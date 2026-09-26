"""PCD (Point Cloud Library) codec - ascii, binary and binary_compressed."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
import operator
import struct
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import Source, open_block, source_name, write_bytes
from polyxios._types import PolyData
from polyxios.codecs._pointcloud import (
    BYTE_SCALE,
    check_float_fmt,
    color_bytes,
    point_cloud,
)
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

try:
    from polyxios.codecs import _lzf as lzf
except ImportError:  # pragma: no cover - the compiled module is optional
    from polyxios.codecs import _lzf_fallback as lzf

EXTENSION: str = ".pcd"

_VERSION: str = "0.7"
_ASCII: str = "ascii"
_BINARY: str = "binary"
_COMPRESSED: str = "binary_compressed"
_DATA_FORMATS: frozenset[str] = frozenset({_ASCII, _BINARY, _COMPRESSED})
_DEFAULT_FLOAT_FMT: str = ".10g"
_DOUBLE_FLOAT_FMT: str = ".17g"

# A field of this name is padding: bytes a binary record holds and nothing
# names. PCL skips it in ascii and compressed files, so here it is skipped
# everywhere but the binary record layout it pads.
_PADDING: str = "_"

_VIEWPOINT_DEFAULT: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
_VIEWPOINT_KEY: str = "pcd_viewpoint"
_WIDTH_KEY: str = "pcd_width"
_HEIGHT_KEY: str = "pcd_height"

_COORDS: tuple[str, str, str] = ("x", "y", "z")
_NORMAL_FIELDS: tuple[str, str, str] = ("normal_x", "normal_y", "normal_z")
_NORMALS: str = "normals"
_COLORS: str = "colors"
_RGB: str = "rgb"
_RGBA: str = "rgba"
_PACKED_SIZE: int = 4

_VERTEX_CODE: int = ELEMENT_TYPES["vertex"]
_EXACT: int = 1 << 53
_U32_MAX: int = (1 << 32) - 1
_I32_MIN: int = -(1 << 31)
# numpy spells a field's byte size in a C int, and a point past that size
# is no cloud anyone wrote; the header is refused before the dtype is built.
_MAX_RECORD_BYTES: int = (1 << 31) - 1

# The longest LZF back-reference spells 264 bytes in 3, so a block cannot
# expand more than 88-fold; a header that says otherwise is refused before
# the expansion is allocated.
_LZF_MAX_RATIO: int = 88

# TYPE letter and SIZE to numpy; PCL writes host order, which is little
# endian everywhere it runs.
_KIND_TO_TYPE: dict[str, str] = {"i": "I", "u": "U", "f": "F"}
_VALID_SIZES: dict[str, frozenset[int]] = {
    "I": frozenset({1, 2, 4, 8}),
    "U": frozenset({1, 2, 4, 8}),
    "F": frozenset({4, 8}),
}
_TYPE_TO_KIND: dict[str, str] = {"I": "i", "U": "u", "F": "f"}


class _Header:
    """What a PCD header declares, resolved and checked."""

    __slots__ = (
        "counts",
        "data",
        "height",
        "names",
        "points",
        "sizes",
        "types",
        "viewpoint",
        "width",
    )

    def __init__(self, fields: dict[str, list[str]], *, name: str) -> None:
        missing = [k for k in ("FIELDS", "SIZE", "TYPE", "DATA") if k not in fields]
        if missing:
            raise CodecError(f"'{name}': PCD header lacks {', '.join(missing)}.")
        self.names = fields["FIELDS"]
        n = len(self.names)
        self.sizes = _ints(fields["SIZE"], "SIZE", n, name)
        self.types = fields["TYPE"]
        if len(self.types) != n:
            raise CodecError(
                f"'{name}': TYPE lists {len(self.types)} entries for {n} fields."
            )
        self.counts = (
            _ints(fields["COUNT"], "COUNT", n, name) if "COUNT" in fields else [1] * n
        )
        for fname, size, typ, count in zip(
            self.names, self.sizes, self.types, self.counts, strict=True
        ):
            if typ not in _VALID_SIZES or size not in _VALID_SIZES[typ]:
                raise CodecError(
                    f"'{name}': field '{fname}' has TYPE {typ} SIZE {size}, "
                    "which PCD does not define."
                )
            if count < 1:
                raise CodecError(f"'{name}': field '{fname}' has COUNT {count}.")
        record = sum(s * c for s, c in zip(self.sizes, self.counts, strict=True))
        if record > _MAX_RECORD_BYTES:
            raise CodecError(
                f"'{name}': SIZE and COUNT make one point {record} bytes, more "
                f"than the {_MAX_RECORD_BYTES} a record can hold."
            )
        tally = Counter(f for f in self.names if f != _PADDING)
        twice = sorted(f for f, k in tally.items() if k > 1)
        if twice:
            warnings.warn(
                f"'{name}': FIELDS names {', '.join(twice)} more than once; a repeat"
                " is read under its name with its position appended.",
                stacklevel=5,
            )
        self.width = _one_int(fields, "WIDTH", name)
        self.height = _one_int(fields, "HEIGHT", name)
        self.points = _one_int(fields, "POINTS", name)
        if self.points is None:
            if self.width is None:
                raise CodecError(f"'{name}': PCD header has neither POINTS nor WIDTH.")
            self.points = self.width * (self.height if self.height is not None else 1)
        if self.height is None:
            self.height = 1
        if self.width is None:
            # A HEIGHT with no WIDTH still names a grid; the product check
            # below refuses a HEIGHT that does not divide POINTS.
            self.width = self.points // self.height if self.height > 0 else self.points
        if min(self.width, self.height, self.points) < 0:
            raise CodecError(
                f"'{name}': WIDTH {self.width}, HEIGHT {self.height} and POINTS "
                f"{self.points} cannot be negative."
            )
        if self.width * self.height != self.points:
            raise CodecError(
                f"'{name}': WIDTH {self.width} x HEIGHT {self.height} is not "
                f"POINTS {self.points}."
            )
        self.data = fields["DATA"][0] if fields["DATA"] else ""
        if self.data not in _DATA_FORMATS:
            raise CodecError(
                f"'{name}': DATA '{self.data}' is not ascii, binary or binary_compressed."
            )
        if "VIEWPOINT" in fields:
            vp = fields["VIEWPOINT"]
            if len(vp) != 7:
                raise CodecError(
                    f"'{name}': VIEWPOINT needs 7 numbers, found {len(vp)}."
                )
            try:
                self.viewpoint = tuple(float(v) for v in vp)
            except ValueError as exc:
                raise CodecError(f"'{name}': VIEWPOINT holds a non-number.") from exc
        else:
            self.viewpoint = _VIEWPOINT_DEFAULT

    def record_dtype(self, *, with_padding: bool) -> np.dtype:
        """The per-point structured dtype the fields describe.

        Parameters
        ----------
        with_padding
            Keep the ``_`` fields as unnamed byte runs, which is the binary
            record layout; drop them, which is the ascii and compressed one.

        Returns
        -------
        numpy.dtype
            One structured record; a field with COUNT above one is a
            sub-array.
        """
        spec: list[tuple[str, Any] | tuple[str, Any, tuple[int]]] = []
        taken: set[str] = set()

        def label(fname: str, k: int) -> str:
            out = fname
            while out in taken:
                out = f"{out}_{k}"
            taken.add(out)
            return out

        for k, (fname, size, typ, count) in enumerate(
            zip(self.names, self.sizes, self.types, self.counts, strict=True)
        ):
            if fname == _PADDING:
                if with_padding:
                    # The space keeps the label clear of every FIELDS token.
                    spec.append((f"_pad {k}", "V1", (size * count,)))
                continue
            base = np.dtype(f"<{_TYPE_TO_KIND[typ]}{size}")
            name = label(fname, k)
            spec.append((name, base) if count == 1 else (name, base, (count,)))
        return np.dtype(spec)

    def data_names(self) -> list[str]:
        """The field names in order with the padding left out."""
        return [n for n in self.names if n != _PADDING]


def _ints(values: list[str], key: str, n: int, name: str) -> list[int]:
    if len(values) != n:
        raise CodecError(f"'{name}': {key} lists {len(values)} entries for {n} fields.")
    try:
        return [int(v) for v in values]
    except ValueError as exc:
        raise CodecError(f"'{name}': {key} holds a non-integer.") from exc


def _one_int(fields: dict[str, list[str]], key: str, name: str) -> int | None:
    if key not in fields:
        return None
    values = fields[key]
    if len(values) != 1:
        raise CodecError(f"'{name}': {key} needs one number, found {len(values)}.")
    try:
        return int(values[0])
    except ValueError as exc:
        raise CodecError(f"'{name}': {key} '{values[0]}' is not an integer.") from exc


def _parse_header(block: bytes | Any, *, name: str) -> tuple[_Header, int]:
    """Read the header lines up to and including ``DATA``.

    Parameters
    ----------
    block
        The file's bytes, or a mapping of them.
    name
        The source, for messages.

    Returns
    -------
    tuple
        The header and the offset of the first body byte.
    """
    fields: dict[str, list[str]] = {}
    pos = 0
    n = len(block)
    while pos < n:
        end = block.find(b"\n", pos)
        if end < 0:
            end = n
        raw = bytes(block[pos:end])
        pos = min(end + 1, n)
        line = raw.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        fields[key] = parts[1:]
        if key == "DATA":
            return _Header(fields, name=name), pos
    raise CodecError(f"'{name}': PCD header has no DATA line.")


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a PCD file and return a point-cloud PolyData.

    Parameters
    ----------
    path
        Path or binary file object of the ``.pcd`` file.
    lazy
        Map a ``DATA binary`` file and hand back arrays that view it,
        read-only, in the file's own dtypes: the coordinates are one strided
        view when ``x``, ``y``, ``z`` sit side by side in one type (else
        ``LazyReadError``), the normals likewise when they are and three
        plain attributes when not, and every other field a view of its own
        column. A packed ``rgb``/``rgba`` field is decoded into ``colors``
        either way. An ``ascii`` or ``binary_compressed`` file raises
        ``LazyReadError``: a number spelled as text or an LZF block has to
        be decoded before it is an array.

    Returns
    -------
    PolyData
        Points only, no elements. ``x``, ``y``, ``z`` are the vertices;
        ``normal_x/y/z`` fold into ``vertex_attrs["normals"]`` and a packed
        ``rgb`` or ``rgba`` into ``vertex_attrs["colors"]``, floats in
        0..1; every other field is a vertex attribute of its own name, two
        dimensional when its COUNT is above one. A file whose HEIGHT is
        above one is an organised cloud and keeps ``pcd_width`` and
        ``pcd_height`` in ``global_attrs``; a VIEWPOINT other than the
        identity is kept as ``pcd_viewpoint``. NaN points are kept.

    Raises
    ------
    CodecError
        On a header that is missing, inconsistent, or declares more points
        than the file holds, on an LZF block that does not expand, and on
        an ascii integer field holding a fraction float64 can see or a
        value its SIZE cannot hold; ``nan`` in such a field reads as 0, as
        PCL reads it.
    LazyReadError
        If ``lazy`` is set on a file that is not ``DATA binary`` or on a
        source that cannot be mapped.

    Notes
    -----
    An ascii ``F 4`` value past float32 range (about 3.4e38) reads as
    ``inf``, with numpy's overflow ``RuntimeWarning`` as the only notice.
    """
    name = source_name(path)
    if lazy:
        # Refusing needs the header, and a buffer cannot be mapped whichever
        # layout it holds; open_block says which before the header is read.
        with open_block(path, fmt=EXTENSION, require_map=True) as data:
            header, body = _parse_header(data, name=name)
            if header.data != _BINARY:
                raise LazyReadError(
                    f"'{name}': PCD DATA {header.data} does not support lazy reads;"
                    " only DATA binary can be viewed without decoding."
                )
            return _read_binary(data, body, header, name=name, lazy=True)

    with open_block(path, fmt=EXTENSION) as data:
        header, body = _parse_header(data, name=name)
        if header.data == _COMPRESSED:
            return _read_compressed(data, body, header, name=name)
        if header.data == _BINARY:
            return _read_binary(data, body, header, name=name, lazy=False)
        return _read_ascii(data, body, header, name=name)


def _read_binary(
    block: Any, body: int, header: _Header, *, name: str, lazy: bool
) -> PolyData:
    dtype = header.record_dtype(with_padding=True)
    n = header.points
    validate_header(n, 0, 0, len(block) - body)
    if body + n * dtype.itemsize > len(block):
        raise CodecError(
            f"'{name}': POINTS {n} of {dtype.itemsize} bytes need "
            f"{n * dtype.itemsize} bytes, the file holds {len(block) - body}."
        )
    raw = np.frombuffer(block, dtype=dtype, count=n, offset=body)
    poly = _assemble(raw, header, name=name, lazy=lazy, block=block, body=body)
    if not lazy:
        # The record view has to go before the block does: a mapping cannot
        # close while an array still points into it.
        del raw
    return poly


def _read_compressed(raw: Any, body: int, header: _Header, *, name: str) -> PolyData:
    if len(raw) < body + 8:
        raise CodecError(f"'{name}': binary_compressed body lacks its two size words.")
    comp_len, full_len = struct.unpack_from("<II", raw, body)
    dtype = header.record_dtype(with_padding=False)
    n = header.points
    validate_header(n, 0, 0, full_len, compressed=True)
    if n * dtype.itemsize != full_len:
        raise CodecError(
            f"'{name}': POINTS {n} of {dtype.itemsize} bytes is "
            f"{n * dtype.itemsize} bytes, the block says {full_len}."
        )
    if body + 8 + comp_len > len(raw):
        raise CodecError(
            f"'{name}': compressed block of {comp_len} bytes runs past the file end."
        )
    if full_len > _LZF_MAX_RATIO * comp_len:
        raise CodecError(
            f"'{name}': compressed block of {comp_len} bytes cannot expand to "
            f"{full_len}; LZF grows at most {_LZF_MAX_RATIO}-fold."
        )
    with memoryview(raw) as view:
        plain = lzf.decompress(view[body + 8 : body + 8 + comp_len], full_len)
    # The block is field-major: every point's first field, then every
    # point's second, so it is read one field at a time into records.
    out = np.empty(n, dtype=dtype)
    offset = 0
    for fname in dtype.names or ():
        sub = dtype[fname]
        width = sub.itemsize
        out[fname] = np.frombuffer(plain, dtype=sub, count=n, offset=offset).reshape(
            (n, *sub.shape) if sub.shape else (n,)
        )
        offset += n * width
    return _assemble(out, header, name=name, lazy=False, block=None, body=0)


def _read_ascii(raw: Any, body: int, header: _Header, *, name: str) -> PolyData:
    dtype = header.record_dtype(with_padding=False)
    n = header.points
    validate_header(n, 0, 0, len(raw) - body)
    with memoryview(raw) as view:
        text = str(view[body:], "ascii", "replace")
    width = sum(
        int(np.prod(dtype[f].shape)) if dtype[f].shape else 1 for f in dtype.names or ()
    )
    if "#" in text:
        text = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
    if text.isspace():
        # numpy reads a body of nothing but whitespace as one value of -1.
        values = np.empty(0, dtype=np.float64)
    else:
        with warnings.catch_warnings():
            # A stray token makes numpy warn that it stopped early; that is a
            # malformed body and reported as one.
            warnings.simplefilter("error", DeprecationWarning)
            try:
                values = np.fromstring(text, dtype=np.float64, sep=" ")
            except (DeprecationWarning, ValueError) as exc:
                raise CodecError(
                    f"'{name}': ascii body holds something that is not a number."
                ) from exc
    if values.size != n * width:
        raise CodecError(
            f"'{name}': POINTS {n} with {width} values each is {n * width} "
            f"numbers, the body holds {values.size}."
        )
    table = values.reshape(n, width)
    tokens: np.ndarray | None = None
    out = np.empty(n, dtype=dtype)
    col = 0
    for fname in dtype.names or ():
        sub = dtype[fname]
        k = int(np.prod(sub.shape)) if sub.shape else 1
        piece = table[:, col : col + k]
        if sub.base.kind in "iu":
            # PCL reads an integer field's nan as 0; a fraction it truncates,
            # which loses what the file spelled, so that is refused instead.
            with np.errstate(invalid="ignore"):
                piece = np.nan_to_num(piece)
            if piece.size and np.any(np.mod(piece, 1) != 0):
                raise CodecError(
                    f"'{name}': field '{fname}' is TYPE {sub.base.kind.upper()} but"
                    " holds a fraction."
                )
            limits = np.iinfo(sub.base)
            if piece.size and (piece.min() < limits.min or piece.max() > limits.max):
                raise CodecError(
                    f"'{name}': field '{fname}' holds a value outside "
                    f"{limits.min}..{limits.max}, what its TYPE and SIZE can hold."
                )
            if piece.size and sub.base.itemsize == 8 and np.abs(piece).max() >= _EXACT:
                # float64 carries 53 bits; an 8-byte integer past that is
                # re-read from its own tokens, which the float pass has
                # already checked are numbers within range.
                if tokens is None:
                    tokens = np.array(text.split(), dtype=str).reshape(n, width)
                try:
                    piece = tokens[:, col : col + k].astype(sub.base)
                except OverflowError as exc:
                    raise CodecError(
                        f"'{name}': field '{fname}' holds a value outside "
                        f"{limits.min}..{limits.max}, what its TYPE and SIZE can hold."
                    ) from exc
                except ValueError:
                    piece = _exact_integers(
                        tokens[:, col : col + k], dtype=sub.base, name=name, fname=fname
                    )
        if fname in (_RGB, _RGBA) and _is_packed(sub) and sub.base.kind == "f":
            # An integer-typed colour already holds the packed value; only
            # the float spelling has to be turned back into bits.
            if tokens is None:
                tokens = np.array(text.split(), dtype=str).reshape(n, width)
            piece = _rgb_from_text(piece, tokens[:, col : col + k])
        out[fname] = piece.reshape((n, *sub.shape)) if sub.shape else piece[:, 0]
        col += k
    return _assemble(out, header, name=name, lazy=False, block=None, body=0)


def _exact_integers(
    tokens: np.ndarray, *, dtype: np.dtype, name: str, fname: str
) -> np.ndarray:
    """Eight-byte integers from tokens numpy does not parse as integers.

    Parameters
    ----------
    tokens
        The field's tokens, ``(n, count)``, already known to be numbers
        within the type's range as floats.
    dtype
        The field's integer dtype.
    name
        The source, for messages.
    fname
        The field, for messages.

    Returns
    -------
    numpy.ndarray
        The tokens' exact values in ``dtype``; ``nan`` is 0 as in every
        other integer field.

    Raises
    ------
    CodecError
        On a token with a fraction, or one outside the type's range.

    Notes
    -----
    A writer that spells an integer as ``9007199254740993.0`` or
    ``9.007199254740993e15`` means that exact value, which float64 rounds
    away past 2**53. ``Decimal`` keeps every digit either way.
    """
    limits = np.iinfo(dtype)
    out = np.empty(tokens.shape, dtype=dtype)
    flat = out.reshape(-1)
    for i, tok in enumerate(tokens.reshape(-1)):
        value = Decimal(tok)
        if value.is_nan():
            flat[i] = 0
            continue
        try:
            whole = int(value)
        except OverflowError:
            whole = None
        if whole is None or not limits.min <= whole <= limits.max:
            raise CodecError(
                f"'{name}': field '{fname}' holds a value outside "
                f"{limits.min}..{limits.max}, what its TYPE and SIZE can hold."
            )
        if whole != value:
            raise CodecError(
                f"'{name}': field '{fname}' is TYPE {dtype.kind.upper()} but"
                " holds a fraction."
            )
        flat[i] = whole
    return out


def _is_packed(sub: np.dtype) -> bool:
    """Whether a field is a colour packed in 32 bits: one scalar of 4 bytes."""
    return (
        not sub.shape and sub.base.kind in "fiu" and sub.base.itemsize == _PACKED_SIZE
    )


def _rgb_from_text(piece: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """A packed ``rgb`` column as an ascii body spells it, back to float bits.

    Parameters
    ----------
    piece
        The column as float64, one value per point.
    tokens
        The same column as the body spells it, one token per value.

    Returns
    -------
    numpy.ndarray
        The column as ``<f4`` whose bits hold ``0xAARRGGBB``, the layout
        every other reader of the field expects.

    Notes
    -----
    PCL writes the packed integer's decimal value, unsigned today and
    signed in older releases: a bare integer token, digits with at most a
    leading ``-`` and within -2**31..2**32-1, is taken as those bits. Any
    other token spells the float itself - ``4.2108e+06`` in the PCL
    tutorial cloud, which PCL before 1.8 wrote, or ``-2.5e+38`` from a
    writer that prints the field as it is stored - and its float32 bits
    are the colour. NaN and infinities are black. The choice is made per
    point, so a body that mixes the two spellings reads every row as its
    own.
    """
    finite = np.isfinite(piece)
    clean = np.where(finite, piece, 0.0)
    bare = np.char.isdigit(np.char.lstrip(tokens, "-"))
    decimal = bare & (clean >= _I32_MIN) & (clean <= _U32_MAX)
    bits = np.where(decimal, clean, 0.0).astype(np.int64) & _U32_MAX
    return np.where(decimal, bits.astype("<u4").view("<f4"), clean.astype("<f4"))


def _assemble(
    raw: np.ndarray,
    header: _Header,
    *,
    name: str,
    lazy: bool,
    block: Any,
    body: int,
) -> PolyData:
    """Turn the record array into a PolyData, folding the known fields."""
    names = list(raw.dtype.names or ())
    n = raw.shape[0]
    used: set[str] = set()

    def triple(fields: tuple[str, str, str], *, required: bool) -> np.ndarray | None:
        """The three fields as one ``(n, 3)`` array, or None.

        A lazy read views them in place, which needs them side by side in
        one type. When they are not, a required triple raises and an
        optional one is left to come back as three plain attributes.
        """
        if any(f not in names or raw.dtype[f].shape for f in fields):
            return None
        if lazy:
            idx = [names.index(f) for f in fields]
            base = raw.dtype[fields[0]]
            offs = [raw.dtype.fields[f][1] for f in fields]  # type: ignore[index]
            same = all(raw.dtype[f] == base for f in fields)
            adjacent = offs[1] - offs[0] == base.itemsize == offs[2] - offs[1]
            if not (
                same and adjacent and idx[1] == idx[0] + 1 and idx[2] == idx[1] + 1
            ):
                if not required:
                    return None
                raise LazyReadError(
                    f"'{name}': fields {', '.join(fields)} are not three adjacent"
                    " values of one type, so they cannot be viewed as (n, 3)."
                )
            used.update(fields)
            if n == 0:
                # Nothing to view; an offset past a mapping's end is refused.
                empty = np.empty((0, 3), dtype=base)
                empty.flags.writeable = False
                return empty
            return np.ndarray(
                (n, 3),
                dtype=base,
                buffer=block,
                offset=body + offs[0],
                strides=(raw.dtype.itemsize, base.itemsize),
            )
        used.update(fields)
        return np.column_stack([raw[f].astype(np.float64) for f in fields])

    vertices = triple(_COORDS, required=True)
    if vertices is None:
        raise CodecError(
            f"'{name}': PCD FIELDS lack x, y and z as three scalar fields."
        )
    attrs: dict[str, np.ndarray] = {}
    normals = triple(_NORMAL_FIELDS, required=False)
    if normals is not None:
        attrs[_NORMALS] = normals
    for packed, channels in ((_RGBA, 4), (_RGB, 3)):
        if packed in names and _is_packed(raw.dtype[packed]):
            attrs[_COLORS] = _unpack_colors(raw[packed], channels)
            used.add(packed)
            break
    for f in names:
        # Padding is the one void-typed field a binary record carries.
        if f in used or raw.dtype[f].base.kind == "V":
            continue
        col = raw[f]
        attrs[f] = col if lazy else np.array(col)
    globals_: dict[str, Any] = {}
    if header.height > 1:
        globals_[_WIDTH_KEY] = header.width
        globals_[_HEIGHT_KEY] = header.height
    if header.viewpoint != _VIEWPOINT_DEFAULT:
        globals_[_VIEWPOINT_KEY] = np.array(header.viewpoint, dtype=np.float64)
    return point_cloud(vertices=vertices, vertex_attrs=attrs, global_attrs=globals_)


def _unpack_colors(col: np.ndarray, channels: int) -> np.ndarray:
    """Split a packed 0xAARRGGBB column into floats in 0..1.

    Parameters
    ----------
    col
        The ``rgb`` or ``rgba`` column, a float32 whose bits hold the
        integer (PCL's spelling) or the integer itself.
    channels
        Three for ``rgb``, four for ``rgba``.

    Returns
    -------
    numpy.ndarray
        ``(n, channels)`` float64, red first, alpha last.
    """
    if col.dtype.kind == "f":
        bits = np.ascontiguousarray(col, dtype="<f4").view("<u4")
    else:
        bits = col.astype("<u4")
    r = (bits >> 16) & 0xFF
    g = (bits >> 8) & 0xFF
    b = bits & 0xFF
    parts = [r, g, b]
    if channels == 4:
        parts.append((bits >> 24) & 0xFF)
    return np.column_stack(parts).astype(np.float64) / BYTE_SCALE


def _pack_colors(colors: np.ndarray) -> np.ndarray:
    """The inverse of ``_unpack_colors``: colour channels to one uint32."""
    scaled = color_bytes(colors).astype("<u4")
    bits = (scaled[:, 0] << 16) | (scaled[:, 1] << 8) | scaled[:, 2]
    if scaled.shape[1] == 4:
        bits |= scaled[:, 3] << 24
    return bits


def _field_type(arr: np.ndarray) -> tuple[str, int, np.dtype] | None:
    """The PCD TYPE, SIZE and file dtype an attribute is written at."""
    kind = arr.dtype.kind
    if kind == "b":
        return "U", 1, np.dtype("u1")
    if kind not in _KIND_TO_TYPE:
        return None
    typ = _KIND_TO_TYPE[kind]
    size = arr.dtype.itemsize
    if typ == "F" and size < 4:
        size = 4
    if size not in _VALID_SIZES[typ]:
        return None
    return typ, size, np.dtype(f"<{_TYPE_TO_KIND[typ]}{size}")


def _grid_size(value: Any) -> int | None:
    """A ``pcd_width`` or ``pcd_height`` value as an int, None when it is not one.

    Parameters
    ----------
    value
        What ``global_attrs`` holds: an integer of any kind, numpy's
        included, is itself; a float or a string counts only when it spells
        a whole number, so ``4.0`` is 4 and ``2.5`` is nothing.

    Returns
    -------
    int or None
    """
    try:
        return operator.index(value)
    except TypeError:
        pass
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return int(as_float) if as_float.is_integer() else None


def write(
    poly: PolyData,
    path: Source,
    *,
    data_format: str = _BINARY,
    double: bool = False,
    float_fmt: str | None = None,
    **opts: Any,
) -> None:
    """Serialise a PolyData's vertices and their attributes as a PCD file.

    Parameters
    ----------
    poly
        The mesh; its vertices are the points. Elements are not part of the
        format: any that are not single vertices are dropped with a
        warning.
    path
        Destination path or binary file object.
    data_format
        ``"binary"`` (the default), ``"ascii"`` or ``"binary_compressed"``,
        which packs the fields with LZF the way PCL does.
    double
        Write the coordinates and normals as ``F 8`` rather than the ``F 4``
        the point-cloud ecosystem expects. At ``F 4`` a value past float32
        range (about 3.4e38) becomes ``inf``: in a binary body on write, in
        an ascii body, which spells the source value, on read; numpy's
        overflow ``RuntimeWarning`` is the only notice. In an ascii body
        ``double`` also raises the float precision to ``.17g``, every digit
        a double carries, unless ``float_fmt`` is given.
    float_fmt
        Format spec for every float in an ascii body. Left unset, it is
        ``.10g``, or ``.17g`` when ``double`` is set; given, it is used as
        is whatever ``double`` says.

    Notes
    -----
    ``vertex_attrs["normals"]`` goes out as ``normal_x normal_y normal_z``,
    ``vertex_attrs["colors"]`` (three or four wide; floats in 0..1, or
    integers already in 0..255) as one
    packed ``rgb`` or ``rgba`` field. Every other numeric vertex attribute
    is a field of its own name, a two-dimensional one with its width as
    COUNT, so a single column reads back one-dimensional; a boolean one
    goes out as ``U 1``. One that is not numeric, whose name a folded
    field already took (``x``, ``normal_x``, ``rgb``, ...), or whose name
    is not a bare ASCII token other than ``_``, is dropped with a warning. A
    one-column ``rgb`` or ``rgba`` of four bytes, which every reader would
    take for a packed colour, is written as eight with a warning so it reads
    back as its values. A
    ``pcd_width`` and ``pcd_height`` pair in ``global_attrs``, whole numbers
    (``4.0`` counts, ``2.5`` does not) whose product is the point count,
    writes an organised cloud, ``pcd_viewpoint`` the VIEWPOINT line at full
    precision; no other global attribute has a place in the format.
    """
    if data_format not in _DATA_FORMATS:
        raise ValueError(
            f"data_format must be one of {sorted(_DATA_FORMATS)}, not {data_format!r}."
        )
    if float_fmt is None:
        float_fmt = _DOUBLE_FLOAT_FMT if double else _DEFAULT_FLOAT_FMT
    check_float_fmt(float_fmt, fmt=EXTENSION)
    n = poly.vertices.shape[0]
    non_vertex = int(np.count_nonzero(poly.element_types != _VERTEX_CODE))
    if non_vertex:
        warnings.warn(
            f".pcd holds points only; {non_vertex} elements dropped.", stacklevel=3
        )
    coord_dtype = np.dtype("<f8" if double else "<f4")
    coord_size = coord_dtype.itemsize
    names: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    columns: list[tuple[np.ndarray, np.dtype]] = []

    def add(
        fname: str, typ: str, size: int, count: int, col: np.ndarray, dt: np.dtype
    ) -> None:
        names.append(fname)
        types.append(typ)
        sizes.append(size)
        counts.append(count)
        columns.append((col, dt))

    for k, axis in enumerate(_COORDS):
        add(axis, "F", coord_size, 1, poly.vertices[:, k], coord_dtype)
    # The folded families claim their field names first, so a plain
    # attribute that happens to share one is the one dropped.
    folded = (_NORMALS, _COLORS)
    ordered = sorted(poly.vertex_attrs.items(), key=lambda kv: kv[0] not in folded)
    for aname, arr in ordered:
        arr = np.asarray(arr)
        if arr.ndim == 0 or arr.shape[0] != n:
            raise CodecError(
                f"vertex attribute '{aname}' has shape {arr.shape} for {n} points."
            )
        if aname == _NORMALS and arr.ndim == 2 and arr.shape[1] == 3:
            for k, axis in enumerate(_NORMAL_FIELDS):
                add(axis, "F", coord_size, 1, arr[:, k], coord_dtype)
            continue
        if aname == _COLORS and arr.ndim == 2 and arr.shape[1] in (3, 4):
            packed = _pack_colors(arr)
            if arr.shape[1] == 4:
                add(_RGBA, "U", 4, 1, packed, np.dtype("<u4"))
            else:
                add(_RGB, "F", 4, 1, packed.view("<f4"), np.dtype("<f4"))
            continue
        if aname in names:
            warnings.warn(
                f".pcd already spells a field '{aname}'; the vertex attribute of"
                " that name is dropped.",
                stacklevel=3,
            )
            continue
        if aname.split() != [aname] or not aname.isascii() or aname == _PADDING:
            # FIELDS is a line of ASCII tokens, and "_" is the padding name
            # every reader skips: either way the file would not read back.
            warnings.warn(
                f".pcd field names are bare ASCII tokens other than '_'; vertex"
                f" attribute {aname!r} dropped.",
                stacklevel=3,
            )
            continue
        spec = _field_type(arr)
        if spec is None or arr.ndim > 2:
            warnings.warn(
                f".pcd cannot hold vertex attribute '{aname}' of dtype {arr.dtype}"
                f" and shape {arr.shape}; dropped.",
                stacklevel=3,
            )
            continue
        typ, size, dt = spec
        count = arr.shape[1] if arr.ndim == 2 else 1
        if count < 1:
            warnings.warn(
                f".pcd cannot hold vertex attribute '{aname}' of no columns; dropped.",
                stacklevel=3,
            )
            continue
        if aname in (_RGB, _RGBA) and count == 1 and size == _PACKED_SIZE:
            # Every reader, this one included, takes a four-byte scalar of
            # that name for a packed colour.
            size = 2 * _PACKED_SIZE
            dt = np.dtype(f"<{_TYPE_TO_KIND[typ]}{size}")
            warnings.warn(
                f"a four-byte '{aname}' field is a packed colour to every PCD reader;"
                f" vertex attribute {aname!r} is written as SIZE {size} so it reads"
                " back as its values.",
                stacklevel=3,
            )
        add(
            aname,
            typ,
            size,
            count,
            arr[:, 0] if count == 1 and arr.ndim == 2 else arr,
            dt,
        )

    width, height = n, 1
    g = poly.global_attrs
    if (_WIDTH_KEY in g) != (_HEIGHT_KEY in g):
        present = _WIDTH_KEY if _WIDTH_KEY in g else _HEIGHT_KEY
        warnings.warn(
            f"{present} without its pair names no grid; written unorganised.",
            stacklevel=3,
        )
    elif _WIDTH_KEY in g:
        w, h = _grid_size(g[_WIDTH_KEY]), _grid_size(g[_HEIGHT_KEY])
        if w is None or h is None:
            warnings.warn(
                f"pcd_width {g[_WIDTH_KEY]!r} x pcd_height {g[_HEIGHT_KEY]!r} is"
                " not a grid of whole numbers; written unorganised.",
                stacklevel=3,
            )
        elif w * h == n and w > 0 and h > 0:
            width, height = w, h
        else:
            warnings.warn(
                f"pcd_width {g[_WIDTH_KEY]!r} x pcd_height {g[_HEIGHT_KEY]!r} is"
                f" not the point count {n}; written unorganised.",
                stacklevel=3,
            )
    viewpoint = _VIEWPOINT_DEFAULT
    if _VIEWPOINT_KEY in g:
        try:
            vp = np.asarray(g[_VIEWPOINT_KEY], dtype=np.float64).ravel()
        except (TypeError, ValueError):
            vp = np.empty(0)
        if vp.shape == (7,):
            viewpoint = tuple(float(v) for v in vp)
        else:
            warnings.warn(
                f"pcd_viewpoint needs 7 numbers, found {g[_VIEWPOINT_KEY]!r};"
                " default written.",
                stacklevel=3,
            )

    record = np.dtype(
        [
            (fname, dt) if count == 1 else (fname, dt, (count,))
            for fname, count, (_, dt) in zip(names, counts, columns, strict=True)
        ]
    )
    table = np.empty(n, dtype=record)
    for fname, (col, dt) in zip(names, columns, strict=True):
        table[fname] = col.astype(dt, copy=False)

    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        f"VERSION {_VERSION}",
        "FIELDS " + " ".join(names),
        "SIZE " + " ".join(map(str, sizes)),
        "TYPE " + " ".join(types),
        "COUNT " + " ".join(map(str, counts)),
        f"WIDTH {width}",
        f"HEIGHT {height}",
        "VIEWPOINT "
        + " ".join(np.format_float_positional(v, trim="-") for v in viewpoint),
        f"POINTS {n}",
        f"DATA {data_format}",
    ]
    header = ("\n".join(lines) + "\n").encode("ascii")

    if data_format == _BINARY:
        body = table.tobytes()
    elif data_format == _COMPRESSED:
        plain = b"".join(np.ascontiguousarray(table[f]).tobytes() for f in names)
        too_big = CodecError(
            f".pcd binary_compressed spells its sizes in 32 bits; {len(plain)}"
            " bytes of points do not fit. Write DATA binary instead."
        )
        if len(plain) > _U32_MAX:
            raise too_big
        packed = lzf.compress(plain)
        # Incompressible input grows, so a block just under the limit can
        # still spill over once packed.
        if len(packed) > _U32_MAX:
            raise too_big
        body = struct.pack("<II", len(packed), len(plain)) + packed
    else:
        body = _ascii_body(table, names, columns, float_fmt=float_fmt)
    write_bytes(path, header + body)


def _ascii_body(
    table: np.ndarray,
    names: list[str],
    columns: list[tuple[np.ndarray, np.dtype]],
    *,
    float_fmt: str,
) -> bytes:
    """Spell the records one point per line.

    Parameters
    ----------
    table
        The records in their file dtypes, the source of every integer.
    names
        The field names in record order.
    columns
        The arrays each field was built from, in the same order. A float
        field is spelled from its source rather than from the ``F 4`` the
        record holds, so ``0.1`` is written as ``0.1`` and not as the ten
        significant digits of its float32 neighbour.
    float_fmt
        Format spec for every float.

    Returns
    -------
    bytes
        The body, one line per point.
    """
    n = table.shape[0]
    if not names or n == 0:
        return b""
    spelled: list[list[str]] = []
    for f, (source, _) in zip(names, columns, strict=True):
        col = table[f]
        flat = col.reshape(n, -1)
        if f == _RGB and _is_packed(table.dtype[f]) and flat.dtype.kind == "f":
            flat = flat.view("<u4")
        if flat.dtype.kind == "f":
            values = np.asarray(source, dtype=np.float64).reshape(n, -1)
            spelled.append([format(v, float_fmt) for v in values.ravel()])
        else:
            spelled.append([str(v) for v in flat.ravel()])
    return _lines(spelled, [len(s) // n for s in spelled], n).encode("ascii")


def _lines(spelled: list[list[str]], widths: list[int], n: int) -> str:
    """Join per-field token lists, each ``n * width`` long, into ``n`` lines."""
    rows: list[list[str]] = [[] for _ in range(n)]
    for tokens, width in zip(spelled, widths, strict=True):
        if width == 1:
            for row, tok in zip(rows, tokens, strict=True):
                row.append(tok)
        else:
            for r, row in enumerate(rows):
                row.extend(tokens[r * width : (r + 1) * width])
    return "\n".join(" ".join(row) for row in rows) + "\n"
