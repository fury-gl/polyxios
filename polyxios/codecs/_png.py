"""PNG codec: gray, gray-alpha, RGB, RGBA and palette images, 1 to 16 bits.

Pixels are decoded to :class:`~polyxios.ImageData` and encoded from one. The
codec uses NumPy and the standard ``zlib`` module only; the scanline
unfiltering is the one kernel that may be compiled.
"""

from collections.abc import Iterator, Mapping
import math
import numbers
import operator
import struct
from typing import Any
import zlib

import numpy as np

from polyxios._image import ImageData, check_readable, raster_pixels, warn_left_out
from polyxios._io import Source, read_bytes, source_name, write_bytes
from polyxios._media import sniff_media
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.codecs._hdf5 import warn_unknown_opts
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.helper import array_to_image
from polyxios.validate import validate_header

try:
    from polyxios.codecs import _raster_kernels as kernels
except ImportError:  # pragma: no cover - the compiled module is optional
    from polyxios.codecs import _raster_kernels_fallback as kernels

EXTENSION: str = ".png"
_FORMAT: str = "PNG"

_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"
_KNOWN_CRITICAL: frozenset[bytes] = frozenset({b"IHDR", b"PLTE", b"IDAT", b"IEND"})
_IDAT_PIECE: int = 1 << 20

# Samples per pixel for each colour type a file may spell.
_CHANNELS: dict[int, int] = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DEPTHS: dict[int, frozenset[int]] = {
    0: frozenset({1, 2, 4, 8, 16}),
    2: frozenset({8, 16}),
    3: frozenset({1, 2, 4, 8}),
    4: frozenset({8, 16}),
    6: frozenset({8, 16}),
}

# Colour type and channel count of each mode :func:`raster_pixels` returns.
_WRITE_COLOR: dict[str, int] = {"L": 0, "LA": 4, "RGB": 2, "RGBA": 6}
_WRITE_CHANNELS: dict[str, int] = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}

# The Adam7 passes as (y0, x0, dy, dx); a non-interlaced image is one pass.
_ADAM7: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 8, 8),
    (0, 4, 8, 8),
    (4, 0, 8, 4),
    (0, 2, 4, 4),
    (2, 0, 4, 2),
    (0, 1, 2, 2),
    (1, 0, 2, 1),
)
_WHOLE: tuple[tuple[int, int, int, int], ...] = ((0, 0, 1, 1),)

_GAMMA_SCALE: int = 100000
_METERS_PER_INCH: float = 0.0254
_PNG_UINT_MAX: int = (1 << 31) - 1  # a PNG four-byte unsigned integer's limit
_FILTER_STRIP_BYTES: int = 1 << 18  # scanline bytes the encoder filters together
_WRITTEN: frozenset[str] = frozenset({"dpi", "gamma", "icc_profile", "text", "exif"})


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Read a PNG file as the lattice of its pixels.

    Parameters
    ----------
    path
        Path or open binary file object.
    lazy
        Not supported: the pixels are compressed, so there is nothing to map.

    Returns
    -------
    PolyData
        One vertex per pixel, with the pixel values as ``colors`` or
        ``intensity``.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        On a file that is not a well-formed PNG.
    ValidationError
        If the header declares more pixels than polyxios will allocate, or
        more than it will expand into vertices.

    Warns
    -----
    UserWarning
        Always, to point at :func:`polyxios.read_image`, which keeps the
        compact :class:`~polyxios.ImageData`; and as :func:`read_image` warns.
    """
    if lazy:
        raise LazyReadError(
            f"{EXTENSION}: lazy reads are not supported; the pixels are compressed."
        )
    image = _decode(path, builds_vertices=True)
    warn_caller(
        f"'{source_name(path)}' is an image format ({_FORMAT}): read() expands "
        "the pixels into a PolyData lattice, one vertex per pixel. Use "
        "polyxios.read_image() to get the compact ImageData."
    )
    return image.to_polydata()


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a lattice of pixels as a PNG file.

    Parameters
    ----------
    poly
        A flat lattice, as :meth:`ImageData.to_polydata` builds it.
    path
        Path or open binary file object.
    **opts
        Forwarded to :func:`write_image`.

    Raises
    ------
    CodecError
        If the lattice is not a flat image that PNG can hold.
    ValidationError
        If the grid is past the caps :func:`read_image` holds a file to.

    Warns
    -----
    UserWarning
        As :func:`write_image` warns.
    """
    write_image(ImageData.from_polydata(poly, fmt=EXTENSION), path, **opts)


def read_image(path: Source, **opts: Any) -> ImageData:
    """Read a PNG file as a flat :class:`~polyxios.ImageData`.

    Parameters
    ----------
    path
        Path or open binary file object.
    **opts
        No option is taken; any given is warned about and ignored.

    Returns
    -------
    ImageData
        One z plane with ``colors`` (RGB, RGBA or palette colors), or
        ``intensity`` (gray, with ``alpha`` for gray-alpha or a transparent
        key). The dtype is ``uint8``, or ``uint16`` for 16-bit files. The
        image holds ``dpi``, ``gamma``, ``icc_profile``, ``exif`` and ``text``
        in its ``global_attrs`` where the file spells them.

    Raises
    ------
    CodecError
        On a file that is not a well-formed PNG, or uses a feature polyxios
        does not read, or whose gray, RGB or palette ``tRNS`` does not fit.
    ValidationError
        If the header declares more pixels than polyxios will allocate.

    Warns
    -----
    UserWarning
        If the compressed data holds more than the image needs, a metadata
        chunk (``pHYs``, ``gAMA``, ``iCCP`` or text) is malformed, a
        text keyword repeats, or a ``tRNS`` chunk sits after ``IDAT``.
    """
    warn_unknown_opts(opts, fmt=EXTENSION, what="read", stacklevel=4)
    return _decode(path, builds_vertices=False)


def _decode(path: Source, *, builds_vertices: bool) -> ImageData:
    """Parse a PNG; ``builds_vertices`` says whether the caller expands it."""
    name = source_name(path)
    data = read_bytes(path)
    if not data.startswith(_SIGNATURE):
        raise _not_png(name, data)
    chunks = _walk(data, name)
    kind, ihdr = next(chunks)
    if kind != b"IHDR" or len(ihdr) != 13:
        raise CodecError(f"'{name}': PNG does not open with a 13-byte IHDR chunk.")
    width, height, depth, color, compression, method, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if not (0 < width < 1 << 31 and 0 < height < 1 << 31):
        raise CodecError(
            f"'{name}': PNG width and height must be 1 to 2**31-1, not "
            f"{width} x {height}."
        )
    validate_header(
        width * height,
        max(width - 1, 0) * max(height - 1, 0),
        4 * max(width - 1, 0) * max(height - 1, 0),
        len(data),
        compressed=True,
        spells_vertices=False,
        spells_connectivity=False,
        builds_vertices=builds_vertices,
    )
    rest = list(chunks)
    _check_structure(name, depth, color, compression, method, interlace, rest)

    first_idat = next(i for i, (k, _) in enumerate(rest) if k == b"IDAT")
    head = rest[:first_idat]
    if any(k == b"tRNS" for k, _ in rest[first_idat:]):
        warn_caller(
            f"'{name}': PNG tRNS chunk after IDAT is out of place; it is ignored."
        )
    idat = b"".join(body for k, body in rest if k == b"IDAT")
    channels = _CHANNELS[color]
    interlaced = interlace == 1
    passes = _passes(width, height, interlaced)
    expected = sum(ph * (1 + _row_bytes(pw, channels, depth)) for *_, ph, pw in passes)
    raw = _inflate_image(idat, expected, name)

    palette = _palette(head, name) if color == 3 else None
    key = _transparency(
        head, color, depth, 0 if palette is None else len(palette), name
    )
    if palette is not None and len(palette) < 1 << depth:
        # An index past the palette is opaque black (PNG 3rd ed. 13.1).
        missing = (1 << depth) - len(palette)
        palette = np.concatenate([palette, np.zeros((missing, 3), np.uint8)])
        if key is not None:
            key = np.concatenate([key, np.full(missing, 255, np.uint8)])
    full: np.ndarray | None = None
    if interlaced:
        dtype = np.uint16 if depth == 16 else np.uint8
        full = np.zeros((height, width, _out_channels(color, key)), dtype)

    bpp = max(1, channels * depth // 8)
    view = memoryview(raw)
    offset = 0
    for y0, x0, dy, dx, ph, pw in passes:
        rb = _row_bytes(pw, channels, depth)
        span = ph * (1 + rb)
        try:
            unfiltered = kernels.png_unfilter(view[offset : offset + span], ph, rb, bpp)
        except CodecError as exc:
            where = ""
            if interlaced:
                where = f" (Adam7 pass {_ADAM7.index((y0, x0, dy, dx)) + 1} of 7)"
            raise CodecError(f"'{name}': {str(exc).removesuffix('.')}{where}.") from exc
        rows = np.frombuffer(unfiltered, dtype=np.uint8).reshape(ph, rb)
        offset += span
        samples = _unpack(rows, pw, channels, depth)
        expanded = _expand(samples, color=color, depth=depth, palette=palette, key=key)
        if full is None:
            full = expanded
        else:
            full[y0::dy, x0::dx] = expanded

    assert full is not None  # a non-empty image has at least one pass
    pixels = full[..., 0] if color == 0 and key is None else full
    return array_to_image(pixels, origin="upper", global_attrs=_metadata(rest, name))


def write_image(
    image: ImageData, path: Source, *, compress_level: int = 6, **opts: Any
) -> None:
    """Write a flat :class:`~polyxios.ImageData` as a PNG file.

    Parameters
    ----------
    image
        The image. ``colors`` (3 or 4 wide) or ``intensity`` (+ ``alpha``)
        point attributes hold the pixels; see
        :func:`polyxios._image.raster_pixels`.
    path
        Path or open binary file object.
    compress_level
        zlib level for the image data and the ICC profile, 0 (none) to 9
        (smallest).
    **opts
        No other option is taken; any given is warned about and ignored.

    Raises
    ------
    CodecError
        If the image is not one flat unrotated layer, its values do not fit
        8 or 16 bits, its metadata is out of range or of a kind a PNG cannot
        spell, or ``compress_level`` is not an int in 0-9.
    ValidationError
        If the grid is past the caps :func:`read_image` holds a file to.

    Warns
    -----
    UserWarning
        Naming whatever the image holds that a PNG has no place for.
    """
    warn_unknown_opts(opts, fmt=EXTENSION, what="write", stacklevel=4)
    try:
        level = operator.index(compress_level)
    except TypeError:
        level = -1
    if isinstance(compress_level, bool) or not 0 <= level <= 9:
        raise CodecError(
            f"{EXTENSION}: compress_level must be an int in 0-9, not "
            f"{_shown(compress_level)}."
        )
    pixels, mode = raster_pixels(
        image, fmt=EXTENSION, dtypes=frozenset({"uint8", "uint16"})
    )
    check_readable(image)
    warn_left_out(image.global_attrs, fmt=EXTENSION, written=_WRITTEN)
    metadata = _metadata_chunks(image.global_attrs, level)  # refuse before filtering
    height, width = pixels.shape[:2]
    depth = 16 if pixels.dtype == np.uint16 else 8
    if depth == 16:
        rows = pixels.reshape(height, -1).astype(">u2").view(np.uint8)
    else:
        rows = pixels.reshape(height, -1)
    bpp = _WRITE_CHANNELS[mode] * (depth // 8)
    header = struct.pack(">IIBBBBB", width, height, depth, _WRITE_COLOR[mode], 0, 0, 0)
    compressed = zlib.compress(_filter_rows(rows, bpp), level)
    chunks = [_chunk(b"IHDR", header)]
    chunks.extend(metadata)
    chunks.extend(
        _chunk(b"IDAT", compressed[at : at + _IDAT_PIECE])
        for at in range(0, len(compressed), _IDAT_PIECE)
    )
    chunks.append(_chunk(b"IEND", b""))
    write_bytes(path, b"".join([_SIGNATURE, *chunks]))


# -----------------------------------------------------------------------------
# Reading


def _not_png(name: str, data: bytes) -> CodecError:
    """Name the failed signature, and the image type the bytes do spell."""
    media = sniff_media(data)
    hint = f"; it holds {media} data, which polyxios does not read" if media else ""
    return CodecError(f"'{name}' is not a {_FORMAT} file{hint}.")


def _walk(data: bytes, name: str) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(type, body)`` of each chunk up to and including IEND.

    Every chunk's CRC is checked as it is reached. Bytes after IEND are not
    part of the image and are never looked at.
    """
    pos = len(_SIGNATURE)
    while True:
        if pos + 12 > len(data):
            raise CodecError(f"'{name}': PNG ends before its IEND chunk.")
        (length,) = struct.unpack_from(">I", data, pos)
        kind = data[pos + 4 : pos + 8]
        body_end = pos + 8 + length
        if body_end + 4 > len(data):
            raise CodecError(
                f"'{name}': PNG {kind.decode('latin-1')} chunk runs past the "
                "end of the file."
            )
        body = data[pos + 8 : body_end]
        (crc,) = struct.unpack_from(">I", data, body_end)
        if zlib.crc32(body, zlib.crc32(kind)) != crc:
            raise CodecError(
                f"'{name}': PNG {kind.decode('latin-1')} chunk fails its CRC."
            )
        yield kind, body
        if kind == b"IEND":
            return
        pos = body_end + 4


def _check_structure(
    name: str,
    depth: int,
    color: int,
    compression: int,
    method: int,
    interlace: int,
    rest: list[tuple[bytes, bytes]],
) -> None:
    """Raise unless the header and chunk order are ones polyxios reads."""
    if color not in _DEPTHS or depth not in _DEPTHS[color]:
        raise CodecError(
            f"'{name}': PNG color type {color} at bit depth {depth} is not valid."
        )
    if compression != 0 or method != 0:
        raise CodecError(
            f"'{name}': PNG spells compression method {compression} and filter "
            f"method {method}; only 0 exists."
        )
    if interlace not in (0, 1):
        raise CodecError(f"'{name}': PNG interlace method {interlace} is not valid.")
    for kind, _ in rest:
        if kind == b"IHDR":
            raise CodecError(f"'{name}': PNG has a second IHDR chunk.")
        if kind[0] & 0x20 == 0 and kind not in _KNOWN_CRITICAL:
            raise CodecError(
                f"'{name}': PNG has an unknown critical chunk "
                f"{kind.decode('latin-1')!r}; polyxios cannot read it."
            )
    kinds = [k for k, _ in rest]
    idat_at = [i for i, k in enumerate(kinds) if k == b"IDAT"]
    if not idat_at:
        raise CodecError(f"'{name}': PNG has no IDAT chunk.")
    if idat_at[-1] - idat_at[0] + 1 != len(idat_at):
        raise CodecError(f"'{name}': PNG IDAT chunks are not consecutive.")
    if color == 3 and (b"PLTE" not in kinds or kinds.index(b"PLTE") > idat_at[0]):
        raise CodecError(f"'{name}': PNG palette image has no PLTE chunk before IDAT.")


def _passes(
    width: int, height: int, interlaced: bool
) -> list[tuple[int, int, int, int, int, int]]:
    """Return ``(y0, x0, dy, dx, rows, columns)`` of each non-empty pass."""
    found = []
    for y0, x0, dy, dx in _ADAM7 if interlaced else _WHOLE:
        pw = len(range(x0, width, dx))
        ph = len(range(y0, height, dy))
        if pw and ph:
            found.append((y0, x0, dy, dx, ph, pw))
    return found


def _row_bytes(pw: int, channels: int, depth: int) -> int:
    """Bytes in one scanline of ``pw`` pixels, without its filter byte."""
    return (pw * channels * depth + 7) // 8


def _inflate_image(idat: bytes, expected: int, name: str) -> bytes:
    """Decompress the image data, to exactly ``expected`` bytes of use."""
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(idat, expected + 1)
    except zlib.error as exc:
        raise CodecError(
            f"'{name}': PNG image data is not a valid zlib stream ({exc})."
        ) from exc
    if len(raw) < expected:
        raise CodecError(f"'{name}': PNG image data ends early.")
    if len(raw) > expected or inflater.unused_data:
        warn_caller(
            f"'{name}': PNG IDAT holds more data than the image; the excess is ignored."
        )
    return raw


def _unpack(rows: np.ndarray, pw: int, channels: int, depth: int) -> np.ndarray:
    """Split scanlines into ``(rows, pw, channels)`` samples at the file depth."""
    ph = rows.shape[0]
    if depth == 16:
        values = np.ascontiguousarray(rows).view(">u2").astype(np.uint16)
    elif depth == 8:
        values = rows
    else:
        per_byte = 8 // depth
        shifts = np.arange(per_byte - 1, -1, -1, dtype=np.uint8) * depth
        mask = (1 << depth) - 1
        values = ((rows[:, :, None] >> shifts) & mask).reshape(ph, -1)
    return values[:, : pw * channels].reshape(ph, pw, channels)


def _expand(
    samples: np.ndarray,
    *,
    color: int,
    depth: int,
    palette: np.ndarray | None,
    key: Any,
) -> np.ndarray:
    """Turn file samples into output pixels of the dtype the depth gives."""
    dtype = np.uint16 if depth == 16 else np.uint8
    top = 65535 if depth == 16 else 255
    if color == 3:
        assert palette is not None
        index = samples[..., 0]
        rgb = palette[index]
        if key is None:
            return rgb
        return np.concatenate([rgb, key[index][..., None]], axis=-1)
    if color == 2:
        rgb = samples.astype(dtype, copy=False)
        if key is None:
            return rgb
        keys = np.asarray(key, dtype=samples.dtype)
        matched = (samples == keys).all(axis=-1)
        alpha = np.full(matched.shape, top, dtype)
        alpha[matched] = 0
        return np.concatenate([rgb, alpha[..., None]], axis=-1)
    if color == 0:
        value = samples[..., 0]
        if depth < 8:
            levels = np.arange(1 << depth) * 255 // ((1 << depth) - 1)
            gray = levels.astype(np.uint8)[value][..., None]
        else:
            gray = value.astype(dtype, copy=False)[..., None]
        if key is None:
            return gray
        alpha = np.full(value.shape, top, dtype)
        alpha[value == key] = 0
        return np.concatenate([gray, alpha[..., None]], axis=-1)
    return samples.astype(dtype, copy=False)


def _out_channels(color: int, key: Any) -> int:
    """Channels of the decoded pixels: alpha is added by a transparent key."""
    if color == 0:
        return 2 if key is not None else 1
    if color in (2, 3):
        return 4 if key is not None else 3
    return _CHANNELS[color]


def _first(chunks: list[tuple[bytes, bytes]], kind: bytes) -> bytes | None:
    """Return the body of the first chunk of ``kind``, or None."""
    for k, body in chunks:
        if k == kind:
            return body
    return None


def _palette(chunks: list[tuple[bytes, bytes]], name: str) -> np.ndarray:
    """Return the palette as ``(entries, 3)`` uint8."""
    body = _first(chunks, b"PLTE") or b""
    if not 0 < len(body) <= 768 or len(body) % 3:
        raise CodecError(f"'{name}': PNG palette is not 1-256 RGB entries.")
    return np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)


def _transparency(
    chunks: list[tuple[bytes, bytes]],
    color: int,
    depth: int,
    palette_size: int,
    name: str,
) -> Any:
    """Return the tRNS key: an int, a triple, an alpha table, or None.

    Colour types 4 and 6 carry alpha already, so their tRNS is ignored. A gray
    or RGB key keeps only the low ``depth`` bits of each sample, as the
    specification says a writer may leave the rest unset.
    """
    body = _first(chunks, b"tRNS")
    if body is None or color in (4, 6):
        return None
    mask = (1 << depth) - 1
    if color == 0:
        if len(body) != 2:
            raise CodecError(f"'{name}': PNG tRNS for gray must be one sample.")
        return struct.unpack(">H", body)[0] & mask
    if color == 2:
        if len(body) != 6:
            raise CodecError(f"'{name}': PNG tRNS for RGB must be three samples.")
        return tuple(v & mask for v in struct.unpack(">3H", body))
    if len(body) > palette_size:
        raise CodecError(
            f"'{name}': PNG tRNS holds {len(body)} alphas for a "
            f"{palette_size}-entry palette."
        )
    alpha = np.full(palette_size, 255, dtype=np.uint8)
    alpha[: len(body)] = np.frombuffer(body, dtype=np.uint8)
    return alpha


def _split(data: bytes) -> tuple[bytes, bytes]:
    """Split at the first NUL; a missing terminator is a ``ValueError``."""
    head, sep, tail = data.partition(b"\0")
    if not sep:
        raise ValueError("no NUL terminator")
    return head, tail


# Bytes all the compressed metadata chunks of one file may expand to together.
_METADATA_BUDGET: int = 1 << 26


def _inflate(packed: bytes, room: list[int]) -> bytes:
    """Inflate a metadata chunk, charging what it inflates to ``room[0]``.

    A few bytes of zlib stream can expand a thousandfold, and a file may carry
    any number of such chunks, so the whole file shares one budget. What a
    chunk inflates is spent whether or not the chunk is kept.
    """
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(packed, room[0] + 1)
    except zlib.error:
        room[0] = 0  # how much it inflated before failing is unknown
        raise
    over = len(out) > room[0]
    room[0] = max(room[0] - len(out), 0)
    if over:
        raise ValueError(
            f"the file's compressed metadata together expands past the "
            f"{_METADATA_BUDGET}-byte budget"
        )
    if not inflater.eof:
        raise ValueError("its zlib stream is cut short")
    return out


def _deflated(packed: bytes) -> bytes:
    """Return the zlib stream behind a compression-method byte, which must be 0."""
    if packed[:1] != b"\0":
        raise ValueError("unknown compression method")
    return packed[1:]


def _unfit_keyword(key: bytes) -> bool:
    """Return whether ``key`` breaks the text keyword rule.

    A keyword is 1-79 printable Latin-1 bytes, with no leading, trailing or
    doubled space.
    """
    return (
        not 1 <= len(key) <= 79
        or any(b < 32 or 126 < b < 161 for b in key)
        or key != key.strip(b" ")
        or b"  " in key
    )


def _keep_text(text: dict[str, str], keyword: bytes, value: str, name: str) -> None:
    """Keep ``value`` under its Latin-1 ``keyword``; a repeated keyword keeps the first.

    A keyword the writer would refuse, or a value holding a NUL, is a
    ``ValueError``: the chunk is malformed.
    """
    if _unfit_keyword(keyword):
        raise ValueError(
            f"text keyword {keyword!r} is not 1-79 printable Latin-1 bytes with no "
            "leading, trailing or doubled space"
        )
    if "\0" in value:
        raise ValueError("the text holds a NUL")
    key = keyword.decode("latin-1")
    if key in text:
        warn_caller(f"'{name}': PNG repeats text keyword {key!r}; the first is kept.")
        return
    text[key] = value


def _metadata(chunks: list[tuple[bytes, bytes]], name: str) -> dict[str, Any]:
    """Collect the metadata chunks polyxios keeps; other chunks are ignored."""
    meta: dict[str, Any] = {}
    text: dict[str, str] = {}
    room = [_METADATA_BUDGET]
    for kind, body in chunks:
        try:
            if kind == b"pHYs":
                if len(body) != 9:
                    raise ValueError(f"{len(body)} bytes, not 9")
                ppx, ppy, unit = struct.unpack(">IIB", body)
                if max(ppx, ppy) > _PNG_UINT_MAX:
                    raise ValueError(f"{max(ppx, ppy)} is past 2**31-1")
                if unit == 1 and ppx and ppy:
                    meta["dpi"] = (ppx * _METERS_PER_INCH, ppy * _METERS_PER_INCH)
            elif kind == b"gAMA":
                if len(body) != 4:
                    raise ValueError(f"{len(body)} bytes, not 4")
                gamma = struct.unpack(">I", body)[0]
                if gamma > _PNG_UINT_MAX:
                    raise ValueError(f"{gamma} is past 2**31-1")
                if gamma:
                    meta["gamma"] = gamma / _GAMMA_SCALE
            elif kind == b"eXIf":
                meta["exif"] = body
            elif kind == b"iCCP":
                _, packed = _split(body)
                meta["icc_profile"] = _inflate(_deflated(packed), room)
            elif kind == b"tEXt":
                keyword, value = _split(body)
                _keep_text(text, keyword, value.decode("latin-1"), name)
            elif kind == b"zTXt":
                keyword, packed = _split(body)
                value = _inflate(_deflated(packed), room)
                _keep_text(text, keyword, value.decode("latin-1"), name)
            elif kind == b"iTXt":
                keyword, tail = _split(body)
                if len(tail) < 2:
                    raise ValueError("it ends before its compression flag and method")
                compressed = tail[0]
                if compressed not in (0, 1):
                    raise ValueError(f"compression flag {compressed} is not 0 or 1")
                if compressed and tail[1:2] != b"\0":
                    raise ValueError("unknown compression method")
                _, tail = _split(tail[2:])
                _, value = _split(tail)
                if compressed:
                    value = _inflate(value, room)
                _keep_text(text, keyword, value.decode("utf-8"), name)
        except (ValueError, zlib.error) as exc:
            warn_caller(
                f"'{name}': PNG {kind.decode('latin-1')} chunk is malformed "
                f"({exc}); it is ignored."
            )
    if text:
        meta["text"] = text
    return meta


# -----------------------------------------------------------------------------
# Writing


def _chunk(kind: bytes, body: bytes) -> bytes:
    """Frame a chunk: length, type, body and the CRC of type and body."""
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body))
    )


def _shown(value: object) -> str:
    """Return ``repr(value)``, or a stand-in when it is too long to print.

    Python refuses the ``repr`` of an integer of thousands of digits, and an
    error message must not raise.
    """
    try:
        return repr(value)
    except ValueError:
        return f"a value of type {type(value).__name__} too long to print"


def _number(value: object, what: str) -> float:
    """Return ``value`` as a float, or refuse it as a :class:`CodecError`."""
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]  # the one-number array a .vti or .vtkhdf holds
    complex_value = isinstance(value, numbers.Complex) and not isinstance(
        value, numbers.Real
    )
    try:
        if complex_value or isinstance(value, (str, bytes, bytearray, bool, np.bool_)):
            raise TypeError("text, booleans and complex numbers are not real numbers")
        return float(value)  # type: ignore[arg-type]
    except OverflowError as exc:
        raise CodecError(f"{EXTENSION}: {what} is too large for a PNG.") from exc
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{EXTENSION}: {what} must be a number, not {_shown(value)}."
        ) from exc


def _blob(value: object, what: str) -> bytes:
    """Return ``value`` as bytes when it is a run of bytes, else refuse it."""
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise CodecError(
            f"{EXTENSION}: {what} must be bytes, not {type(value).__name__}."
        )
    return bytes(value)


def _png_uint(number: float, what: str, given: float) -> int:
    """Round a positive finite ``number`` into a PNG four-byte unsigned integer.

    The specification limits those to 2**31-1.

    ``given`` is the value the caller supplied, which ``number`` was scaled
    from; the messages name it, not the scaled figure.
    """
    if not math.isfinite(given) or given <= 0:
        raise CodecError(
            f"{EXTENSION}: {what} must be positive and finite, not {given!r}."
        )
    # A finite value can scale to infinity, which is too large, not non-finite.
    if not math.isfinite(number) or round(number) > _PNG_UINT_MAX:
        raise CodecError(f"{EXTENSION}: {what} {given!r} is too large for a PNG.")
    rounded = round(number)
    if rounded < 1:
        raise CodecError(
            f"{EXTENSION}: {what} {given!r} rounds to 0, which a PNG cannot hold."
        )
    return rounded


def _text_chunk(keyword: object, value: object) -> bytes:
    """Return a tEXt chunk for latin-1 text, otherwise an uncompressed iTXt."""
    if not isinstance(keyword, str):
        raise CodecError(f"{EXTENSION}: text keyword {_shown(keyword)} must be a str.")
    if not 1 <= len(keyword) <= 79 or "\0" in keyword:
        raise CodecError(
            f"{EXTENSION}: text keyword {keyword!r} must be 1-79 characters with no NUL."
        )
    try:
        key = keyword.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise CodecError(
            f"{EXTENSION}: text keyword {keyword!r} must be latin-1 characters."
        ) from exc
    if _unfit_keyword(key):
        raise CodecError(
            f"{EXTENSION}: text keyword {keyword!r} must be printable Latin-1 with "
            "no leading, trailing or doubled space."
        )
    if not isinstance(value, str):
        raise CodecError(f"{EXTENSION}: text {keyword!r} must be a str.")
    if "\0" in value:
        raise CodecError(
            f"{EXTENSION}: text {keyword!r} holds a NUL, which PNG text cannot."
        )
    try:
        return _chunk(b"tEXt", key + b"\0" + value.encode("latin-1"))
    except UnicodeEncodeError:
        pass
    try:
        utf8 = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CodecError(
            f"{EXTENSION}: text {keyword!r} is not valid Unicode ({exc.reason})."
        ) from exc
    # keyword, NUL, compression flag and method, empty language tag and
    # empty translated keyword, each closed by a NUL, then the UTF-8 text.
    return _chunk(b"iTXt", key + b"\0\0\0\0\0" + utf8)


def _metadata_chunks(meta: dict[str, Any], compress_level: int) -> list[bytes]:
    """Frame the metadata a PNG holds, in the order the spec asks for it."""
    chunks: list[bytes] = []
    if "gamma" in meta:
        gamma = _number(meta["gamma"], "gamma")
        scaled = _png_uint(gamma * _GAMMA_SCALE, "gamma", gamma)
        chunks.append(_chunk(b"gAMA", struct.pack(">I", scaled)))
    if "icc_profile" in meta:
        profile = zlib.compress(
            _blob(meta["icc_profile"], "icc_profile"), compress_level
        )
        chunks.append(_chunk(b"iCCP", b"ICC Profile\0\0" + profile))
    if "dpi" in meta:
        try:
            if isinstance(meta["dpi"], (str, bytes, bytearray)):
                raise TypeError("a run of characters is not a pair of numbers")
            dpi = tuple(meta["dpi"])
        except TypeError as exc:
            raise CodecError(
                f"{EXTENSION}: dpi must be a pair of numbers, not {_shown(meta['dpi'])}."
            ) from exc
        if len(dpi) != 2:
            raise CodecError(
                f"{EXTENSION}: dpi must be a pair of numbers, not {_shown(dpi)}."
            )
        across = _number(dpi[0], "dpi")
        down = _number(dpi[1], "dpi")
        ppx = _png_uint(across / _METERS_PER_INCH, "dpi", across)
        ppy = _png_uint(down / _METERS_PER_INCH, "dpi", down)
        chunks.append(_chunk(b"pHYs", struct.pack(">IIB", ppx, ppy, 1)))
    if "text" in meta:
        text = meta["text"]
        if not isinstance(text, Mapping):
            raise CodecError(
                f"{EXTENSION}: text must be a mapping of keyword to str, not "
                f"{type(text).__name__}."
            )
        for keyword, value in text.items():
            chunks.append(_text_chunk(keyword, value))
    if "exif" in meta:
        # An eXIf chunk holds the TIFF data alone; JPEG's "Exif" ID code, which
        # Pillow and cameras hand over in front of it, is not part of it.
        exif = _blob(meta["exif"], "exif").removeprefix(b"Exif\0\0")
        chunks.append(_chunk(b"eXIf", exif))
    return chunks


def _paeth(left: np.ndarray, up: np.ndarray, upleft: np.ndarray) -> np.ndarray:
    """Paeth predictor over whole rows; ties go to left, then up."""
    estimate = left + up - upleft
    d_left = np.abs(estimate - left)
    d_up = np.abs(estimate - up)
    d_upleft = np.abs(estimate - upleft)
    return np.where(
        (d_left <= d_up) & (d_left <= d_upleft),
        left,
        np.where(d_up <= d_upleft, up, upleft),
    )


def _filter_rows(rows: np.ndarray, bpp: int) -> np.ndarray:
    """Filter every scanline with the cheapest of the five filters.

    The rows are filtered in strips of about ``_FILTER_STRIP_BYTES`` bytes, so
    the 16-bit work arrays are the size of a strip and not of the image.

    Parameters
    ----------
    rows
        ``(height, row_bytes)`` uint8 scanlines.
    bpp
        Bytes per complete pixel, the distance to the left neighbour.

    Returns
    -------
    numpy.ndarray
        ``(height, row_bytes + 1)`` uint8, each row led by its filter type.
        The cost of a filter is the sum of the absolute signed bytes; a tie
        goes to the lower filter number.
    """
    height, row_bytes = rows.shape
    out = np.empty((height, row_bytes + 1), dtype=np.uint8)
    step = max(1, _FILTER_STRIP_BYTES // max(row_bytes, 1))
    for start in range(0, height, step):
        stop = min(start + step, height)
        above = rows[start - 1] if start else None
        out[start:stop] = _filter_strip(rows[start:stop], above, bpp)
    return out


def _filter_strip(rows: np.ndarray, above: np.ndarray | None, bpp: int) -> np.ndarray:
    """Filter consecutive scanlines; ``above`` is the scanline before the first.

    ``above`` is None for the top of the image, where the rows above are zero.
    """
    cur = rows.astype(np.int16)
    left = np.zeros_like(cur)
    left[:, bpp:] = cur[:, :-bpp]
    up = np.zeros_like(cur)
    up[1:] = cur[:-1]
    upleft = np.zeros_like(cur)
    upleft[1:, bpp:] = cur[:-1, :-bpp]
    if above is not None:
        up[0] = above
        upleft[0, bpp:] = above[:-bpp]
    filtered = (
        cur,
        (cur - left) & 0xFF,
        (cur - up) & 0xFF,
        (cur - ((left + up) >> 1)) & 0xFF,
        (cur - _paeth(left, up, upleft)) & 0xFF,
    )
    costs = np.stack(
        [np.minimum(f, 256 - f).sum(axis=1, dtype=np.int64) for f in filtered]
    )
    best = np.argmin(costs, axis=0)
    out = np.empty((rows.shape[0], rows.shape[1] + 1), dtype=np.uint8)
    out[:, 0] = best
    for kind, values in enumerate(filtered):
        pick = best == kind
        out[pick, 1:] = values[pick].astype(np.uint8)
    return out
