"""SPZ codec (compressed 3D Gaussian splats, ``.spz``) - read + write.

An SPZ file is a cloud of Gaussians packed for size: positions as 24-bit
fixed point, opacity, colour and scale as one byte each per component, the
rotation quaternion as three bytes or its smallest three components in four,
and the spherical harmonics as one byte per coefficient. Versions 1 to 3 are
a 16-byte header and the attribute arrays one after the other, the whole
gzip-compressed; version 4 is a 32-byte plaintext header, a table of
contents and each attribute array compressed on its own with zstd.

The Gaussians are read into the attributes a 3D Gaussian splatting PLY file
carries, decoded as a reference implementation decodes them: ``f_dc_0..2``
for the colour's degree-zero harmonic, ``f_rest_<k>`` for the higher ones
with the colour channel slower varying, ``opacity`` as a logit,
``scale_0..2`` as logarithms and ``rot_0..3`` as the quaternion ``w, x, y,
z``. A cloud read here and written as ``.ply`` is a splat PLY other tools
load, and one read from such a PLY writes as SPZ. Coordinates are kept as
the file spells them; no axis convention is converted.
"""

import functools
import gzip
import struct
from types import ModuleType
from typing import Any
import warnings

import numpy as np

from polyxios._dimension import pad_to_3d
from polyxios._io import Source, open_read, source_name, write_bytes
from polyxios._optpkg import TripWire, optional_package
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

try:
    from compression import zstd as _stdlib_zstd
except ImportError:  # Python < 3.14 has no zstd in its standard library
    _stdlib_zstd = None

EXTENSION: str = ".spz"

#: Where a reader records that the splat was trained with antialiasing.
ANTIALIASED_KEY: str = "spz_antialiased"

_MAGIC: bytes = b"NGSP"
_LEGACY = struct.Struct("<4sIIBBBB")
_NGSP = struct.Struct("<4sIIBBBBI12s")
_TOC_ENTRY = struct.Struct("<QQ")
_LATEST_LEGACY: int = 3
_ZSTD_VERSION: int = 4
_SMALLEST_THREE_VERSION: int = 3
_FLOAT16_VERSION: int = 1
_FLAG_ANTIALIASED: int = 0x1
_FLAG_EXTENSIONS: int = 0x2

# Coefficients per colour channel for each harmonic degree.
_SH_DIM: dict[int, int] = {0: 0, 1: 3, 2: 8, 3: 15, 4: 24}
_COLOR_SCALE = np.float32(0.15)
_SQRT1_2 = np.float32(0.707106781186547524401)
_SMALLEST_THREE_MASK: int = (1 << 9) - 1
# The precision the reference writer quantises harmonics to before
# compressing: five bits for degree one, four above. A reader needs neither.
_SH1_BUCKET: int = 1 << (8 - 5)
_SH_REST_BUCKET: int = 1 << (8 - 4)
_DEFAULT_FRACTIONAL_BITS: int = 12
# The reference reader scales by ``1 << fractionalBits`` in a C int, which
# 31 overflows to a negative scale, flipping every position's sign.
_MAX_FRACTIONAL_BITS: int = 30
# The reference writer's zstd level, so a version 4 file comes out as it
# writes one; level 3, zstd's default, is a tenth larger on splat streams.
_ZSTD_LEVEL: int = 12
_FIXED_LIMIT: int = (1 << 23) - 1
_MAX_POINTS: int = (1 << 32) - 1
# The reference reader refuses a version 4 file whose point count, at the
# nine bytes of a position, is past 1024 times its size. zstd reaches 32768
# on constant data, so a writer can make a file it refuses; this reader holds
# each stream to what zstd can reach instead, and the writer warns.
_REFERENCE_RATIO: int = 1024
_MIN_BYTES_PER_POINT: int = 9
_READ_CHUNK: int = 1 << 20
# ``ZSTD_compressBound``'s small-input threshold, and a margin past it.
_ZSTD_SMALL: int = 128 << 10
_ZSTD_SLACK: int = 64
# The largest zstd block, and the fewest bytes one takes: a three-byte block
# header and the one byte of a run.
_ZSTD_BLOCK: int = 128 << 10
_ZSTD_MIN_BLOCK: int = 4
_EXTENSIONS_NOTE: str = "'{name}': the file carries vendor extension records; skipped."

_DC_ATTRS: tuple[str, ...] = ("f_dc_0", "f_dc_1", "f_dc_2")
_SCALE_ATTRS: tuple[str, ...] = ("scale_0", "scale_1", "scale_2")
_ROT_ATTRS: tuple[str, ...] = ("rot_0", "rot_1", "rot_2", "rot_3")
_OPACITY: str = "opacity"
_WRITE_VERSIONS: frozenset[int] = frozenset({2, 3, 4})
_DEFAULT_VERSION: int = 3


@functools.cache
def _zstandard() -> tuple[ModuleType | TripWire, bool]:
    return optional_package("zstandard", extra="zstd")


def _zstd_compress(data: bytes) -> bytes:
    """Compress one stream into a zstd frame that records its own size."""
    if _stdlib_zstd is not None:
        return _stdlib_zstd.compress(data, level=_ZSTD_LEVEL)
    zstandard, _ = _zstandard()
    return zstandard.ZstdCompressor(level=_ZSTD_LEVEL).compress(data)


def _zstd_decompress(data: bytes, size: int, name: str, what: str) -> bytes:
    """Decompress one stream, refusing one that is not exactly ``size`` bytes.

    The output is bounded by ``size`` whatever the frame claims, so a frame
    declaring more than its table of contents does cannot allocate it.
    """
    # Asked for outside the try below, so a missing package raises its own
    # error naming the extra rather than one about the frame.
    decompressor_cls = (
        None if _stdlib_zstd is not None else _zstandard()[0].ZstdDecompressor
    )
    try:
        if decompressor_cls is None:
            stream = _stdlib_zstd.ZstdDecompressor()
            out = stream.decompress(data, max_length=size + 1)
            # This decompressor stops at the end of the first frame, where
            # the other reads on into whatever follows and fails on it.
            if len(out) == size and not stream.eof:
                raise ValueError("it is cut short")
            if stream.unused_data:
                raise ValueError(f"{len(stream.unused_data)} bytes follow its end")
        else:
            reader = decompressor_cls().stream_reader(data)
            chunks: list[bytes] = []
            got = 0
            while got <= size:
                chunk = reader.read(size + 1 - got)
                if not chunk:
                    break
                chunks.append(chunk)
                got += len(chunk)
            out = b"".join(chunks)
    except Exception as exc:  # each implementation raises its own error type
        raise CodecError(
            f"'{name}': the {what} stream is not a valid zstd frame ({exc})."
        ) from exc
    if len(out) != size:
        raise CodecError(
            f"'{name}': the {what} stream holds {len(out)} bytes where the"
            f" table of contents declares {size}."
        )
    return out


def _sizes(n: int, version: int, degree: int) -> list[tuple[str, int]]:
    """The attribute arrays in file order, with their byte sizes."""
    position = 6 if version == _FLOAT16_VERSION else 9
    rotation = 4 if version >= _SMALLEST_THREE_VERSION else 3
    return [
        ("positions", n * position),
        ("alphas", n),
        ("colors", n * 3),
        ("scales", n * 3),
        ("rotations", n * rotation),
        ("sh", n * _SH_DIM[degree] * 3),
    ]


def _inv_sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(x / (np.float32(1) - x))


def _decode(
    arrays: dict[str, bytes],
    n: int,
    version: int,
    degree: int,
    fractional_bits: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Turn the packed arrays into positions and splat PLY attributes."""
    if version == _FLOAT16_VERSION:
        positions = np.frombuffer(arrays["positions"], dtype="<f2").astype(np.float64)
    else:
        raw = np.frombuffer(arrays["positions"], dtype=np.uint8)
        b = raw.reshape(-1, 3).astype(np.int32)
        fixed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        fixed = (fixed ^ 0x800000) - 0x800000
        positions = fixed.astype(np.float64) * 2.0**-fractional_bits
    vertices = positions.reshape(n, 3)

    attrs: dict[str, np.ndarray] = {}
    colors = np.frombuffer(arrays["colors"], dtype=np.uint8).reshape(n, 3)
    dc = ((colors.astype(np.float32) / np.float32(255)) - np.float32(0.5)) / (
        _COLOR_SCALE
    )
    for k, name in enumerate(_DC_ATTRS):
        attrs[name] = dc[:, k].copy()

    dim = _SH_DIM[degree]
    if dim:
        sh = np.frombuffer(arrays["sh"], dtype=np.uint8).reshape(n, dim, 3)
        sh = (sh.astype(np.float32) - np.float32(128)) / np.float32(128)
        for c in range(3):
            for j in range(dim):
                attrs[f"f_rest_{c * dim + j}"] = sh[:, j, c].copy()

    alphas = np.frombuffer(arrays["alphas"], dtype=np.uint8)
    attrs[_OPACITY] = _inv_sigmoid(alphas.astype(np.float32) / np.float32(255))

    scales = np.frombuffer(arrays["scales"], dtype=np.uint8).reshape(n, 3)
    scales = scales.astype(np.float32) / np.float32(16) - np.float32(10)
    for k, name in enumerate(_SCALE_ATTRS):
        attrs[name] = scales[:, k].copy()

    rot = np.frombuffer(arrays["rotations"], dtype=np.uint8)
    if version >= _SMALLEST_THREE_VERSION:
        xyzw = _unpack_smallest_three(rot.reshape(n, 4))
    else:
        # Times the reciprocal, not over 127.5, as the reference decodes: the
        # two differ in the last bit, which ``w`` amplifies near zero.
        xyz = rot.reshape(n, 3).astype(np.float32) * np.float32(1 / 127.5)
        xyz = xyz - np.float32(1)
        w = np.sqrt(np.maximum(np.float32(0), np.float32(1) - (xyz * xyz).sum(axis=1)))
        xyzw = np.column_stack([xyz, w])
    for k, name in zip((3, 0, 1, 2), _ROT_ATTRS):
        attrs[name] = np.ascontiguousarray(xyzw[:, k], dtype=np.float32)
    return vertices, attrs


def _unpack_smallest_three(packed: np.ndarray) -> np.ndarray:
    """Decode ``(n, 4)`` bytes of smallest-three quaternions to ``x, y, z, w``.

    The top two bits name the largest component; the three others follow,
    ten bits each - a sign and nine of magnitude over ``1/sqrt(2)`` - the
    one of highest index in the lowest bits.
    """
    comp = packed.astype(np.uint32)
    comp = comp[:, 0] | (comp[:, 1] << 8) | (comp[:, 2] << 16) | (comp[:, 3] << 24)
    largest = (comp >> 30).astype(np.intp)
    out = np.zeros((comp.size, 4), dtype=np.float32)
    rows = np.arange(comp.size)
    # Walking the components from the highest index down, the one that is
    # the largest is skipped; that makes component ``i``'s field the k-th
    # from the bottom where k counts the smaller indices above it.
    shift = np.zeros(comp.size, dtype=np.uint32)
    sum_squares = np.zeros(comp.size, dtype=np.float32)
    for i in range(3, -1, -1):
        stored = largest != i
        field = (comp >> shift) & np.uint32(0x3FF)
        mag = (field & np.uint32(_SMALLEST_THREE_MASK)).astype(np.float32)
        value = _SQRT1_2 * mag / np.float32(_SMALLEST_THREE_MASK)
        value = np.where((field >> 9) & 1, -value, value)
        out[:, i] = np.where(stored, value, np.float32(0))
        sum_squares += np.where(stored, value * value, np.float32(0))
        shift = shift + np.where(stored, np.uint32(10), np.uint32(0)).astype(np.uint32)
    out[rows, largest] = np.sqrt(np.maximum(np.float32(0), np.float32(1) - sum_squares))
    return out


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an SPZ file and return a point-cloud PolyData.

    Parameters
    ----------
    path
        Path to the ``.spz`` file, or an open binary file object.
    lazy
        Not supported: every attribute is quantised and compressed.

    Returns
    -------
    PolyData
        A point cloud with no elements: float64 positions, and float32
        vertex attributes ``f_dc_0..2``, ``f_rest_<k>`` for each higher
        harmonic coefficient, ``opacity``, ``scale_0..2`` and ``rot_0..3``,
        decoded as the reference implementation decodes them - an opacity
        stored as 0 or 255 reads as an infinite logit, as it does there.
        ``global_attrs["spz_antialiased"]`` is True when the file says the
        splat was trained with antialiasing.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        If the file does not open with the ``NGSP`` magic, is of a version
        other than 1 to 4, declares a harmonic degree past 4, is shorter
        than its header or than the arrays its point count declares, its
        gzip stream is corrupt, or - version 4 - has a table of contents
        that does not match its point count or its own length, a stream
        longer or shorter than zstd ever makes one of its declared size, or
        a stream that does not decompress to the size declared.
    ValidationError
        If the point count is past the safety cap, or past what the file's
        size could hold.
    MissingPackageError
        If the file is version 4 and neither Python's own ``compression.zstd``
        nor the ``zstandard`` package is there to decompress it.

    Notes
    -----
    Vendor extension records are skipped with a warning: in a version 1 to 3
    file they follow the arrays, in a version 4 file they sit between the
    header and the table of contents. Other bytes following the arrays of a
    version 1 to 3 file, or before the table of a version 4 file with no
    extension flag, are skipped with a warning too; in a version 4 file,
    where nothing follows the last stream, they are refused, as the
    reference reader refuses them. A version 4 point count is held to what
    zstd can pack into each stream's declared length, not to the reference
    reader's tighter 1024:1, so a file of many repeated points that reader
    refuses reads here. Skipped bytes are counted as they are
    read and never held, so a gzip stream inflating to far more than the
    splat needs costs time, not memory.
    """
    if lazy:
        raise LazyReadError(
            "SPZ is quantised and compressed and cannot be memory-mapped."
        )
    name = source_name(path)
    # A version 1-3 file is gzip, which the outer open unwraps by content;
    # one gzipped again, a ``.spz.gz``, still holds that member inside, and
    # the inner open unwraps it.
    with open_read(path) as outer, open_read(outer) as fh:
        head = _take(fh, _LEGACY.size, name)
        if len(head) < _LEGACY.size or head[:4] != _MAGIC:
            raise CodecError(f"'{name}': no NGSP header; not an SPZ file.")
        version = struct.unpack_from("<I", head, 4)[0]
        if version == _ZSTD_VERSION:
            poly, note = _read_ngsp(fh, head, name)
        elif 1 <= version <= _LATEST_LEGACY:
            poly, note = _read_legacy(fh, head, name)
        else:
            raise CodecError(
                f"'{name}': SPZ version {version} is not one this codec reads (1 to 4)."
            )
    if note:
        warnings.warn(note, stacklevel=2)
    return poly


def _take(fh: Any, size: int, name: str) -> bytes:
    """Read ``size`` bytes, or all that is left when fewer remain.

    The read goes in chunks, so a size the file declares but does not hold
    allocates only what is there.
    """
    chunks: list[bytes] = []
    got = 0
    try:
        while got < size:
            chunk = fh.read(min(_READ_CHUNK, size - got))
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
    except CodecError as exc:
        raise CodecError(f"'{name}': the gzip stream is corrupt ({exc}).") from exc
    return b"".join(chunks)


def _skip(fh: Any, name: str, *, limit: int | None = None) -> int:
    """Read on to the end, or past ``limit`` bytes, and count what was read."""
    got = 0
    try:
        while limit is None or got < limit:
            want = _READ_CHUNK if limit is None else min(_READ_CHUNK, limit - got)
            chunk = fh.read(want)
            if not chunk:
                break
            got += len(chunk)
    except CodecError as exc:
        raise CodecError(f"'{name}': the gzip stream is corrupt ({exc}).") from exc
    return got


def _read_legacy(fh: Any, head: bytes, name: str) -> tuple[PolyData, str | None]:
    """Read a version 1 to 3 file, gzip already unwrapped, past its header."""
    _, version, n, degree, fractional_bits, flags, _ = _LEGACY.unpack(head)
    _check_degree(degree, name)
    sizes = _sizes(n, version, degree)
    needed = sum(size for _, size in sizes)
    validate_header(n, 0, 0, _LEGACY.size, compressed=True)
    body = _take(fh, needed, name)
    if needed > len(body):
        raise CodecError(
            f"'{name}': the header declares {n} points, {needed} bytes of"
            f" arrays, and the file holds {len(body)} after its header."
        )
    arrays: dict[str, bytes] = {}
    pos = 0
    for key, size in sizes:
        arrays[key] = body[pos : pos + size]
        pos += size
    trailing = _skip(fh, name)
    note = None
    if flags & _FLAG_EXTENSIONS:
        note = _EXTENSIONS_NOTE.format(name=name)
    elif trailing:
        note = f"'{name}': {trailing} bytes follow the splat arrays; ignored."
    return _cloud(arrays, n, version, degree, fractional_bits, flags), note


def _read_ngsp(fh: Any, head: bytes, name: str) -> tuple[PolyData, str | None]:
    """Read a version 4 file: a plaintext header, a table, zstd streams."""
    head += _take(fh, _NGSP.size - len(head), name)
    if len(head) < _NGSP.size:
        raise CodecError(f"'{name}': the file is shorter than its 32-byte header.")
    _, version, n, degree, fractional_bits, flags, n_streams, toc, _ = _NGSP.unpack(
        head
    )
    _check_degree(degree, name)
    validate_header(n, 0, 0, _NGSP.size, compressed=True)
    expected = [(key, size) for key, size in _sizes(n, version, degree) if size]
    if n_streams != len(expected):
        raise CodecError(
            f"'{name}': the header declares {n_streams} streams where {n} points"
            f" of harmonic degree {degree} need {len(expected)}."
        )
    # The extension records are counted, not kept; nothing reads them.
    extensions = _skip(fh, name, limit=max(toc - _NGSP.size, 0))
    table = _take(fh, n_streams * _TOC_ENTRY.size, name)
    length = _NGSP.size + extensions + len(table)
    if (
        toc < _NGSP.size
        or extensions < toc - _NGSP.size
        or len(table) < len(expected) * _TOC_ENTRY.size
    ):
        raise CodecError(
            f"'{name}': the table of contents at byte {toc} does not fit the"
            f" {length + _skip(fh, name)}-byte file."
        )
    # The whole table is checked before any stream is decompressed, so a
    # file that cannot be whole costs no decompression.
    packed_streams: list[tuple[str, bytes, int]] = []
    for k, (key, size) in enumerate(expected):
        packed, unpacked = _TOC_ENTRY.unpack_from(table, k * _TOC_ENTRY.size)
        if unpacked != size:
            raise CodecError(
                f"'{name}': the {key} stream declares {unpacked} bytes where"
                f" {n} points need {size}."
            )
        # A stream shorter than zstd ever makes of its size is refused
        # before it is read, so a header claiming many points in a small
        # file costs nothing.
        if packed < _zstd_floor(size):
            raise CodecError(
                f"'{name}': the {key} stream declares {packed} bytes, fewer than"
                f" zstd ever makes of {size}."
            )
        # Read no further than zstd could need for the size, so a table
        # declaring a stream of any length allocates no more than the
        # point count asks for.
        blob = _take(fh, min(packed, _zstd_bound(size) + 1), name)
        if len(blob) < min(packed, _zstd_bound(size) + 1):
            raise CodecError(
                f"'{name}': the {key} stream runs past the end of the file."
            )
        if len(blob) > _zstd_bound(size):
            raise CodecError(
                f"'{name}': the {key} stream declares {packed} bytes, more than"
                f" zstd ever makes of {size}."
            )
        packed_streams.append((key, blob, size))
    trailing = _skip(fh, name)
    if trailing:
        raise CodecError(
            f"'{name}': {trailing} bytes follow the last stream the table"
            " of contents names."
        )
    arrays = {key: b"" for key, _ in _sizes(n, version, degree)}
    for key, blob, size in packed_streams:
        arrays[key] = _zstd_decompress(blob, size, name, key)
    # Version 4 keeps its extension records between the header and the
    # table of contents, so a flag over no such bytes has nothing skipped.
    note = None
    if flags & _FLAG_EXTENSIONS and extensions:
        note = _EXTENSIONS_NOTE.format(name=name)
    elif extensions:
        note = (
            f"'{name}': {extensions} bytes lie between the header and the table"
            " of contents, with no extension flag; skipped."
        )
    return _cloud(arrays, n, version, degree, fractional_bits, flags), note


def _zstd_bound(size: int) -> int:
    """The longest frame zstd makes of ``size`` bytes, as ``ZSTD_compressBound``.

    A frame header and a checksum are within the margin of the small-input
    term, and the slack covers a dictionary id or a frame from an encoder
    that spells its block headers less tightly.
    """
    small = (_ZSTD_SMALL - size) >> 11 if size < _ZSTD_SMALL else 0
    return size + (size >> 8) + small + _ZSTD_SLACK


def _zstd_floor(size: int) -> int:
    """The shortest frame zstd makes of ``size`` bytes, frame header aside.

    Every block holds at most 128 KiB and takes at least four bytes, a run
    of one repeated byte, so no frame packs ``size`` bytes into fewer.
    """
    return _ZSTD_MIN_BLOCK * max(1, -(-size // _ZSTD_BLOCK))


def _check_degree(degree: int, name: str) -> None:
    if degree not in _SH_DIM:
        raise CodecError(
            f"'{name}': spherical harmonic degree {degree} is past the 4 the"
            " format defines."
        )


def _cloud(
    arrays: dict[str, bytes],
    n: int,
    version: int,
    degree: int,
    fractional_bits: int,
    flags: int,
) -> PolyData:
    vertices, attrs = _decode(arrays, n, version, degree, fractional_bits)
    global_attrs: dict[str, Any] = {}
    if flags & _FLAG_ANTIALIASED:
        global_attrs[ANTIALIASED_KEY] = True
    return PolyData(
        vertices=vertices,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=attrs,
        element_attrs={},
        global_attrs=global_attrs,
    )


def _round(x: np.ndarray) -> np.ndarray:
    """Round half away from zero, as C's ``round`` does.

    The fraction is taken off the truncated value, which is exact, rather
    than ``0.5`` added before flooring, which rounds the largest value below
    one half up to one.
    """
    whole = np.trunc(x)
    with np.errstate(invalid="ignore"):
        half = np.abs(x - whole) >= 0.5
    return whole + np.where(half, np.sign(x), 0).astype(x.dtype)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.clip(np.nan_to_num(_round(x), nan=0.0), 0, 255).astype(np.uint8)


def _column(
    attrs: dict[str, np.ndarray], key: str, n: int, missing: list[str]
) -> np.ndarray:
    """One attribute as float32, zeros when the cloud lacks it."""
    if key not in attrs:
        missing.append(key)
        return np.zeros(n, dtype=np.float32)
    values = np.asarray(attrs[key])
    if values.shape != (n,) or values.dtype.kind not in "biuf":
        raise CodecError(
            f"{EXTENSION}: vertex attribute {key!r} has shape {values.shape} and"
            f" dtype {values.dtype}; SPZ takes one number per Gaussian."
        )
    return values.astype(np.float32)


def _fits(low: float, high: float, bits: int) -> bool:
    """Say whether ``low`` and ``high`` round into a signed 24-bit integer."""
    scale = 2.0**bits
    return bool(
        _round(np.float64(high) * scale) <= _FIXED_LIMIT
        and _round(np.float64(low) * scale) >= -_FIXED_LIMIT - 1
    )


def _fractional_bits(vertices: np.ndarray, chosen: object) -> int:
    """The fractional bits the positions are written with.

    Twelve, as the reference writer uses, unless the cloud reaches past the
    ``[-2048, 2048)`` twelve leave a signed 24-bit integer; then as many as
    still fit. The range is a two's complement one, so ``-2048`` itself
    takes twelve.
    """
    low = float(vertices.min()) if vertices.size else 0.0
    high = float(vertices.max()) if vertices.size else 0.0
    if chosen is not None:
        if (
            not isinstance(chosen, (int, np.integer))
            or isinstance(chosen, bool)
            or not 0 <= chosen <= _MAX_FRACTIONAL_BITS
        ):
            raise CodecError(
                f"{EXTENSION}: fractional_bits= takes an integer from 0 to"
                f" {_MAX_FRACTIONAL_BITS},"
                f" not {chosen!r}."
            )
        bits = int(chosen)
    else:
        bits = _DEFAULT_FRACTIONAL_BITS
        while bits > 0 and not _fits(low, high, bits):
            bits -= 1
    if not _fits(low, high, bits):
        reach = high if abs(high) >= abs(low) else low
        raise CodecError(
            f"{EXTENSION}: a position reaches {reach}, past what a 24-bit fixed"
            f" point number with {bits} fractional bits holds."
        )
    return bits


def _sh_layout(attrs: dict[str, np.ndarray]) -> tuple[int, int]:
    """The harmonic degree written and the stride between colour channels.

    The ``f_rest_<k>`` run, up to the 72 coefficients of degree 4, is split
    into three channels of a third of it each, as the reference PLY reader
    splits it; the degree is the highest whose coefficients fit in a third,
    and the coefficients of each channel past it are dropped.
    """
    count = 0
    while count < 3 * _SH_DIM[4] and f"f_rest_{count}" in attrs:
        count += 1
    stride = count // 3
    return max(d for d, dim in _SH_DIM.items() if dim <= stride), stride


def _attr_order(name: str) -> tuple[str, int]:
    """Sort key putting ``f_rest_10`` after ``f_rest_9``."""
    head, _, tail = name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (name, -1)


def _pack_smallest_three(xyzw: np.ndarray) -> np.ndarray:
    """Encode unit ``x, y, z, w`` quaternions as smallest-three bytes."""
    largest = np.argmax(np.abs(xyzw), axis=1)
    rows = np.arange(xyzw.shape[0])
    negate = xyzw[rows, largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        stored = largest != i
        q = xyzw[:, i]
        negbit = ((q < 0) ^ negate).astype(np.uint32)
        mag = np.float32(_SMALLEST_THREE_MASK) * (np.abs(q) / _SQRT1_2) + np.float32(
            0.5
        )
        mag = np.minimum(mag.astype(np.uint32), np.uint32(_SMALLEST_THREE_MASK))
        packed = (comp << np.uint32(10)) | (negbit << np.uint32(9)) | mag
        comp = np.where(stored, packed, comp)
    out = np.empty((xyzw.shape[0], 4), dtype=np.uint8)
    for k in range(4):
        out[:, k] = (comp >> np.uint32(8 * k)) & np.uint32(0xFF)
    return out


def _pack_first_three(xyzw: np.ndarray) -> np.ndarray:
    """Encode unit quaternions as ``x, y, z`` bytes with ``w`` made positive."""
    sign = np.where(xyzw[:, 3] < 0, np.float32(-127.5), np.float32(127.5))
    q = xyzw * sign[:, None] + np.float32(127.5)
    return _to_uint8(q[:, :3])


def _quantise_sh(x: np.ndarray, bucket: int) -> np.ndarray:
    """Quantise harmonics to bytes, saturating infinities and NaN as zero."""
    q = _round(x * np.float32(128)) + np.float32(128)
    q = np.clip(np.nan_to_num(q, nan=128.0), -1, 256).astype(np.int64)
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def _antialiasing_flag(value: object) -> int:
    """The header flag for ``global_attrs["spz_antialiased"]``.

    A boolean, or a 0 or 1 - also as a one-element array, the shape other
    formats give a global back in - sets or clears the flag; any other
    value clears it, with a warning, where its truthiness would set it.
    """
    arr = np.asarray(value)
    if arr.size == 1 and arr.dtype.kind in "biu" and arr.item() in (0, 1):
        return _FLAG_ANTIALIASED if arr.item() else 0
    warnings.warn(
        f"{EXTENSION}: global attribute {ANTIALIASED_KEY!r} is {value!r}, not a"
        " boolean; the antialiasing flag is not set.",
        stacklevel=3,
    )
    return 0


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise a Gaussian-splat PolyData to an SPZ file.

    Parameters
    ----------
    poly
        Point cloud to write, 2D positions padded to z = 0. Its splat
        attributes are the ones a 3D Gaussian splatting PLY carries:
        ``f_dc_0..2``, ``f_rest_<k>``, ``opacity`` (a logit), ``scale_0..2``
        (logarithms) and ``rot_0..3`` (``w, x, y, z``).
    path
        Output ``.spz`` path, or an open binary file object.
    **opts
        ``version``: 2, 3 or 4. 3 by default: gzip, rotations as their
        smallest three components. 2 stores rotations in three bytes, for
        readers older than version 3; 4 is the zstd layout, which needs
        Python 3.14 or the ``zstandard`` package.
        ``fractional_bits``: the fractional bits of the 24-bit fixed point
        positions, 0 to 30, chosen from the positions' reach when not
        given; the reference reader's scale overflows at 31. Any
        other option is warned about and ignored.

    Raises
    ------
    CodecError
        If ``version`` or ``fractional_bits`` is not one this codec writes,
        the cloud holds more points than the header's 32-bit count, a
        position is not finite or reaches past what the 24 bits hold, or a
        splat attribute is not one number per Gaussian.
    MissingPackageError
        If ``version=4`` and no zstd implementation is installed.

    Notes
    -----
    Every value is quantised as the reference writer quantises it -
    harmonics of degree one to five bits, higher ones to four - so a write
    loses precision a read does not give back. A splat attribute the cloud
    lacks is written as zero, with a warning, and a rotation of zero length
    or not finite as the identity, with a warning. Any other value past
    what its byte holds, infinities included, saturates; a NaN harmonic
    coefficient is written as zero and any other NaN as the lowest byte.
    The ``f_rest_<k>`` run is split into three colour channels of a third
    of it each, as a splat PLY lays them out, and each channel's
    coefficients past the last whole degree are dropped with a warning, as
    are every other vertex attribute, the elements and the tags.
    ``global_attrs["spz_antialiased"]`` sets the antialiasing flag when it
    is True or 1; a value that is not a boolean leaves it clear, with a
    warning. A cloud of no point is written, with a warning: this codec
    reads it back, the reference reader refuses it. So is a version 4 file
    holding more than 1024 times its size in positions, as a cloud of many
    repeated points compresses: the reference reader refuses it too.
    """
    version = opts.pop("version", _DEFAULT_VERSION)
    chosen_bits = opts.pop("fractional_bits", None)
    if opts:
        warnings.warn(
            f"{EXTENSION} write: unrecognized options {set(opts)}; ignored.",
            stacklevel=2,
        )
    if (
        isinstance(version, bool)
        or not isinstance(version, (int, np.integer))
        or version not in _WRITE_VERSIONS
    ):
        raise CodecError(f"{EXTENSION}: version= takes 2, 3 or 4, not {version!r}.")

    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise CodecError(
            f"{EXTENSION}: vertices have shape {vertices.shape}, expected (n, 2)"
            " or (n, 3)."
        )
    if vertices.shape[1] == 2:
        vertices = pad_to_3d(vertices, 2)
    if not np.isfinite(vertices).all():
        raise CodecError(
            f"{EXTENSION}: a position is not finite; fixed point cannot hold it."
        )
    n = vertices.shape[0]
    if n > _MAX_POINTS:
        raise CodecError(
            f"{EXTENSION}: {n} points are past the {_MAX_POINTS} the header's"
            " 32-bit count holds."
        )
    attrs = poly.vertex_attrs or {}
    bits = _fractional_bits(vertices, chosen_bits)
    degree, stride = _sh_layout(attrs)
    dim = _SH_DIM[degree]
    missing: list[str] = []

    fixed = _round(vertices * 2.0**bits).astype(np.int64).reshape(-1)
    positions = np.empty((fixed.size, 3), dtype=np.uint8)
    for k in range(3):
        positions[:, k] = (fixed >> (8 * k)) & 0xFF

    opacity = _column(attrs, _OPACITY, n, missing)
    with np.errstate(over="ignore"):
        sigmoid = np.float32(1) / (np.float32(1) + np.exp(-opacity))
    alphas = _to_uint8(sigmoid * np.float32(255))

    dc = np.column_stack([_column(attrs, key, n, missing) for key in _DC_ATTRS])
    colors = _to_uint8(
        dc * (_COLOR_SCALE * np.float32(255)) + np.float32(0.5) * np.float32(255)
    )

    scale = np.column_stack([_column(attrs, key, n, missing) for key in _SCALE_ATTRS])
    scales = _to_uint8((scale + np.float32(10)) * np.float32(16))

    wxyz = np.column_stack([_column(attrs, key, n, missing) for key in _ROT_ATTRS])
    no_rotation = all(key in missing for key in _ROT_ATTRS)
    if no_rotation:
        wxyz[:, 0] = 1
    xyzw = wxyz[:, [1, 2, 3, 0]]
    norm = np.sqrt((xyzw * xyzw).sum(axis=1))
    degenerate = ~np.isfinite(norm) | (norm == 0)
    xyzw = np.where(degenerate[:, None], np.float32(0), xyzw)
    xyzw[degenerate, 3] = 1
    norm[degenerate] = 1
    xyzw = (xyzw / norm[:, None]).astype(np.float32)
    if version >= _SMALLEST_THREE_VERSION:
        rotations = _pack_smallest_three(xyzw)
    else:
        rotations = _pack_first_three(xyzw)

    sh = np.zeros((n, dim, 3), dtype=np.uint8)
    for c in range(3):
        for j in range(dim):
            values = _column(attrs, f"f_rest_{c * stride + j}", n, missing)
            bucket = _SH1_BUCKET if j < 3 else _SH_REST_BUCKET
            sh[:, j, c] = _quantise_sh(values, bucket)

    flags = _antialiasing_flag((poly.global_attrs or {}).get(ANTIALIASED_KEY, False))
    streams = [
        positions.tobytes(),
        alphas.tobytes(),
        colors.tobytes(),
        scales.tobytes(),
        rotations.tobytes(),
        sh.tobytes(),
    ]

    known = {_OPACITY, *_DC_ATTRS, *_SCALE_ATTRS, *_ROT_ATTRS}
    known.update(f"f_rest_{c * stride + j}" for c in range(3) for j in range(dim))
    dropped = sorted(set(attrs) - known, key=_attr_order)
    if missing:
        warnings.warn(
            f"{EXTENSION}: the cloud carries no {sorted(missing)}; written as"
            " zero" + (", the rotation as the identity." if no_rotation else "."),
            stacklevel=2,
        )
    if int(degenerate.sum()) and not no_rotation:
        warnings.warn(
            f"{EXTENSION}: {int(degenerate.sum())} rotation(s) are of zero length"
            " or not finite; written as the identity.",
            stacklevel=2,
        )
    if dropped:
        warnings.warn(
            f"{EXTENSION}: vertex attribute(s) {dropped} have no place in a splat"
            " record; dropped.",
            stacklevel=2,
        )
    if not n:
        warnings.warn(
            f"{EXTENSION}: the cloud holds no point; written, though the"
            " reference reader refuses a file of none.",
            stacklevel=2,
        )
    if len(poly.element_types) or poly.vertex_tags or poly.element_tags:
        warnings.warn(
            f"{EXTENSION}: a splat file holds points only; elements and tags"
            " were dropped.",
            stacklevel=2,
        )

    if version == _ZSTD_VERSION:
        kept = [stream for stream in streams if stream]
        packed = [_zstd_compress(stream) for stream in kept]
        header = _NGSP.pack(
            _MAGIC, version, n, degree, bits, flags, len(kept), _NGSP.size, bytes(12)
        )
        toc = b"".join(_TOC_ENTRY.pack(len(p), len(s)) for p, s in zip(packed, kept))
        data = header + toc + b"".join(packed)
        if n * _MIN_BYTES_PER_POINT > len(data) * _REFERENCE_RATIO:
            warnings.warn(
                f"{EXTENSION}: {n} points compress to {len(data)} bytes, past the"
                f" {_REFERENCE_RATIO}:1 the reference reader accepts; written,"
                " though it refuses the file. version=3 is not bounded so.",
                stacklevel=2,
            )
        write_bytes(path, data)
        return
    header = _LEGACY.pack(_MAGIC, version, n, degree, bits, flags, 0)
    # Level 6, zlib's default and the reference writer's: level 9 takes five
    # times as long for a file a percent smaller.
    payload = header + b"".join(streams)
    write_bytes(path, gzip.compress(payload, compresslevel=6, mtime=0))
