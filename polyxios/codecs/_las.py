"""LAS (ASPRS LiDAR) codec: versions 1.0-1.4, point formats 0-10, LAZ via lazrs."""

from __future__ import annotations

from collections.abc import Mapping
import functools
import io
import mmap
import struct
from types import ModuleType
from typing import Any, NamedTuple
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import Source, format_suffix, open_block, open_write, source_name
from polyxios._optpkg import TripWire, optional_package
from polyxios._types import PolyData
from polyxios.codecs._pointcloud import point_cloud
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".las"
EXTENSIONS: tuple[str, ...] = (".las", ".laz")

_SIGNATURE: bytes = b"LASF"
_VERSIONS_READ: frozenset[tuple[int, int]] = frozenset(
    {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)}
)
_VERSIONS_WRITE: dict[str, tuple[int, int]] = {
    "1.0": (1, 0),
    "1.1": (1, 1),
    "1.2": (1, 2),
    "1.3": (1, 3),
    "1.4": (1, 4),
}
_HEADER_SIZES: dict[tuple[int, int], int] = {
    (1, 0): 227,
    (1, 1): 227,
    (1, 2): 227,
    (1, 3): 235,
    (1, 4): 375,
}
_HEADER_1_2: str = "<4sHHIHH8sBB32s32sHHHIIBHI5I3d3d6d"
_VLR_HEADER: str = "<H16sHH32s"
_EVLR_HEADER: str = "<H16sHQ32s"
_VLR_HEADER_SIZE: int = struct.calcsize(_VLR_HEADER)
_EVLR_HEADER_SIZE: int = struct.calcsize(_EVLR_HEADER)
_VLR_MAX_BYTES: int = 0xFFFF
_COMPRESSED_BIT: int = 0x80
_WKT_BIT: int = 0x10
_INT32_MIN: int = -(2**31)
_INT32_MAX: int = 2**31 - 1
_U8_MAX: int = 0xFF
_U16_MAX: int = 0xFFFF
_U32_MAX: int = 2**32 - 1
# LAS 1.0 asked every VLR header to open with this; later versions zero it.
_VLR_RESERVED_1_0: int = 0xAABB
# Identical points, the most compressible input, measured at 50000-point
# chunks: LASzip packs about 150 per byte in format 0 and 85 to 100 in
# formats 3, 6 and 8. A header claiming past this many is lying.
_LAZ_MAX_POINTS_PER_BYTE: int = 1024
# The stream is expanded this many points at a time, so a header lying
# about its count costs one piece of memory before the stream runs dry.
_LAZ_PIECE_POINTS: int = 1 << 16

_LASZIP_USER: str = "laszip encoded"
_LASZIP_ID: int = 22204
_SPEC_USER: str = "LASF_Spec"
_EXTRA_BYTES_ID: int = 4
_PROJECTION_USER: str = "LASF_Projection"
_GEOKEYS_ID: int = 34735
_GEODOUBLES_ID: int = 34736
_GEOASCII_ID: int = 34737
_WKT_ID: int = 2112
_EXTRA_DESCRIPTOR: str = "<2sBB32s4s24s24s24s3d3d32s"
_EXTRA_DESCRIPTOR_SIZE: int = struct.calcsize(_EXTRA_DESCRIPTOR)
_EXTRA_SCALE_BIT: int = 0x08
_EXTRA_OFFSET_BIT: int = 0x10
_EXTRA_NAME_BYTES: int = 32
_HEADER_TEXT_BYTES: int = 32

_VERTEX_CODE: int = ELEMENT_TYPES["vertex"]
_COLOR_SCALE: float = 65535.0
_BYTE_TO_16: int = 257
# A scan angle is a whole degree in the legacy formats and a count of
# 0.006 degrees from format 6 on.
_ANGLE_UNIT: float = 0.006

_KEY_VERSION: str = "las_version"
_KEY_FORMAT: str = "las_point_format"
_KEY_SCALE: str = "las_scale"
_KEY_OFFSET: str = "las_offset"
_KEY_SYSTEM: str = "las_system_identifier"
_KEY_SOFTWARE: str = "las_generating_software"
_KEY_SOURCE_ID: str = "las_file_source_id"
_KEY_ENCODING: str = "las_global_encoding"
_KEY_DATE: str = "las_creation_date"
_KEY_VLRS: str = "las_vlrs"
_KEY_WKT: str = "crs_wkt"
_KEY_GEOKEYS: str = "las_geokeys"
_KEY_EXTRA_BYTES: str = "extra_bytes"
_DEFAULT_SOFTWARE: str = "polyxios"
_DEFAULT_SCALE: float = 0.001

_COLORS: str = "colors"
_RGB: tuple[str, str, str] = ("red", "green", "blue")
_WAVE_FIELDS: tuple[tuple[str, str], ...] = (
    ("wavepacket_index", "u1"),
    ("wavepacket_offset", "<u8"),
    ("wavepacket_size", "<u4"),
    ("return_point_wave_location", "<f4"),
    ("x_t", "<f4"),
    ("y_t", "<f4"),
    ("z_t", "<f4"),
)
_LEGACY_BASE: tuple[tuple[str, str], ...] = (
    ("X", "<i4"),
    ("Y", "<i4"),
    ("Z", "<i4"),
    ("intensity", "<u2"),
    ("returns", "u1"),
    ("raw_classification", "u1"),
    ("scan_angle_rank", "i1"),
    ("user_data", "u1"),
    ("point_source_id", "<u2"),
)
_NEW_BASE: tuple[tuple[str, str], ...] = (
    ("X", "<i4"),
    ("Y", "<i4"),
    ("Z", "<i4"),
    ("intensity", "<u2"),
    ("returns", "u1"),
    ("flags", "u1"),
    ("classification", "u1"),
    ("user_data", "u1"),
    ("scan_angle", "<i2"),
    ("point_source_id", "<u2"),
    ("gps_time", "<f8"),
)
_GPS: tuple[tuple[str, str], ...] = (("gps_time", "<f8"),)
_RGB_FIELDS: tuple[tuple[str, str], ...] = tuple((c, "<u2") for c in _RGB)
_NIR: tuple[tuple[str, str], ...] = (("nir", "<u2"),)

_FORMAT_FIELDS: dict[int, tuple[tuple[str, str], ...]] = {
    0: _LEGACY_BASE,
    1: _LEGACY_BASE + _GPS,
    2: _LEGACY_BASE + _RGB_FIELDS,
    3: _LEGACY_BASE + _GPS + _RGB_FIELDS,
    4: _LEGACY_BASE + _GPS + _WAVE_FIELDS,
    5: _LEGACY_BASE + _GPS + _RGB_FIELDS + _WAVE_FIELDS,
    6: _NEW_BASE,
    7: _NEW_BASE + _RGB_FIELDS,
    8: _NEW_BASE + _RGB_FIELDS + _NIR,
    9: _NEW_BASE + _WAVE_FIELDS,
    10: _NEW_BASE + _RGB_FIELDS + _NIR + _WAVE_FIELDS,
}
_RECORD_DTYPES: dict[int, np.dtype] = {
    pf: np.dtype(list(fields)) for pf, fields in _FORMAT_FIELDS.items()
}
_RECORD_LENGTHS: dict[int, int] = {pf: dt.itemsize for pf, dt in _RECORD_DTYPES.items()}
_MIN_VERSION: dict[int, tuple[int, int]] = {
    0: (1, 0),
    1: (1, 0),
    2: (1, 2),
    3: (1, 2),
    4: (1, 3),
    5: (1, 3),
    6: (1, 4),
    7: (1, 4),
    8: (1, 4),
    9: (1, 4),
    10: (1, 4),
}
# 1.0 and 1.1 are written only when asked for: 1.2 is the first version
# with a global encoding field, and the one other readers expect.
_DEFAULT_VERSION: tuple[int, int] = (1, 2)
_NEW_STYLE: frozenset[int] = frozenset(range(6, 11))
_WITH_WAVE: frozenset[int] = frozenset({4, 5, 9, 10})

# Bit fields: attribute name -> (packed field, shift, width), per style.
_LEGACY_BITS: tuple[tuple[str, str, int, int], ...] = (
    ("return_number", "returns", 0, 3),
    ("number_of_returns", "returns", 3, 3),
    ("scan_direction_flag", "returns", 6, 1),
    ("edge_of_flight_line", "returns", 7, 1),
    ("classification", "raw_classification", 0, 5),
    ("synthetic", "raw_classification", 5, 1),
    ("key_point", "raw_classification", 6, 1),
    ("withheld", "raw_classification", 7, 1),
)
_NEW_BITS: tuple[tuple[str, str, int, int], ...] = (
    ("return_number", "returns", 0, 4),
    ("number_of_returns", "returns", 4, 4),
    ("synthetic", "flags", 0, 1),
    ("key_point", "flags", 1, 1),
    ("withheld", "flags", 2, 1),
    ("overlap", "flags", 3, 1),
    ("scanner_channel", "flags", 4, 2),
    ("scan_direction_flag", "flags", 6, 1),
    ("edge_of_flight_line", "flags", 7, 1),
)
_PACKED: frozenset[str] = frozenset({"returns", "raw_classification", "flags"})
_COORDS: frozenset[str] = frozenset({"X", "Y", "Z"})
_DEFAULT_ONE: frozenset[str] = frozenset({"return_number", "number_of_returns"})
_RESERVED: frozenset[str] = _PACKED | _COORDS | frozenset(_RGB)

_NEW_ONLY: frozenset[str] = frozenset(
    {"nir", "overlap", "scanner_channel", "scan_angle"}
)
_WAVE_NAMES: frozenset[str] = frozenset(name for name, _ in _WAVE_FIELDS)
_STANDARD: frozenset[str] = (
    (
        frozenset(name for fields in _FORMAT_FIELDS.values() for name, _ in fields)
        | frozenset(name for name, *_ in _LEGACY_BITS + _NEW_BITS)
        | {_COLORS}
    )
    - _PACKED
    - _COORDS
    - frozenset(_RGB)
)

# Extra bytes data types: the spec's code -> a numpy dtype and a width.
_EXTRA_SCALARS: tuple[str, ...] = (
    "u1", "i1", "<u2", "<i2", "<u4", "<i4", "<u8", "<i8", "<f4", "<f8",
)  # fmt: skip
_EXTRA_TYPES: dict[int, tuple[str, int]] = {0: ("u1", 0)}
for _i, _dt in enumerate(_EXTRA_SCALARS, start=1):
    _EXTRA_TYPES[_i] = (_dt, 1)
    _EXTRA_TYPES[_i + 10] = (_dt, 2)
    _EXTRA_TYPES[_i + 20] = (_dt, 3)
_EXTRA_CODES: dict[tuple[str, int], int] = {v: k for k, v in _EXTRA_TYPES.items() if k}

_GEOKEY_DIRECTORY: str = "<4H"


class _Vlr(NamedTuple):
    user_id: str
    record_id: int
    description: str
    data: bytes


class _Header(NamedTuple):
    version: tuple[int, int]
    point_format: int
    compressed: bool
    record_length: int
    n_points: int
    offset: int
    scale: np.ndarray
    origin: np.ndarray
    vlrs: list[_Vlr]
    globals: dict[str, Any]


class _Extra(NamedTuple):
    name: str
    dtype: str
    width: int
    scale: np.ndarray | None
    offset: np.ndarray | None


class _ExtraOut(NamedTuple):
    name: str
    dtype: Any
    column: np.ndarray
    code: int
    options: int


# ----- lazrs plumbing ----------------------------------------------------------


@functools.cache
def _lazrs() -> tuple[ModuleType | TripWire, bool]:
    return optional_package("lazrs", extra="laz")


# ----- reading -------------------------------------------------------------------


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a LAS or LAZ file and return a point-cloud PolyData.

    Parameters
    ----------
    path
        Path or binary file object of the ``.las`` or ``.laz`` file. Which
        it is comes from the header, not the name: a compressed file under
        either name is decompressed with lazrs (the ``laz`` extra).
    lazy
        Map an uncompressed file and hand back arrays that view it,
        read-only, in the file's own dtypes: the vertices are the ``(n, 3)``
        int32 the file stores, to be scaled by ``las_scale`` and shifted by
        ``las_offset`` from ``global_attrs``, the colours the ``uint16``
        triples and every other field a view of its own column. The bit
        fields of the returns and flags bytes are decoded into small arrays
        of their own, and an extra byte carrying a scale or offset is left
        raw with a warning. A compressed file raises ``LazyReadError``.
        Written back with :func:`write`, the integer vertices go out as
        they are, so the round trip is lossless.

    Returns
    -------
    PolyData
        Points only, no elements. ``X Y Z`` scaled and offset are the
        vertices; ``intensity``, ``return_number``, ``number_of_returns``,
        ``scan_direction_flag``, ``edge_of_flight_line``, ``classification``,
        ``synthetic``, ``key_point``, ``withheld``, ``user_data``,
        ``point_source_id`` are vertex attributes in every format, with
        ``scan_angle_rank`` (whole degrees, int8) in formats 0-5 and
        ``scan_angle`` (0.006 degree steps, int16), ``overlap`` and
        ``scanner_channel`` from format 6; ``gps_time`` where the format
        has it, ``red green blue`` folded into ``colors`` as floats in 0..1,
        ``nir``, and the seven wave packet fields in formats 4, 5, 9 and
        10. Extra bytes described by their VLR are attributes of their own
        name and dtype, scaled and offset when their descriptor says so;
        undescribed ones are one ``extra_bytes`` block of ``uint8``.
        ``global_attrs`` keeps ``las_version``, ``las_point_format``,
        ``las_scale``, ``las_offset``, the system identifier and generating
        software when set (text fields are read as UTF-8, a cut codepoint
        replaced), the file source id, global encoding and creation date
        when nonzero, an OGC WKT VLR as ``crs_wkt``, GeoTIFF keys as
        ``las_geokeys`` and every other VLR or EVLR raw in ``las_vlrs``.
        An extra bytes descriptor with a blank name is read as
        ``extra_<i>``, ``i`` its position among the described fields.

    Raises
    ------
    CodecError
        On a header that is missing, of an unknown version or point
        format, shorter than its version needs, whose VLRs run past the
        point data, whose records are shorter than their format or whose
        point count does not fit the file; on an extra bytes VLR longer
        than the record's extra bytes or naming a data type the spec does
        not; on a LAZ stream that ends early or a compressed file with no
        LASzip VLR.
    LazyReadError
        If ``lazy`` is set on a compressed file or a source that cannot be
        mapped.
    MissingPackageError
        On a compressed file when lazrs is not installed.
    """
    name = source_name(path)
    if lazy:
        with open_block(path, fmt=EXTENSION, require_map=True) as data:
            # The mapping is kept open for the arrays that will view it, so
            # a refusal before any exists has to close it here.
            try:
                header = _parse_header(data, name=name)
                if header.compressed:
                    raise LazyReadError(
                        f"'{name}': the point records are LAZ-compressed, and a "
                        f"mapping would hand back the compressed bytes. Read it "
                        f"eagerly (lazy=False)."
                    )
                layout = _extra_layout(header, name=name)
            except BaseException:
                data.close()
                raise
            return _read_records(data, header, layout, name=name, lazy=True)

    with open_block(path, fmt=EXTENSION) as data:
        header = _parse_header(data, name=name)
        layout = _extra_layout(header, name=name)
        if header.compressed:
            block = _decompress(data, header, name=name)
            return _read_records(
                block, header, layout, name=name, lazy=False, offset=0, private=True
            )
        return _read_records(data, header, layout, name=name, lazy=False)


def _text(raw: bytes) -> str:
    """A NUL-padded text field; UTF-8 read as such, a cut codepoint replaced."""
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").rstrip()


def _parse_header(data: Any, *, name: str) -> _Header:
    """Read the public header block and the VLRs, checking each against the file."""
    size = len(data)
    if size < 4 or bytes(data[:4]) != _SIGNATURE:
        raise CodecError(f"'{name}': not a LAS file; the signature is not 'LASF'.")
    base = struct.calcsize(_HEADER_1_2)
    if size < base:
        raise CodecError(
            f"'{name}': the file is {size} bytes, shorter than the {base}-byte "
            f"LAS header."
        )
    fields = struct.unpack_from(_HEADER_1_2, data, 0)
    (
        _,
        source_id,
        encoding,
        _guid1,
        _guid2,
        _guid3,
        _guid4,
        major,
        minor,
        system_id,
        software,
        day,
        year,
        header_size,
        offset,
        n_vlrs,
        raw_format,
        record_length,
        legacy_count,
        *rest,
    ) = fields
    scale = np.array(rest[5:8], dtype=np.float64)
    origin = np.array(rest[8:11], dtype=np.float64)
    version = (major, minor)
    if version not in _VERSIONS_READ:
        raise CodecError(
            f"'{name}': LAS version {major}.{minor} is not supported; 1.0 to 1.4 are."
        )
    point_format = raw_format & ~_COMPRESSED_BIT
    compressed = bool(raw_format & _COMPRESSED_BIT)
    if point_format not in _FORMAT_FIELDS:
        raise CodecError(
            f"'{name}': point data record format {point_format} is not "
            f"supported; 0 to 10 are."
        )
    needed = _HEADER_SIZES[version]
    if header_size < needed:
        raise CodecError(
            f"'{name}': header size {header_size} is shorter than the {needed} "
            f"bytes a LAS {major}.{minor} header holds."
        )
    if header_size > size:
        raise CodecError(
            f"'{name}': header size {header_size} runs past the {size}-byte file."
        )
    standard = _RECORD_LENGTHS[point_format]
    if record_length < standard:
        raise CodecError(
            f"'{name}': point data record length {record_length} is shorter than "
            f"the {standard} bytes of point format {point_format}."
        )

    n_points = legacy_count
    evlr_start = 0
    n_evlrs = 0
    if version >= (1, 4):
        evlr_start, n_evlrs, count_1_4 = struct.unpack_from("<QIQ", data, 235)
        if count_1_4:
            n_points = count_1_4

    if offset > size:
        raise CodecError(
            f"'{name}': offset to point data {offset} runs past the {size}-byte file."
        )
    vlrs, end = _read_vlrs(data, header_size, n_vlrs, name=name, limit=offset)
    if offset < end:
        raise CodecError(
            f"'{name}': offset to point data {offset} lies inside the header and "
            f"its VLRs, which end at byte {end}."
        )
    if n_evlrs:
        if evlr_start < offset or evlr_start > size:
            raise CodecError(
                f"'{name}': the first EVLR is placed at byte {evlr_start}, outside "
                f"the {size}-byte file's point data."
            )
        vlrs += _read_vlrs(
            data, evlr_start, n_evlrs, name=name, limit=size, extended=True
        )[0]

    validate_header(n_points, 0, 0, size, compressed=compressed)
    if not compressed and offset + n_points * record_length > size:
        raise CodecError(
            f"'{name}': the header declares {n_points} points of {record_length} "
            f"bytes from byte {offset}, past the {size}-byte file."
        )

    globals_: dict[str, Any] = {
        _KEY_VERSION: f"{major}.{minor}",
        _KEY_FORMAT: point_format,
        _KEY_SCALE: scale,
        _KEY_OFFSET: origin,
    }
    if _text(system_id):
        globals_[_KEY_SYSTEM] = _text(system_id)
    if _text(software):
        globals_[_KEY_SOFTWARE] = _text(software)
    if source_id:
        globals_[_KEY_SOURCE_ID] = source_id
    if encoding and version >= _DEFAULT_VERSION:
        globals_[_KEY_ENCODING] = encoding
    if year or day:
        globals_[_KEY_DATE] = (year, day)
    return _Header(
        version,
        point_format,
        compressed,
        record_length,
        n_points,
        offset,
        scale,
        origin,
        vlrs,
        globals_,
    )


def _read_vlrs(
    data: Any,
    start: int,
    count: int,
    *,
    name: str,
    limit: int,
    extended: bool = False,
) -> tuple[list[_Vlr], int]:
    """Read ``count`` (E)VLRs from ``start``, none allowed past ``limit``."""
    fmt = _EVLR_HEADER if extended else _VLR_HEADER
    head = _EVLR_HEADER_SIZE if extended else _VLR_HEADER_SIZE
    kind = "EVLR" if extended else "VLR"
    out: list[_Vlr] = []
    at = start
    for i in range(count):
        if at + head > limit:
            raise CodecError(
                f"'{name}': {kind} {i} at byte {at} runs past byte {limit}, where "
                f"the {'file' if extended else 'point data'} "
                f"{'ends' if extended else 'begins'}."
            )
        _, user_id, record_id, length, description = struct.unpack_from(fmt, data, at)
        at += head
        if at + length > limit:
            raise CodecError(
                f"'{name}': {kind} {i} declares {length} bytes of data from byte "
                f"{at}, past byte {limit}."
            )
        out.append(
            _Vlr(
                _text(user_id),
                record_id,
                _text(description),
                bytes(data[at : at + length]),
            )
        )
        at += length
    return out, at


def _decompress(data: Any, header: _Header, *, name: str) -> bytearray:
    """Expand the LAZ point records into one plain block of records."""
    laszip = [
        v
        for v in header.vlrs
        if v.user_id == _LASZIP_USER and v.record_id == _LASZIP_ID
    ]
    if not laszip:
        raise CodecError(
            f"'{name}': the point format says the records are LAZ-compressed, but "
            f"there is no laszip VLR describing the compression."
        )
    if header.n_points == 0:
        return bytearray()
    lazrs, have = _lazrs()
    if not have:
        lazrs.LasZipDecompressor  # noqa: B018 - raises MissingPackageError
    # The stream's chunk table offset is absolute, so the decompressor gets
    # the whole file and is stood at the point data. A mapping is read in
    # place; wrapping it in BytesIO would copy the file.
    stream = _MappedStream(data) if isinstance(data, mmap.mmap) else io.BytesIO(data)
    stream.seek(header.offset)
    try:
        vlr = lazrs.LazVlr(laszip[0].data)
        chunks = lazrs.read_chunk_table(stream, vlr)
    except lazrs.LazrsError as exc:
        raise CodecError(
            f"'{name}': the LAZ stream has no readable chunk table: {exc}."
        ) from exc
    room = _laz_room(chunks, vlr, len(data) - header.offset, name=name)
    if header.n_points > room:
        raise CodecError(
            f"'{name}': the header declares {header.n_points} points, but the LAZ "
            f"stream's chunk table holds at most {room}."
        )
    stream.seek(header.offset)
    decompressor = lazrs.LasZipDecompressor(stream, laszip[0].data)
    # One piece at a time, a chunk of the stream or less: a header that lies
    # about its count within the chunk table's bound then fails on the piece
    # after the real points, having cost one piece of memory, not n records.
    step = min(header.n_points, vlr.chunk_size(), _LAZ_PIECE_POINTS)
    piece = bytearray(step * header.record_length)
    out = bytearray()
    done = 0
    while done < header.n_points:
        count = min(step, header.n_points - done)
        target = (
            piece
            if count == step
            else memoryview(piece)[: count * header.record_length]
        )
        try:
            decompressor.decompress_many(target)
        except lazrs.LazrsError as exc:
            raise CodecError(
                f"'{name}': the LAZ stream does not hold the {header.n_points} "
                f"points the header declares; it runs dry past point {done}: {exc}."
            ) from exc
        out += target
        done += count
    return out


class _MappedStream:
    """A mapping as the seekable stream lazrs reads.

    Parameters
    ----------
    mapping
        The memory map of the file. Its own ``seek`` returns the new
        position only from Python 3.13, and lazrs needs that position back
        on every version.
    """

    def __init__(self, mapping: mmap.mmap) -> None:
        self._mapping = mapping

    def read(self, size: int = -1) -> bytes:
        return self._mapping.read(None if size < 0 else size)

    def seek(self, offset: int, whence: int = 0) -> int:
        self._mapping.seek(offset, whence)
        return self._mapping.tell()

    def tell(self) -> int:
        return self._mapping.tell()


def _laz_room(
    chunks: list[tuple[int, int]], vlr: Any, available: int, *, name: str
) -> int:
    """The most points a LAZ stream can decode to, from its chunk table.

    Parameters
    ----------
    chunks
        The chunk table: a point count and a byte count per chunk. The
        point count is exact for variable-size chunks and the nominal chunk
        size otherwise, where only the last chunk may hold fewer.
    vlr
        The laszip VLR, for the chunk size and whether chunks vary.
    available
        Bytes from the point data offset to the end of the file, which the
        chunks and their table share.
    name
        The source, for the error.

    Returns
    -------
    int
        A count no honest header exceeds: each chunk's points, capped by
        what its compressed bytes could possibly unpack to.

    Raises
    ------
    CodecError
        When the chunks claim more compressed bytes than the file holds.
    """
    claimed = sum(nbytes for _, nbytes in chunks)
    if claimed > available:
        raise CodecError(
            f"'{name}': the LAZ chunk table declares {claimed} bytes of compressed "
            f"points, but only {available} follow the point data offset."
        )
    variable = vlr.uses_variable_size_chunks()
    return sum(
        min(count if variable else vlr.chunk_size(), nbytes * _LAZ_MAX_POINTS_PER_BYTE)
        for count, nbytes in chunks
    )


def _read_records(
    data: Any,
    header: _Header,
    layout: tuple[list[_Extra], int],
    *,
    name: str,
    lazy: bool,
    offset: int | None = None,
    private: bool = False,
) -> PolyData:
    """Decode the point records.

    Parameters
    ----------
    data
        The mapped or read file, or the block the LAZ records expanded to.
    header
        The parsed header.
    layout
        The described extra bytes fields and the undescribed bytes left.
    name
        The source, for warnings.
    lazy
        Hand back read-only views of the mapping in the file's dtypes.
    offset
        Where the records start in ``data``, the header's offset when None.
    private
        ``data`` is a block nobody else holds, so a column is handed out as
        a view of it rather than copied.
    """
    at = header.offset if offset is None else offset
    n = header.n_points
    pf = header.point_format
    standard = _RECORD_DTYPES[pf]
    extras, remainder = layout
    fields: list[tuple[str, Any]] = list(_FORMAT_FIELDS[pf])
    for i, extra in enumerate(extras):
        shape = (extra.width,) if extra.width > 1 else ()
        fields.append(
            (f"e{i}", extra.dtype, shape) if shape else (f"e{i}", extra.dtype)
        )
    if remainder:
        fields.append((_KEY_EXTRA_BYTES, "u1", (remainder,)))
    record = np.dtype(fields)

    if n == 0:
        raw = np.zeros(0, dtype=record)
    else:
        raw = np.frombuffer(data, dtype=record, count=n, offset=at)

    def take(field: str) -> np.ndarray:
        column = raw[field]
        return column if lazy or private else np.array(column)

    if n == 0 and lazy:
        vertices = np.empty((0, 3), dtype="<i4")
    elif lazy:
        vertices = np.ndarray(
            (n, 3),
            dtype="<i4",
            buffer=data,
            offset=at,
            strides=(record.itemsize, 4),
        )
    else:
        ints = np.column_stack([raw["X"], raw["Y"], raw["Z"]]).astype(np.float64)
        vertices = ints * header.scale + header.origin

    attrs: dict[str, np.ndarray] = {}
    for field_name, _ in standard.descr:
        if field_name in _COORDS or field_name in _PACKED or field_name in _RGB:
            continue
        attrs[field_name] = take(field_name)
    bits = _NEW_BITS if pf in _NEW_STYLE else _LEGACY_BITS
    for attr, packed, shift, width in bits:
        attrs[attr] = ((raw[packed] >> shift) & ((1 << width) - 1)).astype(np.uint8)
    if "red" in standard.names:
        if n == 0 and lazy:
            attrs[_COLORS] = np.empty((0, 3), dtype="<u2")
        elif lazy:
            attrs[_COLORS] = np.ndarray(
                (n, 3),
                dtype="<u2",
                buffer=data,
                offset=at + record.fields["red"][1],
                strides=(record.itemsize, 2),
            )
        else:
            attrs[_COLORS] = (
                np.column_stack([raw[c] for c in _RGB]).astype(np.float64)
                / _COLOR_SCALE
            )
    for i, extra in enumerate(extras):
        column = take(f"e{i}")
        if extra.scale is not None or extra.offset is not None:
            if lazy:
                warnings.warn(
                    f"'{name}': extra bytes field {extra.name!r} carries a scale or "
                    f"offset, which a lazy read leaves unapplied.",
                    stacklevel=5,
                )
            else:
                scale = 1.0 if extra.scale is None else extra.scale[: extra.width]
                shift = 0.0 if extra.offset is None else extra.offset[: extra.width]
                column = column.astype(np.float64) * scale + shift
        attrs[_unique(extra.name, attrs, name=name)] = column
    if remainder:
        attrs[_unique(_KEY_EXTRA_BYTES, attrs, name=name)] = take(_KEY_EXTRA_BYTES)
    if not lazy:
        del raw

    globals_ = dict(header.globals)
    globals_.update(_crs_globals(header.vlrs))
    kept = [
        {
            "user_id": v.user_id,
            "record_id": v.record_id,
            "description": v.description,
            "data": v.data,
        }
        for v in header.vlrs
        if not _consumed(v)
    ]
    if kept:
        globals_[_KEY_VLRS] = kept
    return point_cloud(vertices=vertices, vertex_attrs=attrs, global_attrs=globals_)


def _unique(wanted: str, attrs: dict[str, np.ndarray], *, name: str) -> str:
    """The attribute name an extra byte gets: its own unless that is taken."""
    if wanted not in attrs:
        return wanted
    i = 1
    while f"{wanted}_{i}" in attrs:
        i += 1
    warnings.warn(
        f"'{name}': extra bytes field {wanted!r} is named like another field; "
        f"it is read as {wanted}_{i!r}.",
        stacklevel=5,
    )
    return f"{wanted}_{i}"


def _consumed(vlr: _Vlr) -> bool:
    """Whether a VLR is decoded into something else rather than kept raw."""
    if vlr.user_id == _LASZIP_USER and vlr.record_id == _LASZIP_ID:
        return True
    if vlr.user_id == _SPEC_USER and vlr.record_id == _EXTRA_BYTES_ID:
        return True
    return vlr.user_id == _PROJECTION_USER and vlr.record_id in (
        _WKT_ID,
        _GEOKEYS_ID,
        _GEODOUBLES_ID,
        _GEOASCII_ID,
    )


def _extra_layout(header: _Header, *, name: str) -> tuple[list[_Extra], int]:
    """The extra bytes of a record: the described fields, then the bytes left."""
    room = header.record_length - _RECORD_LENGTHS[header.point_format]
    described = [
        v
        for v in header.vlrs
        if v.user_id == _SPEC_USER and v.record_id == _EXTRA_BYTES_ID
    ]
    extras: list[_Extra] = []
    used = 0
    for vlr in described:
        for at in range(
            0, len(vlr.data) - _EXTRA_DESCRIPTOR_SIZE + 1, _EXTRA_DESCRIPTOR_SIZE
        ):
            (_, code, options, raw_name, _, _, _, _, *numbers, _) = struct.unpack_from(
                _EXTRA_DESCRIPTOR, vlr.data, at
            )
            if code not in _EXTRA_TYPES:
                raise CodecError(
                    f"'{name}': extra bytes field {_text(raw_name)!r} has data type "
                    f"{code}, which the specification does not define."
                )
            dtype, width = _EXTRA_TYPES[code]
            if code == 0:
                width = options
                if width == 0:
                    continue
            scale = np.array(numbers[:3]) if options & _EXTRA_SCALE_BIT else None
            shift = np.array(numbers[3:]) if options & _EXTRA_OFFSET_BIT else None
            if code == 0:
                scale = shift = None
            extras.append(
                _Extra(
                    _text(raw_name) or f"extra_{len(extras)}",
                    dtype,
                    width,
                    scale,
                    shift,
                )
            )
            used += np.dtype(dtype).itemsize * width
    if used > room:
        raise CodecError(
            f"'{name}': the extra bytes VLR describes {used} bytes per point, but "
            f"the {header.record_length}-byte records leave {room} past point "
            f"format {header.point_format}."
        )
    return extras, room - used


def _crs_globals(vlrs: list[_Vlr]) -> dict[str, Any]:
    """``crs_wkt`` and ``las_geokeys`` from the projection VLRs, when present."""
    out: dict[str, Any] = {}
    by_id = {v.record_id: v.data for v in vlrs if v.user_id == _PROJECTION_USER}
    if _WKT_ID in by_id:
        out[_KEY_WKT] = by_id[_WKT_ID].split(b"\0", 1)[0].decode("utf-8", "replace")
    if _GEOKEYS_ID in by_id:
        keys = _decode_geokeys(
            by_id[_GEOKEYS_ID],
            by_id.get(_GEODOUBLES_ID, b""),
            by_id.get(_GEOASCII_ID, b""),
        )
        if keys:
            out[_KEY_GEOKEYS] = keys
    return out


def _decode_geokeys(directory: bytes, doubles: bytes, ascii: bytes) -> dict[int, Any]:
    """Resolve a GeoKeyDirectoryTag into ``{key id: short | float | str}``."""
    if len(directory) < 8:
        return {}
    n_keys = struct.unpack_from(_GEOKEY_DIRECTORY, directory, 0)[3]
    n_doubles = len(doubles) // 8
    values = struct.unpack_from(f"<{n_doubles}d", doubles, 0)
    text = ascii.decode("ascii", "replace")
    out: dict[int, Any] = {}
    for i in range(1, n_keys + 1):
        at = i * 8
        if at + 8 > len(directory):
            break
        key, location, count, value = struct.unpack_from(
            _GEOKEY_DIRECTORY, directory, at
        )
        if location == 0:
            out[key] = value
        elif location == _GEODOUBLES_ID and value + count <= n_doubles:
            out[key] = (
                values[value] if count == 1 else list(values[value : value + count])
            )
        elif location == _GEOASCII_ID:
            out[key] = text[value : value + count].rstrip("|\0")
    return out


# ----- writing -------------------------------------------------------------------


def write(
    poly: PolyData,
    path: Source,
    *,
    point_format: int | None = None,
    version: str | None = None,
    scale: float | tuple[float, float, float] | None = None,
    offset: float | tuple[float, float, float] | None = None,
    compress: bool | None = None,
    **opts: Any,
) -> None:
    """Serialise a PolyData's vertices and their attributes as a LAS or LAZ file.

    Parameters
    ----------
    poly
        The mesh; its vertices are the points. Elements are not part of the
        format: any that are not single vertices are dropped with a warning.
        Vertices of an integer dtype whose ``global_attrs`` carry both
        ``las_scale`` and ``las_offset`` - what a lazy read hands back -
        are the stored ``X Y Z`` and mean ``int * las_scale + las_offset``:
        they go out untouched at that scale and offset, and are rescaled
        through those coordinates when ``scale`` or ``offset`` asks for
        another.
    path
        Destination path or binary file object.
    point_format
        The point data record format, 0 to 10. Left unset, it is
        ``las_point_format`` from ``global_attrs`` when there is one, and
        otherwise the smallest format that holds the attributes present:
        ``gps_time`` asks for 1, ``colors`` for 2, both for 3, a wave packet
        for 4 or 5, and ``nir``, ``scan_angle``, ``overlap`` or
        ``scanner_channel`` for 6 to 10. An attribute the chosen format has
        no field for is dropped with a warning.
    version
        ``"1.0"`` to ``"1.4"``. Left unset, it is ``las_version`` from
        ``global_attrs`` when there is one, raised to what the point format
        and the point count need, else the lowest of 1.2, 1.3 and 1.4 they
        allow; 1.0 and 1.1 are written only when asked for. Given, a
        format it predates is refused.
    scale
        The X, Y and Z scale factors, one number for all three or one per
        axis. Left unset, ``las_scale`` from ``global_attrs`` or 0.001.
    offset
        The X, Y and Z offsets. Left unset, ``las_offset`` from
        ``global_attrs``, else zero when every coordinate fits an int32 at
        the scale and the rounded lower corner of the bounding box when
        not.
    compress
        Whether to LAZ-compress the records with lazrs (the ``laz`` extra).
        Left unset, a destination named ``.laz`` is compressed and any
        other is not; a nameless buffer is not.

    Raises
    ------
    CodecError
        On a NaN or infinite coordinate, one the scale and offset cannot
        bring into an int32, a scale that is not positive, a vertex
        attribute that is not one row per vertex or not numeric, a value
        outside its field's range (an ``intensity`` past 65535, a
        ``return_number`` of 8 in formats 0-5, a ``classification`` above
        31 there, a ``scan_angle`` outside the int8 degrees of formats 0-5,
        a ``scan_angle_rank`` outside the int16 steps of formats 6-10, a
        NaN in an integer field), a header global outside its 16-bit
        field, ``las_geokeys`` that is not a mapping or ``las_vlrs`` that
        is not a list, or a point format and version that do not go
        together.
    MissingPackageError
        When compressing without lazrs installed.

    Notes
    -----
    Vertices of an integer dtype are stored as they are only when
    ``las_scale`` and ``las_offset`` are both in ``global_attrs``; with
    either missing they are coordinates like any other.
    ``vertex_attrs["colors"]`` goes out as ``red green blue``: floats in
    0..1 scaled to 16 bits, ``uint16`` as it is, and any other integer as
    0..255 spread over the 16. A float written into an integer field is
    rounded to the nearest integer. ``return_number`` and
    ``number_of_returns`` default to 1 when absent; every other field to
    0. A ``scan_angle_rank`` written into a format from 6 on, or a
    ``scan_angle`` into one before, is converted between whole degrees and
    0.006-degree steps. Every attribute is taken under its name less
    trailing whitespace, as a reader strips it: ``"intensity "`` fills the
    ``intensity`` field, and one whose stripped name another attribute
    already has is dropped with a warning. Every other numeric vertex
    attribute, one to three columns wide and with a name of 1 to 32
    printable ASCII bytes, goes out as an extra bytes field of its own name
    and dtype (a boolean as ``uint8``, a half-precision float as
    ``float32``), and an ``extra_bytes`` block of ``uint8`` columns
    goes back undescribed, the way it was read; one that does not fit, or
    is named like a raw dimension (``X``, ``Y``, ``Z``, ``red``, ``green``,
    ``blue``, ``returns``, ``flags``, ``raw_classification``), is dropped
    with a warning. ``crs_wkt`` writes an OGC WKT VLR and sets the WKT bit,
    which is cleared when there is no ``crs_wkt`` (or it is too long for
    the version) whatever ``las_global_encoding`` says; ``las_geokeys``
    writes the three GeoTIFF VLRs, the keys in ascending order as GeoTIFF
    asks, a key or short value outside 16 bits,
    or a string or list of doubles past the 65535 bytes or doubles their
    records index, dropped with a warning, and ``las_vlrs`` its entries as
    they are, an
    entry over 65535 bytes as an EVLR in 1.4 and dropped with a warning
    before, as is one the writer builds itself (extra bytes, LASzip, WKT
    or GeoTIFF). ``las_system_identifier``, ``las_generating_software``,
    ``las_file_source_id``, ``las_global_encoding`` and
    ``las_creation_date`` fill the header fields of those names; the two
    strings (bytes are decoded as UTF-8) are cut to their 32 ASCII bytes,
    at a NUL, or with a non-ASCII
    character replaced, each with a warning, the software
    defaults to ``"polyxios"``, the date is left blank unless given, and
    the encoding is written as zero in 1.0 and 1.1, where its field is
    reserved.
    """
    lazrs: ModuleType | TripWire | None = None
    if compress is None:
        compress = format_suffix(path) == ".laz"
    if compress:
        lazrs, have = _lazrs()
        if not have:
            lazrs.LasZipCompressor  # noqa: B018 - raises MissingPackageError
    g = poly.global_attrs
    n = poly.vertices.shape[0]
    non_vertex = int(np.count_nonzero(poly.element_types != _VERTEX_CODE))
    if non_vertex:
        warnings.warn(
            f".las holds points only; {non_vertex} elements dropped.", stacklevel=3
        )

    attrs = _stripped_names(poly.vertex_attrs)
    pf = _choose_format(point_format, attrs, g)
    ver = _choose_version(version, pf, n, g)
    scale_v = _axis_triple(
        _DEFAULT_SCALE if scale is None else scale,
        g,
        _KEY_SCALE,
        given=scale is not None,
    )
    if not np.all(scale_v > 0):
        raise CodecError(
            f".las: the scale factors must be positive, not {scale_v.tolist()}."
        )
    ints, offset_v = _stored_ints(poly.vertices, scale_v, offset, g)

    fields = list(_FORMAT_FIELDS[pf])
    standard = _RECORD_DTYPES[pf]
    extras, trailing = _extra_fields(attrs, n)
    record_fields = fields + [(f"e{i}", e.dtype) for i, e in enumerate(extras)]
    if trailing is not None:
        record_fields.append((_KEY_EXTRA_BYTES, "u1", (trailing.shape[1],)))
    record = np.dtype(record_fields)
    out = np.zeros(n, dtype=record)
    out["X"], out["Y"], out["Z"] = ints[:, 0], ints[:, 1], ints[:, 2]
    _fill_standard(out, attrs, pf, n)
    for i, extra in enumerate(extras):
        out[f"e{i}"] = extra.column
    if trailing is not None:
        out[_KEY_EXTRA_BYTES] = trailing

    vlrs = _vlrs_to_write(g, extras, ver)
    if compress:
        assert lazrs is not None
        laz_vlr = lazrs.LazVlr.new_for_compression(
            pf, record.itemsize - standard.itemsize
        )
        vlrs.insert(0, _Vlr(_LASZIP_USER, _LASZIP_ID, "lazrs", laz_vlr.record_data()))
    short = [v for v in vlrs if len(v.data) <= _VLR_MAX_BYTES]
    long = [v for v in vlrs if len(v.data) > _VLR_MAX_BYTES]
    if long and ver < (1, 4):
        warnings.warn(
            f".las {ver[0]}.{ver[1]} has no EVLRs, so {len(long)} VLR(s) over "
            f"65535 bytes are dropped.",
            stacklevel=3,
        )
        long = []

    header_size = _HEADER_SIZES[ver]
    vlr_bytes = b"".join(_pack_vlr(v, ver=ver) for v in short)
    data_offset = header_size + len(vlr_bytes)
    body: Any = out.view(np.uint8)
    if compress:
        # The chunk table offset lazrs writes is absolute, so the
        # compressor is stood at the point data over a blank header.
        stream = io.BytesIO()
        stream.write(bytes(data_offset))
        compressor = lazrs.LasZipCompressor(stream, laz_vlr)
        compressor.compress_many(body)
        compressor.done()
        body = stream.getbuffer()[data_offset:]
    evlr_start = data_offset + len(body) if long else 0
    evlr_bytes = b"".join(_pack_vlr(v, ver=ver, extended=True) for v in long)
    header = _pack_header(
        g,
        ver,
        pf,
        compress,
        record.itemsize,
        n,
        out,
        scale_v,
        offset_v,
        ints,
        n_vlrs=len(short),
        data_offset=data_offset,
        evlr_start=evlr_start,
        n_evlrs=len(long),
        has_wkt=any(
            v.user_id == _PROJECTION_USER and v.record_id == _WKT_ID
            for v in short + long
        ),
    )
    with open_write(path) as fh:
        fh.write(header)
        fh.write(vlr_bytes)
        fh.write(body)
        fh.write(evlr_bytes)


def _stripped_names(attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The vertex attributes under their names less trailing whitespace.

    A reader strips the NUL padding and trailing blanks of every name it
    finds, so ``"intensity "`` is the ``intensity`` field and ``"height "``
    the extra byte ``height``. One whose stripped name another attribute
    already holds is dropped with a warning; the plain name wins.
    """
    out: dict[str, np.ndarray] = {}
    for name, value in attrs.items():
        stripped = name.rstrip() if isinstance(name, str) else name
        if stripped in out or (stripped != name and stripped in attrs):
            warnings.warn(
                f".las: vertex attribute {name!r} is {stripped!r} without its "
                f"trailing whitespace, a name another attribute has; it is dropped.",
                stacklevel=4,
            )
            continue
        out[stripped] = value
    return out


def _stored_ints(
    vertices: np.ndarray, scale: np.ndarray, offset: Any, g: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """The ``X Y Z`` to store and the offset they are stored against.

    Parameters
    ----------
    vertices
        The points: coordinates as floats, or when of an integer dtype
        while ``las_scale`` and ``las_offset`` are both in ``g``, the
        stored integers a lazy read handed back, which mean
        ``int * las_scale + las_offset``.
    scale
        The three scale factors the file is written at.
    offset
        The ``offset`` argument, or None to take ``las_offset`` from ``g``
        or choose one.
    g
        The global attributes.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The ``(n, 3)`` little-endian int32 records hold, and the ``(3,)``
        offset. Stored integers go out untouched when the scale and offset
        are the file's own, and are rescaled through their coordinates
        otherwise.

    Raises
    ------
    CodecError
        On a NaN or infinite coordinate, or one the scale and offset
        cannot bring into an int32.
    """
    verts = np.asarray(vertices)
    stored = verts.dtype.kind in "iu" and _KEY_SCALE in g and _KEY_OFFSET in g
    if stored:
        file_scale = _axis_triple(None, g, _KEY_SCALE, given=False)
        file_offset = _axis_triple(None, g, _KEY_OFFSET, given=False)
        offset_v = _axis_triple(offset, g, _KEY_OFFSET, given=offset is not None)
        if np.array_equal(scale, file_scale) and np.array_equal(offset_v, file_offset):
            if verts.size and (verts.min() < _INT32_MIN or verts.max() > _INT32_MAX):
                raise CodecError(
                    f".las: a stored coordinate of dtype {verts.dtype} does not fit "
                    f"the int32 the file holds."
                )
            return np.ascontiguousarray(verts, dtype="<i4"), offset_v
        coords = verts.astype(np.float64) * file_scale + file_offset
    else:
        coords = np.asarray(verts, dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        raise CodecError(
            ".las: a NaN or infinite coordinate has no int32 to be stored in."
        )
    offset_v = _pick_offset(offset, coords, scale, g)
    ints = np.rint((coords - offset_v) / scale)
    if ints.size and (ints.min() < _INT32_MIN or ints.max() > _INT32_MAX):
        raise CodecError(
            f".las: at scale {scale.tolist()} and offset {offset_v.tolist()} a "
            f"coordinate does not fit an int32; pass a coarser scale or an offset "
            f"nearer the points."
        )
    return ints.astype("<i4"), offset_v


def _choose_format(
    point_format: int | None, attrs: dict[str, np.ndarray], g: dict[str, Any]
) -> int:
    if point_format is None:
        point_format = g.get(_KEY_FORMAT)
    if point_format is None:
        present = set(attrs)
        new = bool(present & _NEW_ONLY)
        wave = bool(present & _WAVE_NAMES)
        rgb = _COLORS in present or "nir" in present
        if new:
            if wave:
                return 10 if rgb else 9
            if "nir" in present:
                return 8
            return 7 if rgb else 6
        if wave:
            return 5 if rgb else 4
        return (1 if "gps_time" in present else 0) + (2 if rgb else 0)
    try:
        point_format = int(point_format)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"point_format must be 0 to 10, not {point_format!r}."
        ) from exc
    if point_format not in _FORMAT_FIELDS:
        raise ValueError(f"point_format must be 0 to 10, not {point_format!r}.")
    return point_format


def _choose_version(
    version: str | None, pf: int, n: int, g: dict[str, Any]
) -> tuple[int, int]:
    needed = max(_MIN_VERSION[pf], (1, 4) if n > _U32_MAX else (1, 0))
    inherited = version is None
    if inherited:
        version = g.get(_KEY_VERSION)
    if version is None:
        return max(needed, _DEFAULT_VERSION)
    if str(version) not in _VERSIONS_WRITE:
        raise ValueError(
            f"version must be one of {sorted(_VERSIONS_WRITE)}, not {version!r}."
        )
    ver = _VERSIONS_WRITE[str(version)]
    if inherited:
        return max(ver, needed)
    if ver < needed:
        raise CodecError(
            f".las: point format {pf}"
            f"{' with more than 2**32-1 points' if n > _U32_MAX else ''} needs LAS "
            f"{needed[0]}.{needed[1]} or later, not {version}."
        )
    return ver


def _axis_triple(value: Any, g: dict[str, Any], key: str, *, given: bool) -> np.ndarray:
    """Three floats from an argument, the global attribute, or the default."""
    if not given and key in g:
        value = g[key]
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise ValueError(
            f"{key.removeprefix('las_')} must be one number or three, not {value!r}."
        )
    return arr


def _pick_offset(
    offset: Any, coords: np.ndarray, scale: np.ndarray, g: dict[str, Any]
) -> np.ndarray:
    if offset is not None or _KEY_OFFSET in g:
        return _axis_triple(offset, g, _KEY_OFFSET, given=offset is not None)
    if coords.size == 0:
        return np.zeros(3)
    if np.abs(coords / scale).max() <= _INT32_MAX:
        return np.zeros(3)
    return np.floor(coords.min(axis=0))


def _column(
    attrs: dict[str, np.ndarray], attr: str, n: int, *, flat: bool = True
) -> np.ndarray:
    """The attribute as ``n`` rows, one value each when ``flat``, or a CodecError."""
    arr = np.asarray(attrs[attr])
    if arr.ndim == 0 or arr.shape[0] != n or (flat and arr.size != n):
        raise CodecError(
            f".las: vertex attribute {attr!r} has shape {arr.shape}, not one "
            f"{'value' if flat else 'row'} per vertex ({n})."
        )
    return arr.reshape(n) if flat else arr


def _numeric(value: np.ndarray, *, attr: str, field: str) -> np.ndarray:
    """The column as a numeric array, booleans as bytes, or a CodecError."""
    if value.dtype.kind == "b":
        return value.astype(np.uint8)
    if value.dtype.kind not in "iuf":
        raise CodecError(
            f".las: vertex attribute {attr!r} has dtype {value.dtype}, which the "
            f"field {field!r} cannot store."
        )
    return value


def _finite(value: np.ndarray, *, attr: str, field: str) -> np.ndarray:
    """The column as numbers an integer field can take: a NaN or infinity refused."""
    value = _numeric(value, attr=attr, field=field)
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise CodecError(
            f".las: vertex attribute {attr!r} holds a NaN or infinity, which "
            f"the integer field {field!r} cannot store."
        )
    return value


def _integral(value: np.ndarray, *, attr: str, field: str) -> np.ndarray:
    """The column as whole numbers: a float rounded, a NaN or infinity refused."""
    value = _finite(value, attr=attr, field=field)
    return np.rint(value) if value.dtype.kind == "f" else value


def _into_field(out: np.ndarray, field: str, value: np.ndarray, *, attr: str) -> None:
    """Store a column in a record field, refusing what the field cannot hold."""
    target = out.dtype[field]
    if target.kind in "iu":
        value = _integral(value, attr=attr, field=field)
        info = np.iinfo(target)
        if value.size and (value.min() < info.min or value.max() > info.max):
            bad = value.max() if value.max() > info.max else value.min()
            raise CodecError(
                f".las: vertex attribute {attr!r} holds {bad:g}, outside the "
                f"{info.min}..{info.max} of the {target} field {field!r}."
            )
    else:
        value = _numeric(value, attr=attr, field=field)
    out[field] = value


def _fill_standard(
    out: np.ndarray, attrs: dict[str, np.ndarray], pf: int, n: int
) -> None:
    """Write the standard fields, bit fields and colours of one format."""
    names = _RECORD_DTYPES[pf].names or ()
    new = pf in _NEW_STYLE
    for field in names:
        if field in _COORDS or field in _PACKED or field in _RGB:
            continue
        if field == "scan_angle" and field not in attrs and "scan_angle_rank" in attrs:
            out[field] = _angle_to_steps(_column(attrs, "scan_angle_rank", n))
        elif (
            field == "scan_angle_rank" and field not in attrs and "scan_angle" in attrs
        ):
            out[field] = _steps_to_angle(_column(attrs, "scan_angle", n))
        elif field in attrs:
            _into_field(out, field, _column(attrs, field, n), attr=field)
    for attr, packed, shift, width in _NEW_BITS if new else _LEGACY_BITS:
        if attr in attrs:
            value = _integral(_column(attrs, attr, n), attr=attr, field=packed)
        elif attr in _DEFAULT_ONE:
            value = np.ones(n, dtype=np.uint8)
        else:
            continue
        if value.size and (value.min() < 0 or value.max() >= 1 << width):
            raise CodecError(
                f".las: {attr} holds {int(value.max() if value.max() >= 1 << width else value.min())}, "
                f"outside the {width}-bit field point format {pf} gives it."
            )
        out[packed] |= value.astype(np.uint8) << shift
    if "red" in names and _COLORS in attrs:
        rgb = _color_16(_column(attrs, _COLORS, n, flat=False))
        for i, channel in enumerate(_RGB):
            out[channel] = rgb[:, i]
    for attr in _STANDARD & set(attrs):
        if attr not in names and not _covered(attr, names, attrs):
            warnings.warn(
                f".las: point format {pf} has no field for vertex attribute "
                f"{attr!r}; it is dropped.",
                stacklevel=4,
            )


def _covered(attr: str, names: tuple[str, ...], attrs: dict[str, np.ndarray]) -> bool:
    """Whether an attribute lands in a format's field under another name."""
    if attr == _COLORS:
        return "red" in names
    if attr in ("scan_angle", "scan_angle_rank"):
        other = "scan_angle_rank" if attr == "scan_angle" else "scan_angle"
        return other in names and other not in attrs
    bits = (
        {a for a, *_ in _LEGACY_BITS}
        if "raw_classification" in names
        else {a for a, *_ in _NEW_BITS}
    )
    return attr in bits


def _angle_to_steps(rank: np.ndarray) -> np.ndarray:
    """``scan_angle_rank`` degrees as the int16 steps of ``scan_angle``, or a CodecError."""
    rank = _finite(rank, attr="scan_angle_rank", field="scan_angle")
    steps = np.rint(rank.astype(np.float64) / _ANGLE_UNIT)
    info = np.iinfo("<i2")
    if steps.size and (steps.min() < info.min or steps.max() > info.max):
        raise CodecError(
            f".las: scan_angle_rank reaches {rank[np.abs(steps).argmax()]:g} degrees, "
            f"outside the {info.min * _ANGLE_UNIT:.3f}..{info.max * _ANGLE_UNIT:.3f} "
            f"the int16 scan_angle of point formats 6 to 10 spans."
        )
    return steps.astype("<i2")


def _steps_to_angle(steps: np.ndarray) -> np.ndarray:
    """``scan_angle`` steps as the int8 degrees of ``scan_angle_rank``, or a CodecError."""
    steps = _finite(steps, attr="scan_angle", field="scan_angle_rank")
    degrees = np.rint(steps.astype(np.float64) * _ANGLE_UNIT)
    info = np.iinfo("i1")
    if degrees.size and (degrees.min() < info.min or degrees.max() > info.max):
        raise CodecError(
            f".las: scan_angle reaches {degrees[np.abs(degrees).argmax()]:.0f} degrees, "
            f"outside the int8 scan_angle_rank of point formats 0 to 5."
        )
    return degrees.astype("i1")


def _color_16(colors: np.ndarray) -> np.ndarray:
    """Three ``uint16`` channels from floats in 0..1 or integers."""
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise CodecError(f".las: colors must be (n, 3) or (n, 4), not {colors.shape}.")
    rgb = colors[:, :3]
    if rgb.dtype.kind == "u" and rgb.dtype.itemsize == 2:
        return rgb.astype("<u2")
    if rgb.dtype.kind in "iub":
        return (np.clip(rgb.astype(np.int64), 0, 255) * _BYTE_TO_16).astype("<u2")
    scaled = np.rint(np.nan_to_num(rgb.astype(np.float64)) * _COLOR_SCALE)
    return np.clip(scaled, 0, _COLOR_SCALE).astype("<u2")


def _extra_fields(
    attrs: dict[str, np.ndarray], n: int
) -> tuple[list[_ExtraOut], np.ndarray | None]:
    """Every attribute no standard field takes, as extra bytes.

    Returns
    -------
    list of _ExtraOut, numpy.ndarray or None
        The described fields in order, and the ``(n, k)`` uint8 block an
        ``extra_bytes`` attribute puts back after them, undescribed, the
        way it was read.
    """
    out: list[_ExtraOut] = []
    trailing: np.ndarray | None = None
    for name, value in attrs.items():
        if name in _STANDARD:
            continue
        if name in _RESERVED:
            warnings.warn(
                f".las: vertex attribute {name!r} is named like a standard LAS "
                f"dimension, which other readers would take it for; it is dropped.",
                stacklevel=4,
            )
            continue
        arr = np.asarray(value)
        if arr.dtype.kind == "b":
            arr = arr.astype(np.uint8)
        block = arr.ndim == 2 and arr.dtype == np.uint8
        if block and name == _KEY_EXTRA_BYTES and 0 < arr.shape[1] <= _U8_MAX:
            if arr.shape[0] != n:
                raise CodecError(
                    f".las: vertex attribute {name!r} has shape {arr.shape}, not one "
                    f"row per vertex ({n})."
                )
            trailing = arr
            continue
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        width = 1 if arr.ndim == 1 else arr.shape[1] if arr.ndim == 2 else 0
        raw_bytes = block and width > 3
        why = None
        if arr.dtype.kind not in "iuf":
            why = "is not numeric"
        elif arr.ndim > 2 or (width > 3 and not raw_bytes) or width == 0:
            why = f"has shape {arr.shape}, not one to three columns"
        elif width > _U8_MAX:
            why = f"has {width} columns, over the {_U8_MAX} undescribed bytes allow"
        elif not _ascii_within(name, _EXTRA_NAME_BYTES) or not name:
            why = (
                f"has a name that is not 1 to {_EXTRA_NAME_BYTES} printable ASCII bytes"
            )
        elif arr.dtype.itemsize > 8:
            why = f"has dtype {arr.dtype}, which extra bytes cannot spell"
        if why is not None:
            warnings.warn(
                f".las: vertex attribute {name!r} {why}; it is dropped.", stacklevel=4
            )
            continue
        if arr.shape[0] != n:
            raise CodecError(
                f".las: vertex attribute {name!r} has shape {arr.shape}, not one row "
                f"per vertex ({n})."
            )
        if raw_bytes:
            out.append(_ExtraOut(name, ("u1", (width,)), arr, 0, width))
            continue
        if arr.dtype.kind == "f" and arr.dtype.itemsize < 4:
            arr = arr.astype("<f4")
        base = arr.dtype.newbyteorder("<")
        key = base.str if base.itemsize > 1 else base.str[1:]
        arr = np.ascontiguousarray(arr, dtype=key)
        dtype = (key, (width,)) if width > 1 else key
        out.append(_ExtraOut(name, dtype, arr, _EXTRA_CODES[(key, width)], 0))
    return out, trailing


def _vlrs_to_write(
    g: dict[str, Any], extras: list[_ExtraOut], ver: tuple[int, int]
) -> list[_Vlr]:
    vlrs: list[_Vlr] = []
    if extras:
        vlrs.append(
            _Vlr(_SPEC_USER, _EXTRA_BYTES_ID, "extra bytes", _pack_extras(extras))
        )
    if _KEY_WKT in g:
        wkt = _as_text(g[_KEY_WKT])
        if "\0" in wkt:
            warnings.warn(
                f".las: {_KEY_WKT} holds a NUL, which ends its VLR; it is cut there.",
                stacklevel=4,
            )
            wkt = wkt.split("\0", 1)[0]
        vlrs.append(_Vlr(_PROJECTION_USER, _WKT_ID, "OGC WKT", wkt.encode() + b"\0"))
    if _KEY_GEOKEYS in g:
        vlrs += _pack_geokeys(g[_KEY_GEOKEYS])
    entries = g.get(_KEY_VLRS, ())
    if isinstance(entries, (str, bytes, dict)) or not hasattr(entries, "__iter__"):
        raise CodecError(
            f".las: {_KEY_VLRS} must be a list of dicts, one per VLR, not {entries!r}."
        )
    for entry in entries:
        vlr = _vlr_from_entry(entry)
        if vlr is None:
            continue
        if _consumed(vlr):
            warnings.warn(
                f".las: las_vlrs entry {vlr.user_id!r}/{vlr.record_id} is one the "
                f"writer builds from the vertex attributes, crs_wkt or "
                f"las_geokeys; it is dropped.",
                stacklevel=4,
            )
            continue
        vlrs.append(vlr)
    return vlrs


def _vlr_from_entry(entry: Any) -> _Vlr | None:
    """A ``las_vlrs`` entry as a VLR, or None with a warning when malformed."""
    why = None
    if not isinstance(entry, dict):
        why = "is not a dict"
    else:
        user_id = entry.get("user_id", "")
        record_id = entry.get("record_id")
        description = entry.get("description", "")
        data = entry.get("data", b"")
        if not _ascii_within(user_id, 16):
            why = (
                "has a user_id that is not a string of at most 16 printable ASCII bytes"
            )
        elif (
            not isinstance(record_id, (int, np.integer)) or not 0 <= record_id <= 0xFFFF
        ):
            why = "has a record_id that is not a 16-bit integer"
        elif not _ascii_within(description, 32):
            why = (
                "has a description that is not a string of at most 32 printable "
                "ASCII bytes"
            )
        elif not isinstance(data, (bytes, bytearray, memoryview)):
            why = "has data that is not bytes"
    if why is not None:
        warnings.warn(
            f".las: las_vlrs entry {entry!r} {why}; it is dropped.", stacklevel=5
        )
        return None
    return _Vlr(user_id, int(record_id), description, bytes(data))


def _as_text(value: Any) -> str:
    """A global given as text: a string as it is, bytes decoded, anything else spelt."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _ascii_within(value: Any, limit: int) -> bool:
    """Whether a string fits a NUL-padded ASCII field of ``limit`` bytes."""
    return (
        isinstance(value, str)
        and value.isascii()
        and value.isprintable()
        and len(value) <= limit
    )


def _pack_vlr(vlr: _Vlr, *, ver: tuple[int, int], extended: bool = False) -> bytes:
    fmt = _EVLR_HEADER if extended else _VLR_HEADER
    return (
        struct.pack(
            fmt,
            _VLR_RESERVED_1_0 if ver == (1, 0) else 0,
            vlr.user_id.encode("ascii", "replace"),
            vlr.record_id,
            len(vlr.data),
            vlr.description.encode("ascii", "replace"),
        )
        + vlr.data
    )


def _pack_extras(extras: list[_ExtraOut]) -> bytes:
    out = bytearray()
    for extra in extras:
        out += struct.pack(
            _EXTRA_DESCRIPTOR,
            b"\0\0",
            extra.code,
            extra.options,
            extra.name.encode("ascii"),
            b"\0" * 4,
            b"\0" * 24,
            b"\0" * 24,
            b"\0" * 24,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            b"",
        )
    return bytes(out)


def _is_u16(value: Any) -> bool:
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return False
    return as_int == value and 0 <= as_int <= _U16_MAX


def _header_u16(g: dict[str, Any], key: str) -> int:
    """A header global as the 16-bit integer its field holds, or a CodecError."""
    value = g.get(key, 0)
    if not _is_u16(value):
        raise CodecError(
            f".las: {key} must be an integer in 0..{_U16_MAX}, not {value!r}."
        )
    return int(value)


def _header_text(g: dict[str, Any], key: str, default: str) -> bytes:
    """A header string global as its 32-byte ASCII field, cut with a warning."""
    text = _as_text(g.get(key, default))
    kept = text.split("\0", 1)[0]
    raw = kept.encode("ascii", "replace")
    why = None
    if len(raw) > _HEADER_TEXT_BYTES:
        why = f"is {len(raw)} bytes; its header field holds {_HEADER_TEXT_BYTES}"
    elif kept != text:
        why = "holds a NUL, which ends its header field"
    elif not kept.isascii():
        why = "is not ASCII, which its header field is"
    if why is not None:
        warnings.warn(f".las: {key} {why}, so it is cut.", stacklevel=4)
    return raw[:_HEADER_TEXT_BYTES]


def _pack_geokeys(keys: Any) -> list[_Vlr]:
    """The GeoKeyDirectoryTag VLR and, as needed, its doubles and ASCII twins."""
    if not isinstance(keys, Mapping):
        raise CodecError(
            f".las: {_KEY_GEOKEYS} must be a mapping of GeoTIFF key id to value, "
            f"not {keys!r}."
        )
    entries: list[tuple[int, int, int, int]] = []
    doubles: list[float] = []
    text = ""
    for key, value in keys.items():
        if not _is_u16(key):
            warnings.warn(
                f".las: las_geokeys key {key!r} is not a 16-bit integer; it is "
                f"dropped.",
                stacklevel=5,
            )
            continue
        key = int(key)
        if len(entries) == _U16_MAX:
            warnings.warn(
                f".las: las_geokeys holds more than {_U16_MAX} keys, which its "
                f"directory cannot count; key {key} and the rest are dropped.",
                stacklevel=5,
            )
            break
        if isinstance(value, str):
            if len(text) + len(value) + 1 > _U16_MAX:
                warnings.warn(
                    f".las: las_geokeys key {key} holds a string that does not fit "
                    f"the {_U16_MAX} bytes the GeoTIFF ASCII record can hold; it is "
                    f"dropped.",
                    stacklevel=5,
                )
                continue
            entries.append((key, _GEOASCII_ID, len(value) + 1, len(text)))
            text += value + "|"
        elif isinstance(value, (list, tuple, float, np.floating)):
            try:
                numbers = (
                    [float(value)]
                    if isinstance(value, (float, np.floating))
                    else [float(v) for v in value]
                )
            except (TypeError, ValueError):
                warnings.warn(
                    f".las: las_geokeys key {key} holds {value!r}, which is not a "
                    f"list of floats; it is dropped.",
                    stacklevel=5,
                )
                continue
            if len(doubles) + len(numbers) > _U16_MAX:
                warnings.warn(
                    f".las: las_geokeys key {key} holds {len(numbers)} doubles, past "
                    f"the {_U16_MAX} the GeoTIFF double record can index; it is "
                    f"dropped.",
                    stacklevel=5,
                )
                continue
            entries.append((key, _GEODOUBLES_ID, len(numbers), len(doubles)))
            doubles += numbers
        elif _is_u16(value):
            entries.append((key, 0, 1, int(value)))
        else:
            warnings.warn(
                f".las: las_geokeys key {key} holds {value!r}, which is neither a "
                f"16-bit integer, a float, a list of floats nor a string; it is "
                f"dropped.",
                stacklevel=5,
            )
    # GeoTIFF asks for the keys in ascending order; a reader may bisect them.
    entries.sort()
    directory = struct.pack(_GEOKEY_DIRECTORY, 1, 1, 0, len(entries)) + b"".join(
        struct.pack(_GEOKEY_DIRECTORY, *e) for e in entries
    )
    out = [_Vlr(_PROJECTION_USER, _GEOKEYS_ID, "GeoTIFF keys", directory)]
    if doubles:
        out.append(
            _Vlr(
                _PROJECTION_USER,
                _GEODOUBLES_ID,
                "GeoTIFF doubles",
                struct.pack(f"<{len(doubles)}d", *doubles),
            )
        )
    if text:
        out.append(
            _Vlr(
                _PROJECTION_USER,
                _GEOASCII_ID,
                "GeoTIFF ascii",
                text.encode("ascii", "replace"),
            )
        )
    return out


def _pack_header(
    g: dict[str, Any],
    ver: tuple[int, int],
    pf: int,
    compressed: bool,
    record_length: int,
    n: int,
    out: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    ints: np.ndarray,
    *,
    n_vlrs: int,
    data_offset: int,
    evlr_start: int,
    n_evlrs: int,
    has_wkt: bool,
) -> bytes:
    encoding = (_header_u16(g, _KEY_ENCODING) & ~_WKT_BIT) | (
        _WKT_BIT if has_wkt else 0
    )
    if ver < _DEFAULT_VERSION:
        encoding = 0
    year, day = 0, 0
    if _KEY_DATE in g:
        given = g[_KEY_DATE]
        try:
            date = () if isinstance(given, str) else tuple(given)
        except TypeError:
            date = ()
        if len(date) != 2 or not all(_is_u16(v) for v in date):
            raise CodecError(
                f".las: {_KEY_DATE} must be (year, day of year), two integers in "
                f"0..{_U16_MAX}, not {g[_KEY_DATE]!r}."
            )
        year, day = (int(v) for v in date)
    by_return = np.bincount(
        out["returns"] & (0x0F if pf in _NEW_STYLE else 0x07), minlength=16
    ).tolist()
    legacy_returns = by_return[1:6]
    if n == 0:
        box = [0.0] * 6
    else:
        hi = ints.max(axis=0) * scale + offset
        lo = ints.min(axis=0) * scale + offset
        box = [hi[0], lo[0], hi[1], lo[1], hi[2], lo[2]]
    legacy_ok = pf not in _NEW_STYLE and n <= _U32_MAX
    header = struct.pack(
        _HEADER_1_2,
        _SIGNATURE,
        _header_u16(g, _KEY_SOURCE_ID),
        encoding,
        0,
        0,
        0,
        b"",
        ver[0],
        ver[1],
        _header_text(g, _KEY_SYSTEM, ""),
        _header_text(g, _KEY_SOFTWARE, _DEFAULT_SOFTWARE),
        day,
        year,
        _HEADER_SIZES[ver],
        data_offset,
        n_vlrs,
        pf | (_COMPRESSED_BIT if compressed else 0),
        record_length,
        n if legacy_ok else 0,
        *(legacy_returns if legacy_ok else [0] * 5),
        *scale.tolist(),
        *offset.tolist(),
        *box,
    )
    if ver >= (1, 3):
        header += struct.pack("<Q", 0)
    if ver >= (1, 4):
        header += struct.pack(
            "<QIQ15Q",
            evlr_start,
            n_evlrs,
            n,
            *by_return[1:16],
        )
    return header
