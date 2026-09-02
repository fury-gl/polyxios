from bisect import bisect_left
from itertools import accumulate
import mmap
import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    POLYXIOS_TO_VTK,
    VTK_TO_POLYXIOS,
)
from polyxios._globals import globals_for_write
from polyxios._io import (
    Source,
    can_seek,
    is_buffer,
    open_block,
    open_read,
    open_write,
    read_bytes,
    source_name,
)
from polyxios._tags import mask_arrays, tags_from_masks
from polyxios._types import PolyData
from polyxios.exceptions import (
    CodecError,
    IndexOverflowError,
    LazyReadError,
    UnknownElementTypeError,
)
from polyxios.validate import validate_header

try:
    from polyxios._vtk_parse import (  # type: ignore[import]
        parse_ascii_cells_v42,
        parse_ascii_coords,
    )

    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

EXTENSION: str = ".vtk"

MAX_CONNECTIVITY_INDEX_V42: int = 2**31 - 1
MAX_CONNECTIVITY_INDEX_V51: int = 2**63 - 1
MAX_CONNECTIVITY_INDEX: int = MAX_CONNECTIVITY_INDEX_V42

_VTK_DTYPE_MAP: dict[str, str] = {
    "float": "f4",
    "double": "f8",
    "int": "i4",
    "long": "i8",
    "long_long": "i8",
    "unsigned_int": "u4",
    "unsigned_long": "u8",
    "unsigned_long_long": "u8",
    "short": "i2",
    "unsigned_short": "u2",
    "char": "i1",
    "signed_char": "i1",
    "unsigned_char": "u1",
    "vtkidtype": "i8",
    "idtype": "i8",
}

# What a FIELD header calls the dtype an array is held in. The map above reads
# every spelling a file may use; this one writes, so it names one apiece - and
# only the types a legacy reader is sure to know, which is why a bool goes out
# as the unsigned char it is stored as and a float16 as the float it widens to.
_NP_TO_VTK_DTYPE: dict[str, str] = {
    "b1": "unsigned_char",
    "i1": "char",
    "i2": "short",
    "i4": "int",
    "i8": "long",
    "u1": "unsigned_char",
    "u2": "unsigned_short",
    "u4": "unsigned_int",
    "u8": "unsigned_long",
    "f4": "float",
    "f8": "double",
}

# A dtype no FIELD header names - float16, a datetime - is written as the
# double it converts to, the way an attribute array already is.
_FIELD_FALLBACK: str = "double"

# Keys a reader of a structured dataset records to say what grid it expanded.
# The writer spells an UNSTRUCTURED_GRID and has no use for them, and writing
# them as field data would hand the next reader a second copy of a grid it
# already rebuilt.
_GRID_KEYS: frozenset[str] = frozenset({"vtk_dimensions", "vtk_origin", "vtk_spacing"})

# What a binary header is read as when it leaves its type field out. The
# format makes the field mandatory, so this only ever answers a malformed
# one - but every reader here has to answer it the same way, or the same
# file reads back as two different arrays depending on which one opened it.
_DEFAULT_VTK_DTYPE: str = "float"

# What a legacy header field cannot hold: it names an array in a
# whitespace-separated field, and nothing in the format escapes one.
_WHITESPACE: re.Pattern[str] = re.compile(r"\s")

# A binary COLOR_SCALARS component is an unsigned char standing for the 0..1
# float an ASCII file writes; dividing puts both flavours on one scale.
_COLOR_SCALE: float = 255.0

# The keywords that open an attribute section rather than name an array in
# one. They fall through the same branch an unhandled array does, and are no
# more dropped than the section they announce.
_SECTION_KEYWORDS: frozenset[str] = frozenset({"POINT_DATA", "CELL_DATA"})

# The keywords that name an array inside an attribute section. Used to stop a
# METADATA block that was written without its blank terminator, so a
# malformed one costs its own array rather than every array after it.
_ATTRIBUTE_KEYWORDS: frozenset[str] = frozenset(
    {
        "SCALARS",
        "COLOR_SCALARS",
        "VECTORS",
        "NORMALS",
        "TEXTURE_COORDINATES",
        "TENSORS",
        "FIELD",
        "LOOKUP_TABLE",
    }
)

# Every keyword that ends an attribute section, whether it opens another one
# or returns to the geometry.
_ATTRS_STOP_KEYWORDS: tuple[str, ...] = (
    "POINT_DATA",
    "CELL_DATA",
    "POINTS",
    "CELLS",
    "CELL_TYPES",
)

# The keywords that open or continue a cell array. VTK follows one with a
# METADATA block as it does an attribute, so a block sits inside a v5.1 CELLS
# section between its offsets and its connectivity.
_CELL_KEYWORDS: frozenset[str] = frozenset(
    {"OFFSETS", "CONNECTIVITY", "POLYGONS", "LINES", "VERTICES", "TRIANGLE_STRIPS"}
)

# What ends a METADATA block that was written without its blank terminator.
# The geometry keywords belong here as much as the attribute ones: a block
# left open at the end of a POINT_DATA section would otherwise swallow the
# CELLS that follows it and every line to the end of the file, and one left
# open inside a CELLS section would swallow the connectivity.
_METADATA_STOP_KEYWORDS: frozenset[str] = (
    _ATTRIBUTE_KEYWORDS | frozenset(_ATTRS_STOP_KEYWORDS) | _CELL_KEYWORDS
)


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a VTK legacy file (UNSTRUCTURED_GRID or POLYDATA) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtk file.
    lazy
        If True and the file is binary, return arrays backed by mmap (OS-lazy pages).
        Raises LazyReadError for ASCII files.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    LazyReadError
        If lazy=True and the file uses ASCII data sections.
    CodecError
        On unsupported dataset type or malformed data.
    UnknownElementTypeError
        If the file contains a VTK cell type not in _element_types.VTK_TO_POLYXIOS.

    Notes
    -----
    A structured dataset is expanded into an explicit point array, which the
    grid it came from cannot be read back out of, so the grid is kept in
    ``global_attrs`` as ``vtk_dimensions`` and - for ``STRUCTURED_POINTS`` -
    ``vtk_origin`` and ``vtk_spacing``. It is the grid the reader used: a
    ``RECTILINEAR_GRID`` is laid out by its coordinate arrays, so a
    ``DIMENSIONS`` that disagrees with them is warned about and the array
    lengths are what is kept, while a ``STRUCTURED_GRID`` and a
    ``STRUCTURED_POINTS`` have nothing but their header and keep it as
    written. These are read-only: ``write`` always emits an
    ``UNSTRUCTURED_GRID`` and does not consume them.

    Binary ``COLOR_SCALARS`` holds one unsigned char per component where the
    ASCII flavour holds a float in 0..1. The byte is scaled onto that range,
    so the same colour reads back the same way whichever encoding the file
    used.
    """
    # The dataset keyword lives in the header and decides which reader runs,
    # and that reader starts again from the top of the file. A stream that
    # cannot be put back cannot serve both passes: say so before a byte is
    # taken off it, rather than let the second pass start wherever the first
    # one stopped and hand back a mesh of nothing.
    if is_buffer(path) and not can_seek(path):
        raise CodecError(
            f"'{source_name(path)}' is a stream that cannot seek, and this "
            f"format is read by walking its header and then its body from the "
            f"start. Read the stream into io.BytesIO first, or pass a path."
        )

    # The file's size is not measured here. Every reader below reads the
    # whole file anyway, so each takes the size from what it read: measuring a
    # compressed source separately costs a whole decompression pass that the
    # read then repeats.
    with open_read(path) as fh:
        start = fh.tell()
        # The banner declares a version no reader here consults: each asks the
        # body which spelling it uses, which a file gets right more often
        # than its own header does.
        fh.readline()  # version banner
        fh.readline()  # title line
        # VTK v1.0 files can have blank lines between the title and BINARY/ASCII marker.
        data_type = ""
        for _ in range(8):
            data_type = fh.readline().decode("ascii", errors="replace").strip().upper()
            if data_type:
                break
        # Some files also have blank lines before the DATASET line.
        dataset_line = ""
        for _ in range(8):
            dataset_line = (
                fh.readline().decode("ascii", errors="replace").strip().upper()
            )
            if dataset_line:
                break
        # Every reader below starts from the top of the file, so the handle
        # goes back to where the header sniff found it.
        if can_seek(fh):
            fh.seek(start)

    is_binary = data_type == "BINARY"

    if "UNSTRUCTURED_GRID" in dataset_line:
        if is_binary:
            return _read_binary(path, lazy=lazy)
        else:
            if lazy:
                raise LazyReadError("VTK ASCII format does not support lazy reads.")
            return _read_ascii(path)
    elif "POLYDATA" in dataset_line:
        if lazy:
            raise LazyReadError("VTK POLYDATA format does not support lazy reads.")
        if is_binary:
            return _read_polydata_binary(path)
        return _read_polydata_ascii(path)
    elif "RECTILINEAR_GRID" in dataset_line:
        if lazy:
            raise LazyReadError(
                "VTK RECTILINEAR_GRID format does not support lazy reads."
            )
        return _read_rectilinear_grid(path, is_binary=is_binary)
    elif "STRUCTURED_GRID" in dataset_line:
        if lazy:
            raise LazyReadError(
                "VTK STRUCTURED_GRID format does not support lazy reads."
            )
        return _read_structured_grid(path, is_binary=is_binary)
    elif "STRUCTURED_POINTS" in dataset_line:
        if lazy:
            raise LazyReadError(
                "VTK STRUCTURED_POINTS format does not support lazy reads."
            )
        return _read_structured_points(path, is_binary=is_binary)
    elif "FIELD" in dataset_line:
        # The line still carries its 'DATASET ' keyword here, so asking what
        # it starts with never matched and every field-data file was refused
        # by the branch below instead of read.
        if lazy:
            raise LazyReadError("VTK FIELD format does not support lazy reads.")
        return _read_field_data(path)
    else:
        raise CodecError(
            f"VTK codec supports DATASET UNSTRUCTURED_GRID or POLYDATA, got: {dataset_line!r}"
        )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a VTK legacy unstructured grid file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default: False), write binary data sections (big-endian).
    vtk_version
        '4.2' (default) or '5.1'. v4.2 uses classic CELLS layout compatible
        with all VTK readers. v5.1 uses OFFSETS+CONNECTIVITY.

    Raises
    ------
    IndexOverflowError
        If connectivity.max() > MAX_CONNECTIVITY_INDEX for the chosen version.
    """
    binary: bool = bool(opts.get("binary", False))
    vtk_version: str = str(opts.get("vtk_version", "4.2"))

    max_allowed = (
        MAX_CONNECTIVITY_INDEX_V51
        if vtk_version == "5.1"
        else MAX_CONNECTIVITY_INDEX_V42
    )
    if poly.connectivity.size > 0 and int(poly.connectivity.max()) > max_allowed:
        raise IndexOverflowError("vtk", max_allowed, int(poly.connectivity.max()))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    with open_write(path) as fh:
        # ASCII header
        fh.write(f"# vtk DataFile Version {vtk_version}\n".encode())
        fh.write(b"Written by polyxios\n")
        fh.write(b"BINARY\n" if binary else b"ASCII\n")
        fh.write(b"DATASET UNSTRUCTURED_GRID\n")

        # FIELD FieldData, the mesh's own metadata. It sits between the
        # dataset keyword and the geometry, which is where VTK's own writer
        # puts it and the only place a reader looks for a block belonging to
        # the dataset rather than to its points or cells.
        _write_field_block(poly, fh, binary=binary)

        # POINTS
        fh.write(f"POINTS {n_verts} double\n".encode())
        if binary:
            _write_bin_f64(poly.vertices.ravel(), fh)
        else:
            _write_ascii_f64(
                np.ascontiguousarray(poly.vertices, dtype=np.float64).ravel(),
                fh,
                per_line=3,
            )

        if vtk_version == "5.1":
            _write_cells_v51(poly, fh, binary)
        else:
            _write_cells_v42(poly, fh, binary)

        # CELL_TYPES
        fh.write(f"CELL_TYPES {n_elems}\n".encode())
        vtk_types = np.array(
            [_polyxios_to_vtk_code(poly.element_types[i]) for i in range(n_elems)],
            dtype=np.int32,
        )
        if binary:
            _write_bin_i32(vtk_types, fh)
        else:
            fh.write((" ".join(str(t) for t in vtk_types) + "\n").encode())

        # POINT_DATA and CELL_DATA. A tag group travels among them as one
        # column of ones and zeros named for it: the section holds one value
        # per entity, and an element in two groups is named by both columns,
        # which the single reference a Medit or Netgen record carries cannot
        # say. The column is a double like every other array here, and reads
        # back as the group it spells.
        point_arrays = poly.vertex_attrs | mask_arrays(
            poly.vertex_tags, n_verts, fmt=EXTENSION, kind="point"
        )
        cell_arrays = poly.element_attrs | mask_arrays(
            poly.element_tags, n_elems, fmt=EXTENSION, kind="cell"
        )

        if point_arrays:
            fh.write(f"POINT_DATA {n_verts}\n".encode())
            for name, arr in point_arrays.items():
                _write_vtk_array(name, arr, fh, binary=binary)

        if cell_arrays:
            fh.write(f"CELL_DATA {n_elems}\n".encode())
            for name, arr in cell_arrays.items():
                _write_vtk_array(name, arr, fh, binary=binary)


# --- internal helpers ---


def _polyxios_to_vtk_code(type_code: int) -> int:
    name = ELEMENT_TYPES_INV.get(int(type_code))
    if name is None or name not in POLYXIOS_TO_VTK:
        return 7  # fallback to polygon
    return POLYXIOS_TO_VTK[name]


def _write_cells_v42(poly: PolyData, fh: object, binary: bool) -> None:
    """Write v4.2 CELLS + CELL_TYPES sections."""
    n_elems = len(poly.element_types)
    # total_size = connectivity size + n_elems (each cell prefixed by count)
    total_size = len(poly.connectivity) + n_elems
    fh.write(f"CELLS {n_elems} {total_size}\n".encode())  # type: ignore[union-attr]

    if binary:
        # Build interleaved [count, idx0, idx1, ...] int32 stream
        parts: list[np.ndarray] = []
        for i in range(n_elems):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            cnt = e - s
            parts.append(np.array([cnt], dtype=np.int32))
            parts.append(poly.connectivity[s:e].astype(np.int32))
        if parts:
            _write_bin_i32(np.concatenate(parts), fh)
    else:
        for i in range(n_elems):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            face = poly.connectivity[s:e]
            fh.write(
                (str(e - s) + " " + " ".join(str(v) for v in face) + "\n").encode()
            )  # type: ignore[union-attr]


def _write_cells_v51(poly: PolyData, fh: object, binary: bool) -> None:
    """Write v5.1 OFFSETS + CONNECTIVITY sections.

    The two numbers on a v5.1 ``CELLS`` line are the length of the OFFSETS
    array and the length of the CONNECTIVITY array - not the cell count,
    which is one less than the first of them. VTK's own reader takes the
    first number literally and stops with "Error reading cell array
    connectivity header" when it does not match the values that follow.
    """
    conn_size = len(poly.connectivity)
    offsets64 = poly.offsets.astype(np.int64)

    fh.write(f"CELLS {len(offsets64)} {conn_size}\n".encode())  # type: ignore[union-attr]
    fh.write(b"OFFSETS vtktypeint64\n")
    if binary:
        fh.write(offsets64.astype(np.dtype(">i8")).tobytes())  # type: ignore[union-attr]
    else:
        fh.write((" ".join(str(x) for x in offsets64) + "\n").encode())  # type: ignore[union-attr]

    fh.write(b"CONNECTIVITY vtktypeint64\n")
    conn64 = poly.connectivity.astype(np.int64)
    if binary:
        fh.write(conn64.astype(np.dtype(">i8")).tobytes())  # type: ignore[union-attr]
    else:
        fh.write((" ".join(str(x) for x in conn64) + "\n").encode())  # type: ignore[union-attr]


def _spellable_name(name: str) -> bool:
    """Say whether a legacy header can name an array this, warning when not.

    Parameters
    ----------
    name
        The array's name.

    Returns
    -------
    bool
        True when the name is one whitespace-separated token.

    Notes
    -----
    A legacy header names its array in a whitespace-separated field, so a name
    holding a space is read back as a name and a stray token - and the array
    after it as that token's values. There is no escaping in the format to
    fall back on, so the array is dropped and said so. Every section spells
    its names the same way, which is why the attribute writer and the field
    block ask here rather than each carrying the rule.
    """
    if name and not _WHITESPACE.search(name):
        return True
    warnings.warn(
        f".vtk: array name {name!r} holds whitespace, which a legacy"
        " header field cannot spell; the array is not written.",
        UserWarning,
        stacklevel=4,
    )
    return False


def _write_field_block(poly: PolyData, fh: object, *, binary: bool) -> None:
    """Write the mesh's metadata as a ``FIELD FieldData`` block.

    Parameters
    ----------
    poly
        The mesh being written.
    fh
        Open binary file object.
    binary
        Write each payload as big-endian raw bytes rather than spelling it.

    Notes
    -----
    A field array is bound to neither the points nor the cells, so its header
    carries both its component count and its tuple count - the only two
    numbers that say where it ends and the next array begins. Unlike a point
    or cell array, which every branch of this writer spells as a double, a
    field array keeps the type it is held in: it is metadata, and an integer
    that comes home a float is a different value to whatever reads it next.
    """
    spelled: dict[str, np.ndarray] = {}
    for name, arr in globals_for_write(
        poly, reserved=_GRID_KEYS, fmt=EXTENSION
    ).items():
        # A plain loop rather than a comprehension: the warning below counts
        # the frames between itself and the caller, and a comprehension is
        # one more of them on the Python versions that give it its own.
        if _spellable_name(name):
            spelled[name] = arr  # noqa: PERF403 - see the comment above
    if not spelled:
        return
    # The count is written from what survived the name check, not from what
    # the mesh held: a header promising an array the block does not carry
    # sends the next reader into the geometry looking for it.
    fh.write(f"FIELD FieldData {len(spelled)}\n".encode())  # type: ignore[union-attr]
    for name, arr in spelled.items():
        vtk_name = _NP_TO_VTK_DTYPE.get(arr.dtype.str.lstrip("<>|="), _FIELD_FALLBACK)
        np_str = _VTK_DTYPE_MAP[vtk_name]
        n_comp = _components(arr)
        values = np.ascontiguousarray(arr, dtype=np_str)
        fh.write(  # type: ignore[union-attr]
            f"{name} {n_comp} {values.size // n_comp} {vtk_name}\n".encode()
        )
        if binary:
            fh.write(values.astype(">" + np_str).tobytes())  # type: ignore[union-attr]
            fh.write(b"\n")  # type: ignore[union-attr]
        elif np_str[0] == "f":
            _write_ascii_f64(values.ravel().astype(np.float64), fh, per_line=n_comp)
        else:
            spelled_row = " ".join(map(str, values.ravel().tolist()))
            fh.write((spelled_row + "\n").encode())  # type: ignore[union-attr]


def _parse_field_block_ascii(
    lines: list[str], i: int, n_arrays: int
) -> tuple[int, dict[str, np.ndarray]]:
    """Read a top-level ASCII ``FIELD`` block into mesh metadata.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index of the first line after the ``FIELD`` header.
    n_arrays
        Arrays the header says the block holds.

    Returns
    -------
    tuple[int, dict of str to numpy.ndarray]
        The line just past the block, and its arrays keyed by name.

    Raises
    ------
    CodecError
        If the file ends before the block's headers do.
    """
    found: dict[str, np.ndarray] = {}
    n_lines = len(lines)
    for _ in range(n_arrays):
        i = _next_header(lines, i)
        if i >= n_lines:
            raise CodecError(
                f".vtk: FIELD declares {n_arrays} arrays but the file ends"
                " before their headers."
            )
        parts = lines[i].strip().split()
        where = f"line {i + 1}"
        name = parts[0]
        n_comp = _attr_count(parts, 1, where)
        n_tuples = _attr_count(parts, 2, where)
        np_str = _VTK_DTYPE_MAP.get(
            parts[3].lower() if len(parts) > 3 else _DEFAULT_VTK_DTYPE, "f8"
        )
        i += 1
        i, tokens = _read_ascii_tokens(lines, i, n_tuples * n_comp, name=name)
        arr = _field_values(tokens, np_str, name)
        found[name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
    return i, found


def _parse_field_block_binary(
    mm: "mmap.mmap | bytes",
    mv: memoryview,
    pos: int,
    file_size: int,
    n_arrays: int,
) -> tuple[int, dict[str, np.ndarray]]:
    """Read a top-level binary ``FIELD`` block into mesh metadata.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset of the first line after the ``FIELD`` header.
    file_size
        Size of the file.
    n_arrays
        Arrays the header says the block holds.

    Returns
    -------
    tuple[int, dict of str to numpy.ndarray]
        The byte offset just past the block, and its arrays keyed by name.
    """
    found: dict[str, np.ndarray] = {}
    for _ in range(n_arrays):
        pos = _next_binary_header(mm, mv, pos, file_size)
        hdr_start = pos
        hdr_end = mm.find(b"\n", pos)
        if hdr_end == -1:
            break
        hdr = bytes(mv[pos:hdr_end]).decode("ascii", errors="replace").strip()
        pos = hdr_end + 1
        parts = hdr.split()
        if len(parts) < 4:
            continue
        where = f"byte {hdr_start}"
        name = parts[0]
        n_comp = _attr_count(parts, 1, where)
        n_tuples = _attr_count(parts, 2, where)
        np_str = _binary_dtype(parts, 3, where)
        n_bytes = n_tuples * n_comp * np.dtype(np_str).itemsize
        _check_block(pos, n_bytes, file_size, name=name)
        arr = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_str).astype(
            np_str.lstrip(">")
        )
        pos += n_bytes
        pos = _skip_newline(mv, pos, file_size)
        found[name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
    return pos, found


def _write_vtk_array(name: str, arr: np.ndarray, fh: object, *, binary: bool) -> None:
    """Write a single attribute array to a VTK file (SCALARS/VECTORS/TENSORS).

    An array's header says nothing about whether it belongs to the points or
    the cells - the section it is written under does, and the caller has
    already opened that. Nor does the file's version change how a section is
    spelled: every version this writer emits spells them the same way.

    Parameters
    ----------
    name
        Array name.
    arr
        Values, one row per point or cell.
    fh
        Open binary file object.
    binary
        Write the payload as big-endian doubles rather than spelling it.
    """
    if not _spellable_name(name):
        return

    values = np.ascontiguousarray(arr, dtype=np.float64)

    if values.ndim == 1:
        header = f"SCALARS {name} double 1\nLOOKUP_TABLE default\n"
        per_line = values.size
    elif values.ndim == 2 and values.shape[1] == 3:
        header = f"VECTORS {name} double\n"
        per_line = 3
    elif values.ndim == 3 and values.shape[1:] == (3, 3):
        header = f"TENSORS {name} double\n"
        per_line = 3
    elif values.ndim == 2 and values.shape[1] == 6:
        # Voigt 6-component - expand to the full 3x3 a TENSORS section holds.
        values = values[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]]
        header = f"TENSORS {name} double\n"
        per_line = 3
    else:
        # Generic multi-component. Every dimension past the first belongs to
        # the tuple, so a (n, 2, 4) array is eight components and not two.
        header = f"SCALARS {name} double {_components(values)}\n"
        header += "LOOKUP_TABLE default\n"
        per_line = values.size

    fh.write(header.encode())  # type: ignore[union-attr]
    if binary:
        # Every branch writes its payload the same way. The two TENSORS ones
        # used to spell their numbers whatever was asked for, which put a run
        # of ASCII in the middle of a binary file that no reader can take -
        # this one answered it with a CodecError about a short block.
        _write_bin_f64(values.ravel(), fh)
    else:
        _write_ascii_f64(values.ravel(), fh, per_line=per_line)


def _components(arr: np.ndarray) -> int:
    """Count the components one tuple of an attribute array holds.

    Parameters
    ----------
    arr
        The array, one tuple per row.

    Returns
    -------
    int
        Components per tuple; 1 for a one-dimensional array.
    """
    count = 1
    for dim in arr.shape[1:]:
        count *= int(dim)
    return count


def _write_ascii_f64(flat: np.ndarray, fh: object, *, per_line: int) -> None:
    """Spell a run of doubles into a legacy VTK file.

    ``repr`` of a Python float is the shortest decimal that reads back as
    that double, so a value written this way survives the round trip; the
    fixed ``'%.10g'`` this used quietly dropped the last seven digits of
    every coordinate a ``double`` section claimed to hold. The whole block
    is joined and written once: a write per line costs a syscall per point.

    Parameters
    ----------
    flat
        The values, already flat.
    fh
        Open binary file object.
    per_line
        Values per line; ``flat.size`` puts them all on one. A run it does
        not divide goes on one line rather than losing its last few values.
    """
    values = flat.tolist()
    if per_line >= len(values) or len(values) % per_line:
        body = " ".join(map(repr, values))
    else:
        # One shared iterator spells each value once and zip cuts the run
        # into rows; grouping a spelled list by slices copies it again, and
        # a join per row costs more than the one this pays.
        spelled = map(repr, values)
        body = "\n".join(map(" ".join, zip(*[spelled] * per_line)))
    fh.write((body + "\n").encode() if body else b"\n")  # type: ignore[union-attr]


def _write_bin_f64(arr: np.ndarray, fh: object) -> None:
    fh.write(arr.astype(np.dtype(">f8")).tobytes())  # type: ignore[union-attr]


def _write_bin_i32(arr: np.ndarray, fh: object) -> None:
    fh.write(arr.astype(np.dtype(">i4")).tobytes())  # type: ignore[union-attr]


def _read_polydata_ascii(path: Source) -> PolyData:
    """Read a VTK legacy ASCII POLYDATA file and convert to PolyData.

    POLYDATA uses named topology sections (POLYGONS, LINES, VERTICES,
    TRIANGLE_STRIPS) instead of CELLS + CELL_TYPES.  Each section maps
    to a polyxios element type determined by the vertex count per cell:

    POLYGONS:
        3 vertices  -> triangle (code 5)
        4 vertices  -> quad     (code 9)
        N vertices  -> polygon  (code 7)
    LINES:
        2 vertices  -> line      (code 3)
        N vertices  -> poly_line (code 4)
    VERTICES:
        1 vertex    -> vertex      (code 1)
        N vertices  -> poly_vertex (code 2)
    TRIANGLE_STRIPS:
        always      -> triangle_strip (code 6)
    """
    raw = read_bytes(path)
    file_size = len(raw)
    content = raw.decode("ascii", errors="replace")

    lines = content.splitlines()
    # Skip lines until we find POINTS (header may have blank lines / extra lines)
    i = 0
    n_lines = len(lines)

    vertices = np.zeros((0, 3), dtype=np.float64)
    conn_list: list[int] = []
    off_list: list[int] = [0]
    type_list: list[int] = []
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    global_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        upper = line.upper()

        if upper.startswith("FIELD"):
            # A FIELD block out here belongs to the dataset rather than to
            # its points or cells - the one inside a POINT_DATA section is
            # read by that section's own parser, which never returns here.
            n_arrays = _attr_count(line.split(), 2, f"line {i + 1}")
            i, global_attrs = _parse_field_block_ascii(lines, i + 1, n_arrays)

        elif upper.startswith("POINTS"):
            parts = line.split()
            n_verts = _attr_count(parts, 1, f"line {i + 1}")
            i += 1
            validate_header(n_verts, 0, 0, file_size)
            if _HAS_CYTHON:
                vertices = parse_ascii_coords(lines, i, n_verts)
                i += n_verts
            else:
                verts_raw: list[float] = []
                while len(verts_raw) < n_verts * 3:
                    verts_raw.extend(float(x) for x in lines[i].split())
                    i += 1
                vertices = np.array(verts_raw, dtype=np.float64).reshape(n_verts, 3)

        elif (kind := _polydata_section(upper)) is not None:
            parts = line.split()
            where = f"line {i + 1}"
            n_cells = _attr_count(parts, 1, where)
            total_vals = _attr_count(parts, 2, where)
            i += 1

            if i < n_lines and lines[i].strip().upper().startswith("OFFSETS"):
                # v5.1, which every VTK release since 9.0 writes by default:
                # the cells are an offsets array and a flat connectivity one,
                # and the first number on the header line counts the offsets.
                conn_v51, off_v51, i = _parse_v51_cells_ascii(lines, i)
                base = off_list[-1]
                conn_list.extend(conn_v51.tolist())
                off_list.extend((base + off_v51[1:]).tolist())
                counts = np.diff(off_v51)
                type_list.extend(_polydata_cell_type(kind, int(cnt)) for cnt in counts)
                n_elems += len(counts)
                continue

            tokens = parts[3:]
            while len(tokens) < total_vals and i < n_lines:
                tokens.extend(lines[i].split())
                if len(tokens) >= total_vals:
                    break
                i += 1

            idx = 0
            for _ in range(n_cells):
                cnt = int(tokens[idx])
                idx += 1

                conn_list.extend(int(t) for t in tokens[idx : idx + cnt])

                idx += cnt
                off_list.append(off_list[-1] + cnt)
                type_list.append(_polydata_cell_type(kind, cnt))
            n_elems += n_cells
            if len(tokens) >= total_vals:
                i += 1

        elif upper.startswith("POINT_DATA"):
            n_pd = _attr_count(line.split(), 1, f"line {i + 1}")
            i += 1
            i, vertex_attrs = _parse_vtk_data_attrs(
                lines, i, n_pd, n_verts, kind="point"
            )

        elif upper.startswith("CELL_DATA"):
            n_cd = _attr_count(line.split(), 1, f"line {i + 1}")
            i += 1
            i, element_attrs = _parse_vtk_data_attrs(
                lines, i, n_cd, n_elems, kind="cell"
            )

        else:
            i += 1

    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(off_list, dtype=np.int32),
        element_types=np.array(type_list, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


_POLYDATA_CELL_TYPES: dict[str, tuple[dict[int, str], str]] = {
    "POLYGONS": ({3: "triangle", 4: "quad"}, "polygon"),
    "LINES": ({2: "line"}, "poly_line"),
    "VERTICES": ({1: "vertex"}, "poly_vertex"),
    "TRIANGLE_STRIPS": ({}, "triangle_strip"),
}


def _polydata_cell_type(kind: str, count: int) -> int:
    """Name the element a POLYDATA cell of this size in this section is.

    Parameters
    ----------
    kind
        The section keyword, upper case: ``'POLYGONS'``, ``'LINES'``,
        ``'VERTICES'`` or ``'TRIANGLE_STRIPS'``.
    count
        Points the cell holds.

    Returns
    -------
    int
        The polyxios element type code.
    """
    special, general = _POLYDATA_CELL_TYPES[kind]
    return ELEMENT_TYPES[special.get(count, general)]


def _polydata_section(upper: str) -> str | None:
    """Name the POLYDATA cell section a line opens, if it opens one.

    Parameters
    ----------
    upper
        The line, upper case.

    Returns
    -------
    str or None
        The section keyword, or None when the line opens no cell section.
    """
    for kind in _POLYDATA_CELL_TYPES:
        if upper.startswith(kind):
            return kind
    return None


def _read_polydata_binary(path: Source) -> PolyData:
    """Read a VTK legacy binary POLYDATA file."""
    with open_block(path, fmt=".vtk") as mm:
        file_size = len(mm)
        mv = memoryview(mm)
        pos = 0
        for _ in range(4):
            pos = mm.find(b"\n", pos) + 1
        poly = _parse_binary_polydata_body(mm, mv, pos, file_size)
        del mv
    return poly


def _parse_binary_polydata_body(
    mm: mmap.mmap | bytes,
    mv: memoryview,
    start_pos: int,
    file_size: int,
) -> PolyData:
    """Parse binary data sections of a VTK POLYDATA file."""
    pos = start_pos
    vertices = np.zeros((0, 3), dtype=np.float64)
    all_conn: list[np.ndarray] = []
    all_offs: list[int] = [0]
    all_types: list[int] = []
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    global_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while pos < file_size:
        line_start = pos
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            break
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1

        if not line:
            continue

        upper = line.upper()
        parts = line.split()

        if upper.startswith("FIELD"):
            # A FIELD block out here belongs to the dataset rather than to
            # its points or cells - the one inside a POINT_DATA section is
            # read by that section's own parser, which never returns here.
            pos, global_attrs = _parse_field_block_binary(
                mm, mv, pos, file_size, _attr_count(parts, 2, f"byte {line_start}")
            )

        elif upper.startswith("POINTS"):
            n_verts = _attr_count(parts, 1, f"byte {line_start}")
            # The header names the type the block holds, and it is not always
            # a float: reading a POINTS written as 'int' or 'short' at four
            # bytes a value hands back coordinates the file never held.
            np_dt = _binary_dtype(parts, 2, f"byte {line_start}")
            n_bytes = n_verts * 3 * np.dtype(np_dt).itemsize
            validate_header(n_verts, 0, 0, file_size)
            # validate_header bounds the block against the whole file, which a
            # POINTS section sitting behind a long header clears while still
            # running off the end; a short slice reshapes into a ValueError
            # that names neither the array nor the file.
            _check_block(pos, n_bytes, file_size, name=parts[0])
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt)
            vertices = raw.astype(np.float64).reshape(n_verts, 3)
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)

        elif (kind := _polydata_section(upper)) is not None:
            where = f"byte {line_start}"
            n_cells = _attr_count(parts, 1, where)
            total_vals = _attr_count(parts, 2, where)

            next_end = mm.find(b"\n", pos)
            next_line = (
                ""
                if next_end == -1
                else bytes(mv[pos:next_end]).decode("ascii", errors="replace").strip()
            )
            if next_line.upper().startswith("OFFSETS"):
                # v5.1, as every VTK release since 9.0 writes it.
                conn_v51, off_v51, pos = _parse_v51_cells_binary(
                    mm, mv, next_end + 1, n_cells, file_size
                )
                base = all_offs[-1]
                all_conn.append(conn_v51.astype(np.int32))
                all_offs.extend((base + off_v51[1:]).tolist())
                counts = np.diff(off_v51)
                all_types.extend(_polydata_cell_type(kind, int(cnt)) for cnt in counts)
                n_elems += len(counts)
                continue

            n_bytes_cells = total_vals * 4
            _check_block(pos, n_bytes_cells, file_size, name=kind)
            raw_cells = np.frombuffer(
                bytes(mv[pos : pos + n_bytes_cells]), dtype=">i4"
            ).astype(np.int32)
            pos += n_bytes_cells
            pos = _skip_newline(mv, pos, file_size)

            idx = 0
            for _ in range(n_cells):
                cnt = int(raw_cells[idx])
                idx += 1
                cell = raw_cells[idx : idx + cnt]
                idx += cnt
                all_conn.append(cell)
                all_offs.append(all_offs[-1] + cnt)
                all_types.append(_polydata_cell_type(kind, cnt))
            n_elems += n_cells

        elif upper.startswith("POINT_DATA"):
            n_pd = _attr_count(parts, 1, f"byte {line_start}")
            pos, vertex_attrs = _parse_binary_attrs(
                mm, mv, pos, n_pd, file_size, expected=n_verts, kind="point"
            )

        elif upper.startswith("CELL_DATA"):
            n_cd = _attr_count(parts, 1, f"byte {line_start}")
            pos, element_attrs = _parse_binary_attrs(
                mm, mv, pos, n_cd, file_size, expected=n_elems, kind="cell"
            )

    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    connectivity = (
        np.concatenate(all_conn).astype(np.int32)
        if all_conn
        else np.array([], dtype=np.int32)
    )
    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=np.array(all_offs, dtype=np.int32),
        element_types=np.array(all_types, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _read_ascii(path: Source) -> PolyData:
    raw = read_bytes(path)
    file_size = len(raw)
    content = raw.decode("ascii", errors="replace")

    lines = content.splitlines()
    # Skip the 4-line header
    i = 4
    n_lines = len(lines)

    vertices = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.array([], dtype=np.int32)
    offsets = np.array([0], dtype=np.int32)
    element_types_arr = np.array([], dtype=np.uint8)
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    global_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        upper = line.upper()

        if upper.startswith("FIELD"):
            # A FIELD block out here belongs to the dataset rather than to
            # its points or cells - the one inside a POINT_DATA section is
            # read by that section's own parser, which never returns here.
            n_arrays = _attr_count(line.split(), 2, f"line {i + 1}")
            i, global_attrs = _parse_field_block_ascii(lines, i + 1, n_arrays)

        elif upper.startswith("POINTS"):
            parts = line.split()
            n_verts = _attr_count(parts, 1, f"line {i + 1}")
            i += 1
            validate_header(n_verts, 0, 0, file_size)
            if _HAS_CYTHON:
                vertices = parse_ascii_coords(lines, i, n_verts)
                i += n_verts
            else:
                verts_raw: list[float] = []
                while len(verts_raw) < n_verts * 3:
                    verts_raw.extend(float(x) for x in lines[i].split())
                    i += 1
                vertices = np.array(verts_raw, dtype=np.float64).reshape(n_verts, 3)

        elif upper.startswith("CELLS") and not upper.startswith("CELL_TYPES"):
            parts = line.split()
            where = f"line {i + 1}"
            n_elems = _attr_count(parts, 1, where)
            total_size = _attr_count(parts, 2, where)
            i += 1
            validate_header(n_verts, n_elems, total_size, file_size)

            # What follows the header says which spelling this is, not the
            # version in the first line: those compare as strings, so a file
            # declaring 10.0 sorts below 5.1 and its OFFSETS would be read as
            # a v4.2 cell stream. The binary scan has always asked this way.
            if i < n_lines and "OFFSETS" in lines[i].upper():
                connectivity, offsets, i = _parse_v51_cells_ascii(lines, i)
                # The first number on the CELLS line is the length of the
                # offsets array, so the cells are one fewer - and older
                # polyxios files, which put the cell count there, are read
                # by the same count of what OFFSETS actually held.
                n_elems = len(offsets) - 1
            elif _HAS_CYTHON:
                connectivity, offsets = parse_ascii_cells_v42(lines, i, n_elems)
                i += n_elems
            else:
                conn_list: list[int] = []
                off_list: list[int] = [0]
                for _ in range(n_elems):
                    parts2 = lines[i].split()
                    cnt = int(parts2[0])
                    conn_list.extend(int(x) for x in parts2[1 : cnt + 1])
                    off_list.append(off_list[-1] + cnt)
                    i += 1
                connectivity = np.array(conn_list, dtype=np.int32)
                offsets = np.array(off_list, dtype=np.int32)

        elif upper.startswith("CELL_TYPES"):
            n_ct = _attr_count(line.split(), 1, f"line {i + 1}")
            i += 1
            ct_raw: list[int] = []
            while len(ct_raw) < n_ct:
                ct_raw.extend(int(x) for x in lines[i].split())
                i += 1
            type_codes: list[int] = []
            for vtk_code in ct_raw:
                if vtk_code not in VTK_TO_POLYXIOS:
                    raise UnknownElementTypeError("vtk", vtk_code)
                type_codes.append(ELEMENT_TYPES[VTK_TO_POLYXIOS[vtk_code]])
            element_types_arr = np.array(type_codes, dtype=np.uint8)

        elif upper.startswith("POINT_DATA"):
            n_pd = _attr_count(line.split(), 1, f"line {i + 1}")
            i += 1
            i, vertex_attrs = _parse_vtk_data_attrs(
                lines, i, n_pd, n_verts, kind="point"
            )

        elif upper.startswith("CELL_DATA"):
            n_cd = _attr_count(line.split(), 1, f"line {i + 1}")
            i += 1
            i, element_attrs = _parse_vtk_data_attrs(
                lines, i, n_cd, n_elems, kind="cell"
            )

        else:
            i += 1

    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _parse_v51_cells_ascii(
    lines: list[str], i: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse v5.1 OFFSETS + CONNECTIVITY from ASCII lines starting at index i.

    The offsets are counted rather than taken from the ``CELLS`` line, which
    older polyxios releases wrote as the cell count where VTK writes the
    length of the offsets array. Reading to the ``CONNECTIVITY`` keyword
    takes both spellings without having to tell them apart.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index of the ``OFFSETS`` keyword line.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, int]
        Connectivity, offsets, and the line just past the section. The mesh
        has one cell fewer than the offsets array holds values.

    Raises
    ------
    CodecError
        If the ``CONNECTIVITY`` section is missing, or the offsets are not
        a run of integers.
    """
    n_lines = len(lines)
    # line i: "OFFSETS vtktypeint64" or similar
    i += 1
    off_vals: list[int] = []
    while i < n_lines and "CONNECTIVITY" not in lines[i].upper():
        # VTK follows a cell array with its own METADATA block, so one sits
        # between the offsets and the connectivity as often as not. Read as
        # offsets it is a line of words where numbers belong, which used to
        # refuse a file every VTK release since 9.0 writes.
        if lines[i].strip().upper().startswith("METADATA"):
            i = _next_header(lines, i)
            continue
        try:
            off_vals.extend(map(int, lines[i].split()))
        except ValueError as exc:
            raise CodecError(
                f".vtk: line {i + 1} holds {lines[i].strip()!r} where the"
                " OFFSETS of a v5.1 CELLS section should be."
            ) from exc
        i += 1
    if i >= n_lines or not off_vals:
        raise CodecError(
            ".vtk: a v5.1 CELLS section declares OFFSETS but no CONNECTIVITY"
            " follows them."
        )

    # Skip CONNECTIVITY keyword line
    i += 1
    conn_size = off_vals[-1]
    conn_vals: list[int] = []
    while len(conn_vals) < conn_size:
        if i >= n_lines:
            raise CodecError(
                f".vtk: a v5.1 CELLS section declares {conn_size} connectivity"
                f" values but the file ends after {len(conn_vals)}."
            )
        if lines[i].strip().upper().startswith("METADATA"):
            i = _next_header(lines, i)
            continue
        try:
            conn_vals.extend(map(int, lines[i].split()))
        except ValueError as exc:
            raise CodecError(
                f".vtk: line {i + 1} holds {lines[i].strip()!r} where the"
                " CONNECTIVITY of a v5.1 CELLS section should be."
            ) from exc
        i += 1
    return (
        np.array(conn_vals, dtype=np.int32),
        np.array(off_vals, dtype=np.int32),
        i,
    )


def _read_ascii_values(
    lines: list[str], i: int, count: int, *, name: str
) -> tuple[int, list[float]]:
    """Read ``count`` numbers from an ASCII attribute section.

    A legacy file wraps an array over as many lines as it likes, so the only
    way to know an array has ended is to have counted its values. A file that
    ends first is truncated, which is worth saying: reading off the end of the
    line list is an IndexError naming nothing.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index of the first line of values.
    count
        How many numbers the section declares.
    name
        Array name, for the error message.

    Returns
    -------
    tuple[int, list of float]
        The line index just past the values, and the values.

    Raises
    ------
    CodecError
        If the file ends before ``count`` values have been read, or a line
        holds something that is not a number - an array that declares more
        values than it lists runs into the header of the next one, and
        ``float()`` answering that with a bare ``ValueError`` names neither
        the array nor the file.
    """
    n_lines = len(lines)
    vals: list[float] = []
    while len(vals) < count:
        if i >= n_lines:
            raise CodecError(
                f".vtk: array {name!r} declares {count} values but the file ends"
                f" after {len(vals)}."
            )
        try:
            # Parsed whole before it is kept: extending from a generator
            # leaves the good half of a bad line in ``vals``, which the
            # message below would then count as read.
            row = list(map(float, lines[i].split()))
        except ValueError as exc:
            raise CodecError(
                f".vtk: array {name!r} declares {count} values but line"
                f" {i + 1} holds {lines[i].strip()!r}, which is not a row of"
                f" numbers; {len(vals)} were read before it."
            ) from exc
        vals.extend(row)
        i += 1
    return i, vals


def _read_ascii_tokens(
    lines: list[str], i: int, count: int, *, name: str
) -> tuple[int, list[str]]:
    """Read ``count`` unparsed tokens from an ASCII section.

    The twin of :func:`_read_ascii_values` for an array that keeps the type
    its header declared. Handing every token to ``float`` first rounds it to
    a double before the declared type ever sees it, which a ``long`` past
    2**53 does not survive.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index of the first line of values.
    count
        How many values the section declares.
    name
        Array name, for the error message.

    Returns
    -------
    tuple[int, list of str]
        The line index just past the values, and the tokens.

    Raises
    ------
    CodecError
        If the file ends before ``count`` tokens have been read.
    """
    n_lines = len(lines)
    tokens: list[str] = []
    while len(tokens) < count:
        if i >= n_lines:
            raise CodecError(
                f".vtk: array {name!r} declares {count} values but the file ends"
                f" after {len(tokens)}."
            )
        tokens.extend(lines[i].split())
        i += 1
    return i, tokens[:count]


def _field_values(tokens: list[str], np_str: str, name: str) -> np.ndarray:
    """Read a field array's tokens as the type its header declared.

    Parameters
    ----------
    tokens
        The array's whitespace-separated values.
    np_str
        The numpy dtype the declared type maps to.
    name
        Array name, for the error message.

    Returns
    -------
    numpy.ndarray
        The values, in ``np_str``.

    Raises
    ------
    CodecError
        If a token names no number at all. numpy answers that with a bare
        ``ValueError`` about one token, which names neither the array nor
        the file it came out of.
    """
    try:
        return np.array(tokens, dtype=np_str)
    except (ValueError, OverflowError):
        # An integer array spelled with a decimal point, or a value the
        # declared type is too narrow to hold. Neither is worth refusing a
        # whole file over: the tokens are read as doubles and narrowed the
        # way the C reader this file was written for would narrow them.
        try:
            return np.array(tokens, dtype=np.float64).astype(np_str)
        except ValueError as exc:
            raise CodecError(
                f".vtk: field array {name!r} holds a value that is not a"
                f" number ({exc})."
            ) from exc


def _attr_name(parts: list[str], where: str) -> str:
    """Name the array an attribute header declares.

    Parameters
    ----------
    parts
        The header line's whitespace-separated tokens, keyword included.
    where
        Where the header sits, for the error message.

    Returns
    -------
    str
        The array name.

    Raises
    ------
    CodecError
        If the header carries no name. Reading it as ``parts[1]`` answers
        that with an IndexError naming neither the file nor the line.
    """
    if len(parts) < 2:
        raise CodecError(
            f".vtk: {where} holds {' '.join(parts)!r}, a {parts[0]} header"
            " with no array name."
        )
    return parts[1]


def _attr_count(
    parts: list[str], index: int, where: str, *, default: int | None = None
) -> int:
    """Read a count off an attribute header.

    Parameters
    ----------
    parts
        The header line's whitespace-separated tokens, keyword included.
    index
        Which field holds the count.
    where
        Where the header sits, for the error message.
    default
        What the count is when the header leaves the field out, or None
        when the format does not let it.

    Returns
    -------
    int
        The count.

    Raises
    ------
    CodecError
        If a required field is missing, or the field names no integer -
        which ``int()`` answers with a ValueError naming neither the file
        nor the line.
    """
    if len(parts) <= index:
        if default is None:
            raise CodecError(
                f".vtk: {where} holds {' '.join(parts)!r}, a {parts[0]}"
                f" header with no field {index}."
            )
        return default
    try:
        return int(parts[index])
    except ValueError as exc:
        raise CodecError(
            f".vtk: {where} holds {' '.join(parts)!r}, whose {parts[0]}"
            f" count {parts[index]!r} is not a number."
        ) from exc


def _attr_reals(parts: list[str], where: str, count: int) -> list[float]:
    """Read a run of numbers off a header.

    ``ORIGIN`` and ``SPACING`` carry three each, and a header short of one
    or spelling it wrong answers ``float(parts[i])`` with an IndexError or a
    ValueError naming neither the file nor the line.

    Parameters
    ----------
    parts
        The header line's whitespace-separated tokens, keyword included.
    where
        Where the header sits, for the error message.
    count
        How many numbers follow the keyword.

    Returns
    -------
    list of float
        The values, in the order the header spells them.

    Raises
    ------
    CodecError
        If the header is short, or one of the fields names no number.
    """
    if len(parts) <= count:
        raise CodecError(
            f".vtk: {where} holds {' '.join(parts)!r}, a {parts[0]} header"
            f" with fewer than {count} values."
        )
    try:
        return [float(text) for text in parts[1 : count + 1]]
    except ValueError as exc:
        raise CodecError(
            f".vtk: {where} holds {' '.join(parts)!r}, whose {parts[0]}"
            " values are not all numbers."
        ) from exc


def _binary_dtype(parts: list[str], index: int, where: str) -> str:
    """Read a binary payload's numpy dtype off its header.

    A binary block is bytes until its declared type says how wide a value
    is and how to read it, so guessing at a name this reader has no
    equivalent for does not produce a short array or a wrong shape - it
    produces numbers that were never in the file. Refusing is the only
    answer that says what happened.

    Parameters
    ----------
    parts
        The header line's whitespace-separated tokens, keyword included.
    index
        Which field names the type.
    where
        Where the header sits, for the error message.

    Returns
    -------
    str
        Big-endian numpy dtype string, which is what a legacy binary file
        holds whatever the machine writing it was.

    Raises
    ------
    CodecError
        If the field names a type this reader has no numpy equivalent for.
    """
    name = parts[index].lower() if len(parts) > index else _DEFAULT_VTK_DTYPE
    base = _VTK_DTYPE_MAP.get(name)
    if base is None:
        raise CodecError(
            f".vtk: {where} holds {' '.join(parts)!r}, whose data type"
            f" {name!r} is not one this reader can read as numbers."
        )
    return ">" + base


def _parse_vtk_data_attrs(
    lines: list[str], i: int, n_declared: int, n_items: int, *, kind: str = "point"
) -> tuple[int, dict[str, np.ndarray]]:
    """Parse POINT_DATA or CELL_DATA attribute sections.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index of the first line after the section header.
    n_declared
        Tuples the section header says its arrays hold. This is what they
        are read as: it is the only number that says where one array ends
        and the next begins, and reading by the mesh's count instead walks
        an array that disagrees straight into the header of the one after
        it.
    n_items
        Points or cells the mesh has. An array that does not cover them
        belongs to none of them and is dropped rather than attached.
    kind
        ``'point'`` or ``'cell'``, for the warning.

    Returns
    -------
    tuple[int, dict of str to numpy.ndarray]
        The line just past the section, and the arrays that cover the mesh.
    """
    attrs: dict[str, np.ndarray] = {}
    unknown: set[str] = set()
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        upper = line.upper()

        # Stop at next top-level section
        if upper.startswith(_ATTRS_STOP_KEYWORDS):
            break

        if upper.startswith("METADATA"):
            # Every VTK writer since 4.2 puts one of these after each array.
            # It is component names and information keys, not values, and
            # the block ends at a blank line.
            i = _skip_metadata(lines, i)

        elif upper.startswith("SCALARS"):
            parts = line.split()
            where = f"line {i + 1}"
            name = _attr_name(parts, where)
            n_comp = _attr_count(parts, 3, where, default=1)
            i += 1
            # Skip LOOKUP_TABLE line
            if i < n_lines and "LOOKUP_TABLE" in lines[i].upper():
                i += 1
            i, vals = _read_ascii_values(lines, i, n_declared * n_comp, name=name)
            arr = np.array(vals, dtype=np.float64)
            attrs[name] = arr.reshape(n_declared, n_comp) if n_comp > 1 else arr

        elif upper.startswith("COLOR_SCALARS"):
            # An ASCII COLOR_SCALARS line holds one float per component in
            # 0..1; the binary flavour holds one unsigned char instead, which
            # is why this cannot share the SCALARS branch.
            parts = line.split()
            where = f"line {i + 1}"
            name = _attr_name(parts, where)
            n_comp = _attr_count(parts, 2, where, default=1)
            i += 1
            i, vals = _read_ascii_values(lines, i, n_declared * n_comp, name=name)
            arr = np.array(vals, dtype=np.float64)
            attrs[name] = arr.reshape(n_declared, n_comp) if n_comp > 1 else arr

        elif upper.startswith("VECTORS") or upper.startswith("NORMALS"):
            parts = line.split()
            name = _attr_name(parts, f"line {i + 1}")
            i += 1
            i, vals = _read_ascii_values(lines, i, n_declared * 3, name=name)
            attrs[name] = np.array(vals, dtype=np.float64).reshape(n_declared, 3)

        elif upper.startswith("TEXTURE_COORDINATES"):
            parts = line.split()
            where = f"line {i + 1}"
            name = _attr_name(parts, where)
            n_comp = _attr_count(parts, 2, where, default=2)
            i += 1
            i, vals = _read_ascii_values(lines, i, n_declared * n_comp, name=name)
            arr = np.array(vals, dtype=np.float64)
            attrs[name] = arr.reshape(n_declared, n_comp) if n_comp > 1 else arr

        elif upper.startswith("LOOKUP_TABLE") and len(line.split()) > 2:
            # A table definition, not the 'LOOKUP_TABLE name' line a SCALARS
            # section carries - that one is consumed above. It is a palette
            # rather than a value per point or cell, so there is no array to
            # hang it on; its rgba rows are counted past so the arrays after
            # it are still found.
            parts = line.split()
            where = f"line {i + 1}"
            i, _ = _read_ascii_values(
                lines,
                i + 1,
                _attr_count(parts, 2, where) * 4,
                name=_attr_name(parts, where),
            )

        elif upper.startswith("TENSORS"):
            parts = line.split()
            name = _attr_name(parts, f"line {i + 1}")
            i += 1
            i, vals = _read_ascii_values(lines, i, n_declared * 9, name=name)
            attrs[name] = np.array(vals, dtype=np.float64).reshape(n_declared, 3, 3)

        elif upper.startswith("FIELD"):
            parts = line.split()
            n_arrays = _attr_count(parts, 2, f"line {i + 1}")
            i += 1
            for _ in range(n_arrays):
                i = _next_header(lines, i)
                if i >= n_lines:
                    raise CodecError(
                        f".vtk: FIELD declares {n_arrays} arrays but the file"
                        " ends before their headers."
                    )
                fparts = lines[i].strip().split()
                fwhere = f"line {i + 1}"
                fname = fparts[0]
                n_comp_f = _attr_count(fparts, 1, fwhere)
                n_tuples = _attr_count(fparts, 2, fwhere)
                i += 1
                i, vals = _read_ascii_values(lines, i, n_tuples * n_comp_f, name=fname)
                arr = np.array(vals, dtype=np.float64)
                attrs[fname] = arr.reshape(n_tuples, n_comp_f) if n_comp_f > 1 else arr

        else:
            # A keyword no version defines. Text can be stepped over a line
            # at a time, so the scan goes on and the arrays after it are
            # still found - but how many lines this one's values fill is not
            # knowable, so they are skipped too, and the array is gone.
            # Saying so once per keyword is the difference between a short
            # read and a silent one.
            _warn_unhandled_attr(line, unknown)
            i += 1

    return i, _keep_sized_attrs(attrs, expected=n_items, kind=kind, stacklevel=6)


def _warn_unhandled_attr(line: str, seen: set[str], *, stacklevel: int = 6) -> None:
    """Say once that a data keyword with no branch here is being dropped.

    The readers walk their file a line at a time and skip what they do not
    recognise. In a binary file that skip does not step over the keyword's
    payload, so the scan carries on inside it and can read the bytes as
    further keywords - a dropped array is the good outcome. Either way the
    file held something the mesh does not, which is worth a sentence.

    Parameters
    ----------
    line
        The unhandled line, stripped.
    seen
        Keywords already warned about in this file; added to in place, so
        an array wrapped over many lines is named once rather than per line.
    stacklevel
        Frames between here and the caller of ``read``, so the warning is
        blamed on the code that asked for the file rather than on this
        module. The default suits both scans, which sit the same depth
        below ``read``.
    """
    tokens = line.split()
    keyword = tokens[0] if tokens else ""
    if keyword in _SECTION_KEYWORDS or keyword in seen:
        return
    if not _looks_like_keyword(keyword):
        return
    seen.add(keyword)
    warnings.warn(
        f".vtk: attribute keyword {keyword!r} is not one this reader knows;"
        " it and its values are dropped.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _next_header(lines: list[str], i: int) -> int:
    """Find the next header, stepping past blank lines and METADATA blocks.

    An array carries its own ``METADATA`` block, which sits between it and
    whatever comes next - the following array inside a FIELD, or the
    ``CONNECTIVITY`` keyword inside a v5.1 ``CELLS``. Read as data it is a
    line of words where numbers belong, and read as a header it is a name
    with no component count.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index to start looking from.

    Returns
    -------
    int
        Index of the next header line, or past the end when there is none.
    """
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
        elif line.upper().startswith("METADATA"):
            i = _skip_metadata(lines, i)
        else:
            break
    return i


def _skip_metadata(lines: list[str], i: int) -> int:
    """Step past a ``METADATA`` block.

    Every VTK writer since 4.2 follows an array with one of these: component
    names and information keys, terminated by a blank line. It holds no
    values, so there is nothing to keep - but it has to be stepped over, or
    the keywords inside it read as arrays this reader does not know and the
    scan says so about a block that is perfectly ordinary.

    Parameters
    ----------
    lines
        The file's lines.
    i
        Index of the ``METADATA`` line itself.

    Returns
    -------
    int
        The line just past the block: past its blank terminator, or at the
        keyword that ended it when the terminator is missing, so a malformed
        block costs its own array rather than every array after it.
    """
    n_lines = len(lines)
    i += 1
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            return i + 1
        keyword = line.split()[0].upper()
        if keyword in _METADATA_STOP_KEYWORDS:
            return i
        i += 1
    return i


def _looks_like_keyword(token: str) -> bool:
    """Tell a section keyword from a line of values it wraps onto.

    An unknown keyword's values are skipped by the same branch that skipped
    the keyword, so warning on every line would name the numbers as often as
    the array. A legacy keyword is upper case letters and underscores.

    Parameters
    ----------
    token
        First whitespace-separated token of the line.

    Returns
    -------
    bool
        True when the token reads as a keyword rather than as data.
    """
    return token.replace("_", "").isalpha() and token.isupper()


def _read_binary(path: Source, *, lazy: bool) -> PolyData:
    """Read binary VTK file, using mmap (lazy) or direct reads."""
    # A lazy read hands back arrays that view the block, so it has to be a
    # mapping: reading a file object into memory would answer the call with
    # the copy the caller asked not to make.
    with open_block(path, fmt=".vtk", require_map=lazy) as mm:
        file_size = len(mm)
        mv = memoryview(mm)

        # Skip 4 ASCII header lines to reach the data sections
        pos = 0
        for _ in range(4):
            pos = mm.find(b"\n", pos) + 1

        poly = _parse_binary_body(mm, mv, pos, file_size)
        del mv  # release the view before the mapping goes
        if not lazy:
            poly = _materialize(poly)

    return poly


def _materialize(poly: PolyData) -> PolyData:
    """Convert mmap-backed arrays to in-memory copies."""
    import dataclasses

    return dataclasses.replace(
        poly,
        vertices=np.array(poly.vertices),
        connectivity=np.array(poly.connectivity),
        offsets=np.array(poly.offsets),
        element_types=np.array(poly.element_types),
        vertex_attrs={k: np.array(v) for k, v in poly.vertex_attrs.items()},
        element_attrs={k: np.array(v) for k, v in poly.element_attrs.items()},
    )


def _parse_binary_body(
    mm: mmap.mmap | bytes,
    mv: memoryview,
    start_pos: int,
    file_size: int,
) -> PolyData:
    """Parse binary data sections from an mmap object."""
    pos = start_pos
    vertices = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.array([], dtype=np.int32)
    offsets_arr = np.array([0], dtype=np.int32)
    element_types_arr = np.array([], dtype=np.uint8)
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    global_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while pos < file_size:
        line_start = pos
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            break
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1

        if not line:
            continue

        upper = line.upper()

        if upper.startswith("FIELD"):
            # A FIELD block out here belongs to the dataset rather than to
            # its points or cells - the one inside a POINT_DATA section is
            # read by that section's own parser, which never returns here.
            n_arrays = _attr_count(line.split(), 2, f"byte {line_start}")
            pos, global_attrs = _parse_field_block_binary(
                mm, mv, pos, file_size, n_arrays
            )

        elif upper.startswith("POINTS"):
            parts = line.split()
            n_verts = _attr_count(parts, 1, f"byte {line_start}")
            # The header names the type the block holds, and it is not always
            # a float: reading a POINTS written as 'int' or 'short' at four
            # bytes a value hands back coordinates the file never held.
            np_dt = _binary_dtype(parts, 2, f"byte {line_start}")
            n_bytes = n_verts * 3 * np.dtype(np_dt).itemsize
            validate_header(n_verts, 0, 0, file_size)
            # validate_header bounds the block against the whole file, which a
            # POINTS section sitting behind a long header clears while still
            # running off the end; a short slice reshapes into a ValueError
            # that names neither the array nor the file.
            _check_block(pos, n_bytes, file_size, name=parts[0])
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt)
            vertices = raw.astype(np.float64).reshape(n_verts, 3)
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)

        elif upper.startswith("CELLS") and not upper.startswith("CELL_TYPES"):
            parts = line.split()
            where = f"byte {line_start}"
            n_elems = _attr_count(parts, 1, where)
            total_size = _attr_count(parts, 2, where)
            validate_header(n_verts, n_elems, total_size, file_size)

            line_end2 = mm.find(b"\n", pos)
            next_line = (
                bytes(mv[pos:line_end2])
                .decode("ascii", errors="replace")
                .strip()
                .upper()
            )

            if "OFFSETS" in next_line:
                connectivity, offsets_arr, pos = _parse_v51_cells_binary(
                    mm, mv, line_end2 + 1, n_elems, file_size
                )
                n_elems = len(offsets_arr) - 1
            else:
                # v4.2: interleaved [count, idx0, ...] int32
                n_bytes_cells = total_size * 4
                _check_block(pos, n_bytes_cells, file_size, name="CELLS")
                raw = np.frombuffer(
                    bytes(mv[pos : pos + n_bytes_cells]), dtype=">i4"
                ).astype(np.int32)
                pos += n_bytes_cells
                pos = _skip_newline(mv, pos, file_size)
                connectivity, offsets_arr = _unpack_v42_cells(raw, n_elems)

        elif upper.startswith("CELL_TYPES"):
            n_ct = _attr_count(line.split(), 1, f"byte {line_start}")
            n_bytes_ct = n_ct * 4
            _check_block(pos, n_bytes_ct, file_size, name="CELL_TYPES")
            raw_ct = np.frombuffer(
                bytes(mv[pos : pos + n_bytes_ct]), dtype=">i4"
            ).astype(np.int32)
            pos += n_bytes_ct
            pos = _skip_newline(mv, pos, file_size)
            type_codes: list[int] = []
            for vtk_code in raw_ct:
                vtk_code_int = int(vtk_code)
                if vtk_code_int not in VTK_TO_POLYXIOS:
                    raise UnknownElementTypeError("vtk", vtk_code_int)
                type_codes.append(ELEMENT_TYPES[VTK_TO_POLYXIOS[vtk_code_int]])
            element_types_arr = np.array(type_codes, dtype=np.uint8)

        elif upper.startswith("POINT_DATA"):
            n_pd = _attr_count(line.split(), 1, f"byte {line_start}")
            pos, vertex_attrs = _parse_binary_attrs(
                mm, mv, pos, n_pd, file_size, expected=n_verts, kind="point"
            )

        elif upper.startswith("CELL_DATA"):
            n_cd = _attr_count(line.split(), 1, f"byte {line_start}")
            pos, element_attrs = _parse_binary_attrs(
                mm, mv, pos, n_cd, file_size, expected=n_elems, kind="cell"
            )

    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _skip_newline(mv: memoryview, pos: int, file_size: int) -> int:
    """Skip a trailing newline byte if present."""
    if pos < file_size and bytes(mv[pos : pos + 1]) == b"\n":
        return pos + 1
    return pos


def _parse_v51_cells_binary(
    mm: mmap.mmap | bytes, mv: memoryview, pos: int, declared: int, file_size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read a binary v5.1 OFFSETS + CONNECTIVITY pair.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset just past the ``OFFSETS`` keyword line.
    declared
        The first number on the section's header line.
    file_size
        Size of the file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, int]
        Connectivity, offsets, and the byte offset just past the section.
        The section holds one cell fewer than the offsets array has values.
    """
    n_off = _v51_offset_count(mm, mv, pos, declared, file_size)
    n_bytes_off = n_off * 8
    _check_block(pos, n_bytes_off, file_size, name="OFFSETS")
    offsets_arr = np.frombuffer(bytes(mv[pos : pos + n_bytes_off]), dtype=">i8").astype(
        np.int64
    )
    pos = _next_binary_header(
        mm, mv, _skip_newline(mv, pos + n_bytes_off, file_size), file_size
    )

    # skip CONNECTIVITY keyword line
    conn_kw_end = mm.find(b"\n", pos)
    if conn_kw_end == -1:
        raise CodecError(
            ".vtk: a v5.1 cell section declares OFFSETS but no CONNECTIVITY"
            " follows them."
        )
    pos = conn_kw_end + 1
    n_bytes_conn = int(offsets_arr[-1]) * 8
    _check_block(pos, n_bytes_conn, file_size, name="CONNECTIVITY")
    connectivity = np.frombuffer(
        bytes(mv[pos : pos + n_bytes_conn]), dtype=">i8"
    ).astype(np.int64)
    pos = _skip_newline(mv, pos + n_bytes_conn, file_size)
    return connectivity, offsets_arr, pos


def _v51_offset_count(
    mm: "mmap.mmap | bytes", mv: memoryview, pos: int, declared: int, file_size: int
) -> int:
    """Count the int64 offsets a binary v5.1 CELLS section holds.

    The first number on the ``CELLS`` line is the length of the OFFSETS
    array, but polyxios wrote the cell count there until this release, so
    the two spellings differ by one and a file gives no other sign of which
    it used. The block is followed by the ``CONNECTIVITY`` keyword, so the
    length that puts that keyword where it belongs is the length the file
    meant.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset of the first offset value.
    declared
        The first number on the ``CELLS`` line.
    file_size
        Size of the file.

    Returns
    -------
    int
        Values in the OFFSETS array. The spelling VTK writes is tried first,
        so a file that fits both is read the way VTK would read it.

    Raises
    ------
    CodecError
        If neither length is followed by a ``CONNECTIVITY`` keyword.
    """
    for n_off in (declared, declared + 1):
        if n_off < 1:
            continue
        end = pos + n_off * 8
        if end > file_size:
            continue
        probe = _next_binary_header(
            mm, mv, _skip_newline(mv, end, file_size), file_size
        )
        line_end = mm.find(b"\n", probe)
        if line_end == -1:
            line_end = file_size
        keyword = bytes(mv[probe:line_end]).decode("ascii", errors="replace").strip()
        if keyword.upper().startswith("CONNECTIVITY"):
            return n_off
    raise CodecError(
        f".vtk: a v5.1 CELLS section declares {declared} but no CONNECTIVITY"
        " keyword follows either the offsets it names or one more."
    )


def _unpack_v42_cells(raw: np.ndarray, n_elems: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert v4.2 interleaved cell array to CSR connectivity + offsets."""
    conn_list: list[int] = []
    off_list: list[int] = [0]
    idx = 0
    for _ in range(n_elems):
        cnt = int(raw[idx])
        idx += 1
        conn_list.extend(int(raw[idx + j]) for j in range(cnt))
        idx += cnt
        off_list.append(off_list[-1] + cnt)
    return np.array(conn_list, dtype=np.int32), np.array(off_list, dtype=np.int32)


def _check_block(pos: int, n_bytes: int, file_size: int, *, name: str) -> None:
    """Refuse a binary block the file is too short to hold.

    A slice past the end of a mapping is silently short, and the reshape
    that follows fails with a ValueError naming neither the array nor the
    file. Asking first turns a truncated file into a sentence about it.

    Parameters
    ----------
    pos
        Byte offset the block starts at.
    n_bytes
        Bytes the header says the block holds.
    file_size
        Size of the file.
    name
        Array name, for the error message.

    Raises
    ------
    CodecError
        If the block runs past the end of the file.
    """
    if pos + n_bytes > file_size:
        raise CodecError(
            f".vtk: array {name!r} declares {n_bytes} bytes but only"
            f" {max(0, file_size - pos)} remain in the file."
        )


def _skip_binary_metadata(
    mm: mmap.mmap | bytes, mv: memoryview, pos: int, file_size: int
) -> int:
    """Step past a ``METADATA`` block in a binary file.

    The block is text even here, so it is read the way the ASCII reader
    reads it: line by line to the blank one that ends it. Left alone it
    would end the attribute scan and take every array after it.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset just past the ``METADATA`` line.
    file_size
        Size of the file.

    Returns
    -------
    int
        Byte offset just past the block: past its blank terminator, or at
        the keyword that ended it when the terminator is missing.
    """
    while pos < file_size:
        line_start = pos
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            return file_size
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1
        if not line:
            return pos
        keyword = line.split()[0].upper()
        if keyword in _METADATA_STOP_KEYWORDS:
            return line_start
    return pos


def _next_binary_header(
    mm: "mmap.mmap | bytes", mv: memoryview, pos: int, file_size: int
) -> int:
    """Step past blank lines and any METADATA block before the next header.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset to start looking from.
    file_size
        Size of the file.

    Returns
    -------
    int
        Byte offset of the next array header.
    """
    while pos < file_size:
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            return pos
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        if not line:
            pos = line_end + 1
        elif line.upper().startswith("METADATA"):
            pos = _skip_binary_metadata(mm, mv, line_end + 1, file_size)
        else:
            return pos
    return pos


def _parse_binary_attrs(
    mm: mmap.mmap | bytes,
    mv: memoryview,
    pos: int,
    n_items: int,
    file_size: int,
    *,
    expected: int,
    kind: str,
) -> tuple[int, dict[str, np.ndarray]]:
    """Parse binary POINT_DATA or CELL_DATA attribute sections.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset of the first line after the section header.
    n_items
        Tuples the section header says its arrays hold. This is what they
        are read as: in a binary file it is the only thing that says where
        one array's payload ends and the next array's header begins.
    file_size
        Size of the file.
    expected
        Points or cells the mesh has. An array that does not cover them
        belongs to none of them and is dropped rather than attached.
    kind
        ``'point'`` or ``'cell'``, for the warning.

    Returns
    -------
    tuple[int, dict of str to numpy.ndarray]
        The byte offset just past the section, and the arrays that cover
        the mesh.
    """
    attrs: dict[str, np.ndarray] = {}

    while pos < file_size:
        line_start = pos
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            break
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1

        if not line:
            continue

        upper = line.upper()
        where = f"byte {line_start}"
        if upper.startswith(_ATTRS_STOP_KEYWORDS):
            pos = line_start  # back up so outer loop re-reads this line
            break

        if upper.startswith("METADATA"):
            # Written after every array by VTK 4.2 and later, and written as
            # text even in a binary file. Left to the branch below it would
            # end the scan and take every array after it with it.
            pos = _skip_binary_metadata(mm, mv, pos, file_size)

        elif upper.startswith("SCALARS"):
            parts = line.split()
            name = _attr_name(parts, where)
            n_comp = _attr_count(parts, 3, where, default=1)
            np_dt = _binary_dtype(parts, 2, where)
            # Skip the LOOKUP_TABLE line, if the section carries one: the
            # format leaves it optional, and swallowing a line that is not
            # there swallows binary payload up to its first 0x0a byte -
            # or, when the payload holds none, rewinds to the top of the
            # file on the -1 that find answers with.
            pos = _skip_lookup_table(mm, mv, pos, file_size)
            n_bytes = n_items * n_comp * np.dtype(np_dt).itemsize
            _check_block(pos, n_bytes, file_size, name=name)
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, n_comp) if n_comp > 1 else raw

        elif upper.startswith("COLOR_SCALARS"):
            # Binary COLOR_SCALARS is one unsigned char per component - the
            # only attribute section whose type is not named on its line. The
            # ASCII flavour of the same colour is a float in 0..1, so the
            # byte is scaled onto that range: the same file in its two
            # encodings must not read back as two different arrays.
            parts = line.split()
            name = _attr_name(parts, where)
            n_comp = _attr_count(parts, 2, where, default=1)
            n_bytes = n_items * n_comp
            _check_block(pos, n_bytes, file_size, name=name)
            raw = (
                np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np.uint8).astype(
                    np.float64
                )
                / _COLOR_SCALE
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, n_comp) if n_comp > 1 else raw

        elif upper.startswith("VECTORS") or upper.startswith("NORMALS"):
            parts = line.split()
            name = _attr_name(parts, where)
            np_dt = _binary_dtype(parts, 2, where)
            n_bytes = n_items * 3 * np.dtype(np_dt).itemsize
            _check_block(pos, n_bytes, file_size, name=name)
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, 3)

        elif upper.startswith("TEXTURE_COORDINATES"):
            parts = line.split()
            name = _attr_name(parts, where)
            n_comp = _attr_count(parts, 2, where, default=2)
            np_dt = _binary_dtype(parts, 3, where)
            n_bytes = n_items * n_comp * np.dtype(np_dt).itemsize
            _check_block(pos, n_bytes, file_size, name=name)
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, n_comp) if n_comp > 1 else raw

        elif upper.startswith("LOOKUP_TABLE") and len(line.split()) > 2:
            # A table definition, not the 'LOOKUP_TABLE name' line a SCALARS
            # section carries - that one is consumed above. It is a palette
            # rather than a value per point or cell, so there is no array to
            # hang it on; its rgba bytes are stepped over so the arrays after
            # it are still found.
            parts = line.split()
            n_bytes = _attr_count(parts, 2, where) * 4
            _check_block(pos, n_bytes, file_size, name=_attr_name(parts, where))
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)

        elif upper.startswith("TENSORS"):
            parts = line.split()
            name = _attr_name(parts, where)
            np_dt = _binary_dtype(parts, 2, where)
            n_bytes = n_items * 9 * np.dtype(np_dt).itemsize
            _check_block(pos, n_bytes, file_size, name=name)
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, 3, 3)

        elif upper.startswith("FIELD"):
            n_arrays = _attr_count(line.split(), 2, where)
            for _ in range(n_arrays):
                # A FIELD array carries its own METADATA block, which sits
                # between one array and the next; read as a header it eats
                # the array after it.
                pos = _next_binary_header(mm, mv, pos, file_size)
                hdr_start = pos
                hdr_end = mm.find(b"\n", pos)
                if hdr_end == -1:
                    break
                hdr = bytes(mv[pos:hdr_end]).decode("ascii", errors="replace").strip()
                pos = hdr_end + 1
                hparts = hdr.split()
                if len(hparts) < 4:
                    continue
                # Where the header sits, not where its payload does: taken
                # after the step past it, the message would point a reader
                # at the bytes rather than at the line that is wrong.
                fwhere = f"byte {hdr_start}"
                arr_name = hparts[0]
                n_comp_f = _attr_count(hparts, 1, fwhere)
                n_tuples_f = _attr_count(hparts, 2, fwhere)
                np_dt_f = _binary_dtype(hparts, 3, fwhere)
                n_bytes_f = n_tuples_f * n_comp_f * np.dtype(np_dt_f).itemsize
                _check_block(pos, n_bytes_f, file_size, name=arr_name)
                raw_f = np.frombuffer(
                    bytes(mv[pos : pos + n_bytes_f]), dtype=np_dt_f
                ).astype(np.float64)
                pos += n_bytes_f
                pos = _skip_newline(mv, pos, file_size)
                attrs[arr_name] = (
                    raw_f.reshape(n_tuples_f, n_comp_f) if n_comp_f > 1 else raw_f
                )

        else:
            # Every legacy attribute keyword is handled above, so this is one
            # no version defines. Its payload is binary of unknown length, so
            # there is no stepping over it: the scan ends here, and what the
            # file holds after it is lost. Saying so is the difference
            # between a short read and a silent one.
            warnings.warn(
                f".vtk: attribute keyword {line.split()[0]!r} is not one this"
                " reader knows; it and everything after it in this data"
                " section are dropped.",
                UserWarning,
                stacklevel=6,
            )
            pos = line_start  # back up and let the outer loop see the line
            break

    return pos, _keep_sized_attrs(attrs, expected=expected, kind=kind, stacklevel=7)


def _skip_lookup_table(
    mm: "mmap.mmap | bytes", mv: memoryview, pos: int, file_size: int
) -> int:
    """Step past the ``LOOKUP_TABLE`` line a binary SCALARS section may carry.

    Parameters
    ----------
    mm
        The mapping or buffer, for finding line ends.
    mv
        A memoryview of the same bytes, for slicing.
    pos
        Byte offset just past the ``SCALARS`` line.
    file_size
        Size of the file.

    Returns
    -------
    int
        Byte offset of the payload: past the ``LOOKUP_TABLE`` line when
        there is one, and ``pos`` unchanged when there is not.
    """
    line_end = mm.find(b"\n", pos)
    if line_end == -1:
        return pos
    # A LOOKUP_TABLE line is short; reading only that much keeps this from
    # decoding a whole payload's worth of bytes to answer a yes or no.
    head = bytes(mv[pos : min(line_end, pos + 16)])
    if head.upper().startswith(b"LOOKUP_TABLE"):
        return min(line_end + 1, file_size)
    return pos


def _lines_with_offsets(raw: bytes) -> tuple[list[str], list[int]]:
    """Cut a file into stripped lines, and say where each one starts.

    The three structured readers walk their file themselves, so they need
    the byte offset of every line to find a binary payload. Splitting on the
    newline in one pass is what a ``find`` per line was doing four calls at
    a time, and a binary payload carries a newline every few values, so a
    grid of any size is cut into lines by the million.

    Parameters
    ----------
    raw
        The whole file.

    Returns
    -------
    tuple[list of str, list of int]
        The lines, stripped, and the byte offset each one starts at. A
        trailing newline ends the last line rather than opening an empty
        one, which is what a walk stopping at the end of the file did.
    """
    if not raw:
        return [], []
    chunks = raw.split(b"\n")
    if chunks[-1] == b"" and raw.endswith(b"\n"):
        chunks.pop()
    texts = [chunk.decode("ascii", errors="replace").strip() for chunk in chunks]
    offsets = list(accumulate((len(chunk) + 1 for chunk in chunks), initial=0))[:-1]
    return texts, offsets


def _read_rectilinear_grid(path: Source, *, is_binary: bool) -> PolyData:
    """Read a VTK legacy RECTILINEAR_GRID dataset (ASCII or binary).

    Parameters
    ----------
    path
        Path to the .vtk file.
    is_binary
        True when the file header says BINARY.

    Returns
    -------
    PolyData
        Mesh with meshgrid vertices from X/Y/Z_COORDINATES and generated
        hex/quad/line connectivity from DIMENSIONS.
    """
    raw = read_bytes(path)

    texts, line_offsets = _lines_with_offsets(raw)

    nx, ny, nz = 1, 1, 1
    xs: np.ndarray = np.zeros(1)
    ys: np.ndarray = np.zeros(1)
    zs: np.ndarray = np.zeros(1)
    n_points = 0
    in_point_data = False
    # POINT_DATA and CELL_DATA both open a section of attributes; the
    # flag is what tells an unhandled keyword inside one from the
    # header lines above, which are skipped on purpose.
    in_data_section = False
    unhandled: set[str] = set()
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    # What a CELL_DATA section says it holds, against what the grid the
    # header describes actually has; a section is read with the first and
    # kept only when it matches the second.
    n_cells_declared = 0

    i = 0
    n_lines = len(texts)

    while i < n_lines:
        line = texts[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("DIMENSIONS"):
            where = f"line {i + 1}"
            nx = _attr_count(parts, 1, where)
            ny = _attr_count(parts, 2, where)
            nz = _attr_count(parts, 3, where)
        elif upper.startswith("X_COORDINATES"):
            where = f"line {i + 1}"
            n_coord = _attr_count(parts, 1, where)
            i += 1
            if is_binary:
                np_dt = _binary_dtype(parts, 2, where)
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_coord * np.dtype(np_dt).itemsize
                _check_block(data_pos, n_bytes, len(raw), name=parts[0])
                xs = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                i = _skip_payload(line_offsets, i, n_lines, data_pos + n_bytes)
                continue
            else:
                vals: list[float] = []
                while len(vals) < n_coord and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                xs = np.array(vals, dtype=np.float64)
                continue
        elif upper.startswith("Y_COORDINATES"):
            where = f"line {i + 1}"
            n_coord = _attr_count(parts, 1, where)
            i += 1
            if is_binary:
                np_dt = _binary_dtype(parts, 2, where)
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_coord * np.dtype(np_dt).itemsize
                _check_block(data_pos, n_bytes, len(raw), name=parts[0])
                ys = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                i = _skip_payload(line_offsets, i, n_lines, data_pos + n_bytes)
                continue
            else:
                vals = []
                while len(vals) < n_coord and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                ys = np.array(vals, dtype=np.float64)
                continue
        elif upper.startswith("Z_COORDINATES"):
            where = f"line {i + 1}"
            n_coord = _attr_count(parts, 1, where)
            i += 1
            if is_binary:
                np_dt = _binary_dtype(parts, 2, where)
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_coord * np.dtype(np_dt).itemsize
                _check_block(data_pos, n_bytes, len(raw), name=parts[0])
                zs = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                i = _skip_payload(line_offsets, i, n_lines, data_pos + n_bytes)
                continue
            else:
                vals = []
                while len(vals) < n_coord and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                zs = np.array(vals, dtype=np.float64)
                continue
        elif upper.startswith("POINT_DATA"):
            n_points = _attr_count(parts, 1, f"line {i + 1}")
            in_point_data = True
            in_data_section = True
        elif upper.startswith("CELL_DATA"):
            n_cells_declared = _attr_count(parts, 1, f"line {i + 1}", default=0)
            in_point_data = False
            in_data_section = True
        elif in_data_section:
            # The coordinate arrays are the grid, not what DIMENSIONS said
            # about it: the points are their outer product, so a header that
            # disagrees would drop an attribute that covers every point
            # there is.
            gx, gy, gz = len(xs), len(ys), len(zs)
            i = _read_structured_attr(
                texts,
                line_offsets,
                raw,
                i,
                n_items=n_points if in_point_data else n_cells_declared,
                is_binary=is_binary,
                attrs=vertex_attrs if in_point_data else element_attrs,
                expected=(
                    gx * gy * gz
                    if in_point_data
                    else _structured_cell_count(gx, gy, gz)
                ),
                kind="point" if in_point_data else "cell",
                unhandled=unhandled,
            )
            continue
        i += 1

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # As above: the cells have to index the points that exist, so they are
    # generated over the coordinate arrays. A file whose header disagrees
    # with them says two things, and only one of them is the data.
    if (nx, ny, nz) != (len(xs), len(ys), len(zs)):
        warnings.warn(
            f".vtk: DIMENSIONS says {nx} {ny} {nz} but the coordinate arrays"
            f" hold {len(xs)} {len(ys)} {len(zs)}; the arrays are the grid"
            " and the header is ignored.",
            UserWarning,
            stacklevel=4,
        )
    nx, ny, nz = len(xs), len(ys), len(zs)

    # The grid is not recoverable from the point array it was expanded into,
    # so it is kept alongside it - the grid the points are actually laid out
    # on, which above is the coordinate arrays rather than the header they
    # may disagree with. A consumer rebuilding the image from a number the
    # points do not honour rebuilds a different mesh.
    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    grid_meta: dict[str, object] = {"vtk_dimensions": [nx, ny, nz]}

    cells, etype_name = _structured_grid_cells(nx, ny, nz)
    if len(cells) == 0:
        return PolyData(
            vertices=vertices,
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            vertex_attrs=vertex_attrs,
            element_attrs=element_attrs,
            vertex_tags=vertex_tags,
            element_tags=element_tags,
            global_attrs=grid_meta,
        )

    n_cells = len(cells)
    npc = cells.shape[1]
    connectivity = cells.ravel().astype(np.int32)
    offsets_arr = np.arange(0, (n_cells + 1) * npc, npc, dtype=np.int32)
    element_types_arr = np.full(n_cells, ELEMENT_TYPES[etype_name], dtype=np.uint8)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=grid_meta,
    )


def _read_structured_grid(path: Source, *, is_binary: bool) -> PolyData:
    """Read a VTK legacy STRUCTURED_GRID dataset (ASCII or binary).

    Parameters
    ----------
    path
        Path to the .vtk file.
    is_binary
        True when the file header says BINARY.

    Returns
    -------
    PolyData
        Mesh with explicit point coordinates and generated hex/quad/line
        connectivity from the DIMENSIONS keyword.
    """
    raw = read_bytes(path)

    texts, offsets = _lines_with_offsets(raw)

    nx, ny, nz = 1, 1, 1
    n_points = 0
    vertices = np.zeros((0, 3), dtype=np.float64)
    in_point_data = False
    # POINT_DATA and CELL_DATA both open a section of attributes; the
    # flag is what tells an unhandled keyword inside one from the
    # header lines above, which are skipped on purpose.
    in_data_section = False
    unhandled: set[str] = set()
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    # What a data section says it holds, against what the grid the header
    # describes actually has; a section is read with the first and kept only
    # when it matches the second.
    n_points_declared = 0
    n_cells_declared = 0

    i = 0
    n_lines = len(texts)

    while i < n_lines:
        line = texts[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("DIMENSIONS"):
            where = f"line {i + 1}"
            nx = _attr_count(parts, 1, where)
            ny = _attr_count(parts, 2, where)
            nz = _attr_count(parts, 3, where)
        elif upper.startswith("POINTS") and not upper.startswith("POINT_DATA"):
            where = f"line {i + 1}"
            n_points = _attr_count(parts, 1, where)
            i += 1
            if is_binary:
                np_dt = _binary_dtype(parts, 2, where)
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * 3 * np.dtype(np_dt).itemsize
                _check_block(data_pos, n_bytes, len(raw), name=parts[0])
                raw_pts = np.frombuffer(raw[data_pos : data_pos + n_bytes], dtype=np_dt)
                vertices = raw_pts.astype(np.float64).reshape(n_points, 3)
                # The walk already lands on the first line that starts at
                # or after the payload; the separator newline it ends on
                # belongs to the payload's own last line, so stepping over
                # it again cost the section that followed - which is how a
                # CELL_DATA written before POINT_DATA went missing.
                i = _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
                continue
            else:
                vals: list[float] = []
                while len(vals) < n_points * 3 and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                vertices = np.array(vals, dtype=np.float64).reshape(n_points, 3)
                continue
        elif upper.startswith("POINT_DATA"):
            # The section's own count is what its arrays are as long as; the
            # POINTS array is what they have to line up with, and a file that
            # spells the two differently is read by the first and kept by the
            # second.
            n_points_declared = _attr_count(parts, 1, f"line {i + 1}", default=n_points)
            in_point_data = True
            in_data_section = True
            i += 1
            continue
        elif upper.startswith("CELL_DATA"):
            n_cells_declared = _attr_count(parts, 1, f"line {i + 1}", default=0)
            in_point_data = False
            in_data_section = True
        elif in_data_section:
            # The cells are the ones _structured_cells_over will build, not
            # the ones DIMENSIONS describes: a header the POINTS array does
            # not cover leaves the mesh with none, and a CELL_DATA array
            # kept against the count the header named is one validate then
            # refuses.
            i = _read_structured_attr(
                texts,
                offsets,
                raw,
                i,
                n_items=n_points_declared if in_point_data else n_cells_declared,
                is_binary=is_binary,
                attrs=vertex_attrs if in_point_data else element_attrs,
                expected=(
                    len(vertices)
                    if in_point_data
                    else _structured_cell_count_over(nx, ny, nz, len(vertices))
                ),
                kind="point" if in_point_data else "cell",
                unhandled=unhandled,
            )
            continue
        i += 1

    # The grid is not recoverable from the point array, so what the header
    # said is kept alongside it. A STRUCTURED_GRID lays its points out by
    # DIMENSIONS and nothing else, so the header is the only description
    # there is: it is handed back even when the POINTS array does not cover
    # it, which the cells above have already been dropped over.
    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    grid_meta: dict[str, object] = {"vtk_dimensions": [nx, ny, nz]}

    cells, etype_name = _structured_cells_over(nx, ny, nz, len(vertices))
    if len(cells) == 0:
        return PolyData(
            vertices=vertices,
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            vertex_attrs=vertex_attrs,
            element_attrs=element_attrs,
            vertex_tags=vertex_tags,
            element_tags=element_tags,
            global_attrs=grid_meta,
        )

    n_cells = len(cells)
    npc = cells.shape[1]
    connectivity = cells.ravel().astype(np.int32)
    offsets_arr = np.arange(0, (n_cells + 1) * npc, npc, dtype=np.int32)
    element_types_arr = np.full(n_cells, ELEMENT_TYPES[etype_name], dtype=np.uint8)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=grid_meta,
    )


def _read_field_data(path: Source) -> PolyData:
    """Read a VTK FIELD data file (no geometry) into an empty PolyData.

    Parameters
    ----------
    path
        Path to the .vtk file.

    Returns
    -------
    PolyData
        Empty mesh (no vertices, no elements). Field arrays stored in
        ``global_attrs`` keyed by array name.
    """
    warnings.warn(
        f"{source_name(path)}: VTK FIELD dataset has no geometry. "
        "Returning empty PolyData with field arrays in global_attrs.",
        UserWarning,
        stacklevel=4,
    )

    raw = read_bytes(path)
    lines, _ = _lines_with_offsets(raw)

    global_attrs: dict[str, object] = {}
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("FIELD"):
            # FIELD name n_arrays
            i += 1
            continue

        # array header: name n_comp n_tuples dtype
        if len(parts) == 4:
            name, n_comp_s, n_tuples_s, vtk_dt = parts
            try:
                n_comp = int(n_comp_s)
                n_tuples = int(n_tuples_s)
            except ValueError:
                i += 1
                continue
            # skip string arrays - skip exactly n_tuples data lines
            if vtk_dt.lower() == "string":
                skipped = 0
                while skipped < n_tuples and i < n_lines:
                    if lines[i]:
                        skipped += 1
                    i += 1
                continue
            i += 1
            vals: list[float] = []
            while len(vals) < n_tuples * n_comp and i < n_lines:
                for token in lines[i].split():
                    try:
                        vals.append(float(token))
                    except ValueError:
                        pass
                i += 1
            arr = np.array(vals[: n_tuples * n_comp], dtype=np.float64)
            global_attrs[name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
            continue

        i += 1

    return PolyData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        global_attrs=global_attrs,
    )


def _read_structured_attr(
    texts: list[str],
    offsets: list[int],
    raw: bytes,
    i: int,
    *,
    n_items: int,
    is_binary: bool,
    attrs: dict[str, np.ndarray],
    expected: int,
    kind: str,
    unhandled: set[str],
) -> int:
    """Read one attribute of a structured dataset and keep what fits.

    The three structured readers all reach their attributes the same way -
    scan one, drop it if its rows belong to no point or cell of the mesh,
    say so if the keyword is not one this reader knows - so they all do it
    here.

    A section's own header is what says how long its payload is, and a file
    that does not hold that much leaves nowhere to look for the array after
    it. That costs the rest of the section, but not the mesh: the geometry
    was read whole before the scan reached this, and refusing to hand it
    back over a malformed attribute would lose more than it saved.

    Parameters
    ----------
    texts
        The file's lines, stripped.
    offsets
        Byte offset of each line, for finding a binary payload.
    raw
        The whole file.
    i
        Index of the attribute's header line.
    n_items
        Tuples the section declares: points for ``POINT_DATA``, cells for
        ``CELL_DATA``.
    is_binary
        Whether the payload is binary rather than ASCII.
    attrs
        Destination, written in place.
    expected
        Rows an array must have to belong to this mesh.
    kind
        ``'point'`` or ``'cell'``, for the warnings.
    unhandled
        Keywords already warned about in this file; added to in place.

    Returns
    -------
    int
        The line index to carry on from.
    """
    found: dict[str, np.ndarray] = {}
    try:
        nxt = _scan_structured_attr(
            texts,
            offsets,
            raw,
            i,
            n_items=n_items,
            is_binary=is_binary,
            attrs=found,
        )
    except CodecError as exc:
        warnings.warn(
            f"{exc} The rest of this data section is dropped.",
            UserWarning,
            stacklevel=5,
        )
        return _skip_data_section(texts, i)

    if nxt is None:
        _warn_unhandled_attr(texts[i], unhandled)
        return i + 1

    attrs.update(_keep_sized_attrs(found, expected=expected, kind=kind, stacklevel=6))
    return nxt


def _skip_data_section(texts: list[str], i: int) -> int:
    """Find the end of the attribute section a line sits in.

    Parameters
    ----------
    texts
        The file's lines, stripped.
    i
        Index to start looking from.

    Returns
    -------
    int
        Index of the next line that opens a section or returns to the
        geometry, or past the end when there is none.
    """
    n_lines = len(texts)
    i += 1
    while i < n_lines and not texts[i].upper().startswith(_ATTRS_STOP_KEYWORDS):
        i += 1
    return i


def _keep_sized_attrs(
    found: dict[str, np.ndarray], *, expected: int, kind: str, stacklevel: int
) -> dict[str, np.ndarray]:
    """Keep the arrays that cover the mesh, and name the ones that do not.

    A data section declares its own tuple count, and the mesh says how many
    points and cells there really are. When the two disagree the array's
    rows cannot be matched to anything, and attaching it anyway is a
    PolyData that fails validation with a message about lengths rather than
    about the file.

    Parameters
    ----------
    found
        Arrays just read out of one attribute section.
    expected
        Rows an array must have to belong to this mesh.
    kind
        ``'point'`` or ``'cell'``, for the warning.
    stacklevel
        Frames between here and the caller of ``read``, so the warning is
        blamed on the code that asked for the file.

    Returns
    -------
    dict of str to numpy.ndarray
        The arrays that cover the mesh.
    """
    kept: dict[str, np.ndarray] = {}
    for name, arr in found.items():
        if arr.shape[0] != expected:
            warnings.warn(
                f".vtk: {kind} array {name!r} covers {arr.shape[0]} of"
                f" {expected} {kind}s, so its rows cannot be matched to the"
                " mesh; dropped.",
                UserWarning,
                stacklevel=stacklevel,
            )
            continue
        kept[name] = arr
    return kept


def _grid_point_count(nx: int, ny: int, nz: int) -> int:
    """Count the points a grid of these ``DIMENSIONS`` lays out.

    An axis declaring no point at all - a zero, or the negative a malformed
    header spells - carries no plane, so the grid carries no points however
    many the other two declare. Multiplied out unguarded, an odd number of
    negative axes turns the product negative, which is not a count of
    anything: it was compared against a real array length that could never
    match it, and named in the warning that followed.

    Parameters
    ----------
    nx, ny, nz
        Points along each axis, as ``DIMENSIONS`` declares them.

    Returns
    -------
    int
        The point count, never below zero.

    Examples
    --------
    >>> _grid_point_count(3, 3, 3)
    27
    >>> _grid_point_count(-1, 3, 3)
    0
    """
    if min(nx, ny, nz) < 1:
        return 0
    return nx * ny * nz


def _structured_cells_over(
    nx: int, ny: int, nz: int, n_verts: int
) -> tuple[np.ndarray, str]:
    """Build a structured grid's cells, if the points it names are there.

    ``DIMENSIONS`` is the only thing that says how a ``STRUCTURED_GRID``
    lays its points out, and the cells are strides through that layout. A
    ``POINTS`` array that does not cover the grid leaves them indexing
    points the file never held - a mesh ``validate`` refuses, and one that
    silently reads a different vertex where it does not. The points are the
    data, so they are handed back on their own and the topology the header
    described is not.

    A ``RECTILINEAR_GRID`` cannot reach this: its points are the outer
    product of its coordinate arrays, so it reconciles the header against
    them and always covers its own grid.

    Parameters
    ----------
    nx, ny, nz
        Points along each axis, as ``DIMENSIONS`` declares them.
    n_verts
        Points the file actually delivered.

    Returns
    -------
    tuple[numpy.ndarray, str]
        One row of point indices per cell and the element type name, or no
        cells at all when the grid is not covered.
    """
    n_grid = _grid_point_count(nx, ny, nz)
    if n_grid == n_verts:
        return _structured_grid_cells(nx, ny, nz)

    # A file with no POINTS at all is empty rather than inconsistent, and
    # the default 1x1x1 grid it is read with has no cells to warn about.
    if n_verts or n_grid > 1:
        warnings.warn(
            f".vtk: DIMENSIONS says {nx} {ny} {nz}, which is {n_grid}"
            f" point(s), but POINTS holds {n_verts}; the cells they describe"
            " would index points the file does not hold, so the points are"
            " handed back without them.",
            UserWarning,
            stacklevel=5,
        )
    return np.zeros((0, 1), dtype=np.int32), "vertex"


def _structured_cell_count(nx: int, ny: int, nz: int) -> int:
    """Count the cells ``_structured_grid_cells`` will make for a grid.

    The attribute scan needs this before the cells themselves exist, because
    a ``CELL_DATA`` section declares its own count and the two have to be
    the same for its arrays to belong to this mesh.

    Parameters
    ----------
    nx, ny, nz
        Points along each axis, as ``DIMENSIONS`` declares them.

    Returns
    -------
    int
        Cells the grid holds; zero when it extends along no axis, and zero
        when an axis holds no point for it to extend over.
    """
    if min(nx, ny, nz) < 1:
        return 0
    spans = [dim - 1 for dim in (nx, ny, nz) if dim > 1]
    count = 1
    for span in spans:
        count *= span
    return count if spans else 0


def _structured_cell_count_over(nx: int, ny: int, nz: int, n_verts: int) -> int:
    """Count the cells a structured grid will end up with.

    ``_structured_cells_over`` hands back no cells at all when the points
    the file delivered do not cover the grid its header describes. A
    ``CELL_DATA`` section read before that is settled has to be measured
    against the same answer, or its arrays are kept against cells the mesh
    will not have and ``validate`` refuses the result.

    Parameters
    ----------
    nx, ny, nz
        Points along each axis, as ``DIMENSIONS`` declares them.
    n_verts
        Points the file actually delivered.

    Returns
    -------
    int
        Cells the mesh will hold; zero when the grid is not covered.
    """
    if _grid_point_count(nx, ny, nz) != n_verts:
        return 0
    return _structured_cell_count(nx, ny, nz)


def _scan_structured_attr(
    texts: list[str],
    offsets: list[int],
    raw: bytes,
    i: int,
    *,
    n_items: int,
    is_binary: bool,
    attrs: dict[str, np.ndarray],
) -> int | None:
    """Read one attribute of a structured dataset, wherever its section is.

    The three structured readers walk their file themselves rather than
    handing the body to ``_parse_vtk_data_attrs``, because a structured
    dataset has no explicit point array to anchor a byte offset to. They all
    walked it the same way, so they walk it here instead - which is also
    what lets a ``CELL_DATA`` section be read by the code that reads
    ``POINT_DATA``, rather than falling past a chain that only ever asked
    about points.

    Parameters
    ----------
    texts
        The file's lines, stripped.
    offsets
        Byte offset of each line, for finding a binary payload.
    raw
        The whole file.
    i
        Index of the attribute's header line.
    n_items
        Tuples the section declares: points for ``POINT_DATA``, cells for
        ``CELL_DATA``.
    is_binary
        Whether the payload is binary rather than ASCII.
    attrs
        Destination, written in place.

    Returns
    -------
    int or None
        The line index just past the attribute, or None when the header is
        not one this reader knows - which leaves the caller to say so.
    """
    line = texts[i]
    upper = line.upper()
    parts = line.split()
    n_lines = len(texts)
    where = f"line {i + 1}"

    if upper.startswith("METADATA"):
        # Component names and information keys, written after every array by
        # VTK 4.2 and later. No values to keep, but stepping over it is what
        # keeps an ordinary file from being reported as holding keywords this
        # reader does not know.
        return _skip_metadata(texts, i)

    if upper.startswith("LOOKUP_TABLE") and len(parts) > 2:
        # A table definition, not the 'LOOKUP_TABLE name' line a SCALARS
        # section carries - that one is consumed below. It is a palette
        # rather than a value per point or cell, so there is no array to
        # hang it on; its rgba rows are stepped over so the arrays after it
        # are still found.
        n_rows = _attr_count(parts, 2, where)
        table = _attr_name(parts, where)
        if is_binary:
            data_pos = offsets[i + 1] if i + 1 < len(offsets) else len(raw)
            _check_block(data_pos, n_rows * 4, len(raw), name=table)
            return _skip_payload(offsets, i + 1, n_lines, data_pos + n_rows * 4)
        return _read_ascii_values(texts, i + 1, n_rows * 4, name=table)[0]

    if upper.startswith("SCALARS"):
        name = _attr_name(parts, where)
        n_comp = _attr_count(parts, 3, where, default=1)
        i += 1
        if i < n_lines and "LOOKUP_TABLE" in texts[i].upper():
            i += 1
        if is_binary:
            # Resolved inside the binary branch: an ASCII payload is read as
            # text whatever its header names, so refusing a type this reader
            # has no numpy equivalent for would cost an array it can read.
            np_dt = _binary_dtype(parts, 2, where)
            data_pos = offsets[i] if i < len(offsets) else len(raw)
            n_bytes = n_items * n_comp * np.dtype(np_dt).itemsize
            _check_block(data_pos, n_bytes, len(raw), name=name)
            arr = np.frombuffer(raw[data_pos : data_pos + n_bytes], dtype=np_dt).astype(
                np.float64
            )
            attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr
            return _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
        i, values = _read_ascii_values(texts, i, n_items * n_comp, name=name)
        arr = np.array(values, dtype=np.float64)
        attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr
        return i

    if upper.startswith("COLOR_SCALARS"):
        # One unsigned char per component in binary against one float in
        # 0..1 in ASCII; the byte is scaled so the same colour reads back
        # the same way whichever encoding the file used.
        name = _attr_name(parts, where)
        n_comp = _attr_count(parts, 2, where, default=1)
        i += 1
        if is_binary:
            data_pos = offsets[i] if i < len(offsets) else len(raw)
            n_bytes = n_items * n_comp  # unsigned_char = 1 byte each
            _check_block(data_pos, n_bytes, len(raw), name=name)
            arr = (
                np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np.uint8
                ).astype(np.float64)
                / _COLOR_SCALE
            )
            attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr
            return _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
        i, values = _read_ascii_values(texts, i, n_items * n_comp, name=name)
        arr = np.array(values, dtype=np.float64)
        attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr
        return i

    if upper.startswith(("VECTORS", "NORMALS")):
        # NORMALS spells its header the way VECTORS does and holds the same
        # three components, so one branch reads both.
        name = _attr_name(parts, where)
        i += 1
        if is_binary:
            np_dt = _binary_dtype(parts, 2, where)
            data_pos = offsets[i] if i < len(offsets) else len(raw)
            n_bytes = n_items * 3 * np.dtype(np_dt).itemsize
            _check_block(data_pos, n_bytes, len(raw), name=name)
            arr = np.frombuffer(raw[data_pos : data_pos + n_bytes], dtype=np_dt).astype(
                np.float64
            )
            attrs[name] = arr.reshape(n_items, 3)
            return _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
        i, values = _read_ascii_values(texts, i, n_items * 3, name=name)
        attrs[name] = np.array(values, dtype=np.float64).reshape(n_items, 3)
        return i

    if upper.startswith("TEXTURE_COORDINATES"):
        name = _attr_name(parts, where)
        n_comp = _attr_count(parts, 2, where, default=2)
        i += 1
        if is_binary:
            np_dt = _binary_dtype(parts, 3, where)
            data_pos = offsets[i] if i < len(offsets) else len(raw)
            n_bytes = n_items * n_comp * np.dtype(np_dt).itemsize
            _check_block(data_pos, n_bytes, len(raw), name=name)
            arr = np.frombuffer(raw[data_pos : data_pos + n_bytes], dtype=np_dt).astype(
                np.float64
            )
            attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr
            return _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
        i, values = _read_ascii_values(texts, i, n_items * n_comp, name=name)
        arr = np.array(values, dtype=np.float64)
        attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr
        return i

    if upper.startswith("TENSORS"):
        name = _attr_name(parts, where)
        i += 1
        if is_binary:
            np_dt = _binary_dtype(parts, 2, where)
            data_pos = offsets[i] if i < len(offsets) else len(raw)
            n_bytes = n_items * 9 * np.dtype(np_dt).itemsize
            _check_block(data_pos, n_bytes, len(raw), name=name)
            arr = np.frombuffer(raw[data_pos : data_pos + n_bytes], dtype=np_dt).astype(
                np.float64
            )
            attrs[name] = arr.reshape(n_items, 3, 3)
            return _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
        i, values = _read_ascii_values(texts, i, n_items * 9, name=name)
        attrs[name] = np.array(values, dtype=np.float64).reshape(n_items, 3, 3)
        return i

    if upper.startswith("FIELD"):
        n_arrays = _attr_count(parts, 2, where)
        i += 1
        for _ in range(n_arrays):
            i = _next_header(texts, i)
            if i >= n_lines:
                break
            fparts = texts[i].split()
            fwhere = f"line {i + 1}"
            arr_name = fparts[0]
            n_comp_f = _attr_count(fparts, 1, fwhere)
            n_tuples_f = _attr_count(fparts, 2, fwhere)
            i += 1
            if is_binary:
                np_dt_f = _binary_dtype(fparts, 3, fwhere)
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_tuples_f * n_comp_f * np.dtype(np_dt_f).itemsize
                _check_block(data_pos, n_bytes, len(raw), name=arr_name)
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt_f
                ).astype(np.float64)
                i = _skip_payload(offsets, i, n_lines, data_pos + n_bytes)
            else:
                i, fvalues = _read_ascii_values(
                    texts, i, n_tuples_f * n_comp_f, name=arr_name
                )
                arr = np.array(fvalues, dtype=np.float64)
            attrs[arr_name] = arr.reshape(n_tuples_f, n_comp_f) if n_comp_f > 1 else arr
        return i

    return None


def _skip_payload(offsets: list[int], i: int, n_lines: int, data_end: int) -> int:
    """Step the line cursor past a binary payload.

    Parameters
    ----------
    offsets
        Byte offset of each line.
    i
        Index of the first line the payload starts on.
    n_lines
        How many lines there are.
    data_end
        Byte offset just past the payload.

    Returns
    -------
    int
        The first line that starts at or after the end of the payload.
    """
    # A binary payload holds a newline every few values, so the lines it was
    # cut into number in the thousands for a mesh of any size; the offsets
    # ascend, so the end of it is a search rather than a walk.
    return min(bisect_left(offsets, data_end, i), n_lines)


def _structured_grid_cells(nx: int, ny: int, nz: int) -> tuple[np.ndarray, str]:
    """Generate hex/quad/line cell connectivity for a structured grid.

    A dimension of one is a dimension the grid does not extend along, and
    which of the three that is decides nothing but the stride: an ``x``-``z``
    plane is as much a sheet of quads as an ``x``-``y`` one, and a column
    along ``y`` is a run of lines the same way a row along ``x`` is. The
    axes that extend are picked out first, so the cell type follows how many
    there are rather than which ones they happen to be.

    Parameters
    ----------
    nx, ny, nz
        Points along each axis, as ``DIMENSIONS`` declares them.

    Returns
    -------
    tuple[numpy.ndarray, str]
        One row of point indices per cell, and the element type name. A grid
        that extends along no axis is a single point and has no cells.
    """
    if min(nx, ny, nz) < 1:
        # An axis of no points at all empties the whole grid: the point count
        # is a product, so it goes to zero whatever the others say. Counting
        # the other two axes' cells here handed back quads whose corners named
        # points the file never held - ``DIMENSIONS 0 3 3`` read as four cells
        # over no vertices - and the strides they were built from are zero.
        return np.zeros((0, 1), dtype=np.int32), "vertex"

    dims = (nx, ny, nz)
    # A point's index is x + y * nx + z * nx * ny, so this is the step along
    # each axis.
    strides = (1, nx, nx * ny)
    axes = [axis for axis in range(3) if dims[axis] > 1]

    if not axes:
        return np.zeros((0, 1), dtype=np.int32), "vertex"

    spans = np.meshgrid(*(np.arange(dims[a] - 1) for a in axes), indexing="ij")
    v0 = sum(span.ravel() * strides[a] for span, a in zip(spans, axes))

    if len(axes) == 1:
        step = strides[axes[0]]
        return np.column_stack([v0, v0 + step]), "line"

    if len(axes) == 2:
        sa, sb = strides[axes[0]], strides[axes[1]]
        return np.column_stack([v0, v0 + sa, v0 + sa + sb, v0 + sb]), "quad"

    sa, sb, sc = strides
    cells = np.column_stack(
        [
            v0,
            v0 + sa,
            v0 + sa + sb,
            v0 + sb,
            v0 + sc,
            v0 + sa + sc,
            v0 + sa + sb + sc,
            v0 + sb + sc,
        ]
    )
    return cells, "hexahedron"


def _read_structured_points(path: Source, *, is_binary: bool) -> PolyData:
    """Read a VTK legacy STRUCTURED_POINTS dataset (ASCII or binary).

    Parameters
    ----------
    path
        Path to the .vtk file.
    is_binary
        True when the file header says BINARY.

    Returns
    -------
    PolyData
        Mesh with generated hex/quad/line connectivity and preserved
        vertex attributes.
    """
    raw = read_bytes(path)

    # Build (byte_offset, stripped_text) for every line
    texts, offsets = _lines_with_offsets(raw)

    nx, ny, nz = 1, 1, 1
    ox, oy, oz = 0.0, 0.0, 0.0
    sx, sy, sz = 1.0, 1.0, 1.0
    n_points = 0
    in_point_data = False
    # POINT_DATA and CELL_DATA both open a section of attributes; the
    # flag is what tells an unhandled keyword inside one from the
    # header lines above, which are skipped on purpose.
    in_data_section = False
    unhandled: set[str] = set()
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    # What a CELL_DATA section says it holds, against what the grid the
    # header describes actually has; a section is read with the first and
    # kept only when it matches the second.
    n_cells_declared = 0

    i = 0
    n_lines = len(texts)

    while i < n_lines:
        line = texts[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("DIMENSIONS"):
            where = f"line {i + 1}"
            nx = _attr_count(parts, 1, where)
            ny = _attr_count(parts, 2, where)
            nz = _attr_count(parts, 3, where)
        elif upper.startswith("ORIGIN"):
            ox, oy, oz = _attr_reals(parts, f"line {i + 1}", 3)
        elif upper.startswith(("SPACING", "ASPECT_RATIO")):
            sx, sy, sz = _attr_reals(parts, f"line {i + 1}", 3)
        elif upper.startswith("POINT_DATA"):
            n_points = _attr_count(parts, 1, f"line {i + 1}")
            in_point_data = True
            in_data_section = True
        elif upper.startswith("CELL_DATA"):
            n_cells_declared = _attr_count(parts, 1, f"line {i + 1}", default=0)
            in_point_data = False
            in_data_section = True
        elif in_data_section:
            i = _read_structured_attr(
                texts,
                offsets,
                raw,
                i,
                n_items=n_points if in_point_data else n_cells_declared,
                is_binary=is_binary,
                attrs=vertex_attrs if in_point_data else element_attrs,
                expected=(
                    _grid_point_count(nx, ny, nz)
                    if in_point_data
                    else _structured_cell_count(nx, ny, nz)
                ),
                kind="point" if in_point_data else "cell",
                unhandled=unhandled,
            )
            continue
        i += 1

    xs = ox + np.arange(nx, dtype=np.float64) * sx
    ys = oy + np.arange(ny, dtype=np.float64) * sy
    zs = oz + np.arange(nz, dtype=np.float64) * sz
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # A STRUCTURED_POINTS file is its header: without origin and spacing the
    # expanded points cannot be written back as the image they came from.
    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    grid_meta: dict[str, object] = {
        "vtk_dimensions": [nx, ny, nz],
        "vtk_origin": [ox, oy, oz],
        "vtk_spacing": [sx, sy, sz],
    }

    cells, etype_name = _structured_grid_cells(nx, ny, nz)
    if len(cells) == 0:
        return PolyData(
            vertices=vertices,
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            vertex_attrs=vertex_attrs,
            element_attrs=element_attrs,
            vertex_tags=vertex_tags,
            element_tags=element_tags,
            global_attrs=grid_meta,
        )

    n_cells = len(cells)
    npc = cells.shape[1]
    connectivity = cells.ravel().astype(np.int32)
    offsets_arr = np.arange(0, (n_cells + 1) * npc, npc, dtype=np.int32)
    element_types_arr = np.full(n_cells, ELEMENT_TYPES[etype_name], dtype=np.uint8)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=grid_meta,
    )
