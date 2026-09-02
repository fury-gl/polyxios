import struct
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._faces import report_flattened_faces
from polyxios._io import (
    Source,
    can_seek,
    map_read,
    open_block,
    open_read,
    open_write,
    source_size,
)
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".ply"

MAX_CONNECTIVITY_INDEX: int = 2**31 - 1

# PLY property type - numpy dtype string (without endian prefix)
_PLY_DTYPE: dict[str, str] = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}

# The PLY name of each numpy code a column can be written at, which is the
# reverse of _PLY_DTYPE over the eight types the format actually spells.
# Written from the array's own dtype rather than from the dtype it arrived
# under, so the header and the record can never disagree on a field's width.
_PLY_NAME: dict[str, str] = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "u4": "uint",
    "f4": "float",
    "f8": "double",
}

# struct format character per numpy code, so a record holding a list property -
# whose width is only known once its count has been read - can be walked one
# field at a time without a numpy call per field.
_PLY_STRUCT: dict[str, str] = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "f4": "f",
    "f8": "d",
}


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a PLY file and return a PolyData.

    Parameters
    ----------
    path
        Path to the .ply file.
    lazy
        If True, header is parsed eagerly but binary data section is mmap-backed.
        Not supported for ASCII PLY (raises LazyReadError).

    Returns
    -------
    PolyData
        Parsed mesh data. ``element edge`` records follow the faces as ``line``
        elements, and a property both elements declare keeps its values over
        both; one only a face declares is NaN over the edges, and the other
        way round, so every attribute stays one value per element.

    Raises
    ------
    LazyReadError
        If lazy=True and format is ASCII.
    CodecError
        On malformed PLY data.
    """
    # Measured rather than taken from a later read, because the header's
    # counts are checked against it before any of them sizes an array, and
    # the reader that would hand the size back runs after that check. Over a
    # compressed source that costs a decompression pass whose output is
    # thrown away as it is counted - constant memory, unlike holding the
    # whole file to measure it, which is the trade this format's large binary
    # files are on the wrong side of.
    file_size = source_size(path)

    with open_read(path) as fh:
        start = fh.tell()
        header, header_end_offset = _parse_header(fh)
        # The header parse walks the handle forward; every reader below seeks
        # from the file's start, so a caller's handle is put back where it
        # was found rather than left mid-file.
        if can_seek(fh):
            fh.seek(start)

    fmt = header["format"]
    elements = header["elements"]

    vert_elem = next((e for e in elements if e["name"] == "vertex"), None)
    face_elem = next((e for e in elements if e["name"] == "face"), None)

    n_verts = vert_elem["count"] if vert_elem else 0
    n_faces = face_elem["count"] if face_elem else 0

    # Compressed 3D Gaussian Splat format: 'chunk' element + packed vertex properties.
    _chunk_elem = next((e for e in elements if e["name"] == "chunk"), None)
    if _chunk_elem is not None:
        if fmt == "ascii":
            raise CodecError("Compressed 3DGS PLY in ASCII format is not supported.")
        if lazy:
            raise LazyReadError("Compressed 3DGS PLY does not support lazy reads.")
        return _read_compressed_3dgs(path, header, header_end_offset)

    # Estimate connectivity size: assume avg 4 nodes/face
    conn_estimate = n_faces * 4
    validate_header(n_verts, n_faces, conn_estimate, file_size)

    if fmt == "ascii":
        if lazy:
            raise LazyReadError("PLY ASCII format does not support lazy reads.")
        return _read_ascii(path, header, header_end_offset)

    little_endian = fmt == "binary_little_endian"

    if lazy:
        return _read_binary_lazy(path, header, header_end_offset, little_endian)
    return _read_binary(path, header, header_end_offset, little_endian)


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a PLY file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default), write binary little-endian.
    endian
        'little' (default) or 'big'.

    Notes
    -----
    ``line`` elements are written as ``element edge`` records and every other
    element as a face. A PLY element is a block and not a column, so the faces
    are written first and the lines after them: a mesh holding both comes back
    from a round trip with its faces ahead of its lines, whatever order it
    held them in.

    A face record is a flat ring of vertices and PLY has no other shape, so an
    element that is not one - a ``tetra``, a ``quadratic_triangle`` - is
    written as a ring of its nodes in mesh order and reads back as whatever
    that many vertices name: a triangle at three, a quad at four, a polygon
    otherwise. The element it was is not in the file, so it is named in a
    warning rather than lost quietly. The ``element_attrs`` are carried along with it and still
    describe the element they came in on; an index into the element column
    held outside the mesh does not.

    Numeric ``vertex_attrs`` and ``element_attrs`` become properties of their
    element, a multi-component one spelled as one property per column; an
    attribute that is not one row per entity, or whose name is not a bare
    token, has no record to sit in and is skipped with a warning. A column of
    64-bit integers is written at the narrowest PLY type that holds its
    values, since the format spells no integer that wide. The
    ``element_attrs`` are declared on the edge records as well as the faces,
    since an attribute is one value per element and a line's own value would
    otherwise have nowhere to go.
    """
    binary: bool = bool(opts.get("binary", True))
    endian: str = str(opts.get("endian", "little"))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    # PLY spells a two-ended element as 'element edge' with vertex1/vertex2,
    # not as a face list of length two: a reader handed the latter builds a
    # degenerate polygon, which is not the line the mesh held. The split is
    # one pass over the type column rather than a Python loop per element.
    line_code = ELEMENT_TYPES["line"]
    is_edge = np.asarray(poly.element_types) == line_code
    edge_indices = np.flatnonzero(is_edge).tolist()
    face_indices = np.flatnonzero(~is_edge).tolist()

    # Measured before the header is built: a raise once the writing has
    # started leaves a file whose header promises records the body never
    # carries, over whatever was already at that path.
    _check_edge_widths(poly.offsets, edge_indices)
    report_flattened_faces(
        offsets=poly.offsets,
        element_types=poly.element_types,
        face_indices=face_indices,
        fmt=".ply",
    )

    endian_chr = ("<" if endian == "little" else ">") if binary else ""
    vert_attrs = _writable_attrs(poly.vertex_attrs, n_verts, "vertex", endian_chr)
    elem_attrs = _writable_attrs(poly.element_attrs, n_elems, "element", endian_chr)
    count_name, count_code = _list_count_type(poly.offsets, face_indices)

    lines: list[bytes] = []
    lines.append(b"ply")

    if binary:
        fmt_str = f"binary_{endian}_endian"
        lines.append(f"format {fmt_str} 1.0".encode())
    else:
        lines.append(b"format ascii 1.0")

    lines.append(b"comment Written by polyxios")

    # Vertex element
    lines.append(f"element vertex {n_verts}".encode())
    lines.append(b"property double x")
    lines.append(b"property double y")
    lines.append(b"property double z")

    lines.extend(_property_lines(vert_attrs))

    # Face element
    lines.append(f"element face {len(face_indices)}".encode())
    lines.append(f"property list {count_name} int vertex_indices".encode())
    lines.extend(_property_lines(elem_attrs))

    if edge_indices:
        # The edge records carry the same element properties the faces do: an
        # attribute is one value per element, and leaving it off here would
        # drop whatever the mesh held on its lines and read back as NaN.
        lines.append(f"element edge {len(edge_indices)}".encode())
        lines.append(b"property int vertex1")
        lines.append(b"property int vertex2")
        lines.extend(_property_lines(elem_attrs))

    lines.append(b"end_header")

    header_bytes = b"\n".join(lines) + b"\n"

    with open_write(path) as fh:
        fh.write(header_bytes)

        if binary:
            # Cast once for the whole file rather than once per record: the
            # coordinates and the indices are the two largest arrays a mesh
            # holds, and converting a row of either at a time is a numpy call
            # per row that says nothing the one call up here does not.
            coords = np.ascontiguousarray(
                poly.vertices, dtype=np.dtype(endian_chr + "f8")
            )
            indices = np.ascontiguousarray(
                poly.connectivity, dtype=np.dtype(endian_chr + "i4")
            )
            # Vertices: interleaved per-vertex record (x, y, z, extra...)
            for vi in range(n_verts):
                fh.write(coords[vi].tobytes())
                _write_binary_attrs(fh, vert_attrs, vi)
            # Faces: interleaved per-face record (count, indices, extra...)
            pack_count = struct.Struct(endian_chr + _PLY_STRUCT[count_code]).pack
            for i in face_indices:
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                fh.write(pack_count(e - s))
                fh.write(indices[s:e].tobytes())
                _write_binary_attrs(fh, elem_attrs, i)
            # Edges: the two ends and the same properties, one record each.
            for i in edge_indices:
                s = int(poly.offsets[i])
                fh.write(indices[s : s + 2].tobytes())
                _write_binary_attrs(fh, elem_attrs, i)
        else:
            # ASCII: each vertex line = x y z [extra_props...]
            for vi in range(n_verts):
                row = [
                    f"{poly.vertices[vi, 0]:.10g}",
                    f"{poly.vertices[vi, 1]:.10g}",
                    f"{poly.vertices[vi, 2]:.10g}",
                ]
                _extend_ascii_attrs(row, vert_attrs, vi)
                fh.write((" ".join(row) + "\n").encode())
            # Each face line = count v0 v1 ... [extra_props...]
            for i in face_indices:
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                parts = [str(e - s)] + [str(int(v)) for v in poly.connectivity[s:e]]
                _extend_ascii_attrs(parts, elem_attrs, i)
                fh.write((" ".join(parts) + "\n").encode())
            # Each edge line = vertex1 vertex2 [extra_props...]
            for i in edge_indices:
                s = int(poly.offsets[i])
                ends = poly.connectivity[s : s + 2]
                parts = [str(int(ends[0])), str(int(ends[1]))]
                _extend_ascii_attrs(parts, elem_attrs, i)
                fh.write((" ".join(parts) + "\n").encode())


def _property_lines(attrs: dict[str, np.ndarray]) -> list[bytes]:
    """Return the ``property`` header lines a set of attributes declares.

    Parameters
    ----------
    attrs
        The attributes that go on one element, already filtered to the ones a
        record has room for.

    Returns
    -------
    list of bytes
        One line per property, a multi-component attribute spelled one
        property per column.
    """
    lines: list[bytes] = []
    for name, arr in attrs.items():
        dt_str = _PLY_NAME[arr.dtype.str[1:]]
        if arr.ndim == 2:
            lines.extend(
                f"property {dt_str} {name}_{ci}".encode() for ci in range(arr.shape[1])
            )
        else:
            lines.append(f"property {dt_str} {name}".encode())
    return lines


def _write_binary_attrs(fh: Any, attrs: dict[str, np.ndarray], index: int) -> None:
    """Append one record's attribute fields, in the header's declared order.

    The columns already carry the dtype the header declared them at, byte
    order included, so a field is written by taking its bytes - no dtype is
    built and no value is cast here, which on a mesh of a million records is
    a million of each that no longer happen.
    """
    for arr in attrs.values():
        fh.write(arr[index].tobytes())


# Above this a double no longer counts in ones, so a value past it is a
# magnitude rather than a number of things and reads better in the float
# spelling it arrived in.
_EXACT_INTEGER: float = 2.0**53


def _ascii_value(value: Any) -> str:
    """Return one attribute value as an ASCII record spells it.

    Parameters
    ----------
    value
        The value, at whatever dtype its column is written.

    Returns
    -------
    str
        The token to write.

    Notes
    -----
    A whole number is spelled in full rather than through ``%.10g``, which
    turns 12345678901234 into ``1.23456789e+13`` - not a token a reader
    expecting the declared integer property accepts, and one that has already
    dropped the digits telling the values apart. A 64-bit integer column is
    declared ``double`` for want of a PLY integer that wide, so the test is on
    the value and not on the column's dtype; every value ``%.10g`` already
    spelled exactly keeps the spelling it had.
    """
    number = float(value)
    if number.is_integer() and abs(number) < _EXACT_INTEGER:
        return str(int(number))
    return f"{number:.10g}"


def _extend_ascii_attrs(
    row: list[str], attrs: dict[str, np.ndarray], index: int
) -> None:
    """Append one record's attribute fields to an ASCII row."""
    for arr in attrs.values():
        if arr.ndim == 2:
            row.extend(_ascii_value(value) for value in arr[index])
        else:
            row.append(_ascii_value(arr[index]))


def _column_code(values: np.ndarray) -> str:
    """Return the numpy code one attribute column is written at.

    Parameters
    ----------
    values
        The column, as the mesh carries it.

    Returns
    -------
    str
        A code among the eight types PLY spells, without an endian prefix.

    Notes
    -----
    PLY has no 64-bit integer, so a column of one is written at the narrowest
    type that holds every value it carries: ``int`` when they all fit a signed
    32-bit field, ``double`` when they do not. Neither loses a value, where
    declaring ``int`` and writing eight bytes - which is what the two separate
    header and record mappings used to do between them - framed every field
    after it wrong and cost the records that followed.

    A boolean column stays at the ``float`` it has always been written as: it
    reads back the same either way, and narrowing it would rewrite files that
    round-trip today for nothing.
    """
    kind, size = values.dtype.kind, values.dtype.itemsize
    if kind == "f":
        return "f8" if size == 8 else "f4"
    if kind in "iu" and size <= 4:
        return f"{kind}{size}"
    if kind == "b":
        return "f4"
    # A 64-bit column, which the format has no type for. One pass settles
    # whether the values themselves need the width, or only their dtype did.
    if values.size:
        low, high = int(values.min()), int(values.max())
        fits = np.iinfo(np.int32)
        if low < fits.min or high > fits.max:
            return "f8"
    return "i4"


def _writable_attrs(
    attrs: dict[str, np.ndarray] | None, count: int, kind: str, endian_chr: str
) -> dict[str, np.ndarray]:
    """Return the attributes a record of this element has room for.

    Parameters
    ----------
    attrs
        The mesh's ``vertex_attrs`` or ``element_attrs``.
    count
        How many vertices or elements the mesh holds, which is how long an
        attribute has to be to describe them.
    kind
        ``vertex`` or ``element``, named in the warning.
    endian_chr
        Byte order the records are written in, ``<`` or ``>``; empty for the
        ASCII flavour, which has none.

    Returns
    -------
    dict of str to numpy.ndarray
        The attributes that describe these entities, each already at the
        dtype - byte order included - its property is declared at, so the
        header and the records cannot disagree on a field's width. A
        multi-component one is kept and spelled one property per column.

    Notes
    -----
    A PLY header declares a fixed set of properties per record, so an
    attribute that is not one row per entity has nowhere to sit: a short one
    is indexed off the end and a long one leaves values with no record. A
    name is a bare token in the header, so one carrying whitespace would split
    into two properties and read back as neither. All of it is reported rather
    than left to produce a file no reader can parse.

    The cast happens once per column here rather than once per record in the
    writer, which is where the byte order used to be applied a value at a
    time.
    """
    kept: dict[str, np.ndarray] = {}
    refused: set[str] = set()
    for name, raw in (attrs or {}).items():
        values = np.asarray(raw)
        if (
            not name.strip()
            or name.split() != [name]
            or values.ndim not in (1, 2)
            or values.shape[0] != count
            or values.dtype.kind not in "fiub"
        ):
            refused.add(name)
            continue
        kept[name] = np.ascontiguousarray(
            values, dtype=np.dtype(endian_chr + _column_code(values))
        )
    if refused:
        warnings.warn(
            f".ply: only numeric {kind}_attrs named by a bare token, one row"
            f" per {kind}, fit in a record; skipped {sorted(refused)}.",
            stacklevel=3,
        )
    return kept


def _list_count_type(offsets: np.ndarray, face_indices: list[int]) -> tuple[str, str]:
    """Return the type a face's vertex count is declared and written at.

    Parameters
    ----------
    offsets
        The mesh's element offsets.
    face_indices
        Which elements are written as faces.

    Returns
    -------
    tuple of (str, str)
        The PLY type name for the header, and the numpy code the count is
        written at.

    Notes
    -----
    ``uchar`` is what a face list is almost always declared with and stays the
    answer for every mesh whose faces fit in it, so the usual file is byte for
    byte the one this codec has always written. A polygon of more than 255
    vertices does not fit: declaring ``uchar`` and writing the count anyway
    spells a number the header says is one byte, which a reader honouring the
    declaration reads as a different face - so the declaration widens with the
    mesh instead.
    """
    widest = max(
        (int(offsets[i + 1]) - int(offsets[i]) for i in face_indices), default=0
    )
    if widest <= 0xFF:
        return "uchar", "u1"
    return ("ushort", "u2") if widest <= 0xFFFF else ("uint", "u4")


def _check_edge_widths(offsets: np.ndarray, edge_indices: list[int]) -> None:
    """Refuse a line element that does not carry exactly two ends.

    Parameters
    ----------
    offsets
        The mesh's element offsets.
    edge_indices
        Which elements are written as edge records.

    Raises
    ------
    CodecError
        When an element does not hold exactly two vertices; a PLY edge
        record has room for two and no reader would find the rest.

    Notes
    -----
    Every edge is measured here, before the file is opened, rather than as
    its record is written: raising partway through leaves a header promising
    records the body does not carry, over whatever file was already there.
    One subtraction over the whole index array does it, so the check costs a
    pass and not a Python call per edge.
    """
    if not edge_indices:
        return
    picked = np.asarray(edge_indices, dtype=np.int64)
    bounds = np.asarray(offsets)
    widths = bounds[picked + 1] - bounds[picked]
    bad = np.flatnonzero(widths != 2)
    if bad.size:
        first = int(picked[bad[0]])
        raise CodecError(
            f".ply: element {first} is a line with {int(widths[bad[0]])}"
            " vertex(es); a PLY edge record holds exactly two."
        )


def _read_ascii(path: Source, header: dict, header_end_offset: int) -> PolyData:
    """Parse an ASCII PLY body, one element block at a time.

    Parameters
    ----------
    path
        The file being read.
    header
        The parsed header, whose ``elements`` are walked in declared order.
    header_end_offset
        Where the header ends and the records begin.

    Returns
    -------
    PolyData
        The mesh, edges following the faces whatever order the header put
        them in.

    Raises
    ------
    CodecError
        On a block the file ends inside, a malformed record, or a vertex
        index no vertex answers to.

    Notes
    -----
    The blocks are consumed in the order the header declares them, since that
    is the order they sit in the file: a header naming ``edge`` before
    ``face`` would otherwise have its records read as each other's. Every
    ASCII record is one line, so one bound check covers a whole block.
    """
    elements = header["elements"]
    vert_elem = next((e for e in elements if e["name"] == "vertex"), None)
    n_verts = vert_elem["count"] if vert_elem else 0

    with open_read(path) as fh:
        start = fh.tell()
        fh.seek(start + header_end_offset)
        lines = fh.read().decode("ascii", errors="replace").splitlines()
        if can_seek(fh):
            fh.seek(start)

    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    extra_vert_props: dict[str, list] = {}
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    extra_face_props: dict[str, list] = {}
    # Edges are held back and appended after the faces, whatever order the
    # header declares the two in, so a per-face attribute keeps lining up
    # with the faces it describes.
    edge_conn: list[int] = []
    extra_edge_props: dict[str, list] = {}

    idx = 0
    for elem in elements:
        name, count = elem["name"], elem["count"]
        if count < 0:
            raise CodecError(f".ply: element {name!r} declares {count} record(s).")
        if len(lines) - idx < count:
            raise CodecError(
                f".ply: the file ends inside its {count} {name} record(s)."
            )
        block = lines[idx : idx + count]
        idx += count
        if name == "vertex":
            _read_ascii_vertices(block, elem["properties"], vertices, extra_vert_props)
        elif name == "face":
            _read_ascii_faces(
                block, elem["properties"], conn_list, offsets_list, extra_face_props
            )
        elif name == "edge":
            _read_ascii_edges(block, elem["properties"], edge_conn, extra_edge_props)
        # An element this codec has no place for costs its own records and
        # nothing else, since each of them is one line.

    types_list = _face_types(offsets_list)
    n_faces = len(types_list)
    element_attrs = _padded_columns(
        {k: np.array(v, dtype=np.float64) for k, v in extra_face_props.items()},
        n_faces,
    )

    if edge_conn:
        n_edges = len(edge_conn) // 2
        base = offsets_list[-1]
        conn_list.extend(edge_conn)
        offsets_list.extend(range(base + 2, base + 2 * n_edges + 1, 2))
        types_list.extend([ELEMENT_TYPES["line"]] * n_edges)
        element_attrs = _combined_element_attrs(
            element_attrs,
            _padded_columns(
                {k: np.array(v, dtype=np.float64) for k, v in extra_edge_props.items()},
                n_edges,
            ),
            n_faces,
            n_edges,
        )

    return PolyData(
        vertices=vertices,
        connectivity=_checked_connectivity(conn_list, n_verts),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs={k: np.array(v) for k, v in extra_vert_props.items()},
        element_attrs=element_attrs,
    )


def _read_ascii_vertices(
    block: list[str],
    props: list[tuple[str, Any]],
    vertices: np.ndarray,
    extra: dict[str, list],
) -> None:
    """Fill the vertex table and its extra columns from one block of lines.

    Parameters
    ----------
    block
        The element's records, one per line.
    props
        Its ``(name, type)`` properties, in declared order.
    vertices
        The table to fill, shape ``(count, 3)``.
    extra
        Where the properties that are not coordinates collect.

    Raises
    ------
    CodecError
        On a record that is short or does not spell a number.

    Notes
    -----
    Which property sits where is resolved once rather than per record: the
    inner loop runs once per number in the file, and a dictionary lookup
    there is a lookup per number.
    """
    if not block:
        return
    coord_map = {"x": 0, "y": 1, "z": 2}
    coord_slots = [
        (pi, coord_map[name]) for pi, (name, _) in enumerate(props) if name in coord_map
    ]
    extra_slots = [
        (pi, name) for pi, (name, _) in enumerate(props) if name not in coord_map
    ]
    columns = [(pi, extra.setdefault(name, [])) for pi, name in extra_slots]
    for vi, ln in enumerate(block):
        vals = ln.split()
        try:
            for pi, axis in coord_slots:
                vertices[vi, axis] = float(vals[pi])
            for pi, column in columns:
                column.append(float(vals[pi]))
        except (IndexError, ValueError) as exc:
            raise CodecError(f".ply: malformed vertex record {ln!r}.") from exc


def _read_ascii_faces(
    block: list[str],
    props: list[tuple[str, Any]],
    conn_list: list[int],
    offsets_list: list[int],
    extra: dict[str, list],
) -> None:
    """Read one block of face records, the vertex list and its scalars alike.

    Parameters
    ----------
    block
        The element's records, one per line.
    props
        Its ``(name, type)`` properties, in declared order.
    conn_list, offsets_list
        The connectivity being built and its running offsets.
    extra
        Where the scalar properties collect, one column per name.

    Raises
    ------
    CodecError
        On a record that is short, names a negative vertex count, or does not
        spell a number.

    Notes
    -----
    A scalar property may sit either side of the vertex list - Armadillo.ply
    puts one ahead of it - so the record is walked property by property and
    the token positions cannot be resolved ahead of time. Which properties
    those are still can be.
    """
    if not block:
        return
    layout = [(name, ptype[0] == "list") for name, ptype in props]
    # A property only this block declares starts where this block does, so
    # the faces an earlier one read carry no value for it.
    base = len(offsets_list) - 1
    columns: dict[str, list] = {}
    for name, is_list in layout:
        if is_list:
            continue
        column = extra.setdefault(name, [])
        if len(column) < base:
            column.extend([float("nan")] * (base - len(column)))
        columns[name] = column
    running = offsets_list[-1]
    for ln in block:
        vals = ln.split()
        at = 0
        try:
            for name, is_list in layout:
                if is_list:
                    count = int(vals[at])
                    at += 1
                    if count < 0 or at + count > len(vals):
                        raise CodecError(f".ply: malformed face record {ln!r}.")
                    conn_list.extend(map(int, vals[at : at + count]))
                    at += count
                    running += count
                    offsets_list.append(running)
                else:
                    columns[name].append(float(vals[at]))
                    at += 1
        except (IndexError, ValueError) as exc:
            raise CodecError(f".ply: malformed face record {ln!r}.") from exc


def _read_ascii_edges(
    block: list[str],
    props: list[tuple[str, Any]],
    edge_conn: list[int],
    extra: dict[str, list],
) -> None:
    """Read one block of edge records into vertex pairs and scalar columns.

    Parameters
    ----------
    block
        The element's records, one per line.
    props
        Its ``(name, type)`` properties, in declared order.
    edge_conn
        Where the pairs collect.
    extra
        Where the properties that are not the two ends collect, one column
        per name; they describe the line elements the same way a face
        property describes the faces.

    Raises
    ------
    CodecError
        On a record too short to name two ends, or one that does not spell an
        integer.

    Notes
    -----
    The record is walked property by property rather than indexed by position,
    since a list property spends as many tokens as its count declares and
    every property after it would otherwise be read one token short.
    """
    if not block:
        return
    first, second = _edge_end_slots(props)
    layout = [
        (i, name, isinstance(ptype, tuple) and ptype[0] == "list")
        for i, (name, ptype) in enumerate(props)
    ]
    # As with the faces, a property only a later block declares carries no
    # value for the edges an earlier one read.
    base = len(edge_conn) // 2
    columns: dict[str, list] = {}
    for i, name, is_list in layout:
        if is_list or i in (first, second):
            continue
        column = extra.setdefault(name, [])
        if len(column) < base:
            column.extend([float("nan")] * (base - len(column)))
        columns[name] = column
    ends = [0, 0]
    for ln in block:
        vals = ln.split()
        at = 0
        try:
            for i, name, is_list in layout:
                if is_list:
                    count = int(vals[at])
                    at += 1
                    if count < 0 or at + count > len(vals):
                        raise CodecError(f".ply: malformed edge record {ln!r}.")
                    at += count
                    continue
                if i == first:
                    ends[0] = int(vals[at])
                elif i == second:
                    ends[1] = int(vals[at])
                else:
                    columns[name].append(float(vals[at]))
                at += 1
        except (IndexError, ValueError) as exc:
            raise CodecError(f".ply: malformed edge record {ln!r}.") from exc
        edge_conn.extend(ends)


def _face_types(offsets_list: list[int]) -> list[int]:
    """Return the element type each face's vertex count names."""
    tri_code = ELEMENT_TYPES["triangle"]
    quad_code = ELEMENT_TYPES["quad"]
    poly_code = ELEMENT_TYPES["polygon"]
    types: list[int] = []
    for i in range(len(offsets_list) - 1):
        n = offsets_list[i + 1] - offsets_list[i]
        types.append(tri_code if n == 3 else quad_code if n == 4 else poly_code)
    return types


def _edge_end_slots(props: list[tuple[str, Any]]) -> tuple[int, int]:
    """Return the positions of the two properties holding an edge's ends.

    Parameters
    ----------
    props
        The edge element's ``(name, type)`` properties, in declared order.

    Returns
    -------
    tuple of (int, int)
        Where the two ends sit among the properties, which is where they sit
        in an ASCII record and which field of a binary one they name.

    Raises
    ------
    CodecError
        When the element declares fewer than two scalar properties.

    Notes
    -----
    The spec spells the ends ``vertex1``/``vertex2``, but files in the wild
    use ``vertex_index1``/``vertex_index2`` and a few just number two integer
    properties, so the fallback is the first two scalars in declared order.
    """
    scalars = [i for i, (_, ptype) in enumerate(props) if not isinstance(ptype, tuple)]
    at = {props[i][0]: i for i in scalars}
    for first, second in (("vertex1", "vertex2"), ("vertex_index1", "vertex_index2")):
        if first in at and second in at:
            return at[first], at[second]
    if len(scalars) >= 2:
        return scalars[0], scalars[1]
    raise CodecError(
        ".ply: an edge element declares fewer than two scalar properties, so"
        " it names no pair of ends."
    )


def _combined_element_attrs(
    face_attrs: dict[str, np.ndarray],
    edge_attrs: dict[str, np.ndarray],
    n_faces: int,
    n_edges: int,
) -> dict[str, np.ndarray]:
    """Return one column per property name, over the faces then the edges.

    Parameters
    ----------
    face_attrs
        One value per face, in face order.
    edge_attrs
        One value per edge, in edge order.
    n_faces, n_edges
        How many of each the mesh holds; the edges follow the faces.

    Returns
    -------
    dict of str to numpy.ndarray
        The names either element declared, each one value per element. A name
        both declare keeps the values and the width they were read at; one
        only a face declares is NaN over the edges and the other way round,
        since the format spells no missing value and an attribute shorter than
        the mesh is one every reader downstream indexes off the end of.
    """
    combined: dict[str, np.ndarray] = {}
    for name in (*face_attrs, *(n for n in edge_attrs if n not in face_attrs)):
        head, tail = face_attrs.get(name), edge_attrs.get(name)
        combined[name] = np.concatenate(
            [_sized(head, n_faces, tail), _sized(tail, n_edges, head)]
        )
    return combined


def _sized(
    values: np.ndarray | None, count: int, other: np.ndarray | None
) -> np.ndarray:
    """Return the column when it describes these entities, else that many NaN.

    Parameters
    ----------
    values
        The column read for one of the two elements, or None when that
        element did not declare the property.
    count
        How many of that element the mesh holds.
    other
        The column read for the other element, whose width shapes the NaN
        rows when there is nothing else to take it from. At least one of the
        two is always a column.

    Returns
    -------
    numpy.ndarray
        ``count`` rows.
    """
    if values is not None and values.shape[0] == count:
        return values
    # The caller only asks about a name one of the two elements declared, so
    # one of the columns is always there to take a width from; falling back
    # to a plain column keeps that from turning into an AttributeError if a
    # later caller ever asks about a name neither declared.
    like = values if values is not None else other
    shape = (count,) if like is None else (count, *like.shape[1:])
    return np.full(shape, np.nan, dtype=np.float64)


def _decode_binary_faces(
    mv: memoryview,
    pos: int,
    count: int,
    props: list[tuple[str, Any]],
    endian: str,
    conn_list: list[int],
    offsets_list: list[int],
    extra_out: dict[str, np.ndarray],
) -> int:
    """Walk a binary ``face`` block, appending its indices and scalar columns.

    Parameters
    ----------
    mv
        The file's bytes.
    pos
        Where the element's records begin.
    count
        How many of them there are.
    props
        The element's ``(name, type)`` properties, in declared order.
    endian
        ``<`` or ``>``.
    conn_list, offsets_list
        The connectivity being built and its running offsets.
    extra_out
        Where the scalar properties are left, one column per name.

    Returns
    -------
    int
        Where the element's records end.

    Raises
    ------
    CodecError
        When the file ends inside the block, a record names a negative vertex
        count, or a property names a type of no known width.

    Notes
    -----
    The properties are walked in declared order, since a scalar may sit ahead
    of the vertex list - Armadillo.ply puts one there - and reading them out
    of order would take every field after it from the wrong offset. The walk
    uses ``struct`` on the mapping directly: a numpy call per field copies a
    bytes object and builds an array to read one number out of it, which on a
    mesh of a million faces is millions of both.
    """
    # How many faces the earlier blocks left behind, which is where this
    # block's property values belong in their columns.
    n_before = len(offsets_list) - 1
    uniform = _fixed_face_records(mv, pos, count, props, endian)
    if uniform is not None:
        records, list_at, n_nodes, end = uniform
        conn_list.extend(records[f"f{list_at}"].ravel().tolist())
        base = offsets_list[-1]
        offsets_list.extend(
            (base + np.arange(1, count + 1, dtype=np.int64) * n_nodes).tolist()
        )
        for i, (pname, _) in enumerate(props):
            if i != list_at:
                _append_column(
                    extra_out, pname, records[f"f{i}"].astype(np.float64), n_before
                )
        return end

    # (column or None, format, field size, index character, index size) per
    # property; the list property has no column and spends its index size on
    # whatever count it declares.
    layout: list[tuple[list | None, str, int, str, int]] = []
    columns: dict[str, list] = {}
    for pname, ptype in props:
        if isinstance(ptype, tuple) and ptype[0] == "list":
            count_fmt = endian + _PLY_STRUCT[_scalar_code(ptype[1], "face")]
            idx_char = _PLY_STRUCT[_scalar_code(ptype[2], "face")]
            layout.append(
                (
                    None,
                    count_fmt,
                    struct.calcsize(count_fmt),
                    idx_char,
                    struct.calcsize(idx_char),
                )
            )
        else:
            fmt = endian + _PLY_STRUCT[_scalar_code(ptype, "face")]
            column: list = []
            columns[pname] = column
            layout.append((column, fmt, struct.calcsize(fmt), "", 0))

    limit = len(mv)
    running = offsets_list[-1]
    unpack_from = struct.unpack_from
    try:
        for _ in range(count):
            for column, fmt, size, idx_char, idx_size in layout:
                value = unpack_from(fmt, mv, pos)[0]
                pos += size
                if column is not None:
                    column.append(float(value))
                    continue
                n_nodes = int(value)
                if n_nodes < 0 or pos + n_nodes * idx_size > limit:
                    raise CodecError(
                        f".ply: the file ends inside its {count} face record(s)."
                    )
                conn_list.extend(unpack_from(f"{endian}{n_nodes}{idx_char}", mv, pos))
                pos += n_nodes * idx_size
                running += n_nodes
                offsets_list.append(running)
    except struct.error as exc:
        raise CodecError(
            f".ply: the file ends inside its {count} face record(s)."
        ) from exc

    for pname, values in columns.items():
        _append_column(extra_out, pname, np.array(values, dtype=np.float64), n_before)
    return pos


def _append_column(
    extra_out: dict[str, np.ndarray], name: str, column: np.ndarray, base: int
) -> None:
    """Add one block's column to the property, keeping what came before it.

    Parameters
    ----------
    extra_out
        Where the scalar properties collect, one column per name.
    name
        The property this column belongs to.
    column
        The values this block read for it.
    base
        How many faces were read before this block, which is where its
        values belong in the column.

    Notes
    -----
    A header is free to declare ``face`` more than once, and the records of
    every such block are elements of the one mesh. Assigning here would keep
    only the last block's values and hand back a property shorter than the
    element column, which every reader downstream indexes off the end of.

    A property only the later block declares starts where that block does,
    not at the first face, so the run ahead of it is NaN: appending it flush
    would move every value onto a face that never carried it.
    """
    held = extra_out.get(name)
    held = _nan_run(base) if held is None else _resized(held, base)
    extra_out[name] = np.concatenate([held, column]) if held.size else column


def _nan_run(count: int) -> np.ndarray:
    """Return ``count`` missing values, which is what no record spelled."""
    return np.full(count, np.nan, dtype=np.float64)


def _padded_columns(attrs: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    """Return each column at ``count`` rows, NaN over the records it never reached.

    Parameters
    ----------
    attrs
        The columns one element's blocks left behind.
    count
        How many of that element the mesh holds.

    Returns
    -------
    dict of str to numpy.ndarray
        The same mapping when every column already describes them all, which
        is every file that declares its element once.

    Notes
    -----
    A property one block declares and the next does not stops short of the
    element column, and an attribute shorter than the mesh is one every
    reader downstream indexes off the end of.
    """
    if all(column.shape[0] == count for column in attrs.values()):
        return attrs
    return {name: _resized(column, count) for name, column in attrs.items()}


def _resized(column: np.ndarray, count: int) -> np.ndarray:
    """Return one column at ``count`` rows, whichever way it disagrees.

    Notes
    -----
    Short is the ordinary case - a property one block declares and the next
    does not. Long only happens on a file whose element carries values but no
    vertex list, so its records are not faces at all; the values past the last
    face describe nothing, and keeping them is what leaves an attribute the
    mesh cannot be indexed alongside.
    """
    held = column.shape[0]
    if held == count:
        return column
    if held > count:
        return column[:count]
    return np.concatenate([column, _nan_run(count - held)])


def _fixed_face_records(
    mv: memoryview,
    pos: int,
    count: int,
    props: list[tuple[str, Any]],
    endian: str,
) -> tuple[np.ndarray, int, int, int] | None:
    """Return the whole block as one structured read, when every record fits it.

    Parameters
    ----------
    mv
        The file's bytes.
    pos
        Where the element's records begin.
    count
        How many of them there are.
    props
        The element's ``(name, type)`` properties, in declared order.
    endian
        ``<`` or ``>``.

    Returns
    -------
    tuple or None
        The records, which property holds the vertex list, how many vertices
        each face carries, and where the block ends. None when the block is
        not of one width, which sends the caller to its record-by-record walk.

    Notes
    -----
    A record's width varies only with the length of its vertex list, and a
    mesh of one element type - which is nearly every mesh a file holds - has
    the same length in every record. Reading the first one settles the width
    to try; the counts are checked against it afterwards, so a mesh that
    mixes triangles and quads costs one array view and falls back rather than
    reading itself wrong. Fields are named by position, since a file is free
    to name a property anything and a dtype's field names have to be unique.
    """
    lists = [
        i for i, (_, t) in enumerate(props) if isinstance(t, tuple) and t[0] == "list"
    ]
    if count <= 0 or len(lists) != 1:
        return None
    list_at = lists[0]

    head = sum(
        np.dtype(_scalar_code(ptype, "face")).itemsize for _, ptype in props[:list_at]
    )
    count_code = _scalar_code(props[list_at][1][1], "face")
    index_code = _scalar_code(props[list_at][1][2], "face")
    if pos + head + np.dtype(count_code).itemsize > len(mv):
        return None
    n_nodes = int(
        np.frombuffer(mv, dtype=endian + count_code, count=1, offset=pos + head)[0]
    )
    if n_nodes <= 0:
        return None
    # One record of that width has to fit in what is left of the file before
    # a dtype is built for it. A corrupt count - a face declaring 2**31-1
    # vertices - describes a record wider than a C int can measure, which
    # numpy answers with a ValueError about a tuple shape, naming neither
    # the file nor the face it came out of. Falling back sends the block to
    # the record-by-record walk, which reports the truncation it is.
    record_size = (
        head + np.dtype(count_code).itemsize + n_nodes * np.dtype(index_code).itemsize
    )
    if pos + record_size > len(mv):
        return None

    fields: list[tuple] = []
    for i, (_, ptype) in enumerate(props):
        if i == list_at:
            fields.append((f"c{i}", endian + count_code))
            fields.append((f"f{i}", endian + index_code, (n_nodes,)))
        else:
            fields.append((f"f{i}", endian + _scalar_code(ptype, "face")))
    record = np.dtype(fields)

    end = pos + count * record.itemsize
    if end > len(mv):
        return None
    records = np.frombuffer(mv, dtype=record, count=count, offset=pos)
    if not bool(np.all(records[f"c{list_at}"] == n_nodes)):
        return None
    return records, list_at, n_nodes, end


def _checked_connectivity(conn_list: list[int], n_verts: int) -> np.ndarray:
    """Return the connectivity, refusing an index no vertex answers to."""
    connectivity = np.array(conn_list, dtype=np.int64)
    if connectivity.size:
        low, high = int(connectivity.min()), int(connectivity.max())
        if low < 0 or high >= n_verts:
            held = f"0..{n_verts - 1}" if n_verts else "no vertex at all"
            raise CodecError(
                f".ply: an element references vertex {low}..{high}, outside {held}."
            )
    return connectivity.astype(np.int32)


def _read_binary(
    path: Source, header: dict, header_end_offset: int, little_endian: bool
) -> PolyData:
    with open_block(path, fmt=".ply") as block:
        mv = memoryview(block)
        try:
            poly = _decode_binary(mv, header, header_end_offset, little_endian)
        finally:
            del mv  # release the view before the mapping goes
    return poly


def _read_binary_lazy(
    path: Source, header: dict, header_end_offset: int, little_endian: bool
) -> PolyData:
    # The mapping outlives the descriptor: mmap keeps the file alive on its own,
    # so closing fh here leaks nothing while the arrays still reference the
    # mapped pages, which the OS loads on demand.
    mm = map_read(path, fmt=".ply")
    mv = memoryview(mm)
    poly = _decode_binary(mv, header, header_end_offset, little_endian)
    # mm is kept alive by the arrays that view it, and unmapped once they go.
    return poly


def _decode_binary(
    mv: memoryview,
    header: dict,
    header_end_offset: int,
    little_endian: bool,
) -> PolyData:
    endian = "<" if little_endian else ">"
    pos = header_end_offset

    vertices = np.zeros((0, 3), dtype=np.float64)
    vertex_attrs: dict[str, np.ndarray] = {}
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    element_attrs: dict[str, np.ndarray] = {}
    # Edges are held back and appended after the faces, so a per-face
    # attribute keeps lining up with the faces it describes whatever order
    # the header declares the two elements in.
    edge_conn: list[int] = []
    edge_extra: dict[str, np.ndarray] = {}

    for elem in header["elements"]:
        ename = elem["name"]
        count = elem["count"]
        props = elem["properties"]

        if ename == "vertex":
            # One structured read for the usual all-scalar record; an element
            # carrying a list property has no fixed record width and is walked
            # instead, rather than raising on a dtype that cannot describe it.
            dt = _scalar_record_dtype(props, endian, "vertex")
            if dt is not None:
                nbytes = count * dt.itemsize
                if pos + nbytes > len(mv):
                    raise CodecError(
                        f".ply: the file ends inside its {count} vertex record(s)."
                    )
                rec: Any = np.frombuffer(bytes(mv[pos : pos + nbytes]), dtype=dt)
                pos += nbytes
            else:
                rec, pos = _read_mixed_records(mv, pos, count, props, endian, "vertex")

            coords = np.zeros((count, 3), dtype=np.float64)
            coord_map = {"x": 0, "y": 1, "z": 2}
            for pname, ptype in props:
                if isinstance(ptype, tuple):
                    # A list property has no column to become an attribute.
                    continue
                if pname in coord_map:
                    coords[:, coord_map[pname]] = rec[pname].astype(np.float64)
                else:
                    vertex_attrs[pname] = np.array(rec[pname])
            vertices = coords

        elif ename == "face":
            pos = _decode_binary_faces(
                mv, pos, count, props, endian, conn_list, offsets_list, element_attrs
            )

        elif ename == "edge":
            # A PLY edge names its two ends in scalar properties rather than
            # in a face list; read as a face it would build a degenerate
            # polygon instead of the line the file holds.
            first, second = _edge_end_slots(props)
            edge_dt = _scalar_record_dtype(props, endian, "edge")
            if edge_dt is not None:
                nbytes = count * edge_dt.itemsize
                if pos + nbytes > len(mv):
                    raise CodecError(
                        f".ply: the file ends inside its {count} edge record(s)."
                    )
                ends: Any = np.frombuffer(bytes(mv[pos : pos + nbytes]), dtype=edge_dt)
                pos += nbytes
            else:
                ends, pos = _read_mixed_records(mv, pos, count, props, endian, "edge")
            pairs = np.column_stack(
                [
                    ends[props[first][0]].astype(np.int64),
                    ends[props[second][0]].astype(np.int64),
                ]
            )
            # Extended, not assigned: a header declaring the element twice
            # would otherwise keep only its last block.
            edge_base = len(edge_conn) // 2
            edge_conn.extend(pairs.ravel().tolist())
            # Whatever else the record carries describes the line the way a
            # face property describes its face, so it travels with it.
            for pi, (pname, ptype) in enumerate(props):
                if isinstance(ptype, tuple) or pi in (first, second):
                    continue
                _append_column(
                    edge_extra,
                    pname,
                    np.asarray(ends[pname], dtype=np.float64),
                    edge_base,
                )

        else:
            # Skip unknown elements: compute size by summing property sizes
            pos = _skip_binary_element(mv, pos, count, props, endian)

    types_list = _face_types(offsets_list)
    element_attrs = _padded_columns(element_attrs, len(types_list))
    if edge_conn:
        n_faces = len(types_list)
        n_edges = len(edge_conn) // 2
        base = offsets_list[-1]
        conn_list.extend(edge_conn)
        offsets_list.extend(range(base + 2, base + 2 * n_edges + 1, 2))
        types_list.extend([ELEMENT_TYPES["line"]] * n_edges)
        element_attrs = _combined_element_attrs(
            element_attrs, _padded_columns(edge_extra, n_edges), n_faces, n_edges
        )

    return PolyData(
        vertices=vertices,
        connectivity=_checked_connectivity(conn_list, vertices.shape[0]),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def _scalar_code(ptype: Any, where: str) -> str:
    """Return the numpy code a scalar PLY property type names.

    Parameters
    ----------
    ptype
        The property's type as the header spells it.
    where
        The element's name, named in the error.

    Returns
    -------
    str
        The numpy dtype code, without an endian prefix.

    Raises
    ------
    CodecError
        On a type name outside the PLY set. Guessing a width here would read
        every field after it from the wrong offset, so the whole element is
        refused instead.
    """
    code = _PLY_DTYPE.get(ptype) if isinstance(ptype, str) else None
    if code is None:
        raise CodecError(
            f".ply: the {where} element declares an unknown property type {ptype!r}."
        )
    return code


def _scalar_record_dtype(
    props: list[tuple[str, Any]], endian: str, where: str
) -> np.dtype | None:
    """Return one structured dtype for an element, or None when a list bars it.

    Parameters
    ----------
    props
        The element's ``(name, type)`` properties, in declared order.
    endian
        ``<`` or ``>``.
    where
        The element's name, named in any error raised.

    Returns
    -------
    numpy.dtype or None
        A dtype covering one whole record, so the block reads in one call.
        None when a list property makes the record width vary, which is what
        ``_read_mixed_records`` is for.
    """
    if any(isinstance(ptype, tuple) for _, ptype in props):
        return None
    names = [pname for pname, _ in props]
    if len(set(names)) != len(names):
        # A record is read by naming its fields, and two fields answering to
        # one name leave no way to say which column a property is. numpy
        # refuses the dtype outright, which is the right answer said badly:
        # the error names neither the element nor the file.
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise CodecError(
            f".ply: the {where} element declares {repeated} more than once,"
            " so a record names no one column for them."
        )
    return np.dtype(
        [(pname, endian + _scalar_code(ptype, where)) for pname, ptype in props]
    )


def _read_mixed_records(
    mv: memoryview,
    pos: int,
    count: int,
    props: list[tuple[str, Any]],
    endian: str,
    where: str,
) -> tuple[dict[str, np.ndarray], int]:
    """Read an element whose records vary in width, one record at a time.

    Parameters
    ----------
    mv
        The file's bytes.
    pos
        Where the element's records begin.
    count
        How many of them there are.
    props
        The element's ``(name, type)`` properties, in declared order.
    endian
        ``<`` or ``>``.
    where
        The element's name, named in any error raised.

    Returns
    -------
    dict of str to numpy.ndarray
        One column per scalar property; the list properties are stepped over.
    int
        Where the element's records end.

    Raises
    ------
    CodecError
        When the file ends inside the block, or a property names a type of no
        known width.

    Notes
    -----
    Only reached by an element carrying a list property, which no common file
    does outside ``face``: a record whose width depends on a count inside it
    cannot be gathered by one structured read. The per-field walk uses
    ``struct`` rather than numpy, whose per-call overhead is what makes the
    scalar path worth keeping separate.
    """
    # (column, format, field width, index width) per property; a list property
    # has no column and spends its index width on whatever count it declares.
    layout: list[tuple[np.ndarray | None, str, int, int]] = []
    columns: dict[str, np.ndarray] = {}
    for pname, ptype in props:
        if isinstance(ptype, tuple) and ptype[0] == "list":
            fmt = endian + _PLY_STRUCT[_scalar_code(ptype[1], where)]
            idx_size = np.dtype(endian + _scalar_code(ptype[2], where)).itemsize
            layout.append((None, fmt, struct.calcsize(fmt), idx_size))
        else:
            code = _scalar_code(ptype, where)
            fmt = endian + _PLY_STRUCT[code]
            column = np.empty(count, dtype=code)
            columns[pname] = column
            layout.append((column, fmt, struct.calcsize(fmt), 0))

    limit = len(mv)
    for i in range(count):
        for column, fmt, size, idx_size in layout:
            if pos + size > limit:
                raise CodecError(
                    f".ply: the file ends inside its {count} {where} record(s)."
                )
            value = struct.unpack_from(fmt, mv, pos)[0]
            pos += size
            if column is None:
                pos += int(value) * idx_size
            else:
                column[i] = value
    if pos > limit:
        raise CodecError(f".ply: the file ends inside its {count} {where} record(s).")
    return columns, pos


def _skip_binary_element(
    mv: memoryview,
    pos: int,
    count: int,
    props: list[tuple[str, Any]],
    endian: str,
) -> int:
    """Step over an element this codec has no place for, record by record.

    Raises
    ------
    CodecError
        When the file ends inside the block. A list property's length is a
        number inside the record, so it can only be read once the bytes that
        hold it are known to be there; reading past the end instead raises an
        IndexError that names neither the element nor the format.

    Notes
    -----
    The record widths are resolved once, ahead of the walk, rather than per
    field per record: an element skipped here is often the largest in the
    file, and a dtype built per field is a dtype built millions of times.
    """
    limit = len(mv)
    # (format, field width, index width) per property; a list property has no
    # value to keep and spends its index width on whatever count it declares.
    layout: list[tuple[str, int, int]] = []
    for _, ptype in props:
        if isinstance(ptype, tuple) and ptype[0] == "list":
            fmt = endian + _PLY_STRUCT[_PLY_DTYPE.get(ptype[1], "u1")]
            idx_size = np.dtype(endian + _PLY_DTYPE.get(ptype[2], "i4")).itemsize
            layout.append((fmt, struct.calcsize(fmt), idx_size))
        else:
            size = np.dtype(endian + _PLY_DTYPE.get(ptype, "f4")).itemsize
            layout.append(("", size, 0))

    for _ in range(count):
        for fmt, size, idx_size in layout:
            if pos + size > limit:
                raise CodecError(
                    f".ply: the file ends inside its {count} skipped record(s)."
                )
            if not fmt:
                pos += size
                continue
            cnt = int(struct.unpack_from(fmt, mv, pos)[0])
            pos += size
            if cnt < 0:
                raise CodecError(f".ply: a skipped record names {cnt} list value(s).")
            pos += cnt * idx_size
    if pos > limit:
        raise CodecError(f".ply: the file ends inside its {count} skipped record(s).")
    return pos


def _read_compressed_3dgs(
    path: Source, header: dict, header_end_offset: int
) -> PolyData:
    """Decode a compressed 3D Gaussian Splat PLY file.

    The compressed format (produced by super-splat / splat-transform) stores:

    - A ``chunk`` element: one record per group of 256 vertices holding the
      bounding-box ranges needed to dequantise the packed vertex data.
    - A ``vertex`` element: four ``uint32`` per Gaussian -
      ``packed_position``, ``packed_rotation``, ``packed_scale``,
      ``packed_color``.  Each is a 4-byte little-endian integer whose bits
      are sliced into 8-bit normalised components.
    - An optional ``sh`` element: 45 ``uint8`` per Gaussian for higher-order
      SH coefficients.

    Parameters
    ----------
    path
        Path to the PLY file.
    header
        Parsed header dict from ``_parse_header()``.
    header_end_offset
        Byte offset of the first data byte after ``end_header``.

    Returns
    -------
    PolyData
        Point-cloud PolyData (E=0) with dequantised vertex positions and
        per-vertex attributes for scale, rotation, and colour.
    """
    elements = header["elements"]
    chunk_elem = next(e for e in elements if e["name"] == "chunk")
    vert_elem = next(e for e in elements if e["name"] == "vertex")
    sh_elem = next((e for e in elements if e["name"] == "sh"), None)

    n_chunks = chunk_elem["count"]
    n_verts = vert_elem["count"]

    # Build chunk structured dtype from its declared properties
    chunk_dt = np.dtype([(p[0], "<f4") for p in chunk_elem["properties"]])
    vert_dt = np.dtype(
        [
            ("packed_position", "<u4"),
            ("packed_rotation", "<u4"),
            ("packed_scale", "<u4"),
            ("packed_color", "<u4"),
        ]
    )

    with open_block(path, fmt=".ply") as mm:
        try:
            pos = header_end_offset
            chunk_bytes = n_chunks * chunk_dt.itemsize
            chunks = np.frombuffer(bytes(mm[pos : pos + chunk_bytes]), dtype=chunk_dt)
            pos += chunk_bytes

            vert_bytes = n_verts * vert_dt.itemsize
            verts_raw = np.frombuffer(bytes(mm[pos : pos + vert_bytes]), dtype=vert_dt)
            pos += vert_bytes

            sh_data: np.ndarray | None = None
            if sh_elem is not None:
                n_sh_props = len(sh_elem["properties"])
                sh_dt = np.dtype([("data", "u1", n_sh_props)])
                sh_bytes = n_verts * sh_dt.itemsize
                sh_data = np.frombuffer(
                    bytes(mm[pos : pos + sh_bytes]), dtype="u1"
                ).reshape(n_verts, n_sh_props)
        finally:
            mm.close()

    # Each chunk covers exactly CHUNK_SIZE consecutive vertices.
    # Chunk index for vertex i: i // CHUNK_SIZE.
    chunk_size = (n_verts + n_chunks - 1) // n_chunks  # ≈ 256
    chunk_idx = np.minimum(
        np.arange(n_verts, dtype=np.int32) // chunk_size, n_chunks - 1
    )

    def _unpack8(packed: np.ndarray, shift: int) -> np.ndarray:
        """Extract one 8-bit component and normalise to [0, 1]."""
        return ((packed >> shift) & 0xFF).astype(np.float32) / 255.0

    def _dequantise(packed, shift, lo, hi):
        norm = _unpack8(packed, shift)
        return lo + norm * (hi - lo)

    # Dequantise positions using per-chunk bounding boxes
    has_min_max = all(
        k in chunk_dt.names
        for k in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
    )
    if has_min_max:
        cx = chunk_idx
        x = _dequantise(
            verts_raw["packed_position"], 0, chunks["min_x"][cx], chunks["max_x"][cx]
        )
        y = _dequantise(
            verts_raw["packed_position"], 8, chunks["min_y"][cx], chunks["max_y"][cx]
        )
        z = _dequantise(
            verts_raw["packed_position"], 16, chunks["min_z"][cx], chunks["max_z"][cx]
        )
        vertices = np.column_stack([x, y, z]).astype(np.float64)
    else:
        vertices = np.zeros((n_verts, 3), dtype=np.float64)

    # Dequantise scales using per-chunk scale bounds (stored as log-scale)
    vertex_attrs: dict[str, np.ndarray] = {}
    has_scale = all(
        k in chunk_dt.names
        for k in (
            "min_scale_x",
            "max_scale_x",
            "min_scale_y",
            "max_scale_y",
            "min_scale_z",
            "max_scale_z",
        )
    )
    if has_scale:
        cx = chunk_idx
        vertex_attrs["scale_0"] = _dequantise(
            verts_raw["packed_scale"],
            0,
            chunks["min_scale_x"][cx],
            chunks["max_scale_x"][cx],
        )
        vertex_attrs["scale_1"] = _dequantise(
            verts_raw["packed_scale"],
            8,
            chunks["min_scale_y"][cx],
            chunks["max_scale_y"][cx],
        )
        vertex_attrs["scale_2"] = _dequantise(
            verts_raw["packed_scale"],
            16,
            chunks["min_scale_z"][cx],
            chunks["max_scale_z"][cx],
        )

    # Rotation: 8-bit per quaternion component, normalised to [-1, 1]
    for shift, name in ((0, "rot_0"), (8, "rot_1"), (16, "rot_2"), (24, "rot_3")):
        vertex_attrs[name] = ((verts_raw["packed_rotation"] >> shift) & 0xFF).astype(
            np.float32
        ) / 127.5 - 1.0

    # Color (SH DC) and opacity from packed_color
    has_rgb = all(
        k in chunk_dt.names
        for k in ("min_r", "max_r", "min_g", "max_g", "min_b", "max_b")
    )
    if has_rgb:
        cx = chunk_idx
        vertex_attrs["color_r"] = _dequantise(
            verts_raw["packed_color"], 0, chunks["min_r"][cx], chunks["max_r"][cx]
        )
        vertex_attrs["color_g"] = _dequantise(
            verts_raw["packed_color"], 8, chunks["min_g"][cx], chunks["max_g"][cx]
        )
        vertex_attrs["color_b"] = _dequantise(
            verts_raw["packed_color"], 16, chunks["min_b"][cx], chunks["max_b"][cx]
        )
    else:
        for shift, name in ((0, "color_r"), (8, "color_g"), (16, "color_b")):
            vertex_attrs[name] = ((verts_raw["packed_color"] >> shift) & 0xFF).astype(
                np.float32
            )

    vertex_attrs["opacity"] = ((verts_raw["packed_color"] >> 24) & 0xFF).astype(
        np.float32
    )

    # Optional higher-order SH coefficients
    if sh_data is not None:
        for i in range(sh_data.shape[1]):
            vertex_attrs[f"f_rest_{i}"] = sh_data[:, i].astype(np.float32) / 255.0

    return PolyData(
        vertices=vertices,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs={},
    )


def _parse_header(fh: object) -> tuple[dict, int]:
    """Parse PLY header from open binary file handle.

    Parameters
    ----------
    fh
        Open binary file handle positioned at start.

    Returns
    -------
    tuple[dict, int]
        (header_dict, byte_offset_after_header)
    """
    lines: list[str] = []
    offset = 0

    first = fh.readline()  # type: ignore[union-attr]
    offset += len(first)
    if first.strip() != b"ply":
        raise CodecError("Not a PLY file (missing 'ply' header)")

    header: dict[str, Any] = {"format": "ascii", "elements": []}
    current_elem: dict[str, Any] | None = None

    while True:
        raw = fh.readline()  # type: ignore[union-attr]
        if not raw:
            # readline() hands back nothing at the end of the file and goes on
            # doing so, so a header with no 'end_header' - which is every
            # truncated file - would otherwise be read forever.
            raise CodecError(
                ".ply: the file ends inside its header; no 'end_header' line."
            )
        offset += len(raw)
        line = raw.decode("ascii", errors="replace").strip()
        lines.append(line)

        if line == "end_header":
            break

        parts = line.split()
        if not parts:
            continue
        kw = parts[0]

        if kw == "format":
            _need(parts, 2, line)
            header["format"] = parts[1]
        elif kw == "element":
            _need(parts, 3, line)
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise CodecError(
                    f".ply: element {parts[1]!r} declares a count that is not"
                    f" a number, in the header line {line!r}."
                ) from exc
            current_elem = {"name": parts[1], "count": count, "properties": []}
            header["elements"].append(current_elem)
        elif kw == "property" and current_elem is not None:
            _need(parts, 2, line)
            if parts[1] == "list":
                # property list count_type data_type name
                _need(parts, 5, line)
                current_elem["properties"].append(
                    (parts[4], ("list", parts[2], parts[3]))
                )
            else:
                # property type name
                _need(parts, 3, line)
                current_elem["properties"].append((parts[2], parts[1]))
        # ignore comment, obj_info, etc.

    return header, offset


def _need(parts: list[str], count: int, line: str) -> None:
    """Refuse a header line that names fewer fields than its keyword needs.

    Parameters
    ----------
    parts
        The line's whitespace-separated fields.
    count
        How many the keyword spends.
    line
        The line as written, named in the error.

    Raises
    ------
    CodecError
        When the line is short. Indexing past it would raise an IndexError
        that names nothing about the file, where the reader's whole contract
        is that a file it cannot read is reported as one.
    """
    if len(parts) < count:
        raise CodecError(
            f".ply: the header line {line!r} names {len(parts)} field(s),"
            f" and '{parts[0]}' spends {count}."
        )
