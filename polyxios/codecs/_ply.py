import struct
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
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
        elements, and a per-face property is stretched over them with NaN, so
        every attribute stays one value per element.

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
    element as a face. Numeric ``vertex_attrs`` and ``element_attrs`` become
    properties of their element, a multi-component one spelled as one property
    per column; an attribute that is not one row per entity, or whose name is
    not a bare token, has no record to sit in and is skipped with a warning.
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

    vert_attrs = _writable_attrs(poly.vertex_attrs, n_verts, "vertex")
    face_attrs = _writable_attrs(poly.element_attrs, n_elems, "element")

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

    for name, arr in vert_attrs.items():
        dt_str = _np_to_ply_type(arr.dtype)
        if arr.ndim == 2:
            lines.extend(
                f"property {dt_str} {name}_{ci}".encode() for ci in range(arr.shape[1])
            )
        else:
            lines.append(f"property {dt_str} {name}".encode())

    # Face element
    lines.append(f"element face {len(face_indices)}".encode())
    lines.append(b"property list uchar int vertex_indices")

    for name, arr in face_attrs.items():
        dt_str = _np_to_ply_type(arr.dtype)
        if arr.ndim == 2:
            lines.extend(
                f"property {dt_str} {name}_{ci}".encode() for ci in range(arr.shape[1])
            )
        else:
            lines.append(f"property {dt_str} {name}".encode())

    if edge_indices:
        lines.append(f"element edge {len(edge_indices)}".encode())
        lines.append(b"property int vertex1")
        lines.append(b"property int vertex2")

    lines.append(b"end_header")

    header_bytes = b"\n".join(lines) + b"\n"

    with open_write(path) as fh:
        fh.write(header_bytes)

        if binary:
            endian_chr = "<" if endian == "little" else ">"
            # Vertices: interleaved per-vertex record (x, y, z, extra...)
            for vi in range(n_verts):
                fh.write(
                    np.asarray(
                        poly.vertices[vi], dtype=np.dtype(endian_chr + "f8")
                    ).tobytes()
                )
                for arr in vert_attrs.values():
                    val = np.asarray(arr[vi]).ravel()
                    dt = np.dtype(endian_chr + _np_short_dtype(val.dtype))
                    fh.write(val.astype(dt).tobytes())
            # Faces: interleaved per-face record (count, indices, extra...)
            count_dt = np.dtype("u1")
            idx_dt = np.dtype(endian_chr + "i4")
            for i in face_indices:
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                fh.write(np.array([e - s], dtype=count_dt).tobytes())
                fh.write(poly.connectivity[s:e].astype(idx_dt).tobytes())
                for arr in face_attrs.values():
                    val = np.asarray(arr[i]).ravel()
                    dt = np.dtype(endian_chr + _np_short_dtype(val.dtype))
                    fh.write(val.astype(dt).tobytes())
            # Edges: the two ends, one record each.
            for i in edge_indices:
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                fh.write(_edge_ends(poly, i, s, e).astype(idx_dt).tobytes())
        else:
            # ASCII: each vertex line = x y z [extra_props...]
            for vi in range(n_verts):
                row = [
                    f"{poly.vertices[vi, 0]:.10g}",
                    f"{poly.vertices[vi, 1]:.10g}",
                    f"{poly.vertices[vi, 2]:.10g}",
                ]
                for arr in vert_attrs.values():
                    if arr.ndim == 2:
                        row.extend(f"{arr[vi, ci]:.10g}" for ci in range(arr.shape[1]))
                    else:
                        row.append(f"{arr[vi]:.10g}")
                fh.write((" ".join(row) + "\n").encode())
            # Each face line = count v0 v1 ... [extra_props...]
            for i in face_indices:
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                parts = [str(e - s)] + [str(int(v)) for v in poly.connectivity[s:e]]
                for arr in face_attrs.values():
                    if arr.ndim == 2:
                        parts.extend(f"{arr[i, ci]:.10g}" for ci in range(arr.shape[1]))
                    else:
                        parts.append(f"{arr[i]:.10g}")
                fh.write((" ".join(parts) + "\n").encode())
            # Each edge line = vertex1 vertex2
            for i in edge_indices:
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                ends = _edge_ends(poly, i, s, e)
                fh.write(f"{int(ends[0])} {int(ends[1])}\n".encode())


def _writable_attrs(
    attrs: dict[str, np.ndarray] | None, count: int, kind: str
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

    Returns
    -------
    dict of str to numpy.ndarray
        The attributes that describe these entities, each as an array. A
        multi-component one is kept and spelled one property per column.

    Notes
    -----
    A PLY header declares a fixed set of properties per record, so an
    attribute that is not one row per entity has nowhere to sit: a short one
    is indexed off the end and a long one leaves values with no record. A
    name is a bare token in the header, so one carrying whitespace would split
    into two properties and read back as neither. All of it is reported rather
    than left to produce a file no reader can parse.
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
        kept[name] = values
    if refused:
        warnings.warn(
            f".ply: only numeric {kind}_attrs named by a bare token, one row"
            f" per {kind}, fit in a record; skipped {sorted(refused)}.",
            stacklevel=3,
        )
    return kept


def _edge_ends(poly: PolyData, index: int, start: int, end: int) -> np.ndarray:
    """Return an edge's two ends, refusing one that does not have exactly two.

    Parameters
    ----------
    poly
        The mesh being written.
    index
        Which element, named in the error.
    start, end
        The element's slice of the connectivity.

    Returns
    -------
    numpy.ndarray
        The two vertex indices.

    Raises
    ------
    CodecError
        When the element does not hold exactly two vertices; a PLY edge
        record has room for two and no reader would find the rest.
    """
    if end - start != 2:
        raise CodecError(
            f".ply: element {index} is a line with {end - start} vertex(es);"
            " a PLY edge record holds exactly two."
        )
    return np.asarray(poly.connectivity[start:end])


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
            _read_ascii_edges(block, elem["properties"], edge_conn)
        # An element this codec has no place for costs its own records and
        # nothing else, since each of them is one line.

    types_list = _face_types(offsets_list)
    element_attrs = {k: np.array(v) for k, v in extra_face_props.items()}

    if edge_conn:
        n_edges = len(edge_conn) // 2
        base = offsets_list[-1]
        conn_list.extend(edge_conn)
        offsets_list.extend(range(base + 2, base + 2 * n_edges + 1, 2))
        types_list.extend([ELEMENT_TYPES["line"]] * n_edges)
        element_attrs = _padded_face_attrs(element_attrs, len(types_list))

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
    columns = {
        name: extra.setdefault(name, []) for name, is_list in layout if not is_list
    }
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
    block: list[str], props: list[tuple[str, Any]], edge_conn: list[int]
) -> None:
    """Read one block of edge records into a flat run of vertex pairs.

    Parameters
    ----------
    block
        The element's records, one per line.
    props
        Its ``(name, type)`` properties, in declared order.
    edge_conn
        Where the pairs collect.

    Raises
    ------
    CodecError
        On a record too short to name two ends, or one that does not spell an
        integer.
    """
    if not block:
        return
    first, second = _edge_end_slots(props)
    for ln in block:
        vals = ln.split()
        try:
            edge_conn.append(int(vals[first]))
            edge_conn.append(int(vals[second]))
        except (IndexError, ValueError) as exc:
            raise CodecError(f".ply: malformed edge record {ln!r}.") from exc


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


def _padded_face_attrs(
    attrs: dict[str, np.ndarray], n_elems: int
) -> dict[str, np.ndarray]:
    """Return the face attributes stretched over the edges that follow them.

    Parameters
    ----------
    attrs
        One value per face, in face order.
    n_elems
        How many elements the mesh ends up holding, faces and edges together.

    Returns
    -------
    dict of str to numpy.ndarray
        The same names, each one value per element, NaN on the edges - a PLY
        edge record has no room for a face property, and an attribute shorter
        than the mesh is one every reader downstream indexes off the end of.
    """
    padded: dict[str, np.ndarray] = {}
    for name, values in attrs.items():
        stretched = np.full((n_elems, *values.shape[1:]), np.nan, dtype=np.float64)
        stretched[: values.shape[0]] = values
        padded[name] = stretched
    return padded


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
            # Locate the vertex-index list property and its dtype info.
            face_list_prop: tuple[str, tuple] | None = None
            for pname, ptype in props:
                if isinstance(ptype, tuple) and ptype[0] == "list":
                    face_list_prop = (pname, ptype)
                    break

            count_dt_str = (
                _PLY_DTYPE.get(face_list_prop[1][1], "u1") if face_list_prop else "u1"
            )
            idx_dt_str = (
                _PLY_DTYPE.get(face_list_prop[1][2], "i4") if face_list_prop else "i4"
            )
            count_size = np.dtype(endian + count_dt_str).itemsize
            index_size = np.dtype(endian + idx_dt_str).itemsize

            extra_data: dict[str, list] = {}

            # Read face properties in their declared order so that scalar
            # properties appearing before the vertex-index list (e.g. Armadillo.ply
            # which has `property uchar intensity` before `property list ...`)
            # are consumed at the right byte offset.
            for _ in range(count):
                for pname, ptype in props:
                    if isinstance(ptype, tuple) and ptype[0] == "list":
                        cnt = int(
                            np.frombuffer(
                                bytes(mv[pos : pos + count_size]),
                                dtype=endian + count_dt_str,
                            )[0]
                        )
                        pos += count_size
                        nbytes = cnt * index_size
                        indices = np.frombuffer(
                            bytes(mv[pos : pos + nbytes]), dtype=endian + idx_dt_str
                        ).astype(np.int32)
                        conn_list.extend(indices.tolist())
                        offsets_list.append(offsets_list[-1] + cnt)
                        pos += nbytes
                    else:
                        edt = endian + _PLY_DTYPE[ptype]
                        esize = np.dtype(edt).itemsize
                        val = np.frombuffer(bytes(mv[pos : pos + esize]), dtype=edt)[0]
                        extra_data.setdefault(pname, []).append(float(val))
                        pos += esize

            for pname, vals in extra_data.items():
                element_attrs[pname] = np.array(vals)

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
            edge_conn.extend(pairs.ravel().tolist())

        else:
            # Skip unknown elements: compute size by summing property sizes
            pos = _skip_binary_element(mv, pos, count, props, endian)

    types_list = _face_types(offsets_list)
    if edge_conn:
        n_edges = len(edge_conn) // 2
        base = offsets_list[-1]
        conn_list.extend(edge_conn)
        offsets_list.extend(range(base + 2, base + 2 * n_edges + 1, 2))
        types_list.extend([ELEMENT_TYPES["line"]] * n_edges)
        element_attrs = _padded_face_attrs(element_attrs, len(types_list))

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
    for _ in range(count):
        for _, ptype in props:
            if isinstance(ptype, tuple) and ptype[0] == "list":
                count_dt = endian + _PLY_DTYPE.get(ptype[1], "u1")
                cnt_size = np.dtype(count_dt).itemsize
                cnt = int(
                    np.frombuffer(bytes(mv[pos : pos + cnt_size]), dtype=count_dt)[0]
                )
                pos += cnt_size
                idx_size = np.dtype(endian + _PLY_DTYPE.get(ptype[2], "i4")).itemsize
                pos += cnt * idx_size
            else:
                pos += np.dtype(endian + _PLY_DTYPE.get(ptype, "f4")).itemsize
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
            header["format"] = parts[1]
        elif kw == "element":
            current_elem = {"name": parts[1], "count": int(parts[2]), "properties": []}
            header["elements"].append(current_elem)
        elif kw == "property" and current_elem is not None:
            if parts[1] == "list":
                # property list count_type data_type name
                current_elem["properties"].append(
                    (parts[4], ("list", parts[2], parts[3]))
                )
            else:
                # property type name
                current_elem["properties"].append((parts[2], parts[1]))
        # ignore comment, obj_info, etc.

    return header, offset


def _np_to_ply_type(dtype: np.dtype) -> str:
    kind = dtype.kind
    size = dtype.itemsize
    if kind == "f":
        return "double" if size == 8 else "float"
    if kind in ("i", "u"):
        signed = kind == "i"
        mapping = {1: "char", 2: "short", 4: "int"}
        base = mapping.get(size, "int")
        return base if signed else "u" + base
    return "float"


def _np_short_dtype(dtype: np.dtype) -> str:
    kind = dtype.kind
    size = dtype.itemsize
    if kind == "f":
        return "f8" if size == 8 else "f4"
    if kind == "i":
        return f"i{size}"
    if kind == "u":
        return f"u{size}"
    return "f4"
