from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._globals import globals_for_write, text_for_write
from polyxios._io import Source, write_bytes, write_text
from polyxios._tags import tags_from_masks, with_tag_masks
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    Parsed,
    appended_header_type,
    appended_section,
    attr_nbytes,
    decode_da,
    format_attr_da,
    format_da,
    format_field_data,
    header_type_attr,
    join_cells,
    join_piece_attrs,
    parse_xml,
    piece_cells,
    piece_count,
    piece_field_data,
    polygon_types,
    read_field_data,
    release_appended,
    shaped_da,
    spellable_arrays,
    undecodable_type,
    vtk_type_to_np,
)
from polyxios.exceptions import CodecError, UnsupportedFormatError
from polyxios.validate import validate_header

EXTENSION: str = ".vtp"

_SECTION_TYPES = ("Verts", "Lines", "Strips", "Polys")
# Polys is absent: its cells are typed by their point count.
_SECTION_CODES = {
    "Verts": ELEMENT_TYPES["vertex"],
    "Lines": ELEMENT_TYPES["line"],
    "Strips": ELEMENT_TYPES["triangle_strip"],
}


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a VTK PolyData XML file (.vtp) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtp file.
    lazy
        Map the file and hand back arrays that view it, in the dtype and
        byte order the file holds, read-only. Only a raw, uncompressed
        appended section (what VTK writes by default, and what
        ``write(..., appended=True)`` writes) can be viewed. With a single
        Piece holding a single cell section the vertices, connectivity and
        every attribute view the mapping; the offsets and element types are
        derived and so are copies. Pieces and sections are joined by copying.
        A connectivity the file declares as floats is cast to the integers
        an eager read gives, and is a copy too.

    Returns
    -------
    PolyData
        Parsed mesh data combining Verts/Lines/Strips/Polys sections.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set and the source cannot be mapped, or the file keeps
        its arrays inline, base64-encoded or zlib-compressed.
    CodecError
        If the file holds no ``<PolyData>``; if a section declares offsets
        that run backwards or past the end of its connectivity; or if a
        binary block holds bytes that are not a whole number of the values
        its array declares.
    """
    # The size comes back from the read itself: measuring the source
    # separately costs a whole decompression pass over a compressed one,
    # and a stream that cannot seek cannot be measured at all.
    parsed = parse_xml(path, lazy=lazy, fmt=EXTENSION)
    try:
        return _assemble(parsed, lazy=lazy)
    except BaseException:
        release_appended(parsed[1])
        raise


def _assemble(parsed: Parsed, *, lazy: bool) -> PolyData:
    """Build the mesh of a parsed file.

    Parameters
    ----------
    parsed
        What :func:`parse_xml` handed back.
    lazy
        Whether the arrays view the mapping rather than copy out of it.

    Returns
    -------
    PolyData
        The mesh.
    """
    root, appended, header_type, big_endian, compressed, is_base64, file_size = parsed

    def _decode(elem):
        return decode_da(
            elem,
            big_endian=big_endian,
            appended=appended,
            header_type=header_type,
            compressed=compressed,
            is_base64=is_base64,
            copy=not lazy,
        )

    vtk_type = root.get("type", "PolyData")
    if vtk_type != "PolyData":
        raise UnsupportedFormatError(
            f"VTP file declares type='{vtk_type}'; only 'PolyData' is supported "
            "by the built-in VTP reader. A multi-block dataset is read by "
            "polyxios.helper.read_multiblock(), or kept in pieces by "
            "helper.read_blocks(); see examples/read_multiblock_vtp.py."
        )

    pd_elem = root.find("PolyData")
    if pd_elem is None:
        raise CodecError(".vtp: the file holds no <PolyData> element.")

    all_vertices: list[np.ndarray] = []
    n_joined_points = 0
    all_connectivity: list[np.ndarray] = []
    n_joined_conn = 0
    # Cell ends per section, already shifted to where the section's
    # connectivity lands in the joined array, behind the leading zero of the
    # CSR offsets.
    all_offsets: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    all_types: list[np.ndarray] = []
    all_vertex_attrs: dict[str, list[np.ndarray]] = {}
    all_element_attrs: dict[str, list[np.ndarray]] = {}
    # Whole-mesh metadata. VTK puts the block on the dataset and some writers
    # put it on a Piece, so both are read; the dataset's own is taken last,
    # because that is the one the file means when it holds both. The pieces
    # are read in one pass ahead of the walk below, which is what lets a name
    # two of them spell be reported rather than quietly overwritten.
    pieces = pd_elem.findall("Piece")
    global_attrs: dict[str, Any] = piece_field_data(pieces, _decode)

    for index, piece in enumerate(pieces):
        n_points = piece_count(piece, "NumberOfPoints", fmt=".vtp")

        # Where this piece's points land in the joined array: its cells
        # index its own points from zero. Carried along rather than summed
        # per piece, which walks every piece read so far to answer the same
        # question a running count already holds.
        vert_offset = n_joined_points

        points_elem = piece.find("Points")
        if n_points > 0:
            # A piece that declares points and does not deliver them cannot
            # be dropped quietly: its cells index those points, and every
            # later piece is offset by how many there were.
            da = None if points_elem is None else points_elem.find("DataArray")
            # Asked before decoding, so an array of a type this reader has no
            # numbers for is reported as that rather than as an empty one.
            bad_type = None if da is None else undecodable_type(da)
            if bad_type is not None:
                raise CodecError(
                    f".vtp: Piece {index} declares {n_points} points but"
                    f" its Points array has type '{bad_type}', which holds"
                    " no numbers."
                )
            flat = np.array([]) if da is None else _decode(da)
            # The array has to hold whole tuples as well as enough of
            # them: a size that is not a multiple of the point count has
            # no shape to be read as, and reshape answers that with a
            # ValueError naming neither the file nor the Piece.
            if flat.size < n_points * 3 or flat.size % n_points:
                raise CodecError(
                    f".vtp: Piece {index} declares {n_points} points but"
                    f" its Points array holds {flat.size} values, which is"
                    f" not {n_points} tuples of three or more."
                )
            verts = flat.reshape(n_points, -1)[:, :3]
            all_vertices.append(verts if lazy else verts.astype(np.float64))
            n_joined_points += n_points

        for section in _SECTION_TYPES:
            sect_elem = piece.find(section)
            if sect_elem is None:
                continue
            conn_da = sect_elem.find("DataArray[@Name='connectivity']")
            off_da = sect_elem.find("DataArray[@Name='offsets']")
            if conn_da is None or off_da is None:
                continue

            section_ends = _decode(off_da)
            conn, ends = piece_cells(
                _decode(conn_da),
                section_ends,
                vert_offset,
                n_joined_conn,
                where=f".vtp Piece {index} {section}",
            )
            all_connectivity.append(conn if lazy else conn.astype(np.int32))
            all_offsets.append(ends)
            n_joined_conn += conn.size

            code = _SECTION_CODES.get(section)
            if code is None:
                # Polys: the point count is the only thing that tells a
                # triangle from a quad from a polygon.
                n_nodes = np.diff(section_ends, prepend=0)
                all_types.append(polygon_types(n_nodes))
            else:
                all_types.append(np.full(ends.shape, code, dtype=np.uint8))

        pd_data = piece.find("PointData")
        if pd_data is not None:
            for da in pd_data:
                name = da.get("Name", "unknown")
                arr = _decode(da)
                if arr.size == 0:
                    continue
                arr = shaped_da(da, arr)
                all_vertex_attrs.setdefault(name, []).append(arr)

        cd_data = piece.find("CellData")
        if cd_data is not None:
            for da in cd_data:
                name = da.get("Name", "unknown")
                arr = _decode(da)
                if arr.size == 0:
                    continue
                arr = shaped_da(da, arr)
                all_element_attrs.setdefault(name, []).append(arr)

    vertices, connectivity, offsets, element_types = join_cells(
        all_vertices, all_connectivity, all_offsets, all_types, lazy=lazy
    )

    validate_header(
        vertices.shape[0],
        len(element_types),
        len(connectivity),
        file_size,
        compressed=compressed,
    )

    global_attrs |= read_field_data(pd_elem, _decode)

    vertex_attrs = join_piece_attrs(
        all_vertex_attrs, expected=vertices.shape[0], kind="point"
    )
    element_attrs = join_piece_attrs(
        all_element_attrs, expected=len(element_types), kind="cell"
    )

    # A column named for a tag group is that group's membership rather than an
    # attribute over the entities; the name is the only thing that says so.
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a VTK PolyData XML file (.vtp).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default), encode arrays as base64 binary.
    appended
        If True, write the arrays as one raw appended section after the XML
        instead of inline. A third smaller than base64, and the one layout
        ``read(..., lazy=True)`` can map. Field data stays inline.
    """
    binary: bool = bool(opts.get("binary", True))
    blocks: list[bytes] | None = [] if opts.get("appended", False) else None

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    points = poly.vertices.ravel().astype(np.float64)
    conn = poly.connectivity.astype(np.int32)
    off = poly.offsets[1:].astype(np.int32)

    # A tag group travels as one column of ones and zeros named for it: the
    # channel holds one value per entity, and an element in two groups is
    # named by both columns, which one label per element cannot say.
    point_arrays = spellable_arrays(
        with_tag_masks(
            poly.vertex_attrs,
            poly.vertex_tags,
            poly.vertices.shape[0],
            fmt=EXTENSION,
            kind="point",
        ),
        fmt=EXTENSION,
        kind="point",
    )
    cell_arrays = spellable_arrays(
        with_tag_masks(
            poly.element_attrs,
            poly.element_tags,
            len(poly.element_types),
            fmt=EXTENSION,
            kind="cell",
        ),
        fmt=EXTENSION,
        kind="cell",
    )

    field_arrays = globals_for_write(poly, fmt=EXTENSION, text=True)

    # Decided before any array is rendered: the width of every block's byte
    # count is declared once on the root element and every offset counts it.
    header_type = appended_header_type(
        sizes=[
            points.nbytes,
            conn.nbytes,
            off.nbytes,
            *(attr_nbytes(arr=arr) for arr in point_arrays.values()),
            *(attr_nbytes(arr=arr) for arr in cell_arrays.values()),
            *(attr_nbytes(arr=arr) for arr in field_arrays.values()),
        ]
    )

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian"'
        f"{header_type_attr(header_type=header_type)}>"
    )
    lines.append("  <PolyData>")
    lines.extend(
        format_field_data(
            field_arrays,
            text=text_for_write(poly),
            binary=binary,
            indent=4,
            fmt=EXTENSION,
            header_type=header_type,
        )
    )

    n_polys = n_elems  # write all as Polys for generality

    lines.append(
        f'    <Piece NumberOfPoints="{n_verts}" NumberOfVerts="0" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{n_polys}">'
    )

    lines.append("      <Points>")
    lines.append(_da("", points, "Float64", binary, 3, 10, blocks, header_type))
    lines.append("      </Points>")

    lines.append("      <Polys>")
    lines.append(_da("connectivity", conn, "Int32", binary, 1, 10, blocks, header_type))
    lines.append(_da("offsets", off, "Int32", binary, 1, 10, blocks, header_type))
    lines.append("      </Polys>")

    if point_arrays:
        lines.append("      <PointData>")
        for name, arr in point_arrays.items():
            lines.append(
                format_attr_da(
                    name,
                    arr,
                    binary=binary,
                    indent=10,
                    appended=blocks,
                    header_type=header_type,
                )
            )
        lines.append("      </PointData>")

    if cell_arrays:
        lines.append("      <CellData>")
        for name, arr in cell_arrays.items():
            lines.append(
                format_attr_da(
                    name,
                    arr,
                    binary=binary,
                    indent=10,
                    appended=blocks,
                    header_type=header_type,
                )
            )
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </PolyData>")

    if blocks is None:
        lines.append("</VTKFile>")
        write_text(path, "\n".join(lines), encoding="utf-8")
        return
    head = ("\n".join(lines) + "\n").encode("utf-8")
    write_bytes(path, head + appended_section(blocks, tail="</VTKFile>"))


def _da(
    name: str,
    arr: np.ndarray,
    vtk_type: str,
    binary: bool,
    n_comp: int,
    indent: int,
    appended: list[bytes] | None = None,
    header_type: str = "UInt32",
) -> str:
    """Render one ``<DataArray>`` element.

    Parameters
    ----------
    name
        Array name; empty for the unnamed ``Points`` array.
    arr
        Values, flat or one row per tuple.
    vtk_type
        The type name the element declares. The values are cast to the dtype
        it names, so the bytes are what the header says they are on a
        big-endian machine as much as on a little-endian one.
    binary
        Base64 the raw bytes instead of spelling the numbers.
    n_comp
        Components per tuple.
    indent
        Spaces to prefix the line with.
    appended
        Block list to write the bytes to instead, as for ``format_da``.
    header_type
        Width of every block's byte count, as for ``format_da``.

    Returns
    -------
    str
        The ``<DataArray>`` line.
    """
    return format_da(
        name,
        arr,
        vtk_type=vtk_type,
        dtype=np.dtype("<" + (vtk_type_to_np(vtk_type) or "f8")),
        binary=binary,
        n_comp=n_comp,
        indent=indent,
        appended=appended,
        header_type=header_type,
    )
