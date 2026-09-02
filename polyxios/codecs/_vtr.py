from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._globals import globals_for_write
from polyxios._io import Source, write_text
from polyxios._tags import mask_arrays, tags_from_masks
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    decode_da,
    extent_points,
    extent_spans,
    format_attr_da,
    format_field_data,
    grid_axes,
    parse_xml,
    read_field_data,
    require_grid_order,
    require_structured_cells,
    shaped_da,
    sized_attrs,
    structured_cell_shape,
    structured_cells,
    structured_cells_fit,
    structured_spans_from_cells,
    whole_extent,
    xml_extent,
)
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vtr"

# The keys this codec spells from the grid itself on the way out, so they
# never travel as field data - a second copy of the grid, in the wrong shape.
RESERVED_GLOBALS: frozenset[str] = frozenset({"vtr_extents", "vtr_whole_extent"})


def _implied(span: int) -> np.ndarray:
    """The coordinates an axis the file left no array for is read with.

    A 2-D RectilinearGrid writes two arrays and leaves the third to the
    extent, which gives that axis its one plane at zero. An axis the extent
    gives no plane at all - an end before its start - gets no coordinate
    rather than one, so it reads as the empty axis it is rather than as an
    array disagreeing with the extent.

    Parameters
    ----------
    span
        Cells the extent runs along the axis.

    Returns
    -------
    numpy.ndarray
        One zero, or none.

    Examples
    --------
    >>> _implied(0).tolist()
    [0.0]
    >>> _implied(-1).tolist()
    []
    """
    return np.zeros(1 if span >= 0 else 0, dtype=np.float64)


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a VTK rectilinear grid XML file (.vtr) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtr file.
    lazy
        If True, defer array decoding until array is accessed.
        Arrays are stored as bytes in global_attrs and decoded on first use.
        NOTE: In the current implementation, lazy=True raises LazyReadError because
        PolyData is immutable and cannot store deferred arrays.

    Returns
    -------
    PolyData
        Parsed mesh data with structured grid expanded to hex connectivity.

    Raises
    ------
    LazyReadError
        If lazy=True (VTR lazy reads not yet supported in frozen PolyData).
    """
    if lazy:
        raise LazyReadError(
            "VTR lazy reads require mutable array proxies; not supported with frozen PolyData."
        )

    # The size comes back from the read itself: measuring the source
    # separately costs a whole decompression pass over a compressed one,
    # and a stream that cannot seek cannot be measured at all.
    (
        root,
        appended,
        header_type,
        big_endian,
        compressed,
        is_base64,
        file_size,
    ) = parse_xml(path)

    def _decode(elem):
        return decode_da(
            elem,
            big_endian=big_endian,
            appended=appended,
            header_type=header_type,
            compressed=compressed,
            is_base64=is_base64,
        )

    rg = root.find("RectilinearGrid")
    if rg is None:
        raise ValueError("No <RectilinearGrid> element found in VTR file.")

    piece = rg.find("Piece")
    if piece is None:
        raise ValueError("No <Piece> element found.")

    extent_str = piece.get("Extent", "0 0 0 0 0 0")
    extent = xml_extent(extent_str, fmt=".vtr", where="Extent")
    i0, i1, j0, j1, k0, k1 = extent
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    n_verts = extent_points((nx, ny, nz))
    # An extent flat along an axis is a sheet of quads or a run of lines, not
    # a grid of no cells: the cells the grid holds decide the shape of a
    # CellData array, so they are counted before the header is validated.
    n_cells, n_per_cell, cell_kind = structured_cell_shape(nx, ny, nz)

    # The file spells neither the points nor the cells: they are the extent
    # expanded, so the byte-size heuristic has nothing to weigh the counts
    # against and refused a bare 4 x 4 x 4 grid this codec had just written.
    validate_header(
        n_verts,
        n_cells,
        n_cells * n_per_cell,
        file_size,
        compressed=compressed,
        spells_vertices=False,
        spells_connectivity=False,
    )

    coords_elem = piece.find("Coordinates")
    if coords_elem is None:
        raise ValueError("No <Coordinates> element found.")

    coord_arrays = list(coords_elem)
    x_arr, y_arr, z_arr = (
        _decode(coord_arrays[axis]) if axis < len(coord_arrays) else _implied(span)
        for axis, span in enumerate((nx, ny, nz))
    )

    # The extent counts the planes and the arrays spell them, and nothing in
    # the file makes the two agree. A longer array expands to more vertices
    # than the extent declared, and the cells, the offsets and every
    # PointData array are all sized off the extent - so the mesh came back
    # with connectivity over part of itself and attributes covering none of
    # it, past every check the reader makes.
    # Asked of every axis, including those of an extent that ends before it
    # starts: the point count is a product and goes to zero on one such axis,
    # but the coordinates are the file's own and the mesh is built from them,
    # so an axis left unchecked expanded to a mesh the extent said was empty.
    for axis, (coords, span) in enumerate(
        zip((x_arr, y_arr, z_arr), (nx, ny, nz), strict=True)
    ):
        if len(coords) != span + 1:
            raise CodecError(
                f"{EXTENSION}: Extent='{extent_str}' puts {max(span + 1, 0)} "
                f"planes on the {'xyz'[axis]} axis and its coordinate array "
                f"holds {len(coords)} values."
            )

    zz, yy, xx = np.meshgrid(z_arr, y_arr, x_arr, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    connectivity, _, _ = structured_cells(nx, ny, nz)
    offsets = np.arange(n_cells + 1, dtype=np.int32) * n_per_cell
    element_types = np.full(n_cells, ELEMENT_TYPES[cell_kind], dtype=np.uint8)

    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    pd = piece.find("PointData")
    if pd is not None:
        for da in pd:
            point_data[da.get("Name", "unknown")] = shaped_da(da, _decode(da))

    cd = piece.find("CellData")
    if cd is not None:
        for da in cd:
            cell_data[da.get("Name", "unknown")] = shaped_da(da, _decode(da))

    vertex_attrs = sized_attrs(point_data, expected=n_verts, kind="point")
    element_attrs = sized_attrs(cell_data, expected=n_cells, kind="cell")

    # Whole-mesh metadata, under the grid the reader rebuilt: a file naming
    # one of the reader's own keys in its field data is describing the same
    # grid twice, and the grid is what the points were laid out on.
    global_attrs: dict[str, Any] = read_field_data(rg, _decode)
    global_attrs |= read_field_data(piece, _decode)
    global_attrs |= {"vtr_extents": extent}
    whole = rg.get("WholeExtent")
    if whole:
        # Read through the same guard the piece extent is: unpacked straight
        # into ints, a malformed one raised a bare ValueError about a literal,
        # naming neither the file nor the attribute it came out of.
        global_attrs["vtr_whole_extent"] = xml_extent(
            whole, fmt=EXTENSION, where="WholeExtent"
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
    """Serialise PolyData to a VTK rectilinear grid XML file (.vtr).

    Parameters
    ----------
    poly
        PolyData to write. Its cells must be a structured grid - hexahedra,
        or the quadrilaterals or lines a grid flat along one or two axes is
        made of - and its vertices must be that grid expanded, with x varying
        fastest and z slowest. The file holds three coordinate arrays and no
        points of its own, so a mesh that is not that lattice cannot be
        spelled in it.
    path
        Output file path.
    binary
        If True (default: False), encode data as base64 binary.
    """
    binary: bool = bool(opts.get("binary", False))

    # The file holds three coordinate arrays and no points, so the mesh has to
    # be the grid they expand to: its cells give the shape, the coordinates
    # are read off it a stride at a time, and the vertices are checked against
    # the product before any of it is written. A scattered mesh has no three
    # axes that describe it, and a grid in another order would come back with
    # every point attribute on a different point.
    #
    # The extent the mesh was read with is kept while it still describes the
    # mesh, the way ``.vti`` and ``.vts`` keep theirs, so a block that did not
    # begin at zero goes back at the indices it stood on rather than being
    # slid to the origin - which is the one thing a ``.pvtr`` assembling it
    # next to its neighbours reads. A transform since the read leaves it
    # describing the grid the mesh used to be, and what the mesh is now is
    # read off its cells instead.
    ga = poly.global_attrs or {}
    stored = ga.get("vtr_extents")
    spans = extent_spans(stored) if stored is not None else None
    if (
        spans is not None
        and extent_points(spans) == len(poly.vertices)
        and structured_cells_fit(poly.connectivity, poly.element_types, spans)
    ):
        extent = [int(v) for v in stored]
        # The grid this piece belongs to means something only while the piece
        # is still standing where it stood. An extent re-derived below is
        # zero-based and says nothing about the old grid, so the stored one
        # goes with the extent it was read beside.
        whole = ga.get("vtr_whole_extent")
    else:
        whole = None
        spans = structured_spans_from_cells(
            poly.vertices, poly.connectivity, poly.element_types, fmt=EXTENSION
        )
        require_structured_cells(
            poly.connectivity, poly.element_types, spans, fmt=EXTENSION
        )
        extent = [bound for span in spans for bound in (0, span)]

    # An extent that gives one axis no plane at all empties the whole piece,
    # and the coordinates are read off the vertices a stride at a time - so
    # there is no vertex to stride through and the other two axes come back
    # empty as well. Written under an extent that still counts their planes,
    # the file went out with coordinate arrays disagreeing with their own
    # extent, and this reader refused it. A piece holding no points is spelled
    # the way an empty mesh is spelled, on every axis; the grid it belonged to
    # goes with the indices it stood on, the way a re-derived extent's does.
    if extent_points(spans) == 0:
        spans = [-1, -1, -1]
        extent = [bound for span in spans for bound in (0, span)]
        whole = None

    x_coords, y_coords, z_coords = grid_axes(poly.vertices, spans)
    require_grid_order(poly.vertices, (x_coords, y_coords, z_coords), fmt=EXTENSION)

    extent_str = " ".join(str(v) for v in extent)
    whole_str = " ".join(str(v) for v in whole_extent(whole, extent))

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    bo = "LittleEndian"
    lines.append(f'<VTKFile type="RectilinearGrid" version="1.0" byte_order="{bo}">')
    lines.append(f'  <RectilinearGrid WholeExtent="{whole_str}">')
    lines.extend(
        format_field_data(
            globals_for_write(poly, reserved=RESERVED_GLOBALS, fmt=EXTENSION),
            binary=binary,
            indent=4,
        )
    )
    lines.append(f'    <Piece Extent="{extent_str}">')
    lines.append("      <Coordinates>")
    lines.append(_format_data_array("x_coordinates", x_coords, binary, 8))
    lines.append(_format_data_array("y_coordinates", y_coords, binary, 8))
    lines.append(_format_data_array("z_coordinates", z_coords, binary, 8))
    lines.append("      </Coordinates>")

    # A tag group travels as one column of ones and zeros named for it: the
    # channel holds one value per entity, and an element in two groups is
    # named by both columns, which one label per element cannot say.
    point_arrays = poly.vertex_attrs | mask_arrays(
        poly.vertex_tags, poly.vertices.shape[0], fmt=EXTENSION, kind="point"
    )
    cell_arrays = poly.element_attrs | mask_arrays(
        poly.element_tags, len(poly.element_types), fmt=EXTENSION, kind="cell"
    )

    if point_arrays:
        lines.append("      <PointData>")
        for name, arr in point_arrays.items():
            lines.append(_format_data_array(name, arr, binary, 8))
        lines.append("      </PointData>")

    if cell_arrays:
        lines.append("      <CellData>")
        for name, arr in cell_arrays.items():
            lines.append(_format_data_array(name, arr, binary, 8))
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </RectilinearGrid>")
    lines.append("</VTKFile>")

    write_text(path, "\n".join(lines), encoding="utf-8")


def _format_data_array(name: str, arr: np.ndarray, binary: bool, indent: int) -> str:
    """Render one DataArray element in the type the array is held in.

    Parameters
    ----------
    name
        Array name.
    arr
        Values. Every dimension past the first is the component count, which
        the element has to declare or a reader has no way to cut the flat run
        back into tuples.
    binary
        Base64 the raw bytes instead of spelling the numbers.
    indent
        Spaces to prefix the line with.

    Returns
    -------
    str
        The ``<DataArray>`` line.
    """
    return format_attr_da(name, arr, binary=binary, indent=indent)
