from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._globals import globals_for_write, text_for_write
from polyxios._io import Source, write_text
from polyxios._tags import tags_from_masks, with_tag_masks
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    decode_da,
    extent_points,
    extent_spans,
    format_attr_da,
    format_da,
    format_field_data,
    parse_xml,
    read_field_data,
    require_structured_cells,
    shaped_da,
    sized_attrs,
    spellable_arrays,
    structured_cell_shape,
    structured_cells,
    structured_cells_fit,
    structured_spans_from_cells,
    vtk_type_to_np,
    whole_extent,
    xml_extent,
)
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vts"

# The keys this codec spells from the grid itself on the way out, so they
# never travel as field data - a second copy of the grid, in the wrong shape.
RESERVED_GLOBALS: frozenset[str] = frozenset({"vts_extent", "vts_whole_extent"})


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a VTK StructuredGrid XML file (.vts) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vts file.
    lazy
        Deferred decoding is not supported; raises LazyReadError when True.

    Returns
    -------
    PolyData
        Structured grid expanded to explicit hex connectivity.

    Raises
    ------
    LazyReadError
        If lazy=True.
    """
    if lazy:
        raise LazyReadError("VTS lazy reads are not supported with frozen PolyData.")

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

    sg = root.find("StructuredGrid")
    if sg is None:
        raise ValueError("No <StructuredGrid> element found in VTS file.")

    piece = sg.find("Piece")
    if piece is None:
        raise ValueError("No <Piece> element found.")

    extent_str = piece.get("Extent", sg.get("WholeExtent", "0 1 0 1 0 1"))
    extent = xml_extent(extent_str, fmt=".vts", where="Extent")
    i0, i1, j0, j1, k0, k1 = extent
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    n_verts = extent_points((nx, ny, nz))
    # An extent flat along an axis is a sheet of quads or a run of lines, not
    # a grid of no cells: the cells the grid holds decide the shape of a
    # CellData array, so they are counted before the header is validated.
    n_cells, n_per_cell, cell_kind = structured_cell_shape(nx, ny, nz)

    # A StructuredGrid writes its points, so their size is evidence; its
    # cells are the extent expanded and are written nowhere.
    validate_header(
        n_verts,
        n_cells,
        n_cells * n_per_cell,
        file_size,
        compressed=compressed,
        spells_connectivity=False,
    )

    points_elem = piece.find("Points")
    if points_elem is None:
        raise ValueError("No <Points> element found.")
    da = points_elem.find("DataArray")
    if da is None:
        raise ValueError("No <DataArray> under <Points>.")

    flat = _decode(da)
    if not n_verts:
        # An extent of "0 -1 0 -1 0 -1" is how VTK spells a grid with no
        # points, and what this writer spells for a mesh with none. reshape
        # cannot infer a column count from an empty run, so the file this
        # codec wrote itself came back as a ValueError about newaxis.
        if flat.size:
            raise CodecError(
                f"{EXTENSION}: Extent='{extent_str}' spans no points and the "
                f"<Points> array holds {flat.size} values."
            )
        vertices = np.zeros((0, 3), dtype=np.float64)
    elif flat.size % n_verts or flat.size // n_verts < 3:
        # A StructuredGrid carries its points, so the extent and the array
        # have to agree on how many. The column count is inferred from the
        # two, so a file that disagrees came back as a mesh of the wrong
        # width - a point of no coordinates at all, where the extent declared
        # one the array did not carry - or as a bare ValueError from inside
        # numpy naming neither the file nor the extent.
        raise CodecError(
            f"{EXTENSION}: Extent='{extent_str}' declares {n_verts} points and "
            f"the <Points> array holds {flat.size} values, which is not three "
            "or more coordinates each."
        )
    else:
        vertices = flat.reshape(n_verts, -1)[:, :3].astype(np.float64)

    connectivity, _, _ = structured_cells(nx, ny, nz)
    offsets = np.arange(n_cells + 1, dtype=np.int32) * n_per_cell
    element_types = np.full(n_cells, ELEMENT_TYPES[cell_kind], dtype=np.uint8)

    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    pd = piece.find("PointData")
    if pd is not None:
        for da in pd:
            arr = _decode(da)
            if arr.size > 0:
                point_data[da.get("Name", "unknown")] = shaped_da(da, arr)

    cd = piece.find("CellData")
    if cd is not None:
        for da in cd:
            arr = _decode(da)
            if arr.size > 0:
                cell_data[da.get("Name", "unknown")] = shaped_da(da, arr)

    vertex_attrs = sized_attrs(point_data, expected=n_verts, kind="point")
    element_attrs = sized_attrs(cell_data, expected=n_cells, kind="cell")

    # Whole-mesh metadata, under the grid the reader rebuilt: a file naming
    # one of the reader's own keys in its field data is describing the same
    # grid twice, and the grid is what the points were laid out on. The
    # dataset's own block is read last, as in every other codec here: a key
    # spelled on both levels means what the dataset says it means.
    global_attrs: dict[str, Any] = read_field_data(piece, _decode)
    global_attrs |= read_field_data(sg, _decode)
    global_attrs |= {"vts_extent": extent}
    whole = sg.get("WholeExtent")
    if whole:
        # Read through the same guard the piece extent is: unpacked straight
        # into ints, a malformed one raised a bare ValueError about a literal,
        # naming neither the file nor the attribute it came out of.
        global_attrs["vts_whole_extent"] = xml_extent(
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
    """Serialise a structured PolyData to a VTK StructuredGrid XML file (.vts).

    Parameters
    ----------
    poly
        PolyData to write. Its cells must be a structured grid - hexahedra,
        or the quadrilaterals or lines a grid flat along one or two axes is
        made of - since the file carries no connectivity and the reader
        rebuilds it from the extent. Its points need not be a lattice: they
        are written in the mesh's own order, so a warped block, a cylindrical
        shell and an aerofoil O-grid all hold.
    path
        Output file path.
    binary
        If True (default), encode data as base64 binary.
    """
    binary: bool = bool(opts.get("binary", True))

    ga = poly.global_attrs or {}
    stored = ga.get("vts_extent")
    spans = extent_spans(stored) if stored is not None else None

    # A StructuredGrid writes its points, so the extent says only which of
    # them the cells join - and the cells are the one thing the file does not
    # carry, being rebuilt from the extent on the way back in. The extent the
    # mesh was read with is kept while it is still the mesh's own, so a grid
    # that did not begin at zero goes back where it stood; a transform since
    # then leaves it describing the grid the mesh used to be, and what the
    # mesh is now is read off its cells instead.
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
        whole = ga.get("vts_whole_extent")
    else:
        whole = None
        # Off the cells rather than off the coordinates: the points of a
        # StructuredGrid need not be a lattice at all - a warped block, a
        # cylindrical shell and an aerofoil O-grid are all StructuredGrids,
        # and holding those is why the format exists next to .vti - but the
        # cells are a grid whatever the points do.
        spans = structured_spans_from_cells(
            poly.vertices, poly.connectivity, poly.element_types, fmt=EXTENSION
        )
        require_structured_cells(
            poly.connectivity, poly.element_types, spans, fmt=EXTENSION
        )
        extent = [bound for span in spans for bound in (0, span)]

    ext_str = " ".join(str(v) for v in extent)
    whole_str = " ".join(str(v) for v in whole_extent(whole, extent))

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">'
    )
    lines.append(f'  <StructuredGrid WholeExtent="{whole_str}">')
    lines.extend(
        format_field_data(
            globals_for_write(
                poly, reserved=RESERVED_GLOBALS, fmt=EXTENSION, text=True
            ),
            text=text_for_write(poly, reserved=RESERVED_GLOBALS),
            binary=binary,
            indent=4,
            fmt=EXTENSION,
        )
    )
    lines.append(f'    <Piece Extent="{ext_str}">')

    lines.append("      <Points>")
    lines.append(
        _da("", poly.vertices.ravel().astype(np.float64), "Float64", binary, 3, 10)
    )
    lines.append("      </Points>")

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

    if point_arrays:
        lines.append("      <PointData>")
        for name, arr in point_arrays.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </PointData>")

    if cell_arrays:
        lines.append("      <CellData>")
        for name, arr in cell_arrays.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </StructuredGrid>")
    lines.append("</VTKFile>")

    write_text(path, "\n".join(lines), encoding="utf-8")


def _da(
    name: str,
    arr: np.ndarray,
    vtk_type: str,
    binary: bool,
    n_comp: int,
    indent: int,
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
    )
