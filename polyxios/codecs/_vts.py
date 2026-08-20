from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import Source, write_text
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    decode_da,
    format_attr_da,
    format_da,
    parse_xml,
    shaped_da,
    sized_attrs,
    structured_hexahedra,
    vtk_type_to_np,
)
from polyxios.exceptions import LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vts"


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
    extent = [int(v) for v in extent_str.split()]
    i0, i1, j0, j1, k0, k1 = extent
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    n_verts = (nx + 1) * (ny + 1) * (nz + 1)
    n_cells = nx * ny * nz

    validate_header(n_verts, n_cells, n_cells * 8, file_size, compressed=compressed)

    points_elem = piece.find("Points")
    if points_elem is None:
        raise ValueError("No <Points> element found.")
    da = points_elem.find("DataArray")
    if da is None:
        raise ValueError("No <DataArray> under <Points>.")

    flat = _decode(da)
    vertices = flat.reshape(n_verts, -1)[:, :3].astype(np.float64)

    connectivity = structured_hexahedra(nx, ny, nz)
    offsets = np.arange(0, (n_cells + 1) * 8, 8, dtype=np.int32)
    element_types = np.full(n_cells, ELEMENT_TYPES["hexahedron"], dtype=np.uint8)

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

    global_attrs: dict[str, Any] = {"vts_extent": extent}
    whole = sg.get("WholeExtent")
    if whole:
        global_attrs["vts_whole_extent"] = [int(v) for v in whole.split()]

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        global_attrs=global_attrs,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise a hex PolyData to a VTK StructuredGrid XML file (.vts).

    Parameters
    ----------
    poly
        PolyData to write. Must be a structured hex grid.
    path
        Output file path.
    binary
        If True (default), encode data as base64 binary.
    """
    binary: bool = bool(opts.get("binary", True))

    ga = poly.global_attrs or {}
    extent = ga.get("vts_extent")
    if extent is None:
        xu = np.unique(poly.vertices[:, 0])
        yu = np.unique(poly.vertices[:, 1])
        zu = np.unique(poly.vertices[:, 2])
        extent = [0, len(xu) - 1, 0, len(yu) - 1, 0, len(zu) - 1]

    ext_str = " ".join(str(v) for v in extent)

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">'
    )
    lines.append(f'  <StructuredGrid WholeExtent="{ext_str}">')
    lines.append(f'    <Piece Extent="{ext_str}">')

    lines.append("      <Points>")
    lines.append(
        _da("", poly.vertices.ravel().astype(np.float64), "Float64", binary, 3, 10)
    )
    lines.append("      </Points>")

    if poly.vertex_attrs:
        lines.append("      <PointData>")
        for name, arr in poly.vertex_attrs.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </PointData>")

    if poly.element_attrs:
        lines.append("      <CellData>")
        for name, arr in poly.element_attrs.items():
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
