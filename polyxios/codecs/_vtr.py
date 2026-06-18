import base64
from pathlib import Path
from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import decode_da, parse_xml
from polyxios.exceptions import LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vtr"


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
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

    path = Path(path)
    file_size = path.stat().st_size

    root, appended, header_type, big_endian, compressed, is_base64 = parse_xml(path)

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
    extent = [int(x) for x in extent_str.split()]
    i0, i1, j0, j1, k0, k1 = extent
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    n_verts = (nx + 1) * (ny + 1) * (nz + 1)
    n_cells = nx * ny * nz

    validate_header(n_verts, n_cells, n_cells * 8, file_size, compressed=compressed)

    coords_elem = piece.find("Coordinates")
    if coords_elem is None:
        raise ValueError("No <Coordinates> element found.")

    coord_arrays = list(coords_elem)
    x_arr = _decode(coord_arrays[0]) if len(coord_arrays) > 0 else np.array([0.0])
    y_arr = _decode(coord_arrays[1]) if len(coord_arrays) > 1 else np.array([0.0])
    z_arr = _decode(coord_arrays[2]) if len(coord_arrays) > 2 else np.array([0.0])

    zz, yy, xx = np.meshgrid(z_arr, y_arr, x_arr, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    nxp1 = nx + 1
    nyp1 = ny + 1
    connectivity = np.empty(n_cells * 8, dtype=np.int32)
    offsets = np.arange(0, (n_cells + 1) * 8, 8, dtype=np.int32)
    element_types = np.full(n_cells, ELEMENT_TYPES["hexahedron"], dtype=np.uint8)

    cell_idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                v0 = ix + iy * nxp1 + iz * nxp1 * nyp1
                v1 = v0 + 1
                v2 = v0 + 1 + nxp1
                v3 = v0 + nxp1
                v4 = v0 + nxp1 * nyp1
                v5 = v4 + 1
                v6 = v4 + 1 + nxp1
                v7 = v4 + nxp1
                ci = cell_idx * 8
                connectivity[ci : ci + 8] = [v0, v1, v2, v3, v4, v5, v6, v7]
                cell_idx += 1

    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}

    pd = piece.find("PointData")
    if pd is not None:
        for da in pd:
            arr = _decode(da)
            name = da.get("Name", "unknown")
            vertex_attrs[name] = arr

    cd = piece.find("CellData")
    if cd is not None:
        for da in cd:
            arr = _decode(da)
            name = da.get("Name", "unknown")
            element_attrs[name] = arr

    global_attrs: dict[str, Any] = {"vtr_extents": extent}
    whole = rg.get("WholeExtent")
    if whole:
        global_attrs["vtr_whole_extent"] = [int(x) for x in whole.split()]

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        global_attrs=global_attrs,
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a VTK rectilinear grid XML file (.vtr).

    Parameters
    ----------
    poly
        PolyData to write. Must consist of hexahedral elements on a structured grid.
        The vertices are written as coordinate arrays.
    path
        Output file path.
    binary
        If True (default: False), encode data as base64 binary.
    """
    path = Path(path)
    binary: bool = bool(opts.get("binary", False))

    x_coords = np.unique(poly.vertices[:, 0])
    y_coords = np.unique(poly.vertices[:, 1])
    z_coords = np.unique(poly.vertices[:, 2])

    nx = len(x_coords) - 1
    ny = len(y_coords) - 1
    nz = len(z_coords) - 1

    extent_str = f"0 {nx} 0 {ny} 0 {nz}"

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    bo = "LittleEndian"
    lines.append(f'<VTKFile type="RectilinearGrid" version="0.1" byte_order="{bo}">')
    lines.append(f'  <RectilinearGrid WholeExtent="{extent_str}">')
    lines.append(f'    <Piece Extent="{extent_str}">')
    lines.append("      <Coordinates>")
    lines.append(_format_data_array("x_coordinates", x_coords, binary, 8))
    lines.append(_format_data_array("y_coordinates", y_coords, binary, 8))
    lines.append(_format_data_array("z_coordinates", z_coords, binary, 8))
    lines.append("      </Coordinates>")

    if poly.vertex_attrs:
        lines.append("      <PointData>")
        for name, arr in poly.vertex_attrs.items():
            lines.append(_format_data_array(name, arr.ravel(), binary, 8))
        lines.append("      </PointData>")

    if poly.element_attrs:
        lines.append("      <CellData>")
        for name, arr in poly.element_attrs.items():
            lines.append(_format_data_array(name, arr.ravel(), binary, 8))
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </RectilinearGrid>")
    lines.append("</VTKFile>")

    path.write_text("\n".join(lines), encoding="utf-8")


def _format_data_array(name: str, arr: np.ndarray, binary: bool, indent: int) -> str:
    pad = " " * indent
    vtk_type = _np_to_vtk_type(arr.dtype)

    if binary:
        raw = arr.astype("<f8").tobytes()
        length = np.array([len(raw)], dtype="<u4").tobytes()
        encoded = base64.b64encode(length + raw).decode()
        return f'{pad}<DataArray type="{vtk_type}" Name="{name}" format="binary">{encoded}</DataArray>'
    else:
        vals = " ".join(f"{v:.10g}" for v in arr.ravel())
        return f'{pad}<DataArray type="{vtk_type}" Name="{name}" format="ascii">{vals}</DataArray>'


def _np_to_vtk_type(dt: np.dtype) -> str:
    mapping = {
        "f4": "Float32",
        "f8": "Float64",
        "i1": "Int8",
        "i2": "Int16",
        "i4": "Int32",
        "i8": "Int64",
        "u1": "UInt8",
        "u2": "UInt16",
        "u4": "UInt32",
        "u8": "UInt64",
    }
    return mapping.get(dt.str.lstrip("<>|="), "Float64")
