import base64
from pathlib import Path
from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import decode_da, parse_xml
from polyxios.exceptions import LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vti"


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a VTK ImageData XML file (.vti) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vti file.
    lazy
        Deferred decoding is not supported; raises LazyReadError when True.

    Returns
    -------
    PolyData
        Uniform hex grid expanded from the ImageData metadata.

    Raises
    ------
    LazyReadError
        If lazy=True.
    """
    if lazy:
        raise LazyReadError("VTI lazy reads are not supported with frozen PolyData.")

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

    img = root.find("ImageData")
    if img is None:
        raise ValueError("No <ImageData> element found in VTI file.")

    # WholeExtent="x0 x1 y0 y1 z0 z1" gives the global grid dimensions
    whole_extent_str = img.get("WholeExtent", "0 1 0 1 0 1")
    origin_str = img.get("Origin", "0.0 0.0 0.0")
    spacing_str = img.get("Spacing", "1.0 1.0 1.0")
    origin = [float(v) for v in origin_str.split()]
    spacing = [float(v) for v in spacing_str.split()]

    piece = img.find("Piece")
    if piece is None:
        raise ValueError("No <Piece> element found.")

    piece_extent_str = piece.get("Extent", whole_extent_str)
    pe = [int(v) for v in piece_extent_str.split()]
    i0, i1, j0, j1, k0, k1 = pe
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    n_verts = (nx + 1) * (ny + 1) * (nz + 1)
    n_cells = nx * ny * nz

    validate_header(n_verts, n_cells, n_cells * 8, file_size, compressed=compressed)

    x = origin[0] + np.arange(i0, i1 + 1) * spacing[0]
    y = origin[1] + np.arange(j0, j1 + 1) * spacing[1]
    z = origin[2] + np.arange(k0, k1 + 1) * spacing[2]

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    nxp1, nyp1 = nx + 1, ny + 1
    connectivity = np.empty(n_cells * 8, dtype=np.int32)
    offsets = np.arange(0, (n_cells + 1) * 8, 8, dtype=np.int32)
    element_types = np.full(n_cells, ELEMENT_TYPES["hexahedron"], dtype=np.uint8)

    cell_idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                v0 = ix + iy * nxp1 + iz * nxp1 * nyp1
                v1, v2, v3 = v0 + 1, v0 + 1 + nxp1, v0 + nxp1
                v4 = v0 + nxp1 * nyp1
                v5, v6, v7 = v4 + 1, v4 + 1 + nxp1, v4 + nxp1
                ci = cell_idx * 8
                connectivity[ci : ci + 8] = [v0, v1, v2, v3, v4, v5, v6, v7]
                cell_idx += 1

    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}

    pd = piece.find("PointData")
    if pd is not None:
        for da in pd:
            arr = _decode(da)
            if arr.size > 0:
                vertex_attrs[da.get("Name", "unknown")] = arr

    cd = piece.find("CellData")
    if cd is not None:
        for da in cd:
            arr = _decode(da)
            if arr.size > 0:
                element_attrs[da.get("Name", "unknown")] = arr

    global_attrs: dict[str, Any] = {
        "vti_origin": origin,
        "vti_spacing": spacing,
        "vti_extent": pe,
    }

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
    """Serialise a hex PolyData to a VTK ImageData XML file (.vti).

    Parameters
    ----------
    poly
        PolyData to write. Must be a structured hex grid (same topology as
        produced by reading a .vti file).
    path
        Output file path.
    binary
        If True (default), encode data as base64 binary.
    """
    path = Path(path)
    binary: bool = bool(opts.get("binary", True))

    # Recover grid metadata stored by read() or compute from vertex coords
    ga = poly.global_attrs or {}
    origin = ga.get("vti_origin", [0.0, 0.0, 0.0])
    spacing = ga.get("vti_spacing", [1.0, 1.0, 1.0])
    extent = ga.get("vti_extent")

    if extent is None:
        # Infer from unique coordinate values
        xu = np.unique(poly.vertices[:, 0])
        yu = np.unique(poly.vertices[:, 1])
        zu = np.unique(poly.vertices[:, 2])
        extent = [0, len(xu) - 1, 0, len(yu) - 1, 0, len(zu) - 1]
        if len(xu) > 1:
            spacing = [float(xu[1] - xu[0]), float(yu[1] - yu[0]), float(zu[1] - zu[0])]
        origin = [float(xu[0]), float(yu[0]), float(zu[0])]

    ext_str = " ".join(str(v) for v in extent)
    orig_str = " ".join(f"{v:.10g}" for v in origin)
    spac_str = " ".join(f"{v:.10g}" for v in spacing)

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">')
    lines.append(
        f'  <ImageData WholeExtent="{ext_str}" Origin="{orig_str}" Spacing="{spac_str}">'
    )
    lines.append(f'    <Piece Extent="{ext_str}">')

    if poly.vertex_attrs:
        lines.append("      <PointData>")
        for name, arr in poly.vertex_attrs.items():
            n_comp = arr.shape[1] if arr.ndim == 2 else 1
            lines.append(
                _da(name, arr.ravel().astype(np.float64), "Float64", binary, n_comp, 10)
            )
        lines.append("      </PointData>")

    if poly.element_attrs:
        lines.append("      <CellData>")
        for name, arr in poly.element_attrs.items():
            n_comp = arr.shape[1] if arr.ndim == 2 else 1
            lines.append(
                _da(name, arr.ravel().astype(np.float64), "Float64", binary, n_comp, 10)
            )
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </ImageData>")
    lines.append("</VTKFile>")

    path.write_text("\n".join(lines), encoding="utf-8")


def _da(
    name: str,
    arr: np.ndarray,
    vtk_type: str,
    binary: bool,
    n_comp: int,
    indent: int,
) -> str:
    pad = " " * indent
    name_attr = f' Name="{name}"' if name else ""
    comp_attr = f' NumberOfComponents="{n_comp}"' if n_comp > 1 else ""

    if binary:
        raw = arr.tobytes()
        header = np.array([len(raw)], dtype="<u4").tobytes()
        encoded = base64.b64encode(header + raw).decode()
        return (
            f'{pad}<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
            f'format="binary">{encoded}</DataArray>'
        )
    vals = " ".join(f"{v:.10g}" for v in arr.ravel())
    return (
        f'{pad}<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
        f'format="ascii">{vals}</DataArray>'
    )
