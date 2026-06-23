import base64
from pathlib import Path
from typing import Any

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    POLYXIOS_TO_VTK,
    VTK_TO_POLYXIOS,
)
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import decode_da, parse_xml
from polyxios.exceptions import LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vtu"


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a VTK UnstructuredGrid XML file (.vtu) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtu file.
    lazy
        Deferred decoding is not supported; raises LazyReadError when True.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    LazyReadError
        If lazy=True.
    """
    if lazy:
        raise LazyReadError("VTU lazy reads are not supported with frozen PolyData.")

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

    ug = root.find("UnstructuredGrid")
    if ug is None:
        raise ValueError("No <UnstructuredGrid> element found in VTU file.")

    all_vertices: list[np.ndarray] = []
    all_connectivity: list[np.ndarray] = []
    all_offsets: list[int] = [0]
    all_types: list[int] = []
    all_vertex_attrs: dict[str, list[np.ndarray]] = {}
    all_element_attrs: dict[str, list[np.ndarray]] = {}

    for piece in ug.findall("Piece"):
        n_points = int(piece.get("NumberOfPoints", "0"))

        points_elem = piece.find("Points")
        if points_elem is not None and n_points > 0:
            da = points_elem.find("DataArray")
            if da is not None:
                flat = _decode(da)
                if flat.size >= n_points * 3:
                    verts = flat.reshape(n_points, -1)[:, :3].astype(np.float64)
                    all_vertices.append(verts)

        vert_offset = (
            sum(v.shape[0] for v in all_vertices[:-1]) if len(all_vertices) > 1 else 0
        )

        cells_elem = piece.find("Cells")
        if cells_elem is not None:
            conn_da = cells_elem.find("DataArray[@Name='connectivity']")
            off_da = cells_elem.find("DataArray[@Name='offsets']")
            types_da = cells_elem.find("DataArray[@Name='types']")

            if conn_da is not None and off_da is not None and types_da is not None:
                conn = _decode(conn_da).astype(np.int32) + vert_offset
                vtk_offsets = _decode(off_da).astype(np.int32)
                vtk_codes = _decode(types_da).astype(np.uint8)

                prev = all_offsets[-1]
                for i, end in enumerate(vtk_offsets):
                    start_local = int(vtk_offsets[i - 1]) if i > 0 else 0
                    end_local = int(end)
                    all_connectivity.append(conn[start_local:end_local])
                    prev = prev + (end_local - start_local)
                    all_offsets.append(prev)

                for code in vtk_codes:
                    name = VTK_TO_POLYXIOS.get(int(code), "empty_cell")
                    all_types.append(ELEMENT_TYPES.get(name, 0))

        pd_data = piece.find("PointData")
        if pd_data is not None:
            for da in pd_data:
                name = da.get("Name", "unknown")
                arr = _decode(da)
                if arr.size == 0:
                    continue
                n_comp = int(da.get("NumberOfComponents", "1"))
                if n_comp > 1:
                    arr = arr.reshape(-1, n_comp)
                all_vertex_attrs.setdefault(name, []).append(arr)

        cd_data = piece.find("CellData")
        if cd_data is not None:
            for da in cd_data:
                name = da.get("Name", "unknown")
                arr = _decode(da)
                if arr.size == 0:
                    continue
                n_comp = int(da.get("NumberOfComponents", "1"))
                if n_comp > 1:
                    arr = arr.reshape(-1, n_comp)
                all_element_attrs.setdefault(name, []).append(arr)

    vertices = (
        np.concatenate(all_vertices)
        if all_vertices
        else np.zeros((0, 3), dtype=np.float64)
    )
    connectivity = (
        np.concatenate(all_connectivity).astype(np.int32)
        if all_connectivity
        else np.array([], dtype=np.int32)
    )
    offsets = np.array(all_offsets, dtype=np.int32)
    element_types = np.array(all_types, dtype=np.uint8)

    validate_header(
        vertices.shape[0],
        len(element_types),
        len(connectivity),
        file_size,
        compressed=compressed,
    )

    vertex_attrs = {k: np.concatenate(v) for k, v in all_vertex_attrs.items()}
    element_attrs = {k: np.concatenate(v) for k, v in all_element_attrs.items()}

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a VTK UnstructuredGrid XML file (.vtu).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default), encode arrays as base64 binary.
    """
    path = Path(path)
    binary: bool = bool(opts.get("binary", True))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    vtk_types = np.array(
        [
            POLYXIOS_TO_VTK.get(ELEMENT_TYPES_INV.get(int(t), "empty_cell"), 0)
            for t in poly.element_types
        ],
        dtype=np.uint8,
    )
    vtk_offsets = poly.offsets[1:].astype(np.int32)

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
    )
    lines.append("  <UnstructuredGrid>")
    lines.append(f'    <Piece NumberOfPoints="{n_verts}" NumberOfCells="{n_elems}">')

    lines.append("      <Points>")
    lines.append(
        _da("", poly.vertices.ravel().astype(np.float64), "Float64", binary, 3, 10)
    )
    lines.append("      </Points>")

    lines.append("      <Cells>")
    lines.append(
        _da("connectivity", poly.connectivity.astype(np.int32), "Int32", binary, 1, 10)
    )
    lines.append(_da("offsets", vtk_offsets, "Int32", binary, 1, 10))
    lines.append(_da("types", vtk_types, "UInt8", binary, 1, 10))
    lines.append("      </Cells>")

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
    lines.append("  </UnstructuredGrid>")
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
    vals = " ".join(str(v) for v in arr.ravel())
    return (
        f'{pad}<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
        f'format="ascii">{vals}</DataArray>'
    )
