import base64
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.exceptions import LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vtp"

try:
    from polyxios._vtp_base64 import decode_base64_array  # type: ignore[import]

    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

    def decode_base64_array(
        b64_data: bytes, dtype_str: str, big_endian: bool
    ) -> np.ndarray:  # type: ignore[misc]
        """Pure Python fallback for VTP base64 decoding."""
        raw = base64.b64decode(b64_data)
        if len(raw) <= 4:
            return np.array([], dtype=dtype_str)
        data_bytes = raw[4:]
        endian = ">" if big_endian else "<"
        arr = np.frombuffer(data_bytes, dtype=endian + dtype_str)
        return arr.astype(dtype_str)


_SECTION_TYPES = ("Verts", "Lines", "Strips", "Polys")


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a VTK PolyData XML file (.vtp) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtp file.
    lazy
        If True, XML tree is parsed eagerly but array data decoded on access.
        NOTE: Not fully supported with frozen PolyData; currently ignored (eager).

    Returns
    -------
    PolyData
        Parsed mesh data combining Verts/Lines/Strips/Polys sections.
    """
    if lazy:
        raise LazyReadError(
            "VTP lazy reads require mutable array proxies; not supported with frozen PolyData."
        )

    path = Path(path)
    file_size = path.stat().st_size

    tree = ET.parse(str(path))
    root = tree.getroot()

    byte_order = root.get("byte_order", "LittleEndian")
    big_endian = byte_order == "BigEndian"

    pd_elem = root.find("PolyData")
    if pd_elem is None:
        raise ValueError("No <PolyData> element found in VTP file.")

    all_vertices: list[np.ndarray] = []
    all_connectivity: list[np.ndarray] = []
    all_offsets: list[int] = [0]
    all_types: list[int] = []
    all_vertex_attrs: dict[str, list[np.ndarray]] = {}
    all_element_attrs: dict[str, list[np.ndarray]] = {}

    for piece in pd_elem.findall("Piece"):
        n_points = int(piece.get("NumberOfPoints", "0"))

        points_elem = piece.find("Points")
        if points_elem is not None and n_points > 0:
            da = points_elem.find("DataArray")
            if da is not None:
                flat = _decode_da(da, big_endian)
                verts = flat.reshape(n_points, -1)[:, :3].astype(np.float64)
                all_vertices.append(verts)

        vert_offset = (
            sum(v.shape[0] for v in all_vertices[:-1]) if len(all_vertices) > 1 else 0
        )

        for section in _SECTION_TYPES:
            sect_elem = piece.find(section)
            if sect_elem is None:
                continue
            conn_da = sect_elem.find("DataArray[@Name='connectivity']")
            off_da = sect_elem.find("DataArray[@Name='offsets']")
            if conn_da is None or off_da is None:
                continue

            conn = _decode_da(conn_da, big_endian).astype(np.int32) + vert_offset
            piece_offsets = _decode_da(off_da, big_endian).astype(np.int32)

            if section == "Verts":
                code = ELEMENT_TYPES["vertex"]
            elif section == "Lines":
                code = ELEMENT_TYPES["line"]
            elif section == "Strips":
                code = ELEMENT_TYPES["triangle_strip"]
            else:  # Polys
                code = None  # determined per-element

            prev_off = all_offsets[-1]
            for i, end in enumerate(piece_offsets):
                start_local = int(piece_offsets[i - 1]) if i > 0 else 0
                end_local = int(end)
                n_nodes = end_local - start_local
                local_conn = conn[start_local:end_local]
                all_connectivity.append(local_conn)
                new_off = prev_off + n_nodes
                all_offsets.append(new_off)
                prev_off = new_off

                if code is not None:
                    all_types.append(code)
                else:
                    if n_nodes == 3:
                        all_types.append(ELEMENT_TYPES["triangle"])
                    elif n_nodes == 4:
                        all_types.append(ELEMENT_TYPES["quad"])
                    else:
                        all_types.append(ELEMENT_TYPES["polygon"])

        # PointData
        pd_data = piece.find("PointData")
        if pd_data is not None:
            for da in pd_data:
                name = da.get("Name", "unknown")
                arr = _decode_da(da, big_endian)
                n_comp = int(da.get("NumberOfComponents", "1"))
                if n_comp > 1:
                    arr = arr.reshape(-1, n_comp)
                all_vertex_attrs.setdefault(name, []).append(arr)

        # CellData
        cd_data = piece.find("CellData")
        if cd_data is not None:
            for da in cd_data:
                name = da.get("Name", "unknown")
                arr = _decode_da(da, big_endian)
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

    validate_header(vertices.shape[0], len(element_types), len(connectivity), file_size)

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
    """Serialise PolyData to a VTK PolyData XML file (.vtp).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default), encode arrays as base64 binary.
    compressed
        If True (default: False), compress binary data with zlib.
    """
    path = Path(path)
    binary: bool = bool(opts.get("binary", True))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">')
    lines.append("  <PolyData>")

    # Compute per-section element counts
    n_polys = n_elems  # write all as Polys for generality

    lines.append(
        f'    <Piece NumberOfPoints="{n_verts}" NumberOfVerts="0" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{n_polys}">'
    )

    lines.append("      <Points>")
    lines.append(
        _da("", poly.vertices.ravel().astype(np.float64), "Float64", binary, 3, 10)
    )
    lines.append("      </Points>")

    # Build Polys connectivity and offsets
    conn = poly.connectivity.astype(np.int32)
    off = poly.offsets[1:].astype(np.int32)  # VTP offsets are cumulative (no leading 0)

    lines.append("      <Polys>")
    lines.append(_da("connectivity", conn, "Int32", binary, 1, 10))
    lines.append(_da("offsets", off, "Int32", binary, 1, 10))
    lines.append("      </Polys>")

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
    lines.append("  </PolyData>")
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
    else:
        vals = " ".join(f"{v:.10g}" for v in arr.ravel())
        return (
            f'{pad}<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
            f'format="ascii">{vals}</DataArray>'
        )


def _decode_da(elem: ET.Element, big_endian: bool) -> np.ndarray:
    """Decode a VTK <DataArray> element."""
    fmt = elem.get("format", "ascii")
    dtype_str = _vtk_to_np(elem.get("type", "Float64"))
    text = (elem.text or "").strip()

    if fmt == "ascii":
        vals = [float(x) for x in text.split() if x]
        return np.array(vals, dtype=dtype_str)
    elif fmt in ("binary", "base64"):
        return decode_base64_array(text.encode(), dtype_str, big_endian)
    return np.array([], dtype=dtype_str)


def _vtk_to_np(vtk_type: str) -> str:
    mapping = {
        "Float32": "f4",
        "Float64": "f8",
        "Int8": "i1",
        "Int16": "i2",
        "Int32": "i4",
        "Int64": "i8",
        "UInt8": "u1",
        "UInt16": "u2",
        "UInt32": "u4",
        "UInt64": "u8",
    }
    return mapping.get(vtk_type, "f8")
