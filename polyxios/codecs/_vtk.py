import mmap
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    POLYXIOS_TO_VTK,
    VTK_TO_POLYXIOS,
)
from polyxios._types import PolyData
from polyxios.exceptions import (
    CodecError,
    IndexOverflowError,
    LazyReadError,
    UnknownElementTypeError,
)
from polyxios.validate import validate_header

try:
    from polyxios._vtk_parse import (  # type: ignore[import]
        parse_ascii_cells_v42,
        parse_ascii_coords,
    )

    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

EXTENSION: str = ".vtk"

MAX_CONNECTIVITY_INDEX_V42: int = 2**31 - 1
MAX_CONNECTIVITY_INDEX_V51: int = 2**63 - 1
MAX_CONNECTIVITY_INDEX: int = MAX_CONNECTIVITY_INDEX_V42

_VTK_DTYPE_MAP: dict[str, str] = {
    "float": "f4",
    "double": "f8",
    "int": "i4",
    "long": "i8",
    "unsigned_int": "u4",
    "unsigned_long": "u8",
    "short": "i2",
    "unsigned_short": "u2",
    "char": "i1",
    "unsigned_char": "u1",
}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a VTK legacy file (UNSTRUCTURED_GRID or POLYDATA) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtk file.
    lazy
        If True and the file is binary, return arrays backed by mmap (OS-lazy pages).
        Raises LazyReadError for ASCII files.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    LazyReadError
        If lazy=True and the file uses ASCII data sections.
    CodecError
        On unsupported dataset type or malformed data.
    UnknownElementTypeError
        If the file contains a VTK cell type not in _element_types.VTK_TO_POLYXIOS.
    """
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, "rb") as fh:
        header_line = fh.readline().decode("ascii", errors="replace").strip()
        fh.readline()  # title line (unused)
        # VTK v1.0 files can have blank lines between the title and BINARY/ASCII marker.
        data_type = ""
        for _ in range(8):
            data_type = fh.readline().decode("ascii", errors="replace").strip().upper()
            if data_type:
                break
        # Some files also have blank lines before the DATASET line.
        dataset_line = ""
        for _ in range(8):
            dataset_line = (
                fh.readline().decode("ascii", errors="replace").strip().upper()
            )
            if dataset_line:
                break

    is_binary = data_type == "BINARY"
    version = _parse_vtk_version(header_line)

    if "UNSTRUCTURED_GRID" in dataset_line:
        if is_binary:
            return _read_binary(path, file_size, version, lazy=lazy)
        else:
            if lazy:
                raise LazyReadError("VTK ASCII format does not support lazy reads.")
            return _read_ascii(path, file_size, version)
    elif "POLYDATA" in dataset_line:
        if lazy:
            raise LazyReadError("VTK POLYDATA format does not support lazy reads.")
        if is_binary:
            return _read_polydata_binary(path, file_size)
        return _read_polydata_ascii(path, file_size)
    elif "RECTILINEAR_GRID" in dataset_line:
        if lazy:
            raise LazyReadError(
                "VTK RECTILINEAR_GRID format does not support lazy reads."
            )
        return _read_rectilinear_grid(path, is_binary=is_binary)
    elif "STRUCTURED_GRID" in dataset_line:
        if lazy:
            raise LazyReadError(
                "VTK STRUCTURED_GRID format does not support lazy reads."
            )
        return _read_structured_grid(path, is_binary=is_binary)
    elif "STRUCTURED_POINTS" in dataset_line:
        if lazy:
            raise LazyReadError(
                "VTK STRUCTURED_POINTS format does not support lazy reads."
            )
        return _read_structured_points(path, is_binary=is_binary)
    elif dataset_line.startswith("FIELD"):
        return _read_field_data(path)
    else:
        raise CodecError(
            f"VTK codec supports DATASET UNSTRUCTURED_GRID or POLYDATA, got: {dataset_line!r}"
        )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a VTK legacy unstructured grid file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default: False), write binary data sections (big-endian).
    vtk_version
        '4.2' (default) or '5.1'. v4.2 uses classic CELLS layout compatible
        with all VTK readers. v5.1 uses OFFSETS+CONNECTIVITY.

    Raises
    ------
    IndexOverflowError
        If connectivity.max() > MAX_CONNECTIVITY_INDEX for the chosen version.
    """
    path = Path(path)
    binary: bool = bool(opts.get("binary", False))
    vtk_version: str = str(opts.get("vtk_version", "4.2"))

    max_allowed = (
        MAX_CONNECTIVITY_INDEX_V51
        if vtk_version == "5.1"
        else MAX_CONNECTIVITY_INDEX_V42
    )
    if poly.connectivity.size > 0 and int(poly.connectivity.max()) > max_allowed:
        raise IndexOverflowError("vtk", max_allowed, int(poly.connectivity.max()))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    with open(path, "wb") as fh:
        # ASCII header
        fh.write(f"# vtk DataFile Version {vtk_version}\n".encode())
        fh.write(b"Written by polyxios\n")
        fh.write(b"BINARY\n" if binary else b"ASCII\n")
        fh.write(b"DATASET UNSTRUCTURED_GRID\n")

        # POINTS
        fh.write(f"POINTS {n_verts} double\n".encode())
        if binary:
            _write_bin_f64(poly.vertices.ravel(), fh)
        else:
            for v in poly.vertices:
                fh.write(f"{v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n".encode())

        if vtk_version == "5.1":
            _write_cells_v51(poly, fh, binary)
        else:
            _write_cells_v42(poly, fh, binary)

        # CELL_TYPES
        fh.write(f"CELL_TYPES {n_elems}\n".encode())
        vtk_types = np.array(
            [_polyxios_to_vtk_code(poly.element_types[i]) for i in range(n_elems)],
            dtype=np.int32,
        )
        if binary:
            _write_bin_i32(vtk_types, fh)
        else:
            fh.write((" ".join(str(t) for t in vtk_types) + "\n").encode())

        # POINT_DATA
        if poly.vertex_attrs:
            fh.write(f"POINT_DATA {n_verts}\n".encode())
            for name, arr in poly.vertex_attrs.items():
                _write_vtk_array(name, arr, "POINT_DATA", fh, binary, vtk_version)

        # CELL_DATA
        if poly.element_attrs:
            fh.write(f"CELL_DATA {n_elems}\n".encode())
            for name, arr in poly.element_attrs.items():
                _write_vtk_array(name, arr, "CELL_DATA", fh, binary, vtk_version)


# --- internal helpers ---


def _parse_vtk_version(header_line: str) -> str:
    """Extract version string from VTK header line."""
    parts = header_line.split()
    # "# vtk DataFile Version 4.2"
    for i, p in enumerate(parts):
        if p.lower() == "version" and i + 1 < len(parts):
            return parts[i + 1]
    return "4.2"


def _polyxios_to_vtk_code(type_code: int) -> int:
    name = ELEMENT_TYPES_INV.get(int(type_code))
    if name is None or name not in POLYXIOS_TO_VTK:
        return 7  # fallback to polygon
    return POLYXIOS_TO_VTK[name]


def _write_cells_v42(poly: PolyData, fh: object, binary: bool) -> None:
    """Write v4.2 CELLS + CELL_TYPES sections."""
    n_elems = len(poly.element_types)
    # total_size = connectivity size + n_elems (each cell prefixed by count)
    total_size = len(poly.connectivity) + n_elems
    fh.write(f"CELLS {n_elems} {total_size}\n".encode())  # type: ignore[union-attr]

    if binary:
        # Build interleaved [count, idx0, idx1, ...] int32 stream
        parts: list[np.ndarray] = []
        for i in range(n_elems):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            cnt = e - s
            parts.append(np.array([cnt], dtype=np.int32))
            parts.append(poly.connectivity[s:e].astype(np.int32))
        if parts:
            _write_bin_i32(np.concatenate(parts), fh)
    else:
        for i in range(n_elems):
            s = int(poly.offsets[i])
            e = int(poly.offsets[i + 1])
            face = poly.connectivity[s:e]
            fh.write(
                (str(e - s) + " " + " ".join(str(v) for v in face) + "\n").encode()
            )  # type: ignore[union-attr]


def _write_cells_v51(poly: PolyData, fh: object, binary: bool) -> None:
    """Write v5.1 OFFSETS + CONNECTIVITY sections."""
    n_elems = len(poly.element_types)
    conn_size = len(poly.connectivity)

    fh.write(f"CELLS {n_elems} {conn_size}\n".encode())  # type: ignore[union-attr]
    fh.write(b"OFFSETS vtktypeint64\n")
    offsets64 = poly.offsets.astype(np.int64)
    if binary:
        fh.write(offsets64.astype(np.dtype(">i8")).tobytes())  # type: ignore[union-attr]
    else:
        fh.write((" ".join(str(x) for x in offsets64) + "\n").encode())  # type: ignore[union-attr]

    fh.write(b"CONNECTIVITY vtktypeint64\n")
    conn64 = poly.connectivity.astype(np.int64)
    if binary:
        fh.write(conn64.astype(np.dtype(">i8")).tobytes())  # type: ignore[union-attr]
    else:
        fh.write((" ".join(str(x) for x in conn64) + "\n").encode())  # type: ignore[union-attr]


def _write_vtk_array(
    name: str,
    arr: np.ndarray,
    section: str,
    fh: object,
    binary: bool,
    vtk_version: str,
) -> None:
    """Write a single attribute array to a VTK file (SCALARS/VECTORS/TENSORS)."""
    if arr.ndim == 1:
        fh.write(f"SCALARS {name} double 1\n".encode())  # type: ignore[union-attr]
        fh.write(b"LOOKUP_TABLE default\n")
        flat = arr.astype(np.float64)
        if binary:
            _write_bin_f64(flat, fh)
        else:
            fh.write((" ".join(f"{v:.10g}" for v in flat) + "\n").encode())  # type: ignore[union-attr]
    elif arr.ndim == 2 and arr.shape[1] == 3:
        fh.write(f"VECTORS {name} double\n".encode())  # type: ignore[union-attr]
        flat = arr.astype(np.float64).ravel()
        if binary:
            _write_bin_f64(flat, fh)
        else:
            for row in arr:
                fh.write(f"{row[0]:.10g} {row[1]:.10g} {row[2]:.10g}\n".encode())  # type: ignore[union-attr]
    elif arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] == 3:
        fh.write(f"TENSORS {name} double\n".encode())  # type: ignore[union-attr]
        for mat in arr.astype(np.float64):
            for row in mat:
                fh.write(f"{row[0]:.10g} {row[1]:.10g} {row[2]:.10g}\n".encode())  # type: ignore[union-attr]
    elif arr.ndim == 2 and arr.shape[1] == 6:
        # Voigt 6-component - expand to 3×3 and emit TENSORS
        fh.write(f"TENSORS {name} double\n".encode())  # type: ignore[union-attr]
        for row in arr.astype(np.float64):
            mat = np.array(
                [
                    [row[0], row[3], row[4]],
                    [row[3], row[1], row[5]],
                    [row[4], row[5], row[2]],
                ]
            )
            for r in mat:
                fh.write(f"{r[0]:.10g} {r[1]:.10g} {r[2]:.10g}\n".encode())  # type: ignore[union-attr]
    else:
        # Generic multi-component: emit SCALARS with numComp
        n_comp = arr.shape[1] if arr.ndim == 2 else 1
        fh.write(f"SCALARS {name} double {n_comp}\n".encode())  # type: ignore[union-attr]
        fh.write(b"LOOKUP_TABLE default\n")
        flat = arr.astype(np.float64).ravel()
        if binary:
            _write_bin_f64(flat, fh)
        else:
            fh.write((" ".join(f"{v:.10g}" for v in flat) + "\n").encode())  # type: ignore[union-attr]


def _write_bin_f64(arr: np.ndarray, fh: object) -> None:
    fh.write(arr.astype(np.dtype(">f8")).tobytes())  # type: ignore[union-attr]


def _write_bin_i32(arr: np.ndarray, fh: object) -> None:
    fh.write(arr.astype(np.dtype(">i4")).tobytes())  # type: ignore[union-attr]


def _read_polydata_ascii(path: Path, file_size: int) -> PolyData:
    """Read a VTK legacy ASCII POLYDATA file and convert to PolyData.

    POLYDATA uses named topology sections (POLYGONS, LINES, VERTICES,
    TRIANGLE_STRIPS) instead of CELLS + CELL_TYPES.  Each section maps
    to a polyxios element type determined by the vertex count per cell:

    POLYGONS:
        3 vertices  -> triangle (code 5)
        4 vertices  -> quad     (code 9)
        N vertices  -> polygon  (code 7)
    LINES:
        2 vertices  -> line      (code 3)
        N vertices  -> poly_line (code 4)
    VERTICES:
        1 vertex    -> vertex      (code 1)
        N vertices  -> poly_vertex (code 2)
    TRIANGLE_STRIPS:
        always      -> triangle_strip (code 6)
    """
    with open(path, "rb") as fh:
        content = fh.read().decode("ascii", errors="replace")

    lines = content.splitlines()
    # Skip lines until we find POINTS (header may have blank lines / extra lines)
    i = 0
    n_lines = len(lines)

    vertices = np.zeros((0, 3), dtype=np.float64)
    conn_list: list[int] = []
    off_list: list[int] = [0]
    type_list: list[int] = []
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        upper = line.upper()

        if upper.startswith("POINTS"):
            parts = line.split()
            n_verts = int(parts[1])
            i += 1
            validate_header(n_verts, 0, 0, file_size)
            if _HAS_CYTHON:
                vertices = parse_ascii_coords(lines, i, n_verts)
                i += n_verts
            else:
                verts_raw: list[float] = []
                while len(verts_raw) < n_verts * 3:
                    verts_raw.extend(float(x) for x in lines[i].split())
                    i += 1
                vertices = np.array(verts_raw, dtype=np.float64).reshape(n_verts, 3)

        elif (
            upper.startswith("POLYGONS")
            or upper.startswith("LINES")
            or upper.startswith("VERTICES")
            or upper.startswith("TRIANGLE_STRIPS")
        ):
            parts = line.split()
            n_cells = int(parts[1])
            total_vals = int(parts[2])

            tokens = parts[3:]
            i += 1
            while len(tokens) < total_vals and i < n_lines:
                tokens.extend(lines[i].split())
                if len(tokens) >= total_vals:
                    break
                i += 1

            idx = 0
            for _ in range(n_cells):
                cnt = int(tokens[idx])
                idx += 1

                conn_list.extend(int(t) for t in tokens[idx : idx + cnt])

                idx += cnt
                off_list.append(off_list[-1] + cnt)

                if upper.startswith("POLYGONS"):
                    if cnt == 3:
                        type_list.append(ELEMENT_TYPES["triangle"])
                    elif cnt == 4:
                        type_list.append(ELEMENT_TYPES["quad"])
                    else:
                        type_list.append(ELEMENT_TYPES["polygon"])
                elif upper.startswith("LINES"):
                    if cnt == 2:
                        type_list.append(ELEMENT_TYPES["line"])
                    else:
                        type_list.append(ELEMENT_TYPES["poly_line"])
                elif upper.startswith("VERTICES"):
                    if cnt == 1:
                        type_list.append(ELEMENT_TYPES["vertex"])
                    else:
                        type_list.append(ELEMENT_TYPES["poly_vertex"])
                elif upper.startswith("TRIANGLE_STRIPS"):
                    type_list.append(ELEMENT_TYPES["triangle_strip"])
            n_elems += n_cells
            if len(tokens) >= total_vals:
                i += 1

        elif upper.startswith("POINT_DATA"):
            n_pd = int(line.split()[1])
            i += 1
            i, vertex_attrs = _parse_vtk_data_attrs(lines, i, n_pd, n_verts)

        elif upper.startswith("CELL_DATA"):
            n_cd = int(line.split()[1])
            i += 1
            i, element_attrs = _parse_vtk_data_attrs(lines, i, n_cd, n_elems)

        else:
            i += 1

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(off_list, dtype=np.int32),
        element_types=np.array(type_list, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def _read_polydata_binary(path: Path, file_size: int) -> PolyData:
    """Read a VTK legacy binary POLYDATA file."""
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        pos = 0
        for _ in range(4):
            pos = mm.find(b"\n", pos) + 1
        poly = _parse_binary_polydata_body(mm, mv, pos, file_size)
        del mv
        mm.close()
    return poly


def _parse_binary_polydata_body(
    mm: mmap.mmap,
    mv: memoryview,
    start_pos: int,
    file_size: int,
) -> PolyData:
    """Parse binary data sections of a VTK POLYDATA file."""
    pos = start_pos
    vertices = np.zeros((0, 3), dtype=np.float64)
    all_conn: list[np.ndarray] = []
    all_offs: list[int] = [0]
    all_types: list[int] = []
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while pos < file_size:
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            break
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1

        if not line:
            continue

        upper = line.upper()
        parts = line.split()

        if upper.startswith("POINTS"):
            n_verts = int(parts[1])
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            np_dt = ">f8" if vtk_dt == "double" else ">f4"
            n_bytes = n_verts * 3 * np.dtype(np_dt).itemsize
            validate_header(n_verts, 0, 0, file_size)
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt)
            vertices = raw.astype(np.float64).reshape(n_verts, 3)
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)

        elif (
            upper.startswith("POLYGONS")
            or upper.startswith("LINES")
            or upper.startswith("VERTICES")
            or upper.startswith("TRIANGLE_STRIPS")
        ):
            n_cells = int(parts[1])
            total_vals = int(parts[2])
            n_bytes_cells = total_vals * 4
            raw_cells = np.frombuffer(
                bytes(mv[pos : pos + n_bytes_cells]), dtype=">i4"
            ).astype(np.int32)
            pos += n_bytes_cells
            pos = _skip_newline(mv, pos, file_size)

            idx = 0
            for _ in range(n_cells):
                cnt = int(raw_cells[idx])
                idx += 1
                cell = raw_cells[idx : idx + cnt]
                idx += cnt
                all_conn.append(cell)
                all_offs.append(all_offs[-1] + cnt)
                if upper.startswith("POLYGONS"):
                    if cnt == 3:
                        all_types.append(ELEMENT_TYPES["triangle"])
                    elif cnt == 4:
                        all_types.append(ELEMENT_TYPES["quad"])
                    else:
                        all_types.append(ELEMENT_TYPES["polygon"])
                elif upper.startswith("LINES"):
                    if cnt == 2:
                        all_types.append(ELEMENT_TYPES["line"])
                    else:
                        all_types.append(ELEMENT_TYPES["poly_line"])
                elif upper.startswith("VERTICES"):
                    if cnt == 1:
                        all_types.append(ELEMENT_TYPES["vertex"])
                    else:
                        all_types.append(ELEMENT_TYPES["poly_vertex"])
                elif upper.startswith("TRIANGLE_STRIPS"):
                    all_types.append(ELEMENT_TYPES["triangle_strip"])
            n_elems += n_cells

        elif upper.startswith("POINT_DATA"):
            n_pd = int(parts[1])
            pos, vertex_attrs = _parse_binary_attrs(mm, mv, pos, n_pd, file_size)

        elif upper.startswith("CELL_DATA"):
            n_cd = int(parts[1])
            pos, element_attrs = _parse_binary_attrs(mm, mv, pos, n_cd, file_size)

    connectivity = (
        np.concatenate(all_conn).astype(np.int32)
        if all_conn
        else np.array([], dtype=np.int32)
    )
    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=np.array(all_offs, dtype=np.int32),
        element_types=np.array(all_types, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def _read_ascii(path: Path, file_size: int, version: str) -> PolyData:
    with open(path, "rb") as fh:
        content = fh.read().decode("ascii", errors="replace")

    lines = content.splitlines()
    # Skip the 4-line header
    i = 4
    n_lines = len(lines)

    vertices = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.array([], dtype=np.int32)
    offsets = np.array([0], dtype=np.int32)
    element_types_arr = np.array([], dtype=np.uint8)
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        upper = line.upper()

        if upper.startswith("POINTS"):
            parts = line.split()
            n_verts = int(parts[1])
            i += 1
            validate_header(n_verts, 0, 0, file_size)
            if _HAS_CYTHON:
                vertices = parse_ascii_coords(lines, i, n_verts)
                i += n_verts
            else:
                verts_raw: list[float] = []
                while len(verts_raw) < n_verts * 3:
                    verts_raw.extend(float(x) for x in lines[i].split())
                    i += 1
                vertices = np.array(verts_raw, dtype=np.float64).reshape(n_verts, 3)

        elif upper.startswith("CELLS") and not upper.startswith("CELL_TYPES"):
            parts = line.split()
            n_elems = int(parts[1])
            total_size = int(parts[2])
            i += 1
            validate_header(n_verts, n_elems, total_size, file_size)

            if version >= "5.1" and i < n_lines and "OFFSETS" in lines[i].upper():
                connectivity, offsets = _parse_v51_cells_ascii(lines, i, n_elems)
                # advance past OFFSETS + CONNECTIVITY sections
                i += 2 + (n_elems + 1) + 1 + len(connectivity)
                # simpler: skip until CELL_TYPES
                i2 = i
                while i2 < n_lines and "CELL_TYPES" not in lines[i2].upper():
                    i2 += 1
                i = i2
            elif _HAS_CYTHON:
                connectivity, offsets = parse_ascii_cells_v42(lines, i, n_elems)
                i += n_elems
            else:
                conn_list: list[int] = []
                off_list: list[int] = [0]
                for _ in range(n_elems):
                    parts2 = lines[i].split()
                    cnt = int(parts2[0])
                    conn_list.extend(int(x) for x in parts2[1 : cnt + 1])
                    off_list.append(off_list[-1] + cnt)
                    i += 1
                connectivity = np.array(conn_list, dtype=np.int32)
                offsets = np.array(off_list, dtype=np.int32)

        elif upper.startswith("CELL_TYPES"):
            n_ct = int(line.split()[1])
            i += 1
            ct_raw: list[int] = []
            while len(ct_raw) < n_ct:
                ct_raw.extend(int(x) for x in lines[i].split())
                i += 1
            type_codes: list[int] = []
            for vtk_code in ct_raw:
                if vtk_code not in VTK_TO_POLYXIOS:
                    raise UnknownElementTypeError("vtk", vtk_code)
                type_codes.append(ELEMENT_TYPES[VTK_TO_POLYXIOS[vtk_code]])
            element_types_arr = np.array(type_codes, dtype=np.uint8)

        elif upper.startswith("POINT_DATA"):
            n_pd = int(line.split()[1])
            i += 1
            i, vertex_attrs = _parse_vtk_data_attrs(lines, i, n_pd, n_verts)

        elif upper.startswith("CELL_DATA"):
            n_cd = int(line.split()[1])
            i += 1
            i, element_attrs = _parse_vtk_data_attrs(lines, i, n_cd, n_elems)

        else:
            i += 1

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def _parse_v51_cells_ascii(
    lines: list[str], i: int, n_elems: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parse v5.1 OFFSETS + CONNECTIVITY from ASCII lines starting at index i."""
    # line i: "OFFSETS vtktypeint64" or similar
    i += 1
    off_vals: list[int] = []
    while len(off_vals) < n_elems + 1:
        off_vals.extend(int(x) for x in lines[i].split())
        i += 1
    # Skip CONNECTIVITY keyword line
    i += 1
    conn_size = off_vals[-1]
    conn_vals: list[int] = []
    while len(conn_vals) < conn_size:
        conn_vals.extend(int(x) for x in lines[i].split())
        i += 1
    return np.array(conn_vals, dtype=np.int32), np.array(off_vals, dtype=np.int32)


def _parse_vtk_data_attrs(
    lines: list[str], i: int, n_declared: int, n_items: int
) -> tuple[int, dict[str, np.ndarray]]:
    """Parse POINT_DATA or CELL_DATA attribute sections."""
    attrs: dict[str, np.ndarray] = {}
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        upper = line.upper()

        # Stop at next top-level section
        if any(
            upper.startswith(kw)
            for kw in ("POINT_DATA", "CELL_DATA", "POINTS", "CELLS", "CELL_TYPES")
        ):
            break

        if upper.startswith("SCALARS"):
            parts = line.split()
            name = parts[1]
            n_comp = int(parts[3]) if len(parts) > 3 else 1
            i += 1
            # Skip LOOKUP_TABLE line
            if i < n_lines and "LOOKUP_TABLE" in lines[i].upper():
                i += 1
            vals: list[float] = []
            while len(vals) < n_items * n_comp:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            arr = np.array(vals, dtype=np.float64)
            attrs[name] = arr.reshape(n_items, n_comp) if n_comp > 1 else arr

        elif upper.startswith("VECTORS"):
            parts = line.split()
            name = parts[1]
            i += 1
            vals = []
            while len(vals) < n_items * 3:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            attrs[name] = np.array(vals, dtype=np.float64).reshape(n_items, 3)

        elif upper.startswith("TENSORS"):
            parts = line.split()
            name = parts[1]
            i += 1
            vals = []
            while len(vals) < n_items * 9:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            attrs[name] = np.array(vals, dtype=np.float64).reshape(n_items, 3, 3)

        elif upper.startswith("FIELD"):
            parts = line.split()
            n_arrays = int(parts[2])
            i += 1
            for _ in range(n_arrays):
                while i < n_lines and not lines[i].strip():
                    i += 1
                fparts = lines[i].strip().split()
                fname, n_comp_f, n_tuples = fparts[0], int(fparts[1]), int(fparts[2])
                i += 1
                vals = []
                while len(vals) < n_tuples * n_comp_f:
                    vals.extend(float(x) for x in lines[i].split())
                    i += 1
                arr = np.array(vals, dtype=np.float64)
                attrs[fname] = arr.reshape(n_tuples, n_comp_f) if n_comp_f > 1 else arr

        else:
            i += 1

    return i, attrs


def _read_binary(path: Path, file_size: int, version: str, *, lazy: bool) -> PolyData:
    """Read binary VTK file, using mmap (lazy) or direct reads."""
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)

        # Skip 4 ASCII header lines to reach the data sections
        pos = 0
        for _ in range(4):
            pos = mm.find(b"\n", pos) + 1

        poly = _parse_binary_body(mm, mv, pos, file_size, version)
        del mv  # release memoryview before closing mmap
        if not lazy:
            poly = _materialize(poly)
        mm.close()

    return poly


def _materialize(poly: PolyData) -> PolyData:
    """Convert mmap-backed arrays to in-memory copies."""
    import dataclasses

    return dataclasses.replace(
        poly,
        vertices=np.array(poly.vertices),
        connectivity=np.array(poly.connectivity),
        offsets=np.array(poly.offsets),
        element_types=np.array(poly.element_types),
        vertex_attrs={k: np.array(v) for k, v in poly.vertex_attrs.items()},
        element_attrs={k: np.array(v) for k, v in poly.element_attrs.items()},
    )


def _parse_binary_body(
    mm: mmap.mmap,
    mv: memoryview,
    start_pos: int,
    file_size: int,
    version: str,
) -> PolyData:
    """Parse binary data sections from an mmap object."""
    pos = start_pos
    vertices = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.array([], dtype=np.int32)
    offsets_arr = np.array([0], dtype=np.int32)
    element_types_arr = np.array([], dtype=np.uint8)
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    n_verts = 0
    n_elems = 0

    while pos < file_size:
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            break
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1

        if not line:
            continue

        upper = line.upper()

        if upper.startswith("POINTS"):
            parts = line.split()
            n_verts = int(parts[1])
            vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
            np_dt = ">f8" if vtk_dt == "double" else ">f4"
            n_bytes = n_verts * 3 * np.dtype(np_dt).itemsize
            validate_header(n_verts, 0, 0, file_size)
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt)
            vertices = raw.astype(np.float64).reshape(n_verts, 3)
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)

        elif upper.startswith("CELLS") and not upper.startswith("CELL_TYPES"):
            parts = line.split()
            n_elems = int(parts[1])
            total_size = int(parts[2])
            validate_header(n_verts, n_elems, total_size, file_size)

            line_end2 = mm.find(b"\n", pos)
            next_line = (
                bytes(mv[pos:line_end2])
                .decode("ascii", errors="replace")
                .strip()
                .upper()
            )

            if "OFFSETS" in next_line:
                # v5.1: OFFSETS keyword line, then int64 data, then CONNECTIVITY keyword, then int64 data
                pos = line_end2 + 1  # skip OFFSETS keyword line
                n_off = n_elems + 1
                n_bytes_off = n_off * 8
                off_raw = np.frombuffer(bytes(mv[pos : pos + n_bytes_off]), dtype=">i8")
                offsets_arr = off_raw.astype(np.int64)
                pos += n_bytes_off
                pos = _skip_newline(mv, pos, file_size)
                # skip CONNECTIVITY keyword line
                conn_kw_end = mm.find(b"\n", pos)
                pos = conn_kw_end + 1
                conn_size = int(offsets_arr[-1])
                n_bytes_conn = conn_size * 8
                conn_raw = np.frombuffer(
                    bytes(mv[pos : pos + n_bytes_conn]), dtype=">i8"
                )
                connectivity = conn_raw.astype(np.int64)
                pos += n_bytes_conn
                pos = _skip_newline(mv, pos, file_size)
            else:
                # v4.2: interleaved [count, idx0, ...] int32
                n_bytes_cells = total_size * 4
                raw = np.frombuffer(
                    bytes(mv[pos : pos + n_bytes_cells]), dtype=">i4"
                ).astype(np.int32)
                pos += n_bytes_cells
                pos = _skip_newline(mv, pos, file_size)
                connectivity, offsets_arr = _unpack_v42_cells(raw, n_elems)

        elif upper.startswith("CELL_TYPES"):
            n_ct = int(line.split()[1])
            n_bytes_ct = n_ct * 4
            raw_ct = np.frombuffer(
                bytes(mv[pos : pos + n_bytes_ct]), dtype=">i4"
            ).astype(np.int32)
            pos += n_bytes_ct
            pos = _skip_newline(mv, pos, file_size)
            type_codes: list[int] = []
            for vtk_code in raw_ct:
                vtk_code_int = int(vtk_code)
                if vtk_code_int not in VTK_TO_POLYXIOS:
                    raise UnknownElementTypeError("vtk", vtk_code_int)
                type_codes.append(ELEMENT_TYPES[VTK_TO_POLYXIOS[vtk_code_int]])
            element_types_arr = np.array(type_codes, dtype=np.uint8)

        elif upper.startswith("POINT_DATA"):
            n_pd = int(line.split()[1])
            pos, vertex_attrs = _parse_binary_attrs(mm, mv, pos, n_pd, file_size)

        elif upper.startswith("CELL_DATA"):
            n_cd = int(line.split()[1])
            pos, element_attrs = _parse_binary_attrs(mm, mv, pos, n_cd, file_size)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def _skip_newline(mv: memoryview, pos: int, file_size: int) -> int:
    """Skip a trailing newline byte if present."""
    if pos < file_size and bytes(mv[pos : pos + 1]) == b"\n":
        return pos + 1
    return pos


def _unpack_v42_cells(raw: np.ndarray, n_elems: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert v4.2 interleaved cell array to CSR connectivity + offsets."""
    conn_list: list[int] = []
    off_list: list[int] = [0]
    idx = 0
    for _ in range(n_elems):
        cnt = int(raw[idx])
        idx += 1
        conn_list.extend(int(raw[idx + j]) for j in range(cnt))
        idx += cnt
        off_list.append(off_list[-1] + cnt)
    return np.array(conn_list, dtype=np.int32), np.array(off_list, dtype=np.int32)


def _parse_binary_attrs(
    mm: mmap.mmap,
    mv: memoryview,
    pos: int,
    n_items: int,
    file_size: int,
) -> tuple[int, dict[str, np.ndarray]]:
    """Parse binary POINT_DATA or CELL_DATA attribute sections."""
    attrs: dict[str, np.ndarray] = {}

    while pos < file_size:
        line_start = pos
        line_end = mm.find(b"\n", pos)
        if line_end == -1:
            break
        line = bytes(mv[pos:line_end]).decode("ascii", errors="replace").strip()
        pos = line_end + 1

        if not line:
            continue

        upper = line.upper()
        if any(
            upper.startswith(kw)
            for kw in ("POINT_DATA", "CELL_DATA", "POINTS", "CELLS", "CELL_TYPES")
        ):
            pos = line_start  # back up so outer loop re-reads this line
            break

        if upper.startswith("SCALARS"):
            parts = line.split()
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
            n_comp = int(parts[3]) if len(parts) > 3 else 1
            np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
            # Skip LOOKUP_TABLE line
            lt_end = mm.find(b"\n", pos)
            pos = lt_end + 1
            n_bytes = n_items * n_comp * np.dtype(np_dt).itemsize
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, n_comp) if n_comp > 1 else raw

        elif upper.startswith("VECTORS"):
            parts = line.split()
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
            np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
            n_bytes = n_items * 3 * np.dtype(np_dt).itemsize
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, 3)

        elif upper.startswith("TENSORS"):
            parts = line.split()
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "double"
            np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f8")
            n_bytes = n_items * 9 * np.dtype(np_dt).itemsize
            raw = np.frombuffer(bytes(mv[pos : pos + n_bytes]), dtype=np_dt).astype(
                np.float64
            )
            pos += n_bytes
            pos = _skip_newline(mv, pos, file_size)
            attrs[name] = raw.reshape(n_items, 3, 3)

        elif upper.startswith("FIELD"):
            n_arrays = int(line.split()[2])
            for _ in range(n_arrays):
                hdr_end = mm.find(b"\n", pos)
                if hdr_end == -1:
                    break
                hdr = bytes(mv[pos:hdr_end]).decode("ascii", errors="replace").strip()
                pos = hdr_end + 1
                hparts = hdr.split()
                if len(hparts) < 4:
                    continue
                arr_name = hparts[0]
                n_comp_f, n_tuples_f = int(hparts[1]), int(hparts[2])
                vtk_dt_f = hparts[3].lower()
                np_dt_f = ">" + _VTK_DTYPE_MAP.get(vtk_dt_f, "f4")
                n_bytes_f = n_tuples_f * n_comp_f * np.dtype(np_dt_f).itemsize
                raw_f = np.frombuffer(
                    bytes(mv[pos : pos + n_bytes_f]), dtype=np_dt_f
                ).astype(np.float64)
                pos += n_bytes_f
                pos = _skip_newline(mv, pos, file_size)
                attrs[arr_name] = (
                    raw_f.reshape(n_tuples_f, n_comp_f) if n_comp_f > 1 else raw_f
                )

        else:
            pos = line_start  # unknown keyword - back up and let outer loop handle
            break

    return pos, attrs


def _read_rectilinear_grid(path: Path, *, is_binary: bool) -> PolyData:
    """Read a VTK legacy RECTILINEAR_GRID dataset (ASCII or binary).

    Parameters
    ----------
    path
        Path to the .vtk file.
    is_binary
        True when the file header says BINARY.

    Returns
    -------
    PolyData
        Mesh with meshgrid vertices from X/Y/Z_COORDINATES and generated
        hex/quad/line connectivity from DIMENSIONS.
    """
    raw = path.read_bytes()

    line_offsets: list[int] = []
    texts: list[str] = []
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        end = nl if nl != -1 else len(raw)
        line_offsets.append(pos)
        texts.append(raw[pos:end].decode("ascii", errors="replace").strip())
        pos = end + 1

    nx, ny, nz = 1, 1, 1
    xs: np.ndarray = np.zeros(1)
    ys: np.ndarray = np.zeros(1)
    zs: np.ndarray = np.zeros(1)
    n_points = 0
    in_point_data = False
    vertex_attrs: dict[str, np.ndarray] = {}

    i = 0
    n_lines = len(texts)

    while i < n_lines:
        line = texts[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("DIMENSIONS"):
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif upper.startswith("X_COORDINATES"):
            n_coord = int(parts[1])
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_coord * np.dtype(np_dt).itemsize
                xs = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                data_end = data_pos + n_bytes
                while i < n_lines and line_offsets[i] < data_end:
                    i += 1
                continue
            else:
                vals: list[float] = []
                while len(vals) < n_coord and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                xs = np.array(vals, dtype=np.float64)
                continue
        elif upper.startswith("Y_COORDINATES"):
            n_coord = int(parts[1])
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_coord * np.dtype(np_dt).itemsize
                ys = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                data_end = data_pos + n_bytes
                while i < n_lines and line_offsets[i] < data_end:
                    i += 1
                continue
            else:
                vals = []
                while len(vals) < n_coord and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                ys = np.array(vals, dtype=np.float64)
                continue
        elif upper.startswith("Z_COORDINATES"):
            n_coord = int(parts[1])
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_coord * np.dtype(np_dt).itemsize
                zs = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                data_end = data_pos + n_bytes
                while i < n_lines and line_offsets[i] < data_end:
                    i += 1
                continue
            else:
                vals = []
                while len(vals) < n_coord and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                zs = np.array(vals, dtype=np.float64)
                continue
        elif upper.startswith("POINT_DATA"):
            n_points = int(parts[1])
            in_point_data = True
        elif upper.startswith("CELL_DATA"):
            in_point_data = False
        elif in_point_data and upper.startswith("SCALARS"):
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            n_comp = int(parts[3]) if len(parts) > 3 else 1
            i += 1
            if i < n_lines and "LOOKUP_TABLE" in texts[i].upper():
                i += 1
            np_dt_base = _VTK_DTYPE_MAP.get(vtk_dt, "f4")
            if is_binary:
                np_dt = ">" + np_dt_base
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_points * n_comp * np.dtype(np_dt).itemsize
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                data_end = data_pos + n_bytes
                while i < n_lines and line_offsets[i] < data_end:
                    i += 1
                continue
            else:
                svals: list[float] = []
                while len(svals) < n_points * n_comp and i < n_lines:
                    svals.extend(float(x) for x in texts[i].split())
                    i += 1
                arr = np.array(svals, dtype=np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                continue
        elif in_point_data and upper.startswith("VECTORS"):
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                n_bytes = n_points * 3 * np.dtype(np_dt).itemsize
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                vertex_attrs[name] = arr.reshape(n_points, 3)
                data_end = data_pos + n_bytes
                while i < n_lines and line_offsets[i] < data_end:
                    i += 1
                continue
            else:
                vvals: list[float] = []
                while len(vvals) < n_points * 3 and i < n_lines:
                    vvals.extend(float(x) for x in texts[i].split())
                    i += 1
                vertex_attrs[name] = np.array(vvals, dtype=np.float64).reshape(
                    n_points, 3
                )
                continue
        elif in_point_data and upper.startswith("FIELD"):
            n_arrays = int(parts[2])
            i += 1
            for _ in range(n_arrays):
                while i < n_lines and not texts[i]:
                    i += 1
                if i >= n_lines:
                    break
                fparts = texts[i].split()
                arr_name = fparts[0]
                n_comp_f, n_tuples_f = int(fparts[1]), int(fparts[2])
                vtk_dt_f = fparts[3].lower() if len(fparts) > 3 else "float"
                i += 1
                np_dt_base_f = _VTK_DTYPE_MAP.get(vtk_dt_f, "f4")
                if is_binary:
                    np_dt_f = ">" + np_dt_base_f
                    data_pos = line_offsets[i] if i < len(line_offsets) else len(raw)
                    n_bytes = n_tuples_f * n_comp_f * np.dtype(np_dt_f).itemsize
                    arr = np.frombuffer(
                        raw[data_pos : data_pos + n_bytes], dtype=np_dt_f
                    ).astype(np.float64)
                    data_end = data_pos + n_bytes
                    while i < n_lines and line_offsets[i] < data_end:
                        i += 1
                else:
                    fvals: list[float] = []
                    while len(fvals) < n_tuples_f * n_comp_f and i < n_lines:
                        fvals.extend(float(x) for x in texts[i].split())
                        i += 1
                    arr = np.array(fvals, dtype=np.float64)
                vertex_attrs[arr_name] = (
                    arr.reshape(n_tuples_f, n_comp_f) if n_comp_f > 1 else arr
                )
            continue
        i += 1

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    cells, etype_name = _structured_grid_cells(nx, ny, nz)
    if len(cells) == 0:
        return PolyData(
            vertices=vertices,
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            vertex_attrs=vertex_attrs,
        )

    n_cells = len(cells)
    npc = cells.shape[1]
    connectivity = cells.ravel().astype(np.int32)
    offsets_arr = np.arange(0, (n_cells + 1) * npc, npc, dtype=np.int32)
    element_types_arr = np.full(n_cells, ELEMENT_TYPES[etype_name], dtype=np.uint8)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
    )


def _read_structured_grid(path: Path, *, is_binary: bool) -> PolyData:
    """Read a VTK legacy STRUCTURED_GRID dataset (ASCII or binary).

    Parameters
    ----------
    path
        Path to the .vtk file.
    is_binary
        True when the file header says BINARY.

    Returns
    -------
    PolyData
        Mesh with explicit point coordinates and generated hex/quad/line
        connectivity from the DIMENSIONS keyword.
    """
    raw = path.read_bytes()

    offsets: list[int] = []
    texts: list[str] = []
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        end = nl if nl != -1 else len(raw)
        offsets.append(pos)
        texts.append(raw[pos:end].decode("ascii", errors="replace").strip())
        pos = end + 1

    nx, ny, nz = 1, 1, 1
    n_points = 0
    vertices = np.zeros((0, 3), dtype=np.float64)
    in_point_data = False
    vertex_attrs: dict[str, np.ndarray] = {}

    i = 0
    n_lines = len(texts)

    while i < n_lines:
        line = texts[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("DIMENSIONS"):
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif upper.startswith("POINTS") and not upper.startswith("POINT_DATA"):
            n_points = int(parts[1])
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * 3 * np.dtype(np_dt).itemsize
                raw_pts = np.frombuffer(raw[data_pos : data_pos + n_bytes], dtype=np_dt)
                vertices = raw_pts.astype(np.float64).reshape(n_points, 3)
                data_end = data_pos + n_bytes
                while i < n_lines and offsets[i] < data_end:
                    i += 1
                pos = data_end
                if pos < len(raw) and raw[pos : pos + 1] == b"\n":
                    i += 1
                continue
            else:
                vals: list[float] = []
                while len(vals) < n_points * 3 and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                vertices = np.array(vals, dtype=np.float64).reshape(n_points, 3)
                continue
        elif upper.startswith("POINT_DATA"):
            in_point_data = True
            i += 1
            continue
        elif upper.startswith("CELL_DATA"):
            in_point_data = False
        elif in_point_data and upper.startswith("SCALARS"):
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            n_comp = int(parts[3]) if len(parts) > 3 else 1
            i += 1
            if i < n_lines and "LOOKUP_TABLE" in texts[i].upper():
                i += 1
            np_dt_base = _VTK_DTYPE_MAP.get(vtk_dt, "f4")
            if is_binary:
                np_dt = ">" + np_dt_base
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * n_comp * np.dtype(np_dt).itemsize
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                data_end = data_pos + n_bytes
                while i < n_lines and offsets[i] < data_end:
                    i += 1
                continue
            else:
                svals: list[float] = []
                while len(svals) < n_points * n_comp and i < n_lines:
                    svals.extend(float(x) for x in texts[i].split())
                    i += 1
                arr = np.array(svals, dtype=np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                continue
        elif in_point_data and upper.startswith("VECTORS"):
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * 3 * np.dtype(np_dt).itemsize
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                vertex_attrs[name] = arr.reshape(n_points, 3)
                data_end = data_pos + n_bytes
                while i < n_lines and offsets[i] < data_end:
                    i += 1
                continue
            else:
                vvals: list[float] = []
                while len(vvals) < n_points * 3 and i < n_lines:
                    vvals.extend(float(x) for x in texts[i].split())
                    i += 1
                vertex_attrs[name] = np.array(vvals, dtype=np.float64).reshape(
                    n_points, 3
                )
                continue
        elif in_point_data and upper.startswith("FIELD"):
            n_arrays = int(parts[2])
            i += 1
            for _ in range(n_arrays):
                while i < n_lines and not texts[i]:
                    i += 1
                if i >= n_lines:
                    break
                fparts = texts[i].split()
                arr_name = fparts[0]
                n_comp_f, n_tuples_f = int(fparts[1]), int(fparts[2])
                vtk_dt_f = fparts[3].lower() if len(fparts) > 3 else "float"
                i += 1
                np_dt_base_f = _VTK_DTYPE_MAP.get(vtk_dt_f, "f4")
                if is_binary:
                    np_dt_f = ">" + np_dt_base_f
                    data_pos = offsets[i] if i < len(offsets) else len(raw)
                    n_bytes = n_tuples_f * n_comp_f * np.dtype(np_dt_f).itemsize
                    arr = np.frombuffer(
                        raw[data_pos : data_pos + n_bytes], dtype=np_dt_f
                    ).astype(np.float64)
                    data_end = data_pos + n_bytes
                    while i < n_lines and offsets[i] < data_end:
                        i += 1
                else:
                    fvals: list[float] = []
                    while len(fvals) < n_tuples_f * n_comp_f and i < n_lines:
                        fvals.extend(float(x) for x in texts[i].split())
                        i += 1
                    arr = np.array(fvals, dtype=np.float64)
                vertex_attrs[arr_name] = (
                    arr.reshape(n_tuples_f, n_comp_f) if n_comp_f > 1 else arr
                )
            continue
        i += 1

    cells, etype_name = _structured_grid_cells(nx, ny, nz)
    if len(cells) == 0:
        return PolyData(
            vertices=vertices,
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            vertex_attrs=vertex_attrs,
        )

    n_cells = len(cells)
    npc = cells.shape[1]
    connectivity = cells.ravel().astype(np.int32)
    offsets_arr = np.arange(0, (n_cells + 1) * npc, npc, dtype=np.int32)
    element_types_arr = np.full(n_cells, ELEMENT_TYPES[etype_name], dtype=np.uint8)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
    )


def _read_field_data(path: Path) -> PolyData:
    """Read a VTK FIELD data file (no geometry) into an empty PolyData.

    Parameters
    ----------
    path
        Path to the .vtk file.

    Returns
    -------
    PolyData
        Empty mesh (no vertices, no elements). Field arrays stored in
        ``global_attrs`` keyed by array name.
    """
    warnings.warn(
        f"{path.name}: VTK FIELD dataset has no geometry. "
        "Returning empty PolyData with field arrays in global_attrs.",
        UserWarning,
        stacklevel=3,
    )

    raw = path.read_bytes()
    lines: list[str] = []
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        end = nl if nl != -1 else len(raw)
        lines.append(raw[pos:end].decode("ascii", errors="replace").strip())
        pos = end + 1

    global_attrs: dict[str, object] = {}
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("FIELD"):
            # FIELD name n_arrays
            i += 1
            continue

        # array header: name n_comp n_tuples dtype
        if len(parts) == 4:
            name, n_comp_s, n_tuples_s, vtk_dt = parts
            try:
                n_comp = int(n_comp_s)
                n_tuples = int(n_tuples_s)
            except ValueError:
                i += 1
                continue
            # skip string arrays — skip exactly n_tuples data lines
            if vtk_dt.lower() == "string":
                skipped = 0
                while skipped < n_tuples and i < n_lines:
                    if lines[i]:
                        skipped += 1
                    i += 1
                continue
            i += 1
            vals: list[float] = []
            while len(vals) < n_tuples * n_comp and i < n_lines:
                for token in lines[i].split():
                    try:
                        vals.append(float(token))
                    except ValueError:
                        pass
                i += 1
            arr = np.array(vals[: n_tuples * n_comp], dtype=np.float64)
            global_attrs[name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
            continue

        i += 1

    return PolyData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        global_attrs=global_attrs,
    )


def _structured_grid_cells(nx: int, ny: int, nz: int) -> tuple[np.ndarray, str]:
    """Generate hex/quad/line cell connectivity for a structured grid."""
    if nx > 1 and ny > 1 and nz > 1:
        i = np.arange(nx - 1)
        j = np.arange(ny - 1)
        k = np.arange(nz - 1)
        ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
        ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
        sj, sk = nx, nx * ny
        v0 = ii + jj * sj + kk * sk
        cells = np.column_stack(
            [
                v0,
                v0 + 1,
                v0 + 1 + sj,
                v0 + sj,
                v0 + sk,
                v0 + 1 + sk,
                v0 + 1 + sj + sk,
                v0 + sj + sk,
            ]
        )
        return cells, "hexahedron"
    elif nx > 1 and ny > 1:
        i = np.arange(nx - 1)
        j = np.arange(ny - 1)
        ii, jj = np.meshgrid(i, j, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        v0 = ii + jj * nx
        return np.column_stack([v0, v0 + 1, v0 + 1 + nx, v0 + nx]), "quad"
    elif nx > 1:
        i = np.arange(nx - 1)
        return np.column_stack([i, i + 1]), "line"
    elif ny > 1:
        i = np.arange(ny - 1)
        return np.column_stack([i, i + 1]), "line"
    elif nz > 1:
        i = np.arange(nz - 1)
        return np.column_stack([i, i + 1]), "line"
    return np.zeros((0, 1), dtype=np.int32), "vertex"


def _read_structured_points(path: Path, *, is_binary: bool) -> PolyData:
    """Read a VTK legacy STRUCTURED_POINTS dataset (ASCII or binary).

    Parameters
    ----------
    path
        Path to the .vtk file.
    is_binary
        True when the file header says BINARY.

    Returns
    -------
    PolyData
        Mesh with generated hex/quad/line connectivity and preserved
        vertex attributes.
    """
    raw = path.read_bytes()

    # Build (byte_offset, stripped_text) for every line
    offsets: list[int] = []
    texts: list[str] = []
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        end = nl if nl != -1 else len(raw)
        offsets.append(pos)
        texts.append(raw[pos:end].decode("ascii", errors="replace").strip())
        pos = end + 1

    nx, ny, nz = 1, 1, 1
    ox, oy, oz = 0.0, 0.0, 0.0
    sx, sy, sz = 1.0, 1.0, 1.0
    n_points = 0
    in_point_data = False
    vertex_attrs: dict[str, np.ndarray] = {}

    i = 0
    n_lines = len(texts)

    while i < n_lines:
        line = texts[i]
        if not line:
            i += 1
            continue
        upper = line.upper()
        parts = line.split()

        if upper.startswith("DIMENSIONS"):
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
        elif upper.startswith("ORIGIN"):
            ox, oy, oz = float(parts[1]), float(parts[2]), float(parts[3])
        elif upper.startswith(("SPACING", "ASPECT_RATIO")):
            sx, sy, sz = float(parts[1]), float(parts[2]), float(parts[3])
        elif upper.startswith("POINT_DATA"):
            n_points = int(parts[1])
            in_point_data = True
        elif upper.startswith("CELL_DATA"):
            in_point_data = False
        elif in_point_data and upper.startswith("SCALARS"):
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            n_comp = int(parts[3]) if len(parts) > 3 else 1
            i += 1
            if i < n_lines and "LOOKUP_TABLE" in texts[i].upper():
                i += 1
            np_dt_base = _VTK_DTYPE_MAP.get(vtk_dt, "f4")
            if is_binary:
                np_dt = ">" + np_dt_base
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * n_comp * np.dtype(np_dt).itemsize
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                data_end = data_pos + n_bytes
                while i < n_lines and offsets[i] < data_end:
                    i += 1
                continue
            else:
                vals: list[float] = []
                while len(vals) < n_points * n_comp and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                arr = np.array(vals, dtype=np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                continue
        elif in_point_data and upper.startswith("COLOR_SCALARS"):
            name = parts[1]
            n_comp = int(parts[2])
            i += 1
            if is_binary:
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * n_comp  # unsigned_char = 1 byte each
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np.uint8
                ).astype(np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                data_end = data_pos + n_bytes
                while i < n_lines and offsets[i] < data_end:
                    i += 1
                continue
            else:
                vals = []
                while len(vals) < n_points * n_comp and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                arr = np.array(vals, dtype=np.float64)
                vertex_attrs[name] = (
                    arr.reshape(n_points, n_comp) if n_comp > 1 else arr
                )
                continue
        elif in_point_data and upper.startswith("VECTORS"):
            name = parts[1]
            vtk_dt = parts[2].lower() if len(parts) > 2 else "float"
            i += 1
            if is_binary:
                np_dt = ">" + _VTK_DTYPE_MAP.get(vtk_dt, "f4")
                data_pos = offsets[i] if i < len(offsets) else len(raw)
                n_bytes = n_points * 3 * np.dtype(np_dt).itemsize
                arr = np.frombuffer(
                    raw[data_pos : data_pos + n_bytes], dtype=np_dt
                ).astype(np.float64)
                vertex_attrs[name] = arr.reshape(n_points, 3)
                data_end = data_pos + n_bytes
                while i < n_lines and offsets[i] < data_end:
                    i += 1
                continue
            else:
                vals = []
                while len(vals) < n_points * 3 and i < n_lines:
                    vals.extend(float(x) for x in texts[i].split())
                    i += 1
                vertex_attrs[name] = np.array(vals, dtype=np.float64).reshape(
                    n_points, 3
                )
                continue
        elif in_point_data and upper.startswith("FIELD"):
            n_arrays = int(parts[2])
            i += 1
            for _ in range(n_arrays):
                while i < n_lines and not texts[i]:
                    i += 1
                if i >= n_lines:
                    break
                fparts = texts[i].split()
                arr_name = fparts[0]
                n_comp_f, n_tuples_f = int(fparts[1]), int(fparts[2])
                vtk_dt_f = fparts[3].lower() if len(fparts) > 3 else "float"
                i += 1
                np_dt_base_f = _VTK_DTYPE_MAP.get(vtk_dt_f, "f4")
                if is_binary:
                    np_dt_f = ">" + np_dt_base_f
                    data_pos = offsets[i] if i < len(offsets) else len(raw)
                    n_bytes = n_tuples_f * n_comp_f * np.dtype(np_dt_f).itemsize
                    arr = np.frombuffer(
                        raw[data_pos : data_pos + n_bytes], dtype=np_dt_f
                    ).astype(np.float64)
                    data_end = data_pos + n_bytes
                    while i < n_lines and offsets[i] < data_end:
                        i += 1
                else:
                    fvals: list[float] = []
                    while len(fvals) < n_tuples_f * n_comp_f and i < n_lines:
                        fvals.extend(float(x) for x in texts[i].split())
                        i += 1
                    arr = np.array(fvals, dtype=np.float64)
                vertex_attrs[arr_name] = (
                    arr.reshape(n_tuples_f, n_comp_f) if n_comp_f > 1 else arr
                )
            continue
        i += 1

    xs = ox + np.arange(nx, dtype=np.float64) * sx
    ys = oy + np.arange(ny, dtype=np.float64) * sy
    zs = oz + np.arange(nz, dtype=np.float64) * sz
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    cells, etype_name = _structured_grid_cells(nx, ny, nz)
    if len(cells) == 0:
        return PolyData(
            vertices=vertices,
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            vertex_attrs=vertex_attrs,
        )

    n_cells = len(cells)
    npc = cells.shape[1]
    connectivity = cells.ravel().astype(np.int32)
    offsets_arr = np.arange(0, (n_cells + 1) * npc, npc, dtype=np.int32)
    element_types_arr = np.full(n_cells, ELEMENT_TYPES[etype_name], dtype=np.uint8)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets_arr,
        element_types=element_types_arr,
        vertex_attrs=vertex_attrs,
    )
