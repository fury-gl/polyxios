import mmap
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
        data_type = fh.readline().decode("ascii", errors="replace").strip().upper()
        # Some VTK v1.0 files have a blank line before the DATASET line; skip them.
        dataset_line = ""
        for _ in range(8):  # guard against infinite loop on malformed files
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
            raise LazyReadError(
                "VTK ASCII POLYDATA format does not support lazy reads."
            )
        return _read_polydata_ascii(path, file_size)
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

        else:
            pos = line_start  # unknown keyword - back up and let outer loop handle
            break

    return pos, attrs
