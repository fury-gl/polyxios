"""Medit .meshb binary codec (GmFlib format) — read + write."""

import mmap
from pathlib import Path
import struct
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".meshb"

# GmFlib keyword codes — decoded sections
_KW_VERSION = 1
_KW_DIMENSION = 3
_KW_VERTICES = 4
_KW_TRIANGLES = 6
_KW_QUADRILATERALS = 7
_KW_TETRAHEDRA = 8
_KW_HEXAHEDRA = 10

# GmFlib keyword codes — scanned (record size known) but not decoded
_KW_EDGES = 5  # 2 node indices + ref
_KW_PRISMS = 9  # 6 node indices + ref
_KW_NORMALS = 60  # dim floats, no ref (size is dim-dependent)

_KW_END = 54

# Keyword → (polyxios element name, nodes per element).
# Insertion order defines the write and read section order: tri < quad < tetra < hex.
_KW_TO_ELEM: dict[int, tuple[str, int]] = {
    _KW_TRIANGLES: ("triangle", 3),
    _KW_QUADRILATERALS: ("quad", 4),
    _KW_TETRAHEDRA: ("tetra", 4),
    _KW_HEXAHEDRA: ("hexahedron", 8),
}
_ELEM_TO_KW: dict[str, int] = {
    "triangle": _KW_TRIANGLES,
    "quad": _KW_QUADRILATERALS,
    "tetra": _KW_TETRAHEDRA,
    "hexahedron": _KW_HEXAHEDRA,
}

# Fixed record sizes (bytes) for keywords that are scanned but not decoded.
# Lets the scanner skip over INRIA extras (corners, ridges, normals-at-vertices,
# required-* lists) without failing. Anything absent here → UserWarning + stop.
_SKIP_REC: dict[int, int] = {
    _KW_EDGES: 2 * 4 + 4,
    _KW_PRISMS: 6 * 4 + 4,
    13: 1 * 4,  # GmfCorners: 1 vertex index, no ref
    14: 1 * 4,  # GmfRidges: 1 edge index, no ref
    15: 1 * 4,  # GmfRequiredVertices
    16: 1 * 4,  # GmfRequiredEdges
    17: 1 * 4,  # GmfRequiredTriangles
    18: 1 * 4,  # GmfRequiredQuadrilaterals
    19: 1 * 4,  # GmfRequiredTetrahedra
    61: 2 * 4,  # GmfNormalAtVertices: vertex_idx + normal_idx
}

# O(1) lookup array: element_type_code → meshb keyword (-1 = unsupported).
# Index with np.minimum(codes, len-1) so out-of-range codes hit the -1 sentinel.
_max_type_code = max(ELEMENT_TYPES.values(), default=0)
_TYPE_KW_LUT: np.ndarray = np.full(_max_type_code + 2, np.intp(-1), dtype=np.intp)
for _name, _kw in _ELEM_TO_KW.items():
    _code = ELEMENT_TYPES.get(_name)
    if _code is not None:
        _TYPE_KW_LUT[_code] = _kw


def read(path: Path | str) -> PolyData:
    """Parse a Medit binary mesh file (.meshb) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .meshb file.

    Returns
    -------
    PolyData
        Parsed mesh. Elements are returned grouped by type in the fixed order
        triangles → quads → tetrahedra → hexahedra, regardless of the order
        they appear in the file. vertex_attrs["ref"] and element_attrs["ref"]
        are populated only when at least one tag is non-zero.

    Raises
    ------
    CodecError
        On unrecognised magic number, unsupported version, or truncated data.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return _decode(mm)
        finally:
            mm.close()


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a Medit binary mesh file (.meshb).

    Writes version 2 (float64, 32-bit counts). Elements are written grouped
    by type in the fixed order triangles → quads → tetrahedra → hexahedra;
    the original element ordering in ``poly`` is not preserved. If
    element_attrs["ref"] is present those integers are written as section
    reference tags; otherwise reference tags default to 0.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .meshb file path.
    """
    path = Path(path)
    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)
    dim = poly.vertices.shape[1]

    # Map element type codes → meshb keywords via LUT (pure numpy, O(n))
    if n_elems > 0:
        type_codes = poly.element_types.astype(np.intp)
        kw_per_elem = _TYPE_KW_LUT[np.minimum(type_codes, len(_TYPE_KW_LUT) - 1)]
    else:
        kw_per_elem = np.empty(0, dtype=np.intp)
    groups: dict[int, np.ndarray] = {
        kw: np.where(kw_per_elem == kw)[0]
        for kw in _KW_TO_ELEM
        if np.any(kw_per_elem == kw)
    }

    with open(path, "wb") as fh:

        def _wi32(v: int) -> None:
            fh.write(struct.pack("<i", v))

        # File header
        _wi32(_KW_VERSION)
        _wi32(2)  # version 2 = float64
        _wi32(_KW_DIMENSION)
        _wi32(dim)

        # Vertices section
        _wi32(_KW_VERTICES)
        _wi32(n_verts)
        vert_dt = np.dtype([("xyz", "<f8", (dim,)), ("ref", "<i4")])
        buf = np.zeros(n_verts, dtype=vert_dt)
        buf["xyz"] = poly.vertices[:, :dim]
        vref = poly.vertex_attrs.get("ref")
        if vref is not None:
            buf["ref"] = vref
        fh.write(buf.tobytes())

        # Element sections — vectorised gather per type
        refs_attr = poly.element_attrs.get("ref")
        for kw, idx in groups.items():
            _, n_nodes = _KW_TO_ELEM[kw]
            _wi32(kw)
            _wi32(len(idx))
            starts = np.asarray(poly.offsets[idx], dtype=np.intp)
            flat_idx = (starts[:, None] + np.arange(n_nodes, dtype=np.intp)).ravel()
            # Medit uses 1-based indices
            nodes = (poly.connectivity[flat_idx].reshape(len(idx), n_nodes) + 1).astype(
                np.int32
            )
            if refs_attr is not None:
                refs_col = refs_attr[idx].reshape(-1, 1).astype(np.int32)
            else:
                refs_col = np.zeros((len(idx), 1), dtype=np.int32)
            fh.write(np.concatenate([nodes, refs_col], axis=1).tobytes())

        _wi32(_KW_END)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_header(mm: mmap.mmap) -> tuple[int, int, str]:
    """Read magic, version, endian from mmap. Returns (version, dim, endian_char)."""
    if len(mm) < 8:
        raise CodecError("File too short for .meshb header.")

    kw = struct.unpack_from("<i", mm, 0)[0]
    if kw == _KW_VERSION:
        endian = "<"
    else:
        kw_be = struct.unpack_from(">i", mm, 0)[0]
        if kw_be == _KW_VERSION:
            endian = ">"
        else:
            raise CodecError(
                f".meshb: first word is {kw} (LE) / {kw_be} (BE); "
                "expected 1 (GmfMeshVersionFormatted)."
            )

    fmt_i = endian + "i"
    version = struct.unpack_from(fmt_i, mm, 4)[0]
    if version not in (1, 2):
        raise CodecError(f".meshb version {version} not supported (expected 1 or 2).")

    kw2 = struct.unpack_from(fmt_i, mm, 8)[0]
    if kw2 != _KW_DIMENSION:
        raise CodecError(
            f".meshb: expected Dimension keyword (3) at offset 8, got {kw2}."
        )
    dim = struct.unpack_from(fmt_i, mm, 12)[0]

    return version, dim, endian


def _scan_sections(
    mm: mmap.mmap, version: int, dim: int, endian: str
) -> dict[int, tuple[int, int]]:
    """Return {keyword: (count, data_byte_offset)} for decodable sections.

    Sections in _SKIP_REC (corners, ridges, normals, required-* lists, etc.)
    are advanced past silently. Truly unknown keywords emit a UserWarning and
    stop scanning early rather than raising, so partial meshes are returned
    instead of a hard error.
    """
    fmt_i = endian + "i"
    pos = 16  # after header (4+4+4+4 bytes)
    sections: dict[int, tuple[int, int]] = {}
    total = len(mm)
    float_size = 4 if version == 1 else 8
    vert_rec = dim * float_size + 4

    # Derived from _KW_TO_ELEM so any new element type automatically gets the right size
    decode_rec: dict[int, int] = {kw: n * 4 + 4 for kw, (_, n) in _KW_TO_ELEM.items()}

    while pos + 8 <= total:
        kw = struct.unpack_from(fmt_i, mm, pos)[0]
        if kw == _KW_END:
            break
        count = struct.unpack_from(fmt_i, mm, pos + 4)[0]
        data_start = pos + 8

        if kw == _KW_VERTICES:
            rec = vert_rec
            sections[kw] = (count, data_start)
        elif kw == _KW_NORMALS:
            rec = dim * float_size  # dim-dependent, no ref
        elif kw in decode_rec:
            rec = decode_rec[kw]
            sections[kw] = (count, data_start)
        elif kw in _SKIP_REC:
            rec = _SKIP_REC[kw]
        else:
            warnings.warn(
                f".meshb: unknown keyword {kw} at file offset {pos}; "
                "stopping section scan early. Some mesh data may be missing.",
                UserWarning,
                stacklevel=3,
            )
            break

        pos = data_start + count * rec

    return sections


def _decode(mm: mmap.mmap) -> PolyData:
    version, dim, endian = _parse_header(mm)
    sections = _scan_sections(mm, version, dim, endian)

    float_dt = endian + ("f4" if version == 1 else "f8")
    float_size = 4 if version == 1 else 8

    # --- Vertices ---
    vertices = np.zeros((0, 3), dtype=np.float64)
    vertex_attrs: dict[str, np.ndarray] = {}
    if _KW_VERTICES in sections:
        n_verts, vstart = sections[_KW_VERTICES]
        nbytes = n_verts * (dim * float_size + 4)
        vert_dt = np.dtype([("xyz", float_dt, (dim,)), ("ref", endian + "i4")])
        verts_arr = np.frombuffer(bytes(mm[vstart : vstart + nbytes]), dtype=vert_dt)
        xyz = verts_arr["xyz"].astype(np.float64)
        vertices = np.zeros((n_verts, 3), dtype=np.float64)
        vertices[:, :dim] = xyz
        vrefs = verts_arr["ref"].astype(np.int32)
        if vrefs.any():
            vertex_attrs["ref"] = vrefs

    # --- Elements — all numpy, no Python list accumulation ---
    conn_parts: list[np.ndarray] = []
    types_parts: list[np.ndarray] = []
    refs_arr_parts: list[np.ndarray] = []
    elem_sizes: list[int] = []  # n_nodes per element, length = total n_elems

    for kw, (elem_name, n_nodes) in _KW_TO_ELEM.items():
        if kw not in sections:
            continue
        n_elems, estart = sections[kw]
        nbytes = n_elems * (n_nodes * 4 + 4)
        elem_dt = np.dtype(
            [("nodes", endian + "i4", (n_nodes,)), ("ref", endian + "i4")]
        )
        arr = np.frombuffer(bytes(mm[estart : estart + nbytes]), dtype=elem_dt)
        conn_parts.append(arr["nodes"].astype(np.int32).ravel() - 1)  # 0-based
        types_parts.append(
            np.full(n_elems, ELEMENT_TYPES.get(elem_name, 0), dtype=np.uint8)
        )
        refs_arr_parts.append(arr["ref"].astype(np.int32))
        elem_sizes.extend([n_nodes] * n_elems)

    elem_attrs: dict[str, np.ndarray] = {}
    if refs_arr_parts:
        refs_flat = np.concatenate(refs_arr_parts)
        if refs_flat.any():
            elem_attrs["ref"] = refs_flat

    sizes_arr = np.asarray(elem_sizes, dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(sizes_arr)]).astype(np.int32)

    return PolyData(
        vertices=vertices,
        connectivity=np.concatenate(conn_parts).astype(np.int32)
        if conn_parts
        else np.empty(0, np.int32),
        offsets=offsets,
        element_types=np.concatenate(types_parts)
        if types_parts
        else np.empty(0, np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=elem_attrs,
    )
