from pathlib import Path
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError
from polyxios.validate import validate_header

EXTENSION: str = ".mesh"

# MFEM geometry type code → (polyxios element name, number of vertices)
_MFEM_GEOM: dict[int, tuple[str, int]] = {
    0: ("vertex", 1),
    1: ("line", 2),
    2: ("triangle", 3),
    3: ("quad", 4),
    4: ("tetra", 4),
    5: ("hexahedron", 8),
    6: ("wedge", 6),
    7: ("pyramid", 5),
}

# Reverse: polyxios name → MFEM geometry type code
_POLY_TO_MFEM: dict[str, int] = {name: code for code, (name, _) in _MFEM_GEOM.items()}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse an MFEM mesh file (.mesh) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .mesh file.
    lazy
        Ignored; .mesh files are always read eagerly.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    UnsupportedFormatError
        For MFEM INLINE and NURBS mesh variants.
    CodecError
        On malformed or unrecognised mesh data.
    """
    path = Path(path)
    file_size = path.stat().st_size

    header, all_tokens = _read_header_and_tokens(path)

    if not header:
        raise CodecError(f"'{path.name}' is empty.")

    if header.startswith("MFEM INLINE"):
        return _read_inline(path, header, all_tokens)
    if header.startswith("MFEM NURBS") or header.startswith("MFEM NC-Mesh"):
        return _read_nurbs(path, header, all_tokens, file_size)
    if header.startswith("MFEM NC mesh"):
        return _read_nc(path, header, all_tokens, file_size)
    if not header.startswith("MFEM mesh"):
        raise CodecError(
            f"'{path.name}' does not start with 'MFEM mesh'. Got: '{header[:40]}'"
        )

    _read_section_int(all_tokens, "dimension")  # validated but not used directly
    n_elems, elem_data = _read_section_table(all_tokens, "elements")
    # boundary section is optional; we skip it
    n_verts, vert_data = _read_section_table(all_tokens, "vertices")

    validate_header(n_verts, n_elems, n_elems * 4, file_size)

    # Parse vertices.
    # Standard meshes: vertices section = count, coord_dim, then n_verts coordinate rows.
    # High-order meshes: vertices section = count only (next token is 'nodes' keyword);
    # actual coordinates live in the 'nodes' FiniteElementSpace field.
    vert_iter = iter(vert_data)
    coord_dim_token = next(vert_iter, "3")
    vertices = np.zeros((n_verts, 3), dtype=np.float64)

    if coord_dim_token in ("nodes", "elements", "boundary", ""):
        # High-order mesh: read coordinates from the 'nodes' FEM field instead
        vertices = _parse_nodes_field(all_tokens, n_verts)
    else:
        coord_dim = int(coord_dim_token)
        for vi in range(n_verts):
            coords = [float(next(vert_iter)) for _ in range(coord_dim)]
            vertices[vi, : len(coords)] = coords

    # Parse elements: each row = attr type v0 v1 ...
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []

    elem_iter = iter(elem_data)
    for _ in range(n_elems):
        _attr = int(next(elem_iter))
        geom = int(next(elem_iter))
        poly_name, n_nodes = _MFEM_GEOM.get(geom, ("polygon", -1))
        if n_nodes < 0:
            raise CodecError(f"Unknown MFEM geometry type {geom} in '{path.name}'.")
        indices = [int(next(elem_iter)) for _ in range(n_nodes)]
        conn_list.extend(indices)
        offsets_list.append(offsets_list[-1] + n_nodes)
        types_list.append(ELEMENT_TYPES.get(poly_name, 0))

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs={},
        element_attrs={},
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to an MFEM mesh file (.mesh).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    """
    path = Path(path)
    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)
    dim = 3 if np.any(poly.vertices[:, 2] != 0) else 2

    lines: list[str] = []
    lines.append("MFEM mesh v1.0")
    lines.append("")
    lines.append("dimension")
    lines.append(str(dim))
    lines.append("")
    lines.append("elements")
    lines.append(str(n_elems))
    for i in range(n_elems):
        s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
        poly_name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "triangle")
        geom = _POLY_TO_MFEM.get(poly_name, 2)
        indices = " ".join(str(int(v)) for v in poly.connectivity[s:e])
        lines.append(f"1 {geom} {indices}")
    lines.append("")
    lines.append("vertices")
    lines.append(str(n_verts))
    lines.append(str(dim))
    for v in poly.vertices:
        coord = " ".join(f"{c:.10g}" for c in v[:dim])
        lines.append(coord)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _read_inline(path: Path, header: str, tokens: list[str]) -> PolyData:
    """Generate an MFEM INLINE parametric mesh from its recipe.

    INLINE meshes store only a parametric recipe (element type, grid counts,
    domain extents).  This reader materialises the uniform structured grid
    directly, matching MFEM's ``MakeCartesian`` output for all supported types:
    ``segment``, ``quad``, ``tri``, ``hex``, ``tet``, ``wedge``, ``pyramid``.
    """
    # Parse key = value tokens
    params: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i + 1] == "=":
            key = tokens[i]
            val_str = tokens[i + 2] if i + 2 < len(tokens) else ""
            try:
                params[key] = int(val_str)
            except ValueError:
                try:
                    params[key] = float(val_str)
                except ValueError:
                    params[key] = val_str
            i += 3
        elif "=" in tokens[i]:
            k, _, v = tokens[i].partition("=")
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
            i += 1
        else:
            i += 1

    mtype = str(params.get("type", "hex")).strip()
    nx = int(params.get("nx", 1))
    ny = int(params.get("ny", 1))
    nz = int(params.get("nz", 1))
    sx = float(params.get("sx", 1.0))
    sy = float(params.get("sy", 1.0))
    sz = float(params.get("sz", 1.0))

    return _inline_build(mtype, nx, ny, nz, sx, sy, sz, params)


def _inline_build(
    mtype: str,
    nx: int,
    ny: int,
    nz: int,
    sx: float,
    sy: float,
    sz: float,
    params: dict,
) -> PolyData:
    """Materialise an INLINE mesh for any supported MFEM element type."""
    x = np.linspace(0.0, sx, nx + 1)
    y = np.linspace(0.0, sy, ny + 1)
    z = np.linspace(0.0, sz, nz + 1)

    if mtype == "segment":
        vertices = np.column_stack([x, np.zeros(nx + 1), np.zeros(nx + 1)])
        conn = np.array([[i, i + 1] for i in range(nx)], dtype=np.int32).ravel()
        offsets = np.arange(0, (nx + 1) * 2, 2, dtype=np.int32)
        etypes = np.full(nx, ELEMENT_TYPES["line"], dtype=np.uint8)
        return _make_poly(vertices, conn, offsets, etypes, params)

    if mtype in ("quad", "tri"):
        zz, yy, xx = np.meshgrid(np.zeros(1), y, x, indexing="ij")
        verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
        nxp1 = nx + 1

        if mtype == "quad":
            conn, offs, ets = [], [0], []
            for iy in range(ny):
                for ix in range(nx):
                    v0 = ix + iy * nxp1
                    quad = [v0, v0 + 1, v0 + 1 + nxp1, v0 + nxp1]
                    conn.extend(quad)
                    offs.append(offs[-1] + 4)
                    ets.append(ELEMENT_TYPES["quad"])
        else:  # tri: each quad → 2 triangles
            conn, offs, ets = [], [0], []
            for iy in range(ny):
                for ix in range(nx):
                    v0 = ix + iy * nxp1
                    a, b, c, d = v0, v0 + 1, v0 + 1 + nxp1, v0 + nxp1
                    conn.extend([a, b, d])
                    offs.append(offs[-1] + 3)
                    ets.append(ELEMENT_TYPES["triangle"])
                    conn.extend([b, c, d])
                    offs.append(offs[-1] + 3)
                    ets.append(ELEMENT_TYPES["triangle"])

        return _make_poly(
            verts,
            np.array(conn, np.int32),
            np.array(offs, np.int32),
            np.array(ets, np.uint8),
            params,
        )

    # 3-D types: build vertex grid
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    nxp1, nyp1 = nx + 1, ny + 1

    conn, offs, ets = [], [0], []

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                v0 = ix + iy * nxp1 + iz * nxp1 * nyp1
                v1, v2, v3 = v0 + 1, v0 + 1 + nxp1, v0 + nxp1
                v4 = v0 + nxp1 * nyp1
                v5, v6, v7 = v4 + 1, v4 + 1 + nxp1, v4 + nxp1

                if mtype == "hex":
                    conn.extend([v0, v1, v2, v3, v4, v5, v6, v7])
                    offs.append(offs[-1] + 8)
                    ets.append(ELEMENT_TYPES["hexahedron"])

                elif mtype == "tet":
                    # 6 tets per hex — MFEM ordering (alternating by ix+iy+iz parity)
                    if (ix + iy + iz) % 2 == 0:
                        tets = [
                            [v0, v1, v3, v4],
                            [v1, v2, v3, v6],
                            [v4, v5, v6, v1],
                            [v3, v4, v6, v7],
                            [v1, v3, v4, v6],
                        ]
                    else:
                        tets = [
                            [v0, v1, v2, v5],
                            [v0, v2, v3, v7],
                            [v0, v5, v7, v4],
                            [v2, v5, v6, v7],
                            [v0, v2, v5, v7],
                        ]
                    for tet in tets:
                        conn.extend(tet)
                        offs.append(offs[-1] + 4)
                        ets.append(ELEMENT_TYPES["tetra"])

                elif mtype == "wedge":
                    # 2 wedges per hex (split bottom face diagonally)
                    conn.extend([v0, v1, v3, v4, v5, v7])
                    offs.append(offs[-1] + 6)
                    ets.append(ELEMENT_TYPES["wedge"])
                    conn.extend([v1, v2, v3, v5, v6, v7])
                    offs.append(offs[-1] + 6)
                    ets.append(ELEMENT_TYPES["wedge"])

                elif mtype == "pyramid":
                    # 6 pyramids per hex sharing the centre point — MFEM style
                    cx, cy, cz = (
                        verts[v0, 0] + 0.5 * (sx / nx),
                        verts[v0, 1] + 0.5 * (sy / ny),
                        verts[v0, 2] + 0.5 * (sz / nz),
                    )
                    ci = len(verts)
                    verts = np.vstack([verts, [[cx, cy, cz]]])
                    for base in (
                        [v0, v1, v2, v3],
                        [v4, v7, v6, v5],
                        [v0, v4, v5, v1],
                        [v1, v5, v6, v2],
                        [v2, v6, v7, v3],
                        [v3, v7, v4, v0],
                    ):
                        conn.extend(base + [ci])
                        offs.append(offs[-1] + 5)
                        ets.append(ELEMENT_TYPES["pyramid"])

    return _make_poly(
        verts,
        np.array(conn, np.int32),
        np.array(offs, np.int32),
        np.array(ets, np.uint8),
        params,
    )


def _make_poly(vertices, connectivity, offsets, element_types, params) -> PolyData:
    return PolyData(
        vertices=vertices.astype(np.float64),
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs={},
        element_attrs={},
        global_attrs={"mfem_inline_params": params},
    )


def _read_nurbs(path: Path, header: str, tokens: list[str], file_size: int) -> PolyData:
    """Read an MFEM NURBS mesh, returning control points and element topology.

    NURBS meshes use B-spline basis functions to map a parametric domain onto
    physical space.  The file stores **control points** (not actual mesh nodes)
    together with knot vectors and weights.  Evaluating real vertex positions
    requires computing NURBS basis functions at quadrature points — this is
    non-trivial and depends on the polynomial order and knot structure.

    What we return
    --------------
    A PolyData whose ``vertices`` are the **control points** (from the
    ``FiniteElementSpace``/``nodes`` section) and whose connectivity reflects the
    patch topology.  This is *not* a renderable conforming mesh; the control
    points are the B-spline coefficients that define the geometry, not surface
    sample points.

    Additional data is stored in ``global_attrs``:

        ``poly.global_attrs["mfem_nurbs_knotvectors"]``  →  list of knot vectors
        ``poly.global_attrs["mfem_nurbs_weights"]``      →  list of float weights

    How to get actual mesh vertices
    --------------------------------
    Use MFEM to refine/snap the NURBS mesh to an explicit mesh and export::

        import mfem  # pip install mfem (PyMFEM)
        mesh = mfem.Mesh(str(path))
        mesh.UniformRefinement()             # optional: refine
        mesh.SetCurvature(1)                 # linearise to degree-1 (flat)
        mesh.Save("standard.mesh")
        poly = polyxios.read("standard.mesh")
    """
    n_verts, vert_data = _read_section_table(tokens, "vertices")
    n_elems, elem_data = _read_section_table(tokens, "elements")

    validate_header(n_verts, n_elems, n_elems * 4, file_size)

    # Control points live in the nodes FiniteElementSpace field
    vertices = _parse_nodes_field(tokens, n_verts)

    # Parse element connectivity (same layout as standard mesh)
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    elem_iter = iter(elem_data)
    for _ in range(n_elems):
        try:
            _attr = int(next(elem_iter))
            geom = int(next(elem_iter))
        except StopIteration:
            break
        poly_name, n_nodes = _MFEM_GEOM.get(geom, ("polygon", -1))
        if n_nodes < 0:
            break
        indices = [int(next(elem_iter)) for _ in range(n_nodes)]
        conn_list.extend(indices)
        offsets_list.append(offsets_list[-1] + n_nodes)
        types_list.append(ELEMENT_TYPES.get(poly_name, 0))

    # Collect knot vectors and weights into global_attrs
    kv_list: list[list[float]] = []
    try:
        kv_idx = tokens.index("knotvectors")
        n_kv = int(tokens[kv_idx + 1])
        pos = kv_idx + 2
        for _ in range(n_kv):
            order = int(tokens[pos])
            n_knots = int(tokens[pos + 1])
            knots = [float(tokens[pos + 2 + k]) for k in range(n_knots)]
            kv_list.append([order, n_knots] + knots)
            pos += 2 + n_knots
    except (ValueError, IndexError):
        pass

    weights: list[float] = []
    try:
        w_idx = tokens.index("weights")
        pos = w_idx + 1
        while pos < len(tokens):
            try:
                weights.append(float(tokens[pos]))
                pos += 1
            except ValueError:
                break
    except ValueError:
        pass

    warnings.warn(
        f"'{path.name}' is an MFEM NURBS mesh (header: '{header}'). "
        "NURBS meshes store B-spline control points, not actual mesh vertices. "
        "The 'vertices' in the returned PolyData are CONTROL POINTS — they define "
        "the geometry mathematically but are NOT physical mesh nodes. "
        "Rendering this mesh directly will produce incorrect results. "
        f"Control points: {len(vertices)}, elements: {len(types_list)}, "
        f"knot vectors: {len(kv_list)}, weights: {len(weights)}. "
        "Knot vectors are in poly.global_attrs['mfem_nurbs_knotvectors'], "
        "weights in poly.global_attrs['mfem_nurbs_weights']. "
        "To get actual mesh vertices: install PyMFEM (`pip install mfem`), then "
        "`import mfem; mesh = mfem.Mesh(path); mesh.SetCurvature(1)` "
        "(linearises the NURBS geometry to explicit node positions), "
        "`mesh.Save('out.mesh')`, and `polyxios.read('out.mesh')`.",
        UserWarning,
        stacklevel=3,
    )
    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs={},
        element_attrs={},
        global_attrs={
            "mfem_nurbs_knotvectors": kv_list,
            "mfem_nurbs_weights": weights,
        },
    )


def _read_nc(path: Path, header: str, tokens: list[str], file_size: int) -> PolyData:
    """Read an MFEM NC (non-conforming) mesh with full vertex reconstruction.

    NC meshes store a forest-of-octrees refinement tree.  Each entry in the
    ``elements`` section has the format::

        rank  attr  geom  ref_type  node0  node1  ...

    * ``rank == -1``: non-leaf (refined) element — skipped.
    * ``rank >= 0``: leaf element — actual mesh cell with real vertex indices.

    Vertex positions are reconstructed from two sections:

    - ``coordinates``: base (coarse) vertex positions.
    - ``vertex_parents``: midpoint rules ``child p1 p2`` meaning
      ``coords[child] = 0.5 * (coords[p1] + coords[p2])``.

    Processing the parents in listed order (MFEM guarantees parents always
    precede children) gives the position of every vertex referenced by a
    leaf element.

    Note: hanging-node constraints (T-junction enforcement) are not applied.
    The geometry is correct but the mesh is not FEM-conforming at refinement
    boundaries.  For FEM use, export via PyMFEM (``pip install mfem``)::

        import mfem  # pip install mfem (PyMFEM)
        mesh = mfem.Mesh(str(path))
        mesh.Save("standard.mesh")
        poly = polyxios.read("standard.mesh")
    """
    n_elems_total, elem_data = _read_section_table(tokens, "elements")

    # Parse only LEAF elements (rank >= 0)
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    n_leaf = 0

    elem_iter = iter(elem_data)
    for _ in range(n_elems_total):
        try:
            rank = int(next(elem_iter))
            _attr = int(next(elem_iter))
            geom = int(next(elem_iter))
            _ref_type = int(next(elem_iter))
        except StopIteration:
            break
        poly_name, n_nodes = _MFEM_GEOM.get(geom, ("polygon", -1))
        if n_nodes < 0:
            break
        indices_raw = [int(next(elem_iter)) for _ in range(n_nodes)]
        if rank >= 0:
            conn_list.extend(indices_raw)
            offsets_list.append(offsets_list[-1] + n_nodes)
            types_list.append(ELEMENT_TYPES.get(poly_name, 0))
            n_leaf += 1

    # Build full vertex table from base coordinates + midpoint rules
    vertices = np.zeros((0, 3), dtype=np.float64)
    try:
        c_idx = tokens.index("coordinates")
        n_base = int(tokens[c_idx + 1])
        coord_dim = int(tokens[c_idx + 2])
        base_vals = [float(tokens[c_idx + 3 + i]) for i in range(n_base * coord_dim)]
        base_verts = np.array(base_vals, dtype=np.float64).reshape(n_base, coord_dim)

        # Determine full table size from max vertex index referenced by leaf elements
        max_idx = max(conn_list) if conn_list else n_base - 1
        full = np.zeros((max_idx + 1, 3), dtype=np.float64)
        full[:n_base, :coord_dim] = base_verts

        # Apply midpoint rules in listed order (parents always precede children)
        try:
            vp_idx = tokens.index("vertex_parents")
            n_parents = int(tokens[vp_idx + 1])
            for j in range(n_parents):
                child = int(tokens[vp_idx + 2 + j * 3])
                p1 = int(tokens[vp_idx + 3 + j * 3])
                p2 = int(tokens[vp_idx + 4 + j * 3])
                if child <= max_idx:
                    full[child] = 0.5 * (full[p1] + full[p2])
        except (ValueError, IndexError):
            pass

        vertices = full
    except (ValueError, IndexError):
        pass

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs={},
        element_attrs={},
        global_attrs={
            "mfem_nc_n_total_elements": n_elems_total,
            "mfem_nc_n_leaf_elements": n_leaf,
        },
    )


def _parse_nodes_field(tokens: list[str], n_verts: int) -> np.ndarray:
    """Extract vertex coordinates from an MFEM 'nodes' FiniteElementSpace field.

    High-order MFEM meshes omit explicit coordinates in the ``vertices`` section
    and instead provide a ``nodes`` GridFunction.  The field header declares
    ``VDim`` (spatial dimension) and ``Ordering`` (0=by-component, 1=by-node).
    We read all float values after the header and reshape accordingly, returning
    the first *n_verts* rows as the corner-vertex positions.
    """
    try:
        idx = tokens.index("nodes")
    except ValueError:
        return np.zeros((n_verts, 3), dtype=np.float64)

    # Scan forward: skip FiniteElementSpace header keywords until float values start
    vdim = 3
    ordering = 1
    pos = idx + 1
    while pos < len(tokens):
        tok = tokens[pos]
        if tok == "FiniteElementSpace":
            pos += 1
        elif tok.startswith("FiniteElementCollection:"):
            pos += 2  # skip collection name token
        elif tok == "VDim:":
            vdim = int(tokens[pos + 1])
            pos += 2
        elif tok == "Ordering:":
            ordering = int(tokens[pos + 1])
            pos += 2
        else:
            break  # reached the float data

    # Collect all float values
    float_vals: list[float] = []
    for tok in tokens[pos:]:
        try:
            float_vals.append(float(tok))
        except ValueError:
            break

    if not float_vals:
        return np.zeros((n_verts, 3), dtype=np.float64)

    n_nodes = len(float_vals) // vdim
    arr = np.array(float_vals[: n_nodes * vdim], dtype=np.float64)

    if ordering == 0:
        # by-component: [all_x, all_y, all_z]
        coords = arr.reshape(vdim, n_nodes).T
    else:
        # by-node: [x0 y0 z0, x1 y1 z1, ...]
        coords = arr.reshape(n_nodes, vdim)

    take = min(n_verts, len(coords))
    out = np.zeros((n_verts, 3), dtype=np.float64)
    out[:take, :vdim] = coords[:take]
    return out


def _read_header_and_tokens(path: Path) -> tuple[str, list[str]]:
    """Return (first_meaningful_line, list_of_remaining_tokens).

    Comments (``#``-prefixed) are stripped.  The header is the complete first
    non-comment, non-empty line; everything after is split into individual
    tokens for section-based parsing.
    """
    header = ""
    tokens: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.split("#")[0].strip()
            if not stripped:
                continue
            if not header:
                header = stripped
            else:
                tokens.extend(stripped.split())
    return header, tokens


def _read_section_int(tokens: list[str], keyword: str) -> int:
    """Find *keyword* in the flat token list and return the integer that follows."""
    try:
        idx = tokens.index(keyword)
        return int(tokens[idx + 1])
    except (ValueError, IndexError):
        return 0


def _read_section_table(tokens: list[str], keyword: str) -> tuple[int, list[str]]:
    """Find *keyword* section, read the count, return (count, remaining_tokens).

    Returns the tokens that belong to this section (count × row_width tokens)
    identified by the declared count.  Since row width varies by element type,
    we return everything after the count token and rely on the caller to consume
    exactly the right number of values.
    """
    try:
        idx = tokens.index(keyword)
    except ValueError:
        return 0, []
    count = int(tokens[idx + 1])
    # Return the slice starting after the count token
    data = tokens[idx + 2 :]
    return count, data
