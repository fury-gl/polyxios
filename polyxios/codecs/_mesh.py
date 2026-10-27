from collections.abc import Iterator
from typing import Any
import warnings

import numpy as np

from polyxios._dimension import mark_2d, output_dimension
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, open_text, source_name, source_size, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError
from polyxios.validate import validate_header

EXTENSION: str = ".mesh"
# '.mesh' belongs to Medit ASCII as well, so it is shared rather than owned:
# the two are told apart by the keyword the file opens with. MFEM keeps the
# writes, since an output file has no content to sniff and '.mesh' has meant
# MFEM here since before Medit ASCII was read at all.
SNIFF_EXTENSIONS: tuple[str, ...] = (".mesh",)
SNIFF_DEFAULT_WRITER: bool = True
SNIFF_PRIORITY: int = 0

# Every MFEM flavour opens with one of these; the header is mandatory.
_MFEM_HEADERS: tuple[str, ...] = (
    "MFEM MESH",
    "MFEM INLINE",
    "MFEM NURBS",
    "MFEM NC-MESH",
    "MFEM NC MESH",
)


def sniff(head: bytes) -> bool:
    """Report whether a file's opening bytes look like an MFEM mesh.

    Parameters
    ----------
    head
        The file's first bytes, as handed over by the registry.

    Returns
    -------
    bool
        True when the first meaningful line names an MFEM mesh flavour.

    Notes
    -----
    Used to resolve ``.mesh``, which Medit ASCII uses too. The test is
    deliberately narrow: the header is mandatory and opens no other format.
    """
    text = head.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.upper().startswith(_MFEM_HEADERS)
    return False


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

# The geometry codes whose vertices cannot lie in a plane. A mesh of these
# written under a two-column ``vertices`` block is one MFEM does not load: it
# reads exactly as many coordinates per vertex as the block declares, and a
# tetrahedron of two-coordinate vertices is not a cell.
_SOLID_GEOMS: frozenset[int] = frozenset({4, 5, 6, 7})


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an MFEM mesh file (.mesh) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .mesh file.
    lazy
        Ignored, with a warning: MFEM is a text format and always loads
        eagerly.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    CodecError
        On malformed or unrecognised mesh data.

    Warns
    -----
    UserWarning
        For ``lazy=True``, and for an ``MFEM NURBS`` file, whose vertices
        come back as B-spline control points rather than mesh nodes.
    """
    if lazy:
        warnings.warn(
            ".mesh: lazy=True ignored; MFEM is a text format and always loads eagerly.",
            UserWarning,
            stacklevel=2,
        )
    # Measured rather than taken from a later read: the tokens are walked as
    # a stream, so no step below ever holds the whole file to take a length
    # off it. Over a compressed source that is one decompression pass counted
    # and discarded, which is the cheaper half of the trade.
    file_size = source_size(path)

    header, all_tokens = _read_header_and_tokens(path)

    if not header:
        raise CodecError(f"'{source_name(path)}' is empty.")

    # Matched against the same upper-cased spellings the sniffer accepts, and
    # in the same five shapes. Case-sensitively, a file the registry had
    # already claimed as MFEM could reach here and match no branch at all -
    # ``MFEM NC MESH v1.0`` sniffed as MFEM and then failed for not starting
    # with ``MFEM mesh``. The NC variants both belong to the NC reader:
    # ``MFEM NC-Mesh`` was falling through to the NURBS one, which read a
    # refinement forest as control points and said so in a warning naming the
    # wrong variant.
    flavour = header.upper()
    if flavour.startswith("MFEM INLINE"):
        return _read_inline(path, header, all_tokens)
    if flavour.startswith("MFEM NURBS"):
        return _read_nurbs(path, header, all_tokens, file_size)
    if flavour.startswith(("MFEM NC-MESH", "MFEM NC MESH")):
        return _read_nc(path, header, all_tokens, file_size)
    if not flavour.startswith("MFEM MESH"):
        raise CodecError(
            f"'{source_name(path)}' does not start with 'MFEM mesh'. "
            f"Got: '{header[:40]}'"
        )

    _read_section_int(all_tokens, "dimension")  # validated but not used directly
    n_elems, elem_data = _read_section_table(all_tokens, "elements", path=path)
    # boundary section is optional; we skip it
    n_verts, vert_data = _read_section_table(all_tokens, "vertices", path=path)

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
        vertices, coord_dim = _parse_nodes_field(all_tokens, n_verts, path=path)
    else:
        coord_dim = _as_dim(token=coord_dim_token, what="vertices dimension", path=path)
        vertices[:, :coord_dim] = _coordinates(
            tokens=vert_data[1 : 1 + n_verts * coord_dim],
            n_verts=n_verts,
            dim=coord_dim,
            path=path,
        )

    connectivity, offsets_list, types_list = _elements(
        data=elem_data, n_elems=n_elems, n_verts=n_verts, path=path
    )

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs={},
        element_attrs={},
        global_attrs=mark_2d(coord_dim),
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to an MFEM mesh file (.mesh).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.

    Raises
    ------
    CodecError
        When an element does not carry the node count its MFEM geometry
        holds.

    Notes
    -----
    The ``dimension`` and the width of the ``vertices`` block are 2 for a
    mesh whose z column is entirely zero - a mesh carrying
    ``global_attrs["was_2d"]`` and one built flat by hand alike - and 3
    otherwise. A mesh of solid cells keeps three however flat it lies: MFEM
    reads exactly as many coordinates per vertex as the block declares, and a
    tetrahedron of two-coordinate vertices is not a cell.

    MFEM names eight geometries and no higher-order one, so an element it has
    no geometry for - a ``quadratic_tetra``, a ``polygon`` - is skipped with a
    warning naming its type. Writing one under another geometry's code would
    leave its extra nodes to be read as the record that follows.
    """
    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)
    dim = output_dimension(poly, fmt=".mesh", flat_default=2)

    # An MFEM record names a geometry and then exactly the nodes that geometry
    # holds, so an element with no geometry of its own cannot be written at
    # all: spelling one as a triangle and following it with ten indices leaves
    # the reader seven tokens into the next record, and every element after it
    # is read as something else. They are skipped and named instead.
    records: list[str] = []
    skipped: set[str] = set()
    solid = False
    for i in range(n_elems):
        code = int(poly.element_types[i])
        poly_name = ELEMENT_TYPES_INV.get(code, "")
        geom = _POLY_TO_MFEM.get(poly_name)
        if geom is None:
            skipped.add(poly_name or f"type id {code}")
            continue
        solid = solid or geom in _SOLID_GEOMS
        s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
        expected = _MFEM_GEOM[geom][1]
        if e - s != expected:
            raise CodecError(
                f".mesh: element {i} of type {poly_name!r} has {e - s}"
                f" node(s), expected {expected}; an MFEM record holds a fixed"
                " number of nodes per geometry."
            )
        indices = " ".join(str(int(v)) for v in poly.connectivity[s:e])
        records.append(f"1 {geom} {indices}")

    if skipped:
        warnings.warn(
            f".mesh: no MFEM geometry holds {sorted(skipped)}; those elements"
            " were skipped.",
            stacklevel=2,
        )

    if dim == 2 and solid:
        # A flat mesh of solid cells: the coordinates keep their third column,
        # since a tetrahedron or a hexahedron of two-coordinate vertices is a
        # mesh MFEM refuses. Quietly, since nothing is lost - a flat mesh
        # writes a zero z either way.
        dim = 3

    lines: list[str] = []
    lines.append("MFEM mesh v1.0")
    lines.append("")
    lines.append("dimension")
    lines.append(str(dim))
    lines.append("")
    lines.append("elements")
    lines.append(str(len(records)))
    lines.extend(records)
    lines.append("")
    lines.append("vertices")
    lines.append(str(n_verts))
    lines.append(str(dim))
    for v in poly.vertices:
        coord = " ".join(f"{c:.10g}" for c in v[:dim])
        lines.append(coord)
    lines.append("")

    write_text(path, "\n".join(lines), encoding="utf-8")


def _read_inline(path: Source, header: str, tokens: list[str]) -> PolyData:
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
                    # 6 tets per hex - MFEM ordering (alternating by ix+iy+iz parity)
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
                    # 6 pyramids per hex sharing the centre point - MFEM style
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


def _read_nurbs(
    path: Source, header: str, tokens: list[str], file_size: int
) -> PolyData:
    """Read an MFEM NURBS mesh, returning control points and element topology.

    NURBS meshes use B-spline basis functions to map a parametric domain onto
    physical space.  The file stores **control points** (not actual mesh nodes)
    together with knot vectors and weights.  Evaluating real vertex positions
    requires computing NURBS basis functions at quadrature points - this is
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
    n_verts, vert_data = _read_section_table(tokens, "vertices", path=path)
    n_elems, elem_data = _read_section_table(tokens, "elements", path=path)

    validate_header(n_verts, n_elems, n_elems * 4, file_size)

    # Control points live in the nodes FiniteElementSpace field, and its vdim
    # is the geometric dimension the same way a standard mesh's coord_dim is.
    vertices, coord_dim = _parse_nodes_field(tokens, n_verts, path=path)

    # Parse element connectivity (same layout as standard mesh)
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    elem_iter = iter(elem_data)
    for e in range(n_elems):
        # The patch topology ends wherever the records stop being records.
        head = [next(elem_iter, None) for _ in range(2)]
        if None in head:
            break
        _attr = _as_int(token=head[0], what=f"element {e} attribute", path=path)
        geom = _as_int(token=head[1], what=f"element {e} geometry", path=path)
        poly_name, n_nodes = _MFEM_GEOM.get(geom, ("polygon", -1))
        if n_nodes < 0:
            break
        indices = _take_vertices(
            it=elem_iter,
            n_nodes=n_nodes,
            n_verts=n_verts,
            what=f"element {e}",
            path=path,
        )
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
        f"'{source_name(path)}' is an MFEM NURBS mesh (header: '{header}'). "
        "NURBS meshes store B-spline control points, not actual mesh vertices. "
        "The 'vertices' in the returned PolyData are CONTROL POINTS - they define "
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
            **mark_2d(coord_dim),
        },
    )


def _read_nc(path: Source, header: str, tokens: list[str], file_size: int) -> PolyData:
    """Read an MFEM NC (non-conforming) mesh with full vertex reconstruction.

    NC meshes store a forest-of-octrees refinement tree.  Each entry in the
    ``elements`` section has the format::

        rank  attr  geom  ref_type  node0  node1  ...

    * ``rank == -1``: non-leaf (refined) element - skipped.
    * ``rank >= 0``: leaf element - actual mesh cell with real vertex indices.

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
    n_elems_total, elem_data = _read_section_table(tokens, "elements", path=path)

    # Parse only LEAF elements (rank >= 0)
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    n_leaf = 0

    elem_iter = iter(elem_data)
    for e in range(n_elems_total):
        head = [next(elem_iter, None) for _ in range(4)]
        if None in head:
            break
        rank, _attr, geom, _ref_type = (
            _as_int(token=tok, what=f"element {e} header", path=path) for tok in head
        )
        poly_name, n_nodes = _MFEM_GEOM.get(geom, ("polygon", -1))
        if n_nodes < 0:
            break
        indices_raw = [
            _take_int(it=elem_iter, what=f"element {e} vertex", path=path)
            for _ in range(n_nodes)
        ]
        if rank >= 0:
            conn_list.extend(indices_raw)
            offsets_list.append(offsets_list[-1] + n_nodes)
            types_list.append(ELEMENT_TYPES.get(poly_name, 0))
            n_leaf += 1

    vertices, nc_dim = _nc_vertices(
        tokens=tokens, conn_list=conn_list, file_size=file_size, path=path
    )

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
            **mark_2d(nc_dim),
        },
    )


def _nc_vertices(
    *, tokens: list[str], conn_list: list[int], file_size: int, path: Source
) -> tuple[np.ndarray, int]:
    """Build an NC mesh's vertex table from base coordinates and midpoint rules.

    Parameters
    ----------
    tokens
        The file's tokens after the header.
    conn_list
        The vertex indices the leaf elements name.
    file_size
        Size of the source, which bounds the table the indices ask for.
    path
        The source, for messages.

    Returns
    -------
    vertices : numpy.ndarray
        ``(n, 3)`` float64, one row per vertex up to the largest index a leaf
        element names; empty when the file has no ``coordinates`` section and
        names no vertex.
    dim : int
        The dimension the ``coordinates`` section declares, 3 without one.

    Raises
    ------
    CodecError
        If a leaf element names a negative vertex, or vertices without a
        ``coordinates`` section, more vertices than the file can describe, a
        coordinate or a rule that is not a number, or a section shorter than
        its count; or a midpoint rule names a parent past the table.
    """
    if conn_list and min(conn_list) < 0:
        raise CodecError(
            f"'{source_name(path)}': a leaf element names vertex {min(conn_list)}."
        )
    if "coordinates" not in tokens:
        if conn_list:
            raise CodecError(
                f"'{source_name(path)}': leaf elements name vertices but the"
                " file has no coordinates section."
            )
        return np.zeros((0, 3), dtype=np.float64), 3
    c_idx = tokens.index("coordinates")
    n_base = _as_int(
        token=_token_at(tokens=tokens, index=c_idx + 1, path=path),
        what="coordinates count",
        path=path,
    )
    if n_base < 0:
        raise CodecError(
            f"'{source_name(path)}': coordinates count {n_base} is negative."
        )
    coord_dim = _as_dim(
        token=_token_at(tokens=tokens, index=c_idx + 2, path=path),
        what="coordinates dimension",
        path=path,
    )
    n_full = max(max(conn_list) + 1 if conn_list else 0, n_base)
    # Every vertex past the base is spelled by a midpoint rule of three tokens.
    validate_header(n_full, 0, 0, file_size)
    base_vals = [
        _as_float(
            token=_token_at(tokens=tokens, index=c_idx + 3 + i, path=path),
            what="coordinate",
            path=path,
        )
        for i in range(n_base * coord_dim)
    ]
    full = np.zeros((n_full, 3), dtype=np.float64)
    full[:n_base, :coord_dim] = np.array(base_vals, dtype=np.float64).reshape(
        n_base, coord_dim
    )

    # Apply midpoint rules in listed order (parents always precede children)
    if "vertex_parents" in tokens:
        vp_idx = tokens.index("vertex_parents")
        n_parents = _as_int(
            token=_token_at(tokens=tokens, index=vp_idx + 1, path=path),
            what="vertex_parents count",
            path=path,
        )
        for j in range(n_parents):
            child, p1, p2 = (
                _as_int(
                    token=_token_at(
                        tokens=tokens, index=vp_idx + 2 + j * 3 + k, path=path
                    ),
                    what=f"vertex_parents rule {j}",
                    path=path,
                )
                for k in range(3)
            )
            if not (0 <= p1 < n_full and 0 <= p2 < n_full):
                raise CodecError(
                    f"'{source_name(path)}': vertex_parents rule {j} names a"
                    f" parent outside 0..{n_full - 1}."
                )
            if 0 <= child < n_full:
                full[child] = 0.5 * (full[p1] + full[p2])
    return full, coord_dim


def _parse_nodes_field(
    tokens: list[str], n_verts: int, *, path: Source
) -> tuple[np.ndarray, int]:
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
        return np.zeros((n_verts, 3), dtype=np.float64), 3

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
            vdim = _as_dim(
                token=_token_at(tokens=tokens, index=pos + 1, path=path),
                what="VDim",
                path=path,
            )
            pos += 2
        elif tok == "Ordering:":
            ordering = _as_int(
                token=_token_at(tokens=tokens, index=pos + 1, path=path),
                what="Ordering",
                path=path,
            )
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
        return np.zeros((n_verts, 3), dtype=np.float64), 3

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
    return out, vdim


def _read_header_and_tokens(path: Source) -> tuple[str, list[str]]:
    """Return (first_meaningful_line, list_of_remaining_tokens).

    Comments (``#``-prefixed) are stripped.  The header is the complete first
    non-comment, non-empty line; everything after is split into individual
    tokens for section-based parsing.

    Decoded the way :func:`sniff` decodes, byte-order mark and all, so that
    every file the registry hands here is one the dispatch below can read a
    flavour off. A BOM is not whitespace and ``strip`` leaves it in place, so
    a ``MFEM mesh v1.0`` written by an editor that stamps one sniffed as MFEM
    and then failed for not starting with ``MFEM mesh``. The mark is taken off
    the header as well as by the decoder, since a caller may hand in a text
    handle whose own decoding wins over the one asked for here.
    """
    header = ""
    tokens: list[str] = []
    with open_text(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            stripped = line.split("#")[0].strip()
            if not stripped:
                continue
            if not header:
                header = stripped.lstrip("\ufeff")
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


def _read_section_table(
    tokens: list[str], keyword: str, *, path: Source
) -> tuple[int, list[str]]:
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
    count = _as_int(
        token=_token_at(tokens=tokens, index=idx + 1, path=path),
        what=f"{keyword} count",
        path=path,
    )
    if count < 0:
        raise CodecError(f"'{source_name(path)}': {keyword} count {count} is negative.")
    # Return the slice starting after the count token
    data = tokens[idx + 2 :]
    return count, data


def _as_int(*, token: str, what: str, path: Source) -> int:
    """Read ``token`` as an integer, naming ``what`` it is when it is not."""
    try:
        return int(token)
    except ValueError as exc:
        raise CodecError(
            f"'{source_name(path)}': {what} {token[:40]!r} is not an integer."
        ) from exc


def _as_float(*, token: str, what: str, path: Source) -> float:
    """Read ``token`` as a number, naming ``what`` it is when it is not."""
    try:
        return float(token)
    except ValueError as exc:
        raise CodecError(
            f"'{source_name(path)}': {what} {token[:40]!r} is not a number."
        ) from exc


def _as_dim(*, token: str, what: str, path: Source) -> int:
    """Read a coordinate count, refusing one outside 1..3."""
    dim = _as_int(token=token, what=what, path=path)
    if not 1 <= dim <= 3:
        raise CodecError(f"'{source_name(path)}': {what} {dim} is not 1, 2 or 3.")
    return dim


def _token_at(*, tokens: list[str], index: int, path: Source) -> str:
    """Return ``tokens[index]``, refusing a file that ends before it."""
    if index >= len(tokens):
        raise CodecError(f"'{source_name(path)}': the file ends inside a section.")
    return tokens[index]


def _take(*, it: Iterator[str], what: str, path: Source) -> str:
    """Return the next token, refusing a section that ends before ``what``."""
    tok = next(it, None)
    if tok is None:
        raise CodecError(f"'{source_name(path)}': the file ends before {what}.")
    return tok


def _take_int(*, it: Iterator[str], what: str, path: Source) -> int:
    """Return the next token as an integer."""
    return _as_int(token=_take(it=it, what=what, path=path), what=what, path=path)


def _take_vertices(
    *, it: Iterator[str], n_nodes: int, n_verts: int, what: str, path: Source
) -> list[int]:
    """Return an element's ``n_nodes`` vertex indices, each below ``n_verts``.

    Parameters
    ----------
    it
        The section's token stream, at the element's first vertex.
    n_nodes
        How many vertices the element's geometry holds.
    n_verts
        How many vertices the mesh has.
    what
        Names the element in messages.
    path
        The source, for messages.

    Returns
    -------
    list of int
        The indices.

    Raises
    ------
    CodecError
        If the section ends first, a token is not an integer, or an index is
        negative or past the vertices.
    """
    indices = [_take_int(it=it, what=what, path=path) for _ in range(n_nodes)]
    bad = next((i for i in indices if not 0 <= i < n_verts), None)
    if bad is not None:
        raise CodecError(
            f"'{source_name(path)}': {what} names vertex {bad}, outside"
            f" 0..{n_verts - 1}."
        )
    return indices


def _coordinates(
    *, tokens: list[str], n_verts: int, dim: int, path: Source
) -> np.ndarray:
    """Parse a standard mesh's ``vertices`` rows in one pass.

    Parameters
    ----------
    tokens
        The ``n_verts * dim`` tokens after the dimension, or fewer when the
        file ends first.
    n_verts
        How many vertices the section declares.
    dim
        Coordinates per vertex.
    path
        The source, for messages.

    Returns
    -------
    numpy.ndarray
        ``(n_verts, dim)`` float64.

    Raises
    ------
    CodecError
        If the file ends before the last coordinate, or a token is not a
        number.
    """
    if len(tokens) < n_verts * dim:
        raise CodecError(
            f"'{source_name(path)}': the file ends before vertex {len(tokens) // dim}."
        )
    try:
        return np.array(tokens, dtype=np.float64).reshape(n_verts, dim)
    except ValueError:
        k = next(k for k, tok in enumerate(tokens) if not _is_float(tok))
        raise CodecError(
            f"'{source_name(path)}': vertex {k // dim} {tokens[k][:40]!r} is not"
            " a number."
        ) from None


def _is_int(token: str) -> bool:
    """Return whether ``int`` reads ``token``."""
    try:
        int(token)
    except ValueError:
        return False
    return True


def _is_float(token: str) -> bool:
    """Return whether ``float`` reads ``token``."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _elements(
    *, data: list[str], n_elems: int, n_verts: int, path: Source
) -> tuple[np.ndarray, list[int], list[int]]:
    """Walk a standard mesh's ``elements`` records.

    Parameters
    ----------
    data
        The tokens after the section's count.
    n_elems
        How many records the section declares.
    n_verts
        How many vertices the mesh has.
    path
        The source, for messages.

    Returns
    -------
    connectivity : numpy.ndarray
        int32 vertex indices.
    offsets : list of int
        CSR offsets, starting at 0.
    types : list of int
        polyxios type codes.

    Raises
    ------
    CodecError
        If the file ends inside a record, a token is not an integer, a
        geometry is unknown, or an index is negative or past the vertices.
    """
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    pos = 0
    for e in range(n_elems):
        record = data[pos : pos + 2]
        if len(record) < 2:
            raise CodecError(
                f"'{source_name(path)}': the file ends before element {e}."
            )
        try:
            int(record[0])
            geom = int(record[1])
        except ValueError:
            raise CodecError(
                f"'{source_name(path)}': element {e} {' '.join(record)[:40]!r} is"
                " not an attribute and a geometry."
            ) from None
        poly_name, n_nodes = _MFEM_GEOM.get(geom, ("polygon", -1))
        if n_nodes < 0:
            raise CodecError(
                f"Unknown MFEM geometry type {geom} in '{source_name(path)}'."
            )
        nodes = data[pos + 2 : pos + 2 + n_nodes]
        if len(nodes) < n_nodes:
            raise CodecError(
                f"'{source_name(path)}': the file ends inside element {e}."
            )
        try:
            conn_list.extend(map(int, nodes))
        except ValueError:
            bad = next(tok for tok in nodes if not _is_int(tok))
            raise CodecError(
                f"'{source_name(path)}': element {e} {bad[:40]!r} is not an integer."
            ) from None
        offsets_list.append(offsets_list[-1] + n_nodes)
        types_list.append(ELEMENT_TYPES.get(poly_name, 0))
        pos += 2 + n_nodes
    try:
        connectivity = np.array(conn_list, dtype=np.int64)
        in_range = (connectivity >= 0) & (connectivity < n_verts)
    except OverflowError:
        in_range = np.array([0 <= i < n_verts for i in conn_list], dtype=bool)
    if not in_range.all():
        k = int(np.argmin(in_range))
        e = int(np.searchsorted(offsets_list, k, side="right")) - 1
        raise CodecError(
            f"'{source_name(path)}': element {e} names vertex {conn_list[k]}, outside"
            f" 0..{n_verts - 1}."
        )
    return connectivity.astype(np.int32), offsets_list, types_list
