from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    LINE_ELEMENT_TYPES,
    NODES_PER_ELEMENT,
    TOPOLOGICAL_DIMENSION,
)
from polyxios.exceptions import UnknownElementTypeError

_LINE_CODES = LINE_ELEMENT_TYPES

# TOPOLOGICAL_DIMENSION spread over the whole uint8 code space, -1 where no
# element type claims the code. Indexing it dimensions every element in one
# pass, where a dict lookup per element costs a Python call per element.
_DIMENSION_BY_CODE: Final[np.ndarray] = np.array(
    [TOPOLOGICAL_DIMENSION.get(code, -1) for code in range(256)], dtype=np.int8
)


_TRIANGLE_SPLIT = [[0, 1, 2]]
_QUAD_SPLIT = [[0, 1, 2], [0, 2, 3]]

# The triangles each fixed-size surface element is cut into, as positions
# within the element, with the node count the type must hold.
_FIXED_SPLITS: Final[dict[int, tuple[int, np.ndarray]]] = {
    ELEMENT_TYPES[name]: (NODES_PER_ELEMENT[name], np.array(split))
    for name, split in {
        "triangle": _TRIANGLE_SPLIT,
        "quad": _QUAD_SPLIT,
        "pixel": [[0, 1, 3], [0, 3, 2]],
        "quadratic_triangle": _TRIANGLE_SPLIT,
        "biquadratic_triangle": _TRIANGLE_SPLIT,
        "quadratic_quad": _QUAD_SPLIT,
        "biquadratic_quad": _QUAD_SPLIT,
    }.items()
}
_POLYGON_CODE = ELEMENT_TYPES["polygon"]
_STRIP_CODE = ELEMENT_TYPES["triangle_strip"]


def ragged_arange(counts: np.ndarray) -> np.ndarray:
    """Count from zero up to each of ``counts``, all the runs laid end to end.

    Parameters
    ----------
    counts
        Non-negative run lengths.

    Returns
    -------
    numpy.ndarray
        Shape (counts.sum(),), int64: ``arange(n)`` for every ``n`` in
        ``counts``, concatenated, ``[2, 0, 3]`` giving ``[0, 1, 0, 1, 2]``.
    """
    counts = np.asarray(counts, dtype=np.int64)
    return np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)


def surface_triangles(
    connectivity: np.ndarray, *, offsets: np.ndarray, element_types: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split every surface element into triangles that keep its winding.

    Parameters
    ----------
    connectivity, offsets, element_types
        A mesh's CSR layout, as :class:`PolyData` holds it.

    Returns
    -------
    triangles : numpy.ndarray
        Shape (n_tris, 3), in ``connectivity``'s dtype, each triangle wound
        like the element it comes from and in element order. A quadratic
        element is cut along its corner nodes only. A quad is split along
        its 0-2 diagonal, a pixel - whose corners run row by row, not around
        the ring - along its 0-3 one. A triangle strip yields one triangle
        per vertex past the second, every other one turned back so the
        whole strip faces one way: triangle ``j`` is ``(j, j+1, j+2)`` when
        ``j`` is even and ``(j+1, j, j+2)`` when odd, the order OpenGL draws
        them in, so vertex ``j+2`` is always last. A strip triangle naming
        one vertex twice is dropped, as a GPU drops it: it covers no area,
        and strips repeat a vertex on purpose, to stitch several into one or
        to turn the parity of the triangles that follow. A polygon is fanned
        from its first vertex. Lines, volumes and other elements yield none.
    sources : numpy.ndarray
        Shape (n_tris,), int64: the element each triangle was cut from.

    Raises
    ------
    ValueError
        If a fixed-size surface element holds a node count other than its
        type's.
    """
    starts = np.asarray(offsets[:-1], dtype=np.int64)
    sizes = np.asarray(offsets[1:], dtype=np.int64) - starts
    counts = np.zeros(len(element_types), dtype=np.int64)
    fixed: list[tuple[np.ndarray, np.ndarray]] = []
    for code, (n_nodes, split) in _FIXED_SPLITS.items():
        rows = np.flatnonzero(element_types == code)
        if not rows.size:
            continue
        wrong = rows[sizes[rows] != n_nodes]
        if wrong.size:
            raise ValueError(
                f"element {int(wrong[0])} is a {ELEMENT_TYPES_INV[code]} of "
                f"{int(sizes[wrong[0]])} nodes; the type has {n_nodes}"
            )
        counts[rows] = len(split)
        fixed.append((rows, split))
    connectivity = np.asarray(connectivity)
    is_polygon = element_types == _POLYGON_CODE
    counts[is_polygon] = np.maximum(sizes[is_polygon] - 2, 0)
    strip_rows = np.flatnonzero(element_types == _STRIP_CODE)
    n = np.maximum(sizes[strip_rows] - 2, 0)
    strip_owner = np.repeat(strip_rows, n)
    strip_j = ragged_arange(n)
    if strip_owner.size:
        # Judged on the connectivity before any triangle is laid out, so the
        # output is sized once rather than built whole and then filtered.
        k = starts[strip_owner] + strip_j
        nxt = connectivity[1:] != connectivity[:-1]
        skip = connectivity[2:] != connectivity[:-2]
        keep = nxt[k] & nxt[k + 1] & skip[k]
        if not keep.all():
            strip_owner, strip_j = strip_owner[keep], strip_j[keep]
        counts[strip_rows] = np.bincount(
            np.searchsorted(strip_rows, strip_owner), minlength=len(strip_rows)
        )

    first = np.cumsum(counts) - counts
    positions = np.empty((int(counts.sum()), 3), dtype=np.int64)
    for rows, split in fixed:
        at = first[rows][:, None] + np.arange(len(split))
        positions[at] = starts[rows][:, None, None] + split
    polygon_rows = np.flatnonzero(is_polygon)
    n = counts[polygon_rows]
    if n.any():
        j = ragged_arange(n)
        at = np.repeat(first[polygon_rows], n) + j
        tri = np.column_stack([np.zeros_like(j), j + 1, j + 2])
        positions[at] = np.repeat(starts[polygon_rows], n)[:, None] + tri
    if strip_owner.size:
        j = strip_j
        at = np.repeat(first[strip_rows], counts[strip_rows]) + ragged_arange(
            counts[strip_rows]
        )
        tri = np.column_stack([j, j + 1, j + 2])
        odd = j % 2 == 1
        tri[odd, :2] = tri[odd, 1::-1]
        positions[at] = starts[strip_owner][:, None] + tri
    triangles = connectivity[positions]
    return triangles, np.repeat(np.arange(len(element_types)), counts)


@dataclass(frozen=True, slots=True)
class PolyData:
    """Immutable CSR-layout mesh container.

    Parameters
    ----------
    vertices
        Vertex coordinates, shape (n_verts, 3), dtype float64.
    connectivity
        Flat CSR connectivity array, dtype int32 or int64.
    offsets
        CSR offsets (indptr), length n_elements + 1, same dtype as connectivity.
    element_types
        Per-element polyxios type code, dtype uint8, length n_elements.
    vertex_attrs
        Named vertex attributes; each array has length n_verts.
    element_attrs
        Named element attributes; each array has length n_elements.
    vertex_tags
        Named vertex index subsets.
    element_tags
        Named element index subsets; one element may appear in multiple tags.
    global_attrs
        Free-form mesh-level metadata.
    """

    vertices: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    element_types: np.ndarray
    vertex_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    element_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    vertex_tags: dict[str, np.ndarray] = field(default_factory=dict)
    element_tags: dict[str, np.ndarray] = field(default_factory=dict)
    global_attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def topological_dimension(self) -> int:
        """Highest topological dimension present, 0 to 3.

        The dimension of the elements themselves, not of the space they sit
        in: a triangle mesh embedded in 3-D is still 2-D. Points are 0,
        lines 1, surfaces 2, volumes 3, and a mesh holding several of them
        reports the largest - a tetrahedral mesh that also carries its
        boundary triangles is 3-D.

        Returns
        -------
        int
            Maximum dimension over the element types present. 0 when the
            mesh has no elements.

        Raises
        ------
        UnknownElementTypeError
            If an element carries a type code outside ELEMENT_TYPES.
        """
        codes = self.element_types
        if codes.size == 0:
            return 0
        if codes.dtype != np.uint8:
            # A hand-built mesh may hold codes wider than uint8; those cannot
            # index the table at all, so they are refused before it is read.
            codes = np.asarray(codes, dtype=np.int64)
            outside = codes[(codes < 0) | (codes >= _DIMENSION_BY_CODE.size)]
            if outside.size:
                raise UnknownElementTypeError("in-memory PolyData", int(outside[0]))
        dims = _DIMENSION_BY_CODE[codes]
        unknown = np.flatnonzero(dims < 0)
        if unknown.size:
            raise UnknownElementTypeError("in-memory PolyData", int(codes[unknown[0]]))
        return int(dims.max())

    @property
    def faces(self) -> np.ndarray | None:
        """Triangle face indices, shape (n_tris, 3), dtype int32.

        Surface elements are triangulated with their winding kept: quads and
        pixels split into 2 triangles, triangle_strips into one triangle per
        vertex past the second less those naming a vertex twice, polygons
        fan-triangulated. Non-surface elements (lines, volumes) are excluded.

        Returns
        -------
        numpy.ndarray or None
            Array of shape (n_tris, 3) with vertex indices, or None when
            the mesh has no surface elements.

        Notes
        -----
        Recomputes on each access. Store the result if calling repeatedly.
        For a full PolyData with preserved attributes use
        ``polyxios.transforms.triangulate``.
        """
        tris, _ = surface_triangles(
            self.connectivity, offsets=self.offsets, element_types=self.element_types
        )
        return tris.astype(np.int32) if len(tris) else None

    @property
    def lines(self) -> list[np.ndarray] | None:
        """Line connectivity as a list of vertex-index arrays.

        Returns
        -------
        list of numpy.ndarray or None
            Each array has shape (n_pts,) int32 - vertex indices for one
            connected line or poly_line element. Returns None when no
            line/poly_line elements exist.

        Notes
        -----
        Use ``poly.vertices[idx]`` to get coordinates for each segment.
        Recomputes on each access; store the result if calling repeatedly.
        """
        result = []
        for i in range(len(self.element_types)):
            if int(self.element_types[i]) not in _LINE_CODES:
                continue
            result.append(self.connectivity[self.offsets[i] : self.offsets[i + 1]])
        return result if result else None


def make_polydata(
    vertices: np.ndarray,
    element_groups: list[tuple[str, np.ndarray]],
    *,
    vertex_attrs: dict[str, np.ndarray] | None = None,
    element_attrs: dict[str, np.ndarray] | None = None,
    vertex_tags: dict[str, np.ndarray] | None = None,
    element_tags: dict[str, np.ndarray] | None = None,
    global_attrs: dict[str, Any] | None = None,
) -> PolyData:
    """Build a PolyData from a list of (element_type_str, connectivity_2d) pairs.

    Parameters
    ----------
    vertices
        Vertex coordinates array, shape (n_verts, 3).
    element_groups
        List of (type_str, conn_2d) tuples. conn_2d is shape (n_elems, n_nodes).
        All elements in a group must have the same number of nodes.
    vertex_attrs
        Named vertex attribute arrays.
    element_attrs
        Named element attribute arrays.
    vertex_tags
        Named vertex index subsets.
    element_tags
        Named element index subsets.
    global_attrs
        Free-form metadata.

    Returns
    -------
    PolyData
        Validated PolyData with CSR layout. int32 connectivity unless any index
        >= 2**31, in which case int64.
    """
    from polyxios._backend import build_csr
    from polyxios._element_types import ELEMENT_TYPES

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 1:
        vertices = vertices.reshape(-1, 3)

    # First pass: validate types, compute sizes, convert str - (code, int32_arr)
    groups: list[tuple[int, np.ndarray]] = []
    total_conn = 0
    total_elems = 0

    for type_str, conn_arr in element_groups:
        if type_str not in ELEMENT_TYPES:
            raise ValueError(f"Unknown element type: '{type_str}'")
        arr = np.asarray(conn_arr, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        groups.append((int(ELEMENT_TYPES[type_str]), arr))
        total_conn += arr.shape[0] * arr.shape[1]
        total_elems += arr.shape[0]

    # Pre-allocate and fill via Cython hot-path (falls back to Python if not compiled)
    out_connectivity = np.empty(total_conn, dtype=np.int32)
    out_offsets = np.empty(total_elems + 1, dtype=np.int32)
    out_types = np.empty(total_elems, dtype=np.uint8)

    build_csr(groups, out_connectivity, out_offsets, out_types)

    max_idx = int(out_connectivity.max()) if out_connectivity.size > 0 else 0
    idx_dtype = np.int64 if max_idx >= 2**31 else np.int32

    return PolyData(
        vertices=vertices,
        connectivity=out_connectivity.astype(idx_dtype),
        offsets=out_offsets.astype(idx_dtype),
        element_types=out_types,
        vertex_attrs=dict(vertex_attrs or {}),
        element_attrs=dict(element_attrs or {}),
        vertex_tags=dict(vertex_tags or {}),
        element_tags=dict(element_tags or {}),
        global_attrs=dict(global_attrs or {}),
    )
