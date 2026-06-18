from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    LINE_ELEMENT_TYPES,
    QUADRATIC_SURFACE_CORNERS,
    SURFACE_ELEMENT_TYPES,
)

_SURFACE_CODES = SURFACE_ELEMENT_TYPES
_LINE_CODES = LINE_ELEMENT_TYPES
_TRIANGLE_CODE = ELEMENT_TYPES["triangle"]
_QUAD_PIXEL_CODES = frozenset({ELEMENT_TYPES["quad"], ELEMENT_TYPES["pixel"]})


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
    def faces(self) -> np.ndarray | None:
        """Triangle face indices, shape (n_tris, 3), dtype int32.

        Surface elements are triangulated: quads and pixels split into 2
        triangles, polygons and triangle_strips fan-triangulated.
        Non-surface elements (lines, volumes) are excluded.

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
        tris = []
        for i in range(len(self.element_types)):
            etype = int(self.element_types[i])
            if etype not in _SURFACE_CODES:
                continue
            cell = self.connectivity[self.offsets[i] : self.offsets[i + 1]]
            # Quadratic elements: use corner nodes only for linearized rendering.
            n_corners = QUADRATIC_SURFACE_CORNERS.get(etype)
            if n_corners is not None:
                cell = cell[:n_corners]
                etype = _TRIANGLE_CODE if n_corners == 3 else int(ELEMENT_TYPES["quad"])
            if etype == _TRIANGLE_CODE:
                tris.append(cell)
            elif etype in _QUAD_PIXEL_CODES:
                tris.append(cell[[0, 1, 2]])
                tris.append(cell[[0, 2, 3]])
            else:
                tris.extend(cell[[0, j, j + 1]] for j in range(1, len(cell) - 1))
        return np.array(tris, dtype=np.int32) if tris else None

    @property
    def lines(self) -> list[np.ndarray] | None:
        """Line connectivity as a list of vertex-index arrays.

        Returns
        -------
        list of numpy.ndarray or None
            Each array has shape (n_pts,) int32 — vertex indices for one
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
