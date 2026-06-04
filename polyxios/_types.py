from dataclasses import dataclass, field
from typing import Any

import numpy as np


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

    # First pass: validate types, compute sizes, convert str → (code, int32_arr)
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
