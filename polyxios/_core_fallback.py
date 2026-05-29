"""
_core_fallback.py — pure Python fallbacks for _core.pyx
=========================================================

Identical interface to _core.pyx. Used automatically when the compiled
extension is not available (no C compiler, CI environment, editable install
without a build step). Slower but produces bit-identical results.

Imported by _backend.py — do not import this module directly.
"""

import numpy as np


def build_csr(
    element_groups: list[tuple[int, np.ndarray]],
    out_connectivity: np.ndarray,
    out_offsets: np.ndarray,
    out_types: np.ndarray,
) -> None:
    """Fill pre-allocated CSR arrays from (type_code, 2d_conn_array) pairs.

    Parameters
    ----------
    element_groups
        List of (type_code_uint8, connectivity_2d) tuples.
    out_connectivity
        Pre-allocated flat connectivity array.
    out_offsets
        Pre-allocated offsets array (n_total_elements + 1).
    out_types
        Pre-allocated type codes array (n_total_elements).
    """
    conn_pos = 0
    elem_pos = 0
    out_offsets[0] = 0

    for type_code, conn_2d in element_groups:
        arr = np.asarray(conn_2d)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_elems, n_nodes = arr.shape
        flat = arr.ravel()
        out_connectivity[conn_pos : conn_pos + len(flat)] = flat
        for i in range(n_elems):
            elem_pos += 1
            out_offsets[elem_pos] = conn_pos + (i + 1) * n_nodes
            out_types[elem_pos - 1] = type_code
        conn_pos += len(flat)


def has_orphan_vertices(n_verts: int, connectivity: np.ndarray) -> bool:
    """Return True if any vertex index in [0, n_verts) is absent from connectivity.

    Parameters
    ----------
    n_verts
        Total number of vertices.
    connectivity
        Flat connectivity array.
    """
    if connectivity.size == 0:
        return n_verts > 0
    used = np.zeros(n_verts, dtype=bool)
    valid = connectivity[(connectivity >= 0) & (connectivity < n_verts)]
    used[valid] = True
    return bool(~used.all())


def compact_vertex_indices(connectivity: np.ndarray, n_verts: int) -> np.ndarray:
    """Return remap array old_idx -> new_idx (-1 if orphan).

    Parameters
    ----------
    connectivity
        Flat connectivity array.
    n_verts
        Total number of vertices.
    """
    used = np.zeros(n_verts, dtype=bool)
    valid = connectivity[(connectivity >= 0) & (connectivity < n_verts)]
    used[valid] = True
    remap = np.full(n_verts, -1, dtype=np.int32)
    new_idx = 0
    for old in range(n_verts):
        if used[old]:
            remap[old] = new_idx
            new_idx += 1
    return remap
