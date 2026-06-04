# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3
"""
_core.pyx — compiled hot-paths for mesh topology operations
============================================================

Contains every performance-critical mesh operation that benefits from Cython:

  build_csr               Build flat CSR connectivity + offsets arrays from a
                          list of (type_code, 2-D connectivity) pairs. Called
                          by make_polydata() to assemble every new PolyData.

  has_orphan_vertices     Return True when at least one vertex index is never
                          referenced in the connectivity array. Used as a fast
                          early-exit check in remove_orphan_vertices().

  compact_vertex_indices  Produce a remapping array (old_index → new_index,
                          -1 for unreferenced vertices). Called after
                          has_orphan_vertices confirms orphans exist.

If Cython is not available the identical interface is provided in pure Python
by _core_fallback.py — same signatures, same results, just slower.
Binary I/O hot-paths live in the codec-specific Cython files (e.g. _vtk_parse.pyx).
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

cnp.import_array()


cpdef void build_csr(
    list element_groups,
    int[:] out_connectivity,
    int[:] out_offsets,
    unsigned char[:] out_types,
) except *:
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
    cdef Py_ssize_t conn_pos = 0
    cdef Py_ssize_t elem_pos = 0
    cdef Py_ssize_t i, j, n_elems, n_nodes
    cdef int type_code
    cdef int[:, ::1] arr

    out_offsets[0] = 0

    for item in element_groups:
        type_code = item[0]
        raw = np.ascontiguousarray(item[1], dtype=np.int32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        arr = raw
        n_elems = arr.shape[0]
        n_nodes = arr.shape[1]

        for i in range(n_elems):
            for j in range(n_nodes):
                out_connectivity[conn_pos + i * n_nodes + j] = arr[i, j]
            out_types[elem_pos] = type_code
            elem_pos += 1
            out_offsets[elem_pos] = <int>(conn_pos + (i + 1) * n_nodes)

        conn_pos += n_elems * n_nodes


cpdef bint has_orphan_vertices(int n_verts, int[:] connectivity) except -1:
    """Return True if any vertex index in [0, n_verts) is absent from connectivity.

    Parameters
    ----------
    n_verts
        Total number of vertices.
    connectivity
        Flat connectivity array.
    """
    cdef Py_ssize_t i, n = connectivity.shape[0]
    cdef int idx
    cdef unsigned char[:] used

    used_arr = np.zeros(n_verts, dtype=np.uint8)
    used = used_arr

    for i in prange(n, nogil=True):
        idx = connectivity[i]
        if 0 <= idx < n_verts:
            used[idx] = 1

    for i in range(n_verts):
        if used[i] == 0:
            return True
    return False


cpdef int[:] compact_vertex_indices(int[:] connectivity, int n_verts):
    """Return remap array old_idx -> new_idx (-1 if orphan).

    Parameters
    ----------
    connectivity
        Flat connectivity array.
    n_verts
        Total number of vertices.
    """
    cdef Py_ssize_t i, n = connectivity.shape[0]
    cdef int idx, new_idx = 0
    cdef unsigned char[:] used_v
    cdef int[:] remap_v

    used_arr = np.zeros(n_verts, dtype=np.uint8)
    used_v = used_arr

    for i in prange(n, nogil=True):
        idx = connectivity[i]
        if 0 <= idx < n_verts:
            used_v[idx] = 1

    remap_arr = np.full(n_verts, -1, dtype=np.int32)
    remap_v = remap_arr

    for i in range(n_verts):
        if used_v[i]:
            remap_v[i] = new_idx
            new_idx += 1

    return remap_v
