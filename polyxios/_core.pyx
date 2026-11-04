# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3
"""
_core.pyx - compiled hot-paths for mesh topology operations
============================================================

Contains every performance-critical mesh operation that benefits from Cython:

  build_csr               Build flat CSR connectivity + offsets arrays from a
                          list of (type_code, 2-D connectivity) pairs. Called
                          by make_polydata() to assemble every new PolyData.

  has_orphan_vertices     Return True when at least one vertex index is never
                          referenced in the connectivity array. Used as a fast
                          early-exit check in remove_orphan_vertices().

  compact_vertex_indices  Produce a remapping array (old_index - new_index,
                          -1 for unreferenced vertices). Called after
                          has_orphan_vertices confirms orphans exist.

  merge_rows              Group equal rows of a float array in one pass over
                          a hash table, in first-occurrence order. Called by
                          the STL reader to merge the corners its triangles
                          repeat.

If Cython is not available the identical interface is provided in pure Python
by _core_fallback.py - same signatures, same results, just slower.
Binary I/O hot-paths live in the codec-specific Cython files (e.g. _vtk_parse.pyx).
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdint cimport int32_t, int64_t, uint64_t
from libc.string cimport memcpy

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


ctypedef fused real:
    float
    double


cdef inline uint64_t _mix(uint64_t h) noexcept nogil:
    """Return the splitmix64 finaliser of ``h``."""
    h ^= h >> 30
    h *= 0xBF58476D1CE4E5B9ULL
    h ^= h >> 27
    h *= 0x94D049BB133111EBULL
    h ^= h >> 31
    return h


cdef inline uint64_t _row_hash(
    real[:, ::1] rows, Py_ssize_t i, uint64_t seed
) noexcept nogil:
    """Hash row ``i`` so that rows comparing equal hash alike."""
    cdef Py_ssize_t j
    cdef uint64_t h = seed
    cdef uint64_t bits
    cdef double value
    for j in range(rows.shape[1]):
        # -0.0 and 0.0 compare equal, so they have to hash alike.
        value = rows[i, j] + 0.0
        bits = 0
        memcpy(&bits, &value, sizeof(double))
        h = _mix(h ^ bits)
    return h


cdef inline bint _same_row(real[:, ::1] rows, Py_ssize_t a, Py_ssize_t b) noexcept nogil:
    """Say whether rows ``a`` and ``b`` compare equal value by value."""
    cdef Py_ssize_t j
    for j in range(rows.shape[1]):
        if rows[a, j] != rows[b, j]:
            return False
    return True


cdef inline bint _has_nan(real[:, ::1] rows, Py_ssize_t i) noexcept nogil:
    """Say whether row ``i`` holds a NaN, which equals no row."""
    cdef Py_ssize_t j
    for j in range(rows.shape[1]):
        if rows[i, j] != rows[i, j]:
            return True
    return False


def merge_rows(real[:, ::1] rows, *, uint64_t seed):
    """Group equal rows in first-occurrence order.

    The same grouping as ``np.unique(rows, axis=0)`` - rows equal value by
    value, ``-0.0`` equal to ``0.0``, a row holding a NaN equal to no other
    - without its sort: one pass over an open-addressing table keyed by a
    hash of each row. The table grows as rows are added, so it is sized by
    the distinct rows, not by all of them.

    Parameters
    ----------
    rows
        A C-contiguous float32 or float64 array, one row per item.
    seed
        Mixed into every hash. Drawn fresh for each call, so a file cannot
        be built whose rows all land on one chain of the table.

    Returns
    -------
    first : numpy.ndarray
        The index of each distinct row's first occurrence, in that order.
    inverse : numpy.ndarray
        For each row, the position of its group in ``first``; int32 when
        every position fits one, int64 past that.
    """
    cdef Py_ssize_t n = rows.shape[0]
    cdef Py_ssize_t i, k, slot, count = 0
    cdef int64_t uid
    cdef bint wide = n > 2147483647
    cdef Py_ssize_t capacity = 16
    while capacity < n // 8:
        capacity *= 2
    cdef uint64_t mask = capacity - 1

    table_arr = np.full(capacity, -1, dtype=np.int64)
    cdef int64_t[::1] table = table_arr
    first_arr = np.empty(capacity // 2 + 1, dtype=np.int64)
    cdef int64_t[::1] first = first_arr
    inverse_arr = np.empty(n, dtype=np.int64 if wide else np.int32)
    cdef int64_t[::1] inverse64
    cdef int32_t[::1] inverse32
    if wide:
        inverse64 = inverse_arr
    else:
        inverse32 = inverse_arr

    for i in range(n):
        uid = -1
        if _has_nan(rows, i):
            uid = count
        else:
            slot = _row_hash(rows, i, seed) & mask
            while table[slot] != -1:
                if _same_row(rows, i, first[table[slot]]):
                    uid = table[slot]
                    break
                slot = (slot + 1) & mask
            if uid == -1:
                uid = count
                table[slot] = uid
        if uid == count:
            first[count] = i
            count += 1
        if wide:
            inverse64[i] = uid
        else:
            inverse32[i] = <int32_t>uid

        if 2 * count > capacity:
            # Doubled and refilled from the rows already kept: their hashes
            # are recomputed rather than stored, which would cost a word a
            # row for the whole read.
            capacity *= 2
            mask = capacity - 1
            table_arr = np.full(capacity, -1, dtype=np.int64)
            table = table_arr
            grown = np.empty(capacity // 2 + 1, dtype=np.int64)
            grown[:count] = first_arr[:count]
            first_arr = grown
            first = first_arr
            for k in range(count):
                if _has_nan(rows, first[k]):
                    continue
                slot = _row_hash(rows, first[k], seed) & mask
                while table[slot] != -1:
                    slot = (slot + 1) & mask
                table[slot] = k

    return first_arr[:count].copy(), inverse_arr
