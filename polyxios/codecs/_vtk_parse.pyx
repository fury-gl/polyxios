# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3

"""Cython hot-paths for VTK legacy ASCII parsing.

parse_ascii_coords       Parse n_verts lines of 'x y z' floats into a
                         contiguous float64 array. Replaces the pure-Python
                         loop that calls float() on every token.

parse_ascii_cells_v42    Parse n_cells lines of 'count i0 i1 ...' into CSR
                         connectivity + offsets arrays. Used for v4.2 CELLS
                         sections (the classic layout compatible with all
                         VTK/ParaView versions).
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


cpdef cnp.ndarray parse_ascii_coords(
    object lines,
    Py_ssize_t start,
    Py_ssize_t n_verts,
):
    """Parse n_verts lines of ASCII float coordinates.

    Parameters
    ----------
    lines
        List of ASCII text lines.
    start
        Index of first data line.
    n_verts
        Number of vertices to parse.

    Returns
    -------
    np.ndarray
        Shape (n_verts, 3) float64 array.
    """
    cdef Py_ssize_t i
    cdef double[:, ::1] out_v

    out = np.empty((n_verts, 3), dtype=np.float64)
    out_v = out

    for i in range(n_verts):
        parts = lines[start + i].split()
        out_v[i, 0] = float(parts[0])
        out_v[i, 1] = float(parts[1])
        out_v[i, 2] = float(parts[2])

    return out


cpdef tuple parse_ascii_cells_v42(
    object lines,
    Py_ssize_t start,
    Py_ssize_t n_cells,
):
    """Parse n_cells lines of v4.2 CELLS data into CSR arrays.

    Parameters
    ----------
    lines
        List of ASCII text lines.
    start
        Index of first cell line.
    n_cells
        Number of cells to parse.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (connectivity, offsets) as int32 arrays.
    """
    cdef Py_ssize_t i
    cdef int cnt, j, off_acc

    conn_list = []
    offsets_list = [0]
    off_acc = 0

    for i in range(n_cells):
        parts = lines[start + i].split()
        cnt = int(parts[0])
        for j in range(1, cnt + 1):
            conn_list.append(int(parts[j]))
        off_acc += cnt
        offsets_list.append(off_acc)

    return np.array(conn_list, dtype=np.int32), np.array(offsets_list, dtype=np.int32)
