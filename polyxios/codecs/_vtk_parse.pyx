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

    Raises
    ------
    IndexError
        If the file ends inside the block.
    ValueError
        If a line does not hold exactly one vertex. Either says the block is
        not one vertex to a line, which the caller answers by reading it again
        as the run of numbers the header counts.
    """
    cdef Py_ssize_t i
    cdef double[:, ::1] out_v

    # boundscheck=False takes the bounds check off the list indexing below as
    # well as off the memoryview, so a header declaring more rows than the
    # file holds would read past the end of the list rather than raise. The
    # count is a number out of the file; checked once here rather than per
    # row, which costs a well-formed block nothing.
    if start < 0 or n_verts < 0 or start + n_verts > len(lines):
        raise IndexError("POINTS block runs past the end of the file")

    out = np.empty((n_verts, 3), dtype=np.float64)
    out_v = out

    for i in range(n_verts):
        parts = lines[start + i].split()
        # A POINTS block is a run of numbers to the format, and a writer is
        # free to wrap it; refusing a line that is not one whole vertex is
        # what sends the block back to the reader that walks it as that run,
        # rather than reading three of its numbers and dropping the rest.
        if len(parts) != 3:
            raise ValueError("POINTS line does not hold exactly one vertex")
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

    Raises
    ------
    IndexError
        If the file ends inside the block.
    ValueError
        If a row does not list the vertices it declares. Either says the block
        is not one cell to a line, which the caller answers by reading it
        again in Python.
    """
    cdef Py_ssize_t i
    # Py_ssize_t rather than int: the width is a number out of the file, and
    # one past INT_MAX raised an OverflowError out of the conversion itself -
    # before the check below could name the row, and of a kind the caller's
    # fallback did not catch. Same register width on every platform this
    # builds for, so the walk costs no more.
    cdef Py_ssize_t cnt, j, off_acc

    # boundscheck=False takes the bounds check off the list indexing below as
    # well, so a header declaring more cells than the file holds rows would
    # read past the end of the list rather than raise. Checked once here, and
    # a row's own width against its tokens below.
    if start < 0 or n_cells < 0 or start + n_cells > len(lines):
        raise IndexError("CELLS block runs past the end of the file")

    conn_list = []
    offsets_list = [0]
    off_acc = 0

    for i in range(n_cells):
        parts = lines[start + i].split()
        # boundscheck=False takes the check off this indexing too, so a row of
        # no tokens - a blank line inside the block, which the format allows
        # and writers emit - read past the end of the list rather than raise.
        # One length test before the first index, where the row's own width is
        # checked against its tokens below.
        if len(parts) == 0:
            raise ValueError("CELLS row holds no vertex count")
        cnt = int(parts[0])
        # The width is a number out of the file, and a row that does not go on
        # to list it is not one cell to a line. Refusing here is what sends the
        # block back to the Python reader, which says which row it was and
        # reads a wrapped block as the run of numbers the header counts.
        if cnt < 0 or cnt + 1 > len(parts):
            raise ValueError("CELLS row does not list the vertices it declares")
        for j in range(1, cnt + 1):
            conn_list.append(int(parts[j]))
        off_acc += cnt
        offsets_list.append(off_acc)

    return np.array(conn_list, dtype=np.int32), np.array(offsets_list, dtype=np.int32)
