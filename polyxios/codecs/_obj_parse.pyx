# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""One pass over a plain OBJ file, the compiled fast path of ``_obj.read``.

A plain file is what most mesh exporters write: ``v x y z`` and ``f i j k``
lines, comments, blank lines and ``s`` smoothing groups, in ASCII. Anything
else - normals, texture coordinates, groups, materials, a ``v`` of four
values, a negative or slashed index, a byte outside ASCII - makes
:func:`parse_plain` give up and return None, and the Python reader parses the
file from the start. Its result is the one the Python reader gives, value for
value: numbers are read by ``PyOS_string_to_double``, the routine ``float``
uses, and only tokens that routine reads to their last character are taken.
"""

from cpython.object cimport PyObject
from libc.stdint cimport int32_t, int64_t
from libc.stdlib cimport free, realloc
from libc.string cimport memcpy

import numpy as np


cdef extern from "Python.h":
    double PyOS_string_to_double(
        const char *s, char **endptr, PyObject *overflow_exception
    ) except? -1.0


cdef enum:
    MAX_TOKEN = 64
    INT32_MAX = 2147483647


cdef struct Buffer:
    char *data
    Py_ssize_t size
    Py_ssize_t capacity
    Py_ssize_t item


cdef int _reserve(Buffer *buf, Py_ssize_t more) except -1:
    """Grow ``buf`` so it holds ``more`` items past its size."""
    cdef Py_ssize_t capacity = buf.capacity if buf.capacity else 1024
    cdef char *grown
    while buf.size + more > capacity:
        capacity *= 2
    if capacity != buf.capacity:
        grown = <char *>realloc(buf.data, capacity * buf.item)
        if grown == NULL:
            raise MemoryError()
        buf.data = grown
        buf.capacity = capacity
    return 0


cdef inline bint _blank(unsigned char c) noexcept nogil:
    """Whitespace that separates tokens inside a line, as ``str.split`` sees it."""
    return c == 32 or c == 9 or c == 11 or c == 12


cdef inline bint _numeric(unsigned char c) noexcept nogil:
    """A character a plain coordinate may hold."""
    return (48 <= c <= 57) or c == 46 or c == 43 or c == 45 or c == 101 or c == 69


def parse_plain(const unsigned char[::1] raw):
    """Read a plain OBJ file, or return None for any other.

    Parameters
    ----------
    raw
        The whole file.

    Returns
    -------
    tuple or None
        The ``(n, 3)`` float64 vertices, the int32 connectivity (0-based)
        and the int32 vertex count of each face, in file order; None when
        the file is not plain, which leaves it to the Python reader.
    """
    cdef Py_ssize_t n = raw.shape[0]
    cdef Py_ssize_t pos = 0, start, end, line_end, k
    cdef unsigned char c, head
    cdef int n_tokens
    cdef int64_t value
    cdef int64_t n_vertices = 0
    cdef int32_t size
    cdef double coords[3]
    cdef char token[MAX_TOKEN + 1]
    cdef char *stop
    cdef Buffer vertices, connectivity, sizes

    for k in range(n):
        c = raw[k]
        if c >= 128 or 28 <= c <= 31:
            return None

    vertices.data = NULL
    vertices.size = vertices.capacity = 0
    vertices.item = 3 * sizeof(double)
    connectivity.data = NULL
    connectivity.size = connectivity.capacity = 0
    connectivity.item = sizeof(int32_t)
    sizes.data = NULL
    sizes.size = sizes.capacity = 0
    sizes.item = sizeof(int32_t)

    try:
        while pos < n:
            line_end = pos
            while line_end < n and raw[line_end] != 10 and raw[line_end] != 13:
                line_end += 1

            start = pos
            while start < line_end and _blank(raw[start]):
                start += 1
            pos = line_end + 1
            if start == line_end or raw[start] == 35:
                continue

            end = start
            while end < line_end and not _blank(raw[end]):
                end += 1
            if end - start != 1:
                return None
            head = raw[start] | 32

            if head == 115:
                # A smoothing group shades the faces and names no geometry;
                # the Python reader steps over it the same way.
                continue

            if head == 118:
                n_tokens = 0
                while True:
                    start = end
                    while start < line_end and _blank(raw[start]):
                        start += 1
                    if start == line_end:
                        break
                    end = start
                    while end < line_end and not _blank(raw[end]):
                        if not _numeric(raw[end]):
                            return None
                        end += 1
                    if n_tokens == 3 or end - start > MAX_TOKEN:
                        return None
                    memcpy(token, &raw[start], end - start)
                    token[end - start] = 0
                    try:
                        coords[n_tokens] = PyOS_string_to_double(token, &stop, NULL)
                    except ValueError:
                        return None
                    if stop != token + (end - start):
                        return None
                    n_tokens += 1
                if n_tokens != 3:
                    return None
                _reserve(&vertices, 1)
                memcpy(
                    vertices.data + vertices.size * vertices.item,
                    coords,
                    vertices.item,
                )
                vertices.size += 1
                n_vertices += 1
                continue

            if head == 102:
                size = 0
                while True:
                    start = end
                    while start < line_end and _blank(raw[start]):
                        start += 1
                    if start == line_end:
                        break
                    end = start
                    value = 0
                    while end < line_end and not _blank(raw[end]):
                        c = raw[end]
                        if not 48 <= c <= 57 or value > INT32_MAX:
                            return None
                        value = value * 10 + (c - 48)
                        end += 1
                    # An index counts from one and names a vertex declared
                    # above it; zero, or one past those, is an error the
                    # Python reader reports with its line.
                    if value < 1 or value > n_vertices:
                        return None
                    _reserve(&connectivity, 1)
                    (<int32_t *>connectivity.data)[connectivity.size] = <int32_t>(value - 1)
                    connectivity.size += 1
                    size += 1
                _reserve(&sizes, 1)
                (<int32_t *>sizes.data)[sizes.size] = size
                sizes.size += 1
                continue

            return None

        if not n_vertices:
            return None

        out_vertices = np.empty((vertices.size, 3), dtype=np.float64)
        out_connectivity = np.empty(connectivity.size, dtype=np.int32)
        out_sizes = np.empty(sizes.size, dtype=np.int32)
        _copy_out(out_vertices, &vertices)
        _copy_out(out_connectivity, &connectivity)
        _copy_out(out_sizes, &sizes)
        return out_vertices, out_connectivity, out_sizes
    finally:
        free(vertices.data)
        free(connectivity.data)
        free(sizes.data)


cdef int _copy_out(object array, Buffer *buf) except -1:
    """Copy a buffer's items into the numpy array made for them."""
    cdef unsigned char[::1] target = array.reshape(-1).view(np.uint8)
    if buf.size:
        memcpy(&target[0], buf.data, buf.size * buf.item)
    return 0
