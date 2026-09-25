# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""LZF block compression, the compiled twin of ``_lzf_fallback``.

Same algorithm, same window, same match limit, same output byte for byte as
the fallback; only the speed differs. The stream is one any LZF decoder
reads, liblzf's included, not liblzf's own. ``_pcd`` imports whichever of
the two is present.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from polyxios.exceptions import CodecError

cdef enum:
    MAX_LIT = 32
    MAX_OFF = 1 << 13
    MAX_REF = (1 << 8) + (1 << 3)
    HLOG = 16
    HSIZE = 1 << HLOG


def decompress(const unsigned char[::1] data, Py_ssize_t out_len):
    """Expand one LZF block to exactly ``out_len`` bytes.

    Parameters
    ----------
    data
        The compressed bytes.
    out_len
        The length the block expands to.

    Returns
    -------
    bytes
        The expanded bytes.

    Raises
    ------
    CodecError
        When the stream ends mid-instruction, a back-reference reaches
        before the output start, or the block does not expand to ``out_len``
        bytes exactly.
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t i = 0, o = 0, run, length, offset, start, k
    cdef unsigned int ctrl
    cdef unsigned char *out = <unsigned char *> malloc(out_len if out_len > 0 else 1)
    if out == NULL:
        raise MemoryError()
    try:
        while i < n:
            ctrl = data[i]
            i += 1
            if ctrl < MAX_LIT:
                run = ctrl + 1
                if i + run > n:
                    raise CodecError("LZF stream ends inside a literal run.")
                if o + run > out_len:
                    raise CodecError(
                        f"LZF block expands past the {out_len} bytes the header promised."
                    )
                memcpy(out + o, &data[i], run)
                i += run
                o += run
                continue
            length = ctrl >> 5
            if length == 7:
                if i >= n:
                    raise CodecError("LZF stream ends inside a back-reference.")
                length += data[i]
                i += 1
            if i >= n:
                raise CodecError("LZF stream ends inside a back-reference.")
            offset = ((ctrl & 0x1F) << 8) + data[i] + 1
            i += 1
            length += 2
            start = o - offset
            if start < 0:
                raise CodecError("LZF back-reference reaches before the output start.")
            if o + length > out_len:
                raise CodecError(
                    f"LZF block expands past the {out_len} bytes the header promised."
                )
            for k in range(length):
                out[o + k] = out[start + k]
            o += length
        if o != out_len:
            raise CodecError(
                f"LZF block expands to {o} bytes, the header promised {out_len}."
            )
        return PyBytes_FromStringAndSize(<char *> out, out_len)
    finally:
        free(out)


cdef inline unsigned int _hash(const unsigned char *p) noexcept nogil:
    cdef unsigned int h = (p[0] << 16) | (p[1] << 8) | p[2]
    return ((h * 2654435761u) >> (24 - HLOG + 8)) & (HSIZE - 1)


def compress(const unsigned char[::1] data):
    """Compress one block into a stream any LZF decoder reads.

    Parameters
    ----------
    data
        The bytes to compress.

    Returns
    -------
    bytes
        The LZF stream. Incompressible input grows by one byte per 32.
    """
    cdef Py_ssize_t n = data.shape[0]
    if n == 0:
        return b""
    cdef Py_ssize_t cap = n + n // 32 + 8
    cdef unsigned char *out = <unsigned char *> malloc(cap)
    cdef Py_ssize_t *htab = <Py_ssize_t *> malloc(HSIZE * sizeof(Py_ssize_t))
    if out == NULL or htab == NULL:
        free(out)
        free(htab)
        raise MemoryError()
    cdef Py_ssize_t i = 0, o = 0, lit_start = 0, lit = 0
    cdef Py_ssize_t ref, off, length, maxlen, j, chunk
    cdef unsigned int h
    cdef const unsigned char *p = &data[0]
    try:
        for j in range(HSIZE):
            htab[j] = -1
        while i < n:
            if i + 2 < n:
                h = _hash(p + i)
                ref = htab[h]
                htab[h] = i
                off = i - ref - 1
                if (
                    ref >= 0
                    and off < MAX_OFF
                    and p[ref] == p[i]
                    and p[ref + 1] == p[i + 1]
                    and p[ref + 2] == p[i + 2]
                ):
                    # The three hashed bytes are known equal, so the match
                    # starts at 3; a shorter one would encode as a control
                    # byte below 32, which every decoder reads as a literal.
                    length = 3
                    maxlen = n - i - 2
                    if maxlen > MAX_REF:
                        maxlen = MAX_REF
                    while length < maxlen and p[ref + length] == p[i + length]:
                        length += 1
                    length -= 2
                    # Flush the pending literals in runs of at most 32.
                    while lit > 0:
                        chunk = lit if lit < MAX_LIT else MAX_LIT
                        out[o] = <unsigned char> (chunk - 1)
                        o += 1
                        memcpy(out + o, p + lit_start, chunk)
                        o += chunk
                        lit_start += chunk
                        lit -= chunk
                    if length < 7:
                        out[o] = <unsigned char> ((off >> 8) + (length << 5))
                        o += 1
                    else:
                        out[o] = <unsigned char> ((off >> 8) + (7 << 5))
                        out[o + 1] = <unsigned char> (length - 7)
                        o += 2
                    out[o] = <unsigned char> (off & 0xFF)
                    o += 1
                    i += length + 2
                    lit_start = i
                    if i - 1 > 0 and i + 1 < n:
                        htab[_hash(p + i - 1)] = i - 1
                    continue
            lit += 1
            i += 1
        while lit > 0:
            chunk = lit if lit < MAX_LIT else MAX_LIT
            out[o] = <unsigned char> (chunk - 1)
            o += 1
            memcpy(out + o, p + lit_start, chunk)
            o += chunk
            lit_start += chunk
            lit -= chunk
        return PyBytes_FromStringAndSize(<char *> out, o)
    finally:
        free(out)
        free(htab)
