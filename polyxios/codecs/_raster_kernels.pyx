# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""PNG scanline unfiltering, compiled.

The compiled twin of ``_raster_kernels_fallback``: same function, same
results byte for byte, same errors for ``bytes``, a ``bytearray`` or a
contiguous ``memoryview`` of one, and sizes that fit a machine word. ``_png``
imports whichever of the two is present.
"""

from polyxios.exceptions import CodecError


def png_unfilter(
    const unsigned char[::1] data, Py_ssize_t height, Py_ssize_t row_bytes, Py_ssize_t bpp
):
    """Reverse the PNG per-row filters; see the fallback for the contract."""
    if height < 0 or row_bytes < 0 or bpp < 1:
        raise ValueError(
            "png_unfilter needs height >= 0, row_bytes >= 0 and bpp >= 1, got "
            f"{height}, {row_bytes} and {bpp}."
        )
    if height == 0:
        return bytearray()
    cdef Py_ssize_t available = data.shape[0]
    # data.shape[0] < height * (row_bytes + 1), by division so it cannot overflow.
    if available // height <= row_bytes:
        raise CodecError("PNG image data ends early.")
    cdef Py_ssize_t stride = row_bytes + 1
    cdef Py_ssize_t r, i, base, start, up
    cdef int kind, raw, a, b, c, p, pa, pb, pc, pred
    out = bytearray(height * row_bytes)
    cdef unsigned char[::1] o = out
    for r in range(height):
        base = r * stride
        kind = data[base]
        if kind > 4:
            raise CodecError(f"PNG row {r} uses filter type {kind}; only 0-4 exist.")
        start = r * row_bytes
        for i in range(row_bytes):
            raw = <int>data[base + 1 + i]
            up = <int>o[start - row_bytes + i] if r > 0 else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = <int>o[start + i - bpp] if i >= bpp else 0
            elif kind == 2:
                pred = up
            elif kind == 3:
                a = <int>o[start + i - bpp] if i >= bpp else 0
                pred = (a + up) >> 1
            else:
                a = <int>o[start + i - bpp] if i >= bpp else 0
                c = <int>o[start - row_bytes + i - bpp] if (r > 0 and i >= bpp) else 0
                b = up
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    pred = a
                elif pb <= pc:
                    pred = b
                else:
                    pred = c
            o[start + i] = (raw + pred) & 0xFF
    return out
