"""PNG scanline unfiltering in pure Python and NumPy.

The fallback for ``_raster_kernels``, the compiled twin: the same function
with the same results and the same errors for ``bytes``, a ``bytearray`` or a
contiguous ``memoryview`` of one, and sizes that fit a machine word. ``_png``
imports whichever is present.
"""

import numpy as np

from polyxios.exceptions import CodecError


def png_unfilter(
    data: bytes | bytearray | memoryview, height: int, row_bytes: int, bpp: int
) -> bytearray:
    """Reverse the PNG per-row filters.

    Parameters
    ----------
    data
        ``height`` rows of one filter byte followed by ``row_bytes`` filtered
        bytes.
    height
        Number of rows.
    row_bytes
        Bytes per unfiltered row.
    bpp
        Bytes per complete pixel, at least 1; the filters reach back this far.

    Returns
    -------
    bytearray
        The ``height * row_bytes`` unfiltered bytes.

    Raises
    ------
    ValueError
        If ``height`` or ``row_bytes`` is negative or ``bpp`` is below 1.
    CodecError
        If the data is short, or a row uses a filter type above 4.
    """
    if height < 0 or row_bytes < 0 or bpp < 1:
        raise ValueError(
            "png_unfilter needs height >= 0, row_bytes >= 0 and bpp >= 1, got "
            f"{height}, {row_bytes} and {bpp}."
        )
    stride = row_bytes + 1
    if len(data) < height * stride:
        raise CodecError("PNG image data ends early.")
    if height == 0:
        return bytearray()
    out = bytearray(height * row_bytes)
    prev = bytes(row_bytes)
    for r in range(height):
        base = r * stride
        kind = data[base]
        if kind > 4:
            raise CodecError(f"PNG row {r} uses filter type {kind}; only 0-4 exist.")
        raw = data[base + 1 : base + stride]
        if kind == 0:
            cur = raw
        elif kind == 1:
            cur = _unfilter_sub(raw, bpp)
        elif kind == 2:
            cur = (
                np.frombuffer(raw, np.uint8) + np.frombuffer(prev, np.uint8)
            ).tobytes()
        elif kind == 3:
            cur = _unfilter_average(raw, prev, bpp)
        else:
            cur = _unfilter_paeth(raw, prev, bpp)
        start = r * row_bytes
        out[start : start + row_bytes] = cur
        prev = cur
    return out


def _unfilter_sub(raw: bytes, bpp: int) -> bytes:
    """Sub filter: a running sum per byte lane, which NumPy's wrapping cumsum does."""
    if bpp >= len(raw):
        return bytes(raw)  # no byte has a left neighbour
    arr = np.frombuffer(raw, np.uint8)
    pad = (-arr.size) % bpp
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, np.uint8)])
    summed = np.cumsum(arr.reshape(-1, bpp), axis=0, dtype=np.uint8).reshape(-1)
    return summed[: len(raw)].tobytes()


def _unfilter_average(raw: bytes, prev: bytes, bpp: int) -> bytearray:
    cur = bytearray(len(raw))
    for i in range(len(raw)):
        left = cur[i - bpp] if i >= bpp else 0
        cur[i] = (raw[i] + ((left + prev[i]) >> 1)) & 0xFF
    return cur


def _unfilter_paeth(raw: bytes, prev: bytes, bpp: int) -> bytearray:
    cur = bytearray(len(raw))
    for i in range(len(raw)):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
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
        cur[i] = (raw[i] + pred) & 0xFF
    return cur
