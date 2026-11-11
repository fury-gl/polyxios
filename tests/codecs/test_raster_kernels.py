import sys

import numpy as np
import pytest

from polyxios.codecs import _raster_kernels_fallback as fb
from polyxios.exceptions import CodecError

try:
    from polyxios.codecs import _raster_kernels as compiled
except ImportError:  # pragma: no cover - a pure-Python install
    compiled = None

_IMPLS = [m for m in (compiled, fb) if m is not None]
_IDS = [m.__name__.rsplit(".", 1)[-1] for m in _IMPLS]


# ------------------------------------------------------------- helpers


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_filter_reference(rows: np.ndarray, types: list[int], bpp: int) -> bytes:
    """Filter ``rows`` the slow, obvious way; the test's own encoder."""
    out = bytearray()
    prev = [0] * rows.shape[1]
    for r, row in enumerate(rows.tolist()):
        t = types[r]
        out.append(t)
        for i, x in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            pred = {0: 0, 1: a, 2: b, 3: (a + b) // 2, 4: _paeth(a, b, c)}[t]
            out.append((x - pred) & 0xFF)
        prev = row
    return bytes(out)


# ----------------------------------------------------------------- PNG


@pytest.mark.parametrize("impl", _IMPLS, ids=_IDS)
@pytest.mark.parametrize("bpp", range(1, 9))
@pytest.mark.parametrize("pixels", [1, 2, 5])
@pytest.mark.parametrize("first", range(5))
# Bytes below 4 make the Paeth candidates tie, which random bytes almost never do.
@pytest.mark.parametrize("high", [256, 4])
# A row that ends mid-pixel, which the Sub filter has to pad.
@pytest.mark.parametrize("tail", [0, 1])
def test_png_unfilter_inverts_every_filter(
    impl, bpp: int, pixels: int, first: int, high: int, tail: int
) -> None:
    height = 6
    row_bytes = pixels * bpp + tail
    types = [(first + r) % 5 for r in range(height)]
    rng = np.random.default_rng(bpp * 31 + pixels)
    rows = rng.integers(0, high, size=(height, row_bytes), dtype=np.uint8)
    data = _png_filter_reference(rows, types, bpp)
    out = impl.png_unfilter(data, height, row_bytes, bpp)
    assert type(out) is bytearray
    assert bytes(out) == rows.tobytes()


@pytest.mark.parametrize("impl", _IMPLS, ids=_IDS)
def test_png_unfilter_rejects_bad_filter_and_short_data(impl) -> None:
    rows = np.zeros((2, 4), np.uint8)
    data = bytearray(_png_filter_reference(rows, [0, 0], 1))
    data[5] = 5  # row 1's filter byte
    with pytest.raises(
        CodecError, match="PNG row 1 uses filter type 5; only 0-4 exist."
    ):
        impl.png_unfilter(bytes(data), 2, 4, 1)
    with pytest.raises(CodecError, match="PNG image data ends early."):
        impl.png_unfilter(bytes(data[:-1]), 2, 4, 1)


@pytest.mark.parametrize("impl", _IMPLS, ids=_IDS)
@pytest.mark.parametrize(
    ("height", "row_bytes", "bpp"), [(1, 0, 0), (-1, 0, 1), (0, -1, 1)]
)
def test_png_unfilter_rejects_bad_geometry(impl, height, row_bytes, bpp) -> None:
    with pytest.raises(
        ValueError,
        match="png_unfilter needs height >= 0, row_bytes >= 0 and bpp >= 1",
    ):
        impl.png_unfilter(b"\0", height, row_bytes, bpp)


@pytest.mark.parametrize("impl", _IMPLS, ids=_IDS)
def test_png_unfilter_refuses_a_size_that_wraps_around(impl) -> None:
    # 3 * (row_bytes + 1) wraps past the word size to 5, the length of the data.
    row_bytes = (2 * (sys.maxsize + 1) + 5) // 3 - 1
    with pytest.raises(CodecError, match="PNG image data ends early."):
        impl.png_unfilter(bytes(5), 3, row_bytes, 1)


@pytest.mark.parametrize("impl", _IMPLS, ids=_IDS)
@pytest.mark.parametrize("data", [b"", b"\0\0\0"])
@pytest.mark.parametrize("row_bytes", [4, sys.maxsize])
def test_png_unfilter_of_zero_rows_is_empty(impl, data: bytes, row_bytes: int) -> None:
    out = impl.png_unfilter(data, 0, row_bytes, 1)
    assert type(out) is bytearray
    assert out == bytearray()


@pytest.mark.parametrize("impl", _IMPLS, ids=_IDS)
@pytest.mark.parametrize("kind", range(5))
def test_png_unfilter_with_a_pixel_wider_than_the_row_changes_nothing(
    impl, kind: int
) -> None:
    # No byte has a left neighbour, and the row has none above it.
    out = impl.png_unfilter(bytes([kind, 7, 9]), 1, 2, sys.maxsize)
    assert out == bytearray(b"\x07\x09")
