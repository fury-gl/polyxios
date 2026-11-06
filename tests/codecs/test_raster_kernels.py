import numpy as np
import pytest

from polyxios.codecs import _raster_kernels_fallback as fb
from polyxios.exceptions import CodecError

try:
    from polyxios.codecs import _raster_kernels as compiled
except ImportError:  # pragma: no cover - a pure-Python install
    compiled = None

_IMPLS = [m for m in (compiled, fb) if m is not None]
_needs_compiled = pytest.mark.skipif(
    compiled is None, reason="compiled _raster_kernels not built"
)


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


def _canonical(entries: list[tuple[int, int]]):
    """Build a Huffman table from ``(symbol, length)`` pairs.

    Returns the ``(maxcode, valoffset, huffval)`` triple the kernels read, and
    the 256-entry ``codes`` and ``sizes`` the encoders read, indexed by symbol.
    """
    entries = sorted(entries, key=lambda e: (e[1], e[0]))
    maxcode = np.full(18, -1, np.int32)
    maxcode[17] = 0x7FFFFFFF
    valoffset = np.zeros(17, np.int32)
    huffval = np.zeros(256, np.uint8)
    codes = np.zeros(256, np.uint16)
    sizes = np.zeros(256, np.uint8)
    code = 0
    prev = entries[0][1]
    seen: set[int] = set()
    for idx, (sym, length) in enumerate(entries):
        code <<= length - prev
        prev = length
        huffval[idx] = sym
        codes[sym] = code
        sizes[sym] = length
        maxcode[length] = code
        if length not in seen:
            seen.add(length)
            valoffset[length] = idx - code
        code += 1
    return (maxcode, valoffset, huffval), codes, sizes


def _baseline_tables():
    dc, dc_codes, dc_sizes = _canonical([(s, 4) for s in range(12)])
    ac, ac_codes, ac_sizes = _canonical([(s, 8) for s in range(256)])
    enc = (
        np.stack([dc_codes[:12], dc_codes[:12]]),
        np.stack([dc_sizes[:12], dc_sizes[:12]]),
        np.stack([ac_codes, ac_codes]),
        np.stack([ac_sizes, ac_sizes]),
    )
    return dc, ac, enc


def _lossless_table():
    table, codes, sizes = _canonical([(s, 5) for s in range(17)])
    return table, codes[:17], sizes[:17]


def _encode_blocks(coefs: np.ndarray, restart: int) -> bytes:
    """Entropy-code ``coefs`` (rows, cols, 64) as one baseline component.

    Each restart chunk is encoded on its own, so the DC predictor restarts
    exactly where a restart marker sits.
    """
    dc, ac, (dcc, dcs, acc, acs) = _baseline_tables()
    flat = coefs.reshape(-1, 64).astype(np.int32)
    n = flat.shape[0]
    step = restart or n
    chunks = []
    for start in range(0, n, step):
        part = flat[start : start + step]
        zeros = np.zeros(part.shape[0], np.uint8)
        chunks.append(jpeg_encode(part, zeros, dcc, dcs, acc, acs))
    body = bytearray()
    for i, chunk in enumerate(chunks):
        if i:
            body += bytes([0xFF, 0xD0 + ((i - 1) & 7)])
        body += chunk
    return bytes(body) + b"\xff\xd9"


def jpeg_encode(part, comp, dcc, dcs, acc, acs) -> bytes:
    return fb.jpeg_encode_scan(
        part, comp, np.zeros(part.shape[0], np.uint8), dcc, dcs, acc, acs, 1
    )


def _decode_blocks(data: bytes, shape: tuple[int, int], dc, ac, restart: int, impl=fb):
    rows, cols = shape
    coefs = np.zeros((rows, cols, 64), np.int16)
    end = impl.jpeg_decode_scan(
        data,
        0,
        [coefs],
        [1],
        [1],
        [cols],
        [rows],
        [dc],
        [ac],
        cols,
        rows,
        0,
        63,
        0,
        0,
        restart,
        False,
    )
    return coefs, end


def _canonical_lossless_diffs(
    planes: np.ndarray, predictor: int, precision: int, interval: int
) -> np.ndarray:
    """Reference differences in scan order, one ``C``-wide group per pixel."""
    chans, height, width = planes.shape
    start = 1 << (precision - 1)
    out = np.zeros((height, width, chans), np.int64)
    for c in range(chans):
        p = planes[c].astype(np.int64)
        for y in range(height):
            for x in range(width):
                m = y * width + x
                if m == 0 or (interval and m % interval == 0):
                    pred = start
                elif y == 0:
                    pred = int(p[y, x - 1])
                elif x == 0:
                    pred = int(p[y - 1, x])
                else:
                    ra, rb, rc = (
                        int(p[y, x - 1]),
                        int(p[y - 1, x]),
                        int(p[y - 1, x - 1]),
                    )
                    pred = {
                        1: ra,
                        2: rb,
                        3: rc,
                        4: ra + rb - rc,
                        5: ra + ((rb - rc) >> 1),
                        6: rb + ((ra - rc) >> 1),
                        7: (ra + rb) >> 1,
                    }[predictor]
                out[y, x, c] = ((int(p[y, x]) - pred + 32767) & 0xFFFF) - 32767
    return out.reshape(-1)


def _lossless_stream(
    planes: np.ndarray, predictor: int, precision: int, interval: int
) -> bytes:
    chans, height, width = planes.shape
    table, code, size = _lossless_table()
    diffs = _canonical_lossless_diffs(planes, predictor, precision, interval).astype(
        np.int32
    )
    step = (interval or height * width) * chans
    body = bytearray()
    for i, start in enumerate(range(0, diffs.size, step)):
        if i:
            body += bytes([0xFF, 0xD0 + ((i - 1) & 7)])
        body += fb.jpeg_lossless_encode(diffs[start : start + step], code, size)
    return bytes(body) + b"\xff\xd9"


def _decode_lossless(
    data: bytes,
    chans: int,
    height: int,
    width: int,
    predictor: int,
    precision: int,
    interval: int,
    impl=fb,
):
    table, _, _ = _lossless_table()
    samples = [np.zeros((height, width), np.int32) for _ in range(chans)]
    end = impl.jpeg_lossless_decode_scan(
        data,
        0,
        samples,
        [table] * chans,
        predictor,
        0,
        precision,
        interval,
        width,
        height,
    )
    return np.stack(samples), end


# ---------------------------------------------------------------- PNG


@pytest.mark.parametrize("bpp", [1, 3, 4])
def test_png_unfilter_matches_reference_filter(bpp: int) -> None:
    rng = np.random.default_rng(bpp)
    height, row_bytes = 9, 10
    rows = rng.integers(0, 256, size=(height, row_bytes), dtype=np.uint8)
    types = [r % 5 for r in range(height)]
    data = _png_filter_reference(rows, types, bpp)
    out = fb.png_unfilter(data, height, row_bytes, bpp)
    assert bytes(out) == rows.tobytes()


def test_png_unfilter_rejects_bad_filter_and_short_data() -> None:
    rows = np.zeros((2, 4), np.uint8)
    data = bytearray(_png_filter_reference(rows, [0, 0], 1))
    data[5] = 5  # row 1's filter byte
    with pytest.raises(
        CodecError, match="PNG row 1 uses filter type 5; only 0-4 exist."
    ):
        fb.png_unfilter(bytes(data), 2, 4, 1)
    with pytest.raises(CodecError, match="PNG image data ends early."):
        fb.png_unfilter(bytes(data[:-1]), 2, 4, 1)


@pytest.mark.parametrize("impl", _IMPLS)
def test_png_unfilter_implementations_are_identical(impl) -> None:
    rng = np.random.default_rng(7)
    rows = rng.integers(0, 256, size=(12, 17), dtype=np.uint8)
    data = _png_filter_reference(rows, [r % 5 for r in range(12)], 4)
    assert bytes(impl.png_unfilter(data, 12, 17, 4)) == rows.tobytes()
    assert bytes(impl.png_unfilter(data, 12, 17, 4)) == bytes(
        fb.png_unfilter(data, 12, 17, 4)
    )


# ------------------------------------------------------- baseline scans


@pytest.mark.parametrize("restart", [0, 7])
def test_baseline_scan_round_trip(restart: int) -> None:
    rng = np.random.default_rng(restart)
    coefs = np.zeros((3, 4, 64), np.int16)
    coefs[:, :, 0] = rng.integers(-1000, 1001, size=(3, 4))
    ac = rng.integers(-30, 31, size=(3, 4, 64))
    ac[rng.random(ac.shape) < 0.6] = 0
    coefs[:, :, 1:] = ac[:, :, 1:]
    dc, ac_table, _ = _baseline_tables()
    data = _encode_blocks(coefs, restart)
    decoded, end = _decode_blocks(data, (3, 4), dc, ac_table, restart)
    np.testing.assert_array_equal(decoded, coefs)
    assert data[end : end + 2] == b"\xff\xd9"


def test_baseline_scan_rejects_truncation_and_bad_restart() -> None:
    coefs = np.ones((2, 2, 64), np.int16) * 3
    dc, ac, _ = _baseline_tables()
    data = _encode_blocks(coefs, 2)
    with pytest.raises(CodecError, match="ends early"):
        _decode_blocks(data[:6], (2, 2), dc, ac, 2)
    bad = bytearray(data)
    marker = data.index(b"\xff\xd0")
    bad[marker + 1] = 0xD1
    with pytest.raises(CodecError, match="restart marker out of sequence"):
        _decode_blocks(bytes(bad), (2, 2), dc, ac, 2)


def test_baseline_scan_rejects_invalid_huffman_code() -> None:
    # One DC code, "0"; a run of 1-bits matches no code of length up to 16.
    dc, _, _ = _canonical([(0, 1)])
    ac, _, _ = _canonical([(0, 1)])
    data = b"\xff\x00" * 4 + b"\xff\xd9"
    with pytest.raises(CodecError, match="invalid Huffman code"):
        fb.jpeg_decode_scan(
            data,
            0,
            [np.zeros((1, 1, 64), np.int16)],
            [1],
            [1],
            [1],
            [1],
            [dc],
            [ac],
            1,
            1,
            0,
            63,
            0,
            0,
            0,
            False,
        )


def test_encode_scan_rejects_oversized_dc_difference() -> None:
    _, _, (dcc, dcs, acc, acs) = _baseline_tables()
    part = np.zeros((2, 64), np.int32)
    part[1, 0] = 5000
    with pytest.raises(CodecError):
        jpeg_encode(part, np.zeros(2, np.uint8), dcc, dcs, acc, acs)


# ----------------------------------------------------------- lossless


@pytest.mark.parametrize("predictor", [1, 2, 3, 4, 5, 6, 7])
def test_lossless_round_trip_every_predictor(predictor: int) -> None:
    rng = np.random.default_rng(predictor)
    planes = rng.integers(0, 256, size=(1, 5, 6)).astype(np.int32)
    data = _lossless_stream(planes, predictor, 8, 0)
    decoded, end = _decode_lossless(data, 1, 5, 6, predictor, 8, 0)
    np.testing.assert_array_equal(decoded, planes)
    assert data[end : end + 2] == b"\xff\xd9"


@pytest.mark.parametrize(
    ("precision", "interval"),
    [(12, 0), (8, 6), (16, 0)],
)
def test_lossless_round_trip_precision_and_restarts(
    precision: int, interval: int
) -> None:
    rng = np.random.default_rng(precision + interval)
    planes = rng.integers(0, 1 << precision, size=(3, 4, 6)).astype(np.int32)
    data = _lossless_stream(planes, 7, precision, interval)
    decoded, _ = _decode_lossless(data, 3, 4, 6, 7, precision, interval)
    np.testing.assert_array_equal(decoded, planes)


def test_lossless_round_trip_one_line_restart() -> None:
    rng = np.random.default_rng(3)
    planes = rng.integers(0, 256, size=(1, 4, 5)).astype(np.int32)
    data = _lossless_stream(planes, 4, 8, 5)
    decoded, _ = _decode_lossless(data, 1, 4, 5, 4, 8, 5)
    np.testing.assert_array_equal(decoded, planes)


def test_lossless_encode_extremes_round_trip() -> None:
    planes = np.array([[[0, 65535, 0, 65535], [65535, 0, 32768, 1]]], np.int32)
    data = _lossless_stream(planes, 1, 16, 0)
    decoded, _ = _decode_lossless(data, 1, 2, 4, 1, 16, 0)
    np.testing.assert_array_equal(decoded, planes)


def test_lossless_decode_rejects_bad_predictor_and_truncation() -> None:
    planes = np.arange(12, dtype=np.int32).reshape(1, 3, 4)
    data = _lossless_stream(planes, 1, 8, 0)
    with pytest.raises(CodecError, match="predictor 0 is not 1-7"):
        _decode_lossless(data, 1, 3, 4, 0, 8, 0)
    with pytest.raises(CodecError, match="ends early"):
        _decode_lossless(data[:3], 1, 3, 4, 1, 8, 0)


def test_lossless_decode_rejects_out_of_sequence_restart() -> None:
    planes = np.arange(12, dtype=np.int32).reshape(1, 3, 4)
    data = bytearray(_lossless_stream(planes, 1, 8, 4))
    data[data.index(b"\xff\xd0") + 1] = 0xD3
    with pytest.raises(CodecError, match="restart marker out of sequence"):
        _decode_lossless(bytes(data), 1, 3, 4, 1, 8, 4)


def test_lossless_encode_rejects_out_of_range_difference() -> None:
    _, code, size = _lossless_table()
    with pytest.raises(CodecError, match="out of range"):
        fb.jpeg_lossless_encode(np.array([40000], np.int32), code, size)


# ---------------------------------------------------------- agreement


@_needs_compiled
def test_compiled_png_unfilter_agrees_with_fallback() -> None:
    rng = np.random.default_rng(11)
    rows = rng.integers(0, 256, size=(16, 21), dtype=np.uint8)
    data = _png_filter_reference(rows, [r % 5 for r in range(16)], 3)
    assert bytes(compiled.png_unfilter(data, 16, 21, 3)) == bytes(
        fb.png_unfilter(data, 16, 21, 3)
    )


@_needs_compiled
@pytest.mark.parametrize("restart", [0, 5])
def test_compiled_huffman_kernels_agree_with_fallback(restart: int) -> None:
    rng = np.random.default_rng(restart + 1)
    coefs = np.zeros((4, 3, 64), np.int16)
    coefs[:, :, 0] = rng.integers(-800, 801, size=(4, 3))
    coefs[:, :, 1:] = rng.integers(-20, 21, size=(4, 3, 63)) * (
        rng.random((4, 3, 63)) < 0.4
    )
    dc, ac, (dcc, dcs, acc, acs) = _baseline_tables()
    part = coefs.reshape(-1, 64).astype(np.int32)
    zeros = np.zeros(part.shape[0], np.uint8)
    fast = compiled.jpeg_encode_scan(part, zeros, zeros, dcc, dcs, acc, acs, 1)
    assert fast == fb.jpeg_encode_scan(part, zeros, zeros, dcc, dcs, acc, acs, 1)
    data = _encode_blocks(coefs, restart)
    fast_coefs, fast_end = _decode_blocks(data, (4, 3), dc, ac, restart, impl=compiled)
    slow_coefs, slow_end = _decode_blocks(data, (4, 3), dc, ac, restart, impl=fb)
    np.testing.assert_array_equal(fast_coefs, slow_coefs)
    assert fast_end == slow_end


@_needs_compiled
@pytest.mark.parametrize("predictor", [1, 4, 7])
def test_compiled_lossless_kernels_agree_with_fallback(predictor: int) -> None:
    rng = np.random.default_rng(predictor)
    planes = rng.integers(0, 4096, size=(3, 4, 5)).astype(np.int32)
    data = _lossless_stream(planes, predictor, 12, 7)
    fast, fast_end = _decode_lossless(data, 3, 4, 5, predictor, 12, 7, impl=compiled)
    slow, slow_end = _decode_lossless(data, 3, 4, 5, predictor, 12, 7, impl=fb)
    np.testing.assert_array_equal(fast, slow)
    assert fast_end == slow_end
    _, code, size = _lossless_table()
    diffs = _canonical_lossless_diffs(planes, predictor, 12, 7).astype(np.int32)
    assert compiled.jpeg_lossless_encode(diffs, code, size) == fb.jpeg_lossless_encode(
        diffs, code, size
    )
