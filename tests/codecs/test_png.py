"""PNG codec: round trips, Adam7 and palette decoding, structure errors, Pillow."""

from dataclasses import replace
import io
import struct
import sys
import tracemalloc
import warnings
import zlib

import numpy as np
import pytest

import polyxios as px
from polyxios import helper
from polyxios._io import write_bytes
from polyxios.codecs import _png as png_codec, _raster_kernels_fallback
from polyxios.exceptions import CodecError, LazyReadError, ValidationError
from polyxios.helper import array_to_image

_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Adam7 as (y0, x0, dy, dx), written out here rather than imported, so the
# builders below check the codec against the spec and not against itself.
_ADAM7 = (
    (0, 0, 8, 8),
    (0, 4, 8, 8),
    (4, 0, 8, 4),
    (0, 2, 4, 4),
    (2, 0, 4, 2),
    (0, 1, 2, 2),
    (1, 0, 2, 1),
)
_WHOLE = ((0, 0, 1, 1),)


def _chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body))
    )


def _file(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Join chunks behind the signature, closing them with IEND."""
    return _SIGNATURE + b"".join(_chunk(k, b) for k, b in [*chunks, (b"IEND", b"")])


def _header(width: int, height: int, depth: int, color: int, interlaced=False) -> bytes:
    return struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, int(interlaced))


def _pack_row(values: np.ndarray, depth: int) -> bytes:
    """Pack one scanline of samples at the file's bit depth."""
    flat = np.asarray(values, dtype=np.int64).ravel()
    if depth == 16:
        return flat.astype(">u2").tobytes()
    if depth == 8:
        return flat.astype(np.uint8).tobytes()
    per_byte = 8 // depth
    padded = np.zeros(-(-flat.size // per_byte) * per_byte, dtype=np.int64)
    padded[: flat.size] = flat
    shifts = (per_byte - 1 - np.arange(per_byte)) * depth
    packed = (padded.reshape(-1, per_byte) << shifts).sum(axis=1)
    return packed.astype(np.uint8).tobytes()


def _scanlines(samples: np.ndarray, depth: int, interlaced: bool) -> bytes:
    """Filter-0 scanlines of ``samples`` (H, W, C), pass by pass for Adam7."""
    out = bytearray()
    for y0, x0, dy, dx in _ADAM7 if interlaced else _WHOLE:
        block = samples[y0::dy, x0::dx]
        if block.shape[0] == 0 or block.shape[1] == 0:
            continue  # an empty pass has no rows, not even a filter byte
        for row in block:
            out += b"\0" + _pack_row(row, depth)
    return bytes(out)


def _png(
    samples: np.ndarray,
    *,
    depth: int,
    color: int,
    interlaced: bool = False,
    before_idat: list[tuple[bytes, bytes]] | None = None,
) -> bytes:
    height, width = samples.shape[:2]
    data = zlib.compress(_scanlines(samples, depth, interlaced))
    return _file(
        [
            (b"IHDR", _header(width, height, depth, color, interlaced)),
            *(before_idat or []),
            (b"IDAT", data),
        ]
    )


def _gradient(height: int, width: int, channels: int = 1) -> np.ndarray:
    """Uint8 pixels with every channel different, top row first."""
    y, x = np.mgrid[0:height, 0:width]
    stack = [(x * 37 + y * 11 + c * 59) % 256 for c in range(channels)]
    out = np.stack(stack, axis=-1).astype(np.uint8)
    return out[..., 0] if channels == 1 else out


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    write_bytes(path, data)
    return path


def _pil():
    return pytest.importorskip("PIL.Image")


try:
    from polyxios.codecs import _raster_kernels as _compiled
except ImportError:  # pragma: no cover - a pure-Python install
    _compiled = None

_IMPLS = [m for m in (_compiled, _raster_kernels_fallback) if m is not None]


@pytest.fixture(params=_IMPLS, ids=[m.__name__.rsplit(".", 1)[-1] for m in _IMPLS])
def kernels(request, monkeypatch):
    monkeypatch.setattr(png_codec, "kernels", request.param)
    return request.param


# -----------------------------------------------------------------------------
# Round trips


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_round_trip_keeps_pixels_and_metadata(tmp_path, mode, dtype, kernels) -> None:
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    rng = np.random.default_rng(7)
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    pixels = rng.integers(
        0, np.iinfo(dtype).max, size=shape, endpoint=True, dtype=dtype
    )
    meta = {
        "dpi": (300.0, 150.0),
        "gamma": 0.45455,
        "icc_profile": b"\x00profile\xff",
        "exif": b"II*\x00\x08\x00\x00\x00",
        "text": {"Title": "Zo\u00eb", "Note": "\u65e5\u672c"},
    }
    path = tmp_path / "round.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        px.write_image(array_to_image(pixels, global_attrs=meta), path)
        back = px.read_image(path)

    got = helper.image_to_array(back)
    assert got.dtype == dtype
    np.testing.assert_array_equal(got.reshape(pixels.shape), pixels)
    assert back.global_attrs["dpi"] == pytest.approx((300.0, 150.0), rel=1e-4)
    assert back.global_attrs["gamma"] == pytest.approx(0.45455)
    assert back.global_attrs["icc_profile"] == meta["icc_profile"]
    assert back.global_attrs["exif"] == meta["exif"]
    assert back.global_attrs["text"] == meta["text"]


def _chunk_types(data: bytes) -> list[bytes]:
    kinds = []
    pos = len(_SIGNATURE)
    while pos < len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        kinds.append(data[pos + 4 : pos + 8])
        pos += 12 + length
    return kinds


def test_image_data_spans_several_idat_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(png_codec, "_IDAT_PIECE", 16)
    rng = np.random.default_rng(17)
    pixels = rng.integers(0, 256, size=(8, 9, 3), dtype=np.uint8)
    path = tmp_path / "many.png"
    px.write_image(array_to_image(pixels), path)
    assert _chunk_types(path.read_bytes()).count(b"IDAT") > 1
    np.testing.assert_array_equal(helper.image_to_array(px.read_image(path)), pixels)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_a_sub_byte_image_unfilters_a_byte_at_a_time(tmp_path, depth, kernels) -> None:
    rng = np.random.default_rng(depth)
    samples = rng.integers(0, 1 << depth, size=(3, 24, 1))
    raw = bytearray()
    for row in samples:
        packed = np.frombuffer(_pack_row(row, depth), np.uint8).astype(np.int16)
        left = np.concatenate([[0], packed[:-1]])
        raw += b"\x01" + ((packed - left) & 0xFF).astype(np.uint8).tobytes()
    path = _write(
        tmp_path,
        "sub.png",
        _file(
            [
                (b"IHDR", _header(24, 3, depth, 0)),
                (b"IDAT", zlib.compress(bytes(raw))),
            ]
        ),
    )
    got = helper.image_to_array(px.read_image(path)).reshape(3, 24)
    np.testing.assert_array_equal(got, _expected_gray(samples, depth))


def _chunk_body(data: bytes, kind: bytes) -> bytes:
    pos = len(_SIGNATURE)
    while pos < len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        if data[pos + 4 : pos + 8] == kind:
            return data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    raise AssertionError(f"no {kind!r} chunk")


def test_the_writer_rounds_dpi_and_gamma_to_the_values_pillow_writes(tmp_path) -> None:
    attrs = {"dpi": (72, 72), "gamma": 1 / 2.2}
    path = tmp_path / "bytes.png"
    px.write_image(array_to_image(_gradient(2, 2), global_attrs=attrs), path)
    data = path.read_bytes()
    assert _chunk_body(data, b"pHYs") == struct.pack(">IIB", 2835, 2835, 1)
    assert _chunk_body(data, b"gAMA") == struct.pack(">I", 45455)


def test_the_writer_puts_its_metadata_before_the_pixels(tmp_path) -> None:
    meta = {
        "dpi": (300.0, 150.0),
        "gamma": 0.45455,
        "icc_profile": b"profile",
        "exif": b"II*\x00\x08\x00\x00\x00",
        "text": {"Title": "Zo\u00eb", "Note": "\u65e5\u672c"},
    }
    path = tmp_path / "order.png"
    px.write_image(array_to_image(_gradient(2, 2), global_attrs=meta), path)
    kinds = _chunk_types(path.read_bytes())
    assert kinds == [
        b"IHDR",
        b"gAMA",
        b"iCCP",
        b"pHYs",
        b"tEXt",
        b"iTXt",
        b"eXIf",
        b"IDAT",
        b"IEND",
    ]


def test_write_forwards_its_options_to_write_image(tmp_path) -> None:
    poly = array_to_image(np.zeros((64, 64), np.uint8)).to_polydata()
    sizes = {}
    for level in (0, 9):
        path = tmp_path / f"fwd{level}.png"
        px.write(poly, path, compress_level=level)
        sizes[level] = path.stat().st_size
    assert sizes[0] > 4 * sizes[9]
    with pytest.warns(UserWarning, match="unrecognized options"):
        px.write(poly, tmp_path / "bogus.png", bogus=1)


def test_the_default_compress_level_is_6(tmp_path) -> None:
    rng = np.random.default_rng(23)
    image = array_to_image(rng.integers(0, 4, size=(64, 64), dtype=np.uint8))
    bytes_by_level = {}
    for name, opts in (("default", {}), ("six", {"compress_level": 6})):
        path = tmp_path / f"{name}.png"
        px.write_image(image, path, **opts)
        bytes_by_level[name] = path.read_bytes()
    path = tmp_path / "zero.png"
    px.write_image(image, path, compress_level=0)
    assert bytes_by_level["default"] == bytes_by_level["six"]
    assert len(bytes_by_level["default"]) < path.stat().st_size


def test_write_through_public_write_keeps_the_lattice(tmp_path) -> None:
    pixels = _gradient(4, 6, 3)
    poly = array_to_image(pixels).to_polydata()
    path = tmp_path / "lattice.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        px.write(poly, path)
        back = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(back, pixels)


def test_each_row_takes_the_cheapest_filter() -> None:
    ramp = np.arange(57, -1, -3, dtype=np.uint8)[None]
    out = png_codec._filter_rows(np.vstack([ramp, ramp]), 1)
    # Sub for the ramp, whose steps of -3 cost 3 as signed bytes; Up for its repeat.
    assert out[:, 0].tolist() == [1, 2]


def test_the_encoders_paeth_predictor_follows_the_specification() -> None:
    grid = np.arange(16, dtype=np.int16)
    left, up, upleft = (x.ravel() for x in np.meshgrid(grid, grid, grid, indexing="ij"))

    def paeth(a: int, b: int, c: int) -> int:
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        return b if pb <= pc else c

    expected = [
        paeth(a, b, c) for a, b, c in zip(left.tolist(), up.tolist(), upleft.tolist())
    ]
    np.testing.assert_array_equal(
        png_codec._paeth(left, up, upleft), np.array(expected)
    )


def test_writing_filters_in_strips_to_bound_memory(tmp_path) -> None:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
    image = array_to_image(pixels)
    tracemalloc.start()
    try:
        px.write_image(image, tmp_path / "big.png")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Filtered whole, the 16-bit work arrays take about 29 times the image.
    assert peak < 10 * pixels.nbytes


@pytest.mark.parametrize(
    ("strip_bytes", "shape"),
    [
        pytest.param(64, (600, 3, 3), id="rows-fit-a-strip"),
        pytest.param(4, (5, 3, 3), id="a-row-is-wider-than-a-strip"),
    ],
)
def test_filters_agree_across_a_strip_boundary(
    tmp_path, monkeypatch, strip_bytes, shape
) -> None:
    monkeypatch.setattr(png_codec, "_FILTER_STRIP_BYTES", strip_bytes)
    rng = np.random.default_rng(13)
    pixels = rng.integers(0, 256, size=shape, dtype=np.uint8)
    path = tmp_path / "tall.png"
    px.write_image(array_to_image(pixels), path)
    np.testing.assert_array_equal(helper.image_to_array(px.read_image(path)), pixels)
    rows = pixels.reshape(shape[0], -1)
    np.testing.assert_array_equal(
        png_codec._filter_rows(rows, 3), png_codec._filter_strip(rows, None, 3)
    )


def test_read_warns_and_points_at_read_image(tmp_path) -> None:
    path = tmp_path / "w.png"
    px.write_image(array_to_image(_gradient(3, 4, 3)), path)
    with pytest.warns(UserWarning, match=r"image format \(PNG\).*read_image") as caught:
        poly = px.read(path)
    assert {w.filename for w in caught} == {__file__}
    assert len(poly.vertices) == 12
    assert len(poly.vertex_attrs["colors"]) == 12


def test_read_image_emits_no_warning(tmp_path) -> None:
    path = tmp_path / "quiet.png"
    px.write_image(array_to_image(_gradient(3, 4)), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = px.read_image(path)
    assert image.dimensions == (4, 3, 1)


def test_lazy_read_is_refused(tmp_path) -> None:
    path = tmp_path / "lazy.png"
    px.write_image(array_to_image(_gradient(2, 2)), path)
    with pytest.raises(LazyReadError, match="lazy reads are not supported"):
        px.read(path, lazy=True)


def test_read_of_a_file_that_does_not_parse_gives_no_read_image_advice(
    tmp_path,
) -> None:
    path = _write(
        tmp_path, "cut.png", _png(_gradient(2, 2)[..., None], depth=8, color=0)[:-12]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(CodecError, match="IEND"):
            px.read(path)
    assert not [w for w in caught if "read_image" in str(w.message)]


def test_read_keeps_the_implied_vertex_cap_and_read_image_does_not(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cap.png"
    px.write_image(array_to_image(_gradient(4, 4)), path)
    # By path: ``polyxios.validate`` is the function the package re-exports,
    # which shadows the module.
    monkeypatch.setattr(sys.modules["polyxios.validate"], "MAX_IMPLIED_VERTICES", 10)
    assert px.read_image(path).dimensions == (4, 4, 1)
    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        px.read(path)


@pytest.mark.parametrize(
    "bad", [10, -1, 6.0, True, pytest.param(10**5000, id="5000-digits")]
)
def test_compress_level_out_of_range_is_refused(tmp_path, bad) -> None:
    image = array_to_image(_gradient(2, 2))
    with pytest.raises(CodecError, match="compress_level"):
        px.write_image(image, tmp_path / "c.png", compress_level=bad)


@pytest.mark.parametrize("level", [0, 9, np.int64(6)])
def test_compress_level_extremes_round_trip(tmp_path, level) -> None:
    path = tmp_path / f"c{level}.png"
    px.write_image(array_to_image(_gradient(2, 2)), path, compress_level=level)
    np.testing.assert_array_equal(
        helper.image_to_array(px.read_image(path)), _gradient(2, 2)
    )


def test_compress_level_changes_the_size_of_a_compressible_image(tmp_path) -> None:
    image = array_to_image(np.zeros((64, 64), np.uint8))
    sizes = {}
    for level in (0, 9):
        path = tmp_path / f"z{level}.png"
        px.write_image(image, path, compress_level=level)
        sizes[level] = path.stat().st_size
    assert sizes[0] > 4 * sizes[9]
    profile = {"icc_profile": bytes(10000)}
    icc = {}
    for level in (0, 9):
        path = tmp_path / f"icc{level}.png"
        image = array_to_image(np.zeros((2, 2), np.uint8), global_attrs=profile)
        px.write_image(image, path, compress_level=level)
        icc[level] = path.stat().st_size
    assert icc[0] > 10000 > 10 * icc[9]


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"dpi": (0.0, 72.0)}, "positive and finite"),
        ({"dpi": (72.0,)}, "pair of numbers"),
        ({"gamma": 0.0}, "positive and finite"),
        ({"text": {"bad\x00key": "x"}}, "1-79 characters"),
        ({"text": {"": "x"}}, "1-79 characters"),
        ({"text": {"k\u65e5": "x"}}, "latin-1 characters"),
        ({"gamma": "x"}, "gamma must be a number"),
        ({"gamma": True}, "gamma must be a number"),
        ({"gamma": 1e-6}, r"gamma 1e-06 rounds to 0"),
        ({"gamma": 1e304}, r"gamma 1e\+304 is too large"),
        ({"gamma": 30000}, r"gamma 30000\.0 is too large"),
        ({"dpi": (1e307, 72)}, r"dpi 1e\+307 is too large"),
        ({"dpi": (6e7, 72)}, r"dpi 60000000\.0 is too large"),
        ({"text": {b"k": "x"}}, r"keyword b'k' must be a str"),
        ({"gamma": -0.5}, r"gamma must be positive and finite, not -0\.5"),
        ({"dpi": (-1, 72)}, r"dpi must be positive and finite, not -1\."),
        ({"dpi": 300}, "pair of numbers"),
        ({"dpi": "72"}, "pair of numbers"),
        ({"dpi": b"HH"}, "pair of numbers"),
        ({"gamma": b"0.45"}, "gamma must be a number"),
        ({"gamma": "0.45"}, "gamma must be a number"),
        ({"gamma": 0.45 + 1j}, "gamma must be a number"),
        ({"gamma": np.True_}, "gamma must be a number"),
        ({"gamma": np.array([True])}, "gamma must be a number"),
        ({"gamma": np.complex128(0.45 + 1j)}, "gamma must be a number"),
        ({"dpi": (72, 72, 72)}, "pair of numbers"),
        ({"gamma": float("nan")}, "positive and finite"),
        ({"dpi": (float("inf"), 72)}, "positive and finite"),
        ({"exif": 3}, "exif must be bytes"),
        ({"icc_profile": "p"}, "icc_profile must be bytes"),
        ({"text": "abc"}, "text must be a mapping"),
        ({"text": {" T": "x"}}, "printable Latin-1"),
        ({"text": {"T ": "x"}}, "printable Latin-1"),
        ({"text": {"a\nb": "x"}}, "printable Latin-1"),
        ({"text": {"a\x7fb": "x"}}, "printable Latin-1"),
        ({"text": {"a\xa0b": "x"}}, "printable Latin-1"),
        ({"text": {"k": "a\0b"}}, "holds a NUL"),
        ({"text": {"k": 1}}, r"text 'k' must be a str"),
        ({"text": {"k": "a\udc80b"}}, "not valid Unicode"),
        ({"gamma": 10**400}, "gamma is too large"),
        ({"gamma": 10**5000}, "gamma is too large"),
        ({"dpi": (10**400, 72)}, "dpi is too large"),
        ({"dpi": (10**5000,)}, "pair of numbers"),
        ({"gamma": [10**5000]}, "gamma must be a number"),
        ({"text": {10**5000: "x"}}, "text keyword .* must be a str"),
    ],
)
def test_metadata_out_of_range_is_refused(tmp_path, attrs, match) -> None:
    image = replace(array_to_image(_gradient(2, 2)), global_attrs=attrs)
    with pytest.raises(CodecError, match=match):
        px.write_image(image, tmp_path / "bad.png")


def test_unwritten_attributes_warn(tmp_path) -> None:
    image = array_to_image(_gradient(3, 3))
    with pytest.warns(UserWarning, match="cell attributes.*left out"):
        px.write_image(
            replace(image, cell_attrs={"flag": np.zeros((1, 2, 2), np.uint8)}),
            tmp_path / "cells.png",
        )
    with pytest.warns(UserWarning, match=r"global_attrs \['custom'\]"):
        px.write_image(
            replace(image, global_attrs={"custom": 1}), tmp_path / "globals.png"
        )


# -----------------------------------------------------------------------------
# Hand-built files: interlacing, palette, transparency


def _expected_gray(samples: np.ndarray, depth: int) -> np.ndarray:
    values = samples[..., 0].astype(np.int64)
    if depth < 8:
        return (values * 255 // ((1 << depth) - 1)).astype(np.uint8)
    return values.astype(np.uint16 if depth == 16 else np.uint8)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 2), (4, 5), (9, 11)])
@pytest.mark.parametrize("interlaced", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_gray_depths_decode(tmp_path, depth, interlaced, shape, kernels) -> None:
    height, width = shape
    rng = np.random.default_rng(depth)
    samples = rng.integers(
        0, (1 << depth) - 1, size=(height, width, 1), endpoint=True, dtype=np.int64
    )
    path = _write(
        tmp_path,
        "gray.png",
        _png(samples, depth=depth, color=0, interlaced=interlaced),
    )
    image = px.read_image(path)
    got = helper.image_to_array(image).reshape(height, width)
    np.testing.assert_array_equal(got, _expected_gray(samples, depth))


@pytest.mark.parametrize("interlaced", [False, True])
def test_palette_with_trns_decodes_to_rgba(tmp_path, interlaced, kernels) -> None:
    palette = np.array(
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], np.uint8
    )
    alphas = bytes([0, 128, 255, 7, 9])  # as many as the palette has entries
    rng = np.random.default_rng(3)
    indices = rng.integers(0, 5, size=(9, 11, 1), dtype=np.int64)
    path = _write(
        tmp_path,
        "pal.png",
        _png(
            indices,
            depth=4,
            color=3,
            interlaced=interlaced,
            before_idat=[(b"PLTE", palette.tobytes()), (b"tRNS", alphas)],
        ),
    )
    table = np.array([0, 128, 255, 7, 9], np.uint8)
    expected = np.concatenate(
        [palette[indices[..., 0]], table[indices[..., 0]][..., None]], axis=-1
    )
    got = helper.image_to_array(px.read_image(path))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got.reshape(expected.shape), expected)


@pytest.mark.parametrize("interlaced", [False, True])
def test_rgb_trns_key_makes_matching_pixels_transparent(
    tmp_path, interlaced, kernels
) -> None:
    rng = np.random.default_rng(5)
    samples = rng.integers(0, 40, size=(9, 11, 3), dtype=np.int64)
    samples[2, 3] = (10, 20, 30)
    samples[7, 9] = (10, 20, 30)
    path = _write(
        tmp_path,
        "key.png",
        _png(
            samples,
            depth=8,
            color=2,
            interlaced=interlaced,
            before_idat=[(b"tRNS", struct.pack(">3H", 10, 20, 30))],
        ),
    )
    alpha = np.where((samples == (10, 20, 30)).all(axis=-1), 0, 255)
    expected = np.concatenate([samples, alpha[..., None]], axis=-1).astype(np.uint8)
    got = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(got.reshape(expected.shape), expected)
    assert int(got.reshape(9, 11, 4)[2, 3, 3]) == 0


@pytest.mark.parametrize(("channels", "color"), [(3, 2), (2, 4), (4, 6)])
@pytest.mark.parametrize("depth", [8, 16])
def test_interlaced_images_decode_at_every_channel_count(
    tmp_path, channels, color, depth, kernels
) -> None:
    rng = np.random.default_rng(channels + depth)
    samples = rng.integers(0, 1 << depth, size=(9, 11, channels), dtype=np.int64)
    path = _write(
        tmp_path,
        "alpha.png",
        _png(samples, depth=depth, color=color, interlaced=True),
    )
    got = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(got.reshape(samples.shape), samples)


@pytest.mark.parametrize("interlaced", [False, True])
def test_palette_without_trns_decodes_to_rgb(tmp_path, interlaced, kernels) -> None:
    palette = np.array(
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], np.uint8
    )
    rng = np.random.default_rng(3)
    indices = rng.integers(0, 5, size=(9, 11, 1), dtype=np.int64)
    path = _write(
        tmp_path,
        "pal-opaque.png",
        _png(
            indices,
            depth=4,
            color=3,
            interlaced=interlaced,
            before_idat=[(b"PLTE", palette.tobytes())],
        ),
    )
    got = helper.image_to_array(px.read_image(path))
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got.reshape(9, 11, 3), palette[indices[..., 0]])


def test_trailing_bytes_after_iend_are_ignored(tmp_path) -> None:
    samples = _gradient(2, 3)[..., None]
    data = _png(samples, depth=8, color=0) + b"trailing junk"
    path = _write(tmp_path, "tail.png", data)
    got = helper.image_to_array(px.read_image(path)).reshape(2, 3)
    np.testing.assert_array_equal(got, samples[..., 0])


# -----------------------------------------------------------------------------
# Structure errors


def _good_gray(tmp_path):
    return _write(
        tmp_path, "good.png", _png(_gradient(2, 2)[..., None], depth=8, color=0)
    )


def test_flipped_crc_is_refused(tmp_path) -> None:
    data = bytearray(_good_gray(tmp_path).read_bytes())
    data[16] ^= 0x01  # first byte of the IHDR width
    path = _write(tmp_path, "crc.png", bytes(data))
    with pytest.raises(CodecError, match="fails its CRC"):
        px.read_image(path)


def test_missing_iend_is_refused(tmp_path) -> None:
    data = _good_gray(tmp_path).read_bytes()
    path = _write(tmp_path, "noend.png", data[:-12])
    with pytest.raises(CodecError, match="IEND"):
        px.read_image(path)


def test_filter_byte_above_four_is_refused(tmp_path, kernels) -> None:
    raw = b"\x05\x01\x02" + b"\x00\x03\x04"
    path = _write(
        tmp_path,
        "filter.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.raises(CodecError, match="filter.png.*filter type 5"):
        px.read_image(path)
    # Pass 1 of a 9-tall, 11-wide image is two rows of 1 + 2 bytes, so byte 6 is
    # the first filter byte of pass 2.
    raw = bytearray(_scanlines(_gradient(9, 11)[..., None], 8, True))
    raw[6] = 5
    path = _write(
        tmp_path,
        "adam7.png",
        _file(
            [
                (b"IHDR", _header(11, 9, 8, 0, True)),
                (b"IDAT", zlib.compress(bytes(raw))),
            ]
        ),
    )
    with pytest.raises(CodecError, match="Adam7 pass 2 of 7"):
        px.read_image(path)
    # A 1-wide, 9-tall image has no pass 2, so the passes are numbered 1, 3, 5, 7
    # and a count of the non-empty passes would call this one pass 2.
    raw = bytearray(_scanlines(_gradient(9, 1)[..., None], 8, True))
    raw[4] = 5  # the first filter byte of pass 3
    path = _write(
        tmp_path,
        "narrow.png",
        _file(
            [
                (b"IHDR", _header(1, 9, 8, 0, True)),
                (b"IDAT", zlib.compress(bytes(raw))),
            ]
        ),
    )
    with pytest.raises(CodecError, match="Adam7 pass 3 of 7"):
        px.read_image(path)


def test_non_consecutive_idat_is_refused(tmp_path) -> None:
    data = zlib.compress(_scanlines(_gradient(2, 2)[..., None], 8, False))
    half = len(data) // 2
    path = _write(
        tmp_path,
        "split.png",
        _file(
            [
                (b"IHDR", _header(2, 2, 8, 0)),
                (b"IDAT", data[:half]),
                (b"tEXt", b"a\x00b"),
                (b"IDAT", data[half:]),
            ]
        ),
    )
    with pytest.raises(CodecError, match="not consecutive"):
        px.read_image(path)


def test_unknown_critical_chunk_is_refused(tmp_path) -> None:
    data = zlib.compress(_scanlines(_gradient(2, 2)[..., None], 8, False))
    path = _write(
        tmp_path,
        "critical.png",
        _file(
            [
                (b"IHDR", _header(2, 2, 8, 0)),
                (b"XYZW", b""),
                (b"IDAT", data),
            ]
        ),
    )
    with pytest.raises(CodecError, match="unknown critical chunk 'XYZW'"):
        px.read_image(path)


def test_unknown_ancillary_chunk_is_ignored(tmp_path) -> None:
    samples = _gradient(2, 2)[..., None]
    path = _write(
        tmp_path,
        "ancillary.png",
        _png(samples, depth=8, color=0, before_idat=[(b"acTL", b"\0" * 8)]),
    )
    got = helper.image_to_array(px.read_image(path)).reshape(2, 2)
    np.testing.assert_array_equal(got, samples[..., 0])


def test_header_declaring_enormous_image_is_refused_before_allocation(tmp_path) -> None:
    big = 2**31 - 1
    path = _write(
        tmp_path,
        "huge.png",
        _file(
            [
                (b"IHDR", _header(big, big, 8, 0)),
                (b"IDAT", zlib.compress(b"")),
            ]
        ),
    )
    with pytest.raises(ValidationError):
        px.read_image(path)


def test_jpeg_bytes_named_png_are_refused(tmp_path) -> None:
    path = _write(tmp_path, "x.png", b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    with pytest.raises(CodecError, match="holds image/jpeg data, which polyxios does"):
        px.read_image(path)


def test_first_chunk_must_be_ihdr(tmp_path) -> None:
    data = zlib.compress(b"\0\0")
    path = _write(tmp_path, "noihdr.png", _file([(b"IDAT", data)]))
    with pytest.raises(CodecError, match="does not open with"):
        px.read_image(path)


def test_short_image_data_is_refused(tmp_path) -> None:
    raw = b"\x00\x01\x02"  # one row of a 2x2 image: the second row is missing
    path = _write(
        tmp_path,
        "short.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.raises(CodecError, match="ends early"):
        px.read_image(path)


def test_excess_image_data_warns_and_is_ignored(tmp_path) -> None:
    samples = _gradient(2, 2)[..., None]
    raw = _scanlines(samples, 8, False) + b"\x00" * 5
    path = _write(
        tmp_path,
        "excess.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.warns(
        UserWarning, match="IDAT holds more data than the image"
    ) as caught:
        got = helper.image_to_array(px.read_image(path)).reshape(2, 2)
    assert {w.filename for w in caught} == {__file__}
    np.testing.assert_array_equal(got, samples[..., 0])


@pytest.mark.parametrize(
    "tail", [b"garbage after the stream", zlib.compress(b"a second stream")]
)
def test_data_after_the_zlib_stream_warns_and_is_ignored(tmp_path, tail) -> None:
    samples = _gradient(2, 2)[..., None]
    data = zlib.compress(_scanlines(samples, 8, False)) + tail
    path = _write(
        tmp_path,
        "after-stream.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", data)]),
    )
    with pytest.warns(UserWarning, match="IDAT holds more data than the image"):
        got = helper.image_to_array(px.read_image(path)).reshape(2, 2)
    np.testing.assert_array_equal(got, samples[..., 0])


def test_palette_index_past_the_palette_reads_as_opaque_black(tmp_path) -> None:
    palette = np.array([[10, 20, 30], [255, 255, 255]], np.uint8)
    indices = np.array([[[0], [1]], [[2], [3]]], dtype=np.int64)
    plain = _write(
        tmp_path,
        "badpal.png",
        _png(
            indices,
            depth=2,
            color=3,
            before_idat=[(b"PLTE", palette.tobytes())],
        ),
    )
    rgb = helper.image_to_array(px.read_image(plain))
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(
        rgb, [[[10, 20, 30], [255, 255, 255]], [[0, 0, 0], [0, 0, 0]]]
    )
    keyed = _write(
        tmp_path,
        "badpal-trns.png",
        _png(
            indices,
            depth=2,
            color=3,
            before_idat=[(b"PLTE", palette.tobytes()), (b"tRNS", b"\x80")],
        ),
    )
    rgba = helper.image_to_array(px.read_image(keyed))
    assert rgba.dtype == np.uint8
    np.testing.assert_array_equal(
        rgba,
        [
            [[10, 20, 30, 128], [255, 255, 255, 255]],
            [[0, 0, 0, 255], [0, 0, 0, 255]],
        ],
    )


@pytest.mark.parametrize("interlaced", [False, True])
def test_gray_trns_key_is_masked_to_the_depth(tmp_path, interlaced, kernels) -> None:
    samples = (np.arange(24).reshape(4, 6, 1) % 3 == 0).astype(np.int64)
    path = _write(
        tmp_path,
        "gray1.png",
        _png(
            samples,
            depth=1,
            color=0,
            interlaced=interlaced,
            before_idat=[(b"tRNS", struct.pack(">H", 0x0003))],
        ),
    )
    got = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(got[..., 0], samples[..., 0] * 255)
    np.testing.assert_array_equal(got[..., 1], np.where(samples[..., 0] == 1, 0, 255))


@pytest.mark.parametrize("interlaced", [False, True])
def test_rgb16_trns_key(tmp_path, interlaced, kernels) -> None:
    samples = np.array(
        [[[40000, 2, 3], [40000, 2, 4]], [[1, 2, 3], [40000, 2, 3]]], dtype=np.int64
    )
    path = _write(
        tmp_path,
        "rgb16.png",
        _png(
            samples,
            depth=16,
            color=2,
            interlaced=interlaced,
            before_idat=[(b"tRNS", struct.pack(">3H", 40000, 2, 3))],
        ),
    )
    got = helper.image_to_array(px.read_image(path))
    assert got.dtype == np.uint16
    np.testing.assert_array_equal(got[..., :3], samples)
    opaque = ~(samples == (40000, 2, 3)).all(axis=-1)
    np.testing.assert_array_equal(got[..., 3], np.where(opaque, 65535, 0))


def test_gray16_trns_key(tmp_path) -> None:
    samples = np.array([[[0], [40000], [65535]], [[40000], [1], [2]]], dtype=np.int64)
    path = _write(
        tmp_path,
        "gray16.png",
        _png(
            samples,
            depth=16,
            color=0,
            before_idat=[(b"tRNS", struct.pack(">H", 40000))],
        ),
    )
    got = helper.image_to_array(px.read_image(path))
    assert got.dtype == np.uint16
    np.testing.assert_array_equal(got[..., 0], samples[..., 0])
    np.testing.assert_array_equal(
        got[..., 1], np.where(samples[..., 0] == 40000, 0, 65535)
    )


@pytest.mark.parametrize(("channels", "color"), [(2, 4), (4, 6)])
def test_trns_is_ignored_when_the_color_type_has_alpha(
    tmp_path, channels, color
) -> None:
    samples = _gradient(3, 2, channels)
    path = _write(
        tmp_path,
        "alpha.png",
        _png(
            samples,
            depth=8,
            color=color,
            before_idat=[(b"tRNS", b"\0" * 6)],
        ),
    )
    np.testing.assert_array_equal(helper.image_to_array(px.read_image(path)), samples)


def test_trns_after_idat_is_ignored_with_a_warning(tmp_path) -> None:
    samples = _gradient(2, 3, 3)
    path = _write(
        tmp_path,
        "late.png",
        _file(
            [
                (b"IHDR", _header(3, 2, 8, 2)),
                (b"IDAT", zlib.compress(_scanlines(samples, 8, False))),
                (b"tRNS", struct.pack(">3H", 1, 2, 3)),
            ]
        ),
    )
    with pytest.warns(UserWarning, match="out of place"):
        got = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(got, samples)


def test_ztxt_and_compressed_itxt_are_read(tmp_path) -> None:
    path = _write(
        tmp_path,
        "text.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[
                (b"zTXt", b"A\0\0" + zlib.compress("Zo\u00eb".encode("latin-1"))),
                (
                    b"iTXt",
                    b"B\0\x01\0ja\0bee\0" + zlib.compress("\u65e5".encode()),
                ),
            ],
        ),
    )
    assert px.read_image(path).global_attrs["text"] == {
        "A": "Zo\u00eb",
        "B": "\u65e5",
    }


def test_a_malformed_ancillary_chunk_warns_and_the_pixels_read(tmp_path) -> None:
    samples = _gradient(2, 2)[..., None]
    path = _write(
        tmp_path,
        "badtext.png",
        _png(samples, depth=8, color=0, before_idat=[(b"tEXt", b"no terminator")]),
    )
    with pytest.warns(UserWarning, match="tEXt chunk is malformed"):
        image = px.read_image(path)
    np.testing.assert_array_equal(helper.image_to_array(image), samples[..., 0])
    assert "text" not in image.global_attrs


def test_a_repeated_text_keyword_keeps_the_first(tmp_path) -> None:
    path = _write(
        tmp_path,
        "twice.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(b"tEXt", b"K\0first"), (b"tEXt", b"K\0second")],
        ),
    )
    with pytest.warns(UserWarning, match="repeats text keyword 'K'"):
        image = px.read_image(path)
    assert image.global_attrs["text"] == {"K": "first"}


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        pytest.param(b"iTXt", b"k\0", id="itxt-no-flag"),
        pytest.param(
            b"iTXt", b"k\0\x02\0\0\0" + zlib.compress(b"v"), id="itxt-bad-flag"
        ),
        pytest.param(b"tEXt", b"\0v", id="empty-keyword"),
        pytest.param(b"tEXt", b" k\0v", id="leading-space-keyword"),
        pytest.param(b"tEXt", b"k \0v", id="trailing-space-keyword"),
        pytest.param(b"tEXt", b"a  b\0v", id="doubled-space-keyword"),
        pytest.param(b"tEXt", b"k" * 80 + b"\0v", id="long-keyword"),
        pytest.param(b"tEXt", b"a\nb\0v", id="unprintable-keyword"),
        pytest.param(b"tEXt", b"a\x7fb\0v", id="del-in-keyword"),
        pytest.param(b"tEXt", b"a\xa0b\0v", id="nbsp-in-keyword"),
        pytest.param(b"tEXt", b"k\0a\0b", id="nul-in-text"),
        pytest.param(b"iTXt", b"k\0\0\0\0\0a\0b", id="nul-in-itxt"),
    ],
)
def test_a_text_chunk_the_writer_would_refuse_is_malformed(
    tmp_path, kind, body
) -> None:
    path = _write(
        tmp_path,
        "oddtext.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(kind, body)],
        ),
    )
    with pytest.warns(UserWarning, match=f"{kind.decode()} chunk is malformed"):
        image = px.read_image(path)
    assert "text" not in image.global_attrs
    px.write_image(image, tmp_path / "back.png")  # writing it back out still works


@pytest.mark.parametrize(
    "phys",
    [struct.pack(">IIB", 0, 2835, 1), struct.pack(">IIB", 2835, 0, 1)],
)
def test_zero_gamma_and_zero_dpi_are_not_kept(tmp_path, phys) -> None:
    path = _write(
        tmp_path,
        "zero.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(b"gAMA", struct.pack(">I", 0)), (b"pHYs", phys)],
        ),
    )
    attrs = px.read_image(path).global_attrs
    assert "gamma" not in attrs
    assert "dpi" not in attrs


def test_a_resolution_in_meters_reads_as_dots_per_inch(tmp_path) -> None:
    path = _write(
        tmp_path,
        "dpi.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(b"pHYs", struct.pack(">IIB", 2835, 11811, 1))],
        ),
    )
    dpi = px.read_image(path).global_attrs["dpi"]
    assert dpi == pytest.approx((72.009, 299.9994))


@pytest.mark.parametrize(
    ("kind", "head"),
    [
        (b"iCCP", b"p\0\0"),
        (b"zTXt", b"A\0\0"),
        (b"iTXt", b"B\0\x01\0\0\0"),
    ],
)
def test_a_corrupt_compressed_chunk_warns_and_is_ignored(tmp_path, kind, head) -> None:
    path = _write(
        tmp_path,
        "corrupt.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(kind, head + b"x\x9c\xff\xff\xff\xff")],
        ),
    )
    with pytest.warns(UserWarning, match=f"{kind.decode()} chunk is malformed"):
        attrs = px.read_image(path).global_attrs
    assert not {"text", "icc_profile"} & set(attrs)


def test_a_resolution_that_is_no_length_is_not_a_dpi(tmp_path) -> None:
    path = _write(
        tmp_path,
        "aspect.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(b"pHYs", struct.pack(">IIB", 2835, 2835, 0))],
        ),
    )
    assert "dpi" not in px.read_image(path).global_attrs


@pytest.mark.parametrize(("width", "height"), [(2**31, 1), (0, 2), (2, 0)])
def test_a_width_or_height_outside_1_to_2_31_is_refused(
    tmp_path, width, height
) -> None:
    path = _write(
        tmp_path,
        "wide.png",
        _file([(b"IHDR", _header(width, height, 8, 0)), (b"IDAT", zlib.compress(b""))]),
    )
    with pytest.raises(CodecError, match=r"1 to 2\*\*31-1"):
        px.read_image(path)


def _raw_file(ihdr: bytes, *chunks: tuple[bytes, bytes]) -> bytes:
    """A 2x2 8-bit-sized file around ``ihdr``, with ``chunks`` before its IDAT."""
    return _file([(b"IHDR", ihdr), *chunks, (b"IDAT", zlib.compress(b"\0" * 6))])


def _flipped_text_crc() -> bytes:
    data = bytearray(
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(b"tEXt", b"k\0v")],
        )
    )
    data[data.index(b"tEXtk\0v") + 7] ^= 1  # the first byte of that chunk's CRC
    return bytes(data)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        pytest.param(
            lambda: _raw_file(_header(2, 2, 8, 0), (b"tRNS", b"\0")),
            "tRNS for gray",
            id="gray-trns-length",
        ),
        pytest.param(
            lambda: _raw_file(_header(2, 2, 8, 3), (b"PLTE", b"\0" * 4)),
            "1-256 RGB entries",
            id="plte-length",
        ),
        pytest.param(
            lambda: _raw_file(_header(2, 2, 8, 3), (b"PLTE", b"\0" * 771)),
            "1-256 RGB entries",
            id="plte-257-entries",
        ),
        pytest.param(
            lambda: _raw_file(_header(2, 2, 8, 3)),
            "no PLTE chunk before IDAT",
            id="no-plte",
        ),
        pytest.param(
            lambda: _raw_file(_header(2, 2, 4, 2)),
            "is not valid",
            id="color-depth",
        ),
        pytest.param(
            lambda: _raw_file(struct.pack(">IIBBBBB", 2, 2, 8, 0, 1, 0, 0)),
            "only 0 exists",
            id="compression-method",
        ),
        pytest.param(
            lambda: _raw_file(struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 64, 0)),
            "filter method 64",
            id="filter-method",
        ),
        pytest.param(
            lambda: _raw_file(struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 2)),
            "interlace method 2",
            id="interlace-method",
        ),
        pytest.param(
            lambda: _raw_file(_header(2, 2, 8, 0), (b"IHDR", _header(2, 2, 8, 0))),
            "second IHDR",
            id="second-ihdr",
        ),
        pytest.param(
            lambda: _raw_file(
                _header(2, 2, 8, 3), (b"PLTE", b"\0" * 6), (b"tRNS", b"\0\0\0")
            ),
            "3 alphas for a 2-entry palette",
            id="trns-longer-than-palette",
        ),
        pytest.param(_flipped_text_crc, "tEXt chunk fails its CRC", id="ancillary-crc"),
        pytest.param(
            lambda: _png(_gradient(2, 2)[..., None], depth=8, color=0)[:-18],
            "runs past the end",
            id="cut-idat",
        ),
        pytest.param(
            lambda: _file(
                [
                    (b"IHDR", _header(2, 2, 8, 0)[:12]),
                    (b"IDAT", zlib.compress(b"\0" * 6)),
                ]
            ),
            "13-byte IHDR",
            id="short-ihdr",
        ),
        pytest.param(
            lambda: _file(
                [
                    (b"IHDR", _header(2, 2, 8, 3)),
                    (b"IDAT", zlib.compress(b"\0" * 6)),
                    (b"PLTE", b"\0" * 6),
                ]
            ),
            "no PLTE chunk before IDAT",
            id="plte-after-idat",
        ),
        pytest.param(
            lambda: _file([(b"IHDR", _header(2, 2, 8, 0))]),
            "no IDAT chunk",
            id="no-idat",
        ),
        pytest.param(
            lambda: _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", b"junk")]),
            "not a valid zlib stream",
            id="junk-idat",
        ),
        pytest.param(
            lambda: _png(
                _gradient(2, 2, 3),
                depth=8,
                color=2,
                before_idat=[(b"tRNS", b"\0" * 4)],
            ),
            "tRNS for RGB",
            id="rgb-trns-length",
        ),
        pytest.param(
            lambda: _file(
                [
                    (b"IHDR", _header(2, 2, 8, 0) + b"\0"),
                    (b"IDAT", zlib.compress(b"\0" * 6)),
                ]
            ),
            "13-byte IHDR",
            id="long-ihdr",
        ),
        pytest.param(
            lambda: _raw_file(_header(2, 2, 8, 0), (b"tRNS", b"\0" * 3)),
            "tRNS for gray",
            id="gray-trns-too-long",
        ),
        pytest.param(
            lambda: _png(
                _gradient(2, 2, 3),
                depth=8,
                color=2,
                before_idat=[(b"tRNS", b"\0" * 8)],
            ),
            "tRNS for RGB",
            id="rgb-trns-too-long",
        ),
    ],
)
def test_structure_errors_are_refused(tmp_path, build, match) -> None:
    path = _write(tmp_path, "bad.png", build())
    with pytest.raises(CodecError, match=match):
        px.read_image(path)


def test_rgb_trns_key_is_masked_to_the_depth(tmp_path) -> None:
    samples = np.zeros((2, 3, 3), dtype=np.int64)
    samples[0, 1] = (10, 20, 30)
    path = _write(
        tmp_path,
        "mask.png",
        _png(
            samples,
            depth=8,
            color=2,
            before_idat=[(b"tRNS", struct.pack(">3H", 0x010A, 0x0114, 0x011E))],
        ),
    )
    alpha = helper.image_to_array(px.read_image(path))[..., 3]
    assert alpha[0, 1] == 0
    assert np.count_nonzero(alpha == 0) == 1


@pytest.mark.parametrize(("kind", "head"), [(b"iCCP", b"p\0\0"), (b"zTXt", b"A\0\0")])
def test_a_cut_short_compressed_chunk_warns_and_is_ignored(
    tmp_path, kind, head
) -> None:
    cut = zlib.compress(bytes(range(256)) * 2)[:12]
    path = _write(
        tmp_path,
        "cut.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(kind, head + cut)],
        ),
    )
    with pytest.warns(UserWarning, match=f"{kind.decode()} chunk is malformed"):
        attrs = px.read_image(path).global_attrs
    assert not {"text", "icc_profile"} & set(attrs)


@pytest.mark.parametrize(
    ("kind", "head"),
    [
        (b"iCCP", b"p\0\0"),
        (b"zTXt", b"B\0\0"),
        (b"iTXt", b"B\0\x01\0\0\0"),
    ],
)
def test_the_metadata_budget_is_shared_by_every_chunk(
    tmp_path, monkeypatch, kind, head
) -> None:
    monkeypatch.setattr(png_codec, "_METADATA_BUDGET", 10)
    path = _write(
        tmp_path,
        "two.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[
                (b"zTXt", b"A\0\0" + zlib.compress(b"x" * 6)),
                (kind, head + zlib.compress(b"y" * 6)),
            ],
        ),
    )
    with pytest.warns(UserWarning, match=f"{kind.decode()} chunk is malformed.*budget"):
        attrs = px.read_image(path).global_attrs
    assert attrs["text"] == {"A": "x" * 6}
    assert "icc_profile" not in attrs


def test_a_text_chunk_after_idat_is_read(tmp_path) -> None:
    samples = _gradient(2, 2)[..., None]
    path = _write(
        tmp_path,
        "after.png",
        _file(
            [
                (b"IHDR", _header(2, 2, 8, 0)),
                (b"IDAT", zlib.compress(_scanlines(samples, 8, False))),
                (b"tEXt", b"K\0v"),
            ]
        ),
    )
    assert px.read_image(path).global_attrs["text"] == {"K": "v"}


@pytest.mark.parametrize(
    ("kind", "head"),
    [
        (b"iCCP", b"p\0\0"),
        (b"zTXt", b"A\0\0"),
        (b"iTXt", b"B\0\x01\0\0\0"),
    ],
)
def test_a_metadata_bomb_is_ignored_with_a_warning(
    tmp_path, monkeypatch, kind, head
) -> None:
    monkeypatch.setattr(png_codec, "_METADATA_BUDGET", 10)
    path = _write(
        tmp_path,
        "bomb.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(kind, head + zlib.compress(b"x" * 100))],
        ),
    )
    with pytest.warns(UserWarning, match="budget"):
        attrs = px.read_image(path).global_attrs
    assert not {"text", "icc_profile"} & set(attrs)


def test_a_corrupt_stream_spends_what_is_left_of_the_budget(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(png_codec, "_METADATA_BUDGET", 1000)
    bad = bytearray(zlib.compress(b"x" * 6))
    bad[-1] ^= 1  # a wrong Adler-32: zlib fails only after inflating it all
    path = _write(
        tmp_path,
        "corrupt-then-good.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[
                (b"zTXt", b"A\0\0" + bytes(bad)),
                (b"zTXt", b"B\0\0" + zlib.compress(b"y")),
            ],
        ),
    )
    with pytest.warns(UserWarning) as caught:
        attrs = px.read_image(path).global_attrs
    messages = [str(w.message) for w in caught]
    assert "incorrect data check" in messages[0]
    assert len(messages) == 2 and "budget" in messages[1]
    assert "text" not in attrs


def test_a_refused_bomb_still_spends_the_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(png_codec, "_METADATA_BUDGET", 10)
    path = _write(
        tmp_path,
        "spent.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[
                (b"zTXt", b"A\0\0" + zlib.compress(b"x" * 100)),
                (b"zTXt", b"B\0\0" + zlib.compress(b"y")),
            ],
        ),
    )
    with pytest.warns(UserWarning, match="budget") as caught:
        attrs = px.read_image(path).global_attrs
    assert len(caught) == 2
    assert "text" not in attrs


def _peak_reading(path) -> float:
    tracemalloc.start()
    try:
        px.read_image(path)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_metadata_bomb_is_not_inflated_past_the_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(png_codec, "_METADATA_BUDGET", 10)
    bomb = b"A\0\0" + zlib.compress(bytes(8_000_000))
    path = _write(
        tmp_path,
        "bomb-big.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(b"zTXt", bomb)],
        ),
    )
    with pytest.warns(UserWarning, match="budget"):
        peak = _peak_reading(path)
    assert peak < 1e6  # unbounded, it holds 8 MB


def test_excess_image_data_is_skipped_not_inflated(tmp_path) -> None:
    raw = b"\0\x05" + bytes(1 << 24)  # a 1 x 1 picture, then 16 MiB of zeros
    path = _write(
        tmp_path,
        "excess-big.png",
        _file([(b"IHDR", _header(1, 1, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.warns(UserWarning, match="IDAT holds more data"):
        peak = _peak_reading(path)
    assert peak < 2e6  # unbounded, it holds 16 MB


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        (b"iCCP", b"p\0\x01" + zlib.compress(b"x")),
        (b"zTXt", b"A\0\x01" + zlib.compress(b"x")),
        (b"iTXt", b"B\0\x01\x01\0\0" + zlib.compress(b"x")),
    ],
)
def test_an_unknown_compression_method_warns_and_is_ignored(
    tmp_path, kind, body
) -> None:
    path = _write(
        tmp_path,
        "method.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(kind, body)],
        ),
    )
    with pytest.warns(UserWarning, match="unknown compression method"):
        attrs = px.read_image(path).global_attrs
    assert not {"text", "icc_profile"} & set(attrs)


@pytest.mark.parametrize(
    ("kind", "body", "key"),
    [
        (b"gAMA", b"\0\0\0", "gamma"),
        (b"pHYs", b"\0" * 8, "dpi"),
        (b"gAMA", struct.pack(">I", 2**31), "gamma"),
        (b"pHYs", struct.pack(">IIB", 2**31, 1, 1), "dpi"),
        (b"pHYs", struct.pack(">IIB", 1, 2**31, 1), "dpi"),
        (b"gAMA", b"\0" * 5, "gamma"),
        (b"pHYs", b"\0" * 10, "dpi"),
    ],
)
def test_a_gamma_or_resolution_chunk_that_does_not_fit_is_ignored(
    tmp_path, kind, body, key
) -> None:
    path = _write(
        tmp_path,
        "short.png",
        _png(
            _gradient(2, 2)[..., None],
            depth=8,
            color=0,
            before_idat=[(kind, body)],
        ),
    )
    with pytest.warns(UserWarning, match=f"{kind.decode()} chunk is malformed"):
        attrs = px.read_image(path).global_attrs
    assert key not in attrs


def test_bytearrays_and_memoryviews_are_written_as_bytes(tmp_path) -> None:
    attrs = {"exif": bytearray(b"II*\0"), "icc_profile": memoryview(b"profile")}
    path = tmp_path / "views.png"
    px.write_image(array_to_image(_gradient(2, 2), global_attrs=attrs), path)
    back = px.read_image(path).global_attrs
    assert back["exif"] == b"II*\0"
    assert back["icc_profile"] == b"profile"


def test_the_largest_gamma_a_png_can_hold_is_written(tmp_path) -> None:
    attrs = {"gamma": (2**31 - 1) / 100000}
    path = tmp_path / "gamma.png"
    px.write_image(array_to_image(_gradient(2, 2), global_attrs=attrs), path)
    assert px.read_image(path).global_attrs["gamma"] == pytest.approx(21474.83647)


def test_a_one_number_array_is_written_as_the_number(tmp_path) -> None:
    # A .vti or .vtkhdf hands gamma back as a one-element array.
    attrs = {
        "gamma": np.array([0.45455]),
        "dpi": (np.array([300.0]), np.array([300.0])),
    }
    path = tmp_path / "array.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        px.write_image(array_to_image(_gradient(2, 2), global_attrs=attrs), path)
    back = px.read_image(path).global_attrs
    assert back["gamma"] == pytest.approx(0.45455)
    assert back["dpi"] == pytest.approx((300.0, 300.0), rel=1e-4)


def test_the_jpeg_exif_prefix_is_dropped_on_write(tmp_path) -> None:
    image = array_to_image(_gradient(2, 2), global_attrs={"exif": b"Exif\0\0II*\0"})
    path = tmp_path / "exif.png"
    px.write_image(image, path)
    assert px.read_image(path).global_attrs["exif"] == b"II*\0"


def test_a_grid_past_the_point_cap_is_refused_before_it_is_written(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys.modules["polyxios.validate"], "MAX_SAFE_VERTICES", 10)
    path = tmp_path / "big.png"
    with pytest.raises(ValidationError, match="MAX_SAFE_VERTICES"):
        px.write_image(array_to_image(_gradient(4, 4)), path)
    assert not path.exists()


def test_a_one_bit_image_reads_without_widening_every_sample() -> None:
    side = 2000
    raw = (b"\0" + bytes(side // 8)) * side
    data = _file(
        [
            (b"IHDR", _header(side, side, 1, 0)),
            (b"IDAT", zlib.compress(raw)),
        ]
    )
    tracemalloc.start()
    try:
        px.read_image(io.BytesIO(data), fmt=".png")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The picture is 4 MB; widening each sample to 64 bits took 100 MB.
    assert peak < 40e6


# -----------------------------------------------------------------------------
# Pillow cross-checks


def _pillow_cases(Image):
    rng = np.random.default_rng(11)
    gray = _gradient(6, 5)
    bits = rng.integers(0, 2, size=(6, 5)) == 1
    palette_idx = rng.integers(0, 4, size=(6, 5), dtype=np.uint8)
    flat_palette = [10, 20, 30, 200, 100, 0, 0, 0, 255, 255, 255, 90, 90, 90, 1, 2, 3]
    cases = {}

    one = Image.fromarray(bits)
    cases["1"] = (one, np.asarray(one.convert("L")))

    cases["L"] = (Image.fromarray(gray), np.asarray(Image.fromarray(gray)))

    la = rng.integers(0, 256, size=(6, 5, 2), dtype=np.uint8)
    la_im = Image.frombytes("LA", (5, 6), la.tobytes())
    cases["LA"] = (la_im, la)

    pal = Image.frombytes("P", (5, 6), palette_idx.tobytes())
    pal.putpalette(flat_palette + [0] * (768 - len(flat_palette)))
    cases["P"] = (pal, np.asarray(pal.convert("RGB")))

    cases["P+transparency"] = (pal, None)

    rgb = rng.integers(0, 256, size=(6, 5, 3), dtype=np.uint8)
    cases["RGB"] = (Image.fromarray(rgb), rgb)

    rgba = rng.integers(0, 256, size=(6, 5, 4), dtype=np.uint8)
    cases["RGBA"] = (Image.fromarray(rgba), rgba)

    wide = rng.integers(0, 65536, size=(6, 5), dtype=np.uint16)
    wide_im = Image.frombytes("I;16", (5, 6), wide.astype("<u2").tobytes())
    cases["I;16"] = (wide_im, wide)
    return cases


@pytest.mark.parametrize(
    "name", ["1", "L", "LA", "P", "P+transparency", "RGB", "RGBA", "I;16"]
)
def test_reads_files_written_by_pillow(tmp_path, name, kernels) -> None:
    Image = _pil()
    im, expected = _pillow_cases(Image)[name]
    path = tmp_path / f"pil-{name.replace(';', '_').replace('+', '_')}.png"
    if name == "P+transparency":
        im.save(path, transparency=2)
        expected = np.asarray(Image.open(path).convert("RGBA"))
    else:
        im.save(path)
    got = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(
        got.reshape(expected.shape).astype(np.int64),
        np.asarray(expected).astype(np.int64),
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_pillow_reads_files_written_by_polyxios(tmp_path, channels) -> None:
    Image = _pil()
    pixels = _gradient(4, 6, channels)
    path = tmp_path / f"out{channels}.png"
    px.write_image(array_to_image(pixels), path)
    with Image.open(path) as im:
        mode = im.mode
        loaded = np.asarray(im)
    assert mode == {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[channels]
    np.testing.assert_array_equal(loaded.reshape(pixels.shape), pixels)
