"""PNG codec: round trips, Adam7 and palette decoding, structure errors, Pillow."""

from dataclasses import replace
import struct
import warnings
import zlib

import numpy as np
import pytest

import polyxios as px
from polyxios import helper
from polyxios._image import array_to_image
from polyxios._io import write_bytes
from polyxios.exceptions import CodecError, LazyReadError, ValidationError

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
        for row in samples[y0::dy, x0::dx]:
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


# -----------------------------------------------------------------------------
# Round trips


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_round_trip_keeps_pixels_and_metadata(tmp_path, mode, dtype):
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
        "exif": b"Exif\x00\x00II*\x00",
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


def test_write_through_public_write_keeps_the_lattice(tmp_path):
    pixels = _gradient(4, 6, 3)
    poly = array_to_image(pixels).to_polydata()
    path = tmp_path / "lattice.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        px.write(poly, path)
        back = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(back, pixels)


def test_read_warns_and_points_at_read_image(tmp_path):
    path = tmp_path / "w.png"
    px.write_image(array_to_image(_gradient(3, 4, 3)), path)
    with pytest.warns(UserWarning, match=r"image format \(PNG\).*read_image"):
        poly = px.read(path)
    assert len(poly.vertices) == 12
    assert len(poly.vertex_attrs["colors"]) == 12


def test_read_image_emits_no_warning(tmp_path):
    path = tmp_path / "quiet.png"
    px.write_image(array_to_image(_gradient(3, 4)), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = px.read_image(path)
    assert image.dimensions == (4, 3, 1)


def test_lazy_read_is_refused(tmp_path):
    path = tmp_path / "lazy.png"
    px.write_image(array_to_image(_gradient(2, 2)), path)
    with pytest.raises(LazyReadError, match="lazy reads are not supported"):
        px.read(path, lazy=True)


def test_compress_level_is_checked(tmp_path):
    image = array_to_image(_gradient(2, 2))
    for bad in (10, -1, 6.0, True):
        with pytest.raises(CodecError, match="compress_level"):
            px.write_image(image, tmp_path / "c.png", compress_level=bad)
    for level in (0, 9):
        px.write_image(image, tmp_path / f"c{level}.png", compress_level=level)
        np.testing.assert_array_equal(
            helper.image_to_array(px.read_image(tmp_path / f"c{level}.png")),
            _gradient(2, 2),
        )


def test_metadata_out_of_range_is_refused(tmp_path):
    base = array_to_image(_gradient(2, 2))
    cases = [
        {"dpi": (0.0, 72.0)},
        {"dpi": (72.0,)},
        {"gamma": 0.0},
        {"text": {"bad\x00key": "x"}},
        {"text": {"": "x"}},
        {"text": {"k\u65e5": "x"}},
    ]
    for attrs in cases:
        image = replace(base, global_attrs=attrs)
        with pytest.raises(CodecError):
            px.write_image(image, tmp_path / "bad.png")


def test_unwritten_attributes_warn(tmp_path):
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


@pytest.mark.parametrize("interlaced", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_gray_depths_decode(tmp_path, depth, interlaced):
    rng = np.random.default_rng(depth)
    samples = rng.integers(
        0, (1 << depth) - 1, size=(9, 11, 1), endpoint=True, dtype=np.int64
    )
    path = _write(
        tmp_path,
        "gray.png",
        _png(samples, depth=depth, color=0, interlaced=interlaced),
    )
    image = px.read_image(path)
    got = helper.image_to_array(image).reshape(9, 11)
    np.testing.assert_array_equal(got, _expected_gray(samples, depth))


@pytest.mark.parametrize("interlaced", [False, True])
def test_palette_with_trns_decodes_to_rgba(tmp_path, interlaced):
    palette = np.array(
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], np.uint8
    )
    alphas = bytes([0, 128, 255])
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
    table = np.array([0, 128, 255, 255, 255], np.uint8)
    expected = np.concatenate(
        [palette[indices[..., 0]], table[indices[..., 0]][..., None]], axis=-1
    )
    got = helper.image_to_array(px.read_image(path))
    np.testing.assert_array_equal(got.reshape(expected.shape), expected)


@pytest.mark.parametrize("interlaced", [False, True])
def test_rgb_trns_key_makes_matching_pixels_transparent(tmp_path, interlaced):
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


def test_trailing_bytes_after_iend_are_ignored(tmp_path):
    samples = _gradient(2, 3)[..., None]
    data = _png(samples, depth=8, color=0) + b"trailing junk"
    path = _write(tmp_path, "tail.png", data)
    got = helper.image_to_array(px.read_image(path)).reshape(2, 3)
    np.testing.assert_array_equal(got, samples[..., 0])


# -----------------------------------------------------------------------------
# Structure errors


def _good_gray(tmp_path, name="good.png"):
    return _write(tmp_path, name, _png(_gradient(2, 2)[..., None], depth=8, color=0))


def test_flipped_crc_is_refused(tmp_path):
    data = bytearray(_good_gray(tmp_path).read_bytes())
    data[16] ^= 0x01  # first byte of the IHDR width
    path = _write(tmp_path, "crc.png", bytes(data))
    with pytest.raises(CodecError, match="fails its CRC"):
        px.read_image(path)


def test_missing_iend_is_refused(tmp_path):
    data = _good_gray(tmp_path).read_bytes()
    path = _write(tmp_path, "noend.png", data[:-12])
    with pytest.raises(CodecError, match="IEND"):
        px.read_image(path)


def test_filter_byte_above_four_is_refused(tmp_path):
    raw = b"\x05\x01\x02" + b"\x00\x03\x04"
    path = _write(
        tmp_path,
        "filter.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.raises(CodecError, match="filter type 5"):
        px.read_image(path)


def test_non_consecutive_idat_is_refused(tmp_path):
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


def test_unknown_critical_chunk_is_refused(tmp_path):
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


def test_unknown_ancillary_chunk_is_ignored(tmp_path):
    samples = _gradient(2, 2)[..., None]
    path = _write(
        tmp_path,
        "ancillary.png",
        _png(samples, depth=8, color=0, before_idat=[(b"acTL", b"\0" * 8)]),
    )
    got = helper.image_to_array(px.read_image(path)).reshape(2, 2)
    np.testing.assert_array_equal(got, samples[..., 0])


def test_header_declaring_enormous_image_is_refused_before_allocation(tmp_path):
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


def test_jpeg_bytes_named_png_are_refused(tmp_path):
    path = _write(tmp_path, "x.png", b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    with pytest.raises(CodecError, match="holds image/jpeg data"):
        px.read_image(path)


def test_first_chunk_must_be_ihdr(tmp_path):
    data = zlib.compress(b"\0\0")
    path = _write(tmp_path, "noihdr.png", _file([(b"IDAT", data)]))
    with pytest.raises(CodecError, match="does not open with"):
        px.read_image(path)


def test_short_image_data_is_refused(tmp_path):
    raw = b"\x00\x01\x02"  # one row of a 2x2 image: the second row is missing
    path = _write(
        tmp_path,
        "short.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.raises(CodecError, match="ends early"):
        px.read_image(path)


def test_excess_image_data_warns_and_is_ignored(tmp_path):
    samples = _gradient(2, 2)[..., None]
    raw = _scanlines(samples, 8, False) + b"\x00" * 5
    path = _write(
        tmp_path,
        "excess.png",
        _file([(b"IHDR", _header(2, 2, 8, 0)), (b"IDAT", zlib.compress(raw))]),
    )
    with pytest.warns(UserWarning, match="IDAT holds more data than the image"):
        got = helper.image_to_array(px.read_image(path)).reshape(2, 2)
    np.testing.assert_array_equal(got, samples[..., 0])


def test_palette_index_past_the_palette_is_refused(tmp_path):
    palette = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    indices = np.full((2, 2, 1), 3, dtype=np.int64)
    path = _write(
        tmp_path,
        "badpal.png",
        _png(
            indices,
            depth=2,
            color=3,
            before_idat=[(b"PLTE", palette.tobytes())],
        ),
    )
    with pytest.raises(CodecError, match="past the 2-entry palette"):
        px.read_image(path)


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

    pal_t = pal.copy()
    cases["P+transparency"] = (pal_t, np.asarray(pal.convert("RGBA")))

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
def test_reads_files_written_by_pillow(tmp_path, name):
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
def test_pillow_reads_files_written_by_polyxios(tmp_path, channels):
    Image = _pil()
    pixels = _gradient(4, 6, channels)
    path = tmp_path / f"out{channels}.png"
    px.write_image(array_to_image(pixels), path)
    with Image.open(path) as im:
        mode = im.mode
        loaded = np.asarray(im)
    assert mode == {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[channels]
    np.testing.assert_array_equal(loaded.reshape(pixels.shape), pixels)
