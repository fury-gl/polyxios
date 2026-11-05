import struct
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import helper
from polyxios.codecs import _jpeg
from polyxios.exceptions import CodecError, LazyReadError, ValidationError


def _scene(height: int = 40, width: int = 56) -> np.ndarray:
    """A color scene with gradients, edges and one patch of noise."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:height, 0:width]
    red = (xx * 255) // (width - 1)
    green = (yy * 255) // (height - 1)
    blue = ((xx * yy) // 7 + 40) % 256
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    rgb[10:16, 20:34] = rng.integers(0, 256, (6, 14, 3), dtype=np.uint8)
    return rgb


def _gradient(height: int = 24, width: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    red = (xx * 255) // (width - 1)
    green = (yy * 255) // (height - 1)
    blue = ((xx + yy) * 3) % 256
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def _write(path, pixels, **kwargs) -> bytes:
    _jpeg.write_image(helper.array_to_image(pixels), path, **kwargs)
    return path.read_bytes()


def _assert_close(
    ours: np.ndarray, theirs: np.ndarray, *, max_diff: int, mean_diff: float
) -> None:
    assert ours.shape == theirs.shape
    diff = np.abs(ours.astype(np.int64) - theirs.astype(np.int64))
    assert diff.max() <= max_diff, f"max difference {diff.max()}"
    assert diff.mean() <= mean_diff, f"mean difference {diff.mean()}"


# =============================================================================
# Pillow agreement
# =============================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subsampling": 0},
        {"subsampling": 1},
        {"subsampling": 2},
        {"progressive": True},
        {"restart_marker_blocks": 4},
    ],
    ids=["444", "422", "420", "progressive", "restart"],
)
def test_decodes_pillow_color_files(tmp_path, kwargs):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "pillow.jpg"
    Image.fromarray(_scene()).save(path, "JPEG", quality=90, **kwargs)
    ours = helper.image_to_array(_jpeg.read_image(path))
    theirs = np.asarray(Image.open(path).convert("RGB"))
    _assert_close(ours, theirs, max_diff=3, mean_diff=0.5)


def test_decodes_pillow_grayscale_files(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "gray.jpg"
    gray = _scene()[..., 1]
    Image.fromarray(gray).save(path, "JPEG", quality=90)
    ours = helper.image_to_array(_jpeg.read_image(path))
    theirs = np.asarray(Image.open(path).convert("L"))
    _assert_close(ours, theirs, max_diff=3, mean_diff=0.5)


@pytest.mark.parametrize("subsampling", ["4:4:4", "4:2:2", "4:2:0"])
def test_pillow_decodes_our_color_files(tmp_path, subsampling):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "ours.jpg"
    _write(path, _scene(), quality=90, subsampling=subsampling)
    theirs = np.asarray(Image.open(path).convert("RGB"))
    ours = helper.image_to_array(_jpeg.read_image(path))
    _assert_close(ours, theirs, max_diff=3, mean_diff=0.5)


def test_pillow_decodes_our_lossless_gray(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "lossless.jpg"
    gray = _scene()[..., 2]
    _write(path, gray, lossless=True)
    theirs = np.asarray(Image.open(path))
    assert np.array_equal(theirs, gray)


def test_cmyk_file_is_refused(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (16, 16), (10, 20, 30, 40)).save(path, "JPEG")
    with pytest.raises(CodecError):
        _jpeg.read_image(path)


# =============================================================================
# Lossy round trip without Pillow
# =============================================================================


def test_lossy_round_trip_psnr(tmp_path):
    pixels = _gradient()
    path = tmp_path / "gradient.jpg"
    _write(path, pixels, quality=95, subsampling="4:4:4")
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.shape == pixels.shape
    mse = np.mean((back.astype(np.float64) - pixels) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse)
    assert psnr >= 35.0


def test_default_lossy_round_trip_keeps_shape_and_channels(tmp_path):
    pixels = _scene()
    path = tmp_path / "default.jpg"
    _write(path, pixels)
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.shape == pixels.shape
    assert back.dtype == np.uint8


def test_gray_lossy_round_trip(tmp_path):
    pixels = _gradient()[..., 0]
    path = tmp_path / "gray.jpg"
    _write(path, pixels, quality=95)
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.shape == pixels.shape
    mse = np.mean((back.astype(np.float64) - pixels) ** 2)
    assert 10 * np.log10(255.0**2 / mse) >= 35.0


# =============================================================================
# Refusals
# =============================================================================


def test_truncated_file_is_refused(tmp_path):
    path = tmp_path / "full.jpg"
    raw = _write(path, _scene(), quality=90)
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(CodecError):
        _jpeg.read_image(cut)


def test_arithmetic_coded_frame_is_refused(tmp_path):
    path = tmp_path / "base.jpg"
    raw = bytearray(_write(path, _scene(), quality=90))
    index = raw.index(b"\xff\xc0")
    raw[index + 1] = 0xC9  # SOF0 -> SOF9, arithmetic-coded sequential
    patched = tmp_path / "arith.jpg"
    patched.write_bytes(bytes(raw))
    with pytest.raises(CodecError, match="arithmetic-coded"):
        _jpeg.read_image(patched)


def test_twelve_bit_lossy_frame_is_refused(tmp_path):
    path = tmp_path / "base.jpg"
    raw = bytearray(_write(path, _scene(), quality=90))
    index = raw.index(b"\xff\xc0")
    raw[index + 4] = 12  # the precision byte follows the two-byte length
    patched = tmp_path / "twelve.jpg"
    patched.write_bytes(bytes(raw))
    with pytest.raises(CodecError, match="12-bit"):
        _jpeg.read_image(patched)


def test_rgba_image_is_refused_for_writing(tmp_path):
    rgba = np.zeros((4, 5, 4), np.uint8)
    with pytest.raises(CodecError, match="alpha"):
        _jpeg.write_image(helper.array_to_image(rgba), tmp_path / "a.jpg")


def test_rgba_image_is_refused_for_lossless_writing(tmp_path):
    rgba = np.zeros((4, 5, 4), np.uint8)
    with pytest.raises(CodecError, match="alpha"):
        _jpeg.write_image(
            helper.array_to_image(rgba), tmp_path / "a.jpg", lossless=True
        )


@pytest.mark.parametrize("quality", [0, 101, True, 2.5])
def test_bad_quality_is_refused(tmp_path, quality):
    with pytest.raises(CodecError, match="quality"):
        _jpeg.write_image(
            helper.array_to_image(_gradient()), tmp_path / "q.jpg", quality=quality
        )


def test_bad_subsampling_is_refused(tmp_path):
    with pytest.raises(CodecError, match="subsampling"):
        _jpeg.write_image(
            helper.array_to_image(_gradient()), tmp_path / "s.jpg", subsampling="4:1:1"
        )


def test_lossless_with_quality_is_refused(tmp_path):
    with pytest.raises(CodecError, match="lossy JPEG only"):
        _jpeg.write_image(
            helper.array_to_image(_gradient()),
            tmp_path / "l.jpg",
            lossless=True,
            quality=90,
        )


def test_lossless_with_subsampling_is_refused(tmp_path):
    with pytest.raises(CodecError, match="lossy JPEG only"):
        _jpeg.write_image(
            helper.array_to_image(_gradient()),
            tmp_path / "l.jpg",
            lossless=True,
            subsampling="4:4:4",
        )


def test_oversize_frame_is_refused_before_allocation(tmp_path):
    frame = struct.pack(">BHHB", 8, 65535, 65535, 1) + bytes([1, 0x11, 0])
    raw = (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(frame) + 2)
        + frame
        + b"\xff\xd9"
    )
    path = tmp_path / "huge.jpg"
    path.write_bytes(raw)
    with pytest.raises(ValidationError):
        _jpeg.read_image(path)


def test_lazy_read_is_refused(tmp_path):
    path = tmp_path / "x.jpg"
    _write(path, _gradient(), quality=90)
    with pytest.raises(LazyReadError):
        _jpeg.read(path, lazy=True)


# =============================================================================
# Lossless round trips
# =============================================================================


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16], ids=["uint8", "uint16"])
@pytest.mark.parametrize("color", [False, True], ids=["L", "RGB"])
def test_lossless_round_trip_is_exact(tmp_path, dtype, color):
    rng = np.random.default_rng(3)
    top = 255 if dtype is np.uint8 else 65535
    shape = (13, 17, 3) if color else (13, 17)
    pixels = rng.integers(0, top + 1, shape, dtype=dtype)
    path = tmp_path / "exact.jpg"
    raw = _write(path, pixels, lossless=True)
    assert b"\xff\xc3" in raw
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.dtype == dtype
    assert np.array_equal(back, pixels)


def test_lossless_extreme_values_round_trip(tmp_path):
    pixels = np.array([[0, 65535, 0, 32768], [1, 65534, 32767, 2]], np.uint16)
    path = tmp_path / "extreme.jpg"
    _write(path, pixels, lossless=True)
    assert np.array_equal(helper.image_to_array(_jpeg.read_image(path)), pixels)


def test_lossy_round_trip_through_polydata(tmp_path):
    pixels = _gradient()[..., 0]
    poly = helper.array_to_image(pixels).to_polydata()
    path = tmp_path / "poly.jpg"
    _jpeg.write(poly, path, lossless=True)
    assert np.array_equal(helper.image_to_array(_jpeg.read_image(path)), pixels)


# =============================================================================
# Metadata
# =============================================================================


def test_metadata_round_trips(tmp_path):
    pixels = _gradient()
    icc = bytes(range(256)) * 280  # 71680 bytes, two APP2 chunks
    attrs = {
        "dpi": (300.0, 150.0),
        "comment": "a caption, latin-1: \xe9",
        "exif": b"MM\x00*\x00\x00\x00\x08",
        "icc_profile": icc,
    }
    path = tmp_path / "meta.jpg"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _jpeg.write_image(
            helper.array_to_image(pixels, global_attrs=attrs), path, quality=90
        )
    image = _jpeg.read_image(path)
    assert image.global_attrs["dpi"] == pytest.approx((300.0, 150.0))
    assert image.global_attrs["comment"] == attrs["comment"]
    assert image.global_attrs["exif"] == attrs["exif"]
    assert image.global_attrs["icc_profile"] == icc


def test_absent_metadata_stays_absent(tmp_path):
    path = tmp_path / "bare.jpg"
    _write(path, _gradient(), quality=90)
    assert _jpeg.read_image(path).global_attrs == {}


def test_lossless_rgb_reports_dpi_as_left_out(tmp_path):
    path = tmp_path / "rgb.jpg"
    image = helper.array_to_image(_gradient(), global_attrs={"dpi": (72.0, 72.0)})
    with pytest.warns(UserWarning, match="dpi"):
        _jpeg.write_image(image, path, lossless=True)


def test_lossless_gray_keeps_dpi(tmp_path):
    path = tmp_path / "gray.jpg"
    image = helper.array_to_image(
        _gradient()[..., 0], global_attrs={"dpi": (72.0, 72.0)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _jpeg.write_image(image, path, lossless=True)
    assert _jpeg.read_image(path).global_attrs["dpi"] == pytest.approx((72.0, 72.0))


def test_unwritten_global_attrs_warn(tmp_path):
    path = tmp_path / "x.jpg"
    image = helper.array_to_image(_gradient(), global_attrs={"surveyor": "Ada"})
    with pytest.warns(UserWarning, match="surveyor"):
        _jpeg.write_image(image, path, quality=90)


# =============================================================================
# Public API
# =============================================================================


def test_public_read_image_is_silent_and_read_points_at_it(tmp_path):
    path = tmp_path / "api.jpg"
    polyxios.write_image(helper.array_to_image(_gradient()), path, quality=90)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = polyxios.read_image(path)
    assert image.dimensions == (32, 24, 1)
    with pytest.warns(UserWarning, match="read_image"):
        mesh = polyxios.read(path)
    assert mesh.vertices.shape[0] == 32 * 24


@pytest.mark.parametrize("suffix", [".jpeg", ".jpe", ".jfif"])
def test_alternate_extensions_read(tmp_path, suffix):
    path = tmp_path / f"alt{suffix}"
    polyxios.write_image(helper.array_to_image(_gradient()), path, quality=90)
    assert polyxios.read_image(path).dimensions == (32, 24, 1)


def test_read_image_on_mesh_file_is_refused(tmp_path):
    path = tmp_path / "m.stl"
    path.write_text("solid x\nendsolid x\n")
    with pytest.raises(
        polyxios.exceptions.UnsupportedFormatError, match="polyxios.read()"
    ):
        polyxios.read_image(path)


# =============================================================================
# Test-local SOF3 builder, written apart from the codec
# =============================================================================

# Every category 0-16 gets a five-bit code equal to its value: 17 codes of
# length 5, the canonical assignment, with no all-ones code.
_DHT_FIVE_BIT = bytes([0x00]) + bytes([0, 0, 0, 0, 17] + [0] * 11) + bytes(range(17))


class _BitWriter:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, value: int, count: int) -> None:
        for i in range(count - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc = 0
                self.n = 0

    def restart(self, index: int) -> None:
        while self.n:
            self.put(1, 1)
        self.out += bytes([0xFF, 0xD0 + (index & 7)])


def _predict(predictor: int, ra: int, rb: int, rc: int) -> int:
    if predictor == 1:
        return ra
    if predictor == 2:
        return rb
    if predictor == 3:
        return rc
    if predictor == 4:
        return ra + rb - rc
    if predictor == 5:
        return ra + ((rb - rc) >> 1)
    if predictor == 6:
        return rb + ((ra - rc) >> 1)
    return (ra + rb) >> 1


def _sof3_entropy(
    planes: list[np.ndarray], predictor: int, precision: int, restart: int
) -> bytes:
    height, width = planes[0].shape
    writer = _BitWriter()
    since_restart = 0
    marker_index = 0
    for r in range(height):
        for c in range(width):
            if restart and since_restart == restart:
                writer.restart(marker_index)
                marker_index += 1
                since_restart = 0
            for plane in planes:
                x = int(plane[r, c])
                if since_restart == 0:
                    pred = 1 << (precision - 1)
                elif r == 0:
                    pred = int(plane[r, c - 1])
                elif c == 0:
                    pred = int(plane[r - 1, c])
                else:
                    pred = _predict(
                        predictor,
                        int(plane[r, c - 1]),
                        int(plane[r - 1, c]),
                        int(plane[r - 1, c - 1]),
                    )
                diff = (x - pred) % 65536
                if diff > 32768:
                    diff -= 65536
                size = abs(diff).bit_length()
                writer.put(size, 5)
                if 0 < size < 16:
                    writer.put(diff if diff > 0 else diff + (1 << size) - 1, size)
            since_restart += 1
    while writer.n:
        writer.put(1, 1)
    return bytes(writer.out)


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def _sof3_file(
    planes: list[np.ndarray],
    *,
    precision: int,
    predictor: int,
    restart: int = 0,
    interleaved: bool = True,
) -> bytes:
    height, width = planes[0].shape
    ids = [1] if len(planes) == 1 else [82, 71, 66]
    out = bytearray(b"\xff\xd8")
    out += _segment(0xC4, _DHT_FIVE_BIT)
    frame = struct.pack(">BHHB", precision, height, width, len(planes))
    frame += b"".join(bytes([cid, 0x11, 0]) for cid in ids)
    out += _segment(0xC3, frame)
    if restart:
        out += _segment(0xDD, struct.pack(">H", restart))
    groups = (
        [list(range(len(planes)))] if interleaved else [[k] for k in range(len(planes))]
    )
    for group in groups:
        sos = bytes([len(group)]) + b"".join(bytes([ids[k], 0]) for k in group)
        sos += bytes([predictor, 0, 0])
        out += _segment(0xDA, sos)
        out += _sof3_entropy([planes[k] for k in group], predictor, precision, restart)
    out += b"\xff\xd9"
    return bytes(out)


@pytest.mark.parametrize("predictor", range(1, 8))
def test_hand_built_lossless_predictors(tmp_path, predictor):
    rng = np.random.default_rng(predictor)
    plane = rng.integers(0, 256, (9, 13), dtype=np.uint8).astype(np.int64)
    path = tmp_path / f"p{predictor}.jpg"
    path.write_bytes(_sof3_file([plane], precision=8, predictor=predictor))
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.dtype == np.uint8
    assert np.array_equal(back, plane.astype(np.uint8))


def test_hand_built_twelve_bit_gray(tmp_path):
    rng = np.random.default_rng(12)
    plane = rng.integers(0, 4096, (7, 11), dtype=np.int64)
    path = tmp_path / "twelve.jpg"
    path.write_bytes(_sof3_file([plane], precision=12, predictor=4))
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.dtype == np.uint16
    assert np.array_equal(back, plane.astype(np.uint16))


def test_hand_built_restart_interval_of_one_line(tmp_path):
    rng = np.random.default_rng(21)
    plane = rng.integers(0, 256, (5, 9), dtype=np.int64)
    path = tmp_path / "restart.jpg"
    path.write_bytes(_sof3_file([plane], precision=8, predictor=6, restart=9))
    back = helper.image_to_array(_jpeg.read_image(path))
    assert np.array_equal(back, plane.astype(np.uint8))


def test_hand_built_rgb_over_three_scans(tmp_path):
    rng = np.random.default_rng(31)
    planes = [rng.integers(0, 256, (6, 8), dtype=np.int64) for _ in range(3)]
    path = tmp_path / "split.jpg"
    path.write_bytes(_sof3_file(planes, precision=8, predictor=5, interleaved=False))
    back = helper.image_to_array(_jpeg.read_image(path))
    assert back.shape == (6, 8, 3)
    expected = np.stack([p.astype(np.uint8) for p in planes], axis=-1)
    assert np.array_equal(back, expected)
