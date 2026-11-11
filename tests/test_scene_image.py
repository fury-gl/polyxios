"""``SceneImage.decode``: texture bytes, data URIs and local files as ImageData."""

import base64
import io
import os

import numpy as np
import pytest

import polyxios
from polyxios import SceneImage, helper
from polyxios.exceptions import CodecError, UnsupportedFormatError


def _gradient(h: int = 4, w: int = 5, c: int = 3) -> np.ndarray:
    shape = (h, w, c)
    return (np.arange(int(np.prod(shape))) * 7 % 256).astype(np.uint8).reshape(shape)


def _png_bytes(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    polyxios.write_image(helper.array_to_image(pixels), buf, fmt=".png")
    return buf.getvalue()


def test_decode_reads_embedded_bytes() -> None:
    pixels = _gradient(3, 4)
    image = SceneImage(data=_png_bytes(pixels), media_type="image/png").decode()
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)
    # The bytes say what they are, whatever the label claims.
    mislabeled = SceneImage(data=_png_bytes(pixels), media_type="image/jpeg")
    np.testing.assert_array_equal(helper.image_to_array(mislabeled.decode()), pixels)


def test_decode_reads_a_data_uri() -> None:
    pixels = _gradient(3, 4)
    uri = "data:image/png;base64," + base64.b64encode(_png_bytes(pixels)).decode()
    image = SceneImage(uri=uri, name="t").decode()
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)
    with pytest.raises(CodecError, match="data URI without a comma"):
        SceneImage(uri="data:image/png;base64", name="t").decode()
    with pytest.raises(CodecError, match="data URI that does not decode"):
        SceneImage(uri="data:image/png;base64,@@@", name="t").decode()


def test_decode_reads_a_local_file_under_base(tmp_path) -> None:
    pixels = _gradient(3, 4)
    (tmp_path / "t x.png").write_bytes(_png_bytes(pixels))
    image = SceneImage(uri="t%20x.png").decode(base=tmp_path)
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)
    file_uri = SceneImage(uri="file:t%20x.png").decode(base=tmp_path)
    assert file_uri == image


def test_decode_keeps_to_its_directory(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "e.png").write_bytes(_png_bytes(_gradient()))
    with pytest.raises(CodecError, match="escapes the base directory"):
        SceneImage(uri="../e.png").decode(base=tmp_path / "sub")
    with pytest.raises(CodecError, match="escapes the base directory"):
        SceneImage(uri=str(tmp_path / "e.png")).decode(base=tmp_path / "sub")


@pytest.mark.parametrize(
    "uri",
    [
        "//[bad/a.png",
        "http://[::1/a.png",
        "file://[x/a.png",
        "a%00.png",
        "file:///a%00.png",
    ],
)
def test_decode_of_a_malformed_uri_is_a_codec_error(uri) -> None:
    with pytest.raises(CodecError, match="is malformed"):
        SceneImage(uri=uri).decode()


@pytest.mark.skipif(os.name != "nt", reason="a stray colon is a valid POSIX path")
@pytest.mark.parametrize(
    "uri",
    [
        "file:///a:b:c.png",
        "file:///1:/x.png",
        "file:///textures/c:b.png",
        "file:///textures/c|b.png",
        "file:///dir/a%3Ab.png",
        "dir/a%3Ab.png",
        "1:/x.png",
    ],
)
def test_decode_of_a_uri_with_a_stray_colon_is_a_codec_error(uri) -> None:
    with pytest.raises(CodecError, match="is malformed"):
        SceneImage(uri=uri).decode()


def test_decode_refuses_what_it_cannot_read() -> None:
    with pytest.raises(UnsupportedFormatError, match="local files only"):
        SceneImage(uri="http://example.com/a.png").decode()
    with pytest.raises(CodecError, match="neither data nor a uri"):
        SceneImage().decode()
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    with pytest.raises(UnsupportedFormatError, match="image/webp"):
        SceneImage(data=webp, name="w").decode()
    with pytest.raises(UnsupportedFormatError, match="unknown"):
        SceneImage(data=b"????", name="w").decode()


def test_an_image_without_a_name_is_not_called_by_an_empty_one() -> None:
    with pytest.raises(UnsupportedFormatError, match=r"^SceneImage: unknown data"):
        SceneImage(data=b"????").decode()
    with pytest.raises(CodecError, match=r"^SceneImage: data URI without a comma"):
        SceneImage(uri="data:image/png;base64").decode()


def test_decode_reads_an_absolute_file_uri(tmp_path) -> None:
    (tmp_path / "t x.png").write_bytes(_png_bytes(_gradient(3, 4)))
    relative = SceneImage(uri="t%20x.png").decode(base=tmp_path)
    absolute = SceneImage(uri=(tmp_path / "t x.png").as_uri()).decode()
    assert absolute == relative


def test_decode_refuses_a_remote_file_uri() -> None:
    with pytest.raises(UnsupportedFormatError, match="names host 'server'"):
        SceneImage(uri="file://server/share/a.png").decode()


@pytest.mark.skipif(os.name != "nt", reason="a UNC path is local on POSIX")
@pytest.mark.parametrize("uri", ["//server/share/a.png", "file:////server/share/a.png"])
def test_decode_refuses_a_network_share(uri) -> None:
    with pytest.raises(UnsupportedFormatError, match="network share"):
        SceneImage(uri=uri).decode()


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST"])
def test_decode_reads_a_localhost_file_uri(tmp_path, host) -> None:
    pixels = _gradient(3, 4)
    (tmp_path / "t.png").write_bytes(_png_bytes(pixels))
    uri = (tmp_path / "t.png").as_uri().replace("file://", f"file://{host}", 1)
    image = SceneImage(uri=uri).decode()
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)


def test_decode_sniffs_a_local_file(tmp_path) -> None:
    pixels = _gradient(3, 4)
    (tmp_path / "t.jpg").write_bytes(_png_bytes(pixels))
    image = SceneImage(uri="t.jpg").decode(base=tmp_path)
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)


def test_decode_of_a_missing_file_is_an_oserror(tmp_path) -> None:
    with pytest.raises(OSError):
        SceneImage(uri="missing.png").decode(base=tmp_path)


def test_decode_of_an_empty_uri_is_a_codec_error(tmp_path) -> None:
    with pytest.raises(CodecError, match="neither data nor a uri"):
        SceneImage(uri="").decode(base=tmp_path)


def test_decode_names_the_file_or_the_image_a_bad_texture_comes_from(tmp_path) -> None:
    broken = b"\x89PNG\r\n\x1a\n" + b"junk"
    (tmp_path / "bad.png").write_bytes(broken)
    with pytest.raises(CodecError, match="bad.png"):
        SceneImage(uri="bad.png").decode(base=tmp_path)
    with pytest.raises(CodecError, match="albedo"):
        SceneImage(data=broken, name="albedo").decode()


def test_decode_names_the_file_of_an_unsupported_texture(tmp_path) -> None:
    (tmp_path / "t.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes(16))
    with pytest.raises(UnsupportedFormatError, match="t.jpg.*image/jpeg"):
        SceneImage(uri="t.jpg").decode(base=tmp_path)


def test_decode_falls_back_to_the_media_type_and_the_extension(tmp_path) -> None:
    (tmp_path / "t.png").write_bytes(b"garbage")
    with pytest.raises(CodecError, match="not a PNG file"):
        SceneImage(uri="t.png").decode(base=tmp_path)
    with pytest.raises(CodecError, match="'g' is not a PNG file"):
        SceneImage(data=b"garbage", media_type="image/png", name="g").decode()
    (tmp_path / "t.jpg").write_bytes(b"garbage")
    with pytest.raises(CodecError, match="not a PNG file"):
        SceneImage(uri="t.jpg", media_type="image/png").decode(base=tmp_path)
    uri = "data:image/png;base64," + base64.b64encode(b"garbage").decode()
    with pytest.raises(CodecError, match="'g' is not a PNG file"):
        SceneImage(uri=uri, media_type="image/jpeg", name="g").decode()
    with pytest.raises(CodecError, match="'g' is not a PNG file"):
        SceneImage(data=b"garbage", media_type="Image/PNG", name="g").decode()


def test_decode_reads_a_relative_uri_from_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    pixels = _gradient(3, 4)
    (tmp_path / "t x.png").write_bytes(_png_bytes(pixels))
    monkeypatch.chdir(tmp_path)
    image = SceneImage(uri="t%20x.png").decode()
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)


@pytest.mark.skipif(os.name != "nt", reason="a long-path drive exists on Windows only")
def test_decode_reads_a_long_path_drive(tmp_path) -> None:
    pixels = _gradient(3, 4)
    (tmp_path / "t.png").write_bytes(_png_bytes(pixels))
    uri = "\\\\?\\" + str(tmp_path / "t.png")
    for base in (None, tmp_path):
        image = SceneImage(uri=uri).decode(base=base)
        np.testing.assert_array_equal(helper.image_to_array(image), pixels)
