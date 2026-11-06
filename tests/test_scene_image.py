"""``SceneImage.decode``: texture bytes, data URIs and local files as ImageData."""

import base64
import io

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
    with pytest.raises(CodecError, match="outside"):
        SceneImage(uri="../e.png").decode(base=tmp_path / "sub")
    with pytest.raises(CodecError, match="outside"):
        SceneImage(uri=str(tmp_path / "e.png")).decode(base=tmp_path / "sub")


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
