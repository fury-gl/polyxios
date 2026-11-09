"""The media helpers: names, magic bytes and data URIs, as the COLLADA reader and
its writer meet them."""

import base64

import pytest

from polyxios._media import decode_data_uri, media_type_of, sniff_media


@pytest.mark.parametrize(
    ("uri", "media"),
    [
        ("textures/wood.PNG", "image/png"),
        ("a/b/photo.jpeg", "image/jpeg"),
        ("map.tiff", "image/tiff"),
        ("no_extension", None),
        ("archive.zip", None),
    ],
)
def test_a_file_name_spells_its_media_type_by_extension(uri, media) -> None:
    assert media_type_of(uri) == media


@pytest.mark.parametrize(
    ("data", "media"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"rest", "image/png"),
        (b"\xff\xd8\xff\xe0 jfif", "image/jpeg"),
        (b"GIF87a....", "image/gif"),
        (b"GIF89a....", "image/gif"),
        (b"BM....", "image/bmp"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", ""),
        (b"", ""),
        (b"plain text", ""),
    ],
)
def test_the_opening_bytes_spell_the_media_type(data, media) -> None:
    assert sniff_media(data) == media


def test_a_data_uri_decodes_to_its_media_type_and_bytes() -> None:
    payload = b"\x89PNG\r\n\x1a\n"
    uri = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    assert decode_data_uri(uri) == ("image/png", payload)


def test_a_base64_payload_may_be_wrapped_and_the_parameter_any_case() -> None:
    uri = "data:image/jpeg;BASE64,AAE\n=" + "\n"
    assert decode_data_uri(uri) == ("image/jpeg", b"\x00\x01")


def test_a_data_uri_without_a_media_type_gives_none() -> None:
    assert decode_data_uri("data:;base64,AAE=") == (None, b"\x00\x01")


def test_a_percent_encoded_payload_is_unquoted() -> None:
    assert decode_data_uri("data:text/plain,a%20b") == ("text/plain", b"a b")


def test_the_scheme_is_matched_in_any_case() -> None:
    assert decode_data_uri("DATA:text/plain,x") == ("text/plain", b"x")


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("http://example.com/a.png", "that does not begin with 'data:'"),
        ("image/png,AAE=", "that does not begin with 'data:'"),
        ("data:image/png;base64", "without a comma"),
        ("data:image/png;base64,!!!", "that does not decode"),
    ],
)
def test_a_data_uri_that_is_no_data_uri_is_refused(uri, message) -> None:
    with pytest.raises(ValueError, match=message):
        decode_data_uri(uri)
