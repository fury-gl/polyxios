"""Media types of embedded images: names, extensions, magic bytes, data URIs.

Shared by the codecs, which meet images as a file name, a data URI or a run
of bytes.
"""

import base64
from urllib.parse import unquote_to_bytes

MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tga": "image/x-tga",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "ktx": "image/ktx",
    "ktx2": "image/ktx2",
    "dds": "image/vnd-ms.dds",
}
MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def media_type_of(uri: str) -> str | None:
    """Return the media type the extension of ``uri`` spells, or None."""
    ext = uri.rsplit(".", 1)[-1].lower() if "." in uri else ""
    return MEDIA_TYPES.get(ext)


def sniff_media(data: bytes) -> str:
    """Return the media type the magic bytes of ``data`` spell, or an empty string."""
    for magic, media in MAGIC:
        if data.startswith(magic):
            return media
    return "image/webp" if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else ""


def decode_data_uri(uri: str) -> tuple[str | None, bytes]:
    """Return ``(media type or None, bytes)`` of a ``data:`` URI.

    ``uri`` must begin with ``data:``, in any case, and that is checked: a
    string without it is refused rather than sliced.

    Raises
    ------
    ValueError
        When the URI does not begin with ``data:`` (``"that does not begin with
        'data:'."``), has no comma (``"without a comma."``), or its base64
        payload does not decode (``"that does not decode: ..."``). The messages
        complete a sentence the caller starts.
    """
    if uri[:5].lower() != "data:":
        raise ValueError("that does not begin with 'data:'.")
    header, sep, payload = uri[5:].partition(",")
    if not sep:
        raise ValueError("without a comma.")
    media, *params = header.split(";")
    if "base64" in (p.strip().lower() for p in params):
        try:
            data = base64.b64decode("".join(payload.split()), validate=True)
        except ValueError as exc:
            raise ValueError(f"that does not decode: {exc}") from exc
    else:
        data = unquote_to_bytes(payload)
    return media or None, data
