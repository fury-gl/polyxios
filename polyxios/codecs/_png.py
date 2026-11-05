"""PNG codec: gray, gray-alpha, RGB, RGBA and palette images, 1 to 16 bits.

Pixels are decoded to :class:`~polyxios.ImageData` and encoded from one. The
codec uses NumPy and the standard ``zlib`` module only; the scanline
unfiltering is the one kernel that may be compiled.
"""

from collections.abc import Iterator
import math
import struct
from typing import Any
import zlib

import numpy as np

from polyxios._image import ImageData, array_to_image, raster_pixels, warn_left_out
from polyxios._io import Source, read_bytes, source_name, write_bytes
from polyxios._media import sniff_media
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

try:
    from polyxios.codecs import _raster_kernels as kernels
except ImportError:  # pragma: no cover - the compiled module is optional
    from polyxios.codecs import _raster_kernels_fallback as kernels

EXTENSION: str = ".png"
_FORMAT: str = "PNG"

_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"
_KNOWN_CRITICAL: frozenset[bytes] = frozenset({b"IHDR", b"PLTE", b"IDAT", b"IEND"})
_IDAT_PIECE: int = 1 << 20

# Samples per pixel for each colour type a file may spell.
_CHANNELS: dict[int, int] = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DEPTHS: dict[int, frozenset[int]] = {
    0: frozenset({1, 2, 4, 8, 16}),
    2: frozenset({8, 16}),
    3: frozenset({1, 2, 4, 8}),
    4: frozenset({8, 16}),
    6: frozenset({8, 16}),
}

# Colour type and channel count of each mode :func:`raster_pixels` returns.
_WRITE_COLOR: dict[str, int] = {"L": 0, "LA": 4, "RGB": 2, "RGBA": 6}
_WRITE_CHANNELS: dict[str, int] = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}

# The Adam7 passes as (y0, x0, dy, dx); a non-interlaced image is one pass.
_ADAM7: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 8, 8),
    (0, 4, 8, 8),
    (4, 0, 8, 4),
    (0, 2, 4, 4),
    (2, 0, 4, 2),
    (0, 1, 2, 2),
    (1, 0, 2, 1),
)
_WHOLE: tuple[tuple[int, int, int, int], ...] = ((0, 0, 1, 1),)

_GAMMA_SCALE: int = 100000
_METERS_PER_INCH: float = 0.0254
_UINT32_MAX: int = (1 << 32) - 1
_WRITTEN: frozenset[str] = frozenset({"dpi", "gamma", "icc_profile", "text", "exif"})


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Read a PNG file as the lattice of its pixels.

    Parameters
    ----------
    path
        Path or open binary file object.
    lazy
        Not supported: the pixels are compressed, so there is nothing to map.

    Returns
    -------
    PolyData
        One vertex per pixel, with the pixel values as ``colors`` or
        ``intensity``.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        On a file that is not a well-formed PNG.

    Warns
    -----
    UserWarning
        Always, to point at :func:`polyxios.read_image`, which keeps the
        compact :class:`~polyxios.ImageData`.
    """
    if lazy:
        raise LazyReadError(
            f"{EXTENSION}: lazy reads are not supported; the pixels are compressed."
        )
    warn_caller(
        f"'{source_name(path)}' is an image format ({_FORMAT}): read() expands "
        "the pixels into a PolyData lattice, one vertex per pixel. Use "
        "polyxios.read_image() to get the compact ImageData."
    )
    return read_image(path).to_polydata()


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a lattice of pixels as a PNG file.

    Parameters
    ----------
    poly
        A flat lattice, as :meth:`ImageData.to_polydata` builds it.
    path
        Path or open binary file object.
    **opts
        Forwarded to :func:`write_image`.

    Raises
    ------
    CodecError
        If the lattice is not a flat image that PNG can hold.
    """
    write_image(ImageData.from_polydata(poly, fmt=EXTENSION), path, **opts)


def read_image(path: Source) -> ImageData:
    """Read a PNG file as a flat :class:`~polyxios.ImageData`.

    Parameters
    ----------
    path
        Path or open binary file object.

    Returns
    -------
    ImageData
        One z plane with ``colors`` (RGB, RGBA or palette colors), or
        ``intensity`` (gray, with ``alpha`` for gray-alpha or a transparent
        key). The dtype is ``uint8``, or ``uint16`` for 16-bit files. The
        image holds ``dpi``, ``gamma``, ``icc_profile``, ``exif`` and ``text``
        in its ``global_attrs`` where the file spells them.

    Raises
    ------
    CodecError
        On a file that is not a well-formed PNG, or uses a feature polyxios
        does not read.
    ValidationError
        If the header declares more pixels than polyxios will allocate.

    Warns
    -----
    UserWarning
        If the compressed data holds more than the image needs.
    """
    name = source_name(path)
    data = read_bytes(path)
    if not data.startswith(_SIGNATURE):
        raise _not_png(name, data)
    chunks = _walk(data, name)
    kind, ihdr = next(chunks)
    if kind != b"IHDR" or len(ihdr) != 13:
        raise CodecError(f"'{name}': PNG does not open with a 13-byte IHDR chunk.")
    width, height, depth, color, compression, method, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    validate_header(
        width * height,
        max(width - 1, 0) * max(height - 1, 0),
        4 * max(width - 1, 0) * max(height - 1, 0),
        len(data),
        compressed=True,
        spells_vertices=False,
        spells_connectivity=False,
    )
    rest = list(chunks)
    _check_structure(
        name, width, height, depth, color, compression, method, interlace, rest
    )

    idat = b"".join(body for k, body in rest if k == b"IDAT")
    channels = _CHANNELS[color]
    interlaced = interlace == 1
    passes = _passes(width, height, channels, depth, interlaced)
    expected = sum(ph * (1 + _row_bytes(pw, channels, depth)) for *_, ph, pw in passes)
    raw = _inflate_image(idat, expected, name)

    palette = _palette(rest, name) if color == 3 else None
    key = _transparency(rest, color, 0 if palette is None else len(palette), name)
    out_channels = _out_channels(color, key)
    dtype = np.uint16 if depth == 16 else np.uint8
    full = np.zeros((height, width, out_channels), dtype)

    bpp = max(1, channels * depth // 8)
    offset = 0
    for y0, x0, dy, dx, ph, pw in passes:
        rb = _row_bytes(pw, channels, depth)
        span = ph * (1 + rb)
        rows = np.frombuffer(
            kernels.png_unfilter(raw[offset : offset + span], ph, rb, bpp),
            dtype=np.uint8,
        ).reshape(ph, rb)
        offset += span
        samples = _unpack(rows, pw, channels, depth)
        full[y0::dy, x0::dx] = _expand(
            samples, color=color, depth=depth, palette=palette, key=key, name=name
        )

    pixels = full[..., 0] if color == 0 and key is None else full
    return array_to_image(pixels, origin="upper", global_attrs=_metadata(rest, name))


def write_image(image: ImageData, path: Source, *, compress_level: int = 6) -> None:
    """Write a flat :class:`~polyxios.ImageData` as a PNG file.

    Parameters
    ----------
    image
        The image. ``colors`` (3 or 4 wide) or ``intensity`` (+ ``alpha``)
        point attributes hold the pixels; see
        :func:`polyxios._image.raster_pixels`.
    path
        Path or open binary file object.
    compress_level
        zlib level for the image data, 0 (none) to 9 (smallest).

    Raises
    ------
    CodecError
        If the image is not one flat unrotated layer, its values do not fit
        8 or 16 bits, its metadata is out of range, or ``compress_level`` is
        not an int in 0-9.

    Warns
    -----
    UserWarning
        Naming whatever the image holds that a PNG has no place for.
    """
    if (
        isinstance(compress_level, bool)
        or not isinstance(compress_level, int)
        or not 0 <= compress_level <= 9
    ):
        raise CodecError(
            f"{EXTENSION}: compress_level must be an int in 0-9, not "
            f"{compress_level!r}."
        )
    pixels, mode = raster_pixels(
        image, fmt=EXTENSION, dtypes=frozenset({"uint8", "uint16"})
    )
    warn_left_out(image.global_attrs, fmt=EXTENSION, written=_WRITTEN)
    height, width = pixels.shape[:2]
    depth = 16 if pixels.dtype == np.uint16 else 8
    if depth == 16:
        rows = pixels.reshape(height, -1).astype(">u2").view(np.uint8)
    else:
        rows = pixels.reshape(height, -1).astype(np.uint8)
    bpp = _WRITE_CHANNELS[mode] * (depth // 8)
    header = struct.pack(">IIBBBBB", width, height, depth, _WRITE_COLOR[mode], 0, 0, 0)
    compressed = zlib.compress(_filter_rows(rows, bpp).tobytes(), compress_level)
    chunks = [_chunk(b"IHDR", header)]
    chunks.extend(_metadata_chunks(image.global_attrs, compress_level))
    chunks.extend(
        _chunk(b"IDAT", compressed[at : at + _IDAT_PIECE])
        for at in range(0, len(compressed), _IDAT_PIECE)
    )
    chunks.append(_chunk(b"IEND", b""))
    write_bytes(path, _SIGNATURE + b"".join(chunks))


# -----------------------------------------------------------------------------
# Reading


def _not_png(name: str, data: bytes) -> CodecError:
    """Name the failed signature, and the image type the bytes do spell."""
    media = sniff_media(data)
    hint = ""
    if media and media != "image/png":
        hint = f"; it holds {media} data, so pass fmt= to read it as that"
    return CodecError(f"'{name}' is not a {_FORMAT} file{hint}")


def _walk(data: bytes, name: str) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(type, body)`` of each chunk up to and including IEND.

    Every chunk's CRC is checked as it is reached. Bytes after IEND are not
    part of the image and are never looked at.
    """
    pos = len(_SIGNATURE)
    while True:
        if pos + 12 > len(data):
            raise CodecError(f"'{name}': PNG ends before its IEND chunk.")
        (length,) = struct.unpack_from(">I", data, pos)
        kind = data[pos + 4 : pos + 8]
        body_end = pos + 8 + length
        if body_end + 4 > len(data):
            raise CodecError(
                f"'{name}': PNG {kind.decode('latin-1')} chunk runs past the "
                "end of the file."
            )
        body = data[pos + 8 : body_end]
        (crc,) = struct.unpack_from(">I", data, body_end)
        if zlib.crc32(kind + body) != crc:
            raise CodecError(
                f"'{name}': PNG {kind.decode('latin-1')} chunk fails its CRC."
            )
        yield kind, body
        if kind == b"IEND":
            return
        pos = body_end + 4


def _check_structure(
    name: str,
    width: int,
    height: int,
    depth: int,
    color: int,
    compression: int,
    method: int,
    interlace: int,
    rest: list[tuple[bytes, bytes]],
) -> None:
    """Raise unless the header and chunk order are ones polyxios reads."""
    if width == 0 or height == 0:
        raise CodecError(f"'{name}': PNG has a zero-sized dimension.")
    if color not in _DEPTHS or depth not in _DEPTHS[color]:
        raise CodecError(
            f"'{name}': PNG color type {color} at bit depth {depth} is not valid."
        )
    if compression != 0 or method != 0:
        raise CodecError(
            f"'{name}': PNG spells compression method {compression} and filter "
            f"method {method}; only 0 exists."
        )
    if interlace not in (0, 1):
        raise CodecError(f"'{name}': PNG interlace method {interlace} is not valid.")
    for kind, _ in rest:
        if kind == b"IHDR":
            raise CodecError(f"'{name}': PNG has a second IHDR chunk.")
        if kind[0] & 0x20 == 0 and kind not in _KNOWN_CRITICAL:
            raise CodecError(
                f"'{name}': PNG has an unknown critical chunk "
                f"{kind.decode('latin-1')!r}; polyxios cannot read it."
            )
    kinds = [k for k, _ in rest]
    idat_at = [i for i, k in enumerate(kinds) if k == b"IDAT"]
    if not idat_at:
        raise CodecError(f"'{name}': PNG has no IDAT chunk.")
    if idat_at[-1] - idat_at[0] + 1 != len(idat_at):
        raise CodecError(f"'{name}': PNG IDAT chunks are not consecutive.")
    if color == 3 and (b"PLTE" not in kinds or kinds.index(b"PLTE") > idat_at[0]):
        raise CodecError(f"'{name}': PNG palette image has no PLTE chunk before IDAT.")


def _passes(
    width: int, height: int, channels: int, depth: int, interlaced: bool
) -> list[tuple[int, int, int, int, int, int]]:
    """Return ``(y0, x0, dy, dx, rows, columns)`` of each non-empty pass."""
    found = []
    for y0, x0, dy, dx in _ADAM7 if interlaced else _WHOLE:
        pw = len(range(x0, width, dx))
        ph = len(range(y0, height, dy))
        if pw and ph:
            found.append((y0, x0, dy, dx, ph, pw))
    return found


def _row_bytes(pw: int, channels: int, depth: int) -> int:
    """Bytes in one scanline of ``pw`` pixels, without its filter byte."""
    return (pw * channels * depth + 7) // 8


def _inflate_image(idat: bytes, expected: int, name: str) -> bytes:
    """Decompress the image data, to exactly ``expected`` bytes of use."""
    try:
        raw = zlib.decompressobj().decompress(idat, expected + 1)
    except zlib.error as exc:
        raise CodecError(
            f"'{name}': PNG image data is not a valid zlib stream ({exc})."
        ) from exc
    if len(raw) < expected:
        raise CodecError(f"'{name}': PNG image data ends early.")
    if len(raw) > expected:
        warn_caller(
            f"{name}: IDAT holds more data than the image; the excess is ignored."
        )
    return raw


def _unpack(rows: np.ndarray, pw: int, channels: int, depth: int) -> np.ndarray:
    """Split scanlines into ``(rows, pw, channels)`` samples at the file depth."""
    ph = rows.shape[0]
    if depth == 16:
        values = np.ascontiguousarray(rows).view(">u2").astype(np.uint16)
    elif depth == 8:
        values = rows
    else:
        per_byte = 8 // depth
        shifts = np.arange(per_byte - 1, -1, -1, dtype=np.int64) * depth
        mask = (1 << depth) - 1
        values = ((rows[:, :, None] >> shifts) & mask).astype(np.uint8)
        values = values.reshape(ph, -1)
    return values[:, : pw * channels].reshape(ph, pw, channels)


def _expand(
    samples: np.ndarray,
    *,
    color: int,
    depth: int,
    palette: np.ndarray | None,
    key: Any,
    name: str,
) -> np.ndarray:
    """Turn file samples into output pixels of the dtype the depth gives."""
    dtype = np.uint16 if depth == 16 else np.uint8
    top = 65535 if depth == 16 else 255
    if color == 3:
        assert palette is not None
        index = samples[..., 0].astype(np.intp)
        if int(index.max()) >= len(palette):
            raise CodecError(
                f"'{name}': PNG pixel uses palette index {int(index.max())}, past "
                f"the {len(palette)}-entry palette."
            )
        rgb = palette[index]
        if key is None:
            return rgb
        return np.concatenate([rgb, key[index][..., None]], axis=-1)
    if color == 2:
        rgb = samples.astype(dtype)
        if key is None:
            return rgb
        keys = np.asarray(key, dtype=np.int64)
        matched = (samples.astype(np.int64) == keys).all(axis=-1)
        alpha = np.where(matched, 0, top).astype(dtype)[..., None]
        return np.concatenate([rgb, alpha], axis=-1)
    if color == 0:
        value = samples[..., 0].astype(np.int64)
        gray = value * 255 // ((1 << depth) - 1) if depth < 8 else value
        gray = gray.astype(dtype)[..., None]
        if key is None:
            return gray
        alpha = np.where(value == key, 0, top).astype(dtype)[..., None]
        return np.concatenate([gray, alpha], axis=-1)
    return samples.astype(dtype)


def _out_channels(color: int, key: Any) -> int:
    """Channels of the decoded pixels: alpha is added by a transparent key."""
    if color == 0:
        return 2 if key is not None else 1
    if color in (2, 3):
        return 4 if key is not None else 3
    return _CHANNELS[color]


def _first(chunks: list[tuple[bytes, bytes]], kind: bytes) -> bytes | None:
    """Return the body of the first chunk of ``kind``, or None."""
    for k, body in chunks:
        if k == kind:
            return body
    return None


def _palette(chunks: list[tuple[bytes, bytes]], name: str) -> np.ndarray:
    """Return the palette as ``(entries, 3)`` uint8."""
    body = _first(chunks, b"PLTE") or b""
    if not 0 < len(body) <= 768 or len(body) % 3:
        raise CodecError(f"'{name}': PNG palette is not 1-256 RGB entries.")
    return np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)


def _transparency(
    chunks: list[tuple[bytes, bytes]], color: int, palette_size: int, name: str
) -> Any:
    """Return the tRNS key: an int, a triple, an alpha table, or None.

    Colour types 4 and 6 carry alpha already, so their tRNS is ignored.
    """
    body = _first(chunks, b"tRNS")
    if body is None or color in (4, 6):
        return None
    if color == 0:
        if len(body) != 2:
            raise CodecError(f"'{name}': PNG tRNS for gray must be one sample.")
        return struct.unpack(">H", body)[0]
    if color == 2:
        if len(body) != 6:
            raise CodecError(f"'{name}': PNG tRNS for RGB must be three samples.")
        return struct.unpack(">3H", body)
    if len(body) > palette_size:
        raise CodecError(
            f"'{name}': PNG tRNS holds {len(body)} alphas for a "
            f"{palette_size}-entry palette."
        )
    alpha = np.full(palette_size, 255, dtype=np.uint8)
    alpha[: len(body)] = np.frombuffer(body, dtype=np.uint8)
    return alpha


def _split(data: bytes) -> tuple[bytes, bytes]:
    """Split at the first NUL; a missing terminator is a ``ValueError``."""
    head, sep, tail = data.partition(b"\0")
    if not sep:
        raise ValueError("no NUL terminator")
    return head, tail


# Bytes all the compressed metadata chunks of one file may expand to together.
_METADATA_BUDGET: int = 1 << 26


def _inflate(packed: bytes, room: list[int]) -> bytes:
    """Inflate a metadata chunk, charging what it holds to ``room[0]``.

    A few bytes of zlib stream can expand a thousandfold, and a file may carry
    any number of such chunks, so the whole file shares one budget.
    """
    out = zlib.decompressobj().decompress(packed, room[0] + 1)
    if len(out) > room[0]:
        raise ValueError(f"it expands past the {_METADATA_BUDGET}-byte budget")
    room[0] -= len(out)
    return out


def _metadata(chunks: list[tuple[bytes, bytes]], name: str) -> dict[str, Any]:
    """Collect the metadata chunks polyxios keeps; other chunks are ignored."""
    meta: dict[str, Any] = {}
    text: dict[str, str] = {}
    room = [_METADATA_BUDGET]
    for kind, body in chunks:
        try:
            if kind == b"pHYs" and len(body) == 9:
                ppx, ppy, unit = struct.unpack(">IIB", body)
                if unit == 1:
                    meta["dpi"] = (ppx * _METERS_PER_INCH, ppy * _METERS_PER_INCH)
            elif kind == b"gAMA" and len(body) == 4:
                meta["gamma"] = struct.unpack(">I", body)[0] / _GAMMA_SCALE
            elif kind == b"eXIf":
                meta["exif"] = body
            elif kind == b"iCCP":
                _, packed = _split(body)
                meta["icc_profile"] = _inflate(packed[1:], room)
            elif kind == b"tEXt":
                keyword, value = _split(body)
                text[keyword.decode("latin-1")] = value.decode("latin-1")
            elif kind == b"zTXt":
                keyword, packed = _split(body)
                value = _inflate(packed[1:], room)
                text[keyword.decode("latin-1")] = value.decode("latin-1")
            elif kind == b"iTXt":
                keyword, tail = _split(body)
                compressed = tail[0]
                _, tail = _split(tail[2:])
                _, value = _split(tail)
                if compressed:
                    value = _inflate(value, room)
                text[keyword.decode("latin-1")] = value.decode("utf-8")
        except (IndexError, ValueError, zlib.error) as exc:
            raise CodecError(
                f"'{name}': PNG {kind.decode('latin-1')} chunk is malformed ({exc})."
            ) from exc
    if text:
        meta["text"] = text
    return meta


# -----------------------------------------------------------------------------
# Writing


def _chunk(kind: bytes, body: bytes) -> bytes:
    """Frame a chunk: length, type, body and the CRC of type and body."""
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body))
    )


def _uint32(number: float, what: str) -> int:
    """Round a positive finite number into the uint32 a chunk field holds."""
    if not math.isfinite(number) or number <= 0:
        raise CodecError(
            f"{EXTENSION}: {what} must be positive and finite, not {number!r}."
        )
    rounded = round(number)
    if rounded > _UINT32_MAX:
        raise CodecError(f"{EXTENSION}: {what} {number!r} is too large for a PNG.")
    return rounded


def _text_chunk(keyword: object, value: object) -> bytes:
    """Return a tEXt chunk for latin-1 text, otherwise an uncompressed iTXt."""
    if not isinstance(keyword, str) or not 1 <= len(keyword) <= 79 or "\0" in keyword:
        raise CodecError(
            f"{EXTENSION}: text keyword {keyword!r} must be 1-79 characters with no NUL."
        )
    try:
        key = keyword.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise CodecError(
            f"{EXTENSION}: text keyword {keyword!r} must be latin-1 characters."
        ) from exc
    if not isinstance(value, str):
        raise CodecError(f"{EXTENSION}: text {keyword!r} must be a str.")
    try:
        return _chunk(b"tEXt", key + b"\0" + value.encode("latin-1"))
    except UnicodeEncodeError:
        # keyword, NUL, compression flag and method, empty language tag and
        # empty translated keyword, each closed by a NUL, then the UTF-8 text.
        return _chunk(b"iTXt", key + b"\0\0\0\0\0" + value.encode("utf-8"))


def _metadata_chunks(meta: dict[str, Any], compress_level: int) -> list[bytes]:
    """Frame the metadata a PNG holds, in the order the spec asks for it."""
    chunks: list[bytes] = []
    if "gamma" in meta:
        scaled = _uint32(float(meta["gamma"]) * _GAMMA_SCALE, "gamma")
        chunks.append(_chunk(b"gAMA", struct.pack(">I", scaled)))
    if "icc_profile" in meta:
        profile = zlib.compress(bytes(meta["icc_profile"]), compress_level)
        chunks.append(_chunk(b"iCCP", b"ICC Profile\0\0" + profile))
    if "dpi" in meta:
        dpi = tuple(meta["dpi"])
        if len(dpi) != 2:
            raise CodecError(
                f"{EXTENSION}: dpi must be a pair of numbers, not {dpi!r}."
            )
        ppx = _uint32(float(dpi[0]) / _METERS_PER_INCH, "dpi")
        ppy = _uint32(float(dpi[1]) / _METERS_PER_INCH, "dpi")
        chunks.append(_chunk(b"pHYs", struct.pack(">IIB", ppx, ppy, 1)))
    for keyword, value in dict(meta.get("text", {})).items():
        chunks.append(_text_chunk(keyword, value))
    if "exif" in meta:
        chunks.append(_chunk(b"eXIf", bytes(meta["exif"])))
    return chunks


def _paeth(left: np.ndarray, up: np.ndarray, upleft: np.ndarray) -> np.ndarray:
    """Paeth predictor over whole rows; ties go to left, then up."""
    estimate = left + up - upleft
    d_left = np.abs(estimate - left)
    d_up = np.abs(estimate - up)
    d_upleft = np.abs(estimate - upleft)
    return np.where(
        (d_left <= d_up) & (d_left <= d_upleft),
        left,
        np.where(d_up <= d_upleft, up, upleft),
    )


def _filter_rows(rows: np.ndarray, bpp: int) -> np.ndarray:
    """Filter every scanline with the cheapest of the five filters.

    Parameters
    ----------
    rows
        ``(height, row_bytes)`` uint8 scanlines.
    bpp
        Bytes per complete pixel, the distance to the left neighbour.

    Returns
    -------
    numpy.ndarray
        ``(height, row_bytes + 1)`` uint8, each row led by its filter type.
        The cost of a filter is the sum of the absolute signed bytes; a tie
        goes to the lower filter number.
    """
    cur = rows.astype(np.int16)
    left = np.zeros_like(cur)
    left[:, bpp:] = cur[:, :-bpp]
    up = np.zeros_like(cur)
    up[1:] = cur[:-1]
    upleft = np.zeros_like(cur)
    upleft[1:, bpp:] = cur[:-1, :-bpp]
    filtered = (
        cur,
        (cur - left) & 0xFF,
        (cur - up) & 0xFF,
        (cur - ((left + up) >> 1)) & 0xFF,
        (cur - _paeth(left, up, upleft)) & 0xFF,
    )
    costs = np.stack(
        [np.minimum(f, 256 - f).sum(axis=1, dtype=np.int64) for f in filtered]
    )
    best = np.argmin(costs, axis=0)
    out = np.empty((rows.shape[0], rows.shape[1] + 1), dtype=np.uint8)
    out[:, 0] = best
    for kind, values in enumerate(filtered):
        pick = best == kind
        out[pick, 1:] = values[pick].astype(np.uint8)
    return out
