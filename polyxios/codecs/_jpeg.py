"""JPEG codec: baseline, progressive and lossless reads; lossy and lossless writes.

The entropy coding runs in the raster kernels. This module walks the markers,
derives the Huffman and quantisation tables, reconstructs the pixels, and
lays the segments around the scans it writes.
"""

from dataclasses import dataclass
import math
import struct
from typing import Any

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

EXTENSION: str = ".jpg"
EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".jpe", ".jfif")

# Markers, by their second byte.
_TEM = 0x01
_SOF0 = 0xC0
_SOF2 = 0xC2
_SOF3 = 0xC3
_DHT = 0xC4
_HIERARCHICAL = frozenset({0xC5, 0xC6, 0xC7})
_ARITHMETIC = frozenset({0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_RST0 = 0xD0
_RST7 = 0xD7
_SOI = 0xD8
_EOI = 0xD9
_SOS = 0xDA
_DQT = 0xDB
_DNL = 0xDC
_DRI = 0xDD
_APP0 = 0xE0
_APP1 = 0xE1
_APP2 = 0xE2
_APP14 = 0xEE
_COM = 0xFE

_SOI_BYTES = b"\xff\xd8"
_EOI_BYTES = b"\xff\xd9"
_ICC_CHUNK = 65519
_IDCT_BLOCK_ROWS = 64
_RGB_IDS = (82, 71, 66)  # 'R', 'G', 'B'

_Table = tuple[np.ndarray, np.ndarray, np.ndarray]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _zigzag() -> np.ndarray:
    """Natural (row-major) index of each zig-zag position."""
    order: list[int] = []
    for s in range(15):
        rows = range(max(0, s - 7), min(s, 7) + 1)
        order.extend(i * 8 + s - i for i in (rows if s % 2 else reversed(rows)))
    return np.array(order, np.intp)


_ZIGZAG: np.ndarray = _zigzag()


def _dct_matrix() -> np.ndarray:
    """``C[u, x] = c(u) / 2 * cos((2x + 1) u pi / 16)``, the 8-point DCT basis."""
    u = np.arange(8, dtype=np.float64)[:, None]
    x = np.arange(8, dtype=np.float64)[None, :]
    scale = np.where(u == 0, 1 / np.sqrt(2), 1.0)
    return scale / 2 * np.cos((2 * x + 1) * u * np.pi / 16)


_DCT: np.ndarray = _dct_matrix()

# Annex K.1 quantisation tables, natural order.
_LUMA_QUANT: np.ndarray = np.array(
    [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ],
    np.int64,
)
_CHROMA_QUANT: np.ndarray = np.array(
    [
        17,
        18,
        24,
        47,
        99,
        99,
        99,
        99,
        18,
        21,
        26,
        66,
        99,
        99,
        99,
        99,
        24,
        26,
        56,
        99,
        99,
        99,
        99,
        99,
        47,
        66,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
    ],
    np.int64,
)

# Annex K.3 Huffman tables: code-length counts for lengths 1-16, then symbols.
_DC_LUMA_BITS = (0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
_DC_LUMA_VALS = tuple(range(12))
_DC_CHROMA_BITS = (0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
_DC_CHROMA_VALS = tuple(range(12))
_AC_LUMA_BITS = (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125)
_AC_LUMA_VALS = (
    0x01,
    0x02,
    0x03,
    0x00,
    0x04,
    0x11,
    0x05,
    0x12,
    0x21,
    0x31,
    0x41,
    0x06,
    0x13,
    0x51,
    0x61,
    0x07,
    0x22,
    0x71,
    0x14,
    0x32,
    0x81,
    0x91,
    0xA1,
    0x08,
    0x23,
    0x42,
    0xB1,
    0xC1,
    0x15,
    0x52,
    0xD1,
    0xF0,
    0x24,
    0x33,
    0x62,
    0x72,
    0x82,
    0x09,
    0x0A,
    0x16,
    0x17,
    0x18,
    0x19,
    0x1A,
    0x25,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x34,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0x59,
    0x5A,
    0x63,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,
    0x69,
    0x6A,
    0x73,
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,
    0x79,
    0x7A,
    0x83,
    0x84,
    0x85,
    0x86,
    0x87,
    0x88,
    0x89,
    0x8A,
    0x92,
    0x93,
    0x94,
    0x95,
    0x96,
    0x97,
    0x98,
    0x99,
    0x9A,
    0xA2,
    0xA3,
    0xA4,
    0xA5,
    0xA6,
    0xA7,
    0xA8,
    0xA9,
    0xAA,
    0xB2,
    0xB3,
    0xB4,
    0xB5,
    0xB6,
    0xB7,
    0xB8,
    0xB9,
    0xBA,
    0xC2,
    0xC3,
    0xC4,
    0xC5,
    0xC6,
    0xC7,
    0xC8,
    0xC9,
    0xCA,
    0xD2,
    0xD3,
    0xD4,
    0xD5,
    0xD6,
    0xD7,
    0xD8,
    0xD9,
    0xDA,
    0xE1,
    0xE2,
    0xE3,
    0xE4,
    0xE5,
    0xE6,
    0xE7,
    0xE8,
    0xE9,
    0xEA,
    0xF1,
    0xF2,
    0xF3,
    0xF4,
    0xF5,
    0xF6,
    0xF7,
    0xF8,
    0xF9,
    0xFA,
)
_AC_CHROMA_BITS = (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119)
_AC_CHROMA_VALS = (
    0x00,
    0x01,
    0x02,
    0x03,
    0x11,
    0x04,
    0x05,
    0x21,
    0x31,
    0x06,
    0x12,
    0x41,
    0x51,
    0x07,
    0x61,
    0x71,
    0x13,
    0x22,
    0x32,
    0x81,
    0x08,
    0x14,
    0x42,
    0x91,
    0xA1,
    0xB1,
    0xC1,
    0x09,
    0x23,
    0x33,
    0x52,
    0xF0,
    0x15,
    0x62,
    0x72,
    0xD1,
    0x0A,
    0x16,
    0x24,
    0x34,
    0xE1,
    0x25,
    0xF1,
    0x17,
    0x18,
    0x19,
    0x1A,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0x59,
    0x5A,
    0x63,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,
    0x69,
    0x6A,
    0x73,
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,
    0x79,
    0x7A,
    0x82,
    0x83,
    0x84,
    0x85,
    0x86,
    0x87,
    0x88,
    0x89,
    0x8A,
    0x92,
    0x93,
    0x94,
    0x95,
    0x96,
    0x97,
    0x98,
    0x99,
    0x9A,
    0xA2,
    0xA3,
    0xA4,
    0xA5,
    0xA6,
    0xA7,
    0xA8,
    0xA9,
    0xAA,
    0xB2,
    0xB3,
    0xB4,
    0xB5,
    0xB6,
    0xB7,
    0xB8,
    0xB9,
    0xBA,
    0xC2,
    0xC3,
    0xC4,
    0xC5,
    0xC6,
    0xC7,
    0xC8,
    0xC9,
    0xCA,
    0xD2,
    0xD3,
    0xD4,
    0xD5,
    0xD6,
    0xD7,
    0xD8,
    0xD9,
    0xDA,
    0xE2,
    0xE3,
    0xE4,
    0xE5,
    0xE6,
    0xE7,
    0xE8,
    0xE9,
    0xEA,
    0xF2,
    0xF3,
    0xF4,
    0xF5,
    0xF6,
    0xF7,
    0xF8,
    0xF9,
    0xFA,
)
# (DC bits, DC symbols, AC bits, AC symbols) for table 0 (luma) and 1 (chroma).
_HUFFMAN = (
    (_DC_LUMA_BITS, _DC_LUMA_VALS, _AC_LUMA_BITS, _AC_LUMA_VALS),
    (_DC_CHROMA_BITS, _DC_CHROMA_VALS, _AC_CHROMA_BITS, _AC_CHROMA_VALS),
)

_SAMPLING: dict[str, tuple[int, int]] = {
    "4:4:4": (1, 1),
    "4:2:2": (2, 1),
    "4:2:0": (2, 2),
}


# =============================================================================
# Huffman tables
# =============================================================================


def _derive_table(bits: list[int], vals: bytes) -> _Table:
    """Decoder form of a DHT table: maxcode, valoffset and symbols.

    A code of length ``l`` is ``code``; it is ``huffval[code + valoffset[l]]``
    when ``code <= maxcode[l]``. ``maxcode[17]`` is a sentinel.
    """
    maxcode = np.full(18, -1, np.int32)
    valoffset = np.zeros(17, np.int32)
    huffval = np.zeros(256, np.uint8)
    huffval[: len(vals)] = np.frombuffer(vals, np.uint8)
    code = 0
    index = 0
    for length in range(1, 17):
        count = bits[length - 1]
        if count:
            valoffset[length] = index - code
            code += count
            index += count
            maxcode[length] = code - 1
        if code > (1 << length):
            raise CodecError("JPEG Huffman table is invalid: its codes overflow.")
        code <<= 1
    maxcode[17] = 0x7FFFFFFF
    return maxcode, valoffset, huffval


def _code_table(
    bits: tuple[int, ...] | list[int], vals: tuple[int, ...] | list[int], size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Encoder form of a table: the code and its length, per symbol."""
    codes = np.zeros(size, np.uint16)
    lengths = np.zeros(size, np.uint8)
    code = 0
    index = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            symbol = vals[index]
            codes[symbol] = code
            lengths[symbol] = length
            code += 1
            index += 1
        code <<= 1
    return codes, lengths


def _least(weight: list[int], exclude: int) -> int:
    best, best_weight = -1, 0
    for index, w in enumerate(weight):
        if w and index != exclude and (best < 0 or w <= best_weight):
            best, best_weight = index, w
    return best


def _optimal_table(freq: list[int]) -> tuple[list[int], list[int]]:
    """Build a Huffman table limited to 16-bit codes (ITU T.81 Annex K.2).

    ``freq`` counts symbols ``0..len(freq)-1``. A reserved symbol keeps the
    all-ones code unused, as the standard requires.
    """
    reserved = len(freq)
    weight = [*freq, 1]
    codesize = [0] * len(weight)
    others = [-1] * len(weight)
    while True:
        first = _least(weight, -1)
        second = _least(weight, first)
        if second < 0:
            break
        weight[first] += weight[second]
        weight[second] = 0
        codesize[first] += 1
        while others[first] >= 0:
            first = others[first]
            codesize[first] += 1
        others[first] = second
        codesize[second] += 1
        while others[second] >= 0:
            second = others[second]
            codesize[second] += 1
    longest = max(codesize)
    counts = [0] * (max(longest, 16) + 1)
    for size in codesize:
        if size:
            counts[size] += 1
    for length in range(len(counts) - 1, 16, -1):
        while counts[length] > 0:
            j = length - 2
            while counts[j] == 0:
                j -= 1
            counts[length] -= 2
            counts[length - 1] += 1
            counts[j + 1] += 2
            counts[j] -= 1
    top = 16
    while counts[top] == 0:
        top -= 1
    counts[top] -= 1
    vals = [
        symbol
        for length in range(1, longest + 1)
        for symbol in range(reserved)
        if codesize[symbol] == length
    ]
    return counts[1:17], vals


# =============================================================================
# Decoding
# =============================================================================


@dataclass(eq=False)
class _Component:
    """A frame component and the arrays its scans fill."""

    cid: int
    h: int
    v: int
    tq: int
    blocks_x: int = 0
    blocks_y: int = 0
    coefs: np.ndarray | None = None  # lossy: int16 (mcus_y*v, mcus_x*h, 64)
    samples: np.ndarray | None = None  # lossless: int32 (height, width)
    qt: np.ndarray | None = None  # quantisation table, latched at first scan
    shift: int = 0  # lossless point transform
    scanned: bool = False


@dataclass(eq=False)
class _Frame:
    height: int
    width: int
    precision: int
    lossless: bool
    progressive: bool
    comps: list[_Component]
    hmax: int
    vmax: int
    mcus_x: int
    mcus_y: int


class _Decoder:
    """Walks the markers of one JPEG file and builds its :class:`ImageData`."""

    def __init__(self, data: bytes, name: str) -> None:
        self.data = data
        self.name = name
        self.qtables: dict[int, np.ndarray] = {}
        self.huffman: dict[tuple[int, int], _Table] = {}
        self.restart = 0
        self.frame: _Frame | None = None
        self.meta: dict[str, Any] = {}
        self.icc: dict[int, bytes] = {}
        self.adobe: int | None = None

    def _err(self, text: str) -> CodecError:
        return CodecError(f"'{self.name}': {text}")

    def decode(self) -> ImageData:
        self._walk()
        self._finish()
        return self._image()

    # -- markers -------------------------------------------------------------

    def _walk(self) -> None:
        pos = 2  # past SOI
        while True:
            marker, pos = self._marker(pos)
            if marker == _EOI:
                return
            if marker in (_SOI, _TEM) or _RST0 <= marker <= _RST7:
                continue
            if marker == 0:
                raise self._err("JPEG has a stray 0x00 where a marker belongs.")
            seg, pos = self._segment(pos)
            if marker == _SOS:
                pos = self._scan(seg, pos)
            else:
                self._dispatch(marker, seg)

    def _marker(self, pos: int) -> tuple[int, int]:
        """Return the marker at ``pos`` (past any fill bytes) and the offset after it."""
        data = self.data
        if pos + 1 >= len(data) or data[pos] != 0xFF:
            raise self._err("JPEG file ends before its EOI marker, or is corrupt.")
        p = pos + 1
        while p < len(data) and data[p] == 0xFF:
            p += 1
        if p >= len(data):
            raise self._err("JPEG file ends before its EOI marker.")
        return data[p], p + 1

    def _segment(self, pos: int) -> tuple[bytes, int]:
        """Return a segment's payload and the offset after it."""
        data = self.data
        if pos + 2 > len(data):
            raise self._err("JPEG segment header runs past the end of the file.")
        (length,) = struct.unpack_from(">H", data, pos)
        if length < 2 or pos + length > len(data):
            raise self._err("JPEG segment runs past the end of the file.")
        return data[pos + 2 : pos + length], pos + length

    def _dispatch(self, marker: int, seg: bytes) -> None:
        if marker == _DQT:
            self._dqt(seg)
        elif marker == _DHT:
            self._dht(seg)
        elif marker == _DRI:
            self._dri(seg)
        elif marker in (_SOF0, 0xC1, _SOF2, _SOF3):
            self._sof(marker, seg)
        elif marker in _HIERARCHICAL:
            raise CodecError("JPEG hierarchical is not supported.")
        elif marker in _ARITHMETIC:
            raise CodecError("JPEG arithmetic-coded is not supported.")
        elif marker == _DNL:
            raise self._err(
                "JPEG DNL markers (a height set after the frame) are not supported."
            )
        elif marker == _APP0:
            self._app0(seg)
        elif marker == _APP1:
            if seg[:6] == b"Exif\0\0":
                self.meta["exif"] = bytes(seg[6:])
        elif marker == _APP2:
            if seg[:12] == b"ICC_PROFILE\0" and len(seg) >= 14:
                self.icc[seg[12]] = bytes(seg[14:])
        elif marker == _APP14:
            if seg[:5] == b"Adobe" and len(seg) >= 12:
                self.adobe = seg[11]
        elif marker == _COM:
            self.meta["comment"] = bytes(seg).decode("latin-1")

    def _app0(self, seg: bytes) -> None:
        if seg[:5] != b"JFIF\0" or len(seg) < 12:
            return
        units = seg[7]
        x, y = struct.unpack_from(">HH", seg, 8)
        if not (x and y):
            return
        if units == 1:
            self.meta["dpi"] = (float(x), float(y))
        elif units == 2:
            self.meta["dpi"] = (x * 2.54, y * 2.54)

    def _dqt(self, seg: bytes) -> None:
        i = 0
        while i < len(seg):
            pq, tq = seg[i] >> 4, seg[i] & 15
            i += 1
            if pq > 1 or tq > 3:
                raise self._err(
                    "JPEG quantisation table has an out-of-range id or precision."
                )
            n = 128 if pq else 64
            if i + n > len(seg):
                raise self._err("JPEG quantisation table is truncated.")
            if pq:
                zig = np.frombuffer(seg, ">u2", 64, i).astype(np.int64)
            else:
                zig = np.frombuffer(seg, np.uint8, 64, i).astype(np.int64)
            i += n
            natural = np.empty(64, np.int64)
            natural[_ZIGZAG] = zig
            self.qtables[tq] = natural

    def _dht(self, seg: bytes) -> None:
        i = 0
        while i < len(seg):
            tc, th = seg[i] >> 4, seg[i] & 15
            if tc > 1 or th > 3:
                raise self._err("JPEG Huffman table has an out-of-range class or id.")
            if i + 17 > len(seg):
                raise self._err("JPEG Huffman table header is truncated.")
            bits = list(seg[i + 1 : i + 17])
            n = sum(bits)
            if n > 256 or i + 17 + n > len(seg):
                raise self._err("JPEG Huffman table is truncated or too long.")
            self.huffman[(tc, th)] = _derive_table(bits, seg[i + 17 : i + 17 + n])
            i += 17 + n

    def _dri(self, seg: bytes) -> None:
        if len(seg) != 2:
            raise self._err("JPEG restart interval segment is malformed.")
        (self.restart,) = struct.unpack(">H", seg)

    def _sof(self, marker: int, seg: bytes) -> None:
        if self.frame is not None:
            raise self._err("JPEG has a second frame header; only one is supported.")
        if len(seg) < 6:
            raise self._err("JPEG frame header is truncated.")
        precision = seg[0]
        height, width = struct.unpack_from(">HH", seg, 1)
        count = seg[5]
        lossless = marker == _SOF3
        progressive = marker == _SOF2
        if lossless:
            if not 2 <= precision <= 16:
                raise self._err(f"JPEG lossless precision {precision} is outside 2-16.")
        elif precision != 8:
            raise CodecError(f"JPEG {precision}-bit lossy samples are not supported.")
        if height == 0:
            raise self._err(
                "JPEG with a height of 0 (a DNL-defined height) is not supported."
            )
        if width == 0:
            raise self._err("JPEG has a zero width.")
        cells = max(width - 1, 0) * max(height - 1, 0)
        validate_header(
            width * height,
            cells,
            4 * cells,
            len(self.data),
            compressed=True,
            spells_vertices=False,
            spells_connectivity=False,
        )
        if count not in (1, 3):
            raise self._err(
                f"JPEG has {count} components; only gray (1) and color (3) are "
                "supported, CMYK and YCCK included."
            )
        if len(seg) < 6 + 3 * count:
            raise self._err("JPEG frame header is truncated.")
        comps = []
        for k in range(count):
            cid, hv, tq = seg[6 + 3 * k], seg[7 + 3 * k], seg[8 + 3 * k]
            h, v = hv >> 4, hv & 15
            if not (1 <= h <= 4 and 1 <= v <= 4):
                raise self._err(f"JPEG sampling factor {h}x{v} is outside 1-4.")
            comps.append(_Component(cid, h, v, tq))
        hmax = max(c.h for c in comps)
        vmax = max(c.v for c in comps)
        if lossless and any(c.h != 1 or c.v != 1 for c in comps):
            raise self._err(
                "JPEG lossless with subsampled components is not supported."
            )
        mcus_x = _ceil_div(width, 8 * hmax)
        mcus_y = _ceil_div(height, 8 * vmax)
        for c in comps:
            c.blocks_x = _ceil_div(_ceil_div(width * c.h, hmax), 8)
            c.blocks_y = _ceil_div(_ceil_div(height * c.v, vmax), 8)
            if lossless:
                c.samples = np.zeros((height, width), np.int32)
            else:
                c.coefs = np.zeros((mcus_y * c.v, mcus_x * c.h, 64), np.int16)
        self.frame = _Frame(
            height=height,
            width=width,
            precision=precision,
            lossless=lossless,
            progressive=progressive,
            comps=comps,
            hmax=hmax,
            vmax=vmax,
            mcus_x=mcus_x,
            mcus_y=mcus_y,
        )

    # -- scans ---------------------------------------------------------------

    def _table(self, cls: int, ident: int) -> _Table:
        table = self.huffman.get((cls, ident))
        if table is None:
            raise self._err("JPEG scan uses an undefined Huffman table.")
        return table

    def _scan(self, seg: bytes, pos: int) -> int:
        """Decode one scan; return the offset of the marker that follows it."""
        frame = self.frame
        if frame is None:
            raise self._err("JPEG scan header comes before the frame header.")
        if not seg or not 1 <= seg[0] <= 4 or len(seg) < 4 + 2 * seg[0]:
            raise self._err("JPEG scan header is malformed.")
        count = seg[0]
        ss, se, ahal = seg[1 + 2 * count], seg[2 + 2 * count], seg[3 + 2 * count]
        ah, al = ahal >> 4, ahal & 15
        comps: list[_Component] = []
        selectors: list[tuple[int, int]] = []
        for k in range(count):
            cs, tdta = seg[1 + 2 * k], seg[2 + 2 * k]
            comp = next((c for c in frame.comps if c.cid == cs), None)
            if comp is None:
                raise self._err(
                    f"JPEG scan names component {cs}, which the frame does not hold."
                )
            if any(comp is other for other in comps):
                raise self._err("JPEG scan lists a component twice.")
            comps.append(comp)
            selectors.append((tdta >> 4, tdta & 15))
        if frame.lossless:
            return self._lossless_scan(frame, comps, selectors, ss, se, ah, al, pos)
        return self._lossy_scan(frame, comps, selectors, ss, se, ah, al, pos)

    def _lossless_scan(
        self,
        frame: _Frame,
        comps: list[_Component],
        selectors: list[tuple[int, int]],
        ss: int,
        se: int,
        ah: int,
        al: int,
        pos: int,
    ) -> int:
        if not 1 <= ss <= 7 or se != 0 or ah != 0 or al >= frame.precision:
            raise self._err(
                "JPEG lossless scan has a predictor, point transform or spectral "
                "range out of range."
            )
        tables = [self._table(0, dc) for dc, _ in selectors]
        for comp in comps:
            comp.shift = al
            comp.scanned = True
        return kernels.jpeg_lossless_decode_scan(
            self.data,
            pos,
            [c.samples for c in comps],
            tables,
            ss,
            al,
            frame.precision,
            self.restart,
            frame.width,
            frame.height,
        )

    def _lossy_scan(
        self,
        frame: _Frame,
        comps: list[_Component],
        selectors: list[tuple[int, int]],
        ss: int,
        se: int,
        ah: int,
        al: int,
        pos: int,
    ) -> int:
        progressive = frame.progressive
        if not progressive:
            ss, se, ah, al = 0, 63, 0, 0
        elif ss == 0:
            if se != 0:
                raise self._err("JPEG DC scan has a nonzero end of spectral selection.")
        elif len(comps) > 1:
            raise CodecError(
                "JPEG progressive AC scan with more than one component is not supported."
            )
        elif not ss <= se <= 63:
            raise self._err(
                "JPEG progressive scan has an out-of-range spectral selection."
            )
        need_dc = ss == 0 and ah == 0
        need_ac = se > 0
        dc_tables: list[_Table | None] = []
        ac_tables: list[_Table | None] = []
        for comp, (dc, ac) in zip(comps, selectors):
            if comp.qt is None:
                qt = self.qtables.get(comp.tq)
                if qt is None:
                    raise self._err("JPEG scan uses an undefined quantisation table.")
                comp.qt = qt
            dc_tables.append(self._table(0, dc) if need_dc else None)
            ac_tables.append(self._table(1, ac) if need_ac else None)
        nxt = kernels.jpeg_decode_scan(
            self.data,
            pos,
            [c.coefs for c in comps],
            [c.h for c in comps],
            [c.v for c in comps],
            [c.blocks_x for c in comps],
            [c.blocks_y for c in comps],
            dc_tables,
            ac_tables,
            frame.mcus_x,
            frame.mcus_y,
            ss,
            se,
            ah,
            al,
            self.restart,
            progressive,
        )
        for comp in comps:
            comp.scanned = True
        return nxt

    # -- assembly ------------------------------------------------------------

    def _finish(self) -> None:
        if self.frame is None:
            raise self._err("JPEG file has no frame header.")
        if not all(c.scanned for c in self.frame.comps):
            raise self._err("JPEG frame has a component that no scan covers.")
        if self.icc:
            self.meta["icc_profile"] = b"".join(self.icc[k] for k in sorted(self.icc))

    def _is_rgb(self, frame: _Frame) -> bool:
        if self.adobe is not None:
            return self.adobe == 0
        return tuple(c.cid for c in frame.comps) == _RGB_IDS

    def _image(self) -> ImageData:
        frame = self.frame
        assert frame is not None
        if frame.lossless:
            planes = [_lossless_plane(c, frame.precision) for c in frame.comps]
            pixels = planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)
        else:
            planes = [_component_plane(c, frame) for c in frame.comps]
            if len(planes) == 1:
                pixels = planes[0]
            elif self._is_rgb(frame):
                pixels = np.stack(planes, axis=-1)
            else:
                pixels = _ycc_to_rgb(planes)
        return array_to_image(pixels, origin="upper", global_attrs=self.meta)


def _lossless_plane(comp: _Component, precision: int) -> np.ndarray:
    """Shift a lossless plane back by its point transform."""
    limit = (1 << (precision - comp.shift)) - 1
    values = np.clip(comp.samples, 0, limit).astype(np.int64) << comp.shift
    return values.astype(np.uint8 if precision <= 8 else np.uint16)


def _component_plane(comp: _Component, frame: _Frame) -> np.ndarray:
    """Reconstruct a lossy component and upsample it to the full image size."""
    plane = _block_plane(comp)
    rows = _ceil_div(frame.height * comp.v, frame.vmax)
    cols = _ceil_div(frame.width * comp.h, frame.hmax)
    return _upsample(
        plane[:rows, :cols],
        comp.h,
        comp.v,
        frame.hmax,
        frame.vmax,
        frame.height,
        frame.width,
    )


def _block_plane(comp: _Component) -> np.ndarray:
    """Dequantise and inverse-transform every block of a component, in slices."""
    assert comp.coefs is not None and comp.qt is not None
    bx, by = comp.blocks_x, comp.blocks_y
    out = np.empty((by * 8, bx * 8), np.uint8)
    qt = comp.qt
    for r0 in range(0, by, _IDCT_BLOCK_ROWS):
        r1 = min(by, r0 + _IDCT_BLOCK_ROWS)
        zig = comp.coefs[r0:r1, :bx]
        natural = np.empty(zig.shape, np.int64)
        natural[..., _ZIGZAG] = zig
        natural *= qt
        blocks = natural.reshape(r1 - r0, bx, 8, 8).astype(np.float64)
        spatial = np.matmul(np.matmul(_DCT.T, blocks), _DCT) + 128.0
        pixels = np.clip(np.rint(spatial), 0, 255).astype(np.uint8)
        out[r0 * 8 : r1 * 8, : bx * 8] = pixels.transpose(0, 2, 1, 3).reshape(
            (r1 - r0) * 8, bx * 8
        )
    return out


def _fancy_h(cur: np.ndarray, lo: int, hi: int, shift: int) -> np.ndarray:
    """Double the columns of ``cur``, libjpeg's triangle filter, one pass.

    Each output takes three parts of its own sample and one of its neighbour
    (the edge sample itself at the borders), rounded by ``lo`` or ``hi``.
    """
    left = np.concatenate([cur[:, :1], cur[:, :-1]], axis=1)
    right = np.concatenate([cur[:, 1:], cur[:, -1:]], axis=1)
    out = np.empty((cur.shape[0], 2 * cur.shape[1]), np.int32)
    out[:, 0::2] = (3 * cur + left + lo) >> shift
    out[:, 1::2] = (3 * cur + right + hi) >> shift
    return out


def _upsample(
    plane: np.ndarray,
    h: int,
    v: int,
    hmax: int,
    vmax: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Upsample a component plane to ``(height, width)`` uint8."""
    if (h, v) == (hmax, vmax):
        return plane[:height, :width]
    data = plane.astype(np.int32)
    if (hmax, vmax) == (2 * h, v):
        out = _fancy_h(data, 1, 2, 2)
    elif (hmax, vmax) == (2 * h, 2 * v):
        above = np.concatenate([data[:1], data[:-1]], axis=0)
        below = np.concatenate([data[1:], data[-1:]], axis=0)
        out = np.empty((2 * data.shape[0], 2 * data.shape[1]), np.int32)
        out[0::2] = _fancy_h(3 * data + above, 8, 7, 4)
        out[1::2] = _fancy_h(3 * data + below, 8, 7, 4)
    else:
        rows = (np.arange(height) * v) // vmax
        cols = (np.arange(width) * h) // hmax
        return plane[np.ix_(rows, cols)]
    return out[:height, :width].astype(np.uint8)


def _ycc_to_rgb(planes: list[np.ndarray]) -> np.ndarray:
    """JFIF YCbCr to RGB, rounded and clipped to bytes."""
    y = planes[0].astype(np.float64)
    cb = planes[1].astype(np.float64) - 128.0
    cr = planes[2].astype(np.float64) - 128.0
    red = y + 1.402 * cr
    green = y - 0.344136 * cb - 0.714136 * cr
    blue = y + 1.772 * cb
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def read_image(path: Source) -> ImageData:
    """Read a JPEG file as a compact :class:`ImageData`.

    Parameters
    ----------
    path
        File path or binary file object.

    Returns
    -------
    ImageData
        One flat layer. Gray files hold ``'intensity'``, color files
        ``'colors'`` (RGB, after the YCbCr conversion). Lossless files keep
        their precision: 8-bit as ``uint8``, 9-16-bit as ``uint16``. The
        metadata segments that are present land in ``global_attrs`` as
        ``dpi``, ``exif``, ``icc_profile`` and ``comment``.

    Raises
    ------
    CodecError
        If the file is not JPEG, is truncated or corrupt, or uses a variant
        polyxios refuses: CMYK or YCCK, 12-bit lossy, arithmetic-coded,
        hierarchical, or subsampled lossless.
    ValidationError
        If the frame header declares more pixels than the caps allow.
    """
    name = source_name(path)
    data = read_bytes(path)
    if data[:3] != b"\xff\xd8\xff":
        media = sniff_media(data)
        hint = ""
        if media and media != "image/jpeg":
            hint = f"; it holds {media} data, so pass fmt= to read it as that"
        raise CodecError(f"'{name}' is not a JPEG file{hint}")
    return _Decoder(data, name).decode()


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Read a JPEG file as a PolyData lattice, one vertex per pixel.

    Parameters
    ----------
    path
        File path or binary file object.
    lazy
        Not supported; the pixels are compressed.

    Returns
    -------
    PolyData
        The image expanded by :meth:`ImageData.to_polydata`.

    Warns
    -----
    UserWarning
        Always. The pixels are expanded into one vertex each; use
        :func:`polyxios.read_image` to get the compact form.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        On a file :func:`read_image` rejects.
    """
    if lazy:
        raise LazyReadError(
            f"{EXTENSION}: lazy reads are not supported; the pixels are compressed."
        )
    warn_caller(
        f"'{source_name(path)}' is an image format (JPEG): read() expands the "
        "pixels into a PolyData lattice, one vertex per pixel. Use "
        "polyxios.read_image() to get the compact ImageData."
    )
    return read_image(path).to_polydata()


# =============================================================================
# Encoding
# =============================================================================


def _segment(marker: int, payload: bytes) -> bytes:
    if len(payload) > 65533:
        raise CodecError(
            f"{EXTENSION}: a JPEG segment holds at most 65533 bytes; got {len(payload)}."
        )
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def _quality(quality: int | None) -> int:
    if quality is None:
        return 75
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or not 1 <= quality <= 100
    ):
        raise CodecError(
            f"{EXTENSION}: quality must be an int in 1-100, got {quality!r}."
        )
    return quality


def _sampling(subsampling: str | None) -> tuple[int, int]:
    if subsampling is None:
        subsampling = "4:2:0"
    if subsampling not in _SAMPLING:
        raise CodecError(
            f"{EXTENSION}: subsampling must be one of {sorted(_SAMPLING)}, "
            f"got {subsampling!r}."
        )
    return _SAMPLING[subsampling]


def _scaled_table(base: np.ndarray, quality: int) -> np.ndarray:
    """Scale an Annex K table the libjpeg way."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((base * scale + 50) // 100, 1, 255).astype(np.int64)


def _density(value: Any) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise CodecError(f"{EXTENSION}: dpi must be finite, got {value!r}.")
    return min(max(round(number), 1), 65535)


def _dpi(meta: dict[str, Any]) -> tuple[int, int] | None:
    if meta.get("dpi") is None:
        return None
    try:
        x, y = meta["dpi"]
        return _density(x), _density(y)
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{EXTENSION}: dpi must be a pair of numbers, not {meta['dpi']!r}."
        ) from exc


def _jfif_segment(meta: dict[str, Any]) -> bytes:
    """APP0 JFIF 1.01: the density in dots per inch, or 1:1 with no units."""
    dpi = _dpi(meta)
    units, x, y = (1, *dpi) if dpi else (0, 1, 1)
    payload = (
        b"JFIF\0" + bytes([1, 1, units]) + struct.pack(">HH", x, y) + bytes([0, 0])
    )
    return _segment(_APP0, payload)


def _metadata_segments(meta: dict[str, Any]) -> list[bytes]:
    """APP1 Exif, APP2 ICC profile chunks and COM, when present."""
    segments: list[bytes] = []
    exif = meta.get("exif")
    if exif is not None:
        segments.append(_segment(_APP1, b"Exif\0\0" + bytes(exif)))
    icc = bytes(meta.get("icc_profile") or b"")
    count = _ceil_div(len(icc), _ICC_CHUNK)
    if count > 255:
        raise CodecError(
            f"{EXTENSION}: an ICC profile of {len(icc)} bytes is too long."
        )
    for k in range(count):
        chunk = icc[k * _ICC_CHUNK : (k + 1) * _ICC_CHUNK]
        segments.append(
            _segment(_APP2, b"ICC_PROFILE\0" + bytes([k + 1, count]) + chunk)
        )
    comment = meta.get("comment")
    if comment is not None:
        try:
            raw = str(comment).encode("latin-1")
        except UnicodeEncodeError as exc:
            raise CodecError(f"{EXTENSION}: comment must be latin-1 text.") from exc
        segments.append(_segment(_COM, raw))
    return segments


def _encode_lossy(
    pixels: np.ndarray, quality: int, sampling: tuple[int, int], meta: dict[str, Any]
) -> bytes:
    height, width = pixels.shape[:2]
    color = pixels.ndim == 3
    if color:
        hmax, vmax = sampling
        rgb = pixels.astype(np.float64)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        planes = [
            0.299 * red + 0.587 * green + 0.114 * blue,
            -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0,
            0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0,
        ]
    else:
        hmax = vmax = 1
        planes = [pixels.astype(np.float64)]
    mcus_x = _ceil_div(width, 8 * hmax)
    mcus_y = _ceil_div(height, 8 * vmax)
    pad_h, pad_w = mcus_y * 8 * vmax, mcus_x * 8 * hmax
    tables = [_scaled_table(_LUMA_QUANT, quality)]
    if color:
        tables.append(_scaled_table(_CHROMA_QUANT, quality))

    arranged: list[np.ndarray] = []
    for index, plane in enumerate(planes):
        h, v = (hmax, vmax) if index == 0 else (1, 1)
        comp = np.pad(
            plane - 128.0, ((0, pad_h - height), (0, pad_w - width)), mode="edge"
        )
        if index and (hmax > 1 or vmax > 1):
            comp = comp.reshape(pad_h // vmax, vmax, pad_w // hmax, hmax).mean(
                axis=(1, 3)
            )
        by, bx = mcus_y * v, mcus_x * h
        blocks = comp.reshape(by, 8, bx, 8).transpose(0, 2, 1, 3)
        freq = np.matmul(np.matmul(_DCT, blocks), _DCT.T)
        qtab = tables[0 if index == 0 else 1].reshape(8, 8)
        quant = np.sign(freq) * np.floor(np.abs(freq) / qtab + 0.5)
        zig = quant.reshape(by, bx, 64)[..., _ZIGZAG].astype(np.int32)
        arranged.append(
            zig.reshape(mcus_y, v, mcus_x, h, 64)
            .transpose(0, 2, 1, 3, 4)
            .reshape(mcus_y * mcus_x, v * h, 64)
        )

    n_mcu = mcus_y * mcus_x
    counts = [a.shape[1] for a in arranged]
    blocks_all = np.ascontiguousarray(
        np.concatenate(arranged, axis=1).reshape(-1, 64), dtype=np.int32
    )
    slots = np.concatenate([np.full(k, i, np.uint8) for i, k in enumerate(counts)])
    table_ids = np.concatenate(
        [np.full(k, 0 if i == 0 else 1, np.uint8) for i, k in enumerate(counts)]
    )
    dc_code = np.zeros((2, 12), np.uint16)
    dc_size = np.zeros((2, 12), np.uint8)
    ac_code = np.zeros((2, 256), np.uint16)
    ac_size = np.zeros((2, 256), np.uint8)
    for t, (dc_bits, dc_vals, ac_bits, ac_vals) in enumerate(_HUFFMAN):
        dc_code[t], dc_size[t] = _code_table(dc_bits, dc_vals, 12)
        ac_code[t], ac_size[t] = _code_table(ac_bits, ac_vals, 256)
    scan = kernels.jpeg_encode_scan(
        blocks_all,
        np.tile(slots, n_mcu),
        np.tile(table_ids, n_mcu),
        dc_code,
        dc_size,
        ac_code,
        ac_size,
        len(planes),
    )

    segments = [_SOI_BYTES, _jfif_segment(meta)]
    segments.extend(_metadata_segments(meta))
    dqt = b"".join(
        bytes([i]) + bytes(t[_ZIGZAG].astype(np.uint8)) for i, t in enumerate(tables)
    )
    segments.append(_segment(_DQT, dqt))
    frame_comps = b"".join(
        bytes([i + 1, (h << 4) | v, 0 if i == 0 else 1])
        for i, (h, v) in enumerate([(hmax, vmax)] + [(1, 1)] * (len(planes) - 1))
    )
    segments.append(
        _segment(
            _SOF0, struct.pack(">BHHB", 8, height, width, len(planes)) + frame_comps
        )
    )
    dht = b""
    for t in range(len(tables)):
        dc_bits, dc_vals, ac_bits, ac_vals = _HUFFMAN[t]
        dht += bytes([t]) + bytes(dc_bits) + bytes(dc_vals)
        dht += bytes([0x10 | t]) + bytes(ac_bits) + bytes(ac_vals)
    segments.append(_segment(_DHT, dht))
    sos = (
        bytes([len(planes)])
        + b"".join(bytes([i + 1, 0x00 if i == 0 else 0x11]) for i in range(len(planes)))
        + bytes([0, 63, 0])
    )
    segments.append(_segment(_SOS, sos))
    segments.append(scan)
    segments.append(_EOI_BYTES)
    return b"".join(segments)


def _encode_lossless(pixels: np.ndarray, meta: dict[str, Any]) -> bytes:
    height, width = pixels.shape[:2]
    gray = pixels.ndim == 2
    depth = 8 if pixels.dtype == np.uint8 else 16
    planes = pixels.reshape(height, width, -1).astype(np.int64)
    count = planes.shape[2]
    # Predictor 1 (Ra), with the first column predicted from above and the
    # first sample from the mid-level constant.
    pred = np.empty_like(planes)
    pred[:, 1:] = planes[:, :-1]
    pred[1:, 0] = planes[:-1, 0]
    pred[0, 0] = 1 << (depth - 1)
    diff = (planes - pred) & 0xFFFF
    diff = np.where(diff > 32768, diff - 65536, diff).reshape(-1)
    magnitude = np.abs(diff)
    category = np.zeros(diff.shape, np.int64)
    for bit in range(16):
        category += magnitude >= (1 << bit)
    freq = np.bincount(category, minlength=17)
    bits, vals = _optimal_table([int(f) for f in freq])
    code, size = _code_table(bits, vals, 17)
    entropy = kernels.jpeg_lossless_encode(diff.astype(np.int32), code, size)

    ids = (1,) if gray else _RGB_IDS
    segments = [_SOI_BYTES]
    if gray:
        segments.append(_jfif_segment(meta))
    segments.extend(_metadata_segments(meta))
    segments.append(_segment(_DHT, bytes([0x00]) + bytes(bits) + bytes(vals)))
    frame = struct.pack(">BHHB", depth, height, width, count) + b"".join(
        bytes([cid, 0x11, 0]) for cid in ids
    )
    segments.append(_segment(_SOF3, frame))
    sos = bytes([count]) + b"".join(bytes([cid, 0]) for cid in ids) + bytes([1, 0, 0])
    segments.append(_segment(_SOS, sos))
    segments.append(entropy)
    segments.append(_EOI_BYTES)
    return b"".join(segments)


def write_image(
    image: ImageData,
    path: Source,
    *,
    quality: int | None = None,
    subsampling: str | None = None,
    lossless: bool = False,
) -> None:
    """Write a flat :class:`ImageData` as a JPEG file.

    Parameters
    ----------
    image
        A flat image whose pixels sit in ``'intensity'`` or ``'colors'``, as
        :func:`polyxios.array_to_image` makes them.
    path
        File path or binary file object.
    quality
        Lossy only: an int in 1-100, default 75.
    subsampling
        Lossy only: ``'4:4:4'``, ``'4:2:2'`` or ``'4:2:0'`` (default).
    lossless
        Write lossless JPEG (SOF3): exact, at 8 or 16 bits per sample.

    Raises
    ------
    CodecError
        On an argument out of range, an image with alpha, a side longer
        than 65535 pixels, or pixels the file cannot hold.

    Warns
    -----
    UserWarning
        Once, naming what the image holds that the file has no place for.
    """
    if not isinstance(lossless, bool):
        raise CodecError(
            f"{EXTENSION}: lossless must be True or False, got {lossless!r}."
        )
    if lossless:
        if quality is not None or subsampling is not None:
            raise CodecError(
                "quality and subsampling apply to lossy JPEG only; drop them with lossless=True."
            )
        pixels, mode = raster_pixels(
            image, fmt=EXTENSION, dtypes=frozenset({"uint8", "uint16"})
        )
    else:
        quality_value = _quality(quality)
        sampling = _sampling(subsampling)
        pixels, mode = raster_pixels(image, fmt=EXTENSION, dtypes=frozenset({"uint8"}))
    if mode in ("LA", "RGBA"):
        raise CodecError("JPEG holds no alpha channel; drop it from the image first.")
    height, width = pixels.shape[:2]
    if max(height, width) > 65535:
        raise CodecError(
            f"{EXTENSION}: JPEG holds at most 65535 pixels along a side; "
            f"this image is {width} x {height}."
        )
    if lossless:
        data = _encode_lossless(pixels, image.global_attrs)
    else:
        data = _encode_lossy(pixels, quality_value, sampling, image.global_attrs)
    written = {"comment", "exif", "icc_profile"}
    if not lossless or mode == "L":
        written.add("dpi")
    warn_left_out(image.global_attrs, fmt=EXTENSION, written=frozenset(written))
    write_bytes(path, data)


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData image lattice as a JPEG file.

    Parameters
    ----------
    poly
        A mesh that is a uniform 2D lattice, as :meth:`ImageData.from_polydata`
        reads one.
    path
        File path or binary file object.
    **opts
        Passed to :func:`write_image`.
    """
    write_image(ImageData.from_polydata(poly, fmt=EXTENSION), path, **opts)
