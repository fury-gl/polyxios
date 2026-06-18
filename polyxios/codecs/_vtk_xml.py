"""Shared helpers for VTK XML format readers (VTP, VTR, VTU).

For appended data:
- encoding="raw"   → offsets are BYTE offsets from after the '_' marker.
- encoding="base64"→ offsets are CHARACTER offsets from after the '_' marker.
  Each data block (header and each compressed sub-block) is independently
  base64-encoded with its own padding, so Python's b64decode must NOT be
  applied to the whole text at once — it stops at the first '==' it sees.
"""

import base64
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zlib

import numpy as np

_VTK_TO_NP: dict[str, str] = {
    "Float32": "f4",
    "Float64": "f8",
    "Int8": "i1",
    "Int16": "i2",
    "Int32": "i4",
    "Int64": "i8",
    "UInt8": "u1",
    "UInt16": "u2",
    "UInt32": "u4",
    "UInt64": "u8",
}


def vtk_type_to_np(vtk_type: str) -> str | None:
    """Map VTK XML type name to numpy dtype char.

    Parameters
    ----------
    vtk_type
        VTK type string (e.g. "Float32", "Int64").

    Returns
    -------
    str or None
        Numpy dtype string (e.g. "f4", "i8"), or None for non-numeric types
        such as "String".
    """
    return _VTK_TO_NP.get(vtk_type)


def parse_xml(
    path: Path,
) -> tuple[ET.Element, bytes | None, str, bool, bool, bool]:
    """Read a VTK XML file and return parsed state.

    Handles both inline and appended data sections, including raw-binary
    appended data that would break a naive xml.etree parse.

    Parameters
    ----------
    path
        Path to the VTK XML file.

    Returns
    -------
    tuple
        ``(root, appended, header_type, big_endian, compressed, is_base64)``

        * *appended* — raw base64 text (bytes) when ``is_base64=True``, or raw
          binary bytes when ``is_base64=False``, or ``None`` for inline-only files.
        * *header_type* — ``"UInt32"`` or ``"UInt64"``.
        * *compressed* — ``True`` when a vtkZLibDataCompressor is declared.
        * *is_base64* — ``True`` when the appended section uses base64 encoding.
    """
    raw = path.read_bytes()

    preamble = raw[:512]
    big_endian = b'byte_order="BigEndian"' in preamble
    header_type = "UInt64" if b'header_type="UInt64"' in preamble else "UInt32"
    compressed = b"compressor=" in preamble

    app_pos = raw.find(b"<AppendedData")
    if app_pos == -1:
        return (
            ET.fromstring(raw.decode("utf-8")),
            None,
            header_type,
            big_endian,
            compressed,
            False,
        )

    xml_bytes = raw[:app_pos] + b"</VTKFile>"
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))

    app_tag_end = raw.find(b">", app_pos)
    app_tag = raw[app_pos : app_tag_end + 1].decode("ascii", errors="replace")
    use_base64 = 'encoding="base64"' in app_tag

    if use_base64:
        app_close = raw.find(b"</AppendedData>", app_tag_end)
        b64_text = raw[app_tag_end + 1 : app_close].strip()
        if b64_text.startswith(b"_"):
            b64_text = b64_text[1:]
        return root, b64_text, header_type, big_endian, compressed, True

    underscore = raw.find(b"_", app_tag_end)
    return root, raw[underscore + 1 :], header_type, big_endian, compressed, False


def _decode_chars(text: bytes, start: int, byte_count: int) -> bytes:
    """Decode exactly *byte_count* bytes from a base64 text at char offset *start*.

    Computes the necessary number of base64 characters and adds padding as
    needed before decoding.
    """
    if byte_count <= 0:
        return b""
    char_count = math.ceil(byte_count / 3) * 4
    chunk = text[start : start + char_count]
    if not chunk:
        return b""
    rem = len(chunk) % 4
    if rem:
        chunk = chunk + b"=" * (4 - rem)
    return base64.b64decode(chunk)[:byte_count]


def _read_b64_block(
    b64_text: bytes,
    char_offset: int,
    dtype_str: str,
    *,
    h_dt: np.dtype,
    compressed: bool,
    endian: str,
) -> np.ndarray:
    """Read one data block from the base64 appended text at character offset.

    Parameters
    ----------
    b64_text
        Raw base64 text bytes (NOT decoded up front).
    char_offset
        Character offset of this block within *b64_text*.
    dtype_str
        Numpy dtype string for the element data.
    h_dt
        Numpy dtype for the block header (uint32 or uint64).
    compressed
        If True, block uses vtkZLibDataCompressor format.
    endian
        Endianness prefix for numpy dtype (``"<"`` or ``">"``).
    """
    h_size = h_dt.itemsize
    text = b64_text[char_offset:]

    if not compressed:
        header_bytes = _decode_chars(text, 0, h_size)
        if len(header_bytes) < h_size:
            return np.array([], dtype=endian + dtype_str)
        n_bytes = int(np.frombuffer(header_bytes, dtype=h_dt)[0])
        header_chars = math.ceil(h_size / 3) * 4
        data_bytes = _decode_chars(text, header_chars, n_bytes)
        return np.frombuffer(data_bytes, dtype=endian + dtype_str).copy()

    # Compressed block:
    #   header = [n_blocks: h_t][full_block_size: h_t][last_partial_size: h_t]
    #            [compressed_size_0: h_t] ... [compressed_size_n-1: h_t]
    #   Followed by one independently-encoded base64 chunk per compressed block.
    #
    # Read the minimum header (3 × h_size bytes) to discover n_blocks.
    mini_header_bytes = 3 * h_size
    mini_bytes = _decode_chars(text, 0, mini_header_bytes)
    if len(mini_bytes) < h_size:
        return np.array([], dtype=endian + dtype_str)

    n_blocks = int(np.frombuffer(mini_bytes[:h_size], dtype=h_dt)[0])
    if n_blocks == 0:
        return np.array([], dtype=endian + dtype_str)

    total_header_bytes = (3 + n_blocks) * h_size
    header_bytes = _decode_chars(text, 0, total_header_bytes)
    comp_sizes = np.frombuffer(
        header_bytes[3 * h_size : total_header_bytes], dtype=h_dt
    )

    # All compressed sub-blocks are encoded together as ONE base64 chunk.
    total_compressed = int(np.sum(comp_sizes))
    header_chars = math.ceil(total_header_bytes / 3) * 4
    all_data = _decode_chars(text, header_chars, total_compressed)

    parts: list[bytes] = []
    data_off = 0
    for cs in comp_sizes:
        cs_int = int(cs)
        if cs_int > 0:
            parts.append(zlib.decompress(all_data[data_off : data_off + cs_int]))
        data_off += cs_int

    return np.frombuffer(b"".join(parts), dtype=endian + dtype_str).copy()


def _read_raw_block(
    raw_bytes: bytes,
    byte_offset: int,
    dtype_str: str,
    *,
    h_dt: np.dtype,
    compressed: bool,
    endian: str,
) -> np.ndarray:
    """Read one data block from raw appended bytes at byte offset."""
    h_size = h_dt.itemsize
    view = raw_bytes[byte_offset:]

    if not compressed:
        if len(view) < h_size:
            return np.array([], dtype=endian + dtype_str)
        n_bytes = int(np.frombuffer(view[:h_size], dtype=h_dt)[0])
        return np.frombuffer(
            view[h_size : h_size + n_bytes], dtype=endian + dtype_str
        ).copy()

    if len(view) < h_size:
        return np.array([], dtype=endian + dtype_str)

    n_blocks = int(np.frombuffer(view[:h_size], dtype=h_dt)[0])
    if n_blocks == 0:
        return np.array([], dtype=endian + dtype_str)

    total_header_bytes = (3 + n_blocks) * h_size
    comp_sizes = np.frombuffer(view[3 * h_size : total_header_bytes], dtype=h_dt)

    data_pos = total_header_bytes
    parts: list[bytes] = []
    for cs in comp_sizes:
        cs_int = int(cs)
        if cs_int == 0:
            continue
        parts.append(zlib.decompress(view[data_pos : data_pos + cs_int]))
        data_pos += cs_int

    return np.frombuffer(b"".join(parts), dtype=endian + dtype_str).copy()


def decode_da(
    elem: ET.Element,
    *,
    big_endian: bool,
    appended: bytes | None,
    header_type: str,
    compressed: bool,
    is_base64: bool = False,
) -> np.ndarray:
    """Decode a VTK ``<DataArray>`` element to a 1-D numpy array.

    Parameters
    ----------
    elem
        The ``<DataArray>`` XML element.
    big_endian
        True if the file declares ``byte_order="BigEndian"``.
    appended
        Raw base64 text (when ``is_base64=True``) or raw binary bytes
        (when ``is_base64=False``), or None for inline-only files.
    header_type
        ``"UInt32"`` or ``"UInt64"`` — governs block-header size.
    compressed
        True when vtkZLibDataCompressor is active.
    is_base64
        True when the appended section uses ``encoding="base64"``.

    Returns
    -------
    np.ndarray
        Decoded 1-D array.
    """
    fmt = elem.get("format", "ascii")
    dtype_str = vtk_type_to_np(elem.get("type", "Float64"))
    if dtype_str is None:
        return np.array([], dtype=np.float64)
    endian = ">" if big_endian else "<"

    if fmt == "appended":
        if appended is None:
            return np.array([], dtype=dtype_str)
        offset = int(elem.get("offset", "0"))
        h_dt = np.dtype(endian + ("u8" if header_type == "UInt64" else "u4"))
        if is_base64:
            return _read_b64_block(
                appended,
                offset,
                dtype_str,
                h_dt=h_dt,
                compressed=compressed,
                endian=endian,
            )
        return _read_raw_block(
            appended,
            offset,
            dtype_str,
            h_dt=h_dt,
            compressed=compressed,
            endian=endian,
        )

    text = (elem.text or "").strip()

    if fmt == "ascii":
        return np.array([float(x) for x in text.split() if x], dtype=dtype_str)

    # inline binary / base64
    raw = base64.b64decode(text.encode())
    if len(raw) <= 4:
        return np.array([], dtype=dtype_str)
    return np.frombuffer(raw[4:], dtype=endian + dtype_str).copy().astype(dtype_str)
