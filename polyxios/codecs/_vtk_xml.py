"""Shared helpers for VTK XML format readers (VTP, VTR, VTU).

For appended data:
- encoding="raw"   → offsets are BYTE offsets from after the '_' marker.
- encoding="base64"→ offsets are CHARACTER offsets from after the '_' marker.
  Each data block (header and each compressed sub-block) is independently
  base64-encoded with its own padding, so Python's b64decode must NOT be
  applied to the whole text at once - it stops at the first '==' it sees.
"""

import base64
import math
import warnings
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from polyxios._io import Source, read_bytes
from polyxios.exceptions import CodecError

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


_NP_TO_VTK: dict[str, str] = {np_str: name for name, np_str in _VTK_TO_NP.items()}

# What a numeric dtype no VTK type names is written as - bool, float16, a
# datetime all read back as a double, and the bytes have to be that double or
# the header lies about them. A kind that is not numeric at all reaches the
# same fallback and then fails to convert, which format_attr_da names.
_FALLBACK_TYPE: str = "Float64"

# Points per cell and element type name, by how many axes the grid extends
# along. Which axes those are decides nothing but the stride: an x-z plane is
# as much a sheet of quads as an x-y one.
_STRUCTURED_KINDS: dict[int, tuple[str, int]] = {
    1: ("line", 2),
    2: ("quad", 4),
    3: ("hexahedron", 8),
}

# The widest point index an int32 connectivity array can name.
_INT32_MAX: int = int(np.iinfo(np.int32).max)


def np_to_vtk_type(dt: np.dtype) -> tuple[str, np.dtype]:
    """Name a dtype the way a ``DataArray`` header does, and say what it is.

    The two are handed back together because they have to agree: a header
    naming ``Int32`` over the bytes of a double is a file no reader can take.
    A numeric dtype no VTK type names - bool, float16, a datetime - has no
    header of its own, so it is written as the double it converts to. A kind
    that is no number at all is named the same way and refused by the caller
    when the conversion fails, which is where the array's name is known.

    Parameters
    ----------
    dt
        The array's dtype.

    Returns
    -------
    tuple[str, numpy.dtype]
        The VTK type name, and the little-endian dtype whose bytes that name
        describes.

    Examples
    --------
    >>> np_to_vtk_type(np.dtype("int32"))[0]
    'Int32'
    >>> np_to_vtk_type(np.dtype("bool"))[0]
    'Float64'
    """
    name = _NP_TO_VTK.get(dt.str.lstrip("<>|="), _FALLBACK_TYPE)
    return name, np.dtype("<" + _VTK_TO_NP[name])


def format_da(
    name: str,
    arr: np.ndarray,
    *,
    vtk_type: str,
    dtype: np.dtype,
    binary: bool,
    n_comp: int,
    indent: int,
) -> str:
    """Render one ``<DataArray>`` element.

    Parameters
    ----------
    name
        Array name; empty for the unnamed ``Points`` array.
    arr
        Values, of any shape - the element is a flat run either way, and
        ``n_comp`` is what cuts it back into tuples.
    vtk_type
        The type name the element declares.
    dtype
        The dtype ``vtk_type`` describes, little endian. The values are cast
        to it, so the bytes are what the header says they are whichever way
        the machine running this orders them.
    binary
        Base64 the raw bytes instead of spelling the numbers.
    n_comp
        Components per tuple.
    indent
        Spaces to prefix the line with.

    Returns
    -------
    str
        The ``<DataArray>`` line.
    """
    pad = " " * indent
    name_attr = f' Name="{name}"' if name else ""
    comp_attr = f' NumberOfComponents="{n_comp}"' if n_comp > 1 else ""
    values = np.ascontiguousarray(arr, dtype=dtype)

    if binary:
        raw = values.tobytes()
        header = np.array([len(raw)], dtype="<u4").tobytes()
        body = base64.b64encode(header + raw).decode()
        fmt = "binary"
    else:
        body = spell_values(values)
        fmt = "ascii"
    return (
        f'{pad}<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
        f'format="{fmt}">{body}</DataArray>'
    )


def format_attr_da(name: str, arr: np.ndarray, *, binary: bool, indent: int) -> str:
    """Render a point or cell attribute in the type the array is held in.

    An attribute is whatever a caller put in ``vertex_attrs`` or
    ``element_attrs``, and casting it all to a double is a header that
    agrees with the bytes but not with the data: an ``int64`` identifier
    past 2**53 comes back a different number. The dtype names the VTK type,
    and a numeric kind no VTK type names - bool, float16, a datetime - falls
    back to the double it converts to.

    Parameters
    ----------
    name
        Array name.
    arr
        Values, one tuple per row. Every dimension past the first belongs to
        the tuple and is declared as the component count, which is what cuts
        the flat run back into rows on the way in.
    binary
        Base64 the raw bytes instead of spelling the numbers.
    indent
        Spaces to prefix the line with.

    Returns
    -------
    str
        The ``<DataArray>`` line.

    Raises
    ------
    CodecError
        If the values are of a kind no ``DataArray`` can hold - strings, or
        objects. The conversion answers that with a ValueError about one
        element, which names neither the array nor the format refusing it.
    """
    vtk_type, dtype = np_to_vtk_type(arr.dtype)
    try:
        return format_da(
            name,
            arr,
            vtk_type=vtk_type,
            dtype=dtype,
            binary=binary,
            n_comp=components(arr),
            indent=indent,
        )
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"VTK XML: attribute '{name}' holds {arr.dtype!s} values, which"
            " a DataArray has no numeric type for."
        ) from exc


def spell_values(arr: np.ndarray) -> str:
    """Spell an array as the ASCII body of a ``DataArray``.

    ``repr`` of a Python float is the shortest decimal that reads back as
    that double, so a value written this way survives the round trip; a
    fixed precision does not, and ``'%.10g'`` quietly dropped the last seven
    digits of every coordinate. The values are handed to Python as a list
    first: formatting a numpy scalar goes through the array protocol on
    every element, and the conversion costs less than that does.

    Parameters
    ----------
    arr
        Values to spell, of any shape.

    Returns
    -------
    str
        The values, space separated, in C order.
    """
    flat = arr.ravel()
    if flat.dtype.kind == "f" and flat.dtype.itemsize < 8:
        # ``tolist`` widens a float32 to a double, and its shortest decimal
        # is seventeen digits of a value that only carries seven: 0.1 goes
        # out as 0.10000000149011612. numpy spells it at the width it is
        # held at, which reads back as the same float32 in half the bytes.
        return " ".join(flat.astype("U").tolist())
    return " ".join(map(repr, flat.tolist()))


def components(arr: np.ndarray) -> int:
    """Count the components one tuple of an array holds.

    Every dimension past the first belongs to the tuple: a ``(n, 3, 3)``
    tensor array is nine components on n tuples, and declaring the three of
    its second axis cuts the flat run into three times as many rows as the
    mesh has points.

    Parameters
    ----------
    arr
        The array, one tuple per row.

    Returns
    -------
    int
        Components per tuple; 1 for a one-dimensional array.

    Examples
    --------
    >>> components(np.zeros((4, 3, 3)))
    9
    >>> components(np.zeros(4))
    1
    """
    count = 1
    for dim in arr.shape[1:]:
        count *= int(dim)
    return count


def structured_cell_shape(nx: int, ny: int, nz: int) -> tuple[int, int, str]:
    """Say what cells a structured grid of this shape holds, without building them.

    A header has to be validated before an array the size of the grid is
    allocated, so the count comes first and the connectivity second. An
    extent flat along an axis - ``0 4 0 4 0 0``, an image one voxel deep -
    extends along the other two and is a sheet of quads, not a grid of no
    cells at all; reading it as the latter drops every ``CellData`` array it
    carries.

    Parameters
    ----------
    nx, ny, nz
        Cells along each axis, as an extent's spans give them; the grid
        holds one more point than that along each.

    Returns
    -------
    tuple[int, int, str]
        Cells the grid holds, points per cell, and the element type name. A
        grid that extends along no axis is a single point and has no cells,
        which is ``(0, 0, 'vertex')``.

    Examples
    --------
    >>> structured_cell_shape(2, 3, 4)
    (24, 8, 'hexahedron')
    >>> structured_cell_shape(2, 3, 0)
    (6, 4, 'quad')
    >>> structured_cell_shape(0, 0, 0)
    (0, 0, 'vertex')
    """
    spans = [span for span in (nx, ny, nz) if span > 0]
    if not spans:
        return 0, 0, "vertex"

    n_cells = 1
    for span in spans:
        n_cells *= span
    kind, per_cell = _STRUCTURED_KINDS[len(spans)]
    return n_cells, per_cell, kind


def structured_cells(nx: int, ny: int, nz: int) -> tuple[np.ndarray, int, str]:
    """Build the connectivity of a structured grid.

    The three XML grids that expand into an explicit mesh - ``.vti``,
    ``.vts`` and ``.vtr`` - all index their points the same way, so they all
    build the same cells. Done a cell at a time this is one Python loop per
    cell, which on a grid of any size is where the read spends itself; the
    corners are strides from the cell's own origin, so the whole array is
    one add per corner over the origins instead.

    Parameters
    ----------
    nx, ny, nz
        Cells along each axis; the grid holds one more point than that. An
        axis with none is one the grid does not extend along, which drops
        the cells from hexahedra to quads to lines rather than to nothing.

    Returns
    -------
    tuple[numpy.ndarray, int, str]
        Flat connectivity, points per cell, and the element type name. The
        corners of each cell are in the order VTK numbers them, and cells
        run with ``x`` fastest.

    Examples
    --------
    >>> structured_cells(1, 1, 1)[0]
    array([0, 1, 3, 2, 4, 5, 7, 6], dtype=int32)
    >>> structured_cells(1, 1, 0)
    (array([0, 1, 3, 2], dtype=int32), 4, 'quad')
    """
    n_cells, per_cell, kind = structured_cell_shape(nx, ny, nz)
    if not n_cells:
        return np.zeros(0, dtype=np.int32), per_cell, kind

    dims = (nx, ny, nz)
    # A point's index is x + y * (nx + 1) + z * (nx + 1) * (ny + 1), so this
    # is the step from one point to the next along each axis.
    strides = (1, nx + 1, (nx + 1) * (ny + 1))
    axes = [axis for axis in range(3) if dims[axis] > 0]

    # Broadcast the extending axes against each other rather than meshgrid
    # them: the sum is the only array of cell size this builds, and the
    # slowest axis comes first so the cells run with x fastest.
    origins = np.zeros((1,) * len(axes), dtype=np.int64)
    for position, axis in enumerate(reversed(axes)):
        shape = [1] * len(axes)
        shape[position] = dims[axis]
        origins = (
            origins
            + np.arange(dims[axis], dtype=np.int64).reshape(shape) * strides[axis]
        )
    origins = origins.ravel()

    steps = _corner_steps([strides[axis] for axis in axes])
    # An index only fits an int32 while the grid does; a bigger one is kept
    # at the width that holds it rather than wrapped into a negative point.
    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    dtype = np.int32 if n_points <= _INT32_MAX else np.int64

    cells = np.empty((n_cells, per_cell), dtype=dtype)
    for corner, step in enumerate(steps):
        cells[:, corner] = origins + step
    return cells.ravel(), per_cell, kind


def _corner_steps(strides: list[int]) -> tuple[int, ...]:
    """Offset each corner of a structured cell sits at from the cell's origin.

    Parameters
    ----------
    strides
        The point-index step along each axis the grid extends along, slowest
        axis last.

    Returns
    -------
    tuple of int
        One offset per corner, in the order VTK numbers them.
    """
    if len(strides) == 1:
        return 0, strides[0]
    sa, sb = strides[0], strides[1]
    if len(strides) == 2:
        return 0, sa, sa + sb, sb
    sc = strides[2]
    return 0, sa, sa + sb, sb, sc, sa + sc, sa + sb + sc, sb + sc


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
    path: Source,
) -> tuple[ET.Element, bytes | None, str, bool, bool, bool, int]:
    """Read a VTK XML file and return parsed state.

    Handles both inline and appended data sections, including raw-binary
    appended data that would break a naive xml.etree parse.

    Parameters
    ----------
    path
        Path or open file object holding the VTK XML file.

    Returns
    -------
    tuple
        ``(root, appended, header_type, big_endian, compressed, is_base64,
        size)``

        * *appended* - raw base64 text (bytes) when ``is_base64=True``, or raw
          binary bytes when ``is_base64=False``, or ``None`` for inline-only files.
        * *header_type* - ``"UInt32"`` or ``"UInt64"``.
        * *compressed* - ``True`` when a vtkZLibDataCompressor is declared.
        * *is_base64* - ``True`` when the appended section uses base64 encoding.
        * *size* - how many bytes were read, which is what a caller checking a
          declared count against the file it came from wants. It is handed
          back rather than measured again, because measuring a compressed
          source costs a whole decompression pass that this read already paid.
    """
    raw = read_bytes(path)
    size = len(raw)

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
            size,
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
        return root, b64_text, header_type, big_endian, compressed, True, size

    underscore = raw.find(b"_", app_tag_end)
    return (
        root,
        raw[underscore + 1 :],
        header_type,
        big_endian,
        compressed,
        False,
        size,
    )


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


def shaped_da(elem: ET.Element, arr: np.ndarray) -> np.ndarray:
    """Fold a decoded DataArray into one row per tuple.

    A DataArray is written flat whatever its width, and ``NumberOfComponents``
    is the only thing that says how to cut it. Left flat, a three-component
    array on n points is 3n rows long and belongs to no mesh.

    Parameters
    ----------
    elem
        The ``DataArray`` element the values came from.
    arr
        Its decoded values, one dimensional.

    Returns
    -------
    numpy.ndarray
        An ``(n_tuples, n_comp)`` array, or the input unchanged when the
        array is scalar, does not hold whole tuples, or names a width that
        is not a count - which the caller then sees as a row count that
        does not match the mesh.
    """
    try:
        n_comp = int(elem.get("NumberOfComponents", "1"))
    except ValueError:
        # Every other header field here answers a malformed value with a
        # message naming the array; a bare ValueError out of int() names
        # neither it nor the file, and the flat array below is already the
        # shape a caller drops.
        warnings.warn(
            f"VTK XML: DataArray '{elem.get('Name', 'unnamed')}' declares"
            f" NumberOfComponents='{elem.get('NumberOfComponents')}', which"
            " is not a count; read flat.",
            UserWarning,
            stacklevel=4,
        )
        return arr
    if n_comp > 1 and arr.size and arr.size % n_comp == 0:
        return arr.reshape(-1, n_comp)
    return arr


def undecodable_type(elem: ET.Element) -> str | None:
    """Name the type of a DataArray this reader cannot turn into numbers.

    Lets a caller that has no way to carry on - a Points array, whose
    absence would silently shift every later Piece - say what was wrong
    with the array rather than that it held nothing.

    Parameters
    ----------
    elem
        A ``DataArray`` element.

    Returns
    -------
    str or None
        The VTK type name, or None when it decodes to a numpy dtype.
    """
    vtk_type = elem.get("type", "Float64")
    return None if vtk_type_to_np(vtk_type) is not None else vtk_type


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
        ``"UInt32"`` or ``"UInt64"`` - governs block-header size.
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
    vtk_type = elem.get("type", "Float64")
    dtype_str = vtk_type_to_np(vtk_type)
    if dtype_str is None:
        # 'String' is the common case - a label array, which has no numeric
        # form to become an attribute. Whatever it is, the array is skipped
        # and the rest of the piece is read; saying so is what keeps it from
        # looking like the file never held it.
        warnings.warn(
            f"VTK XML: DataArray '{elem.get('Name', 'unnamed')}' has type "
            f"'{vtk_type}', which holds no numbers; skipped.",
            UserWarning,
            stacklevel=5,
        )
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
        return parse_ascii_values(text, dtype_str, name=elem.get("Name", "unnamed"))

    # inline binary / base64
    raw = base64.b64decode(text.encode())
    if len(raw) <= 4:
        return np.array([], dtype=dtype_str)
    return np.frombuffer(raw[4:], dtype=endian + dtype_str).copy().astype(dtype_str)


def parse_ascii_values(
    text: str, dtype_str: str, *, name: str = "unnamed"
) -> np.ndarray:
    """Read the ASCII body of a ``DataArray`` into the type it declares.

    Handing every token to ``float`` first rounds it to a double before the
    declared type ever sees it, which an ``Int64`` identifier past 2**53
    does not survive - ``9007199254740993`` reads back as
    ``9007199254740992``. numpy parses the tokens into the target dtype
    directly, which is exact for the whole integer range and faster than a
    Python-level conversion per value besides.

    Parameters
    ----------
    text
        The element's text, whitespace separated.
    dtype_str
        The numpy dtype character the declared VTK type maps to.
    name
        Array name, for the messages.

    Returns
    -------
    numpy.ndarray
        The values, in ``dtype_str``.

    Raises
    ------
    CodecError
        If a token names no number at all. numpy answers that with a bare
        ``ValueError`` about one token, which names neither the array nor
        the file it came out of.
    """
    tokens = text.split()
    try:
        return np.array(tokens, dtype=dtype_str)
    except (ValueError, OverflowError):
        # Either an integer array spelled with a decimal point, which numpy
        # will not parse, or a value the declared type is too narrow to
        # hold. Neither is worth refusing a whole file over, so the tokens
        # are read as doubles and narrowed the way a C reader would narrow
        # them - which is what the file's own writer did to produce them.
        return _narrow_ascii_values(tokens, dtype_str, name)


def _narrow_ascii_values(tokens: list[str], dtype_str: str, name: str) -> np.ndarray:
    """Read tokens as doubles and narrow them to the declared type.

    Parameters
    ----------
    tokens
        The element's whitespace-separated values.
    dtype_str
        The numpy dtype character the declared VTK type maps to.
    name
        Array name, for the messages.

    Returns
    -------
    numpy.ndarray
        The values, in ``dtype_str``. A value the type cannot hold wraps the
        way a C reader wraps it, and is warned about rather than refused.

    Raises
    ------
    CodecError
        If a token names no number at all.
    """
    try:
        wide = np.array(tokens, dtype=np.float64)
    except ValueError as exc:
        raise CodecError(
            f"VTK XML: DataArray '{name}' holds a value that is not a number ({exc})."
        ) from exc

    narrow = wide.astype(dtype_str)
    # Only the values that had to be narrowed are checked, and only on this
    # path: the common array parsed straight into its own type above and
    # never reaches here.
    if narrow.dtype.kind in "iu" and not np.array_equal(narrow, wide):
        warnings.warn(
            f"VTK XML: DataArray '{name}' holds values its declared type"
            f" '{narrow.dtype.name}' cannot hold; they wrap, as they do for"
            " any reader of this file.",
            UserWarning,
            stacklevel=7,
        )
    return narrow


def piece_count(piece: ET.Element, attr: str, *, fmt: str) -> int:
    """Read a ``Piece``'s declared point or cell count.

    Parameters
    ----------
    piece
        The ``Piece`` element.
    attr
        ``'NumberOfPoints'`` or ``'NumberOfCells'``.
    fmt
        Extension the message names, such as ``'.vtu'``.

    Returns
    -------
    int
        The count, or zero when the attribute is absent - a Piece that
        declares nothing holds nothing.

    Raises
    ------
    CodecError
        If the attribute is present and is not a count. ``int()`` answers
        that with a ValueError naming neither the file nor the Piece.
    """
    text = piece.get(attr)
    if text is None:
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise CodecError(
            f"{fmt}: a Piece declares {attr}='{text}', which is not a count."
        ) from exc


def xml_extent(text: str, *, fmt: str, where: str) -> list[int]:
    """Read a ``WholeExtent`` or ``Extent`` into its six indices.

    Parameters
    ----------
    text
        The attribute's value, six whitespace-separated integers.
    fmt
        Extension the message names, such as ``'.vtr'``.
    where
        Which attribute this is, for the error message.

    Returns
    -------
    list of int
        ``[i0, i1, j0, j1, k0, k1]``.

    Raises
    ------
    CodecError
        If the value is not six integers. Unpacked straight into six names
        it fails with a ValueError about unpacking, which says nothing
        about the file it came out of.
    """
    try:
        extent = [int(value) for value in text.split()]
    except ValueError as exc:
        raise CodecError(
            f"{fmt}: {where}='{text}' is not a run of whole numbers."
        ) from exc
    if len(extent) != 6:
        raise CodecError(
            f"{fmt}: {where}='{text}' holds {len(extent)} indices; an extent holds six."
        )
    return extent


def join_piece_attrs(
    parts: dict[str, list[np.ndarray]], *, expected: int, kind: str
) -> dict[str, np.ndarray]:
    """Concatenate per-piece attribute arrays, dropping any that fall short.

    A multi-piece file may carry an attribute on some pieces and not on
    others - or skip one because polyxios cannot decode its type. Joining
    what is there gives an array shorter than the mesh, whose rows then line
    up against the wrong points from the second piece on. There is no way to
    tell which piece the missing rows belonged to, so the attribute is
    dropped and said out loud.

    Pieces may also disagree about the array's shape, one calling it scalar
    and the next a vector. There is no joined array to make of that either,
    and it is the same answer: drop the array, name it.

    Parameters
    ----------
    parts
        Arrays collected per piece, keyed by array name.
    expected
        Rows the joined array must have: points for point data, cells for
        cell data.
    kind
        ``'point'`` or ``'cell'``, for the warning.

    Returns
    -------
    dict of str to numpy.ndarray
        The arrays that cover the whole mesh.
    """
    joined: dict[str, np.ndarray] = {}
    for name, arrays in parts.items():
        try:
            joined[name] = np.concatenate(arrays)
        except ValueError:
            warnings.warn(
                f"VTK XML: {kind} array '{name}' is shaped differently from"
                f" one Piece to the next ({_shapes(arrays)}), so the pieces"
                " cannot be joined; dropped.",
                UserWarning,
                stacklevel=4,
            )
    return sized_attrs(joined, expected=expected, kind=kind, stacklevel=5)


def sized_attrs(
    found: dict[str, np.ndarray], *, expected: int, kind: str, stacklevel: int = 4
) -> dict[str, np.ndarray]:
    """Keep the arrays that cover the mesh, and name the ones that do not.

    A ``DataArray`` declares nothing about its length; what it holds is
    whatever decoding it gave back, cut into tuples by its component count.
    An array that ends up with a row count the mesh does not have belongs to
    no point and no cell of it, and attaching it anyway is a ``PolyData``
    that fails ``validate`` with a message about lengths rather than about
    the file.

    Parameters
    ----------
    found
        Arrays read out of one attribute section, keyed by name.
    expected
        Rows an array must have: points for point data, cells for cell data.
    kind
        ``'point'`` or ``'cell'``, for the warning.
    stacklevel
        Frames between here and the caller of ``read``, so the warning is
        blamed on the code that asked for the file.

    Returns
    -------
    dict of str to numpy.ndarray
        The arrays that cover the mesh.
    """
    kept: dict[str, np.ndarray] = {}
    for name, arr in found.items():
        if arr.shape[0] != expected:
            warnings.warn(
                f"VTK XML: {kind} array '{name}' covers {arr.shape[0]} of"
                f" {expected} {kind}s, so its rows cannot be matched to the"
                " mesh; dropped.",
                UserWarning,
                stacklevel=stacklevel,
            )
            continue
        kept[name] = arr
    return kept


def _shapes(arrays: list[np.ndarray]) -> str:
    """Spell the shapes of a piece's arrays for a warning.

    Parameters
    ----------
    arrays
        Arrays collected for one attribute, one per piece that carried it.

    Returns
    -------
    str
        The shapes, comma separated.
    """
    return ", ".join(str(arr.shape) for arr in arrays)
