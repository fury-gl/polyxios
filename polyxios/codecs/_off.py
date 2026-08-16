"""OFF (Object File Format) codec - ASCII and binary read + write."""

from collections.abc import Iterator
import io
from itertools import chain, islice
from pathlib import Path
import re
from typing import Any, NamedTuple
import warnings

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    MAX_SAFE_CONN,
    MAX_SAFE_ELEMENTS,
    MAX_SAFE_VERTICES,
)
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".off"

# Header keyword is ``[ST][C][N][4][n]OFF``, optionally followed by ``BINARY``.
# ``N``/``C``/``ST`` add per-vertex normals, colours and texture coordinates;
# ``4`` and ``n`` change the coordinate count itself. Case matters: ``N`` asks
# for normals while ``n`` announces a dimension, so this pattern is matched
# case-sensitively and only a bare ``off`` is folded.
_HEADER_RE: re.Pattern[str] = re.compile(r"^(ST)?(C)?(N)?([34n])?OFF$")

# OFF faces are polygons; a face with fewer than three vertices is not one.
_MIN_FACE_SIZE: int = 3

# Matches the other ASCII surface writers (.obj, .ply, .avs); pass float_fmt to
# trade file size for the full float64 round trip.
_DEFAULT_FLOAT_FMT: str = ".10g"

# Binary OFF is 32-bit big-endian throughout - "Binary data comprise 32-bit
# integers and 32-bit IEEE-format floats, both in big-endian format".
_INT_BE: np.dtype = np.dtype(">i4")
_FLOAT_BE: np.dtype = np.dtype(">f4")
_WORD: int = 4

_BOM: bytes = b"\xef\xbb\xbf"

# Colour channels are held as floats in 0..1 whatever the file said, so a
# consumer never has to ask which scale it is looking at. ASCII OFF also
# permits 0..255 integers; the spelling is remembered per block in
# ``global_attrs`` so a round trip reproduces it. Binary OFF is floats only.
_BYTE_SCALE: float = 255.0
_VERTEX_COLOR_FORMAT_KEY: str = "off_vertex_color_format"
_FACE_COLOR_FORMAT_KEY: str = "off_face_color_format"
_BYTE_FORMAT: str = "byte"
_FLOAT_FORMAT: str = "float"

# Attribute names, shared with the other codecs where one already exists
# (.obj writes vertex normals under "normals").
_NORMALS: str = "normals"
_COLORS: str = "colors"
_TEXCOORDS: str = "texcoords"
_COLOR_INDEX: str = "color_index"

# Binary vertex colours are always four floats; ASCII may spell three or four.
_BINARY_COLOR_COMPONENTS: int = 4
_VALID_COLOR_COMPONENTS: frozenset[int] = frozenset({3, 4})

# Element types OFF can express as a face, and the node order OFF expects.
# VTK-style ``pixel`` numbering runs 0,1,3,2, so it has to be reordered into
# the boundary walk a polygonal face implies.
_TRI_CODE: int = ELEMENT_TYPES["triangle"]
_QUAD_CODE: int = ELEMENT_TYPES["quad"]
_POLY_CODE: int = ELEMENT_TYPES["polygon"]
_PIXEL_CODE: int = ELEMENT_TYPES["pixel"]
_WRITABLE_TYPES: frozenset[int] = frozenset(
    {_TRI_CODE, _QUAD_CODE, _POLY_CODE, _PIXEL_CODE}
)


class _Layout(NamedTuple):
    """What a parsed header keyword says each vertex record carries."""

    has_texcoords: bool
    has_colors: bool
    has_normals: bool
    dim: int | None  # None when the header defers the dimension (``nOFF``)
    binary: bool


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse an OFF file and return a PolyData.

    Parameters
    ----------
    path
        Path to the .off file.
    lazy
        Ignored; OFF is read eagerly in both its ASCII and binary spellings.

    Returns
    -------
    PolyData
        Parsed mesh with triangle, quad, or polygon elements. Per-vertex
        normals, colours and texture coordinates land in ``vertex_attrs``
        under ``"normals"``, ``"colors"`` and ``"texcoords"``; per-face
        colours land in ``element_attrs["colors"]``, and a per-face colormap
        index in ``element_attrs["color_index"]``.

    Raises
    ------
    CodecError
        On malformed OFF data.

    Notes
    -----
    Colour channels are returned as floats in 0..1 regardless of how the file
    spelled them. ASCII OFF also allows 0..255 integers, so the spelling of
    each block is recorded in ``global_attrs`` under
    ``"off_vertex_color_format"`` and ``"off_face_color_format"`` and is
    reproduced on write.

    ``4OFF`` and ``nOFF`` with a dimension other than three cannot be
    represented by PolyData's 3-D vertices and are rejected rather than
    silently flattened.
    """
    if lazy:
        warnings.warn(
            ".off: lazy=True ignored; OFF always loads eagerly.",
            stacklevel=2,
        )

    # The byte-order mark a Windows editor leaves behind would otherwise glue
    # itself to the header keyword; the other text codecs drop it by reading
    # through utf-8-sig, and the binary spelling has to be sniffed as bytes.
    raw = Path(path).read_bytes().removeprefix(_BOM)
    header, body_offset = _sniff_header(raw)
    if header is None:
        raise CodecError(".off: empty file.")
    layout = _parse_header(header)

    if layout.binary:
        return _read_binary(raw[body_offset:], layout)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodecError(".off: file is not valid UTF-8 text.") from exc
    return _read_ascii(text, layout)


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to an OFF file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    **opts
        ``binary`` writes big-endian binary OFF instead of ASCII;
        ``float_fmt`` overrides the ASCII coordinate format specifier.

    Raises
    ------
    CodecError
        If a face references a vertex the mesh does not have, or if
        ``float_fmt`` is not a usable format specifier.

    Notes
    -----
    OFF stores polygonal surfaces only. Elements of any other type - lines and
    volume cells among them - are skipped with a warning, one per type. Every
    vertex is written whether or not a surviving face references it, so vertex
    indices are preserved across a round trip.

    ``vertex_attrs`` entries named ``"normals"``, ``"colors"`` and
    ``"texcoords"``, and ``element_attrs`` entries named ``"colors"`` or
    ``"color_index"``, select the header variant (``NOFF``, ``STCNOFF``, ...)
    and are written back out. An entry whose shape does not fit its channel is
    skipped with a warning rather than corrupting the record layout. Binary
    OFF carries colours as four floats, so a three-channel colour gains an
    opaque alpha and the 0..255 spelling is not available.
    """
    binary = bool(opts.pop("binary", False))
    float_fmt = opts.pop("float_fmt", _DEFAULT_FLOAT_FMT)
    if opts:
        warnings.warn(
            f".off write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )
    try:
        format(0.0, float_fmt)
    except (TypeError, ValueError) as exc:
        raise CodecError(f".off: float_fmt {float_fmt!r} is not usable.") from exc

    n_verts = poly.vertices.shape[0]
    writable = _select_faces(poly)
    cells = [_face_cell(poly, i, n_verts) for i in writable]

    normals = _vertex_channel(poly, _NORMALS, 3, n_verts)
    colors = _vertex_channel(poly, _COLORS, _VALID_COLOR_COMPONENTS, n_verts)
    texcoords = _vertex_channel(poly, _TEXCOORDS, 2, n_verts)
    face_colors, face_index = _face_channels(poly, writable)

    keyword = "".join(
        (
            "ST" if texcoords is not None else "",
            "C" if colors is not None else "",
            "N" if normals is not None else "",
            "OFF",
        )
    )

    path = Path(path)
    if binary:
        path.write_bytes(
            _encode_binary(
                keyword,
                poly.vertices,
                cells,
                normals,
                colors,
                texcoords,
                face_colors,
                face_index,
            )
        )
        return
    path.write_text(
        _encode_ascii(
            keyword,
            poly.vertices,
            cells,
            normals,
            colors,
            texcoords,
            face_colors,
            face_index,
            float_fmt,
            poly.global_attrs,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------


def _sniff_header(raw: bytes) -> tuple[list[str] | None, int]:
    """Return the header line's tokens and the offset just past that line.

    Binary data starts at the byte after the first newline following the
    header, so the header has to be located without decoding the whole file.
    """
    offset = 0
    while offset < len(raw):
        end = raw.find(b"\n", offset)
        stop = len(raw) if end == -1 else end
        # latin-1 never fails, and a header line is ASCII by construction.
        tokens = raw[offset:stop].decode("latin-1").split("#")[0].split()
        offset = stop if end == -1 else end + 1
        if tokens:
            return tokens, offset
    return None, offset


def _parse_header(tokens: list[str]) -> _Layout:
    """Turn a header line's tokens into the record layout they describe."""
    keyword = tokens[0]
    match = _HEADER_RE.match(keyword)
    if match is None:
        if keyword.upper() != "OFF":
            raise CodecError(f".off: not an OFF file; first token is {keyword!r}.")
        match = _HEADER_RE.match("OFF")
    assert match is not None  # noqa: S101 - "OFF" always matches
    st, color, normals, dim_flag = match.groups()

    if dim_flag is None or dim_flag == "3":
        dim: int | None = 3
    elif dim_flag == "4":
        dim = 4
    else:
        dim = None  # nOFF: the dimension leads the counts

    binary = len(tokens) > 1 and tokens[1].upper() == "BINARY"
    return _Layout(bool(st), bool(color), bool(normals), dim, binary)


def _check_dim(dim: int) -> None:
    """Reject a coordinate count PolyData's 3-D vertices cannot hold."""
    if dim != 3:
        raise CodecError(
            f".off: {dim}-D coordinates are not supported; only 3-D OFF is read."
        )


# --------------------------------------------------------------------------
# ASCII read
# --------------------------------------------------------------------------


def _read_ascii(text: str, layout: _Layout) -> PolyData:
    """Parse the ASCII spelling, whose records are one per line.

    The records are consumed from a lazy iterator rather than a materialised
    token table: a table costs one list and one string per token, which for a
    mesh of any size dwarfs the arrays being built.
    """
    lines = _data_lines(text)
    head = next(lines)[1:]

    # ``nOFF`` puts the dimension in front of the three counts, and any of them
    # may sit on the keyword's line or on following ones.
    wanted = 3 if layout.dim is not None else 4
    counts = list(head)
    while len(counts) < wanted:
        more = next(lines, None)
        if more is None:
            raise CodecError(".off: header missing vertex/face/edge counts.")
        counts.extend(more)

    if layout.dim is None:
        dim = _checked_count(counts[0], MAX_SAFE_VERTICES, "dimension")
        counts = counts[1:]
    else:
        dim = layout.dim
    _check_dim(dim)

    n_verts = _checked_count(counts[0], MAX_SAFE_VERTICES, "vertex")
    n_faces = _checked_count(counts[1], MAX_SAFE_ELEMENTS, "face")
    _checked_count(counts[2], MAX_SAFE_CONN, "edge")  # declared, never trusted

    # Counting newlines bounds the records the file can possibly hold without
    # allocating anything, so a header claiming millions of vertices is
    # rejected before the array for them is asked for.
    available = text.count("\n") + 1
    if n_verts + n_faces > available:
        raise CodecError(
            f".off: file truncated - expected {n_verts + n_faces} vertex and"
            f" face lines, file holds at most {available}."
        )

    vertices, vertex_attrs, vertex_color_fmt = _parse_ascii_vertices(
        lines, layout, dim, n_verts
    )
    conn, offsets, types, element_attrs, face_color_fmt = _parse_ascii_faces(
        lines, n_verts, n_faces
    )

    global_attrs: dict[str, Any] = {}
    if vertex_color_fmt is not None:
        global_attrs[_VERTEX_COLOR_FORMAT_KEY] = vertex_color_fmt
    if face_color_fmt is not None:
        global_attrs[_FACE_COLOR_FORMAT_KEY] = face_color_fmt

    return PolyData(
        vertices=vertices,
        connectivity=conn,
        offsets=offsets,
        element_types=types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        global_attrs=global_attrs,
    )


def _parse_ascii_vertices(
    lines: Iterator[list[str]], layout: _Layout, dim: int, n_verts: int
) -> tuple[np.ndarray, dict[str, np.ndarray], str | None]:
    """Read the vertex block, splitting each line into its declared channels."""
    n_normal = 3 if layout.has_normals else 0
    n_tex = 2 if layout.has_texcoords else 0
    n_color = 0
    body: Iterator[list[str]] = islice(lines, n_verts)
    if layout.has_colors and n_verts:
        # The header says a colour is present but not how wide it is; the
        # first record's spare columns are the only thing that can say. Peeking
        # costs one line, which is then put back in front of the rest.
        first = next(body, None)
        if first is None:
            raise CodecError(".off: file truncated in the vertex block.")
        n_color = len(first) - dim - n_normal - n_tex
        if n_color not in _VALID_COLOR_COMPONENTS:
            raise CodecError(
                f".off: vertex colour must be 3 or 4 components, found {n_color}."
            )
        body = chain([first], body)

    wanted = dim + n_normal + n_color + n_tex
    vertices = np.empty((n_verts, 3), dtype=np.float64)
    normals = np.empty((n_verts, 3), dtype=np.float64) if n_normal else None
    colors = np.empty((n_verts, n_color), dtype=np.float64) if n_color else None
    texcoords = np.empty((n_verts, 2), dtype=np.float64) if n_tex else None
    colors_are_bytes = True

    seen = 0
    for i, parts in enumerate(body):
        seen = i + 1
        if len(parts) < wanted:
            raise CodecError(
                f".off: vertex line {i + 1} needs {wanted} values, got"
                f" {len(parts)}: {' '.join(parts)!r}."
            )
        try:
            vertices[i] = [float(tok) for tok in parts[:3]]
            at = dim
            if normals is not None:
                normals[i] = [float(tok) for tok in parts[at : at + 3]]
                at += 3
            if colors is not None:
                chunk = parts[at : at + n_color]
                colors[i] = [float(tok) for tok in chunk]
                colors_are_bytes = colors_are_bytes and all(map(_is_int_token, chunk))
                at += n_color
            if texcoords is not None:
                texcoords[i] = [float(tok) for tok in parts[at : at + 2]]
        except ValueError as exc:
            raise CodecError(
                f".off: non-numeric vertex line {i + 1}: {' '.join(parts)!r}."
            ) from exc

    if seen != n_verts:
        raise CodecError(
            f".off: file truncated - expected {n_verts} vertex lines, got {seen}."
        )

    attrs: dict[str, np.ndarray] = {}
    if normals is not None:
        attrs[_NORMALS] = normals
    if texcoords is not None:
        attrs[_TEXCOORDS] = texcoords
    color_fmt: str | None = None
    if colors is not None:
        color_fmt = _BYTE_FORMAT if colors_are_bytes else _FLOAT_FORMAT
        attrs[_COLORS] = colors / _BYTE_SCALE if colors_are_bytes else colors
    return vertices, attrs, color_fmt


def _parse_ascii_faces(
    lines: Iterator[list[str]], n_verts: int, n_faces: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], str | None]:
    """Read the face block, keeping any colour trailing the vertex indices."""
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    # A face colour trails the indices to end of line. Only non-empty tails are
    # kept, so the common colourless file pays nothing and a file that colours
    # just some of its faces shows up as a short list.
    color_tokens: list[list[str]] = []

    seen = 0
    for i, parts in enumerate(islice(lines, n_faces)):
        seen = i + 1
        n = _checked_count(parts[0], MAX_SAFE_CONN, "face size")
        if n < _MIN_FACE_SIZE:
            raise CodecError(
                f".off: face {i} has {n} vertices; OFF faces need at least"
                f" {_MIN_FACE_SIZE}."
            )
        if len(parts) < 1 + n:
            raise CodecError(
                f".off: face {i} declares {n} vertices but the line holds"
                f" {len(parts) - 1}."
            )
        try:
            indices = [int(tok) for tok in parts[1 : 1 + n]]
        except ValueError as exc:
            raise CodecError(f".off: non-integer vertex index in face {i}.") from exc
        for v in indices:
            if not 0 <= v < n_verts:
                raise CodecError(
                    f".off: face {i} references vertex {v}, outside [0, {n_verts})."
                )
        if len(conn_list) + n > MAX_SAFE_CONN:
            raise CodecError(
                f".off: connectivity size exceeds the safety cap {MAX_SAFE_CONN}."
            )
        conn_list.extend(indices)
        offsets_list.append(offsets_list[-1] + n)
        types_list.append(_face_type(n))

        tail = parts[1 + n :]
        if tail:
            color_tokens.append(tail)

    if seen != n_faces:
        raise CodecError(
            f".off: file truncated - expected {n_faces} face lines, got {seen}."
        )

    attrs, color_fmt = _ascii_face_color_attrs(color_tokens, n_faces)
    # Vertex indices stay below the vertex cap, but the running offset is a
    # cumulative sum that MAX_SAFE_CONN lets past 2**31, where int32 would
    # overflow; widen the pair together so the CSR dtypes stay consistent.
    idx_dtype = np.int64 if offsets_list[-1] > np.iinfo(np.int32).max else np.int32
    return (
        np.array(conn_list, dtype=idx_dtype),
        np.array(offsets_list, dtype=idx_dtype),
        np.array(types_list, dtype=np.uint8),
        attrs,
        color_fmt,
    )


def _ascii_face_color_attrs(
    color_tokens: list[list[str]], n_faces: int
) -> tuple[dict[str, np.ndarray], str | None]:
    """Turn the colour tails of ASCII face lines into element attributes."""
    if not color_tokens:
        return {}, None
    try:
        values = [[float(tok) for tok in tail] for tail in color_tokens]
    except ValueError:
        warnings.warn(
            ".off: face colours are not numeric; dropping them.", stacklevel=2
        )
        return {}, None
    # Integer channels are 0..255 and float channels 0..1, so the spelling is
    # what tells the two scales apart.
    are_bytes = all(_is_int_token(tok) for tail in color_tokens for tok in tail)
    return _face_color_attrs(values, n_faces, are_bytes=are_bytes)


def _face_color_attrs(
    values: list[list[float]], n_faces: int, *, are_bytes: bool
) -> tuple[dict[str, np.ndarray], str | None]:
    """Lay per-face colour channels out as one array.

    A colorspec is nothing, one colormap index, or three or four RGB[A]
    channels. A file that mixes those spellings - including one that colours
    only some of its faces - cannot be laid out as one array, so the colours
    are dropped with a warning and the geometry is kept.
    """
    if not values:
        return {}, None
    widths = {len(v) for v in values}
    if len(values) != n_faces or not (widths <= {3, 4} or widths == {1}):
        warnings.warn(
            ".off: face colours are not uniform across the file; dropping them.",
            stacklevel=2,
        )
        return {}, None

    if widths == {1}:
        index = np.array([int(round(v[0])) for v in values], dtype=np.int32)
        return {_COLOR_INDEX: index}, None

    # A file may spell some faces RGB and others RGBA; the short ones gain an
    # opaque alpha, filled after scaling so it stays opaque on either scale.
    colors = np.full((n_faces, max(widths)), np.nan, dtype=np.float64)
    for i, channels in enumerate(values):
        colors[i, : len(channels)] = channels
    if are_bytes:
        colors /= _BYTE_SCALE
    colors[np.isnan(colors)] = 1.0
    return {_COLORS: colors}, _BYTE_FORMAT if are_bytes else _FLOAT_FORMAT


# --------------------------------------------------------------------------
# binary read
# --------------------------------------------------------------------------


def _read_binary(buf: bytes, layout: _Layout) -> PolyData:
    """Parse the binary spelling: big-endian 32-bit ints and floats."""
    pos = 0
    if layout.dim is None:
        (dim,), pos = _take_counts(buf, pos, 1)
    else:
        dim = layout.dim
    _check_dim(dim)

    (n_verts, n_faces, _n_edges), pos = _take_counts(buf, pos, 3)
    for count, cap, what in (
        (n_verts, MAX_SAFE_VERTICES, "vertex"),
        (n_faces, MAX_SAFE_ELEMENTS, "face"),
    ):
        if count < 0:
            raise CodecError(f".off: negative {what} count {count}.")
        if count > cap:
            raise CodecError(
                f".off: {what} count {count} exceeds the safety cap {cap}."
            )

    n_normal = 3 if layout.has_normals else 0
    n_color = _BINARY_COLOR_COMPONENTS if layout.has_colors else 0
    n_tex = 2 if layout.has_texcoords else 0
    stride = dim + n_normal + n_color + n_tex

    block, pos = _take_floats(buf, pos, n_verts * stride)
    block = block.reshape(n_verts, stride)
    vertices = np.ascontiguousarray(block[:, :3], dtype=np.float64)

    attrs: dict[str, np.ndarray] = {}
    at = dim
    if n_normal:
        attrs[_NORMALS] = np.ascontiguousarray(block[:, at : at + 3], dtype=np.float64)
        at += n_normal
    if n_color:
        colors = np.ascontiguousarray(block[:, at : at + n_color], dtype=np.float64)
        at += n_color
    if n_tex:
        attrs[_TEXCOORDS] = np.ascontiguousarray(
            block[:, at : at + 2], dtype=np.float64
        )
    if n_color:
        attrs[_COLORS] = colors

    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    face_colors: list[list[float]] = []

    # Faces are variable length, so they are walked one at a time - but the
    # walk reads out of one view of the tail rather than slicing a fresh
    # buffer per field. The float view aliases the same words, which is what
    # lets a colour be read without re-interpreting the whole block.
    words = np.frombuffer(
        buf, dtype=_INT_BE, count=(len(buf) - pos) // _WORD, offset=pos
    )
    floats = words.view(_FLOAT_BE)
    total = len(words)
    at = 0

    for i in range(n_faces):
        if at >= total:
            raise CodecError(f".off: binary data truncated at face {i}.")
        n = int(words[at])
        at += 1
        if n < _MIN_FACE_SIZE:
            raise CodecError(
                f".off: face {i} has {n} vertices; OFF faces need at least"
                f" {_MIN_FACE_SIZE}."
            )
        if len(conn_list) + n > MAX_SAFE_CONN:
            raise CodecError(
                f".off: connectivity size exceeds the safety cap {MAX_SAFE_CONN}."
            )
        if at + n + 1 > total:
            raise CodecError(f".off: binary data truncated in face {i}.")
        indices = words[at : at + n]
        at += n
        bad = indices[(indices < 0) | (indices >= n_verts)]
        if bad.size:
            raise CodecError(
                f".off: face {i} references vertex {int(bad[0])}, outside"
                f" [0, {n_verts})."
            )
        conn_list.extend(indices.tolist())
        offsets_list.append(offsets_list[-1] + n)
        types_list.append(_face_type(n))

        n_comp = int(words[at])
        at += 1
        if n_comp not in (0, 1, 3, 4):
            raise CodecError(
                f".off: face {i} declares {n_comp} colour components; the"
                " binary format allows 0, 1, 3 or 4."
            )
        if at + n_comp > total:
            raise CodecError(f".off: binary data truncated in face {i} colour.")
        if n_comp:
            face_colors.append(floats[at : at + n_comp].tolist())
        at += n_comp

    # Binary colour components are floats by definition, so the 0..255 spelling
    # never applies here.
    element_attrs, _ = _face_color_attrs(face_colors, n_faces, are_bytes=False)

    global_attrs: dict[str, Any] = {}
    if n_color:
        global_attrs[_VERTEX_COLOR_FORMAT_KEY] = _FLOAT_FORMAT
    if _COLORS in element_attrs:
        global_attrs[_FACE_COLOR_FORMAT_KEY] = _FLOAT_FORMAT

    idx_dtype = np.int64 if offsets_list[-1] > np.iinfo(np.int32).max else np.int32
    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=idx_dtype),
        offsets=np.array(offsets_list, dtype=idx_dtype),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs=attrs,
        element_attrs=element_attrs,
        global_attrs=global_attrs,
    )


def _take_counts(buf: bytes, pos: int, count: int) -> tuple[list[int], int]:
    """Read int32 values as Python ints.

    Counts get multiplied by a stride and a word size to size the next read.
    Left as numpy int32 scalars those products wrap silently, and a wrapped
    product turns the truncation guard into a pass, so counts leave numpy
    behind the moment they are read.
    """
    values, pos = _take(buf, pos, count, _INT_BE, "integers")
    return values.tolist(), pos


def _take_floats(buf: bytes, pos: int, count: int) -> tuple[np.ndarray, int]:
    """Read ``count`` big-endian float32 values, refusing to run off the end."""
    return _take(buf, pos, count, _FLOAT_BE, "floats")


def _take(
    buf: bytes, pos: int, count: int, dtype: np.dtype, what: str
) -> tuple[np.ndarray, int]:
    """Slice a fixed-width binary record out of the buffer."""
    end = pos + count * _WORD
    if end > len(buf):
        raise CodecError(
            f".off: binary data truncated - needed {count} {what} at byte"
            f" {pos}, only {len(buf) - pos} bytes remain."
        )
    return np.frombuffer(buf, dtype=dtype, count=count, offset=pos), end


# --------------------------------------------------------------------------
# write helpers
# --------------------------------------------------------------------------


def _select_faces(poly: PolyData) -> list[int]:
    """Return the indices of the elements OFF can spell as a face."""
    writable: list[int] = []
    skipped: dict[str, int] = {}
    for i in range(len(poly.element_types)):
        code = int(poly.element_types[i])
        size = int(poly.offsets[i + 1]) - int(poly.offsets[i])
        # A pixel is reordered through its fourth node, so a short one is not
        # writable however forgiving the minimum face size is.
        sized = size >= _MIN_FACE_SIZE and (size == 4 or code != _PIXEL_CODE)
        if code in _WRITABLE_TYPES and sized:
            writable.append(i)
        else:
            name = ELEMENT_TYPES_INV.get(code, f"code {code}")
            skipped[name] = skipped.get(name, 0) + 1
    for name, count in sorted(skipped.items()):
        warnings.warn(
            f".off: {count} element(s) of type {name!r} are not polygonal faces;"
            " skipping.",
            stacklevel=3,
        )
    return writable


def _face_cell(poly: PolyData, i: int, n_verts: int) -> list[int]:
    """Return one element's vertex indices in OFF's boundary-walk order."""
    s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
    cell = [int(v) for v in poly.connectivity[s:e]]
    # An index the file's own vertex block cannot reach makes the OFF
    # unreadable - including by this codec's reader - so refuse to write it.
    for v in cell:
        if not 0 <= v < n_verts:
            raise CodecError(
                f".off: element {i} references vertex {v}, outside [0, {n_verts})."
            )
    if int(poly.element_types[i]) == _PIXEL_CODE:
        cell = [cell[0], cell[1], cell[3], cell[2]]
    return cell


def _vertex_channel(
    poly: PolyData, name: str, width: int | frozenset[int], n_verts: int
) -> np.ndarray | None:
    """Return a per-vertex channel when its shape fits the OFF record."""
    arr = poly.vertex_attrs.get(name)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    allowed = {width} if isinstance(width, int) else set(width)
    if arr.ndim != 2 or arr.shape[0] != n_verts or arr.shape[1] not in allowed:
        warnings.warn(
            f".off: vertex_attrs[{name!r}] has shape {arr.shape}, expected"
            f" ({n_verts}, {'|'.join(str(w) for w in sorted(allowed))});"
            " skipping it.",
            stacklevel=3,
        )
        return None
    return arr


def _face_channels(
    poly: PolyData, writable: list[int]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the per-face colour and colormap-index channels, if usable."""
    n_elems = len(poly.element_types)
    colors = poly.element_attrs.get(_COLORS)
    index = poly.element_attrs.get(_COLOR_INDEX)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if (
            colors.ndim != 2
            or colors.shape[0] != n_elems
            or colors.shape[1] not in _VALID_COLOR_COMPONENTS
        ):
            warnings.warn(
                f".off: element_attrs['colors'] has shape {colors.shape}, expected"
                f" ({n_elems}, 3|4); skipping it.",
                stacklevel=3,
            )
            colors = None
        else:
            colors = colors[writable]

    if index is not None:
        index = np.asarray(index)
        if index.ndim != 1 or index.shape[0] != n_elems:
            warnings.warn(
                f".off: element_attrs['color_index'] has shape {index.shape},"
                f" expected ({n_elems},); skipping it.",
                stacklevel=3,
            )
            index = None
        else:
            index = index[writable]

    if colors is not None and index is not None:
        warnings.warn(
            ".off: element_attrs carries both 'colors' and 'color_index';"
            " writing 'colors'.",
            stacklevel=3,
        )
        index = None
    return colors, index


def _encode_ascii(
    keyword: str,
    vertices: np.ndarray,
    cells: list[list[int]],
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    texcoords: np.ndarray | None,
    face_colors: np.ndarray | None,
    face_index: np.ndarray | None,
    float_fmt: str,
    global_attrs: dict[str, Any],
) -> str:
    """Render the ASCII spelling, one record per line."""
    vertex_bytes = global_attrs.get(_VERTEX_COLOR_FORMAT_KEY) == _BYTE_FORMAT
    face_bytes = global_attrs.get(_FACE_COLOR_FORMAT_KEY) == _BYTE_FORMAT

    lines: list[str] = [keyword, f"{len(vertices)} {len(cells)} 0"]
    for i, vertex in enumerate(vertices):
        parts = [format(float(c), float_fmt) for c in vertex]
        if normals is not None:
            parts.extend(format(float(c), float_fmt) for c in normals[i])
        if colors is not None:
            parts.extend(_color_tokens(colors[i], vertex_bytes, float_fmt))
        if texcoords is not None:
            parts.extend(format(float(c), float_fmt) for c in texcoords[i])
        lines.append(" ".join(parts))

    for i, cell in enumerate(cells):
        parts = [str(len(cell)), *(str(v) for v in cell)]
        if face_colors is not None:
            parts.extend(_color_tokens(face_colors[i], face_bytes, float_fmt))
        elif face_index is not None:
            parts.append(str(int(face_index[i])))
        lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"


def _color_tokens(channels: np.ndarray, as_bytes: bool, float_fmt: str) -> list[str]:
    """Spell one colour back in the scale the file it came from used."""
    if as_bytes:
        scaled = np.clip(np.rint(np.asarray(channels) * _BYTE_SCALE), 0, _BYTE_SCALE)
        return [str(int(c)) for c in scaled]
    # A 0..1 channel that lands on a whole number formats as "1", which any
    # reader - this one included - has to take for the 0..255 spelling. Keep
    # the point so the scale survives the round trip.
    return [
        f"{token}.0" if _is_int_token(token) else token
        for token in (format(float(c), float_fmt) for c in channels)
    ]


def _encode_binary(
    keyword: str,
    vertices: np.ndarray,
    cells: list[list[int]],
    normals: np.ndarray | None,
    colors: np.ndarray | None,
    texcoords: np.ndarray | None,
    face_colors: np.ndarray | None,
    face_index: np.ndarray | None,
) -> bytes:
    """Render the binary spelling: big-endian 32-bit ints and floats."""
    chunks: list[bytes] = [f"{keyword} BINARY\n".encode("ascii")]
    chunks.append(np.array([len(vertices), len(cells), 0], dtype=_INT_BE).tobytes())

    columns: list[np.ndarray] = [np.asarray(vertices, dtype=np.float64)]
    if normals is not None:
        columns.append(normals)
    if colors is not None:
        # Binary vertex colours are four floats; a three-channel colour is
        # written opaque rather than silently shortening the record.
        if colors.shape[1] == 3:
            colors = np.hstack(
                [colors, np.ones((colors.shape[0], 1), dtype=np.float64)]
            )
        columns.append(colors)
    if texcoords is not None:
        columns.append(texcoords)
    block = np.hstack(columns) if len(columns) > 1 else columns[0]
    chunks.append(block.astype(_FLOAT_BE).tobytes())

    for i, cell in enumerate(cells):
        chunks.append(np.array([len(cell), *cell], dtype=_INT_BE).tobytes())
        # Every face carries its colour-component count, even when that is
        # zero; a colormap index is the one-component case.
        if face_colors is not None:
            chunks.append(np.array([face_colors.shape[1]], dtype=_INT_BE).tobytes())
            chunks.append(np.asarray(face_colors[i]).astype(_FLOAT_BE).tobytes())
        elif face_index is not None:
            chunks.append(np.array([1], dtype=_INT_BE).tobytes())
            chunks.append(np.array([face_index[i]], dtype=_FLOAT_BE).tobytes())
        else:
            chunks.append(np.array([0], dtype=_INT_BE).tobytes())
    return b"".join(chunks)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _face_type(n: int) -> int:
    """Map a face's vertex count onto a polyxios element type code."""
    if n == 3:
        return _TRI_CODE
    if n == 4:
        return _QUAD_CODE
    return _POLY_CODE


def _data_lines(text: str) -> Iterator[list[str]]:
    """Yield the tokens of each meaningful line, one line at a time.

    Reading through a StringIO keeps a single line alive rather than a list of
    every line in the file, which for a large mesh is the difference between a
    few megabytes and several times the file's own size.
    """
    for line in io.StringIO(text):
        parts = line.split("#")[0].split()
        if parts:
            yield parts


def _is_int_token(token: str) -> bool:
    """Report whether a colour channel was spelled as an integer.

    OFF reads ``3 or 4 integers`` as 0..255 and ``3 or 4 floats`` as 0..1, so
    the spelling is what tells the two scales apart.
    """
    return token.lstrip("+-").isdigit()


def _checked_count(value: str, cap: int, what: str) -> int:
    """Parse a declared count, rejecting negatives and absurd values."""
    try:
        count = int(value)
    except ValueError as exc:
        raise CodecError(f".off: non-integer {what} count {value!r}.") from exc
    if count < 0:
        raise CodecError(f".off: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".off: {what} count {count} exceeds the safety cap {cap}.")
    return count
