"""ANSYS Fluent mesh codec (``.msh``, resolved by content) - read + write.

A Fluent mesh is a run of parenthesised sections, each opening with an index
that says what it holds: ``(0 "...")`` a comment, ``(2 3)`` the dimension,
``(10 ...)`` nodes, ``(12 ...)`` cells, ``(13 ...)`` faces and ``(45 ...)``
the type and name of a zone. The index is decimal, as is the zone id a
``(45 ...)`` record names; every field of a section header - zone id, range,
type - and every value of a cell or face body is spelled in hexadecimal.
A cell carries no node list of its own: the mesh is described by its faces,
each naming its nodes and the cell on either side, so a reader assembles the
cells from the faces that bound them. Node coordinates, cell types and faces
may each be spelled in ASCII or as raw binary (``2010`` / ``3010`` and their
kin), and both are read - a double-precision section with 32-bit integers,
Fluent's own, or the 64-bit ones another writer spells; the writer spells
ASCII.

A cell comes back as the tetrahedron, hexahedron, wedge or pyramid - in two
dimensions the triangle, quadrilateral or polygon - its faces close, and every
face of a zone that is not interior comes back as an element of its own,
tagged with the zone's name and linked to the cell it bounds through
``face_parent`` / ``face_index`` - see :mod:`polyxios._faces`. Each zone is a
tag, and the zone's type - ``fluid``, ``wall``, ``velocity-inlet`` - is kept
under ``global_attrs["fluent_zone_types"]`` so a write can spell it back.

``.msh`` is Gmsh's extension as well, so it is shared rather than owned and a
read is settled by content; a write there goes to Gmsh unless ``fmt="fluent"``
names this codec.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import re
from typing import Any
import warnings

import numpy as np

from polyxios._dimension import has_solid_cells, mark_2d
from polyxios._element_types import (
    ELEMENT_FACES,
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    MAX_SAFE_CONN,
    MAX_SAFE_ELEMENTS,
    MAX_SAFE_VERTICES,
    NODES_PER_ELEMENT,
)
from polyxios._faces import FACE_INDEX_KEY, FACE_PARENT_KEY
from polyxios._io import Source, read_bytes, write_text
from polyxios._tags import member_indices
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

# ``.fluent`` is not a spelling found in the wild; it exists so a caller can
# name this codec on a write, where ``.msh`` alone still means Gmsh.
EXTENSION: str = ".fluent"
# ``.msh`` is Gmsh's own extension and Fluent uses it too, so it is shared
# rather than owned: the two are told apart by what the file opens with.
SNIFF_EXTENSIONS: tuple[str, ...] = (".msh",)
# After Gmsh's 0: ``$MeshFormat`` opens a Gmsh file and nothing else, where a
# parenthesised section index is the broader of the two tests.
SNIFF_PRIORITY: int = 1

#: Where a reader keeps each zone's type - ``fluid``, ``wall``,
#: ``pressure-outlet`` - keyed by the tag the zone became.
ZONE_TYPES_KEY: str = "fluent_zone_types"

# Section indices. A binary flavour adds 2000 (single precision, 32-bit
# integers) or 3000 (double precision, 64-bit integers) to the ASCII index.
_COMMENT: int = 0
_HEADER: int = 1
_DIMENSION: int = 2
_NODES: int = 10
_CELLS: int = 12
_FACES: int = 13
_ZONE_OLD: int = 39
_ZONE: int = 45
_BINARY_SINGLE: int = 2000
_BINARY_DOUBLE: int = 3000
_FLAVOURS: tuple[int, ...] = (0, _BINARY_SINGLE, _BINARY_DOUBLE)

# The face zone types the format defines, by their boundary-condition code.
# Interior faces sit between two cells of the mesh and are implied by the
# cells; every other type is a boundary a solver would name.
_INTERIOR: int = 2
# A face born of a non-conformal interface carries its type plus this.
_NON_CONFORMAL: int = 1000
_BC_WORD: dict[int, str] = {
    2: "interior",
    3: "wall",
    4: "pressure-inlet",
    5: "pressure-outlet",
    7: "symmetry",
    8: "periodic-shadow",
    9: "pressure-far-field",
    10: "velocity-inlet",
    12: "periodic",
    14: "fan",
    20: "mass-flow-inlet",
    24: "interface",
    31: "parent",
    36: "outflow",
    37: "axis",
}
# The words a (45 ...) record spells for each code, the aliases the format
# groups under one code included.
_BC_CODE: dict[str, int] = {word: code for code, word in _BC_WORD.items()}
_BC_CODE |= {
    "inlet-vent": 4,
    "intake-fan": 4,
    "exhaust-fan": 5,
    "outlet-vent": 5,
    "porous-jump": 14,
    "radiator": 14,
}

# The words a tag name is read for, longest first so ``pressure-inlet``
# is tried before ``inlet-vent`` could shadow it; ``parent`` is no boundary
# a user names.
_BC_WORDS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted((w for w in _BC_CODE if w != "parent"), key=len, reverse=True)
)

# Cell zone types, by the code the (12 ...) header carries.
_CELL_WORD: dict[int, str] = {0: "dead", 1: "fluid", 17: "solid"}
_CELL_CODE: dict[str, int] = {word: code for code, word in _CELL_WORD.items()}
_FLUID: str = "fluid"

# The element-type code of a cell zone, and the node count the explicit
# connectivity another reader writes under it carries. 0 is a mixed zone
# whose body lists one type per cell; 7 is polyhedral, which has no fixed
# node count and is not assembled here.
_ELEM_CODE: dict[str, int] = {
    "triangle": 1,
    "tetra": 2,
    "quad": 3,
    "hexahedron": 4,
    "pyramid": 5,
    "wedge": 6,
}
_ELEM_NAME: dict[int, str] = {code: name for name, code in _ELEM_CODE.items()}
_MIXED: int = 0
_POLYHEDRAL: int = 7

# Face-type codes: 0 mixed (each record opens with its node count), 2 a line,
# 3 a triangle, 4 a quadrilateral, 5 a polygon (also count-prefixed).
_FACE_MIXED: int = 0
_FACE_POLYGON: int = 5
_FACE_WIDTH: dict[int, int] = {2: 2, 3: 3, 4: 4}
_WIDTH_FACE: dict[int, int] = {width: code for code, width in _FACE_WIDTH.items()}
_COUNTED_FACE_TYPES: frozenset[int] = frozenset({_FACE_MIXED, _FACE_POLYGON})

# What each face width reads back as, and the cells and faces written at each
# dimension.
_FACE_TYPE_BY_WIDTH: dict[int, str] = {2: "line", 3: "triangle", 4: "quad"}
_CELL_TYPES: dict[int, tuple[str, ...]] = {
    2: ("triangle", "quad", "polygon"),
    3: ("tetra", "hexahedron", "wedge", "pyramid"),
}
_SIDE_TYPES: dict[int, tuple[str, ...]] = {
    2: ("line",),
    3: ("triangle", "quad"),
}
# The edges of a planar cell, the way ELEMENT_FACES lists a solid's faces.
_CELL_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "triangle": ((0, 1), (1, 2), (2, 0)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
}

# The permutation that mirrors a cell of each type, applied to a cell whose
# faces closed it with the wrong handedness: a node ring reversed, the
# pairing of bottom and top kept.
_MIRROR: dict[str, tuple[int, ...]] = {
    "tetra": (0, 2, 1, 3),
    "hexahedron": (0, 3, 2, 1, 4, 7, 6, 5),
    "wedge": (0, 2, 1, 3, 5, 4),
    "pyramid": (0, 3, 2, 1, 4),
    "triangle": (0, 2, 1),
    "quad": (0, 3, 2, 1),
}

# The default zones a write falls back on for what no tag names.
_DEFAULT_CELL_ZONE: str = "fluid"
_DEFAULT_INTERIOR_ZONE: str = "interior"
_DEFAULT_WALL_ZONE: str = "wall"

# The characters a zone name cannot carry on a (45 ...) record: whitespace
# and parentheses end the token, a quote opens a string.
_UNSAFE_NAME_RE = re.compile(r"[\s()\"']+")

_OPEN_RE = re.compile(rb"\s*\(\s*(\d+)")
_DIMENSION_RE = re.compile(rb"\s*(\w+)\s*\)")
# A Scheme string: a backslash escapes the character after it, a quote
# among them, so a comment may carry one without ending early.
_STRING = rb'"(?:[^"\\]|\\.)*"'
_STRING_RE = re.compile(_STRING, re.DOTALL)
# A ``(a b c ...)`` header: bare tokens, and quoted names that may hold a
# parenthesis of their own.
_HEADER_RE = re.compile(rb"\s*\(((?:" + _STRING + rb'|[^()"])*)\)', re.DOTALL)
# The tokens of a (45 ...) header: a quoted name whole, or a bare word.
_ZONE_TOKEN_RE = re.compile(rb'"((?:[^"\\]|\\.)*)"|([^\s()"]+)')
# What a scan over a skipped section has to stop at.
_DELIMITER_RE = re.compile(rb'[()"]')
_END_OF_BINARY = b"End of Binary Section"
_BOM = b"\xef\xbb\xbf"

# What a file opens with when it is one of these: a comment, the header, the
# dimension or any of the mesh sections, in either flavour.
_SNIFF_RE = re.compile(
    rb'\(\s*(?:0|1|2|10|12|13|18|39|45|2010|3010|2012|3012|2013|3013)[\s("]'
)

# Entries per line when a mixed zone's cell types are written; any number is
# legal, and a bounded line keeps the file readable.
_TYPES_PER_LINE: int = 20

_INT64 = np.iinfo(np.int64)

# The class of each byte in a vectorised scan of a hexadecimal block: a
# digit's value, or one of these two. Fifteen digits is the widest token the
# scan sums in int64 without overflow; a longer one takes the per-token path.
_HEX_SEPARATOR: int = -1
_HEX_OTHER: int = -2
_HEX_MAX_DIGITS: int = 15
_HEX_CLASS = np.full(256, _HEX_OTHER, dtype=np.int8)
_HEX_CLASS[np.frombuffer(b" \t\r\n\f\v", dtype=np.uint8)] = _HEX_SEPARATOR
for _digits, _base in ((b"0123456789", 0), (b"abcdef", 10), (b"ABCDEF", 10)):
    _HEX_CLASS[np.frombuffer(_digits, dtype=np.uint8)] = np.arange(len(_digits)) + _base
_TOKEN_RE = re.compile(rb"\S+")

# The integer widths a binary cell or face section may hold, by flavour.
# A single-precision section holds 32-bit integers. A double-precision one
# holds 32-bit integers when Fluent wrote it and 64-bit ones from a writer
# that widened the integers along with the floats; nothing in the header
# says which.
_INT_WIDTHS: dict[int, tuple[np.dtype, ...]] = {
    _BINARY_SINGLE: (np.dtype("<i4"),),
    _BINARY_DOUBLE: (np.dtype("<i4"), np.dtype("<i8")),
}


def sniff(head: bytes) -> bool:
    """Report whether a file's opening bytes look like a Fluent mesh.

    Parameters
    ----------
    head
        The file's first bytes, as handed over by the registry.

    Returns
    -------
    bool
        True when the first thing in the file is a parenthesised section
        whose index this codec knows - a comment, the header, the dimension
        or a node, cell, face or zone section - in either flavour.

    Notes
    -----
    Used to resolve ``.msh``, which Gmsh owns and Fluent shares. A Gmsh file
    opens with ``$MeshFormat`` and never with a parenthesis, so the two tests
    cannot both claim a file. A byte-order mark does not hide the first
    section.
    """
    return _SNIFF_RE.match(head.removeprefix(_BOM).lstrip()) is not None


# ---------------------------------------------------------------------------
# Reading: the sections
# ---------------------------------------------------------------------------


@dataclass
class _CellZone:
    zone: int
    first: int
    last: int
    kind: int
    element_type: int
    # The explicit node lists another reader writes under a typed zone, None
    # when the header stands alone or the body only lists one type per cell
    # of a mixed zone, which the faces settle on their own.
    conn: np.ndarray | None


@dataclass
class _FaceZone:
    zone: int
    first: int
    last: int
    bc: int
    nodes: np.ndarray
    widths: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclass
class _Sections:
    """Everything one pass over the file collected, before assembly."""

    dim: int | None = None
    n_nodes: int | None = None
    n_cells: int | None = None
    n_faces: int | None = None
    nodes: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    cells: list[_CellZone] = field(default_factory=list)
    faces: list[_FaceZone] = field(default_factory=list)
    names: list[tuple[str, str, str]] = field(default_factory=list)


def _line_of(data: bytes, pos: int) -> int:
    """Return the 1-based line a byte position sits on, for an error."""
    return data.count(b"\n", 0, min(pos, len(data))) + 1


def _fail(data: bytes, pos: int, what: str) -> CodecError:
    return CodecError(f"{EXTENSION}: line {_line_of(data, pos)}: {what}")


def _skip_ws(data: bytes, pos: int) -> int:
    n = len(data)
    while pos < n and data[pos] in b" \t\r\n\f\v":
        pos += 1
    return pos


def _expect(data: bytes, pos: int, byte: bytes, what: str) -> int:
    """Return the position past ``byte``, which has to be the next token."""
    pos = _skip_ws(data, pos)
    if data[pos : pos + 1] != byte:
        found = data[pos : pos + 1].decode("latin-1") or "end of file"
        raise _fail(data, pos, f"expected {byte.decode()!r} {what}, found {found!r}.")
    return pos + 1


def _skip_balanced(data: bytes, pos: int) -> int:
    """Return the position past the group opening at ``pos``.

    Parentheses nest and a quoted string may carry either, so the scan
    counts depth and steps over strings whole, jumping from one delimiter
    to the next so the cell tree or periodic-shadow list of a large adapted
    mesh costs a regex search per parenthesis rather than a loop per byte.
    """
    pos = _expect(data, pos, b"(", "to open a section")
    depth = 1
    while True:
        m = _DELIMITER_RE.search(data, pos)
        if m is None:
            raise _fail(data, len(data), "a section opens and never closes.")
        c = m.group()
        if c == b'"':
            s = _STRING_RE.match(data, m.start())
            if s is None:
                raise _fail(data, m.start(), "a string opens and never closes.")
            pos = s.end()
            continue
        pos = m.end()
        if c == b"(":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return pos


def _skip_binary(data: bytes, pos: int, index: int) -> int:
    """Step over a binary section this codec has no layout for.

    Its size is known only to its writer, but every binary section ends in
    the same marker, so the scan runs to that rather than refusing the file.
    """
    at = data.find(_END_OF_BINARY, pos)
    if at < 0:
        raise _fail(
            data,
            pos,
            f"binary section {index} is not read here and carries no"
            f" '{_END_OF_BINARY.decode()}' marker to skip to.",
        )
    pos = _skip_ws(data, at + len(_END_OF_BINARY))
    while pos < len(data) and data[pos] in b"0123456789":
        pos += 1
    return _expect(data, pos, b")", "after the end marker")


def _close_binary(data: bytes, pos: int) -> int:
    """Return the position past the close of a binary block's section."""
    pos = _expect(data, pos, b")", "to close the binary block")
    pos = _skip_ws(data, pos)
    if data.startswith(_END_OF_BINARY, pos):
        pos = _skip_ws(data, pos + len(_END_OF_BINARY))
        while pos < len(data) and data[pos] in b"0123456789":
            pos += 1
    return _expect(data, pos, b")", "to close the section")


def _hex(data: bytes, pos: int, tok: bytes, what: str) -> int:
    """Convert one hexadecimal token, refusing what int64 cannot hold."""
    try:
        value = int(tok, 16)
    except ValueError:
        raise _fail(data, pos, f"malformed {what} {tok.decode('latin-1')!r}.") from None
    if not _INT64.min <= value <= _INT64.max:
        raise _fail(
            data, pos, f"{what} {tok.decode('latin-1')!r} is wider than 64 bits."
        )
    return value


def _hex_array(data: bytes, pos: int, toks: list[bytes], what: str) -> np.ndarray:
    """Convert a run of hexadecimal tokens to int64, naming a bad one."""
    try:
        return np.asarray([int(t, 16) for t in toks], dtype=np.int64)
    except (ValueError, OverflowError):
        pass
    return np.asarray([_hex(data, pos, t, what) for t in toks], dtype=np.int64)


def _hex_body(
    data: bytes, pos: int, body: bytes, what: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return every hexadecimal token of ``body`` as int64, and where each starts.

    Parameters
    ----------
    data, pos
        The file and the position past the block, for an error.
    body
        The bytes between the block's parentheses.
    what
        What a bad token is called in the error.

    Returns
    -------
    tuple
        ``(values, starts)``: each token's value, and the offset into
        ``body`` of its first byte.

    Notes
    -----
    The block is scanned as one byte array - each byte classed as a digit,
    a separator or neither - and every token is summed from its digits in
    place, so a face zone of millions of records costs a few array passes
    rather than a Python conversion per token. A block holding anything
    else, a sign or a token past fifteen digits, goes through the per-token
    path, which names the bad token.
    """
    cls = _HEX_CLASS[np.frombuffer(body, dtype=np.uint8)]
    is_digit = cls >= 0
    edge = np.diff(is_digit.view(np.int8), prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(edge == 1)
    lengths = np.flatnonzero(edge == -1) - starts
    if cls.min(initial=0) == _HEX_OTHER or (
        lengths.size and lengths.max() > _HEX_MAX_DIGITS
    ):
        body_pos = pos - len(body) - 1
        toks = [(m.start(), m.group()) for m in _TOKEN_RE.finditer(body)]
        values = np.asarray(
            [_hex(data, body_pos + at, tok, what) for at, tok in toks], dtype=np.int64
        )
        return values, np.asarray([at for at, _ in toks], dtype=np.int64)
    if not starts.size:
        return np.empty(0, dtype=np.int64), starts
    digits = cls[is_digit].astype(np.int64)
    first = np.cumsum(lengths) - lengths
    place = np.arange(digits.size) - np.repeat(first, lengths)
    shift = 4 * (np.repeat(lengths, lengths) - place - 1)
    return np.add.reduceat(digits << shift, first), starts


def _read_header(data: bytes, pos: int) -> tuple[list[bytes], int]:
    """Return the tokens of the ``(a b c ...)`` header at ``pos`` and the position past it."""
    m = _HEADER_RE.match(data, pos)
    if m is None:
        raise _fail(data, pos, "expected a (zone first last ...) header.")
    return m.group(1).split(), m.end()


def _ascii_body(data: bytes, pos: int) -> tuple[bytes, int]:
    """Return the bytes of the numeric block at ``pos`` and the position past its close."""
    pos = _expect(data, pos, b"(", "to open the data block")
    end = data.find(b")", pos)
    if end < 0:
        raise _fail(data, pos, "a data block opens and never closes.")
    return data[pos:end], end + 1


def _float_dtype(index: int) -> np.dtype:
    return np.dtype("<f4" if index - index % 1000 == _BINARY_SINGLE else "<f8")


def _from_some_start[T](data: bytes, pos: int, attempt: Callable[[int], T]) -> T:
    """Run ``attempt`` at each byte a binary payload may begin after its ``(``.

    Parameters
    ----------
    data, pos
        The file and the position just past the ``(`` opening the block.
    attempt
        Reads the payload from one byte offset, raising :class:`CodecError`
        when the block does not close from there.

    Returns
    -------
    T
        The result of the first offset whose block closes.

    Notes
    -----
    Fluent ends the line after the ``(`` that opens a binary block, so the
    payload begins on the next; a writer that does not begins it at once.
    A payload may itself open with a line-break byte, so the break is not
    stepped over blindly: the block is read past it first, since that is
    how Fluent spells it, and from the ``(`` itself when that fails, the
    first error standing when both do.
    """
    if data.startswith(b"\r\n", pos):
        past_break = pos + 2
    elif data.startswith(b"\n", pos):
        past_break = pos + 1
    else:
        return attempt(pos)
    try:
        return attempt(past_break)
    except CodecError as first_error:
        try:
            return attempt(pos)
        except CodecError:
            raise first_error from None


def _binary_block(
    data: bytes, pos: int, dtype: np.dtype, count: int, what: str
) -> tuple[np.ndarray, int]:
    """Return ``count`` values of ``dtype`` from the block at ``pos``."""
    pos = _expect(data, pos, b"(", "to open the binary block")
    nbytes = count * dtype.itemsize

    def at(start: int) -> tuple[np.ndarray, int]:
        if start + nbytes > len(data):
            raise _fail(
                data, start, f"the file ends inside a binary block of {count} {what}."
            )
        values = np.frombuffer(data, dtype=dtype, count=count, offset=start)
        return values, _close_binary(data, start + nbytes)

    return _from_some_start(data, pos, at)


def _own_width(values: np.ndarray) -> bool:
    """Say whether a block read narrow holds integers of that width.

    Wider integers read narrow put their high halves at every second
    position, a run of zeros no genuine block shows: a face record names a
    node, which is never 0, at each of those positions.
    """
    return values.size < 2 or bool(values[1::2].any())


def _in_some_width[T](
    index: int, attempt: Callable[[np.dtype], tuple[np.ndarray, T]]
) -> T:
    """Run ``attempt`` with each integer width a section of ``index`` may hold.

    Parameters
    ----------
    index
        The section index, whose flavour lists the widths.
    attempt
        Reads the block at one width, returning the raw values it took and
        the result to hand back; raises :class:`CodecError` when the block
        does not close at that width.

    Returns
    -------
    T
        The result of the first width whose block closes and reads as its
        own; the last width is read outright, its error standing when it
        fails too.
    """
    widths = _INT_WIDTHS[index - index % 1000]
    for dtype in widths[:-1]:
        try:
            values, result = attempt(dtype)
        except CodecError:
            continue
        if _own_width(values):
            return result
    return attempt(widths[-1])[1]


def _binary_ints(
    data: bytes, pos: int, index: int, count: int, what: str
) -> tuple[np.ndarray, int]:
    """Return ``count`` integers of the binary block at ``pos``, as int64."""

    def attempt(dtype: np.dtype) -> tuple[np.ndarray, tuple[np.ndarray, int]]:
        values, end = _binary_block(data, pos, dtype, count, what)
        return values, (values.astype(np.int64), end)

    return _in_some_width(index, attempt)


def _range(
    data: bytes, pos: int, header: list[bytes], what: str
) -> tuple[int, int, int]:
    """Return ``(zone, first, last)`` off a header, checking the range."""
    if len(header) < 4:
        raise _fail(
            data,
            pos,
            f"a {what} header carries {len(header)} field(s), expected 4 or 5.",
        )
    zone, first, last = (_hex(data, pos, t, f"{what} header field") for t in header[:3])
    if zone < 0:
        raise _fail(data, pos, f"{what} zone id {zone} is negative.")
    if zone == 0:
        if last < 0:
            raise _fail(data, pos, f"{what} count {last} is negative.")
    elif first < 1 or last < first:
        raise _fail(
            data,
            pos,
            f"{what} zone {zone} spans {first}..{last}, which is not a range from 1.",
        )
    return zone, first, last


def _checked_count(data: bytes, pos: int, count: int, cap: int, what: str) -> None:
    """Refuse a declared count no file this size can hold.

    Every node, cell and face costs at least one byte of the file - a
    coordinate, a face record, a type - so a count past the file's own size
    is a corrupt header, and refusing it here is what stops the arrays it
    would size from being allocated.
    """
    if count > cap:
        raise _fail(data, pos, f"{what} count {count} exceeds the safety cap {cap}.")
    if count > len(data):
        raise _fail(
            data,
            pos,
            f"the file declares {count} {what}s but is only {len(data)} bytes long.",
        )


def _read_nodes(data: bytes, pos: int, index: int, sections: _Sections) -> int:
    header, pos = _read_header(data, pos)
    zone, first, last = _range(data, pos, header, "node")
    _checked_count(data, pos, last, MAX_SAFE_VERTICES, "node")
    if zone == 0:
        sections.n_nodes = last
        return _expect(data, pos, b")", "to close the node declaration")
    count = last - first + 1
    n_coords = (
        _hex(data, pos, header[4], "node dimension")
        if len(header) > 4
        else sections.dim
    )
    if n_coords is None:
        raise _fail(
            data,
            pos,
            "the node header names no dimension and no (2 ...) section preceded it.",
        )
    if n_coords not in (2, 3):
        raise _fail(data, pos, f"node dimension {n_coords} is not 2 or 3.")
    if sections.dim is None:
        sections.dim = n_coords
    if index == _NODES:
        body, pos = _ascii_body(data, pos)
        try:
            coords = np.asarray(body.split(), dtype=np.float64)
        except ValueError:
            raise _fail(
                data, pos, f"node zone {zone} carries a malformed coordinate."
            ) from None
        if coords.size != count * n_coords:
            raise _fail(
                data,
                pos,
                f"node zone {zone} declares {count} node(s) of {n_coords} coordinates"
                f" but carries {coords.size} value(s).",
            )
        pos = _expect(data, pos, b")", "to close the node section")
    else:
        coords, pos = _binary_block(
            data, pos, _float_dtype(index), count * n_coords, "coordinates"
        )
        coords = coords.astype(np.float64)
    sections.nodes.append((first, last, coords.reshape(count, n_coords)))
    return pos


def _read_cells(data: bytes, pos: int, index: int, sections: _Sections) -> int:
    header, pos = _read_header(data, pos)
    zone, first, last = _range(data, pos, header, "cell")
    _checked_count(data, pos, last, MAX_SAFE_ELEMENTS, "cell")
    if zone == 0:
        sections.n_cells = last
        return _expect(data, pos, b")", "to close the cell declaration")
    count = last - first + 1
    kind = _hex(data, pos, header[3], "cell zone type")
    element_type = (
        _hex(data, pos, header[4], "cell element type") if len(header) > 4 else _MIXED
    )
    at = _skip_ws(data, pos)
    if data[at : at + 1] == b")":
        sections.cells.append(_CellZone(zone, first, last, kind, element_type, None))
        return at + 1
    if element_type == _POLYHEDRAL:
        raise _fail(
            data,
            pos,
            f"cell zone {zone} is polyhedral and carries a body, which is not read.",
        )
    if element_type == _MIXED:
        width = 1
    elif element_type in _ELEM_NAME:
        width = NODES_PER_ELEMENT[_ELEM_NAME[element_type]]
    else:
        raise _fail(
            data,
            pos,
            f"cell zone {zone} has element type {element_type}, which the format"
            " does not define.",
        )
    if index == _CELLS:
        body, pos = _ascii_body(data, pos)
        values, _ = _hex_body(data, pos, body, f"cell zone {zone} entry")
        pos = _expect(data, pos, b")", "to close the cell section")
    else:
        values, pos = _binary_ints(data, pos, index, count * width, "cell entries")
    if values.size != count * width:
        raise _fail(
            data,
            pos,
            f"cell zone {zone} declares {count} cell(s) but carries {values.size}"
            f" value(s), expected {count * width}.",
        )
    conn = None if element_type == _MIXED else values.reshape(count, width)
    sections.cells.append(_CellZone(zone, first, last, kind, element_type, conn))
    return pos


_Faces = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _counted_faces(
    values: np.ndarray, widths: np.ndarray, starts: np.ndarray
) -> _Faces:
    """Gather count-prefixed records into flat nodes + widths + sides.

    Parameters
    ----------
    values
        Every value of the block, int64.
    widths, starts
        The node count of each record and where its first node sits.
    """
    total = int(widths.sum())
    gather = np.repeat(starts, widths) + (
        np.arange(total) - np.repeat(np.cumsum(widths) - widths, widths)
    )
    return (
        values[gather],
        widths,
        values[starts + widths],
        values[starts + widths + 1],
    )


def _counted_faces_ascii(
    data: bytes, pos: int, body: bytes, count: int, zone: int
) -> _Faces:
    """Read count-prefixed face records, one per line, as flat nodes + widths.

    A record is its line: a line short of a value or long by one is refused
    by name rather than read into its neighbour, which a flat token stream
    could not tell apart. The line of each token is the number of newlines
    before it, so the records are cut without a pass over the lines.
    """
    values, starts = _hex_body(data, pos, body, f"face zone {zone} entry")
    newlines = np.flatnonzero(np.frombuffer(body, dtype=np.uint8) == 0x0A)
    per_line = np.bincount(np.searchsorted(newlines, starts))
    n_toks = per_line[per_line > 0]
    if n_toks.size != count:
        raise _fail(
            data,
            pos,
            f"face zone {zone} declares {count} face(s) but carries"
            f" {n_toks.size} record line(s); a count-prefixed record is its"
            " own line.",
        )
    if not count:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    firsts = np.cumsum(n_toks) - n_toks
    widths = values[firsts]
    bad = np.flatnonzero((widths < 2) | (n_toks != widths + 3))
    if bad.size:
        k = int(bad[0])
        raise _fail(
            data,
            pos - len(body) - 1 + int(starts[firsts[k]]),
            f"face zone {zone}: a record opening with {int(widths[k])} node(s)"
            f" carries {int(n_toks[k]) - 1} value(s), expected {int(widths[k]) + 2}.",
        )
    return _counted_faces(values, widths, firsts + 1)


# How many records a walk over count-prefixed binary faces checks at once
# for one shared width, and the most it grows that window to.
_PROBE_MIN: int = 64
_PROBE_MAX: int = 1 << 20
# A run shorter than this says the widths are changing often enough that
# probing costs more than stepping one record at a time for a while.
_SHORT_RUN: int = 8


def _counted_layout(
    values: np.ndarray, count: int, fail: Callable[[str], CodecError]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(widths, starts, used)`` over ``count`` count-prefixed records.

    Parameters
    ----------
    values
        The block's values, a zero-copy view of any integer width.
    count
        How many records the zone declares.
    fail
        Called with a message to build the error for a truncated block or a
        record with fewer than two nodes.

    Notes
    -----
    Each record opens with its own node count, so where the next one starts
    is known only once the last is read. A run of records sharing one width
    is found in one strided comparison, with a window that doubles while it
    keeps matching, so a zone of one width - what Fluent mostly writes
    under the mixed type - costs a handful of array operations; a zone whose
    width keeps changing falls back to stepping one record at a time.
    """
    n = values.size
    widths = np.empty(count, dtype=np.int64)
    starts = np.empty(count, dtype=np.int64)
    at = 0
    k = 0
    probe = _PROBE_MIN
    single_steps = 0
    while k < count:
        if at >= n:
            raise fail("the file ends inside")
        w = int(values[at])
        if w < 2:
            raise fail(f"a record opens with {w} node(s) in")
        stride = w + 3
        if single_steps:
            single_steps -= 1
            widths[k] = w
            starts[k] = at + 1
            k += 1
            at += stride
            continue
        room = min(count - k, probe, (n - at - 1) // stride + 1)
        same = values[at : at + room * stride : stride] == w
        run = room if same.all() else int(same.argmin())
        if run == room and room == probe:
            probe = min(probe * 2, _PROBE_MAX)
        elif run < _SHORT_RUN:
            probe = _PROBE_MIN
            single_steps = _PROBE_MIN
        widths[k : k + run] = w
        starts[k : k + run] = at + 1 + stride * np.arange(run)
        k += run
        at += run * stride
    if at > n:
        raise fail("the file ends inside")
    return widths, starts, at


def _counted_faces_binary(
    data: bytes, pos: int, index: int, count: int, zone: int
) -> tuple[_Faces, int]:
    """Read count-prefixed binary face records, whose total size only they know."""
    pos = _expect(data, pos, b"(", "to open the binary block")

    def fail(what: str) -> CodecError:
        return _fail(data, pos, f"{what} face zone {zone}'s binary block.")

    def attempt(dtype: np.dtype) -> tuple[np.ndarray, tuple[_Faces, int]]:
        def at(start: int) -> tuple[np.ndarray, tuple[_Faces, int]]:
            available = (len(data) - start) // dtype.itemsize
            raw = np.frombuffer(data, dtype=dtype, count=available, offset=start)
            widths, starts, used = _counted_layout(raw, count, fail)
            end = _close_binary(data, start + used * dtype.itemsize)
            taken = raw[:used].astype(np.int64)
            return taken, (_counted_faces(taken, widths, starts), end)

        return _from_some_start(data, pos, at)

    return _in_some_width(index, attempt)


def _uniform_faces(rows: np.ndarray, width: int) -> _Faces:
    return (
        rows[:, :width].ravel(),
        np.full(rows.shape[0], width, dtype=np.int64),
        rows[:, width].copy(),
        rows[:, width + 1].copy(),
    )


def _read_faces(data: bytes, pos: int, index: int, sections: _Sections) -> int:
    header, pos = _read_header(data, pos)
    zone, first, last = _range(data, pos, header, "face")
    _checked_count(data, pos, last, MAX_SAFE_ELEMENTS, "face")
    if zone == 0:
        sections.n_faces = last
        return _expect(data, pos, b")", "to close the face declaration")
    count = last - first + 1
    bc = _hex(data, pos, header[3], "face boundary type")
    if bc >= _NON_CONFORMAL:
        bc -= _NON_CONFORMAL
    face_type = (
        _hex(data, pos, header[4], "face type") if len(header) > 4 else _FACE_MIXED
    )
    if face_type not in _FACE_WIDTH and face_type not in _COUNTED_FACE_TYPES:
        raise _fail(
            data,
            pos,
            f"face zone {zone} has face type {face_type}, which the format does not define.",
        )
    counted = face_type in _COUNTED_FACE_TYPES
    if index == _FACES:
        body, pos = _ascii_body(data, pos)
        if counted:
            faces = _counted_faces_ascii(data, pos, body, count, zone)
        else:
            width = _FACE_WIDTH[face_type]
            values, _ = _hex_body(data, pos, body, f"face zone {zone} entry")
            if values.size != count * (width + 2):
                raise _fail(
                    data,
                    pos,
                    f"face zone {zone} declares {count} face(s) of {width} node(s)"
                    f" but carries {values.size} value(s), expected {count * (width + 2)}.",
                )
            faces = _uniform_faces(values.reshape(count, width + 2), width)
        pos = _expect(data, pos, b")", "to close the face section")
    elif counted:
        faces, pos = _counted_faces_binary(data, pos, index, count, zone)
    else:
        width = _FACE_WIDTH[face_type]
        values, pos = _binary_ints(
            data, pos, index, count * (width + 2), "face entries"
        )
        faces = _uniform_faces(values.reshape(count, width + 2), width)
    sections.faces.append(_FaceZone(zone, first, last, bc, *faces))
    return pos


def _read_zone_name(data: bytes, pos: int, start: int, sections: _Sections) -> int:
    m = _HEADER_RE.match(data, pos)
    if m is None:
        raise _fail(data, pos, "expected a (zone type name) header.")
    header = [
        (quoted or bare).decode("utf-8", errors="replace")
        for quoted, bare in _ZONE_TOKEN_RE.findall(m.group(1))
    ]
    if len(header) >= 2:
        word, name = header[1], header[2] if len(header) > 2 else header[1]
        sections.names.append((header[0], word, name))
    return _skip_balanced(data, start)


def _parse(data: bytes) -> _Sections:
    """Walk every section of the file into a :class:`_Sections`."""
    sections = _Sections()
    pos = 0
    n = len(data)
    while True:
        pos = _skip_ws(data, pos)
        if pos >= n:
            return sections
        m = _OPEN_RE.match(data, pos)
        if m is None:
            raise _fail(data, pos, "expected a '(' opening a section.")
        start = pos
        index = int(m.group(1))
        pos = m.end()
        base, flavour = index % 1000, index - index % 1000
        if index == _DIMENSION:
            m = _DIMENSION_RE.match(data, pos)
            if m is None:
                raise _fail(data, pos, "the dimension section is not (2 <dimension>).")
            pos = m.end()
            dim = _hex(data, pos, m.group(1), "dimension")
            if dim not in (2, 3):
                raise _fail(data, pos, f"dimension {dim} is not 2 or 3.")
            sections.dim = dim
        elif index in (_COMMENT, _HEADER):
            pos = _skip_balanced(data, start)
        elif flavour in _FLAVOURS and base == _NODES:
            pos = _read_nodes(data, pos, index, sections)
        elif flavour in _FLAVOURS and base == _CELLS:
            pos = _read_cells(data, pos, index, sections)
        elif flavour in _FLAVOURS and base == _FACES:
            pos = _read_faces(data, pos, index, sections)
        elif index in (_ZONE, _ZONE_OLD):
            pos = _read_zone_name(data, pos, start, sections)
        elif index >= _BINARY_SINGLE:
            pos = _skip_binary(data, pos, index)
        else:
            pos = _skip_balanced(data, start)


# ---------------------------------------------------------------------------
# Reading: assembling cells from faces
# ---------------------------------------------------------------------------


def _distinct(rows: np.ndarray) -> np.ndarray:
    """Return how many distinct values each row holds."""
    ranked = np.sort(rows, axis=1)
    return 1 + (ranked[:, 1:] != ranked[:, :-1]).sum(axis=1)


def _in_ring(nodes: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Say, per entry of ``nodes`` (m, k), whether it sits in ``ring`` (m, w)."""
    return (nodes[:, :, None] == ring[:, None, :]).any(axis=2)


def _other(nodes: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Return the one node per row of ``nodes`` outside ``ring``, -1 if not exactly one."""
    outside = ~_in_ring(nodes, ring)
    picked = np.where(outside, nodes, -1).max(axis=1)
    return np.where(outside.sum(axis=1) == 1, picked, -1)


def _partners(base: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    """Return the node above each base node, found through the lateral faces.

    Parameters
    ----------
    base
        The base ring of each cell, shape ``(m, w)``.
    lateral
        The quadrilateral faces of each cell other than the base, shape
        ``(m, k, 4)``; the face opposite the base may be among them, since it
        holds no base node and is never chosen.

    Returns
    -------
    numpy.ndarray
        Shape ``(m, w)``: for each base node, the neighbour it has in a
        lateral face that is not itself in the base, or -1 when no lateral
        face gives one.
    """
    m, w = base.shape
    rows = np.arange(m)
    top = np.full((m, w), -1, dtype=np.int64)
    for k in range(w):
        p = base[:, k]
        found = np.zeros(m, dtype=bool)
        for j in range(lateral.shape[1]):
            ring = lateral[:, j, :]
            pos = ring == p[:, None]
            has = pos.any(axis=1) & ~found
            if not has.any():
                continue
            i = pos.argmax(axis=1)
            before = ring[rows, (i - 1) % 4]
            after = ring[rows, (i + 1) % 4]
            before_in = (before[:, None] == base).any(axis=1)
            after_in = (after[:, None] == base).any(axis=1)
            ok = has & (before_in != after_in)
            top[ok, k] = np.where(before_in, after, before)[ok]
            found |= ok
    return top


def _signed_measure(conn: np.ndarray, vertices: np.ndarray, name: str) -> np.ndarray:
    """Return the signed volume (or area, in the plane) of each cell.

    The divergence theorem over the faces ``ELEMENT_FACES`` lists, which are
    outward for a cell of the right handedness, so a cell its faces closed
    the other way round answers negative and is mirrored by the caller.
    """
    pts = vertices[conn]
    if name not in ELEMENT_FACES:
        x, y = pts[:, :, 0], pts[:, :, 1]
        return 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(
            axis=1
        )
    volume = np.zeros(conn.shape[0], dtype=np.float64)
    for ring in ELEMENT_FACES[name]:
        face = pts[:, ring, :]
        for i in range(1, len(ring) - 1):
            volume += np.einsum(
                "ij,ij->i", face[:, 0], np.cross(face[:, i], face[:, i + 1])
            )
    return volume / 6.0


def _oriented(conn: np.ndarray, vertices: np.ndarray, name: str) -> np.ndarray:
    """Return ``conn`` with every cell of negative measure mirrored."""
    if conn.shape[0] == 0:
        return conn
    flipped = _signed_measure(conn, vertices, name) < 0
    if flipped.any():
        conn = conn.copy()
        conn[flipped] = conn[flipped][:, _MIRROR[name]]
    return conn


def _assemble_tets(rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = rings[:, 0]
    apex = _other(rings[:, 1], base)
    conn = np.column_stack([base, apex])
    return conn, (apex >= 0) & (_distinct(rings.reshape(rings.shape[0], -1)) == 4)


def _assemble_hexes(rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = rings.shape[0]
    base = rings[:, 0]
    shared = _in_ring(rings.reshape(m, -1), base).reshape(m, 6, 4).sum(axis=2)
    closes = ((shared[:, 1:] == 2).sum(axis=1) == 4) & (
        (shared[:, 1:] == 0).sum(axis=1) == 1
    )
    top = _partners(base, rings[:, 1:])
    conn = np.hstack([base, top])
    return conn, closes & (top >= 0).all(axis=1) & (_distinct(conn) == 8)


def _assemble_wedges(
    tris: np.ndarray, quads: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base = tris[:, 0]
    top = _partners(base, quads)
    conn = np.hstack([base, top])
    valid = (top >= 0).all(axis=1) & (_distinct(conn) == 6)
    valid &= (np.sort(top, axis=1) == np.sort(tris[:, 1], axis=1)).all(axis=1)
    return conn, valid


def _assemble_pyramids(
    quad: np.ndarray, tris: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    apex = _other(tris[:, 0], quad)
    conn = np.column_stack([quad, apex])
    valid = (apex >= 0) & (_distinct(conn) == 5)
    valid &= _in_ring(tris.reshape(tris.shape[0], -1), conn).all(axis=1)
    return conn, valid


def _assemble_triangles(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = edges[:, 0]
    third = _other(edges[:, 1], first)
    conn = np.column_stack([first, third])
    return conn, (third >= 0) & (_distinct(edges.reshape(edges.shape[0], -1)) == 3)


def _has_edge(edges: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Say whether each cell's edge list holds the edge ``(a, b)``, either way round."""
    pair = np.sort(np.column_stack([a, b]), axis=1)
    return (np.sort(edges, axis=2) == pair[:, None, :]).all(axis=2).any(axis=1)


def _assemble_quads(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = edges.shape[0]
    rows = np.arange(m)
    a, b = edges[:, 0, 0], edges[:, 0, 1]
    c = np.full(m, -1, dtype=np.int64)
    found = np.zeros(m, dtype=bool)
    for j in range(1, 4):
        ring = edges[:, j, :]
        pos = ring == b[:, None]
        has = pos.any(axis=1) & ~found
        c[has] = ring[rows, 1 - pos.argmax(axis=1)][has]
        found |= has
    flat = edges.reshape(m, -1)
    d = np.where(_in_ring(flat, np.column_stack([a, b, c])), -1, flat).max(axis=1)
    conn = np.column_stack([a, b, c, d])
    valid = (c >= 0) & (d >= 0) & (_distinct(flat) == 4) & (_distinct(conn) == 4)
    valid &= _has_edge(edges, c, d) & _has_edge(edges, d, a)
    return conn, valid


def _chain(edges: list[tuple[int, int]]) -> list[int] | None:
    """Order a polygon's edges into one ring, or None when they close none."""
    nxt: dict[int, list[int]] = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    if len(nxt) != len(edges) or any(len(v) != 2 for v in nxt.values()):
        return None
    start = edges[0][0]
    ring = [start]
    prev, cur = start, edges[0][1]
    while cur != start:
        ring.append(cur)
        step = nxt[cur]
        prev, cur = cur, step[1] if step[0] == prev else step[0]
        if len(ring) > len(edges):
            return None
    return ring if len(ring) == len(edges) else None


def _assemble(
    dim: int,
    n_cells: int,
    vertices: np.ndarray,
    face_nodes: np.ndarray,
    face_offsets: np.ndarray,
    face_widths: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    explicit: list[tuple[np.ndarray, str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble every cell from the faces that bound it.

    Parameters
    ----------
    dim
        2 or 3.
    n_cells
        How many cells the file declares.
    vertices
        The node coordinates, for the handedness of each assembled cell.
    face_nodes, face_offsets, face_widths, left, right
        Every face of the file: its nodes as mesh indices, and the 1-based
        cell on either side, 0 for none.
    explicit
        ``(cell ids, type name, connectivity)`` for the cells a typed zone
        spelled outright, which are taken as they are.

    Returns
    -------
    tuple
        ``(kept ids, types, offsets, connectivity)`` over the cells that
        could be assembled, in cell-id order.
    """
    sides = np.concatenate([left, right])
    faces = np.concatenate([np.arange(left.size), np.arange(right.size)])
    keep = sides > 0
    sides, faces = sides[keep] - 1, faces[keep]
    order = np.argsort(sides, kind="stable")
    cell_faces = faces[order]
    counts = np.bincount(sides, minlength=n_cells)
    cf_offsets = np.concatenate([[0], np.cumsum(counts)])

    types = np.full(n_cells, -1, dtype=np.int64)
    widths = np.zeros(n_cells, dtype=np.int64)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    loose: dict[int, list[int]] = {}
    described = np.zeros(n_cells, dtype=bool)

    def take(ids: np.ndarray, name: str, conn: np.ndarray) -> None:
        if ids.size:
            types[ids] = ELEMENT_TYPES[name]
            widths[ids] = conn.shape[1]
            blocks.append((ids, conn))

    for ids, name, conn in explicit:
        take(ids, name, conn)
        described[ids] = True

    def width_count(width: int) -> np.ndarray:
        return np.bincount(
            sides, weights=face_widths[faces] == width, minlength=n_cells
        ).astype(np.int64)

    def group(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask & ~described)

    def faces_of(ids: np.ndarray, count: int) -> np.ndarray:
        return cell_faces[cf_offsets[ids][:, None] + np.arange(count)]

    def rings(picked: np.ndarray, width: int) -> np.ndarray:
        return face_nodes[face_offsets[picked][..., None] + np.arange(width)]

    if dim == 3:
        n3, n4 = width_count(3), width_count(4)
        tets = group((counts == 4) & (n3 == 4))
        if tets.size:
            conn, valid = _assemble_tets(rings(faces_of(tets, 4), 3))
            take(tets[valid], "tetra", _oriented(conn[valid], vertices, "tetra"))
        hexes = group((counts == 6) & (n4 == 6))
        if hexes.size:
            conn, valid = _assemble_hexes(rings(faces_of(hexes, 6), 4))
            take(
                hexes[valid],
                "hexahedron",
                _oriented(conn[valid], vertices, "hexahedron"),
            )
        wedges = group((counts == 5) & (n3 == 2) & (n4 == 3))
        if wedges.size:
            picked = faces_of(wedges, 5)
            is_tri = face_widths[picked] == 3
            conn, valid = _assemble_wedges(
                rings(picked[is_tri].reshape(-1, 2), 3),
                rings(picked[~is_tri].reshape(-1, 3), 4),
            )
            take(wedges[valid], "wedge", _oriented(conn[valid], vertices, "wedge"))
        pyramids = group((counts == 5) & (n3 == 4) & (n4 == 1))
        if pyramids.size:
            picked = faces_of(pyramids, 5)
            is_tri = face_widths[picked] == 3
            conn, valid = _assemble_pyramids(
                rings(picked[~is_tri].reshape(-1), 4),
                rings(picked[is_tri].reshape(-1, 4), 3),
            )
            take(
                pyramids[valid], "pyramid", _oriented(conn[valid], vertices, "pyramid")
            )
    else:
        n2 = width_count(2)
        tris = group((counts == 3) & (n2 == 3))
        if tris.size:
            conn, valid = _assemble_triangles(rings(faces_of(tris, 3), 2))
            take(tris[valid], "triangle", _oriented(conn[valid], vertices, "triangle"))
        quads = group((counts == 4) & (n2 == 4))
        if quads.size:
            conn, valid = _assemble_quads(rings(faces_of(quads, 4), 2))
            take(quads[valid], "quad", _oriented(conn[valid], vertices, "quad"))
        for cid in group((counts >= 5) & (n2 == counts)).tolist():
            picked = cell_faces[cf_offsets[cid] : cf_offsets[cid + 1]]
            starts = face_offsets[picked]
            ring = _chain(
                list(zip(face_nodes[starts].tolist(), face_nodes[starts + 1].tolist()))
            )
            if ring is None:
                continue
            if _signed_measure(np.asarray([ring]), vertices, "polygon")[0] < 0:
                ring = [ring[0], *ring[:0:-1]]
            types[cid] = ELEMENT_TYPES["polygon"]
            widths[cid] = len(ring)
            loose[cid] = ring

    offsets = np.concatenate([[0], np.cumsum(widths)])
    total = int(offsets[-1])
    if total > MAX_SAFE_CONN:
        raise CodecError(
            f"{EXTENSION}: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
        )
    conn = np.empty(total, dtype=np.int64)
    for ids, block in blocks:
        conn[offsets[ids][:, None] + np.arange(block.shape[1])] = block
    for cid, ring in loose.items():
        conn[offsets[cid] : offsets[cid + 1]] = ring

    kept = np.flatnonzero(types >= 0)
    kept_widths = widths[kept]
    kept_offsets = np.concatenate([[0], np.cumsum(kept_widths)])
    gather = np.repeat(offsets[kept], kept_widths) + (
        np.arange(int(kept_offsets[-1])) - np.repeat(kept_offsets[:-1], kept_widths)
    )
    return kept, types[kept], kept_offsets, conn[gather]


def _local_faces(
    parent_types: np.ndarray,
    parent_offsets: np.ndarray,
    conn: np.ndarray,
    parents: np.ndarray,
    face_rings: np.ndarray,
    face_widths: np.ndarray,
) -> np.ndarray:
    """Return which of its parent's faces each face is, -1 where it is none.

    Parameters
    ----------
    parent_types, parent_offsets, conn
        The assembled cells.
    parents
        The parent cell of each face, as an index into the cells, -1 for none.
    face_rings
        The faces' nodes, padded to four columns with -1.
    face_widths
        How many nodes each face carries.
    """
    local = np.full(parents.size, -1, dtype=np.int64)
    live = np.flatnonzero(parents >= 0)
    if not live.size:
        return local
    types = parent_types[parents[live]]
    for code in np.unique(types).tolist():
        faces = ELEMENT_FACES.get(ELEMENT_TYPES_INV.get(code, ""))
        if faces is None:
            continue
        of_type = live[types == code]
        for k, corners in enumerate(faces):
            width = len(corners)
            same = of_type[(face_widths[of_type] == width) & (local[of_type] < 0)]
            if not same.size:
                continue
            held = np.sort(
                conn[parent_offsets[parents[same]][:, None] + np.asarray(corners)],
                axis=1,
            )
            wanted = np.sort(face_rings[same, :width], axis=1)
            local[same[(held == wanted).all(axis=1)]] = k
    return local


def _zone_id(text: str, known: set[int]) -> int | None:
    """Resolve a (45 ...) zone id, which the format spells in decimal.

    A file from another writer may spell it in hexadecimal like the section
    headers, so a decimal reading that names no zone is retried as one.
    Only ASCII digits count: ``str.isdigit`` also passes superscripts and
    circled numerals, which ``int`` then refuses.
    """
    decimal = int(text) if text.isascii() and text.isdigit() else None
    if decimal is not None and decimal in known:
        return decimal
    try:
        hexadecimal = int(text, 16)
    except ValueError:
        return decimal
    return hexadecimal if hexadecimal in known else decimal


def _unique_name(name: str, used: set[str]) -> str:
    candidate, k = name, 1
    while candidate in used:
        k += 1
        candidate = f"{name}-{k}"
    used.add(candidate)
    return candidate


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a Fluent mesh and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.msh`` file, or an open file object.
    lazy
        Ignored; the file is loaded eagerly.

    Returns
    -------
    PolyData
        The cells the faces close, in cell-id order, followed by every face
        of a zone that is not interior - a boundary, a periodic pair, a fan -
        as an element of its own. A two-dimensional file comes back with a
        zero third coordinate and ``global_attrs["was_2d"]``.

    Raises
    ------
    CodecError
        On a file that opens with anything but a section, a section that
        never closes, a malformed number, a node or face zone whose body
        does not match its declared range, a face naming a node or cell the
        file does not hold, two zones claiming one range, a node never given
        coordinates, a dimension other than 2 or 3, a cell zone spelled
        polyhedral with a body, or a declared count past the safety caps or
        the file's own size.

    Notes
    -----
    A cell is assembled from its faces as the tetrahedron, hexahedron, wedge
    or pyramid - in the plane the triangle, quadrilateral or polygon - they
    close, whatever handedness the faces came in; a cell whose faces close
    none of those, a polyhedron or a hanging-node refinement among them, is
    dropped with a warning. A typed cell zone carrying explicit node lists,
    which another reader writes, is read as those lists.

    Each zone becomes an ``element_tags`` entry named as the ``(45 ...)``
    record names it, or ``<type>-<id>`` when no record does, and the zone's
    type word is kept under ``global_attrs["fluent_zone_types"]``. A face
    read as an element carries ``element_attrs["face_parent"]`` and
    ``["face_index"]`` naming the cell it bounds and which of that cell's
    faces it is, the way an Abaqus surface does; in the plane, where an
    edge has no such numbering, the columns are left out.
    """
    if lazy:
        warnings.warn(
            f"{EXTENSION}: lazy=True is not supported; loading eagerly.",
            stacklevel=2,
        )
    data = read_bytes(path).removeprefix(_BOM)
    sections = _parse(data)

    if not sections.nodes and sections.n_nodes is None:
        raise CodecError(f"{EXTENSION}: no node section found; not a Fluent mesh.")
    dim = sections.dim
    if dim is None:
        raise CodecError(f"{EXTENSION}: the file names no dimension.")

    n_nodes = sections.n_nodes
    if n_nodes is None:
        n_nodes = max((last for _, last, _ in sections.nodes), default=0)
    vertices = np.zeros((n_nodes, 3), dtype=np.float64)
    given = np.zeros(n_nodes, dtype=bool)
    for first, last, block in sections.nodes:
        if last > n_nodes:
            raise CodecError(
                f"{EXTENSION}: a node zone runs to {last}, past the {n_nodes}"
                " node(s) declared."
            )
        if given[first - 1 : last].any():
            raise CodecError(f"{EXTENSION}: node zones overlap at {first}..{last}.")
        vertices[first - 1 : last, : block.shape[1]] = block
        given[first - 1 : last] = True
    if not given.all():
        missing = int(np.flatnonzero(~given)[0]) + 1
        raise CodecError(
            f"{EXTENSION}: node {missing} is declared but never given coordinates."
        )

    # Faces, every zone concatenated in file order.
    if sections.faces:
        face_nodes = np.concatenate([z.nodes for z in sections.faces])
        face_widths = np.concatenate([z.widths for z in sections.faces])
        left = np.concatenate([z.left for z in sections.faces])
        right = np.concatenate([z.right for z in sections.faces])
    else:
        face_nodes = face_widths = left = right = np.empty(0, dtype=np.int64)
    n_faces = int(face_widths.size)
    face_offsets = np.concatenate([[0], np.cumsum(face_widths)])
    face_zone = np.zeros(n_faces, dtype=np.int64)
    face_bc = np.full(n_faces, _INTERIOR, dtype=np.int64)
    declared_faces = n_faces if sections.n_faces is None else sections.n_faces
    seen = np.zeros(declared_faces, dtype=bool)
    at = 0
    for z in sections.faces:
        count = z.last - z.first + 1
        if z.last > declared_faces or seen[z.first - 1 : z.last].any():
            raise CodecError(
                f"{EXTENSION}: face zone {z.zone} spans {z.first}..{z.last}, which"
                f" runs past the {declared_faces} face(s) declared or into another zone."
            )
        seen[z.first - 1 : z.last] = True
        face_zone[at : at + count] = z.zone
        face_bc[at : at + count] = z.bc
        at += count
    if face_nodes.size and (face_nodes.min() < 1 or face_nodes.max() > n_nodes):
        bad = int(np.flatnonzero((face_nodes < 1) | (face_nodes > n_nodes))[0])
        which = int(np.searchsorted(face_offsets, bad, side="right"))
        raise CodecError(
            f"{EXTENSION}: face {which} names node {int(face_nodes[bad])},"
            f" outside 1..{n_nodes}."
        )
    face_nodes = face_nodes - 1

    # Cells: how many there are, and which zone each sits in.
    n_cells = sections.n_cells
    if n_cells is None:
        n_cells = max(
            [z.last for z in sections.cells]
            + ([int(max(left.max(), right.max()))] if n_faces else []),
            default=0,
        )
    if n_faces and (
        min(left.min(), right.min()) < 0 or max(left.max(), right.max()) > n_cells
    ):
        raise CodecError(
            f"{EXTENSION}: a face names a cell outside 0..{n_cells}, the {n_cells}"
            " cell(s) declared."
        )
    cell_zone = np.zeros(n_cells, dtype=np.int64)
    cell_kind: dict[int, int] = {}
    explicit: list[tuple[np.ndarray, str, np.ndarray]] = []
    for z in sections.cells:
        if z.last > n_cells or (cell_zone[z.first - 1 : z.last] != 0).any():
            raise CodecError(
                f"{EXTENSION}: cell zone {z.zone} spans {z.first}..{z.last}, which"
                f" runs past the {n_cells} cell(s) declared or into another zone."
            )
        cell_zone[z.first - 1 : z.last] = z.zone
        cell_kind[z.zone] = z.kind
        if z.conn is not None:
            rows = z.conn
            if rows.min() < 1 or rows.max() > n_nodes:
                raise CodecError(
                    f"{EXTENSION}: cell zone {z.zone} names a node outside 1..{n_nodes}."
                )
            name = _ELEM_NAME[z.element_type]
            rows = rows - 1
            if name in _CELL_TYPES[dim]:
                rows = _oriented(rows, vertices, name)
            explicit.append((np.arange(z.first - 1, z.last), name, rows))

    kept, types, offsets, conn = _assemble(
        dim,
        n_cells,
        vertices,
        face_nodes,
        face_offsets,
        face_widths,
        left,
        right,
        explicit,
    )
    n_kept = kept.size
    dropped = n_cells - n_kept
    cell_index = np.full(n_cells, -1, dtype=np.int64)
    cell_index[kept] = np.arange(n_kept)

    # Faces handed back as elements: every one that is not interior.
    emitted = np.flatnonzero((face_bc != _INTERIOR) | (left == 0) | (right == 0))
    n_emitted = emitted.size
    if n_kept + n_emitted > MAX_SAFE_ELEMENTS:
        raise CodecError(
            f"{EXTENSION}: element count {n_kept + n_emitted} exceeds the safety"
            f" cap {MAX_SAFE_ELEMENTS}."
        )
    e_widths = face_widths[emitted]
    e_offsets = np.concatenate([[0], np.cumsum(e_widths)])
    gather = np.repeat(face_offsets[emitted], e_widths) + (
        np.arange(int(e_offsets[-1])) - np.repeat(e_offsets[:-1], e_widths)
    )
    e_conn = face_nodes[gather]
    e_types = np.full(n_emitted, ELEMENT_TYPES["polygon"], dtype=np.int64)
    for width, name in _FACE_TYPE_BY_WIDTH.items():
        e_types[e_widths == width] = ELEMENT_TYPES[name]

    total = int(offsets[-1] + e_offsets[-1])
    if total > MAX_SAFE_CONN:
        raise CodecError(
            f"{EXTENSION}: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
        )
    idx_dtype = np.int64 if total > np.iinfo(np.int32).max else np.int32
    all_types = np.concatenate([types, e_types]).astype(np.uint8)
    all_offsets = np.concatenate([offsets, offsets[-1] + e_offsets[1:]]).astype(
        idx_dtype
    )
    all_conn = np.concatenate([conn, e_conn]).astype(idx_dtype)

    element_attrs: dict[str, np.ndarray] = {}
    if n_emitted and n_kept and dim == 3:
        parent_ids = np.where(left[emitted] > 0, left[emitted], right[emitted]) - 1
        parents = np.where(parent_ids >= 0, cell_index[np.maximum(parent_ids, 0)], -1)
        padded = np.full((n_emitted, 4), -1, dtype=np.int64)
        for width in _FACE_WIDTH.values():
            rows = np.flatnonzero(e_widths == width)
            if rows.size:
                padded[rows, :width] = e_conn[
                    e_offsets[rows][:, None] + np.arange(width)
                ]
        local = _local_faces(types, offsets, conn, parents, padded, e_widths)
        if (local >= 0).any():
            parent_col = np.full(n_kept + n_emitted, -1, dtype=np.int32)
            index_col = np.full(n_kept + n_emitted, -1, dtype=np.int32)
            parent_col[n_kept:] = np.where(local >= 0, parents, -1)
            index_col[n_kept:] = local
            element_attrs = {FACE_PARENT_KEY: parent_col, FACE_INDEX_KEY: index_col}

    # Zones become tags, named by the (45 ...) records where there are any.
    zone_ids = set(np.unique(cell_zone[cell_zone > 0]).tolist()) | set(
        np.unique(face_zone[emitted]).tolist()
    )
    declared: dict[int, tuple[str, str]] = {}
    for token, word, name in sections.names:
        zid = _zone_id(token, zone_ids)
        if zid is not None:
            declared[zid] = (word, name)
    used: set[str] = set()
    element_tags: dict[str, np.ndarray] = {}
    zone_types: dict[str, str] = {}
    emitted_zone = face_zone[emitted]
    emitted_bc = face_bc[emitted]
    for zid in sorted(zone_ids):
        member_cells = np.flatnonzero((cell_zone == zid) & (cell_index >= 0))
        member_faces = np.flatnonzero(emitted_zone == zid)
        if not member_cells.size and not member_faces.size:
            continue
        if zid in declared:
            word, name = declared[zid]
        elif member_cells.size:
            word = _CELL_WORD.get(cell_kind.get(zid, 1), _FLUID)
            name = f"{word}-{zid}"
        else:
            bc = int(emitted_bc[member_faces[0]])
            word = _BC_WORD.get(bc, f"bc-{bc}")
            name = f"{word}-{zid}"
        name = _unique_name(name, used)
        members = np.concatenate([cell_index[member_cells], n_kept + member_faces])
        element_tags[name] = members.astype(np.int32)
        zone_types[name] = word

    if dropped:
        warnings.warn(
            f"{EXTENSION}: {dropped} cell(s) could not be assembled from their"
            " faces - a polyhedron, a hanging-node refinement or a face list"
            " that closes no cell - and were dropped; a zone naming one leaves"
            " it out.",
            stacklevel=2,
        )

    global_attrs: dict[str, Any] = dict(mark_2d(dim))
    if zone_types:
        global_attrs[ZONE_TYPES_KEY] = zone_types
    return PolyData(
        vertices=vertices,
        connectivity=all_conn,
        offsets=all_offsets,
        element_types=all_types,
        element_attrs=element_attrs,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _record_safe(name: str) -> str:
    """Return ``name`` with what cannot sit on a (45 ...) record replaced by ``_``."""
    return _UNSAFE_NAME_RE.sub("_", str(name).strip()) or "zone"


def _safe_name(name: str, used: set[str], renamed: list[tuple[str, str]]) -> str:
    """Return a name that survives its own record, unique against ``used``."""
    safe = _record_safe(name)
    candidate, k = safe, 1
    while candidate in used:
        k += 1
        candidate = f"{safe}-{k}"
    used.add(candidate)
    if candidate != str(name):
        renamed.append((str(name), candidate))
    return candidate


def _bc_word(name: str, remembered: str | None, two_sided: bool) -> str:
    """Pick the boundary-condition word a face zone is written with.

    The word the mesh remembers for the tag wins; failing that, a tag named
    the way Fluent names its zones - ``wall``, ``velocity-inlet-7`` - is read
    for the word it opens with; anything else is a wall, or interior when
    every face of the zone sits between two cells. An interior zone holds
    faces between two cells only, so one with a face on the boundary is
    written as a wall whatever the tag says, and ``parent``, the type of a
    hanging-node refinement's coarse face, is never read off a name.
    """
    word = None
    if remembered in _BC_CODE:
        word = remembered
    else:
        for candidate in _BC_WORDS_BY_LENGTH:
            if name == candidate or name.startswith((f"{candidate}-", f"{candidate}_")):
                word = candidate
                break
    if word is None:
        return _DEFAULT_INTERIOR_ZONE if two_sided else _DEFAULT_WALL_ZONE
    if word == _DEFAULT_INTERIOR_ZONE and not two_sided:
        return _DEFAULT_WALL_ZONE
    return word


def _right_handed(
    dim: int,
    codes: np.ndarray,
    offsets: np.ndarray,
    conn: np.ndarray,
    cells: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Return ``conn`` with every written cell of negative measure mirrored.

    Parameters
    ----------
    dim
        2 or 3.
    codes, offsets, conn
        The mesh's elements; ``conn`` is left as it is, a copy coming back
        when a cell had to be mirrored.
    cells
        The elements written as cells.
    coords
        The node coordinates, ``dim`` columns.

    Notes
    -----
    A face record's node order fixes the side its normal points to - toward
    the second cell, or out of the mesh at the boundary - and the faces are
    read off :data:`ELEMENT_FACES`, which point outward only for a
    right-handed cell. A left-handed one in the mesh would come out with
    every face inverted, which Fluent reports as left-handed faces and
    refuses to run on, so it is mirrored here first.
    """
    out = conn
    for name in _CELL_TYPES[dim]:
        of_type = cells[codes[cells] == ELEMENT_TYPES[name]]
        if not of_type.size:
            continue
        if name == "polygon":
            for e in of_type.tolist():
                ring = conn[offsets[e] : offsets[e + 1]]
                if _signed_measure(ring[None], coords, name)[0] < 0:
                    if out is conn:
                        out = conn.copy()
                    out[offsets[e] + 1 : offsets[e + 1]] = ring[:0:-1]
            continue
        idx = offsets[of_type][:, None] + np.arange(NODES_PER_ELEMENT[name])
        rows = conn[idx]
        oriented = _oriented(rows, coords, name)
        if oriented is not rows:
            if out is conn:
                out = conn.copy()
            out[idx] = oriented
    return out


def _sorted_rows(rows: np.ndarray) -> np.ndarray:
    """Pad rings to four columns with -1 and sort each, for matching by node set."""
    padded = np.full((rows.shape[0], 4), -1, dtype=np.int64)
    padded[:, : rows.shape[1]] = rows
    return np.sort(padded, axis=1)


def _extract_faces(
    dim: int,
    codes: np.ndarray,
    offsets: np.ndarray,
    conn: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every face of the written cells, in cell order.

    Returns
    -------
    tuple
        ``(rings, widths, owner)``: the face's nodes padded to four columns
        with -1, in the outward orientation ``ELEMENT_FACES`` gives; how
        many of them are real; and the position of the cell in the written
        order.
    """
    rings: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    positions = np.arange(cells.size)
    for name in _CELL_TYPES[dim]:
        code = ELEMENT_TYPES[name]
        of_type = np.flatnonzero(codes[cells] == code)
        if not of_type.size:
            continue
        if name == "polygon":
            for pos in of_type.tolist():
                ring = conn[offsets[cells[pos]] : offsets[cells[pos] + 1]]
                edges = np.column_stack([ring, np.roll(ring, -1)])
                padded = np.full((edges.shape[0], 4), -1, dtype=np.int64)
                padded[:, :2] = edges
                rings.append(padded)
                widths.append(np.full(edges.shape[0], 2, dtype=np.int64))
                owners.append(np.full(edges.shape[0], pos, dtype=np.int64))
            continue
        faces = ELEMENT_FACES[name] if dim == 3 else _CELL_EDGES[name]
        rows = conn[
            offsets[cells[of_type]][:, None] + np.arange(NODES_PER_ELEMENT[name])
        ]
        for corners in faces:
            ring = rows[:, corners]
            padded = np.full((ring.shape[0], 4), -1, dtype=np.int64)
            padded[:, : ring.shape[1]] = ring
            rings.append(padded)
            widths.append(np.full(ring.shape[0], ring.shape[1], dtype=np.int64))
            owners.append(positions[of_type])
    return np.vstack(rings), np.concatenate(widths), np.concatenate(owners)


_HEX_FMT: dict[int, str] = {}


def _hex_fmt(n: int) -> str:
    """Return the ``%x %x ...`` format for a row of ``n`` values.

    One ``%`` per row is several times faster than a ``format`` call per
    value, which is where an ASCII face zone of a million records spends
    its time.
    """
    fmt = _HEX_FMT.get(n)
    if fmt is None:
        fmt = _HEX_FMT[n] = " ".join(["%x"] * n)
    return fmt


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a Fluent mesh.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output path, or an open file object.
    **opts
        None are recognised; any given is warned about and ignored.

    Raises
    ------
    CodecError
        If the vertices carry other than two or three columns, an element
        references a vertex that does not exist, the mesh holds no cell this
        codec writes, a two-dimensional mesh has left the plane, or a face
        sits between more than two cells.

    Notes
    -----
    The file is the standard face-based spelling: nodes, then one cell zone
    per ``element_tags`` group holding cells - ``fluid`` for the cells no
    group names - each declaring its element type, then the faces of every
    cell with the cell on either side, interior faces in one zone and the
    boundary in a zone per group naming the matching face elements, ``wall``
    for the rest. A mesh with a solid is written in three dimensions; one of
    triangles, quadrilaterals and polygons is a two-dimensional mesh, which
    Fluent holds in the plane only, so a surface with a third coordinate is
    refused rather than flattened. A face record's node order says which
    side its normal points to, so a left-handed cell is mirrored on the way
    out and its faces spelled outward.

    A face element - a triangle or quadrilateral of a solid mesh, a line of a
    planar one - is written only as the side of a cell it matches by node
    set, under the zone its group names; one that is no side of a written
    cell is dropped with a warning, since the format holds a face only
    between cells. An element of any other type is dropped with a warning.
    A Fluent zone holds each cell or face once, so an element in two groups
    stays with the first, with a warning. The zone's type word is taken from
    ``global_attrs["fluent_zone_types"]`` when the mesh remembers one, and
    otherwise from the tag's name when it opens with a Fluent boundary word.
    A name that cannot sit on a record has its whitespace, parentheses and
    quotes replaced by ``_``, with a warning. Nothing in the format carries
    per-entity data, so attributes and vertex tags are not written.
    """
    if opts:
        warnings.warn(
            f"{EXTENSION} write: unrecognized options {set(opts)}; ignored.",
            stacklevel=2,
        )

    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise CodecError(
            f"{EXTENSION}: vertices have shape {vertices.shape}, expected (n, 3)."
        )
    n_verts = vertices.shape[0]
    n_elems = len(poly.element_types)
    offsets = np.asarray(poly.offsets, dtype=np.int64)
    conn = np.asarray(poly.connectivity, dtype=np.int64)
    codes = np.asarray(poly.element_types, dtype=np.int64)
    if conn.size:
        lo, hi = int(conn.min()), int(conn.max())
        if lo < 0 or hi >= n_verts:
            where = f"0..{n_verts - 1}" if n_verts else "an empty vertex array"
            raise CodecError(
                f"{EXTENSION}: an element references vertex {lo}..{hi}, outside {where}."
            )

    dim = 3 if has_solid_cells(poly) else 2
    if (
        dim == 2
        and vertices.shape[1] == 3
        and (np.isfinite(vertices[:, 2]) & (vertices[:, 2] != 0)).any()
    ):
        raise CodecError(
            f"{EXTENSION}: the mesh holds no solid cell, so it would be a"
            " two-dimensional Fluent mesh, and Fluent holds one in the plane"
            " only - yet a vertex carries a third coordinate. Write the volume"
            " the surface bounds, or drop the third coordinate."
        )
    coords = vertices[:, :dim]
    if coords.shape[1] < dim:
        coords = np.hstack([coords, np.zeros((n_verts, dim - coords.shape[1]))])

    widths = np.diff(offsets)
    cell_codes = {ELEMENT_TYPES[n]: NODES_PER_ELEMENT[n] for n in _CELL_TYPES[dim]}
    side_codes = {ELEMENT_TYPES[n]: NODES_PER_ELEMENT[n] for n in _SIDE_TYPES[dim]}
    expected = np.full(n_elems, -2, dtype=np.int64)
    for code, width in (cell_codes | side_codes).items():
        expected[codes == code] = width
    well_formed = (expected == widths) | ((expected == -1) & (widths >= 3))
    is_cell = np.isin(codes, list(cell_codes)) & well_formed
    is_side = np.isin(codes, list(side_codes)) & well_formed
    unspellable = sorted(
        {
            ELEMENT_TYPES_INV.get(code, "") or f"code {code}"
            for code in np.unique(codes[expected == -2]).tolist()
        }
    )
    malformed = int(((expected != -2) & ~well_formed).sum())
    cells = np.flatnonzero(is_cell)
    if not cells.size:
        raise CodecError(
            f"{EXTENSION}: the mesh holds no cell to write; Fluent holds a mesh"
            f" as the faces of its cells, and the types written at"
            f" {dim} dimensions are {list(_CELL_TYPES[dim])}."
        )

    # Every cell and face belongs to one zone: the first group naming it.
    tag_names = list(poly.element_tags or {})
    zone_of = np.full(n_elems, -1, dtype=np.int64)
    overlapping: list[str] = []
    for t, name in enumerate(tag_names):
        idx = np.unique(member_indices(poly.element_tags[name], n_elems))
        idx = idx[is_cell[idx] | is_side[idx]]
        claimed = zone_of[idx] >= 0
        if claimed.any():
            overlapping.append(name)
        zone_of[idx[~claimed]] = t

    # Cells go out grouped by zone, zones in order of first appearance, so a
    # mesh whose groups are contiguous keeps its own order.
    cell_zone = zone_of[cells]
    _, first_seen = np.unique(cell_zone, return_index=True)
    rank = {int(cell_zone[i]): r for r, i in enumerate(sorted(first_seen.tolist()))}
    rank_of = np.empty(len(tag_names) + 1, dtype=np.int64)
    rank_of[[z + 1 for z in rank]] = list(rank.values())
    order = np.argsort(rank_of[cell_zone + 1], kind="stable")
    cells = cells[order]
    cell_zone = cell_zone[order]

    remembered = (poly.global_attrs or {}).get(ZONE_TYPES_KEY)
    if not isinstance(remembered, dict):
        remembered = {}
    used: set[str] = set()
    renamed: list[tuple[str, str]] = []
    zones: list[str] = []
    cell_zone_lines: list[str] = []
    zone_id = 1
    n_cells = cells.size
    at = 0
    for z in sorted(rank, key=rank.get):
        members = np.flatnonzero(cell_zone == z)
        raw = tag_names[z] if z >= 0 else _DEFAULT_CELL_ZONE
        name = _safe_name(raw, used, renamed)
        word = remembered.get(raw) if z >= 0 else None
        word = word if word in _CELL_CODE and word != "dead" else _FLUID
        zone_id += 1
        types_here = codes[cells[members]]
        distinct = np.unique(types_here)
        first, last = at + 1, at + members.size
        at = last
        if distinct.size == 1:
            # A zone of one type is its header alone; polygons are the
            # polyhedral type, which has no body to spell either.
            etype = _ELEM_CODE.get(ELEMENT_TYPES_INV[int(distinct[0])], _POLYHEDRAL)
            cell_zone_lines.append(
                f"(12 ({zone_id:x} {first:x} {last:x} {_CELL_CODE[word]:x} {etype:x}))"
            )
        else:
            spelled = [
                _ELEM_CODE.get(ELEMENT_TYPES_INV[int(c)], _POLYHEDRAL)
                for c in types_here.tolist()
            ]
            cell_zone_lines.append(
                f"(12 ({zone_id:x} {first:x} {last:x} {_CELL_CODE[word]:x} {_MIXED:x})("
            )
            cell_zone_lines.extend(
                _hex_fmt(len(chunk)) % tuple(chunk)
                for chunk in (
                    spelled[i : i + _TYPES_PER_LINE]
                    for i in range(0, len(spelled), _TYPES_PER_LINE)
                )
            )
            cell_zone_lines.append("))")
        zones.append(f"(45 ({zone_id} {word} {name})())")

    # Faces: each side of each cell, outward of a right-handed cell, matched
    # by node set into one face between at most two cells, then matched
    # against the face elements.
    conn = _right_handed(dim, codes, offsets, conn, cells, coords)
    rings, ring_widths, owner = _extract_faces(dim, codes, offsets, conn, cells)
    sides = np.flatnonzero(is_side)
    side_widths = widths[sides]
    side_parts: list[np.ndarray] = []
    side_order: list[np.ndarray] = []
    for w in np.unique(side_widths).tolist():
        of_width = np.flatnonzero(side_widths == w)
        side_parts.append(
            _sorted_rows(conn[offsets[sides[of_width]][:, None] + np.arange(w)])
        )
        side_order.append(of_width)
    if side_parts:
        side_rows = np.vstack(side_parts)
        sides = sides[np.concatenate(side_order)]
    else:
        side_rows = np.empty((0, 4), dtype=np.int64)
    keys = np.vstack([_sorted_rows(rings), side_rows])
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_faces_all = rings.shape[0]
    face_group = inverse[:n_faces_all]
    side_group = inverse[n_faces_all:]
    n_groups = int(inverse.max()) + 1 if inverse.size else 0
    per_group = np.bincount(face_group, minlength=n_groups)
    if (per_group > 2).any():
        crowded = int(np.flatnonzero(per_group > 2)[0])
        shared = np.flatnonzero(face_group == crowded)
        raise CodecError(
            f"{EXTENSION}: a face is shared by {shared.size} cells"
            f" (elements {sorted(cells[owner[shared]].tolist())}); Fluent holds"
            " a face between at most two."
        )
    by_group = np.lexsort((owner, face_group))
    ranked = face_group[by_group]
    starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
    rep = by_group[starts]
    group_of_rep = face_group[rep]
    two_sided = per_group[group_of_rep] == 2
    second = np.where(
        two_sided, by_group[np.minimum(starts + 1, by_group.size - 1)], -1
    )
    n_unique = rep.size

    # The zone of each unique face: the group of the side element matching
    # it, or a default by whether it is interior.
    face_tag = np.full(n_unique, -1, dtype=np.int64)
    matched_sides = np.zeros(sides.size, dtype=bool)
    if sides.size:
        rep_of_group = np.full(n_groups, -1, dtype=np.int64)
        rep_of_group[group_of_rep] = np.arange(n_unique)
        hit = rep_of_group[side_group]
        fresh = hit >= 0
        _, first_hit = np.unique(hit[fresh], return_index=True)
        chosen = np.flatnonzero(fresh)[first_hit]
        matched_sides[chosen] = True
        face_tag[hit[chosen]] = zone_of[sides[chosen]]
    unmatched = int(sides.size - matched_sides.sum())

    # Face zones: interior first, then one per group, then the default wall.
    face_zone_key = np.where(face_tag >= 0, face_tag, np.where(two_sided, -2, -1))
    zone_keys = [-2, *sorted({int(k) for k in face_tag[face_tag >= 0].tolist()}), -1]
    face_zone_lines: list[str] = []
    n_faces = 0
    for key in zone_keys:
        members = np.flatnonzero(face_zone_key == key)
        if not members.size:
            continue
        if key == -2:
            raw, word = _DEFAULT_INTERIOR_ZONE, _DEFAULT_INTERIOR_ZONE
        elif key == -1:
            raw, word = _DEFAULT_WALL_ZONE, _DEFAULT_WALL_ZONE
        else:
            raw = tag_names[key]
            word = _bc_word(raw, remembered.get(raw), bool(two_sided[members].all()))
        # A group naming both cells and faces is two zones with one name,
        # which the format does not allow; the face zone takes a suffix.
        safe = _record_safe(raw)
        candidate = raw if safe not in used else f"{safe}-faces"
        name = _safe_name(candidate, used, renamed)
        if candidate != raw and (raw, name) not in renamed:
            renamed.append((raw, name))
        zone_id += 1
        w = ring_widths[rep[members]]
        distinct = np.unique(w)
        counted = distinct.size != 1 or int(distinct[0]) not in _WIDTH_FACE
        face_type = _FACE_MIXED if counted else _WIDTH_FACE[int(distinct[0])]
        first, last = n_faces + 1, n_faces + members.size
        n_faces = last
        face_zone_lines.append(
            f"(13 ({zone_id:x} {first:x} {last:x} {_BC_CODE[word]:x} {face_type:x})("
        )
        nodes = (rings[rep[members]] + 1).tolist()
        c0 = (owner[rep[members]] + 1).tolist()
        c1 = np.where(
            second[members] >= 0, owner[np.maximum(second[members], 0)] + 1, 0
        ).tolist()
        if counted:
            for row, wd, a, b in zip(nodes, w.tolist(), c0, c1, strict=True):
                face_zone_lines.append(_hex_fmt(wd + 3) % (wd, *row[:wd], a, b))
        else:
            wd = int(distinct[0])
            fmt = _hex_fmt(wd + 2)
            face_zone_lines.extend(
                fmt % (*row[:wd], a, b) for row, a, b in zip(nodes, c0, c1, strict=True)
            )
        face_zone_lines.append("))")
        zones.append(f"(45 ({zone_id} {word} {name})())")

    non_finite = int((~np.isfinite(coords)).any(axis=1).sum()) if n_verts else 0
    lines = [
        '(0 "Written by polyxios")',
        '(1 "polyxios")',
        f"(2 {dim})",
        f"(10 (0 1 {n_verts:x} 0))",
        f"(10 (1 1 {n_verts:x} 1 {dim})(",
    ]
    coord_fmt = " ".join(["%r"] * dim)
    lines.extend(coord_fmt % tuple(row) for row in coords.tolist())
    lines.append("))")
    lines.append(f"(12 (0 1 {n_cells:x} 0))")
    lines.extend(cell_zone_lines)
    lines.append(f"(13 (0 1 {n_faces:x} 0))")
    lines.extend(face_zone_lines)
    lines.extend(zones)

    if unspellable:
        warnings.warn(
            f"{EXTENSION}: element type(s) {unspellable} are neither a cell nor"
            f" a face Fluent holds at {dim} dimensions; those elements were dropped.",
            stacklevel=2,
        )
    if malformed:
        warnings.warn(
            f"{EXTENSION}: {malformed} element(s) carry a node count that does"
            " not match their type; dropped.",
            stacklevel=2,
        )
    if overlapping:
        warnings.warn(
            f"{EXTENSION}: element tag group(s) {overlapping} name elements an"
            " earlier group already claimed; a Fluent zone holds each cell or"
            " face once, so those members stayed with the first group.",
            stacklevel=2,
        )
    if unmatched:
        warnings.warn(
            f"{EXTENSION}: {unmatched} face element(s) are no side of a written"
            " cell, or repeat one; Fluent holds a face only between cells, so"
            " they were dropped.",
            stacklevel=2,
        )
    if non_finite:
        warnings.warn(
            f"{EXTENSION}: {non_finite} node(s) have non-finite coordinates;"
            " Fluent will not load the file.",
            stacklevel=2,
        )
    if renamed:
        warnings.warn(
            f"{EXTENSION}: a zone name cannot carry whitespace, parentheses,"
            f" quotes or a repeat; wrote {sorted(set(renamed))}.",
            stacklevel=2,
        )

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
