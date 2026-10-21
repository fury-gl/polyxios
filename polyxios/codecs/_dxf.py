"""DXF (AutoCAD Drawing Exchange Format) codec - read + write.

A DXF file is a flat run of group code / value pairs, ASCII or binary, split
into sections; the ``ENTITIES`` section draws, the ``BLOCKS`` section holds
named groups of entities that an ``INSERT`` places. The reader takes the
entities that are mesh geometry - faces, polyface and polygon meshes,
subdivision meshes, polylines, lines and points - expands every insert under
its transform, and tags each element with its layer. The writer spells
AutoCAD R12, which every DXF reader opens: faces as polyface meshes, lines,
3D polylines and points, one layer per element tag group.
"""

from __future__ import annotations

import codecs as _codecs
import math
import re
import struct
from typing import Any

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    QUADRATIC_SURFACE_CORNERS,
)
from polyxios._io import Source, read_bytes, source_name, write_bytes
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".dxf"

_BINARY_SENTINEL: bytes = b"AutoCAD Binary DXF\r\n\x1a\x00"

_VERTEX: int = ELEMENT_TYPES["vertex"]
_POLY_VERTEX: int = ELEMENT_TYPES["poly_vertex"]
_LINE: int = ELEMENT_TYPES["line"]
_POLY_LINE: int = ELEMENT_TYPES["poly_line"]
_TRIANGLE: int = ELEMENT_TYPES["triangle"]
_STRIP: int = ELEMENT_TYPES["triangle_strip"]
_POLYGON: int = ELEMENT_TYPES["polygon"]
_PIXEL: int = ELEMENT_TYPES["pixel"]
_QUAD: int = ELEMENT_TYPES["quad"]
_QUADRATIC_EDGE: int = ELEMENT_TYPES["quadratic_edge"]
_FACES: frozenset[int] = frozenset({_TRIANGLE, _QUAD, _POLYGON})

# $INSUNITS codes, spelled the way 3MF and AMF spell the units they share.
_INSUNITS: dict[int, str] = {
    1: "inch",
    2: "foot",
    3: "mile",
    4: "millimeter",
    5: "centimeter",
    6: "meter",
    7: "kilometer",
    8: "microinch",
    9: "mil",
    10: "yard",
    11: "angstrom",
    12: "nanometer",
    13: "micron",
    14: "decimeter",
    15: "decameter",
    16: "hectometer",
    17: "gigameter",
    18: "astronomical_unit",
    19: "light_year",
    20: "parsec",
    21: "us_survey_foot",
    22: "us_survey_inch",
    23: "us_survey_yard",
    24: "us_survey_mile",
}
_UNIT_CODES: dict[str, int] = {
    **{unit: code for code, unit in _INSUNITS.items()},
    "feet": 2,
    "micrometer": 13,
}

# Entities that draw nothing a mesh holds, passed over without a warning: a
# viewport frames paper space, and the rest belong to an entity read whole.
_SILENT: frozenset[str] = frozenset(
    {"VIEWPORT", "SEQEND", "VERTEX", "ATTRIB", "ATTDEF"}
)
# The entities read through the shared per-type field tables.
_FLAT: frozenset[str] = frozenset({"3DFACE", "SOLID", "TRACE", "LINE", "POINT"})

# A polyface mesh's vertex and face counts are 16-bit in R12, and a face holds
# at most four vertices, so this many faces never outgrow either count.
_POLYFACE_FACES: int = 32767 // 4
_R12_VERSION: str = "AC1009"
# From AutoCAD 2007 (AC1021) every string is UTF-8; before, $DWGCODEPAGE says.
_UTF8_FROM: str = "AC1021"
_UNICODE_ESCAPE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")
# An R12 name in an East Asian code page spells each double-byte character
# \M+nXXXX: n the code page, XXXX the character's two bytes in it.
_MBCS_ESCAPE = re.compile(r"\\M\+([1-5])([0-9A-Fa-f]{4})")
_MBCS_CODE_PAGES: dict[str, str] = {
    "1": "cp932",
    "2": "cp950",
    "3": "cp949",
    "4": "cp1361",
    "5": "cp936",
}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# What no DXF version allows in a layer name, as ezdxf audits it.
_INVALID_LAYER_CHARS: str = '<>/\\":;?*=`'
_INVALID_LAYER = re.compile("[" + re.escape(_INVALID_LAYER_CHARS) + "]")
# Group codes are 16-bit in binary DXF; the largest DXF defines is 1071.
_MAX_GROUP_CODE: int = 32767

# Total vertices and connectivity entries the inserts may expand to, per byte
# of file, before the read is refused: an INSERT is a few dozen bytes however
# many copies it asks for. Each copy is charged this much on top of its own,
# so a block of one point copied millions of times counts the work it costs.
_EXPANSION_PER_BYTE: int = 64
_EXPANSION_FLOOR: int = 1 << 22
_COPY_COST: int = 16
# The width of the bytes arrays lines are parsed through: one numpy sized by
# its longest entry would let a line padded to a few thousand spaces
# multiply the memory of every other.
_FAST_WIDTH: int = 32
_EOF_WINDOW: int = 256
_SHOWN_WHOLE: int = 10**20
_INT64_BOUND: float = 2.0**63


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a DXF drawing and return its mesh geometry.

    Parameters
    ----------
    path
        Path to the ``.dxf`` file, ASCII or binary, or an open binary file
        object.
    lazy
        Not supported: DXF numbers are text, or binary values of mixed
        width between group codes.

    Returns
    -------
    PolyData
        The model-space entities that are mesh geometry, in file order:
        ``3DFACE``, ``SOLID`` and ``TRACE`` as triangles and quads; polyface
        and polygon-mesh ``POLYLINE`` entities and ``MESH`` as faces; other
        ``POLYLINE`` and ``LWPOLYLINE`` entities as poly lines, or as a line
        when they hold two vertices; ``LINE`` as lines; ``POINT`` as
        vertices. An ``INSERT`` adds its block's geometry under its
        position, scale, rotation, extrusion and column and row array,
        nested blocks included. Vertices at equal
        coordinates are one vertex: DXF entities share none. Every element
        is in the element tag group of its layer, an entity on layer ``0``
        inside a block taking the layer of the insert; layer names are
        compared without case, a layer keeping its first spelling. A string
        ends at its first NUL, as in a binary file, and one a file from
        AutoCAD 2007 on does not spell in UTF-8 is read as cp1252 with a
        warning. A ``3DFACE``, ``SOLID`` or ``TRACE`` missing one of its
        first three corners, a ``LINE`` missing an end, a ``POINT``
        missing its location and a polyface face naming no vertex are
        skipped and counted, and a block defined
        twice keeps its first definition, both with a warning; a field
        spelled twice in one entity keeps its first value, and a ``\\U+XXXX``
        or ``\\M+nXXXX`` escape in a string is decoded. ``$INSUNITS`` lands in
        ``global_attrs["unit"]`` and ``$ACADVER`` in
        ``global_attrs["dxf_version"]``.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        When the file holds no section - it is empty, or not a DXF
        drawing - a group code is not an integer, a number is not one, the
        file ends inside a section or a pair, a block has no ``ENDBLK``, a
        polyface face names a vertex its mesh lacks, a polygon mesh holds
        the wrong number of vertices, an insert names a block the file lacks
        or one that inserts itself, an insert places a vertex past the
        float range, or the inserts expand past what the file's size can
        account for, each copy counted as its vertices, its connectivity
        entries and 16 more. What follows the first ``0``/``EOF`` pair of
        an ASCII file is ignored.
    """
    if lazy:
        raise LazyReadError(
            "DXF is a stream of tagged values and cannot be memory-mapped."
        )
    name = source_name(path)
    raw = read_bytes(path)
    if raw.startswith(_BINARY_SENTINEL):
        codes, values = _binary_pairs(raw, name)
    else:
        codes, values = _ascii_pairs(raw, name)
    doc = _Document(
        codes,
        values,
        name,
        budget=max(_EXPANSION_FLOOR, _EXPANSION_PER_BYTE * len(raw)),
    )
    return doc.read()


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as an AutoCAD R12 DXF drawing.

    Parameters
    ----------
    poly
        PolyData to write. Triangles, quads, pixels and polygons of up to
        four corners go out as polyface mesh faces, larger polygons and
        triangle strips split into triangles, a quadratic surface element
        as its corners; lines as ``LINE``, poly lines as 3D ``POLYLINE``
        (closed when their last vertex is their first) - but one of two
        vertices as the ``LINE`` it reads back as - vertices and poly
        vertices as ``POINT``. Volume elements have no DXF entity and are
        dropped with a warning, as is an element with too few vertices to
        draw. Each element tag group is a layer; an element in two groups
        is on the first, groups whose names differ only in case share the
        first one's layer, and the untagged elements are on layer ``0``,
        with a warning when a group is written there too. The characters no
        DXF layer name allows - ``< > / \\ " : ; ? * =`` and the backquote -
        become ``_``, surrounding white space is dropped and a name left
        empty is ``_``, with a warning; length, case and non-ASCII
        characters are kept, the latter as ``\\U+XXXX``.
        ``global_attrs["unit"]`` is written as ``$INSUNITS`` when DXF has a
        code for it; the vertex and element attributes are not written, as
        R12 entities carry none. Only the vertices an element uses are
        written, and a pixel reads back as a quad, a polygon of three or
        four corners as a triangle or a quad.
    path
        Output file path, or an open binary file object.

    Raises
    ------
    CodecError
        When the mesh has elements but none DXF can draw, a vertex an
        element uses is not finite, or a tag group name holds a control
        character a DXF line cannot carry.
    """
    layers = _layers(poly)
    runs = _runs(poly, layers)
    flat = (
        np.concatenate([conn for _, _, conn, _ in runs])
        if runs
        else np.empty(0, np.int64)
    )
    used = np.unique(flat)
    points = np.asarray(poly.vertices[used], dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise CodecError(
            ".dxf: a vertex an element uses is NaN or infinite; DXF has no"
            " spelling for either."
        )
    out: list[str] = []
    _header(out, poly)
    _tables(out, layers.names)
    out.append("  0\nSECTION\n  2\nENTITIES\n")
    spelled = {10: _xyz_lines(points, 10)}
    if any(kind == "line" for kind, *_ in runs):
        spelled[11] = _xyz_lines(points, 11)
    names = [_layer_name(name) for name in layers.names]
    local = np.searchsorted(used, flat).tolist()
    pos = 0
    for kind, layer, conn, offsets in runs:
        stop = pos + len(conn)
        _entities(out, kind, names[layer], spelled, local[pos:stop], offsets.tolist())
        pos = stop
    out.append("  0\nENDSEC\n  0\nEOF\n")
    write_bytes(path, "".join(out).encode("ascii"))


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------


def _ascii_pairs(raw: bytes, name: str) -> tuple[np.ndarray, list[bytes]]:
    """Split an ASCII DXF into its group codes and their raw values.

    What follows the first ``0``/``EOF`` pair is no part of the drawing; it
    is looked for only when the whole file does not split into pairs, so a
    clean file is not searched for it.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return _split_pairs(raw.splitlines(), name)
    except CodecError:
        end = _ascii_end(raw)
        if not raw[end:].strip():
            raise
    return _split_pairs(raw[:end].splitlines(), name)


def _split_pairs(lines: list[bytes], name: str) -> tuple[np.ndarray, list[bytes]]:
    """Return the group codes and raw values the lines of an ASCII DXF spell."""
    if len(lines) % 2:
        if lines and not lines[-1].strip():
            lines.pop()
        else:
            raise CodecError(
                f"{name!r} ends on group code {lines[-1].strip()!r} with no value:"
                " the file is truncated."
            )
    code_lines = lines[0::2]
    try:
        fixed = _narrow(code_lines)
        codes = None if fixed is None else fixed.astype(np.int64)
    except (ValueError, OverflowError):
        codes = None
    if codes is not None and codes.size and np.abs(codes).max() > _MAX_GROUP_CODE:
        codes = None
    if codes is None:
        codes = np.array(
            [_group_code(line, 2 * k + 1, name) for k, line in enumerate(code_lines)],
            dtype=np.int64,
        )
    values = lines[1::2]
    comments = codes == 999
    if comments.any():
        keep = np.flatnonzero(~comments)
        codes = codes[keep]
        values = [values[k] for k in keep.tolist()]
    return codes, values


def _ascii_end(raw: bytes) -> int:
    """Return where the first ``0``/``EOF`` pair of an ASCII DXF ends.

    What follows it, such as a DOS end-of-file byte or NUL padding, is no
    part of the drawing, as a binary file stops there too; the whole length
    when the file has no such pair. Lines are looked for within
    ``_EOF_WINDOW`` bytes of each candidate, so text full of ``EOF`` stays
    linear.
    """
    pos = raw.find(b"EOF")
    while pos >= 0:
        stop = pos + 3
        line = _line_around(raw, pos, stop)
        if line is not None and raw[line[0] : line[1]].strip(b" \t") == b"EOF":
            start = line[0] - 1
            if start > 0 and raw[start] == 10 and raw[start - 1] == 13:
                start -= 1
            code = _line_around(raw, start, start) if start > 0 else None
            if code is not None and raw[code[0] : code[1]].strip(b" \t") == b"0":
                return line[1]
        pos = raw.find(b"EOF", stop)
    return len(raw)


def _line_around(raw: bytes, start: int, stop: int) -> tuple[int, int] | None:
    """Return the bounds of the line holding ``raw[start:stop]``, None if too long."""
    low = max(0, start - _EOF_WINDOW)
    begin = max(raw.rfind(b"\n", low, start), raw.rfind(b"\r", low, start)) + 1
    if not begin and low:
        return None
    high = min(len(raw), stop + _EOF_WINDOW)
    ends = [
        e for e in (raw.find(b"\n", stop, high), raw.find(b"\r", stop, high)) if e >= 0
    ]
    if not ends and high < len(raw):
        return None
    return begin, min(ends) if ends else high


def _narrow(lines: list[bytes]) -> np.ndarray | None:
    """Return ``lines`` as a bytes array of ``_FAST_WIDTH`` per entry.

    numpy cuts an entry to the width without a word, so one that fills it
    is put back stripped, as numpy and ``int`` strip it anyway; None when
    one is still wider, for the caller to parse line by line. A fixed width
    is also faster than one numpy works out from the longest entry.
    """
    fixed = np.array(lines, dtype=f"S{_FAST_WIDTH}")
    full = fixed.view(np.uint8).reshape(-1, _FAST_WIDTH)[:, -1] != 0
    for i in np.flatnonzero(full).tolist():
        stripped = lines[i].strip()
        if len(stripped) > _FAST_WIDTH:
            return None
        fixed[i] = stripped
    return fixed


def _group_code(line: bytes, number: int, name: str) -> int:
    """Return the group code on line ``number``, refusing what is not one.

    The slow path, taken only when numpy cannot read every code at once.
    """
    try:
        code = int(line)
    except ValueError:
        code = None
    if code is None or abs(code) > _MAX_GROUP_CODE:
        raise CodecError(
            f"{name!r}: line {number} should hold a group code, not {line[:40]!r}."
        )
    return code


def _binary_kind(code: int) -> str:
    """Return how a binary DXF stores the value of a group code.

    The binary chunks 310-319 sit inside the 300-369 string range and are
    checked first: a chunk is a length byte and that many bytes, NULs
    included.
    """
    if 310 <= code <= 319 or code == 1004:
        return "b"
    if 0 <= code <= 9 or code in (100, 101, 102, 105) or 300 <= code <= 369:
        return "s"
    if 390 <= code <= 399 or 410 <= code <= 419 or 430 <= code <= 439:
        return "s"
    if 470 <= code <= 481 or code == 999 or 1000 <= code <= 1009 and code != 1004:
        return "s"
    if 10 <= code <= 59 or 110 <= code <= 149 or 210 <= code <= 239:
        return "d"
    if 460 <= code <= 469 or 1010 <= code <= 1059:
        return "d"
    if 60 <= code <= 79 or 170 <= code <= 179 or 270 <= code <= 289:
        return "h"
    if 370 <= code <= 389 or 400 <= code <= 409 or 1060 <= code <= 1070:
        return "h"
    if 90 <= code <= 99 or 420 <= code <= 429 or 440 <= code <= 459 or code == 1071:
        return "i"
    if 160 <= code <= 169:
        return "q"
    if 290 <= code <= 299:
        return "?"
    return ""


_BINARY_WIDTH: dict[str, int] = {"d": 8, "h": 2, "i": 4, "q": 8, "?": 1}


def _binary_pairs(raw: bytes, name: str) -> tuple[np.ndarray, list[Any]]:
    """Split a binary DXF into its group codes and their decoded values.

    R12 spells a group code in one byte, 255 escaping a two-byte one; from
    R13 every code is two bytes. The first code is 0, or the 999 of a
    comment: read as two bytes, either is ``0`` or ``999`` only in a file of
    two-byte codes, as R12 follows a one-byte 0 with the first letter of
    ``SECTION`` and escapes 999 behind a 255. Comments are dropped, as in
    an ASCII file.
    """
    pos = len(_BINARY_SENTINEL)
    if len(raw) < pos + 2:
        raise CodecError(f"{name!r} is a binary DXF holding no group code.")
    wide = struct.unpack_from("<h", raw, pos)[0] in (0, 999)
    codes: list[int] = []
    values: list[Any] = []
    end = len(raw)
    try:
        while pos < end:
            if wide:
                (code,) = struct.unpack_from("<h", raw, pos)
                pos += 2
            else:
                code = raw[pos]
                pos += 1
                if code == 255:
                    (code,) = struct.unpack_from("<h", raw, pos)
                    pos += 2
            kind = _binary_kind(code)
            if kind == "s":
                stop = raw.index(b"\x00", pos)
                value: Any = raw[pos:stop]
                pos = stop + 1
            elif kind == "b":
                size = raw[pos]
                if pos + 1 + size > end:
                    raise IndexError(pos)
                value = raw[pos + 1 : pos + 1 + size]
                pos += 1 + size
            elif kind:
                width = _BINARY_WIDTH[kind]
                (value,) = struct.unpack_from("<" + kind, raw, pos)
                pos += width
            else:
                raise CodecError(
                    f"{name!r}: binary group code {code} at byte {pos} is not one"
                    " DXF defines."
                )
            if code == 999:
                continue
            codes.append(code)
            values.append(value)
            if code == 0 and value == b"EOF":
                break
    except (struct.error, ValueError, IndexError) as exc:
        raise CodecError(
            f"{name!r}: the binary DXF is truncated at byte {pos}."
        ) from exc
    return np.array(codes, dtype=np.int64), values


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class _Piece:
    """Geometry gathered from a run of entities, in the frame they were drawn in.

    Every element carries the index of its layer in the document's layer
    list; inside a block, layer ``0`` stands for the layer of the insert.
    """

    __slots__ = ("conn", "layers", "n_vertices", "sizes", "types", "vertices")

    def __init__(self) -> None:
        self.n_vertices = 0
        self.vertices: list[np.ndarray] = []
        self.conn: list[np.ndarray] = []
        self.sizes: list[np.ndarray] = []
        self.types: list[np.ndarray] = []
        self.layers: list[np.ndarray] = []

    def add(
        self,
        vertices: np.ndarray,
        conn: np.ndarray,
        sizes: np.ndarray,
        types: np.ndarray,
        layers: np.ndarray,
    ) -> None:
        """Append elements over their own vertices, indices local to ``vertices``."""
        base = self.n_vertices
        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.n_vertices += len(vertices)
        self.vertices.append(vertices)
        self.conn.append(np.asarray(conn, dtype=np.int64) + base)
        self.sizes.append(np.asarray(sizes, dtype=np.int64))
        self.types.append(np.asarray(types, dtype=np.uint8))
        self.layers.append(np.asarray(layers, dtype=np.int64))

    def flat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the gathered vertices, connectivity, sizes, types and layers."""
        if not self.vertices:
            return (
                np.empty((0, 3)),
                np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.uint8),
                np.empty(0, np.int64),
            )
        out = (
            np.concatenate(self.vertices),
            np.concatenate(self.conn),
            np.concatenate(self.sizes),
            np.concatenate(self.types),
            np.concatenate(self.layers),
        )
        self.vertices, self.conn, self.sizes, self.types, self.layers = (
            [out[0]],
            [out[1]],
            [out[2]],
            [out[3]],
            [out[4]],
        )
        return out


class _Document:
    """The pairs of one DXF file and what reading them has gathered.

    Parameters
    ----------
    codes
        Every group code, in file order.
    values
        Each code's value: the raw line of an ASCII file, or the decoded
        number or bytes of a binary one.
    name
        The source, for messages.
    budget
        Vertices and connectivity entries the inserts may expand to before
        the read is refused.
    """

    def __init__(
        self,
        codes: np.ndarray,
        values: list[Any],
        name: str,
        *,
        budget: int,
    ) -> None:
        self.codes = codes
        self.values = values
        self.name = name
        self.budget = budget
        self.starts = np.flatnonzero(codes == 0)
        self.ends = np.append(self.starts[1:], len(codes))
        self.kinds = [_upper(values[s]) for s in self.starts.tolist()]
        self.encoding = "utf-8"
        self.version: str | None = None
        self.utf8_only = False
        self.unit_code: int | None = None
        self.layer_names: list[str] = []
        self.layer_ids: dict[str, int] = {}
        self.blocks: dict[str, tuple[np.ndarray, list[int]]] = {}
        self.expanded: dict[str, _Piece] = {}
        self.skipped: dict[str, int] = {}
        self.bulges = 0
        self.misdecoded = 0
        self.produced = 0
        self.duplicates: list[str] = []
        self._layer_cache: dict[Any, int] = {}
        # Entities spelling group 67 as 1 are in paper space: layout, not model.
        at = np.flatnonzero(codes == 67)
        owner = np.searchsorted(self.starts, at, side="right") - 1
        flag = self._numbers([values[i] for i in at.tolist()], "an entity's", codes[at])
        self.paper = np.zeros(len(self.starts), dtype=bool)
        self.paper[owner[(flag == 1) & (owner >= 0)]] = True

    # -- structure ----------------------------------------------------------

    def read(self) -> PolyData:
        entities: list[int] = []
        k = 0
        n = len(self.starts)
        sections = 0
        while k < n:
            kind = self.kinds[k]
            if kind == "EOF":
                break
            if kind != "SECTION":
                k += 1
                continue
            sections += 1
            section = self._text(self._field(k, 2) or b"").strip().upper()
            close = k + 1
            while close < n and self.kinds[close] not in ("ENDSEC", "EOF"):
                close += 1
            if close >= n or self.kinds[close] != "ENDSEC":
                raise CodecError(
                    f"{self.name!r}: section {section or '?'} never ends: the"
                    " file is truncated."
                )
            if section == "HEADER":
                self._header(k)
            elif section == "BLOCKS":
                self._blocks(k + 1, close)
            elif section == "ENTITIES":
                entities.extend(range(k + 1, close))
            k = close + 1
        if not sections:
            raise CodecError(
                f"{self.name!r} holds no DXF section: it is empty or not a DXF drawing."
            )
        if self.duplicates:
            warn_caller(
                f"{self.name!r}: block(s) {self.duplicates} are defined again;"
                " block names ignore case, and each insert places the first"
                " definition."
            )
        piece = self._collect(entities)
        if self.misdecoded:
            warn_caller(
                f"{self.name!r}: {self.misdecoded} string(s) are not the UTF-8"
                f" {self.version} requires; they are read as cp1252."
            )
        if self.bulges:
            warn_caller(
                f"{self.name!r}: {self.bulges} polyline segment(s) are arcs"
                " (bulge); they are read as straight segments."
            )
        if self.skipped:
            named = ", ".join(f"{n} {kind}" for kind, n in sorted(self.skipped.items()))
            warn_caller(
                f"{self.name!r}: entities that are not mesh geometry were skipped: {named}."
            )
        return self._assemble(piece)

    def _header(self, k: int) -> None:
        start, stop = int(self.starts[k]), int(self.ends[k])
        codes = self.codes
        for i in range(start, stop - 1):
            if codes[i] != 9:
                continue
            var = self._text(self.values[i]).strip().upper()
            value = self.values[i + 1]
            if var == "$ACADVER":
                self.version = self._text(value).strip()
            elif var == "$DWGCODEPAGE":
                self.encoding = _codepage(self._text(value).strip(), self.encoding)
            elif var == "$INSUNITS":
                self.unit_code = int(self._number(value, codes[i + 1], "HEADER"))
        if self.version is not None:
            # Version names are compared spelled as AutoCAD spells them.
            self.utf8_only = self.version.upper() >= _UTF8_FROM
        if self.utf8_only:
            self.encoding = "utf-8"

    def _blocks(self, first: int, stop: int) -> None:
        k = first
        while k < stop:
            if self.kinds[k] != "BLOCK":
                k += 1
                continue
            end = k + 1
            while end < stop and self.kinds[end] not in ("ENDBLK", "BLOCK"):
                end += 1
            block = self._text(self._field(k, 2) or b"").strip()
            if end >= stop or self.kinds[end] != "ENDBLK":
                raise CodecError(
                    f"{self.name!r}: block {block!r} never ends: no ENDBLK comes"
                    " before the next block or the end of the section."
                )
            if block.upper() in self.blocks:
                self.duplicates.append(block)
            else:
                base = np.array([self._float_field(k, c, 0.0) for c in (10, 20, 30)])
                self.blocks[block.upper()] = (base, list(range(k + 1, end)))
            k = end + 1

    # -- entities -----------------------------------------------------------

    def _collect(self, entities: list[int]) -> _Piece:
        """Gather the geometry of a run of entities into one piece.

        The fields of every entity of a flat type, and of every polyline
        vertex, are parsed in one pass per type; the entities are then
        added in file order.
        """
        piece = _Piece()
        groups: dict[str, list[int]] = {}
        order: list[tuple[str, int, int, int]] = []
        vertex_ks: list[int] = []
        pos = 0
        while pos < len(entities):
            k = entities[pos]
            kind = self.kinds[k]
            pos += 1
            if kind == "POLYLINE":
                lo = len(vertex_ks)
                while pos < len(entities) and self.kinds[entities[pos]] == "VERTEX":
                    vertex_ks.append(entities[pos])
                    pos += 1
                if not self.paper[k]:
                    order.append((kind, k, lo, len(vertex_ks)))
                continue
            if self.paper[k]:
                continue
            if kind in _FLAT:
                group = groups.setdefault(kind, [])
                order.append((kind, k, len(group), 0))
                group.append(k)
            elif kind in ("LWPOLYLINE", "MESH", "INSERT"):
                order.append((kind, k, 0, 0))
            elif kind not in _SILENT:
                self._skip(kind)
        flat = {kind: self._flat(kind, ks) for kind, ks in groups.items()}
        vertex_fields = self._flat("VERTEX", vertex_ks) if vertex_ks else None
        step = 0
        while step < len(order):
            kind, k, i, hi = order[step]
            step += 1
            if kind in _FLAT:
                # A run of one flat kind is added in one batch, in file order.
                first = i
                while step < len(order) and order[step][0] == kind:
                    step += 1
                    i += 1
                self._add_flat(piece, kind, flat[kind], np.arange(first, i + 1))
            elif kind == "POLYLINE":
                self._polyline(piece, k, vertex_fields, i, hi)
            elif kind == "LWPOLYLINE":
                self._lwpolyline(piece, k)
            elif kind == "MESH":
                self._mesh(piece, k)
            else:
                self._insert(piece, k)
        return piece

    def _flat(self, kind: str, ks: list[int]) -> dict[int | str, np.ndarray]:
        """Return the numeric fields of the entities ``ks``, one array per code.

        A code an entity does not spell is NaN; ``"layer"`` holds each
        entity's layer index.
        """
        wanted = _FLAT_CODES[kind]
        starts = self.starts[ks]
        ends = self.ends[ks]
        lengths = ends - starts
        owner = np.repeat(np.arange(len(ks)), lengths)
        idx = np.arange(int(lengths.sum())) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        idx = idx + np.repeat(starts, lengths)
        codes = self.codes[idx]
        out: dict[int | str, np.ndarray] = {}
        pick = np.isin(codes, wanted)
        sel = idx[pick]
        sel_owner = owner[pick]
        sel_codes = codes[pick]
        numbers = self._numbers([self.values[i] for i in sel.tolist()], kind, sel_codes)
        for code in wanted:
            column = np.full(len(ks), np.nan)
            hit = np.flatnonzero(sel_codes == code)
            # A code spelled twice keeps its first value, as _field does.
            rows, first = np.unique(sel_owner[hit], return_index=True)
            column[rows] = numbers[hit[first]]
            out[code] = column
        layer = np.full(len(ks), self._layer("0"), dtype=np.int64)
        hit = np.flatnonzero(codes == 8)
        rows, first = np.unique(owner[hit], return_index=True)
        for row, i in zip(rows.tolist(), idx[hit[first]].tolist(), strict=True):
            layer[row] = self._raw_layer(self.values[i])
        out["layer"] = layer
        return out

    def _add_flat(
        self, piece: _Piece, kind: str, f: dict[int | str, np.ndarray], rows: np.ndarray
    ) -> None:
        """Add the flat entities ``rows`` of one kind, every one in a single batch."""

        def corner(c: int) -> np.ndarray:
            return np.column_stack([f[c][rows], f[c + 10][rows], f[c + 20][rows]])

        # A required point no code spells would read as the origin, so the
        # entity is skipped and counted instead.
        required, what = _REQUIRED[kind]
        absent = np.zeros(len(rows), dtype=bool)
        for c in required:
            absent |= np.isnan(corner(c)).all(axis=1)
        if absent.any():
            self._skip(f"{kind} without {what}", count=int(absent.sum()))
            rows = rows[~absent]
        layers = f["layer"][rows]
        m = len(rows)
        if kind == "POINT":
            piece.add(
                np.nan_to_num(corner(10)),
                np.arange(m),
                np.ones(m),
                np.full(m, _VERTEX),
                layers,
            )
            return
        if kind == "LINE":
            ends = np.stack([corner(10), corner(11)], axis=1)
            piece.add(
                np.nan_to_num(ends).reshape(-1, 3),
                np.arange(2 * m),
                np.full(m, 2),
                np.full(m, _LINE),
                layers,
            )
            return
        third = corner(12)
        fourth = corner(13)
        missing = np.isnan(fourth).all(axis=1)
        fourth[missing] = third[missing]
        if kind == "3DFACE":
            corners = np.stack([corner(10), corner(11), third, fourth], axis=1)
        else:
            # SOLID and TRACE run their corners 1, 2, 4, 3, in the OCS of their
            # extrusion at the elevation of the first.
            corners = np.stack([corner(10), corner(11), fourth, third], axis=1)
            corners[:, :, 2] = corners[:, :1, 2]
            corners = _ocs_to_wcs_each(np.nan_to_num(corners), _extrusions(f, rows))
        corners = np.nan_to_num(corners)
        quad = (corners[:, 2] != corners[:, 3]).any(axis=1)
        keep = np.ones((m, 4), dtype=bool)
        keep[:, 3] = quad
        sizes = 3 + quad.astype(np.int64)
        piece.add(
            corners[keep],
            np.arange(int(sizes.sum())),
            sizes,
            np.where(quad, _QUAD, _TRIANGLE),
            layers,
        )

    def _polyline(
        self,
        piece: _Piece,
        k: int,
        fields: dict[int | str, np.ndarray] | None,
        lo: int,
        hi: int,
    ) -> None:
        flags = int(self._float_field(k, 70, 0.0))
        layer = self._entity_layer(k)
        if fields is None or hi == lo:
            self._skip("POLYLINE without vertices")
            return
        v = {code: column[lo:hi] for code, column in fields.items()}
        raw_flags = np.nan_to_num(v[70])
        if raw_flags.size and np.abs(raw_flags).max() >= _INT64_BOUND:
            raise CodecError(
                f"{self.name!r}: a VERTEX gives flags"
                f" {np.abs(raw_flags).max():.6g}, past any integer group 70 holds."
            )
        vflags = raw_flags.astype(np.int64)
        coords = np.nan_to_num(np.column_stack([v[10], v[20], v[30]]))
        if flags & 64:
            self._polyface(piece, k, coords, vflags, v, layer)
            return
        if flags & 16:
            self._polygon_mesh(piece, k, coords, vflags, flags, layer)
            return
        keep = (vflags & 16) == 0
        coords = coords[keep]
        if not flags & 8:
            normal = self._entity_extrusion(k)
            coords[:, 2] = self._float_field(k, 30, 0.0)
            coords = _ocs_to_wcs(coords, normal)
            self.bulges += _bends(np.nan_to_num(v[42])[keep], closed=bool(flags & 1))
        self._add_polyline(piece, coords, bool(flags & 1), layer)

    def _add_polyline(
        self, piece: _Piece, coords: np.ndarray, closed: bool, layer: int
    ) -> None:
        n = len(coords)
        if n < 2:
            self._skip("polyline of one vertex")
            return
        conn = np.arange(n)
        if closed and n > 2:
            conn = np.append(conn, 0)
        kind = _LINE if len(conn) == 2 else _POLY_LINE
        piece.add(
            coords, conn, np.array([len(conn)]), np.array([kind]), np.array([layer])
        )

    def _polyface(
        self,
        piece: _Piece,
        k: int,
        coords: np.ndarray,
        vflags: np.ndarray,
        v: dict[int | str, np.ndarray],
        layer: int,
    ) -> None:
        is_face = ((vflags & 128) != 0) & ((vflags & 64) == 0)
        points = coords[~is_face]
        named = np.abs(
            np.nan_to_num(np.column_stack([v[c][is_face] for c in (71, 72, 73, 74)]))
        )
        # Checked before the cast: an index past int64 has no integer to become.
        if named.size and named.max() > len(points):
            raise CodecError(
                f"{self.name!r}: a polyface mesh face names vertex"
                f" {named.max():.15g} but the mesh holds {len(points)}."
            )
        indices = named.astype(np.int64)
        sizes = np.count_nonzero(indices, axis=1)
        nonzero = indices != 0
        keep = sizes > 0
        if not keep.all():
            self._skip("polyface face without vertices", count=int((~keep).sum()))
        indices, nonzero, sizes = indices[keep], nonzero[keep], sizes[keep]
        conn = indices[nonzero] - 1
        types = np.select(
            [sizes == 1, sizes == 2, sizes == 3], [_VERTEX, _LINE, _TRIANGLE], _QUAD
        ).astype(np.uint8)
        piece.add(points, conn, sizes, types, np.full(len(sizes), layer))

    def _polygon_mesh(
        self,
        piece: _Piece,
        k: int,
        coords: np.ndarray,
        vflags: np.ndarray,
        flags: int,
        layer: int,
    ) -> None:
        m = int(self._float_field(k, 71, 0.0))
        n = int(self._float_field(k, 72, 0.0))
        if int(self._float_field(k, 75, 0.0)):
            surface = (vflags & 16) == 0
            if surface.sum() != m * n:
                m = int(self._float_field(k, 73, 0.0))
                n = int(self._float_field(k, 74, 0.0))
            coords = coords[surface]
        if m < 1 or n < 1 or len(coords) != m * n:
            raise CodecError(
                f"{self.name!r}: a {_shown(m)} x {_shown(n)} polygon mesh holds"
                f" {len(coords)} vertices."
            )
        rows = m if flags & 1 else m - 1
        cols = n if flags & 32 else n - 1
        if rows < 1 or cols < 1:
            self._skip("polygon mesh of one row")
            return
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        i, j = i.ravel(), j.ravel()
        i1, j1 = (i + 1) % m, (j + 1) % n
        quads = np.column_stack([i * n + j, i * n + j1, i1 * n + j1, i1 * n + j])
        piece.add(
            coords,
            quads.ravel(),
            np.full(len(quads), 4),
            np.full(len(quads), _QUAD),
            np.full(len(quads), layer),
        )

    def _pairs(self, k: int) -> tuple[list[int], list[Any]]:
        start, stop = int(self.starts[k]), int(self.ends[k])
        return self.codes[start:stop].tolist(), self.values[start:stop]

    def _lwpolyline(self, piece: _Piece, k: int) -> None:
        codes, values = self._pairs(k)
        kind = "LWPOLYLINE"
        flags = 0
        elevation = 0.0
        normal = np.array([0.0, 0.0, 1.0])
        xs: list[float] = []
        ys: list[float] = []
        # The vertex each nonzero bulge follows; it bends the segment leaving it.
        bulged: list[int] = []
        layer = self._layer("0")
        for code, value in zip(codes, values, strict=True):
            if code == 10:
                xs.append(self._number(value, code, kind))
            elif code == 20:
                ys.append(self._number(value, code, kind))
            elif code == 38:
                elevation = self._number(value, code, kind)
            elif code == 70:
                flags = int(self._number(value, code, kind))
            elif code == 42:
                if self._number(value, code, kind) != 0 and xs:
                    bulged.append(len(xs) - 1)
            elif code in (210, 220, 230):
                normal[(code - 210) // 10] = self._number(value, code, kind)
            elif code == 8:
                layer = self._raw_layer(value)
        if len(xs) != len(ys):
            raise CodecError(
                f"{self.name!r}: an LWPOLYLINE spells {len(xs)} x but {len(ys)} y"
                " coordinates."
            )
        closed = bool(flags & 1)
        bulges = np.zeros(len(xs))
        bulges[bulged] = 1.0
        self.bulges += _bends(bulges, closed=closed)
        coords = (
            np.column_stack([xs, ys, np.full(len(xs), elevation)])
            if xs
            else np.empty((0, 3))
        )
        self._add_polyline(piece, _ocs_to_wcs(coords, normal), closed, layer)

    def _mesh(self, piece: _Piece, k: int) -> None:
        codes, values = self._pairs(k)
        kind = "MESH"
        layer = self._layer("0")
        coords: list[float] = []
        faces: list[int] = []
        state = ""
        declared = {"vertices": 0, "faces": 0}
        remaining = 0
        for code, value in zip(codes, values, strict=True):
            if state == "vertices" and code not in (10, 20, 30):
                break
            if state == "faces" and code != 90:
                break
            if code == 8:
                layer = self._raw_layer(value)
            elif code == 92 and not state:
                remaining = declared["vertices"] = int(self._number(value, code, kind))
                state = "vertices" if remaining else "listed"
            elif state == "vertices":
                coords.append(self._number(value, code, kind))
                if code == 30:
                    remaining -= 1
                    if remaining == 0:
                        state = "listed"
            elif code == 93 and state in ("", "listed"):
                remaining = declared["faces"] = int(self._number(value, code, kind))
                state = "faces" if remaining else "done"
            elif state == "faces":
                faces.append(int(self._number(value, code, kind)))
                remaining -= 1
                if remaining == 0:
                    state = "done"
            elif state == "done":
                break
        if state in ("vertices", "faces"):
            held = len(coords) // 3 if state == "vertices" else len(faces)
            what = "vertices" if state == "vertices" else "face list entries"
            raise CodecError(
                f"{self.name!r}: a MESH declares {_shown(declared[state])}"
                f" {what} but holds {held}."
            )
        if len(coords) % 3:
            raise CodecError(f"{self.name!r}: a MESH vertex is not three coordinates.")
        points = np.array(coords, dtype=np.float64).reshape(-1, 3)
        conn: list[int] = []
        sizes: list[int] = []
        pos = 0
        while pos < len(faces):
            size = faces[pos]
            face = faces[pos + 1 : pos + 1 + size]
            if size < 1 or len(face) != size:
                raise CodecError(f"{self.name!r}: a MESH face list is cut short.")
            conn.extend(face)
            sizes.append(size)
            pos += 1 + size
        if conn and (min(conn) < 0 or max(conn) >= len(points)):
            bad = max(conn) if max(conn) >= len(points) else min(conn)
            raise CodecError(
                f"{self.name!r}: a MESH face names vertex {_shown(bad)} but the"
                f" mesh holds {len(points)}."
            )
        size_arr = np.array(sizes, dtype=np.int64)
        types = np.select(
            [size_arr == 1, size_arr == 2, size_arr == 3, size_arr == 4],
            [_VERTEX, _LINE, _TRIANGLE, _QUAD],
            _POLYGON,
        ).astype(np.uint8)
        piece.add(
            points,
            np.array(conn, dtype=np.int64),
            size_arr,
            types,
            np.full(len(sizes), layer),
        )

    def _insert(self, piece: _Piece, k: int) -> None:
        block = self._text(self._field(k, 2) or b"").strip()
        key = block.upper()
        if key not in self.blocks:
            raise CodecError(
                f"{self.name!r}: an INSERT names block {block!r}, which the file lacks."
            )
        layer = self._entity_layer(k)
        inner = self._expand(key)
        vertices, conn, sizes, types, layers = inner.flat()
        if not len(sizes):
            return
        base = self.blocks[key][0]
        scale = np.array([self._float_field(k, c, 1.0) for c in (41, 42, 43)])
        angle = math.radians(self._float_field(k, 50, 0.0))
        cols = max(1, int(self._float_field(k, 70, 1.0)))
        rows = max(1, int(self._float_field(k, 71, 1.0)))
        col_step = self._float_field(k, 44, 0.0)
        row_step = self._float_field(k, 45, 0.0)
        at = np.array([self._float_field(k, c, 0.0) for c in (10, 20, 30)])
        normal = self._entity_extrusion(k)
        copies = cols * rows
        self.produced += copies * (len(vertices) + len(conn) + _COPY_COST)
        if self.produced > self.budget:
            raise CodecError(
                f"{self.name!r}: the inserts expand to more than {self.budget}"
                f" vertices and connectivity entries, each copy counted as"
                f" {_COPY_COST} more, past what a file this size can describe."
            )
        c, s = math.cos(angle), math.sin(angle)
        rotate = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        layers = np.where(layers == self._layer("0"), layer, layers)
        mirrored = np.prod(scale) < 0
        if mirrored:
            conn = _reversed_faces(conn, sizes, types)
        n = len(vertices)
        row, col = np.divmod(np.arange(copies), cols)
        offset = np.column_stack([col * col_step, row * row_step, np.zeros(copies)])
        with np.errstate(over="ignore", invalid="ignore"):
            local = (vertices - base) * scale
            placed = (local[None, :, :] + offset[:, None, :]).reshape(-1, 3)
            placed = _ocs_to_wcs(placed @ rotate.T + at, normal)
        if not np.isfinite(placed).all():
            raise CodecError(
                f"{self.name!r}: an INSERT of block {block!r} places a vertex past"
                " the float range."
            )
        piece.add(
            placed,
            (conn[None, :] + n * np.arange(copies)[:, None]).ravel(),
            np.tile(sizes, copies),
            np.tile(types, copies),
            np.tile(layers, copies),
        )

    def _expand(self, key: str) -> _Piece:
        """Return a block's geometry, its own inserts expanded, in the block's frame.

        The blocks it inserts, and theirs, are expanded first, depth first
        without recursion: a chain of nested blocks as long as a file can
        spell would overflow the Python stack.
        """
        path = [key]
        on_path = {key}
        pending = [self._inserted(key)]
        while path:
            if pending[-1]:
                child = pending[-1].pop()
                if child in on_path:
                    chain = " -> ".join((*path[path.index(child) :], child))
                    raise CodecError(f"{self.name!r}: block {chain} inserts itself.")
                if child not in self.expanded:
                    path.append(child)
                    on_path.add(child)
                    pending.append(self._inserted(child))
                continue
            done = path.pop()
            on_path.discard(done)
            pending.pop()
            if done not in self.expanded:
                self.expanded[done] = self._collect(self.blocks[done][1])
        return self.expanded[key]

    def _inserted(self, key: str) -> list[str]:
        """Return the blocks block ``key`` inserts and has yet to expand, last first.

        A name the file defines no block for is left to :meth:`_insert`, which
        refuses it with the entity at hand.
        """
        if key in self.expanded:
            return []
        out: list[str] = []
        for k in self.blocks[key][1]:
            if self.kinds[k] != "INSERT" or self.paper[k]:
                continue
            child = self._text(self._field(k, 2) or b"").strip().upper()
            if child in self.blocks and child not in self.expanded:
                out.append(child)
        return out[::-1]

    def _skip(self, what: str, *, count: int = 1) -> None:
        self.skipped[what] = self.skipped.get(what, 0) + count

    # -- fields -------------------------------------------------------------

    def _field(self, k: int, code: int) -> Any:
        start, stop = int(self.starts[k]), int(self.ends[k])
        hits = np.flatnonzero(self.codes[start:stop] == code)
        return self.values[start + int(hits[0])] if hits.size else None

    def _float_field(self, k: int, code: int, default: float) -> float:
        value = self._field(k, code)
        return default if value is None else self._number(value, code, self.kinds[k])

    def _entity_layer(self, k: int) -> int:
        value = self._field(k, 8)
        return self._layer("0") if value is None else self._raw_layer(value)

    def _entity_extrusion(self, k: int) -> np.ndarray:
        return np.array(
            [
                self._float_field(k, c, d)
                for c, d in ((210, 0.0), (220, 0.0), (230, 1.0))
            ]
        )

    def _raw_layer(self, value: Any) -> int:
        """Return the layer index of a group 8 value, parsed once per spelling."""
        index = self._layer_cache.get(value)
        if index is None:
            index = self._layer_cache[value] = self._layer(self._text(value).strip())
        return index

    def _layer(self, name: str) -> int:
        """Return the index of a layer, its name compared without case.

        DXF symbol names ignore case, so ``Walls`` and ``WALLS`` are one
        layer, named the way it is first spelled.
        """
        key = name.upper()
        index = self.layer_ids.get(key)
        if index is None:
            index = self.layer_ids[key] = len(self.layer_names)
            self.layer_names.append(name)
        return index

    def _text(self, value: Any) -> str:
        """Return a string value decoded, cut at its first NUL as a binary one is."""
        if isinstance(value, bytes):
            value = value.split(b"\x00", 1)[0]
            try:
                text = value.decode(self.encoding)
            except UnicodeDecodeError:
                if self.utf8_only:
                    self.misdecoded += 1
                text = value.decode("cp1252", errors="replace")
        else:
            text = str(value)
        if "\\U+" in text:
            text = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
            # A character past the BMP is escaped as its two UTF-16 halves.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        if "\\M+" in text:
            text = _MBCS_ESCAPE.sub(_mbcs_char, text)
        return text

    def _number(self, value: Any, code: Any, kind: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise CodecError(
                f"{self.name!r}: {kind} group {int(code)} should be a number, not"
                f" {value!r}."
            ) from None
        if not math.isfinite(number):
            raise CodecError(f"{self.name!r}: {kind} group {int(code)} is {value!r}.")
        return number

    def _numbers(self, values: list[Any], kind: str, codes: np.ndarray) -> np.ndarray:
        """Return ``values`` as float64, naming the first that is not a number."""
        if not values:
            return np.empty(0)
        try:
            if isinstance(values[0], bytes):
                fixed = _narrow(values)
                out = (
                    np.array([float(value) for value in values])
                    if fixed is None
                    else fixed.astype(np.float64)
                )
            else:
                out = np.array(values, dtype=np.float64)
        except ValueError:
            out = None
        if out is None or not np.isfinite(out).all():
            for value, code in zip(values, codes.tolist(), strict=True):
                self._number(value, code, kind)
        if out is None:
            raise CodecError(f"{self.name!r}: a {kind} value is not a number.")
        return out

    # -- result -------------------------------------------------------------

    def _assemble(self, piece: _Piece) -> PolyData:
        vertices, conn, sizes, types, layers = piece.flat()
        global_attrs: dict[str, Any] = {}
        if self.version is not None:
            global_attrs["dxf_version"] = self.version
        if self.unit_code:
            unit = _INSUNITS.get(self.unit_code)
            if unit is None:
                warn_caller(
                    f"{self.name!r}: $INSUNITS {self.unit_code} is not a unit DXF"
                    " defines; it is dropped."
                )
            else:
                global_attrs["unit"] = unit
        if len(vertices):
            unique, first, inverse = np.unique(
                vertices, axis=0, return_index=True, return_inverse=True
            )
            order = np.argsort(first, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            vertices = unique[order]
            conn = rank[inverse.ravel()][conn]
        widest = max(len(vertices), len(conn) + 1)
        index_dtype = np.int32 if widest <= np.iinfo(np.int32).max else np.int64
        offsets = np.zeros(len(sizes) + 1, dtype=index_dtype)
        np.cumsum(sizes, out=offsets[1:])
        # Layers in the order their first element comes.
        order = np.argsort(layers, kind="stable").astype(index_dtype)
        present, first_of, counts = np.unique(
            layers, return_index=True, return_counts=True
        )
        groups = np.split(order, np.cumsum(counts)[:-1])
        tags = {
            self.layer_names[present[g]]: groups[g]
            for g in np.argsort(first_of, kind="stable").tolist()
        }
        return PolyData(
            vertices=np.ascontiguousarray(vertices, dtype=np.float64).reshape(-1, 3),
            connectivity=conn.astype(index_dtype),
            offsets=offsets,
            element_types=types.astype(np.uint8),
            element_tags=tags,
            global_attrs=global_attrs,
        )


# The points a flat entity must spell, and how a skipped one is counted.
_REQUIRED: dict[str, tuple[tuple[int, ...], str]] = {
    "POINT": ((10,), "a location"),
    "LINE": ((10, 11), "a start or an end"),
    "3DFACE": ((10, 11, 12), "corner 1, 2 or 3"),
    "SOLID": ((10, 11, 12), "corner 1, 2 or 3"),
    "TRACE": ((10, 11, 12), "corner 1, 2 or 3"),
}
_FLAT_CODES: dict[str, list[int]] = {
    "POINT": [10, 20, 30],
    "LINE": [10, 20, 30, 11, 21, 31],
    "3DFACE": [10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33],
    "SOLID": [10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33, 210, 220, 230],
    "TRACE": [10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33, 210, 220, 230],
    "VERTEX": [10, 20, 30, 42, 70, 71, 72, 73, 74],
}


def _bends(bulges: np.ndarray, *, closed: bool) -> int:
    """Return how many segments of a polyline its vertices' bulges bend.

    A bulge bends the segment leaving its vertex, so the last vertex's
    bends a segment only when the polyline closes back to the first.
    """
    segments = bulges if closed else bulges[:-1]
    return int(np.count_nonzero(segments))


def _upper(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.split(b"\x00", 1)[0]
        return value.strip().decode("ascii", errors="replace").upper()
    return str(value).strip().upper()


def _mbcs_char(match: re.Match[str]) -> str:
    """Return the character a ``\\M+nXXXX`` escape spells, or the escape as read.

    An escape whose bytes its code page does not decode to one character is
    kept as written rather than guessed at.
    """
    data = bytes.fromhex(match.group(2)).lstrip(b"\x00") or b"\x00"
    try:
        char = data.decode(_MBCS_CODE_PAGES[match.group(1)])
    except UnicodeDecodeError:
        return match.group(0)
    return char if len(char) == 1 else match.group(0)


def _codepage(name: str, fallback: str) -> str:
    """Return the Python codec a ``$DWGCODEPAGE`` names, or ``fallback``."""
    m = re.fullmatch(r"(?i)ansi_(\d+)", name)
    if m is None:
        return fallback
    candidate = f"cp{m.group(1)}"
    try:
        _codecs.lookup(candidate)
    except LookupError:
        return fallback
    return candidate


def _extrusions(f: dict[int | str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Return the extrusion direction of each entity, a code not spelled at its default."""
    return np.column_stack(
        [
            np.nan_to_num(f[code][rows], nan=default)
            for code, default in ((210, 0.0), (220, 0.0), (230, 1.0))
        ]
    )


def _ocs_basis(normals: np.ndarray) -> np.ndarray:
    """Return the ``(n, 3, 3)`` rows x, y, z of each extrusion's object coordinates.

    The arbitrary axis algorithm: z is the extrusion, x is the world y axis
    crossed with it when it lies within 1/64 of the world z axis and the
    world z axis crossed with it otherwise, and y is z crossed with x. A zero
    extrusion stands for the world z axis.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    az = np.where(length > 0, normals / np.where(length > 0, length, 1.0), [0, 0, 1.0])
    near_z = (np.abs(az[:, 0]) < 1 / 64) & (np.abs(az[:, 1]) < 1 / 64)
    world = np.where(near_z[:, None], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    ax = np.cross(world, az)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    ay = np.cross(az, ax)
    return np.stack([ax, ay, az], axis=1)


def _ocs_to_wcs_each(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Return each entity's ``(n, k, 3)`` OCS points in world coordinates."""
    return np.einsum("nkj,nji->nki", points, _ocs_basis(normals))


def _shown(count: int) -> str:
    """Return a count read from the file as a message spells it.

    A count of any integer width is written whole; one only a float could
    have spelled - ``1e300`` - is written as that float, not 301 digits.
    """
    return str(count) if abs(count) < _SHOWN_WHOLE else f"{count:.6g}"


def _ocs_to_wcs(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return OCS ``points`` of one extrusion direction in world coordinates."""
    if np.array_equal(normal, [0.0, 0.0, 1.0]):
        return points
    return points @ _ocs_basis(normal)[0]


def _reversed_faces(
    conn: np.ndarray, sizes: np.ndarray, types: np.ndarray
) -> np.ndarray:
    """Return ``conn`` with every face's corners in reverse order."""
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    owner = np.repeat(np.arange(len(sizes)), sizes)
    flip = np.isin(types, list(_FACES))[owner]
    position = np.arange(len(conn))
    mirrored = offsets[owner] + offsets[owner + 1] - 1 - position
    return conn[np.where(flip, mirrored, position)]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class _Layers:
    """Each element's layer: an index into ``names``, layer ``0`` first."""

    __slots__ = ("index", "names")

    def __init__(self, names: list[str], index: np.ndarray) -> None:
        self.names = names
        self.index = index


def _layers(poly: PolyData) -> _Layers:
    """Return each element's layer, from the tag groups in their order.

    A DXF entity is on one layer, so an element in two groups goes with the
    first and the second is told so. Layer names ignore case, so groups whose
    names differ only in case - or come to after :func:`_layer_label` - are
    one layer, named as the first is. A group whose layer is ``0`` shares it
    with the untagged elements, and is told so.
    """
    n = len(poly.element_types)
    names = ["0"]
    labels = ["0"]
    slots = {"0": 0}
    index = np.zeros(n, dtype=np.int64)
    claimed = np.zeros(n, dtype=bool)
    shared: list[str] = []
    renamed: dict[str, str] = {}
    merged: dict[str, str] = {}
    on_zero: list[str] = []
    for label, members in (poly.element_tags or {}).items():
        if _CONTROL.search(label):
            raise CodecError(
                f".dxf: layer name {label!r} holds a control character a DXF"
                " line cannot carry."
            )
        rows = np.zeros(n, dtype=bool)
        rows[np.asarray(members, dtype=np.int64)] = True
        fresh = rows & ~claimed
        if rows.any() and fresh.sum() < rows.sum():
            shared.append(label)
        if not fresh.any():
            continue
        written = _layer_label(label)
        if written != label:
            renamed[label] = written
        slot = slots.get(written.upper())
        if slot is None:
            slot = slots[written.upper()] = len(names)
            names.append(written)
            labels.append(label)
        elif labels[slot] != label:
            merged[label] = names[slot]
        if slot == 0:
            on_zero.append(label)
        index[fresh] = slot
        claimed |= fresh
    if shared:
        warn_caller(
            f".dxf: element tag group(s) {shared} name elements an earlier group"
            " already holds; a DXF entity is on one layer, so those stayed with"
            " the first."
        )
    if renamed:
        warn_caller(
            f".dxf: layer name(s) {renamed} were rewritten: no DXF layer name"
            f" holds any of {_INVALID_LAYER_CHARS!r}, starts or ends with"
            " white space, or is empty."
        )
    if merged:
        warn_caller(
            f".dxf: element tag group(s) {merged} are written on the layer"
            " named on the right, an earlier group's: the names match once case"
            " is ignored, as DXF ignores it, and refused characters rewritten."
        )
    if on_zero and not claimed.all():
        warn_caller(
            f".dxf: element tag group(s) {on_zero} are written on layer 0, where"
            " the untagged elements go too; they read back as one group."
        )
    return _Layers(names, index)


def _layer_label(name: str) -> str:
    """Return ``name`` as a layer name every DXF reader takes back unchanged.

    The characters no DXF version allows in a layer name become ``_``, and
    the white space a reader strips from a value is dropped; a name left
    empty, which no layer table holds, is ``_``. Length, case and
    characters outside R12's ``A-Z 0-9 $ _ -`` are kept: readers since R13
    take them, R12 files in the wild carry them, and renaming every layer to
    suit R12 itself would lose the tag names a round trip exists to keep.
    """
    return _INVALID_LAYER.sub("_", name.strip()) or "_"


def _layer_name(name: str) -> str:
    """Return a layer name with every non-ASCII character as a ``\\U+XXXX`` escape."""
    return "".join(
        ch if ord(ch) < 128 else "".join(f"\\U+{u:04X}" for u in _utf16(ch))
        for ch in name
    )


def _utf16(ch: str) -> list[int]:
    data = ch.encode("utf-16-le")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]


# One run of elements written as one kind of entity on one layer: the kind,
# the layer index, the flat connectivity and its offsets.
_Run = tuple[str, int, np.ndarray, np.ndarray]


def _runs(poly: PolyData, layers: _Layers) -> list[_Run]:
    """Return the elements as runs of entities, in element order.

    Consecutive cells of one kind on one layer form a run, and the faces of a
    run share polyface meshes, so the elements come back in the order they
    were written. The fixed-size types are gathered in one pass per type,
    the ragged ones in one pass per type and vertex count.
    """
    types = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets, dtype=np.int64)
    connectivity = np.asarray(poly.connectivity, dtype=np.int64)
    # Each block: the source element of each cell, the cells' sizes, their
    # flat connectivity, and the entity kind they are written as.
    owners: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    conns: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    def block(rows: np.ndarray, pattern: np.ndarray | list[int], kind: int) -> None:
        pattern = np.asarray(pattern, dtype=np.int64).reshape(-1, np.shape(pattern)[-1])
        cells = connectivity[offsets[rows][:, None, None] + pattern[None]]
        owners.append(np.repeat(rows, len(pattern)))
        sizes.append(np.full(cells.shape[0] * cells.shape[1], pattern.shape[1]))
        conns.append(cells.ravel())
        kinds.append(np.full(cells.shape[0] * cells.shape[1], kind))

    fixed: dict[int, tuple[list[int], int]] = {
        _TRIANGLE: ([0, 1, 2], _FACE),
        _QUAD: ([0, 1, 2, 3], _FACE),
        _PIXEL: ([0, 1, 3, 2], _FACE),
        _LINE: ([0, 1], _LINE_KIND),
        _QUADRATIC_EDGE: ([0, 1], _LINE_KIND),
        _VERTEX: ([0], _POINT_KIND),
        **{
            code: (list(range(corners)), _FACE)
            for code, corners in QUADRATIC_SURFACE_CORNERS.items()
        },
    }
    for code, (corners, kind) in fixed.items():
        rows = np.flatnonzero(types == code)
        if rows.size:
            block(rows, corners, kind)
    poly_vertex = int((types == _POLY_VERTEX).sum())
    degenerate: dict[str, int] = {}
    lengths = np.diff(offsets)
    for code in (_POLYGON, _STRIP, _POLY_LINE, _POLY_VERTEX):
        rows = np.flatnonzero(types == code)
        for n in np.unique(lengths[rows]).tolist():
            pattern, kind = _ragged_cells(code, n)
            same = rows[lengths[rows] == n]
            if pattern is None:
                label = ELEMENT_TYPES_INV[code]
                degenerate[label] = degenerate.get(label, 0) + len(same)
            else:
                block(same, pattern, kind)
    drawn = set(fixed) | {_POLYGON, _STRIP, _POLY_LINE, _POLY_VERTEX}
    dropped = {
        ELEMENT_TYPES_INV.get(code, str(code)): int((types == code).sum())
        for code in np.unique(types).tolist()
        if code not in drawn
    }
    named = ", ".join(f"{n} {label}" for label, n in dropped.items())
    short = ", ".join(f"{n} {label}" for label, n in degenerate.items())
    if not owners:
        if dropped or degenerate:
            what = "; ".join(
                part for part in (named, short and f"too few vertices: {short}") if part
            )
            raise CodecError(f".dxf: DXF draws none of the mesh's elements: {what}.")
        return []
    if dropped:
        warn_caller(f".dxf: {named} element(s) have no DXF entity and were dropped.")
    if degenerate:
        warn_caller(
            f".dxf: {short} element(s) hold too few vertices to draw and were dropped."
        )
    if poly_vertex:
        warn_caller(
            f".dxf: {poly_vertex} poly_vertex element(s) were written as one"
            " POINT per vertex."
        )
    owner = np.concatenate(owners)
    size = np.concatenate(sizes).astype(np.int64)
    conn = np.concatenate(conns).astype(np.int64)
    kind = np.concatenate(kinds)
    order = np.argsort(owner, kind="stable")
    starts = np.concatenate([[0], np.cumsum(size)])[:-1][order]
    size, kind, owner = size[order], kind[order], owner[order]
    within = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
    conn = conn[np.repeat(starts, size) + within]
    layer = layers.index[owner]
    cell_offsets = np.concatenate([[0], np.cumsum(size)])
    breaks = np.flatnonzero((np.diff(kind) != 0) | (np.diff(layer) != 0)) + 1
    bounds = np.concatenate([[0], breaks, [len(kind)]]).tolist()
    runs: list[_Run] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        span = cell_offsets[lo : hi + 1]
        runs.append(
            (
                _KIND_NAMES[int(kind[lo])],
                int(layer[lo]),
                conn[span[0] : span[-1]],
                span - span[0],
            )
        )
    return runs


def _ragged_cells(code: int, n: int) -> tuple[np.ndarray | None, int]:
    """Return the cells an element of a ragged type and ``n`` vertices is written as.

    Parameters
    ----------
    code
        The element type: polygon, triangle strip, poly line or poly vertex.
    n
        The element's vertex count.

    Returns
    -------
    tuple[numpy.ndarray | None, int]
        The corners of each cell, one row per cell as positions within the
        element, and the entity kind they are written as; None for an element
        with too few vertices to draw.
    """
    if code == _POLY_VERTEX:
        return np.arange(n)[:, None], _POINT_KIND
    if code == _POLY_LINE and n == 2:
        return np.arange(2)[None], _LINE_KIND
    if code == _POLY_LINE and n > 2:
        return np.arange(n)[None], _POLYLINE_KIND
    if code == _POLYGON and 3 <= n <= 4:
        return np.arange(n)[None], _FACE
    if code == _POLYGON and n > 4:
        j = np.arange(1, n - 1)
        return np.column_stack([np.zeros_like(j), j, j + 1]), _FACE
    if code == _STRIP and n >= 3:
        j = np.arange(n - 2)
        odd = j % 2 == 1
        # Every other triangle of a strip turns the other way; swapping its
        # first two corners keeps all of them facing one way.
        return np.column_stack([j + odd, j + 1 - odd, j + 2]), _FACE
    return None, _FACE


_FACE, _LINE_KIND, _POLYLINE_KIND, _POINT_KIND = range(4)
_KIND_NAMES: tuple[str, ...] = ("face", "line", "polyline", "point")


def _header(out: list[str], poly: PolyData) -> None:
    out.append(f"  0\nSECTION\n  2\nHEADER\n  9\n$ACADVER\n  1\n{_R12_VERSION}\n")
    unit = (poly.global_attrs or {}).get("unit")
    code = _UNIT_CODES.get(unit) if isinstance(unit, str) else None
    if unit is not None and code is None:
        warn_caller(
            f".dxf: global_attrs['unit'] {unit!r} has no $INSUNITS code; the"
            " drawing is written unitless."
        )
    if code is not None:
        out.append(f"  9\n$INSUNITS\n 70\n{code}\n")
    out.append("  0\nENDSEC\n")


def _tables(out: list[str], names: list[str]) -> None:
    out.append(
        "  0\nSECTION\n  2\nTABLES\n"
        "  0\nTABLE\n  2\nLTYPE\n 70\n1\n"
        "  0\nLTYPE\n  2\nCONTINUOUS\n 70\n0\n  3\nSolid line\n 72\n65\n 73\n0\n 40\n0.0\n"
        "  0\nENDTAB\n"
        f"  0\nTABLE\n  2\nLAYER\n 70\n{len(names)}\n"
    )
    out.extend(
        f"  0\nLAYER\n  2\n{_layer_name(name)}\n 70\n0\n 62\n7\n  6\nCONTINUOUS\n"
        for name in names
    )
    out.append("  0\nENDTAB\n  0\nENDSEC\n")


def _xyz_lines(points: np.ndarray, first: int) -> list[str]:
    """Return each point as the three group code / value pairs from ``first``.

    Each point is spelled once, however many entities name it; ``repr`` is
    the shortest text that reads back as the same float.
    """
    head = f" {first}\n"
    mid_y = f"\n {first + 10}\n"
    mid_z = f"\n {first + 20}\n"
    return [f"{head}{x!r}{mid_y}{y!r}{mid_z}{z!r}\n" for x, y, z in points.tolist()]


def _entities(
    out: list[str],
    kind: str,
    layer: str,
    spelled: dict[int, list[str]],
    conn: list[int],
    offsets: list[int],
) -> None:
    """Spell one run of elements as R12 entities.

    Parameters
    ----------
    out
        The text written so far, extended in place.
    kind
        The entity kind of the run: ``point``, ``line``, ``polyline`` or
        ``face``.
    layer
        The layer name as written.
    spelled
        Each written point's coordinate pairs, from group code 10 and, when
        the run is lines, from 11.
    conn
        The run's connectivity, as indices into the lists of ``spelled``.
    offsets
        Where each cell of the run starts in ``conn``, and where the last
        one ends.
    """
    at = spelled[10]
    vertex = f"  0\nVERTEX\n  8\n{layer}\n"
    seqend = f"  0\nSEQEND\n  8\n{layer}\n"
    if kind == "point":
        head = f"  0\nPOINT\n  8\n{layer}\n"
        out.extend(head + at[v] for v in conn)
    elif kind == "line":
        head = f"  0\nLINE\n  8\n{layer}\n"
        to = spelled[11]
        out.extend(head + at[a] + to[b] for a, b in zip(conn[::2], conn[1::2]))
    elif kind == "polyline":
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            cell = conn[lo:hi]
            closed = len(cell) > 3 and cell[0] == cell[-1]
            if closed:
                cell = cell[:-1]
            out.append(
                f"  0\nPOLYLINE\n  8\n{layer}\n 66\n1\n 10\n0.0\n 20\n0.0\n 30\n0.0\n"
                f" 70\n{8 | (1 if closed else 0)}\n"
            )
            out.extend(vertex + at[v] + " 70\n32\n" for v in cell)
            out.append(seqend)
    else:
        face = f"{vertex} 10\n0.0\n 20\n0.0\n 30\n0.0\n 70\n128\n"
        spells: dict[int, str] = {}
        for first in range(0, len(offsets) - 1, _POLYFACE_FACES):
            bounds = offsets[first : first + _POLYFACE_FACES + 1]
            span = conn[bounds[0] : bounds[-1]]
            used = sorted(set(span))
            rank = {v: k for k, v in enumerate(used, start=1)}
            out.append(
                f"  0\nPOLYLINE\n  8\n{layer}\n 66\n1\n 10\n0.0\n 20\n0.0\n 30\n0.0\n"
                f" 70\n64\n 71\n{len(used)}\n 72\n{len(bounds) - 1}\n"
            )
            out.extend(vertex + at[v] + " 70\n192\n" for v in used)
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                spell = spells.get(hi - lo)
                if spell is None:
                    spell = spells[hi - lo] = face + "".join(
                        f" {71 + j}\n{{}}\n" for j in range(hi - lo)
                    )
                out.append(spell.format(*[rank[v] for v in conn[lo:hi]]))
            out.append(seqend)
