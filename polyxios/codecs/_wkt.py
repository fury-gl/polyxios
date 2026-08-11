"""WKT (Well-Known Text) .wkt ASCII codec - read + write.

Supports POINT, MULTIPOINT, LINESTRING, MULTILINESTRING, POLYGON,
MULTIPOLYGON and GEOMETRYCOLLECTION, with the optional ``Z`` / ``M`` /
``ZM`` dimension suffixes.  Z values are preserved, 2D coordinates are
padded with ``z=0`` and M (measure) values are read and discarded.  An
EWKT ``SRID=<n>;`` prefix is accepted and dropped.

Polygon interior rings (holes) become separate ``polygon`` elements.
Ring membership is carried by two element attributes:

``wkt_polygon_id``
    Identifier shared by every ring of the same WKT polygon; ``-1`` for
    elements that are not part of a polygon.
``wkt_ring``
    ``0`` for an exterior ring and ``1..n`` for interior rings (holes);
    ``-1`` for elements that are not part of a polygon.

On write, rings are regrouped from those attribute values rather than
from element positions, so holes stay attached to their exterior ring
after elements are filtered or reordered.  Each element carrying
``wkt_ring == 0`` opens a new polygon, which keeps two meshes that were
merged - and therefore reuse the same identifiers - from being welded
into one geometry.

Two lossy write behaviours are worth knowing about:

* WKT has a single polygon and a single line geometry, so ``triangle``,
  ``quad`` and ``polygon`` all come back as ``polygon``, and ``line``
  comes back as ``poly_line``.
* Reading interns identical coordinates into one vertex, so geometries
  that touch share vertices instead of duplicating them.
"""

from collections.abc import Iterator
from pathlib import Path
import re
from typing import Any
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
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".wkt"

#: Element attribute holding the WKT polygon identifier of each ring.
POLYGON_ID_ATTR: str = "wkt_polygon_id"

#: Element attribute holding the ring index (0 = exterior, >0 = hole).
RING_INDEX_ATTR: str = "wkt_ring"

#: Nesting cap for GEOMETRYCOLLECTION - keeps corrupt input from blowing
#: the Python recursion limit.
MAX_NESTING_DEPTH: int = 32

# polyxios element → WKT geometry type (for write)
_POLYXIOS_TO_WKT: dict[str, str] = {
    "vertex": "POINT",
    "poly_line": "LINESTRING",
    "line": "LINESTRING",
    "polygon": "POLYGON",
    "triangle": "POLYGON",
    "quad": "POLYGON",
}

_GEOMETRY_KEYWORDS: frozenset[str] = frozenset(
    (
        "GEOMETRYCOLLECTION",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "POINT",
        "LINESTRING",
        "POLYGON",
    )
)

# ── Tokeniser / recursive-descent parser ─────────────────────────────────────

_NUMBER_PATTERN: str = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
#: ``#`` comments are only recognised at the start of a line; ``;`` and ``=``
#: exist for the EWKT ``SRID=<n>;`` prefix.
_TOKEN_RE = re.compile(rf"(?m)^[^\S\n]*#[^\n]*|[A-Za-z]+|[()]|,|;|=|{_NUMBER_PATTERN}")
_NUMBER_RE = re.compile(rf"{_NUMBER_PATTERN}\Z")


def _line_of(text: str, offset: int) -> int:
    """Return the 1-based line number of a character offset.

    Parameters
    ----------
    text
        WKT source text.
    offset
        Character offset into ``text``.

    Returns
    -------
    int
        Line number containing ``offset``.
    """
    return text.count("\n", 0, offset) + 1


def _iter_tokens(text: str) -> Iterator[tuple[str, int]]:
    """Yield ``(token, offset)`` pairs for WKT text, lazily.

    Tokens are keywords, ``(``, ``)``, ``,``, ``;``, ``=`` or numeric
    literals.  Comment lines are skipped and any non-whitespace character
    that cannot start a token is rejected rather than silently dropped.

    Parameters
    ----------
    text
        WKT source text.

    Yields
    ------
    tuple of (str, int)
        The token and its character offset, in source order.

    Raises
    ------
    CodecError
        On any character that cannot start a token.
    """
    pos = 0
    for match in _TOKEN_RE.finditer(text):
        gap = text[pos : match.start()]
        stripped = gap.strip()
        if stripped:
            bad = pos + gap.index(stripped[0])
            raise CodecError(
                f".wkt:{_line_of(text, bad)}: unexpected character {stripped[0]!r}."
            )
        pos = match.end()
        token = match.group()
        if token.lstrip().startswith("#"):
            continue
        yield token, match.start()

    trailing = text[pos:]
    stripped = trailing.strip()
    if stripped:
        bad = pos + trailing.index(stripped[0])
        raise CodecError(
            f".wkt:{_line_of(text, bad)}: unexpected character {stripped[0]!r}."
        )


def _split_keyword(token: str) -> tuple[str, str]:
    """Split a geometry token into its keyword and dimension suffix.

    Handles both the spaced form (``POINT Z``) and the glued form
    (``POINTZ``) emitted by some writers.

    Parameters
    ----------
    token
        Raw keyword token.

    Returns
    -------
    tuple of (str, str)
        Upper-case keyword and dimension suffix (``''``, ``'Z'``,
        ``'M'`` or ``'ZM'``).
    """
    keyword = token.upper()
    if keyword in _GEOMETRY_KEYWORDS:
        return keyword, ""
    for suffix in ("ZM", "Z", "M"):
        base = keyword[: -len(suffix)]
        if keyword.endswith(suffix) and base in _GEOMETRY_KEYWORDS:
            return base, suffix
    return keyword, ""


class _Parser:
    """Recursive-descent WKT parser.

    Collects parsed geometries into shared vertex / connectivity lists.
    Tokens are pulled lazily from the source text, so a malformed file is
    rejected without materialising a token list for the whole input.

    Parameters
    ----------
    text
        WKT source text.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._tokens = _iter_tokens(text)
        self._lookahead: tuple[str, int] | None = None
        self._offset = 0

        # Shared vertex deduplication
        self._vert_map: dict[tuple[float, float, float], int] = {}
        self._verts: list[list[float]] = []

        # CSR accumulation
        self._conn: list[int] = []
        self._offsets: list[int] = [0]
        self._types: list[int] = []

        # Per-element polygon ring bookkeeping
        self._polygon_ids: list[int] = []
        self._ring_indices: list[int] = []
        self._next_polygon_id = 0

        self._dropped_measures = False

    # ── helpers ───────────────────────────────────────────────────────────

    def _error(self, message: str) -> CodecError:
        """Return a CodecError tagged with the current source line."""
        return CodecError(f".wkt:{_line_of(self._text, self._offset)}: {message}")

    def _fill(self) -> None:
        if self._lookahead is None:
            self._lookahead = next(self._tokens, None)

    def _peek(self) -> str | None:
        self._fill()
        return None if self._lookahead is None else self._lookahead[0]

    def _eof_offset(self) -> int:
        """Return the offset of the last non-blank character of the source."""
        index = len(self._text) - 1
        while index >= 0 and self._text[index].isspace():
            index -= 1
        return max(index, 0)

    def _advance(self) -> str:
        self._fill()
        if self._lookahead is None:
            self._offset = self._eof_offset()
            raise self._error("unexpected end of input.")
        tok, self._offset = self._lookahead
        self._lookahead = None
        return tok

    def _expect(self, expected: str) -> None:
        tok = self._advance()
        if tok != expected:
            raise self._error(f"expected '{expected}', got '{tok}'.")

    def _at_end(self) -> bool:
        return self._peek() is None

    def _at_number(self) -> bool:
        tok = self._peek()
        return tok is not None and _NUMBER_RE.match(tok) is not None

    def _number(self) -> float:
        """Consume one token and return it as a finite float."""
        tok = self._advance()
        if _NUMBER_RE.match(tok) is None:
            raise self._error(f"expected a number, got '{tok}'.")
        return float(tok)

    def _maybe_empty(self) -> bool:
        """Consume an ``EMPTY`` keyword if present."""
        tok = self._peek()
        if tok is not None and tok.upper() == "EMPTY":
            self._advance()
            return True
        return False

    def _maybe_srid(self) -> None:
        """Consume an EWKT ``SRID=<n>;`` prefix if present."""
        tok = self._peek()
        if tok is not None and tok.upper() == "SRID":
            self._advance()
            self._expect("=")
            self._number()
            self._expect(";")

    def _intern_vertex(self, x: float, y: float, z: float) -> int:
        key = (x, y, z)
        idx = self._vert_map.get(key)
        if idx is not None:
            return idx
        if len(self._verts) >= MAX_SAFE_VERTICES:
            raise self._error(
                f"vertex count exceeds the safety cap {MAX_SAFE_VERTICES}."
            )
        idx = len(self._verts)
        self._vert_map[key] = idx
        self._verts.append([x, y, z])
        return idx

    def _add_element(
        self,
        etype_str: str,
        indices: list[int],
        *,
        polygon_id: int = -1,
        ring: int = -1,
    ) -> int:
        """Append one element and return its element index.

        Parameters
        ----------
        etype_str
            polyxios element type name.
        indices
            Vertex indices of the element.
        polygon_id
            WKT polygon identifier, or -1 when not part of a polygon.
        ring
            Ring index within the polygon, or -1 when not part of one.

        Returns
        -------
        int
            Index of the appended element.
        """
        if len(self._types) >= MAX_SAFE_ELEMENTS:
            raise self._error(
                f"element count exceeds the safety cap {MAX_SAFE_ELEMENTS}."
            )
        if len(self._conn) + len(indices) > MAX_SAFE_CONN:
            raise self._error(
                f"connectivity size exceeds the safety cap {MAX_SAFE_CONN}."
            )
        self._conn.extend(indices)
        self._offsets.append(self._offsets[-1] + len(indices))
        self._types.append(ELEMENT_TYPES[etype_str])
        self._polygon_ids.append(polygon_id)
        self._ring_indices.append(ring)
        return len(self._types) - 1

    # ── coordinate parsing ────────────────────────────────────────────────

    def _parse_coord(self, suffix: str) -> tuple[float, float, float]:
        """Parse one ``x y [z] [m]`` coordinate.

        Parameters
        ----------
        suffix
            Dimension suffix of the enclosing geometry (``''``, ``'Z'``,
            ``'M'`` or ``'ZM'``).

        Returns
        -------
        tuple of float
            The x, y and z values; z is 0.0 for 2D and M-only geometries.
        """
        x = self._number()
        y = self._number()
        z = 0.0

        if suffix == "Z":
            z = self._number()
        elif suffix == "M":
            self._number()
            self._dropped_measures = True
        elif suffix == "ZM":
            z = self._number()
            self._number()
            self._dropped_measures = True
        elif self._at_number():
            # Undeclared dimensions: a third value is z, a fourth is a measure.
            z = self._number()
            if self._at_number():
                self._number()
                self._dropped_measures = True

        return x, y, z

    def _parse_coord_list(self, suffix: str) -> list[int]:
        """Parse ``(x y [z], x y [z], ...)`` → list of vertex indices."""
        self._expect("(")
        indices: list[int] = []
        while True:
            x, y, z = self._parse_coord(suffix)
            indices.append(self._intern_vertex(x, y, z))
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")
        return indices

    # ── geometry parsers ──────────────────────────────────────────────────

    def _consume_dimension_suffix(self) -> str:
        """Consume a standalone Z / M / ZM token if present."""
        tok = self._peek()
        if tok is not None and tok.upper() in ("Z", "M", "ZM"):
            return self._advance().upper()
        return ""

    def _parse_geometry(self, suffix: str = "", depth: int = 0) -> None:
        """Parse one geometry from the current position.

        Parameters
        ----------
        suffix
            Dimension suffix inherited from an enclosing collection; an
            explicit suffix on this geometry takes precedence.
        depth
            Current GEOMETRYCOLLECTION nesting depth.
        """
        if depth > MAX_NESTING_DEPTH:
            raise self._error(
                f"geometry nesting deeper than {MAX_NESTING_DEPTH} levels."
            )

        keyword, own_suffix = _split_keyword(self._advance())
        if not own_suffix:
            own_suffix = self._consume_dimension_suffix()
        dim = own_suffix or suffix

        if keyword == "GEOMETRYCOLLECTION":
            self._parse_geometry_collection(dim, depth)
        elif keyword == "MULTIPOINT":
            self._parse_multipoint(dim)
        elif keyword == "MULTILINESTRING":
            self._parse_multilinestring(dim)
        elif keyword == "MULTIPOLYGON":
            self._parse_multipolygon(dim)
        elif keyword == "POINT":
            self._parse_point(dim)
        elif keyword == "LINESTRING":
            self._parse_linestring(dim)
        elif keyword == "POLYGON":
            self._parse_polygon(dim)
        else:
            raise self._error(f"unsupported geometry type '{keyword}'.")

    def _parse_point(self, suffix: str) -> None:
        if self._maybe_empty():
            return
        self._expect("(")
        x, y, z = self._parse_coord(suffix)
        idx = self._intern_vertex(x, y, z)
        self._add_element("vertex", [idx])
        self._expect(")")

    def _parse_linestring(self, suffix: str) -> None:
        if self._maybe_empty():
            return
        indices = self._parse_coord_list(suffix)
        if len(indices) < 2:
            raise self._error("LINESTRING must have at least 2 points.")
        self._add_element("poly_line", indices)

    def _parse_ring(self, suffix: str, *, what: str) -> list[int]:
        """Parse one polygon ring and strip its closing duplicate."""
        indices = self._parse_coord_list(suffix)
        # Some writers repeat the closing point more than once; strip every
        # trailing copy so a ring is always stored in its open form.
        while len(indices) > 1 and indices[0] == indices[-1]:
            indices.pop()
        # Vertices are interned, so distinct indices are distinct points:
        # a ring collapsed onto one or two points encloses no area.
        if len(indices) < 3 or len(set(indices)) < 3:
            raise self._error(f"POLYGON {what} must have at least 3 distinct points.")
        return indices

    def _parse_polygon(self, suffix: str) -> None:
        if self._maybe_empty():
            return
        self._expect("(")

        polygon_id = self._next_polygon_id
        self._next_polygon_id += 1

        ring_indices = self._parse_ring(suffix, what="exterior ring")
        self._add_element("polygon", ring_indices, polygon_id=polygon_id, ring=0)

        ring_num = 1
        while self._peek() == ",":
            self._advance()
            hole_indices = self._parse_ring(suffix, what="interior ring")
            self._add_element(
                "polygon", hole_indices, polygon_id=polygon_id, ring=ring_num
            )
            ring_num += 1

        self._expect(")")

    def _parse_multipoint(self, suffix: str) -> None:
        if self._maybe_empty():
            return
        self._expect("(")
        while True:
            # MULTIPOINT can use ((x y), (x y)) or (x y, x y) syntax
            if not self._maybe_empty():
                if self._peek() == "(":
                    self._expect("(")
                    x, y, z = self._parse_coord(suffix)
                    self._add_element("vertex", [self._intern_vertex(x, y, z)])
                    self._expect(")")
                else:
                    x, y, z = self._parse_coord(suffix)
                    self._add_element("vertex", [self._intern_vertex(x, y, z)])
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    def _parse_multilinestring(self, suffix: str) -> None:
        if self._maybe_empty():
            return
        self._expect("(")
        while True:
            if not self._maybe_empty():
                indices = self._parse_coord_list(suffix)
                if len(indices) < 2:
                    raise self._error(
                        "LINESTRING in MULTILINESTRING must have at least 2 points."
                    )
                self._add_element("poly_line", indices)
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    def _parse_multipolygon(self, suffix: str) -> None:
        if self._maybe_empty():
            return
        self._expect("(")
        while True:
            self._parse_polygon(suffix)
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    def _parse_geometry_collection(self, suffix: str, depth: int) -> None:
        if self._maybe_empty():
            return
        self._expect("(")
        while True:
            self._parse_geometry(suffix, depth + 1)
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    # ── top-level entry point ─────────────────────────────────────────────

    def parse_all(self) -> PolyData:
        """Parse all geometries and return a PolyData.

        Top-level geometries may be separated by whitespace or by commas,
        the shape produced by CSV and database exports.

        Returns
        -------
        PolyData
            Parsed mesh data.
        """
        while not self._at_end():
            self._maybe_srid()
            self._parse_geometry()
            if self._peek() == ",":
                self._advance()

        if self._dropped_measures:
            warnings.warn(
                ".wkt: M (measure) coordinates are not representable; discarded.",
                stacklevel=3,
            )

        if not self._verts:
            return _empty_polydata()

        element_attrs: dict[str, np.ndarray] = {}
        if self._next_polygon_id:
            element_attrs[POLYGON_ID_ATTR] = np.array(self._polygon_ids, dtype=np.int32)
            element_attrs[RING_INDEX_ATTR] = np.array(
                self._ring_indices, dtype=np.int32
            )

        return PolyData(
            vertices=np.array(self._verts, dtype=np.float64),
            connectivity=np.array(self._conn, dtype=np.int32),
            offsets=np.array(self._offsets, dtype=np.int32),
            element_types=np.array(self._types, dtype=np.uint8),
            element_attrs=element_attrs,
        )


def _empty_polydata() -> PolyData:
    """Return a PolyData with no vertices and no elements."""
    return PolyData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
    )


# ── Write helpers ────────────────────────────────────────────────────────────


def _fmt(value: float) -> str:
    """Format one coordinate with full float64 round-trip precision."""
    return repr(float(value))


def _fmt_coords(coords: np.ndarray, *, with_z: bool) -> str:
    """Format a coordinate array as a comma-separated WKT coordinate list."""
    if with_z:
        return ", ".join(f"{_fmt(c[0])} {_fmt(c[1])} {_fmt(c[2])}" for c in coords)
    return ", ".join(f"{_fmt(c[0])} {_fmt(c[1])}" for c in coords)


def _ring_points(coords: np.ndarray) -> np.ndarray:
    """Return a ring's points with its closing duplicates removed.

    An element may store a ring either open or explicitly closed.  The
    open form is what decides whether the ring is degenerate, since the
    reader strips the closing point again; trailing repeats of the first
    point are all removed so that writing stays idempotent.

    Parameters
    ----------
    coords
        Ring coordinates as stored by the element.

    Returns
    -------
    numpy.ndarray
        The ring without its closing duplicates.
    """
    end = len(coords)
    while end > 1 and np.array_equal(coords[0], coords[end - 1]):
        end -= 1
    return coords[:end]


def _is_usable_ring(coords: np.ndarray) -> bool:
    """Return True when an open ring has the 3 distinct points WKT needs.

    Parameters
    ----------
    coords
        Ring coordinates with the closing duplicate already removed.

    Returns
    -------
    bool
        True when the ring encloses an area and survives a read back.
    """
    seen: set[tuple[float, float, float]] = set()
    for point in coords:
        seen.add((float(point[0]), float(point[1]), float(point[2])))
        if len(seen) >= 3:
            return True
    return False


def _fmt_ring(coords: np.ndarray, *, with_z: bool) -> str:
    """Format an open polygon ring, appending its closing point."""
    closed = np.vstack([coords, coords[:1]])
    return f"({_fmt_coords(closed, with_z=with_z)})"


def _element_coords(poly: PolyData, index: int) -> np.ndarray:
    """Return the vertex coordinates of one element."""
    start = int(poly.offsets[index])
    end = int(poly.offsets[index + 1])
    return poly.vertices[poly.connectivity[start:end]]


def _mesh_has_z(poly: PolyData) -> bool:
    """Return True when any referenced vertex carries a non-zero z.

    The whole mesh shares one dimensionality so that a single file does
    not mix ``POINT`` and ``POINT Z`` geometries.

    Parameters
    ----------
    poly
        PolyData about to be written.

    Returns
    -------
    bool
        True when the file must be written with Z coordinates.
    """
    if not len(poly.vertices):
        return False
    z = (
        poly.vertices[poly.connectivity, 2]
        if len(poly.connectivity)
        else poly.vertices[:, 2]
    )
    finite = z[np.isfinite(z)]
    return bool(np.any(finite != 0.0))


def _as_int_attr(values: np.ndarray, *, name: str) -> np.ndarray | None:
    """Return an integer view of a ring attribute, or None if unusable."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    if np.issubdtype(arr.dtype, np.floating) and bool(np.isfinite(arr).all()):
        return arr.astype(np.int64)
    warnings.warn(
        f".wkt write: '{name}' is not an integer attribute; polygon holes ignored.",
        stacklevel=4,
    )
    return None


def _ring_attrs(poly: PolyData, n_elems: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the polygon-id / ring-index attributes, or None if unusable."""
    attrs = poly.element_attrs or {}
    polygon_ids = attrs.get(POLYGON_ID_ATTR)
    ring_indices = attrs.get(RING_INDEX_ATTR)
    if polygon_ids is None or ring_indices is None:
        return None
    if len(polygon_ids) != n_elems or len(ring_indices) != n_elems:
        warnings.warn(
            f".wkt write: '{POLYGON_ID_ATTR}'/'{RING_INDEX_ATTR}' length does not "
            "match the element count; polygon holes ignored.",
            stacklevel=3,
        )
        return None
    ids = _as_int_attr(polygon_ids, name=POLYGON_ID_ATTR)
    rings = _as_int_attr(ring_indices, name=RING_INDEX_ATTR)
    if ids is None or rings is None:
        return None
    return ids, rings


def _polygon_groups(
    poly: PolyData,
    ring_attrs: tuple[np.ndarray, np.ndarray],
    n_elems: int,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Group polygon rings that belong to the same WKT geometry.

    Rings are grouped by ``wkt_polygon_id`` regardless of their position,
    so a hole stays attached to its exterior ring after the elements have
    been filtered or reordered.  Within one identifier, every exterior
    ring (``wkt_ring == 0``) opens a new group, which keeps merged meshes
    that reuse the same identifiers from being welded into one geometry;
    holes seen before any exterior ring join the first group of their
    identifier, or form a group of their own when that identifier has no
    exterior ring left.

    Parameters
    ----------
    poly
        PolyData about to be written.
    ring_attrs
        The polygon-id and ring-index attribute arrays.
    n_elems
        Total element count.

    Returns
    -------
    tuple of (dict, dict)
        Mapping from anchor element index to its member element indices
        (exterior ring first), and mapping from element index to the
        anchor of the group it belongs to.
    """
    polygon_ids, ring_indices = ring_attrs

    members_of: dict[int, list[int]] = {}
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if _POLYXIOS_TO_WKT.get(name) != "POLYGON":
            continue
        polygon_id = int(polygon_ids[i])
        if polygon_id >= 0:
            members_of.setdefault(polygon_id, []).append(i)

    raw_groups: list[list[int]] = []
    for members in members_of.values():
        current: list[int] | None = None
        pending: list[int] = []
        for i in members:
            if int(ring_indices[i]) == 0:
                current = [i, *pending]
                pending = []
                raw_groups.append(current)
            elif current is None:
                pending.append(i)
            else:
                current.append(i)
        if pending:
            raw_groups.append(pending)

    groups: dict[int, list[int]] = {}
    anchor_of: dict[int, int] = {}
    for group in raw_groups:
        group.sort(key=lambda k: (int(ring_indices[k]), k))
        anchor = min(group)
        groups[anchor] = group
        for i in group:
            anchor_of[i] = anchor

    return groups, anchor_of


# ── Public API ────────────────────────────────────────────────────────────────


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a WKT file and return a PolyData.

    The file may contain one geometry per line, a single multi-line
    geometry, or comma-separated geometries.  Blank lines and lines
    starting with ``#`` are ignored.

    Parameters
    ----------
    path
        Path to the .wkt file.
    lazy
        Not supported for WKT - raises LazyReadError.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    LazyReadError
        Always, if lazy=True.
    CodecError
        On malformed or unsupported WKT.
    """
    if lazy:
        raise LazyReadError("WKT format does not support lazy reads (ASCII only).")

    # errors="replace" keeps a non-UTF-8 file inside the codec error
    # contract: the replacement character is rejected by the tokeniser.
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return _Parser(text).parse_all()


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a WKT file.

    Each element is written as one WKT geometry line, except polygon
    rings sharing a ``wkt_polygon_id`` which are written as a single
    POLYGON with interior rings.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    **opts
        Unused; accepted for API uniformity.
    """
    if opts:
        warnings.warn(
            f".wkt write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )

    lines: list[str] = []
    skipped_types: set[str] = set()
    skipped_degenerate = 0
    skipped_nonfinite = 0
    n_elems = len(poly.element_types)

    ring_attrs = _ring_attrs(poly, n_elems)
    if ring_attrs is None:
        groups: dict[int, list[int]] = {}
        anchor_of: dict[int, int] = {}
    else:
        groups, anchor_of = _polygon_groups(poly, ring_attrs, n_elems)

    has_z = _mesh_has_z(poly)
    suffix = " Z" if has_z else ""

    for i in range(n_elems):
        if anchor_of.get(i, i) != i:
            # Already written as an interior ring of an earlier polygon.
            continue

        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        geometry = _POLYXIOS_TO_WKT.get(name)
        if geometry is None:
            skipped_types.add(name or str(int(poly.element_types[i])))
            continue

        rings = [_element_coords(poly, k) for k in groups.get(i, [i])]
        if not all(np.isfinite(r).all() for r in rings):
            skipped_nonfinite += len(rings)
            continue

        coords = rings[0]

        if geometry == "POINT":
            if len(coords) < 1:
                skipped_degenerate += 1
            else:
                lines.append(f"POINT{suffix} ({_fmt_coords(coords[:1], with_z=has_z)})")
        elif geometry == "LINESTRING":
            if len(coords) < 2:
                skipped_degenerate += 1
            else:
                lines.append(
                    f"LINESTRING{suffix} ({_fmt_coords(coords, with_z=has_z)})"
                )
        else:
            # Rings are measured open: a ring stored with its closing
            # duplicate loses that point again when it is read back.
            open_rings = [_ring_points(r) for r in rings]
            if not _is_usable_ring(open_rings[0]):
                # A degenerate exterior ring drops the whole polygon: promoting
                # a hole to exterior would silently invert the geometry.
                skipped_degenerate += len(open_rings)
                continue
            usable = [open_rings[0]] + [r for r in open_rings[1:] if _is_usable_ring(r)]
            skipped_degenerate += len(open_rings) - len(usable)
            body = ", ".join(_fmt_ring(r, with_z=has_z) for r in usable)
            lines.append(f"POLYGON{suffix} ({body})")

    if skipped_types:
        warnings.warn(
            f".wkt write: element types {sorted(skipped_types)} are not "
            "representable in WKT; skipped.",
            stacklevel=2,
        )
    if skipped_degenerate:
        warnings.warn(
            f".wkt write: {skipped_degenerate} element(s) had too few points for "
            "their geometry type; skipped.",
            stacklevel=2,
        )
    if skipped_nonfinite:
        warnings.warn(
            f".wkt write: {skipped_nonfinite} element(s) had non-finite "
            "coordinates; skipped.",
            stacklevel=2,
        )

    text = "\n".join(lines) + "\n" if lines else ""
    Path(path).write_text(text, encoding="utf-8")
