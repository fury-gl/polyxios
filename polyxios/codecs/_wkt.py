"""WKT (Well-Known Text) .wkt ASCII codec - read + write.

Supports POINT, MULTIPOINT, LINESTRING, MULTILINESTRING, POLYGON,
MULTIPOLYGON and GEOMETRYCOLLECTION, with the optional ``Z`` / ``M`` /
``ZM`` dimension suffixes.  Z values are preserved, 2D coordinates are
padded with ``z=0`` and M (measure) values are read and discarded.

Polygon interior rings (holes) become separate ``polygon`` elements.
Ring membership is carried by two element attributes:

``wkt_polygon_id``
    Identifier shared by every ring of the same WKT polygon; ``-1`` for
    elements that are not part of a polygon.
``wkt_ring``
    ``0`` for an exterior ring and ``1..n`` for interior rings (holes);
    ``-1`` for elements that are not part of a polygon.

The association is carried by attribute values rather than by element
positions, so rings stay linked when elements are filtered, reordered or
merged.
"""

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
_TOKEN_RE = re.compile(rf"[A-Za-z]+|[()]|,|{_NUMBER_PATTERN}")
_NUMBER_RE = re.compile(rf"{_NUMBER_PATTERN}\Z")


def _tokenize(text: str) -> list[str]:
    """Split WKT text into a flat list of tokens.

    Tokens are keywords, ``(``, ``)``, ``,`` or numeric literals.  Any
    non-whitespace character that is not part of a token is rejected
    rather than silently dropped.

    Parameters
    ----------
    text
        WKT source text.

    Returns
    -------
    list of str
        The tokens, in source order.

    Raises
    ------
    CodecError
        On any character that cannot start a token.
    """
    tokens: list[str] = []
    pos = 0
    for match in _TOKEN_RE.finditer(text):
        gap = text[pos : match.start()].strip()
        if gap:
            raise CodecError(f".wkt: unexpected character {gap[0]!r}.")
        tokens.append(match.group())
        pos = match.end()
    trailing = text[pos:].strip()
    if trailing:
        raise CodecError(f".wkt: unexpected character {trailing[0]!r}.")
    return tokens


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

    Parameters
    ----------
    tokens
        Token list produced by :func:`_tokenize`.
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

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

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> str:
        if self.pos >= len(self.tokens):
            raise CodecError(".wkt: unexpected end of input.")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, expected: str) -> None:
        tok = self._advance()
        if tok != expected:
            raise CodecError(f".wkt: expected '{expected}', got '{tok}'.")

    def _at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def _at_number(self) -> bool:
        tok = self._peek()
        return tok is not None and _NUMBER_RE.match(tok) is not None

    def _number(self) -> float:
        """Consume one token and return it as a finite float."""
        tok = self._advance()
        if _NUMBER_RE.match(tok) is None:
            raise CodecError(f".wkt: expected a number, got '{tok}'.")
        return float(tok)

    def _maybe_empty(self) -> bool:
        """Consume an ``EMPTY`` keyword if present."""
        tok = self._peek()
        if tok is not None and tok.upper() == "EMPTY":
            self._advance()
            return True
        return False

    def _intern_vertex(self, x: float, y: float, z: float) -> int:
        key = (x, y, z)
        idx = self._vert_map.get(key)
        if idx is not None:
            return idx
        if len(self._verts) >= MAX_SAFE_VERTICES:
            raise CodecError(
                f".wkt: vertex count exceeds the safety cap {MAX_SAFE_VERTICES}."
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
            raise CodecError(
                f".wkt: element count exceeds the safety cap {MAX_SAFE_ELEMENTS}."
            )
        if len(self._conn) + len(indices) > MAX_SAFE_CONN:
            raise CodecError(
                f".wkt: connectivity size exceeds the safety cap {MAX_SAFE_CONN}."
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
            raise CodecError(
                f".wkt: geometry nesting deeper than {MAX_NESTING_DEPTH} levels."
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
            raise CodecError(f".wkt: unsupported geometry type '{keyword}'.")

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
            raise CodecError(".wkt: LINESTRING must have at least 2 points.")
        self._add_element("poly_line", indices)

    def _parse_ring(self, suffix: str, *, what: str) -> list[int]:
        """Parse one polygon ring and strip its closing duplicate."""
        indices = self._parse_coord_list(suffix)
        if len(indices) > 1 and indices[0] == indices[-1]:
            indices = indices[:-1]
        if len(indices) < 3:
            raise CodecError(f".wkt: POLYGON {what} must have at least 3 points.")
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
                    raise CodecError(
                        ".wkt: LINESTRING in MULTILINESTRING must have at least "
                        "2 points."
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

        Returns
        -------
        PolyData
            Parsed mesh data.
        """
        while not self._at_end():
            self._parse_geometry()

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


def _fmt_ring(coords: np.ndarray, *, with_z: bool) -> str:
    """Format a polygon ring, closing it if it is not already closed."""
    if len(coords) > 1 and np.array_equal(coords[0], coords[-1]):
        coords = coords[:-1]
    closed = np.vstack([coords, coords[:1]])
    return f"({_fmt_coords(closed, with_z=with_z)})"


def _element_coords(poly: PolyData, index: int) -> np.ndarray:
    """Return the vertex coordinates of one element."""
    start = int(poly.offsets[index])
    end = int(poly.offsets[index + 1])
    return poly.vertices[poly.connectivity[start:end]]


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
    return np.asarray(polygon_ids), np.asarray(ring_indices)


def _polygon_run_end(
    polygon_ids: np.ndarray, ring_indices: np.ndarray, start: int, n_elems: int
) -> int:
    """Return the exclusive end of the ring run belonging to one polygon.

    The run stops at the next exterior ring (``wkt_ring == 0``) so that
    two polygons that ended up with the same identifier - as happens when
    two meshes are merged - are not welded into a single geometry.

    Parameters
    ----------
    polygon_ids
        Per-element polygon identifiers.
    ring_indices
        Per-element ring indices.
    start
        Index of the first ring of the polygon.
    n_elems
        Total element count.

    Returns
    -------
    int
        Exclusive end index of the run.
    """
    end = start + 1
    while (
        end < n_elems
        and polygon_ids[end] == polygon_ids[start]
        and ring_indices[end] != 0
    ):
        end += 1
    return end


# ── Public API ────────────────────────────────────────────────────────────────


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a WKT file and return a PolyData.

    The file may contain one geometry per line, or a single multi-line
    geometry.  Blank lines and lines starting with ``#`` are ignored.

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

    text = Path(path).read_text(encoding="utf-8-sig")
    # Strip comment lines
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    joined = " ".join(lines)

    tokens = _tokenize(joined)
    if not tokens:
        return _empty_polydata()

    return _Parser(tokens).parse_all()


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

    i = 0
    while i < n_elems:
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        geometry = _POLYXIOS_TO_WKT.get(name)
        if geometry is None:
            skipped_types.add(name or str(int(poly.element_types[i])))
            i += 1
            continue

        # Gather the rings of a multi-ring polygon: a contiguous run of
        # elements sharing the same non-negative polygon id.
        end = i + 1
        if geometry == "POLYGON" and ring_attrs is not None:
            polygon_ids, ring_indices = ring_attrs
            if polygon_ids[i] >= 0:
                end = _polygon_run_end(polygon_ids, ring_indices, i, n_elems)
                order = sorted(range(i, end), key=lambda k: int(ring_indices[k]))
            else:
                order = [i]
        else:
            order = [i]

        rings = [_element_coords(poly, k) for k in order]
        if not all(np.isfinite(r).all() for r in rings):
            skipped_nonfinite += len(rings)
            i = end
            continue

        has_z = any(bool((r[:, 2] != 0.0).any()) for r in rings)
        suffix = " Z" if has_z else ""
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
        elif len(rings[0]) < 3:
            # A degenerate exterior ring drops the whole polygon: promoting a
            # hole to exterior would silently invert the geometry.
            skipped_degenerate += len(rings)
        else:
            usable = [rings[0]] + [r for r in rings[1:] if len(r) >= 3]
            skipped_degenerate += len(rings) - len(usable)
            body = ", ".join(_fmt_ring(r, with_z=has_z) for r in usable)
            lines.append(f"POLYGON{suffix} ({body})")

        i = end

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
