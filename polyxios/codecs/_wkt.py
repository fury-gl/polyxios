"""WKT (Well-Known Text) .wkt ASCII codec — read + write.

Supports POINT, MULTIPOINT, LINESTRING, MULTILINESTRING, POLYGON,
MULTIPOLYGON, and GEOMETRYCOLLECTION.  3D coordinates (Z suffix) are
preserved; 2D coordinates are padded with z=0.
"""

from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".wkt"

# ── WKT geometry type → polyxios element type mapping ────────────────────────

_WKT_TYPE_MAP: dict[str, str] = {
    "POINT": "vertex",
    "LINESTRING": "poly_line",
    "POLYGON": "polygon",
}

# polyxios element → WKT geometry type (for write)
_POLYXIOS_TO_WKT: dict[str, str] = {
    "vertex": "POINT",
    "poly_line": "LINESTRING",
    "line": "LINESTRING",
    "polygon": "POLYGON",
    "triangle": "POLYGON",
    "quad": "POLYGON",
}

# ── Tokeniser / recursive-descent parser ─────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Split WKT text into a flat list of tokens.

    Tokens are: uppercase keywords, ``(``, ``)``, ``,``, or numeric literals.
    """
    return re.findall(
        r"[A-Za-z]+|[()]|,|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",
        text,
    )


class _Parser:
    """Recursive-descent WKT parser.

    Collects parsed geometries into shared vertex / connectivity lists.
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

        # Element tags for polygon holes
        self._hole_tags: dict[str, list[int]] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, expected: str) -> None:
        tok = self._advance()
        if tok != expected:
            raise CodecError(f".wkt: expected '{expected}', got '{tok}'.")

    def _at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def _intern_vertex(self, x: float, y: float, z: float) -> int:
        key = (x, y, z)
        idx = self._vert_map.get(key)
        if idx is not None:
            return idx
        idx = len(self._verts)
        self._vert_map[key] = idx
        self._verts.append([x, y, z])
        return idx

    def _add_element(self, etype_str: str, indices: list[int]) -> int:
        """Append one element and return its element index."""
        self._conn.extend(indices)
        self._offsets.append(self._offsets[-1] + len(indices))
        self._types.append(ELEMENT_TYPES[etype_str])
        return len(self._types) - 1

    # ── coordinate parsing ────────────────────────────────────────────────

    def _parse_coord(self) -> tuple[float, float, float]:
        x = float(self._advance())
        y = float(self._advance())
        z = 0.0
        # Peek: if next token is numeric (not comma/paren/keyword), it's z
        nxt = self._peek()
        if nxt is not None and re.match(r"^[+-]?[\d.]", nxt):
            z = float(self._advance())
        return x, y, z

    def _parse_coord_list(self) -> list[int]:
        """Parse ``(x y [z], x y [z], ...)`` → list of vertex indices."""
        self._expect("(")
        indices: list[int] = []
        while True:
            x, y, z = self._parse_coord()
            indices.append(self._intern_vertex(x, y, z))
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")
        return indices

    # ── geometry parsers ──────────────────────────────────────────────────

    def _consume_dimension_suffix(self) -> None:
        """Consume optional Z / M / ZM suffix after geometry type keyword."""
        nxt = self._peek()
        if nxt is not None and nxt.upper() in ("Z", "M", "ZM"):
            self._advance()

    def _parse_geometry(self) -> None:
        """Parse one geometry from current position."""
        keyword = self._advance().upper()

        # Strip MULTI prefix to decide handler
        if keyword == "GEOMETRYCOLLECTION":
            self._consume_dimension_suffix()
            self._parse_geometry_collection()
        elif keyword == "MULTIPOINT":
            self._consume_dimension_suffix()
            self._parse_multipoint()
        elif keyword == "MULTILINESTRING":
            self._consume_dimension_suffix()
            self._parse_multilinestring()
        elif keyword == "MULTIPOLYGON":
            self._consume_dimension_suffix()
            self._parse_multipolygon()
        elif keyword == "POINT":
            self._consume_dimension_suffix()
            self._parse_point()
        elif keyword == "LINESTRING":
            self._consume_dimension_suffix()
            self._parse_linestring()
        elif keyword == "POLYGON":
            self._consume_dimension_suffix()
            self._parse_polygon()
        else:
            raise CodecError(f".wkt: unsupported geometry type '{keyword}'.")

    def _parse_point(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        self._expect("(")
        x, y, z = self._parse_coord()
        idx = self._intern_vertex(x, y, z)
        self._add_element("vertex", [idx])
        self._expect(")")

    def _parse_linestring(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        indices = self._parse_coord_list()
        if len(indices) < 2:
            raise CodecError(".wkt: LINESTRING must have at least 2 points.")
        self._add_element("poly_line", indices)

    def _parse_polygon(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        self._expect("(")
        # Exterior ring
        ring_indices = self._parse_coord_list()
        # Remove closing duplicate if first == last
        if len(ring_indices) > 1 and ring_indices[0] == ring_indices[-1]:
            ring_indices = ring_indices[:-1]
        if len(ring_indices) < 3:
            raise CodecError(".wkt: POLYGON exterior ring must have at least 3 points.")
        elem_idx = self._add_element("polygon", ring_indices)

        # Interior rings (holes)
        hole_num = 0
        while self._peek() == ",":
            self._advance()
            hole_indices = self._parse_coord_list()
            if len(hole_indices) > 1 and hole_indices[0] == hole_indices[-1]:
                hole_indices = hole_indices[:-1]
            hole_elem_idx = self._add_element("polygon", hole_indices)
            tag_name = f"hole_of_{elem_idx}_{hole_num}"
            self._hole_tags.setdefault(tag_name, []).append(hole_elem_idx)
            hole_num += 1

        self._expect(")")

    def _parse_multipoint(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        self._expect("(")
        while True:
            # MULTIPOINT can use ((x y), (x y)) or (x y, x y) syntax
            if self._peek() == "(":
                self._expect("(")
                x, y, z = self._parse_coord()
                idx = self._intern_vertex(x, y, z)
                self._add_element("vertex", [idx])
                self._expect(")")
            else:
                x, y, z = self._parse_coord()
                idx = self._intern_vertex(x, y, z)
                self._add_element("vertex", [idx])
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    def _parse_multilinestring(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        self._expect("(")
        while True:
            indices = self._parse_coord_list()
            if len(indices) < 2:
                raise CodecError(
                    ".wkt: LINESTRING in MULTILINESTRING must have at least 2 points."
                )
            self._add_element("poly_line", indices)
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    def _parse_multipolygon(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        self._expect("(")
        while True:
            self._parse_polygon()
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    def _parse_geometry_collection(self) -> None:
        if self._peek() is not None and self._peek().upper() == "EMPTY":
            self._advance()
            return
        self._expect("(")
        while True:
            self._parse_geometry()
            if self._peek() == ",":
                self._advance()
            else:
                break
        self._expect(")")

    # ── top-level entry point ─────────────────────────────────────────────

    def parse_all(self) -> PolyData:
        """Parse all geometries and return a PolyData."""
        while not self._at_end():
            self._parse_geometry()

        if not self._verts:
            return PolyData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                connectivity=np.array([], dtype=np.int32),
                offsets=np.array([0], dtype=np.int32),
                element_types=np.array([], dtype=np.uint8),
            )

        element_tags = {
            g: np.array(idxs, dtype=np.int32) for g, idxs in self._hole_tags.items()
        }

        return PolyData(
            vertices=np.array(self._verts, dtype=np.float64),
            connectivity=np.array(self._conn, dtype=np.int32),
            offsets=np.array(self._offsets, dtype=np.int32),
            element_types=np.array(self._types, dtype=np.uint8),
            element_tags=element_tags,
        )


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
        Not supported for WKT — raises LazyReadError.

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

    text = Path(path).read_text(encoding="utf-8")
    # Strip comment lines
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    joined = " ".join(lines)

    tokens = _tokenize(joined)
    if not tokens:
        return PolyData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
        )

    parser = _Parser(tokens)
    return parser.parse_all()


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a WKT file.

    Each element is written as one WKT geometry line.

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
    n_elems = len(poly.element_types)

    # Collect hole element indices so we skip them in the main loop
    # Also build a mapping: parent_elem_idx → list of hole elem indices
    hole_elem_indices: set[int] = set()
    parent_to_holes: dict[int, list[int]] = {}
    if poly.element_tags:
        for tag_name, tag_idxs in poly.element_tags.items():
            if not tag_name.startswith("hole_of_"):
                continue
            # tag format: hole_of_{parent}_{ring}
            parts = tag_name.split("_")
            # parts = ["hole", "of", parent_idx, ring_idx]
            parent_idx = int(parts[2])
            for idx in tag_idxs:
                hole_elem_indices.add(int(idx))
                parent_to_holes.setdefault(parent_idx, []).append(int(idx))

    for i in range(n_elems):
        if i in hole_elem_indices:
            continue

        etype = int(poly.element_types[i])
        name = ELEMENT_TYPES_INV.get(etype, "")
        start = int(poly.offsets[i])
        end = int(poly.offsets[i + 1])
        indices = poly.connectivity[start:end]
        coords = poly.vertices[indices]

        if name == "vertex":
            v = coords[0]
            if v[2] != 0.0:
                lines.append(f"POINT Z ({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})")
            else:
                lines.append(f"POINT ({v[0]:.10g} {v[1]:.10g})")

        elif name in ("line", "poly_line"):
            coord_str = ", ".join(
                f"{c[0]:.10g} {c[1]:.10g} {c[2]:.10g}"
                if c[2] != 0.0
                else f"{c[0]:.10g} {c[1]:.10g}"
                for c in coords
            )
            has_z = any(c[2] != 0.0 for c in coords)
            prefix = "LINESTRING Z" if has_z else "LINESTRING"
            if has_z:
                coord_str = ", ".join(
                    f"{c[0]:.10g} {c[1]:.10g} {c[2]:.10g}" for c in coords
                )
            lines.append(f"{prefix} ({coord_str})")

        elif name in ("triangle", "quad", "polygon"):
            has_z = any(c[2] != 0.0 for c in coords)

            def _fmt_ring(ring_coords: np.ndarray, force_z: bool) -> str:
                # Close the ring: append first point at end
                pts = list(ring_coords) + [ring_coords[0]]
                if force_z:
                    return ", ".join(
                        f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g}" for p in pts
                    )
                return ", ".join(f"{p[0]:.10g} {p[1]:.10g}" for p in pts)

            rings: list[str] = [f"({_fmt_ring(coords, has_z)})"]

            # Collect holes for this element
            for hole_ei in parent_to_holes.get(i, []):
                hs = int(poly.offsets[hole_ei])
                he = int(poly.offsets[hole_ei + 1])
                hole_coords = poly.vertices[poly.connectivity[hs:he]]
                hole_has_z = has_z or any(c[2] != 0.0 for c in hole_coords)
                rings.append(f"({_fmt_ring(hole_coords, hole_has_z)})")

            prefix = "POLYGON Z" if has_z else "POLYGON"
            lines.append(f"{prefix} ({', '.join(rings)})")

        else:
            warnings.warn(
                f".wkt: element type '{name}' not supported by WKT; skipping.",
                stacklevel=2,
            )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
