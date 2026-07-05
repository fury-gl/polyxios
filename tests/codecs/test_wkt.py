from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs._wkt import read, write
from polyxios.exceptions import LazyReadError

# ── Inline WKT test data ─────────────────────────────────────────────────────

_SIMPLE_WKT = (
    "POINT (1 2)\nLINESTRING (0 0, 1 1, 2 0)\nPOLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))\n"
)

_WHITESPACED_WKT = (
    "\n"
    "  point  (  1   2  )\n"
    "\n"
    "  LineString  (  0   0  ,  1   1  ,  2   0  )\n"
    "\n"
    "  POLYGON  ( (  0   0  ,  4   0  ,  4   4  ,  0   4  ,  0   0  ) )\n"
    "\n"
)

# CSV-derived WKT strings — each maps to a geometry type label and its content.
_WKT_SAMPLES: dict[str, str] = {
    "point": "POINT (30 10)\n",
    "linestring": "LINESTRING (30 10, 10 30, 40 40)\n",
    "polygon": "POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))\n",
    "polygon_hole": (
        "POLYGON ((35 10, 10 20, 15 40, 45 45, 35 10),(20 30, 35 35, 30 20, 20 30))\n"
    ),
    "multipoint": "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))\n",
    "multilinestring": (
        "MULTILINESTRING ((10 10, 20 20, 10 40),(40 40, 30 30, 40 20, 30 10))\n"
    ),
    "multipolygon": (
        "MULTIPOLYGON (((30 20, 10 40, 45 40, 30 20)),"
        "((15 5, 40 10, 10 20, 5 10, 15 5)))\n"
    ),
    "multipolygon_hole": (
        "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)),"
        "((20 35, 45 20, 30 5, 10 10, 10 30, 20 35),"
        "(30 20, 20 25, 20 15, 30 20)))\n"
    ),
    "collection": (
        "GEOMETRYCOLLECTION(POLYGON((1 1,2 1,2 2,1 2,1 1)),"
        "POINT(2 3),LINESTRING(2 3,3 4))\n"
    ),
}


def _write_wkt(tmp_path: Path, name: str, content: str) -> Path:
    """Write WKT content to a temp file and return the path."""
    p = tmp_path / f"{name}.wkt"
    p.write_text(content, encoding="utf-8")
    return p


# ── simple / whitespaced tests ────────────────────────────────────────────────


def test_read_simple(tmp_path: Path) -> None:
    """Read simple WKT — one POINT, one LINESTRING, one POLYGON."""
    tmp = _write_wkt(tmp_path, "simple", _SIMPLE_WKT)
    poly = read(tmp)

    # POINT(1 2) → 1 vertex element
    # LINESTRING(0 0, 1 1, 2 0) → 1 poly_line element (3 pts)
    # POLYGON((0 0, 4 0, 4 4, 0 4, 0 0)) → 1 polygon element (4 unique pts)
    assert len(poly.element_types) == 3

    type_codes = list(poly.element_types)
    assert type_codes[0] == ELEMENT_TYPES["vertex"]
    assert type_codes[1] == ELEMENT_TYPES["poly_line"]
    assert type_codes[2] == ELEMENT_TYPES["polygon"]

    # POINT vertex at (1, 2, 0)
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 0])

    # LINESTRING has 3 points
    ls_idx = poly.connectivity[poly.offsets[1] : poly.offsets[2]]
    assert len(ls_idx) == 3

    # POLYGON has 4 unique vertices (closing dup removed)
    pg_idx = poly.connectivity[poly.offsets[2] : poly.offsets[3]]
    assert len(pg_idx) == 4


def test_read_whitespaced(tmp_path: Path) -> None:
    """Whitespaced WKT should produce identical PolyData to simple WKT."""
    tmp_s = _write_wkt(tmp_path, "simple", _SIMPLE_WKT)
    tmp_w = _write_wkt(tmp_path, "whitespaced", _WHITESPACED_WKT)
    poly_s = read(tmp_s)
    poly_w = read(tmp_w)

    np.testing.assert_allclose(poly_s.vertices, poly_w.vertices)
    np.testing.assert_array_equal(poly_s.connectivity, poly_w.connectivity)
    np.testing.assert_array_equal(poly_s.offsets, poly_w.offsets)
    np.testing.assert_array_equal(poly_s.element_types, poly_w.element_types)


# ── Core codec tests ─────────────────────────────────────────────────────────


def test_roundtrip(tmp_path: Path) -> None:
    """Write → read cycle preserves vertices and connectivity."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("polygon", np.array([[0, 1, 3, 2]]))])
    tmp = tmp_path / "roundtrip.wkt"
    write(poly, tmp)
    poly2 = read(tmp)

    assert len(poly2.element_types) == 1
    assert poly2.element_types[0] == ELEMENT_TYPES["polygon"]
    np.testing.assert_allclose(
        poly2.vertices[poly2.connectivity[poly2.offsets[0] : poly2.offsets[1]]],
        poly.vertices[poly.connectivity[poly.offsets[0] : poly.offsets[1]]],
        atol=1e-8,
    )


def test_unsupported_lazy(tmp_path: Path) -> None:
    """lazy=True must raise LazyReadError."""
    tmp = _write_wkt(tmp_path, "lazy", _SIMPLE_WKT)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_empty_geometry(tmp_path: Path) -> None:
    """A file with POINT EMPTY produces empty PolyData."""
    tmp = _write_wkt(tmp_path, "empty", "POINT EMPTY\n")
    poly = read(tmp)
    assert len(poly.element_types) == 0
    assert poly.vertices.shape == (0, 3)


def test_3d_coordinates(tmp_path: Path) -> None:
    """POINT Z preserves z value through read."""
    tmp = _write_wkt(tmp_path, "z", "POINT Z (1 2 3)\n")
    poly = read(tmp)
    assert len(poly.element_types) == 1
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 3])


def test_3d_roundtrip(tmp_path: Path) -> None:
    """3D coordinates survive write → read."""
    verts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    poly = make_polydata(verts, [("vertex", np.array([[0], [1]]))])
    tmp = tmp_path / "z_rt.wkt"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, verts, atol=1e-8)


def test_multipoint(tmp_path: Path) -> None:
    """MULTIPOINT creates multiple vertex elements."""
    tmp = _write_wkt(tmp_path, "mp", "MULTIPOINT ((1 2), (3 4), (5 6))\n")
    poly = read(tmp)
    assert len(poly.element_types) == 3
    assert all(t == ELEMENT_TYPES["vertex"] for t in poly.element_types)


def test_multipoint_flat_syntax(tmp_path: Path) -> None:
    """MULTIPOINT with flat syntax (no inner parens) also works."""
    tmp = _write_wkt(tmp_path, "mp_flat", "MULTIPOINT (1 2, 3 4)\n")
    poly = read(tmp)
    assert len(poly.element_types) == 2


def test_multilinestring(tmp_path: Path) -> None:
    """MULTILINESTRING creates multiple poly_line elements."""
    tmp = _write_wkt(tmp_path, "mls", "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3, 4 4))\n")
    poly = read(tmp)
    assert len(poly.element_types) == 2
    assert all(t == ELEMENT_TYPES["poly_line"] for t in poly.element_types)
    # First line: 2 pts, second: 3 pts
    assert poly.offsets[1] - poly.offsets[0] == 2
    assert poly.offsets[2] - poly.offsets[1] == 3


def test_multipolygon(tmp_path: Path) -> None:
    """MULTIPOLYGON creates multiple polygon elements."""
    tmp = _write_wkt(
        tmp_path,
        "mpg",
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))\n",
    )
    poly = read(tmp)
    assert sum(1 for t in poly.element_types if t == ELEMENT_TYPES["polygon"]) == 2


def test_polygon_with_hole(tmp_path: Path) -> None:
    """POLYGON with a hole stores exterior and hole as separate elements."""
    tmp = _write_wkt(
        tmp_path,
        "hole",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))\n",
    )
    poly = read(tmp)
    # 1 exterior + 1 hole = 2 polygon elements
    assert sum(1 for t in poly.element_types if t == ELEMENT_TYPES["polygon"]) == 2
    assert "hole_of_0_0" in poly.element_tags


def test_geometrycollection(tmp_path: Path) -> None:
    """GEOMETRYCOLLECTION with mixed types parsed correctly."""
    tmp = _write_wkt(
        tmp_path,
        "gc",
        "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))\n",
    )
    poly = read(tmp)
    assert len(poly.element_types) == 2
    assert poly.element_types[0] == ELEMENT_TYPES["vertex"]
    assert poly.element_types[1] == ELEMENT_TYPES["poly_line"]


def test_empty_file(tmp_path: Path) -> None:
    """An empty file returns empty PolyData."""
    tmp = _write_wkt(tmp_path, "empty", "")
    poly = read(tmp)
    assert len(poly.element_types) == 0
    assert poly.vertices.shape == (0, 3)


def test_comment_lines(tmp_path: Path) -> None:
    """Lines starting with # are ignored."""
    tmp = _write_wkt(tmp_path, "comment", "# this is a comment\nPOINT (5 6)\n# end\n")
    poly = read(tmp)
    assert len(poly.element_types) == 1
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [5, 6, 0])


def test_triangle_roundtrip(tmp_path: Path) -> None:
    """Triangles written as POLYGON and read back correctly."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tmp = tmp_path / "tri.wkt"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    # Read back as polygon (WKT doesn't distinguish triangle from polygon)
    assert poly2.element_types[0] == ELEMENT_TYPES["polygon"]
    pg_idx = poly2.connectivity[poly2.offsets[0] : poly2.offsets[1]]
    np.testing.assert_allclose(
        np.sort(poly2.vertices[pg_idx], axis=0),
        np.sort(verts, axis=0),
        atol=1e-8,
    )


def test_write_hole_roundtrip(tmp_path: Path) -> None:
    """Polygon with hole survives write → read roundtrip."""
    verts = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],  # exterior
            [1, 1, 0],
            [2, 1, 0],
            [2, 2, 0],  # hole
        ],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("polygon", np.array([[0, 1, 2, 3]])),
            ("polygon", np.array([[4, 5, 6]])),
        ],
        element_tags={"hole_of_0_0": np.array([1], dtype=np.int32)},
    )
    tmp = tmp_path / "hole_rt.wkt"
    write(poly, tmp)
    poly2 = read(tmp)
    assert "hole_of_0_0" in poly2.element_tags
    assert sum(1 for t in poly2.element_types if t == ELEMENT_TYPES["polygon"]) == 2


def test_registry_auto_discovery() -> None:
    """The .wkt codec must be auto-discovered by the registry."""
    from polyxios._registry import build_default_registry

    registry = build_default_registry()
    assert ".wkt" in registry
    assert callable(registry[".wkt"].read)
    assert callable(registry[".wkt"].write)


def test_top_level_read(tmp_path: Path) -> None:
    """polyxios.read() works with .wkt files via registry dispatch."""
    import polyxios

    tmp = _write_wkt(tmp_path, "simple", _SIMPLE_WKT)
    poly = polyxios.read(str(tmp))
    assert len(poly.element_types) == 3


# ── CSV-derived WKT sample tests ─────────────────────────────────────────────


def test_wkt_point(tmp_path: Path) -> None:
    """POINT (30 10)."""
    poly = read(_write_wkt(tmp_path, "point", _WKT_SAMPLES["point"]))
    assert len(poly.element_types) == 1
    assert poly.element_types[0] == ELEMENT_TYPES["vertex"]
    pt = poly.vertices[poly.connectivity[poly.offsets[0] : poly.offsets[1]][0]]
    np.testing.assert_allclose(pt, [30, 10, 0])


def test_wkt_linestring(tmp_path: Path) -> None:
    """LINESTRING (30 10, 10 30, 40 40)."""
    poly = read(_write_wkt(tmp_path, "linestring", _WKT_SAMPLES["linestring"]))
    assert len(poly.element_types) == 1
    assert poly.element_types[0] == ELEMENT_TYPES["poly_line"]
    n_pts = poly.offsets[1] - poly.offsets[0]
    assert n_pts == 3
    ls_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[ls_idx[0]], [30, 10, 0])
    np.testing.assert_allclose(poly.vertices[ls_idx[1]], [10, 30, 0])
    np.testing.assert_allclose(poly.vertices[ls_idx[2]], [40, 40, 0])


def test_wkt_polygon(tmp_path: Path) -> None:
    """POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))."""
    poly = read(_write_wkt(tmp_path, "polygon", _WKT_SAMPLES["polygon"]))
    assert len(poly.element_types) == 1
    assert poly.element_types[0] == ELEMENT_TYPES["polygon"]
    # 5-point ring with closing duplicate removed → 4 unique vertices
    n_pts = poly.offsets[1] - poly.offsets[0]
    assert n_pts == 4


def test_wkt_polygon_hole(tmp_path: Path) -> None:
    """POLYGON with exterior + 1 interior ring."""
    poly = read(_write_wkt(tmp_path, "polygon_hole", _WKT_SAMPLES["polygon_hole"]))
    # 1 exterior polygon + 1 hole polygon = 2 elements
    polygon_count = sum(1 for t in poly.element_types if t == ELEMENT_TYPES["polygon"])
    assert polygon_count == 2
    assert "hole_of_0_0" in poly.element_tags
    # Exterior: (35 10, 10 20, 15 40, 45 45) → 4 unique pts
    ext_n = poly.offsets[1] - poly.offsets[0]
    assert ext_n == 4
    # Hole: (20 30, 35 35, 30 20) → 3 unique pts
    hole_ei = int(poly.element_tags["hole_of_0_0"][0])
    hole_n = poly.offsets[hole_ei + 1] - poly.offsets[hole_ei]
    assert hole_n == 3


def test_wkt_multipoint(tmp_path: Path) -> None:
    """MULTIPOINT ((10 40), (40 30), (20 20), (30 10))."""
    poly = read(_write_wkt(tmp_path, "multipoint", _WKT_SAMPLES["multipoint"]))
    assert len(poly.element_types) == 4
    assert all(t == ELEMENT_TYPES["vertex"] for t in poly.element_types)
    # Verify all 4 points
    expected = [[10, 40, 0], [40, 30, 0], [20, 20, 0], [30, 10, 0]]
    for i, exp in enumerate(expected):
        idx = poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]]
        np.testing.assert_allclose(poly.vertices[idx[0]], exp)


def test_wkt_multilinestring(tmp_path: Path) -> None:
    """MULTILINESTRING — 2 linestrings."""
    poly = read(
        _write_wkt(tmp_path, "multilinestring", _WKT_SAMPLES["multilinestring"])
    )
    assert len(poly.element_types) == 2
    assert all(t == ELEMENT_TYPES["poly_line"] for t in poly.element_types)
    # First line: 3 points (10 10, 20 20, 10 40)
    assert poly.offsets[1] - poly.offsets[0] == 3
    # Second line: 4 points (40 40, 30 30, 40 20, 30 10)
    assert poly.offsets[2] - poly.offsets[1] == 4


def test_wkt_multipolygon(tmp_path: Path) -> None:
    """MULTIPOLYGON — 2 simple polygons, no holes."""
    poly = read(_write_wkt(tmp_path, "multipolygon", _WKT_SAMPLES["multipolygon"]))
    polygon_count = sum(1 for t in poly.element_types if t == ELEMENT_TYPES["polygon"])
    assert polygon_count == 2
    # First: (30 20, 10 40, 45 40) → triangle, 3 pts
    assert poly.offsets[1] - poly.offsets[0] == 3
    # Second: (15 5, 40 10, 10 20, 5 10) → 4 pts
    assert poly.offsets[2] - poly.offsets[1] == 4


def test_wkt_multipolygon_hole(tmp_path: Path) -> None:
    """MULTIPOLYGON — 2 polygons, second has a hole."""
    poly = read(
        _write_wkt(tmp_path, "multipolygon_hole", _WKT_SAMPLES["multipolygon_hole"])
    )
    # 1st polygon (3 pts) + 2nd polygon exterior (5 pts) + 2nd hole (3 pts) = 3 elements
    polygon_count = sum(1 for t in poly.element_types if t == ELEMENT_TYPES["polygon"])
    assert polygon_count == 3
    assert "hole_of_1_0" in poly.element_tags
    # First polygon: (40 40, 20 45, 45 30) → 3 pts
    assert poly.offsets[1] - poly.offsets[0] == 3
    # Second polygon exterior: (20 35, 45 20, 30 5, 10 10, 10 30) → 5 pts
    assert poly.offsets[2] - poly.offsets[1] == 5


def test_wkt_collection(tmp_path: Path) -> None:
    """GEOMETRYCOLLECTION(POLYGON, POINT, LINESTRING)."""
    poly = read(_write_wkt(tmp_path, "collection", _WKT_SAMPLES["collection"]))
    assert len(poly.element_types) == 3
    assert poly.element_types[0] == ELEMENT_TYPES["polygon"]
    assert poly.element_types[1] == ELEMENT_TYPES["vertex"]
    assert poly.element_types[2] == ELEMENT_TYPES["poly_line"]
    # Polygon: (1 1, 2 1, 2 2, 1 2) → 4 pts
    assert poly.offsets[1] - poly.offsets[0] == 4
    # Point: (2 3)
    pt_idx = poly.connectivity[poly.offsets[1] : poly.offsets[2]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [2, 3, 0])
    # Linestring: (2 3, 3 4) → 2 pts
    assert poly.offsets[3] - poly.offsets[2] == 2


def test_wkt_all_samples_roundtrip(tmp_path: Path) -> None:
    """Every WKT sample must survive a write → read roundtrip."""
    for name, content in _WKT_SAMPLES.items():
        src = _write_wkt(tmp_path, name, content)
        poly = read(src)
        out = tmp_path / f"{name}_rt.wkt"
        write(poly, out)
        poly2 = read(out)
        assert len(poly2.element_types) == len(poly.element_types), (
            f"roundtrip mismatch for {name}"
        )
        np.testing.assert_allclose(
            poly2.vertices,
            poly.vertices,
            atol=1e-8,
            err_msg=f"vertex mismatch for {name}",
        )


def test_wkt_all_samples_via_top_level(tmp_path: Path) -> None:
    """Every WKT sample must be readable via polyxios.read() top-level API."""
    import polyxios

    for name, content in _WKT_SAMPLES.items():
        src = _write_wkt(tmp_path, name, content)
        poly = polyxios.read(str(src))
        assert len(poly.element_types) > 0, f"empty result for {name}"
