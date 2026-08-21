from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs._wkt import POLYGON_ID_ATTR, RING_INDEX_ATTR, read, write
from polyxios.exceptions import CodecError, LazyReadError


def _rings_of(poly, polygon_id: int) -> list[int]:
    """Return the element indices of one WKT polygon, exterior ring first."""
    ids = poly.element_attrs[POLYGON_ID_ATTR]
    rings = poly.element_attrs[RING_INDEX_ATTR]
    members = [i for i in range(len(ids)) if ids[i] == polygon_id]
    return sorted(members, key=lambda i: int(rings[i]))


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
    assert list(poly.element_attrs[POLYGON_ID_ATTR]) == [0, 0]
    assert list(poly.element_attrs[RING_INDEX_ATTR]) == [0, 1]
    assert _rings_of(poly, 0) == [0, 1]


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
        element_attrs={
            POLYGON_ID_ATTR: np.array([0, 0], dtype=np.int32),
            RING_INDEX_ATTR: np.array([0, 1], dtype=np.int32),
        },
    )
    tmp = tmp_path / "hole_rt.wkt"
    write(poly, tmp)
    assert tmp.read_text().count("POLYGON") == 1
    poly2 = read(tmp)
    assert sum(1 for t in poly2.element_types if t == ELEMENT_TYPES["polygon"]) == 2
    assert _rings_of(poly2, 0) == [0, 1]


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
    # Exterior: (35 10, 10 20, 15 40, 45 45) → 4 unique pts
    ext_n = poly.offsets[1] - poly.offsets[0]
    assert ext_n == 4
    # Hole: (20 30, 35 35, 30 20) → 3 unique pts
    ext_ei, hole_ei = _rings_of(poly, 0)
    assert ext_ei == 0
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
    assert list(poly.element_attrs[POLYGON_ID_ATTR]) == [0, 1, 1]
    assert list(poly.element_attrs[RING_INDEX_ATTR]) == [0, 0, 1]
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
        np.testing.assert_array_equal(
            poly2.element_types,
            poly.element_types,
            err_msg=f"element type mismatch for {name}",
        )
        np.testing.assert_array_equal(
            poly2.offsets, poly.offsets, err_msg=f"offsets mismatch for {name}"
        )
        np.testing.assert_array_equal(
            poly2.connectivity,
            poly.connectivity,
            err_msg=f"connectivity mismatch for {name}",
        )
        np.testing.assert_array_equal(
            poly2.vertices, poly.vertices, err_msg=f"vertex mismatch for {name}"
        )
        for attr in (POLYGON_ID_ATTR, RING_INDEX_ATTR):
            if attr in poly.element_attrs:
                np.testing.assert_array_equal(
                    poly2.element_attrs[attr],
                    poly.element_attrs[attr],
                    err_msg=f"{attr} mismatch for {name}",
                )


def test_wkt_all_samples_via_top_level(tmp_path: Path) -> None:
    """Every WKT sample must be readable via polyxios.read() top-level API."""
    import polyxios

    for name, content in _WKT_SAMPLES.items():
        src = _write_wkt(tmp_path, name, content)
        poly = polyxios.read(str(src))
        assert len(poly.element_types) > 0, f"empty result for {name}"


# ── Malformed input ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content",
    [
        "POINT (1 2",  # truncated
        "POINT ()",  # no coordinates
        "POINT (1 2))",  # unbalanced
        "POINT (1)",  # missing y
        "POINT (nan 2)",  # not a number
        "POINT (1; 2)",  # stray character
        "POINT (1 2) $$$",  # trailing junk
        "SRID=;POINT (1 2)",  # EWKT prefix without an SRID value
        "CIRCULARSTRING (1 1, 2 2, 3 1)",  # unsupported type
        "LINESTRING (0 0)",  # too few points
        "POLYGON ((0 0, 1 1, 0 0))",  # degenerate ring
        "POLYGON ((0 0, 1 0, 1 1, 0 0), (2 2, 3 3, 2 2))",  # degenerate hole
    ],
)
def test_malformed_raises_codec_error(tmp_path: Path, content: str) -> None:
    """Malformed WKT must raise CodecError, never a bare Python exception."""
    tmp = _write_wkt(tmp_path, "bad", content)
    with pytest.raises(CodecError):
        read(tmp)


# ── Dimension suffixes ───────────────────────────────────────────────────────


def test_measure_is_not_read_as_z(tmp_path: Path) -> None:
    """POINT M carries a measure, not a z value."""
    tmp = _write_wkt(tmp_path, "m", "POINT M (1 2 5)\n")
    with pytest.warns(UserWarning, match="measure"):
        poly = read(tmp)
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 0])


def test_zm_suffix(tmp_path: Path) -> None:
    """POINT ZM keeps z and drops the measure."""
    tmp = _write_wkt(tmp_path, "zm", "POINT ZM (1 2 3 4)\n")
    with pytest.warns(UserWarning, match="measure"):
        poly = read(tmp)
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 3])


def test_undeclared_fourth_value_is_a_measure(tmp_path: Path) -> None:
    """Four values without a suffix mean XYZM."""
    tmp = _write_wkt(tmp_path, "xyzm", "POINT (1 2 3 4)\n")
    with pytest.warns(UserWarning, match="measure"):
        poly = read(tmp)
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 3])


def test_glued_dimension_suffix(tmp_path: Path) -> None:
    """The glued POINTZ spelling is accepted."""
    tmp = _write_wkt(tmp_path, "glued", "POINTZ (1 2 3)\n")
    poly = read(tmp)
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 3])


def test_collection_propagates_suffix(tmp_path: Path) -> None:
    """A collection's Z suffix applies to its members."""
    tmp = _write_wkt(
        tmp_path,
        "gc_z",
        "GEOMETRYCOLLECTION Z (POINT (1 2 3), LINESTRING (0 0 1, 1 1 2))\n",
    )
    poly = read(tmp)
    assert len(poly.element_types) == 2
    assert poly.vertices[:, 2].tolist() == [3.0, 1.0, 2.0]


def test_deep_nesting_is_rejected(tmp_path: Path) -> None:
    """Deeply nested collections raise CodecError, not RecursionError."""
    depth = 5000
    tmp = _write_wkt(
        tmp_path,
        "deep",
        "GEOMETRYCOLLECTION (" * depth + "POINT (1 2)" + ")" * depth,
    )
    with pytest.raises(CodecError, match="nesting"):
        read(tmp)


def test_utf8_bom(tmp_path: Path) -> None:
    """A UTF-8 BOM does not break parsing."""
    tmp = tmp_path / "bom.wkt"
    tmp.write_text("POINT (1 2)\n", encoding="utf-8-sig")
    poly = read(tmp)
    assert len(poly.element_types) == 1


# ── Nested EMPTY geometries ──────────────────────────────────────────────────


def test_empty_members(tmp_path: Path) -> None:
    """EMPTY members inside multi-geometries are skipped, not parsed."""
    tmp = _write_wkt(
        tmp_path,
        "empties",
        "GEOMETRYCOLLECTION (MULTIPOINT (EMPTY, (1 2)), "
        "MULTILINESTRING (EMPTY, (0 0, 1 1)), "
        "MULTIPOLYGON (EMPTY, ((0 0, 1 0, 1 1, 0 0))))\n",
    )
    poly = read(tmp)
    types = [int(t) for t in poly.element_types]
    assert types == [
        ELEMENT_TYPES["vertex"],
        ELEMENT_TYPES["poly_line"],
        ELEMENT_TYPES["polygon"],
    ]


# ── Write-side behaviour ─────────────────────────────────────────────────────


def test_write_empty_polydata(tmp_path: Path) -> None:
    """Writing an empty mesh produces an empty file, not a blank line."""
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    tmp = tmp_path / "empty.wkt"
    write(poly, tmp)
    assert tmp.read_text() == ""
    assert len(read(tmp).element_types) == 0


def test_write_precision_is_exact(tmp_path: Path) -> None:
    """Coordinates round-trip bit-for-bit."""
    verts = np.array([[0.1234567890123456, 1e-17, 12345678.87654321]])
    poly = make_polydata(verts, [("vertex", np.array([[0]]))])
    tmp = tmp_path / "prec.wkt"
    write(poly, tmp)
    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_write_unsupported_type_warns_once(tmp_path: Path) -> None:
    """Unsupported element types produce a single aggregated warning."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts, [("tetra", np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]))]
    )
    tmp = tmp_path / "tetra.wkt"
    with pytest.warns(UserWarning, match="not representable") as record:
        write(poly, tmp)
    assert len(record) == 1
    assert tmp.read_text() == ""


def test_write_orphan_hole_is_kept(tmp_path: Path) -> None:
    """A hole whose exterior ring is gone is still written out."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("polygon", np.array([[0, 1, 2]]))],
        element_attrs={
            POLYGON_ID_ATTR: np.array([3], dtype=np.int32),
            RING_INDEX_ATTR: np.array([1], dtype=np.int32),
        },
    )
    tmp = tmp_path / "orphan.wkt"
    write(poly, tmp)
    assert tmp.read_text().count("POLYGON") == 1
    assert len(read(tmp).element_types) == 1


def test_write_ignores_mismatched_ring_attrs(tmp_path: Path) -> None:
    """Ring attributes of the wrong length are ignored with a warning."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("polygon", np.array([[0, 1, 2], [1, 2, 3]]))],
        element_attrs={
            POLYGON_ID_ATTR: np.array([0], dtype=np.int32),
            RING_INDEX_ATTR: np.array([0], dtype=np.int32),
        },
    )
    tmp = tmp_path / "mismatch.wkt"
    with pytest.warns(UserWarning, match="element count"):
        write(poly, tmp)
    assert tmp.read_text().count("POLYGON") == 2


def test_write_skips_non_finite(tmp_path: Path) -> None:
    """Non-finite coordinates cannot be expressed in WKT and are skipped."""
    verts = np.array([[0, 0, 0], [np.nan, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("vertex", np.array([[0], [1]]))])
    tmp = tmp_path / "nan.wkt"
    with pytest.warns(UserWarning, match="non-finite"):
        write(poly, tmp)
    assert tmp.read_text().strip() == "POINT (0.0 0.0)"


def test_write_closes_already_closed_ring(tmp_path: Path) -> None:
    """A ring stored with its closing duplicate is not closed twice."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("polygon", np.array([[0, 1, 2, 3]]))])
    tmp = tmp_path / "closed.wkt"
    write(poly, tmp)
    assert tmp.read_text().count(",") == 3
    poly2 = read(tmp)
    assert poly2.offsets[1] - poly2.offsets[0] == 3


def test_write_degenerate_exterior_drops_polygon(tmp_path: Path) -> None:
    """A hole is never promoted to exterior when the exterior is degenerate."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [5, 5, 0], [6, 5, 0], [6, 6, 0]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("polygon", np.array([[0, 1]])),
            ("polygon", np.array([[2, 3, 4]])),
        ],
        element_attrs={
            POLYGON_ID_ATTR: np.array([0, 0], dtype=np.int32),
            RING_INDEX_ATTR: np.array([0, 1], dtype=np.int32),
        },
    )
    tmp = tmp_path / "degenerate.wkt"
    with pytest.warns(UserWarning, match="too few points"):
        write(poly, tmp)
    assert tmp.read_text() == ""


def test_merged_polygons_are_not_welded(tmp_path: Path) -> None:
    """Two merged meshes reusing polygon id 0 stay two polygons."""
    from polyxios.transforms import merge

    src = _write_wkt(
        tmp_path,
        "hole_only",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))\n",
    )
    poly = read(src)
    merged = merge(poly, poly)
    assert list(merged.element_attrs[POLYGON_ID_ATTR]) == [0, 0, 0, 0]

    out = tmp_path / "merged.wkt"
    write(merged, out)
    assert out.read_text().count("POLYGON") == 2
    poly2 = read(out)
    assert list(poly2.element_attrs[RING_INDEX_ATTR]) == [0, 1, 0, 1]
    assert list(poly2.element_attrs[POLYGON_ID_ATTR]) == [0, 0, 1, 1]


def test_holes_survive_element_filtering(tmp_path: Path) -> None:
    """Ring attributes keep holes linked after a transform reorders elements."""
    from polyxios.transforms import filter_element_type

    tmp = _write_wkt(
        tmp_path,
        "mixed",
        "POINT (9 9)\nPOLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))\n",
    )
    poly = read(tmp)
    filtered = filter_element_type(poly, keep="polygon")
    assert len(filtered.element_types) == 2
    ids = filtered.element_attrs[POLYGON_ID_ATTR]
    rings = filtered.element_attrs[RING_INDEX_ATTR]
    assert list(ids) == [0, 0]
    assert list(rings) == [0, 1]

    out = tmp_path / "filtered.wkt"
    write(filtered, out)
    assert out.read_text().count("POLYGON") == 1


def _reorder(poly, order: list[int]):
    """Return a copy of poly with its elements in the given order."""
    from polyxios._types import PolyData

    conn: list[int] = []
    offsets = [0]
    for k in order:
        start, end = int(poly.offsets[k]), int(poly.offsets[k + 1])
        conn.extend(int(c) for c in poly.connectivity[start:end])
        offsets.append(offsets[-1] + (end - start))
    return PolyData(
        vertices=poly.vertices,
        connectivity=np.array(conn, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int32),
        element_types=np.array([poly.element_types[k] for k in order], dtype=np.uint8),
        element_attrs={
            key: np.asarray(values)[order] for key, values in poly.element_attrs.items()
        },
    )


def test_reordered_rings_stay_linked(tmp_path: Path) -> None:
    """A hole separated from its exterior ring is still written as a hole."""
    tmp = _write_wkt(
        tmp_path,
        "mixed",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))\nPOINT (9 9)\n",
    )
    poly = read(tmp)
    # exterior, point, hole - the rings are no longer adjacent
    shuffled = _reorder(poly, [0, 2, 1])

    out = tmp_path / "reordered.wkt"
    write(shuffled, out)
    assert out.read_text().count("POLYGON") == 1

    poly2 = read(out)
    assert list(poly2.element_attrs[POLYGON_ID_ATTR]) == [0, 0, -1]
    assert list(poly2.element_attrs[RING_INDEX_ATTR]) == [0, 1, -1]


def test_holes_survive_a_reversed_mesh(tmp_path: Path) -> None:
    """Reversing the element order keeps every polygon whole."""
    tmp = _write_wkt(
        tmp_path,
        "two",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))\n"
        "POLYGON ((20 0, 30 0, 30 10, 20 10, 20 0), (21 1, 22 1, 22 2, 21 1))\n",
    )
    poly = read(tmp)
    reversed_mesh = _reorder(poly, [3, 2, 1, 0])

    out = tmp_path / "reversed.wkt"
    write(reversed_mesh, out)
    assert out.read_text().count("POLYGON") == 2
    poly2 = read(out)
    assert list(poly2.element_attrs[RING_INDEX_ATTR]) == [0, 1, 0, 1]


def test_non_utf8_file_raises_codec_error(tmp_path: Path) -> None:
    """A non-UTF-8 file must not leak UnicodeDecodeError."""
    tmp = tmp_path / "latin1.wkt"
    tmp.write_bytes(b"POINT (1 2)\nPOINT (3 \xff)\n")
    with pytest.raises(CodecError):
        read(tmp)


def test_error_message_reports_the_line(tmp_path: Path) -> None:
    """Parse errors name the offending source line."""
    tmp = _write_wkt(tmp_path, "bad_line", "POINT (1 2)\nPOINT (3 4)\nPOINT (5 $)\n")
    with pytest.raises(CodecError, match=r"\.wkt:3:"):
        read(tmp)


def test_error_message_reports_the_line_for_syntax(tmp_path: Path) -> None:
    """A structural error also carries its line number."""
    tmp = _write_wkt(tmp_path, "bad_syntax", "POINT (1 2)\nLINESTRING (0 0\n")
    with pytest.raises(CodecError, match=r"\.wkt:2:"):
        read(tmp)


def test_comma_separated_geometries(tmp_path: Path) -> None:
    """Top-level geometries may be comma separated, as CSV exports emit."""
    tmp = _write_wkt(tmp_path, "csv", "POINT (1 2), POINT (3 4), LINESTRING (0 0, 1 1)")
    poly = read(tmp)
    assert len(poly.element_types) == 3
    assert poly.element_types[2] == ELEMENT_TYPES["poly_line"]


def test_ewkt_srid_prefix_is_accepted(tmp_path: Path) -> None:
    """An EWKT SRID prefix is dropped, not rejected."""
    tmp = _write_wkt(
        tmp_path, "ewkt", "SRID=4326;POINT (1 2)\nSRID=4326;LINESTRING (0 0, 1 1)\n"
    )
    poly = read(tmp)
    assert len(poly.element_types) == 2
    pt_idx = poly.connectivity[poly.offsets[0] : poly.offsets[1]]
    np.testing.assert_allclose(poly.vertices[pt_idx[0]], [1, 2, 0])


def test_comment_is_only_recognized_at_line_start(tmp_path: Path) -> None:
    """A stray # inside a geometry is an error, not a comment."""
    tmp = _write_wkt(tmp_path, "midline", "POINT (1 2) # trailing\n")
    with pytest.raises(CodecError):
        read(tmp)


def test_write_uses_one_dimensionality(tmp_path: Path) -> None:
    """A mesh with any non-zero z is written entirely in 3D."""
    verts = np.array([[0, 0, 0], [1, 1, 2]], dtype=np.float64)
    poly = make_polydata(verts, [("vertex", np.array([[0], [1]]))])
    tmp = tmp_path / "mixed_z.wkt"
    write(poly, tmp)
    assert tmp.read_text().splitlines() == [
        "POINT Z (0.0 0.0 0.0)",
        "POINT Z (1.0 1.0 2.0)",
    ]
    np.testing.assert_allclose(read(tmp).vertices, verts)


def test_write_ignores_non_integer_ring_attrs(tmp_path: Path) -> None:
    """Ring attributes that cannot be integers are ignored with a warning."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("polygon", np.array([[0, 1, 2], [1, 2, 3]]))],
        element_attrs={
            POLYGON_ID_ATTR: np.array([0.0, np.nan]),
            RING_INDEX_ATTR: np.array([0.0, 1.0]),
        },
    )
    tmp = tmp_path / "float_attrs.wkt"
    with pytest.warns(UserWarning, match="not an integer attribute"):
        write(poly, tmp)
    assert tmp.read_text().count("POLYGON") == 2


def test_collapsed_ring_is_rejected(tmp_path: Path) -> None:
    """A ring collapsed onto fewer than 3 distinct points is not a polygon."""
    tmp = _write_wkt(tmp_path, "collapsed", "POLYGON ((0 0, 0 0, 0 0, 0 0))\n")
    with pytest.raises(CodecError, match="distinct"):
        read(tmp)


def test_write_drops_collapsed_ring(tmp_path: Path) -> None:
    """A zero-area ring is skipped instead of being written back unreadable."""
    verts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("polygon", np.array([[0, 0, 0, 0]]))])
    tmp = tmp_path / "collapsed.wkt"
    with pytest.warns(UserWarning, match="too few points"):
        write(poly, tmp)
    assert tmp.read_text() == ""


def test_repeated_closing_point_is_stripped(tmp_path: Path) -> None:
    """A ring whose closing point is repeated keeps its open form."""
    tmp = _write_wkt(tmp_path, "closed_twice", "POLYGON ((0 0, 1 0, 1 1, 0 0, 0 0))\n")
    poly = read(tmp)
    assert poly.offsets[1] - poly.offsets[0] == 3


def test_write_read_is_idempotent(tmp_path: Path) -> None:
    """A second write → read cycle must not keep shaving ring points."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float64)
    # The element stores the ring already closed, twice over.
    poly = make_polydata(verts, [("polygon", np.array([[0, 1, 2, 3, 0]]))])
    first = tmp_path / "gen1.wkt"
    write(poly, first)
    poly2 = read(first)
    second = tmp_path / "gen2.wkt"
    write(poly2, second)
    poly3 = read(second)

    assert first.read_text() == second.read_text()
    np.testing.assert_array_equal(poly2.offsets, poly3.offsets)
    np.testing.assert_array_equal(poly2.connectivity, poly3.connectivity)
    np.testing.assert_array_equal(poly2.vertices, poly3.vertices)


# --- meshio #1382: the ISO surface family -----------------------------------


def _wkt(tmp_path: Path, text: str, name: str = "g.wkt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_issue_1382_tin_empty_parses_to_an_empty_mesh(tmp_path: Path) -> None:
    """An empty geometry is a valid one; refusing it fails on legal input."""
    poly = read(_wkt(tmp_path, "TIN EMPTY\n"))
    assert poly.vertices.shape == (0, 3)
    assert len(poly.element_types) == 0


def test_issue_1382_a_tin_reads_as_triangles(tmp_path: Path) -> None:
    path = _wkt(
        tmp_path,
        "TIN Z (((0 0 0, 1 0 0, 0 1 0, 0 0 0)), ((0 0 0, 1 0 0, 0 0 1, 0 0 0)))\n",
    )
    poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]] * 2
    assert poly.vertices.shape == (4, 3)


def test_issue_1382_a_triangle_reads_as_one_triangle(tmp_path: Path) -> None:
    poly = read(_wkt(tmp_path, "TRIANGLE ((0 0, 1 0, 0 1, 0 0))\n"))
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_issue_1382_a_polyhedralsurface_reads_as_its_patches(
    tmp_path: Path,
) -> None:
    path = _wkt(
        tmp_path,
        "POLYHEDRALSURFACE Z (((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0)),"
        " ((0 0 0, 1 0 0, 0 0 1, 0 0 0)))\n",
    )
    poly = read(path)
    assert len(poly.element_types) == 2
    assert poly.vertices.shape == (5, 3)


def test_issue_1382_a_tin_patch_with_four_points_is_refused(tmp_path: Path) -> None:
    """A TIN is triangles; a four-sided patch means the file is not one."""
    path = _wkt(tmp_path, "TIN (((0 0, 1 0, 1 1, 0 1, 0 0)))\n")
    with pytest.raises(CodecError, match="triangular patch"):
        read(path)


def test_issue_1382_a_triangle_with_a_hole_is_refused(tmp_path: Path) -> None:
    path = _wkt(
        tmp_path,
        "TRIANGLE ((0 0, 4 0, 0 4, 0 0), (1 1, 2 1, 1 2, 1 1))\n",
    )
    with pytest.raises(CodecError, match="interior ring"):
        read(path)


def test_issue_1382_a_tin_inside_a_collection_reads(tmp_path: Path) -> None:
    path = _wkt(
        tmp_path,
        "GEOMETRYCOLLECTION (POINT (5 5), TIN (((0 0, 1 0, 0 1, 0 0))))\n",
    )
    poly = read(path)
    assert ELEMENT_TYPES["triangle"] in poly.element_types.tolist()


def test_a_short_triangle_ring_names_the_geometry_it_came_from(
    tmp_path: Path,
) -> None:
    """A TRIANGLE reporting a POLYGON's complaint names a geometry not in the file."""
    path = _wkt(tmp_path, "TRIANGLE ((0 0, 1 0, 0 0))\n")
    with pytest.raises(CodecError, match="TRIANGLE ring"):
        read(path)
