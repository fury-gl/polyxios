from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._types import PolyData
from polyxios.codecs._stl import _BINARY_FACET_SIZE, _HEADER_SIZE, read, write
from polyxios.exceptions import CodecError, LazyReadError


def _sort_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(a.T[::-1])]


def _tetrahedron() -> PolyData:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))],
    )


def test_roundtrip_binary(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(
        _sort_rows(poly2.vertices),
        _sort_rows(poly.vertices),
        atol=1e-6,
    )


def test_roundtrip_ascii(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(
        _sort_rows(poly2.vertices),
        _sort_rows(poly.vertices),
        atol=1e-6,
    )
    assert poly2.connectivity.shape == poly.connectivity.shape


def test_binary_file_is_binary(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    with open(tmp, "rb") as f:
        raw = f.read()
    # Binary STL: 80-byte header + 4-byte count + N*50 bytes
    assert len(raw) == 80 + 4 + 4 * 50


def test_ascii_file_is_text(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=False)
    with open(tmp) as f:
        text = f.read()
    assert text.startswith("solid polyxios")
    assert text.strip().endswith("endsolid polyxios")
    assert text.count("facet normal") == 4


def test_normals_stored_in_element_attrs(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    assert "normals" in poly2.element_attrs
    assert poly2.element_attrs["normals"].shape == (4, 3)


def test_merge_vertices_default(tmp_path: Path) -> None:
    """Shared vertices should be merged on read."""
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    poly2 = read(tmp, merge_vertices=True)
    # tetrahedron has 4 unique vertices
    assert poly2.vertices.shape[0] == 4


def test_no_merge_vertices(tmp_path: Path) -> None:
    """Without merging, each triangle gets its own 3 vertices."""
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    poly2 = read(tmp, merge_vertices=False)
    assert poly2.vertices.shape[0] == 4 * 3  # 4 tris * 3 verts each
    assert poly2.connectivity.shape == (4 * 3,)


def test_lazy_binary(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    poly_lazy = read(tmp, lazy=True)
    # lazy skips deduplication: 4 tris * 3 verts = 12 unmerged vertices
    assert len(poly_lazy.element_types) == 4
    assert poly_lazy.vertices.shape[0] == 12
    unique = np.unique(poly_lazy.vertices, axis=0)
    np.testing.assert_allclose(unique, np.unique(poly.vertices, axis=0), atol=1e-6)
    assert "normals" in poly_lazy.element_attrs


def test_lazy_ascii_raises(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=False)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_write_no_triangles_raises(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 3, 2]]))])
    tmp = tmp_path / "test.stl"
    with pytest.raises(CodecError):
        write(poly, tmp)


def test_degenerate_triangle_normals(tmp_path: Path) -> None:
    """Zero-area triangle must not produce NaN or zero normal."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
    )
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    normals = poly2.element_attrs["normals"]
    assert not np.any(np.isnan(normals)), "NaN normal produced for degenerate triangle"
    norms = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_malformed_ascii_too_many_vertices(tmp_path: Path) -> None:
    """ASCII STL with more than 3 vertices per facet must raise CodecError."""
    bad_stl = (
        b"solid bad\n"
        b"  facet normal 0 0 1\n"
        b"    outer loop\n"
        b"      vertex 0 0 0\n"
        b"      vertex 1 0 0\n"
        b"      vertex 0 1 0\n"
        b"      vertex 0 0 1\n"
        b"    endloop\n"
        b"  endfacet\n"
        b"endsolid bad\n"
    )
    tmp = tmp_path / "bad.stl"
    tmp.write_bytes(bad_stl)
    with pytest.raises(CodecError):
        read(tmp)


def test_malformed_ascii_too_few_vertices(tmp_path: Path) -> None:
    """ASCII STL with fewer than 3 vertices per facet must raise CodecError."""
    bad_stl = (
        b"solid bad\n"
        b"  facet normal 0 0 1\n"
        b"    outer loop\n"
        b"      vertex 0 0 0\n"
        b"      vertex 1 0 0\n"
        b"    endloop\n"
        b"  endfacet\n"
        b"endsolid bad\n"
    )
    tmp = tmp_path / "bad.stl"
    tmp.write_bytes(bad_stl)
    with pytest.raises(CodecError):
        read(tmp)


def test_read_empty_binary_stl(tmp_path: Path) -> None:
    """Binary STL with n_tris=0 must return empty PolyData."""
    stl_file = tmp_path / "empty.stl"
    stl_file.write_bytes(b"\x00" * _HEADER_SIZE + np.array(0, dtype="<u4").tobytes())
    poly = read(str(stl_file))
    assert len(poly.element_types) == 0
    assert poly.vertices.shape == (0, 3)


def test_lazy_binary_stored_normals_returned_as_is(tmp_path: Path) -> None:
    """Lazy read returns STL-embedded normals unchanged (may be all-zero)."""
    poly = _tetrahedron()
    stl_file = tmp_path / "test.stl"
    write(poly, str(stl_file), binary=True)
    raw = bytearray(stl_file.read_bytes())
    data_start = _HEADER_SIZE + 4
    for i in range(4):
        raw[
            data_start + i * _BINARY_FACET_SIZE : data_start
            + i * _BINARY_FACET_SIZE
            + 12
        ] = b"\x00" * 12
    stl_file.write_bytes(bytes(raw))
    poly_lazy = read(str(stl_file), lazy=True)
    normals = poly_lazy.element_attrs["normals"]
    assert normals.shape == (4, 3)
    assert np.all(normals == 0.0)


def test_binary_with_solid_header(tmp_path: Path) -> None:
    """Binary STL whose 80-byte header starts with 'solid' must not be misdetected as ASCII."""
    poly = _tetrahedron()
    stl_file = tmp_path / "test.stl"
    write(poly, str(stl_file), binary=True)
    raw = stl_file.read_bytes()
    solid_hdr = b"solid looks_ascii_but_binary" + b"\x00" * (
        _HEADER_SIZE - len(b"solid looks_ascii_but_binary")
    )
    stl_file.write_bytes(solid_hdr + raw[_HEADER_SIZE:])
    poly2 = read(str(stl_file))
    assert len(poly2.element_types) == 4
    poly_lazy = read(str(stl_file), lazy=True)
    assert len(poly_lazy.element_types) == 4


# --- meshio #1355: binary STL colours ----------------------------------------


def _binary_stl(facets: list[tuple[np.ndarray, int]], header: bytes = b"") -> bytes:
    """Return a binary STL holding the given (3x3 vertices, attribute) facets."""
    out = bytearray(header.ljust(80, b"\x00")[:80])
    out += np.array([len(facets)], dtype="<u4").tobytes()
    for verts, attr in facets:
        out += np.zeros(3, dtype="<f4").tobytes()
        out += np.asarray(verts, dtype="<f4").tobytes()
        out += np.array([attr], dtype="<u2").tobytes()
    return bytes(out)


_TRI_A = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
_TRI_B = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 1]])


def _rgb15(r: int, g: int, b: int) -> int:
    """Pack 5-bit components the VisCAM way: red high, blue low, valid bit set."""
    return 0x8000 | (r << 10) | (g << 5) | b


def _magics15(r: int, g: int, b: int) -> int:
    """Pack them the Magics way: red low, blue high, top bit clear when owned."""
    return (b << 10) | (g << 5) | r


def test_issue_1355_binary_attribute_bytes_read_as_colours(tmp_path) -> None:
    """The attribute word is where a binary STL keeps its per-facet colour."""
    path = tmp_path / "colour.stl"
    path.write_bytes(
        _binary_stl([(_TRI_A, _rgb15(31, 0, 0)), (_TRI_B, _rgb15(0, 31, 31))])
    )
    poly = read(path)
    colors = poly.element_attrs["colors"]
    assert colors.shape == (2, 3)
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(colors[1], [0.0, 1.0, 1.0], atol=1e-6)


def test_issue_1355_facets_without_the_valid_bit_carry_no_colour(tmp_path) -> None:
    """A zero attribute word is what a writer leaves when it has no colour."""
    path = tmp_path / "plain.stl"
    path.write_bytes(_binary_stl([(_TRI_A, 0), (_TRI_B, 0)]))
    assert "colors" not in read(path).element_attrs


def test_issue_1355_a_partly_coloured_file_keeps_the_facets_that_have_one(
    tmp_path,
) -> None:
    """An uncoloured facet in a coloured file must not take the colours along."""
    path = tmp_path / "mixed.stl"
    path.write_bytes(_binary_stl([(_TRI_A, _rgb15(31, 31, 31)), (_TRI_B, 0)]))
    colors = read(path).element_attrs["colors"]
    np.testing.assert_allclose(colors[0], [1.0, 1.0, 1.0], atol=1e-6)
    assert np.isnan(colors[1]).all()


def test_issue_1355_colours_survive_a_round_trip(tmp_path) -> None:
    path = tmp_path / "src.stl"
    path.write_bytes(
        _binary_stl([(_TRI_A, _rgb15(31, 0, 0)), (_TRI_B, _rgb15(0, 0, 31))])
    )
    poly = read(path)
    out = tmp_path / "back.stl"
    write(poly, out, binary=True)
    np.testing.assert_allclose(
        read(out).element_attrs["colors"], poly.element_attrs["colors"], atol=1e-6
    )


def test_issue_1355_ascii_output_drops_colours_with_a_warning(tmp_path) -> None:
    """ASCII STL has no field for a colour; dropping one silently hides it."""
    path = tmp_path / "src.stl"
    path.write_bytes(_binary_stl([(_TRI_A, _rgb15(31, 0, 0))]))
    poly = read(path)
    with pytest.warns(UserWarning, match="colors"):
        write(poly, tmp_path / "ascii.stl", binary=False)


# --- meshio #1470: STL is per-facet, so a conversion needs dedup -------------


def test_issue_1470_coincident_facet_corners_are_merged(tmp_path) -> None:
    """STL repeats a shared corner per facet; keeping them explodes a .ply."""
    path = tmp_path / "shared.stl"
    path.write_bytes(_binary_stl([(_TRI_A, 0), (_TRI_B, 0)]))
    poly = read(path)
    # Six facet corners, four distinct points.
    assert poly.vertices.shape == (4, 3)
    assert len(poly.element_types) == 2


def test_issue_1470_the_merge_can_be_turned_off(tmp_path) -> None:
    path = tmp_path / "shared.stl"
    path.write_bytes(_binary_stl([(_TRI_A, 0), (_TRI_B, 0)]))
    assert read(path, merge_vertices=False).vertices.shape == (6, 3)


def test_issue_1355_a_magics_header_switches_the_colour_convention(tmp_path) -> None:
    """Magics runs red in the low bits and clears the top bit to claim a colour."""
    path = tmp_path / "magics.stl"
    path.write_bytes(
        _binary_stl(
            [(_TRI_A, _magics15(31, 0, 0)), (_TRI_B, _magics15(0, 0, 31))],
            header=b"COLOR=\xff\xff\xff\xff,MATERIAL=solid",
        )
    )
    colors = read(path).element_attrs["colors"]
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(colors[1], [0.0, 0.0, 1.0], atol=1e-6)


def test_issue_1355_a_magics_facet_that_disowns_its_colour_carries_none(
    tmp_path,
) -> None:
    """Magics sets the top bit to say the facet takes the part's colour."""
    path = tmp_path / "inherit.stl"
    path.write_bytes(
        _binary_stl(
            [(_TRI_A, _magics15(31, 0, 0)), (_TRI_B, 0x8000)],
            header=b"COLOR=\xff\xff\xff\xff",
        )
    )
    colors = read(path).element_attrs["colors"]
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.isnan(colors[1]).all()


def test_issue_1355_an_integer_colour_column_counts_to_255(tmp_path) -> None:
    """A uint8 colour scaled as 0..1 saturates and writes a white mesh."""
    path = tmp_path / "src.stl"
    path.write_bytes(_binary_stl([(_TRI_A, _rgb15(31, 0, 0))]))
    poly = read(path)
    poly.element_attrs["colors"] = np.array([[255, 0, 0]], dtype=np.uint8)
    out = tmp_path / "int.stl"
    write(poly, out, binary=True)
    np.testing.assert_allclose(
        read(out).element_attrs["colors"][0], [1.0, 0.0, 0.0], atol=1e-6
    )


def test_issue_1355_lazy_reads_see_the_same_colours(tmp_path) -> None:
    """The lazy path reads the same words and must read them the same way."""
    path = tmp_path / "lazy.stl"
    path.write_bytes(_binary_stl([(_TRI_A, _rgb15(31, 0, 0))]))
    np.testing.assert_allclose(
        read(path, lazy=True).element_attrs["colors"],
        read(path).element_attrs["colors"],
    )
