from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._stl import _HEADER_SIZE, read, write
from polyxios.exceptions import CodecError


def _tetrahedron() -> object:
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
        np.sort(poly2.vertices, axis=0),
        np.sort(poly.vertices, axis=0),
        atol=1e-6,
    )


def test_roundtrip_ascii(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "test.stl"
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(
        np.sort(poly2.vertices, axis=0),
        np.sort(poly.vertices, axis=0),
        atol=1e-6,
    )


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
    from polyxios.exceptions import LazyReadError

    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_write_no_triangles_raises(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 3, 2]]))])
    tmp = tmp_path / "test.stl"
    with pytest.raises(CodecError):
        write(poly, tmp)


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
