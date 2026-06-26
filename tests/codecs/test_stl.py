from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._stl import read, write
from polyxios.exceptions import CodecError


def _tetrahedron() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))],
    )


def test_roundtrip_binary() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(
        np.sort(poly2.vertices, axis=0),
        np.sort(poly.vertices, axis=0),
        atol=1e-6,
    )


def test_roundtrip_ascii() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(
        np.sort(poly2.vertices, axis=0),
        np.sort(poly.vertices, axis=0),
        atol=1e-6,
    )


def test_binary_file_is_binary() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    raw = open(tmp, "rb").read()
    # Binary STL: 80-byte header + 4-byte count + N*50 bytes
    assert len(raw) == 80 + 4 + 4 * 50


def test_ascii_file_is_text() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    text = open(tmp).read()
    assert text.startswith("solid polyxios")
    assert text.strip().endswith("endsolid polyxios")
    assert text.count("facet normal") == 4


def test_normals_stored_in_element_attrs() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    assert "normals" in poly2.element_attrs
    assert poly2.element_attrs["normals"].shape == (4, 3)


def test_merge_vertices_default() -> None:
    """Shared vertices should be merged on read."""
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp, merge_vertices=True)
    # tetrahedron has 4 unique vertices
    assert poly2.vertices.shape[0] == 4


def test_no_merge_vertices() -> None:
    """Without merging, each triangle gets its own 3 vertices."""
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp, merge_vertices=False)
    assert poly2.vertices.shape[0] == 4 * 3  # 4 tris * 3 verts each


def test_lazy_binary() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly_lazy = read(tmp, lazy=True)
    # lazy skips deduplication: 4 tris * 3 verts = 12 unmerged vertices
    assert len(poly_lazy.element_types) == 4
    assert poly_lazy.vertices.shape[0] == 12
    unique = np.unique(poly_lazy.vertices, axis=0)
    np.testing.assert_allclose(unique, np.unique(poly.vertices, axis=0), atol=1e-6)
    assert "normals" in poly_lazy.element_attrs


def test_lazy_ascii_raises() -> None:
    poly = _tetrahedron()
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    from polyxios.exceptions import LazyReadError

    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_write_no_triangles_raises() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 3, 2]]))])
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = f.name
    with pytest.raises(CodecError):
        write(poly, tmp)
