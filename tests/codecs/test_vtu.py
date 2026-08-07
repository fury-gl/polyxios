from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._vtu import read, write
from polyxios.exceptions import LazyReadError


def _tet_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def test_roundtrip_ascii() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    assert len(poly2.element_types) == 1
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_binary() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_lazy() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)
    poly2 = read(tmp, lazy=False)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)


def test_vertex_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pressure = np.array([1.0, 2.0, 3.0, 4.0])
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"pressure": pressure},
    )
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert "pressure" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["pressure"], pressure, atol=1e-6)


def test_element_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    stress = np.array([42.0])
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_attrs={"stress": stress},
    )
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert "stress" in poly2.element_attrs
    np.testing.assert_allclose(poly2.element_attrs["stress"], stress, atol=1e-6)
