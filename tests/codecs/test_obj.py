from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._obj import read, write
from polyxios.exceptions import LazyReadError


def _synthetic_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_ascii() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == len(poly.element_types)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_vertex_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs={"normals": normals}
    )
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "normals" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["normals"], normals, atol=1e-6)


def test_element_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"material": np.array(["steel", "iron"], dtype=object)},
    )
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "material" in poly2.element_attrs


def test_unsupported_lazy() -> None:
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        f.write(b"# empty\n")
        tmp = f.name
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_multi_group_element_tags() -> None:
    """Element 0 in both 'inlet' and 'wall' — both must survive roundtrip."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={
            "inlet": np.array([0], dtype=np.int32),
            "wall": np.array([0, 1], dtype=np.int32),
        },
    )
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "inlet" in poly2.element_tags
    assert "wall" in poly2.element_tags
    assert 0 in poly2.element_tags["inlet"]
    assert 0 in poly2.element_tags["wall"]
