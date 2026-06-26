from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._gmsh import read, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3]]))]
    )


def test_roundtrip_tetra() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles() -> None:
    poly = _tri_mesh()
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 3
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_file_has_sections() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    text = open(tmp).read()
    assert "$MeshFormat" in text
    assert "$Nodes" in text
    assert "$Elements" in text


def test_missing_section_raises() -> None:
    bad = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False, mode="w") as f:
        f.write(bad)
        tmp = f.name
    with pytest.raises(CodecError):
        read(tmp)


def test_phys_tag_in_element_attrs() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "phys_tag" in poly2.element_attrs
    assert len(poly2.element_attrs["phys_tag"]) == 1
