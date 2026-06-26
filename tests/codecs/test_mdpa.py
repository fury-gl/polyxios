from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._mdpa import read, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_tetra() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mdpa", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles() -> None:
    poly = _tri_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mdpa", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_file_has_begin_end_sections() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mdpa", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    text = Path(tmp).read_text()
    assert "Begin Nodes" in text
    assert "End Nodes" in text
    assert "Begin Elements" in text


def test_no_nodes_section_raises() -> None:
    bad = "Begin ModelPartData\nEnd ModelPartData\n"
    with tempfile.NamedTemporaryFile(suffix=".mdpa", delete=False, mode="w") as f:
        f.write(bad)
        tmp = f.name
    with pytest.raises(CodecError):
        read(tmp)


def test_element_class_name_written() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mdpa", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    text = Path(tmp).read_text()
    assert "Element3D4N" in text
