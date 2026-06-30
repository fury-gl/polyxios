from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._abaqus import read, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_file_has_node_element_keywords(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    text = Path(tmp).read_text()
    assert "*Node" in text
    assert "*Element" in text
    assert "C3D4" in text


def test_missing_node_section_raises(tmp_path) -> None:
    bad = "*Heading\n** no node section\n"
    tmp = tmp_path / "bad.inp"
    tmp.write_text(bad)
    with pytest.raises(CodecError):
        read(tmp)


def test_mixed_elements(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2


def test_continuation_line_element(tmp_path) -> None:
    # Real .inp files split long element rows across lines ending with ','
    inp = (
        "*Node\n"
        "1, 0.0, 0.0, 0.0\n"
        "2, 1.0, 0.0, 0.0\n"
        "3, 0.0, 1.0, 0.0\n"
        "4, 0.0, 0.0, 1.0\n"
        "*Element, type=C3D4\n"
        "1, 1,\n"
        "2, 3, 4\n"
    )
    tmp = tmp_path / "cont.inp"
    tmp.write_text(inp)
    poly = read(tmp)
    assert len(poly.element_types) == 1
    assert poly.connectivity.tolist() == [0, 1, 2, 3]


def test_write_unknown_type_id_raises(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([255], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="unknown element type id"):
        write(poly, tmp_path / "out.inp")


def test_write_unmapped_type_raises(tmp_path) -> None:
    # quadratic_tetra (id=21) is readable but has no write mapping
    verts = np.zeros((10, 3), dtype=np.float64)
    poly = PolyData(
        vertices=verts,
        connectivity=np.arange(10, dtype=np.int32),
        offsets=np.array([0, 10], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["quadratic_tetra"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="no write mapping"):
        write(poly, tmp_path / "out.inp")


def test_unrecognised_element_type_warns(tmp_path) -> None:
    inp = "*Node\n1, 0.0, 0.0, 0.0\n2, 1.0, 0.0, 0.0\n*Element, type=BOGUS99\n1, 1, 2\n"
    tmp = tmp_path / "warn.inp"
    tmp.write_text(inp)
    with pytest.warns(UserWarning, match="unrecognised"):
        read(tmp)


def test_2d_node_z_padding(tmp_path) -> None:
    inp = (
        "*Node\n"
        "1, 0.0, 0.0\n"
        "2, 1.0, 0.0\n"
        "3, 0.0, 1.0\n"
        "*Element, type=CPS3\n"
        "1, 1, 2, 3\n"
    )
    tmp = tmp_path / "flat.inp"
    tmp.write_text(inp)
    poly = read(tmp)
    np.testing.assert_array_equal(poly.vertices[:, 2], [0.0, 0.0, 0.0])
