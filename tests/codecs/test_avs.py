from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES_INV
from polyxios.codecs._avs import read, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.avs"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    assert ELEMENT_TYPES_INV[int(poly2.element_types[0])] == "tetra"
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "test.avs"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_mat_id_in_attrs(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.avs"
    write(poly, tmp)
    poly2 = read(tmp)
    assert "mat_id" in poly2.element_attrs
    np.testing.assert_array_equal(poly2.element_attrs["mat_id"], [1])


def test_mat_id_roundtrip(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_attrs={"mat_id": np.array([7], dtype=np.int32)},
    )
    tmp = tmp_path / "test.avs"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.element_attrs["mat_id"], [7])


def test_unknown_element_type_raises(tmp_path) -> None:
    bad = "2 1 0 0 0\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n1 1 bogus 1 2\n"
    tmp = tmp_path / "bad.avs"
    tmp.write_text(bad)
    with pytest.raises(CodecError):
        read(tmp)


def test_truncated_file_raises(tmp_path) -> None:
    # header claims 2 nodes and 1 elem, but only 1 node line present
    truncated = "2 1 0 0 0\n1 0.0 0.0 0.0\n"
    tmp = tmp_path / "truncated.avs"
    tmp.write_text(truncated, encoding="utf-8")
    with pytest.raises(CodecError, match="truncated"):
        read(tmp)


def test_unsupported_elem_type_write_warns(tmp_path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts, [("quadratic_triangle", np.array([[0, 1, 2, 3, 4, 5]]))]
    )
    tmp = tmp_path / "test.avs"
    with pytest.warns(UserWarning, match="not supported"):
        write(poly, tmp)


def test_undeclared_node_raises(tmp_path) -> None:
    # element references node ID 99 which is not declared
    bad = "2 1 0 0 0\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n1 1 tri 1 2 99\n"
    tmp = tmp_path / "bad.avs"
    tmp.write_text(bad, encoding="utf-8")
    with pytest.raises(CodecError, match="undeclared node"):
        read(tmp)


def test_mixed_supported_unsupported_write(tmp_path) -> None:
    # tet uses verts 0-3; quadratic_triangle uses verts 0-5 (adds 2 extra)
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0, 0],
            [0.5, 0.5, 0],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("quadratic_triangle", np.array([[0, 1, 2, 3, 4, 5]])),
        ],
    )
    tmp = tmp_path / "mixed.avs"
    with pytest.warns(UserWarning, match="not supported"):
        write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    assert ELEMENT_TYPES_INV[int(poly2.element_types[0])] == "tetra"
    assert poly2.vertices.shape[0] == 4


def test_malformed_node_line_raises(tmp_path) -> None:
    bad = "2 0 0 0 0\n1 0.0 0.0 NOTANUMBER\n2 1.0 0.0 0.0\n"
    tmp = tmp_path / "bad.avs"
    tmp.write_text(bad, encoding="utf-8")
    with pytest.raises(CodecError, match="malformed node"):
        read(tmp)


def test_negative_header_raises(tmp_path) -> None:
    bad = "-1 0 0 0 0\n"
    tmp = tmp_path / "bad.avs"
    tmp.write_text(bad, encoding="utf-8")
    with pytest.raises(CodecError, match="non-negative"):
        read(tmp)


def test_lazy_ignored_warns(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.avs"
    write(poly, tmp)
    with pytest.warns(UserWarning, match="lazy"):
        poly2 = read(tmp, lazy=True)
    assert len(poly2.element_types) == 1


def test_file_header_format(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "test.avs"
    write(poly, tmp)
    first_line = Path(tmp).read_text().splitlines()[0].strip()
    n_verts, n_elems, *_ = first_line.split()
    assert int(n_verts) == 4
    assert int(n_elems) == 2
