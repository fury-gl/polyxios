from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._mesh import read, write
from polyxios.exceptions import CodecError


def _tet_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_triangle() -> None:
    poly = _tri_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == 2
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_tetra() -> None:
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == 1
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_lazy_ignored() -> None:
    """lazy=True is silently accepted (mesh is always read eagerly)."""
    poly = _tri_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp, lazy=True)
    assert len(poly2.vertices) == 4


def test_inline_generates_geometry() -> None:
    """INLINE mesh generates real vertices and elements — no warning, no empty data."""
    content = "MFEM INLINE mesh v1.0\n\ntype = tet\nnx = 4\nny = 4\nnz = 4\nsx = 1.0\nsy = 1.0\nsz = 1.0\n"
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False, mode="w") as f:
        f.write(content)
        tmp = f.name
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        poly = read(tmp)
    assert not any(issubclass(x.category, UserWarning) for x in w), (
        "No warning expected"
    )
    assert len(poly.vertices) == 125  # (4+1)^3
    assert len(poly.element_types) == 320  # 4^3 * 5 tets
    params = poly.global_attrs["mfem_inline_params"]
    assert params["type"] == "tet"
    assert params["nx"] == 4
    assert params["sx"] == 1.0


def test_nurbs_warns_and_returns_control_points() -> None:
    content = "MFEM NURBS mesh v1.0\n\ndimension\n2\n"
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False, mode="w") as f:
        f.write(content)
        tmp = f.name
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        poly = read(tmp)
    assert any(issubclass(x.category, UserWarning) for x in w)
    assert any("CONTROL POINTS" in str(x.message) for x in w)
    assert "mfem_nurbs_knotvectors" in poly.global_attrs
    assert "mfem_nurbs_weights" in poly.global_attrs


def test_nc_mesh_returns_leaf_elements_no_warning() -> None:
    """NC mesh returns leaf elements and global_attrs with no UserWarning."""
    content = "MFEM NC mesh v1.0\n\ndimension\n3\n"
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False, mode="w") as f:
        f.write(content)
        tmp = f.name
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        poly = read(tmp)
    assert not any(issubclass(x.category, UserWarning) for x in w)
    assert "mfem_nc_n_leaf_elements" in poly.global_attrs
    assert "mfem_nc_n_total_elements" in poly.global_attrs


def test_unknown_header_raises_codec_error() -> None:
    content = "NOT A MESH FILE\n\nsome data\n"
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False, mode="w") as f:
        f.write(content)
        tmp = f.name
    with pytest.raises(CodecError):
        read(tmp)


def test_written_file_is_valid_mfem() -> None:
    """Written .mesh file must start with 'MFEM mesh v1.0'."""
    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    with open(tmp) as fh:
        first_line = fh.readline().strip()
    assert first_line == "MFEM mesh v1.0"


def test_polyxios_dispatch() -> None:
    """polyxios.read() dispatches .mesh to the MFEM codec."""
    import polyxios

    poly = _tet_mesh()
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = polyxios.read(tmp)
    assert len(poly2.vertices) == 4
    assert len(poly2.element_types) == 1
