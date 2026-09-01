from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._mesh import read, write
from polyxios.exceptions import CodecError
from polyxios.fetcher import fetch


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


@pytest.mark.network
@pytest.mark.parametrize(
    "filename,expected_verts,expected_elems",
    [
        ("beam-tri.mesh", 18, 16),
        ("beam-tet.mesh", 36, 48),
        ("beam-hex.mesh", 36, 8),
        ("beam-wedge.mesh", 27, 8),
        ("fichera-mixed.mesh", 26, 14),
        ("equilateral-pyramid.mesh", 5, 1),
    ],
)
def test_real_files(filename: str, expected_verts: int, expected_elems: int) -> None:
    path = fetch(filename)
    poly = read(path)
    assert len(poly.vertices) == expected_verts
    assert len(poly.element_types) == expected_elems
    assert poly.vertices.shape[1] == 3
    assert poly.vertices.dtype == np.float64


@pytest.mark.network
def test_high_order_mesh_reads_coords() -> None:
    """High-order meshes (nodes section) must not return all-zero coordinates."""
    path = fetch("escher-p2.mesh")
    poly = read(path)
    assert len(poly.vertices) > 0
    assert not np.allclose(poly.vertices, 0.0), "Coordinates must not all be zero."


@pytest.mark.network
@pytest.mark.parametrize(
    "filename,expected_verts,expected_elems",
    [
        ("inline-hex.mesh", 125, 64),  # (4+1)^3 verts, 4^3 hexes
        ("inline-quad.mesh", 25, 16),  # (4+1)^2 verts, 4^2 quads
        ("inline-tri.mesh", 25, 32),  # 4^2 * 2 tris
        ("inline-wedge.mesh", 125, 128),  # 4^3 * 2 wedges
    ],
)
def test_inline_real_files(
    filename: str, expected_verts: int, expected_elems: int
) -> None:
    """INLINE real files generate correct vertex and element counts."""
    path = fetch(filename)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        poly = read(path)
    assert not any(issubclass(x.category, UserWarning) for x in w)
    assert len(poly.vertices) == expected_verts
    assert len(poly.element_types) == expected_elems
    assert poly.vertices.dtype == np.float64
    assert not np.allclose(poly.vertices, 0.0)


@pytest.mark.network
def test_nurbs_real_file_control_points() -> None:
    """NURBS real file: control points extracted, knot vectors non-empty."""
    path = fetch("beam-hex-nurbs.mesh")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        poly = read(path)
    assert len(poly.vertices) == 12  # 12 control points declared
    assert len(poly.element_types) == 2
    assert len(poly.global_attrs["mfem_nurbs_knotvectors"]) == 4
    assert len(poly.global_attrs["mfem_nurbs_weights"]) > 0


@pytest.mark.network
def test_nc_real_file_full_reconstruction() -> None:
    """NC real file: vertex_parents midpoints reconstruct all 223 vertices."""
    path = fetch("amr-hex.mesh")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        poly = read(path)
    assert not any(issubclass(x.category, UserWarning) for x in w)
    assert poly.global_attrs["mfem_nc_n_total_elements"] == 137
    assert poly.global_attrs["mfem_nc_n_leaf_elements"] == 120
    # Full reconstruction: 8 base + 215 midpoints = 223 vertices
    assert len(poly.vertices) == 223
    assert not np.allclose(poly.vertices, 0.0)
    # All leaf element indices must be valid
    assert poly.connectivity.max() < len(poly.vertices)


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


def test_an_element_mfem_cannot_hold_is_skipped_not_mislabelled(tmp_path) -> None:
    """A geometry code and a node count that disagree cost every record after."""
    verts = np.arange(30, dtype=np.float64).reshape(10, 3)
    poly = make_polydata(verts, [("quadratic_tetra", np.arange(10).reshape(1, 10))])
    path = tmp_path / "quad.mesh"
    with pytest.warns(UserWarning, match="no MFEM geometry holds"):
        write(poly, path)
    back = read(path)
    assert len(back.element_types) == 0


def test_the_elements_that_do_fit_still_reach_the_file(tmp_path) -> None:
    """Skipping one element must not renumber or drop the rest."""
    verts = np.arange(42, dtype=np.float64).reshape(14, 3)
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("quadratic_tetra", np.arange(4, 14).reshape(1, 10)),
        ],
    )
    path = tmp_path / "mixed.mesh"
    with pytest.warns(UserWarning, match="no MFEM geometry holds"):
        write(poly, path)
    back = read(path)
    assert len(back.element_types) == 1
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2, 3])


@pytest.mark.parametrize(
    "header",
    ["MFEM NC mesh v1.0", "MFEM NC-Mesh v1.0", "MFEM NC MESH v1.0"],
)
def test_every_nc_header_spelling_reaches_the_nc_reader(tmp_path, header: str) -> None:
    """The sniffer accepts five spellings, upper-cased; the reader has to agree.

    ``MFEM NC-Mesh`` was falling through to the NURBS reader, which read a
    refinement forest as control points and warned under the wrong variant's
    name, and an upper-cased header matched no branch at all and was refused
    for not starting with ``MFEM mesh``.
    """
    path = tmp_path / "nc.mesh"
    path.write_text(f"{header}\n\ndimension\n3\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        poly = read(path)

    assert not any(issubclass(w.category, UserWarning) for w in caught)
    assert "mfem_nc_n_leaf_elements" in poly.global_attrs
    assert "mfem_nc_n_total_elements" in poly.global_attrs


def test_a_header_in_another_case_reads_as_the_flavour_it_names(tmp_path) -> None:
    """The registry claims a file by the upper-cased header, so the reader must."""
    path = tmp_path / "shouty.mesh"
    path.write_text("MFEM MESH v1.0\n\ndimension\n3\n")

    poly = read(path)
    assert len(poly.vertices) == 0


def test_a_byte_order_mark_does_not_hide_the_header(tmp_path) -> None:
    """The sniffer decodes the mark away and the reader has to as well."""
    body = (
        "MFEM mesh v1.0\n\ndimension\n2\n\nelements\n1\n1 2 0 1 2\n\n"
        "boundary\n0\n\nvertices\n3\n2\n0 0\n1 0\n0 1\n"
    )
    path = tmp_path / "bom.mesh"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode())

    # A file the registry had already claimed as MFEM: the mark is not
    # whitespace, so strip left it in place and the header matched no branch.
    assert len(read(path).vertices) == 3
