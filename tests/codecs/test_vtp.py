from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._vtp import read, write
from polyxios.exceptions import CodecError, LazyReadError


def _synthetic_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_ascii() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    assert len(poly2.element_types) == 2
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_binary() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_lazy() -> None:
    """VTP lazy raises LazyReadError; eager read gives correct data."""
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)
    # Eager read still works
    poly2 = read(tmp, lazy=False)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)


def test_vertex_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pressure = np.array([1.0, 2.0, 3.0, 4.0])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        vertex_attrs={"pressure": pressure},
    )
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert "pressure" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["pressure"], pressure, atol=1e-6)


def test_element_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    stress = np.array([10.0, 20.0])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"stress": stress},
    )
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert "stress" in poly2.element_attrs
    np.testing.assert_allclose(poly2.element_attrs["stress"], stress, atol=1e-6)


def test_unsupported_lazy() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    # VTP lazy not supported with frozen PolyData - raises LazyReadError
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


# ---------------------------------------------------------------------------
# Pieces that do not line up
# ---------------------------------------------------------------------------


def _polydata_file(points: str, n_points: int, extra: str = "") -> str:
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
        " <PolyData>\n"
        f'  <Piece NumberOfPoints="{n_points}" NumberOfPolys="1">\n'
        "   <Points>\n"
        '    <DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        f"{points}</DataArray>\n"
        "   </Points>\n"
        "   <Polys>\n"
        '    <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2'
        "</DataArray>\n"
        '    <DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>\n'
        "   </Polys>\n"
        f"{extra}"
        "  </Piece>\n"
        " </PolyData>\n"
        "</VTKFile>\n"
    )


def test_a_piece_that_withholds_its_points_is_refused(tmp_path) -> None:
    path = tmp_path / "short.vtp"
    path.write_text(_polydata_file("0 0 0 1 0 0", 3))

    with pytest.raises(CodecError, match="declares 3 points"):
        read(path)


def test_a_point_array_shorter_than_the_mesh_is_dropped(tmp_path) -> None:
    path = tmp_path / "partial.vtp"
    extra = (
        "   <PointData>\n"
        '    <DataArray type="Float64" Name="s" format="ascii">1 2</DataArray>\n'
        "   </PointData>\n"
    )
    path.write_text(_polydata_file("0 0 0 1 0 0 0 1 0", 3, extra))

    with pytest.warns(UserWarning, match="covers 2 of 3"):
        poly = read(path)

    assert "s" not in poly.vertex_attrs


def test_a_points_array_of_ragged_tuples_names_the_piece(tmp_path) -> None:
    """reshape answers a size that is not whole tuples without naming a file."""
    path = tmp_path / "ragged_points.vtp"
    path.write_text(_polydata_file("0 0 0 1 0 0 0 1 0 9", 3))

    with pytest.raises(CodecError, match="not 3 tuples of three or more"):
        read(path)


def test_a_piece_count_that_is_not_a_count_names_the_file(tmp_path) -> None:
    """int() on the attribute answered with a ValueError naming nothing."""
    path = tmp_path / "bad.vtp"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
        " <PolyData>\n"
        '  <Piece NumberOfPoints="many" NumberOfPolys="1">\n'
        '   <Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">0 0 0 1 0 0 0 1 0</DataArray></Points>\n'
        "  </Piece>\n"
        " </PolyData>\n"
        "</VTKFile>\n"
    )

    with pytest.raises(CodecError, match="NumberOfPoints='many'"):
        read(path)
