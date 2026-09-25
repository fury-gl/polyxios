from __future__ import annotations

import mmap
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


def _mapped(arr: np.ndarray) -> bool:
    """Whether the array, through however many views, sits on a mapping."""
    base = arr
    while isinstance(base, np.ndarray):
        base = base.base
    if isinstance(base, memoryview):
        base = base.obj
    return isinstance(base, mmap.mmap)


def _mixed_polys() -> object:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    return make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("polygon", np.array([[1, 4, 5, 2, 3]])),
        ],
        vertex_attrs={"pressure": np.arange(6.0)},
        element_attrs={"stress": np.array([10.0, 20.0, 30.0])},
    )


def test_roundtrip_appended(tmp_path) -> None:
    """A raw appended section reads back eagerly like any other layout,
    and the Polys section types its cells by their point count."""
    poly = _mixed_polys()
    tmp = tmp_path / "mesh.vtp"
    write(poly, tmp, appended=True)
    back = read(tmp)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(
        back.element_attrs["stress"], poly.element_attrs["stress"]
    )
    assert back.vertices.flags.writeable
    assert not _mapped(back.vertices)


def test_lazy_arrays_view_the_mapping(tmp_path) -> None:
    poly = _mixed_polys()
    tmp = tmp_path / "mesh.vtp"
    write(poly, tmp, appended=True)
    back = read(tmp, lazy=True)

    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(
        back.vertex_attrs["pressure"], poly.vertex_attrs["pressure"]
    )
    for arr in (back.vertices, back.connectivity, back.vertex_attrs["pressure"]):
        assert _mapped(arr)
        assert not arr.flags.writeable
    assert back.connectivity.dtype == back.offsets.dtype


def test_lazy_refuses_inline_arrays(tmp_path) -> None:
    poly = _synthetic_mesh()
    tmp = tmp_path / "mesh.vtp"
    write(poly, tmp, binary=True)
    with pytest.raises(LazyReadError, match="inline"):
        read(tmp, lazy=True)
    np.testing.assert_allclose(read(tmp).vertices, poly.vertices)


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


def test_a_vtp_write_holds_the_field_data(tmp_path) -> None:
    """The writer dropped global_attrs, so a time value or a material constant
    did not survive being written."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"TimeValue": 0.25},
    )
    path = tmp_path / "field.vtp"

    write(poly, path)

    np.testing.assert_allclose(read(path).global_attrs["TimeValue"], [0.25])


def test_field_data_on_a_piece_is_read(tmp_path) -> None:
    """VTK puts the block on the dataset and other writers put it on a Piece."""
    path = tmp_path / "piece_field.vtp"
    extra = (
        "   <FieldData>\n"
        '    <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1"'
        ' format="ascii">3.5</DataArray>\n'
        "   </FieldData>\n"
    )
    path.write_text(_polydata_file("0 0 0 1 0 0 0 1 0", 3, extra))

    np.testing.assert_allclose(read(path).global_attrs["TimeValue"], [3.5])
