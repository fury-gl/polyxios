from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._vtu import read, write
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.fetcher import fetch


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


@pytest.mark.network
@pytest.mark.parametrize(
    "filename,expected_verts,expected_cells",
    [
        ("quadraticTetra01.vtu", 22, 3),
        ("Hexahedron.vtu", 26, 7),
        ("QuadraticPyramid.vtu", 153, 48),
        ("QuadraticWedge.vtu", 93, 16),
        ("polyhedron2pieces.vtu", 18, 4),
    ],
)
def test_real_files(filename: str, expected_verts: int, expected_cells: int) -> None:
    path = fetch(filename)
    poly = read(path)
    assert len(poly.vertices) == expected_verts
    assert len(poly.element_types) == expected_cells
    assert poly.vertices.shape[1] == 3
    assert poly.vertices.dtype == np.float64


# ---------------------------------------------------------------------------
# P1.5 - VTU reader/writer hardening
# ---------------------------------------------------------------------------


def _vtu(body: str) -> str:
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        " <UnstructuredGrid>\n"
        f"{body}"
        " </UnstructuredGrid>\n"
        "</VTKFile>\n"
    )


def _piece(
    points: str, n_points: int, cells: str, n_cells: int, extra: str = ""
) -> str:
    return (
        f'  <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n'
        "   <Points>\n"
        '    <DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        f"{points}</DataArray>\n"
        "   </Points>\n"
        "   <Cells>\n"
        f"{cells}"
        "   </Cells>\n"
        f"{extra}"
        "  </Piece>\n"
    )


_TRI_CELLS = (
    '    <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2</DataArray>\n'
    '    <DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>\n'
    '    <DataArray type="UInt8" Name="types" format="ascii">5</DataArray>\n'
)


def test_two_pieces_are_both_read(tmp_path) -> None:
    path = tmp_path / "two.vtu"
    path.write_text(
        _vtu(
            _piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1)
            + _piece("2 0 0 3 0 0 2 1 0", 3, _TRI_CELLS, 1)
        )
    )

    poly = read(path)

    assert poly.vertices.shape == (6, 3)
    assert len(poly.element_types) == 2
    # The second piece's cell indexes the second piece's points.
    np.testing.assert_array_equal(poly.connectivity[3:], [3, 4, 5])


def test_the_writer_declares_a_real_vtk_version(tmp_path) -> None:
    path = tmp_path / "mesh.vtu"

    write(_tet_mesh(), path)

    header = next(line for line in path.read_text().splitlines() if "<VTKFile" in line)
    assert 'version="1.0"' in header


def test_a_file_with_no_points_reads_empty(tmp_path) -> None:
    path = tmp_path / "empty.vtu"
    empty_cells = (
        '    <DataArray type="Int32" Name="connectivity" format="ascii"></DataArray>\n'
        '    <DataArray type="Int32" Name="offsets" format="ascii"></DataArray>\n'
        '    <DataArray type="UInt8" Name="types" format="ascii"></DataArray>\n'
    )
    path.write_text(_vtu(_piece("", 0, empty_cells, 0)))

    poly = read(path)

    assert poly.vertices.shape == (0, 3)
    assert len(poly.element_types) == 0


def test_a_string_data_array_is_skipped_with_a_warning(tmp_path) -> None:
    """A String array holds labels, not numbers; it cannot become an attr."""
    path = tmp_path / "string.vtu"
    extra = (
        "   <PointData>\n"
        '    <DataArray type="String" Name="labels" format="ascii">'
        "97 98 99</DataArray>\n"
        '    <DataArray type="Float64" Name="s" format="ascii">1 2 3</DataArray>\n'
        "   </PointData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    with pytest.warns(UserWarning, match="String"):
        poly = read(path)

    assert "labels" not in poly.vertex_attrs
    np.testing.assert_allclose(poly.vertex_attrs["s"], [1, 2, 3])


def test_an_unreadable_array_does_not_take_the_rest_with_it(
    tmp_path,
) -> None:
    path = tmp_path / "unknown.vtu"
    extra = (
        "   <PointData>\n"
        '    <DataArray type="Float128" Name="odd" format="ascii">1 2 3</DataArray>\n'
        '    <DataArray type="Float64" Name="s" format="ascii">4 5 6</DataArray>\n'
        "   </PointData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    with pytest.warns(UserWarning, match="Float128"):
        poly = read(path)

    assert "odd" not in poly.vertex_attrs
    np.testing.assert_allclose(poly.vertex_attrs["s"], [4, 5, 6])


def test_vertex_order_survives_a_round_trip(tmp_path) -> None:
    """A renumbered point array silently invalidates every external index."""
    verts = np.array(
        [[3, 0, 0], [0, 0, 0], [2, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64
    )
    poly = make_polydata(verts, [("triangle", np.array([[4, 1, 3], [0, 2, 3]]))])
    path = tmp_path / "order.vtu"

    write(poly, path)
    back = read(path)

    np.testing.assert_array_equal(back.vertices, verts)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


# ---------------------------------------------------------------------------
# Pieces that do not line up
# ---------------------------------------------------------------------------


def test_an_attribute_covering_one_piece_of_two_is_dropped(tmp_path) -> None:
    """Joined short, its rows would sit against the second piece's points."""
    path = tmp_path / "partial.vtu"
    extra = (
        "   <PointData>\n"
        '    <DataArray type="Float64" Name="s" format="ascii">1 2 3</DataArray>\n'
        "   </PointData>\n"
    )
    path.write_text(
        _vtu(
            _piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)
            + _piece("2 0 0 3 0 0 2 1 0", 3, _TRI_CELLS, 1)
        )
    )

    with pytest.warns(UserWarning, match="covers 3 of 6"):
        poly = read(path)

    assert "s" not in poly.vertex_attrs
    assert poly.vertices.shape == (6, 3)


def test_a_piece_that_withholds_its_points_is_refused(tmp_path) -> None:
    """Its cells index those points, and later pieces are offset by them."""
    path = tmp_path / "short.vtu"
    path.write_text(_vtu(_piece("0 0 0 1 0 0", 3, _TRI_CELLS, 1)))

    with pytest.raises(CodecError, match="declares 3 points"):
        read(path)


def test_a_piece_with_no_points_element_is_refused(tmp_path) -> None:
    """A missing <Points> shifts every later piece as surely as a short one."""
    pointless = (
        '  <Piece NumberOfPoints="3" NumberOfCells="1">\n'
        f"   <Cells>\n{_TRI_CELLS}   </Cells>\n"
        "  </Piece>\n"
    )
    path = tmp_path / "pointless.vtu"
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1) + pointless))

    with pytest.raises(CodecError, match="declares 3 points"):
        read(path)


def test_an_attribute_the_pieces_shape_differently_is_dropped(tmp_path) -> None:
    """numpy refuses to join them, and the refusal named neither array."""
    scalar = (
        "   <PointData>\n"
        '    <DataArray type="Float64" Name="q" NumberOfComponents="1"'
        ' format="ascii">1 2 3</DataArray>\n'
        "   </PointData>\n"
    )
    vector = (
        "   <PointData>\n"
        '    <DataArray type="Float64" Name="q" NumberOfComponents="3"'
        ' format="ascii">1 2 3 4 5 6 7 8 9</DataArray>\n'
        "   </PointData>\n"
    )
    path = tmp_path / "ragged.vtu"
    path.write_text(
        _vtu(
            _piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, scalar)
            + _piece("2 0 0 3 0 0 2 1 0", 3, _TRI_CELLS, 1, vector)
        )
    )

    with pytest.warns(UserWarning, match="shaped differently"):
        poly = read(path)

    assert "q" not in poly.vertex_attrs
    assert poly.vertices.shape == (6, 3)


def test_a_points_array_of_ragged_tuples_names_the_piece(tmp_path) -> None:
    """reshape answers a size that is not whole tuples without naming a file."""
    path = tmp_path / "ragged_points.vtu"
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0 9", 3, _TRI_CELLS, 1)))

    with pytest.raises(CodecError, match="not 3 tuples of three or more"):
        read(path)


def test_a_points_array_of_a_type_with_no_numbers_names_the_type() -> None:
    """Decoding it empty and blaming the count says nothing about the cause."""
    content = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="2" NumberOfCells="0">
      <Points>
        <DataArray type="String" Name="Points" format="ascii">a b</DataArray>
      </Points>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    with tempfile.NamedTemporaryFile("w", suffix=".vtu", delete=False) as f:
        f.write(content)
        tmp = f.name

    with pytest.raises(CodecError, match="String"):
        read(tmp)


def test_a_failing_piece_is_named_by_its_index() -> None:
    """A file of many pieces gave no way to find the one at fault."""
    good = _piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1)
    bad = _piece("0 0 0 1 0 0", 3, _TRI_CELLS, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".vtu", delete=False) as f:
        f.write(_vtu(good + bad))
        tmp = f.name

    with pytest.raises(CodecError, match="Piece 1 declares 3 points"):
        read(tmp)


@pytest.mark.parametrize("binary", [False, True])
def test_a_tensor_declares_every_component_it_holds(tmp_path, binary: bool) -> None:
    """shape[1] of an (n, 3, 3) array is three, and the tuple is nine."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tensor = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    poly.vertex_attrs["tensor"] = tensor
    path = tmp_path / "tensor.vtu"

    write(poly, path, binary=binary)
    back = read(path)

    assert back.vertex_attrs["tensor"].shape == (3, 9)
    np.testing.assert_array_equal(back.vertex_attrs["tensor"], tensor.reshape(3, 9))


@pytest.mark.parametrize("binary", [False, True])
def test_an_integer_attribute_keeps_every_digit_it_holds(
    tmp_path, binary: bool
) -> None:
    """Cast to a double, an id past 2**53 came back a different number."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    ids = np.array([2**53 + 1, 2**53 + 3, 7], dtype=np.int64)
    poly.vertex_attrs["ids"] = ids
    path = tmp_path / "ids.vtu"

    write(poly, path, binary=binary)
    back = read(path)

    assert back.vertex_attrs["ids"].dtype == np.int64
    np.testing.assert_array_equal(back.vertex_attrs["ids"], ids)


@pytest.mark.parametrize("binary", [False, True])
def test_a_float32_attribute_is_declared_and_read_as_float32(
    tmp_path, binary: bool
) -> None:
    """The header has to name the width the bytes are written at."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    values = np.array([0.1, 1 / 3, 3.4028235e38], dtype=np.float32)
    poly.vertex_attrs["v"] = values
    path = tmp_path / "f32.vtu"

    write(poly, path, binary=binary)
    assert 'type="Float32"' in path.read_text()
    back = read(path)

    assert back.vertex_attrs["v"].dtype == np.float32
    np.testing.assert_array_equal(back.vertex_attrs["v"], values)


def test_a_piece_count_that_is_not_a_count_names_the_file(tmp_path) -> None:
    """int() on the attribute answered with a ValueError naming nothing."""
    path = tmp_path / "bad.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        " <UnstructuredGrid>\n"
        '  <Piece NumberOfPoints="many" NumberOfCells="1">\n'
        '   <Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">0 0 0 1 0 0 0 1 0</DataArray></Points>\n'
        "  </Piece>\n"
        " </UnstructuredGrid>\n"
        "</VTKFile>\n"
    )

    with pytest.raises(CodecError, match="NumberOfPoints='many'"):
        read(path)


def _one_point_grid(point_data: str) -> str:
    """An otherwise empty .vtu carrying one PointData array."""
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        " <UnstructuredGrid>\n"
        '  <Piece NumberOfPoints="1" NumberOfCells="0">\n'
        '   <Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">0 0 0</DataArray></Points>\n'
        "   <Cells>\n"
        '    <DataArray type="Int64" Name="connectivity" format="ascii"></DataArray>\n'
        '    <DataArray type="Int64" Name="offsets" format="ascii"></DataArray>\n'
        '    <DataArray type="UInt8" Name="types" format="ascii"></DataArray>\n'
        "   </Cells>\n"
        f"   <PointData>{point_data}</PointData>\n"
        "  </Piece>\n"
        " </UnstructuredGrid>\n"
        "</VTKFile>\n"
    )


def test_a_value_too_wide_for_its_declared_type_wraps_rather_than_raising(
    tmp_path,
) -> None:
    """numpy answers an out-of-range token with OverflowError, not ValueError."""
    path = tmp_path / "wide.vtu"
    path.write_text(
        _one_point_grid(
            '<DataArray type="UInt8" Name="q" format="ascii">300</DataArray>'
        )
    )

    with pytest.warns(UserWarning, match="cannot hold"):
        poly = read(path)

    np.testing.assert_array_equal(poly.vertex_attrs["q"], [np.uint8(300 % 256)])


def test_an_ascii_value_that_is_no_number_names_the_array(tmp_path) -> None:
    """float() answers that with a ValueError about one token."""
    path = tmp_path / "nan.vtu"
    path.write_text(
        _one_point_grid('<DataArray type="Int64" Name="q" format="ascii">x</DataArray>')
    )

    with pytest.raises(CodecError, match="DataArray 'q'"):
        read(path)


def test_an_integer_array_spelled_with_a_decimal_point_is_read(tmp_path) -> None:
    """numpy will not parse '1.0' as an Int64; a C reader truncates it."""
    path = tmp_path / "dotted.vtu"
    path.write_text(
        _one_point_grid(
            '<DataArray type="Int64" Name="q" format="ascii">7.0</DataArray>'
        )
    )

    poly = read(path)

    np.testing.assert_array_equal(poly.vertex_attrs["q"], [7])
    assert poly.vertex_attrs["q"].dtype == np.int64


def test_an_attribute_no_data_array_can_hold_names_itself(tmp_path) -> None:
    """A label per vertex reached the Float64 fallback and died converting."""
    poly = _tet_mesh()
    poly.vertex_attrs["label"] = np.array(["a", "b", "c", "d"])
    path = tmp_path / "label.vtu"

    with pytest.raises(CodecError, match="attribute 'label'"):
        write(poly, path)
