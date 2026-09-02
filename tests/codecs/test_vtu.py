from __future__ import annotations

import tempfile
import warnings

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


# ---------------------------------------------------------------------------
# <FieldData>: the mesh's own metadata
# ---------------------------------------------------------------------------


def test_issue_1546_a_vtu_write_holds_the_field_data(tmp_path) -> None:
    """The writer dropped global_attrs, so a time value, a material constant
    or a solver tolerance did not survive being written."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"TimeValue": 0.25, "gravity": np.array([0.0, 0.0, -9.81])},
    )
    path = tmp_path / "field.vtu"

    write(poly, path)
    back = read(path)

    np.testing.assert_allclose(back.global_attrs["TimeValue"], [0.25])
    np.testing.assert_allclose(back.global_attrs["gravity"], [0.0, 0.0, -9.81])


@pytest.mark.parametrize("binary", [False, True])
def test_a_field_array_keeps_the_type_it_was_held_in(tmp_path, binary: bool) -> None:
    """An identifier past 2**53 comes home a different number as a double."""
    wide = np.array([9007199254740993], dtype=np.int64)
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"case_id": wide},
    )
    path = tmp_path / "wide.vtu"

    write(poly, path, binary=binary)
    back = read(path)

    assert back.global_attrs["case_id"].dtype == np.int64
    np.testing.assert_array_equal(back.global_attrs["case_id"], wide)


def test_a_global_that_is_text_travels_beside_the_numbers(tmp_path) -> None:
    """A <FieldData> block holds a String array beside its numeric ones, which
    is where a name, a title or a solver's own label belongs."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"solver": "polyxios", "steps": 12},
    )
    path = tmp_path / "mixed.vtu"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    back = read(path)

    assert back.global_attrs["solver"] == "polyxios"
    np.testing.assert_array_equal(back.global_attrs["steps"], [12])


def test_a_global_no_array_of_any_kind_can_hold_is_named_and_dropped(
    tmp_path,
) -> None:
    """A <FieldData> array holds numbers or text; a mapping is neither. The
    mesh is still written - the loss is one key, not the file."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"run": {"id": 3}, "steps": 12},
    )
    path = tmp_path / "unspellable.vtu"

    with pytest.warns(UserWarning, match=r"global_attrs \['run'\]"):
        write(poly, path)
    back = read(path)

    assert "run" not in back.global_attrs
    np.testing.assert_array_equal(back.global_attrs["steps"], [12])


def test_several_strings_under_one_key_come_back_as_the_list_they_were(
    tmp_path,
) -> None:
    """A text array holds one string per tuple, so a list of them travels as
    one array and is cut back apart by the terminator after each."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"notes": ["first pass", "rerun with \u00e9"]},
    )
    path = tmp_path / "notes.vtu"

    write(poly, path)

    assert read(path).global_attrs["notes"] == ["first pass", "rerun with \u00e9"]


def test_a_field_data_name_spelled_twice_keeps_the_first_and_says_so(
    tmp_path,
) -> None:
    """A mapping holds one value per key, and letting the last win answered
    one file two ways depending on the order the blocks were walked in - the
    first is what a file of one array would have given."""
    path = tmp_path / "twice.vtu"
    extra = (
        "   <FieldData>\n"
        '    <DataArray type="Float64" Name="t" NumberOfTuples="1"'
        ' format="ascii">1</DataArray>\n'
        '    <DataArray type="Float64" Name="t" NumberOfTuples="1"'
        ' format="ascii">2</DataArray>\n'
        "   </FieldData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    with pytest.warns(UserWarning, match=r"names array\(s\) \['t'\] more than"):
        poly = read(path)

    np.testing.assert_allclose(poly.global_attrs["t"], [1.0])


def test_a_string_array_in_field_data_is_the_text_it_spells(tmp_path) -> None:
    """VTK writes a label as a String array, whose payload is the characters
    as numbers with a zero after each; read as numbers it was dropped."""
    path = tmp_path / "label.vtu"
    extra = (
        "   <FieldData>\n"
        '    <Array type="String" Name="title" NumberOfTuples="1"'
        ' format="ascii">104 105 0</Array>\n'
        "   </FieldData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    assert read(path).global_attrs["title"] == "hi"


def test_field_data_on_a_piece_is_read(tmp_path) -> None:
    """VTK puts the block on the dataset and other writers put it on a Piece;
    a mesh whose metadata is only read from one place loses the other."""
    path = tmp_path / "piece_field.vtu"
    extra = (
        "   <FieldData>\n"
        '    <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1"'
        ' format="ascii">3.5</DataArray>\n'
        "   </FieldData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    np.testing.assert_allclose(read(path).global_attrs["TimeValue"], [3.5])


def test_field_data_on_the_dataset_wins_over_a_piece(tmp_path) -> None:
    """Both places hold a key of the same name: the dataset's is the file's
    own answer for the mesh, and a piece's is one piece's."""
    path = tmp_path / "both.vtu"
    piece_field = (
        "   <FieldData>\n"
        '    <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1"'
        ' format="ascii">1</DataArray>\n'
        "   </FieldData>\n"
    )
    dataset_field = (
        "  <FieldData>\n"
        '   <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1"'
        ' format="ascii">2</DataArray>\n'
        "  </FieldData>\n"
    )
    path.write_text(
        _vtu(dataset_field + _piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, piece_field))
    )

    np.testing.assert_allclose(read(path).global_attrs["TimeValue"], [2])


def test_an_unnamed_field_array_is_skipped(tmp_path) -> None:
    """global_attrs is keyed by name, and the file gives no other handle - so
    the array is dropped, and counted rather than lost in silence."""
    path = tmp_path / "unnamed.vtu"
    extra = (
        "   <FieldData>\n"
        '    <DataArray type="Float64" NumberOfTuples="1" format="ascii">1</DataArray>\n'
        '    <DataArray type="Float64" Name="kept" NumberOfTuples="1"'
        ' format="ascii">2</DataArray>\n'
        "   </FieldData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    with pytest.warns(UserWarning, match="no Name="):
        poly = read(path)

    assert tuple(poly.global_attrs) == ("kept",)


def test_metadata_with_no_name_to_file_it_under_is_dropped(tmp_path) -> None:
    """Every format here writes the key as the array's own handle, and a key
    that is not one leaves the array unfindable where it does not make the
    file unreadable outright."""
    poly = _tet_mesh()
    poly.global_attrs[""] = 1
    poly.global_attrs["kept"] = 2
    path = tmp_path / "nameless.vtu"

    with pytest.warns(UserWarning, match="have no name a data array can carry"):
        write(poly, path)

    back = read(path)
    assert tuple(back.global_attrs) == ("kept",)


def test_a_key_that_is_not_a_name_is_reported_as_that(tmp_path, recwarn) -> None:
    """A key with a perfectly good number under it is dropped for its name,
    and used to be reported among the values no numeric array can spell."""
    poly = _tet_mesh()
    poly.global_attrs[7] = 1.0
    poly.global_attrs["run"] = {"id": 3}
    path = tmp_path / "misnamed.vtu"

    write(poly, path)

    said = [str(w.message) for w in recwarn]
    assert any("[7] have no name a data array can carry" in m for m in said)
    assert any("['run'] hold values no numeric array" in m for m in said)


def test_a_key_that_is_not_a_name_is_dropped_whatever_text_is_under_it(
    tmp_path, recwarn
) -> None:
    """The name check runs ahead of the split between numbers and text, so a
    string under a key no attribute can carry is reported like any other."""
    poly = _tet_mesh()
    poly.global_attrs[""] = "polyxios"
    path = tmp_path / "blank_key.vtu"

    write(poly, path)

    assert any("have no name a data array can carry" in str(w.message) for w in recwarn)
    assert read(path).global_attrs == {}


# ---------------------------------------------------------------------------
# Tag groups: one membership column apiece
# ---------------------------------------------------------------------------


def test_a_tag_group_travels_as_its_own_column(tmp_path) -> None:
    """A VTU has no set of its own, but PointData and CellData hold one column
    per group - and an element in two groups is named by both, which a format
    spelling one reference per element cannot say."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))],
        vertex_tags={"corner": np.array([0, 3], dtype=np.int32)},
        element_tags={
            "a": np.array([0], dtype=np.int32),
            "b": np.array([0, 1], dtype=np.int32),
        },
    )
    path = tmp_path / "tagged.vtu"

    write(poly, path)
    back = read(path)

    np.testing.assert_array_equal(back.element_tags["a"], [0])
    np.testing.assert_array_equal(back.element_tags["b"], [0, 1])
    np.testing.assert_array_equal(back.vertex_tags["corner"], [0, 3])
    # The columns are tags, not attributes.
    assert back.element_attrs == {}
    assert back.vertex_attrs == {}


def test_a_tag_column_is_named_so_a_reader_can_tell(tmp_path) -> None:
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"wall": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "named.vtu"

    write(poly, path, binary=False)

    assert 'Name="polyxios_tag_wall"' in path.read_text()


def test_a_tag_name_holding_xml_markup_survives(tmp_path) -> None:
    """A name is whatever another format called it. Written as it stands it
    closed the attribute early and left a file no reader could parse."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={'steel & "iron" <2>': np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "markup.vtu"

    write(poly, path)

    assert list(read(path).element_tags) == ['steel & "iron" <2>']


def test_a_name_holding_a_character_xml_cannot_spell_is_dropped(tmp_path) -> None:
    """Escaping carries a markup character through; a control character is
    outside XML's own Char production, so a numeric reference is no way round
    it either and a file holding one parses in no reader at all."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"bell\x07": np.zeros(3), "kept": np.ones(3)},
        element_tags={"nul\x00": np.array([0], dtype=np.int32)},
        global_attrs={"vtab\x0b": np.array([1.0])},
    )
    path = tmp_path / "control.vtu"

    with pytest.warns(UserWarning, match="XML cannot spell"):
        write(poly, path)

    back = read(path)
    assert list(back.vertex_attrs) == ["kept"]
    assert back.element_tags == {}
    assert back.global_attrs == {}


def test_an_attribute_keyed_by_something_that_is_not_text_still_writes(
    tmp_path,
) -> None:
    """Nothing stops a caller keying an attribute by a number, and the name
    rules are about what a file can spell, not about refusing one."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={5: np.zeros(3)},
    )
    path = tmp_path / "numbered.vtu"

    write(poly, path)

    assert list(read(path).vertex_attrs) == ["5"]


def test_a_float_column_named_like_a_tag_stays_an_attribute(tmp_path) -> None:
    """Rounding a member into place would name the wrong element."""
    path = tmp_path / "halves.vtu"
    extra = (
        "   <CellData>\n"
        '    <DataArray type="Float64" Name="polyxios_tag_odd" format="ascii">'
        "0.5</DataArray>\n"
        "   </CellData>\n"
    )
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))

    poly = read(path)

    assert poly.element_tags == {}
    np.testing.assert_allclose(poly.element_attrs["polyxios_tag_odd"], [0.5])


def test_a_name_holding_whitespace_comes_back_the_name_it_was(tmp_path) -> None:
    """An XML parser normalises a literal newline in an attribute value to a
    space, so a name written as it stands came back a name it never was."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"a\nb\tc\rd": np.arange(3.0)},
    )
    path = tmp_path / "white.vtu"

    write(poly, path)

    assert list(read(path).vertex_attrs) == ["a\nb\tc\rd"]


def test_a_field_array_that_miscounts_its_tuples_is_reported(tmp_path) -> None:
    """Nothing else in the file counts a field array, so the declaration is
    the only thing a reader can check it against."""
    path = tmp_path / "miscount.vtu"
    field = (
        "  <FieldData>\n"
        '   <DataArray type="Float64" Name="t" NumberOfTuples="9"'
        ' format="ascii">1 2 3</DataArray>\n'
        "  </FieldData>\n"
    )
    path.write_text(_vtu(field + _piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1)))

    with pytest.warns(UserWarning, match="NumberOfTuples"):
        poly = read(path)

    np.testing.assert_allclose(poly.global_attrs["t"], [1, 2, 3])


def test_a_name_two_pieces_spell_keeps_the_first_and_is_reported(tmp_path) -> None:
    """global_attrs is one mapping over the whole mesh and the pieces are
    joined into one mesh, so a name two of them carry has one slot for two
    values. Folding the pieces with a plain update kept the last, which is
    the one a reader of a single-piece file would never have seen."""
    path = tmp_path / "two_pieces.vtu"
    piece = (
        '<Piece NumberOfPoints="1" NumberOfCells="0">'
        '<FieldData><DataArray type="Int32" Name="run" NumberOfTuples="1"'
        ' format="ascii">{run}</DataArray></FieldData>'
        '<Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">{x} 0 0</DataArray></Points>'
        '<Cells><DataArray type="Int64" Name="connectivity" format="ascii"/>'
        '<DataArray type="Int64" Name="offsets" format="ascii"/>'
        '<DataArray type="UInt8" Name="types" format="ascii"/></Cells></Piece>'
    )
    path.write_text(
        '<?xml version="1.0"?>'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">'
        "<UnstructuredGrid>"
        + piece.format(run=1, x=0)
        + piece.format(run=99, x=1)
        + "</UnstructuredGrid></VTKFile>"
    )

    with pytest.warns(UserWarning, match="more than one Piece"):
        back = read(path)

    np.testing.assert_array_equal(back.global_attrs["run"], [1])


def test_the_dataset_field_data_still_wins_over_a_piece(tmp_path) -> None:
    """A piece describes its own part of the file; the dataset block
    describes the file. Reading the pieces first is what makes them lose
    to it, and reporting their own clash must not change that."""
    path = tmp_path / "both.vtu"
    path.write_text(
        '<?xml version="1.0"?>'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">'
        "<UnstructuredGrid>"
        '<FieldData><DataArray type="Int32" Name="run" NumberOfTuples="1"'
        ' format="ascii">7</DataArray></FieldData>'
        '<Piece NumberOfPoints="1" NumberOfCells="0">'
        '<FieldData><DataArray type="Int32" Name="run" NumberOfTuples="1"'
        ' format="ascii">1</DataArray></FieldData>'
        '<Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">0 0 0</DataArray></Points>'
        '<Cells><DataArray type="Int64" Name="connectivity" format="ascii"/>'
        '<DataArray type="Int64" Name="offsets" format="ascii"/>'
        '<DataArray type="UInt8" Name="types" format="ascii"/></Cells></Piece>'
        "</UnstructuredGrid></VTKFile>"
    )

    np.testing.assert_array_equal(read(path).global_attrs["run"], [7])
