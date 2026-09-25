from __future__ import annotations

import base64
import io
import tempfile
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs import _vtk_xml
from polyxios.codecs._vtk_xml import appended_header_type, format_da, np_to_vtk_type
from polyxios.codecs._vtp import read as vtp_read
from polyxios.codecs._vtu import read, write
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError
from polyxios.fetcher import fetch
from tests.codecs._lazy import mapped


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


def _attr_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2], [0, 1, 3]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        vertex_attrs={
            "pressure": np.arange(4.0),
            "normal": np.ones((4, 3), np.float32),
        },
        element_attrs={"stress": np.array([10.0, 20.0, 30.0])},
    )


def test_roundtrip_appended(tmp_path) -> None:
    """A raw appended section reads back eagerly like any other layout."""
    poly = _attr_mesh()
    tmp = tmp_path / "mesh.vtu"
    write(poly, tmp, appended=True)
    back = read(tmp)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(
        back.vertex_attrs["normal"], poly.vertex_attrs["normal"]
    )
    np.testing.assert_array_equal(
        back.element_attrs["stress"], poly.element_attrs["stress"]
    )
    assert back.vertices.flags.writeable
    assert not mapped(back.vertices)


def test_appended_is_smaller_than_base64(tmp_path) -> None:
    poly = _attr_mesh()
    write(poly, tmp_path / "b64.vtu")
    write(poly, tmp_path / "raw.vtu", appended=True)
    assert (tmp_path / "raw.vtu").stat().st_size < (tmp_path / "b64.vtu").stat().st_size


def test_lazy_arrays_view_the_mapping(tmp_path) -> None:
    """lazy=True hands back the file's own bytes: read-only views on the
    mapping, in the dtype the file holds, and equal to an eager read."""
    poly = _attr_mesh()
    tmp = tmp_path / "mesh.vtu"
    write(poly, tmp, appended=True)
    back = read(tmp, lazy=True)

    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(
        back.vertex_attrs["pressure"], poly.vertex_attrs["pressure"]
    )
    np.testing.assert_array_equal(
        back.vertex_attrs["normal"], poly.vertex_attrs["normal"]
    )
    np.testing.assert_array_equal(
        back.element_attrs["stress"], poly.element_attrs["stress"]
    )

    for arr in (
        back.vertices,
        back.connectivity,
        back.vertex_attrs["pressure"],
        back.vertex_attrs["normal"],
        back.element_attrs["stress"],
    ):
        assert mapped(arr)
        assert not arr.flags.writeable
    assert back.vertex_attrs["normal"].dtype == np.float32
    assert back.connectivity.dtype == back.offsets.dtype


def test_lazy_over_a_handle_at_its_start(tmp_path) -> None:
    poly = _attr_mesh()
    tmp = tmp_path / "mesh.vtu"
    write(poly, tmp, appended=True)
    with tmp.open("rb") as fh:
        back = read(fh, lazy=True)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    assert mapped(back.vertices)


@pytest.mark.parametrize("opts", ({}, {"binary": False}))
def test_lazy_refuses_inline_arrays(tmp_path, opts) -> None:
    """Base64 and ASCII arrays are not the bytes they encode, so there is
    nothing to map; the refusal says so instead of loading eagerly."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.vtu"
    write(poly, tmp, **opts)
    with pytest.raises(LazyReadError, match="inline"):
        read(tmp, lazy=True)
    np.testing.assert_allclose(read(tmp).vertices, poly.vertices)


def test_lazy_refuses_an_in_memory_buffer(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.vtu"
    write(poly, tmp, appended=True)
    with pytest.raises(LazyReadError, match="no file descriptor"):
        read(io.BytesIO(tmp.read_bytes()), lazy=True)


def test_offsets_running_backwards_are_refused(tmp_path) -> None:
    """A Piece whose offsets decrease describes no cells; the reader used
    to hand back empty slices and a mesh that silently lost them."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.vtu"
    write(poly, tmp, binary=False)
    text = tmp.read_text()
    good = 'Name="offsets" format="ascii">4<'
    assert good in text
    tmp.write_text(text.replace(good, 'Name="offsets" format="ascii">-1<'))
    with pytest.raises(CodecError, match="run backwards"):
        read(tmp)


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
# VTU reader/writer hardening
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


def test_a_vtu_write_holds_the_field_data(tmp_path) -> None:
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


# ---------------------------------------------------------------------------
# Hand-built raw appended sections: layouts polyxios' own writer never emits
# ---------------------------------------------------------------------------


def _raw_da(
    blocks: list[bytes],
    name: str,
    arr: np.ndarray,
    *,
    byte_order: str,
    header_type: str,
    n_comp: int = 1,
    n_bytes: int | None = None,
) -> str:
    """Append one block and render the element that points at it.

    ``n_bytes`` overrides the byte count the header declares, for a block
    that lies about itself.
    """
    endian = ">" if byte_order == "BigEndian" else "<"
    h_dt = np.dtype(endian + ("u8" if header_type == "UInt64" else "u4"))
    raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder(endian)).tobytes()
    offset = sum(len(block) for block in blocks)
    declared = len(raw) if n_bytes is None else n_bytes
    blocks.append(np.array([declared], dtype=h_dt).tobytes() + raw)
    vtk_type = np_to_vtk_type(arr.dtype)[0]
    name_attr = f' Name="{name}"' if name else ""
    comp_attr = f' NumberOfComponents="{n_comp}"' if n_comp > 1 else ""
    return (
        f'<DataArray type="{vtk_type}"{name_attr}{comp_attr}'
        f' format="appended" offset="{offset}"/>'
    )


def _raw_piece(
    blocks: list[bytes],
    *,
    points: np.ndarray,
    sections: dict[str, dict[str, np.ndarray]],
    point_data: dict[str, np.ndarray],
    cell_data: dict[str, np.ndarray],
    byte_order: str,
    header_type: str,
    ragged: str | None = None,
    overlong: str | None = None,
) -> str:
    """Render one Piece whose every array sits in the appended section.

    ``sections`` is ``{'Cells': {...}}`` for a grid and ``{'Lines': {...},
    'Polys': {...}}`` for polydata; each holds the named arrays the section
    carries. ``ragged`` names a point array whose block declares three
    bytes fewer than it holds, ``overlong`` one declaring eight more.
    """
    layout = {"byte_order": byte_order, "header_type": header_type}
    counts = "".join(
        f' NumberOf{"Cells" if tag == "Cells" else tag}="{len(arrays["offsets"])}"'
        for tag, arrays in sections.items()
    )
    lines = [
        f'  <Piece NumberOfPoints="{len(points)}"{counts}>',
        "   <Points>",
        "    " + _raw_da(blocks, "", points, n_comp=3, **layout),
        "   </Points>",
    ]
    for tag, arrays in sections.items():
        lines.append(f"   <{tag}>")
        lines.extend(
            "    " + _raw_da(blocks, name, arr, **layout)
            for name, arr in arrays.items()
        )
        lines.append(f"   </{tag}>")
    for tag, arrays in (("PointData", point_data), ("CellData", cell_data)):
        if not arrays:
            continue
        lines.append(f"   <{tag}>")
        for name, arr in arrays.items():
            short = None
            if name == ragged:
                short = arr.nbytes - 3
            elif name == overlong:
                short = arr.nbytes + 8
            lines.append(
                "    "
                + _raw_da(
                    blocks,
                    name,
                    arr,
                    n_comp=1 if arr.ndim == 1 else arr.shape[1],
                    n_bytes=short,
                    **layout,
                )
            )
        lines.append(f"   </{tag}>")
    lines.append("  </Piece>")
    return "\n".join(lines) + "\n"


def _raw_file(
    *, kind: str, pieces: str, blocks: list[bytes], byte_order: str, header_type: str
) -> bytes:
    head = (
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="{kind}" version="1.0" byte_order="{byte_order}"'
        f' header_type="{header_type}">\n'
        f" <{kind}>\n{pieces} </{kind}>\n"
    )
    return (
        head.encode()
        + b'  <AppendedData encoding="raw">\n   _'
        + b"".join(blocks)
        + b"\n  </AppendedData>\n</VTKFile>\n"
    )


_TET_POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _hand_built_vtu(
    path,
    *,
    byte_order: str = "LittleEndian",
    header_type: str = "UInt32",
    n_pieces: int = 1,
    conn_dtype: type = np.int32,
    ragged: str | None = None,
    overlong: str | None = None,
    n_points: int = 4,
):
    """Write a triangle and a tetrahedron per Piece, with three attributes.

    ``n_points`` pads each Piece's points past the four its cells index,
    so a second Piece starts that far into the joined points.
    """
    blocks: list[bytes] = []
    pieces = "".join(
        _raw_piece(
            blocks,
            points=np.resize(_TET_POINTS, (n_points, 3)) + index,
            sections={
                "Cells": {
                    "connectivity": np.array([0, 1, 2, 0, 1, 2, 3], dtype=conn_dtype),
                    "offsets": np.array([3, 7], dtype=np.int32),
                    "types": np.array([5, 10], dtype=np.uint8),
                }
            },
            point_data={
                "pressure": np.arange(float(n_points)) + 10 * index,
                "normal": np.full((n_points, 3), index, dtype=np.float32),
            },
            cell_data={"stress": np.array([1.0, 2.0]) + index},
            byte_order=byte_order,
            header_type=header_type,
            ragged=ragged,
            overlong=overlong,
        )
        for index in range(n_pieces)
    )
    path.write_bytes(
        _raw_file(
            kind="UnstructuredGrid",
            pieces=pieces,
            blocks=blocks,
            byte_order=byte_order,
            header_type=header_type,
        )
    )
    return path


def _hand_built_vtp(path):
    """Write a polyline beside a triangle and a quad, in two sections."""
    blocks: list[bytes] = []
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32
    )
    pieces = _raw_piece(
        blocks,
        points=points,
        sections={
            "Lines": {
                "connectivity": np.array([0, 1, 4], dtype=np.int64),
                "offsets": np.array([3], dtype=np.int64),
            },
            "Polys": {
                "connectivity": np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64),
                "offsets": np.array([3, 7], dtype=np.int64),
            },
        },
        point_data={"pressure": np.arange(5.0)},
        cell_data={"stress": np.array([1.0, 2.0, 3.0])},
        byte_order="LittleEndian",
        header_type="UInt32",
    )
    path.write_bytes(
        _raw_file(
            kind="PolyData",
            pieces=pieces,
            blocks=blocks,
            byte_order="LittleEndian",
            header_type="UInt32",
        )
    )
    return path


def _assert_same_mesh(lazy, eager) -> None:
    np.testing.assert_array_equal(lazy.vertices, eager.vertices)
    np.testing.assert_array_equal(lazy.connectivity, eager.connectivity)
    np.testing.assert_array_equal(lazy.offsets, eager.offsets)
    np.testing.assert_array_equal(lazy.element_types, eager.element_types)
    assert lazy.vertex_attrs.keys() == eager.vertex_attrs.keys()
    assert lazy.element_attrs.keys() == eager.element_attrs.keys()
    for name, arr in eager.vertex_attrs.items():
        np.testing.assert_array_equal(lazy.vertex_attrs[name], arr)
    for name, arr in eager.element_attrs.items():
        np.testing.assert_array_equal(lazy.element_attrs[name], arr)


# Builder, reader, and whether the points (one Piece) and the connectivity
# (one Piece of one cell section) come back as views rather than joins.
_HAND_BUILT = {
    "little_endian": (lambda p: _hand_built_vtu(p), read, True, True),
    "big_endian": (
        lambda p: _hand_built_vtu(p, byte_order="BigEndian"),
        read,
        True,
        True,
    ),
    "uint64_header": (
        lambda p: _hand_built_vtu(p, header_type="UInt64"),
        read,
        True,
        True,
    ),
    "two_pieces": (lambda p: _hand_built_vtu(p, n_pieces=2), read, False, False),
    "vtp_lines_and_polys": (_hand_built_vtp, vtp_read, True, False),
}


@pytest.mark.parametrize("layout", sorted(_HAND_BUILT))
def test_a_hand_built_appended_section_reads_lazily(tmp_path, layout: str) -> None:
    """Layouts polyxios' own writer never emits - big-endian bytes, eight-byte
    block headers, several pieces, several cell sections - read lazily to
    the mesh an eager read gives; a single piece views the mapping, and a
    single piece of a single section views it for its cells too."""
    build, reader, points_view, cells_view = _HAND_BUILT[layout]
    path = build(tmp_path / "hand.vtx")

    eager = reader(path)
    lazy = reader(path, lazy=True)

    assert len(eager.element_types) >= 2
    assert "pressure" in eager.vertex_attrs and "stress" in eager.element_attrs
    _assert_same_mesh(lazy, eager)
    assert mapped(lazy.vertices) is points_view
    assert mapped(lazy.connectivity) is cells_view
    for arr in (lazy.vertices, *lazy.vertex_attrs.values()):
        assert mapped(arr) is points_view
        assert arr.flags.writeable is not points_view
    if layout == "big_endian":
        assert lazy.vertices.dtype.byteorder == ">"


def test_a_two_piece_hand_built_file_shifts_the_second_piece(tmp_path) -> None:
    eager = read(_hand_built_vtu(tmp_path / "two.vtu", n_pieces=2))
    assert eager.vertices.shape == (8, 3)
    np.testing.assert_array_equal(eager.connectivity[7:10], [4, 5, 6])
    np.testing.assert_array_equal(eager.vertex_attrs["pressure"][4:], [10, 11, 12, 13])


def test_a_vtp_of_lines_and_polys_types_each_section(tmp_path) -> None:
    eager = vtp_read(_hand_built_vtp(tmp_path / "hand.vtp"))
    np.testing.assert_array_equal(
        eager.element_types,
        [ELEMENT_TYPES["line"], ELEMENT_TYPES["triangle"], ELEMENT_TYPES["quad"]],
    )
    np.testing.assert_array_equal(eager.offsets, [0, 3, 6, 10])


def test_the_header_widens_only_for_a_block_past_four_gigabytes() -> None:
    """A block's byte count has to fit its header; one that does not needs
    the eight-byte header, and the width is declared once for the file."""
    assert appended_header_type(sizes=[24, 2**32 - 1]) == "UInt32"
    assert appended_header_type(sizes=[24, 2**32]) == "UInt64"
    assert appended_header_type(sizes=[]) == "UInt32"


@pytest.mark.parametrize("opts", ({"appended": True}, {"binary": True}))
def test_a_file_needing_wide_headers_declares_them_and_reads_back(
    tmp_path, monkeypatch, opts
) -> None:
    """The writer sizes every block before rendering any; a block the
    four-byte header cannot count widens every header and says so on the
    root element. Stood in for rather than allocated, four gigabytes being
    what it takes to reach for real."""
    monkeypatch.setattr(
        "polyxios.codecs._vtu.appended_header_type", lambda *, sizes: "UInt64"
    )
    poly = _attr_mesh()
    tmp = tmp_path / "wide.vtu"
    write(poly, tmp, **opts)

    head = tmp.read_bytes()[:512]
    assert b'header_type="UInt64"' in head
    back = read(tmp)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(
        back.vertex_attrs["normal"], poly.vertex_attrs["normal"]
    )
    if "appended" in opts:
        lazy = read(tmp, lazy=True)
        _assert_same_mesh(lazy, back)
        assert mapped(lazy.vertices)


def test_wide_headers_are_eight_bytes_in_the_appended_section() -> None:
    blocks: list[bytes] = []
    format_da(
        "x",
        np.arange(3, dtype=np.int32),
        vtk_type="Int32",
        dtype=np.dtype("<i4"),
        binary=False,
        n_comp=1,
        indent=0,
        appended=blocks,
        header_type="UInt64",
    )
    assert len(blocks[0]) == 8 + 12
    assert int(np.frombuffer(blocks[0], dtype="<u8", count=1)[0]) == 12


def test_unsigned_offsets_running_backwards_are_refused(tmp_path) -> None:
    """On an unsigned offsets array a step backwards wraps to a large
    positive difference, which a differenced check never saw."""
    cells = (
        '    <DataArray type="Int32" Name="connectivity" format="ascii">'
        "0 1 2 0 1 3</DataArray>\n"
        '    <DataArray type="UInt32" Name="offsets" format="ascii">3 2</DataArray>\n'
        '    <DataArray type="UInt8" Name="types" format="ascii">5 5</DataArray>\n'
    )
    path = tmp_path / "unsigned.vtu"
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0 0 0 1", 4, cells, 2)))
    with pytest.raises(CodecError, match="run backwards"):
        read(path)


def test_a_float_connectivity_reads_lazily_as_it_does_eagerly(tmp_path) -> None:
    """An index is a whole number, so a connectivity of floats has no
    dtype worth keeping; the lazy read casts it as the eager one does
    instead of asking numpy for the integer range of a float."""
    path = _hand_built_vtu(tmp_path / "float.vtu", conn_dtype=np.float32)
    eager = read(path)
    lazy = read(path, lazy=True)
    _assert_same_mesh(lazy, eager)
    assert lazy.connectivity.dtype == eager.connectivity.dtype
    assert lazy.offsets.dtype == eager.offsets.dtype
    assert np.issubdtype(lazy.connectivity.dtype, np.integer)


@pytest.mark.parametrize("lazy", (False, True))
def test_a_block_of_ragged_bytes_names_its_array(tmp_path, lazy: bool) -> None:
    path = _hand_built_vtu(tmp_path / "ragged.vtu", ragged="pressure")
    with pytest.raises(CodecError, match="DataArray 'pressure'.*29 bytes"):
        read(path, lazy=lazy)


def test_an_inline_block_of_ragged_bytes_names_its_array(tmp_path) -> None:
    payload = base64.b64encode(np.array([5], dtype="<u4").tobytes() + bytes(5))
    extra = (
        "   <PointData>\n"
        f'    <DataArray type="Float64" Name="pressure" format="binary">'
        f"{payload.decode()}</DataArray>\n"
        "   </PointData>\n"
    )
    path = tmp_path / "ragged.vtu"
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)))
    with pytest.raises(CodecError, match="DataArray 'pressure'.*5 bytes"):
        read(path)


def test_an_inline_block_with_a_wide_header_is_read_past_it(tmp_path) -> None:
    """A file declaring header_type="UInt64" puts eight bytes before every
    inline block too; a reader skipping four reads the count as values."""
    values = np.array([1.5, 2.5, 3.5])
    payload = base64.b64encode(np.array([24], dtype="<u8").tobytes() + values.tobytes())
    extra = (
        "   <PointData>\n"
        f'    <DataArray type="Float64" Name="pressure" format="binary">'
        f"{payload.decode()}</DataArray>\n"
        "   </PointData>\n"
    )
    path = tmp_path / "wide.vtu"
    path.write_text(
        _vtu(_piece("0 0 0 1 0 0 0 1 0", 3, _TRI_CELLS, 1, extra)).replace(
            'byte_order="LittleEndian"',
            'byte_order="LittleEndian" header_type="UInt64"',
        )
    )
    np.testing.assert_array_equal(read(path).vertex_attrs["pressure"], values)


def test_a_file_with_no_grid_element_names_the_format(tmp_path) -> None:
    """A root holding no dataset used to raise a bare ValueError."""
    path = tmp_path / "hollow.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        "</VTKFile>\n"
    )

    with pytest.raises(CodecError, match=r"\.vtu: .*no <UnstructuredGrid>"):
        read(path)


@pytest.mark.parametrize("offset", ("abc", "-5"))
def test_an_offset_that_is_no_byte_offset_names_the_array(tmp_path, offset) -> None:
    """int() answered a word with a ValueError naming nothing, and a negative
    offset read the XML before the section as the block's own header."""
    path = _hand_built_vtu(tmp_path / "offset.vtu")
    raw = path.read_bytes()
    marker = b'Name="pressure" format="appended" offset="'
    assert marker in raw
    head, tail = raw.split(marker, 1)
    rest = tail.split(b'"', 1)[1]
    path.write_bytes(head + marker + offset.encode() + b'"' + rest)

    with pytest.raises(
        CodecError, match=f"DataArray 'pressure' declares offset='{offset}'"
    ):
        read(path)


@pytest.mark.parametrize("n_types", (1, 3))
def test_a_types_array_that_miscounts_the_cells_is_refused(tmp_path, n_types) -> None:
    """The types array is the one cell array nothing else in the Piece sizes;
    read unchecked, a mesh came back typing more or fewer cells than its
    offsets cut, and every cell attribute was sized against the wrong count."""
    cells = (
        '    <DataArray type="Int32" Name="connectivity" format="ascii">'
        "0 1 2 0 1 3</DataArray>\n"
        '    <DataArray type="Int32" Name="offsets" format="ascii">3 6</DataArray>\n'
        f'    <DataArray type="UInt8" Name="types" format="ascii">{"5 " * n_types}'
        "</DataArray>\n"
    )
    path = tmp_path / "miscounted.vtu"
    path.write_text(_vtu(_piece("0 0 0 1 0 0 0 1 0 0 0 1", 4, cells, 2)))

    with pytest.raises(
        CodecError, match=f"cut 2 cells and the types array holds {n_types}"
    ):
        read(path)


def test_a_refused_lazy_read_closes_its_mapping(tmp_path, monkeypatch) -> None:
    """Nothing views the mapping a refusal leaves behind, so it is closed
    rather than left to hold the file until the collector finds it."""
    made: list[object] = []
    real = _vtk_xml.map_read

    def spy(src, *, fmt):
        mapping = real(src, fmt=fmt)
        made.append(mapping)
        return mapping

    monkeypatch.setattr(_vtk_xml, "map_read", spy)
    tmp = tmp_path / "inline.vtu"
    write(_tet_mesh(), tmp)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)
    assert len(made) == 1
    assert made[0].closed


@pytest.mark.parametrize("conn_dtype", [np.uint8, np.int8, np.int16])
def test_a_second_piece_shifts_a_narrow_connectivity_without_wrapping(
    tmp_path, conn_dtype
) -> None:
    """The points before a Piece are added to its indices in an integer
    type that holds the sum, not in the file's own: added in a one-byte
    dtype, three hundred either wraps or is refused by numpy outright."""
    path = _hand_built_vtu(
        tmp_path / "narrow.vtu", n_pieces=2, conn_dtype=conn_dtype, n_points=300
    )
    eager = read(path)
    assert eager.vertices.shape == (600, 3)
    np.testing.assert_array_equal(eager.connectivity[7:10], [300, 301, 302])
    np.testing.assert_array_equal(eager.connectivity[:7], [0, 1, 2, 0, 1, 2, 3])
    _assert_same_mesh(read(path, lazy=True), eager)


@pytest.mark.parametrize(
    ("layout", "conn_dtype"),
    [({"conn_dtype": np.uint8}, np.uint8), ({"byte_order": "BigEndian"}, ">i4")],
)
def test_lazy_offsets_are_native_int32_whatever_the_connectivity_holds(
    tmp_path, layout, conn_dtype
) -> None:
    """The offsets are derived, never read, so the file's index dtype stays
    on the connectivity alone: unsigned offsets would wrap under a step
    backwards, and big-endian ones swap on every use."""
    lazy = read(_hand_built_vtu(tmp_path / "narrow.vtu", **layout), lazy=True)
    assert lazy.connectivity.dtype == np.dtype(conn_dtype)
    assert lazy.offsets.dtype == np.int32
    np.testing.assert_array_equal(lazy.offsets, [0, 3, 7])


def test_a_block_declaring_more_bytes_than_the_section_holds_is_refused(
    tmp_path,
) -> None:
    """A block's header counting past the end of the section is refused,
    not read short: cut to what is left it can still hold a whole number
    of values and pass every later check."""
    path = _hand_built_vtu(tmp_path / "long.vtu", overlong="stress")
    with pytest.raises(CodecError, match="'stress' declares 24 bytes"):
        read(path)
    with pytest.raises(CodecError, match="'stress' declares 24 bytes"):
        read(path, lazy=True)


def _no_grid(path):
    return _hand_built_vtu(path)


def _not_well_formed(path):
    _hand_built_vtu(path)
    path.write_bytes(path.read_bytes().replace(b"</UnstructuredGrid>", b"", 1))
    return path


@pytest.mark.parametrize(
    ("build", "reader", "error", "message"),
    [
        (_no_grid, vtp_read, UnsupportedFormatError, "declares type="),
        (_not_well_formed, read, CodecError, "not well-formed XML"),
    ],
)
def test_a_read_refused_after_mapping_closes_its_mapping(
    tmp_path, monkeypatch, build, reader, error, message
) -> None:
    """A file that maps but is refused before any array views the mapping
    leaves nothing to hold it open, so it is closed rather than left to
    whoever holds the traceback."""
    made: list[object] = []
    real = _vtk_xml.map_read

    def spy(src, *, fmt):
        mapping = real(src, fmt=fmt)
        made.append(mapping)
        return mapping

    monkeypatch.setattr(_vtk_xml, "map_read", spy)
    path = build(tmp_path / "refused.vtx")
    with pytest.raises(error, match=message):
        reader(path, lazy=True)
    assert len(made) == 1
    assert made[0].closed


@pytest.mark.parametrize("lazy", [False, True])
def test_a_raw_section_with_no_marker_is_refused(tmp_path, lazy) -> None:
    """Without the ``_`` the offsets count from nowhere; read from the tag's
    end the XML became a block header declaring a couple of gigabytes."""
    path = _hand_built_vtu(tmp_path / "nomark.vtu")
    path.write_bytes(path.read_bytes().replace(b"\n   _", b"\n   ", 1))
    with pytest.raises(CodecError, match="no '_' marker"):
        read(path, lazy=lazy)
