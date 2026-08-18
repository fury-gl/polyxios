from __future__ import annotations

import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._nastran import read, sniff, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def _hex_mesh():
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    return make_polydata(verts, [("hexahedron", np.arange(8).reshape(1, 8))])


def _write(tmp_path, text, *, name="mesh.bdf"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_roundtrip_hexahedron_uses_continuation(tmp_path) -> None:
    poly = _hex_mesh()
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)

    text = tmp.read_text()
    assert "CHEXA,1,1,1,2,3,4,5,6,+" in text
    assert "+,7,8" in text

    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_file_has_grid_and_element_cards(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)
    text = tmp.read_text()
    assert "GRID" in text
    assert "CTETRA" in text
    assert "BEGIN BULK" in text
    assert "ENDDATA" in text


def test_written_reals_carry_a_decimal_point(tmp_path) -> None:
    verts = np.array([[0, 2, 1e-7], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)

    for line in tmp.read_text().splitlines():
        if line.startswith("GRID,"):
            for value in line.split(",")[3:]:
                assert "." in value.partition("e")[0], line

    np.testing.assert_allclose(read(tmp).vertices, verts)


def test_write_rejects_non_finite_coordinates(tmp_path) -> None:
    verts = np.array([[0, 0, np.nan], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    with pytest.raises(CodecError):
        write(poly, tmp_path / "test.bdf")


def test_write_rejects_unmapped_element_type(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    with pytest.raises(CodecError):
        write(poly, tmp_path / "test.bdf")


def test_write_rejects_unknown_element_type_id(tmp_path) -> None:
    poly = PolyData(
        vertices=np.zeros((3, 3), dtype=np.float64),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([250], dtype=np.uint8),
    )
    with pytest.raises(CodecError):
        write(poly, tmp_path / "test.bdf")


def test_write_preserves_element_order(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 3]])),
        ],
    )
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_empty_deck_reads_as_empty_polydata(tmp_path) -> None:
    tmp = _write(tmp_path, "$ header only\nBEGIN BULK\nENDDATA\n")
    poly = read(tmp)
    assert poly.vertices.shape == (0, 3)
    assert len(poly.element_types) == 0
    assert poly.offsets.tolist() == [0]


def test_write_then_read_empty_polydata(tmp_path) -> None:
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    tmp = tmp_path / "empty.bdf"
    write(poly, tmp)
    assert read(tmp).vertices.shape == (0, 3)


def test_elements_without_grid_cards_raise(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nCTRIA3,1,1,1,2,3\nENDDATA\n")
    with pytest.raises(CodecError, match="undefined GRID"):
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
    tmp = tmp_path / "test.bdf"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2


def test_elements_may_precede_grid_cards(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "CTRIA3,1,1,10,20,30\n"
        "GRID,10,,0.,0.,0.\n"
        "GRID,20,,1.,0.,0.\n"
        "GRID,30,,0.,1.,0.\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_small_field_hexahedron_with_blank_continuation(tmp_path) -> None:
    grids = "".join(
        f"GRID    {i + 1:<8}{0:<8}{float(i):<8}{0.0:<8}{0.0:<8}\n" for i in range(8)
    )
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        + grids
        + "CHEXA          1       1       1       2       3       4       5       6\n"
        "               7       8\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["hexahedron"]]
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))


def test_large_field_grid_cards(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID*                  1               0             0.0             0.0\n"
        "*                    0.0\n"
        "GRID*                  2               0             1.0             0.0\n"
        "*                    0.0\n"
        "GRID*                  3               0             0.0             1.0\n"
        "*                    0.0\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], atol=0)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]


def test_large_field_continuation_with_blank_marker(tmp_path) -> None:
    """A large-field continuation may leave its marker field blank."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID*                  1               0             1.0             2.0\n"
        "                     3.0\n"
        "ENDDATA\n",
    )
    np.testing.assert_allclose(read(tmp).vertices, [[1.0, 2.0, 3.0]])


def test_large_field_element_card(tmp_path) -> None:
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(8))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CHEXA*  1               1               1"
        "               2\n"
        "*       3               4               5               6\n"
        "*       7               8\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["hexahedron"]]
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))


def test_continuation_may_switch_to_large_field(tmp_path) -> None:
    """A '*' marker declares the continuation line itself as large field."""
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(8))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CHEXA   1       1       1       2       3"
        "       4       5       6       *\n"
        "*       7               8               \n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_named_continuation_marker(tmp_path) -> None:
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(8))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CHEXA,1,1,1,2,3,4,5,6,+C1\n+C1,7,8\nENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))


def test_implicit_exponent_reals(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,1.5+3,-2.1-4,1.5D+2\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices[0], [1500.0, -2.1e-4, 150.0])


def test_blank_and_missing_coordinates_default_to_zero(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,,,\n"
        "GRID    2       0       1.      0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_case_control_and_trailing_comments_ignored(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "SOL 101\n"
        "CEND\n"
        "TITLE = GRID STUDY\n"
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.  $ first corner\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n"
        "GRID,4,,9.,9.,9.\n",
    )
    poly = read(tmp)
    assert len(poly.vertices) == 3
    np.testing.assert_allclose(poly.vertices[0], [0.0, 0.0, 0.0])


def test_truncated_element_card_raises(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2\n"
        "ENDDATA\n",
    )
    with pytest.raises(CodecError):
        read(tmp)


def test_undefined_grid_reference_raises(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,99\n"
        "ENDDATA\n",
    )
    with pytest.raises(CodecError):
        read(tmp)


def test_malformed_real_raises(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,0.,abc,0.\nENDDATA\n")
    with pytest.raises(CodecError):
        read(tmp)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_real_raises(tmp_path, value) -> None:
    tmp = _write(tmp_path, f"BEGIN BULK\nGRID,1,,0.,{value},0.\nENDDATA\n")
    with pytest.raises(CodecError):
        read(tmp)


def test_tab_separated_fixed_field(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID\t1\t0\t0.\t0.\t0.\n"
        "GRID\t2\t0\t1.\t0.\t0.\n"
        "GRID\t3\t0\t0.\t1.\t0.\n"
        "CTRIA3\t1\t1\t1\t2\t3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_duplicate_grid_id_keeps_one_vertex(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,1,,9.,9.,9.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    with pytest.warns(UserWarning, match="duplicate GRID"):
        poly = read(tmp)
    assert len(poly.vertices) == 3
    np.testing.assert_allclose(poly.vertices[0], [9.0, 9.0, 9.0])


def test_local_coordinate_system_warns(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,7,0.,0.,0.\n"
        "GRID,2,7,1.,0.,0.\n"
        "GRID,3,7,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    with pytest.warns(UserWarning, match="local coordinate system"):
        read(tmp)


def test_higher_order_card_keeps_corner_nodes(tmp_path) -> None:
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(10))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CTETRA,1,1,1,2,3,4,5,6,+\n+,7,8,9,10\nENDDATA\n",
    )
    poly = read(tmp)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["tetra"]]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_explicit_plus_sign_real_is_not_a_marker(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,+3.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices[0], [0.0, 0.0, 3.0])


def test_indented_free_field_card_is_not_a_continuation(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "          GRID,1,,0.,0.,0.\n"
        "          GRID,2,,1.,0.,0.\n"
        "          GRID,3,,0.,1.,0.\n"
        "          CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    assert len(poly.vertices) == 3
    assert len(poly.element_types) == 1


def test_include_statement_warns(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "INCLUDE 'part.bdf'\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    with pytest.warns(UserWarning, match="INCLUDE"):
        poly = read(tmp)
    assert len(poly.element_types) == 1


def test_reads_meshio_style_deck(tmp_path) -> None:
    """Large-field GRID cards plus a small-field CHEXA, byte-for-byte as
    meshio 5.3.5 writes them: a 16-column id field, a blank CP, and a
    numeric named continuation marker on the element card."""
    grids = "".join(
        f"GRID*   {i + 1:<16}{'':<16}{float(i):>16.1E}{0.0:>16.1E}\n"
        f"*       {0.0:>16.1E}\n"
        for i in range(8)
    )
    tmp = _write(
        tmp_path,
        "$ Nastran file written by meshio v5.3.5\n"
        "BEGIN BULK\n" + grids + "CHEXA   1               1       2       3"
        "       4       5       6       +11     \n"
        "+11     7       8       \n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["hexahedron"]]
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))
    np.testing.assert_allclose(poly.vertices[:, 0], np.arange(8, dtype=np.float64))


def test_file_without_begin_bulk_is_read(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,0.,1.,0.\nCTRIA3,1,1,1,2,3\n",
    )
    poly = read(tmp)
    assert len(poly.vertices) == 3
    assert len(poly.element_types) == 1


def test_grid_only_file_has_no_elements(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,0.,0.,0.\nENDDATA\n")
    poly = read(tmp)
    assert len(poly.vertices) == 1
    assert len(poly.element_types) == 0
    assert poly.offsets.tolist() == [0]


def test_numeric_named_continuation_marker(tmp_path) -> None:
    """A '+11' marker in field 10 is a marker, not a grid id."""
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(11))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CHEXA,1,1,1,2,3,4,5,6,+11\n+11,7,8\nENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))


def test_numeric_named_marker_on_short_line(tmp_path) -> None:
    """A '+11' marker before field 10 is named by the line that follows."""
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(11))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CHEXA,1,1,1,2,3,4,5,+11\n+11,6,7,8\nENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))


def test_star_named_continuation_marker(tmp_path) -> None:
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(8))
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + "CHEXA,1,1,1,2,3,4,5,6,*2\n*2,7,8\nENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_array_equal(poly.connectivity, np.arange(8))


def _precision_mesh():
    verts = np.array(
        [
            [0.1234567890123456, 1e-17, 12345678.87654321],
            [1 / 3, 2.0**-40, -9.87654321e300],
            [0.0, -0.0, 1e20],
        ]
    )
    return verts, make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])


def test_write_precision_is_exact(tmp_path) -> None:
    """Coordinates round-trip bit-for-bit."""
    verts, poly = _precision_mesh()
    tmp = tmp_path / "prec.bdf"
    # 1e-17 and -9.87e300 have no free-field spelling at all, which is a
    # second warning on top of the one about width.
    with pytest.warns(UserWarning, match="no free-field spelling"):
        with pytest.warns(UserWarning, match="more than 8 characters"):
            write(poly, tmp)
    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_short_reals_do_not_warn_about_free_field_width(tmp_path) -> None:
    poly = _tet_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "short.bdf")


def test_large_field_write_round_trips(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "large.bdf"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp, field_format="large")

    lines = tmp.read_text().splitlines()
    assert any(line.startswith("GRID*") for line in lines)
    assert any(line.startswith("*") for line in lines)
    assert all(len(line) <= 80 for line in lines)

    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_large_field_write_warns_when_rounding(tmp_path) -> None:
    verts, poly = _precision_mesh()
    tmp = tmp_path / "large_prec.bdf"
    with pytest.warns(UserWarning, match="16-character large field"):
        write(poly, tmp, field_format="large")

    back = read(tmp).vertices
    np.testing.assert_allclose(back, verts, rtol=1e-9)


def test_large_field_write_stays_finite_at_the_top_of_the_range(tmp_path) -> None:
    """Trimming a coordinate to sixteen characters must not round it to inf.

    The largest double shortens to '1.797693135E+308', which fits the field
    and reads back as infinity; the next shorter form does not.
    """
    huge = np.finfo(np.float64).max
    verts = np.array(
        [[huge, -huge, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])

    tmp = tmp_path / "huge_large.bdf"
    with pytest.warns(UserWarning, match="16-character large field"):
        write(poly, tmp, field_format="large")

    back = read(tmp).vertices
    assert np.isfinite(back).all()
    np.testing.assert_allclose(back, verts, rtol=1e-8)


def test_free_field_write_keeps_the_top_of_the_range_exact(tmp_path) -> None:
    huge = np.finfo(np.float64).max
    verts = np.array(
        [[huge, -huge, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])

    tmp = tmp_path / "huge_free.bdf"
    with pytest.warns(UserWarning):
        write(poly, tmp)
    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_unknown_field_format_raises(tmp_path) -> None:
    with pytest.raises(CodecError, match="unknown field_format"):
        write(_tet_mesh(), tmp_path / "bad.bdf", field_format="small")


@pytest.mark.parametrize(
    ("card", "expected"),
    [
        ("CTRIA6,1,1,1,2,3,4,5,6", "triangle"),
        ("CTRIAR,1,1,1,2,3", "triangle"),
        ("CQUAD8,1,1,1,2,3,4,5,6,+\n+,7,8", "quad"),
        ("CQUADR,1,1,1,2,3,4", "quad"),
        ("CPYRA,1,1,1,2,3,4,5", "pyramid"),
    ],
)
def test_higher_order_and_revised_cards(tmp_path, card, expected) -> None:
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(8))
    tmp = _write(tmp_path, "BEGIN BULK\n" + grids + card + "\nENDDATA\n")
    poly = read(tmp)
    assert poly.element_types.tolist() == [ELEMENT_TYPES[expected]]
    n_nodes = {"triangle": 3, "quad": 4, "pyramid": 5}[expected]
    np.testing.assert_array_equal(poly.connectivity, np.arange(n_nodes))


def test_unsupported_element_cards_warn(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "CBAR,1,1,1,2,0.,0.,1.\n"
        "CROD,2,1,1,2\n"
        "CROD,3,1,2,1\n"
        "ENDDATA\n",
    )
    with pytest.warns(UserWarning, match=r"CBAR \(1\), CROD \(2\)"):
        poly = read(tmp)
    assert len(poly.element_types) == 0


def test_out_of_range_id_raises(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,3000000000,,0.,0.,0.\nENDDATA\n")
    with pytest.raises(CodecError, match="32-bit integer"):
        read(tmp)


def test_coordinate_system_cards_do_not_warn(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "CLOAD,3,1.,1.,10\n"
        "CORD2R,7,0,0.,0.,0.,0.,0.,1.,+\n+,1.,0.,0.\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(tmp)
    assert len(poly.element_types) == 1


def test_write_rejects_out_of_range_connectivity(tmp_path) -> None:
    poly = PolyData(
        vertices=np.zeros((3, 3), dtype=np.float64),
        connectivity=np.array([0, 1, 7], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="outside the vertex array"):
        write(poly, tmp_path / "bad.bdf")


def _pid_mesh(tmp_path):
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(4))
    return _write(
        tmp_path,
        "BEGIN BULK\n"
        + grids
        + "CTRIA3,1,10,1,2,3\nCTRIA3,2,20,2,3,4\nCTRIA3,3,,1,2,4\nENDDATA\n",
    )


def test_property_ids_become_attrs_and_tags(tmp_path) -> None:
    poly = read(_pid_mesh(tmp_path))
    np.testing.assert_array_equal(poly.element_attrs["pid"], [10, 20, 1])
    assert poly.element_attrs["pid"].dtype == np.int32
    assert set(poly.element_tags) == {"pid_1", "pid_10", "pid_20"}
    np.testing.assert_array_equal(poly.element_tags["pid_10"], [0])
    np.testing.assert_array_equal(poly.element_tags["pid_20"], [1])
    np.testing.assert_array_equal(poly.element_tags["pid_1"], [2])


def test_property_ids_round_trip(tmp_path) -> None:
    poly = read(_pid_mesh(tmp_path))
    tmp = tmp_path / "pid.bdf"
    write(poly, tmp)
    assert "CTRIA3,1,10,1,2,3" in tmp.read_text()
    np.testing.assert_array_equal(read(tmp).element_attrs["pid"], [10, 20, 1])


def test_grid_only_deck_has_no_pid_attr(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,0.,0.,0.\nENDDATA\n")
    poly = read(tmp)
    assert poly.element_attrs == {}
    assert poly.element_tags == {}


def test_property_ids_derived_from_tags(tmp_path) -> None:
    """Without a pid attribute, pid_<id> tags drive the property ids."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [1, 2, 3], [0, 1, 3]]))],
        element_tags={"pid_7": np.array([0, 2]), "other": np.array([1])},
    )
    tmp = tmp_path / "tags.bdf"
    write(poly, tmp)
    cards = [ln for ln in tmp.read_text().splitlines() if ln.startswith("CTRIA3")]
    assert cards == ["CTRIA3,1,7,1,2,3", "CTRIA3,2,1,2,3,4", "CTRIA3,3,7,1,2,4"]


def test_first_pid_tag_wins_for_shared_elements(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"pid_3": np.array([0]), "pid_9": np.array([0])},
    )
    tmp = tmp_path / "shared.bdf"
    write(poly, tmp)
    assert "CTRIA3,1,3,1,2,3" in tmp.read_text()


def _tri_polydata(pid_attr):
    return PolyData(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
        element_attrs={"pid": pid_attr},
    )


def test_mismatched_pid_attr_warns_and_falls_back(tmp_path) -> None:
    poly = _tri_polydata(np.array([4, 5, 6], dtype=np.int32))
    tmp = tmp_path / "bad.bdf"
    with pytest.warns(UserWarning, match=r"element_attrs\['pid'\] has shape \(3,\)"):
        write(poly, tmp)
    assert "CTRIA3,1,1,1,2,3" in tmp.read_text()


def test_non_integral_pid_attr_warns_and_falls_back(tmp_path) -> None:
    poly = _tri_polydata(np.array([1.5]))
    tmp = tmp_path / "frac.bdf"
    with pytest.warns(UserWarning, match="not integral"):
        write(poly, tmp)
    assert "CTRIA3,1,1,1,2,3" in tmp.read_text()


def test_integral_float_pid_attr_is_accepted(tmp_path) -> None:
    poly = _tri_polydata(np.array([12.0]))
    tmp = tmp_path / "float.bdf"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp)
    assert "CTRIA3,1,12,1,2,3" in tmp.read_text()


def test_non_positive_pid_warns_and_defaults(tmp_path) -> None:
    poly = _tri_polydata(np.array([0], dtype=np.int32))
    tmp = tmp_path / "zero.bdf"
    with pytest.warns(UserWarning, match="below 1"):
        write(poly, tmp)
    assert "CTRIA3,1,1,1,2,3" in tmp.read_text()


def test_malformed_property_id_raises(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,abc,1,2,3\n"
        "ENDDATA\n",
    )
    with pytest.raises(CodecError, match="property id"):
        read(tmp)


def _eight_grids() -> str:
    return "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(8))


def test_free_field_trailing_comma_continues_the_card(tmp_path) -> None:
    """A blank tenth field is the continuation field, not a grid id."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + _eight_grids() + "CHEXA,1,1,1,2,3,4,5,6,\n,7,8\nENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_named_marker_need_not_match_its_continuation(tmp_path) -> None:
    """Nastran does not tie a continuation to the marker naming it."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        + _eight_grids()
        + "CHEXA,1,1,1,2,3,4,+11\n+12,5,6,7,8\nENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_named_marker_followed_by_blank_continuation(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + _eight_grids() + "CHEXA,1,1,1,2,3,4,+11\n,5,6,7,8\nENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_plus_field_ending_a_card_stays_data(tmp_path) -> None:
    """With no continuation behind it, '+3.' is a coordinate, not a marker."""
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,0.,0.,+3.\nENDDATA\n")
    np.testing.assert_array_equal(read(tmp).vertices, [[0.0, 0.0, 3.0]])


def test_trailing_blank_free_fields_are_ignored(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,1.,2.,3.,,,,\nENDDATA\n")
    np.testing.assert_array_equal(read(tmp).vertices, [[1.0, 2.0, 3.0]])


def test_free_field_line_past_the_tenth_field_raises(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,1.,2.,3.,,,,,9.\nENDDATA\n")
    with pytest.raises(CodecError, match="more than 10 fields"):
        read(tmp)


def test_small_field_continuation_of_a_large_field_card(tmp_path) -> None:
    """A '+' marker means small field, whatever the card it continues used."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        + _eight_grids()
        + "CHEXA*  1               1               1               2\n"
        "+       3       4       5       6\n"
        "+       7       8\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_blank_marker_inherits_the_previous_continuation_width(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        + _eight_grids()
        + "CHEXA*  1               1               1               2\n"
        "+       3       4       5       6\n"
        "        7       8\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_named_marker_inside_a_short_fixed_field_card(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        + _eight_grids()
        + "CHEXA   1       1       1       2       3       4       5       +11\n"
        "+11     6       7       8\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_signed_real_before_a_continuation_stays_data(tmp_path) -> None:
    """A '+1.5' ending a large-field line is a coordinate, not a marker.

    ``GRID*`` always carries a continuation, and its last line-1 field is Y,
    so treating a signed real there as a named marker drops Y and shifts Z
    into it.
    """
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID*   1                               +1.5            +2.5\n"
        "*       3.5\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).vertices, [[1.5, 2.5, 3.5]])


@pytest.mark.parametrize("y", ["+2.5", "+2.5E+00", "+2.5D+00", "+25.-1"])
def test_signed_real_spellings_survive_a_continuation(tmp_path, y) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        f"GRID*   1                               +1.5            {y:<16}\n"
        "*       3.5\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).vertices, [[1.5, 2.5, 3.5]])


def test_signed_real_before_a_small_field_continuation_stays_data(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID    1               0.0     0.0     +3.5\n"
        "+       0\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,1,1,1,2,3\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).vertices[0], [0.0, 0.0, 3.5])


@pytest.mark.parametrize("marker", ["+11", "*11"])
def test_numeric_named_marker_before_field_ten_is_a_marker(tmp_path, marker) -> None:
    """A signed real ends a card, but a bare integer still names one."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        + _eight_grids()
        + f"CHEXA,1,1,1,2,3,4,5,{marker}\n{marker},6,7,8\n"
        "ENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, np.arange(8))


def test_lazy_read_warns(tmp_path) -> None:
    tmp = _write(tmp_path, "BEGIN BULK\nGRID,1,,1.,2.,3.\nENDDATA\n")
    with pytest.warns(UserWarning, match="lazy=True ignored"):
        poly = read(tmp, lazy=True)
    np.testing.assert_array_equal(poly.vertices, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize("field", ["1_0", "1_0.5"])
def test_grouped_digits_are_not_valid_fields(tmp_path, field) -> None:
    """Nastran has no digit grouping; Python's int()/float() do."""
    tmp = _write(tmp_path, f"BEGIN BULK\nGRID,{field},,0.,0.,0.\nENDDATA\n")
    with pytest.raises(CodecError, match="GRID id"):
        read(tmp)
    tmp = _write(tmp_path, f"BEGIN BULK\nGRID,1,,{field},0.,0.\nENDDATA\n")
    with pytest.raises(CodecError, match="coordinate"):
        read(tmp)


def test_undefined_grid_error_names_the_deck_card(tmp_path) -> None:
    """The message points at the card as the deck spells it, not a rewrite."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\n"
        "GRID,2,,1.,0.,0.\n"
        "GRID,3,,0.,1.,0.\n"
        "CTRIA3,7,1,1,2,3\n"
        "CTRIA6,42,1,1,2,99,4,5,6\n"
        "ENDDATA\n",
    )
    with pytest.raises(CodecError, match="CTRIA6 42 references undefined GRID 99"):
        read(tmp)


def test_superelement_set_cards_do_not_warn(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\nCSET,1,123\nCSET1,123,1,2\nGRID,1,,0.,0.,0.\nENDDATA\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(read(tmp).vertices) == 1


def test_pid_attr_warning_points_at_the_caller(tmp_path) -> None:
    """stacklevel must reach the caller of write, not stay in the codec."""
    poly = _tri_mesh()
    poly.element_attrs["pid"] = np.array([1, 2, 3], dtype=np.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, tmp_path / "pid.bdf")
    assert [w for w in caught if w.filename == __file__]


def test_non_ascii_comment_does_not_break_the_read(tmp_path) -> None:
    tmp = tmp_path / "latin.bdf"
    tmp.write_bytes(
        "$ maillage g\xe9n\xe9r\xe9\nBEGIN BULK\nGRID,1,,0.,0.,0.\nENDDATA\n".encode(
            "latin-1"
        )
    )
    assert len(read(tmp).vertices) == 1


def test_registry_roundtrip(tmp_path) -> None:
    import polyxios

    poly = _tet_mesh()
    tmp = tmp_path / "registry.bdf"
    polyxios.write(poly, str(tmp))
    poly2 = polyxios.read(str(tmp))
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_large_field_star_padded_into_column_eight(tmp_path) -> None:
    """The large-field star may sit anywhere in the eight column name field."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID   *1                               0.0             0.0\n"
        "*       0.0\n"
        "GRID   *2                               1.0             0.0\n"
        "*       0.0\n"
        "GRID   *3                               0.0             1.0\n"
        "*       0.0\n"
        "CTRIA3 *1               1               1               2\n"
        "*       3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], atol=0)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_orphan_continuation_warns(tmp_path) -> None:
    """A continuation with no card ahead of it is dropped, but reported."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n+,7,8\nGRID,1,,0.,0.,0.\nENDDATA\n",
    )
    with pytest.warns(UserWarning, match="precede any card"):
        poly = read(tmp)
    assert len(poly.vertices) == 1


def test_include_path_holding_a_comma_still_warns(tmp_path) -> None:
    """A comma in the quoted path makes the line look free field."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\nINCLUDE 'part,1.bdf'\nGRID,1,,0.,0.,0.\nENDDATA\n",
    )
    with pytest.warns(UserWarning, match="INCLUDE"):
        poly = read(tmp)
    assert len(poly.vertices) == 1


def test_aero_cards_are_not_reported_as_elements(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "BEGIN BULK\nCAERO1,1,1,0,0,0,0,0\nGRID,1,,0.,0.,0.\nENDDATA\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(read(tmp).vertices) == 1


def test_non_finite_property_ids_fall_back_without_a_numpy_warning(tmp_path) -> None:
    """A NaN pid must warn and fall back, not raise numpy's cast warning."""
    poly = _tri_mesh()
    poly.element_attrs["pid"] = np.array([np.nan, np.nan])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, tmp_path / "nan.bdf")
    messages = [str(w.message) for w in caught]
    assert any("not integral" in m for m in messages)
    assert not any("invalid value encountered" in m for m in messages)
    assert "CTRIA3,1,1," in (tmp_path / "nan.bdf").read_text()


def test_offsets_length_mismatch_raises(tmp_path) -> None:
    poly = _tri_mesh()
    bad = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets[:-1],
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="offsets has length"):
        write(bad, tmp_path / "bad.bdf")


def test_offsets_past_the_end_of_connectivity_raises(tmp_path) -> None:
    """A truncated connectivity must not write a card short of a grid point."""
    poly = _tri_mesh()
    bad = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity[:-1],
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="offsets end at"):
        write(bad, tmp_path / "bad.bdf")


@pytest.mark.parametrize("shape", [(2, 2), (6,), (2, 3, 1)])
def test_write_rejects_vertices_that_are_not_three_wide(tmp_path, shape) -> None:
    poly = PolyData(
        vertices=np.zeros(shape, dtype=np.float64),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="expected \\(n_vertices, 3\\)"):
        write(poly, tmp_path / "bad.bdf")


def test_write_accepts_an_empty_vertex_array_of_any_shape(tmp_path) -> None:
    poly = PolyData(
        vertices=np.array([], dtype=np.float64),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
    )
    tmp = tmp_path / "empty.bdf"
    write(poly, tmp)
    assert read(tmp).vertices.shape == (0, 3)


def test_marker_past_column_seventy_two_leaves_field_nine_as_data(tmp_path) -> None:
    """A card reaching the continuation field cannot hide a marker before it.

    Field 9 spans columns 64 to 72, so '+9' there is grid point 9, and the
    '+11' past column 72 is the marker.
    """
    grids = "".join(f"GRID,{i + 1},,{float(i)},0.,0.\n" for i in range(12))
    card = "CHEXA   1       1       1       2       3       4       5       +9      +11"
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n" + grids + card + "\n+11     6       7       8\nENDDATA\n",
    )
    np.testing.assert_array_equal(read(tmp).connectivity, [0, 1, 2, 3, 4, 8, 5, 6])


def test_begin_bulk_tolerates_extra_blanks(tmp_path) -> None:
    tmp = _write(
        tmp_path,
        "SOL 101\nCEND\nBEGIN   BULK\nGRID,1,,1.,2.,3.\nENDDATA\n",
    )
    np.testing.assert_allclose(read(tmp).vertices, [[1.0, 2.0, 3.0]])


def test_written_exponents_are_uppercase(tmp_path) -> None:
    """Bulk data spells an exponent 'E'; Python's repr spells it 'e'."""
    verts = np.array([[1e-7, 1e20, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tmp = tmp_path / "exp.bdf"
    write(poly, tmp)

    text = tmp.read_text()
    assert "1.E-07" in text
    assert "e" not in text.partition("BEGIN BULK")[2]
    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_free_field_reals_keep_their_magnitude_when_truncated(tmp_path) -> None:
    """A solver cutting a field to 8 characters must not lose the exponent.

    ``1.2345678901234E-05`` truncates to ``1.234567``, five orders of
    magnitude out; the fixed-point spelling truncates to ``0.000012``.
    """
    verts = np.array([[1.2345678901234e-05, 2.0**-40, 0.0], [1.0, 0.0, 0.0], [0, 1, 0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tmp = tmp_path / "mag.bdf"
    with pytest.warns(UserWarning, match="more than 8 characters"):
        write(poly, tmp)

    fields = next(
        line for line in tmp.read_text().splitlines() if line.startswith("GRID,1,")
    ).split(",")[3:5]
    for text, value in zip(fields, verts[0, :2]):
        assert "E" not in text.upper(), text
        assert float(text[:8]) == pytest.approx(value, rel=0.1)

    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_write_warns_when_no_free_field_spelling_survives(tmp_path) -> None:
    """A huge exponent has no fixed-point spelling worth writing."""
    verts = np.array([[-9.87654321e300, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    with pytest.warns(UserWarning, match="need more than 8 characters"):
        with pytest.warns(UserWarning, match="reads a different magnitude"):
            write(poly, tmp_path / "huge.bdf")


def test_free_field_cards_stay_within_eighty_columns(tmp_path) -> None:
    """A bulk data record is 80 columns wide however long the reals are."""
    verts = np.array(
        [
            [1.2345678901234e-08, 9.8765432109876e-09, 1.1111111111111e-08],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tmp = tmp_path / "wide.bdf"
    with pytest.warns(UserWarning, match="more than 8 characters"):
        write(poly, tmp)

    lines = tmp.read_text().splitlines()
    assert all(len(line) <= 80 for line in lines)
    assert any(line.endswith(",+") for line in lines)
    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_property_id_tags_cover_every_element(tmp_path) -> None:
    """Ids come out ascending, each carrying its elements in deck order."""
    tmp = _write(
        tmp_path,
        "BEGIN BULK\n"
        "GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,0.,1.,0.\n"
        "CTRIA3,1,7,1,2,3\nCTRIA3,2,3,1,2,3\nCTRIA3,3,7,1,2,3\nCTRIA3,4,3,1,2,3\n"
        "ENDDATA\n",
    )
    poly = read(tmp)
    assert list(poly.element_tags) == ["pid_3", "pid_7"]
    np.testing.assert_array_equal(poly.element_tags["pid_3"], [1, 3])
    np.testing.assert_array_equal(poly.element_tags["pid_7"], [0, 2])
    assert poly.element_tags["pid_3"].dtype == np.int32
    np.testing.assert_array_equal(poly.element_attrs["pid"], [7, 3, 7, 3])


# ---------------------------------------------------------------------------
# Content sniffing (resolves the shared '.dat' extension)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "head",
    [
        b"GRID,1,,0.,0.,0.\n",
        b"GRID    1               0.      0.      0.\n",
        b"GRID*   1\n",
        b"$ a banner\n$ and more of it\nGRID,1,,0.,0.,0.\n",
        b"SOL 101\nCEND\n",
        b"NASTRAN SYSTEM(151)=1\n",
        b"BEGIN BULK\n",
    ],
)
def test_sniff_accepts_a_deck(head: bytes) -> None:
    assert sniff(head) is True


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"$ nothing but comments\n$ and no card at all\n",
        b'TITLE = "x"\nVARIABLES = "X" "Y" "Z"\nZONE N=3\n',
        # A table headed with the word GRID is not a GRID card: the delimiter
        # after the keyword is what tells them apart.
        b"GRID POINTS OF THE MODEL\n1 0.0 0.0 0.0\n",
        b"*KEYWORD\n",
    ],
)
def test_sniff_rejects_what_is_not_a_deck(head: bytes) -> None:
    assert sniff(head) is False
