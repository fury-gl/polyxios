from __future__ import annotations

import dataclasses
import io
from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata, read as api_read, write as api_write
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs._permas import read, sniff, write
from polyxios.exceptions import CodecError, UnsupportedFormatError

_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _tet_mesh():
    return make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])


def _mixed_mesh():
    """A tetra and two of its faces, each set naming an entity twice over."""
    return make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2], [0, 1, 3]])),
        ],
        vertex_tags={"top": np.array([3, 0], dtype=np.int32)},
        element_tags={
            "solid": np.array([0], dtype=np.int32),
            "skin": np.array([1, 2], dtype=np.int32),
        },
    )


def _write_deck(tmp_path: Path, text: str, name: str = "m.dato") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


_MINIMAL = """\
$ENTER COMPONENT NAME = DFLT_COMP DOFTYPE = DISP MATH
$STRUCTURE
$COOR
    1 0.0 0.0 0.0
    2 1.0 0.0 0.0
    3 0.0 1.0 0.0
    4 0.0 0.0 1.0
$ELEMENT TYPE = TET4
    1 1 2 3 4
$END STRUCTURE
$EXIT COMPONENT
$FIN
"""


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    path = tmp_path / "t.dato"
    write(poly, path)
    back = read(path)
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)


def test_roundtrip_keeps_element_order_and_sets(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    path = tmp_path / "t.dato"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    # A set comes back in the order the tag held it, not sorted.
    np.testing.assert_array_equal(back.vertex_tags["top"], [3, 0])
    np.testing.assert_array_equal(back.element_tags["solid"], [0])
    np.testing.assert_array_equal(back.element_tags["skin"], [1, 2])


def test_the_deck_has_the_expected_records(tmp_path: Path) -> None:
    path = tmp_path / "t.dato"
    write(_mixed_mesh(), path)
    text = path.read_text()
    assert text.startswith("$ENTER COMPONENT NAME = DFLT_COMP")
    assert "$STRUCTURE\n$COOR\n" in text
    assert "$ELEMENT TYPE = TET4\n    1 1 2 3 4\n$ELEMENT TYPE = TRIA3\n" in text
    assert "$NSET NAME = top\n    4 1\n" in text
    assert "$ESET NAME = skin\n    2 3\n" in text
    assert text.endswith("$END STRUCTURE\n$EXIT COMPONENT\n$FIN\n")


def test_coordinates_survive_exactly(tmp_path: Path) -> None:
    verts = np.array([[0.1, 1e-7, -2.5e3], [1 / 3, 2 / 3, 1e-300]])
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    path = tmp_path / "t.dato"
    write(poly, path)
    np.testing.assert_array_equal(read(path).vertices, verts)


def test_a_fortran_double_exponent_is_read(tmp_path: Path) -> None:
    """A solver-written deck may spell a real as ``1.0D+00``."""
    text = _MINIMAL.replace("    2 1.0 0.0 0.0", "    2 1.0D+00 -2.5d-01 0.0")
    back = read(_write_deck(tmp_path, text))
    np.testing.assert_array_equal(back.vertices[1], [1.0, -0.25, 0.0])


def test_a_tag_is_written_in_its_own_order(tmp_path: Path) -> None:
    """A member spelled twice is written once, at its first spelling."""
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_tags={"t": np.array([2, 0, 2, 3, 0], dtype=np.int32)},
    )
    path = tmp_path / "t.dato"
    write(poly, path)
    assert "$NSET NAME = t\n    3 1 4\n" in path.read_text()
    np.testing.assert_array_equal(read(path).vertex_tags["t"], [2, 0, 3])


@pytest.mark.parametrize(
    "name",
    [
        "vertex",
        "line",
        "quadratic_edge",
        "triangle",
        "quadratic_triangle",
        "quad",
        "quadratic_quad",
        "biquadratic_quad",
        "tetra",
        "quadratic_tetra",
        "hexahedron",
        "quadratic_hexahedron",
        "triquadratic_hexahedron",
        "pyramid",
        "wedge",
        "quadratic_wedge",
    ],
)
def test_every_writable_type_round_trips(tmp_path: Path, name: str) -> None:
    from polyxios._element_types import NODES_PER_ELEMENT

    n = NODES_PER_ELEMENT[name]
    verts = np.arange(3 * n, dtype=np.float64).reshape(n, 3)
    poly = make_polydata(verts, [(name, np.arange(n).reshape(1, n))])
    path = tmp_path / "t.dato"
    write(poly, path)
    back = read(path)
    assert back.element_types[0] == ELEMENT_TYPES[name]
    np.testing.assert_array_equal(back.connectivity, np.arange(n))


def test_api_round_trip_through_both_extensions(tmp_path: Path) -> None:
    poly = _tet_mesh()
    for name in ("t.dato", "t.post"):
        path = tmp_path / name
        api_write(poly, path)
        np.testing.assert_array_equal(api_read(path).connectivity, poly.connectivity)


def test_api_round_trip_through_gzip(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    path = tmp_path / "t.dato.gz"
    api_write(poly, path)
    back = api_read(path)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    assert set(back.element_tags) == {"solid", "skin"}


def test_a_buffer_reads_and_writes(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    out = io.BytesIO()
    write(poly, out)
    back = read(io.BytesIO(out.getvalue()))
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


# ---------------------------------------------------------------------------
# Reading what a solver deck spells
# ---------------------------------------------------------------------------


def test_minimal_deck(tmp_path: Path) -> None:
    back = read(_write_deck(tmp_path, _MINIMAL))
    np.testing.assert_allclose(back.vertices, _TET_VERTS)
    assert back.element_types.tolist() == [ELEMENT_TYPES["tetra"]]
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2, 3])
    assert back.vertex_attrs == {}
    assert back.element_attrs == {}
    assert back.global_attrs == {}


def test_comments_blank_lines_and_free_spacing(tmp_path: Path) -> None:
    text = """\
! a banner
$ENTER COMPONENT NAME=DFLT_COMP

$STRUCTURE
$COOR   NSET=ALL   ! trailing comment
  1   0.0   0.0   0.0
  2   1.0   0.0   0.0    ! and another

  3   0.0   1.0   0.0
$ELEMENT   TYPE   =   TRIA3   ESET=SKIN
  7   1 2 3
$END STRUCTURE
"""
    back = read(_write_deck(tmp_path, text))
    assert back.vertices.shape == (3, 3)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(back.vertex_tags["ALL"], [0, 1, 2])
    np.testing.assert_array_equal(back.element_tags["SKIN"], [0])


def test_a_continuation_line_joins_the_record_before_it(tmp_path: Path) -> None:
    text = """\
$COOR
    1 0 0 0
    2 1 0 0
    3 1 1 0
    4 0 1 0
    5 0 0 1
    6 1 0 1
    7 1 1 1
    8 0 1 1
$ELEMENT TYPE = HEXE8
    1 1 2 3 4
&     5 6 7 8
$NSET NAME = ALL
    1 2 3 4
&   5 6 7 8
"""
    back = read(_write_deck(tmp_path, text))
    np.testing.assert_array_equal(back.connectivity, np.arange(8))
    np.testing.assert_array_equal(back.vertex_tags["ALL"], np.arange(8))


def test_a_continuation_of_a_keyword_record_carries_its_data(tmp_path: Path) -> None:
    """``&`` folds onto a ``$`` record too; numbers there are the record's data."""
    text = """\
$COOR
    1 0 0 0
    2 1 0 0
    3 0 1 0
$COOR NSET = B
&   4 2 2 2
$ELEMENT TYPE = TRIA3
&   1 1 2 3
$NSET NAME = A
&   1 2 3
$ENTER COMPONENT NAME = X DOFTYPE = DISP MATH
"""
    back = read(_write_deck(tmp_path, text))
    assert back.vertices.shape == (4, 3)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(back.vertex_tags["A"], [0, 1, 2])
    np.testing.assert_array_equal(back.vertex_tags["B"], [3])
    assert back.global_attrs == {"permas_component": "X"}


def test_a_continuation_of_a_bare_keyword_record_carries_its_data(
    tmp_path: Path,
) -> None:
    """With no parameter to end the keyword, the first number does."""
    text = """\
$COOR
&   1 0 0 0
    2 1 0 0
    3 0 1 0
$ELEMENT TYPE = TRIA3
    1 1 2 3
$ESET NAME = S
&   1
$NSET
&   1 3
"""
    with pytest.warns(UserWarning, match="no NAME"):
        back = read(_write_deck(tmp_path, text))
    assert back.vertices.shape == (3, 3)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(back.element_tags["S"], [0])
    np.testing.assert_array_equal(back.vertex_tags["nset_1"], [0, 2])


def test_a_flag_word_on_a_record_does_not_hide_it(tmp_path: Path) -> None:
    """A bare word on a record is a flag, not part of the keyword."""
    text = _MINIMAL.replace("$COOR\n", "$COOR CART\n")
    assert read(_write_deck(tmp_path, text)).vertices.shape == (4, 3)
    # A flag between the parameters and the data a & folded onto the record
    # is passed over too, and the data still lands in the block.
    text = _MINIMAL.replace(
        "$COOR\n    1 0.0 0.0 0.0\n", "$COOR NSET = A CART\n& 1 0.0 0.0 0.0\n"
    )
    back = read(_write_deck(tmp_path, text))
    assert back.vertices.shape == (4, 3)
    np.testing.assert_array_equal(back.vertex_tags["A"], [0, 1, 2, 3])


@pytest.mark.parametrize("system", ["CYL", "SPH", "cyl"])
def test_a_non_cartesian_coordinate_system_is_refused(
    tmp_path: Path, system: str
) -> None:
    """A CYL or SPH block holds radii and angles, not the x y z read here."""
    text = _MINIMAL.replace("$COOR\n", f"$COOR {system}\n")
    with pytest.raises(
        CodecError, match=rf"\$COOR on line \d+ is flagged {system.upper()}"
    ):
        read(_write_deck(tmp_path, text))
    # The flag is seen wherever it sits on the record, after a parameter too.
    text = _MINIMAL.replace("$COOR\n", f"$COOR NSET = A {system}\n")
    with pytest.raises(CodecError, match="only Cartesian"):
        read(_write_deck(tmp_path, text))


def test_a_continuation_with_nothing_before_it_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="line 1 continues a record"):
        read(_write_deck(tmp_path, "& 1 2 3\n$COOR\n"))


def test_free_numbering_is_renumbered_and_remembered(tmp_path: Path) -> None:
    text = """\
$COOR
    10 0 0 0
    30 1 0 0
    20 0 1 0
$ELEMENT TYPE = TRIA3
    500 10 30 20
"""
    back = read(_write_deck(tmp_path, text))
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(back.vertex_attrs["original_ids"], [10, 30, 20])
    np.testing.assert_array_equal(back.element_attrs["original_ids"], [500])


def test_remembered_ids_are_written_back(tmp_path: Path) -> None:
    text = """\
$COOR
    10 0 0 0
    30 1 0 0
    20 0 1 0
$ELEMENT TYPE = TRIA3
    500 10 30 20
$NSET NAME = EDGE
    30 20
"""
    poly = read(_write_deck(tmp_path, text))
    out = tmp_path / "out.dato"
    write(poly, out)
    written = out.read_text()
    assert "    500 10 30 20\n" in written
    assert "$NSET NAME = EDGE\n    30 20\n" in written
    np.testing.assert_array_equal(read(out).vertex_attrs["original_ids"], [10, 30, 20])


def test_dense_numbering_records_nothing(tmp_path: Path) -> None:
    back = read(_write_deck(tmp_path, _MINIMAL))
    assert "original_ids" not in back.vertex_attrs
    assert "original_ids" not in back.element_attrs


def test_several_classes_read_as_one_geometry(tmp_path: Path) -> None:
    text = """\
$COOR
    1 0 0 0
    2 1 0 0
    3 1 1 0
    4 0 1 0
$ELEMENT TYPE = SHELL4
    1 1 2 3 4
$ELEMENT TYPE = LOADA4
    2 1 2 3 4
$ELEMENT TYPE = quad4
    3 1 2 3 4
"""
    back = read(_write_deck(tmp_path, text))
    assert back.element_types.tolist() == [ELEMENT_TYPES["quad"]] * 3


def test_two_records_add_to_one_set(tmp_path: Path) -> None:
    text = """\
$COOR
    1 0 0 0
    2 1 0 0
    3 0 1 0
$ELEMENT TYPE = TRIA3 ESET = ALL
    1 1 2 3
$ELEMENT TYPE = TRIA3 ESET = ALL
    2 3 2 1
$NSET NAME = N
    1
$NSET NAME = N
    3 1
"""
    back = read(_write_deck(tmp_path, text))
    np.testing.assert_array_equal(back.element_tags["ALL"], [0, 1])
    # A repeated member names its entity once, in the file's order.
    np.testing.assert_array_equal(back.vertex_tags["N"], [0, 2])


def test_a_component_name_is_kept_unless_it_is_the_default(tmp_path: Path) -> None:
    text = "$ENTER COMPONENT NAME = WING DOFTYPE = DISP\n" + _MINIMAL.split("\n", 1)[1]
    back = read(_write_deck(tmp_path, text))
    assert back.global_attrs == {"permas_component": "WING"}
    out = tmp_path / "out.dato"
    write(back, out)
    assert out.read_text().startswith("$ENTER COMPONENT NAME = WING ")
    assert read(_write_deck(tmp_path, _MINIMAL)).global_attrs == {}


def test_records_this_codec_does_not_read_are_passed_over(tmp_path: Path) -> None:
    text = """\
$ENTER COMPONENT NAME = DFLT_COMP
$SITUATION NAME = STATIC
    LOAD = L1
$STRUCTURE
$COOR
    1 0 0 0
    2 1 0 0
    3 0 1 0
$ELEMENT TYPE = TRIA3
    1 1 2 3
$END STRUCTURE
$SYSTEM
$MATERIAL NAME = STEEL TYPE = ISO
    E = 210000.
$END MATERIAL
$EXIT COMPONENT
$FIN
$COOR
    9 9 9 9
"""
    back = read(_write_deck(tmp_path, text))
    # Nothing past $FIN is read, and no other record is.
    assert back.vertices.shape == (3, 3)


def test_a_second_component_is_not_read(tmp_path: Path) -> None:
    text = (
        _MINIMAL.replace("$FIN\n", "")
        + """\
$ENTER COMPONENT NAME = OTHER
$STRUCTURE
$COOR
    1 5 5 5
$END STRUCTURE
$EXIT COMPONENT
$FIN
"""
    )
    with pytest.warns(UserWarning, match="second \\$ENTER COMPONENT at line 12"):
        back = read(_write_deck(tmp_path, text))
    assert back.vertices.shape == (4, 3)


def test_an_unknown_element_class_is_skipped_with_a_warning(tmp_path: Path) -> None:
    text = """\
$COOR
    1 0 0 0
    2 1 0 0
    3 0 1 0
$ELEMENT TYPE = SPRING1
    1 1
    2 2
$ELEMENT TYPE = TRIA3
    3 1 2 3
$ESET NAME = ALL
    1 2 3
"""
    with (
        pytest.warns(UserWarning, match=r"skipped: SPRING1 \(2\)"),
        pytest.warns(UserWarning, match=r"element set\(s\) \['ALL'\] name 2 id\(s\)"),
    ):
        back = read(_write_deck(tmp_path, text))
    assert back.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(back.element_tags["ALL"], [0])


def test_an_empty_block_of_an_unknown_class_is_silent(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "$ELEMENT TYPE = TET4", "$ELEMENT TYPE = SPRING1\n$ELEMENT TYPE = TET4"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write_deck(tmp_path, text))
    assert back.element_types.tolist() == [ELEMENT_TYPES["tetra"]]


def test_a_set_naming_an_unknown_node_drops_that_member(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "$END STRUCTURE", "$NSET NAME = BAD\n    1 99\n$END STRUCTURE"
    )
    with pytest.warns(UserWarning, match=r"node set\(s\) \['BAD'\] name 1 id\(s\)"):
        back = read(_write_deck(tmp_path, text))
    np.testing.assert_array_equal(back.vertex_tags["BAD"], [0])


def test_a_set_with_no_name_is_named_and_warned_about(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "$END STRUCTURE", "$NSET\n    1 2\n$ESET\n    1\n$END STRUCTURE"
    )
    with pytest.warns(UserWarning, match="2 set\\(s\\) carry no NAME ="):
        back = read(_write_deck(tmp_path, text))
    assert set(back.vertex_tags) == {"nset_1"}
    assert set(back.element_tags) == {"eset_2"}


def test_extra_values_on_a_record_are_ignored_with_a_warning(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    1 0.0 0.0 0.0", "    1 0.0 0.0 0.0 7.0").replace(
        "    1 1 2 3 4", "    1 1 2 3 4 0"
    )
    with pytest.warns(UserWarning, match="2 line\\(s\\) carry more values"):
        back = read(_write_deck(tmp_path, text))
    np.testing.assert_allclose(back.vertices, _TET_VERTS)


def test_lazy_warns_and_loads_eagerly(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="lazy=True is not supported"):
        back = read(_write_deck(tmp_path, _MINIMAL), lazy=True)
    assert back.vertices.shape == (4, 3)


# ---------------------------------------------------------------------------
# What a reader refuses
# ---------------------------------------------------------------------------


def test_a_repeated_node_id_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    2 1.0 0.0 0.0", "    1 1.0 0.0 0.0")
    with pytest.raises(
        CodecError, match="node id 1 on line 5 repeats the one on line 4"
    ):
        read(_write_deck(tmp_path, text))


def test_a_repeated_element_id_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    1 1 2 3 4", "    1 1 2 3 4\n    1 4 3 2 1")
    with pytest.raises(CodecError, match="element id 1 on line 10 repeats"):
        read(_write_deck(tmp_path, text))


def test_a_node_the_deck_never_declares_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    1 1 2 3 4", "    1 1 2 3 9")
    with pytest.raises(CodecError, match="element 1 on line 9 references node 9"):
        read(_write_deck(tmp_path, text))


def test_too_few_nodes_for_the_class_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    1 1 2 3 4", "    1 1 2 3")
    with pytest.raises(
        CodecError, match="line 9 carries 3 node\\(s\\), expected 4 for TET4"
    ):
        read(_write_deck(tmp_path, text))


def test_a_node_without_three_coordinates_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    4 0.0 0.0 1.0", "    4 0.0 0.0")
    with pytest.raises(CodecError, match="node line 7 carries 2 coordinate\\(s\\)"):
        read(_write_deck(tmp_path, text))


def test_a_malformed_number_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("    4 0.0 0.0 1.0", "    4 0.0 zero 1.0")
    with pytest.raises(CodecError, match="malformed node line 7"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", "    one 1 2 3 4")
    with pytest.raises(CodecError, match="malformed element id 'one' on line 9"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", "    1 1 2 3.5 4")
    with pytest.raises(CodecError, match="malformed node reference '3.5' on line 9"):
        read(_write_deck(tmp_path, text))


def test_an_integer_is_only_digits(tmp_path: Path) -> None:
    """``int`` takes ``1_0`` and the digits of other scripts; a deck does not."""
    text = _MINIMAL.replace("    4 0.0 0.0 1.0", "    1_0 0.0 0.0 1.0")
    with pytest.raises(CodecError, match="malformed node id '1_0' on line 7"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", "    1 1 2 3 1_0")
    with pytest.raises(CodecError, match="malformed node reference '1_0' on line 9"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", "    1 1 2 3 ٤")
    with pytest.raises(CodecError, match="malformed node reference"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("$END STRUCTURE", "$NSET NAME = A\n    1_0\n$END STRUCTURE")
    with pytest.raises(CodecError, match="malformed nset member '1_0' on line 11"):
        read(_write_deck(tmp_path, text))
    # A sign is part of how an integer is spelled.
    text = _MINIMAL.replace("    1 1 2 3 4", "    +1 +1 +2 +3 +4")
    np.testing.assert_array_equal(
        read(_write_deck(tmp_path, text)).connectivity, [0, 1, 2, 3]
    )


def test_an_id_wider_than_64_bits_is_refused(tmp_path: Path) -> None:
    """Each id path answers with a CodecError naming the file, not an OverflowError."""
    wide = "99999999999999999999"
    text = _MINIMAL.replace("    4 0.0 0.0 1.0", f"    {wide} 0.0 0.0 1.0")
    with pytest.raises(CodecError, match=f"node id '{wide}' on line 7 is wider"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", f"    {wide} 1 2 3 4")
    with pytest.raises(CodecError, match=f"element id '{wide}' on line 9 is wider"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", f"    1 1 2 3 4\n    2 1 2 3 {wide}")
    with pytest.raises(
        CodecError, match=f"element 2 on line 10 references node {wide}, which is wider"
    ):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace(
        "$END STRUCTURE", f"$NSET NAME = A\n    1 {wide}\n$END STRUCTURE"
    )
    with pytest.raises(CodecError, match=f"set 'A' names {wide}, which is wider"):
        read(_write_deck(tmp_path, text))


def test_a_non_positive_id_is_refused(tmp_path: Path) -> None:
    """PERMAS numbers from one; a zero or negative id is a corrupt deck."""
    text = _MINIMAL.replace("    4 0.0 0.0 1.0", "    0 0.0 0.0 1.0")
    with pytest.raises(CodecError, match="node id 0 on line 7 is not positive"):
        read(_write_deck(tmp_path, text))
    text = _MINIMAL.replace("    1 1 2 3 4", "    -1 1 2 3 4")
    with pytest.raises(CodecError, match="element id -1 on line 9 is not positive"):
        read(_write_deck(tmp_path, text))


def test_a_file_with_no_record_is_refused(tmp_path: Path) -> None:
    for text in ("", "\n\n", "! a banner\n! and nothing else\n"):
        with pytest.raises(CodecError, match="no \\$ record found"):
            read(_write_deck(tmp_path, text))


def test_data_outside_every_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="line 1 carries data outside any"):
        read(_write_deck(tmp_path, "1 0 0 0\n$COOR\n"))


def test_an_element_record_without_a_type_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace("$ELEMENT TYPE = TET4", "$ELEMENT ESET = X")
    with pytest.raises(CodecError, match="line 8 names no TYPE"):
        read(_write_deck(tmp_path, text))


# ---------------------------------------------------------------------------
# What a writer does with what it cannot spell
# ---------------------------------------------------------------------------


def test_element_type_picks_the_class(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    path = tmp_path / "t.dato"
    write(poly, path, element_type={"triangle": "shell3"})
    text = path.read_text()
    assert "$ELEMENT TYPE = SHELL3\n" in text
    assert "$ELEMENT TYPE = TET4\n" in text
    np.testing.assert_array_equal(read(path).element_types, poly.element_types)


def test_element_type_refuses_a_class_of_another_geometry(tmp_path: Path) -> None:
    with pytest.raises(
        CodecError, match="writes 'tetra' as 'HEXE8', which is an? 8-node"
    ):
        write(_tet_mesh(), tmp_path / "t.dato", element_type={"tetra": "HEXE8"})


def test_element_type_refuses_a_geometry_it_does_not_write(tmp_path: Path) -> None:
    with pytest.raises(
        CodecError, match="names 'polygon', which this codec does not write"
    ):
        write(_tet_mesh(), tmp_path / "t.dato", element_type={"polygon": "X"})


def test_element_type_refuses_a_bad_mapping(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="takes a mapping"):
        write(_tet_mesh(), tmp_path / "t.dato", element_type="TET4")
    with pytest.raises(CodecError, match="not a bare class name"):
        write(_tet_mesh(), tmp_path / "t.dato", element_type={"tetra": "TET 4"})
    # An empty non-mapping is still not a mapping.
    with pytest.raises(CodecError, match="takes a mapping"):
        write(_tet_mesh(), tmp_path / "t.dato", element_type=[])


def test_element_type_naming_an_absent_geometry_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match=r"element_type= names \['quad'\]"):
        write(_tet_mesh(), tmp_path / "t.dato", element_type={"quad": "SHELL4"})


def test_an_unspellable_element_is_dropped_and_left_out_of_its_set(
    tmp_path: Path,
) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [("polygon", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[1, 4, 2]]))],
        element_tags={"both": np.array([0, 1], dtype=np.int32)},
    )
    path = tmp_path / "t.dato"
    with (
        pytest.warns(
            UserWarning, match=r"element type\(s\) \['polygon'\] have no PERMAS class"
        ),
        pytest.warns(
            UserWarning,
            match="1 tag member\\(s\\) name an element that was not written",
        ),
    ):
        write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(back.element_tags["both"], [0])
    # The surviving element is numbered from one, not around the gap.
    assert "original_ids" not in back.element_attrs


def test_a_wrong_node_count_is_dropped(tmp_path: Path) -> None:
    from polyxios._types import PolyData

    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 3, 7], dtype=np.int32),
        element_types=np.array(
            [ELEMENT_TYPES["tetra"], ELEMENT_TYPES["tetra"]], dtype=np.uint8
        ),
    )
    path = tmp_path / "t.dato"
    with pytest.warns(UserWarning, match="1 element\\(s\\) carry a node count"):
        write(poly, path)
    assert len(read(path).element_types) == 1


def test_a_class_for_a_geometry_dropped_as_malformed_is_not_unused(
    tmp_path: Path,
) -> None:
    """The mesh holds the geometry; its one element was dropped, not ignored."""
    from polyxios._types import PolyData

    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["tetra"]], dtype=np.uint8),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, tmp_path / "t.dato", element_type={"tetra": "TET4"})
    messages = [str(w.message) for w in caught]
    assert any("carry a node count" in m for m in messages)
    assert not any("element_type= names" in m for m in messages)


def test_a_vertex_outside_the_mesh_is_refused(tmp_path: Path) -> None:
    from polyxios._types import PolyData

    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 9], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["tetra"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="references vertex 0..9, outside 0..3"):
        write(poly, tmp_path / "t.dato")


def test_a_tag_member_outside_the_mesh_is_dropped(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_tags={"v": np.array([0, 40], dtype=np.int32)},
    )
    path = tmp_path / "t.dato"
    with pytest.warns(UserWarning, match="1 tag member\\(s\\) index an entity"):
        write(poly, path)
    np.testing.assert_array_equal(read(path).vertex_tags["v"], [0])


def test_an_unsafe_name_is_made_safe_and_unique(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_tags={
            "top face": np.array([0], dtype=np.int32),
            "top_face": np.array([1], dtype=np.int32),
        },
        global_attrs={"permas_component": "left wing"},
    )
    path = tmp_path / "t.dato"
    with pytest.warns(UserWarning, match="cannot carry whitespace"):
        write(poly, path)
    back = read(path)
    assert set(back.vertex_tags) == {"top_face", "top_face_2"}
    assert back.global_attrs["permas_component"] == "left_wing"


def test_attributes_have_no_record_and_are_dropped_silently(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"p": np.arange(4.0)},
        element_attrs={"q": np.array([1.0])},
        global_attrs={"g": 1},
    )
    path = tmp_path / "t.dato"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    back = read(path)
    assert back.vertex_attrs == {}
    assert back.element_attrs == {}
    assert back.global_attrs == {}


def test_unknown_write_options_warn(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tet_mesh(), tmp_path / "t.dato", binary=True)


def test_a_planar_mesh_is_lifted_with_a_zero_z(tmp_path: Path) -> None:
    """A $COOR line holds three coordinates, so two columns gain a zero."""
    poly = make_polydata(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    path = tmp_path / "t.dato"
    write(poly, path)
    assert "    2 1.0 0.0 0.0\n" in path.read_text()
    np.testing.assert_array_equal(read(path).vertices[:, 2], [0.0, 0.0, 0.0])


def test_a_non_finite_coordinate_is_written_with_a_warning(tmp_path: Path) -> None:
    """The deck is still written - a mesh is still a mesh - but the solver will balk."""
    poly = make_polydata(
        np.array([[0.0, np.nan, 0.0], [1.0, 0.0, np.inf], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    path = tmp_path / "t.dato"
    with pytest.warns(UserWarning, match="2 node\\(s\\) have non-finite coordinates"):
        write(poly, path)
    assert "    1 0.0 nan 0.0\n" in path.read_text()


def test_vertices_of_another_width_are_refused(tmp_path: Path) -> None:
    poly = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    poly = dataclasses.replace(poly, vertices=np.zeros((4, 4)))
    with pytest.raises(CodecError, match=r"expected \(n, 3\)"):
        write(poly, tmp_path / "t.dato")
    # Holding no vertex does not excuse a width nothing could read back.
    poly = dataclasses.replace(poly, vertices=np.zeros((0, 5)))
    with pytest.raises(CodecError, match=r"expected \(n, 3\)"):
        write(poly, tmp_path / "t.dato")


def test_an_empty_mesh_round_trips(tmp_path: Path) -> None:
    from polyxios._types import PolyData

    poly = PolyData(
        vertices=np.zeros((0, 3)),
        connectivity=np.zeros(0, dtype=np.int32),
        offsets=np.zeros(1, dtype=np.int32),
        element_types=np.zeros(0, dtype=np.uint8),
    )
    path = tmp_path / "t.dato"
    write(poly, path)
    back = read(path)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0


# ---------------------------------------------------------------------------
# Sharing .dat
# ---------------------------------------------------------------------------


def test_sniff_recognises_a_deck_and_nothing_else() -> None:
    assert sniff(b"! banner\n\n$ENTER COMPONENT NAME = X\n")
    assert sniff(b"$STRUCTURE\n$COOR\n")
    assert sniff(b"$coor nset = all\n 1 0 0 0\n")
    # A deck may open with another section before its component.
    assert sniff(b"$ENTER MATERIAL\n$MATERIAL NAME = STEEL\n")
    assert sniff(b"$ENTER SYSTEM\n")
    # A byte-order mark does not hide the first record, as read() strips it.
    assert sniff(b"\xef\xbb\xbf$ENTER COMPONENT NAME = X\n")
    assert not sniff(b"$ a Nastran comment\nGRID,1,,0.,0.,0.\n")
    # A bulk data banner puts a space after its $; a keyword sits flush.
    assert not sniff(b"$ STRUCTURE\nGRID,1,,0.,0.,0.\n")
    assert not sniff(b"$ ELEMENT\nCQUAD4,1,1,1,2,3,4\n")
    assert not sniff(b"$\n$COOR\n")
    assert not sniff(b'TITLE = "tecplot"\n')
    assert not sniff(b"")
    assert not sniff(b"! only comments\n")


def test_sniff_does_not_claim_a_banner_spelled_flush(tmp_path: Path) -> None:
    """A Nastran comment can sit flush against its $ and open with a PERMAS word.

    Such a banner parses to a record carrying a word the format never puts
    there, or to one followed by a bulk data card where its data should be;
    either way it is not a deck, and the file has to reach the Nastran codec.
    """
    assert not sniff(b"$ELEMENT PROPERTIES\n$\nGRID,1,,0.,0.,0.\n")
    assert not sniff(b"$COOR SYSTEMS\nGRID,1,,0.,0.,0.\n")
    assert not sniff(b"$COOR\nGRID,1,,0.,0.,0.\n")
    assert not sniff(b"$STRUCTURE wing\nGRID,1,,0.,0.,0.\n")
    assert not sniff(b"$NSET\nGRID,1,,0.,0.,0.\n")
    # What a real record is followed by: its data, a continuation, or a record.
    assert sniff(b"$COOR CART\n    1 0 0 0\n")
    assert sniff(b"$COOR\n& 1 0 0 0\n")
    assert sniff(b"$ELEMENT TYPE = HEXE8\n$END STRUCTURE\n")
    assert sniff(b"$NSET NAME = A 1 2 3\n")
    assert sniff(b"$COOR\n")
    path = tmp_path / "wing.dat"
    path.write_text(
        "$ELEMENT PROPERTIES\n$\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n"
        "CROD,1,1,1,2\nENDDATA\n"
    )
    np.testing.assert_array_equal(api_read(path).connectivity, [0, 1])


def test_a_dat_deck_resolves_by_content(tmp_path: Path) -> None:
    path = _write_deck(tmp_path, _MINIMAL, name="model.dat")
    back = api_read(path)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2, 3])


def test_a_dat_deck_writes_only_with_an_explicit_format(tmp_path: Path) -> None:
    path = tmp_path / "model.dat"
    with pytest.raises(UnsupportedFormatError, match="fmt="):
        api_write(_tet_mesh(), path)
    api_write(_tet_mesh(), path, fmt="dato")
    assert path.read_text().startswith("$ENTER COMPONENT")
