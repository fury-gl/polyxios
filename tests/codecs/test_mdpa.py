from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._ids import IDS_KEY
from polyxios.codecs._mdpa import PROPERTY_KEY, read, write
from polyxios.exceptions import CodecError

_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _tet_mesh(**kwargs):
    return make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))], **kwargs)


def _tri_mesh(**kwargs):
    return make_polydata(
        _TET_VERTS, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))], **kwargs
    )


def _write(text: str, path: Path) -> Path:
    path.write_text(text)
    return path


def _write_mesh(poly, path: Path) -> Path:
    write(poly, path)
    return path


_TWO_TETS = """\
Begin Nodes
 1 0.0 0.0 0.0
 2 1.0 0.0 0.0
 3 0.0 1.0 0.0
 4 0.0 0.0 1.0
 5 1.0 1.0 1.0
End Nodes

Begin Elements Element3D4N
 1 0 1 2 3 4
 2 0 2 3 4 5
End Elements
"""


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    write(poly, tmp_path / "m.mdpa")
    back = read(tmp_path / "m.mdpa")

    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.element_types, poly.element_types)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    write(poly, tmp_path / "m.mdpa")
    back = read(tmp_path / "m.mdpa")

    assert len(back.element_types) == 2
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_element_class_name_written(tmp_path) -> None:
    write(_tet_mesh(), tmp_path / "m.mdpa")
    assert "Begin Elements Element3D4N" in (tmp_path / "m.mdpa").read_text()


def test_a_quad_and_a_tetra_do_not_share_a_class_name(tmp_path) -> None:
    """Both are four nodes: the class name is what tells them apart."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [("quad", np.array([[0, 1, 2, 3]])), ("tetra", np.array([[0, 1, 2, 4]]))],
    )
    write(poly, tmp_path / "m.mdpa")
    text = (tmp_path / "m.mdpa").read_text()
    assert "Element2D4N" in text and "Element3D4N" in text

    back = read(tmp_path / "m.mdpa")
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_a_hexahedron20_keeps_its_node_order(tmp_path) -> None:
    """Kratos lists the vertical mid-edge nodes where polyxios lists the top."""
    verts = np.arange(60, dtype=np.float64).reshape(20, 3)
    poly = make_polydata(
        verts, [("quadratic_hexahedron", np.arange(20).reshape(1, 20))]
    )
    write(poly, tmp_path / "m.mdpa")

    body = [
        line
        for line in (tmp_path / "m.mdpa").read_text().splitlines()
        if line.strip().startswith("1 0 ")
    ]
    spelled = [int(tok) for tok in body[0].split()[2:]]
    # Nodes 13..16 of the file are polyxios's 17..20, and the top face follows.
    assert spelled[12:16] == [17, 18, 19, 20]
    assert spelled[16:20] == [13, 14, 15, 16]

    back = read(tmp_path / "m.mdpa")
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_a_wedge15_keeps_its_node_order(tmp_path) -> None:
    verts = np.arange(45, dtype=np.float64).reshape(15, 3)
    poly = make_polydata(verts, [("quadratic_wedge", np.arange(15).reshape(1, 15))])
    write(poly, tmp_path / "m.mdpa")
    back = read(tmp_path / "m.mdpa")

    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


# ---------------------------------------------------------------------------
# Reading what the format spells
# ---------------------------------------------------------------------------


def test_comments_and_blank_lines_are_ignored(tmp_path) -> None:
    text = "// a header comment\n\n" + _TWO_TETS.replace(
        " 1 0 1 2 3 4", " 1 0 1 2 3 4 // the first cell"
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    assert len(back.element_types) == 2


def test_an_explicit_geometry_name_is_read(tmp_path) -> None:
    text = _TWO_TETS.replace("Element3D4N", "Tetrahedra3D4")
    back = read(_write(text, tmp_path / "m.mdpa"))

    assert set(back.element_types.tolist()) == {ELEMENT_TYPES["tetra"]}


def test_conditions_are_read_as_elements(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin Conditions Element2D3N
 1 0 1 2 3
End Conditions
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    assert back.element_types.tolist() == [
        ELEMENT_TYPES["tetra"],
        ELEMENT_TYPES["tetra"],
        ELEMENT_TYPES["triangle"],
    ]


def test_an_unknown_element_class_is_skipped_with_a_warning(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin Elements SomeCustomElement
 3 0 1 2 3 4
End Elements
"""
    )
    with pytest.warns(UserWarning, match="element class this codec does not know"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert len(back.element_types) == 2


def test_a_section_this_codec_does_not_read_is_reported(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin Tables 1
 0.0 1.0
End Tables
"""
    )
    with pytest.warns(UserWarning, match="section\\(s\\) Tables are not read"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_nodal_data_fills_the_nodes_it_does_not_name(tmp_path) -> None:
    """A variable is listed only where it was set; the rest stay at zero."""
    text = (
        _TWO_TETS
        + """
Begin NodalData TEMPERATURE
 1 0 3.5
 3 0 4.5
End NodalData
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_allclose(
        back.vertex_attrs["TEMPERATURE"], [3.5, 0.0, 4.5, 0.0, 0.0]
    )


def test_nodal_data_without_the_is_fixed_column_is_read(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin NodalData TEMPERATURE
 1 3.5
End NodalData
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_allclose(back.vertex_attrs["TEMPERATURE"][0], 3.5)


def test_a_vector_variable_is_read_as_a_column_of_three(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin NodalData DISPLACEMENT
 2 0 [3] (1.0,2.0,3.0)
End NodalData
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    assert back.vertex_attrs["DISPLACEMENT"].shape == (5, 3)
    np.testing.assert_allclose(back.vertex_attrs["DISPLACEMENT"][1], [1.0, 2.0, 3.0])


def test_a_vector_whose_length_disagrees_with_its_values_is_refused(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin NodalData DISPLACEMENT
 2 0 [3] (1.0,2.0)
End NodalData
"""
    )
    with pytest.raises(CodecError, match="declares a vector of 3"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_model_part_data_becomes_global_attrs(tmp_path) -> None:
    text = (
        """\
Begin ModelPartData
 AMBIENT_TEMPERATURE 250.0
 STEPS 4
 IS_RESTARTED true
 CASE inlet_study
End ModelPartData

"""
        + _TWO_TETS
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    assert back.global_attrs == {
        "AMBIENT_TEMPERATURE": 250.0,
        "STEPS": 4,
        "IS_RESTARTED": True,
        "CASE": "inlet_study",
    }


def test_a_sub_model_part_becomes_tags(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Inlet
 Begin SubModelPartNodes
  1
  3
 End SubModelPartNodes
 Begin SubModelPartElements
  2
 End SubModelPartElements
End SubModelPart
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Inlet"], [0, 2])
    np.testing.assert_array_equal(back.element_tags["Inlet"], [1])


def test_a_nested_sub_model_part_keeps_its_own_name(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Outer
 Begin SubModelPartNodes
  1
 End SubModelPartNodes
 Begin SubModelPart Inner
  Begin SubModelPartNodes
   2
  End SubModelPartNodes
 End SubModelPart
End SubModelPart
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Outer"], [0])
    np.testing.assert_array_equal(back.vertex_tags["Inner"], [1])


def test_a_sub_model_part_member_the_file_never_declares_is_dropped(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Inlet
 Begin SubModelPartNodes
  1
  99
 End SubModelPartNodes
End SubModelPart
"""
    )
    with pytest.warns(UserWarning, match="SubModelPart node member"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Inlet"], [0])


def test_lazy_is_refused_with_a_warning(tmp_path) -> None:
    with pytest.warns(UserWarning, match="lazy=True is not supported"):
        read(_write(_TWO_TETS, tmp_path / "m.mdpa"), lazy=True)


# ---------------------------------------------------------------------------
# Malformed files
# ---------------------------------------------------------------------------


def test_no_nodes_section_raises(tmp_path) -> None:
    text = "Begin ModelPartData\nEnd ModelPartData\n"
    with pytest.raises(CodecError, match="no 'Begin Nodes' section"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_an_unclosed_block_is_refused(tmp_path) -> None:
    text = "Begin Nodes\n 1 0.0 0.0 0.0\n"
    with pytest.raises(CodecError, match="is never closed"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_a_block_closed_under_another_name_is_refused(tmp_path) -> None:
    text = "Begin Nodes\n 1 0.0 0.0 0.0\nEnd Elements\n"
    with pytest.raises(CodecError, match="closes 'Elements' inside a 'Nodes'"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_a_short_node_line_is_refused(tmp_path) -> None:
    text = "Begin Nodes\n 1 0.0 0.0\nEnd Nodes\n"
    with pytest.raises(CodecError, match="expected an id and three coordinates"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_a_repeated_node_id_is_refused(tmp_path) -> None:
    text = "Begin Nodes\n 1 0.0 0.0 0.0\n 1 1.0 0.0 0.0\nEnd Nodes\n"
    with pytest.raises(CodecError, match="node id 1 is declared twice"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_a_short_element_line_is_refused(tmp_path) -> None:
    text = _TWO_TETS.replace(" 2 0 2 3 4 5", " 2 0 2 3 4")
    with pytest.raises(CodecError, match="expected 4 for Element3D4N"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_an_undeclared_node_reference_is_refused(tmp_path) -> None:
    text = _TWO_TETS.replace(" 2 0 2 3 4 5", " 2 0 2 3 4 99")
    with pytest.raises(CodecError, match="references node 99"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_a_repeated_element_id_is_refused(tmp_path) -> None:
    text = _TWO_TETS.replace(" 2 0 2 3 4 5", " 1 0 2 3 4 5")
    with pytest.raises(CodecError, match="element id 1 is declared twice"):
        read(_write(text, tmp_path / "m.mdpa"))


# ---------------------------------------------------------------------------
# The numbering the file gave
# ---------------------------------------------------------------------------


def test_the_ids_a_file_spells_survive_a_round_trip(tmp_path) -> None:
    text = """\
Begin Nodes
 10 0.0 0.0 0.0
 20 1.0 0.0 0.0
 30 0.0 1.0 0.0
 40 0.0 0.0 1.0
End Nodes

Begin Elements Element3D4N
 7 0 10 20 30 40
End Elements
"""
    back = read(_write(text, tmp_path / "m.mdpa"))
    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [10, 20, 30, 40])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [7])

    write(back, tmp_path / "again.mdpa")
    again = (tmp_path / "again.mdpa").read_text()
    assert " 10 0.0 0.0 0.0" in again
    assert " 7 0 10 20 30 40" in again


def test_numbering_from_one_records_nothing(tmp_path) -> None:
    back = read(_write(_TWO_TETS, tmp_path / "m.mdpa"))

    assert IDS_KEY not in back.vertex_attrs
    assert IDS_KEY not in back.element_attrs


def test_a_condition_numbered_like_an_element_drops_the_numbering(tmp_path) -> None:
    """Two id spaces become one here, and a duplicate is unwritable."""
    text = (
        _TWO_TETS
        + """
Begin Conditions Element2D3N
 1 0 1 2 3
End Conditions
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    assert IDS_KEY not in back.element_attrs


def test_ids_a_transform_broke_are_replaced_with_a_warning(tmp_path) -> None:
    poly = _tri_mesh(
        element_attrs={"original_ids": np.array([5, 5], dtype=np.int64)},
    )
    with pytest.warns(UserWarning, match="they are not unique"):
        write(poly, tmp_path / "m.mdpa")

    assert " 1 0 " in (tmp_path / "m.mdpa").read_text()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_a_property_id_survives_a_round_trip(tmp_path) -> None:
    text = _TWO_TETS.replace(" 2 0 2 3 4 5", " 2 3 2 3 4 5")
    back = read(_write(text, tmp_path / "m.mdpa"))
    np.testing.assert_array_equal(back.element_attrs[PROPERTY_KEY], [0, 3])

    write(back, tmp_path / "again.mdpa")
    again = (tmp_path / "again.mdpa").read_text()
    assert "Begin Properties 3" in again
    assert " 2 3 2 3 4 5" in again


def test_one_property_records_nothing(tmp_path) -> None:
    back = read(_write(_TWO_TETS, tmp_path / "m.mdpa"))

    assert PROPERTY_KEY not in back.element_attrs


# ---------------------------------------------------------------------------
# What the writer cannot spell
# ---------------------------------------------------------------------------


def test_a_type_with_no_unambiguous_class_name_is_dropped(tmp_path) -> None:
    """A 3-node line would be written as a triangle's own class name."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("quadratic_edge", np.array([[0, 1, 2]]))])
    with pytest.warns(UserWarning, match="no unambiguous Kratos element class"):
        write(poly, tmp_path / "m.mdpa")

    assert "Begin Elements" not in (tmp_path / "m.mdpa").read_text()


def test_a_tensor_attribute_is_dropped_with_a_warning(tmp_path) -> None:
    poly = _tet_mesh(vertex_attrs={"stress": np.zeros((4, 3, 3))})
    with pytest.warns(UserWarning, match="not one number or one vector"):
        write(poly, tmp_path / "m.mdpa")

    assert "NodalData" not in (tmp_path / "m.mdpa").read_text()


def test_a_name_with_whitespace_is_written_safely(tmp_path) -> None:
    poly = _tet_mesh(element_tags={"inlet face": np.array([0], dtype=np.int32)})
    with pytest.warns(UserWarning, match="whitespace or a comment marker"):
        write(poly, tmp_path / "m.mdpa")

    text = (tmp_path / "m.mdpa").read_text()
    assert "Begin SubModelPart inlet_face" in text


def test_unrecognized_write_options_are_reported(tmp_path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tet_mesh(), tmp_path / "m.mdpa", nonsense=True)


# ---------------------------------------------------------------------------
# Class names an application registered
# ---------------------------------------------------------------------------


def test_an_application_element_class_is_read(tmp_path) -> None:
    """Kratos applications prefix their own name onto the generic suffix."""
    text = """\
Begin Nodes
 1 0.0 0.0 0.0
 2 1.0 0.0 0.0
 3 0.0 1.0 0.0
 4 0.0 0.0 1.0
End Nodes

Begin Elements SmallDisplacementElement3D4N
 1 0 1 2 3 4
End Elements

Begin Conditions SurfaceLoadCondition3D3N
 1 0 1 2 3
End Conditions
"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(
        back.element_types, [ELEMENT_TYPES["tetra"], ELEMENT_TYPES["triangle"]]
    )


def test_a_class_name_with_no_shape_in_it_is_still_unknown(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin Elements MyOwnElement
 9 0 1
End Elements
"""
    )
    with pytest.warns(UserWarning, match="element class this codec does not know"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert len(back.element_types) == 2


def test_a_point_condition_is_read_as_a_vertex(tmp_path) -> None:
    """A one-node class is a point, and Kratos boundary decks are full of them."""
    text = (
        _TWO_TETS
        + """
Begin Conditions PointLoadCondition3D1N
 1 0 1
 2 0 3
End Conditions
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.element_types[2:], [ELEMENT_TYPES["vertex"]] * 2)
    np.testing.assert_array_equal(back.connectivity[back.offsets[2] :], [0, 2])


def test_a_vertex_round_trips(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("vertex", np.array([[0], [1]]))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "m.mdpa")
        back = read(tmp_path / "m.mdpa")

    assert "Begin Elements Element3D1N" in (tmp_path / "m.mdpa").read_text()
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_a_geometry_named_point_is_read_as_a_vertex(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin Conditions Point3D
 1 0 2
End Conditions
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert back.element_types[-1] == ELEMENT_TYPES["vertex"]


# ---------------------------------------------------------------------------
# What a data block can carry
# ---------------------------------------------------------------------------


def test_a_boolean_column_is_written_as_numbers(tmp_path) -> None:
    """'true' is a ModelPartData word; a data block spells numbers."""
    poly = _tet_mesh(vertex_attrs={"FLAG": np.array([True, False, True, False])})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "m.mdpa")
        back = read(tmp_path / "m.mdpa")

    np.testing.assert_array_equal(back.vertex_attrs["FLAG"], [1, 0, 1, 0])


def test_a_one_component_vector_stays_a_column_of_one(tmp_path) -> None:
    poly = _tet_mesh(vertex_attrs={"P": np.arange(4.0).reshape(4, 1)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "m.mdpa")
        back = read(tmp_path / "m.mdpa")

    assert back.vertex_attrs["P"].shape == (4, 1)


def test_a_data_block_that_names_no_variable_is_reported(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin NodalData
 1 0 5.0
End NodalData
"""
    )
    with pytest.warns(UserWarning, match="name no variable"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_allclose(back.vertex_attrs["unnamed"], [5.0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Names, and the tables Kratos looks them up in
# ---------------------------------------------------------------------------


def test_one_name_in_four_tables_is_written_four_times(tmp_path) -> None:
    """A ModelPartData key, two variables and a part are four name spaces."""
    poly = _tet_mesh(
        vertex_attrs={"T": np.arange(4.0)},
        element_attrs={"T": np.array([9.0])},
        global_attrs={"T": 1.5},
        element_tags={"T": np.array([0], dtype=np.int32)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "m.mdpa")
        back = read(tmp_path / "m.mdpa")

    text = (tmp_path / "m.mdpa").read_text()
    assert "Begin NodalData T\n" in text
    assert "Begin ElementalData T\n" in text
    assert "Begin SubModelPart T\n" in text
    assert back.global_attrs == {"T": 1.5}
    np.testing.assert_allclose(back.vertex_attrs["T"], [0, 1, 2, 3])
    np.testing.assert_allclose(back.element_attrs["T"], [9.0])
    np.testing.assert_array_equal(back.element_tags["T"], [0])


def test_two_parts_of_one_name_do_not_pool_their_members(tmp_path) -> None:
    """Kratos asks only that siblings differ, so cousins may share a name."""
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Left
 Begin SubModelPart Skin
  Begin SubModelPartNodes
   1
  End SubModelPartNodes
 End SubModelPart
End SubModelPart

Begin SubModelPart Right
 Begin SubModelPart Skin
  Begin SubModelPartNodes
   2
  End SubModelPartNodes
 End SubModelPart
End SubModelPart
"""
    )
    with pytest.warns(UserWarning, match="used by more than one part"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Skin"], [0])
    np.testing.assert_array_equal(back.vertex_tags["Skin_2"], [1])


def test_two_parts_with_no_name_do_not_pool_their_members(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart
 Begin SubModelPartNodes
  1
 End SubModelPartNodes
End SubModelPart

Begin SubModelPart
 Begin SubModelPartNodes
  2
 End SubModelPartNodes
End SubModelPart
"""
    )
    with pytest.warns(UserWarning, match="used by more than one part"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["submodelpart"], [0])
    np.testing.assert_array_equal(back.vertex_tags["submodelpart_2"], [1])


# ---------------------------------------------------------------------------
# More of what the writer cannot spell
# ---------------------------------------------------------------------------


def test_a_global_value_that_cannot_hold_its_line_is_dropped(tmp_path) -> None:
    poly = _tet_mesh(global_attrs={"note": "a // b", "span": "x\ny", "ok": "plain"})
    with pytest.warns(UserWarning, match="line break or a comment marker"):
        write(poly, tmp_path / "m.mdpa")

    assert read(tmp_path / "m.mdpa").global_attrs == {"ok": "plain"}


def test_a_tag_member_the_mesh_does_not_hold_is_dropped(tmp_path) -> None:
    poly = _tet_mesh(vertex_tags={"g": np.array([0, 99], dtype=np.int32)})
    with pytest.warns(UserWarning, match="index an entity the mesh does not hold"):
        write(poly, tmp_path / "m.mdpa")

    np.testing.assert_array_equal(read(tmp_path / "m.mdpa").vertex_tags["g"], [0])


def test_a_property_column_a_transform_broke_is_reported(tmp_path) -> None:
    poly = _tet_mesh(element_attrs={PROPERTY_KEY: np.array([3.5])})
    with pytest.warns(UserWarning, match="not one whole number per element"):
        write(poly, tmp_path / "m.mdpa")

    assert "Begin Properties 0" in (tmp_path / "m.mdpa").read_text()


def test_a_global_named_after_a_section_word_is_dropped(tmp_path) -> None:
    poly = _tet_mesh(global_attrs={"End": "ModelPartData", "keep": 1})
    with pytest.warns(UserWarning, match="opens or closes a section"):
        write(poly, tmp_path / "m.mdpa")

    assert read(tmp_path / "m.mdpa").global_attrs == {"keep": 1}


def test_a_global_named_begin_does_not_swallow_the_file(tmp_path) -> None:
    poly = _tet_mesh(global_attrs={"begin": "Nodes", "keep": 2})
    with pytest.warns(UserWarning, match="opens or closes a section"):
        write(poly, tmp_path / "m.mdpa")

    back = read(tmp_path / "m.mdpa")
    assert back.global_attrs == {"keep": 2}
    assert back.vertices.shape == (4, 3)


def test_a_global_with_no_value_is_dropped(tmp_path) -> None:
    poly = _tet_mesh(global_attrs={"blank": "", "keep": 3})
    with pytest.warns(UserWarning, match="no ModelPartData entry spells"):
        write(poly, tmp_path / "m.mdpa")

    assert read(tmp_path / "m.mdpa").global_attrs == {"keep": 3}


def test_a_column_of_no_components_is_dropped(tmp_path) -> None:
    poly = _tet_mesh(vertex_attrs={"empty": np.zeros((4, 0))})
    with pytest.warns(UserWarning, match="not one number or one vector"):
        write(poly, tmp_path / "m.mdpa")

    assert "empty" not in read(tmp_path / "m.mdpa").vertex_attrs


def test_a_data_row_the_mesh_does_not_hold_is_reported(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin NodalData TEMPERATURE
 1 0 1.5
 99 0 7.5
End NodalData

Begin ElementalData DENSITY
 1 2.5
 98 9.5
End ElementalData
"""
    )
    path = _write(text, tmp_path / "m.mdpa")
    with pytest.warns(UserWarning, match="NodalData row"):
        with pytest.warns(UserWarning, match="ElementalData row"):
            back = read(path)

    np.testing.assert_array_equal(back.vertex_attrs["TEMPERATURE"], [1.5, 0, 0, 0, 0])
    np.testing.assert_array_equal(back.element_attrs["DENSITY"], [2.5, 0])


def test_a_data_block_naming_nothing_the_mesh_holds_is_reported(tmp_path) -> None:
    text = _TWO_TETS + "\nBegin NodalData TEMPERATURE\n 99 0 7.5\nEnd NodalData\n"
    path = _write(text, tmp_path / "m.mdpa")
    with pytest.warns(UserWarning, match="1 NodalData row"):
        back = read(path)

    assert "TEMPERATURE" not in back.vertex_attrs


def test_parts_nested_past_the_cap_are_refused(tmp_path) -> None:
    depth = 200
    text = (
        _TWO_TETS
        + "".join(f"Begin SubModelPart L{i}\n" for i in range(depth))
        + "End SubModelPart\n" * depth
    )
    with pytest.raises(CodecError, match="nests more than"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_parts_nested_within_the_cap_are_read(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Outer
 Begin SubModelPart Middle
  Begin SubModelPart Inner
   Begin SubModelPartNodes
    2
   End SubModelPartNodes
  End SubModelPart
 End SubModelPart
End SubModelPart
"""
    )
    back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Inner"], [1])


def test_an_empty_nodes_section_says_so(tmp_path) -> None:
    text = "Begin Nodes\nEnd Nodes\n"
    with pytest.raises(CodecError, match="no 'Begin Nodes' section"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_an_element_block_with_no_class_name_is_reported(tmp_path) -> None:
    text = """\
Begin Nodes
 1 0.0 0.0 0.0
 2 1.0 0.0 0.0
 3 0.0 1.0 0.0
End Nodes

Begin Elements
 1 0 1 2 3
End Elements
"""
    with pytest.warns(UserWarning, match=r"named \(unnamed\)"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert len(back.element_types) == 0


def test_an_interleaved_mesh_keeps_its_columns_beside_its_cells(tmp_path) -> None:
    poly = make_polydata(
        np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
        ),
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[1, 2, 3, 4]])),
        ],
        element_attrs={"e": np.array([10.0, 20.0, 30.0])},
        element_tags={"g": np.array([1, 2], dtype=np.int32)},
    )
    back = read(_write_mesh(poly, tmp_path / "m.mdpa"))

    # Grouped by class on the way out, so the triangle comes last - and its
    # value and its tag membership follow it there.
    assert [ELEMENT_TYPES["tetra"]] * 2 + [ELEMENT_TYPES["triangle"]] == list(
        back.element_types
    )
    np.testing.assert_array_equal(back.element_attrs["e"], [10.0, 30.0, 20.0])
    np.testing.assert_array_equal(back.element_tags["g"], [1, 2])


# ---------------------------------------------------------------------------
# The two vector spellings, and the one this codec cannot hold
# ---------------------------------------------------------------------------


def test_a_vector_without_a_declared_length_is_read(tmp_path) -> None:
    """``array_1d<double,3>`` is spelled bare, and DISPLACEMENT is one."""
    text = (
        _TWO_TETS
        + """
Begin NodalData DISPLACEMENT
 1 1 (0.0,0.0,0.0)
 2 0 (1.0,2.0,3.0)
End NodalData
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert back.vertex_attrs["DISPLACEMENT"].shape == (5, 3)
    np.testing.assert_allclose(back.vertex_attrs["DISPLACEMENT"][1], [1.0, 2.0, 3.0])


def test_both_vector_spellings_agree_on_the_width(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin NodalData V
 1 0 [3] (1.0,2.0,3.0)
 2 0 (4.0,5.0,6.0)
End NodalData
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_allclose(
        back.vertex_attrs["V"][:2], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )


def test_a_declared_length_that_does_not_match_is_still_refused(tmp_path) -> None:
    text = _TWO_TETS + "\nBegin NodalData V\n 1 0 [3] (1.0,2.0)\nEnd NodalData\n"
    with pytest.raises(CodecError, match="declares a vector of 3 but carries 2"):
        read(_write(text, tmp_path / "m.mdpa"))


def test_a_matrix_block_is_skipped_rather_than_refused(tmp_path) -> None:
    """A file carrying more than a mesh can is not a malformed file."""
    text = (
        _TWO_TETS
        + """
Begin ElementalData LOCAL_AXES
 1 [3,3] ((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))
End ElementalData

Begin NodalData T
 1 0 1.5
End NodalData
"""
    )
    with pytest.warns(UserWarning, match="spell a matrix per entity"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert "LOCAL_AXES" not in back.element_attrs
    np.testing.assert_allclose(back.vertex_attrs["T"][0], 1.5)


def test_a_matrix_does_not_take_a_scalar_block_of_the_same_name_with_it(
    tmp_path,
) -> None:
    text = (
        _TWO_TETS
        + """
Begin ElementalData X
 1 2.5
End ElementalData

Begin ElementalData X
 1 [2,2] ((1.0,0.0),(0.0,1.0))
End ElementalData
"""
    )
    with pytest.warns(UserWarning, match="spell a matrix per entity"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_allclose(back.element_attrs["X"], [2.5, 0.0])


# ---------------------------------------------------------------------------
# Keys polyxios keeps for itself
# ---------------------------------------------------------------------------


def test_a_variable_named_after_a_reserved_key_is_moved_aside(tmp_path) -> None:
    """A field named original_ids must not come back posing as the numbering."""
    text = (
        _TWO_TETS
        + """
Begin NodalData original_ids
 1 0 7.5
 2 0 8.5
End NodalData
"""
    )
    with pytest.warns(UserWarning, match="already spoken for"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert IDS_KEY not in back.vertex_attrs
    np.testing.assert_allclose(back.vertex_attrs["original_ids_2"][:2], [7.5, 8.5])


def test_a_variable_named_after_the_property_key_is_moved_aside(tmp_path) -> None:
    text = (
        _TWO_TETS
        + f"""
Begin ElementalData {PROPERTY_KEY}
 1 4.5
End ElementalData
"""
    )
    with pytest.warns(UserWarning, match="already spoken for"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    assert PROPERTY_KEY not in back.element_attrs
    np.testing.assert_allclose(back.element_attrs[f"{PROPERTY_KEY}_2"], [4.5, 0.0])


# ---------------------------------------------------------------------------
# What claims a SubModelPart name
# ---------------------------------------------------------------------------


def test_an_empty_part_does_not_take_the_name_of_one_with_members(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Skin
 Begin SubModelPartNodes
 End SubModelPartNodes
End SubModelPart

Begin SubModelPart Skin
 Begin SubModelPartNodes
  1
  2
 End SubModelPartNodes
End SubModelPart
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Skin"], [0, 1])
    assert "Skin_2" not in back.vertex_tags


def test_a_part_that_only_groups_its_children_claims_no_name(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Loads
 Begin SubModelPart Loads_Surface
  Begin SubModelPartNodes
   1
  End SubModelPartNodes
 End SubModelPart
End SubModelPart

Begin SubModelPart Loads
 Begin SubModelPartNodes
  2
 End SubModelPartNodes
End SubModelPart
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Loads"], [1])
    np.testing.assert_array_equal(back.vertex_tags["Loads_Surface"], [0])


def test_a_part_whose_members_are_all_unknown_claims_no_name(tmp_path) -> None:
    text = (
        _TWO_TETS
        + """
Begin SubModelPart Skin
 Begin SubModelPartNodes
  99
 End SubModelPartNodes
End SubModelPart

Begin SubModelPart Skin
 Begin SubModelPartNodes
  1
 End SubModelPartNodes
End SubModelPart
"""
    )
    with pytest.warns(UserWarning, match="name an id the file never declares"):
        back = read(_write(text, tmp_path / "m.mdpa"))

    np.testing.assert_array_equal(back.vertex_tags["Skin"], [0])
    assert "Skin_2" not in back.vertex_tags


# ---------------------------------------------------------------------------
# More of what the writer cannot spell
# ---------------------------------------------------------------------------


def test_a_global_value_of_nothing_but_whitespace_is_dropped(tmp_path) -> None:
    """A blank value writes a line of one word, which is no entry at all."""
    poly = _tet_mesh(global_attrs={"blank": "   ", "keep": 4})
    with pytest.warns(UserWarning, match="no ModelPartData entry spells"):
        write(poly, tmp_path / "m.mdpa")

    assert read(tmp_path / "m.mdpa").global_attrs == {"keep": 4}


def test_a_tag_naming_a_dropped_element_is_reported(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("quadratic_edge", np.array([[0, 1, 2]]))],
        element_tags={"edge": np.array([0], dtype=np.int32)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, tmp_path / "m.mdpa")

    assert any("has no class name for" in str(w.message) for w in caught)
    text = (tmp_path / "m.mdpa").read_text()
    assert "Begin SubModelPart edge" in text
    assert "Begin SubModelPartElements\n    End" in text
