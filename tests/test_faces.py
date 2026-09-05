"""The link a face keeps back to the element it is a side of."""

from __future__ import annotations

import numpy as np

from polyxios import make_polydata
from polyxios._faces import (
    FACE_INDEX_KEY,
    FACE_PARENT_KEY,
    is_parent_face,
    parent_face_columns,
    parent_face_mask,
    parent_faces,
)
import polyxios.transforms as transforms


def _hex_and_face(hex_nodes: list[int], face_nodes: list[int]):
    """A hexahedron and one quad, in that order."""
    return make_polydata(
        np.zeros((8, 3)),
        [
            ("hexahedron", np.array([hex_nodes], dtype=np.int32)),
            ("quad", np.array([face_nodes], dtype=np.int32)),
        ],
    )


def test_a_face_is_its_parents_face_whatever_order_it_holds_it_in() -> None:
    """A format is free to wind a face the other way; it is the same face."""
    poly = _hex_and_face([0, 1, 2, 3, 4, 5, 6, 7], [3, 2, 1, 0])

    assert is_parent_face(poly, 1, 0, 0)


def test_a_collapsed_face_is_not_the_face_it_collapsed_from() -> None:
    """Compared as two sets, a repeat is forgotten - so a face carrying one
    vertex twice answered for a face of three distinct ones, and was written
    back as a side the element does not have."""
    # Face 0 of this hexahedron is (0, 1, 2, 2): degenerate, the deck's own.
    poly = _hex_and_face([0, 1, 2, 2, 4, 5, 6, 7], [0, 1, 1, 2])

    assert not is_parent_face(poly, 1, 0, 0)
    # The face the parent really holds still answers.
    same = _hex_and_face([0, 1, 2, 2, 4, 5, 6, 7], [0, 1, 2, 2])
    assert is_parent_face(same, 1, 0, 0)


def test_a_face_of_no_parent_answers_nothing() -> None:
    poly = _hex_and_face([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3])

    assert not is_parent_face(poly, 1, -1, 0)
    assert not is_parent_face(poly, 1, 9, 0)
    assert not is_parent_face(poly, 1, 1, 0)
    assert not is_parent_face(poly, 1, 0, 99)


def test_the_columns_are_read_back_only_when_they_describe_this_mesh() -> None:
    poly = _hex_and_face([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3])
    assert parent_faces(poly) is None

    poly.element_attrs.update(parent_face_columns({1: (0, 0)}, 2))
    parents, locals_ = parent_faces(poly)
    np.testing.assert_array_equal(parents, [-1, 0])
    np.testing.assert_array_equal(locals_, [-1, 0])

    # A legacy .vtk cell array is a double whatever it was handed, so a
    # column of whole numbers counts however it is spelled.
    poly.element_attrs[FACE_PARENT_KEY] = np.array([-1.0, 0.0])
    parents, locals_ = parent_faces(poly)
    np.testing.assert_array_equal(parents, [-1, 0])
    assert parents.dtype == np.int64

    # One that would have to be rounded into an index does not.
    poly.element_attrs[FACE_PARENT_KEY] = np.array([-1.0, 0.5])
    assert parent_faces(poly) is None

    # A blank does, as the -1 that says "not a face": merge fills a mesh
    # carrying no such column with the blank its dtype spells, and for a
    # float one that is NaN. Refusing the column over it would drop every
    # surface the other mesh did carry.
    poly.element_attrs[FACE_PARENT_KEY] = np.array([-1.0, np.nan])
    parents, _ = parent_faces(poly)
    np.testing.assert_array_equal(parents, [-1, -1])

    poly.element_attrs[FACE_PARENT_KEY] = np.array([-1, 0], dtype=np.int32)
    poly.element_attrs[FACE_INDEX_KEY] = np.array([-1, 0, 0], dtype=np.int32)
    assert parent_faces(poly) is None


def test_a_mesh_carrying_no_face_is_not_worth_two_columns() -> None:
    assert parent_face_columns({}, 3) == {}


def test_merge_renumbers_the_parent_a_face_names() -> None:
    """A face_parent is an element index, and two meshes laid end to end each
    numbered from zero. merge shifts the tag groups for the same reason; a
    column left alone would name an element of whichever mesh went first."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 3]]))],
    )
    poly.element_attrs.update(parent_face_columns({1: (0, 0)}, 2))

    merged = transforms.merge(poly, poly)

    np.testing.assert_array_equal(merged.element_attrs[FACE_PARENT_KEY], [-1, 0, -1, 2])
    # The local face number is not an index into the mesh, so it is not moved.
    np.testing.assert_array_equal(merged.element_attrs[FACE_INDEX_KEY], [-1, 0, -1, 0])
    assert is_parent_face(merged, 3, 2, 0)


def test_merge_leaves_a_column_it_cannot_read_as_indices_alone() -> None:
    """The same thing parent_faces refuses to read: a column that is not one
    whole number per element names no element to shift."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={FACE_PARENT_KEY: np.array([["a"]], dtype=object)},
    )

    merged = transforms.merge(poly, poly)

    assert merged.element_attrs[FACE_PARENT_KEY].tolist() == [["a"], ["a"]]


def test_merge_shifts_a_column_held_as_bools() -> None:
    """parent_faces reads every whole-number column, bools among them, so
    merge has to shift the same ones: one it left alone would go on naming
    an element of whichever mesh went first, and the surface it belongs to
    would be written back as the *Elset it is not."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 3]]))],
        element_attrs={
            FACE_PARENT_KEY: np.array([False, False]),
            FACE_INDEX_KEY: np.array([0, 0], dtype=np.int32),
        },
    )

    merged = transforms.merge(poly, poly)

    np.testing.assert_array_equal(merged.element_attrs[FACE_PARENT_KEY], [0, 0, 2, 2])
    assert is_parent_face(merged, 3, 2, 0)


def test_merge_keeps_the_surfaces_of_a_mesh_whose_columns_are_doubles() -> None:
    """A mesh that went out through legacy .vtk comes back holding these
    columns as doubles, and merging it with one that carries none fills that
    one's rows with the blank a float dtype spells - NaN. Read as a refusal
    it would take every surface the first mesh did carry with it."""
    faced = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 3]]))],
        element_attrs={
            FACE_PARENT_KEY: np.array([-1.0, 0.0]),
            FACE_INDEX_KEY: np.array([-1.0, 0.0]),
        },
    )
    plain = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )

    for merged in (transforms.merge(faced, plain), transforms.merge(plain, faced)):
        columns = parent_faces(merged)
        assert columns is not None
        parents, locals_ = columns
        face = int(np.flatnonzero(parents >= 0)[0])
        assert is_parent_face(merged, face, int(parents[face]), int(locals_[face]))


def test_a_solid_is_not_a_face_it_shares_its_vertices_with() -> None:
    """A tetra's four nodes can be exactly a hexahedron's face, and the
    columns can be made to say so. A face this library built is the quad its
    ring spells, though, so the tetra is a solid that happens to sit there -
    and reading it back as a face would cost the mesh a volume element."""
    poly = make_polydata(
        np.zeros((8, 3)),
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)),
            ("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32)),
        ],
    )

    assert not is_parent_face(poly, 1, 0, 0)


def test_the_column_answer_is_the_one_claim_answer_for_every_claim() -> None:
    """The writer asks of whole columns and the tests ask one claim at a
    time; two spellings of the same comparison would be two chances to
    disagree about what a face is."""
    poly = _hex_and_face([0, 1, 2, 3, 4, 5, 6, 7], [3, 2, 1, 0])
    claims = [
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 9, 0),
        (0, 0, 0),
        (1, 0, 99),
        (1, 0, -1),
        (9, 0, 0),
    ]
    elements, parents, locals_ = (np.array(col) for col in zip(*claims))

    np.testing.assert_array_equal(
        parent_face_mask(poly, elements, parents, locals_),
        [is_parent_face(poly, *claim) for claim in claims],
    )
