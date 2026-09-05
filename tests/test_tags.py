from __future__ import annotations

import numpy as np
import pytest

from polyxios._tags import (
    group_by_value,
    integer_column,
    mask_arrays,
    member_indices,
    member_values,
    members_array,
    values_from_tags,
)


def test_an_empty_column_names_no_groups() -> None:
    """A mesh holding no elements labels none, which is not an error."""
    assert group_by_value(np.zeros(0, dtype=np.int32), "ref_") == {}


def test_a_group_holds_its_members_ascending() -> None:
    values = np.array([7, 2, 7, 2, 9], dtype=np.int32)
    groups = group_by_value(values, "ref_")
    assert list(groups) == ["ref_2", "ref_7", "ref_9"]
    np.testing.assert_array_equal(groups["ref_2"], [1, 3])
    np.testing.assert_array_equal(groups["ref_7"], [0, 2])


def test_a_float_column_is_no_label() -> None:
    """Rounding a label relabels the elements it stands for."""
    assert integer_column(np.array([1.5, 2.5]), 2) is None
    np.testing.assert_array_equal(
        integer_column(np.array([1, 2], dtype=np.int32), 2), [1, 2]
    )


def test_a_group_whose_members_are_not_indices_is_reported() -> None:
    """Nothing checks a tag group's dtype on the way in, and a float indexes none."""
    values, unnamed, named, unusable, oversized, _contested = values_from_tags(
        {"ref_3": np.array([0.0, 1.0])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [0, 0])
    assert not named
    assert unnamed == set()
    assert unusable == {"ref_3"}
    assert oversized == set()


def test_a_group_named_anything_else_is_reported_apart() -> None:
    values, unnamed, named, unusable, oversized, _contested = values_from_tags(
        {"ref_3": np.array([1]), "top": np.array([0])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [0, 3])
    assert named
    assert unnamed == {"top"}
    assert unusable == set()
    assert oversized == set()


def test_a_member_outside_the_mesh_names_nothing() -> None:
    values, _unnamed, named, _unusable, _oversized, _contested = values_from_tags(
        {"ref_5": np.array([0, 9, -1])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [5, 0])
    assert named


def test_a_group_naming_a_label_the_column_cannot_hold_is_reported() -> None:
    """A group name is free text, so it may spell a number no field carries."""
    values, _unnamed, named, unusable, oversized, _contested = values_from_tags(
        {"ref_5000000000": np.array([0])}, "ref_", 1, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [0])
    assert not named
    assert unusable == set()
    assert oversized == {"ref_5000000000"}


def test_a_group_taking_an_element_another_already_labelled_is_reported() -> None:
    """A record carries one reference, so the loser is worth naming."""
    values, _unnamed, named, _unusable, _oversized, contested = values_from_tags(
        {"ref_1": np.array([0, 1]), "ref_2": np.array([1])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [1, 2])
    assert named
    assert contested == {"ref_2"}


def test_groups_that_do_not_overlap_are_not_reported() -> None:
    _values, _unnamed, _named, _unusable, _oversized, contested = values_from_tags(
        {"ref_1": np.array([0]), "ref_2": np.array([1])}, "ref_", 2, dtype=np.int32
    )
    assert contested == set()


def test_a_group_reaching_no_element_names_no_label() -> None:
    """Its members index a mesh it was not built for, so it labels nothing.

    Counting it as a label named would have the caller write a column of
    zeros over a mesh that no group actually labelled.
    """
    held = values_from_tags({"ref_5": np.array([9, 11])}, "ref_", 2, dtype=np.int32)
    np.testing.assert_array_equal(held.values, [0, 0])
    assert not held.named
    assert held.unusable == {"ref_5"}


def test_a_group_reaching_one_element_still_names_a_label() -> None:
    held = values_from_tags({"ref_5": np.array([1, 11])}, "ref_", 2, dtype=np.int32)
    np.testing.assert_array_equal(held.values, [0, 5])
    assert held.named
    assert held.unusable == set()


def test_the_result_names_its_fields() -> None:
    """Six fields, four of them sets of names, is more than a positional
    unpack at a call site can be read back against."""
    held = values_from_tags({"ref_1": np.array([0])}, "ref_", 1, dtype=np.int32)
    assert held.values.tolist() == [1]
    assert held.unnamed == set()
    assert held.named is True
    assert held.oversized == set()
    assert held.contested == set()


# ---------------------------------------------------------------------------
# A group of a shape numpy builds no array of
# ---------------------------------------------------------------------------


_MALFORMED = [
    pytest.param([[0, 1], [2]], id="ragged"),
    pytest.param({"a": 1}, id="mapping"),
    pytest.param(["a", "b"], id="names"),
]


@pytest.mark.parametrize("members", _MALFORMED)
def test_a_group_of_no_shape_names_nothing_rather_than_raising(members) -> None:
    """Nothing checks a tag group on the way in. A ragged list is one numpy
    builds no array of at all, and asking it to raised a bare ValueError out
    of the middle of a write, naming neither the group nor the file."""
    assert member_indices(members, 3).tolist() == []
    assert member_values(members).tolist() == []


def test_a_group_of_no_shape_is_reported_by_the_writer_that_drops_it() -> None:
    """The same loss every other malformed group is reported as: the column
    is short of the members the group named, and the writer says which."""
    with pytest.warns(UserWarning, match=r"tag group\(s\) \['bad'\]"):
        columns = mask_arrays({"bad": [[0, 1], [2]]}, 3, fmt=".vtu", kind="cell")

    np.testing.assert_array_equal(columns["polyxios_tag_bad"], [0, 0, 0])


def test_a_group_that_is_an_array_comes_back_as_one() -> None:
    np.testing.assert_array_equal(members_array([2, 0]), [2, 0])
    assert members_array([[0, 1], [2]]) is None


def test_member_values_keeps_the_order_and_the_members_out_of_range() -> None:
    """The unbounded twin of member_indices: a writer that counts what reached
    no entity has to see those members rather than have them filtered away."""
    np.testing.assert_array_equal(member_values([5, 0, -1]), [5, 0, -1])
    np.testing.assert_array_equal(member_indices([5, 0, -1], 3), [0])
