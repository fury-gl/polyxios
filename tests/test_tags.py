from __future__ import annotations

import numpy as np

from polyxios._tags import group_by_value, integer_column, values_from_tags


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
    values, unnamed, named, unusable = values_from_tags(
        {"ref_3": np.array([0.0, 1.0])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [0, 0])
    assert not named
    assert unnamed == set()
    assert unusable == {"ref_3"}


def test_a_group_named_anything_else_is_reported_apart() -> None:
    values, unnamed, named, unusable = values_from_tags(
        {"ref_3": np.array([1]), "top": np.array([0])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [0, 3])
    assert named
    assert unnamed == {"top"}
    assert unusable == set()


def test_a_member_outside_the_mesh_names_nothing() -> None:
    values, _unnamed, named, _unusable = values_from_tags(
        {"ref_5": np.array([0, 9, -1])}, "ref_", 2, dtype=np.int32
    )
    np.testing.assert_array_equal(values, [5, 0])
    assert named
