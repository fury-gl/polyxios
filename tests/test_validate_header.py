from __future__ import annotations

import pytest

from polyxios.exceptions import ValidationError
from polyxios.validate import validate_header


def test_a_vertex_count_no_machine_can_hold_is_refused() -> None:
    """A corrupt header asking for 100 billion vertices wants 2.4 TB."""
    with pytest.raises(ValidationError, match="MAX_SAFE_VERTICES"):
        validate_header(10**11, 0, 0, 10**6)


def test_a_vertex_count_the_file_is_too_small_to_hold_is_refused() -> None:
    """10M verts need ~240 MB but file is only 1 KB."""
    with pytest.raises(ValidationError, match="file_size"):
        validate_header(10**7, 0, 0, 1000)


def test_reasonable_header_passes() -> None:
    # 100 verts, 50 tris, 150 conn indices - well within any file
    validate_header(100, 50, 150, 10_000)


def test_a_format_that_spells_no_points_is_capped_below_the_loose_one() -> None:
    """The byte heuristic is what made ``MAX_SAFE_VERTICES`` safe to leave loose.

    A file that describes its points rather than writing them is excused that
    heuristic - a 4x4x4 ImageData really does declare 64 points in 231 bytes -
    and on its own the loose cap let a header a few bytes long ask for twelve
    gigabytes of vertices.
    """
    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        validate_header(10**8 + 1, 0, 0, 250, spells_vertices=False)


def test_a_grid_a_machine_can_hold_still_passes_uncounted() -> None:
    """The cap has to clear the largest image anyone would actually expand."""
    validate_header(10**8, 0, 0, 250, spells_vertices=False)


def test_the_implied_cap_leaves_a_format_that_writes_its_points_alone() -> None:
    """A count that big is caught by the byte heuristic there, not by the cap."""
    with pytest.raises(ValidationError, match="file_size"):
        validate_header(10**8 + 1, 0, 0, 250)
