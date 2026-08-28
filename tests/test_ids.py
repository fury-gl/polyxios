"""The original-id policy, held to across every format that numbers freely.

``polyxios/_ids.py`` states the rule: a mesh indexes densely from zero, a file
that numbered its entities otherwise has that numbering recorded in
``vertex_attrs["original_ids"]`` / ``element_attrs["original_ids"]``, and a
writer puts it back when it can still be spelled. The round-trip table below
is one sparsely-numbered fixture per such format, so a codec cannot drift off
the policy without a failing test naming it.
"""

from __future__ import annotations

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._ids import (
    IDS_KEY,
    ids_for_write,
    original_ids,
    record_ids,
    unwritable,
)

# ---------------------------------------------------------------------------
# The helpers themselves
# ---------------------------------------------------------------------------


def test_record_ids_keeps_quiet_about_a_file_numbered_from_one() -> None:
    """A writer renumbering from one reproduces 1..n exactly, so storing it
    would be a column of redundancy on the great majority of real files."""
    assert record_ids([1, 2, 3], count=3) == {}


def test_record_ids_remembers_a_numbering_the_index_does_not_say() -> None:
    out = record_ids([10, 20, 30], count=3)

    assert set(out) == {IDS_KEY}
    np.testing.assert_array_equal(out[IDS_KEY], [10, 20, 30])
    assert out[IDS_KEY].dtype == np.int64


def test_record_ids_remembers_one_from_one_out_of_order() -> None:
    """1..n permuted says something 1..n in order does not, so it is kept
    rather than thrown away as redundant."""
    assert IDS_KEY in record_ids([2, 1, 3], count=3)


@pytest.mark.parametrize(
    ("ids", "count"),
    [
        ([10, 20], 3),  # one short of the mesh
        ([], 0),  # nothing to number
        ([1.5, 2.5, 3.5], 3),  # not integers
        (np.zeros((3, 2), dtype=np.int64), 3),  # not one column
        ([10, 10, 30], 3),  # two entities answering to one number
        ([0, 10, 20], 3),  # a file numbers from one, so 0 is no id at all
    ],
)
def test_record_ids_refuses_what_no_writer_could_spell_back(ids, count: int) -> None:
    """The key's presence is a promise that the numbering survives a round
    trip, so a numbering that could not is not recorded in the first place."""
    assert record_ids(ids, count=count) == {}


def _numbered(ids: list[int] | None = None, **attrs) -> polyxios.PolyData:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if ids is not None:
        attrs = {**attrs, IDS_KEY: np.array(ids, dtype=np.int64)}
    return make_polydata(
        verts, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs=attrs
    )


def test_original_ids_answers_none_for_a_mesh_that_remembers_nothing() -> None:
    assert original_ids(_numbered(), kind="vertex") is None
    assert original_ids(_numbered([10, 20, 30]), kind="element") is None


def test_original_ids_refuses_a_kind_that_names_no_mapping() -> None:
    with pytest.raises(ValueError, match="vertex"):
        original_ids(_numbered(), kind="node")


def test_ids_for_write_numbers_from_one_when_nothing_is_remembered() -> None:
    out = ids_for_write(_numbered(), kind="vertex", count=3, fmt=".bdf")

    np.testing.assert_array_equal(out, [1, 2, 3])
    assert out.dtype == np.int64


def test_ids_for_write_spells_the_ids_the_file_gave() -> None:
    poly = _numbered([7000001, 7000002, 4001])

    out = ids_for_write(poly, kind="vertex", count=3, fmt=".bdf")

    np.testing.assert_array_equal(out, [7000001, 7000002, 4001])


@pytest.mark.parametrize(
    ("ids", "why"),
    [
        ([10, 10, 30], "unique"),
        ([0, 10, 20], "positive"),
        ([-1, 10, 20], "positive"),
        ([10, 20], "one per vertex"),
    ],
)
def test_ids_for_write_renumbers_rather_than_spell_a_file_no_solver_loads(
    ids, why: str
) -> None:
    """A merge collides two meshes numbered from one, a triangulation gives
    both halves of a quad the quad's id. Either would spell two entities
    answering to one number, so the numbering is what gives way - loudly."""
    poly = _numbered(ids)

    with pytest.warns(UserWarning, match=why):
        out = ids_for_write(poly, kind="vertex", count=3, fmt=".bdf")

    np.testing.assert_array_equal(out, [1, 2, 3])


def test_ids_for_write_renumbers_a_column_that_is_not_integers() -> None:
    poly = _numbered()
    poly.vertex_attrs[IDS_KEY] = np.array([1.5, 2.5, 3.5])

    with pytest.warns(UserWarning, match="integers"):
        out = ids_for_write(poly, kind="vertex", count=3, fmt=".bdf")

    np.testing.assert_array_equal(out, [1, 2, 3])


def test_ids_for_write_holds_an_empty_mesh_without_a_warning() -> None:
    poly = make_polydata(np.zeros((0, 3)), [])

    out = ids_for_write(poly, kind="element", count=0, fmt=".bdf")

    assert out.size == 0


@pytest.mark.parametrize(
    ("ids", "count", "why"),
    [
        (np.array([10, 20]), 3, "one per vertex"),
        (np.array([1.5]), 1, "integers"),
        (np.array([0, 1]), 2, "positive"),
        (np.array([5, 5]), 2, "unique"),
        (np.array([], dtype=np.int64), 0, None),
        (np.array([9, 4]), 2, None),
    ],
)
def test_unwritable_is_the_one_test_both_ends_of_the_policy_ask(
    ids, count: int, why
) -> None:
    """Read stores only what write would honour, so the two share a predicate
    rather than drifting apart as codecs are added."""
    got = unwritable(ids, count, "vertex")

    if why is None:
        assert got is None
    else:
        assert why in got


# ---------------------------------------------------------------------------
# Per-format round trips: one sparsely-numbered fixture per id-carrying format
# ---------------------------------------------------------------------------

_MSH_SPARSE = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
101 0.0 0.0 0.0
102 1.0 0.0 0.0
103 0.0 1.0 0.0
104 0.0 0.0 1.0
$EndNodes
$Elements
2
7001 2 2 1 1 101 102 103
7002 4 2 1 1 101 102 103 104
$EndElements
"""


def test_issue_1533_gmsh_keeps_the_numbers_the_file_gave(tmp_path) -> None:
    """meshio #1533 / #1531: a .msh numbering its nodes from 101 and its
    elements from 7001 came back numbered 1..n, so a load case naming node
    103 pointed at a different node after a round trip."""
    src = tmp_path / "sparse.msh"
    src.write_text(_MSH_SPARSE)

    poly = polyxios.read(src)

    np.testing.assert_array_equal(poly.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(poly.element_attrs[IDS_KEY], [7001, 7002])

    out = tmp_path / "out.msh"
    polyxios.write(poly, out)
    back = polyxios.read(out)

    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [7001, 7002])
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_gmsh_numbering_from_one_is_not_recorded(tmp_path) -> None:
    """The common case stays exactly as it was: nothing stored, nothing to
    slice through every transform, and the same file back out."""
    src = tmp_path / "dense.msh"
    src.write_text(
        _MSH_SPARSE.replace("101", "1")
        .replace("102", "2")
        .replace("103", "3")
        .replace("104", "4")
        .replace("7001", "1")
        .replace("7002", "2")
    )

    poly = polyxios.read(src)

    assert IDS_KEY not in poly.vertex_attrs
    assert IDS_KEY not in poly.element_attrs


def test_gmsh_data_rows_follow_the_ids_the_nodes_were_written_under(
    tmp_path,
) -> None:
    """A $NodeData row names a node by its tag. Writing the tags the deck gave
    while tagging the data rows 1..n would point every value at another node."""
    src = tmp_path / "sparse.msh"
    src.write_text(_MSH_SPARSE)
    poly = polyxios.read(src)
    poly.vertex_attrs["temperature"] = np.array([10.0, 20.0, 30.0, 40.0])

    out = tmp_path / "out.msh"
    polyxios.write(poly, out)

    text = out.read_text()
    body = text.split("$NodeData")[1]
    assert "101 10" in body
    np.testing.assert_array_equal(
        polyxios.read(out).vertex_attrs["temperature"], [10.0, 20.0, 30.0, 40.0]
    )
