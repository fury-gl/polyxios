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


def test_ids_for_write_reads_a_whole_float_column_as_the_numbering() -> None:
    """A PLY vertex property and a legacy VTK data array have no integer
    form, so polyxios's own carriers hand the ids back as whole float64s.
    That is the dtype changing under them, not the numbering going wrong."""
    poly = _numbered()
    poly.vertex_attrs[IDS_KEY] = np.array([101.0, 205.0, 7000001.0])

    out = ids_for_write(poly, kind="vertex", count=3, fmt=".bdf")

    np.testing.assert_array_equal(out, [101, 205, 7000001])
    assert out.dtype == np.int64


@pytest.mark.parametrize(
    "ids",
    [
        [1.5, 2.5, 3.5],  # fractions, and no numbering
        [np.nan, 2.0, 3.0],  # a hole, which is not an id either
        [np.inf, 2.0, 3.0],
        [2.0**53, 2.0, 3.0],  # past where float64 spells every whole number
    ],
)
def test_ids_for_write_renumbers_a_column_that_is_not_integers(ids) -> None:
    poly = _numbered()
    poly.vertex_attrs[IDS_KEY] = np.array(ids, dtype=np.float64)

    with pytest.warns(UserWarning, match="integers"):
        out = ids_for_write(poly, kind="vertex", count=3, fmt=".bdf")

    np.testing.assert_array_equal(out, [1, 2, 3])


def test_ids_for_write_renumbers_an_id_wider_than_the_format_spells() -> None:
    """A Gmsh node tag is a 64-bit integer by spec and a Nastran field holds
    31 bits, so a mesh can arrive carrying an id its next format has no room
    for. Writing it anyway would spell a file no reader gets back."""
    poly = _numbered([1, 2, 2**40])

    with pytest.warns(UserWarning, match="run past 2147483647"):
        out = ids_for_write(poly, kind="vertex", count=3, fmt=".bdf", limit=2**31 - 1)

    np.testing.assert_array_equal(out, [1, 2, 3])


def test_ids_for_write_refuses_a_default_that_is_not_one_id_per_entity() -> None:
    """A codec handing over the wrong column is a bug in the codec, not
    anything about the mesh, so it is refused rather than quietly numbering
    the file wrong."""
    with pytest.raises(ValueError, match="one id per vertex"):
        ids_for_write(
            _numbered(), kind="vertex", count=3, fmt=".bdf", default=np.array([1, 2])
        )


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


def test_gmsh_keeps_the_numbers_the_file_gave(tmp_path) -> None:
    """A .msh numbering its nodes from 101 and its elements from 7001 came
    back numbered 1..n, so a load case naming node 103 pointed at a different
    node after a round trip."""
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


_BDF_SPARSE = """$ hand-numbered deck
BEGIN BULK
GRID,101,,0.0,0.0,0.0
GRID,102,,1.0,0.0,0.0
GRID,103,,0.0,1.0,0.0
GRID,104,,0.0,0.0,1.0
CTRIA3,4001,1,101,102,103
CTETRA,4002,1,101,102,103,104
ENDDATA
"""


def test_nastran_keeps_the_grid_and_element_ids(tmp_path) -> None:
    """A deck numbering GRID from 101 and elements from 4001 came back
    numbered 1..n, so every load case and property card in the author's other
    files named the wrong entity."""
    src = tmp_path / "sparse.bdf"
    src.write_text(_BDF_SPARSE)

    poly = polyxios.read(src)

    np.testing.assert_array_equal(poly.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(poly.element_attrs[IDS_KEY], [4001, 4002])

    out = tmp_path / "out.bdf"
    polyxios.write(poly, out)

    text = out.read_text()
    assert "GRID,101" in text
    assert "CTRIA3,4001,1,101,102,103" in text

    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [4001, 4002])
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_nastran_large_field_carries_the_kept_grid_ids(tmp_path) -> None:
    src = tmp_path / "sparse.bdf"
    src.write_text(_BDF_SPARSE)
    poly = polyxios.read(src)

    out = tmp_path / "out.bdf"
    polyxios.write(poly, out, field_format="large")

    assert "GRID*   101" in out.read_text()
    np.testing.assert_array_equal(
        polyxios.read(out).vertex_attrs[IDS_KEY], [101, 102, 103, 104]
    )


def test_nastran_says_so_when_an_id_outruns_the_field_a_solver_reads(
    tmp_path,
) -> None:
    """Nastran keeps only the first eight characters of a free-field entry,
    and a truncated id points a card at another entity rather than at a
    rounded value. The id still goes out whole."""
    poly = _numbered([1, 2, 1234567890])

    out = tmp_path / "wide.bdf"
    with pytest.warns(UserWarning, match="grid id"):
        polyxios.write(poly, out)

    assert "GRID,1234567890" in out.read_text()


def test_nastran_numbering_from_one_is_not_recorded(tmp_path) -> None:
    src = tmp_path / "dense.bdf"
    src.write_text(
        _BDF_SPARSE.replace("101", "1")
        .replace("102", "2")
        .replace("103", "3")
        .replace("104", "4")
        .replace("4001", "1")
        .replace("4002", "2")
    )

    poly = polyxios.read(src)

    assert IDS_KEY not in poly.vertex_attrs
    assert IDS_KEY not in poly.element_attrs


_INP_SPARSE = """*Heading
** hand-numbered deck
*Node
101, 0.0, 0.0, 0.0
102, 1.0, 0.0, 0.0
103, 0.0, 1.0, 0.0
104, 0.0, 0.0, 1.0
*Element, type=C3D4
4001, 101, 102, 103, 104
*Nset, nset=fixed
101, 104
*Elset, elset=body
4001
"""


def test_abaqus_keeps_the_node_and_element_ids(tmp_path) -> None:
    """A deck numbering nodes from 101 came back numbered 1..n, so the *Nset
    the deck's own boundary condition names reached other nodes after a round
    trip."""
    src = tmp_path / "sparse.inp"
    src.write_text(_INP_SPARSE)

    poly = polyxios.read(src)

    np.testing.assert_array_equal(poly.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(poly.element_attrs[IDS_KEY], [4001])

    out = tmp_path / "out.inp"
    polyxios.write(poly, out)

    text = out.read_text()
    assert "101, 0, 0, 0" in text
    assert "4001, 101, 102, 103, 104" in text

    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [4001])
    np.testing.assert_array_equal(back.vertex_tags["fixed"], poly.vertex_tags["fixed"])


def test_abaqus_sets_name_the_ids_the_cards_were_written_under(tmp_path) -> None:
    """An *Nset lists ids, not indices. Writing node 101 and then a set that
    says 1 would name a node the deck no longer defines."""
    src = tmp_path / "sparse.inp"
    src.write_text(_INP_SPARSE)

    out = tmp_path / "out.inp"
    polyxios.write(polyxios.read(src), out)

    body = out.read_text()
    assert "*Nset, nset=fixed\n101, 104" in body
    assert "*Elset, elset=body\n4001" in body


def test_abaqus_numbering_from_one_is_not_recorded(tmp_path) -> None:
    src = tmp_path / "dense.inp"
    src.write_text(
        _INP_SPARSE.replace("101", "1")
        .replace("102", "2")
        .replace("103", "3")
        .replace("104", "4")
        .replace("4001", "1")
    )

    poly = polyxios.read(src)

    assert IDS_KEY not in poly.vertex_attrs
    assert IDS_KEY not in poly.element_attrs


_F3GRID_SPARSE = """* FLAC3D grid
* GRIDPOINTS
G 101  0.0  0.0  0.0
G 102  1.0  0.0  0.0
G 103  0.0  1.0  0.0
G 104  0.0  0.0  1.0
* ZONES
Z  T4  4001  101  102  103  104
* ZONE GROUPS
ZGROUP "rock" SLOT 1
     4001
"""


def test_flac3d_keeps_the_gridpoint_and_record_ids(tmp_path) -> None:
    src = tmp_path / "sparse.f3grid"
    src.write_text(_F3GRID_SPARSE)

    poly = polyxios.read(src)

    np.testing.assert_array_equal(poly.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(poly.element_attrs[IDS_KEY], [4001])

    out = tmp_path / "out.f3grid"
    polyxios.write(poly, out)

    text = out.read_text()
    assert "G 101" in text
    assert "Z  T4  4001  101  102  103  104" in text
    assert "     4001" in text

    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [4001])
    np.testing.assert_array_equal(back.element_tags["rock"], [0])


def test_flac3d_drops_a_numbering_its_two_id_spaces_spell_twice(tmp_path) -> None:
    """FLAC3D numbers zones and faces separately, so a grid holding both may
    spell one number twice. The polyxios writer gives the two one shared
    space, so such a column cannot be put back and is not kept at all."""
    src = tmp_path / "both.f3grid"
    src.write_text(
        _F3GRID_SPARSE.replace(
            "* ZONE GROUPS", "* FACES\nF  T3  4001  101  102  103\n* ZONE GROUPS"
        )
    )

    poly = polyxios.read(src)

    assert IDS_KEY not in poly.element_attrs
    np.testing.assert_array_equal(poly.vertex_attrs[IDS_KEY], [101, 102, 103, 104])


def test_flac3d_numbering_from_one_is_not_recorded(tmp_path) -> None:
    src = tmp_path / "dense.f3grid"
    src.write_text(
        _F3GRID_SPARSE.replace("101", "1")
        .replace("102", "2")
        .replace("103", "3")
        .replace("104", "4")
        .replace("4001", "1")
    )

    poly = polyxios.read(src)

    assert IDS_KEY not in poly.vertex_attrs
    assert IDS_KEY not in poly.element_attrs


# ---------------------------------------------------------------------------
# The numbering travels, and gives way when a transform invalidates it
# ---------------------------------------------------------------------------


def test_the_numbering_is_the_meshs_not_the_files(tmp_path) -> None:
    """The key is deliberately not format-prefixed: an id says something
    about the entity, so a mesh read from a deck and written as a .msh keeps
    the numbering its author gave it."""
    src = tmp_path / "sparse.bdf"
    src.write_text(_BDF_SPARSE)
    out = tmp_path / "out.msh"

    polyxios.write(polyxios.read(src), out)

    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [4001, 4002])


def test_a_format_that_numbers_densely_still_carries_the_ids(tmp_path) -> None:
    """Medit has no id to spell, so it ignores the key on write - but the
    mesh keeps carrying it, which is what lets a later .bdf get it back."""
    src = tmp_path / "sparse.bdf"
    src.write_text(_BDF_SPARSE)
    poly = polyxios.read(src)

    assert IDS_KEY in poly.vertex_attrs
    out = tmp_path / "out.mesh"
    polyxios.write(poly, out)

    assert IDS_KEY not in polyxios.read(out).vertex_attrs


@pytest.mark.parametrize("ext", [".vtu", ".ply", ".vtk"])
def test_a_carrier_that_holds_the_ids_hands_them_back_to_a_deck(
    tmp_path, ext: str
) -> None:
    """The VTK family and PLY have no id of their own but do carry the key,
    which is what lets a later .bdf get the numbering back. A PLY vertex
    property and a legacy VTK data array are float, so the column comes home
    in another dtype - the numbering all the same, and no warning about it."""
    # Surfaces only: PLY has no solid to spell, and a flattened tetra would
    # cost the write a warning that says nothing about the numbering.
    src = tmp_path / "sparse.bdf"
    src.write_text(
        _BDF_SPARSE.replace(
            "CTETRA,4002,1,101,102,103,104", "CTRIA3,4002,1,102,103,104"
        )
    )

    carried = tmp_path / f"carried{ext}"
    polyxios.write(polyxios.read(src), carried)
    out = tmp_path / "out.bdf"
    polyxios.write(polyxios.read(carried), out)

    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs[IDS_KEY], [101, 102, 103, 104])
    np.testing.assert_array_equal(back.element_attrs[IDS_KEY], [4001, 4002])


def test_nastran_refuses_an_id_its_own_reader_could_not_read_back(
    tmp_path,
) -> None:
    """A Gmsh node tag is a 64-bit integer by spec; a Nastran field holds 31
    bits, and this codec's reader refuses anything wider. Spelling the tag
    anyway would land a deck polyxios itself cannot read, so the numbering
    is what gives way."""
    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={IDS_KEY: np.array([1, 2, 2**40], dtype=np.int64)},
    )

    out = tmp_path / "wide.bdf"
    with pytest.warns(UserWarning, match="run past"):
        polyxios.write(poly, out)

    np.testing.assert_array_equal(polyxios.read(out).vertices.shape, (3, 3))


def test_welding_keeps_the_ids_of_the_vertices_that_survive(tmp_path) -> None:
    """A transform that drops entities slices the id column with everything
    else, so what is left still numbers the mesh it describes."""
    src = tmp_path / "sparse.bdf"
    src.write_text(_BDF_SPARSE)
    poly = polyxios.read(src)

    kept = polyxios.transforms.remove_orphan_vertices(poly)

    np.testing.assert_array_equal(kept.vertex_attrs[IDS_KEY], [101, 102, 103, 104])


def test_triangulating_a_quad_gives_up_the_numbering_loudly(tmp_path) -> None:
    """Both halves of a quad carry the quad's id, which is two elements
    answering to one number - a file no solver loads. The numbering is what
    gives way, and the write says so."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        [("quad", np.array([[0, 1, 2, 3]]))],
        element_attrs={IDS_KEY: np.array([4001])},
    )
    split = polyxios.transforms.triangulate(poly)

    np.testing.assert_array_equal(split.element_attrs[IDS_KEY], [4001, 4001])

    out = tmp_path / "out.msh"
    with pytest.warns(UserWarning, match="unique"):
        polyxios.write(split, out)

    np.testing.assert_array_equal(
        polyxios.read(out).element_attrs.get(IDS_KEY, np.empty(0)), []
    )
