from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata, read as api_read, write as api_write
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._su2 import read, write
from polyxios.exceptions import CodecError

_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _tet_mesh():
    return make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    return make_polydata(
        _TET_VERTS,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))],
    )


def _tet_with_boundary():
    """One tetra plus its four faces, two of them tagged."""
    return make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])),
        ],
        element_tags={"wall": np.array([1, 2], dtype=np.int32)},
    )


def _write_su2(tmp_path: Path, text: str, name: str = "m.su2") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


_MINIMAL = """\
NDIME= 3
NELEM= 1
10 0 1 2 3 0
NPOIN= 4
0 0 0
1 0 0
0 1 0
0 0 1
NMARK= 0
"""


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    path = tmp_path / "t.su2"
    write(poly, path)
    back = read(path)
    assert len(back.element_types) == 1
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tri_mesh()
    path = tmp_path / "t.su2"
    write(poly, path)
    back = read(path)
    assert len(back.element_types) == 4
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_file_has_the_declared_sections(tmp_path: Path) -> None:
    path = tmp_path / "t.su2"
    write(_tri_mesh(), path)
    text = path.read_text()
    assert "NDIME=" in text
    assert "NELEM= 4" in text
    assert "NPOIN= 4" in text
    assert "NMARK= 0" in text


def test_a_flat_mesh_is_two_dimensional(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    path = tmp_path / "t.su2"
    write(poly, path)
    assert "NDIME= 2" in path.read_text()
    np.testing.assert_allclose(read(path).vertices, verts)


def test_a_volume_mesh_stays_three_dimensional(tmp_path: Path) -> None:
    """A tetra is 3-D even when every one of its z coordinates is zero."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "t.su2"
    write(poly, path)
    assert "NDIME= 3" in path.read_text()


def test_a_curved_surface_reports_that_su2_will_refuse_it(tmp_path: Path) -> None:
    """SU2 fills an NDIME= 3 domain with 3-D cells, and a triangle is not one."""
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="highest elements are 2-D"):
        write(_tri_mesh(), path)
    assert "NDIME= 3" in path.read_text()
    # The file is still whole: refusing to write it would lose the mesh.
    back = read(path)
    assert len(back.element_types) == 4
    np.testing.assert_allclose(back.vertices, _TET_VERTS)


def test_a_mesh_of_lines_reports_that_su2_will_refuse_it(tmp_path: Path) -> None:
    """NDIME is never 1, so a line mesh fills NELEM of a 2-D file."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64),
        [("line", np.array([[0, 1], [1, 2]]))],
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="highest elements are 1-D"):
        write(poly, path)
    assert "NDIME= 2" in path.read_text()
    assert len(read(path).element_types) == 2


def test_a_flat_surface_mesh_is_not_reported(tmp_path: Path) -> None:
    """A flat triangle mesh is exactly what an NDIME= 2 file holds."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "t.su2")


def test_boundary_elements_are_not_duplicated(tmp_path: Path) -> None:
    """The tagged faces travel as markers only, never in NELEM as well."""
    poly = _tet_with_boundary()
    path = tmp_path / "t.su2"
    write(poly, path)
    text = path.read_text()
    assert "NELEM= 1" in text
    back = read(path)
    assert len(back.element_types) == len(poly.element_types)


def test_boundary_marker_roundtrip(tmp_path: Path) -> None:
    poly = _tet_with_boundary()
    path = tmp_path / "t.su2"
    write(poly, path)
    back = read(path)
    assert set(back.element_tags) == {"wall", "unnamed"}
    assert len(back.element_tags["wall"]) == 2
    assert len(back.element_tags["unnamed"]) == 2
    # The volume cell leads, so the tagged faces follow it.
    tagged = back.element_tags["wall"]
    assert all(back.element_types[i] == ELEMENT_TYPES["triangle"] for i in tagged)


def test_marker_membership_survives_a_second_roundtrip(tmp_path: Path) -> None:
    """read -> write -> read is a fixed point, tags and element count alike."""
    poly = _tet_with_boundary()
    first = tmp_path / "a.su2"
    write(poly, first)
    once = read(first)
    second = tmp_path / "b.su2"
    write(once, second)
    twice = read(second)
    assert len(twice.element_types) == len(once.element_types)
    np.testing.assert_array_equal(twice.connectivity, once.connectivity)
    np.testing.assert_array_equal(twice.offsets, once.offsets)
    for name, members in once.element_tags.items():
        np.testing.assert_array_equal(twice.element_tags[name], members)


def test_untagged_boundary_elements_are_kept(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    path = tmp_path / "t.su2"
    write(poly, path)
    back = read(path)
    assert len(back.element_types) == 2
    assert list(back.element_tags) == ["unnamed"]


def test_a_tag_on_volume_elements_warns(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"fluid": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="no boundary element"):
        write(poly, path)
    assert "NELEM= 1" in path.read_text()
    assert not read(path).element_tags


def test_an_element_in_two_tags_is_written_once(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2], [0, 1, 3]])),
        ],
        element_tags={
            "wall": np.array([1, 2], dtype=np.int32),
            "inlet": np.array([1], dtype=np.int32),
        },
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="more than one tag"):
        write(poly, path)
    back = read(path)
    assert len(back.element_types) == 3
    assert len(back.element_tags["wall"]) == 2


def test_a_tag_named_once_is_marked_once(tmp_path: Path) -> None:
    """A repeated member would be written twice and read back as two elements."""
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={"wall": np.array([1, 1], dtype=np.int32)},
    )
    path = tmp_path / "t.su2"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    assert "MARKER_ELEMS= 1" in path.read_text()
    back = read(path)
    assert len(back.element_types) == 2
    np.testing.assert_array_equal(back.element_tags["wall"], [1])


def test_a_tag_over_both_dimensions_reports_what_it_drops(tmp_path: Path) -> None:
    """Only the boundary half of the tag can travel; the rest must be said."""
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={"mixed": np.array([0, 1], dtype=np.int32)},
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="not boundary elements"):
        write(poly, path)
    assert "MARKER_ELEMS= 1" in path.read_text()
    np.testing.assert_array_equal(read(path).element_tags["mixed"], [1])


def test_a_tag_naming_nothing_is_reported_as_empty(tmp_path: Path) -> None:
    """A tag with no members is a different fault from an unmarkable one."""
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={"hollow": np.array([], dtype=np.int32)},
    )
    with pytest.warns(UserWarning, match=r"\['hollow'\] name no element"):
        write(poly, tmp_path / "t.su2")


def test_an_empty_marker_block_reads_and_is_reported_on_write(
    tmp_path: Path,
) -> None:
    """MARKER_ELEMS= 0 is what leaves an empty tag behind."""
    text = _MINIMAL.replace("NMARK= 0", "NMARK= 1\nMARKER_TAG= hollow\nMARKER_ELEMS= 0")
    poly = read(_write_su2(tmp_path, text))
    assert poly.element_tags["hollow"].size == 0
    with pytest.warns(UserWarning, match="name no element"):
        write(poly, tmp_path / "out.su2")


def test_out_of_range_tag_members_are_reported_as_such(tmp_path: Path) -> None:
    """An index past the element array names no element, boundary or not."""
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={"wall": np.array([1, 99, -3], dtype=np.int32)},
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="outside the mesh's 2 element"):
        write(poly, path)
    assert "MARKER_ELEMS= 1" in path.read_text()
    np.testing.assert_array_equal(read(path).element_tags["wall"], [1])


@pytest.mark.parametrize(
    ("raw", "written"), [("far field", "far_field"), ("wall%1", "wall_1")]
)
def test_a_marker_name_survives_its_own_line(
    tmp_path: Path, raw: str, written: str
) -> None:
    """Whitespace splits the name and '%' comments out its tail."""
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={raw: np.array([1], dtype=np.int32)},
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="marker name"):
        write(poly, path)
    assert f"MARKER_TAG= {written}" in path.read_text()
    assert written in read(path).element_tags


def test_unsupported_element_types_are_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.random.default_rng(0).random((6, 3)),
        [("polygon", np.array([[0, 1, 2, 3, 4, 5]]))],
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="unsupported element type"):
        write(poly, path)
    assert "NELEM= 0" in path.read_text()
    assert len(read(path).element_types) == 0


def test_an_edge_of_a_volume_mesh_is_dropped(tmp_path: Path) -> None:
    """A marker holds the dimension below the volume cells and nothing lower.

    A line written under a marker of an ``NDIME= 3`` mesh is a file SU2 refuses
    to read back, so the edge goes rather than the file's validity.
    """
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("line", np.array([[0, 1]])),
        ],
    )
    path = tmp_path / "t.su2"
    with pytest.warns(UserWarning, match="more than one dimension below"):
        write(poly, path)
    text = path.read_text()
    assert "NELEM= 1" in text
    assert "\n3 0 1" not in text
    back = read(path)
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["tetra"],
        ELEMENT_TYPES["triangle"],
    ]


def test_a_boundary_edge_of_a_surface_mesh_is_kept(tmp_path: Path) -> None:
    """One dimension below a triangle is a line, so a 2-D mesh keeps its edges."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]])), ("line", np.array([[0, 1]]))],
    )
    path = tmp_path / "t.su2"
    write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["line"],
    ]
    assert list(back.element_tags) == ["unnamed"]


def test_write_rejects_an_out_of_range_vertex(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 9], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="references vertex 9"):
        write(poly, tmp_path / "t.su2")


def test_write_rejects_a_wrong_node_count(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="expected 3"):
        write(poly, tmp_path / "t.su2")


def test_write_warns_on_unknown_options(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tet_mesh(), tmp_path / "t.su2", nonesuch=1)


def test_read_of_a_hand_written_file(tmp_path: Path) -> None:
    path = _write_su2(tmp_path, _MINIMAL)
    poly = read(path)
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    text = "% a comment\n\nNDIME= 3  % inline\n" + _MINIMAL.split("NDIME= 3\n", 1)[1]
    poly = read(_write_su2(tmp_path, text))
    assert len(poly.element_types) == 1


def test_a_key_may_carry_a_space_before_its_equals(tmp_path: Path) -> None:
    poly = read(_write_su2(tmp_path, _MINIMAL.replace("NPOIN=", "NPOIN =")))
    assert poly.vertices.shape == (4, 3)


def test_lazy_read_warns(tmp_path: Path) -> None:
    path = _write_su2(tmp_path, _MINIMAL)
    with pytest.warns(UserWarning, match="lazy=True"):
        read(path, lazy=True)


def test_two_dimensional_points_pad_to_z_zero(tmp_path: Path) -> None:
    text = """\
NDIME= 2
NELEM= 1
5 0 1 2 0
NPOIN= 3
0 0
1 0
0 1
NMARK= 0
"""
    poly = read(_write_su2(tmp_path, text))
    np.testing.assert_allclose(poly.vertices[:, 2], 0)
    assert poly.vertices.shape == (3, 3)


def test_marker_elements_follow_the_volume_elements(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0",
        "NMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 2\n5 0 1 2\n5 0 1 3",
    )
    poly = read(_write_su2(tmp_path, text))
    assert len(poly.element_types) == 3
    np.testing.assert_array_equal(poly.element_tags["wall"], [1, 2])
    np.testing.assert_array_equal(poly.offsets, [0, 4, 7, 10])


def test_a_quoted_marker_name_loses_its_quotes(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0",
        'NMARK= 1\nMARKER_TAG= "wall"\nMARKER_ELEMS= 1\n5 0 1 2',
    )
    assert "wall" in read(_write_su2(tmp_path, text)).element_tags


def test_two_markers_of_one_name_merge(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0",
        "NMARK= 2\nMARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 2\n"
        "MARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 3",
    )
    with pytest.warns(UserWarning, match="more than once"):
        poly = read(_write_su2(tmp_path, text))
    np.testing.assert_array_equal(poly.element_tags["wall"], [1, 2])


def test_an_empty_marker_name_gets_one(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0", "NMARK= 1\nMARKER_TAG=\nMARKER_ELEMS= 1\n5 0 1 2"
    )
    with pytest.warns(UserWarning, match="empty MARKER_TAG"):
        poly = read(_write_su2(tmp_path, text))
    assert "marker_1" in poly.element_tags


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("NELEM= 0\nNPOIN= 0\nNMARK= 0\n", "missing NDIME"),
        ("NDIME= 4\nNELEM= 0\nNPOIN= 0\nNMARK= 0\n", "NDIME=4"),
        ("NDIME= x\nNELEM= 0\nNPOIN= 0\nNMARK= 0\n", "non-integer NDIME"),
        ("NDIME= 3\nNPOIN= 0\nNMARK= 0\n", "missing NELEM"),
        ("NDIME= 3\nNELEM= 0\nNMARK= 0\n", "missing NPOIN"),
        ("NDIME= 3\nNELEM= 0\nNPOIN= -5\nNMARK= 0\n", "negative node count"),
        ("NDIME= 3\nNELEM= 0\nNPOIN= 999999999999\nNMARK= 0\n", "safety cap"),
        ("NDIME= 3\nNELEM= 0\nNPOIN= zzz\nNMARK= 0\n", "non-integer node count"),
    ],
)
def test_read_rejects_a_broken_header(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(CodecError, match=match):
        read(_write_su2(tmp_path, text))


def test_read_rejects_an_overdeclared_element_count(tmp_path: Path) -> None:
    text = _MINIMAL.replace("NELEM= 1", "NELEM= 2")
    with pytest.raises(CodecError, match="only 1 are listed"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_a_truncated_file(tmp_path: Path) -> None:
    text = "NDIME= 3\nNELEM= 2\n10 0 1 2 3\nNPOIN= 4\n"
    with pytest.raises(CodecError, match="truncated|only 1 are listed"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_a_short_element_line(tmp_path: Path) -> None:
    text = _MINIMAL.replace("10 0 1 2 3 0", "10 0 1")
    with pytest.raises(CodecError, match="carries 2 node"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_a_short_node_line(tmp_path: Path) -> None:
    text = _MINIMAL.replace("1 0 0\n", "1 0\n", 1)
    with pytest.raises(CodecError, match="node line"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_a_malformed_node_line(tmp_path: Path) -> None:
    text = _MINIMAL.replace("1 0 0\n", "1 nan? 0\n", 1)
    with pytest.raises(CodecError, match="malformed node line"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_an_unsupported_element_type(tmp_path: Path) -> None:
    text = _MINIMAL.replace("10 0 1 2 3 0", "42 0 1 2 3 0")
    with pytest.raises(CodecError, match="unsupported element type 42"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_an_out_of_range_node_reference(tmp_path: Path) -> None:
    text = _MINIMAL.replace("10 0 1 2 3 0", "10 0 1 2 99 0")
    with pytest.raises(CodecError, match="outside 0..3"):
        read(_write_su2(tmp_path, text))


def test_read_rejects_a_marker_missing_its_element_count(tmp_path: Path) -> None:
    text = _MINIMAL.replace("NMARK= 0", "NMARK= 1\nMARKER_TAG= wall")
    with pytest.raises(CodecError, match="no MARKER_ELEMS"):
        read(_write_su2(tmp_path, text))


def test_a_marker_does_not_borrow_the_next_marker_count(tmp_path: Path) -> None:
    """An unbounded MARKER_ELEMS search reads the next marker's block."""
    text = _MINIMAL.replace(
        "NMARK= 0",
        "NMARK= 1\nMARKER_TAG= wall\nMARKER_TAG= inlet\nMARKER_ELEMS= 1\n5 0 1 2",
    )
    with pytest.raises(CodecError, match="marker wall declares no MARKER_ELEMS"):
        read(_write_su2(tmp_path, text))


def test_markers_past_the_declared_count_are_reported(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0",
        "NMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 2\n"
        "MARKER_TAG= inlet\nMARKER_ELEMS= 1\n5 0 1 3",
    )
    with pytest.warns(UserWarning, match="the file holds more"):
        poly = read(_write_su2(tmp_path, text))
    assert list(poly.element_tags) == ["wall"]
    assert len(poly.element_types) == 2


def test_read_rejects_a_missing_marker(tmp_path: Path) -> None:
    text = _MINIMAL.replace("NMARK= 0", "NMARK= 2\nMARKER_TAG= wall\nMARKER_ELEMS= 0")
    with pytest.raises(CodecError, match="only 1 carry a MARKER_TAG"):
        read(_write_su2(tmp_path, text))


def test_lines_past_a_declared_count_are_reported(tmp_path: Path) -> None:
    text = _MINIMAL.replace("10 0 1 2 3 0\n", "10 0 1 2 3 0\n10 0 1 2 3 1\n")
    with pytest.warns(UserWarning, match="past the count"):
        poly = read(_write_su2(tmp_path, text))
    assert len(poly.element_types) == 1


def test_lines_past_a_marker_count_are_reported(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0",
        "NMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 2\n5 0 1 3",
    )
    with pytest.warns(UserWarning, match="past the count"):
        poly = read(_write_su2(tmp_path, text))
    assert len(poly.element_types) == 2


def test_an_overlong_element_line_is_reported(tmp_path: Path) -> None:
    """A type code that understates its element drops a node in silence.

    ``5 0 1 2 3`` is a quad spelled as a triangle: the reader has no width to
    check against, so the fourth node would go without a word.
    """
    text = _MINIMAL.replace("10 0 1 2 3 0", "5 0 1 2 3 7")
    with pytest.warns(UserWarning, match="more tokens than"):
        poly = read(_write_su2(tmp_path, text))
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_an_overlong_marker_element_line_is_reported(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "NMARK= 0",
        "NMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 2 3 7",
    )
    with pytest.warns(UserWarning, match="more tokens than"):
        read(_write_su2(tmp_path, text))


def test_an_overlong_node_line_is_reported(tmp_path: Path) -> None:
    """A third coordinate in an NDIME= 2 file is a z the reader has to drop."""
    text = """\
NDIME= 2
NELEM= 1
5 0 1 2 0
NPOIN= 3
0 0 4 0
1 0 4 1
0 1 4 2
NMARK= 0
"""
    with pytest.warns(UserWarning, match="more values than NDIME=2"):
        poly = read(_write_su2(tmp_path, text))
    np.testing.assert_allclose(poly.vertices[:, 2], 0)


def test_an_index_column_is_not_overlong(tmp_path: Path) -> None:
    """The one trailing token SU2 allows is an index, not a lost node."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read(_write_su2(tmp_path, _MINIMAL))


def test_a_multi_zone_file_is_reported(tmp_path: Path) -> None:
    """Every section repeats per zone and this reader matches the first."""
    text = f"NZONE= 2\nIZONE= 1\n{_MINIMAL}IZONE= 2\n{_MINIMAL}"
    with pytest.warns(UserWarning, match="only the first zone"):
        poly = read(_write_su2(tmp_path, text))
    assert len(poly.element_types) == 1
    assert poly.vertices.shape == (4, 3)


def test_a_zero_nmark_beside_markers_is_reported(tmp_path: Path) -> None:
    """NMARK= 0 next to MARKER_TAG blocks is the file contradicting itself."""
    text = _MINIMAL + "MARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 2\n"
    with pytest.warns(UserWarning, match="declares NMARK= 0"):
        poly = read(_write_su2(tmp_path, text))
    assert not poly.element_tags
    assert len(poly.element_types) == 1


def test_markers_without_nmark_are_reported(tmp_path: Path) -> None:
    text = _MINIMAL.replace("NMARK= 0", "MARKER_TAG= wall\nMARKER_ELEMS= 1\n5 0 1 2")
    with pytest.warns(UserWarning, match="no NMARK"):
        poly = read(_write_su2(tmp_path, text))
    assert not poly.element_tags


def test_an_explicit_zero_nmark_is_believed(tmp_path: Path) -> None:
    """NMARK= 0 says the file has no markers; that is not worth a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read(_write_su2(tmp_path, _MINIMAL))


def test_an_empty_mesh_roundtrips(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    path = tmp_path / "t.su2"
    write(poly, path)
    back = read(path)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0
    np.testing.assert_array_equal(back.offsets, [0])


def test_meshio_reads_what_this_codec_writes(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    path = tmp_path / "t.su2"
    write(_tet_with_boundary(), path)
    mesh = meshio.read(str(path), "su2")
    counts = {block.type: len(block.data) for block in mesh.cells}
    assert counts == {"tetra": 1, "triangle": 4}
    np.testing.assert_allclose(mesh.points, _TET_VERTS)


def test_a_su2_file_with_index_columns_reads(tmp_path: Path) -> None:
    """SU2 closes element and node lines with an optional index column."""
    text = """\
NDIME= 3
NELEM= 1
10 0 1 2 3 0
NPOIN= 4
0.0 0.0 0.0 0
1.0 0.0 0.0 1
0.0 1.0 0.0 2
0.0 0.0 1.0 3
NMARK= 1
MARKER_TAG= 3
MARKER_ELEMS= 1
5 0 1 2
"""
    poly = read(_write_su2(tmp_path, text))
    np.testing.assert_allclose(poly.vertices, _TET_VERTS)
    np.testing.assert_array_equal(poly.element_tags["3"], [1])


def test_the_registry_serves_su2(tmp_path: Path) -> None:
    poly = _tet_mesh()
    path = tmp_path / "t.su2"
    api_write(poly, path)
    np.testing.assert_array_equal(api_read(path).connectivity, poly.connectivity)
