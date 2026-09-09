from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    NODES_PER_ELEMENT,
)
from polyxios._types import PolyData
from polyxios.codecs._obj import read, write
from polyxios.exceptions import CodecError, LazyReadError


def _synthetic_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_ascii() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == len(poly.element_types)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_vertex_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs={"normals": normals}
    )
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "normals" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["normals"], normals, atol=1e-6)


def test_element_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"material": np.array(["steel", "iron"], dtype=object)},
    )
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "material" in poly2.element_attrs


def test_unsupported_lazy() -> None:
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        f.write(b"# empty\n")
        tmp = f.name
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_an_element_in_two_groups_keeps_both() -> None:
    """Element 0 in both 'inlet' and 'wall' - both must survive roundtrip."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={
            "inlet": np.array([0], dtype=np.int32),
            "wall": np.array([0, 1], dtype=np.int32),
        },
    )
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "inlet" in poly2.element_tags
    assert "wall" in poly2.element_tags
    assert 0 in poly2.element_tags["inlet"]
    assert 0 in poly2.element_tags["wall"]


# ---------------------------------------------------------------------------
# Texture coordinates, normals and face indices
# ---------------------------------------------------------------------------


def test_more_texcoords_than_vertices_reads(tmp_path) -> None:
    """OBJ indexes uv per face corner, so n_vt need not equal n_v."""
    path = tmp_path / "uv.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0\nvt 0.5 1\n"
        "vt 0.25 0.25\nvt 0.75 0.25\nvt 0.5 0.5\n"
        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
    )

    poly = read(path)

    assert poly.vertices.shape == (4, 3)
    assert len(poly.element_types) == 2
    uv = poly.vertex_attrs["texcoords"]
    assert uv.shape == (4, 2)
    np.testing.assert_allclose(uv[0], [0.0, 0.0])
    np.testing.assert_allclose(uv[2], [1.0, 1.0])


def test_negative_face_indices_count_back_from_the_end(tmp_path) -> None:
    """A negative OBJ index is relative to what has been declared so far."""
    path = tmp_path / "rel.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")

    poly = read(path)

    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_uv_and_normals_survive_a_round_trip(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": normals, "texcoords": uv},
    )
    path = tmp_path / "uv.obj"

    write(poly, path)
    back = read(path)

    np.testing.assert_allclose(back.vertex_attrs["normals"], normals, atol=1e-8)
    np.testing.assert_allclose(back.vertex_attrs["texcoords"], uv, atol=1e-8)


def test_a_face_index_past_the_vertex_list_is_refused(tmp_path) -> None:
    """Out of range is a corrupt file, not a wrap-around into the last vertex."""
    path = tmp_path / "bad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")

    with pytest.raises(CodecError, match="vertex index"):
        read(path)


def test_a_vertex_with_two_uvs_warns_that_one_is_kept(tmp_path) -> None:
    """A seam needs a uv per corner; the flat layout holds one per vertex."""
    path = tmp_path / "seam.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 0.5\n"
        "f 1/1 2/2 3/3\nf 1/4 3/3 4/1\n"
    )

    with pytest.warns(UserWarning, match="texture coordinate"):
        read(path)


# ---------------------------------------------------------------------------
# Groups on write
# ---------------------------------------------------------------------------


def test_each_tag_group_is_written_as_a_g_directive(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"inlet": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "groups.obj"

    write(poly, path)
    text = path.read_text()

    assert "g inlet" in text
    # The second face is in no group, so it must not inherit 'inlet'.
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["inlet"], [0])


# ---------------------------------------------------------------------------
# Records that cannot be lined up with the vertices
# ---------------------------------------------------------------------------


def test_unmatched_normals_leave_the_attribute_out(tmp_path) -> None:
    """A dropped fold must leave no attribute, not a None where an array goes."""
    path = tmp_path / "loose.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 0 1\nf 1 2 3\n")

    with pytest.warns(UserWarning, match="normal"):
        poly = read(path)

    assert "normals" not in poly.vertex_attrs
    # The dropped fold used to land as None and take the writer down with it.
    write(poly, tmp_path / "again.obj")


def test_a_vertex_no_face_names_is_written_as_a_number(tmp_path) -> None:
    """'vt nan nan' is not a record another OBJ reader takes."""
    path = tmp_path / "loose.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 2 2 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/1 2/2 3/3\n"
    )
    poly = read(path)
    assert np.isnan(poly.vertex_attrs["texcoords"][3]).all()

    out = tmp_path / "again.obj"
    write(poly, out)
    text = out.read_text()

    assert "nan" not in text
    assert len([line for line in text.splitlines() if line.startswith("vt ")]) == 4


def test_a_short_texcoord_row_is_padded_not_indexed_past(tmp_path) -> None:
    """A one-column uv array is a u with no v, not an IndexError."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"texcoords": np.array([[0.25], [0.5], [0.75]])},
    )
    path = tmp_path / "u.obj"

    write(poly, path)

    assert "vt 0.25 0" in path.read_text()


def test_an_attribute_short_of_the_vertices_is_not_written(tmp_path) -> None:
    """Faces indexing records the file does not hold are unreadable."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"texcoords": np.array([[0.0, 0.0]])},
    )
    path = tmp_path / "short.obj"

    with pytest.warns(UserWarning, match="not one row per vertex"):
        write(poly, path)

    text = path.read_text()
    assert "vt " not in text
    # Without the check this was 'f 1/1 2/2 3/3' against a single vt record.
    assert "f 1 2 3" in text
    read(path)


def test_a_one_dimensional_attribute_is_one_value_per_vertex(tmp_path) -> None:
    """A flat array is a column of vertices, not a single wide row."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"texcoords": np.array([0.25, 0.5, 0.75])},
    )
    path = tmp_path / "flat.obj"

    write(poly, path)

    lines = [line for line in path.read_text().splitlines() if line.startswith("vt ")]
    assert lines == ["vt 0.25 0", "vt 0.5 0", "vt 0.75 0"]


@pytest.mark.parametrize(
    "record,match",
    [
        ("v 0 0", "needs at least 3"),
        ("vn 0 0", "needs at least 3"),
        ("vt", "needs at least 1"),
        ("v 0 0 x", "not a row of numbers"),
    ],
)
def test_a_short_or_unreadable_record_names_the_line(
    tmp_path, record: str, match: str
) -> None:
    """float() and list indexing answer these without naming file or line."""
    path = tmp_path / "short.obj"
    path.write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\n{record}\n")

    with pytest.raises(CodecError, match=match):
        read(path)


def test_a_volumetric_vt_keeps_the_two_components_a_surface_uses(tmp_path) -> None:
    """A vt may carry a depth; texcoords holds u and v."""
    path = tmp_path / "vt3.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0 0.5\nvt 1 0 0.5\nvt 0 1 0.5\nf 1/1 2/2 3/3\n"
    )

    poly = read(path)

    assert poly.vertex_attrs["texcoords"].shape == (3, 2)
    np.testing.assert_allclose(poly.vertex_attrs["texcoords"][1], [1, 0])


def test_a_vt_with_one_component_means_zero_for_the_other(tmp_path) -> None:
    """The format lets v go; a per-vertex array still needs both columns."""
    path = tmp_path / "vt1.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0\nvt 1\nvt 0.5\nf 1/1 2/2 3/3\n")

    poly = read(path)

    np.testing.assert_allclose(
        poly.vertex_attrs["texcoords"], [[0, 0], [1, 0], [0.5, 0]]
    )


def test_an_attribute_of_labels_is_dropped_rather_than_crashing(tmp_path) -> None:
    """A vn record has no way to spell a string, and asarray raises on one."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": np.array(["a", "b", "c"], dtype=object)},
    )
    path = tmp_path / "labels.obj"

    with pytest.warns(UserWarning, match="not numbers"):
        write(poly, path)

    text = path.read_text()
    assert "vn " not in text
    assert "f 1 2 3" in text


def test_a_conflict_is_seen_in_whichever_component_it_is_in(tmp_path) -> None:
    """The check asked about the first component, which may itself be NaN."""
    path = tmp_path / "seam.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt nan 0\nvt nan 1\nf 1/1 2/1 3/1\nf 1/2 2/1 3/1\n"
    )

    with pytest.warns(UserWarning, match="more than one texture coordinate"):
        read(path)


def test_a_wider_attribute_says_what_it_leaves_out(tmp_path) -> None:
    """A vt record carries two components; the rest went in silence."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"texcoords": np.arange(9, dtype=np.float64).reshape(3, 3)},
    )
    path = tmp_path / "wide.obj"

    with pytest.warns(UserWarning, match="3 components"):
        write(poly, path)

    assert "vt 0 1" in path.read_text()


@pytest.mark.parametrize(
    "face,match",
    [
        ("f 0 1 2", "index 0 is not valid"),
        ("f a 2 3", "index 'a' is not a number"),
        ("f 1/9 2/1 3/1", "texture coordinate index 9"),
        ("f 1//9 2//1 3//1", "normal index 9"),
    ],
)
def test_a_corner_the_fast_path_cannot_take_still_names_the_line(
    tmp_path, face: str, match: str
) -> None:
    """A plain in-range index is resolved inline; the rest keep the message."""
    path = tmp_path / "corner.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n" + face + "\n"
    )

    with pytest.raises(CodecError, match=match):
        read(path)


@pytest.mark.parametrize("token", ["²", "¹"])
def test_a_superscript_face_index_names_the_line(tmp_path, token: str) -> None:
    """str.isdigit admits superscripts and int() then refuses them."""
    path = tmp_path / "super.obj"
    path.write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 {token}\n")

    with pytest.raises(CodecError, match="is not a number"):
        read(path)


def test_a_directive_with_no_argument_names_nothing(tmp_path) -> None:
    """Kept as the empty string, 'mtllib' wrote back a line naming no file."""
    path = tmp_path / "bare.obj"
    path.write_text("mtllib\no\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    poly = read(path)

    assert "mtl_file" not in poly.global_attrs
    assert "object_name" not in poly.global_attrs


def test_a_material_that_does_not_cover_the_faces_is_not_written(tmp_path) -> None:
    """Indexed per face, a short attribute ran off the end mid-write."""
    poly = make_polydata(
        np.zeros((6, 3)),
        [("triangle", np.array([[0, 1, 2], [3, 4, 5]]))],
        element_attrs={"material": np.array(["a"], dtype=object)},
    )
    path = tmp_path / "short.obj"

    with pytest.warns(UserWarning, match="1 value"):
        write(poly, path)

    assert "usemtl" not in path.read_text()


def _one_of(name: str) -> PolyData:
    """A mesh holding a single element of the named type."""
    k = NODES_PER_ELEMENT[name]
    return PolyData(
        vertices=np.zeros((k, 3)),
        connectivity=np.arange(k, dtype=np.int32),
        offsets=np.array([0, k], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES[name]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("name", "becomes"),
    [("tetra", "quad"), ("hexahedron", "polygon"), ("line", "polygon")],
)
def test_an_element_an_f_record_cannot_hold_is_named(tmp_path, name, becomes) -> None:
    """An 'f' record is a flat ring of vertices, so an element that is not one
    keeps its vertices and loses the type it was."""
    path = tmp_path / "flat.obj"
    with pytest.warns(UserWarning, match=f"{name} \\(1\\) -> {becomes}"):
        write(_one_of(name), path)
    assert ELEMENT_TYPES_INV[int(read(path).element_types[0])] == becomes


@pytest.mark.parametrize("name", ["triangle", "quad"])
def test_an_element_obj_holds_is_not_named(tmp_path, name) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(_one_of(name), tmp_path / "kept.obj")


def test_line_and_point_records_are_reported_rather_than_dropped(tmp_path) -> None:
    """They name geometry this codec has no element for, which is a loss and
    not a directive stepped over."""
    path = tmp_path / "lines.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nl 1 2\nl 2 3\np 1\n")
    with pytest.warns(UserWarning, match=r"'l' \(2\), 'p' \(1\)"):
        poly = read(path)
    assert len(poly.element_types) == 1


def test_a_file_carrying_neither_says_nothing(tmp_path) -> None:
    path = tmp_path / "plain.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\ns off\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(read(path).element_types) == 1


# ---------------------------------------------------------------------------
# Splitting a vertex the corners disagree about
# ---------------------------------------------------------------------------


_SEAM = (
    "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
    "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n"
    "vn 0 0 1\n"
    "f 1/1/1 2/2/1 3/3/1\nf 1/5/1 3/3/1 4/4/1\n"
)


def test_a_seam_keeps_both_uvs_when_the_vertex_is_split(tmp_path) -> None:
    """The corner that disagreed gets its own vertex rather than the last
    value written over the first."""
    path = tmp_path / "seam.obj"
    path.write_text(_SEAM)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(path, split_seams=True)

    assert len(poly.vertices) == 5
    uv = poly.vertex_attrs["texcoords"]
    np.testing.assert_allclose(uv[0], [0, 0])
    np.testing.assert_allclose(uv[4], [0.5, 0.5])
    # The copy sits where the vertex it came from sits.
    np.testing.assert_allclose(poly.vertices[4], poly.vertices[0])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 4, 2, 3])


def test_a_file_whose_corners_agree_reads_the_same_either_way(tmp_path) -> None:
    """Splitting is a no-op where there is nothing to split, so it can be
    left on for a directory of files."""
    path = tmp_path / "plain.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    )

    plain = read(path)
    split = read(path, split_seams=True)

    np.testing.assert_array_equal(split.vertices, plain.vertices)
    np.testing.assert_array_equal(split.connectivity, plain.connectivity)
    np.testing.assert_array_equal(split.offsets, plain.offsets)
    np.testing.assert_array_equal(
        split.vertex_attrs["texcoords"], plain.vertex_attrs["texcoords"]
    )


def test_a_hard_edge_splits_on_the_normal_alone(tmp_path) -> None:
    """A vertex two faces give different normals is the same conflict, in
    the other slot."""
    path = tmp_path / "hard.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vn 0 0 1\nvn 1 0 0\n"
        "f 1//1 2//1 3//1\nf 1//2 3//1 4//1\n"
    )

    poly = read(path, split_seams=True)

    assert len(poly.vertices) == 5
    normals = poly.vertex_attrs["normals"]
    np.testing.assert_allclose(normals[0], [0, 0, 1])
    np.testing.assert_allclose(normals[4], [1, 0, 0])


def test_a_vertex_no_face_names_keeps_its_index_through_a_split(tmp_path) -> None:
    """Copies go on the end, so an unnamed vertex is where it was."""
    path = tmp_path / "orphan.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 9 9 9\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 0.5\n"
        "f 1/1 2/2 3/3\nf 1/4 2/2 3/3\n"
    )

    poly = read(path, split_seams=True)

    assert len(poly.vertices) == 5
    np.testing.assert_allclose(poly.vertices[3], [9, 9, 9])
    assert np.isnan(poly.vertex_attrs["texcoords"][3]).all()


def test_corners_that_name_different_slots_split_on_the_one_that_differs(
    tmp_path,
) -> None:
    """'f 1//1' and 'f 1/2/1' disagree about the texture coordinate, which a
    missing index is as much as a different one."""
    path = tmp_path / "mixed.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "vn 0 0 1\n"
        "f 1//1 2//1 3//1\nf 1/1/1 3//1 4//1\n"
    )

    poly = read(path, split_seams=True)

    assert len(poly.vertices) == 5
    uv = poly.vertex_attrs["texcoords"]
    assert np.isnan(uv[0]).all()
    np.testing.assert_allclose(uv[4], [0, 0])


def test_a_split_leaves_the_elements_where_they_were(tmp_path) -> None:
    """Only the indices inside an f record move; the records do not."""
    path = tmp_path / "tagged.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n"
        "usemtl skin\ng left\nf 1/1 2/2 3/3\n"
        "g right\nf 1/5 3/3 4/4\n"
    )

    with pytest.warns(UserWarning, match="texture coordinate"):
        plain = read(path)
    split = read(path, split_seams=True)

    np.testing.assert_array_equal(split.element_types, plain.element_types)
    np.testing.assert_array_equal(split.offsets, plain.offsets)
    np.testing.assert_array_equal(
        split.element_attrs["material"], plain.element_attrs["material"]
    )
    assert split.element_tags.keys() == plain.element_tags.keys()
    for name, members in plain.element_tags.items():
        np.testing.assert_array_equal(split.element_tags[name], members)


def test_the_seam_warning_names_the_way_out(tmp_path) -> None:
    path = tmp_path / "seam.obj"
    path.write_text(_SEAM)
    with pytest.warns(UserWarning, match="split_seams=True"):
        read(path)


def test_a_split_mesh_round_trips_with_its_uvs(tmp_path) -> None:
    path = tmp_path / "seam.obj"
    path.write_text(_SEAM)
    poly = read(path, split_seams=True)

    out = tmp_path / "out.obj"
    write(poly, out)
    back = read(out)

    np.testing.assert_allclose(back.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_allclose(
        back.vertex_attrs["texcoords"], poly.vertex_attrs["texcoords"], atol=1e-8
    )


def test_records_nothing_indexes_follow_the_vertex_a_copy_came_from(tmp_path) -> None:
    """The vt records line up with the v records and no face names them, so
    a split driven by the normals must not cost them their footing."""
    path = tmp_path / "unindexed.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "vn 0 0 1\nvn 1 0 0\n"
        "f 1//1 2//1 3//1\nf 1//2 3//1 4//1\n"
    )

    with pytest.warns(UserWarning, match="more than one normal"):
        plain = read(path)
    split = read(path, split_seams=True)

    assert len(split.vertices) == 5
    uv = split.vertex_attrs["texcoords"]
    np.testing.assert_array_equal(uv[:4], plain.vertex_attrs["texcoords"])
    np.testing.assert_allclose(uv[4], uv[0])


def test_records_nothing_indexes_are_counted_against_the_file(tmp_path) -> None:
    """Five vt for four v is five vt for four v however many vertices the
    split ends with; a count that matches by accident is not a match."""
    path = tmp_path / "miscounted.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 9 9\n"
        "vn 0 0 1\nvn 1 0 0\n"
        "f 1//1 2//1 3//1\nf 1//2 3//1 4//1\n"
    )

    with pytest.warns(UserWarning, match="5 texture coordinate"):
        split = read(path, split_seams=True)

    assert len(split.vertices) == 5
    assert "texcoords" not in split.vertex_attrs


def test_a_file_naming_no_vt_or_vn_is_left_alone_by_a_split(tmp_path) -> None:
    """Corners that name neither record have nothing to disagree about."""
    path = tmp_path / "bare.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n")

    plain = read(path)
    split = read(path, split_seams=True)

    np.testing.assert_array_equal(split.vertices, plain.vertices)
    np.testing.assert_array_equal(split.connectivity, plain.connectivity)


def test_a_split_resolves_negative_indices_the_way_a_plain_read_does(tmp_path) -> None:
    """A relative corner is an absolute one by the time it is split."""
    path = tmp_path / "relative.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n"
        "f -4/-5 -3/-4 -2/-3\nf -4/-1 -2/-3 -1/-2\n"
    )

    poly = read(path, split_seams=True)

    assert len(poly.vertices) == 5
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 4, 2, 3])
    np.testing.assert_allclose(poly.vertex_attrs["texcoords"][4], [0.5, 0.5])


def test_a_vertex_named_twice_in_one_face_splits(tmp_path) -> None:
    """The corners of a single face disagree as readily as two faces do."""
    path = tmp_path / "degenerate.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/1 2/2 1/3\n"
    )

    poly = read(path, split_seams=True)

    assert len(poly.vertices) == 4
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 3])
    np.testing.assert_allclose(poly.vertices[3], poly.vertices[0])
