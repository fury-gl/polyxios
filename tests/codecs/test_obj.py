from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
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


def test_multi_group_element_tags() -> None:
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
# P1.1 - texture coordinates, normals and face indices
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
# P1.2 - groups on write
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
