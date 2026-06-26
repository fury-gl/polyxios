from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata, read as px_read, write as px_write
from polyxios.codecs._ugrid import read, write
from polyxios.exceptions import CodecError

# A hand-written file in the layout the format fixes: header, nodes, triangles,
# quads, triangle IDs, quad IDs, then the volume sections. Five nodes carry one
# triangle and one tetrahedron, so every boundary between sections is exercised.
_SAMPLE = """5 1 0 1 0 0 0
0 0 0
1 0 0
0 1 0
0 0 1
1 1 1
1 2 3
7
1 2 3 4
"""


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def _mixed_mesh():
    """A mesh holding one element of every section, volumes before faces."""
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    return make_polydata(
        verts,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("wedge", np.array([[0, 1, 2, 3, 4, 5]])),
            ("pyramid", np.array([[0, 1, 2, 3, 4]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )


def test_reads_the_documented_layout(tmp_path: Path) -> None:
    path = tmp_path / "sample.ugrid"
    path.write_text(_SAMPLE)
    poly = read(path)

    assert poly.vertices.shape == (5, 3)
    np.testing.assert_allclose(poly.vertices[4], [1.0, 1.0, 1.0])
    # Triangle first, then the tetrahedron: the surface IDs between them are
    # not connectivity and must not shift the volume section.
    np.testing.assert_array_equal(poly.element_types, [5, 10])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 1, 2, 3])
    np.testing.assert_array_equal(poly.offsets, [0, 3, 7])
    np.testing.assert_array_equal(poly.element_tags["boundary_7"], [0])


def test_header_is_nodes_first(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    write(_tri_mesh(), path)
    counts = [int(v) for v in path.read_text().splitlines()[0].split()]
    # n_nodes n_tri n_quad n_tet n_pyramid n_prism n_hex
    assert counts == [4, 2, 0, 0, 0, 0, 0]


def test_surface_ids_precede_the_volume_sections(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"boundary_4": np.array([0])},
    )
    path = tmp_path / "m.ugrid"
    write(poly, path)
    lines = path.read_text().splitlines()
    # header, 4 nodes, 1 triangle, 1 surface ID, 1 tetrahedron
    assert lines[5] == "1 2 3"
    assert lines[6] == "4"
    assert lines[7] == "1 2 3 4"


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.element_types, poly.element_types)


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tri_mesh()
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    assert len(back.element_types) == 2
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_roundtrip_every_section_in_file_order(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)

    # Written section by section, so the mesh comes back in the format's order.
    names = ["triangle", "quad", "tetra", "pyramid", "wedge", "hexahedron"]
    codes = [5, 9, 10, 14, 13, 12]
    np.testing.assert_array_equal(back.element_types, codes)
    np.testing.assert_allclose(back.vertices, poly.vertices)

    original = {
        int(poly.element_types[i]): poly.connectivity[
            poly.offsets[i] : poly.offsets[i + 1]
        ]
        for i in range(len(poly.element_types))
    }
    for i, (name, code) in enumerate(zip(names, codes)):
        cell = back.connectivity[back.offsets[i] : back.offsets[i + 1]]
        np.testing.assert_array_equal(cell, original[code], err_msg=name)


def test_pyramid_is_permuted_on_disk(tmp_path: Path) -> None:
    verts = np.arange(15, dtype=np.float64).reshape(5, 3)
    poly = make_polydata(verts, [("pyramid", np.array([[0, 1, 2, 3, 4]]))])
    path = tmp_path / "m.ugrid"
    write(poly, path)
    # UGRID spells the apex third; polyxios spells it last.
    assert path.read_text().splitlines()[6] == "2 1 5 3 4"
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2, 3, 4])


def test_tags_follow_the_element_not_its_position(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
    )
    # Volume element first: a writer indexing the tag array by surface position
    # instead of by element index reads the tetrahedron's slot for the triangle.
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 4]])),
        ],
        element_tags={"boundary_9": np.array([1])},
    )
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    assert set(back.element_tags) == {"boundary_9"}
    # The triangle leads the element array after a write.
    np.testing.assert_array_equal(back.element_tags["boundary_9"], [0])
    np.testing.assert_array_equal(
        back.connectivity[back.offsets[0] : back.offsets[1]], [0, 1, 4]
    )


def test_untagged_faces_take_surface_id_one(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    write(_tri_mesh(), path)
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["boundary_1"], [0, 1])
    # A second trip changes nothing.
    again = tmp_path / "again.ugrid"
    write(back, again)
    assert again.read_text() == path.read_text()


def test_untagged_faces_do_not_join_an_existing_tag(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    # boundary_1 already holds ID 1, so the untagged face has to take another
    # one or the two patches come back fused.
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"boundary_1": np.array([0])},
    )
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["boundary_1"], [0])
    np.testing.assert_array_equal(back.element_tags["boundary_2"], [1])


def test_unnumbered_tag_name_takes_a_free_id(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"boundary_1": np.array([0]), "wall": np.array([1])},
    )
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    assert set(back.element_tags) == {"boundary_1", "boundary_2"}
    np.testing.assert_array_equal(back.element_tags["boundary_1"], [0])
    np.testing.assert_array_equal(back.element_tags["boundary_2"], [1])


def test_volume_tag_member_is_dropped_with_a_warning(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"boundary_3": np.array([0])},
    )
    path = tmp_path / "m.ugrid"
    with pytest.warns(UserWarning, match="not boundary faces"):
        write(poly, path)
    assert read(path).element_tags == {}


def test_out_of_range_tag_member_is_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _tri_mesh().vertices,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"boundary_3": np.array([0, 99])},
    )
    with pytest.warns(UserWarning, match="outside the mesh"):
        write(poly, tmp_path / "m.ugrid")


def test_shared_face_keeps_the_first_tag(tmp_path: Path) -> None:
    poly = make_polydata(
        _tri_mesh().vertices,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"boundary_2": np.array([0]), "boundary_5": np.array([0, 1])},
    )
    path = tmp_path / "m.ugrid"
    with pytest.warns(UserWarning, match="more than one tag"):
        write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["boundary_2"], [0])
    np.testing.assert_array_equal(back.element_tags["boundary_5"], [1])


def test_unsupported_element_type_is_skipped(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1]])), ("triangle", np.array([[0, 1, 2]]))],
    )
    path = tmp_path / "m.ugrid"
    with pytest.warns(UserWarning, match="unsupported element type"):
        write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.element_types, [5])


def test_empty_mesh_roundtrips(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0
    assert back.element_tags == {}


def test_short_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("1 0\n")
    with pytest.raises(CodecError, match="expected 7"):
        read(path)


def test_truncated_body_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("3 1 0 0 0 0 0\n0 0 0\n1 0 0\n")
    with pytest.raises(CodecError, match="truncated"):
        read(path)


def test_negative_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("-1 0 0 0 0 0 0\n")
    with pytest.raises(CodecError, match="negative node count"):
        read(path)


def test_absurd_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("999999999999 0 0 0 0 0 0\n")
    with pytest.raises(CodecError, match="safety cap"):
        read(path)


def test_non_integer_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("x 0 0 0 0 0 0\n")
    with pytest.raises(CodecError, match="non-integer node count"):
        read(path)


def test_non_numeric_body_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("1 0 0 0 0 0 0\n0 0 nan_x\n")
    with pytest.raises(CodecError, match="non-numeric"):
        read(path)


def test_non_integer_node_reference_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("3 1 0 0 0 0 0\n0 0 0\n1 0 0\n0 1 0\n1 2.5 3\n1\n")
    with pytest.raises(CodecError, match="non-integer node reference"):
        read(path)


def test_zero_node_reference_raises(tmp_path: Path) -> None:
    # UGRID counts from 1, so 0 is out of range rather than the first node.
    path = tmp_path / "m.ugrid"
    path.write_text("3 1 0 0 0 0 0\n0 0 0\n1 0 0\n0 1 0\n0 1 2\n1\n")
    with pytest.raises(CodecError, match="outside 1..3"):
        read(path)


def test_node_reference_past_the_end_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text("3 1 0 0 0 0 0\n0 0 0\n1 0 0\n0 1 0\n1 2 9\n1\n")
    with pytest.raises(CodecError, match="outside 1..3"):
        read(path)


def test_trailing_values_warn(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    path.write_text(_SAMPLE + "1 2 3\n")
    with pytest.warns(UserWarning, match="follow the sections"):
        read(path)


def test_binary_variant_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.lb8.ugrid"
    path.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(CodecError, match="binary 'lb8'"):
        read(path)
    with pytest.raises(CodecError, match="binary 'lb8'"):
        write(_tri_mesh(), path)


def test_binary_content_is_refused_without_the_infix(tmp_path: Path) -> None:
    # No infix to go by: the seven raw counts a binary file opens with carry
    # NULs, and no ASCII UGRID file holds one anywhere.
    path = tmp_path / "mesh.ugrid"
    path.write_bytes(b"\x00\x00\x00\x05\x00\x00\x00\x01" + b"\x00" * 32)
    with pytest.raises(CodecError, match="opens with binary data"):
        read(path)


def test_ascii_file_named_like_an_infix_is_not_refused(tmp_path: Path) -> None:
    # ``b4`` is the mesh's name here, not a stem-less binary suffix; refusing it
    # would lock the caller out of a file that reads perfectly well.
    path = tmp_path / "b4.ugrid"
    write(_tri_mesh(), path)
    np.testing.assert_array_equal(read(path).element_types, [5, 5])


def test_binary_path_is_refused_before_the_option_warning(tmp_path: Path) -> None:
    path = tmp_path / "m.b8.ugrid"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(CodecError, match="binary 'b8'"):
            write(_tri_mesh(), path, nope=1)
    assert caught == []


def test_missing_file_raises_codec_error(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="cannot read"):
        read(tmp_path / "absent.ugrid")


def test_lazy_warns(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    write(_tri_mesh(), path)
    with pytest.warns(UserWarning, match="lazy=True"):
        read(path, lazy=True)


def test_unknown_option_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tri_mesh(), tmp_path / "m.ugrid", nope=1)


def test_float_fmt_is_honoured(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    write(_tri_mesh(), path, float_fmt=".3f")
    assert path.read_text().splitlines()[1] == "0.000 0.000 0.000"


def test_unusable_float_fmt_raises(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="float_fmt"):
        write(_tri_mesh(), tmp_path / "m.ugrid", float_fmt="d")
    with pytest.raises(CodecError, match="not a number"):
        write(_tri_mesh(), tmp_path / "m.ugrid", float_fmt=",.2f")


def test_element_width_mismatch_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="expected 3"):
        write(broken, tmp_path / "m.ugrid")


def test_vertex_reference_out_of_range_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 99], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="outside 0..3"):
        write(broken, tmp_path / "m.ugrid")


def test_short_offsets_raise(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 2, 0, 1, 3], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([5, 5], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="offsets carry"):
        write(broken, tmp_path / "m.ugrid")


def test_offsets_past_the_connectivity_raise(tmp_path: Path) -> None:
    # Gathering the element would read off the end of the connectivity; the
    # codec has to say so rather than let a bare IndexError out.
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([3, 6], dtype=np.int32),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="outside the connectivity"):
        write(broken, tmp_path / "m.ugrid")


def test_negative_offset_raises(tmp_path: Path) -> None:
    # numpy counts a negative index from the end, so this would otherwise write
    # some other element's nodes without a word.
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([-3, 0], dtype=np.int32),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="outside the connectivity"):
        write(broken, tmp_path / "m.ugrid")


def test_dropped_tag_does_not_consume_a_surface_id(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    # ``wall_a`` names the tetrahedron only, so it writes no face at all;
    # reserving an ID for it would push ``wall_b`` off 1 for nothing.
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"wall_a": np.array([1]), "wall_b": np.array([0])},
    )
    path = tmp_path / "m.ugrid"
    with pytest.warns(UserWarning, match="not boundary faces"):
        write(poly, path)
    back = read(path)
    assert set(back.element_tags) == {"boundary_1"}
    np.testing.assert_array_equal(back.element_tags["boundary_1"], [0])


def test_duplicate_surface_id_is_renumbered(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    # Two names spelling ID 3: written as they stand the patches come back
    # fused into one tag, which no later read can take apart again.
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"boundary_3": np.array([0]), "boundary_03": np.array([1])},
    )
    path = tmp_path / "m.ugrid"
    with pytest.warns(UserWarning, match="already holds"):
        write(poly, path)
    back = read(path)
    assert len(back.element_tags) == 2
    np.testing.assert_array_equal(back.element_tags["boundary_3"], [0])
    assert sorted(np.concatenate(list(back.element_tags.values())).tolist()) == [0, 1]


def test_two_column_vertices_raise(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices[:, :2],
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match=r"expected \(n, 3\)"):
        write(broken, tmp_path / "m.ugrid")


def test_registered_under_its_extension(tmp_path: Path) -> None:
    path = tmp_path / "m.ugrid"
    px_write(_tri_mesh(), path)
    back = px_read(path)
    np.testing.assert_allclose(back.vertices, _tri_mesh().vertices)


@pytest.mark.parametrize("value", [2**53, 2**53 + 1, -(2**53) - 1])
def test_surface_id_float64_cannot_hold_raises(tmp_path: Path, value: int) -> None:
    # 2**53 + 1 is already 2**53 by the time the body is float64, so a bound
    # that let 2**53 through would hand two different files the same tag.
    path = tmp_path / "m.ugrid"
    path.write_text(f"3 1 0 0 0 0 0\n0 0 0\n1 0 0\n0 1 0\n1 2 3\n{value}\n")
    with pytest.raises(CodecError, match="surface ID out of range"):
        read(path)


def test_tag_number_float64_cannot_hold_is_renumbered(tmp_path: Path) -> None:
    # Keeping the number would write an ID the reader refuses outright.
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={f"boundary_{2**53}": np.array([0])},
    )
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["boundary_1"], [0])
    np.testing.assert_array_equal(back.element_tags["boundary_2"], [1])


def test_interleaved_surface_ids_group_in_element_order(tmp_path: Path) -> None:
    # Triangle IDs and quad IDs are two runs in the file but one tag each side
    # of them, so a patch spans both and its members have to come back ascending.
    verts = np.arange(15, dtype=np.float64).reshape(5, 3)
    path = tmp_path / "m.ugrid"
    path.write_text(
        "5 3 2 0 0 0 0\n"
        + "\n".join(" ".join(str(c) for c in xyz) for xyz in verts.tolist())
        + "\n1 2 3\n2 3 4\n1 3 4\n1 2 3 4\n2 3 4 5\n"  # 3 triangles, 2 quads
        + "7 2 7\n2 7\n"  # triangle IDs, then quad IDs
    )
    back = read(path)
    assert set(back.element_tags) == {"boundary_2", "boundary_7"}
    np.testing.assert_array_equal(back.element_tags["boundary_2"], [1, 3])
    np.testing.assert_array_equal(back.element_tags["boundary_7"], [0, 2, 4])


def test_offsets_span_multi_element_sections(tmp_path: Path) -> None:
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2], [1, 2, 3]])),
            ("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]])),
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        ],
    )
    path = tmp_path / "m.ugrid"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.offsets, [0, 3, 6, 10, 14, 22])
    np.testing.assert_array_equal(back.element_types, [5, 5, 10, 10, 12])
    for i in range(len(back.element_types)):
        cell = back.connectivity[back.offsets[i] : back.offsets[i + 1]]
        np.testing.assert_array_equal(
            cell, poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]]
        )
