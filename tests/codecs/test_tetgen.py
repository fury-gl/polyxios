from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata, read as api_read, write as api_write
from polyxios.codecs._tetgen import _MAX_ATTR_COLUMNS as _MAX_ATTR, read, write
from polyxios.exceptions import CodecError

_VERTS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
)


def _tet_mesh(**kwargs):
    return make_polydata(_VERTS[:4], [("tetra", np.array([[0, 1, 2, 3]]))], **kwargs)


def _pair(tmp_path: Path, node: str, ele: str | None = None, stem: str = "mesh"):
    """Write a .node (and optional .ele) pair and return the .ele path.

    The encoding is named because Windows would otherwise pick cp1252, which
    has no spelling for the BOM one of these files opens with.
    """
    (tmp_path / f"{stem}.node").write_text(node, encoding="utf-8")
    if ele is not None:
        (tmp_path / f"{stem}.ele").write_text(ele, encoding="utf-8")
    return tmp_path / f"{stem}.ele"


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path: Path) -> None:
    poly = _tet_mesh()
    ele = tmp_path / "mesh.ele"
    write(poly, ele)
    assert ele.exists()
    assert (tmp_path / "mesh.node").exists()
    back = read(ele)
    assert len(back.element_types) == 1
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)


def test_multiple_tets(tmp_path: Path) -> None:
    poly = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3], [1, 4, 2, 3]]))])
    ele = tmp_path / "mesh.ele"
    write(poly, ele)
    back = read(ele)
    assert len(back.element_types) == 2
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_empty_mesh_roundtrips(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3)), [])
    ele = tmp_path / "mesh.ele"
    write(poly, ele)
    back = read(ele)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0
    np.testing.assert_array_equal(back.offsets, [0])


def test_registered_under_both_extensions(tmp_path: Path) -> None:
    poly = _tet_mesh()
    api_write(poly, tmp_path / "viaapi.ele")
    np.testing.assert_allclose(
        api_read(tmp_path / "viaapi.node").vertices, poly.vertices
    )
    np.testing.assert_allclose(
        api_read(tmp_path / "viaapi.ele").vertices, poly.vertices
    )


# ---------------------------------------------------------------------------
# Index bases and node numbering
# ---------------------------------------------------------------------------


def test_ele_file_uses_1based_indices(tmp_path: Path) -> None:
    write(_tet_mesh(), tmp_path / "mesh.ele")
    lines = (tmp_path / "mesh.ele").read_text().strip().splitlines()
    assert all(int(p) >= 1 for p in lines[1].split())
    assert lines[0].split()[:2] == ["1", "4"]


def test_zero_based_node_file(tmp_path: Path) -> None:
    """A ``-z`` mesh numbers both files from 0 and must not shift."""
    ele = _pair(
        tmp_path,
        "4 3 0 0\n0 0 0 0\n1 1 0 0\n2 0 1 0\n3 0 0 1\n",
        "1 4 0\n0 0 1 2 3\n",
    )
    np.testing.assert_array_equal(read(ele).connectivity, [0, 1, 2, 3])


def test_element_numbering_does_not_shift_nodes(tmp_path: Path) -> None:
    """Elements numbered from 0 beside nodes numbered from 1."""
    ele = _pair(
        tmp_path,
        "4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n",
        "1 4 0\n0 1 2 3 4\n",
    )
    np.testing.assert_array_equal(read(ele).connectivity, [0, 1, 2, 3])


def test_gapped_node_numbering(tmp_path: Path) -> None:
    ele = _pair(
        tmp_path,
        "4 3 0 0\n10 0 0 0\n5 1 0 0\n77 0 1 0\n2 0 0 1\n",
        "1 4 0\n1 77 2 10 5\n",
    )
    back = read(ele)
    # rows are 10, 5, 77, 2 in file order; the element names 77, 2, 10, 5.
    np.testing.assert_array_equal(back.connectivity, [2, 3, 0, 1])


def test_unknown_node_reference_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "2 3 0 0\n1 0 0 0\n2 1 0 0\n", "1 4 0\n1 1 2 3 99\n")
    with pytest.raises(CodecError, match="outside"):
        read(ele)


def test_duplicate_node_number_warns(tmp_path: Path) -> None:
    ele = _pair(
        tmp_path,
        "4 3 0 0\n1 0 0 0\n1 9 9 9\n3 0 1 0\n4 0 0 1\n",
        "1 4 0\n1 1 3 4 1\n",
    )
    with pytest.warns(UserWarning, match="numbers two or more nodes alike"):
        read(ele)


# ---------------------------------------------------------------------------
# Header columns that must be counted, not assumed
# ---------------------------------------------------------------------------


def test_node_attributes_are_read_not_misparsed(tmp_path: Path) -> None:
    ele = _pair(
        tmp_path,
        "2 3 1 1\n1 0 0 0 9.5 1\n2 1 0 0 8.5 2\n",
        "0 4 0\n",
    )
    back = read(ele)
    np.testing.assert_allclose(back.vertices, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(back.vertex_attrs["attr_0"], [9.5, 8.5])
    np.testing.assert_array_equal(back.vertex_tags["boundary_1"], [0])
    np.testing.assert_array_equal(back.vertex_tags["boundary_2"], [1])


def test_two_dimensional_node_file_pads_z(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "3 2 0 0\n1 0 0\n2 1 0\n3 0 1\n", "0 4 0\n")
    np.testing.assert_allclose(read(ele).vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_bad_dimension_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "1 4 0 0\n1 0 0 0 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="dimension 4"):
        read(ele)


def test_element_attributes_named_by_count(tmp_path: Path) -> None:
    nodes = "4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
    one = read(_pair(tmp_path, nodes, "1 4 1\n1 1 2 3 4 7.5\n", stem="a"))
    np.testing.assert_allclose(one.element_attrs["region"], [7.5])

    two = read(_pair(tmp_path, nodes, "1 4 2\n1 1 2 3 4 7.5 8.5\n", stem="b"))
    assert set(two.element_attrs) == {"region_0", "region_1"}
    # One value per element, never one per attribute column.
    assert all(v.shape == (1,) for v in two.element_attrs.values())


def test_quadratic_element_attributes_read_from_the_right_columns(
    tmp_path: Path,
) -> None:
    """A 10-node element pushes its attributes past column 5, not to it."""
    nodes = "10 3 0 0\n" + "".join(f"{i + 1} {i} 0 0\n" for i in range(10))
    body = " ".join(str(i + 1) for i in range(10))
    ele = _pair(tmp_path, nodes, f"1 10 1\n1 {body} 42.5\n")
    with pytest.warns(UserWarning, match="quadratic"):
        back = read(ele)
    np.testing.assert_allclose(back.element_attrs["region"], [42.5])

    two = _pair(tmp_path, nodes, f"1 10 2\n1 {body} 42.5 43.5\n", stem="two")
    with pytest.warns(UserWarning, match="quadratic"):
        back = read(two)
    np.testing.assert_allclose(back.element_attrs["region_0"], [42.5])
    np.testing.assert_allclose(back.element_attrs["region_1"], [43.5])


def test_quadratic_tets_linearized_with_warning(tmp_path: Path) -> None:
    nodes = "10 3 0 0\n" + "".join(f"{i + 1} {i} 0 0\n" for i in range(10))
    body = " ".join(str(i + 1) for i in range(10))
    ele = _pair(tmp_path, nodes, f"1 10 0\n1 {body}\n")
    with pytest.warns(UserWarning, match="quadratic"):
        back = read(ele)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2, 3])


def test_unsupported_element_width_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "3 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n", "1 3 0\n1 1 2 3\n")
    with pytest.raises(CodecError, match="expected 4"):
        read(ele)


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_node_file_missing_raises(tmp_path: Path) -> None:
    ele = tmp_path / "ghost.ele"
    ele.write_text("1 4 0\n1 1 2 3 4\n", encoding="utf-8")
    with pytest.raises(CodecError, match=".node file not found"):
        read(ele)


def test_missing_ele_reads_point_cloud(tmp_path: Path) -> None:
    (tmp_path / "cloud.node").write_text(
        "2 3 0 0\n1 0 0 0\n2 1 0 0\n", encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="point cloud"):
        back = read(tmp_path / "cloud.node")
    assert len(back.element_types) == 0
    np.testing.assert_allclose(back.vertices, [[0, 0, 0], [1, 0, 0]])


def test_named_ele_file_missing_raises(tmp_path: Path) -> None:
    """A caller who named the .ele half gets an error, not a point cloud."""
    (tmp_path / "half.node").write_text("2 3 0 0\n1 0 0 0\n2 1 0 0\n", encoding="utf-8")
    with pytest.raises(CodecError, match=".ele file not found"):
        read(tmp_path / "half.ele")


def test_unrepresentable_node_number_raises(tmp_path: Path) -> None:
    """A number past float64's exact range would resolve to another vertex."""
    ele = _pair(tmp_path, "1 3 0 0\n99999999999999999999 0 0 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="node number out of range"):
        read(ele)


def test_fractional_node_number_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "1 3 0 0\n1.5 0 0 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="non-integer node number"):
        read(ele)


def test_negative_count_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "-5 3 0 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="negative node count"):
        read(ele)


def test_absurd_count_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "99999999999 3 0 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="safety cap"):
        read(ele)


def test_absurd_node_attribute_count_raises(tmp_path: Path) -> None:
    """A column count is not bounded by the body when no rows are declared."""
    ele = _pair(tmp_path, "0 3 100000000 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="node attribute count"):
        read(ele)


def test_absurd_element_attribute_count_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "0 3 0 0\n", "0 4 100000000\n")
    with pytest.raises(CodecError, match="element attribute count"):
        read(ele)


def test_truncated_body_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "3 3 0 0\n1 0 0 0\n2 1 0 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="truncated"):
        read(ele)


def test_non_numeric_body_raises(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "1 3 0 0\n1 0 x 0\n", "0 4 0\n")
    with pytest.raises(CodecError, match="non-numeric"):
        read(ele)


def test_trailing_values_warn(tmp_path: Path) -> None:
    ele = _pair(tmp_path, "1 3 0 0\n1 0 0 0\n2 9 9 9\n", "0 4 0\n")
    with pytest.warns(UserWarning, match="past the 1 node"):
        read(ele)


def test_comments_and_bom_ignored(tmp_path: Path) -> None:
    ele = _pair(
        tmp_path,
        "﻿# a comment\n2 3 0 0\n1 0 0 0 # trailing\n2 1 0 0\n",
        "# header comment\n0 4 0\n",
    )
    np.testing.assert_allclose(read(ele).vertices, [[0, 0, 0], [1, 0, 0]])


def test_lazy_warns(tmp_path: Path) -> None:
    write(_tet_mesh(), tmp_path / "mesh.ele")
    with pytest.warns(UserWarning, match="lazy=True"):
        read(tmp_path / "mesh.ele", lazy=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_to_node_path_does_not_clobber(tmp_path: Path) -> None:
    poly = _tet_mesh()
    write(poly, tmp_path / "pair.node")
    assert (tmp_path / "pair.ele").exists()
    assert (tmp_path / "pair.node").exists()
    np.testing.assert_allclose(read(tmp_path / "pair.ele").vertices, poly.vertices)


def test_write_to_extensionless_path(tmp_path: Path) -> None:
    write(_tet_mesh(), tmp_path / "bare")
    assert (tmp_path / "bare.ele").exists()
    assert (tmp_path / "bare.node").exists()


def test_non_tetra_elements_warn_and_are_skipped(tmp_path: Path) -> None:
    poly = make_polydata(_VERTS[:4], [("triangle", np.array([[0, 1, 2]]))])
    with pytest.warns(UserWarning, match="unsupported element type"):
        write(poly, tmp_path / "mesh.ele")
    assert (tmp_path / "mesh.ele").read_text().split()[0] == "0"
    # Vertices still travel, so a mixed mesh keeps its coordinates.
    np.testing.assert_allclose(read(tmp_path / "mesh.ele").vertices, poly.vertices)


def test_out_of_range_vertex_reference_raises(tmp_path: Path) -> None:
    poly = make_polydata(_VERTS[:4], [("tetra", np.array([[0, 1, 2, 3]]))])
    broken = type(poly)(
        vertices=poly.vertices[:3],
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="outside"):
        write(broken, tmp_path / "mesh.ele")


def test_markers_and_regions_roundtrip(tmp_path: Path) -> None:
    poly = _tet_mesh(
        vertex_tags={
            "boundary_1": np.array([0, 3], dtype=np.int32),
            "boundary_2": np.array([1], dtype=np.int32),
        },
        element_attrs={"region": np.array([7.5])},
    )
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    np.testing.assert_array_equal(back.vertex_tags["boundary_1"], [0, 3])
    np.testing.assert_array_equal(back.vertex_tags["boundary_2"], [1])
    np.testing.assert_allclose(back.element_attrs["region"], [7.5])


def test_unnumbered_tag_names_get_a_marker(tmp_path: Path) -> None:
    poly = _tet_mesh(vertex_tags={"inlet": np.array([0], dtype=np.int32)})
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    assert list(back.vertex_tags) == ["boundary_1"]
    np.testing.assert_array_equal(back.vertex_tags["boundary_1"], [0])


def test_shared_vertex_tag_warns(tmp_path: Path) -> None:
    poly = _tet_mesh(
        vertex_tags={
            "boundary_1": np.array([0, 1], dtype=np.int32),
            "boundary_2": np.array([1], dtype=np.int32),
        }
    )
    with pytest.warns(UserWarning, match="more than one tag"):
        write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    np.testing.assert_array_equal(back.vertex_tags["boundary_1"], [0, 1])
    assert "boundary_2" not in back.vertex_tags


def test_out_of_range_tag_member_warns(tmp_path: Path) -> None:
    poly = _tet_mesh(vertex_tags={"boundary_1": np.array([0, 99], dtype=np.int32)})
    with pytest.warns(UserWarning, match="outside the mesh"):
        write(poly, tmp_path / "mesh.ele")
    np.testing.assert_array_equal(
        read(tmp_path / "mesh.ele").vertex_tags["boundary_1"], [0]
    )


def test_vertex_attrs_roundtrip_by_position(tmp_path: Path) -> None:
    poly = _tet_mesh(
        vertex_attrs={"temp": np.arange(4.0), "pressure": np.arange(4.0) * 2}
    )
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    # Written in name order, read back numbered: pressure first, then temp.
    np.testing.assert_allclose(back.vertex_attrs["attr_0"], np.arange(4.0) * 2)
    np.testing.assert_allclose(back.vertex_attrs["attr_1"], np.arange(4.0))


def test_many_vertex_attrs_keep_their_column_order(tmp_path: Path) -> None:
    """Past ten attributes, name order has to count `attr_10` after `attr_9`."""
    poly = _tet_mesh(
        vertex_attrs={f"attr_{k}": np.full(4, float(k)) for k in range(12)}
    )
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    for k in range(12):
        np.testing.assert_allclose(back.vertex_attrs[f"attr_{k}"], np.full(4, float(k)))


def test_many_boundary_tags_keep_their_markers(tmp_path: Path) -> None:
    poly = make_polydata(
        _VERTS[:4],
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_tags={
            "boundary_2": np.array([0], dtype=np.int32),
            "boundary_10": np.array([1], dtype=np.int32),
        },
    )
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    np.testing.assert_array_equal(back.vertex_tags["boundary_2"], [0])
    np.testing.assert_array_equal(back.vertex_tags["boundary_10"], [1])


def test_boundary_0_tag_is_renumbered(tmp_path: Path) -> None:
    """Marker 0 is TetGen's word for unmarked, so the tag cannot keep it."""
    poly = _tet_mesh(vertex_tags={"boundary_0": np.array([2], dtype=np.int32)})
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    assert list(back.vertex_tags) == ["boundary_1"]
    np.testing.assert_array_equal(back.vertex_tags["boundary_1"], [2])


def test_unreadably_wide_marker_is_renumbered(tmp_path: Path) -> None:
    """A marker past float64's exact range would not survive being read."""
    poly = _tet_mesh(
        vertex_tags={
            f"boundary_{2**63 - 1}": np.array([0], dtype=np.int32),
            "boundary_3": np.array([1], dtype=np.int32),
        }
    )
    write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    # The wide name takes a fresh number that the kept one does not already use.
    np.testing.assert_array_equal(back.vertex_tags["boundary_3"], [1])
    np.testing.assert_array_equal(back.vertex_tags["boundary_1"], [0])


def test_write_keeps_the_case_the_caller_named(tmp_path: Path) -> None:
    """A case-sensitive filesystem gets the file that was asked for."""
    write(_tet_mesh(), tmp_path / "MESH.ELE")
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"MESH.ELE", "MESH.NODE"}
    np.testing.assert_allclose(
        read(tmp_path / "MESH.ELE").vertices, _tet_mesh().vertices
    )


def test_write_to_uppercase_node_half_names_the_sibling_alike(tmp_path: Path) -> None:
    write(_tet_mesh(), tmp_path / "MESH.NODE")
    assert {p.name for p in tmp_path.iterdir()} == {"MESH.ELE", "MESH.NODE"}


def test_uppercase_extensions_find_their_sibling(tmp_path: Path) -> None:
    """Only exercises ``_find_half`` on a case-sensitive filesystem.

    Elsewhere the lower-cased guess already matches the file on disk, so this
    asserts nothing about the fallback. Linux CI is what covers it.
    """
    poly = _tet_mesh()
    write(poly, tmp_path / "mesh.ele")
    (tmp_path / "mesh.node").rename(tmp_path / "MESH.NODE")
    (tmp_path / "mesh.ele").rename(tmp_path / "MESH.ELE")
    np.testing.assert_allclose(read(tmp_path / "MESH.ELE").vertices, poly.vertices)


def test_wide_vertex_attr_skipped_with_warning(tmp_path: Path) -> None:
    poly = _tet_mesh(vertex_attrs={"normals": np.zeros((4, 3))})
    with pytest.warns(UserWarning, match="no column in a .node file"):
        write(poly, tmp_path / "mesh.ele")
    assert (tmp_path / "mesh.node").read_text().splitlines()[0].split()[2] == "0"


def test_extra_element_attrs_warn_and_are_dropped(tmp_path: Path) -> None:
    poly = _tet_mesh(
        element_attrs={"region_0": np.array([1.0]), "region_1": np.array([2.0])}
    )
    with pytest.warns(UserWarning, match=r"\['region_1'\] have no column"):
        write(poly, tmp_path / "mesh.ele")
    back = read(tmp_path / "mesh.ele")
    np.testing.assert_allclose(back.element_attrs["region"], [1.0])


def test_lone_element_attribute_takes_the_region_column(tmp_path: Path) -> None:
    """A mesh from another format names its one attribute its own way."""
    poly = _tet_mesh(element_attrs={"material": np.array([4.0])})
    with pytest.warns(UserWarning, match="reads back as 'region'"):
        write(poly, tmp_path / "mesh.ele")
    np.testing.assert_allclose(
        read(tmp_path / "mesh.ele").element_attrs["region"], [4.0]
    )


def test_first_of_several_unnamed_element_attrs_is_written(tmp_path: Path) -> None:
    poly = _tet_mesh(
        element_attrs={"zone": np.array([1.0]), "material": np.array([2.0])}
    )
    with pytest.warns(UserWarning) as record:
        write(poly, tmp_path / "mesh.ele")
    messages = " ".join(str(w.message) for w in record)
    assert "['zone'] have no column" in messages
    assert "reads back as 'region'" in messages
    # 'material' sorts first, so it is the one that keeps a column.
    np.testing.assert_allclose(
        read(tmp_path / "mesh.ele").element_attrs["region"], [2.0]
    )


def test_write_leaves_no_half_pair_on_failure(tmp_path: Path) -> None:
    poly = _tet_mesh()
    broken = type(poly)(
        vertices=poly.vertices[:3],
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError):
        write(broken, tmp_path / "mesh.ele")
    assert not (tmp_path / "mesh.node").exists()
    assert not (tmp_path / "mesh.ele").exists()


def test_too_many_vertex_attrs_are_capped_to_what_reading_admits(
    tmp_path: Path,
) -> None:
    """A written .node has to be one this codec reads back."""
    poly = _tet_mesh(
        vertex_attrs={f"attr_{k}": np.full(4, float(k)) for k in range(_MAX_ATTR + 5)}
    )
    with pytest.warns(UserWarning, match="past the .* cap"):
        write(poly, tmp_path / "wide.ele")
    back = read(tmp_path / "wide.ele")
    assert len(back.vertex_attrs) == _MAX_ATTR
    # The columns kept are the first by name order, not an arbitrary slice.
    np.testing.assert_allclose(back.vertex_attrs["attr_0"], np.zeros(4))
    np.testing.assert_allclose(
        back.vertex_attrs[f"attr_{_MAX_ATTR - 1}"], np.full(4, float(_MAX_ATTR - 1))
    )


def test_unknown_node_reference_names_the_offending_node(tmp_path: Path) -> None:
    """Not the span every reference covers — the one that is wrong."""
    ele = _pair(
        tmp_path,
        "4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n",
        "1 4 0\n1 1 2 3 99\n",
    )
    with pytest.raises(CodecError, match=r"references node 99, outside the 1\.\.4"):
        read(ele)


def test_unsearchable_directory_raises_codec_error(tmp_path: Path) -> None:
    """``Path.exists`` propagates EACCES; the codec's contract is CodecError."""
    if os.name != "posix" or os.geteuid() == 0:
        pytest.skip("needs POSIX permissions and a non-root user")
    sub = tmp_path / "locked"
    sub.mkdir()
    (sub / "m.node").write_text("0 3 0 0\n", encoding="utf-8")
    (sub / "m.ele").write_text("0 4 0\n", encoding="utf-8")
    sub.chmod(0o000)
    try:
        with pytest.raises(CodecError, match="cannot check for"):
            read(sub / "m.ele")
    finally:
        sub.chmod(0o755)


def test_mixed_case_half_is_found(tmp_path: Path) -> None:
    """A ``Mesh.Node`` pair is neither all-lower nor all-upper.

    Only bites on a case-sensitive filesystem; on macOS/Windows the lookup
    succeeds before the directory scan this exercises is ever reached.
    """
    poly = _tet_mesh()
    write(poly, tmp_path / "mesh.ele")
    (tmp_path / "mesh.node").rename(tmp_path / "Mesh.Node")
    (tmp_path / "mesh.ele").rename(tmp_path / "Mesh.Ele")
    np.testing.assert_allclose(read(tmp_path / "Mesh.Ele").vertices, poly.vertices)


def test_bad_region_attr_warns_and_is_dropped(tmp_path: Path) -> None:
    poly = _tet_mesh(element_attrs={"region": np.zeros((1, 2))})
    with pytest.warns(UserWarning, match="not one scalar per element"):
        write(poly, tmp_path / "mesh.ele")
    assert "region" not in read(tmp_path / "mesh.ele").element_attrs


def test_float_fmt_option(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0.1234567890123, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    write(poly, tmp_path / "wide.ele", float_fmt=".17g")
    np.testing.assert_array_equal(read(tmp_path / "wide.ele").vertices, poly.vertices)


def test_bad_float_fmt_raises(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="float_fmt"):
        write(_tet_mesh(), tmp_path / "mesh.ele", float_fmt="nonsense")


def test_float_fmt_that_writes_unreadable_tokens_raises(tmp_path: Path) -> None:
    """A grouping format is a legal specifier that writes an unreadable file."""
    with pytest.raises(CodecError, match="parse back"):
        write(_tet_mesh(), tmp_path / "mesh.ele", float_fmt=",.2f")
    assert not (tmp_path / "mesh.node").exists()


def test_exponent_float_fmt_is_accepted(tmp_path: Path) -> None:
    """Lossy is not unreadable; only a token the reader chokes on is refused."""
    write(_tet_mesh(), tmp_path / "mesh.ele", float_fmt=".2e")
    assert read(tmp_path / "mesh.ele").vertices.shape == (4, 3)


def test_unknown_option_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tet_mesh(), tmp_path / "mesh.ele", nope=1)


# ---------------------------------------------------------------------------
# Interoperability
# ---------------------------------------------------------------------------


def test_meshio_reads_what_we_write(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    poly = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3], [1, 4, 2, 3]]))])
    write(poly, tmp_path / "m.ele")
    mesh = meshio.read(tmp_path / "m.node", "tetgen")
    np.testing.assert_allclose(mesh.points, poly.vertices)
    assert len(mesh.cells) == 1 and mesh.cells[0].type == "tetra"
    np.testing.assert_array_equal(mesh.cells[0].data.ravel(), poly.connectivity)


def test_we_read_what_meshio_writes(tmp_path: Path) -> None:
    """meshio numbers both files from 0; ours must not shift them."""
    meshio = pytest.importorskip("meshio")
    poly = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3], [1, 4, 2, 3]]))])
    mesh = meshio.Mesh(
        poly.vertices, [("tetra", np.array([[0, 1, 2, 3], [1, 4, 2, 3]]))]
    )
    meshio.write(tmp_path / "mio.node", mesh, "tetgen")
    back = read(tmp_path / "mio.ele")
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_tetgen_numbered_stem(tmp_path: Path) -> None:
    """TetGen names its output ``mesh.1.node`` / ``mesh.1.ele``."""
    poly = _tet_mesh()
    write(poly, tmp_path / "mesh.1.ele")
    assert (tmp_path / "mesh.1.node").exists()
    np.testing.assert_allclose(read(tmp_path / "mesh.1.ele").vertices, poly.vertices)
