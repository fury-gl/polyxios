from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._gmsh import (
    _GMSH_TO_POLYXIOS,
    _READ_ORDER,
    _WRITE_ORDER,
    read,
    write,
)
from polyxios.exceptions import CodecError

_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _tet_mesh():
    return make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    return make_polydata(
        _TET_VERTS, [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3]]))]
    )


def _write_text(tmp_path: Path, name: str, text: str) -> Path:
    out = tmp_path / name
    out.write_text(text)
    return out


# --- element table -------------------------------------------------------


def test_gmsh_type_codes_match_the_msh_spec() -> None:
    # Guards against the mis-assignment of Gmsh's third/fourth-order codes
    # (21, 29) to the second-order elements, which live at 9 and 11.
    assert _GMSH_TO_POLYXIOS[9] == ("quadratic_triangle", 6)
    assert _GMSH_TO_POLYXIOS[11] == ("quadratic_tetra", 10)
    assert 21 not in _GMSH_TO_POLYXIOS
    assert 29 not in _GMSH_TO_POLYXIOS


def test_node_permutations_are_bijections() -> None:
    for name, order in _READ_ORDER.items():
        assert sorted(order) == list(range(len(order))), name
        inverse = _WRITE_ORDER[name]
        assert tuple(inverse[k] for k in order) == tuple(range(len(order)))


# meshio name for each polyxios type whose Gmsh node order needs permuting.
_MESHIO_ALIAS = {
    "quadratic_tetra": "tetra10",
    "quadratic_hexahedron": "hexahedron20",
    "triquadratic_hexahedron": "hexahedron27",
    "quadratic_wedge": "wedge15",
    "quadratic_pyramid": "pyramid13",
}
# Types read straight through; meshio must agree that no permutation applies.
_MESHIO_IDENTITY = {
    "line": 2,
    "line3": 3,
    "triangle": 3,
    "triangle6": 6,
    "quad": 4,
    "quad8": 8,
    "quad9": 9,
    "tetra": 4,
    "hexahedron": 8,
    "wedge": 6,
    "pyramid": 5,
    "vertex": 1,
}


def test_node_permutations_match_meshio() -> None:
    pytest.importorskip("meshio")
    from meshio.gmsh._gmsh22 import _gmsh_to_meshio_order

    for name, order in _READ_ORDER.items():
        alias = _MESHIO_ALIAS.get(name)
        if alias is None:
            continue
        idx = np.arange(len(order))
        ref = _gmsh_to_meshio_order(alias, idx[None, :])[0]
        np.testing.assert_array_equal(ref, order, err_msg=name)


def test_issue_1517_meshio_does_not_permute_the_18_node_prism() -> None:
    """meshio reads Gmsh's prism18 straight through; that is the bug, recorded.

    Kept so the disagreement is deliberate rather than discovered later: our
    table is derived from VTK's own edge and face lists, meshio's is identity.
    """
    pytest.importorskip("meshio")
    from meshio.gmsh._gmsh22 import _gmsh_to_meshio_order

    idx = np.arange(18)
    ref = _gmsh_to_meshio_order("wedge18", idx[None, :])[0]
    np.testing.assert_array_equal(ref, idx)
    assert _READ_ORDER["biquadratic_quadratic_wedge"] != tuple(idx)

    for meshio_name, n_nodes in _MESHIO_IDENTITY.items():
        idx = np.arange(n_nodes)
        ref = _gmsh_to_meshio_order(meshio_name, idx[None, :])[0]
        np.testing.assert_array_equal(ref, idx, err_msg=meshio_name)


def test_written_file_reads_in_meshio(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    tagged.element_attrs["phys_tag"] = np.array([2, 2, 2], dtype=np.int32)
    tagged.element_tags["skin"] = np.array([0, 1, 2], dtype=np.int32)
    out = tmp_path / "meshio.msh"
    write(tagged, out)

    mesh = meshio.read(out, "gmsh")
    np.testing.assert_allclose(mesh.points, poly.vertices)
    np.testing.assert_array_equal(
        mesh.get_cells_type("triangle"), poly.connectivity.reshape(3, 3)
    )
    assert "skin" in mesh.field_data


def test_reads_file_written_by_meshio_v22(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    mesh = meshio.Mesh(
        pts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 4]]))],
        cell_data={"gmsh:physical": [np.array([11]), np.array([22])]},
        field_data={"body": np.array([11, 3]), "wall": np.array([22, 2])},
    )
    out = tmp_path / "from_meshio.msh"
    meshio.write(out, mesh, file_format="gmsh22", binary=False)

    poly = read(out)
    np.testing.assert_allclose(poly.vertices, pts)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3, 0, 1, 4])
    np.testing.assert_array_equal(poly.element_attrs["phys_tag"], [11, 22])
    np.testing.assert_array_equal(poly.element_tags["body"], [0])
    np.testing.assert_array_equal(poly.element_tags["wall"], [1])


def test_reads_file_written_by_meshio_v41(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    mesh = meshio.Mesh(pts, [("tetra", np.array([[0, 1, 2, 3]]))])
    out = tmp_path / "from_meshio41.msh"
    meshio.write(out, mesh, file_format="gmsh", binary=False)
    assert "4.1" in out.read_text()

    poly = read(out)
    np.testing.assert_allclose(poly.vertices, pts)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_v41_physical_names_without_entities_warns(tmp_path: Path) -> None:
    # meshio writes exactly this shape: names declared, no $Entities to bind
    # them to, so the physical tags are genuinely absent from the file.
    text = _V41_TET.replace(
        "$Entities\n0 0 0 1\n1 0 0 0 1 1 1 1 12 0\n$EndEntities\n",
        '$PhysicalNames\n1\n3 12 "body"\n$EndPhysicalNames\n',
    )
    with pytest.warns(UserWarning, match=r"\$Entities is missing"):
        poly = read(_write_text(tmp_path, "noent.msh", text))
    np.testing.assert_array_equal(poly.element_attrs["phys_tag"], [0])
    assert poly.element_tags == {}


# --- roundtrips ----------------------------------------------------------


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.msh"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tri_mesh()
    out = tmp_path / "tri.msh"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 3
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_mixed_elements(tmp_path: Path) -> None:
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("line", np.array([[0, 1]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("quad", np.array([[0, 1, 4, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    out = tmp_path / "mixed.msh"
    write(poly, out)
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


@pytest.mark.parametrize("name", sorted(_READ_ORDER))
def test_roundtrip_permuted_element_types(tmp_path: Path, name: str) -> None:
    # Every higher-order element is permuted on the way out and back; a wrong
    # inverse would surface as reordered connectivity.
    n_nodes = len(_READ_ORDER[name])
    verts = np.arange(3 * n_nodes, dtype=np.float64).reshape(n_nodes, 3)
    poly = make_polydata(verts, [(name, np.arange(n_nodes).reshape(1, n_nodes))])
    out = tmp_path / f"{name}.msh"
    write(poly, out)
    poly2 = read(out)
    assert int(poly2.element_types[0]) == ELEMENT_TYPES[name]
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_quadratic_tetra_read_swaps_last_two_midedge_nodes(tmp_path: Path) -> None:
    # Gmsh orders tet10 mid-edge nodes (0,1)(1,2)(0,2)(0,3)(2,3)(1,3); VTK swaps
    # the final pair. Node tags here are 1..10 so the mapping is readable.
    verts = "\n".join(f"{i + 1} {i} 0 0" for i in range(10))
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f"$Nodes\n10\n{verts}\n$EndNodes\n"
        "$Elements\n1\n1 11 2 0 0 1 2 3 4 5 6 7 8 9 10\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "tet10.msh", text))
    assert int(poly.element_types[0]) == ELEMENT_TYPES["quadratic_tetra"]
    np.testing.assert_array_equal(
        poly.connectivity, np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])
    )


# --- physical groups -----------------------------------------------------


def test_phys_tag_read_from_file(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 7 1 1 2 3\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "phys.msh", text))
    np.testing.assert_array_equal(poly.element_attrs["phys_tag"], [7])


def test_phys_tag_survives_roundtrip(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    tagged.element_attrs["phys_tag"] = np.array([3, 3, 8], dtype=np.int32)
    out = tmp_path / "tagged.msh"
    write(tagged, out)
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_attrs["phys_tag"], [3, 3, 8])


def test_physical_names_become_element_tags(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n1\n2 5 "outer wall"\n$EndPhysicalNames\n'
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n2\n1 2 2 5 1 1 2 3\n2 2 2 6 1 1 2 4\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "names.msh", text))
    np.testing.assert_array_equal(poly.element_tags["outer wall"], [0])


def test_physical_names_roundtrip(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    tagged.element_attrs["phys_tag"] = np.array([4, 4, 9], dtype=np.int32)
    tagged.element_tags["skin"] = np.array([0, 1], dtype=np.int32)
    out = tmp_path / "named.msh"
    write(tagged, out)
    assert '2 4 "skin"' in out.read_text()
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["skin"], [0, 1])


def test_element_tags_without_phys_tag_get_generated_ids(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    tagged.element_tags["inlet"] = np.array([0], dtype=np.int32)
    tagged.element_tags["outlet"] = np.array([2], dtype=np.int32)
    out = tmp_path / "gen.msh"
    write(tagged, out)
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["inlet"], [0])
    np.testing.assert_array_equal(poly2.element_tags["outlet"], [2])


def test_physical_names_without_a_count_header_are_kept(tmp_path: Path) -> None:
    # The record count is what Gmsh writes, but a file that omits it must not
    # lose its first name.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n2 1 "surf"\n$EndPhysicalNames\n'
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 1 1 1 2 3\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "nocount.msh", text))
    np.testing.assert_array_equal(poly.element_tags["surf"], [0])


def test_phys_tag_of_the_wrong_length_warns(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    tagged.element_attrs["phys_tag"] = np.array([5, 5], dtype=np.int32)
    tagged.element_tags["skin"] = np.array([0, 1, 2], dtype=np.int32)
    out = tmp_path / "shorttag.msh"
    with pytest.warns(UserWarning, match="phys_tag"):
        write(tagged, out)
    # The stored tags are unusable, so the group is renumbered from scratch.
    np.testing.assert_array_equal(read(out).element_tags["skin"], [0, 1, 2])


def test_group_without_its_own_tag_warns(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    # "inner" is a subset of "outer", so both resolve to the same physical tag
    # and only the first can be named.
    tagged.element_tags["outer"] = np.array([0, 1], dtype=np.int32)
    tagged.element_tags["inner"] = np.array([1], dtype=np.int32)
    out = tmp_path / "subset.msh"
    with pytest.warns(UserWarning, match="'inner'"):
        write(tagged, out)
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["outer"], [0, 1])
    assert "inner" not in poly2.element_tags


def test_physical_names_sharing_a_tag_split_by_dimension(tmp_path: Path) -> None:
    # Gmsh only makes a physical tag unique within one dimension, so tag 1 can
    # name both a surface and a volume.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n2\n2 1 "wall"\n3 1 "body"\n$EndPhysicalNames\n'
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n2\n1 2 2 1 1 1 2 3\n2 4 2 1 1 1 2 3 4\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "dims.msh", text))
    np.testing.assert_array_equal(poly.element_tags["wall"], [0])
    np.testing.assert_array_equal(poly.element_tags["body"], [1])


def test_physical_name_keeps_members_of_other_dimensions(tmp_path: Path) -> None:
    # No tag collision, so the declared dimension does not filter: a group
    # written as 3D still keeps its boundary triangles.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n1\n3 1 "region"\n$EndPhysicalNames\n'
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n2\n1 2 2 1 1 1 2 3\n2 4 2 1 1 1 2 3 4\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "mixeddim.msh", text))
    np.testing.assert_array_equal(poly.element_tags["region"], [0, 1])


def test_groups_sharing_a_tag_across_dimensions_roundtrip(tmp_path: Path) -> None:
    # A physical tag is scoped to a dimension, so a surface and a volume group
    # may both be tag 1. Dropping one of them on write would leave the survivor
    # claiming the other's elements when the file is read back.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n2\n2 1 "wall"\n3 1 "body"\n$EndPhysicalNames\n'
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n2\n1 2 2 1 1 1 2 3\n2 4 2 1 1 1 2 3 4\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "shared.msh", text))
    out = tmp_path / "shared_out.msh"
    write(poly, out)
    written = out.read_text()
    assert '2 1 "wall"' in written
    assert '3 1 "body"' in written
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["wall"], [0])
    np.testing.assert_array_equal(poly2.element_tags["body"], [1])


def test_group_spanning_dimensions_keeps_its_tag_to_itself(tmp_path: Path) -> None:
    # A reader splits a shared tag by dimension, so a group whose members span
    # two of them cannot let a second name reuse its tag without being torn up.
    poly = make_polydata(
        _TET_VERTS,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    poly.element_tags["both"] = np.array([0, 1], dtype=np.int32)
    poly.element_tags["surface"] = np.array([0], dtype=np.int32)
    out = tmp_path / "spanning.msh"
    with pytest.warns(UserWarning, match="'surface'"):
        write(poly, out)
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["both"], [0, 1])
    assert "surface" not in poly2.element_tags


def test_physical_name_with_tag_zero_is_ignored(tmp_path: Path) -> None:
    # Zero is the tag every untagged element carries, so a group claiming it
    # would sweep up the whole remainder of the mesh.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n1\n2 0 "everything"\n$EndPhysicalNames\n'
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 0 1 2 3\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "tag0.msh", text))
    assert poly.element_tags == {}


def test_quote_in_group_name_is_sanitised(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tagged = make_polydata(
        poly.vertices, [("triangle", poly.connectivity.reshape(3, 3))]
    )
    tagged.element_attrs["phys_tag"] = np.array([2, 2, 2], dtype=np.int32)
    tagged.element_tags['say "hi"'] = np.array([0, 1, 2], dtype=np.int32)
    out = tmp_path / "quoted.msh"
    with pytest.warns(UserWarning, match="cannot be quoted"):
        write(tagged, out)
    assert """2 2 "say 'hi'\"""" in out.read_text()
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["say 'hi'"], [0, 1, 2])


# --- format 4.1 ----------------------------------------------------------


_V41_TET = """$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 0 1
1 0 0 0 1 1 1 1 12 0
$EndEntities
$Nodes
1 4 1 4
3 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
0 0 1
$EndNodes
$Elements
1 1 1 1
3 1 4 1
1 1 2 3 4
$EndElements
"""


def test_read_v41(tmp_path: Path) -> None:
    poly = read(_write_text(tmp_path, "v41.msh", _V41_TET))
    assert int(poly.element_types[0]) == ELEMENT_TYPES["tetra"]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])
    np.testing.assert_allclose(poly.vertices, _TET_VERTS)


def test_v41_physical_tag_from_entities(tmp_path: Path) -> None:
    poly = read(_write_text(tmp_path, "v41.msh", _V41_TET))
    np.testing.assert_array_equal(poly.element_attrs["phys_tag"], [12])


def test_v41_non_contiguous_node_tags(tmp_path: Path) -> None:
    text = (
        _V41_TET.replace("1 4 1 4", "1 4 10 40")
        .replace("3 1 0 4\n1\n2\n3\n4", "3 1 0 4\n10\n20\n30\n40")
        .replace("1 1 2 3 4", "1 10 20 30 40")
    )
    poly = read(_write_text(tmp_path, "sparse.msh", text))
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


# --- error handling ------------------------------------------------------


def test_missing_section_raises(tmp_path: Path) -> None:
    text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
    with pytest.raises(CodecError, match=r"\$Nodes"):
        read(_write_text(tmp_path, "bad.msh", text))


def test_binary_flag_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 1 8\n$EndMeshFormat\n"
        "$Nodes\n0\n$EndNodes\n$Elements\n0\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="binary"):
        read(_write_text(tmp_path, "bin.msh", text))


def test_unsupported_version_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n4.0 0 8\n$EndMeshFormat\n"
        "$Nodes\n0\n$EndNodes\n$Elements\n0\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="unsupported format version"):
        read(_write_text(tmp_path, "v40.msh", text))


def test_truncated_nodes_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n5\n1 0 0 0\n2 1 0 0\n$EndNodes\n"
        "$Elements\n0\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="only 2 follow"):
        read(_write_text(tmp_path, "trunc.msh", text))


def test_absurd_node_count_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n999999999999\n1 0 0 0\n$EndNodes\n"
        "$Elements\n0\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="safety cap"):
        read(_write_text(tmp_path, "huge.msh", text))


def test_undefined_node_tag_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n10 0 0 0\n20 1 0 0\n30 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 0 0 10 20 99\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="undefined node tag"):
        read(_write_text(tmp_path, "dangling.msh", text))


def test_out_of_range_node_tag_raises(tmp_path: Path) -> None:
    # Node tags are 1..n here, so the tag is its own row index and an oversized
    # tag would silently address a vertex that does not exist.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 0 0 1 2 99\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="undefined node tag 99"):
        read(_write_text(tmp_path, "oob.msh", text))


def test_zero_node_tag_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 0 0 1 2 0\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="undefined node tag 0"):
        read(_write_text(tmp_path, "zero.msh", text))


def test_element_record_shorter_than_its_tag_count_raises(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="too short"):
        read(_write_text(tmp_path, "shorttags.msh", text))


def test_element_record_longer_than_its_type_raises(tmp_path: Path) -> None:
    # A triangle record carrying four node fields is misaligned; reading the
    # first three would build an element out of the wrong nodes.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n1\n1 2 2 1 1 1 2 3 4\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="needs 3 nodes"):
        read(_write_text(tmp_path, "longrecord.msh", text))


_OVERFLOW = "99999999999999999999999"


@pytest.mark.parametrize(
    ("name", "text"),
    [
        (
            "v2 node tag",
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            f"$Nodes\n2\n{_OVERFLOW} 0 0 0\n7 1 0 0\n$EndNodes\n"
            "$Elements\n0\n$EndElements\n",
        ),
        (
            "v2 element node tag",
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
            f"$Elements\n1\n1 2 2 1 1 1 2 {_OVERFLOW}\n$EndElements\n",
        ),
        (
            "v2 physical tag",
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
            f"$Elements\n1\n1 2 2 {_OVERFLOW} 1 1 2 3\n$EndElements\n",
        ),
        (
            "v41 node tag",
            "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
            f"$Nodes\n1 1 1 1\n0 1 0 1\n{_OVERFLOW}\n0 0 0\n$EndNodes\n"
            "$Elements\n0 0 1 1\n$EndElements\n",
        ),
    ],
)
def test_out_of_range_integers_raise(tmp_path: Path, name: str, text: str) -> None:
    # A value past the width of the array it lands in must surface as a
    # CodecError, not as the OverflowError numpy raises.
    with pytest.raises(CodecError):
        read(_write_text(tmp_path, "overflow.msh", text))


def test_entities_with_an_infinite_field_drop_physical_tags(tmp_path: Path) -> None:
    text = _V41_TET.replace(
        "$Entities\n0 0 0 1\n1 0 0 0 1 1 1 1 12 0",
        "$Entities\n0 0 0 1\n1e400 0 0 0 1 1 1 1 12 0",
    )
    with pytest.warns(UserWarning, match="malformed .Entities record"):
        poly = read(_write_text(tmp_path, "infent.msh", text))
    np.testing.assert_array_equal(poly.element_attrs["phys_tag"], [0])


def test_large_node_tags_keep_full_precision(tmp_path: Path) -> None:
    # Above 2**53 a float64 no longer holds every integer, so the bulk parse
    # cannot be trusted for the tags.
    big = 2**53 + 1
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f"$Nodes\n2\n{big} 0 0 0\n{big + 2} 1 0 0\n$EndNodes\n"
        f"$Elements\n1\n1 1 2 0 0 {big + 2} {big}\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "bigtags.msh", text))
    np.testing.assert_array_equal(poly.connectivity, [1, 0])


def test_node_record_with_a_missing_field_is_rejected(tmp_path: Path) -> None:
    # The two records hold 3 and 5 fields, which totals the 8 numbers a bulk
    # parse expects; without an alignment check the coordinates would be read
    # from the wrong columns and no error would be raised.
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n2\n1 0 0\n2 1 0 0 0\n$EndNodes\n"
        "$Elements\n0\n$EndElements\n"
    )
    with pytest.raises(CodecError, match="malformed node record"):
        read(_write_text(tmp_path, "ragged.msh", text))


def test_non_contiguous_node_tags_keep_their_coordinates(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n10 0 0 0\n20 1 0 0\n30 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 0 0 10 20 30\n$EndElements\n"
    )
    poly = read(_write_text(tmp_path, "sparse2.msh", text))
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_duplicate_node_tags_warn(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n1 1 0 0\n3 2 0 0\n$EndNodes\n"
        "$Elements\n1\n1 1 2 0 0 1 3\n$EndElements\n"
    )
    with pytest.warns(UserWarning, match="duplicate node tag"):
        poly = read(_write_text(tmp_path, "dupnodes.msh", text))
    # The last definition of tag 1 wins.
    np.testing.assert_array_equal(poly.connectivity, [1, 2])


def test_wrong_node_count_raises_on_write(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="expected 3"):
        write(poly, tmp_path / "badcount.msh")


def test_v41_element_count_mismatch_warns(tmp_path: Path) -> None:
    text = _V41_TET.replace("$Elements\n1 1 1 1", "$Elements\n1 2 1 1")
    with pytest.warns(UserWarning, match="blocks hold 1"):
        poly = read(_write_text(tmp_path, "count.msh", text))
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_unsupported_element_type_warns_and_skips(tmp_path: Path) -> None:
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n2\n1 2 2 0 0 1 2 3\n"
        "2 21 2 0 0 1 2 3 1 2 3 1 2 3 1\n$EndElements\n"
    )
    with pytest.warns(UserWarning, match=r"unsupported Gmsh type"):
        poly = read(_write_text(tmp_path, "skip.msh", text))
    assert len(poly.element_types) == 1


def test_lazy_warns(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.msh"
    write(poly, out)
    with pytest.warns(UserWarning, match="lazy=True ignored"):
        read(out, lazy=True)


def test_unwritable_element_warns(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("polygon", [np.array([0, 1, 3, 2])])])
    with pytest.warns(UserWarning, match="no Gmsh equivalent"):
        write(poly, tmp_path / "poly.msh")


# --- output shape --------------------------------------------------------


def test_file_has_sections(tmp_path: Path) -> None:
    out = tmp_path / "tet.msh"
    write(_tet_mesh(), out)
    text = out.read_text()
    assert "$MeshFormat" in text
    assert "$EndNodes" in text
    assert "$EndElements" in text


def test_float_fmt_option(tmp_path: Path) -> None:
    verts = np.array([[1 / 3, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    out = tmp_path / "fmt.msh"
    write(poly, out, float_fmt=".3f")
    assert "1 0.333 0.000 0.000" in out.read_text()
    np.testing.assert_allclose(read(out).vertices[0, 0], 0.333)


def test_roundtrip_mesh_without_elements(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [])
    out = tmp_path / "empty.msh"
    write(poly, out)
    poly2 = read(out)
    np.testing.assert_allclose(poly2.vertices, verts)
    assert len(poly2.element_types) == 0
    assert poly2.element_attrs == {}


def test_coordinates_roundtrip_exactly(tmp_path: Path) -> None:
    verts = np.array(
        [[0.1, 1 / 3, 1e-17], [np.pi, -2 / 7, 1234567.8901234567], [0, 0, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    out = tmp_path / "prec.msh"
    write(poly, out)
    np.testing.assert_array_equal(read(out).vertices, verts)


# --- meshio #1517: 18-node prism node ordering -------------------------------

# Gmsh numbers a prism18's mid-nodes by its own edge and face tables, VTK by
# its own; the two disagree, so the file's node 7 is not VTK's node 7. The
# corners of a straight prism, and the mid-node each numbering expects.
_P18_CORNERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
_GMSH_P18_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
]
_GMSH_P18_FACES = [(0, 1, 4, 3), (0, 3, 5, 2), (1, 2, 5, 4)]
_VTK_P18_EDGES = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 4),
    (2, 5),
]
_VTK_P18_FACES = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]


def _p18_points() -> np.ndarray:
    """Return the 18 node positions a straight prism has, in Gmsh order."""
    mids = [_P18_CORNERS[list(e)].mean(axis=0) for e in _GMSH_P18_EDGES]
    centres = [_P18_CORNERS[list(f)].mean(axis=0) for f in _GMSH_P18_FACES]
    return np.vstack([_P18_CORNERS, np.array(mids), np.array(centres)])


def _p18_msh(points: np.ndarray) -> str:
    nodes = "\n".join(
        f"{i + 1} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for i, p in enumerate(points)
    )
    refs = " ".join(str(i + 1) for i in range(18))
    return (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f"$Nodes\n18\n{nodes}\n$EndNodes\n"
        f"$Elements\n1\n1 13 2 1 1 {refs}\n$EndElements\n"
    )


def test_issue_1517_prism18_is_permuted_into_vtk_order(tmp_path: Path) -> None:
    """Gmsh's edge and face tables are not VTK's; identity bends the element."""
    points = _p18_points()
    path = _write_text(tmp_path, "p18.msh", _p18_msh(points))
    poly = read(path)

    assert list(poly.element_types) == [ELEMENT_TYPES["biquadratic_quadratic_wedge"]]
    cell = poly.connectivity[: int(poly.offsets[1])]
    assert len(cell) == 18

    np.testing.assert_allclose(poly.vertices[cell[:6]], _P18_CORNERS)
    for slot, edge in enumerate(_VTK_P18_EDGES, start=6):
        np.testing.assert_allclose(
            poly.vertices[cell[slot]],
            _P18_CORNERS[list(edge)].mean(axis=0),
            err_msg=f"VTK slot {slot} must hold the midpoint of edge {edge}",
        )
    for slot, face in enumerate(_VTK_P18_FACES, start=15):
        np.testing.assert_allclose(
            poly.vertices[cell[slot]],
            _P18_CORNERS[list(face)].mean(axis=0),
            err_msg=f"VTK slot {slot} must hold the centre of face {face}",
        )


def test_issue_1517_prism18_survives_a_round_trip(tmp_path: Path) -> None:
    """A permutation applied on read and not on write is the same bug, mirrored."""
    points = _p18_points()
    poly = read(_write_text(tmp_path, "p18.msh", _p18_msh(points)))
    out = tmp_path / "back.msh"
    write(poly, out)
    back = read(out)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_allclose(
        back.vertices[back.connectivity], poly.vertices[poly.connectivity]
    )


def test_pyramid14_is_still_skipped_with_a_warning(tmp_path: Path) -> None:
    """VTK has no 14-node pyramid, so there is nowhere correct to put one."""
    nodes = "\n".join(f"{i + 1} {i}.0 0.0 0.0" for i in range(14))
    refs = " ".join(str(i + 1) for i in range(14))
    path = _write_text(
        tmp_path,
        "p14.msh",
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f"$Nodes\n14\n{nodes}\n$EndNodes\n"
        f"$Elements\n1\n1 14 2 1 1 {refs}\n$EndElements\n",
    )
    with pytest.warns(UserWarning, match="unsupported Gmsh type"):
        poly = read(path)
    assert len(poly.element_types) == 0


# --- meshio #1281: $NodeData / $ElementData ----------------------------------


_TET_HEADER = (
    "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
    "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
    "$Elements\n2\n1 4 2 1 1 1 2 3 4\n2 2 2 1 1 1 2 3\n$EndElements\n"
)


def test_issue_1281_node_data_becomes_a_vertex_attr(tmp_path: Path) -> None:
    """A solution field dropped on read makes the reader useless for results."""
    path = _write_text(
        tmp_path,
        "nodedata.msh",
        _TET_HEADER + '$NodeData\n1\n"T"\n1\n0.0\n3\n0\n1\n4\n'
        "1 10.0\n2 20.0\n3 30.0\n4 40.0\n$EndNodeData\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertex_attrs["T"], [10.0, 20.0, 30.0, 40.0])


def test_issue_1281_node_data_of_any_width_is_kept(tmp_path: Path) -> None:
    """The spec allows any component count, not only 1, 3 and 9."""
    rows = "\n".join(
        f"{i + 1} " + " ".join(str(i * 5 + k) for k in range(5)) for i in range(4)
    )
    path = _write_text(
        tmp_path,
        "nodedata5.msh",
        _TET_HEADER
        + '$NodeData\n1\n"state"\n1\n0.0\n3\n0\n5\n4\n'
        + rows
        + "\n$EndNodeData\n",
    )
    poly = read(path)
    assert poly.vertex_attrs["state"].shape == (4, 5)
    np.testing.assert_allclose(poly.vertex_attrs["state"][2], [10, 11, 12, 13, 14])


def test_issue_1281_element_data_becomes_an_element_attr(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path,
        "ed.msh",
        _TET_HEADER + '$ElementData\n1\n"rho"\n1\n0.0\n3\n0\n1\n2\n'
        "1 1.5\n2 2.5\n$EndElementData\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.element_attrs["rho"], [1.5, 2.5])


def test_issue_1281_several_data_fields_all_arrive(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path,
        "two.msh",
        _TET_HEADER
        + '$NodeData\n1\n"a"\n1\n0.0\n3\n0\n1\n4\n1 1\n2 2\n3 3\n4 4\n$EndNodeData\n'
        '$NodeData\n1\n"b"\n1\n0.0\n3\n0\n1\n4\n1 9\n2 8\n3 7\n4 6\n$EndNodeData\n',
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertex_attrs["a"], [1, 2, 3, 4])
    np.testing.assert_allclose(poly.vertex_attrs["b"], [9, 8, 7, 6])


def test_issue_1281_partial_node_data_is_filled_with_nan(tmp_path: Path) -> None:
    """A field naming half the nodes must not shift onto the wrong ones."""
    path = _write_text(
        tmp_path,
        "part.msh",
        _TET_HEADER + '$NodeData\n1\n"T"\n1\n0.0\n3\n0\n1\n2\n'
        "2 20.0\n4 40.0\n$EndNodeData\n",
    )
    poly = read(path)
    values = poly.vertex_attrs["T"]
    np.testing.assert_allclose(values[[1, 3]], [20.0, 40.0])
    assert np.isnan(values[[0, 2]]).all()


def test_issue_1281_data_naming_an_unknown_node_warns(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path,
        "ghost.msh",
        _TET_HEADER + '$NodeData\n1\n"T"\n1\n0.0\n3\n0\n1\n1\n99 1.0\n$EndNodeData\n',
    )
    with pytest.warns(UserWarning, match="99|unknown"):
        poly = read(path)
    assert "T" not in poly.vertex_attrs


def test_issue_1281_a_data_field_named_like_phys_tag_is_renamed(
    tmp_path: Path,
) -> None:
    """phys_tag is this codec's own element attribute; a clash would hide it."""
    path = _write_text(
        tmp_path,
        "clash.msh",
        _TET_HEADER + '$ElementData\n1\n"phys_tag"\n1\n0.0\n3\n0\n1\n2\n'
        "1 5.0\n2 6.0\n$EndElementData\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_attrs["phys_tag"], [1, 1])
    np.testing.assert_allclose(poly.element_attrs["phys_tag_2"], [5.0, 6.0])


def test_element_data_survives_a_row_naming_a_skipped_element(
    tmp_path: Path,
) -> None:
    """One element of a type with no VTK home must not cost the whole field."""
    path = _write_text(
        tmp_path,
        "partial.msh",
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        # Type 14 is the 14-node pyramid, which has no VTK equivalent.
        "$Elements\n2\n1 2 2 1 1 1 2 3\n2 14 2 1 1 1 2 3 4\n$EndElements\n"
        '$ElementData\n1\n"heat"\n1\n0.0\n3\n0\n1\n2\n'
        "1 5.0\n2 6.0\n$EndElementData\n",
    )
    with pytest.warns(UserWarning):
        poly = read(path)
    np.testing.assert_allclose(poly.element_attrs["heat"], [5.0])


def test_an_unreadable_element_tag_does_not_sink_the_read(tmp_path: Path) -> None:
    """A tag names a data row and nothing else; the mesh is still worth having."""
    path = _write_text(
        tmp_path,
        "oddtag.msh",
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n1\ne1 4 2 1 1 1 2 3 4\n$EndElements\n",
    )
    poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["tetra"]]


def test_malformed_data_section_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path,
        "bad.msh",
        _TET_HEADER
        + '$NodeData\n1\n"T"\n1\n0.0\n3\n0\n1\n2\n1 nope\n2 3.0\n$EndNodeData\n',
    )
    with pytest.warns(UserWarning, match="NodeData"):
        poly = read(path)
    assert "T" not in poly.vertex_attrs


# --- meshio #1421, #1404, #865, #524, #1116 ----------------------------------


def test_issue_1421_single_line_element_mesh_round_trips(tmp_path: Path) -> None:
    """A one-line mesh is the smallest thing Gmsh can hold; it must re-read."""
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    out = tmp_path / "line.msh"
    write(poly, out)
    back = read(out)
    assert list(back.element_types) == [ELEMENT_TYPES["line"]]
    np.testing.assert_allclose(back.vertices, verts)
    np.testing.assert_array_equal(back.connectivity, [0, 1])

    meshio = pytest.importorskip("meshio")
    mesh = meshio.read(out)
    np.testing.assert_array_equal(mesh.cells[0].data, [[0, 1]])


def test_issue_1404_reading_prints_nothing(tmp_path: Path, capsys) -> None:
    """A library that prints on read corrupts whatever the caller pipes it to."""
    path = _write_text(tmp_path, "quiet.msh", _TET_HEADER)
    read(path)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_issue_865_mixed_cell_types_survive_a_write(tmp_path: Path) -> None:
    """CSR holds mixed types natively, so no block splitting can drop one."""
    verts = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("line", np.array([[0, 1]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("vertex", np.array([[4]])),
        ],
    )
    out = tmp_path / "mixed.msh"
    write(poly, out)
    back = read(out)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)


def test_issue_1116_a_flat_mesh_keeps_its_zero_z(tmp_path: Path) -> None:
    """A 2-D mesh is 3-D with z=0; dropping the column changes the geometry."""
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    out = tmp_path / "flat.msh"
    write(poly, out)
    back = read(out)
    assert back.vertices.shape == (3, 3)
    np.testing.assert_allclose(back.vertices[:, 2], 0.0)


def test_issue_1281_vertex_attrs_are_written_as_node_data(tmp_path: Path) -> None:
    """A field read into vertex_attrs and dropped on write is half a codec."""
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["tetra"]], dtype=np.uint8),
        vertex_attrs={
            "T": np.arange(4, dtype=np.float64),
            "v": np.arange(12, dtype=np.float64).reshape(4, 3),
        },
    )
    out = tmp_path / "nodedata.msh"
    write(poly, out)
    back = read(out)
    np.testing.assert_allclose(back.vertex_attrs["T"], [0, 1, 2, 3])
    np.testing.assert_allclose(back.vertex_attrs["v"], poly.vertex_attrs["v"])

    meshio = pytest.importorskip("meshio")
    mesh = meshio.read(out)
    np.testing.assert_allclose(mesh.point_data["T"], [0, 1, 2, 3])


def test_issue_1281_element_attrs_are_written_as_element_data(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 0, 1, 3], dtype=np.int32),
        offsets=np.array([0, 3, 6], dtype=np.int32),
        element_types=np.full(2, ELEMENT_TYPES["triangle"], dtype=np.uint8),
        element_attrs={"rho": np.array([1.5, 2.5])},
    )
    out = tmp_path / "ed.msh"
    write(poly, out)
    np.testing.assert_allclose(read(out).element_attrs["rho"], [1.5, 2.5])


def test_element_data_follows_the_elements_that_were_written(
    tmp_path: Path,
) -> None:
    """A skipped element shifts the numbering; the field has to shift with it."""
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 3, 7], dtype=np.int32),
        element_types=np.array(
            [ELEMENT_TYPES["polyhedron"], ELEMENT_TYPES["tetra"]], dtype=np.uint8
        ),
        element_attrs={"rho": np.array([9.0, 2.5])},
    )
    out = tmp_path / "skip.msh"
    with pytest.warns(UserWarning, match="no Gmsh equivalent"):
        write(poly, out)
    back = read(out)
    assert len(back.element_types) == 1
    np.testing.assert_allclose(back.element_attrs["rho"], [2.5])


def test_attrs_that_have_no_data_section_are_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = PolyData(
        vertices=_TET_VERTS,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["tetra"]], dtype=np.uint8),
        vertex_attrs={"labels": np.array(["a", "b", "c", "d"])},
    )
    with pytest.warns(UserWarning, match="labels"):
        write(poly, tmp_path / "bad.msh")


def test_a_field_covering_part_of_the_mesh_writes_no_missing_value(
    tmp_path: Path,
) -> None:
    """A partial field reads as NaN, and NaN is not a number the format has.

    Written straight through, the elements the field says nothing about would
    each spell the token 'nan' under a count that claims a value for every
    one of them. The rows are left out and the count drops with them, so the
    field goes back out covering the part it came in covering.
    """
    path = _write_text(
        tmp_path,
        "partial.msh",
        _TET_HEADER + '$ElementData\n1\n"rho"\n1\n0.0\n3\n0\n1\n1\n'
        "1 3.5\n$EndElementData\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(np.isnan(poly.element_attrs["rho"]), [False, True])

    out = tmp_path / "partial_out.msh"
    with pytest.warns(UserWarning, match="spells no missing value"):
        write(poly, out)
    text = out.read_text()
    assert "nan" not in text
    assert '$ElementData\n1\n"rho"\n1\n0.0\n3\n0\n1\n1\n1 3.5' in text

    back = read(out)
    np.testing.assert_array_equal(np.isnan(back.element_attrs["rho"]), [False, True])
    np.testing.assert_allclose(back.element_attrs["rho"][0], 3.5)


def test_a_tag_group_indexing_no_element_of_this_mesh_is_dropped(
    tmp_path: Path,
) -> None:
    """A stale index reaches an element that is not the one it named.

    Nothing checks a tag group on the way in, so one carried over from
    another mesh may run past the end of this one. Indexing with it raises
    whatever the stray value happens to raise; dropping it is what lets the
    loss be reported.
    """
    poly = make_polydata(_TET_VERTS, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])
    poly.element_tags["stale"] = np.array([0, 99], dtype=np.int32)
    path = tmp_path / "stale.msh"
    with pytest.warns(UserWarning, match="index no element"):
        write(poly, path)
    assert len(read(path).element_types) == 2


def test_a_data_block_the_file_ends_inside_is_still_read(tmp_path: Path) -> None:
    """The mesh sections and the data blocks come off one walk of the file."""
    path = _write_text(
        tmp_path,
        "unterminated.msh",
        _TET_HEADER + '$ElementData\n1\n"rho"\n1\n0.0\n3\n0\n1\n1\n1 4.5\n',
    )
    rho = read(path).element_attrs["rho"]
    assert rho[0] == 4.5
    assert np.isnan(rho[1])


def test_a_data_field_is_declared_at_a_width_gmsh_loads(tmp_path: Path) -> None:
    """MSH2 declares 1, 3 or 9 components; Gmsh refuses any other count."""
    poly = make_polydata(
        np.arange(12, dtype=np.float64).reshape(4, 3),
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    poly.vertex_attrs["uv"] = np.arange(8, dtype=np.float64).reshape(4, 2)
    path = tmp_path / "uv.msh"
    with pytest.warns(UserWarning, match="padded out"):
        write(poly, path)
    body = path.read_text().split("$NodeData")[1].splitlines()
    # the rest of the keyword line, the string and real tags, then the
    # integer tags: time step, component count, entity count
    assert body[7] == "3"
    back = read(path)
    np.testing.assert_allclose(back.vertex_attrs["uv"][:, :2], poly.vertex_attrs["uv"])


def test_a_field_already_a_legal_width_is_left_alone(tmp_path: Path) -> None:
    poly = make_polydata(
        np.arange(12, dtype=np.float64).reshape(4, 3),
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    poly.vertex_attrs["v"] = np.arange(12, dtype=np.float64).reshape(4, 3)
    path = tmp_path / "v.msh"
    write(poly, path)
    back = read(path)
    np.testing.assert_allclose(back.vertex_attrs["v"], poly.vertex_attrs["v"])
