from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._medit import read, sniff, write
from polyxios.exceptions import CodecError

_TET = """MeshVersionFormatted 2

Dimension
3

Vertices
4
0. 0. 0. 1
1. 0. 0. 1
0. 1. 0. 2
0. 0. 1. 2

Tetrahedra
1
1 2 3 4 7

End
"""

_MFEM_TET = """MFEM mesh v1.0

dimension
3

elements
1
1 4 0 1 2 3

boundary
0

vertices
4
3
0 0 0
1 0 0
0 1 0
0 0 1
"""


def _write_mesh(tmp_path: Path, text: str, name: str = "m.mesh") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _tet_mesh() -> PolyData:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


# --- sharing '.mesh' with MFEM ----------------------------------------------


def test_sniff_recognises_medit_and_not_mfem() -> None:
    assert sniff(_TET.encode())
    assert not sniff(_MFEM_TET.encode())


def test_a_medit_file_named_mesh_reads_as_medit(tmp_path: Path) -> None:
    """'.mesh' is shared, so the keyword the file opens with decides."""
    poly = polyxios.read(_write_mesh(tmp_path, _TET))
    assert poly.vertices.shape == (4, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_an_mfem_file_named_mesh_still_reads_as_mfem(tmp_path: Path) -> None:
    poly = polyxios.read(_write_mesh(tmp_path, _MFEM_TET, "mfem.mesh"))
    assert poly.vertices.shape == (4, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_writing_a_bare_mesh_path_still_writes_mfem(tmp_path: Path) -> None:
    """An output file has no content to sniff, so the owner keeps the writes."""
    out = tmp_path / "out.mesh"
    polyxios.write(_tet_mesh(), out)
    assert out.read_text().startswith("MFEM mesh")


def test_fmt_medit_writes_medit_to_a_mesh_path(tmp_path: Path) -> None:
    out = tmp_path / "out.mesh"
    polyxios.write(_tet_mesh(), out, fmt=".medit")
    assert out.read_text().startswith("MeshVersionFormatted")
    assert polyxios.read(out).vertices.shape == (4, 3)


def test_a_medit_extension_needs_no_fmt(tmp_path: Path) -> None:
    out = tmp_path / "out.medit"
    polyxios.write(_tet_mesh(), out)
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2, 3])


# --- meshio #1311: what bamg writes -----------------------------------------


def test_issue_1311_dimension_on_its_own_line_and_no_end(tmp_path: Path) -> None:
    """bamg puts the count on the keyword's line and omits the closing End."""
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 1\nDimension 2\n"
        "Vertices 3\n0 0 1\n1 0 1\n0 1 1\n"
        "Triangles 1\n1 2 3 4\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_allclose(poly.vertices[:, 2], 0.0)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(poly.element_attrs["ref"], [4])


def test_issue_1311_a_two_dimensional_file_pads_z(tmp_path: Path) -> None:
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n2\nVertices\n2\n"
        "1.5 2.5 0\n3.5 4.5 0\nEdges\n1\n1 2 0\nEnd\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[1.5, 2.5, 0], [3.5, 4.5, 0]])


def test_trailing_junk_after_end_is_not_read(tmp_path: Path) -> None:
    path = _write_mesh(tmp_path, _TET + "Triangles\n1\n1 2 3 9\n")
    poly = read(path)
    assert len(poly.element_types) == 1


# --- meshio #1511, #1508: the sections a Medit file can carry ----------------


@pytest.mark.parametrize(
    ("section", "n_nodes", "kind"),
    [
        ("Edges", 2, "line"),
        ("Triangles", 3, "triangle"),
        ("Quadrilaterals", 4, "quad"),
        ("Tetrahedra", 4, "tetra"),
        ("Pyramids", 5, "pyramid"),
        ("Prisms", 6, "wedge"),
        ("Pentahedra", 6, "wedge"),
        ("Hexahedra", 8, "hexahedron"),
    ],
)
def test_issue_1511_every_entity_section_reads(
    tmp_path: Path, section: str, n_nodes: int, kind: str
) -> None:
    """A section a reader skips is a block of the mesh that vanishes."""
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(n_nodes))
    refs = " ".join(str(i + 1) for i in range(n_nodes))
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n{n_nodes}\n{verts}\n"
        f"{section}\n1\n{refs} 3\nEnd\n",
    )
    poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES[kind]]
    np.testing.assert_array_equal(poly.connectivity, np.arange(n_nodes))


def test_issue_1508_several_sections_land_in_one_mesh(tmp_path: Path) -> None:
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(8))
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n8\n{verts}\n"
        "Triangles\n1\n1 2 3 1\n"
        "Prisms\n1\n1 2 3 4 5 6 2\n"
        "Hexahedra\n1\n1 2 3 4 5 6 7 8 3\nEnd\n",
    )
    poly = read(path)
    assert poly.element_types.tolist() == [
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["wedge"],
        ELEMENT_TYPES["hexahedron"],
    ]
    np.testing.assert_array_equal(poly.offsets, [0, 3, 9, 17])


def test_higher_order_sections_are_skipped_with_a_named_warning(
    tmp_path: Path,
) -> None:
    """libMeshb fixes no node ordering for them; guessing one bends elements."""
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(6))
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n6\n{verts}\n"
        "TrianglesP2\n1\n1 2 3 4 5 6 1\n"
        "Triangles\n1\n1 2 3 1\nEnd\n",
    )
    with pytest.warns(UserWarning, match="TrianglesP2"):
        poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]


def test_a_section_this_codec_does_not_know_warns(tmp_path: Path) -> None:
    path = _write_mesh(tmp_path, _TET.replace("End\n", "Wobbles\n1\n1 2\nEnd\n"))
    with pytest.warns(UserWarning, match="Wobbles"):
        read(path)


def test_a_metadata_section_is_skipped_without_a_word(tmp_path: Path) -> None:
    """A Corners block says nothing about the geometry; it is not a problem."""
    path = _write_mesh(tmp_path, _TET.replace("End\n", "Corners\n1\n1\nEnd\n"))
    poly = read(path)
    assert len(poly.element_types) == 1


# --- meshio #1258: references are labels, and labels have to survive ---------


def test_issue_1258_element_refs_become_tag_groups(tmp_path: Path) -> None:
    """A label kept only as a column of ints does not survive a conversion."""
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(4))
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n4\n{verts}\n"
        "Triangles\n3\n1 2 3 10\n1 2 4 20\n2 3 4 10\nEnd\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_attrs["ref"], [10, 20, 10])
    np.testing.assert_array_equal(poly.element_tags["ref_10"], [0, 2])
    np.testing.assert_array_equal(poly.element_tags["ref_20"], [1])


def test_issue_1258_refs_survive_the_trip_to_vtk(tmp_path: Path) -> None:
    """The headline case: a Medit region label readable in a .vtk."""
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(4))
    src = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n4\n{verts}\n"
        "Triangles\n2\n1 2 3 10\n2 3 4 20\nEnd\n",
    )
    out = tmp_path / "out.vtk"
    polyxios.write(polyxios.read(src), out)
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.element_attrs["ref"], [10, 20])


def test_issue_1258_vertex_refs_are_kept_when_any_is_set(tmp_path: Path) -> None:
    poly = read(_write_mesh(tmp_path, _TET))
    np.testing.assert_array_equal(poly.vertex_attrs["ref"], [1, 1, 2, 2])


def test_all_zero_refs_are_not_invented_as_an_attribute(tmp_path: Path) -> None:
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(3))
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n3\n{verts}\nTriangles\n1\n1 2 3 0\nEnd\n",
    )
    assert "ref" not in read(path).vertex_attrs


def test_refs_survive_a_round_trip(tmp_path: Path) -> None:
    poly = read(_write_mesh(tmp_path, _TET))
    out = tmp_path / "back.medit"
    write(poly, out)
    back = read(out)
    np.testing.assert_array_equal(back.element_attrs["ref"], [7])
    np.testing.assert_array_equal(back.vertex_attrs["ref"], [1, 1, 2, 2])


def test_refs_are_recovered_from_tag_groups_alone(tmp_path: Path) -> None:
    """A mesh that kept the groups but lost the column still writes its labels."""
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
        element_tags={"ref_42": np.array([0], dtype=np.int32)},
    )
    out = tmp_path / "tags.medit"
    write(poly, out)
    np.testing.assert_array_equal(read(out).element_attrs["ref"], [42])


# --- errors ------------------------------------------------------------------


def test_a_file_without_the_keyword_is_refused(tmp_path: Path) -> None:
    path = _write_mesh(tmp_path, "Vertices\n1\n0 0 0 0\nEnd\n", "x.medit")
    with pytest.raises(CodecError, match="MeshVersionFormatted"):
        read(path)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="empty"):
        read(_write_mesh(tmp_path, "", "e.medit"))


def test_a_truncated_section_is_refused(tmp_path: Path) -> None:
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\nVertices\n4\n0 0 0 0\nEnd\n",
        "t.medit",
    )
    with pytest.raises(CodecError, match="declares 4"):
        read(path)


@pytest.mark.parametrize("bad", ["0", "9"])
def test_an_out_of_range_vertex_reference_is_refused(tmp_path: Path, bad: str) -> None:
    """Medit indices are 1-based; a 0 would wrap to the last vertex once shifted."""
    verts = "\n".join(f"{float(i)} 0. 0. 0" for i in range(3))
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\n"
        f"Vertices\n3\n{verts}\nTriangles\n1\n1 2 {bad} 0\nEnd\n",
        "r.medit",
    )
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_an_absurd_count_is_refused(tmp_path: Path) -> None:
    path = _write_mesh(
        tmp_path,
        "MeshVersionFormatted 2\nDimension\n3\nVertices\n999999999999\n",
        "big.medit",
    )
    with pytest.raises(CodecError, match="safety cap"):
        read(path)


def test_a_type_with_no_medit_section_is_skipped(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=np.zeros((3, 3), dtype=np.float64),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["polygon"]], dtype=np.uint8),
    )
    with pytest.warns(UserWarning, match="polygon"):
        write(poly, tmp_path / "poly.medit")


def test_write_warns_on_unknown_options(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tet_mesh(), tmp_path / "o.medit", nonsense=1)


def test_lazy_is_reported_rather_than_ignored(tmp_path: Path) -> None:
    path = _write_mesh(tmp_path, _TET, "l.medit")
    with pytest.warns(UserWarning, match="lazy"):
        read(path, lazy=True)


def test_reads_what_meshio_writes(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    out = tmp_path / "meshio.mesh"
    meshio.Mesh(verts, [("tetra", np.array([[0, 1, 2, 3]]))]).write(
        out, file_format="medit"
    )
    poly = read(out)
    np.testing.assert_allclose(poly.vertices, verts)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_meshio_reads_what_polyxios_writes(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    out = tmp_path / "out.mesh"
    write(_tet_mesh(), out)
    mesh = meshio.read(out, file_format="medit")
    np.testing.assert_allclose(mesh.points, _tet_mesh().vertices)
    np.testing.assert_array_equal(mesh.cells[0].data, [[0, 1, 2, 3]])


def _plane_mesh(tmp_path) -> Path:
    path = tmp_path / "plane.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension\n2\n"
        "Vertices\n3\n0 0 1\n1 0 2\n0 1 3\n"
        "Triangles\n1\n1 2 3 9\nEnd\n"
    )
    return path


def test_a_two_dimensional_file_goes_back_out_in_two_dimensions(tmp_path) -> None:
    """A bamg file promoted to 3-D is one bamg no longer reads."""
    poly = read(_plane_mesh(tmp_path))
    assert poly.vertices.shape == (3, 3)
    assert poly.global_attrs["was_2d"] is True

    out = tmp_path / "back.mesh"
    write(poly, out)
    body = out.read_text()
    assert "Dimension 2" in body
    # Three columns in, two coordinates and a reference out.
    assert body.splitlines()[6].split() == ["0", "0", "1"]
    np.testing.assert_allclose(read(out).vertices, poly.vertices)


def test_a_plane_mesh_lifted_out_of_the_plane_is_written_in_three(tmp_path) -> None:
    """Writing two coordinates for a vertex that has three drops one."""
    poly = read(_plane_mesh(tmp_path))
    poly.vertices[1, 2] = 0.5
    out = tmp_path / "lifted.mesh"
    with pytest.warns(UserWarning, match="third coordinate"):
        write(poly, out)
    assert "Dimension 3" in out.read_text()
    np.testing.assert_allclose(read(out).vertices, poly.vertices)


def test_a_float_vertex_reference_is_refused_rather_than_truncated(
    tmp_path,
) -> None:
    """A reference is a number the file names exactly; rounding relabels it."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.vertex_attrs["ref"] = np.array([1.7, 2.2, 3.9])
    out = tmp_path / "float.mesh"
    with pytest.warns(UserWarning, match="one integer per vertex"):
        write(poly, out)
    assert read(out).vertex_attrs == {}


def test_an_integer_vertex_reference_is_written_and_read_back(tmp_path) -> None:
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.vertex_attrs["ref"] = np.array([4, 5, 6], dtype=np.int32)
    out = tmp_path / "refs.mesh"
    write(poly, out)
    np.testing.assert_array_equal(read(out).vertex_attrs["ref"], [4, 5, 6])


def test_a_dimension_the_format_cannot_mesh_names_itself(tmp_path) -> None:
    """Medit meshes the plane and space; the message must not call 3 a cap."""
    path = tmp_path / "four.mesh"
    path.write_text("MeshVersionFormatted 2\nDimension 4\nVertices\n0\nEnd\n")
    with pytest.raises(CodecError, match="Dimension 4 is not a mesh"):
        read(path)


def test_a_file_declaring_no_vertices_says_so(tmp_path) -> None:
    path = tmp_path / "empty.mesh"
    path.write_text("MeshVersionFormatted 2\nDimension 3\nVertices\n0\nEnd\n")
    with pytest.raises(CodecError, match="declares no vertices"):
        read(path)


def test_a_tag_group_that_holds_no_indices_is_reported(tmp_path) -> None:
    """Nothing checks a tag group's dtype on the way in; a float indexes none."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.element_tags["ref_7"] = np.array([0.0])
    with pytest.warns(UserWarning, match="do not hold"):
        write(poly, tmp_path / "bad_tag.mesh")


def test_a_section_holding_a_word_is_stepped_over_by_its_own_length(
    tmp_path: Path,
) -> None:
    """A skipped section's payload must not be read as the file's structure.

    Sections this codec does not decode used to be stepped over by name
    alone, leaving their records to the scan that looks for the next section.
    That holds while a payload is numbers - a number cannot name a section -
    and stops holding at 'Identifier', whose value is a word: the word reads
    as a section of its own and is reported as one the codec does not
    support.
    """
    path = tmp_path / "identifier.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension 3\n"
        "Identifier\nmygrid\n"
        "GeometricSupport\nsomecad\n"
        "Corners\n2\n1\n2\n"
        "Normals\n2\n0 0 1\n0 1 0\n"
        "NormalAtVertices\n1\n1 1\n"
        "Vertices\n3\n0 0 0 1\n1 0 0 1\n0 1 0 1\n"
        "Triangles\n1\n1 2 3 7\nEnd\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    assert sorted(poly.element_tags) == ["ref_7"]


def test_a_higher_order_section_is_stepped_over_whole(tmp_path: Path) -> None:
    """Its records are known in width even though its ordering is not."""
    path = tmp_path / "ho.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension 3\n"
        "Vertices\n3\n0 0 0 1\n1 0 0 1\n0 1 0 1\n"
        "TrianglesP2\n1\n1 2 3 1 2 3 9\n"
        "Triangles\n1\n1 2 3 7\nEnd\n"
    )
    with pytest.warns(UserWarning, match="higher-order section"):
        poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(poly.element_tags["ref_7"], [0])


def test_a_reference_wider_than_32_bits_is_read_as_it_is_written(
    tmp_path: Path,
) -> None:
    """The ASCII format puts no ceiling on a reference; narrowing one relabels."""
    path = tmp_path / "wide.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension 3\n"
        "Vertices\n3\n0 0 0 5000000000\n1 0 0 1\n0 1 0 1\n"
        "Triangles\n1\n1 2 3 3000000000\nEnd\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["ref"], [5000000000, 1, 1])
    np.testing.assert_array_equal(poly.element_attrs["ref"], [3000000000])
    assert "ref_3000000000" in poly.element_tags


def test_a_second_vertices_section_is_refused(tmp_path: Path) -> None:
    """Replacing the block the elements index moves them onto other geometry."""
    text = (
        "MeshVersionFormatted 2\nDimension 3\n"
        "Vertices\n3\n0 0 0 1\n1 0 0 1\n0 1 0 1\n"
        "Triangles\n1\n1 2 3 5\n"
        "Vertices\n3\n9 9 9 2\n8 8 8 2\n7 7 7 2\nEnd\n"
    )
    with pytest.raises(CodecError, match="second Vertices section"):
        read(_write_mesh(tmp_path, text))


def test_two_groups_naming_one_element_report_the_reference_that_lost(
    tmp_path: Path,
) -> None:
    """A record carries one reference, so the caller is told which won."""
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.element_tags["ref_1"] = np.array([0], dtype=np.int32)
    poly.element_tags["ref_2"] = np.array([0], dtype=np.int32)
    with pytest.warns(UserWarning, match="already labelled"):
        write(poly, tmp_path / "clash.mesh")


def test_a_dimension_after_the_vertices_is_a_codec_error(tmp_path) -> None:
    """A vertex record is as wide as the dimension, so a Dimension arriving
    after the block means every coordinate came out of the wrong token."""
    path = tmp_path / "late.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nVertices 3\n"
        "0 0 0 1\n1 0 0 1\n0 1 0 1\nDimension 2\nEnd\n"
    )
    with pytest.raises(CodecError, match="declared after the Vertices section"):
        read(path)


def test_a_dimension_repeated_at_the_same_width_is_not_refused(tmp_path) -> None:
    """Only a disagreement means the block was read at the wrong width."""
    path = tmp_path / "again.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension 3\nVertices 3\n"
        "0 0 0 1\n1 0 0 1\n0 1 0 1\nDimension 3\nTriangles 1\n1 2 3 4\nEnd\n"
    )
    assert read(path).vertices.shape == (3, 3)


def test_the_element_cap_is_counted_over_the_file_not_the_section(
    tmp_path, monkeypatch
) -> None:
    """A mesh spreads its elements across as many sections as it holds types,
    and a cap each of them passes on its own is one the file does not."""
    monkeypatch.setattr("polyxios.codecs._medit.MAX_SAFE_ELEMENTS", 3)
    path = tmp_path / "many.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension 3\nVertices 3\n"
        "0 0 0 0\n1 0 0 0\n0 1 0 0\n"
        "Triangles 2\n1 2 3 1\n1 2 3 1\n"
        "Edges 2\n1 2 1\n2 3 1\n"
        "End\n"
    )
    with pytest.raises(CodecError, match="element count exceeds the safety cap"):
        read(path)
