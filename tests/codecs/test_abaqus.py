from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._abaqus import read, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_file_has_node_element_keywords(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    text = Path(tmp).read_text()
    assert "*Node" in text
    assert "*Element" in text
    assert "C3D4" in text


def test_missing_node_section_raises(tmp_path) -> None:
    bad = "*Heading\n** no node section\n"
    tmp = tmp_path / "bad.inp"
    tmp.write_text(bad)
    with pytest.raises(CodecError):
        read(tmp)


def test_mixed_elements(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    tmp = tmp_path / "test.inp"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2


def test_continuation_line_element(tmp_path) -> None:
    # Real .inp files split long element rows across lines ending with ','
    inp = (
        "*Node\n"
        "1, 0.0, 0.0, 0.0\n"
        "2, 1.0, 0.0, 0.0\n"
        "3, 0.0, 1.0, 0.0\n"
        "4, 0.0, 0.0, 1.0\n"
        "*Element, type=C3D4\n"
        "1, 1,\n"
        "2, 3, 4\n"
    )
    tmp = tmp_path / "cont.inp"
    tmp.write_text(inp)
    poly = read(tmp)
    assert len(poly.element_types) == 1
    assert poly.connectivity.tolist() == [0, 1, 2, 3]


def test_write_unknown_type_id_raises(tmp_path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([255], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="unknown element type id"):
        write(poly, tmp_path / "out.inp")


def test_write_unmapped_type_raises(tmp_path) -> None:
    # polyhedron has no Abaqus card at all, so it cannot be written
    verts = np.zeros((8, 3), dtype=np.float64)
    poly = PolyData(
        vertices=verts,
        connectivity=np.arange(8, dtype=np.int32),
        offsets=np.array([0, 8], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["polyhedron"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="no write mapping"):
        write(poly, tmp_path / "out.inp")


def test_unrecognised_element_type_warns(tmp_path) -> None:
    inp = "*Node\n1, 0.0, 0.0, 0.0\n2, 1.0, 0.0, 0.0\n*Element, type=BOGUS99\n1, 1, 2\n"
    tmp = tmp_path / "warn.inp"
    tmp.write_text(inp)
    with pytest.warns(UserWarning, match="unrecognised"):
        read(tmp)


def test_2d_node_z_padding(tmp_path) -> None:
    inp = (
        "*Node\n"
        "1, 0.0, 0.0\n"
        "2, 1.0, 0.0\n"
        "3, 0.0, 1.0\n"
        "*Element, type=CPS3\n"
        "1, 1, 2, 3\n"
    )
    tmp = tmp_path / "flat.inp"
    tmp.write_text(inp)
    poly = read(tmp)
    np.testing.assert_array_equal(poly.vertices[:, 2], [0.0, 0.0, 0.0])


def _write_inp(tmp_path: Path, text: str, name: str = "m.inp") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


_TWO_TETRA = (
    "*Node\n"
    "1, 0.0, 0.0, 0.0\n"
    "2, 1.0, 0.0, 0.0\n"
    "3, 0.0, 1.0, 0.0\n"
    "4, 0.0, 0.0, 1.0\n"
    "5, 1.0, 1.0, 1.0\n"
    "*Element, type=C3D4\n"
    "1, 1, 2, 3, 4\n"
    "2, 2, 3, 4, 5\n"
)


# --- meshio #1515, #320, #1393: the element type table ------------------------


@pytest.mark.parametrize(
    ("inp_type", "kind", "n_nodes"),
    [
        ("CPS8R", "quadratic_quad", 8),
        ("CPE4", "quad", 4),
        ("CPE8H", "quadratic_quad", 8),
        ("CAX4P", "quad", 4),
        ("CAX8R", "quadratic_quad", 8),
        ("CPE6M", "quadratic_triangle", 6),
        ("C3D8R", "hexahedron", 8),
        ("C3D8I", "hexahedron", 8),
        ("C3D10M", "quadratic_tetra", 10),
        ("C3D15", "quadratic_wedge", 15),
        ("C3D20RH", "quadratic_hexahedron", 20),
        ("S4R5", "quad", 4),
        ("S8R5", "quadratic_quad", 8),
        ("S9R5", "biquadratic_quad", 9),
        ("S3RS", "triangle", 3),
        ("STRI65", "quadratic_triangle", 6),
        ("R3D3", "triangle", 3),
        ("SC8R", "hexahedron", 8),
        ("B32H", "quadratic_edge", 3),
        ("T2D3", "quadratic_edge", 3),
        ("DC3D8", "hexahedron", 8),
    ],
)
def test_issue_1515_element_suffixes_are_stripped_before_lookup(
    tmp_path: Path, inp_type: str, kind: str, n_nodes: int
) -> None:
    """A reduced-integration or hybrid card is the same element as its base."""
    nodes = "\n".join(f"{i + 1}, {i}.0, 0.0, 0.0" for i in range(n_nodes))
    refs = ", ".join(str(i + 1) for i in range(n_nodes))
    path = _write_inp(
        tmp_path, f"*Node\n{nodes}\n*Element, type={inp_type}\n1, {refs}\n"
    )
    poly = read(path)
    assert list(poly.element_types) == [ELEMENT_TYPES[kind]]
    assert len(poly.connectivity) == n_nodes


def test_issue_1393_unknown_card_warns_and_keeps_the_rest(tmp_path: Path) -> None:
    """An unknown card must not take the blocks around it down with it."""
    path = _write_inp(
        tmp_path,
        "*Node\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n"
        "*Element, type=ZZZ9\n1, 1, 2, 3\n"
        "*Element, type=C3D4\n2, 1, 2, 3, 4\n",
    )
    with pytest.warns(UserWarning, match="ZZZ9"):
        poly = read(path)
    assert list(poly.element_types) == [ELEMENT_TYPES["tetra"]]


def test_issue_1299_element_type_option_forces_the_card(tmp_path: Path) -> None:
    """A solver deck often needs C3D8R where the mesh only says 'hexahedron'."""
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    poly = make_polydata(verts, [("hexahedron", np.arange(8).reshape(1, 8))])
    tmp = tmp_path / "forced.inp"
    write(poly, tmp, element_type={"hexahedron": "C3D8R"})
    assert "type=C3D8R" in tmp.read_text()
    assert list(read(tmp).element_types) == [ELEMENT_TYPES["hexahedron"]]


def test_issue_1299_element_type_for_a_type_not_present_warns(tmp_path: Path) -> None:
    """A silently ignored override writes a deck the user did not ask for."""
    with pytest.warns(UserWarning, match="triangle"):
        write(_tet_mesh(), tmp_path / "o.inp", element_type={"triangle": "S3R"})


def test_quadratic_types_survive_a_round_trip(tmp_path: Path) -> None:
    """Reading a type this codec cannot write back loses the mesh at export."""
    verts = np.arange(30, dtype=np.float64).reshape(10, 3)
    poly = make_polydata(verts, [("quadratic_tetra", np.arange(10).reshape(1, 10))])
    tmp = tmp_path / "q.inp"
    write(poly, tmp)
    back = read(tmp)
    assert list(back.element_types) == [ELEMENT_TYPES["quadratic_tetra"]]
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


# --- meshio #1528, #309: several *NODE blocks --------------------------------


def test_issue_1528_multiple_node_blocks_accumulate(tmp_path: Path) -> None:
    """A second *NODE block adds nodes; overwriting drops the first block."""
    path = _write_inp(
        tmp_path,
        "*Node\n1, 0.0, 0.0, 0.0\n2, 1.0, 0.0, 0.0\n"
        "*Node\n3, 0.0, 1.0, 0.0\n4, 0.0, 0.0, 1.0\n"
        "*Element, type=C3D4\n1, 1, 2, 3, 4\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (4, 3)
    np.testing.assert_allclose(poly.vertices[3], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_issue_309_repeated_node_id_updates_rather_than_appends(
    tmp_path: Path,
) -> None:
    """A repeated id is the same node restated; appending leaves a stray vertex."""
    path = _write_inp(
        tmp_path,
        "*Node\n1, 0.0, 0.0, 0.0\n2, 1.0, 0.0, 0.0\n3, 0.0, 1.0, 0.0\n"
        "4, 0.0, 0.0, 1.0\n"
        "*Node\n4, 0.0, 0.0, 2.0\n"
        "*Element, type=C3D4\n1, 1, 2, 3, 4\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (4, 3)
    np.testing.assert_allclose(poly.vertices[3], [0.0, 0.0, 2.0])


# --- meshio #1527: node and element sets -------------------------------------


def test_issue_1527_nset_on_a_node_block_becomes_a_vertex_tag(
    tmp_path: Path,
) -> None:
    """The NSET names the block's nodes; dropping it loses the boundary."""
    path = _write_inp(
        tmp_path,
        "*Node, nset=fixed\n1, 0.0, 0.0, 0.0\n2, 1.0, 0.0, 0.0\n"
        "*Node\n3, 0.0, 1.0, 0.0\n4, 0.0, 0.0, 1.0\n"
        "*Element, type=C3D4\n1, 1, 2, 3, 4\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_tags["fixed"], [0, 1])


def test_standalone_nset_becomes_a_vertex_tag(tmp_path: Path) -> None:
    path = _write_inp(tmp_path, _TWO_TETRA + "*Nset, nset=corner\n1, 3,\n5\n")
    np.testing.assert_array_equal(read(path).vertex_tags["corner"], [0, 2, 4])


def test_nset_generate_expands_the_range(tmp_path: Path) -> None:
    path = _write_inp(tmp_path, _TWO_TETRA + "*Nset, nset=run, generate\n1, 5, 2\n")
    np.testing.assert_array_equal(read(path).vertex_tags["run"], [0, 2, 4])


def test_elset_on_an_element_block_becomes_an_element_tag(tmp_path: Path) -> None:
    path = _write_inp(
        tmp_path,
        "*Node\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n5, 1, 1, 1\n"
        "*Element, type=C3D4, elset=solid\n1, 1, 2, 3, 4\n2, 2, 3, 4, 5\n",
    )
    np.testing.assert_array_equal(read(path).element_tags["solid"], [0, 1])


def test_standalone_elset_becomes_an_element_tag(tmp_path: Path) -> None:
    path = _write_inp(tmp_path, _TWO_TETRA + "*Elset, elset=second\n2\n")
    np.testing.assert_array_equal(read(path).element_tags["second"], [1])


def test_a_set_naming_an_absent_id_warns(tmp_path: Path) -> None:
    path = _write_inp(tmp_path, _TWO_TETRA + "*Nset, nset=ghost\n99\n")
    with pytest.warns(UserWarning, match="ghost"):
        poly = read(path)
    assert "ghost" not in poly.vertex_tags


def test_a_set_body_names_an_earlier_set_whatever_its_case(tmp_path: Path) -> None:
    """Abaqus matches a set name without regard to case, and so must this."""
    path = _write_inp(
        tmp_path,
        _TWO_TETRA + "*Nset, nset=Top\n1, 3,\n5\n*Nset, nset=all\nTOP\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_tags["all"], [0, 2, 4])


def test_a_set_redeclared_in_another_case_is_one_set(tmp_path: Path) -> None:
    """Two spellings of one name would otherwise open two sets side by side."""
    path = _write_inp(
        tmp_path,
        _TWO_TETRA + "*Nset, nset=Top\n1\n*Nset, nset=TOP\n3\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_tags["Top"], [0, 2])
    assert "TOP" not in poly.vertex_tags


def test_an_element_id_defined_twice_is_reported(tmp_path: Path) -> None:
    """Only the last answers to the id, so a set naming it reaches one cell."""
    path = _write_inp(
        tmp_path,
        "*Node\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n5, 1, 1, 1\n"
        "*Element, type=C3D4\n1, 1, 2, 3, 4\n"
        "*Element, type=C3D4\n1, 2, 3, 4, 5\n",
    )
    with pytest.warns(UserWarning, match="defined twice"):
        poly = read(path)
    assert len(poly.element_types) == 2


# --- meshio #1531: *INCLUDE --------------------------------------------------


def test_issue_1531_include_is_resolved_against_the_parent(tmp_path: Path) -> None:
    """A deck split over files reads as one mesh or it does not read at all."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nodes.inp").write_text(
        "*Node\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n", encoding="utf-8"
    )
    path = _write_inp(
        tmp_path,
        "*Include, input=sub/nodes.inp\n*Element, type=C3D4\n1, 1, 2, 3, 4\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (4, 3)
    assert len(poly.element_types) == 1


def test_issue_1531_include_escaping_the_parent_is_refused(tmp_path: Path) -> None:
    """An .inp is untrusted input; a relative path must not read /etc/passwd."""
    (tmp_path / "secret.inp").write_text("*Node\n1, 0, 0, 0\n", encoding="utf-8")
    (tmp_path / "deck").mkdir()
    path = _write_inp(
        tmp_path / "deck",
        "*Node\n1, 0, 0, 0\n*Include, input=../secret.inp\n",
    )
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_issue_1531_include_recursion_is_capped(tmp_path: Path) -> None:
    """A deck that includes itself would otherwise read until memory runs out."""
    path = _write_inp(tmp_path, "*Node\n1, 0, 0, 0\n*Include, input=m.inp\n")
    with pytest.raises(CodecError, match="nest|depth"):
        read(path)


def test_issue_1531_missing_include_names_the_file(tmp_path: Path) -> None:
    path = _write_inp(tmp_path, "*Node\n1, 0, 0, 0\n*Include, input=gone.inp\n")
    with pytest.raises(CodecError, match="gone.inp"):
        read(path)


def test_include_from_a_buffer_is_refused(tmp_path: Path) -> None:
    """A buffer has no directory, so a relative include names nothing."""
    import io

    buf = io.BytesIO(b"*Node\n1, 0, 0, 0\n*Include, input=other.inp\n")
    with pytest.raises(CodecError, match="INCLUDE"):
        read(buf)


# --- meshio #1569: *SYSTEM ---------------------------------------------------


def test_issue_1569_system_shifts_the_nodes_that_follow(tmp_path: Path) -> None:
    """Ignoring *SYSTEM puts a whole part at the wrong place in the assembly."""
    path = _write_inp(
        tmp_path,
        "*Node\n1, 0.0, 0.0, 0.0\n"
        "*System\n10.0, 0.0, 0.0\n"
        "*Node\n2, 1.0, 0.0, 0.0\n"
        "*System\n"
        "*Node\n3, 2.0, 0.0, 0.0\n"
        "*Element, type=T3D2\n1, 1, 2\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(poly.vertices[1], [11.0, 0.0, 0.0])
    np.testing.assert_allclose(poly.vertices[2], [2.0, 0.0, 0.0])


def test_issue_1569_system_rotates_the_nodes_that_follow(tmp_path: Path) -> None:
    """The two points on the data line define the local axes, not just a shift."""
    path = _write_inp(
        tmp_path,
        "*System\n0.0, 0.0, 0.0, 0.0, 1.0, 0.0\n0.0, 0.0, 1.0\n"
        "*Node\n1, 1.0, 0.0, 0.0\n"
        "*Element, type=T3D2\n1, 1, 1\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices[0], [0.0, 1.0, 0.0], atol=1e-12)


# --- meshio #1456: parts and instances ---------------------------------------


def test_issue_1456_instances_are_merged_with_their_own_numbering(
    tmp_path: Path,
) -> None:
    """Two instances both number their nodes from 1; merging blind folds them."""
    path = _write_inp(
        tmp_path,
        "*Assembly, name=a\n"
        "*Instance, name=i1, part=p\n"
        "*Node\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n"
        "*Element, type=CPS3\n1, 1, 2, 3\n"
        "*End Instance\n"
        "*Instance, name=i2, part=p\n"
        "*Node\n1, 0, 0, 5\n2, 1, 0, 5\n3, 0, 1, 5\n"
        "*Element, type=CPS3\n1, 1, 2, 3\n"
        "*End Instance\n"
        "*End Assembly\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (6, 3)
    assert len(poly.element_types) == 2
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(poly.element_tags["i2"], [1])


# --- meshio #1529: the writer's output ---------------------------------------


def test_issue_1529_meshio_reads_what_polyxios_writes(tmp_path: Path) -> None:
    """An .inp no other reader accepts is not an export, it is a dead end."""
    meshio = pytest.importorskip("meshio")
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))],
    )
    tmp = tmp_path / "out.inp"
    write(poly, tmp)
    mesh = meshio.read(tmp, file_format="abaqus")
    np.testing.assert_allclose(mesh.points, verts)
    np.testing.assert_array_equal(mesh.cells[0].data, [[0, 1, 2, 3], [1, 2, 3, 4]])


def test_issue_1529_element_tags_are_written_as_elsets(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
    )
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32),
        offsets=np.array([0, 4, 8], dtype=np.int32),
        element_types=np.full(2, ELEMENT_TYPES["tetra"], dtype=np.uint8),
        element_tags={"solid": np.array([1], dtype=np.int32)},
        vertex_tags={"fixed": np.array([0, 2], dtype=np.int32)},
    )
    tmp = tmp_path / "sets.inp"
    write(poly, tmp)
    meshio.read(tmp, file_format="abaqus")
    back = read(tmp)
    np.testing.assert_array_equal(back.element_tags["solid"], [1])
    np.testing.assert_array_equal(back.vertex_tags["fixed"], [0, 2])


def test_issue_1529_reads_what_meshio_writes(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tmp = tmp_path / "meshio.inp"
    meshio.Mesh(verts, [("tetra", np.array([[0, 1, 2, 3]]))]).write(
        tmp, file_format="abaqus"
    )
    poly = read(tmp)
    np.testing.assert_allclose(poly.vertices, verts)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_element_type_may_ask_for_a_variant_of_the_same_card(tmp_path: Path) -> None:
    """C3D8R is a reduced-integration brick: the same eight nodes."""
    poly = make_polydata(
        np.arange(24, dtype=np.float64).reshape(8, 3),
        [("hexahedron", np.arange(8).reshape(1, 8))],
    )
    out = tmp_path / "brick.inp"
    write(poly, out, element_type={"hexahedron": "C3D8R"})
    assert "*Element, type=C3D8R" in out.read_text()
    assert read(out).element_types.tolist() == poly.element_types.tolist()


def test_element_type_naming_a_card_of_another_element_is_refused(
    tmp_path: Path,
) -> None:
    """Eight nodes under a twenty-node card is a deck no reader loads."""
    poly = make_polydata(
        np.arange(24, dtype=np.float64).reshape(8, 3),
        [("hexahedron", np.arange(8).reshape(1, 8))],
    )
    with pytest.raises(CodecError, match="20-node"):
        write(poly, tmp_path / "bad.inp", element_type={"hexahedron": "C3D20"})


def test_element_type_cannot_spell_two_element_types_with_one_card(
    tmp_path: Path,
) -> None:
    """One block cannot run rows of two lengths under a card that reads one."""
    verts = np.arange(15, dtype=np.float64).reshape(5, 3)
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("quad", np.array([[0, 1, 2, 4]])),
        ],
    )
    with pytest.raises(CodecError, match="one card cannot hold"):
        write(
            poly,
            tmp_path / "clash.inp",
            # A card this codec does not know is written as asked - but not
            # for two element types at once.
            element_type={"quad": "UEL4", "tetra": "UEL4"},
        )


# --- an assembly keeps its sets outside the instance that numbers them -------

_ASSEMBLY = """*Part, name=P1
*Node
1, 0., 0., 0.
2, 1., 0., 0.
3, 0., 1., 0.
4, 1., 1., 0.
*Element, type=S3
1, 1, 2, 3
2, 2, 4, 3
*End Part
*Assembly, name=A
*Instance, name=I1, part=P1
0., 0., 0.
*End Instance
*Elset, elset=TOP, instance=I1
2,
*Nset, nset=CORNER, instance=I1
1, 4
*End Assembly
"""


def test_an_assembly_level_set_is_numbered_by_the_instance_it_names(
    tmp_path,
) -> None:
    """A deck keeps almost every set out at assembly level, naming INSTANCE=."""
    path = tmp_path / "assembly.inp"
    path.write_text(_ASSEMBLY)
    poly = read(path)
    np.testing.assert_array_equal(poly.element_tags["TOP"], [1])
    np.testing.assert_array_equal(poly.vertex_tags["CORNER"], [0, 3])


def test_an_instance_that_only_places_its_part_shares_its_numbering(
    tmp_path,
) -> None:
    """An instance carrying no *Node of its own means the part's nodes."""
    path = tmp_path / "placed.inp"
    path.write_text(_ASSEMBLY.replace("2,\n", "1, 2\n"))
    np.testing.assert_array_equal(read(path).element_tags["TOP"], [0, 1])


def test_whitespace_inside_a_keyword_does_not_hide_the_card(tmp_path) -> None:
    """'*End  Instance' is the same card spelled loosely."""
    path = tmp_path / "loose.inp"
    path.write_text(_ASSEMBLY.replace("*End Instance", "*End  Instance"))
    np.testing.assert_array_equal(read(path).element_tags["TOP"], [1])


def test_a_set_naming_an_instance_the_deck_never_defines_is_reported(
    tmp_path,
) -> None:
    path = tmp_path / "ghost.inp"
    path.write_text(_ASSEMBLY.replace("instance=I1\n2,", "instance=NOPE\n2,"))
    # The set also reports the entry it lost, so every warning is caught.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read(path)
    assert any("NOPE" in str(w.message) for w in caught)


def test_a_generate_range_wider_than_the_deck_is_walked_from_the_ids(
    tmp_path,
) -> None:
    """'GENERATE 1, 999999999' is a legal card; the range is not a thing to walk."""
    path = tmp_path / "generate.inp"
    path.write_text(
        "*Node\n1, 0., 0., 0.\n2, 1., 0., 0.\n3, 0., 1., 0.\n"
        "*Element, type=S3\n1, 1, 2, 3\n"
        "*Elset, elset=BIG, generate\n1, 999999999, 1\n"
    )
    with pytest.warns(UserWarning, match="999999998 entry"):
        poly = read(path)
    np.testing.assert_array_equal(poly.element_tags["BIG"], [0])


def test_a_generate_range_the_deck_fills_still_reports_the_first_gap(
    tmp_path,
) -> None:
    path = tmp_path / "gap.inp"
    path.write_text(
        "*Node\n1, 0., 0., 0.\n2, 1., 0., 0.\n3, 0., 1., 0.\n4, 1., 1., 0.\n"
        "*Element, type=S3\n1, 1, 2, 3\n3, 2, 4, 3\n"
        "*Elset, elset=SPAN, generate\n1, 4, 1\n"
    )
    with pytest.warns(UserWarning, match="first '2'"):
        poly = read(path)
    np.testing.assert_array_equal(poly.element_tags["SPAN"], [0, 1])
