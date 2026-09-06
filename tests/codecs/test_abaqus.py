from __future__ import annotations

import dataclasses
from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
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


# --- the element type table ---------------------------------------------------


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
def test_element_suffixes_are_stripped_before_lookup(
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


def test_unknown_card_warns_and_keeps_the_rest(tmp_path: Path) -> None:
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


def test_element_type_option_forces_the_card(tmp_path: Path) -> None:
    """A solver deck often needs C3D8R where the mesh only says 'hexahedron'."""
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    poly = make_polydata(verts, [("hexahedron", np.arange(8).reshape(1, 8))])
    tmp = tmp_path / "forced.inp"
    write(poly, tmp, element_type={"hexahedron": "C3D8R"})
    assert "type=C3D8R" in tmp.read_text()
    assert list(read(tmp).element_types) == [ELEMENT_TYPES["hexahedron"]]


def test_element_type_for_a_type_not_present_warns(tmp_path: Path) -> None:
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


# --- several *NODE blocks ----------------------------------------------------


def test_multiple_node_blocks_accumulate(tmp_path: Path) -> None:
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


def test_repeated_node_id_updates_rather_than_appends(
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


# --- node and element sets ---------------------------------------------------


def test_nset_on_a_node_block_becomes_a_vertex_tag(
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


# --- *INCLUDE ----------------------------------------------------------------


def test_include_is_resolved_against_the_parent(tmp_path: Path) -> None:
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


def test_include_escaping_the_parent_is_refused(tmp_path: Path) -> None:
    """An .inp is untrusted input; a relative path must not read /etc/passwd."""
    (tmp_path / "secret.inp").write_text("*Node\n1, 0, 0, 0\n", encoding="utf-8")
    (tmp_path / "deck").mkdir()
    path = _write_inp(
        tmp_path / "deck",
        "*Node\n1, 0, 0, 0\n*Include, input=../secret.inp\n",
    )
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_include_recursion_is_capped(tmp_path: Path) -> None:
    """A deck that includes itself would otherwise read until memory runs out."""
    path = _write_inp(tmp_path, "*Node\n1, 0, 0, 0\n*Include, input=m.inp\n")
    with pytest.raises(CodecError, match="nest|depth"):
        read(path)


def test_missing_include_names_the_file(tmp_path: Path) -> None:
    path = _write_inp(tmp_path, "*Node\n1, 0, 0, 0\n*Include, input=gone.inp\n")
    with pytest.raises(CodecError, match="gone.inp"):
        read(path)


def test_include_from_a_buffer_is_refused(tmp_path: Path) -> None:
    """A buffer has no directory, so a relative include names nothing."""
    import io

    buf = io.BytesIO(b"*Node\n1, 0, 0, 0\n*Include, input=other.inp\n")
    with pytest.raises(CodecError, match="INCLUDE"):
        read(buf)


# --- *SYSTEM -----------------------------------------------------------------


def test_system_shifts_the_nodes_that_follow(tmp_path: Path) -> None:
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


def test_system_rotates_the_nodes_that_follow(tmp_path: Path) -> None:
    """The two points on the data line define the local axes, not just a shift."""
    path = _write_inp(
        tmp_path,
        "*System\n0.0, 0.0, 0.0, 0.0, 1.0, 0.0\n0.0, 0.0, 1.0\n"
        "*Node\n1, 1.0, 0.0, 0.0\n"
        "*Element, type=T3D2\n1, 1, 1\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices[0], [0.0, 1.0, 0.0], atol=1e-12)


# --- parts and instances -----------------------------------------------------


def test_instances_are_merged_with_their_own_numbering(
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


# --- the writer's output -----------------------------------------------------


def test_meshio_reads_what_polyxios_writes(tmp_path: Path) -> None:
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


def test_element_tags_are_written_as_elsets(tmp_path: Path) -> None:
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


def test_reads_what_meshio_writes(tmp_path: Path) -> None:
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


def test_an_include_may_reach_a_sibling_of_its_own_directory(
    tmp_path: Path,
) -> None:
    """The boundary is the deck's directory, not each included file's.

    A deck laid out as main.inp including parts/a.inp including
    ../common/nodes.inp never leaves the directory it was read from. A
    boundary that followed the nesting would narrow to 'parts' and refuse the
    third file for stepping out of it, which is the layout a real deck uses.
    """
    (tmp_path / "parts").mkdir()
    (tmp_path / "common").mkdir()
    (tmp_path / "main.inp").write_text("*Heading\n*INCLUDE, INPUT=parts/a.inp\n")
    (tmp_path / "parts" / "a.inp").write_text(
        "*INCLUDE, INPUT=../common/nodes.inp\n*Element, type=S3\n1, 1, 2, 3\n"
    )
    (tmp_path / "common" / "nodes.inp").write_text(
        "*Node\n1, 0., 0., 0.\n2, 1., 0., 0.\n3, 0., 1., 0.\n"
    )
    poly = read(tmp_path / "main.inp")
    assert poly.vertices.shape == (3, 3)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]


def test_an_include_still_cannot_leave_the_deck_directory(tmp_path: Path) -> None:
    """Pinning the boundary widens it to the deck, never past it."""
    (tmp_path / "deck").mkdir()
    (tmp_path / "secret.inp").write_text("*Node\n1, 0., 0., 0.\n")
    (tmp_path / "deck" / "main.inp").write_text(
        "*Heading\n*INCLUDE, INPUT=../secret.inp\n"
    )
    with pytest.raises(CodecError, match="outside the deck's directory"):
        read(tmp_path / "deck" / "main.inp")


def test_a_vertex_tag_indexing_no_node_is_dropped(tmp_path: Path) -> None:
    """A '*Nset' naming a node no '*Node' card defines is a deck Abaqus refuses."""
    poly = PolyData(
        vertices=np.zeros((3, 3)),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
        vertex_tags={"stale": np.array([0, 99])},
    )
    out = tmp_path / "stale.inp"
    with pytest.warns(UserWarning, match="index no node"):
        write(poly, out)
    body = out.read_text()
    assert "*Nset, nset=stale" in body
    assert "\n1\n" in body


def test_a_float_vertex_tag_is_refused_rather_than_rounded(tmp_path: Path) -> None:
    """Rounding an index moves a label onto another node."""
    poly = PolyData(
        vertices=np.zeros((3, 3)),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
        vertex_tags={"soft": np.array([0.0, 1.9])},
    )
    out = tmp_path / "soft.inp"
    with pytest.warns(UserWarning, match="index no node"):
        write(poly, out)
    assert "nset=soft" not in out.read_text()


def test_a_set_name_carrying_a_separator_is_folded(tmp_path: Path) -> None:
    """A comma or an '=' inside a name ends the card early."""
    poly = PolyData(
        vertices=np.zeros((3, 3)),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
        vertex_tags={"a,b=c\nd": np.array([0])},
    )
    out = tmp_path / "folded.inp"
    write(poly, out)
    assert "*Nset, nset=a_b_c_d" in out.read_text()


def test_a_part_does_not_take_the_decks_own_numbering_with_it(tmp_path: Path) -> None:
    """A '*Nset' out past '*End Part' still names the nodes the deck defined."""
    path = tmp_path / "mixed.inp"
    path.write_text(
        "*Node\n1, 0., 0., 0.\n2, 1., 0., 0.\n3, 0., 1., 0.\n"
        "*Element, type=S3\n1, 1, 2, 3\n"
        "*Part, name=P\n*Node\n1, 0., 0., 1.\n*End Part\n"
        "*Nset, nset=GLOBAL\n1, 2, 3\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_tags["GLOBAL"], [0, 1, 2])


def test_a_set_naming_one_element_twice_is_not_reported_as_lost(tmp_path) -> None:
    """Deduplicating the ids also collapses a repeat, which is not a loss."""
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.element_tags["A"] = np.array([0, 0], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "dup.inp")


def test_a_set_naming_an_element_that_was_not_written_is_still_reported(
    tmp_path,
) -> None:
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.element_tags["A"] = np.array([0, 5], dtype=np.int32)
    with pytest.warns(UserWarning, match="index no element"):
        write(poly, tmp_path / "gone.inp")


def test_two_tag_names_reaching_one_card_name_are_reported(tmp_path) -> None:
    """A card name cannot carry ',' or '=', so two groups can fold into one.

    Every member still reaches the file - Abaqus merges cards naming one set -
    but the groups stop being told apart, which is a loss this codec reports
    everywhere else rather than leaving to be found on the way back in.
    """
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.element_tags["a,b"] = np.array([0], dtype=np.int32)
    poly.element_tags["a=b"] = np.array([0], dtype=np.int32)
    with pytest.warns(UserWarning, match="spell one card name"):
        write(poly, tmp_path / "folded.inp")


def test_two_tag_names_differing_only_in_case_are_reported(tmp_path) -> None:
    """Abaqus matches a set name without regard to case, and so does the read."""
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.vertex_tags["Top"] = np.array([0], dtype=np.int32)
    poly.vertex_tags["TOP"] = np.array([1], dtype=np.int32)
    with pytest.warns(UserWarning, match="spell one card name"):
        write(poly, tmp_path / "cased.inp")


def test_distinct_set_names_are_not_reported_as_merged(tmp_path) -> None:
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.element_tags["A"] = np.array([0], dtype=np.int32)
    poly.element_tags["B"] = np.array([0], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "apart.inp")


# ---------------------------------------------------------------------------
# *HEADING: the deck's own title
# ---------------------------------------------------------------------------


def _deck(heading: str) -> str:
    return (
        f"{heading}"
        "*Node\n"
        "1, 0., 0., 0.\n"
        "2, 1., 0., 0.\n"
        "3, 0., 1., 0.\n"
        "*Element, type=S3\n"
        "1, 1, 2, 3\n"
    )


def test_a_heading_is_read_and_written_back(tmp_path) -> None:
    """The one card in a mesh deck that describes the model rather than a
    node, an element or a set."""
    path = tmp_path / "titled.inp"
    path.write_text(_deck("*Heading\nturbine blade, rev 3\n"))

    poly = read(path)
    assert poly.global_attrs["abaqus_heading"] == "turbine blade, rev 3"

    out = tmp_path / "again.inp"
    write(poly, out)
    assert read(out).global_attrs["abaqus_heading"] == "turbine blade, rev 3"


def test_a_heading_survives_a_trip_through_the_vtk_xml_family(tmp_path) -> None:
    """A <FieldData> block holds a String array beside its numeric ones, so
    the one card describing the model reaches a .vtu and comes home."""
    path = tmp_path / "titled.inp"
    path.write_text(_deck("*Heading\nturbine blade, rev 3\n"))
    through = tmp_path / "through.vtu"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polyxios.write(read(path), through)
    back = tmp_path / "back.inp"
    write(polyxios.read(through), back)

    assert read(back).global_attrs["abaqus_heading"] == "turbine blade, rev 3"


def test_a_heading_a_legacy_vtk_cannot_hold_is_reported(tmp_path) -> None:
    """A FIELD block spells numbers and nothing else, so the heading is lost
    there - said out loud, the way every other unwritable value is."""
    path = tmp_path / "titled.inp"
    path.write_text(_deck("*Heading\nturbine blade, rev 3\n"))

    with pytest.warns(UserWarning, match=r"global_attrs \['abaqus_heading'\]"):
        polyxios.write(read(path), tmp_path / "flat.vtk")


def test_a_heading_of_several_lines_keeps_them_all(tmp_path) -> None:
    path = tmp_path / "long.inp"
    path.write_text(_deck("*Heading\nturbine blade\nrev 3, cold run\n"))

    assert read(path).global_attrs["abaqus_heading"] == (
        "turbine blade\nrev 3, cold run"
    )


def test_a_heading_line_ending_in_a_comma_is_not_joined_to_the_next(
    tmp_path,
) -> None:
    """A trailing comma continues a data line onto the one below it. A
    heading holds no data, so a comma ending one of its lines is punctuation,
    and joining the two handed back one line where the deck spelled two."""
    path = tmp_path / "comma.inp"
    path.write_text(_deck("*Heading\nturbine blade,\nrev 3 of the cold run\n"))

    poly = read(path)
    assert poly.global_attrs["abaqus_heading"] == (
        "turbine blade,\nrev 3 of the cold run"
    )

    out = tmp_path / "again.inp"
    write(poly, out)
    assert read(out).global_attrs["abaqus_heading"] == (
        "turbine blade,\nrev 3 of the cold run"
    )


def test_a_deck_with_no_heading_of_its_own_records_none(tmp_path) -> None:
    """polyxios writes its banner as a comment, so a round trip does not grow
    a heading the caller never asked for."""
    path = tmp_path / "plain.inp"
    path.write_text(_deck(""))

    poly = read(path)
    assert "abaqus_heading" not in poly.global_attrs

    out = tmp_path / "written.inp"
    write(poly, out)
    assert "abaqus_heading" not in read(out).global_attrs


def test_a_heading_that_is_not_text_falls_back_to_the_banner(tmp_path) -> None:
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.global_attrs["abaqus_heading"] = 7
    path = tmp_path / "odd.inp"

    write(poly, path)

    assert "** exported by polyxios" in path.read_text()
    assert "abaqus_heading" not in read(path).global_attrs


# ---------------------------------------------------------------------------
# *SURFACE: a named side of an element
# ---------------------------------------------------------------------------


_TET_DECK = """*Node
1, 0., 0., 0.
2, 1., 0., 0.
3, 0., 1., 0.
4, 0., 0., 1.
*Element, type=C3D4
1, 1, 2, 3, 4
*Elset, elset=solid
1,
"""


def _abaqus_face_nodes() -> dict[str, list[tuple[int, ...]]]:
    """The nodes Abaqus gives each face of each solid, zero-based.

    Straight out of the element library: the faces of C3D4, C3D8, C3D6 and
    C3D5 in the order their S<n> labels number them.
    """
    return {
        "tetra": [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)],
        "hexahedron": [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ],
        "wedge": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
        "pyramid": [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
    }


def test_the_face_label_table_names_the_faces_abaqus_names() -> None:
    """S3 of a C3D4 is a face polyxios also holds, under another number. The
    table between the two numberings is checked against the element library
    rather than against itself."""
    from polyxios._element_types import ELEMENT_FACES
    from polyxios.codecs._abaqus import _ABAQUS_FACE_ORDER

    for type_name, faces in _abaqus_face_nodes().items():
        order = _ABAQUS_FACE_ORDER[type_name]
        assert len(order) == len(faces), type_name
        for label, nodes in enumerate(faces, start=1):
            local = order[label - 1]
            assert set(ELEMENT_FACES[type_name][local]) == set(nodes), (
                f"{type_name} S{label}"
            )


def test_a_surface_becomes_the_face_it_names(tmp_path) -> None:
    """polyxios holds no face set, so a named side of a solid is read as the
    triangle it describes - tagged with the surface's name, and carrying the
    element it is a face of."""
    path = tmp_path / "surf.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=inlet\nsolid, S1\n")

    poly = read(path)

    assert len(poly.element_types) == 2
    assert poly.element_types[1] == ELEMENT_TYPES["triangle"]
    np.testing.assert_array_equal(poly.element_tags["inlet"], [1])
    np.testing.assert_array_equal(poly.element_attrs["face_parent"], [-1, 0])
    # S1 of a tetrahedron is the base, which polyxios numbers last.
    np.testing.assert_array_equal(poly.element_attrs["face_index"], [-1, 3])
    np.testing.assert_array_equal(poly.connectivity[4:], [0, 2, 1])


def test_a_surface_may_name_one_element_rather_than_a_set(tmp_path) -> None:
    path = tmp_path / "one.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=outlet\n1, S2\n")

    poly = read(path)

    np.testing.assert_array_equal(poly.element_tags["outlet"], [1])
    np.testing.assert_array_equal(poly.element_attrs["face_index"], [-1, 0])


def test_a_surface_may_be_declared_before_the_set_it_names(tmp_path) -> None:
    """A deck is free to put the *Surface card above its *Elset."""
    path = tmp_path / "early.inp"
    path.write_text("*Surface, type=ELEMENT, name=inlet\nsolid, S1\n" + _TET_DECK)

    assert read(path).element_tags["inlet"].tolist() == [1]


def test_a_node_surface_becomes_a_vertex_tag(tmp_path) -> None:
    path = tmp_path / "nodes.inp"
    path.write_text(
        _TET_DECK + "*Nset, nset=top\n4,\n*Surface, type=NODE, name=lid\ntop, 1.\n"
    )

    np.testing.assert_array_equal(read(path).vertex_tags["lid"], [3])


def test_a_shell_side_tags_the_element_itself(tmp_path) -> None:
    """SPOS and SNEG name a side of a shell, which is the element."""
    path = tmp_path / "shell.inp"
    path.write_text(
        "*Node\n1, 0., 0., 0.\n2, 1., 0., 0.\n3, 0., 1., 0.\n"
        "*Element, type=S3\n1, 1, 2, 3\n"
        "*Surface, type=ELEMENT, name=skin\n1, SPOS\n"
    )

    poly = read(path)

    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.element_tags["skin"], [0])
    assert "face_parent" not in poly.element_attrs


def test_a_surface_naming_nothing_the_deck_declares_is_reported(tmp_path) -> None:
    path = tmp_path / "missing.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=inlet\nghost, S1\n")

    with pytest.warns(UserWarning, match="neither a set nor an entity"):
        poly = read(path)

    assert "inlet" not in poly.element_tags


def test_a_face_label_the_element_has_no_face_for_is_reported(tmp_path) -> None:
    path = tmp_path / "toohigh.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=inlet\nsolid, S9\n")

    with pytest.warns(UserWarning, match="no face of that element"):
        poly = read(path)

    # The element itself carries the name rather than the surface being lost.
    np.testing.assert_array_equal(poly.element_tags["inlet"], [0])


def test_a_surface_is_written_back_as_a_surface(tmp_path) -> None:
    path = tmp_path / "surf.inp"
    path.write_text(
        _TET_DECK
        + "*Surface, type=ELEMENT, name=inlet\nsolid, S1\n"
        + "*Surface, type=ELEMENT, name=outlet\n1, S2\n"
    )
    poly = read(path)

    out = tmp_path / "again.inp"
    write(poly, out)
    text = out.read_text()

    assert "*Surface, type=ELEMENT, name=inlet" in text
    assert "inlet_S1, S1" in text
    # The face is the surface, not an element card of its own.
    assert text.count("*Element") == 1
    assert "type=S3" not in text

    back = read(out)
    assert set(back.element_tags) == {"solid", "inlet", "outlet"}
    np.testing.assert_array_equal(
        back.element_attrs["face_index"], poly.element_attrs["face_index"]
    )


def test_a_written_surface_does_not_grow_over_a_round_trip(tmp_path) -> None:
    """The elsets a surface generates are marked internal, and are read for
    the surface's sake and then dropped - or every trip would add a group."""
    path = tmp_path / "surf.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=inlet\nsolid, S1\n")

    first = tmp_path / "first.inp"
    write(read(path), first)
    second = tmp_path / "second.inp"
    write(read(first), second)

    assert first.read_text() == second.read_text()
    assert set(read(second).element_tags) == {"solid", "inlet"}


def test_a_face_whose_parent_moved_is_written_as_an_element(tmp_path) -> None:
    """A transform moves a mesh out from under the columns without either
    being wrong. The vertices are what settle it: a face that is no longer
    its parent's is the ordinary element it has become."""
    path = tmp_path / "surf.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=inlet\nsolid, S1\n")
    poly = read(path)

    # Point the column at an element that is not the face's parent.
    poly.element_attrs["face_parent"][1] = 1
    out = tmp_path / "stale.inp"
    write(poly, out)
    text = out.read_text()

    assert "*Surface" not in text
    assert "type=S3" in text


def test_a_group_mixing_faces_and_elements_stays_an_elset(tmp_path) -> None:
    """A deck cannot name one group both ways, so the group that is not all
    faces goes back as the *Elset it is."""
    path = tmp_path / "mixed.inp"
    path.write_text(
        _TET_DECK
        + "*Surface, type=ELEMENT, name=inlet\nsolid, S1\n"
        + "*Elset, elset=inlet\n1,\n"
    )
    poly = read(path)

    out = tmp_path / "again.inp"
    write(poly, out)
    text = out.read_text()

    assert "*Surface" not in text
    assert "*Elset, elset=inlet" in text


def test_a_surface_row_of_nothing_but_separators_is_skipped(tmp_path) -> None:
    """A row that names no entity at all reached the resolver as an empty one
    and took the read down with an IndexError."""
    path = tmp_path / "blank.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=top\n,\n")

    poly = read(path)

    assert len(poly.element_types) == 1
    assert "top" not in poly.element_tags


def test_a_side_named_twice_is_one_face(tmp_path) -> None:
    """Two elements of the same three vertices are one face the deck said
    twice, and the second is geometry the file never had."""
    path = tmp_path / "twice.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=top\n1, S1\n1, S1\n")

    poly = read(path)

    assert len(poly.element_types) == 2
    np.testing.assert_array_equal(poly.element_tags["top"], [1])


def test_a_node_surface_keeps_every_set_a_joined_row_names(tmp_path) -> None:
    """A row ending in a comma continues onto the next, so a node surface
    arrives with every set it named on one row - and each is a member, where
    the token after a set used to be read as a face label and dropped."""
    path = tmp_path / "nodes.inp"
    path.write_text(_TET_DECK + "*Surface, type=NODE, name=lid\n1,\n2,\n")

    np.testing.assert_array_equal(read(path).vertex_tags["lid"], [0, 1])


def test_a_surface_row_naming_more_than_one_pair_is_reported(tmp_path) -> None:
    """An element surface names one set and one face label per row; the rest
    of a joined row is read from and would go without a word."""
    path = tmp_path / "joined.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=top\n1, S1,\n1, S2\n")

    with pytest.warns(UserWarning, match="more than one set"):
        poly = read(path)

    np.testing.assert_array_equal(poly.element_tags["top"], [1])


def test_a_face_a_second_group_names_is_written_as_an_element(tmp_path) -> None:
    """A face on a surface is not written as an element card, so a second
    group naming it would spell an *Elset of an element the deck never
    defines - a deck Abaqus refuses to load."""
    path = tmp_path / "shared.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=top\n1, S1\n")
    poly = read(path)
    poly.element_tags["both"] = np.array([0, 1], dtype=np.int32)

    out = tmp_path / "again.inp"
    write(poly, out)
    text = out.read_text()

    assert "*Surface" not in text
    assert "type=S3" in text
    # Every id an *Elset names is one an *Element card defines.
    back = read(out)
    np.testing.assert_array_equal(back.element_tags["both"], [0, 1])


def test_a_generated_set_does_not_take_a_name_the_mesh_already_uses(
    tmp_path,
) -> None:
    """The sets a surface generates are marked internal, and a reader drops
    those - so one landing on a caller's group would drop the group with it."""
    path = tmp_path / "clash.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=top\n1, S1\n")
    poly = read(path)
    poly.element_tags["top_S1"] = np.array([0], dtype=np.int32)

    out = tmp_path / "again.inp"
    write(poly, out)
    back = read(out)

    np.testing.assert_array_equal(back.element_tags["top_S1"], [0])
    np.testing.assert_array_equal(back.element_tags["top"], [1])


def test_a_surface_group_short_of_a_member_goes_back_as_an_elset(tmp_path) -> None:
    """A *Surface names its faces through their parents and has nowhere to
    say one of them reached no element; an *Elset counts them and reports it,
    so the loss is worth the card."""
    path = tmp_path / "stale.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=top\n1, S1\n")
    poly = read(path)
    poly.element_tags["top"] = np.array([1, 99], dtype=np.int32)

    out = tmp_path / "again.inp"
    with pytest.warns(UserWarning, match="index no element"):
        write(poly, out)
    text = out.read_text()

    assert "*Surface" not in text
    assert "type=S3" in text
    np.testing.assert_array_equal(read(out).element_tags["top"], [1])


def test_a_surface_named_like_a_generated_set_survives_the_read(tmp_path) -> None:
    """The two share a folded key, so dropping the sets a surface generated
    used to take the surface named after one of them with it."""
    path = tmp_path / "named.inp"
    path.write_text(
        _TET_DECK
        + "*Elset, elset=top_S1, internal\n1\n"
        + "*Surface, type=ELEMENT, name=top_S1\n1, S2\n"
        + "*Surface, type=ELEMENT, name=top\n1, S1\n"
    )

    poly = read(path)

    assert "top_S1" in poly.element_tags
    assert "top" in poly.element_tags


def test_a_generated_set_does_not_take_the_name_of_a_surface(tmp_path) -> None:
    """A generated set is marked internal and a reader drops those; one
    landing on a surface's own name would drop the surface. The name a
    surface generates for another is exactly the shape that collides."""
    path = tmp_path / "clash.inp"
    path.write_text(
        _TET_DECK
        + "*Surface, type=ELEMENT, name=top\n1, S1\n"
        + "*Surface, type=ELEMENT, name=top_S1\n1, S2\n"
    )
    poly = read(path)

    out = tmp_path / "again.inp"
    write(poly, out)
    text = out.read_text()

    assert "elset=top_S1, internal" not in text
    back = read(out)
    np.testing.assert_array_equal(back.element_tags["top"], poly.element_tags["top"])
    np.testing.assert_array_equal(
        back.element_tags["top_S1"], poly.element_tags["top_S1"]
    )


def test_two_surfaces_spelling_one_card_name_are_reported(tmp_path) -> None:
    """A name is folded before it is written and Abaqus matches one without
    regard to case, so two groups may reach the same *Surface name. Every face
    still reaches the file - the deck holds both cards and Abaqus merges them,
    and the generated *Elsets are kept apart - but the two surfaces stop being
    told apart, which is the loss *Elset already reports for the groups that
    are not surfaces."""
    path = tmp_path / "two.inp"
    path.write_text(
        _TET_DECK
        + "*Surface, type=ELEMENT, name=top\n1, S1\n"
        + "*Surface, type=ELEMENT, name=side\n1, S2\n"
    )
    poly = read(path)
    poly.element_tags["a*b"] = poly.element_tags.pop("top")
    poly.element_tags["a,b"] = poly.element_tags.pop("side")

    out = tmp_path / "again.inp"
    with pytest.warns(UserWarning, match="one card name between them"):
        write(poly, out)
    text = out.read_text()

    # Both cards are written, and the sets they name are still told apart.
    assert text.count("*Surface, type=ELEMENT, name=a_b\n") == 2
    assert "elset=a_b_S1, internal" in text
    assert "elset=a_b_S2, internal" in text
    back = read(out)
    np.testing.assert_array_equal(np.sort(back.element_tags["a_b"]), [1, 2])


def test_a_mesh_holding_no_tag_mapping_at_all_still_writes(tmp_path) -> None:
    """Nothing stops a caller building a PolyData with the mapping set to
    None, and every other read of it in this codec answers the empty one."""
    bare = dataclasses.replace(_tet_mesh(), element_tags=None)

    out = tmp_path / "bare.inp"
    write(bare, out)

    assert read(out).element_types.size == 1


def test_a_node_surface_reads_its_member_and_not_its_weight_factor(
    tmp_path,
) -> None:
    """An Abaqus node-surface data line is one member and, after it, an
    optional weight factor - and a whole number is as good a weight as a node
    id, so the second field is not a second member."""
    path = tmp_path / "weighted.inp"
    path.write_text(_TET_DECK + "*Surface, type=NODE, name=lid\n1, 2\n")

    poly = read(path)

    np.testing.assert_array_equal(poly.vertex_tags["lid"], [0])


def test_a_node_surface_continued_over_lines_keeps_every_member(tmp_path) -> None:
    """A trailing comma joins the next line onto this one, which is the only
    way several members reach one row - and there they are all members."""
    path = tmp_path / "joined.inp"
    path.write_text(_TET_DECK + "*Surface, type=NODE, name=lid\n1,\n2,\n3\n")

    poly = read(path)

    np.testing.assert_array_equal(poly.vertex_tags["lid"], [0, 1, 2])


def test_a_surface_survives_a_round_trip_through_a_legacy_vtk(tmp_path) -> None:
    """A legacy .vtk cell array is a double whatever it was handed, so the
    two columns that link a face to its parent come back as floats; a deck
    written from that mesh still puts the *Surface card back."""
    path = tmp_path / "deck.inp"
    path.write_text(_TET_DECK + "*Surface, type=ELEMENT, name=inlet\nsolid, S1\n")
    poly = read(path)

    middle = tmp_path / "middle.vtk"
    polyxios.write(poly, str(middle))
    back = polyxios.read(str(middle))
    assert back.element_attrs["face_parent"].dtype.kind == "f"

    out = tmp_path / "again.inp"
    write(back, out)
    text = out.read_text()

    assert "*Surface, type=ELEMENT, name=inlet" in text
    assert read(out).element_tags["inlet"].size == 1


def test_an_internal_set_no_surface_names_is_kept(tmp_path) -> None:
    """The internal sets a surface generates are scaffolding and go once it
    has been read out of them. One no surface names is not scaffolding: it
    used to go with them, and without a word."""
    path = tmp_path / "internal.inp"
    path.write_text(_TET_DECK + "*Elset, elset=keepme, internal\n1,\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(path)

    assert set(poly.element_tags) == {"solid", "keepme"}
    assert poly.element_tags["keepme"].tolist() == [0]


def test_an_internal_set_a_surface_names_still_goes(tmp_path) -> None:
    """The other side of the same rule: a set the surface read its members
    out of is written again on the way out, and would double every trip."""
    path = tmp_path / "generated.inp"
    path.write_text(
        _TET_DECK
        + "*Elset, elset=inlet_S1, internal\n1,\n"
        + "*Surface, type=ELEMENT, name=inlet\ninlet_S1, S1\n"
    )

    poly = read(path)

    assert set(poly.element_tags) == {"solid", "inlet"}


def test_a_heading_that_would_start_a_card_is_refused(tmp_path) -> None:
    """global_attrs travels through every format with a metadata slot, so a
    heading may come back holding anything. A line starting with * ends the
    card and begins the next one: written unchanged, a title of *Node and a
    row below it spells the deck a vertex it never had."""
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.global_attrs["abaqus_heading"] = "rev 3\n*Node\n99, 5., 5., 5."
    path = tmp_path / "injected.inp"

    with pytest.warns(UserWarning, match="start a card"):
        write(poly, path)

    back = read(path)
    # The card line is what goes; the row under it is ordinary text and
    # stays on the heading, where it spells nothing.
    assert back.vertices.shape[0] == 3
    assert back.global_attrs["abaqus_heading"] == "rev 3\n99, 5., 5., 5."


def test_a_heading_holding_a_comment_marker_is_refused(tmp_path) -> None:
    """** opens a comment anywhere on a line, so a title holding one comes
    back truncated at it and one starting with it comes back as nothing."""
    verts = np.arange(9, dtype=np.float64).reshape(3, 3)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.global_attrs["abaqus_heading"] = "run ** 3"
    path = tmp_path / "commented.inp"

    with pytest.warns(UserWarning, match="open a comment"):
        write(poly, path)

    assert "** exported by polyxios" in path.read_text()
    assert "abaqus_heading" not in read(path).global_attrs


def test_a_node_surface_reads_no_weight_factor_as_a_node(tmp_path) -> None:
    """A node surface row is one member and an optional weight factor. The
    join that a trailing comma makes takes the line break with it, so a whole
    number after a set name would be read as a node of the surface - which is
    a node the deck never named."""
    path = tmp_path / "weighted.inp"
    path.write_text(
        _TET_DECK + "*Nset, nset=lid\n2,\n*Surface, type=NODE, name=skin\nlid,\n1\n"
    )

    poly = read(path)

    assert poly.vertex_tags["skin"].tolist() == [1]


def test_a_node_surface_still_keeps_a_joined_row_of_node_numbers(tmp_path) -> None:
    """The other half of the same rule: a row that opens with a node number
    rather than a set name is a row of node numbers, and every one of them is
    a member."""
    path = tmp_path / "numbered.inp"
    path.write_text(_TET_DECK + "*Surface, type=NODE, name=skin\n1,\n2,\n")

    poly = read(path)

    assert poly.vertex_tags["skin"].tolist() == [0, 1]


def test_a_solid_on_another_element_s_face_stays_a_solid(tmp_path) -> None:
    """A tetra's four nodes can be exactly a hexahedron's face. The columns
    say it is one, and its vertices agree - but a face this library built is
    the quad its ring spells, so writing the tetra as a *Surface row would
    cost the deck a volume element and hand back a quad."""
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    poly = make_polydata(
        verts,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    poly.element_attrs["face_parent"] = np.array([-1, 0], dtype=np.int32)
    poly.element_attrs["face_index"] = np.array([-1, 0], dtype=np.int32)
    poly.element_tags["skin"] = np.array([1], dtype=np.int32)
    path = tmp_path / "coincident.inp"

    write(poly, path)

    assert "*Surface" not in path.read_text()
    back = read(path)
    assert [ELEMENT_TYPES_INV[int(t)] for t in back.element_types] == [
        "hexahedron",
        "tetra",
    ]
