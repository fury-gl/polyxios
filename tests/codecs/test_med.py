"""MED: Salome's and code_aster's HDF5 mesh file.

Every test here needs h5py and skips without it, but one, which stands in
for the missing package to check the refusal names the extra that installs
it. Files are built with h5py directly, spelling the layout the MED library
writes, so the reader is tested against the format and not against the
writer beside it.
"""

from __future__ import annotations

import io
from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES_INV
from polyxios._optpkg import TripWire
from polyxios.codecs import _hdf5, _med
from polyxios.codecs._med import read, write
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError

h5py = pytest.importorskip("h5py")

_NO_STEP = "-0000000000000000001-0000000000000000001"
_SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)


def _mesh_attrs(group, dim: int) -> None:
    group.attrs["DIM"] = np.int32(dim)
    group.attrs["ESP"] = np.int32(dim)
    group.attrs["TYP"] = np.int32(0)
    group.attrs["REP"] = np.int32(0)
    group.attrs["SRT"] = np.int32(1)
    group.attrs["DES"] = np.bytes_("test".ljust(200))
    group.attrs["UNT"] = np.bytes_(" " * 16)
    group.attrs["NOM"] = np.bytes_("".join(f"{a:<16}" for a in "XYZ"[:dim]))
    group.attrs["UNI"] = np.bytes_(" " * 16 * dim)


def _put_mesh(
    f,
    name: str,
    points: np.ndarray,
    cells: dict[str, np.ndarray],
    *,
    node_fam: np.ndarray | None = None,
    cell_fam: dict[str, np.ndarray] | None = None,
    node_num: np.ndarray | None = None,
    stepped: bool = True,
) -> None:
    """Spell one mesh as the MED library does: F-ordered COO and NOD, 1-based."""
    dim = points.shape[1]
    mesh = f.require_group("ENS_MAA").create_group(name)
    _mesh_attrs(mesh, dim)
    holder = mesh.create_group(_NO_STEP) if stepped else mesh
    if stepped:
        holder.attrs["NDT"] = np.int32(-1)
        holder.attrs["NOR"] = np.int32(-1)
        holder.attrs["PDT"] = np.float64(-1.0)
    noe = holder.create_group("NOE")
    coo = noe.create_dataset("COO", data=points.ravel(order="F"))
    coo.attrs["NBR"] = np.int32(len(points))
    if node_fam is not None:
        noe.create_dataset("FAM", data=np.asarray(node_fam, dtype=np.int32)).attrs[
            "NBR"
        ] = np.int32(len(points))
    if node_num is not None:
        noe.create_dataset("NUM", data=np.asarray(node_num, dtype=np.int32)).attrs[
            "NBR"
        ] = np.int32(len(points))
    mai = holder.create_group("MAI")
    for med_type, conn in cells.items():
        conn = np.asarray(conn)
        block = mai.create_group(med_type)
        nod = block.create_dataset(
            "NOD", data=(conn + 1).ravel(order="F").astype(np.int32)
        )
        nod.attrs["NBR"] = np.int32(len(conn))
        if cell_fam and med_type in cell_fam:
            fam = block.create_dataset(
                "FAM", data=np.asarray(cell_fam[med_type], dtype=np.int32)
            )
            fam.attrs["NBR"] = np.int32(len(conn))
    fas = f.require_group("FAS").create_group(name)
    fas.create_group("FAMILLE_ZERO").attrs["NUM"] = np.int32(0)


def _put_family(
    f, mesh: str, kind: str, hdf_name: str, number: int, groups: list[str] | None
) -> None:
    """A family under NOEUD or ELEME; ``groups=None`` spells no GRO at all."""
    fam = f["FAS"][mesh].require_group(kind).create_group(hdf_name)
    fam.attrs["NUM"] = np.int32(number)
    if groups is None:
        return
    gro = fam.create_group("GRO")
    gro.attrs["NBR"] = np.int32(len(groups))
    rows = np.zeros((len(groups), 80), dtype=np.int8)
    for i, text in enumerate(groups):
        raw = text.encode()
        rows[i, : len(raw)] = np.frombuffer(raw, dtype=np.int8)
    ds = gro.create_dataset("NOM", (len(groups),), dtype="80int8")
    ds[...] = rows


def _put_field(
    f,
    name: str,
    mesh: str,
    supports: dict[str, np.ndarray],
    *,
    profile: str | None = None,
    time: float = 0.0,
) -> None:
    """A field with one step; ``supports`` maps ``NOE``/``MAI.TR3``/... to (n, nga, nco) values."""
    field = f.require_group("CHA").create_group(name)
    field.attrs["MAI"] = np.bytes_(mesh)
    field.attrs["TYP"] = np.int32(6)
    step = field.create_group("0000000000000000000100000000000000000001")
    step.attrs["NDT"] = np.int32(1)
    step.attrs["NOR"] = np.int32(1)
    step.attrs["PDT"] = np.float64(time)
    for support, values in supports.items():
        values = np.asarray(values, dtype=np.float64)
        node = step.create_group(support)
        pfl = profile or "MED_NO_PROFILE_INTERNAL"
        node.attrs["PFL"] = np.bytes_(pfl)
        node.attrs["GAU"] = np.bytes_("")
        block = node.create_group(pfl)
        block.attrs["NBR"] = np.int32(values.shape[0])
        block.attrs["NGA"] = np.int32(values.shape[1])
        block.create_dataset("CO", data=values.ravel(order="F"))


def _square(path: Path, **kwargs) -> Path:
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"TR3": [[0, 1, 2], [0, 2, 3]]}, **kwargs)
    return path


# ---------------------------------------------------------------------------
# Reading the layout
# ---------------------------------------------------------------------------


def test_a_mesh_reads_with_its_name_and_its_cells(tmp_path: Path) -> None:
    poly = read(_square(tmp_path / "m.med"))
    np.testing.assert_array_equal(poly.vertices, _SQUARE)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 2, 3])
    assert poly.global_attrs["mesh_name"] == "m"
    assert poly.topological_dimension == 2


def test_an_old_layout_without_a_step_group_reads_the_same(tmp_path: Path) -> None:
    poly = read(_square(tmp_path / "m.med", stepped=False))
    assert len(poly.element_types) == 2


def test_a_two_dimensional_file_is_padded_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "flat.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE[:, :2], {"QU4": [[0, 1, 2, 3]]})
    poly = read(path)
    assert poly.vertices.shape == (4, 3)
    assert poly.global_attrs["was_2d"] is True
    np.testing.assert_array_equal(poly.vertices[:, 2], 0)


def test_node_numbers_off_the_index_land_in_original_ids(tmp_path: Path) -> None:
    poly = read(_square(tmp_path / "m.med", node_num=[10, 20, 30, 40]))
    np.testing.assert_array_equal(poly.vertex_attrs["original_ids"], [10, 20, 30, 40])


def test_a_structured_mesh_is_refused_by_name(tmp_path: Path) -> None:
    path = _square(tmp_path / "grid.med")
    with h5py.File(path, "a") as f:
        f["ENS_MAA/m"].attrs["TYP"] = np.int32(1)
    with pytest.raises(CodecError, match="structured"):
        read(path)


def test_a_cell_geometry_polyxios_cannot_hold_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = _square(tmp_path / "poly.med")
    with h5py.File(path, "a") as f:
        f["ENS_MAA/m"][_NO_STEP]["MAI"].create_group("POE")
    with pytest.warns(UserWarning, match=r"\['POE'\]"):
        poly = read(path)
    assert len(poly.element_types) == 2


def test_a_count_larger_than_the_dataset_is_refused(tmp_path: Path) -> None:
    path = _square(tmp_path / "m.med")
    with h5py.File(path, "a") as f:
        f["ENS_MAA/m"][_NO_STEP]["MAI/TR3/NOD"].attrs["NBR"] = np.int32(99)
    with pytest.raises(CodecError, match="NBR=99"):
        read(path)


def test_a_count_short_of_the_dataset_reads_the_first_entities_whole(
    tmp_path: Path,
) -> None:
    """COO and NOD are component-major, so cutting the flat values to the
    declared count would mix one node's X with another's Y; the block is
    shaped by what it holds and cut afterwards."""
    points = np.arange(15.0).reshape(5, 3)
    path = tmp_path / "short.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", points, {"TR3": [[0, 1, 2], [2, 3, 4]]})
        f["ENS_MAA/m"][_NO_STEP]["NOE/COO"].attrs["NBR"] = np.int32(4)
        f["ENS_MAA/m"][_NO_STEP]["MAI/TR3/NOD"].attrs["NBR"] = np.int32(1)
    poly = read(path)
    np.testing.assert_array_equal(poly.vertices, points[:4])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_a_family_or_number_dataset_short_of_its_entities_is_ignored_with_a_warning(
    tmp_path: Path,
) -> None:
    path = _square(tmp_path / "short_fam.med", node_fam=[1, 1, 1, 1], node_num=[1, 2])
    with h5py.File(path, "a") as f:
        f["ENS_MAA/m"][_NO_STEP]["NOE/NUM"].attrs["NBR"] = np.int32(4)
        block = f["ENS_MAA/m"][_NO_STEP]["MAI/TR3"]
        block.create_dataset("FAM", data=np.array([-1], dtype=np.int32))
    with pytest.warns(UserWarning) as record:
        poly = read(path)
    messages = sorted(str(w.message) for w in record)
    assert any("NOE/NUM' holds 2 values for 4 nodes" in m for m in messages)
    assert any("MAI/TR3/FAM' holds 1 values for 2 cells" in m for m in messages)
    assert "original_ids" not in poly.vertex_attrs
    assert poly.element_tags == {}
    assert poly.vertex_tags["family_1"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("step", ["1", 1.5, True, -1])
def test_a_step_that_is_not_a_whole_number_from_zero_is_refused(
    tmp_path: Path, step
) -> None:
    with pytest.raises(CodecError, match="step="):
        read(_square(tmp_path / "m.med"), step=step)


def test_a_node_outside_the_mesh_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"TR3": [[0, 1, 7]]})
    with pytest.raises(CodecError, match="outside 1..4"):
        read(path)


def test_lazy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_square(tmp_path / "m.med"), lazy=True)


def test_a_file_that_is_not_hdf5_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "text.med"
    path.write_text("not hdf5\n")
    with pytest.raises(CodecError, match="not an HDF5 file"):
        read(path)


def test_without_h5py_the_read_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    path = _square(tmp_path / "m.med")
    monkeypatch.setattr(_hdf5, "_h5py", lambda: (TripWire("no h5py"), False))
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
        read(path)
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
        write(make_polydata(_SQUARE, [("quad", np.array([[0, 1, 2, 3]]))]), path)


# ---------------------------------------------------------------------------
# Families and groups
# ---------------------------------------------------------------------------


def test_issue_1050_groups_are_tags_and_a_rebuilt_mesh_writes_them_back(
    tmp_path: Path,
) -> None:
    """A round trip through a rebuilt mesh lost the group names: they lived
    on an attribute the mesh object did not carry. Here they are tag groups,
    and a family listing two groups puts its members in both."""
    path = tmp_path / "groups.med"
    with h5py.File(path, "w") as f:
        _put_mesh(
            f,
            "m",
            _SQUARE,
            {"TR3": [[0, 1, 2], [0, 2, 3]]},
            node_fam=[1, 2, 2, 0],
            cell_fam={"TR3": [-1, 0]},
        )
        _put_family(f, "m", "NOEUD", "left", 1, ["A"])
        _put_family(f, "m", "NOEUD", "both", 2, ["B", "C"])
        _put_family(f, "m", "ELEME", "first", -1, ["D"])
    poly = read(path)
    assert {k: v.tolist() for k, v in poly.vertex_tags.items()} == {
        "A": [0],
        "B": [1, 2],
        "C": [1, 2],
    }
    assert {k: v.tolist() for k, v in poly.element_tags.items()} == {"D": [0]}

    rebuilt = make_polydata(
        poly.vertices,
        [("triangle", poly.connectivity.reshape(-1, 3))],
        vertex_tags=poly.vertex_tags,
        element_tags=poly.element_tags,
    )
    out = tmp_path / "back.med"
    write(rebuilt, out)
    with h5py.File(out) as f:
        node_fam = f["ENS_MAA/mesh"][_NO_STEP]["NOE/FAM"][()]
        cell_fam = f["ENS_MAA/mesh"][_NO_STEP]["MAI/TR3/FAM"][()]
        families = {
            int(fam.attrs["NUM"]): sorted(
                bytes(row.astype(np.uint8)).rstrip(b"\x00").decode()
                for row in fam["GRO/NOM"][()]
            )
            for kind in ("NOEUD", "ELEME")
            for fam in f["FAS/mesh"][kind].values()
        }
    assert families[int(node_fam[1])] == ["B", "C"]
    assert families[int(node_fam[0])] == ["A"]
    assert node_fam[3] == 0
    assert families[int(cell_fam[0])] == ["D"]
    assert cell_fam[1] == 0
    assert all(n > 0 for n in node_fam[:3]) and cell_fam[0] < 0
    again = read(out)
    assert {k: v.tolist() for k, v in again.vertex_tags.items()} == {
        "A": [0],
        "B": [1, 2],
        "C": [1, 2],
    }
    assert again.element_tags["D"].tolist() == [0]


def test_issue_1541_a_family_with_no_groups_reads_without_a_gro(tmp_path: Path) -> None:
    """Gmsh writes one family per elementary entity, and one that belongs to
    no physical group has no GRO subgroup at all; the reader asked for it
    and raised KeyError. Such a family keeps its number as a tag."""
    path = tmp_path / "gmsh.med"
    with h5py.File(path, "w") as f:
        _put_mesh(
            f, "m", _SQUARE, {"TR3": [[0, 1, 2], [0, 2, 3]]}, cell_fam={"TR3": [-1, -1]}
        )
        _put_family(f, "m", "ELEME", "F_2D_1", -1, None)
    poly = read(path)
    assert poly.element_tags["family_-1"].tolist() == [0, 1]


def test_issue_1541_a_file_without_fas_reads_with_no_tags(tmp_path: Path) -> None:
    path = _square(tmp_path / "nofas.med", cell_fam={"TR3": [-1, -1]})
    with h5py.File(path, "a") as f:
        del f["FAS"]
    poly = read(path)
    assert {k: v.tolist() for k, v in poly.element_tags.items()} == {
        "family_-1": [0, 1]
    }


def test_issue_1541_elements_of_one_type_are_one_group_however_they_are_ordered(
    tmp_path: Path,
) -> None:
    """A reader that yields one block per entity handed the writer two
    triangle blocks, which it refused. polyxios holds a flat element list
    and writes one MAI group per geometry whatever the order."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[1, 4, 2]])),
        ],
        element_attrs={"e": np.array([1.0, 2.0, 3.0])},
        element_tags={"skin": np.array([0, 2])},
    )
    path = tmp_path / "interleaved.med"
    write(poly, path)
    with h5py.File(path) as f:
        assert sorted(f["ENS_MAA/mesh"][_NO_STEP]["MAI"]) == ["TE4", "TR3"]
        assert f["ENS_MAA/mesh"][_NO_STEP]["MAI/TR3/NOD"].attrs["NBR"] == 2
    back = read(path)
    assert back.element_types.tolist() == [5, 5, 10]
    np.testing.assert_array_equal(back.element_attrs["e"], [1.0, 3.0, 2.0])
    assert back.element_tags["skin"].tolist() == [0, 1]


def test_issue_1541_a_write_that_fails_leaves_no_file_behind(
    tmp_path: Path, monkeypatch
) -> None:
    """The refusal came after the nodes were written, leaving a file with no
    FAS that the next read fell over."""

    def boom(*args, **kwargs):
        raise RuntimeError("halfway")

    monkeypatch.setattr(_med, "_write_families", boom)
    path = tmp_path / "half.med"
    poly = make_polydata(
        _SQUARE, [("quad", np.array([[0, 1, 2, 3]]))], vertex_tags={"g": np.array([0])}
    )
    with pytest.raises(RuntimeError):
        write(poly, path)
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_issue_1133_several_meshes_merge_tagged_by_name_or_one_is_picked(
    tmp_path: Path,
) -> None:
    """A file holding two meshes was refused outright. Every mesh is read,
    each one's elements tagged with its name, and ``mesh=`` picks one; a
    field says which mesh it belongs to and follows it."""
    path = tmp_path / "two.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "A", _SQUARE, {"QU4": [[0, 1, 2, 3]]})
        _put_mesh(f, "B", _SQUARE, {"TR3": [[0, 1, 2], [0, 2, 3]]})
        _put_field(f, "f", "B", {"NOE": np.arange(4.0)[:, None, None]})
    both = read(path)
    assert len(both.element_types) == 3
    assert both.element_tags["A"].tolist() == [0]
    assert both.element_tags["B"].tolist() == [1, 2]
    assert "mesh_name" not in both.global_attrs
    only = read(path, mesh="B")
    assert len(only.element_types) == 2
    assert only.global_attrs["mesh_name"] == "B"
    np.testing.assert_array_equal(only.vertex_attrs["f"], [0, 1, 2, 3])
    with pytest.raises(CodecError, match=r"\['A', 'B'\]"):
        read(path, mesh="nope")


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("med_type", "ptype", "n"),
    [
        ("TR7", "biquadratic_triangle", 7),
        ("QU9", "biquadratic_quad", 9),
        ("SE4", "cubic_line", 4),
    ],
)
def test_issue_1277_the_biquadratic_faces_read_and_round_trip(
    tmp_path: Path, med_type, ptype, n
) -> None:
    """TR7 and QU9 were not in the type table, and a file holding them was
    met with a KeyError naming the geometry."""
    verts = np.random.default_rng(0).random((n, 3))
    path = tmp_path / f"{med_type}.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", verts, {med_type: [list(range(n))]})
    poly = read(path)
    assert ELEMENT_TYPES_INV[int(poly.element_types[0])] == ptype
    out = tmp_path / "back.med"
    write(poly, out)
    with h5py.File(out) as f:
        assert list(f["ENS_MAA/m"][_NO_STEP]["MAI"]) == [med_type]
    np.testing.assert_array_equal(read(out).connectivity, np.arange(n))


def test_issue_1488_a_pyramid_reads_as_a_pyramid(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], dtype=np.float64
    )
    path = tmp_path / "py.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", verts, {"PY5": [[0, 1, 2, 3, 4]]})
    poly = read(path)
    assert ELEMENT_TYPES_INV[int(poly.element_types[0])] == "pyramid"


def test_the_27_node_hexahedron_s_face_centres_are_reordered_both_ways(
    tmp_path: Path,
) -> None:
    verts = np.random.default_rng(1).random((27, 3))
    med_order = list(range(27))
    path = tmp_path / "h27.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", verts, {"H27": [med_order]})
    poly = read(path)
    # MED's bottom face centre (20) is VTK's z-min face (24), and so on.
    assert poly.connectivity.tolist() == [*range(20), 24, 22, 21, 23, 20, 25, 26]
    out = tmp_path / "back.med"
    write(poly, out)
    with h5py.File(out) as f:
        nod = f["ENS_MAA/m"][_NO_STEP]["MAI/H27/NOD"][()].reshape(1, 27, order="F") - 1
    assert nod[0].tolist() == med_order


def test_per_node_values_follow_the_27_node_hexahedron_s_face_centres(
    tmp_path: Path,
) -> None:
    """An ELNO field lists one value per node in the cell's own order, so
    the values are permuted with the connectivity or they land on the wrong
    node: MED's bottom face centre is VTK's z-min one."""
    verts = np.random.default_rng(2).random((27, 3))
    path = tmp_path / "h27_elno.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", verts, {"H27": [list(range(27))]})
        _put_field(f, "f", "m", {"NOE.H27": np.arange(27.0).reshape(1, 27, 1)})
    poly = read(path)
    # Value i sat on MED node i, which the connectivity now lists at the
    # position VTK gives that node; the value follows.
    np.testing.assert_array_equal(poly.element_attrs["f"][0], poly.connectivity)
    # A third axis the size of the node count is what the writer files as
    # ELNO; a (1, 27) array would be a 27-component cell value.
    elno = make_polydata(
        poly.vertices,
        [("triquadratic_hexahedron", poly.connectivity.reshape(1, 27))],
        element_attrs={"f": poly.element_attrs["f"][:, :, None]},
    )
    out = tmp_path / "back.med"
    write(elno, out)
    with h5py.File(out) as f:
        co = f["CHA/f"][_NO_STEP]["NOE.H27/MED_NO_PROFILE_INTERNAL/CO"][()]
    np.testing.assert_array_equal(co, np.arange(27.0))
    np.testing.assert_array_equal(read(out).element_attrs["f"], poly.element_attrs["f"])


def test_per_node_values_on_a_voxel_follow_its_corners(tmp_path: Path) -> None:
    verts = np.array(
        [[i, j, k] for k in range(2) for j in range(2) for i in range(2)],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [("voxel", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))],
        element_attrs={"e": np.arange(8.0).reshape(1, 8, 1)},
    )
    path = tmp_path / "voxel_elno.med"
    write(poly, path)
    back = read(path)
    # The hexahedron's node j is the voxel's node conn[j]; value e[j] was
    # written at that node, so it still names it.
    np.testing.assert_array_equal(back.element_attrs["e"][0], back.connectivity)


def test_a_pixel_and_a_voxel_go_out_as_a_quad_and_a_hexahedron(tmp_path: Path) -> None:
    verts = np.array(
        [[i, j, k] for k in range(2) for j in range(2) for i in range(2)],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("pixel", np.array([[0, 1, 2, 3]])),
            ("voxel", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        ],
    )
    path = tmp_path / "lattice.med"
    write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [9, 12]
    assert back.connectivity.tolist() == [0, 1, 3, 2, 0, 1, 3, 2, 4, 5, 7, 6]


def test_an_element_type_med_cannot_name_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("polygon", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"e": np.array([1.0, 2.0])},
    )
    path = tmp_path / "drop.med"
    with pytest.warns(UserWarning, match=r"\['polygon'\]"):
        write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [5]
    np.testing.assert_array_equal(back.element_attrs["e"], [2.0])


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def test_issue_1279_a_field_on_some_of_the_geometries_is_nan_on_the_rest(
    tmp_path: Path,
) -> None:
    """A code_aster result defined on the hexahedra and the segments but not
    the quads left a None where the quads' block should be, and the mesh
    fell over measuring it. Also the trace of issue 1403. The missing
    geometry reads as NaN, and Gauss-point values keep their axis."""
    path = tmp_path / "partial.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"TR3": [[0, 1, 2]], "QU4": [[0, 1, 2, 3]]})
        _put_field(f, "f", "m", {"MAI.TR3": np.array([[[1.0]]])})
        _put_field(f, "g", "m", {"MAI.TR3": np.array([[[1.0], [2.0]]])})
        _put_field(
            f,
            "h",
            "m",
            {"MAI.TR3": np.array([[[1.0, 2.0]]]), "MAI.QU4": np.array([[[3.0, 4.0]]])},
        )
    poly = read(path)
    # Blocks read by ascending polyxios code: triangle (5) before quad (9).
    assert poly.element_types.tolist() == [5, 9]
    np.testing.assert_array_equal(poly.element_attrs["f"], [1.0, np.nan])
    assert poly.element_attrs["g"].shape == (2, 2)
    np.testing.assert_array_equal(poly.element_attrs["g"][0], [1.0, 2.0])
    assert np.isnan(poly.element_attrs["g"][1]).all()
    np.testing.assert_array_equal(poly.element_attrs["h"], [[1.0, 2.0], [3.0, 4.0]])
    assert poly.global_attrs["time"] == 0.0


def test_a_nodal_field_on_a_profile_is_spread_with_nan_elsewhere(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"QU4": [[0, 1, 2, 3]]})
        _put_field(
            f,
            "f",
            "m",
            {"NOE": np.array([[[5.0]], [[7.0]]])},
            profile="PROFIL__00000001",
        )
        prof = f.create_group("PROFILS/PROFIL__00000001")
        prof.attrs["NBR"] = np.int32(2)
        prof.create_dataset("PFL", data=np.array([2, 4], dtype=np.int32))
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["f"], [np.nan, 5.0, np.nan, 7.0])


def test_a_profile_that_stops_short_of_the_last_node_still_spreads_over_all(
    tmp_path: Path,
) -> None:
    """A profile says which entities carry values and nothing about how many
    there are; spreading to the largest index it names left a field short
    of the mesh whenever the last node was not in it, and the field was
    dropped as the wrong length."""
    path = tmp_path / "short_profile.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"QU4": [[0, 1, 2, 3]]})
        _put_field(
            f,
            "f",
            "m",
            {"NOE": np.array([[[5.0]], [[7.0]]])},
            profile="PROFIL__00000001",
        )
        prof = f.create_group("PROFILS/PROFIL__00000001")
        prof.attrs["NBR"] = np.int32(2)
        prof.create_dataset("PFL", data=np.array([1, 2], dtype=np.int32))
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["f"], [5.0, 7.0, np.nan, np.nan])
    with h5py.File(path, "a") as f:
        f["PROFILS/PROFIL__00000001/PFL"][...] = np.array([1, 9], dtype=np.int32)
    with pytest.raises(CodecError, match="does not index its support of 4"):
        read(path)


def test_cell_values_win_over_per_node_values_on_one_geometry(tmp_path: Path) -> None:
    path = tmp_path / "both.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"TR3": [[0, 1, 2], [0, 2, 3]]})
        _put_field(
            f,
            "f",
            "m",
            {
                "NOE.TR3": np.arange(6.0).reshape(2, 3, 1),
                "MAI.TR3": np.array([[[1.0]], [[2.0]]]),
            },
        )
    with pytest.warns(UserWarning, match="both cell and per-node values on TR3"):
        poly = read(path)
    np.testing.assert_array_equal(poly.element_attrs["f"], [1.0, 2.0])


def test_a_vertex_and_an_element_attribute_of_one_name_are_one_field(
    tmp_path: Path,
) -> None:
    """MED files a field over the nodes and the cells together, and two
    groups of one name cannot exist anyway."""
    poly = make_polydata(
        _SQUARE,
        [("quad", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"v": np.arange(4.0), "w": np.arange(4.0)},
        element_attrs={"v": np.array([9.0]), "w": np.array([[1.0, 2.0]])},
    )
    path = tmp_path / "shared.med"
    with pytest.warns(UserWarning, match=r"\['w'\] are written under \['w_2'\]"):
        write(poly, path)
    with h5py.File(path) as f:
        assert sorted(f["CHA"]) == ["v", "w", "w_2"]
        assert sorted(f["CHA/v"][_NO_STEP]) == ["MAI.QU4", "NOE"]
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["v"], np.arange(4.0))
    np.testing.assert_array_equal(back.element_attrs["v"], [9.0])
    np.testing.assert_array_equal(back.element_attrs["w_2"], [[1.0, 2.0]])


def test_a_slash_in_a_name_does_not_nest_a_group(tmp_path: Path) -> None:
    """An HDF5 link cannot hold '/', so a tag, mesh or field named with one
    would have landed as a group inside a group the reader never looks in;
    the family's own name carries the real text, the others are spelled
    with '_' and said so."""
    poly = make_polydata(
        _SQUARE,
        [("quad", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"a/b": np.arange(4.0)},
        vertex_tags={"left/right": np.array([0, 1])},
        element_tags={"in/out": np.array([0])},
        global_attrs={"mesh_name": "top/bottom"},
    )
    path = tmp_path / "slash.med"
    with pytest.warns(UserWarning, match=r"\['a/b'\] are written under \['a_b'\]"):
        write(poly, path)
    with h5py.File(path) as f:
        assert list(f["ENS_MAA"]) == ["top_bottom"]
        assert list(f["CHA"]) == ["a_b"]
        assert "/" not in "".join(f["FAS/top_bottom/NOEUD"])
    back = read(path)
    assert back.global_attrs["mesh_name"] == "top_bottom"
    assert back.vertex_tags["left/right"].tolist() == [0, 1]
    assert back.element_tags["in/out"].tolist() == [0]
    np.testing.assert_array_equal(back.vertex_attrs["a_b"], np.arange(4.0))


def test_global_attrs_med_cannot_hold_are_warned_about_once(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("quad", np.array([[0, 1, 2, 3]]))],
        global_attrs={"n": 3, "label": "x", "time": 1.0},
    )
    with pytest.warns(UserWarning) as record:
        write(poly, tmp_path / "globals.med")
    about = [str(w.message) for w in record if "global_attrs" in str(w.message)]
    assert about == [
        ".med write: global_attrs ['label', 'n'] have no place in a MED file; dropped."
    ]


def test_a_field_with_several_steps_reads_at_the_step_asked(tmp_path: Path) -> None:
    path = tmp_path / "steps.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "m", _SQUARE, {"QU4": [[0, 1, 2, 3]]})
        _put_field(f, "f", "m", {"NOE": np.zeros((4, 1, 1))}, time=0.0)
        step = f["CHA/f"].create_group("0000000000000000000200000000000000000002")
        step.attrs["PDT"] = np.float64(0.5)
        node = step.create_group("NOE")
        node.attrs["PFL"] = np.bytes_("MED_NO_PROFILE_INTERNAL")
        block = node.create_group("MED_NO_PROFILE_INTERNAL")
        block.attrs["NBR"] = np.int32(4)
        block.attrs["NGA"] = np.int32(1)
        block.create_dataset("CO", data=np.ones(4))
    first = read(path)
    np.testing.assert_array_equal(first.vertex_attrs["f"], 0)
    second = read(path, step=1)
    np.testing.assert_array_equal(second.vertex_attrs["f"], 1)
    assert second.global_attrs["time"] == 0.5


def test_written_fields_carry_the_time_and_per_node_values_go_out_as_elno(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"s": np.arange(4.0), "v": np.arange(12.0).reshape(4, 3)},
        element_attrs={
            "per_node": np.arange(12.0).reshape(2, 3, 2),
            "count": np.array([1, 2], dtype=np.int64),
        },
        global_attrs={"time": 2.5},
    )
    path = tmp_path / "fields.med"
    write(poly, path)
    with h5py.File(path) as f:
        step = f["CHA/per_node"]["0000000000000000000100000000000000000001"]
        assert list(step) == ["NOE.TR3"]
        assert step["NOE.TR3/MED_NO_PROFILE_INTERNAL"].attrs["NGA"] == 3
        assert f["CHA/per_node"].attrs["NCO"] == 2
        assert step.attrs["PDT"] == 2.5
        assert f["CHA/count"].attrs["TYP"] == 26
        assert f["CHA/v"].attrs["NCO"] == 3
    back = read(path)
    np.testing.assert_array_equal(
        back.element_attrs["per_node"], poly.element_attrs["per_node"]
    )
    np.testing.assert_array_equal(back.element_attrs["count"], [1, 2])
    np.testing.assert_array_equal(back.vertex_attrs["v"], poly.vertex_attrs["v"])
    assert back.global_attrs["time"] == 2.5


def test_a_gauss_point_count_that_is_not_the_node_count_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"gauss": np.zeros((1, 4, 2))},
    )
    with pytest.warns(UserWarning, match="neither one nor its node count"):
        write(poly, tmp_path / "gauss.med")
    assert "gauss" not in read(tmp_path / "gauss.med").element_attrs


# ---------------------------------------------------------------------------
# Writing the layout
# ---------------------------------------------------------------------------


def test_issue_1357_the_mesh_carries_every_attribute_the_library_reads(
    tmp_path: Path,
) -> None:
    """A file without the axis-name attribute was refused by Gmsh, Salome and
    mdump alike: the library's mesh-info call reads nine attributes and
    fails on the first one missing."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts, [("triangle", np.array([[0, 1, 2]])), ("line", np.array([[0, 3]]))]
    )
    path = tmp_path / "attrs.med"
    write(poly, path, mesh_name="shell")
    with h5py.File(path) as f:
        mesh = f["ENS_MAA/shell"]
        for key in ("ESP", "DIM", "TYP", "DES", "UNT", "SRT", "REP", "NOM", "UNI"):
            assert key in mesh.attrs, key
        assert mesh.attrs["ESP"] == 3
        # The mesh dimension is the cells', not the space's: a shell is 2-D.
        assert mesh.attrs["DIM"] == 2
        assert len(mesh.attrs["NOM"]) == len(mesh.attrs["UNI"]) == 48
        assert mesh.attrs["NOM"][:16].strip() == b"X"
        assert f["INFOS_GENERALES"].attrs["MAJ"] == 3
        assert f["ENS_MAA/shell"][_NO_STEP]["MAI/TR3"].attrs["GEO"] == 203
        assert f["ENS_MAA/shell"][_NO_STEP]["MAI/SE2"].attrs["GEO"] == 102
        assert "FAMILLE_ZERO" in f["FAS/shell"]
    assert read(path).global_attrs["mesh_name"] == "shell"


def test_issue_1484_a_med_mesh_of_two_geometries_writes_as_gmsh_4_1(
    tmp_path: Path,
) -> None:
    """A hexahedron with one quad of its skin could not be written as Gmsh
    4.1 for want of entity information MED never carries."""
    verts = np.array(
        [[i, j, k] for k in range(2) for j in range(2) for i in range(2)],
        dtype=np.float64,
    )
    path = tmp_path / "cube.med"
    with h5py.File(path, "w") as f:
        _put_mesh(
            f, "m", verts, {"HE8": [[0, 1, 3, 2, 4, 5, 7, 6]], "QU4": [[0, 1, 3, 2]]}
        )
    poly = polyxios.read(path)
    out = tmp_path / "cube.msh"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polyxios.write(poly, out)
    back = polyxios.read(out)
    assert sorted(back.element_types.tolist()) == [9, 12]


def test_a_buffer_and_a_gzip_name_both_work(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("quad", np.array([[0, 1, 2, 3]]))], vertex_tags={"g": np.array([1])}
    )
    buffer = io.BytesIO()
    polyxios.write(poly, buffer, fmt=".med")
    buffer.seek(0)
    back = polyxios.read(buffer, fmt=".med")
    assert back.vertex_tags["g"].tolist() == [1]
    packed = tmp_path / "m.med.gz"
    polyxios.write(poly, packed)
    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert polyxios.read(packed).vertex_tags["g"].tolist() == [1]


def test_unknown_options_are_warned_about(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("quad", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(poly, tmp_path / "o.med", bogus=1)
    with pytest.warns(UserWarning, match="unrecognized options"):
        read(tmp_path / "o.med", bogus=1)


def test_the_codec_is_registered_for_med() -> None:
    assert ".med" in polyxios.supported_extensions()


# ---------------------------------------------------------------------------
# Edges of the writer
# ---------------------------------------------------------------------------


def test_an_empty_mesh_with_attributes_reads_back_empty(tmp_path: Path) -> None:
    """A field over no node holds no value; the reader took that for a
    field whose values did not divide by its entities."""
    poly = make_polydata(
        np.zeros((0, 3)),
        [],
        vertex_attrs={"s": np.zeros(0), "v": np.zeros((0, 3))},
    )
    path = tmp_path / "empty.med"
    write(poly, path)
    back = read(path)
    assert len(back.vertices) == 0
    assert back.vertex_attrs["s"].shape == (0,)
    assert back.vertex_attrs["v"].shape == (0, 3)


def test_an_attribute_that_is_nan_on_every_element_is_warned_about(
    tmp_path: Path,
) -> None:
    """NaN is a MED field's "no value"; a field of nothing but that would
    read back as a value, so it is left out - and said so, not silently."""
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        element_attrs={"gone": np.full(2, np.nan), "kept": np.array([1.0, np.nan])},
    )
    path = tmp_path / "nan.med"
    with pytest.warns(UserWarning, match=r"\['gone'\] are NaN on every element"):
        write(poly, path)
    back = read(path)
    assert sorted(back.element_attrs) == ["kept"]
    np.testing.assert_array_equal(back.element_attrs["kept"], [1.0, np.nan])


def test_compression_is_passed_to_every_dataset(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"s": np.arange(4.0)},
        vertex_tags={"g": np.array([0, 1])},
    )
    path = tmp_path / "gz.med"
    write(poly, path, compression="gzip", compression_opts=4)
    with h5py.File(path) as f:
        step = f["ENS_MAA/mesh"][_NO_STEP]
        assert step["NOE/COO"].compression == "gzip"
        assert step["NOE/FAM"].compression == "gzip"
        assert step["MAI/TR3/NOD"].compression == "gzip"
        assert f["CHA/s"][_NO_STEP]["NOE/MED_NO_PROFILE_INTERNAL/CO"].compression == (
            "gzip"
        )
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["s"], np.arange(4.0))
    assert back.vertex_tags["g"].tolist() == [0, 1]
    with pytest.raises(CodecError, match="compression_opts names a level"):
        write(poly, tmp_path / "level.med", compression_opts=4)


def test_a_mesh_named_like_a_group_of_another_keeps_both_tags(tmp_path: Path) -> None:
    path = tmp_path / "clash.med"
    with h5py.File(path, "w") as f:
        _put_mesh(f, "A", _SQUARE, {"QU4": [[0, 1, 2, 3]]}, cell_fam={"QU4": [-1]})
        _put_family(f, "A", "ELEME", "FAM_-1_B", -1, ["B"])
        _put_mesh(f, "B", _SQUARE, {"TR3": [[0, 1, 2], [0, 2, 3]]})
    both = read(path)
    assert both.element_tags["A"].tolist() == [0]
    assert both.element_tags["B"].tolist() == [0]
    assert both.element_tags["B_"].tolist() == [1, 2]
