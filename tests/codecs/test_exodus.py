"""Exodus II: Sandia's finite element database, a netCDF file.

Every test here needs netCDF4 and skips without it. Files are built with
netCDF4 directly, in the layout the Exodus library writes, so the reader is
tested against the format and not against the writer beside it.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import struct
import sys
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_FACES
from polyxios._optpkg import TripWire
from polyxios.codecs import _exodus
from polyxios.codecs._exodus import (
    _READ_ORDER,
    _SIDE_ORDER,
    _WRITE_ORDER,
    read,
    write,
)
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    UnknownElementTypeError,
    UnsupportedFormatError,
    ValidationError,
)

netCDF4 = pytest.importorskip("netCDF4")

# netCDF4 1.7's own slicing sets an array's shape, which numpy 2.5 deprecates;
# the codec shields its calls, and the files built here need the same.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_CUBE = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)


def _chars(names: list[str], width: int = 33) -> np.ndarray:
    out = np.zeros((len(names), width), dtype="S1")
    for i, text in enumerate(names):
        raw = text.encode()
        out[i, : len(raw)] = np.frombuffer(raw, dtype="S1")
    return out


class _File:
    """A minimal Exodus file the way the library lays it out."""

    def __init__(
        self,
        path: Path,
        points: np.ndarray,
        *,
        dim: int = 3,
        fmt="NETCDF3_64BIT_OFFSET",
    ):
        self.f = netCDF4.Dataset(path, "w", format=fmt)
        f = self.f
        f.setncattr("api_version", np.float32(8.03))
        f.setncattr("version", np.float32(8.03))
        f.setncattr("floating_point_word_size", np.int32(8))
        f.setncattr("file_size", np.int32(1))
        f.setncattr("title", "")
        f.createDimension("len_string", 33)
        f.createDimension("len_name", 33)
        f.createDimension("time_step", None)
        f.createDimension("num_dim", dim)
        f.createDimension("num_nodes", len(points))
        coord = f.createVariable("coord", "f8", ("num_dim", "num_nodes"))
        coord[:] = np.asarray(points, dtype=np.float64)[:, :dim].T
        self.blocks: list[tuple[str, np.ndarray]] = []
        self.node_sets: list[tuple[str, list[int]]] = []
        self.side_sets: list[tuple[str, list[int], list[int]]] = []
        self.elem_sets: list[tuple[str, list[int]]] = []

    def block(self, elem_type: str, cells, *, name: str = "") -> _File:
        self.blocks.append((elem_type, np.asarray(cells, dtype=np.int64), name))
        return self

    def node_set(self, name: str, nodes: list[int]) -> _File:
        self.node_sets.append((name, nodes))
        return self

    def side_set(self, name: str, elems: list[int], sides: list[int]) -> _File:
        self.side_sets.append((name, elems, sides))
        return self

    def elem_set(self, name: str, elems: list[int]) -> _File:
        self.elem_sets.append((name, elems))
        return self

    def finish(self) -> netCDF4.Dataset:
        """Write the blocks and sets; the handle stays open for more."""
        f = self.f
        n_elems = sum(len(c) for _, c, _ in self.blocks)
        f.createDimension("num_elem", n_elems)
        f.createDimension("num_el_blk", len(self.blocks))
        f.createVariable("eb_prop1", "i4", ("num_el_blk",))[:] = np.arange(
            1, len(self.blocks) + 1
        )
        f.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))[:] = _chars(
            [name for _, _, name in self.blocks]
        )
        for j, (elem_type, cells, _) in enumerate(self.blocks, start=1):
            f.createDimension(f"num_el_in_blk{j}", len(cells))
            f.createDimension(f"num_nod_per_el{j}", cells.shape[1])
            var = f.createVariable(
                f"connect{j}", "i4", (f"num_el_in_blk{j}", f"num_nod_per_el{j}")
            )
            var.setncattr("elem_type", elem_type)
            var[:] = cells + 1
        if self.node_sets:
            f.createDimension("num_node_sets", len(self.node_sets))
            f.createVariable("ns_prop1", "i4", ("num_node_sets",))[:] = np.arange(
                1, len(self.node_sets) + 1
            )
            f.createVariable("ns_names", "S1", ("num_node_sets", "len_name"))[:] = (
                _chars([name for name, _ in self.node_sets])
            )
            for i, (_, nodes) in enumerate(self.node_sets, start=1):
                f.createDimension(f"num_nod_ns{i}", len(nodes))
                f.createVariable(f"node_ns{i}", "i4", (f"num_nod_ns{i}",))[:] = (
                    np.asarray(nodes) + 1
                )
        if self.side_sets:
            f.createDimension("num_side_sets", len(self.side_sets))
            f.createVariable("ss_prop1", "i4", ("num_side_sets",))[:] = np.arange(
                1, len(self.side_sets) + 1
            )
            f.createVariable("ss_names", "S1", ("num_side_sets", "len_name"))[:] = (
                _chars([name for name, _, _ in self.side_sets])
            )
            for i, (_, elems, sides) in enumerate(self.side_sets, start=1):
                f.createDimension(f"num_side_ss{i}", len(elems))
                f.createVariable(f"elem_ss{i}", "i4", (f"num_side_ss{i}",))[:] = (
                    np.asarray(elems) + 1
                )
                f.createVariable(f"side_ss{i}", "i4", (f"num_side_ss{i}",))[:] = (
                    np.asarray(sides)
                )
        if self.elem_sets:
            f.createDimension("num_elem_sets", len(self.elem_sets))
            f.createVariable("els_prop1", "i4", ("num_elem_sets",))[:] = np.arange(
                1, len(self.elem_sets) + 1
            )
            f.createVariable("els_names", "S1", ("num_elem_sets", "len_name"))[:] = (
                _chars([name for name, _ in self.elem_sets])
            )
            for i, (_, elems) in enumerate(self.elem_sets, start=1):
                f.createDimension(f"num_ele_els{i}", len(elems))
                f.createVariable(f"elem_els{i}", "i4", (f"num_ele_els{i}",))[:] = (
                    np.asarray(elems) + 1
                )
        return f


def _tet_file(path: Path) -> netCDF4.Dataset:
    return _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]], name="solid").finish()


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_block_reads_as_its_elements_and_its_name_as_a_tag(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]], name="solid").block(
        "TRI3", [[0, 1, 2], [0, 1, 3]], name="skin"
    ).finish().close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertices, _TET)
    assert poly.element_types.tolist() == [10, 5, 5]
    assert poly.element_tags["solid"].tolist() == [0]
    assert poly.element_tags["skin"].tolist() == [1, 2]
    assert poly.global_attrs == {}


def test_an_unnamed_block_is_tagged_by_its_id(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]]).finish()
    f.variables["eb_prop1"][:] = [7]
    f.close()
    assert read(path).element_tags == {"block_7": np.array([0])}


def test_a_name_ends_at_its_nul_whatever_follows(tmp_path: Path) -> None:
    """A name written over a longer one keeps the old tail past its NUL."""
    path = tmp_path / "m.e"
    f = (
        _File(path, _TET)
        .block("TETRA4", [[0, 1, 2, 3]], name="solid_and_more")
        .finish()
    )
    f.variables["eb_names"][0, :6] = np.frombuffer(b"abc\x00xy", dtype="S1")
    f.close()
    assert list(read(path).element_tags) == ["abc"]


def test_a_step_out_of_range_is_refused_before_the_mesh_is_decoded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.e"
    f = _File(path, _TET).block("TETRA4", [[0, 1, 2, 9]]).finish()
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.close()
    with pytest.raises(CodecError, match="1 time step"):
        read(path, step=3)


def test_coordinates_may_come_as_three_arrays(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET")
    f.createDimension("num_dim", 3)
    f.createDimension("num_nodes", 4)
    f.createDimension("num_el_blk", 1)
    f.createDimension("num_el_in_blk1", 1)
    f.createDimension("num_nod_per_el1", 4)
    for axis, column in zip("xyz", _TET.T):
        f.createVariable(f"coord{axis}", "f8", ("num_nodes",))[:] = column
    var = f.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
    var.setncattr("elem_type", "TETRA")
    var[:] = [[1, 2, 3, 4]]
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertices, _TET)
    assert poly.element_types.tolist() == [10]


def test_a_family_name_is_resolved_by_its_node_count(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA", [[0, 1, 2, 3]]).block(
        "TRIANGLE", [[0, 1, 2]]
    ).block("SHELL", [[0, 1, 2, 3]]).block("BEAM", [[0, 1]]).block(
        "SPHERE", [[0]]
    ).finish().close()
    assert read(path).element_types.tolist() == [10, 5, 9, 3, 1]


def test_a_family_exodus_does_not_define_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("DODECA", [[0, 1, 2, 3]]).finish().close()
    with pytest.raises(UnknownElementTypeError, match="DODECA"):
        read(path)


def test_a_shape_polyxios_cannot_hold_is_skipped_with_its_sets(tmp_path: Path) -> None:
    """HEX9, WEDGE16, PYRAMID14 and TETRA14 are Exodus shapes VTK has no cell for."""
    path = tmp_path / "m.e"
    _File(path, np.vstack([_CUBE, [[0.5, 0.5, 0.5]]])).block(
        "HEX9", [[0, 1, 2, 3, 4, 5, 6, 7, 8]], name="odd"
    ).block("TETRA4", [[0, 1, 3, 4]], name="tet").node_set("all", [0, 8]).elem_set(
        "both", [0, 1]
    ).finish().close()
    with pytest.warns(UserWarning, match=r"HEX9 \(9 nodes\)"):
        poly = read(path)
    assert poly.element_types.tolist() == [10]
    assert sorted(poly.element_tags) == ["both", "tet"]
    assert poly.element_tags["both"].tolist() == [0]
    assert poly.vertex_tags["all"].tolist() == [0, 8]


def test_a_null_block_reads_as_nothing(tmp_path: Path) -> None:
    """The library writes a NULL block as a status of 0 and no table at all."""
    path = tmp_path / "m.e"
    f = netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET")
    f.createDimension("len_name", 33)
    f.createDimension("time_step", None)
    f.createDimension("num_dim", 3)
    f.createDimension("num_nodes", 4)
    f.createVariable("coord", "f8", ("num_dim", "num_nodes"))[:] = _TET.T
    f.createDimension("num_elem", 1)
    f.createDimension("num_el_blk", 3)
    f.createVariable("eb_prop1", "i4", ("num_el_blk",))[:] = [1, 2, 3]
    f.createVariable("eb_status", "i4", ("num_el_blk",))[:] = [0, 1, 0]
    f.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))[:] = _chars(
        ["nothing", "solid", "nothing_either"]
    )
    f.createDimension("num_el_in_blk2", 1)
    f.createDimension("num_nod_per_el2", 4)
    var = f.createVariable("connect2", "i4", ("num_el_in_blk2", "num_nod_per_el2"))
    var.setncattr("elem_type", "TETRA4")
    var[:] = [[1, 2, 3, 4]]
    f.createDimension("num_side_sets", 1)
    f.createVariable("ss_names", "S1", ("num_side_sets", "len_name"))[:] = _chars(
        ["base"]
    )
    f.createDimension("num_side_ss1", 1)
    f.createVariable("elem_ss1", "i4", ("num_side_ss1",))[:] = [1]
    f.createVariable("side_ss1", "i4", ("num_side_ss1",))[:] = [4]
    f.createDimension("num_elem_var", 1)
    f.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))[:] = _chars(
        ["rho"]
    )
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createVariable("vals_elem_var1eb2", "f8", ("time_step", "num_el_in_blk2"))[:] = [
        [7.0]
    ]
    f.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(path)
    assert poly.element_types.tolist() == [10, 5]
    assert list(poly.element_tags) == ["solid", "base"]
    assert poly.element_tags["base"].tolist() == [1]
    assert poly.element_attrs["rho"][0] == 7.0


def test_a_polyhedral_block_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]], name="solid").block(
        "NSIDED", [[0, 1, 2]], name="poly"
    ).elem_set("both", [0, 1]).finish().close()
    with pytest.warns(UserWarning, match=r"NSIDED"):
        poly = read(path)
    assert poly.element_types.tolist() == [10]
    assert sorted(poly.element_tags) == ["both", "solid"]
    assert poly.element_tags["both"].tolist() == [0]


def test_a_connectivity_naming_a_node_the_file_lacks_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA4", [[0, 1, 2, 9]]).finish().close()
    with pytest.raises(CodecError, match="outside 1..4"):
        read(path)


def test_a_node_set_names_vertices_and_an_element_set_elements(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]]).block(
        "TRI3", [[0, 1, 2], [1, 2, 3]]
    ).node_set("base", [0, 1, 2]).elem_set("picked", [2, 0]).finish().close()
    poly = read(path)
    assert {k: v.tolist() for k, v in poly.vertex_tags.items()} == {"base": [0, 1, 2]}
    assert poly.element_tags["picked"].tolist() == [0, 2]


def test_a_set_naming_an_entity_outside_the_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]]).node_set(
        "bad", [7]
    ).finish().close()
    with pytest.raises(CodecError, match="node_ns1"):
        read(path)


def test_a_side_set_reads_as_the_faces_it_names(tmp_path: Path) -> None:
    """Side 6 of a HEX8 is its top face 5-6-7-8 in the Exodus table."""
    path = tmp_path / "m.e"
    _File(path, _CUBE).block("HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]]).side_set(
        "lid", [0, 0], [6, 5]
    ).finish().close()
    poly = read(path)
    assert poly.element_types.tolist() == [12, 9, 9]
    assert set(poly.connectivity[8:12].tolist()) == {4, 5, 6, 7}
    assert set(poly.connectivity[12:16].tolist()) == {0, 1, 2, 3}
    assert poly.element_tags["lid"].tolist() == [1, 2]
    assert poly.element_attrs["face_parent"].tolist() == [-1, 0, 0]
    assert poly.element_attrs["face_index"].tolist() == [-1, 1, 0]


def test_a_side_named_twice_is_one_face_in_both_groups(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _CUBE).block("HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]]).side_set(
        "one", [0], [6]
    ).side_set("two", [0, 0], [6, 6]).finish().close()
    poly = read(path)
    assert len(poly.element_types) == 2
    assert poly.element_tags["one"].tolist() == [1]
    assert poly.element_tags["two"].tolist() == [1]


def test_a_side_number_the_element_lacks_is_skipped_with_its_own_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.e"
    _File(path, _CUBE).block("HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]]).side_set(
        "odd", [0, 0], [7, 6]
    ).finish().close()
    with pytest.warns(UserWarning, match=r"\['odd'\].*no side of"):
        poly = read(path)
    assert poly.element_tags["odd"].tolist() == [1]
    assert set(poly.connectivity[8:12].tolist()) == {4, 5, 6, 7}


def test_side_sets_over_many_blocks_keep_first_mention_order(tmp_path: Path) -> None:
    """Two hex blocks and a wedge block, sets naming sides across all three:
    every face is read once, triangles before quads, each in the order met."""
    points = np.vstack([_CUBE, _CUBE + [2, 0, 0], _CUBE[:6] + [4, 0, 0]])
    hexes = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    wedge = [[16, 17, 18, 19, 20, 21]]
    path = tmp_path / "m.e"
    _File(path, points).block("HEX8", hexes[:1], name="a").block(
        "WEDGE6", wedge, name="w"
    ).block("HEX8", hexes[1:], name="b").side_set(
        "s1", [2, 1, 1, 0], [6, 4, 1, 6]
    ).side_set("s2", [0, 1, 2], [6, 5, 6]).finish().close()
    poly = read(path)
    # Sides met: hexb top (quad), wedge end (tri), wedge lateral (quad),
    # hexa top (quad), then hexb top again, wedge other end (tri), hexa top.
    assert poly.element_types.tolist() == [12, 13, 12, 5, 5, 9, 9, 9]
    assert poly.element_attrs["face_parent"].tolist() == [-1, -1, -1, 1, 1, 2, 1, 0]
    assert poly.element_tags["s1"].tolist() == [3, 5, 6, 7]
    assert poly.element_tags["s2"].tolist() == [4, 5, 7]
    conn = poly.connectivity
    assert set(conn[22:25].tolist()) == {16, 17, 18}
    assert set(conn[25:28].tolist()) == {19, 20, 21}
    assert set(conn[28:32].tolist()) == {12, 13, 14, 15}
    assert set(conn[36:40].tolist()) == {4, 5, 6, 7}


def test_a_side_of_a_shell_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TRI3", [[0, 1, 2]]).side_set(
        "edge", [0], [1]
    ).finish().close()
    with pytest.warns(UserWarning, match=r"\['edge'\].*not a solid"):
        poly = read(path)
    assert poly.element_types.tolist() == [5]
    assert poly.element_tags["edge"].tolist() == []


@pytest.mark.parametrize("type_name", sorted(_SIDE_ORDER))
def test_the_side_numbering_matches_the_exodus_table(type_name: str) -> None:
    """Table 4.2 of the Exodus manual, 1-based corner nodes per side."""
    table = {
        "tetra": [(1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2)],
        "wedge": [(1, 2, 5, 4), (2, 3, 6, 5), (1, 4, 6, 3), (1, 3, 2), (4, 5, 6)],
        "hexahedron": [
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 4, 8, 7),
            (1, 5, 8, 4),
            (1, 4, 3, 2),
            (5, 6, 7, 8),
        ],
        "pyramid": [(1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 1, 5), (1, 4, 3, 2)],
    }
    # A voxel is written as a HEX8 with its corners reordered, so its sides
    # are the hexahedron's, read through that reorder.
    to_file = {n: n for n in range(8)}
    if type_name == "voxel":
        to_file = {n: pos for pos, n in enumerate(_WRITE_ORDER["voxel"])}
        type_name_table = "hexahedron"
    else:
        type_name_table = type_name
    family = next(k for k in table if type_name_table.endswith(k))
    for side, nodes in enumerate(table[family], start=1):
        local = _SIDE_ORDER[type_name][side - 1]
        corners = {to_file[n] for n in ELEMENT_FACES[type_name][local]}
        assert corners == {n - 1 for n in nodes}, (type_name, side)


@pytest.mark.parametrize("type_name", sorted(_READ_ORDER))
def test_a_read_order_is_a_permutation(type_name: str) -> None:
    order = _READ_ORDER[type_name]
    assert sorted(order) == list(range(len(order)))


def test_issue_1538_a_hex20_keeps_its_edges_where_vtk_expects_them(
    tmp_path: Path,
) -> None:
    """Exodus lists the vertical edges before the top ring; VTK after."""
    corners = _CUBE
    edges = {
        (0, 1): 8,
        (1, 2): 9,
        (2, 3): 10,
        (3, 0): 11,
        (0, 4): 12,
        (1, 5): 13,
        (2, 6): 14,
        (3, 7): 15,
        (4, 5): 16,
        (5, 6): 17,
        (6, 7): 18,
        (7, 4): 19,
    }
    points = np.zeros((20, 3))
    points[:8] = corners
    for (a, b), k in edges.items():
        points[k] = (corners[a] + corners[b]) / 2
    path = tmp_path / "m.e"
    _File(path, points).block("HEX20", [list(range(20))]).finish().close()
    poly = read(path)
    assert poly.element_types.tolist() == [22]
    cells = poly.connectivity
    # VTK: 12-15 are the top ring, 16-19 the vertical edges.
    for k, (a, b) in zip(range(12, 16), [(4, 5), (5, 6), (6, 7), (7, 4)]):
        np.testing.assert_allclose(
            poly.vertices[cells[k]], (corners[a] + corners[b]) / 2
        )
    for k, (a, b) in zip(range(16, 20), [(0, 4), (1, 5), (2, 6), (3, 7)]):
        np.testing.assert_allclose(
            poly.vertices[cells[k]], (corners[a] + corners[b]) / 2
        )
    write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as f:
        np.testing.assert_array_equal(f.variables["connect1"][:], [list(range(1, 21))])


def test_a_hex27_face_centre_lands_on_its_face(tmp_path: Path) -> None:
    """Exodus node 22 is the bottom face centre, 23 the top, 24 to 27 the sides;
    VTK puts x-min, x-max, y-min, y-max, z-min, z-max, centre last."""
    points = np.zeros((27, 3))
    points[:8] = _CUBE
    exodus_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    ]
    for k, (a, b) in enumerate(exodus_edges, start=8):
        points[k] = (_CUBE[a] + _CUBE[b]) / 2
    points[20] = [0.5, 0.5, 0.5]
    points[21] = [0.5, 0.5, 0.0]
    points[22] = [0.5, 0.5, 1.0]
    points[23] = [0.0, 0.5, 0.5]
    points[24] = [1.0, 0.5, 0.5]
    points[25] = [0.5, 0.0, 0.5]
    points[26] = [0.5, 1.0, 0.5]
    path = tmp_path / "m.e"
    _File(path, points).block("HEX27", [list(range(27))]).finish().close()
    poly = read(path)
    cells = poly.connectivity
    expected = {
        20: [0.0, 0.5, 0.5],
        21: [1.0, 0.5, 0.5],
        22: [0.5, 0.0, 0.5],
        23: [0.5, 1.0, 0.5],
        24: [0.5, 0.5, 0.0],
        25: [0.5, 0.5, 1.0],
        26: [0.5, 0.5, 0.5],
    }
    for k, where in expected.items():
        np.testing.assert_allclose(poly.vertices[cells[k]], where)
    write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as f:
        np.testing.assert_array_equal(f.variables["connect1"][:], [list(range(1, 28))])


def test_issue_1436_a_lower_dimensional_block_beside_a_solid_reads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET).block("TETRA4", [[0, 1, 2, 3]]).block("BAR2", [[0, 1]]).block(
        "SPHERE", [[3]]
    ).finish().close()
    poly = read(path)
    assert poly.element_types.tolist() == [10, 3, 1]


def test_variables_at_a_step_are_attributes(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = (
        _File(path, _TET)
        .block("TETRA4", [[0, 1, 2, 3]])
        .block("TRI3", [[0, 1, 2]])
        .finish()
    )
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0, 2.5]
    f.createDimension("num_nod_var", 4)
    f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
        ["phi", "disp_x", "disp_y", "disp_z"]
    )
    for i in range(1, 5):
        var = f.createVariable(f"vals_nod_var{i}", "f8", ("time_step", "num_nodes"))
        var[:] = np.arange(8.0).reshape(2, 4) * i
    f.createDimension("num_elem_var", 1)
    f.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))[:] = _chars(
        ["rho"]
    )
    f.createVariable("elem_var_tab", "i4", ("num_el_blk", "num_elem_var"))[:] = [
        [1],
        [0],
    ]
    var = f.createVariable("vals_elem_var1eb1", "f8", ("time_step", "num_el_in_blk1"))
    var[:] = [[1.0], [2.0]]
    f.createDimension("num_glo_var", 1)
    f.createVariable("name_glo_var", "S1", ("num_glo_var", "len_name"))[:] = _chars(
        ["energy"]
    )
    f.createVariable("vals_glo_var", "f8", ("time_step", "num_glo_var"))[:] = [
        [3.0],
        [4.0],
    ]
    f.close()
    poly = read(path, step=1)
    np.testing.assert_array_equal(poly.vertex_attrs["phi"], [4.0, 5.0, 6.0, 7.0])
    assert poly.vertex_attrs["disp"].shape == (4, 3)
    np.testing.assert_array_equal(
        poly.vertex_attrs["disp"][:, 2], 4 * np.array([4.0, 5.0, 6.0, 7.0])
    )
    rho = poly.element_attrs["rho"]
    assert rho[0] == 2.0 and np.isnan(rho[1])
    assert poly.global_attrs == {"time": 2.5, "energy": 4.0}
    assert read(path).global_attrs == {"time": 0.0, "energy": 3.0}


def test_a_global_table_of_the_wrong_width_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createDimension("num_glo_var", 2)
    f.createDimension("one", 1)
    f.createVariable("name_glo_var", "S1", ("num_glo_var", "len_name"))[:] = _chars(
        ["a", "b"]
    )
    f.createVariable("vals_glo_var", "f8", ("time_step", "one"))[:] = [[3.0]]
    f.close()
    with pytest.warns(UserWarning, match="1 values for 2 global"):
        poly = read(path)
    assert poly.global_attrs == {"time": 0.0}


def test_the_legacy_single_nodal_table_reads(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createDimension("num_nod_var", 2)
    f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
        ["a", "b"]
    )
    var = f.createVariable(
        "vals_nod_var", "f8", ("time_step", "num_nod_var", "num_nodes")
    )
    var[0] = [[1, 2, 3, 4], [5, 6, 7, 8]]
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["b"], [5, 6, 7, 8])


def test_a_step_past_the_last_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.close()
    with pytest.raises(CodecError, match="1 time step"):
        read(path, step=3)


def test_the_title_and_the_number_maps_are_kept(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.setncattr("title", "cube of one")
    f.createVariable("node_num_map", "i4", ("num_nodes",))[:] = [10, 20, 30, 40]
    f.createVariable("elem_num_map", "i4", ("num_elem",))[:] = [7]
    f.close()
    poly = read(path)
    assert poly.global_attrs["title"] == "cube of one"
    assert poly.vertex_attrs["original_ids"].tolist() == [10, 20, 30, 40]
    assert poly.element_attrs["original_ids"].tolist() == [7]
    write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as f:
        assert f.getncattr("title") == "cube of one"
        assert f.variables["node_num_map"][:].tolist() == [10, 20, 30, 40]
        assert f.variables["elem_num_map"][:].tolist() == [7]


def test_element_ids_survive_a_side_set_and_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = (
        _File(path, _CUBE)
        .block("HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]])
        .side_set("lid", [0], [6])
        .finish()
    )
    f.createVariable("elem_num_map", "i4", ("num_elem",))[:] = [42]
    f.createVariable("node_num_map", "i4", ("num_nodes",))[:] = list(range(11, 19))
    f.close()
    poly = read(path)
    assert poly.element_attrs["original_ids"].tolist() == [42, 43]
    write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as g:
        assert g.variables["elem_num_map"][:].tolist() == [42]
        assert g.variables["node_num_map"][:].tolist() == list(range(11, 19))
        assert "num_nod_var" not in g.dimensions
    assert read(tmp_path / "back.e").element_attrs["original_ids"].tolist() == [42, 43]


def test_block_attributes_are_element_attrs(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = (
        _File(path, _TET)
        .block("TETRA4", [[0, 1, 2, 3]])
        .block("TRI3", [[0, 1, 2]])
        .finish()
    )
    f.createDimension("num_att_in_blk1", 1)
    f.createVariable("attrib1", "f8", ("num_el_in_blk1", "num_att_in_blk1"))[:] = [
        [9.0]
    ]
    f.createVariable("attrib_name1", "S1", ("num_att_in_blk1", "len_name"))[:] = _chars(
        ["R"]
    )
    f.close()
    r = read(path).element_attrs["R"]
    assert r[0] == 9.0 and np.isnan(r[1])


def test_a_two_dimensional_file_is_padded_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET[:3], dim=2).block("TRI3", [[0, 1, 2]]).finish().close()
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    assert poly.global_attrs["was_2d"] is True
    write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as f:
        assert f.dimensions["num_dim"].size == 2


def test_a_netcdf4_flavoured_file_reads(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET, fmt="NETCDF4").block("TETRA4", [[0, 1, 2, 3]]).finish().close()
    assert read(path).element_types.tolist() == [10]


def test_a_netcdf4_file_behind_a_user_block_reads(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _TET, fmt="NETCDF4").block("TETRA4", [[0, 1, 2, 3]]).finish().close()
    padded = tmp_path / "padded.e"
    padded.write_bytes(bytes(512) + path.read_bytes())
    assert read(padded).element_types.tolist() == [10]


def test_a_file_that_is_not_netcdf_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    path.write_bytes(b"MeshVersionFormatted 2\n")
    with pytest.raises(CodecError, match="not a netCDF file"):
        read(path)


def test_a_declared_count_no_file_can_hold_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as f:
        f.createDimension("num_dim", 3)
        f.createDimension("num_nodes", 2**40)
    with pytest.raises(ValidationError, match=r"\d{10,}"):
        read(path)


def test_lazy_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _tet_file(path).close()
    with pytest.raises(LazyReadError):
        read(path, lazy=True)


def test_without_netcdf4_the_read_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "m.e"
    _tet_file(path).close()
    monkeypatch.setattr(_exodus, "_netcdf4", lambda: (TripWire("no netCDF4"), False))
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[netcdf\]"):
        read(path)
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[netcdf\]"):
        write(make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))]), path)


def test_every_extension_resolves_to_the_codec(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    for ext in (".e", ".exo", ".ex2"):
        path = tmp_path / f"m{ext}"
        _quiet(polyxios.write, poly, path)
        assert polyxios.read(path).element_types.tolist() == [10]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_a_single_type_group_becomes_a_block_and_the_rest_a_set(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2], [0, 1, 3]])),
        ],
        element_tags={
            "skin": np.array([1, 2]),
            "mixed": np.array([0, 1]),
            "one": np.array([1]),
        },
    )
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        f.set_auto_chartostring(False)
        names = [
            b"".join(r).decode().rstrip("\x00") for r in f.variables["eb_names"][:]
        ]
        assert names == ["skin", "tetra"]
        assert f.variables["connect1"].elem_type == "TRI3"
        assert f.variables["connect2"].elem_type == "TETRA4"
        set_names = [
            b"".join(r).decode().rstrip("\x00") for r in f.variables["els_names"][:]
        ]
        # "one" overlaps the block "skin" already took, so it is a set too.
        assert set_names == ["mixed", "one"]
        assert f.variables["elem_els1"][:].tolist() == [1, 3]
        assert f.variables["elem_els2"][:].tolist() == [1]
    back = read(path)
    assert back.element_types.tolist() == [5, 5, 10]
    assert back.element_tags["mixed"].tolist() == [0, 2]
    assert back.element_tags["one"].tolist() == [0]


def test_a_leftover_block_never_takes_a_name_a_group_holds(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={"tetra": np.array([1])},
    )
    path = tmp_path / "m.e"
    write(poly, path)
    back = read(path)
    assert back.element_tags["tetra"].tolist() == [0]
    assert back.element_tags["tetra_2"].tolist() == [1]
    assert back.element_types.tolist() == [5, 10]


def test_a_face_group_goes_back_as_a_side_set(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, _CUBE).block("HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]]).side_set(
        "lid", [0], [6]
    ).finish().close()
    poly = read(path)
    write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as f:
        assert f.dimensions["num_elem"].size == 1
        assert f.variables["elem_ss1"][:].tolist() == [1]
        assert f.variables["side_ss1"][:].tolist() == [6]
        assert "num_elem_sets" not in f.dimensions
    again = read(tmp_path / "back.e")
    assert again.element_tags["lid"].tolist() == [1]
    assert again.element_attrs["face_index"].tolist() == [-1, 1]


def test_a_face_another_group_names_stays_an_element(tmp_path: Path) -> None:
    poly = make_polydata(
        _CUBE,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("quad", np.array([[4, 5, 6, 7]])),
        ],
        element_attrs={
            "face_parent": np.array([-1, 0]),
            "face_index": np.array([-1, 1]),
        },
        element_tags={"lid": np.array([1]), "everything": np.array([0, 1])},
    )
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        assert f.dimensions["num_elem"].size == 2
        assert f.variables["side_ss1"][:].tolist() == [6]
        assert f.variables["elem_els1"][:].tolist() == [1, 2]


def test_a_face_that_no_longer_is_one_is_an_element(tmp_path: Path) -> None:
    poly = make_polydata(
        _CUBE,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("quad", np.array([[0, 1, 5, 4]])),
        ],
        element_attrs={
            "face_parent": np.array([-1, 0]),
            "face_index": np.array([-1, 1]),
        },
        element_tags={"lid": np.array([1])},
    )
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        assert "num_side_sets" not in f.dimensions
        assert f.dimensions["num_elem"].size == 2


def test_a_vector_attribute_is_split_by_axis_and_folded_back(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={
            "v": np.arange(12.0).reshape(4, 3),
            "t": np.arange(8.0).reshape(4, 2),
        },
        element_attrs={"flag": np.array([True])},
    )
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        f.set_auto_chartostring(False)
        names = [
            b"".join(r).decode().rstrip("\x00") for r in f.variables["name_nod_var"][:]
        ]
        assert names == ["v_x", "v_y", "v_z", "t_0", "t_1"]
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["v"], poly.vertex_attrs["v"])
    np.testing.assert_array_equal(back.vertex_attrs["t"], poly.vertex_attrs["t"])
    np.testing.assert_array_equal(back.element_attrs["flag"], [1.0])


def test_an_attribute_that_is_not_numeric_per_entity_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"name": np.array(["a", "b", "c", "d"])},
    )
    with pytest.warns(UserWarning, match=r"\['name'\]"):
        write(poly, tmp_path / "m.e")


def test_a_text_global_other_than_the_title_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        global_attrs={
            "title": "t",
            "note": "hello",
            "n": 3,
            "arr": np.array([1.0, 2.0]),
        },
    )
    path = tmp_path / "m.e"
    with pytest.warns(UserWarning, match=r"\['note'\]"):
        write(poly, path)
    back = read(path)
    assert back.global_attrs["title"] == "t"
    assert back.global_attrs["n"] == 3.0
    np.testing.assert_array_equal(back.global_attrs["arr"], [1.0, 2.0])


def test_a_type_exodus_has_no_block_for_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]])), ("polygon", np.array([[0, 1, 2, 3]]))],
        element_attrs={"rho": np.array([1.0, 2.0])},
    )
    path = tmp_path / "m.e"
    with pytest.warns(UserWarning, match=r"\['polygon'\]"):
        write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [10]
    np.testing.assert_array_equal(back.element_attrs["rho"], [1.0])


def test_a_pixel_and_a_voxel_go_out_as_quad_and_hex(tmp_path: Path) -> None:
    poly = make_polydata(
        _CUBE,
        [
            ("voxel", np.array([[0, 1, 3, 2, 4, 5, 7, 6]])),
            ("pixel", np.array([[0, 1, 3, 2]])),
        ],
    )
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        # Blocks go out by ascending type code: the pixel's quad first.
        assert f.variables["connect1"].elem_type == "QUAD4"
        assert f.variables["connect1"][:].tolist() == [[1, 2, 3, 4]]
        assert f.variables["connect2"].elem_type == "HEX8"
        assert f.variables["connect2"][:].tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_an_empty_mesh_is_written_as_netcdf4_and_reads_back(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3)), [])
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        assert f.data_model == "NETCDF4"
    back = read(path)
    assert len(back.vertices) == 0 and len(back.element_types) == 0


def test_a_buffer_receives_the_bytes_a_path_would(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "m.e"
    write(poly, path)
    buf = io.BytesIO()
    write(poly, buf)
    assert buf.getvalue() == path.read_bytes()
    buf.seek(0)
    assert read(buf).element_types.tolist() == [10]


def test_a_failed_write_leaves_no_half_file(tmp_path: Path, monkeypatch) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "m.e"
    path.write_bytes(b"old")

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(_exodus, "_partition", boom)
    with pytest.raises(RuntimeError):
        write(poly, path)
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_a_variable_name_two_attributes_claim_is_written_once(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"v": np.arange(12.0).reshape(4, 3), "v_x": np.full(4, 9.0)},
    )
    path = tmp_path / "m.e"
    with pytest.warns(UserWarning, match=r"\['v_x'\].*two attributes"):
        write(poly, path)
    with netCDF4.Dataset(path) as f:
        f.set_auto_chartostring(False)
        names = [
            b"".join(r).decode().rstrip("\x00") for r in f.variables["name_nod_var"][:]
        ]
        assert names == ["v_x", "v_y", "v_z"]
    np.testing.assert_array_equal(read(path).vertex_attrs["v"], poly.vertex_attrs["v"])


def test_a_value_on_a_face_written_as_a_side_is_warned_about(tmp_path: Path) -> None:
    poly = make_polydata(
        _CUBE,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("quad", np.array([[4, 5, 6, 7], [0, 1, 2, 3]])),
        ],
        element_attrs={
            "face_parent": np.array([-1, 0, 0]),
            "face_index": np.array([-1, 1, 0]),
            "flux": np.array([np.nan, 3.0, np.nan]),
            "quiet": np.array([1.0, np.nan, np.nan]),
        },
        element_tags={"lid": np.array([1]), "base": np.array([2])},
    )
    path = tmp_path / "m.e"
    with pytest.warns(UserWarning, match=r"\['flux'\].*side sets"):
        write(poly, path)
    with netCDF4.Dataset(path) as f:
        assert f.dimensions["num_elem"].size == 1
        assert f.dimensions["num_side_sets"].size == 2


def test_a_missing_directory_is_reported_by_the_path_given(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    target = tmp_path / "nowhere" / "m.e"
    with pytest.raises(FileNotFoundError) as info:
        write(poly, target)
    assert info.value.filename == str(target)
    assert ".partial" not in str(info.value)


def test_a_long_name_is_cut_on_a_code_point(tmp_path: Path) -> None:
    label = "é" * 200
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={label: np.array([0])},
    )
    path = tmp_path / "m.e"
    with pytest.warns(UserWarning, match="longer than 255 bytes"):
        write(poly, path)
    (back,) = read(path).element_tags
    assert back == "é" * 127
    assert "�" not in back


def test_a_face_of_a_voxel_goes_out_as_a_side_of_its_hex(tmp_path: Path) -> None:
    poly = make_polydata(
        _CUBE,
        [
            ("voxel", np.array([[0, 1, 3, 2, 4, 5, 7, 6]])),
            ("quad", np.array([[4, 5, 7, 6]])),
        ],
        element_attrs={
            "face_parent": np.array([-1, 0]),
            "face_index": np.array([-1, 1]),
        },
        element_tags={"lid": np.array([1])},
    )
    path = tmp_path / "m.e"
    write(poly, path)
    with netCDF4.Dataset(path) as f:
        assert f.dimensions["num_elem"].size == 1
        assert f.variables["elem_ss1"][:].tolist() == [1]
        assert f.variables["side_ss1"][:].tolist() == [6]
    back = read(path)
    assert back.element_types.tolist() == [12, 9]
    assert set(back.connectivity[8:12].tolist()) == {4, 5, 6, 7}
    assert back.element_tags["lid"].tolist() == [1]


def test_a_leftover_block_never_takes_a_name_a_set_holds(tmp_path: Path) -> None:
    """A set and a block of one name read back as one tag, so the leftover
    tetra block steps aside for the mixed group called ``tetra``."""
    poly = make_polydata(
        _TET,
        [
            ("tetra", np.array([[0, 1, 2, 3], [0, 1, 3, 2]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_tags={"tetra": np.array([0, 2])},
    )
    path = tmp_path / "m.e"
    write(poly, path)
    back = read(path)
    # Blocks go out by type code: the triangle block first, the tetras after.
    assert back.element_types.tolist() == [5, 10, 10]
    assert back.element_tags["tetra"].tolist() == [0, 1]
    assert back.element_tags["tetra_2"].tolist() == [1, 2]


def test_unknown_options_are_warned_about(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(poly, tmp_path / "m.e", colour="red")
    with pytest.warns(UserWarning, match="unrecognized options"):
        read(tmp_path / "m.e", colour="red")


# ---------------------------------------------------------------------------
# What a file can say that the reader must not mangle
# ---------------------------------------------------------------------------


def test_a_block_attribute_beside_a_side_set_is_one_value_per_element(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.e"
    f = (
        _File(path, _CUBE)
        .block("HEX8", [[0, 1, 2, 3, 4, 5, 6, 7]])
        .side_set("lid", [0], [6])
        .finish()
    )
    f.createDimension("num_att_in_blk1", 1)
    f.createVariable("attrib1", "f8", ("num_el_in_blk1", "num_att_in_blk1"))[:] = [
        [9.0]
    ]
    f.createVariable("attrib_name1", "S1", ("num_att_in_blk1", "len_name"))[:] = _chars(
        ["R"]
    )
    f.close()
    poly = read(path)
    r = poly.element_attrs["R"]
    assert len(r) == len(poly.element_types) == 2
    assert r[0] == 9.0 and np.isnan(r[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "back.e")
    with netCDF4.Dataset(tmp_path / "back.e") as g:
        assert g.variables["vals_elem_var1eb1"][0].tolist() == [9.0]


def test_a_legacy_nodal_table_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createDimension("num_nod_var", 2)
    f.createDimension("one", 1)
    f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
        ["a", "b"]
    )
    var = f.createVariable("vals_nod_var", "f8", ("time_step", "one", "num_nodes"))
    var[0] = [[1, 2, 3, 4]]
    f.close()
    with pytest.raises(CodecError, match=r"'vals_nod_var' is \(1, 1, 4\)"):
        read(path)


def test_a_variable_of_the_wrong_length_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createDimension("three", 3)
    f.createDimension("num_nod_var", 2)
    f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
        ["short", "full"]
    )
    f.createVariable("vals_nod_var1", "f8", ("time_step", "three"))[0] = [1, 2, 3]
    f.createVariable("vals_nod_var2", "f8", ("time_step", "num_nodes"))[0] = [
        1,
        2,
        3,
        4,
    ]
    f.createDimension("num_elem_var", 1)
    f.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))[:] = _chars(
        ["rho"]
    )
    f.createVariable("vals_elem_var1eb1", "f8", ("time_step", "three"))[0] = [1, 2, 3]
    f.close()
    with (
        pytest.warns(UserWarning, match=r"nodal variable\(s\) \['short'\]"),
        pytest.warns(UserWarning, match=r"element variable\(s\) \['rho \(block 1\)'\]"),
    ):
        poly = read(path)
    assert sorted(poly.vertex_attrs) == ["full"]
    assert "rho" not in poly.element_attrs


def test_a_side_of_a_skipped_block_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _File(path, np.vstack([_CUBE, [[0.5, 0.5, 0.5]]])).block(
        "HEX9", [[0, 1, 2, 3, 4, 5, 6, 7, 8]], name="odd"
    ).block("TETRA4", [[0, 1, 3, 4]], name="tet").side_set(
        "mixed", [0, 1], [1, 1]
    ).finish().close()
    with (
        pytest.warns(UserWarning, match=r"HEX9 \(9 nodes\)"),
        pytest.warns(UserWarning, match=r"\['mixed'\].*skipped block"),
    ):
        poly = read(path)
    assert poly.element_tags["mixed"].tolist() == [1]


def test_a_step_asked_of_a_file_without_steps_is_warned_about(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    _tet_file(path).close()
    with pytest.warns(UserWarning, match="holds no time step, so step=2"):
        poly = read(path, step=2)
    assert "time" not in poly.global_attrs


def test_a_family_beside_its_own_base_name_is_not_folded(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createDimension("num_nod_var", 4)
    f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
        ["v", "v_x", "v_y", "v_z"]
    )
    for i in range(1, 5):
        var = f.createVariable(f"vals_nod_var{i}", "f8", ("time_step", "num_nodes"))
        var[0] = np.full(4, float(i))
    f.close()
    attrs = read(path).vertex_attrs
    assert sorted(attrs) == ["v", "v_x", "v_y", "v_z"]
    assert attrs["v"].tolist() == [1.0] * 4


def test_a_variable_name_given_twice_keeps_the_first(tmp_path: Path) -> None:
    path = tmp_path / "m.e"
    f = _tet_file(path)
    f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0]
    f.createDimension("num_nod_var", 2)
    f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))[:] = _chars(
        ["a", "a"]
    )
    for i in range(1, 3):
        var = f.createVariable(f"vals_nod_var{i}", "f8", ("time_step", "num_nodes"))
        var[0] = np.full(4, float(i))
    f.close()
    with pytest.warns(UserWarning, match=r"\['a'\] .* given twice"):
        attrs = read(path).vertex_attrs
    assert attrs["a"].tolist() == [1.0] * 4


def _declare(path: Path, dim: str, size: int) -> None:
    """Rewrite a CDF5 dimension's length in the header, leaving the data.

    A netCDF library fills every fixed variable to its declared length on
    close, so a table of a hundred billion entries is declared by patching
    the eight-byte length that follows the padded name in the header. Only
    the last fixed variable's dimension can be patched: the header records
    where every later variable begins, and the library refuses a file whose
    offsets contradict its shapes.
    """
    raw = bytearray(path.read_bytes())
    at = raw.index(dim.encode())
    at += len(dim) + (-len(dim)) % 4
    raw[at : at + 8] = struct.pack(">q", size)
    path.write_bytes(raw)


def test_a_table_no_file_can_hold_is_refused_before_it_is_read(
    tmp_path: Path,
) -> None:
    """The header alone declares the count; reading it would allocate it."""
    path = tmp_path / "m.e"
    _File(path, _TET, fmt="NETCDF3_64BIT_DATA").block(
        "TETRA4", [[0, 1, 2, 3]]
    ).node_set("all", [0, 1, 2, 3]).finish().close()
    _declare(path, "num_nod_ns1", 10**11)
    with pytest.raises(CodecError, match="'node_ns1' declares 100000000000 values"):
        read(path)


@pytest.mark.parametrize(
    ("build", "key"),
    [
        pytest.param(
            lambda f: (
                f.createDimension("num_side_sets", 1),
                f.createDimension("num_side_ss1", 10**11),
                f.createVariable("elem_ss1", "i4", ("num_side_ss1",)),
                f.createVariable("side_ss1", "i4", ("num_side_ss1",)),
            ),
            "elem_ss1",
            id="side set",
        ),
        pytest.param(
            lambda f: (
                f.createDimension("num_att_in_blk1", 10**11),
                f.createVariable(
                    "attrib1", "f8", ("num_el_in_blk1", "num_att_in_blk1")
                ),
            ),
            "attrib1",
            id="block attributes",
        ),
        pytest.param(
            lambda f: (
                f.createVariable("time_whole", "f8", ("time_step",)).__setitem__(
                    0, 0.0
                ),
                f.createDimension("num_nod_var", 1),
                f.createDimension("wide", 10**11),
                f.createVariable("vals_nod_var1", "f8", ("time_step", "wide")),
            ),
            "vals_nod_var1",
            id="nodal variable",
        ),
    ],
)
def test_a_table_past_the_cap_is_refused_whatever_the_file_size(
    tmp_path: Path, build, key: str
) -> None:
    """A netCDF-4 file allocates nothing for what is never written, so its
    size says nothing about a table; the hard cap on any read still holds."""
    path = tmp_path / "m.e"
    f = _File(path, _TET, fmt="NETCDF4").block("TETRA4", [[0, 1, 2, 3]]).finish()
    build(f)
    f.close()
    with pytest.raises(CodecError, match=f"'{key}' declares 100000000000 values"):
        read(path)


@pytest.mark.skipif(sys.platform == "win32", reason="a mode bit seals no folder")
def test_a_directory_that_refuses_the_write_is_reported_by_the_path_given(
    tmp_path: Path,
) -> None:
    if getattr(os, "geteuid", lambda: 1)() == 0:
        pytest.skip("root writes anywhere")
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    folder = tmp_path / "sealed"
    folder.mkdir()
    folder.chmod(0o500)
    target = folder / "m.e"
    try:
        with pytest.raises(PermissionError) as info:
            write(poly, target)
    finally:
        folder.chmod(0o700)
    assert info.value.filename == str(target)
    assert ".partial" not in str(info.value)


def test_a_global_triple_goes_out_by_axis(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        global_attrs={"g": np.array([1.0, 2.0, 3.0]), "w": np.array([1.0, 2.0])},
    )
    write(poly, tmp_path / "m.e")
    with netCDF4.Dataset(tmp_path / "m.e") as f:
        names = f.variables["name_glo_var"][:]
    names = [row.tobytes().split(b"\x00", 1)[0].decode() for row in names]
    assert names == ["g_x", "g_y", "g_z", "w_0", "w_1"]
    back = read(tmp_path / "m.e").global_attrs
    assert back["g"].tolist() == [1.0, 2.0, 3.0]
    assert back["w"].tolist() == [1.0, 2.0]


def test_a_title_that_is_not_text_is_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET, [("tetra", np.array([[0, 1, 2, 3]]))], global_attrs={"title": 5}
    )
    with pytest.warns(UserWarning, match=r"\['title'\] is int, not text"):
        write(poly, tmp_path / "m.e")
    assert "title" not in read(tmp_path / "m.e").global_attrs
