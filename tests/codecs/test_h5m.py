"""H5M: MOAB's native HDF5 mesh file.

Every test here needs h5py and skips without it. Files are built with h5py
directly, spelling the ``tstt`` layout MOAB writes, so the reader is tested
against the format and not against the writer beside it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._optpkg import TripWire
from polyxios.codecs import _hdf5
from polyxios.codecs._h5m import read, write
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError

h5py = pytest.importorskip("h5py")

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)


def _file(
    path: Path,
    points: np.ndarray,
    elements: dict[str, np.ndarray],
    *,
    node_start: int = 1,
):
    """The tstt tree: coordinates, then one group per element type, ids running on."""
    f = h5py.File(path, "w")
    tstt = f.create_group("tstt")
    coords = tstt.create_dataset("nodes/coordinates", data=points)
    coords.attrs.create("start_id", node_start, dtype=np.int64)
    next_id = node_start + len(points)
    starts = {}
    for name, conn in elements.items():
        conn = np.asarray(conn, dtype=np.uint64) + node_start
        table = tstt.create_dataset(f"elements/{name}/connectivity", data=conn)
        table.attrs.create("start_id", next_id, dtype=np.int64)
        starts[name] = next_id
        next_id += len(conn)
    tstt.attrs.create("max_id", next_id - 1, dtype=np.uint64)
    return f, starts, next_id


def _sets(
    f, set_start: int, sets: list[tuple[list[int], int]], tags: dict[str, list]
) -> None:
    """Sets as (contents, flags) rows, and sparse tags over them by ordinal."""
    holder = f["tstt"].create_group("sets")
    rows = []
    contents = []
    end = -1
    for members, flags in sets:
        end += len(members)
        rows.append([end, -1, -1, flags])
        contents.extend(members)
    table = holder.create_dataset(
        "list", data=np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    )
    table.attrs.create("start_id", set_start, dtype=np.int64)
    holder.create_dataset("contents", data=np.asarray(contents, dtype=np.uint64))
    for key, values in tags.items():
        group = f["tstt"].require_group(f"tags/{key}")
        group.attrs["class"] = np.int32(1)
        ids = np.asarray([set_start + k for k, _ in values], dtype=np.uint64)
        group.create_dataset("id_list", data=ids)
        if key == "NAME":
            arr = np.zeros(len(values), dtype="V32")
            for i, (_, text) in enumerate(values):
                arr[i] = np.frombuffer(text.encode().ljust(32, b"\x00"), dtype="V32")[0]
            group.create_dataset("values", data=arr)
        else:
            group.create_dataset(
                "values", data=np.asarray([v for _, v in values], dtype=np.int32)
            )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_elements_read_by_type_with_ids_taken_off_the_start_id(tmp_path: Path) -> None:
    path = tmp_path / "t.h5m"
    f, _, _ = _file(
        path, _TET, {"Tet4": [[0, 1, 2, 3]], "Tri3": [[0, 1, 2]]}, node_start=7
    )
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertices, _TET)
    assert poly.element_types.tolist() == [5, 10]
    assert poly.connectivity.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert poly.global_attrs == {}


def test_dense_tags_are_attributes_and_nan_where_a_type_lacks_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tags.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]], "Tri3": [[0, 1, 2]]})
    f["tstt/nodes"].create_dataset("tags/temp", data=np.arange(4.0))
    f["tstt/nodes"].create_dataset(
        "tags/GLOBAL_ID", data=np.array([3, 1, 2, 4], dtype=np.int32)
    )
    f["tstt/elements/Tet4"].create_dataset("tags/rho", data=np.array([9.0]))
    vec = f["tstt/elements/Tet4"].create_dataset(
        "tags/vel", (1,), dtype=np.dtype((np.float64, (3,)))
    )
    vec[0] = [1.0, 2.0, 3.0]
    f["tstt/elements/Tri3"].create_dataset(
        "tags/GLOBAL_ID", data=np.array([1], dtype=np.int32)
    )
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["temp"], np.arange(4.0))
    np.testing.assert_array_equal(poly.vertex_attrs["original_ids"], [3, 1, 2, 4])
    np.testing.assert_array_equal(poly.element_attrs["rho"], [np.nan, 9.0])
    np.testing.assert_array_equal(
        poly.element_attrs["vel"], [[np.nan] * 3, [1.0, 2.0, 3.0]]
    )
    assert "original_ids" not in poly.element_attrs


def test_sets_are_tags_named_by_their_name_or_their_kind(tmp_path: Path) -> None:
    path = tmp_path / "sets.h5m"
    f, starts, next_id = _file(
        path, _TET, {"Tet4": [[0, 1, 2, 3]], "Tri3": [[0, 1, 2], [0, 1, 3]]}
    )
    tri = starts["Tri3"]
    _sets(
        f,
        next_id,
        [([tri, tri + 1], 2), ([starts["Tet4"]], 2), ([1, 2], 2), ([tri, 2], 0x8)],
        {
            "NEUMANN_SET": [(0, 5)],
            "MATERIAL_SET": [(1, 1)],
            "DIRICHLET_SET": [(2, 3)],
            "NAME": [(1, "steel"), (3, "ranged")],
        },
    )
    f.close()
    poly = read(path)
    assert poly.element_tags["neumann_5"].tolist() == [0, 1]
    assert poly.element_tags["steel"].tolist() == [2]
    assert poly.vertex_tags["dirichlet_3"].tolist() == [0, 1]
    # A range-compressed set spells (start, count): the two triangles.
    assert poly.element_tags["ranged"].tolist() == [0, 1]
    assert "ranged" not in poly.vertex_tags


def test_a_global_tag_is_a_global_attr(tmp_path: Path) -> None:
    path = tmp_path / "g.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})
    g = f["tstt"].create_group("tags/answer")
    g.attrs["class"] = np.int32(3)
    g.attrs["global"] = np.array([42], dtype=np.int32)
    f.close()
    assert read(path).global_attrs == {"answer": 42}


def test_an_element_group_polyxios_cannot_hold_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "poly.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})
    f["tstt/elements"].create_group("Polyhedron")
    f.close()
    with pytest.warns(UserWarning, match=r"\['Polyhedron'\]"):
        poly = read(path)
    assert poly.element_types.tolist() == [10]


def test_a_connectivity_of_the_wrong_width_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "wide.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2]]})
    f.close()
    with pytest.raises(CodecError, match=r"Tet4/connectivity"):
        read(path)


def test_a_node_outside_the_mesh_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2, 9]]})
    f.close()
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_a_file_without_tstt_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "nope.h5m"
    with h5py.File(path, "w") as f:
        f.create_group("other")
    with pytest.raises(CodecError, match="'tstt'"):
        read(path)


def test_lazy_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.h5m"
    _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})[0].close()
    with pytest.raises(LazyReadError):
        read(path, lazy=True)


def test_without_h5py_the_read_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "t.h5m"
    _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})[0].close()
    monkeypatch.setattr(_hdf5, "_h5py", lambda: (TripWire("no h5py"), False))
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
        read(path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_issue_1246_triangles_from_several_blocks_are_one_group(tmp_path: Path) -> None:
    """A Gmsh mesh with six physical surfaces came as 28 triangle blocks, and
    the writer created ``Tri3`` once per block; the second creation was
    refused as a name that exists. MOAB has one group per type."""
    poly = make_polydata(
        _SQUARE,
        [
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
            ("line", np.array([[0, 1]])),
            ("triangle", np.array([[1, 2, 3]])),
        ],
        element_attrs={"physical": np.array([1, 1, 7, 2], dtype=np.int32)},
        element_tags={"top": np.array([0, 3])},
    )
    path = tmp_path / "phys.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        conn = f["tstt/elements/Tri3/connectivity"]
        assert conn.shape == (3, 3)
        assert conn.dtype == np.uint64
        assert conn[()].min() == 1
        # Edges (type 3) go out ahead of triangles (5), each group's ids
        # following the last: nodes 1..4, the edge 5, the triangles 6..8.
        assert f["tstt/elements/Edge2/connectivity"].attrs["start_id"] == 5
        assert conn.attrs["start_id"] == 6
        assert f["tstt/elements/Tri3/tags/physical"][()].tolist() == [1, 1, 2]
        assert f["tstt/elements/Edge2/tags/physical"][()].tolist() == [7]
        assert f["tstt/tags/physical"].attrs["class"] == 2
        assert "type" in f["tstt/tags/physical"]
        assert f["tstt"].attrs["max_id"] == 9
        assert "sets" in f["tstt"]
    back = read(path)
    assert back.element_types.tolist() == [3, 5, 5, 5]
    np.testing.assert_array_equal(back.element_attrs["physical"], [7, 1, 1, 2])
    assert back.element_tags["top"].tolist() == [1, 3]
    # GLOBAL_ID numbers the written order, so the grouped mesh reads back
    # numbered 1..n and carries no original_ids it never had.
    with h5py.File(path) as f:
        assert f["tstt/elements/Edge2/tags/GLOBAL_ID"][()].tolist() == [1]
        assert f["tstt/elements/Tri3/tags/GLOBAL_ID"][()].tolist() == [2, 3, 4]
    assert "original_ids" not in back.element_attrs


def test_issue_1243_cell_data_of_any_shape_writes_as_dense_tags(tmp_path: Path) -> None:
    """Any non-empty cell data crashed the writer, which walked it in a layout
    the library had given up long before, and the tags group landed under
    whatever element group came last."""
    poly = make_polydata(
        _SQUARE,
        [
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
            ("quad", np.array([[0, 1, 2, 3]])),
        ],
        element_attrs={"a": np.arange(3.0), "b": np.arange(6.0).reshape(3, 2)},
    )
    path = tmp_path / "cd.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        for group in ("Tri3", "Quad4"):
            assert sorted(f[f"tstt/elements/{group}/tags"]) == ["GLOBAL_ID", "a", "b"]
        assert f["tstt/elements/Quad4/tags/b"][()].tolist() == [[4.0, 5.0]]
        assert f["tstt/tags/b/type"].dtype == np.dtype((np.float64, (2,)))
    back = read(path)
    np.testing.assert_array_equal(back.element_attrs["a"], np.arange(3.0))
    np.testing.assert_array_equal(back.element_attrs["b"], poly.element_attrs["b"])


def test_a_mesh_with_no_element_moab_can_hold_still_writes(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0], [1]]))], vertex_attrs={"s": np.arange(4.0)}
    )
    path = tmp_path / "points.h5m"
    with pytest.warns(UserWarning, match=r"\['vertex'\]"):
        write(poly, path)
    back = read(path)
    assert len(back.element_types) == 0
    np.testing.assert_array_equal(back.vertex_attrs["s"], np.arange(4.0))


def test_tags_go_out_as_named_sets_with_a_kind(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_tags={"fixed": np.array([0, 3])},
        element_tags={"steel": np.array([0])},
    )
    path = tmp_path / "sets.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        table = f["tstt/sets/list"][()]
        assert table.shape == (2, 4)
        assert f["tstt/sets/list"].attrs["start_id"] == 6
        assert f["tstt/sets/contents"][()].tolist() == [1, 4, 5]
        names = [bytes(v).rstrip(b"\x00") for v in f["tstt/tags/NAME/values"][()]]
        assert names == [b"fixed", b"steel"]
        assert f["tstt/tags/NAME/id_list"][()].tolist() == [6, 7]
        assert f["tstt/tags/DIRICHLET_SET/id_list"][()].tolist() == [6]
        assert f["tstt/tags/MATERIAL_SET/id_list"][()].tolist() == [7]
        assert f["tstt/tags/MATERIAL_SET"].attrs["class"] == 1
        assert f["tstt"].attrs["max_id"] == 7
    back = read(path)
    assert back.vertex_tags["fixed"].tolist() == [0, 3]
    assert back.element_tags["steel"].tolist() == [0]


def test_original_ids_become_global_ids_and_come_back(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"original_ids": np.array([10, 20, 30, 40])},
    )
    path = tmp_path / "ids.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["tstt/nodes/tags/GLOBAL_ID"][()].tolist() == [10, 20, 30, 40]
    np.testing.assert_array_equal(
        read(path).vertex_attrs["original_ids"], [10, 20, 30, 40]
    )
    write(poly, path, global_ids=False)
    with h5py.File(path) as f:
        assert "tags" not in f["tstt/nodes"]


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
    path = tmp_path / "lattice.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        assert sorted(f["tstt/elements"]) == ["Hex8", "Quad4"]
        assert f["tstt/elements/Hex8"].attrs["element_type"] == 9
    back = read(path)
    assert back.connectivity.tolist() == [0, 1, 3, 2, 0, 1, 3, 2, 4, 5, 7, 6]


def test_compression_is_passed_to_the_datasets(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "gz.h5m"
    write(poly, path, compression="gzip", compression_opts=4)
    with h5py.File(path) as f:
        assert f["tstt/nodes/coordinates"].compression == "gzip"


def test_a_buffer_and_a_gzip_name_both_work(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    buffer = io.BytesIO()
    polyxios.write(poly, buffer, fmt=".h5m")
    buffer.seek(0)
    assert polyxios.read(buffer, fmt=".h5m").connectivity.tolist() == [0, 1, 2, 3]
    packed = tmp_path / "t.h5m.gz"
    polyxios.write(poly, packed)
    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert polyxios.read(packed).connectivity.tolist() == [0, 1, 2, 3]


def test_the_codec_is_registered_for_h5m() -> None:
    assert ".h5m" in polyxios.supported_extensions()


# ---------------------------------------------------------------------------
# Node order, tag types and the sets' edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "ptype", "vtk_order"),
    [
        ("Prism6", "wedge", [0, 2, 1, 3, 5, 4]),
        (
            "Prism15",
            "quadratic_wedge",
            [0, 2, 1, 3, 5, 4, 8, 7, 6, 14, 13, 12, 9, 11, 10],
        ),
        (
            "Prism18",
            "biquadratic_quadratic_wedge",
            [0, 2, 1, 3, 5, 4, 8, 7, 6, 14, 13, 12, 9, 11, 10, 17, 16, 15],
        ),
        (
            "Hex20",
            "quadratic_hexahedron",
            [*range(12), 16, 17, 18, 19, 12, 13, 14, 15],
        ),
        (
            "Hex27",
            "triquadratic_hexahedron",
            [*range(12), 16, 17, 18, 19, 12, 13, 14, 15, 23, 21, 20, 22, 24, 25, 26],
        ),
    ],
)
def test_moab_node_order_is_permuted_to_vtk_and_back(
    tmp_path: Path, group: str, ptype: str, vtk_order: list[int]
) -> None:
    """MOAB's VTK reader spells these tables (``src/io/VtkUtil.cpp``): a
    prism's triangles run the other way round, a hexahedron's vertical
    edges come before its top ring, its face centres follow MOAB's faces."""
    n = len(vtk_order)
    points = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(
        np.float64
    )
    path = tmp_path / "order.h5m"
    _file(path, points, {group: [list(range(n))]})[0].close()
    poly = read(path)
    assert poly.element_types.tolist() == [int(ELEMENT_TYPES[ptype])]
    assert poly.connectivity.tolist() == vtk_order
    out = tmp_path / "back.h5m"
    write(poly, out)
    with h5py.File(out) as f:
        assert f[f"tstt/elements/{group}/connectivity"][()].tolist() == [
            list(range(1, n + 1))
        ]


def test_a_wedge_keeps_its_volume_positive_in_moab_s_order(tmp_path: Path) -> None:
    """VTK's wedge has its base triangle wound so the normal points away from
    the top; MOAB's prism winds it the other way. Written as MOAB's, the
    base's normal has to point at the top."""
    verts = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("wedge", np.array([[0, 1, 2, 3, 4, 5]]))])
    path = tmp_path / "wedge.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        conn = f["tstt/elements/Prism6/connectivity"][()][0] - 1
    a, b, c = verts[conn[0]], verts[conn[1]], verts[conn[2]]
    normal = np.cross(b - a, c - a)
    assert np.dot(normal, verts[conn[3]] - a) > 0
    assert read(path).connectivity.tolist() == [0, 1, 2, 3, 4, 5]


def test_a_vertex_and_an_element_attribute_of_one_name_keep_their_types(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"v": np.arange(4.0)},
        element_attrs={"v": np.arange(6, dtype=np.int32).reshape(2, 3)},
    )
    path = tmp_path / "twice.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["tstt/tags/v/type"].dtype == np.float64
        assert f["tstt/tags/v__cell/type"].dtype == np.dtype((np.int32, (3,)))
        assert f["tstt/tags/v__cell"].attrs["polyxios_name"] == "v"
        assert "v__cell" in f["tstt/elements/Tri3/tags"]
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["v"], np.arange(4.0))
    np.testing.assert_array_equal(back.element_attrs["v"], poly.element_attrs["v"])
    assert "v__cell" not in back.element_attrs


def test_an_attribute_of_one_name_and_one_type_is_one_tag(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"v": np.arange(4.0)},
        element_attrs={"v": np.arange(2.0)},
    )
    path = tmp_path / "once.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        assert sorted(f["tstt/tags"]) == ["GLOBAL_ID", "v"]


def test_an_unwritable_element_attribute_warns_once_whatever_the_types(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[0, 1]])),
        ],
        element_attrs={"label": np.array(["a", "b", "c", "d"])},
    )
    with pytest.warns(UserWarning, match=r"\['label'\]") as record:
        write(poly, tmp_path / "once.h5m")
    assert sum("label" in str(w.message) for w in record) == 1


def test_a_two_dimensional_global_keeps_its_shape(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET, [("tetra", np.array([[0, 1, 2, 3]]))], global_attrs={"m": np.eye(2)}
    )
    path = tmp_path / "shape.h5m"
    write(poly, path)
    np.testing.assert_array_equal(read(path).global_attrs["m"], np.eye(2))


def test_a_set_kind_with_an_unreadable_value_falls_back_to_the_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "odd.h5m"
    f, starts, next_id = _file(path, _TET, {"Tri3": [[0, 1, 2], [0, 1, 3]]})
    tri = starts["Tri3"]
    _sets(f, next_id, [([tri], 2), ([tri + 1], 2)], {"NAME": [(0, "named")]})
    group = f["tstt"].create_group("tags/MATERIAL_SET")
    group.attrs["class"] = np.int32(1)
    group.create_dataset(
        "id_list", data=np.array([next_id, next_id + 1], dtype=np.uint64)
    )
    group.create_dataset("values", data=np.zeros((2, 2)))
    f.close()
    with pytest.warns(UserWarning, match=r"\['MATERIAL_SET'\]"):
        poly = read(path)
    assert poly.element_tags["named"].tolist() == [0]
    assert not any(k.startswith("material_") for k in poly.element_tags)


def test_an_empty_tag_group_survives_a_round_trip(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_tags={"nowhere": np.array([], dtype=np.int64)},
        element_tags={"nothing": np.array([], dtype=np.int64), "all": np.array([0])},
    )
    path = tmp_path / "empty.h5m"
    write(poly, path)
    back = read(path)
    assert back.vertex_tags["nowhere"].tolist() == []
    assert back.element_tags["nothing"].tolist() == []
    assert back.element_tags["all"].tolist() == [0]


def test_a_sparse_tag_over_entities_is_not_read_and_warns_once(tmp_path: Path) -> None:
    path = tmp_path / "sparse.h5m"
    f, starts, next_id = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})
    _sets(f, next_id, [([starts["Tet4"]], 2)], {"NAME": [(0, "cell")]})
    for key in ("marker", "other"):
        group = f["tstt"].create_group(f"tags/{key}")
        group.attrs["class"] = np.int32(1)
        group.create_dataset("id_list", data=np.array([1, 2], dtype=np.uint64))
        group.create_dataset("values", data=np.array([1.0, 2.0]))
    f.close()
    with pytest.warns(UserWarning, match=r"\['marker', 'other'\]") as record:
        poly = read(path)
    assert sum("sparse" in str(w.message) for w in record) == 1
    assert "marker" not in poly.vertex_attrs
    assert poly.element_tags["cell"].tolist() == [0]


def test_a_sparse_tag_whose_values_are_one_scalar_is_refused_by_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scalar.h5m"
    f, starts, next_id = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})
    _sets(f, next_id, [([starts["Tet4"]], 2)], {})
    group = f["tstt"].create_group("tags/MATERIAL_SET")
    group.create_dataset("id_list", data=np.array([next_id], dtype=np.uint64))
    group.create_dataset("values", data=np.int32(3))
    f.close()
    with pytest.raises(CodecError, match="MATERIAL_SET/values"):
        read(path)


def test_range_compressed_contents_expand_every_run(tmp_path: Path) -> None:
    path = tmp_path / "runs.h5m"
    f, starts, next_id = _file(
        path, _TET, {"Tri3": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]}
    )
    tri = starts["Tri3"]
    _sets(f, next_id, [([tri, 1, tri + 3, 1, 1, 2], 0x8)], {"NAME": [(0, "runs")]})
    f.close()
    poly = read(path)
    assert poly.element_tags["runs"].tolist() == [0, 3]
    assert poly.vertex_tags["runs"].tolist() == [0, 1]


def test_a_negative_run_count_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "neg.h5m"
    f, starts, next_id = _file(path, _TET, {"Tri3": [[0, 1, 2]]})
    holder = f["tstt"].create_group("sets")
    table = holder.create_dataset(
        "list", data=np.array([[1, -1, -1, 0x8]], dtype=np.int64)
    )
    table.attrs.create("start_id", next_id, dtype=np.int64)
    holder.create_dataset(
        "contents", data=np.array([starts["Tri3"], -1], dtype=np.int64)
    )
    group = f["tstt"].create_group("tags/NAME")
    group.create_dataset("id_list", data=np.array([next_id], dtype=np.uint64))
    group.create_dataset(
        "values", data=np.frombuffer(b"x".ljust(32, b"\x00"), dtype="V32")
    )
    f.close()
    with pytest.raises(CodecError, match="negative count"):
        read(path)


def test_an_attribute_named_like_a_set_tag_keeps_the_set_tag_intact(
    tmp_path: Path,
) -> None:
    """A dense ``NAME`` or ``MATERIAL_SET`` attribute used to take the
    declaration the meshsets' sparse tag of that name then overwrote, leaving
    a tag declared sparse with a dense type; it goes out suffixed instead."""
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"NAME": np.arange(4), "DIRICHLET_SET": np.arange(4.0)},
        element_attrs={"MATERIAL_SET": np.arange(2), "NAME": np.arange(2.0)},
        vertex_tags={"g": np.array([0])},
        element_tags={"m": np.array([1])},
    )
    path = tmp_path / "reserved.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        tags = f["tstt/tags"]
        assert tags["NAME"].attrs["class"] == 1
        assert tags["NAME/type"].dtype == np.dtype("V32")
        assert tags["MATERIAL_SET/type"].dtype == np.int32
        assert tags["NAME__attr"].attrs["class"] == 2
        assert tags["NAME__attr"].attrs["polyxios_name"] == "NAME"
        assert tags["NAME__cell"].attrs["polyxios_name"] == "NAME"
        assert tags["MATERIAL_SET__attr"].attrs["polyxios_name"] == "MATERIAL_SET"
        assert tags["DIRICHLET_SET__attr"].attrs["polyxios_name"] == "DIRICHLET_SET"
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["NAME"], np.arange(4))
    np.testing.assert_array_equal(back.vertex_attrs["DIRICHLET_SET"], np.arange(4.0))
    np.testing.assert_array_equal(back.element_attrs["MATERIAL_SET"], np.arange(2))
    np.testing.assert_array_equal(back.element_attrs["NAME"], np.arange(2.0))
    assert back.vertex_tags["g"].tolist() == [0]
    assert back.element_tags["m"].tolist() == [1]


def test_a_global_named_like_a_dense_tag_survives_beside_it(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"s": np.arange(4.0)},
        vertex_tags={"g": np.array([0])},
        global_attrs={"s": 3, "NAME": 7, "answer": 42},
    )
    path = tmp_path / "globals.h5m"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["tstt/tags/s__global"].attrs["polyxios_name"] == "s"
        assert f["tstt/tags/NAME__global"].attrs["polyxios_name"] == "NAME"
        assert "polyxios_name" not in f["tstt/tags/answer"].attrs
    back = read(path)
    assert back.global_attrs == {"s": 3, "NAME": 7, "answer": 42}
    np.testing.assert_array_equal(back.vertex_attrs["s"], np.arange(4.0))


def test_a_dense_tag_of_the_wrong_length_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "short.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})
    f["tstt/nodes"].create_dataset("tags/short", data=np.arange(3.0))
    f.close()
    with pytest.warns(UserWarning, match=r"'tstt/nodes/tags/short'.*3 values for 4"):
        poly = read(path)
    assert "short" not in poly.vertex_attrs


def test_a_global_id_that_is_not_an_id_stays_an_attribute(tmp_path: Path) -> None:
    path = tmp_path / "gid.h5m"
    f, _, _ = _file(path, _TET, {"Tet4": [[0, 1, 2, 3]]})
    f["tstt/nodes"].create_dataset("tags/GLOBAL_ID", data=np.arange(4.0) + 0.5)
    f.close()
    poly = read(path)
    assert "original_ids" not in poly.vertex_attrs
    np.testing.assert_array_equal(poly.vertex_attrs["GLOBAL_ID"], np.arange(4.0) + 0.5)
