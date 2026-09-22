"""CGNS: the CFD General Notation System in its HDF5 (ADFH) form.

Every test here needs h5py and skips without it. Files are built with h5py
directly, spelling the node layout the CGNS library writes - ``name``,
``label`` and ``type`` attributes on every group, the value in a `` data``
dataset - so the reader is tested against the format and not against the
writer beside it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES_INV
from polyxios._optpkg import TripWire
from polyxios.codecs import _hdf5
from polyxios.codecs._cgns import read, write
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    UnknownElementTypeError,
    UnsupportedFormatError,
)

h5py = pytest.importorskip("h5py")

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_CUBE = np.array(
    [[i, j, k] for k in range(2) for j in range(2) for i in range(2)], dtype=np.float64
)
_HEX = [0, 1, 3, 2, 4, 5, 7, 6]


def _node(parent, name: str, label: str, data=None, *, code: str | None = None):
    g = parent.create_group(name)
    g.attrs.create("name", np.bytes_(name), dtype="S33")
    g.attrs.create("label", np.bytes_(label), dtype="S33")
    g.attrs.create("flags", np.array([1], dtype=np.int32))
    if data is None:
        g.attrs.create("type", np.bytes_("MT"), dtype="S3")
        return g
    if isinstance(data, str):
        g.attrs.create("type", np.bytes_("C1"), dtype="S3")
        g.create_dataset(" data", data=np.frombuffer(data.encode(), dtype=np.int8))
        return g
    arr = np.asarray(data)
    if code is None:
        code = {"f": "R8", "i": "I4", "u": "I4"}[arr.dtype.kind]
        if arr.dtype.kind == "f" and arr.dtype.itemsize == 4:
            code = "R4"
        if arr.dtype.kind == "i" and arr.dtype.itemsize == 8:
            code = "I8"
    g.attrs.create("type", np.bytes_(code), dtype="S3")
    g.create_dataset(" data", data=arr)
    return g


def _file(path: Path, *, base: str = "Base", cell_dim: int = 3, phys_dim: int = 3):
    f = h5py.File(path, "w")
    f.attrs.create("name", np.bytes_("HDF5 MotherNode"), dtype="S33")
    f.attrs.create("label", np.bytes_("Root Node of HDF5 File"), dtype="S33")
    f.attrs.create("type", np.bytes_("MT"), dtype="S3")
    f.create_dataset(
        " format", data=np.frombuffer(b"IEEE_LITTLE_32\x00", dtype=np.int8)
    )
    f.create_dataset(
        " hdf5version",
        data=np.frombuffer(b"HDF5 Version 1.8.17".ljust(33, b"\x00"), dtype=np.int8),
    )
    _node(
        f,
        "CGNSLibraryVersion",
        "CGNSLibraryVersion_t",
        np.array([3.3], dtype=np.float32),
    )
    b = _node(f, base, "CGNSBase_t", np.array([cell_dim, phys_dim], dtype=np.int32))
    return f, b


def _zone(base, name: str, points: np.ndarray, n_cells: int, *, itype=np.int32):
    z = _node(
        base, name, "Zone_t", np.array([[len(points)], [n_cells], [0]], dtype=itype)
    )
    _node(z, "ZoneType", "ZoneType_t", "Unstructured")
    grid = _node(z, "GridCoordinates", "GridCoordinates_t")
    for k, axis in enumerate(
        ("CoordinateX", "CoordinateY", "CoordinateZ")[: points.shape[1]]
    ):
        _node(grid, axis, "DataArray_t", np.ascontiguousarray(points[:, k]))
    return z


def _section(
    zone, name: str, code: int, start: int, conn, *, offsets=None, itype=np.int32
):
    s = _node(zone, name, "Elements_t", np.array([code, 0], dtype=np.int32))
    conn = np.asarray(conn, dtype=itype)
    n = (
        len(offsets) - 1
        if offsets is not None
        else (len(conn) // _WIDTH[code] if code in _WIDTH else None)
    )
    _node(
        s, "ElementRange", "IndexRange_t", np.array([start, start + n - 1], dtype=itype)
    )
    _node(s, "ElementConnectivity", "DataArray_t", conn)
    if offsets is not None:
        _node(s, "ElementStartOffset", "DataArray_t", np.asarray(offsets, dtype=itype))
    return s


_WIDTH = {
    2: 1,
    3: 2,
    5: 3,
    7: 4,
    10: 4,
    12: 5,
    14: 6,
    15: 15,
    16: 18,
    17: 8,
    18: 20,
    19: 27,
    13: 14,
}


def _one_tet(path: Path) -> Path:
    f, base = _file(path)
    z = _zone(base, "Zone", _TET, 1)
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    f.close()
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_zone_reads_with_its_names(tmp_path: Path) -> None:
    poly = read(_one_tet(tmp_path / "t.cgns"))
    np.testing.assert_array_equal(poly.vertices, _TET)
    assert poly.connectivity.tolist() == [0, 1, 2, 3]
    assert poly.global_attrs == {"base_name": "Base", "zone_name": "Zone"}


def test_issue_1480_every_zone_is_read_and_tagged_by_its_name(tmp_path: Path) -> None:
    """A second zone was refused: the reader looked for one called Zone1.
    Bases, zones and sections are found by their label, whatever their
    name, and several zones merge with a tag group each."""
    path = tmp_path / "two.cgns"
    f, base = _file(path, base="B")
    z1 = _zone(base, "blk-1", _TET, 1)
    _section(z1, "Cells", 10, 1, [1, 2, 3, 4])
    z2 = _zone(base, "blk-2", _CUBE, 1)
    _section(z2, "Hexas", 17, 1, np.array(_HEX) + 1)
    f.close()
    poly = read(path)
    assert len(poly.vertices) == 12
    assert poly.element_types.tolist() == [10, 12]
    assert poly.connectivity.tolist() == [0, 1, 2, 3, *(np.array(_HEX) + 4)]
    assert poly.element_tags["blk-1"].tolist() == [0]
    assert poly.element_tags["blk-2"].tolist() == [1]
    assert poly.global_attrs["base_name"] == "B"
    assert "zone_name" not in poly.global_attrs
    one = read(path, zone="blk-2")
    assert len(one.vertices) == 8
    assert one.global_attrs["zone_name"] == "blk-2"
    with pytest.raises(CodecError, match=r"\['blk-1', 'blk-2'\]"):
        read(path, zone="nope")
    with pytest.raises(CodecError, match=r"\['B'\]"):
        read(path, base="nope")


def test_issue_1405_a_mixed_section_under_any_names_splits_by_type(
    tmp_path: Path,
) -> None:
    """A Hexpress file names its base 'Unstructured data', its zone 'cubo'
    and writes every section MIXED: a type code ahead of each element's
    nodes. Both were refused for their names alone."""
    path = tmp_path / "mixed.cgns"
    f, base = _file(path, base="Unstructured data")
    verts = np.vstack([_TET, _CUBE + [2, 0, 0]])
    z = _zone(base, "cubo", verts, 2)
    _section(
        z,
        "Elements connectivity",
        20,
        1,
        [10, 1, 2, 3, 4, 17, *(np.array(_HEX) + 5)],
        offsets=[0, 5, 14],
    )
    _section(z, "Mesh faces", 20, 3, [5, 1, 2, 3, 7, 5, 6, 8, 7], offsets=[0, 4, 9])
    f.close()
    poly = read(path)
    assert poly.element_types.tolist() == [5, 9, 10, 12]
    assert poly.connectivity.tolist() == [
        0,
        1,
        2,
        4,
        5,
        7,
        6,
        0,
        1,
        2,
        3,
        *(np.array(_HEX) + 4),
    ]


def test_a_mixed_section_without_offsets_walks_the_codes(tmp_path: Path) -> None:
    path = tmp_path / "old_mixed.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    s = _node(z, "Mixed", "Elements_t", np.array([20, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([1, 2], dtype=np.int32))
    _node(
        s,
        "ElementConnectivity",
        "DataArray_t",
        np.array([10, 1, 2, 3, 4, 5, 1, 2, 3], dtype=np.int32),
    )
    f.close()
    poly = read(path)
    assert poly.element_types.tolist() == [5, 10]


def test_polygons_read_with_offsets_and_with_counts_ahead(tmp_path: Path) -> None:
    square = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float64
    )
    new = tmp_path / "ngon4.cgns"
    f, base = _file(new, cell_dim=2)
    z = _zone(base, "Z", square, 2)
    _section(z, "Faces", 22, 1, [1, 2, 3, 4, 2, 5, 3], offsets=[0, 4, 7])
    f.close()
    poly = read(new)
    assert poly.element_types.tolist() == [7, 7]
    assert poly.connectivity.tolist() == [0, 1, 2, 3, 1, 4, 2]

    old = tmp_path / "ngon3.cgns"
    f, base = _file(old, cell_dim=2)
    z = _zone(base, "Z", square, 2)
    s = _node(z, "Faces", "Elements_t", np.array([22, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([1, 2], dtype=np.int32))
    _node(
        s,
        "ElementConnectivity",
        "DataArray_t",
        np.array([4, 1, 2, 3, 4, 3, 2, 5, 3], dtype=np.int32),
    )
    f.close()
    assert read(old).connectivity.tolist() == [0, 1, 2, 3, 1, 4, 2]


def test_a_polyhedral_section_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "nface.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(
        z,
        "Faces",
        22,
        1,
        [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4],
        offsets=[0, 3, 6, 9, 12],
    )
    _section(z, "Cells", 23, 5, [1, 2, 3, 4], offsets=[0, 4])
    f.close()
    with pytest.warns(UserWarning, match=r"\['NFACE_n'\]"):
        poly = read(path)
    assert poly.element_types.tolist() == [7, 7, 7, 7]


def test_a_type_code_the_sids_does_not_define_is_named(tmp_path: Path) -> None:
    path = tmp_path / "odd.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    s = _node(z, "Cells", "Elements_t", np.array([99, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([1, 1], dtype=np.int32))
    _node(
        s, "ElementConnectivity", "DataArray_t", np.array([1, 2, 3, 4], dtype=np.int32)
    )
    f.close()
    with pytest.raises(UnknownElementTypeError):
        read(path)


def test_the_quadratic_hexahedron_s_edges_are_reordered_both_ways(
    tmp_path: Path,
) -> None:
    verts = np.random.default_rng(2).random((20, 3))
    path = tmp_path / "hex20.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", verts, 1)
    _section(z, "Cells", 18, 1, np.arange(1, 21))
    f.close()
    poly = read(path)
    # The SIDS lists the vertical edges (12..15) before the top ones (16..19).
    assert poly.connectivity.tolist() == [
        *range(8),
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
    ]
    out = tmp_path / "back.cgns"
    write(poly, out)
    with h5py.File(out) as g:
        conn = g["Base/Z/HEXA_20/ElementConnectivity/ data"][()]
    assert conn.tolist() == list(range(1, 21))


def test_a_structured_zone_expands_to_hexahedra(tmp_path: Path) -> None:
    path = tmp_path / "grid.cgns"
    f, base = _file(path)
    z = _node(
        base, "Z", "Zone_t", np.array([[3, 2, 2], [2, 1, 1], [0, 0, 0]], dtype=np.int32)
    )
    _node(z, "ZoneType", "ZoneType_t", "Structured")
    grid = _node(z, "GridCoordinates", "GridCoordinates_t")
    # (ni, nj, nk) = (3, 2, 2) is stored reversed, as (nk, nj, ni).
    x = np.tile(np.arange(3.0), (2, 2, 1))
    y = np.tile(np.arange(2.0)[:, None], (2, 1, 3))
    zc = np.tile(np.arange(2.0)[:, None, None], (1, 2, 3))
    _node(grid, "CoordinateX", "DataArray_t", x)
    _node(grid, "CoordinateY", "DataArray_t", y)
    _node(grid, "CoordinateZ", "DataArray_t", zc)
    f.close()
    poly = read(path)
    assert len(poly.vertices) == 12
    assert poly.element_types.tolist() == [12, 12]
    assert poly.connectivity[:8].tolist() == [0, 1, 4, 3, 6, 7, 10, 9]
    np.testing.assert_array_equal(poly.vertices[1], [1, 0, 0])
    np.testing.assert_array_equal(poly.vertices[3], [0, 1, 0])


def test_boundary_conditions_are_tags_on_faces_or_vertices(tmp_path: Path) -> None:
    path = tmp_path / "bc.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    _section(z, "Faces", 5, 2, [1, 2, 3, 1, 2, 4, 2, 3, 4])
    bc = _node(z, "ZoneBC", "ZoneBC_t")
    wall = _node(bc, "wall", "BC_t", "BCWall")
    _node(wall, "GridLocation", "GridLocation_t", "FaceCenter")
    _node(wall, "PointList", "IndexArray_t", np.array([[2], [4]], dtype=np.int32))
    inlet = _node(bc, "inlet", "BC_t", "BCInflow")
    _node(inlet, "GridLocation", "GridLocation_t", "FaceCenter")
    _node(inlet, "PointRange", "IndexRange_t", np.array([3, 3], dtype=np.int32))
    fixed = _node(bc, "fixed", "BC_t", "BCWall")
    _node(fixed, "PointList", "IndexArray_t", np.array([[1], [4]], dtype=np.int32))
    f.close()
    poly = read(path)
    # The triangles (numbered 2..4) read ahead of the tetrahedron, by type.
    assert poly.element_types.tolist() == [5, 5, 5, 10]
    assert poly.element_tags["wall"].tolist() == [0, 2]
    assert poly.element_tags["inlet"].tolist() == [1]
    assert poly.vertex_tags["fixed"].tolist() == [0, 3]


def test_a_cell_centred_solution_over_the_cells_alone_is_nan_on_the_faces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sol.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Faces", 5, 1, [1, 2, 3])
    _section(z, "Cells", 10, 2, [1, 2, 3, 4])
    sol = _node(z, "FlowSolution", "FlowSolution_t")
    _node(sol, "GridLocation", "GridLocation_t", "CellCenter")
    _node(sol, "Pressure", "DataArray_t", np.array([7.0]))
    _node(sol, "Velocity", "DataArray_t", np.array([[1.0], [2.0], [3.0]]))
    vsol = _node(z, "Vertex", "FlowSolution_t")
    _node(vsol, "Temperature", "DataArray_t", np.arange(4.0))
    _node(vsol, "Odd", "DataArray_t", np.arange(3.0))
    f.close()
    with pytest.warns(UserWarning, match="'Odd'"):
        poly = read(path)
    assert poly.element_types.tolist() == [5, 10]
    np.testing.assert_array_equal(poly.element_attrs["Pressure"], [np.nan, 7.0])
    np.testing.assert_array_equal(
        poly.element_attrs["Velocity"], [[np.nan] * 3, [1.0, 2.0, 3.0]]
    )
    np.testing.assert_array_equal(poly.vertex_attrs["Temperature"], np.arange(4.0))
    assert "Odd" not in poly.vertex_attrs


def test_a_two_dimensional_base_is_padded_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "flat.cgns"
    f, base = _file(path, cell_dim=2, phys_dim=2)
    z = _zone(base, "Z", _TET[:3, :2], 1)
    _section(z, "Cells", 5, 1, [1, 2, 3])
    f.close()
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    assert poly.global_attrs["was_2d"] is True


def test_a_file_without_a_base_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "empty.cgns"
    with h5py.File(path, "w") as f:
        f.create_group("nothing")
    with pytest.raises(CodecError, match="CGNSBase_t"):
        read(path)


def test_lazy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_one_tet(tmp_path / "t.cgns"), lazy=True)


def test_without_h5py_the_read_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    path = _one_tet(tmp_path / "t.cgns")
    monkeypatch.setattr(_hdf5, "_h5py", lambda: (TripWire("no h5py"), False))
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
        read(path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_issue_1012_the_written_file_carries_the_library_s_node_layout(
    tmp_path: Path,
) -> None:
    """A written file had no node attributes, no root datasets and no
    version node, and cgnscheck died on its first probe. Every node the
    library reads is spelled here, typed and labelled."""
    verts = np.vstack([_TET, [[1, 1, 1]]])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    path = tmp_path / "out.cgns"
    write(poly, path)
    with h5py.File(path) as f:
        assert f.attrs["name"].startswith(b"HDF5 MotherNode")
        assert f.attrs["label"].startswith(b"Root Node of HDF5 File")
        assert f.attrs["type"].startswith(b"MT")
        assert f.attrs.get_id("name").dtype.itemsize == 33
        assert f.attrs.get_id("type").dtype.itemsize == 3
        assert bytes(f[" format"][()]).startswith(b"IEEE_LITTLE_32\x00")
        assert f[" hdf5version"].shape == (33,)
        assert f["CGNSLibraryVersion/ data"][()][0] >= 3.0
        assert f["CGNSLibraryVersion"].attrs["type"].startswith(b"R4")
        assert f["Base/ data"][()].tolist() == [3, 3]
        assert f["Base/Zone/ data"][()].tolist() == [[5], [1], [0]]
        assert bytes(f["Base/Zone/ZoneType/ data"][()]) == b"Unstructured"
        for name in (
            "Base",
            "Base/Zone",
            "Base/Zone/GridCoordinates",
            "Base/Zone/TRI_3",
        ):
            for key in ("name", "label", "type", "flags"):
                assert key in f[name].attrs, (name, key)
        assert f["Base/Zone/TRI_3/ data"][()].tolist() == [5, 0]
        assert f["Base/Zone/TETRA_4/ data"][()].tolist() == [10, 0]
        assert f["Base/Zone/TRI_3/ElementRange/ data"][()].tolist() == [1, 1]
        assert f["Base/Zone/TETRA_4/ElementRange/ data"][()].tolist() == [2, 2]
        assert f["Base/Zone/TETRA_4/ElementConnectivity/ data"][()].tolist() == [
            1,
            2,
            3,
            4,
        ]
        assert f["Base/Zone/GridCoordinates/CoordinateX/ data"].compression is None
    back = read(path)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_tags_go_out_as_boundary_conditions_at_the_right_location(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        vertex_tags={"pinned": np.array([0, 2])},
        element_tags={"skin": np.array([1]), "body": np.array([0, 1])},
    )
    path = tmp_path / "bc.cgns"
    write(poly, path)
    with h5py.File(path) as f:
        zone_bc = f["Base/Zone/ZoneBC"]
        assert bytes(zone_bc["pinned/GridLocation/ data"][()]) == b"Vertex"
        assert bytes(zone_bc["skin/GridLocation/ data"][()]) == b"FaceCenter"
        assert bytes(zone_bc["body/GridLocation/ data"][()]) == b"CellCenter"
        assert zone_bc["pinned/PointList/ data"].shape == (2, 1)
        # Sections go out by ascending type: the triangle is element 1.
        assert zone_bc["skin/PointList/ data"][()].ravel().tolist() == [1]
    back = read(path)
    assert back.vertex_tags["pinned"].tolist() == [0, 2]
    assert back.element_types.tolist() == [5, 10]
    assert back.element_tags["skin"].tolist() == [0]
    assert back.element_tags["body"].tolist() == [0, 1]


def test_polygons_write_as_an_ngon_section_with_offsets(tmp_path: Path) -> None:
    square = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float64
    )
    poly = make_polydata(
        square,
        [("polygon", np.array([[0, 1, 2, 3]])), ("polygon", np.array([[1, 4, 2]]))],
        element_attrs={"e": np.array([1.0, 2.0])},
    )
    path = tmp_path / "ngon.cgns"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["Base/Zone/NGON_n/ElementStartOffset/ data"][()].tolist() == [0, 4, 7]
        assert f["Base/Zone/NGON_n/ElementConnectivity/ data"][()].tolist() == [
            1,
            2,
            3,
            4,
            2,
            5,
            3,
        ]
    back = read(path)
    assert back.connectivity.tolist() == poly.connectivity.tolist()
    np.testing.assert_array_equal(back.element_attrs["e"], [1.0, 2.0])


def test_a_vector_attribute_is_one_array_of_components(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={
            "v": np.arange(12.0).reshape(4, 3),
            "flag": np.array([True, False, True, False]),
        },
        global_attrs={"gnum": 42, "note": "hello", "odd": {"a": 1}},
    )
    path = tmp_path / "vec.cgns"
    with pytest.warns(UserWarning, match=r"\['odd'\]"):
        write(poly, path)
    with h5py.File(path) as f:
        assert f["Base/Zone/VertexSolution/v/ data"].shape == (3, 4)
        assert f["Base/Zone/VertexSolution/flag/ data"].dtype == np.int32
        assert bytes(f["Base/polyxios/note/ data"][()]) == b"hello"
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["v"], poly.vertex_attrs["v"])
    assert back.global_attrs["gnum"] == 42
    assert back.global_attrs["note"] == "hello"


def test_an_element_type_cgns_cannot_name_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET,
        [
            ("triangle_strip", np.array([[0, 1, 2, 3]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        element_tags={"all": np.array([0, 1])},
    )
    path = tmp_path / "drop.cgns"
    with pytest.warns(UserWarning, match=r"\['triangle_strip'\]"):
        write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [10]
    assert back.element_tags["all"].tolist() == [0]


def test_names_come_from_the_options_or_the_globals(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET, [("tetra", np.array([[0, 1, 2, 3]]))], global_attrs={"zone_name": "fluid"}
    )
    path = tmp_path / "names.cgns"
    write(poly, path, base_name="Main")
    with h5py.File(path) as f:
        assert list(f["Main"]) == [" data", "fluid"]
    assert read(path).global_attrs == {"base_name": "Main", "zone_name": "fluid"}


def test_compression_is_passed_to_the_datasets(tmp_path: Path) -> None:
    poly = make_polydata(_CUBE, [("hexahedron", np.array([_HEX]))])
    path = tmp_path / "gz.cgns"
    write(poly, path, compression="gzip", compression_opts=4)
    with h5py.File(path) as f:
        assert f["Base/Zone/GridCoordinates/CoordinateX/ data"].compression == "gzip"
    with pytest.raises(CodecError, match="compression_opts"):
        write(poly, path, compression_opts=4)


def test_a_buffer_and_a_gzip_name_both_work(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    buffer = io.BytesIO()
    polyxios.write(poly, buffer, fmt=".cgns")
    buffer.seek(0)
    assert polyxios.read(buffer, fmt=".cgns").connectivity.tolist() == [0, 1, 2, 3]
    packed = tmp_path / "t.cgns.gz"
    polyxios.write(poly, packed)
    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert polyxios.read(packed).connectivity.tolist() == [0, 1, 2, 3]


def test_the_codec_is_registered_for_cgns() -> None:
    assert ".cgns" in polyxios.supported_extensions()
    assert ELEMENT_TYPES_INV[10] == "tetra"


# ---------------------------------------------------------------------------
# Structured zones, partial solutions and the file's own numbering
# ---------------------------------------------------------------------------


def _lattice(path: Path, *, bc=None, solutions=None):
    """A (3, 2, 2) structured zone: 12 vertices, 2 hexahedra along i."""
    f, base = _file(path)
    z = _node(
        base, "Z", "Zone_t", np.array([[3, 2, 2], [2, 1, 1], [0, 0, 0]], dtype=np.int32)
    )
    _node(z, "ZoneType", "ZoneType_t", "Structured")
    grid = _node(z, "GridCoordinates", "GridCoordinates_t")
    _node(grid, "CoordinateX", "DataArray_t", np.tile(np.arange(3.0), (2, 2, 1)))
    _node(
        grid, "CoordinateY", "DataArray_t", np.tile(np.arange(2.0)[:, None], (2, 1, 3))
    )
    _node(
        grid,
        "CoordinateZ",
        "DataArray_t",
        np.tile(np.arange(2.0)[:, None, None], (1, 2, 3)),
    )
    for name, location, node_name, value in bc or ():
        zone_bc = z["ZoneBC"] if "ZoneBC" in z else _node(z, "ZoneBC", "ZoneBC_t")
        cond = _node(zone_bc, name, "BC_t", "BCWall")
        _node(cond, "GridLocation", "GridLocation_t", location)
        _node(cond, node_name, "IndexArray_t", np.asarray(value, dtype=np.int32))
    for name, location, arrays in solutions or ():
        sol = _node(z, name, "FlowSolution_t")
        _node(sol, "GridLocation", "GridLocation_t", location)
        for array_name, values in arrays.items():
            _node(sol, array_name, "DataArray_t", values)
    f.close()
    return path


def test_a_structured_zone_s_solutions_are_laid_on_its_lattice(tmp_path: Path) -> None:
    """A vertex solution shaped (nk, nj, ni) was transposed into (nj*ni, nk)
    and skipped for its length; it flattens i-fastest, as the coordinates do."""
    on_points = np.arange(12.0).reshape(2, 2, 3)
    on_cells = np.array([[[7.0, 8.0]]])
    path = _lattice(
        tmp_path / "grid.cgns",
        solutions=[
            ("Points", "Vertex", {"T": on_points}),
            ("Cells", "CellCenter", {"p": on_cells}),
        ],
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["T"], np.arange(12.0))
    np.testing.assert_array_equal(poly.element_attrs["p"], [7.0, 8.0])
    assert poly.vertex_attrs["T"][1] == 1.0
    np.testing.assert_array_equal(poly.vertices[1], [1, 0, 0])


def test_a_structured_boundary_range_becomes_a_tag_over_the_lattice(
    tmp_path: Path,
) -> None:
    path = _lattice(
        tmp_path / "bc.cgns",
        bc=[
            # The i=1 face: (1,1,1)..(1,2,2), stored as (2, IndexDimension).
            ("imin", "Vertex", "PointRange", [[1, 1, 1], [1, 2, 2]]),
            ("second", "CellCenter", "PointRange", [[2, 1, 1], [2, 1, 1]]),
            ("corners", "Vertex", "PointList", [[1, 1, 1], [3, 2, 2]]),
            ("face", "IFaceCenter", "PointRange", [[1, 1, 1], [1, 2, 2]]),
        ],
    )
    poly = read(path)
    assert poly.vertex_tags["imin"].tolist() == [0, 3, 6, 9]
    assert poly.element_tags["second"].tolist() == [1]
    assert poly.vertex_tags["corners"].tolist() == [0, 11]
    assert "face" not in poly.element_tags and "face" not in poly.vertex_tags


def test_a_solution_with_a_point_list_lands_on_the_listed_elements(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    _section(z, "Faces", 5, 2, [1, 2, 3, 1, 2, 4, 2, 3, 4])
    cells = _node(z, "CellSolution", "FlowSolution_t")
    _node(cells, "GridLocation", "GridLocation_t", "CellCenter")
    _node(cells, "id", "DataArray_t", np.array([10], dtype=np.int32))
    _node(cells, "only", "DataArray_t", np.array([1.5]))
    faces = _node(z, "FaceSolution", "FlowSolution_t")
    _node(faces, "GridLocation", "GridLocation_t", "FaceCenter")
    _node(faces, "PointList", "IndexArray_t", np.array([[4], [2]], dtype=np.int32))
    _node(faces, "id", "DataArray_t", np.array([40, 20], dtype=np.int32))
    _node(faces, "short", "DataArray_t", np.array([1.0]))
    points = _node(z, "PointSolution", "FlowSolution_t")
    _node(points, "PointRange", "IndexRange_t", np.array([2, 3], dtype=np.int32))
    _node(points, "v", "DataArray_t", np.array([[5.0, 6.0], [7.0, 8.0]]))
    f.close()
    with pytest.warns(UserWarning, match="'short'"):
        poly = read(path)
    # Faces numbered 2..4 read ahead of the tetrahedron: elements 0..2, then 3.
    assert poly.element_types.tolist() == [5, 5, 5, 10]
    np.testing.assert_array_equal(poly.element_attrs["id"], [20, np.nan, 40, 10])
    np.testing.assert_array_equal(poly.element_attrs["only"], [np.nan] * 3 + [1.5])
    np.testing.assert_array_equal(
        poly.vertex_attrs["v"], [[np.nan] * 2, [5.0, 7.0], [6.0, 8.0], [np.nan] * 2]
    )


def test_a_face_centred_solution_over_every_face_needs_no_list(tmp_path: Path) -> None:
    path = tmp_path / "faces.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Faces", 5, 1, [1, 2, 3, 1, 2, 4])
    _section(z, "Cells", 10, 3, [1, 2, 3, 4])
    sol = _node(z, "Skin", "FlowSolution_t")
    _node(sol, "GridLocation", "GridLocation_t", "FaceCenter")
    _node(sol, "h", "DataArray_t", np.array([0.1, 0.2]))
    full = _node(z, "All", "FlowSolution_t")
    _node(full, "GridLocation", "GridLocation_t", "CellCenter")
    _node(full, "e", "DataArray_t", np.array([1, 2, 3], dtype=np.int32))
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.element_attrs["h"], [0.1, 0.2, np.nan])
    assert poly.element_attrs["e"].tolist() == [1, 2, 3]
    assert poly.element_attrs["e"].dtype == np.int32


def test_element_values_split_by_dimension_and_the_zone_counts_its_cells(
    tmp_path: Path,
) -> None:
    """A cell-centred solution held every element, faces included, while the
    zone declared the cells alone. The cells' solution is as long as the zone
    says; the faces get one of their own with a PointList, and both come back
    onto one attribute without a NaN, its integers still integers."""
    verts = np.vstack([_TET, [[1, 1, 1]]])
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2], [1, 2, 4]])),
            ("line", np.array([[3, 4]])),
        ],
        element_attrs={
            "n": np.array([10, 20, 30, 40], dtype=np.int32),
            "v": np.arange(12.0).reshape(4, 3),
        },
    )
    path = tmp_path / "split.cgns"
    write(poly, path)
    with h5py.File(path) as f:
        zone = f["Base/Zone"]
        assert zone[" data"][()].ravel().tolist() == [5, 1, 0]
        assert bytes(zone["CellSolution/GridLocation/ data"][()]) == b"CellCenter"
        assert "PointList" not in zone["CellSolution"]
        assert zone["CellSolution/n/ data"][()].tolist() == [10]
        assert bytes(zone["FaceSolution/GridLocation/ data"][()]) == b"FaceCenter"
        # Sections go out by type: the line is 1, the triangles 2..3, the tet 4.
        assert zone["FaceSolution/PointList/ data"][()].ravel().tolist() == [2, 3]
        assert zone["FaceSolution/n/ data"][()].tolist() == [20, 30]
        assert zone["FaceSolution/v/ data"].shape == (3, 2)
        assert bytes(zone["EdgeSolution/GridLocation/ data"][()]) == b"EdgeCenter"
        assert zone["EdgeSolution/PointList/ data"][()].ravel().tolist() == [1]
    back = read(path)
    assert back.element_types.tolist() == [3, 5, 5, 10]
    assert back.element_attrs["n"].tolist() == [40, 20, 30, 10]
    assert back.element_attrs["n"].dtype == np.int32
    np.testing.assert_array_equal(
        back.element_attrs["v"], poly.element_attrs["v"][[3, 1, 2, 0]]
    )


def test_attribute_names_that_collide_as_nodes_are_kept_apart(tmp_path: Path) -> None:
    """Two attributes sharing their first 32 characters, or one called
    GridLocation, crashed the write on a duplicate group."""
    long_a = "x" * 40 + "a"
    long_b = "x" * 40 + "b"
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={
            long_a: np.arange(4.0),
            long_b: np.arange(4.0) + 10,
            "GridLocation": np.arange(4.0) + 20,
        },
        global_attrs={"y" * 40 + "1": 1, "y" * 40 + "2": 2, "z" * 40: "text"},
    )
    path = tmp_path / "names.cgns"
    # Every name that changed is named once, with what it became.
    with pytest.warns(UserWarning, match=r"'GridLocation'.*'GridLocation_2'") as seen:
        write(poly, path)
    assert len(seen) == 1
    assert f"'{long_b}'" in str(seen[0].message)
    assert f"'{'y' * 40 + '2'}'" in str(seen[0].message)
    back = read(path)
    # The suffix stays within the 32 characters a node name has.
    assert sorted(back.vertex_attrs) == ["GridLocation_2", "x" * 30 + "_2", "x" * 32]
    np.testing.assert_array_equal(
        back.vertex_attrs["x" * 30 + "_2"], np.arange(4.0) + 10
    )
    np.testing.assert_array_equal(
        back.vertex_attrs["GridLocation_2"], np.arange(4.0) + 20
    )
    assert back.global_attrs["y" * 32] == 1
    assert back.global_attrs["y" * 30 + "_2"] == 2
    assert back.global_attrs["z" * 32] == "text"


def test_polygons_keep_their_place_among_the_types(tmp_path: Path) -> None:
    """Polygons went out after every other section, so a mesh of a tetrahedron
    then a polygon came back the other way round."""
    verts = np.vstack([_TET, [[1, 1, 0]]])
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("polygon", np.array([[0, 1, 4, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        element_attrs={"e": np.array([1, 2, 3])},
    )
    path = tmp_path / "order.cgns"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["Base/Zone/TRI_3/ElementRange/ data"][()].tolist() == [1, 1]
        assert f["Base/Zone/NGON_n/ElementRange/ data"][()].tolist() == [2, 2]
        assert f["Base/Zone/TETRA_4/ElementRange/ data"][()].tolist() == [3, 3]
    back = read(path)
    assert back.element_types.tolist() == poly.element_types.tolist()
    assert back.connectivity.tolist() == poly.connectivity.tolist()
    assert back.element_attrs["e"].tolist() == [1, 2, 3]


def test_a_range_below_one_or_over_another_section_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "zero.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Cells", 10, 0, [1, 2, 3, 4])
    f.close()
    with pytest.raises(CodecError, match="start at one"):
        read(path)

    path = tmp_path / "overlap.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    _section(z, "Faces", 5, 1, [1, 2, 3])
    f.close()
    with pytest.raises(CodecError, match="both number element 1"):
        read(path)


def test_an_empty_section_is_walked_past(tmp_path: Path) -> None:
    path = tmp_path / "empty.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    s = _node(z, "Nothing", "Elements_t", np.array([5, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([1, 0], dtype=np.int32))
    _node(s, "ElementConnectivity", "DataArray_t", np.zeros(0, dtype=np.int32))
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    f.close()
    assert read(path).element_types.tolist() == [10]


def test_a_zone_declaring_another_vertex_count_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "count.cgns"
    f, base = _file(path)
    z = _node(base, "Z", "Zone_t", np.array([[5], [1], [0]], dtype=np.int32))
    _node(z, "ZoneType", "ZoneType_t", "Unstructured")
    grid = _node(z, "GridCoordinates", "GridCoordinates_t")
    for k, axis in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
        _node(grid, axis, "DataArray_t", np.ascontiguousarray(_TET[:, k]))
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    f.close()
    with pytest.raises(
        CodecError, match="declares 5 vertices and holds coordinates for 4"
    ):
        read(path)


def test_unsigned_values_go_out_as_the_integer_that_fits(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={
            "small": np.array([1, 2, 3, 4], dtype=np.uint8),
            "wide": np.array([1, 2, 3, 2**40], dtype=np.uint64),
        },
    )
    path = tmp_path / "unsigned.cgns"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["Base/Zone/VertexSolution/small"].attrs["type"].startswith(b"I4")
        assert f["Base/Zone/VertexSolution/wide"].attrs["type"].startswith(b"I8")
    back = read(path)
    assert back.vertex_attrs["wide"].tolist() == [1, 2, 3, 2**40]
    too_big = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"huge": np.array([0, 0, 0, 2**63], dtype=np.uint64)},
    )
    with pytest.raises(CodecError, match="'huge'"):
        write(too_big, tmp_path / "huge.cgns")


def test_a_zone_named_like_a_boundary_condition_keeps_both_tags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clash.cgns"
    f, base = _file(path)
    z1 = _zone(base, "wall", _TET, 1)
    _section(z1, "Cells", 10, 1, [1, 2, 3, 4])
    bc = _node(z1, "ZoneBC", "ZoneBC_t")
    wall = _node(bc, "wall", "BC_t", "BCWall")
    _node(wall, "GridLocation", "GridLocation_t", "CellCenter")
    _node(wall, "PointList", "IndexArray_t", np.array([[1]], dtype=np.int32))
    z2 = _zone(base, "other", _CUBE, 1)
    _section(z2, "Hexas", 17, 1, np.array(_HEX) + 1)
    f.close()
    poly = read(path)
    # Zones read in name order: 'other' holds element 0, 'wall' element 1.
    assert poly.element_tags["other"].tolist() == [0]
    assert poly.element_tags["wall"].tolist() == [1]
    assert poly.element_tags["wall_2"].tolist() == [1]


def test_the_27_node_hexahedron_and_18_node_wedge_are_reordered(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    path = tmp_path / "high.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", rng.random((45, 3)), 2)
    _section(z, "Hex", 19, 1, np.arange(1, 28))
    s = _node(z, "Wedge", "Elements_t", np.array([16, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([2, 2], dtype=np.int32))
    _node(s, "ElementConnectivity", "DataArray_t", np.arange(28, 46, dtype=np.int32))
    f.close()
    poly = read(path)
    hexa = poly.connectivity[:27].tolist()
    assert hexa == [
        *range(8),
        *(8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15),
        24,
        22,
        21,
        23,
        20,
        25,
        26,
    ]
    wedge = (poly.connectivity[27:] - 27).tolist()
    assert wedge == [*range(6), 6, 7, 8, 12, 13, 14, 9, 10, 11, 15, 16, 17]
    out = tmp_path / "back.cgns"
    write(poly, out)
    with h5py.File(out) as g:
        assert g["Base/Z/HEXA_27/ElementConnectivity/ data"][()].tolist() == list(
            range(1, 28)
        )
        assert g["Base/Z/PENTA_18/ElementConnectivity/ data"][()].tolist() == list(
            range(28, 46)
        )


def test_a_vertex_point_range_spans_the_vertices(tmp_path: Path) -> None:
    path = tmp_path / "vrange.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    bc = _node(z, "ZoneBC", "ZoneBC_t")
    fixed = _node(bc, "fixed", "BC_t", "BCWall")
    _node(fixed, "PointRange", "IndexRange_t", np.array([[2], [4]], dtype=np.int32))
    f.close()
    assert read(path).vertex_tags["fixed"].tolist() == [1, 2, 3]


def test_an_element_of_the_wrong_width_is_dropped_by_count(tmp_path: Path) -> None:
    poly = polyxios.PolyData(
        vertices=_TET,
        connectivity=np.array([0, 1, 2, 3, 0, 1, 2, 3]),
        offsets=np.array([0, 4, 8]),
        element_types=np.array([5, 10], dtype=np.uint8),
    )
    path = tmp_path / "width.cgns"
    with pytest.warns(UserWarning, match=r"1 element\(s\) hold a node count"):
        write(poly, path)
    assert read(path).element_types.tolist() == [10]


def test_a_condition_spelled_as_an_element_list_or_range_is_a_tag_on_faces(
    tmp_path: Path,
) -> None:
    """CGNS 2 put ``ElementList`` or ``ElementRange`` under a ``BC_t``; the
    numbers are element numbers whatever the GridLocation says."""
    path = tmp_path / "bc2.cgns"
    f, base = _file(path)
    z = _zone(base, "Z", _TET, 1)
    _section(z, "Cells", 10, 1, [1, 2, 3, 4])
    _section(z, "Faces", 5, 2, [1, 2, 3, 1, 2, 4, 2, 3, 4])
    bc = _node(z, "ZoneBC", "ZoneBC_t")
    wall = _node(bc, "wall", "BC_t", "BCWall")
    _node(wall, "ElementList", "IndexArray_t", np.array([[2], [4]], dtype=np.int32))
    inlet = _node(bc, "inlet", "BC_t", "BCInflow")
    _node(inlet, "ElementRange", "IndexRange_t", np.array([3, 3], dtype=np.int32))
    f.close()
    poly = read(path)
    assert poly.element_tags["wall"].tolist() == [0, 2]
    assert poly.element_tags["inlet"].tolist() == [1]
    assert poly.vertex_tags == {}


def test_a_mixed_section_keeps_polygon_widths_in_the_order_they_first_appear(
    tmp_path: Path,
) -> None:
    """Older MIXED streams spell a polygon's node count after its code; the
    triangles gather as one block and the polygons one block per width."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    path = tmp_path / "mixed_ngon.cgns"
    f, base = _file(path, cell_dim=2)
    z = _zone(base, "Z", points, 4)
    s = _node(z, "Mixed", "Elements_t", np.array([20, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([1, 4], dtype=np.int32))
    _node(
        s,
        "ElementConnectivity",
        "DataArray_t",
        np.array(
            [5, 1, 2, 3, 22, 5, 1, 2, 5, 6, 3, 22, 4, 1, 2, 3, 4, 5, 2, 5, 3],
            dtype=np.int32,
        ),
    )
    f.close()
    poly = read(path)
    assert poly.element_types.tolist() == [5, 5, 7, 7]
    assert poly.connectivity.tolist() == [
        *[0, 1, 2, 1, 4, 2],
        *[0, 1, 4, 5, 2],
        *[0, 1, 2, 3],
    ]
    bad = tmp_path / "wide.cgns"
    f, base = _file(bad, cell_dim=2)
    z = _zone(base, "Z", points, 1)
    s = _node(z, "Mixed", "Elements_t", np.array([20, 0], dtype=np.int32))
    _node(s, "ElementRange", "IndexRange_t", np.array([1, 1], dtype=np.int32))
    _node(
        s,
        "ElementConnectivity",
        "DataArray_t",
        np.array([5, 1, 2, 3, 4], dtype=np.int32),
    )
    _node(s, "ElementStartOffset", "DataArray_t", np.array([0, 5], dtype=np.int32))
    f.close()
    with pytest.raises(CodecError, match="element 1 is a TRI_3 of 4 nodes, not 3"):
        read(bad)
