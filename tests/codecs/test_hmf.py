"""HMF: XDMF's model in one HDF5 file.

Every test here needs h5py and skips without it. Files are built with h5py
directly, in the layout the format's originator writes, so the reader is
tested against the format and not against the writer beside it.
"""

from __future__ import annotations

import io
from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._optpkg import TripWire
from polyxios.codecs import _hdf5
from polyxios.codecs._hmf import read, write
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError

h5py = pytest.importorskip("h5py")

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _file(path: Path, points: np.ndarray, topologies: list[tuple[str, np.ndarray]]):
    f = h5py.File(path, "w")
    f.attrs["type"] = "hmf"
    f.attrs["version"] = "0.1-alpha"
    grid = f.create_group("domain").create_group("grid")
    geo = grid.create_dataset("Geometry", data=points)
    geo.attrs["GeometryType"] = "XYZ"[: points.shape[1]]
    for k, (topo, cells) in enumerate(topologies):
        grid.create_dataset(f"Topology{k}", data=np.asarray(cells)).attrs[
            "TopologyType"
        ] = topo
    return f, grid


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_the_originator_s_layout_reads(tmp_path: Path) -> None:
    path = tmp_path / "m.hmf"
    f, grid = _file(
        path, _TET, [("Tetrahedron", [[0, 1, 2, 3]]), ("Triangle", [[0, 1, 2]])]
    )
    grid.create_dataset("NodeAttributes/phi", data=np.arange(4.0))
    grid.create_dataset("CellAttributes/rho", data=np.array([1.0, 2.0]))
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertices, _TET)
    # Topologies keep the order they were numbered in.
    assert poly.element_types.tolist() == [10, 5]
    np.testing.assert_array_equal(poly.vertex_attrs["phi"], np.arange(4.0))
    np.testing.assert_array_equal(poly.element_attrs["rho"], [1.0, 2.0])
    assert poly.global_attrs == {}


def test_topologies_are_read_in_numeric_not_lexical_order(tmp_path: Path) -> None:
    path = tmp_path / "ten.hmf"
    f, grid = _file(path, _TET, [])
    for k in range(11):
        grid.create_dataset(f"Topology{k}", data=np.array([[k % 4]])).attrs[
            "TopologyType"
        ] = "Polyvertex"
    f.close()
    assert read(path).connectivity.tolist() == [k % 4 for k in range(11)]


def test_a_two_dimensional_geometry_is_padded_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "flat.hmf"
    _file(path, _TET[:3, :2], [("Triangle", [[0, 1, 2]])])[0].close()
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    assert poly.global_attrs["was_2d"] is True


def test_an_attribute_of_the_wrong_length_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "odd.hmf"
    f, grid = _file(path, _TET, [("Tetrahedron", [[0, 1, 2, 3]])])
    grid.create_dataset("NodeAttributes/short", data=np.arange(3.0))
    f.close()
    with pytest.warns(UserWarning, match="'NodeAttributes/short'"):
        poly = read(path)
    assert "short" not in poly.vertex_attrs


def test_a_topology_xdmf_does_not_name_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "odd.hmf"
    _file(path, _TET, [("Blob", [[0, 1, 2, 3]])])[0].close()
    with pytest.raises(CodecError, match="'Blob'"):
        read(path)


def test_a_topology_of_the_wrong_width_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "wide.hmf"
    _file(path, _TET, [("Tetrahedron", [[0, 1, 2]])])[0].close()
    with pytest.raises(CodecError, match="3 nodes per Tetrahedron, not 4"):
        read(path)


def test_a_node_outside_the_geometry_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.hmf"
    _file(path, _TET, [("Tetrahedron", [[0, 1, 2, 9]])])[0].close()
    with pytest.raises(CodecError, match="outside 0..3"):
        read(path)


def test_a_file_that_does_not_call_itself_hmf_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.hmf"
    with h5py.File(path, "w") as f:
        f.create_group("domain")
    with pytest.raises(CodecError, match="call itself 'hmf'"):
        read(path)


def test_lazy_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.hmf"
    _file(path, _TET, [("Tetrahedron", [[0, 1, 2, 3]])])[0].close()
    with pytest.raises(LazyReadError):
        read(path, lazy=True)


def test_without_h5py_the_read_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "m.hmf"
    _file(path, _TET, [("Tetrahedron", [[0, 1, 2, 3]])])[0].close()
    monkeypatch.setattr(_hdf5, "_h5py", lambda: (TripWire("no h5py"), False))
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
        read(path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_written_file_is_the_originator_s_layout_plus_sets(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"phi": np.arange(4.0)},
        element_attrs={"rho": np.array([1.0, 2.0])},
        vertex_tags={"base": np.array([0, 1, 2])},
        element_tags={"skin": np.array([1])},
        global_attrs={"gnum": 42, "note": "hello"},
    )
    path = tmp_path / "out.hmf"
    write(poly, path)
    with h5py.File(path) as f:
        assert f.attrs["type"] == "hmf"
        assert f.attrs["version"] == "0.1-alpha"
        grid = f["domain/grid"]
        assert grid["Geometry"].attrs["GeometryType"] == "XYZ"
        # Types go out ascending: the triangle first, then the tetrahedron.
        assert grid["Topology0"].attrs["TopologyType"] == "Triangle"
        assert grid["Topology1"].attrs["TopologyType"] == "Tetrahedron"
        assert grid["CellAttributes/rho"][()].tolist() == [2.0, 1.0]
        assert grid["CellSets/skin"][()].tolist() == [0]
        assert grid["NodeSets/base"][()].tolist() == [0, 1, 2]
        assert grid["Attributes/gnum"][()] == 42
        assert grid["Attributes/note"][()][0] == b"hello"
    back = read(path)
    assert back.element_types.tolist() == [5, 10]
    assert back.element_tags["skin"].tolist() == [0]
    assert back.global_attrs == {"gnum": 42, "note": "hello"}


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
    path = tmp_path / "lattice.hmf"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["domain/grid/Topology0"].attrs["TopologyType"] == "Quadrilateral"
        assert f["domain/grid/Topology1"].attrs["TopologyType"] == "Hexahedron"
    assert read(path).connectivity.tolist() == [0, 1, 3, 2, 0, 1, 3, 2, 4, 5, 7, 6]


def test_an_element_type_xdmf_cannot_name_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET,
        [
            ("triangle_strip", np.array([[0, 1, 2, 3]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    path = tmp_path / "drop.hmf"
    with pytest.warns(UserWarning, match=r"\['triangle_strip'\]"):
        write(poly, path)
    assert read(path).element_types.tolist() == [10]


def test_a_flat_mesh_from_a_flat_file_stays_two_dimensional(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET[:3], [("triangle", np.array([[0, 1, 2]]))], global_attrs={"was_2d": True}
    )
    path = tmp_path / "flat.hmf"
    write(poly, path)
    with h5py.File(path) as f:
        assert f["domain/grid/Geometry"].shape == (3, 2)
        assert f["domain/grid/Geometry"].attrs["GeometryType"] == "XY"


def test_compression_is_passed_to_the_datasets(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "gz.hmf"
    write(poly, path, compression="gzip", compression_opts=4)
    with h5py.File(path) as f:
        assert f["domain/grid/Geometry"].compression == "gzip"


def test_a_buffer_and_a_gzip_name_both_work(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    buffer = io.BytesIO()
    polyxios.write(poly, buffer, fmt=".hmf")
    buffer.seek(0)
    assert polyxios.read(buffer, fmt=".hmf").connectivity.tolist() == [0, 1, 2, 3]
    packed = tmp_path / "t.hmf.gz"
    polyxios.write(poly, packed)
    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert polyxios.read(packed).connectivity.tolist() == [0, 1, 2, 3]


def test_the_codec_is_registered_for_hmf() -> None:
    assert ".hmf" in polyxios.supported_extensions()


def test_polygons_of_one_width_go_out_as_a_polygon_table(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("polygon", np.array([[0, 1, 2, 3], [1, 4, 5, 2]])),
            ("triangle", np.array([[4, 6, 5]])),
        ],
        element_attrs={"rho": np.array([1.0, 2.0, 3.0])},
    )
    path = tmp_path / "poly.hmf"
    write(poly, path)
    with h5py.File(path) as f:
        grid = f["domain/grid"]
        assert grid["Topology0"].attrs["TopologyType"] == "Triangle"
        assert grid["Topology1"].attrs["TopologyType"] == "Polygon"
        assert grid["Topology1"][()].tolist() == [[0, 1, 2, 3], [1, 4, 5, 2]]
        assert grid["CellAttributes/rho"][()].tolist() == [3.0, 1.0, 2.0]
    back = read(path)
    assert back.element_types.tolist() == [5, 7, 7]
    assert back.connectivity.tolist() == [4, 6, 5, 0, 1, 2, 3, 1, 4, 5, 2]


def test_polygons_of_differing_widths_are_dropped_with_a_warning_that_says_so(
    tmp_path: Path,
) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0.5, 0]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("polygon", np.array([[0, 1, 2, 3]])),
            ("polygon", np.array([[1, 4, 2]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    path = tmp_path / "ragged.hmf"
    with pytest.warns(UserWarning, match=r"\['polygon'\].*differ in theirs"):
        write(poly, path)
    assert read(path).element_types.tolist() == [5]


def test_a_mesh_with_nothing_of_polyxios_s_own_is_the_originator_s_layout(
    tmp_path: Path,
) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "plain.hmf"
    write(poly, path)
    with h5py.File(path) as f:
        assert list(f) == ["domain"]
        assert list(f["domain"]) == ["grid"]
        assert sorted(f["domain/grid"]) == [
            "CellAttributes",
            "Geometry",
            "NodeAttributes",
            "Topology0",
        ]


def test_without_h5py_the_write_names_the_extra_before_anything_else(
    tmp_path: Path, monkeypatch
) -> None:
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        global_attrs={"blob": object()},
    )
    monkeypatch.setattr(_hdf5, "_h5py", lambda: (TripWire("no h5py"), False))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
            write(poly, tmp_path / "m.hmf")


def test_another_version_is_read_with_a_warning_that_names_it(tmp_path: Path) -> None:
    path = tmp_path / "v.hmf"
    f, _ = _file(path, _TET, [("Tetrahedron", [[0, 1, 2, 3]])])
    f.attrs["version"] = "0.2"
    f.close()
    with pytest.warns(UserWarning, match=r"is HMF 0\.2; this reader knows 0\.1-alpha"):
        poly = read(path)
    assert poly.element_types.tolist() == [10]
    with h5py.File(path, "r+") as f:
        del f.attrs["version"]
    with pytest.warns(UserWarning, match="of no version"):
        read(path)
