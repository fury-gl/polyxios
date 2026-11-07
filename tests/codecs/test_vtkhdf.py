"""VTKHDF: VTK's own HDF5 layout.

Every test here needs h5py and skips without it, but one, which stands in
for the missing package to check the refusal names the extra that installs
it. Files are built with h5py directly, in the layout VTK's writer emits, so
the reader is tested against the format and not against the writer beside
it. The interop tests at the end need VTK itself and skip without it.
"""

from __future__ import annotations

import dataclasses
import gzip
import io
from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import helper, make_polydata
from polyxios._image import ImageData
from polyxios._optpkg import TripWire
from polyxios.codecs import _hdf5, _vtkhdf
from polyxios.codecs._vtkhdf import read, read_time_series, write, write_time_series
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    UnsupportedFormatError,
    ValidationError,
)

h5py = pytest.importorskip("h5py")

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_SQUARE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float64
)


def _root(path: Path, kind: str, version=(2, 0)):
    f = h5py.File(path, "w")
    root = f.create_group("VTKHDF")
    root.attrs["Version"] = np.array(version, dtype=np.int64)
    root.attrs["Type"] = np.bytes_(kind)
    return f, root


def _ug(path: Path, *, points=_TET, cells=((10, [0, 1, 2, 3]), (5, [0, 1, 2]))):
    """One-part UnstructuredGrid in the layout VTK writes."""
    f, root = _root(path, "UnstructuredGrid")
    conn = np.concatenate([np.asarray(c, dtype=np.int64) for _, c in cells])
    offsets = np.concatenate([[0], np.cumsum([len(c) for _, c in cells])]).astype(
        np.int64
    )
    root.create_dataset("NumberOfPoints", data=np.array([len(points)], dtype=np.int64))
    root.create_dataset("NumberOfCells", data=np.array([len(cells)], dtype=np.int64))
    root.create_dataset(
        "NumberOfConnectivityIds", data=np.array([conn.size], dtype=np.int64)
    )
    root.create_dataset("Points", data=np.asarray(points, dtype=np.float32))
    root.create_dataset("Types", data=np.array([t for t, _ in cells], dtype=np.uint8))
    root.create_dataset("Connectivity", data=conn)
    root.create_dataset("Offsets", data=offsets)
    return f, root


def _steps(root, times, **tables):
    steps = root.create_group("Steps")
    steps.attrs["NSteps"] = np.int64(len(times))
    steps.create_dataset("Values", data=np.asarray(times, dtype=np.float32))
    n = len(times)
    defaults = {
        "PartOffsets": np.zeros(n, dtype=np.int64),
        "NumberOfParts": np.ones(n, dtype=np.int64),
        "PointOffsets": np.zeros(n, dtype=np.int64),
        "CellOffsets": np.zeros(n, dtype=np.int64),
        "ConnectivityIdOffsets": np.zeros(n, dtype=np.int64),
    }
    for key, value in {**defaults, **tables}.items():
        if value is not None:
            steps.create_dataset(key, data=np.asarray(value, dtype=np.int64))
    return steps


# ---------------------------------------------------------------------------
# Reading: UnstructuredGrid
# ---------------------------------------------------------------------------


def test_vtk_s_unstructured_layout_reads(tmp_path: Path) -> None:
    path = tmp_path / "m.vtkhdf"
    f, root = _ug(path)
    root.create_dataset("PointData/phi", data=np.arange(4.0))
    root.create_dataset("PointData/vec", data=np.arange(12.0).reshape(4, 3))
    root.create_dataset("CellData/rho", data=np.array([1, 2], dtype=np.int32))
    root.create_dataset("FieldData/gnum", data=np.array([42.0]))
    root.create_dataset("FieldData/pair", data=np.array([1.0, 2.0]))
    root.create_dataset(
        "FieldData/case",
        data=np.array(["run 3"], dtype=object),
        dtype=h5py.string_dtype(),
    )
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertices, _TET)
    assert poly.vertices.dtype == np.float64
    assert poly.element_types.tolist() == [10, 5]
    assert poly.offsets.tolist() == [0, 4, 7]
    assert poly.connectivity.tolist() == [0, 1, 2, 3, 0, 1, 2]
    np.testing.assert_array_equal(poly.vertex_attrs["phi"], np.arange(4.0))
    assert poly.vertex_attrs["vec"].shape == (4, 3)
    assert poly.element_attrs["rho"].dtype == np.int32
    assert poly.global_attrs["gnum"] == 42.0
    assert poly.global_attrs["pair"].tolist() == [1.0, 2.0]
    assert poly.global_attrs["case"] == "run 3"
    assert poly.global_attrs["vtkhdf_type"] == "UnstructuredGrid"
    assert "time" not in poly.global_attrs


def test_two_partitions_are_merged_and_tagged(tmp_path: Path) -> None:
    path = tmp_path / "p.vtkhdf"
    f, root = _root(path, "UnstructuredGrid")
    root.create_dataset("NumberOfPoints", data=np.array([4, 3], dtype=np.int64))
    root.create_dataset("NumberOfCells", data=np.array([1, 1], dtype=np.int64))
    root.create_dataset(
        "NumberOfConnectivityIds", data=np.array([4, 3], dtype=np.int64)
    )
    root.create_dataset("Points", data=np.vstack([_TET, _TET[:3] + 5]))
    root.create_dataset("Types", data=np.array([10, 5], dtype=np.uint8))
    root.create_dataset(
        "Connectivity", data=np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)
    )
    # Each part's offsets start at zero: VTK writes ncells + 1 per part.
    root.create_dataset("Offsets", data=np.array([0, 4, 0, 3], dtype=np.int64))
    root.create_dataset("PointData/u", data=np.arange(7.0))
    root.create_dataset("CellData/c", data=np.array([1.0, 2.0]))
    f.close()
    poly = read(path)
    assert poly.vertices.shape == (7, 3)
    assert poly.connectivity.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert poly.offsets.tolist() == [0, 4, 7]
    assert poly.element_tags["part_0"].tolist() == [0]
    assert poly.element_tags["part_1"].tolist() == [1]
    np.testing.assert_array_equal(poly.vertex_attrs["u"], np.arange(7.0))


def test_a_vtk_cell_code_polyxios_lacks_reads_as_an_empty_cell(tmp_path: Path) -> None:
    path = tmp_path / "e.vtkhdf"
    f, root = _ug(path, cells=((10, [0, 1, 2, 3]), (200, [0, 1, 2])))
    f.close()
    assert read(path).element_types.tolist() == [10, 0]


def test_tag_mask_columns_read_back_as_tags(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, root = _ug(path)
    root.create_dataset(
        "PointData/polyxios_tag_top", data=np.array([0, 1, 1, 0], np.uint8)
    )
    root.create_dataset("CellData/polyxios_tag_a", data=np.array([1, 0], np.uint8))
    f.close()
    poly = read(path)
    assert poly.vertex_tags["top"].tolist() == [1, 2]
    assert poly.element_tags["a"].tolist() == [0]
    assert not poly.vertex_attrs and not poly.element_attrs


def test_an_array_of_the_wrong_length_is_dropped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    f, root = _ug(path)
    root.create_dataset("PointData/short", data=np.arange(3.0))
    root.create_dataset("PointData/ok", data=np.arange(4.0))
    f.close()
    with pytest.warns(UserWarning, match="'short' holds 3 rows"):
        poly = read(path)
    assert sorted(poly.vertex_attrs) == ["ok"]


def test_a_scalar_point_array_is_dropped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, root = _ug(path)
    root.create_dataset("PointData/one", data=np.float64(1.0))
    f.close()
    with pytest.warns(UserWarning, match="array 'one' is a scalar"):
        assert not read(path).vertex_attrs


def test_a_text_point_array_is_dropped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    f, root = _ug(path)
    root.create_dataset(
        "PointData/name",
        data=np.array(list("abcd"), dtype=object),
        dtype=h5py.string_dtype(),
    )
    f.close()
    with pytest.warns(UserWarning, match="'name' holds object, which is not numeric"):
        assert not read(path).vertex_attrs


# ---------------------------------------------------------------------------
# Reading: PolyData
# ---------------------------------------------------------------------------


def _polydata(path: Path, sections: dict[str, list[list[int]]], *, points=_SQUARE):
    f, root = _root(path, "PolyData")
    root.create_dataset("NumberOfPoints", data=np.array([len(points)], dtype=np.int64))
    root.create_dataset("Points", data=np.asarray(points, dtype=np.float32))
    for name, cells in sections.items():
        conn = (
            np.concatenate([np.asarray(c, dtype=np.int64) for c in cells])
            if cells
            else np.zeros(0, np.int64)
        )
        offsets = (
            np.concatenate([[0], np.cumsum([len(c) for c in cells])]).astype(np.int64)
            if cells
            else np.zeros(0, np.int64)
        )
        root.create_dataset(
            f"{name}/NumberOfCells", data=np.array([len(cells)], dtype=np.int64)
        )
        root.create_dataset(
            f"{name}/NumberOfConnectivityIds",
            data=np.array([conn.size], dtype=np.int64),
        )
        root.create_dataset(f"{name}/Connectivity", data=conn)
        root.create_dataset(f"{name}/Offsets", data=offsets)
    return f, root


def test_vtk_s_polydata_layout_reads_in_vtk_s_cell_order(tmp_path: Path) -> None:
    path = tmp_path / "p.vtkhdf"
    f, root = _polydata(
        path,
        {
            "Vertices": [[4], [0, 1]],
            "Lines": [[0, 4], [0, 1, 2]],
            "Polygons": [[0, 1, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
            "Strips": [[0, 1, 3, 2]],
        },
    )
    root.create_dataset("CellData/c", data=np.arange(8.0))
    f.close()
    poly = read(path)
    assert poly.element_types.tolist() == [1, 2, 3, 4, 5, 9, 7, 6]
    assert poly.offsets.tolist() == [0, 1, 3, 5, 8, 11, 15, 20, 24]
    np.testing.assert_array_equal(poly.element_attrs["c"], np.arange(8.0))
    assert poly.global_attrs["vtkhdf_type"] == "PolyData"


def test_a_polydata_missing_a_section_group_reads(tmp_path: Path) -> None:
    path = tmp_path / "p.vtkhdf"
    f, _ = _polydata(path, {"Polygons": [[0, 1, 2]]})
    f.close()
    assert read(path).element_types.tolist() == [5]


def test_a_polydata_section_counting_fewer_parts_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "p.vtkhdf"
    f, root = _polydata(path, {"Polygons": [[0, 1, 2]]})
    root["NumberOfPoints"][...] = 5
    del root["Polygons/NumberOfCells"]
    root.create_dataset("Polygons/NumberOfCells", data=np.zeros(0, dtype=np.int64))
    f.close()
    with pytest.raises(CodecError, match="counts fewer parts"):
        read(path)


# ---------------------------------------------------------------------------
# Reading: ImageData
# ---------------------------------------------------------------------------


def _image(path: Path, extent, *, origin=(0, 0, 0), spacing=(1, 1, 1), direction=None):
    f, root = _root(path, "ImageData")
    root.attrs["WholeExtent"] = np.array(extent, dtype=np.int64)
    root.attrs["Origin"] = np.asarray(origin, dtype=np.float64)
    root.attrs["Spacing"] = np.asarray(spacing, dtype=np.float64)
    if direction is not None:
        root.attrs["Direction"] = np.asarray(direction, dtype=np.float64).ravel()
    return f, root


def test_an_image_expands_into_hexahedra_with_its_arrays_reshaped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1], origin=(1, 2, 3), spacing=(0.5, 1, 2))
    root.create_dataset("PointData/p", data=np.arange(12.0).reshape(2, 2, 3))
    root.create_dataset("PointData/v", data=np.arange(36.0).reshape(2, 2, 3, 3))
    root.create_dataset("CellData/c", data=np.arange(2.0).reshape(1, 1, 2))
    f.close()
    poly = read(path)
    assert poly.vertices.shape == (12, 3)
    assert poly.vertices[1].tolist() == [1.5, 2.0, 3.0]
    assert poly.vertices[3].tolist() == [1.0, 3.0, 3.0]
    assert poly.element_types.tolist() == [12, 12]
    np.testing.assert_array_equal(poly.vertex_attrs["p"], np.arange(12.0))
    assert poly.vertex_attrs["v"].shape == (12, 3)
    np.testing.assert_array_equal(poly.element_attrs["c"], [0.0, 1.0])
    assert poly.global_attrs["vti_origin"] == [1.0, 2.0, 3.0]
    assert poly.global_attrs["vti_spacing"] == [0.5, 1.0, 2.0]
    assert poly.global_attrs["vti_extent"] == [0, 2, 0, 1, 0, 1]
    assert "vtkhdf_direction" not in poly.global_attrs


def test_a_flat_image_is_a_sheet_of_quads_and_writes_as_vti(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 2, 0, 0])
    root.create_dataset("CellData/c", data=np.arange(4.0).reshape(1, 2, 2))
    f.close()
    poly = read(path)
    assert poly.element_types.tolist() == [9] * 4
    np.testing.assert_array_equal(poly.element_attrs["c"], np.arange(4.0))
    polyxios.write(poly, tmp_path / "i.vti")
    back = polyxios.read(tmp_path / "i.vti")
    np.testing.assert_array_equal(back.vertices, poly.vertices)


def test_a_flat_image_s_arrays_may_leave_out_the_axis_of_size_one(
    tmp_path: Path,
) -> None:
    """VTK's reader wants a size-1 axis left out, its writer keeps it; both read."""
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 2, 0, 0])
    root.create_dataset("PointData/p", data=np.arange(9.0).reshape(3, 3))
    root.create_dataset("PointData/v", data=np.arange(27.0).reshape(3, 3, 3))
    root.create_dataset("PointData/full", data=np.arange(9.0).reshape(1, 3, 3))
    root.create_dataset("CellData/c", data=np.arange(4.0).reshape(2, 2))
    f.close()
    poly = read(path)
    np.testing.assert_array_equal(poly.vertex_attrs["p"], np.arange(9.0))
    np.testing.assert_array_equal(poly.vertex_attrs["full"], np.arange(9.0))
    assert poly.vertex_attrs["v"].shape == (9, 3)
    assert poly.vertex_attrs["v"][4].tolist() == [12.0, 13.0, 14.0]
    np.testing.assert_array_equal(poly.element_attrs["c"], np.arange(4.0))


def test_a_line_image_s_arrays_may_be_one_dimensional(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 3, 0, 0, 0, 0])
    root.create_dataset("PointData/p", data=np.arange(4.0))
    root.create_dataset("PointData/v", data=np.arange(8.0).reshape(4, 2))
    root.create_dataset("CellData/c", data=np.arange(3.0))
    f.close()
    poly = read(path)
    assert poly.element_types.tolist() == [3] * 3
    np.testing.assert_array_equal(poly.vertex_attrs["p"], np.arange(4.0))
    assert poly.vertex_attrs["v"].shape == (4, 2)
    np.testing.assert_array_equal(poly.element_attrs["c"], np.arange(3.0))


def test_a_temporal_flat_image_may_leave_out_the_axis_of_size_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 2, 0, 0])
    root.create_dataset(
        "PointData/p", data=np.stack([np.full((3, 3), t) for t in (10.0, 20.0)])
    )
    steps = _steps(root, [0.0, 1.0])
    steps.create_dataset("PointDataOffsets/p", data=np.arange(2, dtype=np.int64))
    f.close()
    assert read(path, step=1).vertex_attrs["p"].tolist() == [20.0] * 9


def test_a_text_image_array_is_dropped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 1, 0, 0, 0, 0])
    root.create_dataset(
        "PointData/names",
        data=np.array(["a", "b"], dtype=object),
        dtype=h5py.string_dtype(),
    )
    f.close()
    with pytest.warns(UserWarning, match="array 'names' holds .* not numeric"):
        assert not read(path).vertex_attrs


def test_a_rotated_direction_moves_the_points_and_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    f, _ = _image(path, [0, 1, 0, 0, 0, 0], origin=(10, 0, 0), direction=rot)
    f.close()
    poly = read(path)
    assert poly.vertices.tolist() == [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0]]
    np.testing.assert_array_equal(poly.global_attrs["vtkhdf_direction"], rot)
    # The points carry the rotation; the matrix is not field data of the grid.
    out = tmp_path / "back.vtkhdf"
    write(poly, out)
    with h5py.File(out) as f:
        assert "FieldData" not in f["VTKHDF"]
    assert read(out).vertices.tolist() == poly.vertices.tolist()


def test_an_image_array_of_the_wrong_shape_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1])
    root.create_dataset("PointData/p", data=np.zeros((3, 2, 2)))
    f.close()
    with pytest.warns(UserWarning, match="shaped \\(3, 2, 2\\) where the extent asks"):
        assert not read(path).vertex_attrs


def test_an_image_needs_its_extent_and_a_sane_one(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, _ = _root(path, "ImageData")
    f.close()
    with pytest.raises(CodecError, match="needs a 'WholeExtent'"):
        read(path)
    f, _ = _image(path, [2, 0, 0, 1, 0, 1])
    f.close()
    with pytest.raises(CodecError, match="runs backwards"):
        read(path)
    f, root = _image(path, [0, 1, 0, 1, 0, 1])
    root.attrs["Origin"] = np.array([1.0, 2.0])
    f.close()
    with pytest.raises(CodecError, match="'Origin' should be 3 numbers"):
        read(path)


def test_an_extent_that_is_not_whole_numbers_is_refused_not_truncated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 1, 0, 1, 0, 1])
    root.attrs["WholeExtent"] = np.array([0, 1.5, 0, 1, 0, 1])
    f.close()
    with pytest.raises(CodecError, match="six whole numbers, and holds"):
        read(path)
    f, root = _image(path, [0, 1, 0, 1, 0, 1])
    root.attrs["WholeExtent"] = np.array([0, np.inf, 0, 1, 0, 1])
    f.close()
    with pytest.raises(CodecError, match="six whole numbers, and holds"):
        read(path)


def test_an_extent_of_one_point_drops_its_cell_arrays_by_name(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 0, 0, 0, 0, 0])
    root.create_dataset("PointData/p", data=np.ones((1, 1, 1)))
    root.create_dataset("CellData/c", data=np.ones((1, 1, 1)))
    f.close()
    with pytest.warns(
        UserWarning, match="CellData array 'c' is shaped \\(1, 1, 1\\) and"
    ):
        poly = read(path)
    assert poly.vertices.shape == (1, 3)
    assert len(poly.element_types) == 0
    assert poly.vertex_attrs["p"].tolist() == [1.0]
    assert "c" not in poly.element_attrs


def test_an_image_declaring_a_grid_no_file_can_hold_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, _ = _image(path, [0, 2**40, 0, 2**20, 0, 1])
    f.close()
    with pytest.raises(ValidationError):
        read(path)


# ---------------------------------------------------------------------------
# Reading: time steps
# ---------------------------------------------------------------------------


def _temporal_ug(path: Path, times=(0.0, 0.5, 1.0)):
    """A static tetra whose point array and points change per step, laid out
    the way VTK's writer lays out a series: one part per step, points
    appended, cells appended, offsets counting the parts before."""
    n = len(times)
    f, root = _root(path, "UnstructuredGrid")
    root.create_dataset("NumberOfPoints", data=np.full(n, 4, dtype=np.int64))
    root.create_dataset("NumberOfCells", data=np.full(n, 1, dtype=np.int64))
    root.create_dataset("NumberOfConnectivityIds", data=np.full(n, 4, dtype=np.int64))
    root.create_dataset(
        "Points", data=np.vstack([_TET + [t, 0, 0] for t in times]).astype(np.float32)
    )
    root.create_dataset("Types", data=np.full(n, 10, dtype=np.uint8))
    root.create_dataset("Connectivity", data=np.tile([0, 1, 2, 3], n).astype(np.int64))
    root.create_dataset("Offsets", data=np.tile([0, 4], n).astype(np.int64))
    root.create_dataset("PointData/u", data=np.repeat(times, 4).astype(np.float64))
    root.create_dataset("CellData/c", data=10 * np.asarray(times))
    root.create_dataset("FieldData/pair", data=np.repeat(times, 2))
    steps = _steps(
        root,
        times,
        PartOffsets=np.arange(n),
        PointOffsets=4 * np.arange(n),
        CellOffsets=np.arange(n),
        ConnectivityIdOffsets=4 * np.arange(n),
    )
    steps.create_dataset(
        "PointDataOffsets/u", data=4 * np.arange(n + 1, dtype=np.int64)
    )
    steps.create_dataset("CellDataOffsets/c", data=np.arange(n + 1, dtype=np.int64))
    steps.create_dataset(
        "FieldDataOffsets/pair", data=2 * np.arange(n + 1, dtype=np.int64)
    )
    steps.create_dataset(
        "FieldDataSizes/pair", data=np.tile([1, 2], (n, 1)).astype(np.int64)
    )
    return f, root, steps


def test_a_step_that_is_not_a_whole_number_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, _ = _temporal_ug(path)
    f.close()
    with pytest.raises(CodecError, match="step='x' is not a whole number"):
        read(path, step="x")
    with pytest.raises(CodecError, match="step=1.5 is not a whole number"):
        read(path, step=1.5)
    assert read(path, step=np.int64(1)).global_attrs["time"] == 0.5


def test_a_series_reads_at_the_first_step_or_the_one_asked_for(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, _ = _temporal_ug(path)
    f.close()
    first = read(path)
    assert first.global_attrs["time"] == 0.0
    assert first.vertex_attrs["u"].tolist() == [0.0] * 4
    last = read(path, step=-1)
    assert last.global_attrs["time"] == 1.0
    assert last.vertices[:, 0].tolist() == [1.0, 2.0, 1.0, 1.0]
    assert last.vertex_attrs["u"].tolist() == [1.0] * 4
    assert last.element_attrs["c"].tolist() == [10.0]
    assert last.global_attrs["pair"].tolist() == [1.0, 1.0]
    assert last.offsets.tolist() == [0, 4]
    assert read(path, step=1).element_attrs["c"].tolist() == [5.0]


def test_read_time_series_hands_back_every_step(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, _ = _temporal_ug(path)
    f.close()
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 0.5, 1.0]
    assert [m.vertex_attrs["u"][0] for m in meshes] == [0.0, 0.5, 1.0]
    assert len(read_time_series(path)) == 3


def test_a_file_without_steps_is_one_step_and_refuses_step(tmp_path: Path) -> None:
    path = tmp_path / "m.vtkhdf"
    f, _ = _ug(path)
    f.close()
    assert len(read_time_series(path)) == 1
    with pytest.raises(CodecError, match="holds no Steps group"):
        read(path, step=0)


def test_a_steps_group_counting_no_step_says_so(tmp_path: Path) -> None:
    path = tmp_path / "z.vtkhdf"
    f, root = _ug(path)
    steps = _steps(root, [])
    steps.attrs["NSteps"] = np.int64(0)
    f.close()
    assert "time" not in read(path).global_attrs
    with pytest.raises(CodecError, match="the file holds 0 steps"):
        read(path, step=0)


def test_a_step_out_of_range_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, _ = _temporal_ug(path)
    f.close()
    with pytest.raises(CodecError, match="step=3 is out of range"):
        read(path, step=3)
    with pytest.raises(CodecError, match="step=-4 is out of range"):
        read(path, step=-4)


def test_an_array_with_no_offsets_table_is_read_from_zero(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, root, _ = _temporal_ug(path)
    root.create_dataset("PointData/static", data=np.arange(4.0))
    f.close()
    assert read(path, step=2).vertex_attrs["static"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_field_array_without_sizes_is_taken_whole_from_its_offset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.vtkhdf"
    f, root, steps = _temporal_ug(path)
    del steps["FieldDataSizes/pair"]
    f.close()
    assert read(path, step=2).global_attrs["pair"].tolist() == [1.0, 1.0]


def test_a_negative_or_missing_field_size_is_refused_not_read_from_the_end(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.vtkhdf"
    f, root, steps = _temporal_ug(path)
    del steps["FieldDataSizes/pair"]
    steps.create_dataset(
        "FieldDataSizes/pair", data=np.array([[1, 2], [1, -1], [1, 2]])
    )
    f.close()
    assert read(path, step=0).global_attrs["pair"].tolist() == [0.0, 0.0]
    with pytest.raises(CodecError, match="asks for -1 from 2"):
        read(path, step=1)
    f, root, steps = _temporal_ug(path)
    del steps["FieldDataSizes/pair"]
    steps.create_dataset("FieldDataSizes/pair", data=np.zeros((3, 0), dtype=np.int64))
    f.close()
    with pytest.raises(CodecError, match="FieldDataSizes of 'pair' hold no entry"):
        read(path, step=1)


def test_nsteps_may_be_a_one_element_array_and_must_be_a_whole_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, steps = _temporal_ug(path)
    steps.attrs["NSteps"] = np.array([3], dtype=np.int64)
    f.close()
    assert len(read_time_series(path)) == 3
    with h5py.File(path, "r+") as f:
        f["VTKHDF/Steps"].attrs["NSteps"] = np.array([1, 2], dtype=np.int64)
    with pytest.raises(CodecError, match=r"NSteps should be one whole number"):
        read(path)
    with h5py.File(path, "r+") as f:
        f["VTKHDF/Steps"].attrs["NSteps"] = 2.5
    with pytest.raises(CodecError, match=r"NSteps should be one whole number"):
        read(path)


def test_steps_declaring_more_steps_than_values_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, steps = _temporal_ug(path)
    steps.attrs["NSteps"] = np.int64(5)
    f.close()
    with pytest.raises(CodecError, match="declares 5 steps and holds 3"):
        read(path)


def test_a_scalar_or_text_steps_values_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, steps = _temporal_ug(path)
    del steps["Values"]
    steps.create_dataset("Values", data=np.float64(0.5))
    f.close()
    with pytest.raises(CodecError, match=r"Steps/Values' should be one number per"):
        read(path)
    with h5py.File(path, "r+") as f:
        del f["VTKHDF/Steps/Values"]
        f["VTKHDF/Steps"].create_dataset("Values", data=[b"0", b"1", b"2"])
    with pytest.raises(CodecError, match=r"Steps/Values' should be one number per"):
        read_time_series(path)


def test_a_step_naming_parts_the_file_lacks_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    f, _, steps = _temporal_ug(path)
    steps["NumberOfParts"][...] = 2
    f.close()
    with pytest.raises(CodecError, match="names parts 2 to 4 and the file holds 3"):
        read(path, step=2)


def test_a_temporal_polydata_keeps_one_offset_per_section(tmp_path: Path) -> None:
    path = tmp_path / "p.vtkhdf"
    f, root = _polydata(path, {"Polygons": [[0, 1, 2], [0, 1, 2, 3]]})
    root.create_dataset("CellData/c", data=np.array([1.0, 2.0, 3.0, 4.0]))
    steps = _steps(
        root,
        [0.0, 1.0],
        CellOffsets=np.zeros((2, 4)),
        ConnectivityIdOffsets=np.zeros((2, 4)),
    )
    steps.create_dataset("CellDataOffsets/c", data=np.array([0, 2], dtype=np.int64))
    f.close()
    assert read(path, step=1).element_attrs["c"].tolist() == [3.0, 4.0]
    steps_wrong = tmp_path / "w.vtkhdf"
    f, root = _polydata(steps_wrong, {"Polygons": [[0, 1, 2]]})
    _steps(
        root,
        [0.0],
        CellOffsets=np.zeros((1, 3)),
        ConnectivityIdOffsets=np.zeros((1, 4)),
    )
    f.close()
    with pytest.raises(
        CodecError, match="holds 3 value\\(s\\) per step where 4 belong"
    ):
        read(steps_wrong)


def test_a_temporal_image_keeps_its_steps_along_a_leading_axis(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1])
    root.create_dataset(
        "PointData/p",
        data=np.stack([np.full((2, 2, 3), t) for t in (10.0, 20.0, 30.0)]),
    )
    steps = _steps(root, [0.0, 1.0, 2.0])
    steps.create_dataset("PointDataOffsets/p", data=np.arange(3, dtype=np.int64))
    f.close()
    assert read(path, step=2).vertex_attrs["p"].tolist() == [30.0] * 12
    assert read(path).global_attrs["time"] == 0.0


def test_a_temporal_image_array_kept_once_reads_whole_at_every_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1])
    root.create_dataset("PointData/p", data=np.zeros((3, 2, 2, 3)))
    root.create_dataset("PointData/static", data=np.arange(12.0).reshape(2, 2, 3))
    root.create_dataset("CellData/c", data=np.arange(2.0).reshape(1, 1, 2))
    steps = _steps(root, [0.0, 1.0, 2.0])
    steps.create_dataset("PointDataOffsets/p", data=np.arange(3, dtype=np.int64))
    f.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(path, step=2)
    assert poly.vertex_attrs["static"].tolist() == list(range(12))
    assert poly.element_attrs["c"].tolist() == [0.0, 1.0]


def test_a_temporal_image_array_without_its_step_axis_names_it(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1])
    root.create_dataset("PointData/p", data=np.zeros((2, 2, 3)))
    steps = _steps(root, [0.0, 1.0])
    steps.create_dataset("PointDataOffsets/p", data=np.arange(2, dtype=np.int64))
    f.close()
    with pytest.warns(UserWarning, match=r"a step axis before \(2, 2, 3\)"):
        assert not read(path, step=1).vertex_attrs


# ---------------------------------------------------------------------------
# Reading: refusals
# ---------------------------------------------------------------------------


def test_a_file_that_is_not_hdf5_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    path.write_bytes(b"not hdf5 at all")
    with pytest.raises(CodecError, match="not an HDF5 file"):
        read(path)


def test_a_file_without_the_root_group_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    with h5py.File(path, "w") as f:
        f.create_group("Other")
    with pytest.raises(CodecError, match="holds no '/VTKHDF' group"):
        read(path)


def test_the_root_needs_a_version_this_reader_knows(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    with h5py.File(path, "w") as f:
        f.create_group("VTKHDF").attrs["Type"] = np.bytes_("UnstructuredGrid")
    with pytest.raises(CodecError, match="carries no 'Version'"):
        read(path)
    f, _ = _ug(path)
    f["VTKHDF"].attrs["Version"] = np.array([3, 0], dtype=np.int64)
    f.close()
    with pytest.raises(CodecError, match="version \\[3, 0\\] is not one"):
        read(path)


def test_version_one_reads(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, _ = _ug(path)
    f["VTKHDF"].attrs["Version"] = np.array([1, 0], dtype=np.int64)
    f.close()
    assert read(path).element_types.tolist() == [10, 5]


def test_the_root_needs_a_type_polyxios_reads(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, _ = _ug(path)
    del f["VTKHDF"].attrs["Type"]
    f.close()
    with pytest.raises(CodecError, match="carries no 'Type'"):
        read(path)
    f, _ = _root(path, "MultiBlockDataSet")
    f.close()
    with pytest.raises(
        UnsupportedFormatError, match="type 'MultiBlockDataSet' is not read"
    ):
        read(path)


def test_a_count_the_points_do_not_hold_is_refused_by_number(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    root["NumberOfPoints"][...] = 10
    f.close()
    with pytest.raises(CodecError, match="holds 4 points, and the file asks for 10"):
        read(path)


def test_a_count_no_file_can_hold_is_refused_before_anything_is_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    root["NumberOfPoints"][...] = 2**62
    f.close()
    with pytest.raises((CodecError, ValidationError), match="4611686018427387904"):
        read(path)


def test_negative_and_non_integer_counts_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    root["NumberOfCells"][...] = -1
    f.close()
    with pytest.raises(CodecError, match="negative count"):
        read(path)
    f, root = _ug(path)
    del root["NumberOfCells"]
    root.create_dataset("NumberOfCells", data=np.array([2.0]))
    f.close()
    with pytest.raises(CodecError, match="holds no whole numbers"):
        read(path)


def test_offsets_running_backwards_or_past_the_ids_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    root["Offsets"][...] = [0, 5, 3]
    f.close()
    with pytest.raises(CodecError, match="run backwards"):
        read(path)
    f, root = _ug(path)
    root["Offsets"][...] = [0, 4, 9]
    f.close()
    with pytest.raises(
        CodecError, match="reach 9 connectivity ids and the part holds 7"
    ):
        read(path)


def test_a_connectivity_id_past_the_points_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    root["Connectivity"][...] = [0, 1, 2, 9, 0, 1, 2]
    f.close()
    with pytest.raises(CodecError, match="names point 9 and holds 4 points"):
        read(path)


def test_points_that_are_not_coordinates_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    del root["Points"]
    root.create_dataset("Points", data=np.zeros((4, 2)))
    f.close()
    with pytest.raises(CodecError, match="not \\(n, 3\\) coordinates"):
        read(path)


def test_a_group_where_a_dataset_belongs_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, root = _ug(path)
    del root["Types"]
    root.create_group("Types")
    f.close()
    with pytest.raises(CodecError, match="is a group, not a dataset"):
        read(path)
    f, root = _ug(path)
    del root["NumberOfPoints"]
    f.close()
    with pytest.raises(CodecError, match="holds no 'NumberOfPoints'"):
        read(path)


def test_lazy_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, _ = _ug(path)
    f.close()
    with pytest.raises(LazyReadError, match="decoded whole"):
        read(path, lazy=True)


def test_unknown_options_are_warned_about(tmp_path: Path) -> None:
    path = tmp_path / "x.vtkhdf"
    f, _ = _ug(path)
    f.close()
    with pytest.warns(UserWarning, match="unrecognized options"):
        read(path, colour="red")
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(read(path), tmp_path / "y.vtkhdf", colour="red")


def test_without_h5py_the_read_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "x.vtkhdf"
    f, _ = _ug(path)
    f.close()
    monkeypatch.setattr(_hdf5, "_h5py", lambda: (TripWire("no h5py"), False))
    with pytest.raises(
        UnsupportedFormatError, match='pip install "polyxios\\[hdf5\\]"'
    ):
        read(path)
    with pytest.raises(
        UnsupportedFormatError, match='pip install "polyxios\\[hdf5\\]"'
    ):
        write(make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))]), path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _mesh(**kw):
    return make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        **kw,
    )


def test_the_written_file_is_vtk_s_layout(tmp_path: Path) -> None:
    path = tmp_path / "w.vtkhdf"
    write(
        _mesh(
            vertex_attrs={"u": np.arange(4.0)},
            element_attrs={"c": np.array([1, 2], dtype=np.int32)},
            vertex_tags={"top": np.array([1, 2])},
            global_attrs={"gnum": 42, "case": "run 3", "pair": [1.0, 2.0]},
        ),
        path,
    )
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        assert root.attrs["Version"].tolist() == [2, 0]
        assert root.attrs["Type"] == b"UnstructuredGrid"
        assert root.attrs.get_id("Type").dtype == np.dtype("S16")
        assert root["NumberOfPoints"][()].tolist() == [4]
        assert root["NumberOfCells"][()].tolist() == [2]
        assert root["NumberOfConnectivityIds"][()].tolist() == [7]
        assert root["Points"].dtype == np.float64 and root["Points"].shape == (4, 3)
        assert root["Types"][()].tolist() == [10, 5] and root["Types"].dtype == np.uint8
        assert root["Offsets"][()].tolist() == [0, 4, 7]
        assert root["Connectivity"][()].tolist() == [0, 1, 2, 3, 0, 1, 2]
        assert root["PointData/u"][()].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert root["PointData/polyxios_tag_top"][()].tolist() == [0, 1, 1, 0]
        assert root["CellData/c"].dtype == np.int32
        assert root["FieldData/gnum"][()].tolist() == [42]
        assert root["FieldData/pair"][()].tolist() == [1.0, 2.0]
        assert root["FieldData/case"][()].tolist() == [b"run 3"]
        assert "Steps" not in root
        assert root["Points"].chunks is None
    back = read(path)
    assert back.vertex_tags["top"].tolist() == [1, 2]
    assert back.global_attrs["gnum"] == 42
    assert back.global_attrs["case"] == "run 3"


def test_polydata_is_written_by_section_with_its_cell_data_reordered(
    tmp_path: Path,
) -> None:
    path = tmp_path / "p.vtkhdf"
    poly = make_polydata(
        _SQUARE,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("line", np.array([[0, 4]])),
            ("vertex", np.array([[3]])),
            ("quad", np.array([[0, 1, 2, 3]])),
        ],
        element_attrs={"c": np.array([1.0, 2.0, 3.0, 4.0])},
        element_tags={"a": np.array([0, 2])},
    )
    write(poly, path, polydata=True)
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"PolyData"
        assert root["Vertices/Connectivity"][()].tolist() == [3]
        assert root["Lines/Offsets"][()].tolist() == [0, 2]
        assert root["Polygons/Connectivity"][()].tolist() == [0, 1, 4, 0, 1, 2, 3]
        assert root["Polygons/NumberOfCells"][()].tolist() == [2]
        assert root["Strips/NumberOfCells"][()].tolist() == [0]
        assert root["CellData/c"][()].tolist() == [3.0, 2.0, 1.0, 4.0]
        assert root["CellData/polyxios_tag_a"][()].tolist() == [1, 0, 1, 0]
    back = read(path)
    assert back.element_types.tolist() == [1, 3, 5, 9]
    assert back.element_tags["a"].tolist() == [0, 2]
    assert back.element_attrs["c"].tolist() == [3.0, 2.0, 1.0, 4.0]


def test_a_volume_cell_refuses_polydata(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="mesh holds \\['tetra'\\]"):
        write(_mesh(), tmp_path / "p.vtkhdf", polydata=True)


def test_the_type_a_mesh_was_read_from_is_the_one_written_back(tmp_path: Path) -> None:
    path = tmp_path / "p.vtkhdf"
    f, _ = _polydata(path, {"Polygons": [[0, 1, 2]]})
    f.close()
    out = tmp_path / "q.vtkhdf"
    write(read(path), out)
    with h5py.File(out) as f:
        assert f["VTKHDF"].attrs["Type"] == b"PolyData"
    write(read(path), out, polydata=False)
    with h5py.File(out) as f:
        assert f["VTKHDF"].attrs["Type"] == b"UnstructuredGrid"


def test_a_time_writes_a_steps_group_of_one(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    write(_mesh(), path, time=0.5)
    with h5py.File(path) as f:
        steps = f["VTKHDF/Steps"]
        assert steps.attrs["NSteps"] == 1
        assert steps["Values"][()].tolist() == [0.5]
        assert steps["CellOffsets"].shape == (1,)
        assert "FieldData" not in f["VTKHDF"]
    assert read(path).global_attrs["time"] == 0.5
    write(_mesh(global_attrs={"time": 2.0}), path)
    assert read(path).global_attrs["time"] == 2.0
    write(_mesh(global_attrs={"time": "noon"}), path)
    assert read(path).global_attrs["time"] == "noon"
    with pytest.raises(CodecError, match="time 'noon', which is not a number"):
        write(_mesh(), path, time="noon")


def test_a_text_time_is_dropped_by_name_when_a_step_time_is_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.vtkhdf"
    noon = _mesh(global_attrs={"time": "noon", "case": "run 3"})
    with pytest.warns(UserWarning, match="global 'time' holds 'noon', and the step"):
        write(noon, path, time=0.5)
    with h5py.File(path) as f:
        assert "time" not in f["VTKHDF/FieldData"]
        assert "case" in f["VTKHDF/FieldData"]
    assert read(path).global_attrs["time"] == 0.5
    with pytest.warns(UserWarning, match="global 'time' holds 'noon'"):
        write_time_series([(0.0, noon), (1.0, noon)], path)
    with h5py.File(path) as f:
        assert "time" not in f["VTKHDF/FieldData"]
    assert [m.global_attrs["time"] for m in read_time_series(path)] == [0.0, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(_mesh(global_attrs={"time": 2.0}), path, time=0.5)
    assert read(path).global_attrs["time"] == 0.5


def test_one_step_is_laid_out_whole_not_chunked_to_grow(tmp_path: Path) -> None:
    path = tmp_path / "t.vtkhdf"
    write(_mesh(vertex_attrs={"u": np.arange(4.0)}), path, time=0.5)
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        for key in ("Points", "NumberOfPoints", "PointData/u", "Connectivity"):
            assert root[key].chunks is None, key
            assert root[key].maxshape == root[key].shape, key
        assert root["Steps/PointDataOffsets/u"][()].tolist() == [0]
    assert read(path).vertex_attrs["u"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_series_step_whose_time_is_not_a_number_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    with pytest.raises(CodecError, match="step 0 is at time 'dawn', which is not"):
        write_time_series([("dawn", _mesh())], path)
    with pytest.raises(CodecError, match="step 1 is at time None, which is not"):
        write_time_series([(0.0, _mesh()), (None, _mesh())], path)


def test_an_array_named_with_a_slash_is_renamed_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "n.vtkhdf"
    mesh = _mesh(
        vertex_attrs={"a/b": np.arange(4.0), "a_b": np.arange(4.0) + 10},
        element_attrs={"": np.array([1.0, 2.0])},
        element_tags={"x/y": np.array([0])},
        global_attrs={"run/1": 3, "run/2": "two"},
    )
    with pytest.warns(UserWarning, match="named as no HDF5 dataset can be") as record:
        write(mesh, path)
    assert sorted(str(w.message).split(" array(s) ")[0] for w in record) == [
        ".vtkhdf write: CellData",
        ".vtkhdf write: FieldData",
        ".vtkhdf write: PointData",
    ]
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        assert set(root["PointData"]) == {"a_b", "a_b_2"}
        assert set(root["CellData"]) == {"array", "polyxios_tag_x_y"}
        assert set(root["FieldData"]) == {"run_1", "run_2"}
    back = read(path)
    assert back.vertex_attrs["a_b_2"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert back.vertex_attrs["a_b"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert back.element_attrs["array"].tolist() == [1.0, 2.0]
    assert back.element_tags["x_y"].tolist() == [0]
    assert back.global_attrs["run_1"] == 3 and back.global_attrs["run_2"] == "two"
    path2 = tmp_path / "s.vtkhdf"
    with pytest.warns(UserWarning, match="named as no HDF5 dataset can be"):
        write_time_series([(0.0, mesh), (1.0, mesh)], path2)
    with h5py.File(path2) as f:
        assert set(f["VTKHDF/Steps/PointDataOffsets"]) == {"a_b", "a_b_2"}
        assert set(f["VTKHDF/Steps/FieldDataOffsets"]) == {"run_1", "run_2"}


def _step(t: float, *, moved: bool = False):
    verts = _TET + ([t, 0, 0] if moved else [0, 0, 0])
    return make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"u": np.full(4, t)},
        element_attrs={"c": np.full(2, 10 * t)},
        global_attrs={"pair": [t, t], "gnum": 42},
    )


def test_a_series_writes_the_cells_once_and_appends_the_arrays(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    write_time_series(((0.5 * i, _step(0.5 * i)) for i in range(3)), path)
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        assert root["Points"].shape == (4, 3)
        assert root["Types"].shape == (2,)
        assert root["NumberOfPoints"][()].tolist() == [4]
        assert root["PointData/u"][()].tolist() == [0.0] * 4 + [0.5] * 4 + [1.0] * 4
        steps = root["Steps"]
        assert steps.attrs["NSteps"] == 3
        assert steps["Values"][()].tolist() == [0.0, 0.5, 1.0]
        assert steps["PointOffsets"][()].tolist() == [0, 0, 0]
        assert steps["PartOffsets"][()].tolist() == [0, 0, 0]
        assert steps["PointDataOffsets/u"][()].tolist() == [0, 4, 8]
        assert steps["CellDataOffsets/c"][()].tolist() == [0, 2, 4]
        assert steps["FieldDataOffsets/pair"][()].tolist() == [0, 2, 4]
        assert steps["FieldDataSizes/pair"][()].tolist() == [[1, 2], [1, 2], [1, 2]]
        assert steps["FieldDataSizes/gnum"][()].tolist() == [[1, 1]] * 3
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 0.5, 1.0]
    assert [m.element_attrs["c"][0] for m in meshes] == [0.0, 5.0, 10.0]
    assert [m.global_attrs["pair"].tolist() for m in meshes] == [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ]
    assert meshes[2].global_attrs["gnum"] == 42


def test_a_step_whose_points_moved_writes_its_own(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    write_time_series([(0.0, _step(0.0)), (1.0, _step(1.0, moved=True))], path)
    with h5py.File(path) as f:
        assert f["VTKHDF/Points"].shape == (8, 3)
        assert f["VTKHDF/Steps/PointOffsets"][()].tolist() == [0, 4]
    assert read(path, step=1).vertices[1].tolist() == [2.0, 0.0, 0.0]
    assert read(path, step=0).vertices[1].tolist() == [1.0, 0.0, 0.0]


def test_a_step_equal_to_the_last_written_shares_its_points(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    write_time_series(
        [
            (0.0, _step(0.0)),
            (1.0, _step(1.0, moved=True)),
            (2.0, _step(1.0, moved=True)),
            (3.0, _step(0.0)),
        ],
        path,
    )
    with h5py.File(path) as f:
        assert f["VTKHDF/Points"].shape == (12, 3)
        assert f["VTKHDF/Steps/PointOffsets"][()].tolist() == [0, 4, 4, 8]
    assert read(path, step=2).vertices[1].tolist() == [2.0, 0.0, 0.0]
    assert read(path, step=3).vertices[1].tolist() == [1.0, 0.0, 0.0]


def test_a_generator_moving_one_mesh_in_place_writes_each_step_as_it_stood(
    tmp_path: Path,
) -> None:
    path = tmp_path / "g.vtkhdf"
    mesh = _step(0.0)

    def solver():
        for k in range(3):
            if k:
                mesh.vertices[:, 0] += 1.0
                mesh.vertex_attrs["u"][:] = float(k)
            yield float(k), mesh

    write_time_series(solver(), path)
    with h5py.File(path) as f:
        assert f["VTKHDF/Points"].shape == (12, 3)
        assert f["VTKHDF/Steps/PointOffsets"][()].tolist() == [0, 4, 8]
    assert [read(path, step=k).vertices[1, 0] for k in range(3)] == [1.0, 2.0, 3.0]
    assert [read(path, step=k).vertex_attrs["u"][0] for k in range(3)] == [0, 1, 2]

    def rewired():
        yield 0.0, mesh
        mesh.connectivity[0], mesh.connectivity[1] = 1, 0
        yield 1.0, mesh

    with pytest.raises(CodecError, match="step 1 holds different elements"):
        write_time_series(rewired(), path)


def test_a_nan_vertex_still_shares_its_points_between_equal_steps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "n.vtkhdf"
    hole = _step(0.0)
    hole.vertices[3, 2] = np.nan
    write_time_series([(0.0, hole), (1.0, hole), (2.0, hole)], path)
    with h5py.File(path) as f:
        assert f["VTKHDF/Points"].shape == (4, 3)
        assert f["VTKHDF/Steps/PointOffsets"][()].tolist() == [0, 0, 0]


def test_a_series_whose_elements_change_is_refused(tmp_path: Path) -> None:
    other = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(CodecError, match="step 1 holds different elements"):
        write_time_series([(0.0, _step(0.0)), (1.0, other)], tmp_path / "s.vtkhdf")
    fewer = make_polydata(
        _TET[:3],
        [("triangle", np.array([[0, 1, 2]]))],
    )
    same_cells = make_polydata(
        np.vstack([_TET, [[2, 2, 2]]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
    )
    with pytest.raises(CodecError, match="holds 5 vertices and step 0 holds 4"):
        write_time_series([(0.0, _mesh()), (1.0, same_cells)], tmp_path / "s.vtkhdf")
    wide = dataclasses.replace(_mesh(), vertices=np.hstack([_TET, np.zeros((4, 1))]))
    with pytest.raises(CodecError, match=r"shaped \(4, 4\), not \(n, 3\)"):
        write_time_series([(0.0, _mesh()), (1.0, wide)], tmp_path / "s.vtkhdf")
    del fewer


def test_a_series_whose_arrays_change_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    missing = _step(1.0)
    missing.vertex_attrs.pop("u")
    with pytest.raises(CodecError, match="lacks PointData array\\(s\\) \\['u'\\]"):
        write_time_series([(0.0, _step(0.0)), (1.0, missing)], path)
    extra = _step(1.0)
    extra.element_attrs["d"] = np.zeros(2)
    with pytest.raises(CodecError, match="adds CellData array\\(s\\) \\['d'\\]"):
        write_time_series([(0.0, _step(0.0)), (1.0, extra)], path)
    reshaped = _step(1.0)
    reshaped.vertex_attrs["u"] = np.zeros((4, 2))
    with pytest.raises(CodecError, match="changes shape or dtype"):
        write_time_series([(0.0, _step(0.0)), (1.0, reshaped)], path)
    # The names are checked before any of the step's rows are appended, so
    # a step both renaming and reshaping is refused by name.
    both = _step(1.0)
    both.vertex_attrs["u"] = np.zeros((4, 2))
    both.vertex_attrs["w"] = np.zeros(4)
    with pytest.raises(CodecError, match="adds PointData array\\(s\\) \\['w'\\]"):
        write_time_series([(0.0, _step(0.0)), (1.0, both)], path)
    renamed_text = _step(1.0)
    renamed_text.global_attrs["case"] = "run 3"
    with pytest.raises(CodecError, match="adds FieldData array\\(s\\) \\['case'\\]"):
        write_time_series([(0.0, _step(0.0)), (1.0, renamed_text)], path)
    # A dtype or a component count that drifts is refused by name before
    # any row of the step is appended, not by the dataset it would land in.
    narrower = _step(0.0)
    narrower.vertex_attrs["u"] = np.zeros(4, dtype=np.float32)
    with pytest.raises(
        CodecError,
        match="PointData array 'u' from \\(4,\\) float32 to \\(4,\\) float64",
    ):
        write_time_series([(0.0, narrower), (1.0, _step(1.0))], path)
    wider_field = _step(1.0)
    wider_field.global_attrs["gnum"] = np.zeros((1, 2))
    with pytest.raises(
        CodecError, match="FieldData array 'gnum' from \\(1,\\) int64 to \\(1, 2\\)"
    ):
        write_time_series([(0.0, _step(0.0)), (1.0, wider_field)], path)
    now_text = _step(1.0)
    now_text.global_attrs["gnum"] = "forty-two"
    with pytest.raises(CodecError, match="'gnum' as text where step 0 wrote numbers"):
        write_time_series([(0.0, _step(0.0)), (1.0, now_text)], path)
    was_text = _step(0.0)
    was_text.global_attrs["gnum"] = "forty-two"
    with pytest.raises(CodecError, match="'gnum' as numbers where step 0 wrote text"):
        write_time_series([(0.0, was_text), (1.0, _step(1.0))], path)
    # A field array may hold another number of tuples per step, and a text
    # another number of lines: FieldDataSizes spells each step's.
    longer = _step(1.0)
    longer.global_attrs["pair"] = [1.0, 1.0, 1.0]
    write_time_series([(0.0, _step(0.0)), (1.0, longer)], path)
    assert read(path, step=1).global_attrs["pair"].tolist() == [1.0, 1.0, 1.0]


def test_an_empty_series_is_refused_and_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    with pytest.raises(CodecError, match="at least one step"):
        write_time_series([], path)
    assert not path.exists()


def test_a_polydata_series_keeps_four_offsets_per_step(tmp_path: Path) -> None:
    path = tmp_path / "s.vtkhdf"
    surface = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
    )
    write_time_series([(0.0, surface), (1.0, surface)], path, polydata=True)
    with h5py.File(path) as f:
        assert f["VTKHDF/Steps/CellOffsets"].shape == (2, 4)
        assert f["VTKHDF/Steps/ConnectivityIdOffsets"].shape == (2, 4)
    assert read(path, step=1).element_types.tolist() == [5, 9]


def test_an_element_type_vtk_lacks_is_dropped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "w.vtkhdf"
    poly = make_polydata(
        _TET,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"c": np.array([1.0, 2.0])},
        element_tags={"a": np.array([1])},
    )
    # A code polyxios has no name for: nothing to spell it as.
    object.__setattr__(poly, "element_types", np.array([10, 200], dtype=np.uint8))
    with pytest.warns(UserWarning, match="\\['type_200'\\] have no VTK cell type"):
        write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [10]
    assert back.element_attrs["c"].tolist() == [1.0]
    assert back.element_tags["a"].tolist() == []


def test_arrays_that_are_not_numeric_are_dropped_and_odd_shapes_flattened(
    tmp_path: Path,
) -> None:
    path = tmp_path / "w.vtkhdf"
    poly = _mesh(
        vertex_attrs={
            "names": np.array(list("abcd")),
            "flag": np.array([True, False, True, False]),
            "tensor": np.arange(36.0).reshape(4, 3, 3),
        },
        global_attrs={"odd": {"a": 1}},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, path)
    messages = [str(w.message) for w in caught]
    assert any("point array(s) ['names'] are not numeric" in m for m in messages)
    assert any("odd" in m for m in messages)
    back = read(path)
    assert back.vertex_attrs["flag"].dtype == np.uint8
    assert back.vertex_attrs["tensor"].shape == (4, 9)
    assert "odd" not in back.global_attrs


def test_vertices_that_are_not_a_coordinate_block_are_refused(tmp_path: Path) -> None:
    poly = _mesh()
    object.__setattr__(poly, "vertices", np.zeros((4, 2)))
    with pytest.raises(CodecError, match="shaped \\(4, 2\\), not \\(n, 3\\)"):
        write(poly, tmp_path / "w.vtkhdf")


def test_compression_is_passed_to_the_datasets(tmp_path: Path) -> None:
    path = tmp_path / "c.vtkhdf"
    write(
        _mesh(vertex_attrs={"u": np.arange(4.0)}),
        path,
        compression="gzip",
        compression_opts=4,
    )
    with h5py.File(path) as f:
        assert f["VTKHDF/Points"].compression == "gzip"
        assert f["VTKHDF/PointData/u"].compression_opts == 4
    assert read(path).vertex_attrs["u"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_compressed_file_is_read_whatever_its_size_says(tmp_path: Path) -> None:
    # Two hundred thousand points of zeros shrink to a few kilobytes, far
    # below what the count would need uncompressed.
    path = tmp_path / "c.vtkhdf"
    n = 200_000
    poly = make_polydata(np.zeros((n, 3)), [("vertex", np.array([[0], [1]]))])
    write(poly, path, compression="gzip")
    assert path.stat().st_size * 4 < n * 24
    assert read(path).vertices.shape == (n, 3)
    write(poly, path, compression="gzip", time=0.5)
    assert read_time_series(path)[0].vertices.shape == (n, 3)


def test_text_globals_are_spelled_the_way_vtk_reads_them(tmp_path: Path) -> None:
    # VTK's reader converts variable-length ASCII and not variable-length
    # UTF-8, so the bytes are UTF-8 under the type VTK's own writer uses.
    path = tmp_path / "t.vtkhdf"
    write(_mesh(global_attrs={"label": "h\u00e9llo"}), path)
    with h5py.File(path) as f:
        node = f["VTKHDF/FieldData/label"]
        assert h5py.check_string_dtype(node.dtype).encoding == "ascii"
        assert node[()].tolist() == ["h\u00e9llo".encode()]
    assert read(path).global_attrs["label"] == "h\u00e9llo"
    write_time_series(
        [
            (0.0, _mesh(global_attrs={"label": "h\u00e9llo"})),
            (1.0, _mesh(global_attrs={"label": "world"})),
        ],
        path,
    )
    with h5py.File(path) as f:
        node = f["VTKHDF/FieldData/label"]
        assert h5py.check_string_dtype(node.dtype).encoding == "ascii"
        assert node[()].tolist() == ["h\u00e9llo".encode(), b"world"]
    assert [m.global_attrs["label"] for m in read_time_series(path)] == [
        "h\u00e9llo",
        "world",
    ]


def test_a_text_global_holding_a_nul_is_written_without_it(tmp_path: Path) -> None:
    path = tmp_path / "n.vtkhdf"
    with pytest.warns(UserWarning, match="text global 'label' holds a NUL"):
        write(_mesh(global_attrs={"label": "run\x003"}), path)
    assert read(path).global_attrs["label"] == "run3"


def test_a_buffer_and_a_gzip_name_both_work(tmp_path: Path) -> None:
    poly = _mesh(vertex_attrs={"u": np.arange(4.0)})
    buf = io.BytesIO()
    write(poly, buf)
    buf.seek(0)
    assert read(buf).vertex_attrs["u"].tolist() == [0.0, 1.0, 2.0, 3.0]
    packed = tmp_path / "m.vtkhdf.gz"
    polyxios.write(poly, packed)
    assert gzip.decompress(packed.read_bytes())[:8] == _hdf5.HDF5_MAGIC
    assert polyxios.read(packed).element_types.tolist() == [10, 5]


def test_the_codec_is_registered_and_the_helper_dispatches_to_it(
    tmp_path: Path,
) -> None:
    assert polyxios.supported_extensions().count(".vtkhdf") == 1
    path = tmp_path / "s.vtkhdf"
    helper.write_time_series([(0.0, _step(0.0)), (1.0, _step(1.0))], path)
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 1.0] and len(meshes) == 2
    assert polyxios.read(path, step=1).vertex_attrs["u"][0] == 1.0


# ---------------------------------------------------------------------------
# Interop with VTK itself, when it is installed
# ---------------------------------------------------------------------------


def _vtk():
    vtk = pytest.importorskip("vtk")
    if not hasattr(vtk, "vtkHDFWriter"):
        pytest.skip("this VTK has no vtkHDFWriter")
    return vtk


def _vtk_read(vtk, path: Path, time: float | None = None):
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    if time is None:
        reader.Update()
    else:
        reader.UpdateTimeStep(time)
    return reader.GetOutput()


def test_vtk_reads_what_polyxios_writes(tmp_path: Path) -> None:
    vtk = _vtk()
    from vtk.util.numpy_support import vtk_to_numpy

    path = tmp_path / "w.vtkhdf"
    labelled = _step(0.5)
    labelled.global_attrs["label"] = "h\u00e9llo"
    write(labelled, path)
    out = _vtk_read(vtk, path)
    assert out.GetNumberOfPoints() == 4
    assert [out.GetCellType(i) for i in range(out.GetNumberOfCells())] == [10, 5]
    assert vtk_to_numpy(out.GetPointData().GetArray("u")).tolist() == [0.5] * 4
    assert vtk_to_numpy(out.GetCellData().GetArray("c")).tolist() == [5.0, 5.0]
    assert vtk_to_numpy(out.GetFieldData().GetArray("pair")).tolist() == [0.5, 0.5]
    assert out.GetFieldData().GetAbstractArray("label").GetValue(0) == "h\u00e9llo"

    surface = make_polydata(
        _SQUARE, [("triangle", np.array([[0, 1, 4]])), ("line", np.array([[0, 4]]))]
    )
    write(surface, path, polydata=True)
    out = _vtk_read(vtk, path)
    assert out.GetClassName() == "vtkPolyData"
    assert [out.GetCellType(i) for i in range(out.GetNumberOfCells())] == [3, 5]

    later = _step(1.0, moved=True)
    later.global_attrs["label"] = "world"
    write_time_series([(0.0, labelled), (1.0, later)], path)
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    info = reader.GetOutputInformation(0)
    assert list(info.Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())) == [
        0.0,
        1.0,
    ]
    out = _vtk_read(vtk, path, 1.0)
    assert out.GetPoint(1) == (2.0, 0.0, 0.0)
    assert vtk_to_numpy(out.GetPointData().GetArray("u")).tolist() == [1.0] * 4
    assert out.GetFieldData().GetAbstractArray("label").GetValue(0) == "world"


def test_polyxios_reads_what_vtk_writes(tmp_path: Path) -> None:
    vtk = _vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    points = vtk.vtkPoints()
    for p in _SQUARE:
        points.InsertNextPoint(*p)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, [0, 1, 4])
    grid.InsertNextCell(vtk.VTK_QUAD, 4, [0, 1, 2, 3])
    scalar = numpy_to_vtk(np.arange(5.0))
    scalar.SetName("scalar")
    grid.GetPointData().AddArray(scalar)
    writer = vtk.vtkHDFWriter()
    writer.SetFileName(str(tmp_path / "vtk.vtkhdf"))
    writer.SetInputData(grid)
    writer.Write()
    poly = read(tmp_path / "vtk.vtkhdf")
    np.testing.assert_array_equal(poly.vertices, _SQUARE)
    assert poly.element_types.tolist() == [5, 9]
    assert poly.vertex_attrs["scalar"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    image = vtk.vtkImageData()
    image.SetDimensions(3, 2, 2)
    image.SetSpacing(0.5, 1, 2)
    values = numpy_to_vtk(np.arange(12.0))
    values.SetName("p")
    image.GetPointData().AddArray(values)
    writer = vtk.vtkHDFWriter()
    writer.SetFileName(str(tmp_path / "img.vtkhdf"))
    writer.SetInputData(image)
    writer.Write()
    poly = read(tmp_path / "img.vtkhdf")
    assert poly.vertices.shape == (12, 3) and poly.element_types.tolist() == [12, 12]
    assert poly.vertex_attrs["p"].tolist() == list(range(12))
    assert poly.global_attrs["vti_spacing"] == [0.5, 1.0, 2.0]


# The layouts VTK's writer stores an ImageData in, which leaves out each axis that
# holds one point, and keeps one row for a single point. Each entry: the extent,
# whether the grid is rotated, and the stored shapes of the point scalar ``p``, the
# point vector ``v`` and the cell scalar ``c`` (None when there are no cells).
_LAYOUTS = {
    "flat": ((0, 3, 0, 2, 0, 0), False, ((3, 4), (3, 4, 3), (2, 3))),
    "line": ((0, 3, 0, 0, 0, 0), False, ((4,), (4, 3), (3,))),
    "xz plane": ((0, 2, 0, 0, 0, 1), False, ((2, 3), (2, 3, 3), (1, 2))),
    "yz plane": ((0, 0, 0, 2, 0, 1), False, ((2, 3), (2, 3, 3), (1, 2))),
    "single point": ((0, 0, 0, 0, 0, 0), False, ((1,), (1, 3), None)),
    "rotated volume": ((1, 3, 2, 3, 0, 1), True, ((2, 2, 3), (2, 2, 3, 3), (1, 1, 2))),
}
_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _layout_image(extent, rotated: bool) -> ImageData:
    i0, i1, j0, j1, k0, k1 = extent
    pts = (k1 - k0 + 1, j1 - j0 + 1, i1 - i0 + 1)
    n = int(np.prod(pts))
    point = {
        "p": np.arange(n, dtype=np.float64).reshape(pts),
        "v": np.arange(3 * n, dtype=np.float64).reshape(pts + (3,)),
    }
    cell = {}
    if n > 1:
        cells = tuple(max(p - 1, 1) for p in pts)
        count = int(np.prod(cells))
        cell["c"] = np.arange(count, dtype=np.float64).reshape(cells) + 100.0
    return ImageData(
        extent=extent,
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 1.0, 2.0),
        direction=_ROTATION if rotated else None,
        point_attrs=point,
        cell_attrs=cell,
        global_attrs={"title": "x", "pair": np.array([0.25, 0.5])},
    )


def _assert_image_arrays(
    back: ImageData, image: ImageData, *, fields: bool = True
) -> None:
    for want, got in (
        (image.point_attrs, back.point_attrs),
        (image.cell_attrs, back.cell_attrs),
    ):
        assert sorted(got) == sorted(want)
        for key, arr in want.items():
            np.testing.assert_array_equal(got[key], arr, err_msg=key)
    assert back.extent == image.extent
    if fields:
        np.testing.assert_array_equal(
            back.global_attrs["pair"], image.global_attrs["pair"]
        )
    if image.direction is None:
        assert back.direction is None
    else:
        np.testing.assert_array_equal(back.direction, image.direction)


@pytest.mark.parametrize("name", list(_LAYOUTS))
def test_image_arrays_are_stored_in_vtks_own_layout(tmp_path: Path, name: str) -> None:
    """A file of any shape stores each array as VTK's writer does, with each axis that
    holds one point left out, and reads back to the full lattice."""
    extent, rotated, stored = _LAYOUTS[name]
    image = _layout_image(extent, rotated)
    path = tmp_path / "layout.vtkhdf"
    polyxios.write_image(image, path)
    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root["PointData/p"].shape == stored[0]
        assert root["PointData/v"].shape == stored[1]
        if stored[2] is None:
            assert "CellData/c" not in root
        else:
            assert root["CellData/c"].shape == stored[2]
    _assert_image_arrays(polyxios.read_image(path), image)


@pytest.mark.parametrize("name", list(_LAYOUTS))
def test_read_image_takes_the_full_lattice(tmp_path: Path, name: str) -> None:
    """A file that keeps every axis of each array - the form earlier releases read -
    reads back the same as one that leaves out the axes of one point, for points and
    cells alike."""
    extent, rotated, _ = _LAYOUTS[name]
    image = _layout_image(extent, rotated)
    path = tmp_path / "lattice.vtkhdf"
    polyxios.write_image(image, path)
    with h5py.File(path, "r+") as f:
        for group, arrays in (
            ("PointData", image.point_attrs),
            ("CellData", image.cell_attrs),
        ):
            for key, arr in arrays.items():
                data = f[f"VTKHDF/{group}/{key}"][()]
                del f[f"VTKHDF/{group}/{key}"]
                f[f"VTKHDF/{group}"].create_dataset(key, data=data.reshape(arr.shape))
    _assert_image_arrays(polyxios.read_image(path), image)


def test_read_image_reads_a_plane_built_by_hand_in_vtks_layout(tmp_path: Path) -> None:
    """An xz plane as VTK's writer lays it out, built with h5py alone: its cells span
    one layer of the plane, and are kept as cells."""
    path = tmp_path / "plane.vtkhdf"
    with h5py.File(path, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array((2, 0), dtype=np.int64)
        root.attrs["Type"] = np.bytes_("ImageData")
        root.attrs["WholeExtent"] = np.array((0, 2, 0, 0, 0, 1), dtype=np.int64)
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.ones(3)
        root.create_group("PointData")["p"] = np.arange(6.0).reshape(2, 3)
        root.create_group("CellData")["c"] = np.arange(2.0).reshape(1, 2)
    image = polyxios.read_image(path)
    np.testing.assert_array_equal(
        image.point_attrs["p"], np.arange(6.0).reshape(2, 1, 3)
    )
    np.testing.assert_array_equal(
        image.cell_attrs["c"], np.arange(2.0).reshape(1, 1, 2)
    )


@pytest.mark.parametrize("name", list(_LAYOUTS))
def test_vtk_reads_every_image_layout_polyxios_writes(
    tmp_path: Path, name: str
) -> None:
    vtk = _vtk()
    from vtk.util.numpy_support import vtk_to_numpy

    extent, rotated, _ = _LAYOUTS[name]
    image = _layout_image(extent, rotated)
    path = tmp_path / "img.vtkhdf"
    polyxios.write_image(image, path)
    out = _vtk_read(vtk, path)
    assert out.GetNumberOfPoints() == image.point_attrs["p"].size
    for key, arr in image.point_attrs.items():
        got = vtk_to_numpy(out.GetPointData().GetArray(key))
        np.testing.assert_array_equal(got.reshape(arr.shape), arr, err_msg=key)
    for key, arr in image.cell_attrs.items():
        got = vtk_to_numpy(out.GetCellData().GetArray(key))
        np.testing.assert_array_equal(got.reshape(arr.shape), arr, err_msg=key)
    pair = vtk_to_numpy(out.GetFieldData().GetArray("pair"))
    np.testing.assert_array_equal(pair, [0.25, 0.5])
    if rotated:
        m = out.GetDirectionMatrix()
        assert [[m.GetElement(i, j) for j in range(3)] for i in range(3)] == (
            _ROTATION.tolist()
        )


@pytest.mark.parametrize("name", list(_LAYOUTS))
def test_polyxios_reads_every_image_layout_vtk_writes(
    tmp_path: Path, name: str
) -> None:
    vtk = _vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    extent, rotated, _ = _LAYOUTS[name]
    want = _layout_image(extent, rotated)
    grid = vtk.vtkImageData()
    grid.SetExtent(*extent)
    grid.SetOrigin(1.0, 2.0, 3.0)
    grid.SetSpacing(0.5, 1.0, 2.0)
    if rotated:
        grid.SetDirectionMatrix(_ROTATION.ravel().tolist())
    for key, arr in want.point_attrs.items():
        values = numpy_to_vtk(np.ascontiguousarray(arr.reshape((-1,) + arr.shape[3:])))
        values.SetName(key)
        grid.GetPointData().AddArray(values)
    for key, arr in want.cell_attrs.items():
        values = numpy_to_vtk(np.ascontiguousarray(arr.reshape(-1)))
        values.SetName(key)
        grid.GetCellData().AddArray(values)
    # VTK's writer keeps no field data on an image, so none is written here.
    writer = vtk.vtkHDFWriter()
    writer.SetFileName(str(tmp_path / "vtk.vtkhdf"))
    writer.SetInputData(grid)
    writer.Write()
    _assert_image_arrays(
        polyxios.read_image(tmp_path / "vtk.vtkhdf"), want, fields=False
    )


def _vtk_temporal_source(vtk, output_type: str, times: list[float], build):
    """A pipeline source handing out ``build(output, t)`` at each of ``times``."""
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    sddp = vtk.vtkStreamingDemandDrivenPipeline

    class Source(VTKPythonAlgorithmBase):
        def __init__(self):
            VTKPythonAlgorithmBase.__init__(
                self, nInputPorts=0, nOutputPorts=1, outputType=output_type
            )

        def RequestInformation(self, request, in_info, out_info):
            info = out_info.GetInformationObject(0)
            info.Set(sddp.TIME_STEPS(), times, len(times))
            info.Set(sddp.TIME_RANGE(), [times[0], times[-1]], 2)
            if output_type == "vtkImageData":
                info.Set(sddp.WHOLE_EXTENT(), [0, 2, 0, 1, 0, 1], 6)
            return 1

        def RequestData(self, request, in_info, out_info):
            info = out_info.GetInformationObject(0)
            t = info.Get(sddp.UPDATE_TIME_STEP())
            out = info.Get(vtk.vtkDataObject.DATA_OBJECT())
            build(out, t)
            out.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), t)
            return 1

    return Source()


def _vtk_write_series(vtk, source, path: Path) -> None:
    writer = vtk.vtkHDFWriter()
    writer.SetInputConnection(source.GetOutputPort())
    writer.SetFileName(str(path))
    writer.SetWriteAllTimeSteps(True)
    writer.Write()


def test_polyxios_reads_a_partitioned_series_vtk_writes(tmp_path: Path) -> None:
    """VTK lays a partitioned series out as one Types run per step, with one
    extra Offsets row per part: the second part's offsets at step 1 begin
    at CellOffsets[1] + 1, not CellOffsets[1]."""
    vtk = _vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    def grid(t, shift, n_cells):
        points = vtk.vtkPoints()
        for p in _TET:
            points.InsertNextPoint(p[0] + shift + t, p[1], p[2])
        g = vtk.vtkUnstructuredGrid()
        g.SetPoints(points)
        g.InsertNextCell(vtk.VTK_TETRA, 4, [0, 1, 2, 3])
        if n_cells == 2:
            g.InsertNextCell(vtk.VTK_TRIANGLE, 3, [0, 1, 2])
        u = numpy_to_vtk(np.full(4, t + shift))
        u.SetName("u")
        g.GetPointData().AddArray(u)
        c = numpy_to_vtk(np.full(n_cells, 10 * t + shift))
        c.SetName("c")
        g.GetCellData().AddArray(c)
        return g

    def build(out, t):
        out.SetNumberOfPartitions(2)
        out.SetPartition(0, grid(t, 0.0, 1))
        out.SetPartition(1, grid(t, 10.0, 2))

    path = tmp_path / "parts.vtkhdf"
    source = _vtk_temporal_source(vtk, "vtkPartitionedDataSet", [0.0, 1.0], build)
    _vtk_write_series(vtk, source, path)
    with h5py.File(path) as f:
        assert f["VTKHDF/Steps/NumberOfParts"][()].tolist() == [2, 2]
        assert f["VTKHDF/Steps/CellOffsets"][()].tolist() == [0, 3]
        assert f["VTKHDF/Offsets"].shape == (10,)
    for step in (0, 1):
        poly = read(path, step=step)
        assert poly.element_types.tolist() == [10, 10, 5]
        assert poly.offsets.tolist() == [0, 4, 8, 11]
        assert poly.connectivity.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6]
        assert poly.vertices[:, 0].tolist() == [
            step,
            step + 1,
            step,
            step,
            step + 10,
            step + 11,
            step + 10,
            step + 10,
        ]
        assert poly.vertex_attrs["u"].tolist() == [step] * 4 + [step + 10] * 4
        assert poly.element_attrs["c"].tolist() == [10 * step] + [10 * step + 10] * 2
        assert poly.element_tags["part_0"].tolist() == [0]
        assert poly.element_tags["part_1"].tolist() == [1, 2]
        assert poly.global_attrs["time"] == step


def test_polyxios_reads_a_temporal_image_vtk_writes(tmp_path: Path) -> None:
    vtk = _vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    def build(out, t):
        out.SetExtent(0, 2, 0, 1, 0, 1)
        p = numpy_to_vtk(np.arange(12.0) + 100 * t)
        p.SetName("p")
        out.GetPointData().AddArray(p)
        c = numpy_to_vtk(np.full(2, t))
        c.SetName("c")
        out.GetCellData().AddArray(c)

    path = tmp_path / "timg.vtkhdf"
    source = _vtk_temporal_source(vtk, "vtkImageData", [0.0, 1.0, 2.0], build)
    _vtk_write_series(vtk, source, path)
    with h5py.File(path) as f:
        assert f["VTKHDF/PointData/p"].shape == (3, 2, 2, 3)
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 1.0, 2.0]
    for t, poly in zip(times, meshes, strict=True):
        assert poly.vertices.shape == (12, 3)
        assert poly.element_types.tolist() == [12, 12]
        assert poly.vertex_attrs["p"].tolist() == (np.arange(12.0) + 100 * t).tolist()
        assert poly.element_attrs["c"].tolist() == [t, t]
    assert read(path, step=-1).vertex_attrs["p"][0] == 200.0


def test_the_module_lists_its_public_names() -> None:
    assert set(_vtkhdf.__all__) == {
        "EXTENSION",
        "LABEL",
        "read",
        "read_image",
        "read_time_series",
        "write",
        "write_image",
        "write_time_series",
    }


def test_read_image_keeps_the_grid_the_mesh_read_expands(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1], origin=(1, 2, 3), spacing=(0.5, 1, 2))
    root.create_dataset("PointData/p", data=np.arange(12.0).reshape(2, 2, 3))
    root.create_dataset("PointData/v", data=np.arange(36.0).reshape(2, 2, 3, 3))
    root.create_dataset("CellData/c", data=np.arange(2.0).reshape(1, 1, 2))
    root.create_dataset("FieldData/gnum", data=np.array([42.0]))
    f.close()
    image = polyxios.read_image(path)
    assert image.extent == (0, 2, 0, 1, 0, 1)
    assert image.origin == (1.0, 2.0, 3.0)
    assert image.spacing == (0.5, 1.0, 2.0)
    assert image.direction is None
    assert image.point_attrs["v"].shape == (2, 2, 3, 3)
    assert image.cell_attrs["c"].shape == (1, 1, 2)
    expanded = image.to_polydata()
    full = read(path)
    np.testing.assert_array_equal(expanded.vertices, full.vertices)
    np.testing.assert_array_equal(expanded.connectivity, full.connectivity)
    for name, values in full.vertex_attrs.items():
        np.testing.assert_array_equal(expanded.vertex_attrs[name], values)
    for name, values in full.element_attrs.items():
        np.testing.assert_array_equal(expanded.element_attrs[name], values)
    # The mesh reader spells the grid in global_attrs; the image holds it itself.
    assert set(full.global_attrs) - set(expanded.global_attrs) == {
        "vti_origin",
        "vti_spacing",
        "vti_extent",
        "vtkhdf_type",
    }
    assert expanded.global_attrs["gnum"] == full.global_attrs["gnum"]


def test_read_image_keeps_a_rotation(tmp_path: Path) -> None:
    path = tmp_path / "r.vtkhdf"
    swap_xy = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    f, _ = _image(path, [0, 1, 0, 1, 0, 0], origin=(1, 2, 3), direction=swap_xy)
    f.close()
    image = polyxios.read_image(path)
    np.testing.assert_array_equal(image.direction, swap_xy)
    np.testing.assert_array_equal(image.to_polydata().vertices, read(path).vertices)


def test_read_image_of_a_mesh_names_the_other_reader(tmp_path: Path) -> None:
    path = tmp_path / "m.vtkhdf"
    f, _ = _ug(path)
    f.close()
    with pytest.raises(UnsupportedFormatError, match=r"holds no image.*polyxios\.read"):
        polyxios.read_image(path)


def _volume(**changes) -> ImageData:
    """A 3 x 2 x 2 grid that starts away from zero, with every kind of array."""
    fields = {
        "extent": (1, 3, 2, 3, 0, 1),
        "origin": (10.0, 20.0, 30.0),
        "spacing": (0.5, 2.0, 4.0),
        "point_attrs": {
            "p": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            "v": np.arange(36, dtype=np.int16).reshape(2, 2, 3, 3),
            "mask": (np.arange(12) % 2 == 0).reshape(2, 2, 3),
        },
        "cell_attrs": {"c": np.arange(2, dtype=np.int32).reshape(1, 1, 2) + 5},
        "global_attrs": {"title": "grid", "gnum": np.array([42.0])},
    }
    return ImageData(**{**fields, **changes})


def test_write_image_round_trips_the_grid_and_its_arrays(tmp_path: Path) -> None:
    path = tmp_path / "v.vtkhdf"
    image = _volume()
    polyxios.write_image(image, path)
    back = polyxios.read_image(path)
    assert back.extent == image.extent
    assert back.origin == image.origin
    assert back.spacing == image.spacing
    assert back.direction is None
    np.testing.assert_array_equal(back.point_attrs["p"], image.point_attrs["p"])
    np.testing.assert_array_equal(back.point_attrs["v"], image.point_attrs["v"])
    # A boolean is written as a byte.
    np.testing.assert_array_equal(back.point_attrs["mask"], image.point_attrs["mask"])
    assert back.point_attrs["mask"].dtype == np.uint8
    np.testing.assert_array_equal(back.cell_attrs["c"], image.cell_attrs["c"])
    assert back.global_attrs["title"] == "grid"
    assert back.global_attrs["gnum"] == 42.0


def test_write_image_lays_arrays_out_the_way_the_format_does(tmp_path: Path) -> None:
    path = tmp_path / "v.vtkhdf"
    polyxios.write_image(_volume(), path)
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        assert bytes(root.attrs["Type"]) == b"ImageData"
        assert root.attrs["WholeExtent"].tolist() == [1, 3, 2, 3, 0, 1]
        assert root.attrs["Origin"].tolist() == [10.0, 20.0, 30.0]
        assert root.attrs["Spacing"].tolist() == [0.5, 2.0, 4.0]
        assert root.attrs["Direction"].tolist() == np.eye(3).ravel().tolist()
        assert root["PointData/p"].shape == (2, 2, 3)
        assert root["PointData/v"].shape == (2, 2, 3, 3)
        assert root["CellData/c"].shape == (1, 1, 2)
        assert "Steps" not in root


def test_write_image_agrees_with_the_mesh_read(tmp_path: Path) -> None:
    path = tmp_path / "v.vtkhdf"
    image = _volume()
    polyxios.write_image(image, path)
    mesh = read(path)
    expanded = image.to_polydata()
    np.testing.assert_array_equal(mesh.vertices, expanded.vertices)
    np.testing.assert_array_equal(mesh.vertex_attrs["p"], expanded.vertex_attrs["p"])
    np.testing.assert_array_equal(mesh.element_attrs["c"], [5, 6])
    assert mesh.global_attrs["vtkhdf_type"] == "ImageData"


def test_write_image_keeps_a_rotation(tmp_path: Path) -> None:
    path = tmp_path / "r.vtkhdf"
    swap_xy = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0), origin=(1.0, 2.0, 3.0), direction=swap_xy
    )
    polyxios.write_image(image, path)
    back = polyxios.read_image(path)
    np.testing.assert_array_equal(back.direction, swap_xy)
    np.testing.assert_array_equal(
        back.to_polydata().vertices, image.to_polydata().vertices
    )


def test_write_image_writes_a_flat_picture(tmp_path: Path) -> None:
    path = tmp_path / "flat.vtkhdf"
    pixels = (np.arange(4 * 5 * 3) * 7 % 256).astype(np.uint8).reshape(4, 5, 3)
    polyxios.write_image(helper.array_to_image(pixels), path)
    back = polyxios.read_image(path)
    assert back.extent == (0, 4, 0, 3, 0, 0)
    np.testing.assert_array_equal(helper.image_to_array(back), pixels)
    assert read(path).element_types.tolist() == [9] * 12


def test_write_image_passes_compression_to_h5py(tmp_path: Path) -> None:
    path = tmp_path / "z.vtkhdf"
    image = ImageData(
        extent=(0, 31, 0, 31, 0, 0),
        point_attrs={"p": np.zeros((1, 32, 32), np.float32)},
    )
    polyxios.write_image(image, path, compression="gzip")
    with h5py.File(path) as f:
        assert f["VTKHDF/PointData/p"].compression == "gzip"
    np.testing.assert_array_equal(
        polyxios.read_image(path).point_attrs["p"], image.point_attrs["p"]
    )


def test_write_image_refuses_what_no_file_can_hold(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="holds no point"):
        polyxios.write_image(
            ImageData(extent=(0, -1, 0, 1, 0, 0)), tmp_path / "e.vtkhdf"
        )
    with pytest.raises(ValueError, match="holds 4 numbers"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.zeros(4))
    with pytest.raises(CodecError, match="PointData array 'p' has shape"):
        polyxios.write_image(
            ImageData(extent=(0, 2, 0, 1, 0, 0), point_attrs={"p": np.zeros((6,))}),
            tmp_path / "s.vtkhdf",
        )


def test_write_image_names_what_it_drops(tmp_path: Path) -> None:
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={
            "p": np.zeros((1, 2, 2)),
            "label": np.array([["a", "b"], ["c", "d"]])[None],
        },
        cell_attrs={"c": np.zeros((1, 1, 1))},
        global_attrs={"bad": {"a": 1}, "vti_origin": [1.0, 2.0, 3.0]},
    )
    path = tmp_path / "w.vtkhdf"
    with pytest.warns(UserWarning) as caught:
        polyxios.write_image(image, path)
    text = " ".join(str(w.message) for w in caught)
    assert "PointData array(s) ['label']" in text
    assert "global_attrs ['bad']" in text
    back = polyxios.read_image(path)
    assert sorted(back.point_attrs) == ["p"]
    assert "vti_origin" not in back.global_attrs


def test_a_tensor_array_is_written_with_its_components_flat(tmp_path: Path) -> None:
    """One component axis past the lattice, as the .vti writer spells a tensor."""
    tensor = np.arange(2 * 2 * 3 * 9, dtype=np.float64).reshape(2, 2, 3, 3, 3)
    image = ImageData(extent=(0, 2, 0, 1, 0, 1), point_attrs={"t": tensor})
    path = tmp_path / "t.vtkhdf"
    polyxios.write_image(image, path)
    with h5py.File(path, "r") as f:
        assert f["VTKHDF/PointData/t"].shape == (2, 2, 3, 9)
    back = polyxios.read_image(path)
    np.testing.assert_array_equal(back.point_attrs["t"], tensor.reshape(2, 2, 3, 9))


def test_a_vti_whole_extent_is_never_spelled_as_field_data(tmp_path: Path) -> None:
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={"p": np.zeros((1, 2, 2))},
        global_attrs={"vti_whole_extent": [0, 9, 0, 9, 0, 0], "title": "x"},
    )
    path = tmp_path / "w.vtkhdf"
    with pytest.warns(
        UserWarning, match="global 'vti_whole_extent' is dropped"
    ) as caught:
        polyxios.write_image(image, path)
    assert {w.filename for w in caught} == {__file__}
    with h5py.File(path, "r") as f:
        assert list(f["VTKHDF/FieldData"]) == ["title"]
    assert polyxios.read_image(path).global_attrs == {"title": "x"}


def test_read_image_takes_a_step_of_a_temporal_file(tmp_path: Path) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1])
    root.create_dataset(
        "PointData/p",
        data=np.stack([np.full((2, 2, 3), t) for t in (10.0, 20.0, 30.0)]),
    )
    steps = _steps(root, [0.0, 1.0, 2.0])
    steps.create_dataset("PointDataOffsets/p", data=np.arange(3, dtype=np.int64))
    f.close()
    image = polyxios.read_image(path, step=2)
    np.testing.assert_array_equal(image.point_attrs["p"], np.full((2, 2, 3), 30.0))
    assert image.global_attrs == {"time": 2.0}


@pytest.mark.parametrize("entry", [polyxios.read_image, polyxios.read])
def test_an_image_array_the_extent_cannot_hold_is_blamed_on_the_caller(
    tmp_path: Path, entry
) -> None:
    path = tmp_path / "i.vtkhdf"
    f, root = _image(path, [0, 2, 0, 1, 0, 1])
    root.create_dataset("PointData/bad", data=np.zeros((2, 2, 2)))
    f.close()
    with pytest.warns(UserWarning, match="array 'bad' is shaped") as caught:
        entry(path)
    assert caught[0].filename == __file__


def test_a_mesh_keeps_a_vti_whole_extent_as_field_data(tmp_path: Path) -> None:
    """Only the image path reserves it; the mesh a .vti piece read as keeps it."""
    mesh = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    mesh.global_attrs["vti_whole_extent"] = [0, 9, 0, 9, 0, 0]
    path = tmp_path / "m.vtkhdf"
    polyxios.write(mesh, path)
    with h5py.File(path, "r") as f:
        assert "vti_whole_extent" in f["VTKHDF/FieldData"]
    assert polyxios.read(path).global_attrs["vti_whole_extent"].tolist() == [
        0,
        9,
        0,
        9,
        0,
        0,
    ]


def test_write_image_warnings_are_blamed_on_the_caller(tmp_path: Path) -> None:
    """Every warning the image writer raises names the caller's file, whichever
    helper raised it: the drops, the renamed names, the NUL and the globals."""
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={
            "p": np.zeros((1, 2, 2)),
            "label": np.array(list("abcd")).reshape(1, 2, 2),
        },
        cell_attrs={"a/b": np.zeros((1, 1, 1))},
        global_attrs={"bad": {"x": 1}, "t": "has\x00nul", "": 3},
    )
    with pytest.warns(UserWarning) as caught:
        polyxios.write_image(image, tmp_path / "w.vtkhdf")
    messages = " | ".join(str(w.message) for w in caught)
    for part in (
        "PointData array(s) ['label']",
        "CellData array(s) {'a/b': 'a_b'}",
        "global_attrs ['bad']",
        "text global 't' holds a NUL",
        "global_attrs ['']",
    ):
        assert part in messages
    assert {w.filename for w in caught} == {__file__}


def _mesh_with_bad_globals() -> polyxios.PolyData:
    mesh = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    mesh.global_attrs.update({"bad": {"x": 1}, "sl/ash": 2.0})
    return mesh


def test_a_mesh_write_is_blamed_on_the_caller(tmp_path: Path) -> None:
    """The mesh writer's drops and renames point at the line that asked for them,
    not at polyxios' own entry point."""
    with pytest.warns(UserWarning) as caught:
        polyxios.write(_mesh_with_bad_globals(), tmp_path / "m.vtkhdf")
    messages = " | ".join(str(w.message) for w in caught)
    assert "global_attrs ['bad'] hold values no" in messages
    assert "FieldData array(s) {'sl/ash': 'sl_ash'}" in messages
    assert {w.filename for w in caught} == {__file__}


def test_a_series_write_is_blamed_on_the_caller(tmp_path: Path) -> None:
    """write_time_series reaches the same writer through the helper; the warnings
    still name this file."""
    with pytest.warns(UserWarning) as caught:
        helper.write_time_series(
            ((0.0, _mesh_with_bad_globals()),), tmp_path / "s.vtkhdf"
        )
    messages = " | ".join(str(w.message) for w in caught)
    assert "global_attrs ['bad'] hold values no" in messages
    assert "FieldData array(s) {'sl/ash': 'sl_ash'}" in messages
    assert {w.filename for w in caught} == {__file__}


def test_the_whole_extent_warning_comes_after_the_empty_grid_refusal(
    tmp_path: Path,
) -> None:
    """An empty grid is refused outright; nothing about its whole extent is said
    first, since the write never happens."""
    empty = ImageData(
        extent=(0, -1, 0, 1, 0, 0),
        global_attrs={"vti_whole_extent": [0, 9, 0, 9, 0, 0]},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(CodecError, match="holds no point"):
            polyxios.write_image(empty, tmp_path / "e.vtkhdf")
    assert not any("vti_whole_extent" in str(w.message) for w in caught)


def test_read_keeps_the_vtk_laid_out_arrays_the_old_reader_dropped(
    tmp_path: Path,
) -> None:
    """A VTK-written single point's vector and an xz plane's cells used to be dropped
    with a warning by read(); they are kept now, and nothing else about the mesh
    changes."""
    plane = tmp_path / "plane.vtkhdf"
    with h5py.File(plane, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array((2, 0), dtype=np.int64)
        root.attrs["Type"] = np.bytes_("ImageData")
        root.attrs["WholeExtent"] = np.array((0, 2, 0, 0, 0, 1), dtype=np.int64)
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.ones(3)
        root.create_group("PointData")["p"] = np.arange(6.0).reshape(2, 3)
        root.create_group("CellData")["c"] = np.arange(2.0).reshape(1, 2)
    point = tmp_path / "point.vtkhdf"
    with h5py.File(point, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array((2, 0), dtype=np.int64)
        root.attrs["Type"] = np.bytes_("ImageData")
        root.attrs["WholeExtent"] = np.array((0, 0, 0, 0, 0, 0), dtype=np.int64)
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.ones(3)
        root.create_group("PointData")["v"] = np.arange(3.0).reshape(1, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from_plane = read(plane)
        from_point = read(point)
    assert not caught
    np.testing.assert_array_equal(from_plane.element_attrs["c"], [0.0, 1.0])
    np.testing.assert_array_equal(from_point.vertex_attrs["v"], [[0.0, 1.0, 2.0]])


# Rows of the spelling sweep that earlier releases read: a cell array of one voxel and
# of a (2, 2) plane, a scalar on a single point, and a vector over a single point. Each
# is read back as earlier releases read it.
@pytest.mark.parametrize(
    ("extent", "group", "shape", "kept"),
    [
        ((0, 1, 0, 2, 0, 2), "CellData", (2, 2), [0.0, 1.0, 2.0, 3.0]),
        ((0, 1, 0, 1, 0, 1), "CellData", (1,), [0.0]),
        ((0, 1, 0, 1, 0, 1), "CellData", (3,), [0.0, 1.0, 2.0]),
        ((0, 0, 0, 0, 0, 0), "PointData", (), [0.0]),
        ((0, 0, 0, 0, 0, 0), "PointData", (3,), [0.0, 1.0, 2.0]),
    ],
)
def test_read_keeps_each_spelling_of_an_image_array_earlier_releases_kept(
    tmp_path: Path, extent, group, shape, kept
) -> None:
    path = tmp_path / "rows.vtkhdf"
    with h5py.File(path, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array((2, 0), dtype=np.int64)
        root.attrs["Type"] = np.bytes_("ImageData")
        root.attrs["WholeExtent"] = np.array(extent, dtype=np.int64)
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.ones(3)
        values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
        root.create_group(group)["a"] = values
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = read(path)
    assert not [w for w in caught if "array 'a'" in str(w.message)]
    attrs = mesh.element_attrs if group == "CellData" else mesh.vertex_attrs
    np.testing.assert_array_equal(np.asarray(attrs["a"]).ravel(), kept)


def test_a_single_points_cell_array_keeps_the_warning_earlier_releases_gave(
    tmp_path: Path,
) -> None:
    """A (1, 2) cell array over a single point spans no cell: it is dropped, and the
    warning gives its shape and the extent's, as earlier releases did."""
    path = tmp_path / "point.vtkhdf"
    with h5py.File(path, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array((2, 0), dtype=np.int64)
        root.attrs["Type"] = np.bytes_("ImageData")
        root.attrs["WholeExtent"] = np.zeros(6, dtype=np.int64)
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.ones(3)
        root.create_group("CellData")["a"] = np.arange(2.0).reshape(1, 2)
    with pytest.warns(
        UserWarning,
        match=r"CellData array 'a' is shaped \(1, 2\) where the extent asks for "
        r"\(1, 1, 1\); dropped\.",
    ):
        read(path)


def test_an_extent_past_int64_reads_as_a_mesh_and_refuses_as_an_image(
    tmp_path: Path,
) -> None:
    """A mesh of one vertex read from such an extent is what read() always gave; an
    ImageData cannot hold the extent, so read_image refuses it as a CodecError."""
    path = tmp_path / "huge.vtkhdf"
    with h5py.File(path, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array((2, 0), dtype=np.int64)
        root.attrs["Type"] = np.bytes_("ImageData")
        root.attrs["WholeExtent"] = np.array([1e19, 1e19, 0, 0, 0, 0])
        root.attrs["Origin"] = np.zeros(3)
        root.attrs["Spacing"] = np.ones(3)
    assert read(path).vertices.tolist() == [[1e19, 0.0, 0.0]]
    with pytest.raises(CodecError, match="outside the 64-bit"):
        polyxios.read_image(path)


def test_a_bare_grid_past_the_mesh_cap_reads_compact(tmp_path: Path) -> None:
    """A bare grid of more points than a mesh may hold builds no vertex, so read_image
    reads it; read() still refuses it as the mesh cap always has, and so does
    to_polydata."""
    n = 465
    image = ImageData(extent=(0, n - 1, 0, n - 1, 0, n - 1))
    path = tmp_path / "bare.vtkhdf"
    polyxios.write_image(image, path)
    assert polyxios.read_image(path).dimensions == (n, n, n)
    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        read(path)
    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        image.to_polydata()


def test_a_grid_past_the_point_cap_is_refused_before_it_is_written(
    tmp_path: Path,
) -> None:
    """A reader refuses a compact image past MAX_SAFE_VERTICES, so the writer refuses it
    first and leaves no file that cannot be read back."""
    path = tmp_path / "huge.vtkhdf"
    with pytest.raises(ValidationError, match="MAX_SAFE_VERTICES"):
        polyxios.write_image(ImageData(extent=(0, 999, 0, 999, 0, 999)), path)
    assert not path.exists()


@pytest.mark.parametrize("first", [2**53, 2**63 - 2])
def test_a_wide_extent_round_trips_exactly(tmp_path: Path, first: int) -> None:
    """A WholeExtent past 2**53 is spelled as the integers it is, so a two-point grid
    there reads back as the two points it was, not one rounded onto the other."""
    image = ImageData(
        extent=(first, first + 1, 0, 0, 0, 0),
        point_attrs={"a": np.array([[[1.0, 2.0]]])},
    )
    path = tmp_path / "wide.vtkhdf"
    polyxios.write_image(image, path)
    back = polyxios.read_image(path)
    assert back == image
    assert back.extent == (first, first + 1, 0, 0, 0, 0)
