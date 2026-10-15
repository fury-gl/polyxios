from __future__ import annotations

import dataclasses
import gzip
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import tracemalloc
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._io import GZIP_SUFFIXES
from polyxios._registry import Codec, _make_dispatcher, build_default_registry
from polyxios.codecs import _exodus, _tecplot
from polyxios.codecs._vtk_xml import structured_cells
from polyxios.exceptions import CodecError, UnsupportedFormatError
from polyxios.napari import (
    _GRID_GLOBALS,
    _NOT_READ,
    _PLAIN_ONLY,
    _READ_TESTS,
    _SHARED_SUFFIXES,
    META_KEY,
    _grid_spans,
    _head,
    get_reader,
    layer_data,
    write_points,
    write_surface,
)
from polyxios.transforms import extract_surface, triangulate

_MANIFEST = Path(polyxios.__file__).parent / "napari.yaml"

# The writer extensions that need an optional package, and which one.
_NEEDS: dict[str, str] = {
    ".vtkhdf": "h5py",
    ".med": "h5py",
    ".exo": "netCDF4",
    ".laz": "lazrs",
}

_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 3.0]],
)


def _surface() -> polyxios.PolyData:
    return make_polydata(
        _VERTS,
        [("quad", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"temp": np.arange(4.0), "normals": np.ones((4, 3))},
    )


def _cloud() -> polyxios.PolyData:
    return make_polydata(
        _VERTS,
        [],
        vertex_attrs={
            "intensity": np.array([1.0, 2.0, 3.0, 4.0]),
            "colors": np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32
            ),
        },
    )


def _manifest() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(_MANIFEST.read_text())


def _tri_set(triangles: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(sorted(t)) for t in np.asarray(triangles).tolist()}


def test_surface_layer_reverses_axes_and_triangulates():
    [(data, kwargs, layer_type)] = layer_data(_surface(), name="m")
    vertices, faces, values = data
    assert layer_type == "surface"
    np.testing.assert_array_equal(vertices, _VERTS[:, ::-1])
    assert faces.shape == (2, 3)
    np.testing.assert_array_equal(values, np.arange(4.0))
    info = kwargs["metadata"][META_KEY]
    assert info["axis_order"] == "zyx"
    assert info["values"] == "temp"
    assert kwargs["name"] == "m"


def test_surface_layer_xyz_passes_axes_through():
    [(data, kwargs, _)] = layer_data(_surface(), name="m", axis_order="xyz")
    np.testing.assert_array_equal(data[0], _VERTS)
    assert kwargs["metadata"][META_KEY]["axis_order"] == "xyz"


def test_surface_without_scalars_has_no_values():
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    assert len(data) == 2
    assert "values" not in kwargs["metadata"][META_KEY]


def test_volume_mesh_opens_as_its_boundary():
    poly = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    [(data, _, layer_type)] = layer_data(poly, name="m")
    assert layer_type == "surface"
    assert data[1].shape == (4, 3)


def test_boundary_patch_on_a_volume_is_drawn_once():
    # A gmsh or Abaqus deck lists the skin triangles next to the tetrahedra.
    tet = np.array([[0, 1, 2, 3]])
    skin = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]])
    poly = make_polydata(_VERTS, [("tetra", tet), ("triangle", skin)])
    [(data, _, _)] = layer_data(poly, name="m")
    assert data[1].shape == (4, 3)
    assert _tri_set(data[1]) == _tri_set(skin)


def test_surface_cell_off_the_skin_is_kept_next_to_a_volume():
    verts = np.vstack([_VERTS, [[5.0, 5.0, 5.0]]])
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 4]]))],
    )
    [(data, _, _)] = layer_data(poly, name="m")
    assert data[1].shape == (5, 3)


def test_triangles_match_the_transforms_on_a_mixed_mesh():
    rng = np.random.default_rng(0)
    verts = rng.random((30, 3))
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("quad", np.array([[3, 4, 5, 6], [7, 8, 9, 10]])),
            ("polygon", np.array([[11, 12, 13, 14, 15]])),
            ("quadratic_triangle", np.array([[16, 17, 18, 19, 20, 21]])),
            ("tetra", np.array([[0, 1, 2, 22]])),
            ("hexahedron", np.array([[3, 4, 5, 6, 23, 24, 25, 26]])),
            ("line", np.array([[27, 28]])),
        ],
    )
    with pytest.warns(UserWarning, match="line cells are not drawn"):
        [(data, _, _)] = layer_data(poly, name="m")
    reference = np.concatenate(
        [
            triangulate(poly).connectivity.reshape(-1, 3),
            triangulate(extract_surface(poly)).connectivity.reshape(-1, 3),
        ]
    )
    # The two cells the solids cover - triangle 0 1 2 and quad 3 4 5 6 - are
    # drawn once; the transforms draw them a second time.
    assert _tri_set(data[1]) == _tri_set(reference)
    assert len(data[1]) == len(reference) - 3


def test_pixel_is_split_along_its_diagonal():
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    poly = make_polydata(verts, [("pixel", np.array([[0, 1, 2, 3]]))])
    [(data, _, _)] = layer_data(poly, name="p")
    assert _tri_set(data[1]) == {(0, 1, 3), (0, 2, 3)}


def test_triangle_strip_alternates_its_triangles():
    verts = np.array([[0.0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    poly = make_polydata(verts, [("triangle_strip", np.array([[0, 1, 2, 3]]))])
    [(data, _, _)] = layer_data(poly, name="s", axis_order="xyz")
    np.testing.assert_array_equal(data[1], [[0, 1, 2], [2, 1, 3]])


def _drawn_normal(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # napari draws a Surface with its columns and its faces both reversed.
    v = np.asarray(vertices)[:, ::-1]
    a, b, c = v[np.asarray(faces)[0, ::-1]]
    return np.cross(b - a, c - a)


@pytest.mark.parametrize("axis_order", ["zyx", "xyz"])
def test_triangles_face_the_way_the_file_winds_them(tmp_path, axis_order):
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="t", axis_order=axis_order)
    normal = _drawn_normal(data[0], data[1])
    # Under xyz napari draws the file mirrored, x and z swapped.
    expected = [0, 0, 1] if axis_order == "zyx" else [1, 0, 0]
    np.testing.assert_allclose(normal, expected)
    out = tmp_path / "t.vtp"
    write_surface(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).connectivity, [0, 1, 2])


def test_a_surface_drawn_in_napari_keeps_its_drawn_facing(tmp_path):
    # No metadata: taken as (z, y, x), faces wound for that order.
    zyx = np.array([[0.0, 0, 0], [0, 0, 1], [0, 1, 0]])
    faces = np.array([[2, 1, 0]])
    out = tmp_path / "t.vtp"
    write_surface(str(out), (zyx, faces), {})
    back = polyxios.read(out)
    a, b, c = back.vertices[back.connectivity]
    np.testing.assert_allclose(np.cross(b - a, c - a), _drawn_normal(zyx, faces))


@pytest.mark.parametrize("writer", [write_surface, write_points])
def test_an_unknown_axis_order_in_the_metadata_is_refused(tmp_path, writer):
    [(data, kwargs, _)] = layer_data(_surface(), name="m")
    kwargs["metadata"][META_KEY]["axis_order"] = "yxz"
    if writer is write_points:
        data = data[0]
    with pytest.raises(ValueError, match="'yxz'"):
        writer(str(tmp_path / "m.vtp"), data, kwargs)


def test_values_prefer_a_measured_field_over_an_index():
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "original_index": np.arange(4),
            "mask": np.ones(4, dtype=bool),
            "temp": np.arange(4.0) / 2,
        },
    )
    [(data, kwargs, _)] = layer_data(poly, name="m")
    assert kwargs["metadata"][META_KEY]["values"] == "temp"
    np.testing.assert_array_equal(data[2], np.arange(4.0) / 2)


def test_point_cloud_layer_keeps_features_and_colors():
    [(data, kwargs, layer_type)] = layer_data(_cloud(), name="c")
    assert layer_type == "points"
    np.testing.assert_array_equal(data, _VERTS[:, ::-1])
    assert list(kwargs["features"]) == ["intensity"]
    assert kwargs["face_color"].shape == (4, 4)
    assert kwargs["metadata"][META_KEY]["colors"]["names"] == ["colors"]
    assert kwargs["size"] == pytest.approx(3.0 * 0.002)


def test_normals_are_not_taken_for_colours(tmp_path):
    poly = make_polydata(_VERTS, [], vertex_attrs={"normals": np.full((4, 3), 0.5)})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    assert "face_color" not in kwargs
    assert "colors" not in kwargs["metadata"][META_KEY]
    out = tmp_path / "n.pcd"
    write_points(str(out), data, {**kwargs, "face_color": np.ones((4, 4))})
    back = polyxios.read(out)
    assert "colors" not in back.vertex_attrs
    np.testing.assert_allclose(back.vertex_attrs["normals"], 0.5)


def test_colours_with_no_finite_value_are_not_shown():
    poly = make_polydata(_VERTS, [], vertex_attrs={"colors": np.full((4, 3), np.nan)})
    [(_, kwargs, _)] = layer_data(poly, name="c")
    assert "face_color" not in kwargs


def test_colours_of_another_length_are_not_shown():
    # napari refuses a colour array whose length is not the vertex count.
    poly = make_polydata(_VERTS, [], vertex_attrs={"colors": np.ones((3, 3))})
    [(_, kwargs, _)] = layer_data(poly, name="c")
    assert "face_color" not in kwargs
    assert "colors" in kwargs["metadata"][META_KEY]["vertex_attrs"]


@pytest.mark.parametrize(
    "channels",
    [
        {"colors": np.array([[np.nan, 0, 0], [1, 1, 1]], dtype=np.float32)},
        {
            "red": np.array([255, 0], dtype=np.uint8),
            "green": np.array([0, 9], dtype=np.uint8),
            "blue": np.array([0, 1], dtype=np.uint8),
            "alpha": np.array([np.nan, 0.5], dtype=np.float32),
        },
    ],
)
def test_a_nan_colour_channel_shows_at_full(channels):
    poly = make_polydata(_VERTS[:2], [], vertex_attrs=channels)
    [(_, kwargs, _)] = layer_data(poly, name="c")
    assert np.isfinite(kwargs["face_color"]).all()
    assert kwargs["face_color"][0, 0] == 1.0
    assert kwargs["face_color"][0, 3] == 1.0


def test_an_empty_cloud_with_normals_opens_and_saves(tmp_path):
    normals = {k: np.empty(0, np.float32) for k in ("nx", "ny", "nz")}
    src = tmp_path / "e.ply"
    polyxios.write(make_polydata(np.empty((0, 3)), [], vertex_attrs=normals), src)
    [(data, kwargs, _)] = get_reader(path=str(src))(str(src))
    assert data.shape == (0, 3)
    out = tmp_path / "out.ply"
    write_points(str(out), data, kwargs)
    assert len(polyxios.read(out).vertices) == 0
    with pytest.warns(UserWarning, match="not written"):
        write_points(str(out), np.ones((1, 3)), kwargs)
    assert len(polyxios.read(out).vertices) == 1


def test_empty_points_layer_writes_its_colours_without_a_warning(tmp_path):
    # napari hands an empty Points layer its current colour as one row.
    [(_, kwargs, _)] = layer_data(_cloud(), name="c")
    out = tmp_path / "c.vtp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_points(
            str(out),
            np.empty((0, 3)),
            {**kwargs, "features": {}, "face_color": np.ones((1, 4))},
        )
    assert len(polyxios.read(out).vertices) == 0


def test_ply_colour_channels_show_and_write_back(tmp_path):
    src = tmp_path / "c.ply"
    src.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nproperty uchar red\n"
        "property uchar green\nproperty uchar blue\nend_header\n"
        "0 0 0 255 0 0\n1 1 1 0 128 255\n"
    )
    [(data, kwargs, _)] = get_reader(path=str(src))(str(src))
    np.testing.assert_allclose(
        kwargs["face_color"][:, :3], [[1, 0, 0], [0, 128 / 255, 1]], atol=1e-6
    )
    assert "red" not in kwargs["features"]
    out = tmp_path / "out.ply"
    write_points(str(out), data, kwargs)
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs["red"], [255, 0])
    np.testing.assert_array_equal(back.vertex_attrs["green"], [0, 128])
    np.testing.assert_array_equal(back.vertex_attrs["blue"], [0, 255])


def test_sixteen_bit_colours_keep_their_range(tmp_path):
    rgb = np.array([[65535, 0, 0], [0, 32768, 0], [0, 0, 65535], [1, 2, 3]])
    poly = make_polydata(_VERTS, [], vertex_attrs={"colors": rgb.astype(np.uint16)})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    np.testing.assert_allclose(kwargs["face_color"][0], [1, 0, 0, 1])
    out = tmp_path / "c.vtp"
    write_points(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["colors"], rgb)


def test_sixteen_bit_colours_below_255_are_not_taken_for_8_bit(tmp_path):
    rgb = np.array([[10, 20, 30]] * 4, dtype=np.uint16)
    poly = make_polydata(_VERTS, [], vertex_attrs={"colors": rgb})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    np.testing.assert_allclose(kwargs["face_color"][0, :3], rgb[0] / 65535, rtol=1e-6)
    out = tmp_path / "c.vtp"
    write_points(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["colors"], rgb)


def test_float_colours_keep_their_fractions(tmp_path):
    rgb = np.array([[0.5, 100.7, 3.0]] * 4)
    poly = make_polydata(_VERTS, [], vertex_attrs={"colors": rgb})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    out = tmp_path / "c.vtp"
    write_points(str(out), data, kwargs)
    np.testing.assert_allclose(
        polyxios.read(out).vertex_attrs["colors"], rgb, atol=1e-4
    )


def test_an_alpha_set_in_napari_is_written_where_the_colours_can_hold_it(tmp_path):
    [(data, kwargs, _)] = layer_data(_cloud(), name="c")
    rgba = kwargs["face_color"].copy()
    rgba[:, 3] = 0.5
    out = tmp_path / "c.vtp"
    write_points(str(out), data, {**kwargs, "face_color": rgba})
    np.testing.assert_allclose(polyxios.read(out).vertex_attrs["colors"], rgba)


def test_an_alpha_set_in_napari_adds_a_ply_alpha_channel(tmp_path):
    src = tmp_path / "c.ply"
    src.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nproperty uchar red\n"
        "property uchar green\nproperty uchar blue\nend_header\n"
        "0 0 0 255 0 0\n1 1 1 0 128 255\n"
    )
    [(data, kwargs, _)] = get_reader(path=str(src))(str(src))
    rgba = kwargs["face_color"].copy()
    rgba[0, 3] = 0.2
    out = tmp_path / "out.ply"
    write_points(str(out), data, {**kwargs, "face_color": rgba})
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["alpha"], [51, 255])


def test_an_alpha_with_no_channel_to_go_to_is_warned_about(tmp_path):
    channels = {k: np.full(4, 0.5) for k in ("color_r", "color_g", "color_b")}
    poly = make_polydata(_VERTS, [], vertex_attrs=channels)
    [(data, kwargs, _)] = layer_data(poly, name="c")
    rgba = kwargs["face_color"].copy()
    rgba[:, 3] = 0.5
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="alpha was not written"):
        write_points(str(out), data, {**kwargs, "face_color": rgba})
    assert set(polyxios.read(out).vertex_attrs) == set(channels)


def test_rgb_colours_on_a_layer_read_with_alpha_write_it_opaque(tmp_path):
    rgba = np.array([[1, 0, 0, 0.2], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]])
    channels = dict(zip(("red", "green", "blue", "alpha"), (rgba * 255).T))
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={k: v.astype(np.uint8) for k, v in channels.items()},
    )
    [(data, kwargs, _)] = layer_data(poly, name="c")
    out = tmp_path / "c.ply"
    write_surface(str(out), data, {**kwargs, "vertex_colors": rgba[:, :3]})
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["alpha"], [255] * 4)


def test_a_float_alpha_beside_byte_colours_keeps_its_range_and_dtype(tmp_path):
    src = tmp_path / "c.ply"
    channels = {
        "red": np.array([255, 0, 0, 9], dtype=np.uint8),
        "green": np.array([0, 128, 0, 9], dtype=np.uint8),
        "blue": np.array([0, 255, 64, 9], dtype=np.uint8),
        "alpha": np.array([0.2, 1.0, 0.5, 0.0], dtype=np.float32),
    }
    polyxios.write(make_polydata(_VERTS, [], vertex_attrs=channels), src)
    [(data, kwargs, _)] = get_reader(path=str(src))(str(src))
    np.testing.assert_allclose(kwargs["face_color"][:, 3], channels["alpha"], atol=1e-6)
    out = tmp_path / "out.ply"
    write_points(str(out), data, kwargs)
    back = polyxios.read(out).vertex_attrs
    for name, values in channels.items():
        assert back[name].dtype == values.dtype, name
        np.testing.assert_allclose(back[name], values, atol=1e-6)


@pytest.mark.parametrize("layer_type", ["surface", "points"])
def test_an_added_alpha_does_not_replace_the_files_own_alpha(tmp_path, layer_type):
    # An "alpha" that is not one number per vertex is not a colour channel;
    # it rides along with the layer and must come back as it was.
    rgb = {c: np.array([1, 2, 3, 4], dtype=np.uint8) for c in ("red", "green", "blue")}
    other = np.arange(8.0).reshape(4, 2)
    cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
    poly = make_polydata(
        _VERTS,
        cells if layer_type == "surface" else [],
        vertex_attrs={**rgb, "alpha": other},
    )
    [(data, kwargs, _)] = layer_data(poly, name="m")
    key = "vertex_colors" if layer_type == "surface" else "face_color"
    kwargs[key] = kwargs[key].copy()
    kwargs[key][:, 3] = 0.5
    write = write_surface if layer_type == "surface" else write_points
    out = tmp_path / "out.vtp"
    with pytest.warns(UserWarning, match="alpha was not written"):
        write(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["alpha"], other)


def test_a_flag_named_alpha_is_not_an_opacity():
    rgb = {c: np.array([1, 2, 3, 4], dtype=np.uint8) for c in ("red", "green", "blue")}
    flag = np.array([True, False, True, True])
    poly = make_polydata(_VERTS, [], vertex_attrs={**rgb, "alpha": flag})
    [(_, kwargs, _)] = layer_data(poly, name="c")
    np.testing.assert_array_equal(kwargs["face_color"][:, 3], 1.0)
    np.testing.assert_array_equal(kwargs["features"]["alpha"], flag)


def test_a_flag_named_alpha_keeps_its_name_beside_a_set_opacity(tmp_path):
    rgb = {c: np.array([1, 2, 3, 4], dtype=np.uint8) for c in ("red", "green", "blue")}
    flag = np.array([1, 0, 1, 1], dtype=np.int8)
    poly = make_polydata(_VERTS, [], vertex_attrs={**rgb, "alpha": flag > 0})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    see_through = np.array(kwargs["face_color"])
    see_through[:, 3] = 0.5
    out = tmp_path / "out.vtp"
    with pytest.warns(UserWarning, match="alpha was not written"):
        write_points(str(out), data, kwargs | {"face_color": see_through})
    back = polyxios.read(out).vertex_attrs
    assert "alpha_1" not in back
    np.testing.assert_array_equal(back["alpha"], flag > 0)


def test_ply_alpha_channel_is_a_colour(tmp_path):
    src = tmp_path / "c.ply"
    src.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nproperty uchar red\n"
        "property uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        "end_header\n0 0 0 255 0 0 51\n1 1 1 0 128 255 255\n"
    )
    [(data, kwargs, _)] = get_reader(path=str(src))(str(src))
    np.testing.assert_allclose(kwargs["face_color"][:, 3], [0.2, 1.0], atol=1e-6)
    assert "alpha" not in kwargs["features"]
    out = tmp_path / "out.ply"
    write_points(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["alpha"], [51, 255])


def test_surface_colours_show_and_write_back(tmp_path):
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], np.uint8)
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        vertex_attrs={"colors": rgb},
    )
    [(data, kwargs, _)] = layer_data(poly, name="m")
    np.testing.assert_allclose(kwargs["vertex_colors"][:, :3], rgb / 255)
    out = tmp_path / "out.ply"
    write_surface(str(out), data, kwargs)
    back = polyxios.read(out)
    stored = back.vertex_attrs.get("colors")
    if stored is None:
        stored = np.column_stack([back.vertex_attrs[f"colors_{i}"] for i in range(3)])
    np.testing.assert_array_equal(stored, rgb)


def test_values_skip_normal_and_texture_components():
    tri = [("triangle", np.array([[0, 1, 2]]))]
    ply_like = {
        "nx": np.zeros(4),
        "ny": np.zeros(4),
        "nz": np.ones(4),
        "s": np.zeros(4),
        "t": np.zeros(4),
        "temp": np.arange(4.0),
    }
    [(_, kwargs, _)] = layer_data(
        make_polydata(_VERTS, tri, vertex_attrs=ply_like), name="m"
    )
    assert kwargs["metadata"][META_KEY]["values"] == "temp"
    del ply_like["temp"]
    [(data, kwargs, _)] = layer_data(
        make_polydata(_VERTS, tri, vertex_attrs=ply_like), name="m"
    )
    assert len(data) == 2
    assert set(kwargs["metadata"][META_KEY]["vertex_attrs"]) == set(ply_like)


@pytest.mark.parametrize(
    ("columns", "picked"),
    [
        (("t",), "t"),
        (("u", "v", "w"), "u"),
        (("s", "v"), "s"),
        (("u", "v", "temp"), "temp"),
        (("s", "t"), None),
        (("U", "V"), None),
    ],
)
def test_values_skip_texture_components_only_in_pairs(columns, picked):
    tri = [("triangle", np.array([[0, 1, 2]]))]
    attrs = {c: np.arange(4.0) for c in columns}
    [(_, kwargs, _)] = layer_data(
        make_polydata(_VERTS, tri, vertex_attrs=attrs), name="m"
    )
    assert kwargs["metadata"][META_KEY].get("values") == picked


def test_line_set_opens_as_points():
    poly = make_polydata(_VERTS, [("line", np.array([[0, 1], [2, 3]]))])
    [(_, _, layer_type)] = layer_data(poly, name="l")
    assert layer_type == "points"


@pytest.mark.parametrize(
    ("kind", "width"),
    [("quadratic_linear_quad", 6), ("lagrange_triangle", 6), ("polyhedron", 4)],
)
def test_cells_no_layer_draws_warn_when_opening_as_points(kind, width):
    verts = np.arange(width * 3.0).reshape(width, 3)
    poly = make_polydata(verts, [(kind, np.arange(width)[None])])
    with pytest.warns(UserWarning, match=f"{kind} cells are not drawn"):
        [(_, kwargs, layer_type)] = layer_data(poly, name="q")
    assert layer_type == "points"
    assert kwargs["metadata"][META_KEY]["cells"] is not None


def test_line_set_keeps_its_lines_on_save(tmp_path):
    lines = np.array([[0, 1], [2, 3]])
    poly = make_polydata(_VERTS, [("line", lines)])
    [(data, kwargs, _)] = layer_data(poly, name="l")
    out = tmp_path / "l.vtp"
    write_points(str(out), data, kwargs)
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.connectivity.reshape(-1, 2), lines)


def test_points_keep_vector_attributes_and_tags(tmp_path):
    normals = np.tile([0.0, 0.0, -1.0], (4, 1))
    poly = make_polydata(
        _VERTS,
        [],
        vertex_attrs={"normals": normals},
        vertex_tags={"top": np.array([3])},
    )
    [(data, kwargs, _)] = layer_data(poly, name="c")
    out = tmp_path / "c.vtp"
    write_points(str(out), data, kwargs)
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs["normals"], normals)


def test_surface_keeps_the_attributes_it_does_not_show(tmp_path):
    [(data, kwargs, _)] = layer_data(_surface(), name="m")
    out = tmp_path / "m.vtp"
    with pytest.warns(UserWarning, match="quad cells are written as triangles"):
        write_surface(str(out), data, kwargs)
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs["normals"], np.ones((4, 3)))


def test_edited_layer_warns_and_drops_what_no_longer_lines_up(tmp_path):
    poly = make_polydata(
        _VERTS,
        [("line", np.array([[0, 1], [2, 3]]))],
        vertex_attrs={"normals": np.ones((4, 3))},
    )
    [(data, kwargs, _)] = layer_data(poly, name="c")
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="3 vertices where the file had 4"):
        write_points(str(out), data[:3], kwargs)
    back = polyxios.read(out)
    assert len(back.vertices) == 3
    assert "normals" not in back.vertex_attrs
    assert len(back.element_types) == 0


def test_point_deleted_and_another_added_drops_what_no_longer_lines_up(tmp_path):
    normals = np.arange(12.0).reshape(4, 3)
    poly = make_polydata(
        _VERTS,
        [],
        vertex_attrs={"normals": normals},
        vertex_tags={"top": np.array([3])},
    )
    [(data, kwargs, _)] = layer_data(poly, name="c")
    edited = np.vstack([np.delete(data, 1, axis=0), [[9.0, 9.0, 9.0]]])
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="deleted, added or reordered"):
        write_points(str(out), edited, kwargs)
    assert "normals" not in polyxios.read(out).vertex_attrs


def test_moved_point_keeps_its_attributes(tmp_path):
    flow = np.arange(12.0).reshape(4, 3)
    poly = make_polydata(_VERTS, [], vertex_attrs={"flow": flow})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    moved = data.copy()
    moved[1] += 0.5
    out = tmp_path / "c.vtp"
    write_points(str(out), moved, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["flow"], flow)


def test_moved_point_drops_the_normals_it_no_longer_has(tmp_path):
    poly = make_polydata(
        _VERTS,
        [],
        vertex_attrs={
            "normals": np.ones((4, 3)),
            "nx": np.zeros(4),
            "u": np.arange(4.0),
        },
    )
    [(data, kwargs, _)] = layer_data(poly, name="c")
    moved = data.copy()
    moved[1] += 0.5
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="normals, nx describe the points as read"):
        write_points(str(out), moved, kwargs)
    back = polyxios.read(out).vertex_attrs
    assert "normals" not in back
    assert "nx" not in back
    np.testing.assert_array_equal(back["u"], np.arange(4.0))


def test_moved_surface_vertex_drops_its_normals(tmp_path):
    [(data, kwargs, _)] = layer_data(_surface(), name="m")
    vertices = data[0].copy()
    vertices[0] += 0.5
    out = tmp_path / "m.vtp"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_surface(str(out), (vertices, *data[1:]), kwargs)
    assert any("normals describe" in str(w.message) for w in caught)
    assert "normals" not in polyxios.read(out).vertex_attrs


def _normal_cloud() -> polyxios.PolyData:
    return make_polydata(
        _VERTS,
        [],
        vertex_attrs={"nx": np.arange(4.0), "temp": np.arange(4.0) + 10},
    )


def test_point_deleted_and_another_moved_drops_the_normal_features(tmp_path):
    # napari deletes a point's feature row with it, so the features still
    # follow the points; the moved one's normal is stale all the same.
    [(data, kwargs, _)] = layer_data(_normal_cloud(), name="c")
    keep = [1, 2, 3]
    edited = data[keep].copy()
    edited[1] += 0.5
    features = {k: np.asarray(v)[keep] for k, v in kwargs["features"].items()}
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="nx describe the points as read"):
        write_points(str(out), edited, kwargs | {"features": features})
    back = polyxios.read(out).vertex_attrs
    assert "nx" not in back
    np.testing.assert_array_equal(back["temp"], [11.0, 12.0, 13.0])


def test_point_added_drops_the_normal_features(tmp_path):
    [(data, kwargs, _)] = layer_data(_normal_cloud(), name="c")
    edited = np.vstack([data, [[9.0, 9.0, 9.0]]])
    features = {k: np.append(v, v[-1]) for k, v in kwargs["features"].items()}
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="nx describe the points as read"):
        write_points(str(out), edited, kwargs | {"features": features})
    assert "nx" not in polyxios.read(out).vertex_attrs


def test_point_deleted_alone_keeps_the_normal_features(tmp_path):
    [(data, kwargs, _)] = layer_data(_normal_cloud(), name="c")
    features = {k: np.asarray(v)[1:] for k, v in kwargs["features"].items()}
    out = tmp_path / "c.vtp"
    write_points(str(out), data[1:], kwargs | {"features": features})
    np.testing.assert_array_equal(
        polyxios.read(out).vertex_attrs["nx"], [1.0, 2.0, 3.0]
    )


def test_unmoved_points_keep_their_normals_without_a_warning(tmp_path):
    poly = make_polydata(_VERTS, [], vertex_attrs={"nx": np.zeros(4)})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    out = tmp_path / "c.vtp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_points(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["nx"], np.zeros(4))


def test_text_features_warn_that_they_are_not_written(tmp_path):
    out = tmp_path / "c.vtp"
    features = {"label": np.array(["a", "b", "c", "d"]), "w": np.arange(4.0)}
    with pytest.warns(UserWarning, match="features label are not one number per point"):
        write_points(str(out), _VERTS, {"features": features})
    back = polyxios.read(out)
    assert "label" not in back.vertex_attrs
    np.testing.assert_array_equal(back.vertex_attrs["w"], np.arange(4.0))


def test_gltf_write_returns_its_buffer_file(tmp_path):
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.gltf"
    written = write_surface(str(out), data, kwargs)
    assert written == [str(out), str(tmp_path / "m.bin")]
    assert all(Path(p).is_file() for p in written)
    assert write_surface(str(tmp_path / "m.glb"), data, kwargs) == [
        str(tmp_path / "m.glb")
    ]


def test_stale_gltf_buffer_is_not_returned(tmp_path, monkeypatch):
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.gltf"
    sidecar = tmp_path / "m.bin"
    write_surface(str(out), data, kwargs)
    old = sidecar.read_bytes()

    def json_only(poly, path, **opts):
        Path(path).write_text("{}")

    monkeypatch.setattr(polyxios, "write", json_only)
    assert write_surface(str(out), data, kwargs) == [str(out)]
    # Not the save's file, but not the save's to delete either.
    assert sidecar.read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.bin", "m.gltf"]


def test_failed_gltf_save_leaves_the_old_buffer(tmp_path, monkeypatch):
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.gltf"
    sidecar = tmp_path / "m.bin"
    write_surface(str(out), data, kwargs)
    old = sidecar.read_bytes()

    def half_write(poly, path, **opts):
        sidecar.write_bytes(b"partial")
        raise CodecError("disk full")

    monkeypatch.setattr(polyxios, "write", half_write)
    with pytest.raises(CodecError, match="disk full"):
        write_surface(str(out), data, kwargs)
    assert sidecar.read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.bin", "m.gltf"]


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs a directory its owner cannot write to",
)
def test_gltf_save_into_a_locked_folder_names_the_target(tmp_path):
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.gltf"
    write_surface(str(out), data, kwargs)
    old = (tmp_path / "m.bin").read_bytes()
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(PermissionError) as err:
            write_surface(str(out), data, kwargs)
    finally:
        tmp_path.chmod(0o755)
    assert err.value.filename == str(out)
    assert (tmp_path / "m.bin").read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.bin", "m.gltf"]


def test_gltf_buffer_rewritten_within_the_clock_tick_is_returned(tmp_path, monkeypatch):
    # A FAT or SMB mtime can miss a rewrite of the same size.
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.gltf"
    sidecar = tmp_path / "m.bin"
    write_surface(str(out), data, kwargs)
    stamp = sidecar.stat()
    real_write = polyxios.write

    def coarse_write(*args, **opts):
        real_write(*args, **opts)
        os.utime(sidecar, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

    monkeypatch.setattr(polyxios, "write", coarse_write)
    assert write_surface(str(out), data, kwargs) == [str(out), str(sidecar)]


def test_global_attrs_are_written_back(tmp_path):
    flat = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"was_2d": True},
    )
    [(data, kwargs, _)] = layer_data(flat, name="f")
    out = tmp_path / "f.inp"
    write_surface(str(out), data, kwargs)
    assert polyxios.read(out).global_attrs.get("was_2d") is True


def test_layer_keeps_the_vertex_dtype_it_is_given():
    # A lazy read hands over vertices in the file's own dtype.
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    poly = dataclasses.replace(poly, vertices=_VERTS.astype(np.float32))
    [(data, _, _)] = layer_data(poly, name="m")
    assert data[0].dtype == np.float32


def test_time_series_values_are_written_per_vertex(tmp_path):
    [(data, kwargs, _)] = layer_data(_surface(), name="m")
    series = np.arange(12.0).reshape(3, 4)
    out = tmp_path / "m.vtp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_surface(str(out), (data[0], data[1], series), kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["temp"], series.T)


def test_napari_default_values_are_not_written(tmp_path):
    # A napari Surface always has values, all ones when it was given none.
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.vtp"
    write_surface(str(out), (*data, np.ones(4)), kwargs)
    assert "values" not in polyxios.read(out).vertex_attrs
    write_surface(str(out), (_VERTS, np.array([[0, 1, 2]]), np.ones(4)), {})
    assert "values" not in polyxios.read(out).vertex_attrs
    write_surface(str(out), (_VERTS, np.array([[0, 1, 2]]), np.ones((3, 4))), {})
    assert "values" not in polyxios.read(out).vertex_attrs


def test_values_set_on_a_layer_without_them_are_written(tmp_path):
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"values": np.ones((4, 3))},
    )
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.vtp"
    write_surface(str(out), (*data, np.arange(4.0)), kwargs)
    back = polyxios.read(out).vertex_attrs
    np.testing.assert_array_equal(back["values"], np.ones((4, 3)))
    np.testing.assert_array_equal(back["values_1"], np.arange(4.0))


def test_read_values_of_all_ones_are_written(tmp_path):
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"temp": np.ones(4)},
    )
    [(data, kwargs, _)] = layer_data(poly, name="m")
    out = tmp_path / "m.vtp"
    write_surface(str(out), data, kwargs)
    np.testing.assert_array_equal(polyxios.read(out).vertex_attrs["temp"], np.ones(4))


def test_values_that_fit_no_vertex_count_warn(tmp_path):
    poly = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    [(data, kwargs, _)] = layer_data(poly, name="m")
    with pytest.warns(UserWarning, match="values have shape"):
        write_surface(str(tmp_path / "m.vtp"), (*data, np.zeros(7)), kwargs)


def test_unknown_axis_order_is_refused():
    with pytest.raises(ValueError, match="axis_order"):
        layer_data(_surface(), name="m", axis_order="yxz")


def test_reader_refuses_foreign_extensions():
    assert get_reader(path="image.tif") is None
    assert get_reader(path=["mesh.ply", "image.tif"]) is None
    assert get_reader(path=[]) is None


_FOREIGN: dict[str, bytes] = {
    "settings.xml": b'<?xml version="1.0"?>\n<config><item>1</item></config>\n',
    "table.dat": b"1 2 3\n4 5 6\n",
    "binary.plt": b"#!TDV112" + bytes(64),
    "notes.e": b"not netCDF\n",
    "volume.vol": b"\x00\x01raw voxels",
    "drawing.svg": b'<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
    "water.xyz": b"3\nwater\nO 0.0 0.0 0.0\nH 0.76 0.59 0.0\nH -0.76 0.59 0.0\n",
    "face.pts": b"version: 1\nn_points: 2\n{\n1.0 2.0\n3.0 4.0\n}\n",
    "mesh_facet_region.xml": (
        b'<?xml version="1.0"?>\n<dolfin xmlns:dolfin="http://fenicsproject.org">\n'
        b'  <mesh_function type="uint" dim="2" size="1">\n'
        b'    <entity index="0" value="1"/>\n  </mesh_function>\n</dolfin>\n'
    ),
    "ogre.mesh.xml": (
        b'<mesh>\n  <sharedgeometry vertexcount="0"/>\n  <submeshes/>\n</mesh>\n'
    ),
    "label.xyz": b"station name\n",
}

# A gzip header with nothing readable behind it.
_BROKEN_GZIP: bytes = b"\x1f\x8b\x08\x00garbage"


@pytest.mark.parametrize("name", ["b.dat", "b.dat.gz", "b.xml", "b.e", "b.xyz"])
def test_reader_declines_a_broken_gzip_stream(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(_BROKEN_GZIP)
    assert get_reader(path=str(path)) is None


@pytest.mark.parametrize("suffix", [".xyz", ".pts", ".ptx"])
def test_reader_takes_point_clouds_under_shared_extensions(tmp_path, suffix):
    path = tmp_path / f"c{suffix}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polyxios.write(_cloud(), path)
    [(data, _, layer_type)] = get_reader(path=str(path))(str(path))
    assert layer_type == "points"
    assert len(data) == 4


@pytest.mark.parametrize("name", sorted(_FOREIGN))
def test_reader_declines_foreign_files_under_shared_extensions(tmp_path, name):
    # napari stops at the first reader that raises; declining lets it try
    # the next plugin claiming the same extension.
    path = tmp_path / name
    path.write_bytes(_FOREIGN[name])
    assert get_reader(path=str(path)) is None
    gz = tmp_path / f"{name}.gz"
    gz.write_bytes(gzip.compress(_FOREIGN[name]))
    assert get_reader(path=str(gz)) is None


@pytest.mark.parametrize(
    ("written", "renamed"),
    [
        ("m.xml", "m.xml"),
        ("m.vol", "m.vol"),
        ("m.tec", "m.plt"),
        ("m.tec", "m.dat"),
        ("m.bdf", "m.dat"),
        ("m.e", "m.e"),
    ],
)
def test_reader_takes_meshes_under_shared_extensions(tmp_path, written, renamed):
    if written.endswith(".e"):
        pytest.importorskip("netCDF4")
    tet = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    src = tmp_path / written
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polyxios.write(tet, src)
    path = tmp_path / f"in_{renamed}"
    src.rename(path)
    reader = get_reader(path=str(path))
    assert reader is not None
    [(data, _, layer_type)] = reader(str(path))
    assert layer_type == "surface"
    assert len(data[1]) == 4


def test_shared_suffixes_are_all_read():
    assert _SHARED_SUFFIXES <= set(polyxios.supported_extensions())


def test_a_dat_codec_added_to_the_registry_is_sniffed_too(tmp_path, monkeypatch):
    # The content test follows the registry: a new competitor for .dat is
    # asked by its own sniffer, here the .xyz one, which takes a bare table.
    path = tmp_path / "table.dat"
    path.write_bytes(_FOREIGN["table.dat"])
    assert get_reader(path=str(path)) is None
    dispatcher = _make_dispatcher(".dat", [("xyz", polyxios._REGISTRY[".xyz"])])
    monkeypatch.setitem(polyxios._REGISTRY, ".dat", dispatcher)
    assert get_reader(path=str(path)) is not None


def _nastran_deck(tmp_path) -> bytes:
    bdf = tmp_path / "deck.bdf"
    tet = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polyxios.write(tet, bdf)
    return bdf.read_bytes()


@pytest.mark.parametrize("banner_lines", [10, 400])
def test_a_dat_is_taken_only_when_read_would_take_it(tmp_path, banner_lines):
    # The .dat dispatcher sniffs a short window; a comment banner pushing the
    # first card past it fails polyxios.read, and napari must not be offered
    # a file its reader then raises on.
    path = tmp_path / "banner.dat"
    banner = b"$ a comment banner line, padded to some width for this test\n"
    path.write_bytes(banner * banner_lines + _nastran_deck(tmp_path))
    try:
        polyxios.read(path)
    except UnsupportedFormatError:
        reads = False
    else:
        reads = True
    assert reads == (banner_lines == 10)
    assert (get_reader(path=str(path)) is not None) == reads


def test_the_content_test_does_not_need_the_codec_imported(tmp_path, monkeypatch):
    # A codec module not yet in sys.modules still has its test asked, rather
    # than the file being taken unchecked.
    monkeypatch.delitem(sys.modules, "polyxios.codecs._exodus")
    path = tmp_path / "notes.e"
    path.write_bytes(_FOREIGN["notes.e"])
    assert get_reader(path=str(path)) is None


def _mine(head: bytes) -> bool:
    return head.startswith(b"MINE")


def test_a_dat_candidate_replaced_is_asked_by_its_own_sniffer(tmp_path, monkeypatch):
    # A codec standing in for a built-in under its label is judged by its
    # sniffer, the one the dispatcher runs, not by the built-in's.
    codec = Codec(_third_party_read, _third_party_write, _mine)
    monkeypatch.setitem(
        polyxios._REGISTRY, ".dat", _make_dispatcher(".dat", [("tecplot", codec)])
    )
    mine = tmp_path / "mine.dat"
    mine.write_bytes(b"MINE 1 2 3\n")
    assert get_reader(path=str(mine)) is not None
    tec = tmp_path / "m.tec"
    tet = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polyxios.write(tet, tec)
    theirs = tmp_path / "theirs.dat"
    tec.rename(theirs)
    assert get_reader(path=str(theirs)) is None


def test_a_raising_dat_sniffer_leaves_the_next_one_to_answer(tmp_path, monkeypatch):
    def boom(head):
        raise RuntimeError("broken sniffer")

    broken = Codec(_third_party_read, _third_party_write, boom)
    entries = [
        ("broken", broken),
        ("mine", Codec(_third_party_read, _third_party_write, _mine)),
    ]
    monkeypatch.setitem(polyxios._REGISTRY, ".dat", _make_dispatcher(".dat", entries))
    path = tmp_path / "mine.dat"
    path.write_bytes(b"MINE\n")
    assert get_reader(path=str(path)) is not None
    other = tmp_path / "other.dat"
    other.write_bytes(b"1 2 3\n")
    assert get_reader(path=str(other)) is None


def test_a_dat_no_sniffer_takes_goes_to_the_extension_owner(tmp_path, monkeypatch):
    # polyxios.read hands such a file to the owner the dispatcher falls back
    # to, so the reader must not refuse it.
    owner = Codec(_third_party_read, _third_party_write, _mine)
    dispatcher = _make_dispatcher(".dat", [("owner", owner)], ("owner", owner))
    monkeypatch.setitem(polyxios._REGISTRY, ".dat", dispatcher)
    path = tmp_path / "other.dat"
    path.write_bytes(b"1 2 3\n")
    assert polyxios.read(path) is not None
    assert get_reader(path=str(path)) is not None


def test_features_named_alike_once_spelled_as_text_are_both_written(tmp_path):
    out = tmp_path / "c.vtp"
    features = {1: np.zeros(4), "1": np.ones(4)}
    with pytest.warns(UserWarning, match="written as 1_1"):
        write_points(str(out), _VERTS, {"features": features})
    back = polyxios.read(out).vertex_attrs
    np.testing.assert_array_equal(back["1"], np.zeros(4))
    np.testing.assert_array_equal(back["1_1"], np.ones(4))


def _third_party_read(*, path, lazy=False):
    return make_polydata(_VERTS, [])


def _third_party_write(*, poly, path):
    raise NotImplementedError


def test_a_codec_installed_for_a_shared_extension_takes_its_files(
    tmp_path, monkeypatch
):
    # An entry point claiming .dat owns it, and polyxios.read hands it every
    # .dat file; the reader must not decline one the built-ins do not know.
    path = tmp_path / "table.dat"
    path.write_bytes(_FOREIGN["table.dat"])
    codec = Codec(_third_party_read, _third_party_write)
    monkeypatch.setitem(polyxios._REGISTRY, ".dat", codec)
    reader = get_reader(path=str(path))
    assert reader is not None
    [(data, _, layer_type)] = reader(str(path))
    assert layer_type == "points"
    assert len(data) == 4


def test_a_sniffer_that_raises_declines_the_file(tmp_path, monkeypatch):
    # napari stops at a reader hook that raises, so a broken sniffer must
    # decline rather than keep every other plugin from the file.
    def boom(head):
        raise RuntimeError("broken sniffer")

    monkeypatch.setattr(_tecplot, "is_ascii", boom)
    path = tmp_path / "table.dat"
    path.write_bytes(_FOREIGN["table.dat"])
    assert get_reader(path=str(path)) is None
    bdf = tmp_path / "m.bdf"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tet = make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
        polyxios.write(tet, bdf)
    dat = tmp_path / "nastran.dat"
    bdf.rename(dat)
    # The other .dat sniffers still answer.
    assert get_reader(path=str(dat)) is not None


def test_svg_is_still_write_only(tmp_path):
    # The manifest leaves .svg out of the reader patterns for this reason.
    path = tmp_path / "d.svg"
    path.write_bytes(_FOREIGN["drawing.svg"])
    with pytest.raises(polyxios.UnsupportedFormatError, match="write-only"):
        polyxios.read(path)


_PARALLEL_INDEX = """<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="1.0">
  <PUnstructuredGrid GhostLevel="0">
{entries}
  </PUnstructuredGrid>
</VTKFile>
"""
_MULTIBLOCK_INDEX = """<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0">
  <vtkMultiBlockDataSet>
{entries}
  </vtkMultiBlockDataSet>
</VTKFile>
"""


def _write_pieces(tmp_path: Path) -> list[str]:
    tri = np.array([[0, 1, 2]])
    names = []
    for i in range(2):
        name = f"piece_{i}.vtu"
        polyxios.write(
            make_polydata(_VERTS[:3] + i, [("triangle", tri)]), tmp_path / name
        )
        names.append(name)
    return names


@pytest.mark.parametrize("suffix", [".pvtu", ".vtm"])
def test_reader_opens_an_index_as_its_merged_pieces(tmp_path, suffix):
    names = _write_pieces(tmp_path)
    if suffix == ".pvtu":
        entries = "\n".join(f'    <Piece Source="{n}"/>' for n in names)
        text = _PARALLEL_INDEX.format(entries=entries)
    else:
        entries = "\n".join(
            f'    <DataSet index="{i}" file="{n}"/>' for i, n in enumerate(names)
        )
        text = _MULTIBLOCK_INDEX.format(entries=entries)
    index = tmp_path / f"case{suffix}"
    index.write_text(text)
    reader = get_reader(path=str(index))
    assert reader is not None
    [(data, kwargs, layer_type)] = reader(str(index))
    assert layer_type == "surface"
    assert kwargs["name"] == "case"
    assert len(data[0]) == 6
    assert len(data[1]) == 2


def test_reader_opens_a_multiblock_vtp(tmp_path):
    # ParaView writes a multi-block set under .vtp as well as .vtm.
    names = _write_pieces(tmp_path)
    entries = "\n".join(
        f'    <DataSet index="{i}" file="{n}"/>' for i, n in enumerate(names)
    )
    index = tmp_path / "case.vtp"
    index.write_text(_MULTIBLOCK_INDEX.format(entries=entries))
    [(data, _, layer_type)] = get_reader(path=str(index))(str(index))
    assert layer_type == "surface"
    assert len(data[0]) == 6


def test_a_vtp_no_reader_takes_keeps_the_codec_error(tmp_path):
    path = tmp_path / "grid.vtp"
    path.write_text('<?xml version="1.0"?>\n<VTKFile type="ImageData"></VTKFile>\n')
    with pytest.raises(polyxios.UnsupportedFormatError, match="ImageData"):
        get_reader(path=str(path))(str(path))


def test_a_multiblock_vtp_missing_its_pieces_says_so(tmp_path):
    # Only a real multi-block set raises past the ValueError the fallback
    # catches, and then its own error is the one worth reading.
    index = tmp_path / "case.vtp"
    entry = '    <DataSet index="0" file="gone.vtp"/>'
    index.write_text(_MULTIBLOCK_INDEX.format(entries=entry))
    with pytest.raises(FileNotFoundError, match="case.vtp"):
        get_reader(path=str(index))(str(index))


def test_reader_declines_a_gzipped_index(tmp_path):
    # Its pieces are named relative to it, and the index readers read plain
    # XML only.
    entries = '    <Piece Source="piece_0.vtu"/>'
    index = tmp_path / "case.pvtu.gz"
    index.write_bytes(gzip.compress(_PARALLEL_INDEX.format(entries=entries).encode()))
    assert get_reader(path=str(index)) is None


def test_reader_reads_plain_gzipped_and_lists(tmp_path):
    plain = tmp_path / "mesh.ply"
    packed = tmp_path / "Mesh.PLY.gz"
    polyxios.write(_surface(), plain)
    polyxios.write(_surface(), packed)

    layers = get_reader(path=[str(plain), str(packed)])([str(plain), str(packed)])
    assert [kwargs["name"] for _, kwargs, _ in layers] == ["mesh", "Mesh"]
    for data, _, _ in layers:
        np.testing.assert_allclose(data[0], _VERTS[:, ::-1])
    [(data, _, _)] = get_reader(path=str(plain))(str(plain))
    np.testing.assert_allclose(data[0], _VERTS[:, ::-1])


@pytest.mark.parametrize("axis_order", ["zyx", "xyz"])
def test_surface_round_trips_through_the_writer(tmp_path, axis_order):
    [(data, kwargs, _)] = layer_data(_surface(), name="m", axis_order=axis_order)
    out = tmp_path / "out.vtp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert write_surface(str(out), data, kwargs) == [str(out)]
    back = polyxios.read(out)
    np.testing.assert_allclose(back.vertices, _VERTS)
    np.testing.assert_array_equal(back.vertex_attrs["temp"], np.arange(4.0))


@pytest.mark.parametrize("axis_order", ["zyx", "xyz"])
def test_points_round_trip_through_the_writer(tmp_path, axis_order):
    [(data, kwargs, _)] = layer_data(_cloud(), name="c", axis_order=axis_order)
    out = tmp_path / "out.pcd"
    assert write_points(str(out), data, kwargs) == [str(out)]
    back = polyxios.read(out)
    np.testing.assert_allclose(back.vertices, _VERTS)
    np.testing.assert_array_equal(back.vertex_attrs["intensity"], [1, 2, 3, 4])
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], _cloud().vertex_attrs["colors"], atol=1 / 255
    )


def test_layer_drawn_in_napari_is_taken_as_zyx(tmp_path):
    out = tmp_path / "drawn.ply"
    write_points(str(out), _VERTS[:, ::-1], {"face_color": np.ones((4, 4))})
    back = polyxios.read(out)
    np.testing.assert_allclose(back.vertices, _VERTS)
    assert "colors" not in back.vertex_attrs


def test_2d_layer_is_padded_to_3d(tmp_path):
    out = tmp_path / "flat.ply"
    write_points(str(out), np.array([[2.0, 1.0], [4.0, 3.0]]), {})
    np.testing.assert_allclose(
        polyxios.read(out).vertices, [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
    )


def test_writer_refuses_nd_layers(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_points(str(tmp_path / "x.ply"), np.zeros((2, 4)), {})


def test_npe2_accepts_the_manifest():
    # npe2 still uses pydantic's json_encoders, deprecated since pydantic 2.0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npe2 = pytest.importorskip("npe2")
    manifest = npe2.PluginManifest.from_file(_MANIFEST)
    assert manifest.name == "polyxios"


def test_manifest_names_the_distribution_and_its_hooks():
    manifest = _manifest()
    assert manifest["name"] == "polyxios"
    for command in manifest["contributions"]["commands"]:
        module, _, attr = command["python_name"].partition(":")
        assert callable(getattr(importlib.import_module(module), attr))


def test_manifest_has_one_reader():
    # napari keys readers by plugin, so a second reader of the same patterns
    # is never offered as a choice and makes the default a coin toss.
    assert len(_manifest()["contributions"]["readers"]) == 1


class _PluginEntryPoint:
    def load(self):
        return lambda: (".plugin", polyxios._REGISTRY[".ply"])


def _built_in_extensions(monkeypatch) -> list[str]:
    # A codec installed through an entry point is not in the static manifest,
    # so the drift check compares against what polyxios ships alone.
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: [])
    return polyxios.supported_extensions(registry=build_default_registry())


def test_built_in_extensions_leave_entry_point_codecs_out(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda **_: [_PluginEntryPoint()]
    )
    assert ".plugin" in build_default_registry()
    assert ".plugin" not in _built_in_extensions(monkeypatch)


def test_manifest_reader_patterns_match_the_registry(monkeypatch):
    expected = {
        f"*{ext}{gz}"
        for ext in _built_in_extensions(monkeypatch)
        if ext not in _NOT_READ
        for gz in ("", *(() if ext in _PLAIN_ONLY else GZIP_SUFFIXES))
    }
    for reader in _manifest()["contributions"]["readers"]:
        assert set(reader["filename_patterns"]) == expected


@pytest.mark.parametrize(
    ("layer_type", "make", "write", "n_cells"),
    [
        ("surface", _surface, write_surface, 2),
        ("points", _cloud, write_points, 0),
    ],
)
def test_manifest_writer_extensions_all_write(
    tmp_path, layer_type, make, write, n_cells
):
    extensions = [
        ext
        for writer in _manifest()["contributions"]["writers"]
        if writer["layer_types"] == [layer_type]
        for ext in writer["filename_extensions"]
    ]
    assert extensions
    for ext in extensions:
        needs = _NEEDS.get(ext)
        if needs and importlib.util.find_spec(needs) is None:
            continue
        [(data, kwargs, _)] = layer_data(make(), name="m")
        out = tmp_path / f"out{ext}"
        # A format that cannot hold an attribute says so; dropping it is the
        # expected outcome here, not what this test checks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write(str(out), data, kwargs)
            back = polyxios.read(out)
        assert len(back.vertices), ext
        assert len(back.element_types) == n_cells, ext


def _hex_grid(n: int) -> polyxios.PolyData:
    conn, _, _ = structured_cells(n, n, n)
    g = np.arange(n + 1.0)
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return make_polydata(verts, [("hexahedron", conn.reshape(-1, 8))])


def test_structured_grid_skin_matches_the_generic_skin(tmp_path):
    out = tmp_path / "g.vti"
    polyxios.write(_hex_grid(3), out)
    grid = polyxios.read(out)
    assert "vti_extent" in grid.global_attrs
    [(data, _, _)] = layer_data(grid, name="g")
    loose = dataclasses.replace(grid, global_attrs={})
    [(reference, _, _)] = layer_data(loose, name="g")
    assert len(data[1]) == 6 * 9 * 2
    # Same triangles, wound the same way.
    assert {tuple(t) for t in data[1].tolist()} == {
        tuple(t) for t in reference[1].tolist()
    }


def test_structured_grid_skin_costs_its_surface_not_its_cells(tmp_path):
    # A .vti image is a grid of voxels; sorting six faces per cell to find
    # its skin took gigabytes on a 256^3 image.
    out = tmp_path / "g.vti"
    polyxios.write(_hex_grid(40), out)
    grid = polyxios.read(out)
    tracemalloc.start()
    try:
        [(data, _, _)] = layer_data(grid, name="g")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(data[1]) == 6 * 40 * 40 * 2
    # The faces of every cell alone would be 64000 * 6 * 4 int64, 12 MB.
    assert peak < 4_000_000


def _legacy_grid(path: Path, *, n: int, dataset: str) -> None:
    head = f"# vtk DataFile Version 3.0\ngrid\nASCII\nDATASET {dataset}\n"
    dims = f"DIMENSIONS {n + 1} {n + 1} {n + 1}\n"
    if dataset == "STRUCTURED_POINTS":
        body = "ORIGIN 0 0 0\nSPACING 1 1 1\n"
    else:
        g = np.arange(n + 1.0)
        zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        rows = "\n".join(" ".join(f"{v:g}" for v in p) for p in pts)
        body = f"POINTS {len(pts)} double\n{rows}\n"
    path.write_text(head + dims + body)


@pytest.mark.parametrize("dataset", ["STRUCTURED_POINTS", "STRUCTURED_GRID"])
def test_legacy_vtk_grid_skin_matches_the_generic_skin(tmp_path, dataset):
    out = tmp_path / "g.vtk"
    _legacy_grid(out, n=3, dataset=dataset)
    grid = polyxios.read(out)
    assert "vtk_dimensions" in grid.global_attrs
    assert _grid_spans(grid) == [3, 3, 3]
    [(data, _, _)] = layer_data(grid, name="g")
    loose = dataclasses.replace(grid, global_attrs={})
    [(reference, _, _)] = layer_data(loose, name="g")
    assert len(data[1]) == 6 * 9 * 2
    assert {tuple(t) for t in data[1].tolist()} == {
        tuple(t) for t in reference[1].tolist()
    }


def test_legacy_vtk_grid_with_other_cells_takes_the_generic_skin(tmp_path):
    out = tmp_path / "g.vtk"
    _legacy_grid(out, n=2, dataset="STRUCTURED_POINTS")
    grid = polyxios.read(out)
    rows = np.asarray(grid.connectivity).reshape(-1, 8)
    swapped = dataclasses.replace(grid, connectivity=rows[::-1].ravel())
    assert _grid_spans(swapped) is None


def test_legacy_vtk_grid_skin_costs_its_surface_not_its_cells(tmp_path):
    out = tmp_path / "g.vtk"
    _legacy_grid(out, n=40, dataset="STRUCTURED_POINTS")
    grid = polyxios.read(out)
    tracemalloc.start()
    try:
        [(data, _, _)] = layer_data(grid, name="g")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(data[1]) == 6 * 40 * 40 * 2
    assert peak < 4_000_000


def test_layout_globals_are_dropped_once_the_rows_moved(tmp_path):
    src = tmp_path / "grid.pcd"
    organised = make_polydata(
        _VERTS, [], global_attrs={"pcd_width": 2, "pcd_height": 2}
    )
    polyxios.write(organised, src)
    [(data, kwargs, _)] = layer_data(polyxios.read(src), name="g")
    edited = np.vstack([np.delete(data, 1, axis=0), [[9.0, 9.0, 9.0]]])
    out = tmp_path / "edited.pcd"
    with pytest.warns(UserWarning, match="pcd_width, pcd_height no longer line up"):
        write_points(str(out), edited, kwargs)
    back = polyxios.read(out)
    assert back.global_attrs.get("pcd_height", 1) == 1
    # A point deleted: the napari warning says so, and the PCD writer has no
    # grid of the wrong size left to warn about.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_points(str(out), data[:3], kwargs)
    assert [str(w.message) for w in caught if "pcd_width" in str(w.message)] == [
        "the layer has 3 vertices where the file had 4; pcd_width, pcd_height "
        "no longer line up and were not written."
    ]


def test_cloud_with_an_infinite_point_is_sized_by_its_finite_points():
    verts = np.vstack([_VERTS, [[np.inf, 0.0, 0.0]]])
    [(_, kwargs, _)] = layer_data(make_polydata(verts, []), name="c")
    assert np.isfinite(kwargs["size"])
    assert kwargs["size"] == pytest.approx(3.0 * 0.002)


def test_a_feature_named_like_the_colours_is_written_under_another_name(tmp_path):
    src = tmp_path / "c.ply"
    channels = {
        name: np.array([255, 0, 0, 9], dtype=np.uint8)
        for name in ("red", "green", "blue")
    }
    polyxios.write(make_polydata(_VERTS, [], vertex_attrs=channels), src)
    [(data, kwargs, _)] = layer_data(polyxios.read(src), name="c")
    assert "red" not in kwargs["features"]
    red = np.array([0.5, 1.5, 2.5, 3.5])
    out = tmp_path / "out.ply"
    with pytest.warns(UserWarning, match="feature red .* written as red_1"):
        write_points(str(out), data, kwargs | {"features": {"red": red}})
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs["red"], channels["red"])
    np.testing.assert_array_equal(back.vertex_attrs["red_1"], red)


def test_a_feature_named_like_a_kept_attribute_does_not_replace_it(tmp_path):
    normals = np.arange(12.0).reshape(4, 3)
    poly = make_polydata(_VERTS, [], vertex_attrs={"normals": normals})
    [(data, kwargs, _)] = layer_data(poly, name="c")
    out = tmp_path / "c.vtp"
    with pytest.warns(UserWarning, match="written as normals_1"):
        write_points(str(out), data, kwargs | {"features": {"normals": np.ones(4)}})
    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertex_attrs["normals"], normals)
    np.testing.assert_array_equal(back.vertex_attrs["normals_1"], np.ones(4))


def test_integer_values_keep_their_dtype(tmp_path):
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"label": np.array([1, 2, 3, 4], dtype=np.int32)},
    )
    [(data, kwargs, _)] = layer_data(poly, name="m")
    assert data[2].dtype == np.int32
    out = tmp_path / "m.vtp"
    write_surface(str(out), data, kwargs)
    assert polyxios.read(out).vertex_attrs["label"].dtype == np.int32


def test_the_row_keys_cost_one_word_per_vertex():
    n = 1000
    cloud = make_polydata(
        np.random.default_rng(0).random((n, 3)), [], vertex_attrs={"nx": np.ones(n)}
    )
    [(_, kwargs, _)] = layer_data(cloud, name="c")
    assert kwargs["metadata"][META_KEY]["vertex_keys"].nbytes <= 8 * n


def test_points_saved_in_fewer_dimensions_drop_what_was_kept(tmp_path):
    cloud = make_polydata(
        _VERTS, [], vertex_attrs={"w": np.arange(4.0), "t": np.ones((4, 2))}
    )
    [(data, kwargs, _)] = layer_data(cloud, name="c")
    out = tmp_path / "flat.vtp"
    with pytest.warns(UserWarning, match="changed dimension.*t no longer line up"):
        write_points(str(out), data[:, 1:], kwargs)
    assert sorted(polyxios.read(out).vertex_attrs) == ["w"]


def test_skin_is_exact_when_face_hashes_collide(monkeypatch):
    # Faces too wide to pack into one integer are sorted by a hash; two
    # different faces sharing one must still not be taken for one face.
    monkeypatch.setattr(
        polyxios.napari, "_row_hash", lambda bits: np.zeros(len(bits), np.uint64)
    )
    big = 3_000_000
    verts = np.zeros((big + 8, 3))
    hexes = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, big, big + 1, big + 2, big + 3]]
    )
    poly = make_polydata(verts, [("hexahedron", hexes)])
    [(data, _, _)] = layer_data(poly, name="h")
    # Two hexahedra sharing one face: ten outer quads, twenty triangles.
    assert len(data[1]) == 20


def test_a_feature_named_like_an_added_alpha_is_written_under_another_name(
    tmp_path,
):
    src = tmp_path / "c.ply"
    channels = {
        name: np.array([255, 0, 0, 9], dtype=np.uint8)
        for name in ("red", "green", "blue")
    }
    polyxios.write(make_polydata(_VERTS, [], vertex_attrs=channels), src)
    [(data, kwargs, _)] = layer_data(polyxios.read(src), name="c")
    see_through = np.array(kwargs["face_color"])
    see_through[:, 3] = 0.5
    weight = np.array([0.5, 1.5, 2.5, 3.5])
    out = tmp_path / "out.ply"
    with pytest.warns(UserWarning, match="written as alpha_1"):
        write_points(
            str(out),
            data,
            kwargs | {"face_color": see_through, "features": {"alpha": weight}},
        )
    back = polyxios.read(out)
    assert back.vertex_attrs["alpha"].dtype == np.uint8
    np.testing.assert_array_equal(back.vertex_attrs["alpha_1"], weight)


def test_a_surface_drops_the_structured_grid_it_was_read_from(tmp_path):
    # The skin of a .vti is triangles; an extent written beside them would
    # describe a grid the file no longer holds.
    src = tmp_path / "g.vti"
    polyxios.write(_hex_grid(2), src)
    grid = polyxios.read(src)
    assert {"vti_extent", "vti_origin", "vti_spacing"} <= set(grid.global_attrs)
    [((verts, faces), kwargs, _)] = layer_data(grid, name="g")
    out = tmp_path / "skin.vtu"
    with pytest.warns(UserWarning, match="the structured grid vti_extent"):
        write_surface(str(out), (verts, faces), kwargs)
    back = polyxios.read(out)
    assert not _GRID_GLOBALS.intersection(back.global_attrs)
    assert len(back.element_types) == len(faces)


@pytest.mark.parametrize(
    ("kind", "kept"), [("ImageData", False), ("PolyData", True), ("x", True)]
)
def test_a_surface_drops_a_vtkhdf_image_type(tmp_path, kind, kept):
    grid = dataclasses.replace(_hex_grid(2), global_attrs={"vtkhdf_type": kind})
    [((verts, faces), kwargs, _)] = layer_data(grid, name="g")
    out = tmp_path / "skin.vtu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_surface(str(out), (verts, faces), kwargs)
    back = polyxios.read(out)
    assert ("vtkhdf_type" in back.global_attrs) is kept


def test_moved_points_drop_a_vtkhdf_image_type(tmp_path):
    poly = make_polydata(
        _VERTS,
        [("line", np.array([[0, 1], [2, 3]]))],
        global_attrs={"vtkhdf_type": "ImageData", "note": "kept"},
    )
    [(data, kwargs, _)] = layer_data(poly, name="l")
    out = tmp_path / "l.vtp"
    with pytest.warns(UserWarning, match="vtkhdf_type"):
        write_points(str(out), data[:-1], kwargs)
    back = polyxios.read(out)
    assert "vtkhdf_type" not in back.global_attrs
    assert back.global_attrs["note"] == "kept"


def test_content_tests_are_keyed_by_the_registered_readers():
    readers = set()
    for codec in polyxios._REGISTRY.values():
        readers.add(codec.read)
        readers.update(entry.read for _, entry in codec.entries)
    assert set(_READ_TESTS) <= readers
    assert _READ_TESTS[polyxios._REGISTRY[".plt"].read] is _tecplot.is_ascii
    assert _READ_TESTS[polyxios._REGISTRY[".e"].read] is _exodus.is_netcdf


def test_a_full_text_window_loses_its_cut_line(tmp_path):
    path = tmp_path / "m.vol"
    path.write_bytes(b"# comment\nmesh3d\ndimension\n")
    assert _head(str(path), size=20) == b"# comment\nmesh3d"
    assert _head(str(path), size=100) == path.read_bytes()
    binary = tmp_path / "m.e"
    binary.write_bytes(b"\x89HDF\r\n\x1a\n\0\0\n\x01\x02")
    assert _head(str(binary), size=12) == binary.read_bytes()[:12]


def test_upper_case_extensions_are_read(tmp_path):
    # npe2 lower-cases the extension before matching the patterns.
    path = tmp_path / "PART.STL"
    tri = make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))])
    polyxios.write(tri, path, fmt=".stl")
    reader = get_reader(path=str(path))
    assert reader is not None
    [(data, kwargs, layer_type)] = reader(str(path))
    assert layer_type == "surface"
    assert kwargs["name"] == "PART"


def test_a_writer_missing_its_extra_names_the_install_line(tmp_path, monkeypatch):
    monkeypatch.setattr(_exodus, "_netcdf4", lambda: (None, False))
    [(data, kwargs, _)] = layer_data(_surface(), name="m")
    out = tmp_path / "m.exo"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(UnsupportedFormatError) as err:
            write_surface(str(out), data, kwargs)
    assert 'pip install "polyxios[netcdf]"' in str(err.value)
    assert "'m.exo'" in str(err.value)
    assert not out.exists()
