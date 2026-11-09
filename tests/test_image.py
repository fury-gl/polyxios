"""ImageData, the array helpers and the image API."""

import dataclasses

import numpy as np
import pytest

import polyxios
from polyxios import ImageData, helper
from polyxios._image import raster_pixels, warn_left_out
from polyxios.exceptions import CodecError, UnsupportedFormatError

_PNG = frozenset({"uint8", "uint16"})


def _gradient(h: int = 4, w: int = 5, c: int | None = 3) -> np.ndarray:
    shape = (h, w) if c is None else (h, w, c)
    return (np.arange(int(np.prod(shape))) * 7 % 256).astype(np.uint8).reshape(shape)


# ---------------------------------------------------------------------------
# Arrays <-> ImageData <-> PolyData
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 1, 2, 3, 4])
def test_an_array_survives_the_trip_through_a_mesh(channels) -> None:
    pixels = _gradient(c=channels)
    image = helper.array_to_image(pixels)
    poly = image.to_polydata()
    again = ImageData.from_polydata(poly, fmt=".png")
    assert again == image
    np.testing.assert_array_equal(helper.image_to_array(again), pixels.squeeze())
    np.testing.assert_array_equal(helper.image_to_array(poly), pixels.squeeze())


def test_the_top_row_of_the_picture_sits_at_the_highest_y() -> None:
    pixels = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(2, 3, 3)
    poly = helper.array_to_image(pixels).to_polydata()
    assert len(poly.vertices) == 6
    assert len(poly.element_types) == 2
    top = poly.vertices[:, 1] == 1
    np.testing.assert_array_equal(poly.vertex_attrs["colors"][top], pixels[0])
    np.testing.assert_array_equal(poly.vertex_attrs["colors"][~top], pixels[1])


def test_lower_origin_keeps_vtk_order() -> None:
    pixels = _gradient(2, 3, None)
    image = helper.array_to_image(pixels, origin="lower")
    np.testing.assert_array_equal(image.point_attrs["intensity"][0], pixels)
    np.testing.assert_array_equal(helper.image_to_array(image, origin="lower"), pixels)
    np.testing.assert_array_equal(helper.image_to_array(image), pixels[::-1])


def test_a_gray_image_with_alpha_is_two_attributes_and_comes_back_whole() -> None:
    pixels = _gradient(3, 3, 2)
    image = helper.array_to_image(pixels)
    assert sorted(image.point_attrs) == ["alpha", "intensity"]
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)


def test_the_array_is_copied_and_keeps_its_dtype() -> None:
    pixels = _gradient(2, 2, 3).astype(np.uint16)
    image = helper.array_to_image(pixels)
    pixels[:] = 0
    assert image.point_attrs["colors"].dtype == np.uint16
    assert image.point_attrs["colors"].any()


def test_a_volume_comes_back_as_depth_height_width() -> None:
    volume = ImageData(
        extent=(0, 2, 0, 1, 0, 3),
        point_attrs={"intensity": np.arange(24, dtype=np.float32).reshape(4, 2, 3)},
    )
    assert volume.dimensions == (3, 2, 4)
    poly = volume.to_polydata()
    assert len(poly.vertices) == 24
    assert len(poly.element_types) == 6
    assert helper.image_to_array(volume, origin="lower").shape == (4, 2, 3)
    assert ImageData.from_polydata(poly, fmt=".vti") == volume


def test_cell_attributes_and_globals_travel_both_ways() -> None:
    image = ImageData(
        extent=(0, 2, 0, 1, 0, 0),
        origin=(1.0, 2.0, 0.0),
        spacing=(0.5, 2.0, 1.0),
        point_attrs={"p": np.arange(6.0).reshape(1, 2, 3)},
        cell_attrs={"c": np.arange(2.0).reshape(1, 1, 2)},
        global_attrs={"note": "x"},
    )
    poly = image.to_polydata()
    np.testing.assert_array_equal(poly.element_attrs["c"], [0.0, 1.0])
    assert poly.global_attrs == {"note": "x"}
    poly.global_attrs["more"] = 1
    assert "more" not in image.global_attrs
    assert poly.vertices[1].tolist() == [1.5, 2.0, 0.0]
    assert ImageData.from_polydata(poly, fmt=".vti") == dataclasses.replace(
        image, global_attrs={"note": "x", "more": 1}
    )


def test_a_rotation_is_applied_to_the_lattice_and_then_the_origin_added() -> None:
    swap_xy = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])
    image = ImageData(
        extent=(0, 1, 0, 0, 0, 0), origin=(10.0, 0.0, 0.0), direction=swap_xy
    )
    assert image.to_polydata().vertices.tolist() == [[10.0, 0, 0], [10.0, 1, 0]]


def test_an_empty_axis_holds_no_points_and_no_cells() -> None:
    poly = ImageData(extent=(0, -1, 0, 2, 0, 2)).to_polydata()
    assert poly.vertices.shape == (0, 3)
    assert len(poly.element_types) == 0


def test_equality_compares_values_not_identity() -> None:
    a = helper.array_to_image(_gradient())
    b = helper.array_to_image(_gradient())
    assert a == b
    assert a != helper.array_to_image(_gradient() + 1)
    assert a != "not an image"
    rotated = ImageData(extent=a.extent, direction=np.eye(3)[::-1], point_attrs={})
    assert rotated != ImageData(extent=a.extent)
    assert ImageData(extent=(0, 1, 0, 1, 0, 0), spacing=(1.0, 1.0, 1.0)) != ImageData(
        extent=(0, 1, 0, 1, 0, 0), spacing=(2.0, 1.0, 1.0)
    )
    with pytest.raises(TypeError):
        hash(a)


def test_an_attribute_of_the_wrong_shape_is_refused_by_name() -> None:
    image = ImageData(extent=(0, 2, 0, 1, 0, 0), point_attrs={"p": np.zeros((1, 3, 2))})
    with pytest.raises(ValueError, match="point attribute 'p'"):
        image.to_polydata()
    single_point = ImageData(
        extent=(0, 0, 0, 0, 0, 0), cell_attrs={"c": np.zeros((1, 1, 1))}
    )
    with pytest.raises(ValueError, match="cell attribute 'c'"):
        single_point.to_polydata()


def test_a_mesh_that_is_not_a_lattice_is_refused() -> None:
    mesh = polyxios.make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    with pytest.raises(CodecError):
        ImageData.from_polydata(mesh, fmt=".png")


def test_the_grid_keys_a_codec_spells_do_not_become_globals() -> None:
    poly = helper.array_to_image(_gradient()).to_polydata()
    poly.global_attrs.update(vti_origin=[0, 0, 0], vtkhdf_type="ImageData", keep="me")
    assert ImageData.from_polydata(poly, fmt=".png").global_attrs == {"keep": "me"}


# ---------------------------------------------------------------------------
# helper: error contracts
# ---------------------------------------------------------------------------


def test_image_to_array_names_the_choices_when_it_cannot_pick() -> None:
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={"a": np.zeros((1, 2, 2)), "b": np.zeros((1, 2, 2))},
    )
    with pytest.raises(ValueError, match=r"name one of \['a', 'b'\]"):
        helper.image_to_array(image)
    with pytest.raises(ValueError, match=r"no point attribute 'z'.*\['a', 'b'\]"):
        helper.image_to_array(image, name="z")
    assert helper.image_to_array(image, name="a").shape == (2, 2)
    with pytest.raises(ValueError, match="no point attributes"):
        helper.image_to_array(ImageData(extent=(0, 1, 0, 1, 0, 0)))
    with pytest.raises(ValueError, match="origin"):
        helper.image_to_array(image, name="a", origin="left")


def test_the_only_attribute_is_the_default() -> None:
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0), point_attrs={"temp": np.arange(4.0).reshape(1, 2, 2)}
    )
    assert helper.image_to_array(image, origin="lower").tolist() == [[0, 1], [2, 3]]


@pytest.mark.parametrize("shape", [(4,), (2, 2, 5), (2, 2, 2, 3), (2, 2, 0)], ids=str)
def test_array_to_image_takes_only_pictures(shape) -> None:
    with pytest.raises(ValueError, match="expects"):
        helper.array_to_image(np.zeros(shape))
    with pytest.raises(ValueError, match="origin"):
        helper.array_to_image(np.zeros((2, 2)), origin="middle")


# ---------------------------------------------------------------------------
# raster_pixels: what a flat file can hold
# ---------------------------------------------------------------------------


def test_raster_pixels_hands_back_top_row_first_with_its_mode() -> None:
    pixels = _gradient(3, 4, 4)
    got, mode = raster_pixels(helper.array_to_image(pixels), fmt=".png", dtypes=_PNG)
    assert mode == "RGBA"
    np.testing.assert_array_equal(got, pixels)
    assert got.flags.c_contiguous
    gray, mode = raster_pixels(
        helper.array_to_image(pixels[..., 0]), fmt=".png", dtypes=_PNG
    )
    assert mode == "L"
    assert gray.shape == (3, 4)
    _, mode = raster_pixels(
        helper.array_to_image(pixels[..., :2]), fmt=".png", dtypes=_PNG
    )
    assert mode == "LA"


def test_a_downward_y_axis_means_the_rows_are_already_top_first() -> None:
    pixels = _gradient(3, 3, 3)
    image = helper.array_to_image(pixels, origin="lower")
    flipped = ImageData(
        extent=image.extent, spacing=(1.0, -1.0, 1.0), point_attrs=image.point_attrs
    )
    got, _ = raster_pixels(flipped, fmt=".png", dtypes=_PNG)
    np.testing.assert_array_equal(got, pixels)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: ImageData(
                extent=(0, 1, 0, 1, 0, 2),
                point_attrs={"intensity": np.zeros((3, 2, 2), np.uint8)},
            ),
            "3 planes along z",
        ),
        (
            lambda: ImageData(
                extent=(0, 1, 0, 1, 0, 0),
                direction=np.eye(3)[::-1],
                point_attrs={"intensity": np.zeros((1, 2, 2), np.uint8)},
            ),
            "rotated",
        ),
        (
            lambda: ImageData(
                extent=(0, 1, 0, 1, 0, 0),
                point_attrs={"temp": np.zeros((1, 2, 2), np.uint8)},
            ),
            r"needs a 'colors'.*\['temp'\]",
        ),
        (lambda: ImageData(extent=(0, -1, 0, 1, 0, 0)), "no pixels"),
        (
            lambda: ImageData(
                extent=(0, 1, 0, 1, 0, 0),
                point_attrs={"colors": np.zeros((2, 2, 2, 3), np.uint8)},
            ),
            r"point attribute 'colors' is shaped \(2, 2, 2, 3\)",
        ),
    ],
    ids=["planes", "rotated", "no-pixels-attr", "empty", "colors-shape"],
)
def test_raster_pixels_refuses_what_a_flat_file_cannot_hold(make, match) -> None:
    with pytest.raises(CodecError, match=match):
        raster_pixels(make(), fmt=".png", dtypes=_PNG)


def test_what_the_file_has_no_place_for_is_named_once() -> None:
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        origin=(5.0, 0.0, 0.0),
        spacing=(2.0, 1.0, 1.0),
        point_attrs={
            "intensity": np.zeros((1, 2, 2), np.uint8),
            "extra": np.zeros((1, 2, 2)),
        },
        cell_attrs={"c": np.zeros((1, 1, 1))},
    )
    with pytest.warns(UserWarning) as caught:
        raster_pixels(image, fmt=".jpg", dtypes=frozenset({"uint8"}))
    assert len(caught) == 1
    message = str(caught[0].message)
    for part in (
        "point attributes ['extra']",
        "cell attributes ['c']",
        "origin",
        "spacing",
    ):
        assert part in message
    assert message.endswith("have no place in the file and are left out.")


@pytest.mark.parametrize(
    ("values", "dtypes", "expected"),
    [
        (np.array([0.0, 0.5, 1.0]), _PNG, [0, 128, 255]),
        (np.array([0, 255, 7], dtype=np.int64), _PNG, [0, 255, 7]),
        (np.array([0, 256, 65535], dtype=np.int32), _PNG, [0, 256, 65535]),
        (np.array([0, 300], dtype=np.uint16), frozenset({"uint8"}), None),
        (np.array([0, 200], dtype=np.uint16), frozenset({"uint8"}), [0, 200]),
        (np.array([0.0, 1.5]), _PNG, None),
        (np.array([0.0, np.nan]), _PNG, None),
        (np.array([-1, 3], dtype=np.int8), _PNG, None),
        (np.array([True, False]), _PNG, None),
        (np.array([0, 70000], dtype=np.int64), _PNG, None),
    ],
    ids=[
        "float",
        "int-small",
        "int-wide",
        "u16-too-wide",
        "u16-narrow",
        "float-over",
        "nan",
        "negative",
        "bool",
        "over-u16",
    ],
)
def test_samples_are_converted_or_refused(values, dtypes, expected) -> None:
    image = ImageData(
        extent=(0, len(values) - 1, 0, 0, 0, 0),
        point_attrs={"intensity": values.reshape(1, 1, -1)},
    )
    if expected is None:
        with pytest.raises(CodecError):
            raster_pixels(image, fmt=".x", dtypes=dtypes)
        return
    got, _ = raster_pixels(image, fmt=".x", dtypes=dtypes)
    assert got.ravel().tolist() == expected


def test_global_attrs_the_file_cannot_hold_are_listed() -> None:
    with pytest.warns(UserWarning, match=r"global_attrs \['b'\] have no place"):
        warn_left_out({"a": 1, "b": 2}, fmt=".png", written=frozenset({"a"}))
    warn_left_out({"a": 1}, fmt=".png", written=frozenset({"a"}))


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


def test_write_image_takes_an_image_and_nothing_else(tmp_path) -> None:
    with pytest.raises(TypeError, match="takes an ImageData, got ndarray"):
        polyxios.write_image(np.zeros((2, 2)), tmp_path / "a.png")


def test_a_mesh_format_holds_no_image(tmp_path) -> None:
    path = tmp_path / "m.stl"
    path.write_bytes(b"")
    with pytest.raises(UnsupportedFormatError, match=r"use polyxios\.read\(\)"):
        polyxios.read_image(path)
    with pytest.raises(UnsupportedFormatError, match=r"use polyxios\.write\(\)"):
        polyxios.write_image(helper.array_to_image(_gradient()), path)


def test_read_image_is_quiet_and_read_expands_it_without_a_warning(tmp_path) -> None:
    pixels = _gradient(4, 6, 3)
    path = tmp_path / "x.vti"
    polyxios.write_image(helper.array_to_image(pixels), path)
    image = polyxios.read_image(path)  # a warning here fails the suite
    poly = polyxios.read(path)  # a grid is already the mesh form: no warning
    np.testing.assert_array_equal(
        helper.image_to_array(poly), helper.image_to_array(image)
    )
    np.testing.assert_array_equal(helper.image_to_array(image), pixels)


def test_an_image_crosses_between_formats(tmp_path) -> None:
    pytest.importorskip("h5py")
    pixels = _gradient(4, 6, 3)
    vti = tmp_path / "x.vti"
    polyxios.write_image(helper.array_to_image(pixels), vti)
    hdf = tmp_path / "x.vtkhdf"
    polyxios.write_image(polyxios.read_image(vti), hdf)
    back = polyxios.read_image(hdf)
    np.testing.assert_array_equal(helper.image_to_array(back), pixels)


def test_an_explicit_identity_is_the_same_grid_as_no_rotation() -> None:
    plain = ImageData(extent=(0, 1, 0, 1, 0, 0))
    assert ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.eye(3)) == plain
    assert ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.eye(3).ravel()) == plain
    assert plain != ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.eye(3)[::-1])


def test_extent_takes_whole_numbers_and_steps_take_three_numbers() -> None:
    image = ImageData(extent=(0.0, 2.0, 0, 1, 0, 0), origin=(1, 2, 3))
    assert image.extent == (0, 2, 0, 1, 0, 0)
    assert all(type(v) is int for v in image.extent)
    assert image.origin == (1.0, 2.0, 3.0)
    assert all(type(v) is float for v in image.origin)
    with pytest.raises(ValueError, match="not a whole number"):
        ImageData(extent=(0.0, 2.5, 0, 1, 0, 0))
    with pytest.raises(ValueError, match="not a whole number"):
        ImageData(extent=(0, np.nan, 0, 1, 0, 0))
    with pytest.raises(ValueError, match="holds 5 indices"):
        ImageData(extent=(0, 1, 0, 1, 0))
    with pytest.raises(ValueError, match="holds 2 numbers"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), spacing=(1.0, 1.0))


def test_a_single_plane_keeps_its_step_and_offset_extent_through_a_mesh() -> None:
    image = ImageData(
        extent=(2, 4, 0, 1, 0, 0),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 1.0, 4.0),
        point_attrs={"p": np.arange(6.0).reshape(1, 2, 3)},
    )
    poly = image.to_polydata()
    # What the .vti reader leaves on the mesh: the grid it was read from.
    poly.global_attrs.update(
        vti_origin=[1.0, 2.0, 3.0],
        vti_spacing=[0.5, 1.0, 4.0],
        vti_extent=[2, 4, 0, 1, 0, 0],
    )
    assert ImageData.from_polydata(poly, fmt=".vti") == image
    # Without it the grid is re-measured: zero-based, and the single plane's
    # step is the one-unit default, which the mesh never carried.
    bare = ImageData.from_polydata(image.to_polydata(), fmt=".vti")
    assert bare.extent == (0, 2, 0, 1, 0, 0)
    assert bare.origin == (2.0, 2.0, 3.0)
    assert bare.spacing == (0.5, 1.0, 1.0)


def test_a_wider_whole_extent_travels_only_with_the_grid_it_belongs_to() -> None:
    image = ImageData(
        extent=(2, 4, 0, 1, 0, 0),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 1.0, 4.0),
        point_attrs={"p": np.arange(6.0).reshape(1, 2, 3)},
        global_attrs={"vti_whole_extent": [0, 9, 0, 9, 0, 0]},
    )
    poly = image.to_polydata()
    poly.global_attrs.update(
        vti_origin=[1.0, 2.0, 3.0],
        vti_spacing=[0.5, 1.0, 4.0],
        vti_extent=[2, 4, 0, 1, 0, 0],
    )
    assert ImageData.from_polydata(poly, fmt=".vti") == image
    # Re-measured, the grid is zero-based and no longer the one the whole
    # extent described, so the whole extent goes with the stored one.
    moved = image.to_polydata()
    moved.global_attrs["vti_whole_extent"] = [0, 9, 0, 9, 0, 0]
    assert (
        "vti_whole_extent"
        not in ImageData.from_polydata(moved, fmt=".vti").global_attrs
    )


def test_a_negative_y_step_means_row_zero_is_already_the_top() -> None:
    """image_to_array and raster_pixels agree: only a positive y step flips."""
    pixels = _gradient(3, 3, 3)
    down = ImageData(
        extent=(0, 2, 0, 2, 0, 0),
        spacing=(1.0, -1.0, 1.0),
        point_attrs={"colors": pixels.reshape(1, 3, 3, 3)},
    )
    np.testing.assert_array_equal(helper.image_to_array(down), pixels)
    got, _ = raster_pixels(down, fmt=".png", dtypes=_PNG)
    np.testing.assert_array_equal(got, pixels)
    up = dataclasses.replace(down, spacing=(1.0, 1.0, 1.0))
    np.testing.assert_array_equal(helper.image_to_array(up), pixels[::-1])
    got, _ = raster_pixels(up, fmt=".png", dtypes=_PNG)
    np.testing.assert_array_equal(got, pixels[::-1])


def test_the_vtr_advice_is_kept_by_from_polydata_and_not_by_image_to_array() -> None:
    """Planes spaced 1 and 2 apart: a rectilinear grid, which has no step.

    ``from_polydata`` says to write a .vtr, which is advice for the writer;
    ``image_to_array`` only asks for a grid of pixels, so it does not repeat it.
    """
    mesh = polyxios.make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [3, 1, 0]]),
        [("quad", np.array([[0, 1, 4, 3], [1, 2, 5, 4]]))],
    )
    with pytest.raises(CodecError, match=r"\.vtr"):
        ImageData.from_polydata(mesh, fmt=".png")
    with pytest.raises(CodecError) as caught:
        helper.image_to_array(mesh)
    assert (
        str(caught.value) == "image_to_array: the mesh is not a uniform grid of pixels."
    )


def test_image_to_array_refuses_an_attribute_not_laid_out_along_the_extent() -> None:
    """A (2, 3) array on a 3 x 2 grid used to come back with its columns reversed."""
    image = ImageData(
        extent=(0, 2, 0, 1, 0, 0), point_attrs={"p": np.arange(6).reshape(2, 3)}
    )
    with pytest.raises(ValueError, match=r"point attribute 'p' is shaped \(2, 3\)"):
        helper.image_to_array(image)
    transposed = ImageData(
        extent=(0, 2, 0, 1, 0, 0), point_attrs={"p": np.zeros((1, 3, 2))}
    )
    with pytest.raises(ValueError, match=r"\(nz, ny, nx\) = \(1, 2, 3\)"):
        helper.image_to_array(transposed, name="p")
    gray = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={"intensity": np.zeros((1, 2, 2)), "alpha": np.zeros((1, 2, 3))},
    )
    with pytest.raises(ValueError, match="point attribute 'alpha'"):
        helper.image_to_array(gray)


def test_a_direction_is_stored_as_a_three_by_three_matrix() -> None:
    swap = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    flat = ImageData(extent=(0, 1, 0, 1, 0, 0), direction=swap.ravel().tolist())
    assert flat.direction.shape == (3, 3)
    assert flat.direction.dtype == np.float64
    assert flat == ImageData(extent=(0, 1, 0, 1, 0, 0), direction=swap)
    assert ImageData(extent=(0, 1, 0, 1, 0, 0), direction=None).direction is None
    with pytest.raises(ValueError, match="holds 4 numbers"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.zeros(4))
    with pytest.raises(ValueError, match="not a run of numbers"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction="abc")


@pytest.mark.parametrize("ext", [".vti", ".vtkhdf"])
def test_both_writers_drop_what_a_file_cannot_hold_by_name(tmp_path, ext) -> None:
    """A cell array on a single point, and a text array, are dropped by both
    codecs with a warning blamed on the caller, not refused by one of them."""
    if ext == ".vtkhdf":
        pytest.importorskip("h5py")
    single = ImageData(
        extent=(0, 0, 0, 0, 0, 0),
        point_attrs={"p": np.zeros((1, 1, 1))},
        cell_attrs={"c": np.zeros((1, 1, 1))},
    )
    path = tmp_path / f"single{ext}"
    with pytest.warns(UserWarning, match=r"CellData array\(s\) \['c'\]") as caught:
        polyxios.write_image(single, path)
    assert {w.filename for w in caught} == {__file__}
    assert polyxios.read_image(path).cell_attrs == {}
    text = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={
            "p": np.zeros((1, 2, 2)),
            "label": np.array(list("abcd")).reshape(1, 2, 2),
        },
    )
    path = tmp_path / f"text{ext}"
    with pytest.warns(UserWarning, match=r"PointData array\(s\) \['label'\]") as caught:
        polyxios.write_image(text, path)
    assert {w.filename for w in caught} == {__file__}
    assert sorted(polyxios.read_image(path).point_attrs) == ["p"]


def test_a_cell_array_on_a_single_point_says_it_holds_no_cell() -> None:
    image = ImageData(extent=(0, 0, 0, 0, 0, 0), cell_attrs={"c": np.zeros((1, 1, 1))})
    with pytest.raises(ValueError, match="this extent holds no cell"):
        image.to_polydata()


@pytest.mark.parametrize("ext", [".vti", ".vtkhdf"])
def test_an_unrecognised_image_option_is_blamed_on_the_caller(tmp_path, ext) -> None:
    if ext == ".vtkhdf":
        pytest.importorskip("h5py")
    path = tmp_path / f"o{ext}"
    with pytest.warns(UserWarning, match="unrecognized options") as written:
        polyxios.write_image(helper.array_to_image(_gradient()), path, bogus=1)
    assert {w.filename for w in written} == {__file__}
    with pytest.warns(UserWarning, match="unrecognized options") as read_back:
        polyxios.read_image(path, bogus=1)
    assert {w.filename for w in read_back} == {__file__}


@pytest.mark.parametrize(
    ("origin", "match"),
    [
        (("a", "b", "c"), "origin is not a run of numbers"),
        (1.0, "origin is not a run of numbers"),
        ((0.0, 1.0), "origin holds 2 numbers"),
    ],
)
def test_a_step_that_is_no_number_is_a_value_error(origin, match) -> None:
    with pytest.raises(ValueError, match=match):
        ImageData(extent=(0, 1, 0, 1, 0, 0), origin=origin)


@pytest.mark.parametrize(
    ("extent", "match"),
    [
        (("a", 0, 0, 0, 0, 0), "not a number"),
        (5, "not a run of six numbers"),
    ],
)
def test_an_extent_that_is_no_number_is_a_value_error(extent, match) -> None:
    with pytest.raises(ValueError, match=match):
        ImageData(extent=extent)


def test_from_polydata_names_itself_in_its_default_messages() -> None:
    """A mesh no lattice describes: the message leads with the call, not a format."""
    mesh = polyxios.make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [3, 1, 0]]),
        [("quad", np.array([[0, 1, 4, 3], [1, 2, 5, 4]]))],
    )
    with pytest.raises(CodecError, match=r"^ImageData\.from_polydata: "):
        ImageData.from_polydata(mesh)


def test_point_and_cell_arrays_compare_by_value_whatever_they_were_built_from() -> None:
    listed = ImageData(extent=(0, 0, 0, 0, 0, 0), point_attrs={"p": [[[1, 2]]]})
    array = ImageData(
        extent=(0, 0, 0, 0, 0, 0), point_attrs={"p": np.array([[[1, 2]]])}
    )
    assert listed == array
    assert isinstance(listed.point_attrs["p"], np.ndarray)


def test_to_polydata_hands_back_views_of_the_image_arrays() -> None:
    """The Returns section says the arrays are views; this holds it to that."""
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0), point_attrs={"p": np.arange(4.0).reshape(1, 2, 2)}
    )
    poly = image.to_polydata()
    assert np.shares_memory(poly.vertex_attrs["p"], image.point_attrs["p"])


@pytest.mark.parametrize("ext", [".vti", ".vtkhdf"])
def test_both_image_writers_keep_the_same_keys_off_the_field_data(
    tmp_path, ext
) -> None:
    """One reserved set for image writes: the grid and structure keys never come
    back as field data from either codec, whichever it is."""
    if ext == ".vtkhdf":
        pytest.importorskip("h5py")
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={"p": np.zeros((1, 2, 2))},
        global_attrs={
            "vti_origin": [9.0, 9.0, 9.0],
            "vtkhdf_type": "ImageData",
            "vtkhdf_direction": np.eye(3),
            "keep": 1.0,
        },
    )
    path = tmp_path / f"keys{ext}"
    polyxios.write_image(image, path)
    assert set(polyxios.read_image(path).global_attrs) == {"keep"}


@pytest.mark.parametrize("ext", [".vti", ".vtkhdf"])
def test_an_array_with_no_component_is_dropped_by_name(tmp_path, ext) -> None:
    """A zero-width array holds no value per point, so both writers drop it with a
    warning naming it, blamed on the caller, rather than silently."""
    if ext == ".vtkhdf":
        pytest.importorskip("h5py")
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        point_attrs={"p": np.zeros((1, 2, 2)), "none": np.zeros((1, 2, 2, 0))},
    )
    path = tmp_path / f"zero{ext}"
    with pytest.warns(UserWarning, match=r"PointData array\(s\) \['none'\]") as caught:
        polyxios.write_image(image, path)
    assert {w.filename for w in caught} == {__file__}
    assert sorted(polyxios.read_image(path).point_attrs) == ["p"]


def test_a_grid_helper_names_its_format_and_blames_the_caller() -> None:
    """A bad origin in the mesh is reported with the call that found it, not as the
    .vti codec's message, and blamed on this file."""
    image = ImageData(extent=(0, 1, 0, 1, 0, 0), point_attrs={"p": np.zeros((1, 2, 2))})
    poly = image.to_polydata()
    poly.global_attrs["vti_origin"] = [0.0, 1.0]
    with pytest.warns(UserWarning, match="vti_origin spells 2 numbers") as caught:
        ImageData.from_polydata(poly, fmt=".png")
    assert [str(w.message).split(":")[0] for w in caught] == [".png"]
    assert {w.filename for w in caught} == {__file__}
    with pytest.warns(
        UserWarning, match="ImageData.from_polydata: vti_origin"
    ) as caught:
        helper.image_to_array(poly)
    assert {w.filename for w in caught} == {__file__}


def test_a_rotated_vtkhdf_image_is_refused_by_both_helpers_with_read_image(
    tmp_path,
) -> None:
    """A mesh holds no rotation to give back, so the refusal names read_image."""
    pytest.importorskip("h5py")
    swap = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    path = tmp_path / "rot.vtkhdf"
    polyxios.write_image(
        ImageData(
            extent=(0, 1, 0, 1, 0, 0),
            direction=swap,
            point_attrs={"p": np.zeros((1, 2, 2))},
        ),
        path,
    )
    mesh = polyxios.read(path)
    with pytest.raises(CodecError, match="polyxios.read_image"):
        helper.image_to_array(mesh)
    with pytest.raises(CodecError, match="read the file with polyxios.read_image"):
        ImageData.from_polydata(mesh)


def test_a_slice_written_with_lower_origin_comes_back_with_its_rows() -> None:
    """What image_to_array(origin='lower') gives out, array_to_image(origin='lower')
    takes back in: the rows keep their order, which the default 'upper' would flip."""
    plane = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float64)
    image = helper.array_to_image(plane, origin="lower")
    np.testing.assert_array_equal(helper.image_to_array(image, origin="lower"), plane)
    flipped = helper.array_to_image(plane)
    np.testing.assert_array_equal(helper.image_to_array(flipped), plane)
    # Read back in the other order, the same image is upside down: the mismatch the
    # usage example used to make.
    np.testing.assert_array_equal(
        helper.image_to_array(flipped, origin="lower"), plane[::-1]
    )


@pytest.mark.parametrize("origin", ["upper", "lower"])
def test_image_to_array_always_hands_back_a_copy(origin) -> None:
    """Whether or not the rows were flipped, writing into the result leaves the image
    as it was."""
    image = helper.array_to_image(_gradient(3, 4, 3), origin="lower")
    before = image.point_attrs["colors"].copy()
    got = helper.image_to_array(image, origin=origin)
    assert not np.shares_memory(got, image.point_attrs["colors"])
    got[...] = 0
    np.testing.assert_array_equal(image.point_attrs["colors"], before)


@pytest.mark.parametrize("channels", [None, 1, 2, 3, 4])
@pytest.mark.parametrize("origin", ["upper", "lower"])
def test_array_to_image_takes_its_own_copy_of_every_channel_layout(
    channels, origin
) -> None:
    """Changing the array after the call never reaches the image, whatever its
    channel count and origin."""
    pixels = _gradient(3, 4, channels)
    expected = pixels.copy()
    image = helper.array_to_image(pixels, origin=origin)
    pixels[...] = 0
    back = helper.image_to_array(image, origin=origin)
    np.testing.assert_array_equal(back, expected.reshape(back.shape))


def test_missing_global_attrs_is_no_metadata_and_a_non_mapping_is_refused() -> None:
    """As PolyData allows, an ImageData with no global_attrs holds none; every writer
    and conversion reads it as empty, and a value that is not a mapping is refused
    where it is given, not later in a writer."""
    image = ImageData(extent=(0, 1, 0, 1, 0, 0), global_attrs=None)
    assert image.global_attrs == {}
    poly = image.to_polydata()
    assert poly.global_attrs == {}
    assert ImageData(extent=(0, 1, 0, 1, 0, 0), global_attrs=None) == image
    with pytest.raises(ValueError, match="global_attrs is a list, not a mapping"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), global_attrs=[("a", 1)])


def test_image_to_array_reads_a_mesh_with_no_metadata_map() -> None:
    """A mesh whose global_attrs is None has no rotation to refuse; its pixels come
    back as they would from one with an empty map."""
    pixels = _gradient(2, 3, 3)
    mesh = helper.array_to_image(pixels).to_polydata()
    mesh = dataclasses.replace(mesh, global_attrs=None)
    np.testing.assert_array_equal(helper.image_to_array(mesh), pixels)


@pytest.mark.parametrize("attr", ["point_attrs", "cell_attrs"])
def test_attribute_maps_take_none_as_empty_and_refuse_a_non_mapping(attr) -> None:
    """point_attrs and cell_attrs follow global_attrs: None holds nothing, and a value
    that is not a mapping is refused where it is given, not in a writer."""
    extent = (0, 1, 0, 1, 0, 0)
    assert getattr(ImageData(extent=extent, **{attr: None}), attr) == {}
    with pytest.raises(ValueError, match=f"{attr} is a list, not a mapping"):
        ImageData(extent=extent, **{attr: [1, 2]})


def test_from_polydata_names_a_point_attribute_the_grid_cannot_hold() -> None:
    poly = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    poly.vertex_attrs["bad"] = np.zeros(5)
    with pytest.raises(
        CodecError,
        match=r"point attribute 'bad' holds 5 rows; the grid has 6 points",
    ):
        ImageData.from_polydata(poly)


def test_from_polydata_names_a_cell_attribute_the_grid_cannot_hold() -> None:
    poly = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    n_cells = len(poly.element_types)
    poly.element_attrs["bad"] = np.zeros(n_cells + 1)
    with pytest.raises(
        CodecError,
        match=rf"cell attribute 'bad' holds {n_cells + 1} rows; the grid has {n_cells} cells",
    ):
        ImageData.from_polydata(poly)


def test_image_to_array_names_an_unrelated_attribute_it_cannot_lay_out() -> None:
    """The attribute asked for is sound; another the grid cannot hold is still named,
    as the mesh cannot be read as pixels with it in it."""
    poly = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    poly.vertex_attrs["bad"] = np.zeros(5)
    with pytest.raises(
        CodecError, match="image_to_array: point attribute 'bad' holds 5 rows"
    ):
        helper.image_to_array(poly, name="intensity")


def _one_point_mesh():
    """A mesh of one point and no cell, as a grid of no cells expands to."""
    return ImageData(
        extent=(0, 0, 0, 0, 0, 0), point_attrs={"s": np.ones((1, 1, 1))}
    ).to_polydata()


def test_a_cell_array_with_rows_is_refused_on_a_mesh_with_no_cell() -> None:
    """A mesh with no cell holds no cell array: one that holds rows is refused by
    name, and one that holds none is simply not kept."""
    point = _one_point_mesh()
    bad = dataclasses.replace(point, element_attrs={"bad": np.zeros(3)})
    with pytest.raises(
        CodecError, match="cell attribute 'bad' holds 3 rows; the grid has 0 cells"
    ):
        ImageData.from_polydata(bad)
    with pytest.raises(
        CodecError, match="image_to_array: cell attribute 'bad' holds 3 rows"
    ):
        helper.image_to_array(bad)
    empty = dataclasses.replace(point, element_attrs={"empty": np.zeros(0)})
    assert ImageData.from_polydata(empty).cell_attrs == {}


def test_from_polydata_warns_on_a_cell_group_that_names_no_cell() -> None:
    """A tag group naming a cell the mesh does not hold is dropped with the warning
    a mesh with cells gives, whatever the cell count."""
    mesh = dataclasses.replace(_one_point_mesh(), element_tags={"g": np.array([0])})
    with pytest.warns(
        UserWarning, match=r"ImageData.from_polydata read: cell tag group"
    ):
        image = ImageData.from_polydata(mesh)
    assert image.cell_attrs == {}


def test_from_polydata_and_image_to_array_refuse_what_is_not_a_mesh_or_image() -> None:
    with pytest.raises(TypeError, match="takes a PolyData, got NoneType"):
        ImageData.from_polydata(None)
    with pytest.raises(
        TypeError,
        match=r"image_to_array\(\) takes an ImageData or a PolyData, got NoneType",
    ):
        helper.image_to_array(None)


def test_array_to_image_hands_global_attrs_to_imagedata_without_an_alias() -> None:
    """The metadata is validated by ImageData, so a bad value is ImageData's own
    refusal, and the image holds a copy: changing the caller's dict does not reach
    it."""
    with pytest.raises(ValueError, match="ImageData: global_attrs is a list"):
        helper.array_to_image(np.ones((2, 2)), global_attrs=[1])
    attrs = {"k": 1.0}
    image = helper.array_to_image(np.ones((2, 2)), global_attrs=attrs)
    attrs["k"] = 2.0
    assert image.global_attrs == {"k": 1.0}


def test_an_extent_beyond_int64_is_refused_where_it_is_given() -> None:
    """A file spells each extent value as an int64, so an ImageData refuses a value
    outside that range when it is made, not in a writer that overflows on it; the
    limits themselves are kept exactly."""
    with pytest.raises(ValueError, match="ImageData: extent .* outside the 64-bit"):
        ImageData(extent=(0, 2**63, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="outside the 64-bit"):
        ImageData(extent=(-(2**63) - 1, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="outside the 64-bit"):
        ImageData(extent=(0, 1e30, 0, 0, 0, 0))
    assert ImageData(extent=(0, 2**63 - 1, 0, 0, 0, 0)).extent[1] == 2**63 - 1
    assert ImageData(extent=(-(2**63), 0, 0, 0, 0, 0)).extent[0] == -(2**63)


def test_from_polydata_names_a_shadowed_attribute_as_kept_not_written() -> None:
    """Reading a mesh into an image writes nothing, so the warning for a group that
    takes a cell attribute's name says the group is kept."""
    poly = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    n_cells = len(poly.element_types)
    mesh = dataclasses.replace(
        poly,
        element_tags={"g": np.array([0])},
        element_attrs={"polyxios_tag_g": np.zeros(n_cells)},
    )
    with pytest.warns(UserWarning) as caught:
        ImageData.from_polydata(mesh)
    [message] = [str(w.message) for w in caught if "polyxios_tag_g" in str(w.message)]
    assert message.startswith("ImageData.from_polydata read: cell attribute(s)")
    assert message.endswith("the group is kept and the attribute is dropped.")


@pytest.mark.parametrize("attr", ["point_attrs", "cell_attrs"])
def test_a_ragged_attribute_is_refused_by_name(attr) -> None:
    """A value NumPy cannot make an array of is refused as an ImageData error naming the
    attribute, not with NumPy's own message about an inhomogeneous shape."""
    with pytest.raises(ValueError, match=f"ImageData: {attr} 'a' is not an array"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), **{attr: {"a": [[1], [2, 3]]}})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("extent", "012345", "extent is not a run of six numbers"),
        ("extent", ("0", "3", 0, 0, 0, 0), "holds '0', which is text"),
        ("origin", "123", "origin is not a run of numbers"),
        ("origin", ("1", 2.0, 3.0), "'1' is text"),
        ("spacing", "111", "spacing is not a run of numbers"),
        ("spacing", (1.0, b"1", 1.0), "b'1' is text"),
    ],
)
def test_text_is_refused_for_the_extent_origin_and_spacing(field, value, match) -> None:
    """A string parses as a number and iterates as characters, so text is refused as a
    whole and as an item, before anything is read as a number."""
    kwargs = {"extent": (0, 1, 0, 1, 0, 0)} | {field: value}
    with pytest.raises(ValueError, match=match):
        ImageData(**kwargs)


def test_a_mesh_read_past_int64_is_a_codec_error_from_both_entry_points(
    tmp_path,
) -> None:
    """A .vti whose extent a 64-bit integer cannot hold reads as a mesh that keeps the
    extent; turning that mesh back into a grid is refused as a CodecError, not a bare
    ValueError, by both ImageData.from_polydata and helper.image_to_array."""
    huge = "9223372036854775808 9223372036854775808 0 0 0 0"
    path = tmp_path / "huge.vti"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{huge}" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="{huge}">\n    </Piece>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    mesh = polyxios.read(path)
    with pytest.raises(CodecError, match=r"^ImageData\.from_polydata: .*64-bit"):
        ImageData.from_polydata(mesh)
    with pytest.raises(CodecError, match="not a uniform grid of pixels"):
        helper.image_to_array(mesh)


@pytest.mark.parametrize(
    ("direction", "match"),
    [
        (["0", "-1", "0", "1", "0", "0", "0", "0", "1"], "text is not a number"),
        ("012345678", "text is not a number"),
        (np.eye(3) * 1j, "complex numbers"),
        ([1, 0, 0, 0, 1, 0, 0, 0, None], "is not a number"),
        ([None] * 9, "is not a number"),
        (
            np.array(["0", "1", "0", "1", "0", "0", "0", "0", "1"], dtype=object),
            "'0' is text",
        ),
    ],
)
def test_a_direction_that_is_text_or_complex_is_a_value_error(direction, match) -> None:
    """A string parses as a number and a complex value would lose its imaginary part, so
    neither is a rotation; both are refused where the direction is given."""

    with pytest.raises(ValueError, match=match):
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction=direction)


def test_a_nan_direction_is_accepted_as_origin_and_spacing_accept_nan() -> None:
    """Origin and spacing take NaN and inf, since float() does; direction takes NaN in
    the same way, so the three fields follow one rule."""
    image = ImageData(
        extent=(0, 1, 0, 1, 0, 0),
        origin=(np.nan, 0.0, 0.0),
        spacing=(np.inf, 1.0, 1.0),
        direction=[1, 0, 0, 0, 1, 0, 0, 0, np.nan],
    )
    assert np.isnan(image.origin[0])
    assert np.isnan(image.direction[2, 2])


@pytest.mark.parametrize("stored", ["identity", [1.0, 0.0, 0.0]])
def test_a_stored_direction_that_is_no_rotation_is_a_codec_error(stored) -> None:
    """A mesh carrying a vtkhdf_direction that is text or not nine numbers is refused
    as a CodecError naming the caller, from both entry points."""
    mesh = helper.array_to_image(np.zeros((2, 2), np.uint8)).to_polydata()
    mesh.global_attrs["vtkhdf_direction"] = stored
    with pytest.raises(CodecError, match=r"^ImageData\.from_polydata: global"):
        ImageData.from_polydata(mesh)
    with pytest.raises(CodecError, match=r"^image_to_array: global"):
        helper.image_to_array(mesh)


def test_array_to_image_names_ragged_pixels() -> None:
    """Pixels NumPy cannot make an array of are refused by name, not with NumPy's
    message about an inhomogeneous shape."""
    with pytest.raises(ValueError, match="array_to_image: pixels is not an array"):
        helper.array_to_image([[1, 2], [3]])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extent", (0, np.complex128(1 + 5j), 0, 1, 0, 0)),
        ("extent", np.array([0, 1, 0, 1, 0, 0], dtype=complex)),
        ("origin", np.array([1 + 2j, 0, 0])),
        ("spacing", (1 + 0j, 1.0, 1.0)),
    ],
)
def test_a_complex_extent_origin_or_spacing_is_a_value_error(field, value) -> None:
    """A complex number is no index, coordinate or step: its imaginary part would be
    dropped without a word, so each field refuses it where it is given."""
    kwargs = {"extent": (0, 1, 0, 1, 0, 0)} | {field: value}
    with pytest.raises(ValueError, match="which is complex, not a real number"):
        ImageData(**kwargs)


@pytest.mark.parametrize("value", [np.float64(1.0), 1.0])
def test_a_point_attribute_with_no_rows_is_a_codec_error(value) -> None:
    """A scalar point attribute has no row to be one per point: it is refused by name
    from both entry points, not with a bare TypeError from len()."""
    mesh = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    mesh.vertex_attrs["a"] = value
    with pytest.raises(
        CodecError, match="point attribute 'a' holds no rows; the grid has 6 points"
    ):
        ImageData.from_polydata(mesh)
    with pytest.raises(CodecError, match="image_to_array: point attribute 'a' holds"):
        helper.image_to_array(mesh)


@pytest.mark.parametrize("value", [np.float64(1.0), 1.0])
def test_a_cell_attribute_with_no_rows_is_a_codec_error(value) -> None:
    mesh = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    mesh.element_attrs["c"] = value
    with pytest.raises(
        CodecError, match=r"cell attribute 'c' holds no rows; the grid has \d+ cells"
    ):
        ImageData.from_polydata(mesh)
    with pytest.raises(CodecError, match="image_to_array: cell attribute 'c' holds"):
        helper.image_to_array(mesh)


def test_a_mesh_attribute_given_as_a_list_is_read_as_an_array() -> None:
    """PolyData keeps what it is given, so a list of one value per point is still a
    per-point attribute, read as an array."""
    mesh = helper.array_to_image(np.zeros((2, 3), np.uint8)).to_polydata()
    mesh.vertex_attrs["a"] = list(range(6))
    image = ImageData.from_polydata(mesh)
    np.testing.assert_array_equal(image.point_attrs["a"].ravel(), np.arange(6))
