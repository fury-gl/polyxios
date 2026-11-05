"""ImageData, the array helpers, the image API and ``SceneImage.decode``."""

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
    ],
    ids=["planes", "rotated", "no-pixels-attr", "empty"],
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
