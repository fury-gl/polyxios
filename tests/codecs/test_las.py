from __future__ import annotations

import io
import struct
import tracemalloc
import types
import warnings

import numpy as np
import pytest

import polyxios
from polyxios._optpkg import TripWire
from polyxios._types import PolyData, make_polydata
from polyxios.codecs import _las
from polyxios.codecs._pointcloud import point_cloud
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    MissingPackageError,
    ValidationError,
)
from tests.codecs._lazy import mapped

_HEADER_1_2 = 227
_HEADER_1_4 = 375


def _cloud(n: int = 4, **extra: np.ndarray) -> PolyData:
    rng = np.random.default_rng(1)
    verts = rng.uniform(-10, 10, (n, 3)).round(3)
    attrs: dict[str, np.ndarray] = {
        "intensity": np.arange(n, dtype=np.uint16) * 100,
        "return_number": np.resize([1, 2, 1, 3], n).astype(np.uint8),
        "number_of_returns": np.resize([2, 2, 1, 3], n).astype(np.uint8),
        "classification": np.resize([2, 5, 6, 31], n).astype(np.uint8),
        "gps_time": np.linspace(1e5, 1e5 + 1, n),
        "colors": np.resize(np.eye(3), (n, 3)),
    }
    attrs.update(extra)
    return make_polydata(verts, [("vertex", np.arange(n)[:, None])], vertex_attrs=attrs)


def _write(poly: PolyData, **opts) -> bytes:
    buf = io.BytesIO()
    polyxios.write(poly, buf, fmt=".las", **opts)
    return buf.getvalue()


def _read(data: bytes, **opts) -> PolyData:
    return polyxios.read(io.BytesIO(data), fmt=".las", **opts)


def _u16(data: bytes, at: int) -> int:
    return struct.unpack_from("<H", data, at)[0]


def _u32(data: bytes, at: int) -> int:
    return struct.unpack_from("<I", data, at)[0]


def _patched(data: bytes, fmt: str, at: int, value: int) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, at, value)
    return bytes(out)


# ---------------------------------------------------------------------------
# Header and layout
# ---------------------------------------------------------------------------


def test_the_default_header_is_las_1_2_point_format_3_for_a_coloured_timed_cloud():
    data = _write(_cloud(), scale=0.001)
    assert data[:4] == b"LASF"
    assert (data[24], data[25]) == (1, 2)
    assert _u16(data, 94) == _HEADER_1_2
    assert _u32(data, 96) == _HEADER_1_2
    assert data[104] == 3
    assert _u16(data, 105) == 34
    assert _u32(data, 107) == 4
    assert len(data) == _HEADER_1_2 + 4 * 34


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({}, 0),
        ({"gps_time"}, 1),
        ({"colors"}, 2),
        ({"gps_time", "colors"}, 3),
        ({"gps_time", "nir"}, 8),
        ({"colors", "nir"}, 8),
        ({"scan_angle"}, 6),
        ({"scan_angle", "colors"}, 7),
        ({"scanner_channel", "gps_time"}, 6),
        ({"wavepacket_size", "gps_time"}, 4),
        ({"wavepacket_size", "colors"}, 5),
        ({"wavepacket_size", "scan_angle"}, 9),
        ({"wavepacket_size", "nir"}, 10),
    ],
)
def test_the_point_format_follows_the_attributes(attrs, expected) -> None:
    base = _cloud()
    keep = {
        k: v
        for k, v in base.vertex_attrs.items()
        if k not in {"gps_time", "colors"} or k in attrs
    }
    n = base.vertices.shape[0]
    for name in attrs:
        if name in ("gps_time", "colors"):
            continue
        keep[name] = np.arange(n, dtype=np.uint16)
    poly = make_polydata(
        base.vertices, [("vertex", np.arange(n)[:, None])], vertex_attrs=keep
    )
    data = _write(poly)
    assert data[104] == expected
    assert _u16(data, 105) == _las._RECORD_LENGTHS[expected]


def test_a_new_style_point_format_writes_a_1_4_header() -> None:
    data = _write(_cloud(), point_format=7)
    assert (data[24], data[25]) == (1, 4)
    assert _u16(data, 94) == _HEADER_1_4
    assert struct.unpack_from("<Q", data, 247)[0] == 4
    # Legacy counts have to be zero for the new formats.
    assert _u32(data, 107) == 0


def test_a_legacy_point_format_in_a_1_4_file_fills_the_legacy_counts() -> None:
    data = _write(_cloud(), version="1.4")
    assert data[104] == 3
    assert _u32(data, 107) == 4
    assert struct.unpack_from("<Q", data, 247)[0] == 4
    back = _read(data)
    assert back.global_attrs["las_version"] == "1.4"
    assert back.vertices.shape == (4, 3)


def test_points_by_return_are_counted() -> None:
    data = _write(_cloud())
    by_return = struct.unpack_from("<5I", data, 111)
    assert by_return == (2, 1, 1, 0, 0)
    data = _write(_cloud(), point_format=7)
    assert struct.unpack_from("<15Q", data, 255)[:3] == (2, 1, 1)


def test_the_records_are_written_from_the_array_without_a_copy(tmp_path) -> None:
    n = 200_000
    poly = point_cloud(
        vertices=np.arange(3 * n, dtype=np.int32).reshape(n, 3),
        vertex_attrs={},
        global_attrs={"las_scale": [0.001] * 3, "las_offset": [0.0] * 3},
    )
    record_bytes = n * _las._RECORD_LENGTHS[0]
    tracemalloc.start()
    try:
        polyxios.write(poly, tmp_path / "c.las", point_format=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * record_bytes
    assert (tmp_path / "c.las").stat().st_size == _HEADER_1_2 + record_bytes


def test_the_bounding_box_is_the_written_coordinates() -> None:
    poly = _cloud()
    data = _write(poly, scale=0.001)
    back = _read(data)
    box = struct.unpack_from("<6d", data, 179)
    assert box[0] == back.vertices[:, 0].max()
    assert box[1] == back.vertices[:, 0].min()
    assert box[5] == back.vertices[:, 2].min()


def test_scale_and_offset_come_back_and_round_trip_the_coordinates() -> None:
    poly = _cloud()
    data = _write(poly, scale=(0.001, 0.001, 0.0001), offset=(1.0, 2.0, 3.0))
    back = _read(data)
    np.testing.assert_allclose(back.vertices, poly.vertices, atol=1e-9)
    np.testing.assert_array_equal(back.global_attrs["las_scale"], [0.001, 0.001, 1e-4])
    np.testing.assert_array_equal(back.global_attrs["las_offset"], [1.0, 2.0, 3.0])
    again = _write(back)
    assert again[131:179] == data[131:179]


def test_a_scalar_scale_applies_to_every_axis() -> None:
    back = _read(_write(_cloud(), scale=0.01))
    np.testing.assert_array_equal(back.global_attrs["las_scale"], [0.01] * 3)


def test_the_default_offset_is_zero_when_the_cloud_fits_and_a_corner_when_not():
    data = _write(_cloud(), scale=0.001)
    assert struct.unpack_from("<3d", data, 155) == (0.0, 0.0, 0.0)
    far = make_polydata(
        np.array([[5e6, 5e6, 0.0], [5e6 + 1, 5e6 + 1, 1.0]]),
        [("vertex", np.array([[0], [1]]))],
    )
    data = _write(far, scale=0.001)
    assert struct.unpack_from("<3d", data, 155) == (5e6, 5e6, 0.0)
    np.testing.assert_allclose(_read(data).vertices, far.vertices, atol=1e-9)


def test_a_coordinate_no_scale_can_hold_is_refused() -> None:
    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1e10, 0.0, 0.0]]),
        [("vertex", np.array([[0], [1]]))],
    )
    with pytest.raises(CodecError, match="int32"):
        _write(poly, scale=0.001, offset=0.0)


def test_a_nan_coordinate_is_refused() -> None:
    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
        [("vertex", np.array([[0], [1]]))],
    )
    with pytest.raises(CodecError, match="NaN"):
        _write(poly)


def test_a_zero_or_negative_scale_is_refused_on_write() -> None:
    with pytest.raises(CodecError, match="scale"):
        _write(_cloud(), scale=0.0)
    with pytest.raises(CodecError, match="scale"):
        _write(_cloud(), scale=(0.001, -0.001, 0.001))


def test_an_empty_cloud_round_trips() -> None:
    poly = make_polydata(np.empty((0, 3)), [])
    back = _read(_write(poly))
    assert back.vertices.shape == (0, 3)
    assert back.vertex_attrs["intensity"].shape == (0,)
    lazy_back = polyxios.read(io.BytesIO(_write(poly)), fmt=".las")
    assert lazy_back.vertices.shape == (0, 3)


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def test_every_standard_field_reads_back_in_its_file_dtype() -> None:
    poly = _cloud(
        scan_direction_flag=np.array([0, 1, 0, 1], np.uint8),
        edge_of_flight_line=np.array([1, 0, 0, 0], np.uint8),
        synthetic=np.array([1, 0, 0, 0], np.uint8),
        key_point=np.array([0, 1, 0, 0], np.uint8),
        withheld=np.array([0, 0, 1, 0], np.uint8),
        scan_angle_rank=np.array([-90, 0, 45, 90], np.int8),
        user_data=np.array([7, 8, 9, 10], np.uint8),
        point_source_id=np.array([1, 2, 3, 65535], np.uint16),
    )
    back = _read(_write(poly, scale=0.001))
    for name, value in poly.vertex_attrs.items():
        if name == "colors":
            continue
        np.testing.assert_array_equal(back.vertex_attrs[name], value, err_msg=name)
        assert back.vertex_attrs[name].dtype == value.dtype, name
    np.testing.assert_allclose(back.vertex_attrs["colors"], poly.vertex_attrs["colors"])
    assert back.vertex_attrs["colors"].dtype == np.float64


def test_the_new_style_fields_read_back() -> None:
    poly = _cloud(
        overlap=np.array([1, 0, 0, 1], np.uint8),
        scanner_channel=np.array([0, 1, 2, 3], np.uint8),
        scan_angle=np.array([-30000, 0, 15000, 30000], np.int16),
        nir=np.array([0, 1, 2, 65535], np.uint16),
        return_number=np.array([1, 15, 8, 3], np.uint8),
        number_of_returns=np.array([15, 15, 8, 3], np.uint8),
        classification=np.array([0, 64, 200, 255], np.uint8),
    )
    data = _write(poly)
    assert data[104] == 8
    back = _read(data)
    for name in (
        "overlap",
        "scanner_channel",
        "scan_angle",
        "nir",
        "return_number",
        "number_of_returns",
        "classification",
    ):
        np.testing.assert_array_equal(
            back.vertex_attrs[name], poly.vertex_attrs[name], err_msg=name
        )
    assert "scan_angle_rank" not in back.vertex_attrs
    assert "overlap" in back.vertex_attrs


def test_the_wave_packet_fields_read_back() -> None:
    n = 4
    poly = _cloud(
        wavepacket_index=np.array([1, 1, 2, 2], np.uint8),
        wavepacket_offset=np.arange(n, dtype=np.uint64) * 1000,
        wavepacket_size=np.full(n, 256, np.uint32),
        return_point_wave_location=np.linspace(0, 1, n, dtype=np.float32),
        x_t=np.full(n, 0.5, np.float32),
        y_t=np.full(n, -0.5, np.float32),
        z_t=np.full(n, 1.0, np.float32),
    )
    data = _write(poly)
    assert data[104] == 5
    back = _read(data)
    for name in (
        "wavepacket_index",
        "wavepacket_offset",
        "wavepacket_size",
        "return_point_wave_location",
        "x_t",
        "y_t",
        "z_t",
    ):
        np.testing.assert_array_equal(back.vertex_attrs[name], poly.vertex_attrs[name])
        assert back.vertex_attrs[name].dtype == poly.vertex_attrs[name].dtype


def test_a_missing_return_number_defaults_to_one_of_one() -> None:
    poly = make_polydata(np.zeros((2, 3)), [("vertex", np.array([[0], [1]]))])
    back = _read(_write(poly))
    np.testing.assert_array_equal(back.vertex_attrs["return_number"], [1, 1])
    np.testing.assert_array_equal(back.vertex_attrs["number_of_returns"], [1, 1])
    np.testing.assert_array_equal(back.vertex_attrs["classification"], [0, 0])


@pytest.mark.parametrize(
    ("name", "value", "point_format"),
    [
        ("return_number", 8, 0),
        ("number_of_returns", 8, 3),
        ("classification", 32, 1),
        ("return_number", 16, 6),
        ("scanner_channel", 4, 6),
        ("scan_direction_flag", 2, 0),
        ("synthetic", 2, 6),
    ],
)
def test_a_bit_field_past_its_width_is_refused(name, value, point_format) -> None:
    poly = _cloud(**{name: np.array([0, 0, 0, value], np.uint8)})
    with pytest.raises(CodecError, match=name):
        _write(poly, point_format=point_format)


def test_a_scan_angle_converts_between_the_two_spellings() -> None:
    legacy = _cloud(scan_angle_rank=np.array([-90, 0, 45, 90], np.int8))
    back = _read(_write(legacy, point_format=7))
    np.testing.assert_array_equal(
        back.vertex_attrs["scan_angle"], [-15000, 0, 7500, 15000]
    )
    new = _cloud(scan_angle=np.array([-15000, 0, 7500, 15000], np.int16))
    back = _read(_write(new, point_format=3))
    np.testing.assert_array_equal(
        back.vertex_attrs["scan_angle_rank"], [-90, 0, 45, 90]
    )


def test_a_scan_angle_past_the_legacy_rank_is_refused() -> None:
    poly = _cloud(scan_angle=np.array([0, 0, 0, 30000], np.int16))
    with pytest.raises(CodecError, match="scan_angle"):
        _write(poly, point_format=3)


def test_a_nan_scan_angle_is_refused_in_either_spelling() -> None:
    rank = _cloud(scan_angle_rank=np.array([0.0, 1.0, np.nan, 3.0]))
    with pytest.raises(CodecError, match="scan_angle_rank.*NaN"):
        _write(rank, point_format=7)
    steps = _cloud(scan_angle=np.array([0.0, 100.0, np.nan, 300.0]))
    with pytest.raises(CodecError, match="'scan_angle'.*NaN"):
        _write(steps, point_format=3)
    with pytest.raises(CodecError, match="'scan_angle'.*infinity"):
        _write(_cloud(scan_angle=np.array([0.0, 1.0, np.inf, 3.0])), point_format=3)


def test_a_scan_angle_rank_past_the_int16_steps_is_refused() -> None:
    poly = _cloud(scan_angle_rank=np.array([0.0, 1.0, 2.0, 500.0]))
    with pytest.raises(CodecError, match="scan_angle_rank reaches 500 degrees"):
        _write(poly, point_format=7)
    edge = _cloud(scan_angle_rank=np.array([-196, 0, 90, 196], np.int16))
    np.testing.assert_array_equal(
        _read(_write(edge, point_format=7)).vertex_attrs["scan_angle"],
        [-32667, 0, 15000, 32667],
    )


def test_a_scan_angle_the_format_lacks_is_dropped_with_a_warning_when_both_are_given():
    poly = _cloud(
        scan_angle_rank=np.array([1, 2, 3, 4], np.int8),
        scan_angle=np.array([10, 20, 30, 40], np.int16),
    )
    with pytest.warns(UserWarning, match="'scan_angle'"):
        data = _write(poly, point_format=3)
    np.testing.assert_array_equal(
        _read(data).vertex_attrs["scan_angle_rank"], [1, 2, 3, 4]
    )
    with pytest.warns(UserWarning, match="'scan_angle_rank'"):
        data = _write(poly, point_format=7)
    np.testing.assert_array_equal(
        _read(data).vertex_attrs["scan_angle"], [10, 20, 30, 40]
    )


def test_a_value_its_field_cannot_hold_is_refused_and_a_float_is_rounded() -> None:
    with pytest.raises(CodecError, match="intensity.*70000.*65535"):
        _write(_cloud(intensity=np.array([0.0, 1.0, 2.0, 70000.0])))
    with pytest.raises(CodecError, match="point_source_id.*-1"):
        _write(_cloud(point_source_id=np.array([0, 1, 2, -1], np.int64)))
    with pytest.raises(CodecError, match="user_data.*NaN"):
        _write(_cloud(user_data=np.array([0.0, 1.0, np.nan, 3.0])))
    back = _read(_write(_cloud(intensity=np.array([0.4, 1.5, 2.6, 65535.0]))))
    np.testing.assert_array_equal(back.vertex_attrs["intensity"], [0, 2, 3, 65535])
    assert back.vertex_attrs["intensity"].dtype == np.uint16


def test_a_bit_field_is_rounded_and_refuses_a_nan_like_any_integer_field() -> None:
    with pytest.raises(CodecError, match="classification.*NaN"):
        _write(_cloud(classification=np.array([np.nan, 2.0, 3.0, 4.0])))
    back = _read(_write(_cloud(classification=np.array([1.6, 2.4, 3.5, 4.0]))))
    np.testing.assert_array_equal(back.vertex_attrs["classification"], [2, 2, 4, 4])


@pytest.mark.parametrize("attr", ["intensity", "gps_time", "classification"])
def test_a_non_numeric_column_is_refused_by_name(attr) -> None:
    with pytest.raises(CodecError, match=f"{attr}.*dtype"):
        _write(_cloud(**{attr: np.array(["a", "b", "c", "d"])}))


def test_an_attribute_that_is_not_one_row_per_vertex_is_refused_by_name() -> None:
    for attr, value in [
        ("intensity", np.arange(3, dtype=np.uint16)),
        ("intensity", np.zeros((4, 2), np.uint16)),
        ("return_number", np.ones(5, np.uint8)),
        ("colors", np.zeros((3, 3))),
        ("height", np.zeros(3)),
    ]:
        poly = _cloud()
        poly.vertex_attrs[attr] = value
        with pytest.raises(CodecError, match=f"{attr}.*per vertex"):
            _write(poly)


def test_a_coordinate_of_minus_2_to_the_31_fits_an_int32() -> None:
    poly = make_polydata(np.array([[-(2**31), 0.0, 2**31 - 1]]), [])
    back = _read(_write(poly, scale=1.0, offset=0.0))
    np.testing.assert_array_equal(back.vertices, [[-(2**31), 0.0, 2**31 - 1]])


def test_integer_colors_count_bytes_unless_they_are_uint16() -> None:
    poly = _cloud(colors=np.array([[255, 0, 0]] * 4, np.int64))
    data = _write(poly)
    back = _read(data)
    np.testing.assert_allclose(back.vertex_attrs["colors"], [[1.0, 0.0, 0.0]] * 4)
    poly = _cloud(colors=np.array([[65535, 256, 0]] * 4, np.uint16))
    back = _read(_write(poly))
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], [[1.0, 256 / 65535, 0.0]] * 4
    )


def test_big_endian_uint16_colors_are_written_as_they_are() -> None:
    poly = _cloud(colors=np.array([[65535, 256, 0]] * 4, ">u2"))
    back = _read(_write(poly))
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], [[1.0, 256 / 65535, 0.0]] * 4
    )


def test_an_attribute_named_like_a_reserved_dimension_is_dropped_with_a_warning():
    poly = _cloud(
        X=np.zeros(4), red=np.zeros(4, np.uint16), flags=np.zeros(4, np.uint8)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = _write(poly)
    messages = " ".join(str(w.message) for w in caught)
    for name in ("X", "red", "flags"):
        assert f"'{name}'" in messages
    assert _u32(data, 100) == 0
    assert not {"X", "red", "flags"} & set(_read(data).vertex_attrs)


def test_an_attribute_with_an_empty_name_is_dropped_with_a_warning() -> None:
    with pytest.warns(UserWarning, match="''.*dropped"):
        data = _write(_cloud(**{"": np.zeros(4)}))
    assert "" not in _read(data).vertex_attrs


def test_an_attribute_the_point_format_lacks_is_dropped_with_a_warning() -> None:
    with pytest.warns(UserWarning, match="colors"):
        data = _write(_cloud(), point_format=1)
    back = _read(data)
    assert "colors" not in back.vertex_attrs
    assert "gps_time" in back.vertex_attrs


def test_elements_are_dropped_with_a_warning() -> None:
    poly = make_polydata(
        np.eye(3), [("triangle", np.array([[0, 1, 2]]))], vertex_attrs={}
    )
    with pytest.warns(UserWarning, match="1 elements dropped"):
        data = _write(poly)
    assert len(_read(data).element_types) == 0


# ---------------------------------------------------------------------------
# Extra bytes
# ---------------------------------------------------------------------------


def test_other_attributes_go_out_as_extra_bytes_and_read_back_by_name() -> None:
    poly = _cloud(
        height=np.array([1.5, 2.5, 3.5, 4.5]),
        label=np.array([1, 2, 3, 4], np.int32),
        normals=np.tile([0.0, 0.0, 1.0], (4, 1)).astype(np.float32),
        flag=np.array([True, False, True, False]),
    )
    data = _write(poly)
    n_vlr = _u32(data, 100)
    assert n_vlr == 1
    assert _u16(data, 105) == 34 + 8 + 4 + 12 + 1
    back = _read(data)
    np.testing.assert_array_equal(back.vertex_attrs["height"], [1.5, 2.5, 3.5, 4.5])
    assert back.vertex_attrs["height"].dtype == np.float64
    np.testing.assert_array_equal(back.vertex_attrs["label"], [1, 2, 3, 4])
    assert back.vertex_attrs["label"].dtype == np.int32
    np.testing.assert_array_equal(
        back.vertex_attrs["normals"], poly.vertex_attrs["normals"]
    )
    assert back.vertex_attrs["normals"].dtype == np.float32
    np.testing.assert_array_equal(back.vertex_attrs["flag"], [1, 0, 1, 0])
    assert back.vertex_attrs["flag"].dtype == np.uint8


def test_a_scaled_extra_byte_is_applied_on_an_eager_read() -> None:
    data = _write(_cloud(height=np.array([1, 2, 3, 4], np.int16)))
    vlr_at = _HEADER_1_2 + 54
    data = _patched(data, "<B", vlr_at + 3, 0x18)
    data = _patched(data, "<d", vlr_at + 112, 0.5)
    data = _patched(data, "<d", vlr_at + 136, 10.0)
    back = _read(data)
    np.testing.assert_array_equal(back.vertex_attrs["height"], [10.5, 11.0, 11.5, 12.0])


def test_an_extra_byte_name_that_is_a_standard_field_or_repeats_is_suffixed():
    data = _write(_cloud(height=np.arange(4, dtype=np.uint8)))
    vlr_at = _HEADER_1_2 + 54
    named = bytearray(data)
    named[vlr_at + 4 : vlr_at + 36] = b"intensity".ljust(32, b"\0")
    with pytest.warns(UserWarning, match="intensity"):
        back = _read(bytes(named))
    assert "intensity" in back.vertex_attrs
    np.testing.assert_array_equal(back.vertex_attrs["intensity_1"], [0, 1, 2, 3])


def test_an_extra_byte_with_a_blank_name_reads_as_its_position() -> None:
    data = _write(_cloud(height=np.arange(4, dtype=np.uint8)))
    vlr_at = _HEADER_1_2 + 54
    blank = bytearray(data)
    blank[vlr_at + 4 : vlr_at + 36] = b"\0" * 32
    back = _read(bytes(blank))
    assert "" not in back.vertex_attrs
    np.testing.assert_array_equal(back.vertex_attrs["extra_0"], [0, 1, 2, 3])


def test_an_extra_byte_name_is_stripped_of_trailing_whitespace_on_write() -> None:
    data = _write(_cloud(**{"height ": np.arange(4, dtype=np.uint8)}))
    vlr_at = _HEADER_1_2 + 54
    assert data[vlr_at + 4 : vlr_at + 36] == b"height".ljust(32, b"\0")
    np.testing.assert_array_equal(_read(data).vertex_attrs["height"], [0, 1, 2, 3])


def test_a_standard_name_with_trailing_whitespace_fills_its_field() -> None:
    poly = _cloud()
    attrs = dict(poly.vertex_attrs)
    attrs["intensity "] = attrs.pop("intensity")
    poly = make_polydata(poly.vertices, [], vertex_attrs=attrs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = _write(poly)
    np.testing.assert_array_equal(
        _read(data).vertex_attrs["intensity"], [0, 100, 200, 300]
    )
    assert _u32(data, 100) == 0


def test_a_name_that_strips_to_another_attribute_is_dropped_with_a_warning() -> None:
    poly = _cloud(**{"intensity ": np.zeros(4, np.uint16)})
    with pytest.warns(UserWarning, match="'intensity '.*dropped"):
        data = _write(poly)
    np.testing.assert_array_equal(
        _read(data).vertex_attrs["intensity"], [0, 100, 200, 300]
    )
    assert _u32(data, 100) == 0


def test_undescribed_extra_bytes_read_as_one_byte_block() -> None:
    data = _write(_cloud())
    n = 4
    body = bytearray(data[_HEADER_1_2:])
    widened = b"".join(
        bytes(body[i * 34 : (i + 1) * 34]) + bytes([i, 100]) for i in range(n)
    )
    data = _patched(data[:_HEADER_1_2], "<H", 105, 36) + widened
    back = _read(data)
    np.testing.assert_array_equal(
        back.vertex_attrs["extra_bytes"], [[0, 100], [1, 100], [2, 100], [3, 100]]
    )


def test_extra_bytes_descriptors_past_the_record_are_refused() -> None:
    data = _write(_cloud(height=np.arange(4, dtype=np.float64)))
    vlr_at = _HEADER_1_2 + 54
    # float64 -> its 3-array twin, 24 bytes where the record holds 8.
    data = _patched(data, "<B", vlr_at + 2, 30)
    with pytest.raises(CodecError, match="extra bytes"):
        _read(data)


def test_an_unknown_extra_byte_data_type_is_refused() -> None:
    data = _write(_cloud(height=np.arange(4, dtype=np.float64)))
    data = _patched(data, "<B", _HEADER_1_2 + 54 + 2, 31)
    with pytest.raises(CodecError, match="data type"):
        _read(data)


def test_wide_byte_attributes_write_as_undocumented_extra_bytes() -> None:
    poly = _cloud(
        reserved=np.arange(28, dtype=np.uint8).reshape(4, 7),
        extra_bytes=np.arange(16, dtype=np.uint8).reshape(4, 4),
    )
    data = _write(poly)
    assert _u16(data, 105) == 34 + 7 + 4
    vlr_at = _HEADER_1_2 + 54
    assert data[vlr_at + 2] == 0
    assert data[vlr_at + 3] == 7
    assert len(data[_HEADER_1_2 + 54 : _u32(data, 96)]) == 192
    back = _read(data)
    np.testing.assert_array_equal(
        back.vertex_attrs["reserved"], poly.vertex_attrs["reserved"]
    )
    np.testing.assert_array_equal(
        back.vertex_attrs["extra_bytes"], poly.vertex_attrs["extra_bytes"]
    )


@pytest.mark.parametrize("width", [1, 2, 3])
def test_a_narrow_extra_bytes_block_goes_back_undescribed_in_its_shape(width):
    poly = _cloud(extra_bytes=np.ones((4, width), np.uint8))
    data = _write(poly)
    assert _u32(data, 100) == 0
    back = _read(data)
    assert back.vertex_attrs["extra_bytes"].shape == (4, width)


def test_an_attribute_name_with_a_control_character_is_dropped_with_a_warning():
    with pytest.warns(UserWarning, match="printable"):
        data = _write(_cloud(**{"a\0b": np.zeros(4)}))
    assert _u32(data, 100) == 0
    assert "a" not in _read(data).vertex_attrs


def test_a_byte_block_over_255_columns_is_dropped_with_a_warning() -> None:
    poly = _cloud(wide=np.zeros((4, 300), np.uint8))
    with pytest.warns(UserWarning, match="wide.*300 columns"):
        data = _write(poly)
    assert "wide" not in _read(data).vertex_attrs


def test_a_big_endian_attribute_goes_out_as_a_little_endian_extra_byte() -> None:
    poly = _cloud(height=np.array([1.5, 2.5, 3.5, 4.5], dtype=">f8"))
    back = _read(_write(poly))
    np.testing.assert_array_equal(back.vertex_attrs["height"], [1.5, 2.5, 3.5, 4.5])
    assert back.vertex_attrs["height"].dtype == np.dtype("<f8")


def test_a_bool_or_half_float_extra_byte_widens_to_a_type_the_spec_has() -> None:
    poly = _cloud(
        flag=np.array([True, False, True, False]),
        half=np.arange(4, dtype=np.float16) / 4,
    )
    attrs = _read(_write(poly)).vertex_attrs
    assert attrs["flag"].dtype == np.uint8
    np.testing.assert_array_equal(attrs["flag"], [1, 0, 1, 0])
    assert attrs["half"].dtype == np.float32
    np.testing.assert_array_equal(attrs["half"], [0, 0.25, 0.5, 0.75])


def test_an_undescribed_remainder_does_not_overwrite_a_described_extra_bytes():
    poly = _cloud(extra_bytes=np.array([1.5, 2.5, 3.5, 4.5]))
    data = _write(poly)
    at = _u32(data, 96)
    body = bytearray(data[at:])
    widened = b"".join(
        bytes(body[i * 42 : (i + 1) * 42]) + bytes([i, 100]) for i in range(4)
    )
    data = _patched(data[:at], "<H", 105, 44) + widened
    with pytest.warns(UserWarning, match="extra_bytes_1"):
        back = _read(data)
    np.testing.assert_array_equal(
        back.vertex_attrs["extra_bytes"], [1.5, 2.5, 3.5, 4.5]
    )
    np.testing.assert_array_equal(
        back.vertex_attrs["extra_bytes_1"], [[0, 100], [1, 100], [2, 100], [3, 100]]
    )


def test_a_legacy_format_defaults_to_1_2_and_1_0_only_when_asked() -> None:
    poly = _cloud()
    del poly.vertex_attrs["colors"]
    del poly.vertex_attrs["gps_time"]
    data = _write(poly)
    assert data[104] == 0
    assert (data[24], data[25]) == (1, 2)
    poly.global_attrs["las_global_encoding"] = 1
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 7, "description": "d", "data": b"\x01"}
    ]
    data = _write(poly, version="1.0")
    assert (data[24], data[25]) == (1, 0)
    assert _u16(data, 6) == 0
    assert _u16(data, _HEADER_1_2) == 0xAABB
    assert "las_global_encoding" not in _read(data).global_attrs
    data = _write(poly, version="1.2")
    assert _u16(data, 6) == 1
    assert _u16(data, _HEADER_1_2) == 0
    assert _read(data).global_attrs["las_global_encoding"] == 1


def test_an_inherited_version_is_a_floor_that_the_format_raises() -> None:
    back = _read(_write(_cloud()))
    assert back.global_attrs["las_version"] == "1.2"
    data = _write(back, point_format=7)
    assert (data[24], data[25]) == (1, 4)
    data = _write(back, point_format=5)
    assert (data[24], data[25]) == (1, 3)
    with pytest.raises(CodecError, match="1.4"):
        _write(back, point_format=7, version="1.2")


def test_a_1_1_file_read_back_writes_as_1_1() -> None:
    poly = _cloud()
    del poly.vertex_attrs["colors"]
    data = _write(poly, version="1.1", point_format=1)
    assert (data[24], data[25]) == (1, 1)
    back = _read(data)
    assert back.global_attrs["las_version"] == "1.1"
    again = _write(back)
    assert (again[24], again[25]) == (1, 1)


def test_an_attribute_with_no_extra_byte_type_is_dropped_with_a_warning() -> None:
    poly = _cloud(
        wide=np.zeros((4, 5)),
        text=np.array(["a", "b", "c", "d"]),
        **{"a" * 40: np.zeros(4)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = _write(poly)
    messages = " ".join(str(w.message) for w in caught)
    assert "wide" in messages and "text" in messages and "a" * 40 in messages
    back = _read(data)
    assert not {"wide", "text", "a" * 40} & set(back.vertex_attrs)


# ---------------------------------------------------------------------------
# VLRs, CRS and the other global attributes
# ---------------------------------------------------------------------------


def test_a_wkt_crs_writes_a_vlr_and_sets_the_wkt_bit() -> None:
    poly = _cloud()
    poly.global_attrs["crs_wkt"] = 'GEOGCS["WGS 84"]'
    data = _write(poly)
    assert _u16(data, 6) & 0x10
    assert _u32(data, 100) == 1
    back = _read(data)
    assert back.global_attrs["crs_wkt"] == 'GEOGCS["WGS 84"]'
    assert "las_vlrs" not in back.global_attrs


def test_a_wkt_too_long_for_a_vlr_clears_the_wkt_bit_before_1_4() -> None:
    poly = _cloud()
    poly.global_attrs["crs_wkt"] = "PROJCS[" + "x" * 70000 + "]"
    with pytest.warns(UserWarning, match="65535"):
        data = _write(poly, version="1.2")
    assert _u16(data, 6) & 0x10 == 0
    assert "crs_wkt" not in _read(data).global_attrs
    data = _write(poly, version="1.4")
    assert _u16(data, 6) & 0x10
    assert _read(data).global_attrs["crs_wkt"] == poly.global_attrs["crs_wkt"]


def test_the_wkt_bit_is_cleared_when_the_crs_is_dropped() -> None:
    poly = _cloud()
    poly.global_attrs["crs_wkt"] = 'GEOGCS["WGS 84"]'
    back = _read(_write(poly))
    assert back.global_attrs["las_global_encoding"] & 0x10
    del back.global_attrs["crs_wkt"]
    data = _write(back)
    assert not _u16(data, 6) & 0x10
    assert _u32(data, 100) == 0


def test_a_geokey_outside_16_bits_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["las_geokeys"] = {1024: 1, 3072: -1, 70000: 2, 2048: object()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = _write(poly)
    messages = " ".join(str(w.message) for w in caught)
    assert "3072" in messages and "70000" in messages and "2048" in messages
    assert _read(data).global_attrs["las_geokeys"] == {1024: 1}


def test_a_geokey_list_with_a_non_numeric_item_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["las_geokeys"] = {1024: 1, 2054: ["a"], 2055: [1.5, 2.5]}
    with pytest.warns(UserWarning, match="2054"):
        data = _write(poly)
    assert _read(data).global_attrs["las_geokeys"] == {1024: 1, 2055: [1.5, 2.5]}


def test_a_geokey_past_what_its_record_can_index_is_dropped_with_a_warning() -> None:
    keys = {1024: 2, 2049: "x" * 70_000, 2057: [1.0] * 70_000, 3076: 9001}
    poly = _cloud()
    poly.global_attrs["las_geokeys"] = keys
    with pytest.warns(UserWarning) as record:
        data = _write(poly)
    messages = sorted(str(w.message) for w in record)
    assert len(messages) == 2
    assert "key 2049 holds a string" in messages[0]
    assert "key 2057 holds 70000 doubles" in messages[1]
    assert _read(data).global_attrs["las_geokeys"] == {1024: 2, 3076: 9001}


def test_a_header_global_outside_16_bits_is_refused_by_name() -> None:
    for key, value in [
        ("las_file_source_id", 70000),
        ("las_global_encoding", -1),
        ("las_creation_date", (2024, 70000)),
    ]:
        poly = _cloud()
        poly.global_attrs[key] = value
        with pytest.raises(CodecError, match=key):
            _write(poly)


def test_a_creation_date_that_is_not_a_pair_is_refused_by_name() -> None:
    for bad in (2024, "2024-100", (2024,), np.array(2024)):
        poly = _cloud()
        poly.global_attrs["las_creation_date"] = bad
        with pytest.raises(CodecError, match="las_creation_date"):
            _write(poly)


def test_geotiff_keys_round_trip_through_their_three_vlrs() -> None:
    poly = _cloud()
    poly.global_attrs["las_geokeys"] = {
        1024: 1,
        3072: 32610,
        2049: "WGS 84",
        3076: 9001,
        2054: 0.0174532925199433,
    }
    data = _write(poly)
    assert _u32(data, 100) == 3
    back = _read(data)
    assert back.global_attrs["las_geokeys"] == poly.global_attrs["las_geokeys"]


def test_geotiff_keys_are_written_in_ascending_order() -> None:
    poly = _cloud()
    poly.global_attrs["las_geokeys"] = {3076: 9001, 1024: 2, 2049: "WGS 84"}
    data = _write(poly)
    header = _las._parse_header(data, name="x")
    directory = next(v.data for v in header.vlrs if v.record_id == 34735)
    ids = [struct.unpack_from("<H", directory, 8 * i)[0] for i in range(1, 4)]
    assert ids == [1024, 2049, 3076]
    assert list(_read(data).global_attrs["las_geokeys"]) == [1024, 2049, 3076]


def test_geokeys_that_are_not_a_mapping_are_refused_by_name() -> None:
    for bad in ("EPSG:4326", 4326, [(1024, 2)]):
        poly = _cloud()
        poly.global_attrs["las_geokeys"] = bad
        with pytest.raises(CodecError, match="las_geokeys.*mapping"):
            _write(poly)


def test_other_vlrs_are_kept_raw_and_written_back() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 7, "description": "d", "data": b"\x01\x02"}
    ]
    data = _write(poly)
    back = _read(data)
    assert back.global_attrs["las_vlrs"] == poly.global_attrs["las_vlrs"]


def test_a_long_vlr_becomes_an_evlr_in_1_4_and_is_dropped_before() -> None:
    poly = _cloud()
    big = {"user_id": "acme", "record_id": 7, "description": "", "data": bytes(70000)}
    poly.global_attrs["las_vlrs"] = [big]
    data = _write(poly, version="1.4")
    assert _u32(data, 100) == 0
    assert _u32(data, 243) == 1
    back = _read(data)
    assert back.global_attrs["las_vlrs"] == [big]
    with pytest.warns(UserWarning, match="65535"):
        data = _write(poly, version="1.2")
    assert "las_vlrs" not in _read(data).global_attrs


def test_a_vlr_the_writer_makes_itself_is_not_written_twice() -> None:
    poly = _cloud(height=np.zeros(4))
    poly.global_attrs["crs_wkt"] = "PROJCS[]"
    descriptor = _las._pack_extras([_las._ExtraOut("height", "<f8", None, 10, 0)])
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "LASF_Spec", "record_id": 4, "description": "", "data": descriptor},
        {"user_id": "LASF_Projection", "record_id": 2112, "data": b"x\0"},
        {"user_id": "laszip encoded", "record_id": 22204, "data": b""},
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = _write(poly)
    messages = " ".join(str(w.message) for w in caught)
    for user_id in ("LASF_Spec", "LASF_Projection", "laszip encoded"):
        assert user_id in messages
    assert _u32(data, 100) == 2
    back = _read(data)
    np.testing.assert_array_equal(back.vertex_attrs["height"], np.zeros(4))
    assert back.global_attrs["crs_wkt"] == "PROJCS[]"
    assert "las_vlrs" not in back.global_attrs


def test_a_vlr_entry_with_a_non_ascii_id_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acm\u00e9", "record_id": 1, "data": b""}
    ]
    with pytest.warns(UserWarning, match="user_id"):
        data = _write(poly)
    assert _u32(data, 100) == 0
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 1, "description": "d\u00e9", "data": b""}
    ]
    with pytest.warns(UserWarning, match="description"):
        data = _write(poly)
    assert _u32(data, 100) == 0


def test_a_vlr_entry_with_a_nul_in_its_id_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [{"user_id": "ac\0me", "record_id": 1, "data": b""}]
    with pytest.warns(UserWarning, match="user_id.*printable"):
        data = _write(poly)
    assert _u32(data, 100) == 0


def test_a_vlr_entry_takes_a_numpy_record_id() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": np.uint16(7), "description": "", "data": b"z"}
    ]
    back = _read(_write(poly))
    assert back.global_attrs["las_vlrs"][0]["record_id"] == 7


def test_a_vlr_entry_takes_memoryview_data() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 7, "description": "", "data": memoryview(b"z")}
    ]
    back = _read(_write(poly))
    assert back.global_attrs["las_vlrs"][0]["data"] == b"z"


def test_a_malformed_vlr_entry_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [{"user_id": "x" * 20, "record_id": 1, "data": b""}]
    with pytest.warns(UserWarning, match="user_id"):
        data = _write(poly)
    assert _u32(data, 100) == 0


def test_vlrs_that_are_not_a_list_are_refused_by_name() -> None:
    for bad in (5, "abc", {"user_id": "x", "record_id": 1, "data": b""}):
        poly = _cloud()
        poly.global_attrs["las_vlrs"] = bad
        with pytest.raises(CodecError, match="las_vlrs.*list of dicts"):
            _write(poly)


def test_the_identifying_globals_round_trip() -> None:
    poly = _cloud()
    poly.global_attrs.update(
        las_system_identifier="scanner",
        las_generating_software="me",
        las_file_source_id=12,
        las_creation_date=(2026, 100),
        las_global_encoding=1,
    )
    back = _read(_write(poly))
    for key, value in poly.global_attrs.items():
        assert back.global_attrs[key] == value, key


def test_a_header_string_over_32_bytes_is_cut_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["las_system_identifier"] = "s" * 40
    poly.global_attrs["las_generating_software"] = "g" * 33
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = _write(poly)
    messages = " ".join(str(w.message) for w in caught)
    assert "las_system_identifier" in messages and "32" in messages
    assert "las_generating_software" in messages
    back = _read(data)
    assert back.global_attrs["las_system_identifier"] == "s" * 32
    assert back.global_attrs["las_generating_software"] == "g" * 32


@pytest.mark.parametrize(
    ("value", "match", "expected"),
    [("ab\0cd", "NUL", "ab"), ("h\u00e9llo", "ASCII", "h?llo")],
)
def test_a_header_string_the_field_cannot_hold_is_cut_with_a_warning(
    value, match, expected
) -> None:
    poly = _cloud()
    poly.global_attrs["las_system_identifier"] = value
    with pytest.warns(UserWarning, match=f"las_system_identifier.*{match}"):
        data = _write(poly)
    assert _read(data).global_attrs["las_system_identifier"] == expected


def test_a_utf8_header_string_and_extra_byte_name_read_as_text() -> None:
    data = _write(_cloud(height=np.arange(4, dtype=np.uint8)))
    patched = bytearray(data)
    patched[26:58] = "h\u00e9llo".encode().ljust(32, b"\0")
    vlr_at = _HEADER_1_2 + 54
    patched[vlr_at + 4 : vlr_at + 36] = "h\u00e9ight".encode().ljust(32, b"\0")
    back = _read(bytes(patched))
    assert back.global_attrs["las_system_identifier"] == "h\u00e9llo"
    np.testing.assert_array_equal(back.vertex_attrs["h\u00e9ight"], [0, 1, 2, 3])


def test_a_wkt_with_a_nul_is_cut_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs["crs_wkt"] = 'GEOGCS["a"]\0junk'
    with pytest.warns(UserWarning, match="crs_wkt.*NUL"):
        data = _write(poly)
    assert _read(data).global_attrs["crs_wkt"] == 'GEOGCS["a"]'


def test_a_text_global_given_as_bytes_is_written_as_its_text() -> None:
    poly = _cloud()
    poly.global_attrs["crs_wkt"] = b'GEOGCS["WGS 84"]'
    poly.global_attrs["las_system_identifier"] = b"scanner"
    g = _read(_write(poly)).global_attrs
    assert g["crs_wkt"] == 'GEOGCS["WGS 84"]'
    assert g["las_system_identifier"] == "scanner"


def test_the_defaults_leave_no_date_and_name_polyxios() -> None:
    back = _read(_write(_cloud()))
    assert back.global_attrs["las_generating_software"] == "polyxios"
    assert "las_creation_date" not in back.global_attrs
    assert "las_file_source_id" not in back.global_attrs
    assert "las_system_identifier" not in back.global_attrs


def test_a_global_point_format_and_version_are_honoured_on_write() -> None:
    poly = _cloud()
    poly.global_attrs.update(las_point_format=7, las_version="1.4")
    data = _write(poly)
    assert data[104] == 7
    assert (data[24], data[25]) == (1, 4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"point_format": 6, "version": "1.2"}, "1.4"),
        ({"point_format": 4, "version": "1.2"}, "1.3"),
        ({"point_format": 11}, "point_format"),
        ({"point_format": 2, "version": "1.1"}, "1.2"),
        ({"version": "1.5"}, "version"),
        ({"version": "2.0"}, "version"),
    ],
)
def test_a_format_and_version_that_do_not_go_together_are_refused(kwargs, match):
    with pytest.raises((CodecError, ValueError), match=match):
        _write(_cloud(), **kwargs)


# ---------------------------------------------------------------------------
# Refusals on read
# ---------------------------------------------------------------------------


def test_a_bad_signature_is_refused() -> None:
    with pytest.raises(CodecError, match="LASF"):
        _read(b"LASX" + _write(_cloud())[4:])


def test_a_truncated_header_is_refused() -> None:
    with pytest.raises(CodecError, match="header"):
        _read(_write(_cloud())[:100])


def test_an_unknown_version_is_refused() -> None:
    with pytest.raises(CodecError, match=r"2\.0"):
        _read(_patched(_patched(_write(_cloud()), "<B", 24, 2), "<B", 25, 0))


def test_an_unknown_point_format_is_refused() -> None:
    with pytest.raises(CodecError, match="11"):
        _read(_patched(_write(_cloud()), "<B", 104, 11))


def test_a_header_shorter_than_its_version_needs_is_refused() -> None:
    with pytest.raises(CodecError, match="header size"):
        _read(_patched(_write(_cloud()), "<H", 94, 200))


def test_a_point_offset_inside_the_header_is_refused() -> None:
    with pytest.raises(CodecError, match="offset"):
        _read(_patched(_write(_cloud()), "<I", 96, 100))


def test_a_record_shorter_than_its_format_is_refused() -> None:
    with pytest.raises(CodecError, match="record length"):
        _read(_patched(_write(_cloud()), "<H", 105, 20))


def test_a_count_past_the_file_is_refused_naming_it() -> None:
    with pytest.raises((CodecError, ValidationError), match="4000000"):
        _read(_patched(_write(_cloud()), "<I", 107, 4_000_000))


def test_a_vlr_running_past_the_point_offset_is_refused() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 7, "description": "", "data": b"\x01" * 8}
    ]
    data = _write(poly)
    with pytest.raises(CodecError, match="VLR"):
        _read(_patched(data, "<H", _HEADER_1_2 + 20, 5000))


def test_a_point_offset_past_the_file_is_refused_before_the_vlrs_are_walked():
    poly = make_polydata(np.empty((0, 3)), [])
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 7, "description": "", "data": b"\x01" * 8}
    ]
    data = _patched(_write(poly), "<I", 96, 10**6)
    data = _patched(data, "<I", 100, 2)
    with pytest.raises(CodecError, match="offset to point data 1000000"):
        _read(data)


def test_an_evlr_past_the_file_is_refused() -> None:
    poly = _cloud()
    poly.global_attrs["las_vlrs"] = [
        {"user_id": "acme", "record_id": 7, "description": "", "data": bytes(70000)}
    ]
    data = _write(poly, version="1.4")
    with pytest.raises(CodecError, match="EVLR"):
        _read(_patched(data, "<Q", 235, len(data) - 10))


def test_a_1_4_file_with_no_new_count_falls_back_to_the_legacy_one() -> None:
    data = _write(_cloud(), version="1.4")
    data = _patched(data, "<Q", 247, 0)
    assert _read(data).vertices.shape == (4, 3)


# ---------------------------------------------------------------------------
# Lazy reads
# ---------------------------------------------------------------------------


def test_a_lazy_read_hands_back_views_in_the_file_dtypes(tmp_path) -> None:
    poly = _cloud(height=np.arange(4, dtype=np.float32))
    path = tmp_path / "c.las"
    polyxios.write(poly, path, scale=0.001)
    lazy = polyxios.read(path, lazy=True)
    eager = polyxios.read(path)
    assert lazy.vertices.dtype == np.dtype("<i4")
    assert mapped(lazy.vertices)
    assert not lazy.vertices.flags.writeable
    scaled = (
        lazy.vertices * lazy.global_attrs["las_scale"] + lazy.global_attrs["las_offset"]
    )
    np.testing.assert_allclose(scaled, eager.vertices)
    for name in ("intensity", "gps_time", "height", "user_data", "point_source_id"):
        assert mapped(lazy.vertex_attrs[name]), name
        np.testing.assert_array_equal(lazy.vertex_attrs[name], eager.vertex_attrs[name])
    assert lazy.vertex_attrs["colors"].dtype == np.uint16
    assert mapped(lazy.vertex_attrs["colors"])
    np.testing.assert_allclose(
        lazy.vertex_attrs["colors"] / 65535.0, eager.vertex_attrs["colors"]
    )
    # Bit fields are decoded: small arrays of their own.
    np.testing.assert_array_equal(
        lazy.vertex_attrs["return_number"], eager.vertex_attrs["return_number"]
    )
    np.testing.assert_array_equal(
        lazy.vertex_attrs["classification"], eager.vertex_attrs["classification"]
    )


@pytest.mark.parametrize("point_format", [0, 2, 7])
def test_an_empty_cloud_reads_lazily_in_every_format(tmp_path, point_format) -> None:
    path = tmp_path / "empty.las"
    polyxios.write(make_polydata(np.empty((0, 3)), []), path, point_format=point_format)
    back = polyxios.read(path, lazy=True)
    assert back.vertices.shape == (0, 3)
    if point_format:
        assert back.vertex_attrs["colors"].shape == (0, 3)


def test_a_lazy_read_leaves_a_scaled_extra_byte_raw_with_a_warning(tmp_path) -> None:
    data = _write(_cloud(height=np.array([1, 2, 3, 4], np.int16)))
    vlr_at = _HEADER_1_2 + 54
    data = _patched(data, "<B", vlr_at + 3, 0x08)
    data = _patched(data, "<d", vlr_at + 112, 0.5)
    path = tmp_path / "c.las"
    path.write_bytes(data)
    with pytest.warns(UserWarning, match="height"):
        lazy = polyxios.read(path, lazy=True)
    np.testing.assert_array_equal(lazy.vertex_attrs["height"], [1, 2, 3, 4])
    assert lazy.vertex_attrs["height"].dtype == np.int16


def test_a_lazily_read_cloud_writes_back_bit_for_bit(tmp_path) -> None:
    poly = _cloud(height=np.arange(4, dtype=np.float32))
    path = tmp_path / "c.las"
    polyxios.write(poly, path, scale=0.001, offset=(1.0, 2.0, 3.0))
    lazy = polyxios.read(path, lazy=True)
    again = tmp_path / "again.las"
    polyxios.write(lazy, again)
    assert again.read_bytes() == path.read_bytes()
    np.testing.assert_allclose(polyxios.read(again).vertices, poly.vertices, atol=1e-9)


def test_a_lazily_read_cloud_is_rescaled_when_asked_for_another_scale(tmp_path):
    poly = _cloud()
    path = tmp_path / "c.las"
    polyxios.write(poly, path, scale=0.001, offset=(1.0, 2.0, 3.0))
    lazy = polyxios.read(path, lazy=True)
    coarse = _write(lazy, scale=0.01)
    np.testing.assert_allclose(_read(coarse).vertices, poly.vertices, atol=0.005)
    np.testing.assert_array_equal(_read(coarse).global_attrs["las_scale"], [0.01] * 3)
    shifted = _write(lazy, offset=0.0)
    np.testing.assert_allclose(_read(shifted).vertices, poly.vertices, atol=1e-9)
    np.testing.assert_array_equal(_read(shifted).global_attrs["las_offset"], [0.0] * 3)


def test_integer_vertices_without_the_file_globals_are_coordinates() -> None:
    ints = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    poly = point_cloud(vertices=ints, vertex_attrs={})
    np.testing.assert_allclose(_read(_write(poly)).vertices, ints)
    poly.global_attrs["las_scale"] = np.array([0.001] * 3)
    np.testing.assert_allclose(_read(_write(poly)).vertices, ints)
    poly.global_attrs["las_offset"] = np.zeros(3)
    np.testing.assert_allclose(_read(_write(poly)).vertices, ints * 0.001)


def test_stored_integers_past_an_int32_are_refused() -> None:
    poly = point_cloud(
        vertices=np.array([[2**31, 0, 0]], np.int64),
        vertex_attrs={},
        global_attrs={"las_scale": [0.001] * 3, "las_offset": [0.0] * 3},
    )
    with pytest.raises(CodecError, match="int64"):
        _write(poly)


def test_a_lazy_read_of_a_buffer_is_refused() -> None:
    with pytest.raises(LazyReadError):
        _read(_write(_cloud()), lazy=True)


def test_a_lazy_read_that_is_refused_closes_its_mapping(tmp_path, monkeypatch) -> None:
    import polyxios._io as pio

    mappings: list = []
    real = pio.map_read

    def recording(*args, **kwargs):
        mappings.append(real(*args, **kwargs))
        return mappings[-1]

    monkeypatch.setattr(pio, "map_read", recording)
    bad = tmp_path / "bad.las"
    bad.write_bytes(_patched(_write(_cloud()), "<H", 94, 100))
    with pytest.raises(CodecError, match="header size"):
        polyxios.read(bad, lazy=True)
    assert len(mappings) == 1 and mappings[0].closed


# ---------------------------------------------------------------------------
# LAZ
# ---------------------------------------------------------------------------


def _laz(poly: PolyData, **opts) -> bytes:
    buf = io.BytesIO()
    polyxios.write(poly, buf, fmt=".laz", compress=True, **opts)
    return buf.getvalue()


def test_a_laz_write_is_smaller_flags_the_format_and_reads_back(tmp_path) -> None:
    pytest.importorskip("lazrs")
    poly = _cloud(64, height=np.linspace(0, 1, 64))
    data = _laz(poly, scale=0.001)
    assert data[104] == 3 | 0x80
    assert _u32(data, 100) == 2
    back = _read(data)
    np.testing.assert_allclose(back.vertices, poly.vertices, atol=1e-9)
    np.testing.assert_array_equal(
        back.vertex_attrs["height"], poly.vertex_attrs["height"]
    )
    np.testing.assert_allclose(back.vertex_attrs["colors"], poly.vertex_attrs["colors"])
    assert back.global_attrs["las_point_format"] == 3
    assert "las_vlrs" not in back.global_attrs
    path = tmp_path / "c.laz"
    polyxios.write(poly, path, scale=0.001)
    assert path.read_bytes()[104] == 3 | 0x80
    plain = tmp_path / "c.las"
    polyxios.write(poly, plain, scale=0.001)
    assert len(path.read_bytes()) < len(plain.read_bytes())


def test_compress_overrides_the_suffix(tmp_path) -> None:
    pytest.importorskip("lazrs")
    poly = _cloud()
    path = tmp_path / "c.laz"
    polyxios.write(poly, path, compress=False)
    assert path.read_bytes()[104] == 3
    path = tmp_path / "c.las"
    polyxios.write(poly, path, compress=True)
    assert path.read_bytes()[104] == 3 | 0x80
    assert polyxios.read(path).vertices.shape == (4, 3)


def test_a_laz_read_hands_back_views_of_the_one_decompressed_block() -> None:
    pytest.importorskip("lazrs")
    back = _read(_laz(_cloud(height=np.arange(4, dtype=np.float32))))
    attrs = back.vertex_attrs
    assert np.may_share_memory(attrs["intensity"], attrs["gps_time"])
    assert np.may_share_memory(attrs["intensity"], attrs["height"])
    np.testing.assert_array_equal(attrs["height"], [0, 1, 2, 3])


def test_a_laz_file_of_no_points_reads_back() -> None:
    pytest.importorskip("lazrs")
    poly = make_polydata(np.empty((0, 3)), [])
    assert _read(_laz(poly)).vertices.shape == (0, 3)


def test_a_lazy_read_of_a_laz_file_is_refused(tmp_path) -> None:
    pytest.importorskip("lazrs")
    path = tmp_path / "c.laz"
    polyxios.write(_cloud(), path)
    with pytest.raises(LazyReadError, match="compressed"):
        polyxios.read(path, lazy=True)


def test_a_lazy_read_of_a_laz_file_closes_its_mapping(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lazrs")
    import polyxios._io as pio

    mappings: list = []
    real = pio.map_read

    def recording(*args, **kwargs):
        mappings.append(real(*args, **kwargs))
        return mappings[-1]

    monkeypatch.setattr(pio, "map_read", recording)
    path = tmp_path / "c.laz"
    polyxios.write(_cloud(), path)
    with pytest.raises(LazyReadError, match="compressed"):
        polyxios.read(path, lazy=True)
    assert len(mappings) == 1 and mappings[0].closed


def test_a_laz_file_with_no_laszip_vlr_is_refused() -> None:
    pytest.importorskip("lazrs")
    data = _laz(_cloud())
    # Drop the LASzip VLR by renaming its record id.
    at = _HEADER_1_2
    while _u16(data, at + 18) != 22204:
        at += 54 + _u16(data, at + 20)
    with pytest.raises(CodecError, match="laszip"):
        _read(_patched(data, "<H", at + 18, 1))


def test_a_laz_stream_that_ends_early_is_refused() -> None:
    pytest.importorskip("lazrs")
    data = _laz(_cloud(64))
    with pytest.raises(CodecError, match="LAZ"):
        _read(data[:-40])


def test_a_laz_count_past_its_chunk_table_is_refused_before_allocating(monkeypatch):
    pytest.importorskip("lazrs")
    data = _patched(_laz(_cloud()), "<I", 107, 400_000_000)

    def no_allocation(*args):
        raise AssertionError("allocated the point records from the forged count")

    monkeypatch.setattr(_las, "bytearray", no_allocation, raising=False)
    with pytest.raises(CodecError, match="400000000 points.*chunk table"):
        _read(data)


def test_a_laz_chunk_table_is_bounded_per_chunk_and_by_the_bytes_that_follow(
    monkeypatch,
) -> None:
    lazrs = pytest.importorskip("lazrs")
    chunks: list[tuple[int, int]] = []
    forged = types.SimpleNamespace(
        LazVlr=lazrs.LazVlr,
        LazrsError=lazrs.LazrsError,
        LasZipDecompressor=lazrs.LasZipDecompressor,
        read_chunk_table=lambda stream, vlr: chunks,
    )
    data = _laz(_cloud())
    monkeypatch.setattr(_las, "_lazrs", lambda: (forged, True))

    def no_allocation(*args):
        raise AssertionError("allocated the point records from the forged table")

    monkeypatch.setattr(_las, "bytearray", no_allocation, raising=False)
    chunks[:] = [(40_000, 8)]
    with pytest.raises(CodecError, match="40000 points.*chunk table.*8192"):
        _read(_patched(data, "<I", 107, 40_000))
    chunks[:] = [(4, 10**6)]
    with pytest.raises(CodecError, match="chunk table.*1000000 bytes"):
        _read(data)


def test_a_laz_stream_is_expanded_one_piece_at_a_time(monkeypatch) -> None:
    pytest.importorskip("lazrs")
    n = 300
    poly = _cloud(n, height=np.random.default_rng(2).uniform(0, 1, n))
    data = _laz(poly)
    header = _las._parse_header(data, name="x")
    sizes: list[int] = []
    real = bytearray

    def counting(*args):
        sizes.append(args[0] if args and isinstance(args[0], int) else 0)
        return real(*args)

    monkeypatch.setattr(_las, "bytearray", counting, raising=False)
    monkeypatch.setattr(_las, "_LAZ_PIECE_POINTS", 1000)
    with pytest.raises(CodecError, match="40000 points.*runs dry past point"):
        _read(_patched(data, "<I", 107, 40_000))
    assert max(sizes) <= 1000 * header.record_length
    monkeypatch.setattr(_las, "_LAZ_PIECE_POINTS", 7)
    back = _read(data)
    np.testing.assert_array_equal(
        back.vertex_attrs["height"], poly.vertex_attrs["height"]
    )
    np.testing.assert_allclose(back.vertices, poly.vertices, atol=1e-9)


def test_a_laz_file_of_several_chunks_reads_back_whole() -> None:
    pytest.importorskip("lazrs")
    n = 120_000
    rng = np.random.default_rng(5)
    poly = point_cloud(
        vertices=rng.uniform(-100, 100, (n, 3)).round(3),
        vertex_attrs={"intensity": rng.integers(0, 65535, n).astype(np.uint16)},
    )
    plain = _read(_write(poly))
    back = _read(_laz(poly))
    np.testing.assert_array_equal(back.vertices, plain.vertices)
    np.testing.assert_array_equal(
        back.vertex_attrs["intensity"], plain.vertex_attrs["intensity"]
    )


def test_a_laz_extra_bytes_vlr_is_checked_before_the_stream_is_expanded(
    monkeypatch,
) -> None:
    pytest.importorskip("lazrs")
    data = _laz(_cloud(64, height=np.zeros(64)))
    header = _las._parse_header(data, name="x")
    at = _HEADER_1_2
    for vlr in header.vlrs:
        if vlr.record_id == 4:
            break
        at += 54 + len(vlr.data)
    data = _patched(data, "B", at + 54 + 2, 31)

    def no_allocation(*args):
        raise AssertionError("expanded the stream before checking the extra bytes")

    monkeypatch.setattr(_las, "bytearray", no_allocation, raising=False)
    with pytest.raises(CodecError, match="height.*data type 31"):
        _read(data)


def test_a_laz_path_is_decompressed_from_its_mapping_without_a_copy(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("lazrs")
    path = tmp_path / "c.laz"
    polyxios.write(_cloud(), path)

    def no_copy(*args):
        raise AssertionError("copied the mapped file into a BytesIO")

    monkeypatch.setattr(_las, "io", types.SimpleNamespace(BytesIO=no_copy))
    assert polyxios.read(path).vertices.shape == (4, 3)


def test_the_mapped_stream_answers_seek_with_the_position_for_lazrs() -> None:
    lazrs = pytest.importorskip("lazrs")
    data = _laz(_cloud(64))
    header = _las._parse_header(data, name="x")

    class SeekReturnsNone(io.BytesIO):
        def seek(self, offset: int, whence: int = 0) -> None:
            super().seek(offset, whence)

    stream = _las._MappedStream(SeekReturnsNone(data))
    assert stream.seek(header.offset) == header.offset
    assert stream.tell() == header.offset
    laszip = next(v for v in header.vlrs if v.record_id == 22204)
    out = bytearray(64 * header.record_length)
    lazrs.LasZipDecompressor(stream, laszip.data).decompress_many(out)
    assert (
        np.frombuffer(out, dtype="<i4", count=1)[0]
        == struct.unpack_from("<i", _write(_cloud(64)), _HEADER_1_2)[0]
    )


def test_laz_without_lazrs_names_the_extra(monkeypatch) -> None:
    pytest.importorskip("lazrs")
    data = _laz(_cloud())
    monkeypatch.setattr(
        _las, "_lazrs", lambda: (TripWire('pip install "polyxios[laz]"'), False)
    )
    with pytest.raises(MissingPackageError, match=r"polyxios\[laz\]") as info:
        _read(data)
    assert info.value.__context__ is None
    with pytest.raises(MissingPackageError, match=r"polyxios\[laz\]"):
        _laz(_cloud())


def test_a_laz_named_file_with_plain_records_reads_as_plain(tmp_path) -> None:
    path = tmp_path / "c.laz"
    path.write_bytes(_write(_cloud()))
    assert polyxios.read(path).vertices.shape == (4, 3)


def test_an_unnamed_buffer_writes_plain_unless_asked_to_compress() -> None:
    buf = io.BytesIO()
    polyxios.write(_cloud(), buf, fmt=".laz")
    assert buf.getvalue()[104] == 3


# ---------------------------------------------------------------------------
# Interoperability with a reference implementation
# ---------------------------------------------------------------------------


def _interop_cloud() -> PolyData:
    n = 16
    rng = np.random.default_rng(3)
    poly = _cloud(
        n,
        height=np.linspace(0, 1, n, dtype=np.float32),
        normals=rng.uniform(-1, 1, (n, 3)),
        count=np.arange(n, dtype=np.int32),
        scan_angle_rank=np.resize([-10, 0, 10], n).astype(np.int8),
    )
    poly.global_attrs["crs_wkt"] = 'GEOGCS["WGS 84"]'
    poly.global_attrs["las_geokeys"] = {
        1024: 2,
        2048: 4326,
        2049: "WGS 84",
        2057: 6378137.0,
        2059: [0.5, 1.5],
    }
    return poly


@pytest.mark.parametrize("compress", [False, True])
def test_a_written_file_is_read_by_a_reference_implementation(compress) -> None:
    laspy = pytest.importorskip("laspy")
    if compress:
        pytest.importorskip("lazrs")
    poly = _interop_cloud()
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polyxios.write(poly, buf, fmt=".las", compress=compress, version="1.4")
    buf.seek(0)
    las = laspy.read(buf)
    assert las.header.are_points_compressed is compress
    assert las.header.point_format.id == 3
    np.testing.assert_allclose(
        np.column_stack([las.x, las.y, las.z]), poly.vertices, atol=1e-9
    )
    attrs = poly.vertex_attrs
    np.testing.assert_array_equal(las.intensity, attrs["intensity"])
    np.testing.assert_array_equal(las.classification, attrs["classification"])
    np.testing.assert_array_equal(las.return_number, attrs["return_number"])
    np.testing.assert_array_equal(las.scan_angle_rank, attrs["scan_angle_rank"])
    np.testing.assert_array_equal(las.gps_time, attrs["gps_time"])
    np.testing.assert_array_equal(
        np.column_stack([las.red, las.green, las.blue]), attrs["colors"] * 65535
    )
    assert list(las.point_format.extra_dimension_names) == [
        "height",
        "normals",
        "count",
    ]
    assert las["height"].dtype == np.float32
    np.testing.assert_array_equal(las["height"], attrs["height"])
    np.testing.assert_array_equal(las["normals"], attrs["normals"])
    np.testing.assert_array_equal(las["count"], attrs["count"])
    by_id = {v.record_id: v for v in las.vlrs}
    assert by_id[2112].string == 'GEOGCS["WGS 84"]'
    assert las.header.global_encoding.wkt
    keys = {
        k.id: (k.tiff_tag_location, k.count, k.value_offset)
        for k in by_id[34735].geo_keys
    }
    assert keys == {
        1024: (0, 1, 2),
        2048: (0, 1, 4326),
        2049: (34737, 7, 0),
        2057: (34736, 1, 0),
        2059: (34736, 2, 1),
    }
    assert by_id[34737].strings == ["WGS 84|"]
    assert [d.value for d in by_id[34736].doubles] == [6378137.0, 0.5, 1.5]


def test_a_new_style_file_without_a_crs_is_read_by_a_reference_implementation():
    laspy = pytest.importorskip("laspy")
    poly = _cloud(scan_angle=np.arange(4, dtype=np.int16) * 100)
    buf = io.BytesIO()
    polyxios.write(poly, buf, fmt=".las")
    buf.seek(0)
    las = laspy.read(buf)
    assert las.header.point_format.id == 7
    assert not las.header.global_encoding.wkt
    np.testing.assert_array_equal(las.scan_angle, [0, 100, 200, 300])


@pytest.mark.parametrize("compress", [False, True])
def test_a_file_from_a_reference_implementation_reads_back(compress) -> None:
    laspy = pytest.importorskip("laspy")
    if compress:
        pytest.importorskip("lazrs")
    n = 8
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [100.0, 200.0, 300.0]
    header.add_extra_dim(laspy.ExtraBytesParams(name="height", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="triple", type="3u1"))
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="temp",
            type=np.int16,
            scales=np.array([0.5]),
            offsets=np.array([10.0]),
        )
    )
    header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr('GEOGCS["WGS 84"]'))
    header.global_encoding.wkt = True
    geokeys = laspy.vlrs.known.GeoKeyDirectoryVlr()
    geokeys.geo_keys = [
        laspy.vlrs.known.GeoKeyEntryStruct(
            id=1024, tiff_tag_location=0, count=1, value_offset=2
        ),
        laspy.vlrs.known.GeoKeyEntryStruct(
            id=2048, tiff_tag_location=0, count=1, value_offset=4326
        ),
    ]
    geokeys.geo_keys_header.number_of_keys = 2
    header.vlrs.append(geokeys)
    las = laspy.LasData(header)
    las.x = np.arange(n) * 1.5 + 100
    las.y = np.arange(n) * 2.0 + 200
    las.z = np.arange(n) * 0.25 + 300
    las.intensity = np.arange(n) * 10
    las.classification = np.full(n, 2, np.uint8)
    las.gps_time = np.linspace(0, 1, n)
    las.scan_angle = np.arange(n) * 100
    las.return_number = np.ones(n, np.uint8)
    las.number_of_returns = np.full(n, 2, np.uint8)
    las["height"] = np.linspace(0, 1, n).astype(np.float32)
    las["triple"] = np.tile([1, 2, 3], (n, 1))
    las["temp"] = np.arange(n) * 0.5 + 10
    buf = io.BytesIO()
    las.write(buf, do_compress=compress)
    back = _read(buf.getvalue())
    attrs = back.vertex_attrs
    g = back.global_attrs
    assert (g["las_version"], g["las_point_format"]) == ("1.4", 6)
    np.testing.assert_allclose(
        back.vertices, np.column_stack([las.x, las.y, las.z]), atol=1e-9
    )
    np.testing.assert_array_equal(g["las_scale"], [0.01, 0.01, 0.01])
    np.testing.assert_array_equal(g["las_offset"], [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(attrs["intensity"], las.intensity)
    np.testing.assert_array_equal(attrs["classification"], 2)
    np.testing.assert_array_equal(attrs["number_of_returns"], 2)
    np.testing.assert_array_equal(attrs["scan_angle"], np.arange(n) * 100)
    np.testing.assert_array_equal(attrs["gps_time"], las.gps_time)
    assert attrs["height"].dtype == np.float32
    np.testing.assert_array_equal(attrs["height"], las["height"])
    np.testing.assert_array_equal(attrs["triple"], np.tile([1, 2, 3], (n, 1)))
    np.testing.assert_allclose(attrs["temp"], np.arange(n) * 0.5 + 10)
    assert g["crs_wkt"] == 'GEOGCS["WGS 84"]'
    assert g["las_geokeys"] == {1024: 2, 2048: 4326}
    assert "las_vlrs" not in g
    assert g["las_generating_software"].startswith("laspy")
