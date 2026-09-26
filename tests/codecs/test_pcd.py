from __future__ import annotations

import io
import struct
import warnings

import numpy as np
import pytest

import polyxios
from polyxios._types import PolyData, make_polydata
from polyxios.codecs import _lzf_fallback, _pcd
from polyxios.exceptions import CodecError, LazyReadError, ValidationError
from tests.codecs._lazy import mapped

try:
    from polyxios.codecs import _lzf
except ImportError:  # pragma: no cover - a pure-Python install
    _lzf = None

_IMPLS = [m for m in (_lzf, _lzf_fallback) if m is not None]
_needs_compiled = pytest.mark.skipif(_lzf is None, reason="compiled _lzf not built")

# A cloud the way the PCL tutorial lays one out, ascii, with a packed colour
# spelled the way PCL writes it today: 4210752 is 0x404040, grey, as the
# integer's decimal value.
_PCL_ASCII = """\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
0.93773 0.33763 0 4210752
0.90805 0.35641 0 4210752
nan nan nan 4210752
"""


def _cloud(n: int = 4) -> PolyData:
    rng = np.random.default_rng(1)
    verts = rng.uniform(-1, 1, (n, 3))
    return make_polydata(
        verts,
        [("vertex", np.arange(n)[:, None])],
        vertex_attrs={
            "normals": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "colors": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)[:n],
            "intensity": np.arange(n, dtype=np.uint16),
            "curvature": np.linspace(0, 1, n, dtype=np.float32),
            "moments": np.arange(2 * n, dtype=np.float64).reshape(n, 2),
            "label": np.arange(n, dtype=np.int64),
        },
    )


def _write(poly: PolyData, **opts) -> bytes:
    buf = io.BytesIO()
    polyxios.write(poly, buf, fmt=".pcd", **opts)
    return buf.getvalue()


def _read(data: bytes, **opts) -> PolyData:
    return polyxios.read(io.BytesIO(data), fmt=".pcd", **opts)


def _header(data: bytes) -> dict[str, str]:
    head = data.split(b"DATA")[0].decode()
    return dict(
        line.split(" ", 1) for line in head.splitlines() if not line.startswith("#")
    )


# ---------------------------------------------------------------------------
# LZF
# ---------------------------------------------------------------------------


@pytest.fixture(params=_IMPLS, ids=[m.__name__.rsplit(".", 1)[-1] for m in _IMPLS])
def lzf(request, monkeypatch):
    monkeypatch.setattr(_pcd, "lzf", request.param)
    return request.param


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"a",
        b"abc" * 50,
        bytes(range(256)) * 40,
        b"\x00" * 5000,
        bytes(range(256)) * 300,
        b"abc\x00abc\x01",
        b"\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x80?\x00\x00\x00@",
    ],
    ids=[
        "empty",
        "one",
        "abc",
        "ramp",
        "zeros",
        "long-ramp",
        "tail-match",
        "tail-floats",
    ],
)
@_needs_compiled
def test_lzf_round_trips_and_both_implementations_agree(blob: bytes) -> None:
    fast, slow = _lzf.compress(blob), _lzf_fallback.compress(blob)
    assert fast == slow
    assert _lzf.decompress(fast, len(blob)) == blob
    assert _lzf_fallback.decompress(fast, len(blob)) == blob


def test_lzf_a_match_at_the_tail_is_never_a_two_byte_reference(lzf) -> None:
    """A back-reference shorter than three bytes has a control byte below 32,
    which every decoder reads as a literal run; the match has to be three."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        blob = rng.integers(0, 3, int(rng.integers(4, 40)), dtype=np.uint8).tobytes()
        packed = lzf.compress(blob)
        for impl in _IMPLS:
            assert impl.decompress(packed, len(blob)) == blob


def test_lzf_random_bytes_grow_by_at_most_one_in_32(lzf) -> None:
    blob = np.random.default_rng(0).integers(0, 256, 70_000, dtype=np.uint8).tobytes()
    packed = lzf.compress(blob)
    assert len(packed) <= len(blob) + len(blob) // 32 + 8
    assert lzf.decompress(packed, len(blob)) == blob


def test_lzf_streams_cross_with_liblzf(lzf) -> None:
    """liblzf decodes what is written here and the reverse; the streams
    themselves differ, since the two hash functions pick different matches."""
    liblzf = pytest.importorskip("lzf")
    rng = np.random.default_rng(0)
    blobs = [
        b"abc" * 50,
        bytes(range(256)) * 300,
        rng.integers(0, 256, 20_000, dtype=np.uint8).tobytes(),
        np.sin(np.arange(30_000) / 7).astype(np.float32).tobytes(),
    ]
    for blob in blobs:
        ours = lzf.compress(blob)
        theirs = liblzf.compress(blob, len(blob) + len(blob) // 32 + 64)
        assert liblzf.decompress(ours, len(blob)) == blob
        assert lzf.decompress(theirs, len(blob)) == blob


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (b"\x20\x00", "before the output start"),
        (b"\x05ab", "inside a literal run"),
        (b"\xe0", "inside a back-reference"),
        (b"\x00a", "expands to 1 bytes, the header promised 5"),
        (b"\x05abcdef", "expands past the 5 bytes the header promised"),
    ],
)
def test_lzf_malformed_stream_is_a_codec_error(
    lzf, stream: bytes, message: str
) -> None:
    with pytest.raises(CodecError, match=message):
        lzf.decompress(stream, 5)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_format", ["ascii", "binary", "binary_compressed"])
def test_round_trip_keeps_every_field(lzf, data_format: str) -> None:
    poly = _cloud()
    back = _read(_write(poly, data_format=data_format))

    np.testing.assert_allclose(back.vertices, poly.vertices, rtol=1e-6)
    assert back.element_types.size == 0
    for key, value in poly.vertex_attrs.items():
        np.testing.assert_allclose(back.vertex_attrs[key], value, rtol=1e-6)
    assert back.vertex_attrs["intensity"].dtype == np.uint16
    assert back.vertex_attrs["label"].dtype == np.int64
    assert back.vertex_attrs["curvature"].dtype == np.float32
    assert back.global_attrs == {}


def test_a_tiny_cloud_survives_binary_compressed(lzf) -> None:
    """Two points whose last bytes repeat an earlier run: the match starts
    three bytes from the block's end."""
    verts = np.array([[2, 2, 1], [2, 2, 2]], dtype=np.float32)
    poly = make_polydata(verts, [("vertex", np.arange(2)[:, None])])
    back = _read(_write(poly, data_format="binary_compressed"))
    np.testing.assert_array_equal(back.vertices, verts)


def test_a_one_column_attribute_writes_as_count_one_and_reads_back_flat() -> None:
    poly = _cloud()
    poly.vertex_attrs["weight"] = np.arange(4.0)[:, None]
    data = _write(poly)
    assert _header(data)["COUNT"].endswith(" 1")
    np.testing.assert_array_equal(_read(data).vertex_attrs["weight"], np.arange(4.0))


def test_a_scalar_attribute_is_a_codec_error_not_an_index_error() -> None:
    poly = _cloud()
    poly.vertex_attrs["bad"] = np.float64(1.0)
    with pytest.raises(CodecError, match="has shape \\(\\) for 4 points"):
        _write(poly)


def test_an_attribute_named_like_a_folded_field_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.vertex_attrs["rgb"] = np.zeros(4, dtype=np.float32)
    poly.vertex_attrs["normal_x"] = np.zeros(4, dtype=np.float32)
    with pytest.warns(UserWarning, match="already spells a field 'rgb'"):
        with pytest.warns(UserWarning, match="already spells a field 'normal_x'"):
            data = _write(poly)
    assert _header(data)["FIELDS"].split().count("rgb") == 1
    back = _read(data)
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], _cloud().vertex_attrs["colors"]
    )
    assert "rgb" not in back.vertex_attrs


def test_a_nan_colour_writes_as_black_without_a_warning() -> None:
    poly = _cloud()
    poly.vertex_attrs["colors"][0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = _read(_write(poly))
    np.testing.assert_array_equal(back.vertex_attrs["colors"][0], [0, 0, 0])


def test_the_viewpoint_keeps_its_full_precision() -> None:
    poly = _cloud()
    pose = [0.123456789, -1e-7, 2.5, 0.7071067811865476, 0, 0, 0.7071067811865476]
    poly.global_attrs["pcd_viewpoint"] = pose
    data = _write(poly)
    np.testing.assert_array_equal(_read(data).global_attrs["pcd_viewpoint"], pose)


def test_the_header_spells_the_fields_the_way_pcl_does() -> None:
    head = _header(_write(_cloud()))
    assert head["VERSION"] == "0.7"
    assert head["FIELDS"] == (
        "x y z normal_x normal_y normal_z rgb intensity curvature moments label"
    )
    assert head["SIZE"] == "4 4 4 4 4 4 4 2 4 8 8"
    assert head["TYPE"] == "F F F F F F F U F F I"
    assert head["COUNT"] == "1 1 1 1 1 1 1 1 1 2 1"
    assert head["WIDTH"] == "4"
    assert head["HEIGHT"] == "1"
    assert head["VIEWPOINT"] == "0 0 0 1 0 0 0"
    assert head["POINTS"] == "4"


def test_double_writes_eight_byte_coordinates_and_normals() -> None:
    data = _write(_cloud(), double=True)
    assert _header(data)["SIZE"].startswith("8 8 8 8 8 8")
    np.testing.assert_array_equal(_read(data).vertices, _cloud().vertices)


def test_double_spells_an_ascii_body_at_every_digit_unless_float_fmt_says() -> None:
    """``F 8`` buys nothing in an ascii body spelled at ten digits, so
    ``double`` raises the default to ``.17g``; a ``float_fmt`` given out
    loud is used as is."""
    poly = make_polydata(
        np.array([[0.1234567890123456, 0, 0]]), [("vertex", np.array([[0]]))]
    )
    poly.vertex_attrs["normals"] = np.array([[0, 0, 1 / 3]])

    def body(**opts) -> bytes:
        return _write(poly, data_format="ascii", **opts).split(b"DATA ascii\n")[1]

    assert body() == b"0.123456789 0 0 0 0 0.3333333333\n"
    assert body(double=True) == b"0.12345678901234559 0 0 0 0 0.33333333333333331\n"
    assert body(double=True, float_fmt=".5g") == b"0.12346 0 0 0 0 0.33333\n"
    back = _read(_write(poly, data_format="ascii", double=True))
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.vertex_attrs["normals"], [[0, 0, 1 / 3]])


def test_the_pcl_tutorial_float_spelling_reads_as_the_float_bits() -> None:
    """The tutorial file spells ``rgb`` as ``4.2108e+06``, the float itself
    at six digits as PCL before 1.8 printed it. That is not the decimal
    value 4210800: its float32 bits are 0x4a8080e0, so the colour is
    (128, 128, 224), what that PCL and every float-bits reader give."""
    back = _read(_PCL_ASCII.replace("4210752", "4.2108e+06").encode())
    assert back.vertices.shape == (3, 3)
    assert np.isnan(back.vertices[2]).all()
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], [[128 / 255, 128 / 255, 224 / 255]] * 3
    )
    # Written back in ascii, the colour is PCL's decimal spelling of the bits.
    line = _write(back, data_format="ascii").decode().splitlines()[-3]
    assert line.split()[-1] == str(0x8080E0)
    # The same digits as a bare integer are the decimal value, grey.
    grey = _read(_PCL_ASCII.replace("4210752", "4210800").encode())
    np.testing.assert_allclose(
        grey.vertex_attrs["colors"], [[64 / 255] * 2 + [0x70 / 255]] * 3
    )


def test_a_four_channel_colour_goes_out_as_rgba_and_comes_back() -> None:
    poly = _cloud()
    poly.vertex_attrs["colors"] = np.column_stack(
        [poly.vertex_attrs["colors"], np.array([1.0, 0.5, 0.0, 1.0])]
    )
    data = _write(poly)
    assert "rgba" in _header(data)["FIELDS"]
    assert _header(data)["TYPE"].split()[6] == "U"
    back = _read(data)
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], poly.vertex_attrs["colors"], atol=1 / 255
    )


def test_an_organised_cloud_keeps_its_grid_and_viewpoint() -> None:
    poly = _cloud()
    poly.global_attrs.update(
        {"pcd_width": 2, "pcd_height": 2, "pcd_viewpoint": [1, 2, 3, 1, 0, 0, 0]}
    )
    data = _write(poly)
    head = _header(data)
    assert (head["WIDTH"], head["HEIGHT"]) == ("2", "2")
    assert head["VIEWPOINT"] == "1 2 3 1 0 0 0"
    back = _read(data)
    assert back.global_attrs["pcd_width"] == 2
    assert back.global_attrs["pcd_height"] == 2
    np.testing.assert_array_equal(
        back.global_attrs["pcd_viewpoint"], [1, 2, 3, 1, 0, 0, 0]
    )


def test_a_grid_that_does_not_match_the_count_is_written_flat_with_a_warning() -> None:
    poly = _cloud()
    poly.global_attrs.update({"pcd_width": 3, "pcd_height": 3})
    with pytest.warns(UserWarning, match="is not the point count 4"):
        data = _write(poly)
    assert _header(data)["WIDTH"] == "4"


@pytest.mark.parametrize("key", ["pcd_width", "pcd_height"])
def test_half_a_grid_is_written_flat_with_a_warning(key: str) -> None:
    poly = _cloud()
    poly.global_attrs[key] = 4
    with pytest.warns(UserWarning, match=f"{key} without its pair"):
        data = _write(poly)
    assert (_header(data)["WIDTH"], _header(data)["HEIGHT"]) == ("4", "1")


def test_a_padding_field_is_skipped_in_every_layout(lzf, tmp_path) -> None:
    head = (
        "VERSION 0.7\nFIELDS x y z _ i\nSIZE 4 4 4 4 2\nTYPE F F F U U\n"
        "COUNT 1 1 1 1 1\nWIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\n"
    )
    record = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("pad", "V4"), ("i", "<u2")]
    )
    table = np.zeros(2, dtype=record)
    table["x"] = [1, 2]
    table["i"] = [7, 9]
    binary = (head + "DATA binary\n").encode() + table.tobytes()
    ascii_ = (head + "DATA ascii\n").encode() + b"1 0 0 7\n2 0 0 9\n"
    plain = (
        table["x"].tobytes()
        + table["y"].tobytes()
        + table["z"].tobytes()
        + table["i"].tobytes()
    )
    packed = lzf.compress(plain)
    compressed = (
        (head + "DATA binary_compressed\n").encode()
        + struct.pack("<II", len(packed), len(plain))
        + packed
    )
    path = tmp_path / "pad.pcd"
    for data in (binary, ascii_, compressed):
        path.write_bytes(data)
        # A buffer is read into memory, a path is mapped; both go through
        # every layout's decoder.
        for back in (_read(data), polyxios.read(path)):
            np.testing.assert_array_equal(back.vertices[:, 0], [1, 2])
            np.testing.assert_array_equal(back.vertex_attrs["i"], [7, 9])
            assert set(back.vertex_attrs) == {"i"}
    path.write_bytes(binary)
    lazy = polyxios.read(path, lazy=True)
    assert set(lazy.vertex_attrs) == {"i"}
    np.testing.assert_array_equal(lazy.vertex_attrs["i"], [7, 9])


@pytest.mark.parametrize(
    ("fields", "sizes", "types", "record"),
    [
        (
            "_pad1 _ x y z",
            "2 4 4 4 4",
            "U U F F F",
            [("a", "<u2"), ("pad", "V4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
        ),
        (
            "x y z _ _pad3",
            "4 4 4 4 2",
            "F F F U U",
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("pad", "V4"), ("a", "<u2")],
        ),
    ],
    ids=["before", "after"],
)
def test_a_field_named_like_a_padding_label_keeps_its_own_name(
    lzf, fields: str, sizes: str, types: str, record: list
) -> None:
    """Padding is labelled inside the binary record; a real field spelled
    like that label, before or after the padding, must neither collide
    with it nor be taken for it, and reads the same in every layout."""
    aname = fields.split()[0] if fields.startswith("_pad") else fields.split()[-1]
    head = (
        f"VERSION 0.7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\n"
        "COUNT 1 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\n"
    )
    table = np.zeros(2, dtype=np.dtype(record))
    table["a"] = [3, 4]
    table["x"] = [1, 2]
    binary = (head + "DATA binary\n").encode() + table.tobytes()
    rows = (
        ["3 1 0 0", "4 2 0 0"] if fields.startswith("_pad") else ["1 0 0 3", "2 0 0 4"]
    )
    ascii_ = (head + "DATA ascii\n").encode() + "\n".join(rows).encode() + b"\n"
    plain = b"".join(table[f].tobytes() for f in table.dtype.names if f != "pad")
    packed = lzf.compress(plain)
    compressed = (
        (head + "DATA binary_compressed\n").encode()
        + struct.pack("<II", len(packed), len(plain))
        + packed
    )
    for data in (binary, ascii_, compressed):
        back = _read(data)
        assert set(back.vertex_attrs) == {aname}
        np.testing.assert_array_equal(back.vertex_attrs[aname], [3, 4])
        np.testing.assert_array_equal(back.vertices[:, 0], [1, 2])


@pytest.mark.parametrize("bad", ["my field", "température", "_", ""])
def test_a_name_that_is_not_a_bare_ascii_token_is_dropped_with_a_warning(
    bad: str,
) -> None:
    """FIELDS is a line of ASCII tokens and ``_`` is the padding name, so any
    of these would write a file that does not read back as written."""
    poly = _cloud()
    poly.vertex_attrs[bad] = np.arange(4.0)
    with pytest.warns(UserWarning, match="bare ASCII tokens"):
        data = _write(poly)
    back = _read(data)
    assert bad not in back.vertex_attrs
    assert set(back.vertex_attrs) == set(_cloud().vertex_attrs)


def test_an_eight_byte_integer_survives_ascii_beyond_float_precision() -> None:
    poly = _cloud()
    poly.vertex_attrs["id"] = np.array([2**60 + 1, 5, -(2**62) - 3, 0], np.int64)
    poly.vertex_attrs["big"] = np.array([2**64 - 1, 2**53 + 1, 0, 1], np.uint64)
    back = _read(_write(poly, data_format="ascii"))
    np.testing.assert_array_equal(back.vertex_attrs["id"], poly.vertex_attrs["id"])
    np.testing.assert_array_equal(back.vertex_attrs["big"], poly.vertex_attrs["big"])
    assert back.vertex_attrs["big"].dtype == np.uint64


@pytest.mark.parametrize(
    ("typ", "spelling", "value"),
    [
        ("I", "9007199254740993.0", 2**53 + 1),
        ("I", "9.007199254740993e15", 2**53 + 1),
        ("I", "-9007199254740993.00", -(2**53) - 1),
        ("U", "1.8446744073709551615e19", 2**64 - 1),
    ],
    ids=["I .0", "I exponent", "I negative", "U 8 top"],
)
def test_an_eight_byte_integer_spelled_as_a_float_keeps_every_digit(
    typ: str, spelling: str, value: int
) -> None:
    """Past 2**53 a trailing ``.0`` or an exponent still means the exact
    value the digits spell, not the float64 that rounds it; ``nan`` in the
    same column is 0 as in every other integer field."""
    data = (
        _PCL_ASCII.replace("FIELDS x y z rgb", "FIELDS x y z big")
        .replace("SIZE 4 4 4 4", "SIZE 4 4 4 8")
        .replace("TYPE F F F F", f"TYPE F F F {typ}")
        .replace("4210752", spelling, 1)
        .replace("4210752", "nan", 1)
        .encode()
    )
    back = _read(data)
    assert back.vertex_attrs["big"].tolist() == [value, 0, 4210752]
    fraction = data.replace(spelling.encode(), b"9007199254740993.5")
    with pytest.raises(CodecError, match="'big' is TYPE . but holds a fraction"):
        _read(fraction)


@pytest.mark.parametrize(
    ("typ", "value"),
    [("I", 2**63), ("I", -(2**63) - 1), ("U", 2**64), ("I", "9.3e18"), ("U", "2e19")],
    ids=["I 8 high", "I 8 low", "U 8 high", "I 8 exponent", "U 8 exponent"],
)
def test_an_ascii_integer_past_eight_bytes_is_refused(typ: str, value) -> None:
    """Past 2**53 the float pass cannot tell 2**63 from 2**63 - 1, so the
    token is re-read as an integer, and one that does not fit is refused
    as such rather than escaping numpy's own OverflowError."""
    data = (
        _PCL_ASCII.replace("FIELDS x y z rgb", "FIELDS x y z big")
        .replace("SIZE 4 4 4 4", "SIZE 4 4 4 8")
        .replace("TYPE F F F F", f"TYPE F F F {typ}")
        .replace("4210752", str(value))
        .encode()
    )
    with pytest.raises(CodecError, match="'big' holds a value outside"):
        _read(data)


@pytest.mark.parametrize("data_format", ["ascii", "binary", "binary_compressed"])
def test_a_plain_attribute_named_rgb_that_is_not_four_bytes_stays_one(
    data_format: str,
) -> None:
    """A packed colour is 32 bits; an eight-byte ``rgb`` is a field of its
    own, spelled as its value and read back as one, not as a colour."""
    poly = make_polydata(np.zeros((3, 3)), [("vertex", np.arange(3)[:, None])])
    poly.vertex_attrs["rgb"] = np.array([1.5, 2.5, 3.5])
    poly.vertex_attrs["rgba"] = np.array([1, 2, 3], np.int64)
    data = _write(poly, data_format=data_format)
    assert _header(data)["SIZE"] == "4 4 4 8 8"
    back = _read(data)
    assert "colors" not in back.vertex_attrs
    np.testing.assert_array_equal(back.vertex_attrs["rgb"], [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(back.vertex_attrs["rgba"], [1, 2, 3])


def test_ascii_spells_a_float_from_its_source_not_its_float32_neighbour() -> None:
    poly = make_polydata(np.array([[0.1, 0.2, 0.3]]), [("vertex", np.array([[0]]))])
    poly.vertex_attrs["m"] = np.array([[1.5, 2.5]])
    body = _write(poly, data_format="ascii").split(b"DATA ascii\n")[1]
    assert body == b"0.1 0.2 0.3 1.5 2.5\n"
    back = _read(_write(poly, data_format="ascii"))
    np.testing.assert_array_equal(back.vertices, _read(_write(poly)).vertices)


def test_a_bad_float_fmt_is_refused_as_a_codec_error() -> None:
    with pytest.raises(CodecError, match="float_fmt 'zz' is not usable"):
        _write(_cloud(), data_format="ascii", float_fmt="zz")


@pytest.mark.parametrize("width", ["two", 2.5, np.float32(2.5), float("nan")])
def test_a_grid_that_is_not_whole_numbers_is_written_flat_with_a_warning(
    width,
) -> None:
    """``2.5 x 2`` must not be truncated into the ``2 x 2`` grid of 4 points."""
    poly = _cloud()
    poly.global_attrs.update({"pcd_width": width, "pcd_height": 2})
    with pytest.warns(UserWarning, match="is not a grid of whole numbers"):
        data = _write(poly)
    assert (_header(data)["WIDTH"], _header(data)["HEIGHT"]) == ("4", "1")


@pytest.mark.parametrize(
    "width", [2, np.int64(2), np.uint8(2), 2.0, np.float64(2.0), "2"]
)
def test_a_whole_number_grid_is_accepted_whatever_its_type(width) -> None:
    poly = _cloud()
    poly.global_attrs.update({"pcd_width": width, "pcd_height": 2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = _write(poly)
    assert (_header(data)["WIDTH"], _header(data)["HEIGHT"]) == ("2", "2")


def test_header_comments_blank_lines_and_lower_case_keys_are_accepted() -> None:
    data = _PCL_ASCII.replace("VERSION", "\n# note\nversion").encode()
    assert _read(data).vertices.shape == (3, 3)


def test_ascii_comment_lines_inside_the_body_are_skipped() -> None:
    data = _PCL_ASCII.replace("0.90805", "# mid-body\n0.90805").encode()
    assert _read(data).vertices.shape == (3, 3)


def test_a_cloud_of_no_points_round_trips(lzf) -> None:
    poly = make_polydata(np.empty((0, 3)), [], vertex_attrs={})
    for data_format in ("ascii", "binary", "binary_compressed"):
        back = _read(_write(poly, data_format=data_format))
        assert back.vertices.shape == (0, 3)


@pytest.mark.parametrize("data_format", ["ascii", "binary", "binary_compressed"])
def test_a_cloud_of_no_points_with_colours_round_trips(data_format, lzf) -> None:
    """An empty ascii body is no tokens; the packed rgb re-read has to cope."""
    poly = make_polydata(
        np.empty((0, 3)),
        [],
        vertex_attrs={
            "colors": np.empty((0, 3)),
            "normals": np.empty((0, 3)),
            "big": np.empty((0,), dtype=np.int64),
        },
    )
    back = _read(_write(poly, data_format=data_format))
    assert back.vertices.shape == (0, 3)
    assert back.vertex_attrs["colors"].shape == (0, 3)
    assert back.vertex_attrs["normals"].shape == (0, 3)
    assert back.vertex_attrs["big"].shape == (0,)
    assert back.vertex_attrs["big"].dtype == np.int64


_NO_NEWLINE_HEAD = (
    "VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
    "WIDTH 0\nHEIGHT 1\nPOINTS 0\nDATA "
)


@pytest.mark.parametrize("data_format", ["ascii", "binary"])
def test_no_points_and_no_newline_after_data_reads_empty(data_format, tmp_path) -> None:
    """The body starts at the file end, not one byte past it."""
    data = (_NO_NEWLINE_HEAD + data_format).encode()
    back = _read(data)
    assert back.vertices.shape == (0, 3)
    assert back.vertex_attrs["colors"].shape == (0, 3)
    if data_format == "binary":
        path = tmp_path / "bare.pcd"
        path.write_bytes(data)
        assert polyxios.read(path, lazy=True).vertices.shape == (0, 3)


@pytest.mark.parametrize("body", ["\n", "   \n\n", "  "])
def test_no_points_with_a_blank_body_reads_empty(body) -> None:
    """numpy turns whitespace alone into one number; the reader must not."""
    back = _read((_NO_NEWLINE_HEAD + "ascii\n" + body).encode())
    assert back.vertices.shape == (0, 3)
    assert back.vertex_attrs["colors"].shape == (0, 3)


def test_no_newline_after_data_binary_compressed_lacks_its_size_words() -> None:
    with pytest.raises(CodecError, match="two size words"):
        _read((_NO_NEWLINE_HEAD + "binary_compressed").encode())


# ---------------------------------------------------------------------------
# What is dropped, and how it says so
# ---------------------------------------------------------------------------


def test_elements_that_are_not_points_are_dropped_with_a_warning() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    with pytest.warns(UserWarning, match="points only; 1 elements dropped"):
        back = _read(_write(poly))
    assert back.vertices.shape == (3, 3)


def test_a_text_attribute_is_dropped_with_a_warning() -> None:
    poly = _cloud()
    poly.vertex_attrs["name"] = np.array(["a", "b", "c", "d"])
    with pytest.warns(UserWarning, match="cannot hold vertex attribute 'name'"):
        back = _read(_write(poly))
    assert "name" not in back.vertex_attrs


def test_an_unknown_data_format_is_refused_before_anything_is_written() -> None:
    with pytest.raises(ValueError, match="data_format must be one of"):
        _write(_cloud(), data_format="base64")


# ---------------------------------------------------------------------------
# Malformed files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("FIELDS x y z rgb", "FIELDS x y w rgb"), "lack x, y and z"),
        (("TYPE F F F F", "TYPE F F F Q"), "which PCD does not define"),
        (("SIZE 4 4 4 4", "SIZE 4 4 4 3"), "which PCD does not define"),
        (("SIZE 4 4 4 4", "SIZE 4 4 4"), "SIZE lists 3 entries for 4 fields"),
        (("WIDTH 3", "WIDTH 2"), "WIDTH 2 x HEIGHT 1 is not POINTS 3"),
        (("DATA ascii", "DATA base64"), "is not ascii, binary or binary_compressed"),
        (("VIEWPOINT 0 0 0 1 0 0 0", "VIEWPOINT 0 0 0"), "VIEWPOINT needs 7 numbers"),
        (("WIDTH 3\nHEIGHT 1", "WIDTH -3\nHEIGHT -1"), "cannot be negative"),
        (("POINTS 3\n", "POINTS 3\nHEIGHT 1\nWIDTH 3\n"), None),
    ],
)
def test_a_bad_header_names_what_is_wrong(edit, message) -> None:
    data = _PCL_ASCII.replace(*edit).encode()
    if message is None:
        assert _read(data).vertices.shape == (3, 3)
        return
    with pytest.raises(CodecError, match=message):
        _read(data)


def test_a_header_without_a_data_line_is_refused() -> None:
    with pytest.raises(CodecError, match="has no DATA line"):
        _read(_PCL_ASCII.split("DATA")[0].encode())


def test_ascii_with_too_few_numbers_is_refused() -> None:
    data = _PCL_ASCII.replace("nan nan nan 4210752\n", "").encode()
    with pytest.raises(
        CodecError, match="POINTS 3 with 4 values each is 12 numbers, the body holds 8"
    ):
        _read(data)


def test_ascii_with_a_value_its_type_cannot_hold_is_refused() -> None:
    data = _PCL_ASCII.replace("TYPE F F F F", "TYPE F F F U").replace(
        "SIZE 4 4 4 4", "SIZE 4 4 4 1"
    )
    with pytest.raises(CodecError, match="'rgb' holds a value outside 0..255"):
        _read(data.encode())


def test_ascii_rgb_spelled_signed_or_as_a_float_reads_the_same_colour() -> None:
    """PCL today spells the packed colour unsigned, older releases signed,
    and some writers spell the float whose bits hold it; all three are the
    one colour, and NaN is black. Alpha 0xFF makes the float a whole number
    near -2.5e38, alpha 0 a subnormal with a fraction; both are float bits.
    The field's TYPE may be F, U or I: an integer-typed column holds the
    packed value itself and is not float bits to reinterpret."""
    grey = 64 / 255
    opaque = np.array([0xFF404040], dtype="<u4")
    clear = np.array([0x00404040], dtype="<u4")
    unsigned = str(int(opaque[0]))
    signed = str(int(opaque.astype("<i4")[0]))
    spellings = {
        "F unsigned": ("F", unsigned),
        "F signed": ("F", signed),
        "F float, opaque": ("F", str(opaque.view("<f4")[0])),
        "F float, no alpha": ("F", str(clear.view("<f4")[0])),
        "U unsigned": ("U", unsigned),
        "I signed": ("I", signed),
    }
    assert signed.startswith("-")
    assert "e+38" in spellings["F float, opaque"][1]
    assert "e-39" in spellings["F float, no alpha"][1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for label, (typ, spelling) in spellings.items():
            text = _PCL_ASCII.replace("TYPE F F F F", f"TYPE F F F {typ}")
            back = _read(text.replace("4210752", spelling).encode())
            np.testing.assert_allclose(
                back.vertex_attrs["colors"], [[grey] * 3] * 3, err_msg=label
            )
        back = _read(_PCL_ASCII.replace("4210752", "nan", 1).encode())
    np.testing.assert_array_equal(back.vertex_attrs["colors"][0], [0, 0, 0])
    np.testing.assert_allclose(back.vertex_attrs["colors"][1:], [[grey] * 3] * 2)


def test_ascii_rgb_mixing_decimal_and_float_spellings_reads_each_row_as_its_own() -> (
    None
):
    """The decimal-or-float choice is per point: a decimal ``rgb`` beside a
    float-spelled one is still the colour its integer names."""
    opaque = np.array([0xFF404040], dtype="<u4").view("<f4")[0]
    text = _PCL_ASCII.replace("4210752", str(0xFF8000), 1).replace(
        "4210752", str(opaque), 1
    )
    back = _read(text.encode())
    np.testing.assert_allclose(
        back.vertex_attrs["colors"],
        [[1, 128 / 255, 0], [64 / 255] * 3, [64 / 255] * 3],
    )


@pytest.mark.parametrize("typ", ["F", "U"])
def test_ascii_rgba_reads_its_alpha_whatever_its_type(typ: str) -> None:
    text = _PCL_ASCII.replace("FIELDS x y z rgb", "FIELDS x y z rgba").replace(
        "TYPE F F F F", f"TYPE F F F {typ}"
    )
    back = _read(text.replace("4210752", str(0x80404040)).encode())
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], [[64 / 255] * 3 + [128 / 255]] * 3
    )


def test_a_height_without_a_width_names_the_grid() -> None:
    text = _PCL_ASCII.replace("WIDTH 3\n", "").replace("HEIGHT 1", "HEIGHT 3")
    back = _read(text.encode())
    assert (back.global_attrs["pcd_width"], back.global_attrs["pcd_height"]) == (1, 3)
    with pytest.raises(CodecError, match="WIDTH 1 x HEIGHT 2 is not POINTS 3"):
        _read(text.replace("HEIGHT 3", "HEIGHT 2").encode())
    with pytest.raises(CodecError, match="WIDTH 3 x HEIGHT 0 is not POINTS 3"):
        _read(text.replace("HEIGHT 3", "HEIGHT 0").encode())


def test_an_ascii_fraction_in_an_integer_field_is_refused_and_nan_is_zero() -> None:
    data = (
        _PCL_ASCII.replace("TYPE F F F F", "TYPE F F F I")
        .replace("FIELDS x y z rgb", "FIELDS x y z label")
        .encode()
    )
    with pytest.raises(CodecError, match="'label' is TYPE I but holds a fraction"):
        _read(data.replace(b"4210752", b"2.5", 1))
    back = _read(data.replace(b"4210752", b"nan", 1))
    np.testing.assert_array_equal(back.vertex_attrs["label"], [0, 4210752, 4210752])


def test_a_field_named_twice_is_read_under_its_position_with_a_warning() -> None:
    data = _PCL_ASCII.replace("FIELDS x y z rgb", "FIELDS x y z x").encode()
    with pytest.warns(UserWarning, match="FIELDS names x more than once"):
        back = _read(data)
    np.testing.assert_array_equal(back.vertex_attrs["x_3"], [4210752] * 3)


def test_ascii_with_a_word_in_the_body_is_refused() -> None:
    data = _PCL_ASCII.replace("0.90805", "oops").encode()
    with pytest.raises(CodecError, match="not a number"):
        _read(data)


def test_a_binary_body_shorter_than_its_count_is_refused() -> None:
    data = _write(_cloud())
    with pytest.raises(CodecError, match="the file holds"):
        _read(data[:-5])


def test_a_count_no_file_can_hold_is_refused_before_allocation() -> None:
    data = _PCL_ASCII.replace("POINTS 3", f"POINTS {2**62}").replace(
        "WIDTH 3", f"WIDTH {2**62}"
    )
    with pytest.raises((CodecError, ValidationError), match=str(2**62)):
        _read(data.encode())


def test_a_compressed_block_that_does_not_expand_is_refused(lzf) -> None:
    data = _write(_cloud(), data_format="binary_compressed")
    head, body = data.split(b"DATA binary_compressed\n")
    comp_len, full_len = struct.unpack_from("<II", body)
    bad = (
        head
        + b"DATA binary_compressed\n"
        + struct.pack("<II", comp_len, full_len + 8)
        + body[8:]
    )
    with pytest.raises(CodecError, match="POINTS 4 of"):
        _read(bad)
    truncated = (
        head
        + b"DATA binary_compressed\n"
        + struct.pack("<II", comp_len + 50, full_len)
        + body[8:]
    )
    with pytest.raises(CodecError, match="runs past the file end"):
        _read(truncated)
    garbage = (
        head
        + b"DATA binary_compressed\n"
        + struct.pack("<II", 5, full_len)
        + b"\x20\x00\x00\x00\x00"
    )
    with pytest.raises(CodecError, match="before the output start"):
        _read(garbage)


def test_a_compressed_block_that_cannot_expand_that_far_is_refused_unread(
    lzf, monkeypatch
) -> None:
    """A tiny block claiming a huge expansion is refused before the
    expansion is allocated: LZF grows a byte 88-fold at most."""
    data = _write(_cloud(), data_format="binary_compressed")
    head, body = data.split(b"DATA binary_compressed\n")
    comp_len, full_len = struct.unpack_from("<II", body)
    monkeypatch.setattr(
        lzf, "decompress", lambda *a, **k: pytest.fail("decompress was called")
    )
    bad = (
        head
        + b"DATA binary_compressed\n"
        + struct.pack("<II", 2, full_len)
        + body[8:10]
    )
    with pytest.raises(CodecError, match="2 bytes cannot expand to"):
        _read(bad)


def test_write_binary_compressed_refuses_a_block_past_32_bits(monkeypatch) -> None:
    """The two size words are 32-bit; a block they cannot spell is refused
    with a CodecError before struct.pack turns it into a struct.error."""

    class Huge(bytes):
        def __len__(self) -> int:
            return 1 << 32

    monkeypatch.setattr(_pcd.lzf, "compress", lambda plain: Huge())
    with pytest.raises(CodecError, match="32 bits"):
        _write(_cloud(), data_format="binary_compressed")


def test_write_binary_compressed_refuses_a_block_past_32_bits_unpacked(
    monkeypatch,
) -> None:
    """A block the size words cannot spell is refused before it is
    compressed, not after the work is done."""

    def never(plain: bytes) -> bytes:
        raise AssertionError("compressed a block already known to be too big")

    monkeypatch.setattr(_pcd, "_U32_MAX", 16)
    monkeypatch.setattr(_pcd.lzf, "compress", never)
    with pytest.raises(CodecError, match="32 bits"):
        _write(_cloud(), data_format="binary_compressed")


# ---------------------------------------------------------------------------
# Lazy reads
# ---------------------------------------------------------------------------


def test_a_lazy_binary_read_views_the_file_in_its_own_dtypes(tmp_path) -> None:
    poly = _cloud()
    path = tmp_path / "cloud.pcd"
    polyxios.write(poly, path)
    back = polyxios.read(path, lazy=True)

    assert mapped(back.vertices)
    assert back.vertices.dtype == np.float32
    assert not back.vertices.flags.writeable
    np.testing.assert_allclose(back.vertices, poly.vertices, rtol=1e-6)
    assert mapped(back.vertex_attrs["normals"])
    assert mapped(back.vertex_attrs["intensity"])
    assert back.vertex_attrs["intensity"].dtype == np.uint16
    assert mapped(back.vertex_attrs["moments"])
    assert back.vertex_attrs["moments"].shape == (4, 2)
    # A packed colour has to be decoded; it is the one copy.
    assert not mapped(back.vertex_attrs["colors"])
    np.testing.assert_allclose(back.vertex_attrs["colors"], poly.vertex_attrs["colors"])


def test_a_lazy_read_matches_the_eager_one(tmp_path) -> None:
    path = tmp_path / "cloud.pcd"
    polyxios.write(_cloud(), path)
    eager, lazy = polyxios.read(path), polyxios.read(path, lazy=True)
    np.testing.assert_allclose(eager.vertices, lazy.vertices)
    assert set(eager.vertex_attrs) == set(lazy.vertex_attrs)
    for key in eager.vertex_attrs:
        np.testing.assert_allclose(eager.vertex_attrs[key], lazy.vertex_attrs[key])


@pytest.mark.parametrize("data_format", ["ascii", "binary_compressed"])
def test_a_lazy_read_of_a_decoded_layout_is_refused(tmp_path, data_format: str) -> None:
    path = tmp_path / "cloud.pcd"
    polyxios.write(_cloud(), path, data_format=data_format)
    with pytest.raises(
        LazyReadError, match=f"DATA {data_format} does not support lazy"
    ):
        polyxios.read(path, lazy=True)


def test_a_lazy_read_of_a_buffer_is_refused() -> None:
    with pytest.raises(LazyReadError):
        _read(_write(_cloud()), lazy=True)


def test_a_lazy_read_needs_adjacent_coordinates_of_one_type(tmp_path) -> None:
    head = (
        "VERSION 0.7\nFIELDS x i y z\nSIZE 4 2 4 4\nTYPE F U F F\nCOUNT 1 1 1 1\n"
        "WIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA binary\n"
    )
    record = np.dtype([("x", "<f4"), ("i", "<u2"), ("y", "<f4"), ("z", "<f4")])
    path = tmp_path / "split.pcd"
    path.write_bytes(head.encode() + np.zeros(1, dtype=record).tobytes())
    with pytest.raises(LazyReadError, match="not three adjacent values"):
        polyxios.read(path, lazy=True)
    assert polyxios.read(path).vertices.shape == (1, 3)


def test_a_lazy_read_of_split_normals_keeps_them_as_three_fields(tmp_path) -> None:
    """Only the coordinates have to be viewable as (n, 3); normals that are
    not adjacent come back as the file spells them rather than refusing."""
    head = (
        "VERSION 0.7\nFIELDS x y z normal_x i normal_y normal_z\n"
        "SIZE 4 4 4 4 2 4 4\nTYPE F F F F U F F\nCOUNT 1 1 1 1 1 1 1\n"
        "WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n"
    )
    record = np.dtype(
        [(f, "<f4") for f in ("x", "y", "z", "nx")]
        + [("i", "<u2"), ("ny", "<f4"), ("nz", "<f4")]
    )
    path = tmp_path / "split.pcd"
    path.write_bytes(head.encode() + np.ones(1, dtype=record).tobytes())
    lazy = polyxios.read(path, lazy=True)
    assert set(lazy.vertex_attrs) == {"normal_x", "i", "normal_y", "normal_z"}
    assert mapped(lazy.vertex_attrs["normal_y"])
    eager = polyxios.read(path)
    assert set(eager.vertex_attrs) == {"normals", "i"}


def test_a_lazy_read_of_no_points_is_empty_not_an_error(tmp_path) -> None:
    path = tmp_path / "empty.pcd"
    polyxios.write(make_polydata(np.empty((0, 3)), []), path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = polyxios.read(path, lazy=True)
    assert back.vertices.shape == (0, 3)


def test_a_lazy_read_of_no_points_with_a_field_before_x_is_empty(tmp_path) -> None:
    """The coordinate view would start past the end of an empty body."""
    path = tmp_path / "empty.pcd"
    path.write_text(
        "VERSION 0.7\nFIELDS i x y z\nSIZE 4 4 4 4\nTYPE U F F F\nCOUNT 1 1 1 1\n"
        "WIDTH 0\nHEIGHT 1\nPOINTS 0\nDATA binary\n"
    )
    back = polyxios.read(path, lazy=True)
    assert back.vertices.shape == (0, 3)
    assert back.vertices.dtype == np.float32
    assert not back.vertices.flags.writeable
    assert back.vertex_attrs["i"].shape == (0,)


def test_integer_colours_count_0_to_255_like_every_other_codec() -> None:
    poly = _cloud()
    poly.vertex_attrs["colors"] = np.array(
        [[128, 0, 0], [0, 255, 0], [0, 0, 64], [255, 255, 255]], dtype=np.uint8
    )
    back = _read(_write(poly))
    np.testing.assert_allclose(
        back.vertex_attrs["colors"], poly.vertex_attrs["colors"] / 255.0
    )


def test_a_count_past_what_a_record_can_hold_is_refused() -> None:
    """numpy refuses a field past a C int with its own ValueError; the
    header is refused first, with the record size it asked for."""
    data = (
        _PCL_ASCII.replace("COUNT 1 1 1 1", "COUNT 1 1 1 2000000000")
        .replace("FIELDS x y z rgb", "FIELDS x y z w")
        .encode()
    )
    with pytest.raises(CodecError, match="one point 8000000012 bytes"):
        _read(data)
    with pytest.raises(CodecError, match="one point 2400000012 bytes"):
        _read(data.replace(b"2000000000", b"600000000"))


@pytest.mark.parametrize(
    ("fname", "dtype"),
    [
        ("rgb", np.float32),
        ("rgb", np.int32),
        ("rgb", np.float16),
        ("rgba", np.uint32),
        ("rgba", np.float32),
    ],
)
@pytest.mark.parametrize("data_format", ["ascii", "binary", "binary_compressed"])
def test_a_four_byte_attribute_named_rgb_is_widened_so_it_is_not_a_colour(
    fname: str, dtype: type, data_format: str
) -> None:
    """A four-byte scalar of that name is a packed colour to every reader,
    so a plain attribute is written at eight bytes and reads back as its
    values rather than as ``colors``."""
    poly = make_polydata(np.zeros((2, 3)), [("vertex", np.arange(2)[:, None])])
    poly.vertex_attrs[fname] = np.array([7, 3], dtype)
    with pytest.warns(UserWarning, match=f"'{fname}' field is a packed colour"):
        data = _write(poly, data_format=data_format)
    head = _header(data)
    assert head["FIELDS"].split()[-1] == fname
    assert head["SIZE"].split()[-1] == "8"
    back = _read(data)
    assert "colors" not in back.vertex_attrs
    np.testing.assert_array_equal(back.vertex_attrs[fname], [7, 3])
    assert back.vertex_attrs[fname].dtype.itemsize == 8


@pytest.mark.parametrize("viewpoint", ["abc", [1, "a", 3, 1, 0, 0, 0], object()])
def test_a_viewpoint_that_is_not_numbers_warns_and_writes_the_default(
    viewpoint: object,
) -> None:
    poly = _cloud()
    poly.global_attrs["pcd_viewpoint"] = viewpoint
    with pytest.warns(UserWarning, match="pcd_viewpoint needs 7 numbers"):
        data = _write(poly)
    assert _header(data)["VIEWPOINT"] == "0 0 0 1 0 0 0"
    assert "pcd_viewpoint" not in _read(data).global_attrs


def test_warnings_point_at_the_caller() -> None:
    data = _PCL_ASCII.replace("FIELDS x y z rgb", "FIELDS x y z x").encode()
    with pytest.warns(UserWarning, match="more than once") as caught:
        _read(data)
    assert {w.filename for w in caught} == {__file__}
    poly = _cloud()
    poly.vertex_attrs["rgb"] = np.zeros(4, dtype=np.float32)
    poly.global_attrs["pcd_width"] = 4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write(poly)
    assert len(caught) == 2
    assert {w.filename for w in caught} == {__file__}
