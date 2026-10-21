from __future__ import annotations

import gzip
import io
from pathlib import Path
import struct
import tracemalloc
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._optpkg import TripWire
from polyxios._types import PolyData
from polyxios.codecs import _spz
from polyxios.codecs._spz import read, write
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    MissingPackageError,
    ValidationError,
)

# Three Gaussians of harmonic degree 1, trained with antialiasing, packed by
# the reference implementation as version 2, 3 and 4.
_REF_V2 = bytes.fromhex(
    "1f8b0800000000000013f3730f0e60626060600662461e4606060e068637ff191418181cd8"
    "80420c27fe33fcfbcf5003623e746d989cb3aca133f24c43734246c3820d0213264ce89db5"
    "5cada7a7a1a16142c68a0d0e176e487ce82898302361c586801d053322020e1868bc100000"
    "314db77264000000"
)
_REF_V3 = bytes.fromhex(
    "1f8b0800000000000013f3730f0e606660600061461e4606060e068637ff191418181cd8"
    "80420c27fe33fcfbcf5003623e746d989cb3aca133f24c43734246c3820d0213264cb817"
    "b0e8c82fe54e15a0fc8109192b36385cb821f1a1a360c28c84151b027614cc88083860a0"
    "f14200002555da4a67000000"
)
_REF_V4 = bytes.fromhex(
    "4e4753500400000003000000010c010620000000000000000000000000000000240000"
    "00000000001b000000000000000c000000000000000300000000000000120000000000"
    "000009000000000000001200000000000000090000000000000015000000000000000c"
    "0000000000000024000000000000001b0000000000000028b52ffd201bd90000000800"
    "00ecff00200000400600000000c8ff00feff007c0000000028b52ffd2003190000e145"
    "8028b52ffd2009490000936ca6808959cc808328b52ffd2009490000606880a0b01090"
    "909028b52ffd200c610000de50a2c4fa238924000000c028b52ffd201bd90000906"
    "8a8b040d0d818f08870909860a8b050b870985850c03028e810"
)
# The same Gaussians with no harmonics, packed by the reference as version 4.
_REF_V4_DEGREE_0 = bytes.fromhex(
    "4e4753500400000003000000000c010520000000000000000000000000000000240000"
    "00000000001b000000000000000c000000000000000300000000000000120000000000"
    "000009000000000000001200000000000000090000000000000015000000000000000c"
    "0000000000000028b52ffd201bd9000000080000ecff00200000400600000000c8ff00"
    "feff007c0000000028b52ffd2003190000e1458028b52ffd2009490000936ca6808959"
    "cc808328b52ffd2009490000606880a0b01090909028b52ffd200c610000de50a2c4fa"
    "238924000000c0"
)

# What the reference implementation decodes those files to.
_POSITIONS = [[0.5, -1.25, 2.0], [100.0, 0.0, -3.5], [-0.125, 7.75, 0.0]]
_SCALES = [[-4.0, -3.5, -2.0], [0.0, 1.0, -9.0], [-1.0, -1.0, -1.0]]
_OPACITY = [2.0149030685424805, -0.9916400909423828, 0.007843292318284512]
_DC = [
    [0.5098040699958801, -0.5098038911819458, 1.0065360069274902],
    [0.013072093017399311, 0.24836619198322296, -1.0065358877182007],
    [2.0, 0.013072093017399311, 0.09150345623493195],
]
# x, y, z, w per Gaussian.
_ROT_FIRST_THREE = [
    [0.10588240623474121, 0.20784318447113037, 0.30980396270751953, 0.9217438101768494],
    [-0.7019608020782471, 0.09803926944732666, 0.09803926944732666, 0.6985897421836853],
    [
        0.003921627998352051,
        0.003921627998352051,
        0.003921627998352051,
        0.9999769330024719,
    ],
]
_ROT_SMALLEST_THREE = [
    [0.10239902138710022, 0.20479804277420044, 0.30719706416130066, 0.9236894249916077],
    [
        0.6999171376228333,
        -0.09963148087263107,
        -0.09963148087263107,
        -0.7001879215240479,
    ],
    [0.0, 0.0, 0.0, 1.0],
]
# Coefficient-major, channel-minor, as the format stores them.
_SH = np.array(
    [
        [0.125, -0.1875, 0.3125, 0.375, -0.5, 0.625, 0.6875, -0.8125, 0.875],
        [0.0625, -0.125, 0.125, 0.1875, -0.25, 0.3125, 0.375, -0.375, 0.4375],
        [-0.125, 0.1875, -0.3125, -0.375, 0.5, -0.625, -0.6875, 0.8125, -0.875],
    ]
).reshape(3, 3, 3)

_HAS_ZSTD = _spz._stdlib_zstd is not None or _spz._zstandard()[1]
needs_zstd = pytest.mark.skipif(not _HAS_ZSTD, reason="no zstd implementation")


def _file(tmp_path: Path, data: bytes, name: str = "s.spz") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _check_reference(poly: PolyData, rotations) -> None:
    np.testing.assert_array_equal(poly.vertices, _POSITIONS)
    attrs = poly.vertex_attrs
    for k in range(3):
        np.testing.assert_array_equal(attrs[f"scale_{k}"], np.float32(_SCALES)[:, k])
        np.testing.assert_array_equal(attrs[f"f_dc_{k}"], np.float32(_DC)[:, k])
    np.testing.assert_array_equal(attrs["opacity"], np.float32(_OPACITY))
    xyzw = np.column_stack(
        [attrs["rot_1"], attrs["rot_2"], attrs["rot_3"], attrs["rot_0"]]
    )
    # The derived component can differ from the reference in its last bit,
    # by whether its build fuses the multiply-adds of a squared norm.
    np.testing.assert_allclose(xyzw, np.float32(rotations), rtol=0, atol=2e-7)
    for c in range(3):
        for j in range(3):
            np.testing.assert_array_equal(
                attrs[f"f_rest_{3 * c + j}"], np.float32(_SH[:, j, c])
            )
    assert "f_rest_9" not in attrs
    assert poly.global_attrs == {"spz_antialiased": True}
    assert poly.element_types.size == 0
    for arr in attrs.values():
        assert arr.dtype == np.float32


def _original_cloud() -> PolyData:
    """The cloud the reference implementation packed into those files."""
    xyzw = np.float32(
        [
            [
                0.10259784013032913,
                0.20519568026065826,
                0.307793527841568,
                0.9233804941177368,
            ],
            [
                -0.699999988079071,
                0.10000000149011612,
                0.10000000149011612,
                0.699999988079071,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    dc = np.float32([[0.5, -0.5, 1.0], [0.0, 0.25, -1.0], [2.0, 0.0, 0.1]])
    base = np.float32([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9])
    sh = (base * np.float32([[1.0], [0.5], [-1.0]])).reshape(3, 3, 3)
    attrs = {"opacity": np.float32([2.0, -1.0, 0.0])}
    for k in range(3):
        attrs[f"scale_{k}"] = np.float32(_SCALES)[:, k]
        attrs[f"f_dc_{k}"] = dc[:, k]
    for k, name in zip((3, 0, 1, 2), ("rot_0", "rot_1", "rot_2", "rot_3")):
        attrs[name] = xyzw[:, k]
    for c in range(3):
        for j in range(3):
            attrs[f"f_rest_{3 * c + j}"] = sh[:, j, c]
    return PolyData(
        vertices=np.array(_POSITIONS),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=attrs,
        element_attrs={},
        global_attrs={"spz_antialiased": True},
    )


def _legacy(
    *,
    n=1,
    version=2,
    degree=0,
    bits=12,
    flags=0,
    body: bytes | None = None,
) -> bytes:
    """A version 1 to 3 payload, before gzip."""
    per_point = (6 if version == 1 else 9) + 7 + (4 if version >= 3 else 3)
    per_point += 3 * _spz._SH_DIM.get(degree, 0)
    if body is None:
        body = bytes(n * per_point)
    return struct.pack("<4sIIBBBB", b"NGSP", version, n, degree, bits, flags, 0) + body


# --- reading ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "rotations"),
    [(_REF_V2, _ROT_FIRST_THREE), (_REF_V3, _ROT_SMALLEST_THREE)],
    ids=["v2", "v3"],
)
def test_a_reference_file_decodes_as_the_reference_does(
    tmp_path: Path, data: bytes, rotations
) -> None:
    _check_reference(read(_file(tmp_path, data)), rotations)


@needs_zstd
def test_a_reference_version_4_file_decodes_as_the_reference_does(
    tmp_path: Path,
) -> None:
    _check_reference(read(_file(tmp_path, _REF_V4)), _ROT_SMALLEST_THREE)


def test_a_version_1_file_holds_half_float_positions(tmp_path: Path) -> None:
    positions = np.array([[1.5, -2.0, 0.25]], dtype="<f2").tobytes()
    body = positions + bytes([255, 128, 128, 128, 160, 160, 160, 128, 128, 255])
    poly = read(_file(tmp_path, gzip.compress(_legacy(version=1, body=body))))
    np.testing.assert_array_equal(poly.vertices, [[1.5, -2.0, 0.25]])
    assert poly.vertex_attrs["opacity"][0] == np.inf
    np.testing.assert_array_equal(poly.vertex_attrs["scale_0"], [0.0])


def test_a_negative_fixed_point_position_is_sign_extended(tmp_path: Path) -> None:
    fixed = (-3 * 4096) & 0xFFFFFF
    body = fixed.to_bytes(3, "little") * 3 + bytes(10)
    poly = read(_file(tmp_path, gzip.compress(_legacy(body=body))))
    np.testing.assert_array_equal(poly.vertices, [[-3.0, -3.0, -3.0]])


def test_an_uncompressed_payload_reads_too(tmp_path: Path) -> None:
    poly = read(_file(tmp_path, gzip.decompress(_REF_V3)))
    _check_reference(poly, _ROT_SMALLEST_THREE)


def test_a_buffer_reads_and_writes(tmp_path: Path) -> None:
    _check_reference(
        polyxios.read(io.BytesIO(_REF_V3), fmt=".spz"), _ROT_SMALLEST_THREE
    )
    buf = io.BytesIO()
    polyxios.write(_original_cloud(), buf, fmt=".spz")
    assert gzip.decompress(buf.getvalue()) == gzip.decompress(_REF_V3)


def test_extension_records_are_skipped_with_a_warning(tmp_path: Path) -> None:
    data = _legacy(flags=_spz._FLAG_EXTENSIONS) + b"vendor bytes"
    with pytest.warns(UserWarning, match="vendor extension records"):
        read(_file(tmp_path, gzip.compress(data)))


def test_bytes_after_the_arrays_are_ignored_with_a_warning(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="3 bytes follow the splat arrays"):
        read(_file(tmp_path, gzip.compress(_legacy() + b"abc")))


def test_lazy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_file(tmp_path, _REF_V3), lazy=True)


def test_a_cloud_of_no_points_round_trips(tmp_path: Path) -> None:
    poly = read(_file(tmp_path, gzip.compress(_legacy(n=0))))
    assert poly.vertices.shape == (0, 3)
    path = tmp_path / "out.spz"
    # The reference reader refuses a point count of zero in every version.
    with pytest.warns(UserWarning, match="reference reader refuses"):
        write(poly, path)
    assert read(path).vertices.shape == (0, 3)


@needs_zstd
def test_an_empty_version_4_cloud_writes_no_stream(tmp_path: Path) -> None:
    poly = read(_file(tmp_path, gzip.compress(_legacy(n=0))))
    path = tmp_path / "out.spz"
    with pytest.warns(UserWarning, match="holds no point"):
        write(poly, path, version=4)
    data = path.read_bytes()
    assert len(data) == 32 and data[15] == 0
    assert read(path).vertices.shape == (0, 3)


def _with_extension_block(data: bytes, block: bytes) -> bytes:
    """A version 4 file with ``block`` between its header and its table."""
    header = bytearray(data[:32])
    header[14] |= _spz._FLAG_EXTENSIONS
    struct.pack_into("<I", header, 16, 32 + len(block))
    return bytes(header) + block + data[32:]


@needs_zstd
def test_version_4_extension_records_before_the_table_are_skipped(
    tmp_path: Path,
) -> None:
    data = _with_extension_block(_REF_V4, b"\x01\x00\x00\x00vendor")
    with pytest.warns(UserWarning, match="vendor extension records"):
        poly = read(_file(tmp_path, data))
    np.testing.assert_array_equal(poly.vertices, _POSITIONS)


@needs_zstd
@pytest.mark.parametrize("version", [3, 4])
def test_the_extension_warning_names_the_caller_of_read(
    tmp_path: Path, version: int
) -> None:
    if version == 4:
        data = _with_extension_block(_REF_V4, b"\x01\x00\x00\x00vendor")
    else:
        data = gzip.compress(_legacy(flags=_spz._FLAG_EXTENSIONS) + b"vendor")
    with pytest.warns(UserWarning, match="vendor extension records") as record:
        read(_file(tmp_path, data))
    assert record[0].filename == __file__


@needs_zstd
def test_a_version_4_extension_flag_over_no_records_is_silent(
    tmp_path: Path,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(_file(tmp_path, _with_extension_block(_REF_V4, b"")))
    np.testing.assert_array_equal(poly.vertices, _POSITIONS)


@needs_zstd
def test_version_4_bytes_before_the_table_unflagged_are_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    data = bytearray(_with_extension_block(_REF_V4, b"vendor"))
    data[14] &= ~_spz._FLAG_EXTENSIONS
    with pytest.warns(UserWarning, match="6 bytes lie between the header"):
        poly = read(_file(tmp_path, bytes(data)))
    np.testing.assert_array_equal(poly.vertices, _POSITIONS)


@needs_zstd
def test_version_4_bytes_after_the_streams_are_refused_even_flagged(
    tmp_path: Path,
) -> None:
    data = _with_extension_block(_REF_V4, b"") + b"vendor"
    with pytest.raises(CodecError, match="6 bytes follow the last stream"):
        read(_file(tmp_path, data))


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"", "no NGSP header"),
        (b"PLY? not a splat at all", "no NGSP header"),
        (gzip.compress(_legacy(version=0)), "version 0"),
        (gzip.compress(_legacy(version=7)), "version 7"),
        (gzip.compress(_legacy(degree=5)), "degree 5"),
        (gzip.compress(_legacy()[:-1]), "declares 1 points"),
        (struct.pack("<4sI", b"NGSP", 4) + bytes(10), "shorter than its 32-byte"),
    ],
)
def test_a_malformed_file_is_refused(tmp_path: Path, data: bytes, match: str) -> None:
    with pytest.raises(CodecError, match=match):
        read(_file(tmp_path, data))


def test_a_point_count_no_file_could_hold_is_refused(tmp_path: Path) -> None:
    data = gzip.compress(_legacy(n=2**32 - 1, body=b""))
    with pytest.raises((CodecError, ValidationError), match=str(2**32 - 1)):
        read(_file(tmp_path, data))


def _ngsp(*, n=3, degree=1, streams=None, toc=32, entries=None, tail=b"") -> bytes:
    sizes = [size for _, size in _spz._sizes(n, 4, degree) if size]
    blobs = [b"x" * 4 for _ in sizes] if streams is None else streams
    if entries is None:
        entries = [(len(b), s) for b, s in zip(blobs, sizes)]
    header = struct.pack(
        "<4sIIBBBBI12s", b"NGSP", 4, n, degree, 12, 0, len(entries), toc, bytes(12)
    )
    table = b"".join(struct.pack("<QQ", c, u) for c, u in entries)
    return header + table + b"".join(blobs) + tail


@needs_zstd
@pytest.mark.parametrize(
    ("data", "match"),
    [
        (_ngsp(entries=[(4, 27)]), "declares 1 streams where 3 points"),
        (_ngsp(toc=8), "table of contents at byte 8"),
        (_ngsp(toc=10**6), "table of contents at byte 1000000"),
        (_ngsp(entries=[(4, 10)] + [(4, s) for s in (3, 9, 9, 12, 27)]), "declares 10"),
        (_ngsp(entries=[(400, 27)] + [(4, s) for s in (3, 9, 9, 12, 27)]), "runs past"),
        (_ngsp(tail=b"zz"), "2 bytes follow the last stream"),
        (
            _ngsp(streams=[b"x" * 500] + [b"x" * 4] * 5),
            "declares 500 bytes, more than zstd ever makes of 27",
        ),
        (_ngsp(), "not a valid zstd frame"),
    ],
)
def test_a_malformed_version_4_file_is_refused(
    tmp_path: Path, data: bytes, match: str
) -> None:
    with pytest.raises(CodecError, match=match):
        read(_file(tmp_path, data))


@needs_zstd
def test_a_stream_shorter_than_zstd_could_pack_its_size_is_refused(
    tmp_path: Path,
) -> None:
    data = _ngsp(n=100_000)
    with pytest.raises(
        CodecError, match="positions stream declares 4 bytes, fewer than zstd"
    ):
        read(_file(tmp_path, data))


@needs_zstd
def test_a_version_4_cloud_of_repeated_points_reads_back(tmp_path: Path) -> None:
    n = 200_000
    poly = make_polydata(np.zeros((n, 3)), [])
    poly.vertex_attrs.update(_original_cloud().vertex_attrs)
    for key, values in poly.vertex_attrs.items():
        poly.vertex_attrs[key] = np.full(n, values[0], dtype=np.float32)
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="past the 1024:1 the reference reader"):
        write(poly, path, version=4)
    assert n * 9 > path.stat().st_size * 1024
    back = read(path)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    assert back.vertex_attrs["scale_0"].shape == (n,)
    assert np.unique(back.vertex_attrs["scale_0"]).size == 1


@needs_zstd
def test_a_frame_holding_more_than_its_table_says_is_refused(tmp_path: Path) -> None:
    sizes = [size for _, size in _spz._sizes(3, 4, 1) if size]
    blobs = [_spz._zstd_compress(bytes(s + 1)) for s in sizes]
    data = _ngsp(streams=blobs)
    with pytest.raises(
        CodecError, match="holds 28 bytes where the table of contents declares 27"
    ):
        read(_file(tmp_path, data))


@needs_zstd
@pytest.mark.parametrize("tail", [b"junk", b"another frame"])
def test_bytes_after_a_zstd_frame_are_refused(tmp_path: Path, tail: bytes) -> None:
    sizes = [size for _, size in _spz._sizes(3, 4, 1) if size]
    blobs = [_spz._zstd_compress(bytes(s)) for s in sizes]
    blobs[0] += _spz._zstd_compress(tail) if tail == b"another frame" else tail
    with pytest.raises(CodecError, match="not a valid zstd frame|holds 28 bytes"):
        read(_file(tmp_path, _ngsp(streams=blobs)))


def test_version_4_without_zstd_names_the_extra(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_spz, "_stdlib_zstd", None)
    monkeypatch.setattr(
        _spz, "_zstandard", lambda: (TripWire("zstandard is gone"), False)
    )
    with pytest.raises(MissingPackageError, match="gone"):
        read(_file(tmp_path, _REF_V4))
    with pytest.raises(MissingPackageError, match="gone"):
        write(_original_cloud(), tmp_path / "o.spz", version=4)


# --- writing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "data", "rotations"),
    [(2, _REF_V2, _ROT_FIRST_THREE), (3, _REF_V3, _ROT_SMALLEST_THREE)],
)
def test_the_writer_packs_as_the_reference_does(
    tmp_path: Path, version: int, data: bytes, rotations
) -> None:
    path = tmp_path / "o.spz"
    write(_original_cloud(), path, version=version)
    assert gzip.decompress(path.read_bytes()) == gzip.decompress(data)


@needs_zstd
def test_a_version_4_file_is_written_as_the_reference_writes_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "o.spz"
    write(_original_cloud(), path, version=4)
    assert path.read_bytes() == _REF_V4


@needs_zstd
def test_a_degree_0_version_4_file_omits_the_harmonics_stream(
    tmp_path: Path,
) -> None:
    poly = _original_cloud()
    for k in range(9):
        del poly.vertex_attrs[f"f_rest_{k}"]
    path = tmp_path / "o.spz"
    write(poly, path, version=4)
    # The reference writer and reader both skip a stream of no bytes, so
    # the file holds five where a cloud with harmonics holds six.
    assert path.read_bytes() == _REF_V4_DEGREE_0
    back = read(_file(tmp_path, _REF_V4_DEGREE_0))
    assert not any(key.startswith("f_rest_") for key in back.vertex_attrs)
    np.testing.assert_array_equal(back.vertices, _POSITIONS)


def test_the_default_version_is_3(tmp_path: Path) -> None:
    path = tmp_path / "o.spz"
    write(_original_cloud(), path)
    assert struct.unpack_from("<I", gzip.decompress(path.read_bytes()), 4)[0] == 3


def test_a_cloud_with_no_splat_attributes_is_written_with_defaults(
    tmp_path: Path,
) -> None:
    poly = make_polydata(np.array([[1.0, 2.0, 3.0]]), [])
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="the rotation as the identity"):
        write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.vertices, [[1, 2, 3]])
    assert back.vertex_attrs["rot_0"][0] == 1.0
    assert back.vertex_attrs["scale_0"][0] == 0.0


def test_a_degenerate_rotation_is_written_as_the_identity(tmp_path: Path) -> None:
    poly = _original_cloud()
    for name in ("rot_0", "rot_1", "rot_2", "rot_3"):
        poly.vertex_attrs[name][1] = 0.0
    poly.vertex_attrs["rot_2"][2] = np.nan
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match=r"2 rotation\(s\) are of zero length"):
        write(poly, path)
    back = read(path)
    assert back.vertex_attrs["rot_0"][1:].tolist() == [1.0, 1.0]


def test_a_rotation_missing_only_w_is_written_as_spelled(tmp_path: Path) -> None:
    poly = _original_cloud()
    del poly.vertex_attrs["rot_0"]
    poly.vertex_attrs["rot_1"][2] = 0.0
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    messages = " ".join(str(w.message) for w in record)
    assert "['rot_0']; written as zero." in messages
    assert "1 rotation(s) are of zero length" in messages
    back = read(path).vertex_attrs
    assert back["rot_0"][1] == 0.0 and back["rot_0"][2] == 1.0
    assert abs(back["rot_1"][1]) == pytest.approx(0.98, abs=1e-2)


def test_non_finite_harmonics_saturate_on_their_own_side(tmp_path: Path) -> None:
    poly = _original_cloud()
    poly.vertex_attrs["f_rest_0"] = np.float32([np.inf, -np.inf, np.nan])
    path = tmp_path / "o.spz"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    assert read(path).vertex_attrs["f_rest_0"].tolist() == [127 / 128, -1.0, 0.0]


def test_a_position_just_short_of_half_a_step_rounds_down(tmp_path: Path) -> None:
    below_half = np.nextafter(0.5, 0.0) / 4096
    poly = make_polydata(np.array([[below_half, -below_half, 0.5 / 4096]]), [])
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="written as zero"):
        write(poly, path, fractional_bits=12)
    assert read(path).vertices.tolist() == [[0.0, 0.0, 1 / 4096]]


def test_rounding_is_half_away_from_zero_at_every_magnitude() -> None:
    x = np.float32([np.nextafter(np.float32(0.5), np.float32(0)), -0.5, 0.5, 2.5])
    assert _spz._round(x).tolist() == [0.0, -1.0, 1.0, 3.0]


@pytest.mark.parametrize(
    ("count", "degree"), [(0, 0), (9, 1), (24, 2), (45, 3), (72, 4)]
)
def test_the_harmonic_degree_follows_the_coefficients(
    tmp_path: Path, count: int, degree: int
) -> None:
    poly = _original_cloud()
    for k in range(9):
        del poly.vertex_attrs[f"f_rest_{k}"]
    for k in range(count):
        poly.vertex_attrs[f"f_rest_{k}"] = np.full(3, 0.25, dtype=np.float32)
    path = tmp_path / "o.spz"
    write(poly, path)
    assert gzip.decompress(path.read_bytes())[12] == degree
    rest = sorted(k for k in read(path).vertex_attrs if k.startswith("f_rest_"))
    assert len(rest) == count


def test_coefficients_past_a_whole_degree_are_dropped(tmp_path: Path) -> None:
    poly = _original_cloud()
    poly.vertex_attrs["f_rest_9"] = np.zeros(3, dtype=np.float32)
    with pytest.warns(UserWarning, match=r"\['f_rest_9'\] have no place"):
        write(poly, tmp_path / "o.spz")


def test_channels_are_strided_by_a_third_of_the_coefficients(tmp_path: Path) -> None:
    # Thirty coefficients are ten per channel, as a splat PLY lays them out:
    # degree 2 takes the first eight of each ten.
    poly = _original_cloud()
    for k in range(9):
        del poly.vertex_attrs[f"f_rest_{k}"]
    for c in range(3):
        for j in range(10):
            value = (c + 1) * 0.25 - (j % 2) * 0.125
            poly.vertex_attrs[f"f_rest_{10 * c + j}"] = np.full(3, value, np.float32)
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="have no place") as record:
        write(poly, path)
    assert "'f_rest_8', 'f_rest_9', 'f_rest_18'" in str(record[0].message)
    back = read(path).vertex_attrs
    for c in range(3):
        for j in range(8):
            expected = poly.vertex_attrs[f"f_rest_{10 * c + j}"][0]
            assert back[f"f_rest_{8 * c + j}"][0] == expected


def test_far_positions_take_fewer_fractional_bits(tmp_path: Path) -> None:
    poly = _original_cloud()
    poly = PolyData(
        vertices=np.array([[5000.25, -70000.5, 0.0]] * 3),
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
        vertex_attrs=poly.vertex_attrs,
        element_attrs={},
    )
    path = tmp_path / "o.spz"
    write(poly, path)
    assert gzip.decompress(path.read_bytes())[13] == 6
    np.testing.assert_array_equal(read(path).vertices, poly.vertices)


def test_the_most_negative_fixed_point_position_keeps_twelve_bits(
    tmp_path: Path,
) -> None:
    poly = make_polydata(np.array([[-2048.0, 0.0, 2047.5]]), [])
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="written as zero"):
        write(poly, path)
    assert gzip.decompress(path.read_bytes())[13] == 12
    np.testing.assert_array_equal(read(path).vertices, poly.vertices)


def test_a_positive_2048_still_takes_eleven_bits(tmp_path: Path) -> None:
    poly = make_polydata(np.array([[2048.0, 0.0, -2048.0]]), [])
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="written as zero"):
        write(poly, path)
    assert gzip.decompress(path.read_bytes())[13] == 11


def test_fractional_bits_can_be_chosen(tmp_path: Path) -> None:
    path = tmp_path / "o.spz"
    write(_original_cloud(), path, fractional_bits=4)
    assert gzip.decompress(path.read_bytes())[13] == 4
    write(_original_cloud(), path, fractional_bits=np.int64(5), version=np.int64(2))
    payload = gzip.decompress(path.read_bytes())
    assert payload[13] == 5 and payload[4] == 2
    np.testing.assert_array_equal(read(path).vertices[0], [0.5, -1.25, 2.0])


@pytest.mark.parametrize(
    ("opts", "match"),
    [
        ({"version": 1}, "version= takes 2, 3 or 4"),
        ({"version": True}, "version= takes 2, 3 or 4"),
        ({"version": 3.0}, "version= takes 2, 3 or 4"),
        ({"version": "3"}, "version= takes 2, 3 or 4"),
        ({"fractional_bits": -1}, "fractional_bits= takes an integer"),
        ({"fractional_bits": 2.5}, "fractional_bits= takes an integer"),
        ({"fractional_bits": 20}, "past what a 24-bit fixed point number"),
        ({"fractional_bits": 31}, "takes an integer from 0 to 30"),
    ],
)
def test_a_bad_option_is_refused(tmp_path: Path, opts, match: str) -> None:
    with pytest.raises(CodecError, match=match):
        write(_original_cloud(), tmp_path / "o.spz", **opts)


def test_a_position_too_far_for_any_fixed_point_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(np.array([[1e7, 0.0, 0.0]]), [])
    with pytest.raises(CodecError, match="reaches 10000000.0"):
        write(poly, tmp_path / "o.spz")


def test_2d_positions_are_padded_to_z_0(tmp_path: Path) -> None:
    poly = make_polydata(np.array([[0.5, -1.25], [2.0, 3.0]]), [])
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="carries no"):
        write(poly, path)
    np.testing.assert_array_equal(read(path).vertices, [[0.5, -1.25, 0], [2, 3, 0]])


def test_a_non_finite_position_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(np.array([[np.nan, 0.0, 0.0]]), [])
    with pytest.raises(CodecError, match="not finite"):
        write(poly, tmp_path / "o.spz")


def test_more_points_than_the_header_counts_are_refused(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(_spz, "_MAX_POINTS", 2, raising=False)
    with pytest.raises(CodecError, match="3 points are past the 2"):
        write(_original_cloud(), tmp_path / "o.spz")
    assert not (tmp_path / "o.spz").exists()


@pytest.mark.parametrize(
    ("value", "flagged"),
    [(True, True), (np.True_, True), (1, True), (np.array([1], np.uint8), True)]
    + [(False, False), (0, False), (np.array([0]), False)],
)
def test_the_antialiasing_flag_takes_a_boolean_in_any_shape(
    tmp_path: Path, value, flagged: bool
) -> None:
    poly = _original_cloud()
    poly.global_attrs["spz_antialiased"] = value
    path = tmp_path / "o.spz"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    assert read(path).global_attrs.get("spz_antialiased", False) is flagged


@pytest.mark.parametrize("value", ["False", "no", 2, np.array([1, 1]), None])
def test_an_antialiasing_value_that_is_not_a_boolean_leaves_the_flag_clear(
    tmp_path: Path, value
) -> None:
    poly = _original_cloud()
    poly.global_attrs["spz_antialiased"] = value
    path = tmp_path / "o.spz"
    with pytest.warns(UserWarning, match="not a boolean") as record:
        write(poly, path)
    assert record[0].filename == __file__
    assert "spz_antialiased" not in read(path).global_attrs


def test_a_splat_attribute_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    poly = _original_cloud()
    poly.vertex_attrs["opacity"] = np.zeros((3, 2))
    with pytest.raises(CodecError, match="'opacity' has shape"):
        write(poly, tmp_path / "o.spz")


def test_elements_tags_and_other_attributes_are_dropped_with_warnings(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        np.eye(3),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"temperature": np.ones(3)},
    )
    with pytest.warns(UserWarning) as record:
        write(poly, tmp_path / "o.spz")
    messages = " ".join(str(w.message) for w in record)
    assert "['temperature'] have no place" in messages
    assert "elements and tags were dropped" in messages


def test_unknown_write_options_warn(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_original_cloud(), tmp_path / "o.spz", level=9)


def test_an_opacity_far_out_does_not_overflow(tmp_path: Path) -> None:
    poly = _original_cloud()
    poly.vertex_attrs["opacity"] = np.float32([-1e6, 1e6, 0.0])
    path = tmp_path / "o.spz"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
        back = read(path)
    assert back.vertex_attrs["opacity"][:2].tolist() == [-np.inf, np.inf]


def test_a_cloud_survives_a_trip_through_a_splat_ply(tmp_path: Path) -> None:
    first = read(_file(tmp_path, _REF_V3))
    polyxios.write(first, tmp_path / "s.ply")
    again = polyxios.read(tmp_path / "s.ply")
    np.testing.assert_array_equal(again.vertices, first.vertices)
    assert sorted(again.vertex_attrs) == sorted(first.vertex_attrs)
    for name, values in first.vertex_attrs.items():
        np.testing.assert_array_equal(again.vertex_attrs[name], values)


def test_a_gzipped_spz_reads_through_both_layers(tmp_path: Path) -> None:
    _check_reference(read(_file(tmp_path, gzip.compress(_REF_V3))), _ROT_SMALLEST_THREE)


def test_a_corrupt_inner_gzip_member_is_refused(tmp_path: Path) -> None:
    data = gzip.compress(_REF_V3[:-12] + bytes(12))
    with pytest.raises(CodecError, match="gzip stream is corrupt"):
        read(_file(tmp_path, data))


_PADDING = 64 << 20


def _peak_read(path: Path) -> int:
    """Read ``path`` and return the peak of what the read allocated."""
    tracemalloc.start()
    try:
        read(path)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("layers", [1, 2], ids=["spz", "spz.gz"])
def test_bytes_after_the_arrays_are_counted_not_held(
    tmp_path: Path, layers: int
) -> None:
    data = _legacy() + bytes(_PADDING)
    for _ in range(layers):
        data = gzip.compress(data, compresslevel=1)
    path = _file(tmp_path, data)
    del data
    with pytest.warns(UserWarning, match=f"{_PADDING} bytes follow the splat arrays"):
        assert _peak_read(path) < _PADDING // 8


@needs_zstd
def test_version_4_extension_records_are_skipped_not_held(tmp_path: Path) -> None:
    data = gzip.compress(
        _with_extension_block(_REF_V4, bytes(_PADDING)), compresslevel=1
    )
    path = _file(tmp_path, data)
    del data
    with pytest.warns(UserWarning, match="vendor extension records"):
        assert _peak_read(path) < _PADDING // 8
