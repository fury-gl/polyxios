"""The HDF5 plumbing MED, CGNS, H5M and HMF share.

Every test here needs h5py and skips without it. The codecs' own tests cover
what each format does with the plumbing; these cover the plumbing on its
own, so a change to it is caught once rather than four times.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._types import PolyData
from polyxios.codecs import _hdf5
from polyxios.exceptions import CodecError

h5py = pytest.importorskip("h5py")

_TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_a_failed_write_leaves_neither_a_half_file_nor_a_partial(tmp_path: Path):
    path = tmp_path / "out.med"
    path.write_bytes(b"stale")
    with pytest.raises(RuntimeError):
        with _hdf5.open_hdf5_write(path, fmt=".med") as f:
            f.create_dataset("x", data=np.zeros(3))
            raise RuntimeError("halfway")
    assert path.read_bytes() == b"stale"
    assert list(tmp_path.iterdir()) == [path]


def test_a_finished_write_replaces_the_stale_file(tmp_path: Path):
    path = tmp_path / "out.med"
    path.write_bytes(b"stale")
    with _hdf5.open_hdf5_write(path, fmt=".med") as f:
        f.create_dataset("x", data=np.zeros(3))
    with _hdf5.open_hdf5_read(path, fmt=".med") as f:
        assert f["x"].shape == (3,)
    assert list(tmp_path.iterdir()) == [path]


def test_a_buffer_and_a_gz_name_both_read_back(tmp_path: Path):
    buf = io.BytesIO()
    with _hdf5.open_hdf5_write(buf, fmt=".med") as f:
        f.create_dataset("x", data=np.arange(3))
    assert _hdf5.is_hdf5(buf.getvalue())
    with _hdf5.open_hdf5_read(io.BytesIO(buf.getvalue()), fmt=".med") as f:
        np.testing.assert_array_equal(f["x"][()], [0, 1, 2])

    packed = tmp_path / "out.med.gz"
    with _hdf5.open_hdf5_write(packed, fmt=".med") as f:
        f.create_dataset("x", data=np.arange(3))
    assert _hdf5.is_hdf5(gzip.decompress(packed.read_bytes()))
    with _hdf5.open_hdf5_read(packed, fmt=".med") as f:
        np.testing.assert_array_equal(f["x"][()], [0, 1, 2])


def test_a_file_without_the_magic_is_refused_by_name(tmp_path: Path):
    path = tmp_path / "not.med"
    path.write_bytes(b"# not hdf5\n" * 4)
    with pytest.raises(CodecError, match=r"'not\.med': not an HDF5 file"):
        with _hdf5.open_hdf5_read(path, fmt=".med"):
            pass
    with pytest.raises(CodecError, match="not an HDF5 file"):
        with _hdf5.open_hdf5_read(io.BytesIO(b"# not hdf5\n" * 4), fmt=".med"):
            pass


def test_a_truncated_file_names_h5py_s_complaint(tmp_path: Path):
    path = tmp_path / "cut.med"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(1000))
    path.write_bytes(path.read_bytes()[:600])
    with pytest.raises(CodecError, match="h5py could not open it"):
        with _hdf5.open_hdf5_read(path, fmt=".med"):
            pass


# ---------------------------------------------------------------------------
# Datasets and attributes
# ---------------------------------------------------------------------------


def test_a_swapped_dataset_reads_native(tmp_path: Path):
    path = tmp_path / "swap.med"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(3, dtype=">i4"))
        f.create_dataset("t", data=np.array([b"a"], dtype="S1"))
        f.create_group("g")
    with h5py.File(path, "r") as f:
        x = _hdf5.as_array(f["x"], name="swap.med", where="x")
        assert x.dtype.isnative
        np.testing.assert_array_equal(x, [0, 1, 2])
        assert _hdf5.as_array(f["t"], name="swap.med", where="t").dtype.kind == "S"
        with pytest.raises(CodecError, match="'g' is a group, not a dataset"):
            _hdf5.as_array(f["g"], name="swap.med", where="g")
        with pytest.raises(CodecError, match="'/' holds no 'y'"):
            _hdf5.child(f, "y", name="swap.med", where="/")


def test_text_of_decodes_every_spelling_and_refuses_the_rest():
    assert _hdf5.text_of(b"abc\x00\x00") == "abc"
    assert _hdf5.text_of(np.bytes_(" abc ")) == "abc"
    assert _hdf5.text_of("abc") == "abc"
    assert _hdf5.text_of(np.array(b"abc")) == "abc"
    assert _hdf5.text_of(np.array([b"a", b"b"])) is None
    assert _hdf5.text_of(np.int32(3)) is None
    assert _hdf5.text_of(b"\xff") == "�"


def test_attr_text_answers_none_for_a_missing_or_numeric_attribute(tmp_path: Path):
    path = tmp_path / "attrs.med"
    with h5py.File(path, "w") as f:
        f.attrs["s"] = "text"
        f.attrs["n"] = 3
    with h5py.File(path, "r") as f:
        assert _hdf5.attr_text(f, "s") == "text"
        assert _hdf5.attr_text(f, "n") is None
        assert _hdf5.attr_text(f, "missing") is None


def test_dataset_options_pop_the_h5py_keys_and_pair_the_level_with_a_method():
    opts = {"compression": "gzip", "compression_opts": 4, "other": 1}
    assert _hdf5.dataset_options(opts, fmt=".med") == {
        "compression": "gzip",
        "compression_opts": 4,
    }
    assert opts == {"other": 1}
    assert _hdf5.dataset_options({}, fmt=".med") == {}
    with pytest.raises(CodecError, match="compression_opts names a level"):
        _hdf5.dataset_options({"compression_opts": 4}, fmt=".med")


def test_unknown_options_warn_once_and_name_themselves():
    with pytest.warns(UserWarning, match=r"\.med read: unrecognized options"):
        _hdf5.warn_unknown_opts({"bogus": 1}, fmt=".med", what="read")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _hdf5.warn_unknown_opts({}, fmt=".med", what="read")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _sizes() -> dict[int, int]:
    return {3: 2, 5: 3, 9: 4}


def test_blocks_come_out_by_ascending_code_in_mesh_order():
    poly = make_polydata(
        _TRI,
        [
            ("quad", [[0, 1, 3, 2]]),
            ("line", [[0, 1], [2, 3]]),
            ("triangle", [[0, 1, 2]]),
        ],
    )
    blocks, dropped = _hdf5.type_blocks(poly, sizes=_sizes(), fmt=".x")
    assert dropped == set()
    assert [(c, i.tolist(), cells.shape) for c, i, cells in blocks] == [
        (3, [1, 2], (2, 2)),
        (5, [3], (1, 3)),
        (9, [0], (1, 4)),
    ]
    assert all(cells.dtype == np.int64 for _, _, cells in blocks)
    np.testing.assert_array_equal(_hdf5.kept_index(blocks), [1, 2, 3, 0])
    assert _hdf5.kept_index([]).shape == (0,)


def test_a_type_the_format_has_no_name_for_is_reported_not_warned():
    poly = make_polydata(_TRI, [("triangle", [[0, 1, 2]]), ("tetra", [[0, 1, 2, 3]])])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blocks, dropped = _hdf5.type_blocks(poly, sizes=_sizes(), fmt=".x")
    assert dropped == {10}
    assert [c for c, _, _ in blocks] == [5]


def test_an_element_of_the_wrong_width_is_warned_about_and_its_type_kept():
    """A triangle of four nodes is malformed, not a type the format lacks:
    the well-formed triangles are written and the warning says what it is."""
    poly = PolyData(
        vertices=_TRI,
        connectivity=np.array([0, 1, 2, 0, 1, 2, 3, 1, 2, 3], dtype=np.int64),
        offsets=np.array([0, 3, 7, 10], dtype=np.int64),
        element_types=np.array([5, 5, 5], dtype=np.uint8),
    )
    with pytest.warns(
        UserWarning, match=r"\.x write: 1 element\(s\) hold a node count"
    ):
        blocks, dropped = _hdf5.type_blocks(poly, sizes=_sizes(), fmt=".x")
    assert dropped == set()
    assert [(c, i.tolist()) for c, i, _ in blocks] == [(5, [0, 2])]
    np.testing.assert_array_equal(blocks[0][2], [[0, 1, 2], [1, 2, 3]])


def test_a_type_whose_every_element_is_malformed_yields_no_block():
    poly = PolyData(
        vertices=_TRI,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.warns(UserWarning, match=r"1 element\(s\)"):
        blocks, dropped = _hdf5.type_blocks(poly, sizes=_sizes(), fmt=".x")
    assert blocks == [] and dropped == set()


def test_tags_are_renumbered_onto_the_written_order_and_empty_ones_kept():
    kept = np.array([1, 2, 3, 0])
    tags = {
        "a": np.array([0, 3]),
        "b": np.array([1]),
        "stale": np.array([7, -1]),
    }
    out = _hdf5.reindexed_tags(tags, kept, 4)
    assert out["a"].tolist() == [2, 3]
    assert out["b"].tolist() == [0]
    assert out["stale"].tolist() == []
    assert _hdf5.reindexed_tags(None, kept, 4) == {}


def test_a_tag_over_an_unwritten_element_is_dropped_from_the_group():
    out = _hdf5.reindexed_tags({"a": np.array([0, 1, 2])}, np.array([2, 0]), 3)
    assert out["a"].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_the_plumbing_is_not_a_codec():
    assert not hasattr(_hdf5, "EXTENSION")
    assert not hasattr(_hdf5, "read")
    assert ".hdf5" not in polyxios.supported_extensions()


# ---------------------------------------------------------------------------
# User blocks and the temporary file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("userblock", [512, 1024, 4096])
def test_a_file_with_a_user_block_is_hdf5_from_a_path_and_a_buffer(
    tmp_path: Path, userblock: int
):
    """The superblock sits past the user block, where h5py itself finds it."""
    path = tmp_path / "ub.med"
    with h5py.File(path, "w", userblock_size=userblock) as f:
        f.create_dataset("x", data=np.arange(3))
    raw = path.read_bytes()
    assert raw[: len(_hdf5.HDF5_MAGIC)] != _hdf5.HDF5_MAGIC
    assert _hdf5.is_hdf5(raw)
    assert not _hdf5.is_hdf5(raw[:userblock])
    with _hdf5.open_hdf5_read(path, fmt=".med") as f:
        np.testing.assert_array_equal(f["x"][()], [0, 1, 2])
    with _hdf5.open_hdf5_read(io.BytesIO(raw), fmt=".med") as f:
        np.testing.assert_array_equal(f["x"][()], [0, 1, 2])


def test_a_missing_file_is_still_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        with _hdf5.open_hdf5_read(tmp_path / "gone.med", fmt=".med"):
            pass


def test_the_finished_file_has_the_permissions_a_plain_open_gives(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.write_bytes(b"")
    path = tmp_path / "out.med"
    with _hdf5.open_hdf5_write(path, fmt=".med") as f:
        f.create_dataset("x", data=np.zeros(3))
    assert path.stat().st_mode == plain.stat().st_mode


def test_two_writers_on_one_path_never_share_a_temporary(tmp_path: Path):
    """Each write gets a temporary of its own, so a slower writer cannot have
    its half file moved into place, or deleted, by a faster one."""
    path = tmp_path / "out.med"
    with _hdf5.open_hdf5_write(path, fmt=".med") as outer:
        outer.create_dataset("x", data=np.zeros(1))
        with _hdf5.open_hdf5_write(path, fmt=".med") as inner:
            inner.create_dataset("x", data=np.zeros(2))
        assert {p.suffix for p in tmp_path.iterdir()} == {".partial", ".med"}
    with _hdf5.open_hdf5_read(path, fmt=".med") as f:
        assert f["x"].shape == (1,)
    assert list(tmp_path.iterdir()) == [path]


def test_a_move_that_fails_leaves_no_temporary_behind(tmp_path: Path):
    path = tmp_path / "taken.med"
    path.mkdir()
    with pytest.raises(OSError):
        with _hdf5.open_hdf5_write(path, fmt=".med") as f:
            f.create_dataset("x", data=np.zeros(3))
    assert list(tmp_path.iterdir()) == [path]
