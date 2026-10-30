"""PVD: ParaView's collection file, an index of datasets over time.

The index holds no geometry, so every test writes the datasets it names
with the VTK XML codecs and checks what the collection makes of them.
"""

from __future__ import annotations

import io
from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import helper, make_polydata
from polyxios.codecs import _pvd
from polyxios.codecs._pvd import read, read_time_series, write, write_time_series
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _step(t: float, *, moved: bool = False, **kw):
    verts = _TET + ([t, 0, 0] if moved else [0, 0, 0])
    return make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"u": np.full(4, t)},
        **kw,
    )


def _index(
    path: Path, datasets: list[tuple[str, str]], *, root_type="Collection"
) -> Path:
    """Write an index whose DataSet elements are spelled as given."""
    lines = [
        '<?xml version="1.0"?>',
        f'<VTKFile type="{root_type}" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
        *(f"    <DataSet {attrs} file={file!r}/>" for attrs, file in datasets),
        "  </Collection>",
        "</VTKFile>",
    ]
    path.write_text("\n".join(lines).replace("'", '"'))
    return path


def _series(tmp_path: Path, times=(0.0, 0.5, 1.0)) -> Path:
    for k, t in enumerate(times):
        polyxios.write(_step(t), tmp_path / f"run_{k}.vtu")
    return _index(
        tmp_path / "run.pvd",
        [
            (f'timestep="{t}" group="" part="0"', f"run_{k}.vtu")
            for k, t in enumerate(times)
        ],
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_collection_reads_at_the_first_step_or_the_one_asked_for(
    tmp_path: Path,
) -> None:
    path = _series(tmp_path)
    first = read(path)
    assert first.vertex_attrs["u"].tolist() == [0.0] * 4
    assert first.global_attrs["time"] == 0.0
    assert first.element_types.tolist() == [10, 5]
    last = read(path, step=-1)
    assert last.vertex_attrs["u"].tolist() == [1.0] * 4
    assert last.global_attrs["time"] == 1.0
    assert polyxios.read(path, step=1).global_attrs["time"] == 0.5


def test_steps_come_back_in_ascending_time_whatever_the_index_order(
    tmp_path: Path,
) -> None:
    for k, t in enumerate((1.0, 0.0, 0.5)):
        polyxios.write(_step(t), tmp_path / f"r_{k}.vtu")
    path = _index(
        tmp_path / "r.pvd",
        [
            ('timestep="1.0"', "r_0.vtu"),
            ('timestep="0.0"', "r_1.vtu"),
            ('timestep="0.5"', "r_2.vtu"),
        ],
    )
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 0.5, 1.0]
    assert [m.vertex_attrs["u"][0] for m in meshes] == [0.0, 0.5, 1.0]
    assert len(read_time_series(path)) == 3


def test_datasets_at_one_time_are_merged_and_tagged(tmp_path: Path) -> None:
    polyxios.write(_step(0.0), tmp_path / "a.vtu")
    polyxios.write(_step(0.0, moved=True), tmp_path / "b.vtu")
    polyxios.write(_step(0.0), tmp_path / "c.vtu")
    path = _index(
        tmp_path / "m.pvd",
        [
            ('timestep="0" group="fluid" part="0"', "a.vtu"),
            ('timestep="0" group="" part="1"', "b.vtu"),
            ('timestep="0" group="fluid" part="2"', "c.vtu"),
        ],
    )
    poly = read(path)
    assert poly.vertices.shape == (12, 3)
    assert poly.offsets.tolist() == [0, 4, 7, 11, 14, 18, 21]
    # One group is one object across its datasets, as ParaView has it: the
    # tag names every element of both.
    assert poly.element_tags["fluid"].tolist() == [0, 1, 4, 5]
    assert poly.element_tags["part_1"].tolist() == [2, 3]
    assert set(poly.element_tags) == {"fluid", "part_1"}
    assert poly.global_attrs["time"] == 0.0
    assert poly.vertex_attrs["u"].shape == (12,)


def test_an_untagged_part_is_numbered_at_its_own_step(tmp_path: Path) -> None:
    for name in ("a", "b", "c", "d"):
        polyxios.write(_step(0.0), tmp_path / f"{name}.vtu")
    path = _index(
        tmp_path / "p.pvd",
        [
            ('timestep="0" part="0"', "a.vtu"),
            ('timestep="0" part="1"', "b.vtu"),
            ('timestep="1" part="0"', "c.vtu"),
            ('timestep="1" part="1"', "d.vtu"),
        ],
    )
    for step in (0, 1):
        assert set(read(path, step=step).element_tags) == {"part_0", "part_1"}
    # The place at the step is what numbers a part, so an index spelling
    # every dataset part="0" still reads them apart.
    same = _index(
        tmp_path / "q.pvd",
        [('timestep="0" part="0"', "a.vtu"), ('timestep="0" part="0"', "b.vtu")],
    )
    assert set(read(same).element_tags) == {"part_0", "part_1"}


def test_a_tag_a_dataset_already_carries_is_left_to_it(tmp_path: Path) -> None:
    polyxios.write(
        _step(0.0, element_tags={"fluid": np.array([0])}), tmp_path / "a.vtu"
    )
    polyxios.write(_step(0.0), tmp_path / "b.vtu")
    path = _index(
        tmp_path / "t.pvd",
        [
            ('timestep="0" group="fluid"', "a.vtu"),
            ('timestep="0" group="fluid"', "b.vtu"),
        ],
    )
    poly = read(path)
    assert poly.element_tags["fluid"].tolist() == [0]
    assert poly.element_tags["fluid_1"].tolist() == [0, 1, 2, 3]


def test_datasets_whose_globals_differ_warn_and_the_last_wins(tmp_path: Path) -> None:
    polyxios.write(
        _step(0.0, global_attrs={"run": "a", "n": 1, "v": [1.0, 2.0]}),
        tmp_path / "a.vtu",
    )
    polyxios.write(
        _step(0.0, global_attrs={"run": "b", "n": 1, "v": [1.0, 2.0]}),
        tmp_path / "b.vtu",
    )
    path = _index(
        tmp_path / "g.pvd",
        [('timestep="0" part="0"', "a.vtu"), ('timestep="0" part="1"', "b.vtu")],
    )
    with pytest.warns(UserWarning, match="spell global 'run' differently") as record:
        poly = read(path)
    assert len(record) == 1
    assert poly.global_attrs["run"] == "b" and poly.global_attrs["n"] == 1


def test_a_windows_index_spells_its_subdirectory_with_a_backslash(
    tmp_path: Path,
) -> None:
    (tmp_path / "sub").mkdir()
    polyxios.write(_step(0.5), tmp_path / "sub" / "run_0.vtu")
    path = _index(tmp_path / "w.pvd", [('timestep="0.5"', "sub\\run_0.vtu")])
    assert read(path).vertex_attrs["u"].tolist() == [0.5] * 4


def test_the_index_s_time_wins_over_a_dataset_s_own(tmp_path: Path) -> None:
    polyxios.write(_step(0.0, global_attrs={"time": 9.0}), tmp_path / "a.vtu")
    path = _index(tmp_path / "t.pvd", [('timestep="2.5"', "a.vtu")])
    assert read(path).global_attrs["time"] == 2.5


def test_a_dataset_with_no_timestep_is_at_zero(tmp_path: Path) -> None:
    polyxios.write(_step(0.0), tmp_path / "a.vtu")
    path = _index(tmp_path / "t.pvd", [("", "a.vtu")])
    assert read(path).global_attrs["time"] == 0.0
    times, _ = helper.read_time_series(path)
    assert times.tolist() == [0.0]


def test_a_parallel_index_named_by_the_collection_is_read_whole(tmp_path: Path) -> None:
    polyxios.write(_step(0.0), tmp_path / "piece_0.vtu")
    polyxios.write(_step(0.0, moved=True), tmp_path / "piece_1.vtu")
    (tmp_path / "whole.pvtu").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PUnstructuredGrid" version="1.0">'
        '<PUnstructuredGrid GhostLevel="0"><Piece Source="piece_0.vtu"/>'
        '<Piece Source="piece_1.vtu"/></PUnstructuredGrid></VTKFile>\n'
    )
    path = _index(tmp_path / "p.pvd", [('timestep="0"', "whole.pvtu")])
    assert read(path).vertices.shape == (8, 3)


def test_lazy_passes_through_to_a_lone_dataset(tmp_path: Path) -> None:
    polyxios.write(_step(0.0), tmp_path / "a.vtu", appended=True)
    path = _index(tmp_path / "l.pvd", [('timestep="0"', "a.vtu")])
    poly = polyxios.read(path, lazy=True)
    assert not poly.vertices.flags.writeable
    assert poly.vertices.dtype == np.float64


def test_lazy_reaches_a_dataset_that_cannot_be_mapped_and_raises_there(
    tmp_path: Path,
) -> None:
    pytest.importorskip("h5py")
    polyxios.write(_step(0.0), tmp_path / "a.vtkhdf")
    path = _index(tmp_path / "l.pvd", [('timestep="0"', "a.vtkhdf")])
    assert polyxios.read(path).element_types.tolist() == [10, 5]
    with pytest.raises(LazyReadError, match="a.vtkhdf"):
        polyxios.read(path, lazy=True)


def test_a_step_out_of_range_or_not_a_whole_number_is_refused(tmp_path: Path) -> None:
    path = _series(tmp_path)
    with pytest.raises(CodecError, match="step=3 is out of range"):
        read(path, step=3)
    with pytest.raises(CodecError, match="step=-4 is out of range"):
        read(path, step=-4)
    with pytest.raises(CodecError, match="step='x' is not a whole number"):
        read(path, step="x")
    with pytest.raises(CodecError, match="step=1.5 is not a whole number"):
        read(path, step=1.5)
    assert read(path, step=np.int64(1)).global_attrs["time"] == 0.5


def test_a_dataset_outside_the_index_s_directory_is_refused(tmp_path: Path) -> None:
    polyxios.write(_step(0.0), tmp_path / "a.vtu")
    inside = tmp_path / "sub"
    inside.mkdir()
    path = _index(inside / "x.pvd", [('timestep="0"', "../a.vtu")])
    with pytest.raises(CodecError, match="outside the index's own directory"):
        read(path)
    path = _index(inside / "y.pvd", [('timestep="0"', str(tmp_path / "a.vtu"))])
    with pytest.raises(CodecError, match="outside the index's own directory"):
        read(path)


def test_a_dataset_the_index_names_and_the_directory_lacks_raises(
    tmp_path: Path,
) -> None:
    path = _index(tmp_path / "x.pvd", [('timestep="0"', "gone.vtu")])
    with pytest.raises(FileNotFoundError):
        read(path)


def test_a_file_that_is_not_a_collection_is_refused(tmp_path: Path) -> None:
    path = _index(
        tmp_path / "x.pvd", [('timestep="0"', "a.vtu")], root_type="UnstructuredGrid"
    )
    with pytest.raises(CodecError, match="not a PVD file"):
        read(path)
    (tmp_path / "bad.pvd").write_text("<VTKFile type='Collection'>")
    with pytest.raises(CodecError, match="not a PVD file"):
        read(tmp_path / "bad.pvd")
    (tmp_path / "empty.pvd").write_text('<VTKFile type="Collection"></VTKFile>')
    with pytest.raises(CodecError, match="holds no <Collection>"):
        read(tmp_path / "empty.pvd")
    with pytest.raises(CodecError, match="names no dataset"):
        read(_index(tmp_path / "none.pvd", []))
    with pytest.raises(CodecError):
        read(tmp_path / "missing.pvd")


def test_a_dataset_spelled_badly_is_refused_by_number(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="DataSet 0 names no file"):
        read(_index(tmp_path / "a.pvd", [('timestep="0"', "")]))
    with pytest.raises(CodecError, match="timestep='soon', which is not a number"):
        read(_index(tmp_path / "b.pvd", [('timestep="soon"', "a.vtu")]))
    with pytest.raises(CodecError, match="timestep=nan"):
        read(_index(tmp_path / "c.pvd", [('timestep="nan"', "a.vtu")]))
    with pytest.raises(CodecError, match="timestep=inf, which is not a finite"):
        read(_index(tmp_path / "e.pvd", [('timestep="inf"', "a.vtu")]))
    with pytest.raises(CodecError, match="part='1e400', which is not a whole"):
        read(_index(tmp_path / "f.pvd", [('part="1e400"', "a.vtu")]))
    with pytest.raises(CodecError, match="part='1.5', which is not a whole"):
        read(_index(tmp_path / "g.pvd", [('part="1.5"', "a.vtu")]))
    with pytest.raises(CodecError, match="another collection"):
        read(_index(tmp_path / "d.pvd", [('timestep="0"', "b.pvd")]))


def test_a_buffer_or_a_gzip_name_is_refused_both_ways(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="a file object is not enough"):
        read(io.BytesIO(b"<VTKFile/>"))
    with pytest.raises(CodecError, match="a file object is not enough"):
        write(_step(0.0), io.BytesIO())
    with pytest.raises(CodecError, match="gzip is not handled here"):
        write(_step(0.0), tmp_path / "x.pvd.gz")


def test_unknown_options_are_warned_about(tmp_path: Path) -> None:
    path = _series(tmp_path)
    with pytest.warns(UserWarning, match="unrecognized options"):
        read(path, colour="red")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_puts_the_dataset_beside_the_index(tmp_path: Path) -> None:
    path = tmp_path / "out.pvd"
    write(_step(0.0, global_attrs={"gnum": 42}), path)
    assert (tmp_path / "out_0.vtu").exists()
    text = path.read_text()
    assert '<VTKFile type="Collection"' in text
    assert '<DataSet timestep="0.0" group="" part="0" file="out_0.vtu"/>' in text
    back = polyxios.read(path)
    assert back.element_types.tolist() == [10, 5]
    assert back.global_attrs["time"] == 0.0
    assert back.global_attrs["gnum"].tolist() == [42]


def test_the_time_comes_from_the_option_or_the_globals(tmp_path: Path) -> None:
    path = tmp_path / "t.pvd"
    write(_step(0.0), path, time=1.5)
    assert 'timestep="1.5"' in path.read_text()
    write(_step(0.0, global_attrs={"time": 2.5}), path)
    assert 'timestep="2.5"' in path.read_text()
    # The index spells the time; the dataset does not say it again.
    assert "time" not in polyxios.read(tmp_path / "t_0.vtu").global_attrs
    write(_step(0.0, global_attrs={"time": 2.5}), path, time=3.0)
    assert 'timestep="3.0"' in path.read_text()
    with pytest.raises(CodecError, match="time 'noon', which is not a number"):
        write(_step(0.0), path, time="noon")
    with pytest.raises(CodecError, match="step 1 is at time None, which is not"):
        write_time_series([(0.0, _step(0.0)), (None, _step(1.0))], path)


def test_a_text_time_is_dropped_by_name_and_a_number_silently(tmp_path: Path) -> None:
    path = tmp_path / "t.pvd"
    noon = _step(0.0, global_attrs={"time": "noon", "case": "run 3"})
    with pytest.warns(UserWarning, match="global 'time' holds 'noon', and the index"):
        write(noon, path, time=1.5)
    back = polyxios.read(tmp_path / "t_0.vtu")
    assert "time" not in back.global_attrs
    assert back.global_attrs["case"] == "run 3"
    assert read(path).global_attrs["time"] == 1.5
    with pytest.warns(UserWarning, match="index spells step 1's time"):
        write_time_series([(0.0, _step(0.0)), (1.0, noon)], path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(_step(0.0, global_attrs={"time": 2.5}), path, time=3.0)
    assert "time" not in polyxios.read(tmp_path / "t_0.vtu").global_attrs


def test_format_picks_the_dataset_s_extension_and_options_reach_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "f.pvd"
    surface = make_polydata(_TET[:3], [("triangle", np.array([[0, 1, 2]]))])
    write(surface, path, format="vtp", binary=False)
    dataset = tmp_path / "f_0.vtp"
    assert dataset.exists()
    assert 'format="ascii"' in dataset.read_text()
    assert polyxios.read(path).element_types.tolist() == [5]
    with pytest.raises(CodecError, match="names another collection"):
        write(surface, path, format=".pvd")
    with pytest.raises(UnsupportedFormatError):
        write(surface, path, format=".nope")


def test_a_series_writes_one_dataset_per_step(tmp_path: Path) -> None:
    path = tmp_path / "s.pvd"
    write_time_series(((0.5 * i, _step(0.5 * i, moved=True)) for i in range(3)), path)
    assert sorted(p.name for p in tmp_path.glob("s_*.vtu")) == [
        "s_0.vtu",
        "s_1.vtu",
        "s_2.vtu",
    ]
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 0.5, 1.0]
    assert [m.vertices[0, 0] for m in meshes] == [0.0, 0.5, 1.0]
    assert [m.global_attrs["time"] for m in meshes] == [0.0, 0.5, 1.0]


def test_a_series_may_change_its_mesh_from_step_to_step(tmp_path: Path) -> None:
    path = tmp_path / "s.pvd"
    other = make_polydata(_TET[:3], [("triangle", np.array([[0, 1, 2]]))])
    write_time_series([(0.0, _step(0.0)), (1.0, other)], path)
    assert polyxios.read(path, step=1).element_types.tolist() == [5]


def test_two_steps_at_one_time_or_no_step_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.pvd"
    with pytest.raises(CodecError, match="two steps are at time 1.0"):
        write_time_series([(1.0, _step(0.0)), (1.0, _step(1.0))], path)
    with pytest.raises(CodecError, match="at least one step"):
        write_time_series([], path)
    assert not path.exists()


def test_a_time_of_nan_is_refused_as_it_is_on_read(tmp_path: Path) -> None:
    path = tmp_path / "n.pvd"
    with pytest.raises(CodecError, match="step 0 is at time nan"):
        write(_step(0.0), path, time=float("nan"))
    with pytest.raises(CodecError, match="step 0 is at time nan"):
        write(_step(0.0, global_attrs={"time": np.nan}), path)
    with pytest.raises(CodecError, match="step 1 is at time nan"):
        write_time_series([(0.0, _step(0.0)), (float("nan"), _step(1.0))], path)
    with pytest.raises(CodecError, match="step 0 is at time inf"):
        write(_step(0.0), path, time=float("inf"))
    assert not path.exists()


def test_the_codec_is_registered_and_the_helper_dispatches_to_it(
    tmp_path: Path,
) -> None:
    assert polyxios.supported_extensions().count(".pvd") == 1
    path = tmp_path / "h.pvd"
    helper.write_time_series(
        [(0.0, _step(0.0)), (1.0, _step(1.0))], path, format=".vtu"
    )
    times, meshes = helper.read_time_series(path)
    assert times.tolist() == [0.0, 1.0] and len(meshes) == 2
    with pytest.raises(
        UnsupportedFormatError, match="VTKHDF \\(.vtkhdf\\) and PVD \\(.pvd\\)"
    ):
        helper.read_time_series(tmp_path / "h.vtu")


def test_the_module_lists_its_public_names() -> None:
    assert set(_pvd.__all__) == {
        "EXTENSION",
        "LABEL",
        "read",
        "read_time_series",
        "write",
        "write_time_series",
    }


@pytest.mark.parametrize("encoding", ["utf-9", "cp932"])
def test_an_encoding_expat_cannot_decode_is_refused(tmp_path: Path, encoding) -> None:
    """expat raises LookupError for an unknown encoding and ValueError for a
    multi-byte one, and neither is the ParseError the reader caught."""
    path = tmp_path / "bad.pvd"
    path.write_bytes(
        f'<?xml version="1.0" encoding="{encoding}"?>'
        '<VTKFile type="Collection"><Collection/></VTKFile>'.encode()
    )
    with pytest.raises(CodecError, match="not a PVD file"):
        read(path)
