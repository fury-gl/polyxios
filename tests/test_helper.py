from __future__ import annotations

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios.helper import read_multiblock_vtp, read_polydata, resolve_path

_INDEX_TEMPLATE = """<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0">
  <vtkMultiBlockDataSet>
{entries}
  </vtkMultiBlockDataSet>
</VTKFile>
"""


def _write_piece(path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]], dtype=np.int32))])
    polyxios.write(poly, str(path))


def _write_index(path, names) -> None:
    entries = "\n".join(
        f'    <DataSet index="{i}" file="{name}"/>' for i, name in enumerate(names)
    )
    path.write_text(_INDEX_TEMPLATE.format(entries=entries), encoding="utf-8")


def test_read_multiblock_vtp_merges_pieces(tmp_path) -> None:
    pieces = tmp_path / "blocks"
    pieces.mkdir()
    for i in range(2):
        _write_piece(pieces / f"piece_{i}.vtp")

    index = tmp_path / "index.vtp"
    _write_index(index, ["blocks/piece_0.vtp", "blocks/piece_1.vtp"])

    poly = read_multiblock_vtp(index)
    assert len(poly.vertices) == 6
    assert len(poly.element_types) == 2


def test_read_multiblock_vtp_accepts_str_path(tmp_path) -> None:
    pieces = tmp_path / "blocks"
    pieces.mkdir()
    _write_piece(pieces / "piece_0.vtp")

    index = tmp_path / "index.vtp"
    _write_index(index, ["blocks/piece_0.vtp"])

    poly = read_multiblock_vtp(str(index))
    assert len(poly.vertices) == 3


def test_read_multiblock_vtp_skips_missing_sub_files(tmp_path) -> None:
    """A partially downloaded companion directory still loads."""
    pieces = tmp_path / "blocks"
    pieces.mkdir()
    _write_piece(pieces / "piece_0.vtp")

    index = tmp_path / "index.vtp"
    _write_index(index, ["blocks/piece_0.vtp", "blocks/piece_1.vtp"])

    poly = read_multiblock_vtp(index)
    assert len(poly.vertices) == 3


def test_read_multiblock_vtp_skips_unreadable_sub_files(tmp_path) -> None:
    """A corrupt piece is skipped like a missing one instead of failing the load."""
    pieces = tmp_path / "blocks"
    pieces.mkdir()
    _write_piece(pieces / "piece_0.vtp")
    (pieces / "piece_1.vtp").write_text("not a vtp file at all", encoding="utf-8")

    index = tmp_path / "index.vtp"
    _write_index(index, ["blocks/piece_0.vtp", "blocks/piece_1.vtp"])

    poly = read_multiblock_vtp(index)
    assert len(poly.vertices) == 3


def test_read_multiblock_vtp_all_missing_raises(tmp_path) -> None:
    index = tmp_path / "index.vtp"
    _write_index(index, ["blocks/piece_0.vtp"])

    with pytest.raises(FileNotFoundError):
        read_multiblock_vtp(index)


def test_read_multiblock_vtp_rejects_path_traversal(tmp_path) -> None:
    outside = tmp_path / "outside.vtp"
    _write_piece(outside)

    nested = tmp_path / "nested"
    nested.mkdir()
    index = nested / "index.vtp"
    _write_index(index, ["../outside.vtp"])

    with pytest.raises(PermissionError, match="Path traversal"):
        read_multiblock_vtp(index)


def test_read_multiblock_vtp_requires_multiblock_element(tmp_path) -> None:
    index = tmp_path / "index.vtp"
    index.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PolyData"></VTKFile>\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vtkMultiBlockDataSet"):
        read_multiblock_vtp(index)


def test_read_polydata_falls_back_to_multiblock(tmp_path) -> None:
    """A .vtp index file is unreadable as PolyData but loads as a multiblock."""
    pieces = tmp_path / "blocks"
    pieces.mkdir()
    _write_piece(pieces / "piece_0.vtp")

    index = tmp_path / "index.vtp"
    _write_index(index, ["blocks/piece_0.vtp"])

    poly = read_polydata(str(index))
    assert len(poly.vertices) == 3


def test_read_polydata_missing_file_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("polyxios.fetcher.POLYXIOS_HOME", str(tmp_path))
    monkeypatch.setattr("polyxios.fetcher._catalog_cache", None)
    monkeypatch.setattr("polyxios.fetcher._ext_map_cache", None)
    monkeypatch.setattr(
        "polyxios.helper.fetch",
        lambda name, **kwargs: (_ for _ in ()).throw(RuntimeError("no catalog")),
    )
    with pytest.raises(FileNotFoundError, match="No such file"):
        read_polydata(str(tmp_path / "nope.obj"))


def test_resolve_path_does_not_fetch_for_a_path_like_name(
    tmp_path, monkeypatch
) -> None:
    """A missing path is reported as missing, never swapped for a catalog asset."""

    def _boom(*args, **kwargs):
        raise AssertionError("a path with a directory must not be fetched")

    monkeypatch.setattr("polyxios.helper.fetch", _boom)

    with pytest.raises(FileNotFoundError, match="No such file"):
        resolve_path(tmp_path / "data" / "bunny.obj")
    with pytest.raises(FileNotFoundError, match="No such file"):
        resolve_path("data/bunny.obj")


def test_resolve_path_fetches_a_bare_name(tmp_path, monkeypatch) -> None:
    asset = tmp_path / "bunny.obj"
    asset.write_text("x", encoding="utf-8")
    monkeypatch.setattr("polyxios.helper.fetch", lambda name, **kwargs: str(asset))

    assert resolve_path("bunny.obj") == asset
