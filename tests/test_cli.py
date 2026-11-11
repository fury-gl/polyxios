import io
import json
import logging
import os
import re
import sys
import types
import urllib.request
import warnings
import zipfile

import numpy as np
import pytest

import polyxios
from polyxios import helper, make_polydata
from polyxios.cli import _log_warning, _setup_logging, logger as cli_logger, main
from polyxios.exceptions import CodecError, UnsupportedFormatError
from polyxios.fetcher import fetch
from polyxios.helper import visualize_mesh


@pytest.fixture
def temp_polyxios_home(tmp_path, monkeypatch):
    """Fixture to set up a clean, localized POLYXIOS_HOME for CLI testing without network dependencies."""
    monkeypatch.setenv("POLYXIOS_HOME", str(tmp_path))
    # The CLI reaches the fetcher through the module, so patching it once here
    # covers both.
    monkeypatch.setattr("polyxios.fetcher.POLYXIOS_HOME", str(tmp_path))
    # Module-level caches outlive a single test, so a stale entry from another
    # POLYXIOS_HOME must not leak into this one.
    monkeypatch.setattr("polyxios.fetcher._catalog_cache", None)
    monkeypatch.setattr("polyxios.fetcher._ext_map_cache", None)
    monkeypatch.setattr("polyxios.fetcher._verified_cache", None)

    models_data = {
        "ext_to_package": {
            "inp": "abaqus",
            "fem": "nastran",
            "vol": "netgen",
            "f3grid": "flac3d",
            "tec": "tecplot",
            "meshb": "medit",
            "xml": "dolfin",
            "msh": "gmsh",
        },
        "formats": {
            "obj": {
                "bunny.obj": {
                    "url": "https://example.com/obj/bunny.obj",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
                "armadillo.obj": {
                    "url": "https://example.com/obj/armadillo.obj",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
            },
            "ply": {
                "cow.ply": {
                    "url": "https://example.com/ply/cow.ply",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                }
            },
            "vtk": {
                "1.vtk": {
                    "url": "https://example.com/vtk/1.vtk",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
                "2.vtk": {
                    "url": "https://example.com/vtk/2.vtk",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
                "armadillo.vtk": {
                    "url": "https://example.com/vtk/armadillo.vtk",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
            },
            "gmsh": {
                "insulated-2.2.msh": {
                    "url": "https://example.com/gmsh/insulated-2.2.msh",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
                "insulated-4.1.msh": {
                    "url": "https://example.com/gmsh/insulated-4.1.msh",
                    "sha256": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
                },
            },
        },
    }
    with open(tmp_path / "models.json", "w") as f:
        json.dump(models_data, f)

    def mock_load_catalog():
        with open(tmp_path / "models.json", encoding="utf-8") as fh:
            return json.load(fh)

    monkeypatch.setattr("polyxios.fetcher._load_models_catalog", mock_load_catalog)
    monkeypatch.setattr("polyxios.fetcher._verify_sha256", lambda path, sha: True)
    return tmp_path


def create_real_model(path):
    """Helper to write a real valid PolyData model to the disk."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]], dtype=np.int32))])
    polyxios.write(poly, str(path))
    return poly


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pxios", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Polyxios CLI" in captured.out


def test_cli_fetch(temp_polyxios_home, monkeypatch, capsys):
    obj_dir = temp_polyxios_home / "obj"
    obj_dir.mkdir()
    model_path = obj_dir / "bunny.obj"
    create_real_model(model_path)

    monkeypatch.setattr(sys, "argv", ["pxios", "fetch", "bunny.obj"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Successfully fetched to:" in captured.out
    assert str(model_path) in captured.out


def test_cli_viz(temp_polyxios_home, monkeypatch, capsys):
    pytest.importorskip("fury")

    def mock_show(actors):
        assert isinstance(actors, list)
        assert len(actors) > 0

    monkeypatch.setattr("fury.window.show", mock_show)

    obj_dir = temp_polyxios_home / "obj"
    obj_dir.mkdir()
    model_path = obj_dir / "armadillo.obj"
    create_real_model(model_path)

    monkeypatch.setattr(sys, "argv", ["pxios", "viz", "armadillo.obj"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert f"Reading {model_path}" in captured.out


def test_cli_viz_lines(temp_polyxios_home, monkeypatch, capsys):
    pytest.importorskip("fury")

    def mock_show(actors):
        assert isinstance(actors, list)
        assert len(actors) > 0

    monkeypatch.setattr("fury.window.show", mock_show)

    obj_dir = temp_polyxios_home / "obj"
    obj_dir.mkdir()
    model_path = obj_dir / "armadillo.obj"
    create_real_model(model_path)

    monkeypatch.setattr(sys, "argv", ["pxios", "viz", "armadillo.obj", "--lines"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "No line elements found. Rendering surface wireframe." in captured.out


def test_cli_viz_points(temp_polyxios_home, monkeypatch, capsys):
    pytest.importorskip("fury")

    def mock_show(actors):
        assert isinstance(actors, list)
        assert len(actors) > 0

    monkeypatch.setattr("fury.window.show", mock_show)

    obj_dir = temp_polyxios_home / "obj"
    obj_dir.mkdir()
    model_path = obj_dir / "armadillo.obj"
    create_real_model(model_path)

    monkeypatch.setattr(sys, "argv", ["pxios", "viz", "armadillo.obj", "--points"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Opening FURY window ..." in captured.out
    assert "Rendering strictly as point cloud" in captured.out


def test_cli_viz_no_fury(temp_polyxios_home, monkeypatch, capsys):
    # Mask fury module
    monkeypatch.setitem(sys.modules, "fury", None)

    obj_dir = temp_polyxios_home / "obj"
    obj_dir.mkdir()
    model_path = obj_dir / "armadillo.obj"
    create_real_model(model_path)

    monkeypatch.setattr(sys, "argv", ["pxios", "viz", "armadillo.obj"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "FURY is not installed" in captured.err
    # No window is about to open, so the CLI must not say one is.
    assert "Opening FURY window" not in captured.out


def test_cli_list_local(temp_polyxios_home, monkeypatch, capsys):
    vtk_dir = temp_polyxios_home / "vtk"
    vtk_dir.mkdir()
    tag_file = vtk_dir / ".tag"
    tag_file.write_text(
        "latest:0ae5335020cfc8b520d90fcb5b7898a7f377520b4f6db672ba6a20770e7c7dde"
    )

    p1 = vtk_dir / "1.vtk"
    p2 = vtk_dir / "2.vtk"
    p3 = vtk_dir / "armadillo.vtk"
    create_real_model(p1)
    create_real_model(p2)
    create_real_model(p3)

    monkeypatch.setattr(sys, "argv", ["pxios", "list", "vtk", "--local"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "Cached .vtk files:" in captured.out
    assert str(p1) in captured.out
    assert str(p2) in captured.out


def test_cli_list_local_all(temp_polyxios_home, monkeypatch, capsys):

    obj_dir = temp_polyxios_home / "obj"
    obj_dir.mkdir()
    vtk_dir = temp_polyxios_home / "vtk"
    vtk_dir.mkdir()

    p1 = obj_dir / "bunny.obj"
    p2 = vtk_dir / "armadillo.vtk"
    create_real_model(p1)
    create_real_model(p2)

    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--local"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "Cached files:" in captured.out
    assert "[obj]" in captured.out
    assert str(p1) in captured.out
    assert "[vtk]" in captured.out
    assert str(p2) in captured.out


def test_cli_viz_no_filename(temp_polyxios_home, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pxios", "viz"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2


def test_cli_convert(temp_polyxios_home, monkeypatch, capsys):
    input_path = temp_polyxios_home / "armadillo.obj"
    output_path = temp_polyxios_home / "output.vtk"
    create_real_model(input_path)

    monkeypatch.setattr(
        sys, "argv", ["pxios", "convert", str(input_path), str(output_path)]
    )

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert f"Reading '{input_path}'..." in captured.out
    assert f"Writing to '{output_path}'..." in captured.out
    assert "Conversion successful." in captured.out

    assert output_path.exists()
    poly_out = polyxios.read(str(output_path))
    assert len(poly_out.vertices) == 3


def _write_seam_obj(path):
    """Write a triangle pair whose shared vertices carry two different normals."""
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\n"
        "f 1//1 2//1 3//1\nf 1//2 2//2 3//2\n"
    )


@pytest.mark.filterwarnings("default")
def test_cli_reports_codec_warnings_without_source_location(
    temp_polyxios_home, monkeypatch, capsys
):
    """A codec warning reads like CLI advice, not a Python traceback fragment."""
    input_path = temp_polyxios_home / "seam.obj"
    output_path = temp_polyxios_home / "seam.vtk"
    _write_seam_obj(input_path)

    monkeypatch.setattr(
        sys, "argv", ["pxios", "convert", str(input_path), str(output_path)]
    )
    hook_before = warnings.showwarning
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert warnings.showwarning is hook_before

    captured = capsys.readouterr()
    lines = [ln for ln in captured.err.splitlines() if ln.startswith("WARNING: ")]
    assert len(lines) == 1
    assert re.fullmatch(
        r"WARNING: \.obj: a vertex is given more than one normal, .*[^)]", lines[0]
    )


@pytest.mark.filterwarnings("default")
def test_cli_verbose_keeps_codec_warning_source_location(
    temp_polyxios_home, monkeypatch, capsys
):
    """``--verbose`` appends the ``(file:line)`` a bug report needs."""
    input_path = temp_polyxios_home / "seam.obj"
    output_path = temp_polyxios_home / "seam.vtk"
    _write_seam_obj(input_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["pxios", "convert", str(input_path), str(output_path), "--verbose"],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    lines = [ln for ln in captured.err.splitlines() if ln.startswith("WARNING: ")]
    assert len(lines) == 1
    assert re.search(r" \(.+\.py:\d+\)$", lines[0])


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (UserWarning, "WARNING: careful"),
        (type("ObjWarning", (UserWarning,), {}), "WARNING: careful"),
        (DeprecationWarning, "WARNING: DeprecationWarning: careful"),
    ],
)
def test_cli_warning_prefix_names_only_actionable_classes(
    capsys, monkeypatch, category, expected
):
    """A ``UserWarning`` subclass reads as plain advice; other classes are named."""
    monkeypatch.setattr(cli_logger, "handlers", [])
    _setup_logging()
    _log_warning("careful", category, "codec.py", 12)
    assert capsys.readouterr().err.splitlines() == [expected]


def test_visualize_mesh_empty_does_not_announce_window(monkeypatch, capsys):
    """An empty mesh opens no window, so nothing may say one is opening."""
    fake_fury = types.ModuleType("fury")
    fake_fury.actor = types.SimpleNamespace()
    fake_fury.window = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "fury", fake_fury)
    monkeypatch.setattr(cli_logger, "handlers", [])
    _setup_logging()

    empty = make_polydata(np.empty((0, 3), dtype=np.float64), [])
    visualize_mesh(empty)

    assert "Opening FURY window" not in capsys.readouterr().out


def test_cli_list(temp_polyxios_home, monkeypatch, capsys):
    models_file = temp_polyxios_home / "models.json"
    mock_catalog = {
        "formats": {
            "obj": {"bunny.obj": {}, "armadillo.obj": {}},
            "ply": {"Armadillo.ply": {}},
        }
    }
    models_file.write_text(json.dumps(mock_catalog), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["pxios", "list"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Available files for fetch:" in captured.out
    assert "[obj]" in captured.out
    assert "bunny.obj" in captured.out
    assert "[ply]" in captured.out
    assert "Armadillo.ply" in captured.out


def test_cli_list_filtered(temp_polyxios_home, monkeypatch, capsys):
    models_file = temp_polyxios_home / "models.json"
    mock_catalog = {
        "formats": {
            "obj": {"bunny.obj": {}, "armadillo.obj": {}},
            "ply": {"Armadillo.ply": {}},
        }
    }
    models_file.write_text(json.dumps(mock_catalog), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["pxios", "list", "obj"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Available files for fetch (obj):" in captured.out
    assert "bunny.obj" in captured.out
    assert "Armadillo.ply" not in captured.out

    monkeypatch.setattr(sys, "argv", ["pxios", "list", "invalid"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "No package/extension found matching 'invalid'" in captured.err


def test_cli_fetch_folder(temp_polyxios_home, monkeypatch, capsys):
    gmsh_dir = temp_polyxios_home / "gmsh"
    gmsh_dir.mkdir()
    (gmsh_dir / "insulated-2.2.msh").write_text("dummy msh file")
    (gmsh_dir / "insulated-4.1.msh").write_text("dummy msh file")

    monkeypatch.setattr(sys, "argv", ["pxios", "fetch", "gmsh"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Successfully fetched 2 file(s):" in captured.out
    assert str(gmsh_dir / "insulated-2.2.msh") in captured.out
    assert str(gmsh_dir / "insulated-4.1.msh") in captured.out


def test_cli_list_extensions(temp_polyxios_home, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--extensions"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "File formats available in remote catalog:" in captured.out
    assert "abaqus (.inp)" in captured.out
    assert "dolfin (.xml)" in captured.out
    assert "gmsh (.msh)" in captured.out
    assert "obj (.obj)" in captured.out
    assert "vtk (.vtk)" in captured.out


def test_cli_list_codecs(temp_polyxios_home, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--codecs"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "File formats supported by polyxios codecs:" in captured.out
    assert "obj (.obj)" in captured.out
    assert "vtk (.vtk)" in captured.out
    assert "abaqus (.inp)" in captured.out


def test_cli_list_codecs_names_a_shared_extension_honestly(
    temp_polyxios_home, monkeypatch, capsys
):
    """'.dat' resolves by content, so it cannot be listed under one format."""
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--codecs"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    # Binary Tecplot groups with the codec that reports it, not on its own.
    assert "tecplot (.plt, .tec)" in captured.out
    assert "tecplot/nastran (.dat)" in captured.out
    assert "plt (.plt)" not in captured.out
    assert "dat (.dat)" not in captured.out


def test_cli_list_modes_are_mutually_exclusive(temp_polyxios_home, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--local", "--codecs"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["pxios", "--verbose", "list", "--codecs"],
        ["pxios", "list", "--codecs", "--verbose"],
    ],
    ids=["before-subcommand", "after-subcommand"],
)
def test_cli_verbose_accepted_on_either_side(
    temp_polyxios_home, monkeypatch, capsys, argv
):
    """--verbose reaches the handler whether it precedes or follows the subcommand."""
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert cli_logger.level == logging.DEBUG
    assert "not recognized" not in capsys.readouterr().err


def test_cli_verbose_defaults_to_info(temp_polyxios_home, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--codecs"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert cli_logger.level == logging.INFO


def test_cli_verbose_logs_a_traceback_on_failure(
    temp_polyxios_home, monkeypatch, capsys
):
    """A failing command reports the traceback only when --verbose is given."""

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("polyxios.fetcher.get_fetchable_files", _boom)

    monkeypatch.setattr(sys, "argv", ["pxios", "list"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert "Traceback" not in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--verbose"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "RuntimeError: network down" in err


def test_cli_fetch_with_zip_companion(temp_polyxios_home, monkeypatch):
    models_file = temp_polyxios_home / "models.json"
    mock_catalog = {
        "ext_to_package": {
            "vtp": "vtp",
            "zip": "vtp",
        },
        "formats": {
            "vtp": {
                "mock_dataset.vtp": {
                    "url": "https://example.com/vtp/mock_dataset.vtp",
                    "sha256": "fake_sha",
                    "size_bytes": 10,
                },
                "mock_dataset.zip": {
                    "url": "https://example.com/vtp/mock_dataset.zip",
                    "sha256": "fake_sha",
                    "size_bytes": 50,
                },
            }
        },
    }
    models_file.write_text(json.dumps(mock_catalog), encoding="utf-8")

    vtp_data = b"vtp data"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("mock_dataset/piece_0.vtp", "sub-piece content")
    zip_data = zip_buffer.getvalue()

    def mock_urlopen(req, timeout=None):
        class MockResponse:
            def __init__(self, data):
                self.data = data
                self.headers = {"Content-Length": str(len(data))}

            def read(self, chunk_size=None):
                if chunk_size is None:
                    return self.data
                # Return data in chunks to emulate stream reading
                res = self.data
                self.data = b""
                return res

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        url = req.full_url
        if url.endswith(".zip"):
            return MockResponse(zip_data)
        else:
            return MockResponse(vtp_data)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    path = fetch("mock_dataset.vtp")

    assert os.path.exists(path)
    extracted_file = os.path.join(os.path.dirname(path), "mock_dataset", "piece_0.vtp")
    assert os.path.exists(extracted_file)
    with open(extracted_file) as f:
        assert f.read() == "sub-piece content"


def test_cli_convert_refuses_existing_output(temp_polyxios_home, monkeypatch, capsys):
    input_path = temp_polyxios_home / "armadillo.obj"
    output_path = temp_polyxios_home / "output.vtk"
    create_real_model(input_path)
    output_path.write_text("do not clobber me", encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["pxios", "convert", str(input_path), str(output_path)]
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert output_path.read_text(encoding="utf-8") == "do not clobber me"


def test_cli_convert_force_overwrites(temp_polyxios_home, monkeypatch, capsys):
    input_path = temp_polyxios_home / "armadillo.obj"
    output_path = temp_polyxios_home / "output.vtk"
    create_real_model(input_path)
    output_path.write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["pxios", "convert", str(input_path), str(output_path), "--force"]
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    assert "Conversion successful." in capsys.readouterr().out
    assert len(polyxios.read(str(output_path)).vertices) == 3


def test_cli_list_extensions_catalog_failure(temp_polyxios_home, monkeypatch, capsys):
    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr("polyxios.fetcher._load_models_catalog", _boom)
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--extensions"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Catalog retrieval failed" in captured.err
    # The built-in fallback map is still listed.
    assert "abaqus (.inp)" in captured.out


def test_cli_convert_missing_input(temp_polyxios_home, monkeypatch, capsys):
    missing = temp_polyxios_home / "nope.obj"
    output_path = temp_polyxios_home / "out.vtk"

    monkeypatch.setattr(
        sys, "argv", ["pxios", "convert", str(missing), str(output_path)]
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert "Failed to convert model" in capsys.readouterr().err


def _picture(path, shape=(4, 6, 3)):
    pixels = (np.arange(int(np.prod(shape))) * 5 % 256).astype(np.uint8)
    pixels = pixels.reshape(shape)
    polyxios.write_image(helper.array_to_image(pixels), path)
    return pixels


def _convert(monkeypatch, capsys, source, target):
    monkeypatch.setattr(sys, "argv", ["pxios", "convert", str(source), str(target)])
    with pytest.raises(SystemExit) as excinfo:
        main()
    captured = capsys.readouterr()
    return excinfo.value.code, [
        ln for ln in captured.err.splitlines() if ln.startswith("WARNING: ")
    ]


@pytest.mark.filterwarnings("default")
def test_cli_convert_between_image_formats_stays_compact(
    temp_polyxios_home, monkeypatch, capsys
):
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "a.vti"
    target = temp_polyxios_home / "b.vtkhdf"
    pixels = _picture(source)
    assert _convert(monkeypatch, capsys, source, target) == (0, [])
    assert polyxios.read_image(target).extent == (0, 5, 0, 3, 0, 0)
    back = temp_polyxios_home / "c.vti"
    assert _convert(monkeypatch, capsys, target, back) == (0, [])
    np.testing.assert_array_equal(
        helper.image_to_array(polyxios.read_image(back)), pixels
    )


@pytest.mark.filterwarnings("default")
def test_cli_convert_keeps_a_bare_grid_past_the_mesh_cap_compact(
    temp_polyxios_home, monkeypatch, capsys
):
    """A bare grid of more points than a mesh may hold is still a grid: converted
    between the two image formats it is read and written compact, not expanded."""
    pytest.importorskip("h5py")
    n = 465
    source = temp_polyxios_home / "big.vti"
    polyxios.write_image(
        polyxios.ImageData(extent=(0, n - 1, 0, n - 1, 0, n - 1)), source
    )
    target = temp_polyxios_home / "big.vtkhdf"
    assert _convert(monkeypatch, capsys, source, target) == (0, [])
    assert polyxios.read_image(target).dimensions == (n, n, n)
    back = temp_polyxios_home / "again.vti"
    assert _convert(monkeypatch, capsys, target, back) == (0, [])
    assert polyxios.read_image(back).dimensions == (n, n, n)


@pytest.mark.filterwarnings("default")
def test_cli_convert_to_a_lattice_format_expands_the_image(
    temp_polyxios_home, monkeypatch, capsys
):
    source = temp_polyxios_home / "a.vti"
    target = temp_polyxios_home / "b.vtk"
    _picture(source)
    assert _convert(monkeypatch, capsys, source, target) == (0, [])
    mesh = polyxios.read(target)
    assert len(mesh.vertices) == 24
    assert "colors" in mesh.vertex_attrs


def test_cli_convert_takes_a_mesh_vtkhdf_through_the_mesh_path(
    temp_polyxios_home, monkeypatch, capsys
):
    """A .vtkhdf both reads and writes images, but this one holds cells: the
    image reader refuses it, and the conversion goes on through the mesh path."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "m.vtkhdf"
    out = temp_polyxios_home / "n.vtkhdf"
    mesh = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    polyxios.write(mesh, source)
    refused = []
    read_as_image = polyxios.read_image

    def _spy(path, **opts):
        try:
            return read_as_image(path, **opts)
        except UnsupportedFormatError as exc:
            refused.append(exc)
            raise

    monkeypatch.setattr(polyxios, "read_image", _spy)
    assert _convert(monkeypatch, capsys, source, out) == (0, [])
    assert len(refused) == 1
    assert polyxios.read(out).vertices.shape == (3, 3)


def test_cli_convert_an_empty_vti_takes_the_mesh_path(
    temp_polyxios_home, monkeypatch, capsys
):
    """An empty grid has no image form a writer takes; the mesh path carries it."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "empty.vti"
    source.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 -1 0 1 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="0 -1 0 1 0 0">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    out = temp_polyxios_home / "empty.vtkhdf"
    assert _convert(monkeypatch, capsys, source, out) == (0, [])
    assert polyxios.read(out).vertices.shape == (0, 3)


def test_cli_convert_between_vti_files_keeps_the_whole_extent(
    temp_polyxios_home, monkeypatch, capsys
):
    """A piece of a larger grid goes back out declaring the grid it belongs to."""
    source = temp_polyxios_home / "piece.vti"
    source.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 9 0 4 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="1 3 2 3 0 0">\n'
        '      <PointData Scalars="p">\n'
        '        <DataArray type="Float64" Name="p" NumberOfComponents="1" format="ascii">\n'
        "          0 1 2 3 4 5\n"
        "        </DataArray>\n"
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    out = temp_polyxios_home / "again.vti"
    assert _convert(monkeypatch, capsys, source, out) == (0, [])
    assert 'WholeExtent="0 9 0 4 0 0"' in out.read_text(encoding="utf-8")
    assert polyxios.read_image(out).global_attrs["vti_whole_extent"] == [
        0,
        9,
        0,
        4,
        0,
        0,
    ]


def test_cli_convert_keeps_the_grid_a_field_entry_named_like_a_key_does_not_move(
    temp_polyxios_home, monkeypatch, capsys
):
    """The file's own WholeExtent is the grid; a FieldData entry called
    vti_whole_extent does not replace it, either in .vti or in a .vtkhdf written from
    it, which has no whole extent to drop and so says nothing."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "grid.vti"
    source.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 1 0 1 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <FieldData><DataArray type="Float64" Name="vti_whole_extent" '
        'NumberOfComponents="1" format="ascii">0 5 0 5 0 0</DataArray></FieldData>\n'
        '    <Piece Extent="0 1 0 1 0 0">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    out = temp_polyxios_home / "again.vti"
    assert _convert(monkeypatch, capsys, source, out) == (0, [])
    assert 'WholeExtent="0 1 0 1 0 0"' in out.read_text(encoding="utf-8")
    assert polyxios.read_image(out).extent == (0, 1, 0, 1, 0, 0)

    hdf = temp_polyxios_home / "grid.vtkhdf"
    assert _convert(monkeypatch, capsys, source, hdf) == (0, [])


def _vti_text(extent: str) -> str:
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="{extent}">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )


@pytest.mark.filterwarnings("default")
def test_cli_convert_takes_an_extent_past_int64_by_the_mesh_path(
    temp_polyxios_home, monkeypatch, capsys
):
    """An extent a 64-bit integer cannot hold is a grid the image path refuses; convert
    takes such a file the way it always has, as a mesh, to either target."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "huge.vti"
    source.write_text(_vti_text("9223372036854775808 9223372036854775808 0 0 0 0"))
    assert _convert(monkeypatch, capsys, source, temp_polyxios_home / "o.vti") == (
        0,
        [],
    )
    assert _convert(monkeypatch, capsys, source, temp_polyxios_home / "o.vtkhdf") == (
        0,
        [],
    )


def test_cli_convert_reports_a_corrupt_file_as_the_mesh_path_does(
    temp_polyxios_home, monkeypatch, capsys
):
    """A file the image path refuses for being malformed is read by the mesh path, which
    reports the error it would report on its own."""
    source = temp_polyxios_home / "corrupt.vti"
    source.write_text(_vti_text("0 1 0 1 0 x"))
    with pytest.raises(CodecError) as direct:
        polyxios.read(source)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pxios", "convert", str(source), str(temp_polyxios_home / "o.vti")],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert str(direct.value) in capsys.readouterr().err


def _vti_grid(extent: str, *, origin: str = "0 0 0", point_values=None) -> str:
    """A .vti piece with an optional ``p`` point array over its ``extent``."""
    piece = ""
    if point_values is not None:
        values = " ".join(str(v) for v in point_values)
        piece = (
            '      <PointData Scalars="p">\n'
            '        <DataArray type="Float64" Name="p" NumberOfComponents="1" '
            f'format="ascii">\n          {values}\n        </DataArray>\n'
            "      </PointData>\n"
        )
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="{origin}" Spacing="1 1 1">\n'
        f'    <Piece Extent="{extent}">\n'
        f"{piece}"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )


def _count(warns, text: str) -> int:
    return sum(text in line for line in warns)


@pytest.mark.filterwarnings("default")
def test_cli_convert_says_an_empty_grids_warning_once(
    temp_polyxios_home, monkeypatch, capsys
):
    """An empty grid is read by the image path, which falls back to the mesh path; the
    warning its array raised is said once, by the mesh path."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "empty.vti"
    source.write_text(_vti_grid("0 -1 0 2 0 2", point_values=list(range(27))))
    code, warns = _convert(
        monkeypatch, capsys, source, temp_polyxios_home / "out.vtkhdf"
    )
    assert code == 0
    assert _count(warns, "point array 'p'") == 1
    assert len(warns) == len(set(warns))


@pytest.mark.filterwarnings("default")
def test_cli_convert_says_an_origin_warning_once_past_int64(
    temp_polyxios_home, monkeypatch, capsys
):
    """A grid past int64 falls back to the mesh path, which reads the file's origin
    and says its warning once."""
    pytest.importorskip("h5py")
    huge = "9223372036854775808 9223372036854775808 0 0 0 0"
    source = temp_polyxios_home / "huge.vti"
    source.write_text(_vti_grid(huge, origin="1 2"))
    code, warns = _convert(monkeypatch, capsys, source, temp_polyxios_home / "out.vti")
    assert code == 0
    assert _count(warns, "Origin spells 2 numbers") == 1
    assert len(warns) == len(set(warns))


@pytest.mark.filterwarnings("default")
def test_cli_convert_says_a_kept_images_warning_once(
    temp_polyxios_home, monkeypatch, capsys
):
    """An image the image path keeps says its warning once, replayed from the read."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "short.vti"
    source.write_text(_vti_grid("0 2 0 2 0 0", point_values=list(range(8))))
    code, warns = _convert(
        monkeypatch, capsys, source, temp_polyxios_home / "out.vtkhdf"
    )
    assert code == 0
    assert _count(warns, "point array 'p'") == 1
    assert polyxios.read_image(temp_polyxios_home / "out.vtkhdf").extent == (
        0,
        2,
        0,
        2,
        0,
        0,
    )


def _vti_with_strings(extent: str) -> str:
    """A .vti whose PointData holds two String arrays named ``label``, and a short
    ``p`` array: two different warnings, the first of them raised twice."""
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="{extent}">\n'
        "      <PointData>\n"
        '        <DataArray type="String" Name="label" format="ascii">a</DataArray>\n'
        '        <DataArray type="String" Name="label" format="ascii">b</DataArray>\n'
        '        <DataArray type="Float64" Name="p" NumberOfComponents="1" '
        'format="ascii">0 1 2 3 4 5 6 7</DataArray>\n'
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )


@pytest.mark.filterwarnings("default")
def test_cli_convert_says_a_replayed_warning_once(
    temp_polyxios_home, monkeypatch, capsys
):
    """A warning the image read raised twice is replayed once, as the filters show a
    warning once per place; a different warning from the same read still shows."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "dup.vti"
    source.write_text(_vti_with_strings("0 2 0 2 0 0"))
    code, warns = _convert(monkeypatch, capsys, source, temp_polyxios_home / "o.vti")
    assert code == 0
    assert _count(warns, "has type 'String'") == 1
    assert _count(warns, "point array 'p'") == 1


@pytest.mark.filterwarnings("default")
def test_cli_convert_under_an_error_filter_still_raises_the_replayed_warning(
    temp_polyxios_home, monkeypatch, capsys
):
    """Under an error filter the replayed warning is an exception, as it is for any
    other warning the read raises; convert reports the failure."""
    pytest.importorskip("h5py")
    source = temp_polyxios_home / "dup.vti"
    source.write_text(_vti_with_strings("0 2 0 2 0 0"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        code, _ = _convert(monkeypatch, capsys, source, temp_polyxios_home / "o.vti")
    assert code == 1


@pytest.mark.filterwarnings("default")
def test_cli_convert_a_png_to_a_lattice_format_points_at_read_image(
    temp_polyxios_home, monkeypatch, capsys
):
    source = temp_polyxios_home / "a.png"
    target = temp_polyxios_home / "b.vtk"
    _picture(source)
    code, warnings_shown = _convert(monkeypatch, capsys, source, target)
    assert code == 0
    assert len(warnings_shown) == 1
    assert "read_image()" in warnings_shown[0]
    mesh = polyxios.read(target)
    assert len(mesh.vertices) == 24
    assert "colors" in mesh.vertex_attrs


@pytest.mark.filterwarnings("default")
def test_cli_convert_between_a_png_and_an_image_grid(
    temp_polyxios_home, monkeypatch, capsys
):
    source = temp_polyxios_home / "a.png"
    grid = temp_polyxios_home / "b.vti"
    back = temp_polyxios_home / "c.png"
    pixels = _picture(source)
    assert _convert(monkeypatch, capsys, source, grid) == (0, [])
    assert polyxios.read_image(grid).extent == (0, 5, 0, 3, 0, 0)
    assert _convert(monkeypatch, capsys, grid, back) == (0, [])
    np.testing.assert_array_equal(
        helper.image_to_array(polyxios.read_image(back)), pixels
    )
