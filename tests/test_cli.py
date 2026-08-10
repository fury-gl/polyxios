import json
import sys

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios.cli import main


@pytest.fixture
def temp_polyxios_home(tmp_path, monkeypatch):
    """Fixture to set up a clean, localized POLYXIOS_HOME for CLI testing without network dependencies."""
    monkeypatch.setenv("POLYXIOS_HOME", str(tmp_path))
    monkeypatch.setattr("polyxios.fetcher.POLYXIOS_HOME", str(tmp_path))
    monkeypatch.setattr("polyxios.cli.POLYXIOS_HOME", str(tmp_path))
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

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
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

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
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

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys, "argv", ["pxios", "viz", "armadillo.obj", "--points"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
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

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
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

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
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
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
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
    assert "[obj]" in captured.out
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
    assert "Successfully fetched package to:" in captured.out
    assert str(gmsh_dir) in captured.out


def test_cli_list_extensions(temp_polyxios_home, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
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
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys, "argv", ["pxios", "list", "--codecs"])

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "File formats supported by polyxios codecs:" in captured.out
    assert "obj (.obj)" in captured.out
    assert "vtk (.vtk)" in captured.out
    assert "abaqus (.inp)" in captured.out


def test_cli_fetch_with_zip_companion(temp_polyxios_home, monkeypatch):
    import io
    import os
    import urllib.request
    import zipfile

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

    from polyxios.fetcher import fetch

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
    monkeypatch.setattr("polyxios.cli._load_models_catalog", _boom)
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
