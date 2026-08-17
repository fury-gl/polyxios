from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import pytest

from polyxios import fetcher
from polyxios.exceptions import FetcherError
from polyxios.fetcher import (
    _safe_extract_zip,
    fetch,
    get_cached_files,
    get_package_name,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point the fetcher at an empty cache and clear its module-level caches."""
    monkeypatch.setenv("POLYXIOS_HOME", str(tmp_path))
    monkeypatch.setattr("polyxios.fetcher.POLYXIOS_HOME", str(tmp_path))
    monkeypatch.setattr("polyxios.fetcher._catalog_cache", None)
    monkeypatch.setattr("polyxios.fetcher._ext_map_cache", None)
    monkeypatch.setattr("polyxios.fetcher._verified_cache", None)
    return tmp_path


def _write_catalog(home, formats, ext_to_package=None):
    catalog = {"formats": formats, "ext_to_package": ext_to_package or {}}
    (home / "models.json").write_text(json.dumps(catalog), encoding="utf-8")
    return catalog


def _mock_urlopen(payload: bytes):
    class _Response:
        def __init__(self):
            self._data = payload
            self.headers = {"Content-Length": str(len(payload))}

        def read(self, chunk_size=None):
            data, self._data = self._data, b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def _urlopen(req, timeout=None):
        return _Response()

    return _urlopen


def test_get_package_name_builtin_map(home) -> None:
    assert get_package_name("inp") == "abaqus"
    assert get_package_name(".inp") == "abaqus"
    assert get_package_name("xml") == "dolfin"
    assert get_package_name(".meshb") == "medit"
    assert get_package_name("obj") == "obj"


def test_get_package_name_prefers_catalog(home) -> None:
    _write_catalog(home, {}, ext_to_package={"inp": "custom-abaqus"})
    assert get_package_name("inp") == "custom-abaqus"
    assert get_package_name("xml") == "dolfin"


def test_checksum_mismatch_raises_and_leaves_no_file(home, monkeypatch) -> None:
    _write_catalog(
        home,
        {
            "obj": {
                "bunny.obj": {"url": "https://example.com/bunny.obj", "sha256": "00"}
            }
        },
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", _mock_urlopen(b"not the right bytes")
    )

    with pytest.raises(FetcherError, match="Integrity verification failed"):
        fetch("bunny.obj")

    assert not os.path.exists(home / "obj" / "bunny.obj")


def test_non_https_url_is_refused(home, monkeypatch) -> None:
    _write_catalog(
        home,
        {"obj": {"bunny.obj": {"url": "http://example.com/bunny.obj", "sha256": "00"}}},
    )
    called = []
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: called.append(1) or None
    )

    with pytest.raises(FetcherError, match="not https"):
        fetch("bunny.obj")
    assert not called


def test_stale_cached_file_is_redownloaded(home, monkeypatch) -> None:
    """A cached file whose checksum no longer matches the catalog is refetched."""
    obj_dir = home / "obj"
    obj_dir.mkdir()
    (obj_dir / "bunny.obj").write_bytes(b"old contents")

    new_payload = b"new contents"
    _write_catalog(
        home,
        {
            "obj": {
                "bunny.obj": {
                    "url": "https://example.com/bunny.obj",
                    "sha256": hashlib.sha256(new_payload).hexdigest(),
                }
            }
        },
    )
    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen(new_payload))

    path = fetch("bunny.obj")
    assert Path(path).read_bytes() == new_payload


def test_safe_extract_zip_rejects_traversal(home) -> None:
    archive = home / "evil.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    archive.write_bytes(buffer.getvalue())

    target = home / "pkg"
    target.mkdir()
    with pytest.raises(FetcherError, match="resolves outside"):
        _safe_extract_zip(str(archive), str(target))

    assert not (home / "escaped.txt").exists()


def test_safe_extract_zip_extracts_normal_members(home) -> None:
    archive = home / "ok.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("pieces/piece_0.vtp", "content")
    archive.write_bytes(buffer.getvalue())

    target = home / "pkg"
    target.mkdir()
    _safe_extract_zip(str(archive), str(target))
    assert (target / "pieces" / "piece_0.vtp").read_text() == "content"


def test_get_cached_files_lists_without_network(home, monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("get_cached_files must not hit the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    vtp_dir = home / "vtp"
    vtp_dir.mkdir()
    (vtp_dir / "a.vtp").write_text("x")
    (vtp_dir / "b.vtp").write_text("x")
    (vtp_dir / "a.zip").write_text("x")
    (vtp_dir / ".hidden").write_text("x")

    assert get_cached_files("vtp") == [
        str(vtp_dir / "a.vtp"),
        str(vtp_dir / "b.vtp"),
    ]
    assert get_cached_files(".vtp") == get_cached_files("vtp")


def test_get_cached_files_missing_package(home) -> None:
    assert get_cached_files("obj") == []


def test_catalog_entry_without_checksum_is_refused(home, monkeypatch) -> None:
    _write_catalog(home, {"obj": {"bunny.obj": {"url": "https://example.com/b.obj"}}})
    called = []
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: called.append(1) or None
    )

    with pytest.raises(FetcherError, match="no sha256"):
        fetch("bunny.obj")
    assert not called


def test_overwrite_is_not_satisfied_by_cache_when_catalog_is_down(
    home, monkeypatch
) -> None:
    obj_dir = home / "obj"
    obj_dir.mkdir()
    (obj_dir / "bunny.obj").write_bytes(b"cached")

    def _boom(*args, **kwargs):
        raise FetcherError("catalog unreachable")

    monkeypatch.setattr("polyxios.fetcher._load_models_catalog", _boom)

    # Without overwrite the intact cached copy is still good enough.
    assert fetch("bunny.obj") == str(obj_dir / "bunny.obj")

    with pytest.raises(FetcherError, match="catalog unreachable"):
        fetch("bunny.obj", overwrite=True)


def test_missing_companion_refetches_only_the_archive(home, monkeypatch) -> None:
    """An intact asset with unpacked pieces missing pulls the archive alone."""
    payload = b"index contents"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("dataset/piece_0.vtp", "piece")
    zip_payload = buffer.getvalue()

    _write_catalog(
        home,
        {
            "vtp": {
                "dataset.vtp": {
                    "url": "https://example.com/dataset.vtp",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "dataset.zip": {
                    "url": "https://example.com/dataset.zip",
                    "sha256": hashlib.sha256(zip_payload).hexdigest(),
                },
            }
        },
    )

    vtp_dir = home / "vtp"
    vtp_dir.mkdir()
    (vtp_dir / "dataset.vtp").write_bytes(payload)

    requested = []
    zip_urlopen = _mock_urlopen(zip_payload)

    def _urlopen(req, timeout=None):
        requested.append(req.full_url)
        return zip_urlopen(req, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    path = fetch("dataset.vtp")

    assert requested == ["https://example.com/dataset.zip"]
    assert Path(path).read_bytes() == payload
    assert (vtp_dir / "dataset" / "piece_0.vtp").read_text() == "piece"

    # A second call finds the extraction flag and stays off the network.
    requested.clear()
    fetch("dataset.vtp")
    assert requested == []


def _flaky_urlopen(payload: bytes, failures: int, exc: Exception):
    """urlopen that drops the transfer `failures` times, then succeeds."""
    calls = {"n": 0}
    ok = _mock_urlopen(payload)

    def _urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return ok(req, timeout)

    _urlopen.calls = calls
    return _urlopen


def test_dropped_transfer_is_retried(home, monkeypatch) -> None:
    payload = b"mesh bytes"
    _write_catalog(
        home,
        {
            "obj": {
                "cube.obj": {
                    "url": "https://example.invalid/cube.obj",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            }
        },
    )
    flaky = _flaky_urlopen(
        payload, 2, http.client.RemoteDisconnected("Remote end closed connection")
    )
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)

    path = fetch("cube.obj", package="obj")
    assert Path(path).read_bytes() == payload
    assert flaky.calls["n"] == 3


def test_retries_are_bounded(home, monkeypatch) -> None:
    _write_catalog(
        home,
        {
            "obj": {
                "cube.obj": {
                    "url": "https://example.invalid/cube.obj",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                }
            }
        },
    )
    flaky = _flaky_urlopen(b"x", 99, ConnectionResetError("connection reset"))
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)

    with pytest.raises(FetcherError, match="connection reset"):
        fetch("cube.obj", package="obj")
    assert flaky.calls["n"] == fetcher._DOWNLOAD_ATTEMPTS


def test_not_found_is_not_retried(home, monkeypatch) -> None:
    _write_catalog(
        home,
        {
            "obj": {
                "cube.obj": {
                    "url": "https://example.invalid/cube.obj",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                }
            }
        },
    )
    flaky = _flaky_urlopen(
        b"x",
        99,
        urllib.error.HTTPError("https://example.invalid/cube.obj", 404, "", {}, None),
    )
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)

    with pytest.raises(FetcherError, match="not found on remote server"):
        fetch("cube.obj", package="obj")
    assert flaky.calls["n"] == 1


def test_server_error_is_retried(home, monkeypatch) -> None:
    payload = b"mesh bytes"
    _write_catalog(
        home,
        {
            "obj": {
                "cube.obj": {
                    "url": "https://example.invalid/cube.obj",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            }
        },
    )
    flaky = _flaky_urlopen(
        payload,
        1,
        urllib.error.HTTPError("https://example.invalid/cube.obj", 503, "", {}, None),
    )
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetcher.time, "sleep", lambda _s: None)

    assert Path(fetch("cube.obj", package="obj")).read_bytes() == payload
    assert flaky.calls["n"] == 2
