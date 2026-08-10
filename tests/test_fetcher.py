from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile

import pytest

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
    assert open(path, "rb").read() == new_payload


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
