"""Transparent gzip, for every format at once.

Compression is handled by the IO layer, not by any codec: a source opening
with the gzip magic is decompressed on the way in, and a destination named
``.gz`` is compressed on the way out. A codec sees plain bytes either way,
so a format gains ``.gz`` support without a line of its own.

The tests below are parametrised over the same table the round-trip matrix
uses, so a new codec is covered the day it lands.
"""

from __future__ import annotations

import gzip
import io

import numpy as np
import pytest

import polyxios
from polyxios._io import source_size
from polyxios.exceptions import CodecError, LazyReadError
from tests.test_buffer_io import _BUFFERABLE, _quietly, _same_mesh
from tests.test_roundtrip import CANONICAL, CAPABILITIES

# TetGen writes two files, and .plt is refused before any byte is written;
# both are covered by their own tests and have nothing to say about gzip.
_GZIPPABLE: tuple[str, ...] = _BUFFERABLE


@pytest.mark.parametrize("ext", _GZIPPABLE)
def test_a_gzipped_file_reads_like_a_plain_one(tmp_path, ext: str) -> None:
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    plain = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, plain)
    packed = tmp_path / f"mesh{ext}.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))

    _same_mesh(_quietly(polyxios.read, plain), _quietly(polyxios.read, packed))


@pytest.mark.parametrize("ext", _GZIPPABLE)
def test_writing_a_gz_name_compresses(tmp_path, ext: str) -> None:
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    plain = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, plain)
    packed = tmp_path / f"mesh{ext}.gz"
    _quietly(polyxios.write, poly, packed)

    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert gzip.decompress(packed.read_bytes()) == plain.read_bytes()


@pytest.mark.parametrize("ext", _GZIPPABLE)
def test_a_mesh_round_trips_through_gzip(tmp_path, ext: str) -> None:
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    plain = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, plain)
    packed = tmp_path / f"mesh{ext}.gz"
    _quietly(polyxios.write, poly, packed)

    _same_mesh(_quietly(polyxios.read, plain), _quietly(polyxios.read, packed))


def test_the_compressed_file_is_the_same_every_time(tmp_path) -> None:
    """No timestamp, no embedded name: the bytes depend on the mesh alone."""
    poly = CANONICAL["surface"]()
    first = tmp_path / "a.obj.gz"
    second = tmp_path / "b.obj.gz"
    _quietly(polyxios.write, poly, first)
    _quietly(polyxios.write, poly, second)

    assert first.read_bytes() == second.read_bytes()


def test_gz_names_the_compression_not_the_format(tmp_path) -> None:
    """'mesh.vol.gz' is a Netgen file, and resolves to the Netgen codec."""
    poly = CANONICAL["volume"]()
    path = tmp_path / "mesh.vol.gz"
    _quietly(polyxios.write, poly, path)

    back = _quietly(polyxios.read, path)

    assert back.vertices.shape == poly.vertices.shape


def test_gzip_is_recognised_by_content_not_by_name(tmp_path) -> None:
    """A file compressed without being renamed still reads."""
    poly = CANONICAL["surface"]()
    plain = tmp_path / "mesh.obj"
    _quietly(polyxios.write, poly, plain)

    # Same name, compressed content - what a well-meaning pipeline leaves.
    misnamed = tmp_path / "packed.obj"
    misnamed.write_bytes(gzip.compress(plain.read_bytes()))

    back = _quietly(polyxios.read, misnamed)

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_gz_name_over_plain_content_is_read_as_it_is(tmp_path) -> None:
    """A name promising compression does not make a file compressed."""
    poly = CANONICAL["surface"]()
    plain = tmp_path / "plain.obj"
    _quietly(polyxios.write, poly, plain)

    misnamed = tmp_path / "mesh.obj.gz"
    misnamed.write_bytes(plain.read_bytes())

    back = _quietly(polyxios.read, misnamed)

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_gzipped_buffer_reads(tmp_path) -> None:
    poly = CANONICAL["surface"]()
    plain = tmp_path / "mesh.obj"
    _quietly(polyxios.write, poly, plain)

    buf = io.BytesIO(gzip.compress(plain.read_bytes()))
    back = _quietly(polyxios.read, buf, fmt=".obj")

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_writing_a_gz_named_handle_compresses(tmp_path) -> None:
    """A handle is compressed on its own name, the way a path is."""
    poly = CANONICAL["surface"]()
    path = tmp_path / "mesh.obj.gz"

    with path.open("wb") as fh:
        _quietly(polyxios.write, poly, fh)

    assert path.read_bytes()[:2] == b"\x1f\x8b"
    np.testing.assert_allclose(_quietly(polyxios.read, path).vertices, poly.vertices)


def test_lazy_reading_a_compressed_file_is_refused(tmp_path) -> None:
    """A mapping would hand back the compressed bytes, so it is refused."""
    poly = CANONICAL["mixed"]()
    plain = tmp_path / "mesh.vtk"
    _quietly(polyxios.write, poly, plain, binary=True)
    packed = tmp_path / "mesh.vtk.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))

    with pytest.raises(LazyReadError, match="gzip-compressed"):
        polyxios.read(packed, lazy=True)


def test_a_header_is_checked_against_the_decompressed_size(tmp_path) -> None:
    """Compressed bytes are the wrong yardstick for what a header claims."""
    poly = CANONICAL["mixed"]()
    plain = tmp_path / "mesh.vtk"
    _quietly(polyxios.write, poly, plain)
    packed = tmp_path / "mesh.vtk.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))

    assert source_size(packed) >= plain.stat().st_size


def test_an_empty_gzip_member_is_reported_by_the_codec(tmp_path) -> None:
    """Decompression is transparent, so an empty file fails as an empty file."""
    path = tmp_path / "mesh.vol.gz"
    path.write_bytes(gzip.compress(b""))

    with pytest.raises(CodecError, match="empty file"):
        _quietly(polyxios.read, path)
