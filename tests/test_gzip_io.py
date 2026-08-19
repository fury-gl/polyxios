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


@pytest.mark.parametrize("ext", _GZIPPABLE)
def test_a_gzipped_buffer_reads_like_a_gzipped_path(tmp_path, ext: str) -> None:
    """The path matrix above says nothing about a handle.

    A codec that walks a header and then re-opens the source to parse the body
    reads it twice, and over a compressed handle the second read has to start
    where the first one did - not wherever the decompressor's read-ahead left
    the bytes underneath it.
    """
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    plain = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, plain)
    packed = io.BytesIO(gzip.compress(plain.read_bytes()))

    _same_mesh(
        _quietly(polyxios.read, plain),
        _quietly(polyxios.read, packed, fmt=ext),
    )


@pytest.mark.parametrize("ext", _GZIPPABLE)
def test_a_gzipped_buffer_reads_from_where_it_stands(tmp_path, ext: str) -> None:
    """A member reached at an offset is the member that is read."""
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    plain = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, plain)
    prefix = b"NOT THE MESH"
    packed = io.BytesIO(prefix + gzip.compress(plain.read_bytes()))
    packed.seek(len(prefix))

    _same_mesh(
        _quietly(polyxios.read, plain),
        _quietly(polyxios.read, packed, fmt=ext),
    )


# ---------------------------------------------------------------------------
# Detecting compression on sources that make it hard
# ---------------------------------------------------------------------------


class _Drip(io.RawIOBase):
    """A stream handing back one byte at a time, like a socket mid-transfer.

    A ``BufferedReader`` over it answers ``peek(2)`` with a single byte, which
    is the whole point: ``peek`` is allowed to return less than it was asked
    for, and a short answer must not be read as 'this is not compressed'.
    """

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b) -> int:
        chunk = self._buf.read(1)
        b[: len(chunk)] = chunk
        return len(chunk)


def test_a_stream_whose_peek_falls_short_is_still_seen_as_gzip() -> None:
    """A one-byte peek is not an answer; the opening has to be spent and
    given back, or the codec parses compressed bytes as though they were the
    format and hands back an empty mesh without an error."""
    poly = CANONICAL["surface"]()
    plain = io.BytesIO()
    _quietly(polyxios.write, poly, plain, fmt=".obj")

    stream = io.BufferedReader(_Drip(gzip.compress(plain.getvalue())))
    assert len(stream.peek(2)) < 2

    back = _quietly(polyxios.read, stream, fmt=".obj")

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_gzip_member_part_way_into_a_handle_reads() -> None:
    """gzip rewinds to absolute zero, which is not where the member began.

    ``.vtk`` is the format to test it with: its reader walks the header and
    then puts the handle back at the start, which is the rewind that used to
    land on the front of the buffer rather than on the front of the member -
    and left the codec parsing the prefix into an empty mesh, silently.
    """
    poly = CANONICAL["mixed"]()
    plain = io.BytesIO()
    _quietly(polyxios.write, poly, plain, fmt=".vtk")

    prefix = b"PREFIX" * 4
    buf = io.BytesIO(prefix + gzip.compress(plain.getvalue()))
    buf.seek(len(prefix))

    back = _quietly(polyxios.read, buf, fmt=".vtk")

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_size_the_trailer_understates_is_counted(tmp_path) -> None:
    """gzip's size trailer is smaller than the input for a file that does not
    compress; a codec dividing by the size cannot take that number."""
    # One splat is 32 bytes: three float32 coordinates and then the scale,
    # rotation and colour fields, filled here with bytes gzip cannot shrink
    # past its own header and trailer - so the trailer's size, which is what
    # gzip records, lands below the size of the input it belongs to.
    coords = np.array([1.0, 2.0, 3.0], dtype="<f4").tobytes()
    noise = np.random.default_rng(0).integers(0, 256, 20, dtype=np.uint8).tobytes()
    raw = coords + noise

    plain = tmp_path / "mesh.splat"
    plain.write_bytes(raw)
    packed = tmp_path / "mesh.splat.gz"
    packed.write_bytes(gzip.compress(raw))
    assert packed.stat().st_size > len(raw), "fixture must be incompressible"

    assert source_size(packed, exact=True) == len(raw)
    _same_mesh(_quietly(polyxios.read, plain), _quietly(polyxios.read, packed))


def test_a_gz_name_does_not_smuggle_past_a_format_s_own_refusal(tmp_path) -> None:
    """'.plt' is refused for writing; '.plt.gz' is the same file compressed."""
    with pytest.raises(CodecError, match="binary Tecplot flavour"):
        _quietly(polyxios.write, CANONICAL["surface"](), tmp_path / "mesh.plt.gz")
