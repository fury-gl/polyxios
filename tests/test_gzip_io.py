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
from polyxios import _io as _io_module
from polyxios._io import read_bytes, source_size
from polyxios.codecs._splat import _SPLAT_DTYPE as SPLAT_DTYPE
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError
from tests.test_buffer_io import (
    _BUFFERABLE,
    _TECPLOT,
    _quietly,
    _same_mesh,
    _Unseekable,
)
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


def test_a_size_the_trailer_understates_is_still_the_size(tmp_path) -> None:
    """gzip's size trailer is smaller than the input for a file that does not
    compress; it is still the size, and a codec dividing by it needs that and
    not the length of the compressed bytes it sits in."""
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

    assert source_size(packed) == len(raw)
    _same_mesh(_quietly(polyxios.read, plain), _quietly(polyxios.read, packed))


def test_a_gz_name_does_not_smuggle_past_a_format_s_own_refusal(tmp_path) -> None:
    """'.plt' is refused for writing; '.plt.gz' is the same file compressed."""
    with pytest.raises(CodecError, match="binary Tecplot flavour"):
        _quietly(polyxios.write, CANONICAL["surface"](), tmp_path / "mesh.plt.gz")


def test_a_gz_fmt_does_not_smuggle_past_a_format_s_own_refusal(tmp_path) -> None:
    """A codec validates its destination by name, so it has to still see one.

    Compressing by wrapping the destination in an open gzip member would hand
    the codec a handle with no name on it, and every guard reading the name -
    the one refusing to write ASCII under a binary format's name - would wave
    through the file it means to refuse.
    """
    poly = CANONICAL["surface"]()

    with pytest.raises(CodecError, match="binary Tecplot flavour"):
        _quietly(polyxios.write, poly, tmp_path / "mesh.plt", fmt=".plt.gz")

    with pytest.raises(CodecError, match="binary 'lb8' UGRID variant"):
        _quietly(polyxios.write, poly, tmp_path / "m.lb8.ugrid", fmt=".ugrid.gz")


def test_a_gz_name_does_not_smuggle_past_a_name_the_codec_reads(tmp_path) -> None:
    """'.gz' names the compression, so the name under it is still the name."""
    with pytest.raises(CodecError, match="binary 'lb8' UGRID variant"):
        _quietly(polyxios.write, CANONICAL["surface"](), tmp_path / "m.lb8.ugrid.gz")


def test_a_destination_a_codec_refuses_is_not_created(tmp_path) -> None:
    """Compressing used to open the destination before the codec saw it, so a
    refusal left a truncated stub where the caller's file had been."""
    dst = tmp_path / "mesh.plt"
    dst.write_bytes(b"THE FILE THAT WAS ALREADY THERE")

    with pytest.raises(CodecError, match="binary Tecplot flavour"):
        _quietly(polyxios.write, CANONICAL["surface"](), dst, fmt=".plt.gz")

    assert dst.read_bytes() == b"THE FILE THAT WAS ALREADY THERE"


def test_a_format_that_opens_its_own_files_refuses_gzip(tmp_path) -> None:
    """TetGen finds its second half beside the first and opens it itself, so
    it never reaches the layer that unwraps gzip: a '.node.gz' would be parsed
    as compressed bytes read as text, and written as a plain file under a name
    promising otherwise."""
    poly = CANONICAL["volume"]()

    with pytest.raises(CodecError, match="gzip is not handled here"):
        _quietly(polyxios.write, poly, tmp_path / "mesh.node.gz")
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(CodecError, match="gzip is not handled here"):
        _quietly(polyxios.write, poly, tmp_path / "mesh.node", fmt=".node.gz")
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(CodecError, match="gzip is not handled here"):
        polyxios.read(tmp_path / "mesh.node.gz")


def test_a_bare_gz_name_says_it_names_no_format(tmp_path) -> None:
    """'.gz' is stripped to choose a codec, and 'mesh.gz' has nothing under
    it - a different problem from an extension nothing is registered under."""
    with pytest.raises(UnsupportedFormatError, match="names the compression"):
        polyxios.read(tmp_path / "mesh.gz")


# ---------------------------------------------------------------------------
# A member that does not run to the end of what it sits in
# ---------------------------------------------------------------------------


def test_a_member_with_data_after_it_ends_where_the_member_ends() -> None:
    """An archive entry has the archive's own bytes after it.

    ``gzip.GzipFile`` reads on past a finished member expecting either another
    one or the end of the file, so those bytes used to raise ``BadGzipFile``
    after the mesh had already decompressed perfectly well.
    """
    poly = CANONICAL["mixed"]()
    plain = io.BytesIO()
    _quietly(polyxios.write, poly, plain, fmt=".vtk")

    prefix = b"ARCHIVE HEADER"
    buf = io.BytesIO(prefix + gzip.compress(plain.getvalue()) + b"NEXT ENTRY")
    buf.seek(len(prefix))

    back = _quietly(polyxios.read, buf, fmt=".vtk")

    np.testing.assert_allclose(back.vertices, poly.vertices)
    assert buf.tell() == len(prefix), "the handle is left at the member's front"


def test_a_size_is_the_member_s_and_not_the_bytes_that_follow_it() -> None:
    """gzip's size trailer is the last four bytes of the file it belongs to.

    Those are some other data's bytes for a member sitting inside a larger
    buffer, and the number they spell is not a size at all - a codec checking
    a header against it would wave through anything.
    """
    payload = b"MESH" * 64
    buf = io.BytesIO(gzip.compress(payload) + b"TRAILING JUNK AFTER THE MEMBER")

    assert source_size(buf) == len(payload)
    assert buf.tell() == 0


def test_a_member_larger_than_a_chunk_decompresses_whole() -> None:
    """Decompression is bounded, so a member is handed out a chunk at a time
    rather than expanded into memory whole - a chunk of compressed input can
    carry a thousand times its own size."""
    payload = b"MESHLINE " * 200_000
    buf = io.BytesIO(gzip.compress(payload))

    assert read_bytes(buf) == payload

    buf.seek(0)
    bites: list[bytes] = []
    with _io_module.open_read(buf) as fh:
        while chunk := fh.read(7_777):
            bites.append(chunk)

    assert b"".join(bites) == payload


def test_members_written_back_to_back_read_as_one_stream() -> None:
    """A 'cat a.gz b.gz' file is one gzip stream, and stays one here."""
    buf = io.BytesIO(gzip.compress(b"first ") + gzip.compress(b"second"))

    assert read_bytes(buf) == b"first second"


def test_members_larger_than_a_chunk_read_back_to_back() -> None:
    """The two properties above, crossed - which is where they used to break.

    A member is decompressed a chunk at a time, and at the end of one zlib
    reports the bytes past it twice: once as input it did not get to, once as
    data past the member. Taking both put the next member into the feed
    twice, so it decompressed twice and left a copy behind to do it again -
    a read that never ended and grew without bound. Under a chunk a member
    ends in a single step and neither test above could see it.
    """
    first = b"MESHLINE " * 20_000
    second = bytes(range(256)) * 400
    assert len(first) > _io_module._GZIP_CHUNK
    assert len(second) > _io_module._GZIP_CHUNK

    packed = gzip.compress(first) + gzip.compress(second)

    assert read_bytes(io.BytesIO(packed)) == first + second
    assert source_size(io.BytesIO(packed)) == len(first) + len(second)

    # Three of them, read a chunk at a time rather than in one call, and then
    # rewound: a run that re-fed itself would diverge on any of the three.
    third = gzip.compress(first) + gzip.compress(second) + gzip.compress(first)
    whole = first + second + first
    with _io_module.open_read(io.BytesIO(third)) as fh:
        bites: list[bytes] = []
        while chunk := fh.read(7_777):
            bites.append(chunk)
        assert b"".join(bites) == whole
        fh.seek(0)
        assert fh.read() == whole


def test_a_member_larger_than_a_chunk_ends_where_the_member_ends() -> None:
    """Trailing bytes after a big member are still not decompressed as one."""
    payload = b"MESHLINE " * 20_000
    assert len(payload) > _io_module._GZIP_CHUNK
    buf = io.BytesIO(gzip.compress(payload) + b"TRAILING JUNK AFTER THE MEMBER")

    assert read_bytes(buf) == payload


def test_members_written_back_to_back_are_measured_whole(tmp_path) -> None:
    """The size of a 'cat a.gz b.gz' file is both members, not the last one.

    gzip's four-byte trailer records the size of the member it ends, so
    reading it off the end of a multi-member file reports the last member
    alone - and the reader above decodes them all. The two answers have to
    agree or a codec measures one file and parses another.
    """
    packed = tmp_path / "pair.gz"
    packed.write_bytes(gzip.compress(b"A" * 1000) + gzip.compress(b"B" * 3000))

    assert source_size(packed) == 4000
    assert len(read_bytes(packed)) == 4000

    buf = io.BytesIO(packed.read_bytes())
    assert source_size(buf) == 4000


def test_a_multi_member_splat_is_not_rejected_as_a_bad_size(tmp_path) -> None:
    """The size skew above reached the user as 'not a valid .splat file'.

    '.splat' is 32 bytes a record and checks the file size against that, so a
    two-member file measured by its last member alone was refused outright.
    """
    poly = CANONICAL["points"]()
    plain = tmp_path / "cloud.splat"
    _quietly(polyxios.write, poly, plain)

    raw = plain.read_bytes()
    cut = 32 * (len(raw) // 64) + 16  # a split that lands mid-record
    packed = tmp_path / "cloud.splat.gz"
    packed.write_bytes(gzip.compress(raw[:cut]) + gzip.compress(raw[cut:]))

    # '.splat' keeps points and drops topology, so the vertices are what
    # survives a round trip and what the size check would have cost us.
    back = _quietly(polyxios.read, packed)
    np.testing.assert_allclose(back.vertices, poly.vertices, rtol=1e-3)


def test_a_broken_gzip_path_fails_the_way_a_broken_gzip_buffer_does(
    tmp_path,
) -> None:
    """One broken file, one error, whichever way it arrived.

    ``gzip.GzipFile`` decodes lazily, so a path cut short used to surface as
    a bare ``EOFError`` and one that was never gzip as ``BadGzipFile`` -
    neither of them the ``CodecError`` that ``read()`` documents, and neither
    of them what the same bytes raise through a buffer.
    """
    whole = gzip.compress(b"MESH" * 4096)
    cut = whole[: len(whole) // 2]
    # A header that reads as gzip all the way through, over a body that does
    # not. Two magic bytes with junk behind them are not this: they are not a
    # member at all, and are read as the content they are.
    rotten = whole[:10] + bytes(b ^ 0xFF for b in whole[10:])

    for payload, message in (
        (cut, "part-way through a gzip member"),
        (rotten, "not readable as gzip"),
    ):
        packed = tmp_path / "broken.gz"
        packed.write_bytes(payload)

        with pytest.raises(CodecError, match=message):
            read_bytes(packed)
        with pytest.raises(CodecError, match=message):
            read_bytes(io.BytesIO(payload))
        with pytest.raises(CodecError, match=message):
            source_size(packed)


def test_two_magic_bytes_are_not_a_member(tmp_path) -> None:
    """A gzip header is four bytes, and a mesh may open with the first two.

    1f 8b opens one file in every 65536 by chance. In a headerless binary
    format they are an ordinary coordinate's low mantissa bytes: a '.splat'
    whose first x is 10.658965 opens with exactly them, and reading it as a
    broken archive loses a file that was never compressed to begin with.
    """
    splat = np.zeros(2, dtype=SPLAT_DTYPE)
    splat["x"] = np.frombuffer(b"\x1f\x8b\x2a\x41", dtype="<f4")[0]
    payload = splat.tobytes()
    assert payload[:2] == b"\x1f\x8b", "the point of the test"

    back = polyxios.read(io.BytesIO(payload), fmt=".splat")
    assert back.vertices.shape == (2, 3)
    np.testing.assert_allclose(back.vertices[0, 0], splat["x"][0])

    packed = tmp_path / "splat_that_is_not_gzip.splat"
    packed.write_bytes(payload)
    np.testing.assert_array_equal(polyxios.read(packed).vertices, back.vertices)


def test_a_member_that_stops_half_way_is_reported() -> None:
    """Truncated compressed data is an error, not a short mesh."""
    whole = gzip.compress(b"MESH" * 256)
    buf = io.BytesIO(whole[: len(whole) // 2])

    with pytest.raises(CodecError, match="part-way through a gzip member"):
        read_bytes(buf)


def test_a_compressed_stream_that_cannot_seek_is_refused_where_sniffing_needs_it() -> (
    None
):
    """'.dat' is shared, so the opening has to be read to choose a codec.

    A gzip wrapper answers 'seekable' for itself and not for the stream under
    it, so asking the wrapper let a compressed unseekable stream through to
    fail on the rewind with a bare ``io.UnsupportedOperation``.
    """
    stream = io.BufferedReader(_Unseekable(gzip.compress(_TECPLOT)))

    with pytest.raises(CodecError, match="cannot seek back"):
        polyxios.read(stream, fmt=".dat")


def test_a_stream_refused_for_sniffing_keeps_its_opening() -> None:
    """The refusal comes before a byte is spent, so the caller can retry."""
    stream = io.BufferedReader(_Unseekable(_TECPLOT))

    with pytest.raises(CodecError, match="cannot seek back"):
        polyxios.read(stream, fmt=".dat")

    assert stream.read() == _TECPLOT


# ---------------------------------------------------------------------------
# Asking a nameless buffer for compression
# ---------------------------------------------------------------------------


def test_a_gz_in_fmt_compresses_a_nameless_buffer() -> None:
    """``io.BytesIO`` has no name to end in '.gz', so ``fmt`` says it."""
    poly = CANONICAL["surface"]()
    plain = io.BytesIO()
    _quietly(polyxios.write, poly, plain, fmt=".obj")

    packed = io.BytesIO()
    _quietly(polyxios.write, poly, packed, fmt=".obj.gz")

    assert packed.getvalue()[:2] == b"\x1f\x8b"
    assert gzip.decompress(packed.getvalue()) == plain.getvalue()
    assert not packed.closed


def test_a_gz_in_fmt_names_the_compression_not_the_format() -> None:
    """'.obj.gz' resolves the '.obj' codec rather than falling off the
    registry as an extension nothing is registered under."""
    poly = CANONICAL["surface"]()
    packed = io.BytesIO()
    _quietly(polyxios.write, poly, packed, fmt="obj.gz")
    packed.seek(0)

    back = _quietly(polyxios.read, packed, fmt=".obj.gz")

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_gz_in_fmt_does_not_compress_a_name_that_already_says_so(
    tmp_path,
) -> None:
    """The suffix asks for compression once, not once per place it appears."""
    poly = CANONICAL["surface"]()
    path = tmp_path / "mesh.obj.gz"
    _quietly(polyxios.write, poly, path, fmt=".obj.gz")

    assert gzip.decompress(path.read_bytes())[:2] != b"\x1f\x8b"
    back = _quietly(polyxios.read, path)
    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_gz_in_fmt_compresses_a_path_that_does_not_say_so(tmp_path) -> None:
    """A path is named by the caller, and the override outranks the name."""
    poly = CANONICAL["surface"]()
    path = tmp_path / "mesh.obj"
    _quietly(polyxios.write, poly, path, fmt=".obj.gz")

    assert path.read_bytes()[:2] == b"\x1f\x8b"
    back = _quietly(polyxios.read, path, fmt=".obj")
    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_path_and_a_buffer_read_one_compressed_file_the_same_way(
    tmp_path,
) -> None:
    """A path used to go through gzip.GzipFile and a handle through _GzipRun.

    Two readers is two behaviours, and they disagreed on the shapes a real
    gzip file comes in: a member with an archive's own bytes after it read
    whole through a buffer and raised BadGzipFile through a path.
    """
    body = b"v 0 0 0\n" * 50
    shapes = {
        "one member": gzip.compress(body),
        "members back to back": gzip.compress(body[:8]) + gzip.compress(body[8:]),
        "a member then other bytes": gzip.compress(body) + b"NOTAMEMBER",
    }

    for name, payload in shapes.items():
        packed = tmp_path / "shape.gz"
        packed.write_bytes(payload)

        assert read_bytes(packed) == body, name
        assert read_bytes(io.BytesIO(payload)) == body, name
        assert source_size(packed) == len(body), name
        assert source_size(io.BytesIO(payload)) == len(body), name


def test_a_compressed_header_is_walked_line_by_line_without_quadratic_cost(
    tmp_path,
) -> None:
    """PLY and VTK read their headers straight off the handle, a line at a time.

    A raw stream's inherited readline takes one byte per call and re-slices
    the decompressed block for each, so a header cost time in the square of
    the block size. The reader is buffered to keep that a scan over bytes
    already in hand.
    """
    header = b"property float x\n" * 2000
    packed = tmp_path / "many.gz"
    packed.write_bytes(gzip.compress(header + b"end_header\n"))

    with _io_module.open_read(packed) as fh:
        lines = [fh.readline() for _ in range(2000)]

    assert lines[0] == b"property float x\n"
    assert len(lines) == 2000
    # A buffered reader is what makes the loop above a scan rather than a
    # byte-at-a-time walk; asserting the wrapper keeps the fix from being
    # dropped as an implementation detail.
    assert isinstance(fh, io.BufferedReader)


def test_a_handle_onto_a_gz_name_is_not_told_it_has_no_name() -> None:
    """'.gz' says how a file is packed and not what is in it - for both.

    The nameless-buffer case was asked first, so a handle onto 'mesh.gz' -
    which has a name, just not a useful one - was told it had none.
    """
    named = io.BytesIO(gzip.compress(b"x"))
    named.name = "mesh.gz"

    with pytest.raises(UnsupportedFormatError, match="names the compression"):
        polyxios.read(named)


def test_a_gzip_stand_in_reads_through_to_the_destination_it_stands_for() -> None:
    """``is_buffer`` answers for the destination, like every other helper.

    A ``fmt='.obj.gz'`` puts a stand-in in front of the caller's destination.
    Reading the stand-in itself rather than through it called a path a buffer,
    which is the wrong answer for any codec branching on it.
    """
    target = _io_module._GzipTarget("mesh.obj")

    assert not _io_module.is_buffer(target)
    assert _io_module.is_buffer(_io_module._GzipTarget(io.BytesIO()))


def test_a_destination_that_compresses_itself_is_not_compressed_twice(
    tmp_path,
) -> None:
    """A caller's own gzip handle is already the compression, not a name.

    ``gzip.open(p, 'wb')`` carries the '.gz' path it opened as its name, and
    that name is what a plain ``open(p, 'wb')`` carries too - where it means
    the opposite. Reading the name alone buried a member inside a member, and
    the file read back as a mesh of no vertices with nothing raised.
    """
    poly = CANONICAL["surface"]()

    for fmt in (".obj", ".obj.gz"):
        path = tmp_path / "own.obj.gz"
        with gzip.open(path, "wb") as gz:
            _quietly(polyxios.write, poly, gz, fmt=fmt)

        assert gzip.decompress(path.read_bytes())[:2] != b"\x1f\x8b", fmt
        np.testing.assert_allclose(
            _quietly(polyxios.read, path).vertices, poly.vertices
        )


def test_a_plain_handle_onto_a_gz_name_is_still_compressed(tmp_path) -> None:
    """The fix above reads what the handle is, not what it is called.

    A binary handle onto a '.gz' path is not compressing anything itself, so
    polyxios still owes it the compression its name asks for.
    """
    poly = CANONICAL["surface"]()
    path = tmp_path / "plain.obj.gz"
    with open(path, "wb") as fh:
        _quietly(polyxios.write, poly, fh, fmt=".obj")

    assert path.read_bytes()[:2] == b"\x1f\x8b"
    np.testing.assert_allclose(_quietly(polyxios.read, path).vertices, poly.vertices)


def _gzip_in_place(path) -> None:
    """Compress a file, keeping the name it already has."""
    path.write_bytes(gzip.compress(path.read_bytes()))


def test_a_format_that_opens_its_own_files_sees_gzip_by_content(tmp_path) -> None:
    """Everywhere else the content decides, so a file compressed without being
    renamed reads as gzip. TetGen opens its own files and so cannot unwrap
    one, but it must still say that is what it is holding rather than parse
    the compressed bytes as text and complain about a malformed header."""
    poly = CANONICAL["volume"]()
    _quietly(polyxios.write, poly, tmp_path / "mesh.node")
    _gzip_in_place(tmp_path / "mesh.node")
    _gzip_in_place(tmp_path / "mesh.ele")

    with pytest.raises(CodecError, match="holds gzip-compressed data"):
        _quietly(polyxios.read, tmp_path / "mesh.node")


def test_a_gzipped_second_half_is_named_and_not_parsed(tmp_path) -> None:
    """Only one half of the pair is ever named by the caller, so the half
    found beside it is looked at too."""
    poly = CANONICAL["volume"]()
    _quietly(polyxios.write, poly, tmp_path / "mesh.node")
    _gzip_in_place(tmp_path / "mesh.ele")

    with pytest.raises(CodecError, match="'mesh.ele' holds gzip-compressed"):
        _quietly(polyxios.read, tmp_path / "mesh.node")


def test_a_plain_pair_is_untouched_by_the_content_check(tmp_path) -> None:
    """The guard reads four bytes and gets out of the way."""
    poly = CANONICAL["volume"]()
    _quietly(polyxios.write, poly, tmp_path / "mesh.node")

    back = _quietly(polyxios.read, tmp_path / "mesh.node")

    _same_mesh(back, _quietly(polyxios.read, tmp_path / "mesh.ele"))


def test_writing_over_a_compressed_file_is_not_refused(tmp_path) -> None:
    """The content check is for the way in only. A destination holds either
    nothing or the file about to be replaced, and refusing to overwrite a
    compressed one would refuse a perfectly ordinary write."""
    poly = CANONICAL["volume"]()
    _quietly(polyxios.write, poly, tmp_path / "mesh.node")
    _gzip_in_place(tmp_path / "mesh.node")
    _gzip_in_place(tmp_path / "mesh.ele")

    _quietly(polyxios.write, poly, tmp_path / "mesh.node")

    back = _quietly(polyxios.read, tmp_path / "mesh.node")
    assert back.vertices.shape == poly.vertices.shape


def test_a_missing_tetgen_half_still_reports_itself(tmp_path) -> None:
    """A path that cannot be opened peeks as empty, so the codec's own error
    is what the caller sees - not a claim about compression."""
    with pytest.raises(CodecError, match="not found"):
        _quietly(polyxios.read, tmp_path / "absent.node")
