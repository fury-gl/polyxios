"""Reading and writing through a file object instead of a path.

``polyxios.read`` and ``polyxios.write`` take an open binary handle wherever
they take a path, so a mesh can round-trip through ``io.BytesIO``, a socket's
``makefile('rb')`` or a file inside an archive without ever touching disk.

The parametrised tests below are the contract: for every writable format, a
buffer must produce the same bytes as a path, and reading those bytes back must
produce the same mesh. Anything a handle cannot do - inferring a format it has
no name for, mmap over an in-memory buffer, a format split across two files -
has its own test naming what is refused and why.

The mesh table lives in ``tests/test_roundtrip.py``: this file is about the
plumbing, not about what each format keeps.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest

import polyxios
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError
from tests.test_roundtrip import CANONICAL, CAPABILITIES

# TetGen is the one format that is not a stream: a mesh is a '.node' and an
# '.ele' file, and the second half is found beside the first by name.
_NEEDS_A_PATH: tuple[str, ...] = (".node", ".ele")

_BUFFERABLE: tuple[str, ...] = tuple(
    ext for ext in sorted(CAPABILITIES) if ext not in _NEEDS_A_PATH
)


def _quietly(fn, *args, **kwargs):
    """Run a codec call whose warnings the round-trip matrix already asserts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _same_mesh(a, b) -> None:
    """Assert two PolyData carry the same geometry and the same fields."""
    np.testing.assert_array_equal(a.vertices, b.vertices)
    np.testing.assert_array_equal(a.connectivity, b.connectivity)
    np.testing.assert_array_equal(a.offsets, b.offsets)
    np.testing.assert_array_equal(a.element_types, b.element_types)
    assert sorted(a.vertex_attrs) == sorted(b.vertex_attrs)
    assert sorted(a.element_attrs) == sorted(b.element_attrs)
    for name, values in a.vertex_attrs.items():
        np.testing.assert_array_equal(values, b.vertex_attrs[name])
    for name, values in a.element_attrs.items():
        np.testing.assert_array_equal(values, b.element_attrs[name])
    for name, ids in a.vertex_tags.items():
        np.testing.assert_array_equal(ids, b.vertex_tags[name])
    for name, ids in a.element_tags.items():
        np.testing.assert_array_equal(ids, b.element_tags[name])


# ---------------------------------------------------------------------------
# The matrix: a buffer does what a path does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", _BUFFERABLE)
def test_writing_to_a_buffer_matches_writing_to_a_path(tmp_path, ext: str) -> None:
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    on_disk = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, on_disk)

    buf = io.BytesIO()
    _quietly(polyxios.write, poly, buf, fmt=ext)

    assert buf.getvalue() == on_disk.read_bytes()


@pytest.mark.parametrize("ext", _BUFFERABLE)
def test_reading_from_a_buffer_matches_reading_from_a_path(tmp_path, ext: str) -> None:
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    on_disk = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, on_disk)
    from_path = _quietly(polyxios.read, on_disk)

    from_buffer = _quietly(polyxios.read, io.BytesIO(on_disk.read_bytes()), fmt=ext)

    _same_mesh(from_path, from_buffer)


@pytest.mark.parametrize("ext", _BUFFERABLE)
def test_a_mesh_round_trips_through_memory_alone(ext: str) -> None:
    """Write to a buffer, read the same buffer back; disk is never involved."""
    poly = CANONICAL[CAPABILITIES[ext].mesh]()

    buf = io.BytesIO()
    _quietly(polyxios.write, poly, buf, fmt=ext)
    buf.seek(0)
    back = _quietly(polyxios.read, buf, fmt=ext)

    assert back.vertices.shape[1] == 3


# ---------------------------------------------------------------------------
# What a handle may and may not assume
# ---------------------------------------------------------------------------


def test_a_nameless_buffer_needs_the_format_named() -> None:
    with pytest.raises(UnsupportedFormatError, match="no file name"):
        polyxios.read(io.BytesIO(b"whatever"))
    with pytest.raises(UnsupportedFormatError, match="no file name"):
        polyxios.write(CANONICAL["surface"](), io.BytesIO())


def test_an_open_file_carries_its_own_extension(tmp_path) -> None:
    """A handle from ``open()`` has a name, so fmt= is not needed."""
    poly = CANONICAL["surface"]()
    path = tmp_path / "mesh.obj"
    _quietly(polyxios.write, poly, path)

    with path.open("rb") as fh:
        back = _quietly(polyxios.read, fh)

    assert back.vertices.shape == poly.vertices.shape


def test_a_text_handle_works_for_a_text_format(tmp_path) -> None:
    """An ASCII format has nothing to lose to a handle that decodes for it."""
    poly = CANONICAL["surface"]()
    path = tmp_path / "mesh.obj"
    _quietly(polyxios.write, poly, path)

    with path.open("r") as fh:
        back = _quietly(polyxios.read, fh)

    assert back.vertices.shape == poly.vertices.shape

    out = io.StringIO()
    _quietly(polyxios.write, poly, out, fmt=".obj")
    assert out.getvalue().startswith("# Written by polyxios")


def test_a_text_handle_is_refused_where_bytes_are_needed(tmp_path) -> None:
    """A binary section decoded as text is corrupt, so the handle is refused."""
    path = tmp_path / "mesh.ply"
    _quietly(polyxios.write, CANONICAL["surface"](), path)

    with path.open("r") as fh:
        with pytest.raises(CodecError, match="text mode"):
            polyxios.read(fh)

    with pytest.raises(CodecError, match="text mode"):
        _quietly(polyxios.write, CANONICAL["surface"](), io.StringIO(), fmt=".ply")


def test_a_caller_s_handle_is_left_open() -> None:
    """polyxios closes what it opens, and nothing that it was handed."""
    poly = CANONICAL["surface"]()

    out = io.BytesIO()
    _quietly(polyxios.write, poly, out, fmt=".obj")
    assert not out.closed

    src = io.BytesIO(out.getvalue())
    _quietly(polyxios.read, src, fmt=".obj")
    assert not src.closed


def test_writing_appends_where_the_handle_stands() -> None:
    """A handle is written at its position, so a caller can prepend a header."""
    buf = io.BytesIO()
    buf.write(b"### caller's own preamble\n")
    _quietly(polyxios.write, CANONICAL["surface"](), buf, fmt=".obj")

    assert buf.getvalue().startswith(b"### caller's own preamble\n")
    assert b"\nv " in buf.getvalue()


# ---------------------------------------------------------------------------
# The formats that need more than a stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", _NEEDS_A_PATH)
def test_tetgen_refuses_a_file_object(ext: str) -> None:
    """Two files, one handle: the pair cannot be found without a directory."""
    with pytest.raises(CodecError, match="a file object is not enough"):
        polyxios.read(io.BytesIO(b"0 3 0 0\n"), fmt=ext)
    with pytest.raises(CodecError, match="a file object is not enough"):
        _quietly(polyxios.write, CANONICAL["volume"](), io.BytesIO(), fmt=ext)


def test_lazy_reading_an_in_memory_buffer_is_refused() -> None:
    """mmap needs a file descriptor, and a BytesIO has none."""
    buf = io.BytesIO()
    _quietly(polyxios.write, CANONICAL["mixed"](), buf, fmt=".vtk", binary=True)
    buf.seek(0)

    with pytest.raises(LazyReadError, match="no file descriptor"):
        polyxios.read(buf, fmt=".vtk", lazy=True)


def test_lazy_reading_a_real_file_handle_still_maps(tmp_path) -> None:
    """A handle over a regular file has a descriptor, so lazy still works."""
    poly = CANONICAL["mixed"]()
    path = tmp_path / "mesh.vtk"
    _quietly(polyxios.write, poly, path, binary=True)

    with path.open("rb") as fh:
        back = _quietly(polyxios.read, fh, lazy=True)

    np.testing.assert_allclose(back.vertices, poly.vertices)


# ---------------------------------------------------------------------------
# Content sniffing over a buffer
# ---------------------------------------------------------------------------

_TECPLOT = (
    b'TITLE = "buffer"\n'
    b'VARIABLES = "X" "Y" "Z"\n'
    b'ZONE T="z", N=3, E=1, F=FEPOINT, ET=TRIANGLE\n'
    b"0.0 0.0 0.0\n"
    b"1.0 0.0 0.0\n"
    b"0.0 1.0 0.0\n"
    b"1 2 3\n"
)


def test_a_contested_extension_sniffs_a_buffer_and_rewinds() -> None:
    """The sniffer reads the opening bytes and gives them back to the codec."""
    back = _quietly(polyxios.read, io.BytesIO(_TECPLOT), fmt=".dat")

    assert back.vertices.shape == (3, 3)
    assert len(back.element_types) == 1


class _Unseekable(io.RawIOBase):
    """A read-only stream with no seek, like a socket or a pipe."""

    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b) -> int:
        return self._buf.readinto(b)


def test_a_contested_extension_refuses_a_stream_that_cannot_rewind() -> None:
    """Sniffing spends bytes the codec needs, and this stream cannot give them
    back; saying so beats handing the codec a file missing its opening."""
    stream = io.BufferedReader(_Unseekable(_TECPLOT))

    with pytest.raises(CodecError, match="cannot seek back"):
        polyxios.read(stream, fmt=".dat")


def test_an_unambiguous_format_reads_from_a_stream_that_cannot_rewind() -> None:
    """Only the sniffing dispatcher needs to seek; a named format does not."""
    obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    stream = io.BufferedReader(_Unseekable(obj))

    back = _quietly(polyxios.read, stream, fmt=".obj")

    assert back.vertices.shape == (3, 3)


# ---------------------------------------------------------------------------
# What a handle is allowed to be missing
# ---------------------------------------------------------------------------


class _BareRead:
    """A source offering nothing but ``read`` - no seek, no readline, no mode.

    The public contract asks a source for a binary ``read`` and no more, so
    everything the codecs and ``io.TextIOWrapper`` reach for beyond that has
    to be supplied rather than assumed.
    """

    def __init__(self, payload: bytes, name: str | None = None) -> None:
        self._buf = io.BytesIO(payload)
        if name is not None:
            self.name = name

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)


_OBJ = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def test_a_handle_with_nothing_but_read_is_enough() -> None:
    """A codec reading line by line must not need io.IOBase underneath it."""
    back = _quietly(polyxios.read, _BareRead(_OBJ, name="mesh.obj"))

    assert back.vertices.shape == (3, 3)


def test_a_handle_that_cannot_say_whether_it_seeks_is_refused_politely() -> None:
    """A source with no 'seekable' is a stream, not an AttributeError."""
    with pytest.raises(CodecError, match="cannot seek back"):
        polyxios.read(_BareRead(_TECPLOT, name="mesh.dat"))


def test_a_handle_that_cannot_seek_is_refused_where_a_size_is_needed() -> None:
    """Formats that check a header against the file size need to measure it."""
    stream = io.BufferedReader(_Unseekable(b"solid x\nendsolid x\n"))

    with pytest.raises(CodecError, match="cannot seek"):
        polyxios.read(stream, fmt=".stl", lazy=True)


class _SeekableBareRead(_BareRead):
    """A bare source that can seek: the codecs that re-read must be able to."""

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._buf.seek(offset, whence)

    def tell(self) -> int:
        return self._buf.tell()

    def seekable(self) -> bool:
        return True


def test_a_bare_handle_that_seeks_is_re_read_from_the_start() -> None:
    """A '.vtk' walks the header and then parses the body from the top, so a
    source wrapped for the io protocol has to move the source itself."""
    poly = CANONICAL["mixed"]()
    written = io.BytesIO()
    _quietly(polyxios.write, poly, written, fmt=".vtk")

    back = _quietly(polyxios.read, _SeekableBareRead(written.getvalue(), "m.vtk"))

    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_a_handle_is_read_from_where_it_stands() -> None:
    """A mesh at an offset into a larger buffer is the mesh that is read."""
    buf = io.BytesIO(b"NOT A MESH AT ALL" + _OBJ)
    buf.seek(17)

    back = _quietly(polyxios.read, buf, fmt=".obj")

    assert back.vertices.shape == (3, 3)


# ---------------------------------------------------------------------------
# Lazy reads, which need a mapping and so need the top of a file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", (".stl", ".ply", ".vtk"))
def test_a_lazy_read_of_a_handle_at_its_start_maps_the_file(tmp_path, ext) -> None:
    """A handle over a real file is as mappable as the path that opened it."""
    poly = CANONICAL["mixed"]()
    path = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, path, binary=True)

    with path.open("rb") as fh:
        back = _quietly(polyxios.read, fh, fmt=ext, lazy=True)

    _same_mesh(_quietly(polyxios.read, path, fmt=ext, lazy=True), back)


@pytest.mark.parametrize("ext", (".ply", ".vtk"))
def test_a_lazy_read_of_a_handle_part_way_in_is_refused(tmp_path, ext) -> None:
    """mmap addresses a file from byte zero and starts only on an allocation
    boundary, so a handle standing part-way in cannot be mapped from where it
    stands. Mapping from the top instead read a header field off the padding:
    a mesh came back with no elements at all and no error to say why.

    Only the formats that hand back arrays viewing the mapping are refused.
    STL's lazy mode copies what it reads - it skips vertex deduplication and
    nothing else - so it has its own test saying it reads either way."""
    poly = CANONICAL["mixed"]()
    path = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, path, binary=True)
    padded = tmp_path / f"padded{ext}"
    padded.write_bytes(b"NOT THE MESH" + path.read_bytes())

    with padded.open("rb") as fh:
        fh.seek(len(b"NOT THE MESH"))
        with pytest.raises(LazyReadError, match="standing at byte"):
            polyxios.read(fh, fmt=ext, lazy=True)


def test_a_lazy_read_that_copies_needs_no_mapping(tmp_path) -> None:
    """STL's lazy mode only skips deduplication, so it needs no descriptor.

    Refusing a buffer here would send a caller to ``lazy=False``, which for
    this format is not the same read: it merges vertices, so the mesh that
    came back would differ from the one the lazy read was asked for.
    """
    poly = CANONICAL["mixed"]()
    path = tmp_path / "mesh.stl"
    _quietly(polyxios.write, poly, path, binary=True)
    reference = _quietly(polyxios.read, path, fmt=".stl", lazy=True)

    buffered = _quietly(
        polyxios.read, io.BytesIO(path.read_bytes()), fmt=".stl", lazy=True
    )
    _same_mesh(reference, buffered)

    padded = tmp_path / "padded.stl"
    padded.write_bytes(b"NOT THE MESH" + path.read_bytes())
    with padded.open("rb") as fh:
        fh.seek(len(b"NOT THE MESH"))
        _same_mesh(reference, _quietly(polyxios.read, fh, fmt=".stl", lazy=True))


def test_a_lazy_read_of_a_stream_spends_none_of_it_before_refusing() -> None:
    """A stream cannot be put back, so a refusal has to come before the read.

    Looking at the opening to decide costs the opening, and a caller told to
    fall back to an eager read would find those bytes already gone.
    """
    payload = b"solid x\nendsolid x\n"
    stream = io.BufferedReader(_Unseekable(payload))

    with pytest.raises((LazyReadError, CodecError)):
        polyxios.read(stream, fmt=".stl", lazy=True)

    assert stream.read() == payload, "the stream is still whole"


def test_a_broken_file_reports_itself_and_not_the_mapping(tmp_path) -> None:
    """A path is mapped and a buffer is not, and both have to fail alike.

    A parse that gives up half-way leaves an array or two viewing the block,
    still alive in the traceback carrying the failure - and closing a mapping
    whose pages are exported raises BufferError. That reached the caller in
    place of the codec's own error, so the same broken file said one thing
    read from a path and another read from a buffer.
    """
    poly = CANONICAL[CAPABILITIES[".ply"].mesh]()
    good = tmp_path / "good.ply"
    _quietly(polyxios.write, poly, good, binary=True)

    # The header stays intact and the body loses bytes, so the failure lands
    # inside the decode with the arrays already built.
    raw = bytearray(good.read_bytes())
    del raw[-7:]
    broken = tmp_path / "broken.ply"
    broken.write_bytes(bytes(raw))

    from_path: Exception | None = None
    from_buffer: Exception | None = None
    try:
        _quietly(polyxios.read, broken)
    except Exception as exc:  # noqa: BLE001 - the type is the assertion
        from_path = exc
    try:
        _quietly(polyxios.read, io.BytesIO(bytes(raw)), fmt=".ply")
    except Exception as exc:  # noqa: BLE001
        from_buffer = exc

    assert from_path is not None
    assert not isinstance(from_path, BufferError)
    assert type(from_path) is type(from_buffer)
    assert str(from_path) == str(from_buffer)


@pytest.mark.parametrize("ext", (".stl", ".ply", ".vtk"))
def test_an_eager_read_of_a_handle_part_way_in_still_reads(tmp_path, ext) -> None:
    """Only the mapping needs the top of the file; reading does not."""
    poly = CANONICAL["mixed"]()
    path = tmp_path / f"mesh{ext}"
    _quietly(polyxios.write, poly, path, binary=True)
    padded = tmp_path / f"padded{ext}"
    padded.write_bytes(b"NOT THE MESH" + path.read_bytes())

    with padded.open("rb") as fh:
        fh.seek(len(b"NOT THE MESH"))
        back = _quietly(polyxios.read, fh, fmt=ext)

    np.testing.assert_allclose(
        np.sort(back.vertices, axis=0),
        np.sort(_quietly(polyxios.read, path, fmt=ext).vertices, axis=0),
    )


class _Trickle(_BareRead):
    """A bare source that answers a read with less than it was asked for.

    A socket, a pipe and an HTTP response all do this: ``read(n)`` hands back
    what has arrived, not what was wanted. A handle from ``open()`` never
    does, so the codecs ask for n bytes and count on n.
    """

    _MOST: int = 3

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._buf.read()
        return self._buf.read(min(size, self._MOST))


def test_a_source_that_answers_short_is_read_in_full() -> None:
    """A short answer from the source must not become a short answer to the
    codec: the wrapper asks again until the read is satisfied or the source
    is spent."""
    from polyxios._io import open_read

    payload = bytes(range(256)) * 4

    with open_read(_Trickle(payload)) as fh:
        assert fh.read(200) == payload[:200]
        assert fh.read(200) == payload[200:400]
        assert fh.read() == payload[400:]
        assert fh.read(10) == b""


def test_a_source_that_answers_short_still_reads_a_mesh() -> None:
    """The whole point of the wrapper: a trickling stream reads like a file."""
    back = _quietly(polyxios.read, _Trickle(_OBJ, name="mesh.obj"))

    assert back.vertices.shape == (3, 3)


def test_nothing_is_read_past_what_was_asked_for() -> None:
    """The source is left exactly where the codec's reading leaves it, so a
    handle shared with the caller keeps the bytes the codec did not take."""
    from polyxios._io import open_read

    src = _Trickle(b"HEAD" + b"TAIL" * 4)

    with open_read(src) as fh:
        assert fh.read(4) == b"HEAD"

    assert src.read() == b"TAIL" * 4
