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
