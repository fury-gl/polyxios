"""Path-or-buffer input and output shared by every codec.

A codec never opens a file itself. It asks this module for the bytes or the
text behind whatever the caller passed - a path, or an already-open file
object - so that widening the public API to buffers costs each codec one
call site instead of one rewrite.

Three rules hold everywhere:

- a handle the caller owns is never closed here, and is left wherever the
  codec's reading or writing leaves it - except over a gzip member, which is
  rewound to the front of the member for the reason ``open_read`` gives;
- a path is opened and closed inside the helper that was given it;
- a handle is read from and written at wherever it stands, never from the
  start of the buffer it happens to sit in - so a mesh reached through an
  archive member or an offset into a larger stream is the mesh that is read.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import gzip
import io
import mmap
import os
from pathlib import Path
import stat as _stat
from typing import IO, Any
import zlib

from polyxios.exceptions import CodecError, LazyReadError

# What read() and write() accept in place of a path. A file object is only
# ever duck-typed - a binary ``read`` is all a source has to offer and a binary
# ``write`` all a destination does - so the alias documents the intent rather
# than gating on a type. Anything short of the full io protocol is wrapped on
# the way in, so a handle need not subclass io.IOBase to be read line by line.
Source = str | os.PathLike[str] | IO[Any]

# A gzip member opens with these two bytes, followed by the compression
# method and the flag byte. Reading them is what decides whether a source is
# decompressed on the way in; the file name is only consulted on the way out,
# where there is no content to look at yet.
#
# All four are checked rather than the magic alone, because two bytes are not
# evidence: 1f 8b opens one file in every 65536 by chance, and in a headerless
# binary format they are an ordinary coordinate's low mantissa bytes - a
# '.splat' whose first x is 10.658965 opens with exactly them. Deflate is the
# only compression method gzip defines and the top three flag bits are
# reserved and zero, so a real member never fails this and a mesh that only
# looks like one almost never passes it.
_GZIP_MAGIC: bytes = b"\x1f\x8b"
_GZIP_HEADER: int = 4
_GZIP_DEFLATE: int = 8
_GZIP_FLG_RESERVED: int = 0b1110_0000

# Suffixes that mean "the format is the one before me". Stripped when a
# format is inferred, so 'mesh.vol.gz' resolves to the .vol codec.
GZIP_SUFFIXES: tuple[str, ...] = (".gz", ".gzip")

# Deflate level for the files polyxios compresses. 9 costs several times the
# CPU of 6 for a per cent or two of size on mesh text, which is the wrong
# trade for a file measured in hundreds of megabytes.
_GZIP_LEVEL: int = 6

# What to hand zlib so it reads a gzip wrapper rather than a bare deflate
# stream, and how much compressed input to feed it at a time.
_ZLIB_GZIP_WBITS: int = 16 + zlib.MAX_WBITS
_GZIP_CHUNK: int = 1 << 16

__all__ = [
    "GZIP_SUFFIXES",
    "Source",
    "can_seek",
    "format_suffix",
    "is_buffer",
    "is_gzip",
    "map_read",
    "open_block",
    "open_read",
    "open_target",
    "open_text",
    "open_write",
    "read_bytes",
    "read_text",
    "require_path",
    "source_name",
    "source_size",
    "source_suffix",
    "strip_gzip",
    "wants_gzip",
    "write_bytes",
    "write_text",
]


class _GzipTarget:
    """A destination to compress into, standing in for the caller's own.

    ``fmt='.obj.gz'`` asks for compression that the destination's own name
    does not spell - a nameless buffer has no name to spell it in. Handing the
    codec an open gzip member would answer that and cost two things. The codec
    would lose the name it validates its output against, so a guard reading
    the destination's extension - the one refusing to write ASCII under a
    binary format's name - would wave through the file it means to refuse. And
    the destination would be opened, and a path truncated, before the codec
    had the chance to refuse it at all, leaving a stub behind.

    So the request is carried rather than the handle: the name reads through
    this to the destination underneath, and ``open_write`` compresses only
    once the codec asks for somewhere to write.
    """

    __slots__ = ("dst",)

    def __init__(self, dst: Source) -> None:
        self.dst = dst


def _unwrap(src: Source) -> Source:
    """Return the destination a source stands in for, or the source itself."""
    return src.dst if isinstance(src, _GzipTarget) else src


def _opens_gzip(head: bytes) -> bool:
    """Say whether these opening bytes start a gzip member.

    Parameters
    ----------
    head
        The first bytes of a source, however few of them there are.

    Returns
    -------
    bool
        True when the magic, the compression method and the flag byte all
        read as a gzip header. Fewer bytes than a header is False: what
        cannot be checked is not compressed data anyone can read.
    """
    if len(head) < _GZIP_HEADER or head[: len(_GZIP_MAGIC)] != _GZIP_MAGIC:
        return False
    return head[2] == _GZIP_DEFLATE and not head[3] & _GZIP_FLG_RESERVED


def is_buffer(src: Source) -> bool:
    """Say whether a source is an open file object rather than a path.

    Parameters
    ----------
    src
        Path or file object handed to a codec.

    Returns
    -------
    bool
        True for a file object, False for anything ``os.fspath`` accepts. A
        destination standing in for another - the one a gzip ``fmt=`` puts in
        front of a path - answers for the destination underneath it, the same
        way ``source_name`` and ``source_suffix`` read through it. Reading the
        stand-in itself would call a path a buffer.
    """
    return not isinstance(_unwrap(src), (str, os.PathLike))


def source_name(src: Source) -> str:
    """Name a source for an error message.

    Parameters
    ----------
    src
        Path or file object handed to a codec.

    Returns
    -------
    str
        The file name for a path, the handle's own ``name`` when it has a
        usable one, and ``'<buffer>'` for a nameless handle - an in-memory
        buffer has nothing better to be called.
    """
    src = _unwrap(src)
    if not is_buffer(src):
        return Path(os.fspath(src)).name  # type: ignore[arg-type]
    name = getattr(src, "name", None)
    # A raw file descriptor is exposed as an int name; that is not a name a
    # message can show, and neither is anything else non-textual.
    if isinstance(name, str) and name:
        return Path(name).name
    return "<buffer>"


def source_suffix(src: Source) -> str:
    """Return the lower-case extension of a source, dot included.

    Parameters
    ----------
    src
        Path or file object handed to a codec.

    Returns
    -------
    str
        The suffix, or an empty string when the source has no name to take
        one from. A codec branching on the suffix must treat the empty
        string as 'unknown', never as 'not that format'.
    """
    src = _unwrap(src)
    if not is_buffer(src):
        return Path(os.fspath(src)).suffix.lower()  # type: ignore[arg-type]
    name = getattr(src, "name", None)
    if isinstance(name, str) and name:
        return Path(name).suffix.lower()
    return ""


def format_suffix(src: Source) -> str:
    """Return the extension that names a source's format, past any ``.gz``.

    ``.gz`` names the compression, not the format, so a codec branching on the
    extension - to tell a binary flavour from a text one, say - has to look at
    the suffix underneath it or it will read ``mesh.plt.gz`` as neither.

    Parameters
    ----------
    src
        Path or file object handed to a codec.

    Returns
    -------
    str
        The lower-case suffix, dot included, of the name with a gzip suffix
        stripped off it. Empty when there is no name to take one from.
    """
    ext = source_suffix(src)
    if ext not in GZIP_SUFFIXES:
        return ext
    name = source_name(src)
    return Path(name[: -len(ext)]).suffix.lower()


def strip_gzip(ext: str) -> str:
    """Return a name or extension with a trailing gzip suffix taken off it.

    ``.gz`` names the compression rather than the format wherever it turns up
    - in a file name or in a ``fmt=`` override - so ``'.vtk.gz'`` names the
    ``.vtk`` codec. A bare ``'.gz'`` has no format under it and is left as it
    is, so it fails as the unknown extension it is rather than as an empty one.

    Parameters
    ----------
    ext
        Extension, format string or whole file name, already lower-cased. A
        codec that tells its flavours apart by an infix rather than by the
        last suffix - ``mesh.lb8.ugrid.gz`` - passes the name.

    Returns
    -------
    str
        The input without its gzip suffix, or unchanged when it has none.
    """
    for gz in GZIP_SUFFIXES:
        if ext.endswith(gz) and len(ext) > len(gz):
            return ext[: -len(gz)]
    return ext


def wants_gzip(fmt: str | None) -> bool:
    """Say whether a format override asks for the output to be compressed.

    Parameters
    ----------
    fmt
        Format override as the caller spelled it, or None.

    Returns
    -------
    bool
        True when the override carries a gzip suffix over a format, as
        ``'.obj.gz'`` does and ``'.gz'`` alone does not.
    """
    if fmt is None:
        return False
    ext = fmt.strip().lower()
    return strip_gzip(ext) != ext


def can_seek(handle: Any) -> bool:
    """Say whether a handle can be seeked, without assuming it says so.

    A file object is only ever duck-typed here, so ``seekable`` is not a
    method every source has. Asking for it directly is what turns a stream
    polyxios means to refuse politely into an ``AttributeError``.

    Parameters
    ----------
    handle
        Open file object, or anything pretending to be one.

    Returns
    -------
    bool
        True only when the handle claims to seek and offers the ``seek`` and
        ``tell`` that claim implies.
    """
    try:
        seekable = handle.seekable
    except AttributeError:
        return False
    try:
        if not seekable():
            return False
    except Exception:
        return False
    return callable(getattr(handle, "seek", None)) and callable(
        getattr(handle, "tell", None)
    )


class _Stream(io.RawIOBase):
    """A duck-typed handle presented as a real raw stream.

    Two jobs, both about handles that are not already ``io`` objects. It puts
    ``head`` - bytes taken off a source that cannot seek back, to look at its
    opening - in front of the stream again; and it supplies the ``readable``,
    ``readline`` and friends that ``io.TextIOWrapper`` and the codecs reach
    for and that a bare ``read`` does not provide.

    Nothing is read ahead: the source ends up exactly where the codec left it,
    which is the promise a caller's handle is given. Seeking is passed through
    to a source that offers it, so a codec that walks a header and then goes
    back to the start moves the source itself and not a copy of it - unless
    bytes are being served from ``head``, which no position over the source
    accounts for. Closing this wrapper does not close the source underneath it.
    """

    def __init__(self, src: Any, *, head: bytes = b"") -> None:
        self._src = src
        self._head = head
        self._seekable = not head and can_seek(src)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return self._seekable

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if not self._seekable:
            raise io.UnsupportedOperation("seek")
        return int(self._src.seek(offset, whence))

    def tell(self) -> int:
        if not self._seekable:
            raise io.UnsupportedOperation("tell")
        return int(self._src.tell())

    def fileno(self) -> int:
        # A duck-typed handle over a real file keeps its descriptor through
        # this wrapper, so a lazy read still maps the file it was given. The
        # inherited RawIOBase.fileno would refuse every wrapped handle alike,
        # and a source with a descriptor would be told it has none.
        fileno = getattr(self._src, "fileno", None)
        if not callable(fileno):
            raise io.UnsupportedOperation("fileno")
        return int(fileno())

    def readinto(self, buf: Any) -> int:
        want = len(buf)
        if not want:
            return 0
        out = b""
        if self._head:
            out = self._head[:want]
            self._head = self._head[len(out) :]
            want -= len(out)
        # Asked for until the buffer is full, not asked once: a bare ``read``
        # is allowed to hand back less than it was asked for - a socket
        # returns what has arrived rather than what was wanted - and a raw
        # stream's ``read(n)`` is a single ``readinto``, so a short answer
        # here is a short answer to the codec. A handle from ``open()`` never
        # gives one, so the codecs ask for n bytes and count on n; this is
        # what makes a duck-typed source keep the same promise. Nothing is
        # read past what was asked for, so the source is still left exactly
        # where the codec's reading leaves it.
        while want > 0:
            chunk = self._src.read(want)
            if not chunk:
                break
            # Clamped: a source handing back more than it was asked for would
            # otherwise overrun the caller's buffer.
            chunk = bytes(chunk)[:want]
            out += chunk
            want -= len(chunk)
        buf[: len(out)] = out
        return len(out)


class _GzipRun(io.RawIOBase):
    """The gzip members a handle opens with, decompressed, and no further.

    ``gzip.GzipFile`` is the obvious reader and the wrong one for a member
    that sits inside something bigger. Two of its habits break there. It
    rewinds by seeking its file object to absolute zero, which for a handle
    standing part-way into a buffer is not where the member began. And once a
    member ends it reads on, expecting either another member or the end of the
    file, so an archive's own bytes after the mesh raise ``BadGzipFile`` rather
    than ending the stream - the mesh decompresses correctly and the read
    fails anyway.

    zlib says how much input it did not use, so this stops decompressing at
    the first thing that is not another gzip member rather than failing on it.
    Concatenated members are still read as the one stream they spell, which is
    what makes a ``cat a.gz b.gz`` file work. Positions are counted over the
    decompressed bytes, and a rewind re-runs the members from the front of the
    first rather than from the front of the buffer.

    The source handle itself is left wherever the block reads ran past the
    end of the last member; putting it back is ``open_read``'s job, and it
    puts it back to the front of the first member - see the note there.
    """

    def __init__(self, src: Any, *, start: int | None, name: str) -> None:
        self._src = src
        self._start = start
        self._name = name
        self._reset()

    def _reset(self) -> None:
        """Start the run over, from the front of the first member."""
        self._dec: Any | None = None
        self._feed = b""
        self._out = b""
        self._spent = False
        self._drained = False
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        # Forward is always reachable by reading; going back means re-running
        # the members, which needs a handle that can be put back to the front.
        return self._start is not None

    def tell(self) -> int:
        return self._pos

    def _more(self) -> bool:
        """Pull another block of compressed input. False when there is none."""
        if self._spent:
            return False
        chunk = self._src.read(_GZIP_CHUNK)
        if not chunk:
            self._spent = True
            return False
        self._feed += bytes(chunk)
        return True

    def _fill(self) -> None:
        """Decompress until there is something to hand out, or the run ends."""
        while not self._out and not self._drained:
            if self._dec is None:
                # Between members: another member may follow, and anything
                # else belongs to whatever the member is sitting in.
                while len(self._feed) < _GZIP_HEADER and self._more():
                    pass
                if not _opens_gzip(self._feed[:_GZIP_HEADER]):
                    self._drained = True
                    break
                self._dec = zlib.decompressobj(_ZLIB_GZIP_WBITS)
                continue
            if self._dec.eof:
                self._feed = self._dec.unused_data + self._feed
                self._dec = None
                continue
            if not self._feed and not self._more():
                raise CodecError(
                    f"'{self._name}' ends part-way through a gzip member; "
                    f"the compressed data is truncated."
                )
            data, self._feed = self._feed, b""
            try:
                # Bounded: a chunk of compressed input can expand a thousand
                # times over, and a whole member's worth of it does not belong
                # in memory to answer a read for a few bytes. What zlib did not
                # get to is fed back on the next turn.
                self._out = self._dec.decompress(data, _GZIP_CHUNK)
            except zlib.error as exc:
                raise CodecError(
                    f"'{self._name}' is not readable as gzip: {exc}"
                ) from exc
            # Only while the member is still running. Once it ends, zlib
            # reports the same trailing bytes twice - as the tail it did not
            # get to and as the data past the member - and taking both puts
            # the next member into the feed twice over, which decompresses it
            # twice and leaves a copy behind to do it again. The member's end
            # is handled by the ``eof`` branch above, which takes those bytes
            # once.
            if not self._dec.eof and self._dec.unconsumed_tail:
                self._feed = self._dec.unconsumed_tail + self._feed

    def readinto(self, buf: Any) -> int:
        want = len(buf)
        if not want:
            return 0
        self._fill()
        out = self._out[:want]
        self._out = self._out[len(out) :]
        self._pos += len(out)
        buf[: len(out)] = out
        return len(out)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_CUR:
            offset += self._pos
        elif whence != os.SEEK_SET:
            raise io.UnsupportedOperation("seek from the end of a gzip stream")
        if offset < 0:
            raise ValueError("negative seek position")
        if offset < self._pos:
            if self._start is None:
                raise io.UnsupportedOperation("seek")
            self._src.seek(self._start)
            self._reset()
        # Forward is reading and dropping: there is no index into a deflate
        # stream to jump with.
        while self._pos < offset:
            step = self.read(min(_GZIP_CHUNK, offset - self._pos))
            if not step:
                break
        return self._pos


def require_path(src: Source, *, fmt: str, reason: str, reading: bool = False) -> Path:
    """Return the source as a Path, or refuse a file object.

    A few formats are not one stream of bytes: TetGen splits a mesh across
    ``.node`` and ``.ele`` siblings, and finding the sibling needs a
    directory to look in. Those codecs call this instead of ``open_read``.

    Parameters
    ----------
    src
        Path or file object handed to a codec.
    fmt
        Extension the message should name, dot included.
    reason
        Why this format needs a real path, phrased to follow 'because'.
    reading
        Set on the way in, where the file already holds bytes to look at, so
        that a compressed one is caught by its content the way it is
        everywhere else in this module and not only by its name. Left unset
        on the way out, where the only thing to go on is the name: a
        destination holds either nothing or the file about to be replaced,
        and reading that one would refuse a plain write for the sake of
        whatever happened to be sitting there.

    Returns
    -------
    Path
        The source as a path.

    Raises
    ------
    CodecError
        If the source is a file object, or carries a gzip name or a ``fmt=``
        asking for compression, or - when ``reading`` is set - holds gzip
        data under any name at all. A format that opens its own files does
        not go through the layer that unwraps gzip, so a compressed one
        would be read as text or written uncompressed under a ``.gz`` name.
    """
    if isinstance(src, _GzipTarget) or source_suffix(src) in GZIP_SUFFIXES:
        raise CodecError(
            f"{fmt}: gzip is not handled here, because {reason}. "
            f"Use uncompressed paths."
        )
    if is_buffer(src):
        raise CodecError(
            f"{fmt}: a file object is not enough, because {reason}. "
            f"Pass a path instead."
        )
    # After the buffer is refused, so this only ever opens a path - and a
    # path that cannot be opened peeks as empty, leaving the codec to raise
    # the real error when it opens the same file itself.
    if reading and is_gzip(src):
        raise CodecError(
            f"{fmt}: '{source_name(src)}' holds gzip-compressed data, and "
            f"gzip is not handled here, because {reason}. Decompress it "
            f"first."
        )
    return Path(os.fspath(src))  # type: ignore[arg-type]


def _is_text_handle(handle: Any) -> bool:
    """Say whether a handle takes and returns str rather than bytes."""
    if isinstance(handle, io.TextIOBase):
        return True
    mode = getattr(handle, "mode", None)
    if isinstance(mode, str) and mode:
        return "b" not in mode
    # Neither a known base class nor a mode string: ask the handle what it
    # holds. An empty read is cheap and tells the truth for StringIO and for
    # any other duck-typed buffer.
    peek = getattr(handle, "read", None)
    if callable(peek):
        try:
            pos = handle.tell()
            sample = handle.read(0)
            handle.seek(pos)
        except Exception:
            return False
        return isinstance(sample, str)
    return False


def _peek(src: Source, n: int) -> bytes:
    """Return the first ``n`` bytes of a source without consuming them.

    Parameters
    ----------
    src
        Path or open binary file object.
    n
        How many bytes to look at.

    Returns
    -------
    bytes
        The opening bytes. Fewer than ``n`` of them means either that the
        source is shorter than that or that it is a stream which can neither
        peek that far nor seek back - a caller that cannot act on a short
        answer has to tell the two apart itself. Nothing is consumed.
    """
    if not is_buffer(src):
        try:
            with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
                return bytes(fh.read(n))
        except OSError:
            # Not this function's error to report: the codec is about to
            # open the same path and will raise the real one.
            return b""

    # Reading and seeking back is the exact answer, so it is preferred over
    # peeking: 'peek' is allowed to return less than it was asked for, and a
    # short answer read as the whole opening is how a compressed file gets
    # parsed as though it were plain.
    if can_seek(src):
        here = src.tell()  # type: ignore[union-attr]
        try:
            return bytes(src.read(n))  # type: ignore[union-attr]
        finally:
            src.seek(here)  # type: ignore[union-attr]

    peek = getattr(src, "peek", None)
    if callable(peek):
        try:
            return bytes(peek(n))[:n]
        except Exception:
            return b""

    return b""


def _read_exactly(src: Any, n: int) -> bytes:
    """Take ``n`` bytes off a stream, or everything left if there are fewer."""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = src.read(n - got)
        if not chunk:
            break
        chunks.append(bytes(chunk))
        got += len(chunk)
    return b"".join(chunks)


def _sniffed(src: Any) -> tuple[bytes, Any]:
    """Return a buffer's opening bytes and the handle to read it through.

    A stream that cannot seek back and whose ``peek`` came up short leaves no
    way to look at the opening without spending it, so the bytes are taken and
    put back in front of the stream. The handle that comes back is the one to
    read from; the caller's own is never left short of what it started with.
    """
    head = _peek(src, _GZIP_HEADER)
    if len(head) == _GZIP_HEADER or can_seek(src):
        return head, src
    head = _read_exactly(src, _GZIP_HEADER)
    return head, _Stream(src, head=head)


def is_gzip(src: Source) -> bool:
    """Say whether a source holds gzip-compressed data.

    The answer comes from the content, not the name: a ``.vol`` file gzipped
    by a well-meaning pipeline reads the same as a ``.vol.gz``, and a file
    named ``.gz`` that is not compressed is not treated as though it were.

    Parameters
    ----------
    src
        Path or open binary file object.

    Returns
    -------
    bool
        True when the source opens with the gzip magic.
    """
    if is_buffer(src) and _is_text_handle(src):
        return False
    return _opens_gzip(_peek(src, _GZIP_HEADER))


def _compresses_already(dst: Source) -> bool:
    """Say whether a destination does its own gzip compression.

    A caller who opened ``gzip.open(path, 'wb')`` is already compressing, and
    the name that handle carries is the ``.gz`` path they opened - which is
    the same thing a plain ``open(path, 'wb')`` onto a ``.gz`` name carries
    and there means the opposite. Told apart by what the handle is rather
    than by what it is called, because compressing into a gzip writer buries
    a member inside a member: the file reads back as a mesh of no vertices,
    with nothing raised anywhere to say so.
    """
    return isinstance(_unwrap(dst), gzip.GzipFile)


def _writes_gzip(dst: Source) -> bool:
    """Say whether a destination's name asks for gzip compression."""
    return source_suffix(dst) in GZIP_SUFFIXES and not _compresses_already(dst)


def _gzip_reader(src: Any, *, start: int | None, name: str) -> IO[bytes]:
    """Return a buffered reader over the gzip members a source opens with.

    ``_GzipRun`` is a raw stream, and a raw stream's ``readline`` is the one
    inherited from ``io.IOBase``: a byte at a time, each byte re-slicing the
    block the decompressor just produced. That is quadratic in the block, and
    the codecs that walk a header line by line - PLY, VTK legacy - walk it
    straight off this handle. The buffer in front turns those reads back into
    one memchr over bytes already in hand.
    """
    return io.BufferedReader(_GzipRun(src, start=start, name=name))


@contextmanager
def open_read(src: Source) -> Iterator[IO[bytes]]:
    """Yield a binary handle for reading a source.

    Parameters
    ----------
    src
        Path to open, or an already-open binary file object to pass through.

    Yields
    ------
    IO[bytes]
        The handle to read from, decompressed when the source opens with the
        gzip magic. A handle the caller owns is yielded as it is and left
        open; a path is closed on the way out.

    Raises
    ------
    CodecError
        If the file object was opened in text mode, or if the source opens
        with the gzip magic and is not readable as gzip all the way through.
    """
    if not is_buffer(src):
        # One open, not two: the magic is read off the handle already in hand
        # rather than by opening the path a second time to look at it.
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            packed = _opens_gzip(fh.read(_GZIP_HEADER))
            fh.seek(0)
            if packed:
                # The same reader a buffer gets, not ``gzip.GzipFile``, so
                # that one compressed file reads the same whichever way it
                # was handed over - the same members, the same stop at the
                # first thing that is not one, the same error when it breaks.
                # GzipFile is the faster of the two by a third of a pass over
                # bytes that inflate at gigabytes a second, which is not worth
                # a file that reads through a path and fails through a handle.
                yield _gzip_reader(fh, start=0, name=source_name(src))
            else:
                yield fh
        return

    if _is_text_handle(src):
        raise CodecError(
            f"'{source_name(src)}' is open in text mode, and this format is "
            f"read as bytes. Open it with mode='rb'."
        )

    head, handle = _sniffed(src)

    if _opens_gzip(head):
        here = int(handle.tell()) if can_seek(handle) else None
        try:
            yield _gzip_reader(handle, start=here, name=source_name(src))
        finally:
            # A decompressor reads ahead in blocks, so where it leaves the
            # handle says nothing about how much of the mesh the codec
            # consumed - a codec that opens the same source twice, as the ones
            # that walk a header before parsing a body do, would find the
            # second read starting in the middle of the compressed stream. The
            # handle goes back to the front of the member instead, which is the
            # only position over compressed bytes that means anything to the
            # next reader.
            if here is not None:
                handle.seek(here)
        return

    # A bare duck type is given the rest of the io protocol - readline and the
    # readable/seekable trio - that the codecs and TextIOWrapper reach for.
    if not isinstance(handle, io.IOBase):
        handle = _Stream(handle)

    yield handle


@contextmanager
def open_write(dst: Source) -> Iterator[IO[bytes]]:
    """Yield a binary handle for writing a source.

    Parameters
    ----------
    dst
        Path to create, or an already-open binary file object to pass
        through.

    Yields
    ------
    IO[bytes]
        The handle to write to, compressed when the destination is named
        ``.gz``. A handle the caller owns is yielded as it is and left open,
        so a caller can keep writing after polyxios is done; a path is
        closed on the way out.

    Raises
    ------
    CodecError
        If the file object was opened in text mode.
    """
    if isinstance(dst, _GzipTarget):
        # The compression a fmt= asked for, applied where the codec finally
        # writes rather than where the caller handed the destination over -
        # unless the destination compresses on its own, in which case the
        # compression asked for is the one already there.
        with open_write(dst.dst) as fh:
            if _compresses_already(dst.dst):
                yield fh
            else:
                with _gzip_writer(fh) as gz:
                    yield gz
        return

    if not is_buffer(dst):
        with open(os.fspath(dst), "wb") as fh:  # type: ignore[arg-type]
            if _writes_gzip(dst):
                with _gzip_writer(fh) as gz:
                    yield gz
            else:
                yield fh
        return

    if _is_text_handle(dst):
        raise CodecError(
            f"'{source_name(dst)}' is open in text mode, and this format is "
            f"written as bytes. Open it with mode='wb'."
        )

    if _writes_gzip(dst):
        with _gzip_writer(dst) as gz:  # type: ignore[arg-type]
            yield gz
        return

    yield dst  # type: ignore[misc]


@contextmanager
def _gzip_writer(fh: IO[bytes]) -> Iterator[IO[bytes]]:
    """Wrap a handle in a gzip member with a reproducible header.

    The name and the timestamp gzip would otherwise embed are what make two
    identical meshes compress to two different files; both are pinned so the
    output depends on the mesh alone.
    """
    gz = gzip.GzipFile(
        filename="", mode="wb", compresslevel=_GZIP_LEVEL, fileobj=fh, mtime=0
    )
    with gz:
        yield gz  # type: ignore[misc]


@contextmanager
def open_target(dst: Source, *, fmt: str | None) -> Iterator[Source]:
    """Yield the destination a codec should write to, compressed if asked.

    A destination named ``.gz`` is compressed by ``open_write`` on the codec's
    own call, which is where a path gets to say what it wants. A nameless
    buffer has no name to say it in - ``io.BytesIO`` cannot end in ``.gz`` -
    so a ``fmt=`` carrying the suffix says it here instead, and the codec is
    handed a gzip member to write into rather than the buffer itself.

    Parameters
    ----------
    dst
        Path or open binary file object the caller passed.
    fmt
        Format override as the caller spelled it, or None.

    Yields
    ------
    Source
        The destination to hand the codec: ``dst`` itself whenever the
        compression is already accounted for, and a stand-in that compresses
        on the codec's own ``open_write`` otherwise. The stand-in is not an
        open handle - see :class:`_GzipTarget` for why the destination is not
        opened here. Nothing the caller owns is closed either way.
    """
    if not wants_gzip(fmt) or _writes_gzip(dst):
        yield dst
        return

    yield _GzipTarget(dst)  # type: ignore[misc]


def read_bytes(src: Source) -> bytes:
    """Read a whole source into memory.

    Parameters
    ----------
    src
        Path or open binary file object.

    Returns
    -------
    bytes
        Everything left in the source.
    """
    with open_read(src) as fh:
        return fh.read()


def read_text(src: Source, *, encoding: str = "utf-8", errors: str = "strict") -> str:
    """Read a whole source and decode it.

    Parameters
    ----------
    src
        Path or open file object. A handle opened in text mode is read
        directly, and its own decoding wins over the arguments here.
    encoding
        Codec name used to decode the bytes.
    errors
        Decoding error policy, as taken by ``bytes.decode``.

    Returns
    -------
    str
        The decoded contents.
    """
    if is_buffer(src) and _is_text_handle(src):
        return src.read()  # type: ignore[union-attr,no-any-return]
    return read_bytes(src).decode(encoding, errors)


def write_bytes(dst: Source, data: bytes) -> None:
    """Write bytes to a path or an open binary file object.

    Parameters
    ----------
    dst
        Path to create, or open binary file object to append to.
    data
        Payload to write.
    """
    with open_write(dst) as fh:
        fh.write(data)


def write_text(dst: Source, text: str, *, encoding: str = "utf-8") -> None:
    """Encode text and write it to a path or an open file object.

    Parameters
    ----------
    dst
        Path to create, or open file object. A handle opened in text mode
        takes the string as it is, and does its own encoding - and its own
        newline translation with it, so on Windows a handle from ``open(p,
        'w')`` turns every ``'\n'`` here into ``'\r\n'``. Every format
        polyxios writes is newline-terminated rather than newline-delimited,
        so that stays readable; a destination that has to be byte-for-byte
        LF - one being hashed or compared - wants ``'wb'`` or a path, both of
        which are written as bytes and translate nothing.
    text
        Payload to write.
    encoding
        Codec name used to encode the text for a binary destination.
    """
    if is_buffer(dst) and _is_text_handle(dst):
        dst.write(text)  # type: ignore[union-attr]
        return
    write_bytes(dst, text.encode(encoding))


def _decompressed_length(src: Source) -> int:
    """Count what a gzip source decompresses to, a chunk at a time.

    Nothing bigger than a chunk is held while it counts, so measuring a file
    costs its decompression and not its size in memory.
    """
    total = 0
    with open_read(src) as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
    return total


def _gzip_size(src: Source) -> int:
    """Return what a gzip source decompresses to, by decompressing it.

    The four-byte ISIZE trailer looks like the cheap answer and is the wrong
    one. It records the size of the member it ends, and a gzip file is allowed
    to hold several members back to back - a ``cat a.gz b.gz`` file is one
    valid gzip stream, and both readers here decode it whole. Taking the
    trailer would report the last member alone: half of a two-member file,
    which a codec then rejects as corrupt or slices short. Nor are those bytes
    a member's at all when a handle holds a member with an archive's own bytes
    after it.

    So the bytes are counted. The cost is one decompression pass with the
    output thrown away, which inflate does at gigabytes a second - cheaper
    than the parse that follows it, and the only answer that is right for
    every gzip file rather than for the common one.

    Raises
    ------
    CodecError
        If the source is a compressed stream that cannot seek - counting it
        would spend the bytes the codec still needs - or, from the read
        itself, if the source is not readable as gzip all the way through.
    """
    if is_buffer(src) and not can_seek(src):
        raise CodecError(
            f"'{source_name(src)}' is a compressed stream that cannot seek, "
            f"and this format needs the file's size before it can parse it. "
            f"Read the stream into io.BytesIO first, or pass a path."
        )
    # A file that is not gzip all the way down fails here rather than in the
    # codec, one read earlier than it otherwise would. ``open_read`` reports
    # it, so there is nothing to translate: a broken file has one error
    # whether it was measured or parsed.
    return _decompressed_length(src)


def source_size(src: Source) -> int:
    """Return the size of a source in bytes.

    Parameters
    ----------
    src
        Path or open binary file object. A handle is measured by seeking to
        its end and back, so an unseekable stream cannot be measured.

    Returns
    -------
    int
        Size of the file for a path; for a handle, the number of bytes it has
        left to offer, counted from where it stands - the same bytes
        ``open_read`` would go on to read. A gzip source reports the size of
        what it decompresses to, which is the number a codec is checking a
        header against, and it is the size itself rather than a bound on it -
        a codec that divides by it gets the same answer as one that compares
        against it.

    Raises
    ------
    CodecError
        If the source is a file object that cannot seek.
    """
    if is_gzip(src):
        return _gzip_size(src)

    if not is_buffer(src):
        return os.stat(os.fspath(src)).st_size  # type: ignore[arg-type]

    if not can_seek(src):
        raise CodecError(
            f"'{source_name(src)}' is a stream that cannot seek, and this "
            f"format needs the file's size before it can parse it. Read the "
            f"stream into io.BytesIO first, or pass a path."
        )
    here = int(src.tell())  # type: ignore[union-attr]
    end = int(src.seek(0, os.SEEK_END))  # type: ignore[union-attr]
    src.seek(here)  # type: ignore[union-attr]
    return end - here


@contextmanager
def open_text(
    src: Source, *, encoding: str = "utf-8", errors: str = "strict"
) -> Iterator[Any]:
    """Yield a text handle for reading a source line by line.

    Reading line by line is the point: a codec that only needs the whole
    string should call ``read_text`` instead. The wrapper this puts around a
    binary handle is detached rather than closed, so a handle the caller owns
    survives the ``with``.

    Parameters
    ----------
    src
        Path or open file object. A handle already in text mode is yielded
        as it is, and its own decoding wins over the arguments here.
    encoding
        Codec name used to decode the bytes.
    errors
        Decoding error policy.

    Yields
    ------
    TextIO
        The handle to iterate.
    """
    if is_buffer(src) and _is_text_handle(src):
        yield src
        return

    if not is_buffer(src) and not is_gzip(src):
        with open(os.fspath(src), encoding=encoding, errors=errors) as fh:  # type: ignore[arg-type]
            yield fh
        return

    with open_read(src) as raw:
        wrapper = io.TextIOWrapper(
            raw if isinstance(raw, io.IOBase) else _Stream(raw),
            encoding=encoding,
            errors=errors,
        )
        try:
            yield wrapper
        finally:
            # Closing the wrapper would close the handle underneath it - the
            # caller's own, or the decompressor open_read is holding for the
            # length of the `with` - so it is detached instead.
            wrapper.flush()
            wrapper.detach()


@contextmanager
def open_block(
    src: Source, *, fmt: str, require_map: bool = False
) -> Iterator[mmap.mmap | bytes]:
    """Yield a whole source as one randomly-addressable block of bytes.

    A path is mapped, so a large binary mesh costs no copy; a file object is
    read into memory instead, because a buffer has no descriptor to map. Both
    answer ``len``, slicing, ``find`` and ``struct.unpack_from`` the same way,
    which is all a binary parser asks of them.

    Parameters
    ----------
    src
        Path or open binary file object.
    fmt
        Extension an error message should name, dot included.
    require_map
        Set by a reader that hands back arrays viewing the block rather than
        copies. Reading such a source into memory would defeat the point, so
        a file object with nothing to map is refused instead, and the mapping
        is left open for the arrays that outlive this call.

    Yields
    ------
    mmap.mmap or bytes
        The whole source. A mapping is closed on the way out unless
        ``require_map`` asked for one to keep.

    Raises
    ------
    LazyReadError
        If ``require_map`` is set and the source cannot be mapped.
    """
    if require_map:
        # Not closed: the caller asked for a mapping precisely because the
        # arrays it hands back view these pages, and closing a mapping whose
        # pages are still exported raises BufferError rather than freeing
        # anything. The mapping goes when the last array viewing it does.
        yield map_read(src, fmt=fmt)
        return

    if not is_buffer(src) and not is_gzip(src):
        mm = map_read(src, fmt=fmt)
        try:
            yield mm
        except BaseException:
            # A parse that fails part-way has usually built an array or two
            # over the block already, and those are still alive in the
            # traceback carrying the failure. Closing a mapping whose pages
            # are exported raises BufferError, which would land in the
            # caller's lap in place of the codec's own error - the same file
            # read from a buffer, where there is no mapping to close, reports
            # what actually went wrong. Nothing leaks by leaving it: the
            # mapping is freed when the last view of it goes.
            try:
                mm.close()
            except BufferError:
                pass
            raise
        mm.close()
        return

    yield read_bytes(src)


def map_read(src: Source, *, fmt: str) -> mmap.mmap:
    """Map a source read-only, for the codecs that decode lazily.

    The mapping is returned rather than yielded because it outlives this
    call: arrays handed back by a lazy read stay backed by it. A path is
    opened and its descriptor closed straight away - the mapping keeps the
    file alive on its own - and a caller's handle is left open and where it
    was.

    Parameters
    ----------
    src
        Path, or a file object over a real file standing at its start. A
        buffer with no file descriptor cannot be mapped.
    fmt
        Extension the error message should name, dot included.

    Returns
    -------
    mmap.mmap
        The whole file, mapped read-only.

    Raises
    ------
    LazyReadError
        If the source is a file object that no file descriptor backs, whose
        descriptor is not a regular file, or that stands part-way into one.
    CodecError
        If the mapping itself fails.
    """
    if is_gzip(src):
        raise LazyReadError(
            f"{fmt}: '{source_name(src)}' is gzip-compressed, and a mapping "
            f"would hand back the compressed bytes. Read it eagerly "
            f"(lazy=False), or decompress it first."
        )

    if not is_buffer(src):
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            return _map_handle(fh, src, fmt=fmt)

    # A stream that cannot seek cannot say where it stands, cannot be put back
    # to the top, and has no descriptor worth mapping - and opening it to find
    # that out spends the opening bytes it can never give back. Refuse it
    # before anything is read off it, so a caller that falls back to an eager
    # read still has the whole stream to read.
    if not can_seek(src):
        raise LazyReadError(
            f"{fmt}: '{source_name(src)}' is a stream that cannot seek, and a "
            f"mapping has to start at the top of a file. Read it eagerly "
            f"(lazy=False), or pass a path."
        )

    # A mapping addresses a file from byte zero, and mmap will only start it
    # at a multiple of the allocation granularity, so a handle standing
    # part-way in cannot be mapped from where it stands. Reading it from the
    # top instead would hand the parser bytes the caller never pointed at -
    # a header field landing on padding, a mesh coming back empty - so say so
    # rather than decode the wrong file quietly.
    if int(src.tell()) != 0:  # type: ignore[union-attr]
        raise LazyReadError(
            f"{fmt}: '{source_name(src)}' is a file object standing at byte "
            f"{int(src.tell())}, and a mapping can only start at the top of "  # type: ignore[union-attr]
            f"the file. Read it eagerly (lazy=False), or pass a path."
        )

    with open_read(src) as fh:
        return _map_handle(fh, src, fmt=fmt)


def _map_handle(fh: IO[bytes], src: Source, *, fmt: str) -> mmap.mmap:
    """Map an open handle, translating the ways that can fail."""
    try:
        fileno = fh.fileno()
        st_mode = os.fstat(fileno).st_mode
    except Exception as exc:
        raise LazyReadError(
            f"{fmt}: '{source_name(src)}' is a file object with no file "
            f"descriptor, and mmap needs one. Read it eagerly (lazy=False), "
            f"or pass a path."
        ) from exc

    if not _stat.S_ISREG(st_mode):
        raise LazyReadError(
            f"{fmt}: '{source_name(src)}' is not a regular file, and mmap "
            f"needs one. Read it eagerly (lazy=False), or pass a path."
        )

    try:
        return mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError) as exc:
        raise CodecError(f"{fmt}: cannot mmap '{source_name(src)}': {exc}") from exc
