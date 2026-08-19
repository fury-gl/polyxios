"""Path-or-buffer input and output shared by every codec.

A codec never opens a file itself. It asks this module for the bytes or the
text behind whatever the caller passed - a path, or an already-open file
object - so that widening the public API to buffers costs each codec one
call site instead of one rewrite.

Three rules hold everywhere:

- a handle the caller owns is never closed here, and is left wherever the
  codec's reading or writing leaves it;
- a path is opened and closed inside the helper that was given it;
- a handle is read from and written at wherever it stands, never from the
  start of the buffer it happens to sit in - so a mesh reached through an
  archive member or an offset into a larger stream is the mesh that is read.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import gzip
import io
import mmap
import os
from pathlib import Path
import stat as _stat
from typing import IO, Any

from polyxios.exceptions import CodecError, LazyReadError

# What read() and write() accept in place of a path. A file object is only
# ever duck-typed - a binary ``read`` is all a source has to offer and a binary
# ``write`` all a destination does - so the alias documents the intent rather
# than gating on a type. Anything short of the full io protocol is wrapped on
# the way in, so a handle need not subclass io.IOBase to be read line by line.
Source = str | os.PathLike[str] | IO[Any]

# A gzip member opens with these two bytes. Reading them is what decides
# whether a source is decompressed on the way in; the file name is only
# consulted on the way out, where there is no content to look at yet.
_GZIP_MAGIC: bytes = b"\x1f\x8b"

# Suffixes that mean "the format is the one before me". Stripped when a
# format is inferred, so 'mesh.vol.gz' resolves to the .vol codec.
GZIP_SUFFIXES: tuple[str, ...] = (".gz", ".gzip")

# Deflate level for the files polyxios compresses. 9 costs several times the
# CPU of 6 for a per cent or two of size on mesh text, which is the wrong
# trade for a file measured in hundreds of megabytes.
_GZIP_LEVEL: int = 6

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
    "open_text",
    "open_write",
    "read_bytes",
    "read_text",
    "require_path",
    "source_name",
    "source_size",
    "source_suffix",
    "write_bytes",
    "write_text",
]


def is_buffer(src: Source) -> bool:
    """Say whether a source is an open file object rather than a path.

    Parameters
    ----------
    src
        Path or file object handed to a codec.

    Returns
    -------
    bool
        True for a file object, False for anything ``os.fspath`` accepts.
    """
    return not isinstance(src, (str, os.PathLike))


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

    def readinto(self, buf: Any) -> int:
        want = len(buf)
        if not want:
            return 0
        out = b""
        if self._head:
            out = self._head[:want]
            self._head = self._head[len(out) :]
            want -= len(out)
        if want:
            chunk = self._src.read(want)
            if chunk:
                out += bytes(chunk)
        buf[: len(out)] = out
        return len(out)


class _Window:
    """The tail of a seekable handle, presented as a stream of its own.

    ``gzip`` rewinds by seeking its file object to absolute zero, which for a
    handle standing part-way into a buffer is not where the member began -
    the decompressor would resume over whatever came before it and hand back
    an empty mesh rather than an error. Position zero is mapped back onto
    wherever the handle stood instead.
    """

    def __init__(self, src: Any, start: int) -> None:
        self._src = src
        self._start = start

    def read(self, size: int = -1) -> bytes:
        return bytes(self._src.read(size))

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            return int(self._src.seek(self._start + offset)) - self._start
        return int(self._src.seek(offset, whence)) - self._start

    def tell(self) -> int:
        return int(self._src.tell()) - self._start

    def seekable(self) -> bool:
        return True


def require_path(src: Source, *, fmt: str, reason: str) -> Path:
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

    Returns
    -------
    Path
        The source as a path.

    Raises
    ------
    CodecError
        If the source is a file object.
    """
    if is_buffer(src):
        raise CodecError(
            f"{fmt}: a file object is not enough, because {reason}. "
            f"Pass a path instead."
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
    head = _peek(src, len(_GZIP_MAGIC))
    if len(head) == len(_GZIP_MAGIC) or can_seek(src):
        return head, src
    head = _read_exactly(src, len(_GZIP_MAGIC))
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
    return _peek(src, 2) == _GZIP_MAGIC


def _writes_gzip(dst: Source) -> bool:
    """Say whether a destination's name asks for gzip compression."""
    return source_suffix(dst) in GZIP_SUFFIXES


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
        If the file object was opened in text mode.
    """
    if not is_buffer(src):
        # One open, not two: the magic is read off the handle already in hand
        # rather than by opening the path a second time to look at it.
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            packed = fh.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC
            fh.seek(0)
            if packed:
                with gzip.GzipFile(fileobj=fh, mode="rb") as gz:
                    yield gz  # type: ignore[misc]
            else:
                yield fh
        return

    if _is_text_handle(src):
        raise CodecError(
            f"'{source_name(src)}' is open in text mode; mesh files are read "
            f"as bytes. Open it with mode='rb'."
        )

    head, handle = _sniffed(src)

    if head == _GZIP_MAGIC:
        # Closing the wrapper flushes it without closing what it wraps, so
        # the caller's handle survives the decompression.
        base, restore = _gzip_base(handle)
        try:
            with gzip.GzipFile(fileobj=base, mode="rb") as gz:
                yield gz  # type: ignore[misc]
        finally:
            restore()
        return

    # A bare duck type is given the rest of the io protocol - readline and the
    # readable/seekable trio - that the codecs and TextIOWrapper reach for.
    if not isinstance(handle, io.IOBase):
        handle = _Stream(handle)

    yield handle  # type: ignore[misc]


def _gzip_base(handle: Any) -> tuple[Any, Callable[[], None]]:
    """Return what to hand ``gzip``, and how to put the handle back after.

    Two things a decompressor does to the handle underneath it have to be
    undone. It rewinds by seeking to absolute zero, which for a handle
    standing part-way into a buffer is not where the member began, so the
    handle is windowed and zero mapped back onto where it stood. And it reads
    ahead in blocks, so where it leaves the handle says nothing about how much
    of the mesh the codec consumed - a codec that opens the same source twice,
    as the ones that walk a header before parsing a body do, would find the
    second read starting in the middle of the compressed stream. The handle
    goes back to the front of the member instead, which is the only position
    over compressed bytes that means anything to the next reader.

    A stream that cannot seek has neither problem to fix and nothing that
    could be put back; it is passed through.
    """
    if not can_seek(handle):
        return handle, lambda: None
    here = int(handle.tell())
    base = _Window(handle, here) if here else handle
    return base, lambda: handle.seek(here)


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
            f"'{source_name(dst)}' is open in text mode; mesh files are "
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
        takes the string as it is, and does its own encoding.
    text
        Payload to write.
    encoding
        Codec name used to encode the text for a binary destination.
    """
    if is_buffer(dst) and _is_text_handle(dst):
        dst.write(text)  # type: ignore[union-attr]
        return
    write_bytes(dst, text.encode(encoding))


def _compressed_span(src: Source) -> int:
    """Return how many compressed bytes a gzip source still has to offer."""
    if not is_buffer(src):
        return os.stat(os.fspath(src)).st_size  # type: ignore[arg-type]
    if not can_seek(src):
        raise CodecError(
            f"'{source_name(src)}' is a compressed stream that cannot seek, "
            f"and this format needs the file's size before it can parse it. "
            f"Read the stream into io.BytesIO first, or pass a path."
        )
    here = int(src.tell())  # type: ignore[union-attr]
    end = int(src.seek(0, os.SEEK_END))  # type: ignore[union-attr]
    src.seek(here)  # type: ignore[union-attr]
    return end - here


def _gzip_trailer_size(src: Source) -> int:
    """Return the uncompressed size a gzip source records in its trailer.

    The trailer is the last four bytes of the source, so this is the member's
    own number only when the member runs to the end of what it sits in. A
    handle holding a member with data after it reports whatever those last
    four bytes happen to say, which is why the caller treats a value it cannot
    corroborate as a bound rather than as the size.
    """
    if not is_buffer(src):
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            fh.seek(-4, os.SEEK_END)
            trailer = fh.read(4)
        return int.from_bytes(trailer, "little")

    here = int(src.tell())  # type: ignore[union-attr]
    try:
        src.seek(-4, os.SEEK_END)  # type: ignore[union-attr]
        trailer = bytes(src.read(4))  # type: ignore[union-attr]
    finally:
        src.seek(here)  # type: ignore[union-attr]
    return int.from_bytes(trailer, "little")


def _decompressed_length(src: Source) -> int:
    """Count what a gzip source decompresses to, a chunk at a time.

    The trailer is cheaper and is what ``_gzip_size`` reaches for first; this
    is the fallback for the one case the trailer cannot answer, and it holds
    no more than a chunk of the file in memory while it counts.
    """
    total = 0
    with open_read(src) as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
    return total


def _gzip_size(src: Source, *, exact: bool) -> int:
    """Return what a gzip source decompresses to, from its ISIZE trailer.

    gzip records the uncompressed size modulo 2**32 in the last four bytes,
    which is the true size for every file under 4 GiB and a wrapped one above
    it. A value at least as large as the compressed input cannot have wrapped
    below it, so it is taken as exact; anything smaller is either a wrap or an
    incompressible file, and the two are indistinguishable from the trailer
    alone. A caller that only needs an upper bound - a ceiling on what a
    header may claim - gets the larger of the two numbers, which never
    understates and so never rejects a valid file. A caller that needs the
    real number gets the file counted.
    """
    compressed = _compressed_span(src)
    if compressed < 4:
        return compressed
    trailer = _gzip_trailer_size(src)
    if trailer >= compressed:
        return trailer
    if exact:
        return _decompressed_length(src)
    return compressed


def source_size(src: Source, *, exact: bool = False) -> int:
    """Return the size of a source in bytes.

    Parameters
    ----------
    src
        Path or open binary file object. A handle is measured by seeking to
        its end and back, so an unseekable stream cannot be measured.
    exact
        Set by a codec that divides the number rather than comparing against
        it. Only a gzip source can answer inexactly, and only when its size
        trailer is smaller than the compressed input it belongs to; asking
        for an exact size there costs one pass over the file.

    Returns
    -------
    int
        Size of the file for a path; for a handle, the number of bytes it has
        left to offer, counted from where it stands - the same bytes
        ``open_read`` would go on to read. A gzip source reports the size of
        what it decompresses to, which is the number a codec is checking a
        header against.

    Raises
    ------
    CodecError
        If the source is a file object that cannot seek.
    """
    if is_gzip(src):
        return _gzip_size(src, exact=exact)

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
        a file object is refused instead.

    Yields
    ------
    mmap.mmap or bytes
        The whole source. A mapping is closed on the way out.

    Raises
    ------
    LazyReadError
        If ``require_map`` is set and the source cannot be mapped.
    """
    if require_map or (not is_buffer(src) and not is_gzip(src)):
        mm = map_read(src, fmt=fmt)
        try:
            yield mm
        finally:
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
        Path, or a file object over a real file. A buffer with no file
        descriptor cannot be mapped.
    fmt
        Extension the error message should name, dot included.

    Returns
    -------
    mmap.mmap
        The whole file, mapped read-only.

    Raises
    ------
    LazyReadError
        If the source is a file object that no file descriptor backs, or
        whose descriptor is not a regular file.
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
