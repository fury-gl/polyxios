"""Path-or-buffer input and output shared by every codec.

A codec never opens a file itself. It asks this module for the bytes or the
text behind whatever the caller passed - a path, or an already-open file
object - so that widening the public API to buffers costs each codec one
call site instead of one rewrite.

Two rules hold everywhere:

- a handle the caller owns is never closed here, and is left wherever the
  codec's reading or writing leaves it;
- a path is opened and closed inside the helper that was given it.
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

from polyxios.exceptions import CodecError, LazyReadError

# What read() and write() accept in place of a path. A file object is only
# ever duck-typed - anything with the read/write methods a codec reaches for
# qualifies - so the alias documents the intent rather than gating on a type.
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
        The opening bytes, or an empty result when the source is a stream
        that can neither peek nor seek back - nothing has been consumed
        either way.
    """
    if not is_buffer(src):
        try:
            with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
                return bytes(fh.read(n))
        except OSError:
            # Not this function's error to report: the codec is about to
            # open the same path and will raise the real one.
            return b""

    peek = getattr(src, "peek", None)
    if callable(peek):
        try:
            return bytes(peek(n))[:n]
        except Exception:
            return b""

    if not getattr(src, "seekable", lambda: False)():
        return b""
    here = src.tell()  # type: ignore[union-attr]
    try:
        return bytes(src.read(n))  # type: ignore[union-attr]
    finally:
        src.seek(here)  # type: ignore[union-attr]


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
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            if is_gzip(src):
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

    if is_gzip(src):
        # Closing the wrapper flushes it without closing what it wraps, so
        # the caller's handle survives the decompression.
        with gzip.GzipFile(fileobj=src, mode="rb") as gz:  # type: ignore[arg-type]
            yield gz  # type: ignore[misc]
        return

    yield src  # type: ignore[misc]


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


def _gzip_size(src: Source) -> int:
    """Return what a gzip source decompresses to, from its ISIZE trailer.

    gzip records the uncompressed size modulo 2**32 in the last four bytes.
    Past 4 GiB that wraps, and a wrapped value would understate the file - so
    the compressed size wins whenever it is the larger of the two. The number
    is only ever used as an upper bound on what a header may claim, where
    overstating is harmless and understating rejects a valid file.
    """
    if not is_buffer(src):
        compressed = os.stat(os.fspath(src)).st_size  # type: ignore[arg-type]
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            if compressed < 4:
                return compressed
            fh.seek(-4, os.SEEK_END)
            trailer = fh.read(4)
        return max(int.from_bytes(trailer, "little"), compressed)

    if not getattr(src, "seekable", lambda: False)():
        raise CodecError(
            f"'{source_name(src)}' is a compressed stream that cannot seek, "
            f"and this format needs the file's size before it can parse it. "
            f"Read the stream into io.BytesIO first, or pass a path."
        )
    here = src.tell()  # type: ignore[union-attr]
    try:
        compressed = int(src.seek(0, os.SEEK_END))  # type: ignore[union-attr]
        if compressed < 4:
            return compressed
        src.seek(-4, os.SEEK_END)  # type: ignore[union-attr]
        trailer = bytes(src.read(4))  # type: ignore[union-attr]
    finally:
        src.seek(here)  # type: ignore[union-attr]
    return max(int.from_bytes(trailer, "little"), compressed)


def source_size(src: Source) -> int:
    """Return the total size of a source in bytes.

    Parameters
    ----------
    src
        Path or open binary file object. A handle is measured by seeking to
        its end and back, so an unseekable stream cannot be measured.

    Returns
    -------
    int
        Size of the file for a path; for a handle, the number of bytes it
        holds in total, counted from its start and not from where it stands.
        A gzip source reports the size of what it decompresses to, which is
        the number a codec is checking a header against.

    Raises
    ------
    CodecError
        If the source is a file object that cannot seek.
    """
    if is_gzip(src):
        return _gzip_size(src)

    if not is_buffer(src):
        return os.stat(os.fspath(src)).st_size  # type: ignore[arg-type]

    if not getattr(src, "seekable", lambda: False)():
        raise CodecError(
            f"'{source_name(src)}' is a stream that cannot seek, and this "
            f"format needs the file's size before it can parse it. Read the "
            f"stream into io.BytesIO first, or pass a path."
        )
    here = src.tell()  # type: ignore[union-attr]
    size = src.seek(0, os.SEEK_END)  # type: ignore[union-attr]
    src.seek(here)  # type: ignore[union-attr]
    return int(size)


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
        wrapper = io.TextIOWrapper(raw, encoding=encoding, errors=errors)
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
