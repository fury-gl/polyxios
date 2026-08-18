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

__all__ = [
    "Source",
    "is_buffer",
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
        The handle to read from. A handle the caller owns is yielded as it
        is and left open; a path is closed on the way out.

    Raises
    ------
    CodecError
        If the file object was opened in text mode.
    """
    if not is_buffer(src):
        with open(os.fspath(src), "rb") as fh:  # type: ignore[arg-type]
            yield fh
        return

    if _is_text_handle(src):
        raise CodecError(
            f"'{source_name(src)}' is open in text mode; mesh files are read "
            f"as bytes. Open it with mode='rb'."
        )
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
        The handle to write to. A handle the caller owns is yielded as it
        is and left open, so a caller can keep writing after polyxios is
        done; a path is closed on the way out.

    Raises
    ------
    CodecError
        If the file object was opened in text mode.
    """
    if not is_buffer(dst):
        with open(os.fspath(dst), "wb") as fh:  # type: ignore[arg-type]
            yield fh
        return

    if _is_text_handle(dst):
        raise CodecError(
            f"'{source_name(dst)}' is open in text mode; mesh files are "
            f"written as bytes. Open it with mode='wb'."
        )
    yield dst  # type: ignore[misc]


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

    Raises
    ------
    CodecError
        If the source is a file object that cannot seek.
    """
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

    if not is_buffer(src):
        with open(os.fspath(src), encoding=encoding, errors=errors) as fh:  # type: ignore[arg-type]
            yield fh
        return

    with open_read(src) as raw:
        wrapper = io.TextIOWrapper(raw, encoding=encoding, errors=errors)
        try:
            yield wrapper
        finally:
            # Closing the wrapper would close the caller's handle with it,
            # and the caller may still want to read past what we consumed.
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
    if require_map or not is_buffer(src):
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
