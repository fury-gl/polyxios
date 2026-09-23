"""HDF5 plumbing shared by the codecs whose file *is* an HDF5 file.

MED, CGNS, H5M and HMF each keep the whole mesh in one HDF5 file, so the four
share one way of getting h5py, one way of opening the file to read - through
the IO layer, so a buffer or a gzip-packed file reads like a path does - and
one way of writing it back. XDMF keeps its own plumbing: its HDF5 file is a
sidecar beside the XML one, found by name, which none of this covers.

Nothing here is a codec: the module exposes no ``EXTENSION`` and no ``read``,
so the registry walks past it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import functools
import io
import os
from pathlib import Path
import tempfile
from types import ModuleType
from typing import Any, TypeGuard
import warnings

import numpy as np

from polyxios._io import (
    GZIP_SUFFIXES,
    Source,
    is_gzip,
    read_bytes,
    source_name,
    source_suffix,
    write_bytes,
)
from polyxios._optpkg import TripWire, optional_package
from polyxios._tags import member_indices
from polyxios.exceptions import CodecError, LazyReadError, UnsupportedFormatError

# The eight bytes every HDF5 file opens with, superblock version regardless.
HDF5_MAGIC: bytes = b"\x89HDF\r\n\x1a\n"


@functools.cache
def _h5py() -> tuple[ModuleType | TripWire, bool]:
    """h5py and whether it is there, imported on first use.

    Importing h5py costs about a third of ``import polyxios``, and only the
    HDF5-backed formats need it; every other format should not pay for it.
    """
    return optional_package("h5py", min_version="3.0", extra="hdf5")


def require_h5py(*, fmt: str, name: str, verb: str) -> ModuleType:
    """Return h5py, or refuse with the install line that fixes its absence.

    Parameters
    ----------
    fmt
        The format's own extension, ``".med"`` and the like.
    name
        The file being read or written, for the message.
    verb
        ``"reading"`` or ``"writing"``, for the message.

    Returns
    -------
    module
        The imported h5py.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed, or is older than 3.0. Not an ImportError:
        a missing optional package is a format polyxios cannot do here, and
        is refused the way any other unsupported format is.
    """
    h5py, have_h5py = _h5py()
    if not have_h5py:
        raise UnsupportedFormatError(
            f"'{name}': {fmt} is an HDF5 file, and {verb} it needs h5py:"
            ' pip install "polyxios[hdf5]".'
        )
    return h5py


def is_hdf5(head: bytes) -> bool:
    """Say whether a file's opening bytes hold an HDF5 superblock.

    Parameters
    ----------
    head
        The first bytes of the file; the whole file, to be sure of one with
        a user block.

    Returns
    -------
    bool
        True when the file is HDF5, whatever it holds inside. The superblock
        sits at byte 0, or past a user block at 512, 1024, 2048 and every
        doubling after, which is where the library itself looks for it.
    """
    n = len(HDF5_MAGIC)
    offset = 0
    while offset + n <= len(head):
        if head[offset : offset + n] == HDF5_MAGIC:
            return True
        offset = offset * 2 if offset else 512
    return False


def refuse_lazy(lazy: bool, *, fmt: str, name: str) -> None:
    """Raise LazyReadError for a ``lazy=True`` read of an HDF5-backed format.

    Parameters
    ----------
    lazy
        The ``lazy`` argument the reader was handed.
    fmt
        The format's own extension, for the message.
    name
        The file being read, for the message.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set. The arrays live in HDF5 datasets that are
        decoded whole; a frozen PolyData has nothing to defer.
    """
    if lazy:
        raise LazyReadError(
            f"'{name}': {fmt} lazy reads are not supported; the arrays live"
            " in HDF5 datasets and are decoded whole."
        )


def warn_unknown_opts(opts: dict[str, Any], *, fmt: str, what: str) -> None:
    """Warn once about the options a reader or writer did not consume.

    Parameters
    ----------
    opts
        Whatever is left of ``**opts`` after the known keys were popped.
    fmt
        The format's own extension, for the message.
    what
        ``"read"`` or ``"write"``, for the message.
    """
    if opts:
        warnings.warn(
            f"{fmt} {what}: unrecognized options {set(opts)}; ignored.",
            UserWarning,
            stacklevel=3,
        )


def dataset_options(opts: dict[str, Any], *, fmt: str) -> dict[str, Any]:
    """Pop the h5py dataset options a writer forwards to ``create_dataset``.

    Parameters
    ----------
    opts
        The writer's ``**opts``; ``compression`` and ``compression_opts``
        are taken out of it.
    fmt
        The format's own extension, for the message.

    Returns
    -------
    dict
        Keyword arguments for ``create_dataset``: only the keys that were
        given, so a dataset too small to chunk is left contiguous.

    Raises
    ------
    CodecError
        If ``compression_opts`` names a level and ``compression`` names no
        method: h5py would raise a TypeError about a filter, naming neither
        the file nor the option at fault.
    """
    compression = opts.pop("compression", None)
    level = opts.pop("compression_opts", None)
    if level is not None and compression is None:
        raise CodecError(
            f"{fmt} write: compression_opts names a level and compression"
            " names no method; pass compression='gzip' (or another h5py filter)."
        )
    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if level is not None:
        kwargs["compression_opts"] = level
    return kwargs


def _plain_path(src: Source) -> TypeGuard[str | os.PathLike[str]]:
    """A path h5py can open itself: a real file name that is not a gzip target."""
    return isinstance(src, (str, os.PathLike)) and (
        source_suffix(src) not in GZIP_SUFFIXES
    )


@contextmanager
def open_hdf5_read(path: Source, *, fmt: str) -> Iterator[Any]:
    """Open an HDF5 file for reading, from a path, a buffer or a gzip member.

    Parameters
    ----------
    path
        Path or open binary file object. A plain path is handed to h5py as
        it is; anything else - a buffer, a gzip-packed file - is read whole
        through the IO layer and opened from memory, so the codec sees the
        same handle either way.
    fmt
        The format's own extension, for the messages.

    Yields
    ------
    h5py.File
        The open file, closed on exit.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If the file does not open with the HDF5 magic, or h5py cannot open it.
    """
    name = source_name(path)
    h5py = require_h5py(fmt=fmt, name=name, verb="reading")
    if _plain_path(path) and not is_gzip(path):
        with open(path, "rb") as fh:
            head = fh.read(len(HDF5_MAGIC))
        # A user block pushes the superblock down the file; the library's
        # own check walks those offsets on disk without reading it in.
        hdf5 = head == HDF5_MAGIC or bool(h5py.is_hdf5(os.fspath(path)))
        target: Any = path
    else:
        data = read_bytes(path)
        hdf5 = is_hdf5(data)
        target = io.BytesIO(data)
    if not hdf5:
        raise CodecError(f"'{name}': not an HDF5 file, so not {fmt}.")
    try:
        handle = h5py.File(target, "r")
    except OSError as exc:
        raise CodecError(f"'{name}': h5py could not open it ({exc}).") from None
    try:
        yield handle
    finally:
        handle.close()


@contextmanager
def open_hdf5_write(path: Source, *, fmt: str) -> Iterator[Any]:
    """Open an HDF5 file for writing, to a path, a buffer or a gzip target.

    Parameters
    ----------
    path
        Path or open binary file object. A plain path is written by h5py
        directly; anything else is built in memory and handed to the IO
        layer whole once the body has run through, which is what makes a
        buffer or a ``.gz`` name work without the codec knowing.
    fmt
        The format's own extension, for the messages.

    Yields
    ------
    h5py.File
        The open file, closed - and, for a buffer, delivered - on exit. A
        path is written under a temporary name beside it and moved into
        place on success, so a failed write leaves no half file; a path
        that is a symlink becomes a regular file, the link is not followed.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    """
    name = source_name(path)
    h5py = require_h5py(fmt=fmt, name=name, verb="writing")
    if _plain_path(path):
        # Written under a temporary name of its own and moved into place once
        # the body has run through, so a write that fails halfway leaves
        # neither a truncated file nor a stale one from an earlier write,
        # and two writers aimed at one path never share a half file.
        target = Path(path)
        fd, tmp = tempfile.mkstemp(
            dir=target.parent, prefix=target.name + ".", suffix=".partial"
        )
        os.close(fd)
        partial = Path(tmp)
        # mkstemp makes its file private; the finished mesh should carry
        # the permissions a plain open() would have given it.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(partial, 0o666 & ~umask)
        try:
            with h5py.File(partial, "w") as handle:
                yield handle
            os.replace(partial, target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        yield handle
    write_bytes(path, buffer.getvalue())


def as_array(node: Any, *, name: str, where: str) -> np.ndarray:
    """Read an HDF5 dataset whole, in the machine's byte order.

    Parameters
    ----------
    node
        The h5py object found at ``where``; a group is refused.
    name
        The file, for the message.
    where
        The dataset's path inside the file, for the message.

    Returns
    -------
    numpy.ndarray
        The dataset's values, native-endian so a swapped block never
        travels on in the mesh.

    Raises
    ------
    CodecError
        If ``node`` is a group rather than a dataset.
    """
    if not hasattr(node, "shape"):
        raise CodecError(f"'{name}': '{where}' is a group, not a dataset.")
    values = np.asarray(node[()])
    if values.dtype.kind in "biufc":
        return values.astype(values.dtype.newbyteorder("="), copy=False)
    return values


def child(group: Any, key: str, *, name: str, where: str) -> Any:
    """Return ``group[key]``, or refuse with the file and path named.

    Parameters
    ----------
    group
        An h5py group.
    key
        The child's name.
    name
        The file, for the message.
    where
        The group's path inside the file, for the message.

    Raises
    ------
    CodecError
        If the child is missing. h5py's own KeyError names the key and
        nothing else.
    """
    if key not in group:
        raise CodecError(f"'{name}': '{where}' holds no '{key}'.")
    return group[key]


def attr_text(node: Any, key: str) -> str | None:
    """Return a string attribute decoded and stripped, or None when absent.

    Parameters
    ----------
    node
        An h5py group or dataset.
    key
        The attribute's name.

    Returns
    -------
    str or None
        The text, NUL padding and surrounding blanks removed. An attribute
        holding something other than text - a number - answers None.
    """
    if key not in node.attrs:
        return None
    return text_of(node.attrs[key])


def text_of(value: Any) -> str | None:
    """Decode whatever h5py handed back for a string, or None when it is not one.

    Parameters
    ----------
    value
        Bytes, str, a numpy bytes scalar or a zero-dimensional array of one.

    Returns
    -------
    str or None
        The text with NUL padding and surrounding blanks removed.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value[()]
        else:
            return None
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip("\x00").strip()
    return None


def type_blocks(
    poly: Any, *, sizes: dict[int, int], fmt: str
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], set[int]]:
    """Split a mesh's elements into one fixed-width block per type.

    Parameters
    ----------
    poly
        The mesh being written.
    sizes
        Node count per polyxios type code, for the types the format can
        hold. A type not here is left out and reported through the return
        value; an element whose width is not its type's is left out with a
        warning naming the count, so the caller's "no such type" warning
        stays true.
    fmt
        The format's own extension, for the warning.

    Returns
    -------
    tuple
        The blocks as ``(code, index, cells)`` with the codes ascending,
        ``index`` the mesh indices of the block's elements in mesh order and
        ``cells`` their ``(n, k)`` int64 connectivity; and the set of codes
        the format has no name for. Every block is in mesh order, so a
        reader that walks the types by ascending code hands back a mesh
        whose elements are in the order they were written whenever the
        mesh was already grouped by type.
    """
    codes = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets)
    conn = np.asarray(poly.connectivity)
    widths = np.diff(offsets)
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []
    dropped: set[int] = set()
    malformed = 0
    for code in np.unique(codes):
        code = int(code)
        here = np.flatnonzero(codes == code)
        width = sizes.get(code)
        if width is None:
            dropped.add(code)
            continue
        fits = widths[here] == width
        if not fits.all():
            malformed += int(np.count_nonzero(~fits))
            here = here[fits]
            if not here.size:
                continue
        cells = conn[offsets[here][:, None] + np.arange(width)[None, :]]
        blocks.append((code, here, cells.astype(np.int64)))
    if malformed:
        warnings.warn(
            f"{fmt} write: {malformed} element(s) hold a node count other than"
            " their type's; those elements and the values and tags on them are"
            " not written.",
            UserWarning,
            stacklevel=3,
        )
    return blocks, dropped


def kept_index(blocks: list[tuple[int, np.ndarray, np.ndarray]]) -> np.ndarray:
    """The mesh indices of every element the blocks hold, in the order written.

    Parameters
    ----------
    blocks
        What :func:`type_blocks` returned.

    Returns
    -------
    numpy.ndarray
        One mesh index per written element, so the arrays and tags over the
        elements can be cut and reordered to match the file.
    """
    if not blocks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([index for _, index, _ in blocks])


def reindexed_tags(
    tags: dict[str, np.ndarray] | None, kept: np.ndarray, n_items: int
) -> dict[str, np.ndarray]:
    """Renumber tag groups onto the written order, dropping what was not written.

    Parameters
    ----------
    tags
        The mesh's tag groups.
    kept
        Mesh index per written entity, in the order written.
    n_items
        How many entities the mesh holds.

    Returns
    -------
    dict of str to numpy.ndarray
        The groups over written positions; a group left with no member is
        kept, empty, so its name survives the file.
    """
    position = np.full(n_items, -1, dtype=np.int64)
    position[kept] = np.arange(len(kept))
    out: dict[str, np.ndarray] = {}
    for name, members in (tags or {}).items():
        picked = position[member_indices(members, n_items)]
        out[name] = np.sort(picked[picked >= 0])
    return out
