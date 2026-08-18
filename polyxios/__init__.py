from polyxios import transforms
from polyxios._io import Source
from polyxios._registry import Codec, build_default_registry, resolve
from polyxios._types import PolyData, make_polydata
from polyxios.fetcher import fetch
from polyxios.helper import read_polydata, visualize_mesh
from polyxios.validate import validate
from polyxios.version import version as __version__

_REGISTRY: dict[str, Codec] = build_default_registry()


def supported_extensions(*, registry: dict | None = None) -> list[str]:
    """List the file extensions the registry resolves.

    Almost every one of them both reads and writes. Three kinds do not, and
    are listed all the same because resolving to a message that names the
    problem beats resolving to nothing:

    - an extension several formats share, such as ``.dat``, reads by content
      and needs ``fmt=`` to write;
    - ``.plt`` resolves only to the error saying binary Tecplot is not read;
    - the ParaView meta-file extensions read but raise on write.

    Parameters
    ----------
    registry
        Custom codec registry. Uses the built-in registry if None.

    Returns
    -------
    list of str
        Sorted extensions, each with its leading dot (e.g. '.obj').
    """
    return sorted(registry or _REGISTRY)


def read(
    path: Source,
    *,
    fmt: str | None = None,
    lazy: bool = False,
    registry: dict | None = None,
) -> PolyData:
    """Read a mesh file and return a PolyData.

    Parameters
    ----------
    path
        File path to read, or an open binary file object - anything with a
        ``read``, such as ``io.BytesIO`` or a socket's ``makefile('rb')``.
        A handle is read from wherever it stands and is never closed here.
    fmt
        Format override (e.g. '.vtk'). The leading dot and the case are
        both optional. Inferred from the file extension if None, which a
        file object only has when it carries a usable ``name``; pass ``fmt``
        for any other buffer.
    lazy
        If True and the format supports it, use mmap for the binary data
        section so pages are loaded on demand. A file object has to be
        backed by a real file for that; an in-memory buffer raises
        LazyReadError.
    registry
        Custom codec registry. Uses the built-in registry if None.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    UnsupportedFormatError
        If no codec matches, or if a nameless file object was passed
        without ``fmt``.
    """
    codec = resolve(path, fmt, registry or _REGISTRY)
    return codec.read(path=path, lazy=lazy)


def write(
    poly: PolyData,
    path: Source,
    *,
    fmt: str | None = None,
    registry: dict | None = None,
    **opts: object,
) -> None:
    """Write a PolyData to a mesh file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object - anything with a
        ``write``, such as ``io.BytesIO``. A handle is written at wherever
        it stands and is never closed here, so a caller can keep appending
        to it afterwards.
    fmt
        Format override (e.g. '.vtk'). The leading dot and the case are
        both optional. Inferred from the file extension if None, which a
        file object only has when it carries a usable ``name``; pass ``fmt``
        for any other buffer.
    registry
        Custom codec registry. Uses the built-in registry if None.
    **opts
        Format-specific options passed to the codec's write function.

    Raises
    ------
    UnsupportedFormatError
        If no codec matches, or if a nameless file object was passed
        without ``fmt``.
    """
    codec = resolve(path, fmt, registry or _REGISTRY)
    codec.write(poly=poly, path=path, **opts)


__all__ = [
    "Codec",
    "PolyData",
    "Source",
    "__version__",
    "fetch",
    "make_polydata",
    "read",
    "read_polydata",
    "supported_extensions",
    "transforms",
    "validate",
    "visualize_mesh",
    "write",
]
