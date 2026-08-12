from polyxios import transforms
from polyxios._registry import Codec, build_default_registry, resolve
from polyxios._types import PolyData, make_polydata
from polyxios.fetcher import fetch
from polyxios.helper import read_polydata, visualize_mesh
from polyxios.validate import validate
from polyxios.version import version as __version__

_REGISTRY: dict[str, Codec] = build_default_registry()


def supported_extensions(*, registry: dict | None = None) -> list[str]:
    """List the file extensions the registry can read and write.

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
    path: str,
    *,
    fmt: str | None = None,
    lazy: bool = False,
    registry: dict | None = None,
) -> PolyData:
    """Read a mesh file and return a PolyData.

    Parameters
    ----------
    path
        File path to read.
    fmt
        Format override (e.g. '.vtk'). The leading dot and the case are
        both optional. Inferred from the file extension if None.
    lazy
        If True and the format supports it, use mmap for the binary data
        section so pages are loaded on demand.
    registry
        Custom codec registry. Uses the built-in registry if None.

    Returns
    -------
    PolyData
        Parsed mesh data.
    """
    codec = resolve(path, fmt, registry or _REGISTRY)
    return codec.read(path=path, lazy=lazy)


def write(
    poly: PolyData,
    path: str,
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
        Output file path.
    fmt
        Format override (e.g. '.vtk'). The leading dot and the case are
        both optional. Inferred from the file extension if None.
    registry
        Custom codec registry. Uses the built-in registry if None.
    **opts
        Format-specific options passed to the codec's write function.
    """
    codec = resolve(path, fmt, registry or _REGISTRY)
    codec.write(poly=poly, path=path, **opts)


__all__ = [
    "Codec",
    "PolyData",
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
