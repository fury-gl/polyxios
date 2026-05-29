from polyxios._registry import Codec, build_default_registry, resolve
from polyxios._types import PolyData, make_polydata
import polyxios.transforms as transforms
from polyxios.validate import validate

_REGISTRY: dict[str, Codec] = build_default_registry()


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
        Format override (e.g. '.vtk'). Inferred from file extension if None.
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
    return codec.read(path, lazy=lazy)


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
        Format override (e.g. '.vtk'). Inferred from file extension if None.
    registry
        Custom codec registry. Uses the built-in registry if None.
    **opts
        Format-specific options passed to the codec's write function.
    """
    codec = resolve(path, fmt, registry or _REGISTRY)
    codec.write(poly, path, **opts)


__all__ = [
    "Codec",
    "PolyData",
    "make_polydata",
    "read",
    "transforms",
    "validate",
    "write",
]
