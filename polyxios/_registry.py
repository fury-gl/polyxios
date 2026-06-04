from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from polyxios.exceptions import UnsupportedFormatError


class Codec(NamedTuple):
    """A read+write pair for a mesh format."""

    read: Callable
    write: Callable


def build_default_registry() -> dict[str, Codec]:
    """Scan the codecs package and return a fresh registry dict.

    A module is registered when it exposes all three of:
      - ``read``  : callable
      - ``write`` : callable
      - ``EXTENSION`` : str  (e.g. ``".vtk"``)

    Also loads third-party codecs declared under the ``polyxios.codecs``
    entry-point group.

    Returns
    -------
    dict[str, Codec]
        Mapping from file extension (e.g. '.vtk') to Codec.
    """
    import importlib
    from pathlib import Path as _Path
    import pkgutil

    import polyxios.codecs as _codecs_pkg

    # meson-python editable installs expose __file__ but leave __path__ empty;
    # fall back to the directory of __file__ to locate codec modules on disk.
    _search_path = list(_codecs_pkg.__path__) or [
        str(_Path(_codecs_pkg.__file__).parent)
    ]

    registry: dict[str, Codec] = {}

    for mod_info in pkgutil.iter_modules(_search_path):
        if mod_info.name.startswith("_") and not mod_info.name.startswith("__"):
            try:
                mod = importlib.import_module(f"polyxios.codecs.{mod_info.name}")
            except Exception:
                continue

            ext = getattr(mod, "EXTENSION", None)
            read_fn = getattr(mod, "read", None)
            write_fn = getattr(mod, "write", None)

            if isinstance(ext, str) and callable(read_fn) and callable(write_fn):
                registry[ext] = Codec(read_fn, write_fn)

    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="polyxios.codecs"):
            ext, codec = ep.load()()
            registry[ext] = codec
    except Exception:
        pass

    return registry


def resolve(
    path: Path | str,
    fmt: str | None,
    registry: dict[str, Codec],
) -> Codec:
    """Resolve a file path or explicit format string to a Codec.

    Parameters
    ----------
    path
        File path (used to infer extension if fmt is None).
    fmt
        Explicit format override (e.g. '.vtk').
    registry
        Codec registry to search.

    Returns
    -------
    Codec
        Matching codec.

    Raises
    ------
    UnsupportedFormatError
        If no codec is registered for the resolved extension.
    """
    ext = fmt if fmt is not None else Path(path).suffix.lower()
    if ext not in registry:
        raise UnsupportedFormatError(f"No codec for '{ext}'")
    return registry[ext]
