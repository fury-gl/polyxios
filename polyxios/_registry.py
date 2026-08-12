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

    A format known by several extensions may also expose
    ``EXTENSIONS : tuple[str, ...]``, and every entry is registered against
    the same Codec. ``EXTENSION`` stays the canonical spelling — the one the
    docs quote and the one a writer should prefer — and is registered first
    whether or not ``EXTENSIONS`` repeats it, so a typo there cannot drop a
    format's own extension.

    Also loads third-party codecs declared under the ``polyxios.codecs``
    entry-point group; those stay one extension per entry point.

    Returns
    -------
    dict[str, Codec]
        Mapping from file extension (e.g. '.vtk') to Codec. Two extensions
        of one format map to the same Codec instance.
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

            if not (isinstance(ext, str) and callable(read_fn) and callable(write_fn)):
                continue

            # A bare string is iterable, so it would register one entry per
            # character; only a real sequence widens the registration.
            exts = getattr(mod, "EXTENSIONS", ())
            if not isinstance(exts, (tuple, list)) or not all(
                isinstance(e, str) for e in exts
            ):
                exts = ()

            codec = Codec(read_fn, write_fn)
            # EXTENSION leads, so a module whose EXTENSIONS forgets its own
            # canonical spelling still answers to it; dict.fromkeys drops the
            # repeat the usual case produces. resolve() looks a key up in
            # lower case, so a capitalised spelling has to be folded here or
            # it registers a key nothing can reach.
            for alias in dict.fromkeys(e.lower() for e in (ext, *exts)):
                registry[alias] = codec

    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="polyxios.codecs"):
            ext, codec = ep.load()()
            # Same folding as the built-in codecs: resolve() looks keys up in
            # lower case, and a key it cannot reach is worse than none.
            if isinstance(ext, str):
                registry[ext.lower()] = codec
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
        Explicit format override (e.g. '.vtk'). The leading dot and the case
        are both optional, so 'vtk' and 'VTK' resolve the same way.
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
    if fmt is None:
        ext = Path(path).suffix.lower()
    else:
        ext = fmt.strip().lower()
        ext = ext if ext.startswith(".") else f".{ext}"
    if ext not in registry:
        raise UnsupportedFormatError(f"No codec for '{ext}'")
    return registry[ext]
