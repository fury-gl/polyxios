from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
import warnings

from polyxios.exceptions import UnsupportedFormatError


class Codec(NamedTuple):
    """A read+write pair for a mesh format.

    Attributes
    ----------
    read
        Reader callable, invoked as ``read(path=..., lazy=...)``.
    write
        Writer callable, invoked as ``write(poly=..., path=..., **opts)``.
    sniff
        Optional content test, ``sniff(head: bytes) -> bool``, answering
        whether the opening bytes of a file look like this format. Only a
        codec competing for an extension another format also uses needs one.
    candidates
        Empty for a codec that is one format. Populated only on the
        dispatcher a contested extension resolves to, naming the formats
        competing for it in the order their ``sniff`` is tried, so a caller
        can report the ambiguity without hard-coding the candidate list.
    """

    read: Callable
    write: Callable
    sniff: Callable | None = None
    candidates: tuple[str, ...] = ()


# How much of a file's opening the sniffers see. Large enough that a Nastran
# deck's comment banner does not hide its first real card, small enough that
# reading it costs one page.
_SNIFF_BYTES: int = 8192


def _make_dispatcher(ext: str, entries: list[tuple[str, Codec]]) -> Codec:
    """Build the Codec that resolves one contested extension by content.

    Parameters
    ----------
    ext
        The contested extension, lower case and with its leading dot.
    entries
        Candidate ``(label, codec)`` pairs, already in the order their
        ``sniff`` should be tried.

    Returns
    -------
    Codec
        A codec whose ``read`` delegates to the first candidate that
        recognises the file, whose ``write`` always raises - an output file
        has no content to sniff yet - and whose ``candidates`` names the
        competitors.
    """
    labels = tuple(label for label, _ in entries)
    named = ", ".join(labels)
    # One competitor is still a contested extension - the other formats using
    # it simply have no codec here - so the wording has to stop claiming a
    # plurality the list does not show.
    if len(labels) > 1:
        shared = f"'{ext}' is used by several formats ({named})"
        no_match = "the file's content matches none of them"
    else:
        shared = f"'{ext}' is shared between formats, of which only {named} reads here"
        no_match = f"the file's content does not look like {named}"

    def read(path: Path | str, *, lazy: bool = False) -> object:
        # A file that cannot be opened is not an ambiguous file: let the OS
        # error through, so a missing or unreadable path raises the same
        # thing here as it does under an extension one codec owns.
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)

        # A full buffer means the read stopped mid-line, and half a line
        # answers a sniffer's question by accident either way; drop it. A
        # window holding no newline at all is left alone - there is nothing
        # to drop back to, and the sniffers anchor at the start regardless.
        if len(head) == _SNIFF_BYTES and b"\n" in head:
            head = head.rsplit(b"\n", 1)[0]

        broken: list[str] = []
        for label, codec in entries:
            try:
                matched = bool(codec.sniff(head))
            except Exception as exc:  # a broken sniffer must not mask the rest
                broken.append(f"{label} ({exc})")
                warnings.warn(
                    f"{label} sniffer raised on '{Path(path).name}': {exc}",
                    stacklevel=2,
                )
                continue
            if matched:
                return codec.read(path=path, lazy=lazy)

        # A sniffer that raised never answered, so reporting that the content
        # matched nothing would state a verdict no one reached; say what
        # actually happened, and name the failures either way.
        if len(broken) == len(entries):
            detail = f"no sniffer could answer, each one raising: {'; '.join(broken)}"
        else:
            detail = no_match

        parts = [f"'{Path(path).name}': {shared} and {detail}."]
        if broken and len(broken) != len(entries):
            noun = "sniffer" if len(broken) == 1 else "sniffers"
            parts.append(f"The {noun} {'; '.join(broken)} raised instead of answering.")
        parts.append("Pass fmt= to choose one explicitly.")

        raise UnsupportedFormatError(" ".join(parts))

    def write(poly: object, path: Path | str, **opts: object) -> None:
        raise UnsupportedFormatError(
            f"{shared}, so a writer cannot be chosen from the extension "
            f"alone. Pass fmt= to choose one explicitly."
        )

    return Codec(read, write, None, labels)


def build_default_registry() -> dict[str, Codec]:
    """Scan the codecs package and return a fresh registry dict.

    A module is registered when it exposes all three of:
      - ``read``  : callable
      - ``write`` : callable
      - ``EXTENSION`` : str  (e.g. ``".vtk"``)

    A format known by several extensions may also expose
    ``EXTENSIONS : tuple[str, ...]``, and every entry is registered against
    the same Codec. ``EXTENSION`` stays the canonical spelling - the one the
    docs quote and the one a writer should prefer - and is registered first
    whether or not ``EXTENSIONS`` repeats it, so a typo there cannot drop a
    format's own extension.

    An extension several unrelated formats use - ``.dat`` belongs to Tecplot,
    Nastran, LS-DYNA and plain ASCII tables alike - is claimed by none of them
    outright. A codec competing for one declares it under
    ``SNIFF_EXTENSIONS : tuple[str, ...]`` and supplies
    ``sniff(head: bytes) -> bool``; the extension then resolves to a
    dispatcher that reads the file's opening bytes and delegates to the first
    codec recognising them. ``SNIFF_PRIORITY : int`` orders the attempts, the
    lowest first, ties broken by module name; a codec whose test is narrow
    should sort ahead of one whose test is broad. An extension a codec owns
    outright through ``EXTENSION``/``EXTENSIONS`` is never contested.

    Also loads third-party codecs declared under the ``polyxios.codecs``
    entry-point group; those stay one extension per entry point, and they are
    registered last. A built-in codec owning an extension keeps it against a
    built-in sniffer competing for it, but an entry point naming the same
    extension replaces whatever holds it, dispatcher included: installing a
    codec for ``.dat`` is a deliberate claim by the person installing it, and
    it should win over polyxios' own guess at the file's content.

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
    # Contested extension -> the codecs competing for it, each carrying the
    # sort key that decides which sniffer runs first.
    contested: dict[str, list[tuple[int, str, Codec]]] = {}

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

            sniff_fn = getattr(mod, "sniff", None)
            if not callable(sniff_fn):
                sniff_fn = None

            codec = Codec(read_fn, write_fn, sniff_fn)
            # EXTENSION leads, so a module whose EXTENSIONS forgets its own
            # canonical spelling still answers to it; dict.fromkeys drops the
            # repeat the usual case produces. resolve() looks a key up in
            # lower case, so a capitalised spelling has to be folded here or
            # it registers a key nothing can reach.
            for alias in dict.fromkeys(e.lower() for e in (ext, *exts)):
                registry[alias] = codec

            # Same shape guard as EXTENSIONS: a bare string is iterable, and
            # a codec declaring one without a sniffer has nothing to compete
            # with, so it is dropped rather than allowed to claim the key.
            sniff_exts = getattr(mod, "SNIFF_EXTENSIONS", ())
            if (
                sniff_fn is None
                or not isinstance(sniff_exts, (tuple, list))
                or not all(isinstance(e, str) for e in sniff_exts)
            ):
                sniff_exts = ()

            priority = getattr(mod, "SNIFF_PRIORITY", 0)
            if not isinstance(priority, int) or isinstance(priority, bool):
                priority = 0

            for alias in dict.fromkeys(e.lower() for e in sniff_exts):
                contested.setdefault(alias, []).append((priority, mod_info.name, codec))

    for alias, competing in contested.items():
        # An extension one codec owns outright is not contested, whatever
        # another codec believes it competes for: the owner keeps the key.
        if alias in registry:
            continue
        competing.sort(key=lambda entry: (entry[0], entry[1]))
        registry[alias] = _make_dispatcher(
            alias, [(name.lstrip("_"), codec) for _, name, codec in competing]
        )

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
