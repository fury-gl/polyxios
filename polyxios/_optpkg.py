"""Optional packages: import one if it is there, and say so clearly if not.

A codec that needs a package polyxios does not depend on - h5py for the
HDF5-backed formats - cannot ``import`` it at module level, since the codec
registry imports every codec module on start-up and a missing package would
drop the format from the registry without a word. It also cannot raise at
import time in a way the caller can act on: the caller asked to read a file,
not to import a module.

:func:`optional_package` is the one rule for that. It imports the package when
it can, and hands back a :class:`TripWire` in its place when it cannot: an
object that raises :class:`~polyxios.exceptions.MissingPackageError` the
moment anything is asked of it, naming the package, the reason the import
failed and the pip extra that installs it. A codec keeps the module-level
spelling it would have used anyway, and the error arrives at the call that
needed the package, from inside the read or write that needed it.

A minimum version can be asked for, and is read from the module's own
``__version__`` or, failing that, the distribution's metadata, so nothing
here depends on a version-parsing package either.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from types import ModuleType
from typing import Any

from polyxios.exceptions import MissingPackageError

# The numeric prefix of a version string: "3.10.0", "3.10.0rc1" and
# "3.10.0.post2" all compare as (3, 10, 0). A pre-release is not ordered below
# its release here, and does not need to be: a minimum is a floor, not a pin.
_VERSION_DIGITS: re.Pattern[str] = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


class TripWire:
    """A stand-in for a package that could not be imported.

    Every attribute access and every call raises
    :class:`~polyxios.exceptions.MissingPackageError` carrying the message it
    was built with, so a codec that reaches for ``h5py.File`` gets an error
    naming h5py rather than one naming a ``NoneType``. It is false in a
    boolean context, which is what lets ``if h5py:`` read as a check.

    Parameters
    ----------
    msg
        What to say when the package is used.

    Examples
    --------
    >>> h5py = TripWire("h5py is not installed")
    >>> bool(h5py)
    False
    >>> h5py.File  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    polyxios.exceptions.MissingPackageError: h5py is not installed
    """

    __slots__ = ("_msg",)

    def __init__(self, msg: str) -> None:
        object.__setattr__(self, "_msg", msg)

    def __getattr__(self, name: str) -> Any:
        raise MissingPackageError(self._msg)

    def __setattr__(self, name: str, value: object) -> None:
        raise MissingPackageError(self._msg)

    def __call__(self, *args: object, **kwargs: object) -> Any:
        raise MissingPackageError(self._msg)

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"TripWire({self._msg!r})"


def is_tripwire(obj: object) -> bool:
    """Say whether *obj* is the stand-in a missing package leaves behind.

    Parameters
    ----------
    obj
        Anything :func:`optional_package` might have handed back.

    Returns
    -------
    bool
        True for a :class:`TripWire`, False for a module or anything else.

    Examples
    --------
    >>> is_tripwire(TripWire("gone"))
    True
    >>> import os
    >>> is_tripwire(os)
    False
    """
    return isinstance(obj, TripWire)


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """Turn the numeric prefix of a version string into a comparable tuple."""
    match = _VERSION_DIGITS.match(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _installed_version(pkg: ModuleType, name: str) -> str | None:
    """Find a package's version: its own ``__version__`` first, then pip's record."""
    version = getattr(pkg, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_hint(name: str, extra: str | None) -> str:
    """Spell the pip command that brings the package in."""
    if extra:
        return f'pip install "polyxios[{extra}]"'
    return f"pip install {name}"


def optional_package(
    name: str,
    *,
    min_version: str | None = None,
    extra: str | None = None,
    trip_msg: str | None = None,
) -> tuple[ModuleType | TripWire, bool]:
    """Import an optional package, or a stand-in that says why it is missing.

    Parameters
    ----------
    name
        The importable name - ``"h5py"``, or a dotted submodule such as
        ``"scipy.spatial"``.
    min_version
        The lowest version that will do, as ``"3.8"`` or ``"3.8.0"``. A
        package older than this, or one whose version cannot be read at all,
        is treated as missing, and the message says which. Compared on the
        numeric prefix of the version string alone, so no version-parsing
        package is needed to ask.
    extra
        The name of the ``polyxios[...]`` extra that installs the package,
        so the message can spell the exact ``pip install`` command. Without
        one it names the bare package.
    trip_msg
        The whole message to raise instead of the one built here. For the
        rare caller with something better to say; the default names the
        package, what went wrong and how to install it.

    Returns
    -------
    tuple[types.ModuleType | TripWire, bool]
        The module and True when it imported, or a :class:`TripWire` and
        False when it did not. The bool is there so a codec can test once at
        module level rather than catching the error at every use.

    Examples
    --------
    >>> pkg, have_pkg = optional_package("os.path")
    >>> have_pkg, hasattr(pkg, "dirname")
    (True, True)
    >>> pkg, have_pkg = optional_package("not_a_package_anyone_has")
    >>> have_pkg
    False
    >>> pkg.anything()  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    polyxios.exceptions.MissingPackageError: ...

    Notes
    -----
    The error the import raised travels in the message: a package that is
    installed but broken - its compiled extension built against another
    numpy, say - is a different problem from one that is not there, and the
    caller is told which. Only :class:`ImportError` is caught; a package
    that raises anything else on import is not a missing package, and that
    error is let through.
    """
    hint = _install_hint(name, extra)
    try:
        pkg = importlib.import_module(name)
    except ImportError as exc:
        reason = trip_msg or (
            f"polyxios needs the package '{name}' for this, and"
            f" `import {name}` failed: {exc}. Install it with `{hint}`."
        )
        return TripWire(reason), False

    if min_version is None:
        return pkg, True

    wanted = _version_tuple(min_version)
    if wanted is None:
        raise ValueError(f"min_version {min_version!r} is not a version number.")
    current = _installed_version(pkg, name)
    found = None if current is None else _version_tuple(current)
    if found is not None and found >= wanted:
        return pkg, True

    if trip_msg is None:
        if current is None:
            trip_msg = (
                f"polyxios needs '{name}' {min_version} or later for this, and"
                f" the installed one carries no version it can read; the"
                f" installation may be incomplete. Reinstall it with `{hint}`."
            )
        else:
            trip_msg = (
                f"polyxios needs '{name}' {min_version} or later for this, and"
                f" `import {name}` found {current}. Upgrade it with `{hint}`."
            )
    return TripWire(trip_msg), False
