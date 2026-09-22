"""Optional packages: a module when it imports, a stand-in that says why not."""

from __future__ import annotations

import types

import pytest

from polyxios._optpkg import TripWire, optional_package
from polyxios.exceptions import MissingPackageError, UnsupportedFormatError


def test_a_present_package_is_handed_back_as_itself() -> None:
    pkg, have = optional_package("os.path")
    assert have is True
    assert isinstance(pkg, types.ModuleType)
    assert callable(pkg.dirname)


def test_a_missing_package_is_a_tripwire_that_names_the_fix() -> None:
    pkg, have = optional_package("polyxios_no_such_package", extra="hdf5")
    assert have is False
    assert isinstance(pkg, TripWire)
    assert not pkg
    with pytest.raises(
        MissingPackageError, match=r'pip install "polyxios\[hdf5\]"'
    ) as info:
        pkg.File("x")
    assert "polyxios_no_such_package" in str(info.value)
    assert "No module named" in str(info.value)


def test_the_error_is_an_unsupported_format_and_an_attribute_error() -> None:
    """A caller's ``except UnsupportedFormatError`` catches it, and ``hasattr``
    on the stand-in answers False rather than raising."""
    pkg, _ = optional_package("polyxios_no_such_package")
    with pytest.raises(UnsupportedFormatError):
        _ = pkg.anything
    with pytest.raises(AttributeError):
        _ = pkg.anything
    assert not hasattr(pkg, "anything")
    assert getattr(pkg, "anything", "fallback") == "fallback"


def test_calling_or_assigning_on_a_tripwire_raises_too() -> None:
    wire = TripWire("gone")
    with pytest.raises(MissingPackageError, match="gone"):
        wire()
    with pytest.raises(MissingPackageError, match="gone"):
        wire.x = 1
    assert repr(wire) == "TripWire('gone')"


def test_without_an_extra_the_bare_package_is_named() -> None:
    pkg, _ = optional_package("polyxios_no_such_package")
    with pytest.raises(
        MissingPackageError, match="pip install polyxios_no_such_package"
    ):
        _ = pkg.x


def test_a_custom_message_replaces_the_default() -> None:
    pkg, _ = optional_package("polyxios_no_such_package", trip_msg="use the other one")
    with pytest.raises(MissingPackageError, match="^use the other one$"):
        _ = pkg.x


def test_a_version_floor_is_honoured() -> None:
    pkg, have = optional_package("numpy", min_version="1.0")
    assert have is True and pkg.__name__ == "numpy"
    pkg, have = optional_package("numpy", min_version="999.0")
    assert have is False
    with pytest.raises(MissingPackageError, match="999.0 or later.*found") as info:
        _ = pkg.array
    assert "Upgrade it with `pip install numpy`" in str(info.value)


def test_a_version_floor_compares_as_numbers_not_as_tuples(monkeypatch) -> None:
    """``3.0`` is ``3.0.0``: a floor spelled longer than the installed
    version must not read as above it."""
    import numpy

    monkeypatch.setattr(numpy, "__version__", "3.0")
    _, have = optional_package("numpy", min_version="3.0.0")
    assert have is True
    _, have = optional_package("numpy", min_version="3")
    assert have is True
    _, have = optional_package("numpy", min_version="3.0.1")
    assert have is False
    monkeypatch.setattr(numpy, "__version__", "3")
    _, have = optional_package("numpy", min_version="3.0")
    assert have is True


def test_a_version_floor_that_is_not_a_version_is_refused() -> None:
    with pytest.raises(ValueError, match="not a version number"):
        optional_package("numpy", min_version="latest")


def test_a_package_with_no_readable_version_is_treated_as_missing(monkeypatch) -> None:
    import importlib.metadata

    fake = types.ModuleType("polyxios_fake_versionless")
    monkeypatch.setitem(__import__("sys").modules, "polyxios_fake_versionless", fake)

    def no_dist(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", no_dist)
    pkg, have = optional_package("polyxios_fake_versionless", min_version="1.0")
    assert have is False
    with pytest.raises(MissingPackageError, match="carries no version"):
        _ = pkg.x


def test_a_broken_package_carries_its_own_error(monkeypatch, tmp_path) -> None:
    """Installed but failing to import is a different problem from absent,
    and the message says which."""
    (tmp_path / "polyxios_broken_pkg.py").write_text(
        "raise ImportError('no libfoo.so')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg, have = optional_package("polyxios_broken_pkg")
    assert have is False
    with pytest.raises(MissingPackageError, match="no libfoo.so"):
        _ = pkg.x


def test_a_submodule_takes_its_version_from_the_top_level_package() -> None:
    pkg, have = optional_package("numpy.linalg", min_version="1.0")
    assert have is True and pkg.__name__ == "numpy.linalg"
    pkg, have = optional_package("numpy.linalg", min_version="999.0")
    assert have is False
    with pytest.raises(MissingPackageError, match="999.0 or later.*found"):
        _ = pkg.norm


def test_deleting_from_a_tripwire_raises_the_same_error() -> None:
    pkg = TripWire("gone")
    with pytest.raises(MissingPackageError, match="gone"):
        del pkg.anything
