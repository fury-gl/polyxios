from __future__ import annotations

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata, supported_extensions
from polyxios._registry import Codec, build_default_registry, resolve
from polyxios.exceptions import UnsupportedFormatError


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])


def _stub_registry():
    return {".vtk": Codec(read=lambda *a, **k: None, write=lambda *a, **k: None)}


def test_every_registered_extension_starts_with_a_dot() -> None:
    assert all(ext.startswith(".") for ext in supported_extensions())


def test_nastran_family_shares_one_codec() -> None:
    registry = build_default_registry()
    assert registry[".nas"] is registry[".bdf"]
    assert registry[".fem"] is registry[".bdf"]


def test_dat_is_not_claimed() -> None:
    """'.dat' belongs to LS-DYNA and Tecplot too, so no codec binds it."""
    assert ".dat" not in build_default_registry()


@pytest.mark.parametrize("ext", [".bdf", ".nas", ".fem"])
def test_nastran_round_trips_under_every_extension(tmp_path, ext) -> None:
    poly = _tri_mesh()
    path = tmp_path / f"mesh{ext}"
    polyxios.write(poly, path)
    back = polyxios.read(path)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_dat_reads_through_an_explicit_format(tmp_path) -> None:
    poly = _tri_mesh()
    path = tmp_path / "mesh.dat"
    polyxios.write(poly, path, fmt=".bdf")
    with pytest.raises(UnsupportedFormatError, match=r"\.dat"):
        polyxios.read(path)
    np.testing.assert_array_equal(
        polyxios.read(path, fmt=".bdf").connectivity, poly.connectivity
    )


@pytest.mark.parametrize("fmt", [".vtk", "vtk", "VTK", ".VTK"])
def test_resolve_normalises_an_explicit_format(fmt) -> None:
    """A leading dot and the case of `fmt` are both optional."""
    registry = _stub_registry()
    assert resolve("mesh.bin", fmt, registry) is registry[".vtk"]


def test_resolve_falls_back_to_the_path_suffix() -> None:
    registry = _stub_registry()
    assert resolve("mesh.VTK", None, registry) is registry[".vtk"]


def test_resolve_rejects_an_unknown_format() -> None:
    with pytest.raises(UnsupportedFormatError, match="No codec for '.nope'"):
        resolve("mesh.nope", None, _stub_registry())


def test_resolve_reports_a_dotless_format_with_its_dot() -> None:
    with pytest.raises(UnsupportedFormatError, match="No codec for '.nope'"):
        resolve("mesh.bin", "nope", _stub_registry())


def test_a_single_extension_codec_still_registers() -> None:
    """EXTENSIONS is optional; EXTENSION alone remains the contract."""
    registry = build_default_registry()
    assert ".obj" in registry
    assert ".stl" in registry
