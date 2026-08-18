from __future__ import annotations

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata, supported_extensions
from polyxios._registry import Codec, build_default_registry, resolve
from polyxios.exceptions import CodecError, UnsupportedFormatError


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


_TECPLOT_DAT = """TITLE = "one triangle"
VARIABLES = "X" "Y" "Z"
ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1 2 3
"""

_NASTRAN_DAT = """$ a banner a sniffer has to step over
$ and a second line of it
GRID,1,,0.,0.,0.
GRID,2,,1.,0.,0.
GRID,3,,0.,1.,0.
CTRIA3,1,1,1,2,3
"""

# Neither a Tecplot header nor a bulk data card: a plain ASCII table, which is
# the third thing '.dat' is used for.
_TABLE_DAT = """# x y z value
0.0 0.0 0.0 1.0
1.0 0.0 0.0 2.0
"""


def test_dat_is_shared_not_owned() -> None:
    """'.dat' resolves, but through a dispatcher no one codec owns."""
    registry = build_default_registry()
    assert ".dat" in registry
    assert registry[".dat"] is not registry[".bdf"]
    assert registry[".dat"] is not registry[".tec"]


def test_dat_reads_as_tecplot_when_it_looks_like_tecplot(tmp_path) -> None:
    path = tmp_path / "mesh.dat"
    path.write_text(_TECPLOT_DAT)
    poly = polyxios.read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_dat_reads_as_nastran_when_it_looks_like_nastran(tmp_path) -> None:
    path = tmp_path / "mesh.dat"
    path.write_text(_NASTRAN_DAT)
    poly = polyxios.read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_dat_matching_no_codec_names_the_candidates(tmp_path) -> None:
    """The ambiguity is reported, never guessed at."""
    path = tmp_path / "table.dat"
    path.write_text(_TABLE_DAT)
    with pytest.raises(UnsupportedFormatError, match="tecplot, nastran"):
        polyxios.read(path)


def test_dat_write_refuses_to_guess(tmp_path) -> None:
    """An output file has no content to sniff, so fmt= is required."""
    with pytest.raises(UnsupportedFormatError, match="fmt="):
        polyxios.write(_tri_mesh(), tmp_path / "mesh.dat")


def test_a_narrow_sniffer_is_tried_before_a_broad_one() -> None:
    """SNIFF_PRIORITY orders the attempts; Tecplot's test is the narrow one."""
    import polyxios.codecs._nastran as nastran
    import polyxios.codecs._tecplot as tecplot

    assert tecplot.SNIFF_PRIORITY < nastran.SNIFF_PRIORITY


def test_a_malformed_sniff_extensions_contests_nothing(monkeypatch) -> None:
    """Same shape guard as EXTENSIONS: a bare string is not a sequence."""
    import polyxios.codecs._nastran as nastran
    import polyxios.codecs._tecplot as tecplot

    monkeypatch.setattr(nastran, "SNIFF_EXTENSIONS", ".dat", raising=True)
    monkeypatch.setattr(tecplot, "SNIFF_EXTENSIONS", ".dat", raising=True)
    registry = build_default_registry()
    assert ".dat" not in registry
    assert "d" not in registry


def test_a_codec_without_a_sniffer_cannot_contest(monkeypatch) -> None:
    """Declaring a contested extension without a test claims nothing."""
    import polyxios.codecs._nastran as nastran
    import polyxios.codecs._tecplot as tecplot

    monkeypatch.delattr(nastran, "sniff", raising=True)
    monkeypatch.delattr(tecplot, "sniff", raising=True)
    assert ".dat" not in build_default_registry()


def test_an_owned_extension_is_never_contested(monkeypatch) -> None:
    """A codec owning '.dat' outright keeps it, whatever else competes."""
    import polyxios.codecs._nastran as nastran

    monkeypatch.setattr(nastran, "EXTENSIONS", (".bdf", ".dat"), raising=True)
    registry = build_default_registry()
    assert registry[".dat"] is registry[".bdf"]


def test_plt_resolves_to_a_clear_binary_tecplot_error(tmp_path) -> None:
    """'.plt' is registered so the error names the format, not the extension."""
    assert ".plt" in build_default_registry()
    path = tmp_path / "mesh.plt"
    path.write_bytes(b"#!TDV112" + b"\x00" * 32)
    with pytest.raises(CodecError, match="binary Tecplot"):
        polyxios.read(path)


@pytest.mark.parametrize("ext", [".bdf", ".nas", ".fem"])
def test_nastran_round_trips_under_every_extension(tmp_path, ext) -> None:
    poly = _tri_mesh()
    path = tmp_path / f"mesh{ext}"
    polyxios.write(poly, path)
    back = polyxios.read(path)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_dat_reads_through_an_explicit_format(tmp_path) -> None:
    """fmt= still forces the issue, and skips the sniffing entirely."""
    poly = _tri_mesh()
    path = tmp_path / "mesh.dat"
    polyxios.write(poly, path, fmt=".bdf")
    np.testing.assert_array_equal(
        polyxios.read(path, fmt=".bdf").connectivity, poly.connectivity
    )


def test_a_deck_polyxios_wrote_to_dat_reads_back_unaided(tmp_path) -> None:
    """The writer's own output has to survive the sniffer."""
    poly = _tri_mesh()
    path = tmp_path / "mesh.dat"
    polyxios.write(poly, path, fmt=".bdf")
    np.testing.assert_array_equal(polyxios.read(path).connectivity, poly.connectivity)


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


def test_extension_registers_even_when_extensions_omits_it(monkeypatch) -> None:
    """EXTENSION leads, so a typo in EXTENSIONS cannot drop the canonical one."""
    import polyxios.codecs._nastran as nastran

    monkeypatch.setattr(nastran, "EXTENSIONS", (".nas", ".fem"), raising=True)
    registry = build_default_registry()
    assert registry[".bdf"] is registry[".nas"] is registry[".fem"]


def test_a_malformed_extensions_falls_back_to_extension(monkeypatch) -> None:
    import polyxios.codecs._nastran as nastran

    monkeypatch.setattr(nastran, "EXTENSIONS", ".bdf", raising=True)
    registry = build_default_registry()
    assert ".bdf" in registry
    assert "b" not in registry


def test_a_capitalised_extension_registers_in_lower_case(monkeypatch) -> None:
    """resolve() looks keys up folded, so an unfolded key is unreachable."""
    import polyxios.codecs._nastran as nastran

    monkeypatch.setattr(nastran, "EXTENSIONS", (".BDF", ".Nas"), raising=True)
    registry = build_default_registry()
    assert ".BDF" not in registry
    assert registry[".bdf"] is registry[".nas"]
