from __future__ import annotations

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata, supported_extensions
from polyxios._registry import (
    _SNIFF_BYTES,
    Codec,
    _make_dispatcher,
    build_default_registry,
    resolve,
)
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


def test_the_dispatcher_names_its_candidates() -> None:
    """The candidate list is readable, so no caller has to hard-code it."""
    registry = build_default_registry()
    assert registry[".dat"].candidates == ("tecplot", "nastran")
    # A codec that is one format competes with nobody and says so.
    assert registry[".tec"].candidates == ()


def test_a_dat_that_cannot_be_opened_raises_what_any_extension_would(
    tmp_path,
) -> None:
    """An unopenable file is not an ambiguous one; the OS error stands."""
    missing = tmp_path / "absent.dat"
    with pytest.raises(FileNotFoundError):
        polyxios.read(missing)
    # Same error the caller gets under an extension one codec owns outright.
    with pytest.raises(FileNotFoundError):
        polyxios.read(tmp_path / "absent.tec")


def test_a_raising_sniffer_is_reported_and_stepped_over(tmp_path, monkeypatch) -> None:
    """A broken sniffer warns and yields to the next candidate."""
    import polyxios.codecs._tecplot as tecplot

    def _boom(head: bytes) -> bool:
        raise RuntimeError("sniffer is broken")

    monkeypatch.setattr(tecplot, "sniff", _boom, raising=True)
    registry = build_default_registry()

    path = tmp_path / "mesh.dat"
    path.write_text(_NASTRAN_DAT)
    with pytest.warns(UserWarning, match="tecplot sniffer raised"):
        poly = polyxios.read(path, registry=registry)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_every_sniffer_raising_leaves_the_ambiguity_reported(
    tmp_path, monkeypatch
) -> None:
    """A broken sniffer must not turn the ambiguity into a crash."""
    import polyxios.codecs._nastran as nastran
    import polyxios.codecs._tecplot as tecplot

    def _boom(head: bytes) -> bool:
        raise RuntimeError("sniffer is broken")

    monkeypatch.setattr(nastran, "sniff", _boom, raising=True)
    monkeypatch.setattr(tecplot, "sniff", _boom, raising=True)
    registry = build_default_registry()

    path = tmp_path / "mesh.dat"
    path.write_text(_TECPLOT_DAT)
    with pytest.warns(UserWarning, match="sniffer raised"):
        with pytest.raises(UnsupportedFormatError, match="tecplot, nastran"):
            polyxios.read(path, registry=registry)


def test_a_raising_sniffer_is_named_in_the_ambiguity(tmp_path, monkeypatch) -> None:
    """A sniffer that raised never answered, so the error has to say so."""
    import polyxios.codecs._nastran as nastran

    def _boom(head: bytes) -> bool:
        raise RuntimeError("sniffer is broken")

    monkeypatch.setattr(nastran, "sniff", _boom, raising=True)
    registry = build_default_registry()

    path = tmp_path / "table.dat"
    path.write_text(_TABLE_DAT)
    with pytest.warns(UserWarning, match="sniffer raised"):
        with pytest.raises(
            UnsupportedFormatError,
            match="nastran .* raised instead of answering",
        ):
            polyxios.read(path, registry=registry)


def test_every_sniffer_raising_is_not_reported_as_a_verdict(
    tmp_path, monkeypatch
) -> None:
    """No sniffer answered, so the error must not claim the content matched none."""
    import polyxios.codecs._nastran as nastran
    import polyxios.codecs._tecplot as tecplot

    def _boom(head: bytes) -> bool:
        raise RuntimeError("sniffer is broken")

    monkeypatch.setattr(nastran, "sniff", _boom, raising=True)
    monkeypatch.setattr(tecplot, "sniff", _boom, raising=True)
    registry = build_default_registry()

    path = tmp_path / "mesh.dat"
    path.write_text(_TECPLOT_DAT)
    with pytest.warns(UserWarning, match="sniffer raised"):
        with pytest.raises(UnsupportedFormatError) as excinfo:
            polyxios.read(path, registry=registry)

    message = str(excinfo.value)
    assert "no sniffer could answer" in message
    assert "matches none of them" not in message


def test_a_lone_candidate_is_not_described_as_several(tmp_path) -> None:
    """One competitor is still a shared extension, but not a plural one."""
    registry = build_default_registry()
    dispatcher = _make_dispatcher(".dat", [("tecplot", registry[".tec"])])

    path = tmp_path / "table.dat"
    path.write_text(_TABLE_DAT)
    with pytest.raises(UnsupportedFormatError) as excinfo:
        dispatcher.read(path)
    assert "several formats" not in str(excinfo.value)
    assert "only tecplot reads here" in str(excinfo.value)

    with pytest.raises(UnsupportedFormatError, match="only tecplot reads here"):
        dispatcher.write(_tri_mesh(), path)


def _spying_dispatcher(seen: list[bytes]):
    """A one-candidate dispatcher recording exactly what its sniffer is shown."""

    def _spy(head: bytes) -> bool:
        seen.append(head)
        return False

    return _make_dispatcher(".dat", [("spy", Codec(None, None, _spy))])


def test_a_line_split_by_the_sniff_window_is_not_offered_to_a_sniffer(
    tmp_path,
) -> None:
    """Half a line answers a sniffer's question by accident; it is dropped."""
    seen: list[bytes] = []
    path = tmp_path / "long.dat"
    # A complete line, then one the window has to cut in half.
    path.write_bytes(b"x" * (_SNIFF_BYTES - 20) + b"\nCUT-IN-HALF-BY-THE-WINDOW")

    with pytest.raises(UnsupportedFormatError):
        _spying_dispatcher(seen).read(path)

    assert seen[0] == b"x" * (_SNIFF_BYTES - 20)


def test_a_sniff_window_holding_no_newline_is_left_whole(tmp_path) -> None:
    """There is nothing to fall back to, and the sniffers anchor at the start."""
    seen: list[bytes] = []
    path = tmp_path / "oneline.dat"
    path.write_bytes(b"y" * (_SNIFF_BYTES * 2))

    with pytest.raises(UnsupportedFormatError):
        _spying_dispatcher(seen).read(path)

    assert seen[0] == b"y" * _SNIFF_BYTES


def test_a_window_that_did_not_fill_keeps_its_last_line(tmp_path) -> None:
    """A short file ends on a real line, so nothing may be trimmed off it."""
    seen: list[bytes] = []
    path = tmp_path / "short.dat"
    path.write_bytes(b"first line\nGRID,1,,0.,0.,0.")

    with pytest.raises(UnsupportedFormatError):
        _spying_dispatcher(seen).read(path)

    assert seen[0] == b"first line\nGRID,1,,0.,0.,0."


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
    """A codec owning '.dat' outright keeps it, whatever else competes.

    The claim has to come from a codec that does not also list '.dat' among
    its sniffed extensions: listing an extension in both places is how a
    codec says it shares its own extension, which is the opposite claim.
    """
    import polyxios.codecs._off as off

    monkeypatch.setattr(off, "EXTENSIONS", (".off", ".dat"), raising=False)
    registry = build_default_registry()
    assert registry[".dat"] is registry[".off"]
    assert registry[".dat"].candidates == ()


def test_a_sniffed_extension_a_codec_does_not_own_keeps_no_writes(
    monkeypatch,
) -> None:
    """Competing to read '.dat' is no claim on writing it."""
    import polyxios.codecs._nastran as nastran

    monkeypatch.setattr(nastran, "SNIFF_DEFAULT_WRITER", True, raising=False)
    registry = build_default_registry()
    with pytest.raises(UnsupportedFormatError, match="fmt="):
        registry[".dat"].write(_tri_mesh(), "out.dat")


def test_two_owners_cannot_both_keep_the_writes_of_a_shared_extension(
    monkeypatch, tmp_path
) -> None:
    """Which one won would otherwise come down to the module walk order."""
    import polyxios.codecs._medit as medit

    monkeypatch.setattr(medit, "SNIFF_DEFAULT_WRITER", True, raising=False)
    with pytest.warns(UserWarning, match="default writer"):
        registry = build_default_registry()
    # Neither keeps it: a bare write raises and names the way out.
    with pytest.raises(UnsupportedFormatError, match="fmt="):
        registry[".mesh"].write(_tri_mesh(), tmp_path / "shared.mesh")


def test_an_entry_point_outranks_the_dispatcher_it_collides_with(monkeypatch) -> None:
    """Installing a codec for '.dat' is a claim; it beats polyxios' own guess."""
    import importlib.metadata

    claimed = Codec(read=lambda *a, **k: None, write=lambda *a, **k: None)

    class _EntryPoint:
        def load(self):
            return lambda: (".DAT", claimed)

    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda **kw: [_EntryPoint()]
    )
    registry = build_default_registry()
    # Folded to lower case like every other key, and holding the key outright.
    assert registry[".dat"] is claimed
    assert registry[".dat"].candidates == ()


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
