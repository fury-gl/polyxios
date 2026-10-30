"""The fuzz harness stays runnable without the fuzzing engine.

``fuzz/`` is not a package and atheris is not a test dependency, so nothing
else would notice a target losing its entry point, a new format joining the
registry without a fuzzer, a writer change leaving a target seedless, or the
oracle starting to reject what the writers themselves produce. Every seed is
read back through the same oracle the engine uses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import polyxios
from polyxios._types import PolyData

FUZZ = Path(__file__).parents[1] / "fuzz"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, FUZZ / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("_harness")
make_corpus = _load("make_corpus")

# Parsed by libhdf5 / libnetcdf, which are fuzzed upstream.
_LIBRARY_PARSED = frozenset(
    {".cgns", ".e", ".ex2", ".exo", ".h5m", ".hmf", ".med", ".vtkhdf", ".xdmf", ".xmf"}
)
# Read through a directory of the files they index, or never read at all.
_NOT_READ_FROM_BYTES = frozenset(
    {".pvd", ".pvti", ".pvtp", ".pvtr", ".pvts", ".pvtu", ".vtm", ".svg"}
)
# Extensions reaching a reader another target already fuzzes.
_ALIASES = {
    ".dato": ".post",
    ".fem": ".bdf",
    ".nas": ".bdf",
    ".ele": ".node",
    ".plt": ".tec",
}


def test_every_target_has_an_entry_point_and_back() -> None:
    entry_points = {p.stem.removeprefix("fuzz_") for p in FUZZ.glob("fuzz_*.py")}
    assert entry_points == set(harness.TARGETS) | set(harness.LAZY_TARGETS)


def test_every_format_read_from_bytes_is_fuzzed() -> None:
    fuzzed = set(harness.TARGETS.values())
    outstanding = {
        ext
        for ext in polyxios.supported_extensions()
        if ext not in fuzzed
        and ext not in _LIBRARY_PARSED
        and ext not in _NOT_READ_FROM_BYTES
        and _ALIASES.get(ext) not in fuzzed
    }
    assert not outstanding


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("corpus")
    make_corpus.main(out=out)
    return out


@pytest.mark.parametrize("name", sorted(harness.TARGETS))
def test_every_seed_passes_the_oracle(corpus, name: str) -> None:
    seeds = sorted((corpus / name).iterdir())
    if not seeds and harness.TARGETS[name] == ".laz":
        pytest.importorskip("lazrs")
    assert seeds, f"{name} has no seed"
    for seed in seeds:
        harness.read_one(seed.read_bytes(), fmt=harness.TARGETS[name])


@pytest.mark.parametrize("name", sorted(harness.LAZY_TARGETS))
def test_every_lazy_seed_passes_the_oracle(corpus, tmp_path, name: str) -> None:
    seeds = sorted((corpus / name).iterdir())
    assert seeds, f"{name} has no seed"
    for seed in seeds:
        harness.read_one(
            seed.read_bytes(), fmt=harness.LAZY_TARGETS[name], scratch=tmp_path
        )
    assert not any(tmp_path.iterdir())


def test_a_lazy_target_reads_through_a_mapping(tmp_path, monkeypatch) -> None:
    calls = []
    real = polyxios.read

    def spy(source, **kwargs):
        calls.append(kwargs.get("lazy"))
        return real(source, **kwargs)

    monkeypatch.setattr(harness.polyxios, "read", spy)
    data = (FUZZ / "make_corpus.py").read_bytes()
    harness.read_one(data, fmt=".stl", scratch=tmp_path)
    assert calls == [True]


def test_the_oracle_refuses_a_mesh_pointing_past_its_vertices() -> None:
    poly = PolyData(
        vertices=np.zeros((2, 3)),
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.raises(harness.InvalidMesh, match="index 2"):
        harness.check(poly, fmt=".test")


def test_the_oracle_refuses_offsets_short_of_the_connectivity() -> None:
    poly = PolyData(
        vertices=np.zeros((3, 3)),
        connectivity=np.array([0, 1, 2, 0], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([5], dtype=np.uint8),
    )
    with pytest.raises(harness.InvalidMesh, match="over 4"):
        harness.check(poly, fmt=".test")


def test_a_polyxios_error_is_an_honest_answer() -> None:
    harness.read_one(b"not a mesh", fmt=".stl")
