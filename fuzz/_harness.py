"""What every read fuzzer asks of a reader.

A reader handed arbitrary bytes has two honest answers: a mesh that holds
together, or a polyxios error naming what is wrong with the file. Anything
else - an ``IndexError`` from a short buffer, a ``struct.error``, a
``UnicodeDecodeError``, a ``RecursionError``, a hang, an allocation the file
never paid for, or a mesh whose connectivity points past its vertices - is a
finding. The harness turns the second kind of answer into an uncaught exception
so the fuzzing engine records the input that produced it.

``TARGETS`` maps a fuzzer name to the format it reads. Each ``fuzz_<name>.py``
beside this file is the entry point the engine runs (one binary per target, as
OSS-Fuzz builds them), and ``make_corpus.py`` writes their seed files.
"""

from __future__ import annotations

import dataclasses
import io
import itertools
import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np

import polyxios
from polyxios.exceptions import PolyxiosError, ValidationError
from polyxios.validate import validate

# The engine is needed only to fuzz; the oracle and the corpus run without it.
try:
    import atheris
except ImportError:
    atheris = None

# HDF5 and netCDF formats are left out: their bytes are parsed by libhdf5 and
# libnetcdf, which are fuzzed upstream, and polyxios only walks the tree those
# libraries hand back. Index formats (.pvd, .pvt*, .vtm) need a directory and
# are reached through the files they point at.
TARGETS: dict[str, str] = {
    "3mf": ".3mf",
    "amf": ".amf",
    "avs": ".avs",
    "bdf": ".bdf",
    "dae": ".dae",
    "dat": ".dat",
    "dolfin": ".xml",
    "dxf": ".dxf",
    "f3grid": ".f3grid",
    "fluent": ".fluent",
    "glb": ".glb",
    "gltf": ".gltf",
    "inp": ".inp",
    "las": ".las",
    "laz": ".laz",
    "mdpa": ".mdpa",
    "medit": ".medit",
    "meshb": ".meshb",
    "mfem": ".mesh",
    "msh": ".msh",
    "node": ".node",
    "obj": ".obj",
    "off": ".off",
    "pcd": ".pcd",
    "permas": ".post",
    "ply": ".ply",
    "pts": ".pts",
    "ptx": ".ptx",
    "splat": ".splat",
    "spz": ".spz",
    "stl": ".stl",
    "su2": ".su2",
    "tec": ".tec",
    "ugrid": ".ugrid",
    "unv": ".unv",
    "vol": ".vol",
    "vti": ".vti",
    "vtk": ".vtk",
    "vtp": ".vtp",
    "vtr": ".vtr",
    "vts": ".vts",
    "vtu": ".vtu",
    "wkt": ".wkt",
    "xyz": ".xyz",
}

# Readers that map a file rather than read it, fuzzed through a real file
# with ``lazy=True``: the arrays they hand back are views into the mapping,
# so every offset and count the file declares is checked against its size
# by nothing but the reader itself.
LAZY_TARGETS: dict[str, str] = {
    "lazy_las": ".las",
    "lazy_meshb": ".meshb",
    "lazy_pcd": ".pcd",
    "lazy_ply": ".ply",
    "lazy_splat": ".splat",
    "lazy_stl": ".stl",
    "lazy_vtk": ".vtk",
    "lazy_vtp": ".vtp",
    "lazy_vtu": ".vtu",
}


class InvalidMesh(AssertionError):
    """A reader returned a mesh that breaks the PolyData invariants."""


def check(poly: object, *, fmt: str, lazy: bool = False) -> None:
    """Raise InvalidMesh unless ``poly`` is a structurally sound PolyData.

    Parameters
    ----------
    poly
        What the reader returned.
    fmt
        The format read, for the message.
    lazy
        Whether the mesh was read lazily, keeping the file's dtypes.

    Raises
    ------
    InvalidMesh
        If ``validate`` refuses the mesh, or its offsets do not run from 0 to
        the connectivity length without stepping back.
    """
    if lazy:
        # A lazy mesh keeps the file's dtypes; validate() asks for the eager
        # ones, so the check runs on a copy, which also touches every page.
        poly = dataclasses.replace(
            poly,
            vertices=np.asarray(poly.vertices, dtype=np.float64),
            connectivity=np.asarray(poly.connectivity).astype(np.int64),
            offsets=np.asarray(poly.offsets).astype(np.int64),
            element_types=np.asarray(poly.element_types).astype(np.uint8),
        )
    try:
        validate(poly)
    except ValidationError as err:
        raise InvalidMesh(f"{fmt} reader returned an invalid mesh: {err}") from err
    offsets = np.asarray(poly.offsets)
    if offsets[0] != 0 or offsets[-1] != poly.connectivity.size:
        raise InvalidMesh(
            f"{fmt} reader returned offsets {offsets[0]}..{offsets[-1]} "
            f"over {poly.connectivity.size} connectivity entries"
        )
    if offsets.size > 1 and np.any(np.diff(offsets) < 0):
        raise InvalidMesh(f"{fmt} reader returned decreasing offsets")


_names = itertools.count()


def read_one(data: bytes, *, fmt: str, scratch: Path | None = None) -> None:
    """Read ``data`` as ``fmt``; return on a sound mesh or a polyxios error.

    Parameters
    ----------
    data
        The file's bytes.
    fmt
        Format to read them as.
    scratch
        A directory to write ``data`` into and read it back lazily from, the
        file removed once read; read from memory when None.

    Raises
    ------
    InvalidMesh
        If the reader returned a mesh breaking the PolyData invariants.
    Exception
        Whatever the reader raised that is not a ``PolyxiosError``.
    """
    if scratch is not None:
        # A fresh name each time: rewriting a file a live mapping still views
        # would fault on the pages past its new end. The pid keeps the forked
        # workers of ``-jobs`` apart in a shared directory.
        path = scratch / f"polyxios-fuzz-{os.getpid()}-{next(_names)}{fmt}"
        path.write_bytes(data)
        try:
            _read_and_check(path, fmt=fmt, lazy=True)
        finally:
            path.unlink()
        return
    _read_and_check(io.BytesIO(data), fmt=fmt, lazy=False)


def _read_and_check(source: object, *, fmt: str, lazy: bool) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            poly = polyxios.read(source, fmt=fmt, lazy=lazy)
        except PolyxiosError:
            return
    check(poly, fmt=fmt, lazy=lazy)


def main(*, name: str) -> None:
    """Run the fuzzing engine on the reader of target ``name``.

    Parameters
    ----------
    name
        A key of ``TARGETS`` or ``LAZY_TARGETS``.

    Raises
    ------
    SystemExit
        If atheris is not installed.
    """
    if atheris is None:
        sys.exit('fuzzing needs atheris: pip install "atheris>=2.3"')
    fmt = TARGETS.get(name) or LAZY_TARGETS[name]
    # The system's temporary directory itself, not one made for the run:
    # libFuzzer ends the process from C, so no cleanup would remove it.
    scratch = Path(tempfile.gettempdir()) if name in LAZY_TARGETS else None

    def test_one_input(data: bytes) -> None:
        read_one(data, fmt=fmt, scratch=scratch)

    atheris.instrument_all()
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
