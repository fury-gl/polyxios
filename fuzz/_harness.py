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

import io
import sys
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


class InvalidMesh(AssertionError):
    """A reader returned a mesh that breaks the PolyData invariants."""


def check(poly: object, *, fmt: str) -> None:
    """Raise InvalidMesh unless ``poly`` is a structurally sound PolyData.

    Parameters
    ----------
    poly
        What the reader returned.
    fmt
        The format read, for the message.

    Raises
    ------
    InvalidMesh
        If ``validate`` refuses the mesh, or its offsets do not run from 0 to
        the connectivity length without stepping back.
    """
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


def read_one(data: bytes, *, fmt: str) -> None:
    """Read ``data`` as ``fmt``; return on a sound mesh or a polyxios error.

    Parameters
    ----------
    data
        The file's bytes.
    fmt
        Format to read them as.

    Raises
    ------
    InvalidMesh
        If the reader returned a mesh breaking the PolyData invariants.
    Exception
        Whatever the reader raised that is not a ``PolyxiosError``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            poly = polyxios.read(io.BytesIO(data), fmt=fmt)
        except PolyxiosError:
            return
    check(poly, fmt=fmt)


def main(*, name: str) -> None:
    """Run the fuzzing engine on the reader of target ``name``.

    Parameters
    ----------
    name
        A key of ``TARGETS``.

    Raises
    ------
    SystemExit
        If atheris is not installed.
    """
    if atheris is None:
        sys.exit('fuzzing needs atheris: pip install "atheris>=2.3"')
    fmt = TARGETS[name]

    def test_one_input(data: bytes) -> None:
        read_one(data, fmt=fmt)

    atheris.instrument_all()
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
