"""Read a file with polyxios: ``python read_polyxios.py PATH [--lazy]``."""

from __future__ import annotations

import sys

from _measure import measure

import polyxios


def load(path: str, *, lazy: bool) -> tuple[object, int]:
    """Read ``path`` and return its vertices and element count.

    Parameters
    ----------
    path
        The file to read.
    lazy
        Map the file's arrays rather than copy them into memory.

    Returns
    -------
    tuple
        The vertices and the number of elements.
    """
    mesh = polyxios.read(path, lazy=True) if lazy else polyxios.read(path)
    return mesh.vertices, len(mesh.element_types)


if __name__ == "__main__":
    lazy = "--lazy" in sys.argv[2:]
    measure(load=lambda p: load(p, lazy=lazy), path=sys.argv[1])
