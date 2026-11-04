"""Read a file with PyVista: ``python read_pyvista.py PATH``."""

from __future__ import annotations

import sys

from _measure import measure
import pyvista


def load(path: str) -> tuple[object, int]:
    """Read ``path`` and return its points and cell count.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    tuple
        The points and the number of cells.
    """
    mesh = pyvista.read(path)
    return mesh.points, mesh.n_cells


if __name__ == "__main__":
    measure(load=load, path=sys.argv[1])
