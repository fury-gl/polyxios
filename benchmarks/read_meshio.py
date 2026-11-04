"""Read a file with meshio: ``python read_meshio.py PATH``."""

from __future__ import annotations

import sys

from _measure import measure
import meshio


def load(path: str) -> tuple[object, int]:
    """Read ``path`` and return its points and cell count.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    tuple
        The points and the number of cells over every cell block.
    """
    mesh = meshio.read(path)
    return mesh.points, sum(len(block.data) for block in mesh.cells)


if __name__ == "__main__":
    measure(load=load, path=sys.argv[1])
