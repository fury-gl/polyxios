"""Read a file with laspy: ``python read_laspy.py PATH``."""

from __future__ import annotations

import sys

from _measure import measure
import laspy


def load(path: str) -> tuple[object, int]:
    """Read ``path`` and return its scaled coordinates.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    tuple
        The ``(n, 3)`` coordinates with scale and offset applied, as
        polyxios returns them, and no cells.
    """
    return laspy.read(path).xyz, 0


if __name__ == "__main__":
    measure(load=load, path=sys.argv[1])
