"""Read a file with trimesh: ``python read_trimesh.py PATH``."""

from __future__ import annotations

import sys

from _measure import measure
import trimesh


def load(path: str) -> tuple[object, int]:
    """Read ``path`` and return its vertices and face count.

    ``process=False`` keeps the vertices as the file lists them: trimesh
    otherwise merges duplicates and drops degenerate faces, which is work
    the other readers do not do.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    tuple
        The vertices and the number of faces; none for a point cloud.
    """
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    return loaded.vertices, len(getattr(loaded, "faces", ()))


if __name__ == "__main__":
    measure(load=load, path=sys.argv[1])
