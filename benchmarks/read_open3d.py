"""Read a file with Open3D: ``python read_open3d.py PATH [--points]``."""

from __future__ import annotations

import sys

from _measure import measure
import numpy as np
import open3d


def load(path: str, *, points: bool) -> tuple[object, int]:
    """Read ``path`` and return its vertices and triangle count.

    Parameters
    ----------
    path
        The file to read.
    points
        Read it as a point cloud rather than a triangle mesh; Open3D has a
        reader for each.

    Returns
    -------
    tuple
        The vertices and the number of triangles; none for a point cloud.
    """
    if points:
        cloud = open3d.io.read_point_cloud(path)
        return np.asarray(cloud.points), 0
    mesh = open3d.io.read_triangle_mesh(path)
    return np.asarray(mesh.vertices), len(np.asarray(mesh.triangles))


if __name__ == "__main__":
    as_points = "--points" in sys.argv[2:]
    measure(load=lambda p: load(p, points=as_points), path=sys.argv[1])
