"""Fetch and visualize a VTK legacy dataset (UNSTRUCTURED_GRID or POLYDATA) with FURY."""

import argparse
import sys

from fury import actor, window
import numpy as np

import polyxios
from polyxios._element_types import ELEMENT_TYPES
from polyxios.fetcher import fetch, fetch_by_extension
import polyxios.transforms as transforms

_LINE_CODES = frozenset({ELEMENT_TYPES["line"], ELEMENT_TYPES["poly_line"]})


def _extract_lines(poly):
    """Extract line segments from line/poly_line elements.

    Parameters
    ----------
    poly : PolyData

    Returns
    -------
    list of numpy.ndarray or None
        List of (n_pts, 3) float64 arrays, one per connected line, or None.
    """
    segments = []
    for i in range(len(poly.element_types)):
        if int(poly.element_types[i]) not in _LINE_CODES:
            continue
        idx = poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]]
        segments.append(poly.vertices[idx].astype(np.float64))
    return segments if segments else None


def _build_actors(*, poly, render_lines=False):
    if len(poly.vertices) == 0:
        return None
    faces = poly.faces
    if faces is None:
        surface = transforms.extract_surface(poly)
        faces = surface.faces
    if faces is not None and len(faces) > 0:
        colors = transforms.vertex_colors(poly)
        return [
            actor.surface(
                poly.vertices,
                faces,
                colors=colors if colors is not None else (0.8, 0.7, 0.6),
            )
        ]
    if render_lines:
        lines = _extract_lines(poly)
        if lines:
            print(f"  Rendering {len(lines)} line segment(s) with actor.line.")
            return [actor.line(lines, colors=(0.2, 0.8, 0.2))]
    print("  No renderable geometry — rendering as point cloud.")
    return [actor.point(poly.vertices, colors=(0.9, 0.9, 0.9))]


def visualize(*, path, render_lines=False):
    """Fetch, read, and display a single VTK file.

    Parameters
    ----------
    path : str
        Local path to a VTK legacy file.
    render_lines : bool
        If True, render line/poly_line elements with actor.line instead of
        falling back to a point cloud.
    """
    print(f"Reading {path} ...")
    poly = polyxios.read(path)
    print(
        f"  {len(poly.vertices)} vertices | "
        f"{len(poly.element_types)} elements | "
        f"vertex attrs: {list(poly.vertex_attrs) or 'none'}"
    )
    actors = _build_actors(poly=poly, render_lines=render_lines)
    if actors is None:
        print("  No geometry (FIELD data) — skipping window.")
        return
    window.show(actors)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a VTK dataset via polyxios + FURY."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="VTK filename to fetch and visualize (e.g. 'mesh.vtk'). "
        "Omit to render the first available file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List locally cached VTK files and exit.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Render line/poly_line elements with actor.line instead of point cloud.",
    )
    args = parser.parse_args()

    if args.list:
        paths = fetch_by_extension("vtk")
        if not paths:
            print("No local VTK files cached. Run with a filename to download one.")
        else:
            print("Cached VTK files:")
            for p in paths:
                print(f"  {p}")
        sys.exit(0)

    if args.filename:
        path = fetch(args.filename)
    else:
        paths = fetch_by_extension("vtk")
        if not paths:
            print("No VTK files cached. Provide a filename argument to download one.")
            sys.exit(1)
        path = paths[0]
        print(f"No filename given — using first cached file: {path}")

    visualize(path=path, render_lines=args.lines)


if __name__ == "__main__":
    main()
