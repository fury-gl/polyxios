"""Fetch and visualize a VTK legacy dataset (UNSTRUCTURED_GRID or POLYDATA) with FURY."""

import argparse
import sys

from fury import actor, window

import polyxios
from polyxios.fetcher import fetch, fetch_by_extension
import polyxios.transforms as transforms


def _build_actors(*, poly):
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
    print("  No renderable geometry — rendering as point cloud.")
    return [actor.point(poly.vertices, colors=(0.9, 0.9, 0.9))]


def visualize(*, path):
    """Fetch, read, and display a single VTK file.

    Parameters
    ----------
    path : str
        Local path to a VTK legacy file.
    """
    print(f"Reading {path} ...")
    poly = polyxios.read(path)
    print(
        f"  {len(poly.vertices)} vertices | "
        f"{len(poly.element_types)} elements | "
        f"vertex attrs: {list(poly.vertex_attrs) or 'none'}"
    )
    window.show(_build_actors(poly=poly))


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

    visualize(path=path)


if __name__ == "__main__":
    main()
