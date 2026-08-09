"""Fetch and visualize a mesh file (any format supported by polyxios) with FURY.

Supported formats
-----------------
polyxios can read: .obj  .ply  .vtk  .vtp  .vtr  .vtu  .vti  .vts
                   .pvtp .pvtr .pvtu .pvti .pvts .vtm  .stl

Sample packs available via the fetcher (auto-downloaded from GitHub releases):
  obj  ply  vtk  vtp  vtr  vtu  stl
"""

import argparse
from pathlib import Path
import sys

import polyxios
from polyxios.fetcher import fetch, fetch_by_extension
from polyxios.helper import visualize_mesh

_FETCHABLE_EXTS = ("obj", "ply", "vtk", "vtp", "vtr", "vtu", "stl")


def visualize(*, path, render_lines=False):
    """Read and display a single mesh file.

    Parameters
    ----------
    path : str or Path
        Local path to the mesh file (any format polyxios supports).
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
    if len(poly.vertices) == 0:
        print("  No geometry (FIELD data) — skipping window.")
        return
    visualize_mesh(poly, lines=render_lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a mesh file via polyxios + FURY.\n\n"
            "Accepted formats: .obj  .ply  .vtk  .vtp  .vtr  .vtu  .vti  .vts\n"
            "                  .pvtp .pvtr .pvtu .pvti .pvts .vtm  .stl\n\n"
            "Sample packs for obj / ply / vtk / vtp / vtr / vtu / stl are fetched "
            "automatically from the polyxios-data GitHub release and cached under "
            "~/.polyxios/<ext>/. Pass --list to see what is already cached.\n"
            "STL example: python visualize_mesh.py 20mm-xyz-cube.stl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help=(
            "Filename to fetch and visualize (e.g. 'mesh.vtk', 'bunny.obj'), or a "
            "local path (relative or absolute). "
            "The extension determines which sample pack is downloaded when the file "
            "is not already on disk. "
            f"Fetchable extensions: {', '.join(_FETCHABLE_EXTS)}. "
            "Omit to use the first locally cached file for --ext (default: vtk)."
        ),
    )
    parser.add_argument(
        "--ext",
        default="vtk",
        metavar="EXT",
        help=(
            "Which sample pack to use when no filename is given "
            f"({', '.join(_FETCHABLE_EXTS)}). Default: vtk."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List locally cached files for --ext and exit.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Render line/poly_line elements with actor.line instead of point cloud.",
    )
    args = parser.parse_args()

    ext = args.ext.lower().lstrip(".")

    if args.list:
        paths = fetch_by_extension(ext)
        if not paths:
            print(
                f"No local .{ext} files cached.\n"
                f"Run without --list to download the sample pack."
            )
        else:
            print(f"Cached .{ext} files:")
            for p in paths:
                print(f"  {p}")
        sys.exit(0)

    if args.filename:
        p = Path(args.filename)
        if p.exists() or p.parent != Path("."):
            path = str(p)
        else:
            path = fetch(args.filename)
    else:
        paths = fetch_by_extension(ext)
        if not paths:
            print(
                f"No .{ext} files found in the sample pack. Try a different --ext.",
                file=sys.stderr,
            )
            sys.exit(1)
        path = paths[0]
        print(f"No filename given — using first cached .{ext} file: {path}")

    visualize(path=path, render_lines=args.lines)


if __name__ == "__main__":
    main()
