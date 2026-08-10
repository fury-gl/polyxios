"""
Reading a vtkMultiBlockDataSet VTP file
========================================

Background
----------
A ``.vtp`` file normally contains a single ``PolyData`` mesh.  Some VTK exporters
instead write a *multi-block dataset*: the ``.vtp`` file is a pure index — it has
no geometry of its own, only an XML list of paths to individual PolyData sub-files
stored in a companion sub-directory.

Example structure on disk::

    ExportBunny.vtp                  ← the index file
    ExportBunny/
        ExportBunny_0.vtp            ← piece 0 (genuine PolyData)
        ExportBunny_1.vtp            ← piece 1
        ...
        ExportBunny_46.vtp           ← piece 46

``polyxios.read()`` raises ``UnsupportedFormatError`` for these index files and
forwards you here.  This script shows the four steps needed to load such a dataset
and produce a single ``PolyData`` object.
"""

import argparse
from pathlib import Path
import sys

from polyxios.fetcher import fetch, get_cached_files
from polyxios.helper import read_multiblock_vtp


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read a vtkMultiBlockDataSet .vtp index file and merge all pieces into a "
            "single PolyData.\n\n"
            "Files are fetched automatically from the polyxios-data release package "
            "(vtp.zip, hosted on GitHub) and cached under ~/.polyxios/vtp/. "
            "Use --list to see what is already cached locally. "
            "Pass a local path directly if the file is already on disk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help=(
            "VTP filename to fetch and read (e.g. 'ExportBunny.vtp'), or a local "
            "path (relative or absolute). "
            "Omit to use the default multiblock sample 'ExportBunny.vtp' "
            "(auto-downloaded if not cached)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List locally cached VTP files and exit.",
    )
    args = parser.parse_args()

    if args.list:
        paths = get_cached_files("vtp")
        if not paths:
            print(
                "No local VTP files cached.\n"
                "Run without --list (optionally with a filename) to download the sample pack."
            )
        else:
            print("Cached VTP files:")
            for p in paths:
                print(f"  {p}")
        sys.exit(0)

    if args.filename:
        p = Path(args.filename)
        if p.exists() or p.parent != Path("."):
            vtp_path = p
        else:
            vtp_path = Path(fetch(args.filename))
    else:
        vtp_path = Path(fetch("ExportBunny.vtp"))
        print(f"No filename given — using default: {vtp_path}")

    print(f"\nReading multi-block VTP: {vtp_path}\n")
    poly = read_multiblock_vtp(vtp_path)
    print(
        f"\nMerged result:\n"
        f"  vertices : {len(poly.vertices):,}\n"
        f"  elements : {len(poly.element_types):,}\n"
        f"  vertex attrs  : {list(poly.vertex_attrs) or 'none'}\n"
        f"  element attrs : {list(poly.element_attrs) or 'none'}"
    )


if __name__ == "__main__":
    main()
