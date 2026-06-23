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
import xml.etree.ElementTree as ET

import polyxios
from polyxios.fetcher import fetch, fetch_by_extension
import polyxios.transforms as transforms


def read_multiblock_vtp(path: str | Path) -> polyxios.PolyData:
    """Load a vtkMultiBlockDataSet .vtp index file into a merged PolyData.

    Parameters
    ----------
    path
        Path to the top-level .vtp index file.

    Returns
    -------
    PolyData
        All present sub-meshes merged into one.

    Raises
    ------
    FileNotFoundError
        If none of the referenced sub-files exist on disk (e.g. the companion
        directory was never downloaded).
    ValueError
        If the file does not contain a <vtkMultiBlockDataSet> element.
    """
    path = Path(path)

    # ------------------------------------------------------------------
    # Step 1 — Parse the index XML
    #
    # The file may have a binary <AppendedData> section after the XML, which
    # would cause xml.etree.ElementTree to raise ParseError on the raw binary
    # bytes.  We truncate the byte string at the "<AppendedData" marker before
    # parsing, so the parser only ever sees valid UTF-8 XML.
    # ------------------------------------------------------------------
    raw = path.read_bytes()
    app_marker = raw.find(b"<AppendedData")
    xml_bytes = (raw[:app_marker] + b"</VTKFile>") if app_marker != -1 else raw
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))

    # ------------------------------------------------------------------
    # Step 2 — Locate the <vtkMultiBlockDataSet> element
    #
    # The root <VTKFile> element wraps a <vtkMultiBlockDataSet> child that
    # contains one <DataSet index="N" file="relative/path.vtp"/> entry per
    # mesh piece.  The file attribute is always a path relative to the
    # directory that contains the index file — never absolute.
    # ------------------------------------------------------------------
    block = root.find("vtkMultiBlockDataSet")
    if block is None:
        raise ValueError(
            f"No <vtkMultiBlockDataSet> element found in '{path}'. "
            f"The file declares type='{root.get('type', '?')}', not 'vtkMultiBlockDataSet'."
        )

    sub_paths = [
        path.parent / ds.get("file")
        for ds in block.findall("DataSet")
        if ds.get("file")
    ]
    print(f"Index file lists {len(sub_paths)} sub-file(s).")

    # ------------------------------------------------------------------
    # Step 3 — Load each present sub-file, skip missing ones with a warning
    #
    # When the companion directory is absent (e.g. you only downloaded the
    # index .vtp without its sibling folder), sub-files will not exist.
    # We skip them and report which paths were missing so the user knows
    # what to fetch, rather than crashing without explanation.
    # ------------------------------------------------------------------
    polys: list[polyxios.PolyData] = []
    for sub in sub_paths:
        if not sub.exists():
            print(f"  WARNING: sub-file not found, skipping — {sub}")
            continue
        print(f"  Loading {sub.name} …")
        polys.append(polyxios.read(sub))

    if not polys:
        missing = "\n  ".join(str(p) for p in sub_paths[:5])
        raise FileNotFoundError(
            f"No sub-files found for '{path}'.\n"
            f"Expected files such as:\n  {missing}\n"
            "Ensure the companion directory is present alongside the index file."
        )

    print(f"Loaded {len(polys)} of {len(sub_paths)} sub-file(s).")

    # ------------------------------------------------------------------
    # Step 4 — Merge all pieces into a single PolyData
    #
    # transforms.merge() takes care of:
    #   • offsetting connectivity indices so each piece's vertex indices do
    #     not collide with those from earlier pieces;
    #   • concatenating offsets correctly so element boundaries are preserved;
    #   • taking the union of vertex and element attribute keys, filling any
    #     attribute absent from a particular piece with NaN (floats) or -1
    #     (integers).
    #
    # The result is a frozen PolyData identical in structure to what
    # polyxios.read() returns for a plain single-piece .vtp file.
    # ------------------------------------------------------------------
    merged = transforms.merge(*polys)
    return merged


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
        paths = fetch_by_extension("vtp")
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
