"""
Reading parallel and multi-block VTK files
==========================================

Background
----------
Several VTK file formats are *meta-files*: they contain no geometry themselves,
only an XML list of references to individual data files that hold the actual
mesh pieces.  polyxios raises ``UnsupportedFormatError`` for all of these and
forwards you here.

The affected extensions and their VTK dataset types:

============  ===========================  ===========================
Extension     VTK type                     Sub-file attribute
============  ===========================  ===========================
``.vtm``      vtkMultiBlockDataSet         ``<DataSet file="..."/>``
``.pvtu``     Parallel UnstructuredGrid    ``<Piece Source="..."/>``
``.pvtp``     Parallel PolyData            ``<Piece Source="..."/>``
``.pvtr``     Parallel RectilinearGrid     ``<Piece Source="..."/>``
``.pvts``     Parallel StructuredGrid      ``<Piece Source="..."/>``
``.pvti``     Parallel ImageData           ``<Piece Source="..."/>``
============  ===========================  ===========================

In every case the loading strategy is the same:

  1. Parse the index XML to collect sub-file paths.
  2. Resolve each relative path against the index file's directory.
  3. Load every present sub-file with ``polyxios.read()``.
  4. Merge all pieces into one ``PolyData`` with ``transforms.merge()``.

This script implements that strategy for both format families.
"""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import polyxios
import polyxios.transforms as transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_xml_safe(path: Path) -> ET.Element:
    """Parse a VTK XML file, stripping binary AppendedData if present.

    Some meta-files may embed compressed field data after the XML.
    Truncating at ``<AppendedData`` prevents xml.etree from choking on
    non-UTF-8 bytes.
    """
    raw = path.read_bytes()
    app = raw.find(b"<AppendedData")
    xml_bytes = (raw[:app] + b"</VTKFile>") if app != -1 else raw
    return ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))


def _collect_and_merge(index_path: Path, sub_paths: list[Path]) -> polyxios.PolyData:
    """Load present sub-files, warn on missing ones, merge and return."""
    polys: list[polyxios.PolyData] = []
    for sub in sub_paths:
        if not sub.exists():
            print(f"  WARNING: sub-file not found, skipping — {sub}")
            continue
        print(f"  Loading {sub.name} …")
        polys.append(polyxios.read(sub))

    if not polys:
        sample = "\n  ".join(str(p) for p in sub_paths[:5])
        raise FileNotFoundError(
            f"No sub-files found for '{index_path}'.\n"
            f"Expected files such as:\n  {sample}\n"
            "Ensure sub-files are present alongside the index file."
        )

    print(f"Loaded {len(polys)} of {len(sub_paths)} piece(s). Merging …")
    return transforms.merge(*polys)


# ---------------------------------------------------------------------------
# vtkMultiBlockDataSet  (.vtm)
# ---------------------------------------------------------------------------


def read_vtm(path: str | Path) -> polyxios.PolyData:
    """Load a ``.vtm`` vtkMultiBlockDataSet index file into a merged PolyData.

    Parameters
    ----------
    path
        Path to the ``.vtm`` index file.

    Returns
    -------
    PolyData
        All geometry from all blocks merged into one.

    Notes
    -----
    A ``.vtm`` file groups datasets into named ``<Block>`` elements, each
    containing ``<DataSet index="N" file="rel/path.ext"/>`` entries.  Blocks
    can be nested.  This reader flattens the hierarchy: every leaf
    ``<DataSet>`` whose ``file`` attribute points to a supported extension
    is loaded and merged.
    """
    path = Path(path)

    # ------------------------------------------------------------------
    # Step 1 — Parse the index XML
    # The file may carry compressed AppendedData; strip it before parsing.
    # ------------------------------------------------------------------
    root = _parse_xml_safe(path)

    block = root.find("vtkMultiBlockDataSet")
    if block is None:
        vtk_type = root.get("type", "?")
        raise ValueError(
            f"Expected <vtkMultiBlockDataSet> but got type='{vtk_type}' in '{path}'."
        )

    # ------------------------------------------------------------------
    # Step 2 — Walk the (possibly nested) block tree and collect file refs
    #
    # Blocks can be nested arbitrarily deep.  ``iter("DataSet")`` flattens
    # the tree so we don't need a recursive walk.  Each <DataSet> entry may
    # point to any VTK format that polyxios supports.
    # ------------------------------------------------------------------
    sub_paths = [
        path.parent / ds.get("file") for ds in block.iter("DataSet") if ds.get("file")
    ]
    print(f"vtm index lists {len(sub_paths)} dataset(s).")

    # ------------------------------------------------------------------
    # Step 3 & 4 — Load present sub-files and merge
    # (see _collect_and_merge above)
    # ------------------------------------------------------------------
    return _collect_and_merge(path, sub_paths)


# ---------------------------------------------------------------------------
# Parallel formats  (.pvtu / .pvtp / .pvtr / .pvts / .pvti)
# ---------------------------------------------------------------------------

# Each parallel format wraps a different VTK dataset type, but the index
# structure is always the same: <P{Type}> → <Piece Source="rel/path.vXu"/>.
_PARALLEL_ROOTS = {
    ".pvtu": "PUnstructuredGrid",
    ".pvtp": "PPolyData",
    ".pvtr": "PRectilinearGrid",
    ".pvts": "PStructuredGrid",
    ".pvti": "PImageData",
}


def read_parallel(path: str | Path) -> polyxios.PolyData:
    """Load any parallel VTK index file (.pvtu / .pvtp / .pvtr / .pvts / .pvti).

    Parameters
    ----------
    path
        Path to the parallel index file.

    Returns
    -------
    PolyData
        All pieces merged into one PolyData.

    Notes
    -----
    Parallel VTK files list one ``<Piece Source="relative/path.vXu"/>`` per
    MPI rank.  Each piece is an ordinary single-format VTK file (e.g. ``.vtu``
    for ``.pvtu``).  The ``<PPointData>`` / ``<PCellData>`` blocks in the
    index only declare *schema*; the actual array data lives in the pieces.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # ------------------------------------------------------------------
    # Step 1 — Identify the root element name for this extension
    # ------------------------------------------------------------------
    root_tag = _PARALLEL_ROOTS.get(ext)
    if root_tag is None:
        raise ValueError(
            f"Unknown parallel VTK extension '{ext}'. "
            f"Supported: {list(_PARALLEL_ROOTS)}"
        )

    # ------------------------------------------------------------------
    # Step 2 — Parse the index XML and collect <Piece Source="..."/> refs
    #
    # Unlike .vtm, parallel formats use a flat list of Piece elements.
    # The Source attribute holds the path to the actual data file.
    # ------------------------------------------------------------------
    root = _parse_xml_safe(path)
    dataset_elem = root.find(root_tag)
    if dataset_elem is None:
        raise ValueError(f"No <{root_tag}> element found in '{path}'.")

    sub_paths = [
        path.parent / piece.get("Source")
        for piece in dataset_elem.findall("Piece")
        if piece.get("Source")
    ]
    print(f"{ext} index lists {len(sub_paths)} piece(s).")

    # ------------------------------------------------------------------
    # Step 3 & 4 — Load present sub-files and merge
    # ------------------------------------------------------------------
    return _collect_and_merge(path, sub_paths)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def read_meta_vtk(path: str | Path) -> polyxios.PolyData:
    """Dispatch to the right loader based on file extension.

    Parameters
    ----------
    path
        Path to a ``.vtm`` or parallel VTK index file.

    Returns
    -------
    PolyData
        Merged mesh from all sub-files.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".vtm":
        return read_vtm(path)
    if ext in _PARALLEL_ROOTS:
        return read_parallel(path)
    raise ValueError(
        f"Unsupported extension '{ext}'. Supported: .vtm, {', '.join(_PARALLEL_ROOTS)}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python read_parallel_vtk.py <file.vtm|file.pvtu|...>\n"
            "Loads a parallel/multi-block VTK index file and prints mesh stats."
        )
        sys.exit(1)

    meta_path = sys.argv[1]
    print(f"Reading meta-file: {meta_path}\n")

    poly = read_meta_vtk(meta_path)

    print(
        f"\nMerged result:\n"
        f"  vertices      : {len(poly.vertices):,}\n"
        f"  elements      : {len(poly.element_types):,}\n"
        f"  vertex attrs  : {list(poly.vertex_attrs) or 'none'}\n"
        f"  element attrs : {list(poly.element_attrs) or 'none'}"
    )
