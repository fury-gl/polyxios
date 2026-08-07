from pathlib import Path
import xml.etree.ElementTree as ET

import polyxios
from polyxios.fetcher import fetch
import polyxios.transforms as transforms


def read_multiblock_vtp(path: Path) -> polyxios.PolyData:
    """Load a vtkMultiBlockDataSet .vtp index file and merge its components.

    Reads the provided .vtp file, extracts references to individual sub-dataset
    files (under the <vtkMultiBlockDataSet> element), reads each sub-file, and
    merges them into a single consolidated PolyData object.

    Parameters
    ----------
    path : Path
        The file path to the parent .vtp index file.

    Returns
    -------
    polyxios.PolyData
        A merged PolyData containing geometry and properties of all sub-datasets.

    Raises
    ------
    ValueError
        If the index file does not contain a <vtkMultiBlockDataSet> element.
    FileNotFoundError
        If no referenced sub-files could be successfully found/read.
    """
    raw = path.read_bytes()
    app_marker = raw.find(b"<AppendedData")
    xml_bytes = (raw[:app_marker] + b"</VTKFile>") if app_marker != -1 else raw
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))

    block = root.find("vtkMultiBlockDataSet")
    if block is None:
        raise ValueError(f"No <vtkMultiBlockDataSet> element found in '{path}'.")

    sub_paths = [
        path.parent / ds.get("file")
        for ds in block.findall("DataSet")
        if ds.get("file")
    ]

    polys = []
    for sub in sub_paths:
        if not sub.exists():
            raise FileNotFoundError(f"Sub-file not found: {sub}")
        polys.append(polyxios.read(str(sub)))

    if not polys:
        missing = "\n  ".join(str(p) for p in sub_paths[:5])
        raise FileNotFoundError(
            f"No sub-files found for '{path}'.\nExpected files such as:\n  {missing}"
        )

    return transforms.merge(*polys)


def read_polydata(filename: str) -> polyxios.PolyData:
    """Read a PolyData object from the given file, resolving fetches and VTP index files.

    Checks if the file exists locally. If it doesn't, resolves it by calling the
    local fetcher. If the file has a `.vtp` extension and fails to read normally,
    attempts to load and parse it as a VTK multiblock dataset file.

    Parameters
    ----------
    filename : str
        The target filename or full path to the 3D model file.

    Returns
    -------
    polyxios.PolyData
        The parsed PolyData object containing vertices, cells, and attributes.

    Raises
    ------
    RuntimeError
        If a .vtp file fails both single PolyData and multiblock parsing.
    Exception
        Any codec or format reading exceptions raised by the polyxios backend.
    """
    p = Path(filename)
    if p.exists() or p.parent != Path("."):
        path = p
    else:
        path = Path(fetch(filename))

    try:
        return polyxios.read(str(path))
    except Exception as e:
        if path.suffix.lower() == ".vtp":
            try:
                return read_multiblock_vtp(path)
            except Exception as multiblock_err:
                raise RuntimeError(
                    f"Failed to read VTP as single PolyData or MultiBlockDataSet:\n"
                    f"  Single PolyData error: {e}\n"
                    f"  MultiBlockDataSet error: {multiblock_err}"
                ) from multiblock_err
        raise
