import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import polyxios
from polyxios.fetcher import fetch
import polyxios.transforms as transforms

logger = logging.getLogger("polyxios.helper")

_DEFAULT_SURFACE_COLOR = (0.8, 0.7, 0.6)
_DEFAULT_POINT_COLOR = (0.9, 0.9, 0.9)

_VOLUME_ELEMENT_TYPES = frozenset(
    {
        "tetra",
        "hexahedron",
        "wedge",
        "pyramid",
        "quadratic_tetra",
        "quadratic_hexahedron",
    }
)


def read_multiblock_vtp(path: str | Path) -> polyxios.PolyData:
    """Load a vtkMultiBlockDataSet .vtp index file and merge its components.

    Reads the provided .vtp file, extracts references to individual sub-dataset
    files (under the <vtkMultiBlockDataSet> element), reads each sub-file, and
    merges them into a single consolidated PolyData object. Referenced sub-files
    that are missing on disk are skipped with a warning, so a partially
    downloaded companion directory still loads.

    Parameters
    ----------
    path : str or Path
        The file path to the parent .vtp index file.

    Returns
    -------
    polyxios.PolyData
        A merged PolyData containing geometry and properties of all sub-datasets.

    Raises
    ------
    ValueError
        If the index file does not contain a <vtkMultiBlockDataSet> element.
    PermissionError
        If a referenced sub-file resolves outside the index file's directory.
    FileNotFoundError
        If no referenced sub-files could be successfully found/read.
    """
    path = Path(path)
    raw = path.read_bytes()
    app_marker = raw.find(b"<AppendedData")
    xml_bytes = (raw[:app_marker] + b"</VTKFile>") if app_marker != -1 else raw
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))

    block = root.find("vtkMultiBlockDataSet")
    if block is None:
        raise ValueError(f"No <vtkMultiBlockDataSet> element found in '{path}'.")

    sub_paths = []
    parent_resolve = path.parent.resolve()
    for ds in block.findall("DataSet"):
        name = ds.get("file")
        if name:
            sub = (path.parent / name).resolve()
            if not sub.is_relative_to(parent_resolve):
                raise PermissionError(
                    f"Path traversal detected: '{name}' is outside '{path.parent}'"
                )
            sub_paths.append(sub)

    if not sub_paths:
        raise ValueError(f"No sub-files found in index file '{path}'.")

    polys = []
    for sub in sub_paths:
        if not sub.exists():
            logger.warning(f"  Sub-file not found, skipping: {sub}")
            continue
        polys.append(polyxios.read(str(sub)))

    if not polys:
        missing = "\n  ".join(str(p) for p in sub_paths[:5])
        raise FileNotFoundError(
            f"No sub-files found for '{path}'.\n"
            f"Expected files such as:\n  {missing}\n"
            "Ensure the companion directory is present alongside the index file."
        )

    logger.info(f"Loaded {len(polys)} of {len(sub_paths)} sub-file(s).")
    return transforms.merge(*polys)


def _uniform_colors(color: tuple[float, float, float], n_verts: int) -> np.ndarray:
    """Broadcast a single RGB color to one float32 entry per vertex.

    Parameters
    ----------
    color : tuple of float
        RGB components in [0, 1].
    n_verts : int
        Number of vertices to color.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (n_verts, 3). FURY's surface actor rejects a
        single color tuple, so per-vertex colors are always supplied.
    """
    return np.tile(np.asarray(color, dtype=np.float32), (n_verts, 1))


def resolve_path(filename: str | Path) -> Path:
    """Resolve a filename to a local path, fetching it from the catalog if needed.

    Parameters
    ----------
    filename : str or Path
        A local path, or the bare name of an asset listed in the remote catalog.

    Returns
    -------
    Path
        The local path of the file.

    Raises
    ------
    FileNotFoundError
        If the file exists neither locally nor in the remote catalog.
    """
    p = Path(filename)
    if p.exists():
        return p
    try:
        return Path(fetch(p.name))
    except Exception as e:
        raise FileNotFoundError(
            f"No such file: '{filename}' (and not found in remote catalog: {e})"
        ) from e


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
    path = resolve_path(filename)

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


def visualize_mesh(
    polydata: polyxios.PolyData, *, lines: bool = False, points: bool = False
) -> None:
    """Visualize a PolyData mesh using FURY.

    Parameters
    ----------
    polydata : PolyData
        The PolyData object to visualize.
    lines : bool, optional
        Render line elements or surface wireframe.
    points : bool, optional
        Render strictly as a point cloud.
    """
    try:
        from fury import actor, window
    except ImportError as e:
        raise ImportError(
            "FURY is not installed. Please install it to visualize models:\n"
            "  pip install polyxios[viz]"
        ) from e

    if len(polydata.vertices) == 0:
        return

    # transforms.vertex_colors picks the first (n_verts, >= 3) attribute and
    # normalizes it to floats in [0, 1], which is what the actors expect.
    colors = transforms.vertex_colors(polydata)
    surface_colors = _uniform_colors(_DEFAULT_SURFACE_COLOR, len(polydata.vertices))

    has_volume = any(t in _VOLUME_ELEMENT_TYPES for t in polydata.element_types)

    actors = []
    if points:
        logger.info("  Rendering strictly as point cloud per --points request.")
        actors.append(
            actor.point(
                polydata.vertices,
                colors=colors if colors is not None else _DEFAULT_POINT_COLOR,
            )
        )
    elif lines:
        lines_list = polydata.lines
        if lines_list and len(lines_list) > 0:
            lines_coords = [
                polydata.vertices[idx].astype("float64") for idx in lines_list
            ]
            logger.info(
                f"  Rendering {len(lines_coords)} line segment(s) with actor.line."
            )
            actors.append(actor.line(lines_coords, colors=(0.2, 0.8, 0.2)))
        else:
            faces = polydata.faces
            if faces is None or has_volume:
                surface = transforms.extract_surface(polydata)
                faces = surface.faces
            if faces is not None and len(faces) > 0:
                logger.info("  No line elements found. Rendering surface wireframe.")
                surf_actor = actor.surface(
                    vertices=polydata.vertices,
                    faces=faces,
                    colors=colors if colors is not None else surface_colors,
                )
                surf_actor.material.wireframe = True
                coords = polydata.vertices
                bbox_min = coords.min(axis=0)
                bbox_max = coords.max(axis=0)
                diag = float(np.linalg.norm(bbox_max - bbox_min))
                surf_actor.material.wireframe_thickness = max(diag * 0.005, 0.5)
                actors.append(surf_actor)
            else:
                logger.warning(
                    "  No line elements or surface faces found to render as wireframe."
                )
                actors.append(
                    actor.point(
                        polydata.vertices,
                        colors=colors if colors is not None else _DEFAULT_POINT_COLOR,
                    )
                )
    else:
        faces = polydata.faces
        if faces is None or has_volume:
            surface = transforms.extract_surface(polydata)
            faces = surface.faces
        if faces is not None and len(faces) > 0:
            actors.append(
                actor.surface(
                    vertices=polydata.vertices,
                    faces=faces,
                    colors=colors if colors is not None else surface_colors,
                )
            )
        else:
            logger.info("  No renderable geometry - rendering as point cloud.")
            actors.append(
                actor.point(
                    polydata.vertices,
                    colors=colors if colors is not None else _DEFAULT_POINT_COLOR,
                )
            )

    window.show(actors)
