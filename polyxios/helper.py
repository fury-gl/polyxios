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
            raise FileNotFoundError(f"Sub-file not found: {sub}")
        polys.append(polyxios.read(str(sub)))

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
    if p.exists():
        path = p
    else:
        try:
            path = Path(fetch(p.name))
        except Exception as e:
            raise FileNotFoundError(
                f"No such file: '{filename}' (and not found in remote catalog: {e})"
            ) from e

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

    colors = None
    for col_attr in ["colors", "rgb", "red"]:
        if col_attr in polydata.vertex_attrs:
            colors = polydata.vertex_attrs[col_attr]
            break
    if colors is None:
        colors = transforms.vertex_colors(polydata)

    has_volume = any(
        t
        in [
            "tetra",
            "hexahedron",
            "wedge",
            "pyramid",
            "quadratic_tetra",
            "quadratic_hexahedron",
        ]
        for t in polydata.element_types
    )

    import logging

    logger = logging.getLogger("polyxios.helper")

    actors = []
    if points:
        logger.info("  Rendering strictly as point cloud per --points request.")
        actors.append(
            actor.point(
                polydata.vertices,
                colors=colors if colors is not None else (0.9, 0.9, 0.9),
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
                    colors=colors if colors is not None else (0.8, 0.7, 0.6),
                )
                surf_actor.material.wireframe = True
                try:
                    import numpy as np

                    coords = polydata.vertices
                    if len(coords) > 0:
                        bbox_min = coords.min(axis=0)
                        bbox_max = coords.max(axis=0)
                        diag = np.linalg.norm(bbox_max - bbox_min)
                        surf_actor.material.wireframe_thickness = max(diag * 0.05, 0.5)
                except Exception:
                    pass
                actors.append(surf_actor)
            else:
                logger.warning(
                    "  No line elements or surface faces found to render as wireframe."
                )
                actors.append(
                    actor.point(
                        polydata.vertices,
                        colors=colors if colors is not None else (0.9, 0.9, 0.9),
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
                    colors=colors if colors is not None else (0.8, 0.7, 0.6),
                )
            )
        else:
            logger.info("  No renderable geometry - rendering as point cloud.")
            actors.append(
                actor.point(
                    polydata.vertices,
                    colors=colors if colors is not None else (0.9, 0.9, 0.9),
                )
            )

    window.show(actors)
