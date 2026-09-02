from __future__ import annotations

import logging
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import polyxios
from polyxios.exceptions import UnsupportedFormatError
from polyxios.fetcher import fetch
import polyxios.transforms as transforms

logger = logging.getLogger("polyxios.helper")

_DEFAULT_SURFACE_COLOR = (0.8, 0.7, 0.6)
_DEFAULT_POINT_COLOR = (0.9, 0.9, 0.9)

# Wireframe thickness is a screen-space width in pixels, so it must not be
# scaled by the mesh bounding box.
_WIREFRAME_THICKNESS = 1.0

# Kept in sync with the surface extractor rather than hand-listed, so a volume
# element type added there is never rendered as raw interior faces here.
_VOLUME_ELEMENT_TYPES = frozenset(transforms._VOL_ELEMENT_FACES)


# The meta-file extensions: an index of sub-files, holding no geometry of its
# own. ``read`` refuses each of them by name, because a mesh reader that
# sometimes hands back several meshes is a different function; the ones here
# are where the several are put back together.
META_SUFFIXES: frozenset[str] = frozenset(
    {".vtm", ".pvtu", ".pvtp", ".pvtr", ".pvts", ".pvti"}
)

# How deep a chain of indexes naming indexes is followed. The walk already
# refuses one that names a parent of its own, but a chain of files each
# naming the next is not a cycle and would recurse as far as it is long -
# past the interpreter's own limit, which arrives as a RecursionError out of
# whatever frame happened to be deepest. A real case is a handful deep.
_MAX_INDEX_DEPTH: int = 32


class Traversal(PermissionError):
    """A sub-file reference that points outside its own index's directory.

    A :class:`PermissionError` by inheritance, which is what the readers here
    have always raised for this and what their callers catch. Its own class is
    what separates it from the one the operating system raises over a sub-file
    the caller may not open: the walk skips that block and carries on, the way
    it does for one that is missing, and refuses the whole file for this.
    """


def _index_root(path: Path) -> ET.Element:
    """Parse an index file, stepping over any appended binary payload.

    Parameters
    ----------
    path
        The index file.

    Returns
    -------
    xml.etree.ElementTree.Element
        The document root.
    """
    raw = path.read_bytes()
    app_marker = raw.find(b"<AppendedData")
    xml_bytes = (raw[:app_marker] + b"</VTKFile>") if app_marker != -1 else raw
    return ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))


def _referenced_paths(root: ET.Element, parent: Path) -> list[Path]:
    """Resolve every sub-file an index names, in the order it names them.

    Parameters
    ----------
    root
        The index document's root element.
    parent
        The directory the index sits in; a reference is relative to it.

    Returns
    -------
    list of pathlib.Path
        The resolved sub-file paths.

    Raises
    ------
    Traversal
        If a reference resolves outside ``parent``. An index is data, and one
        naming ``../../etc/passwd`` is asking for a file the caller did not.
        A :class:`PermissionError`, so a caller catching that still catches
        this; its own class is what tells it apart from the one the operating
        system raises over a sub-file nobody may read.

    Notes
    -----
    A multi-block index names its sub-files in ``file=`` and a parallel one in
    ``Source=``, and both nest them - inside ``<Block>``, or under a
    ``<PUnstructuredGrid>``. Walking the whole tree for either attribute reads
    both families without a table of which element holds which, and a document
    that names none is empty rather than misread.
    """
    resolved_parent = parent.resolve()
    found: list[Path] = []
    for elem in root.iter():
        name = elem.get("file") or elem.get("Source")
        if not name:
            continue
        sub = (parent / name).resolve()
        if not sub.is_relative_to(resolved_parent):
            raise Traversal(f"Path traversal detected: '{name}' is outside '{parent}'")
        found.append(sub)
    return found


def read_blocks(path: str | Path) -> list[polyxios.PolyData]:
    """Read every mesh a VTK index file names, one PolyData per sub-file.

    ``read`` hands back one mesh, always - a function that sometimes returns a
    sequence puts the branch in every caller. A file that holds several is an
    index rather than a mesh, so it is read here instead, and what to do with
    the pieces is the caller's: merge them with
    :func:`~polyxios.transforms.merge`, or keep them apart.

    Parameters
    ----------
    path
        A ``.vtm``, ``.pvtu``, ``.pvtp``, ``.pvtr``, ``.pvts`` or ``.pvti``
        index, or a ``.vtp`` holding a ``<vtkMultiBlockDataSet>``.

    Returns
    -------
    list of polyxios.PolyData
        One mesh per sub-file that was found and read, in the order the index
        names them. A nested ``.vtm`` or ``.pvt*`` contributes its own blocks
        in place, so a tree of them reads as one flat list; a nested ``.vtp``
        index contributes one, its own blocks merged, since at that depth
        there is nothing to tell a block of a mesh from a mesh.

    Raises
    ------
    ValueError
        If the file names no sub-files at all.
    Traversal
        If a reference resolves outside the index file's directory. A
        :class:`PermissionError`, and the only one this raises: the operating
        system's own, over a sub-file nobody may open, names a block that
        cannot be read like any other and is skipped with a warning.
    FileNotFoundError
        If none of the referenced sub-files could be read.

    Notes
    -----
    A sub-file that is missing, or that fails to read, is skipped with a
    warning rather than refusing the rest: a partially downloaded companion
    directory still loads.

    Examples
    --------
    >>> from polyxios import helper, transforms       # doctest: +SKIP
    >>> blocks = helper.read_blocks("case.vtm")       # doctest: +SKIP
    >>> whole = transforms.merge(*blocks)             # doctest: +SKIP
    """
    return _read_blocks(Path(path), seen=frozenset())


def _read_one(sub: Path, *, seen: frozenset[Path]) -> polyxios.PolyData:
    """Read one sub-file a block index named, whichever kind it turns out to be.

    Parameters
    ----------
    sub
        The sub-file, already resolved inside the index's own directory.
    seen
        The index files this sub-file hangs under, passed on so a nested index
        cannot walk back into one of its own parents.

    Returns
    -------
    polyxios.PolyData
        The mesh, or the blocks of a nested index merged into one.

    Notes
    -----
    Only the six meta extensions name an index by their spelling. A ``.vtp``
    is the exception: the same extension holds a mesh and a
    ``vtkMultiBlockDataSet`` index, and only the document says which. Rather
    than parse every one twice to find out, the reader is asked first and its
    refusal - the one it raises for a ``.vtp`` that is not ``PolyData`` - is
    what sends the file back here as the index it is.
    """
    try:
        return polyxios.read(str(sub))
    except UnsupportedFormatError:
        if sub.suffix.lower() != ".vtp":
            raise
        return transforms.merge(*_read_blocks(sub, seen=seen))


def _read_blocks(
    path: Path, *, seen: frozenset[Path], root: ET.Element | None = None
) -> list[polyxios.PolyData]:
    """Read one index file's blocks, following the indexes it names.

    Parameters
    ----------
    path
        The index file.
    seen
        The index files this one hangs under, so a pair of them naming each
        other is read once rather than for ever. The chain of parents rather
        than every index the walk has opened: two indexes that both name a
        third are naming two blocks, and a set of everything seen would hand
        back one of them and drop the other without the caller asking.
    root
        The document root, when the caller has already parsed it. Parsed here
        otherwise.

    Returns
    -------
    list of polyxios.PolyData
        The meshes, in the order the index names them.

    Raises
    ------
    Traversal
        If a reference resolves outside its own index's directory, at any
        depth. It is the one failure a sub-file is not skipped over: the
        others say a block is missing, this one says the file is asking for
        something it was not given, and a walk that swallowed it would answer
        a traversal with a warning in a log. Caught by its own class rather
        than as the :class:`PermissionError` it is, so that the one the
        operating system raises over a sub-file the caller may not open is
        skipped the way every other unreadable block is.
    RecursionError
        If the chain of indexes above this one is longer than
        :data:`_MAX_INDEX_DEPTH`. Raised rather than followed, and caught by
        the walk one level up, so the block is skipped with a warning naming
        it the way an unreadable one is.
    """
    if len(seen) >= _MAX_INDEX_DEPTH:
        raise RecursionError(
            f"index files are nested more than {_MAX_INDEX_DEPTH} deep"
        )
    seen = seen | {path.resolve()}
    sub_paths = _referenced_paths(
        _index_root(path) if root is None else root, path.parent
    )
    if not sub_paths:
        raise ValueError(f"No sub-files found in index file '{path}'.")

    blocks: list[polyxios.PolyData] = []
    for sub in sub_paths:
        if not sub.exists():
            logger.warning(f"  Sub-file not found, skipping: {sub}")
            continue
        if sub in seen:
            logger.warning(f"  Index names one of its own parents, skipping: {sub}")
            continue
        try:
            if sub.suffix.lower() in META_SUFFIXES:
                blocks.extend(_read_blocks(sub, seen=seen))
            else:
                blocks.append(_read_one(sub, seen=seen))
        except Traversal:
            raise
        except Exception as e:
            logger.warning(f"  Sub-file could not be read, skipping: {sub} ({e})")

    if not blocks:
        missing = "\n  ".join(str(p) for p in sub_paths[:5])
        raise FileNotFoundError(
            f"No sub-files found for '{path}'.\n"
            f"Expected files such as:\n  {missing}\n"
            "Ensure the companion directory is present alongside the index file."
        )

    logger.info(f"Loaded {len(blocks)} block(s) from {len(sub_paths)} reference(s).")
    return blocks


def read_multiblock(path: str | Path) -> polyxios.PolyData:
    """Read a VTK index file and merge every mesh it names into one.

    Parameters
    ----------
    path
        A ``.vtm``, ``.pvtu``, ``.pvtp``, ``.pvtr``, ``.pvts`` or ``.pvti``
        index, or a ``.vtp`` holding a ``<vtkMultiBlockDataSet>``.

    Returns
    -------
    polyxios.PolyData
        The blocks, merged. ``merge`` renumbers each block's connectivity
        against the joined vertex array and drops ``global_attrs``, since two
        meshes carry two sets of metadata.

    Raises
    ------
    ValueError
        If the file names no sub-files at all.
    Traversal
        If a reference resolves outside the index file's directory. A
        :class:`PermissionError`, and the only one this raises: the operating
        system's own, over a sub-file nobody may open, names a block that
        cannot be read like any other and is skipped with a warning.
    FileNotFoundError
        If none of the referenced sub-files could be read.

    See Also
    --------
    read_blocks : the same meshes, kept apart.
    """
    return transforms.merge(*read_blocks(path))


def read_multiblock_vtp(path: str | Path) -> polyxios.PolyData:
    """Load a vtkMultiBlockDataSet .vtp index file and merge its components.

    Reads the provided .vtp file, extracts references to individual sub-dataset
    files (under the <vtkMultiBlockDataSet> element), reads each sub-file, and
    merges them into a single consolidated PolyData object. Referenced sub-files
    that are missing on disk, or that fail to read, are skipped with a warning,
    so a partially downloaded companion directory still loads.

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
    Traversal
        If a reference resolves outside the index file's directory. A
        :class:`PermissionError`, and the only one this raises: the operating
        system's own, over a sub-file nobody may open, names a block that
        cannot be read like any other and is skipped with a warning.
    FileNotFoundError
        If no referenced sub-files could be successfully found/read.

    See Also
    --------
    read_multiblock : the same, for every index spelling rather than this one.
    read_blocks : the blocks kept apart instead of merged.

    Notes
    -----
    This is :func:`read_multiblock` with one extra demand: the document has to
    hold a ``<vtkMultiBlockDataSet>`` element, which is what tells a ``.vtp``
    index from a ``.vtp`` mesh. A ``.vtm`` or a ``.pvtu`` is named by its own
    extension and needs no such check, so it goes to :func:`read_multiblock`.
    """
    path = Path(path)
    # Parsed once and handed on: the index is read whole to find the element,
    # and reading it a second time to find the same references is a file the
    # size of the whole document for nothing.
    root = _index_root(path)
    if root.find("vtkMultiBlockDataSet") is None:
        raise ValueError(f"No <vtkMultiBlockDataSet> element found in '{path}'.")
    return transforms.merge(*_read_blocks(path, seen=frozenset(), root=root))


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

    Only a bare asset name (no directory component) is looked up in the remote
    catalog. A path such as ``data/bunny.obj`` that does not exist is reported
    as missing rather than silently replaced by an unrelated catalog asset that
    happens to share its basename.

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
    if p.parent != Path("."):
        raise FileNotFoundError(f"No such file: '{filename}'")
    try:
        return Path(fetch(p.name))
    except Exception as e:
        raise FileNotFoundError(
            f"No such file: '{filename}' (and not found in remote catalog: {e})"
        ) from e


def read_polydata(filename: str | Path) -> polyxios.PolyData:
    """Read a PolyData object from the given file, resolving fetches and VTP index files.

    Checks if the file exists locally. If it doesn't, resolves it by calling the
    local fetcher. If the file has a `.vtp` extension and fails to read normally,
    attempts to load and parse it as a VTK multiblock dataset file.

    Parameters
    ----------
    filename : str or Path
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
    n_verts = len(polydata.vertices)

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
        if lines_list:
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
                    colors=colors
                    if colors is not None
                    else _uniform_colors(_DEFAULT_SURFACE_COLOR, n_verts),
                )
                surf_actor.material.wireframe = True
                surf_actor.material.wireframe_thickness = _WIREFRAME_THICKNESS
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
                    colors=colors
                    if colors is not None
                    else _uniform_colors(_DEFAULT_SURFACE_COLOR, n_verts),
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
