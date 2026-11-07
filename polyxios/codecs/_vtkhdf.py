"""VTKHDF codec - VTK's own HDF5 layout, the file ParaView is moving to.

A VTKHDF file is one HDF5 file with a ``/VTKHDF`` group at its root, two
attributes on that group - ``Version``, two integers, and ``Type``, the VTK
data object it holds - and the object's arrays as datasets under it. An
``UnstructuredGrid`` keeps ``Points``, ``Connectivity``, ``Offsets`` and
``Types``; a ``PolyData`` its points and four cell groups, ``Vertices``,
``Lines``, ``Polygons`` and ``Strips``, each with a connectivity and offsets
of its own; an ``ImageData`` no points at all, only the ``WholeExtent``,
``Origin``, ``Spacing`` and ``Direction`` attributes its lattice is expanded
from. Each holds its arrays under ``PointData``, ``CellData`` and
``FieldData``. The file is partitioned - ``NumberOfPoints`` and
``NumberOfCells`` hold one entry per part - and, when a ``Steps`` group is
there, temporal: every dataset is the steps' arrays laid end to end, and
``Steps`` says where each step's begin.

:func:`read` hands back one mesh, always: every part at one step merged, the
first step unless ``step=`` says otherwise. The whole series is read by
:func:`read_time_series` and written by :func:`write_time_series`, both
reached through :mod:`polyxios.helper`.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
import operator
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV, POLYXIOS_TO_VTK
from polyxios._globals import spell_globals, spell_text
from polyxios._image import (
    DIRECTION_KEY as _DIRECTION_KEY,
    IMAGE_RESERVED_GLOBALS,
    TYPE_KEY as _TYPE_KEY,
    ImageData,
    build_image,
    check_readable,
    grid_rows,
    is_identity,
    lattice_mesh,
    vtk_lattice,
)
from polyxios._io import Source, source_name, source_size
from polyxios._tags import tags_from_masks, with_tag_masks
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.codecs._hdf5 import (
    _h5py,
    as_array,
    attr_text,
    child,
    dataset_options,
    open_hdf5_read,
    open_hdf5_write,
    refuse_lazy,
    require_h5py,
    text_of,
    warn_unknown_opts,
)
from polyxios.codecs._vtk_xml import (
    extent_points,
    polygon_types,
    structured_cell_shape,
    vtk_code_types,
)
from polyxios.exceptions import CodecError, UnsupportedFormatError
from polyxios.validate import validate_header

EXTENSION: str = ".vtkhdf"
LABEL: str = "VTKHDF"

_ROOT: str = "VTKHDF"
# The version written. VTK reads any 2.x, and 2.0 is the one that defined
# every layout used here: PolyData, partitions and the Steps group.
_VERSION: tuple[int, int] = (2, 0)

_UNSTRUCTURED: str = "UnstructuredGrid"
_POLYDATA: str = "PolyData"
_IMAGE: str = "ImageData"
_READ_TYPES: tuple[str, ...] = (_UNSTRUCTURED, _POLYDATA, _IMAGE)

# The cell groups of a PolyData, in the order VTK numbers their cells, which
# is the order a CellData array runs over them.
_SECTIONS: tuple[str, ...] = ("Vertices", "Lines", "Polygons", "Strips")
_SECTION_OF: dict[int, str] = {
    ELEMENT_TYPES["vertex"]: "Vertices",
    ELEMENT_TYPES["poly_vertex"]: "Vertices",
    ELEMENT_TYPES["line"]: "Lines",
    ELEMENT_TYPES["poly_line"]: "Lines",
    ELEMENT_TYPES["triangle"]: "Polygons",
    ELEMENT_TYPES["quad"]: "Polygons",
    ELEMENT_TYPES["polygon"]: "Polygons",
    ELEMENT_TYPES["triangle_strip"]: "Strips",
}

_TIME_KEY: str = "time"
_RESERVED_GLOBALS: frozenset[str] = frozenset(
    {_TYPE_KEY, _TIME_KEY, _DIRECTION_KEY, "vti_origin", "vti_spacing", "vti_extent"}
)
_IDENTITY: np.ndarray = np.eye(3).ravel()


# =============================================================================
# Reading
# =============================================================================


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a VTKHDF file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.vtkhdf`` file, or an open binary file object.
    lazy
        Not supported: the arrays live in HDF5 datasets that are decoded
        whole, and ``True`` raises.
    **opts
        ``step`` picks the time step of a temporal file, counted from zero
        and negative from the end the way a list is; the first is read
        without it. Every partition at that step is merged into the one
        mesh, each part's elements tagged ``part_<k>`` when there are
        several.

    Returns
    -------
    PolyData
        The mesh. An ``UnstructuredGrid`` reads as its cells are, a
        ``PolyData`` with its vertices, lines, polygons and strips typed by
        their point count, an ``ImageData`` expanded into the hexahedra,
        quadrilaterals or lines its extent spans, the way ``.vti`` is, its
        arrays taken in the lattice's shape whether or not the file left
        out an axis of size one. ``PointData`` and ``CellData`` are
        ``vertex_attrs`` and ``element_attrs``, ``FieldData`` is
        ``global_attrs``; the VTK type
        the file declared is ``global_attrs["vtkhdf_type"]`` and a temporal
        file's step time ``global_attrs["time"]``.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    CodecError
        If the file is not HDF5, holds no ``/VTKHDF`` group, declares a
        version other than 1.x or 2.x, names a count its datasets do not
        hold, or ``step`` is not a whole number or names a step the file
        does not have. A 1.x file is read as leniently as a 2.x one: the
        layouts 2.0 defined - ``PolyData``, several parts, ``Steps`` - are
        taken as written when a 1.x file holds them, not refused.
    ValidationError
        If the file declares more points, cells or connectivity ids than
        any mesh may hold, before anything is allocated for them.
    UnsupportedFormatError
        If h5py is not installed, or the file's ``Type`` is one polyxios
        does not read - a multi-block or AMR file.
    """
    step = opts.pop("step", None)
    warn_unknown_opts(opts, fmt=EXTENSION, what="read")
    name = source_name(path)
    refuse_lazy(lazy, fmt=EXTENSION, name=name)
    size = source_size(path)
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        root, kind, n_steps = _root(handle, name)
        index = _step_index(step, n_steps, name, absent=_no_step(root))
        return _read_step(root, kind, index, name=name, size=size)


def read_image(path: Source, **opts: Any) -> ImageData:
    """Read the ImageData a VTKHDF file holds, without expanding it.

    Parameters
    ----------
    path
        Path to the ``.vtkhdf`` file, or an open binary file object.
    **opts
        ``step`` picks the time step of a temporal file, as for
        :func:`read`.

    Returns
    -------
    ImageData
        The grid with its ``PointData`` and ``CellData`` laid out along it,
        its ``Direction`` when the file rotates it, and ``FieldData`` as
        ``global_attrs``, plus ``time`` for a temporal file's step.

    Raises
    ------
    UnsupportedFormatError
        If the file's ``Type`` is not ``ImageData``; read it with
        :func:`polyxios.read`.
    CodecError
        As for :func:`read`.
    """
    step = opts.pop("step", None)
    warn_unknown_opts(opts, fmt=EXTENSION, what="read", stacklevel=4)
    name = source_name(path)
    size = source_size(path)
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        root, kind, n_steps = _root(handle, name)
        index = _step_index(step, n_steps, name, absent=_no_step(root))
        if kind != _IMAGE:
            raise UnsupportedFormatError(
                f"'{name}': VTKHDF type '{kind}' holds no image; "
                "read it with polyxios.read()."
            )
        parts, _ = _read_image_grid(
            root, index, name=name, size=size, builds_vertices=False
        )
        if index is not None:
            parts["global_attrs"][_TIME_KEY] = float(root["Steps"]["Values"][index])
        return build_image(name, **parts)


def read_time_series(path: Source, **opts: Any) -> list[PolyData]:
    """Read every step of a temporal file, one PolyData each.

    Parameters
    ----------
    path
        Path to the ``.vtkhdf`` file, or an open binary file object.
    **opts
        None are taken; any given is warned about and ignored.

    Returns
    -------
    list of PolyData
        One mesh per step, in the file's order, each carrying its time under
        ``global_attrs["time"]``. A file with no ``Steps`` group is one step.
    """
    warn_unknown_opts(opts, fmt=EXTENSION, what="read")
    name = source_name(path)
    size = source_size(path)
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        root, kind, n_steps = _root(handle, name)
        if n_steps == 0:
            return [_read_step(root, kind, None, name=name, size=size)]
        return [_read_step(root, kind, k, name=name, size=size) for k in range(n_steps)]


def _root(handle: Any, name: str) -> tuple[Any, str, int]:
    """The ``/VTKHDF`` group, its type and how many steps it holds."""
    if _ROOT not in handle:
        raise CodecError(
            f"'{name}': the file holds no '/VTKHDF' group, so is not VTKHDF."
        )
    root = handle[_ROOT]
    if "Version" not in root.attrs:
        raise CodecError(f"'{name}': '/VTKHDF' carries no 'Version' attribute.")
    version = np.atleast_1d(np.asarray(root.attrs["Version"])).ravel()
    if (
        version.size == 0
        or version.dtype.kind not in "iu"
        or int(version[0]) not in (1, 2)
    ):
        raise CodecError(
            f"'{name}': VTKHDF version {version.tolist()} is not one this reader"
            " knows; versions 1.x and 2.x are."
        )
    kind = attr_text(root, "Type")
    if kind is None:
        raise CodecError(f"'{name}': '/VTKHDF' carries no 'Type' attribute.")
    if kind not in _READ_TYPES:
        raise UnsupportedFormatError(
            f"'{name}': VTKHDF type '{kind}' is not read; only"
            f" {', '.join(_READ_TYPES)} are."
        )
    n_steps = 0
    if "Steps" in root:
        steps = root["Steps"]
        where = "/VTKHDF/Steps/Values"
        values = as_array(
            child(steps, "Values", name=name, where="/VTKHDF/Steps"),
            name=name,
            where=where,
        )
        if values.ndim != 1 or values.dtype.kind not in "biuf":
            raise CodecError(
                f"'{name}': '{where}' should be one number per step, and is"
                f" shaped {values.shape} ({values.dtype})."
            )
        n_steps = _n_steps(steps, len(values), name=name)
        if n_steps < 0 or n_steps > len(values):
            raise CodecError(
                f"'{name}': Steps declares {n_steps} steps and holds"
                f" {len(values)} time values."
            )
    return root, kind, n_steps


def _no_step(root: Any) -> str:
    """Why a file has no step to pick: no Steps group, or one that counts none."""
    return "holds 0 steps" if "Steps" in root else "holds no Steps group"


def _n_steps(steps: Any, default: int, *, name: str) -> int:
    """``Steps/NSteps`` as an int, a scalar or one-element array alike."""
    if "NSteps" not in steps.attrs:
        return default
    held = np.atleast_1d(np.asarray(steps.attrs["NSteps"])).ravel()
    if held.size != 1 or held.dtype.kind not in "iu":
        raise CodecError(
            f"'{name}': Steps/NSteps should be one whole number, and holds"
            f" {held.tolist()}."
        )
    return int(held[0])


def _step_index(step: Any, n_steps: int, name: str, *, absent: str) -> int | None:
    """Turn ``step=`` into an index into the Steps group, or None without one.

    ``absent`` says why the file has no step when ``step=`` asks for one.
    """
    if step is None:
        return 0 if n_steps else None
    index = _whole_step(step, name)
    if n_steps == 0:
        raise CodecError(f"'{name}': step={step} was asked for, and the file {absent}.")
    if index < 0:
        index += n_steps
    if not 0 <= index < n_steps:
        raise CodecError(
            f"'{name}': step={step} is out of range; the file holds {n_steps} step(s)."
        )
    return index


def _whole_step(step: Any, name: str) -> int:
    """``step=`` as an index, refusing a value that is not a whole number."""
    try:
        return operator.index(step)
    except TypeError:
        raise CodecError(
            f"'{name}': step={step!r} is not a whole number; a step is counted"
            " from zero, and negative from the end."
        ) from None


def _read_step(
    root: Any, kind: str, index: int | None, *, name: str, size: int
) -> PolyData:
    if kind == _IMAGE:
        parts, grid = _read_image_grid(
            root, index, name=name, size=size, builds_vertices=True
        )
        poly = lattice_mesh(**parts)
        poly.global_attrs.update(grid)
    elif kind == _POLYDATA:
        poly = _read_polydata(root, index, name=name, size=size)
    else:
        poly = _read_unstructured(root, index, name=name, size=size)
    poly.global_attrs[_TYPE_KEY] = kind
    if index is not None:
        poly.global_attrs[_TIME_KEY] = float(root["Steps"]["Values"][index])
    return poly


# ----- the datasets ------------------------------------------------------------


def _counts(group: Any, key: str, *, name: str, where: str) -> np.ndarray:
    """A per-part count dataset, as int64."""
    values = as_array(child(group, key, name=name, where=where), name=name, where=where)
    if values.dtype.kind not in "iu":
        raise CodecError(f"'{name}': '{where}/{key}' holds no whole numbers.")
    values = values.reshape(-1).astype(np.int64)
    if values.size and values.min() < 0:
        raise CodecError(f"'{name}': '{where}/{key}' holds a negative count.")
    return values


def _rows(
    node: Any, start: int, count: int, *, name: str, where: str, what: str
) -> np.ndarray:
    """Rows ``start`` to ``start + count`` of a dataset, native byte order.

    Only the rows asked for are read, so a temporal file's other steps stay
    on disk; a count the dataset does not hold is refused naming it.
    """
    if not hasattr(node, "shape"):
        raise CodecError(f"'{name}': '{where}' is a group, not a dataset.")
    if node.ndim == 0:
        raise CodecError(f"'{name}': '{where}' is a scalar, not an array of {what}.")
    held = int(node.shape[0])
    if start < 0 or count < 0 or start + count > held:
        raise CodecError(
            f"'{name}': '{where}' holds {held} {what}, and the file asks for"
            f" {count} from {start}."
        )
    values = np.asarray(node[start : start + count])
    if values.dtype.kind in "biufc":
        return values.astype(values.dtype.newbyteorder("="), copy=False)
    return values


def _points(root: Any, start: int, count: int, *, name: str) -> np.ndarray:
    pts = _rows(
        child(root, "Points", name=name, where="/VTKHDF"),
        start,
        count,
        name=name,
        where="/VTKHDF/Points",
        what="points",
    )
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.dtype.kind not in "iuf":
        raise CodecError(
            f"'{name}': '/VTKHDF/Points' is shaped {pts.shape} ({pts.dtype}),"
            " not (n, 3) coordinates."
        )
    return np.ascontiguousarray(pts[:, :3], dtype=np.float64)


def _window(
    root: Any, index: int | None, n_parts_total: int, *, name: str, sections: int
) -> tuple[range, int, np.ndarray, np.ndarray]:
    """Which parts a step holds, and where its points, cells and ids start."""
    if index is None:
        zeros = np.zeros(sections, dtype=np.int64)
        return range(n_parts_total), 0, zeros, zeros
    steps = root["Steps"]
    where = "/VTKHDF/Steps"

    def at(key: str, default: int, width: int) -> np.ndarray:
        if key not in steps:
            return np.full(width, default, dtype=np.int64)
        values = as_array(steps[key], name=name, where=f"{where}/{key}")
        if values.ndim == 0 or index >= len(values):
            raise CodecError(
                f"'{name}': '{where}/{key}' holds no entry for step {index}."
            )
        row = np.atleast_1d(values[index]).astype(np.int64).ravel()
        if row.size != width:
            raise CodecError(
                f"'{name}': '{where}/{key}' holds {row.size} value(s) per step"
                f" where {width} belong."
            )
        return row

    part0 = int(at("PartOffsets", 0, 1)[0])
    n_parts = int(at("NumberOfParts", 1, 1)[0])
    if part0 < 0 or n_parts < 0 or part0 + n_parts > n_parts_total:
        raise CodecError(
            f"'{name}': step {index} names parts {part0} to {part0 + n_parts}"
            f" and the file holds {n_parts_total}."
        )
    point0 = int(at("PointOffsets", 0, 1)[0])
    return (
        range(part0, part0 + n_parts),
        point0,
        at("CellOffsets", 0, sections),
        at("ConnectivityIdOffsets", 0, sections),
    )


def _section_cells(
    group: Any,
    parts: range,
    cell0: int,
    conn0: int,
    n_cells: np.ndarray,
    n_ids: np.ndarray,
    *,
    name: str,
    where: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """One cell group's connectivity and cell ends, one entry per part.

    Returns the connectivity of each part indexing the part's own points,
    the ends of each part's cells relative to its own connectivity, and the
    number of cells per part, so the caller can place them in the joined
    mesh.
    """
    conns: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    counts: list[int] = []
    cell_cursor = cell0
    conn_cursor = conn0
    for g in parts:
        nc = int(n_cells[g])
        ni = int(n_ids[g])
        offsets = _rows(
            child(group, "Offsets", name=name, where=where),
            cell_cursor + g,
            nc + 1 if nc else 0,
            name=name,
            where=f"{where}/Offsets",
            what="offsets",
        ).astype(np.int64)
        conn = _rows(
            child(group, "Connectivity", name=name, where=where),
            conn_cursor,
            ni,
            name=name,
            where=f"{where}/Connectivity",
            what="connectivity ids",
        )
        if conn.dtype.kind not in "iu":
            raise CodecError(f"'{name}': '{where}/Connectivity' holds no indices.")
        if nc:
            if offsets[0] < 0 or np.any(offsets[1:] < offsets[:-1]):
                raise CodecError(f"'{name}': '{where}/Offsets' run backwards.")
            if offsets[-1] > ni:
                raise CodecError(
                    f"'{name}': '{where}/Offsets' reach {int(offsets[-1])}"
                    f" connectivity ids and the part holds {ni}."
                )
            conn = conn[int(offsets[0]) : int(offsets[-1])]
            offsets = offsets[1:] - offsets[0]
        else:
            conn = conn[:0]
            offsets = offsets[:0]
        conns.append(conn.astype(np.int64, copy=False))
        ends.append(offsets)
        counts.append(nc)
        cell_cursor += nc
        conn_cursor += ni
    return conns, ends, counts


def _assemble(
    vertices: np.ndarray,
    parts: list[tuple[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Join the parts' sections into one CSR mesh.

    Parameters
    ----------
    vertices
        Every part's points, laid end to end.
    parts
        Per part: how many points it holds, and its sections as
        ``(connectivity, ends, types)`` triples in cell order.
    name
        The file, for the message.

    Returns
    -------
    tuple
        The connectivity, offsets and element types, and one ``part_<k>``
        tag per part when there are several.
    """
    conn_blocks: list[np.ndarray] = []
    end_blocks: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    type_blocks: list[np.ndarray] = []
    tags: dict[str, np.ndarray] = {}
    vert_offset = 0
    conn_offset = 0
    cell_offset = 0
    for k, (n_points, sections) in enumerate(parts):
        n_cells_here = 0
        for conn, ends, types in sections:
            if conn.size and (conn.min() < 0 or conn.max() >= n_points):
                raise CodecError(
                    f"'{name}': part {k} names point {int(conn.max())} and"
                    f" holds {n_points} points."
                )
            conn_blocks.append(conn + vert_offset)
            end_blocks.append(ends + conn_offset)
            type_blocks.append(types)
            conn_offset += conn.size
            n_cells_here += types.size
        if len(parts) > 1:
            tags[f"part_{k}"] = np.arange(
                cell_offset, cell_offset + n_cells_here, dtype=np.int32
            )
        cell_offset += n_cells_here
        vert_offset += n_points
    connectivity = (
        np.concatenate(conn_blocks) if conn_blocks else np.zeros(0, dtype=np.int64)
    )
    offsets = np.concatenate(end_blocks)
    types = np.concatenate(type_blocks) if type_blocks else np.zeros(0, dtype=np.uint8)
    idx = np.int64 if offsets[-1] >= 2**31 or len(vertices) >= 2**31 else np.int32
    return connectivity.astype(idx), offsets.astype(idx), types, tags


def _read_unstructured(
    root: Any, index: int | None, *, name: str, size: int
) -> PolyData:
    where = "/VTKHDF"
    n_points_all = _counts(root, "NumberOfPoints", name=name, where=where)
    n_cells_all = _counts(root, "NumberOfCells", name=name, where=where)
    n_ids_all = _counts(root, "NumberOfConnectivityIds", name=name, where=where)
    n_parts_total = len(n_points_all)
    if len(n_cells_all) < n_parts_total or len(n_ids_all) < n_parts_total:
        raise CodecError(
            f"'{name}': NumberOfPoints names {n_parts_total} part(s) and"
            " NumberOfCells or NumberOfConnectivityIds fewer."
        )
    parts, point0, cell0, conn0 = _window(
        root, index, n_parts_total, name=name, sections=1
    )
    n_points = int(n_points_all[parts.start : parts.stop].sum())
    n_cells = int(n_cells_all[parts.start : parts.stop].sum())
    n_ids = int(n_ids_all[parts.start : parts.stop].sum())
    validate_header(n_points, n_cells, n_ids, size, compressed=True)
    vertices = _points(root, point0, n_points, name=name)
    conns, ends, counts = _section_cells(
        root,
        parts,
        int(cell0[0]),
        int(conn0[0]),
        n_cells_all,
        n_ids_all,
        name=name,
        where=where,
    )
    types_all = _rows(
        child(root, "Types", name=name, where=where),
        int(cell0[0]),
        n_cells,
        name=name,
        where=f"{where}/Types",
        what="cell types",
    )
    if types_all.dtype.kind not in "iu":
        raise CodecError(f"'{name}': '{where}/Types' holds no cell type codes.")
    assembled: list[tuple[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
    cursor = 0
    for k, g in enumerate(parts):
        types = vtk_code_types(types_all[cursor : cursor + counts[k]])
        cursor += counts[k]
        assembled.append((int(n_points_all[g]), [(conns[k], ends[k], types)]))
    connectivity, offsets, element_types, tags = _assemble(
        vertices, assembled, name=name
    )
    return _finish(
        root, index, vertices, connectivity, offsets, element_types, tags, name=name
    )


def _read_polydata(root: Any, index: int | None, *, name: str, size: int) -> PolyData:
    where = "/VTKHDF"
    n_points_all = _counts(root, "NumberOfPoints", name=name, where=where)
    n_parts_total = len(n_points_all)
    parts, point0, cell0, conn0 = _window(
        root, index, n_parts_total, name=name, sections=len(_SECTIONS)
    )
    n_points = int(n_points_all[parts.start : parts.stop].sum())
    per_section: list[tuple[np.ndarray, np.ndarray]] = []
    n_cells = 0
    n_ids = 0
    for section in _SECTIONS:
        if section not in root:
            empty = np.zeros(n_parts_total, dtype=np.int64)
            per_section.append((empty, empty))
            continue
        group = root[section]
        cells = _counts(group, "NumberOfCells", name=name, where=f"{where}/{section}")
        ids = _counts(
            group, "NumberOfConnectivityIds", name=name, where=f"{where}/{section}"
        )
        if len(cells) < n_parts_total or len(ids) < n_parts_total:
            raise CodecError(
                f"'{name}': '{where}/{section}' counts fewer parts than"
                f" NumberOfPoints names ({n_parts_total})."
            )
        per_section.append((cells, ids))
        n_cells += int(cells[parts.start : parts.stop].sum())
        n_ids += int(ids[parts.start : parts.stop].sum())
    validate_header(n_points, n_cells, n_ids, size, compressed=True)
    vertices = _points(root, point0, n_points, name=name)
    # Read section by section - each is one run of the file - then regroup
    # part by part, since a part's cells run vertices, lines, polygons,
    # strips and its CellData with them.
    by_section: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    for s, section in enumerate(_SECTIONS):
        cells, ids = per_section[s]
        if section not in root:
            by_section.append(
                (
                    [np.zeros(0, dtype=np.int64) for _ in parts],
                    [np.zeros(0, dtype=np.int64) for _ in parts],
                )
            )
            continue
        conns, ends, _ = _section_cells(
            root[section],
            parts,
            int(cell0[s]),
            int(conn0[s]),
            cells,
            ids,
            name=name,
            where=f"{where}/{section}",
        )
        by_section.append((conns, ends))
    assembled: list[tuple[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
    for k, g in enumerate(parts):
        sections: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for s, section in enumerate(_SECTIONS):
            conn, ends = by_section[s][0][k], by_section[s][1][k]
            sections.append((conn, ends, _section_types(section, ends)))
        assembled.append((int(n_points_all[g]), sections))
    connectivity, offsets, element_types, tags = _assemble(
        vertices, assembled, name=name
    )
    return _finish(
        root, index, vertices, connectivity, offsets, element_types, tags, name=name
    )


def _section_types(section: str, ends: np.ndarray) -> np.ndarray:
    """Element codes for one PolyData cell group, by point count."""
    widths = np.diff(ends, prepend=0)
    if section == "Vertices":
        out = np.full(widths.shape, ELEMENT_TYPES["poly_vertex"], dtype=np.uint8)
        out[widths == 1] = ELEMENT_TYPES["vertex"]
        return out
    if section == "Lines":
        out = np.full(widths.shape, ELEMENT_TYPES["poly_line"], dtype=np.uint8)
        out[widths == 2] = ELEMENT_TYPES["line"]
        return out
    if section == "Polygons":
        return polygon_types(widths)
    return np.full(widths.shape, ELEMENT_TYPES["triangle_strip"], dtype=np.uint8)


def _read_image_grid(
    root: Any,
    index: int | None,
    *,
    name: str,
    size: int,
    builds_vertices: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read an ImageData's parts, and the keys the mesh reader spells it with."""
    extent = (
        _attr_numbers(root, "WholeExtent", 6, name=name)
        if builds_vertices
        else _attr_integers(root, "WholeExtent", 6, name=name)
    )
    if extent is None:
        raise CodecError(f"'{name}': an ImageData needs a 'WholeExtent' attribute.")
    if not all(math.isfinite(v) and v == int(v) for v in extent):
        raise CodecError(
            f"'{name}': WholeExtent should be six whole numbers, and holds {extent}."
        )
    extent = [int(v) for v in extent]
    origin = _attr_numbers(root, "Origin", 3, name=name) or [0.0, 0.0, 0.0]
    spacing = _attr_numbers(root, "Spacing", 3, name=name) or [1.0, 1.0, 1.0]
    direction = _attr_numbers(root, "Direction", 9, name=name) or list(_IDENTITY)
    i0, i1, j0, j1, k0, k1 = extent
    spans = (i1 - i0, j1 - j0, k1 - k0)
    if min(spans) < 0:
        raise CodecError(f"'{name}': WholeExtent {extent} runs backwards.")
    nx, ny, nz = spans
    n_verts = extent_points(spans)
    n_cells, n_per_cell, cell_kind = structured_cell_shape(nx, ny, nz)
    validate_header(
        n_verts,
        n_cells,
        n_cells * n_per_cell,
        size,
        compressed=True,
        spells_vertices=False,
        spells_connectivity=False,
        builds_vertices=builds_vertices,
    )
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    rotated = not is_identity(matrix)
    point_shape = (nz + 1, ny + 1, nx + 1)
    cell_shape = (max(nz, 1), max(ny, 1), max(nx, 1))
    vertex_attrs = _image_arrays(
        root, "PointData", index, point_shape, point_shape, n_verts, name=name
    )
    element_attrs = _image_arrays(
        root, "CellData", index, cell_shape, point_shape, n_cells, name=name
    )
    parts: dict[str, Any] = {
        "extent": tuple(extent),
        "origin": tuple(float(v) for v in origin),
        "spacing": tuple(float(v) for v in spacing),
        "direction": matrix if rotated else None,
        "point_attrs": {
            k: v.reshape(point_shape + v.shape[1:]) for k, v in vertex_attrs.items()
        },
        "cell_attrs": {
            k: v.reshape(cell_shape + v.shape[1:]) for k, v in element_attrs.items()
        },
        "global_attrs": _read_field_data(root, index, name=name),
    }
    grid: dict[str, Any] = {
        "vti_origin": [float(v) for v in origin],
        "vti_spacing": [float(v) for v in spacing],
        "vti_extent": extent,
    }
    if rotated:
        grid[_DIRECTION_KEY] = matrix
    return parts, grid


def _attr_integers(root: Any, key: str, count: int, *, name: str) -> list[float] | None:
    """An attribute that holds integers, as the integers it holds; anything else goes
    through :func:`_attr_numbers`, which takes or refuses it. An int64 is kept exact,
    where a float would round a value past 2**53."""
    if key not in root.attrs:
        return None
    values = np.atleast_1d(np.asarray(root.attrs[key])).ravel()
    if values.dtype.kind not in "iu" or values.size != count:
        return _attr_numbers(root, key, count, name=name)
    return [int(v) for v in values]


def _attr_numbers(root: Any, key: str, count: int, *, name: str) -> list[float] | None:
    if key not in root.attrs:
        return None
    values = np.atleast_1d(np.asarray(root.attrs[key])).ravel()
    if values.dtype.kind not in "biuf" or values.size != count:
        raise CodecError(
            f"'{name}': '/VTKHDF' attribute '{key}' should be {count} numbers,"
            f" and holds {values.tolist()}."
        )
    return [float(v) for v in values]


def _image_arrays(
    root: Any,
    group_name: str,
    index: int | None,
    shape: tuple[int, int, int],
    points: tuple[int, int, int],
    n_rows: int,
    *,
    name: str,
) -> dict[str, np.ndarray]:
    """The arrays of an ImageData's PointData or CellData, one row per entity.

    The file may spell each array's lattice three ways: in full, ``shape``, slowest
    axis first; in VTK's own layout, which leaves out each axis that holds one point
    (``points`` is the points' lattice, see :func:`~polyxios._image.vtk_lattice`);
    or with every axis of one left out, the form earlier releases read. All three are
    read. A single point spans no cell, so its cell arrays are looked for in the first
    and last spellings only, the two earlier releases read. In a temporal file an array
    with an entry under ``Steps/<group>Offsets`` adds a leading step axis the offset
    counts along; one with no entry is kept once, and every step reads it whole.
    """
    if group_name not in root:
        return {}
    group = root[group_name]
    offsets = _offsets_group(root, index, group_name)
    squeezed = tuple(d for d in shape if d > 1)
    forms = (
        (shape, squeezed)
        if n_rows == 0
        else (shape, vtk_lattice(shape, points), squeezed)
    )
    out: dict[str, np.ndarray] = {}
    for key in group:
        node = group[key]
        if not hasattr(node, "shape"):
            continue
        if node.dtype.kind not in "biuf":
            _warn_not_numeric(key, node.dtype, group_name)
            continue
        where = f"/VTKHDF/{group_name}/{key}"
        stepped = offsets is not None and key in offsets
        held = tuple(node.shape[1:]) if stepped else tuple(node.shape)
        if (stepped and node.ndim == 0) or not _spells_lattice(held, forms):
            _warn_shape(
                key,
                node.shape,
                shape,
                group_name,
                stepped=stepped,
            )
            continue
        if n_rows == 0:
            # An extent of one point spans no cell, and a (1, 1, 1) array
            # still spells its lattice: there is no row to attach it to.
            _warn_no_rows(key, node.shape, group_name)
            continue
        if stepped:
            start = _offset_of(offsets, key, index, name=name)
            values = _rows(node, start, 1, name=name, where=where, what="steps")[0]
        else:
            values = np.asarray(node[()])
        values = values.astype(values.dtype.newbyteorder("="), copy=False)
        values = values.reshape(n_rows, -1)
        out[key] = values[:, 0] if values.shape[1] == 1 else values
    return out


def _spells_lattice(held: tuple[int, ...], forms: tuple[tuple[int, ...], ...]) -> bool:
    """Whether a dataset's axes are one of the lattice's ``forms``, with one component
    axis or none."""
    for wanted in forms:
        n = len(wanted)
        if held[:n] == wanted and len(held) - n <= 1:
            return True
    return False


def _warn_shape(
    key: str,
    held: tuple,
    wanted: tuple,
    group_name: str,
    *,
    stepped: bool,
) -> None:
    asked = f"a step axis before {wanted}" if stepped else f"{wanted}"
    warn_caller(
        f"{EXTENSION}: {group_name} array '{key}' is shaped {tuple(held)} where"
        f" the extent asks for {asked}; dropped."
    )


def _warn_no_rows(key: str, held: tuple, group_name: str) -> None:
    warn_caller(
        f"{EXTENSION}: {group_name} array '{key}' is shaped {tuple(held)} and the"
        " extent spans no entity to hold it; dropped."
    )


def _warn_scalar(key: str, group_name: str, *, stacklevel: int) -> None:
    warnings.warn(
        f"{EXTENSION}: {group_name} array '{key}' is a scalar, not one row per"
        " entity; dropped.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _warn_not_numeric(
    key: str, dtype: Any, group_name: str, *, stacklevel: int | None = None
) -> None:
    """Name an array that holds a non-numeric type.

    ``stacklevel`` is None on the image path, which blames the caller of the
    reader; the mesh reader passes the frame it always has.
    """
    message = (
        f"{EXTENSION}: {group_name} array '{key}' holds {dtype}, which is not"
        " numeric; dropped."
    )
    if stacklevel is None:
        warn_caller(message)
    else:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _offsets_group(root: Any, index: int | None, group_name: str) -> Any | None:
    if index is None or "Steps" not in root:
        return None
    steps = root["Steps"]
    key = f"{group_name}Offsets"
    return steps[key] if key in steps else None


def _offset_of(offsets: Any | None, key: str, index: int, *, name: str) -> int:
    """Where one array's step begins, zero for an array the file keeps once."""
    if offsets is None or key not in offsets:
        return 0
    node = offsets[key]
    values = as_array(node, name=name, where=f"/VTKHDF/Steps/{key}")
    if values.ndim == 0 or index >= len(values):
        raise CodecError(
            f"'{name}': offsets of '{key}' hold no entry for step {index}."
        )
    return int(values[index])


def _read_arrays(
    root: Any, group_name: str, index: int | None, count: int, *, name: str
) -> dict[str, np.ndarray]:
    """The arrays of a PointData or CellData group, ``count`` rows each."""
    if group_name not in root:
        return {}
    group = root[group_name]
    offsets = _offsets_group(root, index, group_name)
    out: dict[str, np.ndarray] = {}
    for key in group:
        node = group[key]
        if not hasattr(node, "shape"):
            continue
        if node.dtype.kind not in "biuf":
            _warn_not_numeric(key, node.dtype, group_name, stacklevel=5)
            continue
        if node.ndim == 0:
            _warn_scalar(key, group_name, stacklevel=5)
            continue
        start = 0 if index is None else _offset_of(offsets, key, index, name=name)
        held = int(node.shape[0])
        if start < 0 or start + count > held:
            # The same answer the VTK XML readers give an array of the wrong
            # length: it belongs to no point or cell of this mesh, so it is
            # named and left rather than attached to the wrong ones.
            warnings.warn(
                f"{EXTENSION}: {group_name} array '{key}' holds {held} rows,"
                f" and the mesh asks for {count} from {start}; dropped.",
                UserWarning,
                stacklevel=4,
            )
            continue
        where = f"/VTKHDF/{group_name}/{key}"
        out[key] = _rows(node, start, count, name=name, where=where, what="rows")
    return out


def _read_field_data(root: Any, index: int | None, *, name: str) -> dict[str, Any]:
    if "FieldData" not in root:
        return {}
    group = root["FieldData"]
    steps = root["Steps"] if index is not None and "Steps" in root else None
    out: dict[str, Any] = {}
    for key in group:
        node = group[key]
        if not hasattr(node, "shape"):
            continue
        if steps is None or node.ndim == 0:
            values = np.asarray(node[()])
        else:
            start = _offset_of(
                steps["FieldDataOffsets"] if "FieldDataOffsets" in steps else None,
                key,
                index,
                name=name,
            )
            values = _rows(
                node,
                start,
                _field_tuples(steps, key, index, node, start, name=name),
                name=name,
                where=f"/VTKHDF/FieldData/{key}",
                what="tuples",
            )
        held = _global_value(values)
        if held is not None:
            out[key] = held
    return out


def _field_tuples(
    steps: Any, key: str, index: int, node: Any, start: int, *, name: str
) -> int:
    """How many tuples of a FieldData array belong to one step.

    ``FieldDataSizes/<name>`` spells ``[components, tuples]`` per step; an
    array with no entry there is taken whole from its offset.
    """
    if "FieldDataSizes" not in steps or key not in steps["FieldDataSizes"]:
        return max(int(node.shape[0]) - start, 0)
    sizes = as_array(
        steps["FieldDataSizes"][key],
        name=name,
        where=f"/VTKHDF/Steps/FieldDataSizes/{key}",
    )
    if sizes.ndim == 0 or index >= len(sizes):
        raise CodecError(
            f"'{name}': FieldDataSizes of '{key}' hold no entry for step {index}."
        )
    row = np.atleast_1d(sizes[index]).ravel()
    if row.size == 0:
        raise CodecError(
            f"'{name}': FieldDataSizes of '{key}' hold no entry for step {index}."
        )
    return int(row[-1])


def _global_value(values: np.ndarray) -> Any:
    """One FieldData array as the value ``global_attrs`` keeps for it."""
    if values.dtype.kind in "biuf":
        values = values.astype(values.dtype.newbyteorder("="), copy=False)
        return values.reshape(-1)[0] if values.size == 1 else values
    texts = [
        t for t in (text_of(v) for v in np.atleast_1d(values).ravel()) if t is not None
    ]
    if not texts:
        return None
    return texts[0] if len(texts) == 1 else texts


def _finish(
    root: Any,
    index: int | None,
    vertices: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    element_types: np.ndarray,
    tags: dict[str, np.ndarray],
    *,
    name: str,
) -> PolyData:
    vertex_attrs = _read_arrays(root, "PointData", index, len(vertices), name=name)
    element_attrs = _read_arrays(root, "CellData", index, len(element_types), name=name)
    vertex_attrs, vertex_tags = tags_from_masks(vertex_attrs)
    element_attrs, element_tags = tags_from_masks(element_attrs)
    for tag, members in tags.items():
        if tag in element_tags:
            warnings.warn(
                f"{EXTENSION}: CellData names a tag '{tag}', which the partition"
                " tag of the same name replaces.",
                UserWarning,
                stacklevel=4,
            )
        element_tags[tag] = members
    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=_read_field_data(root, index, name=name),
    )


# =============================================================================
# Writing
# =============================================================================


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as a VTKHDF file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object.
    **opts
        ``polydata=True`` writes a VTK ``PolyData`` - vertices, lines,
        polygons and strips - and ``False`` an ``UnstructuredGrid``; without
        it the type the mesh was read from decides, through
        ``global_attrs["vtkhdf_type"]``, and an ``UnstructuredGrid`` is
        written otherwise. ``time`` writes a ``Steps`` group of one step at
        that time; without it, a number under ``global_attrs["time"]`` does.
        Either way ``global_attrs["time"]`` is then the step's time and
        travels through ``Steps``, not ``FieldData``: a text held there is
        dropped with a warning, since the step's time reads back over it.
        Without a time, a text ``time`` is field data like any other.
        ``compression`` and ``compression_opts`` go to h5py for every
        dataset.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If a ``PolyData`` is asked for and the mesh holds a cell it cannot -
        a tetrahedron, a hexahedron - or the mesh's vertices are not an
        ``(n, 3)`` block.

    Warns
    -----
    UserWarning
        For an element type VTK has no code for, an attribute that is not
        numeric, a ``global_attrs`` value that is neither a number nor
        text, and a text ``global_attrs["time"]`` when a time is written:
        each is dropped by name.
    """
    require_h5py(fmt=EXTENSION, name=source_name(path), verb="writing")
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    time = opts.pop("time", None)
    polydata = opts.pop("polydata", None)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write")
    name = source_name(path)
    kind = _write_kind(poly, polydata, name=name)
    time = _held_time(poly) if time is None else _time_value(time, name=name, index=0)
    arrays = _step_arrays(poly, timed=time is not None, stacklevel=4)
    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        root = _new_root(handle, kind)
        # One step appends nothing, so its datasets are laid out whole; the
        # Steps group records row 0 of each.
        store = _Store(root, dataset_kw, growing=False)
        record = _write_body(store, poly, kind, arrays, geometry=True)
        if time is not None:
            _write_steps(root, kind, [time], [record])


def write_image(image: ImageData, path: Source, **opts: Any) -> None:
    """Write an ImageData as a VTKHDF ``ImageData`` file.

    The grid is written as it stands: ``WholeExtent``, ``Origin``,
    ``Spacing`` and ``Direction`` attributes, with the point and cell arrays
    laid out along it in VTK's own layout: ``(nz, ny, nx[, k])`` with each axis that
    holds one point left out, so a flat image's points are ``(ny, nx[, k])`` and a
    single point's ``(1[, k])``. A volume costs its samples and no coordinates or
    cells.

    Parameters
    ----------
    image
        The grid to write.
    path
        Output file path, or an open binary file object.
    **opts
        ``compression`` and ``compression_opts`` go to h5py for every
        dataset. No other option is taken; any other is warned about and
        ignored.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If an axis holds no point, or an attribute is not laid out along the
        extent; a grid past the caps a reader holds a compact image to is a
        :class:`~polyxios.exceptions.ValidationError`, so no file is written that
        cannot be read back.

    Warns
    -----
    UserWarning
        For an array that is not numeric, a ``global_attrs`` value that is
        neither a number nor text, and a cell array of an extent that spans no
        cell: each is dropped by name. A ``vti_whole_extent`` is dropped too,
        with a warning, as a VTKHDF ImageData has no whole extent to spell.

    Notes
    -----
    One step is written, with no ``Steps`` group: a ``global_attrs["time"]``
    is field data like any other. The ``vti_*`` and ``vtkhdf_*`` keys a mesh
    read spells the grid with are never written as field data.
    """
    name = source_name(path)
    require_h5py(fmt=EXTENSION, name=name, verb="writing")
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write", stacklevel=4)
    nx, ny, nz = image.dimensions
    if min(nx, ny, nz) == 0:
        raise CodecError(
            f"{EXTENSION} write: the extent {tuple(image.extent)} holds no point,"
            " and an ImageData file cannot hold an empty grid."
        )
    check_readable(image)
    if "vti_whole_extent" in image.global_attrs:
        warn_caller(
            f"{EXTENSION} write: global 'vti_whole_extent' is dropped; a VTKHDF"
            " ImageData has no whole extent of its own, only the WholeExtent of"
            " its extent."
        )
    direction = np.eye(3) if image.direction is None else image.direction
    spans = image.spans
    n_cells = structured_cell_shape(*spans)[0]
    point = _image_data_arrays(
        image.point_attrs,
        (nz, ny, nx),
        nx * ny * nz,
        point_lead=(nz, ny, nx),
        group="PointData",
    )
    cell = _image_data_arrays(
        image.cell_attrs,
        tuple(max(s, 1) for s in (spans[2], spans[1], spans[0])),
        n_cells,
        point_lead=(nz, ny, nx),
        group="CellData",
    )
    held = _field_arrays(
        image.global_attrs, reserved=IMAGE_RESERVED_GLOBALS, stacklevel=4
    )
    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        root = _new_root(handle, _IMAGE)
        root.attrs["WholeExtent"] = np.array(image.extent, dtype=np.int64)
        root.attrs["Origin"] = np.array(image.origin, dtype=np.float64)
        root.attrs["Spacing"] = np.array(image.spacing, dtype=np.float64)
        root.attrs["Direction"] = direction.ravel()
        store = _Store(root, dataset_kw, growing=False)
        for key, arr in point.items():
            store.put(f"PointData/{key}", arr)
        for key, arr in cell.items():
            store.put(f"CellData/{key}", arr)
        for key, arr in held["field"].items():
            store.put(f"FieldData/{key}", arr)
        for key, lines in held["text"].items():
            store.put_text(f"FieldData/{key}", lines)


def _image_data_arrays(
    arrays: dict[str, np.ndarray],
    lead: tuple[int, ...],
    n_rows: int,
    *,
    point_lead: tuple[int, ...],
    group: str,
) -> dict[str, np.ndarray]:
    """The arrays an ImageData group can hold, each laid out along the grid.

    ``lead`` is the leading shape every array must have in memory, ``(nz, ny, nx)``
    for points; which arrays are kept, and the errors for the rest, are
    :func:`grid_rows`'s. The file holds each one in VTK's own shape, with the axes
    that hold one point left out (see :func:`~polyxios._image.vtk_lattice`), since
    VTK's reader cannot open a file that keeps them. Booleans are written as bytes,
    and the components past the lattice are one axis, as VTKHDF holds them.
    """
    rows = grid_rows(arrays, lead, n_rows, fmt=EXTENSION, group=group)
    out: dict[str, np.ndarray] = {}
    for key, arr in rows.items():
        if arr.dtype.kind == "b":
            arr = arr.astype(np.uint8)
        if arr.ndim > 2:
            # VTKHDF holds one component axis after the lattice, as .vti spells
            # a tensor: its nine components are one axis of nine, not two.
            arr = arr.reshape(n_rows, -1)
        out[key] = arr.reshape(vtk_lattice(lead, point_lead) + arr.shape[1:])
    return _linked(out, group=group)


def write_time_series(
    steps: Iterable[tuple[float, PolyData]], path: Source, **opts: Any
) -> None:
    """Write a sequence of meshes as one temporal VTKHDF file.

    Parameters
    ----------
    steps
        ``(time, mesh)`` pairs in time order. The first mesh's cells are
        written once and every step refers to them; a later mesh whose
        vertices moved since the last ones written writes its own points,
        while one whose elements differ is refused - that is another mesh,
        not another step of this one. Any iterable will do, a generator
        included: each step's arrays are appended as it arrives, and a
        generator that moves one mesh in place between yields is read as
        it stands at each, since what was written is what a step is
        compared with. Each step's time goes to ``Steps``, so a
        ``global_attrs["time"]`` is never field data here: a number is
        left to the pair's time, a text dropped with a warning.
    path
        Output file path, or an open binary file object.
    **opts
        As :func:`write`, ``time`` excepted: each step carries its own.

    Raises
    ------
    CodecError
        If ``steps`` is empty, a step's time is not a number, a step's
        elements differ from the first's, or a later step lacks an array
        the first one had, adds one, or holds one in another dtype or
        shape - refused before any of that step's rows are appended.
    """
    require_h5py(fmt=EXTENSION, name=source_name(path), verb="writing")
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    polydata = opts.pop("polydata", None)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write")
    remaining = iter(steps)
    head = next(remaining, None)
    if head is None:
        raise CodecError(
            f"'{source_name(path)}': a time series needs at least one step."
        )
    first_time, first = head
    name = source_name(path)
    kind = _write_kind(first, polydata, name=name)
    first_arrays = _step_arrays(first, timed=True, stacklevel=4)
    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        root = _new_root(handle, kind)
        store = _Store(root, dataset_kw, growing=True)
        times = [_time_value(first_time, name=name, index=0)]
        records = [_write_body(store, first, kind, first_arrays, geometry=True)]
        # Copies, not the mesh's own arrays: a generator that moves one mesh
        # in place would otherwise be compared with itself.
        cells = _cells_of(first)
        written = np.array(first.vertices, dtype=np.float64)
        for index, (time, poly) in enumerate(remaining, start=1):
            time = _time_value(time, name=name, index=index)
            _require_same_elements(cells, len(written), poly, index=index, name=name)
            arrays = _step_arrays(poly, timed=True, stacklevel=4)
            _require_same_arrays(first_arrays, arrays, index=index, name=name)
            moved = not np.array_equal(written, poly.vertices, equal_nan=True)
            record = _write_body(
                store,
                poly,
                kind,
                arrays,
                geometry=moved,
                previous=records[-1],
            )
            if moved:
                written = np.array(poly.vertices, dtype=np.float64)
            times.append(time)
            records.append(record)
        _write_steps(root, kind, times, records)


class _Store:
    """Datasets under the root, contiguous once or growing step by step.

    A single write creates each dataset whole. A temporal write creates it
    resizable on the first step and appends every later one, handing back
    the row each append began at, which is what the ``Steps`` group records.
    """

    def __init__(self, root: Any, dataset_kw: dict[str, Any], *, growing: bool) -> None:
        self.root = root
        self.dataset_kw = dataset_kw
        self.growing = growing

    def put(self, where: str, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        if where in self.root:
            node = self.root[where]
            start = int(node.shape[0])
            if node.shape[1:] != arr.shape[1:] or node.dtype != arr.dtype:
                raise CodecError(
                    f"{EXTENSION} write: '{where}' changes shape or dtype from"
                    f" one step to the next ({node.shape[1:]} {node.dtype} to"
                    f" {arr.shape[1:]} {arr.dtype})."
                )
            node.resize(start + arr.shape[0], axis=0)
            if arr.shape[0]:
                node[start:] = arr
            return start
        kwargs: dict[str, Any] = dict(self.dataset_kw) if arr.size > 1 else {}
        if self.growing:
            kwargs["maxshape"] = (None, *arr.shape[1:])
            kwargs["chunks"] = True
        self.root.create_dataset(where, data=arr, **kwargs)
        return 0

    def put_text(self, where: str, lines: tuple[str, ...]) -> int:
        h5py, _ = _h5py()
        # Variable-length ASCII holding UTF-8 bytes, the way VTK's own writer
        # spells a string array: its reader has no conversion for the UTF-8
        # string type and hands back '' for one.
        data = np.array([line.encode("utf-8") for line in lines], dtype=object)
        if where in self.root:
            node = self.root[where]
            start = int(node.shape[0])
            node.resize(start + len(data), axis=0)
            node[start:] = data
            return start
        kwargs: dict[str, Any] = {}
        if self.growing:
            kwargs["maxshape"] = (None,)
            kwargs["chunks"] = True
        self.root.create_dataset(
            where, data=data, dtype=h5py.string_dtype("ascii"), **kwargs
        )
        return 0


def _new_root(handle: Any, kind: str) -> Any:
    root = handle.create_group(_ROOT)
    root.attrs["Version"] = np.array(_VERSION, dtype=np.int64)
    # A fixed-length string of the name's own size, the way VTK spells it;
    # its reader takes the attribute's size as the string's.
    root.attrs["Type"] = np.bytes_(kind)
    return root


def _write_kind(poly: PolyData, polydata: Any, *, name: str) -> str:
    if polydata is None:
        polydata = (poly.global_attrs or {}).get(_TYPE_KEY) == _POLYDATA
    if not polydata:
        return _UNSTRUCTURED
    codes = np.unique(np.asarray(poly.element_types))
    outside = [
        ELEMENT_TYPES_INV.get(int(c), f"type_{int(c)}")
        for c in codes
        if int(c) not in _SECTION_OF
    ]
    if outside:
        raise CodecError(
            f"'{name}': a VTKHDF PolyData holds vertices, lines, polygons and"
            f" strips, and the mesh holds {outside}; write it as an"
            " UnstructuredGrid (polydata=False)."
        )
    return _POLYDATA


def _held_time(poly: PolyData) -> float | None:
    """The number under ``global_attrs["time"]``, or None when there is none."""
    held = (poly.global_attrs or {}).get(_TIME_KEY)
    if isinstance(held, bool) or not isinstance(
        held, (int, float, np.integer, np.floating)
    ):
        return None
    return float(held)


def _time_value(time: Any, *, name: str, index: int) -> float:
    try:
        return float(time)
    except (TypeError, ValueError):
        raise CodecError(
            f"'{name}': step {index} is at time {time!r}, which is not a number."
        ) from None


def _step_arrays(
    poly: PolyData, *, timed: bool, stacklevel: int
) -> dict[str, dict[str, Any]]:
    """The arrays one step writes, gathered before any dataset is created.

    Parameters
    ----------
    poly
        The step's mesh.
    timed
        Whether the step's time is written to ``Steps``. It then reads back
        as ``global_attrs["time"]`` over any field data of that name, so
        ``time`` is reserved: a number there is the time itself, a text is
        dropped with a warning. Without a time it is field data like any
        other.
    stacklevel
        Where the warnings point.

    Returns
    -------
    dict
        ``point`` and ``cell``: name to numeric array, one row per entity,
        tag masks included; ``field``: name to numeric array; ``text``:
        name to the lines of a text value. A series checks these names
        against the first step's before it appends a row.
    """
    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)
    if timed and _TIME_KEY in (poly.global_attrs or {}) and _held_time(poly) is None:
        warnings.warn(
            f"{EXTENSION} write: global '{_TIME_KEY}' holds"
            f" {poly.global_attrs[_TIME_KEY]!r}, and the step's time is written"
            " to Steps, which reads back over it; dropped.",
            UserWarning,
            stacklevel=stacklevel,
        )
    reserved = _RESERVED_GLOBALS if timed else _RESERVED_GLOBALS - {_TIME_KEY}
    held = _field_arrays(
        poly.global_attrs, reserved=reserved, stacklevel=stacklevel + 1
    )
    return {
        "point": _linked(
            _numeric_arrays(
                with_tag_masks(
                    poly.vertex_attrs,
                    poly.vertex_tags,
                    n_verts,
                    fmt=EXTENSION,
                    kind="point",
                ),
                n_verts,
                kind="point",
            ),
            group="PointData",
        ),
        "cell": _linked(
            _numeric_arrays(
                with_tag_masks(
                    poly.element_attrs,
                    poly.element_tags,
                    n_elems,
                    fmt=EXTENSION,
                    kind="cell",
                ),
                n_elems,
                kind="cell",
            ),
            group="CellData",
        ),
        "field": held["field"],
        "text": held["text"],
    }


def _field_arrays(
    global_attrs: dict[str, Any] | None,
    *,
    reserved: frozenset[str] | set[str],
    stacklevel: int,
) -> dict[str, dict[str, Any]]:
    """The FieldData one step or one image writes, numbers and text by name.

    Parameters
    ----------
    global_attrs
        The whole-mesh or whole-image metadata.
    reserved
        The keys the writer spells itself, which are not field data: the grid
        keys, and for a step whose time is written, ``time`` (see
        :func:`_step_arrays`).
    stacklevel
        Where the warnings point.

    Returns
    -------
    dict
        ``field``: name to numeric array, one row per tuple; ``text``: name
        to the lines of a text value.
    """
    numeric = spell_globals(
        global_attrs,
        reserved=reserved,
        fmt=EXTENSION,
        text=True,
        stacklevel=stacklevel + 1,
    )
    field: dict[str, np.ndarray] = {}
    for key, arr in numeric.items():
        arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
        field[key] = arr.astype(np.uint8) if arr.dtype.kind == "b" else arr
    text = spell_text(global_attrs, reserved=reserved)
    for key, lines in text.items():
        if any("\x00" in line for line in lines):
            # A variable-length HDF5 string is a C string, ended by its NUL.
            warnings.warn(
                f"{EXTENSION} write: text global '{key}' holds a NUL, which a"
                " VTKHDF string cannot; written without it.",
                UserWarning,
                stacklevel=stacklevel,
            )
            text[key] = tuple(line.replace("\x00", "") for line in lines)
    linked = _linked({**field, **text}, group="FieldData")
    return {
        "field": {k: v for k, v in linked.items() if isinstance(v, np.ndarray)},
        "text": {k: v for k, v in linked.items() if not isinstance(v, np.ndarray)},
    }


def _linked(arrays: dict[str, Any], *, group: str) -> dict[str, Any]:
    """The arrays under names an HDF5 link can carry, one dataset each.

    A ``/`` in a name would nest the dataset in a subgroup the reader
    walks past, and an empty name, ``.`` or ``..`` is no link at all. Each
    ``/`` becomes ``_`` and an empty name ``array``; a name that then
    matches another's gets a number appended. Every rename is warned about,
    since the array reads back under the new name.
    """
    out: dict[str, Any] = {}
    renamed: dict[str, str] = {}
    for key, arr in arrays.items():
        link = key.replace("/", "_")
        if link in ("", ".", ".."):
            link = "array"
        base = link
        n = 1
        while link in out or (link != key and link in arrays):
            n += 1
            link = f"{base}_{n}"
        if link != key:
            renamed[key] = link
        out[link] = arr
    if renamed:
        warn_caller(
            f"{EXTENSION} write: {group} array(s) {renamed} are named as no"
            " HDF5 dataset can be - a '/' in the name, or none - and are"
            " written under the names shown."
        )
    return out


def _write_body(
    store: _Store,
    poly: PolyData,
    kind: str,
    arrays: dict[str, dict[str, Any]],
    *,
    geometry: bool,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one step's geometry and arrays, and say where each began.

    Parameters
    ----------
    store
        Where the datasets go.
    poly
        The step's mesh.
    kind
        ``UnstructuredGrid`` or ``PolyData``.
    arrays
        The step's arrays, from :func:`_step_arrays`.
    geometry
        Whether the points are written; the cells are written only when
        the store holds none yet.
    previous
        The step before's record, when this is a later one: the cells are
        shared with every step, and the points with it unless ``geometry``
        says they moved, so only the arrays are appended.

    Returns
    -------
    dict
        ``points``: the row the points begin at; ``cells`` and ``ids``: the
        row each cell group's cells and connectivity ids begin at, per
        section for a PolyData; ``point_data``, ``cell_data``: name to row;
        ``field_data``: name to ``(row, components, tuples)``.
    """
    n_verts = poly.vertices.shape[0]
    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise CodecError(
            f"{EXTENSION} write: vertices are shaped {vertices.shape}, not (n, 3)."
        )
    record: dict[str, Any] = {}
    if previous is None:
        record["points"] = store.put("Points", vertices)
        if kind == _POLYDATA:
            record["cells"], record["ids"], kept = _write_polydata_cells(store, poly)
        else:
            record["cells"], record["ids"], kept = _write_unstructured_cells(
                store, poly
            )
        store.put("NumberOfPoints", np.array([n_verts], dtype=np.int64))
        record["kept"] = kept
    else:
        record["points"] = (
            store.put("Points", vertices) if geometry else previous["points"]
        )
        record["cells"] = previous["cells"]
        record["ids"] = previous["ids"]
        kept = previous["kept"]
        record["kept"] = kept
    record["point_data"] = {
        key: store.put(f"PointData/{key}", arr) for key, arr in arrays["point"].items()
    }
    record["cell_data"] = {
        key: store.put(f"CellData/{key}", arr[kept] if kept is not None else arr)
        for key, arr in arrays["cell"].items()
    }
    field: dict[str, tuple[int, int, int]] = {}
    for key, arr in arrays["field"].items():
        components = arr.shape[1] if arr.ndim == 2 else 1
        field[key] = (store.put(f"FieldData/{key}", arr), components, arr.shape[0])
    for key, lines in arrays["text"].items():
        field[key] = (store.put_text(f"FieldData/{key}", lines), 1, len(lines))
    record["field_data"] = field
    return record


def _write_unstructured_cells(
    store: _Store, poly: PolyData
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    codes = np.asarray(poly.element_types)
    vtk_types = np.array(
        [POLYXIOS_TO_VTK.get(ELEMENT_TYPES_INV.get(int(c), ""), -1) for c in codes],
        dtype=np.int64,
    )
    unknown = vtk_types < 0
    if unknown.any():
        warn_caller(
            f"{EXTENSION} write: element type(s)"
            f" {sorted({ELEMENT_TYPES_INV.get(int(c), f'type_{int(c)}') for c in codes[unknown]})}"
            " have no VTK cell type; those elements and the values and tags on"
            " them are not written."
        )
    kept = None if not unknown.any() else np.flatnonzero(~unknown)
    offsets = np.asarray(poly.offsets, dtype=np.int64)
    conn = np.asarray(poly.connectivity, dtype=np.int64)
    if kept is not None:
        widths = np.diff(offsets)[kept]
        starts = offsets[kept]
        conn = (
            np.concatenate(
                [conn[s : s + w] for s, w in zip(starts, widths, strict=True)]
            )
            if kept.size
            else conn[:0]
        )
        offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        vtk_types = vtk_types[kept]
    store.put("Types", vtk_types.astype(np.uint8))
    store.put("Connectivity", conn)
    store.put("Offsets", offsets)
    store.put("NumberOfCells", np.array([len(vtk_types)], dtype=np.int64))
    store.put("NumberOfConnectivityIds", np.array([conn.size], dtype=np.int64))
    return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), kept


def _write_polydata_cells(
    store: _Store, poly: PolyData
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    codes = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets, dtype=np.int64)
    conn = np.asarray(poly.connectivity, dtype=np.int64)
    order: list[np.ndarray] = []
    for section in _SECTIONS:
        here = np.flatnonzero(
            np.isin(codes, [c for c, s in _SECTION_OF.items() if s == section])
        )
        order.append(here)
        widths = np.diff(offsets)[here]
        starts = offsets[here]
        section_conn = (
            np.concatenate(
                [conn[s : s + w] for s, w in zip(starts, widths, strict=True)]
            )
            if here.size
            else conn[:0]
        )
        section_offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        store.put(f"{section}/Connectivity", section_conn)
        store.put(f"{section}/Offsets", section_offsets)
        store.put(f"{section}/NumberOfCells", np.array([here.size], dtype=np.int64))
        store.put(
            f"{section}/NumberOfConnectivityIds",
            np.array([section_conn.size], dtype=np.int64),
        )
    kept = np.concatenate(order)
    if np.array_equal(kept, np.arange(len(codes))):
        kept = None
    return (
        np.zeros(len(_SECTIONS), dtype=np.int64),
        np.zeros(len(_SECTIONS), dtype=np.int64),
        kept,
    )


def _numeric_arrays(
    arrays: dict[str, np.ndarray], n_rows: int, *, kind: str
) -> dict[str, np.ndarray]:
    """The arrays a data group can hold: numeric, one row per entity."""
    out: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for key, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.dtype.kind not in "biuf" or arr.ndim == 0 or arr.shape[0] != n_rows:
            dropped.append(str(key))
            continue
        if arr.dtype.kind == "b":
            arr = arr.astype(np.uint8)
        if arr.ndim > 2:
            arr = arr.reshape(n_rows, -1)
        out[str(key)] = arr
    if dropped:
        warn_caller(
            f"{EXTENSION} write: {kind} array(s) {dropped} are not numeric or not"
            f" one row per {kind}; dropped."
        )
    return out


def _write_steps(
    root: Any, kind: str, times: list[float], records: list[dict[str, Any]]
) -> None:
    n = len(times)
    steps = root.create_group("Steps")
    steps.attrs["NSteps"] = np.int64(n)
    steps.create_dataset("Values", data=np.asarray(times, dtype=np.float64))
    steps.create_dataset("PartOffsets", data=np.zeros(n, dtype=np.int64))
    steps.create_dataset("NumberOfParts", data=np.ones(n, dtype=np.int64))
    steps.create_dataset(
        "PointOffsets", data=np.array([r["points"] for r in records], dtype=np.int64)
    )
    cells = np.array([r["cells"] for r in records], dtype=np.int64)
    ids = np.array([r["ids"] for r in records], dtype=np.int64)
    if kind != _POLYDATA:
        cells = cells.reshape(n)
        ids = ids.reshape(n)
    steps.create_dataset("CellOffsets", data=cells)
    steps.create_dataset("ConnectivityIdOffsets", data=ids)
    for group_name, key in (("PointData", "point_data"), ("CellData", "cell_data")):
        holder = steps.create_group(f"{group_name}Offsets")
        for name in records[0][key]:
            holder.create_dataset(
                name, data=np.array([r[key][name] for r in records], dtype=np.int64)
            )
    offsets = steps.create_group("FieldDataOffsets")
    sizes = steps.create_group("FieldDataSizes")
    for name in records[0]["field_data"]:
        offsets.create_dataset(
            name,
            data=np.array([r["field_data"][name][0] for r in records], dtype=np.int64),
        )
        sizes.create_dataset(
            name,
            data=np.array([r["field_data"][name][1:] for r in records], dtype=np.int64),
        )


def _cells_of(poly: PolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copies of the cells step 0 wrote, to hold every later step against."""
    return (
        np.array(poly.element_types),
        np.array(poly.offsets),
        np.array(poly.connectivity),
    )


def _require_same_elements(
    cells: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_verts: int,
    poly: PolyData,
    *,
    index: int,
    name: str,
) -> None:
    """Refuse a step whose cells or vertex count are not what step 0 wrote."""
    types, offsets, connectivity = cells
    same = (
        np.array_equal(types, poly.element_types)
        and np.array_equal(offsets, poly.offsets)
        and np.array_equal(connectivity, poly.connectivity)
    )
    if not same:
        raise CodecError(
            f"'{name}': step {index} holds different elements from step 0. A"
            " time series is one mesh over time; a mesh whose elements change"
            " is another mesh."
        )
    if len(poly.vertices) != n_verts:
        raise CodecError(
            f"'{name}': step {index} holds {len(poly.vertices)} vertices and"
            f" step 0 holds {n_verts}."
        )


def _require_same_arrays(
    first: dict[str, dict[str, Any]],
    arrays: dict[str, dict[str, Any]],
    *,
    index: int,
    name: str,
) -> None:
    """Refuse a step whose arrays are not the first step's, by name or kind.

    Checked before the step's first row is appended, so the refusal
    creates no dataset the file would then carry half-written and appends
    no row of a step the file then lacks the rest of. A numeric array has
    to keep its dtype and its components, since every step's rows go into
    the one dataset; a field array may change how many tuples it holds,
    and a text how many lines, as ``FieldDataSizes`` spells each step's.
    """
    for keys, label in (
        (("point",), "PointData"),
        (("cell",), "CellData"),
        (("field", "text"), "FieldData"),
    ):
        was = {k for key in keys for k in first[key]}
        now = {k for key in keys for k in arrays[key]}
        missing = sorted(was - now)
        extra = sorted(now - was)
        if missing:
            raise CodecError(
                f"'{name}': step {index} lacks {label} array(s) {missing} that"
                " step 0 wrote; every step of a series carries the same arrays."
            )
        if extra:
            raise CodecError(
                f"'{name}': step {index} adds {label} array(s) {extra} that step 0"
                " did not write; every step of a series carries the same arrays."
            )
        for key in keys:
            for k, before in first[key].items():
                after = arrays[key].get(k)
                if after is None:
                    was_text = key == "text"
                    raise CodecError(
                        f"'{name}': step {index} spells {label} '{k}' as"
                        f" {'numbers' if was_text else 'text'} where step 0 wrote"
                        f" {'text' if was_text else 'numbers'}."
                    )
                if key == "text":
                    continue
                if after.dtype != before.dtype or after.shape[1:] != before.shape[1:]:
                    raise CodecError(
                        f"'{name}': step {index} changes shape or dtype of"
                        f" {label} array '{k}' from {before.shape} {before.dtype}"
                        f" to {after.shape} {after.dtype}; every step of a"
                        " series holds an array the way step 0 did."
                    )


__all__ = [
    "EXTENSION",
    "LABEL",
    "read",
    "read_image",
    "read_time_series",
    "write",
    "write_image",
    "write_time_series",
]
