"""HMF codec - XDMF's model in one HDF5 file, with no XML beside it.

HMF is the experimental format meshio invented for the case XDMF serves
badly: a mesh whose arrays are all in the HDF5 file anyway, and whose XML
description therefore only says where they are. The file is one ``domain``
group holding one ``grid``, and the grid holds a ``Geometry`` dataset, one
``Topology<k>`` dataset per element type - named for XDMF's topology types,
``Triangle``, ``Tetrahedron_10`` and so on - and the arrays over the nodes
and the cells under ``NodeAttributes`` and ``CellAttributes``. polyxios adds
``NodeSets`` and ``CellSets`` for its tag groups and ``Attributes`` for the
mesh-wide values, each only when the mesh has some: the originator's reader
asserts on any key it does not know, so a mesh with none of them is written
in exactly the layout it reads, and one with them is polyxios' own.

The format is versioned ``0.1-alpha`` and calls itself experimental; this
codec reads and writes that version.
"""

from __future__ import annotations

import re
from typing import Any
import warnings

import numpy as np

from polyxios._dimension import mark_2d, output_dimension, pad_to_3d
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV, NODES_PER_ELEMENT
from polyxios._globals import globals_for_write, text_for_write
from polyxios._io import Source, source_name, source_size
from polyxios._tags import member_indices
from polyxios._types import PolyData, make_polydata
from polyxios.codecs._hdf5 import (
    _h5py,
    as_array,
    attr_text,
    child,
    dataset_options,
    kept_index,
    open_hdf5_read,
    open_hdf5_write,
    refuse_lazy,
    reindexed_tags,
    require_h5py,
    text_of,
    type_blocks,
    warn_unknown_opts,
)
from polyxios.codecs._xdmf import WRITE_MAP, XDMF_TO_POLYXIOS
from polyxios.exceptions import CodecError
from polyxios.validate import validate_header

EXTENSION: str = ".hmf"

_TYPE: str = "hmf"
_VERSION: str = "0.1-alpha"
_TOPOLOGY: re.Pattern[str] = re.compile(r"Topology(\d*)$")
# The topologies whose node count XDMF fixes; a polyvertex, polyline or
# polygon dataset is read at whatever width it has.
_WRITE_SIZES: dict[int, int] = {
    int(code): NODES_PER_ELEMENT[ELEMENT_TYPES_INV[int(code)]]
    for code in WRITE_MAP
    if NODES_PER_ELEMENT[ELEMENT_TYPES_INV[int(code)]] > 0
}
_POLYGON: int = int(ELEMENT_TYPES["polygon"])
_RESERVED_GLOBALS: frozenset[str] = frozenset({"was_2d"})


# ----- reading ----------------------------------------------------------------


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read an HMF file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.hmf`` file, or an open binary file object.
    lazy
        Not supported; raises LazyReadError when True.
    **opts
        None are taken; any given are warned about and ignored.

    Returns
    -------
    PolyData
        The mesh. ``NodeAttributes`` are ``vertex_attrs``, ``CellAttributes``
        ``element_attrs``, ``NodeSets`` and ``CellSets`` the tag groups and
        ``Attributes`` the ``global_attrs``.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    UnsupportedFormatError
        If h5py is not installed. The message names the extra that installs it.
    CodecError
        If the file is not HDF5, does not call itself ``hmf``, holds no
        grid or geometry, names a topology XDMF does not define, or a
        topology names a node outside the geometry.

    Warns
    -----
    UserWarning
        For an attribute whose length fits neither the nodes nor the cells,
        which is skipped; for a file whose ``version`` is not the one this
        reader knows, which is read as that one.
    """
    name = source_name(path)
    refuse_lazy(lazy, fmt=EXTENSION, name=name)
    warn_unknown_opts(opts, fmt=EXTENSION, what="read")
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        ctx = {"name": name, "file_size": source_size(path)}
        if attr_text(handle, "type") != _TYPE:
            raise CodecError(f"'{name}': the file does not call itself 'hmf'.")
        version = attr_text(handle, "version")
        if version != _VERSION:
            warnings.warn(
                f"{EXTENSION} read: '{name}' is HMF {version or 'of no version'};"
                f" this reader knows {_VERSION}, and reads it as that.",
                UserWarning,
                stacklevel=2,
            )
        domain = child(handle, "domain", name=name, where="/")
        grid = child(domain, "grid", name=name, where="domain")
        geometry = child(grid, "Geometry", name=name, where="domain/grid")
        coords = as_array(geometry, name=name, where="domain/grid/Geometry")
        if (
            coords.ndim != 2
            or coords.dtype.kind not in "iuf"
            or coords.shape[1] not in (1, 2, 3)
        ):
            raise CodecError(
                f"'{name}': 'Geometry' is not an (n, dim) coordinate block."
            )
        n_verts = len(coords)
        validate_header(n_verts, 0, 0, ctx["file_size"], compressed=True)
        dim = coords.shape[1]
        vertices = (
            pad_to_3d(coords, dim)
            if dim > 1
            else np.column_stack([coords[:, 0], np.zeros(n_verts), np.zeros(n_verts)])
        )
        blocks = _read_topologies(grid, n_verts, ctx)
        n_elems = sum(len(cells) for _, cells in blocks)
        vertex_attrs = _read_arrays(grid, "NodeAttributes", n_verts, "nodes", name)
        element_attrs = _read_arrays(grid, "CellAttributes", n_elems, "cells", name)
        vertex_tags = _read_sets(grid, "NodeSets", n_verts, name)
        element_tags = _read_sets(grid, "CellSets", n_elems, name)
        global_attrs = _read_globals(grid, name)
    global_attrs.update(mark_2d(dim))
    return make_polydata(
        vertices,
        blocks,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _read_topologies(
    grid: Any, n_verts: int, ctx: dict[str, Any]
) -> list[tuple[str, np.ndarray]]:
    """Every Topology dataset, in the numbered order they were written."""
    name = ctx["name"]
    keys = []
    for key in grid:
        match = _TOPOLOGY.match(key)
        if match and hasattr(grid[key], "shape"):
            keys.append((int(match.group(1) or 0), key))
    blocks: list[tuple[str, np.ndarray]] = []
    for _, key in sorted(keys):
        node = grid[key]
        topo = attr_text(node, "TopologyType")
        if topo is None:
            raise CodecError(f"'{name}': '{key}' names no TopologyType.")
        ptype = XDMF_TO_POLYXIOS.get(topo.lower())
        if ptype is None:
            raise CodecError(
                f"'{name}': '{key}' is a {topo!r}, which XDMF does not name."
            )
        cells = as_array(node, name=name, where=f"domain/grid/{key}")
        if cells.dtype.kind not in "iu" or cells.ndim != 2:
            raise CodecError(f"'{name}': '{key}' is not an (n, k) index table.")
        width = NODES_PER_ELEMENT[ptype]
        if width > 0 and cells.shape[1] != width:
            raise CodecError(
                f"'{name}': '{key}' holds {cells.shape[1]} nodes per {topo}, not {width}."
            )
        validate_header(
            n_verts, len(cells), cells.size, ctx["file_size"], compressed=True
        )
        cells = cells.astype(np.int64)
        if cells.size and (cells.min() < 0 or cells.max() >= n_verts):
            raise CodecError(
                f"'{name}': '{key}' names a node outside 0..{n_verts - 1}."
            )
        blocks.append((ptype, cells))
    return blocks


def _read_arrays(
    grid: Any, key: str, count: int, what: str, name: str
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if key not in grid:
        return out
    for attr_name in grid[key]:
        node = grid[key][attr_name]
        if not hasattr(node, "shape"):
            continue
        values = as_array(node, name=name, where=f"domain/grid/{key}/{attr_name}")
        if values.dtype.kind not in "biuf":
            continue
        if values.ndim == 0 or values.shape[0] != count:
            warnings.warn(
                f"{EXTENSION} read: '{key}/{attr_name}' in '{name}' holds"
                f" {values.shape[0] if values.ndim else 1} values for {count} {what};"
                " skipped.",
                UserWarning,
                stacklevel=4,
            )
            continue
        out[attr_name] = values
    return out


def _read_sets(grid: Any, key: str, count: int, name: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if key not in grid:
        return out
    for set_name in grid[key]:
        node = grid[key][set_name]
        if not hasattr(node, "shape"):
            continue
        values = as_array(
            node, name=name, where=f"domain/grid/{key}/{set_name}"
        ).ravel()
        if values.dtype.kind not in "iu":
            continue
        picked = values.astype(np.int64)
        out[set_name] = np.unique(picked[(picked >= 0) & (picked < count)])
    return out


def _read_globals(grid: Any, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "Attributes" not in grid:
        return out
    holder = grid["Attributes"]
    for key in holder:
        node = holder[key]
        if not hasattr(node, "shape"):
            continue
        values = np.asarray(node[()])
        if values.dtype.kind in "biuf":
            out[key] = values.reshape(-1)[0] if values.size == 1 else values
        else:
            texts = [
                t
                for t in (text_of(v) for v in np.atleast_1d(values).ravel())
                if t is not None
            ]
            if texts:
                out[key] = texts[0] if len(texts) == 1 else texts
    return out


# ----- writing ----------------------------------------------------------------


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as an HMF file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object.
    **opts
        ``compression`` and ``compression_opts`` are passed to h5py for the
        datasets - ``compression="gzip"`` with ``compression_opts=4`` is the
        usual pair.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If the mesh's vertices are not a coordinate block.

    Warns
    -----
    UserWarning
        Once for the element types XDMF has no topology for, polygons of
        differing widths among them - those elements, and the values and
        tags on them, are not written; once for the ``global_attrs`` values
        that are neither numbers nor text.

    Notes
    -----
    One ``Topology<k>`` dataset per element type, the types in ascending
    polyxios code order; a pixel and a voxel go out as the quadrilateral and
    the hexahedron they are, polygons all of one width as a ``Polygon``
    table. The ``Geometry`` is ``XYZ``, or ``XY`` for a mesh that came from
    a two-dimensional file and stayed in the plane. ``NodeSets``,
    ``CellSets`` and ``Attributes`` are written only when the mesh has tag
    groups or mesh-wide values, since the originator's reader refuses a
    key it does not know.
    """
    require_h5py(fmt=EXTENSION, name=source_name(path), verb="writing")
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write")
    dim = output_dimension(poly, fmt=EXTENSION)
    numeric = globals_for_write(
        poly, reserved=_RESERVED_GLOBALS, fmt=EXTENSION, text=True
    )
    texts = text_for_write(poly, reserved=_RESERVED_GLOBALS)
    n_verts = len(poly.vertices)
    n_elems = len(poly.element_types)
    blocks, dropped = type_blocks(poly, sizes=_WRITE_SIZES, fmt=EXTENSION)
    polygons = _polygon_block(poly)
    if polygons is not None:
        dropped.discard(_POLYGON)
        blocks.append(polygons)
        blocks.sort(key=lambda block: block[0])
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: element type(s)"
            f" {sorted(ELEMENT_TYPES_INV.get(c, f'type_{c}') for c in dropped)}"
            " have no XDMF topology"
            + (
                " of one width, and the polygons here differ in theirs"
                if _POLYGON in dropped
                else ""
            )
            + "; those elements and the values and tags on them are not written.",
            UserWarning,
            stacklevel=2,
        )
    kept = kept_index(blocks)
    element_tags = reindexed_tags(poly.element_tags, kept, n_elems)

    def store(group: Any, key: str, arr: np.ndarray) -> Any:
        return group.create_dataset(
            key, data=np.ascontiguousarray(arr), **(dataset_kw if arr.size > 1 else {})
        )

    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        handle.attrs["type"] = _TYPE
        handle.attrs["version"] = _VERSION
        grid = handle.create_group("domain").create_group("grid")
        geometry = store(
            grid, "Geometry", np.asarray(poly.vertices, dtype=np.float64)[:, :dim]
        )
        geometry.attrs["GeometryType"] = "XYZ"[:dim]
        for k, (code, _, conn) in enumerate(blocks):
            topo, order = WRITE_MAP[code]
            if order is not None:
                conn = conn[:, list(order)]
            store(grid, f"Topology{k}", conn).attrs["TopologyType"] = topo
        _write_arrays(grid, "NodeAttributes", poly.vertex_attrs, n_verts, None, store)
        _write_arrays(grid, "CellAttributes", poly.element_attrs, n_elems, kept, store)
        if poly.vertex_tags:
            holder = grid.create_group("NodeSets")
            for tag_name, members in poly.vertex_tags.items():
                store(
                    holder,
                    str(tag_name),
                    member_indices(members, n_verts).astype(np.int64),
                )
        if element_tags:
            holder = grid.create_group("CellSets")
            for tag_name, picked in element_tags.items():
                store(holder, str(tag_name), picked.astype(np.int64))
        if numeric or texts:
            holder = grid.create_group("Attributes")
            for key, arr in numeric.items():
                store(holder, key, arr)
            for key, lines in texts.items():
                holder.create_dataset(
                    key, data=np.array(list(lines), dtype=object), dtype=_string_dtype()
                )


def _string_dtype() -> Any:
    h5py, _ = _h5py()
    return h5py.string_dtype("utf-8")


def _polygon_block(poly: PolyData) -> tuple[int, np.ndarray, np.ndarray] | None:
    """The polygons as one ``Polygon`` block, when they are all of one width.

    A ``Topology`` dataset is an ``(n, k)`` table, so polygons of differing
    widths have no place in it; those are left to the caller's warning.
    """
    codes = np.asarray(poly.element_types)
    here = np.flatnonzero(codes == _POLYGON)
    if not here.size:
        return None
    offsets = np.asarray(poly.offsets)
    widths = np.diff(offsets)[here]
    width = int(widths[0])
    if width < 3 or (widths != width).any():
        return None
    conn = np.asarray(poly.connectivity)
    cells = conn[offsets[here][:, None] + np.arange(width)[None, :]]
    return _POLYGON, here, cells.astype(np.int64)


def _write_arrays(
    grid: Any,
    key: str,
    attrs: dict[str, np.ndarray],
    count: int,
    index: np.ndarray | None,
    store: Any,
) -> None:
    kept: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for attr_name, values in attrs.items():
        arr = np.asarray(values)
        if arr.dtype.kind == "b":
            arr = arr.astype(np.int32)
        if arr.dtype.kind not in "iuf" or arr.ndim == 0 or arr.shape[0] != count:
            dropped.append(attr_name)
            continue
        kept[attr_name] = arr if index is None else arr[index]
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: attribute(s) {sorted(dropped)} are not one numeric"
            " value per entity; dropped.",
            UserWarning,
            stacklevel=3,
        )
    holder = grid.create_group(key)
    for attr_name, arr in kept.items():
        store(holder, attr_name, arr)
