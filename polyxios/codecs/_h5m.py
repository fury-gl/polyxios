"""H5M codec - MOAB's native HDF5 mesh file.

Everything sits under ``/tstt``: the coordinates in ``nodes/coordinates``,
one group per element type under ``elements`` (``Tri3``, ``Tet4``, ``Hex8``
and so on, named for the shape and its node count) holding a ``connectivity``
table, and ``sets`` - the meshsets MOAB builds materials, boundary conditions
and partitions out of. Every entity has a *handle id*: the nodes are numbered
from the ``start_id`` on their coordinates, then each element group from its
own, in the order the file assigns them, and both the connectivity and the
sets speak in those ids. Values on the entities are *dense tags*, one dataset
per name under the owner's ``tags`` group, each also declared under
``/tstt/tags`` with its datatype; a *sparse tag* names its entities in an
``id_list`` beside its ``values``.

A set lands here as a tag group, named by its ``NAME`` tag when it has one and
by its ``MATERIAL_SET``, ``NEUMANN_SET`` or ``DIRICHLET_SET`` number
otherwise; a set holding both nodes and elements becomes one group of each.
Node order is MOAB's canonical one, which is VTK's for every type both name
but the wedges and the quadratic hexahedra: MOAB's prism is VTK's wedge with
its triangles turned over, and MOAB numbers a hexahedron's vertical edges
before its top ring where VTK numbers them after; both are permuted on the
way in and out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import warnings

import numpy as np

from polyxios._dimension import output_dimension
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV, NODES_PER_ELEMENT
from polyxios._globals import globals_for_write
from polyxios._ids import IDS_KEY, record_ids
from polyxios._io import Source, source_name, source_size
from polyxios._tags import member_indices
from polyxios._types import PolyData, make_polydata
from polyxios.codecs._hdf5 import (
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
from polyxios.exceptions import CodecError
from polyxios.validate import validate_header

EXTENSION: str = ".h5m"

# MOAB's entity types, as its ``elemtypes`` enum numbers them.
_MOAB_ENUM: dict[str, int] = {
    "Edge": 1,
    "Tri": 2,
    "Quad": 3,
    "Polygon": 4,
    "Tet": 5,
    "Pyramid": 6,
    "Prism": 7,
    "Knife": 8,
    "Hex": 9,
    "Polyhedron": 10,
}
# The MOAB group names polyxios can hold and what they read as; MOAB names a
# group for the shape and its node count.
_H5M_TO_POLYXIOS: dict[str, str] = {
    "Edge2": "line",
    "Edge3": "quadratic_edge",
    "Edge4": "cubic_line",
    "Tri3": "triangle",
    "Tri6": "quadratic_triangle",
    "Tri7": "biquadratic_triangle",
    "Quad4": "quad",
    "Quad8": "quadratic_quad",
    "Quad9": "biquadratic_quad",
    "Tet4": "tetra",
    "Tet10": "quadratic_tetra",
    "Pyramid5": "pyramid",
    "Pyramid13": "quadratic_pyramid",
    "Prism6": "wedge",
    "Prism15": "quadratic_wedge",
    "Prism18": "biquadratic_quadratic_wedge",
    "Hex8": "hexahedron",
    "Hex20": "quadratic_hexahedron",
    "Hex27": "triquadratic_hexahedron",
}
_POLYXIOS_TO_H5M: dict[str, str] = {v: k for k, v in _H5M_TO_POLYXIOS.items()}
_LATTICE: dict[str, tuple[str, tuple[int, ...]]] = {
    "pixel": ("Quad4", (0, 1, 3, 2)),
    "voxel": ("Hex8", (0, 1, 3, 2, 4, 5, 7, 6)),
}
_WRITE_SIZES: dict[int, int] = {
    int(ELEMENT_TYPES[name]): NODES_PER_ELEMENT[name]
    for name in (*_POLYXIOS_TO_H5M, *_LATTICE)
}
# MOAB node index per VTK node, as MOAB's own VTK reader spells it in
# src/io/VtkUtil.cpp (the ``wedge`` and ``qhex`` tables): a prism's triangles
# run the other way round, and a hexahedron's mid-edge nodes go bottom ring,
# verticals, top ring where VTK goes bottom, top, verticals, its face centres
# by MOAB's face order where VTK goes -x, +x, -y, +y, -z, +z.
_WEDGE_ORDER: tuple[int, ...] = (0, 2, 1, 3, 5, 4, 8, 7, 6, 14, 13, 12, 9, 11, 10)
_HEX_ORDER: tuple[int, ...] = (*range(12), 16, 17, 18, 19, 12, 13, 14, 15)
_READ_ORDER: dict[str, tuple[int, ...]] = {
    "wedge": _WEDGE_ORDER[:6],
    "quadratic_wedge": _WEDGE_ORDER,
    "biquadratic_quadratic_wedge": (*_WEDGE_ORDER, 17, 16, 15),
    "quadratic_hexahedron": _HEX_ORDER,
    "triquadratic_hexahedron": (*_HEX_ORDER, 23, 21, 20, 22, 24, 25, 26),
}
_WRITE_ORDER: dict[str, tuple[int, ...]] = {
    name: tuple(order.index(i) for i in range(len(order)))
    for name, order in _READ_ORDER.items()
}

_DENSE: int = 2
_SPARSE: int = 1
_MESH: int = 3
# Set flags MOAB writes: bit 3 says the contents are (start, count) ranges.
_RANGE_BIT: int = 0x8
_SET_FLAG: int = 0x2
_NAME_TAG: str = "NAME"
_NAME_WIDTH: int = 32
_SET_KINDS: tuple[str, ...] = ("MATERIAL_SET", "NEUMANN_SET", "DIRICHLET_SET")
_SET_PREFIX: dict[str, str] = {
    "MATERIAL_SET": "material_",
    "NEUMANN_SET": "neumann_",
    "DIRICHLET_SET": "dirichlet_",
}
_GLOBAL_ID: str = "GLOBAL_ID"
# A dense tag one owner declares with a type the other cannot share is
# written for the elements under a suffixed name, and the declaration
# remembers the name it stands for; so is one named like a tag the sets
# themselves are written with, or a mesh-wide value named like a dense tag.
_CELL_SUFFIX: str = "__cell"
_ATTR_SUFFIX: str = "__attr"
_GLOBAL_SUFFIX: str = "__global"
_STANDS_FOR: str = "polyxios_name"
_SET_TAGS: frozenset[str] = frozenset({_NAME_TAG, *_SET_KINDS})


# ----- reading ----------------------------------------------------------------


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read an H5M file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.h5m`` file, or an open binary file object.
    lazy
        Not supported; raises LazyReadError when True.
    **opts
        None are taken; any given are warned about and ignored.

    Returns
    -------
    PolyData
        The mesh. Dense tags on the nodes are ``vertex_attrs``, dense tags
        on the elements ``element_attrs`` - a tag one element type carries
        and another does not is NaN on the other - and each meshset is a
        tag group over the nodes and the elements it holds, an empty set
        an empty group over the entities its kind says. Mesh-wide (global)
        tags with a value are ``global_attrs``. A sparse tag over the nodes
        or the elements themselves, rather than over sets, is not read.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    UnsupportedFormatError
        If h5py is not installed. The message names the extra that installs it.
    CodecError
        If the file is not HDF5, holds no ``tstt`` tree, a connectivity
        names a node the file does not hold, or a sparse tag's values are
        not one per id.

    Warns
    -----
    UserWarning
        For each element group MOAB names and polyxios does not - polygons of
        MOAB's own layout, polyhedra, knives - which is skipped; once for the
        sparse tags over entities, which are not read; once for a set-kind
        tag whose values are neither numbers nor text.
    """
    name = source_name(path)
    refuse_lazy(lazy, fmt=EXTENSION, name=name)
    warn_unknown_opts(opts, fmt=EXTENSION, what="read")
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        ctx = {"name": name, "file_size": source_size(path)}
        tstt = child(handle, "tstt", name=name, where="/")
        nodes = child(tstt, "nodes", name=name, where="tstt")
        coords = as_array(
            child(nodes, "coordinates", name=name, where="tstt/nodes"),
            name=name,
            where="tstt/nodes/coordinates",
        )
        if coords.ndim != 2 or coords.dtype.kind not in "iuf":
            raise CodecError(
                f"'{name}': 'tstt/nodes/coordinates' is not an (n, dim) block."
            )
        n_verts = len(coords)
        validate_header(n_verts, 0, 0, ctx["file_size"], compressed=True)
        node_start = _start_id(nodes["coordinates"])
        vertices = np.zeros((n_verts, 3), dtype=np.float64)
        vertices[:, : min(3, coords.shape[1])] = coords[:, :3]

        blocks, ranges = _read_elements(tstt, n_verts, node_start, ctx)
        n_elems = sum(len(cells) for _, _, cells in blocks)
        # Handle id -> mesh index, for the nodes and for the elements.
        node_lookup = (node_start, n_verts)

        declared = tstt["tags"] if "tags" in tstt else None
        vertex_attrs = _dense_tags(nodes, n_verts, name, "tstt/nodes", declared)
        element_attrs = _element_tags(tstt, blocks, n_elems, name, declared)
        vertex_tags: dict[str, np.ndarray] = {}
        element_tags: dict[str, np.ndarray] = {}
        _read_sets(tstt, node_lookup, ranges, n_elems, vertex_tags, element_tags, ctx)
        global_attrs = _global_tags(tstt, name)
    for attrs, count in ((vertex_attrs, n_verts), (element_attrs, n_elems)):
        ids = attrs.get(_GLOBAL_ID)
        if ids is not None and ids.ndim == 1 and ids.dtype.kind in "iu":
            del attrs[_GLOBAL_ID]
            attrs.update(record_ids(ids, count=count))
    return make_polydata(
        vertices,
        [(ptype, cells) for ptype, _, cells in blocks],
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _start_id(node: Any) -> int:
    value = node.attrs.get("start_id")
    return int(np.asarray(value).ravel()[0]) if value is not None else 1


def _read_elements(
    tstt: Any, n_verts: int, node_start: int, ctx: dict[str, Any]
) -> tuple[list[tuple[str, str, np.ndarray]], list[tuple[int, int, int]]]:
    """Every element group by ascending polyxios type code, with its id range.

    Returns the blocks as ``(polyxios type, group name, cells)`` and, for
    each block, ``(start_id, count, mesh offset)`` so a handle id can be
    turned into a mesh index.
    """
    name = ctx["name"]
    if "elements" not in tstt:
        return [], []
    elements = tstt["elements"]
    unknown: list[str] = []
    found: list[tuple[int, str, Any]] = []
    for key in elements:
        ptype = _H5M_TO_POLYXIOS.get(key)
        if ptype is None:
            unknown.append(key)
            continue
        found.append((int(ELEMENT_TYPES[ptype]), key, elements[key]))
    if unknown:
        warnings.warn(
            f"{EXTENSION} read: element group(s) {sorted(unknown)} in '{name}'"
            " have no polyxios element type; those elements are skipped.",
            UserWarning,
            stacklevel=4,
        )
    found.sort(key=lambda item: item[0])
    blocks: list[tuple[str, str, np.ndarray]] = []
    ranges: list[tuple[int, int, int]] = []
    offset = 0
    for code, key, group in found:
        ptype = ELEMENT_TYPES_INV[code]
        width = NODES_PER_ELEMENT[ptype]
        where = f"tstt/elements/{key}"
        conn = as_array(
            child(group, "connectivity", name=name, where=where),
            name=name,
            where=f"{where}/connectivity",
        )
        if conn.dtype.kind not in "iu" or conn.ndim != 2 or conn.shape[1] != width:
            raise CodecError(
                f"'{name}': '{where}/connectivity' is not an (n, {width}) table."
            )
        n_cells = len(conn)
        validate_header(
            n_verts, n_cells, n_cells * width, ctx["file_size"], compressed=True
        )
        cells = conn.astype(np.int64) - node_start
        if n_cells and (cells.min() < 0 or cells.max() >= n_verts):
            raise CodecError(
                f"'{name}': '{where}/connectivity' names a node outside the"
                f" {n_verts} the file holds."
            )
        order = _READ_ORDER.get(ptype)
        if order is not None:
            cells = cells[:, list(order)]
        blocks.append((ptype, key, cells))
        ranges.append((_start_id(group["connectivity"]), n_cells, offset))
        offset += n_cells
    return blocks, ranges


def _dense_tags(
    owner: Any, count: int, name: str, where: str, declared: Any
) -> dict[str, np.ndarray]:
    """The dense tags of one owner, under the names they stand for."""
    out: dict[str, np.ndarray] = {}
    if "tags" not in owner:
        return out
    for key in owner["tags"]:
        node = owner["tags"][key]
        if not hasattr(node, "shape"):
            continue
        values = as_array(node, name=name, where=f"{where}/tags/{key}")
        if values.dtype.kind in "VSU":
            continue
        if values.ndim == 0 or values.shape[0] != count:
            warnings.warn(
                f"{EXTENSION} read: tag '{where}/tags/{key}' in '{name}' holds"
                f" {values.shape[0] if values.ndim else 1} values for {count}"
                " entities; skipped.",
                UserWarning,
                stacklevel=5,
            )
            continue
        out[_stands_for(declared, key)] = values
    return out


def _stands_for(declared: Any, key: str) -> str:
    """The name a tag was written for, when its declaration remembers one."""
    if declared is not None and key in declared:
        return attr_text(declared[key], _STANDS_FOR) or key
    return key


def _element_tags(
    tstt: Any,
    blocks: list[tuple[str, str, np.ndarray]],
    n_elems: int,
    name: str,
    declared: Any,
) -> dict[str, np.ndarray]:
    """Dense tags over the elements, laid over every block, NaN where a block lacks one."""
    parts: dict[str, list[tuple[int, np.ndarray]]] = {}
    offset = 0
    for _, key, cells in blocks:
        group = tstt["elements"][key]
        for tag_name, values in _dense_tags(
            group, len(cells), name, f"tstt/elements/{key}", declared
        ).items():
            parts.setdefault(tag_name, []).append((offset, values))
        offset += len(cells)
    out: dict[str, np.ndarray] = {}
    for tag_name, pieces in parts.items():
        shapes = {v.shape[1:] for _, v in pieces}
        if len(shapes) != 1:
            warnings.warn(
                f"{EXTENSION} read: tag {tag_name!r} in '{name}' has a different"
                f" shape on each element type ({sorted(shapes)}); skipped.",
                UserWarning,
                stacklevel=5,
            )
            continue
        trailing = shapes.pop()
        covered = sum(len(v) for _, v in pieces)
        dtype = np.result_type(*(v.dtype for _, v in pieces))
        if covered != n_elems and dtype.kind != "f":
            dtype = np.dtype(np.float64)
        fill = np.nan if dtype.kind == "f" else 0
        arr = np.full((n_elems, *trailing), fill, dtype=dtype)
        for offset, values in pieces:
            arr[offset : offset + len(values)] = values
        out[tag_name] = arr
    return out


def _global_tags(tstt: Any, name: str) -> dict[str, Any]:
    """Mesh-wide tags: the ones declared with a value and no entity."""
    out: dict[str, Any] = {}
    if "tags" not in tstt:
        return out
    for key in tstt["tags"]:
        group = tstt["tags"][key]
        if hasattr(group, "shape") or "global" not in group.attrs:
            continue
        value = np.asarray(group.attrs["global"])
        stands_for = attr_text(group, _STANDS_FOR) or key
        if value.dtype.kind in "biuf":
            out[stands_for] = (
                value.reshape(-1)[0] if value.ndim <= 1 and value.size == 1 else value
            )
        else:
            text = text_of(value if value.ndim == 0 else value.reshape(-1)[0])
            if text is not None:
                out[stands_for] = text
    return out


def _read_sets(
    tstt: Any,
    node_lookup: tuple[int, int],
    ranges: list[tuple[int, int, int]],
    n_elems: int,
    vertex_tags: dict[str, np.ndarray],
    element_tags: dict[str, np.ndarray],
    ctx: dict[str, Any],
) -> None:
    name = ctx["name"]
    if "sets" not in tstt or "list" not in tstt["sets"]:
        return
    sets = tstt["sets"]
    table = as_array(sets["list"], name=name, where="tstt/sets/list")
    if table.ndim != 2 or table.shape[1] < 4 or table.dtype.kind not in "iu":
        raise CodecError(f"'{name}': 'tstt/sets/list' is not an (n, 4) table.")
    table = table.astype(np.int64)
    n_sets = len(table)
    contents = (
        as_array(sets["contents"], name=name, where="tstt/sets/contents")
        .ravel()
        .astype(np.int64)
        if "contents" in sets
        else np.empty(0, dtype=np.int64)
    )
    set_start = _start_id(sets["list"])
    labels = _set_labels(tstt, set_start, n_sets, name)
    _warn_sparse_on_entities(tstt, set_start, n_sets, name)
    node_start, n_verts = node_lookup
    empty = np.empty(0, dtype=np.int64)
    begin = 0
    for k in range(n_sets):
        end = int(table[k, 0]) + 1
        flags = int(table[k, 3])
        piece = contents[begin:end] if end > begin else empty
        begin = max(end, begin)
        found = labels.get(k)
        if found is None:
            continue
        label, kind = found
        if not piece.size:
            # An empty set says nothing about what it would hold; its kind
            # does - the writer numbers a vertex group DIRICHLET.
            holder = vertex_tags if kind == "DIRICHLET_SET" else element_tags
            holder.setdefault(label, empty)
            continue
        ids = _expand(piece, ctx) if flags & _RANGE_BIT else piece
        picked_nodes = (
            ids[(ids >= node_start) & (ids < node_start + n_verts)] - node_start
        )
        picked_elems = _element_indices(ids, ranges, n_elems)
        if picked_nodes.size:
            vertex_tags[label] = np.union1d(vertex_tags.get(label, empty), picked_nodes)
        if picked_elems.size:
            element_tags[label] = np.union1d(
                element_tags.get(label, empty), picked_elems
            )


def _warn_sparse_on_entities(tstt: Any, set_start: int, n_sets: int, name: str) -> None:
    """Warn once for the sparse tags whose ids name nodes or elements, not sets."""
    if "tags" not in tstt:
        return
    over_entities: list[str] = []
    for key in tstt["tags"]:
        group = tstt["tags"][key]
        if hasattr(group, "shape") or "id_list" not in group:
            continue
        ids = np.asarray(group["id_list"][()]).ravel()
        if ids.size and ids.dtype.kind in "iu":
            ids = ids.astype(np.int64)
            if ((ids < set_start) | (ids >= set_start + n_sets)).any():
                over_entities.append(key)
    if over_entities:
        warnings.warn(
            f"{EXTENSION} read: sparse tag(s) {sorted(over_entities)} in '{name}'"
            " are set over nodes or elements rather than sets; not read.",
            UserWarning,
            stacklevel=5,
        )


def _expand(piece: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    """Range-compressed contents are (start, count) pairs."""
    if piece.size % 2:
        raise CodecError(
            f"'{ctx['name']}': a range-compressed set has an odd contents length."
        )
    starts = piece[0::2]
    counts = piece[1::2]
    if counts.size and counts.min() < 0:
        raise CodecError(
            f"'{ctx['name']}': a range-compressed set holds a negative count."
        )
    total = int(counts.sum()) if counts.size else 0
    validate_header(0, total, 0, ctx["file_size"], compressed=True)
    if not total:
        return piece[:0]
    # Every run at once: each output slot takes its run's start plus its
    # position inside the run.
    first = np.cumsum(counts) - counts
    return np.repeat(starts - first, counts) + np.arange(total, dtype=np.int64)


def _element_indices(
    ids: np.ndarray, ranges: list[tuple[int, int, int]], n_elems: int
) -> np.ndarray:
    picked = []
    for start, count, offset in ranges:
        inside = ids[(ids >= start) & (ids < start + count)]
        if inside.size:
            picked.append(inside - start + offset)
    if not picked:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(picked)
    return np.unique(out[(out >= 0) & (out < n_elems)])


def _set_labels(
    tstt: Any, set_start: int, n_sets: int, name: str
) -> dict[int, tuple[str, str | None]]:
    """A name per set, with the kind that numbered it.

    The name is the set's NAME tag, else its material, Neumann or Dirichlet
    number; the kind is the set-kind tag it carries, None when it has none.
    """
    labels: dict[int, tuple[str, str | None]] = {}
    if "tags" not in tstt:
        return labels
    tags = tstt["tags"]
    unreadable: list[str] = []
    for kind in reversed(_SET_KINDS):
        for k, value in _sparse_on_sets(tags, kind, set_start, n_sets, name):
            number = (
                int(value) if isinstance(value, (int, np.integer)) else text_of(value)
            )
            if number is None:
                unreadable.append(kind)
                labels.setdefault(k, ("", kind))
                continue
            labels[k] = (f"{_SET_PREFIX[kind]}{number}", kind)
    for k, value in _sparse_on_sets(tags, _NAME_TAG, set_start, n_sets, name):
        text = _opaque_text(value)
        if text:
            labels[k] = (text, labels[k][1] if k in labels else None)
    if unreadable:
        warnings.warn(
            f"{EXTENSION} read: tag(s) {sorted(set(unreadable))} in '{name}' hold"
            " values that are neither numbers nor text; the sets they number"
            " and no NAME names are skipped.",
            UserWarning,
            stacklevel=5,
        )
    return {k: found for k, found in labels.items() if found[0]}


def _sparse_on_sets(
    tags: Any, key: str, set_start: int, n_sets: int, name: str
) -> list[tuple[int, Any]]:
    if key not in tags or "id_list" not in tags[key] or "values" not in tags[key]:
        return []
    group = tags[key]
    where = f"tstt/tags/{key}"
    ids = (
        as_array(group["id_list"], name=name, where=f"{where}/id_list")
        .ravel()
        .astype(np.int64)
    )
    values = np.asarray(group["values"][()])
    if values.ndim == 0:
        raise CodecError(f"'{name}': '{where}/values' is one value, not one per id.")
    if len(values) != len(ids):
        return []
    ordinal = ids - set_start
    inside = np.flatnonzero((ordinal >= 0) & (ordinal < n_sets))
    return [(int(ordinal[i]), values[i]) for i in inside]


def _opaque_text(value: Any) -> str | None:
    if isinstance(value, np.void):
        value = value.tobytes()
    if isinstance(value, (bytes, np.bytes_)):
        return (
            bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        )
    return text_of(value)


# ----- writing ----------------------------------------------------------------


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as an H5M file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object.
    **opts
        ``compression`` and ``compression_opts`` are passed to h5py for the
        datasets - ``compression="gzip"`` with ``compression_opts=4`` is the
        usual pair. ``global_ids`` adds a ``GLOBAL_ID`` tag numbering the
        nodes and the elements from one, which MOAB's partitioner wants;
        on by default, and ``original_ids`` are written under it when the
        mesh carries them.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If the mesh's vertices are not a coordinate block.

    Warns
    -----
    UserWarning
        Once per element type MOAB has no group for - those elements, and
        the values and tags on them, are not written; once for the
        attributes no dense tag can hold; once for the ``global_attrs`` that
        no numeric array can spell.

    Notes
    -----
    One group per element type under ``tstt/elements``, the types in
    ascending polyxios code order, each element's handle id following the
    last; a pixel and a voxel go out as the quad and the hexahedron they
    are. ``vertex_attrs`` and ``element_attrs`` are dense tags, each declared
    under ``tstt/tags`` with its type - one named like the ``NAME`` or
    set-kind tag the meshsets carry, or an element attribute whose type
    differs from the vertex attribute of its name, is declared under a
    suffixed name that says which it stands for; the tag groups are meshsets, one per
    group, each carrying a ``NAME`` tag with the group's name and - so MOAB's
    tools list them - a ``MATERIAL_SET`` number for an element group and a
    ``DIRICHLET_SET`` number for a vertex group. Numeric ``global_attrs`` are
    mesh tags with a ``global`` value.
    """
    h5py = require_h5py(fmt=EXTENSION, name=source_name(path), verb="writing")
    global_ids = bool(opts.pop("global_ids", True))
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write")
    output_dimension(poly, fmt=EXTENSION)
    numeric = globals_for_write(poly, reserved=frozenset({"was_2d"}), fmt=EXTENSION)
    n_verts = len(poly.vertices)
    n_elems = len(poly.element_types)
    blocks, dropped = type_blocks(poly, sizes=_WRITE_SIZES, fmt=EXTENSION)
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: element type(s)"
            f" {sorted(ELEMENT_TYPES_INV.get(c, f'type_{c}') for c in dropped)}"
            " have no MOAB element type; those elements and the values and tags"
            " on them are not written.",
            UserWarning,
            stacklevel=2,
        )
    kept = kept_index(blocks)
    element_tags = reindexed_tags(poly.element_tags, kept, n_elems)
    # GLOBAL_ID numbers the written order, so a mesh already grouped by type
    # reads back numbered 1..n and carries no original_ids it never had.
    element_ids = np.zeros(n_elems, dtype=np.int32)
    element_ids[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    node_arrays = _tag_arrays(
        poly.vertex_attrs,
        n_verts,
        global_ids,
        np.arange(1, n_verts + 1, dtype=np.int32),
    )
    element_arrays = _tag_arrays(poly.element_attrs, n_elems, global_ids, element_ids)
    node_spelled, element_spelled = _spellings(node_arrays, element_arrays)
    enum_dtype = h5py.enum_dtype(_MOAB_ENUM, basetype="i")

    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        tstt = handle.create_group("tstt")
        tstt["elemtypes"] = enum_dtype
        stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
        tstt.create_dataset(
            "history",
            data=np.array(
                [b"polyxios", stamp.encode()], dtype=h5py.string_dtype("ascii")
            ),
        )
        declared = tstt.create_group("tags")

        next_id = 1
        nodes = tstt.create_group("nodes")
        coords = nodes.create_dataset(
            "coordinates",
            data=np.ascontiguousarray(
                np.asarray(poly.vertices, dtype=np.float64)[:, :3]
            ),
            **(dataset_kw if n_verts > 1 else {}),
        )
        coords.attrs.create("start_id", next_id, dtype=np.int64)
        node_start = next_id
        next_id += n_verts
        _write_dense(nodes, declared, node_arrays, node_spelled, dataset_kw)

        elements = tstt.create_group("elements")
        element_start = next_id
        for code, index, conn in blocks:
            ptype = ELEMENT_TYPES_INV[code]
            h5m_name, order = _h5m_spelling(ptype)
            if order is not None:
                conn = conn[:, list(order)]
            group = elements.create_group(h5m_name)
            group.attrs.create(
                "element_type",
                _MOAB_ENUM[h5m_name.rstrip("0123456789")],
                dtype=enum_dtype,
            )
            table = group.create_dataset(
                "connectivity",
                data=np.ascontiguousarray(conn + node_start).astype(np.uint64),
                **(dataset_kw if conn.size > 1 else {}),
            )
            table.attrs.create("start_id", next_id, dtype=np.int64)
            _write_dense(
                group,
                declared,
                {key: arr[index] for key, arr in element_arrays.items()},
                element_spelled,
                dataset_kw,
            )
            next_id += len(index)

        sets = tstt.create_group("sets")
        sets.create_group("tags")
        _write_sets(
            sets,
            declared,
            poly.vertex_tags,
            element_tags,
            n_verts,
            node_start,
            element_start,
            next_id,
            dataset_kw,
        )
        n_sets = len(poly.vertex_tags) + len(element_tags)
        next_id += n_sets
        taken = set(declared)
        for key, arr in numeric.items():
            spelled = _free_name(key + _GLOBAL_SUFFIX, taken) if key in taken else key
            group = declared.create_group(spelled)
            group["type"] = arr.dtype
            group.attrs.create("class", _MESH, dtype=np.int32)
            group.attrs.create("global", np.ascontiguousarray(arr))
            if spelled != key:
                group.attrs[_STANDS_FOR] = key
        tstt.attrs.create("max_id", next_id - 1, dtype=np.uint64)


def _h5m_spelling(ptype: str) -> tuple[str, tuple[int, ...] | None]:
    if ptype in _LATTICE:
        return _LATTICE[ptype]
    return _POLYXIOS_TO_H5M[ptype], _WRITE_ORDER.get(ptype)


def _tag_arrays(
    attrs: dict[str, np.ndarray], count: int, global_ids: bool, ids: np.ndarray
) -> dict[str, np.ndarray]:
    """The dense tags an owner writes, over every entity of the mesh.

    Parameters
    ----------
    attrs
        The owner's attributes; one that is not a number or a vector per
        entity is dropped, all of them named in one warning.
    count
        How many entities the owner holds.
    global_ids
        Whether a ``GLOBAL_ID`` tag is wanted.
    ids
        The numbering ``GLOBAL_ID`` takes when the mesh carries no
        ``original_ids`` of its own.
    """
    out: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for key, values in attrs.items():
        arr = np.asarray(values)
        if arr.dtype.kind == "b":
            arr = arr.astype(np.int32)
        if (
            arr.dtype.kind not in "iuf"
            or arr.ndim == 0
            or arr.shape[0] != count
            or arr.ndim > 2
        ):
            dropped.append(key)
            continue
        if key == IDS_KEY:
            if global_ids:
                out[_GLOBAL_ID] = arr.astype(np.int32)
            continue
        out[key] = arr
    if global_ids and _GLOBAL_ID not in out:
        out[_GLOBAL_ID] = ids
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: attribute(s) {sorted(dropped)} are not one number"
            " or one vector per entity; dropped.",
            UserWarning,
            stacklevel=3,
        )
    return out


def _tag_dtype(arr: np.ndarray) -> np.dtype:
    """MOAB's record type for a dense tag: a k-tuple per entity, not an (n, k) table."""
    return np.dtype((arr.dtype, (arr.shape[1],))) if arr.ndim == 2 else arr.dtype


def _spellings(
    node_arrays: dict[str, np.ndarray], element_arrays: dict[str, np.ndarray]
) -> tuple[dict[str, str], dict[str, str]]:
    """The dense tags that cannot go out under their own name, per owner.

    A MOAB tag has one type, so a vertex attribute and an element attribute
    of one name and different types are two tags; the elements' goes out
    under a suffixed name. An attribute named like the ``NAME`` or set-kind
    tag the meshsets are written with would share its declaration with a
    sparse tag of another type, so it is suffixed too. Every suffixed
    declaration says which name it stands for, and the reader undoes it.
    """
    taken = set(node_arrays) | set(element_arrays) | _SET_TAGS
    nodes: dict[str, str] = {}
    for key in node_arrays:
        if key in _SET_TAGS:
            nodes[key] = _free_name(key + _ATTR_SUFFIX, taken)
    elements: dict[str, str] = {}
    for key, arr in element_arrays.items():
        other = node_arrays.get(key)
        if other is not None and _tag_dtype(other) == _tag_dtype(arr):
            if key in nodes:
                elements[key] = nodes[key]
        elif other is not None:
            elements[key] = _free_name(key + _CELL_SUFFIX, taken)
        elif key in _SET_TAGS:
            elements[key] = _free_name(key + _ATTR_SUFFIX, taken)
    return nodes, elements


def _free_name(wanted: str, taken: set[str]) -> str:
    """``wanted`` with underscores until no other tag spells it; then taken."""
    while wanted in taken:
        wanted += "_"
    taken.add(wanted)
    return wanted


def _write_dense(
    owner: Any,
    declared: Any,
    arrays: dict[str, np.ndarray],
    renamed: dict[str, str],
    dataset_kw: dict[str, Any],
) -> None:
    if not arrays:
        return
    tags = owner.create_group("tags")
    for key, arr in arrays.items():
        spelled = renamed.get(key, key)
        dtype = _tag_dtype(arr)
        if arr.ndim == 2:
            node = tags.create_dataset(
                spelled,
                (len(arr),),
                dtype=dtype,
                **(dataset_kw if arr.size > 1 else {}),
            )
            node[...] = arr
        else:
            tags.create_dataset(
                spelled, data=arr, **(dataset_kw if arr.size > 1 else {})
            )
        if spelled not in declared:
            group = declared.create_group(spelled)
            group["type"] = dtype
            group.attrs.create("class", _DENSE, dtype=np.int32)
            if spelled != key:
                group.attrs[_STANDS_FOR] = key


def _write_sets(
    sets: Any,
    declared: Any,
    vertex_tags: dict[str, np.ndarray],
    element_tags: dict[str, np.ndarray],
    n_verts: int,
    node_start: int,
    element_start: int,
    set_start: int,
    dataset_kw: dict[str, Any],
) -> None:
    """One meshset per tag group, named by a NAME tag and numbered by its kind.

    The element groups are numbered on from one another, so an element's
    handle is the first element's plus its position in the written order -
    which is the order ``element_tags`` were renumbered onto.
    """
    rows: list[list[int]] = []
    contents: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[tuple[str, int]] = []
    total = 0
    for k, (tag_name, members) in enumerate(vertex_tags.items()):
        ids = member_indices(members, n_verts).astype(np.int64) + node_start
        total += ids.size
        rows.append([total - 1, -1, -1, _SET_FLAG])
        contents.append(ids)
        names.append(str(tag_name))
        kinds.append(("DIRICHLET_SET", k + 1))
    for k, (tag_name, members) in enumerate(element_tags.items()):
        ids = np.asarray(members, dtype=np.int64) + element_start
        total += ids.size
        rows.append([total - 1, -1, -1, _SET_FLAG])
        contents.append(ids)
        names.append(str(tag_name))
        kinds.append(("MATERIAL_SET", k + 1))
    if not rows:
        return
    table = sets.create_dataset("list", data=np.asarray(rows, dtype=np.int64))
    table.attrs.create("start_id", set_start, dtype=np.int64)
    flat = (
        np.concatenate(contents).astype(np.uint64)
        if total
        else np.empty(0, dtype=np.uint64)
    )
    sets.create_dataset("contents", data=flat, **(dataset_kw if flat.size > 1 else {}))
    sets.create_dataset("children", data=np.empty(0, dtype=np.uint64))
    sets.create_dataset("parents", data=np.empty(0, dtype=np.uint64))
    handles = np.arange(set_start, set_start + len(rows), dtype=np.uint64)

    name_dtype = np.dtype(f"V{_NAME_WIDTH}")
    values = np.zeros(len(names), dtype=name_dtype)
    for k, text in enumerate(names):
        raw = text.encode("utf-8")[: _NAME_WIDTH - 1]
        values[k] = np.frombuffer(raw.ljust(_NAME_WIDTH, b"\x00"), dtype=name_dtype)[0]
    _write_sparse(declared, _NAME_TAG, handles, values, name_dtype)
    for kind in _SET_KINDS:
        picked = [
            (h, number) for h, (found, number) in zip(handles, kinds) if found == kind
        ]
        if picked:
            ids = np.asarray([h for h, _ in picked], dtype=np.uint64)
            numbers = np.asarray([n for _, n in picked], dtype=np.int32)
            _write_sparse(declared, kind, ids, numbers, np.dtype(np.int32))


def _write_sparse(
    declared: Any, key: str, ids: np.ndarray, values: np.ndarray, dtype: np.dtype
) -> None:
    group = declared.create_group(key)
    group["type"] = dtype
    group.attrs.create("class", _SPARSE, dtype=np.int32)
    group.create_dataset("id_list", data=ids)
    group.create_dataset("values", data=values)
