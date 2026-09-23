"""CGNS codec - the CFD General Notation System, in its HDF5 form.

A CGNS file is a tree of typed nodes. In the HDF5 mapping every node is a
group carrying its ``name``, its ``label`` (the SIDS type, ``Zone_t`` and the
like) and the ``type`` of its value (``I4``, ``R8``, ``C1``, ``MT`` for none),
and the value itself is a dataset called `` data`` - with the leading space -
whose dimensions are the CGNS ones reversed, since CGNS counts in Fortran
order. The mesh sits under a ``CGNSBase_t`` as one or more ``Zone_t`` nodes:
an unstructured zone has its coordinates under ``GridCoordinates`` and its
cells in ``Elements_t`` sections, each section one element type - or
``MIXED`` - over a consecutive range of element numbers; a structured zone
has only its coordinates, on a lattice, and its cells are implied.

Every zone is read, and when there are several they are merged into the one
mesh, each zone's elements tagged with the zone's name - the first thing a
multi-zone file is opened for is usually the zones themselves. ``ZoneBC_t``
boundary conditions become tag groups over the elements or the vertices
their ``GridLocation`` says, ``FlowSolution_t`` arrays become attributes on
the same terms, and a ``UserDefinedData_t`` holding ``DataArray_t`` nodes
becomes ``global_attrs``. Node order in a section follows the SIDS, which
puts the mid-edge nodes of the quadratic hexahedron and wedge in another
order than VTK does; they are permuted on the way in and out.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from polyxios import transforms
from polyxios._dimension import mark_2d, output_dimension, pad_to_3d
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV, NODES_PER_ELEMENT
from polyxios._globals import globals_for_write, text_for_write
from polyxios._ids import IDS_KEY
from polyxios._io import Source, source_name, source_size
from polyxios._tags import member_indices
from polyxios._types import _DIMENSION_BY_CODE, PolyData, make_polydata
from polyxios.codecs._hdf5 import (
    _h5py,
    as_array,
    attr_text,
    dataset_options,
    kept_index,
    open_hdf5_read,
    open_hdf5_write,
    refuse_lazy,
    reindexed_tags,
    text_of,
    type_blocks,
    warn_unknown_opts,
)
from polyxios.codecs._vtk_xml import structured_cells
from polyxios.exceptions import CodecError, UnknownElementTypeError
from polyxios.validate import validate_header

EXTENSION: str = ".cgns"

# The SIDS element type codes polyxios can hold, and what they read as.
_CGNS_TO_POLYXIOS: dict[int, str] = {
    2: "vertex",
    3: "line",
    4: "quadratic_edge",
    5: "triangle",
    6: "quadratic_triangle",
    7: "quad",
    8: "quadratic_quad",
    9: "biquadratic_quad",
    10: "tetra",
    11: "quadratic_tetra",
    12: "pyramid",
    14: "wedge",
    15: "quadratic_wedge",
    16: "biquadratic_quadratic_wedge",
    17: "hexahedron",
    18: "quadratic_hexahedron",
    19: "triquadratic_hexahedron",
    21: "quadratic_pyramid",
    22: "polygon",
    24: "cubic_line",
}
_POLYXIOS_TO_CGNS: dict[str, int] = {v: k for k, v in _CGNS_TO_POLYXIOS.items()}
_CGNS_NAMES: dict[int, str] = {
    2: "NODE",
    3: "BAR_2",
    4: "BAR_3",
    5: "TRI_3",
    6: "TRI_6",
    7: "QUAD_4",
    8: "QUAD_8",
    9: "QUAD_9",
    10: "TETRA_4",
    11: "TETRA_10",
    12: "PYRA_5",
    13: "PYRA_14",
    14: "PENTA_6",
    15: "PENTA_15",
    16: "PENTA_18",
    17: "HEXA_8",
    18: "HEXA_20",
    19: "HEXA_27",
    20: "MIXED",
    21: "PYRA_13",
    22: "NGON_n",
    23: "NFACE_n",
    24: "BAR_4",
}
_MIXED: int = 20
_NGON: int = 22
_NFACE: int = 23
# Node counts the SIDS fixes for the types it names and polyxios does not;
# a MIXED section has to be walked past them.
_CGNS_SIZES: dict[int, int] = {
    13: 14,
    25: 9,
    26: 10,
    27: 12,
    28: 16,
    29: 16,
    30: 20,
    31: 21,
    32: 29,
    33: 30,
    34: 24,
    35: 38,
    36: 40,
    37: 32,
    38: 56,
    39: 64,
    40: 5,
    41: 12,
    42: 15,
    43: 16,
    44: 25,
    45: 22,
    46: 34,
    47: 35,
    48: 29,
    49: 50,
    50: 55,
    51: 33,
    52: 66,
    53: 75,
    54: 44,
    55: 98,
    56: 125,
}
# Mid-edge node order. VTK lists a hexahedron's bottom edges, then its top
# edges, then the vertical ones; the SIDS puts the vertical edges before the
# top. The wedge likewise. Entry ``i`` is the SIDS position of the node VTK
# puts at ``i``; the 27-node hexahedron adds the face centres, which the
# SIDS lists bottom, front, right, back, left, top and VTK by axis.
_HEX_EDGES: tuple[int, ...] = (8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15)
_READ_ORDER: dict[str, tuple[int, ...]] = {
    "quadratic_hexahedron": (*range(8), *_HEX_EDGES),
    "triquadratic_hexahedron": (*range(8), *_HEX_EDGES, 24, 22, 21, 23, 20, 25, 26),
    "quadratic_wedge": (*range(6), 6, 7, 8, 12, 13, 14, 9, 10, 11),
    "biquadratic_quadratic_wedge": (
        *range(6),
        6,
        7,
        8,
        12,
        13,
        14,
        9,
        10,
        11,
        15,
        16,
        17,
    ),
}
_WRITE_ORDER: dict[str, tuple[int, ...]] = {
    name: tuple(order.index(i) for i in range(len(order)))
    for name, order in _READ_ORDER.items()
}
_LATTICE: dict[str, tuple[int, tuple[int, ...]]] = {
    "pixel": (7, (0, 1, 3, 2)),
    "voxel": (17, (0, 1, 3, 2, 4, 5, 7, 6)),
}
_WRITE_SIZES: dict[int, int] = {
    int(ELEMENT_TYPES[name]): NODES_PER_ELEMENT[name]
    for name in (*_POLYXIOS_TO_CGNS, *_LATTICE)
    if NODES_PER_ELEMENT[name] > 0
}
_POLYGON: int = int(ELEMENT_TYPES["polygon"])

_ZONE_NAME_KEY: str = "zone_name"
_BASE_NAME_KEY: str = "base_name"
_RESERVED_GLOBALS: frozenset[str] = frozenset(
    {_ZONE_NAME_KEY, _BASE_NAME_KEY, "was_2d"}
)
_USER_DATA: str = "polyxios"
_LABEL_WIDTH: int = 33
_INT32_MAX: int = 2**31 - 1


# ----- reading ----------------------------------------------------------------


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a CGNS file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.cgns`` file, or an open binary file object.
    lazy
        Not supported; raises LazyReadError when True.
    **opts
        ``zone`` names the zone to read when the base holds several; without
        it every zone is merged into one mesh, each zone's elements tagged
        with the zone's name. ``base`` names the base the same way; the
        first is read without it.

    Returns
    -------
    PolyData
        The mesh. Boundary conditions are tag groups, over the vertices when
        their ``GridLocation`` is ``Vertex`` and over the elements otherwise
        - an ``ElementList`` or ``ElementRange``, as CGNS 2 spelled a
        condition on faces, names elements whatever the location says; on
        a structured zone a ``PointRange`` or ``PointList`` of index
        triples is spanned over the lattice at ``Vertex`` or ``CellCenter``,
        while a face-centred one has no element to land on and is skipped.
        Flow solutions are ``vertex_attrs`` or ``element_attrs`` on the same
        terms: a solution over the elements of its location's dimension
        alone - cell-centred over the volume cells, face-centred over the
        faces - or one with a ``PointList`` or ``PointRange`` lands on those
        entities, and what several solutions leave uncovered is NaN.
        ``DataArray_t`` nodes under a ``UserDefinedData_t`` are
        ``global_attrs``, as are the base's and the zone's names.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    UnsupportedFormatError
        If h5py is not installed. The message names the extra that installs it.
    CodecError
        If the file is not HDF5, holds no base or zone, names a base or zone
        it does not have, a zone declares more or fewer vertices than its
        coordinates hold, or a section's range, connectivity and type do not
        agree - a range starting below one or overlapping another included.
    UnknownElementTypeError
        If a section carries an element type code the SIDS does not define.

    Warns
    -----
    UserWarning
        For each element type the SIDS names and polyxios does not - the
        polyhedra of an ``NFACE_n`` section, the 14-node pyramid - which is
        skipped, and for a solution array whose length fits neither the
        vertices nor the elements of its location.
    """
    name = source_name(path)
    refuse_lazy(lazy, fmt=EXTENSION, name=name)
    wanted_zone = opts.pop("zone", None)
    wanted_base = opts.pop("base", None)
    warn_unknown_opts(opts, fmt=EXTENSION, what="read")
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        ctx = {"name": name, "file_size": source_size(path)}
        bases = _children(handle, "CGNSBase_t")
        if not bases:
            raise CodecError(f"'{name}': no CGNSBase_t node; not a CGNS mesh.")
        if wanted_base is not None:
            picked = [(k, b) for k, b in bases if k == wanted_base]
            if not picked:
                raise CodecError(
                    f"'{name}': no base named {wanted_base!r}; the file holds"
                    f" {[k for k, _ in bases]}."
                )
            bases = picked
        base_name, base = bases[0]
        dims = (
            as_array(base[" data"], name=name, where=f"{base_name}/ data").ravel()
            if " data" in base
            else np.array([3, 3])
        )
        cell_dim, phys_dim = (int(dims[0]), int(dims[1])) if dims.size >= 2 else (3, 3)
        zones = _children(base, "Zone_t")
        if not zones:
            raise CodecError(f"'{name}': base {base_name!r} holds no Zone_t.")
        if wanted_zone is not None:
            picked = [(k, z) for k, z in zones if k == wanted_zone]
            if not picked:
                raise CodecError(
                    f"'{name}': no zone named {wanted_zone!r}; base {base_name!r}"
                    f" holds {[k for k, _ in zones]}."
                )
            zones = picked
        globals_ = _read_user_data(base, ctx)
        globals_[_BASE_NAME_KEY] = base_name
        meshes = [
            _read_zone(zone, zone_name, base_name, cell_dim, phys_dim, ctx)
            for zone_name, zone in zones
        ]
    if len(meshes) == 1:
        poly = meshes[0]
        return PolyData(
            vertices=poly.vertices,
            connectivity=poly.connectivity,
            offsets=poly.offsets,
            element_types=poly.element_types,
            vertex_attrs=poly.vertex_attrs,
            element_attrs=poly.element_attrs,
            vertex_tags=poly.vertex_tags,
            element_tags=poly.element_tags,
            global_attrs={**globals_, **poly.global_attrs},
        )
    tagged = []
    for poly, (zone_name, _) in zip(meshes, zones):
        tags = dict(poly.element_tags)
        tags[_unique_name(tags, zone_name, width=None)] = np.arange(
            len(poly.element_types), dtype=np.int64
        )
        tagged.append(
            PolyData(
                vertices=poly.vertices,
                connectivity=poly.connectivity,
                offsets=poly.offsets,
                element_types=poly.element_types,
                vertex_attrs=poly.vertex_attrs,
                element_attrs=poly.element_attrs,
                vertex_tags=poly.vertex_tags,
                element_tags=tags,
                global_attrs={
                    k: v for k, v in poly.global_attrs.items() if k != _ZONE_NAME_KEY
                },
            )
        )
    merged = transforms.merge(*tagged)
    return PolyData(
        vertices=merged.vertices,
        connectivity=merged.connectivity,
        offsets=merged.offsets,
        element_types=merged.element_types,
        vertex_attrs=merged.vertex_attrs,
        element_attrs=merged.element_attrs,
        vertex_tags=merged.vertex_tags,
        element_tags=merged.element_tags,
        global_attrs={**globals_, **merged.global_attrs},
    )


def _children(node: Any, label: str) -> list[tuple[str, Any]]:
    """The child groups carrying this SIDS label, in name order."""
    found = []
    for key in node:
        sub = node[key]
        if hasattr(sub, "shape"):
            continue
        if attr_text(sub, "label") == label:
            found.append((attr_text(sub, "name") or key, sub))
    return found


def _first_child(node: Any, label: str) -> Any | None:
    found = _children(node, label)
    return found[0][1] if found else None


def _data(node: Any, ctx: dict[str, Any], where: str) -> np.ndarray | None:
    if " data" not in node:
        return None
    return as_array(node[" data"], name=ctx["name"], where=f"{where}/ data")


def _text_data(node: Any, ctx: dict[str, Any], where: str) -> str | None:
    """A ``C1`` node's value: the characters, stored as bytes."""
    values = _data(node, ctx, where)
    if values is None:
        return None
    if values.dtype.kind in "iu":
        return (
            bytes(values.ravel().astype(np.uint8))
            .decode("utf-8", errors="replace")
            .strip("\x00")
            .strip()
        )
    return text_of(values) if values.ndim == 0 else None


def _read_zone(
    zone: Any,
    zone_name: str,
    base_name: str,
    cell_dim: int,
    phys_dim: int,
    ctx: dict[str, Any],
) -> PolyData:
    name = ctx["name"]
    where = f"{base_name}/{zone_name}"
    zone_type = "Unstructured"
    type_node = _first_child(zone, "ZoneType_t")
    if type_node is not None:
        zone_type = _text_data(type_node, ctx, f"{where}/ZoneType") or zone_type
    size = _data(zone, ctx, where)
    if size is None:
        raise CodecError(f"'{name}': zone {zone_name!r} carries no size.")
    vertices, extents = _read_coordinates(zone, phys_dim, ctx, where)
    n_verts = len(vertices)
    _check_zone_size(size, n_verts, ctx, where)

    structured = zone_type == "Structured"
    if structured:
        blocks, element_index = _structured_blocks(extents, n_verts, ctx, where)
    else:
        blocks, element_index = _read_sections(zone, n_verts, ctx, where)
    n_elems = sum(len(cells) for _, cells in blocks)
    # The dimension of each element by mesh index, so a solution or a
    # condition at FaceCenter or CellCenter knows which elements it spans.
    dims = (
        _dimension_of(
            np.concatenate(
                [
                    np.full(len(cells), int(ELEMENT_TYPES[ptype]))
                    for ptype, cells in blocks
                ]
            )
        )
        if blocks
        else np.empty(0, dtype=np.int8)
    )
    lattice = extents if structured else None

    vertex_tags: dict[str, np.ndarray] = {}
    element_tags: dict[str, np.ndarray] = {}
    _read_boundary(
        zone, n_verts, element_index, lattice, vertex_tags, element_tags, ctx, where
    )
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    _read_solutions(
        zone,
        n_verts,
        n_elems,
        cell_dim,
        element_index,
        dims,
        lattice,
        vertex_attrs,
        element_attrs,
        ctx,
        where,
    )
    global_attrs: dict[str, Any] = {_ZONE_NAME_KEY: zone_name, **mark_2d(phys_dim)}
    return make_polydata(
        vertices,
        blocks,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _read_coordinates(
    zone: Any, phys_dim: int, ctx: dict[str, Any], where: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    name = ctx["name"]
    grid = _first_child(zone, "GridCoordinates_t")
    if grid is None:
        raise CodecError(f"'{name}': zone '{where}' has no GridCoordinates.")
    columns = []
    shape: tuple[int, ...] | None = None
    for axis in ("CoordinateX", "CoordinateY", "CoordinateZ"):
        if axis not in grid:
            break
        values = _data(grid[axis], ctx, f"{where}/GridCoordinates/{axis}")
        if values is None or values.dtype.kind not in "iuf":
            raise CodecError(
                f"'{name}': '{where}/GridCoordinates/{axis}' holds no numbers."
            )
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise CodecError(
                f"'{name}': '{where}/GridCoordinates/{axis}' is shaped"
                f" {values.shape} where CoordinateX is {shape}."
            )
        columns.append(values.ravel(order="C").astype(np.float64))
    if not columns or shape is None:
        raise CodecError(f"'{name}': zone '{where}' has no CoordinateX.")
    n_verts = int(np.prod(shape))
    validate_header(n_verts, 0, 0, ctx["file_size"], compressed=True)
    coords = np.column_stack(columns)
    dim = coords.shape[1]
    if dim < 3:
        coords = (
            pad_to_3d(coords, dim)
            if dim == 2
            else np.column_stack([coords[:, 0], np.zeros(n_verts), np.zeros(n_verts)])
        )
    # HDF5 holds the CGNS dimensions reversed, so a structured zone's
    # (ni, nj, nk) array reads as (nk, nj, ni): i is the fastest index.
    return coords, tuple(int(n) for n in reversed(shape))


def _check_zone_size(
    size: np.ndarray, n_verts: int, ctx: dict[str, Any], where: str
) -> None:
    """Refuse a zone whose declared vertex count is not its coordinates'.

    The zone's `` data`` is ``(IndexDimension, 3)`` in CGNS - vertex sizes,
    cell sizes, boundary vertex sizes - so ``(3, IndexDimension)`` here, and
    the vertex sizes multiply to the vertex count.
    """
    if size.dtype.kind not in "iu" or size.size % 3:
        return
    declared = int(np.prod(size.reshape(3, -1)[0]))
    if declared != n_verts:
        raise CodecError(
            f"'{ctx['name']}': zone '{where}' declares {declared} vertices and"
            f" holds coordinates for {n_verts}."
        )


def _structured_blocks(
    extents: tuple[int, ...], n_verts: int, ctx: dict[str, Any], where: str
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """The cells a lattice implies, and the mesh index of each cell number."""
    spans = [max(n - 1, 0) for n in extents] + [0] * (3 - len(extents))
    conn, per_cell, kind = structured_cells(*spans[:3])
    n_cells = len(conn) // per_cell if per_cell else 0
    validate_header(n_verts, n_cells, len(conn), ctx["file_size"], compressed=True)
    if not n_cells:
        return [], np.empty(0, dtype=np.int64)
    cells = conn.reshape(n_cells, per_cell).astype(np.int64)
    return [(kind, cells)], np.arange(n_cells, dtype=np.int64)


def _read_sections(
    zone: Any, n_verts: int, ctx: dict[str, Any], where: str
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """Every Elements_t section, in element-number order, as typed blocks.

    Returns the blocks and the mesh index of each element number from one
    (``-1`` for a skipped element), which is what a boundary condition or a
    solution with a ``PointList`` is indexed by.
    """
    name = ctx["name"]
    sections = []
    for key, node in _children(zone, "Elements_t"):
        header = _data(node, ctx, f"{where}/{key}")
        range_node = node.get("ElementRange")
        if header is None or header.size < 1 or range_node is None:
            raise CodecError(
                f"'{name}': section '{where}/{key}' lacks its type or ElementRange."
            )
        span = _data(range_node, ctx, f"{where}/{key}/ElementRange")
        if span is None or span.size < 2:
            raise CodecError(f"'{name}': section '{where}/{key}' has no ElementRange.")
        start, end = int(span.ravel()[0]), int(span.ravel()[1])
        if start < 1 or end < start - 1:
            raise CodecError(
                f"'{name}': section '{where}/{key}' ranges {start}..{end}; element"
                " numbers start at one."
            )
        if end >= start:
            sections.append((start, end, key, int(header.ravel()[0]), node))
    sections.sort(key=lambda s: s[0])
    for (_, end, key, _, _), (start, _, other, _, _) in zip(sections, sections[1:]):
        if start <= end:
            raise CodecError(
                f"'{name}': sections '{where}/{key}' and '{where}/{other}' both"
                f" number element {start}."
            )

    blocks: dict[int, list[np.ndarray]] = {}
    block_index: dict[int, list[np.ndarray]] = {}
    max_number = max((end for _, end, *_ in sections), default=0)
    validate_header(n_verts, max_number, 0, ctx["file_size"], compressed=True)
    element_index = np.full(max_number + 1, -1, dtype=np.int64)
    unknown: set[str] = set()
    running = 0
    for start, end, key, code, node in sections:
        here = f"{where}/{key}"
        n_declared = end - start + 1
        conn_node = node.get("ElementConnectivity")
        conn = (
            _data(conn_node, ctx, f"{here}/ElementConnectivity")
            if conn_node is not None
            else None
        )
        if conn is None or conn.dtype.kind not in "iu":
            raise CodecError(f"'{name}': section '{here}' has no ElementConnectivity.")
        conn = conn.ravel().astype(np.int64)
        offsets_node = node.get("ElementStartOffset")
        offsets = (
            _data(offsets_node, ctx, f"{here}/ElementStartOffset")
            if offsets_node is not None
            else None
        )
        if offsets is not None:
            offsets = offsets.ravel().astype(np.int64)
        validate_header(
            n_verts, n_declared, conn.size, ctx["file_size"], compressed=True
        )
        for ptype, cells, numbered in _section_cells(
            code, conn, offsets, n_declared, start, here, ctx, unknown
        ):
            if cells.size and (cells.min() < 0 or cells.max() >= n_verts):
                raise CodecError(
                    f"'{name}': section '{here}' names a node outside 1..{n_verts}."
                )
            pcode = int(ELEMENT_TYPES[ptype])
            blocks.setdefault(pcode, []).append(cells)
            block_index.setdefault(pcode, []).append(numbered)
    if unknown:
        warnings.warn(
            f"{EXTENSION} read: element type(s) {sorted(unknown)} in '{name}' have"
            " no polyxios element type; those elements are skipped.",
            UserWarning,
            stacklevel=4,
        )
    out: list[tuple[str, np.ndarray]] = []
    for pcode in sorted(blocks):
        parts = blocks[pcode]
        if len({p.shape[1] for p in parts}) > 1:
            # Polygons of several sizes: one block per width keeps make_polydata's
            # rectangular contract; they are still one type.
            for cells, numbered in zip(parts, block_index[pcode]):
                out.append((ELEMENT_TYPES_INV[pcode], cells))
                element_index[numbered] = running + np.arange(len(cells))
                running += len(cells)
            continue
        cells = np.concatenate(parts) if len(parts) > 1 else parts[0]
        numbered = np.concatenate(block_index[pcode])
        out.append((ELEMENT_TYPES_INV[pcode], cells))
        element_index[numbered] = running + np.arange(len(cells))
        running += len(cells)
    return out, element_index[1:]


def _dimension_of(codes: np.ndarray) -> np.ndarray:
    return _DIMENSION_BY_CODE[np.asarray(codes, dtype=np.int64)]


def _section_cells(
    code: int,
    conn: np.ndarray,
    offsets: np.ndarray | None,
    n_declared: int,
    start: int,
    here: str,
    ctx: dict[str, Any],
    unknown: set[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The typed cells a section holds, each with its element numbers."""
    name = ctx["name"]
    if code == _MIXED:
        return _mixed_cells(conn, offsets, n_declared, start, here, ctx, unknown)
    if code == _NGON:
        return _ngon_cells(conn, offsets, n_declared, start, here, ctx)
    if code == _NFACE:
        unknown.add(_CGNS_NAMES[code])
        return []
    ptype = _CGNS_TO_POLYXIOS.get(code)
    if ptype is None:
        if code in _CGNS_SIZES or code in _CGNS_NAMES:
            unknown.add(_CGNS_NAMES.get(code, f"type {code}"))
            return []
        raise UnknownElementTypeError(name, code)
    width = NODES_PER_ELEMENT[ptype]
    if conn.size != n_declared * width:
        raise CodecError(
            f"'{name}': section '{here}' declares {n_declared} {_CGNS_NAMES[code]}"
            f" and holds {conn.size} indices, not {n_declared * width}."
        )
    cells = conn.reshape(n_declared, width) - 1
    order = _READ_ORDER.get(ptype)
    if order is not None:
        cells = cells[:, list(order)]
    numbers = np.arange(start, start + n_declared, dtype=np.int64)
    return [(ptype, cells, numbers)]


def _mixed_cells(
    conn: np.ndarray,
    offsets: np.ndarray | None,
    n_declared: int,
    start: int,
    here: str,
    ctx: dict[str, Any],
    unknown: set[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Walk a MIXED stream: a type code, then that type's nodes, per element.

    The elements of one type are gathered at once from their start offsets;
    only a stream without ``ElementStartOffset`` has to be paced through,
    since each element's width is known only once its code is read.
    """
    name = ctx["name"]
    if offsets is not None:
        if offsets.size != n_declared + 1:
            raise CodecError(
                f"'{name}': section '{here}' has {offsets.size} start offsets for"
                f" {n_declared} elements."
            )
        starts = offsets
    else:
        starts = _pace_mixed(conn, n_declared, here, name)
    first, last = starts[:-1], starts[1:]
    _check_bounds(first, last, conn.size, here, name, "element")
    codes = conn[first]
    # Nodes per element, the count ahead of an older NGON's nodes included.
    widths = last - first - 1
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for code in _in_first_appearance(codes):
        picked = np.flatnonzero(codes == code)
        ptype = _CGNS_TO_POLYXIOS.get(code)
        if ptype is None:
            if code in _CGNS_SIZES or code in _CGNS_NAMES:
                unknown.add(_CGNS_NAMES.get(code, f"type {code}"))
                continue
            raise UnknownElementTypeError(name, code)
        if ptype == "polygon":
            skip = 2 if offsets is None else 1
            out.extend(
                _polygon_blocks(
                    conn,
                    first[picked] + skip,
                    widths[picked] - (skip - 1),
                    start + picked,
                )
            )
            continue
        width = NODES_PER_ELEMENT[ptype]
        wrong = np.flatnonzero(widths[picked] != width)
        if wrong.size:
            k = int(picked[wrong[0]])
            raise CodecError(
                f"'{name}': section '{here}' element {k + 1} is a"
                f" {_CGNS_NAMES[code]} of {widths[k]} nodes, not {width}."
            )
        cells = conn[(first[picked] + 1)[:, None] + np.arange(width)] - 1
        order = _READ_ORDER.get(ptype)
        if order is not None:
            cells = cells[:, list(order)]
        out.append((ptype, cells.astype(np.int64), (start + picked).astype(np.int64)))
    return out


def _pace_mixed(conn: np.ndarray, n_declared: int, here: str, name: str) -> np.ndarray:
    """The start of each element of a MIXED stream that carries no offsets."""
    flat = conn.tolist()
    size = len(flat)
    starts = np.empty(n_declared + 1, dtype=np.int64)
    widths: dict[int, int | None] = {}
    pos = 0
    for k in range(n_declared):
        if pos >= size:
            raise CodecError(
                f"'{name}': section '{here}' ends before its {n_declared} elements."
            )
        code = flat[pos]
        if code not in widths:
            widths[code] = _width_of(code, name)
        width = widths[code]
        if width is None:
            # An NGON inside MIXED spells its own node count after the code.
            width = flat[pos + 1] + 1 if pos + 1 < size else 0
        starts[k] = pos
        pos += 1 + width
    starts[n_declared] = pos
    return starts


def _check_bounds(
    first: np.ndarray, last: np.ndarray, size: int, here: str, name: str, what: str
) -> None:
    """Refuse the first element whose span is empty or runs past the stream."""
    bad = np.flatnonzero((last <= first) | (last > size))
    if bad.size:
        raise CodecError(
            f"'{name}': section '{here}' {what} {int(bad[0]) + 1} runs past its data."
        )


def _in_first_appearance(values: np.ndarray) -> list[int]:
    """The distinct values, each where it first occurs."""
    distinct, index = np.unique(values, return_index=True)
    return [int(v) for v in distinct[np.argsort(index)]]


def _polygon_blocks(
    conn: np.ndarray, first: np.ndarray, widths: np.ndarray, numbers: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Polygons gathered one width at a time, the widths as they first occur."""
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for width in _in_first_appearance(widths):
        rows = np.flatnonzero(widths == width)
        cells = conn[first[rows][:, None] + np.arange(max(width, 0))] - 1
        out.append(("polygon", cells.astype(np.int64), numbers[rows].astype(np.int64)))
    return out


def _width_of(code: int, name: str) -> int | None:
    ptype = _CGNS_TO_POLYXIOS.get(code)
    if ptype is not None:
        width = NODES_PER_ELEMENT[ptype]
        return width if width > 0 else None
    if code in _CGNS_SIZES:
        return _CGNS_SIZES[code]
    if code in _CGNS_NAMES:
        return None
    raise UnknownElementTypeError(name, code)


def _ngon_cells(
    conn: np.ndarray,
    offsets: np.ndarray | None,
    n_declared: int,
    start: int,
    here: str,
    ctx: dict[str, Any],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Polygons: with start offsets (CGNS 4) or a node count ahead of each (older)."""
    name = ctx["name"]
    if offsets is not None:
        if offsets.size != n_declared + 1:
            raise CodecError(
                f"'{name}': section '{here}' has {offsets.size} start offsets for"
                f" {n_declared} polygons."
            )
        bounds = offsets
    else:
        flat = conn.tolist()
        bounds = np.empty(n_declared + 1, dtype=np.int64)
        bounds[0] = pos = 0
        for k in range(n_declared):
            if pos >= len(flat):
                raise CodecError(
                    f"'{name}': section '{here}' ends before its polygons."
                )
            pos += 1 + flat[pos]
            bounds[k + 1] = pos
    first, last = bounds[:-1], bounds[1:]
    _check_bounds(first, last, conn.size, here, name, "polygon")
    skip = 0 if offsets is not None else 1
    numbers = np.arange(start, start + n_declared, dtype=np.int64)
    return _polygon_blocks(conn, first + skip, last - first - skip, numbers)


def _location(node: Any, ctx: dict[str, Any], where: str, default: str) -> str:
    loc = _first_child(node, "GridLocation_t")
    if loc is None:
        return default
    return _text_data(loc, ctx, f"{where}/GridLocation") or default


def _read_boundary(
    zone: Any,
    n_verts: int,
    element_index: np.ndarray,
    lattice: tuple[int, ...] | None,
    vertex_tags: dict[str, np.ndarray],
    element_tags: dict[str, np.ndarray],
    ctx: dict[str, Any],
    where: str,
) -> None:
    zone_bc = _first_child(zone, "ZoneBC_t")
    if zone_bc is None:
        return
    for key, bc in _children(zone_bc, "BC_t"):
        here = f"{where}/ZoneBC/{key}"
        location = _location(bc, ctx, here, "Vertex")
        members = _point_set(bc, ctx, here, lattice=lattice, location=location)
        if members is None and lattice is None:
            # CGNS 2 spelled a condition on faces as ElementList or
            # ElementRange; the numbers are element numbers whatever the
            # GridLocation says, which is then Vertex by default.
            members = _element_set(bc, ctx, here)
            location = "FaceCenter"
        if members is None:
            continue
        if location == "Vertex":
            picked = members - 1
            picked = picked[(picked >= 0) & (picked < n_verts)]
            vertex_tags[key] = np.unique(picked)
        else:
            picked = _indexed(members, element_index)
            element_tags[key] = np.unique(picked[picked >= 0])


def _indexed(members: np.ndarray, element_index: np.ndarray) -> np.ndarray:
    """The mesh index of each element number, ``-1`` for one the zone lacks."""
    out = np.full(len(members), -1, dtype=np.int64)
    inside = (members >= 1) & (members <= len(element_index))
    out[inside] = element_index[members[inside] - 1]
    return out


def _element_set(node: Any, ctx: dict[str, Any], where: str) -> np.ndarray | None:
    """An ``ElementList``'s numbers, or the numbers an ``ElementRange`` spans."""
    if "ElementList" in node:
        values = _data(node["ElementList"], ctx, f"{where}/ElementList")
        if values is None or values.dtype.kind not in "iu":
            return None
        return values.ravel().astype(np.int64)
    if "ElementRange" in node:
        values = _data(node["ElementRange"], ctx, f"{where}/ElementRange")
        if values is None or values.size != 2 or values.dtype.kind not in "iu":
            return None
        lo, hi = (int(v) for v in values.ravel())
        validate_header(0, max(hi - lo + 1, 0), 0, ctx["file_size"], compressed=True)
        return np.arange(lo, hi + 1, dtype=np.int64)
    return None


def _lattice_sizes(lattice: tuple[int, ...], location: str) -> tuple[int, ...] | None:
    """How many entities a structured zone has along each index, by location.

    ``Vertex`` counts the lattice points, ``CellCenter`` the cells between
    them. A face-centred location names faces the expanded mesh does not
    hold, so it has no sizes.
    """
    if location == "Vertex":
        return lattice
    if location == "CellCenter":
        return tuple(max(n - 1, 0) for n in lattice)
    return None


def _point_set(
    node: Any,
    ctx: dict[str, Any],
    where: str,
    *,
    lattice: tuple[int, ...] | None,
    location: str,
) -> np.ndarray | None:
    """A PointList's numbers, or the numbers a PointRange spans.

    On a structured zone both are index tuples - ``(IndexDimension, n)`` and
    ``(IndexDimension, 2)`` in CGNS, so ``(n, IndexDimension)`` and
    ``(2, IndexDimension)`` here - turned into the numbers of the expanded
    mesh, ``i`` fastest, over the lattice's vertices or its cells.
    """
    if "PointList" in node:
        values = _data(node["PointList"], ctx, f"{where}/PointList")
        if values is None or values.dtype.kind not in "iu":
            return None
        if lattice is None:
            return values.ravel().astype(np.int64)
        sizes = _lattice_sizes(lattice, location)
        if sizes is None or values.size % len(sizes):
            return None
        return _lattice_numbers(values.reshape(-1, len(sizes)).T, sizes)
    if "PointRange" in node:
        values = _data(node["PointRange"], ctx, f"{where}/PointRange")
        if values is None or values.size < 2 or values.dtype.kind not in "iu":
            return None
        if lattice is None:
            if values.size != 2:
                return None
            sizes = None
            bounds = values.reshape(2, 1).astype(np.int64)
        else:
            sizes = _lattice_sizes(lattice, location)
            if sizes is None or values.size != 2 * len(sizes):
                return None
            bounds = values.reshape(2, -1).astype(np.int64)
        spans = [max(int(hi) - int(lo) + 1, 0) for lo, hi in bounds.T]
        validate_header(0, int(np.prod(spans)), 0, ctx["file_size"], compressed=True)
        axes = [np.arange(lo, hi + 1, dtype=np.int64) for lo, hi in bounds.T]
        if sizes is None:
            return axes[0]
        grid = np.meshgrid(*axes, indexing="ij")
        return _lattice_numbers(np.stack([g.ravel(order="F") for g in grid]), sizes)
    return None


def _lattice_numbers(tuples: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    """The 1-based numbers of ``(IndexDimension, n)`` index tuples, ``i`` fastest."""
    inside = np.ones(tuples.shape[1], dtype=bool)
    for axis, n in zip(tuples, sizes):
        inside &= (axis >= 1) & (axis <= n)
    if not inside.all():
        tuples = tuples[:, inside]
    if not tuples.size:
        return np.empty(0, dtype=np.int64)
    flat = np.ravel_multi_index(tuple(tuples - 1), sizes, order="F")
    return np.asarray(flat, dtype=np.int64) + 1


def _read_solutions(
    zone: Any,
    n_verts: int,
    n_elems: int,
    cell_dim: int,
    element_index: np.ndarray,
    dims: np.ndarray,
    lattice: tuple[int, ...] | None,
    vertex_attrs: dict[str, np.ndarray],
    element_attrs: dict[str, np.ndarray],
    ctx: dict[str, Any],
    where: str,
) -> None:
    """Lay every FlowSolution_t over the vertices or the elements.

    A solution names its entities by a ``PointList`` or ``PointRange``, or
    spans every entity of its location - all the vertices, or all the
    elements of the location's dimension in element-number order, which is
    what a cell-centred solution over the volume cells alone is. Several
    solutions can each fill part of one attribute; what none covers is NaN.
    """
    name = ctx["name"]
    by_dim: dict[int, np.ndarray] = {}
    over_vertices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    over_elements: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, sol in _children(zone, "FlowSolution_t"):
        here = f"{where}/{key}"
        location = _location(sol, ctx, here, "Vertex")
        members = _point_set(sol, ctx, here, lattice=lattice, location=location)
        if location == "Vertex":
            store, count, what = over_vertices, n_verts, "vertices"
            listed = members - 1 if members is not None else None
            if listed is not None:
                listed[(listed < 0) | (listed >= n_verts)] = -1
        else:
            store, count, what = over_elements, n_elems, "elements"
            listed = _indexed(members, element_index) if members is not None else None
        for array_name, node in _children(sol, "DataArray_t"):
            values = _data(node, ctx, f"{here}/{array_name}")
            if values is None or values.dtype.kind not in "biuf":
                continue
            values = _as_rows(values, lattice is not None)
            if listed is not None:
                if len(values) != len(listed):
                    _length_warning(array_name, len(values), len(listed), what, name)
                    continue
                index = listed
            elif len(values) == count:
                index = None
            elif store is over_elements:
                dim = _location_dim(location, cell_dim)
                if dim not in by_dim:
                    by_dim[dim] = _by_number(element_index, dims, dim)
                if len(values) != len(by_dim[dim]):
                    _length_warning(array_name, len(values), count, what, name)
                    continue
                index = by_dim[dim]
            else:
                _length_warning(array_name, len(values), count, what, name)
                continue
            _lay(store, array_name, index, values, count)
    for store, attrs in ((over_vertices, vertex_attrs), (over_elements, element_attrs)):
        for array_name, (out, filled) in store.items():
            if filled.all():
                attrs[array_name] = out
            else:
                out = out.astype(np.float64)
                out[~filled] = np.nan
                attrs[array_name] = out


def _location_dim(location: str, cell_dim: int) -> int:
    """The dimension of the elements a solution at this location spans."""
    if location == "EdgeCenter":
        return 1
    if location.endswith("FaceCenter"):
        return 2
    return cell_dim


def _by_number(element_index: np.ndarray, dims: np.ndarray, dim: int) -> np.ndarray:
    """The mesh indices of the elements of one dimension, in element-number order."""
    present = element_index[element_index >= 0]
    picked: np.ndarray = present[dims[present] == dim]
    return picked


def _lay(
    store: dict[str, tuple[np.ndarray, np.ndarray]],
    array_name: str,
    index: np.ndarray | None,
    values: np.ndarray,
    count: int,
) -> None:
    """Put a solution's rows at ``index`` of the attribute it fills, or over all."""
    if index is None:
        store[array_name] = (values, np.ones(count, dtype=bool))
        return
    inside = index >= 0
    good = index[inside]
    if len(good) != len(index):
        values = values[inside]
    if array_name in store and store[array_name][0].shape[1:] == values.shape[1:]:
        out, filled = store[array_name]
        if out.dtype != values.dtype:
            out = out.astype(np.result_type(out.dtype, values.dtype))
    else:
        out = np.zeros((count, *values.shape[1:]), dtype=values.dtype)
        filled = np.zeros(count, dtype=bool)
    out[good] = values
    filled[good] = True
    store[array_name] = (out, filled)


def _as_rows(values: np.ndarray, structured: bool) -> np.ndarray:
    """One row per entity: a CGNS ``(n, k)`` array reads as ``(k, n)``.

    A structured zone's solution is laid on its lattice, ``(nk, nj, ni)``
    here, one value per point or cell; it flattens to the order the
    coordinates and the cells were read in, ``i`` fastest.
    """
    if values.ndim == 1:
        return values
    if structured:
        return values.reshape(-1)
    if values.ndim == 2:
        return np.ascontiguousarray(values.T)
    return np.ascontiguousarray(values.reshape(values.shape[0], -1).T)


def _length_warning(
    array_name: str, held: int, wanted: int, what: str, name: str
) -> None:
    warnings.warn(
        f"{EXTENSION} read: solution {array_name!r} in '{name}' holds {held}"
        f" values for {wanted} {what}; skipped.",
        UserWarning,
        stacklevel=5,
    )


def _read_user_data(base: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, udd in _children(base, "UserDefinedData_t"):
        for array_name, node in _children(udd, "DataArray_t"):
            if attr_text(node, "type") == "C1":
                text = _text_data(node, ctx, f"{key}/{array_name}")
                if text is not None:
                    out[array_name] = text
                continue
            values = _data(node, ctx, f"{key}/{array_name}")
            if values is None:
                continue
            if values.dtype.kind in "biuf":
                out[array_name] = (
                    values.reshape(-1) if values.size != 1 else values.reshape(-1)[0]
                )
        for desc_name, node in _children(udd, "Descriptor_t"):
            text = _text_data(node, ctx, f"{key}/{desc_name}")
            if text is not None:
                out[desc_name] = text
    return out


# ----- writing ----------------------------------------------------------------


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as a CGNS file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object.
    **opts
        ``zone_name`` and ``base_name`` name the zone and the base; without
        them ``global_attrs["zone_name"]`` and ``["base_name"]`` are used,
        and ``"Zone"`` and ``"Base"`` failing that. ``compression`` and
        ``compression_opts`` are passed to h5py for the datasets; a file
        with compressed datasets still reads through the CGNS library
        wherever its HDF5 has the filter.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If the mesh's vertices are not a coordinate block.

    Warns
    -----
    UserWarning
        Once per element type the SIDS has no code for - those elements, and
        the values and tags on them, are not written; once for the attributes
        no DataArray can hold; once for the ``global_attrs`` values that are
        neither numbers nor text; once for the tag, attribute and global
        names that had to change - a CGNS node name is 32 characters, holds
        no ``/`` and cannot repeat among its siblings - naming each and what
        it became.

    Notes
    -----
    One base of the mesh's own dimensions, one unstructured zone, one
    ``Elements_t`` section per element type over consecutive element
    numbers, the types in ascending polyxios code order, polygons among
    them as an ``NGON_n`` section with ``ElementStartOffset``, as CGNS 4
    spells it; a pixel and a voxel go out as the quad and the hexahedron
    they are. Tag groups are ``BC_t`` nodes under ``ZoneBC``, a vertex
    group at ``Vertex`` and an element group at ``CellCenter``, each with a
    ``PointList``; ``vertex_attrs`` go under a ``FlowSolution_t`` at
    ``Vertex``, ``element_attrs`` under one at ``CellCenter`` over the cells
    of the mesh's own dimension - as many as the zone declares - and, for
    the elements below them, under one at ``FaceCenter`` or ``EdgeCenter``
    with a ``PointList`` naming them; ``global_attrs`` under a
    ``UserDefinedData_t`` on the base, numbers as ``DataArray_t`` and text
    as ``Descriptor_t``.
    """
    zone_name = opts.pop("zone_name", None)
    base_name = opts.pop("base_name", None)
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write")
    if zone_name is None:
        zone_name = text_of(poly.global_attrs.get(_ZONE_NAME_KEY)) or "Zone"
    if base_name is None:
        base_name = text_of(poly.global_attrs.get(_BASE_NAME_KEY)) or "Base"
    zone_name, base_name = _node_name(str(zone_name)), _node_name(str(base_name))
    numeric = globals_for_write(
        poly, reserved=_RESERVED_GLOBALS, fmt=EXTENSION, text=True
    )
    texts = text_for_write(poly, reserved=_RESERVED_GLOBALS)
    phys_dim = output_dimension(poly, fmt=EXTENSION)
    n_verts = len(poly.vertices)
    n_elems = len(poly.element_types)

    blocks, dropped = type_blocks(poly, sizes=_WRITE_SIZES, fmt=EXTENSION)
    polygons = _polygon_rows(poly)
    dropped.discard(_POLYGON)
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: element type(s)"
            f" {sorted(ELEMENT_TYPES_INV.get(c, f'type_{c}') for c in dropped)}"
            " have no CGNS element type; those elements and the values and tags"
            " on them are not written.",
            UserWarning,
            stacklevel=2,
        )
    # Polygons take their place among the fixed-width blocks by code, so the
    # sections go out in the order the reader sorts them back into.
    sections: list[tuple[int, np.ndarray, np.ndarray | None]] = sorted(
        [*blocks, (_POLYGON, polygons[0], None)] if polygons[0].size else blocks,
        key=lambda block: block[0],
    )
    kept = kept_index([(code, index, index) for code, index, _ in sections])
    cell_dim = int(poly.topological_dimension) if n_elems else 3
    codes = (
        np.asarray(poly.element_types)[kept]
        if kept.size
        else np.empty(0, dtype=np.int64)
    )
    dims = _dimension_of(codes)
    n_cells = int((dims == cell_dim).sum())
    wide = (
        n_verts > _INT32_MAX
        or n_elems > _INT32_MAX
        or int(poly.connectivity.max(initial=0)) + 1 > _INT32_MAX
    )
    itype = np.int64 if wide else np.int32
    element_tags = reindexed_tags(poly.element_tags, kept, n_elems)
    vertex_arrays = _solution_arrays(poly.vertex_attrs, n_verts)
    element_arrays = _solution_arrays(poly.element_attrs, n_elems)

    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        _root(handle)
        _node(
            handle,
            "CGNSLibraryVersion",
            "CGNSLibraryVersion_t",
            np.array([4.2], dtype=np.float32),
        )
        base = _node(
            handle,
            _unique_name(handle, base_name),
            "CGNSBase_t",
            np.array([cell_dim, phys_dim], dtype=np.int32),
        )
        size = np.array([[n_verts], [n_cells], [0]], dtype=itype)
        zone = _node(base, _unique_name(base, zone_name), "Zone_t", size)
        _node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        grid = _node(zone, "GridCoordinates", "GridCoordinates_t", None)
        coords = np.asarray(poly.vertices, dtype=np.float64)
        for k, axis in enumerate(
            ("CoordinateX", "CoordinateY", "CoordinateZ")[:phys_dim]
        ):
            _node(
                grid,
                axis,
                "DataArray_t",
                np.ascontiguousarray(coords[:, k]),
                **dataset_kw,
            )

        number = 1
        for code, index, conn in sections:
            n = len(index)
            if conn is None:
                cgns_code = _NGON
                flat, offsets = polygons[1], polygons[2]
            else:
                cgns_code, order = _cgns_spelling(ELEMENT_TYPES_INV[code])
                if order is not None:
                    conn = conn[:, list(order)]
                flat, offsets = conn.ravel(), None
            section = _node(
                zone,
                _unique_name(zone, _CGNS_NAMES[cgns_code]),
                "Elements_t",
                np.array([cgns_code, 0], dtype=np.int32),
            )
            _node(
                section,
                "ElementRange",
                "IndexRange_t",
                np.array([number, number + n - 1], dtype=itype),
            )
            _node(
                section,
                "ElementConnectivity",
                "DataArray_t",
                np.ascontiguousarray(flat + 1).astype(itype),
                **dataset_kw,
            )
            if offsets is not None:
                _node(
                    section,
                    "ElementStartOffset",
                    "DataArray_t",
                    offsets.astype(itype),
                    **dataset_kw,
                )
            number += n

        renamed: dict[str, str] = {}
        zone_bc = None
        for tag_name, members in poly.vertex_tags.items():
            picked = member_indices(members, n_verts)
            if zone_bc is None:
                zone_bc = _node(zone, "ZoneBC", "ZoneBC_t", None)
            _boundary(
                zone_bc, str(tag_name), "Vertex", picked + 1, itype, dataset_kw, renamed
            )
        for tag_name, picked in element_tags.items():
            if zone_bc is None:
                zone_bc = _node(zone, "ZoneBC", "ZoneBC_t", None)
            _boundary(
                zone_bc,
                str(tag_name),
                _bc_location(codes[picked], cell_dim),
                picked + 1,
                itype,
                dataset_kw,
                renamed,
            )

        _write_solution(
            zone,
            "VertexSolution",
            "Vertex",
            vertex_arrays,
            None,
            None,
            itype,
            dataset_kw,
            renamed,
        )
        for sol_name, location, positions, listed in _solution_parts(dims, cell_dim):
            _write_solution(
                zone,
                sol_name,
                location,
                element_arrays,
                kept[positions],
                positions + 1 if listed else None,
                itype,
                dataset_kw,
                renamed,
            )

        if numeric or texts:
            udd = _node(base, _unique_name(base, _USER_DATA), "UserDefinedData_t", None)
            for key, arr in numeric.items():
                _node(
                    udd,
                    _child_name(udd, key, renamed),
                    "DataArray_t",
                    np.ascontiguousarray(arr),
                )
            for key, lines in texts.items():
                _node(
                    udd,
                    _child_name(udd, key, renamed),
                    "Descriptor_t",
                    "\n".join(lines),
                )
    if renamed:
        warnings.warn(
            f"{EXTENSION} write: name(s) {sorted(renamed)} are written as"
            f" {[renamed[k] for k in sorted(renamed)]}; a CGNS node name is 32"
            " characters, holds no '/' and cannot repeat among its siblings.",
            UserWarning,
            stacklevel=2,
        )


def _solution_parts(
    dims: np.ndarray, cell_dim: int
) -> list[tuple[str, str, np.ndarray, bool]]:
    """Where the element values go: one solution per dimension of element written.

    Parameters
    ----------
    dims
        The dimension of each written element, in written order.
    cell_dim
        The mesh's own dimension: its cells are the elements of this one.

    Returns
    -------
    list of tuple
        ``(name, location, positions, listed)`` per solution: the cells go
        first at ``CellCenter`` without a list, since a cell-centred solution
        spans the zone's cells by definition; every lower dimension gets a
        solution of its own with a ``PointList``, at ``FaceCenter`` for faces,
        ``EdgeCenter`` for edges and ``CellCenter`` for what is neither.
    """
    parts: list[tuple[str, str, np.ndarray, bool]] = []
    cells = np.flatnonzero(dims == cell_dim)
    if cells.size:
        parts.append(("CellSolution", "CellCenter", cells, False))
    for dim in sorted({int(d) for d in np.unique(dims)} - {cell_dim}, reverse=True):
        positions = np.flatnonzero(dims == dim)
        if dim == 2:
            parts.append(("FaceSolution", "FaceCenter", positions, True))
        elif dim == 1:
            parts.append(("EdgeSolution", "EdgeCenter", positions, True))
        else:
            parts.append((f"Solution{dim}D", "CellCenter", positions, True))
    return parts


def _bc_location(codes: np.ndarray, cell_dim: int) -> str:
    """Where a condition sits: on the faces or edges below the cells, or on cells."""
    if not codes.size:
        return "CellCenter"
    dims = _dimension_of(codes)
    if cell_dim == 3 and (dims == 2).all():
        return "FaceCenter"
    if cell_dim == 2 and (dims == 1).all():
        return "EdgeCenter"
    return "CellCenter"


def _polygon_rows(poly: PolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The polygons' mesh indices, flat connectivity and start offsets."""
    codes = np.asarray(poly.element_types)
    index = np.flatnonzero(codes == _POLYGON).astype(np.int64)
    if not index.size:
        return index, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    offsets = np.asarray(poly.offsets)
    conn = np.asarray(poly.connectivity)
    widths = offsets[index + 1] - offsets[index]
    pieces = [conn[offsets[i] : offsets[i + 1]] for i in index]
    flat = np.concatenate(pieces).astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(widths))).astype(np.int64)
    return index, flat, starts


def _cgns_spelling(ptype: str) -> tuple[int, tuple[int, ...] | None]:
    if ptype in _LATTICE:
        return _LATTICE[ptype]
    return _POLYXIOS_TO_CGNS[ptype], _WRITE_ORDER.get(ptype)


def _node_name(text: str) -> str:
    """A CGNS node name: 32 characters, no slash, not HDF5's ``.`` or ``..``."""
    cleaned = text.replace("/", "_").strip()
    if cleaned in {"", ".", ".."}:
        cleaned = "unnamed"
    return cleaned[: _LABEL_WIDTH - 1]


def _child_name(parent: Any, wanted: str, renamed: dict[str, str]) -> str:
    """The node name ``wanted`` gets under ``parent``, recorded when it differs."""
    name = _unique_name(parent, _node_name(wanted))
    if name != wanted:
        renamed[wanted] = name
    return name


def _unique_name(
    parent: Any, wanted: str, *, width: int | None = _LABEL_WIDTH - 1
) -> str:
    """``wanted``, or ``wanted_2`` and up when ``parent`` already holds it.

    Parameters
    ----------
    parent
        An h5py group, or a dict, whose keys the name must not repeat.
    wanted
        The name asked for.
    width
        The longest name allowed - a CGNS node name is 32 characters, so
        the suffix takes the place of the name's tail rather than exceeding
        it; None for a dict, which has no such limit.
    """
    name = wanted
    k = 2
    while name in parent:
        suffix = f"_{k}"
        stem = wanted if width is None else wanted[: width - len(suffix)]
        name = f"{stem}{suffix}"
        k += 1
    return name


def _root(handle: Any) -> None:
    handle.attrs.create("name", np.bytes_("HDF5 MotherNode"), dtype=f"S{_LABEL_WIDTH}")
    handle.attrs.create(
        "label", np.bytes_("Root Node of HDF5 File"), dtype=f"S{_LABEL_WIDTH}"
    )
    handle.attrs.create("type", np.bytes_("MT"), dtype="S3")
    handle.create_dataset(" format", data=_chars("IEEE_LITTLE_32", 15))
    handle.create_dataset(" hdf5version", data=_chars(_hdf5_version(), _LABEL_WIDTH))


def _hdf5_version() -> str:
    h5py, _ = _h5py()
    text = f"HDF5 Version {h5py.version.hdf5_version}"
    return text[: _LABEL_WIDTH - 1]


def _chars(text: str, width: int | None = None) -> np.ndarray:
    raw = text.encode("utf-8")
    if width is not None:
        raw = raw[:width].ljust(width, b"\x00")
    return np.frombuffer(raw, dtype=np.int8).copy()


def _node(parent: Any, name: str, label: str, data: Any, **dataset_kw: Any) -> Any:
    """Create one CGNS node: a group with its name, label, type and `` data``."""
    group = parent.create_group(name)
    group.attrs.create("name", np.bytes_(name), dtype=f"S{_LABEL_WIDTH}")
    group.attrs.create("label", np.bytes_(label), dtype=f"S{_LABEL_WIDTH}")
    group.attrs.create("flags", np.array([1], dtype=np.int32))
    if data is None:
        group.attrs.create("type", np.bytes_("MT"), dtype="S3")
        return group
    if isinstance(data, str):
        group.attrs.create("type", np.bytes_("C1"), dtype="S3")
        group.create_dataset(" data", data=_chars(data))
        return group
    arr = np.asarray(data)
    if arr.dtype.kind == "b":
        arr = arr.astype(np.int32)
    if arr.dtype.kind == "f":
        arr = (
            arr.astype(np.float32)
            if arr.dtype.itemsize <= 4
            else arr.astype(np.float64)
        )
        code = "R4" if arr.dtype.itemsize == 4 else "R8"
    elif arr.dtype.kind in "iu":
        if arr.dtype.kind == "u" and arr.dtype.itemsize == 8:
            if arr.size and int(arr.max()) > np.iinfo(np.int64).max:
                raise CodecError(
                    f"{EXTENSION} write: node {name!r} holds a value above what"
                    " I8, the widest CGNS integer, can hold."
                )
            arr = arr.astype(np.int64)
        elif arr.dtype.kind == "u" and arr.dtype.itemsize == 4:
            arr = arr.astype(
                np.int64 if arr.size and arr.max() > _INT32_MAX else np.int32
            )
        else:
            arr = arr.astype(np.int32 if arr.dtype.itemsize <= 4 else np.int64)
        code = "I4" if arr.dtype.itemsize == 4 else "I8"
    else:
        raise CodecError(
            f"{EXTENSION} write: node {name!r} holds {arr.dtype}, which CGNS cannot."
        )
    group.attrs.create("type", np.bytes_(code), dtype="S3")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    kwargs = dict(dataset_kw) if arr.size > 1 else {}
    group.create_dataset(" data", data=np.ascontiguousarray(arr), **kwargs)
    return group


def _boundary(
    zone_bc: Any,
    tag_name: str,
    location: str,
    numbers: np.ndarray,
    itype: type,
    dataset_kw: dict[str, Any],
    renamed: dict[str, str],
) -> None:
    bc = _node(
        zone_bc, _child_name(zone_bc, tag_name, renamed), "BC_t", "BCTypeUserDefined"
    )
    _node(bc, "GridLocation", "GridLocation_t", location)
    # A PointList is (IndexDimension, n) in CGNS, so (n, 1) here.
    _node(
        bc,
        "PointList",
        "IndexArray_t",
        numbers.astype(itype).reshape(-1, 1),
        **dataset_kw,
    )


def _solution_arrays(attrs: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    """The attributes a DataArray can hold, warning once for the rest."""
    arrays: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for attr_name, values in attrs.items():
        if attr_name == IDS_KEY:
            continue
        arr = np.asarray(values)
        if arr.dtype.kind == "b":
            arr = arr.astype(np.int32)
        if (
            arr.dtype.kind not in "iuf"
            or arr.ndim == 0
            or arr.shape[0] != count
            or arr.ndim > 2
        ):
            dropped.append(attr_name)
            continue
        arrays[attr_name] = arr
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: attribute(s) {sorted(dropped)} are not one number"
            " or one vector per entity; dropped.",
            UserWarning,
            stacklevel=3,
        )
    return arrays


def _write_solution(
    zone: Any,
    sol_name: str,
    location: str,
    arrays: dict[str, np.ndarray],
    kept: np.ndarray | None,
    numbers: np.ndarray | None,
    itype: type,
    dataset_kw: dict[str, Any],
    renamed: dict[str, str],
) -> None:
    """One FlowSolution_t: the arrays cut to ``kept``, listed by ``numbers`` if any."""
    if not arrays:
        return
    sol = _node(zone, _unique_name(zone, sol_name), "FlowSolution_t", None)
    _node(sol, "GridLocation", "GridLocation_t", location)
    if numbers is not None:
        _node(
            sol,
            "PointList",
            "IndexArray_t",
            numbers.astype(itype).reshape(-1, 1),
            **dataset_kw,
        )
    for attr_name, arr in arrays.items():
        if kept is not None:
            arr = arr[kept]
        # CGNS dims (n, k) are (k, n) in HDF5: one dataset row per component.
        data = arr if arr.ndim == 1 else np.ascontiguousarray(arr.T)
        _node(
            sol, _child_name(sol, attr_name, renamed), "DataArray_t", data, **dataset_kw
        )
