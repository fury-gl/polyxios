"""XDMF codec - an XML description of a mesh whose arrays live wherever it says.

An XDMF file is two things: the *light data*, an XML tree naming the grids,
their topology and geometry, the attributes on them and the time each grid
belongs to; and the *heavy data*, the arrays themselves, which every
``<DataItem>`` in the tree either holds inline (``Format="XML"``), names in
an HDF5 file (``Format="HDF"``, the usual case and the reason the format
exists) or names in a raw binary file (``Format="Binary"``). The HDF5 case
needs h5py, which is optional: without it the XML and binary flavours still
read and write, and an HDF5 reference is refused with the ``pip install``
line that fixes it.

A file holds one uniform grid, several of them, or a temporal collection of
them - a time series, each step a grid of its own. :func:`read` hands back
one mesh, always: several grids at one time are merged, each grid's elements
tagged with the grid's name; a time series is read at one step, the first
unless ``step=`` says otherwise. The whole series is read by
:func:`read_time_series`, and written by :func:`write_time_series`, both
reached through :mod:`polyxios.helper`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
import math
from pathlib import Path
import re
from typing import Any
import warnings
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

import numpy as np

from polyxios import transforms
from polyxios._dimension import mark_2d, output_dimension, pad_to_3d
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV, NODES_PER_ELEMENT
from polyxios._globals import globals_for_write, text_for_write
from polyxios._io import (
    Source,
    is_buffer,
    read_bytes,
    require_path,
    source_name,
    write_text,
)
from polyxios._optpkg import optional_package
from polyxios._tags import member_indices, member_values
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    spellable_arrays,
    structured_cell_shape,
    structured_cells,
)
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    UnknownElementTypeError,
    UnsupportedFormatError,
)
from polyxios.validate import validate_header

EXTENSION: str = ".xdmf"
EXTENSIONS: tuple[str, ...] = (".xdmf", ".xmf")

h5py, HAVE_H5PY = optional_package("h5py", min_version="3.0", extra="hdf5")

_XI: str = "{http://www.w3.org/2001/XInclude}"
_XI_URI: str = "http://www.w3.org/2001/XInclude"

# ----- topology types ---------------------------------------------------------

# XDMF's own names for the cells polyxios can hold, lower-cased for the lookup,
# each with the polyxios type it reads as. The mixed-topology code of each is
# what the Xdmf library numbers them; a reader meeting a code not here is met
# with UnknownElementTypeError rather than a KeyError.
_XDMF_TO_POLYXIOS: dict[str, str] = {
    "polyvertex": "vertex",
    "polyline": "line",
    "polygon": "polygon",
    "triangle": "triangle",
    "quadrilateral": "quad",
    "tetrahedron": "tetra",
    "pyramid": "pyramid",
    "wedge": "wedge",
    "hexahedron": "hexahedron",
    "edge_3": "quadratic_edge",
    "triangle_6": "quadratic_triangle",
    "tri_6": "quadratic_triangle",
    "quadrilateral_8": "quadratic_quad",
    "quad_8": "quadratic_quad",
    "quadrilateral_9": "biquadratic_quad",
    "quad_9": "biquadratic_quad",
    "tetrahedron_10": "quadratic_tetra",
    "tet_10": "quadratic_tetra",
    "pyramid_13": "quadratic_pyramid",
    "wedge_15": "quadratic_wedge",
    "wedge_18": "biquadratic_quadratic_wedge",
    "hexahedron_20": "quadratic_hexahedron",
    "hex_20": "quadratic_hexahedron",
    "hexahedron_24": "biquadratic_quadratic_hexahedron",
    "hex_24": "biquadratic_quadratic_hexahedron",
    "hexahedron_27": "triquadratic_hexahedron",
    "hex_27": "triquadratic_hexahedron",
}

# The Lagrange hexahedra past 27 nodes: XDMF names each by its node count, and
# polyxios holds them all as one free-size type, so the count is what tells
# them apart on the way back out.
_LAGRANGE_HEX_NODES: tuple[int, ...] = (64, 125, 216, 343, 512, 729, 1000, 1331)
for _count in _LAGRANGE_HEX_NODES:
    _XDMF_TO_POLYXIOS[f"hexahedron_{_count}"] = "lagrange_hexahedron"
    _XDMF_TO_POLYXIOS[f"hex_{_count}"] = "lagrange_hexahedron"

# Mixed-topology codes, as XdmfTopologyType numbers them.
_CODE_TO_XDMF: dict[int, str] = {
    0x1: "polyvertex",
    0x2: "polyline",
    0x3: "polygon",
    0x4: "triangle",
    0x5: "quadrilateral",
    0x6: "tetrahedron",
    0x7: "pyramid",
    0x8: "wedge",
    0x9: "hexahedron",
    0x10: "polyhedron",
    0x22: "edge_3",
    0x23: "quadrilateral_9",
    0x24: "triangle_6",
    0x25: "quadrilateral_8",
    0x26: "tetrahedron_10",
    0x27: "pyramid_13",
    0x28: "wedge_15",
    0x29: "wedge_18",
    0x30: "hexahedron_20",
    0x31: "hexahedron_24",
    0x32: "hexahedron_27",
    0x33: "hexahedron_64",
    0x34: "hexahedron_125",
    0x35: "hexahedron_216",
    0x36: "hexahedron_343",
    0x37: "hexahedron_512",
    0x38: "hexahedron_729",
    0x39: "hexahedron_1000",
    0x40: "hexahedron_1331",
}
_XDMF_TO_CODE: dict[str, int] = {name: code for code, name in _CODE_TO_XDMF.items()}
_MIXED_CODE: int = 0x70
_POLYHEDRON_CODE: int = 0x10

# The three free-size kinds: in a mixed stream each is followed by its node
# count, which is how the Xdmf library and VTK both spell them.
_POLY_CODES: frozenset[int] = frozenset({0x1, 0x2, 0x3})
_POLY_XDMF: frozenset[str] = frozenset({"polyvertex", "polyline", "polygon"})
_POLY_DEFAULT_NODES: dict[str, int] = {"polyvertex": 1, "polyline": 2}

# Nodes per XDMF type, for the fixed-size ones.
_XDMF_NODES: dict[str, int] = {
    name: NODES_PER_ELEMENT[ptype]
    for name, ptype in _XDMF_TO_POLYXIOS.items()
    if name not in _POLY_XDMF and NODES_PER_ELEMENT[ptype] > 0
}
for _count in _LAGRANGE_HEX_NODES:
    _XDMF_NODES[f"hexahedron_{_count}"] = _count
    _XDMF_NODES[f"hex_{_count}"] = _count

# What a mixed-topology code means, in one lookup: the node count - 0 for a
# free-size kind, whose count follows in the stream - and the polyxios type.
# The polyhedron is not here; it is stepped over, not read.
_MIXED_CELL: dict[int, tuple[int, int]] = {
    code: (
        0 if code in _POLY_CODES else _XDMF_NODES[name],
        ELEMENT_TYPES[_XDMF_TO_POLYXIOS[name]],
    )
    for code, name in _CODE_TO_XDMF.items()
    if code != _POLYHEDRON_CODE
}
# A free-size cell at its natural count is the fixed kind: one node is a
# vertex, two a line. Indexed by ``count == _POLY_SPLIT[code]``.
_POLY_SPLIT: dict[int, int] = {0x1: 1, 0x2: 2, 0x3: -1}
_POLY_TYPE_CODES: dict[int, tuple[int, int]] = {
    0x1: (ELEMENT_TYPES["poly_vertex"], ELEMENT_TYPES["vertex"]),
    0x2: (ELEMENT_TYPES["poly_line"], ELEMENT_TYPES["line"]),
    0x3: (ELEMENT_TYPES["polygon"], ELEMENT_TYPES["polygon"]),
}

# polyxios type code -> XDMF name and the node order the file wants. A pixel
# and a voxel are VTK's axis-aligned quad and hexahedron with the corners in
# lattice order; XDMF has only the general kind, so they go out reordered.
_WRITE_MAP: dict[int, tuple[str, tuple[int, ...] | None]] = {
    ELEMENT_TYPES["vertex"]: ("Polyvertex", None),
    ELEMENT_TYPES["poly_vertex"]: ("Polyvertex", None),
    ELEMENT_TYPES["line"]: ("Polyline", None),
    ELEMENT_TYPES["poly_line"]: ("Polyline", None),
    ELEMENT_TYPES["polygon"]: ("Polygon", None),
    ELEMENT_TYPES["triangle"]: ("Triangle", None),
    ELEMENT_TYPES["quad"]: ("Quadrilateral", None),
    ELEMENT_TYPES["pixel"]: ("Quadrilateral", (0, 1, 3, 2)),
    ELEMENT_TYPES["tetra"]: ("Tetrahedron", None),
    ELEMENT_TYPES["pyramid"]: ("Pyramid", None),
    ELEMENT_TYPES["wedge"]: ("Wedge", None),
    ELEMENT_TYPES["hexahedron"]: ("Hexahedron", None),
    ELEMENT_TYPES["voxel"]: ("Hexahedron", (0, 1, 3, 2, 4, 5, 7, 6)),
    ELEMENT_TYPES["quadratic_edge"]: ("Edge_3", None),
    ELEMENT_TYPES["quadratic_triangle"]: ("Triangle_6", None),
    ELEMENT_TYPES["quadratic_quad"]: ("Quadrilateral_8", None),
    ELEMENT_TYPES["biquadratic_quad"]: ("Quadrilateral_9", None),
    ELEMENT_TYPES["quadratic_tetra"]: ("Tetrahedron_10", None),
    ELEMENT_TYPES["quadratic_pyramid"]: ("Pyramid_13", None),
    ELEMENT_TYPES["quadratic_wedge"]: ("Wedge_15", None),
    ELEMENT_TYPES["biquadratic_quadratic_wedge"]: ("Wedge_18", None),
    ELEMENT_TYPES["quadratic_hexahedron"]: ("Hexahedron_20", None),
    ELEMENT_TYPES["biquadratic_quadratic_hexahedron"]: ("Hexahedron_24", None),
    ELEMENT_TYPES["triquadratic_hexahedron"]: ("Hexahedron_27", None),
}
_LAGRANGE_HEX: int = ELEMENT_TYPES["lagrange_hexahedron"]

# The structured topologies: points on a lattice, cells implied by its shape.
# ``Dimensions`` on these runs slowest axis first - K J I - and so do the
# origin and spacing of a co-rectilinear grid, which is how VTK reads them.
_STRUCTURED: dict[str, tuple[int, str]] = {
    "2dsmesh": (2, "smesh"),
    "3dsmesh": (3, "smesh"),
    "2drectmesh": (2, "rect"),
    "3drectmesh": (3, "rect"),
    "2dcorectmesh": (2, "regular"),
    "3dcorectmesh": (3, "regular"),
}

# ----- number types -----------------------------------------------------------

_NUMBER_TYPES: dict[tuple[str, int], np.dtype] = {
    ("float", 4): np.dtype("float32"),
    ("float", 8): np.dtype("float64"),
    ("int", 1): np.dtype("int8"),
    ("int", 2): np.dtype("int16"),
    ("int", 4): np.dtype("int32"),
    ("int", 8): np.dtype("int64"),
    ("uint", 1): np.dtype("uint8"),
    ("uint", 2): np.dtype("uint16"),
    ("uint", 4): np.dtype("uint32"),
    ("uint", 8): np.dtype("uint64"),
    ("char", 1): np.dtype("int8"),
    ("uchar", 1): np.dtype("uint8"),
}
_DEFAULT_PRECISION: dict[str, int] = {
    "float": 4,
    "int": 4,
    "uint": 4,
    "char": 1,
    "uchar": 1,
}
_NUMBER_TYPE_OF_KIND: dict[str, str] = {"f": "Float", "i": "Int", "u": "UInt"}

# Keys the writer spells on its own rather than as Information or a
# Grid-centred attribute.
_TIME_KEY: str = "time"
_RESERVED_GLOBALS: frozenset[str] = frozenset({_TIME_KEY, "was_2d"})

_MESH_GRID_NAME: str = "mesh"
_SERIES_GRID_NAME: str = "TimeSeries"
_H5_SUFFIX: str = ".h5"
_BIN_SUFFIX: str = ".bin"

# Values per line for a flat array written inline.
_INLINE_PER_ROW: int = 12

# The include pointers met in practice are a small XPath dialect:
# ``xpointer(//Grid[@Name="a"]/Grid[@Name="b"]/*[self::Topology or self::Geometry])``
# from one writer, ``xpointer(//Grid[@Name="a"]/Grid[1]/*[...])`` from another,
# ``xpointer(/Xdmf/Domain/Grid[@Name="a"]/Topology)`` from a third. Each step
# is a tag or ``*`` with at most one predicate: a Name, a 1-based position, or
# the ``self::`` tests that pick the children wanted.
_XPOINTER: re.Pattern[str] = re.compile(r"^\s*xpointer\((.*)\)\s*$", re.S)
_POINTER_STEP: re.Pattern[str] = re.compile(r"(\*|\w+)(?:\[([^\]]*)\])?")
_NAME_TEST: re.Pattern[str] = re.compile(r"""^@Name\s*=\s*(["'])(.*)\1$""", re.S)
_SELF_TAGS: re.Pattern[str] = re.compile(r"self::(\w+)")
_DRIVE: re.Pattern[str] = re.compile(r"^[A-Za-z]:[\\/]")


# =============================================================================
# Reading
# =============================================================================


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read an XDMF file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.xdmf`` / ``.xmf`` file, or an open file object. A file
        whose arrays live in an HDF5 or binary sidecar needs a path, since
        the sidecar is found beside it; one holding its arrays inline reads
        from anything.
    lazy
        Not supported; raises LazyReadError when True.
    **opts
        ``step`` picks the time step of a temporal collection, counted from
        zero and negative from the end the way a list is; the first is read
        without it. Every uniform grid at that step is merged into the one
        mesh, each grid's elements tagged with the grid's name when there
        are several.

    Returns
    -------
    PolyData
        The mesh. A ``<Time>`` value lands in ``global_attrs["time"]``, a
        ``<Set>`` in the tags, an ``<Information>`` in ``global_attrs`` as
        text and a Grid-centred ``<Attribute>`` there as an array.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    CodecError
        If the file is not XDMF, a grid lacks its topology or geometry, a
        ``DataItem`` declares a size it does not hold, an HDF5 or binary
        reference resolves outside the file's own directory, or ``step``
        names a step the file does not have.
    UnsupportedFormatError
        If an HDF5 reference is met and h5py is not installed. The message
        names the extra that installs it.
    UnknownElementTypeError
        If a mixed topology carries a cell code XDMF does not define.
    """
    if lazy:
        raise LazyReadError(
            f"'{source_name(path)}': XDMF lazy reads are not supported; the"
            " arrays live in the sidecar the file names and are decoded whole."
        )
    step = opts.pop("step", None)
    _warn_unknown_opts(opts, "read")
    ctx = _open_document(path)
    try:
        return _read_document(ctx, step=step)
    finally:
        _close_document(ctx)


def read_time_series(path: Source, **opts: Any) -> list[PolyData]:
    """Read every step of a temporal collection, one PolyData each.

    Parameters
    ----------
    path
        Path to the ``.xdmf`` / ``.xmf`` file, or an open file object.
    **opts
        None are taken; any given is warned about and ignored.

    Returns
    -------
    list of PolyData
        One mesh per step, in the collection's order, each carrying its time
        under ``global_attrs["time"]``. A file with no temporal collection is
        one step.
    """
    _warn_unknown_opts(opts, "read")
    ctx = _open_document(path)
    # The arrays every step includes from one grid are read once and copied
    # out per step, not decoded again for each.
    ctx["item_cache"] = {}
    try:
        n_steps = _count_steps(ctx)
        if n_steps == 0:
            return [_read_document(ctx, step=None)]
        return [_read_document(ctx, step=k) for k in range(n_steps)]
    finally:
        _close_document(ctx)


def _warn_unknown_opts(opts: dict[str, Any], what: str) -> None:
    if opts:
        warnings.warn(
            f"{EXTENSION} {what}: unrecognized options {set(opts)}; ignored.",
            UserWarning,
            stacklevel=3,
        )


def _open_document(path: Source) -> dict[str, Any]:
    """Parse the XML and gather what every DataItem lookup needs."""
    raw = read_bytes(path)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise CodecError(
            f"'{source_name(path)}': not well-formed XML ({exc})."
        ) from None
    if root.tag.lower() != "xdmf":
        raise CodecError(
            f"'{source_name(path)}': the root element is <{root.tag}>, not"
            " <Xdmf>; this is not an XDMF file."
        )
    base_dir = None if is_buffer(path) else Path(path).resolve().parent  # type: ignore[arg-type]
    return {
        "root": root,
        "name": source_name(path),
        "base_dir": base_dir,
        "file_size": len(raw),
        "h5_files": {},
        "holders": None,
        "shared_items": set(),
        "item_cache": None,
        "pointer_cache": {},
        "temporal": None,
        "time": None,
    }


def _close_document(ctx: dict[str, Any]) -> None:
    for handle in ctx["h5_files"].values():
        handle.close()
    ctx["h5_files"].clear()
    ctx["item_cache"] = None


def _grid_kind(grid: ET.Element) -> str:
    """Say what a ``<Grid>`` is: uniform, temporal, spatial, tree or subset."""
    grid_type = grid.get("GridType", "Uniform").lower()
    if grid_type == "collection":
        return grid.get("CollectionType", "Spatial").lower()
    return grid_type


def _temporal_collections(ctx: dict[str, Any]) -> list[ET.Element]:
    """The file's temporal collections, found once: a series asks per step."""
    if ctx["temporal"] is None:
        ctx["temporal"] = [
            g for g in ctx["root"].iter("Grid") if _grid_kind(g) == "temporal"
        ]
    return ctx["temporal"]


def _count_steps(ctx: dict[str, Any]) -> int:
    """How many steps the file's time series holds, 0 for a file with none."""
    temporal = _temporal_collections(ctx)
    if not temporal:
        return 0
    return len(temporal[0].findall("Grid"))


def _pointer_targets(pointer: str, ctx: dict[str, Any]) -> list[ET.Element] | None:
    """The elements an ``xpointer(...)`` names, or None for a form not read here.

    Every step of a series carries the same pointer, and resolving one
    walks the tree, so the answer is kept per document.
    """
    cache: dict[str, list[ET.Element] | None] = ctx["pointer_cache"]
    if pointer not in cache:
        cache[pointer] = _resolve_pointer(pointer, ctx["root"])
    return cache[pointer]


def _resolve_pointer(pointer: str, root: ET.Element) -> list[ET.Element] | None:
    """Walk an ``xpointer(...)`` path down the tree.

    Each step of the path narrows a set of elements: ``//`` at the front
    searches the whole tree for the first step, every later step looks at
    the children. A ``[@Name="x"]`` keeps the elements so named, a ``[n]``
    the n-th match under each parent, and a ``[self::A or self::B]`` on a
    ``*`` step the children of those tags.
    """
    match = _XPOINTER.match(pointer)
    if match is None:
        return None
    path = match.group(1).strip()
    steps = _POINTER_STEP.findall(path)
    if not steps:
        return None
    descend = path.startswith("//")
    if not descend and steps[0][0].lower() == "xdmf" and not steps[0][1]:
        steps = steps[1:]
    nodes: list[ET.Element] = [root]
    for index, (tag, predicate) in enumerate(steps):
        narrowed: list[ET.Element] = []
        for node in nodes:
            deep = index == 0 and descend
            found = _apply_predicate(_matching(node, tag, deep=deep), predicate)
            # A named grid is looked for among the descendants too: writers
            # spell the chain loosely, and a name is unambiguous where a
            # position is not.
            if not found and not deep and _NAME_TEST.match(predicate.strip()):
                found = _apply_predicate(_matching(node, tag, deep=True), predicate)
            narrowed.extend(found)
        nodes = narrowed
        if not nodes:
            break
    return nodes


def _matching(node: ET.Element, tag: str, *, deep: bool) -> list[ET.Element]:
    pool: Iterable[ET.Element] = (
        (e for e in node.iter() if e is not node) if deep else node
    )
    return [e for e in pool if tag == "*" or e.tag == tag]


def _apply_predicate(found: list[ET.Element], predicate: str) -> list[ET.Element]:
    predicate = predicate.strip()
    if not predicate:
        return found
    if predicate.isdigit():
        position = int(predicate)
        return found[position - 1 : position] if position >= 1 else []
    name = _NAME_TEST.match(predicate)
    if name is not None:
        return [e for e in found if e.get("Name") == name.group(2)]
    tags = set(_SELF_TAGS.findall(predicate))
    if tags:
        return [e for e in found if e.tag in tags]
    return []


def _shared(ctx: dict[str, Any]) -> tuple[set[int], set[int]]:
    """What the ``xi:include`` pointers draw from: the lending grids, and their DataItems.

    By identity rather than name: two grids may share a name, and a
    positional pointer names none. The DataItems are what a series reads
    once and keeps, rather than once per step.
    """
    root: ET.Element = ctx["root"]
    parents = {id(child): parent for parent in root.iter() for child in parent}
    holders: set[int] = set()
    items: set[int] = set()
    for include in root.iter(f"{_XI}include"):
        targets = _pointer_targets(include.get("xpointer", ""), ctx)
        for target in targets or ():
            owner = target if target.tag == "Grid" else parents.get(id(target))
            if owner is not None and owner.tag == "Grid":
                holders.add(id(owner))
            items.update(id(item) for item in target.iter("DataItem"))
    return holders, items


def _read_document(ctx: dict[str, Any], *, step: int | None) -> PolyData:
    root = ctx["root"]
    ctx["time"] = None
    domains = [child for child in root if child.tag == "Domain"]
    if not domains:
        raise CodecError(f"'{ctx['name']}': the file holds no <Domain>.")
    if step is not None and not _temporal_collections(ctx):
        raise CodecError(
            f"'{ctx['name']}': step={step} was asked for, and the file holds"
            " no temporal collection to take a step of."
        )
    if ctx["holders"] is None:
        ctx["holders"], ctx["shared_items"] = _shared(ctx)
    holders = ctx["holders"]
    leaves: list[ET.Element] = []
    globals_: dict[str, Any] = {}
    for domain in domains:
        _read_information(domain, globals_)
        leaves.extend(_leaves(domain, step, ctx, holders))
    if not leaves:
        raise CodecError(f"'{ctx['name']}': the file holds no grid to read.")

    polys = [_read_uniform(grid, ctx) for grid in leaves]
    if len(polys) == 1:
        poly = polys[0]
        merged_globals = {**globals_, **poly.global_attrs}
    else:
        poly, merged_globals = _merge_grids(leaves, polys, globals_)
    if ctx["time"] is not None:
        merged_globals[_TIME_KEY] = ctx["time"]
    return PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
        vertex_attrs=poly.vertex_attrs,
        element_attrs=poly.element_attrs,
        vertex_tags=poly.vertex_tags,
        element_tags=poly.element_tags,
        global_attrs=merged_globals,
    )


def _merge_grids(
    leaves: list[ET.Element], polys: list[PolyData], globals_: dict[str, Any]
) -> tuple[PolyData, dict[str, Any]]:
    """Merge several uniform grids, tagging each grid's elements by its name."""
    tagged: list[PolyData] = []
    merged_globals = dict(globals_)
    for index, (grid, poly) in enumerate(zip(leaves, polys)):
        name = grid.get("Name") or f"grid_{index}"
        counter = 0
        while name in poly.element_tags:
            name = f"grid_{index}" if counter == 0 else f"grid_{index}_{counter}"
            counter += 1
        tags = {**poly.element_tags, name: np.arange(len(poly.element_types))}
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
                global_attrs={},
            )
        )
        merged_globals.update(poly.global_attrs)
    return transforms.merge(*tagged), merged_globals


def _leaves(
    parent: ET.Element,
    step: int | None,
    ctx: dict[str, Any],
    holders: set[int],
) -> list[ET.Element]:
    """The uniform grids under *parent* that belong to the step being read."""
    out: list[ET.Element] = []
    grids = parent.findall("Grid")
    beside_a_series = any(_grid_kind(g) == "temporal" for g in grids)
    for grid in grids:
        kind = _grid_kind(grid)
        if kind == "uniform":
            # A grid another grid includes its mesh from, lying beside a time
            # series rather than in it, is the series' shared mesh and not a
            # mesh of its own: reading it too would draw the mesh twice.
            if beside_a_series and id(grid) in holders:
                continue
            out.append(grid)
        elif kind == "temporal":
            steps = grid.findall("Grid")
            if not steps:
                raise CodecError(
                    f"'{ctx['name']}': the temporal collection"
                    f" '{grid.get('Name', '')}' holds no step."
                )
            k = 0 if step is None else step
            if k < 0:
                k += len(steps)
            if not 0 <= k < len(steps):
                raise CodecError(
                    f"'{ctx['name']}': step={step} is out of range; the"
                    f" temporal collection holds {len(steps)} step(s)."
                )
            chosen = steps[k]
            time = _step_time(grid, chosen, k, len(steps), ctx)
            if time is not None:
                ctx["time"] = time
            if _grid_kind(chosen) == "uniform":
                out.append(chosen)
            else:
                out.extend(_leaves(chosen, step, ctx, holders))
        elif kind in ("spatial", "tree"):
            out.extend(_leaves(grid, step, ctx, holders))
        else:
            warnings.warn(
                f"'{ctx['name']}': grid '{grid.get('Name', '')}' is a"
                f" {kind} grid, which is not read; skipped.",
                UserWarning,
                stacklevel=4,
            )
    return out


def _step_time(
    collection: ET.Element,
    chosen: ET.Element,
    k: int,
    n_steps: int,
    ctx: dict[str, Any],
) -> float | None:
    """The time of step *k*: the step's own ``<Time>``, else the collection's.

    A collection's ``<Time>`` spells every step at once: a ``List`` of
    values, a ``HyperSlab`` of start and stride, or a ``Range`` of first and
    last, spread evenly over the steps.
    """
    own = chosen.find("Time")
    if own is not None and own.get("Value") is not None:
        return _as_float(own.get("Value"), ctx, "Time Value")
    shared = collection.find("Time")
    if shared is None:
        return None
    time_type = shared.get("TimeType", "Single").lower()
    if time_type == "single" and shared.get("Value") is not None:
        return _as_float(shared.get("Value"), ctx, "Time Value")
    item = shared.find("DataItem")
    if item is None:
        return None
    values = _item_array(item, ctx).ravel().astype(np.float64)
    if time_type == "list":
        return float(values[k]) if k < values.size else None
    if time_type == "hyperslab" and values.size >= 2:
        return float(values[0] + k * values[1])
    if time_type == "range" and values.size >= 2:
        if n_steps < 2:
            return float(values[0])
        return float(values[0] + k * (values[1] - values[0]) / (n_steps - 1))
    return None


def _as_float(text: str | None, ctx: dict[str, Any], what: str) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        raise CodecError(f"'{ctx['name']}': {what} '{text}' is not a number.") from None


def _as_int(text: str | None, ctx: dict[str, Any], what: str) -> int:
    try:
        return int(str(text).strip())
    except ValueError:
        raise CodecError(
            f"'{ctx['name']}': {what} '{text}' is not a whole number."
        ) from None


def _read_information(parent: ET.Element, globals_: dict[str, Any]) -> None:
    """Collect ``<Information Name Value>`` entries; a repeated name lists."""
    for info in parent.findall("Information"):
        name = info.get("Name")
        # A text under the planarity key would be read as a flag on the way
        # back out; a text "time" is harmless, and stays as text.
        if not name or name == "was_2d":
            continue
        value = info.get("Value")
        if value is None:
            value = (info.text or "").strip()
        held = globals_.get(name)
        if held is None:
            globals_[name] = value
        elif isinstance(held, list):
            held.append(value)
        else:
            globals_[name] = [held, value]


# ----- one uniform grid ------------------------------------------------------


def _grid_children(grid: ET.Element, ctx: dict[str, Any]) -> list[ET.Element]:
    """The grid's children, with every ``xi:include`` replaced by its target."""
    children: list[ET.Element] = []
    for child in grid:
        if child.tag != f"{_XI}include":
            children.append(child)
            continue
        children.extend(_included(child, grid, ctx))
    return children


def _included(
    include: ET.Element, grid: ET.Element, ctx: dict[str, Any]
) -> list[ET.Element]:
    """Resolve one ``xi:include`` to the elements it points at, or none."""
    where = f"'{ctx['name']}': grid '{grid.get('Name', '')}'"
    if include.get("href"):
        warnings.warn(
            f"{where} includes another document ('{include.get('href')}'),"
            " which is not followed; the include is skipped.",
            UserWarning,
            stacklevel=6,
        )
        return []
    pointer = include.get("xpointer", "")
    targets = _pointer_targets(pointer, ctx)
    if targets is None:
        warnings.warn(
            f"{where} carries an include pointer '{pointer}' of a form not"
            " read here; the include is skipped.",
            UserWarning,
            stacklevel=6,
        )
        return []
    # A pointer at a whole grid lends everything in it but its own nested
    # grids and includes, which belong to that grid's reading, not this one.
    found = [
        child
        for target in targets
        for child in ([target] if target.tag != "Grid" else target)
        if child.tag != "Grid" and child.tag != f"{_XI}include"
    ]
    if found:
        return found
    warnings.warn(
        f"{where} includes '{pointer}', which names nothing in the file;"
        " the include is skipped.",
        UserWarning,
        stacklevel=6,
    )
    return []


def _read_uniform(grid: ET.Element, ctx: dict[str, Any]) -> PolyData:
    """Read one ``<Grid GridType="Uniform">`` into a PolyData."""
    where = f"'{ctx['name']}': grid '{grid.get('Name', '')}'"
    children = _grid_children(grid, ctx)
    topology = next((c for c in children if c.tag == "Topology"), None)
    geometry = next((c for c in children if c.tag == "Geometry"), None)
    if topology is None:
        raise CodecError(f"{where} has no <Topology>.")
    if geometry is None:
        raise CodecError(f"{where} has no <Geometry>.")

    topo_type = (topology.get("TopologyType") or topology.get("Type") or "").lower()
    if not topo_type:
        raise CodecError(f"{where}: its <Topology> names no TopologyType.")

    if topo_type in _STRUCTURED:
        vertices, connectivity, offsets, element_types, globals_ = _read_structured(
            topology, geometry, topo_type, ctx, where
        )
        kept: np.ndarray | None = None
    else:
        vertices, globals_ = _read_geometry(geometry, ctx, where)
        connectivity, offsets, element_types, kept = _read_topology(
            topology, topo_type, vertices.shape[0], ctx, where
        )

    n_verts = vertices.shape[0]
    n_elems = len(element_types)
    vertex_attrs: dict[str, np.ndarray] = {}
    element_attrs: dict[str, np.ndarray] = {}
    vertex_tags: dict[str, np.ndarray] = {}
    element_tags: dict[str, np.ndarray] = {}

    for child in children:
        if child.tag == "Attribute":
            _read_attribute(
                child,
                ctx,
                where,
                n_verts,
                n_elems,
                kept,
                vertex_attrs,
                element_attrs,
                globals_,
            )
        elif child.tag == "Set":
            _read_set(
                child, ctx, where, n_verts, n_elems, kept, vertex_tags, element_tags
            )
        elif child.tag == "Time" and child.get("Value") is not None:
            globals_[_TIME_KEY] = _as_float(child.get("Value"), ctx, "Time Value")
    _read_information(grid, globals_)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=globals_,
    )


def _read_geometry(
    geometry: ET.Element, ctx: dict[str, Any], where: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an unstructured grid's points; a planar geometry marks ``was_2d``."""
    geo_type = (geometry.get("GeometryType") or geometry.get("Type") or "XYZ").upper()
    items = geometry.findall("DataItem")
    if not items:
        raise CodecError(f"{where}: its <Geometry> holds no DataItem.")
    if geo_type in ("XYZ", "XY"):
        dim = 3 if geo_type == "XYZ" else 2
        n_verts = _declared_size(items[0], ctx) // dim
        validate_header(n_verts, 0, 0, ctx["file_size"], compressed=True)
        flat = _item_array(items[0], ctx)
        if flat.size % dim:
            raise CodecError(
                f"{where}: a {geo_type} geometry holds {flat.size} values,"
                f" which is not a whole number of {dim}-tuples."
            )
        coords = flat.reshape(-1, dim)
    elif geo_type in ("X_Y_Z", "X_Y"):
        dim = 3 if geo_type == "X_Y_Z" else 2
        if len(items) < dim:
            raise CodecError(
                f"{where}: a {geo_type} geometry needs {dim} DataItems and"
                f" holds {len(items)}."
            )
        n_verts = _declared_size(items[0], ctx)
        validate_header(n_verts, 0, 0, ctx["file_size"], compressed=True)
        axes = [_item_array(item, ctx).ravel() for item in items[:dim]]
        if len({axis.size for axis in axes}) != 1:
            raise CodecError(
                f"{where}: the {geo_type} geometry's axes differ in length"
                f" ({[axis.size for axis in axes]})."
            )
        coords = np.column_stack(axes)
    else:
        raise CodecError(
            f"{where}: GeometryType '{geo_type}' is not read for an"
            " unstructured topology."
        )
    return pad_to_3d(coords, dim), mark_2d(dim)


def _read_topology(
    topology: ET.Element,
    topo_type: str,
    n_verts: int,
    ctx: dict[str, Any],
    where: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read an unstructured topology into CSR connectivity.

    Returns the connectivity, offsets and element types, plus the mask of the
    file's cells that were kept - None when every cell was - so the arrays
    that are one value per cell can be filtered to match.
    """
    item = topology.find("DataItem")
    declared = topology.get("NumberOfElements") or topology.get("Dimensions")
    n_declared = None
    if declared is not None:
        n_declared = math.prod(_dims_of(declared, ctx))
    if item is None:
        if n_declared == 0:
            return _empty_topology()
        raise CodecError(f"{where}: its <Topology> holds no DataItem.")
    size = _declared_size(item, ctx)
    validate_header(n_verts, n_declared or 0, size, ctx["file_size"], compressed=True)

    if topo_type == "mixed":
        stream = _item_array(item, ctx).ravel()
        return _read_mixed(stream, n_verts, ctx, where)

    if topo_type == "polyhedron":
        raise CodecError(
            f"{where}: a Polyhedron topology has no element type here and is not read."
        )
    ptype = _XDMF_TO_POLYXIOS.get(topo_type)
    if ptype is None:
        raise CodecError(
            f"{where}: TopologyType '{topology.get('TopologyType')}' is not known."
        )

    if topo_type in _POLY_XDMF:
        spelled = topology.get("NodesPerElement")
        if spelled is not None:
            per_cell = _as_int(spelled, ctx, "NodesPerElement")
        elif topo_type in _POLY_DEFAULT_NODES:
            per_cell = _POLY_DEFAULT_NODES[topo_type]
        else:
            raise CodecError(f"{where}: a Polygon topology needs NodesPerElement.")
        if per_cell < 1:
            raise CodecError(f"{where}: NodesPerElement={per_cell} is not positive.")
        ptype = _poly_type(topo_type, per_cell)
    else:
        per_cell = _XDMF_NODES[topo_type]

    flat = _item_array(item, ctx).ravel()
    if flat.size % per_cell:
        raise CodecError(
            f"{where}: the {topology.get('TopologyType')} topology holds"
            f" {flat.size} indices, which is not a whole number of"
            f" {per_cell}-node cells."
        )
    n_cells = flat.size // per_cell
    if n_declared is not None and n_declared != n_cells:
        if n_declared > n_cells:
            raise CodecError(
                f"{where}: the topology declares {n_declared} cells and holds"
                f" {n_cells}."
            )
        flat = flat[: n_declared * per_cell]
        n_cells = n_declared
    connectivity = _as_indices(flat, n_verts, where)
    offsets = np.arange(0, (n_cells + 1) * per_cell, per_cell, dtype=connectivity.dtype)
    element_types = np.full(n_cells, ELEMENT_TYPES[ptype], dtype=np.uint8)
    return connectivity, offsets, element_types, None


def _empty_topology() -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    return (
        np.zeros(0, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.zeros(0, dtype=np.uint8),
        None,
    )


def _poly_type(topo_type: str, per_cell: int) -> str:
    """The polyxios type a free-size XDMF kind reads as at this node count."""
    if topo_type == "polyvertex":
        return "vertex" if per_cell == 1 else "poly_vertex"
    if topo_type == "polyline":
        return "line" if per_cell == 2 else "poly_line"
    return "polygon"


def _read_mixed(
    stream: np.ndarray, n_verts: int, ctx: dict[str, Any], where: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Walk a Mixed topology stream: a code, a count for the free-size kinds, the nodes.

    The walk is sequential by nature - each cell's length says where the
    next one starts - so it is a Python loop, but one step per cell, not
    per node: it records where each cell's nodes lie and numpy gathers them
    in one go afterwards.
    """
    values = stream.tolist()
    n = len(values)
    starts: list[int] = []
    counts: list[int] = []
    types: list[int] = []
    kept: list[bool] = []
    n_polyhedra = 0
    truncated = f"{where}: the mixed topology ends inside a cell."
    i = 0
    while i < n:
        code = int(values[i])
        i += 1
        info = _MIXED_CELL.get(code)
        if info is None:
            if code == _POLYHEDRON_CODE:
                # A polyhedron is a face count and then each face's own
                # count and nodes; it has no element type here, so it is
                # stepped over.
                if i >= n:
                    raise CodecError(truncated)
                n_faces = int(values[i])
                i += 1
                for _ in range(n_faces):
                    if i >= n:
                        raise CodecError(truncated)
                    i += 1 + int(values[i])
                if i > n:
                    raise CodecError(truncated)
                n_polyhedra += 1
                kept.append(False)
                continue
            raise UnknownElementTypeError(EXTENSION, code)
        count, ptype_code = info
        if count == 0:
            if i >= n:
                raise CodecError(truncated)
            count = int(values[i])
            i += 1
            if count < 1:
                raise CodecError(
                    f"{where}: a mixed-topology cell of type"
                    f" {_CODE_TO_XDMF[code]} declares {count} nodes."
                )
            ptype_code = _POLY_TYPE_CODES[code][count == _POLY_SPLIT[code]]
        if i + count > n:
            raise CodecError(truncated)
        starts.append(i)
        counts.append(count)
        types.append(ptype_code)
        kept.append(True)
        i += count
    if n_polyhedra:
        warnings.warn(
            f"{where}: {n_polyhedra} polyhedron cell(s) in the mixed topology"
            " have no element type here and were dropped.",
            UserWarning,
            stacklevel=6,
        )
    cell_starts = np.asarray(starts, dtype=np.int64)
    cell_sizes = np.asarray(counts, dtype=np.int64)
    ends = np.cumsum(cell_sizes)
    local = np.arange(int(ends[-1]) if ends.size else 0) - np.repeat(
        ends - cell_sizes, cell_sizes
    )
    flat = stream[np.repeat(cell_starts, cell_sizes) + local]
    connectivity = _as_indices(flat, n_verts, where)
    offsets = np.zeros(len(counts) + 1, dtype=connectivity.dtype)
    offsets[1:] = ends
    mask = None if all(kept) else np.asarray(kept, dtype=bool)
    return connectivity, offsets, np.asarray(types, dtype=np.uint8), mask


def _as_indices(flat: np.ndarray, n_verts: int, where: str) -> np.ndarray:
    """Check a connectivity block against the point count, and size its dtype."""
    if flat.size == 0:
        return np.zeros(0, dtype=np.int32)
    flat = _whole_numbers(flat, where, "the topology")
    low = int(flat.min())
    high = int(flat.max())
    if low < 0 or high >= n_verts:
        raise CodecError(
            f"{where}: the topology indexes point {low if low < 0 else high},"
            f" and the geometry holds {n_verts} point(s)."
        )
    dtype = np.int32 if high < 2**31 else np.int64
    return flat.astype(dtype, copy=False)


def _whole_numbers(flat: np.ndarray, where: str, what: str) -> np.ndarray:
    """Indices as integers: a float block is accepted only when every value is whole."""
    if flat.dtype.kind != "f":
        return flat
    if not np.all(np.mod(flat, 1) == 0):
        raise CodecError(f"{where}: {what} holds non-integer indices.")
    return flat.astype(np.int64)


def _read_structured(
    topology: ET.Element,
    geometry: ET.Element,
    topo_type: str,
    ctx: dict[str, Any],
    where: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Expand a lattice topology into explicit points and cells."""
    dim, kind = _STRUCTURED[topo_type]
    spelled = topology.get("Dimensions") or topology.get("NumberOfElements")
    if spelled is None:
        raise CodecError(
            f"{where}: a {topology.get('TopologyType')} topology needs Dimensions."
        )
    dims = _dims_of(spelled, ctx)
    if len(dims) != dim:
        raise CodecError(
            f"{where}: a {topology.get('TopologyType')} topology names {dim}"
            f" dimensions, and Dimensions='{spelled}' names {len(dims)}."
        )
    # K J I in the file; nx ny nz here.
    counts = list(reversed(dims)) + [1] * (3 - dim)
    nx, ny, nz = counts
    if min(counts) < 1:
        raise CodecError(f"{where}: Dimensions='{spelled}' names an empty lattice.")
    n_points = nx * ny * nz
    n_cells, per_cell, _ = structured_cell_shape(nx - 1, ny - 1, nz - 1)
    validate_header(
        n_points,
        n_cells,
        n_cells * per_cell,
        ctx["file_size"],
        compressed=True,
        spells_vertices=False,
        spells_connectivity=False,
    )

    geo_type = (geometry.get("GeometryType") or geometry.get("Type") or "").upper()
    items = geometry.findall("DataItem")
    if kind == "smesh":
        if geo_type not in ("XYZ", "XY", "X_Y_Z", "X_Y", ""):
            raise CodecError(
                f"{where}: a curvilinear lattice takes an XYZ or XY geometry,"
                f" not '{geo_type}'."
            )
        vertices, globals_ = _read_geometry(geometry, ctx, where)
        if vertices.shape[0] != n_points:
            raise CodecError(
                f"{where}: the lattice has {n_points} points and the geometry"
                f" holds {vertices.shape[0]}."
            )
    elif kind == "rect":
        if len(items) < dim:
            raise CodecError(
                f"{where}: a {geo_type or 'VXVY(VZ)'} geometry needs {dim}"
                f" DataItems and holds {len(items)}."
            )
        axes = [
            _item_array(item, ctx).ravel().astype(np.float64) for item in items[:dim]
        ]
        if [axis.size for axis in axes] != counts[:dim]:
            raise CodecError(
                f"{where}: the lattice is {counts[:dim]} points along its"
                f" axes and the geometry's axes hold"
                f" {[axis.size for axis in axes]}."
            )
        axes += [np.zeros(1)] * (3 - dim)
        vertices = _lattice_points(*axes)
        globals_ = mark_2d(dim)
    else:
        if len(items) < 2:
            raise CodecError(
                f"{where}: an ORIGIN_DXDYDZ geometry needs two DataItems and"
                f" holds {len(items)}."
            )
        origin = _item_array(items[0], ctx).ravel().astype(np.float64)
        spacing = _item_array(items[1], ctx).ravel().astype(np.float64)
        if origin.size < dim or spacing.size < dim:
            raise CodecError(f"{where}: the origin and spacing need {dim} values each.")
        # Slowest axis first in the file, like Dimensions.
        origin = origin[:dim][::-1]
        spacing = spacing[:dim][::-1]
        axes = [
            origin[a] + spacing[a] * np.arange(counts[a], dtype=np.float64)
            for a in range(dim)
        ]
        axes += [np.zeros(1)] * (3 - dim)
        vertices = _lattice_points(*axes)
        globals_ = mark_2d(dim)

    connectivity, per_cell, kind_name = structured_cells(nx - 1, ny - 1, nz - 1)
    n_cells = connectivity.size // per_cell if per_cell else 0
    offsets = np.arange(0, (n_cells + 1) * per_cell, per_cell, dtype=connectivity.dtype)
    if per_cell == 0:
        offsets = np.zeros(1, dtype=np.int32)
    element_types = np.full(n_cells, ELEMENT_TYPES[kind_name], dtype=np.uint8)
    return vertices, connectivity, offsets, element_types, globals_


def _lattice_points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Every point of a lattice, x varying fastest."""
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _read_attribute(
    attribute: ET.Element,
    ctx: dict[str, Any],
    where: str,
    n_verts: int,
    n_elems: int,
    kept: np.ndarray | None,
    vertex_attrs: dict[str, np.ndarray],
    element_attrs: dict[str, np.ndarray],
    globals_: dict[str, Any],
) -> None:
    name = attribute.get("Name") or "unnamed"
    center = (attribute.get("Center") or "Node").lower()
    item_type = (attribute.get("ItemType") or "").lower()
    item = attribute.find("DataItem")
    if item is None:
        warnings.warn(
            f"{where}: attribute '{name}' holds no DataItem; skipped.",
            UserWarning,
            stacklevel=5,
        )
        return
    if item_type == "finiteelementfunction" or center not in ("node", "cell", "grid"):
        what = (
            "a finite element function"
            if item_type == "finiteelementfunction"
            else f"centred on '{attribute.get('Center')}'"
        )
        warnings.warn(
            f"{where}: attribute '{name}' is {what}, which is neither one"
            " value per point nor one per cell; skipped.",
            UserWarning,
            stacklevel=5,
        )
        return
    if center == "grid" and name in _RESERVED_GLOBALS:
        warnings.warn(
            f"{where}: grid attribute '{name}' has the name of a key polyxios"
            " sets itself; skipped.",
            UserWarning,
            stacklevel=5,
        )
        return
    values = _item_array(item, ctx)
    if center == "grid":
        globals_[name] = values
        return
    target, count = (
        (vertex_attrs, n_verts) if center == "node" else (element_attrs, n_elems)
    )
    if center == "cell" and kept is not None:
        # Shaped against the file's cell count first, so a flat block reads
        # the same way whether or not a polyhedron was dropped.
        rows = _rows_of(values, kept.size)
        if rows is None:
            warnings.warn(
                f"{where}: cell attribute '{name}' does not hold one value"
                f" per cell ({kept.size}); skipped.",
                UserWarning,
                stacklevel=5,
            )
            return
        values = rows[kept]
    rows = _rows_of(values, count)
    if rows is None:
        warnings.warn(
            f"{where}: {center} attribute '{name}' does not hold one value"
            f" per {'point' if center == 'node' else 'cell'} ({count});"
            " skipped.",
            UserWarning,
            stacklevel=5,
        )
        return
    target[name] = rows


def _rows_of(values: np.ndarray, count: int) -> np.ndarray | None:
    """Shape an attribute as one row per entity, or None when it cannot be."""
    if values.ndim == 0:
        values = values.reshape(1)
    if values.shape[0] == count:
        if values.ndim == 2 and values.shape[1] == 1:
            return values[:, 0]
        return values
    if values.size == count:
        return values.reshape(count)
    if count and values.size % count == 0:
        return values.reshape(count, -1)
    return None


def _read_set(
    set_elem: ET.Element,
    ctx: dict[str, Any],
    where: str,
    n_verts: int,
    n_elems: int,
    kept: np.ndarray | None,
    vertex_tags: dict[str, np.ndarray],
    element_tags: dict[str, np.ndarray],
) -> None:
    name = set_elem.get("Name") or "unnamed"
    set_type = (set_elem.get("SetType") or "Node").lower()
    item = set_elem.find("DataItem")
    if item is None:
        return
    if set_type not in ("node", "cell"):
        warnings.warn(
            f"{where}: set '{name}' is a {set_elem.get('SetType')} set, which"
            " names neither points nor cells; skipped.",
            UserWarning,
            stacklevel=5,
        )
        return
    ids = _whole_numbers(_item_array(item, ctx).ravel(), where, f"set '{name}'")
    if set_type == "node":
        vertex_tags[name] = member_indices(ids, n_verts).astype(np.int32)
        return
    if kept is not None:
        # Indices in the file count every cell; the mesh counts the kept ones.
        remap = np.cumsum(kept) - 1
        ids = member_indices(ids, kept.size)
        ids = remap[ids[kept[ids]]]
    element_tags[name] = member_indices(ids, n_elems).astype(np.int32)


# ----- DataItem ---------------------------------------------------------------


def _dims_of(text: str, ctx: dict[str, Any]) -> list[int]:
    parts = text.split()
    if not parts:
        raise CodecError(f"'{ctx['name']}': an empty Dimensions attribute.")
    try:
        dims = [int(part) for part in parts]
    except ValueError:
        raise CodecError(
            f"'{ctx['name']}': Dimensions='{text}' is not whole numbers."
        ) from None
    if any(d < 0 for d in dims):
        raise CodecError(f"'{ctx['name']}': Dimensions='{text}' is negative.")
    return dims


def _declared_size(item: ET.Element, ctx: dict[str, Any]) -> int:
    """How many values a DataItem says it holds, before any of them is read."""
    if item.get("Reference") is not None:
        return _declared_size(_reference_target(item, ctx), ctx)
    dims = item.get("Dimensions")
    if dims is None:
        return 0
    return math.prod(_dims_of(dims, ctx))


def _item_dtype(item: ET.Element, ctx: dict[str, Any]) -> np.dtype:
    number_type = (item.get("NumberType") or item.get("DataType") or "Float").lower()
    if number_type not in _DEFAULT_PRECISION:
        raise CodecError(f"'{ctx['name']}': NumberType '{number_type}' is not known.")
    precision = item.get("Precision")
    width = (
        _DEFAULT_PRECISION[number_type]
        if precision is None
        else _as_int(precision, ctx, "Precision")
    )
    dtype = _NUMBER_TYPES.get((number_type, width))
    if dtype is None:
        raise CodecError(
            f"'{ctx['name']}': NumberType '{number_type}' has no {width}-byte form."
        )
    return dtype


def _item_array(item: ET.Element, ctx: dict[str, Any]) -> np.ndarray:
    """Read one ``<DataItem>`` into an array of its declared shape.

    An item several grids include is read once per document when the
    caller keeps a cache, and each reader gets its own copy: the steps of a
    series must not share one array between them.
    """
    cache = ctx["item_cache"]
    if cache is None or id(item) not in ctx["shared_items"]:
        return _decode_item(item, ctx)
    held = cache.get(id(item))
    if held is None:
        held = cache[id(item)] = _decode_item(item, ctx)
    return held.copy()


def _decode_item(item: ET.Element, ctx: dict[str, Any]) -> np.ndarray:
    if item.get("Reference") is not None:
        return _item_array(_reference_target(item, ctx), ctx)
    item_type = (item.get("ItemType") or "Uniform").lower()
    if item_type == "uniform":
        return _uniform_item(item, ctx)
    if item_type == "hyperslab":
        return _hyperslab_item(item, ctx)
    raise CodecError(
        f"'{ctx['name']}': a DataItem of ItemType '{item.get('ItemType')}' is not read."
    )


def _reference_target(item: ET.Element, ctx: dict[str, Any]) -> ET.Element:
    """Follow a ``Reference`` chain to the DataItem that holds the values.

    A reference may name another reference; the chain is walked to its end,
    and one that comes back to an item already passed is refused rather
    than followed forever.
    """
    seen: set[int] = {id(item)}
    target = item
    while target.get("Reference") is not None:
        xpath = _reference_xpath(target, ctx)
        target = _find_reference(xpath, ctx)
        if id(target) in seen:
            raise CodecError(
                f"'{ctx['name']}': a DataItem Reference '{xpath}' leads back to itself."
            )
        seen.add(id(target))
    return target


def _reference_xpath(item: ET.Element, ctx: dict[str, Any]) -> str:
    reference = (item.get("Reference") or "").strip()
    xpath = (item.text or "").strip() if reference.upper() == "XML" else reference
    if not xpath:
        raise CodecError(f"'{ctx['name']}': a DataItem Reference names no path.")
    return xpath


def _find_reference(xpath: str, ctx: dict[str, Any]) -> ET.Element:
    root: ET.Element = ctx["root"]
    path = xpath
    if path.startswith("//"):
        path = "." + path
    elif path.startswith("/"):
        head, _, rest = path[1:].partition("/")
        if head.lower() != root.tag.lower():
            raise CodecError(
                f"'{ctx['name']}': a DataItem Reference '{xpath}' does not"
                " start at the document's root."
            )
        path = rest
    try:
        target = root.find(path) if path else None
    except SyntaxError:
        target = None
    if target is None:
        raise CodecError(
            f"'{ctx['name']}': a DataItem Reference '{xpath}' names nothing"
            " in the file."
        )
    return target


def _uniform_item(item: ET.Element, ctx: dict[str, Any]) -> np.ndarray:
    dtype = _item_dtype(item, ctx)
    spelled = item.get("Dimensions")
    dims = None if spelled is None else _dims_of(spelled, ctx)
    fmt = (item.get("Format") or "XML").lower()
    if fmt == "xml":
        values = _xml_values(item, dtype, ctx)
    elif fmt in ("hdf", "hdf5"):
        values = _hdf_values(item, dims, ctx)
    elif fmt == "binary":
        values = _binary_values(item, dtype, dims, ctx)
    else:
        raise CodecError(
            f"'{ctx['name']}': a DataItem Format '{item.get('Format')}' is not read."
        )
    if dims is None:
        return values
    wanted = math.prod(dims)
    if values.size < wanted:
        raise CodecError(
            f"'{ctx['name']}': a DataItem declares Dimensions='{spelled}'"
            f" ({wanted} values) and holds {values.size}."
        )
    # Dimensions is a window on the block, and other readers honour it as
    # one: a longer dataset is read to the declared size, not refused.
    if values.size > wanted:
        values = values.reshape(-1)[:wanted]
    return values.reshape(dims)


def _xml_values(item: ET.Element, dtype: np.dtype, ctx: dict[str, Any]) -> np.ndarray:
    tokens = (item.text or "").split()
    try:
        if dtype.kind == "f":
            return np.array(tokens, dtype=np.float64).astype(dtype, copy=False)
        try:
            return np.array(tokens, dtype=dtype)
        except (ValueError, OverflowError):
            # An integer block some writers spell with a decimal point, "1.0";
            # read as doubles and kept only when every value is whole.
            doubles = np.array(tokens, dtype=np.float64)
            if not np.all(np.mod(doubles, 1) == 0):
                raise ValueError from None
            return doubles.astype(dtype)
    except (ValueError, OverflowError):
        raise CodecError(
            f"'{ctx['name']}': an inline DataItem holds a value that is not a"
            f" {item.get('NumberType') or 'Float'}."
        ) from None


def _sidecar(reference: str, ctx: dict[str, Any], what: str) -> Path:
    """Resolve a sidecar file named in the XML, inside the document's own directory."""
    base_dir: Path | None = ctx["base_dir"]
    if base_dir is None:
        raise CodecError(
            f"'{ctx['name']}': the file names its arrays in {what} '{reference}',"
            " which is found beside the file, and a file object has no beside."
            " Pass a path instead."
        )
    candidate = (base_dir / reference).resolve()
    if base_dir != candidate and base_dir not in candidate.parents:
        raise CodecError(
            f"'{ctx['name']}': {what} '{reference}' resolves outside the"
            " file's own directory and is refused."
        )
    return candidate


def _hdf_dataset(item: ET.Element, ctx: dict[str, Any]) -> tuple[Any, str]:
    """Open the dataset an HDF DataItem names; the handle is kept for the read."""
    text = (item.text or "").strip()
    # The first colon splits file from dataset, a Windows drive letter's
    # excepted; a colon inside the dataset path stays with the dataset.
    drive = 2 if _DRIVE.match(text) else 0
    file_part, sep, dataset = text[drive:].partition(":")
    file_part = text[:drive] + file_part
    if not sep or not file_part or not dataset:
        raise CodecError(
            f"'{ctx['name']}': an HDF DataItem reads '{text}', not 'file.h5:/dataset'."
        )
    if not HAVE_H5PY:
        raise UnsupportedFormatError(
            f"'{ctx['name']}': the arrays live in the HDF5 file '{file_part}',"
            ' and reading it needs h5py: pip install "polyxios[hdf5]".'
        )
    path = _sidecar(file_part, ctx, "the HDF5 file")
    handle = ctx["h5_files"].get(path)
    if handle is None:
        try:
            handle = h5py.File(path, "r")
        except OSError as exc:
            raise CodecError(
                f"'{ctx['name']}': the HDF5 file '{file_part}' could not be"
                f" opened ({exc})."
            ) from None
        ctx["h5_files"][path] = handle
    if dataset not in handle:
        raise CodecError(
            f"'{ctx['name']}': '{file_part}' holds no dataset '{dataset}'."
        )
    node = handle[dataset]
    where = f"'{dataset}' in '{file_part}'"
    if not hasattr(node, "shape"):
        raise CodecError(f"'{ctx['name']}': {where} is a group, not a dataset.")
    if node.dtype.kind not in "biuf":
        raise CodecError(
            f"'{ctx['name']}': {where} holds {node.dtype}, which is not numbers."
        )
    return node, where


def _hdf_values(
    item: ET.Element, dims: list[int] | None, ctx: dict[str, Any]
) -> np.ndarray:
    node, where = _hdf_dataset(item, ctx)
    if dims is not None:
        wanted = math.prod(dims)
        if node.size < wanted:
            raise CodecError(
                f"'{ctx['name']}': {where} holds {node.size} values and the"
                f" DataItem declares {wanted}."
            )
    values = np.asarray(node[()])
    # Read in the dataset's own byte order; back to the machine's, so a
    # swapped block never travels on in the mesh.
    return values.astype(values.dtype.newbyteorder("="), copy=False)


def _plain_hdf_item(item: ET.Element) -> bool:
    """A DataItem that is one HDF5 dataset, read as is: the one h5py can slice."""
    return (
        item.get("Reference") is None
        and (item.get("ItemType") or "Uniform").lower() == "uniform"
        and (item.get("Format") or "").lower() in ("hdf", "hdf5")
    )


def _binary_values(
    item: ET.Element, dtype: np.dtype, dims: list[int] | None, ctx: dict[str, Any]
) -> np.ndarray:
    text = (item.text or "").strip()
    if not text:
        raise CodecError(f"'{ctx['name']}': a Binary DataItem names no file.")
    endian = (item.get("Endian") or "Native").lower()
    if endian == "little":
        dtype = dtype.newbyteorder("<")
    elif endian == "big":
        dtype = dtype.newbyteorder(">")
    elif endian != "native":
        raise CodecError(
            f"'{ctx['name']}': Endian '{item.get('Endian')}' is not known."
        )
    seek = 0 if item.get("Seek") is None else _as_int(item.get("Seek"), ctx, "Seek")
    path = _sidecar(text, ctx, "the binary file")
    if dims is None:
        raise CodecError(f"'{ctx['name']}': a Binary DataItem needs Dimensions.")
    count = math.prod(dims)
    try:
        available = path.stat().st_size
    except OSError as exc:
        raise CodecError(
            f"'{ctx['name']}': the binary file '{text}' could not be read ({exc})."
        ) from None
    needed = seek + count * dtype.itemsize
    if needed > available:
        raise CodecError(
            f"'{ctx['name']}': the binary file '{text}' holds {available}"
            f" bytes and the DataItem asks for {count} values ending at"
            f" byte {needed}."
        )
    values = np.fromfile(path, dtype=dtype, count=count, offset=seek)
    # Back to the machine's own byte order, so a swapped block from the file
    # never travels on in the mesh.
    return values.astype(dtype.newbyteorder("="), copy=False)


def _hyperslab_item(item: ET.Element, ctx: dict[str, Any]) -> np.ndarray:
    items = item.findall("DataItem")
    if len(items) < 2:
        raise CodecError(
            f"'{ctx['name']}': a HyperSlab DataItem needs a selection and a"
            f" source, and holds {len(items)} DataItem(s)."
        )
    selection = _item_array(items[0], ctx).ravel().astype(np.int64)
    source_item = items[1]
    spelled_source = source_item.get("Dimensions")
    source_dims = None if spelled_source is None else _dims_of(spelled_source, ctx)
    rank = None if source_dims is None else len(source_dims)

    # A slab of an HDF5 dataset is read through h5py's own slicing, so a
    # time step cut from one big dataset costs the step, not the dataset.
    node = None
    if rank is not None and _plain_hdf_item(source_item):
        node, _ = _hdf_dataset(source_item, ctx)
        if tuple(node.shape) != tuple(source_dims or ()):
            node = None
    source: np.ndarray | None = None
    if node is None:
        source = _item_array(source_item, ctx)
        rank = source.ndim
        source_shape = source.shape
    else:
        source_shape = tuple(node.shape)

    if selection.size != 3 * rank:
        raise CodecError(
            f"'{ctx['name']}': a HyperSlab selection over a rank-{rank} source"
            f" needs {3 * rank} numbers and holds {selection.size}."
        )
    start, stride, count = selection.reshape(3, rank)
    if np.any(stride < 1) or np.any(count < 0) or np.any(start < 0):
        raise CodecError(f"'{ctx['name']}': a HyperSlab selection is out of range.")
    ends = start + stride * (count - 1) + 1
    if np.any((count > 0) & (ends > np.asarray(source_shape))):
        raise CodecError(
            f"'{ctx['name']}': a HyperSlab selection reaches past its source"
            f" (shape {source_shape})."
        )
    slices = tuple(
        slice(int(s), int(s + st * c), int(st))
        for s, st, c in zip(start, stride, count)
    )
    picked = np.asarray(node[slices] if source is None else source[slices])
    spelled = item.get("Dimensions")
    if spelled is not None:
        dims = _dims_of(spelled, ctx)
        if math.prod(dims) != picked.size:
            raise CodecError(
                f"'{ctx['name']}': a HyperSlab declares Dimensions='{spelled}'"
                f" and selects {picked.size} values."
            )
        picked = picked.reshape(dims)
    return np.ascontiguousarray(picked, dtype=picked.dtype.newbyteorder("="))


# =============================================================================
# Writing
# =============================================================================


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as an XDMF file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open file object for the ``"xml"`` data
        format only: the other two write a sidecar beside the file, and a
        file object has no beside.
    **opts
        ``data_format`` says where the arrays go: ``"hdf"`` (the default)
        writes them to an HDF5 file named after this one, ``out.h5`` beside
        ``out.xdmf``, and needs h5py; ``"binary"`` writes them to one raw
        ``out.bin`` beside it; ``"xml"`` keeps them inline in the XML.
        ``compression`` and ``compression_opts`` are passed to h5py for the
        HDF5 datasets - ``compression="gzip"`` with ``compression_opts=4``
        is the usual pair - and mean nothing to the other two formats.
        ``time`` is written as the grid's ``<Time Value>``; without it, a
        number under ``global_attrs["time"]`` is.

    Raises
    ------
    CodecError
        If ``data_format`` names none of the three, a sidecar format is asked
        for on a file object or a gzip path, or the mesh's vertices are not
        a coordinate block.
    UnsupportedFormatError
        If the HDF5 format is asked for and h5py is not installed.

    Notes
    -----
    The mesh is one ``<Grid GridType="Uniform">``: a ``<Geometry>`` of
    ``XYZ`` - ``XY`` for a mesh that came from a two-dimensional file and
    stayed in the plane - and one ``<Topology>``, named for the element type
    when every element is of one type and ``Mixed`` otherwise. A pixel and a
    voxel go out as the quadrilateral and hexahedron they are, corners
    reordered; an element type XDMF has no name for is dropped with a
    warning per type, and the arrays and tags over the elements are cut to
    match. ``vertex_attrs`` and ``element_attrs`` are ``<Attribute>``
    elements centred on ``Node`` and ``Cell``; the tags are ``<Set>``
    elements of ``Node`` and ``Cell`` type; a numeric ``global_attrs`` value
    is an ``<Attribute Center="Grid">``, a text one an ``<Information>``.
    """
    data_format, sink_opts, time = _write_options(opts, "write")
    target = _target_path(path, data_format)
    with _open_sink(data_format, target, sink_opts) as store:
        body = _grid_lines(
            poly,
            store,
            prefix="mesh",
            indent=6,
            time=time,
            geometry=True,
            topology=True,
            include=None,
        )
    _write_xml(
        path,
        [
            f'    <Grid Name="{_MESH_GRID_NAME}" GridType="Uniform">',
            *body,
            "    </Grid>",
        ],
    )


def write_time_series(
    steps: Iterable[tuple[float, PolyData]], path: Source, **opts: Any
) -> None:
    """Write a sequence of meshes as one temporal collection.

    Parameters
    ----------
    steps
        ``(time, mesh)`` pairs in time order. The first mesh's topology is
        written once and every later step includes it; a later mesh whose
        vertices differ writes its own geometry, so a mesh that moves is
        held, while one whose elements differ is refused - that is another
        mesh, not another step of this one. Any iterable will do, a
        generator included: each step's arrays are written as it arrives.
    path
        Output file path.
    **opts
        As :func:`write`, ``time`` excepted: each step carries its own.

    Raises
    ------
    CodecError
        If ``steps`` is empty, or a step's elements differ from the first's.
    """
    data_format, sink_opts, _ = _write_options(
        opts, "write_time_series", takes_time=False
    )
    target = _target_path(path, data_format)
    # The first step is taken before the sidecar is opened, so an empty
    # series leaves no empty file behind.
    remaining = iter(steps)
    head = next(remaining, None)
    if head is None:
        raise CodecError(
            f"'{source_name(path)}': a time series needs at least one step."
        )
    first_time, first = head
    # Every step holds the first's elements, so which of them XDMF can hold
    # is settled once, and the warning for the rest is given once.
    kept = _kept_elements(first, stacklevel=4)
    lines: list[str] = [
        f'    <Grid Name="{_SERIES_GRID_NAME}" GridType="Collection"'
        ' CollectionType="Temporal">'
    ]
    with _open_sink(data_format, target, sink_opts) as store:
        body = _grid_lines(
            first,
            store,
            prefix="step0",
            indent=8,
            time=float(first_time),
            geometry=True,
            topology=True,
            include=None,
            kept=kept,
        )
        lines.append('      <Grid Name="step0" GridType="Uniform">')
        lines.extend(body)
        lines.append("      </Grid>")
        for index, (time, poly) in enumerate(remaining, start=1):
            _require_same_elements(first, poly, index, path)
            moved = not np.array_equal(first.vertices, poly.vertices)
            body = _grid_lines(
                poly,
                store,
                prefix=f"step{index}",
                indent=8,
                time=float(time),
                geometry=moved,
                topology=False,
                include=("Topology",) if moved else ("Topology", "Geometry"),
                kept=kept,
            )
            lines.append(f'      <Grid Name="step{index}" GridType="Uniform">')
            lines.extend(body)
            lines.append("      </Grid>")
    lines.append("    </Grid>")
    _write_xml(path, lines)


def _require_same_elements(
    first: PolyData, poly: PolyData, index: int, path: Source
) -> None:
    same = (
        np.array_equal(first.element_types, poly.element_types)
        and np.array_equal(first.offsets, poly.offsets)
        and np.array_equal(first.connectivity, poly.connectivity)
    )
    if not same:
        raise CodecError(
            f"'{source_name(path)}': step {index} holds different elements"
            " from step 0. A time series is one mesh over time; a mesh whose"
            " elements change is another mesh."
        )
    if first.vertices.shape != poly.vertices.shape:
        raise CodecError(
            f"'{source_name(path)}': step {index} holds"
            f" {poly.vertices.shape[0]} vertices and step 0 holds"
            f" {first.vertices.shape[0]}."
        )


def _write_options(
    opts: dict[str, Any], what: str, *, takes_time: bool = True
) -> tuple[str, dict[str, Any], float | None]:
    """Sort the write options; ``time`` is left in for the warning when a series is written."""
    spelled = str(opts.pop("data_format", "hdf")).strip().lower()
    data_format = {
        "hdf": "hdf",
        "hdf5": "hdf",
        "h5": "hdf",
        "xml": "xml",
        "binary": "binary",
        "bin": "binary",
    }.get(spelled)
    if data_format is None:
        raise CodecError(
            f"{EXTENSION} {what}: data_format={spelled!r} is none of 'hdf',"
            " 'xml' or 'binary'."
        )
    sink_opts = {
        "compression": opts.pop("compression", None),
        "compression_opts": opts.pop("compression_opts", None),
    }
    time = opts.pop("time", None) if takes_time else None
    if time is not None:
        try:
            time = float(time)
        except (TypeError, ValueError):
            raise CodecError(
                f"{EXTENSION} {what}: time={time!r} is not a number."
            ) from None
    _warn_unknown_opts(opts, what)
    return data_format, sink_opts, time


def _target_path(path: Source, data_format: str) -> Path | None:
    """Where the sidecar goes: beside the file, which a file object has not."""
    if data_format == "xml":
        return None
    return require_path(
        path,
        fmt=EXTENSION,
        reason=f"the arrays go to a{'n HDF5' if data_format == 'hdf' else ' binary'} file beside it",
    )


def _write_xml(path: Source, grid_lines: list[str]) -> None:
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<Xdmf Version="3.0" xmlns:xi="{_XI_URI}">',
        "  <Domain>",
        *grid_lines,
        "  </Domain>",
        "</Xdmf>",
        "",
    ]
    write_text(path, "\n".join(lines), encoding="utf-8")


# A sink stores one array and answers with the DataItem attributes that say
# where it went, and the text that goes inside the element.
Store = Callable[[str, np.ndarray], tuple[dict[str, str], str]]


@contextmanager
def _open_sink(
    data_format: str, target: Path | None, sink_opts: dict[str, Any]
) -> Iterator[Store]:
    if data_format == "xml":
        yield _xml_store
        return
    assert target is not None
    if data_format == "hdf":
        if not HAVE_H5PY:
            raise UnsupportedFormatError(
                f"'{target.name}': the arrays go to an HDF5 file beside it,"
                ' which needs h5py: pip install "polyxios[hdf5]". Or pass'
                ' data_format="xml" to keep them inline, or "binary" for a'
                " raw sidecar."
            )
        h5_path = target.with_name(target.stem + _H5_SUFFIX)
        with h5py.File(h5_path, "w") as handle:
            yield _hdf_store(handle, h5_path.name, sink_opts)
        return
    bin_path = target.with_name(target.stem + _BIN_SUFFIX)
    with open(bin_path, "wb") as handle:
        yield _binary_store(handle, bin_path.name)


def _xml_store(hint: str, arr: np.ndarray) -> tuple[dict[str, str], str]:
    if arr.size == 0:
        return {"Format": "XML"}, ""
    # numpy spells a float as the shortest string that reads back to the
    # same value, and does it in C; that is both a smaller file and several
    # times quicker than a printf format per value.
    if arr.dtype.kind == "f":
        tokens = arr.reshape(-1).astype(str).tolist()
    else:
        tokens = list(map(str, arr.reshape(-1).tolist()))
    # One value per line is a long file for a flat array; a mixed topology
    # stream or a set of ids reads as well a dozen to a row.
    per_row = _INLINE_PER_ROW if arr.ndim == 1 else math.prod(arr.shape[1:])
    rows = [" ".join(tokens[i : i + per_row]) for i in range(0, len(tokens), per_row)]
    return {"Format": "XML"}, "\n" + "\n".join(rows) + "\n"


def _hdf_store(handle: Any, h5_name: str, sink_opts: dict[str, Any]) -> Store:
    taken: set[str] = set()
    kwargs = {k: v for k, v in sink_opts.items() if v is not None}

    def store(hint: str, arr: np.ndarray) -> tuple[dict[str, str], str]:
        name = "/" + hint.replace("//", "/").strip("/")
        base, counter = name, 1
        while name in taken:
            name = f"{base}_{counter}"
            counter += 1
        taken.add(name)
        # Compression needs chunking, and h5py refuses to chunk an empty
        # dataset; nothing is lost by writing that one plain.
        extra = kwargs if arr.size else {}
        handle.create_dataset(name, data=arr, **extra)
        return {"Format": "HDF"}, f"{h5_name}:{name}"

    return store


def _binary_store(handle: Any, bin_name: str) -> Store:
    def store(hint: str, arr: np.ndarray) -> tuple[dict[str, str], str]:
        offset = handle.tell()
        # A contiguous array is its own buffer; no bytes copy on the way out.
        handle.write(np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<")))
        return {"Format": "Binary", "Endian": "Little", "Seek": str(offset)}, bin_name

    return store


def _h5_safe(name: str) -> str:
    """A dataset name from an attribute name: a slash would open a group."""
    return name.replace("/", "_") or "_"


def _storable(arr: np.ndarray) -> np.ndarray | None:
    """The array as XDMF can type it, or None when it holds no numbers."""
    if arr.dtype.kind == "b":
        return arr.astype(np.uint8)
    if arr.dtype.kind == "f" and arr.dtype.itemsize < 4:
        return arr.astype(np.float32)
    if arr.dtype.kind == "f" and arr.dtype.itemsize > 8:
        return arr.astype(np.float64)
    if arr.dtype.kind in "iu" and arr.dtype.itemsize > 8:
        return None
    if arr.dtype.kind not in "iuf":
        return None
    return arr


def _data_item(arr: np.ndarray, store: Store, hint: str, indent: int) -> str:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    number_type = _NUMBER_TYPE_OF_KIND[arr.dtype.kind]
    dims = " ".join(str(n) for n in arr.shape)
    attrs, text = store(hint, arr)
    spelled = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    pad = " " * indent
    if "\n" in text:
        rows = text.strip("\n").split("\n")
        text = "".join(f"\n{pad}  {row}" for row in rows) + f"\n{pad}"
    return (
        f'{pad}<DataItem Dimensions="{dims}" NumberType="{number_type}"'
        f' Precision="{arr.dtype.itemsize}" {spelled}>{text}</DataItem>'
    )


def _grid_lines(
    poly: PolyData,
    store: Store,
    *,
    prefix: str,
    indent: int,
    time: float | None,
    geometry: bool,
    topology: bool,
    include: tuple[str, ...] | None,
    kept: np.ndarray | None = None,
) -> list[str]:
    """The children of one ``<Grid>``, as lines.

    ``kept`` is the mask of elements XDMF can hold, when the caller has
    settled it already; otherwise it is found here, with the warning for
    the ones it cannot.
    """
    pad = " " * indent
    lines: list[str] = []
    if time is None:
        held = poly.global_attrs.get(_TIME_KEY)
        if isinstance(held, (int, float, np.integer, np.floating)) and not isinstance(
            held, bool
        ):
            time = float(held)
    if time is not None:
        lines.append(f'{pad}<Time Value="{time!r}"/>')
    if include:
        pointer = "".join(
            f"/Grid[@Name=&quot;{name}&quot;]" for name in (_SERIES_GRID_NAME, "step0")
        )
        tests = " or ".join(f"self::{tag}" for tag in include)
        lines.append(f'{pad}<xi:include xpointer="xpointer(/{pointer}/*[{tests}])"/>')

    if geometry:
        dim = output_dimension(poly, fmt=EXTENSION, flat_default=3, stacklevel=5)
        coords = np.ascontiguousarray(poly.vertices[:, :dim], dtype=np.float64)
        lines.append(f'{pad}<Geometry GeometryType="{"XYZ" if dim == 3 else "XY"}">')
        lines.append(_data_item(coords, store, f"{prefix}/geometry", indent + 2))
        lines.append(f"{pad}</Geometry>")

    if kept is None:
        kept = _kept_elements(poly, stacklevel=6)
    n_elems = int(kept.sum())
    if topology:
        lines.extend(_topology_lines(poly, kept, n_elems, store, prefix, indent))

    n_verts = poly.vertices.shape[0]
    lines.extend(
        _attribute_lines(
            spellable_arrays(
                poly.vertex_attrs, fmt=EXTENSION, kind="point", stacklevel=5
            ),
            "Node",
            n_verts,
            None,
            store,
            f"{prefix}/node",
            indent,
        )
    )
    lines.extend(
        _attribute_lines(
            spellable_arrays(
                poly.element_attrs, fmt=EXTENSION, kind="cell", stacklevel=5
            ),
            "Cell",
            len(poly.element_types),
            kept,
            store,
            f"{prefix}/cell",
            indent,
        )
    )
    numeric = globals_for_write(
        poly, reserved=_RESERVED_GLOBALS, fmt=EXTENSION, text=True, stacklevel=5
    )
    lines.extend(
        _attribute_lines(
            spellable_arrays(numeric, fmt=EXTENSION, kind="field", stacklevel=5),
            "Grid",
            None,
            None,
            store,
            f"{prefix}/grid",
            indent,
        )
    )
    lines.extend(
        _set_lines(
            spellable_arrays(
                poly.vertex_tags, fmt=EXTENSION, kind="point", stacklevel=5
            ),
            "Node",
            n_verts,
            None,
            store,
            f"{prefix}/node_sets",
            indent,
        )
    )
    lines.extend(
        _set_lines(
            spellable_arrays(
                poly.element_tags, fmt=EXTENSION, kind="cell", stacklevel=5
            ),
            "Cell",
            len(poly.element_types),
            kept,
            store,
            f"{prefix}/cell_sets",
            indent,
        )
    )
    lines.extend(
        f"{pad}<Information Name={quoteattr(str(name))} Value={quoteattr(value)}/>"
        for name, strings in text_for_write(poly, reserved=_RESERVED_GLOBALS).items()
        for value in strings
    )
    return lines


def _kept_elements(poly: PolyData, *, stacklevel: int) -> np.ndarray:
    """The mask of elements XDMF can hold, warning once for the ones it cannot."""
    kept, dropped_types = _writable_elements(poly)
    if dropped_types:
        warnings.warn(
            f"{EXTENSION} write: element type(s) {sorted(dropped_types)} have"
            " no XDMF topology; those elements and the values and tags on"
            " them are not written.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return kept


def _writable_elements(poly: PolyData) -> tuple[np.ndarray, set[str]]:
    """The mask of elements XDMF can hold, and the names of the types it cannot."""
    codes = np.asarray(poly.element_types)
    kept = np.ones(len(codes), dtype=bool)
    dropped: set[str] = set()
    sizes = np.diff(poly.offsets)
    for code in np.unique(codes):
        code = int(code)
        if code in _WRITE_MAP:
            continue
        here = codes == code
        if code == _LAGRANGE_HEX:
            fits = np.isin(sizes[here], _LAGRANGE_HEX_NODES)
            if fits.all():
                continue
            kept[np.flatnonzero(here)[~fits]] = False
        else:
            kept[here] = False
        dropped.add(ELEMENT_TYPES_INV.get(code, f"type_{code}"))
    return kept, dropped


def _xdmf_name(code: int, size: int) -> tuple[str, tuple[int, ...] | None]:
    if code == _LAGRANGE_HEX:
        return f"Hexahedron_{size}", None
    return _WRITE_MAP[code]


def _topology_lines(
    poly: PolyData,
    kept: np.ndarray,
    n_elems: int,
    store: Store,
    prefix: str,
    indent: int,
) -> list[str]:
    pad = " " * indent
    codes = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets)
    sizes = np.diff(offsets)
    conn = np.asarray(poly.connectivity)
    if n_elems == 0:
        empty = np.zeros(0, dtype=np.int64)
        return [
            f'{pad}<Topology TopologyType="Polyvertex" NumberOfElements="0" NodesPerElement="1">',
            _data_item(empty, store, f"{prefix}/topology", indent + 2),
            f"{pad}</Topology>",
        ]
    index = np.flatnonzero(kept)
    names = {_xdmf_name(int(c), int(s))[0] for c, s in zip(codes[index], sizes[index])}
    uniform_size = len(np.unique(sizes[index])) == 1
    if len(names) == 1 and uniform_size:
        name = next(iter(names))
        per_cell = int(sizes[index[0]])
        cells = _reordered_cells(conn, offsets, codes, index, per_cell)
        extra = (
            f' NodesPerElement="{per_cell}"'
            if name in ("Polyvertex", "Polyline", "Polygon")
            else ""
        )
        return [
            f'{pad}<Topology TopologyType="{name}" NumberOfElements="{n_elems}"{extra}>',
            _data_item(cells, store, f"{prefix}/topology", indent + 2),
            f"{pad}</Topology>",
        ]
    stream = _mixed_stream(conn, offsets, codes, sizes, index)
    return [
        f'{pad}<Topology TopologyType="Mixed" NumberOfElements="{n_elems}">',
        _data_item(stream, store, f"{prefix}/topology", indent + 2),
        f"{pad}</Topology>",
    ]


def _reordered_cells(
    conn: np.ndarray,
    offsets: np.ndarray,
    codes: np.ndarray,
    index: np.ndarray,
    per_cell: int,
) -> np.ndarray:
    starts = offsets[index]
    cells = conn[starts[:, None] + np.arange(per_cell)[None, :]].astype(np.int64)
    for code in np.unique(codes[index]):
        order = _WRITE_MAP.get(int(code), ("", None))[1]
        if order is not None:
            rows = codes[index] == code
            cells[rows] = cells[rows][:, list(order)]
    return np.asarray(cells)


def _mixed_stream(
    conn: np.ndarray,
    offsets: np.ndarray,
    codes: np.ndarray,
    sizes: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Lay the kept cells out as a Mixed topology, without a Python loop per cell."""
    kept_codes = codes[index]
    kept_sizes = sizes[index].astype(np.int64)
    xdmf_codes = np.empty(len(index), dtype=np.int64)
    is_poly = np.zeros(len(index), dtype=bool)
    for code in np.unique(kept_codes):
        rows = kept_codes == code
        if int(code) == _LAGRANGE_HEX:
            for size in np.unique(kept_sizes[rows]):
                xdmf_codes[rows & (kept_sizes == size)] = _XDMF_TO_CODE[
                    f"hexahedron_{int(size)}"
                ]
            continue
        name = _WRITE_MAP[int(code)][0].lower()
        xdmf_codes[rows] = _XDMF_TO_CODE[name]
        if name in _POLY_XDMF:
            is_poly[rows] = True
    head = np.where(is_poly, 2, 1)
    out_sizes = head + kept_sizes
    starts = np.concatenate([[0], np.cumsum(out_sizes)[:-1]])
    stream = np.empty(int(out_sizes.sum()), dtype=np.int64)
    stream[starts] = xdmf_codes
    stream[starts[is_poly] + 1] = kept_sizes[is_poly]
    cell_of = np.repeat(np.arange(len(index)), kept_sizes)
    local = np.arange(int(kept_sizes.sum())) - np.repeat(
        np.cumsum(kept_sizes) - kept_sizes, kept_sizes
    )
    node_pos = starts[cell_of] + head[cell_of] + local
    cell_starts = np.cumsum(kept_sizes) - kept_sizes
    gathered = conn[np.repeat(offsets[index], kept_sizes) + local].astype(np.int64)
    # A pixel or voxel in a mixed stream is reordered the same way as alone.
    for code in np.unique(kept_codes):
        order = _WRITE_MAP.get(int(code), ("", None))[1]
        if order is None:
            continue
        rows = np.flatnonzero(kept_codes == code)
        slots = cell_starts[rows][:, None] + np.arange(len(order))[None, :]
        gathered[slots] = gathered[slots][:, list(order)]
    stream[node_pos] = gathered
    return stream


def _attribute_type(shape: tuple[int, ...]) -> str:
    if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
        return "Scalar"
    if len(shape) == 2 and shape[1] in (2, 3):
        return "Vector"
    if len(shape) == 2 and shape[1] == 6:
        return "Tensor6"
    if (len(shape) == 2 and shape[1] == 9) or shape[1:] == (3, 3):
        return "Tensor"
    return "Matrix"


def _attribute_lines(
    arrays: dict[str, Any],
    center: str,
    count: int | None,
    kept: np.ndarray | None,
    store: Store,
    prefix: str,
    indent: int,
) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    unspellable: list[str] = []
    misfit: list[str] = []
    for name, held in arrays.items():
        arr = _storable(np.asarray(held))
        if arr is None:
            unspellable.append(str(name))
            continue
        if count is not None:
            if arr.ndim == 0 or arr.shape[0] != count:
                misfit.append(str(name))
                continue
            if kept is not None:
                arr = arr[kept]
        elif arr.ndim == 0:
            arr = arr.reshape(1)
        lines.append(
            f'{pad}<Attribute Name={quoteattr(str(name))} Center="{center}"'
            f' AttributeType="{_attribute_type(arr.shape)}">'
        )
        lines.append(
            _data_item(arr, store, f"{prefix}/{_h5_safe(str(name))}", indent + 2)
        )
        lines.append(f"{pad}</Attribute>")
    kind = {"Node": "vertex", "Cell": "element", "Grid": "global"}[center]
    if unspellable:
        warnings.warn(
            f"{EXTENSION} write: {kind} attribute(s) {sorted(unspellable)} hold"
            " no numbers XDMF can type; those arrays are not written.",
            UserWarning,
            stacklevel=6,
        )
    if misfit:
        warnings.warn(
            f"{EXTENSION} write: {kind} attribute(s) {sorted(misfit)} are not"
            f" one value per {kind} ({count}); those arrays are not written.",
            UserWarning,
            stacklevel=6,
        )
    return lines


def _set_lines(
    tags: dict[str, Any],
    set_type: str,
    count: int,
    kept: np.ndarray | None,
    store: Store,
    prefix: str,
    indent: int,
) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    trimmed: list[str] = []
    for name, members in tags.items():
        ids = member_indices(members, count)
        if ids.size != member_values(members).size:
            trimmed.append(str(name))
        if kept is not None:
            remap = np.cumsum(kept) - 1
            ids = remap[ids[kept[ids]]]
        lines.append(f'{pad}<Set Name={quoteattr(str(name))} SetType="{set_type}">')
        lines.append(
            _data_item(
                ids.astype(np.int64),
                store,
                f"{prefix}/{_h5_safe(str(name))}",
                indent + 2,
            )
        )
        lines.append(f"{pad}</Set>")
    if trimmed:
        kind = "vertex" if set_type == "Node" else "element"
        warnings.warn(
            f"{EXTENSION} write: {kind} tag group(s) {sorted(trimmed)} name"
            f" members that are not {kind} indices of this mesh; those"
            " members are not written.",
            UserWarning,
            stacklevel=6,
        )
    return lines


__all__ = [
    "EXTENSION",
    "EXTENSIONS",
    "HAVE_H5PY",
    "read",
    "read_time_series",
    "write",
    "write_time_series",
]
