"""napari reader and writers, registered through ``polyxios/napari.yaml``.

napari finds them through the ``napari.manifest`` entry point, so installing
polyxios next to napari is all a user does. Nothing here imports napari: a
reader hands back plain ``(data, kwargs, layer_type)`` tuples and a writer is
given numpy arrays, so polyxios gains no dependency for it.

napari indexes data the way it indexes an image, slowest axis first, so a
vertex ``(x, y, z)`` is shown as ``(z, y, x)`` by default. That is what lets a
mesh sit on the volume it was segmented from. Each layer records the order it
was opened in under ``metadata["polyxios"]``, and the writers undo exactly
that.

A layer shows less than a file holds: a Surface draws one scalar field and a
Points layer has no cells. What the layer cannot show - the other vertex
attributes, the vertex tags, the lines of a line set, the mesh-level
``global_attrs`` - rides along in the same metadata entry, and the writer puts
it back as long as the layer's points are still the file's points, moved or
not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Literal

import numpy as np

import polyxios
from polyxios._element_types import (
    ELEMENT_FACES,
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    QUADRATIC_SURFACE_CORNERS,
    SURFACE_ELEMENT_TYPES,
    TOPOLOGICAL_DIMENSION,
)
from polyxios._io import GZIP_SUFFIXES, format_suffix, open_read, source_suffix
from polyxios._registry import Codec, sniff_head
from polyxios._types import PolyData, make_polydata
from polyxios._warn import warn_caller
from polyxios.codecs import _exodus, _gltf, _tecplot, _vti, _vtr, _vts
from polyxios.codecs._vtk_xml import (
    extent_points,
    extent_spans,
    structured_cells_fit,
)
from polyxios.exceptions import CodecError, UnsupportedFormatError
from polyxios.helper import META_SUFFIXES, read_multiblock, read_multiblock_vtp

__all__ = [
    "META_KEY",
    "get_reader",
    "layer_data",
    "write_points",
    "write_surface",
]

AxisOrder = Literal["zyx", "xyz"]
LayerData = tuple[Any, dict[str, Any], str]

#: The key under a layer's ``metadata`` that the reader fills and the writers read.
META_KEY: Final[str] = "polyxios"

_AXIS_ORDERS: Final[tuple[str, ...]] = ("zyx", "xyz")

_SURFACE_CODES: Final[np.ndarray] = np.array(sorted(SURFACE_ELEMENT_TYPES))
_PIXEL_CODE: Final[int] = ELEMENT_TYPES["pixel"]
_STRIP_CODE: Final[int] = ELEMENT_TYPES["triangle_strip"]
_SOLID_FACES: Final[dict[int, tuple[tuple[int, ...], ...]]] = {
    ELEMENT_TYPES[name]: faces for name, faces in ELEMENT_FACES.items()
}
_SOLID_CODES: Final[np.ndarray] = np.array(sorted(_SOLID_FACES))
_HEX_FACES: Final[tuple[tuple[int, ...], ...]] = ELEMENT_FACES["hexahedron"]
# Where each corner of a VTK hexahedron sits in its cell, along x, y and z.
_HEX_CORNERS: Final[np.ndarray] = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)
# The extent a structured VTK grid is read with, under each reader's own key.
_GRID_EXTENT_KEYS: Final[tuple[str, ...]] = ("vti_extent", "vtr_extents", "vts_extent")
# The points along x, y and z a legacy .vtk grid is read with.
_LEGACY_GRID_KEY: Final[str] = "vtk_dimensions"
_HEX_CODE: Final[int] = ELEMENT_TYPES["hexahedron"]

_COLOR_ATTR: Final[str] = "colors"
# PLY spells a colour as three or four scalar properties and a splat as three
# bytes; both arrive as separate vertex attributes rather than one array.
_COLOR_CHANNELS: Final[tuple[tuple[tuple[str, str, str], str | None], ...]] = (
    (("red", "green", "blue"), "alpha"),
    (("color_r", "color_g", "color_b"), None),
)
_ALPHA_NAMES: Final[dict[tuple[str, ...], str]] = {
    rgb: alpha for rgb, alpha in _COLOR_CHANNELS if alpha is not None
}
# The full range of an integer colour channel, by its width. A wider integer
# says nothing about the range its values were meant in.
_COLOR_RANGE: Final[dict[str, float]] = {"u1": 255.0, "u2": 65535.0}
# Integer columns that number things rather than measure them: colouring a
# surface by a running index shows nothing.
_ID_SUFFIXES: Final[tuple[str, ...]] = ("_id", "_ids", "_index")
# Vertex attributes that a point's position decides: moving the point leaves
# them describing where it was. A texture coordinate is a label the point
# carries, and stays right wherever it goes.
_NORMAL_NAMES: Final[frozenset[str]] = frozenset({"nx", "ny", "nz"})
_NORMAL_PREFIX: Final[str] = "normal"
# Scalar columns that are one component of a direction or a texture lookup,
# as PLY spells them - a normal's, or these: colouring a surface by one shows
# nothing about it. A texture coordinate is only ever a pair; a lone ``t`` is
# a temperature or a time, and ``u v w`` a velocity.
_TEXTURE_PAIRS: Final[tuple[tuple[frozenset[str], str | None], ...]] = (
    (frozenset({"u", "v"}), "w"),
    (frozenset({"s", "t"}), None),
)
_GEOMETRY_PREFIXES: Final[tuple[str, ...]] = (
    _NORMAL_PREFIX,
    "texcoord",
    "tex_coord",
    "texture_",
)

# Files a format writes beside the one it is named after, as the codec names
# them from that path: a .gltf keeps its buffers in a .bin.
_SIDECARS: Final[dict[str, tuple[Callable[..., Path], ...]]] = {
    ".gltf": (_gltf.buffer_path,)
}

# Formats polyxios writes and never reads; a reader offered for them would
# fail on every file.
_NOT_READ: Final[frozenset[str]] = frozenset({".svg"})
# Index files, which name their pieces relative to themselves and are read
# by helper.read_multiblock rather than by a codec, as plain XML only.
_PLAIN_ONLY: Final[frozenset[str]] = META_SUFFIXES
_VTP: Final[str] = ".vtp"
# How much of a file the content test of a codec owning its extension sees.
# The registry's own sniff window is too short for an HDF5 signature behind a
# user block, which a netCDF-4 Exodus file may carry; this is the window the
# Exodus reader itself looks for one in. A dispatcher's candidates see the
# registry's window, since it is all polyxios.read shows them, so .e must
# stay one codec's for these files to open.
_HEAD_BYTES: Final[int] = 1 << 16

# Extensions other software writes too, so a file under one is often not a
# mesh polyxios reads. napari hands a file to the first reader whose pattern
# matches it, and one that then raises stops it trying the next; these are
# checked by content, with the sniffers of the codecs the registry resolves
# them to, before the reader takes them.
_SHARED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".dat", ".e", ".plt", ".pts", ".vol", ".xml", ".xyz"}
)
# The content test of a codec whose sniffer takes files its reader refuses
# (a binary .plt), or that has none, keyed by the codec's reader.
_READ_TESTS: Final[dict[Callable[..., PolyData], Callable[[bytes], bool]]] = {
    _exodus.read: _exodus.is_netcdf,
    _tecplot.read: _tecplot.is_ascii,
}
_CODEC_PACKAGE: Final[str] = "polyxios.codecs."

# The structured grid a reader expanded the vertices and cells from. It no
# longer describes a Surface, whose cells are triangles, nor a Points layer
# whose cells could not be kept.
_GRID_GLOBALS: Final[frozenset[str]] = (
    _vti.RESERVED_GLOBALS
    | _vtr.RESERVED_GLOBALS
    | _vts.RESERVED_GLOBALS
    | {"vtk_dimensions", "vtk_origin", "vtk_spacing", "vtkhdf_direction"}
)
# A VTKHDF file names its dataset type; only an image is a structured grid,
# and the type of a PolyData or an UnstructuredGrid still fits triangles.
_VTKHDF_TYPE: Final[str] = "vtkhdf_type"
_VTKHDF_GRID: Final[str] = "ImageData"
# Global attributes that lay the vertices out in rows - an organised PCD
# grid, the scans of a PTX file - and so describe the file's rows, not the
# layer's, once those rows have moved.
_LAYOUT_GLOBALS: Final[frozenset[str]] = (
    frozenset({"pcd_width", "pcd_height", "ptx_dimensions", "ptx_transforms"})
    | _GRID_GLOBALS
)

# napari sizes points in data units, and its default of 10 is a speck on a
# survey in metres and a blob on a scan in millimetres. A fraction of the
# cloud's extent reads the same at either scale.
_POINT_SIZE_FRACTION: Final[float] = 0.002

# The splitmix64 finaliser's constants: every bit of a row's words reaches
# every bit of its hash, which a multiply-and-xor over whole words does not.
_MIX_MULTIPLIERS: Final[tuple[np.uint64, np.uint64]] = (
    np.uint64(0xBF58476D1CE4E5B9),
    np.uint64(0x94D049BB133111EB),
)
_MIX_SHIFTS: Final[tuple[np.uint64, np.uint64, np.uint64]] = (
    np.uint64(30),
    np.uint64(27),
    np.uint64(31),
)
_WORD_BITS: Final[np.uint64] = np.uint64(32)


def _grid_keys(global_attrs: dict[str, Any]) -> list[str]:
    """Return the keys of *global_attrs* that describe a structured grid."""
    return sorted(
        k
        for k, v in global_attrs.items()
        if k in _GRID_GLOBALS
        or (k == _VTKHDF_TYPE and isinstance(v, str) and v == _VTKHDF_GRID)
    )


def _float_coords(coords: Any) -> np.ndarray:
    coords = np.asarray(coords)
    return coords if coords.dtype.kind == "f" else coords.astype(np.float64)


def _flips(axis_order: AxisOrder) -> bool:
    """Whether *axis_order* mirrors the vertices, and so the triangles' winding.

    napari reverses a Surface's columns and its faces again to draw it, so a
    face must be wound for the order its layer holds the columns in.
    """
    return axis_order == "zyx"


def _to_napari(vertices: np.ndarray, *, axis_order: AxisOrder) -> np.ndarray:
    return vertices[:, ::-1] if _flips(axis_order) else vertices


def _from_napari(coords: Any, *, axis_order: AxisOrder) -> np.ndarray:
    coords = _float_coords(coords)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(
            f"polyxios writes 2-D or 3-D coordinates; this layer's data has "
            f"shape {coords.shape}."
        )
    if _flips(axis_order):
        coords = coords[:, ::-1]
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords), coords.dtype)])
    return coords


def _cell_blocks(
    poly: PolyData, *, codes: np.ndarray
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(type code, (m, width) vertex rows)`` for cells of *codes*.

    Cells are grouped by type and width, so every block is a plain 2-D array
    and the work on it is a numpy expression rather than a loop over cells.
    """
    types = np.asarray(poly.element_types)
    selected = np.flatnonzero(np.isin(types, codes))
    if not len(selected):
        return
    offsets = np.asarray(poly.offsets, dtype=np.int64)
    widths = np.diff(offsets)[selected]
    connectivity = np.asarray(poly.connectivity)
    key = types[selected].astype(np.int64) * (int(widths.max()) + 1) + widths
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for group in np.split(order, bounds):
        cells = selected[group]
        width = int(widths[group[0]])
        if width == 0:
            continue
        rows = connectivity[offsets[cells, None] + np.arange(width)]
        yield int(types[cells[0]]), rows


def _fan(rings: np.ndarray) -> np.ndarray:
    """Split (m, k) rings into triangles fanned from each ring's first corner.

    Right for every convex ring. A concave polygon fanned from a corner its
    notch hides draws triangles outside its outline.
    """
    k = rings.shape[1]
    if k < 3:
        return np.empty((0, 3), dtype=rings.dtype)
    picks = np.array([[0, j, j + 1] for j in range(1, k - 1)])
    return rings[:, picks].reshape(-1, 3)


def _surface_rings(poly: PolyData) -> Iterator[np.ndarray]:
    """Yield the surface cells as (m, k) rings, quadratic ones by their corners."""
    for code, rows in _cell_blocks(poly, codes=_SURFACE_CODES):
        corners = QUADRATIC_SURFACE_CORNERS.get(code)
        if corners is not None:
            yield rows[:, :corners]
        elif code == _PIXEL_CODE:
            # A pixel numbers its corners row by row, not around the ring.
            yield rows[:, [0, 1, 3, 2]]
        elif code == _STRIP_CODE:
            if rows.shape[1] >= 3:
                # Every other triangle of a strip is wound backwards.
                picks = np.array(
                    [
                        [j, j + 1, j + 2] if j % 2 == 0 else [j + 1, j, j + 2]
                        for j in range(rows.shape[1] - 2)
                    ]
                )
                yield rows[:, picks].reshape(-1, 3)
        else:
            yield rows


def _ring_keys(rings: np.ndarray) -> np.ndarray:
    """Pack each ring's sorted vertices into as few int64 columns as hold them.

    The packing is exact: a column holds as many vertex indices as its base,
    the largest index plus one, can raise to without passing 2**63.
    """
    ordered = np.sort(rings, axis=1).astype(np.int64, copy=False)
    width = ordered.shape[1]
    base = int(ordered.max()) + 1 if ordered.size else 1
    per = 1
    if ordered.size and int(ordered.min()) >= 0:
        while per < width and max(base, 2) ** (per + 1) < 2**63:
            per += 1
    columns = []
    for start in range(0, width, per):
        key = ordered[:, start].copy()
        for column in range(start + 1, min(start + per, width)):
            key = key * base + ordered[:, column]
        columns.append(key)
    return np.column_stack(columns)


def _ring_groups(rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group (m, k) rings that hold the same vertices, in any order.

    Returns
    -------
    order : ndarray
        The permutation that sorts the rings by their vertices.
    group : ndarray
        The group of each ring in that sorted order, counting from 0.
    """
    keys = _ring_keys(rings)
    # One integer per ring sorts several times faster than the rows viewed as
    # bytes, which is what np.unique sorts, and than np.lexsort. Rings too
    # wide to pack into one are sorted by a hash of their columns, which
    # keeps equal rings together unless two different ones collide - and
    # then the rows themselves are sorted.
    if keys.shape[1] == 1:
        order = np.argsort(keys[:, 0])
        ordered = keys[order]
        starts = np.ones(len(ordered), dtype=bool)
        starts[1:] = ordered[1:, 0] != ordered[:-1, 0]
        return order, np.cumsum(starts) - 1
    hashes = _row_hash(keys.view(np.uint64))
    order = np.argsort(hashes)
    ordered = keys[order]
    starts = np.ones(len(ordered), dtype=bool)
    starts[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    hashes = hashes[order]
    if (starts[1:] & (hashes[1:] == hashes[:-1])).any():
        rows = np.ascontiguousarray(keys)
        order = np.argsort(rows.view(np.dtype((np.void, rows.shape[1] * 8))).ravel())
        ordered = keys[order]
        starts[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    return order, np.cumsum(starts) - 1


def _unshared(faces: np.ndarray) -> np.ndarray:
    """Return the rows of *faces* whose vertices no other row holds."""
    if not len(faces):
        return faces
    order, group = _ring_groups(faces)
    single = np.empty(len(faces), dtype=bool)
    single[order] = np.bincount(group)[group] == 1
    return faces[single]


def _among(rings: np.ndarray, *, faces: np.ndarray) -> np.ndarray:
    """Say which of *rings* holds the vertices of one of *faces*."""
    if not len(rings) or not len(faces):
        return np.zeros(len(rings), dtype=bool)
    order, group = _ring_groups(np.concatenate([faces, rings]))
    hit = np.zeros(int(group[-1]) + 1, dtype=bool)
    hit[group[order < len(faces)]] = True
    found = np.empty(len(order), dtype=bool)
    found[order] = hit[group]
    return found[len(faces) :]


def _legacy_grid_fits(poly: PolyData, *, spans: Sequence[int]) -> bool:
    """Say whether *poly* holds the hexahedra a legacy ``.vtk`` grid expands to.

    The legacy reader numbers points ``x`` fastest, as the XML grids do, but
    runs its cells ``z`` fastest; checked a corner at a time, so nothing wider
    than one column of the connectivity is held at once.
    """
    cx, cy, cz = spans
    n_cells = cx * cy * cz
    types = np.asarray(poly.element_types)
    connectivity = np.asarray(poly.connectivity)
    if (
        len(types) != n_cells
        or len(connectivity) != 8 * n_cells
        or not bool(np.all(types == _HEX_CODE))
    ):
        return False
    rows = connectivity.reshape(cx, cy, cz, 8)
    i, j, k = np.ogrid[:cx, :cy, :cz]
    origins = i + (cx + 1) * (j + (cy + 1) * k)
    return all(
        bool(np.array_equal(rows[..., corner], origins + _corner_step(c, spans=spans)))
        for corner, c in enumerate(_HEX_CORNERS)
    )


def _corner_step(corner: np.ndarray, *, spans: Sequence[int]) -> int:
    """Return the point-index offset of a hexahedron *corner* in its grid."""
    return int(corner[0] + (spans[0] + 1) * (corner[1] + (spans[1] + 1) * corner[2]))


def _grid_spans(poly: PolyData) -> list[int] | None:
    """Return the cells along x, y and z when *poly* is a structured hex grid.

    That is a ``.vti``, ``.vtr`` or ``.vts`` grid, or a legacy ``.vtk``
    ``STRUCTURED_POINTS``, ``STRUCTURED_GRID`` or ``RECTILINEAR_GRID``, whose
    cells are still the ones its extent or dimensions expand to; None for
    any other mesh.
    """
    n_points = len(poly.vertices)
    for key in _GRID_EXTENT_KEYS:
        stored = poly.global_attrs.get(key)
        spans = extent_spans(stored) if stored is not None else None
        if (
            spans is not None
            and min(spans) > 0
            and extent_points(spans) == n_points
            and structured_cells_fit(
                np.asarray(poly.connectivity), np.asarray(poly.element_types), spans
            )
        ):
            return spans
    dims = poly.global_attrs.get(_LEGACY_GRID_KEY)
    try:
        spans = [int(d) - 1 for d in dims] if dims is not None else None
    except (TypeError, ValueError):
        spans = None
    if (
        spans is not None
        and len(spans) == 3
        and min(spans) > 0
        and extent_points(spans) == n_points
        and _legacy_grid_fits(poly, spans=spans)
    ):
        return spans
    return None


def _grid_skin(*, spans: Sequence[int]) -> np.ndarray:
    """Return the outer faces of a structured hex grid, from its spans alone.

    The cells on each side of the box give that side's faces, so the work is
    the size of the surface rather than six faces for every cell: a 256^3
    image would otherwise sort a hundred million faces to find its skin. The
    corners are point indices, ``x`` fastest, which is how every structured
    grid reader numbers its points, whatever order it runs its cells in.
    """
    nx, ny, nz = spans
    parts = []
    for local in _HEX_FACES:
        corners = _HEX_CORNERS[list(local)]
        axis = int(np.flatnonzero((corners == corners[0]).all(axis=0))[0])
        along = [np.arange(n) for n in spans]
        along[axis] = np.array([0 if corners[0, axis] == 0 else spans[axis] - 1])
        k, j, i = np.meshgrid(along[2], along[1], along[0], indexing="ij")
        origins = (i + (nx + 1) * (j + (ny + 1) * k)).ravel()
        steps = np.array([_corner_step(c, spans=spans) for c in corners])
        parts.append(origins[:, None] + steps)
    return np.concatenate(parts)


def _skin(poly: PolyData) -> dict[int, np.ndarray]:
    """Return the faces of the solid cells that no other solid shares.

    Keyed by ring width; each face keeps the outward winding
    :data:`~polyxios._element_types.ELEMENT_FACES` gives it.
    """
    spans = _grid_spans(poly)
    if spans is not None:
        return {4: _grid_skin(spans=spans)}
    by_width: dict[int, list[np.ndarray]] = {}
    for code, rows in _cell_blocks(poly, codes=_SOLID_CODES):
        for local in _SOLID_FACES[code]:
            by_width.setdefault(len(local), []).append(rows[:, list(local)])
    return {
        width: _unshared(np.concatenate(parts)) for width, parts in by_width.items()
    }


def _triangles(poly: PolyData) -> np.ndarray:
    """Return every surface of *poly* as an (n, 3) triangle index array.

    Surface cells are triangulated as they are, and solid cells contribute the
    faces on the outside of the volume, so a tetrahedral FEM mesh opens as the
    skin of the part. A surface cell lying on that skin - a boundary patch a
    mesher writes next to the volume it bounds - is drawn once, not twice.
    """
    skin = _skin(poly)
    parts = []
    for rings in _surface_rings(poly):
        on_skin = skin.get(rings.shape[1])
        if on_skin is not None:
            rings = rings[~_among(rings, faces=on_skin)]
        parts.append(_fan(rings))
    parts.extend(_fan(faces) for faces in skin.values())
    if not parts:
        return np.empty((0, 3), dtype=np.int64)
    return np.concatenate(parts).astype(np.int64, copy=False)


def _scalar_columns(attrs: dict[str, Any], *, n: int) -> dict[str, np.ndarray]:
    columns = {}
    for name, arr in attrs.items():
        arr = np.asarray(arr)
        if (
            arr.ndim == 1
            and len(arr) == n
            and (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_)
        ):
            columns[name] = arr
    return columns


def _geometry_names(names: Sequence[str]) -> set[str]:
    """Return those of *names* that are a component of a normal or texture lookup.

    ``nx``, ``ny``, ``nz`` and names starting ``normal`` or ``texcoord`` are
    always one. ``u`` and ``v`` are only when both are there and no ``w``,
    and ``s`` and ``t`` only when both are there.
    """
    lower = {name: name.lower() for name in names}
    present = set(lower.values())
    texture: set[str] = set()
    for pair, third in _TEXTURE_PAIRS:
        if pair <= present and (third is None or third not in present):
            texture |= pair
    return {
        name
        for name, low in lower.items()
        if low in _NORMAL_NAMES or low in texture or low.startswith(_GEOMETRY_PREFIXES)
    }


def _is_normal(name: str) -> bool:
    name = name.lower()
    return name in _NORMAL_NAMES or name.startswith(_NORMAL_PREFIX)


def _values_column(scalars: dict[str, np.ndarray]) -> str | None:
    """Pick the attribute a Surface is coloured by: a measured field first.

    A normal or texture-coordinate component is never picked, and neither is
    an integer index.
    """
    geometry = _geometry_names(list(scalars))
    floats = [
        k for k, v in scalars.items() if v.dtype.kind == "f" and k not in geometry
    ]
    if floats:
        return floats[0]
    counts = [
        k
        for k, v in scalars.items()
        if v.dtype.kind in "iu"
        and not k.lower().endswith(_ID_SUFFIXES)
        and k not in geometry
    ]
    return counts[0] if counts else None


def _channel_range(raw: np.ndarray) -> tuple[float, bool]:
    """Return the full range of the colour channels *raw*, and if they count.

    An 8- or 16-bit integer channel spans its type's range. Any other type
    says nothing about its range, so its peak decides: floats up to 1 are
    fractions, values up to 255 are 8-bit, larger ones 16-bit.

    A reader may hand 8-bit channels over as floats (ASCII PLY does); whole
    numbers in a 0..255 range are then counts to round back to. Fractions are
    not, and neither are 0..1 floats that happen to be 0 or 1, which napari
    may set to anything in between.
    """
    finite = raw[np.isfinite(raw)]
    scale = _COLOR_RANGE.get(raw.dtype.str[1:])
    if scale is None:
        peak = float(finite.max()) if finite.size else 0.0
        scale = 1.0 if raw.dtype.kind == "f" and peak <= 1.0 else 255.0
        if peak > 255.0:
            scale = 65535.0
    return scale, scale != 1.0 and bool(np.all(finite == np.rint(finite)))


def _colors(poly: PolyData, *, n: int) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Return the vertex colours as RGBA in 0..1, and where they came from.

    Only an attribute that names itself a colour counts. A normal is the same
    shape as an RGB triple, and guessing from shape would show one as the
    other and write it back as a colour attribute the file never had.

    The RGB channels share one range, and a separate alpha channel has its
    own: a PLY may store ``uchar`` colours beside a ``float`` alpha in 0..1.
    Each channel keeps its own dtype for the writer. A NaN in a channel,
    which napari cannot draw, shows as the channel at full, and is written
    back so.
    """
    named = poly.vertex_attrs.get(_COLOR_ATTR)
    if named is not None:
        named = np.asarray(named)
    if (
        named is not None
        and named.ndim == 2
        and named.shape[1] in (3, 4)
        and len(named) == n
    ):
        names = [_COLOR_ATTR]
        groups = [named]
        dtypes = [named.dtype.str] * named.shape[1]
        alpha_held = False
    else:

        def channel(name: str | None) -> Any:
            col = poly.vertex_attrs.get(name) if name else None
            col = np.asarray(col) if col is not None else None
            return col if col is not None and col.ndim == 1 and len(col) == n else None

        for rgb, alpha in _COLOR_CHANNELS:
            cols = [channel(c) for c in rgb]
            if any(c is None for c in cols):
                continue
            names = list(rgb)
            groups = [np.column_stack(cols)]
            alpha_col = channel(alpha)
            # A flag or a label named alpha is not an opacity.
            if alpha_col is not None and alpha_col.dtype.kind in "iuf":
                names.append(alpha)
                cols.append(alpha_col)
                groups.append(alpha_col[:, None])
            alpha_held = alpha not in names and alpha in poly.vertex_attrs
            dtypes = [c.dtype.str for c in cols]
            break
        else:
            return None
    if groups[0].dtype.kind not in "iuf" or not np.isfinite(groups[0]).any():
        return None
    rgba = np.ones((n, 4), dtype=np.float32)
    scales: list[float] = []
    integral: list[bool] = []
    for group in groups:
        scale, counts = _channel_range(group)
        at = len(scales)
        rgba[:, at : at + group.shape[1]] = np.clip(
            np.nan_to_num(group / scale, nan=1.0), 0.0, 1.0
        )
        scales.extend([scale] * group.shape[1])
        integral.extend([counts] * group.shape[1])
    source = {
        "names": names,
        "width": len(scales),
        "scales": scales,
        "dtypes": dtypes,
        "integral": integral,
        "alpha_held": alpha_held,
    }
    return rgba, source


def _restore_colors(
    rgba: Any, *, source: dict[str, Any], n: int, taken: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Turn the layer's RGBA back into the colour attributes they were read from.

    An alpha the file's colours lack is added under the alpha name of their
    channels, unless the file held that name for something else, or *taken*
    - the attributes written beside them - already holds it.
    """
    rgba = np.asarray(rgba, dtype=np.float64)
    if rgba.ndim != 2 or len(rgba) != n or rgba.shape[1] < 3:
        warn_caller(
            f"the layer's colours have shape {rgba.shape} for {n} vertices; "
            f"they were not written."
        )
        return {}
    if rgba.shape[1] == 3:
        # napari draws RGB colours opaque.
        rgba = np.column_stack([rgba, np.ones(n)])
    names = list(source["names"])
    width = min(int(source["width"]), 4)
    scales = [float(s) for s in source["scales"]][:width]
    dtypes = [np.dtype(d) for d in source["dtypes"]][:width]
    integral = [bool(c) for c in source["integral"]][:width]
    if width == 3 and not np.all(rgba[:, 3] == 1.0):
        alpha = _ALPHA_NAMES.get(tuple(names))
        if alpha is not None and (alpha in taken or source.get("alpha_held")):
            warn_caller(
                f"the layer's colours are not opaque, and the file's own "
                f"{alpha} is not a colour channel; the alpha was not written."
            )
        elif names == [_COLOR_ATTR] or alpha is not None:
            if alpha is not None:
                names.append(alpha)
            width = 4
            # An alpha the file never had is written the way its colours are.
            scales.append(scales[-1])
            dtypes.append(dtypes[-1])
            integral.append(integral[-1])
        else:
            warn_caller(
                f"the layer's colours are not opaque, and "
                f"{' '.join(names)} has no alpha channel; the alpha was not "
                f"written."
            )
    channels = []
    for i in range(width):
        values = rgba[:, i] * scales[i]
        dtype = dtypes[i]
        if dtype.kind in "iu" or integral[i]:
            # napari keeps colours as float32 fractions; truncating one back
            # would turn 255 into 254.
            top = (
                min(scales[i], np.iinfo(dtype).max) if dtype.kind in "iu" else scales[i]
            )
            values = np.clip(np.rint(values), 0.0, top)
        channels.append(values.astype(dtype))
    if names == [_COLOR_ATTR]:
        return {_COLOR_ATTR: np.column_stack(channels)}
    return dict(zip(names, channels, strict=True))


def _type_names(poly: PolyData) -> dict[int, str]:
    return {
        int(c): ELEMENT_TYPES_INV[int(c)]
        for c in np.unique(np.asarray(poly.element_types))
    }


def _undrawn(names: dict[int, str]) -> list[str]:
    """Name the cell types a Surface draws nothing of: lines, vertices."""
    return [
        n
        for c, n in names.items()
        if c not in SURFACE_ELEMENT_TYPES and c not in _SOLID_FACES
    ]


def _undrawn_shapes(names: dict[int, str]) -> list[str]:
    """Name the 2-D and 3-D cell types of a mesh that drew no triangle.

    Those are the ones without a triangulation here - a polyhedron, a
    Lagrange or a quadratic-linear cell - and a point cloud is all that is
    left of them on screen. Lines and vertices are what a Points layer shows.
    """
    return [n for c, n in names.items() if TOPOLOGICAL_DIMENSION.get(c, 0) >= 2]


def _not_kept(poly: PolyData) -> list[str]:
    """Say what of *poly* a Surface layer, being triangles alone, cannot keep."""
    names = _type_names(poly)
    surface = [n for c, n in names.items() if c in SURFACE_ELEMENT_TYPES]
    solid = [n for c, n in names.items() if c in _SOLID_FACES]
    other = _undrawn(names)
    notes = []
    if any(n != "triangle" for n in surface):
        reshaped = ", ".join(n for n in surface if n != "triangle")
        notes.append(f"{reshaped} cells are written as triangles")
    if solid:
        notes.append(f"{', '.join(solid)} cells are written as their skin")
    if other:
        notes.append(f"{', '.join(other)} cells are not written")
    element_data = sorted({*poly.element_attrs, *poly.element_tags})
    if element_data:
        notes.append(f"element data {', '.join(element_data)} is not written")
    grid = _grid_keys(poly.global_attrs)
    if grid:
        notes.append(f"the structured grid {', '.join(grid)} is not written")
    return notes


def layer_data(
    poly: PolyData,
    *,
    name: str,
    axis_order: AxisOrder = "zyx",
) -> list[LayerData]:
    """Turn a PolyData into napari layer data.

    A mesh with surface or volume cells becomes one Surface layer; one with
    neither - a point cloud, or a line set - becomes one Points layer.

    Parameters
    ----------
    poly
        Mesh to show.
    name
        Layer name.
    axis_order
        ``"zyx"`` reverses the vertex columns into napari's image order, so
        the mesh overlays a volume of the same space, and reverses each
        triangle with them, since napari reverses both again to draw it and
        a mirrored triangle faces the other way. ``"xyz"`` passes both
        through as stored.

    Returns
    -------
    list of tuple
        ``[(data, kwargs, layer_type)]``, the shape a napari reader returns.
        Vertices keep the file's floating dtype.

        A Surface's data is ``(vertices, faces)``, or ``(vertices, faces,
        values)`` when the mesh carries a scalar per vertex: the first
        floating one that is not a normal or texture coordinate component,
        else the first integer one that is not an index. ``u`` and ``v`` are
        taken for a texture coordinate only side by side with no ``w``, and
        ``s`` and ``t`` only side by side. Polygons are fanned
        from their first corner, which draws a concave one wrong. A mesh with
        both surface and solid cells shows the solid's skin once, with any
        surface cell lying on it dropped. A structured ``.vti``, ``.vtr`` or
        ``.vts`` grid, or a legacy ``.vtk`` one, finds its skin from its
        extent, without sorting the faces of every cell.

        A Points layer keeps every scalar vertex attribute as a feature.

        Either layer shows vertex colours - a ``colors`` attribute, or PLY's
        ``red green blue`` and optional ``alpha`` - as ``vertex_colors`` or
        ``face_color``. What the layer cannot show is kept in
        ``kwargs["metadata"]["polyxios"]`` for the writers: every other
        vertex attribute, the vertex tags, the cells of a Points layer, and
        ``global_attrs``, with the vertex rows they belong to. A Surface does
        not keep the structured grid it was read from (``vti_extent``,
        ``vtk_dimensions``, a ``vtkhdf_type`` of ``ImageData``, ...): its
        cells are triangles, which no grid describes.

    Warns
    -----
    UserWarning
        When a mesh opening as a Surface also has cells it draws nothing of,
        such as lines or vertices; or when a mesh opens as Points although it
        has surface or volume cells, because none of them is a type drawn
        here - a polyhedron, a Lagrange cell, a quadratic-linear quad.

    Raises
    ------
    ValueError
        If ``axis_order`` is neither ``"zyx"`` nor ``"xyz"``.
    """
    if axis_order not in _AXIS_ORDERS:
        raise ValueError(
            f"axis_order must be one of {_AXIS_ORDERS}, got {axis_order!r}."
        )
    vertices = _to_napari(_float_coords(poly.vertices), axis_order=axis_order)
    n_verts = len(vertices)
    info: dict[str, Any] = {
        "axis_order": axis_order,
        "n_vertices": n_verts,
        "global_attrs": dict(poly.global_attrs),
        "vertex_tags": dict(poly.vertex_tags),
    }
    shown: set[str] = set()
    colors = _colors(poly, n=n_verts)
    colors_shown = set(colors[1]["names"]) if colors is not None else set()
    if colors is not None:
        info["colors"] = colors[1]
    shown.update(colors_shown)
    scalars = {
        k: v
        for k, v in _scalar_columns(poly.vertex_attrs, n=n_verts).items()
        if k not in shown
    }

    faces = _triangles(poly)
    if _flips(axis_order):
        faces = faces[:, ::-1]
    if len(faces):
        data: tuple[np.ndarray, ...] = (vertices, faces)
        values_name = _values_column(scalars)
        if values_name is not None:
            data = (vertices, faces, scalars[values_name])
            info["values"] = values_name
            shown.add(values_name)
        kwargs: dict[str, Any] = {"name": name, "metadata": {META_KEY: info}}
        if colors is not None:
            kwargs["vertex_colors"] = colors[0]
        info["not_kept"] = _not_kept(poly)
        grid = _grid_keys(info["global_attrs"])
        info["global_attrs"] = {
            k: v for k, v in info["global_attrs"].items() if k not in grid
        }
        undrawn = _undrawn(_type_names(poly))
        if undrawn:
            warn_caller(
                f"{', '.join(undrawn)} cells are not drawn: the Surface layer "
                f"{name!r} shows triangles only."
            )
        layer: LayerData = (data, kwargs, "surface")
    else:
        shown.update(scalars)
        kwargs = {"name": name, "features": scalars, "metadata": {META_KEY: info}}
        if colors is not None:
            kwargs["face_color"] = colors[0]
        finite = vertices[np.isfinite(vertices).all(axis=1)]
        if len(finite):
            extent = float(np.ptp(finite, axis=0).max())
            if extent > 0:
                kwargs["size"] = extent * _POINT_SIZE_FRACTION
        undrawn = _undrawn_shapes(_type_names(poly))
        if undrawn:
            warn_caller(
                f"{', '.join(undrawn)} cells are not drawn: the file opens as "
                f"the Points layer {name!r}, and they are kept for saving."
            )
        if len(poly.element_types):
            info["cells"] = {
                "connectivity": poly.connectivity,
                "offsets": poly.offsets,
                "element_types": poly.element_types,
                "element_attrs": dict(poly.element_attrs),
                "element_tags": dict(poly.element_tags),
            }
        layer = (vertices, kwargs, "points")
    info["vertex_attrs"] = {
        k: v for k, v in poly.vertex_attrs.items() if k not in shown
    }
    info["normals"] = [
        k for k in poly.vertex_attrs if _is_normal(k) and k not in colors_shown
    ]
    if (
        info["vertex_attrs"]
        or info["normals"]
        or info["vertex_tags"]
        or "cells" in info
        or _LAYOUT_GLOBALS.intersection(info["global_attrs"])
    ):
        info["vertex_keys"] = _row_hash(_row_bits(vertices))
        info["vertex_width"] = vertices.shape[1]
    return [layer]


def _layer_name(path: str) -> str:
    name = Path(path).name
    suffix = format_suffix(path)
    return name[: name.lower().rfind(suffix)] if suffix else name


def _read(path: str | Sequence[str]) -> list[LayerData]:
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    layers: list[LayerData] = []
    for p in paths:
        p = os.fspath(p)
        read = read_multiblock if format_suffix(p) in META_SUFFIXES else _read_one
        layers.extend(layer_data(read(p), name=_layer_name(p)))
    return layers


def _read_one(path: str) -> PolyData:
    """Read *path*, or merge its pieces when it is a multi-block ``.vtp``.

    ParaView saves a multi-block set under ``.vtp`` too, which the VTP codec
    refuses for not being PolyData.
    """
    try:
        return polyxios.read(path)
    except UnsupportedFormatError as exc:
        if source_suffix(path) != _VTP:
            raise
        try:
            return read_multiblock_vtp(path)
        except (ValueError, SyntaxError):
            # No multi-block set in it either: the codec's reason stands.
            raise exc from None


def _read_test(codec: Codec) -> Callable[[bytes], bool] | None:
    """Return the test a file must pass for *codec*.

    That is the built-in codec's own content test when its sniffer takes files
    its reader refuses, or it has none; else ``codec.sniff``, which is None
    for a codec with nothing to test a file by.
    """
    return _READ_TESTS.get(codec.read, codec.sniff)


def _passes(test: Callable[[bytes], bool], head: bytes) -> bool:
    try:
        return bool(test(head))
    except Exception:
        # A sniffer that raises has not recognised the file, and raising
        # from get_reader would stop napari trying the next plugin.
        return False


def _head(path: str, *, size: int | None) -> bytes | None:
    """Return the opening of *path* that its codec's content test sees.

    With *size*, a full window of text loses its last, possibly cut, line, as
    :func:`~polyxios._registry.sniff_head` drops it from the dispatchers'
    window. A binary window is left whole: the line it would cut is no line,
    and the bytes after its last newline may hold an HDF5 signature.
    """
    try:
        if size is None:
            return sniff_head(path)
        with open_read(path) as fh:
            head = fh.read(size)
    except (OSError, EOFError, CodecError):
        # A file that is not even readable gzip is no mesh either, and
        # another plugin may still know what it is.
        return None
    if len(head) == size and b"\n" in head and b"\0" not in head:
        head = head.rsplit(b"\n", 1)[0]
    return head


def _dispatched(path: str, *, codec: Codec) -> bool:
    """Say whether the dispatcher *codec* hands *path* to a reader taking it.

    The dispatcher gives the file to the first candidate whose sniffer takes
    the registry's window of it, so that one alone decides. A file none of
    them takes goes to the codec owning the extension, when it has one, and
    is taken as :func:`_owned` would take it under that codec.
    """
    head = _head(path, size=None)
    if head is None:
        return False
    for _label, entry in codec.entries:
        if entry.sniff is not None and _passes(entry.sniff, head):
            test = _read_test(entry)
            return test is entry.sniff or _passes(test, head)
    return codec.owner is not None and _owned(path, codec=codec.owner)


def _owned(path: str, *, codec: Codec) -> bool:
    """Say whether *codec*, owning the extension of *path*, reads the file.

    A codec installed through an entry point owns the extension outright,
    and so does a built-in with nothing to test a file by.
    """
    module = getattr(codec.read, "__module__", None) or ""
    if not module.startswith(_CODEC_PACKAGE):
        return True
    test = _read_test(codec)
    if test is None:
        return True
    head = _head(path, size=_HEAD_BYTES)
    return head is not None and _passes(test, head)


def _readable(path: str) -> bool:
    suffix = format_suffix(path)
    codec = polyxios._REGISTRY.get(suffix)
    if codec is None or suffix in _NOT_READ:
        return False
    if suffix in _PLAIN_ONLY and source_suffix(path) in GZIP_SUFFIXES:
        return False
    if suffix not in _SHARED_SUFFIXES:
        return True
    if codec.entries:
        return _dispatched(path, codec=codec)
    return _owned(path, codec=codec)


def _accepts(path: str | Sequence[str]) -> bool:
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    return bool(paths) and all(_readable(os.fspath(p)) for p in paths)


def get_reader(
    *, path: str | Sequence[str]
) -> Callable[[str | Sequence[str]], list[LayerData]] | None:
    """napari reader hook: open in napari's (z, y, x) order.

    Parameters
    ----------
    path
        One path or a list of them, as napari passes it.

    Returns
    -------
    callable or None
        A reader taking the same paths positionally - napari calls it that
        way - or None when a path is not one polyxios reads, so napari moves
        on to the next plugin. An extension other software also uses -
        ``.dat``, ``.e``, ``.plt``, ``.pts``, ``.vol``, ``.xml``, ``.xyz`` -
        is taken only when the sniffer of a codec reading that extension
        recognises the file, and not when it is a broken gzip stream. A
        ``.dat`` goes to the codec :func:`polyxios.read` would hand it to,
        chosen from the same opening bytes, and is taken when that codec
        reads it; a deck whose first card sits past those bytes is not. A codec
        installed for one through the ``polyxios.codecs`` entry point owns
        it, so its files are taken as :func:`polyxios.read` takes them.
        ``.svg``, which polyxios writes and never reads, is never taken. A
        ``.vtm``, ``.pvtu``, ``.pvtp``, ``.pvtr``, ``.pvts`` or ``.pvti``
        index opens as its pieces merged into one layer, through
        :func:`polyxios.helper.read_multiblock`, and so does a plain
        ``.vtp`` holding a multi-block set; a gzipped index is not taken,
        and a gzipped multi-block ``.vtp`` fails with the VTP reader's
        error.
    """
    return _read if _accepts(path) else None


def _layer_info(attributes: dict[str, Any]) -> dict[str, Any]:
    metadata = attributes.get("metadata") or {}
    return metadata.get(META_KEY) or {}


def _layer_axis_order(info: dict[str, Any]) -> AxisOrder:
    axis_order = info.get("axis_order", "zyx")
    if axis_order not in _AXIS_ORDERS:
        raise ValueError(
            f"the layer's metadata[{META_KEY!r}] gives axis_order "
            f"{axis_order!r}, which is none of {_AXIS_ORDERS}; set it to the "
            f"order the layer's columns are in."
        )
    return axis_order


def _row_bits(coords: Any) -> np.ndarray:
    """Return each vertex row's float32 bit pattern, as an (n, d) uint32 array.

    Rows are compared at float32, so a layer napari stored at a narrower
    precision than the file still matches its own vertices.
    """
    rows = np.ascontiguousarray(np.asarray(coords), dtype=np.float32)
    width = int(np.prod(rows.shape[1:], dtype=np.int64))
    return rows.reshape(len(rows), width).view(np.uint32)


def _mix(lane: np.ndarray) -> np.ndarray:
    """Scramble each uint64 of *lane* with the splitmix64 finaliser, a bijection."""
    first, second, third = _MIX_SHIFTS
    lane = (lane ^ (lane >> first)) * _MIX_MULTIPLIERS[0]
    lane = (lane ^ (lane >> second)) * _MIX_MULTIPLIERS[1]
    return lane ^ (lane >> third)


def _row_hash(bits: np.ndarray) -> np.ndarray:
    """Hash each row of an (n, d) uint32 or uint64 array to one uint64.

    Looking rows up among a million means sorting them, and sorting one
    integer per row is several times faster than sorting the rows (or than
    :func:`numpy.isin`). Two uint32 words are packed into each 64-bit lane,
    so a row of three costs two rounds of mixing. A collision makes a moved
    point look shifted, which drops what was kept with a warning, or
    unmoved; a plain multiply-and-xor over the words had them on regular
    grids, which this has not.
    """
    width = bits.shape[1]
    step = 2 if bits.dtype.itemsize == 4 else 1
    out = np.zeros(len(bits), dtype=np.uint64)
    for k in range(0, width, step):
        lane = bits[:, k].astype(np.uint64)
        if step == 2 and k + 1 < width:
            lane |= bits[:, k + 1].astype(np.uint64) << _WORD_BITS
        out = _mix(out ^ lane)
    return out


def _misplaced(info: dict[str, Any], *, coords: Any) -> tuple[str | None, bool]:
    """Say how the layer's vertices stopped lining up with the file's, if they did.

    A point moved in place keeps its index and so its attributes. A point
    deleted shifts every later one down by a row, and one added in its place
    restores the count; what gives that away is a changed row now holding the
    coordinates of another row of the file. A point deleted and another added
    where it was reads as a move. A point moved onto the exact coordinates of
    another vertex of the file reads as a reorder, and drops what was kept.

    Returns
    -------
    reason : str or None
        Why the rows no longer line up, or None while they do.
    moved : bool
        Whether a row of the layer holds coordinates no vertex of the file
        had: a point moved, or one added, whether or not the rows still line
        up.
    """
    n = len(coords)
    expected = info.get("n_vertices")
    keys = info.get("vertex_keys")
    bits = _row_bits(coords) if keys is not None and n else None
    same_dim = bits is not None and bits.shape[1] == info.get("vertex_width")
    now = _row_hash(bits) if same_dim else None
    if expected is not None and expected != n:
        reason = f"the layer has {n} vertices where the file had {expected}"
        return reason, now is not None and bool((~_known_rows(keys, rows=now)).any())
    if bits is None:
        return None, False
    if now is None:
        return "the layer's vertices changed dimension since the file was read", False
    changed = now != keys
    if not changed.any():
        return None, False
    known = _known_rows(keys, rows=now[changed])
    if known.any():
        return (
            "points were deleted, added or reordered since the file was read",
            bool((~known).any()),
        )
    return None, True


def _known_rows(keys: np.ndarray, *, rows: np.ndarray) -> np.ndarray:
    """Say which of the row hashes *rows* is among the row hashes *keys*."""
    if not len(keys):
        return np.zeros(len(rows), dtype=bool)
    known = np.sort(keys)
    at = np.minimum(np.searchsorted(known, rows), len(known) - 1)
    return known[at] == rows


def _kept(info: dict[str, Any], *, coords: Any) -> dict[str, Any]:
    """Return what the reader set aside, or nothing when the vertices changed.

    Every array the reader kept is per vertex or indexes the vertices, so
    once the layer's rows are no longer the file's - a vertex gained, lost or
    shifted to another row - none of it lines up any more. Saying so is
    better than writing attributes onto the wrong points. So is a global
    attribute laying the rows out, such as an organised PCD's
    ``pcd_width`` and ``pcd_height``. The colour source survives: the
    colours themselves are the layer's own and follow its vertices.

    A point moved, or added, has no normal of the file's: one moved keeps
    the normal of the surface as it was, and one added a copy of another's.
    Once any is, the file's normals are listed under ``"stale"``, set aside
    or shown as a feature, for the writer to leave out - whether or not the
    rows still line up, since a Points layer's features follow its points
    through a deletion.
    """
    global_attrs = dict(info.get("global_attrs") or {})
    kept = {
        "vertex_attrs": dict(info.get("vertex_attrs") or {}),
        "vertex_tags": dict(info.get("vertex_tags") or {}),
        "cells": info.get("cells"),
        "colors": info.get("colors"),
        "global_attrs": global_attrs,
        "stale": frozenset(),
    }
    reason, moved = _misplaced(info, coords=coords)
    normals = list(info.get("normals") or ())
    if moved and normals:
        kept["stale"] = frozenset(normals)
        # Normals set aside are named once, below, when the rows no longer
        # line up at all; only those shown as features are left to name.
        named = (
            normals
            if reason is None
            else [k for k in normals if k not in kept["vertex_attrs"]]
        )
        if named:
            warn_caller(
                f"points were moved or added since the file was read; "
                f"{', '.join(named)} describe the points as read and were not "
                f"written."
            )
        if reason is None:
            kept["vertex_attrs"] = {
                k: v for k, v in kept["vertex_attrs"].items() if k not in normals
            }
    if reason is None:
        return kept
    grid = _grid_keys(global_attrs)
    layout = [k for k in global_attrs if k in _LAYOUT_GLOBALS or k in grid]
    lost = [*kept["vertex_attrs"], *kept["vertex_tags"], *layout]
    if kept["cells"] is not None:
        lost.append("the cells")
    if lost:
        warn_caller(
            f"{reason}; {', '.join(lost)} no longer line up and were not written."
        )
    return {
        **kept,
        "vertex_attrs": {},
        "vertex_tags": {},
        "cells": None,
        "global_attrs": {k: v for k, v in global_attrs.items() if k not in layout},
    }


def _free_name(name: str, *, taken: dict[str, Any]) -> str:
    """Return *name*, or the first ``name_k`` not in *taken*."""
    k = 0
    candidate = name
    while candidate in taken:
        k += 1
        candidate = f"{name}_{k}"
    return candidate


def _write(
    path: str,
    *,
    coords: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
    vertex_attrs: dict[str, np.ndarray],
    kept: dict[str, Any],
) -> list[str]:
    poly = make_polydata(
        coords,
        groups,
        vertex_attrs=vertex_attrs,
        vertex_tags=kept["vertex_tags"],
        global_attrs=kept["global_attrs"],
    )
    cells = kept["cells"]
    if cells is not None:
        poly = dataclasses.replace(poly, **cells)
    sidecars = [name(path=path) for name in _SIDECARS.get(format_suffix(path), ())]
    # An earlier save's sidecar is set aside rather than compared by mtime,
    # which a FAT or SMB share cannot tell from a rewrite of the same size:
    # whatever stands there after the write is this save's.
    aside: list[tuple[Path, Path]] = []
    try:
        aside.extend(
            (p, _set_aside(path=p, target=path)) for p in sidecars if p.is_file()
        )
        polyxios.write(poly, path)
    except BaseException:
        for p, old in aside:
            os.replace(old, p)
        raise
    written = [os.fspath(p) for p in sidecars if p.is_file()]
    for p, old in aside:
        if p.is_file():
            os.unlink(old)
        else:
            os.replace(old, p)
    return [path, *written]


def _set_aside(*, path: Path, target: str) -> Path:
    """Move *path* to a free name beside it, and return that name.

    Parameters
    ----------
    path
        The sidecar to move.
    target
        The file being saved, which an error names: the free name is one the
        caller never chose.

    Returns
    -------
    Path
        Where *path* now is.

    Raises
    ------
    OSError
        If *path* cannot be moved, with *target* as its ``filename``; one
        carrying no errno keeps its message, prefixed with *target*.
    """
    name = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        os.replace(path, name)
    except BaseException as exc:
        if name is not None:
            if path.exists():
                Path(name).unlink(missing_ok=True)
            else:
                os.replace(name, path)
        if not isinstance(exc, OSError):
            raise
        if exc.errno is None:
            raise type(exc)(f"'{target}': {exc}") from exc
        raise type(exc)(exc.errno, exc.strerror, target) from exc
    return Path(name)


def write_surface(path: str, data: Any, attributes: dict[str, Any]) -> list[str]:
    """napari writer hook for one Surface layer.

    napari calls this positionally, so the signature is its own.

    Parameters
    ----------
    path
        Destination; its extension picks the format.
    data
        ``(vertices, faces)`` or ``(vertices, faces, values)``. 2-D
        vertices are written as 3-D ones in the ``z = 0`` plane. Values of
        shape (n,) are written as one attribute; a time series of shape
        (T, n) as one attribute of shape (n, T). napari puts the vertices
        last, so a square (n, n) array is a time series too. Values keep
        the dtype the layer holds them in. On a layer read without
        values, or drawn in napari, values of all ones are napari's default
        and are not written; other values are written as ``values``, or
        ``values_1`` and on when the file has an attribute of that name.
    attributes
        The layer's state as napari hands it over. The ``metadata`` entry the
        reader left tells which axis order to undo, what the values were
        called, and what else the file held; a layer drawn in napari has none
        and is taken as (z, y, x), its triangles wound for that order. Vertex
        colours are written back under the name they were read from, with an
        alpha channel added when the layer's is not opaque and the file's
        colours can hold one. The layer's ``scale`` and ``translate`` are
        not applied: the file gets the layer's data coordinates.

    Returns
    -------
    list of str
        Every file written: the path, then any file the format keeps beside
        it, such as the ``.bin`` of a ``.gltf``.

    Warns
    -----
    UserWarning
        When the file had cells other than triangles, element data or a
        structured grid, which a Surface layer cannot carry; when the values
        or colours do not fit the vertices, or the colours have an alpha no
        channel of the file's holds; when vertices were moved or added, so
        the file's normals no longer fit them; or when the layer's vertices are no
        longer the file's, so what the reader set aside cannot be written
        back.

    Raises
    ------
    ValueError
        If the ``metadata`` entry names an axis order other than ``"zyx"``
        or ``"xyz"``, or the vertices are not 2-D or 3-D points.
    """
    info = _layer_info(attributes)
    axis_order = _layer_axis_order(info)
    vertices, faces, *rest = data
    coords = _from_napari(vertices, axis_order=axis_order)
    n = len(coords)
    kept = _kept(info, coords=vertices)
    vertex_attrs = dict(kept["vertex_attrs"])
    values = np.asarray(rest[0]) if rest else None
    if values is not None and info.get("values") is None:
        if np.all(values == 1):
            # napari fills the values of a Surface given none with ones.
            values = None
    if values is not None:
        name = info.get("values") or _free_name("values", taken=vertex_attrs)
        if values.ndim == 1 and len(values) == n:
            vertex_attrs[name] = values
        elif values.ndim == 2 and values.shape[1] == n:
            vertex_attrs[name] = np.ascontiguousarray(values.T)
        else:
            warn_caller(
                f"the layer's values have shape {values.shape} for {n} "
                f"vertices; they were not written."
            )
    colors = attributes.get("vertex_colors")
    if kept["colors"] is not None and colors is not None:
        vertex_attrs.update(
            _restore_colors(colors, source=kept["colors"], n=n, taken=vertex_attrs)
        )
    not_kept = info.get("not_kept") or []
    if not_kept:
        warn_caller(f"a Surface layer holds triangles only: {'; '.join(not_kept)}.")
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if _flips(axis_order):
        triangles = triangles[:, ::-1]
    return _write(
        path,
        coords=coords,
        groups=[("triangle", triangles)],
        vertex_attrs=vertex_attrs,
        kept=kept,
    )


def write_points(path: str, data: Any, attributes: dict[str, Any]) -> list[str]:
    """napari writer hook for one Points layer.

    napari calls this positionally, so the signature is its own.

    Parameters
    ----------
    path
        Destination; its extension picks the format.
    data
        Point coordinates, shape (n, 2) or (n, 3). 2-D points are written
        as 3-D ones in the ``z = 0`` plane.
    attributes
        The layer's state as napari hands it over. Numeric ``features``
        columns are written as vertex attributes; ``face_color`` is written
        as colours only when the layer was read with colours, since every
        napari layer has a face colour whether the file had one or not; an
        alpha is added as for :func:`write_surface`. The cells of a line set
        and the attributes a feature cannot hold come back from the reader's
        ``metadata`` entry. The layer's ``scale`` and ``translate`` are not
        applied: the file gets the layer's data coordinates.

    Returns
    -------
    list of str
        Every file written: the path, then any file the format keeps beside
        it, such as the ``.bin`` of a ``.gltf``.

    Warns
    -----
    UserWarning
        When a feature column is not one number per point, so no mesh
        attribute holds it; or when the layer's points are no longer the
        file's - gained, lost or shifted to another row - so what the reader
        set aside cannot be written back; when points were moved or added,
        so the file's normals, set aside or shown as features, no longer fit
        them and are not written; when the colours have an alpha
        no channel of the file's holds; or when a feature has the name of
        the file's colours, of an attribute the reader set aside, or of
        another feature once spelled as text, so it is written under another.

    Raises
    ------
    ValueError
        If the ``metadata`` entry names an axis order other than ``"zyx"``
        or ``"xyz"``, or the points are not 2-D or 3-D.
    """
    info = _layer_info(attributes)
    coords = _from_napari(data, axis_order=_layer_axis_order(info))
    n = len(coords)
    kept = _kept(info, coords=data)
    features = attributes.get("features")
    columns: dict[str, np.ndarray] = {}
    for key in features if features is not None else ():
        name = _free_name(str(key), taken=columns)
        if name != str(key):
            warn_caller(
                f"feature {key!r} is spelled {str(key)!r} like another one; it "
                f"was written as {name}."
            )
        columns[name] = np.asarray(features[key])
    numeric = {
        k: v for k, v in _scalar_columns(columns, n=n).items() if k not in kept["stale"]
    }
    dropped = [k for k in columns if k not in numeric and k not in kept["stale"]]
    if dropped:
        warn_caller(
            f"features {', '.join(dropped)} are not one number per point, "
            f"which is what a mesh attribute holds; they were not written."
        )
    face_color = attributes.get("face_color")
    if not n:
        # napari hands an empty layer its current colour as a single row.
        face_color = np.empty((0, 4))
    colors = {}
    if kept["colors"] is not None and face_color is not None:
        colors = _restore_colors(
            face_color, source=kept["colors"], n=n, taken=kept["vertex_attrs"]
        )
    vertex_attrs = {**kept["vertex_attrs"], **colors}
    taken = {**vertex_attrs, **numeric}
    for key, column in numeric.items():
        if key in vertex_attrs:
            free = _free_name(key, taken=taken)
            taken[free] = column
            warn_caller(
                f"feature {key} has the name of the file's own {key}; it was "
                f"written as {free}."
            )
            key = free
        vertex_attrs[key] = column
    return _write(
        path,
        coords=coords,
        groups=[],
        vertex_attrs=vertex_attrs,
        kept=kept,
    )
