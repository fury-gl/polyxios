from collections import defaultdict
from collections.abc import Callable
import dataclasses
from functools import reduce
from typing import Final

import numpy as np

from polyxios._element_types import (
    ELEMENT_FACES,
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    QUADRATIC_SURFACE_CORNERS,
    SURFACE_ELEMENT_TYPES,
)
from polyxios._faces import FACE_PARENT_KEY
from polyxios._tags import member_indices, members_array
from polyxios._types import PolyData

_SURFACE_CODES = SURFACE_ELEMENT_TYPES
_TRIANGLE_CODE = ELEMENT_TYPES["triangle"]
_QUAD_PIXEL_CODES = frozenset({ELEMENT_TYPES["quad"], ELEMENT_TYPES["pixel"]})

# The vertex attribute a codec writes colours to when the format names them
# (.off, and anything else that grows colour support).
_COLOR_ATTR_NAME = "colors"


def pipeline(*fns: Callable[[PolyData], PolyData]) -> Callable[[PolyData], PolyData]:
    """Left-to-right function composition for PolyData transforms.

    Parameters
    ----------
    *fns
        Transform functions to compose.

    Returns
    -------
    Callable
        A single function applying all transforms in order.
    """
    return lambda poly: reduce(lambda p, f: f(p), fns, poly)


def _remapped_tags(
    tags: dict[str, np.ndarray],
    remap: np.ndarray,
    n_items: int,
    *,
    collapse: bool,
) -> dict[str, np.ndarray]:
    """Return tag groups carried through an index remap.

    Parameters
    ----------
    tags
        Tag groups as the mesh carries them, name to member indices.
    remap
        New index for each old index, negative where the item is gone.
    n_items
        How many items the groups were built against.
    collapse
        Whether members landing on one new index become a single member.
        True where the remap welds items together, False where it only
        moves them.

    Returns
    -------
    dict of str to numpy.ndarray
        The groups over the new indices. A member that indexes nothing in
        this mesh is dropped rather than raising: nothing checks a group on
        the way in, so one may hold floats, a negative index or an index
        past the end of the mesh it was built for.
    """
    out: dict[str, np.ndarray] = {}
    for name, members in tags.items():
        held = members_array(members)
        moved = remap[member_indices(held, n_items)]
        moved = moved[moved >= 0]
        if collapse:
            moved = np.unique(moved)
        # The group's own width is kept where it has one. A group numpy builds
        # no array of has none, and names nothing either, so what it comes back
        # as is the empty column the remap left.
        dtype = moved.dtype if held is None else held.dtype
        out[name] = moved.astype(dtype, copy=False) if dtype.kind in "iu" else moved
    return out


def remove_orphan_vertices(poly: PolyData) -> PolyData:
    """Return a new PolyData with unreferenced vertices removed and indices remapped.

    Parameters
    ----------
    poly
        Input PolyData.

    Returns
    -------
    PolyData
        New PolyData without orphan vertices.
    """
    from polyxios._backend import compact_vertex_indices, has_orphan_vertices

    n_verts = poly.vertices.shape[0]
    conn32 = poly.connectivity.astype(np.int32, copy=False)

    if not has_orphan_vertices(n_verts, conn32):
        return poly

    remap = np.asarray(compact_vertex_indices(conn32, n_verts))

    kept = remap >= 0
    new_vertices = poly.vertices[kept]
    new_connectivity = remap[poly.connectivity]

    new_vertex_attrs = {k: v[kept] for k, v in poly.vertex_attrs.items()}
    new_vertex_tags = _remapped_tags(poly.vertex_tags, remap, n_verts, collapse=False)

    return dataclasses.replace(
        poly,
        vertices=new_vertices,
        connectivity=new_connectivity.astype(poly.connectivity.dtype),
        vertex_attrs=new_vertex_attrs,
        vertex_tags=new_vertex_tags,
    )


def _fill_like(ref: np.ndarray, length: int) -> np.ndarray:
    """Build a placeholder block matching a reference attribute's channels.

    An attribute is not always one number per vertex or element - normals,
    colours and texture coordinates are (n, k) - so the placeholder has to
    keep the reference's trailing shape or the concatenation has nothing to
    line up against.
    """
    return np.full(
        (length, *ref.shape[1:]),
        np.nan if np.issubdtype(ref.dtype, np.floating) else -1,
        dtype=ref.dtype,
    )


def _concat_attr(parts: list[np.ndarray], what: str) -> np.ndarray:
    """Join one attribute's blocks, naming it if their channels disagree."""
    shapes = {p.shape[1:] for p in parts}
    if len(shapes) > 1:
        raise ValueError(
            f"cannot merge {what}: channel counts differ across inputs "
            f"({', '.join(str(s) for s in sorted(shapes))})"
        )
    return np.concatenate(parts)


def _shifted_parents(column: np.ndarray, offset: int) -> np.ndarray:
    """Renumber a ``face_parent`` column onto the mesh a merge is building.

    Parameters
    ----------
    column
        The column as one of the merged meshes carries it.
    offset
        Where that mesh's elements start in the merged one.

    Returns
    -------
    numpy.ndarray
        The column with every parent it names shifted by ``offset``, and the
        ``-1`` that says "not a face" left alone. The column unchanged when it
        is not one number per element, which is the same thing
        :func:`~polyxios._faces.parent_faces` refuses to read.

    Notes
    -----
    The one element attribute that is an index into the mesh rather than a
    value. ``merge`` shifts the tag groups for the same reason; a column left
    unshifted would name an element of whichever mesh went first, which
    :func:`~polyxios._faces.is_parent_face` then answers by writing every
    surface back as the ``*Elset`` it is not.

    A column carrying whole numbers as doubles counts, because a mesh that
    went out through a format spelling every attribute as one - legacy
    ``.vtk`` among them - comes back holding them that way. So does one
    carrying them as bools: every kind
    :func:`~polyxios._faces.parent_faces` reads back is a kind this shifts,
    or a merged mesh keeps a column that reader still trusts pointing into
    whichever mesh went first.

    An unsigned column is the one kind that cannot spell the ``-1`` saying
    "not a face", so every row of one is shifted as the index it claims to
    be. Nothing is broken by that - :func:`~polyxios._faces.is_parent_face`
    checks each claim against the vertices before a writer acts on it - but
    it is why ``-1`` is what these columns are built with.
    """
    held = np.asarray(column)
    if offset == 0 or held.ndim != 1 or held.dtype.kind not in "biuf":
        return held
    if held.dtype.kind == "b":
        # A bool column cannot spell the -1 that says "not a face", but it is
        # one whole number per element, which is all parent_faces asks of it.
        # Widened rather than left alone: a shift is what keeps it pointing
        # at the element it pointed at before the merge.
        held = held.astype(np.int64)
    is_face = held >= 0
    if not is_face.any():
        return held
    dtype = held.dtype
    if dtype.kind in "iu" and int(held[is_face].max()) + offset > np.iinfo(dtype).max:
        # An element index fits in the column a reader built for it, but two
        # meshes laid end to end are counted past the end of one mesh's.
        dtype = np.dtype(np.int64)
    shifted = held.astype(dtype, copy=True)
    shifted[is_face] += offset
    return shifted


def merge(*polys: PolyData) -> PolyData:
    """Merge multiple PolyData into one by concatenating vertices and elements.

    Parameters
    ----------
    *polys
        PolyData objects to merge.

    Returns
    -------
    PolyData
        Single PolyData with all vertices and elements from inputs.
    """
    if not polys:
        raise ValueError("merge requires at least one PolyData")
    if len(polys) == 1:
        return polys[0]

    all_vertices = np.concatenate([p.vertices for p in polys])

    # Offset connectivity indices for each chunk
    conn_parts: list[np.ndarray] = []
    vert_offset = 0
    for p in polys:
        conn_parts.append(p.connectivity + vert_offset)
        vert_offset += p.vertices.shape[0]

    all_connectivity = np.concatenate(conn_parts)

    # Correct offset concatenation: shift each poly's internal offsets by previous conn size
    offset_acc = 0
    merged_offsets_list: list[np.ndarray] = []
    for p in polys:
        if not merged_offsets_list:
            merged_offsets_list.append(p.offsets)
        else:
            merged_offsets_list.append(p.offsets[1:] + offset_acc)
        offset_acc += int(p.offsets[-1]) if len(p.offsets) > 0 else 0

    all_offsets = np.concatenate(merged_offsets_list)
    all_element_types = np.concatenate([p.element_types for p in polys])

    # Merge attrs: only include keys present in all polys, fill missing with nan/-1
    all_vertex_attr_keys: set[str] = set()
    all_element_attr_keys: set[str] = set()
    for p in polys:
        all_vertex_attr_keys.update(p.vertex_attrs)
        all_element_attr_keys.update(p.element_attrs)

    merged_vertex_attrs: dict[str, np.ndarray] = {}
    for key in all_vertex_attr_keys:
        parts = []
        for p in polys:
            if key in p.vertex_attrs:
                parts.append(p.vertex_attrs[key])
            else:
                ref = next(q.vertex_attrs[key] for q in polys if key in q.vertex_attrs)
                parts.append(_fill_like(ref, p.vertices.shape[0]))
        merged_vertex_attrs[key] = _concat_attr(parts, f"vertex_attrs['{key}']")

    # Where each mesh's elements land, for the one attribute that is an
    # element index rather than a value: a face_parent numbered from zero in
    # its own mesh names another mesh's element once they are laid end to end.
    elem_starts: list[int] = []
    running = 0
    for p in polys:
        elem_starts.append(running)
        running += len(p.element_types)

    merged_element_attrs: dict[str, np.ndarray] = {}
    for key in all_element_attr_keys:
        parts = []
        for start, p in zip(elem_starts, polys):
            if key in p.element_attrs:
                held = p.element_attrs[key]
                parts.append(
                    _shifted_parents(held, start) if key == FACE_PARENT_KEY else held
                )
            else:
                ref = next(
                    q.element_attrs[key] for q in polys if key in q.element_attrs
                )
                parts.append(_fill_like(ref, len(p.element_types)))
        merged_element_attrs[key] = _concat_attr(parts, f"element_attrs['{key}']")

    # Merge element_tags: shift element indices
    merged_element_tags: dict[str, np.ndarray] = {}
    all_etag_keys: set[str] = set()
    for p in polys:
        all_etag_keys.update(p.element_tags)

    elem_offset = 0
    per_poly_etags: list[dict[str, np.ndarray]] = []
    for p in polys:
        shifted = {k: v + elem_offset for k, v in p.element_tags.items()}
        per_poly_etags.append(shifted)
        elem_offset += len(p.element_types)

    for key in all_etag_keys:
        parts = [d[key] for d in per_poly_etags if key in d]
        merged_element_tags[key] = np.concatenate(parts)

    # Merge vertex_tags: shift vertex indices
    merged_vertex_tags: dict[str, np.ndarray] = {}
    all_vtag_keys: set[str] = set()
    for p in polys:
        all_vtag_keys.update(p.vertex_tags)

    vert_offset2 = 0
    per_poly_vtags: list[dict[str, np.ndarray]] = []
    for p in polys:
        shifted = {k: v + vert_offset2 for k, v in p.vertex_tags.items()}
        per_poly_vtags.append(shifted)
        vert_offset2 += p.vertices.shape[0]

    for key in all_vtag_keys:
        parts = [d[key] for d in per_poly_vtags if key in d]
        merged_vertex_tags[key] = np.concatenate(parts)

    idx_dtype = (
        np.int64
        if all_connectivity.size > 0 and all_connectivity.max() >= 2**31
        else np.int32
    )

    return PolyData(
        vertices=all_vertices,
        connectivity=all_connectivity.astype(idx_dtype),
        offsets=all_offsets.astype(idx_dtype),
        element_types=all_element_types,
        vertex_attrs=merged_vertex_attrs,
        element_attrs=merged_element_attrs,
        vertex_tags=merged_vertex_tags,
        element_tags=merged_element_tags,
        global_attrs={},
    )


def filter_element_type(poly: PolyData, *, keep: str | list[str]) -> PolyData:
    """Return a new PolyData containing only elements of the specified type(s).

    Parameters
    ----------
    poly
        Input PolyData.
    keep
        Element type name(s) to keep (e.g. "triangle" or ["triangle", "quad"]).

    Returns
    -------
    PolyData
        New PolyData with only the requested element types.
    """
    if isinstance(keep, str):
        keep = [keep]
    keep_codes = {ELEMENT_TYPES[t] for t in keep}

    mask = np.isin(poly.element_types, list(keep_codes))
    elem_indices = np.where(mask)[0]

    if elem_indices.size == 0:
        return dataclasses.replace(
            poly,
            connectivity=np.array([], dtype=poly.connectivity.dtype),
            offsets=np.array([0], dtype=poly.offsets.dtype),
            element_types=np.array([], dtype=np.uint8),
            element_attrs={k: v[[]].copy() for k, v in poly.element_attrs.items()},
            element_tags={
                k: np.array([], dtype=v.dtype) for k, v in poly.element_tags.items()
            },
        )

    conn_parts: list[np.ndarray] = []
    new_offsets: list[int] = [0]

    for i in elem_indices:
        start = int(poly.offsets[i])
        end = int(poly.offsets[i + 1])
        conn_parts.append(poly.connectivity[start:end])
        new_offsets.append(new_offsets[-1] + (end - start))

    new_connectivity = (
        np.concatenate(conn_parts)
        if conn_parts
        else np.array([], dtype=poly.connectivity.dtype)
    )
    new_element_types = poly.element_types[elem_indices]
    new_element_attrs = {k: v[elem_indices] for k, v in poly.element_attrs.items()}

    # Remap element_tags to new indices
    idx_map = np.full(len(poly.element_types), -1, dtype=np.int64)
    idx_map[elem_indices] = np.arange(len(elem_indices))
    new_element_tags = _remapped_tags(
        poly.element_tags, idx_map, len(poly.element_types), collapse=False
    )

    return dataclasses.replace(
        poly,
        connectivity=new_connectivity,
        offsets=np.array(new_offsets, dtype=poly.offsets.dtype),
        element_types=new_element_types,
        element_attrs=new_element_attrs,
        element_tags=new_element_tags,
    )


def merge_duplicate_vertices(poly: PolyData, *, tol: float = 0.0) -> PolyData:
    """Return a new PolyData with coincident vertices welded into one.

    The equivalent of ParaView's "Clean to Grid": formats that write a
    corner per element - STL above all - hand back a soup of unconnected
    vertices, and welding them is what turns it back into a surface.

    Parameters
    ----------
    poly
        Input PolyData.
    tol
        Distance below which two vertices count as the same point. Zero
        welds only exactly equal coordinates. A positive value snaps each
        coordinate to a grid of that step before comparing, so vertices
        merge when they land in the same cell - two points closer than
        ``tol`` that straddle a cell boundary stay apart.

    Returns
    -------
    PolyData
        New PolyData whose vertices are unique. The survivor of each group
        is its lowest original index, and the surviving vertices stay in
        their original relative order, so the result does not depend on
        which duplicate the file listed first. The survivor keeps its own
        coordinates and vertex attributes - a tolerance decides who merges,
        it never moves a point.

    Raises
    ------
    ValueError
        If ``tol`` is negative, or so small that snapping a coordinate
        overflows to infinity - every overflowed point would then weld,
        however far apart the points really are.

    Notes
    -----
    Welding is not culling: a vertex no element references is not a
    duplicate of anything and is kept. Compose with
    ``remove_orphan_vertices`` to drop those as well. Elements that become
    degenerate because two of their corners welded together are kept as
    they are - deciding what a collapsed element means belongs to the
    caller, not here.
    """
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    n_verts = poly.vertices.shape[0]
    if n_verts == 0:
        return poly

    # Snap to a grid so near-coincident points share a key, then sort: a
    # lexsort is reproducible where grouping through a dict is not.
    if tol > 0:
        with np.errstate(over="ignore"):
            keys = np.round(poly.vertices / tol)
        # A tol small enough to send a finite coordinate to infinity would
        # weld every point that overflowed, however far apart they are.
        if (
            not np.isfinite(keys).all()
            and (~np.isfinite(keys) & np.isfinite(poly.vertices)).any()
        ):
            raise ValueError(
                f"tol={tol} is too small for these coordinates: snapping "
                "overflows to infinity and would weld distinct points."
            )
    else:
        keys = poly.vertices
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]

    starts_group = np.empty(n_verts, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    if starts_group.all():
        return poly

    group_of = np.cumsum(starts_group) - 1
    # The survivor is the lowest original index in its group, which keeps
    # the kept vertices in the order the file listed them. reduceat over the
    # group starts does it in one pass - np.minimum.at would not.
    survivor = np.minimum.reduceat(order, np.flatnonzero(starts_group))
    kept = np.sort(survivor)

    remap = np.empty(n_verts, dtype=np.int64)
    remap[order] = np.searchsorted(kept, survivor)[group_of]

    new_vertex_tags = _remapped_tags(poly.vertex_tags, remap, n_verts, collapse=True)

    return dataclasses.replace(
        poly,
        vertices=poly.vertices[kept],
        connectivity=remap[poly.connectivity].astype(
            poly.connectivity.dtype, copy=False
        ),
        vertex_attrs={k: v[kept] for k, v in poly.vertex_attrs.items()},
        vertex_tags=new_vertex_tags,
    )


def reindex(poly: PolyData) -> PolyData:
    """Return a new PolyData with compact vertex indices (removes orphans).

    Parameters
    ----------
    poly
        Input PolyData.

    Returns
    -------
    PolyData
        New PolyData with orphan vertices removed.
    """
    return remove_orphan_vertices(poly)


def triangulate(poly: PolyData) -> PolyData:
    """Return a new PolyData with all surface elements converted to triangles.

    Quads and pixels are split into 2 triangles. Polygons and triangle_strips
    are fan-triangulated. Non-surface elements (lines, volumes) are dropped.

    Parameters
    ----------
    poly
        Input PolyData (may contain mixed element types).

    Returns
    -------
    PolyData
        New PolyData with only triangle elements. Vertex attrs preserved.
        Element attrs expanded: each source element's values repeated once
        per generated triangle. Non-surface elements are dropped.
    """
    conn_parts: list[np.ndarray] = []
    src_indices: list[int] = []

    for i in range(len(poly.element_types)):
        etype = int(poly.element_types[i])
        if etype not in _SURFACE_CODES:
            continue
        cell = poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]]
        # Quadratic elements: linearize to corner nodes before triangulating.
        n_corners = QUADRATIC_SURFACE_CORNERS.get(etype)
        if n_corners is not None:
            cell = cell[:n_corners]
            etype = _TRIANGLE_CODE if n_corners == 3 else int(ELEMENT_TYPES["quad"])
        if etype == _TRIANGLE_CODE:
            conn_parts.append(cell)
            src_indices.append(i)
        elif etype in _QUAD_PIXEL_CODES:
            conn_parts.append(cell[[0, 1, 2]])
            conn_parts.append(cell[[0, 2, 3]])
            src_indices.extend([i, i])
        else:
            for j in range(1, len(cell) - 1):
                conn_parts.append(cell[[0, j, j + 1]])
                src_indices.append(i)

    if not conn_parts:
        return dataclasses.replace(
            poly,
            connectivity=np.array([], dtype=poly.connectivity.dtype),
            offsets=np.array([0], dtype=poly.offsets.dtype),
            element_types=np.array([], dtype=np.uint8),
            element_attrs={k: v[[]].copy() for k, v in poly.element_attrs.items()},
            element_tags={
                k: np.array([], dtype=v.dtype) for k, v in poly.element_tags.items()
            },
        )

    idx = np.array(src_indices, dtype=np.int64)
    n_tris = len(src_indices)
    new_connectivity = np.concatenate(conn_parts).astype(poly.connectivity.dtype)
    new_offsets = np.arange(0, (n_tris + 1) * 3, 3, dtype=poly.offsets.dtype)
    new_element_types = np.full(n_tris, _TRIANGLE_CODE, dtype=np.uint8)
    new_element_attrs = {k: v[idx] for k, v in poly.element_attrs.items()}

    old_to_new: dict[int, list[int]] = {}
    for new_i, old_i in enumerate(src_indices):
        old_to_new.setdefault(old_i, []).append(new_i)

    new_element_tags: dict[str, np.ndarray] = {}
    for k, v in poly.element_tags.items():
        new_inds: list[int] = []
        for old_i in v:
            new_inds.extend(old_to_new.get(int(old_i), []))
        new_element_tags[k] = np.array(new_inds, dtype=v.dtype)

    return dataclasses.replace(
        poly,
        connectivity=new_connectivity,
        offsets=new_offsets,
        element_types=new_element_types,
        element_attrs=new_element_attrs,
        element_tags=new_element_tags,
    )


def vertex_colors(poly: PolyData) -> np.ndarray | None:
    """Extract per-vertex RGB colors from the first eligible vertex attribute.

    Parameters
    ----------
    poly
        Input PolyData.

    Returns
    -------
    numpy.ndarray or None
        Float32 array of shape (n_verts, 3) in [0, 1], or None if no
        vertex attribute looks like a colour.

    Notes
    -----
    An attribute named for colour wins outright. Otherwise the first (n, >= 3)
    attribute is taken, skipping any that carries a negative value: codecs
    that read colours also read normals, and a normal is the same shape as an
    RGB triple, so shape alone would hand back surface directions as colours.
    """
    named = poly.vertex_attrs.get(_COLOR_ATTR_NAME)
    candidates = [named] if named is not None else []
    candidates.extend(
        arr for name, arr in poly.vertex_attrs.items() if name != _COLOR_ATTR_NAME
    )

    for arr in candidates:
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        rgb = arr[:, :3].astype(np.float32)
        if arr is not named and rgb.min() < 0.0:
            continue
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        return rgb
    return None


# Local corner-node face definitions for 3D volumetric element types.
# Each entry is a list of tuples of local vertex indices that form one face.
# Quadratic elements reuse corner nodes only (indices match linear sub-element).
# The table moved to polyxios._element_types, which is where a per-element-type
# fact belongs and where the codecs can reach it without importing this module.
# The old name is kept: it is what extract_surface and helper.py read it by.
_VOL_ELEMENT_FACES: Final[dict[str, tuple[tuple[int, ...], ...]]] = ELEMENT_FACES


def extract_surface(poly: PolyData) -> PolyData:
    """Return the boundary surface of a volumetric PolyData.

    A boundary face is one shared by exactly one element. Surface elements
    already present in the mesh (triangle, quad, polygon, etc.) are ignored
    during extraction - only 3D volumetric elements contribute.

    Parameters
    ----------
    poly
        Input PolyData (may contain volumetric elements).

    Returns
    -------
    PolyData
        New PolyData containing only boundary faces (triangles, quads,
        polygons). Vertices and vertex_attrs are preserved unchanged.
        Call ``remove_orphan_vertices`` afterwards to compact the vertex array.
    """
    # sorted-vertex-key → (count, actual_face_vertices)
    face_count: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)

    for i in range(len(poly.element_types)):
        etype_name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]))
        local_faces = _VOL_ELEMENT_FACES.get(etype_name or "")
        if local_faces is None:
            continue
        cell = poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]]
        for local_idx in local_faces:
            face_verts = tuple(int(cell[j]) for j in local_idx)
            key = tuple(sorted(face_verts))
            face_count[key].append(face_verts)

    conn_list: list[int] = []
    off_list: list[int] = [0]
    type_list: list[int] = []

    for appearances in face_count.values():
        if len(appearances) != 1:
            continue
        face = appearances[0]
        n = len(face)
        conn_list.extend(face)
        off_list.append(off_list[-1] + n)
        if n == 3:
            type_list.append(ELEMENT_TYPES["triangle"])
        elif n == 4:
            type_list.append(ELEMENT_TYPES["quad"])
        else:
            type_list.append(ELEMENT_TYPES["polygon"])

    return dataclasses.replace(
        poly,
        connectivity=np.array(conn_list, dtype=poly.connectivity.dtype),
        offsets=np.array(off_list, dtype=poly.offsets.dtype),
        element_types=np.array(type_list, dtype=np.uint8),
        element_attrs={},
        element_tags={},
    )
