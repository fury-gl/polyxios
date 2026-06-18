from collections import defaultdict
from collections.abc import Callable
import dataclasses
from functools import reduce

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    SURFACE_ELEMENT_TYPES,
)
from polyxios._types import PolyData

_SURFACE_CODES = SURFACE_ELEMENT_TYPES
_TRIANGLE_CODE = ELEMENT_TYPES["triangle"]
_QUAD_PIXEL_CODES = frozenset({ELEMENT_TYPES["quad"], ELEMENT_TYPES["pixel"]})


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
    new_vertex_tags = {k: remap[v[v < n_verts]] for k, v in poly.vertex_tags.items()}
    # filter out -1 from remapped tag arrays
    new_vertex_tags = {k: v[v >= 0] for k, v in new_vertex_tags.items()}

    return dataclasses.replace(
        poly,
        vertices=new_vertices,
        connectivity=new_connectivity.astype(poly.connectivity.dtype),
        vertex_attrs=new_vertex_attrs,
        vertex_tags=new_vertex_tags,
    )


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
                fill = np.full(
                    p.vertices.shape[0],
                    np.nan if np.issubdtype(ref.dtype, np.floating) else -1,
                    dtype=ref.dtype,
                )
                parts.append(fill)
        merged_vertex_attrs[key] = np.concatenate(parts)

    merged_element_attrs: dict[str, np.ndarray] = {}
    for key in all_element_attr_keys:
        parts = []
        for p in polys:
            if key in p.element_attrs:
                parts.append(p.element_attrs[key])
            else:
                ref = next(
                    q.element_attrs[key] for q in polys if key in q.element_attrs
                )
                fill = np.full(
                    len(p.element_types),
                    np.nan if np.issubdtype(ref.dtype, np.floating) else -1,
                    dtype=ref.dtype,
                )
                parts.append(fill)
        merged_element_attrs[key] = np.concatenate(parts)

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
    new_element_tags: dict[str, np.ndarray] = {}
    for k, v in poly.element_tags.items():
        remapped = idx_map[v[v < len(poly.element_types)]]
        new_element_tags[k] = remapped[remapped >= 0].astype(v.dtype)

    return dataclasses.replace(
        poly,
        connectivity=new_connectivity,
        offsets=np.array(new_offsets, dtype=poly.offsets.dtype),
        element_types=new_element_types,
        element_attrs=new_element_attrs,
        element_tags=new_element_tags,
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
        vertex attribute with 3 or more channels exists.
    """
    for arr in poly.vertex_attrs.values():
        if arr.ndim == 2 and arr.shape[1] >= 3:
            rgb = arr[:, :3].astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            return rgb
    return None


# Local corner-node face definitions for 3D volumetric element types.
# Each entry is a list of tuples of local vertex indices that form one face.
# Quadratic elements reuse corner nodes only (indices match linear sub-element).
_VOL_ELEMENT_FACES: dict[str, list[tuple[int, ...]]] = {
    "tetra": [
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 2, 1),
    ],
    "hexahedron": [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ],
    # VTK voxel: bit-encoded ordering differs from hex
    "voxel": [
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    ],
    "wedge": [
        (0, 1, 2),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ],
    "pyramid": [
        (0, 1, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ],
    "pentagonal_prism": [
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (0, 1, 6, 5),
        (1, 2, 7, 6),
        (2, 3, 8, 7),
        (3, 4, 9, 8),
        (4, 0, 5, 9),
    ],
    "hexagonal_prism": [
        (0, 1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10, 11),
        (0, 1, 7, 6),
        (1, 2, 8, 7),
        (2, 3, 9, 8),
        (3, 4, 10, 9),
        (4, 5, 11, 10),
        (5, 0, 6, 11),
    ],
}
# Quadratic elements: reuse corner-node faces of their linear counterparts.
_VOL_ELEMENT_FACES["quadratic_tetra"] = _VOL_ELEMENT_FACES["tetra"]
_VOL_ELEMENT_FACES["triquadratic_hexahedron"] = _VOL_ELEMENT_FACES["hexahedron"]
_VOL_ELEMENT_FACES["biquadratic_quadratic_wedge"] = _VOL_ELEMENT_FACES["wedge"]
_VOL_ELEMENT_FACES["quadratic_hexahedron"] = _VOL_ELEMENT_FACES["hexahedron"]
_VOL_ELEMENT_FACES["quadratic_wedge"] = _VOL_ELEMENT_FACES["wedge"]
_VOL_ELEMENT_FACES["quadratic_pyramid"] = _VOL_ELEMENT_FACES["pyramid"]


def extract_surface(poly: PolyData) -> PolyData:
    """Return the boundary surface of a volumetric PolyData.

    A boundary face is one shared by exactly one element. Surface elements
    already present in the mesh (triangle, quad, polygon, etc.) are ignored
    during extraction — only 3D volumetric elements contribute.

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
