"""MED codec - the HDF5 mesh file of Salome and code_aster.

A MED file is one HDF5 file with a fixed tree: the meshes under ``ENS_MAA``,
each a group named for the mesh holding its nodes (``NOE``) and its cells by
geometry (``MAI/<TYPE>``), the families under ``FAS`` and the fields under
``CHA``. Every node and every cell carries a *family* number, and a family
belongs to any number of named *groups*: that is how a MED mesh spells its
sets, and a node or a cell in two groups is a family of its own. Families
land here as tag groups - one per MED group, members being the entities of
every family the group names - so nothing about the grouping is lost, and a
mesh with an element in two groups writes back as one family per
combination, which is what Salome itself does.

The coordinates and the connectivity are stored *component-major* (every X,
then every Y, then every Z; every first node, then every second node),
which is why the reshapes here are Fortran-ordered. Node numbering follows
Salome's, which is VTK's for every type MED names; the triquadratic
hexahedron's face centres are the one place the two disagree.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from polyxios import transforms
from polyxios._dimension import mark_2d, output_dimension, pad_to_3d
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV, NODES_PER_ELEMENT
from polyxios._ids import IDS_KEY, ids_for_write, record_ids
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
    text_of,
    type_blocks,
    warn_unknown_opts,
)
from polyxios.exceptions import CodecError
from polyxios.validate import validate_header

EXTENSION: str = ".med"

# The MED geometry names polyxios can hold and what they read as. Node order
# is Salome's, which is VTK's for every one of them except H27, whose six
# face centres MED lists bottom, front, right, back, left, top and VTK lists
# by axis: x-min, x-max, y-min, y-max, z-min, z-max. Entry ``i`` of a
# permutation is the MED position of the node VTK puts at ``i``.
_MED_TO_POLYXIOS: dict[str, str] = {
    "PO1": "vertex",
    "SE2": "line",
    "SE3": "quadratic_edge",
    "SE4": "cubic_line",
    "TR3": "triangle",
    "TR6": "quadratic_triangle",
    "TR7": "biquadratic_triangle",
    "QU4": "quad",
    "QU8": "quadratic_quad",
    "QU9": "biquadratic_quad",
    "TE4": "tetra",
    "T10": "quadratic_tetra",
    "PY5": "pyramid",
    "P13": "quadratic_pyramid",
    "PE6": "wedge",
    "P15": "quadratic_wedge",
    "P18": "biquadratic_quadratic_wedge",
    "HE8": "hexahedron",
    "H20": "quadratic_hexahedron",
    "H27": "triquadratic_hexahedron",
}
_POLYXIOS_TO_MED: dict[str, str] = {v: k for k, v in _MED_TO_POLYXIOS.items()}
_READ_ORDER: dict[str, tuple[int, ...]] = {
    "H27": (*range(20), 24, 22, 21, 23, 20, 25, 26),
}
_WRITE_ORDER: dict[str, tuple[int, ...]] = {
    name: tuple(order.index(i) for i in range(len(order)))
    for name, order in _READ_ORDER.items()
}
# A pixel and a voxel are a quad and a hexahedron with their corners in
# lattice order; MED names only the latter.
_LATTICE: dict[str, tuple[str, tuple[int, ...]]] = {
    "pixel": ("QU4", (0, 1, 3, 2)),
    "voxel": ("HE8", (0, 1, 3, 2, 4, 5, 7, 6)),
}
# The types the writer can name, by polyxios code, with their node counts.
_WRITE_SIZES: dict[int, int] = {
    int(ELEMENT_TYPES[name]): NODES_PER_ELEMENT[name]
    for name in (*_POLYXIOS_TO_MED, *_LATTICE)
}

# MED's own field value types, as ``TYP`` on a field.
_MED_FLOAT64: int = 6
_MED_INT32: int = 24
_MED_INT64: int = 26
_NO_PROFILE: str = "MED_NO_PROFILE_INTERNAL"
# A mesh with no time step is filed under this NDT/NOR pair by every writer.
_NO_STEP: str = "-0000000000000000001-0000000000000000001"
_FIELD_STEP: str = "0000000000000000000100000000000000000001"
_NAME_WIDTH: int = 80
_DESCRIPTION_WIDTH: int = 200
# Topological dimension per MED geometry, for the GEO code the library
# writes on each cell group and reads back to know what the group holds.
_MED_DIMENSION: dict[str, int] = {
    "PO1": 0,
    "SE2": 1,
    "SE3": 1,
    "SE4": 1,
    "TR3": 2,
    "TR6": 2,
    "TR7": 2,
    "QU4": 2,
    "QU8": 2,
    "QU9": 2,
    "TE4": 3,
    "T10": 3,
    "PY5": 3,
    "P13": 3,
    "PE6": 3,
    "P15": 3,
    "P18": 3,
    "HE8": 3,
    "H20": 3,
    "H27": 3,
}
_COMPONENT_WIDTH: int = 16
_MESH_NAME_KEY: str = "mesh_name"
_TIME_KEY: str = "time"
_RESERVED_GLOBALS: frozenset[str] = frozenset({_MESH_NAME_KEY, "was_2d", _TIME_KEY})
_FAMILY_PREFIX: str = "family_"


# ----- reading ----------------------------------------------------------------


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a MED file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.med`` file, or an open binary file object.
    lazy
        Not supported; raises LazyReadError when True.
    **opts
        ``mesh`` names the mesh to read when the file holds several;
        without it every mesh is merged into one, each mesh's elements
        tagged with the mesh's name. ``step`` picks the time step of the
        fields, counted from zero; the first is read without it. A mesh
        that steps too is read at the same step, or at its last one when
        it has fewer.

    Returns
    -------
    PolyData
        The mesh. MED groups become tag groups; a family that belongs to no
        group but is not family zero becomes ``family_<n>``. Node and cell
        numbers other than ``1..n`` land in ``original_ids``. Fields on the
        nodes are ``vertex_attrs``, fields on the cells ``element_attrs``
        - one value per Gauss point or per node of the cell where the field
        has those - and the mesh's name is ``global_attrs["mesh_name"]``.
        A field step with a time puts it in ``global_attrs["time"]``.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    UnsupportedFormatError
        If h5py is not installed. The message names the extra that installs it.
    CodecError
        If the file is not HDF5, holds no MED mesh, holds a structured mesh,
        names a mesh it does not have, a dataset declares more entities than
        it holds values for, or ``step`` is not a whole number from zero.

    Warns
    -----
    UserWarning
        For each cell geometry MED names and polyxios does not - polygons and
        polyhedra among them - which is skipped; for a family or number
        dataset shorter than its entities, which is ignored; for a field
        whose parts cannot share one array; and for a field holding both
        cell and per-node values on one geometry, where the cell values win.
    """
    name = source_name(path)
    refuse_lazy(lazy, fmt=EXTENSION, name=name)
    wanted = opts.pop("mesh", None)
    step = _step_option(opts.pop("step", 0))
    warn_unknown_opts(opts, fmt=EXTENSION, what="read")
    with open_hdf5_read(path, fmt=EXTENSION) as handle:
        ctx = {"name": name, "file_size": source_size(path), "handle": handle}
        ensemble = child(handle, "ENS_MAA", name=name, where="/")
        names = list(ensemble)
        if not names:
            raise CodecError(f"'{name}': 'ENS_MAA' holds no mesh.")
        if wanted is not None:
            if wanted not in ensemble:
                raise CodecError(
                    f"'{name}': no mesh named {wanted!r}; the file holds {names}."
                )
            names = [wanted]
        meshes = [_read_mesh(ensemble[n], n, step, ctx) for n in names]
    if len(meshes) == 1:
        return meshes[0]
    return _merge_meshes(meshes, names)


def _merge_meshes(meshes: list[PolyData], names: list[str]) -> PolyData:
    """Merge several meshes, tagging each one's elements with its name."""
    tagged = []
    # A mesh's label must not be a group of any mesh, or the merge would
    # pour the mesh into that group.
    taken: set[str] = set().union(*(poly.element_tags for poly in meshes))
    for poly, mesh_name in zip(meshes, names):
        tags = dict(poly.element_tags)
        label = mesh_name
        while label in taken:
            label += "_"
        taken.add(label)
        tags[label] = np.arange(len(poly.element_types), dtype=np.int64)
        globals_ = {k: v for k, v in poly.global_attrs.items() if k != _MESH_NAME_KEY}
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
                global_attrs=globals_,
            )
        )
    return transforms.merge(*tagged)


def _step_group(mesh: Any, step: int, ctx: dict[str, Any], mesh_name: str) -> Any:
    """The group holding ``NOE`` and ``MAI``: the mesh itself, or one of its steps."""
    if "NOE" in mesh or "MAI" in mesh:
        return mesh
    steps = sorted(k for k in mesh if isinstance(mesh[k], type(mesh)))
    if not steps:
        raise CodecError(
            f"'{ctx['name']}': mesh {mesh_name!r} holds neither nodes nor a step."
        )
    # The fields usually step many times over a mesh that steps once, so a
    # step the mesh does not have reads the last one the mesh does.
    return mesh[steps[min(max(step, 0), len(steps) - 1)]]


def _step_option(step: Any) -> int:
    """The ``step`` option as a whole number from zero, or a refusal naming it."""
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
        raise CodecError(
            f"{EXTENSION} read: step={step!r} is not a whole number from zero."
        )
    if step < 0:
        raise CodecError(f"{EXTENSION} read: step={step} is below zero.")
    return int(step)


def _count(node: Any, fallback: int, ctx: dict[str, Any], where: str) -> int:
    """The ``NBR`` a dataset declares, checked against what it holds."""
    declared = node.attrs.get("NBR")
    if declared is None:
        return fallback
    declared = int(np.asarray(declared).ravel()[0])
    if declared < 0 or declared > fallback:
        raise CodecError(
            f"'{ctx['name']}': '{where}' declares NBR={declared} and holds"
            f" values for {fallback}."
        )
    return declared


def _component_major(flat: np.ndarray, width: int, count: int) -> np.ndarray:
    """The first ``count`` rows of a component-major block, ``(count, width)``.

    Every component runs over all the entities the dataset holds before the
    next begins, so the block is shaped by what it holds and cut afterwards;
    cutting the flat values first would mix one entity's components with
    another's whenever ``NBR`` is short of the dataset.
    """
    return flat.reshape(flat.size // width, width, order="F")[:count]


def _short_warning(key: str, held: int, wanted: int, what: str, where: str) -> None:
    warnings.warn(
        f"{EXTENSION} read: '{where}/{key}' holds {held} values for {wanted}"
        f" {what}; ignored.",
        UserWarning,
        stacklevel=6,
    )


def _read_mesh(mesh: Any, mesh_name: str, step: int, ctx: dict[str, Any]) -> PolyData:
    name = ctx["name"]
    where = f"ENS_MAA/{mesh_name}"
    mesh_type = mesh.attrs.get("TYP")
    if mesh_type is not None and int(np.asarray(mesh_type).ravel()[0]) != 0:
        raise CodecError(
            f"'{name}': mesh {mesh_name!r} is a structured (grid) mesh, which"
            " is not read; only unstructured MED meshes are."
        )
    dim = mesh.attrs.get("ESP", mesh.attrs.get("DIM"))
    if dim is None:
        raise CodecError(f"'{name}': '{where}' names no space dimension (ESP).")
    dim = int(np.asarray(dim).ravel()[0])
    if dim not in (1, 2, 3):
        raise CodecError(f"'{name}': '{where}' has space dimension {dim}.")
    group = _step_group(mesh, step, ctx, mesh_name)

    vertices, vertex_extra = _read_nodes(group, dim, ctx, where)
    n_verts = len(vertices)
    node_families = vertex_extra.pop("families", None)

    blocks, element_families, element_ids = _read_cells(group, n_verts, ctx, where)
    element_groups = [(ptype, cells) for ptype, cells in blocks]

    families = _read_families(ctx["handle"], mesh_name, group)
    vertex_tags = _tags_from_families(node_families, families["node"], n_verts)
    n_elems = sum(len(cells) for _, cells in element_groups)
    element_tags = _tags_from_families(element_families, families["cell"], n_elems)

    vertex_attrs: dict[str, np.ndarray] = dict(vertex_extra)
    element_attrs: dict[str, np.ndarray] = dict(record_ids(element_ids, count=n_elems))
    global_attrs: dict[str, Any] = {_MESH_NAME_KEY: mesh_name, **mark_2d(dim)}
    if "CHA" in ctx["handle"]:
        _read_fields(
            ctx["handle"]["CHA"],
            ctx,
            mesh_name,
            step,
            [ptype for ptype, _ in blocks],
            [len(cells) for _, cells in blocks],
            n_verts,
            vertex_attrs,
            element_attrs,
            global_attrs,
        )
    return make_polydata(
        vertices,
        element_groups,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _read_nodes(
    group: Any, dim: int, ctx: dict[str, Any], where: str
) -> tuple[np.ndarray, dict[str, Any]]:
    name = ctx["name"]
    nodes = child(group, "NOE", name=name, where=where)
    coo = as_array(
        child(nodes, "COO", name=name, where=f"{where}/NOE"),
        name=name,
        where=f"{where}/NOE/COO",
    )
    if coo.dtype.kind not in "iuf":
        raise CodecError(f"'{name}': '{where}/NOE/COO' holds {coo.dtype}, not numbers.")
    flat = coo.ravel()
    if flat.size % dim:
        raise CodecError(
            f"'{name}': '{where}/NOE/COO' holds {flat.size} values, not a"
            f" multiple of the {dim} components."
        )
    n_verts = _count(nodes["COO"], flat.size // dim, ctx, f"{where}/NOE/COO")
    validate_header(n_verts, 0, 0, ctx["file_size"], compressed=True)
    coords = _component_major(flat, dim, n_verts)
    vertices = (
        pad_to_3d(coords, dim)
        if dim >= 2
        else np.column_stack([coords[:, 0], np.zeros(n_verts), np.zeros(n_verts)])
    )
    extra: dict[str, Any] = {}
    fam = _per_entity(nodes, "FAM", n_verts, name, f"{where}/NOE", "nodes")
    if fam is not None:
        extra["families"] = fam
    num = _per_entity(nodes, "NUM", n_verts, name, f"{where}/NOE", "nodes")
    if num is not None:
        extra.update(record_ids(num, count=n_verts))
    return vertices, extra


def _read_cells(
    group: Any, n_verts: int, ctx: dict[str, Any], where: str
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray | None, np.ndarray | None]:
    """Every cell block the mesh holds, by ascending polyxios type code."""
    name = ctx["name"]
    if "MAI" not in group:
        return [], None, None
    cells_group = group["MAI"]
    unknown: list[str] = []
    found: list[tuple[int, str, Any]] = []
    for med_type in cells_group:
        ptype = _MED_TO_POLYXIOS.get(med_type)
        if ptype is None:
            unknown.append(med_type)
            continue
        found.append((int(ELEMENT_TYPES[ptype]), med_type, cells_group[med_type]))
    if unknown:
        warnings.warn(
            f"{EXTENSION} read: cell geometry {sorted(unknown)} in '{name}' has"
            " no polyxios element type; those cells are skipped.",
            UserWarning,
            stacklevel=4,
        )
    found.sort(key=lambda item: item[0])
    blocks: list[tuple[str, np.ndarray]] = []
    families: list[np.ndarray | None] = []
    ids: list[np.ndarray | None] = []
    for code, med_type, node in found:
        ptype = ELEMENT_TYPES_INV[code]
        width = NODES_PER_ELEMENT[ptype]
        here = f"{where}/MAI/{med_type}"
        nod = as_array(
            child(node, "NOD", name=name, where=here), name=name, where=f"{here}/NOD"
        )
        if nod.dtype.kind not in "iu":
            raise CodecError(f"'{name}': '{here}/NOD' holds {nod.dtype}, not indices.")
        flat = nod.ravel()
        if flat.size % width:
            raise CodecError(
                f"'{name}': '{here}/NOD' holds {flat.size} indices, not a"
                f" multiple of the {width} nodes of a {med_type}."
            )
        n_cells = _count(node["NOD"], flat.size // width, ctx, f"{here}/NOD")
        validate_header(
            n_verts, n_cells, n_cells * width, ctx["file_size"], compressed=True
        )
        cells = _component_major(flat, width, n_cells).astype(np.int64) - 1
        if n_cells and (cells.min() < 0 or cells.max() >= n_verts):
            raise CodecError(
                f"'{name}': '{here}/NOD' names a node outside 1..{n_verts}."
            )
        order = _READ_ORDER.get(med_type)
        if order is not None:
            cells = cells[:, list(order)]
        blocks.append((ptype, cells))
        families.append(_per_entity(node, "FAM", n_cells, name, here, "cells"))
        ids.append(_per_entity(node, "NUM", n_cells, name, here, "cells"))
    all_families = None
    if any(f is not None for f in families):
        all_families = np.concatenate(
            [
                f if f is not None else np.zeros(len(c), dtype=np.int64)
                for f, (_, c) in zip(families, blocks)
            ]
        )
    all_ids: np.ndarray | None = None
    if ids and all(i is not None for i in ids):
        all_ids = np.concatenate([i for i in ids if i is not None])
    return blocks, all_families, all_ids


def _per_entity(
    node: Any, key: str, count: int, name: str, where: str, what: str
) -> np.ndarray | None:
    """A ``FAM`` or ``NUM`` dataset as one int64 per entity, or None with a warning."""
    if key not in node:
        return None
    values = as_array(node[key], name=name, where=f"{where}/{key}").ravel()
    if values.dtype.kind not in "iu":
        warnings.warn(
            f"{EXTENSION} read: '{where}/{key}' holds {values.dtype}, not"
            " whole numbers; ignored.",
            UserWarning,
            stacklevel=5,
        )
        return None
    if values.size < count:
        _short_warning(key, values.size, count, what, where)
        return None
    return values[:count].astype(np.int64)


def _read_families(
    handle: Any, mesh_name: str, group: Any
) -> dict[str, dict[int, list[str]]]:
    """Family number to group names, for the nodes and for the cells.

    MED 3 files a family under ``FAS/<mesh>/NOEUD`` or ``FAS/<mesh>/ELEME``;
    MED 2.3 files them all directly under ``FAS/<mesh>``, where the sign of
    the number says which they are - positive for nodes, negative for
    cells. Family zero belongs to both and names no group.
    """
    out: dict[str, dict[int, list[str]]] = {"node": {}, "cell": {}}
    fas = None
    if "FAS" in handle and mesh_name in handle["FAS"]:
        fas = handle["FAS"][mesh_name]
    elif "FAS" in group:
        fas = group["FAS"]
    if fas is None:
        return out
    for key in fas:
        node = fas[key]
        if not hasattr(node, "attrs") or hasattr(node, "shape"):
            continue
        if key == "NOEUD":
            _collect_families(node, out["node"], out["cell"], kind="node")
        elif key == "ELEME":
            _collect_families(node, out["node"], out["cell"], kind="cell")
        else:
            _one_family(key, node, out["node"], out["cell"], kind=None)
    return out


def _collect_families(
    parent: Any, nodes: dict[int, list[str]], cells: dict[int, list[str]], *, kind: str
) -> None:
    for key in parent:
        _one_family(key, parent[key], nodes, cells, kind=kind)


def _one_family(
    key: str,
    node: Any,
    nodes: dict[int, list[str]],
    cells: dict[int, list[str]],
    *,
    kind: str | None,
) -> None:
    if not hasattr(node, "attrs") or hasattr(node, "shape") or "NUM" not in node.attrs:
        return
    number = int(np.asarray(node.attrs["NUM"]).ravel()[0])
    groups = _group_names(node)
    if kind is None:
        kind = "cell" if number < 0 else "node"
    if number == 0:
        return
    target = nodes if kind == "node" else cells
    target[number] = groups


def _group_names(family: Any) -> list[str]:
    """The names of the groups a family belongs to, as ``GRO/NOM`` spells them."""
    if "GRO" not in family or "NOM" not in family["GRO"]:
        return []
    raw = np.asarray(family["GRO"]["NOM"][()])
    if raw.dtype.kind == "V" and raw.dtype.names:
        raw = raw[raw.dtype.names[0]]
    names: list[str] = []
    if raw.dtype.kind in "iu":
        rows = raw.reshape(-1, raw.shape[-1]) if raw.ndim > 1 else raw.reshape(1, -1)
        for row in rows:
            text = bytes(row.astype(np.uint8)).decode("utf-8", errors="replace")
            names.append(text.rstrip("\x00").strip())
    else:
        for value in np.atleast_1d(raw).ravel():
            decoded = text_of(value)
            if decoded is not None:
                names.append(decoded)
    declared = family["GRO"].attrs.get("NBR")
    if declared is not None:
        names = names[: max(0, int(np.asarray(declared).ravel()[0]))]
    return [n for n in names if n]


def _tags_from_families(
    per_entity: np.ndarray | None, families: dict[int, list[str]], n_items: int
) -> dict[str, np.ndarray]:
    """One tag group per MED group; a group-less family keeps its number."""
    if per_entity is None or per_entity.size == 0:
        return {}
    tags: dict[str, list[np.ndarray]] = {}
    for number in np.unique(per_entity):
        number = int(number)
        if number == 0:
            continue
        members = np.flatnonzero(per_entity == number).astype(np.int64)
        names = families.get(number)
        if not names:
            names = [f"{_FAMILY_PREFIX}{number}"]
        for group_name in names:
            tags.setdefault(group_name, []).append(members)
    return {
        group_name: np.sort(np.concatenate(parts)) if len(parts) > 1 else parts[0]
        for group_name, parts in tags.items()
    }


# ----- fields -----------------------------------------------------------------


def _read_fields(
    fields: Any,
    ctx: dict[str, Any],
    mesh_name: str,
    step: int,
    block_types: list[str],
    block_sizes: list[int],
    n_verts: int,
    vertex_attrs: dict[str, np.ndarray],
    element_attrs: dict[str, np.ndarray],
    global_attrs: dict[str, Any],
) -> None:
    name = ctx["name"]
    profiles = ctx["handle"].get("PROFILS")
    n_elems = sum(block_sizes)
    starts = np.cumsum([0, *block_sizes])
    for field_name in fields:
        field = fields[field_name]
        if hasattr(field, "shape"):
            continue
        on_mesh = attr_text(field, "MAI")
        if on_mesh is not None and on_mesh != mesh_name:
            continue
        steps = sorted(k for k in field if not hasattr(field[k], "shape"))
        if not steps:
            continue
        if step >= len(steps):
            warnings.warn(
                f"{EXTENSION} read: field {field_name!r} in '{name}' has"
                f" {len(steps)} step(s), none numbered {step}; skipped.",
                UserWarning,
                stacklevel=5,
            )
            continue
        at = field[steps[step]]
        n_comp = max(int(np.asarray(field.attrs.get("NCO", 1)).ravel()[0]), 1)
        time = at.attrs.get("PDT")
        if time is not None and _TIME_KEY not in global_attrs:
            time = float(np.asarray(time).ravel()[0])
            if time != -1.0:
                global_attrs[_TIME_KEY] = time
        parts: dict[str, np.ndarray] = {}
        # Sorted, so a geometry's cell values (MAI.*) come before its per-node
        # values (NOE.*) and are the ones kept when a field holds both.
        for support in sorted(at):
            node = at[support]
            if hasattr(node, "shape"):
                continue
            here = f"CHA/{field_name}/{steps[step]}/{support}"
            if support == "NOE":
                values = _support_values(node, profiles, ctx, here, n_verts, n_comp)
                if values is None:
                    continue
                if len(values) != n_verts:
                    warnings.warn(
                        f"{EXTENSION} read: field {field_name!r} holds"
                        f" {len(values)} node values for {n_verts} nodes; skipped.",
                        UserWarning,
                        stacklevel=5,
                    )
                    continue
                vertex_attrs[field_name] = _squeezed(values)
                continue
            kind, _, med_type = support.partition(".")
            ptype = _MED_TO_POLYXIOS.get(med_type)
            if ptype is None or ptype not in block_types:
                continue
            index = block_types.index(ptype)
            values = _support_values(
                node, profiles, ctx, here, block_sizes[index], n_comp
            )
            if values is None:
                continue
            if len(values) != block_sizes[index]:
                warnings.warn(
                    f"{EXTENSION} read: field {field_name!r} holds {len(values)}"
                    f" values for {block_sizes[index]} {med_type} cells; skipped.",
                    UserWarning,
                    stacklevel=5,
                )
                continue
            if ptype in parts:
                warnings.warn(
                    f"{EXTENSION} read: field {field_name!r} holds both cell and"
                    f" per-node values on {med_type}; the per-node values are"
                    " skipped.",
                    UserWarning,
                    stacklevel=5,
                )
                continue
            order = _READ_ORDER.get(med_type)
            if (
                kind == "NOE"
                and order is not None
                and values.shape[1] == NODES_PER_ELEMENT[ptype]
            ):
                values = values[:, list(order), :]
            parts[ptype] = values
        if parts:
            _place_cell_field(
                field_name, parts, block_types, starts, n_elems, element_attrs, name
            )


def _support_values(
    node: Any,
    profiles: Any,
    ctx: dict[str, Any],
    where: str,
    count: int,
    n_comp_hint: int = 1,
) -> np.ndarray | None:
    """The values a field holds on one support, ``(n, n_gauss, n_comp)``.

    ``count`` is how many entities the support has in the mesh; a profiled
    field is spread over that many, NaN (or zero, for an integer field)
    where the profile does not reach. ``n_comp_hint`` is the field's
    declared component count, used only when the support holds no entity.
    """
    name = ctx["name"]
    profile = attr_text(node, "PFL") or _NO_PROFILE
    if profile not in node:
        return None
    block = node[profile]
    if "CO" not in block:
        return None
    raw = as_array(block["CO"], name=name, where=f"{where}/{profile}/CO")
    if raw.dtype.kind not in "biuf":
        return None
    n_gauss = int(np.asarray(block.attrs.get("NGA", 1)).ravel()[0]) or 1
    n_entities = block.attrs.get("NBR")
    flat = raw.ravel()
    if n_entities is None:
        return None
    n_entities = int(np.asarray(n_entities).ravel()[0])
    per = n_entities * n_gauss
    if (per == 0 and flat.size) or (per and flat.size % per):
        raise CodecError(
            f"'{name}': '{where}/{profile}/CO' holds {flat.size} values for"
            f" {n_entities} entities of {n_gauss} point(s) each."
        )
    # A field over no entity holds no value to count components from; the
    # field's own NCO says how many it would have had.
    n_comp = flat.size // per if per else n_comp_hint
    values = flat.reshape(n_entities, n_gauss, n_comp, order="F")
    if profile != _NO_PROFILE:
        if (
            profiles is None
            or profile not in profiles
            or "PFL" not in profiles[profile]
        ):
            raise CodecError(
                f"'{name}': field profile {profile!r} is not under 'PROFILS'."
            )
        index = (
            as_array(
                profiles[profile]["PFL"], name=name, where=f"PROFILS/{profile}/PFL"
            )
            .ravel()
            .astype(np.int64)
            - 1
        )
        if index.size != n_entities or (
            index.size and (index.min() < 0 or index.max() >= count)
        ):
            raise CodecError(
                f"'{name}': field profile {profile!r} does not index its"
                f" support of {count}."
            )
        if values.dtype.kind == "f":
            spread = np.full((count, n_gauss, n_comp), np.nan, dtype=values.dtype)
        else:
            spread = np.zeros((count, n_gauss, n_comp), dtype=values.dtype)
        spread[index] = values
        values = spread
    return values


def _squeezed(values: np.ndarray) -> np.ndarray:
    """Drop the Gauss and component axes a field does not use.

    ``(n, 1, 1)`` is a scalar per entity, ``(n, 1, c)`` a vector, ``(n, g,
    1)`` one value per Gauss point or per node of the cell, and ``(n, g, c)``
    stays as it is.
    """
    if values.shape[2] == 1:
        values = values[:, :, 0]
        if values.shape[1] == 1:
            values = values[:, 0]
    elif values.shape[1] == 1:
        values = values[:, 0, :]
    return np.ascontiguousarray(values)


def _place_cell_field(
    field_name: str,
    parts: dict[str, np.ndarray],
    block_types: list[str],
    starts: np.ndarray,
    n_elems: int,
    element_attrs: dict[str, np.ndarray],
    name: str,
) -> None:
    """Lay the per-type parts of a cell field over the whole element range."""
    shapes = {v.shape[1:] for v in parts.values()}
    if len(shapes) != 1:
        warnings.warn(
            f"{EXTENSION} read: field {field_name!r} in '{name}' has a different"
            f" shape on each cell type ({sorted(shapes)}); skipped.",
            UserWarning,
            stacklevel=6,
        )
        return
    trailing = shapes.pop()
    dtypes = {v.dtype for v in parts.values()}
    dtype = np.result_type(*dtypes)
    fill = np.nan if dtype.kind == "f" else 0
    if dtype.kind != "f" and len(parts) != len(block_types):
        # An integer field over some of the types has no value for the rest,
        # and 0 would be a value; a float column can say "none".
        dtype = np.dtype(np.float64)
        fill = np.nan
    out = np.full((n_elems, *trailing), fill, dtype=dtype)
    for ptype, values in parts.items():
        index = block_types.index(ptype)
        out[starts[index] : starts[index + 1]] = values
    element_attrs[field_name] = _squeezed(out)


# ----- writing ----------------------------------------------------------------


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as a MED file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object.
    **opts
        ``mesh_name`` names the mesh inside the file; without it
        ``global_attrs["mesh_name"]`` is used, and ``"mesh"`` failing that.
        ``time`` is the time the fields are written at; without it a number
        under ``global_attrs["time"]`` is, and failing that the fields
        carry no time step. ``compression`` and ``compression_opts`` are
        passed to h5py for the datasets - ``compression="gzip"`` with
        ``compression_opts=4`` is the usual pair; the MED library reads a
        filtered dataset as it reads any other.

    Raises
    ------
    UnsupportedFormatError
        If h5py is not installed.
    CodecError
        If the mesh's vertices are not a coordinate block, or the connectivity
        indexes past what the file's int32 fields can hold.

    Warns
    -----
    UserWarning
        Once per element type MED has no name for - those elements, and the
        values and tags on them, are not written; once for the attributes a
        MED field cannot carry (text, or a third axis that is neither the
        cell's node count nor a Gauss point count); once for the element
        attributes that are NaN on every element, which a MED field would
        turn into a value; once for ``global_attrs`` other than
        ``mesh_name``, which MED has no place for.

    Notes
    -----
    The file is MED 3.0, which every Salome since 7 and every code_aster
    since 11 reads. Tag groups become MED groups: each distinct combination
    of groups an entity belongs to is a family - numbered upward for nodes,
    downward for cells, as MED expects - and the family belongs to every
    group in the combination. ``vertex_attrs`` are ``NOEU`` fields,
    ``element_attrs`` are ``MAI.<TYPE>`` fields, one part per cell type, with
    a third axis written as Gauss points (``NGA``) when it is not the cell's
    node count and as ``ELNO`` values when it is. A pixel or a voxel is
    written as the quad or hexahedron it is, corners reordered.
    """
    name = source_name(path)
    mesh_name = opts.pop("mesh_name", None)
    time = opts.pop("time", None)
    dataset_kw = dataset_options(opts, fmt=EXTENSION)
    warn_unknown_opts(opts, fmt=EXTENSION, what="write")
    if time is None:
        time = poly.global_attrs.get(_TIME_KEY)
    if time is not None:
        try:
            time = float(time)
        except (TypeError, ValueError):
            raise CodecError(
                f"{EXTENSION} write: time={time!r} is not a number."
            ) from None
    if mesh_name is None:
        mesh_name = text_of(poly.global_attrs.get(_MESH_NAME_KEY)) or "mesh"
    mesh_name = _link_name(str(mesh_name)[:64], "mesh")
    dropped_globals = sorted(
        (k for k in (poly.global_attrs or {}) if k not in _RESERVED_GLOBALS),
        key=repr,
    )
    if dropped_globals:
        warnings.warn(
            f"{EXTENSION} write: global_attrs {dropped_globals} have no place in"
            " a MED file; dropped.",
            UserWarning,
            stacklevel=2,
        )
    dim = output_dimension(poly, fmt=EXTENSION)
    n_verts = len(poly.vertices)
    n_elems = len(poly.element_types)
    mesh_dim = min(dim, int(poly.topological_dimension)) if n_elems else dim
    blocks, dropped = type_blocks(poly, sizes=_WRITE_SIZES, fmt=EXTENSION)
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: element type(s)"
            f" {sorted(ELEMENT_TYPES_INV.get(c, f'type_{c}') for c in dropped)}"
            " have no MED geometry; those elements and the values and tags on"
            " them are not written.",
            UserWarning,
            stacklevel=2,
        )
    kept = kept_index(blocks)
    if n_verts >= 2**31 - 1:
        raise CodecError(
            f"'{name}': {n_verts} nodes do not fit MED's int32 connectivity."
        )
    node_ids = ids_for_write(
        poly, kind="vertex", count=n_verts, fmt=EXTENSION, limit=2**31 - 1
    )
    cell_ids = ids_for_write(
        poly, kind="element", count=n_elems, fmt=EXTENSION, limit=2**31 - 1
    )[kept]

    node_fam, node_families = _families(poly.vertex_tags, n_verts, sign=1)
    element_tags = reindexed_tags(poly.element_tags, kept, n_elems)
    cell_fam, cell_families = _families(element_tags, len(kept), sign=-1)

    def store(group: Any, key: str, arr: np.ndarray) -> Any:
        return group.create_dataset(
            key, data=np.ascontiguousarray(arr), **(dataset_kw if arr.size > 1 else {})
        )

    with open_hdf5_write(path, fmt=EXTENSION) as handle:
        info = handle.create_group("INFOS_GENERALES")
        info.attrs.create("MAJ", 3, dtype=np.int32)
        info.attrs.create("MIN", 0, dtype=np.int32)
        info.attrs.create("REL", 0, dtype=np.int32)

        # Every one of these nine is read by the MED library's mesh-info
        # call, and a missing one refuses the whole file in Salome and Gmsh.
        mesh = handle.create_group("ENS_MAA").create_group(mesh_name)
        mesh.attrs.create("DIM", mesh_dim, dtype=np.int32)
        mesh.attrs.create("ESP", dim, dtype=np.int32)
        mesh.attrs.create("REP", 0, dtype=np.int32)
        mesh.attrs.create("UNT", _padded([""], _COMPONENT_WIDTH))
        mesh.attrs.create("UNI", _padded([""] * dim, _COMPONENT_WIDTH))
        mesh.attrs.create("SRT", 1, dtype=np.int32)
        mesh.attrs.create("NOM", _padded(["X", "Y", "Z"][:dim], _COMPONENT_WIDTH))
        mesh.attrs.create(
            "DES", _padded(["Mesh written by polyxios"], _DESCRIPTION_WIDTH)
        )
        mesh.attrs.create("TYP", 0, dtype=np.int32)
        mesh.attrs.create("NXI", -1, dtype=np.int32)
        mesh.attrs.create("NXT", -1, dtype=np.int32)
        step = mesh.create_group(_NO_STEP)
        step.attrs.create("CGT", 1, dtype=np.int32)
        step.attrs.create("NDT", -1, dtype=np.int32)
        step.attrs.create("NOR", -1, dtype=np.int32)
        step.attrs.create("PDT", -1.0, dtype=np.float64)

        nodes = step.create_group("NOE")
        nodes.attrs.create("CGT", 1, dtype=np.int32)
        nodes.attrs.create("CGS", 1, dtype=np.int32)
        nodes.attrs.create("PFL", np.bytes_(_NO_PROFILE))
        coords = np.asarray(poly.vertices, dtype=np.float64)[:, :dim]
        coo = store(nodes, "COO", coords.ravel(order="F"))
        coo.attrs.create("CGT", 1, dtype=np.int32)
        coo.attrs.create("NBR", n_verts, dtype=np.int32)
        _counted(nodes, "FAM", node_fam.astype(np.int32), n_verts, store)
        if not _dense(node_ids):
            _counted(nodes, "NUM", node_ids.astype(np.int32), n_verts, store)

        cells = step.create_group("MAI")
        cells.attrs.create("CGT", 1, dtype=np.int32)
        offset = 0
        for code, index, conn in blocks:
            ptype = ELEMENT_TYPES_INV[code]
            med_type, order = _med_spelling(ptype)
            if order is not None:
                conn = conn[:, list(order)]
            n_cells = len(index)
            block = cells.create_group(med_type)
            block.attrs.create("CGT", 1, dtype=np.int32)
            block.attrs.create("CGS", 1, dtype=np.int32)
            block.attrs.create("GEO", _geo_code(med_type), dtype=np.int32)
            block.attrs.create("PFL", np.bytes_(_NO_PROFILE))
            nod = store(block, "NOD", (conn + 1).ravel(order="F").astype(np.int32))
            nod.attrs.create("CGT", 1, dtype=np.int32)
            nod.attrs.create("NBR", n_cells, dtype=np.int32)
            _counted(
                block,
                "FAM",
                cell_fam[offset : offset + n_cells].astype(np.int32),
                n_cells,
                store,
            )
            if not _dense(cell_ids):
                _counted(
                    block,
                    "NUM",
                    cell_ids[offset : offset + n_cells].astype(np.int32),
                    n_cells,
                    store,
                )
            offset += n_cells

        fas = handle.create_group("FAS").create_group(mesh_name)
        zero = fas.create_group("FAMILLE_ZERO")
        zero.attrs.create("NUM", 0, dtype=np.int32)
        if node_families:
            _write_families(fas.create_group("NOEUD"), node_families)
        if cell_families:
            _write_families(fas.create_group("ELEME"), cell_families)

        fields = handle.create_group("CHA")
        renamed: dict[str, str] = {}
        all_nan: list[str] = []
        for attr_name, values in poly.vertex_attrs.items():
            if attr_name == IDS_KEY:
                continue
            _write_node_field(
                fields, mesh_name, attr_name, values, n_verts, time, renamed, store
            )
        for attr_name, values in poly.element_attrs.items():
            if attr_name == IDS_KEY:
                continue
            if not _write_cell_field(
                fields,
                mesh_name,
                attr_name,
                values,
                blocks,
                n_elems,
                time,
                renamed,
                store,
            ):
                all_nan.append(str(attr_name))
        if all_nan:
            warnings.warn(
                f"{EXTENSION} write: element attribute(s) {sorted(all_nan)} are"
                " NaN on every element, which a MED field would turn into a"
                " value; not written.",
                UserWarning,
                stacklevel=2,
            )
        if renamed:
            warnings.warn(
                f"{EXTENSION} write: attribute(s) {sorted(renamed)} are written"
                f" under {[renamed[k] for k in sorted(renamed)]}; a MED field's"
                " name is its HDF5 link, which cannot hold '/' or repeat.",
                UserWarning,
                stacklevel=2,
            )


def _geo_code(med_type: str) -> int:
    """MED's numeric geometry code: the dimension, then the node count."""
    return (
        _MED_DIMENSION[med_type] * 100 + NODES_PER_ELEMENT[_MED_TO_POLYXIOS[med_type]]
    )


def _med_spelling(ptype: str) -> tuple[str, tuple[int, ...] | None]:
    if ptype in _LATTICE:
        return _LATTICE[ptype]
    med_type = _POLYXIOS_TO_MED[ptype]
    return med_type, _WRITE_ORDER.get(med_type)


def _dense(ids: np.ndarray) -> bool:
    return bool(np.array_equal(ids, np.arange(1, len(ids) + 1)))


def _counted(group: Any, key: str, values: np.ndarray, count: int, store: Any) -> None:
    node = store(group, key, values)
    node.attrs.create("CGT", 1, dtype=np.int32)
    node.attrs.create("NBR", count, dtype=np.int32)


def _padded(names: list[str], width: int) -> np.bytes_:
    return np.bytes_("".join(f"{n[:width]:<{width}}" for n in names))


def _families(
    tags: dict[str, np.ndarray] | None, n_items: int, *, sign: int
) -> tuple[np.ndarray, dict[int, list[str]]]:
    """One family per distinct combination of groups; number and names of each."""
    per_item = np.zeros(n_items, dtype=np.int64)
    if not tags or n_items == 0:
        return per_item, {}
    names = [str(n) for n in tags]
    membership = np.zeros((n_items, len(names)), dtype=bool)
    for column, members in enumerate(tags.values()):
        membership[member_indices(members, n_items), column] = True
    combos, inverse = np.unique(membership, axis=0, return_inverse=True)
    families: dict[int, list[str]] = {}
    number = 0
    numbers = np.zeros(len(combos), dtype=np.int64)
    for row, combo in enumerate(combos):
        if not combo.any():
            continue
        number += 1
        numbers[row] = sign * number
        families[sign * number] = [names[c] for c in np.flatnonzero(combo)]
    per_item = numbers[np.asarray(inverse).ravel()]
    return per_item, families


def _link_name(text: str, fallback: str) -> str:
    """``text`` as an HDF5 link: no ``/``, and not the ``.`` or ``..`` HDF5 reserves."""
    link = text.replace("/", "_")
    return fallback if link in ("", ".", "..") else link


def _field_link(fields: Any, attr_name: object, renamed: dict[str, str]) -> str:
    """A link no field holds yet, ``renamed`` recording a name that had to change."""
    wanted = str(attr_name)
    link = base = _link_name(wanted, "field")
    n = 1
    while link in fields:
        n += 1
        link = f"{base}_{n}"
    if link != wanted:
        renamed[wanted] = link
    return link


def _write_families(parent: Any, families: dict[int, list[str]]) -> None:
    for number, names in families.items():
        label = _link_name(f"FAM_{number}_{'_'.join(names)}"[:_NAME_WIDTH], "FAM")
        family = parent.create_group(label)
        family.attrs.create("NUM", number, dtype=np.int32)
        group = family.create_group("GRO")
        group.attrs.create("NBR", len(names), dtype=np.int32)
        rows = np.zeros((len(names), _NAME_WIDTH), dtype=np.int8)
        for i, group_name in enumerate(names):
            raw = group_name.encode("utf-8")[:_NAME_WIDTH]
            rows[i, : len(raw)] = np.frombuffer(raw, dtype=np.int8)
        dataset = group.create_dataset("NOM", (len(names),), dtype=f"{_NAME_WIDTH}int8")
        dataset[...] = rows


def _field_values(values: object, attr_name: object, count: int) -> np.ndarray | None:
    """An attribute as ``(n, n_gauss, n_comp)`` MED can hold, or None with a warning."""
    arr = np.asarray(values)
    if arr.dtype.kind == "b":
        arr = arr.astype(np.int32)
    if (
        arr.dtype.kind not in "iuf"
        or arr.ndim == 0
        or arr.shape[0] != count
        or arr.ndim > 3
    ):
        warnings.warn(
            f"{EXTENSION} write: attribute {attr_name!r} is not a numeric array"
            f" of one value, vector or matrix per entity ({arr.dtype}, shape"
            f" {arr.shape}); dropped.",
            UserWarning,
            stacklevel=4,
        )
        return None
    if arr.ndim == 1:
        arr = arr[:, None, None]
    elif arr.ndim == 2:
        arr = arr[:, None, :]
    return arr


def _field_type(arr: np.ndarray) -> tuple[int, np.dtype]:
    if arr.dtype.kind == "f":
        return _MED_FLOAT64, np.dtype(np.float64)
    if arr.dtype.itemsize > 4 or (arr.dtype.kind == "u" and arr.dtype.itemsize == 4):
        return _MED_INT64, np.dtype(np.int64)
    return _MED_INT32, np.dtype(np.int32)


def _field_group(
    fields: Any,
    mesh_name: str,
    attr_name: object,
    arr: np.ndarray,
    time: float | None,
    renamed: dict[str, str],
    *,
    join: bool,
) -> Any:
    """The step group a field's supports go under.

    A new field, or - when ``join`` allows it - the one already filed under
    the name with the same component count and value type, so a vertex
    attribute and an element attribute sharing a name become one MED field
    over the nodes and the cells, which is how MED spells such a field.
    """
    n_comp = arr.shape[2]
    typ, dtype = _field_type(arr)
    link = _link_name(str(attr_name), "field")
    if join and link in fields:
        existing = fields[link]
        if int(existing.attrs["NCO"]) == n_comp and int(existing.attrs["TYP"]) == typ:
            return existing[_NO_STEP if time is None else _FIELD_STEP]
    field = fields.create_group(_field_link(fields, attr_name, renamed))
    field.attrs.create("MAI", np.bytes_(mesh_name))
    field.attrs.create("TYP", typ, dtype=np.int32)
    field.attrs.create("UNT", np.bytes_(""))
    field.attrs.create("NCO", n_comp, dtype=np.int32)
    field.attrs.create(
        "NOM", _padded([f"V{i + 1}" for i in range(n_comp)], _COMPONENT_WIDTH)
    )
    field.attrs.create("UNI", _padded([""] * n_comp, _COMPONENT_WIDTH))
    if time is None:
        step = field.create_group(_NO_STEP)
        step.attrs.create("NDT", -1, dtype=np.int32)
        step.attrs.create("NOR", -1, dtype=np.int32)
        step.attrs.create("PDT", -1.0, dtype=np.float64)
    else:
        step = field.create_group(_FIELD_STEP)
        step.attrs.create("NDT", 1, dtype=np.int32)
        step.attrs.create("NOR", 1, dtype=np.int32)
        step.attrs.create("PDT", time, dtype=np.float64)
    step.attrs.create("RDT", -1, dtype=np.int32)
    step.attrs.create("ROR", -1, dtype=np.int32)
    return step


def _write_support(step: Any, support: str, arr: np.ndarray, store: Any) -> None:
    n, n_gauss, _ = arr.shape
    _, dtype = _field_type(arr)
    node = step.create_group(support)
    node.attrs.create("GAU", np.bytes_(""))
    node.attrs.create("PFL", np.bytes_(_NO_PROFILE))
    profile = node.create_group(_NO_PROFILE)
    profile.attrs.create("NBR", n, dtype=np.int32)
    profile.attrs.create("NGA", n_gauss, dtype=np.int32)
    profile.attrs.create("GAU", np.bytes_(""))
    store(profile, "CO", arr.astype(dtype).ravel(order="F"))


def _write_node_field(
    fields: Any,
    mesh_name: str,
    attr_name: object,
    values: object,
    n_verts: int,
    time: float | None,
    renamed: dict[str, str],
    store: Any,
) -> None:
    arr = _field_values(values, attr_name, n_verts)
    if arr is None:
        return
    if arr.shape[1] != 1:
        warnings.warn(
            f"{EXTENSION} write: vertex attribute {attr_name!r} has a third axis,"
            " which a node field cannot carry; dropped.",
            UserWarning,
            stacklevel=3,
        )
        return
    step = _field_group(fields, mesh_name, attr_name, arr, time, renamed, join=False)
    _write_support(step, "NOE", arr, store)


def _write_cell_field(
    fields: Any,
    mesh_name: str,
    attr_name: object,
    values: object,
    blocks: list[tuple[int, np.ndarray, np.ndarray]],
    n_elems: int,
    time: float | None,
    renamed: dict[str, str],
    store: Any,
) -> bool:
    """Write one element attribute; False when NaN on every element kept it out."""
    arr = _field_values(values, attr_name, n_elems)
    if arr is None or not blocks:
        return True
    if arr.dtype.kind == "f" and arr.shape[1] == 1 and np.isnan(arr).all():
        return False
    step = None
    for code, index, _ in blocks:
        part = arr[index]
        if part.dtype.kind == "f" and part.shape[1] == 1 and np.isnan(part).all():
            # A field read off a MED file that never had this type: NaN was
            # its "no value", and writing NaN back would give it one.
            continue
        ptype = ELEMENT_TYPES_INV[code]
        med_type, order = _med_spelling(ptype)
        if part.shape[1] == 1:
            support = f"MAI.{med_type}"
        elif part.shape[1] == NODES_PER_ELEMENT[ptype]:
            # A third axis the size of the cell's node count is one value per
            # node of each cell, which MED calls ELNO and files under NOE.
            support = f"NOE.{med_type}"
            if order is not None:
                part = part[:, list(order), :]
        else:
            # Gauss-point values need a localisation MED's readers look up
            # under GAUSS, which nothing in the mesh describes.
            warnings.warn(
                f"{EXTENSION} write: element attribute {attr_name!r} holds"
                f" {part.shape[1]} values per {ptype}, neither one nor its"
                " node count; that part is dropped.",
                UserWarning,
                stacklevel=3,
            )
            continue
        if step is None:
            step = _field_group(
                fields, mesh_name, attr_name, arr, time, renamed, join=True
            )
        _write_support(step, support, part, store)
    return True
