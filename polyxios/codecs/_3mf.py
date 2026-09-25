"""3MF (3D Manufacturing Format) codec - read + write.

A ``.3mf`` file is a ZIP package holding an XML model part. Every ``<object>``
placed by the ``<build>`` is read - a ``<mesh>`` outright, a ``<components>``
assembly through its parts, each ``transform`` composed on the way down - and
comes back as a group of triangles tagged with the object's name. A material
a triangle names is read as its colour; on write the tag groups become the
objects and the colours the materials.
"""

from __future__ import annotations

import io
import re
from typing import Any
import urllib.parse
import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    QUADRATIC_SURFACE_CORNERS,
)
from polyxios._io import Source, read_bytes, source_name, write_bytes
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".3mf"

_MODEL_PART: str = "3D/3dmodel.model"
_RELS_PART: str = "_rels/.rels"
_CONTENT_TYPES_PART: str = "[Content_Types].xml"
_MODEL_REL_TYPE: str = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_ZIP_EPOCH: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
_CORE_NS: str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

_UNITS: frozenset[str] = frozenset(
    {"micron", "millimeter", "centimeter", "inch", "foot", "meter"}
)
_DEFAULT_UNIT: str = "millimeter"
# The metadata names the core specification defines; any other name has to
# carry a namespace prefix, so a global attribute polyxios cannot spell that
# way stays out of the file.
_METADATA_NAMES: frozenset[str] = frozenset(
    {
        "Title",
        "Designer",
        "Description",
        "Copyright",
        "LicenseTerms",
        "Rating",
        "CreationDate",
        "ModificationDate",
        "Application",
    }
)
_COLOR_KEY: str = "colors"
# An integer colour column counts 0..255 the way every image format does; a
# float one runs 0..1.
_INT_COLOR_MAX: float = 255.0
_HEX_COLOR = re.compile(r"^#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")
# XML 1.0 admits no control character but tab, newline and return; a lone
# surrogate has no UTF-8 spelling at all.
_XML_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")

_TRIANGLE: int = ELEMENT_TYPES["triangle"]
_QUAD: int = ELEMENT_TYPES["quad"]
_PIXEL: int = ELEMENT_TYPES["pixel"]
_POLYGON: int = ELEMENT_TYPES["polygon"]
_STRIP: int = ELEMENT_TYPES["triangle_strip"]

# For each element type of a fixed corner count, the corner triples of the
# triangles it fans into; a quadratic surface fans like its linear corners.
_TRIANGLE_FAN: np.ndarray = np.array([[0, 1, 2]])
_QUAD_FAN: np.ndarray = np.array([[0, 1, 2], [0, 2, 3]])
_PIXEL_FAN: np.ndarray = np.array([[0, 1, 3], [0, 3, 2]])
_FIXED_FANS: dict[int, np.ndarray] = {
    _TRIANGLE: _TRIANGLE_FAN,
    _QUAD: _QUAD_FAN,
    _PIXEL: _PIXEL_FAN,
    **{
        code: _TRIANGLE_FAN if corners == 3 else _QUAD_FAN
        for code, corners in QUADRATIC_SURFACE_CORNERS.items()
    },
}

# A 4x3 affine matrix in the row-vector convention the specification uses -
# ``m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32``, the last row the
# translation - is kept here as the 4x4 it embeds in, so a chain of them
# composes by matrix product.
_IDENTITY: np.ndarray = np.eye(4, dtype=np.float64)


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a 3MF package and return its build as a triangle mesh.

    Parameters
    ----------
    path
        Path to the ``.3mf`` file, or an open binary file object.
    lazy
        Not supported: a ZIP member has to be inflated before it is parsed.

    Returns
    -------
    PolyData
        Every triangle of every object the build places, the objects'
        transforms applied. An object's triangles form an element tag group
        named after the object - its ``name`` attribute, or ``object_<id>``
        - so an object placed twice, or two objects of one name, contribute
        to the one group.
        A triangle whose material is a base material or a colour is read
        into ``element_attrs["colors"]``, RGBA in 0..1, NaN on the
        triangles that name none. The model's ``unit`` and its ``metadata``
        entries land in ``global_attrs``.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        When the package is not a ZIP, holds no model part, or the model
        names a vertex or an object it does not hold, or gives two
        resources one id.
    """
    if lazy:
        raise LazyReadError("3MF is a ZIP package and cannot be memory-mapped.")
    name = source_name(path)
    raw = read_bytes(path)
    try:
        package = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise CodecError(f"{name!r} is not a ZIP package: {exc}") from exc
    with package:
        model_part = _model_part(package, name)
        try:
            root = ET.fromstring(package.read(model_part))
        except ET.ParseError as exc:
            raise CodecError(
                f"{name!r}: {model_part} is not well-formed XML: {exc}"
            ) from exc
    if _local(root.tag) != "model":
        raise CodecError(
            f"{name!r}: {model_part} holds a <{_local(root.tag)}> where a"
            " <model> was expected."
        )
    return _read_model(root, name)


def write(
    poly: PolyData, path: Source, *, unit: str | None = None, **opts: Any
) -> None:
    """Write a PolyData as a 3MF package.

    Parameters
    ----------
    poly
        PolyData to write. Triangles are written as they are; quads,
        pixels, polygons and triangle strips are split into triangles; every
        other element has no 3MF triangle and is dropped with a warning.
    path
        Output file path, or an open binary file object.
    unit
        The model's unit of length: ``micron``, ``millimeter``,
        ``centimeter``, ``inch``, ``foot`` or ``meter``. Falls back to
        ``global_attrs["unit"]``, then to millimetres; a mesh unit the
        format lacks is written as millimetres with a warning.

    Raises
    ------
    CodecError
        When the mesh has elements but none of them is a surface, a vertex
        a triangle uses is not finite, a tag group name or metadata value
        holds a character XML cannot carry, or ``unit`` is not one the
        format defines.
    """
    unit = _pick_unit(poly, unit)
    tris, source = _triangles(poly)
    if tris.size and not np.isfinite(poly.vertices[np.unique(tris)]).all():
        raise CodecError(
            ".3mf: a vertex a triangle uses is NaN or infinite; 3MF has no"
            " spelling for either."
        )
    colors = _facet_colors(poly, source)
    palette, material = _palette(colors)
    objects = _partition(poly, source, len(tris))

    parts: list[str] = []
    next_id = 1
    if palette:
        parts.append(_basematerials_xml(next_id, palette))
        material_id = next_id
        next_id += 1
    else:
        material_id = 0
    items: list[int] = []
    for obj_name, rows in objects:
        if obj_name is None and f"object_{next_id}" in (poly.element_tags or {}):
            warnings.warn(
                f".3mf: the untagged triangles are object {next_id}, which reads"
                f" back as 'object_{next_id}' - the name of a tag group; the two"
                " merge on read.",
                stacklevel=2,
            )
        parts.append(
            _object_xml(
                next_id,
                obj_name,
                poly.vertices,
                tris[rows],
                material_id,
                material[rows] if palette else None,
            )
        )
        items.append(next_id)
        next_id += 1

    metadata = "".join(
        f'  <metadata name="{key}">{_escape(str(value), key)}</metadata>\n'
        for key, value in (poly.global_attrs or {}).items()
        if key in _METADATA_NAMES and isinstance(value, (str, int, float))
    )
    build = "".join(f'    <item objectid="{i}" />\n' for i in items)
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="{unit}" xml:lang="en-US" xmlns="{_CORE_NS}">\n'
        f"{metadata}"
        "  <resources>\n"
        f"{''.join(parts)}"
        "  </resources>\n"
        "  <build>\n"
        f"{build}"
        "  </build>\n"
        "</model>\n"
    )
    write_bytes(path, _package(model))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    """Return an element tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _model_part(package: zipfile.ZipFile, name: str) -> str:
    """Return the name of the model part the package's root relationships point to.

    A package without a ``_rels/.rels`` - or with one that names no model - is
    read through the conventional ``3D/3dmodel.model`` when it holds one.
    """
    members = set(package.namelist())
    target: str | None = None
    if _RELS_PART in members:
        try:
            rels = ET.fromstring(package.read(_RELS_PART))
        except ET.ParseError as exc:
            raise CodecError(
                f"{name!r}: {_RELS_PART} is not well-formed XML: {exc}"
            ) from exc
        for rel in rels:
            if _local(rel.tag) == "Relationship" and rel.get("Type") == _MODEL_REL_TYPE:
                # OPC spells a part name as a URI, so a space is ``%20``.
                target = urllib.parse.unquote(rel.get("Target", "")).lstrip("/")
                break
    if target is None:
        target = _MODEL_PART
    if target not in members:
        raise CodecError(f"{name!r} holds no model part ({target}).")
    return target


def _read_model(root: ET.Element, name: str) -> PolyData:
    """Assemble the build of a parsed ``<model>``."""
    unit = root.get("unit", _DEFAULT_UNIT)
    if unit not in _UNITS:
        warnings.warn(
            f"{name!r}: unit {unit!r} is not one 3MF defines"
            f" ({', '.join(sorted(_UNITS))}); it is kept as read.",
            stacklevel=3,
        )
    global_attrs: dict[str, Any] = {"unit": unit}
    objects: dict[int, ET.Element] = {}
    properties: dict[int, np.ndarray | None] = {}
    for child in root:
        tag = _local(child.tag)
        if tag == "metadata":
            key = child.get("name")
            if key:
                global_attrs[key] = child.text or ""
        elif tag == "resources":
            for res in child:
                kind = _local(res.tag)
                if kind == "object":
                    rid = _int(res.get("id"), f"{name!r}: an <object> id")
                    if rid in objects or rid in properties:
                        raise CodecError(f"{name!r} gives two resources id {rid}.")
                    objects[rid] = res
                    continue
                if res.get("id") is None:
                    continue
                rid = _int(res.get("id"), f"{name!r}: a <{kind}> id")
                if rid in objects or rid in properties:
                    raise CodecError(f"{name!r} gives two resources id {rid}.")
                if kind in ("basematerials", "colorgroup"):
                    properties[rid] = _read_colors(res, kind, name)
                else:
                    # Textures, composites and multi-properties carry no
                    # colour a triangle could be read as.
                    properties[rid] = None

    meshes: dict[int, _Mesh] = {}
    parts: list[_Part] = []
    for child in root:
        if _local(child.tag) != "build":
            continue
        for item in child:
            if _local(item.tag) != "item":
                continue
            oid = _int(item.get("objectid"), f"{name!r}: a build item's objectid")
            matrix = _transform(item.get("transform"), name)
            _place(oid, matrix, objects, properties, meshes, parts, name, ())

    return _assemble(parts, global_attrs)


class _Mesh:
    """One object's mesh, parsed once and placed as many times as it is built."""

    __slots__ = ("colors", "triangles", "vertices")

    def __init__(
        self, vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray | None
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.colors = colors


# One placed mesh: the object's label, its vertices under the transform, its
# triangles and its per-triangle RGBA.
_Part = tuple[str, np.ndarray, np.ndarray, np.ndarray]


def _place(
    oid: int,
    matrix: np.ndarray,
    objects: dict[int, ET.Element],
    properties: dict[int, np.ndarray | None],
    meshes: dict[int, _Mesh],
    parts: list[_Part],
    name: str,
    stack: tuple[int, ...],
) -> None:
    """Place object ``oid`` under ``matrix``, recursing through its components."""
    if oid in stack:
        chain = " -> ".join(str(i) for i in (*stack, oid))
        raise CodecError(f"{name!r}: object {chain} contains itself.")
    obj = objects.get(oid)
    if obj is None:
        raise CodecError(f"{name!r} names object {oid}, which it does not hold.")
    for child in obj:
        tag = _local(child.tag)
        if tag == "mesh":
            mesh = meshes.get(oid)
            if mesh is None:
                mesh = _read_mesh(child, obj, properties, name)
                meshes[oid] = mesh
            label = obj.get("name") or f"object_{oid}"
            parts.append((label, *_placed(mesh, matrix)))
        elif tag == "components":
            for comp in child:
                if _local(comp.tag) != "component":
                    continue
                cid = _int(comp.get("objectid"), f"{name!r}: a component's objectid")
                local = _transform(comp.get("transform"), name)
                _place(
                    cid,
                    local @ matrix,
                    objects,
                    properties,
                    meshes,
                    parts,
                    name,
                    (*stack, oid),
                )


def _placed(
    mesh: _Mesh, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mesh's vertices, triangles and colours under ``matrix``.

    A transform that mirrors turns every triangle inside out, so the winding
    is reversed to keep the surface facing the way the modeller left it.
    """
    vertices = mesh.vertices @ matrix[:3, :3] + matrix[3, :3]
    triangles = mesh.triangles
    if np.linalg.det(matrix[:3, :3]) < 0:
        triangles = triangles[:, [0, 2, 1]]
    colors = mesh.colors
    if colors is None:
        colors = np.full((len(triangles), 4), np.nan)
    return vertices, triangles, colors


def _read_mesh(
    mesh: ET.Element,
    obj: ET.Element,
    properties: dict[int, np.ndarray | None],
    name: str,
) -> _Mesh:
    """Parse one ``<mesh>``: its vertices, triangles and per-triangle colour."""
    oid = obj.get("id")
    vertex_rows: list[tuple[str | None, str | None, str | None]] = []
    tri_rows: list[tuple[str | None, str | None, str | None]] = []
    pids: list[str | None] = []
    p1s: list[str | None] = []
    for child in mesh:
        tag = _local(child.tag)
        if tag == "vertices":
            vertex_rows = [
                (v.get("x"), v.get("y"), v.get("z"))
                for v in child
                if _local(v.tag) == "vertex"
            ]
        elif tag == "triangles":
            for t in child:
                if _local(t.tag) != "triangle":
                    continue
                tri_rows.append((t.get("v1"), t.get("v2"), t.get("v3")))
                pids.append(t.get("pid"))
                p1s.append(t.get("p1"))
    try:
        vertices = np.array(vertex_rows, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{name!r}: object {oid} has a vertex that is not three numbers."
        ) from exc
    try:
        triangles = np.array(tri_rows, dtype=np.int64).reshape(-1, 3)
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{name!r}: object {oid} has a triangle that is not three vertex indices."
        ) from exc
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        bad = int(
            triangles.max() if triangles.max() >= len(vertices) else triangles.min()
        )
        raise CodecError(
            f"{name!r}: object {oid} names vertex {bad} but holds {len(vertices)}."
        )
    colors = _triangle_colors(obj, pids, p1s, properties, name)
    return _Mesh(vertices, triangles, colors)


def _triangle_colors(
    obj: ET.Element,
    pids: list[str | None],
    p1s: list[str | None],
    properties: dict[int, np.ndarray | None],
    name: str,
) -> np.ndarray | None:
    """Return the RGBA of each triangle from its material, or None when none has one.

    A triangle without ``pid`` takes the object's ``pid``, one without ``p1``
    the object's ``pindex``, each on its own; a property a texture or
    composite defines is not a colour and reads as NaN. Only ``p1`` is read:
    3MF lets a triangle name three properties, one per corner, and a colour
    per corner has no element to land on.
    """
    if not pids:
        return None
    obj_pid, obj_pindex = obj.get("pid"), obj.get("pindex")
    colors = np.full((len(pids), 4), np.nan)
    any_color = False
    for i, (pid, p1) in enumerate(zip(pids, p1s, strict=True)):
        if pid is None:
            pid = obj_pid
        if p1 is None:
            p1 = obj_pindex
        if pid is None or p1 is None:
            continue
        group = _int(pid, f"{name!r}: a triangle's pid")
        index = _int(p1, f"{name!r}: a triangle's p1")
        if group not in properties:
            raise CodecError(
                f"{name!r} names property group {group}, which it does not hold."
            )
        table = properties[group]
        if table is None:
            continue
        if index < 0 or index >= len(table):
            raise CodecError(
                f"{name!r}: property group {group} holds {len(table)} entries;"
                f" p1={index} names none of them."
            )
        colors[i] = table[index]
        any_color = True
    return colors if any_color else None


def _read_colors(group: ET.Element, kind: str, name: str) -> np.ndarray:
    """Return the ``(n, 4)`` RGBA table of a ``<basematerials>`` or ``<colorgroup>``."""
    attr = "displaycolor" if kind == "basematerials" else "color"
    rows = []
    for entry in group:
        value = entry.get(attr)
        if value is None:
            rows.append((np.nan,) * 4)
            continue
        m = _HEX_COLOR.match(value.strip())
        if m is None:
            raise CodecError(
                f"{name!r}: {value!r} is not an sRGB colour (#RRGGBB or #RRGGBBAA)."
            )
        rgb = bytes.fromhex(m.group(1))
        alpha = int(m.group(2), 16) if m.group(2) else 255
        rows.append(tuple(c / _INT_COLOR_MAX for c in (*rgb, alpha)))
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _transform(text: str | None, name: str) -> np.ndarray:
    """Return the 4x4 (row-vector convention) a ``transform`` attribute spells."""
    if text is None:
        return _IDENTITY
    values = text.split()
    if len(values) != 12:
        raise CodecError(f"{name!r}: a transform needs 12 numbers, got {text!r}.")
    try:
        m = np.array(values, dtype=np.float64).reshape(4, 3)
    except ValueError as exc:
        raise CodecError(f"{name!r}: a transform is not numbers: {text!r}.") from exc
    out = np.eye(4)
    out[:, :3] = m
    return out


def _int(text: str | None, what: str) -> int:
    """Parse a required integer attribute, naming it when it is missing or not one."""
    if text is None:
        raise CodecError(f"{what} is missing.")
    try:
        return int(text)
    except ValueError as exc:
        raise CodecError(f"{what} is not an integer: {text!r}.") from exc


def _assemble(parts: list[_Part], global_attrs: dict[str, Any]) -> PolyData:
    """Concatenate the placed parts into one PolyData, one tag per object name."""
    if not parts:
        return PolyData(
            vertices=np.empty((0, 3), dtype=np.float64),
            connectivity=np.array([], dtype=np.int32),
            offsets=np.zeros(1, dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
            global_attrs=global_attrs,
        )
    vertex_blocks: list[np.ndarray] = []
    tri_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    tags: dict[str, list[np.ndarray]] = {}
    n_verts = 0
    n_tris = 0
    for label, vertices, triangles, colors in parts:
        vertex_blocks.append(vertices)
        tri_blocks.append(triangles + n_verts)
        color_blocks.append(colors)
        tags.setdefault(label, []).append(np.arange(n_tris, n_tris + len(triangles)))
        n_verts += len(vertices)
        n_tris += len(triangles)
    # The offsets run to 3 * n_tris, so they can outgrow int32 before the
    # vertex indices do.
    widest = max(n_verts, 3 * n_tris + 1)
    index_dtype = np.int32 if widest <= np.iinfo(np.int32).max else np.int64
    connectivity = np.concatenate(tri_blocks).astype(index_dtype).ravel()
    colors = np.concatenate(color_blocks)
    element_attrs: dict[str, np.ndarray] = {}
    if not np.isnan(colors).all():
        element_attrs[_COLOR_KEY] = colors
    return PolyData(
        vertices=np.concatenate(vertex_blocks),
        connectivity=connectivity,
        offsets=np.arange(0, 3 * n_tris + 1, 3, dtype=index_dtype),
        element_types=np.full(n_tris, _TRIANGLE, dtype=np.uint8),
        element_attrs=element_attrs,
        element_tags={
            label: np.concatenate(rows).astype(index_dtype)
            for label, rows in tags.items()
        },
        global_attrs=global_attrs,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _pick_unit(poly: PolyData, unit: str | None) -> str:
    """Return the unit to write, from the argument, the mesh, or the default.

    An explicit ``unit`` the format lacks is refused; one the mesh carries
    from another format is replaced by millimetres with a warning, so a
    mesh read elsewhere can still be written.
    """
    if unit is not None:
        if unit not in _UNITS:
            raise CodecError(
                f".3mf: {unit!r} is not a 3MF unit; one of {sorted(_UNITS)} is."
            )
        return unit
    unit = (poly.global_attrs or {}).get("unit", _DEFAULT_UNIT)
    if unit not in _UNITS:
        warnings.warn(
            f".3mf: global_attrs['unit'] {unit!r} is not a 3MF unit; written"
            f" as {_DEFAULT_UNIT!r}. Pass unit= to choose one of {sorted(_UNITS)}.",
            stacklevel=3,
        )
        return _DEFAULT_UNIT
    return unit


def _triangles(poly: PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return the triangles to write and the element each came from.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(n, 3)`` vertex indices and the ``(n,)`` index of the source
        element, so a per-element colour or tag can follow its triangles.
    """
    types = np.asarray(poly.element_types)
    connectivity = np.asarray(poly.connectivity)
    offsets = np.asarray(poly.offsets)
    tris: list[np.ndarray] = []
    source: list[np.ndarray] = []
    # Every type of a fixed corner count is fanned in one gather over the
    # elements of that type; only the ragged types are walked one by one.
    for code, fan in _FIXED_FANS.items():
        rows = np.flatnonzero(types == code)
        if rows.size == 0:
            continue
        starts = offsets[rows]
        for corner_fan in fan:
            tris.append(connectivity[starts[:, None] + corner_fan[None, :]])
            source.append(rows)
    for code in (_POLYGON, _STRIP):
        for i in np.flatnonzero(types == code).tolist():
            cell = connectivity[offsets[i] : offsets[i + 1]]
            n = len(cell) - 2
            if n <= 0:
                continue
            if code == _POLYGON:
                fan = np.column_stack([np.full(n, cell[0]), cell[1:-1], cell[2:]])
            else:
                fan = np.column_stack([cell[:-2], cell[1:-1], cell[2:]])
                fan[1::2, :2] = fan[1::2, 1::-1]
            tris.append(fan)
            source.append(np.full(n, i))
    dropped: dict[str, int] = {}
    for code in np.unique(types).tolist():
        if code not in _FIXED_FANS and code not in (_POLYGON, _STRIP):
            label = ELEMENT_TYPES_INV.get(code, str(code))
            dropped[label] = int((types == code).sum())
    named = ", ".join(f"{n} {label}" for label, n in dropped.items())
    if not tris:
        if dropped:
            raise CodecError(
                f".3mf: 3MF holds triangles only, and the mesh has none: {named}."
            )
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64)
    if dropped:
        warnings.warn(
            f".3mf: 3MF holds triangles only; {named} element(s) have no"
            " triangle to be written as and were dropped.",
            stacklevel=3,
        )
    all_source = np.concatenate(source).astype(np.int64)
    order = np.argsort(all_source, kind="stable")
    return np.concatenate(tris).astype(np.int64)[order], all_source[order]


def _facet_colors(poly: PolyData, source: np.ndarray) -> np.ndarray | None:
    """Return the RGBA of each triangle written, or None when the mesh carries none.

    An integer attribute is read as 0..255 and a floating point one as 0..1;
    a three-column one is opaque. A row that is not finite is no colour.
    """
    stored = (poly.element_attrs or {}).get(_COLOR_KEY)
    if stored is None or len(source) == 0:
        return None
    values = np.asarray(stored)
    if (
        values.ndim != 2
        or values.shape[0] != len(poly.element_types)
        or values.shape[1] not in (3, 4)
        or values.dtype.kind not in "fiub"
    ):
        warnings.warn(
            f".3mf: element_attrs['{_COLOR_KEY}'] is not three or four"
            " components per element; the colours were not written.",
            stacklevel=3,
        )
        return None
    picked = values[source].astype(np.float64, copy=False)
    if values.dtype.kind in "iu":
        picked = picked / _INT_COLOR_MAX
    if picked.shape[1] == 3:
        picked = np.hstack([picked, np.ones((len(picked), 1))])
    return picked


def _palette(colors: np.ndarray | None) -> tuple[list[str], np.ndarray]:
    """Return the distinct colours as ``#RRGGBBAA`` and each triangle's index into them.

    A triangle without a colour gets index -1.
    """
    if colors is None:
        return [], np.empty(0, dtype=np.int64)
    valid = np.isfinite(colors).all(axis=1)
    material = np.full(len(colors), -1, dtype=np.int64)
    if not valid.any():
        return [], material
    bytes_ = np.clip(np.rint(colors[valid] * _INT_COLOR_MAX), 0, 255).astype(np.uint8)
    unique, inverse = np.unique(bytes_, axis=0, return_inverse=True)
    material[valid] = inverse.ravel()
    palette = ["#" + row.tobytes().hex().upper() for row in unique]
    return palette, material


def _partition(
    poly: PolyData, source: np.ndarray, n_tris: int
) -> list[tuple[str | None, np.ndarray]]:
    """Split the triangles into objects, one per tag group, the rest in one more.

    A tag group takes the triangles of its members that no earlier group
    took: a 3MF object owns its triangles, so an element in two groups goes
    with the first and the second is told so. A group left with nothing is
    not written, since a mesh with no triangle is not a 3MF object.
    """
    claimed = np.zeros(n_tris, dtype=bool)
    objects: list[tuple[str | None, np.ndarray]] = []
    shared: list[str] = []
    for label, members in (poly.element_tags or {}).items():
        in_group = np.isin(source, np.asarray(members))
        rows = np.flatnonzero(in_group & ~claimed)
        if in_group.any() and rows.size < in_group.sum():
            shared.append(label)
        if rows.size:
            objects.append((label, rows))
            claimed[rows] = True
    if shared:
        warnings.warn(
            f".3mf: element tag group(s) {shared} name elements an earlier"
            " group already wrote; a 3MF object owns its triangles, so those"
            " stayed with the first.",
            stacklevel=3,
        )
    rest = np.flatnonzero(~claimed)
    if rest.size:
        objects.append((None, rest))
    return objects


def _basematerials_xml(rid: int, palette: list[str]) -> str:
    """Return one ``<basematerials>`` with a ``<base>`` per palette colour."""
    entries = "".join(
        f'      <base name="{color}" displaycolor="{color}" />\n' for color in palette
    )
    return f'    <basematerials id="{rid}">\n{entries}    </basematerials>\n'


def _object_xml(
    rid: int,
    label: str | None,
    vertices: np.ndarray,
    tris: np.ndarray,
    material_id: int,
    material: np.ndarray | None,
) -> str:
    """Return one ``<object>`` holding ``tris`` over the vertices they use."""
    used, local = np.unique(tris, return_inverse=True)
    local = local.reshape(tris.shape)
    coords = vertices[used].astype(np.float64)
    vertex_lines = "".join(
        f'          <vertex x="{x!r}" y="{y!r}" z="{z!r}" />\n'
        for x, y, z in coords.tolist()
    )
    if material is None:
        tri_lines = "".join(
            f'          <triangle v1="{a}" v2="{b}" v3="{c}" />\n'
            for a, b, c in local.tolist()
        )
    else:
        tri_lines = "".join(
            f'          <triangle v1="{a}" v2="{b}" v3="{c}" />\n'
            if m < 0
            else f'          <triangle v1="{a}" v2="{b}" v3="{c}" pid="{material_id}" p1="{m}" />\n'
            for (a, b, c), m in zip(local.tolist(), material.tolist(), strict=True)
        )
    name_attr = (
        f' name="{_escape(label, "the tag group name")}"' if label is not None else ""
    )
    return (
        f'    <object id="{rid}" type="model"{name_attr}>\n'
        "      <mesh>\n"
        "        <vertices>\n"
        f"{vertex_lines}"
        "        </vertices>\n"
        "        <triangles>\n"
        f"{tri_lines}"
        "        </triangles>\n"
        "      </mesh>\n"
        "    </object>\n"
    )


def _escape(text: str, what: str) -> str:
    """Return ``text`` with the four characters XML reserves as entities.

    Raises
    ------
    CodecError
        When ``text`` holds a control character XML 1.0 has no spelling for,
        which would make the model part malformed; ``what`` names the value.
    """
    if _XML_FORBIDDEN.search(text):
        raise CodecError(
            f".3mf: {what} {text!r} holds a control character XML cannot carry."
        )
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _package(model: str) -> bytes:
    """Wrap the model part in the OPC package 3MF requires."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels"'
        ' ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
        '  <Default Extension="model"'
        ' ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
        "</Types>\n"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        f'  <Relationship Target="/{_MODEL_PART}" Id="rel0" Type="{_MODEL_REL_TYPE}" />\n'
        "</Relationships>\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as package:
        for member, text in (
            (_CONTENT_TYPES_PART, content_types),
            (_RELS_PART, rels),
            (_MODEL_PART, model),
        ):
            # A fixed timestamp keeps two writes of one mesh byte-identical;
            # the ZIP epoch is the earliest a ZIP entry can carry.
            info = zipfile.ZipInfo(member, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(info, text)
    return buf.getvalue()
