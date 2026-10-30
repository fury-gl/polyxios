"""3MF (3D Manufacturing Format) codec - read + write.

A ``.3mf`` file is a ZIP package holding an XML model part. Every ``<object>``
placed by the ``<build>`` is read - a ``<mesh>`` outright, a ``<components>``
assembly through its parts, each ``transform`` composed on the way down - and
comes back as a group of triangles tagged with the object's name. A material
a triangle names is read as its colour; on write the tag groups become the
objects and the colours the materials.
"""

from __future__ import annotations

import array
from collections.abc import Iterator
import io
import lzma
import re
from typing import Any
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    QUADRATIC_SURFACE_CORNERS,
)
from polyxios._io import Source, read_bytes, source_name, write_bytes
from polyxios._types import PolyData
from polyxios._units import conversion
from polyxios._warn import warn_caller
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".3mf"

_MODEL_PART: str = "3D/3dmodel.model"
_RELS_PART: str = "_rels/.rels"
# Entities are refused whatever expat's amplification limits, which a
# Python linked against an older system expat lacks.
_ENTITY_DECLS: tuple[bytes, ...] = tuple(
    "<!ENTITY".encode(enc) for enc in ("utf-8", "utf-16-le", "utf-16-be")
)
# Bytes kept from one chunk of a streamed part to find a declaration the
# next chunk completes.
_ENTITY_SPAN: int = max(len(decl) for decl in _ENTITY_DECLS) - 1
_CONTENT_TYPES_PART: str = "[Content_Types].xml"
_MODEL_REL_TYPE: str = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_ZIP_EPOCH: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
_CORE_NS: str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
# A member is refused past this many bytes inflated per byte stored, once it
# inflates past the floor. A mesh's XML stores at 5:1 to 15:1; deflate tops
# out near 1000:1, which is what a member built to exhaust memory reaches.
_INFLATE_RATIO: int = 200
_INFLATE_FLOOR: int = 1 << 28
# What opening a ZIP or reading a member raises beyond a bad archive: an
# encrypted member (RuntimeError), a compression method or ZIP version
# zipfile lacks (NotImplementedError), and offsets that point before the
# archive or a name flagged UTF-8 that is not (ValueError). A bzip2 or LZMA
# member that does not decompress raises OSError or LZMAError; every source
# is opened from memory, so no OSError here is about a file.
_ZIP_ERRORS: tuple[type[Exception], ...] = (
    zipfile.BadZipFile,
    zlib.error,
    lzma.LZMAError,
    EOFError,
    OSError,
    RuntimeError,
    NotImplementedError,
    ValueError,
)

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
_XML_FORBIDDEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")

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
        named after the object - its ``name`` attribute, empty or not, or
        ``object_<id>`` when it has none - so an object placed twice, or two
        objects of one name, contribute to the one group.
        A triangle whose material is a base material or a colour is read
        into ``element_attrs["colors"]``, RGBA in 0..1, NaN on the
        triangles that name none. The model's ``unit`` and its ``metadata``
        entries land in ``global_attrs``.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        When the package is not a ZIP or not one ``zipfile`` can open,
        holds no model part or one that does not inflate, is encrypted, is
        stored in a method ``zipfile`` lacks or inflates the way only a ZIP
        bomb does, the model or the root relationships declare an XML
        entity, or the model names a vertex or an object it does not hold,
        or gives two resources one id.
    """
    if lazy:
        raise LazyReadError("3MF is a ZIP package and cannot be memory-mapped.")
    name = source_name(path)
    raw = read_bytes(path)
    try:
        package = zipfile.ZipFile(io.BytesIO(raw))
    except _ZIP_ERRORS as exc:
        raise CodecError(f"{name!r} is not a ZIP package: {exc}") from exc
    with package:
        model_part = _model_part(package, name)
        _check_inflation(package.getinfo(model_part), name)
        target = _ModelTarget(name)
        parser = ET.XMLParser(target=target)
        try:
            where = f"{name!r}: {model_part}"
            tail = b""
            for chunk in _inflated(package, model_part, name):
                if not tail:
                    _check_declared_encoding(chunk, where)
                _refuse_entities(tail + chunk[:_ENTITY_SPAN], where, fmt="3MF")
                _refuse_entities(chunk, where, fmt="3MF")
                tail = chunk[-_ENTITY_SPAN:]
                parser.feed(chunk)
            root = parser.close()
        except (ET.ParseError, LookupError) as exc:
            raise CodecError(
                f"{name!r}: {model_part} is not well-formed XML: {exc}"
            ) from exc
    if _local(root.tag) != "model":
        raise CodecError(
            f"{name!r}: {model_part} holds a <{_local(root.tag)}> where a"
            " <model> was expected."
        )
    return _read_model(root, target.meshes, name)


def _inflated(package: zipfile.ZipFile, part: str, name: str) -> Iterator[bytes]:
    """Yield a member of ``package`` as it inflates, ``_PARSE_CHUNK`` at a time.

    Only opening and inflating the member is guarded, so an error raised
    by whatever consumes a chunk is never reported as one of the archive.

    Raises
    ------
    CodecError
        When the member does not open or does not inflate.
    """
    try:
        member = package.open(part)
    except _ZIP_ERRORS as exc:
        raise CodecError(f"{name!r}: {part} does not inflate: {exc}") from exc
    with member:
        while True:
            try:
                chunk = member.read(_PARSE_CHUNK)
            except _ZIP_ERRORS as exc:
                raise CodecError(f"{name!r}: {part} does not inflate: {exc}") from exc
            if not chunk:
                return
            yield chunk


# How much of the model part is handed to expat at a time. Fed the whole
# part at once, expat grows one buffer to hold it and cannot grow it past
# 1 GiB - a mesh of ten million vertices - so it is fed as it is inflated.
_PARSE_CHUNK: int = 1 << 20

# A triangle's own material: its index in the mesh, its ``pid`` and its
# ``p1``, either of which it may leave to the object.
_Material = tuple[int, str | None, str | None]


class _RawMesh:
    """The numbers of one ``<mesh>``, gathered as its elements are parsed.

    Parameters
    ----------
    oid
        The ``id`` of the ``<object>`` the mesh belongs to, for messages.
    """

    __slots__ = ("coords", "indices", "materials", "oid")

    def __init__(self, oid: str | None) -> None:
        self.oid = oid
        self.coords = array.array("d")
        self.indices = array.array("q")
        self.materials: list[_Material] = []


class _ModelTarget:
    """A parser target that keeps the model's tree but not its geometry.

    A ``<vertex>`` or ``<triangle>`` is one Element and one attribute dict
    apiece in a parsed tree, and a mesh of ten million vertices has thirty
    million of them - more memory than the file inflated. Those two elements
    are read off the parser as they close and their numbers appended to the
    enclosing mesh's buffers instead; everything else is built into the tree
    as usual, so the model is walked the way it always was. Only the layout
    the specification gives is read - ``<vertex>`` straight under
    ``<vertices>`` straight under ``<mesh>``, and the triangles likewise -
    the way a walk of the tree read them; one nested in an extension
    element is left in the tree and not counted.

    Parameters
    ----------
    name
        The source, for messages.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._builder = ET.TreeBuilder()
        # One entry per open element: its local name, or None for a vertex
        # or triangle read off the parser, which the builder never saw and
        # must not be told to end.
        self._stack: list[str | None] = []
        self._oid: str | None = None
        self._mesh: _RawMesh | None = None
        self.meshes: dict[int, _RawMesh] = {}

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        local = _local(tag)
        mesh = self._mesh
        if (
            mesh is not None
            and local == "vertex"
            and self._stack[-2:] == ["mesh", "vertices"]
        ):
            try:
                mesh.coords.extend(
                    (float(attrs["x"]), float(attrs["y"]), float(attrs["z"]))
                )
            except (KeyError, ValueError) as exc:
                raise CodecError(
                    f"{self._name!r}: object {mesh.oid} has a vertex that is"
                    " not three numbers."
                ) from exc
            self._stack.append(None)
            return
        if (
            mesh is not None
            and local == "triangle"
            and self._stack[-2:] == ["mesh", "triangles"]
        ):
            try:
                mesh.indices.extend(
                    (int(attrs["v1"]), int(attrs["v2"]), int(attrs["v3"]))
                )
            except (KeyError, ValueError, OverflowError) as exc:
                raise CodecError(
                    f"{self._name!r}: object {mesh.oid} has a triangle that is"
                    " not three vertex indices."
                ) from exc
            pid, p1 = attrs.get("pid"), attrs.get("p1")
            if pid is not None or p1 is not None:
                mesh.materials.append((len(mesh.indices) // 3 - 1, pid, p1))
            self._stack.append(None)
            return
        if local == "object":
            self._oid = attrs.get("id")
        elem = self._builder.start(tag, attrs)
        if local == "mesh":
            self._mesh = self.meshes[id(elem)] = _RawMesh(self._oid)
        self._stack.append(local)

    def end(self, tag: str) -> None:
        local = self._stack.pop()
        if local is None:
            return
        self._builder.end(tag)
        if local == "mesh":
            self._mesh = None

    def data(self, text: str) -> None:
        # The whitespace between ten million vertices is ten million strings
        # the builder would hold until </vertices> closed; nothing in a
        # vertex, a triangle or the block around them is text worth keeping.
        if self._mesh is not None and self._stack[-1] in (
            None,
            "vertices",
            "triangles",
        ):
            return
        self._builder.data(text)

    def close(self) -> ET.Element:
        return self._builder.close()


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
        format lacks is written as millimetres with a warning, the vertices
        scaled to millimetres when that unit is a length polyxios knows -
        ``yard`` or ``kilometer`` read from DXF, say.

    Raises
    ------
    CodecError
        When the mesh has elements but none of them is a surface, a vertex
        a triangle uses is not finite, a tag group name or metadata value
        holds a character XML cannot carry, or ``unit`` is not one the
        format defines.
    """
    unit, scale = _pick_unit(poly, unit)
    tris, source = _triangles(poly)
    vertices = _scaled(poly.vertices, scale)
    if tris.size and not np.isfinite(vertices[np.unique(tris)]).all():
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
            warn_caller(
                f".3mf: the untagged triangles are object {next_id}, which reads"
                f" back as 'object_{next_id}' - the name of a tag group; the two"
                " merge on read."
            )
        parts.append(
            _object_xml(
                next_id,
                obj_name,
                vertices,
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


_XML_DECLARATION = re.compile(rb"\A(?:\xef\xbb\xbf)?<\?xml\b[^>]*?\?>")


def _check_declared_encoding(head: bytes, where: str) -> None:
    """Refuse a document whose XML declaration names an encoding expat lacks.

    Parameters
    ----------
    head
        The document's opening bytes.
    where
        The document, or the part of a package, as a message starts.

    Raises
    ------
    CodecError
        When expat cannot decode the declared encoding: it raises
        LookupError for one it does not know and ValueError for a multi-byte
        one. The declaration is parsed on its own so that neither is
        confused with what a parser target raises later.
    """
    match = _XML_DECLARATION.match(head)
    if match is None:
        return
    try:
        ET.fromstring(match.group(0) + b"<a/>")
    except (LookupError, ValueError) as exc:
        raise CodecError(f"{where} is not well-formed XML: {exc}") from exc
    except ET.ParseError:
        # Anything else wrong with the declaration is the parser's to report.
        return


def _refuse_entities(data: bytes, where: str, *, fmt: str) -> None:
    """Refuse XML text that declares an entity.

    Parameters
    ----------
    data
        The XML bytes, or a chunk of them.
    where
        The document, or the part of a package, the bytes come from, as a
        message starts.
    fmt
        The format's name, for the message.

    Raises
    ------
    CodecError
        When ``data`` holds ``<!ENTITY`` in UTF-8 or either UTF-16.
    """
    if any(decl in data for decl in _ENTITY_DECLS):
        raise CodecError(
            f"{where} declares an XML entity, which {fmt} never needs; refused"
            " rather than expanded."
        )


def _check_inflation(info: zipfile.ZipInfo, name: str) -> None:
    """Refuse a ZIP member that inflates far past what any mesh stores as.

    zipfile stops a member at the size its header declares, so the declared
    size bounds what reading it costs.

    Raises
    ------
    CodecError
        When the member declares more than ``_INFLATE_FLOOR`` bytes and
        more than ``_INFLATE_RATIO`` times its stored size.
    """
    if info.file_size > max(_INFLATE_FLOOR, _INFLATE_RATIO * info.compress_size):
        raise CodecError(
            f"{name!r}: {info.filename} inflates from {info.compress_size} to"
            f" {info.file_size} bytes, past what a mesh stores as; refused as"
            " a ZIP bomb."
        )


def _model_part(package: zipfile.ZipFile, name: str) -> str:
    """Return the name of the model part the package's root relationships point to.

    A package without a ``_rels/.rels`` - or with one that names no model - is
    read through the conventional ``3D/3dmodel.model`` when it holds one.
    """
    members = set(package.namelist())
    target: str | None = None
    if _RELS_PART in members:
        _check_inflation(package.getinfo(_RELS_PART), name)
        data = b"".join(_inflated(package, _RELS_PART, name))
        _refuse_entities(data, f"{name!r}: {_RELS_PART}", fmt="3MF")
        _check_declared_encoding(data, f"{name!r}: {_RELS_PART}")
        try:
            rels = ET.fromstring(data)
        except (ET.ParseError, LookupError) as exc:
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


def _read_model(root: ET.Element, raw: dict[int, _RawMesh], name: str) -> PolyData:
    """Assemble the build of a parsed ``<model>``.

    Parameters
    ----------
    root
        The ``<model>`` element, its meshes stripped of their geometry.
    raw
        Each mesh's geometry, keyed by the id of its ``<mesh>`` element.
    name
        The source, for messages.
    """
    unit = root.get("unit", _DEFAULT_UNIT)
    if unit not in _UNITS:
        warn_caller(
            f"{name!r}: unit {unit!r} is not one 3MF defines"
            f" ({', '.join(sorted(_UNITS))}); it is kept as read."
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
            _place(oid, matrix, objects, properties, meshes, parts, name, (), raw)

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
    raw: dict[int, _RawMesh],
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
                mesh = _read_mesh(raw[id(child)], obj, properties, name)
                meshes[oid] = mesh
            label = obj.get("name")
            if label is None:
                label = f"object_{oid}"
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
                    raw,
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
    raw: _RawMesh,
    obj: ET.Element,
    properties: dict[int, np.ndarray | None],
    name: str,
) -> _Mesh:
    """Shape one ``<mesh>``'s gathered numbers into vertices, triangles and colours."""
    oid = obj.get("id")
    vertices = np.frombuffer(raw.coords, dtype=np.float64).reshape(-1, 3)
    triangles = np.frombuffer(raw.indices, dtype=np.int64).reshape(-1, 3)
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        bad = int(
            triangles.max() if triangles.max() >= len(vertices) else triangles.min()
        )
        raise CodecError(
            f"{name!r}: object {oid} names vertex {bad} but holds {len(vertices)}."
        )
    colors = _triangle_colors(obj, len(triangles), raw.materials, properties, name)
    return _Mesh(vertices, triangles, colors)


def _triangle_colors(
    obj: ET.Element,
    n_tris: int,
    materials: list[_Material],
    properties: dict[int, np.ndarray | None],
    name: str,
) -> np.ndarray | None:
    """Return the RGBA of each triangle from its material, or None when none has one.

    A triangle without ``pid`` takes the object's ``pid``, one without ``p1``
    the object's ``pindex``, each on its own; a property a texture or
    composite defines is not a colour and reads as NaN. Only ``p1`` is read:
    3MF lets a triangle name three properties, one per corner, and a colour
    per corner has no element to land on.

    Parameters
    ----------
    obj
        The ``<object>`` the mesh belongs to.
    n_tris
        Triangles in the mesh.
    materials
        The ``pid`` and ``p1`` of each triangle that spells either, with
        the triangle's index; a triangle that spells neither is not listed.
    properties
        Each property group's colour table, None for a group with none.
    name
        The source, for messages.
    """
    if n_tris == 0:
        return None
    obj_pid, obj_pindex = obj.get("pid"), obj.get("pindex")
    colors = np.full((n_tris, 4), np.nan)
    any_color = False
    if len(materials) < n_tris:
        default = _color_of(obj_pid, obj_pindex, properties, name)
        if default is not None:
            colors[:] = default
            any_color = True
    for index, pid, p1 in materials:
        row = _color_of(
            obj_pid if pid is None else pid,
            obj_pindex if p1 is None else p1,
            properties,
            name,
        )
        colors[index] = np.nan if row is None else row
        any_color = any_color or row is not None
    return colors if any_color else None


def _color_of(
    pid: str | None,
    p1: str | None,
    properties: dict[int, np.ndarray | None],
    name: str,
) -> np.ndarray | None:
    """Return the RGBA a ``pid``/``p1`` pair names, or None when it names no colour."""
    if pid is None or p1 is None:
        return None
    group = _int(pid, f"{name!r}: a triangle's pid")
    index = _int(p1, f"{name!r}: a triangle's p1")
    if group not in properties:
        raise CodecError(
            f"{name!r} names property group {group}, which it does not hold."
        )
    table = properties[group]
    if table is None:
        return None
    if index < 0 or index >= len(table):
        raise CodecError(
            f"{name!r}: property group {group} holds {len(table)} entries;"
            f" p1={index} names none of them."
        )
    return table[index]


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


def _pick_unit(poly: PolyData, unit: str | None) -> tuple[str, float]:
    """Return the unit to write and the factor the vertices are scaled by.

    An explicit ``unit`` the format lacks is refused; one the mesh carries
    from another format is replaced by millimetres with a warning, so a
    mesh read elsewhere can still be written, and the vertices are scaled
    to millimetres when the mesh's unit is a length polyxios knows.
    """
    if unit is not None:
        if unit not in _UNITS:
            raise CodecError(
                f".3mf: {unit!r} is not a 3MF unit; one of {sorted(_UNITS)} is."
            )
        return unit, 1.0
    unit = (poly.global_attrs or {}).get("unit", _DEFAULT_UNIT)
    if unit in _UNITS:
        return unit, 1.0
    return _DEFAULT_UNIT, _fallback_scale(unit, EXTENSION, sorted(_UNITS))


def _fallback_scale(unit: Any, fmt: str, units: list[str]) -> float:
    """Warn that a mesh unit the format lacks is written as millimetres.

    Parameters
    ----------
    unit
        The mesh's ``global_attrs["unit"]``.
    fmt
        The extension the message names.
    units
        The units the format defines, for the message.

    Returns
    -------
    float
        The factor that turns the mesh's lengths into millimetres, or 1 when
        its unit is not a length polyxios knows and the numbers are kept.
    """
    scale = conversion(unit, _DEFAULT_UNIT)
    kept = (
        "the coordinates are kept as they are"
        if scale is None
        else f"the coordinates are scaled by {scale:.15g} to match"
    )
    label = _label(fmt)
    article = "an" if label[0] in "AEIOU" else "a"
    warn_caller(
        f"{fmt}: global_attrs['unit'] {unit!r} is not {article} {label} unit;"
        f" written as {_DEFAULT_UNIT!r} and {kept}. Pass unit= to choose one"
        f" of {units}."
    )
    return 1.0 if scale is None else scale


def _scaled(vertices: np.ndarray, scale: float) -> np.ndarray:
    """Return ``vertices`` multiplied by ``scale``, the array itself when it is 1."""
    if scale == 1.0:
        return vertices
    with np.errstate(over="ignore"):
        return np.asarray(vertices, dtype=np.float64) * scale


def _triangles(
    poly: PolyData, *, fmt: str = EXTENSION
) -> tuple[np.ndarray, np.ndarray]:
    """Return the triangles to write and the element each came from.

    Parameters
    ----------
    poly
        The mesh being written.
    fmt
        The extension the messages name, for a codec sharing this one.

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
                f"{fmt}: {_label(fmt)} holds triangles only, and the mesh has"
                f" none: {named}."
            )
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64)
    if dropped:
        warn_caller(
            f"{fmt}: {_label(fmt)} holds triangles only; {named} element(s) have no"
            " triangle to be written as and were dropped."
        )
    all_source = np.concatenate(source).astype(np.int64)
    order = np.argsort(all_source, kind="stable")
    return np.concatenate(tris).astype(np.int64)[order], all_source[order]


def _facet_colors(
    poly: PolyData, source: np.ndarray, *, fmt: str = EXTENSION
) -> np.ndarray | None:
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
        warn_caller(
            f"{fmt}: element_attrs['{_COLOR_KEY}'] is not three or four"
            " components per element; the colours were not written."
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
    poly: PolyData, source: np.ndarray, n_tris: int, *, fmt: str = EXTENSION
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
        warn_caller(
            f"{fmt}: element tag group(s) {shared} name elements an earlier"
            f" group already wrote; {_label(fmt)} objects own their triangles, so"
            " those stayed with the first."
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


def _label(fmt: str) -> str:
    """Return the format name an extension stands for, ``3MF`` for ``.3mf``."""
    return fmt.lstrip(".").upper()


def _escape(text: str, what: str, *, fmt: str = EXTENSION) -> str:
    """Return ``text`` with the characters XML reserves or normalizes as entities.

    A literal tab, newline or return inside an attribute value is read back
    as a space, and a literal return anywhere as a newline; as character
    references they come back as themselves.

    Raises
    ------
    CodecError
        When ``text`` holds a control character XML 1.0 has no spelling for,
        which would make the model part malformed; ``what`` names the value.
    """
    if _XML_FORBIDDEN.search(text):
        raise CodecError(
            f"{fmt}: {what} {text!r} holds a control character XML cannot carry."
        )
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\t", "&#9;")
        .replace("\n", "&#10;")
        .replace("\r", "&#13;")
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
