"""AMF (Additive Manufacturing File Format) codec - read + write.

An ``.amf`` file is an XML document, or a ZIP archive holding one, that
describes ``<object>`` meshes: a vertex list and one or more ``<volume>``
blocks of triangles, each volume a region of one material. Every object comes
back as a group of triangles tagged with the object's name; colours given on
the object, its volumes, its materials and its triangles are read into one
per-triangle column, and a ``<constellation>`` places its objects under the
rotation and translation of each instance. Writing shares the 3MF mapping:
the tag groups become the objects and the colours go on the triangles.
"""

from __future__ import annotations

import array
from collections.abc import Callable
import dataclasses
import io
import math
from pathlib import PurePosixPath
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from polyxios._io import Source, read_bytes, source_name, write_bytes
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.codecs._3mf import (
    _ZIP_ERRORS,
    _assemble,
    _check_declared_encoding,
    _check_inflation,
    _escape,
    _facet_colors,
    _fallback_scale,
    _partition,
    _refuse_entities,
    _scaled,
    _triangles,
)
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".amf"

_UNITS: frozenset[str] = frozenset({"millimeter", "inch", "feet", "meter", "micron"})
_DEFAULT_UNIT: str = "millimeter"
# Other spellings of the same lengths: 3MF's and DXF's "foot", and the SI
# "micrometer", so a mesh read from another format writes without a warning.
_UNIT_ALIASES: dict[str, str] = {"foot": "feet", "micrometer": "micron"}
# The metadata types the specification lists; any other global attribute is
# polyxios's or another format's and stays out of the file.
_METADATA_TYPES: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "url",
        "author",
        "company",
        "cad",
        "revision",
        "tolerance",
        "volume",
    }
)
_COLOR_KEY: str = "colors"
_NORMAL_KEY: str = "normals"
_ZIP_MAGIC: bytes = b"PK\x03\x04"
_ZIP_EPOCH: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
_PARSE_CHUNK: int = 1 << 20
# Vertices and triangles the constellations may place, per byte of document,
# before the read is refused: an instance is a few dozen bytes however much
# it places. Each placement is charged this much on top of its object's, so
# a chain doubling a one-vertex object counts the work it costs.
_EXPANSION_PER_BYTE: int = 64
_EXPANSION_FLOOR: int = 1 << 22
_PLACEMENT_COST: int = 16
# The material id the specification reserves for void, which a document
# need not define.
_VOID_MATERIAL: str = "0"
# An instance's rotation is spelled in degrees about x, then y, then z.
_ROTATIONS: tuple[str, ...] = ("rx", "ry", "rz")
_AXES: tuple[str, ...] = ("x", "y", "z")
_CORNERS: tuple[str, ...] = ("v1", "v2", "v3")
# What the int64 buffer of a volume's triangles holds.
_INDEX_MIN: int = -(1 << 63)
_INDEX_MAX: int = (1 << 63) - 1
_DELTAS: tuple[str, ...] = ("deltax", "deltay", "deltaz")
# White space at either end of a text that holds a line break is the layout
# of an indented document, not part of a name or a value.
_LAYOUT = re.compile(r"\A\s*[\r\n]\s*|\s*[\r\n]\s*\Z")
# Elements holding one number, which start() has nothing to do for.
_LEAVES: frozenset[str] = frozenset(
    {*_AXES, *_CORNERS, "r", "g", "b", "a", "nx", "ny", "nz", *_DELTAS, *_ROTATIONS}
)


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read an AMF document and return its objects as one triangle mesh.

    Parameters
    ----------
    path
        Path to the ``.amf`` file, plain or zipped, or an open binary file
        object.
    lazy
        Not supported: every number in the document is XML text.

    Returns
    -------
    PolyData
        Every triangle of every object, placed by the constellations when
        the document has any. An object's triangles form an element tag
        group named after its ``name`` metadata, empty or not, or
        ``object_<id>`` when it has none. The
        colour of each triangle - its own, else its volume's, else its
        object's, else its volume's material's - is read into
        ``element_attrs["colors"]`` as RGBA in 0..1, NaN where none is
        given, or where the volume's material is one the document lacks,
        with a warning unless it is the void material ``0``; vertex colours
        and normals land in ``vertex_attrs["colors"]`` and
        ``vertex_attrs["normals"]`` the same way. The document's
        ``unit`` and its metadata land in ``global_attrs``. White space at
        either end of a name or a metadata value is kept, unless it holds a
        line break: that is the indentation of the document, and dropped.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        When the document is not well-formed XML or its root is not an
        ``<amf>``, refused at the root before the rest is parsed, a
        ZIP archive holds no AMF document or one that is encrypted, stored
        in a method ``zipfile`` lacks or inflates the way only a ZIP bomb
        does, a vertex is not one set of three coordinates, a triangle is
        not one each of ``<v1>``, ``<v2>`` and ``<v3>``, a triangle names a
        vertex its object does not hold, an instance an object the document
        lacks, two objects share an id, a constellation contains itself,
        the constellations place a vertex past the float range, or the
        constellations place more vertices and triangles than a
        document of its size - on disk, zipped or not - can describe, each
        placement counted as 16 on top of its object's, and as 16 again in
        each constellation it passes through.
    """
    if lazy:
        raise LazyReadError("AMF is XML text and cannot be memory-mapped.")
    name = source_name(path)
    raw = read_bytes(path)
    # The bytes on disk, not inflated: whitespace deflates a thousandfold.
    budget = max(_EXPANSION_FLOOR, _EXPANSION_PER_BYTE * len(raw))
    if raw.startswith(_ZIP_MAGIC):
        raw = _unzip(raw, name)
    _refuse_entities(raw, repr(name), fmt="AMF")
    _check_declared_encoding(raw, repr(name))
    target = _AmfTarget(name)
    parser = ET.XMLParser(target=target)
    try:
        for start in range(0, len(raw), _PARSE_CHUNK):
            parser.feed(raw[start : start + _PARSE_CHUNK])
        parser.close()
    except (ET.ParseError, LookupError) as exc:
        raise CodecError(f"{name!r} is not well-formed XML: {exc}") from exc
    if target.root is None:
        raise CodecError(f"{name!r} holds no XML element.")
    return _read_document(target, name, budget=budget)


def write(
    poly: PolyData,
    path: Source,
    *,
    unit: str | None = None,
    compress: bool = False,
    **opts: Any,
) -> None:
    """Write a PolyData as an AMF document.

    Parameters
    ----------
    poly
        PolyData to write. Triangles are written as they are; quads,
        pixels, polygons and triangle strips are split into triangles; every
        other element has no AMF triangle and is dropped with a warning.
        Each element tag group becomes an object of its name, the untagged
        triangles one last unnamed object, which reads back as the tag group
        ``object_<id>``. ``element_attrs["colors"]`` goes on the triangles,
        ``vertex_attrs["colors"]`` and ``vertex_attrs["normals"]`` on the
        vertices; a colour channel outside 0..1 is clipped with a warning.
        An object holds the vertices its triangles use, so a vertex no
        triangle uses is not written and one two objects share is written
        in each. A string or number in
        ``global_attrs`` under a metadata type the specification lists -
        ``name``, ``description``, ``url``, ``author``, ``company``,
        ``cad``, ``revision``, ``tolerance``, ``volume`` - is written as
        document metadata, which is text: a number reads back as a string.
        A name or a value whose edge white space holds a line break is
        written with a warning, as the reader drops that white space.
    path
        Output file path, or an open binary file object.
    unit
        The document's unit of length: ``millimeter``, ``inch``, ``feet``,
        ``meter`` or ``micron``. Falls back to ``global_attrs["unit"]``,
        then to millimetres; a mesh unit the format lacks is written as
        millimetres with a warning, the vertices scaled to millimetres when
        that unit is a length polyxios knows - ``centimeter`` read from 3MF,
        or ``yard`` from DXF.
    compress
        Write the document inside a ZIP archive, as the specification
        allows, rather than as plain XML.

    Raises
    ------
    CodecError
        When the mesh has elements but none of them is a surface, a vertex
        a triangle uses is not finite, a tag group name or metadata value
        holds a character XML cannot carry, or ``unit`` is not one the
        format defines.
    """
    unit, scale = _pick_unit(poly, unit)
    tris, source = _triangles(poly, fmt=EXTENSION)
    vertices = _scaled(poly.vertices, scale)
    if tris.size and not np.isfinite(vertices[np.unique(tris)]).all():
        raise CodecError(
            ".amf: a vertex a triangle uses is NaN or infinite; AMF has no"
            " spelling for either."
        )
    colors = _facet_colors(poly, source, fmt=EXTENSION)
    vertex_colors = _vertex_column(poly, _COLOR_KEY, (3, 4))
    colors = _clipped(colors, "element_attrs")
    vertex_colors = _clipped(vertex_colors, "vertex_attrs")
    normals = _vertex_column(poly, _NORMAL_KEY, (3,))
    objects = _partition(poly, source, len(tris), fmt=EXTENSION)
    tags = poly.element_tags or {}
    meta_items = [
        (key, str(value))
        for key, value in (poly.global_attrs or {}).items()
        if key in _METADATA_TYPES
        and isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    ]
    _warn_layout(
        [label for label, _ in objects if label is not None]
        + [value for _, value in meta_items]
    )

    parts: list[str] = []
    for oid, (label, rows) in enumerate(objects, start=1):
        if label is None and f"object_{oid}" in tags:
            warn_caller(
                f".amf: the untagged triangles are object {oid}, which reads"
                f" back as 'object_{oid}' - the name of a tag group; the two"
                " merge on read."
            )
        parts.append(
            _object_xml(
                oid,
                label,
                vertices,
                tris[rows],
                None if colors is None else colors[rows],
                vertex_colors,
                normals,
            )
        )
    metadata = "".join(
        f'  <metadata type="{key}">{_escape(value, key, fmt=EXTENSION)}</metadata>\n'
        for key, value in meta_items
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<amf unit="{unit}" version="1.1">\n'
        f"{metadata}"
        f"{''.join(parts)}"
        "</amf>\n"
    ).encode()
    if compress:
        document = _zip(document, path)
    write_bytes(path, document)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    """Return an element tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _unzip(raw: bytes, name: str) -> bytes:
    """Return the AMF document a ZIP archive holds.

    The specification has the archive hold the one document under any
    name; a member ending in ``.amf`` is taken first, with a warning when
    there are several, then a lone member, refused when it inflates the way
    only a ZIP bomb does.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [i for i in archive.infolist() if not i.is_dir()]
            amf = [i for i in members if i.filename.lower().endswith(".amf")]
            picked = amf or (members if len(members) == 1 else [])
            if not picked:
                raise CodecError(
                    f"{name!r} is a ZIP archive holding no .amf document"
                    f" ({len(members)} member(s))."
                )
            if len(amf) > 1:
                warn_caller(
                    f"{name!r} holds {len(amf)} .amf documents; only"
                    f" {amf[0].filename!r} is read."
                )
            _check_inflation(picked[0], name)
            return archive.read(picked[0])
    except _ZIP_ERRORS as exc:
        raise CodecError(f"{name!r} is not a readable ZIP archive: {exc}") from exc


class _Volume:
    """One ``<volume>``: its material, its colour and its triangles."""

    __slots__ = ("color", "indices", "material", "tri_colors")

    def __init__(self, material: str | None) -> None:
        self.material = material
        self.color: np.ndarray | None = None
        self.indices = array.array("q")
        self.tri_colors: list[tuple[int, np.ndarray]] = []


class _Object:
    """One ``<object>``, gathered as its elements are parsed."""

    __slots__ = (
        "color",
        "coords",
        "label",
        "normals",
        "oid",
        "vertex_colors",
        "volumes",
    )

    def __init__(self, oid: str) -> None:
        self.oid = oid
        self.label: str | None = None
        self.color: np.ndarray | None = None
        self.coords = array.array("d")
        self.vertex_colors: list[tuple[int, np.ndarray]] = []
        self.normals: list[tuple[int, np.ndarray]] = []
        self.volumes: list[_Volume] = []


# One placement: the id of an object or a constellation and the six numbers
# of its instance, deltas first.
_Instance = tuple[str, tuple[float, ...]]


class _AmfTarget:
    """A parser target that reads the document straight into numbers.

    A ``<vertex>`` is seven elements and a ``<triangle>`` four; a tree of
    them costs many times the file, so nothing is built and each leaf's text
    is read off the parser as it closes, by its place in the document. An
    element the reader has no use for - ``<edges>``, ``<texmap>``,
    ``<composite>``, ``<texture>`` - is passed over, its children with it.

    Parameters
    ----------
    name
        The source, for messages.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._stack: list[str] = []
        self._text: list[str] = []
        self._object: _Object | None = None
        self._volume: _Volume | None = None
        self._vertex: dict[str, float] = {}
        self._color: dict[str, float] = {}
        self._normal: dict[str, float] = {}
        self._instance: dict[str, float] = {}
        self._material: str | None = None
        self._constellation: str | None = None
        self._meta_type: str | None = None
        self._vertex_index = 0
        self._tri_index = 0
        self._corners: dict[str, int] = {}
        self._skip_depth = 0
        self._locals: dict[str, str] = {}
        self.root: str | None = None
        self.attrs: dict[str, str] = {}
        self.metadata: list[tuple[str, str]] = []
        self.objects: dict[str, _Object] = {}
        self.materials: dict[str, np.ndarray | None] = {}
        self.constellations: dict[str, list[_Instance]] = {}
        self.order: list[str] = []
        self.formulas = 0
        self.skipped: set[str] = set()

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        local = self._locals.get(tag)
        if local is None:
            local = self._locals[tag] = _local(tag)
        parent = self._stack[-1] if self._stack else None
        self._stack.append(local)
        self._text = []
        if self._skip_depth:
            self._skip_depth += 1
            return
        if parent is None:
            if local != "amf":
                raise CodecError(
                    f"{self._name!r}: the document root is <{local}>, not <amf>."
                )
            self.root = local
            self.attrs = dict(attrs)
            return
        if local in _LEAVES:
            return
        if local == "object" and parent == "amf":
            oid = self._id(attrs, "an <object>", self.objects, "objects")
            self._object = self.objects[oid] = _Object(oid)
            self.order.append(oid)
        elif local == "volume" and self._object is not None and parent == "mesh":
            self._volume = _Volume(attrs.get("materialid"))
            self._object.volumes.append(self._volume)
        elif local == "material" and parent == "amf":
            mid = attrs.get("id")
            if mid is None:
                raise CodecError(f"{self._name!r}: a <material> has no id.")
            if mid in self.materials:
                raise CodecError(f"{self._name!r} gives two materials id {mid}.")
            self._material = mid
            self.materials[mid] = None
        elif local == "constellation" and parent == "amf":
            cid = self._id(
                attrs, "a <constellation>", self.constellations, "constellations"
            )
            self._constellation = cid
            self.constellations[cid] = []
        elif local == "instance" and self._constellation is not None:
            self._instance = {"objectid": attrs.get("objectid")}  # type: ignore[dict-item]
        elif local == "vertex" and self._object is not None:
            self._vertex_index = len(self._object.coords) // 3
        elif local == "triangle" and self._volume is not None:
            self._tri_index = len(self._volume.indices) // 3
            self._corners = {}
        elif local == "metadata":
            self._meta_type = attrs.get("type")
        elif local == "color":
            self._color = {}
        elif local == "coordinates":
            self._vertex = {}
        elif local == "normal":
            self._normal = {}
        elif local in _SKIPPED:
            self.skipped.add(local)
            self._skip_depth = 1

    def _id(self, attrs: dict[str, str], what: str, taken: dict, kind: str) -> str:
        """Return an element's id, refused when missing or already given its kind.

        Objects and constellations are numbered apart: files in the wild give
        a constellation the id of an object, and an instance names the object.
        """
        oid = attrs.get("id")
        if oid is None:
            raise CodecError(f"{self._name!r}: {what} has no id.")
        if oid in taken:
            raise CodecError(f"{self._name!r} gives two {kind} id {oid}.")
        return oid

    def data(self, text: str) -> None:
        self._text.append(text)

    def end(self, tag: str) -> None:
        local = self._stack.pop()
        parent = self._stack[-1] if self._stack else None
        raw = "".join(self._text)
        text = raw.strip()
        self._text = []
        if self._skip_depth:
            self._skip_depth -= 1
            return
        obj = self._object
        if parent == "coordinates" and local in _AXES:
            if local in self._vertex:
                raise CodecError(f"{self._name!r}: a vertex gives <{local}> twice.")
            self._vertex[local] = self._number(text, "a vertex coordinate")
        elif local == "coordinates" and obj is not None:
            if len(self._vertex) != 3:
                raise CodecError(
                    f"{self._name!r}: object {obj.oid} has a vertex that is not"
                    " three coordinates."
                )
            v = self._vertex
            obj.coords.extend((v["x"], v["y"], v["z"]))
        elif local == "vertex" and parent == "vertices" and obj is not None:
            if len(obj.coords) != 3 * (self._vertex_index + 1):
                raise CodecError(
                    f"{self._name!r}: object {obj.oid} has a vertex that is not"
                    " one set of coordinates."
                )
        elif parent == "triangle" and local in _CORNERS:
            if self._volume is not None:
                if local in self._corners:
                    raise CodecError(
                        f"{self._name!r}: object {obj.oid if obj else '?'} has a"
                        f" triangle that gives <{local}> twice."
                    )
                self._corners[local] = self._index(text)
        elif local == "triangle" and self._volume is not None:
            if len(self._corners) != 3:
                raise CodecError(
                    f"{self._name!r}: object {obj.oid if obj else '?'} has a"
                    " triangle that is not three vertex indices."
                )
            c = self._corners
            self._volume.indices.extend((c["v1"], c["v2"], c["v3"]))
        elif parent == "color" and local in ("r", "g", "b", "a"):
            self._color[local] = self._channel(text)
        elif local == "color":
            self._place_color(parent)
        elif parent == "normal" and local in ("nx", "ny", "nz"):
            self._normal[local] = self._number(text, "a normal component")
        elif local == "normal" and parent == "vertex" and obj is not None:
            row = np.array([self._normal.get(k, np.nan) for k in ("nx", "ny", "nz")])
            obj.normals.append((self._vertex_index, row))
        elif local == "metadata":
            self._place_metadata(parent, _LAYOUT.sub("", raw))
        elif parent == "instance" and local in (*_DELTAS, *_ROTATIONS):
            self._instance[local] = self._number(text, f"an instance's {local}")
        elif local == "instance" and self._constellation is not None:
            self._place_instance()
        elif local == "object":
            self._object = None
        elif local == "volume":
            self._volume = None
        elif local == "material":
            self._material = None
        elif local == "constellation":
            self._constellation = None

    def _place_color(self, parent: str | None) -> None:
        """File the colour just closed with the element that holds it."""
        channels = self._color
        if any(math.isnan(channels.get(c, 0.0)) for c in "rgba"):
            rgba = None
        else:
            rgba = np.array(
                [channels.get(c, 0.0) for c in "rgb"] + [channels.get("a", 1.0)]
            )
        obj, volume = self._object, self._volume
        if parent == "triangle" and volume is not None:
            if rgba is not None:
                volume.tri_colors.append((self._tri_index, rgba))
        elif parent == "vertex" and obj is not None:
            if rgba is not None:
                obj.vertex_colors.append((self._vertex_index, rgba))
        elif parent == "volume" and volume is not None:
            volume.color = rgba
        elif parent == "object" and obj is not None:
            obj.color = rgba
        elif parent == "material" and self._material is not None:
            self.materials[self._material] = rgba

    def _place_metadata(self, parent: str | None, text: str) -> None:
        """Keep a document's metadata, and an object's name."""
        kind = self._meta_type
        if parent == "amf" and kind:
            self.metadata.append((kind, text))
        elif parent == "object" and kind == "name" and self._object is not None:
            self._object.label = text

    def _place_instance(self) -> None:
        cid = self._constellation
        assert cid is not None
        target = self._instance.get("objectid")
        if target is None:
            raise CodecError(
                f"{self._name!r}: an instance in constellation {cid} has no objectid."
            )
        numbers = tuple(
            float(self._instance.get(key, 0.0)) for key in (*_DELTAS, *_ROTATIONS)
        )
        self.constellations[cid].append((str(target), numbers))

    def _number(self, text: str, what: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise CodecError(
                f"{self._name!r}: {what} is not a number: {text!r}."
            ) from exc
        if not math.isfinite(value):
            raise CodecError(f"{self._name!r}: {what} is not finite: {text!r}.")
        return value

    def _index(self, text: str) -> int:
        try:
            value = int(text)
        except ValueError as exc:
            raise CodecError(
                f"{self._name!r}: a triangle's vertex index is not an integer:"
                f" {text!r}."
            ) from exc
        if not _INDEX_MIN <= value <= _INDEX_MAX:
            raise CodecError(
                f"{self._name!r}: a triangle names vertex {text[:32]!r}, past"
                " any vertex an object can hold."
            )
        return value

    def _channel(self, text: str) -> float:
        """Return a colour channel, NaN for a formula in x, y and z."""
        try:
            value = float(text)
        except ValueError:
            self.formulas += 1
            return math.nan
        return value if math.isfinite(value) else math.nan

    def close(self) -> None:
        return None


def _read_document(target: _AmfTarget, name: str, *, budget: int) -> PolyData:
    """Assemble the parsed document into one PolyData within ``budget``.

    ``budget`` bounds the vertices and triangles the placements produce.
    """
    if target.formulas:
        warn_caller(
            f"{name!r}: {target.formulas} colour channel(s) are formulas, which"
            " polyxios does not evaluate; those colours read as NaN."
        )
    for kind in sorted(target.skipped):
        warn_caller(f"{name!r}: <{kind}> is not read; " + _SKIPPED[kind])
    unit = target.attrs.get("unit", _DEFAULT_UNIT)
    if _UNIT_ALIASES.get(unit, unit) not in _UNITS:
        warn_caller(
            f"{name!r}: unit {unit!r} is not one AMF defines"
            f" ({', '.join(sorted(_UNITS))}); it is kept as read."
        )
    global_attrs: dict[str, Any] = {**dict(target.metadata), "unit": unit}

    missing: set[str] = set()
    meshes = {
        oid: _mesh_of(obj, target.materials, name, missing=missing)
        for oid, obj in target.objects.items()
    }
    if missing:
        warn_caller(
            f"{name!r}: material(s) {sorted(missing)} are named by a volume but"
            " not defined; those triangles take no colour from them."
        )
    # An overflow is refused below, not warned about by whichever BLAS
    # numpy uses, some of which warn and some of which do not.
    with np.errstate(over="ignore", invalid="ignore"):
        which, frames = _placements(target, name, budget=budget)
        poly = _place_all(target, meshes, which, frames, global_attrs)
    if not np.isfinite(poly.vertices).all():
        raise CodecError(
            f"{name!r}: the constellations place a vertex past the float range."
        )
    return poly


def _place_all(
    target: _AmfTarget,
    meshes: dict[str, tuple[np.ndarray, ...]],
    which: np.ndarray,
    frames: np.ndarray,
    global_attrs: dict[str, Any],
) -> PolyData:
    """Return every placement of every object as one PolyData, in placement order.

    Placement ``p`` puts object ``target.order[which[p]]`` under ``frames[p]``.
    Each object's placements are transformed in one batch and written to the
    rows their place in the order gives them, so the cost is per object, not
    per placement.
    """
    if not len(which):
        return _assemble([], global_attrs)
    oids = target.order
    n_vertices = np.array([len(meshes[oid][0]) for oid in oids], dtype=np.int64)
    n_triangles = np.array([len(meshes[oid][1]) for oid in oids], dtype=np.int64)
    per_v, per_t = n_vertices[which], n_triangles[which]
    v_start = np.cumsum(per_v) - per_v
    t_start = np.cumsum(per_t) - per_t
    vertices = np.empty((int(per_v.sum()), 3))
    placed_oids = [oids[k] for k in np.unique(which).tolist()]
    normals = vcolors = None
    if any(meshes[oid][4] is not None for oid in placed_oids):
        normals = np.full((len(vertices), 3), np.nan)
    if any(meshes[oid][3] is not None for oid in placed_oids):
        vcolors = np.full((len(vertices), 4), np.nan)
    triangles = np.empty((int(per_t.sum()), 3), dtype=np.int64)
    colors = np.empty((len(triangles), 4))
    by_object = np.argsort(which, kind="stable")
    bounds = np.searchsorted(which[by_object], np.arange(len(oids) + 1))
    for k, oid in enumerate(oids):
        rows = by_object[bounds[k] : bounds[k + 1]]
        if not rows.size:
            continue
        local, tris, tri_colors, local_colors, local_normals = meshes[oid]
        frame = frames[rows]
        turn = frame[:, :3, :3].transpose(0, 2, 1)
        vdest = _rows_of(v_start[rows], int(n_vertices[k]))
        tdest = _rows_of(t_start[rows], int(n_triangles[k]))
        moved = np.matmul(local, turn) + frame[:, None, :3, 3]
        vertices[vdest] = moved.reshape(-1, 3)
        if local_normals is not None:
            normals[vdest] = np.matmul(local_normals, turn).reshape(-1, 3)
        if local_colors is not None:
            vcolors[vdest] = np.tile(local_colors, (len(rows), 1))
        triangles[tdest] = (tris[None] + v_start[rows][:, None, None]).reshape(-1, 3)
        colors[tdest] = np.tile(tri_colors, (len(rows), 1))
    poly = _assemble([("", vertices, triangles, colors)], global_attrs)
    labels: dict[str, int] = {}
    named = [target.objects[oid].label for oid in oids]
    label_of = np.array(
        [
            labels.setdefault(f"object_{oid}" if label is None else label, len(labels))
            for oid, label in zip(oids, named, strict=True)
        ],
        dtype=np.int64,
    )
    # Tag groups come in the order their first placement does.
    placed_label = label_of[which]
    present, first = np.unique(placed_label, return_index=True)
    names = list(labels)
    tri_label = np.repeat(placed_label, per_t)
    grouped = np.argsort(tri_label, kind="stable")
    cuts = np.searchsorted(tri_label[grouped], np.arange(len(names) + 1))
    index_dtype = poly.offsets.dtype
    tags = {
        names[li]: grouped[cuts[li] : cuts[li + 1]].astype(index_dtype)
        for li in present[np.argsort(first, kind="stable")].tolist()
    }
    vertex_attrs: dict[str, np.ndarray] = {}
    if vcolors is not None and not np.isnan(vcolors).all():
        vertex_attrs[_COLOR_KEY] = vcolors
    if normals is not None and not np.isnan(normals).all():
        vertex_attrs[_NORMAL_KEY] = normals
    return dataclasses.replace(poly, element_tags=tags, vertex_attrs=vertex_attrs)


_SKIPPED: dict[str, str] = {
    "edges": "curved triangles are read flat, between their corners.",
    "texmap": "texture coordinates are dropped.",
    "texture": "textures are dropped.",
    "composite": "a composite material reads as its own colour only.",
}


def _mesh_of(
    obj: _Object,
    materials: dict[str, np.ndarray | None],
    name: str,
    *,
    missing: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return an object's vertices, triangles, triangle and vertex colours, normals.

    The vertex colours and normals are None when no vertex has one, so a
    document with neither allocates nothing for them.

    A volume naming a material the document lacks takes no colour from it,
    and the id is added to ``missing``; the void material ``0`` needs no
    definition.
    """
    vertices = np.frombuffer(obj.coords, dtype=np.float64).reshape(-1, 3)
    n = len(vertices)
    blocks: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for volume in obj.volumes:
        tris = np.frombuffer(volume.indices, dtype=np.int64).reshape(-1, 3)
        if tris.size and (tris.min() < 0 or tris.max() >= n):
            bad = int(tris.max() if tris.max() >= n else tris.min())
            raise CodecError(
                f"{name!r}: object {obj.oid} names vertex {bad} but holds {n}."
            )
        base = volume.color
        if base is None:
            base = obj.color
        if base is None and volume.material is not None:
            if volume.material in materials:
                base = materials[volume.material]
            elif volume.material != _VOID_MATERIAL:
                missing.add(volume.material)
        rgba = np.full((len(tris), 4), np.nan)
        if base is not None:
            rgba[:] = base
        for index, row in volume.tri_colors:
            rgba[index] = row
        blocks.append(tris)
        colors.append(rgba)
    triangles = np.concatenate(blocks) if blocks else np.empty((0, 3), dtype=np.int64)
    tri_colors = np.concatenate(colors) if colors else np.empty((0, 4))
    vcolors = None
    if obj.vertex_colors:
        vcolors = np.full((n, 4), np.nan)
        for index, row in obj.vertex_colors:
            vcolors[index] = row
    normals = None
    if obj.normals:
        normals = np.full((n, 3), np.nan)
        for index, row in obj.normals:
            normals[index] = row
    return vertices, triangles, tri_colors, vcolors, normals


def _rows_of(starts: np.ndarray, length: int) -> slice | np.ndarray:
    """Return the rows of blocks of ``length`` rows from each of ``starts``.

    One block is a slice, which numpy writes without an index array: a
    document placing one large object once pays nothing for the batching.
    """
    if len(starts) == 1:
        first = int(starts[0])
        return slice(first, first + length)
    return (starts[:, None] + np.arange(length)).ravel()


def _placements(
    target: _AmfTarget, name: str, *, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the index in ``target.order`` and the 4x4 transform of each placement.

    The constellations no instance names are the roots, expanded in
    document order; an object no constellation reaches is placed once where
    it stands, so a document mixing the two loses neither. Every
    constellation is checked for a cycle and what the roots expand to is
    counted - each placement as ``_PLACEMENT_COST`` on top of its object's
    vertices and triangles - before any transform is built, so a chain of
    constellations each placing the next twice is refused rather than
    expanded. Each constellation's placements are then built once, as one
    stack of matrices, from those of the constellations it places; every
    stack is held until the roots are built, so each placement a
    constellation that is not a root holds counts ``_PLACEMENT_COST`` again.
    """
    order = _constellation_order(target, name)
    objects = target.objects
    # Objects and constellations are numbered apart and an instance names
    # the object first, so their costs are kept apart too: a constellation
    # sharing an object's id never stands in for it.
    object_cost = {oid: _object_cost(obj) for oid, obj in objects.items()}
    cost: dict[str, int] = {}
    count: dict[str, int] = {}
    for cid in order:
        instances = target.constellations[cid]
        cost[cid] = sum(
            object_cost[t] if t in objects else cost[t] for t, _ in instances
        )
        count[cid] = sum(1 if t in objects else count[t] for t, _ in instances)
        # A root costs at least what any constellation under it does, so one
        # past the budget is refused here, before a doubling chain grows the
        # sums to integers of as many bits as the chain is long.
        if cost[cid] > budget:
            raise CodecError(
                f"{name!r}: constellation {cid} and those in it place more than {budget}"
                f" vertices and triangles, each placement counted as"
                f" {_PLACEMENT_COST} more, past what a document this size can"
                " describe."
            )
    named = {
        target_id
        for instances in target.constellations.values()
        for target_id, _ in instances
        if target_id not in target.objects
    }
    roots = [cid for cid in target.constellations if cid not in named]
    reached = {
        target_id
        for instances in target.constellations.values()
        for target_id, _ in instances
    }
    loose = [oid for oid in target.order if oid not in reached]
    total = sum(cost[cid] for cid in roots) + sum(object_cost[oid] for oid in loose)
    root_set = set(roots)
    total += _PLACEMENT_COST * sum(count[cid] for cid in order if cid not in root_set)
    if total > budget:
        raise CodecError(
            f"{name!r}: the constellations place more than {budget} vertices"
            f" and triangles, each placement counted as {_PLACEMENT_COST} more,"
            " past what a document this size can describe."
        )
    index = {oid: k for k, oid in enumerate(target.order)}
    placed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cid in order:
        whiches: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        for target_id, numbers in target.constellations[cid]:
            matrix = _instance_matrix(numbers)
            if target_id in target.objects:
                whiches.append(np.array([index[target_id]]))
                frames.append(matrix[None])
            else:
                inner_which, inner_frames = placed[target_id]
                whiches.append(inner_which)
                frames.append(matrix @ inner_frames)
        placed[cid] = _stacked(whiches, frames)
    whiches = [placed[cid][0] for cid in roots]
    frames = [placed[cid][1] for cid in roots]
    if loose:
        whiches.append(np.array([index[oid] for oid in loose]))
        frames.append(np.broadcast_to(np.eye(4), (len(loose), 4, 4)))
    return _stacked(whiches, frames)


def _stacked(
    whiches: list[np.ndarray], frames: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the placements of each block one after the other."""
    if not whiches:
        return np.empty(0, dtype=np.int64), np.empty((0, 4, 4))
    return (
        np.concatenate(whiches).astype(np.int64, copy=False),
        np.concatenate(frames),
    )


def _object_cost(obj: _Object) -> int:
    """Return what placing ``obj`` once counts against the budget.

    Its vertices and its triangles both count: an object of three vertices
    and thousands of triangles costs the triangles every time it is placed.
    """
    triangles = sum(len(volume.indices) for volume in obj.volumes) // 3
    return len(obj.coords) // 3 + triangles + _PLACEMENT_COST


def _constellation_order(target: _AmfTarget, name: str) -> list[str]:
    """Return every constellation after those it places, refusing a cycle."""
    order: list[str] = []
    done: set[str] = set()
    for start in target.constellations:
        if start in done:
            continue
        # Depth-first, without recursion: a chain of constellations as long
        # as the document can spell would overflow the Python stack.
        path: list[str] = [start]
        on_path = {start}
        pending: list[list[str]] = [_children(start, target, name)]
        while path:
            if pending[-1]:
                child = pending[-1].pop()
                if child in on_path:
                    chain = " -> ".join((*path[path.index(child) :], child))
                    raise CodecError(
                        f"{name!r}: constellation {chain} contains itself."
                    )
                if child not in done:
                    path.append(child)
                    on_path.add(child)
                    pending.append(_children(child, target, name))
                continue
            cid = path.pop()
            on_path.discard(cid)
            pending.pop()
            if cid not in done:
                done.add(cid)
                order.append(cid)
    return order


def _children(cid: str, target: _AmfTarget, name: str) -> list[str]:
    """Return the constellations ``cid`` places, refusing an id the document lacks."""
    out: list[str] = []
    for target_id, _ in target.constellations[cid]:
        if target_id in target.objects:
            continue
        if target_id not in target.constellations:
            raise CodecError(
                f"{name!r}: constellation {cid} names object {target_id}, which"
                " the document does not hold."
            )
        out.append(target_id)
    return out


def _instance_matrix(numbers: tuple[float, ...]) -> np.ndarray:
    """Return the 4x4 an instance's deltas and rotations (degrees) spell."""
    dx, dy, dz, rx, ry, rz = numbers
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    about_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    about_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    about_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    out = np.eye(4)
    out[:3, :3] = about_z @ about_y @ about_x
    out[:3, 3] = (dx, dy, dz)
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _pick_unit(poly: PolyData, unit: str | None) -> tuple[str, float]:
    """Return the unit to write and the factor the vertices are scaled by.

    The argument wins, then the mesh's unit, then the default; a mesh unit
    AMF lacks is written as millimetres, the vertices scaled to match when
    it is a length polyxios knows.
    """
    if unit is not None:
        unit = _UNIT_ALIASES.get(unit, unit)
        if unit not in _UNITS:
            raise CodecError(
                f".amf: {unit!r} is not an AMF unit; one of {sorted(_UNITS)} is."
            )
        return unit, 1.0
    stored = (poly.global_attrs or {}).get("unit", _DEFAULT_UNIT)
    known = _UNIT_ALIASES.get(stored, stored) if isinstance(stored, str) else stored
    if known in _UNITS:
        return known, 1.0
    return _DEFAULT_UNIT, _fallback_scale(stored, EXTENSION, sorted(_UNITS))


def _vertex_column(
    poly: PolyData, key: str, widths: tuple[int, ...]
) -> np.ndarray | None:
    """Return a numeric per-vertex column as float64, or None when absent or unfit.

    An integer colour column counts 0..255 and is scaled to 0..1, and a
    three-column one is made opaque; a column of the wrong shape is warned
    about and not written.
    """
    stored = (poly.vertex_attrs or {}).get(key)
    if stored is None:
        return None
    values = np.asarray(stored)
    if (
        values.ndim != 2
        or values.shape[0] != len(poly.vertices)
        or values.shape[1] not in widths
        or values.dtype.kind not in "fiub"
    ):
        shape = " or ".join(str(w) for w in widths)
        warn_caller(
            f".amf: vertex_attrs['{key}'] is not {shape} components per vertex;"
            " it was not written."
        )
        return None
    out = values.astype(np.float64)
    if key == _COLOR_KEY:
        if values.dtype.kind in "iu":
            out = out / 255.0
        if out.shape[1] == 3:
            out = np.hstack([out, np.ones((len(out), 1))])
    return out


def _clipped(rows: np.ndarray | None, what: str) -> np.ndarray | None:
    """Return colours with every finite channel outside 0..1 clipped, with a warning.

    Parameters
    ----------
    rows
        The RGBA of each triangle or vertex, or None for none.
    what
        The attribute mapping the colours come from, for the message.
    """
    if rows is None:
        return None
    outside = (rows < 0.0) | (rows > 1.0)
    if not outside.any():
        return rows
    outside &= np.isfinite(rows)
    count = int(np.count_nonzero(outside))
    if not count:
        return rows
    warn_caller(
        f".amf: {what}['{_COLOR_KEY}'] holds {count} channel(s) outside 0..1;"
        " they were clipped, as AMF colours run 0..1."
    )
    return np.where(outside, np.clip(rows, 0.0, 1.0), rows)


def _warn_layout(texts: list[str]) -> None:
    """Warn about names and metadata whose edge white space a reader drops."""
    lost = [text for text in texts if _LAYOUT.search(text)]
    if lost:
        warn_caller(
            f".amf: {lost} start or end with white space holding a line break,"
            " which reads back as the indentation of the document and is"
            " dropped."
        )


def _color_xml(rgba: list[float]) -> str:
    r, g, b, a = rgba
    return f"<color><r>{r!r}</r><g>{g!r}</g><b>{b!r}</b><a>{a!r}</a></color>"


def _normal_xml(normal: list[float]) -> str:
    nx, ny, nz = normal
    return f"<normal><nx>{nx!r}</nx><ny>{ny!r}</ny><nz>{nz!r}</nz></normal>"


def _spelled(
    rows: np.ndarray | None, *, n: int, spell: Callable[[list[float]], str]
) -> list[str]:
    """Return each row of ``rows`` as ``spell`` writes it, ``""`` where not finite.

    Parameters
    ----------
    rows
        The values, one row per vertex or triangle, or None for none.
    n
        The number of rows to return when ``rows`` is None.
    spell
        Turns one row, as a list of floats, into its XML.
    """
    if rows is None:
        return [""] * n
    finite = np.isfinite(rows).all(axis=1).tolist()
    values = rows.astype(np.float64).tolist()
    return [spell(row) if ok else "" for row, ok in zip(values, finite, strict=True)]


def _object_xml(
    oid: int,
    label: str | None,
    vertices: np.ndarray,
    tris: np.ndarray,
    colors: np.ndarray | None,
    vertex_colors: np.ndarray | None,
    normals: np.ndarray | None,
) -> str:
    """Return one ``<object>`` holding ``tris`` over the vertices they use."""
    used, local = np.unique(tris, return_inverse=True)
    local = local.reshape(tris.shape)
    coords = vertices[used].astype(np.float64).tolist()
    vc = _spelled(
        None if vertex_colors is None else vertex_colors[used],
        n=len(used),
        spell=_color_xml,
    )
    nm = _spelled(
        None if normals is None else normals[used], n=len(used), spell=_normal_xml
    )
    lines = [
        "        <vertex><coordinates>"
        f"<x>{x!r}</x><y>{y!r}</y><z>{z!r}</z></coordinates>{c}{n}</vertex>\n"
        for (x, y, z), c, n in zip(coords, vc, nm, strict=True)
    ]
    tc = _spelled(colors, n=len(local), spell=_color_xml)
    tri_lines = [
        f"        <triangle>{color}<v1>{a}</v1><v2>{b}</v2><v3>{c}</v3></triangle>\n"
        for (a, b, c), color in zip(local.tolist(), tc, strict=True)
    ]
    name = (
        f'    <metadata type="name">{_escape(label, "the tag group name", fmt=EXTENSION)}'
        "</metadata>\n"
        if label is not None
        else ""
    )
    return (
        f'  <object id="{oid}">\n'
        f"{name}"
        "    <mesh>\n"
        "      <vertices>\n"
        f"{''.join(lines)}"
        "      </vertices>\n"
        "      <volume>\n"
        f"{''.join(tri_lines)}"
        "      </volume>\n"
        "    </mesh>\n"
        "  </object>\n"
    )


def _zip(document: bytes, path: Source) -> bytes:
    """Wrap the document in the ZIP archive a compressed AMF is."""
    name = source_name(path)
    # A nameless buffer is "<buffer>", a name no file system on Windows takes.
    stem = "" if name == "<buffer>" else PurePosixPath(name.replace("\\", "/")).stem
    stem = stem or "model"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        info = zipfile.ZipInfo(f"{stem}.amf", date_time=_ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, document)
    return buf.getvalue()
