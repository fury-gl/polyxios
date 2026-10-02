"""COLLADA (``.dae``): the XML scene interchange format, as a SceneData.

A COLLADA document is a set of libraries - geometries, effects, materials,
images, controllers, animations, visual scenes - tied together by ``#id``
references, and a ``<scene>`` naming the visual scene to show. Each
``<geometry>`` holds a ``<mesh>`` of ``<source>`` arrays and primitive
blocks (``<triangles>``, ``<polylist>``, ``<lines>``, ...) whose ``<p>``
index tuples name one entry of each ``<input>`` per corner, so a corner
carries its own normal and texture coordinate and a vertex two corners
disagree about is split here, as OBJ readers do. ``<node>`` trees carry
transforms as ``<matrix>``, ``<translate>``, ``<rotate>`` and ``<scale>``
elements in document order, instance geometries, controllers (skins) and
other nodes; ``<animation>`` channels target a node's transform element
by the ``sid`` it declares.

Read gives a :class:`~polyxios.SceneData`: one PolyData per geometry with
``normals``, ``texcoords`` and ``colors`` as vertex attributes and
``element_attrs["material"]`` as indices into the scene's materials;
effects as PBR-ish :class:`~polyxios.SceneMaterial` entries whose phong
pieces ride in ``extras``; nodes with their local matrix and the transform
elements they were spelled with in ``extras["transforms"]``, so an
animation still finds its target on write; animations under
``global_attrs["animations"]``, in the shape glTF uses. Skins are not read
or written yet. Write spells the scene back in 1.4.1 syntax with fixed
timestamps, so the same scene always writes the same bytes.
"""

from __future__ import annotations

import base64
import codecs
import dataclasses
from datetime import datetime
import re
from typing import Any
from urllib.parse import unquote_to_bytes
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, is_buffer, read_bytes, source_name, write_text
from polyxios._scene import (
    SceneData,
    SceneImage,
    SceneMaterial,
    SceneNode,
    SceneTexture,
)
from polyxios._trs import trs_of_matrix
from polyxios._types import PolyData
from polyxios._warn import warn_caller as _warn
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header
from polyxios.version import version as __version__

EXTENSION: str = ".dae"
LABEL: str = "COLLADA"

_NS = "http://www.collada.org/2005/11/COLLADASchema"

# How much of the document is handed to expat at a time. Fed whole, expat
# grows one buffer to hold it and cannot grow it past 1 GiB.
_PARSE_CHUNK: int = 1 << 20
# expat reads UTF-8, UTF-16 and Latin-1 itself; any other declared encoding
# is transcoded here first.
_DECLARED_ENCODING = re.compile(
    rb"(\A(?:\xef\xbb\xbf)?<\?xml\b[^>]*?\bencoding\s*=\s*[\"'])"
    rb"([A-Za-z][\w.\-]*)([\"'][^>]*\?>)"
)
# Entities are refused whatever expat's amplification limits, which a
# Python linked against an older system expat lacks.
_ENTITY_DECLS = tuple(
    "<!ENTITY".encode(enc) for enc in ("utf-8", "utf-16-le", "utf-16-be")
)

_XML_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
# XML's NameStartChar and NameChar, less the colon an NCName forbids;
# ``\w`` admits digits and marks of other scripts that no name starts with.
_NAME_START = (
    "A-Z_a-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02ff\u0370-\u037d"
    "\u037f-\u1fff\u200c\u200d\u2070-\u218f\u2c00-\u2fef\u3001-\ud7ff"
    "\uf900-\ufdcf\ufdf0-\ufffd\U00010000-\U000effff"
)
_NAME_REST = _NAME_START + "0-9\u00b7\u0300-\u036f\u203f\u2040\\-"
_NCNAME = re.compile(f"[{_NAME_START}][{_NAME_REST}.]*\\Z")
# A transform sid ends a channel target, where a dot starts the member.
_TRANSFORM_SID = re.compile(f"[{_NAME_START}][{_NAME_REST}]*\\Z")
_NMTOKEN = re.compile(f"[{_NAME_REST}.]+\\Z")
# The member part of a channel target: ``.X`` / ``.ANGLE`` or ``(i)(j)``.
_MEMBER = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|(?:\(\d+\))+)\Z")
# A parser folds a literal newline, tab or return in an attribute to a
# space; only the character reference survives a round trip.
_ATTR_ENTITIES = {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"}

_VERSION = re.compile(r"1\.[45](?:\.|\Z)")
# The ``<contributor>`` children kept, in the schema's order.
_CONTRIBUTOR_KEYS = ("author", "authoring_tool", "copyright")
# Asset keys a write spells; ``version`` and ``generator`` are another
# format's own, which the file written here replaces.
_ASSET_KEYS = frozenset(
    {
        "up_axis",
        "unit",
        "created",
        "modified",
        "version",
        "generator",
        *_CONTRIBUTOR_KEYS,
    }
)
_UP_AXES = ("X_UP", "Y_UP", "Z_UP")
# The extras a write spells; ``camera`` and ``light`` are dropped with their
# own warning.
_NODE_EXTRAS = frozenset({"id", "sid", "type", "transforms", "camera", "light"})
_MATERIAL_EXTRAS = frozenset(
    {
        "shading",
        "ambient",
        "specular",
        "reflective",
        "shininess",
        "reflectivity",
        "index_of_refraction",
        "transparent_mode",
    }
)

# A fixed timestamp keeps two writes of one scene byte-identical.
_EPOCH = "1970-01-01T00:00:00Z"
# xs:dateTime, which ISO 8601 and ``datetime.fromisoformat`` exceed.
_DATE_TIME = re.compile(
    r"-?\d{4,}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)?\Z"
)
# Every character a document may hold that ASCII spells, to test whether an
# encoding spells them as ASCII does.
_ASCII_PROBE = "".join(map(chr, (9, 10, 13, *range(32, 127))))

# ``<instance_node>`` copies a subtree each time it appears, so a few nested
# instances spell exponentially many nodes; the tree may grow to one node
# per eight bytes of document, and never less than this.
_MIN_NODE_CAP: int = 1 << 18
# A geometry bound to other materials by another instance gets a material
# column per binding, so a few instances of one large mesh can spell many;
# they may hold sixteen bytes per byte of document, and never less than this.
_MIN_COPY_CAP: int = 1 << 28

_TRI = ELEMENT_TYPES["triangle"]
_QUAD = ELEMENT_TYPES["quad"]
_POLYGON = ELEMENT_TYPES["polygon"]
_LINE = ELEMENT_TYPES["line"]
_POLY_LINE = ELEMENT_TYPES["poly_line"]
_STRIP = ELEMENT_TYPES["triangle_strip"]
_VERTEX = ELEMENT_TYPES["vertex"]

# Which primitive block each element type is written in.
_BLOCK_OF_TYPE = {
    _TRI: "triangles",
    _QUAD: "polylist",
    _POLYGON: "polylist",
    _LINE: "lines",
    _POLY_LINE: "linestrips",
    _STRIP: "tristrips",
}

_BLOCKS = ("triangles", "polylist", "lines", "linestrips", "tristrips")
# The vertex count each written element type allows, low and high.
_CORNERS = {
    _TRI: (3, 3),
    _QUAD: (4, 4),
    _POLYGON: (3, np.iinfo(np.int64).max),
    _LINE: (2, 2),
    _POLY_LINE: (2, np.iinfo(np.int64).max),
    _STRIP: (3, np.iinfo(np.int64).max),
}
_BLOCK_INDEX = {code: _BLOCKS.index(block) for code, block in _BLOCK_OF_TYPE.items()}

_WRAP_TO_GL = {
    "WRAP": 10497,
    "MIRROR": 33648,
    "CLAMP": 33071,
    "BORDER": 33069,
    "NONE": 33071,
}
_GL_TO_WRAP = {10497: "WRAP", 33648: "MIRROR", 33071: "CLAMP", 33069: "BORDER"}
_FILTER_TO_GL = {
    "NEAREST": 9728,
    "LINEAR": 9729,
    "NEAREST_MIPMAP_NEAREST": 9984,
    "LINEAR_MIPMAP_NEAREST": 9985,
    "NEAREST_MIPMAP_LINEAR": 9986,
    "LINEAR_MIPMAP_LINEAR": 9987,
}
_GL_TO_FILTER = {v: k for k, v in _FILTER_TO_GL.items()}

_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tga": "image/x-tga",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "ktx": "image/ktx",
    "ktx2": "image/ktx2",
    "dds": "image/vnd-ms.dds",
}
_EXT_OF_MEDIA = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif"}
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

_PATH_OF_KIND = {
    "matrix": "matrix",
    "translate": "translation",
    "rotate": "rotation",
    "scale": "scale",
    "lookat": "lookat",
}
_KIND_OF_PATH = {v: k for k, v in _PATH_OF_KIND.items()}
_KIND_SIZE = {"matrix": 16, "translate": 3, "rotate": 4, "scale": 3, "lookat": 9}

_PARAM_WIDTH = {
    "float2": 2,
    "float3": 3,
    "float4": 4,
    "float2x2": 4,
    "float3x3": 9,
    "float4x4": 16,
}

_ARRAY_KINDS = frozenset(
    {"float_array", "int_array", "Name_array", "IDREF_array", "SIDREF_array"}
)

_SHADINGS = ("phong", "lambert", "blinn", "constant")
# Of the terms a material's extras may carry, the ones each shading holds.
_SHADING_TERMS = {
    "phong": ("ambient", "specular", "shininess"),
    "blinn": ("ambient", "specular", "shininess"),
    "lambert": ("ambient",),
    "constant": (),
}
_OPAQUE_MODES = ("A_ONE", "A_ZERO", "RGB_ZERO", "RGB_ONE")
_INTERPOLATIONS = ("LINEAR", "BEZIER", "HERMITE", "CARDINAL", "BSPLINE", "STEP")
_MATERIAL_TEXTURES = ("base_color_texture", "normal_texture", "emissive_texture")

# Paths of a channel without a sid that are baked into ``<matrix>`` keys, and
# the interpolations such a channel may use.
_BAKED_PATHS = ("translation", "rotation", "scale")
_BAKED_INTERPOLATIONS = ("LINEAR", "STEP", "CUBICSPLINE")
# COLLADA blends a matrix element by element, which shrinks a node rotating
# between two keys; keys this close in angle keep that below 1%.
_BAKE_MAX_DEGREES: float = 15.0
_BAKE_CUBIC_PIECES: int = 4
_BAKE_MAX_PASSES: int = 8

_IDENTITY = np.eye(4, dtype=np.float64)


# =============================================================================
# read
# =============================================================================


def read_scene(path: Source, **opts: Any) -> SceneData:
    """Read a COLLADA document and return its scene.

    Parameters
    ----------
    path
        Path or open binary file object of a ``.dae`` document.
    **opts
        Accepted for interface symmetry and ignored.

    Returns
    -------
    SceneData
        One mesh per ``<geometry>``, in library order (every
        ``<library_geometries>`` of the document, as with the other
        libraries), its ``name`` as ``global_attrs["mesh_name"]``, then a
        variant of a geometry, sharing its arrays under another ``material``
        column, for each further way its instances resolve its symbols; the
        visual scenes' node trees with the instanced one active, effects as
        materials with their textures and images, and animations under
        ``global_attrs["animations"]`` (each named by its ``name``, else its
        ``id``; a channel addressing a node instanced several times gets a
        target per copy). A document
        whose visual scenes hold no node (or that has none) gets one root
        node per geometry, all in one scene, so its meshes still show.
        COLLADA has no alpha mask, so ``alpha_mode`` is ``BLEND`` when the
        material's alpha is below one and ``OPAQUE`` otherwise. Positions,
        vertex attributes, matrices and sampler keys are float64 whatever precision or
        array type the file's text carries. Every ``<vertices>`` position is
        a vertex, whether or not a primitive names it. A mesh without
        texture coordinate set 0 has its sets renumbered down from the
        lowest, so its first set is ``texcoords``; a ``P`` column that is
        zero throughout is dropped.

    Raises
    ------
    CodecError
        When the document is not well-formed XML, declares an encoding
        Python does not know or an XML entity (in its own encoding or once
        transcoded), its root is not ``<COLLADA>`` of version 1.4 or 1.5,
        an array's ``count`` disagrees with its content, an accessor reads
        past its array, an index tuple names an entry a source lacks, a
        ``<p>`` or ``<vcount>`` length disagrees with the block's ``count``,
        an input inside ``<vertices>`` has a row count other than the
        positions', a ``#url`` names an id the document does not hold or an
        ``<instance_node>`` names one that is not a node, a transform
        element has the wrong number of values, a node instances itself,
        the node tree nests deeper than the interpreter's recursion limit,
        ``<instance_node>`` expands it past one node per eight bytes of
        document (and past 262144 nodes), the material columns of
        geometries bound to other materials pass sixteen bytes per byte of
        document (and 256 MiB), an animation sampler lacks ``INPUT`` or
        ``OUTPUT``, its values or tangents do not number its keys, a channel
        names a source that is not a sampler, or channels addressing
        instanced nodes expand past the node cap.

    Warns
    -----
    UserWarning
        For what is not read: skins (the mesh is read in its bind shape,
        one warning counting the skinned instances). For an animation
        channel whose target reaches no node transform (it is dropped), a
        sampler mixing interpolations (the first is taken for all keys) or
        naming one the specification does not (read as ``LINEAR``). For
        metadata read leniently:
        an ``up_axis`` that is not one of the three the specification
        allows is dropped, a unit's meter spelled with a decimal comma is
        read as a point, a morph controller contributes only its base
        geometry, a unit's meter that is not a finite positive number is
        read as 1, an alpha that is not finite is read as opaque, an
        instance of a node, geometry, controller, effect or visual scene in another document is dropped (a material without
        its effect is a bare phong one, and the first visual scene is
        active), a node's second ``<instance_camera>`` or
        ``<instance_light>`` is dropped for the first, and an
        ``<instance_material>`` symbol bound twice keeps its first binding.
        An ``<input>`` without a ``semantic`` is dropped. A material symbol
        an instance's binding omits takes the material of that id. An image
        the reader would have to create (``<create_2d>``, ...) is empty.
    """
    name = source_name(path)
    raw = read_bytes(path)
    root = _parse(raw, name)
    doc = _Doc(root=root, name=name, size=len(raw), ids=_ids(root))

    global_attrs: dict[str, Any] = {}
    asset = _read_asset(doc)
    if asset:
        global_attrs["asset"] = asset

    images = _read_images(doc)
    image_index: dict[str, int] = {}
    for i, img_id in enumerate(images[1]):
        if img_id:
            image_index.setdefault(img_id, i)
    tex = _Textures(image_index=image_index)
    materials, material_index = _read_materials(doc, tex)

    geometries = _library(root, "library_geometries", "geometry")
    meshes: list[_Mesh | None] = [None] * len(geometries)
    # Keyed by element, not id: a url resolves to the first element holding
    # an id, and a second geometry reusing it must not take its instances.
    slot_of_geometry = {id(geo): i for i, geo in enumerate(geometries)}
    st = _State(
        doc=doc,
        geometries=geometries,
        slot_of_geometry=slot_of_geometry,
        meshes=meshes,
        material_index=material_index,
        max_nodes=max(_MIN_NODE_CAP, len(raw) // 8),
        max_copied=max(_MIN_COPY_CAP, 16 * len(raw)),
    )
    try:
        scenes, active, scene_name = _read_scenes(st)
    except RecursionError as exc:
        raise CodecError(
            f"{name!r}: the node tree nests deeper than this reader follows."
        ) from exc

    for i in range(len(meshes)):
        _resolve_deferred(st, i)
    if not st.nodes:
        st.nodes = [SceneNode(mesh=i) for i in range(len(meshes))]
        scenes = (tuple(range(len(meshes))),) if meshes else ()
        active = 0
    polys = (
        *(_finish_mesh(st, m) for m in meshes if m is not None),
        *st.variants,
    )

    if st.skinned:
        _warn(
            f"{name!r}: skins are not read; {st.skinned} skinned mesh "
            "instance(s) are read in their bind shape, without joints or "
            "weights."
        )
    animations = _read_animations(st)
    if animations:
        global_attrs["animations"] = animations

    return SceneData(
        meshes=polys,
        nodes=tuple(st.nodes),
        materials=materials,
        textures=tuple(tex.textures),
        images=images[0],
        scenes=scenes,
        active_scene=active,
        name=scene_name,
        global_attrs=global_attrs,
    )


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a COLLADA document flattened to one PolyData.

    Parameters
    ----------
    path
        Path or open binary file object of a ``.dae`` document.
    lazy
        Not supported: the geometry is XML text.
    **opts
        Accepted for interface symmetry and ignored.

    Returns
    -------
    PolyData
        Every instanced mesh of the active scene under its node's world
        transform, merged.

    Warns
    -----
    UserWarning
        Always: the scene graph, materials and animations are dropped;
        :func:`read_scene` keeps them.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        Whatever :func:`read_scene` refuses.
    """
    if lazy:
        raise LazyReadError("COLLADA is XML text and cannot be memory-mapped.")
    _warn(
        f"'{source_name(path)}' is a scene format (COLLADA): read() flattens "
        "the scene graph, materials and animations into a single "
        "PolyData. Use polyxios.read_scene() to preserve the full scene."
    )
    return read_scene(path).to_polydata()


# -----------------------------------------------------------------------------
# document plumbing
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class _Doc:
    """The parsed document and the lookups every phase needs."""

    root: ET.Element
    name: str
    size: int
    ids: dict[str, ET.Element]
    arrays: dict[int, np.ndarray | list[str]] = dataclasses.field(default_factory=dict)
    warned: set[str] = dataclasses.field(default_factory=set)
    handed: set[int] = dataclasses.field(default_factory=set)


def _refuse_entities(raw: bytes, name: str) -> None:
    if any(decl in raw for decl in _ENTITY_DECLS):
        raise CodecError(
            f"{name!r} declares an XML entity, which COLLADA never needs; "
            "refused rather than expanded."
        )


def _parse(raw: bytes, name: str) -> ET.Element:
    _refuse_entities(raw, name)
    try:
        try:
            root = _feed(raw)
        except (ValueError, LookupError):
            # An encoding such as UTF-7 hides a declaration from the check
            # on the raw bytes.
            utf8 = _as_utf8(raw, name)
            _refuse_entities(utf8, name)
            root = _feed(utf8)
    except ET.ParseError as exc:
        raise CodecError(f"{name!r} is not well-formed XML: {exc}") from exc
    if _local(root.tag) != "COLLADA":
        raise CodecError(
            f"{name!r}: the document root is <{_local(root.tag)}>, not <COLLADA>."
        )
    version = root.get("version", "")
    if not _VERSION.match(version):
        raise CodecError(
            f"{name!r}: COLLADA version {version!r} is not one this reader "
            "knows (1.4 or 1.5)."
        )
    return root


def _feed(raw: bytes) -> ET.Element:
    parser = ET.XMLParser()
    for start in range(0, len(raw), _PARSE_CHUNK):
        parser.feed(raw[start : start + _PARSE_CHUNK])
    return parser.close()


def _as_utf8(raw: bytes, name: str) -> bytes:
    """Transcode a document in an encoding expat cannot read to UTF-8.

    Parameters
    ----------
    raw
        The document's bytes.
    name
        The document's name, for messages.

    Returns
    -------
    bytes
        The document in UTF-8, its declaration naming UTF-8.

    Raises
    ------
    CodecError
        When the declared encoding is unknown or the bytes do not decode in
        it.
    """
    match = _DECLARED_ENCODING.match(raw)
    encoding = match.group(2).decode("ascii") if match else None
    try:
        if encoding is None:
            raise LookupError("no encoding declared")
        text = raw[match.end(0) :].decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise CodecError(
            f"{name!r} is not well-formed XML: encoding {encoding!r} cannot "
            f"be read ({exc})."
        ) from exc
    head = match.group(1) + b"UTF-8" + match.group(3)
    return head.removeprefix(b"\xef\xbb\xbf") + text.encode("utf-8")


def _ids(root: ET.Element) -> dict[str, ET.Element]:
    ids: dict[str, ET.Element] = {}
    for elem in root.iter():
        eid = elem.get("id")
        if eid is not None:
            ids.setdefault(eid, elem)
    return ids


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(elem: ET.Element | None, tag: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [child for child in elem if _local(child.tag) == tag]


def _child(elem: ET.Element | None, tag: str) -> ET.Element | None:
    if elem is None:
        return None
    for child in elem:
        if _local(child.tag) == tag:
            return child
    return None


def _library(root: ET.Element, library: str, tag: str) -> list[ET.Element]:
    """Return the ``tag`` entries of every ``<library>`` of the root, in document order."""
    return [entry for lib in _children(root, library) for entry in _children(lib, tag)]


def _ref(doc: _Doc, url: str | None, what: str) -> ET.Element:
    """Return the element a ``#id`` reference names."""
    if url is None:
        raise CodecError(f"{doc.name!r}: {what} has no url.")
    if not url.startswith("#"):
        raise CodecError(
            f"{doc.name!r}: {what} names {url!r}, an external document, "
            "which this reader does not follow."
        )
    elem = doc.ids.get(url[1:])
    if elem is None:
        raise CodecError(
            f"{doc.name!r}: {what} names {url!r}, which the document does not hold."
        )
    return elem


def _instanced(doc: _Doc, inst: ET.Element, what: str) -> ET.Element | None:
    """Return the element an ``<instance_*>`` names, None for another document's.

    An instance of an element in another document is legal COLLADA that
    this reader cannot follow; it is dropped rather than failing the scene.
    """
    url = inst.get("url")
    if url is not None and not url.startswith("#"):
        _warn(
            f"{doc.name!r}: {what} names {url!r}, an external document, which "
            "this reader does not follow; the instance is dropped."
        )
        return None
    return _ref(doc, url, what)


def _int(text: str | None, what: str, *, default: int | None = None) -> int:
    if text is None:
        if default is None:
            raise CodecError(f"{what} is missing.")
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise CodecError(f"{what} is {text!r}, not an integer.") from exc
    if value < 0:
        raise CodecError(f"{what} is {value}, which is negative.")
    return value


def _ints(text: str | None, what: str) -> np.ndarray:
    try:
        return np.array((text or "").split(), dtype=np.int64)
    except (ValueError, OverflowError) as exc:
        raise CodecError(f"{what} is not integers.") from exc


def _floats(text: str | None, what: str, *, n: int | None = None) -> np.ndarray:
    values = (text or "").split()
    if n is not None and len(values) != n:
        raise CodecError(f"{what} needs {n} numbers, got {len(values)}.")
    try:
        return np.array(values, dtype=np.float64)
    except ValueError as exc:
        raise CodecError(f"{what} is not numbers: {(text or '')[:60]!r}.") from exc


def _array(doc: _Doc, elem: ET.Element, what: str) -> np.ndarray | list[str]:
    """Return the values of a ``<float_array>``, ``<int_array>`` or name array."""
    cached = doc.arrays.get(id(elem))
    if cached is not None:
        return cached
    kind = _local(elem.tag)
    count = _int(elem.get("count"), f"{doc.name!r}: the count of {what}")
    if count > doc.size:
        raise CodecError(
            f"{doc.name!r}: {what} declares count={count}, more values than "
            f"the file has bytes ({doc.size})."
        )
    tokens = (elem.text or "").split()
    if len(tokens) != count:
        raise CodecError(
            f"{doc.name!r}: {what} declares count={count} but holds "
            f"{len(tokens)} values."
        )
    if kind not in _ARRAY_KINDS:
        raise CodecError(
            f"{doc.name!r}: {what} is a <{kind}>, which holds neither numbers "
            "nor names an input of this reader reads."
        )
    values: np.ndarray | list[str]
    if kind == "float_array":
        try:
            values = np.array(tokens, dtype=np.float64)
        except ValueError as exc:
            raise CodecError(
                f"{doc.name!r}: {what} is not numbers: {(elem.text or '')[:60]!r}."
            ) from exc
    elif kind == "int_array":
        try:
            values = np.array(tokens, dtype=np.int64)
        except (ValueError, OverflowError) as exc:
            raise CodecError(
                f"{doc.name!r}: {what} is not integers: {(elem.text or '')[:60]!r}."
            ) from exc
    else:
        values = tokens
    doc.arrays[id(elem)] = values
    return values


def _accessor(doc: _Doc, src: ET.Element, what: str) -> tuple[ET.Element, ET.Element]:
    """Return the ``<accessor>`` of a ``<source>`` and the array element it reads."""
    sid = src.get("id", "?")
    acc = _child(_child(src, "technique_common"), "accessor")
    if acc is None:
        raise CodecError(f"{doc.name!r}: source '{sid}' ({what}) has no <accessor>.")
    return acc, _ref(doc, acc.get("source"), f"the accessor of source '{sid}'")


def _source(doc: _Doc, src: ET.Element, what: str) -> np.ndarray | list[str]:
    """Return a ``<source>`` as its accessor reads it: floats ``(count, k)`` or names.

    The accessor may start past the array's head (``offset``) and step
    wider than the components it names (``stride``), as two accessors
    interleaved over one array do, so the bound is the last value a row
    touches, not ``offset + count * stride``.
    """
    sid = src.get("id", "?")
    acc, arr_elem = _accessor(doc, src, what)
    count = _int(acc.get("count"), f"{doc.name!r}: the count of accessor '{sid}'")
    stride = _int(acc.get("stride"), f"{doc.name!r}: stride of '{sid}'", default=1)
    offset = _int(acc.get("offset"), f"{doc.name!r}: offset of '{sid}'", default=0)
    if stride == 0:
        raise CodecError(f"{doc.name!r}: the accessor of source '{sid}' has stride 0.")
    values = _array(doc, arr_elem, f"the array of source '{sid}'")
    used: list[int] = []
    at = 0
    for p in _children(acc, "param"):
        width = _PARAM_WIDTH.get(p.get("type", ""), 1)
        if p.get("name"):
            used.extend(range(at, at + width))
        at += width
    # ``cols`` is None when every column of the stride is read: a stride is
    # file text, and spelling its columns out would allocate before the
    # bound below is checked.
    cols: list[int] | None = None
    width = stride
    if isinstance(values, list):
        width = 1
    elif used and max(used) < stride and len(used) != stride:
        cols = used
        width = max(used) + 1
    need = offset + (count - 1) * stride + width if count else 0
    if need > len(values):
        raise CodecError(
            f"{doc.name!r}: the accessor of source '{sid}' reads {need} values "
            f"from an array of {len(values)}."
        )
    if isinstance(values, list):
        return [values[offset + i * stride] for i in range(count)]
    if cols is None:
        return values[offset:need].reshape(count, stride)
    return values[offset + np.arange(count)[:, None] * stride + np.array(cols)]


def _float_source(doc: _Doc, src: ET.Element, what: str) -> np.ndarray:
    table = _source(doc, src, what)
    if isinstance(table, list):
        raise CodecError(
            f"{doc.name!r}: source '{src.get('id', '?')}' ({what}) holds names, "
            "not numbers."
        )
    return table.astype(np.float64, copy=False)


def _name_source(doc: _Doc, src: ET.Element, what: str) -> list[str]:
    table = _source(doc, src, what)
    if not isinstance(table, list):
        raise CodecError(
            f"{doc.name!r}: source '{src.get('id', '?')}' ({what}) holds numbers, "
            "not names."
        )
    return table


# -----------------------------------------------------------------------------
# asset, images, effects, materials
# -----------------------------------------------------------------------------


def _read_asset(doc: _Doc) -> dict[str, Any]:
    asset = _child(doc.root, "asset")
    out: dict[str, Any] = {}
    if asset is None:
        return out
    up = _child(asset, "up_axis")
    if up is not None:
        value = (up.text or "").strip()
        if value in _UP_AXES:
            out["up_axis"] = value
        else:
            _warn(
                f"{doc.name!r}: up_axis is {value!r}, not one of "
                f"{', '.join(_UP_AXES)}; it is dropped."
            )
    unit = _child(asset, "unit")
    if unit is not None:
        spelled = unit.get("meter", "1")
        if "," in spelled and "." not in spelled:
            # Some exporters write the unit in the machine's locale.
            _warn(
                f"{doc.name!r}: the unit's meter is spelled {spelled!r} with a "
                "decimal comma; it is read as a point."
            )
            spelled = spelled.replace(",", ".")
        try:
            meter = float(_floats(spelled, "meter", n=1)[0])
        except CodecError:
            meter = float("nan")
        if not (np.isfinite(meter) and meter > 0.0):
            _warn(
                f"{doc.name!r}: the unit's meter is {spelled!r}, not a finite "
                "positive number; it is read as 1."
            )
            meter = 1.0
        out["unit"] = {"name": unit.get("name", "meter"), "meter": meter}
    contributor = _child(asset, "contributor")
    for key in _CONTRIBUTOR_KEYS:
        elem = _child(contributor, key)
        if elem is not None and elem.text and elem.text.strip():
            out[key] = elem.text.strip()
    for key in ("created", "modified"):
        elem = _child(asset, key)
        if elem is not None and elem.text:
            out[key] = elem.text.strip()
    return out


def _media_type(uri: str) -> str | None:
    ext = uri.rsplit(".", 1)[-1].lower() if "." in uri else ""
    return _MEDIA_TYPES.get(ext)


def _read_images(doc: _Doc) -> tuple[tuple[SceneImage, ...], list[str]]:
    images: list[SceneImage] = []
    ids: list[str] = []
    # COLLADA 1.4 also lets an effect or its profile hold its own <image>.
    local = [
        e
        for fx in _library(doc.root, "library_effects", "effect")
        for e in fx.iter()
        if _local(e.tag) == "image"
    ]
    for img in (*_library(doc.root, "library_images", "image"), *local):
        init = _child(img, "init_from")
        name = img.get("name", "")
        image = SceneImage(name=name)
        inline = _child(img, "data")
        if init is None and inline is not None:
            data = _hex_bytes(doc, inline, img.get("id", "?"), "data")
            image = SceneImage(
                data=data, media_type=_sniff_media(data) or None, name=name
            )
        elif init is None:
            made = [
                _local(c.tag)
                for c in img
                if _local(c.tag) in ("create_2d", "create_3d", "create_cube")
            ]
            if made:
                _warn(
                    f"{doc.name!r}: image '{img.get('id', '?')}' is a <{made[0]}>, "
                    "which this reader does not decode; it is an empty image."
                )
        if init is not None:
            ref = _child(init, "ref")
            hexed = _child(init, "hex")
            if hexed is not None:
                data = _hex_bytes(doc, hexed, img.get("id", "?"), "hex")
                fmt = hexed.get("format", "").lower()
                image = SceneImage(
                    data=data, media_type=_MEDIA_TYPES.get(fmt), name=name
                )
            else:
                uri = ((ref.text if ref is not None else init.text) or "").strip()
                if uri:
                    image = _image_of_uri(doc, uri, name, img.get("id", "?"))
        images.append(image)
        ids.append(img.get("id", ""))
    return tuple(images), ids


def _hex_bytes(doc: _Doc, elem: ET.Element, img_id: str, tag: str) -> bytes:
    try:
        return bytes.fromhex("".join((elem.text or "").split()))
    except ValueError as exc:
        raise CodecError(
            f"{doc.name!r}: image '{img_id}' holds <{tag}> that is not "
            f"hexadecimal: {exc}"
        ) from exc


def _image_of_uri(doc: _Doc, uri: str, name: str, img_id: str) -> SceneImage:
    if uri[:5].lower() == "data:":
        header, sep, payload = uri[5:].partition(",")
        if not sep:
            raise CodecError(
                f"{doc.name!r}: image '{img_id}' has a data URI without a comma."
            )
        media, *params = header.split(";")
        if "base64" in (p.strip().lower() for p in params):
            try:
                data = base64.b64decode("".join(payload.split()), validate=True)
            except ValueError as exc:
                raise CodecError(
                    f"{doc.name!r}: image '{img_id}' has a data URI that does not "
                    f"decode: {exc}"
                ) from exc
        else:
            data = unquote_to_bytes(payload)
        return SceneImage(data=data, media_type=media or None, name=name)
    return SceneImage(uri=uri, media_type=_media_type(uri), name=name)


def _color_of(elem: ET.Element | None, what: str) -> tuple[float, ...] | None:
    color = _child(elem, "color")
    if color is None:
        return None
    values = _floats(color.text, what)
    if len(values) not in (3, 4):
        raise CodecError(f"{what} needs 3 or 4 numbers, got {len(values)}.")
    if len(values) == 3:
        values = np.append(values, 1.0)
    return tuple(float(v) for v in values)


def _float_of(elem: ET.Element | None, what: str) -> float | None:
    f = _child(elem, "float")
    if f is None:
        return None
    return float(_floats(f.text, what, n=1)[0])


@dataclasses.dataclass
class _Textures:
    """The texture table a read builds, deduplicated by settings."""

    image_index: dict[str, int]
    textures: list[SceneTexture] = dataclasses.field(default_factory=list)
    index: dict[SceneTexture, int] = dataclasses.field(default_factory=dict)


def _texture_of(
    doc: _Doc,
    effect: ET.Element,
    params: dict[str | None, ET.Element],
    elem: ET.Element | None,
    tex_table: _Textures,
) -> int | None:
    """Resolve a ``<texture texture=sid>`` through sampler and surface to a texture index."""
    tex = _child(elem, "texture")
    if tex is None:
        return None
    ref = tex.get("texture", "")
    sampler = _child(params.get(ref), "sampler2D")
    image_id = ref
    settings: dict[str, Any] = {}
    if sampler is not None:
        source = _child(sampler, "source")
        surface_sid = (source.text or "").strip() if source is not None else ""
        surface = _child(params.get(surface_sid), "surface")
        init = _child(surface, "init_from")
        image_id = (init.text or "").strip() if init is not None else ""
        inst = _child(sampler, "instance_image")
        if inst is not None:
            image_id = inst.get("url", "").removeprefix("#")
        for key, table in (("wrap_s", _WRAP_TO_GL), ("wrap_t", _WRAP_TO_GL)):
            node = _child(sampler, key)
            if node is not None and (node.text or "").strip() in table:
                settings[key] = table[(node.text or "").strip()]
        for key, tag in (("min_filter", "minfilter"), ("mag_filter", "magfilter")):
            node = _child(sampler, tag)
            if node is not None and (node.text or "").strip() in _FILTER_TO_GL:
                settings[key] = _FILTER_TO_GL[(node.text or "").strip()]
    if image_id not in tex_table.image_index:
        _warn(
            f"{doc.name!r}: effect '{effect.get('id', '?')}' names texture "
            f"{ref!r}, which reaches no image; the texture is dropped."
        )
        return None
    texture = SceneTexture(image=tex_table.image_index[image_id], **settings)
    slot = tex_table.index.get(texture)
    if slot is None:
        slot = len(tex_table.textures)
        tex_table.textures.append(texture)
        tex_table.index[texture] = slot
    return slot


def _luminance(rgb: tuple[float, ...]) -> float:
    return 0.212671 * rgb[0] + 0.715160 * rgb[1] + 0.072169 * rgb[2]


def _read_effect(
    doc: _Doc, effect: ET.Element, name: str, tex_table: _Textures
) -> SceneMaterial:
    profile = _child(effect, "profile_COMMON")
    technique = _child(profile, "technique")
    shading_elem = None
    for child in technique if technique is not None else ():
        if _local(child.tag) in _SHADINGS:
            shading_elem = child
            break
    what = f"{doc.name!r}: effect '{effect.get('id', '?')}'"
    extras: dict[str, Any] = {
        "shading": _local(shading_elem.tag) if shading_elem is not None else "phong"
    }
    if shading_elem is None:
        return SceneMaterial(name=name, metallic=0.0, extras=extras)

    # Effect-scope params, then the common profile's, which shadow them; a
    # GLSL or CG profile's param of the same sid is not the common one's.
    params = {p.get("sid"): p for p in _children(effect, "newparam")}
    params.update(
        (p.get("sid"), p) for p in profile.iter() if _local(p.tag) == "newparam"
    )
    diffuse = _child(shading_elem, "diffuse")
    base = _color_of(diffuse, f"{what} diffuse") or (1.0, 1.0, 1.0, 1.0)
    base_tex = _texture_of(doc, effect, params, diffuse, tex_table)
    emission = _child(shading_elem, "emission")
    emissive_color = _color_of(emission, f"{what} emission")
    emissive_tex = _texture_of(doc, effect, params, emission, tex_table)
    # A PBR emissive factor scales its texture; black would switch it off.
    unset = (1.0, 1.0, 1.0, 1.0) if emissive_tex is not None else (0.0, 0.0, 0.0, 1.0)
    emissive = (emissive_color or unset)[:3]
    for key in ("ambient", "specular", "reflective"):
        color = _color_of(_child(shading_elem, key), f"{what} {key}")
        if color is not None:
            extras[key] = color
    for key in ("shininess", "reflectivity", "index_of_refraction"):
        value = _float_of(_child(shading_elem, key), f"{what} {key}")
        if value is not None:
            extras[key] = value

    transparent = _child(shading_elem, "transparent")
    mode = transparent.get("opaque", "A_ONE") if transparent is not None else "A_ONE"
    if mode not in _OPAQUE_MODES:
        _warn(
            f"{what} has transparent opaque={mode!r}, which the specification does "
            "not name; it is read as A_ONE."
        )
        mode = "A_ONE"
    tcolor = _color_of(transparent, f"{what} transparent")
    amount = _float_of(_child(shading_elem, "transparency"), f"{what} transparency")
    alpha = base[3]
    if transparent is not None or amount is not None:
        amount = 1.0 if amount is None else amount
        if tcolor is None:
            tcolor = (0.0, 0.0, 0.0, 1.0)
        if mode == "A_ONE" and amount == 0.0 and tcolor == (1.0, 1.0, 1.0, 1.0):
            # SketchUp, Google Earth and Blender before 2.8 spell an opaque
            # material this way; read literally it would be invisible.
            if "opaque" not in doc.warned:
                doc.warned.add("opaque")
                _warn(
                    f"{what} has A_ONE transparency 0 with a white transparent "
                    "colour, the exporter convention for opaque; it and any "
                    "other such effect of the document are read as opaque."
                )
            amount = 1.0
        if mode == "A_ONE":
            alpha = tcolor[3] * amount
        elif mode == "A_ZERO":
            alpha = 1.0 - tcolor[3] * amount
        elif mode == "RGB_ZERO":
            alpha = 1.0 - _luminance(tcolor) * amount
        elif mode == "RGB_ONE":
            alpha = _luminance(tcolor) * amount
        alpha *= base[3]
        extras["transparent_mode"] = mode
    if not np.isfinite(alpha):
        _warn(f"{what} has an alpha that is not finite; it is read as opaque.")
        alpha = 1.0
    alpha = min(max(alpha, 0.0), 1.0)

    normal_tex = None
    for extra in effect.iter():
        if _local(extra.tag) == "bump":
            normal_tex = _texture_of(doc, effect, params, extra, tex_table)
            break
    double_sided = any(
        _local(e.tag) == "double_sided" and (e.text or "").strip() in ("1", "true")
        for e in effect.iter()
    )
    return SceneMaterial(
        name=name,
        base_color=(base[0], base[1], base[2], alpha),
        metallic=0.0,
        roughness=1.0,
        emissive=tuple(emissive),
        alpha_mode="BLEND" if alpha < 1.0 else "OPAQUE",
        double_sided=double_sided,
        base_color_texture=base_tex,
        normal_texture=normal_tex,
        emissive_texture=emissive_tex,
        extras=extras,
    )


def _read_materials(
    doc: _Doc, tex_table: _Textures
) -> tuple[tuple[SceneMaterial, ...], dict[str, int]]:
    materials: list[SceneMaterial] = []
    index: dict[str, int] = {}
    for mat in _library(doc.root, "library_materials", "material"):
        mid = mat.get("id", "")
        inst = _child(mat, "instance_effect")
        name = mat.get("name", "")
        effect = (
            _instanced(doc, inst, f"material '{mid}'") if inst is not None else None
        )
        if effect is None:
            materials.append(
                SceneMaterial(name=name, metallic=0.0, extras={"shading": "phong"})
            )
        else:
            materials.append(_read_effect(doc, effect, name, tex_table))
        if mid:
            index.setdefault(mid, len(materials) - 1)
    return tuple(materials), index


# -----------------------------------------------------------------------------
# geometry
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class _Mesh:
    """One geometry, read once and shared by every node instancing it.

    ``key`` is the material each symbol resolves to under the first
    instance's binding, which its own ``materials`` hold; ``variants`` maps
    every other resolution seen to the scene mesh index of its variant.
    """

    poly: PolyData
    positions_of: np.ndarray
    n_positions: int
    symbols: list[tuple[str | None, int]]
    gid: str
    materials: np.ndarray | None = None
    key: tuple[int, ...] | None = None
    variants: dict[tuple[int, ...], int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _State:
    """What the node walk accumulates.

    ``variants`` holds a mesh for each extra material resolution, scene
    mesh ``len(meshes) + k``; ``copied`` counts the bytes of their material
    columns against ``max_copied``.
    """

    doc: _Doc
    geometries: list[ET.Element]
    slot_of_geometry: dict[int, int]
    meshes: list[_Mesh | None]
    material_index: dict[str, int]
    max_nodes: int
    max_copied: int
    nodes: list[SceneNode | None] = dataclasses.field(default_factory=list)
    nodes_by_id: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    variants: list[PolyData] = dataclasses.field(default_factory=list)
    copied: int = 0
    skinned: int = 0


def _prim_rows(
    doc: _Doc, prim: ET.Element, stride: int, gid: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (corners, sizes, types) for one primitive block."""
    tag = _local(prim.tag)
    what = f"{doc.name!r}: <{tag}> of geometry '{gid}'"
    count = _int(prim.get("count"), f"{what} count")

    def ints(text: str | None, label: str) -> np.ndarray:
        return _ints(text, f"{what}: {label}")

    p_elem = _child(prim, "p")
    if tag in ("triangles", "lines"):
        per = 3 if tag == "triangles" else 2
        p = ints(p_elem.text if p_elem is not None else "", "<p>")
        if len(p) != count * per * stride:
            raise CodecError(
                f"{what} declares count={count}, which needs {count * per * stride} "
                f"indices, but its <p> holds {len(p)}."
            )
        sizes = np.full(count, per, dtype=np.int64)
        types = np.full(count, _TRI if per == 3 else _LINE, dtype=np.uint8)
        return p.reshape(-1, stride), sizes, types

    if tag == "polylist":
        vc = _child(prim, "vcount")
        vcount = ints(vc.text if vc is not None else "", "<vcount>")
        if len(vcount) != count:
            raise CodecError(
                f"{what} declares count={count} but its <vcount> holds {len(vcount)}."
            )
        if len(vcount) and vcount.min() < 1:
            raise CodecError(f"{what} has a <vcount> entry below 1.")
        p = ints(p_elem.text if p_elem is not None else "", "<p>")
        if len(vcount) and int(vcount.max()) * stride > len(p):
            raise CodecError(
                f"{what} has a <vcount> entry of {int(vcount.max())} corners, more "
                f"than its <p> of {len(p)} indices holds."
            )
        need = int(vcount.sum()) * stride
        if len(p) != need:
            raise CodecError(
                f"{what}: <vcount> sums to {need} indices but <p> holds {len(p)}."
            )
        return p.reshape(-1, stride), vcount, _types_of_sizes(vcount)

    blocks: list[np.ndarray] = []
    holes = False
    for child in prim:
        kind = _local(child.tag)
        if kind == "p":
            blocks.append(ints(child.text, "<p>"))
        elif kind == "ph":
            outer = _child(child, "p")
            if outer is None:
                raise CodecError(f"{what} has a <ph> without a <p>.")
            holes = True
            blocks.append(ints(outer.text, "<ph><p>"))
    if len(blocks) != count:
        raise CodecError(
            f"{what} declares count={count} but holds {len(blocks)} <p> blocks."
        )
    if holes:
        _warn(f"{what} has polygons with holes; the holes are dropped.")
    for block in blocks:
        if len(block) % stride:
            raise CodecError(
                f"{what}: a <p> of {len(block)} indices is not a whole number of "
                f"{stride}-index corners."
            )
    corners = [b.reshape(-1, stride) for b in blocks]
    sizes = np.array([len(c) for c in corners], dtype=np.int64)
    if tag == "polygons":
        if len(sizes) and sizes.min() < 1:
            raise CodecError(f"{what} has an empty <p>.")
        types = _types_of_sizes(sizes)
    elif tag == "linestrips":
        if len(sizes) and sizes.min() < 2:
            raise CodecError(f"{what} has a strip of fewer than 2 corners.")
        types = np.full(len(sizes), _POLY_LINE, dtype=np.uint8)
    elif tag == "tristrips":
        if len(sizes) and sizes.min() < 3:
            raise CodecError(f"{what} has a strip of fewer than 3 corners.")
        types = np.full(len(sizes), _STRIP, dtype=np.uint8)
    else:  # trifans
        if len(sizes) and sizes.min() < 3:
            raise CodecError(f"{what} has a fan of fewer than 3 corners.")
        flat = np.concatenate(corners) if corners else np.zeros((0, stride), np.int64)
        n_tris = sizes - 2
        starts = np.cumsum(sizes) - sizes
        fan = np.repeat(np.arange(len(sizes)), n_tris)
        j = np.arange(len(fan)) - np.repeat(np.cumsum(n_tris) - n_tris, n_tris)
        head = starts[fan]
        tris = np.stack([head, head + 1 + j, head + 2 + j], axis=1).ravel()
        corners = [flat[tris]]
        sizes = np.full(len(fan), 3, dtype=np.int64)
        types = np.full(len(sizes), _TRI, dtype=np.uint8)
    stacked = np.concatenate(corners) if corners else np.zeros((0, stride), np.int64)
    return stacked, sizes, types


def _types_of_sizes(sizes: np.ndarray) -> np.ndarray:
    types = np.full(len(sizes), _POLYGON, dtype=np.uint8)
    types[sizes == 3] = _TRI
    types[sizes == 4] = _QUAD
    types[sizes == 2] = _LINE
    types[sizes == 1] = _VERTEX
    return types


def _attr_name(semantic: str, set_no: str) -> str:
    """Return the vertex attribute key of an input, its set spelled as a number."""
    base = {"NORMAL": "normals", "COLOR": "colors", "TEXCOORD": "texcoords"}.get(
        semantic, semantic.lower()
    )
    set_no = set_no.strip()
    if set_no.isascii() and set_no.isdigit():
        set_no = str(int(set_no))
    return base if set_no in ("", "0") else f"{base}_{set_no}"


def _tidy_texcoords(attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Name a mesh's lowest texture coordinate set ``texcoords``, and drop an empty P.

    Some exporters number their first map channel 1, which would leave a
    mesh with ``texcoords_1`` and no ``texcoords`` for a material to sample.
    A ``P`` column that is zero throughout is a UV set spelled as UVW.

    Parameters
    ----------
    attrs
        The mesh's vertex attributes, keyed as :func:`_attr_name` spells.

    Returns
    -------
    dict
        The same attributes in their order; without a set 0, every texcoord
        set is renumbered down by the lowest one.
    """
    sets = sorted(
        int(k.removeprefix("texcoords_"))
        for k in attrs
        if k.startswith("texcoords_") and k.removeprefix("texcoords_").isdigit()
    )
    shift = sets[0] if sets and "texcoords" not in attrs else 0
    rename = {f"texcoords_{k}": _attr_name("TEXCOORD", str(k - shift)) for k in sets}
    out: dict[str, np.ndarray] = {}
    for key, values in attrs.items():
        if key == "texcoords" or key in rename:
            if values.shape[1] == 3 and not values[:, 2].any():
                values = values[:, :2]
        out[rename.get(key, key)] = values
    return out


def _read_geometry(doc: _Doc, geo: ET.Element, slot: int) -> _Mesh:
    gid = geo.get("id") or f"geometry_{slot}"
    label = geo.get("name")
    global_attrs = {"mesh_name": label} if label else {}
    mesh = _child(geo, "mesh")
    if mesh is None:
        kinds = [_local(c.tag) for c in geo if _local(c.tag) not in ("asset", "extra")]
        _warn(
            f"{doc.name!r}: geometry '{gid}' holds <{kinds[0] if kinds else '?'}>, "
            "which this reader does not read; it is an empty mesh."
        )
        empty = dataclasses.replace(_empty(), global_attrs=global_attrs)
        return _Mesh(empty, np.zeros(0, np.int64), 0, [], gid)

    vertices = _child(mesh, "vertices")
    if vertices is None:
        raise CodecError(f"{doc.name!r}: geometry '{gid}' has no <vertices>.")
    positions = None
    per_position: list[tuple[str, np.ndarray]] = []
    source_of: dict[str, ET.Element] = {}
    for inp in _children(vertices, "input"):
        semantic = inp.get("semantic", "")
        if not semantic:
            _warn_unnamed_input(doc, f"<vertices> of geometry '{gid}'")
            continue
        src = _ref(doc, inp.get("source"), f"the {semantic} input of '{gid}'")
        table = _float_source(doc, src, f"{semantic} of '{gid}'")
        if semantic == "POSITION":
            if table.shape[1] != 3:
                raise CodecError(
                    f"{doc.name!r}: geometry '{gid}' has POSITION with "
                    f"{table.shape[1]} components, not 3."
                )
            positions = table
            source_of[""] = src
        else:
            attr = _attr_name(semantic, inp.get("set", ""))
            if attr not in source_of:
                per_position.append((attr, table))
                source_of[attr] = src
    if positions is None:
        raise CodecError(f"{doc.name!r}: geometry '{gid}' has no POSITION input.")
    n_pos = len(positions)
    for attr, table in per_position:
        if len(table) != n_pos:
            raise CodecError(
                f"{doc.name!r}: geometry '{gid}' gives {attr} inside <vertices> "
                f"with {len(table)} rows for {n_pos} positions."
            )

    # One column per distinct (semantic, set, source): two blocks naming the
    # same semantic from different sources index different tables.
    columns: list[tuple[str, np.ndarray]] = [("", positions)]
    column_of: dict[tuple[str, str, int], int] = {}
    corner_blocks: list[np.ndarray] = []
    sizes_blocks: list[np.ndarray] = []
    types_blocks: list[np.ndarray] = []
    symbols: list[tuple[str | None, int]] = []
    n_tokens = 0
    for prim in mesh:
        tag = _local(prim.tag)
        if tag not in (
            "triangles",
            "polylist",
            "polygons",
            "lines",
            "linestrips",
            "tristrips",
            "trifans",
        ):
            continue
        where = f"<{tag}> of geometry '{gid}'"
        what = f"{doc.name!r}: {where}"
        inputs = _children(prim, "input")
        offsets = [
            _int(i.get("offset"), f"{what} input offset", default=0) for i in inputs
        ]
        stride = max(offsets, default=-1) + 1
        vertex_input = None
        mapping: list[tuple[int, int, int]] = []  # (p column, mesh column, count)
        in_block: set[str] = set()
        for inp, offset in zip(inputs, offsets, strict=True):
            semantic = inp.get("semantic", "")
            if not semantic:
                _warn_unnamed_input(doc, what.removeprefix(f"{doc.name!r}: "))
                continue
            if semantic == "VERTEX":
                src = _ref(doc, inp.get("source"), f"{where} VERTEX input")
                if src is not vertices:
                    raise CodecError(
                        f"{what} names {inp.get('source')!r} as VERTEX, which is not "
                        "the mesh's <vertices>."
                    )
                vertex_input = offset
                mapping.append((offset, 0, n_pos))
                continue
            src = _ref(doc, inp.get("source"), f"{where} {semantic} input")
            key = (semantic, inp.get("set", ""), id(src))
            attr = _attr_name(semantic, inp.get("set", ""))
            if attr in in_block:
                _warn(
                    f"{what} gives {attr} twice (input {semantic} set "
                    f"{inp.get('set', '')!r}); the first is kept."
                )
                continue
            in_block.add(attr)
            col = column_of.get(key)
            if col is None:
                table = _float_source(doc, src, f"{semantic} of '{gid}'")
                if stride == 1 and len(table) == n_pos:
                    # Every input shares the one index, so this is a value per
                    # position, and the vertices stay the positions themselves.
                    # A later block naming another source for the same
                    # attribute takes the column path and splits its corners.
                    known = source_of.get(attr)
                    if known is None:
                        per_position.append((attr, table))
                        source_of[attr] = src
                    if known is None or known is src:
                        continue
                columns.append((attr, table))
                col = len(columns) - 1
                column_of[key] = col
            mapping.append((offset, col, len(columns[col][1])))
        if vertex_input is None:
            raise CodecError(f"{what} has no VERTEX input.")
        corners, sizes, types = _prim_rows(doc, prim, stride, gid)
        n_tokens += corners.size
        wide = np.full((len(corners), len(columns)), -1, dtype=np.int64)
        for p_col, mesh_col, limit in mapping:
            idx = corners[:, p_col]
            if len(idx) and (idx.min() < 0 or idx.max() >= limit):
                bad = int(idx.min()) if idx.min() < 0 else int(idx.max())
                raise CodecError(
                    f"{what} names entry {bad} of a source holding {limit} "
                    f"({columns[mesh_col][0] or 'POSITION'})."
                )
            wide[:, mesh_col] = idx
        corner_blocks.append(wide)
        sizes_blocks.append(sizes)
        types_blocks.append(types)
        symbols.append((prim.get("material"), len(sizes)))

    n_elements = sum(len(s) for s in sizes_blocks)
    validate_header(n_pos, n_elements, n_tokens, doc.size)

    if corner_blocks:
        width = max(b.shape[1] for b in corner_blocks)
        all_corners = np.concatenate(
            [
                np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=-1)
                for b in corner_blocks
            ]
        )
        sizes = np.concatenate(sizes_blocks)
        types = np.concatenate(types_blocks)
    else:
        all_corners = np.zeros((0, len(columns)), dtype=np.int64)
        sizes = np.zeros(0, dtype=np.int64)
        types = np.zeros(0, dtype=np.uint8)

    if len(columns) == 1:
        connectivity = all_corners[:, 0]
        positions_of = np.arange(n_pos, dtype=np.int64)
        verts = _own(doc, source_of[""], positions)
        attrs = {name: _own(doc, source_of[name], t) for name, t in per_position}
    else:
        unique, first, inverse = np.unique(
            all_corners, axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        unique = unique[order]
        connectivity = rank[inverse.ravel()]
        positions_of = unique[:, 0]
        verts = positions[positions_of]
        attrs = {name: table[positions_of] for name, table in per_position}
        covered: dict[str, np.ndarray] = {}
        for col in range(1, len(columns)):
            name, table = columns[col]
            idx = unique[:, col]
            have = idx >= 0
            if name in attrs:
                previous = attrs[name]
                if previous.shape[1] != table.shape[1]:
                    _warn(
                        f"{doc.name!r}: geometry '{gid}' gives {name} two widths; "
                        "the later source is dropped, and a corner only it gives "
                        f"{name} to keeps the earlier source's value, or zero."
                    )
                else:
                    previous[have] = table[idx[have]]
            else:
                values = np.zeros((len(unique), table.shape[1]), dtype=np.float64)
                values[have] = table[idx[have]]
                attrs[name] = values
                covered[name] = np.zeros(len(unique), dtype=bool)
            if name in covered:
                covered[name] |= have
        partial = [name for name, seen in covered.items() if not seen.all()]
        if partial:
            _warn(
                f"{doc.name!r}: geometry '{gid}' gives {partial} in some primitive "
                "blocks but not others; the corners without one are zero."
            )
        seen = np.zeros(n_pos, dtype=bool)
        seen[positions_of] = True
        unused = np.flatnonzero(~seen)
        if len(unused):
            # A position no corner names is still a vertex of the mesh, as it
            # is when no input splits corners.
            by_position = dict(per_position)
            positions_of = np.concatenate([positions_of, unused])
            verts = positions[positions_of]
            for name, values in attrs.items():
                table = by_position.get(name)
                tail = (
                    table[unused]
                    if table is not None and table.shape[1] == values.shape[1]
                    else np.zeros((len(unused), values.shape[1]), dtype=values.dtype)
                )
                attrs[name] = np.concatenate([values, tail])

    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    poly = PolyData(
        vertices=np.ascontiguousarray(verts, dtype=np.float64),
        connectivity=connectivity.astype(np.int32),
        offsets=offsets.astype(np.int32),
        element_types=types,
        vertex_attrs={
            k: np.ascontiguousarray(v) for k, v in _tidy_texcoords(attrs).items()
        },
        global_attrs=global_attrs,
    )
    return _Mesh(poly, positions_of, n_pos, symbols, gid)


def _warn_unnamed_input(doc: _Doc, where: str) -> None:
    """Warn that an ``<input>`` without a ``semantic`` is dropped.

    Parameters
    ----------
    doc
        The document being read.
    where
        The element holding the input, for the message.
    """
    _warn(
        f"{doc.name!r}: an <input> of the {where} has no semantic, which "
        "names no attribute; it is dropped."
    )


def _own(doc: _Doc, src: ET.Element, table: np.ndarray) -> np.ndarray:
    """Return ``table`` as is the first time its array is handed out, a copy after.

    A geometry or sampler keeps its ``<source>`` tables as views of the
    parsed arrays; a second reader of the same ``<float_array>``, through
    the same ``<source>`` or another one over it, gets its own memory so
    editing one mesh or animation leaves the others alone.
    """
    key = id(_accessor(doc, src, "")[1])
    if key in doc.handed:
        return table.copy()
    doc.handed.add(key)
    return table


def _empty() -> PolyData:
    return PolyData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        connectivity=np.zeros(0, dtype=np.int32),
        offsets=np.zeros(1, dtype=np.int32),
        element_types=np.zeros(0, dtype=np.uint8),
    )


def _mesh_at(st: _State, slot: int, bound: dict[str, int] | None) -> int:
    """Return the scene mesh index of geometry ``slot`` under ``bound``.

    The geometry is read on first use. Its first instance's binding resolves
    its own materials; an instance whose binding resolves its symbols to
    other materials gets a variant: the same arrays, PolyData being
    immutable, under its own ``material`` column.
    """
    mesh = st.meshes[slot]
    if mesh is None:
        mesh = _read_geometry(st.doc, st.geometries[slot], slot)
        st.meshes[slot] = mesh
    if not any(sym is not None for sym, _ in mesh.symbols):
        return slot
    key = _resolved(st, mesh, bound or {})
    if mesh.key is None:
        mesh.key = key
        mesh.materials = _resolve_symbols(st, mesh, bound or {})
        return slot
    if key == mesh.key:
        return slot
    if key not in mesh.variants:
        size = 4 * sum(n for _, n in mesh.symbols)
        if st.copied + size > st.max_copied:
            raise CodecError(
                f"{st.doc.name!r}: instances binding geometry '{mesh.gid}' to "
                f"other materials need material columns past {st.max_copied} "
                f"bytes, more than a document of {st.doc.size} bytes is read "
                "into."
            )
        st.copied += size
        materials = _resolve_symbols(st, mesh, bound or {})
        st.variants.append(
            dataclasses.replace(
                mesh.poly,
                vertex_attrs=dict(mesh.poly.vertex_attrs),
                element_attrs={**mesh.poly.element_attrs, "material": materials},
                vertex_tags=dict(mesh.poly.vertex_tags),
                element_tags=dict(mesh.poly.element_tags),
                global_attrs=dict(mesh.poly.global_attrs),
            )
        )
        mesh.variants[key] = len(st.meshes) + len(st.variants) - 1
    return mesh.variants[key]


def _resolved(st: _State, mesh: _Mesh, bound: dict[str, int]) -> tuple[int, ...]:
    """Return the material each of ``mesh``'s symbols resolves to under ``bound``."""
    out = []
    for sym, _ in mesh.symbols:
        idx = bound.get(sym) if sym is not None else None
        if idx is None and sym is not None:
            idx = st.material_index.get(sym)
        out.append(-1 if idx is None else idx)
    return tuple(out)


def _resolve_deferred(st: _State, slot: int) -> None:
    """Read a geometry no node instances and resolve its symbols by material id."""
    mesh = st.meshes[slot]
    if mesh is None:
        mesh = _read_geometry(st.doc, st.geometries[slot], slot)
        st.meshes[slot] = mesh
    if mesh.materials is None and any(sym is not None for sym, _ in mesh.symbols):
        mesh.materials = _resolve_symbols(st, mesh, {})


def _resolve_symbols(st: _State, mesh: _Mesh, bound: dict[str, int]) -> np.ndarray:
    """Return the material index of every element, one lookup per distinct symbol."""
    index_of: dict[str | None, int] = {None: -1}
    missing: list[str] = []
    guessed: list[str] = []
    for sym, _ in mesh.symbols:
        if sym in index_of:
            continue
        idx = bound.get(sym)
        if idx is None:
            idx = st.material_index.get(sym)
            if idx is not None and bound:
                guessed.append(sym)
        if idx is None:
            idx = -1
            missing.append(sym)
        index_of[sym] = idx
    per_run = np.array([index_of[sym] for sym, _ in mesh.symbols], dtype=np.int32)
    out = np.repeat(per_run, [n for _, n in mesh.symbols])
    if missing:
        _warn(
            f"{st.doc.name!r}: geometry '{mesh.gid}' names material symbol(s) "
            f"{missing} that no binding or material id resolves; those "
            "elements have material -1."
        )
    if guessed:
        _warn(
            f"{st.doc.name!r}: geometry '{mesh.gid}' is bound without material "
            f"symbol(s) {guessed}; the material(s) whose id they spell are used."
        )
    return out


def _finish_mesh(st: _State, mesh: _Mesh) -> PolyData:
    if mesh.materials is None:
        return mesh.poly
    return dataclasses.replace(
        mesh.poly, element_attrs={**mesh.poly.element_attrs, "material": mesh.materials}
    )


def _bind(st: _State, inst: ET.Element) -> dict[str, int]:
    """Return the ``symbol -> material index`` map an instance's bind_material gives."""
    out: dict[str, int] = {}
    tc = _child(_child(inst, "bind_material"), "technique_common")
    for im in _children(tc, "instance_material"):
        symbol = im.get("symbol")
        target = im.get("target", "")
        if symbol is None:
            continue
        if symbol in out:
            _warn(
                f"{st.doc.name!r}: instance_material '{symbol}' is bound twice; "
                "the first binding is kept."
            )
            continue
        if not target.startswith("#") or target[1:] not in st.material_index:
            _warn(
                f"{st.doc.name!r}: instance_material '{symbol}' targets "
                f"{target!r}, which is not a material of the document."
            )
            continue
        out[symbol] = st.material_index[target[1:]]
    return out


# -----------------------------------------------------------------------------
# nodes
# -----------------------------------------------------------------------------


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return _IDENTITY.copy()
    x, y, z = axis / norm
    c = np.cos(np.radians(degrees))
    s = np.sin(np.radians(degrees))
    t = 1.0 - c
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    return out


def _lookat(values: np.ndarray) -> np.ndarray:
    """Return the camera-to-parent matrix of a ``<lookat>``.

    An ``up`` parallel to the view direction leaves the roll undefined; the
    world axis least aligned with the view direction stands in for it, so
    the matrix stays a rotation whatever the direction.
    """
    eye, target, up = values[:3], values[3:6], values[6:]
    z = eye - target
    zn = np.linalg.norm(z)
    z = z / zn if zn else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    xn = np.linalg.norm(x)
    if not xn:
        x = np.cross(np.eye(3)[np.argmin(np.abs(z))], z)
        xn = np.linalg.norm(x)
    x = x / xn
    y = np.cross(z, x)
    out = np.eye(4, dtype=np.float64)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, y, z, eye
    return out


def _matrix_of(kind: str, values: np.ndarray) -> np.ndarray:
    if kind == "matrix":
        return values.reshape(4, 4)
    if kind == "translate":
        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = values
        return out
    if kind == "rotate":
        return _rotation(values[:3], float(values[3]))
    if kind == "scale":
        return np.diag([values[0], values[1], values[2], 1.0])
    return _lookat(values)


def _node_transforms(
    doc: _Doc, node: ET.Element, what: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.eye(4, dtype=np.float64)
    spelled: list[dict[str, Any]] = []
    for child in node:
        kind = _local(child.tag)
        if kind == "skew":
            _warn(f"{what} has a <skew>, which this reader ignores.")
            continue
        size = _KIND_SIZE.get(kind)
        if size is None:
            continue
        values = _floats(child.text, f"{what} <{kind}>", n=size)
        matrix = matrix @ _matrix_of(kind, values)
        spelled.append(
            {"kind": kind, "sid": child.get("sid"), "values": values.tolist()}
        )
    return matrix, spelled


def _controller_geometry(
    st: _State, ctrl: ET.Element, stack: tuple[str, ...]
) -> tuple[int, ET.Element | None]:
    """Return (geometry slot, skin element or None) a controller ends at."""
    cid = ctrl.get("id", "?")
    if cid in stack:
        raise CodecError(f"{st.doc.name!r}: controller '{cid}' contains itself.")
    skin = _child(ctrl, "skin")
    morph = _child(ctrl, "morph")
    body = skin if skin is not None else morph
    if body is None:
        raise CodecError(
            f"{st.doc.name!r}: controller '{cid}' has neither <skin> nor <morph>."
        )
    if morph is not None and skin is None:
        _warn(
            f"{st.doc.name!r}: controller '{cid}' is a morph; only its base geometry is read."
        )
    target = _ref(st.doc, body.get("source"), f"controller '{cid}'")
    if _local(target.tag) == "controller":
        slot, inner = _controller_geometry(st, target, (*stack, cid))
        return slot, skin if skin is not None else inner
    if id(target) not in st.slot_of_geometry:
        raise CodecError(
            f"{st.doc.name!r}: controller '{cid}' names {body.get('source')!r}, "
            "which is not a geometry."
        )
    return st.slot_of_geometry[id(target)], skin


def _walk(
    st: _State,
    node: ET.Element,
    stack: tuple[tuple[int, str], ...],
) -> int:
    """Walk one ``<node>`` and its subtree into ``st.nodes``.

    Parameters
    ----------
    st
        The walk's state.
    node
        The element to walk.
    stack
        The elements above it, as ``(id(element), label)`` pairs; an element
        met again below itself is a cycle. Compared by identity, since a
        document may reuse an id for another node.

    Returns
    -------
    int
        The node's index.
    """
    doc = st.doc
    nid = node.get("id")
    label = nid or node.get("name") or "?"
    where = f"node '{label}'"
    what = f"{doc.name!r}: {where}"
    if any(key == id(node) for key, _ in stack):
        chain = " -> ".join((*(name for _, name in stack), label))
        raise CodecError(f"{doc.name!r}: node {chain} contains itself.")
    idx = _new_node(st, None)
    inner = (*stack, (id(node), label))
    if nid is not None:
        st.nodes_by_id.setdefault(nid, []).append(idx)
    sid = node.get("sid")

    matrix, spelled = _node_transforms(doc, node, what)
    extras: dict[str, Any] = {}
    if nid is not None:
        extras["id"] = nid
    if sid is not None:
        extras["sid"] = sid
    if node.get("type") == "JOINT":
        extras["type"] = "JOINT"
    if spelled:
        extras["transforms"] = spelled

    mesh_idx: int | None = None
    children: list[int] = []
    for child in node:
        tag = _local(child.tag)
        if tag == "node":
            children.append(_walk(st, child, inner))
        elif tag == "instance_node":
            target = _instanced(doc, child, f"{where} instance_node")
            if target is None:
                continue
            if _local(target.tag) != "node":
                raise CodecError(
                    f"{what} instances {child.get('url')!r}, which is not a node."
                )
            children.append(_walk(st, target, inner))
        elif tag == "instance_geometry":
            geo = _instanced(doc, child, f"{where} instance_geometry")
            if geo is None:
                continue
            slot = st.slot_of_geometry.get(id(geo))
            if slot is None:
                raise CodecError(
                    f"{what} instances {child.get('url')!r}, which is not a geometry."
                )
            mesh = _mesh_at(st, slot, _bind(st, child))
            if mesh_idx is None:
                mesh_idx = mesh
            else:
                children.append(
                    _synth_node(st, mesh, st.geometries[slot].get("name", ""))
                )
        elif tag == "instance_controller":
            ctrl = _instanced(doc, child, f"{where} instance_controller")
            if ctrl is None:
                continue
            if _local(ctrl.tag) != "controller":
                raise CodecError(
                    f"{what} instances {child.get('url')!r}, which is not a controller."
                )
            slot, skin = _controller_geometry(st, ctrl, ())
            mesh = _mesh_at(st, slot, _bind(st, child))
            if skin is not None:
                st.skinned += 1
            if mesh_idx is None:
                mesh_idx = mesh
            else:
                children.append(_synth_node(st, mesh, ctrl.get("name", "")))
        elif tag in ("instance_camera", "instance_light"):
            kind = tag[len("instance_") :]
            url = child.get("url", "")
            if kind in extras:
                _warn(
                    f"{what} instances a second {kind}, {url!r}; only the first, "
                    f"{extras[kind]!r}, is kept."
                )
                continue
            extras[kind] = url[1:] if url.startswith("#") else url

    st.nodes[idx] = SceneNode(
        name=node.get("name", ""),
        mesh=mesh_idx,
        children=tuple(children),
        matrix=matrix,
        extras=extras,
    )
    return idx


def _synth_node(st: _State, mesh: int, name: str) -> int:
    return _new_node(st, SceneNode(name=name, mesh=mesh))


def _new_node(st: _State, node: SceneNode | None) -> int:
    """Append ``node`` to the walk's node list and return its index."""
    if len(st.nodes) >= st.max_nodes:
        raise CodecError(
            f"{st.doc.name!r}: instancing expands the node tree past "
            f"{st.max_nodes} nodes, more than a document of {st.doc.size} bytes "
            "is read into."
        )
    st.nodes.append(node)
    return len(st.nodes) - 1


def _read_scenes(st: _State) -> tuple[tuple[tuple[int, ...], ...], int, str]:
    doc = st.doc
    visual = _library(doc.root, "library_visual_scenes", "visual_scene")
    scenes: list[tuple[int, ...]] = []
    for vs in visual:
        roots: list[int] = []
        for child in vs:
            tag = _local(child.tag)
            if tag == "node":
                roots.append(_walk(st, child, ()))
            elif tag == "instance_node":
                where = "visual_scene instance_node"
                what = f"{doc.name!r}: {where}"
                target = _instanced(doc, child, where)
                if target is None:
                    continue
                if _local(target.tag) != "node":
                    raise CodecError(
                        f"{what} instances {child.get('url')!r}, which is not a node."
                    )
                roots.append(_walk(st, target, ()))
        scenes.append(tuple(roots))
    active = 0
    inst = _child(_child(doc.root, "scene"), "instance_visual_scene")
    target = _instanced(doc, inst, "<scene>") if inst is not None and visual else None
    if target is not None:
        for i, vs in enumerate(visual):
            if vs is target:
                active = i
                break
        else:
            raise CodecError(
                f"{doc.name!r}: <scene> instances {inst.get('url')!r}, which is not "
                "a visual_scene."
            )
    name = visual[active].get("name", "") if visual else ""
    return tuple(scenes), active, name


# -----------------------------------------------------------------------------
# animations
# -----------------------------------------------------------------------------


def _sampler(
    st: _State, elem: ET.Element, cache: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    doc = st.doc
    known = cache.get(id(elem))
    if known is not None:
        return known
    what = f"sampler '{elem.get('id', '?')}'"
    out: dict[str, Any] = {"interpolation": "LINEAR"}
    times = values = None
    for inp in _children(elem, "input"):
        semantic = inp.get("semantic")
        src = _ref(doc, inp.get("source"), f"{what} {semantic} input")
        if semantic == "INPUT":
            times = _own(doc, src, _float_source(doc, src, f"{what} INPUT")[:, 0])
        elif semantic == "OUTPUT":
            values = _own(doc, src, _float_source(doc, src, f"{what} OUTPUT"))
        elif semantic == "INTERPOLATION":
            names = [n.upper() for n in _name_source(doc, src, f"{what} INTERPOLATION")]
            if names:
                if len(set(names)) > 1:
                    _warn(
                        f"{doc.name!r}: {what} mixes interpolations; {names[0]} is taken for all keys."
                    )
                if names[0] in _INTERPOLATIONS:
                    out["interpolation"] = names[0]
                else:
                    _warn(
                        f"{doc.name!r}: {what} has interpolation {names[0]!r}, which the "
                        "specification does not name; it is read as LINEAR."
                    )
        elif semantic in ("IN_TANGENT", "OUT_TANGENT"):
            out[semantic.lower() + "s"] = _own(
                doc, src, _float_source(doc, src, f"{what} {semantic}")
            )
    if times is None or values is None:
        raise CodecError(f"{doc.name!r}: {what} needs INPUT and OUTPUT.")
    if len(times) != len(values):
        raise CodecError(
            f"{doc.name!r}: {what} has {len(times)} keys but {len(values)} output values."
        )
    for key in ("in_tangents", "out_tangents"):
        if key in out and len(out[key]) != len(times):
            raise CodecError(
                f"{doc.name!r}: {what} has {len(times)} keys but {len(out[key])} {key[:-1]} values."
            )
    out["times"] = times
    out["values"] = values[:, 0] if values.shape[1] == 1 else values
    cache[id(elem)] = out
    return out


def _descendant_by_sid(st: _State, node_idx: int, sid: str) -> int | None:
    """Return the first node below ``node_idx``, breadth-first, whose sid is ``sid``."""
    queue = list(st.nodes[node_idx].children)
    k = 0
    while k < len(queue):
        idx = queue[k]
        k += 1
        node = st.nodes[idx]
        if node.extras.get("sid") == sid:
            return idx
        queue.extend(node.children)
    return None


def _transform_by_sid(st: _State, node_idx: int, sid: str) -> tuple[int, str] | None:
    """Return the node holding the transform ``sid`` and the transform's kind.

    The node itself is looked at first, then the nodes below it,
    breadth-first, so an address that skips the nodes between its id and
    the transform still reaches it.
    """
    queue = [node_idx]
    k = 0
    while k < len(queue):
        idx = queue[k]
        k += 1
        node = st.nodes[idx]
        for t in node.extras.get("transforms", ()):
            if t["sid"] == sid:
                return idx, t["kind"]
        queue.extend(node.children)
    return None


def _targets(st: _State, target: str) -> list[dict[str, Any]]:
    """Return one target per node the address reaches.

    A ``<library_nodes>`` node instanced twice is walked into two nodes that
    share its id, and a channel addressing that id moves both, so each copy
    gets the channel. Each sid of the address is looked up breadth-first
    below the node before it, as the specification's address syntax says:
    a node sid among its descendants, the transform sid on the node the
    last of them reaches and then on its descendants.
    """
    parts = target.split("/")
    if len(parts) < 2:
        return []
    last = parts[-1]
    member: str | None = None
    for i, ch in enumerate(last):
        if ch in ".(":
            member = last[i + 1 :] if ch == "." else last[i:]
            last = last[:i]
            break
    out: list[dict[str, Any]] = []
    for node_idx in st.nodes_by_id.get(parts[0], []):
        for part in parts[1:-1]:
            node_idx = _descendant_by_sid(st, node_idx, part)
            if node_idx is None:
                break
        if node_idx is None:
            continue
        hit = _transform_by_sid(st, node_idx, last) if last else None
        if hit is not None:
            out.append(
                {
                    "node": hit[0],
                    "path": _PATH_OF_KIND[hit[1]],
                    "sid": last,
                    "member": member,
                }
            )
    return out


def _read_animations(st: _State) -> list[dict[str, Any]]:
    doc = st.doc
    out: list[dict[str, Any]] = []
    n_targets = 0
    for anim in _library(doc.root, "library_animations", "animation"):
        # Per animation, so one reached from two owns its arrays in each.
        cache: dict[int, dict[str, Any]] = {}
        samplers: list[dict[str, Any]] = []
        slot_of: dict[int, int] = {}
        channels: list[dict[str, Any]] = []
        for channel in (e for e in anim.iter() if _local(e.tag) == "channel"):
            target_text = channel.get("target", "")
            targets = _targets(st, target_text)
            if not targets:
                _warn(
                    f"{doc.name!r}: animation '{anim.get('id', '?')}' targets "
                    f"{target_text!r}, which reaches no node transform; the channel is dropped."
                )
                continue
            elem = _ref(doc, channel.get("source"), f"channel {target_text!r}")
            if _local(elem.tag) != "sampler":
                raise CodecError(
                    f"{doc.name!r}: channel {target_text!r} names {channel.get('source')!r}, "
                    "which is not a sampler."
                )
            n_targets += len(targets)
            if n_targets > st.max_nodes:
                raise CodecError(
                    f"{doc.name!r}: channels addressing instanced nodes expand "
                    f"past {st.max_nodes} targets, more than a document of "
                    f"{doc.size} bytes is read into."
                )
            sampler = _sampler(st, elem, cache)
            slot = slot_of.get(id(elem))
            if slot is None:
                slot = slot_of[id(elem)] = len(samplers)
                samplers.append(sampler)
            channels.extend({"sampler": slot, "target": t} for t in targets)
        if not channels:
            continue
        out.append(
            {
                "name": anim.get("name") or anim.get("id", ""),
                "channels": channels,
                "samplers": samplers,
            }
        )
    return out


# =============================================================================
# write
# =============================================================================


def write_scene(
    scene: SceneData,
    path: Source,
    *,
    up_axis: str | None = None,
    unit: dict[str, Any] | None = None,
    **opts: Any,
) -> None:
    """Write a SceneData as a COLLADA 1.4.1 document.

    Parameters
    ----------
    scene
        Scene to serialize. Every mesh becomes a ``<geometry>`` named by its
        ``global_attrs["mesh_name"]``, every node a ``<node>`` spelled with
        the transform elements it was read with when ``extras["transforms"]``
        still composes to its matrix, else one ``<matrix>``; materials
        become phong effects and animations channels targeting those
        transform elements. The ``translation``, ``rotation`` and ``scale``
        channels without a ``sid`` that animate a node written as one
        ``<matrix>`` - glTF's, whose rotation is a quaternion no
        ``<rotate>`` spells - are baked, per animation and node, into one
        channel of matrix keys: the keys are every such channel's times, a
        path none animates holds the node's rest value, and since COLLADA
        blends a matrix element by element, extra keys split a span turning
        more than 15 degrees (unless the rotation steps), each
        ``CUBICSPLINE`` span in four, and hold a
        ``STEP`` channel's value up to its next key when others blend. A
        node reached twice (a second parent, or the
        root of a second scene) is written once in ``<library_nodes>`` and
        instanced from there by every parent; the schema lists those
        instances before a parent's child ``<node>`` elements, and a scene
        root reached so gets a wrapper node to hold its instance. A node
        the reader copied out of an ``<instance_node>`` - one holding the
        ``id`` of a node written before it, whose subtree it matches node
        for node - is instanced from that node again, its channels written
        once for both, rather than written in full under a generated id.
        COLLADA has no alpha mask: a material's
        alpha goes out as an ``A_ONE`` transparency when it is below one or
        ``alpha_mode`` is not ``OPAQUE``, and ``alpha_mode`` and
        ``alpha_cutoff`` themselves are not written; nor are ``metallic``
        and ``roughness``, which phong has no term for. Integer ``colors``
        are scaled to COLLADA's 0..1 floats by their dtype's maximum (255
        for a signed dtype, with a warning when one holds values outside
        0..255) and read back as float64. A mesh is written as one primitive
        block (``<triangles>``, ``<polylist>``, ``<lines>``,
        ``<linestrips>``, ``<tristrips>``) per material, so elements whose
        kinds or materials interleave read back grouped by block and
        material, each group in its first element's place: lines and
        triangles given as ``[line, triangle, line]`` come back as
        ``[line, line, triangle]``. Triangles share the ``<polylist>`` of a
        mesh that has quads or polygons, and keep their order among them. A
        ``<polylist>`` records only each element's vertex count, so a
        polygon of three or four vertices reads back as a triangle or a
        quad. A material's ``base_color`` of three values is opaque.
    path
        Output file path, open binary file object, or text handle encoding
        UTF-8.
    up_axis
        ``X_UP``, ``Y_UP`` or ``Z_UP``; overrides ``global_attrs["asset"]``.
    unit
        ``{"name": ..., "meter": ...}``; overrides ``global_attrs["asset"]``.
    **opts
        Accepted for interface symmetry and ignored.

    Warns
    -----
    UserWarning
        For the scene's ``skins``, which are not written, an animation
        channel naming a node or sampler the written scene lacks, whose
        target the written node lacks or whose ``member`` is neither a name
        nor ``(i)(j)`` indices, a channel whose sampler has no keys, keys
        or tangents that are not finite or key times that do not increase,
        a second channel on a target the animation already animates, a
        channel only a copy of an instanced node holds (it is written for
        the node and reaches every instance), a channel without a ``sid``
        on a node
        spelled with its own transform elements whose path is ``rotation``
        (a quaternion, not a ``<rotate>``'s axis and angle), whose node
        spells no single element of its kind, or whose keys are not that
        element's width or interpolation one COLLADA names, a channel to
        bake whose keys are not its path's width, not finite or not
        increasing, whose interpolation is not ``LINEAR``, ``STEP`` or
        ``CUBICSPLINE``, or that animates a path of its node a second time
        in one animation, a node to bake whose matrix holds a shear or
        projection (its keys drop it),
        for an asset key COLLADA has no place for (it keeps ``up_axis``,
        ``unit``, ``created``, ``modified``, ``author`` and ``copyright``,
        writes itself as the authoring tool, and replaces another format's
        ``version`` and ``generator`` silently), an asset ``created`` or
        ``modified`` that is not an ``xs:dateTime`` (the epoch is written;
        a ``datetime`` is written in ISO form), a unit name that is not an
        XML name token (the unit is written without one), elements COLLADA
        cannot hold (points, volume cells; a mesh with nothing else keeps its positions as a ``<mesh>``
        without primitives), vertex attributes it has no input for
        (``joints`` and ``weights`` among them), element attributes other
        than ``material``, a ``metallic_roughness_texture`` or
        ``occlusion_texture`` (the common profile has no slot for either),
        a base colour or emissive tinting its texture, a base colour on a
        ``constant`` material (a diffuse holds a colour or a texture, and
        ``constant`` has none), a set suffix of ``normals``, ``texcoords``,
        ``colors``, ``tangent``, ``binormal``, ``textangent`` or
        ``texbinormal`` that is not a free number (it is written under the
        lowest free one and reads back under it), a node no scene reaches
        (it is not written), a node ``id`` or ``sid`` that is not an XML
        name or an ``id`` an earlier node holds (a fresh id is written), a
        transform ``sid`` that is not a name (a letter or ``_``, then
        letters, digits, ``_`` or ``-``), an emissive texture under a black
        ``emissive`` (the texture is dropped), a vertex attribute of a set
        other than 0 on a mesh with no primitive, a node whose extras name a
        ``camera`` or ``light`` (this writer carries neither, so the
        instance is dropped), or node or material ``extras`` keys COLLADA
        has no place for.

    Raises
    ------
    CodecError
        When the node graph has a cycle, names a node the scene lacks or
        nests deeper than the interpreter's recursion limit, a name holds a
        character XML cannot carry, a mesh's ``material`` column does not
        number its elements or holds values other than whole numbers, a
        material's ``ambient``, ``specular`` or ``reflective`` extra is not
        3 or 4 finite numbers or its ``shininess``, ``reflectivity`` or
        ``index_of_refraction`` not a finite number, ``up_axis`` is not one
        of the three, ``unit`` or the asset is not a dict or its meter not a
        finite positive number, a scene names a root node the scene lacks,
        or ``path`` is a text handle encoding something other than UTF-8,
        unless the encoding spells ASCII as ASCII and the document is
        ASCII, ``global_attrs["animations"]`` is not a list, an animation
        entry, channel or sampler is not a dict, a sampler lacks ``times``
        or ``values``, its values or tangents do not number its ``times``,
        or its ``interpolation`` is not one of the six the specification
        names. Also when a mesh's vertices are not (n, 3) real numbers, its element types not
        integer codes in 0..255, its offsets not integers rising from 0 to
        the connectivity's length, its connectivity names a vertex it lacks
        or an element has a vertex count its type does not allow, a node's
        matrix is not 4x4 real numbers, or a material's ``base_color`` is
        not 3 or 4 numbers.
    """
    name = source_name(path)
    asset = scene.global_attrs.get("asset", {})
    if not isinstance(asset, dict):
        raise CodecError(
            f"{name!r}: global_attrs['asset'] must be a dict, not {asset!r}."
        )
    axis = up_axis or asset.get("up_axis", "Y_UP")
    if axis not in _UP_AXES:
        raise CodecError(
            f"{name!r}: up_axis must be one of {', '.join(_UP_AXES)}, not {axis!r}."
        )
    if unit is None:
        unit = asset.get("unit")
    if unit is None:
        unit = {"name": "meter", "meter": 1.0}
    if not isinstance(unit, dict):
        raise CodecError(
            f"{name!r}: unit must be a dict with 'name' and 'meter', not {unit!r}."
        )
    try:
        meter = float(unit.get("meter", 1.0))
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{name!r}: the unit's meter must be a number, not {unit.get('meter')!r}."
        ) from exc
    if not (np.isfinite(meter) and meter > 0.0):
        raise CodecError(
            f"{name!r}: the unit's meter must be a finite positive number, not {meter!r}."
        )

    for n, node in enumerate(scene.nodes):
        m = np.asarray(node.matrix)
        if m.shape != (4, 4) or m.dtype.kind not in "iuf":
            raise CodecError(
                f"{name!r}: node {n} matrix must be 4x4 real numbers, not "
                f"{m.dtype} of shape {m.shape}."
            )
    for kind, entries, known in (
        ("node", scene.nodes, _NODE_EXTRAS),
        ("material", scene.materials, _MATERIAL_EXTRAS),
    ):
        unknown = sorted(
            {str(k) for entry in entries for k in entry.extras if k not in known}
        )
        if unknown:
            _warn(
                f"{name!r}: {kind} extras {unknown} have no COLLADA counterpart; "
                "they are dropped."
            )
    roots_per_scene = list(scene.scenes) if scene.scenes else [_roots(scene)]
    order, alias = _emission_order(scene, roots_per_scene)
    ids = _Ids(scene, alias)
    for key, fate in (
        ("id", "written with a generated id"),
        ("sid", "written without a sid"),
    ):
        unnamed = [
            n
            for n, node in enumerate(scene.nodes)
            if node.extras.get(key) is not None
            and _xml_name(node.extras.get(key)) is None
        ]
        if unnamed:
            _warn(
                f"{name!r}: the {key} of node(s) {unnamed} is not an XML name; "
                f"they are {fate}."
            )
    repeated = [
        n
        for n, node in enumerate(scene.nodes)
        if _xml_name(node.extras.get("id")) is not None
        and ids.node(n) != node.extras["id"]
    ]
    if repeated:
        _warn(
            f"{name!r}: node(s) {repeated} repeat an id an earlier node holds; "
            "they are written with a generated one."
        )
    bad_sids = [
        (n, t["sid"])
        for n, node in enumerate(scene.nodes)
        if _spelled(node)[1]
        for t in node.extras["transforms"]
        if t.get("sid") is not None and _transform_sid(t["sid"]) is None
    ]
    if bad_sids:
        _warn(
            f"{name!r}: transform sid(s) {bad_sids} (node, sid) are not names a "
            "channel target can end in (a letter or underscore, then letters, "
            "digits, '_' or '-'); the transforms are written without them."
        )
    unkept = sorted(str(k) for k in asset if k not in _ASSET_KEYS)
    if unkept:
        _warn(
            f"{name!r}: asset key(s) {unkept} have no COLLADA counterpart; "
            "they are dropped."
        )
    dates = {
        key: _date_time(asset.get(key, _EPOCH), key, name)
        for key in ("created", "modified")
    }
    unit_name = str(unit.get("name", "meter"))
    if not _NMTOKEN.match(unit_name):
        _warn(
            f"{name!r}: unit name {unit_name!r} is not an XML name token; the "
            "unit is written without a name."
        )
        unit_attr = ""
    else:
        unit_attr = f' name="{unit_name}"'
    textured: set[int] = set()
    parts: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>\n',
        f'<COLLADA xmlns="{_NS}" version="1.4.1">\n',
        "  <asset>\n",
        f"    <contributor>{_contributor_xml(asset)}</contributor>\n",
        f"    <created>{dates['created']}</created>\n",
        f"    <modified>{dates['modified']}</modified>\n",
        f'    <unit{unit_attr} meter="{meter!r}"/>\n',
        f"    <up_axis>{axis}</up_axis>\n",
        "  </asset>\n",
    ]
    if scene.images:
        parts.append("  <library_images>\n")
        for i, img in enumerate(scene.images):
            parts.append(_image_xml(ids.image(i), img))
        parts.append("  </library_images>\n")
    if scene.materials:
        parts.append("  <library_effects>\n")
        for i, mat in enumerate(scene.materials):
            xml, has_texture = _effect_xml(scene, ids, i, mat, name)
            parts.append(xml)
            if has_texture:
                textured.add(i)
        parts.append("  </library_effects>\n  <library_materials>\n")
        for i, mat in enumerate(scene.materials):
            parts.append(
                f'    <material id="{ids.material(i)}"{_name_attr(mat.name)}>'
                f'<instance_effect url="#{ids.effect(i)}"/></material>\n'
            )
        parts.append("  </library_materials>\n")

    written = set(order)
    if len(written) < len(scene.nodes):
        _warn(
            f"{name!r}: {len(scene.nodes) - len(written)} node(s) are in no scene "
            "and are not written."
        )
    lost = sorted(
        k for k in scene.global_attrs if k not in ("asset", "skins", "animations")
    )
    if lost:
        _warn(
            f"{name!r}: global_attrs {lost} of the scene have no COLLADA "
            "counterpart; they are dropped."
        )
    if scene.global_attrs.get("skins"):
        _warn(
            f"{name!r}: global_attrs ['skins'] are not written; COLLADA skins "
            "are not supported yet."
        )

    binds: list[str] = []
    parts.append("  <library_geometries>\n")
    for i, mesh in enumerate(scene.meshes):
        xml, used, uv_set = _geometry_xml(mesh, ids, i, len(scene.materials), name)
        parts.append(xml)
        binds.append(_bind_xml(ids, used, textured, uv_set))
    parts.append("  </library_geometries>\n")

    animations = _animations_xml(scene, ids, written, name)
    if animations:
        parts.append(
            "  <library_animations>\n" + animations + "  </library_animations>\n"
        )

    try:
        shared_order = _shared_nodes(scene, roots_per_scene, alias, name)
        shared = set(shared_order)
        if shared_order:
            parts.append("  <library_nodes>\n")
            parts.extend(
                _node_xml(scene, ids, n, binds, shared, 2, name) for n in shared_order
            )
            parts.append("  </library_nodes>\n")
        active = (
            scene.active_scene if 0 <= scene.active_scene < len(roots_per_scene) else 0
        )
        parts.append("  <library_visual_scenes>\n")
        for s, roots in enumerate(roots_per_scene):
            label = _name_attr(scene.name) if s == active else ""
            parts.append(f'    <visual_scene id="{ids.scene(s)}"{label}>\n')
            for r in roots:
                r = alias.get(r, r)
                if r in shared:
                    parts.append(
                        f'      <node id="{ids.fresh("instance")}">'
                        f'<instance_node url="#{ids.node(r)}"/></node>\n'
                    )
                else:
                    parts.append(_node_xml(scene, ids, r, binds, shared, 3, name))
            parts.append("    </visual_scene>\n")
    except RecursionError as exc:
        raise CodecError(
            f"{name!r}: the node tree nests deeper than this writer follows."
        ) from exc
    parts.append("  </library_visual_scenes>\n")
    parts.append(
        f'  <scene><instance_visual_scene url="#{ids.scene(active)}"/></scene>\n</COLLADA>\n'
    )
    text = "".join(parts)
    _check_handle_encoding(path, text, name)
    write_text(path, text)


def _check_handle_encoding(path: Source, text: str, name: str) -> None:
    """Refuse a text handle that would encode a UTF-8 document otherwise.

    The declaration names UTF-8; a handle opened with another encoding
    writes its own bytes under it, which no reader decodes unless that
    encoding spells ASCII as ASCII and the text holds nothing else.
    """
    encoding = getattr(path, "encoding", None) if is_buffer(path) else None
    if not isinstance(encoding, str):
        return
    try:
        canonical = codecs.lookup(encoding).name
        ascii_safe = _ASCII_PROBE.encode(encoding) == _ASCII_PROBE.encode("ascii")
    except LookupError:
        canonical, ascii_safe = encoding, False
    if canonical not in ("utf-8", "utf-8-sig") and not (ascii_safe and text.isascii()):
        raise CodecError(
            f"{name!r} is a text handle encoding {encoding!r}; COLLADA is "
            "written as UTF-8, so open it with encoding='utf-8' or in "
            "binary mode."
        )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as a one-node COLLADA scene.

    Parameters
    ----------
    poly
        Mesh to serialize. ``element_attrs["material"]`` values become
        materials named ``material_<value>``; points and volume cells are
        skipped with a warning.
    path
        Output file path or open binary file object.
    **opts
        Passed to :func:`write_scene` (``up_axis``, ``unit``).

    Raises
    ------
    CodecError
        Whatever :func:`write_scene` refuses.
    """
    materials: tuple[SceneMaterial, ...] = ()
    column = poly.element_attrs.get("material")
    if column is not None:
        column = _material_column(column, f"{source_name(path)!r}: the mesh")
        used = np.unique(column[column >= 0])
        materials = tuple(
            SceneMaterial(
                name=f"material_{int(m)}", metallic=0.0, extras={"shading": "phong"}
            )
            for m in used
        )
        new_column = np.where(column >= 0, np.searchsorted(used, column), -1).astype(
            np.int32
        )
        poly = dataclasses.replace(
            poly, element_attrs={**poly.element_attrs, "material": new_column}
        )
    global_attrs = {}
    if "asset" in poly.global_attrs:
        global_attrs["asset"] = poly.global_attrs["asset"]
        poly = dataclasses.replace(
            poly,
            global_attrs={k: v for k, v in poly.global_attrs.items() if k != "asset"},
        )
    scene = SceneData(
        meshes=(poly,),
        nodes=(SceneNode(mesh=0),),
        materials=materials,
        scenes=((0,),),
        global_attrs=global_attrs,
    )
    write_scene(scene, path, **opts)


# -----------------------------------------------------------------------------
# write helpers
# -----------------------------------------------------------------------------


class _Ids:
    """The XML ids a write uses, given ones kept where they are valid and unique.

    A generated id also serves as the stem of ``<stem>-positions``,
    ``<stem>-array`` and the like, so a stem is not handed out while a
    given node id begins with it and a dash. Nor is a given node sid, so
    no generated id spells another node's sid.

    A node of a copied subtree (``alias``, see :func:`_emission_order`)
    shares the id of the node it copies, as the copies a reader makes of
    an ``<instance_node>`` do.
    """

    def __init__(self, scene: SceneData, alias: dict[int, int]) -> None:
        self.alias = alias
        self.given_sids: list[str | None] = [
            n.extras["sid"]
            if isinstance(n.extras.get("sid"), str) and _NCNAME.match(n.extras["sid"])
            else None
            for n in scene.nodes
        ]
        self.sids = {sid for sid in self.given_sids if sid is not None}
        self.taken: set[str] = set()
        self.stems: set[str] = set()
        self.node_ids: list[str] = []
        given = [
            n.extras.get("id")
            if isinstance(n.extras.get("id"), str) and _NCNAME.match(n.extras["id"])
            else None
            for n in scene.nodes
        ]
        for i in alias:
            given[i] = None
        for i, g in enumerate(given):
            if g is not None and g not in self.taken:
                self.taken.add(g)
                head = g
                while "-" in head:
                    head = head.rsplit("-", 1)[0]
                    self.stems.add(head)
            elif g is not None:
                given[i] = None
        for i, g in enumerate(given):
            if i in alias:
                self.node_ids.append("")
            else:
                self.node_ids.append(g if g is not None else self.fresh(f"node{i}"))
        for i, node in alias.items():
            self.node_ids[i] = self.node_ids[node]
        self._cache: dict[tuple[str, int], str] = {}

    def fresh(self, base: str) -> str:
        candidate = base
        k = 0
        while (
            candidate in self.taken or candidate in self.stems or candidate in self.sids
        ):
            k += 1
            candidate = f"{base}_{k}"
        self.taken.add(candidate)
        return candidate

    def _of(self, prefix: str, i: int) -> str:
        key = (prefix, i)
        if key not in self._cache:
            self._cache[key] = self.fresh(f"{prefix}{i}")
        return self._cache[key]

    def image(self, i: int) -> str:
        return self._of("image", i)

    def effect(self, i: int) -> str:
        return self._of("effect", i)

    def material(self, i: int) -> str:
        return self._of("material", i)

    def geometry(self, i: int) -> str:
        return self._of("geometry", i)

    def scene(self, i: int) -> str:
        return self._of("scene", i)

    def animation(self, i: int) -> str:
        return self._of("animation", i)

    def node(self, i: int) -> str:
        return self.node_ids[i]


def _date_time(value: Any, key: str, name: str) -> str:
    """Return an asset date as written, the epoch when it is not an ``xs:dateTime``."""
    text = value.isoformat() if isinstance(value, datetime) else str(value)
    if _DATE_TIME.match(text):
        try:
            datetime.fromisoformat(text)
        except ValueError:
            pass
        else:
            return text
    _warn(
        f"{name!r}: asset {key} {text!r} is not an ISO 8601 date and time; "
        f"{_EPOCH} is written."
    )
    return _EPOCH


def _contributor_xml(asset: dict[str, Any]) -> str:
    """Return the children of the ``<contributor>`` a write spells.

    Parameters
    ----------
    asset
        ``global_attrs["asset"]``; its ``author`` and ``copyright`` are
        kept, and the authoring tool is always this writer.

    Returns
    -------
    str
        The ``<author>``, ``<authoring_tool>`` and ``<copyright>`` elements.
    """
    given = {"authoring_tool": f"polyxios {__version__}"}
    for key in ("author", "copyright"):
        if asset.get(key) not in (None, ""):
            given[key] = _esc(str(asset[key]), f"asset {key}")
    return "".join(
        f"<{key}>{given[key]}</{key}>" for key in _CONTRIBUTOR_KEYS if key in given
    )


def _xml_name(value: Any) -> str | None:
    return value if isinstance(value, str) and _NCNAME.match(value) else None


def _transform_sid(value: Any) -> str | None:
    return value if isinstance(value, str) and _TRANSFORM_SID.match(value) else None


def _index(value: Any) -> int | None:
    """Return ``value`` as a plain int when it is an integer, numpy's included."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _esc(text: str, what: str) -> str:
    if _XML_FORBIDDEN.search(text):
        raise CodecError(f"{what} holds a character XML cannot carry: {text!r}.")
    # A parser folds a literal return in text to a newline.
    return escape(text, {"\r": "&#13;"})


def _attr(text: str, what: str) -> str:
    if _XML_FORBIDDEN.search(text):
        raise CodecError(f"{what} holds a character XML cannot carry: {text!r}.")
    return '"' + escape(text, _ATTR_ENTITIES) + '"'


def _name_attr(name: Any) -> str:
    text = "" if name is None else str(name)
    return f" name={_attr(text, 'a name')}" if text else ""


def _nums(values: np.ndarray) -> str:
    flat = np.asarray(values).ravel()
    words = list(map(str, flat.tolist()))
    if flat.dtype.kind == "f":
        for i in np.flatnonzero(~np.isfinite(flat)).tolist():
            v = flat[i]
            words[i] = "NaN" if v != v else ("INF" if v > 0 else "-INF")
    return " ".join(words)


def _source_xml(
    sid: str,
    table: np.ndarray,
    params: tuple[str, ...],
    *,
    indent: int = 8,
    kind: str = "float",
) -> str:
    table = np.asarray(table)
    stride = table.shape[1] if table.ndim == 2 else 1
    count = table.shape[0] if table.ndim else 0
    pad = " " * indent
    plist = "".join(
        f'<param name="{p}" type="{kind if p != "TRANSFORM" else "float4x4"}"/>'
        for p in params
    )
    return (
        f'{pad}<source id="{sid}">\n'
        f'{pad}  <float_array id="{sid}-array" count="{count * stride}">{_nums(table)}</float_array>\n'
        f'{pad}  <technique_common><accessor source="#{sid}-array" count="{count}" stride="{stride}">{plist}</accessor></technique_common>\n'
        f"{pad}</source>\n"
    )


def _name_source_xml(
    sid: str, names: list[str], param: str, *, indent: int = 8, by_id: bool = False
) -> str:
    pad = " " * indent
    kind, ptype = ("IDREF_array", "IDREF") if by_id else ("Name_array", "name")
    return (
        f'{pad}<source id="{sid}">\n'
        f'{pad}  <{kind} id="{sid}-array" count="{len(names)}">{" ".join(names)}</{kind}>\n'
        f'{pad}  <technique_common><accessor source="#{sid}-array" count="{len(names)}" stride="1"><param name="{param}" type="{ptype}"/></accessor></technique_common>\n'
        f"{pad}</source>\n"
    )


def _sniff_media(data: bytes) -> str:
    """Return the media type the magic bytes of ``data`` spell, or an empty string."""
    for magic, media in _MAGIC:
        if data.startswith(magic):
            return media
    return "image/webp" if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else ""


def _image_xml(iid: str, img: SceneImage) -> str:
    """Return one ``<image>``.

    A data URI with no type reads back as ``media_type=None``. An image with
    neither bytes nor a uri gets an empty ``<init_from>``, since the schema
    requires one, and reads back as neither.
    """
    if img.data is not None:
        media = img.media_type or _sniff_media(img.data)
        uri = f"data:{media};base64,{base64.b64encode(img.data).decode('ascii')}"
    elif img.uri is not None:
        uri = img.uri
    else:
        uri = ""
    return (
        f'    <image id="{iid}"{_name_attr(img.name)}>'
        f"<init_from>{_esc(uri, 'an image uri')}</init_from></image>\n"
    )


def _extra_color(
    mat: SceneMaterial, key: str, i: int, name: str
) -> tuple[float, ...] | None:
    """Return ``mat.extras[key]`` as 3 or 4 finite floats, None when absent."""
    value = mat.extras.get(key)
    if value is None:
        return None
    try:
        rgba = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        rgba = np.zeros(0)
    if rgba.ndim != 1 or len(rgba) not in (3, 4) or not np.isfinite(rgba).all():
        raise CodecError(
            f"{name!r}: material {i} extras[{key!r}] must be 3 or 4 finite "
            f"numbers, not {value!r}."
        )
    return tuple(rgba.tolist())


def _extra_float(mat: SceneMaterial, key: str, i: int, name: str) -> float | None:
    """Return ``mat.extras[key]`` as a finite float, None when absent."""
    value = mat.extras.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not np.isfinite(number):
        raise CodecError(
            f"{name!r}: material {i} extras[{key!r}] must be a finite number, "
            f"not {value!r}."
        )
    return number


def _color_xml(tag: str, rgba: tuple[float, ...]) -> str:
    values = tuple(rgba) + (1.0,) * (4 - len(rgba))
    return f"          <{tag}><color>{_nums(np.array(values[:4]))}</color></{tag}>\n"


def _effect_xml(
    scene: SceneData, ids: _Ids, i: int, mat: SceneMaterial, name: str
) -> tuple[str, bool]:
    """Return the ``<effect>`` of material ``i`` and whether it samples a texture.

    A ``base_color`` of three values is opaque.
    """
    rgba = _float_array(mat.base_color, f"{name!r}: material {i} base_color").ravel()
    if rgba.size == 3:
        rgba = np.append(rgba, 1.0)
    if rgba.size != 4:
        raise CodecError(
            f"{name!r}: material {i} base_color holds {rgba.size} values, not 3 or 4."
        )
    mat = dataclasses.replace(mat, base_color=tuple(rgba.tolist()))
    eid = ids.effect(i)
    samplers: dict[str, str] = {}
    params: list[str] = []
    glow = tuple(float(v) for v in mat.emissive[:3])
    for key in _MATERIAL_TEXTURES:
        t = getattr(mat, key)
        if t is None:
            continue
        if not (0 <= t < len(scene.textures)) or not (
            0 <= scene.textures[t].image < len(scene.images)
        ):
            _warn(
                f"{name!r}: material {i} names texture {t}, which reaches no image; it is dropped."
            )
            continue
        if key == "emissive_texture" and glow == (0.0,) * 3:
            _warn(
                f"{name!r}: material {i} has a black emissive factor, which "
                "switches its emissive texture off; a COLLADA emission texture "
                "always glows, so the texture is dropped."
            )
            continue
        tex = scene.textures[t]
        surface = f"{eid}-surface{t}"
        sampler = f"{eid}-sampler{t}"
        settings = ""
        if tex.wrap_s != 10497:
            settings += f"<wrap_s>{_GL_TO_WRAP.get(tex.wrap_s, 'WRAP')}</wrap_s>"
        if tex.wrap_t != 10497:
            settings += f"<wrap_t>{_GL_TO_WRAP.get(tex.wrap_t, 'WRAP')}</wrap_t>"
        if tex.min_filter in _GL_TO_FILTER:
            settings += f"<minfilter>{_GL_TO_FILTER[tex.min_filter]}</minfilter>"
        if tex.mag_filter in _GL_TO_FILTER:
            settings += f"<magfilter>{_GL_TO_FILTER[tex.mag_filter]}</magfilter>"
        if sampler not in samplers.values():
            params.append(
                f'        <newparam sid="{surface}"><surface type="2D"><init_from>{ids.image(tex.image)}</init_from></surface></newparam>\n'
                f'        <newparam sid="{sampler}"><sampler2D><source>{surface}</source>{settings}</sampler2D></newparam>\n'
            )
        samplers[key] = sampler

    lost = [
        key
        for key in ("metallic_roughness_texture", "occlusion_texture")
        if getattr(mat, key) is not None
    ]
    if lost:
        _warn(
            f"{name!r}: material {i} has {lost}, which the COLLADA common "
            "profile has no slot for; they are dropped."
        )
    shading = mat.extras.get("shading", "phong")
    if shading not in _SHADINGS:
        _warn(
            f"{name!r}: material {i} has shading {shading!r}, which is not one "
            f"of {', '.join(_SHADINGS)}; it is written as phong."
        )
        shading = "phong"
    unheld = [
        key
        for key in ("ambient", "specular", "shininess")
        if mat.extras.get(key) is not None and key not in _SHADING_TERMS[shading]
    ]
    if unheld:
        _warn(
            f"{name!r}: material {i} has {unheld}, which {shading} shading has "
            "no term for; they are dropped."
        )
    body: list[str] = []

    def channel(tag: str, key: str | None, rgba: tuple[float, ...] | None) -> None:
        sampler = samplers.get(key) if key else None
        if sampler is not None:
            body.append(
                f'          <{tag}><texture texture="{sampler}" texcoord="UVMap"/></{tag}>\n'
            )
        elif rgba is not None:
            body.append(_color_xml(tag, rgba))

    tinted = tuple(float(v) for v in mat.base_color[:3]) != (1.0, 1.0, 1.0)
    if shading == "constant" and (tinted or "base_color_texture" in samplers):
        _warn(
            f"{name!r}: material {i} has constant shading, which has no diffuse; "
            "its base colour and base colour texture are dropped."
        )
    elif tinted and "base_color_texture" in samplers:
        _warn(
            f"{name!r}: material {i} tints its base colour texture by "
            f"{tuple(mat.base_color[:3])}; a COLLADA diffuse holds a texture or a "
            "colour, not both, so the tint is dropped."
        )
    if "emissive_texture" in samplers and glow != (1.0,) * 3:
        _warn(
            f"{name!r}: material {i} tints its emissive texture by {glow}; a "
            "COLLADA emission holds a texture or a colour, not both, so the tint "
            "is dropped."
        )
    channel("emission", "emissive_texture", glow)
    if shading != "constant":
        ambient = _extra_color(mat, "ambient", i, name)
        if ambient is not None:
            channel("ambient", None, ambient)
        channel(
            "diffuse",
            "base_color_texture",
            (mat.base_color[0], mat.base_color[1], mat.base_color[2], 1.0),
        )
    if shading in ("phong", "blinn"):
        specular = _extra_color(mat, "specular", i, name)
        if specular is not None:
            channel("specular", None, specular)
        shininess = _extra_float(mat, "shininess", i, name)
        if shininess is not None:
            body.append(
                f"          <shininess><float>{shininess!r}</float></shininess>\n"
            )
    reflective = _extra_color(mat, "reflective", i, name)
    if reflective is not None:
        channel("reflective", None, reflective)
    reflectivity = _extra_float(mat, "reflectivity", i, name)
    if reflectivity is not None:
        body.append(
            f"          <reflectivity><float>{reflectivity!r}</float></reflectivity>\n"
        )
    alpha = float(mat.base_color[3])
    if alpha < 1.0 or mat.alpha_mode != "OPAQUE":
        body.append(
            f'          <transparent opaque="A_ONE"><color>1 1 1 {_nums(np.array([alpha]))}</color></transparent>\n'
            "          <transparency><float>1.0</float></transparency>\n"
        )
    ior = _extra_float(mat, "index_of_refraction", i, name)
    if ior is not None:
        body.append(
            f"          <index_of_refraction><float>{ior!r}</float></index_of_refraction>\n"
        )
    bump = samplers.get("normal_texture")
    extra = ""
    if bump is not None:
        extra = (
            '        <extra><technique profile="FCOLLADA"><bump>'
            f'<texture texture="{bump}" texcoord="UVMap"/></bump></technique></extra>\n'
        )
    double = (
        '      <extra><technique profile="GOOGLEEARTH"><double_sided>1</double_sided></technique></extra>\n'
        if mat.double_sided
        else ""
    )
    xml = (
        f'    <effect id="{eid}">\n      <profile_COMMON>\n'
        + "".join(params)
        + f'        <technique sid="common">\n        <{shading}>\n'
        + "".join(body)
        + f"        </{shading}>\n"
        + extra
        + "        </technique>\n"
        + double
        + "      </profile_COMMON>\n    </effect>\n"
    )
    return xml, bool(samplers)


_ATTR_PARAMS = {
    "normals": (3, ("X", "Y", "Z"), "NORMAL"),
    "colors": (None, ("R", "G", "B", "A"), "COLOR"),
    "texcoords": (None, ("S", "T", "P"), "TEXCOORD"),
    "textangent": (3, ("X", "Y", "Z"), "TEXTANGENT"),
    "texbinormal": (3, ("X", "Y", "Z"), "TEXBINORMAL"),
    "tangent": (3, ("X", "Y", "Z"), "TANGENT"),
    "binormal": (3, ("X", "Y", "Z"), "BINORMAL"),
}


def _geometry_xml(
    mesh: PolyData, ids: _Ids, i: int, n_materials: int, name: str
) -> tuple[str, list[int], int | None]:
    """Return the ``<geometry>`` of mesh ``i``, its blocks' materials and UV set.

    The third value is the lowest ``TEXCOORD`` set written, None when the
    mesh has none; a textured material binds its sampler to that set.
    """
    gid = ids.geometry(i)
    verts, types, offs = _mesh_arrays(mesh, f"{name!r}: mesh {i}")
    n = len(verts)
    label = mesh.global_attrs.get("mesh_name")
    parts = [f'    <geometry id="{gid}"{_name_attr(label)}>\n      <mesh>\n']
    parts.append(_source_xml(f"{gid}-positions", verts, ("X", "Y", "Z")))

    inputs: list[str] = []
    local_inputs: list[tuple[str, str, int]] = []
    uv_sets: list[int] = []
    dropped: list[str] = []
    renamed: list[str] = []
    # Set numbers spelled by a key are reserved up front, so a key whose
    # suffix is not a free number takes one no key spells.
    used_sets: dict[str, set[int]] = {}
    for base in _ATTR_PARAMS:
        suffixes = [_set_of(k, base) for k in mesh.vertex_attrs]
        used_sets[base] = {int(s) for s in suffixes if s}
    given_sets = {base: set(taken) for base, taken in used_sets.items()}
    for key, table in mesh.vertex_attrs.items():
        if key in ("joints", "weights"):
            dropped.append(key)
            continue
        base = key
        set_no = 0
        for prefix in _ATTR_PARAMS:
            suffix = _set_of(key, prefix)
            if suffix is None:
                continue
            base = prefix
            if suffix and int(suffix) in given_sets[prefix]:
                set_no = int(suffix)
                given_sets[prefix].discard(set_no)
            else:
                set_no = 0
                while set_no in used_sets[prefix]:
                    set_no += 1
                used_sets[prefix].add(set_no)
                renamed.append(f"{key} as set {set_no}")
        spec = _ATTR_PARAMS.get(base)
        arr = np.asarray(table)
        if (
            spec is None
            or arr.ndim != 2
            or len(arr) != n
            or arr.shape[1] > len(spec[1])
            or arr.shape[1] < 2
            or (spec[0] is not None and arr.shape[1] != spec[0])
            or arr.dtype.kind not in "iuf"
        ):
            dropped.append(key)
            continue
        if base == "colors" and arr.shape[1] == 2:
            dropped.append(key)
            continue
        sid = f"{gid}-{base}" + (f"_{set_no}" if set_no else "")
        values = arr.astype(np.float64)
        if base == "colors" and arr.dtype.kind in "iu":
            values /= np.iinfo(arr.dtype).max if arr.dtype.kind == "u" else 255.0
            if (
                arr.dtype.kind == "i"
                and arr.size
                and (arr.min() < 0 or arr.max() > 255)
            ):
                _warn(
                    f"{name!r}: {key} of mesh {i} is {arr.dtype} holding values "
                    "outside 0..255; a signed colour is scaled by 255, so they "
                    "read back outside 0..1."
                )
        parts.append(_source_xml(sid, values, spec[1][: arr.shape[1]]))
        if base == "texcoords":
            uv_sets.append(set_no)
        inputs.append(
            f'<input semantic="{spec[2]}" source="#{sid}" offset="0" set="{set_no}"/>'
        )
        local_inputs.append(
            (key, f'<input semantic="{spec[2]}" source="#{sid}"/>', set_no)
        )
    if renamed:
        _warn(
            f"{name!r}: vertex attribute(s) {renamed} of mesh {i} have a set that "
            "is not a free number as a reader spells it back (COLLADA's set is "
            "an unsigned integer, set 0 being the bare name); each reads back "
            "under the number it is written with."
        )
    if dropped:
        _warn(
            f"{name!r}: vertex attribute(s) {dropped} of mesh {i} have no COLLADA "
            "input (only normals, tangents and binormals of 3 columns, texcoords "
            "of 2-3, colors of 3-4); they "
            "are dropped."
        )
    others = sorted(k for k in mesh.element_attrs if k != "material")
    if others:
        _warn(
            f"{name!r}: element attribute(s) {others} of mesh {i} have no COLLADA "
            "counterpart (only material does); they are dropped."
        )
    tags = sorted({*mesh.vertex_tags, *mesh.element_tags})
    if tags:
        _warn(
            f"{name!r}: tag group(s) {tags} of mesh {i} have no COLLADA "
            "counterpart; they are dropped."
        )
    lost = sorted(k for k in mesh.global_attrs if k != "mesh_name")
    if lost:
        _warn(
            f"{name!r}: global_attrs {lost} of mesh {i} have no COLLADA "
            "counterpart (only mesh_name does); they are dropped."
        )
    material = mesh.element_attrs.get("material")
    if material is not None:
        mat_of = _material_column(material, f"{name!r}: mesh {i}")
        if len(mat_of) != len(types):
            raise CodecError(
                f"{name!r}: mesh {i} has {len(mat_of)} material entries for "
                f"{len(types)} elements."
            )
        if len(mat_of) and mat_of.max() >= n_materials:
            raise CodecError(
                f"{name!r}: mesh {i} names material {int(mat_of.max())}, but the "
                f"scene holds {n_materials} material(s)."
            )
        mat_of = np.maximum(mat_of, -1)
    else:
        mat_of = np.full(len(types), -1, dtype=np.int64)
    block_of = np.array([_BLOCK_INDEX.get(code, -1) for code in range(256)])[types]
    # A mesh mixing triangles with quads or polygons writes them all in one
    # polylist, so those elements keep their order on the way back.
    if np.isin(types, [_QUAD, _POLYGON]).any():
        block_of[types == _TRI] = _BLOCKS.index("polylist")
    writable = block_of >= 0
    skipped = [
        ELEMENT_TYPES_INV.get(int(code), str(code))
        for code in np.unique(types[~writable])
    ]
    if skipped:
        _warn(
            f"{name!r}: element type(s) {skipped} of mesh {i} have no COLLADA "
            "primitive; they are skipped"
            + (", leaving the positions alone." if not writable.any() else ".")
        )
    # With no primitive to carry their inputs, the attributes move into
    # <vertices>, whose inputs take no set: only set 0 fits there.
    in_vertices = ""
    if not writable.any():
        in_vertices = "".join(xml for _, xml, set_no in local_inputs if not set_no)
        unset = [key for key, _, set_no in local_inputs if set_no]
        if unset:
            _warn(
                f"{name!r}: vertex attribute(s) {unset} of mesh {i} have a set "
                "other than 0, which only a primitive's input carries, and the "
                "mesh writes no primitive; they are dropped."
            )
    parts.append(
        f'        <vertices id="{gid}-vertices"><input semantic="POSITION" source="#{gid}-positions"/>{in_vertices}</vertices>\n'
    )

    used: list[int] = []
    conn = np.asarray(mesh.connectivity)
    input_xml = (
        f'<input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>'
        + "".join(inputs)
    )
    keys = np.where(writable, block_of * (n_materials + 1) + mat_of + 1, -1)
    by_key = np.argsort(keys, kind="stable")
    groups = np.split(by_key, np.flatnonzero(np.diff(keys[by_key])) + 1)
    # A stable sort leaves each group's first element at its head.
    groups.sort(key=lambda g: int(g[0]) if len(g) else -1)
    for elems in groups:
        if not len(elems) or keys[elems[0]] < 0:
            continue
        start = int(elems[0])
        block = _BLOCKS[int(block_of[start])]
        m = int(mat_of[start])
        mat_attr = ""
        if m >= 0:
            mat_attr = f' material="{ids.material(m)}"'
            if m not in used:
                used.append(m)
        sizes = offs[elems + 1] - offs[elems]
        parts.append(f'        <{block} count="{len(elems)}"{mat_attr}>{input_xml}')
        if block in ("linestrips", "tristrips"):
            parts.append(
                "".join(
                    f"<p>{_nums(conn[offs[e] : offs[e + 1]])}</p>"
                    for e in elems.tolist()
                )
            )
        else:
            gathered = conn[_gather(offs[elems], sizes)]
            if block == "polylist":
                parts.append(f"<vcount>{_nums(sizes)}</vcount>")
            parts.append(f"<p>{_nums(gathered)}</p>")
        parts.append(f"</{block}>\n")
    parts.append("      </mesh>\n    </geometry>\n")
    return "".join(parts), used, min(uv_sets, default=None)


def _mesh_arrays(
    mesh: PolyData, what: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mesh's vertices, element types and offsets in the dtypes written.

    Parameters
    ----------
    mesh
        The mesh to write.
    what
        Names the mesh in messages.

    Returns
    -------
    tuple
        Vertices as (n, 3) float64, element types as int64 and offsets as
        int64.

    Raises
    ------
    CodecError
        When the vertices are not (n, 3) real numbers, an element type is
        not an integer code in 0..255, the offsets are not integers rising
        from 0 to the connectivity's length, one per element plus one, the
        connectivity names a vertex the mesh lacks, or an element has a
        vertex count its type does not allow (a line two, a triangle three,
        a quad four, a polygon or strip at least three, a polyline at least
        two), any of which a reader could not read back.
    """
    verts = np.asarray(mesh.vertices)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.dtype.kind not in "iuf":
        raise CodecError(
            f"{what} vertices must be (n, 3) real numbers, not {verts.dtype} "
            f"of shape {verts.shape}."
        )
    types = np.asarray(mesh.element_types)
    if types.dtype.kind not in "iu" or (
        types.size and (int(types.min()) < 0 or int(types.max()) > 255)
    ):
        raise CodecError(
            f"{what} element_types must be integer codes in 0..255, not {types.dtype}"
            + (f" from {types.min()} to {types.max()}." if types.size else ".")
        )
    offs = np.asarray(mesh.offsets)
    if offs.dtype.kind not in "iu":
        raise CodecError(f"{what} offsets must be integers, not {offs.dtype}.")
    conn = np.asarray(mesh.connectivity)
    if conn.ndim != 1 or conn.dtype.kind not in "iu":
        raise CodecError(
            f"{what} connectivity must be 1-D integers, not {conn.dtype} of "
            f"shape {conn.shape}."
        )
    if (
        offs.ndim != 1
        or len(offs) != len(types) + 1
        or (len(offs) and (int(offs[0]) != 0 or int(offs[-1]) != len(conn)))
        or np.any(offs[1:] < offs[:-1])
    ):
        raise CodecError(
            f"{what} offsets must rise from 0 to the connectivity's length "
            f"({len(conn)}), one per element ({len(types)}) plus one."
        )
    if conn.size and (int(conn.min()) < 0 or int(conn.max()) >= len(verts)):
        raise CodecError(
            f"{what} connectivity names vertices {int(conn.min())} to "
            f"{int(conn.max())}, outside 0..{len(verts) - 1}."
        )
    sizes = np.diff(offs)
    for code, (low, high) in _CORNERS.items():
        bad = (types == code) & ((sizes < low) | (sizes > high))
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            raise CodecError(
                f"{what} element {k}, a {ELEMENT_TYPES_INV[code]}, has "
                f"{int(sizes[k])} vertices; COLLADA writes it with "
                + (f"{low}." if low == high else f"at least {low}.")
            )
    return (
        verts.astype(np.float64, copy=False),
        types.astype(np.int64, copy=False),
        offs.astype(np.int64, copy=False),
    )


def _material_column(material: Any, what: str) -> np.ndarray:
    """Return a ``material`` column as int64, refusing anything but whole numbers.

    Parameters
    ----------
    material
        The ``element_attrs["material"]`` column.
    what
        The mesh the column belongs to, for the error message.

    Returns
    -------
    np.ndarray
        The column, flattened, as int64.

    Raises
    ------
    CodecError
        When the column holds values other than whole finite numbers.
    """
    column = np.asarray(material).ravel()
    whole = column.dtype.kind in "iub" or (
        column.dtype.kind == "f"
        and bool(np.isfinite(column).all())
        and bool((column == np.round(column)).all())
    )
    if not whole:
        raise CodecError(
            f"{what} has a material column of {column.dtype} that is not whole "
            "numbers; it must number the scene's materials."
        )
    return column.astype(np.int64)


def _set_of(key: str, base: str) -> str | None:
    """Return the set number ``key`` spells under ``base``.

    ``"0"`` for ``base`` itself and the digits of ``base_<n>`` when they are
    how the reader spells set ``n`` (no leading zero, no ``_0``); an empty
    string for any other suffix, None when ``key`` is not under ``base``.
    """
    if key == base:
        return "0"
    if not key.startswith(base + "_"):
        return None
    suffix = key[len(base) + 1 :]
    canonical = suffix.isascii() and suffix.isdigit() and str(int(suffix)) == suffix
    return suffix if canonical and suffix != "0" else ""


def _gather(starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Return the indices of every ``conn[start:start + size]`` run, concatenated."""
    total = int(sizes.sum())
    run = np.repeat(np.arange(len(sizes)), sizes)
    heads = np.cumsum(sizes) - sizes
    return starts[run] + np.arange(total) - heads[run]


def _emission_order(
    scene: SceneData, roots_per_scene: list[tuple[int, ...]]
) -> tuple[dict[int, int], dict[int, int]]:
    """Return the rank of every node a scene reaches, and the copies among them.

    The rank is the order in which the writer spells each node's ``<node>``
    element, depth first. A reader walks those elements in the same order:
    a node reached again is instanced only after its ``<node>`` element
    (see :func:`_node_xml`), so a reader meets every copy after the node.

    A reader expands each ``<instance_node>`` into a copy of the subtree it
    names, every copy holding the node's own ``id``. A node whose ``id`` an
    earlier-ranked node holds, and whose subtree is that node's node for
    node, is such a copy: it is instanced from that node rather than
    written again, and each node of its subtree takes the rank of the node
    it copies.

    Parameters
    ----------
    scene
        The scene being written.
    roots_per_scene
        The root nodes of each scene, in the order they are written.

    Returns
    -------
    order : dict
        ``node -> rank`` for every node a scene reaches, copies included.
    alias : dict
        ``copy -> node`` for every node of a copied subtree.
    """
    order: dict[int, int] = {}
    alias: dict[int, int] = {}
    first_of_id: dict[str, int] = {}
    rank = 0
    for roots in roots_per_scene:
        for root in roots:
            stack = [root]
            while stack:
                idx = stack.pop()
                if idx in order or not (0 <= idx < len(scene.nodes)):
                    continue
                given = _xml_name(scene.nodes[idx].extras.get("id"))
                held = first_of_id.get(given) if given is not None else None
                pairs = (
                    _copy_pairs(scene, idx, held, order) if held is not None else None
                )
                if pairs is not None:
                    for copy, node in pairs:
                        alias[copy] = alias.get(node, node)
                        order[copy] = order[node]
                    continue
                order[idx] = rank
                rank += 1
                if given is not None:
                    first_of_id.setdefault(given, idx)
                stack.extend(reversed(scene.nodes[idx].children))
    return order, alias


def _copy_pairs(
    scene: SceneData, copy: int, node: int, order: dict[int, int]
) -> list[tuple[int, int]] | None:
    """Return ``(copy, node)`` pairs over two subtrees when one copies the other.

    Parameters
    ----------
    scene
        The scene being written.
    copy
        The root of the subtree that may be a copy, not ranked yet.
    node
        The root of the subtree it may copy, already ranked.
    order
        The ranks given so far.

    Returns
    -------
    list or None
        One pair per node of the two subtrees, matched child by child; None
        when they differ in a name, mesh, matrix, extras or child count,
        when a node of the copy is already ranked or appears twice, or when
        a node it would copy is not ranked yet.
    """
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    stack = [(copy, node)]
    n_nodes = len(scene.nodes)
    while stack:
        a, b = stack.pop()
        if not (0 <= a < n_nodes and 0 <= b < n_nodes):
            return None
        if a == b or a in seen or a in order or b not in order:
            return None
        na, nb = scene.nodes[a], scene.nodes[b]
        if (
            na.name != nb.name
            or na.mesh != nb.mesh
            or len(na.children) != len(nb.children)
            or not np.array_equal(na.matrix, nb.matrix)
            or not _same(na.extras, nb.extras)
        ):
            return None
        seen.add(a)
        pairs.append((a, b))
        stack.extend(zip(na.children, nb.children, strict=True))
    return pairs


def _same(a: Any, b: Any) -> bool:
    """Return whether two extras compare equal, False when they cannot be compared.

    Parameters
    ----------
    a, b
        The values to compare.

    Returns
    -------
    bool
        True when ``a == b`` holds as a single truth value.
    """
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _float_array(value: Any, what: str) -> np.ndarray:
    """Return ``value`` as a float64 array, refusing what does not convert."""
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CodecError(f"{what} is not numeric: {exc}") from exc


def _spelled(node: SceneNode) -> tuple[list[dict[str, Any]], bool]:
    """Return the transform elements to spell for a node, and whether they are its own.

    The elements of ``extras["transforms"]`` are kept when they still compose
    to ``node.matrix``; otherwise one ``<matrix sid="transform">`` stands in
    and the second value is False. A sid that is not an XML name, or that
    an earlier element of the node already spells, is None: sids are unique
    within their parent, and a channel reaches the first element holding it.
    """
    spelled = node.extras.get("transforms")
    if isinstance(spelled, list) and spelled:
        numbers: list[list[float]] = []
        try:
            matrix = np.eye(4)
            for t in spelled:
                values = np.asarray(t["values"], dtype=np.float64).ravel()
                if t["kind"] not in _KIND_SIZE or values.size != _KIND_SIZE[t["kind"]]:
                    raise ValueError
                matrix = matrix @ _matrix_of(t["kind"], values)
                numbers.append(values.tolist())
        except (KeyError, TypeError, ValueError):
            matrix = None
        # numpy's default rtol would accept a far node moved by a
        # fraction of a unit, and a fixed atol a tiny node rescaled by
        # orders of magnitude; both would write stale transforms back.
        target = np.asarray(node.matrix, dtype=np.float64)
        if matrix is not None and np.allclose(
            matrix,
            target,
            rtol=1e-12,
            atol=1e-12 * max(np.abs(target).max(), np.abs(matrix).max()),
        ):
            out: list[dict[str, Any]] = []
            seen: set[str] = set()
            for t, values in zip(spelled, numbers, strict=True):
                sid = _transform_sid(t.get("sid"))
                if sid in seen:
                    sid = None
                elif sid is not None:
                    seen.add(sid)
                out.append({**t, "sid": sid, "values": values})
            return out, True
    return [
        {
            "kind": "matrix",
            "sid": "transform",
            "values": np.asarray(node.matrix, dtype=np.float64).ravel().tolist(),
        }
    ], False


def _roots(scene: SceneData) -> tuple[int, ...]:
    children: set[int] = set()
    for node in scene.nodes:
        children.update(node.children)
    return tuple(i for i in range(len(scene.nodes)) if i not in children)


def _shared_nodes(
    scene: SceneData,
    roots_per_scene: list[tuple[int, ...]],
    alias: dict[int, int],
    name: str,
) -> list[int]:
    """Return the nodes reached more than once, in depth-first order.

    Each is written once in ``<library_nodes>`` and instanced from there by
    every parent and scene reaching it; a reader resolves an
    ``<instance_node>`` only reliably against that library.

    Raises
    ------
    CodecError
        When the node graph has a cycle or names a node the scene lacks.
    """
    n_nodes = len(scene.nodes)
    refs: dict[int, int] = {}
    order: list[int] = []
    state: dict[int, int] = {}
    for roots in roots_per_scene:
        for root in roots:
            if not (0 <= root < n_nodes):
                raise CodecError(
                    f"{name!r}: a scene names root node {root}, which the scene lacks."
                )
            root = alias.get(root, root)
            refs[root] = refs.get(root, 0) + 1
            if root in state:
                continue
            stack = [(root, iter(scene.nodes[root].children))]
            state[root] = 1
            order.append(root)
            while stack:
                idx, it = stack[-1]
                child = next(it, None)
                if child is None:
                    state[idx] = 2
                    stack.pop()
                    continue
                if not (0 <= child < n_nodes):
                    raise CodecError(
                        f"{name!r}: node {idx} names child {child}, which the "
                        "scene lacks."
                    )
                child = alias.get(child, child)
                refs[child] = refs.get(child, 0) + 1
                if state.get(child) == 1:
                    raise CodecError(
                        f"{name!r}: the node graph has a cycle through node {child}."
                    )
                if child not in state:
                    state[child] = 1
                    order.append(child)
                    stack.append((child, iter(scene.nodes[child].children)))
    return [n for n in order if refs[n] > 1]


def _node_xml(
    scene: SceneData,
    ids: _Ids,
    idx: int,
    binds: list[str],
    shared: set[int],
    depth: int,
    name: str,
) -> str:
    """Return the ``<node>`` of node ``idx`` and the subtree below it.

    A child in ``shared`` is written in ``<library_nodes>`` and instanced
    here with ``<instance_node>``, which the schema places before the child
    ``<node>`` elements, so a reader lists such children first.
    """
    node = scene.nodes[idx]
    pad = "  " * depth
    joint = ' type="JOINT"' if node.extras.get("type") == "JOINT" else ""
    given = ids.given_sids[idx]
    sid_attr = f' sid="{given}"' if given is not None else ""
    parts = [
        f'{pad}<node id="{ids.node(idx)}"{sid_attr}{_name_attr(node.name)}{joint}>\n'
    ]
    spelled, own = _spelled(node)
    if own and any(
        t["sid"] is None
        and _transform_sid(node.extras["transforms"][k].get("sid")) is not None
        for k, t in enumerate(spelled)
    ):
        _warn(
            f"{name!r}: node {idx} spells a transform sid that repeats an "
            "earlier one; it is written without one."
        )
    for t in spelled:
        sid = f' sid="{t["sid"]}"' if t.get("sid") else ""
        parts.append(
            f"{pad}  <{t['kind']}{sid}>{_nums(np.asarray(t['values']))}</{t['kind']}>\n"
        )
    if node.mesh is not None:
        if not (0 <= node.mesh < len(scene.meshes)):
            raise CodecError(
                f"{name!r}: node {idx} names mesh {node.mesh}, which the scene lacks."
            )
        parts.append(
            f'{pad}  <instance_geometry url="#{ids.geometry(node.mesh)}">'
            f"{binds[node.mesh]}</instance_geometry>\n"
        )
    dropped = [k for k in ("camera", "light") if node.extras.get(k) is not None]
    if dropped:
        _warn(
            f"{name!r}: node {idx} instances a {' and a '.join(dropped)}, which "
            "this writer does not carry; the instance is dropped."
        )
    instanced: list[str] = []
    nested: list[str] = []
    for child in node.children:
        child = ids.alias.get(child, child)
        if child in shared:
            instanced.append(f'{pad}  <instance_node url="#{ids.node(child)}"/>\n')
        else:
            nested.append(_node_xml(scene, ids, child, binds, shared, depth + 1, name))
    parts.extend(instanced)
    parts.extend(nested)
    parts.append(f"{pad}</node>\n")
    return "".join(parts)


def _bind_xml(
    ids: _Ids, used: list[int], textured: set[int], uv_set: int | None
) -> str:
    """Return the ``<bind_material>`` of a mesh whose blocks name ``used``.

    An effect samples its textures through the ``UVMap`` texcoord semantic,
    which only a ``<bind_vertex_input>`` ties to the mesh's lowest written
    ``TEXCOORD`` set, ``uv_set``; a mesh without one binds nothing.
    """
    if not used:
        return ""
    entries = []
    for m in used:
        mid = ids.material(m)
        uv = (
            '<bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" '
            f'input_set="{uv_set}"/>'
            if m in textured and uv_set is not None
            else ""
        )
        entries.append(
            f'<instance_material symbol="{mid}" target="#{mid}">{uv}</instance_material>'
        )
    return (
        "<bind_material><technique_common>"
        + "".join(entries)
        + "</technique_common></bind_material>"
    )


_PARAMS_OF_WIDTH = {
    1: ("X",),
    2: ("X", "Y"),
    3: ("X", "Y", "Z"),
    4: ("X", "Y", "Z", "W"),
    16: ("TRANSFORM",),
}


def _animations_xml(scene: SceneData, ids: _Ids, written: set[int], name: str) -> str:
    out: list[str] = []
    animations = scene.global_attrs.get("animations", [])
    if not isinstance(animations, (list, tuple)):
        raise CodecError(
            f"{name!r}: global_attrs['animations'] must be a list, not {animations!r}."
        )
    for a, anim in enumerate(animations):
        if not isinstance(anim, dict):
            raise CodecError(f"{name!r}: animation {a} is not a dict: {anim!r}.")
        aid = ids.animation(a)
        samplers = anim.get("samplers", [])
        channel_list = anim.get("channels", [])
        if not isinstance(samplers, (list, tuple)) or not isinstance(
            channel_list, (list, tuple)
        ):
            raise CodecError(
                f"{name!r}: animation {a} needs 'channels' and 'samplers' as lists."
            )
        sources: list[str] = []
        sampler_xml: list[str] = []
        # A node index holds the place of the one channel its baked keys get.
        channels: list[str | int] = []
        emitted: set[int] = set()
        baked: dict[int, dict[str, Any]] = {}
        # A reader gives each copy of an instanced node the channel of the
        # node; written once for the node, it reaches every copy again.
        copied: dict[tuple[Any, ...], bool] = {}
        lone: dict[tuple[Any, ...], int] = {}
        targeted: set[str] = set()
        unusable: dict[int, str | None] = {}
        for c, channel in enumerate(channel_list):
            if not isinstance(channel, dict):
                raise CodecError(
                    f"{name!r}: animation {a} channel {c} is not a dict: {channel!r}."
                )
            target = channel.get("target")
            if not isinstance(target, dict):
                target = {}
            node_idx = _index(target.get("node"))
            s = _index(channel.get("sampler"))
            if (
                node_idx is None
                or node_idx not in written
                or s is None
                or not (0 <= s < len(samplers))
            ):
                _warn(
                    f"{name!r}: animation {a} channel {c} names a node or sampler "
                    "the written scene lacks; it is dropped."
                )
                continue
            is_copy = node_idx in ids.alias
            copy_idx = node_idx
            node_idx = ids.alias.get(node_idx, node_idx)
            key = (
                node_idx,
                s,
                *(repr(target.get(k)) for k in ("sid", "path", "member")),
            )
            if not is_copy:
                lone.pop(key, None)
            if key in copied and (is_copy or copied[key]):
                continue
            if key not in copied and is_copy:
                lone[key] = copy_idx
            copied.setdefault(key, is_copy)
            spelled, own = _spelled(scene.nodes[node_idx])
            sid = target.get("sid")
            path = target.get("path")
            member = target.get("member")
            if sid is None and not own and path in _BAKED_PATHS and not member:
                tracks = baked.get(node_idx, {})
                if path in tracks:
                    _warn(
                        f"{name!r}: animation {a} channel {c} animates the {path} of "
                        f"node {node_idx} a second time; the first is kept."
                    )
                    continue
                track = _trs_track(samplers[s], path, a, s, name)
                if isinstance(track, str):
                    _warn(
                        f"{name!r}: animation {a} channel {c} has no sid and {track}; "
                        "it is dropped."
                    )
                    continue
                if not tracks:
                    address = f"{ids.node(node_idx)}/transform"
                    if address in targeted:
                        _warn(
                            f"{name!r}: animation {a} channel {c} animates "
                            f"{address!r} a second time; the first is kept."
                        )
                        continue
                    targeted.add(address)
                    channels.append(node_idx)
                    baked[node_idx] = tracks
                tracks[path] = track
                continue
            if sid is None:
                reason = _sidless_reason(spelled, path, member, samplers[s])
                if reason is not None:
                    _warn(
                        f"{name!r}: animation {a} channel {c} has no sid and {reason}; "
                        "it is dropped."
                    )
                    continue
            hit = None
            for t in spelled:
                if (sid is not None and t.get("sid") == sid) or (
                    sid is None and t["kind"] == _KIND_OF_PATH.get(path)
                ):
                    hit = t
                    break
            if hit is None or not hit.get("sid"):
                _warn(
                    f"{name!r}: animation {a} channel {c} targets {sid or path!r} on node "
                    f"{node_idx}, which is written without such a transform; it is dropped."
                )
                continue
            if (
                member is not None
                and member != ""
                and (not isinstance(member, str) or not _MEMBER.match(member))
            ):
                _warn(
                    f"{name!r}: animation {a} channel {c} has member {member!r}, "
                    "which is neither a name nor (i)(j) indices; it is dropped."
                )
                continue
            if s not in unusable:
                unusable[s] = _keys_reason(samplers[s])
            if unusable[s] is not None:
                _warn(
                    f"{name!r}: animation {a} channel {c} has {unusable[s]}; "
                    "it is dropped."
                )
                continue
            base = f"{aid}-s{s}"
            suffix = (
                ""
                if not member
                else (member if member.startswith("(") else f".{member}")
            )
            address = f"{ids.node(node_idx)}/{hit['sid']}{suffix}"
            if address in targeted:
                _warn(
                    f"{name!r}: animation {a} channel {c} animates {address!r} "
                    "a second time; the first is kept."
                )
                continue
            targeted.add(address)
            if s not in emitted:
                emitted.add(s)
                src, smp = _sampler_xml(samplers[s], base, a, s, name)
                sources.append(src)
                sampler_xml.append(smp)
            channels.append(
                f'      <channel source="#{base}-sampler" target="{address}"/>\n'
            )
        for key, copy_idx in lone.items():
            _warn(
                f"{name!r}: animation {a} animates node {copy_idx}, a copy of "
                f"instanced node {key[0]}, alone; its channel is written for the "
                "node and reaches every instance."
            )
        for k, (node_idx, tracks) in enumerate(baked.items()):
            base = f"{aid}-bake{k}"
            keys = _bake_trs(scene.nodes[node_idx].matrix, tracks, node_idx, name)
            src, smp = _sampler_xml(keys, base, a, len(samplers) + k, name)
            sources.append(src)
            sampler_xml.append(smp)
            channels[channels.index(node_idx)] = (
                f'      <channel source="#{base}-sampler" '
                f'target="{ids.node(node_idx)}/transform"/>\n'
            )
        if channels:
            # The schema's sequence is every source, then every sampler, then
            # every channel; interleaving them per sampler fails validation.
            out.append(
                f'    <animation id="{aid}"{_name_attr(str(anim.get("name") or ""))}>\n'
                + "".join(sources)
                + "".join(sampler_xml)
                + "".join(channels)
                + "    </animation>\n"
            )
    return "".join(out)


def _sidless_reason(
    spelled: list[dict[str, Any]], path: Any, member: Any, sampler: Any
) -> str | None:
    """Return why a channel without a sid cannot be bound, None when it can.

    Without a sid only the path says which element moves, so it binds when
    exactly one element of the node has that kind and each key holds that
    element's whole value. A ``rotation`` never does: a ``<rotate>`` is an
    axis and an angle in degrees, while a channel with no sid comes from a
    format whose rotation is a quaternion of the same width.
    """
    kind = _KIND_OF_PATH.get(path) if isinstance(path, str) else None
    if kind is None or kind == "rotate":
        return f"path {path!r}, which names no single COLLADA transform"
    if member not in (None, ""):
        return f"member {member!r}, which only a sid can place"
    same = [t for t in spelled if t["kind"] == kind]
    if len(same) != 1:
        return f"its node spells {len(same)} <{kind}> elements, not one"
    interp = (
        sampler.get("interpolation", "LINEAR") if isinstance(sampler, dict) else None
    )
    if isinstance(interp, str):
        interp = interp.upper()
    if interp not in _INTERPOLATIONS:
        return f"interpolation {interp!r}, which COLLADA does not name"
    values = sampler.get("values") if isinstance(sampler, dict) else None
    try:
        width = _rows(np.asarray(values, dtype=np.float64)).shape[1]
    except (TypeError, ValueError):
        return None
    if width != _KIND_SIZE[kind]:
        return f"{width} values per key where <{kind}> holds {_KIND_SIZE[kind]}"
    return None


def _keys_reason(sampler: Any) -> str | None:
    """Return why a sampler's keys cannot be written, None when they can.

    A sampler :func:`_sampler_xml` refuses outright - not a dict, without
    ``times`` or ``values``, or holding something other than numbers - is
    left for it to refuse.
    """
    if not isinstance(sampler, dict):
        return None
    try:
        times = np.asarray(sampler.get("times"), dtype=np.float64).ravel()
        values = np.asarray(sampler.get("values"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if "times" not in sampler or "values" not in sampler:
        return None
    if not len(times):
        return "a sampler without keys"
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        return "keys that are not finite"
    if (np.diff(times) <= 0).any():
        return "key times that do not increase"
    for key in ("in_tangents", "out_tangents"):
        try:
            tangents = np.asarray(sampler.get(key, 0.0), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(tangents).all():
            return f"{key} that are not finite"
    return None


def _rows(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as one row per key, a matrix per key flattened to 16."""
    if values.ndim == 2:
        return values
    n = values.shape[0] if values.ndim else 1
    return values.reshape(n, values.size // n if n else 1)


def _trs_track(
    sampler: Any, path: str, a: int, s: int, name: str
) -> tuple[np.ndarray, np.ndarray, str] | str:
    """Return a glTF-style sampler as ``(times, values, interpolation)`` to bake.

    Parameters
    ----------
    sampler
        The animation's sampler the channel names.
    path
        ``translation``, ``rotation`` or ``scale``.
    a, s
        The animation and sampler indices, for messages.
    name
        The output's name, for messages.

    Returns
    -------
    tuple or str
        The key times, the values - ``(n, k)``, or ``(n, 3, k)`` in-tangent,
        value, out-tangent rows for ``CUBICSPLINE`` - and the interpolation;
        or the reason the channel cannot be baked.

    Raises
    ------
    CodecError
        When the sampler is not a dict holding numeric ``times`` and
        ``values``, as :func:`_sampler_xml` refuses.
    """
    if (
        not isinstance(sampler, dict)
        or "times" not in sampler
        or "values" not in sampler
    ):
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} needs 'times' and 'values'."
        )
    try:
        times = np.asarray(sampler["times"], dtype=np.float64).ravel()
        values = _rows(np.asarray(sampler["values"], dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} has times or values that are not numbers."
        ) from exc
    interp = sampler.get("interpolation", "LINEAR")
    if isinstance(interp, str):
        interp = interp.upper()
    if interp not in _BAKED_INTERPOLATIONS:
        return f"interpolation {interp!r}, which a {path} key cannot be baked with"
    width = 4 if path == "rotation" else 3
    rows = 3 * len(times) if interp == "CUBICSPLINE" else len(times)
    if not len(times):
        return "a sampler without keys"
    if values.shape != (rows, width):
        return (
            f"{values.shape[0]} keys of {values.shape[1]} values where its "
            f"{len(times)} {interp} {path} keys need {rows} of {width}"
        )
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        return "keys that are not finite"
    if (np.diff(times) <= 0).any():
        return "key times that do not increase"
    if interp == "CUBICSPLINE":
        values = values.reshape(len(times), 3, width)
    return times, values, interp


def _rotations_of_quats(q: np.ndarray) -> np.ndarray:
    """Return the ``(n, 3, 3)`` rotation matrices of ``(n, 4)`` unit quaternions."""
    x, y, z, w = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def _unit_quats(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return np.where(norm > 0, q / np.where(norm > 0, norm, 1.0), [0.0, 0.0, 0.0, 1.0])


def _slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return the shortest-arc interpolation of quaternion rows at fractions ``u``."""
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sin = np.sin(theta)
    near = sin < 1e-9
    safe = np.where(near, 1.0, sin)
    w0 = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / safe)
    w1 = np.where(near, u, np.sin(u * theta) / safe)
    return _unit_quats(w0[:, None] * q0 + w1[:, None] * q1)


def _evaluate(
    track: tuple[np.ndarray, np.ndarray, str], at: np.ndarray, *, rotation: bool
) -> np.ndarray:
    """Return a track's value at every time of ``at``, held past its ends.

    ``LINEAR`` blends a rotation by slerp, as glTF specifies, and
    ``CUBICSPLINE`` is glTF's Hermite spline, its tangents scaled by the
    key interval.
    """
    times, values, interp = track
    keys = values[:, 1] if interp == "CUBICSPLINE" else values
    n = len(times)
    if n == 1 or interp == "STEP":
        out = keys[np.clip(np.searchsorted(times, at, side="right") - 1, 0, n - 1)]
    else:
        k = np.clip(np.searchsorted(times, at, side="right") - 1, 0, n - 2)
        dt = times[k + 1] - times[k]
        u = np.clip((at - times[k]) / dt, 0.0, 1.0)
        if interp == "LINEAR":
            if rotation:
                return _slerp(_unit_quats(keys[k]), _unit_quats(keys[k + 1]), u)
            out = keys[k] + u[:, None] * (keys[k + 1] - keys[k])
        else:
            u2, u3 = u * u, u * u * u
            out = (
                (2 * u3 - 3 * u2 + 1)[:, None] * keys[k]
                + (dt * (u3 - 2 * u2 + u))[:, None] * values[k, 2]
                + (3 * u2 - 2 * u3)[:, None] * keys[k + 1]
                + (dt * (u3 - u2))[:, None] * values[k + 1, 0]
            )
    return _unit_quats(out) if rotation else out


def _bake_trs(
    matrix: Any,
    tracks: dict[str, tuple[np.ndarray, np.ndarray, str]],
    node_idx: int,
    name: str,
) -> dict[str, Any]:
    """Return one ``<matrix>`` sampler composing a node's translation, rotation and scale keys.

    COLLADA animates a transform element, and the rotation a glTF-style
    channel holds is a quaternion no ``<rotate>`` spells, so the node's
    tracks are composed into its matrix instead. The keys are every track's
    times; a path no track animates holds the node's rest value. Since
    COLLADA blends a matrix element by element, a span rotating more than
    ``_BAKE_MAX_DEGREES`` under a rotation that blends and every span of a
    ``CUBICSPLINE`` track are split by extra keys. When some tracks are ``STEP`` and others are not,
    a key just before each step holds the value before it; when all are,
    the sampler is ``STEP``.

    Parameters
    ----------
    matrix
        The node's rest matrix.
    tracks
        ``path -> (times, values, interpolation)``, from :func:`_trs_track`.
    node_idx
        The node's index, for messages.
    name
        The output's name, for messages.

    Returns
    -------
    dict
        A sampler with ``times``, ``(n, 16)`` row-major ``values`` and an
        ``interpolation``.
    """
    rest_t, rest_q, rest_s, exact = trs_of_matrix(matrix)
    if not exact:
        _warn(
            f"{name!r}: node {node_idx} has a matrix that is not a translation, "
            "rotation and scale; its baked animation keys drop the rest (shear "
            "or projection)."
        )
    times = np.unique(np.concatenate([t[0] for t in tracks.values()]))
    kinds = {t[2] for t in tracks.values()}
    if kinds != {"STEP"}:
        if "CUBICSPLINE" in kinds:
            times = _split_spans(times, np.full(len(times) - 1, _BAKE_CUBIC_PIECES))
        # A rotation that steps turns at its keys alone, which no split of
        # the span before one brings under the cap. A cubic span can
        # overshoot between its keys, so the spans are measured again after
        # each split.
        turning = "rotation" in tracks and tracks["rotation"][2] != "STEP"
        for _ in range(_BAKE_MAX_PASSES if turning else 0):
            q = _evaluate(tracks["rotation"], times, rotation=True)
            dot = np.clip(np.abs(np.sum(q[:-1] * q[1:], axis=1)), 0.0, 1.0)
            degrees = 2.0 * np.degrees(np.arccos(dot))
            pieces = np.ceil(degrees / _BAKE_MAX_DEGREES - 1e-9).astype(np.int64)
            if (pieces <= 1).all():
                break
            times = _split_spans(times, np.maximum(pieces, 1))
        steps = [t[0][1:] for t in tracks.values() if t[2] == "STEP"]
        if steps:
            # Added after the splits, which would cut the span between a
            # held key and its step into keys no float32 tells apart. Many
            # importers read times as float32, where a float64 ulp before
            # the step would land on the step itself.
            at_step = np.concatenate(steps).astype(np.float32)
            held = np.nextafter(at_step, np.float32(-np.inf)).astype(np.float64)
            times = np.unique(np.concatenate([times, held]))
    n = len(times)

    def at(path: str, rest: np.ndarray) -> np.ndarray:
        if path in tracks:
            return _evaluate(tracks[path], times, rotation=path == "rotation")
        return np.tile(rest, (n, 1))

    keys = np.zeros((n, 4, 4), dtype=np.float64)
    keys[:, :3, :3] = (
        _rotations_of_quats(at("rotation", rest_q)) * at("scale", rest_s)[:, None, :]
    )
    keys[:, :3, 3] = at("translation", rest_t)
    keys[:, 3, 3] = 1.0
    return {
        "times": times,
        "values": keys.reshape(n, 16),
        "interpolation": "STEP" if kinds == {"STEP"} else "LINEAR",
    }


def _split_spans(times: np.ndarray, pieces: np.ndarray) -> np.ndarray:
    """Return ``times`` with span ``k`` cut into ``pieces[k]`` equal parts."""
    span = np.repeat(np.arange(len(pieces)), pieces)
    step = np.arange(len(span)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    split = times[span] + (times[span + 1] - times[span]) * step / pieces[span]
    return np.unique(np.concatenate([split, times]))


def _sampler_xml(
    sampler: dict[str, Any], base: str, a: int, s: int, name: str
) -> tuple[str, str]:
    """Return the sources and the ``<sampler>`` of one animation sampler, apart."""
    if (
        not isinstance(sampler, dict)
        or "times" not in sampler
        or "values" not in sampler
    ):
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} needs 'times' and 'values'."
        )
    try:
        times = np.asarray(sampler["times"], dtype=np.float64).ravel()
        values = np.asarray(sampler["values"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} has times or values that are not numbers."
        ) from exc
    values = _rows(values)
    if len(times) != len(values):
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} has {len(times)} times but {len(values)} values."
        )
    width = values.shape[1]
    params = _PARAMS_OF_WIDTH.get(width, tuple(f"C{k}" for k in range(width)))
    spelled = str(sampler.get("interpolation", "LINEAR"))
    interp = spelled.upper()
    if interp not in _INTERPOLATIONS:
        raise CodecError(
            f"{name!r}: animation {a} sampler {s} has interpolation {spelled!r}; "
            f"COLLADA names {', '.join(_INTERPOLATIONS)}."
        )
    body: list[str] = []
    inputs = [
        f'<input semantic="INPUT" source="#{base}-input"/>',
        f'<input semantic="OUTPUT" source="#{base}-output"/>',
        f'<input semantic="INTERPOLATION" source="#{base}-interpolation"/>',
    ]
    body.append(_source_xml(f"{base}-input", times.reshape(-1, 1), ("TIME",), indent=6))
    body.append(_source_xml(f"{base}-output", values, params, indent=6))
    body.append(
        _name_source_xml(
            f"{base}-interpolation", [interp] * len(times), "INTERPOLATION", indent=6
        )
    )
    for key, semantic in (
        ("in_tangents", "IN_TANGENT"),
        ("out_tangents", "OUT_TANGENT"),
    ):
        tangents = sampler.get(key)
        if tangents is not None:
            try:
                tangents = _rows(np.asarray(tangents, dtype=np.float64))
            except (TypeError, ValueError) as exc:
                raise CodecError(
                    f"{name!r}: animation {a} sampler {s} has {key} that are not numbers."
                ) from exc
            if len(tangents) != len(times):
                raise CodecError(
                    f"{name!r}: animation {a} sampler {s} has {len(times)} times "
                    f"but {len(tangents)} {key[:-1]} values."
                )
            tp = _PARAMS_OF_WIDTH.get(
                tangents.shape[1], tuple(f"C{k}" for k in range(tangents.shape[1]))
            )
            body.append(_source_xml(f"{base}-{key}", tangents, tp, indent=6))
            inputs.append(f'<input semantic="{semantic}" source="#{base}-{key}"/>')
    return "".join(
        body
    ), f'      <sampler id="{base}-sampler">{"".join(inputs)}</sampler>\n'
