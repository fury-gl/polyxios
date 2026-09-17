"""glTF 2.0 codec — reads and writes ``.gltf`` and ``.glb`` files.

Registered for both ``.gltf`` (JSON + external ``.bin``) and ``.glb``
(single binary container).  :func:`read` flattens the scene graph into a
single :class:`~polyxios.PolyData` and warns the caller; use
:func:`read_scene` to obtain the full :class:`~polyxios._scene.SceneData`.
"""

import base64
import dataclasses
import json
from pathlib import Path
import struct
from typing import Any
import urllib.parse
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, QUADRATIC_SURFACE_CORNERS
from polyxios._io import (
    Source,
    format_suffix,
    is_buffer,
    read_bytes,
    source_name,
    write_bytes,
    write_text,
)
from polyxios._scene import (
    SceneData,
    SceneImage,
    SceneMaterial,
    SceneNode,
    SceneTexture,
)
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header
from polyxios.version import version as __version__

EXTENSION: str = ".gltf"
EXTENSIONS: tuple[str, ...] = (".gltf", ".glb")

# ----- glTF binary constants -------------------------------------------------

_GLB_MAGIC: int = 0x46546C67  # b'glTF' little-endian
_CHUNK_JSON: int = 0x4E4F534A  # b'JSON'
_CHUNK_BIN: int = 0x004E4942  # b'BIN\x00'
_GLB_HEADER_SIZE: int = 12
_CHUNK_HEADER_SIZE: int = 8

# ----- accessor component type → numpy dtype (little-endian) ----------------

_COMPONENT_DTYPE: dict[int, str] = {
    5120: "<i1",
    5121: "<u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}

# ----- accessor type string → component count --------------------------------

_TYPE_COMPONENTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

# ----- element type codes used during write ----------------------------------

_TRI_CODE = ELEMENT_TYPES["triangle"]
_QUAD_CODE = ELEMENT_TYPES["quad"]
_POLY_CODE = ELEMENT_TYPES["polygon"]
_LINE_CODE = ELEMENT_TYPES["line"]
_VERTEX_CODE = ELEMENT_TYPES["vertex"]
_TRI_STRIP_CODE = ELEMENT_TYPES["triangle_strip"]
_POLY_LINE_CODE = ELEMENT_TYPES["poly_line"]
_PIXEL_CODE = ELEMENT_TYPES["pixel"]

# Volume element codes — skipped with a warning on write.
_VOLUME_CODES: frozenset[int] = frozenset(
    {
        ELEMENT_TYPES["tetra"],
        ELEMENT_TYPES["hexahedron"],
        ELEMENT_TYPES["wedge"],
        ELEMENT_TYPES["pyramid"],
        ELEMENT_TYPES["voxel"],
        ELEMENT_TYPES["pentagonal_prism"],
        ELEMENT_TYPES["hexagonal_prism"],
    }
)

# Pixel and quadratic surface codes — no glTF mode; skipped with a warning.
_SKIP_WARN_CODES: frozenset[int] = frozenset({_PIXEL_CODE, *QUADRATIC_SURFACE_CORNERS})


def _safe_resolve(base_dir: Path, uri: str) -> Path:
    """Resolve *uri* relative to *base_dir*, rejecting path traversal.

    Parameters
    ----------
    base_dir
        Directory that owns the ``.gltf`` file.
    uri
        Percent-encoded relative URI from the glTF JSON (not a data: URI).

    Raises
    ------
    CodecError
        If the decoded URI is absolute or would escape *base_dir*.
    """
    decoded = urllib.parse.unquote(uri)
    p = Path(decoded)
    if p.is_absolute():
        raise CodecError(
            f"glTF: external URI {uri!r} is absolute; rejected for security."
        )
    resolved = (base_dir.resolve() / decoded).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise CodecError(
            f"glTF: external URI {uri!r} escapes the base directory "
            f"({base_dir}); rejected for security."
        )
    return resolved


# =============================================================================
# GLB parsing
# =============================================================================


def _parse_glb(data: bytes) -> tuple[dict, bytes | None]:
    """Parse a GLB binary and return (gltf_json_dict, bin_chunk_bytes_or_None)."""
    if len(data) < _GLB_HEADER_SIZE:
        raise CodecError("GLB file is too short to contain a valid header.")

    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise CodecError(
            f"Not a GLB file: expected magic 0x{_GLB_MAGIC:08X}, got 0x{magic:08X}."
        )
    if version != 2:
        raise CodecError(
            f"Unsupported GLB version {version}; only version 2 is supported."
        )
    if length != len(data):
        raise CodecError(
            f"GLB header reports {length} bytes but file has {len(data)} bytes."
        )

    pos = _GLB_HEADER_SIZE
    gltf_json: dict | None = None
    bin_data: bytes | None = None

    while pos < len(data):
        if pos + _CHUNK_HEADER_SIZE > len(data):
            raise CodecError("GLB chunk header extends past end of file.")
        chunk_length, chunk_type = struct.unpack_from("<II", data, pos)
        pos += _CHUNK_HEADER_SIZE
        if pos + chunk_length > len(data):
            raise CodecError("GLB chunk data extends past end of file.")
        chunk_data = data[pos : pos + chunk_length]
        pos += chunk_length

        if chunk_type == _CHUNK_JSON:
            gltf_json = json.loads(chunk_data.rstrip(b" ").decode("utf-8"))
        elif chunk_type == _CHUNK_BIN:
            bin_data = bytes(chunk_data)

    if gltf_json is None:
        raise CodecError("GLB file contains no JSON chunk.")
    return gltf_json, bin_data


# =============================================================================
# Accessor reading
# =============================================================================


def _read_accessor(
    gltf: dict,
    accessor_index: int,
    buffers: list[bytes],
) -> np.ndarray:
    """Decode a glTF accessor into a numpy array.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    accessor_index
        Index into ``gltf["accessors"]``.
    buffers
        Raw buffer bytes, one entry per ``gltf["buffers"]`` entry.

    Returns
    -------
    numpy.ndarray
        Shape ``(count,)`` for SCALAR, ``(count, n_components)`` otherwise.

    Raises
    ------
    CodecError
        On unknown component type, unknown accessor type, or out-of-bounds
        buffer access.
    """
    acc = gltf["accessors"][accessor_index]
    count: int = acc["count"]
    comp_type: int = acc["componentType"]
    acc_type: str = acc["type"]

    if comp_type not in _COMPONENT_DTYPE:
        raise CodecError(f"glTF: unknown accessor componentType {comp_type}.")
    if acc_type not in _TYPE_COMPONENTS:
        raise CodecError(f"glTF: unknown accessor type '{acc_type}'.")

    # Reject sparse accessors — silent zero geometry is worse than an error.
    if acc.get("sparse"):
        raise CodecError(
            f"glTF: accessor {accessor_index} uses sparse encoding which is not "
            "supported; re-export with dense accessors."
        )

    dtype = np.dtype(_COMPONENT_DTYPE[comp_type])
    n_comp = _TYPE_COMPONENTS[acc_type]
    byte_offset_acc: int = acc.get("byteOffset", 0)

    if "bufferView" not in acc:
        # No data (no bufferView, no sparse): return zeros.
        shape = (count,) if n_comp == 1 else (count, n_comp)
        return np.zeros(shape, dtype=dtype)

    bv = gltf["bufferViews"][acc["bufferView"]]
    buf_index: int = bv["buffer"]
    byte_offset_bv: int = bv.get("byteOffset", 0)
    byte_stride: int | None = bv.get("byteStride")
    buf = buffers[buf_index]

    start = byte_offset_bv + byte_offset_acc
    elem_size = dtype.itemsize * n_comp

    if byte_stride is not None and byte_stride != elem_size:
        # Vectorized interleaved read using stride_tricks.
        buf_u8 = np.frombuffer(buf, dtype=np.uint8)
        total = start + (count - 1) * byte_stride + elem_size
        if total > len(buf):
            raise CodecError(
                f"glTF: accessor {accessor_index} declares {count} elements "
                f"but buffer read at offset {total} exceeds buffer length {len(buf)}."
            )
        raw_u8 = (
            np.lib.stride_tricks.as_strided(
                buf_u8[start:],
                shape=(count, elem_size),
                strides=(byte_stride, 1),
            )
            .copy()
            .ravel()
        )
        raw = raw_u8.view(dtype).reshape((count, n_comp) if n_comp > 1 else (count,))
    else:
        end = start + count * elem_size
        if end > len(buf):
            raise CodecError(
                f"glTF: accessor {accessor_index} declares {count} elements "
                f"({end} bytes needed) but buffer only has {len(buf)} bytes."
            )
        raw = np.frombuffer(buf, dtype=dtype, count=count * n_comp, offset=start)
        if n_comp == 1:
            raw = raw.copy()
        else:
            raw = raw.reshape(count, n_comp).copy()

    # Honor the normalized flag — integer types scale to [-1, 1] or [0, 1].
    if acc.get("normalized"):
        if comp_type == 5121:  # UBYTE → [0, 1]
            return raw.astype(np.float64) / 255.0
        if comp_type == 5123:  # USHORT → [0, 1]
            return raw.astype(np.float64) / 65535.0
        if comp_type == 5120:  # BYTE → [-1, 1]
            return np.maximum(raw.astype(np.float64) / 127.0, -1.0)
        if comp_type == 5122:  # SHORT → [-1, 1]
            return np.maximum(raw.astype(np.float64) / 32767.0, -1.0)

    return raw


# =============================================================================
# Primitive → PolyData
# =============================================================================


def _primitive_to_polydata(
    gltf: dict,
    primitive: dict,
    buffers: list[bytes],
    file_size: int,
) -> PolyData:
    """Convert one glTF mesh primitive to a PolyData.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    primitive
        One entry from ``mesh["primitives"]``.
    buffers
        Raw buffer bytes.
    file_size
        Total source file size, forwarded to :func:`validate_header`.

    Returns
    -------
    PolyData
        Geometry and vertex attributes for this primitive.
    """
    attrs = primitive.get("attributes", {})
    if "POSITION" not in attrs:
        raise CodecError("glTF primitive is missing a POSITION attribute.")

    pos_raw = _read_accessor(gltf, attrs["POSITION"], buffers)
    vertices = pos_raw.astype(np.float64)
    n_verts = vertices.shape[0]

    mode: int = primitive.get("mode", 4)

    if "indices" in primitive:
        idx_raw = _read_accessor(gltf, primitive["indices"], buffers)
        indices = idx_raw.astype(np.int32).ravel()
    else:
        indices = np.arange(n_verts, dtype=np.int32)

    if indices.size > 0 and indices.max() >= n_verts:
        raise CodecError(
            f"glTF: primitive has index {indices.max()} but only {n_verts} vertices."
        )

    if mode == 4:  # TRIANGLES — vectorized
        n_tris = len(indices) // 3
        connectivity = indices[: n_tris * 3].astype(np.int32)
        offsets = np.arange(0, n_tris * 3 + 1, 3, dtype=np.int32)
        element_types = np.full(n_tris, _TRI_CODE, dtype=np.uint8)
        n_elements = n_tris
        conn_size = n_tris * 3
    else:
        conn_list: list[int] = []
        offsets_list: list[int] = [0]
        type_codes: list[int] = []

        if mode == 0:  # POINTS
            conn_list = indices.tolist()
            offsets_list = list(range(len(conn_list) + 1))
            type_codes = [_VERTEX_CODE] * len(conn_list)
        elif mode == 1:  # LINES
            n_pairs = len(indices) // 2
            pairs = indices[: n_pairs * 2].reshape(-1, 2)
            conn_list = pairs.ravel().tolist()
            offsets_list = list(range(0, len(conn_list) + 1, 2))
            type_codes = [_LINE_CODE] * n_pairs
        elif mode == 2:  # LINE_LOOP: strip + close
            if len(indices) > 0:
                loop = indices.tolist() + [int(indices[0])]
                conn_list.extend(loop)
                offsets_list.append(offsets_list[-1] + len(loop))
                type_codes.append(_POLY_LINE_CODE)
        elif mode == 3:  # LINE_STRIP
            conn_list.extend(indices.tolist())
            offsets_list.append(offsets_list[-1] + len(indices))
            type_codes.append(_POLY_LINE_CODE)
        elif mode == 5:  # TRIANGLE_STRIP
            conn_list.extend(indices.tolist())
            offsets_list.append(offsets_list[-1] + len(indices))
            type_codes.append(_TRI_STRIP_CODE)
        elif mode == 6:  # TRIANGLE_FAN: fan-triangulate
            for i in range(1, len(indices) - 1):
                conn_list.extend(
                    [int(indices[0]), int(indices[i]), int(indices[i + 1])]
                )
                offsets_list.append(offsets_list[-1] + 3)
                type_codes.append(_TRI_CODE)
        else:
            raise CodecError(f"glTF: unsupported primitive mode {mode}.")

        n_elements = len(type_codes)
        conn_size = len(conn_list)
        connectivity = np.array(conn_list, dtype=np.int32)
        offsets = np.array(offsets_list, dtype=np.int32)
        element_types = np.array(type_codes, dtype=np.uint8)
    validate_header(n_verts, n_elements, conn_size, file_size)

    # Vertex attributes.
    vertex_attrs: dict[str, np.ndarray] = {}

    for gltf_name, poly_name in _ATTR_MAP.items():
        if gltf_name not in attrs:
            continue
        raw = _read_accessor(gltf, attrs[gltf_name], buffers)
        if gltf_name == "JOINTS_0":
            vertex_attrs[poly_name] = raw.astype(np.int32)
        else:
            vertex_attrs[poly_name] = raw.astype(np.float64)

    if "COLOR_0" in attrs:
        raw = _read_accessor(gltf, attrs["COLOR_0"], buffers)
        vertex_attrs["colors"] = raw.astype(np.float64)

    for gltf_name in attrs:
        if (
            gltf_name not in _ATTR_MAP
            and gltf_name != "COLOR_0"
            and gltf_name != "POSITION"
        ):
            raw = _read_accessor(gltf, attrs[gltf_name], buffers)
            if gltf_name.startswith("TEXCOORD_"):
                n = gltf_name[len("TEXCOORD_") :]
                key = f"texcoords_{n}" if n != "0" else "texcoords"
            else:
                key = gltf_name.lower()
            vertex_attrs[key] = raw

    mat_idx: int = primitive.get("material", -1)
    element_attrs: dict[str, np.ndarray] = {}
    if mat_idx >= 0:
        element_attrs["material"] = np.full(n_elements, mat_idx, dtype=np.int32)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


# =============================================================================
# Mesh → merged PolyData
# =============================================================================


def _mesh_to_polydata(
    gltf: dict,
    mesh_index: int,
    buffers: list[bytes],
    file_size: int,
) -> PolyData:
    """Merge all primitives of one glTF mesh into a single PolyData.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    mesh_index
        Index into ``gltf["meshes"]``.
    buffers
        Raw buffer bytes.
    file_size
        Total source file size forwarded to :func:`validate_header`.

    Returns
    -------
    PolyData
        All primitives merged; ``element_attrs["material"]`` holds per-element
        material indices into the parent :class:`~polyxios._scene.SceneData`.
    """
    from polyxios import transforms

    mesh = gltf["meshes"][mesh_index]
    primitives = mesh.get("primitives", [])
    if not primitives:
        return PolyData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
        )

    polys = [
        _primitive_to_polydata(gltf, prim, buffers, file_size) for prim in primitives
    ]
    result = transforms.merge(*polys) if len(polys) > 1 else polys[0]
    mesh_name: str = mesh.get("name", "")
    if mesh_name:
        result = dataclasses.replace(
            result, global_attrs={**result.global_attrs, "mesh_name": mesh_name}
        )
    return result


# =============================================================================
# Node local transform
# =============================================================================


def _node_local_matrix(node: dict) -> np.ndarray:
    """Return the 4×4 local transform matrix for a glTF node dict."""
    if "matrix" in node:
        # glTF stores column-major; numpy is row-major, so transpose.
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T

    t = node.get("translation", [0.0, 0.0, 0.0])
    r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # xyzw
    s = node.get("scale", [1.0, 1.0, 1.0])

    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = t

    S = np.diag([s[0], s[1], s[2], 1.0])

    # Quaternion xyzw → 3×3 rotation matrix.
    x, y, z, w = r
    R = np.eye(4, dtype=np.float64)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)
    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - x * w)
    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * (x * x + y * y)

    return T @ R @ S


# =============================================================================
# read_scene
# =============================================================================


def _decode_animations(
    gltf: dict,
    animations: list[dict],
    buffers: list[bytes],
) -> list[dict]:
    """Decode glTF animation sampler accessors into numpy arrays.

    Replaces raw accessor indices with decoded ``times`` and ``values``
    arrays so callers do not need to re-read the buffer.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    animations
        The raw ``gltf["animations"]`` list.
    buffers
        Raw buffer bytes.

    Returns
    -------
    list[dict]
        One dict per animation, each with:
        - ``"name"`` : str
        - ``"channels"`` : list[dict] (raw — ``target.node`` and ``target.path``)
        - ``"samplers"`` : list[dict] with ``"times"`` (ndarray, shape (N,)),
          ``"values"`` (ndarray, shape (N, K) or (N,) for SCALAR), and
          ``"interpolation"`` str (``"LINEAR"``, ``"STEP"``, or
          ``"CUBICSPLINE"``).
    """
    result = []
    for anim in animations:
        decoded_samplers: list[dict] = []
        for s in anim.get("samplers", []):
            times = _read_accessor(gltf, s["input"], buffers).astype(np.float64)
            values = _read_accessor(gltf, s["output"], buffers).astype(np.float64)
            decoded_samplers.append(
                {
                    "times": times,
                    "values": values,
                    "interpolation": s.get("interpolation", "LINEAR"),
                }
            )
        result.append(
            {
                "name": anim.get("name", ""),
                "channels": anim.get("channels", []),
                "samplers": decoded_samplers,
            }
        )
    return result


def read_scene(path: Source) -> SceneData:
    """Read a glTF or GLB file and return a SceneData.

    Parameters
    ----------
    path
        Path to a ``.gltf`` or ``.glb`` file.

    Returns
    -------
    SceneData
        Full scene graph with meshes, nodes, materials, textures, and images.

    Raises
    ------
    CodecError
        On invalid GLB magic, unsupported version, missing JSON chunk,
        unknown accessor types, or buffer overreads.
    """
    raw = read_bytes(path)
    file_size = len(raw)
    suffix = format_suffix(path).lower()
    if suffix == ".glb" or raw[:4] == b"glTF":
        gltf, bin_chunk = _parse_glb(raw)
        buffers: list[bytes] = []
        # Iterate ALL buffers in order; for entry 0 without a uri the BIN
        # chunk (if any) is authoritative, otherwise resolve the uri normally.
        base_dir_glb = Path(str(path)).parent if not is_buffer(path) else None
        for _buf_idx, _buf_entry in enumerate(gltf.get("buffers", [])):
            _uri = _buf_entry.get("uri", "")
            if _buf_idx == 0 and bin_chunk is not None and not _uri:
                buffers.append(bin_chunk)
            elif _uri.startswith("data:"):
                _parts = _uri.split(",", 1)
                if len(_parts) != 2:
                    raise CodecError(
                        f"glTF: malformed data URI (no comma): {_uri[:80]!r}"
                    )
                _, _payload = _parts
                buffers.append(base64.b64decode(_payload))
            elif _uri:
                if base_dir_glb is None:
                    raise CodecError(
                        "glTF: cannot resolve external buffer URIs from a stream; "
                        "provide a filesystem path."
                    )
                buffers.append(read_bytes(_safe_resolve(base_dir_glb, _uri)))
            else:
                # No uri and no applicable BIN chunk — placeholder keeps index alignment.
                buffers.append(b"")
    else:
        try:
            gltf = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CodecError(f"glTF: invalid JSON — {exc}") from exc
        # Raise early when a stream has non-data URI buffers to resolve.
        raw_bufs = gltf.get("buffers", [])
        has_external = any(
            b.get("uri", "").strip() and not b.get("uri", "").startswith("data:")
            for b in raw_bufs
        )
        if has_external and is_buffer(path):
            raise CodecError(
                "glTF: cannot resolve external buffer URIs from a stream; "
                "provide a filesystem path."
            )
        base_dir = Path(str(path)).parent if not is_buffer(path) else Path(".")
        buffers = []
        for buf_entry in raw_bufs:
            uri: str = buf_entry.get("uri", "")
            if uri.startswith("data:"):
                parts = uri.split(",", 1)
                if len(parts) != 2:
                    raise CodecError(
                        f"glTF: malformed data URI (no comma): {uri[:80]!r}"
                    )
                _, payload = parts
                buffers.append(base64.b64decode(payload))
            elif uri:
                # Percent-decode and validate before resolving.
                buffers.append(read_bytes(_safe_resolve(base_dir, uri)))
            else:
                # Missing/empty uri — append placeholder to keep index alignment.
                buffers.append(b"")
        file_size = sum(len(b) for b in buffers) + len(raw)

    asset = gltf.get("asset", {})
    ver = str(asset.get("version", ""))
    if not ver.startswith("2"):
        raise CodecError(
            f"glTF: unsupported asset version '{ver}'; only 2.x is supported."
        )

    # Meshes.
    meshes = tuple(
        _mesh_to_polydata(gltf, i, buffers, file_size)
        for i in range(len(gltf.get("meshes", [])))
    )

    # Materials.
    materials = tuple(_build_material(m) for m in gltf.get("materials", []))

    # Samplers (needed for textures).
    samplers = gltf.get("samplers", [])

    # Textures.
    _tex_list: list[SceneTexture] = []
    for t in gltf.get("textures", []):
        src = t.get("source")
        if src is None:
            for ext_val in t.get("extensions", {}).values():
                if isinstance(ext_val, dict) and "source" in ext_val:
                    src = ext_val["source"]
                    break
        if src is None:
            # Placeholder preserves index alignment — dropping the entry would
            # shift every subsequent texture index by one.
            _tex_list.append(SceneTexture(image=-1))
            continue
        _tex_list.append(
            SceneTexture(
                image=src,
                **_sampler_fields(samplers[t["sampler"]] if "sampler" in t else {}),
            )
        )
    textures = tuple(_tex_list)

    # Images.
    images = tuple(_build_image(img, buffers, gltf) for img in gltf.get("images", []))

    # Nodes — synthesize when absent.
    raw_nodes = gltf.get("nodes", [])
    if raw_nodes:
        nodes = tuple(
            SceneNode(
                name=n.get("name", ""),
                mesh=n.get("mesh"),
                children=tuple(n.get("children", ())),
                matrix=_node_local_matrix(n),
                extras={
                    k: n[k]
                    for k in ("camera", "skin", "extensions", "extras")
                    if k in n
                },
            )
            for n in raw_nodes
        )
    else:
        # Mesh-only file: synthesize one identity node per mesh.
        nodes = tuple(SceneNode(mesh=i) for i in range(len(meshes)))

    # Scenes.
    raw_scenes = gltf.get("scenes", [])
    if raw_scenes:
        scenes: tuple[tuple[int, ...], ...] = tuple(
            tuple(s.get("nodes", ())) for s in raw_scenes
        )
    elif nodes:
        all_children: set[int] = set()
        for n in nodes:
            all_children.update(n.children)
        roots = tuple(i for i in range(len(nodes)) if i not in all_children)
        scenes = (roots,)
    else:
        scenes = ()

    active_scene: int = gltf.get("scene", 0)

    # Global attrs: preserve asset metadata and deferred data.
    global_attrs: dict[str, Any] = {"asset": asset}
    for key in ("extensions", "extras"):
        if key in gltf:
            global_attrs[key] = gltf[key]
    if "animations" in gltf:
        global_attrs["animations"] = _decode_animations(
            gltf, gltf["animations"], buffers
        )
    if "skins" in gltf:
        global_attrs["skins"] = gltf["skins"]

    return SceneData(
        meshes=meshes,
        nodes=nodes,
        materials=materials,
        textures=textures,
        images=images,
        scenes=scenes,
        active_scene=active_scene,
        name=gltf.get("name", ""),
        global_attrs=global_attrs,
    )


def _sampler_fields(sampler: dict) -> dict:
    """Extract SceneTexture kwargs from a glTF sampler dict."""
    out: dict = {}
    if "magFilter" in sampler:
        out["mag_filter"] = sampler["magFilter"]
    if "minFilter" in sampler:
        out["min_filter"] = sampler["minFilter"]
    if "wrapS" in sampler:
        out["wrap_s"] = sampler["wrapS"]
    if "wrapT" in sampler:
        out["wrap_t"] = sampler["wrapT"]
    return out


def _build_material(m: dict) -> SceneMaterial:
    """Build a SceneMaterial from a glTF material dict."""
    pbr = m.get("pbrMetallicRoughness", {})

    def _tex(slot: str) -> int | None:
        info = m.get(slot) or pbr.get(slot)
        return info["index"] if info else None

    return SceneMaterial(
        name=m.get("name", ""),
        base_color=tuple(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])),
        metallic=float(pbr.get("metallicFactor", 1.0)),
        roughness=float(pbr.get("roughnessFactor", 1.0)),
        emissive=tuple(m.get("emissiveFactor", [0.0, 0.0, 0.0])),
        alpha_mode=m.get("alphaMode", "OPAQUE"),
        alpha_cutoff=float(m.get("alphaCutoff", 0.5)),
        double_sided=bool(m.get("doubleSided", False)),
        base_color_texture=_tex("baseColorTexture"),
        normal_texture=_tex("normalTexture"),
        metallic_roughness_texture=_tex("metallicRoughnessTexture"),
        occlusion_texture=_tex("occlusionTexture"),
        emissive_texture=_tex("emissiveTexture"),
        extras=m.get("extras", {}),
    )


def _build_image(img: dict, buffers: list[bytes], gltf: dict) -> SceneImage:
    """Build a SceneImage from a glTF image dict."""
    if "bufferView" in img:
        bv = gltf["bufferViews"][img["bufferView"]]
        buf = buffers[bv["buffer"]]
        start = bv.get("byteOffset", 0)
        data = buf[start : start + bv["byteLength"]]
        return SceneImage(
            data=bytes(data), media_type=img.get("mimeType"), name=img.get("name", "")
        )
    uri: str = img.get("uri", "")
    if uri.startswith("data:"):
        _img_parts = uri.split(",", 1)
        if len(_img_parts) != 2:
            raise CodecError(f"glTF: malformed data URI (no comma): {uri[:80]!r}")
        mime, payload = _img_parts
        media_type = mime.split(";")[0].split(":", 1)[1] if ":" in mime else None
        return SceneImage(
            data=base64.b64decode(payload),
            media_type=media_type,
            name=img.get("name", ""),
        )
    return SceneImage(uri=uri, name=img.get("name", ""))


# =============================================================================
# read
# =============================================================================


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Read a glTF or GLB file and return a flattened PolyData.

    Parameters
    ----------
    path
        Path to a ``.gltf`` or ``.glb`` file.
    lazy
        Not supported for glTF; raises :exc:`~polyxios.exceptions.LazyReadError`.

    Returns
    -------
    PolyData
        All scene meshes merged into one flat mesh.  Scene hierarchy,
        materials, and textures are discarded; use :func:`read_scene` to
        preserve them.

    Warns
    -----
    UserWarning
        Always, to inform the caller that scene data is being discarded.

    Raises
    ------
    LazyReadError
        If ``lazy=True``.
    CodecError
        On malformed or unsupported glTF/GLB content.
    """
    if lazy:
        raise LazyReadError("glTF format does not support lazy reads.")
    warnings.warn(
        f"'{source_name(path)}' is a scene format (glTF): read() flattens "
        "the scene graph, materials, textures and hierarchy into a single "
        "PolyData. Use polyxios.read_scene() to preserve the full scene.",
        stacklevel=3,  # user → read → here
    )
    return read_scene(path).to_polydata()


# =============================================================================
# sniff
# =============================================================================


def sniff(head: bytes) -> bool:
    """Return True if the opening bytes look like a GLB file."""
    return len(head) >= 4 and head[:4] == b"glTF"


# =============================================================================
# GLB / glTF assembly helpers
# =============================================================================


def _pad4(data: bytes, *, pad_byte: bytes = b"\x00") -> bytes:
    """Return *data* padded to the next 4-byte boundary."""
    rem = len(data) % 4
    return data + pad_byte * (4 - rem) if rem else data


def _make_glb(gltf_dict: dict, bin_data: bytes) -> bytes:
    """Assemble a GLB binary from a JSON dict and binary buffer."""
    try:
        json_bytes = _pad4(
            json.dumps(gltf_dict, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            ),
            pad_byte=b" ",
        )
    except ValueError as exc:
        raise CodecError("glTF: mesh data contains NaN or Inf values") from exc
    chunks: list[bytes] = []
    chunks.append(struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes)
    if bin_data:
        padded_bin = _pad4(bin_data)
        chunks.append(struct.pack("<II", len(padded_bin), _CHUNK_BIN) + padded_bin)
    body = b"".join(chunks)
    total = _GLB_HEADER_SIZE + len(body)
    header = struct.pack("<III", _GLB_MAGIC, 2, total)
    return header + body


class _BinBuilder:
    """Accumulates binary buffer data and tracks bufferViews / accessors."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._offset: int = 0
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def add(
        self,
        data: np.ndarray,
        *,
        acc_type: str,
        component_type: int,
        normalized: bool = False,
        target: int | None = 34962,
    ) -> int:
        """Append *data* and register a bufferView + accessor; return accessor index."""
        raw = _pad4(data.tobytes())
        bv_idx = len(self.buffer_views)
        bv: dict = {
            "buffer": 0,
            "byteOffset": self._offset,
            "byteLength": len(raw),
        }
        if target is not None:
            bv["target"] = target
        self.buffer_views.append(bv)
        self._chunks.append(raw)
        self._offset += len(raw)

        acc_idx = len(self.accessors)
        entry: dict = {
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": component_type,
            "count": data.shape[0],
            "type": acc_type,
        }
        if normalized:
            entry["normalized"] = True
        # glTF spec requires min/max on POSITION (VEC3) accessors.
        if acc_type == "VEC3" and data.ndim == 2 and data.shape[0] > 0:
            entry["min"] = data.min(axis=0).tolist()
            entry["max"] = data.max(axis=0).tolist()
        self.accessors.append(entry)
        return acc_idx

    def add_indices(self, indices: np.ndarray) -> int:
        """Append an index buffer and return its accessor index."""
        raw = _pad4(indices.tobytes())
        bv_idx = len(self.buffer_views)
        self.buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": self._offset,
                "byteLength": len(raw),
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            }
        )
        self._chunks.append(raw)
        self._offset += len(raw)

        comp_type = 5123 if indices.dtype == np.uint16 else 5125  # USHORT / UINT
        acc_idx = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": bv_idx,
                "byteOffset": 0,
                "componentType": comp_type,
                "count": len(indices),
                "type": "SCALAR",
            }
        )
        return acc_idx

    def add_image(self, data: bytes, media_type: str | None) -> int:
        """Append raw image bytes and return the bufferView index."""
        raw = _pad4(data)
        bv_idx = len(self.buffer_views)
        bv: dict = {
            "buffer": 0,
            "byteOffset": self._offset,
            "byteLength": len(raw),
        }
        if media_type:
            bv["_media_type"] = media_type  # carried for JSON; cleaned before output
        self.buffer_views.append(bv)
        self._chunks.append(raw)
        self._offset += len(raw)
        return bv_idx

    @property
    def bin_data(self) -> bytes:
        """Return the accumulated binary buffer."""
        return b"".join(self._chunks)


# =============================================================================
# PolyData → glTF primitives
# =============================================================================

_GLTF_ATTR_MAP: dict[str, tuple[str, int]] = {
    # vertex_attr key → (glTF semantic, componentType)
    "normals": ("NORMAL", 5126),
    "texcoords": ("TEXCOORD_0", 5126),
    "texcoords_1": ("TEXCOORD_1", 5126),
    "colors": ("COLOR_0", 5126),
    "tangents": ("TANGENT", 5126),
    "joints": ("JOINTS_0", 5121),  # UBYTE
    "weights": ("WEIGHTS_0", 5126),
}

_GLTF_ACC_TYPE: dict[str, str] = {
    "normals": "VEC3",
    "texcoords": "VEC2",
    "texcoords_1": "VEC2",
    "tangents": "VEC4",
    "joints": "VEC4",
    "weights": "VEC4",
}

_ATTR_MAP: dict[str, str] = {
    "NORMAL": "normals",
    "TEXCOORD_0": "texcoords",
    "TEXCOORD_1": "texcoords_1",
    "TANGENT": "tangents",
    "JOINTS_0": "joints",
    "WEIGHTS_0": "weights",
}


def _polydata_to_primitives(
    poly: PolyData,
    bb: _BinBuilder,
    mat_remap: dict[int, int] | None = None,
) -> list[dict]:
    """Convert a PolyData into a list of glTF primitive dicts.

    Parameters
    ----------
    poly
        Source mesh.
    bb
        Binary buffer builder to append vertex/index data into.
    mat_remap
        Mapping from raw ``element_attrs["material"]`` values to the
        sequential indices used in the output ``materials`` array.  ``None``
        passes raw values through unchanged (valid when material indices are
        already dense and zero-based).

    Returns
    -------
    list[dict]
        glTF primitive dicts ready for ``mesh["primitives"]``.
    """
    # POSITION accessor (float32).
    pos_f32 = poly.vertices.astype(np.float32)
    pos_idx = bb.add(pos_f32, acc_type="VEC3", component_type=5126)

    # Vertex attribute accessors.
    va_accessors: dict[str, int] = {}
    for key, (semantic, comp_type) in _GLTF_ATTR_MAP.items():
        if key not in poly.vertex_attrs:
            continue
        arr = poly.vertex_attrs[key]

        if key == "joints":
            # Joints with index > 255 require UNSIGNED_SHORT.
            if arr.size > 0 and arr.max() > 255:
                data = arr.astype(np.uint16)
                comp_type = 5123  # UNSIGNED_SHORT
            else:
                data = arr.astype(np.uint8)
        elif key == "colors":
            # Integer color arrays must be divided by 255 before writing as float32.
            if np.issubdtype(arr.dtype, np.integer):
                data = (arr / 255.0).astype(np.float32)
            else:
                data = arr.astype(np.float32)
        else:
            data = arr.astype(np.float32)

        # Derive acc_type from array shape, not attribute name.
        if arr.ndim == 1:
            acc_type = "SCALAR"
        elif arr.shape[1] == 2:
            acc_type = "VEC2"
        elif arr.shape[1] == 3:
            acc_type = "VEC3"
        elif arr.shape[1] == 4:
            acc_type = "VEC4"
        else:
            raise CodecError(
                f"glTF: attribute '{key}' has unsupported shape {arr.shape}"
            )

        va_accessors[semantic] = bb.add(
            data, acc_type=acc_type, component_type=comp_type
        )

    # Group elements by (element_type_code, material_index).
    mat_col: np.ndarray | None = poly.element_attrs.get("material")
    primitives: list[dict] = []

    # Warn once for each unsupported category.
    has_volume = bool(np.any(np.isin(poly.element_types, list(_VOLUME_CODES))))
    if has_volume:
        warnings.warn(
            "glTF: volume elements (tetra, hexahedron, etc.) have no glTF "
            "primitive mode and were skipped.",
            stacklevel=5,  # user → write/write_scene → _polydata_to_primitives → here
        )
    has_skip = bool(np.any(np.isin(poly.element_types, list(_SKIP_WARN_CODES))))
    if has_skip:
        warnings.warn(
            "glTF: pixel and quadratic surface elements have no glTF "
            "primitive mode and were skipped.",
            stacklevel=5,  # user → write/write_scene → _polydata_to_primitives → here
        )

    _WRITABLE_MODES: dict[int, int] = {
        _TRI_CODE: 4,
        _QUAD_CODE: 4,  # fan-triangulate below
        _POLY_CODE: 4,  # fan-triangulate below
        _LINE_CODE: 1,
        _VERTEX_CODE: 0,
        # _TRI_STRIP_CODE excluded — strips must never be merged across
        # element boundaries (bridging triangles appear at seams); handled
        # individually in the per-element loop below, like poly_line.
    }

    _SKIP_ALL: frozenset[int] = _VOLUME_CODES | _SKIP_WARN_CODES

    # Group mergeable element types by (glTF mode, material index).
    # poly_line and triangle_strip are excluded — each must be its own primitive.
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for elem_i in range(len(poly.element_types)):
        code = int(poly.element_types[elem_i])
        if code in _SKIP_ALL or code == _POLY_LINE_CODE or code == _TRI_STRIP_CODE:
            continue
        mode = _WRITABLE_MODES.get(code)
        if mode is None:
            continue
        mat = int(mat_col[elem_i]) if mat_col is not None else -1
        groups.setdefault((mode, mat), []).append((code, elem_i))

    for (mode, mat), code_elem_pairs in groups.items():
        index_list: list[int] = []
        for code, ei in code_elem_pairs:
            seg = poly.connectivity[poly.offsets[ei] : poly.offsets[ei + 1]].tolist()
            if code in (_QUAD_CODE, _POLY_CODE):
                for j in range(1, len(seg) - 1):
                    index_list.extend([seg[0], seg[j], seg[j + 1]])
            else:
                index_list.extend(seg)

        indices_arr = np.array(index_list, dtype=np.uint32)
        if indices_arr.max() < 65536:
            indices_arr = indices_arr.astype(np.uint16)
        idx_accessor = bb.add_indices(indices_arr)

        prim: dict = {
            "attributes": {"POSITION": pos_idx, **va_accessors},
            "indices": idx_accessor,
            "mode": mode,
        }
        if mat >= 0:
            prim["material"] = mat_remap[mat] if mat_remap else mat
        primitives.append(prim)

    # poly_line → LINE_STRIP (mode 3): one primitive per element, each strip
    # must remain independent — merging would falsely connect their endpoints.
    # triangle_strip → TRIANGLE_STRIP (mode 5): same reason — merged strips
    # produce bridging triangles across seams.
    for elem_i in range(len(poly.element_types)):
        code = int(poly.element_types[elem_i])
        if code == _POLY_LINE_CODE:
            strip_mode = 3  # LINE_STRIP
        elif code == _TRI_STRIP_CODE:
            strip_mode = 5  # TRIANGLE_STRIP
        else:
            continue
        seg = poly.connectivity[poly.offsets[elem_i] : poly.offsets[elem_i + 1]]
        indices_arr = seg.astype(np.uint32)
        if indices_arr.size > 0 and indices_arr.max() < 65536:
            indices_arr = indices_arr.astype(np.uint16)
        idx_accessor = bb.add_indices(indices_arr)
        mat = int(mat_col[elem_i]) if mat_col is not None else -1
        prim = {
            "attributes": {"POSITION": pos_idx, **va_accessors},
            "indices": idx_accessor,
            "mode": strip_mode,
        }
        if mat >= 0:
            prim["material"] = mat_remap[mat] if mat_remap else mat
        primitives.append(prim)

    return primitives


# =============================================================================
# write
# =============================================================================


def write(
    poly: PolyData, path: Source, *, binary: bool | None = None, **opts: Any
) -> None:
    """Write a PolyData to a glTF or GLB file.

    Parameters
    ----------
    poly
        Mesh to serialize.  Volume elements are skipped with a warning.
        Quads and polygons are fan-triangulated into triangles.
    path
        Output file path.
    binary
        ``True`` writes a self-contained ``.glb`` binary container.
        ``False`` writes a ``.gltf`` JSON file plus a companion ``.bin``
        file; *path* must be a filesystem path, not a stream.
        Defaults to ``False`` when *path* ends in ``.gltf``, ``True``
        otherwise.

    Raises
    ------
    CodecError
        If no writable elements remain after filtering, or if
        ``binary=False`` and *path* is a stream rather than a filesystem
        path.
    """
    if binary is None:
        binary = format_suffix(path).lower() != ".gltf"

    mat_col = poly.element_attrs.get("material")
    mat_remap: dict[int, int] = {}
    gltf_materials: list[dict] = []
    if mat_col is not None:
        unique_mats = sorted({int(m) for m in mat_col if m >= 0})
        mat_remap = {m: i for i, m in enumerate(unique_mats)}
        gltf_materials = [
            {
                "name": f"material_{m}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
            for m in unique_mats
        ]

    bb = _BinBuilder()
    primitives = _polydata_to_primitives(poly, bb, mat_remap=mat_remap or None)

    if not primitives:
        raise CodecError(
            "glTF: no writable elements in PolyData after filtering out "
            "volume elements.  At least one triangle, line, or point is needed."
        )

    asset_meta: dict = {"version": "2.0", "generator": f"polyxios {__version__}"}
    ga = poly.global_attrs.get("asset", {})
    if "copyright" in ga:
        asset_meta["copyright"] = ga["copyright"]

    gltf_dict: dict = {
        "asset": asset_meta,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "accessors": bb.accessors,
        "bufferViews": bb.buffer_views,
        "buffers": [{"byteLength": len(bb.bin_data)}],
    }
    if gltf_materials:
        gltf_dict["materials"] = gltf_materials

    _write_output(gltf_dict, bb.bin_data, path, binary=binary)


# =============================================================================
# write_scene
# =============================================================================


def _matrix_to_trs(matrix: np.ndarray) -> dict:
    """Decompose a 4×4 transform into glTF translation/rotation/scale.

    Parameters
    ----------
    matrix
        4×4 float64 row-major transform matrix.

    Returns
    -------
    dict
        Dict with ``"translation"``, ``"rotation"`` (x, y, z, w), and
        ``"scale"`` keys.  Identity components are omitted.
    """
    t = matrix[:3, 3].tolist()
    col0 = matrix[:3, 0]
    col1 = matrix[:3, 1]
    col2 = matrix[:3, 2]
    sx = float(np.linalg.norm(col0))
    sy = float(np.linalg.norm(col1))
    sz = float(np.linalg.norm(col2))

    # Detect degenerate (zero) scale components.
    if sx <= 0 or sy <= 0 or sz <= 0:
        warnings.warn(
            "glTF: matrix has a zero-scale component "
            f"(sx={sx:.6g}, sy={sy:.6g}, sz={sz:.6g}); TRS decomposition may be inaccurate.",
            stacklevel=4,  # user → write_scene → _matrix_to_trs → here
        )

    # Detect negative determinant (reflection); TRS cannot represent it.
    det = float(np.linalg.det(matrix[:3, :3]))
    if det < 0:
        warnings.warn(
            "glTF: matrix has a negative determinant (reflection); "
            "glTF TRS cannot represent reflections — absolute-value scale will be used.",
            stacklevel=4,  # user → write_scene → _matrix_to_trs → here
        )
        sx, sy, sz = abs(sx), abs(sy), abs(sz)

    # Rotation sub-matrix (columns normalised).
    r = np.column_stack(
        [
            col0 / sx if sx > 0 else col0,
            col1 / sy if sy > 0 else col1,
            col2 / sz if sz > 0 else col2,
        ]
    )

    try:
        from scipy.spatial.transform import Rotation as _Rot

        q = _Rot.from_matrix(r).as_quat()  # [x, y, z, w]
    except (ImportError, ModuleNotFoundError):
        # Pure-numpy Shepherd method fallback.
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q = np.array(
                [
                    (r[2, 1] - r[1, 2]) * s,
                    (r[0, 2] - r[2, 0]) * s,
                    (r[1, 0] - r[0, 1]) * s,
                    0.25 / s,
                ]
            )
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q = np.array(
                [
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[2, 1] - r[1, 2]) / s,
                ]
            )
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q = np.array(
                [
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                    (r[0, 2] - r[2, 0]) / s,
                ]
            )
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q = np.array(
                [
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                    (r[1, 0] - r[0, 1]) / s,
                ]
            )

    # Normalise quaternion; fall back to identity if it is degenerate.
    q_norm = float(np.linalg.norm(q))
    if q_norm > 0:
        q = q / q_norm
    else:
        warnings.warn(
            "glTF: quaternion extracted from matrix has zero norm; "
            "falling back to identity rotation.",
            stacklevel=4,  # user → write_scene → _matrix_to_trs → here
        )
        q = np.array([0.0, 0.0, 0.0, 1.0])

    out: dict = {}
    _identity_t = [0.0, 0.0, 0.0]
    _identity_r = [0.0, 0.0, 0.0, 1.0]
    _identity_s = [1.0, 1.0, 1.0]
    if not np.allclose(t, _identity_t):
        out["translation"] = t
    if not np.allclose(q.tolist(), _identity_r, atol=1e-6):
        out["rotation"] = q.tolist()
    if not np.allclose([sx, sy, sz], _identity_s, atol=1e-6):
        out["scale"] = [sx, sy, sz]
    return out


def write_scene(
    scene: SceneData,
    path: Source,
    *,
    binary: bool | None = None,
    **opts: Any,
) -> None:
    """Write a SceneData to a glTF or GLB file preserving the scene graph.

    Parameters
    ----------
    scene
        Scene to serialize.  Each :class:`~polyxios._scene.SceneNode` becomes
        a glTF node; transforms, hierarchy, and materials are preserved.
    path
        Output file path.
    binary
        ``True`` writes a ``.glb`` container.  ``False`` writes
        ``.gltf`` + ``.bin``; *path* must be a filesystem path.
        Defaults to ``False`` when *path* ends in ``.gltf``, ``True``
        otherwise.

    Raises
    ------
    CodecError
        If ``binary=False`` and *path* is a stream, or if any mesh produces
        no writable primitives.

    Notes
    -----
    ``write_scene`` is lossy for skinned meshes: skin definitions
    (``global_attrs["skins"]``), ``JOINTS_0`` / ``WEIGHTS_0`` vertex
    attributes, and per-node ``skin`` / ``camera`` extras are not written.
    Use the round-tripped ``SceneData`` only for static geometry and
    material inspection.
    """
    if binary is None:
        binary = format_suffix(path).lower() != ".gltf"
    bb = _BinBuilder()

    # Meshes.
    gltf_meshes: list[dict] = []
    for mesh_idx, mesh_poly in enumerate(scene.meshes):
        prims = _polydata_to_primitives(mesh_poly, bb)
        if not prims:
            raise CodecError(
                f"glTF: mesh {mesh_idx} produced no writable primitives; "
                "remove it or filter out unsupported element types first."
            )
        gltf_meshes.append({"primitives": prims})

    # Materials.
    gltf_materials: list[dict] = [_scene_material_to_gltf(m) for m in scene.materials]

    # Images — embed bufferView data or use URI.
    gltf_images: list[dict] = []
    gltf_bvs_for_images: list[int] = []
    for img in scene.images:
        img_entry: dict = {}
        if img.name:
            img_entry["name"] = img.name
        if img.data is not None:
            bv_idx = bb.add_image(img.data, img.media_type)
            img_entry["bufferView"] = bv_idx
            if img.media_type:
                img_entry["mimeType"] = img.media_type
            gltf_bvs_for_images.append(bv_idx)
        elif img.uri:
            img_entry["uri"] = img.uri
        gltf_images.append(img_entry)

    # Textures.
    gltf_textures: list[dict] = []
    gltf_samplers: list[dict] = []
    for tex in scene.textures:
        tex_entry: dict = {}
        if tex.image >= 0:
            tex_entry["source"] = tex.image
        sampler: dict = {}
        if tex.mag_filter is not None:
            sampler["magFilter"] = tex.mag_filter
        if tex.min_filter is not None:
            sampler["minFilter"] = tex.min_filter
        if tex.wrap_s != 10497:
            sampler["wrapS"] = tex.wrap_s
        if tex.wrap_t != 10497:
            sampler["wrapT"] = tex.wrap_t
        if sampler:
            tex_entry["sampler"] = len(gltf_samplers)
            gltf_samplers.append(sampler)
        gltf_textures.append(tex_entry)

    # Nodes.
    # Nodes targeted by animation channels MUST use T/R/S, not matrix.
    identity = np.eye(4, dtype=np.float64)
    animated_node_indices: set[int] = set()
    for anim in scene.global_attrs.get("animations", []):
        for ch in anim.get("channels", []):
            node_idx = ch.get("target", {}).get("node")
            if node_idx is not None:
                animated_node_indices.add(node_idx)

    gltf_nodes: list[dict] = []
    for node_i, node in enumerate(scene.nodes):
        n_entry: dict = {}
        if node.name:
            n_entry["name"] = node.name
        if node.mesh is not None:
            n_entry["mesh"] = node.mesh
        if node.children:
            n_entry["children"] = list(node.children)
        if node_i in animated_node_indices:
            # glTF spec: animated nodes MUST NOT carry matrix; always use
            # explicit T/R/S even when the rest-pose transform is identity.
            trs = _matrix_to_trs(node.matrix)
            n_entry["translation"] = trs.get("translation", [0.0, 0.0, 0.0])
            n_entry["rotation"] = trs.get("rotation", [0.0, 0.0, 0.0, 1.0])
            n_entry["scale"] = trs.get("scale", [1.0, 1.0, 1.0])
        elif not np.allclose(node.matrix, identity):
            # glTF column-major = transpose of numpy row-major.
            n_entry["matrix"] = node.matrix.T.ravel().tolist()
        gltf_nodes.append(n_entry)

    # Scenes.
    if scene.scenes:
        gltf_scenes = [{"nodes": list(s)} for s in scene.scenes]
    else:
        all_children: set[int] = set()
        for node in scene.nodes:
            all_children.update(node.children)
        roots = [i for i in range(len(gltf_nodes)) if i not in all_children]
        gltf_scenes = [{"nodes": roots}]

    asset_meta: dict = {"version": "2.0", "generator": f"polyxios {__version__}"}
    ga = scene.global_attrs.get("asset", {})
    if "copyright" in ga:
        asset_meta["copyright"] = ga["copyright"]
    if scene.name:
        asset_meta["name"] = scene.name
    # Encode animations before assembling gltf_dict: _encode_animations appends
    # to bb.accessors and bb.buffer_views, so bvs and buffer byte-length must
    # be computed after all data is in the builder.
    gltf_animations = _encode_animations(scene.global_attrs.get("animations", []), bb)

    bvs = [
        {k: v for k, v in bv.items() if k != "_media_type"} for bv in bb.buffer_views
    ]

    gltf_dict: dict = {
        "asset": asset_meta,
        "scene": scene.active_scene,
        "scenes": gltf_scenes,
        "nodes": gltf_nodes,
        "meshes": gltf_meshes,
        "accessors": bb.accessors,
        "bufferViews": bvs,
        "buffers": [{"byteLength": len(bb.bin_data)}],
    }
    if gltf_materials:
        gltf_dict["materials"] = gltf_materials
    if gltf_images:
        gltf_dict["images"] = gltf_images
    if gltf_textures:
        gltf_dict["textures"] = gltf_textures
    if gltf_samplers:
        gltf_dict["samplers"] = gltf_samplers
    if gltf_animations:
        gltf_dict["animations"] = gltf_animations

    _write_output(gltf_dict, bb.bin_data, path, binary=binary)


# glTF path → (accessor type string, number of components)
_PATH_ACCESSOR: dict[str, tuple[str, int]] = {
    "translation": ("VEC3", 3),
    "rotation": ("VEC4", 4),
    "scale": ("VEC3", 3),
}


def _encode_animations(animations: list[dict], bb: _BinBuilder) -> list[dict]:
    """Encode decoded animation data back into glTF animation dicts.

    Appends accessor and bufferView entries to *bb* for each sampler's
    time and value arrays and returns a list of glTF animation objects
    ready to embed in the JSON.  All interpolation modes (LINEAR, STEP,
    CUBICSPLINE) are preserved; for CUBICSPLINE the values array already
    holds 3 × n_keyframes rows (in-tangent, value, out-tangent) as decoded
    by :func:`_decode_animations`.

    Parameters
    ----------
    animations
        Decoded animation list from ``SceneData.global_attrs["animations"]``.
        Each sampler must have ``"times"`` and ``"values"`` as ndarrays.
    bb
        Binary buffer builder to append accessor data into.

    Returns
    -------
    list[dict]
        glTF animation dicts ready for ``gltf["animations"]``.
    """
    result: list[dict] = []
    for anim in animations:
        channels = anim.get("channels", [])
        samplers = anim.get("samplers", [])
        if not channels or not samplers:
            continue

        # Build both lists in a single loop so a skipped channel never
        # shifts the sampler index relative to the channel list.
        gltf_samplers: list[dict] = []
        gltf_channels: list[dict] = []
        for ch in channels:
            sampler_idx = ch.get("sampler", 0)
            if sampler_idx >= len(samplers):
                continue
            s = samplers[sampler_idx]
            times: np.ndarray = np.asarray(s["times"], dtype=np.float32)
            values: np.ndarray = np.asarray(s["values"], dtype=np.float32)
            path = ch.get("target", {}).get("path", "")

            # Derive acc_type from values shape; "weights" is SCALAR.
            if (
                path == "weights"
                or values.ndim == 1
                or (values.ndim == 2 and values.shape[1] == 1)
            ):
                acc_type = "SCALAR"
            elif values.ndim == 2 and values.shape[1] == 2:
                acc_type = "VEC2"
            elif values.ndim == 2 and values.shape[1] == 4:
                acc_type = "VEC4"
            else:
                acc_type, _ = _PATH_ACCESSOR.get(path, ("VEC3", 3))

            t_idx = bb.add(times, acc_type="SCALAR", component_type=5126, target=None)
            # glTF spec requires min/max on animation sampler input (time) accessors.
            if len(times) > 0:
                bb.accessors[t_idx]["min"] = [float(times.min())]
                bb.accessors[t_idx]["max"] = [float(times.max())]
            v_idx = bb.add(values, acc_type=acc_type, component_type=5126, target=None)
            interp = s.get("interpolation", "LINEAR")

            gltf_samplers.append(
                {
                    "input": t_idx,
                    "output": v_idx,
                    "interpolation": interp,
                }
            )
            gltf_channels.append(
                {
                    "sampler": len(gltf_samplers) - 1,
                    "target": ch.get("target", {}),
                }
            )

        entry: dict = {"channels": gltf_channels, "samplers": gltf_samplers}
        if anim.get("name"):
            entry["name"] = anim["name"]
        result.append(entry)

    return result


def _scene_material_to_gltf(mat: SceneMaterial) -> dict:
    """Convert a SceneMaterial to a glTF material dict."""
    pbr: dict = {
        "baseColorFactor": list(mat.base_color),
        "metallicFactor": mat.metallic,
        "roughnessFactor": mat.roughness,
    }
    if mat.base_color_texture is not None:
        pbr["baseColorTexture"] = {"index": mat.base_color_texture}
    if mat.metallic_roughness_texture is not None:
        pbr["metallicRoughnessTexture"] = {"index": mat.metallic_roughness_texture}

    entry: dict = {"pbrMetallicRoughness": pbr}
    if mat.name:
        entry["name"] = mat.name
    if mat.emissive != (0.0, 0.0, 0.0):
        entry["emissiveFactor"] = list(mat.emissive)
    if mat.emissive_texture is not None:
        entry["emissiveTexture"] = {"index": mat.emissive_texture}
    if mat.normal_texture is not None:
        entry["normalTexture"] = {"index": mat.normal_texture}
    if mat.occlusion_texture is not None:
        entry["occlusionTexture"] = {"index": mat.occlusion_texture}
    if mat.alpha_mode != "OPAQUE":
        entry["alphaMode"] = mat.alpha_mode
    if mat.alpha_mode == "MASK":
        entry["alphaCutoff"] = mat.alpha_cutoff
    if mat.double_sided:
        entry["doubleSided"] = True
    return entry


def _write_output(
    gltf_dict: dict,
    bin_data: bytes,
    path: Source,
    *,
    binary: bool,
) -> None:
    """Write the assembled glTF dict and binary buffer to *path*."""
    if binary:
        write_bytes(path, _make_glb(gltf_dict, bin_data))
        return

    if is_buffer(path):
        raise CodecError(
            "Writing .gltf with a separate .bin requires a filesystem path, "
            "not a stream.  Use binary=True for a self-contained GLB."
        )

    p = Path(str(path))
    bin_name = p.stem + ".bin"
    if bin_data:
        gltf_dict = dict(gltf_dict)
        gltf_dict["buffers"] = [{"uri": bin_name, "byteLength": len(bin_data)}]
        write_bytes(p.with_name(bin_name), bin_data)
    else:
        gltf_dict = {k: v for k, v in gltf_dict.items() if k != "buffers"}

    try:
        json_str = json.dumps(gltf_dict, indent=2, allow_nan=False)
    except ValueError as exc:
        raise CodecError("glTF: mesh data contains NaN or Inf values") from exc
    write_text(path, json_str, encoding="utf-8")
