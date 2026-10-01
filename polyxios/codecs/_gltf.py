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
from polyxios._trs import matrix_of_trs, trs_of_matrix
from polyxios._types import PolyData
from polyxios._warn import warn_caller as _warn_caller
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

    return matrix_of_trs(
        translation=node.get("translation", [0.0, 0.0, 0.0]),
        rotation=node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
        scale=node.get("scale", [1.0, 1.0, 1.0]),
    )


# =============================================================================
# read_scene
# =============================================================================


def _is_float_mat4(accessor: Any) -> bool:
    """Return whether an accessor entry holds float MAT4 elements.

    Parameters
    ----------
    accessor
        One entry of ``gltf["accessors"]``.

    Returns
    -------
    bool
        True for an object of ``type`` MAT4 and ``componentType`` FLOAT, the
        only kind glTF allows for inverse bind matrices.
    """
    return (
        isinstance(accessor, dict)
        and accessor.get("type") == "MAT4"
        and accessor.get("componentType") == 5126
    )


def _is_index(value: Any, n: int) -> bool:
    """Return whether a JSON value is an index into a list of ``n`` entries.

    Parameters
    ----------
    value
        The value a glTF object gives where an index is expected.
    n
        The length of the list it indexes.

    Returns
    -------
    bool
        True for an integer from 0 to ``n - 1``; False for a boolean, which
        Python would take as 0 or 1, and for a negative integer, which would
        count from the end.
    """
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < n


def _decode_skins(gltf: dict, skins: Any, buffers: list[bytes]) -> Any:
    """Return the skins with their inverse bind matrices decoded.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    skins
        The raw ``gltf["skins"]`` list.
    buffers
        Raw buffer bytes.

    Returns
    -------
    Any
        ``skins`` itself, with a warning, when it is not a list. Otherwise
        one dict per skin: its raw keys, plus ``"inverse_bind_matrices"``,
        ``(len(joints), 4, 4)`` row-major float64, when it names an
        ``inverseBindMatrices`` accessor of at least one float MAT4 per
        joint, any beyond the joints left out. A skin that is not an object
        is kept as it is, and one whose ``joints`` is not a list or is empty,
        whose ``inverseBindMatrices`` names no accessor, an accessor
        that is not float MAT4, one that cannot be read, one with no
        ``bufferView`` (all zeros, which no bind pose inverts to) or one
        shorter than the joints is kept without the key; each with a warning.
    """
    if not isinstance(skins, list):
        _warn_caller("glTF: skins is not a list; kept as it is.")
        return skins
    out = []
    accessors = gltf.get("accessors", ())
    n_accessors = len(accessors) if isinstance(accessors, list) else 0
    for k, skin in enumerate(skins):
        if not isinstance(skin, dict):
            _warn_caller(f"glTF: skin {k} is not an object; kept as it is.")
            out.append(skin)
            continue
        entry = dict(skin)
        acc = skin.get("inverseBindMatrices")
        joints = skin.get("joints", [])
        if acc is None:
            pass
        elif not isinstance(joints, list):
            _warn_caller(
                f"glTF: skin {k} joints is not a list; its inverse bind "
                "matrices are not decoded."
            )
        elif not joints:
            _warn_caller(
                f"glTF: skin {k} has no joints; its inverse bind matrices are "
                "not decoded."
            )
        elif not _is_index(acc, n_accessors):
            _warn_caller(
                f"glTF: skin {k} inverseBindMatrices {acc!r} names no accessor; "
                "its inverse bind matrices are not decoded."
            )
        elif not _is_float_mat4(accessors[acc]):
            _warn_caller(
                f"glTF: skin {k} inverseBindMatrices accessor {acc} is not a "
                "float MAT4 one; its inverse bind matrices are not decoded."
            )
        else:
            n_joints = len(joints)
            try:
                ibm = _read_accessor(gltf, acc, buffers)
            except (CodecError, KeyError, IndexError, TypeError, ValueError) as exc:
                _warn_caller(
                    f"glTF: skin {k} inverseBindMatrices accessor {acc} cannot "
                    f"be read ({exc}); its inverse bind matrices are not decoded."
                )
            else:
                if "bufferView" not in accessors[acc]:
                    _warn_caller(
                        f"glTF: skin {k} inverseBindMatrices accessor {acc} has no "
                        "bufferView, so holds only zeros; its inverse bind "
                        "matrices are not decoded."
                    )
                elif ibm.ndim != 2 or ibm.shape[1] != 16 or len(ibm) < n_joints:
                    _warn_caller(
                        f"glTF: skin {k} has {n_joints} joints but its "
                        f"inverseBindMatrices accessor holds shape {ibm.shape}, "
                        "not one MAT4 per joint; its inverse bind matrices are "
                        "not decoded."
                    )
                else:
                    # Column-major in the file; transposed to numpy's row-major.
                    entry["inverse_bind_matrices"] = np.ascontiguousarray(
                        ibm[:n_joints].reshape(-1, 4, 4).transpose(0, 2, 1),
                        dtype=np.float64,
                    )
        out.append(entry)
    return out


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

    Raises
    ------
    CodecError
        If ``animations`` is not a list, an animation is not an object
        holding a list of channels and a list of samplers, a channel or a
        sampler is not an object, or a sampler's ``input`` or ``output`` is
        not the index of a readable accessor, SCALAR for the ``input``.
    """
    if not isinstance(animations, list):
        raise CodecError("glTF: animations is not a list.")
    accessors = gltf.get("accessors")
    n_accessors = len(accessors) if isinstance(accessors, list) else 0
    result = []
    for k, anim in enumerate(animations):
        if not isinstance(anim, dict) or not all(
            isinstance(anim.get(key, []), list) for key in ("channels", "samplers")
        ):
            raise CodecError(
                f"glTF: animation {k} is not an object holding a list of "
                "channels and a list of samplers."
            )
        for j, ch in enumerate(anim.get("channels", [])):
            if not isinstance(ch, dict):
                raise CodecError(
                    f"glTF: channel {j} of animation {k} is not an object."
                )
        decoded_samplers: list[dict] = []
        for j, s in enumerate(anim.get("samplers", [])):
            if not isinstance(s, dict):
                raise CodecError(
                    f"glTF: sampler {j} of animation {k} is not an object."
                )
            for key in ("input", "output"):
                if not _is_index(s.get(key), n_accessors):
                    raise CodecError(
                        f"glTF: sampler {j} of animation {k} names no accessor "
                        f"as its {key} ({s.get(key)!r})."
                    )
            try:
                times = _read_accessor(gltf, s["input"], buffers).astype(np.float64)
                values = _read_accessor(gltf, s["output"], buffers).astype(np.float64)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise CodecError(
                    f"glTF: sampler {j} of animation {k} cannot be read ({exc!r})."
                ) from exc
            if times.ndim != 1:
                raise CodecError(
                    f"glTF: sampler {j} of animation {k} takes its times from "
                    f"accessor {s['input']}, which is not a SCALAR one."
                )
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
        global_attrs["skins"] = _decode_skins(gltf, gltf["skins"], buffers)

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
    _warn_caller(
        f"'{source_name(path)}' is a scene format (glTF): read() flattens "
        "the scene graph, materials, textures and hierarchy into a single "
        "PolyData. Use polyxios.read_scene() to preserve the full scene."
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


def _json_default(value: Any) -> Any:
    """Return a numpy scalar or array as the Python value JSON can hold.

    Parameters
    ----------
    value
        A value :func:`json.dumps` cannot serialize by itself.

    Returns
    -------
    Any
        The ``int``, ``float``, ``bool`` or nested ``list`` it holds.

    Raises
    ------
    TypeError
        If it is neither a numpy scalar nor an array.
    """
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _pad4(data: bytes, *, pad_byte: bytes = b"\x00") -> bytes:
    """Return *data* padded to the next 4-byte boundary."""
    rem = len(data) % 4
    return data + pad_byte * (4 - rem) if rem else data


def _make_glb(gltf_dict: dict, bin_data: bytes) -> bytes:
    """Assemble a GLB binary from a JSON dict and binary buffer."""
    try:
        json_bytes = _pad4(
            json.dumps(
                gltf_dict,
                separators=(",", ":"),
                allow_nan=False,
                default=_json_default,
            ).encode("utf-8"),
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
        _warn_caller(
            "glTF: volume elements (tetra, hexahedron, etc.) have no glTF "
            "primitive mode and were skipped."
        )
    has_skip = bool(np.any(np.isin(poly.element_types, list(_SKIP_WARN_CODES))))
    if has_skip:
        _warn_caller(
            "glTF: pixel and quadratic surface elements have no glTF "
            "primitive mode and were skipped."
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
        If ``binary=False`` and *path* is a stream, if any mesh produces
        no writable primitives, or if the matrix of an animated node holds
        NaN or Inf.

    Notes
    -----
    ``write_scene`` is lossy for skinned meshes: skin definitions
    (``global_attrs["skins"]``), ``JOINTS_0`` / ``WEIGHTS_0`` vertex
    attributes, and per-node ``skin`` / ``camera`` extras are not written.
    Use the round-tripped ``SceneData`` of a skinned scene only for static
    geometry and material inspection.

    An animation channel that is not a glTF one is skipped with a warning
    rather than written with the wrong meaning: a target with fields glTF
    lacks (another format's ``sid`` or ``member``), a path other than
    ``translation``, ``rotation``, ``scale``, ``weights`` or ``pointer``, a
    node that is not one of the scene's, an interpolation other than
    ``LINEAR``, ``STEP`` or ``CUBICSPLINE``, a sampler that is not an
    object, holds no key, NaN, Inf or a value float32 cannot hold, times
    that are not strictly increasing, a rotation of zero length, or whose
    values are not three per key for a translation or scale, four for a
    rotation (three times as many under ``CUBICSPLINE``), or a node path an
    earlier channel of the same animation already animates. A channel
    naming no sampler of its animation is dropped, and does not count as
    animating its node; an animation that is not an object is skipped with
    a warning.

    The values of a ``pointer`` channel are one row per key, written as the
    accessor type as wide as a row: a scalar, 2, 3 or 4 components, or the
    9 or 16 of a matrix. Any other shape is skipped with a warning.

    A ``matrix`` channel of ``LINEAR`` or ``STEP`` keys that animates a
    node's whole transform - the node spelled with that one matrix alone,
    or with no spelling at all - is written as translation, rotation and
    scale channels instead, each key split into the three (a shear or
    projection is dropped, with a warning); between keys glTF blends those
    rather than the matrix. It is skipped with a warning when another
    channel of its animation already animates one of the node's three
    paths.
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
    animations = _split_matrix_channels(
        _animation_objects(scene.global_attrs.get("animations")), scene.nodes
    )
    animated_node_indices: set[int] = set()
    n_nodes = len(scene.nodes)
    for anim in animations:
        samplers = anim.get("samplers", [])
        for ch in anim.get("channels", []):
            if _sampler_index(ch, samplers) is None:
                continue
            if _foreign_channel(ch, samplers, n_nodes=n_nodes):
                continue
            node_idx = ch["target"].get("node")
            if node_idx is not None:
                animated_node_indices.add(int(node_idx))

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
            if not np.isfinite(node.matrix).all():
                raise CodecError(
                    f"glTF: the matrix of animated node {node_i} holds NaN or "
                    "Inf values."
                )
            t, q, sc, exact = trs_of_matrix(node.matrix)
            if not exact:
                _warn_caller(
                    f"glTF: node {node_i} is animated, so it is written as "
                    "translation, rotation and scale, and its matrix holds a "
                    "shear or projection, which those cannot; it is dropped."
                )
            n_entry["translation"] = t.tolist()
            n_entry["rotation"] = q.tolist()
            n_entry["scale"] = sc.tolist()
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
    gltf_animations = _encode_animations(animations, bb, n_nodes=n_nodes)

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


_GLTF_PATHS = frozenset({"translation", "rotation", "scale", "weights", "pointer"})
_GLTF_INTERPOLATIONS = frozenset({"LINEAR", "STEP", "CUBICSPLINE"})
_GLTF_TARGET_KEYS = frozenset({"node", "path", "extensions", "extras"})

# glTF path → (accessor type string, number of components)
_PATH_ACCESSOR: dict[str, tuple[str, int]] = {
    "translation": ("VEC3", 3),
    "rotation": ("VEC4", 4),
    "scale": ("VEC3", 3),
}

# Width of one key of a pointer channel → accessor type; four is read as a
# VEC4, the commoner of the two types that wide.
_WIDTH_ACCESSOR: dict[int, str] = {
    1: "SCALAR",
    2: "VEC2",
    3: "VEC3",
    4: "VEC4",
    9: "MAT3",
    16: "MAT4",
}

_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _animation_objects(animations: Any) -> list[dict]:
    """Return the animations that are objects holding channels and samplers.

    Parameters
    ----------
    animations
        ``SceneData.global_attrs["animations"]``, or None when it has none.

    Returns
    -------
    list[dict]
        The entries that are dicts whose ``channels`` and ``samplers`` are
        lists or tuples, in order; any other is left out with a warning, as
        is the whole value when it is not a list or tuple.
    """
    if animations is None:
        return []
    if not isinstance(animations, (list, tuple)):
        _warn_caller("glTF: animations is not a list; none is written.")
        return []
    out = []
    for k, anim in enumerate(animations):
        if isinstance(anim, dict) and all(
            isinstance(anim.get(key, []), (list, tuple))
            for key in ("channels", "samplers")
        ):
            out.append(anim)
        else:
            _warn_caller(
                f"glTF: animation {k} is not an object holding a list of "
                "channels and a list of samplers; it is not written."
            )
    return out


def _bad_sampler(sampler: Any, path: str) -> str | None:
    """Return why a sampler cannot be written for a path, None when it can.

    Parameters
    ----------
    sampler
        One entry of an animation's ``samplers``.
    path
        The glTF path of the channel using it.

    Returns
    -------
    str or None
        The reason, None when the sampler holds at least one key of
        strictly increasing times and of values float32 can hold, in an
        interpolation glTF has, the values as many per key as the path
        takes: three or four for a translation, rotation or scale, any whole
        number for morph weights, and for a pointer one row per key of a
        width glTF has an accessor type for. A rotation key of zero length
        is refused too.
    """
    if not isinstance(sampler, dict):
        return f"its sampler {sampler!r} is not an object"
    interp = sampler.get("interpolation", "LINEAR")
    if not isinstance(interp, str) or interp not in _GLTF_INTERPOLATIONS:
        return f"its sampler interpolates {interp!r}, which glTF lacks"
    if "times" not in sampler or "values" not in sampler:
        return "its sampler has no times or no values"
    try:
        times = np.asarray(sampler["times"], dtype=np.float64)
        values = np.asarray(sampler["values"], dtype=np.float64)
    except (TypeError, ValueError):
        return "its sampler times or values are not numbers"
    if not times.size:
        return "its sampler holds no key"
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        return "its sampler holds NaN or Inf"
    if max(np.abs(times).max(), np.abs(values).max(initial=0.0)) > _FLOAT32_MAX:
        return "its sampler holds a value float32, the width glTF stores, cannot"
    if not (np.diff(times.ravel()) > 0).all():
        return "its sampler times are not strictly increasing"
    n_rows = times.size * (3 if interp == "CUBICSPLINE" else 1)
    if path in _PATH_ACCESSOR:
        width = _PATH_ACCESSOR[path][1]
        if values.size != n_rows * width:
            return (
                f"its sampler holds {values.size} values, not {width} for each "
                f"of its {times.size} {interp} keys"
            )
        if path == "rotation":
            # Under CUBICSPLINE only the middle row of three is a rotation.
            keys = values.reshape(-1, 4)
            if interp == "CUBICSPLINE":
                keys = keys[1::3]
            if not np.any(keys, axis=1).all():
                return "its sampler holds a rotation of zero length"
    elif path == "weights":
        if not values.size or values.size % n_rows:
            return (
                f"its sampler holds {values.size} values, not a whole number for "
                f"each of its {times.size} {interp} keys"
            )
    elif (
        values.ndim not in (1, 2)
        or len(values) != n_rows
        or (values.ndim == 2 and values.shape[1] not in _WIDTH_ACCESSOR)
    ):
        return (
            f"its sampler values, of shape {values.shape}, are not one scalar, "
            f"vector or matrix glTF has a type for per row of its {times.size} "
            f"{interp} keys"
        )
    return None


def _sampler_index(ch: Any, samplers: Any) -> int | None:
    """Return the index of the sampler a channel names, None when it names none.

    Parameters
    ----------
    ch
        One entry of an animation's ``channels``.
    samplers
        That animation's ``samplers``.

    Returns
    -------
    int or None
        The index, None when the channel is not an object or its ``sampler``
        is not an integer within ``samplers``.
    """
    s = ch.get("sampler", 0) if isinstance(ch, dict) else None
    if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
        return None
    if not isinstance(samplers, (list, tuple)) or not 0 <= s < len(samplers):
        return None
    return int(s)


def _foreign_channel(ch: Any, samplers: Any, *, n_nodes: int) -> str | None:
    """Return why a channel is not a glTF one, None when it is.

    Another format's channel can share glTF's shape but not its meaning: a
    COLLADA ``rotation`` animates one ``<rotate>`` element's angle in
    degrees (its target carries ``sid`` and ``member``), not the node's
    quaternion, and its sampler may interpolate ``BEZIER``. Written as is,
    it would be a valid-looking glTF file that animates the wrong thing.
    A sampler :func:`_bad_sampler` refuses makes its channel foreign too.

    Parameters
    ----------
    ch
        One entry of an animation's ``channels``.
    samplers
        That animation's ``samplers``.
    n_nodes
        The number of nodes of the scene.

    Returns
    -------
    str or None
        The reason the channel cannot be written, None when it can.
    """
    target = ch.get("target") if isinstance(ch, dict) else None
    if not isinstance(target, dict):
        return f"its target {target!r} is not an object"
    extra = sorted(set(target) - _GLTF_TARGET_KEYS, key=str)
    if extra:
        return f"its target carries {extra}, which glTF has no field for"
    path = target.get("path", "")
    if not isinstance(path, str) or path not in _GLTF_PATHS:
        return f"its path {path!r} is not a glTF one"
    node = target.get("node")
    if node is not None and (
        isinstance(node, bool)
        or not isinstance(node, (int, np.integer))
        or not 0 <= node < n_nodes
    ):
        return f"its node {node!r} is not one of the scene's {n_nodes}"
    sampler_idx = _sampler_index(ch, samplers)
    if sampler_idx is None:
        return None
    return _bad_sampler(samplers[sampler_idx], path)


_MATRIX_TARGET_KEYS = frozenset({"node", "path", "sid", "member"})
_MATRIX_INTERPOLATIONS = frozenset({"LINEAR", "STEP"})
_TRS_PATHS = ("translation", "rotation", "scale")


def _whole_matrix_channel(
    ch: Any, samplers: Any, nodes: tuple[SceneNode, ...]
) -> tuple[int, np.ndarray, np.ndarray, str] | None:
    """Return the keys of a channel animating a node's whole matrix.

    A COLLADA ``matrix`` channel animates one ``<matrix>`` element; it is
    the node's whole local transform only when the node is spelled with
    that element alone, or carries no spelling at all.

    Parameters
    ----------
    ch
        One entry of an animation's ``channels``.
    samplers
        That animation's ``samplers``.
    nodes
        The scene's nodes.

    Returns
    -------
    tuple or None
        The node index, the key times, the ``(n, 4, 4)`` row-major key
        matrices and the interpolation; None when the channel is not one
        whose keys can be split into translation, rotation and scale.
    """
    target = ch.get("target") if isinstance(ch, dict) else None
    if not isinstance(target, dict) or target.get("path") != "matrix":
        return None
    if set(target) - _MATRIX_TARGET_KEYS or target.get("member") not in (None, ""):
        return None
    node = target.get("node")
    if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
        return None
    if not 0 <= node < len(nodes):
        return None
    spelled = nodes[node].extras.get("transforms")
    if spelled is not None and not (
        isinstance(spelled, list)
        and len(spelled) == 1
        and isinstance(spelled[0], dict)
        and spelled[0].get("kind") == "matrix"
        and spelled[0].get("sid") == target.get("sid")
    ):
        return None
    s = _sampler_index(ch, samplers)
    if s is None:
        return None
    sampler = samplers[s]
    if not isinstance(sampler, dict):
        return None
    interp = sampler.get("interpolation", "LINEAR")
    if not isinstance(interp, str) or interp not in _MATRIX_INTERPOLATIONS:
        return None
    try:
        times = np.asarray(sampler.get("times"), dtype=np.float64).ravel()
        values = np.asarray(sampler.get("values"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not len(times) or values.size != 16 * len(times):
        return None
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        return None
    if max(np.abs(times).max(), np.abs(values).max()) > _FLOAT32_MAX:
        return None
    return int(node), times, values.reshape(len(times), 4, 4), interp


def _split_matrix_channels(animations: Any, nodes: tuple[SceneNode, ...]) -> Any:
    """Return the animations with whole-matrix channels split into TRS ones.

    Each channel :func:`_whole_matrix_channel` accepts is replaced, in its
    place, by a translation, a rotation and a scale channel over the same
    times, their samplers appended to the animation's. Every other channel
    and sampler is kept as is, so its index does not move. Successive
    quaternions keep the same hemisphere, so glTF's slerp takes the short
    way between keys as the matrix did.

    Parameters
    ----------
    animations
        ``SceneData.global_attrs["animations"]``.
    nodes
        The scene's nodes.

    Returns
    -------
    Any
        A new list of animations, the input left untouched; the input
        itself when it is not a list or tuple.
    """
    if not isinstance(animations, (list, tuple)):
        return animations
    out: list = []
    for anim in animations:
        channels = anim.get("channels", []) if isinstance(anim, dict) else None
        samplers = anim.get("samplers", []) if isinstance(anim, dict) else None
        if not isinstance(channels, (list, tuple)):
            out.append(anim)
            continue
        wholes = [_whole_matrix_channel(ch, samplers, nodes) for ch in channels]
        if all(w is None for w in wholes):
            out.append(anim)
            continue
        taken = {
            (ch["target"].get("node"), ch["target"].get("path"))
            for ch, w in zip(channels, wholes, strict=True)
            if w is None
            and isinstance(ch, dict)
            and isinstance(ch.get("target"), dict)
            and _sampler_index(ch, samplers) is not None
            and _foreign_channel(ch, samplers, n_nodes=len(nodes)) is None
        }
        new_channels: list = []
        new_samplers = list(samplers)
        for ch, whole in zip(channels, wholes, strict=True):
            if whole is None:
                new_channels.append(ch)
                continue
            node, times, keys, interp = whole
            paths = {(node, p) for p in _TRS_PATHS}
            if paths & taken:
                _warn_caller(
                    "glTF: an animation channel is not written: its matrix keys "
                    f"would animate the translation, rotation and scale of node "
                    f"{node}, which another channel of the animation already "
                    "animates."
                )
                continue
            taken |= paths
            split = [trs_of_matrix(k) for k in keys]
            if not all(exact for *_, exact in split):
                _warn_caller(
                    f"glTF: a matrix animation key of node {node} holds a shear "
                    "or projection, which translation, rotation and scale "
                    "cannot; it is dropped from the key."
                )
            flips = np.diff([np.prod(sc) < 0 for _, _, sc, _ in split])
            if flips.any():
                _warn_caller(
                    f"glTF: the matrix animation keys of node {node} change "
                    "handedness between keys; the reflection is a negative x "
                    "scale, so the keys between two such keys swing through it."
                )
            quats = np.array([q for _, q, _, _ in split])
            for k in range(1, len(quats)):
                if np.dot(quats[k], quats[k - 1]) < 0:
                    quats[k] = -quats[k]
            tracks = (
                np.array([t for t, _, _, _ in split]),
                quats,
                np.array([sc for _, _, sc, _ in split]),
            )
            for path, values in zip(_TRS_PATHS, tracks, strict=True):
                new_channels.append(
                    {
                        "sampler": len(new_samplers),
                        "target": {"node": node, "path": path},
                    }
                )
                new_samplers.append(
                    {"times": times, "values": values, "interpolation": interp}
                )
        out.append({**anim, "channels": new_channels, "samplers": new_samplers})
    return out


def _increasing_float32(times: np.ndarray) -> np.ndarray:
    """Return key times that stay strictly increasing once stored as float32.

    Two float64 times closer than a float32 step, a key held just before a
    step say, round onto one float32 time, the only width a glTF time
    accessor takes. Such a key moves down to the float32 just below the key
    after it, keeping the step a step; when that would push the first time,
    not negative to begin with, below zero, which glTF forbids, the
    colliding keys move up instead. Only the first time is kept from
    turning negative: after a negative first time, which glTF forbids
    anyway, a later zero may move below zero. Times that were not strictly
    increasing to begin with are left as they are.

    Parameters
    ----------
    times
        Key times, as float64.

    Returns
    -------
    np.ndarray
        The times as float32 values.
    """
    out = times.astype(np.float32)
    if out.ndim != 1 or (np.diff(out) > 0).all():
        return out
    if not (times[:-1] < times[1:]).all():
        return out
    down = out.copy()
    for k in range(len(down) - 2, -1, -1):
        if down[k] >= down[k + 1]:
            down[k] = np.nextafter(down[k + 1], np.float32(-np.inf))
    if times[0] < 0 or down[0] >= 0:
        return down
    for k in range(1, len(out)):
        if out[k] <= out[k - 1]:
            out[k] = np.nextafter(out[k - 1], np.float32(np.inf))
    return out


def _encode_animations(
    animations: list[dict], bb: _BinBuilder, *, n_nodes: int
) -> list[dict]:
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
    n_nodes
        The number of nodes of the scene, a channel targeting any other
        being skipped.

    Notes
    -----
    A sampler several channels name is written once, and so is a times
    array several samplers share. A channel :func:`_foreign_channel`
    refuses, or animating a node path an earlier channel of its animation
    already does, is skipped with a warning.

    Returns
    -------
    list[dict]
        glTF animation dicts ready for ``gltf["animations"]``.
    """
    result: list[dict] = []
    time_accessors: dict[int, int] = {}
    for anim in animations:
        channels = anim.get("channels", [])
        samplers = anim.get("samplers", [])
        if not channels or not samplers:
            continue

        gltf_samplers: list[dict] = []
        gltf_channels: list[dict] = []
        written: dict[tuple[int, str], int] = {}
        animated: set[tuple[int, str]] = set()
        for ch in channels:
            sampler_idx = _sampler_index(ch, samplers)
            if sampler_idx is None:
                continue
            reason = _foreign_channel(ch, samplers, n_nodes=n_nodes)
            if reason is not None:
                _warn_caller(f"glTF: an animation channel is not written: {reason}.")
                continue
            target = dict(ch["target"])
            path = target["path"]
            if target.get("node") is None:
                target.pop("node", None)
            else:
                target["node"] = int(target["node"])
                if path != "pointer":
                    if (target["node"], path) in animated:
                        _warn_caller(
                            "glTF: an animation channel is not written: an "
                            f"earlier channel of the animation already animates "
                            f"the {path} of node {target['node']}."
                        )
                        continue
                    animated.add((target["node"], path))
            s = samplers[sampler_idx]
            values: np.ndarray = np.asarray(s["values"], dtype=np.float32)

            if path in _PATH_ACCESSOR:
                acc_type, width = _PATH_ACCESSOR[path]
                values = values.reshape(-1, width)
            elif path == "weights":
                acc_type = "SCALAR"
                values = values.ravel()
            else:
                acc_type = _WIDTH_ACCESSOR[1 if values.ndim == 1 else values.shape[1]]
                if acc_type == "SCALAR":
                    values = values.ravel()

            if (sampler_idx, acc_type) not in written:
                # Keyed by the array itself: the three samplers a matrix
                # channel splits into share one, and so one accessor.
                t_idx = time_accessors.get(id(s["times"]))
                if t_idx is None:
                    times = _increasing_float32(
                        np.asarray(s["times"], dtype=np.float64).ravel()
                    )
                    t_idx = bb.add(
                        times, acc_type="SCALAR", component_type=5126, target=None
                    )
                    # glTF requires min/max on a sampler's input accessor.
                    bb.accessors[t_idx]["min"] = [float(times.min())]
                    bb.accessors[t_idx]["max"] = [float(times.max())]
                    time_accessors[id(s["times"])] = t_idx
                v_idx = bb.add(
                    values, acc_type=acc_type, component_type=5126, target=None
                )
                written[sampler_idx, acc_type] = len(gltf_samplers)
                gltf_samplers.append(
                    {
                        "input": t_idx,
                        "output": v_idx,
                        "interpolation": s.get("interpolation", "LINEAR"),
                    }
                )
            gltf_channels.append(
                {"sampler": written[sampler_idx, acc_type], "target": target}
            )

        if not gltf_channels:
            continue
        entry: dict = {"channels": gltf_channels, "samplers": gltf_samplers}
        if anim.get("name") and isinstance(anim["name"], str):
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
        json_str = json.dumps(
            gltf_dict, indent=2, allow_nan=False, default=_json_default
        )
    except ValueError as exc:
        raise CodecError("glTF: mesh data contains NaN or Inf values") from exc
    write_text(path, json_str, encoding="utf-8")
