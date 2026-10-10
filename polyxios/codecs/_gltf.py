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
import re
import struct
from typing import Any
import urllib.parse

import numpy as np

from polyxios import transforms
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


def _primitive_attributes(gltf: dict, primitive: Any, where: str) -> dict:
    """Return the validated ``attributes`` object of a primitive.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    primitive
        One entry from ``mesh["primitives"]``.
    where
        Names the primitive in error messages.

    Returns
    -------
    dict
        Attribute semantic to accessor index, with ``POSITION`` present and
        every index naming an accessor of the file.

    Raises
    ------
    CodecError
        If the primitive or its attributes are not objects, ``POSITION`` is
        missing, or an attribute names no accessor.
    """
    if not isinstance(primitive, dict):
        raise CodecError(f"glTF: {where} is not an object.")
    attrs = primitive.get("attributes")
    if not isinstance(attrs, dict):
        raise CodecError(f"glTF: {where} has no attributes object.")
    if "POSITION" not in attrs:
        raise CodecError("glTF primitive is missing a POSITION attribute.")
    n_accessors = len(gltf.get("accessors", ()))
    for name, acc in attrs.items():
        if not _is_index(acc, n_accessors):
            raise CodecError(
                f"glTF: {where} attribute {name} names no accessor ({acc!r})."
            )
    return attrs


def _read_vertex_set(gltf: dict, attrs: dict, buffers: list[bytes]) -> PolyData:
    """Read the vertices and vertex attributes one ``attributes`` object names.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    attrs
        A primitive's ``attributes``, as :func:`_primitive_attributes` returns.
    buffers
        Raw buffer bytes.

    Returns
    -------
    PolyData
        The vertices and vertex attributes, with no elements.

    Raises
    ------
    CodecError
        If an attribute holds a different number of entries than ``POSITION``.
    """
    vertices = _read_accessor(gltf, attrs["POSITION"], buffers).astype(np.float64)
    n_verts = vertices.shape[0]
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

    for key, arr in vertex_attrs.items():
        if arr.shape[0] != n_verts:
            raise CodecError(
                f"glTF: vertex attribute '{key}' holds {arr.shape[0]} entries "
                f"but POSITION holds {n_verts}."
            )

    return PolyData(
        vertices=vertices,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=vertex_attrs,
    )


def _primitive_elements(
    gltf: dict,
    primitive: dict,
    buffers: list[bytes],
    *,
    n_verts: int,
    file_size: int,
    where: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the elements of one glTF mesh primitive.

    Parameters
    ----------
    gltf
        Parsed glTF JSON dict.
    primitive
        One entry from ``mesh["primitives"]``.
    buffers
        Raw buffer bytes.
    n_verts
        Number of vertices the primitive's attributes hold.
    file_size
        Total source file size, forwarded to :func:`validate_header`.
    where
        Names the primitive in error messages.

    Returns
    -------
    connectivity : numpy.ndarray
        Vertex indices into the primitive's own vertex set; int32 unless
        the set holds more than 2**31 vertices, then int64.
    offsets : numpy.ndarray
        int64 CSR offsets, starting at 0.
    element_types : numpy.ndarray
        uint8 polyxios type codes.

    Raises
    ------
    CodecError
        If ``indices`` names no accessor, or one that is not an unsigned
        integer SCALAR or points past the vertex set, or the mode is
        unsupported.
    """
    idx_dtype = np.int64 if n_verts > 2**31 else np.int32
    if "indices" in primitive:
        acc = primitive["indices"]
        if not _is_index(acc, len(gltf.get("accessors", ()))):
            raise CodecError(f"glTF: {where} indices names no accessor ({acc!r}).")
        idx_raw = _read_accessor(gltf, acc, buffers)
        if idx_raw.ndim != 1 or idx_raw.dtype.kind != "u":
            raise CodecError(
                f"glTF: {where} indices accessor {acc} is not an unsigned "
                "integer SCALAR."
            )
        if idx_raw.size > 0 and int(idx_raw.max()) >= n_verts:
            raise CodecError(
                f"glTF: primitive has index {idx_raw.max()} but only "
                f"{n_verts} vertices."
            )
        indices = idx_raw.astype(idx_dtype)
    else:
        indices = np.arange(n_verts, dtype=idx_dtype)
    n_idx = len(indices)

    mode: int = primitive.get("mode", 4)
    if isinstance(mode, bool):
        raise CodecError(f"glTF: unsupported primitive mode {mode}.")
    if mode == 4:  # TRIANGLES
        n_elements = n_idx // 3
        connectivity = indices[: n_elements * 3]
        offsets = np.arange(0, n_elements * 3 + 1, 3, dtype=np.int64)
        element_types = np.full(n_elements, _TRI_CODE, dtype=np.uint8)
    elif mode == 0:  # POINTS
        connectivity = indices
        offsets = np.arange(n_idx + 1, dtype=np.int64)
        element_types = np.full(n_idx, _VERTEX_CODE, dtype=np.uint8)
    elif mode == 1:  # LINES
        n_elements = n_idx // 2
        connectivity = indices[: n_elements * 2]
        offsets = np.arange(0, n_elements * 2 + 1, 2, dtype=np.int64)
        element_types = np.full(n_elements, _LINE_CODE, dtype=np.uint8)
    elif mode == 2:  # LINE_LOOP: strip + close
        connectivity = np.append(indices, indices[:1])
        offsets = np.array([0, n_idx + 1] if n_idx else [0], dtype=np.int64)
        element_types = np.full(min(n_idx, 1), _POLY_LINE_CODE, dtype=np.uint8)
    elif mode in (3, 5):  # LINE_STRIP, TRIANGLE_STRIP
        connectivity = indices
        offsets = np.array([0, n_idx], dtype=np.int64)
        code = _POLY_LINE_CODE if mode == 3 else _TRI_STRIP_CODE
        element_types = np.array([code], dtype=np.uint8)
    elif mode == 6:  # TRIANGLE_FAN: fan-triangulate
        n_elements = max(n_idx - 2, 0)
        connectivity = np.column_stack(
            [
                np.full(n_elements, indices[0] if n_idx else 0, dtype=idx_dtype),
                indices[1 : 1 + n_elements],
                indices[2 : 2 + n_elements],
            ]
        ).ravel()
        offsets = np.arange(0, n_elements * 3 + 1, 3, dtype=np.int64)
        element_types = np.full(n_elements, _TRI_CODE, dtype=np.uint8)
    else:
        raise CodecError(f"glTF: unsupported primitive mode {mode}.")

    validate_header(n_verts, len(element_types), len(connectivity), file_size)
    return connectivity, offsets, element_types


# =============================================================================
# Mesh → merged PolyData
# =============================================================================


def _stack_elements(
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    vertex_bases: list[int],
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay element blocks end to end over one vertex array.

    Parameters
    ----------
    blocks
        ``(connectivity, offsets, element_types)`` per primitive, as
        :func:`_primitive_elements` returns them.
    vertex_bases
        Row of the merged vertex array where each block's vertex set starts.
    n_vertices
        Number of rows in the merged vertex array.

    Returns
    -------
    connectivity : numpy.ndarray
        Indices into the merged vertex array; int32 unless the vertex count
        or the connectivity length reaches 2**31, then int64.
    offsets : numpy.ndarray
        CSR offsets, same dtype as ``connectivity``.
    element_types : numpy.ndarray
        uint8 polyxios type codes.
    """
    conn_sizes = [len(conn) for conn, _, _ in blocks]
    idx_dtype = np.int64 if max(n_vertices, sum(conn_sizes)) >= 2**31 else np.int32
    connectivity = np.concatenate(
        [np.zeros(0, dtype=idx_dtype)]
        + [
            conn.astype(idx_dtype, copy=False) + base if base else conn
            for (conn, _, _), base in zip(blocks, vertex_bases, strict=True)
        ],
        dtype=idx_dtype,
    )
    conn_starts = np.cumsum([0] + conn_sizes[:-1]).tolist()
    offsets = np.concatenate(
        [np.zeros(1, dtype=idx_dtype)]
        + [
            off[1:] + start
            for (_, off, _), start in zip(blocks, conn_starts, strict=True)
        ],
        dtype=idx_dtype,
    )
    element_types = np.concatenate(
        [np.zeros(0, dtype=np.uint8)] + [types for _, _, types in blocks]
    )
    return connectivity, offsets, element_types


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
        All primitives merged, their elements in primitive order;
        ``element_attrs["material"]`` holds per-element material indices into
        the parent :class:`~polyxios._scene.SceneData`, -1 for a primitive
        with none. Primitives with the same ``attributes`` share one vertex
        set, read once.
    """
    mesh = gltf["meshes"][mesh_index]
    primitives = mesh.get("primitives", [])
    if not primitives:
        return PolyData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
        )

    vertex_sets: dict[tuple, PolyData] = {}
    set_keys: list[tuple] = []
    for p, prim in enumerate(primitives):
        attrs = _primitive_attributes(gltf, prim, f"mesh {mesh_index} primitive {p}")
        key = tuple(sorted(attrs.items()))
        if key not in vertex_sets:
            vertex_sets[key] = _read_vertex_set(gltf, attrs, buffers)
        set_keys.append(key)

    sets = list(vertex_sets.values())
    starts = np.cumsum([0] + [s.vertices.shape[0] for s in sets[:-1]])
    set_start = dict(zip(vertex_sets, starts.tolist(), strict=True))
    blocks = [
        _primitive_elements(
            gltf,
            prim,
            buffers,
            n_verts=vertex_sets[key].vertices.shape[0],
            file_size=file_size,
            where=f"mesh {mesh_index} primitive {p}",
        )
        for p, (prim, key) in enumerate(zip(primitives, set_keys, strict=True))
    ]
    connectivity, offsets, element_types = _stack_elements(
        blocks,
        vertex_bases=[set_start[key] for key in set_keys],
        n_vertices=int(starts[-1]) + sets[-1].vertices.shape[0],
    )

    element_attrs: dict[str, np.ndarray] = {}
    materials = [prim.get("material", -1) for prim in primitives]
    n_materials = len(gltf.get("materials", ()))
    for p, mat in enumerate(materials):
        if mat != -1 and not _is_index(mat, n_materials):
            raise CodecError(
                f"glTF: mesh {mesh_index} primitive {p} names no material ({mat!r})."
            )
    if any(m >= 0 for m in materials):
        element_attrs["material"] = np.repeat(
            np.array(materials, dtype=np.int32),
            [len(types) for _, _, types in blocks],
        )

    shared = transforms.merge(*sets)
    result = dataclasses.replace(
        shared,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        element_attrs=element_attrs,
    )
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
        raise CodecError(
            "glTF: a value is NaN or Inf; the scene cannot be written."
        ) from exc
    except TypeError as exc:
        raise CodecError(f"glTF: {exc}; the scene cannot be written.") from exc
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
}

_INFLUENCE_KEY = re.compile(r"(joints|weights)(_[1-9][0-9]*)?")

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


_FRAMES = (("textangent", "texbinormal"), ("tangent", "binormal"))


def _rows(attrs: dict[str, Any], *, key: str, n: int, width: int) -> np.ndarray | None:
    """Return ``attrs[key]`` as float64 when it is ``width`` numbers per vertex.

    Parameters
    ----------
    attrs
        The mesh's vertex attributes.
    key
        The attribute to look up.
    n
        The mesh's vertex count.
    width
        The numbers each vertex must hold.

    Returns
    -------
    ndarray or None
        ``(n, width)`` float64, None when the key is missing or of another
        shape or kind.
    """
    arr = np.asarray(attrs.get(key))
    if arr.shape != (n, width) or arr.dtype.kind not in "iuf":
        return None
    return arr.astype(np.float64)


_SAFE_BAND = (1e-100, 1e100)


def scaled_rows(rows: np.ndarray) -> np.ndarray:
    """Return ``rows`` each divided by its largest magnitude.

    A row keeps its direction and side but no longer overflows or underflows
    a norm or a dot product, as a finite row of values near the float64
    limit, or near its smallest normal number, would. A row of zeros or
    holding a non-finite value is left as is.

    Parameters
    ----------
    rows
        ``(n, 3)`` float64 rows.

    Returns
    -------
    ndarray
        The scaled rows, or ``rows`` itself when every nonzero finite row's
        largest magnitude lies within ``_SAFE_BAND``.
    """
    size = np.abs(rows)
    peak = np.maximum(np.maximum(size[:, 0], size[:, 1]), size[:, 2])
    scale = np.isfinite(peak) & (peak > 0)
    low, high = _SAFE_BAND
    if not (scale & ((peak < low) | (peak > high))).any():
        return rows
    return rows / np.where(scale, peak, 1.0)[:, None]


def _tangents_of_frame(
    attrs: dict[str, Any], *, n: int, what: str
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Return glTF tangents rebuilt from a COLLADA tangent frame, and its keys.

    A COLLADA read names a tangent frame ``textangent`` and ``texbinormal``
    (or ``tangent`` and ``binormal``), where glTF holds the tangent's xyz and
    the handedness ``w`` of the bitangent ``w * cross(normal, tangent)``.
    Only set 0 of each is looked at. The first frame holding both a tangent
    and a bitangent of 3 numbers per vertex is taken, else the first holding
    a tangent alone, whose handedness is then +1, with a warning; so is
    that of a row whose bitangent lies on neither side of
    ``cross(normal, tangent)`` (zero, NaN, or at right angles to it to
    within rounding, where the side is noise).

    Parameters
    ----------
    attrs
        The mesh's vertex attributes, without ``tangents`` glTF can hold.
    n
        The mesh's vertex count.
    what
        The mesh's name in the warnings.

    Returns
    -------
    tangents : ndarray or None
        ``(n, 4)`` float64 tangents of unit xyz, None when no frame holds a
        tangent of 3 numbers per vertex, or, with a warning, when the mesh
        has no ``normals`` of 3 columns (glTF has clients ignore a
        ``TANGENT`` without a ``NORMAL``) or a tangent row of zero or
        non-finite length, which no unit vector stands for.
    keys : tuple of str
        The attributes the tangents stand for, or the frame left out.
    """
    vec = {
        key: arr
        for frame in _FRAMES
        for key in frame
        if (arr := _rows(attrs, key=key, n=n, width=3)) is not None
    }
    tangents = [t for t, _ in _FRAMES if t in vec]
    if not tangents:
        return None, ()
    complete = [(t, b) for t, b in _FRAMES if t in vec and b in vec]
    t_key, b_key = complete[0] if complete else (tangents[0], None)
    keys = (t_key, b_key) if b_key else (t_key,)
    normals = _rows(attrs, key="normals", n=n, width=3)
    if normals is None:
        _warn_caller(
            f"{what}: {t_key} has no normals of 3 columns beside it, and glTF "
            "ignores a TANGENT without a NORMAL; it is not written."
        )
        return None, keys
    tangent = scaled_rows(vec[t_key])
    length = np.linalg.norm(tangent, axis=1)
    bad = ~(np.isfinite(length) & (length > 0))
    if bad.any():
        _warn_caller(
            f"{what}: {t_key} holds {int(bad.sum())} row(s) of zero or "
            "non-finite length, which a glTF TANGENT of unit length cannot "
            "hold; it is not written."
        )
        return None, keys
    tangent /= length[:, None]
    if b_key is None:
        _warn_caller(
            f"{what}: {t_key} goes out as TANGENT without a bitangent of 3 "
            "columns to give its handedness; it is taken as +1."
        )
        return np.column_stack([tangent, np.ones(n)]), keys
    across = np.cross(scaled_rows(normals), tangent)
    bitangent = scaled_rows(vec[b_key])
    side = np.einsum("ij,ij->i", across, bitangent)
    # The dot product carries a rounding error of a few ulps of the product
    # of the lengths, so a side within that is no side at all.
    noise = (
        8
        * np.finfo(np.float64).eps
        * np.linalg.norm(across, axis=1)
        * np.linalg.norm(bitangent, axis=1)
    )
    flat = ~(np.abs(side) > noise)
    if flat.any():
        _warn_caller(
            f"{what}: {b_key} holds {int(flat.sum())} row(s) on neither side of "
            f"cross(normals, {t_key}); their handedness is taken as +1."
        )
    w = np.where((side < 0) & ~flat, -1.0, 1.0)
    return np.column_stack([tangent, w]), keys


def _polydata_to_primitives(
    poly: PolyData,
    bb: _BinBuilder,
    mat_remap: dict[int, int] | None = None,
    *,
    what: str = "glTF",
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
    what
        The mesh's name in the warnings, ``glTF`` alone or with its index.

    Returns
    -------
    list[dict]
        glTF primitive dicts ready for ``mesh["primitives"]``. Vertex and
        element attributes glTF has no place for, and tags, are left out
        with a warning.
    """
    # POSITION accessor (float32).
    pos_f32 = poly.vertices.astype(np.float32)
    pos_idx = bb.add(pos_f32, acc_type="VEC3", component_type=5126)

    # Vertex attribute accessors.
    va_accessors: dict[str, int] = {}
    handled: set[str] = set()
    sets, rejected = _influence_sets(poly.vertex_attrs, what=what)
    if rejected:
        handled.update(k for k in poly.vertex_attrs if _INFLUENCE_KEY.fullmatch(k))
    for n, (joints_key, weights_key) in enumerate(sets):
        joints = np.asarray(poly.vertex_attrs[joints_key])
        weights = np.asarray(poly.vertex_attrs[weights_key])
        wide = joints.size > 0 and joints.max() > 255
        va_accessors[f"JOINTS_{n}"] = bb.add(
            joints.astype(np.uint16 if wide else np.uint8),
            acc_type="VEC4",
            component_type=5123 if wide else 5121,
        )
        if weights.dtype.kind == "u":
            weights = weights / np.iinfo(weights.dtype).max
        va_accessors[f"WEIGHTS_{n}"] = bb.add(
            weights.astype(np.float32), acc_type="VEC4", component_type=5126
        )
        handled.update((joints_key, weights_key))
    n_verts = len(pos_f32)
    own_tangents = (
        _rows(poly.vertex_attrs, key="tangents", n=n_verts, width=4) is not None
    )
    if "tangents" in poly.vertex_attrs and not own_tangents:
        handled.add("tangents")
        _warn_caller(
            f"{what}: tangents of shape "
            f"{np.shape(poly.vertex_attrs['tangents'])} are not the 4 numbers "
            f"for each of the {n_verts} vertices a glTF TANGENT holds; they "
            "are not written."
        )
    for key, (semantic, comp_type) in _GLTF_ATTR_MAP.items():
        if key not in poly.vertex_attrs or key in handled:
            continue
        handled.add(key)
        arr = poly.vertex_attrs[key]

        if key == "colors":
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
    if not own_tangents:
        tangents, keys = _tangents_of_frame(poly.vertex_attrs, n=n_verts, what=what)
        if tangents is not None:
            va_accessors["TANGENT"] = bb.add(
                tangents.astype(np.float32), acc_type="VEC4", component_type=5126
            )
        handled.update(keys)
    unwritten = sorted(set(poly.vertex_attrs) - handled)
    if unwritten:
        _warn_caller(
            f"{what}: vertex attribute(s) {unwritten} have no glTF semantic; they "
            "are not written."
        )
    others = sorted(k for k in poly.element_attrs if k != "material")
    if others:
        _warn_caller(
            f"{what}: element attribute(s) {others} have no glTF counterpart "
            "(only material does); they are not written."
        )
    tags = sorted({*poly.vertex_tags, *poly.element_tags})
    if tags:
        _warn_caller(
            f"{what}: tag group(s) {tags} have no glTF counterpart; they are not "
            "written."
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


def _influence_key(name: str, n: int) -> str:
    """Return the vertex attribute holding influence set ``n`` of ``name``."""
    return name if n == 0 else f"{name}_{n}"


def _influence_sets(
    attrs: dict[str, Any], *, what: str
) -> tuple[list[tuple[str, str]], bool]:
    """Return the ``(joints, weights)`` keys of the influence sets to write.

    Set ``n`` is ``joints`` and ``weights`` for 0, ``joints_<n>`` and
    ``weights_<n>`` after, as a read names ``JOINTS_<n>`` and
    ``WEIGHTS_<n>``. glTF numbers them from 0 with no gap, so the first set
    that cannot be written ends the list, with a warning.

    Parameters
    ----------
    attrs
        The mesh's vertex attributes.
    what
        The mesh's name in the warning.

    Returns
    -------
    tuple
        The keys of each set, in order, and whether a set was rejected, its
        warning then covering every influence key left out.
    """
    sets: list[tuple[str, str]] = []
    while True:
        n = len(sets)
        keys = (_influence_key("joints", n), _influence_key("weights", n))
        if keys[0] not in attrs and keys[1] not in attrs:
            return sets, False
        why = _unwritable_influences(attrs, *keys)
        if why is not None:
            _warn_caller(
                f"{what}: {why}; JOINTS_{n} and WEIGHTS_{n} are not written, nor "
                "any set after them."
            )
            return sets, True
        sets.append(keys)


def _unwritable_influences(
    attrs: dict[str, Any], joints_key: str, weights_key: str
) -> str | None:
    """Return why an influence set cannot go out, None when it can.

    glTF takes it as a pair of ``VEC4`` accessors, one row per vertex,
    joints whole numbers in 0..65535 and weights floats, or uint8 or uint16
    normalised as glTF stores them.

    Parameters
    ----------
    attrs
        The mesh's vertex attributes, holding at least one of the two keys.
    joints_key, weights_key
        The keys of the set's joints and weights.

    Returns
    -------
    str or None
        The reason, for the warning.
    """
    if joints_key not in attrs or weights_key not in attrs:
        return f"{joints_key} and {weights_key} come as a pair, and only one is given"
    joints = np.asarray(attrs[joints_key])
    weights = np.asarray(attrs[weights_key])
    if joints.ndim != 2 or joints.shape[1] != 4 or weights.shape != joints.shape:
        return (
            f"{joints_key} {joints.shape} and {weights_key} {weights.shape} are "
            "not four per vertex"
        )
    if weights.dtype.kind != "f" and weights.dtype not in (np.uint8, np.uint16):
        return f"{weights_key} are neither floats nor normalised uint8 or uint16"
    bad = f"{joints_key} are not integers in 0..65535"
    if joints.dtype.kind not in "iuf":
        return bad
    if joints.size == 0:
        return None
    if joints.dtype.kind == "f" and not (
        np.isfinite(joints).all() and np.array_equal(joints, np.round(joints))
    ):
        return bad
    if joints.min() < 0 or joints.max() > 65535:
        return bad
    return None


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

    Notes
    -----
    ``global_attrs["mesh_name"]`` becomes the mesh's ``name``, and
    ``global_attrs["asset"]`` gives its ``copyright``. Vertex attributes
    glTF has no semantic for, element attributes other than ``material``,
    tags and other globals are not written, each with a warning.
    Influences are written as :func:`write_scene` writes them, but with
    no skin: a flat write has no node to hang one on.
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
    mesh_entry = _mesh_entry(primitives, poly.global_attrs, "glTF", keep={"asset"})

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
        "meshes": [mesh_entry],
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
        no writable primitives, if the matrix of an animated node holds
        NaN or Inf, or if a value passed through as it is, such as an
        animation target's ``extras``, is not JSON.

    Notes
    -----
    Skins in ``global_attrs["skins"]`` are written with their ``name``,
    ``joints`` and ``skeleton`` node indices, their ``extras`` and their
    decoded ``inverse_bind_matrices`` (identity when absent) as a float
    MAT4 accessor; a node's ``extras["skin"]`` index becomes its ``skin``,
    and its mesh's ``joints`` and ``weights`` its ``JOINTS_0`` and
    ``WEIGHTS_0``. A ``bind_shape_matrix``, which glTF lacks, is folded
    into every inverse bind matrix: a joint then moves the bind shape the
    same way. A skin is not written, with a warning, when it is not a dict,
    its joints are not distinct node indices sharing one root, its inverse
    bind matrices are not one finite 4x4 per joint float32 can hold, with
    a last row of ``[0, 0, 0, 1]`` once the bind shape is folded in (or an
    ``inverseBindMatrices`` accessor was never decoded) or its
    ``bind_shape_matrix`` is not a finite 4x4; a ``skeleton`` that is not
    an ancestor of (or is) every joint is dropped, and so are a ``name``
    that is not a string, ``extras`` that are not JSON, ``extensions``
    (whose ``extensionsUsed`` the writer does not keep) and keys glTF has
    no place for, each with a warning. A node keeps no ``skin``, with a
    warning, when it names one the scene lacks, has no mesh, or its mesh
    has no ``JOINTS_0`` and ``WEIGHTS_0`` or a joint index past the skin's
    joints. ``joints`` and ``weights`` are written as a pair, four per
    vertex, joints whole numbers in 0..65535 and weights floats; uint8 and
    uint16 weights are taken as normalised, as glTF stores them, and
    written divided by 255 or 65535. ``joints_<n>`` and ``weights_<n>``
    go out as ``JOINTS_<n>`` and ``WEIGHTS_<n>``, numbered from 1 with no
    gap; the first set that breaks any of this is not written, nor any
    set after it, with a warning. Per-node ``camera`` extras are not
    written.

    Each mesh's ``global_attrs["mesh_name"]`` becomes its ``name``, and the
    scene's ``global_attrs["extras"]`` the file's ``extras``. Vertex
    attributes glTF has no semantic for, element attributes other than
    ``material``, tags, other mesh globals, the scene's ``extensions``
    and other scene globals are not written, each with a warning.

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
        what = f"glTF: mesh {mesh_idx}"
        prims = _polydata_to_primitives(mesh_poly, bb, what=what)
        if not prims:
            raise CodecError(
                f"glTF: mesh {mesh_idx} produced no writable primitives; "
                "remove it or filter out unsupported element types first."
            )
        gltf_meshes.append(_mesh_entry(prims, mesh_poly.global_attrs, what))
    gltf_skins, skin_of = _skins_of_nodes(scene, bb, gltf_meshes)

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
        if node_i in skin_of:
            n_entry["skin"] = skin_of[node_i]
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
    if gltf_skins:
        gltf_dict["skins"] = gltf_skins
    extras = _scene_extras(scene.global_attrs)
    if extras is not None:
        gltf_dict["extras"] = extras
    if gltf_animations:
        gltf_dict["animations"] = gltf_animations

    _write_output(gltf_dict, bb.bin_data, path, binary=binary)


def _mesh_entry(
    primitives: list[dict],
    global_attrs: dict[str, Any],
    what: str,
    *,
    keep: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """Return the glTF mesh object of ``primitives``, named by ``mesh_name``.

    Parameters
    ----------
    primitives
        The mesh's primitives.
    global_attrs
        The mesh's global attributes; ``mesh_name`` becomes its ``name``.
    what
        The mesh's name in the warnings.
    keep
        Further keys the caller writes elsewhere; any other is left out
        with a warning.

    Returns
    -------
    dict
        The mesh object.
    """
    entry: dict = {"primitives": primitives}
    name = global_attrs.get("mesh_name")
    if isinstance(name, str):
        if name:
            entry["name"] = name
    elif name is not None:
        _warn_caller(f"{what}: mesh_name {name!r} is not a string; it is dropped.")
    lost = sorted(str(k) for k in global_attrs if k != "mesh_name" and k not in keep)
    if lost:
        _warn_caller(
            f"{what}: global_attrs {lost} have no glTF counterpart (only "
            "mesh_name does); they are not written."
        )
    return entry


# The scene globals a write reads; ``extensions`` is not among them, since an
# extension must also be listed in ``extensionsUsed``.
_SCENE_KEYS = frozenset({"asset", "skins", "animations", "extras"})


def _scene_extras(global_attrs: dict[str, Any]) -> Any:
    """Return the scene's ``extras`` to write, None when there are none.

    ``extras`` that are not JSON, ``extensions`` and globals glTF has no
    place for are left out with a warning.
    """
    extras = global_attrs.get("extras")
    if extras is not None and not _is_json(extras):
        _warn_caller("glTF: global_attrs['extras'] are not JSON; they are dropped.")
        extras = None
    if "extensions" in global_attrs:
        _warn_caller(
            "glTF: global_attrs['extensions'] are not written: glTF needs each "
            "listed in extensionsUsed, which the writer does not keep."
        )
    lost = sorted(
        str(k) for k in global_attrs if k not in _SCENE_KEYS and k != "extensions"
    )
    if lost:
        _warn_caller(
            f"glTF: global_attrs {lost} have no glTF counterpart; they are not written."
        )
    return extras


# The keys of a ``global_attrs["skins"]`` entry a write reads; the raw
# ``inverseBindMatrices`` accessor index is superseded by its decoded matrices,
# and COLLADA's ``bind_shape_matrix`` is folded into them. ``extensions`` is
# left out: an extension must also be listed in ``extensionsUsed``, which the
# writer does not keep.
_SKIN_KEYS = frozenset(
    {
        "name",
        "joints",
        "skeleton",
        "inverse_bind_matrices",
        "inverseBindMatrices",
        "bind_shape_matrix",
        "extras",
    }
)


def _node_index(value: Any, n: int) -> int | None:
    """Return ``value`` as an index below ``n``, None when it is not one."""
    if isinstance(value, np.integer):
        value = int(value)
    return value if _is_index(value, n) else None


def _matrices(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    """Return ``value`` as finite float64 of ``shape``, None when it is not."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != shape or not np.isfinite(arr).all():
        return None
    return arr


def _is_json(value: Any) -> bool:
    """Return whether ``value`` serialises to JSON with no NaN or Inf."""
    try:
        json.dumps(value, allow_nan=False, default=_json_default)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _skin_matrices(
    what: str, skin: dict, n_joints: int
) -> tuple[bool, np.ndarray | None]:
    """Return the inverse bind matrices skin ``what`` writes, bind shape folded in.

    Parameters
    ----------
    what
        The skin's name in the warnings.
    skin
        The ``global_attrs["skins"]`` entry.
    n_joints
        The number of its joints.

    Returns
    -------
    tuple
        Whether the skin can be written (False comes with a warning), and
        its ``(n_joints, 4, 4)`` row-major matrices, None when it has neither
        matrices nor bind shape, glTF's identity.
    """
    ibm = skin.get("inverse_bind_matrices")
    if ibm is None and skin.get("inverseBindMatrices") is not None:
        _warn_caller(
            f"{what} names inverse bind matrices that were never decoded; it is "
            "not written."
        )
        return False, None
    shape = (n_joints, 4, 4)
    matrices = None if ibm is None else _matrices(ibm, shape)
    if ibm is not None and matrices is None:
        _warn_caller(
            f"{what} has inverse bind matrices that are not {n_joints} finite "
            "4x4 matrices, one per joint; it is not written."
        )
        return False, None
    bsm = skin.get("bind_shape_matrix")
    if bsm is not None:
        bind_shape = _matrices(bsm, (4, 4))
        if bind_shape is None:
            _warn_caller(
                f"{what} has a bind_shape_matrix that is not a finite 4x4 matrix; "
                "it is not written."
            )
            return False, None
        if matrices is None:
            matrices = np.broadcast_to(np.eye(4), shape)
        matrices = matrices @ bind_shape
    if matrices is None:
        return True, None
    if np.abs(matrices).max() > np.finfo(np.float32).max:
        _warn_caller(
            f"{what} has inverse bind matrices float32 cannot hold; it is not written."
        )
        return False, None
    if not np.allclose(matrices[:, 3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
        _warn_caller(
            f"{what} has inverse bind matrices whose last row is not "
            "[0, 0, 0, 1], which glTF requires; it is not written."
        )
        return False, None
    matrices = matrices.copy()
    matrices[:, 3] = (0.0, 0.0, 0.0, 1.0)
    return True, matrices


def _encode_skin(
    k: int, skin: Any, bb: _BinBuilder, parent: dict[int, int], n_nodes: int
) -> dict | None:
    """Return skin ``k`` as a glTF skin object, None when it cannot be one.

    Parameters
    ----------
    k
        The skin's index in ``global_attrs["skins"]``, for the warnings.
    skin
        The entry itself.
    bb
        The buffer builder its inverse bind matrices are appended to.
    parent
        Each node's parent, for checking that the joints share a root and
        that ``skeleton`` is an ancestor of them.
    n_nodes
        The number of nodes of the scene.

    Returns
    -------
    dict or None
        None, with a warning, when the skin is not written.
    """
    what = f"glTF: skin {k}"
    if not isinstance(skin, dict):
        _warn_caller(f"{what} is not a dict; it is not written.")
        return None
    joints = skin.get("joints")
    if isinstance(joints, np.ndarray):
        joints = joints.tolist()
    nodes = (
        [_node_index(j, n_nodes) for j in joints]
        if isinstance(joints, (list, tuple))
        else []
    )
    if not nodes or None in nodes or len(set(nodes)) != len(nodes):
        _warn_caller(
            f"{what} joints {joints!r} are not distinct node indices; it is not "
            "written."
        )
        return None
    if len({_root(j, parent) for j in nodes}) > 1:
        _warn_caller(
            f"{what} joints {nodes} do not share a root node, which glTF "
            "requires; it is not written."
        )
        return None
    ok, matrices = _skin_matrices(what, skin, len(nodes))
    if not ok:
        return None
    entry: dict = {}
    name = skin.get("name")
    if isinstance(name, str):
        entry["name"] = name
    elif name is not None:
        _warn_caller(f"{what} name {name!r} is not a string; it is dropped.")
    entry["joints"] = nodes
    if matrices is not None:
        column_major = matrices.transpose(0, 2, 1).astype(np.float32)
        entry["inverseBindMatrices"] = bb.add(
            np.ascontiguousarray(column_major).reshape(-1, 16),
            acc_type="MAT4",
            component_type=5126,
            target=None,
        )
    if skin.get("skeleton") is not None:
        top = _node_index(skin["skeleton"], n_nodes)
        if top is not None and all(_rooted(j, top, parent) for j in nodes):
            entry["skeleton"] = top
        else:
            _warn_caller(
                f"{what} skeleton {skin['skeleton']!r} is not an ancestor of every "
                "joint; it is dropped."
            )
    if skin.get("extras") is not None:
        if _is_json(skin["extras"]):
            entry["extras"] = skin["extras"]
        else:
            _warn_caller(f"{what} extras are not JSON; they are dropped.")
    if "extensions" in skin:
        _warn_caller(
            f"{what} extensions are not written: glTF needs each listed in "
            "extensionsUsed, which the writer does not keep."
        )
    dropped = sorted(
        str(key) for key in skin if key not in _SKIN_KEYS and key != "extensions"
    )
    if dropped:
        _warn_caller(f"{what} keys {dropped} have no glTF field; they are not written.")
    return entry


def _root(node: int, parent: dict[int, int]) -> int:
    """Return the topmost ancestor of ``node``, ``node`` itself when it has none."""
    seen = {node}
    while node in parent and parent[node] not in seen:
        node = parent[node]
        seen.add(node)
    return node


def _rooted(node: int, top: int, parent: dict[int, int]) -> bool:
    """Return whether ``top`` is ``node`` or one of its ancestors."""
    seen: set[int] = set()
    while node != top:
        if node not in parent or node in seen:
            return False
        seen.add(node)
        node = parent[node]
    return True


def _skins_of_nodes(
    scene: SceneData, bb: _BinBuilder, gltf_meshes: list[dict]
) -> tuple[list[dict], dict[int, int]]:
    """Return the glTF skins of ``scene`` and the skin each node takes.

    Parameters
    ----------
    scene
        The scene being written.
    bb
        The buffer builder the inverse bind matrices are appended to.
    gltf_meshes
        The meshes as written, whose primitives say whether ``JOINTS_0``
        and ``WEIGHTS_0`` went out.

    Returns
    -------
    tuple
        The ``skins`` array and a ``node -> skin`` map over the nodes whose
        ``extras["skin"]`` names a written skin their mesh can wear; a node
        naming a skin that was not written is left unskinned silently, the
        skin's own warning covering it.
    """
    skins = scene.global_attrs.get("skins")
    if skins is None:
        skins = []
    elif not isinstance(skins, (list, tuple)):
        _warn_caller("glTF: global_attrs['skins'] is not a list; it is not written.")
        skins = []
    n_nodes = len(scene.nodes)
    parent: dict[int, int] = {}
    for i, node in enumerate(scene.nodes):
        for child in node.children:
            parent.setdefault(child, i)
    out: list[dict] = []
    written: dict[int, int] = {}
    for k, skin in enumerate(skins):
        entry = _encode_skin(k, skin, bb, parent, n_nodes)
        if entry is not None:
            written[k] = len(out)
            out.append(entry)

    skin_of: dict[int, int] = {}
    unskinned: dict[str, list[int]] = {}
    for i, node in enumerate(scene.nodes):
        ref = node.extras.get("skin")
        if ref is None:
            continue
        k = _node_index(ref, len(skins))
        mesh = _node_index(node.mesh, len(gltf_meshes))
        if k is None:
            why = "name a skin global_attrs['skins'] lacks"
        elif k not in written:
            continue
        elif mesh is None:
            why = "name a skin but no mesh for it to deform"
        else:
            why = _misfit(scene.meshes[mesh], gltf_meshes[mesh], out[written[k]])
            if why is None:
                skin_of[i] = written[k]
                continue
        unskinned.setdefault(why, []).append(i)
    for why, nodes in unskinned.items():
        _warn_caller(f"glTF: node(s) {_few(nodes)} {why}; they are written unskinned.")
    return out, skin_of


def _few(indices: list[int], *, limit: int = 10) -> str:
    """Return ``indices`` for a warning, the first ``limit`` and a count of the rest."""
    if len(indices) <= limit:
        return str(indices)
    return f"{indices[:limit]} and {len(indices) - limit} more"


def _misfit(poly: PolyData, mesh: dict, skin: dict) -> str | None:
    """Return why ``mesh`` cannot wear ``skin``, None when it can.

    Every primitive of a skinned mesh needs ``JOINTS_0`` and ``WEIGHTS_0``,
    and every joint index of every ``JOINTS_<n>`` must name one of the
    skin's joints, a padding slot of zero weight included.
    """
    if not all(
        "JOINTS_0" in p["attributes"] and "WEIGHTS_0" in p["attributes"]
        for p in mesh["primitives"]
    ):
        return "name a skin but their mesh has no JOINTS_0 and WEIGHTS_0"
    attributes = mesh["primitives"][0]["attributes"]
    top = max(
        int(joints.max()) if joints.size else -1
        for joints in (
            np.asarray(poly.vertex_attrs[_influence_key("joints", n)])
            for n in range(len(attributes))
            if f"JOINTS_{n}" in attributes
        )
    )
    n_joints = len(skin["joints"])
    if top >= n_joints:
        return (
            f"name a skin of {n_joints} joint(s) but their mesh's joints reach "
            f"index {top}"
        )
    return None


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
        raise CodecError(
            "glTF: a value is NaN or Inf; the scene cannot be written."
        ) from exc
    except TypeError as exc:
        raise CodecError(f"glTF: {exc}; the scene cannot be written.") from exc
    write_text(path, json_str, encoding="utf-8")
