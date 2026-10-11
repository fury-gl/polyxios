"""Tests for the glTF/GLB codec.

All tests use real file I/O via ``tmp_path``; no mocks or patches.
The ``_make_glb`` helper assembles valid GLB bytes from a JSON dict and
an optional binary chunk, independently of the codec's own GLB writer.
"""

import dataclasses
import io
import json
from pathlib import Path
import struct
from typing import Any
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._scene import SceneData, SceneMaterial, SceneNode, SceneTexture
from polyxios._trs import matrix_of_trs
from polyxios._types import PolyData
from polyxios.codecs import _collada
from polyxios.codecs._gltf import (
    _decode_skins,
    _increasing_float32,
    _parse_glb,
    _stack_elements,
    _tangents_of_frame,
    read,
    read_scene,
    write,
    write_scene,
)
from polyxios.exceptions import CodecError

# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942


def _make_glb(gltf_json: dict, bin_data: bytes | None = None) -> bytes:
    """Assemble a valid GLB from a JSON dict and optional binary chunk."""

    def _pad4(b: bytes, pad: bytes = b" ") -> bytes:
        rem = len(b) % 4
        return b + pad * (4 - rem) if rem else b

    json_bytes = _pad4(json.dumps(gltf_json).encode("utf-8"), pad=b" ")
    chunks = struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
    if bin_data is not None:
        padded = _pad4(bin_data, pad=b"\x00")
        chunks += struct.pack("<II", len(padded), _CHUNK_BIN) + padded
    total = 12 + len(chunks)
    return struct.pack("<III", _GLB_MAGIC, 2, total) + chunks


def _triangle_glb() -> tuple[bytes, np.ndarray]:
    """Return (glb_bytes, vertices) for a single triangle GLB."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)

    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    # Align indices to 4-byte boundary: 12*3=36 bytes for verts, pad 2.
    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()

    # BufferView 0: positions (36 bytes), BV 1: indices (6 bytes, offset 38)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 38, "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    return _make_glb(gltf, bin_data), verts.astype(np.float64)


def _triangle_poly() -> PolyData:
    """A single-triangle PolyData for write tests."""
    return make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )


# ---------------------------------------------------------------------------
# Read tests
# ---------------------------------------------------------------------------


def test_read_minimal_glb(tmp_path: Path) -> None:
    glb = _make_glb({"asset": {"version": "2.0"}})
    p = tmp_path / "minimal.glb"
    p.write_bytes(glb)
    with pytest.warns(UserWarning, match="scene format.*read_scene"):
        poly = read(p)
    assert poly.vertices.shape == (0, 3)
    assert len(poly.element_types) == 0


def test_read_triangle_glb(tmp_path: Path) -> None:
    glb, expected_verts = _triangle_glb()
    p = tmp_path / "tri.glb"
    p.write_bytes(glb)
    scene = read_scene(p)
    assert len(scene.meshes) == 1
    poly = scene.meshes[0]
    assert poly.vertices.shape == (3, 3)
    assert list(poly.element_types) == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_read_with_normals_and_texcoords(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    texcoords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)

    bin_data = (
        verts.tobytes()
        + normals.tobytes()
        + texcoords.tobytes()
        + b"\x00\x00"  # pad to 4-byte boundary before indices
        + indices.tobytes()
    )
    offsets = {
        "verts": 0,
        "normals": 36,
        "texcoords": 36 + 36,
        "indices": 36 + 36 + 24 + 2,
    }
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "mode": 4,
                    }
                ]
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets["verts"], "byteLength": 36},
            {"buffer": 0, "byteOffset": offsets["normals"], "byteLength": 36},
            {"buffer": 0, "byteOffset": offsets["texcoords"], "byteLength": 24},
            {"buffer": 0, "byteOffset": offsets["indices"], "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    p = tmp_path / "attrs.glb"
    p.write_bytes(_make_glb(gltf, bin_data))
    scene = read_scene(p)
    poly = scene.meshes[0]
    assert poly.vertex_attrs["normals"].shape == (3, 3)
    assert poly.vertex_attrs["texcoords"].shape == (3, 2)


def test_read_non_indexed_primitive(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}],
    }
    p = tmp_path / "nonindexed.glb"
    p.write_bytes(_make_glb(gltf, verts.tobytes()))
    scene = read_scene(p)
    poly = scene.meshes[0]
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_read_multiple_primitives(tmp_path: Path) -> None:
    verts = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    # Two primitives sharing the same position accessor, different materials.
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                        "material": 0,
                    },
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                        "material": 1,
                    },
                ]
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 38, "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "materials": [{"name": "mat0"}, {"name": "mat1"}],
    }
    p = tmp_path / "multi_prim.glb"
    p.write_bytes(_make_glb(gltf, bin_data))
    scene = read_scene(p)
    poly = scene.meshes[0]
    assert poly.vertices.shape[0] == 3
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(poly.offsets, [0, 3, 6])
    np.testing.assert_array_equal(poly.element_attrs["material"], [0, 1])


_QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)


def _primitives_glb(
    primitives: list[Any],
    *,
    index_blocks: tuple[np.ndarray, ...] = (),
    n_normals: int = 4,
) -> bytes:
    """Build a one-mesh GLB over a 4-vertex quad.

    Accessor 0 and 1 are two POSITION accessors over the same quad, accessor
    2 a NORMAL accessor of ``n_normals`` entries, and accessors 3 onwards the
    ``index_blocks``, each in the component type of its numpy dtype.
    """
    comp = {
        np.dtype(np.int8): 5120,
        np.dtype(np.uint8): 5121,
        np.dtype(np.uint16): 5123,
        np.dtype(np.uint32): 5125,
        np.dtype(np.float32): 5126,
    }
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (n_normals, 1))
    bin_data = _QUAD.tobytes() + normals.tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": 48},
        {"buffer": 0, "byteOffset": 48, "byteLength": normals.nbytes},
    ]
    accessors: list[dict] = [
        {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
        {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
        {"bufferView": 1, "componentType": 5126, "count": n_normals, "type": "VEC3"},
    ]
    for block in index_blocks:
        bin_data += b"\x00" * (-len(bin_data) % 4)
        views.append(
            {"buffer": 0, "byteOffset": len(bin_data), "byteLength": block.nbytes}
        )
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": comp[block.dtype],
                "count": len(block),
                "type": "SCALAR",
            }
        )
        bin_data += block.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": primitives}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(bin_data)}],
        "materials": [{"name": f"mat{k}"} for k in range(3)],
    }
    return _make_glb(gltf, bin_data)


def _read_primitives(tmp_path: Path, primitives: list[Any], **kwargs: Any) -> PolyData:
    p = tmp_path / "prims.glb"
    p.write_bytes(_primitives_glb(primitives, **kwargs))
    return read_scene(p).meshes[0]


def test_read_primitives_with_own_accessors_keep_their_vertices_and_order(
    tmp_path: Path,
) -> None:
    tri = np.array([0, 1, 2], dtype=np.uint16)
    poly = _read_primitives(
        tmp_path,
        [
            {"attributes": {"POSITION": 0}, "indices": 3, "material": 0},
            {"attributes": {"POSITION": 1}, "indices": 3, "material": 1},
            {"attributes": {"POSITION": 0}, "indices": 3, "material": 2},
        ],
        index_blocks=(tri,),
    )
    np.testing.assert_array_equal(poly.vertices, np.vstack([_QUAD, _QUAD]))
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 4, 5, 6, 0, 1, 2])
    np.testing.assert_array_equal(poly.offsets, [0, 3, 6, 9])
    np.testing.assert_array_equal(poly.element_attrs["material"], [0, 1, 2])


def test_read_shared_vertices_fill_a_missing_material_with_minus_one(
    tmp_path: Path,
) -> None:
    tri = np.array([0, 1, 2], dtype=np.uint16)
    poly = _read_primitives(
        tmp_path,
        [
            {"attributes": {"POSITION": 0}, "indices": 3, "material": 1},
            {"attributes": {"POSITION": 0}, "indices": 3},
        ],
        index_blocks=(tri,),
    )
    assert poly.vertices.shape[0] == 4
    np.testing.assert_array_equal(poly.element_attrs["material"], [1, -1])


def test_read_shared_vertices_across_primitive_modes(tmp_path: Path) -> None:
    attrs = {"POSITION": 0, "NORMAL": 2}
    poly = _read_primitives(
        tmp_path,
        [
            {"attributes": attrs, "indices": 3, "mode": 1},
            {"attributes": attrs, "indices": 4, "mode": 4},
            {"attributes": attrs, "mode": 0},
        ],
        index_blocks=(
            np.array([0, 1, 1, 2], dtype=np.uint8),
            np.array([0, 2, 3], dtype=np.uint32),
        ),
    )
    assert poly.vertices.shape[0] == 4
    assert poly.vertex_attrs["normals"].shape == (4, 3)
    line, tri, vertex = (ELEMENT_TYPES[k] for k in ("line", "triangle", "vertex"))
    np.testing.assert_array_equal(
        poly.element_types, [line, line, tri, vertex, vertex, vertex, vertex]
    )
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 1, 2, 0, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(poly.offsets, [0, 2, 4, 7, 8, 9, 10, 11])


_LINE, _TRI = ELEMENT_TYPES["line"], ELEMENT_TYPES["triangle"]
_POLY_LINE, _VERTEX = ELEMENT_TYPES["poly_line"], ELEMENT_TYPES["vertex"]
_TRI_STRIP = ELEMENT_TYPES["triangle_strip"]


@pytest.mark.parametrize(
    ("mode", "indices", "connectivity", "offsets", "types"),
    [
        (0, [3, 1], [3, 1], [0, 1, 2], [_VERTEX] * 2),
        (1, [0, 1, 2, 3, 1], [0, 1, 2, 3], [0, 2, 4], [_LINE] * 2),
        (2, [0, 1, 2], [0, 1, 2, 0], [0, 4], [_POLY_LINE]),
        (2, [], [], [0], []),
        (3, [0, 1, 2], [0, 1, 2], [0, 3], [_POLY_LINE]),
        (4, [0, 1, 2, 3], [0, 1, 2], [0, 3], [_TRI]),
        (5, [0, 1, 3, 2], [0, 1, 3, 2], [0, 4], [_TRI_STRIP]),
        (6, [0, 1, 2, 3], [0, 1, 2, 0, 2, 3], [0, 3, 6], [_TRI] * 2),
        (6, [0, 1], [], [0], []),
    ],
)
def test_read_primitive_modes(
    tmp_path: Path,
    mode: int,
    indices: list[int],
    connectivity: list[int],
    offsets: list[int],
    types: list[int],
) -> None:
    block = np.array(indices or [0], dtype=np.uint16)
    poly = _read_primitives(
        tmp_path,
        [{"attributes": {"POSITION": 0}, "indices": 3, "mode": mode}],
        index_blocks=(block[: len(indices)] if indices else block[:0],),
    )
    np.testing.assert_array_equal(poly.connectivity, connectivity)
    np.testing.assert_array_equal(poly.offsets, offsets)
    np.testing.assert_array_equal(poly.element_types, types)


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (np.array([0, 1, 0xFFFFFFFF], dtype=np.uint32), "index 4294967295"),
        (np.array([0, 1, -1], dtype=np.int8), "not an unsigned integer"),
        (np.array([0.0, 1.0, 2.5], dtype=np.float32), "not an unsigned integer"),
    ],
)
def test_read_rejects_indices_that_name_no_vertex(
    tmp_path: Path, block: np.ndarray, match: str
) -> None:
    """An index past the vertex set, or one stored as a signed or float
    component, used to be cast to int32 and read as a vertex."""
    with pytest.raises(CodecError, match=match):
        _read_primitives(
            tmp_path,
            [{"attributes": {"POSITION": 0}, "indices": 3}],
            index_blocks=(block,),
        )


@pytest.mark.parametrize(
    ("primitive", "match"),
    [
        ({"attributes": {"POSITION": -1}}, r"POSITION names no accessor \(-1\)"),
        ({"attributes": {"POSITION": 99}}, r"POSITION names no accessor \(99\)"),
        ({"attributes": {"POSITION": 0, "NORMAL": True}}, "NORMAL names no"),
        ({"attributes": {"POSITION": 0}, "indices": 99}, "indices names no"),
        ({"attributes": ["POSITION"]}, "has no attributes object"),
        ("triangle", "primitive 0 is not an object"),
        ({"attributes": {"POSITION": 0}, "material": "a"}, "names no material"),
        ({"attributes": {"POSITION": 0}, "material": 3}, r"no material \(3\)"),
        ({"attributes": {"POSITION": 0}, "mode": True}, "mode True"),
    ],
)
def test_read_rejects_a_malformed_primitive(
    tmp_path: Path, primitive: Any, match: str
) -> None:
    """A negative accessor index used to read the last accessor, a material
    past the list and a boolean mode were kept; the others raised a raw
    IndexError, TypeError or ValueError."""
    with pytest.raises(CodecError, match=match):
        _read_primitives(tmp_path, [primitive])


def test_read_rejects_a_vertex_attribute_of_another_length(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="'normals' holds 3 entries"):
        _read_primitives(
            tmp_path, [{"attributes": {"POSITION": 0, "NORMAL": 2}}], n_normals=3
        )


def test_stack_elements_widens_indices_past_int32() -> None:
    conn = np.array([0, 1, 2], dtype=np.int64)
    offsets = np.array([0, 3], dtype=np.int64)
    types = np.array([_TRI], dtype=np.uint8)
    connectivity, offsets_out, _ = _stack_elements(
        [(conn, offsets, types)] * 2, vertex_bases=[0, 2**31], n_vertices=2**31 + 3
    )
    assert connectivity.dtype == np.int64
    assert offsets_out.dtype == np.int64
    np.testing.assert_array_equal(connectivity[3:], [2**31, 2**31 + 1, 2**31 + 2])
    small, small_offsets, _ = _stack_elements(
        [(conn, offsets, types)] * 2, vertex_bases=[0, 4], n_vertices=8
    )
    assert small.dtype == small_offsets.dtype == np.int32


def test_write_read_multi_material_mesh_keeps_its_vertex_count(
    tmp_path: Path,
) -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    n = np.tile([0.0, 0.0, 1.0], (4, 1))
    quad = make_polydata(v, [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))])
    quad = dataclasses.replace(
        quad,
        vertex_attrs={"normals": n},
        element_attrs={"material": np.array([0, 1], dtype=np.int32)},
    )
    scene = SceneData(
        meshes=(quad,),
        nodes=(SceneNode(mesh=0),),
        materials=(SceneMaterial(name="a"), SceneMaterial(name="b")),
        scenes=((0,),),
    )
    p = tmp_path / "quad.glb"
    for _ in range(3):
        write_scene(scene, p)
        scene = read_scene(p)
        poly = scene.meshes[0]
        np.testing.assert_allclose(poly.vertices, v)
        np.testing.assert_allclose(poly.vertex_attrs["normals"], n)
        np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 2, 3])
        np.testing.assert_array_equal(poly.element_attrs["material"], [0, 1])


def test_read_node_transform_baked(tmp_path: Path) -> None:
    verts = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [1.0, 0.0, 0.0]}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 38, "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    p = tmp_path / "transform.glb"
    p.write_bytes(_make_glb(gltf, bin_data))
    scene = read_scene(p)
    poly = scene.to_polydata()
    np.testing.assert_allclose(poly.vertices[:, 0], 1.0, atol=1e-6)


def test_read_scene_materials(tmp_path: Path) -> None:
    gltf = {
        "asset": {"version": "2.0"},
        "materials": [
            {"name": "red", "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}},
            {"name": "blue", "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 1, 1]}},
        ],
    }
    p = tmp_path / "mats.glb"
    p.write_bytes(_make_glb(gltf))
    scene = read_scene(p)
    assert len(scene.materials) == 2
    assert scene.materials[0].base_color == (1.0, 0.0, 0.0, 1.0)
    assert scene.materials[1].base_color == (0.0, 0.0, 1.0, 1.0)


def test_read_scene_hierarchy(tmp_path: Path) -> None:
    verts = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"mesh": 0, "children": [1], "translation": [1.0, 0.0, 0.0]},
            {"mesh": 0, "translation": [0.0, 2.0, 0.0]},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 38, "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    p = tmp_path / "hierarchy.glb"
    p.write_bytes(_make_glb(gltf, bin_data))
    scene = read_scene(p)
    assert scene.nodes[0].children == (1,)
    world = scene.world_transform(1)
    # Child world = parent T(1,0,0) @ child T(0,2,0) → translation (1,2,0).
    np.testing.assert_allclose(world[0, 3], 1.0, atol=1e-10)
    np.testing.assert_allclose(world[1, 3], 2.0, atol=1e-10)


def test_read_gltf_json(tmp_path: Path) -> None:
    verts = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_bytes = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 38, "byteLength": 6},
        ],
        "buffers": [{"uri": "mesh.bin", "byteLength": len(bin_bytes)}],
    }
    (tmp_path / "mesh.bin").write_bytes(bin_bytes)
    (tmp_path / "mesh.gltf").write_text(json.dumps(gltf), encoding="utf-8")
    scene = read_scene(tmp_path / "mesh.gltf")
    assert len(scene.meshes) == 1
    assert scene.meshes[0].vertices.shape == (3, 3)


def test_invalid_magic_raises(tmp_path: Path) -> None:
    bad = struct.pack("<III", 0xDEADBEEF, 2, 12)
    p = tmp_path / "bad.glb"
    p.write_bytes(bad)
    with pytest.raises(CodecError, match="magic"):
        read_scene(p)


def test_unsupported_version_raises(tmp_path: Path) -> None:
    # Build a GLB that has glTF magic but version 1 in the header.
    json_bytes = b'{"asset":{"version":"1.0"}}  '  # pad to 4-byte
    json_chunk = struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
    total = 12 + len(json_chunk)
    header = struct.pack("<III", _GLB_MAGIC, 2, total)
    p = tmp_path / "v1.glb"
    p.write_bytes(header + json_chunk)
    with pytest.raises(CodecError, match="version"):
        read_scene(p)


def test_triangle_fan_triangulated(tmp_path: Path) -> None:
    # 4 vertices, TRIANGLE_FAN (mode 6) → 2 triangles.
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.uint16)
    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 6}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 50, "byteLength": 8},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    p = tmp_path / "fan.glb"
    p.write_bytes(_make_glb(gltf, bin_data))
    scene = read_scene(p)
    poly = scene.meshes[0]
    assert len(poly.element_types) == 2
    assert all(c == ELEMENT_TYPES["triangle"] for c in poly.element_types)


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------


def test_write_triangle_glb(tmp_path: Path) -> None:
    poly = _triangle_poly()
    out = tmp_path / "out.glb"
    write(poly, out)
    scene = read_scene(out)
    assert len(scene.meshes) == 1
    back = scene.meshes[0]
    assert back.vertices.shape == (3, 3)
    np.testing.assert_allclose(back.vertices, poly.vertices, atol=1e-6)
    assert len(back.element_types) == 1


def test_write_read_roundtrip_normals(tmp_path: Path) -> None:
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": normals},
    )
    out = tmp_path / "normals.glb"
    write(poly, out)
    scene = read_scene(out)
    back = scene.meshes[0]
    assert "normals" in back.vertex_attrs
    np.testing.assert_allclose(back.vertex_attrs["normals"], normals, atol=1e-6)


def test_write_quad_fan_triangulated(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        [("quad", np.array([[0, 1, 2, 3]]))],
    )
    out = tmp_path / "quad.glb"
    write(poly, out)
    scene = read_scene(out)
    back = scene.meshes[0]
    # One quad → two triangles after fan-triangulation.
    assert len(back.element_types) == 2
    assert all(c == ELEMENT_TYPES["triangle"] for c in back.element_types)


def test_write_volume_elements_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64
        ),
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    out = tmp_path / "vol.glb"
    with pytest.warns(UserWarning, match="volume"):
        write(poly, out)
    scene = read_scene(out)
    back = scene.meshes[0]
    assert all(c == ELEMENT_TYPES["triangle"] for c in back.element_types)


def test_write_gltf_separate_bin(tmp_path: Path) -> None:
    poly = _triangle_poly()
    out = tmp_path / "mesh.gltf"
    write(poly, out, binary=False)
    assert out.exists()
    assert (tmp_path / "mesh.bin").exists()
    scene = read_scene(out)
    assert len(scene.meshes) == 1


def test_write_scene_preserves_hierarchy(tmp_path: Path) -> None:
    mesh_poly = _triangle_poly()
    t = np.eye(4, dtype=np.float64)
    t[1, 3] = 2.0  # child translation (0, 2, 0)
    scene = SceneData(
        meshes=(mesh_poly, mesh_poly),
        nodes=(
            SceneNode(name="parent", mesh=0, children=(1,)),
            SceneNode(name="child", mesh=1, matrix=t),
        ),
        scenes=((0,),),
        active_scene=0,
    )
    out = tmp_path / "scene.glb"
    write_scene(scene, out)
    back = read_scene(out)
    assert back.nodes[0].children == (1,)
    np.testing.assert_allclose(back.nodes[1].matrix[1, 3], 2.0, atol=1e-10)


def test_write_scene_materials(tmp_path: Path) -> None:
    mesh_poly = _triangle_poly()
    mat0 = SceneMaterial(name="red", base_color=(1.0, 0.0, 0.0, 1.0))
    mat1 = SceneMaterial(name="blue", base_color=(0.0, 0.0, 1.0, 1.0))
    scene = SceneData(
        meshes=(mesh_poly,),
        nodes=(SceneNode(mesh=0),),
        materials=(mat0, mat1),
        scenes=((0,),),
        active_scene=0,
    )
    out = tmp_path / "mats.glb"
    write_scene(scene, out)
    back = read_scene(out)
    assert len(back.materials) == 2
    np.testing.assert_allclose(
        back.materials[0].base_color, (1.0, 0.0, 0.0, 1.0), atol=1e-6
    )
    np.testing.assert_allclose(
        back.materials[1].base_color, (0.0, 0.0, 1.0, 1.0), atol=1e-6
    )


def test_read_mesh_only_no_nodes(tmp_path: Path) -> None:
    verts = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_data = verts.tobytes() + b"\x00\x00" + indices.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 38, "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    p = tmp_path / "meshonly.glb"
    p.write_bytes(_make_glb(gltf, bin_data))
    scene = read_scene(p)
    assert len(scene.meshes) == 1
    assert len(scene.nodes) == 1  # synthesized
    assert scene.nodes[0].mesh == 0
    with pytest.warns(UserWarning, match="scene format"):
        poly = read(p)
    assert poly.vertices.shape == (3, 3)


# ---------------------------------------------------------------------------
# Animation tests
# ---------------------------------------------------------------------------


def _make_animation_glb(
    times: list[float],
    values: list[list[float]],
    path: str,
    node_idx: int = 0,
    interpolation: str = "LINEAR",
) -> bytes:
    """Build a GLB with one mesh and one animation channel targeting a node.

    Parameters
    ----------
    times
        Keyframe timestamps in seconds.
    values
        Keyframe values; shape depends on *path*
        (translation/scale → vec3, rotation → vec4).
    path
        glTF animation target path: ``"translation"``, ``"rotation"``,
        or ``"scale"``.
    node_idx
        Index of the target node.
    interpolation
        Sampler interpolation: ``"LINEAR"``, ``"STEP"``, or
        ``"CUBICSPLINE"``.
    """
    verts = np.zeros((3, 3), dtype=np.float32)
    tri_indices = np.array([0, 1, 2], dtype=np.uint16)
    times_arr = np.array(times, dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)

    # Binary layout:  verts | pad | tri_indices | times | values
    vert_bytes = verts.tobytes()
    pad = b"\x00\x00"
    idx_bytes = tri_indices.tobytes()
    time_bytes = times_arr.tobytes()
    val_bytes = values_arr.tobytes()

    # Pad time_bytes and val_bytes to 4-byte alignment.
    def _pad4(b: bytes) -> bytes:
        rem = len(b) % 4
        return b + b"\x00" * (4 - rem) if rem else b

    time_bytes_p = _pad4(time_bytes)
    val_bytes_p = _pad4(val_bytes)

    bin_data = vert_bytes + pad + idx_bytes + time_bytes_p + val_bytes_p

    off_vert = 0
    off_idx = len(vert_bytes) + len(pad)
    off_time = off_idx + len(idx_bytes)
    off_val = off_time + len(time_bytes_p)

    acc_type = "VEC4" if path == "rotation" else "VEC3"
    n_kf = len(times)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": n_kf, "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5126, "count": n_kf, "type": acc_type},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_vert, "byteLength": len(vert_bytes)},
            {"buffer": 0, "byteOffset": off_idx, "byteLength": len(idx_bytes)},
            {"buffer": 0, "byteOffset": off_time, "byteLength": len(time_bytes)},
            {"buffer": 0, "byteOffset": off_val, "byteLength": len(val_bytes)},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
        "animations": [
            {
                "name": "test_anim",
                "channels": [
                    {"sampler": 0, "target": {"node": node_idx, "path": path}}
                ],
                "samplers": [{"input": 2, "output": 3, "interpolation": interpolation}],
            }
        ],
    }
    return _make_glb(gltf, bin_data)


def test_animation_decoded_on_read(tmp_path: Path) -> None:
    times = [0.0, 1.0, 2.0]
    values = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    p = tmp_path / "anim.glb"
    p.write_bytes(_make_animation_glb(times, values, "translation"))
    scene = read_scene(p)
    anims = scene.global_attrs.get("animations", [])
    assert len(anims) == 1
    assert anims[0]["name"] == "test_anim"
    assert len(anims[0]["channels"]) == 1
    assert anims[0]["channels"][0]["target"]["path"] == "translation"
    s = anims[0]["samplers"][0]
    assert "times" in s and "values" in s
    assert isinstance(s["times"], np.ndarray)
    assert isinstance(s["values"], np.ndarray)
    np.testing.assert_allclose(s["times"], times, atol=1e-6)
    np.testing.assert_allclose(s["values"], values, atol=1e-6)


def test_animation_times_values_shapes(tmp_path: Path) -> None:
    times = [0.0, 0.5, 1.0, 1.5]
    values = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.707, 0.0, 0.707],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.707, 0.0, -0.707],
    ]
    p = tmp_path / "rot.glb"
    p.write_bytes(_make_animation_glb(times, values, "rotation"))
    scene = read_scene(p)
    s = scene.global_attrs["animations"][0]["samplers"][0]
    assert s["times"].shape == (4,)
    assert s["values"].shape == (4, 4)  # VEC4 quaternions
    assert s["interpolation"] == "LINEAR"


def test_animation_step_interpolation(tmp_path: Path) -> None:
    times = [0.0, 1.0]
    values = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    p = tmp_path / "step.glb"
    p.write_bytes(
        _make_animation_glb(times, values, "translation", interpolation="STEP")
    )
    scene = read_scene(p)
    s = scene.global_attrs["animations"][0]["samplers"][0]
    assert s["interpolation"] == "STEP"
    np.testing.assert_allclose(s["times"], times, atol=1e-6)


def test_animation_scale_channel(tmp_path: Path) -> None:
    times = [0.0, 1.0]
    values = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    p = tmp_path / "scale.glb"
    p.write_bytes(_make_animation_glb(times, values, "scale"))
    scene = read_scene(p)
    s = scene.global_attrs["animations"][0]["samplers"][0]
    assert s["values"].shape == (2, 3)  # VEC3 scale
    np.testing.assert_allclose(s["values"], values, atol=1e-6)


def test_animation_roundtrip_write_scene(tmp_path: Path) -> None:
    times = [0.0, 1.0, 2.0]
    values = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    p_in = tmp_path / "anim_in.glb"
    p_in.write_bytes(_make_animation_glb(times, values, "translation"))
    scene = read_scene(p_in)
    p_out = tmp_path / "anim_out.glb"
    write_scene(scene, p_out)
    scene2 = read_scene(p_out)
    anims = scene2.global_attrs.get("animations", [])
    assert len(anims) == 1
    s = anims[0]["samplers"][0]
    np.testing.assert_allclose(s["times"], times, atol=1e-5)
    np.testing.assert_allclose(s["values"], values, atol=1e-5)
    assert anims[0]["channels"][0]["target"]["path"] == "translation"


def test_animation_name_preserved(tmp_path: Path) -> None:
    p = tmp_path / "named.glb"
    p.write_bytes(
        _make_animation_glb([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], "translation")
    )
    scene = read_scene(p)
    assert scene.global_attrs["animations"][0]["name"] == "test_anim"
    p_out = tmp_path / "named_out.glb"
    write_scene(scene, p_out)
    scene2 = read_scene(p_out)
    assert scene2.global_attrs["animations"][0]["name"] == "test_anim"


def test_no_animation_field_when_absent(tmp_path: Path) -> None:
    p = tmp_path / "no_anim.glb"
    p.write_bytes(_make_glb({"asset": {"version": "2.0"}}))
    scene = read_scene(p)
    assert "animations" not in scene.global_attrs


# ---------------------------------------------------------------------------
# Primitive-mode tests (modes 0, 1, 3, 5)
# ---------------------------------------------------------------------------


def test_primitive_mode_points(tmp_path: Path) -> None:
    """Round-trip POINTS (mode 0): all element types are vertex codes."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("vertex", np.array([[0], [1], [2]]))],
    )
    out = tmp_path / "pts.glb"
    write(poly, out)
    back = read_scene(out).meshes[0]
    assert len(back.element_types) == 3
    assert all(c == ELEMENT_TYPES["vertex"] for c in back.element_types)


def test_primitive_mode_lines(tmp_path: Path) -> None:
    """Round-trip LINES (mode 1): all element types are line codes."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1], [2, 3]]))],
    )
    out = tmp_path / "lines.glb"
    write(poly, out)
    back = read_scene(out).meshes[0]
    assert len(back.element_types) == 2
    assert all(c == ELEMENT_TYPES["line"] for c in back.element_types)


def test_primitive_mode_poly_line(tmp_path: Path) -> None:
    """Round-trip poly_line: written as LINE_STRIP (mode 3), one element on read."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("poly_line", np.array([[0, 1, 2, 3]]))],
    )
    out = tmp_path / "polyline.glb"
    write(poly, out)
    back = read_scene(out).meshes[0]
    assert len(back.element_types) == 1
    assert back.element_types[0] == ELEMENT_TYPES["poly_line"]


def test_triangle_strip_two_primitives(tmp_path: Path) -> None:
    """Two triangle_strip elements sharing a material produce two glTF primitives
    each with mode 5 (strips are never merged across element boundaries)."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("triangle_strip", np.array([[0, 1, 2, 3]])),
            ("triangle_strip", np.array([[2, 3, 4, 5]])),
        ],
    )
    out = tmp_path / "strips.glb"
    write(poly, out)
    gltf, _ = _parse_glb(out.read_bytes())
    primitives = gltf["meshes"][0]["primitives"]
    assert len(primitives) == 2
    assert all(p["mode"] == 5 for p in primitives)


# ---------------------------------------------------------------------------
# Accessor error paths
# ---------------------------------------------------------------------------


def test_sparse_accessor_raises(tmp_path: Path) -> None:
    """A GLB accessor with 'sparse' raises CodecError on read."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    bin_data = verts.tobytes()
    gltf_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(bin_data)}],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "sparse": {"count": 1, "indices": {}, "values": {}},
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "scene": 0,
    }
    p = tmp_path / "sparse.glb"
    p.write_bytes(_make_glb(gltf_json, bin_data))
    with pytest.raises(CodecError, match="sparse"):
        read_scene(p)


def test_normalised_ubyte_color(tmp_path: Path) -> None:
    """UBYTE COLOR_0 accessor with normalized:True decodes to float in [0, 1]."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    # verts: 3 × 3 × 4 = 36 bytes; colors: 3 × 3 × 1 = 9 bytes
    bin_data = verts.tobytes() + colors.tobytes()
    gltf_json = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 9},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {
                "bufferView": 1,
                "componentType": 5121,
                "count": 3,
                "type": "VEC3",
                "normalized": True,
            },
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}],
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "scene": 0,
    }
    p = tmp_path / "colors.glb"
    p.write_bytes(_make_glb(gltf_json, bin_data))
    poly = read_scene(p).meshes[0]
    colors_out = poly.vertex_attrs["colors"]
    assert np.issubdtype(colors_out.dtype, np.floating)
    assert float(colors_out.max()) <= 1.0


# ---------------------------------------------------------------------------
# Material remapping
# ---------------------------------------------------------------------------


def test_noncontiguous_material_remap(tmp_path: Path) -> None:
    """Material indices [0, 5] are remapped to dense [0, 1] in the output GLB."""
    verts = np.zeros((6, 3), dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("triangle", np.array([[3, 4, 5]])),
        ],
        element_attrs={"material": np.array([0, 5], dtype=np.int32)},
    )
    out = tmp_path / "mat.glb"
    write(poly, out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert len(gltf.get("materials", [])) == 2
    prim_mats = sorted(p["material"] for p in gltf["meshes"][0]["primitives"])
    assert prim_mats == [0, 1]


# ---------------------------------------------------------------------------
# write_scene error paths
# ---------------------------------------------------------------------------


def test_write_scene_volume_only_raises(tmp_path: Path) -> None:
    """write_scene raises CodecError when a mesh yields no writable primitives."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    scene = SceneData(
        meshes=(poly,),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
    )
    out = tmp_path / "vol_scene.glb"
    with pytest.warns(UserWarning, match="volume"), pytest.raises(CodecError):
        write_scene(scene, out)


# ---------------------------------------------------------------------------
# Animation skipped-channel sampler pairing
# ---------------------------------------------------------------------------


def test_animation_skipped_channel_sampler_index(tmp_path: Path) -> None:
    """A channel referencing an out-of-range sampler is dropped; the remaining
    valid channel is assigned sampler index 0 without shift."""
    times = np.array([0.0, 1.0], dtype=np.float32)
    values = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    sampler = {"times": times, "values": values, "interpolation": "LINEAR"}
    scene = SceneData(
        meshes=(_triangle_poly(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
        global_attrs={
            "animations": [
                {
                    "samplers": [sampler],
                    "channels": [
                        # channel 0: sampler index 99 is out of range → skipped
                        {"sampler": 99, "target": {"node": 0, "path": "translation"}},
                        # channel 1: sampler index 0 is valid → written as sampler 0
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                    ],
                }
            ]
        },
    )
    out = tmp_path / "skip_anim.glb"
    write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    anim = gltf["animations"][0]
    assert len(anim["channels"]) == 1
    assert anim["channels"][0]["sampler"] == 0
    assert len(anim["samplers"]) == 1


def test_placeholder_texture_source_omitted(tmp_path: Path) -> None:
    """SceneTexture(image=-1) must not emit 'source': -1 in the written GLB."""
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
        textures=(SceneTexture(image=-1),),
    )
    out = tmp_path / "placeholder.glb"
    write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    tex = gltf["textures"][0]
    assert "source" not in tex, f"expected no 'source' key, got {tex!r}"


@pytest.mark.parametrize(
    ("target", "interp", "match"),
    [
        ({"node": 0, "path": "matrix"}, "LINEAR", "path 'matrix'"),
        ({"node": 0, "path": "translation"}, "BEZIER", "interpolates 'BEZIER'"),
        (
            {"node": 0, "path": "rotation", "sid": "rz", "member": "ANGLE"},
            "LINEAR",
            "'member', 'sid'",
        ),
    ],
)
def test_foreign_animation_channel_skipped(
    tmp_path: Path, target: dict, interp: str, match: str
) -> None:
    """A channel glTF cannot hold is skipped, and an animation left without
    channels is not written, since glTF requires at least one."""
    sampler = {
        "times": np.array([0.0, 1.0]),
        "values": np.array([0.0, 90.0]),
        "interpolation": interp,
    }
    scene = SceneData(
        meshes=(_triangle_poly(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={
            "animations": [
                {"samplers": [sampler], "channels": [{"sampler": 0, "target": target}]}
            ]
        },
    )
    out = tmp_path / "foreign.glb"
    with pytest.warns(UserWarning, match=match):
        write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "animations" not in gltf


def _quat(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """Return the ``(x, y, z, w)`` quaternion turning ``degrees`` about ``axis``."""
    half = np.radians(degrees) / 2
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.r_[unit * np.sin(half), np.cos(half)]


def _trs_matrix(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Return the 4x4 row-major matrix of translation, quaternion and scale."""
    return matrix_of_trs(translation=t, rotation=q, scale=s)


def _matrix_scene(keys: np.ndarray, *, interp: str = "LINEAR", extras=None, more=()):
    """One node animated by a matrix channel of ``keys``, row-major (n, 16)."""
    samplers = [
        {
            "times": np.arange(len(keys), dtype=np.float64),
            "values": keys,
            "interpolation": interp,
        }
    ]
    channels = [
        {
            "sampler": 0,
            "target": {"node": 0, "path": "matrix", "sid": "transform", "member": None},
        }
    ]
    for target, sampler in more:
        channels.append({"sampler": len(samplers), "target": target})
        samplers.append(sampler)
    return SceneData(
        meshes=(_triangle_poly(),),
        nodes=(SceneNode(mesh=0, extras=extras or {}),),
        scenes=((0,),),
        global_attrs={"animations": [{"channels": channels, "samplers": samplers}]},
    )


def _tracks(scene: SceneData) -> dict[str, dict]:
    """Return the first animation's samplers keyed by their channel's path."""
    anim = scene.global_attrs["animations"][0]
    return {
        c["target"]["path"]: anim["samplers"][c["sampler"]] for c in anim["channels"]
    }


def test_matrix_channel_written_as_trs(tmp_path: Path) -> None:
    """A matrix channel animating a node's whole transform is split into
    translation, rotation and scale channels, and the node written as TRS."""
    t = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    q = np.array([_quat((0, 0, 1), 0), _quat((1, 1, 0), 120), _quat((0, 1, 0), 250)])
    sc = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 1.0, 3.0]])
    keys = np.stack([_trs_matrix(*k).ravel() for k in zip(t, q, sc, strict=True)])
    out = tmp_path / "split.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(_matrix_scene(keys), out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "matrix" not in gltf["nodes"][0]
    tracks = _tracks(read_scene(out))
    assert sorted(tracks) == ["rotation", "scale", "translation"]
    np.testing.assert_allclose(tracks["translation"]["values"], t, atol=1e-6)
    np.testing.assert_allclose(tracks["scale"]["values"], sc, atol=1e-5)
    rot = tracks["rotation"]["values"]
    # Same hemisphere key to key, and each the same rotation as its source.
    assert (np.sum(rot[1:] * rot[:-1], axis=1) >= 0).all()
    np.testing.assert_allclose(np.abs(np.sum(rot * q, axis=1)), 1.0, atol=1e-6)


def test_gltf_animation_survives_a_collada_round_trip(tmp_path: Path) -> None:
    """glTF TRS channels baked into COLLADA matrix keys come back as TRS
    channels holding the original values at the original key times, with
    time keys that stay strictly increasing in float32."""
    rotation = np.array([_quat((0, 0, 1), 0), _quat((0, 0, 1), 170)])
    translation = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scene = SceneData(
        meshes=(_triangle_poly(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={
            "animations": [
                {
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
                        {"sampler": 1, "target": {"node": 0, "path": "translation"}},
                    ],
                    "samplers": [
                        {
                            "times": np.array([0.0, 1.0]),
                            "values": rotation,
                            "interpolation": "LINEAR",
                        },
                        {
                            "times": np.array([0.0, 0.5, 1.0]),
                            "values": translation,
                            "interpolation": "STEP",
                        },
                    ],
                }
            ]
        },
    )
    dae = tmp_path / "mid.dae"
    glb = tmp_path / "back.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _collada.write_scene(scene, dae)
        write_scene(_collada.read_scene(dae), glb)
    tracks = _tracks(read_scene(glb))
    assert sorted(tracks) == ["rotation", "scale", "translation"]
    for track in tracks.values():
        assert (np.diff(track["times"]) > 0).all()

    def at(path: str, time: float) -> np.ndarray:
        track = tracks[path]
        return track["values"][np.flatnonzero(track["times"] == np.float32(time))[-1]]

    for time, value in zip((0.0, 0.5, 1.0), translation, strict=True):
        np.testing.assert_allclose(at("translation", time), value, atol=1e-6)
    for time, value in zip((0.0, 1.0), rotation, strict=True):
        assert abs(np.dot(at("rotation", time), value)) == pytest.approx(1.0, abs=1e-6)
    # The translation still steps at 0.5: the key just before holds 0.
    times = tracks["translation"]["times"]
    before = int(np.flatnonzero(times == np.float32(0.5))[0]) - 1
    np.testing.assert_allclose(tracks["translation"]["values"][before], 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"interp": "BEZIER"}, "not written"),
        (
            {
                "extras": {
                    "transforms": [
                        {"kind": "translate", "sid": "t", "values": [0, 0, 0]},
                        {
                            "kind": "matrix",
                            "sid": "transform",
                            "values": np.eye(4).ravel().tolist(),
                        },
                    ]
                }
            },
            "not written",
        ),
        (
            {
                "extras": {
                    "transforms": [
                        {
                            "kind": "matrix",
                            "sid": "other",
                            "values": np.eye(4).ravel().tolist(),
                        }
                    ]
                }
            },
            "not written",
        ),
    ],
)
def test_matrix_channel_not_the_whole_transform_is_skipped(
    tmp_path: Path, change: dict, match: str
) -> None:
    """A matrix channel that is not the node's whole transform, or whose
    interpolation glTF cannot blend a TRS by, is still skipped."""
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    out = tmp_path / "skip.glb"
    with pytest.warns(UserWarning, match=match):
        write_scene(_matrix_scene(keys, **change), out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "animations" not in gltf


def test_matrix_channel_clashing_with_a_trs_channel_is_skipped(tmp_path: Path) -> None:
    """Two channels of one animation must not target the same node path."""
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    native = {
        "times": np.array([0.0, 1.0]),
        "values": np.zeros((2, 3)),
        "interpolation": "LINEAR",
    }
    scene = _matrix_scene(keys, more=[({"node": 0, "path": "translation"}, native)])
    out = tmp_path / "clash.glb"
    with pytest.warns(UserWarning, match="already animates"):
        write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    channels = gltf["animations"][0]["channels"]
    assert [c["target"]["path"] for c in channels] == ["translation"]


def test_matrix_channel_key_with_a_shear_warns(tmp_path: Path) -> None:
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    keys[1, 1] = 0.5
    with pytest.warns(UserWarning, match="shear"):
        write_scene(_matrix_scene(keys), tmp_path / "shear.glb")


def test_matrix_channel_split_leaves_the_scene_alone(tmp_path: Path) -> None:
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    scene = _matrix_scene(keys)
    write_scene(scene, tmp_path / "same.glb")
    anim = scene.global_attrs["animations"][0]
    assert len(anim["channels"]) == 1 and len(anim["samplers"]) == 1
    assert anim["channels"][0]["target"]["path"] == "matrix"


def test_increasing_float32_moves_a_hold_key_below_its_step() -> None:
    step = 0.1
    times = np.array([0.0, np.nextafter(step, -np.inf), step, 1.0])
    out = _increasing_float32(times)
    assert out.dtype == np.float32
    assert (np.diff(out) > 0).all()
    assert out[2] == np.float32(step)


def test_animated_node_with_a_mirrored_rest_matrix_is_written(tmp_path: Path) -> None:
    """The rest pose of an animated node splits as its keys do, reflection
    included, instead of refusing a negative determinant."""
    mirror = np.diag([1.0, -1.0, 1.0, 1.0])
    keys = np.stack([np.eye(4).ravel(), mirror.ravel()])
    scene = _matrix_scene(keys)
    scene = dataclasses.replace(
        scene, nodes=(dataclasses.replace(scene.nodes[0], matrix=mirror),)
    )
    out = tmp_path / "mirror.glb"
    with pytest.warns(UserWarning, match="handedness"):
        write_scene(scene, out)
    node = _parse_glb(out.read_bytes())[0]["nodes"][0]
    rebuilt = _trs_matrix(
        np.asarray(node["translation"]),
        np.asarray(node["rotation"]),
        np.asarray(node["scale"]),
    )
    np.testing.assert_allclose(rebuilt, mirror, atol=1e-12)


def test_matrix_keys_printed_to_six_digits_are_not_a_shear(tmp_path: Path) -> None:
    """Six significant digits, as COLLADA exporters print, rebuild the key."""
    q = [_quat((1, 2, 3), a) for a in (0, 37, 81)]
    keys = np.stack([_trs_matrix(np.zeros(3), k, np.ones(3)).ravel() for k in q])
    keys = np.vectorize(lambda v: float(f"{v:.6g}"))(keys)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(_matrix_scene(keys), tmp_path / "six.glb")


def test_native_channel_times_stay_increasing_in_float32(tmp_path: Path) -> None:
    """Times of any channel, not only a split matrix one, that float32 would
    collapse are kept strictly increasing."""
    times = np.array([0.0, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 2.0])
    scene = _matrix_scene(np.eye(4).ravel()[None, :])
    scene = dataclasses.replace(
        scene,
        global_attrs={
            "animations": [
                {
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                    "samplers": [
                        {
                            "times": times,
                            "values": np.zeros((5, 3)),
                            "interpolation": "STEP",
                        }
                    ],
                }
            ]
        },
    )
    out = tmp_path / "times.glb"
    write_scene(scene, out)
    back = _tracks(read_scene(out))["translation"]["times"]
    assert (np.diff(back.astype(np.float32)) > 0).all()


def test_increasing_float32_leaves_unordered_times_alone() -> None:
    times = np.array([2.0, 1.0, 0.0])
    np.testing.assert_array_equal(_increasing_float32(times), times)


def test_animation_channel_warning_names_the_caller(tmp_path: Path) -> None:
    """The warning points at the line that wrote, however deep the codec is."""
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    scene = _matrix_scene(keys, interp="BEZIER")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_scene(scene, tmp_path / "caller.glb")
    assert [w.filename for w in caught] == [__file__]


def _skinned_glb(*, ibm_count: int = 2) -> tuple[bytes, np.ndarray]:
    """Return a skinned triangle GLB and its row-major inverse bind matrices.

    The second joint's inverse bind matrix translates by ``-1`` along x,
    which only a column-major read puts in ``[0, 3]``.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    joints = np.array([[0, 1, 0, 0]] * 3, dtype=np.uint8)
    weights = np.array([[0.5, 0.5, 0, 0]] * 3, dtype=np.float32)
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[1, 0, 3] = -1.0
    stored = np.tile(ibm.transpose(0, 2, 1), (ibm_count, 1, 1))[:ibm_count]
    blobs = [verts.tobytes(), joints.tobytes(), weights.tobytes()]
    blobs.append(stored.astype(np.float32).tobytes())
    views, offset = [], 0
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        offset += len(blob)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"mesh": 0, "skin": 0},
            {"name": "hip", "children": [2]},
            {"name": "knee", "translation": [1.0, 0.0, 0.0]},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "mode": 4,
                    }
                ]
            }
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": 3}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": ibm_count,
                "type": "MAT4",
            },
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    return _make_glb(gltf, b"".join(blobs)), ibm


def test_read_skin_inverse_bind_matrices_decoded_row_major(tmp_path: Path) -> None:
    data, ibm = _skinned_glb()
    p = tmp_path / "skin.glb"
    p.write_bytes(data)
    skin = read_scene(p).global_attrs["skins"][0]
    assert skin["inverseBindMatrices"] == 3 and skin["joints"] == [1, 2]
    assert skin["inverse_bind_matrices"].dtype == np.float64
    np.testing.assert_array_equal(skin["inverse_bind_matrices"], ibm)


def test_read_skin_inverse_bind_matrices_short_accessor_warns(
    tmp_path: Path,
) -> None:
    data, _ = _skinned_glb(ibm_count=1)
    p = tmp_path / "short.glb"
    p.write_bytes(data)
    with pytest.warns(UserWarning, match="not one MAT4 per joint"):
        skin = read_scene(p).global_attrs["skins"][0]
    assert "inverse_bind_matrices" not in skin


@pytest.mark.parametrize(
    ("skin", "match"),
    [
        (5, "not an object"),
        ({"joints": 5, "inverseBindMatrices": 0}, "joints is not a list"),
        ({"joints": [0], "inverseBindMatrices": 9}, "names no accessor"),
        ({"joints": [0], "inverseBindMatrices": -1}, "names no accessor"),
        ({"joints": [0], "inverseBindMatrices": "0"}, "names no accessor"),
    ],
)
def test_malformed_skin_is_kept_with_a_warning(skin: Any, match: str) -> None:
    gltf = {"accessors": [{"count": 1, "componentType": 5126, "type": "MAT4"}]}
    with pytest.warns(UserWarning, match=match):
        (out,) = _decode_skins(gltf, [skin], [])
    assert out == skin


_MAT4_VIEW = [{"buffer": 0, "byteLength": 64}]


@pytest.mark.parametrize(
    ("accessor", "buffers", "match"),
    [
        (
            {"bufferView": 0, "count": 1, "componentType": 5126, "type": "MAT4"},
            [b"\0" * 8],
            "cannot be read",
        ),
        (
            {
                "count": 1,
                "componentType": 5126,
                "type": "MAT4",
                "sparse": {"count": 1},
            },
            [],
            "cannot be read",
        ),
        ({"componentType": 5126, "type": "MAT4"}, [], "cannot be read"),
        (
            {"bufferView": 7, "count": 1, "componentType": 5126, "type": "MAT4"},
            [b"\0" * 64],
            "cannot be read",
        ),
        (
            {"bufferView": 0, "count": 1, "componentType": 5121, "type": "MAT4"},
            [b"\1" * 64],
            "not a float MAT4",
        ),
        (
            {"bufferView": 0, "count": 4, "componentType": 5126, "type": "VEC4"},
            [b"\0" * 64],
            "not a float MAT4",
        ),
        (5, [], "not a float MAT4"),
    ],
)
def test_skin_with_an_undecodable_accessor_is_kept_with_a_warning(
    accessor: Any, buffers: list, match: str
) -> None:
    """An inverse bind accessor that cannot be decoded does not fail the read."""
    gltf = {"accessors": [accessor], "bufferViews": _MAT4_VIEW}
    skin = {"joints": [0], "inverseBindMatrices": 0}
    with pytest.warns(UserWarning, match=match):
        (out,) = _decode_skins(gltf, [skin], buffers)
    assert out == skin


def test_read_scene_survives_a_short_inverse_bind_buffer(tmp_path: Path) -> None:
    data, _ = _skinned_glb()
    gltf, bin_chunk = _parse_glb(data)
    gltf["accessors"][3]["count"] = 50
    p = tmp_path / "short_buffer.glb"
    p.write_bytes(_make_glb(gltf, bin_chunk))
    with pytest.warns(UserWarning, match="cannot be read"):
        scene = read_scene(p)
    assert "inverse_bind_matrices" not in scene.global_attrs["skins"][0]
    assert len(scene.meshes) == 1


@pytest.mark.parametrize("skin", [{"inverseBindMatrices": 0}, {"joints": []}])
def test_skin_without_joints_is_not_decoded(skin: dict) -> None:
    gltf = {
        "accessors": [
            {"bufferView": 0, "count": 1, "componentType": 5126, "type": "MAT4"}
        ],
        "bufferViews": _MAT4_VIEW,
    }
    skin = {**skin, "inverseBindMatrices": 0}
    with pytest.warns(UserWarning, match="has no joints"):
        (out,) = _decode_skins(gltf, [skin], [b"\0" * 64])
    assert out == skin


def test_skin_matrices_beyond_the_joints_are_left_out(tmp_path: Path) -> None:
    data, ibm = _skinned_glb(ibm_count=3)
    p = tmp_path / "surplus.glb"
    p.write_bytes(data)
    skin = read_scene(p).global_attrs["skins"][0]
    np.testing.assert_array_equal(skin["inverse_bind_matrices"], ibm)


def test_skins_that_are_not_a_list_are_kept_with_a_warning() -> None:
    skins = {"joints": [0]}
    with pytest.warns(UserWarning, match="skins is not a list"):
        assert _decode_skins({}, skins, []) is skins


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _skinned_scene(
    skin: Any,
    *,
    joints: Any = None,
    weights: Any = None,
    node_skin: Any = 0,
    mesh: int | None = 0,
    omit: tuple[str, ...] = (),
    extra_attrs: dict[str, Any] | None = None,
) -> SceneData:
    """Return a triangle skinned by ``skin`` over a two-bone chain.

    Node 0 holds the mesh, node 1 is the hip and node 2 the knee below it.
    """
    if joints is None:
        joints = np.array([[0, 1, 0, 0]] * 3, dtype=np.uint8)
    if weights is None:
        weights = np.array([[0.5, 0.5, 0, 0]] * 3, dtype=np.float32)
    influences = {"joints": joints, "weights": weights, **(extra_attrs or {})}
    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={k: v for k, v in influences.items() if k not in omit},
    )
    return SceneData(
        meshes=(poly,),
        nodes=(
            SceneNode(mesh=mesh, extras={"skin": node_skin}),
            SceneNode(name="hip", children=(2,)),
            SceneNode(name="knee", matrix=_translation(1.0, 0.0, 0.0)),
        ),
        scenes=((0, 1),),
        global_attrs={"skins": [skin]},
    )


def _written_json(scene: SceneData, tmp_path: Path) -> dict:
    p = tmp_path / "skin.gltf"
    write_scene(scene, p, binary=False)
    return json.loads(p.read_text())


def test_skin_round_trips(tmp_path: Path) -> None:
    data, ibm = _skinned_glb()
    src = tmp_path / "in.glb"
    src.write_bytes(data)
    scene = read_scene(src)
    out = tmp_path / "out.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out)
    (skin,) = back.global_attrs["skins"]
    assert skin["joints"] == [1, 2]
    np.testing.assert_allclose(skin["inverse_bind_matrices"], ibm)
    assert back.nodes[0].extras["skin"] == 0
    for key in ("joints", "weights"):
        np.testing.assert_array_equal(
            back.meshes[0].vertex_attrs[key], scene.meshes[0].vertex_attrs[key]
        )


def test_skin_is_written_with_its_name_skeleton_and_matrices(tmp_path: Path) -> None:
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[1, 0, 3] = -1.0
    skin = {"name": "rig", "joints": [1, 2], "skeleton": 1}
    gltf = _written_json(
        _skinned_scene({**skin, "inverse_bind_matrices": ibm}), tmp_path
    )
    (out,) = gltf["skins"]
    acc = gltf["accessors"][out.pop("inverseBindMatrices")]
    assert out == skin
    assert (acc["type"], acc["componentType"], acc["count"]) == ("MAT4", 5126, 2)
    assert "target" not in gltf["bufferViews"][acc["bufferView"]]
    assert gltf["nodes"][0]["skin"] == 0


def test_skin_without_matrices_writes_no_accessor(tmp_path: Path) -> None:
    gltf = _written_json(_skinned_scene({"joints": np.array([1, 2])}), tmp_path)
    assert gltf["skins"] == [{"joints": [1, 2]}]


def test_bind_shape_matrix_is_folded_into_the_inverse_bind_matrices(
    tmp_path: Path,
) -> None:
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[1, 0, 3] = -1.0
    bsm = _translation(0.0, 0.0, 2.0)
    skin = {"joints": [1, 2], "inverse_bind_matrices": ibm, "bind_shape_matrix": bsm}
    out = tmp_path / "bsm.glb"
    write_scene(_skinned_scene(skin), out)
    (back,) = read_scene(out).global_attrs["skins"]
    assert "bind_shape_matrix" not in back
    np.testing.assert_allclose(back["inverse_bind_matrices"], ibm @ bsm)


@pytest.mark.parametrize(
    ("skin", "match"),
    [
        ("rig", "is not a dict"),
        ({"joints": []}, "are not distinct node indices"),
        ({"joints": [1, 9]}, "are not distinct node indices"),
        ({"joints": [1, 1]}, "are not distinct node indices"),
        ({"joints": [1, True]}, "are not distinct node indices"),
        ({"joints": [1, 2], "inverseBindMatrices": 3}, "never decoded"),
        ({"joints": [1, 2], "inverse_bind_matrices": np.eye(4)}, "one per joint"),
        (
            {"joints": [1, 2], "inverse_bind_matrices": np.full((2, 4, 4), np.nan)},
            "one per joint",
        ),
        (
            {"joints": [1, 2], "inverse_bind_matrices": np.full((2, 4, 4), 1e300)},
            "float32 cannot hold",
        ),
        ({"joints": [1, 2], "bind_shape_matrix": "eye"}, "bind_shape_matrix"),
        (
            {"joints": [1, 2], "inverse_bind_matrices": np.full((2, 4, 4), 0.5)},
            r"last row is not \[0, 0, 0, 1\]",
        ),
        (
            {"joints": [1, 2], "bind_shape_matrix": np.diag([1.0, 1.0, 1.0, 2.0])},
            r"last row is not \[0, 0, 0, 1\]",
        ),
        ({"joints": [0, 1]}, "do not share a root node"),
    ],
)
def test_unwritable_skin_is_left_out_with_a_warning(
    tmp_path: Path, skin: Any, match: str
) -> None:
    with pytest.warns(UserWarning, match=match):
        gltf = _written_json(_skinned_scene(skin), tmp_path)
    assert "skins" not in gltf
    assert "skin" not in gltf["nodes"][0]


def test_last_row_noise_is_written_as_exactly_affine(tmp_path: Path) -> None:
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[:, 3, 0] = 1e-9
    out = tmp_path / "noise.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(
            _skinned_scene({"joints": [1, 2], "inverse_bind_matrices": ibm}), out
        )
    (back,) = read_scene(out).global_attrs["skins"]
    np.testing.assert_array_equal(
        back["inverse_bind_matrices"][:, 3], [[0, 0, 0, 1]] * 2
    )


def test_skeleton_that_does_not_root_the_joints_is_dropped(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="not an ancestor of every joint"):
        gltf = _written_json(
            _skinned_scene({"joints": [1, 2], "skeleton": 2}), tmp_path
        )
    assert gltf["skins"] == [{"joints": [1, 2]}]


def test_skin_extras_are_written(tmp_path: Path) -> None:
    skin = {"joints": [1, 2], "extras": {"a": np.int64(1), "b": [np.float32(0.5)]}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gltf = _written_json(_skinned_scene(skin), tmp_path)
    assert gltf["skins"] == [{"joints": [1, 2], "extras": {"a": 1, "b": [0.5]}}]


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"extras": {"a": object()}}, "extras are not JSON"),
        ({"extras": {"a": float("nan")}}, "extras are not JSON"),
        ({"extensions": {"EXT_x": {}}}, "extensions are not written: glTF needs"),
        ({"pose": 1}, r"keys \['pose'\] have no glTF field"),
        ({"name": b"rig"}, "is not a string"),
    ],
)
def test_skin_parts_glTF_cannot_hold_are_dropped_with_a_warning(
    tmp_path: Path, extra: dict, match: str
) -> None:
    with pytest.warns(UserWarning, match=match):
        gltf = _written_json(_skinned_scene({"joints": [1, 2], **extra}), tmp_path)
    assert gltf["skins"] == [{"joints": [1, 2]}]
    assert gltf["nodes"][0]["skin"] == 0


def test_float32_overflow_warns_of_nothing_but_the_skin(tmp_path: Path) -> None:
    skin = {
        "joints": [1, 2],
        "inverse_bind_matrices": np.full((2, 4, 4), 1e300),
        "extensions": {},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gltf = _written_json(_skinned_scene(skin), tmp_path)
    assert [str(w.message) for w in caught] == [
        "glTF: skin 0 has inverse bind matrices float32 cannot hold; it is not written."
    ]
    assert "skins" not in gltf


def test_nodes_follow_their_skin_past_one_left_out(tmp_path: Path) -> None:
    scene = _skinned_scene({"joints": [1, 2]}, node_skin=1)
    scene = dataclasses.replace(
        scene,
        nodes=(*scene.nodes, SceneNode(mesh=0, extras={"skin": 0})),
        scenes=((0, 1, 3),),
        global_attrs={"skins": ["rig", {"name": "b", "joints": [1, 2]}]},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gltf = _written_json(scene, tmp_path)
    assert [str(w.message) for w in caught] == [
        "glTF: skin 0 is not a dict; it is not written."
    ]
    assert gltf["skins"] == [{"name": "b", "joints": [1, 2]}]
    assert gltf["nodes"][0]["skin"] == 0
    assert "skin" not in gltf["nodes"][3]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"node_skin": 3}, "name a skin global_attrs"),
        ({"mesh": None}, "no mesh for it to deform"),
        (
            {"joints": np.array([[0, 2, 0, 0]] * 3, dtype=np.uint8)},
            r"skin of 2 joint\(s\) but their mesh's joints reach index 2",
        ),
    ],
)
def test_node_that_cannot_wear_its_skin_is_written_unskinned(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    with pytest.warns(UserWarning, match=match):
        gltf = _written_json(_skinned_scene({"joints": [1, 2]}, **kwargs), tmp_path)
    assert "skin" not in gltf["nodes"][0]


@pytest.mark.parametrize(
    "joints",
    [
        np.array([[0.5, 1, 0, 0]] * 3),
        np.array([[0, -1, 0, 0]] * 3, dtype=np.int16),
        np.array([[0, 70000, 0, 0]] * 3),
        np.array([["a", "b", "c", "d"]] * 3),
    ],
)
def test_joints_glTF_cannot_hold_are_not_written(
    tmp_path: Path, joints: np.ndarray
) -> None:
    with (
        pytest.warns(UserWarning, match="has no JOINTS_0 and WEIGHTS_0"),
        pytest.warns(UserWarning, match="not integers in 0..65535"),
    ):
        gltf = _written_json(
            _skinned_scene({"joints": [1, 2]}, joints=joints), tmp_path
        )
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_0" not in attrs and "WEIGHTS_0" not in attrs


@pytest.mark.parametrize(
    ("influences", "match"),
    [
        ({"joints": np.array([0, 1, 0])}, "are not four per vertex"),
        ({"joints": np.array([[0, 1]] * 3)}, "are not four per vertex"),
        (
            {"weights": np.array([[1.0, 0.0]] * 3, dtype=np.float32)},
            "are not four per vertex",
        ),
        ({"omit": ("weights",)}, "only one is given"),
        ({"omit": ("joints",)}, "only one is given"),
        ({"weights": np.array([["a"] * 4] * 3)}, "weights are neither floats"),
        ({"weights": np.array([[1, 0, 0, 0]] * 3)}, "weights are neither floats"),
    ],
)
def test_influences_that_are_not_two_vec4_are_not_written(
    tmp_path: Path, influences: dict, match: str
) -> None:
    with (
        pytest.warns(UserWarning, match="has no JOINTS_0 and WEIGHTS_0"),
        pytest.warns(UserWarning, match=match),
    ):
        gltf = _written_json(_skinned_scene({"joints": [1, 2]}, **influences), tmp_path)
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_0" not in attrs and "WEIGHTS_0" not in attrs
    assert "skin" not in gltf["nodes"][0]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_weights_are_written_normalised(tmp_path: Path, dtype: type) -> None:
    top = np.iinfo(dtype).max
    weights = np.array([[top, 0, 0, 0]] * 3, dtype=dtype)
    out = tmp_path / "w.glb"
    write_scene(_skinned_scene({"joints": [1, 2]}, weights=weights), out)
    np.testing.assert_allclose(
        read_scene(out).meshes[0].vertex_attrs["weights"], [[1, 0, 0, 0]] * 3
    )


@pytest.mark.parametrize("binary", [True, False])
def test_a_value_json_cannot_hold_raises_a_codec_error(
    tmp_path: Path, binary: bool
) -> None:
    scene = _channel_scene(
        {"sampler": 0, "target": {"node": 0, "path": "scale", "extras": object()}}
    )
    with pytest.raises(CodecError, match="not JSON serializable"):
        write_scene(scene, tmp_path / ("a.glb" if binary else "a.gltf"))


def test_nan_in_a_passed_through_value_is_not_blamed_on_the_mesh(
    tmp_path: Path,
) -> None:
    scene = _channel_scene(
        {"sampler": 0, "target": {"node": 0, "path": "scale", "extras": [np.nan]}}
    )
    with pytest.raises(CodecError, match="a value is NaN or Inf"):
        write_scene(scene, tmp_path / "a.glb")


def test_many_unskinned_nodes_are_counted_past_ten(tmp_path: Path) -> None:
    scene = _skinned_scene({"joints": [1, 2]}, node_skin=7)
    scene = dataclasses.replace(scene, nodes=(*scene.nodes, *[scene.nodes[0]] * 11))
    with pytest.warns(UserWarning, match=r"11\] and 2 more name a skin"):
        _written_json(scene, tmp_path)


def test_second_influence_set_is_written(tmp_path: Path) -> None:
    second = {
        "joints_1": np.array([[1, 0, 0, 0]] * 3, dtype=np.uint8),
        "weights_1": np.array([[0.5, 0, 0, 0]] * 3),
    }
    weights = np.array([[0.25, 0.25, 0, 0]] * 3)
    scene = _skinned_scene({"joints": [1, 2]}, weights=weights, extra_attrs=second)
    out = tmp_path / "eight.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out)
    attrs = back.meshes[0].vertex_attrs
    np.testing.assert_array_equal(attrs["joints_1"], second["joints_1"])
    np.testing.assert_allclose(attrs["weights_1"], second["weights_1"])
    np.testing.assert_allclose(attrs["weights"].sum(1) + attrs["weights_1"].sum(1), 1)
    assert back.nodes[0].extras["skin"] == 0


@pytest.mark.parametrize(
    ("second", "match"),
    [
        (
            {
                "joints_1": np.array([[0, 2, 0, 0]] * 3),
                "weights_1": np.zeros((3, 4)),
            },
            r"skin of 2 joint\(s\) but their mesh's joints reach index 2",
        ),
        (
            {"joints_1": np.array([[0, 1]] * 3), "weights_1": np.zeros((3, 2))},
            "JOINTS_1 and WEIGHTS_1 are not written",
        ),
        (
            {"joints_2": np.zeros((3, 4)), "weights_2": np.zeros((3, 4))},
            r"vertex attribute\(s\) \['joints_2', 'weights_2'\] have no glTF",
        ),
    ],
)
def test_second_influence_set_glTF_cannot_hold_warns(
    tmp_path: Path, second: dict, match: str
) -> None:
    with pytest.warns(UserWarning, match=match):
        gltf = _written_json(
            _skinned_scene({"joints": [1, 2]}, extra_attrs=second), tmp_path
        )
    assert "JOINTS_0" in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_mesh_name_round_trips(tmp_path: Path) -> None:
    scene = _skinned_scene({"joints": [1, 2]})
    poly = dataclasses.replace(scene.meshes[0], global_attrs={"mesh_name": "body"})
    out = tmp_path / "named.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(dataclasses.replace(scene, meshes=(poly,)), out)
        write(poly, tmp_path / "flat.glb")
    assert read_scene(out).meshes[0].global_attrs["mesh_name"] == "body"
    assert read_scene(tmp_path / "flat.glb").meshes[0].global_attrs["mesh_name"] == (
        "body"
    )


@pytest.mark.parametrize(
    ("mesh_globals", "match"),
    [
        ({"mesh_name": 3}, "mesh_name 3 is not a string"),
        ({"gnum": 1}, r"mesh 0: global_attrs \['gnum'\] have no glTF"),
    ],
)
def test_mesh_globals_glTF_cannot_hold_warn(
    tmp_path: Path, mesh_globals: dict, match: str
) -> None:
    scene = _skinned_scene({"joints": [1, 2]})
    poly = dataclasses.replace(scene.meshes[0], global_attrs=mesh_globals)
    with pytest.warns(UserWarning, match=match):
        gltf = _written_json(dataclasses.replace(scene, meshes=(poly,)), tmp_path)
    assert "name" not in gltf["meshes"][0]


def test_scene_extras_round_trip(tmp_path: Path) -> None:
    scene = dataclasses.replace(
        _skinned_scene({"joints": [1, 2]}),
        global_attrs={"skins": [{"joints": [1, 2]}], "extras": {"a": [1, 2]}},
    )
    out = tmp_path / "extras.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    assert read_scene(out).global_attrs["extras"] == {"a": [1, 2]}


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"extras": {"a": object()}}, r"global_attrs\['extras'\] are not JSON"),
        ({"extensions": {}}, r"global_attrs\['extensions'\] are not written"),
        ({"units": "m"}, r"global_attrs \['units'\] have no glTF counterpart"),
    ],
)
def test_scene_globals_glTF_cannot_hold_warn(
    tmp_path: Path, extra: dict, match: str
) -> None:
    scene = _skinned_scene({"joints": [1, 2]})
    scene = dataclasses.replace(scene, global_attrs={**scene.global_attrs, **extra})
    with pytest.warns(UserWarning, match=match):
        gltf = _written_json(scene, tmp_path)
    assert "extras" not in gltf


def test_skinned_collada_scene_keeps_its_rig_in_gltf(tmp_path: Path) -> None:
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[1, 0, 3] = -1.0
    bsm = _translation(0.0, 0.0, 2.0)
    scene = _skinned_scene(
        {
            "name": "rig",
            "joints": [1, 2],
            "skeleton": 1,
            "inverse_bind_matrices": ibm,
            "bind_shape_matrix": bsm,
        }
    )
    dae = tmp_path / "mid.dae"
    glb = tmp_path / "back.glb"
    _collada.write_scene(scene, dae)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mid = _collada.read_scene(dae)
        write_scene(mid, glb)
    back = read_scene(glb)
    (skin,) = back.global_attrs["skins"]
    assert [back.nodes[j].name for j in skin["joints"]] == ["hip", "knee"]
    assert back.nodes[skin["skeleton"]].name == "hip"
    np.testing.assert_allclose(skin["inverse_bind_matrices"], ibm @ bsm, atol=1e-6)
    (skinned,) = [n for n in back.nodes if n.extras.get("skin") == 0]
    mesh = back.meshes[skinned.mesh]
    order = np.lexsort(mesh.vertices.T[::-1])
    source = scene.meshes[0]
    expect = np.lexsort(source.vertices.T[::-1])
    for key in ("joints", "weights"):
        np.testing.assert_array_equal(
            mesh.vertex_attrs[key][order], source.vertex_attrs[key][expect]
        )


def test_whole_float_joints_are_written_as_integers(tmp_path: Path) -> None:
    joints = np.array([[0.0, 1.0, 0.0, 0.0]] * 3)
    gltf = _written_json(_skinned_scene({"joints": [1, 2]}, joints=joints), tmp_path)
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert gltf["accessors"][attrs["JOINTS_0"]]["componentType"] == 5121
    assert gltf["nodes"][0]["skin"] == 0


def test_skins_that_are_not_a_list_are_not_written(tmp_path: Path) -> None:
    scene = dataclasses.replace(
        _skinned_scene({"joints": [1, 2]}), global_attrs={"skins": {"joints": [1]}}
    )
    with (
        pytest.warns(UserWarning, match="name a skin global_attrs"),
        pytest.warns(UserWarning, match="is not a list; it is not written"),
    ):
        gltf = _written_json(scene, tmp_path)
    assert "skins" not in gltf


def _channel_scene(channel: Any, *, matrix: np.ndarray | None = None) -> SceneData:
    """Return a one-node scene whose one animation holds ``channel``."""
    sampler = {
        "times": np.array([0.0, 1.0]),
        "values": np.zeros((2, 3)),
        "interpolation": "LINEAR",
    }
    node = SceneNode(mesh=0) if matrix is None else SceneNode(mesh=0, matrix=matrix)
    return SceneData(
        meshes=(_triangle_poly(),),
        nodes=(node,),
        scenes=((0,),),
        global_attrs={"animations": [{"samplers": [sampler], "channels": [channel]}]},
    )


@pytest.mark.parametrize("sampler", [99, -1, "0", None, 0.5, True])
def test_channel_naming_no_sampler_leaves_its_node_a_matrix(
    tmp_path: Path, sampler: Any
) -> None:
    """A dropped channel neither animates its node nor borrows a sampler."""
    moved = np.eye(4)
    moved[0, 3] = 2.0
    channel = {"sampler": sampler, "target": {"node": 0, "path": "translation"}}
    out = tmp_path / "no_sampler.glb"
    write_scene(_channel_scene(channel, matrix=moved), out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "animations" not in gltf
    assert "matrix" in gltf["nodes"][0] and "translation" not in gltf["nodes"][0]


@pytest.mark.parametrize(
    ("target", "match"),
    [
        (None, "is not an object"),
        ("node", "is not an object"),
        ({"node": 99, "path": "translation"}, "not one of the scene's"),
        ({"node": -1, "path": "translation"}, "not one of the scene's"),
        ({"node": "a", "path": "translation"}, "not one of the scene's"),
        ({"node": 0, "path": 3}, "not a glTF one"),
    ],
)
def test_channel_with_a_target_glTF_cannot_hold_is_skipped(
    tmp_path: Path, target: Any, match: str
) -> None:
    out = tmp_path / "target.glb"
    with pytest.warns(UserWarning, match=match):
        write_scene(_channel_scene({"sampler": 0, "target": target}), out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "animations" not in gltf


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_animated_node_with_a_non_finite_matrix_raises(
    tmp_path: Path, bad: float
) -> None:
    matrix = np.eye(4)
    matrix[0, 0] = bad
    channel = {"sampler": 0, "target": {"node": 0, "path": "translation"}}
    with pytest.raises(CodecError, match="NaN or Inf"):
        write_scene(_channel_scene(channel, matrix=matrix), tmp_path / "nan.glb")


def test_increasing_float32_never_turns_a_time_negative() -> None:
    out = _increasing_float32(np.array([0.0, 1e-50, 1e-49, 1.0]))
    assert out[0] == 0 and (np.diff(out) > 0).all()
    np.testing.assert_array_equal(out[-1], np.float32(1.0))


def test_increasing_float32_keeps_negative_times_moving_down() -> None:
    out = _increasing_float32(np.array([-1.0, -1.0 + 1e-12, 0.0]))
    assert (np.diff(out) > 0).all() and out[1] == np.float32(-1.0)


def test_increasing_float32_leaves_repeated_times_alone() -> None:
    times = np.array([0.0, 1.0, 1.0, 1.0 + 1e-12, 2.0])
    np.testing.assert_array_equal(_increasing_float32(times), times.astype(np.float32))


def _sampler_scene(sampler: Any, *, path: str = "translation") -> SceneData:
    """Return a one-node scene whose one channel of ``path`` uses ``sampler``."""
    return SceneData(
        meshes=(_triangle_poly(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={
            "animations": [
                {
                    "samplers": [sampler],
                    "channels": [{"sampler": 0, "target": {"node": 0, "path": path}}],
                }
            ]
        },
    )


_T2 = np.array([0.0, 1.0])


@pytest.mark.parametrize(
    ("sampler", "path", "match"),
    [
        (5, "translation", "is not an object"),
        ({}, "translation", "no times or no values"),
        ({"times": _T2}, "translation", "no times or no values"),
        (
            {"times": _T2, "values": np.zeros((2, 3)), "interpolation": ["LINEAR"]},
            "translation",
            "interpolates",
        ),
        ({"times": ["a"], "values": np.zeros((1, 3))}, "translation", "not numbers"),
        ({"times": np.zeros(0), "values": np.zeros((0, 3))}, "translation", "no key"),
        ({"times": _T2, "values": np.full((2, 4), np.nan)}, "rotation", "NaN or Inf"),
        ({"times": _T2, "values": np.zeros((2, 3))}, "rotation", "not 4 for each"),
        ({"times": _T2, "values": np.zeros((1, 3))}, "scale", "not 3 for each"),
        (
            {"times": _T2, "values": np.zeros((2, 3)), "interpolation": "CUBICSPLINE"},
            "translation",
            "not 3 for each",
        ),
        ({"times": _T2, "values": np.zeros(3)}, "weights", "not a whole number"),
    ],
)
def test_channel_with_a_sampler_glTF_cannot_hold_is_skipped(
    tmp_path: Path, sampler: Any, path: str, match: str
) -> None:
    """A sampler that is no object, holds no usable keys, or whose values do
    not fit its path is skipped with a warning instead of raising or being
    written as an accessor of the wrong type or count."""
    out = tmp_path / "sampler.glb"
    with pytest.warns(UserWarning, match=match):
        write_scene(_sampler_scene(sampler, path=path), out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "animations" not in gltf
    assert "matrix" not in gltf["nodes"][0] and "rotation" not in gltf["nodes"][0]


def test_matrix_channel_with_an_unhashable_interpolation_is_skipped(
    tmp_path: Path,
) -> None:
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    with pytest.warns(UserWarning, match="not written"):
        write_scene(_matrix_scene(keys, interp=["LINEAR"]), tmp_path / "interp.glb")


def test_second_channel_animating_the_same_node_path_is_skipped(
    tmp_path: Path,
) -> None:
    """glTF forbids two channels of one animation on the same target."""
    scene = _sampler_scene({"times": _T2, "values": np.zeros((2, 3))})
    anim = scene.global_attrs["animations"][0]
    anim["samplers"].append({"times": _T2, "values": np.ones((2, 3))})
    anim["channels"].append({"sampler": 1, "target": {"node": 0, "path": "scale"}})
    anim["channels"].append(
        {"sampler": 1, "target": {"node": 0, "path": "translation"}}
    )
    out = tmp_path / "twice.glb"
    with pytest.warns(UserWarning, match="already animates the translation of node 0"):
        write_scene(scene, out)
    tracks = _tracks(read_scene(out))
    assert sorted(tracks) == ["scale", "translation"]
    np.testing.assert_array_equal(tracks["translation"]["values"], np.zeros((2, 3)))


@pytest.mark.parametrize(
    ("animations", "match"),
    [
        ([5], "animation 0 is not an object"),
        ([{"channels": 5, "samplers": []}], "animation 0 is not an object"),
        (5, "animations is not a list"),
    ],
)
def test_animation_that_is_not_an_object_is_skipped(
    tmp_path: Path, animations: Any, match: str
) -> None:
    scene = dataclasses.replace(
        _sampler_scene({}), global_attrs={"animations": animations}
    )
    out = tmp_path / "anim.glb"
    with pytest.warns(UserWarning, match=match):
        write_scene(scene, out)
    assert "animations" not in _parse_glb(out.read_bytes())[0]


def test_animations_set_to_none_write_no_animation(tmp_path: Path) -> None:
    scene = dataclasses.replace(_sampler_scene({}), global_attrs={"animations": None})
    out = tmp_path / "none.glb"
    write_scene(scene, out)
    assert "animations" not in _parse_glb(out.read_bytes())[0]


def test_flat_values_are_written_at_the_width_of_their_path(tmp_path: Path) -> None:
    """Six flat values of a translation are two VEC3 keys, not six scalars,
    and morph weights of two targets are four scalars, not two."""
    out = tmp_path / "flat.glb"
    write_scene(_sampler_scene({"times": _T2, "values": np.arange(6.0)}), out)
    gltf, _ = _parse_glb(out.read_bytes())
    output = gltf["accessors"][gltf["animations"][0]["samplers"][0]["output"]]
    assert (output["type"], output["count"]) == ("VEC3", 2)

    weights = np.array([[0.0, 1.0], [0.5, 0.25]])
    write_scene(_sampler_scene({"times": _T2, "values": weights}, path="weights"), out)
    gltf, _ = _parse_glb(out.read_bytes())
    output = gltf["accessors"][gltf["animations"][0]["samplers"][0]["output"]]
    assert (output["type"], output["count"]) == ("SCALAR", 4)
    np.testing.assert_array_equal(
        _tracks(read_scene(out))["weights"]["values"], weights.ravel()
    )


def test_shared_times_and_samplers_are_written_once(tmp_path: Path) -> None:
    """The three channels a matrix one splits into share one time accessor,
    and two channels naming one sampler share that sampler."""
    out = tmp_path / "shared.glb"
    write_scene(_matrix_scene(np.tile(np.eye(4).ravel(), (2, 1))), out)
    anim = _parse_glb(out.read_bytes())[0]["animations"][0]
    assert len(anim["samplers"]) == 3
    assert len({s["input"] for s in anim["samplers"]}) == 1

    scene = _sampler_scene({"times": _T2, "values": np.ones((2, 3))})
    scene.global_attrs["animations"][0]["channels"].append(
        {"sampler": 0, "target": {"node": 0, "path": "scale"}}
    )
    write_scene(scene, out)
    anim = _parse_glb(out.read_bytes())[0]["animations"][0]
    assert [c["sampler"] for c in anim["channels"]] == [0, 0]
    assert len(anim["samplers"]) == 1


def test_numpy_integers_in_the_scene_are_written(tmp_path: Path) -> None:
    """A node, mesh or child index held as a numpy integer is plain JSON."""
    scene = _sampler_scene({"times": _T2, "values": np.zeros((2, 3))})
    scene.global_attrs["animations"][0]["channels"][0]["target"]["node"] = np.int64(0)
    scene = dataclasses.replace(
        scene,
        nodes=(SceneNode(mesh=np.int32(0), children=(np.int64(1),)), SceneNode()),
    )
    out = tmp_path / "numpy.glb"
    write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert gltf["nodes"][0]["children"] == [1] and gltf["nodes"][0]["mesh"] == 0
    assert gltf["animations"][0]["channels"][0]["target"]["node"] == 0


def test_skin_accessor_without_a_buffer_view_is_not_decoded() -> None:
    """Such an accessor reads as zeros, which are no inverse bind matrices."""
    gltf = {"accessors": [{"count": 1, "componentType": 5126, "type": "MAT4"}]}
    skin = {"joints": [0], "inverseBindMatrices": 0}
    with pytest.warns(UserWarning, match="has no bufferView"):
        (out,) = _decode_skins(gltf, [skin], [])
    assert out == skin


def _animated_glb() -> tuple[dict, bytes]:
    """Return the JSON and binary chunk of a GLB holding one animation."""
    buf = io.BytesIO()
    write_scene(_sampler_scene({"times": _T2, "values": np.zeros((2, 3))}), buf)
    return _parse_glb(buf.getvalue())


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda g: g["animations"][0]["samplers"][0].pop("input"), "sampler 0 of"),
        (lambda g: g["animations"][0]["samplers"][0].update(output=99), "sampler 0"),
        (lambda g: g.update(animations=[5]), "animation 0 is not an object"),
        (lambda g: g.update(animations=5), "animations is not a list"),
    ],
)
def test_read_malformed_animation_raises_codec_error(change: Any, match: str) -> None:
    gltf, bin_chunk = _animated_glb()
    change(gltf)
    with pytest.raises(CodecError, match=match):
        read_scene(io.BytesIO(_make_glb(gltf, bin_chunk)))


def test_skipped_element_warning_names_the_caller(tmp_path: Path) -> None:
    tetra = PolyData(
        vertices=np.eye(4, 3),
        connectivity=np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 3, 7], dtype=np.int32),
        element_types=np.array(
            [ELEMENT_TYPES["triangle"], ELEMENT_TYPES["tetra"]], dtype=np.uint8
        ),
    )
    scene = SceneData(meshes=(tetra,), nodes=(SceneNode(mesh=0),), scenes=((0,),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_scene(scene, tmp_path / "tetra.glb")
    assert [w.filename for w in caught] == [__file__]


def _pointer_scene(values: Any) -> SceneData:
    """Return a scene whose one channel animates a pointer with ``values``."""
    scene = _sampler_scene({"times": _T2, "values": values}, path="pointer")
    target = scene.global_attrs["animations"][0]["channels"][0]["target"]
    del target["node"]
    target["extensions"] = {"KHR_animation_pointer": {"pointer": "/nodes/0/scale"}}
    return scene


@pytest.mark.parametrize(
    ("values", "acc_type"),
    [
        (np.arange(2.0), "SCALAR"),
        (np.arange(2.0).reshape(2, 1), "SCALAR"),
        (np.arange(4.0).reshape(2, 2), "VEC2"),
        (np.arange(6.0).reshape(2, 3), "VEC3"),
        (np.arange(8.0).reshape(2, 4), "VEC4"),
        (np.arange(18.0).reshape(2, 9), "MAT3"),
        (np.arange(32.0).reshape(2, 16), "MAT4"),
    ],
)
def test_pointer_values_are_written_at_their_own_width(
    tmp_path: Path, values: np.ndarray, acc_type: str
) -> None:
    """One accessor element per key, of the type as wide as a key's row."""
    out = tmp_path / "pointer.glb"
    write_scene(_pointer_scene(values), out)
    gltf, bin_chunk = _parse_glb(out.read_bytes())
    output = gltf["accessors"][gltf["animations"][0]["samplers"][0]["output"]]
    assert (output["type"], output["count"]) == (acc_type, 2)
    view = gltf["bufferViews"][output["bufferView"]]
    assert view["byteLength"] == values.size * 4
    back = read_scene(out).global_attrs["animations"][0]["samplers"][0]["values"]
    np.testing.assert_array_equal(back.ravel(), values.ravel())


@pytest.mark.parametrize(
    "values", [np.zeros((2, 5)), np.zeros((2, 2, 2)), np.zeros(6), np.zeros((3, 3))]
)
def test_pointer_values_of_no_glTF_type_are_skipped(
    tmp_path: Path, values: np.ndarray
) -> None:
    """A width glTF has no accessor type for is not written as a VEC3."""
    out = tmp_path / "pointer.glb"
    with pytest.warns(UserWarning, match="glTF has a type for"):
        write_scene(_pointer_scene(values), out)
    assert "animations" not in _parse_glb(out.read_bytes())[0]


def test_channel_target_with_a_null_node_is_written_without_it(
    tmp_path: Path,
) -> None:
    """A target's ``node`` is an integer or absent in glTF, never null."""
    scene = _sampler_scene({"times": _T2, "values": np.zeros((2, 3))})
    scene.global_attrs["animations"][0]["channels"][0]["target"]["node"] = None
    out = tmp_path / "null_node.glb"
    write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert gltf["animations"][0]["channels"][0]["target"] == {"path": "translation"}


@pytest.mark.parametrize(
    ("sampler", "path", "match"),
    [
        ({"times": _T2, "values": np.full((2, 3), 1e39)}, "translation", "float32"),
        (
            {"times": np.array([0.0, 1e39]), "values": np.zeros((2, 3))},
            "translation",
            "float32",
        ),
        (
            {"times": np.array([1.0, 0.0]), "values": np.zeros((2, 3))},
            "translation",
            "not strictly increasing",
        ),
        (
            {"times": np.array([0.0, 0.0]), "values": np.zeros((2, 3))},
            "scale",
            "not strictly increasing",
        ),
        ({"times": _T2, "values": np.zeros((2, 4))}, "rotation", "zero length"),
        (
            {
                "times": _T2,
                "values": np.tile(
                    [[1.0, 0, 0, 0], [0, 0, 0, 0], [1.0, 0, 0, 0]], (2, 1)
                ),
                "interpolation": "CUBICSPLINE",
            },
            "rotation",
            "zero length",
        ),
    ],
)
def test_sampler_glTF_would_store_wrong_is_skipped(
    tmp_path: Path, sampler: dict, path: str, match: str
) -> None:
    """A value float32 overflows on, times out of order and a zero quaternion
    skip their channel with a warning rather than fail or spoil the file."""
    out = tmp_path / "sampler.glb"
    with pytest.warns(UserWarning, match=match):
        write_scene(_sampler_scene(sampler, path=path), out)
    gltf, _ = _parse_glb(out.read_bytes())
    assert "animations" not in gltf and "rotation" not in gltf["nodes"][0]


def test_cubicspline_rotation_with_zero_tangents_is_written(tmp_path: Path) -> None:
    """Only the middle row of each three is a rotation; tangents may be zero."""
    values = np.tile([[0.0, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 0, 0]], (2, 1))
    sampler = {"times": _T2, "values": values, "interpolation": "CUBICSPLINE"}
    out = tmp_path / "spline.glb"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(_sampler_scene(sampler, path="rotation"), out)
    np.testing.assert_array_equal(
        _tracks(read_scene(out))["rotation"]["values"], values
    )


def test_matrix_channel_float32_cannot_hold_is_skipped_whole(tmp_path: Path) -> None:
    """Not split into a skipped translation beside a written rotation."""
    keys = np.tile(np.eye(4).ravel(), (2, 1))
    keys[1, 3] = 1e39
    out = tmp_path / "huge.glb"
    with pytest.warns(UserWarning, match="not written"):
        write_scene(_matrix_scene(keys), out)
    assert "animations" not in _parse_glb(out.read_bytes())[0]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda g: g["animations"][0].update(channels=5), "animation 0 is not"),
        (lambda g: g["animations"][0].update(channels=[5]), "channel 0 of"),
        (lambda g: g["animations"][0].update(samplers=[5]), "sampler 0 of"),
        (
            lambda g: g["animations"][0]["samplers"][0].update(input=-1),
            "names no accessor as its input",
        ),
        (
            lambda g: g["animations"][0]["samplers"][0].update(input=True),
            "names no accessor as its input",
        ),
        (
            lambda g: g["animations"][0]["samplers"][0].update(output=-1),
            "names no accessor as its output",
        ),
        (
            lambda g: g["animations"][0]["samplers"][0].update(
                input=g["animations"][0]["samplers"][0]["output"]
            ),
            "not a SCALAR one",
        ),
    ],
)
def test_read_animation_naming_no_accessor_raises_codec_error(
    change: Any, match: str
) -> None:
    """A negative or boolean index reads no accessor from the end or as 0/1,
    and channels are checked as the samplers are."""
    gltf, bin_chunk = _animated_glb()
    change(gltf)
    with pytest.raises(CodecError, match=match):
        read_scene(io.BytesIO(_make_glb(gltf, bin_chunk)))


def test_animation_name_that_is_not_a_string_is_left_out(tmp_path: Path) -> None:
    scene = _sampler_scene({"times": _T2, "values": np.zeros((2, 3))})
    scene.global_attrs["animations"][0]["name"] = 5
    out = tmp_path / "name.glb"
    write_scene(scene, out)
    assert "name" not in _parse_glb(out.read_bytes())[0]["animations"][0]


# ---------------------------------------------------------------------------
# TANGENT rebuilt from a COLLADA tangent frame
# ---------------------------------------------------------------------------

_FRAME_TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_TANGENTS = np.array([[1.0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 1]])
_NORMALS = np.tile([0.0, 0, 1], (3, 1))
_BITANGENT = _TANGENTS[:, 3:] * np.cross(_NORMALS, _TANGENTS[:, :3])


def _frame_scene(attrs: dict[str, np.ndarray]) -> SceneData:
    poly = make_polydata(
        _FRAME_TRI, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs=attrs
    )
    return SceneData(meshes=(poly,), nodes=(SceneNode(mesh=0),))


def _written_attrs(tmp_path: Path, attrs: dict[str, np.ndarray]) -> dict:
    write_scene(_frame_scene(attrs), tmp_path / "a.glb")
    return read_scene(tmp_path / "a.glb").meshes[0].vertex_attrs


@pytest.mark.parametrize(
    ("t_key", "b_key"), [("textangent", "texbinormal"), ("tangent", "binormal")]
)
def test_write_rebuilds_tangents_from_a_frame(
    tmp_path: Path, t_key: str, b_key: str
) -> None:
    attrs = {"normals": _NORMALS, t_key: _TANGENTS[:, :3], b_key: _BITANGENT}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = _written_attrs(tmp_path, attrs)
    assert sorted(back) == ["normals", "tangents"]
    np.testing.assert_array_equal(back["tangents"], _TANGENTS)


def test_write_takes_handedness_plus_one_without_a_bitangent(
    tmp_path: Path,
) -> None:
    attrs = {"normals": _NORMALS, "textangent": _TANGENTS[:, :3]}
    with pytest.warns(UserWarning, match="it is taken as \\+1"):
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"][:, 3], [1, 1, 1])


def test_write_keeps_its_own_tangents_over_a_frame(tmp_path: Path) -> None:
    attrs = {"normals": _NORMALS, "tangents": _TANGENTS, "textangent": _NORMALS}
    with pytest.warns(UserWarning, match=r"\['textangent'\] have no glTF semantic"):
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"], _TANGENTS)


def test_write_leaves_out_a_frame_without_normals(tmp_path: Path) -> None:
    attrs = {"textangent": _TANGENTS[:, :3], "texbinormal": _BITANGENT}
    with pytest.warns(UserWarning, match="TANGENT without a NORMAL"):
        back = _written_attrs(tmp_path, attrs)
    assert back == {}


def test_write_scales_frame_tangents_to_unit_length(tmp_path: Path) -> None:
    attrs = {
        "normals": _NORMALS,
        "textangent": _TANGENTS[:, :3] * [[2.0], [0.5], [7.0]],
        "texbinormal": _BITANGENT,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"], _TANGENTS)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_write_leaves_out_a_frame_with_a_tangent_of_no_length(
    tmp_path: Path, bad: float
) -> None:
    tangent = _TANGENTS[:, :3].copy()
    tangent[1] = bad
    attrs = {"normals": _NORMALS, "textangent": tangent, "texbinormal": _BITANGENT}
    with pytest.warns(UserWarning, match="1 row\\(s\\) of zero or non-finite"):
        back = _written_attrs(tmp_path, attrs)
    assert sorted(back) == ["normals"]


def test_write_prefers_a_complete_frame_over_a_lone_textangent(
    tmp_path: Path,
) -> None:
    attrs = {
        "normals": _NORMALS,
        "textangent": _TANGENTS[:, :3],
        "tangent": _TANGENTS[:, :3],
        "binormal": _BITANGENT,
    }
    with pytest.warns(UserWarning, match=r"\['textangent'\] have no glTF semantic"):
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"], _TANGENTS)


def test_write_warns_on_a_bitangent_on_neither_side(tmp_path: Path) -> None:
    bitangent = _BITANGENT.copy()
    bitangent[2] = _NORMALS[2]
    attrs = {
        "normals": _NORMALS,
        "textangent": _TANGENTS[:, :3],
        "texbinormal": bitangent,
    }
    with pytest.warns(UserWarning, match="1 row\\(s\\) on neither side"):
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"][:, 3], [1, -1, 1])


def test_write_takes_handedness_plus_one_on_a_bitangent_of_rounding_noise(
    tmp_path: Path,
) -> None:
    bitangent = _BITANGENT.copy()
    bitangent[2] = [0, -1e-17, 1]
    attrs = {
        "normals": _NORMALS,
        "textangent": _TANGENTS[:, :3],
        "texbinormal": bitangent,
    }
    with pytest.warns(UserWarning, match="1 row\\(s\\) on neither side"):
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"][:, 3], [1, -1, 1])


@pytest.mark.parametrize("width", [2, 3, 5])
def test_write_falls_back_to_the_frame_when_tangents_are_not_vec4(
    tmp_path: Path, width: int
) -> None:
    attrs = {
        "normals": _NORMALS,
        "tangents": np.ones((3, width)),
        "textangent": _TANGENTS[:, :3],
        "texbinormal": _BITANGENT,
    }
    with pytest.warns(UserWarning, match="not the 4 numbers for each of the"):
        back = _written_attrs(tmp_path, attrs)
    assert sorted(back) == ["normals", "tangents"]
    np.testing.assert_array_equal(back["tangents"], _TANGENTS)


def test_write_rebuilds_tangents_from_a_frame_near_the_float64_limit(
    tmp_path: Path,
) -> None:
    attrs = {
        "normals": _NORMALS,
        "textangent": _TANGENTS[:, :3] * 1e200,
        "texbinormal": _BITANGENT * 1e300,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = _written_attrs(tmp_path, attrs)
    np.testing.assert_array_equal(back["tangents"], _TANGENTS)


@pytest.mark.parametrize("keys", [("textangent",), ("normals", "texbinormal")])
def test_write_rebuilds_tangents_from_a_frame_near_the_smallest_float64(
    keys: tuple[str, ...],
) -> None:
    attrs = {
        "normals": _NORMALS,
        "textangent": _TANGENTS[:, :3],
        "texbinormal": _BITANGENT,
    }
    for key in keys:
        attrs[key] = attrs[key] * 1e-170
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = _tangents_of_frame(attrs, n=3, what="glTF")[0]
    np.testing.assert_array_equal(back, _TANGENTS)


class _Untouchable:
    """Normals that fail the test when anything reads them as an array."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("normals were read without a tangent frame")


def test_write_leaves_normals_unread_without_a_tangent_frame() -> None:
    attrs = {"normals": _Untouchable(), "texcoords": np.zeros((3, 2))}
    assert _tangents_of_frame(attrs, n=3, what="glTF") == (None, ())


def test_write_names_the_vertex_count_tangents_miss(tmp_path: Path) -> None:
    attrs = {"normals": _NORMALS, "tangents": np.ones((2, 4))}
    with pytest.warns(UserWarning, match=r"for each of the 3 vertices"):
        back = _written_attrs(tmp_path, attrs)
    assert sorted(back) == ["normals"]
