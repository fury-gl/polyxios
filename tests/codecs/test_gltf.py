"""Tests for the glTF/GLB codec.

All tests use real file I/O via ``tmp_path``; no mocks or patches.
The ``_make_glb`` helper assembles valid GLB bytes from a JSON dict and
an optional binary chunk, independently of the codec's own GLB writer.
"""

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._scene import SceneData, SceneMaterial, SceneNode
from polyxios._types import PolyData
from polyxios.codecs._gltf import read, read_scene, write, write_scene
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
    assert poly.vertices.shape[0] == 6
    mat_col = poly.element_attrs["material"]
    assert int(mat_col[0]) == 0
    assert int(mat_col[1]) == 1


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


def test_read_scene_returns_scenedata(tmp_path: Path) -> None:
    p = tmp_path / "m.glb"
    p.write_bytes(_make_glb({"asset": {"version": "2.0"}}))
    assert isinstance(read_scene(p), SceneData)


def test_read_flattens_to_polydata(tmp_path: Path) -> None:
    p = tmp_path / "m.glb"
    p.write_bytes(_make_glb({"asset": {"version": "2.0"}}))
    with pytest.warns(UserWarning, match=r"scene format.*read_scene"):
        result = read(p)
    assert isinstance(result, PolyData)


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
