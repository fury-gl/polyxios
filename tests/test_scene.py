"""Tests for the SceneData type and its to_polydata / world_transform methods."""

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._scene import SceneData, SceneNode
from polyxios._types import PolyData
from polyxios.codecs._gltf import _parse_glb, write_scene


def _triangle_mesh() -> PolyData:
    """A single triangle PolyData for scene tests."""
    return make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )


def _translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Return a 4×4 translation matrix."""
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def _empty_mesh() -> PolyData:
    """An empty PolyData with no vertices or elements."""
    return PolyData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
    )


def test_empty_scene_to_polydata() -> None:
    scene = SceneData(
        meshes=(_empty_mesh(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
    )
    result = scene.to_polydata()
    assert result.vertices.shape == (0, 3)
    assert len(result.element_types) == 0


def test_single_mesh_identity_transform() -> None:
    mesh = _triangle_mesh()
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
    )
    result = scene.to_polydata()
    np.testing.assert_array_equal(result.vertices, mesh.vertices)


def test_transform_applied() -> None:
    mesh = _triangle_mesh()
    t = _translation_matrix(5.0, 0.0, 0.0)
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0, matrix=t),),
        scenes=((0,),),
        active_scene=0,
    )
    result = scene.to_polydata()
    np.testing.assert_allclose(
        result.vertices[:, 0], mesh.vertices[:, 0] + 5.0, atol=1e-10
    )
    np.testing.assert_allclose(result.vertices[:, 1:], mesh.vertices[:, 1:], atol=1e-10)


def test_hierarchical_transform() -> None:
    mesh = _triangle_mesh()
    parent_t = _translation_matrix(1.0, 0.0, 0.0)
    child_t = _translation_matrix(0.0, 2.0, 0.0)
    scene = SceneData(
        meshes=(mesh,),
        nodes=(
            SceneNode(mesh=None, children=(1,), matrix=parent_t),
            SceneNode(mesh=0, matrix=child_t),
        ),
        scenes=((0,),),
        active_scene=0,
    )
    world = scene.world_transform(1)
    np.testing.assert_allclose(world[0, 3], 1.0, atol=1e-10)
    np.testing.assert_allclose(world[1, 3], 2.0, atol=1e-10)

    result = scene.to_polydata()
    np.testing.assert_allclose(
        result.vertices[:, 0], mesh.vertices[:, 0] + 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        result.vertices[:, 1], mesh.vertices[:, 1] + 2.0, atol=1e-10
    )


def test_instancing() -> None:
    mesh = _triangle_mesh()
    t = _translation_matrix(10.0, 0.0, 0.0)
    scene = SceneData(
        meshes=(mesh,),
        nodes=(
            SceneNode(mesh=0),
            SceneNode(mesh=0, matrix=t),
        ),
        scenes=((0, 1),),
        active_scene=0,
    )
    result = scene.to_polydata()
    assert result.vertices.shape[0] == 6  # 3 verts × 2 instances


def test_material_indices_survive_merge() -> None:
    mesh0 = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"material": np.array([0], dtype=np.int32)},
    )
    mesh1 = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"material": np.array([1], dtype=np.int32)},
    )
    scene = SceneData(
        meshes=(mesh0, mesh1),
        nodes=(SceneNode(mesh=0), SceneNode(mesh=1)),
        scenes=((0, 1),),
        active_scene=0,
    )
    result = scene.to_polydata()
    mat = result.element_attrs["material"]
    assert len(mat) == 2
    assert int(mat[0]) == 0
    assert int(mat[1]) == 1


def test_scene_selection() -> None:
    mesh = _triangle_mesh()
    t = _translation_matrix(100.0, 0.0, 0.0)
    scene = SceneData(
        meshes=(mesh,),
        nodes=(
            SceneNode(mesh=0),
            SceneNode(mesh=0, matrix=t),
        ),
        scenes=((0,), (1,)),
        active_scene=0,
    )
    result0 = scene.to_polydata(scene=0)
    result1 = scene.to_polydata(scene=1)
    assert result0.vertices.shape[0] == 3
    assert result1.vertices.shape[0] == 3
    np.testing.assert_allclose(
        result1.vertices[:, 0], mesh.vertices[:, 0] + 100.0, atol=1e-10
    )


def test_world_transform_identity_chain() -> None:
    scene = SceneData(
        meshes=(_triangle_mesh(),),
        nodes=(
            SceneNode(children=(1,)),
            SceneNode(children=(2,)),
            SceneNode(mesh=0),
        ),
        scenes=((0,),),
        active_scene=0,
    )
    world = scene.world_transform(2)
    np.testing.assert_allclose(world, np.eye(4), atol=1e-10)


def test_no_nodes_merges_all_meshes() -> None:
    mesh0 = _triangle_mesh()
    mesh1 = _triangle_mesh()
    scene = SceneData(
        meshes=(mesh0, mesh1),
        nodes=(),
        scenes=(),
        active_scene=0,
    )
    result = scene.to_polydata()
    assert result.vertices.shape[0] == 6


# ---------------------------------------------------------------------------
# Animated-node TRS round-trip
# ---------------------------------------------------------------------------


def test_animated_node_written_as_trs(tmp_path: Path) -> None:
    """An animated node (identity rest-pose) is written with explicit T/R/S keys
    and no 'matrix' key, regardless of whether the transform is identity."""
    mesh = _triangle_mesh()
    times = np.array([0.0, 1.0], dtype=np.float32)
    values = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
        global_attrs={
            "animations": [
                {
                    "samplers": [
                        {"times": times, "values": values, "interpolation": "LINEAR"}
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                }
            ]
        },
    )
    out = tmp_path / "trs.glb"
    write_scene(scene, out)
    gltf, _ = _parse_glb(out.read_bytes())
    node_entry = gltf["nodes"][0]
    assert "matrix" not in node_entry
    assert "translation" in node_entry
    assert "rotation" in node_entry
    assert "scale" in node_entry


# ---------------------------------------------------------------------------
# Root derivation without explicit scenes
# ---------------------------------------------------------------------------


def test_root_derivation_no_scenes() -> None:
    """With scenes=(), only meshes reachable from derived roots are traversed."""
    mesh = _triangle_mesh()
    # node 0: parent (no mesh), node 1: child of 0 (has mesh)
    # With scenes=(), node 0 is derived as root (node 1 is a child).
    scene = SceneData(
        meshes=(mesh,),
        nodes=(
            SceneNode(children=(1,)),
            SceneNode(mesh=0),
        ),
        scenes=(),
        active_scene=0,
    )
    result = scene.to_polydata()
    assert len(result.element_types) == 1


# ---------------------------------------------------------------------------
# Cycle guard
# ---------------------------------------------------------------------------


def test_cycle_raises() -> None:
    """A cycle in the scene graph raises ValueError."""
    scene = SceneData(
        meshes=(_triangle_mesh(),),
        nodes=(
            SceneNode(mesh=0, children=(1,)),
            SceneNode(children=(0,)),
        ),
        scenes=((0,),),
        active_scene=0,
    )
    with pytest.raises(ValueError, match="cycle"):
        scene.world_transform(0)


# ---------------------------------------------------------------------------
# Scene index validation
# ---------------------------------------------------------------------------


def test_invalid_scene_index_raises() -> None:
    """to_polydata(scene=N) raises ValueError when N is out of range."""
    scene = SceneData(
        meshes=(_triangle_mesh(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=0,
    )
    with pytest.raises(ValueError):
        scene.to_polydata(scene=999)


# ---------------------------------------------------------------------------
# Normal transformation
# ---------------------------------------------------------------------------


def test_normals_rotated_with_world_transform() -> None:
    """Normals are transformed by the inverse-transpose of the rotation matrix."""
    # 90° rotation about Z: +Y → −X
    angle = np.pi / 2
    rot = np.eye(4, dtype=np.float64)
    rot[0, 0] = np.cos(angle)
    rot[0, 1] = -np.sin(angle)
    rot[1, 0] = np.sin(angle)
    rot[1, 1] = np.cos(angle)

    input_normals = np.array([[0.0, 1.0, 0.0]] * 3)
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": input_normals},
    )
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0, matrix=rot),),
        scenes=((0,),),
        active_scene=0,
    )
    result = scene.to_polydata(apply_transforms=True)
    result_normals = result.vertex_attrs["normals"]
    # Normals must have changed from the input.
    assert not np.allclose(result_normals, input_normals, atol=1e-6)
    # After 90° rotation about Z, +Y normal becomes −X.
    np.testing.assert_allclose(result_normals, [[-1.0, 0.0, 0.0]] * 3, atol=1e-6)


def test_tangents_transformed_with_world_transform() -> None:
    """Tangents transform covariantly (forward rotation), not by inverse-transpose.

    Under non-uniform scale (2, 1, 1) with no rotation, the pre-fix code used
    pinv(R) = diag(0.5, 1, 1) instead of R.T = R = diag(2, 1, 1), producing
    the wrong direction.
    """
    # Non-uniform scale: x-axis stretched by 2.
    scale = np.eye(4, dtype=np.float64)
    scale[0, 0] = 2.0

    # Tangent pointing at 45° in XY: (1, 1, 0) normalised, w=1.
    input_tgt = np.array([[1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0, 1.0]] * 3)
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"tangents": input_tgt},
    )
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0, matrix=scale),),
        scenes=((0,),),
        active_scene=0,
    )
    result = scene.to_polydata(apply_transforms=True)
    result_tgts = result.vertex_attrs["tangents"]

    # Forward transform: R = diag(2, 1, 1).
    # (1/√2, 1/√2, 0) → (2/√2, 1/√2, 0) → normalised: (2, 1, 0) / √5.
    expected_xyz = np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0)
    np.testing.assert_allclose(result_tgts[:, :3], [expected_xyz] * 3, atol=1e-6)
    # w component must be preserved.
    np.testing.assert_allclose(result_tgts[:, 3], [1.0] * 3, atol=1e-6)

    # Confirm the pre-fix result would differ: pinv(diag(2,1,1)) = diag(0.5,1,1)
    # gives direction (0.5/√2, 1/√2, 0) → normalised: (1, 2, 0) / √5 — wrong.
    wrong_xyz = np.array([1.0, 2.0, 0.0]) / np.sqrt(5.0)
    assert not np.allclose(result_tgts[:, :3], [wrong_xyz] * 3, atol=1e-6), (
        "tangent direction matches the pre-fix (wrong) result"
    )
