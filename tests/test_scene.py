"""Tests for the SceneData type and its to_polydata / world_transform methods."""

import numpy as np

from polyxios import make_polydata
from polyxios._scene import SceneData, SceneMaterial, SceneNode
from polyxios._types import PolyData


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


def test_scene_material_defaults() -> None:
    m = SceneMaterial()
    assert m.base_color == (1.0, 1.0, 1.0, 1.0)
    assert m.metallic == 1.0
    assert m.roughness == 1.0
    assert m.alpha_mode == "OPAQUE"
    assert m.double_sided is False


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
