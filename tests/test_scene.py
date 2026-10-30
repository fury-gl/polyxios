"""Tests for the SceneData type and its to_polydata / world_transform methods."""

from pathlib import Path

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._scene import SceneData, SceneMaterial, SceneNode
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


def _animated(times: np.ndarray, *, extra: np.ndarray | None = None) -> SceneData:
    """Return a one-node scene whose sampler and node extras hold arrays.

    Parameters
    ----------
    times
        The sampler's times and values.
    extra
        The node's ``extras["weights"]``, or no extras when None.

    Returns
    -------
    SceneData
        The scene.
    """
    node_extras = {} if extra is None else {"weights": extra}
    return SceneData(
        meshes=(),
        nodes=(SceneNode(extras=node_extras),),
        global_attrs={
            "animations": [{"samplers": [{"times": times, "values": times}]}]
        },
    )


def test_scene_equality_compares_nested_arrays() -> None:
    """Arrays inside global_attrs and node extras compare by value, not raise."""
    t = np.arange(3.0)
    assert _animated(t) == _animated(t.copy())
    assert _animated(t) != _animated(t + 1.0)
    assert _animated(t) != _animated(np.arange(4.0))
    assert _animated(t, extra=t) == _animated(t, extra=t.copy())
    assert _animated(t, extra=t) != _animated(t, extra=[0.0, 1.0, 2.0])


def test_scene_equality_nan_arrays_and_0d_scalars() -> None:
    """A NaN array equals itself, and a 0-d array its scalar, as under ``==``."""
    t = np.array([0.0, np.nan])
    assert _animated(t) == _animated(t.copy())
    assert _animated(t, extra=np.array(1.0)) == _animated(t, extra=np.float64(1.0))
    names = np.array(["a", "b"])
    assert _animated(t, extra=names) == _animated(t, extra=names.copy())


def test_scene_equality_nan_scalars_and_identity() -> None:
    """A scene equals itself, and NaN scalars compare as NaN arrays do."""
    t = np.arange(3.0)
    assert _animated(t, extra=float("nan")) == _animated(t, extra=float("nan"))
    assert _animated(t, extra=np.array(np.nan)) == _animated(t, extra=np.float32("nan"))
    assert _animated(t, extra=float("nan")) != _animated(t, extra=1.0)
    scene = _animated(t, extra=np.array([{"a": t}, None], dtype=object))
    assert scene == scene


def test_scene_equality_object_arrays_compare_element_by_element() -> None:
    t = np.arange(3.0)

    def nested(values: np.ndarray) -> np.ndarray:
        out = np.empty(2, dtype=object)
        out[0], out[1] = {"a": values}, "b"
        return out

    assert _animated(t, extra=nested(t)) == _animated(t, extra=nested(t.copy()))
    assert _animated(t, extra=nested(t)) != _animated(t, extra=nested(t + 1.0))


def test_material_equality_compares_arrays_in_extras() -> None:
    """Arrays in a material's extras compare by value instead of raising."""
    t = np.arange(3.0)
    assert SceneMaterial(extras={"a": t}) == SceneMaterial(extras={"a": t.copy()})
    assert SceneMaterial(extras={"a": t}) != SceneMaterial(extras={"a": t + 1.0})
    assert SceneMaterial(name="a") != SceneMaterial(name="b")
    assert SceneMaterial() == SceneMaterial()


def test_scene_equality_complex_nan_scalars() -> None:
    """A complex NaN scalar equals itself, as it does inside an array."""
    t = np.arange(3.0)
    nan = complex("nan")
    assert _animated(t, extra=nan) == _animated(t, extra=complex("nan"))
    assert _animated(t, extra=np.array([nan])) == _animated(t, extra=np.array([nan]))


def test_material_equality_compares_every_field_by_value() -> None:
    """A NaN factor equals itself, as a NaN in ``extras`` does."""
    assert SceneMaterial(metallic=float("nan")) == SceneMaterial(metallic=float("nan"))
    assert SceneMaterial(metallic=0.5) != SceneMaterial(metallic=0.25)


def test_scene_equality_compares_meshes_by_value() -> None:
    """Two scenes read from one file are equal, their meshes distinct objects."""

    def scene(z: float) -> SceneData:
        mesh = make_polydata(
            np.array([[0.0, 0.0, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            [("triangle", np.array([[0, 1, 2]]))],
        )
        return SceneData(meshes=(mesh,), nodes=(SceneNode(mesh=0),))

    assert scene(0.0) == scene(0.0)
    assert scene(0.0) != scene(1.0)
    assert scene(0.0) != SceneData(meshes=(), nodes=(SceneNode(mesh=0),))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("normals", np.ones((3, 2))),
        ("tangents", np.ones((3, 2))),
        ("normals", np.ones(3)),
    ],
)
def test_directions_without_three_columns_are_kept_unturned(name, values) -> None:
    """A COLLADA NORMAL source of two params read as (n, 2) normals, and the
    transform multiplied them by a 3x3 matrix and raised numpy's ValueError."""
    rot = np.eye(4)
    rot[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    mesh = make_polydata(
        np.eye(3),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={name: values},
    )
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0, matrix=rot),),
        scenes=((0,),),
        active_scene=0,
    )
    result = scene.to_polydata(apply_transforms=True)
    np.testing.assert_array_equal(result.vertex_attrs[name], values)


@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.float32])
def test_integer_tangents_are_turned_in_floating_point(dtype) -> None:
    """The tangents were turned in place in their own dtype, so an integer
    array raised numpy's UFuncTypeError on the normalising division."""
    rot = np.eye(4)
    rot[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    mesh = make_polydata(
        np.eye(3),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"tangents": np.tile(np.array([1, 0, 0, 1], dtype=dtype), (3, 1))},
    )
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0, matrix=rot),),
        scenes=((0,),),
        active_scene=0,
    )
    tangents = scene.to_polydata().vertex_attrs["tangents"]
    assert tangents.dtype == np.result_type(dtype, np.float32)
    np.testing.assert_allclose(tangents, [[0, 1, 0, 1]] * 3, atol=1e-6)


# ---------------------------------------------------------------------------
# Flattening a scene read back from glTF
# ---------------------------------------------------------------------------


def _round_trip(scene: SceneData, path: Path) -> PolyData:
    """Write ``scene`` to ``path``, read it back and flatten it."""
    polyxios.write_scene(scene, path)
    return polyxios.read_scene(path).to_polydata()


@pytest.mark.parametrize("ext", ["glb", "gltf"])
def test_triangle_strip_keeps_its_triangles(tmp_path: Path, ext: str) -> None:
    """A strip came back fanned from its first vertex, which joins vertices
    the strip never does and turns half its triangles over."""
    vertices = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=float
    )
    poly = make_polydata(vertices, [("triangle_strip", np.array([[0, 1, 2, 3, 4]]))])
    scene = SceneData(meshes=(poly,), nodes=(SceneNode(mesh=0),), scenes=((0,),))
    faces = _round_trip(scene, tmp_path / f"strip.{ext}").faces
    np.testing.assert_array_equal(faces, [[0, 1, 2], [2, 1, 3], [2, 3, 4]])


def _front_facing(poly: PolyData) -> np.ndarray:
    """Dot each triangle's geometric normal with its first vertex normal."""
    tris = poly.vertices[poly.faces]
    geometric = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = poly.vertex_attrs["normals"][poly.faces[:, 0]]
    return np.einsum("ij,ij->i", geometric, normals)


@pytest.mark.parametrize("ext", ["glb", "gltf"])
def test_mirrored_node_keeps_the_front_face(tmp_path: Path, ext: str) -> None:
    """The reflection was baked into the points and the normals but not the
    winding, so the triangle faced away from its own normals."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    poly = make_polydata(
        vertices,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": np.tile([0.0, 0.0, 1.0], (3, 1))},
    )
    scene = SceneData(
        meshes=(poly,),
        nodes=(SceneNode(mesh=0, matrix=np.diag([-1.0, 1.0, 1.0, 1.0])),),
        scenes=((0,),),
    )
    assert (_front_facing(_round_trip(scene, tmp_path / f"mirror.{ext}")) > 0).all()


@pytest.mark.parametrize("n_verts", [5, 6])
def test_mirrored_strip_keeps_the_front_face(n_verts: int) -> None:
    """A strip is flipped by reading it backwards or, when that would keep
    its parity, by starting it one vertex earlier."""
    x = np.arange(n_verts) // 2
    y = np.arange(n_verts) % 2
    vertices = np.column_stack([x, y, np.zeros(n_verts)]).astype(float)
    poly = make_polydata(
        vertices,
        [("triangle_strip", np.arange(n_verts)[None])],
        vertex_attrs={"normals": np.tile([0.0, 0.0, -1.0], (n_verts, 1))},
    )
    assert (_front_facing(poly) > 0).all()
    scene = SceneData(
        meshes=(poly,),
        nodes=(SceneNode(mesh=0, matrix=np.diag([1.0, -1.0, 1.0, 1.0])),),
        scenes=((0,),),
    )
    dots = _front_facing(scene.to_polydata())
    assert (dots[~np.isclose(dots, 0)] > 0).all()
    assert np.count_nonzero(dots > 0) == n_verts - 2


def test_double_mirror_keeps_the_winding() -> None:
    """Two reflections make a rotation; nothing is flipped."""
    mesh = _triangle_mesh()
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0, matrix=np.diag([-1.0, -1.0, 1.0, 1.0])),),
        scenes=((0,),),
    )
    np.testing.assert_array_equal(scene.to_polydata().faces, mesh.faces)


@pytest.mark.parametrize("ext", ["glb", "gltf"])
def test_mixed_rgb_and_rgba_meshes_flatten(tmp_path: Path, ext: str) -> None:
    """One mesh's colours carried alpha and another's did not, and merging
    them raised over the channel counts."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    rgb = np.tile([0.25, 0.5, 0.75], (3, 1))
    rgba = np.tile([0.1, 0.2, 0.3, 0.4], (3, 1))
    meshes = tuple(
        make_polydata(
            vertices,
            [("triangle", np.array([[0, 1, 2]]))],
            vertex_attrs={"colors": colors},
        )
        for colors in (rgb, rgba)
    )
    scene = SceneData(
        meshes=meshes,
        nodes=(SceneNode(mesh=0), SceneNode(mesh=1)),
        scenes=((0, 1),),
    )
    flat = _round_trip(scene, tmp_path / f"mixed.{ext}")
    assert flat.vertices.shape == (6, 3)
    assert flat.faces.shape == (2, 3)
    np.testing.assert_allclose(
        flat.vertex_attrs["colors"],
        np.vstack([np.column_stack([rgb, np.ones(3)]), rgba]),
        rtol=1e-6,
    )
