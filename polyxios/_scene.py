"""Scene-graph container for multi-mesh formats (glTF, FBX, USD, COLLADA).

`SceneData` holds the full scene hierarchy, materials, and texture images
returned by scene-aware codecs.  `PolyData` remains the flat single-mesh
type; `SceneData` sits alongside it, never replaces it.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polyxios._types import PolyData
import polyxios.transforms as transforms


@dataclass(frozen=True, slots=True)
class SceneImage:
    """A texture image: external URI or embedded bytes.

    Parameters
    ----------
    uri
        File-system path or ``data:`` URI.  Mutually exclusive with *data*.
    data
        Raw image bytes decoded from a glTF buffer view or a ``data:`` URI.
    media_type
        MIME type string, e.g. ``"image/png"`` or ``"image/jpeg"``.
    name
        Optional human-readable label from the source file.
    """

    uri: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    name: str = ""


@dataclass(frozen=True, slots=True)
class SceneTexture:
    """A texture: image index plus OpenGL sampler settings.

    Parameters
    ----------
    image
        Index into :attr:`SceneData.images`.
    mag_filter
        GL magnification filter enum (e.g. 9729 = ``GL_LINEAR``).
    min_filter
        GL minification filter enum (e.g. 9987 = ``GL_LINEAR_MIPMAP_LINEAR``).
    wrap_s
        GL wrap mode for the S axis (default 10497 = ``GL_REPEAT``).
    wrap_t
        GL wrap mode for the T axis (default 10497 = ``GL_REPEAT``).
    """

    image: int
    mag_filter: int | None = None
    min_filter: int | None = None
    wrap_s: int = 10497
    wrap_t: int = 10497


@dataclass(frozen=True, slots=True)
class SceneMaterial:
    """PBR metallic-roughness material.

    Parameters
    ----------
    name
        Material name from the source file.
    base_color
        RGBA base-color factor in linear space, each component in [0, 1].
    metallic
        Metallic factor in [0, 1].
    roughness
        Roughness factor in [0, 1].
    emissive
        RGB emissive factor in linear space.
    alpha_mode
        One of ``"OPAQUE"``, ``"MASK"``, or ``"BLEND"``.
    alpha_cutoff
        Alpha cutoff threshold used when *alpha_mode* is ``"MASK"``.
    double_sided
        Whether the material is rendered on both sides.
    base_color_texture
        Index into :attr:`SceneData.textures`, or ``None``.
    normal_texture
        Index into :attr:`SceneData.textures`, or ``None``.
    metallic_roughness_texture
        Index into :attr:`SceneData.textures`, or ``None``.
    occlusion_texture
        Index into :attr:`SceneData.textures`, or ``None``.
    emissive_texture
        Index into :attr:`SceneData.textures`, or ``None``.
    extras
        Format-specific extra data preserved for round-tripping.
    """

    name: str = ""
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 1.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    base_color_texture: int | None = None
    normal_texture: int | None = None
    metallic_roughness_texture: int | None = None
    occlusion_texture: int | None = None
    emissive_texture: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneNode:
    """One node in the scene graph.

    Parameters
    ----------
    name
        Node name from the source file.
    mesh
        Index into :attr:`SceneData.meshes`, or ``None`` if this node
        carries no geometry (e.g. a grouping or camera node).
    children
        Indices of child nodes into :attr:`SceneData.nodes`.
    matrix
        Local 4×4 transform matrix (float64, column-major convention).
        Defaults to the identity transform.
    extras
        Format-specific extra data (cameras, lights, skins) preserved for
        round-tripping.
    """

    name: str = ""
    mesh: int | None = None
    children: tuple[int, ...] = ()
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneData:
    """Multi-mesh scene with hierarchy, materials, and texture images.

    This is the type returned by scene-aware codecs (glTF, FBX, USD,
    COLLADA).  Each entry in *meshes* is a flat :class:`~polyxios.PolyData`;
    ``element_attrs["material"]`` holds ``int32`` indices into *materials*
    (``-1`` meaning no material).  Node transforms are stored as local
    matrices; world transforms are computed on demand by
    :meth:`world_transform`.

    Parameters
    ----------
    meshes
        One :class:`~polyxios.PolyData` per glTF mesh (primitives merged).
    nodes
        Scene-graph nodes with local transforms and parent–child wiring.
    materials
        PBR materials referenced by element_attrs["material"] indices.
    textures
        Textures referencing entries in *images*.
    images
        Raw image data or URI references.
    scenes
        Each entry is a tuple of root-node indices defining one scene.
    active_scene
        Index of the default scene in *scenes*.
    name
        Optional top-level name from the source file.
    global_attrs
        Format-level metadata: ``asset``, raw ``animations``, ``skins``, etc.
    """

    meshes: tuple[PolyData, ...]
    nodes: tuple[SceneNode, ...]
    materials: tuple[SceneMaterial, ...] = ()
    textures: tuple[SceneTexture, ...] = ()
    images: tuple[SceneImage, ...] = ()
    scenes: tuple[tuple[int, ...], ...] = ()
    active_scene: int = 0
    name: str = ""
    global_attrs: dict[str, Any] = field(default_factory=dict)

    def world_transform(self, node_index: int) -> np.ndarray:
        """Return the world-space 4×4 transform for the given node.

        Parameters
        ----------
        node_index
            Index into :attr:`nodes`.

        Returns
        -------
        numpy.ndarray
            Shape (4, 4), dtype float64.  Product of all ancestor local
            matrices from the root down to this node.
        """
        parent: dict[int, int] = {}
        for i, node in enumerate(self.nodes):
            for child in node.children:
                parent[child] = i

        chain: list[int] = []
        idx: int | None = node_index
        while idx is not None:
            chain.append(idx)
            idx = parent.get(idx)
        chain.reverse()

        result = np.eye(4, dtype=np.float64)
        for i in chain:
            result = result @ self.nodes[i].matrix
        return result

    def to_polydata(
        self,
        *,
        scene: int | None = None,
        apply_transforms: bool = True,
    ) -> PolyData:
        """Flatten the scene graph into a single merged PolyData.

        Parameters
        ----------
        scene
            Which entry in :attr:`scenes` to flatten.  Defaults to
            :attr:`active_scene`.  Ignored when :attr:`nodes` is empty.
        apply_transforms
            When ``True``, each mesh's vertices are multiplied by the
            node's world-space transform before merging.

        Returns
        -------
        PolyData
            All scene geometry merged into one mesh.
        """
        if len(self.nodes) == 0:
            if not self.meshes:
                return PolyData(
                    vertices=np.zeros((0, 3), dtype=np.float64),
                    connectivity=np.array([], dtype=np.int32),
                    offsets=np.array([0], dtype=np.int32),
                    element_types=np.array([], dtype=np.uint8),
                )
            return transforms.merge(*self.meshes)

        scene_idx = scene if scene is not None else self.active_scene
        if self.scenes:
            root_indices = list(self.scenes[scene_idx])
        else:
            all_children: set[int] = set()
            for node in self.nodes:
                all_children.update(node.children)
            root_indices = [i for i in range(len(self.nodes)) if i not in all_children]

        collected: list[PolyData] = []
        stack: list[tuple[int, np.ndarray]] = [
            (r, np.eye(4, dtype=np.float64)) for r in reversed(root_indices)
        ]
        while stack:
            node_idx, parent_world = stack.pop()
            node = self.nodes[node_idx]
            world = parent_world @ node.matrix
            if node.mesh is not None:
                mesh = self.meshes[node.mesh]
                if apply_transforms and not np.allclose(world, np.eye(4)):
                    n = mesh.vertices.shape[0]
                    ones = np.ones((n, 1), dtype=np.float64)
                    homo = np.hstack([mesh.vertices, ones])
                    transformed = (homo @ world.T)[:, :3]
                    mesh = dataclasses.replace(mesh, vertices=transformed)
                collected.append(mesh)
            stack.extend((child, world) for child in reversed(node.children))

        if not collected:
            return PolyData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                connectivity=np.array([], dtype=np.int32),
                offsets=np.array([0], dtype=np.int32),
                element_types=np.array([], dtype=np.uint8),
            )
        return transforms.merge(*collected)
