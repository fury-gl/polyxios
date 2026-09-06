from __future__ import annotations

import numpy as np
import pytest

from polyxios import make_polydata, transforms
from polyxios.transforms import (
    extract_surface,
    filter_element_type,
    merge,
    merge_duplicate_vertices,
    pipeline,
    remove_orphan_vertices,
)


def _tri_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_pipeline_compose() -> None:
    def add_one(poly):  # type: ignore[no-untyped-def]
        import dataclasses

        return dataclasses.replace(poly, vertices=poly.vertices + 1.0)

    def scale_two(poly):  # type: ignore[no-untyped-def]
        import dataclasses

        return dataclasses.replace(poly, vertices=poly.vertices * 2.0)

    poly = _tri_mesh()
    fn = pipeline(add_one, scale_two)
    result = fn(poly)
    expected = (poly.vertices + 1.0) * 2.0
    np.testing.assert_allclose(result.vertices, expected)


def test_remove_orphan_vertices() -> None:
    # 5 vertices but only first 3 referenced
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [99, 99, 99],
            [88, 88, 88],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    assert poly.vertices.shape[0] == 5

    result = remove_orphan_vertices(poly)
    assert result.vertices.shape[0] == 3
    np.testing.assert_allclose(result.vertices, verts[:3])


def test_merge() -> None:
    verts1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly1 = make_polydata(verts1, [("triangle", np.array([[0, 1, 2]]))])

    verts2 = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float64)
    poly2 = make_polydata(verts2, [("triangle", np.array([[0, 1, 2]]))])

    merged = merge(poly1, poly2)
    assert merged.vertices.shape[0] == 6
    assert len(merged.element_types) == 2
    # Second mesh connectivity should be shifted by 3
    assert int(merged.connectivity[3]) == 3
    assert int(merged.connectivity[4]) == 4
    assert int(merged.connectivity[5]) == 5


def test_filter_element_type() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("quad", np.array([[0, 1, 3, 2]])),
        ],
    )
    assert len(poly.element_types) == 2

    tris_only = filter_element_type(poly, keep="triangle")
    assert len(tris_only.element_types) == 1

    from polyxios._element_types import ELEMENT_TYPES

    assert tris_only.element_types[0] == ELEMENT_TYPES["triangle"]


def test_extract_surface_two_tets_shared_face() -> None:
    # 2 tets sharing face (0,1,2): 4 unique faces each, minus 1 shared = 6 boundary
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
    )
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3], [0, 1, 2, 4]]))])
    surf = extract_surface(poly)
    assert len(surf.element_types) == 6
    from polyxios._element_types import ELEMENT_TYPES

    assert all(int(t) == ELEMENT_TYPES["triangle"] for t in surf.element_types)
    assert surf.faces is not None
    assert surf.faces.shape == (6, 3)


def test_extract_surface_single_hex() -> None:
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))])
    surf = extract_surface(poly)
    assert len(surf.element_types) == 6
    from polyxios._element_types import ELEMENT_TYPES

    assert all(int(t) == ELEMENT_TYPES["quad"] for t in surf.element_types)


def test_extract_surface_wedge() -> None:
    # 1 wedge: 2 tri faces + 3 quad faces = 5 boundary faces
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("wedge", np.array([[0, 1, 2, 3, 4, 5]]))])
    surf = extract_surface(poly)
    assert len(surf.element_types) == 5
    from polyxios._element_types import ELEMENT_TYPES

    n_tri = sum(1 for t in surf.element_types if int(t) == ELEMENT_TYPES["triangle"])
    n_quad = sum(1 for t in surf.element_types if int(t) == ELEMENT_TYPES["quad"])
    assert n_tri == 2
    assert n_quad == 3


def test_extract_surface_skips_surface_elements() -> None:
    # Mixed mesh: 1 surface tri + 1 tet. extract_surface ignores the tri.
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    surf = extract_surface(poly)
    # tet has 4 faces; one of them (0,1,2) matches the surface tri vertex set
    # but the surface tri is NOT a volumetric element — boundary detection only
    # counts volumetric-element faces, so face (0,1,2) appears once → boundary
    assert len(surf.element_types) == 4


def test_extract_surface_pipeline_composable() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    fn = pipeline(extract_surface, remove_orphan_vertices)
    result = fn(poly)
    assert len(result.element_types) == 4
    assert result.vertices.shape[0] == 4  # all 4 verts are on boundary


def test_extract_surface_corpus_ball() -> None:
    import os

    path = os.path.expanduser("~/.polyxios/vtk/ball.vtk")
    if not os.path.exists(path):
        import pytest

        pytest.skip("ball.vtk not in local cache")
    import polyxios

    poly = polyxios.read(path)
    assert poly.faces is None  # pure tet mesh
    surf = extract_surface(poly)
    assert len(surf.element_types) > 0
    assert surf.faces is not None


def test_merge_fills_missing_multichannel_attr() -> None:
    """A missing (n, k) attribute needs a placeholder of matching width."""
    a = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": np.ones((3, 3))},
    )
    b = make_polydata(np.zeros((3, 3)), [("triangle", np.array([[0, 1, 2]]))])
    merged = transforms.merge(a, b)
    assert merged.vertex_attrs["normals"].shape == (6, 3)
    np.testing.assert_allclose(merged.vertex_attrs["normals"][:3], 1.0)
    assert np.isnan(merged.vertex_attrs["normals"][3:]).all()


def test_merge_fills_missing_multichannel_element_attr() -> None:
    a = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"colors": np.ones((1, 4))},
    )
    b = make_polydata(np.zeros((3, 3)), [("triangle", np.array([[0, 1, 2]]))])
    merged = transforms.merge(a, b)
    assert merged.element_attrs["colors"].shape == (2, 4)


def test_merge_rejects_mismatched_channel_counts() -> None:
    a = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"colors": np.ones((3, 3))},
    )
    b = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"colors": np.ones((3, 4))},
    )
    with pytest.raises(ValueError, match="channel counts differ"):
        transforms.merge(a, b)


def test_vertex_colors_prefers_the_colors_attribute() -> None:
    """Normals are the same shape as an RGB triple; they are not colours."""
    poly = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "normals": np.tile([0.0, 0.0, -1.0], (3, 1)),
            "colors": np.tile([1.0, 0.0, 0.0], (3, 1)),
        },
    )
    np.testing.assert_allclose(transforms.vertex_colors(poly), [[1, 0, 0]] * 3)


def test_vertex_colors_skips_attrs_with_negative_values() -> None:
    poly = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": np.tile([0.0, 0.0, -1.0], (3, 1))},
    )
    assert transforms.vertex_colors(poly) is None


def test_vertex_colors_still_falls_back_to_an_unnamed_attr() -> None:
    poly = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"rgb": np.tile([255.0, 0.0, 0.0], (3, 1))},
    )
    np.testing.assert_allclose(transforms.vertex_colors(poly), [[1, 0, 0]] * 3)


def test_merge_duplicate_vertices_welds_coincident_points() -> None:
    """Two triangles written corner-by-corner share an edge once welded."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2], [3, 4, 5]]))])

    result = merge_duplicate_vertices(poly)

    assert result.vertices.shape[0] == 4
    np.testing.assert_allclose(result.vertices, verts[[0, 1, 2, 5]])
    np.testing.assert_array_equal(
        result.connectivity.reshape(2, 3), np.array([[0, 1, 2], [1, 2, 3]])
    )
    assert result.connectivity.dtype == poly.connectivity.dtype


def test_merge_duplicate_vertices_keeps_a_clean_mesh_untouched() -> None:
    poly = _tri_mesh()
    result = merge_duplicate_vertices(poly)
    np.testing.assert_allclose(result.vertices, poly.vertices)
    np.testing.assert_array_equal(result.connectivity, poly.connectivity)


def test_merge_duplicate_vertices_keeps_the_first_of_each_group() -> None:
    """The survivor is the lowest original index, so vertex order is stable."""
    verts = np.array(
        [[2, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("line", np.array([[0, 1], [2, 3], [4, 0]]))])

    result = merge_duplicate_vertices(poly)

    np.testing.assert_allclose(result.vertices, verts[[0, 1, 3]])
    np.testing.assert_array_equal(
        result.connectivity.reshape(3, 2), np.array([[0, 1], [0, 2], [1, 0]])
    )


def test_merge_duplicate_vertices_is_order_independent() -> None:
    """Welding must not depend on the order the duplicates arrive in."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("line", np.array([[0, 1], [2, 3], [4, 1]]))])
    result = merge_duplicate_vertices(poly)
    assert result.vertices.shape[0] == 2
    np.testing.assert_array_equal(
        result.connectivity.reshape(3, 2), np.array([[0, 1], [0, 1], [0, 1]])
    )


def test_merge_duplicate_vertices_tolerance() -> None:
    """Points within the tolerance weld; points outside it survive apart."""
    verts = np.array(
        [[0, 0, 0], [1e-9, 0, 0], [1, 0, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])

    assert merge_duplicate_vertices(poly).vertices.shape[0] == 3
    welded = merge_duplicate_vertices(poly, tol=1e-6)
    assert welded.vertices.shape[0] == 2
    np.testing.assert_allclose(welded.vertices, verts[[0, 2]])


def test_merge_duplicate_vertices_keeps_the_original_coordinates() -> None:
    """A tolerance decides who merges; it must not move the survivor."""
    verts = np.array([[0.3333333, 0, 0], [0.3333334, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    result = merge_duplicate_vertices(poly, tol=1e-3)
    np.testing.assert_allclose(result.vertices, verts[[0]])


def test_merge_duplicate_vertices_rejects_a_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tol"):
        merge_duplicate_vertices(_tri_mesh(), tol=-1.0)


def test_merge_duplicate_vertices_carries_attrs_and_tags() -> None:
    """The survivor keeps its own attributes, and tags follow the remap."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 3], [2, 1, 3]]))],
        vertex_attrs={"temperature": np.array([1.0, 2.0, 3.0, 4.0])},
        vertex_tags={"inlet": np.array([0, 2], dtype=np.int32)},
        element_attrs={"material": np.array([7, 8], dtype=np.int32)},
    )

    result = merge_duplicate_vertices(poly)

    np.testing.assert_allclose(result.vertex_attrs["temperature"], [1.0, 2.0, 4.0])
    # 0 and 2 weld into one vertex, so the tag names it once.
    np.testing.assert_array_equal(result.vertex_tags["inlet"], [0])
    np.testing.assert_array_equal(result.element_attrs["material"], [7, 8])
    assert result.vertex_tags["inlet"].dtype == poly.vertex_tags["inlet"].dtype


def test_merge_duplicate_vertices_of_an_empty_mesh() -> None:
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    assert merge_duplicate_vertices(poly).vertices.shape[0] == 0


def test_merge_duplicate_vertices_leaves_orphans_alone() -> None:
    """Welding is not culling - an unreferenced vertex is not a duplicate."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    assert merge_duplicate_vertices(poly).vertices.shape[0] == 4


def test_welding_a_facet_soup_makes_it_a_surface(tmp_path) -> None:
    """The STL corner-per-facet soup becomes a connected mesh once welded."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2], [3, 4, 5]]))])
    welded = merge_duplicate_vertices(poly)
    assert welded.vertices.shape[0] == 4
    assert welded.topological_dimension == 2


def test_merge_duplicate_vertices_rejects_a_tolerance_that_overflows() -> None:
    """A tol that snaps a finite coordinate to infinity would weld everything."""
    verts = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    with pytest.raises(ValueError, match="overflows"):
        merge_duplicate_vertices(poly, tol=1e-310)


def test_merge_duplicate_vertices_welds_infinite_coordinates() -> None:
    """A tolerance must not turn a mesh's own infinities into an error."""
    verts = np.array([[np.inf, 0, 0], [np.inf, 0, 0], [1, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    assert merge_duplicate_vertices(poly, tol=1e-6).vertices.shape[0] == 2


@pytest.mark.parametrize(
    "members",
    [
        np.array([-1, 0], dtype=np.int32),
        np.array([0, 99], dtype=np.int32),
        np.array([0.0, 2.0]),
    ],
    ids=["negative", "past-the-end", "float"],
)
def test_transforms_drop_tag_members_that_index_nothing(
    members: np.ndarray,
) -> None:
    """A stray tag member is dropped, never wrapped onto another vertex."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [9, 9, 9]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1]]))],
        vertex_tags={"inlet": members},
    )

    welded = merge_duplicate_vertices(poly).vertex_tags["inlet"]
    compacted = remove_orphan_vertices(poly).vertex_tags["inlet"]

    # Index 0 is the only member that names a vertex of this mesh; -1 must not
    # reach the last one and 99 must not reach anything at all.
    assert welded.tolist() in ([], [0])
    assert compacted.tolist() in ([], [0])


def test_filter_element_type_drops_stray_tag_members() -> None:
    """The same holds for element tags crossing a filter."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1]])), ("triangle", np.array([[0, 1, 2]]))],
        element_tags={"walls": np.array([-1, 1], dtype=np.int32)},
    )
    kept = filter_element_type(poly, keep="triangle").element_tags["walls"]
    np.testing.assert_array_equal(kept, [0])


def test_float_tag_members_do_not_raise() -> None:
    """A float group indexes nothing; it is dropped, not raised on."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [9, 9, 9]], dtype=np.float64),
        [("line", np.array([[0, 1]]))],
        vertex_tags={"inlet": np.array([0.0, 1.0])},
    )
    assert remove_orphan_vertices(poly).vertex_tags["inlet"].size == 0


# One reference element per volume type, its nodes in polyxios's own order, so
# a face's ring can be wound against the solid it came off rather than against
# a second copy of the table it is being checked.
_REFERENCE_SOLIDS: dict[str, np.ndarray] = {
    "tetra": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    "hexahedron": np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    ),
    "voxel": np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    ),
    "wedge": np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype=np.float64,
    ),
    "pyramid": np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float64,
    ),
    "pentagonal_prism": np.array(
        [
            [np.cos(a), np.sin(a), z]
            for z in (0.0, 1.0)
            for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)
        ],
        dtype=np.float64,
    ),
    "hexagonal_prism": np.array(
        [
            [np.cos(a), np.sin(a), z]
            for z in (0.0, 1.0)
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ],
        dtype=np.float64,
    ),
}


def _points_outward(solid: np.ndarray, ring: tuple[int, ...]) -> bool:
    """Say whether a face's ring is wound so its normal leaves the solid."""
    face = solid[list(ring)]
    normal = sum(
        (np.cross(face[i], face[(i + 1) % len(face)]) for i in range(len(face))),
        np.zeros(3),
    )
    return bool(np.dot(normal, face.mean(axis=0) - solid.mean(axis=0)) > 0)


@pytest.mark.parametrize("type_name", sorted(_REFERENCE_SOLIDS))
def test_every_face_of_a_solid_is_wound_outward(type_name: str) -> None:
    """A skin is shaded by its normals, and every base among these was wound
    the other way: five sides lit from outside and one from within, which is
    the shape that reads as a hole in the mesh."""
    from polyxios._element_types import ELEMENT_FACES

    solid = _REFERENCE_SOLIDS[type_name]
    inward = [
        (index, ring)
        for index, ring in enumerate(ELEMENT_FACES[type_name])
        if not _points_outward(solid, ring)
    ]

    assert not inward
