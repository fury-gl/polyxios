from __future__ import annotations

import numpy as np

from polyxios import make_polydata
from polyxios.transforms import (
    extract_surface,
    filter_element_type,
    merge,
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
