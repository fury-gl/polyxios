from __future__ import annotations

import numpy as np

from polyxios import make_polydata
from polyxios.transforms import (
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
