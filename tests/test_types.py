from __future__ import annotations

import numpy as np
import pytest

from polyxios import PolyData, make_polydata
from polyxios._element_types import (
    ELEMENT_TYPES,
    NODES_PER_ELEMENT,
    TOPOLOGICAL_DIMENSION,
)
from polyxios.exceptions import UnknownElementTypeError


def _tri_mesh() -> PolyData:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def _mesh_with_type_code(code: int, dtype: object) -> PolyData:
    """Build a one-element mesh carrying a raw type code, bypassing the names."""
    return PolyData(
        vertices=np.zeros((1, 3), dtype=np.float64),
        connectivity=np.array([0], dtype=np.int32),
        offsets=np.array([0, 1], dtype=np.int32),
        element_types=np.array([code], dtype=dtype),
    )


def test_issue_1551_topological_dimension_of_a_surface() -> None:
    """A triangle mesh is two-dimensional however it is embedded in space."""
    assert _tri_mesh().topological_dimension == 2


def test_issue_1551_topological_dimension_takes_the_maximum() -> None:
    """A mesh carrying its boundary must not be demoted to that boundary."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("vertex", np.array([[0]])),
            ("line", np.array([[0, 1]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    assert poly.topological_dimension == 3


@pytest.mark.parametrize(
    ("type_str", "expected"),
    [
        ("vertex", 0),
        ("poly_vertex", 0),
        ("line", 1),
        ("poly_line", 1),
        ("quadratic_edge", 1),
        ("cubic_line", 1),
        ("bezier_curve", 1),
        ("triangle", 2),
        ("quad", 2),
        ("pixel", 2),
        ("polygon", 2),
        ("triangle_strip", 2),
        ("biquadratic_quad", 2),
        ("lagrange_triangle", 2),
        ("tetra", 3),
        ("hexahedron", 3),
        ("wedge", 3),
        ("pyramid", 3),
        ("voxel", 3),
        ("polyhedron", 3),
        ("triquadratic_hexahedron", 3),
        ("lagrange_hexahedron", 3),
    ],
)
def test_issue_1551_topological_dimension_per_element_type(
    type_str: str, expected: int
) -> None:
    verts = np.zeros((16, 3), dtype=np.float64)
    n_nodes = NODES_PER_ELEMENT[type_str]
    if n_nodes < 0:
        n_nodes = 4
    conn = np.arange(n_nodes, dtype=np.int32).reshape(1, -1)
    poly = make_polydata(verts, [(type_str, conn)])
    assert poly.topological_dimension == expected


def test_issue_1551_every_element_type_has_a_dimension() -> None:
    """A type without an entry would silently read as a point cloud."""
    missing = set(ELEMENT_TYPES.values()) - set(TOPOLOGICAL_DIMENSION)
    assert not missing


def test_issue_1551_topological_dimension_of_an_empty_mesh() -> None:
    """No elements is not an error, and nothing in it rises above a point."""
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    assert poly.topological_dimension == 0


def test_issue_1551_topological_dimension_rejects_an_unknown_code() -> None:
    """An unnamed code must not pass for a point cloud."""
    poly = _mesh_with_type_code(200, np.uint8)
    with pytest.raises(UnknownElementTypeError, match="200"):
        _ = poly.topological_dimension


def test_issue_1551_topological_dimension_rejects_a_code_past_uint8() -> None:
    """A code too wide to be an element type is refused, not wrapped."""
    poly = _mesh_with_type_code(9999, np.int64)
    with pytest.raises(UnknownElementTypeError, match="9999"):
        _ = poly.topological_dimension


def test_issue_1551_topological_dimension_rejects_a_negative_code() -> None:
    """A negative code must not index the table from the far end."""
    poly = _mesh_with_type_code(-3, np.int64)
    with pytest.raises(UnknownElementTypeError, match="-3"):
        _ = poly.topological_dimension
