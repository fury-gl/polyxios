from __future__ import annotations

import numpy as np
import pytest

from polyxios import PolyData, make_polydata, validate
from polyxios.exceptions import ValidationError


def _make_tri_mesh() -> PolyData:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_valid_passes_through() -> None:
    poly = _make_tri_mesh()
    result = validate(poly)
    assert result is poly


def test_bad_dtype_raises() -> None:
    poly = _make_tri_mesh()
    bad_verts = poly.vertices.astype(np.float32)
    import dataclasses

    bad_poly = dataclasses.replace(poly, vertices=bad_verts)
    with pytest.raises(ValidationError, match="float64"):
        validate(bad_poly)


def test_out_of_bounds_raises() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    # connectivity references vertex index 99 which doesn't exist
    connectivity = np.array([0, 1, 99], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int32)
    element_types = np.array([5], dtype=np.uint8)  # triangle
    poly = PolyData(
        vertices=verts,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
    )
    with pytest.raises(ValidationError, match="index"):
        validate(poly)


def test_shape_mismatch_raises() -> None:
    poly = _make_tri_mesh()
    import dataclasses

    # offsets length should be n_elements + 1 = 3, give wrong length
    bad_offsets = np.array([0, 3], dtype=np.int32)  # only 2 elements, need 3
    bad_poly = dataclasses.replace(poly, offsets=bad_offsets)
    with pytest.raises(ValidationError, match="offsets"):
        validate(bad_poly)


def test_element_attrs_length_mismatch_raises() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={
            "pressure": np.array([1.0, 2.0, 3.0])
        },  # 3 values but 2 elements
    )
    with pytest.raises(ValidationError, match="element_attrs"):
        validate(poly)
