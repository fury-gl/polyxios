"""Translation, rotation and scale of a node matrix."""

import numpy as np
import pytest

from polyxios._trs import matrix_of_trs, quat_of_rotation, trs_of_matrix


def _compose(t, q, s) -> np.ndarray:
    """Return the 4x4 matrix of translation ``t``, quaternion ``q`` and scale ``s``."""
    return matrix_of_trs(translation=t, rotation=q, scale=s)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_half_turn_about_each_axis(axis: int) -> None:
    r = -np.eye(3)
    r[axis, axis] = 1.0
    q = quat_of_rotation(r)
    expected = np.zeros(4)
    expected[axis] = 1.0
    np.testing.assert_allclose(np.abs(q), expected, atol=1e-12)


@pytest.mark.parametrize(
    "scale", [(2.0, 3.0, 4.0), (-2.0, 3.0, 4.0), (2.0, -3.0, -4.0), (0.0, 1.0, 1.0)]
)
def test_rebuilds_matrix_with_any_scale(scale) -> None:
    q = np.array([0.1, -0.3, 0.5, 0.8])
    q /= np.linalg.norm(q)
    m = _compose([1.0, 2.0, 3.0], q, np.asarray(scale))
    t, rot, s, exact = trs_of_matrix(m)
    assert exact
    np.testing.assert_allclose(_compose(t, rot, s), m, atol=1e-12)
    assert np.sign(np.prod(s)) == np.sign(np.prod(scale))


def test_rotation_printed_to_six_digits_is_exact() -> None:
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    q = np.r_[axis * np.sin(0.35), np.cos(0.35)]
    m = np.array([float(f"{v:.6g}") for v in _compose([0, 0, 0], q, 1.0).ravel()])
    assert trs_of_matrix(m)[3]
    assert trs_of_matrix(m.astype(np.float32))[3]


def test_shear_and_projection_are_not_exact() -> None:
    shear = np.eye(4)
    shear[0, 1] = 0.5
    assert not trs_of_matrix(shear)[3]
    projection = np.eye(4)
    projection[3, 2] = -1.0
    assert not trs_of_matrix(projection)[3]


def test_matrix_of_trs_places_translation_rotation_and_scale() -> None:
    """A quarter turn about z of a scaled x axis, then moved."""
    half = np.sqrt(0.5)
    m = matrix_of_trs(
        translation=[1.0, 2.0, 3.0], rotation=[0.0, 0.0, half, half], scale=[2, 1, 1]
    )
    np.testing.assert_allclose(m @ [1.0, 0.0, 0.0, 1.0], [1.0, 4.0, 3.0, 1.0])
    assert m.dtype == np.float64
