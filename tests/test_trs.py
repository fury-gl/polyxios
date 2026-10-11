"""Translation, rotation and scale of a node matrix."""

import numpy as np
import pytest

from polyxios._trs import (
    matrices_of_element,
    matrix_of_element,
    matrix_of_trs,
    quat_of_rotation,
    trs_of_matrices,
    trs_of_matrix,
)


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


def test_trs_of_matrices_takes_each_branch_as_one_matrix_does() -> None:
    """Every quaternion branch, a reflection, a shear and a projection."""
    rng = np.random.default_rng(3)
    stack = [_compose(rng.normal(size=3), q, [1.0, 2.0, 0.5]) for q in np.eye(4)]
    stack.append(np.diag([-1.0, 2.0, 3.0, 1.0]))
    shear = np.eye(4)
    shear[0, 1] = 0.5
    projection = np.eye(4)
    projection[3, 2] = 0.1
    stack += [shear, projection]
    for _ in range(20):
        q = rng.normal(size=4)
        stack.append(_compose(rng.normal(size=3), q / np.linalg.norm(q), [1, 1, 1]))
    move, quats, size, exact = trs_of_matrices(np.stack(stack))
    for k, m in enumerate(stack):
        t, q, s, e = trs_of_matrix(m)
        np.testing.assert_array_equal(move[k], t)
        np.testing.assert_allclose(quats[k], q, atol=1e-15)
        np.testing.assert_array_equal(size[k], s)
        assert exact[k] == e
    assert not exact[5] and not exact[6]


def test_trs_of_matrices_of_no_matrix() -> None:
    move, quats, size, exact = trs_of_matrices(np.empty((0, 4, 4)))
    assert move.shape == (0, 3) and quats.shape == (0, 4) and exact.shape == (0,)


@pytest.mark.parametrize(
    ("kind", "values", "expected"),
    [
        ("translate", [1.0, 2.0, 3.0], [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]),
        ("scale", [2.0, 3.0, 4.0], [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0]]),
        ("rotate", [0.0, 0.0, 2.0, 90.0], [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]),
        ("rotate", [0.0, 0.0, 0.0, 90.0], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        (
            "lookat",
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5]],
        ),
    ],
)
def test_matrix_of_element(kind: str, values: list, expected: list) -> None:
    m = matrix_of_element(kind=kind, values=np.asarray(values))
    np.testing.assert_allclose(m[:3], expected, atol=1e-12)
    np.testing.assert_array_equal(m[3], [0.0, 0.0, 0.0, 1.0])


def test_lookat_with_up_along_the_view_stays_a_rotation() -> None:
    values = np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2)
    for m in matrices_of_element(kind="lookat", values=values):
        np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(m[:3, 3], [0.0, 0.0, 5.0])
