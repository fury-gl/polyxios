"""Translation, rotation and scale of a node matrix, shared by scene codecs.

glTF animates a node by translation, quaternion rotation and scale; COLLADA
animates the transform elements a node is spelled with, often one
``<matrix>``. Moving an animation from one to the other splits each matrix
key into the three, the same way whichever codec does it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Relative to the matrix's largest basis entry.
_EXACT_TOLERANCE: float = 1e-6


def quat_of_rotation(r: np.ndarray) -> np.ndarray:
    """Return the unit quaternion of a rotation matrix.

    Parameters
    ----------
    r
        A 3x3 rotation matrix.

    Returns
    -------
    np.ndarray
        The quaternion as ``(x, y, z, w)``, of unit norm.
    """
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        k = 0.5 / np.sqrt(trace + 1.0)
        q = [
            (r[2, 1] - r[1, 2]) * k,
            (r[0, 2] - r[2, 0]) * k,
            (r[1, 0] - r[0, 1]) * k,
            0.25 / k,
        ]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        k = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [
            0.25 * k,
            (r[0, 1] + r[1, 0]) / k,
            (r[0, 2] + r[2, 0]) / k,
            (r[2, 1] - r[1, 2]) / k,
        ]
    elif r[1, 1] > r[2, 2]:
        k = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [
            (r[0, 1] + r[1, 0]) / k,
            0.25 * k,
            (r[1, 2] + r[2, 1]) / k,
            (r[0, 2] - r[2, 0]) / k,
        ]
    else:
        k = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [
            (r[0, 2] + r[2, 0]) / k,
            (r[1, 2] + r[2, 1]) / k,
            0.25 * k,
            (r[1, 0] - r[0, 1]) / k,
        ]
    out = np.array(q, dtype=np.float64)
    return out / np.linalg.norm(out)


def matrix_of_trs(*, translation: Any, rotation: Any, scale: Any) -> np.ndarray:
    """Return the node matrix of a translation, a quaternion and a scale.

    Parameters
    ----------
    translation
        The three translation components.
    rotation
        The quaternion ``(x, y, z, w)``.
    scale
        The three scale factors, or one for all three axes.

    Returns
    -------
    np.ndarray
        The 4x4 row-major float64 transform, translation times rotation
        times scale.
    """
    x, y, z, w = rotation
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot * np.asarray(scale, dtype=np.float64)
    m[:3, 3] = translation
    return m


def trs_of_matrix(matrix: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return a node matrix as translation, quaternion and scale.

    A reflection is carried by a negative x scale; shear and projection
    have no place in the three.

    Parameters
    ----------
    matrix
        A 4x4 row-major transform, or its 16 values.

    Returns
    -------
    translation : np.ndarray
        The three translation components.
    rotation : np.ndarray
        The unit quaternion ``(x, y, z, w)``.
    scale : np.ndarray
        The three scale factors.
    exact : bool
        Whether the three rebuild the matrix to about six significant
        digits, False when it holds a shear or a projection.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    basis = m[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    if np.linalg.det(basis) < 0:
        scale[0] = -scale[0]
    u, _, vt = np.linalg.svd(basis / np.where(scale != 0, scale, 1.0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] = -u[:, -1]
    rot = u @ vt
    # Files print six or so digits, or store float32: a tolerance tighter
    # than that would call every such rotation a shear.
    tol = _EXACT_TOLERANCE * max(1.0, float(np.abs(basis).max()))
    exact = bool(
        np.allclose(rot * scale, basis, rtol=0.0, atol=tol)
        and np.allclose(m[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=_EXACT_TOLERANCE)
    )
    return m[:3, 3].copy(), quat_of_rotation(rot), scale, exact
