"""Translation, rotation and scale of a node matrix, shared by scene codecs.

glTF animates a node by translation, quaternion rotation and scale; COLLADA
animates the transform elements a node is spelled with, often one
``<matrix>``. Moving an animation from one to the other splits each matrix
key into the three, the same way whichever codec does it, and composes
COLLADA's elements into the matrix they spell the same way too.
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
    translation, rotation, scale, exact = trs_of_matrices(
        np.asarray(matrix, dtype=np.float64).reshape(1, 4, 4)
    )
    return translation[0], rotation[0], scale[0], bool(exact[0])


def _quats_of_rotations(r: np.ndarray) -> np.ndarray:
    """Return the unit quaternions of ``(n, 3, 3)`` rotations, as ``(n, 4)``.

    The branch each rotation takes is the one :func:`quat_of_rotation`
    takes, so the two agree on every rotation.
    """
    r00, r11, r22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    trace = r00 + r11 + r22
    q = np.empty((len(r), 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = 0.5 / np.sqrt(trace + 1.0)
        by_w = np.stack(
            [
                (r[:, 2, 1] - r[:, 1, 2]) * k,
                (r[:, 0, 2] - r[:, 2, 0]) * k,
                (r[:, 1, 0] - r[:, 0, 1]) * k,
                0.25 / k,
            ],
            axis=1,
        )
        k = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
        by_x = np.stack(
            [
                0.25 * k,
                (r[:, 0, 1] + r[:, 1, 0]) / k,
                (r[:, 0, 2] + r[:, 2, 0]) / k,
                (r[:, 2, 1] - r[:, 1, 2]) / k,
            ],
            axis=1,
        )
        k = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
        by_y = np.stack(
            [
                (r[:, 0, 1] + r[:, 1, 0]) / k,
                0.25 * k,
                (r[:, 1, 2] + r[:, 2, 1]) / k,
                (r[:, 0, 2] - r[:, 2, 0]) / k,
            ],
            axis=1,
        )
        k = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
        by_z = np.stack(
            [
                (r[:, 0, 2] + r[:, 2, 0]) / k,
                (r[:, 1, 2] + r[:, 2, 1]) / k,
                0.25 * k,
                (r[:, 1, 0] - r[:, 0, 1]) / k,
            ],
            axis=1,
        )
    w = trace > 0
    x = ~w & (r00 > r11) & (r00 > r22)
    y = ~w & ~x & (r11 > r22)
    z = ~(w | x | y)
    for mask, value in ((w, by_w), (x, by_x), (y, by_y), (z, by_z)):
        q[mask] = value[mask]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def trs_of_matrices(
    matrices: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return node matrices as translations, quaternions and scales.

    :func:`trs_of_matrix` for a stack of matrices at once, with the same
    results.

    Parameters
    ----------
    matrices
        ``(n, 4, 4)`` row-major transforms, or ``(n, 16)`` values.

    Returns
    -------
    translation : np.ndarray
        ``(n, 3)`` translations.
    rotation : np.ndarray
        ``(n, 4)`` unit quaternions ``(x, y, z, w)``.
    scale : np.ndarray
        ``(n, 3)`` scale factors.
    exact : np.ndarray
        ``(n,)`` booleans, whether each key rebuilds its matrix to about
        six significant digits.
    """
    m = np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4)
    if not len(m):
        empty = np.empty((0, 3))
        return empty, np.empty((0, 4)), empty.copy(), np.empty(0, dtype=bool)
    basis = m[:, :3, :3]
    scale = np.linalg.norm(basis, axis=1)
    scale[:, 0] = np.where(np.linalg.det(basis) < 0, -scale[:, 0], scale[:, 0])
    unit = basis / np.where(scale != 0, scale, 1.0)[:, None, :]
    u, _, vt = np.linalg.svd(unit)
    flip = np.linalg.det(u @ vt) < 0
    u[flip, :, -1] = -u[flip, :, -1]
    rot = u @ vt
    tol = _EXACT_TOLERANCE * np.maximum(1.0, np.abs(basis).max(axis=(1, 2)))
    exact = (np.abs(rot * scale[:, None, :] - basis).max(axis=(1, 2)) <= tol) & (
        np.abs(m[:, 3] - (0.0, 0.0, 0.0, 1.0)).max(axis=1) <= _EXACT_TOLERANCE
    )
    return m[:, :3, 3].copy(), _quats_of_rotations(rot), scale, exact


def _rotations(axis: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(axis, axis=1)
    unit = axis / np.where(norm == 0.0, 1.0, norm)[:, None]
    x, y, z = unit.T
    c = np.cos(np.radians(degrees))
    s = np.sin(np.radians(degrees))
    t = 1.0 - c
    out = np.tile(np.eye(4), (len(axis), 1, 1))
    out[:, :3, :3] = np.stack(
        [
            np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], axis=1),
            np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], axis=1),
            np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], axis=1),
        ],
        axis=1,
    )
    out[norm == 0.0] = np.eye(4)
    return out


def _lookats(values: np.ndarray) -> np.ndarray:
    """Return the camera-to-parent matrices of ``<lookat>`` elements.

    An ``up`` parallel to the view direction leaves the roll undefined; the
    world axis least aligned with the view direction stands in for it, so
    the matrix stays a rotation whatever the direction.
    """
    eye, target, up = values[:, :3], values[:, 3:6], values[:, 6:]
    z = eye - target
    zn = np.linalg.norm(z, axis=1)
    z = np.where(
        (zn == 0)[:, None], (0.0, 0.0, 1.0), z / np.where(zn == 0, 1.0, zn)[:, None]
    )
    x = np.cross(up, z)
    xn = np.linalg.norm(x, axis=1)
    lone = xn == 0
    if lone.any():
        axis = np.eye(3)[np.argmin(np.abs(z[lone]), axis=1)]
        x[lone] = np.cross(axis, z[lone])
        xn[lone] = np.linalg.norm(x[lone], axis=1)
    x = x / xn[:, None]
    y = np.cross(z, x)
    out = np.tile(np.eye(4), (len(values), 1, 1))
    out[:, :3, 0], out[:, :3, 1], out[:, :3, 2], out[:, :3, 3] = x, y, z, eye
    return out


def matrices_of_element(*, kind: str, values: np.ndarray) -> np.ndarray:
    """Return the matrices of a COLLADA transform element at several values.

    Parameters
    ----------
    kind
        ``matrix``, ``translate``, ``rotate``, ``scale`` or ``lookat``.
    values
        ``(n, k)`` values of the element, one row per matrix: 16 row-major
        for a matrix, three for a translate or scale, an axis and an angle
        in degrees for a rotate, and eye, target and up for a lookat.

    Returns
    -------
    np.ndarray
        ``(n, 4, 4)`` row-major float64 transforms.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if kind == "matrix":
        return values.reshape(n, 4, 4).copy()
    if kind == "rotate":
        return _rotations(values[:, :3], values[:, 3])
    if kind == "lookat":
        return _lookats(values)
    out = np.tile(np.eye(4), (n, 1, 1))
    if kind == "translate":
        out[:, :3, 3] = values
    else:
        out[:, [0, 1, 2], [0, 1, 2]] = values
    return out


def matrix_of_element(*, kind: str, values: np.ndarray) -> np.ndarray:
    """Return the matrix of one COLLADA transform element.

    Parameters
    ----------
    kind
        ``matrix``, ``translate``, ``rotate``, ``scale`` or ``lookat``.
    values
        The element's values: 16 row-major for a matrix, three for a
        translate or scale, an axis and an angle in degrees for a rotate,
        and eye, target and up for a lookat.

    Returns
    -------
    np.ndarray
        The 4x4 row-major float64 transform.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(1, -1)
    return matrices_of_element(kind=kind, values=flat)[0]
