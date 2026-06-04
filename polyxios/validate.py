import numpy as np

from polyxios._element_types import MAX_SAFE_CONN, MAX_SAFE_ELEMENTS, MAX_SAFE_VERTICES
from polyxios._types import PolyData
from polyxios.exceptions import ValidationError


def validate(poly: PolyData) -> PolyData:
    """Check structural integrity of a PolyData. Raise ValidationError on any violation.

    Parameters
    ----------
    poly
        PolyData to validate.

    Returns
    -------
    PolyData
        The input unchanged if valid (allows use in pipelines).

    Raises
    ------
    ValidationError
        On dtype mismatch, shape mismatch, out-of-bounds indices, or attribute
        length mismatches.
    """
    if poly.vertices.ndim != 2 or poly.vertices.shape[1] != 3:
        raise ValidationError(
            f"vertices must be shape (n, 3), got {poly.vertices.shape}"
        )
    if poly.vertices.dtype != np.float64:
        raise ValidationError(f"vertices must be float64, got {poly.vertices.dtype}")

    n_verts = poly.vertices.shape[0]
    n_elems = poly.element_types.shape[0]

    if poly.element_types.dtype != np.uint8:
        raise ValidationError(
            f"element_types must be uint8, got {poly.element_types.dtype}"
        )
    if poly.element_types.ndim != 1:
        raise ValidationError("element_types must be 1-D")

    if poly.offsets.ndim != 1:
        raise ValidationError("offsets must be 1-D")
    if poly.offsets.shape[0] != n_elems + 1:
        raise ValidationError(
            f"offsets length must be n_elements+1={n_elems + 1}, "
            f"got {poly.offsets.shape[0]}"
        )
    if poly.offsets.dtype not in (np.int32, np.int64):
        raise ValidationError(
            f"offsets must be int32 or int64, got {poly.offsets.dtype}"
        )

    if poly.connectivity.ndim != 1:
        raise ValidationError("connectivity must be 1-D")
    if poly.connectivity.dtype not in (np.int32, np.int64):
        raise ValidationError(
            f"connectivity must be int32 or int64, got {poly.connectivity.dtype}"
        )

    if n_elems > 0 and poly.connectivity.size > 0:
        max_idx = int(poly.connectivity.max())
        if max_idx >= n_verts:
            raise ValidationError(
                f"connectivity contains index {max_idx} but n_verts={n_verts}"
            )
        min_idx = int(poly.connectivity.min())
        if min_idx < 0:
            raise ValidationError(f"connectivity contains negative index {min_idx}")

    for name, arr in poly.vertex_attrs.items():
        if len(arr) != n_verts:
            raise ValidationError(
                f"vertex_attrs['{name}'] length {len(arr)} != n_verts {n_verts}"
            )

    for name, arr in poly.element_attrs.items():
        if len(arr) != n_elems:
            raise ValidationError(
                f"element_attrs['{name}'] length {len(arr)} != n_elements {n_elems}"
            )

    return poly


def validate_header(
    declared_n_verts: int,
    declared_n_elems: int,
    declared_conn_size: int,
    file_size_bytes: int,
) -> None:
    """Validate header counts against file size before any array allocation.

    Parameters
    ----------
    declared_n_verts
        Number of vertices declared in the file header.
    declared_n_elems
        Number of elements declared in the file header.
    declared_conn_size
        Total connectivity size declared in the file header.
    file_size_bytes
        Actual file size in bytes.

    Raises
    ------
    ValidationError
        If declared counts exceed hard caps or are implausible given file size.
    """
    if declared_n_verts > MAX_SAFE_VERTICES:
        raise ValidationError(
            f"declared_n_verts={declared_n_verts} exceeds MAX_SAFE_VERTICES="
            f"{MAX_SAFE_VERTICES}. Possible corrupt or malicious file."
        )
    if declared_n_elems > MAX_SAFE_ELEMENTS:
        raise ValidationError(
            f"declared_n_elems={declared_n_elems} exceeds MAX_SAFE_ELEMENTS="
            f"{MAX_SAFE_ELEMENTS}. Possible corrupt or malicious file."
        )
    if declared_conn_size > MAX_SAFE_CONN:
        raise ValidationError(
            f"declared_conn_size={declared_conn_size} exceeds MAX_SAFE_CONN="
            f"{MAX_SAFE_CONN}. Possible corrupt or malicious file."
        )

    # coords require 3 * 8 bytes per vertex; allow 4× slack for headers/ASCII overhead
    if declared_n_verts * 3 * 8 > file_size_bytes * 4:
        raise ValidationError(
            f"declared_n_verts={declared_n_verts} implies "
            f"{declared_n_verts * 24} bytes of vertex data but "
            f"file_size_bytes={file_size_bytes}. Possible corrupt file."
        )
    # connectivity requires 4 bytes per index; allow 4× slack
    if declared_conn_size * 4 > file_size_bytes * 4:
        raise ValidationError(
            f"declared_conn_size={declared_conn_size} implies "
            f"{declared_conn_size * 4} bytes of index data but "
            f"file_size_bytes={file_size_bytes}. Possible corrupt file."
        )
