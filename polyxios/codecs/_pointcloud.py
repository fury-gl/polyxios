"""What the point-cloud codecs share: colour bytes, float formats, a cloud."""

from __future__ import annotations

from typing import Any

import numpy as np

from polyxios._types import PolyData
from polyxios.exceptions import CodecError

BYTE_SCALE: float = 255.0


def color_bytes(colors: np.ndarray) -> np.ndarray:
    """Colour channels as 0..255, from floats in 0..1 or integers already there.

    Parameters
    ----------
    colors
        The channels, any shape. An integer column counts 0..255 the way
        every image format does, the convention the colour-carrying codecs
        share; a float one runs 0..1. NaN is black.

    Returns
    -------
    numpy.ndarray
        float64 of the same shape, every value clipped to 0..255 and, for a
        float input, rounded to a whole number.
    """
    if colors.dtype.kind in "iub":
        return np.clip(colors.astype(np.float64), 0, BYTE_SCALE)
    return np.clip(
        np.rint(np.nan_to_num(colors.astype(np.float64)) * BYTE_SCALE), 0, BYTE_SCALE
    )


def check_float_fmt(float_fmt: str, *, fmt: str) -> None:
    """Refuse a format spec ``format()`` cannot apply to a float, up front.

    Parameters
    ----------
    float_fmt
        The spec every float in an ascii body is written with.
    fmt
        The format's name for the message, ``".pcd"`` or ``".xyz"``.

    Raises
    ------
    CodecError
        When ``format(0.0, float_fmt)`` does not.
    """
    try:
        format(0.0, float_fmt)
    except (TypeError, ValueError) as exc:
        raise CodecError(f"{fmt}: float_fmt {float_fmt!r} is not usable.") from exc


def point_cloud(
    *,
    vertices: np.ndarray,
    vertex_attrs: dict[str, np.ndarray],
    vertex_tags: dict[str, np.ndarray] | None = None,
    global_attrs: dict[str, Any] | None = None,
) -> PolyData:
    """A PolyData of points and nothing else.

    Parameters
    ----------
    vertices
        The ``(n, 3)`` coordinates, kept as given so a lazy view stays one.
    vertex_attrs
        Per-point attributes.
    vertex_tags
        Named index subsets of the points.
    global_attrs
        Whole-cloud metadata.

    Returns
    -------
    PolyData
        No elements: empty connectivity, a single zero offset, no types.
    """
    return PolyData(
        vertices=vertices,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs={},
        vertex_tags=vertex_tags or {},
        global_attrs=global_attrs or {},
    )
