from pathlib import Path
from typing import Any

import numpy as np

from polyxios._types import PolyData
from polyxios.exceptions import CodecError
from polyxios.validate import validate_header

EXTENSION: str = ".splat"

# 32-byte per-Gaussian binary layout used by the WebGL Gaussian Splat Viewer
# (antimatter15 / Kevin Kwok) and compatible tools.
_SPLAT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("color_r", "u1"),
        ("color_g", "u1"),
        ("color_b", "u1"),
        ("opacity", "u1"),
        ("rot_0", "u1"),
        ("rot_1", "u1"),
        ("rot_2", "u1"),
        ("rot_3", "u1"),
    ]
)

assert _SPLAT_DTYPE.itemsize == 32, "SPLAT record must be exactly 32 bytes"

_ATTR_NAMES = (
    "scale_0",
    "scale_1",
    "scale_2",
    "color_r",
    "color_g",
    "color_b",
    "opacity",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a binary 3D Gaussian Splat file (.splat) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .splat file.
    lazy
        Ignored; .splat files are always read eagerly (they are small
        flat binary arrays with no seekable structure).

    Returns
    -------
    PolyData
        Point-cloud PolyData with E=0.  Per-Gaussian attributes are stored
        in vertex_attrs: scale_0/1/2, color_r/g/b, opacity, rot_0/1/2/3.

    Raises
    ------
    CodecError
        If the file size is not a multiple of 32 bytes.
    """
    path = Path(path)
    file_size = path.stat().st_size

    if file_size % 32 != 0:
        raise CodecError(
            f"'{path.name}' has size {file_size} bytes which is not a multiple "
            "of 32. Not a valid .splat file."
        )

    n_splats = file_size // 32
    validate_header(n_splats, 0, 0, file_size)

    raw = np.frombuffer(path.read_bytes(), dtype=_SPLAT_DTYPE)

    vertices = np.column_stack(
        [
            raw["x"].astype(np.float64),
            raw["y"].astype(np.float64),
            raw["z"].astype(np.float64),
        ]
    )

    vertex_attrs = {name: np.array(raw[name]) for name in _ATTR_NAMES}

    return PolyData(
        vertices=vertices,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs={},
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise a Gaussian-splat PolyData to a .splat binary file.

    Parameters
    ----------
    poly
        PolyData to write.  Must have vertex_attrs containing at minimum
        ``scale_0``, ``scale_1``, ``scale_2``, ``color_r``, ``color_g``,
        ``color_b``, ``opacity``, ``rot_0``, ``rot_1``, ``rot_2``, ``rot_3``.
        Missing attributes are written as zeros.
    path
        Output file path.
    """
    path = Path(path)
    n = poly.vertices.shape[0]

    out = np.zeros(n, dtype=_SPLAT_DTYPE)
    out["x"] = poly.vertices[:, 0].astype("<f4")
    out["y"] = poly.vertices[:, 1].astype("<f4")
    out["z"] = poly.vertices[:, 2].astype("<f4")

    for name in _ATTR_NAMES:
        if name in poly.vertex_attrs:
            out[name] = poly.vertex_attrs[name].astype(_SPLAT_DTYPE[name])

    path.write_bytes(out.tobytes())
