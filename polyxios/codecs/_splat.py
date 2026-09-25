from typing import Any

import numpy as np

from polyxios._io import Source, open_block, source_name, write_bytes
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


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a binary 3D Gaussian Splat file (.splat) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .splat file.
    lazy
        Map the file and hand back arrays that view it, read-only. The file
        is a run of 32-byte records, so the positions are a strided
        ``float32`` view three columns wide and every attribute a strided
        view of its own column; nothing is decoded or copied. Needs a path
        or a handle over a regular file at its start.

    Returns
    -------
    PolyData
        Point-cloud PolyData with E=0.  Per-Gaussian attributes are stored
        in vertex_attrs: scale_0/1/2, color_r/g/b, opacity, rot_0/1/2/3.
        Eagerly the positions are float64; lazily they keep the file's
        float32.

    Raises
    ------
    CodecError
        If the file size is not a multiple of 32 bytes.
    LazyReadError
        If ``lazy`` is set and the source cannot be mapped.
    """
    # The size is the length of what was read, not a separate measurement of
    # the source: measuring a compressed one costs a whole decompression pass
    # that the read about to follow would only repeat, and a stream that
    # cannot seek can be read even though it cannot be measured.
    with open_block(path, fmt=EXTENSION, require_map=lazy) as data:
        file_size = len(data)

        if file_size % 32 != 0:
            raise CodecError(
                f"'{source_name(path)}' has size {file_size} bytes which is not a "
                "multiple of 32. Not a valid .splat file."
            )

        n_splats = file_size // 32
        validate_header(n_splats, 0, 0, file_size)

        raw = np.frombuffer(data, dtype=_SPLAT_DTYPE)

        if lazy:
            # x, y, z are the record's first three fields, so a float32 view
            # of the record stream, one row per record, is the coordinates.
            vertices = np.ndarray(
                (n_splats, 3),
                dtype="<f4",
                buffer=data,
                offset=0,
                strides=(_SPLAT_DTYPE.itemsize, 4),
            )
            vertex_attrs = {name: raw[name] for name in _ATTR_NAMES}
        else:
            vertices = np.column_stack(
                [
                    raw["x"].astype(np.float64),
                    raw["y"].astype(np.float64),
                    raw["z"].astype(np.float64),
                ]
            )
            vertex_attrs = {name: np.array(raw[name]) for name in _ATTR_NAMES}
            # The record view has to go before the block does: a mapping
            # cannot close while an array still points into it.
            del raw

    return PolyData(
        vertices=vertices,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs={},
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
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
    n = poly.vertices.shape[0]

    out = np.zeros(n, dtype=_SPLAT_DTYPE)
    out["x"] = poly.vertices[:, 0].astype("<f4")
    out["y"] = poly.vertices[:, 1].astype("<f4")
    out["z"] = poly.vertices[:, 2].astype("<f4")

    for name in _ATTR_NAMES:
        if name in poly.vertex_attrs:
            out[name] = poly.vertex_attrs[name].astype(_SPLAT_DTYPE[name])

    write_bytes(path, out.tobytes())
