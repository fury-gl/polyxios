"""Write the seed corpus every fuzzer starts from.

Seeds are files polyxios writes itself, a few small meshes in every flavour a
writer has (ASCII and binary, appended and inline, each point format), so the
engine starts on inputs that reach deep into a reader instead of spending its
first hours rediscovering the magic number. Nothing is checked in: the corpus
is rebuilt from the writers, and follows them as they change.

Usage::

    python fuzz/make_corpus.py [OUT_DIR]

``OUT_DIR`` defaults to ``fuzz/corpus``; each target gets ``OUT_DIR/<name>/``.
"""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import struct
import sys
import warnings
import zlib

from _harness import LAZY_TARGETS, TARGETS
import numpy as np

import polyxios
from polyxios import make_polydata
from polyxios._types import PolyData
from polyxios.exceptions import PolyxiosError
from polyxios.helper import array_to_image


def _surface() -> PolyData:
    """Two triangles and a quad, with vertex and element attributes."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], dtype=np.float64
    )
    return make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 4], [1, 2, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
        ],
        vertex_attrs={
            "scalar": np.arange(5, dtype=np.float64),
            "normals": np.tile([0.0, 0.0, 1.0], (5, 1)),
        },
        element_attrs={"eint": np.arange(3, dtype=np.int32)},
        element_tags={"top": np.array([0, 1])},
    )


def _volume() -> PolyData:
    """Two tetrahedra and a hexahedron."""
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [2, 0, 0],
        ],
        dtype=np.float64,
    )
    return make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 3, 4], [1, 8, 2, 5]])),
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        ],
        element_attrs={"efloat": np.array([0.5, 1.5, 2.5])},
    )


def _points() -> PolyData:
    """Six points with colours and an intensity."""
    rng = np.random.default_rng(0)
    verts = rng.random((6, 3))
    return make_polydata(
        verts,
        [("vertex", np.arange(6).reshape(6, 1))],
        vertex_attrs={
            "colors": rng.random((6, 3)),
            "intensity": np.arange(6, dtype=np.float32),
        },
    )


def _lines() -> PolyData:
    """A polyline and a segment."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 2, 2]], dtype=np.float64)
    return make_polydata(verts, [("line", np.array([[0, 1], [1, 2], [2, 3]]))])


def _grid() -> PolyData:
    """A 2x2x2 lattice of hexahedra, the shape .vti, .vtr and .vts hold."""
    axis = np.arange(3.0)
    points = np.array([[i, j, k] for k in axis for j in axis for i in axis])

    def node(i: int, j: int, k: int) -> int:
        return k * 9 + j * 3 + i

    conn = [
        [
            node(i, j, k),
            node(i + 1, j, k),
            node(i + 1, j + 1, k),
            node(i, j + 1, k),
            node(i, j, k + 1),
            node(i + 1, j, k + 1),
            node(i + 1, j + 1, k + 1),
            node(i, j + 1, k + 1),
        ]
        for k in range(2)
        for j in range(2)
        for i in range(2)
    ]
    return make_polydata(
        points,
        [("hexahedron", np.array(conn))],
        vertex_attrs={"scalar": np.arange(27, dtype=np.float64)},
        element_attrs={"eint": np.arange(8, dtype=np.int32)},
    )


def _image() -> PolyData:
    """A 3-wide, 4-tall RGB picture as the lattice of quads an image file expands into."""
    return array_to_image(
        np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
    ).to_polydata()


MESHES = (_surface, _volume, _points, _lines, _grid, _image)

# Writer options worth a seed of their own; every target also gets the
# writer's defaults.
OPTIONS: dict[str, tuple[dict, ...]] = {
    "stl": ({"binary": False},),
    "ply": ({"binary": False}, {"endian": "big"}),
    "vtk": (
        {"binary": True},
        {"vtk_version": "5.1"},
        {"binary": True, "vtk_version": "5.1"},
    ),
    "vtu": ({"binary": False}, {"appended": True}),
    "vtp": ({"binary": False}, {"appended": True}),
    "vti": ({"binary": False}, {"appended": True}),
    "vtr": ({"binary": False}, {"appended": True}),
    "vts": ({"binary": False}, {"appended": True}),
    "pcd": ({"data_format": "ascii"}, {"data_format": "binary_compressed"}),
    "las": tuple({"point_format": k} for k in (0, 1, 2, 3, 6, 7, 8)),
    "laz": ({"point_format": 3}, {"point_format": 7}),
    "spz": ({"version": 2}, {"version": 4}),
    "glb": ({"binary": True},),
    "gltf": ({"binary": False},),
    "amf": ({"compress": True},),
}


def _png_metadata_seed() -> bytes:
    """A 16-bit gray PNG carrying every metadata key the writer spells."""
    image = array_to_image(
        (np.arange(12, dtype=np.uint16) * 5000).reshape(3, 4),
        global_attrs={
            "dpi": (72.0, 72.0),
            "gamma": 0.45455,
            "icc_profile": b"icc",
            "exif": b"II*\x00\x08\x00\x00\x00",
            "text": {"T": "x", "N": "\u65e5"},
        },
    )
    buf = io.BytesIO()
    polyxios.write_image(image, buf, fmt=".png")
    return buf.getvalue()


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    """Frame a PNG chunk with its length and CRC."""
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body))
    )


def _png_row(samples: np.ndarray, depth: int) -> bytes:
    """One filter-0 scanline of ``samples`` packed at ``depth`` bits."""
    flat = np.asarray(samples, np.uint32).ravel()
    if depth == 16:
        return b"\0" + flat.astype(">u2").tobytes()
    if depth == 8:
        return b"\0" + flat.astype(np.uint8).tobytes()
    per_byte = 8 // depth
    padded = np.zeros(-(-flat.size // per_byte) * per_byte, np.uint32)
    padded[: flat.size] = flat
    shifts = (per_byte - 1 - np.arange(per_byte)) * depth
    packed = (padded.reshape(-1, per_byte) << shifts).sum(axis=1)
    return b"\0" + packed.astype(np.uint8).tobytes()


def _png_built(
    samples: np.ndarray,
    depth: int,
    color: int,
    *,
    interlaced: bool,
    before_idat: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    """A PNG laid out by hand, which the writer cannot make: Adam7, palettes."""
    height, width = samples.shape[:2]
    adam7 = (
        (0, 0, 8, 8),
        (0, 4, 8, 8),
        (4, 0, 8, 4),
        (0, 2, 4, 4),
        (2, 0, 4, 2),
        (0, 1, 2, 2),
        (1, 0, 2, 1),
    )
    raw = b""
    for y0, x0, dy, dx in adam7 if interlaced else ((0, 0, 1, 1),):
        block = samples[y0::dy, x0::dx]
        if block.shape[0] and block.shape[1]:
            raw += b"".join(_png_row(row, depth) for row in block)
    header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, int(interlaced))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + b"".join(_png_chunk(kind, body) for kind, body in before_idat)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _png_decoder_seeds() -> tuple[bytes, ...]:
    """PNGs that reach the decoder's hard paths: Adam7, palettes, sub-byte depths.

    Every chunk carries a CRC, so a mutation of a file that has none of these
    stops at the CRC check instead of reaching them.
    """
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 256, (10, 3), np.uint8).tobytes()
    return (
        _png_built(
            rng.integers(0, 10, (11, 9, 1)),
            4,
            3,
            interlaced=True,
            before_idat=((b"PLTE", palette), (b"tRNS", bytes([0, 128, 255]))),
        ),
        _png_built(
            rng.integers(0, 2, (11, 9, 1)),
            1,
            0,
            interlaced=True,
            before_idat=(
                (b"tRNS", struct.pack(">H", 1)),
                (b"zTXt", b"A\0\0" + zlib.compress(b"x")),
                (b"iTXt", b"B\0\x01\0\0\0" + zlib.compress("\u65e5".encode())),
            ),
        ),
        _png_built(rng.integers(0, 256, (5, 7, 2)), 8, 4, interlaced=True),
        _png_built(rng.integers(0, 1 << 16, (3, 4, 4)), 16, 6, interlaced=False),
        _png_built(
            rng.integers(0, 1 << 16, (3, 4, 3)),
            16,
            2,
            interlaced=False,
            before_idat=((b"tRNS", struct.pack(">3H", 1, 2, 3)),),
        ),
    )


# Writers that need a path for a sibling file leave their target seedless, and
# some targets need a seed the meshes above never write; a hand-built seed
# stands in.
HANDWRITTEN: dict[str, tuple[bytes, ...]] = {
    "node": (b"4 3 0 1\n1 0 0 0 1\n2 1 0 0 1\n3 0 1 0 2\n4 0 0 1 2\n",),
    "png": (_png_metadata_seed(), *_png_decoder_seeds()),
}


# A shared extension is seeded by every writer whose files it can hold.
WRITTEN_AS: dict[str, tuple[str, ...]] = {
    "dat": (".bdf", ".post", ".tec"),
}


def _write(poly: PolyData, *, fmt: str, opts: dict) -> bytes | None:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            polyxios.write(poly, buf, fmt=fmt, **opts)
        except (PolyxiosError, TypeError, ValueError):
            return None
    return buf.getvalue()


def main(*, out: Path) -> None:
    """Write every target's seeds under ``out``.

    Parameters
    ----------
    out
        Directory receiving one subdirectory per target.
    """
    for name, fmt in TARGETS.items():
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        seeds = 0
        for build in MESHES:
            for written_as in WRITTEN_AS.get(name, (fmt,)):
                for opts in ({}, *OPTIONS.get(name, ())):
                    data = _write(build(), fmt=written_as, opts=opts)
                    if data:
                        (target / f"{build.__name__[1:]}_{seeds}").write_bytes(data)
                        seeds += 1
        for data in HANDWRITTEN.get(name, ()):
            (target / f"handwritten_{seeds}").write_bytes(data)
            seeds += 1
        print(f"{name:8} {seeds} seeds")
    # A lazy target starts from the same files as the eager one.
    for name in LAZY_TARGETS:
        shutil.copytree(
            out / name.removeprefix("lazy_"), out / name, dirs_exist_ok=True
        )


if __name__ == "__main__":
    main(
        out=Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "corpus"
    )
