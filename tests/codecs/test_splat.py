from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios._types import PolyData
from polyxios.codecs._splat import read, write
from polyxios.exceptions import CodecError


def _synthetic_splats(n: int = 4) -> PolyData:
    rng = np.random.default_rng(0)
    verts = rng.uniform(-1, 1, (n, 3)).astype(np.float64)
    return PolyData(
        vertices=verts,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs={
            "scale_0": rng.uniform(0, 1, n).astype("<f4"),
            "scale_1": rng.uniform(0, 1, n).astype("<f4"),
            "scale_2": rng.uniform(0, 1, n).astype("<f4"),
            "color_r": rng.integers(0, 255, n, dtype="u1"),
            "color_g": rng.integers(0, 255, n, dtype="u1"),
            "color_b": rng.integers(0, 255, n, dtype="u1"),
            "opacity": rng.integers(0, 255, n, dtype="u1"),
            "rot_0": rng.integers(0, 255, n, dtype="u1"),
            "rot_1": rng.integers(0, 255, n, dtype="u1"),
            "rot_2": rng.integers(0, 255, n, dtype="u1"),
            "rot_3": rng.integers(0, 255, n, dtype="u1"),
        },
        element_attrs={},
    )


def test_roundtrip() -> None:
    poly = _synthetic_splats(8)
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    for name in ("scale_0", "scale_1", "scale_2"):
        np.testing.assert_allclose(
            poly2.vertex_attrs[name], poly.vertex_attrs[name], atol=1e-6
        )
    for name in (
        "color_r",
        "color_g",
        "color_b",
        "opacity",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ):
        np.testing.assert_array_equal(poly2.vertex_attrs[name], poly.vertex_attrs[name])


def test_file_size_32_bytes_per_splat() -> None:
    poly = _synthetic_splats(5)
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    import os

    assert os.path.getsize(tmp) == 5 * 32


def test_read_wrong_size_raises() -> None:
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        f.write(b"\x00" * 33)  # not a multiple of 32
        tmp = f.name
    with pytest.raises(CodecError):
        read(tmp)


def test_missing_attrs_written_as_zeros() -> None:
    """PolyData with no vertex_attrs writes zeros for all attribute fields."""
    verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([], dtype=np.int32),
        offsets=np.array([0], dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
        vertex_attrs={},
        element_attrs={},
    )
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, verts, atol=1e-6)
    assert np.all(poly2.vertex_attrs["opacity"] == 0)
    assert np.all(poly2.vertex_attrs["scale_0"] == 0.0)


def test_polyxios_read_dispatch() -> None:
    """polyxios.read() must dispatch .splat files to the SPLAT codec."""
    import polyxios

    poly = _synthetic_splats(3)
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = polyxios.read(tmp)
    assert len(poly2.vertices) == 3
    assert len(poly2.element_types) == 0
    assert "opacity" in poly2.vertex_attrs
