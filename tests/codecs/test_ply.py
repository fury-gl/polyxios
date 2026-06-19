from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._ply import read, write
from polyxios.exceptions import LazyReadError


def _synthetic_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_ascii() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    assert len(poly2.element_types) == 2
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_binary() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_lazy() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly_lazy = read(tmp, lazy=True)
    np.testing.assert_allclose(poly_lazy.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_array_equal(poly_lazy.connectivity, poly.connectivity)


def test_vertex_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    nx = np.array([0, 0, 1, 0], dtype=np.float64)
    poly = make_polydata(
        verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))], vertex_attrs={"nx": nx}
    )
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert "nx" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["nx"], nx, atol=1e-6)


def test_element_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    flag = np.array([1.0, 2.0])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"flag": flag},
    )
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    assert "flag" in poly2.element_attrs
    np.testing.assert_allclose(poly2.element_attrs["flag"], flag, atol=1e-6)


def test_ascii_lazy_raises() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_face_scalar_prop_before_list_binary() -> None:
    """Non-list face property declared before vertex_indices must be read correctly."""
    # Construct a binary big-endian PLY with `intensity` before `vertex_indices`
    # (mirrors the layout of Armadillo.ply)
    import struct

    header = (
        b"ply\n"
        b"format binary_big_endian 1.0\n"
        b"element vertex 3\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"element face 1\n"
        b"property uchar intensity\n"
        b"property list uchar int vertex_indices\n"
        b"end_header\n"
    )
    # 3 vertices: (0,0,0), (1,0,0), (0,1,0)
    verts_bytes = (
        struct.pack(">fff", 0, 0, 0)
        + struct.pack(">fff", 1, 0, 0)
        + struct.pack(">fff", 0, 1, 0)
    )
    # 1 face: intensity=42, then 3 indices [0,1,2]
    face_bytes = (
        struct.pack(">B", 42) + struct.pack(">B", 3) + struct.pack(">iii", 0, 1, 2)
    )

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        f.write(header + verts_bytes + face_bytes)
        tmp = f.name

    poly = read(tmp)
    assert len(poly.vertices) == 3
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    assert "intensity" in poly.element_attrs
    assert int(poly.element_attrs["intensity"][0]) == 42


def test_3dgs_ascii_chunk_raises() -> None:
    """Compressed 3DGS PLY in ASCII format raises CodecError."""
    from polyxios.exceptions import CodecError

    header = (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element chunk 1\n"
        b"property float min_x\n"
        b"property float max_x\n"
        b"element vertex 2\n"
        b"property uint packed_position\n"
        b"end_header\n"
        b"0.0 1.0\n"
        b"100\n"
        b"200\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        f.write(header)
        tmp = f.name

    with pytest.raises(CodecError):
        read(tmp)


def test_3dgs_compressed_ply_positions() -> None:
    """Compressed 3DGS PLY returns real world coordinates, not zeros."""
    from polyxios.fetcher import fetch

    path = fetch("gs_Halo_Believe.cleaned.compressed.ply")
    poly = read(path)
    assert len(poly.vertices) == 345217
    assert len(poly.element_types) == 0
    # Positions must not all be zero
    assert not np.allclose(poly.vertices, 0.0)
    # Coords should be within the known scene bbox (roughly -5..5 range)
    assert poly.vertices[:, 0].min() > -20.0
    assert poly.vertices[:, 0].max() < 20.0
    assert "scale_0" in poly.vertex_attrs
    assert "rot_0" in poly.vertex_attrs
    assert "opacity" in poly.vertex_attrs


def test_real_armadillo() -> None:
    """Armadillo.ply: binary big-endian with face scalar before vertex list."""
    from polyxios.fetcher import fetch

    path = fetch("Armadillo.ply")
    poly = read(path)
    assert len(poly.vertices) == 172974
    assert len(poly.element_types) == 345944
    assert poly.faces is not None and len(poly.faces) > 0
    assert "intensity" in poly.element_attrs
