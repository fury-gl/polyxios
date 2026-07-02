from __future__ import annotations

import dataclasses
import struct

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._meshb import read, write


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))],
    )


def test_roundtrip_tetra(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_binary_file_has_correct_magic(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    with open(tmp, "rb") as f:
        raw = f.read(8)
    kw, version = struct.unpack("<ii", raw)
    assert kw == 1  # GmfMeshVersionFormatted
    assert version == 2  # float64


def test_ref_not_stored_when_all_zero(tmp_path) -> None:
    """Default (all-zero) refs are not stored to avoid false-positive ref checks."""
    poly = _tri_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert "ref" not in poly2.element_attrs


def test_ref_roundtrip(tmp_path) -> None:
    poly = _tri_mesh()
    refs = np.array([10, 20, 30, 40], dtype=np.int32)
    poly = dataclasses.replace(poly, element_attrs={"ref": refs})
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.element_attrs["ref"], refs)


def test_mixed_elements(tmp_path) -> None:
    """write reorders elements by type (tri < quad < tet < hex); roundtrip preserves content."""
    from polyxios._element_types import ELEMENT_TYPES

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    # Input order: tetra first, then triangle — write will emit triangle section first.
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    # After roundtrip elements are reordered: triangle first, tetra second.
    expected_types = np.array(
        [ELEMENT_TYPES["triangle"], ELEMENT_TYPES["tetra"]], dtype=np.uint8
    )
    np.testing.assert_array_equal(poly2.element_types, expected_types)


def test_vertex_ref_roundtrip(tmp_path) -> None:
    """Nonzero vertex reference tags survive a write/read roundtrip."""
    poly = _tet_mesh()
    poly = dataclasses.replace(
        poly, vertex_attrs={"ref": np.array([1, 2, 3, 4], dtype=np.int32)}
    )
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.vertex_attrs["ref"], [1, 2, 3, 4])


def test_vertex_ref_not_stored_when_all_zero(tmp_path) -> None:
    """All-zero vertex refs (default) are not stored."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    poly2 = read(tmp)
    assert "ref" not in poly2.vertex_attrs


def test_unknown_keyword_warns(tmp_path) -> None:
    """Unknown GmFlib keyword emits UserWarning and stops scan (partial mesh)."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    data = tmp.read_bytes()
    # Inject unknown keyword 999 after the header, before vertex data.
    # Scanner warns and stops, so result has no elements.
    injected = data[:16] + struct.pack("<ii", 999, 0) + data[16:]
    bad = tmp_path / "bad.meshb"
    bad.write_bytes(injected)
    with pytest.warns(UserWarning, match="unknown keyword 999"):
        result = read(bad)
    assert len(result.element_types) == 0


def test_known_skip_keyword_transparent(tmp_path) -> None:
    """GmFlib sections in _SKIP_REC (e.g. GmfCorners=13) are silently skipped."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly, tmp)
    data = tmp.read_bytes()
    # Inject GmfCorners (kw=13, count=1, 1 int32 record) before vertex data.
    injected = data[:16] + struct.pack("<iii", 13, 1, 0) + data[16:]
    patched = tmp_path / "patched.meshb"
    patched.write_bytes(injected)
    poly2 = read(patched)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
