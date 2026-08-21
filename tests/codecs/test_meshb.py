from __future__ import annotations

import dataclasses
import struct

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
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
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
    assert len(poly2.element_types) == 4
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


def test_binary_file_has_correct_magic(tmp_path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    with open(tmp, "rb") as f:
        raw = f.read(8)
    kw, version = struct.unpack("<ii", raw)
    assert kw == 1  # GmfMeshVersionFormatted
    assert version == 2  # float64


def test_ref_not_stored_when_all_zero(tmp_path) -> None:
    """Default (all-zero) refs are not stored to avoid false-positive ref checks."""
    poly = _tri_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
    assert "ref" not in poly2.element_attrs


def test_ref_roundtrip(tmp_path) -> None:
    poly = _tri_mesh()
    refs = np.array([10, 20, 30, 40], dtype=np.int32)
    poly = dataclasses.replace(poly, element_attrs={"ref": refs})
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
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
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
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
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
    np.testing.assert_array_equal(poly2.vertex_attrs["ref"], [1, 2, 3, 4])


def test_vertex_ref_not_stored_when_all_zero(tmp_path) -> None:
    """All-zero vertex refs (default) are not stored."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    poly2 = read(path=tmp)
    assert "ref" not in poly2.vertex_attrs


def test_unknown_keyword_warns(tmp_path) -> None:
    """Unknown GmFlib keyword emits UserWarning and stops scan (partial mesh)."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    data = tmp.read_bytes()
    # Inject unknown keyword 999 after the header, before vertex data.
    # Scanner warns and stops, so result has no elements.
    injected = data[:16] + struct.pack("<ii", 999, 0) + data[16:]
    bad = tmp_path / "bad.meshb"
    bad.write_bytes(injected)
    with pytest.warns(UserWarning, match="unknown keyword 999"):
        result = read(path=bad)
    assert len(result.element_types) == 0


def test_known_skip_keyword_transparent(tmp_path) -> None:
    """GmFlib sections in _SKIP_REC (e.g. GmfCorners=13) are silently skipped."""
    poly = _tet_mesh()
    tmp = tmp_path / "mesh.meshb"
    write(poly=poly, path=tmp)
    data = tmp.read_bytes()
    # Inject GmfCorners (kw=13, count=1, 1 int32 record) before vertex data.
    injected = data[:16] + struct.pack("<iii", 13, 1, 0) + data[16:]
    patched = tmp_path / "patched.meshb"
    patched.write_bytes(injected)
    poly2 = read(path=patched)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


# --- meshio #1511, #1508: sections Medit binary can carry --------------------


@pytest.mark.parametrize(
    ("kind", "n_nodes"),
    [
        ("line", 2),
        ("triangle", 3),
        ("quad", 4),
        ("tetra", 4),
        ("pyramid", 5),
        ("wedge", 6),
        ("hexahedron", 8),
    ],
)
def test_issue_1511_every_entity_section_round_trips(
    tmp_path, kind: str, n_nodes: int
) -> None:
    """A section skipped on read is a block of the mesh that vanishes."""
    verts = np.arange(3 * n_nodes, dtype=np.float64).reshape(n_nodes, 3)
    poly = make_polydata(verts, [(kind, np.arange(n_nodes).reshape(1, n_nodes))])
    out = tmp_path / f"{kind}.meshb"
    write(poly=poly, path=out)
    back = read(path=out)
    assert back.element_types.tolist() == [ELEMENT_TYPES[kind]]
    np.testing.assert_array_equal(back.connectivity, np.arange(n_nodes))
    np.testing.assert_allclose(back.vertices, verts)


def test_issue_1508_prisms_and_pyramids_travel_with_the_rest(tmp_path) -> None:
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("pyramid", np.array([[0, 1, 2, 3, 4]])),
            ("wedge", np.array([[0, 1, 2, 3, 4, 5]])),
            ("hexahedron", np.arange(8).reshape(1, 8)),
        ],
    )
    out = tmp_path / "mixed.meshb"
    write(poly=poly, path=out)
    back = read(path=out)
    assert sorted(back.element_types.tolist()) == sorted(poly.element_types.tolist())
    assert back.offsets[-1] == 3 + 5 + 6 + 8


# --- meshio #1258: references are labels ------------------------------------


def test_issue_1258_element_refs_become_tag_groups(tmp_path) -> None:
    """A label kept only as a column of ints does not survive a conversion."""
    verts = np.arange(12, dtype=np.float64).reshape(4, 3)
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 2, 1, 2, 3, 0, 1, 3], dtype=np.int32),
        offsets=np.array([0, 3, 6, 9], dtype=np.int32),
        element_types=np.full(3, ELEMENT_TYPES["triangle"], dtype=np.uint8),
        element_attrs={"ref": np.array([10, 20, 10], dtype=np.int32)},
    )
    out = tmp_path / "refs.meshb"
    write(poly=poly, path=out)
    back = read(path=out)
    np.testing.assert_array_equal(back.element_attrs["ref"], [10, 20, 10])
    np.testing.assert_array_equal(back.element_tags["ref_10"], [0, 2])
    np.testing.assert_array_equal(back.element_tags["ref_20"], [1])


def test_issue_1258_refs_survive_the_trip_to_vtk(tmp_path) -> None:
    """The headline case: a Medit region label readable in a .vtk."""
    import polyxios

    verts = np.arange(12, dtype=np.float64).reshape(4, 3)
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 2, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 3, 6], dtype=np.int32),
        element_types=np.full(2, ELEMENT_TYPES["triangle"], dtype=np.uint8),
        element_attrs={"ref": np.array([10, 20], dtype=np.int32)},
    )
    src = tmp_path / "src.meshb"
    write(poly=poly, path=src)
    out = tmp_path / "out.vtk"
    polyxios.write(polyxios.read(path=src), out)
    np.testing.assert_array_equal(
        polyxios.read(path=out).element_attrs["ref"], [10, 20]
    )


def test_all_zero_refs_invent_no_tag_groups(tmp_path) -> None:
    out = tmp_path / "plain.meshb"
    write(poly=_tri_mesh(), path=out)
    back = read(path=out)
    assert not back.element_tags
    assert "ref" not in back.element_attrs


def test_a_float_ref_column_is_refused_rather_than_truncated(tmp_path) -> None:
    """A reference is a number the file names exactly; rounding one relabels."""
    poly = _tri_mesh()
    n = len(poly.element_types)
    poly.element_attrs["ref"] = np.full(n, 2.7, dtype=np.float64)
    poly.element_tags["ref_5"] = np.arange(n, dtype=np.int32)
    out = tmp_path / "float.meshb"
    with pytest.warns(UserWarning, match="not one integer per element"):
        write(poly=poly, path=out)
    # The tag groups spell the label the float column could not.
    assert set(read(path=out).element_attrs["ref"].tolist()) == {5}


def test_a_tag_group_that_holds_no_indices_is_reported(tmp_path) -> None:
    """A float tag group indexes nothing, so the reference reaches nothing."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.element_tags["ref_7"] = np.array([0.0])
    with pytest.warns(UserWarning, match="do not hold"):
        write(poly=poly, path=tmp_path / "bad_tag.meshb")
