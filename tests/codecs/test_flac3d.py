from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._flac3d import _FACES, _signed_volume, read, write
from polyxios.exceptions import CodecError

# Tetrahedral decomposition of each element, in polyxios (VTK) corner order.
# Deliberately independent of the codec's own face tables so the orientation
# assertions below cannot pass by agreeing with a bug in _FACES.
_TET_DECOMPOSITION: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "tetra": ((0, 1, 2, 3),),
    "pyramid": ((0, 1, 2, 4), (0, 2, 3, 4)),
    "wedge": ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)),
    "hexahedron": (
        (0, 1, 2, 6),
        (0, 1, 6, 5),
        (0, 3, 6, 2),
        (0, 3, 7, 6),
        (0, 4, 5, 6),
        (0, 4, 6, 7),
    ),
}


def _element_volume(poly: PolyData, index: int, name: str) -> float:
    """Six times the signed volume of one element, in polyxios corner order."""
    nodes = poly.connectivity[poly.offsets[index] : poly.offsets[index + 1]]
    p = poly.vertices[np.asarray(nodes)]
    return float(
        sum(
            np.dot(p[b] - p[a], np.cross(p[c] - p[a], p[d] - p[a]))
            for a, b, c, d in _TET_DECOMPOSITION[name]
        )
    )


def _mirrored(verts: np.ndarray) -> np.ndarray:
    """Same mesh reflected through x = 0, i.e. every element inverted."""
    out = verts.copy()
    out[:, 0] *= -1.0
    return out


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _hex_mesh():
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
        ],
        dtype=np.float64,
    )
    return make_polydata(verts, [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))])


def _pyramid_mesh():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float64,
    )
    return make_polydata(verts, [("pyramid", np.array([[0, 1, 2, 3, 4]]))])


def _wedge_mesh():
    # vtkWedge parametric coordinates, i.e. a positively oriented wedge.
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    return make_polydata(verts, [("wedge", np.array([[0, 1, 2, 3, 4, 5]]))])


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.f3grid"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)


def test_roundtrip_hex(tmp_path: Path) -> None:
    poly = _hex_mesh()
    out = tmp_path / "hex.f3grid"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)


def test_roundtrip_pyramid(tmp_path: Path) -> None:
    poly = _pyramid_mesh()
    out = tmp_path / "pyr.f3grid"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)


def test_roundtrip_wedge(tmp_path: Path) -> None:
    poly = _wedge_mesh()
    out = tmp_path / "wedge.f3grid"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)


def test_dt4_zone_parsed_as_tetra(tmp_path: Path) -> None:
    f = tmp_path / "dt4.f3grid"
    f.write_text(
        "GRIDPOINT 1  0.0  0.0  0.0\n"
        "GRIDPOINT 2  1.0  0.0  0.0\n"
        "GRIDPOINT 3  0.0  1.0  0.0\n"
        "GRIDPOINT 4  0.0  0.0  1.0\n"
        "ZONE  DT4  1  1  2  3  4\n"
    )
    with pytest.warns(UserWarning, match="DT4"):
        poly = read(f)
    assert len(poly.element_types) == 1
    assert len(poly.vertices) == 4


def test_file_uses_flac3d_record_keywords(tmp_path: Path) -> None:
    out = tmp_path / "tet.f3grid"
    write(_tet_mesh(), out)
    records = [
        ln.split()[0] for ln in out.read_text().splitlines() if ln and ln[0] != "*"
    ]
    assert set(records) == {"G", "Z"}
    assert records.count("G") == 4
    assert records.count("Z") == 1


def test_zone_keyword_written_as_t4(tmp_path: Path) -> None:
    out = tmp_path / "tet.f3grid"
    write(_tet_mesh(), out)
    zone_lines = [ln for ln in out.read_text().splitlines() if ln.startswith("Z ")]
    assert len(zone_lines) == 1
    assert zone_lines[0].split()[1] == "T4"


def test_no_gridpoints_raises(tmp_path: Path) -> None:
    f = tmp_path / "empty.f3grid"
    f.write_text("* empty\n")
    with pytest.raises(CodecError):
        read(f)


def test_zone_undefined_gridpoint_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.f3grid"
    f.write_text(
        "GRIDPOINT 1  0.0  0.0  0.0\nZONE  T4  1  1  2  3  99\n"  # 99 never defined
    )
    with pytest.raises(CodecError, match="undefined GRIDPOINT"):
        read(f)


def test_unknown_zone_type_skipped(tmp_path: Path) -> None:
    f = tmp_path / "mixed.f3grid"
    f.write_text(
        "GRIDPOINT 1  0.0  0.0  0.0\n"
        "GRIDPOINT 2  1.0  0.0  0.0\n"
        "GRIDPOINT 3  0.0  1.0  0.0\n"
        "GRIDPOINT 4  0.0  0.0  1.0\n"
        "ZONE  UNKNOWNTYPE  1  1  2  3\n"
        "ZONE  T4  2  1  2  3  4\n"
    )
    with pytest.warns(UserWarning, match="unknown zone type"):
        poly = read(f)
    assert len(poly.element_types) == 1


def test_malformed_gridpoint_warns(tmp_path: Path) -> None:
    f = tmp_path / "bad_gp.f3grid"
    f.write_text(
        "GRIDPOINT 1  0.0  0.0  0.0\n"
        "GRIDPOINT 2  1.0\n"  # too few fields
        "GRIDPOINT 3  0.0  1.0  0.0\n"
        "GRIDPOINT 4  0.0  0.0  1.0\n"
        "ZONE  T4  1  1  3  4  1\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="malformed GRIDPOINT"):
        poly = read(f)
    assert len(poly.vertices) == 3


def test_short_zone_warns(tmp_path: Path) -> None:
    f = tmp_path / "short_zone.f3grid"
    f.write_text(
        "GRIDPOINT 1  0.0  0.0  0.0\n"
        "GRIDPOINT 2  1.0  0.0  0.0\n"
        "GRIDPOINT 3  0.0  1.0  0.0\n"
        "GRIDPOINT 4  0.0  0.0  1.0\n"
        "ZONE  T4  1  1  2\n"  # T4 needs 4 node ids; only 2 given
        "ZONE  T4  2  1  2  3  4\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="short ZONE"):
        poly = read(f)
    assert len(poly.element_types) == 1


def test_write_truncated_offsets_raises(tmp_path: Path) -> None:
    poly = _tet_mesh()
    bad = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets[:0],  # empty — shorter than n_elems + 1
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="offsets length"):
        write(bad, tmp_path / "bad.f3grid")


def test_write_unsupported_type_warns(tmp_path: Path) -> None:
    """FLAC3D has no record for 1D elements."""
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    out = tmp_path / "line.f3grid"
    with pytest.warns(UserWarning, match="skipped"):
        write(poly, out)


def test_lazy_warns(tmp_path: Path) -> None:
    out = tmp_path / "tet.f3grid"
    write(_tet_mesh(), out)
    with pytest.warns(UserWarning, match="lazy"):
        read(out, lazy=True)


def test_non_sequential_gridpoint_ids(tmp_path: Path) -> None:
    f = tmp_path / "nseq.f3grid"
    f.write_text(
        "GRIDPOINT 100  0.0  0.0  0.0\n"
        "GRIDPOINT 200  1.0  0.0  0.0\n"
        "GRIDPOINT 300  0.0  1.0  0.0\n"
        "GRIDPOINT 400  0.0  0.0  1.0\n"
        "ZONE  T4  1  100  200  300  400\n",
        encoding="ascii",
    )
    poly = read(f)
    assert len(poly.vertices) == 4
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_roundtrip_mixed_elements(tmp_path: Path) -> None:
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
            [2, 0, 1],
            [3, 0, 1],
            [3, 1, 1],
            [2, 1, 1],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("hexahedron", np.array([[4, 5, 6, 7, 8, 9, 10, 11]])),
        ],
    )
    out = tmp_path / "mixed.f3grid"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)


def test_write_corrupt_connectivity_raises(tmp_path: Path) -> None:
    poly = _tet_mesh()
    bad = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity[
            :2
        ],  # truncated — offsets[-1] > len(connectivity)
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="offsets\\[-1\\]"):
        write(bad, tmp_path / "bad.f3grid")


def test_no_recognised_zones_warns(tmp_path: Path) -> None:
    f = tmp_path / "nozones.f3grid"
    f.write_text(
        "GRIDPOINT 1  0.0  0.0  0.0\n"
        "GRIDPOINT 2  1.0  0.0  0.0\n"
        "ZONE  UNKNOWNTYPE  1  1  2\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match=r"unknown zone type\(s\) skipped"):
        with pytest.warns(UserWarning, match="no recognised ZONE"):
            poly = read(f)
    assert len(poly.element_types) == 0


def _zone_ids(path: Path) -> list[int]:
    """Node ids (1-based, file order) of the single zone record in a file."""
    zone_lines = [ln for ln in path.read_text().splitlines() if ln.startswith("Z ")]
    assert len(zone_lines) == 1
    return [int(tok) for tok in zone_lines[0].split()[3:]]


def test_real_flac3d_syntax_read(tmp_path: Path) -> None:
    """A file in the syntax FLAC3D itself emits: G/Z records plus ZGROUP."""
    f = tmp_path / "real.f3grid"
    f.write_text(
        "* FLAC3D grid produced by FLAC3D\n"
        "* GRIDPOINTS\n"
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 1.0 1.0 0.0\n"
        "G 4 0.0 1.0 0.0\n"
        "G 5 0.0 0.0 1.0\n"
        "G 6 1.0 0.0 1.0\n"
        "G 7 1.0 1.0 1.0\n"
        "G 8 0.0 1.0 1.0\n"
        "* ZONES\n"
        "Z B8 1 1 2 4 5 3 8 6 7\n"
        "* GROUPS\n"
        'ZGROUP "Rock" SLOT 1\n'
        "     1\n",
        encoding="ascii",
    )
    poly = read(f)
    assert len(poly.vertices) == 8
    assert len(poly.element_types) == 1
    # B8 corners 1 2 4 5 3 8 6 7 map back to the polyxios hexahedron order.
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(poly.element_tags["Rock"], [0])


def test_face_records_read_as_surface_elements(tmp_path: Path) -> None:
    """`F` records become 2D elements, in the order they appear in the file."""
    f = tmp_path / "faces.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        "F T3 1 1 2 3\n"
        "F Q4 2 1 2 3 4\n",
        encoding="ascii",
    )
    poly = read(f)
    np.testing.assert_array_equal(
        poly.element_types,
        [ELEMENT_TYPES["tetra"], ELEMENT_TYPES["triangle"], ELEMENT_TYPES["quad"]],
    )
    np.testing.assert_array_equal(poly.offsets, [0, 4, 7, 11])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3])


def test_unknown_face_type_skipped(tmp_path: Path) -> None:
    f = tmp_path / "badface.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        "F T6 1 1 2 3 1 2 3\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="unknown face type"):
        poly = read(f)
    assert len(poly.element_types) == 1


def test_write_hex_uses_flac3d_corner_order(tmp_path: Path) -> None:
    out = tmp_path / "hex.f3grid"
    write(_hex_mesh(), out)
    # polyxios -> FLAC3D permutation [0, 1, 3, 4, 2, 7, 5, 6], 1-based.
    assert _zone_ids(out) == [1, 2, 4, 5, 3, 8, 6, 7]


def test_write_pyramid_uses_flac3d_corner_order(tmp_path: Path) -> None:
    out = tmp_path / "pyr.f3grid"
    write(_pyramid_mesh(), out)
    assert _zone_ids(out) == [1, 2, 4, 5, 3]


def test_write_wedge_uses_flac3d_corner_order(tmp_path: Path) -> None:
    out = tmp_path / "wedge.f3grid"
    write(_wedge_mesh(), out)
    assert _zone_ids(out) == [1, 2, 4, 3, 5, 6]


@pytest.mark.parametrize(
    ("name", "mesh_fn"),
    [
        ("tetra", _tet_mesh),
        ("pyramid", _pyramid_mesh),
        ("wedge", _wedge_mesh),
        ("hexahedron", _hex_mesh),
    ],
)
def test_face_tables_are_outward_wound(name: str, mesh_fn) -> None:
    """`_FACES` must agree with an independent tetrahedral decomposition.

    VTK's own wedge face table is wound inward; copying it verbatim made every
    valid wedge look inverted and get mirrored on write.
    """
    poly = mesh_fn()
    px, py, pz = (poly.vertices[:, k].tolist() for k in range(3))
    nodes = poly.connectivity.tolist()
    assert _signed_volume(px, py, pz, nodes, _FACES[name]) == pytest.approx(
        _element_volume(poly, 0, name)
    )


def test_write_flips_left_handed_zone(tmp_path: Path) -> None:
    """FLAC3D needs positive zone volume; inverted tetra is reordered."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("tetra", np.array([[0, 2, 1, 3]]))])
    out = tmp_path / "flip.f3grid"
    write(poly, out)
    ids = _zone_ids(out)
    assert ids == [1, 2, 3, 4]
    corners = verts[np.array(ids) - 1]
    det = np.dot(
        corners[1] - corners[0],
        np.cross(corners[2] - corners[0], corners[3] - corners[0]),
    )
    assert det > 0
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.connectivity, [0, 1, 2, 3])


def test_non_numeric_gridpoint_raises(tmp_path: Path) -> None:
    f = tmp_path / "nan_gp.f3grid"
    f.write_text("G 1 x y z\n", encoding="ascii")
    with pytest.raises(CodecError, match="non-numeric GRIDPOINT"):
        read(f)


def test_non_numeric_zone_raises(tmp_path: Path) -> None:
    f = tmp_path / "nan_zone.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 four\n",
        encoding="ascii",
    )
    with pytest.raises(CodecError, match="non-numeric ZONE"):
        read(f)


def test_comment_inside_quoted_name_kept(tmp_path: Path) -> None:
    f = tmp_path / "quoted.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4  * trailing comment\n"
        'ZGROUP "Rock*Salt" SLOT 1\n'
        "     1\n",
        encoding="ascii",
    )
    poly = read(f)
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])
    np.testing.assert_array_equal(poly.element_tags["Rock*Salt"], [0])


def test_write_out_of_range_connectivity_raises(tmp_path: Path) -> None:
    poly = _tet_mesh()
    bad = PolyData(
        vertices=poly.vertices[:3],  # one vertex short of what connectivity needs
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="out of range"):
        write(bad, tmp_path / "bad.f3grid")


@pytest.mark.parametrize(
    ("name", "mesh_fn"),
    [
        ("tetra", _tet_mesh),
        ("pyramid", _pyramid_mesh),
        ("wedge", _wedge_mesh),
        ("hexahedron", _hex_mesh),
    ],
)
def test_matches_meshio(tmp_path: Path, name: str, mesh_fn) -> None:
    meshio = pytest.importorskip("meshio")
    poly = mesh_fn()
    out = tmp_path / f"{name}.f3grid"
    write(poly, out)
    mesh = meshio.read(out, "flac3d")
    np.testing.assert_allclose(mesh.points, poly.vertices)
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == name
    np.testing.assert_array_equal(mesh.cells[0].data.ravel(), poly.connectivity)


def test_write_degenerate_zone_warns(tmp_path: Path) -> None:
    """Flat tetra has zero volume; FLAC3D rejects such zones."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="zero volume"):
        write(poly, tmp_path / "flat.f3grid")


def test_inline_zone_group_becomes_tag(tmp_path: Path) -> None:
    """FLAC3D 7 appends `GROUP "name"` to the zone record itself."""
    f = tmp_path / "inline_group.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        'Z T4 1 1 2 3 4 GROUP "Rock Salt" SLOT "Default"\n',
        encoding="ascii",
    )
    poly = read(f)
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])
    np.testing.assert_array_equal(poly.element_tags["Rock Salt"], [0])


@pytest.mark.parametrize(
    ("name", "mesh_fn"),
    [
        ("tetra", _tet_mesh),
        ("pyramid", _pyramid_mesh),
        ("wedge", _wedge_mesh),
        ("hexahedron", _hex_mesh),
    ],
)
def test_write_flips_inverted_element(tmp_path: Path, name: str, mesh_fn) -> None:
    """Mirrored elements are reordered so the FLAC3D zone volume is positive."""
    poly = mesh_fn()
    conn = poly.connectivity.reshape(1, -1)
    inverted = make_polydata(_mirrored(poly.vertices), [(name, conn)])
    assert _element_volume(inverted, 0, name) < 0

    out = tmp_path / f"{name}_flip.f3grid"
    write(inverted, out)
    poly2 = read(out)

    # Same corners, now positively oriented (FLAC3D rejects negative zones).
    # The zone read back in polyxios order is the orientation FLAC3D sees; the
    # determinant of the first four file nodes is only a valid proxy for T4,
    # so it is asserted in test_write_flips_left_handed_zone instead.
    assert set(poly2.connectivity.tolist()) == set(conn.ravel().tolist())
    assert _element_volume(poly2, 0, name) > 0


def test_concave_hex_not_flipped(tmp_path: Path) -> None:
    """Valid hex whose corner-0 frame is left-handed must not be reordered.

    Deciding handedness from the first four corners alone would flip this
    zone and emit a twisted element; the signed volume of the whole element
    is positive, so it must be written in the unflipped corner order.
    """
    verts = np.array(
        [
            [0.6, 0.6, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.6, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("hexahedron", np.arange(8).reshape(1, 8))])
    assert _element_volume(poly, 0, "hexahedron") > 0
    # Corner-0 frame (polyxios corners 0, 1, 3, 4) is left-handed.
    frame = np.dot(
        verts[1] - verts[0], np.cross(verts[3] - verts[0], verts[4] - verts[0])
    )
    assert frame < 0

    out = tmp_path / "concave.f3grid"
    write(poly, out)
    assert _zone_ids(out) == [1, 2, 4, 5, 3, 8, 6, 7]  # unflipped order
    poly2 = read(out)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_zgroup_roundtrip(tmp_path: Path) -> None:
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [3, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))],
        element_tags={
            "Rock": np.array([0], dtype=np.int32),
            "Soil": np.array([1], dtype=np.int32),
        },
    )
    out = tmp_path / "groups.f3grid"
    write(poly, out)
    text = out.read_text()
    assert 'ZGROUP "Rock"' in text
    assert 'ZGROUP "Soil"' in text

    poly2 = read(out)
    np.testing.assert_array_equal(poly2.element_tags["Rock"], [0])
    np.testing.assert_array_equal(poly2.element_tags["Soil"], [1])


def test_write_tag_with_no_written_record_warns(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1]]))],
        element_tags={"Edge": np.array([0], dtype=np.int32)},
    )
    with pytest.warns(UserWarning, match=r"element\(s\) skipped"):
        with pytest.warns(UserWarning, match="element tag"):
            write(poly, tmp_path / "line.f3grid")


def test_group_referencing_unknown_zone_warns(tmp_path: Path) -> None:
    f = tmp_path / "badgroup.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        'ZGROUP "Rock"\n'
        "     1 77\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="group member"):
        poly = read(f)
    np.testing.assert_array_equal(poly.element_tags["Rock"], [0])


def test_fgroup_block_becomes_tag(tmp_path: Path) -> None:
    """FGROUP members resolve against face ids, not zone ids."""
    f = tmp_path / "fgroup.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        "F T3 1 1 2 3\n"
        'ZGROUP "Rock" SLOT 1\n'
        "     1\n"
        'FGROUP "Top" SLOT 1\n'
        "     1\n",
        encoding="ascii",
    )
    poly = read(f)
    # Zone 1 and face 1 are different records even though they share an id.
    np.testing.assert_array_equal(poly.element_tags["Rock"], [0])
    np.testing.assert_array_equal(poly.element_tags["Top"], [1])


def test_fgroup_member_without_face_warns(tmp_path: Path) -> None:
    f = tmp_path / "fgroup_only.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        'FGROUP "Top" SLOT 1\n'
        "     1\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="group member"):
        poly = read(f)
    assert poly.element_tags == {}


def test_duplicate_gridpoint_id_warns(tmp_path: Path) -> None:
    f = tmp_path / "dup.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 1 9.0 9.0 9.0\n"  # same id redeclared
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="duplicate GRIDPOINT"):
        poly = read(f)
    assert len(poly.vertices) == 4  # no orphan vertex
    np.testing.assert_allclose(poly.vertices[0], [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_zone_before_gridpoints(tmp_path: Path) -> None:
    """Gridpoint ids are resolved after the whole file is read."""
    f = tmp_path / "forward.f3grid"
    f.write_text(
        "Z T4 1 1 2 3 4\n"
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n",
        encoding="ascii",
    )
    poly = read(f)
    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_malformed_lines_warn_once(tmp_path: Path) -> None:
    """Broken records are counted, not warned about one line at a time."""
    f = tmp_path / "many_bad.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        + "".join(f"G {i} 1.0\n" for i in range(2, 12))
        + "".join(f"Z T4 {i} 1\n" for i in range(1, 8)),
        encoding="ascii",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        poly = read(f)
    messages = [str(w.message) for w in caught]
    assert sum("malformed GRIDPOINT" in m for m in messages) == 1
    assert sum("short ZONE" in m for m in messages) == 1
    assert any("10 malformed GRIDPOINT" in m for m in messages)
    assert any("7 short ZONE" in m for m in messages)
    assert len(poly.vertices) == 1


def test_write_wrong_corner_count_warns(tmp_path: Path) -> None:
    poly = _tet_mesh()
    bad = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=np.array([0, 3], dtype=np.int32),  # tetra with 3 corners
        element_types=poly.element_types,
    )
    with pytest.warns(UserWarning, match="wrong corner count"):
        write(bad, tmp_path / "bad.f3grid")


def test_groups_match_meshio(tmp_path: Path) -> None:
    """ZGROUP blocks must carry the SLOT token meshio's parser requires."""
    meshio = pytest.importorskip("meshio")
    poly = _tet_mesh()
    poly = make_polydata(
        poly.vertices,
        [("tetra", poly.connectivity.reshape(1, -1))],
        element_tags={"Rock": np.array([0], dtype=np.int32)},
    )
    out = tmp_path / "grouped.f3grid"
    write(poly, out)
    assert 'ZGROUP "Rock" SLOT 1' in out.read_text()

    mesh = meshio.read(out, "flac3d")
    assert "zone:Rock:1" in mesh.cell_sets
    np.testing.assert_array_equal(np.asarray(mesh.cell_sets["zone:Rock:1"][0]), [0])

    # And a file written by meshio reads back with its groups intact.
    mio_file = tmp_path / "from_meshio.f3grid"
    meshio.write(mio_file, mesh, "flac3d")
    poly2 = read(mio_file)
    assert len(poly2.element_tags) == 1
    np.testing.assert_array_equal(next(iter(poly2.element_tags.values())), [0])


def test_duplicate_zone_id_warns(tmp_path: Path) -> None:
    f = tmp_path / "dupzone.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 7 1 2 3 4\n"
        "Z T4 7 1 2 3 4\n"  # id reused
        'ZGROUP "Rock" SLOT 1\n'
        "     7\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="duplicate ZONE id"):
        poly = read(f)
    assert len(poly.element_types) == 2
    np.testing.assert_array_equal(poly.element_tags["Rock"], [1])


def test_group_block_ends_at_next_text_line(tmp_path: Path) -> None:
    """A non-numeric line closes the block; its ids are not partly absorbed."""
    f = tmp_path / "block_end.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        'ZGROUP "Rock" SLOT 1\n'
        "     1\n"
        "1 stray text\n",
        encoding="ascii",
    )
    poly = read(f)
    np.testing.assert_array_equal(poly.element_tags["Rock"], [0])


def test_roundtrip_preserves_float64_coordinates(tmp_path: Path) -> None:
    """Coordinates are written at full float64 precision, not truncated."""
    verts = np.array(
        [
            [0.1234567890123456, -9.876543210987654e-8, 1.0],
            [1.0000000000000002, 0.0, 0.0],
            [0.0, 3.141592653589793, 0.0],
            [0.0, 0.0, 2.718281828459045],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    out = tmp_path / "precise.f3grid"
    write(poly, out)
    np.testing.assert_array_equal(read(out).vertices, verts)


def test_non_ascii_group_name_roundtrips(tmp_path: Path) -> None:
    """Group names outside ASCII survive; the file is latin-1, like read()."""
    poly = make_polydata(
        _tet_mesh().vertices,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"Grès schisteux": np.array([0], dtype=np.int32)},
    )
    out = tmp_path / "accent.f3grid"
    write(poly, out)
    np.testing.assert_array_equal(read(out).element_tags["Grès schisteux"], [0])


def test_group_name_outside_ascii_roundtrips(tmp_path: Path) -> None:
    """The file is UTF-8, so names no single-byte encoding can hold survive."""
    poly = make_polydata(
        _tet_mesh().vertices,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"岩塩": np.array([0], dtype=np.int32)},
    )
    out = tmp_path / "cjk.f3grid"
    write(poly, out)
    np.testing.assert_array_equal(read(out).element_tags["岩塩"], [0])


def test_utf8_bom_is_not_read_as_a_record(tmp_path: Path) -> None:
    """A BOM used to swallow the first record and orphan every zone."""
    f = tmp_path / "bom.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n",
        encoding="utf-8-sig",
    )
    poly = read(f)
    assert len(poly.vertices) == 4
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])


def test_undecodable_bytes_do_not_raise(tmp_path: Path) -> None:
    """Legacy single-byte files still load; only the bad characters change."""
    f = tmp_path / "legacy.f3grid"
    f.write_bytes(
        b"* commentaire g\xe9n\xe9r\xe9 par FLAC3D\n"
        b"G 1 0.0 0.0 0.0\n"
        b"G 2 1.0 0.0 0.0\n"
        b"G 3 0.0 1.0 0.0\n"
        b"G 4 0.0 0.0 1.0\n"
        b"Z T4 1 1 2 3 4\n"
    )
    poly = read(f)
    assert len(poly.element_types) == 1


def test_meshio_reads_non_ascii_group_name(tmp_path: Path) -> None:
    """Written files must be decodable by the wider (UTF-8) ecosystem."""
    meshio = pytest.importorskip("meshio")
    poly = make_polydata(
        _tet_mesh().vertices,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"Grès": np.array([0], dtype=np.int32)},
    )
    out = tmp_path / "accent.f3grid"
    write(poly, out)
    mesh = meshio.read(out, "flac3d")
    assert "zone:Grès:1" in mesh.cell_sets


def test_partly_dropped_tag_members_warn(tmp_path: Path) -> None:
    """A tag keeping some members but losing others is not silently trimmed."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("line", np.array([[0, 4]]))],
        element_tags={"Mixed": np.array([0, 1, 99], dtype=np.int32)},
    )
    out = tmp_path / "partial.f3grid"
    # The line element is unwritable, which is warned about on its own.
    with pytest.warns(UserWarning, match=r"element\(s\) skipped"):
        with pytest.warns(UserWarning, match="tag member"):
            write(poly, out)
    np.testing.assert_array_equal(read(out).element_tags["Mixed"], [0])


def test_near_degenerate_zone_warns(tmp_path: Path) -> None:
    """Sliver zones are reported; the zero test is relative to element size."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1e-15]], dtype=np.float64
    )
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="zero volume"):
        write(poly, tmp_path / "sliver.f3grid")


def test_small_but_valid_zone_not_degenerate(tmp_path: Path) -> None:
    """A tiny yet well-shaped zone is not mistaken for a degenerate one."""
    verts = 1e-6 * np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "tiny.f3grid")


def _face_ids(path: Path) -> list[int]:
    """Node ids (1-based, file order) of the single face record in a file."""
    face_lines = [ln for ln in path.read_text().splitlines() if ln.startswith("F ")]
    assert len(face_lines) == 1
    return [int(tok) for tok in face_lines[0].split()[3:]]


def _triangle_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])


def _quad_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    return make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])


@pytest.mark.parametrize(
    ("name", "mesh_fn", "code"),
    [("triangle", _triangle_mesh, "T3"), ("quad", _quad_mesh, "Q4")],
)
def test_roundtrip_surface_element(
    tmp_path: Path, name: str, mesh_fn, code: str
) -> None:
    poly = mesh_fn()
    out = tmp_path / f"{name}.f3grid"
    write(poly, out)
    # Surface elements are face records, not zones.
    assert [ln for ln in out.read_text().splitlines() if ln.startswith("Z ")] == []
    assert _face_ids(out) == list(range(1, len(poly.connectivity) + 1))
    face_lines = [ln for ln in out.read_text().splitlines() if ln.startswith("F ")]
    assert face_lines[0].split()[1] == code

    poly2 = read(out)
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)
    np.testing.assert_array_equal(poly2.element_types, poly.element_types)


def test_write_moves_faces_after_zones(tmp_path: Path) -> None:
    """FLAC3D keeps faces in their own section, so a mixed mesh is regrouped."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("quad", np.array([[0, 1, 4, 2]])),
        ],
        element_tags={
            "Skin": np.array([0, 2], dtype=np.int32),
            "Body": np.array([1], dtype=np.int32),
        },
    )
    out = tmp_path / "mixed_dim.f3grid"
    write(poly, out)
    records = [
        ln.split()[0] for ln in out.read_text().splitlines() if ln and ln[0] != "*"
    ]
    # Every Z record precedes every F record.
    assert records.index("Z") < records.index("F")

    poly2 = read(out)
    np.testing.assert_array_equal(
        poly2.element_types,
        [ELEMENT_TYPES["tetra"], ELEMENT_TYPES["triangle"], ELEMENT_TYPES["quad"]],
    )
    np.testing.assert_array_equal(poly2.offsets, [0, 4, 7, 11])
    np.testing.assert_array_equal(poly2.connectivity, [0, 1, 2, 3, 0, 1, 2, 0, 1, 4, 2])
    # Tags follow the elements to their new indices.
    np.testing.assert_array_equal(poly2.element_tags["Body"], [0])
    np.testing.assert_array_equal(poly2.element_tags["Skin"], [1, 2])


def test_tag_spanning_zones_and_faces_is_split(tmp_path: Path) -> None:
    """One tag over both dimensions needs a ZGROUP and an FGROUP block."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
        element_tags={"Both": np.array([0, 1], dtype=np.int32)},
    )
    out = tmp_path / "both.f3grid"
    write(poly, out)
    text = out.read_text()
    assert 'ZGROUP "Both" SLOT 1' in text
    assert 'FGROUP "Both" SLOT 1' in text

    # Zone and face groups of one name merge back into a single tag.
    np.testing.assert_array_equal(read(out).element_tags["Both"], [0, 1])


def test_zone_and_face_ids_do_not_collide(tmp_path: Path) -> None:
    """Group ids are unique across both sections, as FLAC3D itself writes them."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
    )
    out = tmp_path / "ids.f3grid"
    write(poly, out)
    ids = [
        int(ln.split()[2])
        for ln in out.read_text().splitlines()
        if ln.startswith(("Z ", "F "))
    ]
    assert ids == [1, 2]


def test_faces_match_meshio(tmp_path: Path) -> None:
    """meshio must recognise the face records and face groups written here."""
    meshio = pytest.importorskip("meshio")
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]])), ("quad", np.array([[0, 1, 2, 3]]))],
        element_tags={"Top": np.array([1], dtype=np.int32)},
    )
    out = tmp_path / "faces_meshio.f3grid"
    write(poly, out)

    mesh = meshio.read(out, "flac3d")
    np.testing.assert_allclose(mesh.points, poly.vertices)
    assert {cb.type for cb in mesh.cells} == {"triangle", "quad"}
    blocks = {cb.type: cb.data for cb in mesh.cells}
    np.testing.assert_array_equal(blocks["triangle"].ravel(), [0, 1, 2])
    np.testing.assert_array_equal(blocks["quad"].ravel(), [0, 1, 2, 3])
    assert "face:Top:1" in mesh.cell_sets


def test_b7_zone_read_as_hexahedron(tmp_path: Path) -> None:
    """B7 is a hexahedron whose eighth corner repeats the seventh."""
    f = tmp_path / "b7.f3grid"
    f.write_text(
        "".join(f"G {i} {i}.0 0.0 0.0\n" for i in range(1, 8))
        + "Z B7 1 1 2 3 4 5 6 7\n",
        encoding="ascii",
    )
    with pytest.warns(UserWarning, match="B7"):
        poly = read(f)
    assert len(poly.element_types) == 1
    assert poly.element_types[0] == ELEMENT_TYPES["hexahedron"]
    # Same permutation as B8, applied to [1 2 3 4 5 6 7 7] (0-based).
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 4, 2, 3, 6, 6, 5])


def test_empty_quoted_group_name_ignored(tmp_path: Path) -> None:
    """An unnamed block must not adopt `SLOT` as its name."""
    f = tmp_path / "noname.f3grid"
    f.write_text(
        "G 1 0.0 0.0 0.0\n"
        "G 2 1.0 0.0 0.0\n"
        "G 3 0.0 1.0 0.0\n"
        "G 4 0.0 0.0 1.0\n"
        "Z T4 1 1 2 3 4\n"
        'ZGROUP "" SLOT 1\n'
        "     1\n",
        encoding="ascii",
    )
    assert read(f).element_tags == {}


def test_non_finite_coordinate_warns(tmp_path: Path) -> None:
    """nan/inf coordinates produce a file FLAC3D cannot load."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, np.inf]], dtype=np.float64
    )
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="non-finite"):
        write(poly, tmp_path / "inf.f3grid")


def test_face_with_wrong_corner_count_warns(tmp_path: Path) -> None:
    poly = _triangle_mesh()
    bad = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=np.array([0, 2], dtype=np.int32),  # triangle with 2 corners
        element_types=poly.element_types,
    )
    with pytest.warns(UserWarning, match="wrong corner count"):
        write(bad, tmp_path / "bad_face.f3grid")


def test_write_ignores_unknown_options(tmp_path: Path) -> None:
    """The generic write signature passes format options to every codec."""
    write(_tet_mesh(), tmp_path / "opts.f3grid", float_fmt=".6f")
    assert len(read(tmp_path / "opts.f3grid").element_types) == 1


def test_issue_584_a_large_grid_reads_without_boxing_every_number(
    tmp_path: Path,
) -> None:
    """A read that boxes each coordinate costs several times the file itself.

    The accumulators are machine-number arrays rather than Python lists, which
    is what keeps a large grid's peak within a fixed multiple of its size. The
    bound is loose on purpose - it is a guard against the boxed-object
    regression, not a benchmark.
    """
    import tracemalloc

    n = 20
    ids = {
        (i, j, k): 1 + (i * n + j) * n + k
        for i in range(n)
        for j in range(n)
        for k in range(n)
    }
    lines = ["* GRIDPOINTS"]
    lines.extend(
        f"G {gid} {float(p[0])} {float(p[1])} {float(p[2])}" for p, gid in ids.items()
    )
    lines.append("* ZONES")
    zid = 0
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                zid += 1
                corners = [
                    ids[(i, j, k)],
                    ids[(i + 1, j, k)],
                    ids[(i, j + 1, k)],
                    ids[(i + 1, j + 1, k)],
                    ids[(i, j, k + 1)],
                    ids[(i + 1, j, k + 1)],
                    ids[(i, j + 1, k + 1)],
                    ids[(i + 1, j + 1, k + 1)],
                ]
                lines.append(f"Z B8 {zid} " + " ".join(str(c) for c in corners))
    path = tmp_path / "big.f3grid"
    path.write_text("\n".join(lines))

    size = path.stat().st_size
    tracemalloc.start()
    try:
        poly = read(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(poly.element_types) == (n - 1) ** 3
    assert peak < 12 * size, f"peak {peak} is {peak / size:.1f}x the file"
