from __future__ import annotations

import dataclasses
from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata, read as px_read, write as px_write
from polyxios._element_types import ELEMENT_TYPES, NODES_PER_ELEMENT
from polyxios.codecs import _netgen
from polyxios.codecs._netgen import read, write
from polyxios.exceptions import CodecError

# A file in the layout Netgen writes: the commented column headers, the four
# leading fields a surface element carries before its node count, and the point
# section last. One triangle and one tetrahedron over four nodes.
_SAMPLE = """mesh3d

dimension
3

geomtype
0

# surfnr    bcnr   domin  domout      np      p1      p2      p3
surfaceelements
1
1 2 0 0 3 1 2 3

#  matnr      np      p1      p2      p3      p4
volumeelements
1
7 4 1 2 3 4

#          X             Y             Z
points
4
0 0 0
1 0 0
0 1 0
0 0 1

endmesh
"""


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def _mixed_mesh():
    """One element of every linear type the format spells."""
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    return make_polydata(
        verts,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("wedge", np.array([[0, 1, 2, 3, 4, 5]])),
            ("pyramid", np.array([[0, 1, 2, 3, 4]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("line", np.array([[0, 1]])),
            ("vertex", np.array([[3]])),
        ],
    )


def _roundtrip(poly, tmp_path: Path, name: str = "mesh.vol"):
    path = tmp_path / name
    write(poly, path)
    return read(path)


# ---------------------------------------------------------------- reading


def test_reads_the_documented_layout(tmp_path: Path) -> None:
    path = tmp_path / "sample.vol"
    path.write_text(_SAMPLE)
    poly = read(path)

    assert poly.vertices.shape == (4, 3)
    np.testing.assert_allclose(poly.vertices[3], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        poly.element_types,
        [ELEMENT_TYPES["triangle"], ELEMENT_TYPES["tetra"]],
    )
    np.testing.assert_array_equal(poly.offsets, [0, 3, 7])
    # The tetrahedron's second and third nodes swap on the way in.
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 2, 1, 3])


def test_reads_the_element_indices_as_tags(tmp_path: Path) -> None:
    path = tmp_path / "sample.vol"
    path.write_text(_SAMPLE)
    poly = read(path)

    # bcnr 2 on the face, matnr 7 on the cell; the two are numbered apart.
    assert set(poly.element_tags) == {"bc_2", "material_7"}
    np.testing.assert_array_equal(poly.element_tags["bc_2"], [0])
    np.testing.assert_array_equal(poly.element_tags["material_7"], [1])


def test_index_zero_is_not_a_tag(tmp_path: Path) -> None:
    path = tmp_path / "unset.vol"
    path.write_text(
        "mesh3d\nsurfaceelements\n1\n0 0 0 0 3 1 2 3\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    assert read(path).element_tags == {}


def test_names_sections_name_the_tags(tmp_path: Path) -> None:
    path = tmp_path / "named.vol"
    path.write_text(
        "mesh3d\ndimension\n3\n"
        "surfaceelements\n1\n1 2 0 0 3 1 2 3\n"
        "volumeelements\n1\n1 4 1 2 3 4\n"
        "points\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "materials\n1\n1 steel\n"
        "bcnames\n2\n1 inlet\n2 outlet\n"
        "endmesh\n"
    )
    poly = read(path)
    assert set(poly.element_tags) == {"outlet", "steel"}


def test_one_name_over_several_indices_is_one_tag(tmp_path: Path) -> None:
    """Netgen spells one boundary carried by several surfaces this way."""
    path = tmp_path / "shared.vol"
    path.write_text(
        "mesh3d\ndimension\n3\nsurfaceelements\n3\n"
        "1 1 0 0 3 1 2 3\n2 2 0 0 3 1 2 4\n3 3 0 0 3 1 3 4\n"
        "points\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "bcnames\n3\n1 wall\n2 wall\n3 wall\nendmesh\n"
    )
    poly = read(path)
    assert set(poly.element_tags) == {"wall"}
    # No face is dropped on the way: a later index must not overwrite an earlier.
    np.testing.assert_array_equal(poly.element_tags["wall"], [0, 1, 2])


def test_one_name_over_two_codimensions_stays_apart(tmp_path: Path) -> None:
    """A face and a cell spelled alike are two things and keep two tags."""
    path = tmp_path / "codim.vol"
    path.write_text(
        "mesh3d\ndimension\n3\nsurfaceelements\n1\n1 1 0 0 3 1 2 3\n"
        "volumeelements\n1\n1 4 1 2 3 4\n"
        "points\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "bcnames\n1\n1 steel\nmaterials\n1\n1 steel\nendmesh\n"
    )
    poly = read(path)
    assert len(poly.element_tags) == 2
    assert sorted(int(v[0]) for v in poly.element_tags.values()) == [0, 1]


def test_quadratic_surface_and_volume_types(tmp_path: Path) -> None:
    path = tmp_path / "quad2.vol"
    nodes6 = " ".join(str(i) for i in range(1, 7))
    nodes10 = " ".join(str(i) for i in range(1, 11))
    coords = "\n".join("0 0 0" for _ in range(10))
    path.write_text(
        f"mesh3d\nsurfaceelements\n1\n1 1 0 0 6 {nodes6}\n"
        f"volumeelements\n1\n1 10 {nodes10}\n"
        f"points\n10\n{coords}\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(
        poly.element_types,
        [ELEMENT_TYPES["quadratic_triangle"], ELEMENT_TYPES["quadratic_tetra"]],
    )
    # triangle6 midpoints are permuted; the corners are not.
    np.testing.assert_array_equal(poly.connectivity[:6], [0, 1, 2, 5, 3, 4])


def test_one_section_may_hold_two_widths(tmp_path: Path) -> None:
    path = tmp_path / "widths.vol"
    path.write_text(
        "mesh3d\nvolumeelements\n2\n1 4 1 2 3 4\n2 8 1 2 3 4 5 6 7 8\n"
        "points\n8\n" + "\n".join("0 0 0" for _ in range(8)) + "\n"
    )
    poly = read(path)
    assert {int(t) for t in poly.element_types} == {
        ELEMENT_TYPES["tetra"],
        ELEMENT_TYPES["hexahedron"],
    }
    np.testing.assert_array_equal(poly.offsets, [0, 4, 12])


def test_gi_trailing_fields_are_ignored(tmp_path: Path) -> None:
    """surfaceelementsgi repeats the nodes as geometry info after them."""
    path = tmp_path / "gi.vol"
    path.write_text(
        "mesh3d\nsurfaceelementsgi\n1\n1 1 0 0 3 1 2 3 11 12 13\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_uv_trailing_parameters_are_floats(tmp_path: Path) -> None:
    """surfaceelementsuv writes two float parameters per node after the nodes."""
    path = tmp_path / "uv.vol"
    path.write_text(
        "mesh3d\nsurfaceelementsuv\n1\n"
        "1 1 0 0 3 1 2 3 0.5 0.25 0.75 0.125 0.0 1.0\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(poly.element_tags["bc_1"], [0])


def test_gi2_trailing_geometry_is_float(tmp_path: Path) -> None:
    """The one-line gi2 spelling carries two float distances in its tail."""
    path = tmp_path / "gi2.vol"
    path.write_text(
        "mesh3d\nedgesegmentsgi2\n1\n1 0 1 2 1 1 0 0 1 0.0 1 1.5\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_types, [ELEMENT_TYPES["line"]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1])


def test_two_line_edgesegments(tmp_path: Path) -> None:
    path = tmp_path / "edges.vol"
    path.write_text(
        "mesh3d\nsurf1 surf2 p1 p2\nedgesegmentsgi2\n2\n"
        "3 0 1 2\n-1 -1 0 0 1 0 1 0\n"
        "3 0 2 3\n-1 -1 0 0 1 0 1 0\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_types, [ELEMENT_TYPES["line"]] * 2)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 1, 2])
    np.testing.assert_array_equal(poly.element_tags["cd2_3"], [0, 1])


def test_two_line_edgesegments_without_the_marker(tmp_path: Path) -> None:
    """The marker is a comment in some Netgen versions, so it is gone by here."""
    path = tmp_path / "edges2.vol"
    path.write_text(
        "mesh3d\nedgesegmentsgi2\n2\n"
        "3 0 1 2\n-1 -1 0 0 1 0 1 0\n"
        "3 0 2 3\n-1 -1 0 0 1 0 1 0\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 1, 2])


def test_bare_one_line_edgesegments_stay_one_line(tmp_path: Path) -> None:
    """Records that stop at the nodes on every line are not a split record."""
    path = tmp_path / "edges3.vol"
    path.write_text(
        "mesh3d\nedgesegmentsgi2\n2\n3 0 1 2\n3 0 2 3\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_types, [ELEMENT_TYPES["line"]] * 2)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 1, 2])


def test_a_single_bare_edgesegment_does_not_swallow_the_next_section(
    tmp_path: Path,
) -> None:
    """The line behind a lone bare record is a keyword, not the tail of one.

    A section declaring one record has nothing behind it to compare widths with
    but the keyword line of the section that follows, which is as narrow as a
    geometry tail; pairing the two would take the whole next section with it.
    """
    path = tmp_path / "edges4.vol"
    path.write_text(
        "mesh3d\nedgesegmentsgi2\n1\n3 0 1 2\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_types, [ELEMENT_TYPES["line"]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1])
    assert poly.vertices.shape == (3, 3)


def test_a_single_two_line_edgesegment_is_still_paired(tmp_path: Path) -> None:
    """One record over two lines is still one record; the tail is numbers."""
    path = tmp_path / "edges5.vol"
    path.write_text(
        "mesh3d\nedgesegmentsgi2\n1\n3 0 1 2\n-1 -1 0 0 1 0 1 0\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_types, [ELEMENT_TYPES["line"]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1])
    assert poly.vertices.shape == (3, 3)


def test_point_and_edge_sections(tmp_path: Path) -> None:
    path = tmp_path / "low.vol"
    path.write_text(
        "mesh3d\nedgesegments\n1\n4 0 1 2\npointelements\n1\n3 5\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    poly = read(path)
    np.testing.assert_array_equal(
        poly.element_types, [ELEMENT_TYPES["line"], ELEMENT_TYPES["vertex"]]
    )
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    assert set(poly.element_tags) == {"cd2_4", "cd3_5"}


def test_identifications_do_not_shift_the_sections(tmp_path: Path) -> None:
    """identificationtypes counts values on one line, not lines."""
    path = tmp_path / "ident.vol"
    path.write_text(
        "mesh3d\nidentifications\n2\n1 2 1\n2 3 1\n"
        "identificationtypes\n3\n1 2 3\n"
        "face_colours\n1\n1 0.5 0.5 0.5\n"
        "surfaceelements\n1\n1 1 0 0 3 1 2 3\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.warns(UserWarning) as caught:
        poly = read(path)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    # Each section is stepped over, and each says what it dropped on the way.
    said = " ".join(str(record.message) for record in caught)
    for section in ("identifications", "identificationtypes", "face_colours"):
        assert f"the {section!r} section holds" in said


def test_empty_identificationtypes_does_not_eat_the_next_section(
    tmp_path: Path,
) -> None:
    """A count of zero writes no values line for the skip to step over.

    It also drops nothing, so stepping over it is worth no warning.
    """
    path = tmp_path / "ident0.vol"
    path.write_text(
        "mesh3d\nidentificationtypes\n0\nidentifications\n0\n"
        "surfaceelements\n1\n1 1 0 0 3 1 2 3\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\nendmesh\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_unknown_section_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "odd.vol"
    path.write_text(
        "mesh3d\nunheardof\n2\n1 2 3\n4 5 6\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.warns(UserWarning, match="unknown section"):
        poly = read(path)
    assert poly.vertices.shape == (3, 3)


def test_uncommented_column_header_is_ignored(tmp_path: Path) -> None:
    """A header is not a section; stepping over it would eat the section below."""
    path = tmp_path / "header.vol"
    path.write_text(
        "mesh3d\nsurfnr bcnr domin domout np p1 p2 p3\n"
        "surfaceelements\n1\n1 1 0 0 3 1 2 3\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.warns(UserWarning, match="unrecognized line"):
        poly = read(path)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_warnings_point_at_the_caller(tmp_path: Path) -> None:
    """A stacklevel past the caller reports the warning against warnings.py."""
    path = tmp_path / "wide.vol"
    path.write_text(
        "mesh3d\nsurfaceelements\n1\n1 1 0 0 5 1 2 3 4 5\n"
        "points\n5\n" + "\n".join("0 0 0" for _ in range(5)) + "\n"
    )
    with pytest.warns(UserWarning, match="not an element type") as caught:
        read(path)
    assert Path(caught[0].filename) == Path(__file__)

    poly = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"outer wall": np.array([0])},
    )
    with pytest.warns(UserWarning, match="whitespace") as caught:
        write(poly, tmp_path / "space2.vol")
    assert Path(caught[0].filename) == Path(__file__)


def test_unspellable_width_is_skipped_with_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "wide.vol"
    path.write_text(
        "mesh3d\nsurfaceelements\n1\n1 1 0 0 5 1 2 3 4 5\n"
        "points\n5\n" + "\n".join("0 0 0" for _ in range(5)) + "\n"
    )
    with pytest.warns(UserWarning, match="not an element type"):
        poly = read(path)
    assert len(poly.element_types) == 0


def test_two_dimensional_points_are_padded(tmp_path: Path) -> None:
    path = tmp_path / "flat.vol"
    path.write_text("mesh3d\ndimension\n2\npoints\n2\n1 2\n3 4\n")
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[1, 2, 0], [3, 4, 0]])


def test_two_dimensional_names_follow_the_codimension(tmp_path: Path) -> None:
    """In a 2D mesh 'materials' names the faces, not the cells."""
    path = tmp_path / "flat2.vol"
    path.write_text(
        "mesh3d\ndimension\n2\nsurfaceelements\n1\n1 3 0 0 3 1 2 3\n"
        "materials\n3\n1 a\n2 b\n3 plate\n"
        "points\n3\n0 0\n1 0\n0 1\n"
    )
    assert set(read(path).element_tags) == {"plate"}


def test_bom_and_crlf(tmp_path: Path) -> None:
    path = tmp_path / "bom.vol"
    path.write_bytes("mesh3d\r\npoints\r\n2\r\n0 0 0\r\n1 1 1\r\n".encode("utf-8-sig"))
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 1, 1]])


def test_lazy_warns(tmp_path: Path) -> None:
    path = tmp_path / "lazy.vol"
    path.write_text(_SAMPLE)
    with pytest.warns(UserWarning, match="lazy=True"):
        read(path, lazy=True)


# ----------------------------------------------------------- read errors


def test_missing_points_raises(tmp_path: Path) -> None:
    path = tmp_path / "nopoints.vol"
    path.write_text("mesh3d\ndimension\n3\n")
    with pytest.raises(CodecError, match="no 'points' section"):
        read(path)


def test_not_a_netgen_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.vol"
    path.write_text("solid cube\nfacet normal 0 0 1\n")
    with pytest.raises(CodecError, match="not a Netgen mesh"):
        read(path)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.vol"
    path.write_text("")
    with pytest.raises(CodecError, match="an empty file"):
        read(path)


def test_a_gzipped_vol_reads_like_a_plain_one(tmp_path: Path) -> None:
    """gzip is unwrapped by the IO layer, so the codec never sees it."""
    import gzip

    plain = tmp_path / "mesh.vol"
    plain.write_text(_SAMPLE)
    packed = tmp_path / "mesh.vol.gz"
    packed.write_bytes(gzip.compress(_SAMPLE.encode()))

    from_plain = px_read(plain)
    from_gzip = px_read(packed)

    np.testing.assert_array_equal(from_gzip.vertices, from_plain.vertices)
    np.testing.assert_array_equal(from_gzip.connectivity, from_plain.connectivity)


def test_missing_file_raises_codec_error(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="cannot read"):
        read(tmp_path / "absent.vol")


def test_truncated_section_raises(tmp_path: Path) -> None:
    path = tmp_path / "short.vol"
    path.write_text("mesh3d\npoints\n4\n0 0 0\n1 0 0\n")
    with pytest.raises(CodecError, match="truncated"):
        read(path)


def test_non_integer_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "badcount.vol"
    path.write_text("mesh3d\npoints\nmany\n")
    with pytest.raises(CodecError, match="non-integer point count"):
        read(path)


def test_negative_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "neg.vol"
    path.write_text("mesh3d\npoints\n-3\n")
    with pytest.raises(CodecError, match="negative point count"):
        read(path)


def test_negative_number_under_a_keyword_is_not_a_count(tmp_path: Path) -> None:
    """A count is a run of digits, so a signed number is a header, not a section."""
    path = tmp_path / "negsection.vol"
    path.write_text("mesh3d\nweird\n-5\npoints\n3\n0 0 0\n1 0 0\n0 1 0\nendmesh\n")
    with pytest.warns(UserWarning, match="unrecognized line"):
        poly = read(path)
    assert poly.vertices.shape == (3, 3)


def test_truncated_two_line_edges_count_in_records(tmp_path: Path) -> None:
    """The count line is written in records, so the shortfall is reported in them."""
    path = tmp_path / "shortedges.vol"
    path.write_text(
        "mesh3d\npoints\n2\n0 0 0\n1 0 0\n"
        "surf1 surf2 p1 p2\nedgesegmentsgi2\n3\n"
        "1 0 1 2\n-1 -1 0 0 1 0 1 0\n1 0 1 2\nendmesh\n"
    )
    with pytest.raises(CodecError, match=r"declares 3 record\(s\), found 2"):
        read(path)


def test_absurd_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "huge.vol"
    path.write_text("mesh3d\npoints\n999999999999\n")
    with pytest.raises(CodecError, match="safety cap"):
        read(path)


def test_non_numeric_coordinate_raises(tmp_path: Path) -> None:
    path = tmp_path / "nan.vol"
    path.write_text("mesh3d\npoints\n1\nx y z\n")
    with pytest.raises(CodecError, match="non-numeric point coordinate"):
        read(path)


def test_record_too_short_raises(tmp_path: Path) -> None:
    path = tmp_path / "shortrec.vol"
    path.write_text(
        "mesh3d\nsurfaceelements\n1\n1 1 0 0 3 1 2\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.raises(CodecError, match="too few"):
        read(path)


def test_non_integer_node_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "badnp.vol"
    path.write_text(
        "mesh3d\nvolumeelements\n1\n1 four 1 2 3 4\npoints\n4\n"
        + "\n".join("0 0 0" for _ in range(4))
        + "\n"
    )
    with pytest.raises(CodecError, match="non-integer node count"):
        read(path)


def test_non_integer_node_reference_raises(tmp_path: Path) -> None:
    path = tmp_path / "badref.vol"
    path.write_text(
        "mesh3d\nsurfaceelements\n1\n1 1 0 0 3 1 2 x\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.raises(CodecError, match="non-integer or out-of-range value"):
        read(path)


@pytest.mark.parametrize("ref", ["0", "9"])
def test_node_reference_out_of_range_raises(tmp_path: Path, ref: str) -> None:
    path = tmp_path / "range.vol"
    path.write_text(
        f"mesh3d\nsurfaceelements\n1\n1 1 0 0 3 1 2 {ref}\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.raises(CodecError, match="outside 1..3"):
        read(path)


def test_node_reference_too_large_for_int64_raises(tmp_path: Path) -> None:
    path = tmp_path / "big.vol"
    path.write_text(
        "mesh3d\nsurfaceelements\n1\n1 1 0 0 3 1 2 99999999999999999999999\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.raises(CodecError, match="out-of-range value"):
        read(path)


def test_uppercase_header_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "shout.vol"
    path.write_text("MESH3D\nPOINTS\n1\n1 2 3\n")
    np.testing.assert_allclose(read(path).vertices, [[1, 2, 3]])


def test_count_at_end_of_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "nocount.vol"
    path.write_text("mesh3d\npoints\n")
    with pytest.raises(CodecError, match="ends where the point count"):
        read(path)


def test_point_with_one_coordinate_raises(tmp_path: Path) -> None:
    path = tmp_path / "flat.vol"
    path.write_text("mesh3d\npoints\n1\n5\n")
    with pytest.raises(CodecError, match="coordinate"):
        read(path)


def test_record_too_short_for_the_node_count_column_raises(tmp_path: Path) -> None:
    """A surface record has to reach column 4 before its node count can be read."""
    path = tmp_path / "stub.vol"
    path.write_text("mesh3d\nsurfaceelements\n1\n1 1 0 0\npoints\n1\n0 0 0\n")
    with pytest.raises(CodecError, match="too few to hold its node count"):
        read(path)


def test_negative_node_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "negnp.vol"
    path.write_text("mesh3d\nsurfaceelements\n1\n1 1 0 0 -3 1 2 3\npoints\n1\n0 0 0\n")
    with pytest.raises(CodecError, match="negative node count"):
        read(path)


def test_malformed_names_records_are_skipped(tmp_path: Path) -> None:
    """A blank or non-numeric names entry names nothing and is stepped over."""
    path = tmp_path / "oddnames.vol"
    path.write_text(
        "mesh3d\ndimension\n3\n"
        "surfaceelements\n1\n1 3 0 0 3 1 2 3\n"
        "points\n3\n0 0 0\n1 0 0\n0 1 0\n"
        "bcnames\n3\n1\nx wall\n3 inlet\n"
        "endmesh\n"
    )
    poly = read(path)
    assert set(poly.element_tags) == {"inlet"}


def test_dimension_at_end_of_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "nodim.vol"
    path.write_text("mesh3d\ndimension\n")
    with pytest.raises(CodecError, match="ends where the dimension"):
        read(path)


def test_non_integer_dimension_raises(tmp_path: Path) -> None:
    path = tmp_path / "baddim.vol"
    path.write_text("mesh3d\ndimension\nflat\npoints\n1\n0 0 0\n")
    with pytest.raises(CodecError, match="non-integer dimension"):
        read(path)


def test_geomtype_at_end_of_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "nogeom.vol"
    path.write_text("mesh3d\ngeomtype\n")
    with pytest.raises(CodecError, match="ends where the geomtype"):
        read(path)


def test_total_element_cap_raises(tmp_path: Path, monkeypatch) -> None:
    """Two sections each under the cap can still add up to more than it."""
    monkeypatch.setattr(_netgen, "MAX_SAFE_ELEMENTS", 1)
    path = tmp_path / "manyelems.vol"
    path.write_text(
        "mesh3d\n"
        "surfaceelements\n1\n1 1 0 0 3 1 2 3\n"
        "volumeelements\n1\n1 4 1 2 3 4\n"
        "points\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
    )
    with pytest.raises(CodecError, match="element count 2 exceeds"):
        read(path)


def test_total_connectivity_cap_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_netgen, "MAX_SAFE_CONN", 3)
    path = tmp_path / "manynodes.vol"
    path.write_text(
        "mesh3d\nvolumeelements\n1\n1 4 1 2 3 4\npoints\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
    )
    with pytest.raises(CodecError, match="connectivity 4 exceeds"):
        read(path)


def test_two_points_sections_raise(tmp_path: Path) -> None:
    path = tmp_path / "twice.vol"
    path.write_text("mesh3d\npoints\n1\n0 0 0\npoints\n1\n1 1 1\n")
    with pytest.raises(CodecError, match="more than one 'points'"):
        read(path)


# ---------------------------------------------------------------- writing


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    back = _roundtrip(poly, tmp_path)
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.element_types, poly.element_types)


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tri_mesh()
    back = _roundtrip(poly, tmp_path)
    np.testing.assert_allclose(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_roundtrip_every_type(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    back = _roundtrip(poly, tmp_path)
    # Written section by section: faces, cells, edges, then point elements.
    expected = ["triangle", "quad", "tetra", "pyramid", "wedge", "hexahedron"]
    expected += ["line", "vertex"]
    np.testing.assert_array_equal(
        back.element_types, [ELEMENT_TYPES[n] for n in expected]
    )
    # Every element keeps its own nodes through the permutations.
    original = {
        int(t): poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]].tolist()
        for i, t in enumerate(poly.element_types)
    }
    for i, code in enumerate(back.element_types):
        got = back.connectivity[back.offsets[i] : back.offsets[i + 1]].tolist()
        assert got == original[int(code)]


def test_roundtrip_quadratic_types(tmp_path: Path) -> None:
    verts = np.arange(60, dtype=np.float64).reshape(20, 3)
    poly = make_polydata(
        verts,
        [
            ("quadratic_triangle", np.array([list(range(6))])),
            ("quadratic_quad", np.array([list(range(8))])),
            ("quadratic_tetra", np.array([list(range(10))])),
            ("quadratic_pyramid", np.array([list(range(13))])),
            ("quadratic_wedge", np.array([list(range(15))])),
            ("quadratic_hexahedron", np.array([list(range(20))])),
        ],
    )
    back = _roundtrip(poly, tmp_path)
    original = {
        int(t): poly.connectivity[poly.offsets[i] : poly.offsets[i + 1]].tolist()
        for i, t in enumerate(poly.element_types)
    }
    assert len(back.element_types) == len(poly.element_types)
    for i, code in enumerate(back.element_types):
        got = back.connectivity[back.offsets[i] : back.offsets[i + 1]].tolist()
        assert got == original[int(code)]


def test_header_and_sections(tmp_path: Path) -> None:
    path = tmp_path / "head.vol"
    write(_tet_mesh(), path)
    text = path.read_text()
    assert text.startswith("mesh3d\n")
    assert "\nvolumeelements\n1\n" in text
    assert "\nsurfaceelements\n0\n" in text
    assert "\npoints\n4\n" in text
    assert text.rstrip().endswith("endmesh")


def test_numbered_tags_keep_their_index(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"bc_4": np.array([0]), "bc_9": np.array([1])},
    )
    path = tmp_path / "tags.vol"
    write(poly, path)
    text = path.read_text()
    # bcnr keeps the tag's number; surfnr is numbered densely from 1.
    assert "1 4 0 0 3 1 2 3" in text
    assert "2 9 0 0 3 1 2 4" in text
    # A tag already named after its index needs no names section.
    assert "bcnames" not in text
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["bc_4"], [0])
    np.testing.assert_array_equal(back.element_tags["bc_9"], [1])


def test_named_tags_round_trip_through_the_names_section(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        element_tags={"inlet": np.array([0]), "steel": np.array([1])},
    )
    path = tmp_path / "named.vol"
    write(poly, path)
    text = path.read_text()
    assert "bcnames\n1\n1 inlet" in text
    assert "materials\n1\n1 steel" in text
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["inlet"], [0])
    np.testing.assert_array_equal(back.element_tags["steel"], [1])


def test_tag_names_with_whitespace_are_folded(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"outer wall": np.array([0])},
    )
    path = tmp_path / "space.vol"
    with pytest.warns(UserWarning, match="whitespace"):
        write(poly, path)
    assert "1 outer_wall" in path.read_text()
    assert set(read(path).element_tags) == {"outer_wall"}


def test_untagged_elements_take_a_free_index(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"bc_1": np.array([0])},
    )
    back = _roundtrip(poly, tmp_path)
    np.testing.assert_array_equal(back.element_tags["bc_1"], [0])
    # The untagged face cannot share index 1; it takes the next free one.
    assert "bc_2" in back.element_tags
    np.testing.assert_array_equal(back.element_tags["bc_2"], [1])


def test_colliding_tag_numbers_are_renumbered(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={"bc_3": np.array([0]), "bc_03": np.array([1])},
    )
    path = tmp_path / "collide.vol"
    with pytest.warns(UserWarning, match="already holds"):
        write(poly, path)
    back = read(path)
    assert len(back.element_tags) == 2


def test_shared_elements_keep_the_first_tag(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"bc_1": np.array([0]), "bc_2": np.array([0])},
    )
    path = tmp_path / "shared.vol"
    with pytest.warns(UserWarning, match="more than one tag"):
        write(poly, path)
    assert set(read(path).element_tags) == {"bc_1"}


def test_tag_index_outside_the_mesh_warns_once(tmp_path: Path) -> None:
    """Indices are handed out per dimension; the warning is still one warning."""
    poly = make_polydata(
        np.zeros((4, 3)),
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        element_tags={"bc_1": np.array([0, 1, 7])},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, tmp_path / "outside.vol")
    astray = [w for w in caught if "outside the mesh's" in str(w.message)]
    assert len(astray) == 1


def test_one_stray_index_in_two_tags_is_counted_once(tmp_path: Path) -> None:
    """The warning counts indices the mesh does not hold, not tag memberships."""
    poly = make_polydata(
        np.zeros((4, 3)),
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"a": np.array([0, 7]), "b": np.array([0, 7])},
    )
    # Both elements are also in two tags, which warns on its own; catching every
    # warning keeps that one from leaking out of the test.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write(poly, tmp_path / "twice.vol")
    astray = [w for w in caught if "outside the mesh's" in str(w.message)]
    assert len(astray) == 1
    assert "name 1 index(es) outside" in str(astray[0].message)


def test_tag_over_two_dimensions_comes_back_split(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((4, 3)),
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        element_tags={"steel": np.array([0, 1])},
    )
    back = _roundtrip(poly, tmp_path, "split.vol")
    assert len(back.element_tags) == 2
    members = sorted(int(v[0]) for v in back.element_tags.values())
    assert members == [0, 1]


def test_huge_tag_index_drops_the_names_section(tmp_path: Path) -> None:
    """A cell tag spelled the way a face tag is keeps the number and the name."""
    poly = make_polydata(
        np.zeros((4, 3)),
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"bc_999999": np.array([0])},
    )
    path = tmp_path / "huge.vol"
    with pytest.warns(UserWarning) as caught:
        write(poly, path)
    said = " ".join(str(record.message) for record in caught)
    assert "too large" in said
    assert "run past" in said
    text = path.read_text()
    assert "materials" not in text
    # The index itself survives, so the tag still comes back by number.
    assert "material_999999" in read(path).element_tags


def test_a_huge_index_warns_even_with_no_names_section(tmp_path: Path) -> None:
    """A tag already named after its own index writes no name to warn about.

    ``material_999999`` on a cell is the name reading would give index 999999
    anyway, so nothing goes into a ``materials`` section and the warning about
    dropping one never fires. The index is still written as it stands, and
    Netgen sizes its per-dimension tables off it.
    """
    poly = make_polydata(
        np.zeros((4, 3)),
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"material_999999": np.array([0])},
    )
    path = tmp_path / "unnamed.vol"
    with pytest.warns(UserWarning, match="run past 100000"):
        write(poly, path)
    assert "materials" not in path.read_text()
    assert "999999 4 " in path.read_text()
    assert "material_999999" in read(path).element_tags


def test_tag_index_past_int64_is_renumbered(tmp_path: Path) -> None:
    """A number the index array cannot hold takes a fresh one, name and all.

    ``element_tags`` is the caller's to fill, so a name may spell a number no
    int64 holds. Writing it into the per-element array would raise a bare
    OverflowError; it is renumbered the way a negative index is instead, and the
    name carries the original through the names section.
    """
    poly = make_polydata(
        np.zeros((4, 3)),
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"material_99999999999999999999": np.array([0])},
    )
    path = tmp_path / "past64.vol"
    write(poly, path)
    assert "1 4 1 3 2 4" in path.read_text()
    back = read(path)
    np.testing.assert_array_equal(
        back.element_tags["material_99999999999999999999"], [0]
    )
    # And the second pass sees the same name again and settles on the same file.
    second = tmp_path / "past64b.vol"
    write(back, second)
    assert second.read_text() == path.read_text()


def test_names_section_leaves_no_blank_entry(tmp_path: Path) -> None:
    """A gap in the names list is written with the name reading gives it anyway.

    Netgen reads the section as ``index >> name`` off the stream, so an entry
    written as a bare number takes the next index for its name and every entry
    behind it lands one field out.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3]])), ("tetra", np.array([[0, 1, 3, 2]]))],
        element_tags={"material_1": np.array([0]), "steel": np.array([1])},
    )
    path = tmp_path / "gap.vol"
    write(poly, path)
    section = path.read_text().split("materials\n")[1].splitlines()
    assert section[:3] == ["2", "1 material_1", "2 steel"]
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["material_1"], [0])
    np.testing.assert_array_equal(back.element_tags["steel"], [1])


def test_negatively_numbered_tag_is_renumbered(tmp_path: Path) -> None:
    """A negative index is not something the format spells, so it takes a fresh one."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"bc_-3": np.array([0])},
    )
    path = tmp_path / "negtag.vol"
    write(poly, path)
    assert "bcnames\n1\n1 bc_-3" in path.read_text()
    back = read(path)
    np.testing.assert_array_equal(back.element_tags["bc_-3"], [0])


def test_element_types_of_a_wider_dtype_still_group(tmp_path: Path) -> None:
    """The type scan reads uint8 off in one pass and falls back on any other dtype."""
    poly = _mixed_mesh()
    wide = dataclasses.replace(
        poly, element_types=np.asarray(poly.element_types, dtype=np.int32)
    )
    back = _roundtrip(wide, tmp_path, "wide.vol")
    # Writing reorders the elements section by section, so it is the multiset of
    # types that has to survive, not their order.
    np.testing.assert_array_equal(
        np.sort(back.element_types), np.sort(poly.element_types)
    )


def test_empty_names_section_is_not_written() -> None:
    assert _netgen._name_lines("materials", {}) == []


def test_a_line_of_no_fields_is_not_numbers() -> None:
    assert not _netgen._is_numbers("   ")


def test_unsupported_element_type_warns(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((5, 3)),
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("polygon", np.array([[0, 1, 2, 3, 4]])),
        ],
    )
    path = tmp_path / "poly.vol"
    with pytest.warns(UserWarning, match="unsupported element type"):
        write(poly, path)
    assert len(read(path).element_types) == 1


def test_float_fmt_option(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[1 / 3, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    path = tmp_path / "fmt.vol"
    write(poly, path, float_fmt=".17g")
    assert "0.33333333333333331" in path.read_text()


def test_the_default_format_is_not_bit_exact(tmp_path: Path) -> None:
    """``.10g`` does not name a float64 exactly; ``.17g`` is what does."""
    verts = np.array([[1 / 3, np.pi, 0.1], [1, 0, 0], [0, 1, 0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    lossy = tmp_path / "lossy.vol"
    write(poly, lossy)
    assert not np.array_equal(read(lossy).vertices, verts)
    exact = tmp_path / "exact.vol"
    write(poly, exact, float_fmt=".17g")
    np.testing.assert_array_equal(read(exact).vertices, verts)


def test_unusable_float_fmt_raises(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="not usable"):
        write(_tri_mesh(), tmp_path / "bad.vol", float_fmt="q")


def test_unparseable_float_fmt_raises(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="not a number"):
        write(_tri_mesh(), tmp_path / "bad.vol", float_fmt=",.2f")


def test_unknown_option_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tri_mesh(), tmp_path / "opt.vol", precision=3)


def test_bad_vertex_shape_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=np.zeros((4, 2)),
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match=r"expected \(n, 3\)"):
        write(broken, tmp_path / "shape.vol")


def test_offsets_not_closing_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets[:-1],
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="offsets carry"):
        write(broken, tmp_path / "off.vol")


def test_wrong_node_count_for_type_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=np.array([0, 2, 6], dtype=poly.offsets.dtype),
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="expected 3"):
        write(broken, tmp_path / "width.vol")


def test_offset_past_the_connectivity_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=poly.connectivity[:4],
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="fall outside the connectivity"):
        write(broken, tmp_path / "conn.vol")


def test_vertex_reference_out_of_range_raises(tmp_path: Path) -> None:
    poly = _tri_mesh()
    broken = type(poly)(
        vertices=poly.vertices[:2],
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="outside 0..1"):
        write(broken, tmp_path / "vert.vol")


def test_empty_mesh_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3)), [])
    path = tmp_path / "empty.vol"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
        back = read(path)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0


def test_second_roundtrip_changes_nothing(tmp_path: Path) -> None:
    """Writing what reading gave back reproduces the same file."""
    poly = make_polydata(
        np.arange(24, dtype=np.float64).reshape(8, 3),
        [
            ("triangle", np.array([[0, 1, 2], [1, 2, 3]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
        element_tags={"inlet": np.array([0]), "wall": np.array([1])},
    )
    first = tmp_path / "one.vol"
    second = tmp_path / "two.vol"
    write(poly, first)
    once = read(first)
    write(once, second)
    assert second.read_text() == first.read_text()
    twice = read(second)
    assert set(twice.element_tags) == set(once.element_tags)


def test_clean_roundtrip_is_quiet(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _roundtrip(_mixed_mesh(), tmp_path)


# ------------------------------------------------------------ integration


def test_registry_resolves_the_extension(tmp_path: Path) -> None:
    path = tmp_path / "viapx.vol"
    px_write(_tet_mesh(), path)
    poly = px_read(path)
    assert len(poly.element_types) == 1


def test_meshio_reads_what_this_codec_writes(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    path = tmp_path / "compat.vol"
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )
    write(poly, path)
    mesh = meshio.read(str(path), "netgen")
    blocks = {block.type: block.data for block in mesh.cells}
    np.testing.assert_allclose(mesh.points, poly.vertices)
    np.testing.assert_array_equal(blocks["triangle"], [[0, 1, 2]])
    np.testing.assert_array_equal(blocks["tetra"], [[0, 1, 2, 3]])


# Every type whose node order differs from Netgen's, and what meshio calls it.
# A wrong permutation writes a cell that is inside out and reads back the same
# way, so a round trip through this codec alone cannot catch one — the order has
# to be checked against something outside it.
_PERMUTED = [
    ("quadratic_triangle", "triangle6"),
    ("quadratic_quad", "quad8"),
    ("tetra", "tetra"),
    ("quadratic_tetra", "tetra10"),
    ("pyramid", "pyramid"),
    ("quadratic_pyramid", "pyramid13"),
    ("wedge", "wedge"),
    ("quadratic_wedge", "wedge15"),
    ("hexahedron", "hexahedron"),
    ("quadratic_hexahedron", "hexahedron20"),
]


def test_node_order_agrees_with_meshio() -> None:
    """The permutation tables are meshio's, type for type.

    Read back through meshio the volume types below are identity, but two of
    them — pyramid13 and wedge15 — meshio cannot read at all, so the tables
    themselves are what is compared.
    """
    netgen_module = pytest.importorskip("meshio.netgen._netgen")
    for name, meshio_name in _PERMUTED:
        assert (
            _netgen._TO_POLYXIOS[name]
            == (netgen_module.netgen_to_meshio_pmap[meshio_name])
        ), name


@pytest.mark.parametrize(
    "name",
    [
        "quadratic_triangle",
        "quadratic_quad",
        "pyramid",
        "wedge",
        "hexahedron",
        "quadratic_tetra",
        "quadratic_hexahedron",
    ],
)
def test_meshio_reads_the_permuted_types_back(tmp_path: Path, name: str) -> None:
    meshio = pytest.importorskip("meshio")
    meshio_name = dict(_PERMUTED)[name]
    width = NODES_PER_ELEMENT[name]
    verts = np.arange(3 * width, dtype=np.float64).reshape(width, 3)
    poly = make_polydata(verts, [(name, np.arange(width).reshape(1, width))])
    path = tmp_path / f"{name}.vol"
    write(poly, path)
    blocks = {
        block.type: block.data for block in meshio.read(str(path), "netgen").cells
    }
    np.testing.assert_array_equal(blocks[meshio_name], [list(range(width))])


def test_reads_what_meshio_writes(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    path = tmp_path / "frommeshio.vol"
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    meshio.write(
        str(path),
        meshio.Mesh(
            points,
            [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
        ),
        file_format="netgen",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, points)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 1, 2, 3])
    np.testing.assert_array_equal(
        poly.element_types, [ELEMENT_TYPES["triangle"], ELEMENT_TYPES["tetra"]]
    )


def test_issue_1369_a_facedescriptors_section_is_stepped_over(tmp_path) -> None:
    """Netgen writes one per surface; a second-order mesh always has them."""
    path = tmp_path / "fd.vol"
    path.write_text(
        "mesh3d\n\ndimension\n3\n\ngeomtype\n0\n\n"
        "facedescriptors\n1\n1 1 0 0\n\n"
        "surfaceelements\n0\n\n"
        "volumeelements\n1\n1 4 1 3 2 4\n\n"
        "points\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n\nendmesh\n"
    )
    # A known section holding data polyxios has no home for is reported as
    # such, rather than as a section nobody recognised.
    with pytest.warns(UserWarning, match="'facedescriptors' section holds"):
        poly = read(path)
    assert poly.vertices.shape == (4, 3)
    assert len(poly.element_types) == 1


def test_issue_1369_an_empty_facedescriptors_section_is_silent(tmp_path) -> None:
    path = tmp_path / "fd0.vol"
    path.write_text(
        "mesh3d\n\ndimension\n3\n\ngeomtype\n0\n\n"
        "facedescriptors\n0\n\n"
        "surfaceelements\n0\n\n"
        "volumeelements\n1\n1 4 1 3 2 4\n\n"
        "points\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n\nendmesh\n"
    )
    assert len(read(path).element_types) == 1
