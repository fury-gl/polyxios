from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios.codecs._fluent import ZONE_TYPES_KEY, _signed_measure, read, sniff, write
from polyxios.exceptions import CodecError

_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

# One tetrahedron, the way a solver writes it: a comment, the header, hex
# zone ids in the section headers and decimal ones in the (45 ...) records,
# a count-prefixed face zone and a plain one.
_TET = """\
(0 "Grid (written by hand)")
(1 "Fluent Interface")
(2 3)
(10 (0 1 4 0))
(10 (a 1 4 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
))
(12 (0 1 1 0))
(12 (b 1 1 1 2))
(13 (0 1 4 0))
(13 (c 1 2 3 0)(
3 1 3 2 1 0
3 1 2 4 1 0
))
(13 (d 3 4 a 3)(
2 3 4 1 0
3 1 4 1 0
))
(45 (10 fluid fluid-body)())
(45 (11 fluid fluid-body)())
(45 (12 wall bottom)())
(45 (13 velocity-inlet inlet)())
"""

# Two triangles sharing an edge, in the plane.
_TWO_TRIANGLES = """\
(2 2)
(10 (0 1 4 0))
(10 (1 1 4 1 2)(
0 0
1 0
1 1
0 1
))
(12 (0 1 2 0))
(12 (2 1 2 1 1))
(13 (0 1 5 0))
(13 (3 1 1 2 2)(
1 3 1 2
))
(13 (4 2 5 3 2)(
1 2 1 0
2 3 1 0
3 4 2 0
4 1 2 0
))
(45 (2 fluid sheet)())
(45 (3 interior interior-3)())
(45 (4 wall rim)())
"""


def _names(poly) -> list[str]:
    return [ELEMENT_TYPES_INV[int(t)] for t in poly.element_types]


def _rows(poly) -> list[list[int]]:
    return [
        poly.connectivity[poly.offsets[e] : poly.offsets[e + 1]].tolist()
        for e in range(len(poly.element_types))
    ]


def _tet_mesh():
    return make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])


def _two_tets():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
    )
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))])


def _cube_verts() -> np.ndarray:
    return np.array(
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


def _write(tmp_path: Path, text: str, name: str = "m.msh") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _measure(poly, e: int) -> float:
    conn = poly.connectivity[poly.offsets[e] : poly.offsets[e + 1]]
    return float(_signed_measure(conn[None], poly.vertices, _names(poly)[e])[0])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_tetrahedron_is_assembled_from_its_faces(tmp_path: Path) -> None:
    poly = read(_write(tmp_path, _TET))
    assert _names(poly) == ["tetra", "triangle", "triangle", "triangle", "triangle"]
    np.testing.assert_allclose(poly.vertices, _TET_VERTS)
    assert sorted(_rows(poly)[0]) == [0, 1, 2, 3]
    assert _measure(poly, 0) > 0
    assert poly.global_attrs == {
        ZONE_TYPES_KEY: {
            "fluid-body": "fluid",
            "bottom": "wall",
            "inlet": "velocity-inlet",
        }
    }


def test_zones_become_tags_named_by_the_45_records(tmp_path: Path) -> None:
    """Section headers spell the zone id in hex and (45 ...) in decimal."""
    poly = read(_write(tmp_path, _TET))
    assert {k: v.tolist() for k, v in poly.element_tags.items()} == {
        "fluid-body": [0],
        "bottom": [1, 2],
        "inlet": [3, 4],
    }


def test_a_boundary_face_links_back_to_its_cell(tmp_path: Path) -> None:
    poly = read(_write(tmp_path, _TET))
    assert poly.element_attrs["face_parent"].tolist() == [-1, 0, 0, 0, 0]
    local = poly.element_attrs["face_index"].tolist()
    assert local[0] == -1 and sorted(local[1:]) == [0, 1, 2, 3]
    # Each face is the side its index names: the parent's corners, as a set.
    from polyxios._faces import is_parent_face

    assert all(is_parent_face(poly, e, 0, local[e]) for e in range(1, 5))


def test_an_unnamed_zone_is_named_by_its_type_and_id(tmp_path: Path) -> None:
    text = "\n".join(line for line in _TET.splitlines() if not line.startswith("(45"))
    poly = read(_write(tmp_path, text))
    assert set(poly.element_tags) == {"fluid-11", "wall-12", "velocity-inlet-13"}
    assert poly.global_attrs[ZONE_TYPES_KEY]["velocity-inlet-13"] == "velocity-inlet"


def test_a_two_dimensional_file_is_padded_and_flagged(tmp_path: Path) -> None:
    poly = read(_write(tmp_path, _TWO_TRIANGLES))
    assert _names(poly) == ["triangle", "triangle", "line", "line", "line", "line"]
    assert poly.vertices.shape == (4, 3) and not poly.vertices[:, 2].any()
    assert poly.global_attrs["was_2d"] is True
    assert all(_measure(poly, e) > 0 for e in range(2))
    assert {k: v.tolist() for k, v in poly.element_tags.items()} == {
        "sheet": [0, 1],
        "rim": [2, 3, 4, 5],
    }
    # An edge has no face numbering, so the link columns are left out.
    assert poly.element_attrs == {}


def test_an_interior_zone_is_implied_by_the_cells(tmp_path: Path) -> None:
    """Interior faces are not elements; the zone leaves no tag behind."""
    poly = read(_write(tmp_path, _TWO_TRIANGLES))
    assert "interior-3" not in poly.element_tags
    assert "interior-3" not in poly.global_attrs[ZONE_TYPES_KEY]


def test_a_hexahedron_a_wedge_and_a_pyramid_are_assembled(tmp_path: Path) -> None:
    verts = np.vstack(
        [_cube_verts(), [[0.5, 0.5, 2.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.0]]]
    )
    poly = make_polydata(
        verts,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("pyramid", np.array([[4, 5, 6, 7, 8]])),
            ("wedge", np.array([[1, 9, 2, 5, 10, 6]])),
        ],
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    back = read(path)
    assert _names(back)[:3] == ["hexahedron", "pyramid", "wedge"]
    for e, expected in enumerate([1.0, 1.0 / 3.0, 0.5]):
        assert _measure(back, e) == pytest.approx(expected)
    assert [sorted(r) for r in _rows(back)[:3]] == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [1, 2, 5, 6, 9, 10],
    ]


def test_a_cell_is_oriented_by_its_geometry_not_its_faces(tmp_path: Path) -> None:
    """The faces' orientation is not trusted: a cell comes back right-handed."""
    mirrored = _TET.replace("3 1 3 2 1 0", "3 1 2 3 1 0").replace(
        "3 1 2 4 1 0", "3 1 4 2 1 0"
    )
    mirrored = mirrored.replace("2 3 4 1 0", "2 4 3 1 0").replace(
        "3 1 4 1 0", "3 4 1 1 0"
    )
    poly = read(_write(tmp_path, mirrored))
    assert _measure(poly, 0) > 0


def test_explicit_node_lists_under_a_typed_zone_are_read(tmp_path: Path) -> None:
    """Another reader writes a cell zone's connectivity outright, faces or not."""
    text = """\
(2 3)
(10 (0 1 4 0))
(10 (1 1 4 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
))
(12 (0 1 1 0))
(12 (2 1 1 1 2)(
1 2 3 4
))
"""
    poly = read(_write(tmp_path, text))
    assert _names(poly) == ["tetra"]
    assert _rows(poly) == [[0, 1, 2, 3]]
    assert poly.element_tags["fluid-2"].tolist() == [0]


def test_a_polygon_cell_is_chained_from_its_edges(tmp_path: Path) -> None:
    text = """\
(2 2)
(10 (0 1 5 0))
(10 (1 1 5 1 2)(
0 0
1 0
1.5 1
0.5 2
-0.5 1
))
(12 (0 1 1 0))
(12 (2 1 1 1 7))
(13 (0 1 5 0))
(13 (3 1 5 3 2)(
3 4 1 0
1 2 1 0
5 1 1 0
2 3 1 0
4 5 1 0
))
"""
    poly = read(_write(tmp_path, text))
    assert _names(poly)[0] == "polygon"
    ring = _rows(poly)[0]
    assert sorted(ring) == [0, 1, 2, 3, 4]
    assert _measure(poly, 0) > 0
    # Consecutive ring nodes are edges of the file.
    pairs = {frozenset(p) for p in zip(ring, [*ring[1:], ring[0]])}
    assert pairs == {frozenset(p) for p in [(2, 3), (0, 1), (4, 0), (1, 2), (3, 4)]}


def test_a_cell_its_faces_do_not_close_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    """Three faces of a tetrahedron, declared polyhedral: nothing to assemble."""
    text = """\
(2 3)
(10 (0 1 4 0))
(10 (1 1 4 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
))
(12 (0 1 1 0))
(12 (2 1 1 1 7))
(13 (0 1 3 0))
(13 (3 1 3 3 3)(
1 3 2 1 0
1 2 4 1 0
2 3 4 1 0
))
(45 (2 fluid open)())
(45 (3 wall skin)())
"""
    with pytest.warns(UserWarning, match=r"1 cell\(s\) could not be assembled"):
        poly = read(_write(tmp_path, text))
    assert _names(poly) == ["triangle"] * 3
    # The zone that held only the dropped cell is gone; the faces keep theirs
    # and their parent is unknown.
    assert set(poly.element_tags) == {"skin"}
    assert poly.element_attrs == {}


def _binary_tet(
    floats: str,
    ints: str,
    flavour: int,
    *,
    counted: bool = False,
    after_open: bytes = b"",
    before_close: bytes = b")",
) -> bytes:
    """A binary tetrahedron: floats and integers of the given widths.

    ``after_open`` sits between the ``(`` opening each block and its
    payload, ``before_close`` between the payload and the end marker.
    """
    n, c, fc = flavour + 10, flavour + 12, flavour + 13
    parts = [f'(0 "b")\n(2 3)\n(10 (0 1 4 0))\n({n} (1 1 4 1 3)('.encode()]
    parts.append(after_open + _TET_VERTS.astype(floats).tobytes())
    parts.append(before_close + f"End of Binary Section   {n})\n".encode())
    parts.append(f"(12 (0 1 1 0))\n({c} (2 1 1 1 0)(".encode())
    parts.append(after_open + np.array([2], ints).tobytes())
    parts.append(before_close + f"End of Binary Section   {c})\n".encode())
    faces = [[1, 3, 2, 1, 0], [1, 2, 4, 1, 0], [2, 3, 4, 1, 0], [3, 1, 4, 1, 0]]
    if counted:
        faces = [[3, *row] for row in faces]
    parts.append(f"(13 (0 1 4 0))\n({fc} (3 1 4 3 {0 if counted else 3})(".encode())
    parts.append(after_open + np.array(faces, ints).tobytes())
    parts.append(before_close + f"End of Binary Section   {fc})\n".encode())
    parts.append(b"(45 (2 fluid f)())\n(45 (3 wall w)())\n")
    return b"".join(parts)


# Single precision, and double precision with Fluent's own 32-bit integers
# and with the 64-bit ones another writer widens them to.
_BINARY_WIDTHS = [("<f4", "<i4", 2000), ("<f8", "<i4", 3000), ("<f8", "<i8", 3000)]
# Where the payload sits: on the line after the ``(``, the way Fluent and
# other writers spell it, on the same byte, or after a Windows line break.
_BINARY_SPELLINGS = [
    (b"\n", b"\n)\n"),
    (b"", b")\n"),
    (b"\r\n", b"\r\n)\r\n"),
    (b"\n", b"\n)"),
]


@pytest.mark.parametrize(("after_open", "before_close"), _BINARY_SPELLINGS)
@pytest.mark.parametrize(("floats", "ints", "flavour"), _BINARY_WIDTHS)
def test_binary_sections_are_read_in_every_flavour_and_width(
    tmp_path: Path,
    floats: str,
    ints: str,
    flavour: int,
    after_open: bytes,
    before_close: bytes,
) -> None:
    path = tmp_path / "m.msh"
    path.write_bytes(
        _binary_tet(
            floats, ints, flavour, after_open=after_open, before_close=before_close
        )
    )
    poly = read(path)
    assert _names(poly) == ["tetra", "triangle", "triangle", "triangle", "triangle"]
    np.testing.assert_allclose(poly.vertices, _TET_VERTS)
    assert {k: v.tolist() for k, v in poly.element_tags.items()} == {
        "f": [0],
        "w": [1, 2, 3, 4],
    }


@pytest.mark.parametrize(("after_open", "before_close"), _BINARY_SPELLINGS)
@pytest.mark.parametrize(("floats", "ints", "flavour"), _BINARY_WIDTHS)
def test_count_prefixed_binary_faces_are_read(
    tmp_path: Path,
    floats: str,
    ints: str,
    flavour: int,
    after_open: bytes,
    before_close: bytes,
) -> None:
    path = tmp_path / "m.msh"
    path.write_bytes(
        _binary_tet(
            floats,
            ints,
            flavour,
            counted=True,
            after_open=after_open,
            before_close=before_close,
        )
    )
    poly = read(path)
    assert _names(poly) == ["tetra", "triangle", "triangle", "triangle", "triangle"]
    assert _rows(poly)[1:] == [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]


def test_a_payload_opening_with_a_line_break_byte_is_not_stepped_over(
    tmp_path: Path,
) -> None:
    """A face block whose first node is 10 begins with the byte 0x0a."""
    verts = np.vstack([np.zeros((9, 3)), _TET_VERTS])
    parts = [b"(2 3)\n(10 (0 1 d 0))\n(2010 (1 1 d 1 3)("]
    parts.append(verts.astype("<f4").tobytes())
    parts.append(
        b")\nEnd of Binary Section   2010)\n(12 (0 1 1 0))\n(12 (2 1 1 1 2))\n"
    )
    faces = [
        [10, 12, 11, 1, 0],
        [10, 11, 13, 1, 0],
        [11, 12, 13, 1, 0],
        [12, 10, 13, 1, 0],
    ]
    parts.append(b"(13 (0 1 4 0))\n(2013 (3 1 4 3 3)(")
    parts.append(np.array(faces, "<i4").tobytes())
    parts.append(b")\nEnd of Binary Section   2013)\n")
    path = tmp_path / "m.msh"
    path.write_bytes(b"".join(parts))
    poly = read(path)
    assert _names(poly) == ["tetra", "triangle", "triangle", "triangle", "triangle"]
    assert sorted(_rows(poly)[0]) == [9, 10, 11, 12]


def test_a_wide_integer_block_that_closes_narrow_by_chance_is_not_misread(
    tmp_path: Path,
) -> None:
    """Read narrow, 64-bit integers show their high halves as every second value."""
    data = _binary_tet("<f8", "<i8", 3000)
    # The 4-byte reading of the face block ends mid-data, on the third
    # record's first node; plant a ')' pair there so that reading closes.
    # The wide reading then sees node 0x2929 and refuses it by name, which
    # the narrow one, taken at face value, would have read as node 3.
    open_ = b"(3013 (3 1 4 3 3)("
    at = data.index(open_) + len(open_) + 4 * 20
    assert data[at : at + 8] == (2).to_bytes(8, "little")
    path = tmp_path / "m.msh"
    path.write_bytes(data[:at] + b"))" + data[at + 2 :])
    with pytest.raises(CodecError, match="names node 10537, outside 1..4"):
        read(path)


def test_a_binary_section_this_codec_does_not_read_is_skipped(tmp_path: Path) -> None:
    """A cell tree or a periodic-shadow list in binary is stepped over by its marker."""
    parts = [_TET.encode()]
    parts.append(
        b"(2059 (1 1 1 2 3)(" + bytes(range(40)) + b")\nEnd of Binary Section   2059)\n"
    )
    path = tmp_path / "m.msh"
    path.write_bytes(b"".join(parts))
    assert _names(read(path))[0] == "tetra"


def test_a_comment_may_carry_parentheses_and_a_bom_is_stepped_over(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.msh"
    path.write_bytes(
        b"\xef\xbb\xbf" + _TET.replace('"Grid (written by hand)"', '"a ) ( b"').encode()
    )
    assert _names(read(path))[0] == "tetra"


def test_an_unknown_ascii_section_is_skipped_whole(tmp_path: Path) -> None:
    text = _TET + "(18 (1 2 3 4)(\n1 2\n3 4\n))\n(58 (1 1 1)((1 2) (3 4)))\n"
    assert _names(read(_write(tmp_path, text)))[0] == "tetra"


def test_reading_from_a_buffer_and_a_gzip_matches_a_path(tmp_path: Path) -> None:
    path = _write(tmp_path, _TET)
    expected = read(path)
    buffered = read(io.BytesIO(_TET.encode()))
    np.testing.assert_array_equal(buffered.connectivity, expected.connectivity)
    zipped = tmp_path / "m.msh.gz"
    zipped.write_bytes(gzip.compress(_TET.encode()))
    np.testing.assert_array_equal(
        polyxios.read(zipped).connectivity, expected.connectivity
    )


def test_lazy_warns_and_loads_eagerly(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="lazy=True is not supported"):
        poly = read(_write(tmp_path, _TET), lazy=True)
    assert _names(poly)[0] == "tetra"


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (("(2 3)", "(2 4)"), "dimension 4 is not 2 or 3"),
        (("0 0 1\n", "0 0 x\n"), "malformed coordinate"),
        (("(10 (a 1 4 1 3)", "(10 (a 1 5 1 3)"), "declares 5 node"),
        (("3 1 3 2 1 0", "3 1 3 9 1 0"), "names node 9, outside 1..4"),
        (("3 1 3 2 1 0", "3 1 3 2 7 0"), "names a cell outside 0..1"),
        (("3 1 3 2 1 0", "3 1 3 2 1"), "expected 5"),
        (("(13 (d 3 4 a 3)", "(13 (d 2 3 a 3)"), "into another zone"),
        (
            ("(12 (b 1 1 1 2))", "(12 (b 1 1 1 7)(\n1\n))"),
            "polyhedral and carries a body",
        ),
        (("(12 (b 1 1 1 2))", "(12 (b 1 1 1 9)(\n1 2 3 4\n))"), "does not define"),
        (("(2 3)", "(2 3"), "dimension section"),
        (('"Grid (written by hand)"', '"Grid'), "never closes"),
        (("(45 (13 velocity-inlet inlet)())", "hello"), "expected a '\\('"),
        (("(10 (a 1 4 1 3)", "(10 (a 1 4 1 zz)"), "malformed node dimension"),
        (("3 1 2 4 1 0", "3 1 2 4 1 0g"), "line 16: malformed face zone 12 entry '0g'"),
        (("2 3 4 1 0", "2 3 4 1 0x"), "malformed face zone 13 entry '0x'"),
        (("2 3 4 1 0", "2 3 4 1 10000000000000000"), "wider than 64 bits"),
        (("(12 (b 1 1 1 2))", "(12 (b 1 1 1 0)(\n2 3\n))"), "carries 2 value"),
    ],
)
def test_a_malformed_file_is_refused_with_the_line_named(
    tmp_path: Path, edit: tuple[str, str], match: str
) -> None:
    text = _TET.replace(*edit)
    assert text != _TET
    with pytest.raises(CodecError, match=match):
        read(_write(tmp_path, text))


def test_a_node_never_given_coordinates_is_refused(tmp_path: Path) -> None:
    text = _TET.replace("(10 (0 1 4 0))", "(10 (0 1 5 0))")
    with pytest.raises(
        CodecError, match="node 5 is declared but never given coordinates"
    ):
        read(_write(tmp_path, text))


def test_a_file_with_no_node_section_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="no node section"):
        read(_write(tmp_path, '(0 "x")\n(2 3)\n'))


def test_a_file_naming_no_dimension_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="names no dimension"):
        read(_write(tmp_path, "(10 (0 1 1 0))\n(10 (1 1 1 1)(\n0 0 0\n))\n"))


def test_a_count_past_the_file_is_refused_before_allocation(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="declares 100000 nodes but is only"):
        read(_write(tmp_path, "(2 3)\n(10 (0 1 186a0 0))\n"))


@pytest.mark.parametrize("after_open", [b"", b"\n"])
def test_a_truncated_binary_block_is_refused(tmp_path: Path, after_open: bytes) -> None:
    path = tmp_path / "m.msh"
    path.write_bytes(
        b"(2 3)\n(10 (0 1 4 0))\n(3010 (1 1 4 1 3)(" + after_open + b"\x00" * 10
    )
    with pytest.raises(CodecError, match="ends inside a binary block"):
        read(path)


def test_a_non_conformal_face_is_read_by_the_type_under_its_offset(
    tmp_path: Path,
) -> None:
    """An interface's faces spell their type with 1000 added."""
    poly = make_polydata(
        _two_tets().vertices,
        [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))],
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert "(13 (3 1 1 2 3)(" in text and "(13 (4 2 7 3 3)(" in text
    text = text.replace("(13 (3 1 1 2 3)(", "(13 (3 1 1 3ea 3)(").replace(
        "(13 (4 2 7 3 3)(", "(13 (4 2 7 3eb 3)("
    )
    text = "\n".join(line for line in text.splitlines() if not line.startswith("(45"))
    back = read(_write(tmp_path, text))
    assert _names(back) == ["tetra", "tetra"] + ["triangle"] * 6
    assert set(back.element_tags) == {"fluid-2", "wall-4"}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_file_has_the_expected_sections(tmp_path: Path) -> None:
    path = tmp_path / "m.fluent"
    write(_two_tets(), path)
    text = path.read_text()
    assert text.startswith('(0 "Written by polyxios")\n(1 "polyxios")\n(2 3)\n')
    assert "(10 (0 1 5 0))\n(10 (1 1 5 1 3)(\n" in text
    assert "(12 (0 1 2 0))\n(12 (2 1 2 1 2))\n" in text
    assert "(13 (0 1 7 0))\n(13 (3 1 1 2 3)(\n" in text
    assert "(13 (4 2 7 3 3)(\n" in text
    assert text.endswith(
        "(45 (2 fluid fluid)())\n(45 (3 interior interior)())\n(45 (4 wall wall)())\n"
    )


def test_every_face_is_written_once_with_the_cell_on_either_side(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.fluent"
    write(_two_tets(), path)
    text = path.read_text()
    interior = text.split("(13 (3 1 1 2 3)(\n", 1)[1].split("\n", 1)[0].split()
    assert sorted(int(t, 16) for t in interior[:3]) == [2, 3, 4]
    assert [int(t, 16) for t in interior[3:]] == [1, 2]
    boundary = text.split("(13 (4 2 7 3 3)(\n", 1)[1].split("\n))", 1)[0].splitlines()
    assert len(boundary) == 6
    assert all(line.split()[-1] == "0" for line in boundary)


def test_a_round_trip_keeps_the_cells_and_their_zones(tmp_path: Path) -> None:
    poly = make_polydata(
        _two_tets().vertices,
        [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))],
        element_tags={"left": np.array([0]), "right": np.array([1])},
        global_attrs={ZONE_TYPES_KEY: {"right": "solid"}},
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert "(12 (2 1 1 1 2))\n(12 (3 2 2 11 2))\n" in text
    assert "(45 (2 fluid left)())\n(45 (3 solid right)())" in text
    back = read(path)
    assert _names(back)[:2] == ["tetra", "tetra"]
    assert [sorted(r) for r in _rows(back)[:2]] == [[0, 1, 2, 3], [1, 2, 3, 4]]
    assert back.element_tags["left"].tolist() == [0]
    assert back.element_tags["right"].tolist() == [1]
    assert back.global_attrs[ZONE_TYPES_KEY] == {
        "left": "fluid",
        "right": "solid",
        "wall": "wall",
    }


def test_a_face_element_names_the_zone_of_the_side_it_matches(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 2, 1], [1, 2, 3]])),
        ],
        element_tags={"bottom": np.array([1]), "inlet": np.array([2])},
        global_attrs={ZONE_TYPES_KEY: {"inlet": "velocity-inlet"}},
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert (
        "(45 (3 wall bottom)())\n(45 (4 velocity-inlet inlet)())\n(45 (5 wall wall)())"
        in text
    )
    back = read(path)
    assert {k: len(v) for k, v in back.element_tags.items()} == {
        "fluid": 1,
        "bottom": 1,
        "inlet": 1,
        "wall": 2,
    }
    assert sorted(_rows(back)[back.element_tags["inlet"][0]]) == [1, 2, 3]


def test_the_boundary_word_is_read_off_a_fluent_style_tag_name(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 2, 1]]))],
        element_tags={"pressure-outlet-7": np.array([1])},
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert "(13 (3 1 1 5 3)(" in path.read_text()
    assert "(45 (3 pressure-outlet pressure-outlet-7)())" in path.read_text()


def test_a_side_matching_an_interior_face_is_an_interior_zone(tmp_path: Path) -> None:
    """A named face between two cells - a baffle - is written as its own zone."""
    poly = make_polydata(
        _two_tets().vertices,
        [
            ("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]])),
            ("triangle", np.array([[1, 2, 3]])),
        ],
        element_tags={"baffle": np.array([2])},
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert "(13 (3 1 1 2 3)(\n" in text
    assert "(45 (3 interior baffle)())" in text
    assert "interior interior" not in text
    back = read(path)
    assert "baffle" not in back.element_tags


def test_a_group_naming_cells_and_faces_is_two_zones(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 2, 1]]))],
        element_tags={"body": np.array([0, 1])},
    )
    path = tmp_path / "m.fluent"
    with pytest.warns(UserWarning, match="body-faces"):
        write(poly, path)
    assert "(45 (2 fluid body)())\n(45 (3 wall body-faces)())" in path.read_text()


def test_a_planar_mesh_is_written_in_two_dimensions(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]])), ("quad", np.array([[0, 1, 2, 3]]))],
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert "(2 2)\n" in text and "(10 (1 1 4 1 2)(\n0.0 0.0\n" in text
    back = read(path)
    assert back.global_attrs["was_2d"] is True
    assert _names(back)[:2] == ["triangle", "quad"]


def test_a_surface_off_the_plane_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    with pytest.raises(CodecError, match="in the plane only"):
        write(poly, tmp_path / "m.fluent")


def test_a_mesh_without_a_cell_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_TET_VERTS[:3], [("line", np.array([[0, 1]]))])
    with pytest.raises(CodecError, match="no cell to write"):
        write(poly, tmp_path / "m.fluent")


def test_an_element_of_another_type_is_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        np.vstack([_TET_VERTS, [[2, 0, 0]]]),
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[0, 4]])),
            ("vertex", np.array([[4]])),
        ],
    )
    path = tmp_path / "m.fluent"
    with pytest.warns(
        UserWarning, match=r"\['line', 'vertex'\] are neither a cell nor a face"
    ):
        write(poly, path)
    assert _names(read(path)) == [
        "tetra",
        "triangle",
        "triangle",
        "triangle",
        "triangle",
    ]


def test_a_face_element_that_is_no_side_of_a_cell_is_dropped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.vstack([_TET_VERTS, [[2, 0, 0]]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[1, 2, 4]]))],
    )
    with pytest.warns(
        UserWarning, match=r"1 face element\(s\) are no side of a written cell"
    ):
        write(poly, tmp_path / "m.fluent")


def test_an_element_in_two_groups_stays_with_the_first(tmp_path: Path) -> None:
    poly = make_polydata(
        _two_tets().vertices,
        [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))],
        element_tags={"a": np.array([0, 1]), "b": np.array([1])},
    )
    path = tmp_path / "m.fluent"
    with pytest.warns(
        UserWarning, match=r"element tag group\(s\) \['b'\] name elements"
    ):
        write(poly, path)
    back = read(path)
    assert back.element_tags["a"].tolist() == [0, 1]
    assert "b" not in back.element_tags


def test_a_mixed_zone_lists_one_type_per_cell(tmp_path: Path) -> None:
    verts = np.vstack([_cube_verts(), [[0.5, 0.5, 2.0]]])
    poly = make_polydata(
        verts,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
            ("pyramid", np.array([[4, 5, 6, 7, 8]])),
        ],
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert "(12 (2 1 2 1 0)(\n4 5\n))\n" in path.read_text()
    assert _names(read(path))[:2] == ["hexahedron", "pyramid"]


def _outward(text: str, poly) -> list[bool]:
    """Say, per boundary face record, whether its normal points out of its cell.

    Cells are written in ``poly``'s order under one zone, so a record's
    ``c0`` names the input cell ``c0 - 1``.
    """
    verts = np.hstack([poly.vertices, np.zeros((len(poly.vertices), 1))])[:, :3]
    boundary = text[text.rindex("(13 (") :]
    records = boundary.split("(\n", 1)[1].split("\n))", 1)[0].splitlines()
    out = []
    for line in records:
        toks = [int(t, 16) for t in line.split()]
        ring, c0 = np.asarray(toks[:-2]) - 1, toks[-2] - 1
        cell = poly.connectivity[poly.offsets[c0] : poly.offsets[c0 + 1]]
        centre = verts[cell].mean(axis=0)
        if ring.size == 3:
            n = np.cross(
                verts[ring[1]] - verts[ring[0]], verts[ring[2]] - verts[ring[0]]
            )
        else:
            d = verts[ring[1]] - verts[ring[0]]
            n = np.array([d[1], -d[0], 0.0])
        out.append(bool(n @ (verts[ring].mean(axis=0) - centre) > 0))
    return out


def test_a_left_handed_cell_is_written_with_its_faces_outward(tmp_path: Path) -> None:
    """A face record's node order fixes its normal's side, so the cell is mirrored."""
    poly = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 2, 1, 3]]))])
    before = poly.connectivity.copy()
    path = tmp_path / "m.fluent"
    write(poly, path)
    np.testing.assert_array_equal(poly.connectivity, before)
    assert _outward(path.read_text(), poly) == [True] * 4
    right = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    write(right, path)
    assert _outward(path.read_text(), right) == [True] * 4
    assert _measure(read(path), 0) > 0


def test_a_clockwise_planar_cell_is_written_counter_clockwise(tmp_path: Path) -> None:
    verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("quad", np.array([[0, 3, 2, 1]])), ("triangle", np.array([[1, 4, 2]]))],
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert _outward(path.read_text(), poly) == [True] * 5
    back = read(path)
    assert all(_measure(back, e) > 0 for e in range(2))


def test_a_clockwise_polygon_is_written_counter_clockwise(tmp_path: Path) -> None:
    verts = np.array([[0, 0], [1, 0], [1.5, 1], [0.5, 2], [-0.5, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("polygon", np.array([[0, 4, 3, 2, 1]]))])
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert _outward(path.read_text(), poly) == [True] * 5
    back = read(path)
    assert _measure(back, 0) > 0
    assert _rows(back)[0][0] == 0


def test_a_face_between_three_cells_is_refused(tmp_path: Path) -> None:
    verts = np.vstack([_TET_VERTS, [[1, 1, 1], [-1, -1, -1]]])
    poly = make_polydata(
        verts, [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 5]]))]
    )
    with pytest.raises(CodecError, match="shared by 3 cells"):
        write(poly, tmp_path / "m.fluent")


def test_a_zone_name_is_made_safe_for_its_record(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"left (upper)": np.array([0])},
    )
    path = tmp_path / "m.fluent"
    with pytest.warns(UserWarning, match="left_upper_"):
        write(poly, path)
    assert "(45 (2 fluid left_upper_)())" in path.read_text()


def test_a_non_finite_coordinate_is_written_with_a_warning(tmp_path: Path) -> None:
    verts = _TET_VERTS.copy()
    verts[3, 2] = np.nan
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="non-finite"):
        write(poly, tmp_path / "m.fluent")


def test_an_unrecognised_option_is_warned_about(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tet_mesh(), tmp_path / "m.fluent", binary=True)


def test_a_vertex_out_of_range_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    import dataclasses

    broken = dataclasses.replace(
        poly, connectivity=np.array([0, 1, 2, 9], dtype=np.int32)
    )
    with pytest.raises(CodecError, match="outside 0..3"):
        write(broken, tmp_path / "m.fluent")


def test_writing_to_a_buffer_matches_a_path(tmp_path: Path) -> None:
    path = tmp_path / "m.fluent"
    write(_two_tets(), path)
    buffer = io.BytesIO()
    write(_two_tets(), buffer)
    assert buffer.getvalue() == path.read_bytes()


def test_a_structured_block_round_trips_every_hexahedron(tmp_path: Path) -> None:
    from tests.test_roundtrip import _structured

    poly = _structured()
    path = tmp_path / "m.fluent"
    with pytest.warns(UserWarning, match="earlier group"):
        write(poly, path)
    back = read(path)
    assert _names(back).count("hexahedron") == 8
    assert _names(back).count("quad") == 24
    assert all(_measure(back, e) == pytest.approx(1.0) for e in range(8))
    assert sorted(sorted(r) for r in _rows(back)[:8]) == sorted(
        sorted(r) for r in _rows(poly)
    )


# ---------------------------------------------------------------------------
# Sniffing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "head",
    [
        b'(0 "Grid")\n',
        b'  \n(1 "Fluent Interface")',
        b"(2 3)",
        b"(10 (0 1 4 0))",
        b"(2010 (1 1 4 1 3)(",
        b'\xef\xbb\xbf(0 "bom")',
    ],
)
def test_sniff_claims_a_file_opening_with_a_section(head: bytes) -> None:
    assert sniff(head)


@pytest.mark.parametrize(
    "head",
    [
        b"$MeshFormat\n2.2 0 8\n",
        b"",
        b"(7 1)",
        b"POLYGON((0 0, 1 0))",
        b"MFEM mesh v1.0",
        b"(10x",
    ],
)
def test_sniff_declines_what_is_not_a_fluent_mesh(head: bytes) -> None:
    assert not sniff(head)


def test_the_api_resolves_a_fluent_msh_by_content(tmp_path: Path) -> None:
    path = _write(tmp_path, _TET, "grid.msh")
    poly = polyxios.read(path)
    assert _names(poly)[0] == "tetra"
    assert ELEMENT_TYPES["tetra"] == poly.element_types[0]


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", ["10", "12", "13"])
def test_a_negative_declared_count_is_refused(tmp_path: Path, section: str) -> None:
    text = _TET.replace(f"({section} (0 1 ", f"({section} (0 1 -")
    assert text != _TET
    with pytest.raises(CodecError, match="count -.* is negative"):
        read(_write(tmp_path, text))


def test_two_column_vertices_under_a_solid_are_written_in_three(
    tmp_path: Path,
) -> None:
    poly = make_polydata(_TET_VERTS[:, :2], [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert "(10 (1 1 4 1 3)(\n0.0 0.0 0.0\n" in path.read_text()
    back = read(path)
    np.testing.assert_array_equal(back.vertices[:, 2], 0.0)
    assert _names(back)[0] == "tetra"


def test_a_nan_third_coordinate_does_not_put_a_planar_mesh_off_the_plane(
    tmp_path: Path,
) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, np.nan]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert "(2 2)" in path.read_text()


def test_a_group_of_cells_and_faces_needing_a_safe_name_is_two_zones(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 2, 1]]))],
        element_tags={"my zone": np.array([0, 1])},
    )
    path = tmp_path / "m.fluent"
    with pytest.warns(UserWarning) as caught:
        write(poly, path)
    text = path.read_text()
    assert "(45 (2 fluid my_zone)())\n(45 (3 wall my_zone-faces)())" in text
    (message,) = [str(w.message) for w in caught if "zone name" in str(w.message)]
    assert "('my zone', 'my_zone')" in message
    assert "('my zone', 'my_zone-faces')" in message
    assert message.count("my zone") == 2


def test_a_bad_count_prefixed_record_names_its_own_line(tmp_path: Path) -> None:
    text = _TET.replace("3 1 2 4 1 0\n", "3 1 2 4 1\n")
    with pytest.raises(
        CodecError, match=r"line 16: .*carries 4 value\(s\), expected 5"
    ):
        read(_write(tmp_path, text))


@pytest.mark.parametrize(
    "pattern",
    [
        [3] * 300,
        [3, 4] * 150,
        ([3] * 3 + [4] * 3) * 50,
        ([3] * 100 + [4] * 100 + [5] * 100),
        [4] + [3] * 299,
        [3] * 299 + [4],
        [3],
        [3, 4],
    ],
)
def test_count_prefixed_binary_widths_are_walked_in_runs(
    tmp_path: Path, pattern: list[int]
) -> None:
    from polyxios.codecs._fluent import _counted_layout

    records = [[w, *range(1, w + 1), 1, 0] for w in pattern]
    raw = np.frombuffer(b"".join(np.array(r, "<i4").tobytes() for r in records), "<i4")
    widths, starts, used = _counted_layout(raw, len(pattern), RuntimeError)
    lengths = np.array([len(r) for r in records])
    assert widths.tolist() == pattern
    assert starts.tolist() == (np.cumsum(lengths) - lengths + 1).tolist()
    assert used == raw.size


def test_a_count_prefixed_binary_block_short_of_a_record_is_refused() -> None:
    from polyxios.codecs._fluent import _counted_layout

    raw = np.frombuffer(np.array([3, 1, 2, 3, 1, 0, 3, 1], "<i4").tobytes(), "<i4")
    with pytest.raises(RuntimeError, match="ends inside"):
        _counted_layout(raw, 2, RuntimeError)
    with pytest.raises(RuntimeError, match=r"opens with 1 node\(s\)"):
        _counted_layout(np.array([1, 1, 0, 0], "<i4"), 1, RuntimeError)


def test_a_comment_may_carry_an_escaped_quote(tmp_path: Path) -> None:
    text = _TET.replace('(0 "Grid (written by hand)")', '(0 "a \\"quoted\\" (word)")')
    assert _names(read(_write(tmp_path, text)))[0] == "tetra"


def test_a_quoted_zone_name_is_read_whole(tmp_path: Path) -> None:
    text = _TET.replace("(45 (12 wall bottom)())", '(45 (12 wall "the floor")())')
    poly = read(_write(tmp_path, text))
    assert "the floor" in poly.element_tags
    assert poly.global_attrs[ZONE_TYPES_KEY]["the floor"] == "wall"


def test_a_zone_id_that_is_a_digit_only_to_unicode_is_not_a_crash(
    tmp_path: Path,
) -> None:
    """``str.isdigit`` passes a superscript; ``int`` does not, and the record is skipped."""
    text = _TET.replace("(45 (12 wall bottom)())", "(45 (² wall bottom)())")
    poly = read(_write(tmp_path, text))
    assert "bottom" not in poly.element_tags
    assert "wall-12" in poly.element_tags


def test_explicit_node_lists_come_back_right_handed(tmp_path: Path) -> None:
    """A typed zone's own connectivity is oriented the way an assembled cell is."""
    text = """\
(2 3)
(10 (0 1 4 0))
(10 (1 1 4 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
))
(12 (0 1 1 0))
(12 (2 1 1 1 2)(
1 3 2 4
))
"""
    poly = read(_write(tmp_path, text))
    assert _names(poly) == ["tetra"]
    assert _measure(poly, 0) > 0
    assert sorted(_rows(poly)[0]) == [0, 1, 2, 3]


def test_a_zone_of_polygons_is_its_header_alone(tmp_path: Path) -> None:
    verts = np.array([[0, 0], [1, 0], [1.5, 1], [0.5, 2], [-0.5, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("polygon", np.array([[0, 1, 2, 3, 4]]))])
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert "(12 (2 1 1 1 7))\n" in text
    assert "(12 (2 1 1 1 0)(" not in text
    back = read(path)
    assert _names(back)[0] == "polygon"
    assert sorted(_rows(back)[0]) == [0, 1, 2, 3, 4]


def test_the_hexadecimal_scan_matches_the_per_token_path() -> None:
    from polyxios.codecs._fluent import _hex_body

    body = b"1 A ff\n 7fffffffffffff 10 \t0"
    values, starts = _hex_body(body + b")", len(body) + 1, body, "entry")
    assert values.tolist() == [1, 10, 255, 0x7FFFFFFFFFFFFF, 16, 0]
    assert starts.tolist() == [0, 2, 4, 8, 23, 27]
    # A sixteen-digit token takes the per-token path and reads the same.
    wide = b"1 A ff\n 007fffffffffffff 10 \t0"
    values, starts = _hex_body(wide + b")", len(wide) + 1, wide, "entry")
    assert values.tolist() == [1, 10, 255, 0x7FFFFFFFFFFFFF, 16, 0]
    assert starts.tolist() == [0, 2, 4, 8, 25, 29]
    empty, _ = _hex_body(b" \n ", 3, b" \n ", "entry")
    assert empty.size == 0
    with pytest.raises(CodecError, match="line 2: malformed entry 'zz'"):
        _hex_body(b"1\nzz)", 5, b"1\nzz", "entry")


def test_a_quoted_zone_name_may_carry_a_parenthesis(tmp_path: Path) -> None:
    text = _TET.replace("(45 (12 wall bottom)())", '(45 (12 wall "floor (z=0)")())')
    poly = read(_write(tmp_path, text))
    assert "floor (z=0)" in poly.element_tags
    assert poly.global_attrs[ZONE_TYPES_KEY]["floor (z=0)"] == "wall"


def test_a_skipped_section_holding_a_string_with_parentheses_is_stepped_over(
    tmp_path: Path,
) -> None:
    text = _TET + '(58 (1 1 1)("a ) b" (2 "(") 3))\n(0 "tail")\n'
    assert _names(read(_write(tmp_path, text)))[0] == "tetra"


def test_a_comment_that_never_closes_is_refused_at_the_end(tmp_path: Path) -> None:
    text = _TET + "(0 (never\n"
    with pytest.raises(CodecError, match=r"line \d+: a section opens and never closes"):
        read(_write(tmp_path, text))


def test_sniff_claims_a_comment_with_no_space_before_its_string() -> None:
    assert sniff(b'(0"Grid")\n(2 3)')


def test_a_remembered_interior_zone_with_a_boundary_face_is_written_as_a_wall(
    tmp_path: Path,
) -> None:
    """An interior zone holds faces between two cells only."""
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 2, 1]]))],
        element_tags={"skin": np.array([1])},
        global_attrs={ZONE_TYPES_KEY: {"skin": "interior"}},
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    text = path.read_text()
    assert "(13 (3 1 1 3 3)(" in text
    assert "(45 (3 wall skin)())" in text


def test_a_tag_opening_with_parent_is_not_read_as_that_type(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 2, 1]]))],
        element_tags={"parent-block": np.array([1])},
    )
    path = tmp_path / "m.fluent"
    write(poly, path)
    assert "(45 (3 wall parent-block)())" in path.read_text()


def test_two_count_prefixed_records_on_one_line_are_refused_by_line(
    tmp_path: Path,
) -> None:
    text = _TET.replace("3 1 3 2 1 0\n3 1 2 4 1 0", "3 1 3 2 1 0 3 1 2 4 1 0")
    with pytest.raises(CodecError, match=r"carries 1 record line\(s\)"):
        read(_write(tmp_path, text))
