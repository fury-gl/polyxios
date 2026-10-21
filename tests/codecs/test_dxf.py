"""DXF: group code / value pairs in sections, ASCII or binary.

Files are spelled pair by pair from the group code reference, so the reader
is tested against the format and not against the writer beside it.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import struct
import time
import tracemalloc
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs import _dxf
from polyxios.codecs._dxf import read, write
from polyxios.exceptions import CodecError, LazyReadError

_T = ELEMENT_TYPES

Pair = tuple[int, object]


def _xyz(first: int, point) -> list[Pair]:
    return [(first + 10 * k, c) for k, c in enumerate(point)]


def _face(points, layer: str = "0") -> list[Pair]:
    pairs: list[Pair] = [(0, "3DFACE"), (8, layer)]
    for k, p in enumerate(points):
        pairs += _xyz(10 + k, p)
    return pairs


def _line(a, b, layer: str = "0") -> list[Pair]:
    return [(0, "LINE"), (8, layer), *_xyz(10, a), *_xyz(11, b)]


def _point(p, layer: str = "0") -> list[Pair]:
    return [(0, "POINT"), (8, layer), *_xyz(10, p)]


def _polyline(flags: int, vertices, *, layer: str = "0", head=()) -> list[Pair]:
    pairs: list[Pair] = [(0, "POLYLINE"), (8, layer), (66, 1), *head, (70, flags)]
    for v in vertices:
        coords, vflags, extra = v
        pairs += [(0, "VERTEX"), (8, layer), *_xyz(10, coords), (70, vflags), *extra]
    return [*pairs, (0, "SEQEND")]


def _polyface(points, faces, layer: str = "0") -> list[Pair]:
    verts = [(p, 192, ()) for p in points]
    verts += [
        ((0, 0, 0), 128, tuple((71 + j, i) for j, i in enumerate(face)))
        for face in faces
    ]
    head = ((71, len(points)), (72, len(faces)))
    return _polyline(64, verts, layer=layer, head=head)


def _block(name: str, entities: list[Pair], base=(0, 0, 0)) -> list[Pair]:
    return [
        (0, "BLOCK"),
        (8, "0"),
        (2, name),
        (70, 0),
        *_xyz(10, base),
        *entities,
        (0, "ENDBLK"),
    ]


def _insert(block: str, *, layer: str = "0", at=(0, 0, 0), **fields) -> list[Pair]:
    codes = {
        "sx": 41,
        "sy": 42,
        "sz": 43,
        "rotation": 50,
        "cols": 70,
        "rows": 71,
        "dx": 44,
        "dy": 45,
        "nx": 210,
        "ny": 220,
        "nz": 230,
    }
    pairs: list[Pair] = [(0, "INSERT"), (8, layer), (2, block), *_xyz(10, at)]
    pairs += [(codes[key], value) for key, value in fields.items()]
    return pairs


def _section(name: str, pairs: list[Pair]) -> list[Pair]:
    return [(0, "SECTION"), (2, name), *pairs, (0, "ENDSEC")]


def _doc(
    entities: list[Pair],
    *,
    header: list[Pair] | None = None,
    blocks: list[Pair] | None = None,
) -> list[Pair]:
    pairs: list[Pair] = []
    if header is not None:
        pairs += _section("HEADER", header)
    if blocks is not None:
        pairs += _section("BLOCKS", blocks)
    return [*pairs, *_section("ENTITIES", entities), (0, "EOF")]


def _ascii(pairs: list[Pair]) -> bytes:
    return "".join(f"{code:>3}\n{value}\n" for code, value in pairs).encode("utf-8")


def _file(tmp_path: Path, pairs: list[Pair] | bytes, name: str = "a.dxf") -> Path:
    path = tmp_path / name
    path.write_bytes(pairs if isinstance(pairs, bytes) else _ascii(pairs))
    return path


def _binary(pairs: list[Pair], *, wide: bool = True) -> bytes:
    """Spell pairs as a binary DXF, the value type taken from the group code."""
    out = bytearray(_dxf._BINARY_SENTINEL)
    for code, value in pairs:
        if wide:
            out += struct.pack("<h", code)
        elif code >= 255:
            out += b"\xff" + struct.pack("<h", code)
        else:
            out.append(code)
        kind = _dxf._binary_kind(code)
        if kind == "s":
            out += str(value).encode() + b"\x00"
        elif kind == "b":
            out += bytes([len(value)]) + value
        else:
            out += struct.pack("<" + kind, value)
    return bytes(out)


def _cells(mesh) -> list[list[int]]:
    off = mesh.offsets.tolist()
    conn = mesh.connectivity.tolist()
    return [conn[a:b] for a, b in zip(off[:-1], off[1:], strict=True)]


def _coords(mesh) -> list[list[list[float]]]:
    return [mesh.vertices[c].tolist() for c in _cells(mesh)]


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


def test_a_3dface_is_a_quad_or_a_triangle(tmp_path: Path) -> None:
    quad = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    tri = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (1, 1, 1)]
    mesh = read(_file(tmp_path, _doc(_face(quad) + _face(tri))))
    assert mesh.element_types.tolist() == [_T["quad"], _T["triangle"]]
    assert _coords(mesh) == [
        [list(map(float, p)) for p in quad],
        [list(map(float, p)) for p in tri[:3]],
    ]


def test_a_3dface_without_a_fourth_corner_is_a_triangle(tmp_path: Path) -> None:
    pairs = [(0, "3DFACE"), (8, "0"), *_xyz(10, (0, 0, 0)), *_xyz(11, (1, 0, 0))]
    pairs += _xyz(12, (0, 1, 0))
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.element_types.tolist() == [_T["triangle"]]


@pytest.mark.parametrize("kind", ["3DFACE", "SOLID"])
def test_a_face_missing_a_required_corner_is_skipped(tmp_path: Path, kind) -> None:
    pairs = [(0, kind), (8, "0"), *_xyz(10, (0, 0, 0)), *_xyz(11, (1, 0, 0))]
    pairs += _face([(0, 0, 5), (1, 0, 5), (0, 1, 5)])
    with pytest.warns(UserWarning, match=f"1 {kind} without corner 1, 2 or 3"):
        mesh = read(_file(tmp_path, _doc(pairs)))
    assert _coords(mesh) == [[[0, 0, 5], [1, 0, 5], [0, 1, 5]]]


def test_lines_and_points(tmp_path: Path) -> None:
    mesh = read(_file(tmp_path, _doc(_line((0, 0, 0), (1, 2, 3)) + _point((5, 5, 5)))))
    assert mesh.element_types.tolist() == [_T["line"], _T["vertex"]]
    assert _coords(mesh) == [[[0, 0, 0], [1, 2, 3]], [[5, 5, 5]]]


def test_vertices_at_one_place_are_one_vertex(tmp_path: Path) -> None:
    pairs = _face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)])
    pairs += _face([(0, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0)])
    pairs += _point((0, 0, 0))
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert len(mesh.vertices) == 4
    assert _cells(mesh) == [[0, 1, 2], [0, 2, 3], [0]]


def test_a_polyface_mesh(tmp_path: Path) -> None:
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    pairs = _polyface(points, [(1, 2, 3, 4), (1, -2, 5), (4, 5)])
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.element_types.tolist() == [_T["quad"], _T["triangle"], _T["line"]]
    assert _coords(mesh)[1] == [[0, 0, 0], [1, 0, 0], [0, 0, 1]]


def test_a_line_or_point_missing_its_coordinates_is_skipped(tmp_path: Path) -> None:
    pairs = [(0, "LINE"), (8, "0"), *_xyz(10, (1, 1, 1))]
    pairs += [(0, "POINT"), (8, "0"), *_point((2, 2, 2))]
    with pytest.warns(UserWarning, match="1 LINE without a start or an end, 1 POINT"):
        mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.vertices.tolist() == [[2, 2, 2]]
    assert mesh.element_types.tolist() == [_T["vertex"]]


def test_a_polyface_face_without_vertices_is_skipped_with_a_count(
    tmp_path: Path,
) -> None:
    pairs = _polyface([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(1, 2, 3), (0, 0, 0)])
    with pytest.warns(UserWarning, match="1 polyface face without vertices"):
        mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.element_types.tolist() == [_T["triangle"]]


def test_a_code_spelled_twice_keeps_its_first_value(tmp_path: Path) -> None:
    pairs = [*_point((1, 2, 3), layer="A"), (10, 9.0), (8, "B")]
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.vertices.tolist() == [[1, 2, 3]]
    assert list(mesh.element_tags) == ["A"]


def test_a_polyface_face_past_its_vertices_is_refused(tmp_path: Path) -> None:
    pairs = _polyface([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(1, 2, 4)])
    with pytest.raises(CodecError, match="names vertex 4 but the mesh holds 3"):
        read(_file(tmp_path, _doc(pairs)))


def _grid(m: int, n: int) -> list:
    return [((i, j, i * j), 64, ()) for i in range(m) for j in range(n)]


def test_an_open_polygon_mesh_is_quads(tmp_path: Path) -> None:
    pairs = _polyline(16, _grid(3, 4), head=((71, 3), (72, 4)))
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert len(mesh.element_types) == 2 * 3
    assert set(mesh.element_types.tolist()) == {_T["quad"]}
    assert _coords(mesh)[0] == [[0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 0]]


def test_a_polygon_mesh_closed_both_ways_wraps(tmp_path: Path) -> None:
    pairs = _polyline(16 | 1 | 32, _grid(3, 4), head=((71, 3), (72, 4)))
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert len(mesh.element_types) == 3 * 4
    assert _coords(mesh)[-1] == [[2, 3, 6], [2, 0, 0], [0, 0, 0], [0, 3, 0]]


def test_a_polygon_mesh_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    pairs = _polyline(16, _grid(3, 3), head=((71, 3), (72, 4)))
    with pytest.raises(CodecError, match="3 x 4 polygon mesh holds 9 vertices"):
        read(_file(tmp_path, _doc(pairs)))


def test_a_smoothed_polygon_mesh_reads_its_surface_vertices(tmp_path: Path) -> None:
    control = [((9, 9, 9), 16, ())] * 4
    head = ((71, 2), (72, 2), (73, 2), (74, 3), (75, 6))
    pairs = _polyline(16 | 4, control + _grid(2, 3), head=head)
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert len(mesh.element_types) == 2
    assert [9.0, 9.0, 9.0] not in mesh.vertices.tolist()


def test_a_closed_3d_polyline_repeats_its_first_vertex(tmp_path: Path) -> None:
    verts = [((0, 0, 0), 32, ()), ((1, 0, 1), 32, ()), ((1, 1, 2), 32, ())]
    mesh = read(_file(tmp_path, _doc(_polyline(8 | 1, verts))))
    assert mesh.element_types.tolist() == [_T["poly_line"]]
    assert _cells(mesh) == [[0, 1, 2, 0]]


def test_a_two_vertex_polyline_is_a_line(tmp_path: Path) -> None:
    verts = [((0, 0, 0), 32, ()), ((1, 0, 1), 32, ())]
    mesh = read(_file(tmp_path, _doc(_polyline(8, verts))))
    assert mesh.element_types.tolist() == [_T["line"]]


def test_a_spline_polyline_skips_its_control_points(tmp_path: Path) -> None:
    verts = [((5, 5, 5), 16, ()), ((0, 0, 0), 8, ()), ((1, 0, 0), 8, ())]
    mesh = read(_file(tmp_path, _doc(_polyline(8 | 4, verts))))
    assert mesh.vertices.tolist() == [[0, 0, 0], [1, 0, 0]]


def test_a_2d_polyline_is_in_its_object_coordinate_system(tmp_path: Path) -> None:
    # Extrusion -z: the OCS x axis is world -x, its elevation world -z.
    verts = [((1, 2, 0), 0, ()), ((3, 4, 0), 0, ())]
    head = (*_xyz(10, (0, 0, 5)), (210, 0), (220, 0), (230, -1))
    mesh = read(_file(tmp_path, _doc(_polyline(0, verts, head=head))))
    np.testing.assert_allclose(mesh.vertices, [[-1, 2, -5], [-3, 4, -5]])


def test_a_bulge_is_read_straight_with_a_warning(tmp_path: Path) -> None:
    verts = [((0, 0, 0), 0, ((42, 1.0),)), ((1, 0, 0), 0, ())]
    with pytest.warns(UserWarning, match=r"1 polyline segment\(s\) are arcs"):
        mesh = read(_file(tmp_path, _doc(_polyline(0, verts))))
    assert len(mesh.vertices) == 2


def test_an_lwpolyline(tmp_path: Path) -> None:
    pairs = [(0, "LWPOLYLINE"), (8, "W"), (90, 3), (70, 1), (38, 2.0)]
    pairs += [(10, 0), (20, 0), (10, 1), (20, 0), (10, 1), (20, 1)]
    pairs += [(210, 1), (220, 0), (230, 0)]
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.element_types.tolist() == [_T["poly_line"]]
    assert _cells(mesh) == [[0, 1, 2, 0]]
    # Extrusion +x: OCS x is world y, OCS y world z, elevation world x.
    np.testing.assert_allclose(mesh.vertices, [[2, 0, 0], [2, 1, 0], [2, 1, 1]])


def test_an_lwpolyline_with_unpaired_coordinates_is_refused(tmp_path: Path) -> None:
    pairs = [(0, "LWPOLYLINE"), (8, "0"), (90, 2), (10, 0), (20, 0), (10, 1)]
    with pytest.raises(CodecError, match="2 x but 1 y"):
        read(_file(tmp_path, _doc(pairs)))


def test_a_solid_corners_run_one_two_four_three(tmp_path: Path) -> None:
    pairs = [(0, "SOLID"), (8, "0")]
    pairs += _xyz(10, (0, 0, 3)) + _xyz(11, (1, 0, 3)) + _xyz(12, (0, 1, 3))
    pairs += _xyz(13, (1, 1, 3))
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert _coords(mesh) == [[[0, 0, 3], [1, 0, 3], [1, 1, 3], [0, 1, 3]]]


def test_a_mesh_entity(tmp_path: Path) -> None:
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)]
    pairs: list[Pair] = [(0, "MESH"), (8, "M"), (100, "AcDbSubDMesh"), (71, 2), (72, 0)]
    pairs += [(91, 0), (92, len(points))]
    for p in points:
        pairs += _xyz(10, p)
    faces = [3, 0, 1, 4, 4, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4]
    pairs += [(93, len(faces)), *[(90, f) for f in faces], (94, 1), (90, 0), (90, 1)]
    pairs += [(95, 0), (90, 0)]
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.element_types.tolist() == [_T["triangle"], _T["quad"], _T["polygon"]]
    assert _cells(mesh)[0] == [0, 1, 4]


@pytest.mark.parametrize(
    ("faces", "match"),
    [([3, 0, 1, 9], "names vertex 9"), ([4, 0, 1], "face list is cut short")],
)
def test_a_bad_mesh_entity_is_refused(tmp_path: Path, faces, match) -> None:
    pairs: list[Pair] = [(0, "MESH"), (8, "0"), (92, 3)]
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        pairs += _xyz(10, p)
    pairs += [(93, len(faces)), *[(90, f) for f in faces]]
    with pytest.raises(CodecError, match=match):
        read(_file(tmp_path, _doc(pairs)))


def test_other_entities_are_skipped_with_a_count(tmp_path: Path) -> None:
    pairs = [(0, "CIRCLE"), (8, "0"), *_xyz(10, (0, 0, 0)), (40, 1.0)]
    pairs += [(0, "TEXT"), (8, "0"), (1, "hi"), (0, "CIRCLE"), (8, "0")]
    pairs += _point((1, 1, 1))
    with pytest.warns(UserWarning, match="skipped: 2 CIRCLE, 1 TEXT"):
        mesh = read(_file(tmp_path, _doc(pairs)))
    assert len(mesh.element_types) == 1


def test_paper_space_entities_are_not_read(tmp_path: Path) -> None:
    pairs = _point((1, 1, 1)) + [(0, "POINT"), (8, "0"), (67, 1), *_xyz(10, (2, 2, 2))]
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.vertices.tolist() == [[1, 1, 1]]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

_TRI = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)]


def test_an_insert_places_its_block(tmp_path: Path) -> None:
    blocks = _block("B", _face(_TRI), base=(1, 0, 0))
    entities = _insert("B", at=(10, 0, 0), sx=2, sy=2, sz=2, rotation=90)
    mesh = read(_file(tmp_path, _doc(entities, blocks=blocks)))
    # (p - base) * 2, turned 90 degrees, moved to (10, 0, 0).
    np.testing.assert_allclose(
        mesh.vertices, [[10, -2, 0], [10, 0, 0], [8, -2, 0]], atol=1e-12
    )


def test_an_insert_array_places_one_copy_per_cell(tmp_path: Path) -> None:
    blocks = _block("B", _point((0, 0, 0)))
    entities = _insert("B", cols=3, rows=2, dx=10, dy=5)
    mesh = read(_file(tmp_path, _doc(entities, blocks=blocks)))
    assert sorted(map(tuple, mesh.vertices.tolist())) == sorted(
        (10.0 * c, 5.0 * r, 0.0) for r in range(2) for c in range(3)
    )


def test_an_insert_follows_its_extrusion(tmp_path: Path) -> None:
    blocks = _block("B", _point((1, 0, 0)))
    entities = _insert("B", at=(0, 0, 2), nx=0, ny=0, nz=-1)
    mesh = read(_file(tmp_path, _doc(entities, blocks=blocks)))
    np.testing.assert_allclose(mesh.vertices, [[-1, 0, -2]])


def test_a_mirroring_insert_keeps_faces_facing_out(tmp_path: Path) -> None:
    blocks = _block("B", _face(_TRI))
    mesh = read(_file(tmp_path, _doc(_insert("B", sx=-1), blocks=blocks)))
    a, b, c = mesh.vertices[_cells(mesh)[0]]
    assert np.cross(b - a, c - a)[2] > 0


def test_nested_inserts_compose(tmp_path: Path) -> None:
    blocks = _block("INNER", _point((1, 0, 0)))
    blocks += _block("OUTER", _insert("INNER", at=(0, 5, 0)))
    mesh = read(_file(tmp_path, _doc(_insert("OUTER", at=(100, 0, 0)), blocks=blocks)))
    np.testing.assert_allclose(mesh.vertices, [[101, 5, 0]])


def test_block_names_are_case_insensitive(tmp_path: Path) -> None:
    blocks = _block("Part", _point((1, 0, 0)))
    mesh = read(_file(tmp_path, _doc(_insert("PART"), blocks=blocks)))
    assert len(mesh.vertices) == 1


def test_layer_0_in_a_block_takes_the_layer_of_the_insert(tmp_path: Path) -> None:
    blocks = _block("B", _point((0, 0, 0)) + _point((1, 0, 0), layer="OWN"))
    entities = _insert("B", layer="A") + _insert("B", layer="C", at=(0, 9, 0))
    mesh = read(_file(tmp_path, _doc(entities, blocks=blocks)))
    assert {k: v.tolist() for k, v in mesh.element_tags.items()} == {
        "A": [0],
        "OWN": [1, 3],
        "C": [2],
    }


def test_an_insert_of_a_missing_block_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="names block 'NOPE'"):
        read(_file(tmp_path, _doc(_insert("NOPE"))))


def test_a_block_inserting_itself_is_refused(tmp_path: Path) -> None:
    blocks = _block("A", _insert("B")) + _block("B", _insert("A"))
    with pytest.raises(CodecError, match="block A -> B -> A inserts itself"):
        read(_file(tmp_path, _doc(_insert("A"), blocks=blocks)))


def test_inserts_expanding_past_the_file_size_are_refused(tmp_path: Path) -> None:
    blocks = _block("B", _point((0, 0, 0)))
    blocks += _block("C", _insert("B", cols=3000, rows=3000))
    entities = _insert("C", cols=3000, rows=3000)
    with pytest.raises(CodecError, match="expand to more than"):
        read(_file(tmp_path, _doc(entities, blocks=blocks)))


def test_a_million_copies_of_one_point_are_refused(tmp_path: Path) -> None:
    # A file of a few hundred bytes: the copies cost work whatever their size.
    blocks = _block("B", _point((0, 0, 0)))
    entities = _insert("B", cols=1000, rows=1000)
    with pytest.raises(CodecError, match="each copy counted as 16 more"):
        read(_file(tmp_path, _doc(entities, blocks=blocks)))


def test_an_insert_array_runs_row_by_row(tmp_path: Path) -> None:
    blocks = _block("B", _line((0, 0, 0), (0, 0, 1)))
    entities = _insert("B", cols=3, rows=2, dx=10, dy=5, rotation=90)
    mesh = read(_file(tmp_path, _doc(entities, blocks=blocks)))
    starts = [c[0] for c in _coords(mesh)]
    np.testing.assert_allclose(
        starts,
        [[-5.0 * r, 10.0 * c, 0.0] for r in range(2) for c in range(3)],
        atol=1e-12,
    )
    assert mesh.element_types.tolist() == [_T["line"]] * 6


def test_a_deep_chain_of_nested_blocks_is_read(tmp_path: Path) -> None:
    blocks = _block("B0", _point((0, 0, 0)))
    for k in range(1, 2000):
        blocks += _block(f"B{k}", _insert(f"B{k - 1}", at=(1, 0, 0)))
    mesh = read(_file(tmp_path, _doc(_insert("B1999"), blocks=blocks)))
    np.testing.assert_allclose(mesh.vertices, [[1999, 0, 0]])


def test_a_block_defined_twice_keeps_the_first_with_a_warning(
    tmp_path: Path,
) -> None:
    blocks = _block("B", _point((1, 0, 0))) + _block("b", _point((2, 0, 0)))
    with pytest.warns(UserWarning, match=r"block\(s\) \['b'\] are defined again"):
        mesh = read(_file(tmp_path, _doc(_insert("B"), blocks=blocks)))
    assert mesh.vertices.tolist() == [[1, 0, 0]]


# ---------------------------------------------------------------------------
# Layers, header and text
# ---------------------------------------------------------------------------


def test_layers_are_tags_in_the_order_they_come(tmp_path: Path) -> None:
    pairs = _point((0, 0, 0), layer="Z") + _point((1, 0, 0)) + _point((2, 0, 0), "Z")
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert {k: v.tolist() for k, v in mesh.element_tags.items()} == {
        "Z": [0, 2],
        "0": [1],
    }
    assert list(mesh.element_tags) == ["Z", "0"]


def test_layer_names_ignore_case(tmp_path: Path) -> None:
    pairs = _point((0, 0, 0), "Walls") + _point((1, 0, 0), "WALLS")
    pairs += _point((2, 0, 0), "walls")
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert {k: v.tolist() for k, v in mesh.element_tags.items()} == {"Walls": [0, 1, 2]}


def test_a_unicode_escape_in_a_layer_name_is_decoded(tmp_path: Path) -> None:
    mesh = read(_file(tmp_path, _doc(_point((0, 0, 0), layer="W\\U+00E4nde"))))
    assert list(mesh.element_tags) == ["Wände"]


def test_an_r12_layer_name_is_read_in_its_code_page(tmp_path: Path) -> None:
    header = [(9, "$ACADVER"), (1, "AC1009"), (9, "$DWGCODEPAGE"), (3, "ANSI_1252")]
    raw = _ascii(_doc(_point((0, 0, 0), layer="LAYER_X"), header=header))
    path = _file(tmp_path, raw.replace(b"LAYER_X", "Wände".encode("cp1252")))
    assert list(read(path).element_tags) == ["Wände"]


def test_a_version_spelled_in_lower_case_keeps_its_code_page(tmp_path: Path) -> None:
    header = [(9, "$ACADVER"), (1, "ac1009"), (9, "$DWGCODEPAGE"), (3, "ANSI_1253")]
    raw = _ascii(_doc(_point((0, 0, 0), layer="LAYER_X"), header=header))
    path = _file(tmp_path, raw.replace(b"LAYER_X", "Ωμέγα".encode("cp1253")))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(read(path).element_tags) == ["Ωμέγα"]


@pytest.mark.parametrize(
    ("escaped", "name"),
    [
        ("\\M+193FA\\M+1967B", "日本"),
        ("\\M+5D6D0\\M+5CEC4", "中文"),
        ("A\\M+1FFFF", "A\\M+1FFFF"),
    ],
)
def test_a_code_page_escape_in_a_layer_name_is_decoded(
    tmp_path: Path, escaped: str, name: str
) -> None:
    mesh = read(_file(tmp_path, _doc(_point((0, 0, 0), layer=escaped))))
    assert list(mesh.element_tags) == [name]


def test_a_name_padded_with_nul_bytes_ends_at_the_first(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point((0, 0, 0), layer="LAYER_X")))
    path = _file(
        tmp_path,
        raw.replace(b"\nPOINT\n", b"\nPOINT\x00\n").replace(b"LAYER_X", b"L1\x00\x00"),
    )
    mesh = read(path)
    assert list(mesh.element_tags) == ["L1"]
    write(mesh, tmp_path / "back.dxf")


def test_a_name_that_is_not_the_utf8_its_version_requires_warns(
    tmp_path: Path,
) -> None:
    header = [(9, "$ACADVER"), (1, "AC1027")]
    raw = _ascii(_doc(_point((0, 0, 0), layer="LAYER_X"), header=header))
    path = _file(tmp_path, raw.replace(b"LAYER_X", "Wände".encode("cp1252")))
    with pytest.warns(UserWarning, match="1 string.* not the UTF-8 AC1027 requires"):
        assert list(read(path).element_tags) == ["Wände"]


def test_header_units_and_version_are_global_attrs(tmp_path: Path) -> None:
    header = [(9, "$ACADVER"), (1, "AC1027"), (9, "$INSUNITS"), (70, 6)]
    mesh = read(_file(tmp_path, _doc(_point((0, 0, 0)), header=header)))
    assert mesh.global_attrs == {"dxf_version": "AC1027", "unit": "meter"}


def test_an_unknown_insunits_code_is_dropped_with_a_warning(tmp_path: Path) -> None:
    header = [(9, "$INSUNITS"), (70, 99)]
    with pytest.warns(UserWarning, match=r"\$INSUNITS 99 is not a unit"):
        mesh = read(_file(tmp_path, _doc(_point((0, 0, 0)), header=header)))
    assert mesh.global_attrs == {}


def test_comments_crlf_and_a_bom_are_read(tmp_path: Path) -> None:
    raw = _ascii([(999, "made by hand"), *_doc(_point((1, 2, 3)))])
    path = _file(tmp_path, b"\xef\xbb\xbf" + raw.replace(b"\n", b"\r\n"))
    assert read(path).vertices.tolist() == [[1, 2, 3]]


def test_section_and_entity_names_padded_with_spaces_are_read(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point((1, 2, 3))))
    raw = raw.replace(b"\nENTITIES\n", b"\nENTITIES  \n")
    raw = raw.replace(b"\nPOINT\n", b"\nPOINT \n")
    assert read(_file(tmp_path, raw)).vertices.tolist() == [[1, 2, 3]]


def test_an_empty_drawing_reads_as_an_empty_mesh(tmp_path: Path) -> None:
    mesh = read(_file(tmp_path, _doc([])))
    assert mesh.vertices.shape == (0, 3)
    assert len(mesh.element_types) == 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [b"", b"\r\n", b"  0\nEOF\n", b"999\nnotes\n"])
def test_a_file_holding_no_section_is_refused(tmp_path: Path, raw: bytes) -> None:
    with pytest.raises(CodecError, match="holds no DXF section"):
        read(_file(tmp_path, raw))


def test_a_group_code_out_of_range_is_refused(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point((0, 0, 0))))
    raw = raw.replace(b" 20\n", b"99999999999999999999\n", 1)
    with pytest.raises(CodecError, match="should hold a group code"):
        read(_file(tmp_path, raw))


def test_a_group_code_past_16_bits_is_refused(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point((0, 0, 0))))
    raw = raw.replace(b" 20\n", b"40000\n", 1)
    with pytest.raises(CodecError, match="line .* should hold a group code"):
        read(_file(tmp_path, raw))


@pytest.mark.parametrize("padded", [b" 20", b"0.5"])
def test_a_padded_line_does_not_widen_every_other(tmp_path: Path, padded) -> None:
    # numpy sizes a bytes array by its longest entry: one line of 20000
    # characters among 5000 points would cost 300 MB at that width.
    pairs: list[Pair] = []
    for k in range(5000):
        pairs += _point((k, 0.5, 0))
    raw = _ascii(_doc(pairs)).replace(padded + b"\n", padded + b" " * 20000 + b"\n", 1)
    tracemalloc.start()
    try:
        mesh = read(_file(tmp_path, raw))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(mesh.vertices) == 5000
    assert peak < 50 * 2**20


def test_a_group_67_that_is_not_a_number_names_its_code(tmp_path: Path) -> None:
    pairs = [*_point((0, 0, 0)), (67, "x")]
    with pytest.raises(CodecError, match="an entity's group 67 should be a number"):
        read(_file(tmp_path, _doc(pairs)))


def test_a_block_without_endblk_is_refused(tmp_path: Path) -> None:
    blocks = _block("A", _face(_TRI))[:-1] + _block("B", _face(_TRI))
    entities = _insert("B")
    with pytest.raises(CodecError, match="block 'A' never ends"):
        read(_file(tmp_path, _doc(entities, blocks=blocks)))


def test_a_huge_polyface_index_is_refused_quietly(tmp_path: Path) -> None:
    pairs = _polyface([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(1, 2, "1e300")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(CodecError, match="names vertex 1e\\+300 but"):
            read(_file(tmp_path, _doc(pairs)))


@pytest.mark.parametrize(("flags", "arcs"), [(0, 0), (1, 1)])
def test_the_last_bulge_of_an_lwpolyline_bends_only_when_closed(
    tmp_path: Path, flags: int, arcs: int
) -> None:
    pairs = [(0, "LWPOLYLINE"), (8, "0"), (90, 2), (70, flags)]
    pairs += [(10, 0.0), (20, 0.0), (10, 1.0), (20, 0.0), (42, 1.0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read(_file(tmp_path, _doc(pairs)))
    bends = [w for w in caught if "bulge" in str(w.message)]
    assert len(bends) == arcs
    if arcs:
        assert "1 polyline segment(s)" in str(bends[0].message)


def test_a_group_code_that_is_not_an_integer_is_refused(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point((0, 0, 0)))).replace(b" 20\n", b"2x\n", 1)
    with pytest.raises(CodecError, match="should hold a group code"):
        read(_file(tmp_path, raw))


def test_a_coordinate_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point(("abc", 0, 0))))
    with pytest.raises(CodecError, match="POINT group 10 should be a number"):
        read(_file(tmp_path, raw))


def test_a_file_ending_on_a_group_code_is_refused(tmp_path: Path) -> None:
    raw = _ascii(_doc(_point((0, 0, 0))))[: -len(b"EOF\n")]
    with pytest.raises(CodecError, match="truncated"):
        read(_file(tmp_path, raw))


def test_a_section_that_never_ends_is_refused(tmp_path: Path) -> None:
    raw = _ascii(_section("ENTITIES", _point((0, 0, 0)))[:-1])
    with pytest.raises(CodecError, match="section ENTITIES never ends"):
        read(_file(tmp_path, raw))


def test_lazy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_file(tmp_path, _doc(_point((0, 0, 0)))), lazy=True)


# ---------------------------------------------------------------------------
# Binary DXF
# ---------------------------------------------------------------------------

_BINARY_DOC = _doc(
    _face([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], layer="F")
    + _polyface([(0, 0, 1), (1, 0, 1), (1, 1, 1)], [(1, 2, 3)])
    + [(0, "LWPOLYLINE"), (8, "W"), (90, 2), (70, 0), (10, 0.0), (20, 0.0)]
    + [(10, 1.0), (20, 0.0)],
    header=[(9, "$INSUNITS"), (70, 4)],
)


@pytest.mark.parametrize("wide", [True, False], ids=["r13", "r12"])
def test_binary_matches_ascii(tmp_path: Path, wide: bool) -> None:
    text = read(_file(tmp_path, _BINARY_DOC, "t.dxf"))
    binary = read(_file(tmp_path, _binary(_BINARY_DOC, wide=wide), "b.dxf"))
    np.testing.assert_array_equal(binary.vertices, text.vertices)
    np.testing.assert_array_equal(binary.connectivity, text.connectivity)
    np.testing.assert_array_equal(binary.element_types, text.element_types)
    assert binary.global_attrs == text.global_attrs == {"unit": "millimeter"}
    assert list(binary.element_tags) == ["F", "0", "W"]


def test_a_binary_dxf_opening_on_a_comment_is_read(tmp_path: Path) -> None:
    pairs = [(999, "made by hand"), *_BINARY_DOC]
    text = read(_file(tmp_path, _BINARY_DOC, "t.dxf"))
    binary = read(_file(tmp_path, _binary(pairs), "b.dxf"))
    np.testing.assert_array_equal(binary.vertices, text.vertices)
    assert list(binary.element_tags) == list(text.element_tags)


def test_a_truncated_binary_dxf_is_refused(tmp_path: Path) -> None:
    raw = _binary(_BINARY_DOC)[:-30]
    with pytest.raises(CodecError, match="binary DXF is truncated"):
        read(_file(tmp_path, raw))


def test_an_unknown_binary_group_code_is_refused(tmp_path: Path) -> None:
    raw = _dxf._BINARY_SENTINEL + struct.pack("<h", 0) + b"SECTION\x00"
    raw += struct.pack("<h", 2000) + b"\x00\x00"
    with pytest.raises(CodecError, match="binary group code 2000"):
        read(_file(tmp_path, raw))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)


def _round_trip(poly, tmp_path: Path, **kwargs):
    path = tmp_path / "out.dxf"
    write(poly, path, **kwargs)
    return read(path)


def test_writes_ascii_r12(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("quad", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "out.dxf"
    write(poly, path)
    text = path.read_text()
    assert text.startswith("  0\nSECTION\n  2\nHEADER\n  9\n$ACADVER\n  1\nAC1009\n")
    assert text.endswith("  0\nEOF\n")
    assert read(path).global_attrs == {"dxf_version": "AC1009"}


def test_faces_lines_points_and_polylines_round_trip(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[0, 2]])),
            ("vertex", np.array([[4]])),
            ("poly_line", np.array([[0, 1, 2, 0]])),
            ("poly_line", np.array([[3, 2, 1]])),
        ],
    )
    back = _round_trip(poly, tmp_path)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    original = [poly.vertices[c].tolist() for c in _cells(poly)]
    assert _coords(back) == original


def test_tag_groups_are_layers(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
        element_tags={"walls": np.array([1])},
    )
    back = _round_trip(poly, tmp_path)
    assert {k: v.tolist() for k, v in back.element_tags.items()} == {
        "0": [0],
        "walls": [1],
    }


def test_an_element_in_two_groups_stays_on_the_first(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]]))],
        element_tags={"a": np.array([0]), "b": np.array([0])},
    )
    with pytest.warns(UserWarning, match=r"\['b'\] name elements an earlier group"):
        back = _round_trip(poly, tmp_path)
    assert list(back.element_tags) == ["a"]


def test_a_group_on_layer_0_beside_untagged_elements_is_warned_about(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4], [0, 1, 2]]))],
        element_tags={"0": np.array([1])},
    )
    with pytest.warns(UserWarning, match=r"\['0'\] are written on layer 0"):
        back = _round_trip(poly, tmp_path)
    assert {k: v.tolist() for k, v in back.element_tags.items()} == {"0": [0, 1]}
    whole = make_polydata(
        _SQUARE, [("triangle", np.array([[0, 1, 4]]))], element_tags={"0": [0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _round_trip(whole, tmp_path)


def test_a_non_ascii_layer_name_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0]]))], element_tags={"Wände 😀": np.array([0])}
    )
    path = tmp_path / "out.dxf"
    write(poly, path)
    path.read_bytes().decode("ascii")
    assert list(read(path).element_tags) == ["Wände 😀"]


def test_layer_names_differing_in_case_share_a_layer(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("vertex", np.array([[0], [1], [2]]))],
        element_tags={"Walls": np.array([0]), "WALLS": np.array([1])},
    )
    with pytest.warns(
        UserWarning, match=r"\{'WALLS': 'Walls'\} are written on the layer"
    ):
        back = _round_trip(poly, tmp_path)
    assert {k: v.tolist() for k, v in back.element_tags.items()} == {
        "Walls": [0, 1],
        "0": [2],
    }
    text = (tmp_path / "out.dxf").read_text()
    assert text.count("  0\nLAYER\n") == 2


@pytest.mark.parametrize(
    ("label", "written"),
    [
        ("a/b", "a_b"),
        (" walls ", "walls"),
        ("x\\U+0041", "x_U+0041"),
        ("  ", "_"),
        ("", "_"),
    ],
)
def test_a_layer_name_no_reader_takes_back_is_rewritten(
    tmp_path: Path, label: str, written: str
) -> None:
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0]]))], element_tags={label: np.array([0])}
    )
    with pytest.warns(UserWarning, match="were rewritten"):
        back = _round_trip(poly, tmp_path)
    assert list(back.element_tags) == [written]


def test_groups_whose_names_match_once_rewritten_share_a_layer(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("vertex", np.array([[0], [1]]))],
        element_tags={"a_b": np.array([0]), "a/b": np.array([1])},
    )
    with pytest.warns(UserWarning, match=r"\{'a/b': 'a_b'\} are written on the layer"):
        with pytest.warns(UserWarning, match="were rewritten"):
            back = _round_trip(poly, tmp_path)
    assert {k: v.tolist() for k, v in back.element_tags.items()} == {"a_b": [0, 1]}


def test_degenerate_elements_are_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("polygon", np.array([[0, 1]])),
            ("triangle_strip", np.array([[0, 1]])),
            ("poly_line", np.array([[3]])),
        ],
    )
    with pytest.warns(UserWarning, match="hold too few vertices to draw"):
        back = _round_trip(poly, tmp_path)
    assert back.element_types.tolist() == [_T["triangle"]]


def test_a_mesh_of_degenerate_elements_only_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("poly_line", np.array([[3]]))])
    with pytest.raises(CodecError, match="too few vertices: 1 poly_line"):
        write(poly, tmp_path / "out.dxf")


def test_a_layer_name_with_a_newline_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0]]))], element_tags={"a\nb": np.array([0])}
    )
    with pytest.raises(CodecError, match="control character"):
        write(poly, tmp_path / "out.dxf")


def test_large_polygons_and_strips_are_split_facing_one_way(tmp_path: Path) -> None:
    hexagon = [[math.cos(a), math.sin(a), 0] for a in np.arange(6) * math.pi / 3]
    zigzag = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    poly = make_polydata(
        np.array(hexagon + zigzag),
        [
            ("polygon", np.array([[0, 1, 2, 3, 4, 5]])),
            ("triangle_strip", np.array([[6, 7, 8, 9]])),
        ],
    )
    back = _round_trip(poly, tmp_path)
    assert back.element_types.tolist() == [_T["triangle"]] * 6
    normals = [
        np.cross(b - a, c - a)[2]
        for a, b, c in (back.vertices[c] for c in _cells(back))
    ]
    assert all(n > 0 for n in normals[:4])
    assert normals[4] < 0 and normals[5] < 0


def test_a_pixel_is_written_as_a_quad(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("pixel", np.array([[0, 1, 3, 2]]))])
    back = _round_trip(poly, tmp_path)
    assert back.element_types.tolist() == [_T["quad"]]
    assert _coords(back) == [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]


def test_a_poly_vertex_is_one_point_per_vertex(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("poly_vertex", np.array([[0, 1, 2]]))])
    with pytest.warns(UserWarning, match="one POINT per vertex"):
        back = _round_trip(poly, tmp_path)
    assert back.element_types.tolist() == [_T["vertex"]] * 3


def test_volume_elements_are_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]])), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    with pytest.warns(UserWarning, match="1 tetra element"):
        back = _round_trip(poly, tmp_path)
    assert len(back.element_types) == 1


def test_a_mesh_of_volume_elements_only_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(CodecError, match="draws none of the mesh's elements"):
        write(poly, tmp_path / "out.dxf")


def test_a_vertex_that_is_not_finite_is_refused(tmp_path: Path) -> None:
    verts = _SQUARE.copy()
    verts[1, 0] = np.nan
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 4]]))])
    with pytest.raises(CodecError, match="NaN or infinite"):
        write(poly, tmp_path / "out.dxf")


def test_the_unit_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0]]))], global_attrs={"unit": "inch"}
    )
    assert _round_trip(poly, tmp_path).global_attrs["unit"] == "inch"


def test_a_unit_without_a_code_is_written_unitless(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0]]))], global_attrs={"unit": "cubit"}
    )
    with pytest.warns(UserWarning, match="has no \\$INSUNITS code"):
        back = _round_trip(poly, tmp_path)
    assert "unit" not in back.global_attrs


def test_big_runs_of_faces_split_into_several_polyface_meshes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(_dxf, "_POLYFACE_FACES", 2)
    rng = np.random.default_rng(0)
    verts = rng.random((30, 3))
    faces = np.array([rng.choice(30, 3, replace=False) for _ in range(7)])
    poly = make_polydata(verts, [("triangle", faces)])
    path = tmp_path / "out.dxf"
    write(poly, path)
    assert path.read_text().count("\n 70\n64\n") == 4
    back = read(path)
    assert _coords(back) == [verts[f].tolist() for f in faces]


def test_an_empty_mesh_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(np.empty((0, 3)), [])
    back = _round_trip(poly, tmp_path)
    assert len(back.element_types) == 0


def test_writes_to_a_buffer_and_through_the_registry(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("quad", np.array([[0, 1, 2, 3]]))])
    buf = io.BytesIO()
    write(poly, buf)
    path = tmp_path / "reg.dxf"
    polyxios.write(poly, path)
    assert path.read_bytes() == buf.getvalue()
    assert len(polyxios.read(path).element_types) == 1


def test_ezdxf_reads_what_is_written(tmp_path: Path) -> None:
    recover = pytest.importorskip("ezdxf.recover")
    poly = make_polydata(
        _SQUARE,
        [
            ("quad", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[0, 4]])),
            ("poly_line", np.array([[0, 1, 2, 0]])),
            ("vertex", np.array([[4]])),
        ],
        element_tags={"walls": np.array([0])},
        global_attrs={"unit": "meter"},
    )
    path = tmp_path / "out.dxf"
    write(poly, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc, auditor = recover.readfile(path)
    assert not auditor.errors
    kinds = [e.dxftype() for e in doc.modelspace()]
    assert kinds == ["POLYLINE", "LINE", "POLYLINE", "POINT"]
    assert doc.modelspace()[0].dxf.layer == "walls"
    # ezdxf leaves $INSUNITS out of the R12 files it writes, but reads it back
    # from an R12 header all the same.
    assert doc.header["$INSUNITS"] == 6


def test_ezdxf_audits_rewritten_layer_names_clean(tmp_path: Path) -> None:
    recover = pytest.importorskip("ezdxf.recover")
    tags = {"a/b": np.array([0]), "Wände": np.array([1]), "x" * 40: np.array([2])}
    poly = make_polydata(
        _SQUARE, [("vertex", np.array([[0], [1], [2]]))], element_tags=tags
    )
    path = tmp_path / "out.dxf"
    with pytest.warns(UserWarning, match="were rewritten"):
        write(poly, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc, auditor = recover.readfile(path)
    assert not auditor.errors
    assert [e.dxf.layer for e in doc.modelspace()] == ["a_b", "Wände", "x" * 40]


def test_a_mesh_entity_holding_fewer_vertices_than_it_declares_is_refused(
    tmp_path: Path,
) -> None:
    pairs: list[Pair] = [(0, "MESH"), (8, "0"), (92, 3), *_xyz(10, (0, 0, 0))]
    pairs += [(93, 4), (90, 3), (90, 0), (90, 0), (90, 0)]
    with pytest.raises(CodecError, match="declares 3 vertices but holds 1"):
        read(_file(tmp_path, _doc(pairs)))


def test_inserts_of_a_mesh_with_many_faces_count_its_connectivity(
    tmp_path: Path,
) -> None:
    # Three vertices and 2000 faces: the copies cost their faces, not 19 each.
    pairs: list[Pair] = [(0, "MESH"), (8, "0"), (92, 3)]
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        pairs += _xyz(10, p)
    faces = [3, 0, 1, 2] * 2000
    pairs += [(93, len(faces)), *[(90, f) for f in faces]]
    blocks = _block("B", pairs)
    entities = _insert("B", cols=10, rows=100)
    with pytest.raises(CodecError, match="vertices and connectivity entries"):
        read(_file(tmp_path, _doc(entities, blocks=blocks)))


def test_an_insert_placing_a_vertex_past_the_float_range_is_refused(
    tmp_path: Path,
) -> None:
    blocks = _block("B", _point((10, 0, 0)))
    entities = _insert("B", sx=1e308)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(CodecError, match="past the float range"):
            read(_file(tmp_path, _doc(entities, blocks=blocks)))


@pytest.mark.parametrize(
    "tail",
    [b"\x1a", b"\x00" * 7, b"junk\n", b"junk\nmore\n", b"  0\nEOF\n"],
    ids=["dos-eof", "nul-padding", "odd-junk", "even-junk", "second-eof"],
)
def test_what_follows_the_eof_pair_is_ignored(tmp_path: Path, tail: bytes) -> None:
    raw = _ascii(_doc(_point((1, 2, 3))))
    mesh = read(_file(tmp_path, raw + tail))
    assert mesh.vertices.tolist() == [[1, 2, 3]]


def test_eof_as_a_value_is_not_the_end_of_the_file(tmp_path: Path) -> None:
    pairs = _point((1, 2, 3)) + [(0, "TEXT"), (8, "0"), (1, "EOF")]
    pairs += _point((4, 5, 6))
    with pytest.warns(UserWarning, match="1 TEXT"):
        mesh = read(_file(tmp_path, _doc(pairs)))
    assert mesh.vertices.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("code", [310, 1004])
def test_a_binary_chunk_cut_short_is_refused(tmp_path: Path, code: int) -> None:
    raw = _binary(_BINARY_DOC[:-1]) + struct.pack("<h", code) + b"\xc8" + b"x" * 5
    with pytest.raises(CodecError, match=f"truncated at byte {len(raw) - 6}"):
        read(_file(tmp_path, raw))


def test_a_binary_chunk_holding_nul_bytes_is_read_whole(tmp_path: Path) -> None:
    chunk = b"\x00\x01\x00ab\x00"
    pairs = [*_point((1, 2, 3)), (310, chunk), *_point((4, 5, 6))]
    mesh = read(_file(tmp_path, _binary(_doc(pairs))))
    assert mesh.vertices.tolist() == [[1, 2, 3], [4, 5, 6]]
    codes, values = _dxf._binary_pairs(_binary(_doc(pairs)), "a.dxf")
    assert values[codes.tolist().index(310)] == chunk


def _one_point_per_layer(tmp_path: Path, n: int) -> Path:
    body = "".join(
        f"  0\nPOINT\n  8\nL{i}\n 10\n{i}\n 20\n0\n 30\n0\n" for i in range(n)
    )
    raw = f"  0\nSECTION\n  2\nENTITIES\n{body}  0\nENDSEC\n  0\nEOF\n".encode()
    return _file(tmp_path, raw, f"layers{n}.dxf")


def test_tagging_one_element_per_layer_is_not_quadratic(tmp_path: Path) -> None:
    small = _one_point_per_layer(tmp_path, 25_000)
    large = _one_point_per_layer(tmp_path, 100_000)
    times = []
    for path in (small, large):
        start = time.perf_counter()
        mesh = read(path)
        times.append(time.perf_counter() - start)
    assert len(mesh.element_tags) == 100_000
    assert mesh.element_tags["L99999"].tolist() == [99_999]
    # Four times the layers: about four times the time, sixteen if quadratic.
    assert times[1] < 10 * times[0]


def test_a_polygon_mesh_of_a_huge_size_is_refused_in_a_short_message(
    tmp_path: Path,
) -> None:
    pairs = _polyline(16, _grid(3, 3), head=((71, 1e300), (72, 3)))
    with pytest.raises(CodecError, match=r"1e\+300 x 3 polygon mesh") as info:
        read(_file(tmp_path, _doc(pairs)))
    assert len(str(info.value)) < 200


def test_a_mesh_of_no_vertices_reads_past_its_override_block(tmp_path: Path) -> None:
    # The override block's 92 counts properties, not vertices.
    pairs: list[Pair] = [(0, "MESH"), (8, "0"), (100, "AcDbSubDMesh"), (71, 2)]
    pairs += [(72, 0), (91, 0), (92, 0), (94, 0), (95, 0), (90, 1), (91, 0)]
    pairs += [(92, 1), (90, 0), (63, 1)]
    pairs += _point((1, 1, 1))
    mesh = read(_file(tmp_path, _doc(pairs)))
    assert len(mesh.element_types) == 1


def test_a_vertex_flag_past_int64_is_refused(tmp_path: Path) -> None:
    pairs = _polyline(0, [((0, 0, 0), 1e300, ()), ((1, 0, 0), 0, ())])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(CodecError, match="a VERTEX gives flags 1e\\+300"):
            read(_file(tmp_path, _doc(pairs)))


def test_a_two_vertex_poly_line_is_written_as_the_line_it_reads_back_as(
    tmp_path: Path,
) -> None:
    poly = make_polydata(_SQUARE, [("poly_line", np.array([[0, 2]]))])
    path = tmp_path / "out.dxf"
    write(poly, path)
    data = path.read_bytes()
    assert b"POLYLINE" not in data
    assert data.count(b"\nLINE\n") == 1
    back = read(path)
    assert back.element_types.tolist() == [_T["line"]]
    assert _coords(back) == [[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]


def test_a_bulge_before_the_first_lwpolyline_vertex_bends_nothing(
    tmp_path: Path,
) -> None:
    pairs = [(0, "LWPOLYLINE"), (8, "0"), (90, 2), (70, 0), (42, 0.5)]
    pairs += [(10, 0.0), (20, 0.0), (10, 1.0), (20, 0.0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = read(_file(tmp_path, _doc(pairs)))
    assert not [w for w in caught if "bulge" in str(w.message)]
    assert mesh.element_types.tolist() == [_T["line"]]


def test_write_warnings_point_at_the_caller(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"a/b": np.array([0])},
        global_attrs={"unit": "cubit"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polyxios.write(poly, tmp_path / "out.dxf")
    assert len(caught) == 2
    assert {w.filename for w in caught} == {__file__}
