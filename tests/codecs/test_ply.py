from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs._ply import read, write
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.fetcher import fetch


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


@pytest.mark.network
def test_3dgs_compressed_ply_positions() -> None:
    """Compressed 3DGS PLY returns real world coordinates, not zeros."""
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


@pytest.mark.network
def test_real_armadillo() -> None:
    """Armadillo.ply: binary big-endian with face scalar before vertex list."""
    path = fetch("Armadillo.ply")
    poly = read(path)
    assert len(poly.vertices) == 172974
    assert len(poly.element_types) == 345944
    assert poly.faces is not None and len(poly.faces) > 0
    assert "intensity" in poly.element_attrs


# --- meshio #1394: PLY edge elements -----------------------------------------


def _edge_mesh():
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    return make_polydata(
        verts,
        [
            ("line", np.array([[0, 1], [2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )


@pytest.mark.parametrize("binary", [False, True], ids=["ascii", "binary"])
def test_issue_1394_lines_are_written_as_ply_edges(tmp_path, binary: bool) -> None:
    """A line written as a degenerate face is not what a PLY reader expects."""
    poly = _edge_mesh()
    out = tmp_path / "edges.ply"
    write(poly, out, binary=binary)

    head = out.read_bytes().split(b"end_header")[0].decode("ascii")
    assert "element edge 2" in head
    assert "property int vertex1" in head
    assert "property int vertex2" in head
    assert "element face 1" in head

    back = read(out)
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["line"],
    ]
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2, 0, 1, 2, 3])


@pytest.mark.parametrize("binary", [False, True], ids=["ascii", "binary"])
def test_issue_1394_an_edge_only_mesh_round_trips(tmp_path, binary: bool) -> None:
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("line", np.array([[0, 1], [1, 2]]))])
    out = tmp_path / "lines.ply"
    write(poly, out, binary=binary)
    back = read(out)
    assert len(back.element_types) == 2
    np.testing.assert_array_equal(back.connectivity, [0, 1, 1, 2])
    np.testing.assert_allclose(back.vertices, verts)


def test_issue_1394_an_edge_element_is_read_from_a_hand_written_file(
    tmp_path,
) -> None:
    """Files in the wild spell the two ends vertex1/vertex2, not a face list."""
    path = tmp_path / "hand.ply"
    path.write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "element edge 2\nproperty int vertex1\nproperty int vertex2\n"
        "end_header\n"
        "0 0 0\n1 0 0\n2 0 0\n"
        "0 1\n1 2\n"
    )
    poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["line"]] * 2
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 1, 2])


def test_an_edge_element_naming_a_missing_vertex_is_refused(tmp_path) -> None:
    path = tmp_path / "bad.ply"
    path.write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
        "element edge 1\nproperty int vertex1\nproperty int vertex2\n"
        "end_header\n"
        "0 0 0\n1 0 0\n"
        "0 9\n"
    )
    with pytest.raises(CodecError, match="vertex"):
        read(path)


def _ascii_ply_with_edges(extra_face_prop: bool) -> str:
    prop = "property float quality\n" if extra_face_prop else ""
    values = " 0.25" if extra_face_prop else ""
    return (
        "ply\nformat ascii 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n" + prop + "element edge 1\n"
        "property int vertex1\nproperty int vertex2\n"
        "end_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"
        "3 0 1 2" + values + "\n"
        "0 2\n"
    )


def test_a_face_property_is_stretched_over_the_edges_that_follow(tmp_path) -> None:
    """An attribute shorter than the mesh is one every reader indexes off."""
    path = tmp_path / "edges.ply"
    path.write_text(_ascii_ply_with_edges(extra_face_prop=True))
    poly = read(path)
    quality = poly.element_attrs["quality"]
    assert quality.shape[0] == len(poly.element_types) == 2
    assert quality[0] == 0.25
    assert np.isnan(quality[1])


def test_a_short_edge_record_is_a_codec_error(tmp_path) -> None:
    """A truncated record must name the format, not raise a bare IndexError."""
    path = tmp_path / "short.ply"
    path.write_text(
        _ascii_ply_with_edges(extra_face_prop=False).replace("0 2\n", "0\n")
    )
    with pytest.raises(CodecError, match="edge record"):
        read(path)


def test_a_file_ending_inside_its_edges_is_a_codec_error(tmp_path) -> None:
    path = tmp_path / "cut.ply"
    path.write_text(_ascii_ply_with_edges(extra_face_prop=False).replace("0 2\n", ""))
    with pytest.raises(CodecError, match="ends inside"):
        read(path)


# --- the header's own order decides which records are whose -----------------


def _ascii_ply(elements: str, body: str) -> str:
    return "ply\nformat ascii 1.0\n" + elements + "end_header\n" + body


_VERTEX_BLOCK = (
    "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
)
_FACE_BLOCK = "element face 1\nproperty list uchar int vertex_indices\n"
_EDGE_BLOCK = "element edge 1\nproperty int vertex1\nproperty int vertex2\n"
_VERTEX_ROWS = "0 0 0\n1 0 0\n0 1 0\n1 1 0\n"


def test_an_edge_element_declared_before_the_faces_is_still_read_as_edges(
    tmp_path,
) -> None:
    """The blocks sit in the file in the order the header declares them."""
    path = tmp_path / "edge_first.ply"
    path.write_text(
        _ascii_ply(
            _VERTEX_BLOCK + _EDGE_BLOCK + _FACE_BLOCK,
            _VERTEX_ROWS + "2 3\n3 0 1 2\n",
        )
    )
    poly = read(path)
    # The face keeps its place ahead of the edge whatever the header said.
    assert poly.element_types.tolist() == [
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["line"],
    ]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 2, 3])


def test_an_element_this_codec_has_no_place_for_costs_only_its_own_records(
    tmp_path,
) -> None:
    """An unknown block read as another's records shifts the whole mesh."""
    path = tmp_path / "material.ply"
    path.write_text(
        _ascii_ply(
            _VERTEX_BLOCK
            + "element material 2\nproperty float red\nproperty float green\n"
            + _FACE_BLOCK,
            _VERTEX_ROWS + "0.5 0.5\n0.25 0.25\n3 0 1 2\n",
        )
    )
    poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_a_short_vertex_record_names_the_format(tmp_path) -> None:
    """A truncated record must not escape as a bare IndexError."""
    path = tmp_path / "short.ply"
    path.write_text(_ascii_ply(_VERTEX_BLOCK, "0 0 0\n1 0\n0 1 0\n1 1 0\n"))
    with pytest.raises(CodecError, match="vertex record"):
        read(path)


def test_a_face_naming_more_indices_than_it_carries_names_the_format(
    tmp_path,
) -> None:
    path = tmp_path / "cut.ply"
    path.write_text(_ascii_ply(_VERTEX_BLOCK + _FACE_BLOCK, _VERTEX_ROWS + "4 0 1 2\n"))
    with pytest.raises(CodecError, match="face record"):
        read(path)


def _binary_ply(header: str, body: bytes) -> bytes:
    """Return a little-endian binary PLY from its element declarations."""
    return (
        b"ply\nformat binary_little_endian 1.0\n"
        + header.encode()
        + b"end_header\n"
        + body
    )


def test_an_edge_carrying_a_list_property_still_names_its_two_ends(
    tmp_path,
) -> None:
    """A list property makes an edge record vary in width, not unreadable.

    The scalar path gathers a whole block with one structured read, which a
    list property has no fixed width for. Falling back to a record-by-record
    walk is what keeps such a file readable instead of raising a bare KeyError
    from a dtype that cannot describe the record.
    """
    import struct

    path = tmp_path / "edge_list.ply"
    path.write_bytes(
        _binary_ply(
            "element vertex 3\nproperty float x\nproperty float y\n"
            "property float z\n"
            "element edge 1\nproperty int vertex1\nproperty int vertex2\n"
            "property list uchar int tags\n",
            struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
            + struct.pack("<ii", 0, 1)
            + struct.pack("<BI", 1, 5),
        )
    )
    poly = read(path)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["line"]]
    np.testing.assert_array_equal(poly.connectivity, [0, 1])


def test_a_vertex_carrying_a_list_property_still_names_its_coordinates(
    tmp_path,
) -> None:
    """The same varying width on the element every PLY file carries."""
    import struct

    path = tmp_path / "vertex_list.ply"
    path.write_bytes(
        _binary_ply(
            "element vertex 2\nproperty float x\nproperty float y\n"
            "property float z\nproperty list uchar int extra\n"
            "element edge 1\nproperty int vertex1\nproperty int vertex2\n",
            struct.pack("<3f", 0, 0, 0)
            + struct.pack("<BII", 2, 7, 8)
            + struct.pack("<3f", 1, 0, 0)
            + struct.pack("<BI", 1, 9)
            + struct.pack("<ii", 0, 1),
        )
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1])


def test_a_multi_component_element_attribute_is_one_property_per_column(
    tmp_path,
) -> None:
    """An STL's facet colours are three numbers per element, not one.

    A header declaring one property where the record holds three describes a
    file no reader can walk, so the columns are spelled out the way a vertex
    attribute's already are.
    """
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.element_attrs["colors"] = np.array([[1.0, 0.0, 0.5]])
    path = tmp_path / "colors.ply"
    write(poly, path, binary=False)
    text = path.read_text()
    assert "property double colors_0" in text
    assert "property double colors_2" in text

    back = read(path)
    for column, value in enumerate((1.0, 0.0, 0.5)):
        np.testing.assert_allclose(back.element_attrs[f"colors_{column}"], [value])


@pytest.mark.parametrize("binary", [False, True])
def test_an_attribute_that_does_not_describe_the_mesh_is_reported(
    tmp_path, binary: bool
) -> None:
    """A short column has no record to sit in and must not index off the end."""
    poly = make_polydata(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.vertex_attrs["v"] = np.arange(2, dtype=np.float64)
    poly.element_attrs["two names"] = np.array([1.0])
    path = tmp_path / "bad_attrs.ply"
    with pytest.warns(UserWarning, match="fit in a record"):
        write(poly, path, binary=binary)

    back = read(path)
    assert "v" not in back.vertex_attrs
    assert "two" not in back.element_attrs
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2])


def _face_and_edge_mesh(values: np.ndarray) -> object:
    from polyxios._types import PolyData

    return PolyData(
        vertices=np.array(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
        ),
        connectivity=np.array([0, 1, 2, 2, 3], dtype=np.int32),
        offsets=np.array([0, 3, 5], dtype=np.int32),
        element_types=np.array(
            [ELEMENT_TYPES["triangle"], ELEMENT_TYPES["line"]], dtype=np.uint8
        ),
        element_attrs={"mat": values},
    )


@pytest.mark.parametrize("binary", [False, True], ids=["ascii", "binary"])
def test_an_element_property_reaches_the_edges_too(tmp_path, binary: bool) -> None:
    """A line's own value has nowhere to go if only the faces carry properties."""
    out = tmp_path / "both.ply"
    write(_face_and_edge_mesh(np.array([7, 8], dtype=np.int32)), out, binary=binary)

    head = out.read_bytes().split(b"end_header")[0].decode("ascii")
    assert head.count("property int mat") == 2

    back = read(out)
    np.testing.assert_array_equal(back.element_attrs["mat"], [7, 8])


def test_a_property_only_the_edges_declare_is_nan_over_the_faces(tmp_path) -> None:
    """The two elements need not declare the same properties."""
    path = tmp_path / "edge_only.ply"
    path.write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\n"
        "element edge 1\n"
        "property int vertex1\nproperty int vertex2\nproperty float weight\n"
        "end_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"
        "3 0 1 2\n"
        "0 2 0.5\n"
    )
    weight = read(path).element_attrs["weight"]
    assert np.isnan(weight[0])
    assert weight[1] == 0.5


def test_a_binary_face_block_of_one_width_reads_as_one_block(tmp_path) -> None:
    """Every record the same width is the case worth reading in one call."""
    import struct

    path = tmp_path / "uniform.ply"
    path.write_bytes(
        _binary_ply(
            "element vertex 4\nproperty float x\nproperty float y\n"
            "property float z\n"
            "element face 2\nproperty list uchar int vertex_indices\n"
            "property uchar quality\n",
            struct.pack("<12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0)
            + struct.pack("<B3iB", 3, 0, 1, 2, 7)
            + struct.pack("<B3iB", 3, 1, 3, 2, 9),
        )
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])
    np.testing.assert_array_equal(poly.element_attrs["quality"], [7, 9])


def test_a_binary_face_block_of_mixed_widths_falls_back_to_the_walk(tmp_path) -> None:
    """A triangle beside a quad is a block no single record width describes."""
    import struct

    path = tmp_path / "mixed.ply"
    path.write_bytes(
        _binary_ply(
            "element vertex 4\nproperty float x\nproperty float y\n"
            "property float z\n"
            "element face 2\nproperty list uchar int vertex_indices\n",
            struct.pack("<12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0)
            + struct.pack("<B3i", 3, 0, 1, 2)
            + struct.pack("<B4i", 4, 0, 1, 3, 2),
        )
    )
    poly = read(path)
    assert poly.element_types.tolist() == [
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["quad"],
    ]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 0, 1, 3, 2])


def test_a_file_ending_inside_its_faces_names_the_format(tmp_path) -> None:
    """A truncated binary face block must not escape as a bare struct error."""
    import struct

    path = tmp_path / "cut.ply"
    path.write_bytes(
        _binary_ply(
            "element vertex 3\nproperty float x\nproperty float y\n"
            "property float z\n"
            "element face 2\nproperty list uchar int vertex_indices\n",
            struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
            + struct.pack("<B3i", 3, 0, 1, 2)
            + struct.pack("<Bi", 3, 0),
        )
    )
    with pytest.raises(CodecError, match="face record"):
        read(path)
