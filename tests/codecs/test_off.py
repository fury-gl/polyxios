from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs._off import read, write
from polyxios.exceptions import CodecError


def _tetrahedron():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))],
    )


def _write_off(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "mesh.off"
    path.write_text(text, encoding="utf-8")
    return path


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "tetra.off"
    write(poly, tmp)
    poly2 = read(tmp)
    assert poly2.vertices.shape == poly.vertices.shape
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    assert len(poly2.element_types) == 4
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)


def test_roundtrip_quads(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])
    tmp = tmp_path / "quad.off"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    assert int(poly2.element_types[0]) == ELEMENT_TYPES["quad"]
    np.testing.assert_array_equal(poly2.connectivity, [0, 1, 2, 3])


def test_file_header(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "tetra.off"
    write(poly, tmp)
    text = tmp.read_text(encoding="utf-8")
    assert text.startswith("OFF\n")
    assert "4 4 0" in text


def test_polygon_face(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1.5, 0], [0, 1, 0]], dtype=np.float64
    )
    poly = make_polydata(verts, [("polygon", np.array([[0, 1, 2, 3, 4]]))])
    tmp = tmp_path / "poly.off"
    write(poly, tmp)
    poly2 = read(tmp)
    assert int(poly2.element_types[0]) == ELEMENT_TYPES["polygon"]


def test_empty_mesh_roundtrip(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    tmp = tmp_path / "empty.off"
    write(poly, tmp)
    poly2 = read(tmp)
    assert poly2.vertices.shape == (0, 3)
    assert len(poly2.element_types) == 0
    np.testing.assert_array_equal(poly2.offsets, [0])


def test_comments_and_blank_lines(tmp_path: Path) -> None:
    path = _write_off(
        tmp_path,
        "OFF\n# a comment\n\n3 1 0\n0 0 0\n1 0 0  # trailing comment\n0 1 0\n3 0 1 2\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_counts_on_header_line(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_bad_header_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "NOT_A_MESH\n4 4 0\n")
    with pytest.raises(CodecError, match="not an OFF file"):
        read(path)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "")
    with pytest.raises(CodecError, match="empty file"):
        read(path)


def test_lowercase_keyword_accepted(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "off\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_4d_variant_rejected(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "4OFF\n3 1 0\n0 0 0 0\n1 0 0 0\n0 1 0 0\n3 0 1 2\n")
    with pytest.raises(CodecError, match="4-D coordinates are not supported"):
        read(path)


def test_noff_with_dimension_three(tmp_path: Path) -> None:
    """`nOFF` defers the dimension to the header; three is readable."""
    path = _write_off(tmp_path, "nOFF\n3 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_noff_with_other_dimension_rejected(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "nOFF\n5 3 1 0\n")
    with pytest.raises(CodecError, match="5-D coordinates are not supported"):
        read(path)


def test_coff_reads_vertex_colors(tmp_path: Path) -> None:
    """Colour columns must not be read as the next vertex's coordinates."""
    path = _write_off(
        tmp_path,
        "COFF\n3 1 0\n"
        "0 0 0 255 0 0 255\n"
        "1 0 0 0 255 0 255\n"
        "0 1 0 0 0 255 255\n"
        "3 0 1 2\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    assert int(poly.element_types[0]) == ELEMENT_TYPES["triangle"]
    np.testing.assert_allclose(
        poly.vertex_attrs["colors"],
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
    )
    assert poly.global_attrs["off_vertex_color_format"] == "byte"


def test_float_vertex_colors_keep_their_scale(tmp_path: Path) -> None:
    """Integer channels are 0..255; float channels are already 0..1."""
    path = _write_off(
        tmp_path,
        "COFF\n3 1 0\n"
        "0 0 0 1.0 0.0 0.0\n"
        "1 0 0 0.0 1.0 0.0\n"
        "0 1 0 0.0 0.0 1.0\n"
        "3 0 1 2\n",
    )
    poly = read(path)
    np.testing.assert_allclose(
        poly.vertex_attrs["colors"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert poly.global_attrs["off_vertex_color_format"] == "float"


def test_bad_vertex_color_width_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "COFF\n1 0 0\n0 0 0 128 128\n")
    with pytest.raises(CodecError, match="3 or 4 components"):
        read(path)


def test_noff_reads_vertex_normals(tmp_path: Path) -> None:
    path = _write_off(
        tmp_path,
        "NOFF\n3 1 0\n0 0 0 0 0 1\n1 0 0 0 0 1\n0 1 0 0 0 1\n3 0 1 2\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(poly.vertex_attrs["normals"], np.tile([0, 0, 1], (3, 1)))
    assert "colors" not in poly.vertex_attrs


def test_stcnoff_all_channels_roundtrip(tmp_path: Path) -> None:
    path = _write_off(
        tmp_path,
        "STCNOFF\n3 1 0\n"
        "0 0 0  0 0 1  255 0 0 255  0 0\n"
        "1 0 0  0 0 1  0 255 0 255  1 0\n"
        "0 1 0  0 0 1  0 0 255 255  0 1\n"
        "3 0 1 2 255 255 0\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertex_attrs["texcoords"], [[0, 0], [1, 0], [0, 1]])
    np.testing.assert_allclose(poly.element_attrs["colors"], [[1, 1, 0]])

    out = tmp_path / "again.off"
    write(poly, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("STCNOFF\n")
    assert "255 0 0 255" in text

    poly2 = read(out)
    np.testing.assert_allclose(
        poly2.vertex_attrs["colors"], poly.vertex_attrs["colors"]
    )
    np.testing.assert_allclose(
        poly2.vertex_attrs["normals"], poly.vertex_attrs["normals"]
    )
    np.testing.assert_allclose(
        poly2.vertex_attrs["texcoords"], poly.vertex_attrs["texcoords"]
    )
    np.testing.assert_allclose(
        poly2.element_attrs["colors"], poly.element_attrs["colors"]
    )


def test_face_colormap_index(tmp_path: Path) -> None:
    """A single trailing number is a colormap index, not an RGB channel."""
    path = _write_off(
        tmp_path,
        "OFF\n4 2 0\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2 7\n3 1 3 2 9\n",
    )
    poly = read(path)
    np.testing.assert_array_equal(poly.element_attrs["color_index"], [7, 9])
    out = tmp_path / "again.off"
    write(poly, out)
    np.testing.assert_array_equal(read(out).element_attrs["color_index"], [7, 9])


def test_partial_face_colors_dropped(tmp_path: Path) -> None:
    """Colouring only some faces cannot be laid out as one array."""
    path = _write_off(
        tmp_path,
        "OFF\n4 2 0\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2 255 0 0\n3 1 3 2\n",
    )
    with pytest.warns(UserWarning, match="not uniform"):
        poly = read(path)
    assert "colors" not in poly.element_attrs
    assert len(poly.element_types) == 2


def test_face_color_is_ignored(tmp_path: Path) -> None:
    """A trailing per-face colour must not be mistaken for the next face."""
    path = _write_off(
        tmp_path,
        "OFF\n4 2 0\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2 255 0 0\n3 1 3 2 0 255 0\n",
    )
    poly = read(path)
    assert len(poly.element_types) == 2
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])


def test_negative_vertex_count_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n-5 1 0\n3 0 1 2\n")
    with pytest.raises(CodecError, match="negative vertex count"):
        read(path)


def test_absurd_vertex_count_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n999999999999 1 0\n")
    with pytest.raises(CodecError, match="safety cap"):
        read(path)


def test_negative_face_size_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n-3 0 1 2\n")
    with pytest.raises(CodecError, match="negative face size count"):
        read(path)


def test_degenerate_face_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n2 0 1\n")
    with pytest.raises(CodecError, match="at least 3"):
        read(path)


def test_out_of_range_index_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 99\n")
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_negative_index_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 -2\n")
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_short_face_line_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n4 0 1 2\n")
    with pytest.raises(CodecError, match="declares 4 vertices"):
        read(path)


def test_truncated_file_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n")
    with pytest.raises(CodecError, match="truncated"):
        read(path)


def test_malformed_vertex_line_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n2 0 0\n0 0 0\n1 0\n")
    with pytest.raises(CodecError, match="vertex line 2 needs 3 values"):
        read(path)


def test_non_numeric_vertex_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n2 0 0\n0 0 0\nx y z\n")
    with pytest.raises(CodecError, match="non-numeric vertex line"):
        read(path)


def test_missing_counts_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1\n")
    with pytest.raises(CodecError, match="missing vertex/face/edge counts"):
        read(path)


def test_non_integer_count_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\nthree 1 0\n0 0 0\n")
    with pytest.raises(CodecError, match="non-integer vertex count"):
        read(path)


def test_non_integer_face_index_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 two\n")
    with pytest.raises(CodecError, match="non-integer vertex index"):
        read(path)


def test_non_utf8_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "latin1.off"
    path.write_bytes(b"OFF\n1 0 0\n0 0 0 # \xe9\xe8\xea\n")
    with pytest.raises(CodecError, match="not valid UTF-8"):
        read(path)


def test_lazy_warns(tmp_path: Path) -> None:
    poly = _tetrahedron()
    tmp = tmp_path / "tetra.off"
    write(poly, tmp)
    with pytest.warns(UserWarning, match="lazy=True ignored"):
        read(tmp, lazy=True)


def test_unknown_write_option_warns(tmp_path: Path) -> None:
    poly = _tetrahedron()
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(poly, tmp_path / "tetra.off", bogus=1)


def test_float_fmt_option(tmp_path: Path) -> None:
    verts = np.array([[1 / 3, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    tmp = tmp_path / "precise.off"
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        write(poly, tmp, float_fmt=".17g")
    np.testing.assert_array_equal(read(tmp).vertices, verts)


def test_write_rejects_out_of_range_index(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 7], dtype=np.int32),
        offsets=poly.offsets,
        element_types=poly.element_types,
    )
    with pytest.raises(CodecError, match="outside"):
        write(broken, tmp_path / "broken.off")


def test_non_surface_elements_skipped(tmp_path: Path) -> None:
    """A tetra is not a face; writing it as one would round-trip as a quad."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    tmp = tmp_path / "tetra.off"
    with pytest.warns(UserWarning, match="not polygonal faces|not polygonal"):
        write(poly, tmp)
    assert tmp.read_text(encoding="utf-8").splitlines()[1] == "4 0 0"
    poly2 = read(tmp)
    assert len(poly2.element_types) == 0
    assert poly2.vertices.shape == (4, 3)


def test_mixed_mesh_keeps_faces_and_vertex_indices(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1]])), ("triangle", np.array([[1, 2, 3]]))],
    )
    tmp = tmp_path / "mixed.off"
    with pytest.warns(UserWarning):
        write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_array_equal(poly2.connectivity, [1, 2, 3])
    np.testing.assert_allclose(poly2.vertices, verts)


def test_pixel_reordered_to_polygon_walk(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("pixel", np.array([[0, 1, 2, 3]]))])
    tmp = tmp_path / "pixel.off"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.connectivity, [0, 1, 3, 2])


def test_registry_resolves_off(tmp_path: Path) -> None:
    import polyxios

    poly = _tetrahedron()
    tmp = tmp_path / "tetra.off"
    polyxios.write(poly, tmp)
    poly2 = polyxios.read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices)


# --------------------------------------------------------------------------
# binary OFF
# --------------------------------------------------------------------------


def _be(values, dtype: str) -> bytes:
    return np.array(values, dtype=np.dtype(dtype)).tobytes()


def test_binary_read_hand_built(tmp_path: Path) -> None:
    """Big-endian 32-bit ints and floats, face colour preceded by its count."""
    path = tmp_path / "hand.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([0], ">i4")
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    assert int(poly.element_types[0]) == ELEMENT_TYPES["triangle"]


def test_binary_read_face_color(tmp_path: Path) -> None:
    path = tmp_path / "colored.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([4], ">i4")
        + _be([1.0, 0.0, 0.0, 1.0], ">f4")
    )
    poly = read(path)
    np.testing.assert_allclose(poly.element_attrs["colors"], [[1, 0, 0, 1]])


def test_binary_vertex_channels(tmp_path: Path) -> None:
    """Binary vertex colour is always four floats, after the normal."""
    path = tmp_path / "cn.off"
    path.write_bytes(
        b"CNOFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be(
            [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
            ],
            ">f4",
        )
        + _be([3, 0, 1, 2], ">i4")
        + _be([0], ">i4")
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(poly.vertex_attrs["normals"], np.tile([0, 0, 1], (3, 1)))
    np.testing.assert_allclose(
        poly.vertex_attrs["colors"], [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    )


def test_binary_roundtrip(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        vertex_attrs={"normals": np.tile([0.0, 0.0, 1.0], (4, 1))},
        element_attrs={"colors": np.array([[1.0, 0, 0, 1.0], [0, 1.0, 0, 1.0]])},
    )
    out = tmp_path / "bin.off"
    write(poly, out, binary=True)
    assert out.read_bytes().startswith(b"CNOFF BINARY\n") is False
    assert out.read_bytes().startswith(b"NOFF BINARY\n")

    poly2 = read(out)
    np.testing.assert_allclose(poly2.vertices, verts)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_allclose(
        poly2.vertex_attrs["normals"], poly.vertex_attrs["normals"]
    )
    np.testing.assert_allclose(
        poly2.element_attrs["colors"], poly.element_attrs["colors"]
    )


def test_binary_three_channel_color_gains_alpha(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"colors": np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])},
    )
    out = tmp_path / "bin.off"
    write(poly, out, binary=True)
    poly2 = read(out)
    np.testing.assert_allclose(
        poly2.vertex_attrs["colors"],
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
    )


def test_binary_empty_mesh_roundtrip(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3), dtype=np.float64), [])
    out = tmp_path / "empty.off"
    write(poly, out, binary=True)
    poly2 = read(out)
    assert poly2.vertices.shape == (0, 3)
    assert len(poly2.element_types) == 0


def test_binary_truncated_raises(tmp_path: Path) -> None:
    path = tmp_path / "short.off"
    path.write_bytes(b"OFF BINARY\n" + _be([3, 1, 0], ">i4") + _be([0, 0, 0], ">f4"))
    with pytest.raises(CodecError, match="binary data truncated"):
        read(path)


def test_binary_bad_color_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([2], ">i4")
        + _be([0.0, 0.0], ">f4")
    )
    with pytest.raises(CodecError, match="colour components"):
        read(path)


def test_binary_out_of_range_index_raises(tmp_path: Path) -> None:
    path = tmp_path / "oob.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 42], ">i4")
        + _be([0], ">i4")
    )
    with pytest.raises(CodecError, match="outside"):
        read(path)


def test_binary_absurd_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "absurd.off"
    path.write_bytes(b"OFF BINARY\n" + _be([-4, 1, 0], ">i4"))
    with pytest.raises(CodecError, match="negative vertex count"):
        read(path)


# --------------------------------------------------------------------------
# writer robustness
# --------------------------------------------------------------------------


def test_bad_float_fmt_raises(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="float_fmt"):
        write(_tetrahedron(), tmp_path / "bad.off", float_fmt="qq")


def test_misshaped_vertex_attr_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": np.zeros((2, 3))},
    )
    out = tmp_path / "bad.off"
    with pytest.warns(UserWarning, match="skipping it"):
        write(poly, out)
    assert out.read_text(encoding="utf-8").startswith("OFF\n")


def test_misshaped_face_color_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"colors": np.zeros((1, 2))},
    )
    out = tmp_path / "bad.off"
    with pytest.warns(UserWarning, match="skipping it"):
        write(poly, out)
    assert "colors" not in read(out).element_attrs


def test_colors_win_over_color_index(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={
            "colors": np.array([[1.0, 0.0, 0.0]]),
            "color_index": np.array([4], dtype=np.int32),
        },
    )
    out = tmp_path / "both.off"
    with pytest.warns(UserWarning, match="both 'colors' and 'color_index'"):
        write(poly, out)
    poly2 = read(out)
    np.testing.assert_allclose(poly2.element_attrs["colors"], [[1, 0, 0]])
    assert "color_index" not in poly2.element_attrs


def test_short_pixel_is_skipped_not_crashed(tmp_path: Path) -> None:
    """Reordering a pixel walks its fourth node; a short one must not crash."""
    poly = make_polydata(np.zeros((3, 3), dtype=np.float64), [])
    broken = type(poly)(
        vertices=poly.vertices,
        connectivity=np.array([0, 1, 2], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["pixel"]], dtype=np.uint8),
    )
    out = tmp_path / "pixel.off"
    with pytest.warns(UserWarning, match="not polygonal faces"):
        write(broken, out)
    assert len(read(out).element_types) == 0


def test_face_colors_follow_writable_subset(tmp_path: Path) -> None:
    """Skipping a non-face element must not shift the remaining colours."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("line", np.array([[0, 1]])), ("triangle", np.array([[1, 2, 3]]))],
        element_attrs={"colors": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
    )
    out = tmp_path / "mixed.off"
    with pytest.warns(UserWarning, match="not polygonal faces"):
        write(poly, out)
    poly2 = read(out)
    np.testing.assert_allclose(poly2.element_attrs["colors"], [[1, 0, 0]])


def test_float_scale_colors_survive_ascii_roundtrip(tmp_path: Path) -> None:
    """A 0..1 channel that lands on a whole number must not read back as 0..255."""
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"colors": np.array([[1.0, 0.0, 0.0]] * 3)},
    )
    out = tmp_path / "float.off"
    write(poly, out)
    poly2 = read(out)
    np.testing.assert_allclose(poly2.vertex_attrs["colors"], [[1, 0, 0]] * 3)
    assert poly2.global_attrs["off_vertex_color_format"] == "float"


def test_binary_color_index_roundtrip(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"color_index": np.array([12], dtype=np.int32)},
    )
    out = tmp_path / "idx.off"
    write(poly, out, binary=True)
    np.testing.assert_array_equal(read(out).element_attrs["color_index"], [12])


def test_binary_header_after_comment(tmp_path: Path) -> None:
    """Binary data starts after the header line, not after the file's first."""
    path = tmp_path / "commented.off"
    path.write_bytes(
        b"# written by a tool\nOFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([0], ">i4")
    )
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_binary_crlf_header(tmp_path: Path) -> None:
    path = tmp_path / "crlf.off"
    path.write_bytes(
        b"OFF BINARY\r\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([0], ">i4")
    )
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_binary_oversized_count_does_not_overflow(tmp_path: Path) -> None:
    """A count whose byte size overflows int32 must still trip the guard."""
    path = tmp_path / "huge.off"
    path.write_bytes(b"OFF BINARY\n" + _be([0x14000000, 1, 0], ">i4"))
    with pytest.raises(CodecError, match="binary data truncated"):
        read(path)


def test_binary_negative_face_size_raises(tmp_path: Path) -> None:
    path = tmp_path / "negface.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([-3, 0, 1, 2], ">i4")
    )
    with pytest.raises(CodecError, match="at least 3"):
        read(path)


def test_non_numeric_face_colors_dropped(tmp_path: Path) -> None:
    path = _write_off(
        tmp_path, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2 red red red\n"
    )
    with pytest.warns(UserWarning, match="not numeric"):
        poly = read(path)
    assert "colors" not in poly.element_attrs


def test_binary_noff_dimension_header(tmp_path: Path) -> None:
    """`nOFF BINARY` puts the dimension in front of the counts."""
    path = tmp_path / "ndim.off"
    path.write_bytes(
        b"nOFF BINARY\n"
        + _be([3], ">i4")
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([0], ">i4")
    )
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_binary_absurd_face_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "cap.off"
    path.write_bytes(b"OFF BINARY\n" + _be([1, 2_000_000_001, 0], ">i4"))
    with pytest.raises(CodecError, match="safety cap"):
        read(path)


def test_binary_texcoords_roundtrip(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"texcoords": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])},
    )
    out = tmp_path / "st.off"
    write(poly, out, binary=True)
    assert out.read_bytes().startswith(b"STOFF BINARY\n")
    np.testing.assert_allclose(
        read(out).vertex_attrs["texcoords"], poly.vertex_attrs["texcoords"]
    )


def test_misshaped_color_index_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.zeros((3, 3), dtype=np.float64),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"color_index": np.zeros((1, 2), dtype=np.int32)},
    )
    out = tmp_path / "bad.off"
    with pytest.warns(UserWarning, match="skipping it"):
        write(poly, out)
    assert "color_index" not in read(out).element_attrs


def test_binary_truncated_at_face_start(tmp_path: Path) -> None:
    path = tmp_path / "cut.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
    )
    with pytest.raises(CodecError, match="truncated at face 0"):
        read(path)


def test_binary_truncated_mid_face(tmp_path: Path) -> None:
    path = tmp_path / "cut.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0], ">i4")
    )
    with pytest.raises(CodecError, match="truncated in face 0"):
        read(path)


def test_binary_truncated_in_face_color(tmp_path: Path) -> None:
    path = tmp_path / "cut.off"
    path.write_bytes(
        b"OFF BINARY\n"
        + _be([3, 1, 0], ">i4")
        + _be([0, 0, 0, 1, 0, 0, 0, 1, 0], ">f4")
        + _be([3, 0, 1, 2], ">i4")
        + _be([4], ">i4")
        + _be([1.0, 0.0], ">f4")
    )
    with pytest.raises(CodecError, match="truncated in face 0 colour"):
        read(path)


def test_byte_order_mark_is_ignored(tmp_path: Path) -> None:
    """A BOM must not glue itself to the header keyword."""
    path = tmp_path / "bom.off"
    path.write_bytes(b"\xef\xbb\xbf" + b"OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_crlf_ascii_file(tmp_path: Path) -> None:
    path = tmp_path / "crlf.off"
    path.write_bytes(b"OFF\r\n3 1 0\r\n0 0 0\r\n1 0 0\r\n0 1 0\r\n3 0 1 2\r\n")
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_tab_separated_file(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n3\t1\t0\n0\t0\t0\n1 0 0\n0 1 0\n3\t0\t1\t2\n")
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_ascii_write_is_idempotent(tmp_path: Path) -> None:
    """Re-writing what was read must reproduce the same bytes."""
    src = _write_off(
        tmp_path,
        "COFF\n4 2 0\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n1 1 0 255 255 0\n"
        "3 0 1 2 10 20 30\n3 1 3 2 40 50 60\n",
    )
    first = tmp_path / "first.off"
    write(read(src), first)
    second = tmp_path / "second.off"
    write(read(first), second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == src.read_bytes()


def test_binary_write_is_idempotent(tmp_path: Path) -> None:
    src = _write_off(
        tmp_path,
        "COFF\n3 1 0\n0 0 0 1.0 0.0 0.0\n1 0 0 0.0 1.0 0.0\n0 1 0 0.0 0.0 1.0\n"
        "3 0 1 2\n",
    )
    first = tmp_path / "first.off"
    write(read(src), first, binary=True)
    second = tmp_path / "second.off"
    write(read(first), second, binary=True)
    assert first.read_bytes() == second.read_bytes()


def test_counts_beyond_file_length_raise_before_allocating(tmp_path: Path) -> None:
    """A header claiming more records than the file has lines is refused."""
    path = _write_off(tmp_path, "OFF 100 100 0\n0 0 0\n")
    with pytest.raises(CodecError, match="file holds at most"):
        read(path)


def test_missing_vertex_block_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "COFF\n1 0 0\n")
    with pytest.raises(CodecError, match="truncated in the vertex block"):
        read(path)


def test_truncated_face_block_raises(tmp_path: Path) -> None:
    path = _write_off(tmp_path, "OFF\n1 2 0\n0 0 0\n3 0 0 0\n")
    with pytest.raises(CodecError, match="expected 2 face lines, got 1"):
        read(path)
