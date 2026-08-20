from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._vtk import read, write
from polyxios.exceptions import CodecError, LazyReadError


def _synthetic_mesh() -> object:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def test_roundtrip_ascii() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == 2
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_binary() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_lazy() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly_lazy = read(tmp, lazy=True)
    # Force access to load pages
    np.testing.assert_allclose(poly_lazy.vertices, poly.vertices, atol=1e-8)
    np.testing.assert_array_equal(poly_lazy.connectivity, poly.connectivity)


def test_vertex_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pressure = np.array([1.0, 2.0, 3.0, 4.0])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        vertex_attrs={"pressure": pressure},
    )
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "pressure" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["pressure"], pressure, atol=1e-6)


def test_element_attrs() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    stress = np.array([10.0, 20.0])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"stress": stress},
    )
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    poly2 = read(tmp)
    assert "stress" in poly2.element_attrs
    np.testing.assert_allclose(poly2.element_attrs["stress"], stress, atol=1e-6)


def test_vtk_version_42_has_cells_keyword() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp, vtk_version="4.2")
    assert "CELLS" in Path(tmp).read_text()
    assert "OFFSETS" not in Path(tmp).read_text()


def test_ascii_lazy_raises() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def _write_tmp(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        f.write(content)
        return f.name


def test_v1_blank_line_before_binary_marker() -> None:
    """VTK v1.0 files can have a blank line between the title and BINARY/ASCII."""
    # Minimal ASCII UNSTRUCTURED_GRID with v1.0 blank-line quirk.
    content = (
        b"# vtk DataFile Version 1.0\n"
        b"Test mesh\n"
        b"\n"  # blank line before ASCII/BINARY marker
        b"ASCII\n"
        b"\n"  # blank line before DATASET
        b"DATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n"
        b"0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n"
        b"3 0 1 2\n"
        b"CELL_TYPES 1\n"
        b"5\n"
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 3
    assert len(poly.element_types) == 1


def test_v1_blank_line_unsupported_dataset_gives_clear_error() -> None:
    """v1.0 blank-line quirk: CodecError names the dataset type, not 'BINARY'."""
    content = (
        b"# vtk DataFile Version 1.0\n"
        b"Some grid\n"
        b"\n"
        b"BINARY\n"
        b"\n"
        b"DATASET CUSTOM_GRID\n"
        b"DIMENSIONS 2 2 2\n"
    )
    tmp = _write_tmp(content)
    with pytest.raises(CodecError, match="CUSTOM_GRID"):
        read(tmp)


def _make_binary_polydata_lines() -> bytes:
    """Build a minimal binary VTK POLYDATA file with a LINES section."""

    header = (
        b"# vtk DataFile Version 3.0\ntest polydata binary\nBINARY\nDATASET POLYDATA\n"
    )
    # 4 points
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=">f4").tobytes()
    points_hdr = b"POINTS 4 float\n"

    # 1 LINES cell with 4 points: [count=4, 0, 1, 2, 3] → total_vals = 5
    cell_data = np.array([4, 0, 1, 2, 3], dtype=">i4").tobytes()
    lines_hdr = b"LINES 1 5\n"

    return header + points_hdr + pts + lines_hdr + cell_data


def test_binary_polydata_lines() -> None:
    """Binary POLYDATA with LINES section reads correctly."""
    tmp = _write_tmp(_make_binary_polydata_lines())
    poly = read(tmp)
    assert len(poly.vertices) == 4
    assert len(poly.element_types) == 1
    # poly_line (cnt=4 > 2)
    from polyxios._element_types import ELEMENT_TYPES

    assert int(poly.element_types[0]) == ELEMENT_TYPES["poly_line"]
    np.testing.assert_allclose(poly.vertices[0], [0, 0, 0])
    np.testing.assert_allclose(poly.vertices[3], [3, 0, 0])


def test_binary_polydata_polygons() -> None:
    """Binary POLYDATA with POLYGONS: triangles and quads map to correct element types."""

    header = b"# vtk DataFile Version 3.0\ntest polygons\nBINARY\nDATASET POLYDATA\n"
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]], dtype=">f4"
    ).tobytes()
    points_hdr = b"POINTS 5 float\n"

    # 2 cells: triangle [3,0,1,2] + quad [4,0,1,3,2] → total_vals = 4+5 = 9
    cell_data = np.array([3, 0, 1, 2, 4, 0, 1, 3, 2], dtype=">i4").tobytes()
    polys_hdr = b"POLYGONS 2 9\n"

    content = header + points_hdr + pts + polys_hdr + cell_data
    tmp = _write_tmp(content)
    poly = read(tmp)

    from polyxios._element_types import ELEMENT_TYPES

    assert len(poly.element_types) == 2
    assert int(poly.element_types[0]) == ELEMENT_TYPES["triangle"]
    assert int(poly.element_types[1]) == ELEMENT_TYPES["quad"]


def test_binary_polydata_lazy_raises() -> None:
    """Binary POLYDATA does not support lazy reads."""
    tmp = _write_tmp(_make_binary_polydata_lines())
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_rectilinear_grid_ascii() -> None:
    """RECTILINEAR_GRID ASCII builds correct meshgrid vertices."""
    content = (
        b"# vtk DataFile Version 2.0\n"
        b"test\n"
        b"ASCII\n"
        b"DATASET RECTILINEAR_GRID\n"
        b"DIMENSIONS 3 2 1\n"
        b"X_COORDINATES 3 float\n"
        b"0.0 1.0 3.0\n"
        b"Y_COORDINATES 2 float\n"
        b"0.0 2.0\n"
        b"Z_COORDINATES 1 float\n"
        b"0.0\n"
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 6  # 3*2*1
    assert len(poly.element_types) == 2  # (3-1)*(2-1) = 2 quads
    from polyxios._element_types import ELEMENT_TYPES

    assert all(t == ELEMENT_TYPES["quad"] for t in poly.element_types)
    # vertex ordering: ij meshgrid → x varies first
    np.testing.assert_allclose(poly.vertices[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(poly.vertices[1], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(poly.vertices[2], [1.0, 0.0, 0.0])


@pytest.mark.parametrize("fname,expected", [("RectGrid2.vtk", (17061, 14720))])
def test_rectilinear_grid_real_files(fname: str, expected: tuple) -> None:
    """Real RECTILINEAR_GRID corpus file reads with correct counts."""
    import os

    path = os.path.expanduser(f"~/.polyxios/vtk/{fname}")
    if not os.path.exists(path):
        pytest.skip(f"{fname} not in local cache")
    poly = read(path)
    assert len(poly.vertices) == expected[0]
    assert len(poly.element_types) == expected[1]


def test_structured_grid_ascii() -> None:
    """STRUCTURED_GRID ASCII with explicit curvilinear points."""
    content = (
        b"# vtk DataFile Version 3.0\n"
        b"test\n"
        b"ASCII\n"
        b"DATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 1\n"
        b"POINTS 4 float\n"
        b"0 0 0\n1.5 0 0\n0 2.5 0\n1.5 2.5 0\n"
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 4
    assert len(poly.element_types) == 1  # 1 quad
    from polyxios._element_types import ELEMENT_TYPES

    assert int(poly.element_types[0]) == ELEMENT_TYPES["quad"]
    np.testing.assert_allclose(poly.vertices[1], [1.5, 0.0, 0.0])


def test_structured_grid_binary() -> None:
    """STRUCTURED_GRID binary reads correct vertex coordinates."""
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=">f4",
    ).tobytes()
    content = (
        b"# vtk DataFile Version 3.0\n"
        b"test\n"
        b"BINARY\n"
        b"DATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 2\n"
        b"POINTS 8 float\n" + pts
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 8
    assert len(poly.element_types) == 1  # 1 hex
    from polyxios._element_types import ELEMENT_TYPES

    assert int(poly.element_types[0]) == ELEMENT_TYPES["hexahedron"]


@pytest.mark.parametrize(
    "fname,expected_verts",
    [("SampleStructGrid.vtk", 24000), ("office.binary.vtk", 8400)],
)
def test_structured_grid_real_files(fname: str, expected_verts: int) -> None:
    """Real STRUCTURED_GRID corpus files read with correct vertex count."""
    import os

    path = os.path.expanduser(f"~/.polyxios/vtk/{fname}")
    if not os.path.exists(path):
        pytest.skip(f"{fname} not in local cache")
    poly = read(path)
    assert len(poly.vertices) == expected_verts


def test_structured_points_ascii_2d() -> None:
    """STRUCTURED_POINTS ASCII 2D generates quad connectivity."""
    content = (
        b"# vtk DataFile Version 2.0\n"
        b"test\n"
        b"ASCII\n"
        b"DATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 3 3 1\n"
        b"ORIGIN 0 0 0\n"
        b"SPACING 1 1 1\n"
        b"POINT_DATA 9\n"
        b"SCALARS values float\n"
        b"LOOKUP_TABLE default\n"
        b"0 1 2 3 4 5 6 7 8\n"
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 9
    assert len(poly.element_types) == 4  # (3-1)*(3-1) = 4 quads
    from polyxios._element_types import ELEMENT_TYPES

    assert all(t == ELEMENT_TYPES["quad"] for t in poly.element_types)
    assert "values" in poly.vertex_attrs
    np.testing.assert_allclose(poly.vertex_attrs["values"], np.arange(9, dtype=float))


def test_structured_points_ascii_3d() -> None:
    """STRUCTURED_POINTS ASCII 3D generates hexahedron connectivity."""
    content = (
        b"# vtk DataFile Version 2.0\n"
        b"test 3d\n"
        b"ASCII\n"
        b"DATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 2 2\n"
        b"ORIGIN 0 0 0\n"
        b"SPACING 1 1 1\n"
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 8
    assert len(poly.element_types) == 1  # 1 hex
    from polyxios._element_types import ELEMENT_TYPES

    assert int(poly.element_types[0]) == ELEMENT_TYPES["hexahedron"]


def test_structured_points_aspect_ratio_keyword() -> None:
    """VTK v1.0 ASPECT_RATIO keyword is treated the same as SPACING."""
    content = (
        b"# vtk DataFile Version 1.0\n"
        b"v1 grid\n"
        b"ASCII\n"
        b"DATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 2 1\n"
        b"ORIGIN 0 0 0\n"
        b"ASPECT_RATIO 2 3 1\n"
    )
    tmp = _write_tmp(content)
    poly = read(tmp)
    assert len(poly.vertices) == 4
    # ij indexing: (i=0,j=1) → vertex[1]; (i=1,j=0) → vertex[2]
    np.testing.assert_allclose(poly.vertices[1], [0.0, 3.0, 0.0])
    np.testing.assert_allclose(poly.vertices[2], [2.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "fname,min_verts",
    [
        ("heart.vtk", 12000),
        ("matrix.vtk", 50),
        ("texThres2.vtk", 100),
    ],
)
def test_structured_points_real_files(fname: str, min_verts: int) -> None:
    """Real STRUCTURED_POINTS files from the test corpus read without error."""
    import os

    path = os.path.expanduser(f"~/.polyxios/vtk/{fname}")
    if not os.path.exists(path):
        pytest.skip(f"{fname} not in local cache")
    poly = read(path)
    assert len(poly.vertices) >= min_verts


@pytest.mark.parametrize("fname", ["faults.vtk", "track1.binary.vtk"])
def test_binary_polydata_real_files(fname: str) -> None:
    """Real binary POLYDATA files from the test corpus read without error."""
    import os

    path = os.path.expanduser(f"~/.polyxios/vtk/{fname}")
    if not os.path.exists(path):
        pytest.skip(f"{fname} not in local cache")
    poly = read(path)
    assert len(poly.vertices) > 0
    assert len(poly.element_types) > 0


# ---------------------------------------------------------------------------
# P1.3 - legacy binary read failures
# ---------------------------------------------------------------------------


def _binary_grid(
    *,
    newline: bytes = b"\n",
    point_dtype: str = ">f8",
    point_type: bytes = b"double",
    trailer: bytes = b"",
    extra: bytes = b"",
) -> bytes:
    """A one-triangle binary UNSTRUCTURED_GRID, with the quirks dialled in."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=point_dtype).tobytes()
    cells = np.array([3, 0, 1, 2], dtype=">i4").tobytes()
    types = np.array([5], dtype=">i4").tobytes()
    header = newline.join(
        [
            b"# vtk DataFile Version 4.2",
            b"binary quirks",
            b"BINARY",
            b"DATASET UNSTRUCTURED_GRID",
            b"POINTS 3 " + point_type + trailer,
            b"",
        ]
    )
    return (
        header
        + pts
        + b"\nCELLS 1 4\n"
        + cells
        + b"\nCELL_TYPES 1\n"
        + types
        + b"\n"
        + extra
    )


def test_binary_grid_with_crlf_header_reads() -> None:
    """A file written on Windows carries CRLF in the ASCII header."""
    tmp = _write_tmp(_binary_grid(newline=b"\r\n"))

    poly = read(tmp)

    assert len(poly.vertices) == 3
    assert len(poly.element_types) == 1


def test_trailing_space_after_the_points_keyword_reads() -> None:
    """Some writers pad the keyword line; the data still starts after it."""
    tmp = _write_tmp(_binary_grid(trailer=b"  "))

    poly = read(tmp)

    np.testing.assert_allclose(poly.vertices[2], [0, 1, 0])


def test_binary_float32_points_read_as_written() -> None:
    tmp = _write_tmp(_binary_grid(point_dtype=">f4", point_type=b"float"))

    poly = read(tmp)

    np.testing.assert_allclose(poly.vertices[1], [1, 0, 0])


def test_color_scalars_are_read_not_refused() -> None:
    """COLOR_SCALARS is unsigned char per component, not a double array."""
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    extra = b"POINT_DATA 3\nCOLOR_SCALARS rgb 3\n" + colors.tobytes() + b"\n"
    tmp = _write_tmp(_binary_grid(extra=extra))

    poly = read(tmp)

    assert "rgb" in poly.vertex_attrs
    assert poly.vertex_attrs["rgb"].shape == (3, 3)


def test_ascii_color_scalars_and_normals_are_read() -> None:
    """The ASCII flavour holds floats in 0..1, and NORMALS is a vector."""
    content = (
        b"# vtk DataFile Version 4.2\n"
        b"ascii attributes\n"
        b"ASCII\n"
        b"DATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n"
        b"0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n"
        b"3 0 1 2\n"
        b"CELL_TYPES 1\n"
        b"5\n"
        b"POINT_DATA 3\n"
        b"COLOR_SCALARS rgb 3\n"
        b"1 0 0\n0 1 0\n0 0 1\n"
        b"NORMALS n float\n"
        b"0 0 1\n0 0 1\n0 0 1\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_allclose(poly.vertex_attrs["rgb"][2], [0, 0, 1])
    np.testing.assert_allclose(poly.vertex_attrs["n"][0], [0, 0, 1])


def test_binary_normals_are_read() -> None:
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=">f4")
    extra = b"POINT_DATA 3\nNORMALS n float\n" + normals.tobytes() + b"\n"
    tmp = _write_tmp(_binary_grid(extra=extra))

    poly = read(tmp)

    np.testing.assert_allclose(poly.vertex_attrs["n"][1], [0, 0, 1])


# ---------------------------------------------------------------------------
# P1.4 - POLYDATA sections and the legacy structured datasets
# ---------------------------------------------------------------------------


def test_polydata_reads_all_four_cell_sections() -> None:
    """VERTICES, LINES, POLYGONS and TRIANGLE_STRIPS in one file."""
    content = (
        b"# vtk DataFile Version 4.2\n"
        b"every section\n"
        b"ASCII\n"
        b"DATASET POLYDATA\n"
        b"POINTS 5 float\n"
        b"0 0 0\n1 0 0\n0 1 0\n1 1 0\n2 0 0\n"
        b"VERTICES 1 2\n"
        b"1 4\n"
        b"LINES 1 3\n"
        b"2 0 1\n"
        b"POLYGONS 1 4\n"
        b"3 0 1 2\n"
        b"TRIANGLE_STRIPS 1 5\n"
        b"4 0 1 2 3\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    from polyxios._element_types import ELEMENT_TYPES

    kinds = [int(t) for t in poly.element_types]
    assert ELEMENT_TYPES["vertex"] in kinds
    assert ELEMENT_TYPES["line"] in kinds
    assert ELEMENT_TYPES["triangle"] in kinds
    assert ELEMENT_TYPES["triangle_strip"] in kinds


def test_structured_points_keeps_origin_spacing_dimensions() -> None:
    """The grid the file describes is lost the moment the points are built."""
    content = (
        b"# vtk DataFile Version 4.2\n"
        b"image\n"
        b"ASCII\n"
        b"DATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 3 1\n"
        b"ORIGIN 1 2 3\n"
        b"SPACING 0.5 0.25 1\n"
        b"POINT_DATA 6\n"
        b"SCALARS s float 1\n"
        b"LOOKUP_TABLE default\n"
        b"0 1 2 3 4 5\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert list(poly.global_attrs["vtk_dimensions"]) == [2, 3, 1]
    np.testing.assert_allclose(poly.global_attrs["vtk_origin"], [1, 2, 3])
    np.testing.assert_allclose(poly.global_attrs["vtk_spacing"], [0.5, 0.25, 1])


def test_legacy_structured_grid_keeps_its_dimensions() -> None:
    content = (
        b"# vtk DataFile Version 4.2\n"
        b"curvilinear\n"
        b"ASCII\n"
        b"DATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 1\n"
        b"POINTS 4 float\n"
        b"0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert list(poly.global_attrs["vtk_dimensions"]) == [2, 2, 1]


def test_legacy_rectilinear_grid_keeps_its_dimensions() -> None:
    content = (
        b"# vtk DataFile Version 4.2\n"
        b"rect\n"
        b"ASCII\n"
        b"DATASET RECTILINEAR_GRID\n"
        b"DIMENSIONS 2 2 1\n"
        b"X_COORDINATES 2 float\n"
        b"0 1\n"
        b"Y_COORDINATES 2 float\n"
        b"0 1\n"
        b"Z_COORDINATES 1 float\n"
        b"0\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert list(poly.global_attrs["vtk_dimensions"]) == [2, 2, 1]


# ---------------------------------------------------------------------------
# Attribute sections that run out, and the one whose scale differs by flavour
# ---------------------------------------------------------------------------


def _ascii_grid(attrs: bytes) -> bytes:
    return (
        b"# vtk DataFile Version 4.2\n"
        b"attrs\n"
        b"ASCII\n"
        b"DATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n"
        b"0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n"
        b"3 0 1 2\n"
        b"CELL_TYPES 1\n"
        b"5\n"
        b"POINT_DATA 3\n" + attrs
    )


@pytest.mark.parametrize(
    "attrs",
    [
        b"COLOR_SCALARS rgb 3\n1 0 0\n",
        b"SCALARS s float 1\nLOOKUP_TABLE default\n1 2\n",
        b"NORMALS n float\n0 0 1\n",
        b"VECTORS v float\n0 0 1\n",
        b"TENSORS t float\n1 0 0 0 1 0 0 0 1\n",
        b"FIELD FieldData 1\nf 1 3 float\n1 2\n",
    ],
)
def test_a_truncated_attribute_is_named_not_an_index_error(attrs: bytes) -> None:
    """Reading off the end of the line list names nothing; say what ran out."""
    tmp = _write_tmp(_ascii_grid(attrs))

    with pytest.raises(CodecError, match="the file ends"):
        read(tmp)


def test_a_field_header_that_never_arrives_is_named() -> None:
    tmp = _write_tmp(_ascii_grid(b"FIELD FieldData 2\nf 1 3 float\n1 2 3\n"))

    with pytest.raises(CodecError, match="FIELD declares"):
        read(tmp)


def test_color_scalars_read_the_same_from_ascii_and_binary() -> None:
    """One byte stands for the 0..1 float the ASCII flavour spells out."""
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    binary = _write_tmp(
        _binary_grid(extra=b"POINT_DATA 3\nCOLOR_SCALARS rgb 3\n" + colors.tobytes())
    )
    ascii_ = _write_tmp(_ascii_grid(b"COLOR_SCALARS rgb 3\n1 0 0\n0 1 0\n0 0 1\n"))

    np.testing.assert_allclose(
        read(binary).vertex_attrs["rgb"], read(ascii_).vertex_attrs["rgb"]
    )
    assert read(binary).vertex_attrs["rgb"].max() == 1.0
