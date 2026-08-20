from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._vtk import read, write
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate


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


def _ascii_grid(attributes: bytes) -> bytes:
    """A one-triangle ASCII UNSTRUCTURED_GRID carrying the given POINT_DATA."""
    return (
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
        b"POINT_DATA 3\n" + attributes
    )


def test_ascii_texture_coordinates_are_read() -> None:
    """TEXTURE_COORDINATES names its dimension, not its type, in column three."""
    tmp = _write_tmp(_ascii_grid(b"TEXTURE_COORDINATES tc 2 float\n0 0\n1 0\n0 1\n"))

    poly = read(tmp)

    assert poly.vertex_attrs["tc"].shape == (3, 2)
    np.testing.assert_allclose(poly.vertex_attrs["tc"][2], [0, 1])


def test_a_lookup_table_does_not_hide_the_arrays_after_it() -> None:
    """A palette is no attribute, but it must still be counted past."""
    tmp = _write_tmp(
        _ascii_grid(
            b"LOOKUP_TABLE palette 2\n"
            b"1 0 0 1\n0 1 0 1\n"
            b"SCALARS after float 1\n"
            b"LOOKUP_TABLE default\n"
            b"7 8 9\n"
        )
    )

    poly = read(tmp)

    assert "palette" not in poly.vertex_attrs
    np.testing.assert_allclose(poly.vertex_attrs["after"], [7, 8, 9])


def test_binary_texture_coordinates_and_lookup_table_are_stepped_over() -> None:
    """In binary an unhandled keyword loses every array after it."""
    tc = np.array([[0, 0], [1, 0], [0, 1]], dtype=">f4").tobytes()
    palette = bytes([255, 0, 0, 255, 0, 255, 0, 255])
    after = np.array([7, 8, 9], dtype=">f4").tobytes()
    extra = (
        b"POINT_DATA 3\n"
        b"TEXTURE_COORDINATES tc 2 float\n" + tc + b"\n"
        b"LOOKUP_TABLE palette 2\n" + palette + b"\n"
        b"SCALARS after float 1\nLOOKUP_TABLE default\n" + after + b"\n"
    )
    tmp = _write_tmp(_binary_grid(extra=extra))

    poly = read(tmp)

    assert poly.vertex_attrs["tc"].shape == (3, 2)
    assert "palette" not in poly.vertex_attrs
    np.testing.assert_allclose(poly.vertex_attrs["after"], [7, 8, 9])


def test_an_unknown_binary_attribute_keyword_says_what_it_costs() -> None:
    """The scan cannot go on past it; a short read must not be a silent one."""
    after = np.array([7, 8, 9], dtype=">f4").tobytes()
    extra = (
        b"POINT_DATA 3\n"
        b"WIDGETS w float\n" + after + b"\n"
        b"SCALARS after float 1\nLOOKUP_TABLE default\n" + after + b"\n"
    )
    tmp = _write_tmp(_binary_grid(extra=extra))

    with pytest.warns(UserWarning, match="WIDGETS"):
        poly = read(tmp)

    assert "after" not in poly.vertex_attrs


def test_a_binary_attribute_running_past_the_file_names_itself() -> None:
    """A short slice used to fail in a reshape that named nothing."""
    extra = b"POINT_DATA 3\nNORMALS n float\n" + b"\x00\x00\x00\x00"
    tmp = _write_tmp(_binary_grid(extra=extra))

    with pytest.raises(CodecError, match="'n'"):
        read(tmp)


@pytest.mark.parametrize(
    "dataset",
    [
        b"DATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1\nORIGIN 0 0 0\nSPACING 1 1 1\n",
        b"DATASET RECTILINEAR_GRID\nDIMENSIONS 2 2 1\n"
        b"X_COORDINATES 2 float\n0 1\n"
        b"Y_COORDINATES 2 float\n0 1\n"
        b"Z_COORDINATES 1 float\n0\n",
        b"DATASET STRUCTURED_GRID\nDIMENSIONS 2 2 1\n"
        b"POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n",
    ],
    ids=["structured_points", "rectilinear_grid", "structured_grid"],
)
def test_structured_datasets_read_normals_and_colors(dataset: bytes) -> None:
    """These three scan their attributes themselves and knew only three."""
    content = (
        b"# vtk DataFile Version 4.2\nstructured\nASCII\n" + dataset + b"POINT_DATA 4\n"
        b"NORMALS n float\n0 0 1\n0 0 1\n0 0 1\n0 0 1\n"
        b"COLOR_SCALARS rgb 3\n1 0 0\n0 1 0\n0 0 1\n1 1 1\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_allclose(poly.vertex_attrs["n"][0], [0, 0, 1])
    np.testing.assert_allclose(poly.vertex_attrs["rgb"][1], [0, 1, 0])


@pytest.mark.parametrize(
    "dataset",
    [
        b"DATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1\nORIGIN 0 0 0\nSPACING 1 1 1\n",
        b"DATASET RECTILINEAR_GRID\nDIMENSIONS 2 2 1\n"
        b"X_COORDINATES 2 float\n0 1\n"
        b"Y_COORDINATES 2 float\n0 1\n"
        b"Z_COORDINATES 1 float\n0\n",
        b"DATASET STRUCTURED_GRID\nDIMENSIONS 2 2 1\n"
        b"POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n",
    ],
    ids=["structured_points", "rectilinear_grid", "structured_grid"],
)
def test_structured_datasets_read_cell_data(dataset: bytes) -> None:
    """A CELL_DATA section used to fall past a chain that asked about points."""
    content = (
        b"# vtk DataFile Version 4.2\nstructured\nASCII\n" + dataset + b"CELL_DATA 1\n"
        b"SCALARS region int\nLOOKUP_TABLE default\n7\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_allclose(poly.element_attrs["region"], [7])


def test_structured_cell_data_of_the_wrong_length_is_dropped_out_loud() -> None:
    """Rows that match no cell cannot be attached, and going quiet hides why."""
    content = (
        b"# vtk DataFile Version 4.2\nstructured\nASCII\n"
        b"DATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"CELL_DATA 3\nSCALARS region int\nLOOKUP_TABLE default\n7 8 9\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="covers 3 of 1 cells"):
        poly = read(tmp)

    assert "region" not in poly.element_attrs


def test_an_unknown_ascii_attribute_keyword_says_it_is_dropping_the_array() -> None:
    """The binary scan said so; the ASCII one skipped the lines in silence."""
    content = (
        b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
        b"POINT_DATA 3\nBOGUS foo 1\n1 2 3\n"
        b"SCALARS good float\nLOOKUP_TABLE default\n7 8 9\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="'BOGUS'"):
        poly = read(tmp)

    # The arrays after the unknown one are still found.
    np.testing.assert_allclose(poly.vertex_attrs["good"], [7, 8, 9])


def test_an_unhandled_structured_keyword_says_it_is_dropping_the_array() -> None:
    """Skipping a line does not step over a payload, so the array is gone."""
    content = (
        b"# vtk DataFile Version 4.2\nstructured\nASCII\n"
        b"DATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"POINT_DATA 4\nBOGUS foo 1\n1 2 3 4\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="'BOGUS'"):
        read(tmp)


def test_an_ascii_array_running_into_the_next_header_names_itself() -> None:
    """float() alone answers a truncated array without naming it or the file."""
    content = (
        b"# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
        b"POINT_DATA 3\nSCALARS s float\nLOOKUP_TABLE default\n1 2\n"
        b"VECTORS v float\n0 0 1\n0 0 1\n0 0 1\n"
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match="'s'"):
        read(tmp)


@pytest.mark.parametrize(
    "dims,expected_type,expected_cells",
    [
        ((3, 3, 3), "hexahedron", 8),
        ((3, 3, 1), "quad", 4),
        ((3, 1, 3), "quad", 4),
        ((1, 3, 3), "quad", 4),
        ((3, 1, 1), "line", 2),
        ((1, 3, 1), "line", 2),
        ((1, 1, 3), "line", 2),
        ((1, 1, 1), "vertex", 0),
    ],
)
def test_a_structured_grid_extends_along_whichever_axes_it_declares(
    dims: tuple[int, int, int], expected_type: str, expected_cells: int
) -> None:
    """An x-z plane is as much a sheet of quads as an x-y one."""
    from polyxios.codecs._vtk import _structured_cell_count, _structured_grid_cells

    cells, etype = _structured_grid_cells(*dims)

    assert etype == expected_type
    assert len(cells) == expected_cells
    # The count the attribute scan uses has to agree with the cells made.
    assert _structured_cell_count(*dims) == expected_cells


def test_a_y_z_plane_reads_as_quads_over_its_own_points() -> None:
    """The old chain called it a run of lines and indexed the wrong points."""
    content = (
        b"# vtk DataFile Version 4.2\nplane\nASCII\n"
        b"DATASET STRUCTURED_POINTS\nDIMENSIONS 1 3 3\nORIGIN 0 0 0\nSPACING 1 1 1\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert len(poly.vertices) == 9
    assert len(poly.element_types) == 4
    np.testing.assert_array_equal(poly.connectivity[:4], [0, 1, 4, 3])


def test_a_metadata_block_does_not_end_the_attribute_scan_in_ascii() -> None:
    """Every VTK writer since 4.2 puts one after each array."""
    content = (
        b"# vtk DataFile Version 4.2\nm\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 2 float\n0 0 0\n1 0 0\n"
        b"METADATA\nINFORMATION 0\n\n"
        b"CELLS 1 3\n2 0 1\nCELL_TYPES 1\n3\n"
        b"POINT_DATA 2\n"
        b"SCALARS a float 1\nLOOKUP_TABLE default\n1 2\n"
        b"METADATA\nINFORMATION 0\n\n"
        b"SCALARS b float 1\nLOOKUP_TABLE default\n3 4\n"
    )
    tmp = _write_tmp(content)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(tmp)

    np.testing.assert_array_equal(poly.vertex_attrs["a"], [1.0, 2.0])
    np.testing.assert_array_equal(poly.vertex_attrs["b"], [3.0, 4.0])


def test_a_metadata_block_does_not_end_the_attribute_scan_in_binary() -> None:
    """The block is text even in a binary file, and used to end the scan."""
    content = (
        b"# vtk DataFile Version 4.2\nm\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 2 float\n"
        + np.array([0, 0, 0, 1, 0, 0], dtype=">f4").tobytes()
        + b"\nCELLS 1 3\n"
        + np.array([2, 0, 1], dtype=">i4").tobytes()
        + b"\nCELL_TYPES 1\n"
        + np.array([3], dtype=">i4").tobytes()
        + b"\nPOINT_DATA 2\nSCALARS a float 1\nLOOKUP_TABLE default\n"
        + np.array([1, 2], dtype=">f4").tobytes()
        + b"\nMETADATA\nINFORMATION 0\n\n"
        b"SCALARS b float 1\nLOOKUP_TABLE default\n"
        + np.array([3, 4], dtype=">f4").tobytes()
        + b"\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_array_equal(poly.vertex_attrs["a"], [1.0, 2.0])
    np.testing.assert_array_equal(poly.vertex_attrs["b"], [3.0, 4.0])


def test_a_metadata_block_inside_a_field_does_not_eat_the_next_array() -> None:
    """A FIELD array carries its own block, between one array and the next."""
    content = (
        b"# vtk DataFile Version 4.2\nm\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 2 float\n0 0 0\n1 0 0\n"
        b"CELLS 1 3\n2 0 1\nCELL_TYPES 1\n3\n"
        b"POINT_DATA 2\nFIELD FieldData 2\n"
        b"first 1 2 double\n1 2\n"
        b"METADATA\nCOMPONENT_NAMES\ncx\n\n"
        b"second 1 2 double\n3 4\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_array_equal(poly.vertex_attrs["first"], [1.0, 2.0])
    np.testing.assert_array_equal(poly.vertex_attrs["second"], [3.0, 4.0])


def test_a_v51_cells_line_counting_offsets_reads_every_cell() -> None:
    """VTK puts the length of the offsets array there, not the cell count."""
    content = (
        b"# vtk DataFile Version 5.1\nv\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
        b"CELLS 3 6\nOFFSETS vtktypeint64\n0 3 6\n"
        b"CONNECTIVITY vtktypeint64\n0 1 2 1 3 2\n"
        b"CELL_TYPES 2\n5 5\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert len(poly.element_types) == 2
    np.testing.assert_array_equal(poly.offsets, [0, 3, 6])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])


def test_a_v51_cells_line_counting_cells_still_reads() -> None:
    """Files older polyxios wrote put the cell count on that line."""
    content = (
        b"# vtk DataFile Version 5.1\nv\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
        b"CELLS 2 6\nOFFSETS vtktypeint64\n0 3 6\n"
        b"CONNECTIVITY vtktypeint64\n0 1 2 1 3 2\n"
        b"CELL_TYPES 2\n5 5\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert len(poly.element_types) == 2
    np.testing.assert_array_equal(poly.offsets, [0, 3, 6])


@pytest.mark.parametrize("binary", [False, True])
def test_a_v51_write_declares_the_length_of_its_offsets_array(binary: bool) -> None:
    """VTK's own reader takes that number literally and finds no cells."""
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp, vtk_version="5.1", binary=binary)

    header = Path(tmp).read_bytes().split(b"OFFSETS")[0].split(b"CELLS ")[1]
    assert header.split()[0] == str(len(poly.offsets)).encode()

    back = read(tmp)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


@pytest.mark.parametrize("section", [b"POLYGONS", b"LINES", b"VERTICES"])
def test_v51_polydata_cell_sections_are_read(section: bytes) -> None:
    """Every VTK release since 9.0 writes polydata cells this way."""
    counts = {b"POLYGONS": 3, b"LINES": 2, b"VERTICES": 1}
    n = counts[section]
    conn = " ".join(str(k) for k in range(n)).encode()
    content = (
        b"# vtk DataFile Version 5.1\np\nASCII\nDATASET POLYDATA\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        + section
        + b" 2 "
        + str(n).encode()
        + b"\nOFFSETS vtktypeint64\n0 "
        + str(n).encode()
        + b"\nCONNECTIVITY vtktypeint64\n"
        + conn
        + b"\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.connectivity, list(range(n)))


def test_a_binary_structured_grid_keeps_a_cell_data_written_first() -> None:
    """The line after the POINTS payload was stepped over twice."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nBINARY\nDATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 1\nPOINTS 4 float\n"
        + np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=">f4").tobytes()
        + b"\nCELL_DATA 1\nSCALARS c float 1\nLOOKUP_TABLE default\n"
        + np.array([7], dtype=">f4").tobytes()
        + b"\nPOINT_DATA 4\nSCALARS p float 1\nLOOKUP_TABLE default\n"
        + np.array([1, 2, 3, 4], dtype=">f4").tobytes()
        + b"\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_array_equal(poly.element_attrs["c"], [7.0])
    np.testing.assert_array_equal(poly.vertex_attrs["p"], [1.0, 2.0, 3.0, 4.0])


def test_a_rectilinear_grid_follows_its_coordinates_not_its_header() -> None:
    """The points are the outer product of the coordinate arrays."""
    content = (
        b"# vtk DataFile Version 4.2\nr\nASCII\nDATASET RECTILINEAR_GRID\n"
        b"DIMENSIONS 3 2 1\n"
        b"X_COORDINATES 2 float\n0 1\n"
        b"Y_COORDINATES 2 float\n0 1\n"
        b"Z_COORDINATES 1 float\n0\n"
        b"POINT_DATA 4\nSCALARS s float 1\nLOOKUP_TABLE default\n1 2 3 4\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="DIMENSIONS"):
        poly = read(tmp)

    assert len(poly.vertices) == 4
    np.testing.assert_array_equal(poly.vertex_attrs["s"], [1.0, 2.0, 3.0, 4.0])
    # The cells have to index points that exist.
    assert int(poly.connectivity.max()) < len(poly.vertices)


@pytest.mark.parametrize("shape", [(4, 3, 3), (4, 6)])
def test_a_binary_tensor_is_written_as_binary(shape: tuple[int, ...]) -> None:
    """Both tensor branches spelled their numbers into a binary file."""
    poly = _synthetic_mesh()
    poly.vertex_attrs["T"] = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(
        shape
    )
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name

    write(poly, tmp, binary=True)
    back = read(tmp)

    assert back.vertex_attrs["T"].shape == (4, 3, 3)
    assert b"TENSORS T double\n0.0" not in Path(tmp).read_bytes()


@pytest.mark.parametrize("binary", [False, True])
def test_a_double_section_holds_every_digit_of_a_double(binary: bool) -> None:
    """Ten significant digits is seven short of what a double carries."""
    verts = np.array([[1 / 3, 2 / 7, 1 / 9]] * 3, dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    poly.vertex_attrs["s"] = np.array([1 / 3, 2 / 7, 1 / 9])
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name

    write(poly, tmp, binary=binary)
    back = read(tmp)

    np.testing.assert_array_equal(back.vertices, verts)
    np.testing.assert_array_equal(back.vertex_attrs["s"], poly.vertex_attrs["s"])


def test_an_unterminated_metadata_block_ends_at_the_geometry() -> None:
    """Left open it swallowed the CELLS after it and the rest of the file."""
    content = (
        b"# vtk DataFile Version 4.2\nm\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"POINT_DATA 3\nSCALARS s float 1\nLOOKUP_TABLE default\n1 2 3\n"
        b"METADATA\nINFORMATION 1\n"
        b"CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    assert len(poly.element_types) == 1
    np.testing.assert_array_equal(poly.vertex_attrs["s"], [1.0, 2.0, 3.0])


def test_a_section_declaring_more_than_the_file_holds_costs_the_section() -> None:
    """The geometry was already whole; refusing it lost more than it saved."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"POINT_DATA 8\nSCALARS p float 1\nLOOKUP_TABLE default\n1 2 3 4 5 6 7 8\n"
        b"CELL_DATA 99\nSCALARS c float 1\nLOOKUP_TABLE default\n9\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="declares 99 values"):
        poly = read(tmp)

    np.testing.assert_array_equal(poly.vertex_attrs["p"], [1, 2, 3, 4, 5, 6, 7, 8])
    assert "c" not in poly.element_attrs


def test_a_structured_grid_whose_header_outruns_its_points_keeps_the_points() -> None:
    """The cells DIMENSIONS describes indexed points the file never held."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 3 3 1\nPOINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
        b"POINT_DATA 4\nSCALARS s float 1\nLOOKUP_TABLE default\n1 2 3 4\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="DIMENSIONS says 3 3 1"):
        poly = read(tmp)

    assert len(poly.vertices) == 4
    assert len(poly.element_types) == 0
    assert len(poly.connectivity) == 0
    np.testing.assert_array_equal(poly.vertex_attrs["s"], [1, 2, 3, 4])
    assert poly.global_attrs["vtk_dimensions"] == [3, 3, 1]
    validate(poly)


def test_a_structured_grid_point_section_is_read_by_its_own_count() -> None:
    """Reading by the mesh's count walked one array into the next header."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 1\nPOINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
        b"POINT_DATA 6\nSCALARS s float 1\nLOOKUP_TABLE default\n1 2 3 4 5 6\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="covers 6 of 4 points"):
        poly = read(tmp)

    assert "s" not in poly.vertex_attrs
    assert len(poly.element_types) == 1


def test_an_unstructured_point_section_is_read_by_its_own_count() -> None:
    """Read by the mesh's count, the first array walks into the next header.

    The arrays here belong to no point of this mesh and are dropped, but the
    section is still walked by the length it declares, so the CELL_DATA
    after it is found rather than parsed as more of the last array.
    """
    content = (
        b"# vtk DataFile Version 4.2\nu\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
        b"POINT_DATA 5\nSCALARS s float 1\nLOOKUP_TABLE default\n1 2 3 4 5\n"
        b"VECTORS v float\n1 0 0\n0 1 0\n0 0 1\n1 1 0\n0 1 1\n"
        b"CELL_DATA 1\nSCALARS c float 1\nLOOKUP_TABLE default\n7\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="covers 5 of 3 points"):
        poly = read(tmp)

    assert poly.vertex_attrs == {}
    np.testing.assert_array_equal(poly.element_attrs["c"], [7.0])


def test_a_binary_scalars_without_a_lookup_table_reads_its_own_values() -> None:
    """The line skipped unconditionally was payload up to its first newline."""
    content = (
        b"# vtk DataFile Version 4.2\nb\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n"
        + np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=">f4").tobytes()
        + b"\nCELLS 1 4\n"
        + np.array([3, 0, 1, 2], dtype=">i4").tobytes()
        + b"\nCELL_TYPES 1\n"
        + np.array([5], dtype=">i4").tobytes()
        + b"\nPOINT_DATA 3\nSCALARS s double 1\n"
        + np.array([10.0, 20.0, 30.0], dtype=">f8").tobytes()
        + b"\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_array_equal(poly.vertex_attrs["s"], [10.0, 20.0, 30.0])


@pytest.mark.parametrize("dataset", [b"UNSTRUCTURED_GRID", b"POLYDATA"])
def test_a_truncated_binary_points_block_names_itself(dataset: bytes) -> None:
    """The whole-file bound clears a block that still runs off the end."""
    content = (
        b"# vtk DataFile Version 4.2\np\nBINARY\nDATASET " + dataset + b"\n"
        b"# " + b"x" * 5000 + b"\n"
        b"POINTS 10 double\n" + np.zeros(12, dtype=">f8").tobytes()
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match="POINTS"):
        read(tmp)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (b"SCALARS", "no array name"),
        (b"VECTORS", "no array name"),
        (b"TENSORS", "no array name"),
        (b"COLOR_SCALARS", "no array name"),
        (b"SCALARS s float x", "is not a number"),
        (b"FIELD FieldData", "no field 2"),
    ],
)
def test_a_malformed_attribute_header_names_the_line(header: bytes, match: str) -> None:
    """These fell out of parts[1] and int() naming neither file nor line."""
    content = (
        b"# vtk DataFile Version 4.2\nu\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
        b"POINT_DATA 3\n" + header + b"\n"
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match=match):
        read(tmp)


def test_a_malformed_binary_attribute_header_names_the_byte() -> None:
    """A binary file has no line to name, so the offset stands in for one."""
    content = (
        b"# vtk DataFile Version 4.2\nb\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n"
        + np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=">f4").tobytes()
        + b"\nCELLS 1 4\n"
        + np.array([3, 0, 1, 2], dtype=">i4").tobytes()
        + b"\nCELL_TYPES 1\n"
        + np.array([5], dtype=">i4").tobytes()
        + b"\nPOINT_DATA 3\nSCALARS\n"
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match="no array name"):
        read(tmp)


def test_a_malformed_structured_header_costs_its_section_only() -> None:
    """The geometry was whole before the scan reached the bad header."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 1 1\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"POINT_DATA 2\nSCALARS\n1 2\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning, match="no array name"):
        poly = read(tmp)

    assert len(poly.vertices) == 2
    assert poly.vertex_attrs == {}


def test_cell_data_is_dropped_when_the_grid_leaves_the_mesh_no_cells() -> None:
    """Kept against DIMENSIONS, the array outlived the cells it belonged to."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 2\nPOINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n"
        b"CELL_DATA 1\nSCALARS q float 1\nLOOKUP_TABLE default\n7\n"
    )
    tmp = _write_tmp(content)

    with pytest.warns(UserWarning) as caught:
        poly = read(tmp)

    assert any("covers 1 of 0 cells" in str(w.message) for w in caught)
    assert len(poly.element_types) == 0
    assert poly.element_attrs == {}
    validate(poly)


def test_binary_points_are_read_as_the_type_they_declare() -> None:
    """'POINTS n int' read at four bytes a float gave coordinates from nowhere."""
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=">i4").tobytes()
    content = (
        b"# vtk DataFile Version 4.2\np\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 int\n" + pts + b"\nCELLS 1 4\n"
        b"" + np.array([3, 0, 1, 2], dtype=">i4").tobytes() + b"\nCELL_TYPES 1\n"
        b"" + np.array([5], dtype=">i4").tobytes() + b"\n"
    )
    tmp = _write_tmp(content)

    poly = read(tmp)

    np.testing.assert_array_equal(poly.vertices, [[0, 0, 0], [2, 0, 0], [0, 3, 0]])


def test_a_binary_array_of_an_unknown_type_names_it() -> None:
    """Guessing a width reads numbers the file never held."""
    content = _binary_grid(
        extra=b"POINT_DATA 3\nSCALARS s quadruple 1\n" + b"\x00" * 24
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match="'quadruple'"):
        read(tmp)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (b"POINT_DATA x\nSCALARS s float 1\n1 2 3\n", "count 'x' is not a number"),
        (b"POINT_DATA\n", "no field 1"),
    ],
)
def test_a_data_section_count_that_is_not_a_count_names_the_line(
    header: bytes, match: str
) -> None:
    """int() on the header answered with a ValueError naming nothing."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n" + header
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match=match):
        read(tmp)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (b"DIMENSIONS 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n", "no field 3"),
        (b"DIMENSIONS 2 2 1\nORIGIN a b c\nSPACING 1 1 1\n", "not all numbers"),
        (b"DIMENSIONS 2 2 1\nORIGIN 0 0\nSPACING 1 1 1\n", "fewer than 3 values"),
    ],
)
def test_a_structured_points_header_that_is_not_numbers_names_the_line(
    header: bytes, match: str
) -> None:
    """A short ORIGIN was an IndexError naming an axis, not a file."""
    content = (
        b"# vtk DataFile Version 4.2\ns\nASCII\nDATASET STRUCTURED_POINTS\n" + header
    )
    tmp = _write_tmp(content)

    with pytest.raises(CodecError, match=match):
        read(tmp)
