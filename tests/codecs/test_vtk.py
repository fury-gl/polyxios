from __future__ import annotations

import dataclasses
from pathlib import Path
import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES_INV
from polyxios.codecs._vtk import read, write
from polyxios.exceptions import (
    CodecError,
    IndexOverflowError,
    LazyReadError,
    UnknownElementTypeError,
)
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


def test_a_v42_write_spells_cells_not_offsets() -> None:
    poly = _synthetic_mesh()
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp = f.name
    write(poly, tmp, vtk_version="4.2")
    content = Path(tmp).read_text()
    assert "CELLS" in content
    assert "OFFSETS" not in content


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


def test_a_field_dataset_is_read_rather_than_refused() -> None:
    """The dispatch asked what a line starts with, keyword and all."""
    tmp = _write_tmp(
        b"# vtk DataFile Version 2.0\ns\nASCII\nDATASET FIELD FieldData 1\n"
        b"arr 1 2 float\n1 2\n"
    )

    with pytest.warns(UserWarning, match="no geometry"):
        poly = read(tmp)

    np.testing.assert_allclose(poly.global_attrs["arr"], [1.0, 2.0])


def test_v51_cells_are_found_whatever_version_the_header_declares() -> None:
    """Versions compared as strings sort '10.0' below '5.1'."""
    tmp = _write_tmp(
        b"# vtk DataFile Version 10.0\ns\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 2 3\nOFFSETS vtktypeint64\n0 3\n"
        b"CONNECTIVITY vtktypeint64\n0 1 2\nCELL_TYPES 1\n5\n"
    )

    poly = read(tmp)

    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(poly.offsets, [0, 3])
    validate(poly)


@pytest.mark.parametrize(
    "cells",
    [
        pytest.param(
            b"CELLS 2 3\nOFFSETS vtktypeint64\n0 3\nMETADATA\nINFORMATION 0\n\n"
            b"CONNECTIVITY vtktypeint64\n0 1 2\n",
            id="after-offsets",
        ),
        pytest.param(
            b"CELLS 2 3\nOFFSETS vtktypeint64\n0 3\n"
            b"CONNECTIVITY vtktypeint64\nMETADATA\nINFORMATION 0\n\n0 1 2\n",
            id="after-connectivity-keyword",
        ),
        pytest.param(
            b"CELLS 2 3\nOFFSETS vtktypeint64\n0 3\nMETADATA\nINFORMATION 0\n"
            b"CONNECTIVITY vtktypeint64\n0 1 2\n",
            id="unterminated",
        ),
    ],
)
def test_metadata_inside_a_v51_cells_section_is_stepped_over(cells: bytes) -> None:
    """VTK follows a cell array with a METADATA block; it is not offsets."""
    tmp = _write_tmp(
        b"# vtk DataFile Version 5.1\ns\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n" + cells + b"CELL_TYPES 1\n5\n"
    )

    poly = read(tmp)

    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(poly.offsets, [0, 3])
    validate(poly)


def test_metadata_inside_a_binary_v51_cells_section_is_stepped_over() -> None:
    """The binary flavour writes the same block as text between the arrays."""
    tmp = _write_tmp(
        b"# vtk DataFile Version 5.1\ns\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 3 float\n"
        + np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=">f4").tobytes()
        + b"\nCELLS 2 3\nOFFSETS vtktypeint64\n"
        + np.array([0, 3], dtype=">i8").tobytes()
        + b"\nMETADATA\nINFORMATION 0\n\nCONNECTIVITY vtktypeint64\n"
        + np.array([0, 1, 2], dtype=">i8").tobytes()
        + b"\nCELL_TYPES 1\n"
        + np.array([5], dtype=">i4").tobytes()
        + b"\n"
    )

    poly = read(tmp)

    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])
    np.testing.assert_array_equal(poly.offsets, [0, 3])
    validate(poly)


@pytest.mark.parametrize(
    "dataset",
    [
        ("DATASET STRUCTURED_POINTS\nDIMENSIONS 0 3 3\nORIGIN 0 0 0\nSPACING 1 1 1\n"),
        (
            "DATASET RECTILINEAR_GRID\n"
            "DIMENSIONS 0 3 3\n"
            "X_COORDINATES 0 float\n"
            "Y_COORDINATES 3 float\n"
            "0 1 2\n"
            "Z_COORDINATES 3 float\n"
            "0 1 2\n"
        ),
    ],
    ids=["structured_points", "rectilinear_grid"],
)
def test_an_axis_of_no_points_holds_no_cells(dataset: str, tmp_path) -> None:
    """One empty axis empties the grid - the point count is a product.

    The other two axes' cells were counted anyway, so a grid of no vertices
    came back holding four quads whose corners named points the file never
    laid out. The strides those corners were built from are zero as well.
    """
    path = tmp_path / "empty_axis.vtk"
    path.write_text(f"# vtk DataFile Version 3.0\ntitle\nASCII\n{dataset}")

    poly = read(path)
    assert len(poly.vertices) == 0
    assert len(poly.element_types) == 0
    assert len(poly.connectivity) == 0
    validate(poly)


def test_dimensions_the_points_do_not_cover_are_counted_from_zero(tmp_path) -> None:
    """A negative DIMENSIONS is no count of points, and is not reported as one."""
    path = tmp_path / "negative.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\ntitle\nASCII\n"
        "DATASET STRUCTURED_GRID\n"
        "DIMENSIONS -1 3 3\n"
        "POINTS 9 float\n"
        "0 0 0 1 0 0 2 0 0 0 1 0 1 1 0 2 1 0 0 2 0 1 2 0 2 2 0\n"
    )

    with pytest.warns(UserWarning, match="which is 0 point"):
        poly = read(path)
    assert len(poly.vertices) == 9
    assert len(poly.element_types) == 0
    validate(poly)


# ---------------------------------------------------------------------------
# FIELD FieldData: the mesh's own metadata
# ---------------------------------------------------------------------------


def _field_grid(field: bytes) -> bytes:
    """A one-triangle ASCII grid carrying the given dataset FIELD block."""
    return (
        b"# vtk DataFile Version 4.2\n"
        b"field data\n"
        b"ASCII\n"
        b"DATASET UNSTRUCTURED_GRID\n" + field + b"POINTS 3 float\n"
        b"0 0 0\n1 0 0\n0 1 0\n"
        b"CELLS 1 4\n"
        b"3 0 1 2\n"
        b"CELL_TYPES 1\n"
        b"5\n"
    )


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("vtk_version", ["4.2", "5.1"])
def test_a_vtk_write_holds_the_field_data(
    tmp_path, binary: bool, vtk_version: str
) -> None:
    """The writer dropped global_attrs, so a time value, a material constant
    or a solver tolerance did not survive being written."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"TimeValue": 0.25, "steps": 12},
    )
    path = tmp_path / "field.vtk"

    write(poly, path, binary=binary, vtk_version=vtk_version)
    back = read(path)

    np.testing.assert_allclose(back.global_attrs["TimeValue"], [0.25])
    np.testing.assert_array_equal(back.global_attrs["steps"], [12])
    # An integer that came home a double is a different value to whatever
    # reads the file next.
    assert back.global_attrs["steps"].dtype.kind == "i"


def test_a_dataset_field_block_is_read() -> None:
    """A block between the DATASET keyword and the geometry belongs to the
    mesh; read as an attribute section it would have been dropped for
    covering neither the points nor the cells."""
    path = _write_tmp(
        _field_grid(b"FIELD FieldData 2\nTimeValue 1 1 double\n0.5\nid 1 1 int\n7\n")
    )

    poly = read(path)

    np.testing.assert_allclose(poly.global_attrs["TimeValue"], [0.5])
    np.testing.assert_array_equal(poly.global_attrs["id"], [7])
    assert poly.vertices.shape == (3, 3)
    assert len(poly.element_types) == 1


def test_a_multi_component_field_array_comes_back_in_rows() -> None:
    path = _write_tmp(
        _field_grid(b"FIELD FieldData 1\nbounds 3 2 double\n0 0 0\n1 1 1\n")
    )

    np.testing.assert_allclose(
        read(path).global_attrs["bounds"], [[0, 0, 0], [1, 1, 1]]
    )


def test_a_point_data_field_block_stays_an_attribute() -> None:
    """FIELD inside POINT_DATA names arrays over the points, not the mesh."""
    path = _write_tmp(
        _field_grid(b"") + b"POINT_DATA 3\nFIELD FieldData 1\nf 1 3 double\n1 2 3\n"
    )

    poly = read(path)

    np.testing.assert_allclose(poly.vertex_attrs["f"], [1, 2, 3])
    assert poly.global_attrs == {}


def test_a_field_value_wider_than_a_double_survives(tmp_path) -> None:
    """Read through float() an id past 2**53 comes back a different number."""
    wide = np.array([9007199254740993], dtype=np.int64)
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"case_id": wide},
    )
    path = tmp_path / "wide.vtk"

    write(poly, path)
    np.testing.assert_array_equal(read(path).global_attrs["case_id"], wide)


def test_a_global_no_field_array_can_hold_is_named_and_dropped(tmp_path) -> None:
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"solver": "polyxios", "steps": 12},
    )
    path = tmp_path / "mixed.vtk"

    with pytest.warns(UserWarning, match=r"global_attrs \['solver'\]"):
        write(poly, path)

    assert "solver" not in read(path).global_attrs


def test_the_grid_a_structured_read_recorded_travels_as_field_data(tmp_path) -> None:
    """vtk_dimensions describes a grid this writer does not spell - it writes
    an UNSTRUCTURED_GRID - so skipping it would drop it without a word where
    the XML family of the same metadata carries it."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"vtk_dimensions": [2, 2, 1], "steps": 3},
    )
    path = tmp_path / "grid.vtk"

    write(poly, path)

    back = read(path).global_attrs
    np.testing.assert_array_equal(back["vtk_dimensions"], [2, 2, 1])
    np.testing.assert_array_equal(back["steps"], [3])


def test_a_structured_read_takes_its_own_grid_over_the_field_block(
    tmp_path,
) -> None:
    """The grid the points were laid out on wins the keys it spells itself, so
    carrying them as field data cannot hand a reader a second copy of one."""
    path = _write_tmp(
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"FIELD FieldData 1\nvtk_dimensions 3 1 int\n9 9 9\n"
    )

    np.testing.assert_array_equal(read(path).global_attrs["vtk_dimensions"], [2, 2, 2])


def test_a_tag_group_travels_as_its_own_column(tmp_path) -> None:
    """Legacy VTK has no set of its own; a POINT_DATA or CELL_DATA column
    named for a group carries it, and an element in two groups is named by
    both columns."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))],
        vertex_tags={"corner": np.array([0, 3], dtype=np.int32)},
        element_tags={
            "a": np.array([0], dtype=np.int32),
            "b": np.array([0, 1], dtype=np.int32),
        },
    )
    path = tmp_path / "tagged.vtk"

    write(poly, path)
    back = read(path)

    np.testing.assert_array_equal(back.element_tags["a"], [0])
    np.testing.assert_array_equal(back.element_tags["b"], [0, 1])
    np.testing.assert_array_equal(back.vertex_tags["corner"], [0, 3])
    assert back.element_attrs == {}


def test_a_tag_name_a_legacy_header_cannot_spell_is_reported(tmp_path) -> None:
    """A legacy header names its array in a whitespace-separated field, and
    nothing in the format escapes one: written anyway, the name would be read
    back as a name and a stray token, and the array after it as its values."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"outer wall": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "spaced.vtk"

    with pytest.warns(UserWarning, match="holds whitespace"):
        write(poly, path)

    assert read(path).element_tags == {}


def test_a_metadata_name_a_legacy_header_cannot_spell_is_reported(
    tmp_path,
) -> None:
    """A FIELD header names its array in the same whitespace-separated field
    an attribute header does; written anyway, the file reads back as neither
    the array nor the geometry after it."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.global_attrs["time step"] = 1.5
    poly.global_attrs["kept"] = 2.5
    path = tmp_path / "spaced.vtk"

    with pytest.warns(UserWarning, match="holds whitespace"):
        write(poly, path)

    # The header counts what survived, so the reader finds the geometry where
    # the block ends rather than one array further on.
    back = read(path)
    assert "time step" not in back.global_attrs
    np.testing.assert_allclose(back.global_attrs["kept"], [2.5])
    np.testing.assert_allclose(back.vertices, poly.vertices)


def test_metadata_of_no_components_is_dropped_rather_than_written(
    tmp_path,
) -> None:
    """A field header carries a component count and a tuple count, and an
    array of no components has neither; it used to divide by zero."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.global_attrs["hollow"] = np.zeros((2, 0))
    path = tmp_path / "hollow.vtk"

    with pytest.warns(UserWarning, match="no"):
        write(poly, path)

    assert read(path).global_attrs == {}


def test_a_tag_group_naming_no_element_of_this_mesh_is_reported(
    tmp_path,
) -> None:
    """A column of ones and zeros cannot say a member was dropped, so the
    writer says it, the way the *Elset writers already do."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"stale": np.array([0, 99], dtype=np.int32)},
    )
    path = tmp_path / "stale.vtk"

    with pytest.warns(UserWarning, match="index no cell"):
        write(poly, path)

    np.testing.assert_array_equal(read(path).element_tags["stale"], [0])


_STRUCTURED_BODIES: dict[str, bytes] = {
    "STRUCTURED_POINTS": b"DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n",
    "RECTILINEAR_GRID": (
        b"DIMENSIONS 2 2 2\n"
        b"X_COORDINATES 2 float\n0 1\n"
        b"Y_COORDINATES 2 float\n0 1\n"
        b"Z_COORDINATES 2 float\n0 1\n"
    ),
    "STRUCTURED_GRID": (
        b"DIMENSIONS 2 2 2\n"
        b"POINTS 8 float\n"
        b"0 0 0 1 0 0 0 1 0 1 1 0 0 0 1 1 0 1 0 1 1 1 1 1\n"
    ),
}


@pytest.mark.parametrize("dataset", sorted(_STRUCTURED_BODIES))
def test_a_structured_dataset_field_block_is_read(dataset: str) -> None:
    """VTK's own writer puts a time value here, and these three readers used
    to hand back the grid they rebuilt and nothing else."""
    path = _write_tmp(
        b"# vtk DataFile Version 4.2\nfield data\nASCII\nDATASET "
        + dataset.encode()
        + b"\n"
        + _STRUCTURED_BODIES[dataset]
        + b"FIELD FieldData 2\nTimeValue 1 1 double\n0.5\nid 1 1 int\n7\n"
        b"POINT_DATA 8\nSCALARS s double 1\nLOOKUP_TABLE default\n"
        b"0 1 2 3 4 5 6 7\n"
    )

    poly = read(path)

    np.testing.assert_allclose(poly.global_attrs["TimeValue"], [0.5])
    np.testing.assert_array_equal(poly.global_attrs["id"], [7])
    # The type the header declared, not the double every attribute here is.
    assert poly.global_attrs["id"].dtype.kind == "i"
    # The grid the reader rebuilt still wins the keys it spells itself.
    np.testing.assert_array_equal(poly.global_attrs["vtk_dimensions"], [2, 2, 2])
    np.testing.assert_allclose(poly.vertex_attrs["s"], np.arange(8.0))


def test_a_binary_structured_dataset_field_block_is_read() -> None:
    """The payload is raw big-endian bytes, so the line cursor has to step
    over it by width rather than by looking for the next newline."""
    path = _write_tmp(
        b"# vtk DataFile Version 4.2\nfield data\nBINARY\n"
        b"DATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"FIELD FieldData 1\nid 1 1 int\n"
        + np.array([7], dtype=">i4").tobytes()
        + b"\nPOINT_DATA 8\nSCALARS s double 1\nLOOKUP_TABLE default\n"
        + np.arange(8, dtype=">f8").tobytes()
        + b"\n"
    )

    poly = read(path)

    np.testing.assert_array_equal(poly.global_attrs["id"], [7])
    np.testing.assert_allclose(poly.vertex_attrs["s"], np.arange(8.0))


def test_a_second_dataset_field_block_adds_to_the_first() -> None:
    """Two blocks used to leave only the last: the second rebound the name
    the first had been read into."""
    path = _write_tmp(
        _field_grid(
            b"FIELD FieldData 1\na 1 1 int\n1\nFIELD FieldData 1\nb 1 1 int\n2\n"
        )
    )

    poly = read(path)

    np.testing.assert_array_equal(poly.global_attrs["a"], [1])
    np.testing.assert_array_equal(poly.global_attrs["b"], [2])


def test_an_attribute_a_tag_column_shadows_is_reported(tmp_path) -> None:
    """One array of that name goes out, and a reader takes it for the group
    whichever of the two wrote it - so the attribute is lost either way."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"polyxios_tag_z": np.array([7])},
        element_tags={"z": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "shadow.vtk"

    with pytest.warns(UserWarning, match="what a tag group"):
        write(poly, path)

    back = read(path)
    np.testing.assert_array_equal(back.element_tags["z"], [0])
    assert back.element_attrs == {}


def test_a_tag_group_with_no_name_is_reported(tmp_path) -> None:
    """The column's name is the only thing that says it is a group; a group
    with none reads back as neither a group nor an attribute."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "unnamed.vtk"

    with pytest.warns(UserWarning, match="no name a data array can carry"):
        write(poly, path)

    assert read(path).element_tags == {}


def test_a_wide_integer_field_array_is_spelled_a_tuple_to_a_row(tmp_path) -> None:
    """The whole block used to go on one line, which a mesh carrying a large
    metadata array turns into a single enormous string."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    poly.global_attrs["rows"] = np.arange(12, dtype=np.int64).reshape(4, 3)
    path = tmp_path / "rows.vtk"

    write(poly, path)

    body = path.read_text().splitlines()
    assert "rows 3 4 long" in body
    assert body[body.index("rows 3 4 long") + 1] == "0 1 2"
    np.testing.assert_array_equal(
        read(path).global_attrs["rows"], np.arange(12).reshape(4, 3)
    )


@pytest.mark.parametrize("title", [b"FIELD data", b"Field data from run 3"])
@pytest.mark.parametrize(
    ("dataset", "grid"),
    [
        (b"STRUCTURED_POINTS", b"DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n"),
        (
            b"RECTILINEAR_GRID",
            b"DIMENSIONS 2 1 1\nX_COORDINATES 2 float\n0 1\n"
            b"Y_COORDINATES 1 float\n0\nZ_COORDINATES 1 float\n0\n",
        ),
        (b"STRUCTURED_GRID", b"DIMENSIONS 2 1 1\nPOINTS 2 float\n0 0 0\n1 0 0\n"),
    ],
)
def test_a_title_line_is_not_read_as_a_field_block(
    title: bytes, dataset: bytes, grid: bytes
) -> None:
    """These readers scan from the top of the file - the header may carry
    blank lines, so the DATASET line is at no fixed index - which puts the
    free-text title in front of every keyword branch. A title of three words
    or more spells a count where a block header carries one."""
    path = _write_tmp(
        b"# vtk DataFile Version 4.2\n" + title + b"\nASCII\n"
        b"DATASET " + dataset + b"\n" + grid
    )

    assert "vtk_dimensions" in read(path).global_attrs


def test_a_field_header_with_no_type_reads_the_same_in_either_encoding() -> None:
    """The format makes the type field mandatory, so a header without one is
    malformed - but both scans have to answer it the same way, or one file
    reads back as two different meshes depending on how it was written."""
    body_ascii = (
        "# vtk DataFile Version 4.2\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "FIELD FieldData 1\narr 1 2\n1.5 2.5\n"
        "POINTS 3 double\n0 0 0\n1 0 0\n0 1 0\n"
        "CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
    )
    ascii_poly = read(_write_tmp(body_ascii.encode()))

    body_binary = (
        b"# vtk DataFile Version 4.2\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 1\narr 1 2\n"
        + np.array([1.5, 2.5], dtype=">f4").tobytes()
        + b"\nPOINTS 3 double\n"
        + np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=">f8").tobytes()
        + b"\nCELLS 1 4\n"
        + np.array([3, 0, 1, 2], dtype=">i4").tobytes()
        + b"\nCELL_TYPES 1\n"
        + np.array([5], dtype=">i4").tobytes()
        + b"\n"
    )
    binary_poly = read(_write_tmp(body_binary))

    np.testing.assert_allclose(ascii_poly.global_attrs["arr"], [1.5, 2.5])
    np.testing.assert_allclose(binary_poly.global_attrs["arr"], [1.5, 2.5])


def test_a_short_field_header_is_refused_in_either_encoding() -> None:
    """A header short of a name, a component count and a tuple count says
    nothing about where its payload ends. Reading on would frame the next
    header out of the array's own bytes, so both scans refuse it."""
    body_ascii = (
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 1\narr 1\n1.5\n"
        b"POINTS 1 double\n0 0 0\n"
    )
    body_binary = (
        b"# vtk DataFile Version 4.2\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 1\narr 1\n"
        + np.array([1.5], dtype=">f4").tobytes()
        + b"\nPOINTS 1 double\n"
        + np.array([[0.0, 0, 0]], dtype=">f8").tobytes()
        + b"\n"
    )
    for body in (body_ascii, body_binary):
        with pytest.raises(CodecError, match="no field 2"):
            read(_write_tmp(body))


def test_a_field_block_short_of_the_arrays_it_declares_is_refused() -> None:
    """The count is the only thing that says where the block ends, so a file
    that stops inside one is refused however the reader walks it - as a
    dataset of its own, and under a structured header."""
    unstructured = (
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 2\narr 1 1\n1.5\n"
    )
    structured = (
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET STRUCTURED_POINTS\n"
        b"DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\n"
        b"FIELD FieldData 2\narr 1 1\n1.5\n"
    )
    for body in (unstructured, structured):
        with pytest.raises(CodecError, match="ends before their headers"):
            read(_write_tmp(body))


def test_a_bool_is_spelled_the_way_the_xml_family_spells_one(tmp_path) -> None:
    """No VTK type names a bool, here or in the XML family, so both writers
    widen it to the double it converts to rather than naming one file's
    metadata two different types."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        global_attrs={"flag": True},
    )
    path = tmp_path / "flag.vtk"

    write(poly, path)

    assert b"flag 1 1 double" in path.read_bytes()
    np.testing.assert_array_equal(read(path).global_attrs["flag"], [1.0])


def test_an_attribute_keyed_by_something_that_is_not_text_still_writes(
    tmp_path,
) -> None:
    """A legacy header names its array in a whitespace-separated field. A
    number is one such field; the rule is about what the field can hold, not
    about refusing a caller's key."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={5: np.zeros(3)},
    )
    path = tmp_path / "numbered.vtk"

    write(poly, path)

    assert list(read(path).vertex_attrs) == ["5"]


def test_a_section_whose_names_are_all_unspellable_writes_no_header(
    tmp_path,
) -> None:
    """Every section here declares itself ahead of its contents, so the names
    have to be checked before the header goes down: a POINT_DATA promising
    arrays that all turn out to be unnameable is a header over nothing."""
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"bad name": np.zeros(3), "worse name": np.zeros(3)},
    )
    path = tmp_path / "unnameable.vtk"

    with pytest.warns(UserWarning, match="holds whitespace"):
        write(poly, path)

    assert b"POINT_DATA" not in path.read_bytes()
    assert read(path).vertex_attrs == {}


def test_a_cells_block_wrapped_over_several_lines_reads(tmp_path) -> None:
    """A v4.2 CELLS block is a run of numbers the header's second field
    counts, not a line apiece, and a writer is free to wrap it. Read one cell
    to a line only, such a file was refused for a row that was never short."""
    path = tmp_path / "wrapped.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 4 double\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "CELLS 1 5\n4 0 1\n2 3\nCELL_TYPES 1\n10\n"
    )

    poly = read(path)

    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 3])
    np.testing.assert_array_equal(poly.offsets, [0, 4])


def test_a_points_block_wrapped_over_several_lines_reads(tmp_path) -> None:
    """The twin of the CELLS block: POINTS declares a vertex count and the
    coordinates are a run of numbers under it, however they are broken up."""
    path = tmp_path / "wrapped_points.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 double\n0 0 0 1 0 0\n0 1 0\n"
        "CELLS 1 4\n3 0 1 2\nCELL_TYPES 1\n5\n"
    )

    poly = read(path)

    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_a_wrapped_block_reads_as_the_same_mesh_as_an_unwrapped_one(
    tmp_path,
) -> None:
    """One file spelled two ways is one mesh; the two readings are what a
    reader that only knew one of the layouts could not answer alike."""
    head = (
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 4 double\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
    )
    rows = tmp_path / "rows.vtk"
    rows.write_text(head + "CELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n10\n")
    wrapped = tmp_path / "wrapped.vtk"
    wrapped.write_text(head + "CELLS 1 5\n4\n0 1\n2 3\nCELL_TYPES 1\n10\n")

    one, two = read(rows), read(wrapped)

    np.testing.assert_array_equal(one.connectivity, two.connectivity)
    np.testing.assert_array_equal(one.offsets, two.offsets)
    np.testing.assert_array_equal(one.element_types, two.element_types)


def test_a_blank_line_inside_a_cells_block_reads_as_the_wrap_it_is(
    tmp_path,
) -> None:
    """A row of no tokens is not one cell to a line, so the block is read
    again as the run of numbers its header counts. The compiled walk indexes
    the row's first token with its bounds check off, so an empty one has to
    be refused there before it is indexed rather than read past the end of
    the list."""
    path = tmp_path / "blank_row.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 4 double\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "CELLS 2 8\n3 0 1 2\n\n3 1 2 3\nCELL_TYPES 2\n5\n5\n"
    )

    poly = read(path)

    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 2, 3])
    np.testing.assert_array_equal(poly.offsets, [0, 3, 6])
    np.testing.assert_array_equal(poly.element_types, [5, 5])


def test_a_cells_row_shorter_than_the_width_it_claims_is_refused(
    tmp_path,
) -> None:
    """The width is a number out of the file. Trusted twice over, a short row
    used to take the indices it had while the offsets advanced by the width
    it claimed, which cuts every cell after it out of the wrong place."""
    path = tmp_path / "short_row.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 double\n0 0 0\n1 0 0\n0 1 0\n"
        "CELLS 1 4\n3 0 1\nCELL_TYPES 1\n5\n"
    )

    with pytest.raises(CodecError, match="does not list"):
        read(path)


def test_a_cells_row_declaring_a_width_past_a_machine_int_is_refused(
    tmp_path,
) -> None:
    """The width is converted before it is checked, and the Cython walk held
    it in a C int: one past INT_MAX raised an OverflowError out of the
    conversion itself, which named neither the file nor the row and was not
    of a kind the caller's fallback caught."""
    path = tmp_path / "wide_row.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 double\n0 0 0\n1 0 0\n0 1 0\n"
        f"CELLS 1 4\n{2**31} 0 1\nCELL_TYPES 1\n5\n"
    )

    with pytest.raises(CodecError, match="does not list"):
        read(path)


def test_a_polydata_cell_shorter_than_the_width_it_claims_is_refused(
    tmp_path,
) -> None:
    """The twin of the CELLS row, in the section a POLYDATA file spells its
    cells in. Reading past the tokens raised a bare IndexError naming neither
    the file nor the cell."""
    path = tmp_path / "short_poly.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET POLYDATA\n"
        "POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\nPOLYGONS 1 4\n3 0 1\n"
    )

    with pytest.raises(CodecError, match="does not list"):
        read(path)


def test_a_binary_cells_block_shorter_than_its_cells_is_refused(
    tmp_path,
) -> None:
    """A v4.2 cell array is a width followed by that many indices, and both
    numbers come off the file; a width the block cannot hold used to reach
    numpy as an index out of bounds."""
    path = tmp_path / "short_block.vtk"
    head = (
        b"# vtk DataFile Version 3.0\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 1 double\n" + b"\x00" * 24 + b"\n"
    )
    # One cell declaring five vertices in a block of three numbers.
    cells = b"CELLS 1 3\n" + b"".join(n.to_bytes(4, "big") for n in (5, 0, 1))
    path.write_bytes(
        head + cells + b"\nCELL_TYPES 1\n" + (5).to_bytes(4, "big") + b"\n"
    )

    with pytest.raises(CodecError, match="too short to hold"):
        read(path)


def test_a_binary_cells_block_that_ends_early_is_refused(tmp_path) -> None:
    """The count is the only thing that says how many cells the block holds,
    and walking past its end reached numpy as an index out of bounds."""
    path = tmp_path / "early_end.vtk"
    head = (
        b"# vtk DataFile Version 3.0\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 1 double\n" + b"\x00" * 24 + b"\n"
    )
    cells = b"CELLS 2 4\n" + b"".join(n.to_bytes(4, "big") for n in (3, 0, 1, 0))
    path.write_bytes(
        head + cells + b"\nCELL_TYPES 1\n" + (5).to_bytes(4, "big") + b"\n"
    )

    with pytest.raises(CodecError, match="ends after 1"):
        read(path)


def test_a_cell_types_value_that_is_not_a_number_names_itself(tmp_path) -> None:
    """CELL_TYPES declaring more values than it lists runs into the keyword
    after it, which int() answers with a bare ValueError."""
    path = tmp_path / "bad_types.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 double\n0 0 0\n1 0 0\n0 1 0\n"
        "CELLS 1 4\n3 0 1 2\nCELL_TYPES 2\n5\nPOINT_DATA 3\n"
    )

    with pytest.raises(CodecError, match="CELL_TYPES"):
        read(path)


def test_structured_points_shorter_than_the_header_declares_are_refused(
    tmp_path,
) -> None:
    """A STRUCTURED_GRID lays its points out on the grid its header names, so
    a POINTS array short of that grid reshapes into nothing; it used to be a
    bare ValueError about a shape."""
    path = tmp_path / "short_points.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET STRUCTURED_GRID\n"
        "DIMENSIONS 2 2 1\nPOINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n"
    )

    with pytest.raises(CodecError, match="POINTS declares 4 points"):
        read(path)


def test_a_coordinate_array_that_is_not_numbers_names_itself(tmp_path) -> None:
    """A coordinate array declaring more values than it lists runs into the
    keyword after it, and float() names neither the array nor the file."""
    path = tmp_path / "bad_coords.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET RECTILINEAR_GRID\n"
        "DIMENSIONS 2 2 1\nX_COORDINATES 3 float\n0 1\n"
        "Y_COORDINATES 2 float\n0 1\nZ_COORDINATES 1 float\n0\n"
    )

    with pytest.raises(CodecError, match="X_COORDINATES"):
        read(path)


def test_a_field_dataset_array_the_file_runs_out_inside_is_dropped(
    tmp_path,
) -> None:
    """The header is the only thing that says where a field array ends, so
    one the file runs out inside has no shape the file spells; reshaped it
    was a bare ValueError, and kept flat it would be a different array under
    the same name."""
    path = tmp_path / "short_field.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET FIELD FieldData 2\n"
        "whole 1 2 double\n1 2\nshort 3 2 double\n1 2 3\n"
    )

    with pytest.warns(UserWarning, match="short"):
        back = read(path)

    assert "short" not in back.global_attrs
    np.testing.assert_allclose(back.global_attrs["whole"], [1.0, 2.0])


def test_a_free_text_title_is_not_read_as_the_section_it_names() -> None:
    """The title is line two and its author writes what they like. A reader
    scanning from the top of the file took one beginning with a keyword for
    that keyword's header, and refused a file every other reader opens."""
    for title in (
        b"Field data from run 3",
        b"Vertices of a cow",
        b"Points of interest",
    ):
        body = (
            b"# vtk DataFile Version 4.2\n" + title + b"\nASCII\nDATASET POLYDATA\n"
            b"POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\nPOLYGONS 1 4\n3 0 1 2\n"
        )
        poly = read(_write_tmp(body))
        assert poly.vertices.shape == (3, 3)
        assert len(poly.element_types) == 1


def test_a_header_carrying_a_blank_line_still_finds_where_the_mesh_begins() -> None:
    """The DATASET line is at no fixed index once the header holds a blank
    one, so it is found rather than counted."""
    body = (
        b"# vtk DataFile Version 4.2\nt\n\nASCII\n\nDATASET UNSTRUCTURED_GRID\n"
        b"POINTS 1 double\n0 0 0\nCELLS 1 2\n1 0\nCELL_TYPES 1\n1\n"
    )
    assert read(_write_tmp(body)).vertices.shape == (1, 3)


def test_a_field_header_counting_backwards_is_refused_in_either_encoding() -> None:
    """The two counts are multiplied into the length of a payload. A negative
    one walks the binary scan back off the front of the file, and reshapes to
    a dimension numpy infers rather than the one the header claimed."""
    ascii_body = (
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 1\narr 4 -20 int\n1 2 3 4\n"
        b"POINTS 1 double\n0 0 0\n"
    )
    binary_body = (
        b"# vtk DataFile Version 4.2\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 1\narr 4 -20 int\n"
        + np.array([1, 2, 3, 4], dtype=">i4").tobytes()
        + b"\nPOINTS 1 double\n"
        + np.array([[0.0, 0, 0]], dtype=">f8").tobytes()
        + b"\n"
    )
    for body in (ascii_body, binary_body):
        with pytest.raises(CodecError, match="a count no array has"):
            read(_write_tmp(body))


def test_a_field_block_naming_an_array_twice_keeps_the_first() -> None:
    """A mesh's metadata holds one value per name, and the XML family answers
    a name spelled twice with the first of them. The same here, so one file
    does not read back two ways depending on which spelling of the format it
    was written in."""
    bodies = (
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 2\na 1 1 int\n1\na 1 1 int\n2\n"
        b"POINTS 1 double\n0 0 0\n",
        b"# vtk DataFile Version 4.2\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n"
        b"FIELD FieldData 2\na 1 1 int\n"
        + np.array([1], dtype=">i4").tobytes()
        + b"\na 1 1 int\n"
        + np.array([2], dtype=">i4").tobytes()
        + b"\nPOINTS 1 double\n"
        + np.array([[0.0, 0, 0]], dtype=">f8").tobytes()
        + b"\n",
        b"# vtk DataFile Version 4.2\nt\nASCII\nDATASET RECTILINEAR_GRID\n"
        b"DIMENSIONS 2 1 1\nFIELD FieldData 2\na 1 1 int\n1\na 1 1 int\n2\n"
        b"X_COORDINATES 2 float\n0 1\n"
        b"Y_COORDINATES 1 float\n0\nZ_COORDINATES 1 float\n0\n",
    )
    for body in bodies:
        with pytest.warns(UserWarning, match="more than once"):
            poly = read(_write_tmp(body))
        assert poly.global_attrs["a"].tolist() == [1]


def test_a_field_dataset_string_array_of_no_tuples_is_stepped_over(tmp_path) -> None:
    """A string array's lines are counted from its header, so one declaring
    none never reached a line to step over and read the same one for ever."""
    path = tmp_path / "empty_string.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET FIELD FieldData 2\n"
        "label 1 0 string\nkept 1 1 double\n7\n"
    )

    with pytest.warns(UserWarning, match="no geometry"):
        back = read(path)

    assert "label" not in back.global_attrs
    np.testing.assert_allclose(back.global_attrs["kept"], [7.0])


def test_a_field_dataset_string_array_ends_where_its_count_says(tmp_path) -> None:
    """Its data lines were counted from the header itself, so the last of
    them was left behind to be read as the header of an array of its own."""
    path = tmp_path / "string_rows.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET FIELD FieldData 2\n"
        "label 1 1 string\nnot 1 1 double\nkept 1 1 double\n7\n"
    )

    with pytest.warns(UserWarning, match="no geometry"):
        back = read(path)

    assert "not" not in back.global_attrs
    np.testing.assert_allclose(back.global_attrs["kept"], [7.0])


def test_a_field_dataset_array_of_a_negative_count_is_skipped(tmp_path) -> None:
    """Neither count is a length: they slice the values read from the wrong
    end and reshape to a dimension numpy infers rather than the declared one,
    so the array came back empty under a name the file never spelled."""
    path = tmp_path / "negative_field.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\nt\nASCII\nDATASET FIELD FieldData 2\n"
        "backwards 1 -3 double\n1 2\nkept 1 1 double\n7\n"
    )

    with pytest.warns(UserWarning, match="no geometry"):
        back = read(path)

    assert "backwards" not in back.global_attrs
    np.testing.assert_allclose(back.global_attrs["kept"], [7.0])


def test_a_connectivity_index_past_int32_is_refused(tmp_path) -> None:
    """Legacy 4.2 spells its connectivity in 32-bit ints, so a wider index
    would be written truncated - a silently different mesh."""
    verts = np.zeros((4, 3), dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    big = dataclasses.replace(
        poly,
        connectivity=np.array([0, 1, 2**31], dtype=np.int64),
        offsets=np.array([0, 3], dtype=np.int64),
    )
    with pytest.raises(IndexOverflowError):
        write(big, tmp_path / "big.vtk", vtk_version="4.2")


def test_an_unknown_cell_type_is_named_not_an_index_error(
    tmp_path,
) -> None:
    """VTK type 99 is not in the table; looking it up must not index past it."""
    text = (
        "# vtk DataFile Version 4.2\n"
        "Test mesh\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 float\n"
        "0 0 0\n1 0 0\n0 1 0\n"
        "CELLS 1 4\n"
        "3 0 1 2\n"
        "CELL_TYPES 1\n"
        "99\n"
    )
    out = tmp_path / "unknown.vtk"
    out.write_text(text)
    with pytest.raises(UnknownElementTypeError):
        read(out)


@pytest.mark.parametrize(
    ("code", "name", "n_nodes"),
    [
        (68, "lagrange_curve", 2),
        (69, "lagrange_triangle", 3),
        (70, "lagrange_quadrilateral", 4),
        (71, "lagrange_tetrahedron", 4),
        (72, "lagrange_hexahedron", 8),
        (73, "lagrange_wedge", 6),
        (74, "lagrange_pyramid", 5),
    ],
)
def test_a_lagrange_cell_type_is_read_rather_than_refused(
    tmp_path, code: int, name: str, n_nodes: int
) -> None:
    """The high-order codes are real VTK types; the table has to carry them.

    A Lagrange cell is arbitrary-order, so its node count is a property of the
    cell rather than of the type. Each case here uses the linear count, the
    smallest one the shape admits.
    """
    nodes = " ".join(str(i) for i in range(n_nodes))
    text = (
        "# vtk DataFile Version 4.2\n"
        "Lagrange\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 8 float\n"
        "0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
        "0 0 1\n1 0 1\n1 1 1\n0 1 1\n"
        f"CELLS 1 {n_nodes + 1}\n"
        f"{n_nodes} {nodes}\n"
        "CELL_TYPES 1\n"
        f"{code}\n"
    )
    out = tmp_path / "lagrange.vtk"
    out.write_text(text)

    poly = read(out)
    assert ELEMENT_TYPES_INV[int(poly.element_types[0])] == name

    # The reverse map is what a write leans on, so send the type back out.
    back_out = tmp_path / "lagrange_again.vtk"
    write(poly, back_out, vtk_version="4.2")
    assert f"\n{code}\n" in back_out.read_text()
    back = read(back_out)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_a_rank_two_element_attr_is_written_as_tensors(tmp_path) -> None:
    """A 3x3 per element is a TENSORS array; a FIELD block reads back flat."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    stress = np.eye(3)[np.newaxis, :, :].repeat(2, axis=0)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"stress": stress},
    )
    out = tmp_path / "tensor.vtk"
    write(poly, out)
    content = out.read_text()
    assert "TENSORS" in content
    assert "FIELD FieldData" not in content

    back = read(out)
    assert back.element_attrs["stress"].shape == (2, 3, 3)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("n_comp", [5, 7, 9])
def test_a_tuple_too_wide_for_scalars_travels_as_a_field_array(
    tmp_path, binary: bool, n_comp: int
) -> None:
    """A legacy SCALARS header names at most four components.

    Five is the first width past it, and a reader that holds the format to
    its own range refuses a SCALARS wider than four outright. A FIELD array
    carries its component count in its own header and so has no ceiling.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    wide = np.arange(2.0 * n_comp).reshape(2, n_comp)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"wide": wide},
    )
    out = tmp_path / "wide.vtk"
    write(poly, out, binary=binary)

    written = out.read_bytes()
    assert b"SCALARS wide" not in written
    # The header itself, not only the absence of the other one: a FIELD array
    # whose counts are wrong is the same unreadable file by another route.
    assert f"wide {n_comp} 2 double".encode() in written
    np.testing.assert_allclose(read(out).element_attrs["wide"], wide)


def test_a_tuple_of_no_components_is_dropped_before_its_header(tmp_path) -> None:
    """Zero components is as far outside a SCALARS header's range as nine.

    The drop happens before the section header is written, so a section left
    with nothing to carry is never declared - a POINT_DATA over no arrays is
    a promise the file does not keep.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"empty": np.zeros((2, 0)), "kept": np.arange(2.0)},
        vertex_attrs={"empty": np.zeros((4, 0))},
    )
    out = tmp_path / "empty.vtk"
    with pytest.warns(UserWarning, match="no components"):
        write(poly, out)

    content = out.read_text()
    assert "double 0" not in content
    assert "POINT_DATA" not in content
    assert "SCALARS kept double 1" in content
    np.testing.assert_allclose(read(out).element_attrs["kept"], [0.0, 1.0])


@pytest.mark.parametrize("n_comp", [1, 2, 3, 4])
def test_a_tuple_scalars_can_hold_keeps_its_scalars_spelling(
    tmp_path, n_comp: int
) -> None:
    """Four components or fewer stay SCALARS or VECTORS - the narrow spellings.

    Sending those through a FIELD array too would cost every reader the
    lookup table a SCALARS section carries.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    values = np.arange(2.0 * n_comp).reshape(2, n_comp)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"narrow": values},
    )
    out = tmp_path / "narrow.vtk"
    write(poly, out)
    content = out.read_text()

    expected = "VECTORS narrow" if n_comp == 3 else f"SCALARS narrow double {n_comp}"
    assert expected in content
    # Scoped to the section: the dataset's own metadata is a FIELD block too,
    # and this says nothing about that one.
    assert "FIELD" not in content.split("CELL_DATA")[1]
    # A single component comes home flat: the format has no trailing 1-axis.
    np.testing.assert_allclose(
        read(out).element_attrs["narrow"].ravel(), values.ravel()
    )


@pytest.mark.parametrize("vtk_version", ["4.2", "5.1"])
def test_a_binary_block_is_closed_by_the_newline_the_format_expects(
    tmp_path, vtk_version: str
) -> None:
    """A binary block ends on a newline, and the next keyword line follows it.

    Without one the terminator a reader finds is the first 0x0a inside the
    block itself, so the keyword line it reads starts in the middle of the
    numbers: a reader that holds the format to this refuses the file, where
    this one recovered by looking for the keyword anyway.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"s": np.arange(2.0)},
        vertex_attrs={"v": np.arange(12.0).reshape(4, 3)},
    )
    out = tmp_path / "b.vtk"
    write(poly, out, binary=True, vtk_version=vtk_version)
    written = out.read_bytes()

    for keyword in (b"CELL_TYPES 2", b"POINT_DATA 4", b"CELL_DATA 2"):
        assert b"\n" + keyword + b"\n" in written, keyword
    assert written.endswith(b"\n")
    np.testing.assert_allclose(read(out).vertex_attrs["v"], poly.vertex_attrs["v"])


@pytest.mark.parametrize("vtk_version", ["4.2", "5.1"])
def test_a_binary_file_written_without_those_newlines_still_reads(
    tmp_path, vtk_version: str
) -> None:
    """Files this writer already wrote have no terminator; they must still read.

    The reader steps over the newline when there is one rather than requiring
    it, so the fix to the writer costs nothing already on disk.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"s": np.arange(2.0)},
    )
    out = tmp_path / "old.vtk"
    write(poly, out, binary=True, vtk_version=vtk_version)
    # What the writer produced before it closed its blocks: the same bytes
    # with the terminator after each payload taken back out. Only the
    # keywords that follow a payload - a v5.1 OFFSETS follows the CELLS
    # header line, whose newline is the header's own.
    after_payload = [b"CELLS ", b"CELL_TYPES ", b"CELL_DATA "]
    if vtk_version == "5.1":
        after_payload.append(b"CONNECTIVITY ")
    stripped = out.read_bytes()
    for keyword in after_payload:
        stripped = stripped.replace(b"\n" + keyword, keyword)
    old = tmp_path / "stripped.vtk"
    old.write_bytes(stripped)

    back = read(old)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_allclose(back.element_attrs["s"], [0.0, 1.0])


def test_a_symmetric_tensor_keeps_its_shear_terms_where_vtk_puts_them(
    tmp_path,
) -> None:
    """Six components are XX, YY, ZZ, XY, YZ, XZ - VTK's order, not Voigt's.

    Classical Voigt runs its last three the other way, and reading them that
    way puts XZ where YZ belongs. Nothing downstream can tell the result from
    a tensor that was always that way, so the order is pinned here.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    sym = np.array([[10.0, 20, 30, 40, 50, 60], [1.0, 2, 3, 4, 5, 6]])
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"stress": sym},
    )
    out = tmp_path / "sym.vtk"
    write(poly, out)
    assert "TENSORS stress double" in out.read_text()

    back = read(out).element_attrs["stress"]
    np.testing.assert_allclose(back[0], [[10, 40, 60], [40, 20, 50], [60, 50, 30]])
    np.testing.assert_allclose(back[1], [[1, 4, 6], [4, 2, 5], [6, 5, 3]])


@pytest.mark.parametrize("binary", [False, True])
def test_a_tensors6_section_is_read_as_the_six_it_holds(tmp_path, binary: bool) -> None:
    """VTK 9 spells a symmetric tensor TENSORS6, which carries six not nine.

    Read as a TENSORS the header would be taken for nine, which in a binary
    file reads six values plus whatever follows them and leaves the scan
    standing in the middle of the next array.
    """
    payload = (
        np.array([[10.0, 20, 30, 40, 50, 60], [1.0, 2, 3, 4, 5, 6]])
        .astype(">f8")
        .tobytes()
        if binary
        else b"10 20 30 40 50 60\n1 2 3 4 5 6\n"
    )
    after = np.array([7.0, 8.0]).astype(">f8").tobytes() if binary else b"7 8\n"
    head = (
        f"# vtk DataFile Version 4.2\nsym\n{'BINARY' if binary else 'ASCII'}\n"
        "DATASET UNSTRUCTURED_GRID\nPOINTS 4 double\n"
    ).encode()
    points = (
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(">f8").tobytes()
        + b"\n"
        if binary
        else b"0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
    )
    cells = (
        np.array([3, 0, 1, 2, 3, 0, 1, 3], dtype=">i4").tobytes() + b"\n"
        if binary
        else b"3 0 1 2\n3 0 1 3\n"
    )
    types = np.array([5, 5], dtype=">i4").tobytes() + b"\n" if binary else b"5 5\n"
    out = tmp_path / "t6.vtk"
    out.write_bytes(
        head
        + points
        + b"CELLS 2 8\n"
        + cells
        + b"CELL_TYPES 2\n"
        + types
        + b"CELL_DATA 2\nTENSORS6 sym double\n"
        + payload
        + (b"\n" if binary else b"")
        + b"SCALARS after double 1\nLOOKUP_TABLE default\n"
        + after
    )

    back = read(out)
    np.testing.assert_allclose(
        back.element_attrs["sym"][0], [[10, 40, 60], [40, 20, 50], [60, 50, 30]]
    )
    # The array behind it survives: a tuple counted wrong takes everything
    # after it down with the scan.
    np.testing.assert_allclose(back.element_attrs["after"], [7.0, 8.0])


@pytest.mark.parametrize("vtk_version", ["4.2", "5.1"])
@pytest.mark.parametrize("binary", [False, True])
def test_another_reader_takes_what_this_writer_writes(
    tmp_path, binary: bool, vtk_version: str
) -> None:
    """A file only this reader takes back is a file that failed to be written.

    Both of the things this guards - a binary block closed by its newline and
    a wide tuple spelled as a FIELD array - read here whether or not they are
    right, because this reader recovers from either. A reader that holds the
    format to its own rules is what tells the two apart.
    """
    meshio = pytest.importorskip("meshio")
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"wide": np.arange(18.0).reshape(2, 9), "s": np.arange(2.0)},
        vertex_attrs={"v": np.arange(12.0).reshape(4, 3)},
        global_attrs={"time": np.array([2.5])},
    )
    out = tmp_path / "interop.vtk"
    write(poly, out, binary=binary, vtk_version=vtk_version)

    mesh = meshio.read(out)
    np.testing.assert_allclose(mesh.points, verts)
    np.testing.assert_allclose(mesh.cell_data["wide"][0], poly.element_attrs["wide"])
    np.testing.assert_allclose(mesh.point_data["v"], poly.vertex_attrs["v"])
