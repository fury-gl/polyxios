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
    """v1.0 blank-line quirk: unsupported dataset shows real type, not 'BINARY'."""
    content = (
        b"# vtk DataFile Version 1.0\n"
        b"Some grid\n"
        b"\n"
        b"BINARY\n"
        b"\n"
        b"DATASET STRUCTURED_GRID\n"
        b"DIMENSIONS 2 2 2\n"
    )
    tmp = _write_tmp(content)
    with pytest.raises(CodecError, match="STRUCTURED_GRID"):
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
