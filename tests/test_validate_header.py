from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from polyxios.exceptions import (
    IndexOverflowError,
    UnknownElementTypeError,
    ValidationError,
)
from polyxios.validate import validate_header


def test_corrupt_large_n_verts() -> None:
    """Guard against meshio #1562: 100B vertices would need 2.4 TB."""
    with pytest.raises(ValidationError, match="MAX_SAFE_VERTICES"):
        validate_header(10**11, 0, 0, 10**6)


def test_n_verts_exceeds_file_size() -> None:
    """10M verts need ~240 MB but file is only 1 KB."""
    with pytest.raises(ValidationError, match="file_size"):
        validate_header(10**7, 0, 0, 1000)


def test_reasonable_header_passes() -> None:
    # 100 verts, 50 tris, 150 conn indices — well within any file
    validate_header(100, 50, 150, 10_000)


def test_int64_overflow_at_write() -> None:
    """Connectivity index > 2**31-1 must raise IndexOverflowError for v4.2."""
    import dataclasses

    from polyxios import make_polydata

    verts = np.zeros((4, 3), dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    # Manually set a large connectivity index
    big_conn = np.array([0, 1, 2**31], dtype=np.int64)
    big_off = np.array([0, 3], dtype=np.int64)
    big_poly = dataclasses.replace(poly, connectivity=big_conn, offsets=big_off)

    from polyxios.codecs._vtk import write as vtk_write

    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp_path = f.name

    with pytest.raises(IndexOverflowError):
        vtk_write(big_poly, tmp_path, vtk_version="4.2")


def test_unknown_vtk_type_raises() -> None:
    """VTK cell type 99 → UnknownElementTypeError, not IndexError."""
    import tempfile

    vtk_content = (
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
    with tempfile.NamedTemporaryFile(suffix=".vtk", mode="w", delete=False) as f:
        f.write(vtk_content)
        tmp_path = f.name

    from polyxios.codecs._vtk import read as vtk_read

    with pytest.raises(UnknownElementTypeError):
        vtk_read(tmp_path)


def test_vtk_version_42_roundtrip_paraview_compatible() -> None:
    """v4.2 output must contain CELLS keyword, not OFFSETS."""
    import tempfile

    from polyxios import make_polydata
    from polyxios.codecs._vtk import write as vtk_write

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])

    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        tmp_path = f.name

    vtk_write(poly, tmp_path, vtk_version="4.2")
    content = Path(tmp_path).read_text()
    assert "CELLS" in content
    assert "OFFSETS" not in content


def test_vtk_tensor_written_correctly() -> None:
    """element_attrs of shape (n, 3, 3) must emit TENSORS keyword."""
    import tempfile

    from polyxios import make_polydata
    from polyxios.codecs._vtk import read as vtk_read, write as vtk_write

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    stress = np.eye(3)[np.newaxis, :, :].repeat(2, axis=0)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_attrs={"stress": stress},
    )

    with tempfile.NamedTemporaryFile(suffix=".vtk", mode="wb", delete=False) as f:
        tmp_path = f.name

    vtk_write(poly, tmp_path)
    content = Path(tmp_path).read_text()
    assert "TENSORS" in content
    assert "FIELD FieldData" not in content

    poly2 = vtk_read(tmp_path)
    assert "stress" in poly2.element_attrs
    assert poly2.element_attrs["stress"].shape == (2, 3, 3)


def test_multi_group_element_tags() -> None:
    """Element 0 belongs to 'inlet' and 'wall' — both tags must survive roundtrip."""
    import tempfile

    from polyxios import make_polydata
    from polyxios.codecs._obj import read as obj_read, write as obj_write

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))],
        element_tags={
            "inlet": np.array([0], dtype=np.int32),
            "wall": np.array([0, 1], dtype=np.int32),
        },
    )

    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        tmp_path = f.name

    obj_write(poly, tmp_path)
    poly2 = obj_read(tmp_path)

    assert "inlet" in poly2.element_tags, f"Tags: {list(poly2.element_tags)}"
    assert "wall" in poly2.element_tags
    assert 0 in poly2.element_tags["inlet"]
    assert 0 in poly2.element_tags["wall"]
