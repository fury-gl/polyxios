from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios.codecs._vts import read, write
from polyxios.exceptions import LazyReadError
from polyxios.validate import validate


def _synthetic_vts() -> object:
    """Build and write a 2×2×1 StructuredGrid, return the PolyData."""
    # 2×2×1 grid: 3×3×2 = 18 vertices, 4 hex cells
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    z = np.linspace(0, 0.5, 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    pts_str = "\n".join(f"{r[0]:.6g} {r[1]:.6g} {r[2]:.6g}" for r in verts)

    with tempfile.NamedTemporaryFile(suffix=".vts", delete=False) as f:
        tmp = f.name

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        '  <StructuredGrid WholeExtent="0 2 0 2 0 1">\n'
        '    <Piece Extent="0 2 0 2 0 1">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        f"          {pts_str}\n"
        "        </DataArray>\n"
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )
    with open(tmp, "w") as fh:
        fh.write(xml)
    return read(tmp)


def test_read_basic() -> None:
    poly = _synthetic_vts()
    assert len(poly.vertices) == 18  # 3×3×2
    assert len(poly.element_types) == 4  # 2×2×1
    assert poly.vertices.dtype == np.float64


def test_roundtrip_ascii() -> None:
    poly = _synthetic_vts()
    with tempfile.NamedTemporaryFile(suffix=".vts", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    assert len(poly2.element_types) == len(poly.element_types)


def test_roundtrip_binary() -> None:
    poly = _synthetic_vts()
    with tempfile.NamedTemporaryFile(suffix=".vts", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == len(poly.element_types)


def test_lazy_raises() -> None:
    poly = _synthetic_vts()
    with tempfile.NamedTemporaryFile(suffix=".vts", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_vertex_attrs() -> None:
    poly = _synthetic_vts()
    from polyxios._types import PolyData

    pressure = np.arange(len(poly.vertices), dtype=np.float64)
    poly_attr = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
        vertex_attrs={"pressure": pressure},
        element_attrs={},
        global_attrs=poly.global_attrs,
    )
    with tempfile.NamedTemporaryFile(suffix=".vts", delete=False) as f:
        tmp = f.name
    write(poly_attr, tmp, binary=False)
    poly2 = read(tmp)
    assert "pressure" in poly2.vertex_attrs
    np.testing.assert_allclose(poly2.vertex_attrs["pressure"], pressure, atol=1e-6)


def test_a_flat_extent_is_a_sheet_of_quads_with_its_cell_data(tmp_path) -> None:
    """A grid one point deep held no cells, so its CellData was dropped."""
    path = tmp_path / "flat.vts"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        ' <StructuredGrid WholeExtent="0 2 0 2 0 0">\n'
        '  <Piece Extent="0 2 0 2 0 0">\n'
        '   <Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">0 0 0 1 0 0 2 0 0 0 1 0 1 1 0 2 1 0'
        " 0 2 0 1 2 0 2 2 0</DataArray></Points>\n"
        '   <CellData><DataArray type="Float64" Name="c"'
        ' format="ascii">1 2 3 4</DataArray></CellData>\n'
        "  </Piece>\n"
        " </StructuredGrid>\n"
        "</VTKFile>\n"
    )

    poly = read(path)

    assert len(poly.element_types) == 4
    np.testing.assert_allclose(poly.element_attrs["c"], [1.0, 2.0, 3.0, 4.0])
    validate(poly)
