from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios.codecs._vtr import read, write
from polyxios.exceptions import LazyReadError


def _synthetic_rectilinear() -> object:
    """Create a simple 2x2x2 rectilinear grid PolyData from VTR round-trip."""
    # Build via write+read since VTR is structured
    vtr_content = """<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
  <RectilinearGrid WholeExtent="0 2 0 2 0 2">
    <Piece Extent="0 2 0 2 0 2">
      <Coordinates>
        <DataArray type="Float64" Name="x_coordinates" format="ascii">0.0 1.0 2.0</DataArray>
        <DataArray type="Float64" Name="y_coordinates" format="ascii">0.0 1.0 2.0</DataArray>
        <DataArray type="Float64" Name="z_coordinates" format="ascii">0.0 1.0 2.0</DataArray>
      </Coordinates>
    </Piece>
  </RectilinearGrid>
</VTKFile>"""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".vtr", mode="w", delete=False) as f:
        f.write(vtr_content)
        tmp = f.name
    return read(tmp), tmp


def test_roundtrip_ascii() -> None:
    poly, _ = _synthetic_rectilinear()
    assert poly.vertices.shape[0] == 27  # 3x3x3 grid points
    assert len(poly.element_types) == 8  # 2x2x2 hexahedra

    with tempfile.NamedTemporaryFile(suffix=".vtr", delete=False) as f:
        tmp2 = f.name
    write(poly, tmp2, binary=False)
    poly2 = read(tmp2)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == len(poly.element_types)


def test_vertex_attrs() -> None:
    vtr_content = """<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
  <RectilinearGrid WholeExtent="0 1 0 1 0 1">
    <Piece Extent="0 1 0 1 0 1">
      <Coordinates>
        <DataArray type="Float64" Name="x_coordinates" format="ascii">0.0 1.0</DataArray>
        <DataArray type="Float64" Name="y_coordinates" format="ascii">0.0 1.0</DataArray>
        <DataArray type="Float64" Name="z_coordinates" format="ascii">0.0 1.0</DataArray>
      </Coordinates>
      <PointData>
        <DataArray type="Float64" Name="pressure" format="ascii">
          1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
        </DataArray>
      </PointData>
    </Piece>
  </RectilinearGrid>
</VTKFile>"""

    with tempfile.NamedTemporaryFile(suffix=".vtr", mode="w", delete=False) as f:
        f.write(vtr_content)
        tmp = f.name
    poly = read(tmp)
    assert "pressure" in poly.vertex_attrs
    assert len(poly.vertex_attrs["pressure"]) == 8


def test_element_attrs() -> None:
    vtr_content = """<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
  <RectilinearGrid WholeExtent="0 1 0 1 0 1">
    <Piece Extent="0 1 0 1 0 1">
      <Coordinates>
        <DataArray type="Float64" Name="x_coordinates" format="ascii">0.0 1.0</DataArray>
        <DataArray type="Float64" Name="y_coordinates" format="ascii">0.0 1.0</DataArray>
        <DataArray type="Float64" Name="z_coordinates" format="ascii">0.0 1.0</DataArray>
      </Coordinates>
      <CellData>
        <DataArray type="Float64" Name="velocity" format="ascii">42.0</DataArray>
      </CellData>
    </Piece>
  </RectilinearGrid>
</VTKFile>"""

    with tempfile.NamedTemporaryFile(suffix=".vtr", mode="w", delete=False) as f:
        f.write(vtr_content)
        tmp = f.name
    poly = read(tmp)
    assert "velocity" in poly.element_attrs
    assert len(poly.element_attrs["velocity"]) == 1  # 1x1x1 = 1 cell


def test_unsupported_lazy() -> None:
    _, tmp = _synthetic_rectilinear()
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)
