import tempfile
import warnings

import numpy as np

from polyxios.codecs._mesh import read as read_mesh
from polyxios.codecs._ply import read as read_ply
from polyxios.codecs._vtu import read as read_vtu

# Smallest complete .mesh models (ASCII)
BEAM_TRI_MESH = b"""MFEM mesh v1.0

dimension
2

elements
16
1 2 10 0 1
1 2 0 10 9
1 2 11 1 2
1 2 1 11 10
1 2 12 2 3
1 2 2 12 11
1 2 13 3 4
1 2 3 13 12
2 2 14 4 5
2 2 4 14 13
2 2 15 5 6
2 2 5 15 14
2 2 16 6 7
2 2 6 16 15
2 2 17 7 8
2 2 7 17 16

boundary
18
3 1 0 1
3 1 1 2
3 1 2 3
3 1 3 4
3 1 4 5
3 1 5 6
3 1 6 7
3 1 7 8
3 1 10 9
3 1 11 10
3 1 12 11
3 1 13 12
3 1 14 13
3 1 15 14
3 1 16 15
3 1 17 16
1 1 9 0
2 1 8 17

vertices
18
2
0 0
1 0
2 0
3 0
4 0
5 0
6 0
7 0
8 0
0 1
1 1
2 1
3 1
4 1
5 1
6 1
7 1
8 1
"""

BEAM_HEX_NURBS_MESH = b"""MFEM NURBS mesh v1.0

dimension
3

elements
2
1 5 0 1 4 5 6 7 10 11
2 5 1 2 3 4 7 8 9 10

boundary
10
3 3 5 4 1 0
3 3 0 1 7 6
3 3 4 5 11 10
1 3 5 0 6 11
3 3 6 7 10 11
3 3 4 3 2 1
3 3 1 2 8 7
2 3 2 3 9 8
3 3 3 4 10 9
3 3 7 8 9 10

edges
20
0 0 1
0 5 4
0 6 7
0 11 10
1 1 2
1 4 3
1 7 8
1 10 9
2 0 5
2 1 4
2 2 3
2 6 11
2 7 10
2 8 9
3 0 6
3 1 7
3 2 8
3 3 9
3 4 10
3 5 11

vertices
12

knotvectors
4
1 5 0 0 1 2 3 4 4
1 5 0 0 1 2 3 4 4
1 2 0 0 1 1
1 2 0 0 1 1

weights
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1

FiniteElementSpace
FiniteElementCollection: NURBS1
VDim: 3
Ordering: 1

0 0 0
4 0 0
8 0 0
8 1 0
4 1 0
0 1 0
0 0 1
4 0 1
8 0 1
8 1 1
4 1 1
0 1 1
1 0 0
2 0 0
3 0 0
3 1 0
2 1 0
1 1 0
1 0 1
2 0 1
3 0 1
3 1 1
2 1 1
1 1 1
5 0 0
6 0 0
7 0 0
7 1 0
6 1 0
5 1 0
5 0 1
6 0 1
7 0 1
7 1 1
6 1 1
5 1 1
"""

# Smallest complete .vtu model (XML ASCII)
QUADRATIC_TETRA_VTU = b"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="22" NumberOfCells="3">
      <PointData Scalars="scalars">
        <DataArray type="Float32" Name="scalars" format="ascii">
          1 1 1 1 0 0 0 0 0 0
          1 1 1 0 0 0
          1 1 1 0 0 0
        </DataArray>
      </PointData>
      <CellData>
      </CellData>
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          0 0 0
          1 0 0
          0.5 0.8 0
          0.5 0.4 1
          0.5 0 -0.2
          0.6 0.6 0
          0.3 0.4 0
          0.4 0.2 0.5
          0.85 0.3 0.5
          0.5 0.6 0.45

          2 -0.5 0
          3 0 0
          2 0.5 0
          2.5 -0.25 0
          2.5 0.25 0
          1.75 0 0

          1 -0.5 -1
          1 0 -2
          1 0.5 -1
          1 -0.25 -1.5
          1 0.25 -1.5
          1 0 -0.75

        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2 3 4 5 6 7 8 9
          10 11 12 13 14 15
          16 17 18 19 20 21
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          10 16 22
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          24 22 22
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

# Smallest complete .ply model (ASCII)
TET_PLY = b"""ply
format ascii 1.0
comment single tetrahedron with colored faces
element vertex 4
comment tetrahedron vertices
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
property uchar red
property uchar green
property uchar blue
end_header
0 0 0
0 1 1
1 0 1
1 1 0
3 0 1 2 255 255 255
3 0 2 3 255 0 0
3 0 1 3 0 255 0
3 1 2 3 0 0 255
"""


def test_inline_mesh_beam_tri() -> None:
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        f.write(BEAM_TRI_MESH)
        tmp = f.name
    poly = read_mesh(tmp)
    assert len(poly.vertices) == 18
    assert len(poly.element_types) == 16
    assert poly.vertices.shape[1] == 3  # 3D vertices (padded with 0 for z)
    assert poly.vertices.dtype == np.float64


def test_inline_mesh_beam_hex_nurbs() -> None:
    with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
        f.write(BEAM_HEX_NURBS_MESH)
        tmp = f.name
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        poly = read_mesh(tmp)
    assert len(poly.vertices) == 12
    assert len(poly.element_types) == 2
    assert "mfem_nurbs_knotvectors" in poly.global_attrs


def test_inline_vtu_quadratic_tetra() -> None:
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        f.write(QUADRATIC_TETRA_VTU)
        tmp = f.name
    poly = read_vtu(tmp)
    assert len(poly.vertices) == 22
    assert len(poly.element_types) == 3
    assert "scalars" in poly.vertex_attrs


def test_inline_ply_tetra() -> None:
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        f.write(TET_PLY)
        tmp = f.name
    poly = read_ply(tmp)
    assert len(poly.vertices) == 4
    assert len(poly.element_types) == 4
    assert "red" in poly.element_attrs
    assert "green" in poly.element_attrs
    assert "blue" in poly.element_attrs


def test_get_package_name() -> None:
    from polyxios.fetcher import get_package_name

    assert get_package_name("inp") == "abaqus"
    assert get_package_name(".inp") == "abaqus"
    assert get_package_name("xml") == "dolfin"
    assert get_package_name(".meshb") == "medit"
    assert get_package_name("obj") == "obj"
