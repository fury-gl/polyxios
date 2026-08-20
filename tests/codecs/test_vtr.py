from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._vtr import read, write
from polyxios.exceptions import CodecError, LazyReadError


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


def _grid() -> object:
    xs = np.array([0.0, 1.0, 2.0])
    verts = np.stack(
        [g.ravel() for g in np.meshgrid(xs, xs, xs, indexing="ij")], axis=1
    )
    quads = np.array([[0, 1, 4, 3]], dtype=np.int32)
    return make_polydata(verts, [("quad", quads)])


@pytest.mark.parametrize("binary", [False, True])
def test_a_vector_attribute_keeps_its_components(tmp_path, binary: bool) -> None:
    """Without NumberOfComponents a reader has no way to cut the flat run."""
    poly = _grid()
    poly.vertex_attrs["vector"] = np.arange(27 * 3, dtype=np.float64).reshape(27, 3)
    path = tmp_path / "grid.vtr"

    write(poly, path, binary=binary)
    back = read(path)

    assert back.vertex_attrs["vector"].shape == (27, 3)
    np.testing.assert_allclose(back.vertex_attrs["vector"], poly.vertex_attrs["vector"])


def test_a_binary_integer_attribute_is_written_as_the_type_it_declares(
    tmp_path,
) -> None:
    """It was cast to float64 under an Int32 header and read back as noise."""
    poly = _grid()
    poly.vertex_attrs["ints"] = np.arange(27, dtype=np.int32)
    path = tmp_path / "grid.vtr"

    write(poly, path, binary=True)
    back = read(path)

    np.testing.assert_array_equal(back.vertex_attrs["ints"], np.arange(27))


@pytest.mark.parametrize("binary", [False, True])
def test_a_dtype_no_vtk_type_names_is_written_as_the_one_declared(
    tmp_path, binary: bool
) -> None:
    """Only the header fell back to Float64; the bytes stayed booleans."""
    poly = _grid()
    poly.vertex_attrs["mask"] = np.arange(27) % 2 == 0
    path = tmp_path / "grid.vtr"

    write(poly, path, binary=binary)
    back = read(path)

    np.testing.assert_array_equal(
        back.vertex_attrs["mask"], poly.vertex_attrs["mask"].astype(np.float64)
    )


@pytest.mark.parametrize("binary", [False, True])
def test_a_tensor_declares_every_component_it_holds(tmp_path, binary: bool) -> None:
    """shape[1] of an (n, 3, 3) array is three, and the tuple is nine."""
    poly = _grid()
    tensor = np.arange(27 * 9, dtype=np.float64).reshape(27, 3, 3)
    poly.vertex_attrs["tensor"] = tensor
    path = tmp_path / "grid.vtr"

    write(poly, path, binary=binary)
    back = read(path)

    assert back.vertex_attrs["tensor"].shape == (27, 9)
    np.testing.assert_array_equal(back.vertex_attrs["tensor"], tensor.reshape(27, 9))


def test_an_attribute_that_covers_no_mesh_is_dropped(tmp_path) -> None:
    """It used to reach PolyData and fail validate with a length message."""
    path = tmp_path / "short.vtr"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="0 1 0 1 0 1">\n'
        '    <Piece Extent="0 1 0 1 0 1">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" Name="x" format="ascii">0 1</DataArray>\n'
        '        <DataArray type="Float64" Name="y" format="ascii">0 1</DataArray>\n'
        '        <DataArray type="Float64" Name="z" format="ascii">0 1</DataArray>\n'
        "      </Coordinates>\n"
        "      <PointData>\n"
        '        <DataArray type="Float64" Name="half" format="ascii">1 2 3</DataArray>\n'
        "      </PointData>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )

    with pytest.warns(UserWarning, match="covers 3 of 8 points"):
        back = read(path)

    assert "half" not in back.vertex_attrs


@pytest.mark.parametrize(
    ("extent", "match"),
    [
        ("0 1 0 1", "holds 4 indices"),
        ("a b c d e f", "not a run of whole numbers"),
    ],
)
def test_an_extent_that_is_not_six_numbers_names_the_file(
    tmp_path, extent: str, match: str
) -> None:
    """Unpacked into six names it failed with a message about unpacking."""
    path = tmp_path / "bad.vtr"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        ' <RectilinearGrid WholeExtent="0 1 0 1 0 0">\n'
        f'  <Piece Extent="{extent}">\n'
        "   <Coordinates>\n"
        '    <DataArray type="Float64" format="ascii">0 1</DataArray>\n'
        '    <DataArray type="Float64" format="ascii">0 1</DataArray>\n'
        '    <DataArray type="Float64" format="ascii">0</DataArray>\n'
        "   </Coordinates>\n"
        "  </Piece>\n"
        " </RectilinearGrid>\n"
        "</VTKFile>\n"
    )

    with pytest.raises(CodecError, match=match):
        read(path)
