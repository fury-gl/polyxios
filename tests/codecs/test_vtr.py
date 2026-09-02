from __future__ import annotations

from dataclasses import replace
import tempfile

import numpy as np
import pytest

from polyxios import PolyData, make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios.codecs._vtk_xml import structured_cells
from polyxios.codecs._vtr import read, write
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.transforms import remove_orphan_vertices
from polyxios.validate import validate


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
    """A 3x3x3 grid in the order .vtr reads one back: x fastest, z slowest.

    Built the other way round the points are the same cube, but every point
    attribute lands on the point mirrored through it - which is what the
    writer now refuses rather than quietly writes.
    """
    xs = np.array([0.0, 1.0, 2.0])
    zz, yy, xx = np.meshgrid(xs, xs, xs, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    # The grid's own cells, since the file carries no connectivity: the reader
    # rebuilds them from the extent, so a mesh holding any others is not the
    # one that comes back.
    hexes = structured_cells(2, 2, 2)[0].reshape(-1, 8)
    return make_polydata(verts, [("hexahedron", hexes)])


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


def test_a_flat_extent_is_a_sheet_of_quads_with_its_cell_data(tmp_path) -> None:
    """A grid one point deep held no cells, so its CellData was dropped."""
    path = tmp_path / "flat.vtr"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        ' <RectilinearGrid WholeExtent="0 2 0 2 0 0">\n'
        '  <Piece Extent="0 2 0 2 0 0">\n'
        "   <Coordinates>\n"
        '    <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        '    <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        '    <DataArray type="Float64" format="ascii">0</DataArray>\n'
        "   </Coordinates>\n"
        '   <CellData><DataArray type="Float64" Name="c"'
        ' format="ascii">1 2 3 4</DataArray></CellData>\n'
        "  </Piece>\n"
        " </RectilinearGrid>\n"
        "</VTKFile>\n"
    )

    poly = read(path)

    assert len(poly.element_types) == 4
    np.testing.assert_allclose(poly.element_attrs["c"], [1.0, 2.0, 3.0, 4.0])
    validate(poly)


def test_an_axis_that_runs_downwards_keeps_its_direction(tmp_path) -> None:
    """A coordinate array is written out in full, so nothing says it ascends.

    The axes used to be taken as the sorted distinct value on each column,
    which turned a descending axis round under its own point data.
    """
    xs = np.array([2.0, 1.0, 0.0])
    ys = np.array([0.0, 1.0, 2.0])
    verts = np.array([[i, j, k] for k in ys for j in ys for i in xs])
    poly = make_polydata(
        verts, [("hexahedron", structured_cells(2, 2, 2)[0].reshape(-1, 8))]
    )
    poly.vertex_attrs["scalar"] = np.arange(27, dtype=np.float64)

    path = tmp_path / "down.vtr"
    write(poly, path, binary=False)

    back = read(path)
    np.testing.assert_allclose(back.vertices, verts)
    np.testing.assert_allclose(back.vertex_attrs["scalar"], poly.vertex_attrs["scalar"])
    validate(back)


def test_cells_that_are_not_the_grids_own_are_refused(tmp_path) -> None:
    """The file holds three coordinate arrays and no cells at all."""
    poly = _grid()
    quad = make_polydata(poly.vertices, [("quad", np.array([[0, 1, 4, 3]]))])

    with pytest.raises(CodecError):
        write(quad, tmp_path / "quad.vtr")


def test_the_extent_a_grid_stood_on_is_written_back(tmp_path) -> None:
    """A block that did not begin at zero goes back at its own indices.

    Its place in the extent is what a .pvtr assembling it next to its
    neighbours reads; sliding it to the origin puts it on top of them.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="2 4 1 2 0 0">\n'
        '    <Piece Extent="2 4 1 2 0 0">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" format="ascii">10 20 30</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">5 6</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0</DataArray>\n'
        "      </Coordinates>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    source = tmp_path / "block.vtr"
    source.write_text(body)

    poly = read(source)
    assert poly.global_attrs["vtr_extents"] == [2, 4, 1, 2, 0, 0]

    out = tmp_path / "again.vtr"
    write(poly, out, binary=False)
    assert 'Extent="2 4 1 2 0 0"' in out.read_text()

    back = read(out)
    np.testing.assert_allclose(back.vertices, poly.vertices)
    assert back.global_attrs["vtr_extents"] == [2, 4, 1, 2, 0, 0]


def test_an_extent_a_transform_has_outgrown_is_re_derived(tmp_path) -> None:
    """The stored extent is checked against the mesh, not trusted.

    Pruning the grid to one of its cells leaves it counting points the file
    no longer holds, so the writer reads the shape off the cells instead.
    """
    poly = _grid()
    poly.global_attrs["vtr_extents"] = [5, 7, 5, 7, 5, 7]

    # The eight corners of the grid's first cell, in the order a lattice
    # numbers them: index i + 3j + 9k, so x fastest and z slowest.
    corners = poly.vertices[[0, 1, 3, 4, 9, 10, 12, 13]].copy()
    cell = make_polydata(
        corners, [("hexahedron", np.array([[0, 1, 3, 2, 4, 5, 7, 6]]))]
    )
    cell.global_attrs["vtr_extents"] = [5, 7, 5, 7, 5, 7]

    path = tmp_path / "pruned.vtr"
    write(cell, path, binary=False)
    assert 'Extent="0 1 0 1 0 1"' in path.read_text()
    validate(read(path))


def test_an_axis_whose_extent_ends_before_it_starts_holds_nothing(tmp_path) -> None:
    """VTK spells an empty grid with an end before its start.

    One such axis empties the whole grid - the point count is a product. The
    other two axes' cells used to be counted anyway, handing back four quads
    whose corners named points the file never spelled.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="0 -1 0 2 0 2">\n'
        '    <Piece Extent="0 -1 0 2 0 2">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" format="ascii"></DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        "      </Coordinates>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "empty_axis.vtr"
    path.write_text(body)

    poly = read(path)
    assert len(poly.vertices) == 0
    assert len(poly.element_types) == 0
    assert len(poly.connectivity) == 0
    validate(poly)


def test_a_coordinate_array_the_extent_does_not_count_names_its_axis(
    tmp_path,
) -> None:
    """The extent counts the planes, the arrays spell them, nothing joins them.

    A longer array expanded to more vertices than the extent declared, while
    the cells, the offsets and every PointData array were sized off the
    extent - a mesh whose connectivity covered part of itself and whose
    attributes covered none of it, past every check the reader made.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="0 1 0 1 0 0">\n'
        '    <Piece Extent="0 1 0 1 0 0">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" format="ascii">0 1 2 3 4</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0</DataArray>\n'
        "      </Coordinates>\n"
        "      <PointData>\n"
        '        <DataArray type="Float64" Name="s" format="ascii">1 2 3 4</DataArray>\n'
        "      </PointData>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "wide.vtr"
    path.write_text(body)

    with pytest.raises(CodecError, match="x axis"):
        read(path)


def test_a_piece_an_extent_empties_is_written_back_empty_on_every_axis(
    tmp_path,
) -> None:
    """An extent that empties one axis empties the piece, and the file says so.

    The stored extent still described the mesh - no points, and none declared -
    so it was kept, but the coordinates are read off the vertices a stride at a
    time and there is no vertex to stride through. The other two axes went out
    empty under an extent that still counted three planes on each, and this
    reader refused the file it had just written.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="0 -1 0 2 0 2">\n'
        '    <Piece Extent="0 -1 0 2 0 2">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" format="ascii"></DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        "      </Coordinates>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "empty_axis.vtr"
    path.write_text(body)
    poly = read(path)
    assert poly.global_attrs["vtr_extents"] == [0, -1, 0, 2, 0, 2]

    out = tmp_path / "back.vtr"
    write(poly, out)
    assert 'Extent="0 -1 0 -1 0 -1"' in out.read_text()

    again = read(out)
    assert len(again.vertices) == 0
    assert len(again.element_types) == 0
    assert again.global_attrs["vtr_extents"] == [0, -1, 0, -1, 0, -1]
    validate(again)


def test_a_coordinate_array_on_an_axis_the_extent_empties_is_named(tmp_path) -> None:
    """The check is asked of every axis, not only of a grid that holds points.

    An extent ending before it starts empties the grid - the point count is a
    product - but the vertices are built from the coordinate arrays, which are
    the file's own. Skipping the comparison there handed back nine vertices
    under an extent that declared none, with no cells and no attributes over
    them: the shape the check exists to refuse.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="0 -1 0 2 0 2">\n'
        '    <Piece Extent="0 -1 0 2 0 2">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" format="ascii">0</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1 2</DataArray>\n'
        "      </Coordinates>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "phantom.vtr"
    path.write_text(body)

    with pytest.raises(CodecError, match="0 planes on the x axis"):
        read(path)


def test_a_two_dimensional_grid_may_leave_out_its_third_axis(tmp_path) -> None:
    """A missing array is an axis the extent gives one plane at zero."""
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <RectilinearGrid WholeExtent="0 1 0 1 0 0">\n'
        '    <Piece Extent="0 1 0 1 0 0">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" format="ascii">0 1</DataArray>\n'
        '        <DataArray type="Float64" format="ascii">0 1</DataArray>\n'
        "      </Coordinates>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "flat.vtr"
    path.write_text(body)

    poly = read(path)
    assert len(poly.vertices) == 4
    assert np.all(poly.vertices[:, 2] == 0.0)
    validate(poly)


def test_an_extent_global_attrs_cannot_spell_is_re_derived(tmp_path) -> None:
    """Whatever ``global_attrs`` held may be no extent at all.

    A bare number has no length, a string that looks like one has eleven
    characters rather than six numbers, and either used to reach ``len`` and
    fail with a TypeError from inside the writer. There is nothing to keep,
    so the mesh's own extent is read off its cells.
    """
    conn, _, _ = structured_cells(1, 1, 1)
    verts = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    for junk in (5, "0 1 0 1 0 1", [0, 1], [0, "x", 0, 1, 0, 1]):
        poly = PolyData(
            vertices=verts,
            connectivity=conn,
            offsets=np.array([0, 8], dtype=np.int32),
            element_types=np.full(1, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
            global_attrs={"vtr_extents": junk},
        )
        path = tmp_path / "junk.vtr"
        write(poly, path)
        assert 'WholeExtent="0 1 0 1 0 1"' in path.read_text()
        assert np.allclose(read(path).vertices, verts)


def _piece_xml(whole: str = "0 9 0 4 0 0") -> str:
    """One piece of a larger grid, the shape a .pvtr assembles."""
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
        f'  <RectilinearGrid WholeExtent="{whole}">\n'
        '    <Piece Extent="6 9 2 4 0 0">\n'
        "      <Coordinates>\n"
        '        <DataArray type="Float64" Name="x" format="ascii">6 7 8 9</DataArray>\n'
        '        <DataArray type="Float64" Name="y" format="ascii">2 3 4</DataArray>\n'
        '        <DataArray type="Float64" Name="z" format="ascii">0</DataArray>\n'
        "      </Coordinates>\n"
        "    </Piece>\n"
        "  </RectilinearGrid>\n"
        "</VTKFile>\n"
    )


def test_the_grid_a_piece_belongs_to_travels_with_it(tmp_path) -> None:
    """WholeExtent is the one thing a .pvtr assembling the piece reads."""
    path = tmp_path / "piece.vtr"
    path.write_text(_piece_xml())
    piece = read(path)
    assert piece.global_attrs["vtr_whole_extent"] == [0, 9, 0, 4, 0, 0]

    out = tmp_path / "back.vtr"
    write(piece, out)
    body = out.read_text()
    # Written as the piece extent, the block went back out claiming to be the
    # whole domain - the one thing keeping the piece indices was to prevent.
    assert 'WholeExtent="0 9 0 4 0 0"' in body
    assert 'Extent="6 9 2 4 0 0"' in body


def test_a_pruned_piece_drops_the_grid_it_no_longer_stands_in(tmp_path) -> None:
    """A re-derived extent is zero-based and says nothing about the old grid."""
    path = tmp_path / "piece.vtr"
    path.write_text(_piece_xml())
    piece = read(path)

    one_cell = remove_orphan_vertices(
        replace(
            piece,
            connectivity=piece.connectivity[:4],
            offsets=np.array([0, 4]),
            element_types=piece.element_types[:1],
        )
    )
    out = tmp_path / "pruned.vtr"
    write(one_cell, out)
    body = out.read_text()
    assert 'Extent="0 1 0 1 0 0"' in body
    assert 'WholeExtent="0 1 0 1 0 0"' in body


def test_a_malformed_whole_extent_names_itself(tmp_path) -> None:
    """Unpacked straight into ints it failed with a literal and no file."""
    path = tmp_path / "bad.vtr"
    path.write_text(_piece_xml(whole="nonsense"))
    with pytest.raises(CodecError, match="WholeExtent"):
        read(path)


def test_a_bare_grid_is_readable_by_the_codec_that_wrote_it(tmp_path) -> None:
    """A RectilinearGrid spells three axes, not a point per vertex."""
    axis = np.arange(8.0)
    pts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    grid = PolyData(
        vertices=pts,
        connectivity=structured_cells(7, 7, 7)[0],
        offsets=np.arange(344) * 8,
        element_types=np.full(343, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
    )
    path = tmp_path / "bare.vtr"
    write(grid, path)

    # 512 points against 24 coordinates on disk: the header check weighed the
    # count against bytes that are only there when a format spells its points.
    assert np.array_equal(read(path).vertices, pts)
