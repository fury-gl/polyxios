from __future__ import annotations

from dataclasses import replace
import tempfile

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import structured_cells
from polyxios.codecs._vts import read, write
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.transforms import remove_orphan_vertices
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


def test_an_empty_mesh_round_trips_through_its_own_writer(tmp_path) -> None:
    """The empty extent it writes came back as a ValueError about newaxis."""
    path = tmp_path / "empty.vts"
    poly = PolyData(
        vertices=np.zeros((0, 3)),
        connectivity=np.zeros(0, dtype=np.int64),
        offsets=np.zeros(1, dtype=np.int64),
        element_types=np.zeros(0, dtype=np.uint8),
    )

    write(poly, path, binary=False)
    assert 'WholeExtent="0 -1 0 -1 0 -1"' in path.read_text()

    back = read(path)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0
    validate(back)


def test_a_stale_extent_is_re_derived_rather_than_trusted(tmp_path) -> None:
    """A transform leaves the stored extent describing the grid the mesh was."""
    axis = np.arange(3.0)
    pts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    conn: list[int] = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                b = k * 9 + j * 3 + i
                conn += [b, b + 1, b + 4, b + 3, b + 9, b + 10, b + 13, b + 12]
    grid = PolyData(
        vertices=pts,
        connectivity=np.array(conn),
        offsets=np.arange(9) * 8,
        element_types=np.full(8, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
    )
    path = tmp_path / "grid.vts"
    write(grid, path, binary=False)
    poly = read(path)
    assert poly.global_attrs["vts_extent"] == [0, 2, 0, 2, 0, 2]

    one_cell = remove_orphan_vertices(
        replace(
            poly,
            connectivity=poly.connectivity[:8],
            offsets=np.array([0, 8]),
            element_types=poly.element_types[:1],
        )
    )
    out = tmp_path / "pruned.vts"
    write(one_cell, out, binary=False)

    assert 'WholeExtent="0 1 0 1 0 1"' in out.read_text()
    back = read(out)
    assert back.vertices.shape == (8, 3)
    assert len(back.element_types) == 1
    validate(back)


def test_a_curvilinear_grid_is_written_as_the_grid_it_is(tmp_path) -> None:
    """A StructuredGrid writes its points, so they need not be a lattice.

    The extent was inferred from the distinct value on each coordinate, which
    for a warped block is every point on it: a 3x3x3 grid went out declaring
    a 27x27x27 one and its own reader refused the file.
    """
    axis = np.arange(3.0)
    verts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    verts[:, 0] += 0.3 * np.sin(verts[:, 2])
    verts[:, 1] += 0.2 * verts[:, 0] ** 2
    poly = make_polydata(
        verts, [("hexahedron", structured_cells(2, 2, 2)[0].reshape(-1, 8))]
    )
    poly.vertex_attrs["scalar"] = np.arange(27, dtype=np.float64)

    path = tmp_path / "warped.vts"
    write(poly, path, binary=False)

    assert 'WholeExtent="0 2 0 2 0 2"' in path.read_text()
    back = read(path)
    np.testing.assert_allclose(back.vertices, verts)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_allclose(back.vertex_attrs["scalar"], poly.vertex_attrs["scalar"])
    validate(back)


def test_cells_that_are_not_the_grids_own_are_refused(tmp_path) -> None:
    """The extent is the only thing that says which points the cells join."""
    axis = np.arange(3.0)
    verts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    tets = make_polydata(verts, [("tetra", np.array([[0, 1, 3, 9], [1, 2, 4, 10]]))])

    with pytest.raises(CodecError):
        write(tets, tmp_path / "tets.vts")


def test_an_axis_whose_extent_ends_before_it_starts_holds_nothing(tmp_path) -> None:
    """VTK spells an empty grid with an end before its start.

    One such axis empties the whole grid - the point count is a product. The
    other two axes' cells used to be counted anyway, so the mesh came back
    with four quads over no vertices at all, indexing point zero.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <StructuredGrid WholeExtent="0 -1 0 2 0 2">\n'
        '    <Piece Extent="0 -1 0 2 0 2">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'format="ascii"></DataArray>\n'
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "empty_axis.vts"
    path.write_text(body)

    poly = read(path)
    assert len(poly.vertices) == 0
    assert len(poly.element_types) == 0
    assert len(poly.connectivity) == 0
    validate(poly)


def test_an_extent_that_runs_backwards_holds_no_points(tmp_path) -> None:
    """An end two or more before its start turned the count negative.

    reshape took it as a second unknown dimension and failed from inside
    numpy, naming neither the file nor the extent.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <StructuredGrid WholeExtent="0 -5 0 2 0 2">\n'
        '    <Piece Extent="0 -5 0 2 0 2">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'format="ascii"></DataArray>\n'
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "backwards.vts"
    path.write_text(body)

    poly = read(path)
    assert len(poly.vertices) == 0
    validate(poly)


@pytest.mark.parametrize(
    ("extent", "points"),
    [
        ("-1 -1 0 0 0 0", ""),
        ("0 1 0 0 0 0", "0 0 1 0"),
        ("0 1 0 0 0 0", "0 0 0 1 0"),
    ],
    ids=["a point the array omits", "two coordinates each", "a ragged run"],
)
def test_points_the_extent_does_not_count_name_the_file(
    tmp_path, extent: str, points: str
) -> None:
    """A StructuredGrid carries its points; the extent and the array must agree.

    The column count is inferred from the two, so a file that disagreed came
    back as a mesh of the wrong width - a point of no coordinates at all -
    and only failed later, on a shape nothing in the file explained.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        f'  <StructuredGrid WholeExtent="{extent}">\n'
        f'    <Piece Extent="{extent}">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" '
        f'format="ascii">{points}</DataArray>\n'
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "short.vts"
    path.write_text(body)

    with pytest.raises(CodecError, match="coordinates each"):
        read(path)


def test_a_points_array_wider_than_three_keeps_its_first_three(tmp_path) -> None:
    """A fourth component is padding; the mesh is the three that matter."""
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        '  <StructuredGrid WholeExtent="0 1 0 0 0 0">\n'
        '    <Piece Extent="0 1 0 0 0 0">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'format="ascii">0 0 0 9 1 0 0 9</DataArray>\n'
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "wide.vts"
    path.write_text(body)

    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    validate(poly)


def _piece_xml(whole: str = "0 9 0 4 0 0") -> str:
    """One piece of a larger grid, the shape a .pvts assembles."""
    points = " ".join(f"{x} {y} 0" for y in (2, 3, 4) for x in (6, 7, 8, 9))
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">\n'
        f'  <StructuredGrid WholeExtent="{whole}">\n'
        '    <Piece Extent="6 9 2 4 0 0">\n'
        "      <Points>\n"
        f'        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{points}</DataArray>\n'
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )


def test_the_grid_a_piece_belongs_to_travels_with_it(tmp_path) -> None:
    """WholeExtent is the one thing a .pvts assembling the piece reads."""
    path = tmp_path / "piece.vts"
    path.write_text(_piece_xml())
    piece = read(path)
    assert piece.global_attrs["vts_whole_extent"] == [0, 9, 0, 4, 0, 0]

    out = tmp_path / "back.vts"
    write(piece, out, binary=False)
    body = out.read_text()
    assert 'WholeExtent="0 9 0 4 0 0"' in body
    assert 'Extent="6 9 2 4 0 0"' in body


def test_a_pruned_piece_drops_the_grid_it_no_longer_stands_in(tmp_path) -> None:
    """A re-derived extent is zero-based and says nothing about the old grid."""
    path = tmp_path / "piece.vts"
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
    out = tmp_path / "pruned.vts"
    write(one_cell, out, binary=False)
    body = out.read_text()
    assert 'Extent="0 1 0 1 0 0"' in body
    assert 'WholeExtent="0 1 0 1 0 0"' in body


def test_a_malformed_whole_extent_names_itself(tmp_path) -> None:
    """Unpacked straight into ints it failed with a literal and no file."""
    path = tmp_path / "bad.vts"
    path.write_text(_piece_xml(whole="nonsense"))
    with pytest.raises(CodecError, match="WholeExtent"):
        read(path)
