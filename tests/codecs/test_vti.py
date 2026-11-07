from __future__ import annotations

from dataclasses import replace
import tempfile
import tracemalloc
import warnings

import numpy as np
import pytest

import polyxios
from polyxios._element_types import ELEMENT_TYPES
from polyxios._image import ImageData
from polyxios._types import PolyData
from polyxios.codecs import _vtk_xml
from polyxios.codecs._vti import read, read_image, write, write_image
from polyxios.codecs._vtk_xml import (
    lattice_vertices,
    structured_cells,
    structured_offsets,
)
from polyxios.exceptions import CodecError, LazyReadError, ValidationError
from polyxios.transforms import remove_orphan_vertices
from polyxios.validate import validate


def _synthetic_vti() -> object:
    """Write then read a 2×2×2 ImageData, return the PolyData."""
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 2 0 2 0 2" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="0 2 0 2 0 2">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    with open(tmp, "w") as fh:
        fh.write(xml)
    return read(tmp)


def test_read_basic() -> None:
    poly = _synthetic_vti()
    # 2×2×2 grid → 3×3×3 = 27 vertices, 8 hex cells
    assert len(poly.vertices) == 27
    assert len(poly.element_types) == 8
    assert poly.vertices.dtype == np.float64
    assert poly.vertices.shape[1] == 3


def test_roundtrip_ascii() -> None:
    poly = _synthetic_vti()
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    assert len(poly2.element_types) == len(poly.element_types)


def test_roundtrip_binary() -> None:
    poly = _synthetic_vti()
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == len(poly.element_types)


def test_lazy_raises() -> None:
    poly = _synthetic_vti()
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_origin_and_spacing() -> None:
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 1 0 1 0 1" Origin="1 2 3" Spacing="0.5 0.5 0.5">\n'
        '    <Piece Extent="0 1 0 1 0 1">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    with open(tmp, "w") as fh:
        fh.write(xml)
    poly = read(tmp)
    # 1×1×1 grid → 8 vertices at origin + spacing
    assert len(poly.vertices) == 8
    np.testing.assert_allclose(poly.vertices.min(axis=0), [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(poly.vertices.max(axis=0), [1.5, 2.5, 3.5], atol=1e-12)


def test_cell_data_roundtrip() -> None:
    poly = _synthetic_vti()
    from polyxios._types import PolyData

    poly_with_attr = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
        vertex_attrs={},
        element_attrs={"pressure": np.arange(8, dtype=np.float64)},
        global_attrs=poly.global_attrs,
    )
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly_with_attr, tmp, binary=False)
    poly2 = read(tmp)
    assert "pressure" in poly2.element_attrs
    np.testing.assert_allclose(
        poly2.element_attrs["pressure"], np.arange(8, dtype=np.float64), atol=1e-6
    )


def test_a_flat_extent_is_a_sheet_of_quads_with_its_cell_data(tmp_path) -> None:
    """An image one voxel deep held no cells, so its CellData was dropped."""
    path = tmp_path / "flat.vti"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        ' <ImageData WholeExtent="0 2 0 2 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        '  <Piece Extent="0 2 0 2 0 0">\n'
        '   <CellData><DataArray type="Float64" Name="c"'
        ' format="ascii">1 2 3 4</DataArray></CellData>\n'
        "  </Piece>\n"
        " </ImageData>\n"
        "</VTKFile>\n"
    )

    poly = read(path)

    assert len(poly.element_types) == 4
    np.testing.assert_array_equal(poly.offsets, [0, 4, 8, 12, 16])
    np.testing.assert_allclose(poly.element_attrs["c"], [1.0, 2.0, 3.0, 4.0])
    validate(poly)


def test_a_mesh_flat_in_z_is_written_rather_than_raising(tmp_path) -> None:
    """Only the x axis was asked whether it had a step; y and z were indexed."""
    path = tmp_path / "flat.vti"
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
    )
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 3, 2]),
        offsets=np.array([0, 4]),
        element_types=np.array([ELEMENT_TYPES["quad"]], dtype=np.uint8),
    )

    write(poly, path, binary=False)

    body = path.read_text()
    assert 'WholeExtent="0 1 0 1 0 0"' in body
    # The step is measured per axis, so y keeps its 2 and the flat z falls
    # back to the default rather than borrowing x's.
    assert 'Spacing="1 2 1"' in body


def test_an_empty_mesh_writes_an_empty_image(tmp_path) -> None:
    """Every other VTK writer here spells an empty file; this one raised."""
    path = tmp_path / "empty.vti"
    poly = PolyData(
        vertices=np.zeros((0, 3)),
        connectivity=np.zeros(0, dtype=np.int64),
        offsets=np.zeros(1, dtype=np.int64),
        element_types=np.zeros(0, dtype=np.uint8),
    )

    write(poly, path, binary=False)

    body = path.read_text()
    # "0 -1" is how VTK spells an axis with no plane on it, and what the
    # .vts and .vtr writers here already spell. "0 0" is one point, so an
    # empty mesh came back holding a vertex it never had.
    assert 'WholeExtent="0 -1 0 -1 0 -1"' in body
    assert 'Origin="0 0 0"' in body

    back = read(path)
    assert back.vertices.shape == (0, 3)
    assert len(back.element_types) == 0
    validate(back)


def test_a_short_stored_spacing_still_writes_three_axes(tmp_path) -> None:
    """Zipping against a two-value spacing dropped the z axis silently.

    The axis it leaves out is given a step the mesh never carried, which is
    worth a word: this codec writes no coordinates, so the step is the
    geometry.
    """
    path = tmp_path / "short.vti"
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1]),
        offsets=np.array([0, 2]),
        element_types=np.array([ELEMENT_TYPES["line"]], dtype=np.uint8),
        global_attrs={"vti_spacing": [5.0, 6.0]},
    )

    with pytest.warns(UserWarning, match="vti_spacing spells 2 numbers"):
        write(poly, path, binary=False)

    body = path.read_text()
    # x measures its own step of 1; y keeps the 6 it was given; z was never
    # given one and falls back to the default rather than off the end.
    assert 'Spacing="1 6 1"' in body


def test_a_stale_extent_is_re_derived_rather_than_trusted(tmp_path) -> None:
    """A transform leaves the stored extent describing the grid the mesh was."""
    grid = tmp_path / "grid.vti"
    axis = np.arange(3.0)
    pts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    conn: list[int] = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                base = k * 9 + j * 3 + i
                conn += [
                    base,
                    base + 1,
                    base + 4,
                    base + 3,
                    base + 9,
                    base + 10,
                    base + 13,
                    base + 12,
                ]
    write(
        PolyData(
            vertices=pts,
            connectivity=np.array(conn),
            offsets=np.arange(9) * 8,
            element_types=np.full(8, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
        ),
        grid,
        binary=False,
    )
    poly = read(grid)
    assert poly.global_attrs["vti_extent"] == [0, 2, 0, 2, 0, 2]

    one_cell = remove_orphan_vertices(
        replace(
            poly,
            connectivity=poly.connectivity[:8],
            offsets=np.array([0, 8]),
            element_types=poly.element_types[:1],
        )
    )
    out = tmp_path / "pruned.vti"
    write(one_cell, out, binary=False)

    # The 3x3x3 extent it still carries would declare 27 points over the 8 in
    # the file, and its own reader would hand back a mesh that never existed.
    assert 'WholeExtent="0 1 0 1 0 1"' in out.read_text()
    back = read(out)
    assert back.vertices.shape == (8, 3)
    assert len(back.element_types) == 1
    validate(back)


def test_an_extent_that_still_fits_is_written_where_it_stood(tmp_path) -> None:
    """A grid that did not start at zero keeps its place; only a stale one goes.

    The extent is an index range, so the coordinates it stands for are the
    origin plus the index times the step: a grid running from index 5 with an
    origin of zero and a step of one sits at five, and the vertices have to be
    there or the extent is not this mesh's.
    """
    path = tmp_path / "offset.vti"
    verts = np.array(
        [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0], [6.0, 6.0, 0.0]]
    )
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1, 3, 2]),
        offsets=np.array([0, 4]),
        element_types=np.array([ELEMENT_TYPES["quad"]], dtype=np.uint8),
        global_attrs={"vti_extent": [5, 6, 5, 6, 0, 0]},
    )

    write(poly, path, binary=False)

    assert 'WholeExtent="5 6 5 6 0 0"' in path.read_text()


def _grid() -> PolyData:
    """A 2x2x2 grid of hexahedra, the mesh an ImageData is shaped for."""
    axis = np.arange(3.0)
    pts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    return PolyData(
        vertices=pts,
        connectivity=structured_cells(2, 2, 2)[0],
        offsets=np.arange(9) * 8,
        element_types=np.full(8, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
    )


def test_a_moved_grid_writes_the_origin_it_moved_to(tmp_path) -> None:
    """The origin travels with the mesh and was trusted after a transform."""
    path = tmp_path / "grid.vti"
    write(_grid(), path, binary=False)
    moved = replace(read(path), vertices=read(path).vertices + 5.0)

    out = tmp_path / "moved.vti"
    write(moved, out, binary=False)

    # An ImageData holds no points, so the origin is where they are: writing
    # the one the file was read with left the grid five units behind.
    assert 'Origin="5 5 5"' in out.read_text()
    np.testing.assert_allclose(read(out).vertices, moved.vertices)


def test_a_scaled_grid_writes_the_step_it_was_scaled_to(tmp_path) -> None:
    """The step travels with the mesh and was trusted the same way."""
    path = tmp_path / "grid.vti"
    write(_grid(), path, binary=False)
    scaled = replace(read(path), vertices=read(path).vertices * 10.0)

    out = tmp_path / "scaled.vti"
    write(scaled, out, binary=False)

    assert 'Spacing="10 10 10"' in out.read_text()
    np.testing.assert_allclose(read(out).vertices, scaled.vertices)


def test_an_unevenly_spaced_lattice_is_refused_rather_than_evened_out(
    tmp_path,
) -> None:
    """One step per axis is all the format has, and the first one was taken."""
    poly = _grid()
    stretched = poly.vertices.copy()
    stretched[stretched[:, 0] == 2.0, 0] = 5.0

    with pytest.raises(CodecError, match="evenly spaced"):
        write(replace(poly, vertices=stretched), tmp_path / "uneven.vti")


def test_cells_that_are_not_the_grids_own_are_refused(tmp_path) -> None:
    """The file carries no connectivity: these tets would come back as hexes."""
    poly = _grid()
    tets = replace(
        poly,
        connectivity=np.array([0, 1, 3, 9, 1, 2, 4, 10]),
        offsets=np.array([0, 4, 8]),
        element_types=np.full(2, ELEMENT_TYPES["tetra"], dtype=np.uint8),
    )

    with pytest.raises(CodecError):
        write(tets, tmp_path / "tets.vti")


def test_a_scalar_spacing_is_taken_for_every_axis(tmp_path) -> None:
    """A caller writing one number means it of all three, not of a slice."""
    path = tmp_path / "scalar.vti"
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1]),
        offsets=np.array([0, 2]),
        element_types=np.array([ELEMENT_TYPES["line"]], dtype=np.uint8),
        global_attrs={"vti_spacing": 0.5},
    )

    write(poly, path, binary=False)

    # x measures its own step; the two axes with one plane keep the 0.5 they
    # were handed rather than raising TypeError on the way past it.
    assert 'Spacing="1 0.5 0.5"' in path.read_text()


def test_a_short_origin_in_the_file_is_read_rather_than_indexed_past(
    tmp_path,
) -> None:
    """Two numbers where three belong raised IndexError from inside the parse.

    The third axis is given a value the file never spelled, so the read says
    so rather than handing back a grid standing somewhere the file did not
    put it.
    """
    path = tmp_path / "short.vti"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 1 0 1 0 1" Origin="0 0" Spacing="1 1">\n'
        '    <Piece Extent="0 1 0 1 0 1">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>"
    )

    # The file is short in both, and each names itself rather than the other.
    with pytest.warns(UserWarning, match="spells 2 numbers") as caught:
        poly = read(path)

    said = sorted(str(w.message).split(": ")[1].split()[0] for w in caught)
    assert said == ["Origin", "Spacing"]

    assert poly.vertices.shape == (8, 3)
    np.testing.assert_allclose(poly.vertices[:, 2], [0, 0, 0, 0, 1, 1, 1, 1])
    validate(poly)


def test_an_axis_whose_extent_ends_before_it_starts_holds_nothing(tmp_path) -> None:
    """VTK spells an empty grid with an end before its start.

    One such axis empties the whole grid - the point count is a product. The
    other two axes' cells used to be counted anyway, so the mesh came back
    with four quads over no vertices at all, indexing point zero.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 -1 0 2 0 2" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="0 -1 0 2 0 2">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "empty_axis.vti"
    path.write_text(body)

    poly = read(path)
    assert len(poly.vertices) == 0
    assert len(poly.element_types) == 0
    assert len(poly.connectivity) == 0
    validate(poly)


def test_an_origin_of_more_than_three_numbers_names_the_axes_it_drops(
    tmp_path,
) -> None:
    """A mesh has three axes; a fourth number describes nothing to write.

    Both wrong lengths warn; a dropped fourth number names no axis, where a
    missing third is an axis given a value the file never spelled.
    """
    body = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 1 0 1 0 0" Origin="1 2 3 4" '
        'Spacing="1 1 1">\n'
        '    <Piece Extent="0 1 0 1 0 0">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "long_origin.vti"
    path.write_text(body)

    with pytest.warns(UserWarning, match="Origin spells 4 numbers"):
        poly = read(path)

    assert poly.global_attrs["vti_origin"] == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(poly.vertices[0], [1.0, 2.0, 3.0])


def test_a_spacing_of_one_number_is_the_step_on_every_axis(tmp_path) -> None:
    """``vti_spacing=0.5`` means half a unit on all three, not a sliceable one."""
    poly = _grid()
    poly.global_attrs["vti_spacing"] = 0.5

    path = tmp_path / "bare.vti"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path, binary=False)

    validate(read(path))


def test_an_origin_that_is_no_number_falls_back_rather_than_raising(tmp_path) -> None:
    """``global_attrs`` may hold anything; the writer names it and re-derives.

    A string origin reached ``float`` and died with a bare ValueError from
    inside the writer. Taking the default instead leaves the writer's own
    check against the vertices to fail, so the origin and the step are read
    off the mesh - a file describing the grid that is there, rather than one
    built on a value nothing could read.
    """
    verts = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    poly = PolyData(
        vertices=verts,
        connectivity=np.array([0, 1]),
        offsets=np.array([0, 2]),
        element_types=np.array([ELEMENT_TYPES["line"]], dtype=np.uint8),
        global_attrs={"vti_origin": "2 0 0", "vti_spacing": ["a", "b", "c"]},
    )
    path = tmp_path / "junk.vti"

    with pytest.warns(UserWarning, match="is not a number or a run of them"):
        write(poly, path)

    assert np.allclose(read(path).vertices, verts)


def test_the_origin_and_step_are_written_at_the_width_a_double_holds(
    tmp_path,
) -> None:
    """An ImageData holds no points, so these six numbers are the geometry."""
    origin, spacing = 0.12345678901234, 1.0000000001234
    pts = np.array(
        [
            [origin + i * spacing, float(j), float(k)]
            for k in range(2)
            for j in range(2)
            for i in range(3)
        ]
    )
    grid = PolyData(
        vertices=pts,
        connectivity=structured_cells(2, 1, 1)[0],
        offsets=np.arange(3) * 8,
        element_types=np.full(2, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
    )
    path = tmp_path / "fine.vti"
    write(grid, path, binary=False)

    # A ``.10g`` field kept ten of the seventeen digits a double carries: the
    # origin came back 2.6e-10 away and the step came back as a flat 1, with
    # the error growing by one step per plane.
    assert np.array_equal(read(path).vertices, pts)


def test_an_axis_far_from_the_origin_is_measured_against_its_own_step(
    tmp_path,
) -> None:
    """A relative tolerance is a fraction of where the axis sits, not of it."""
    axis = np.array([1e6, 1e6 + 1.0, 1e6 + 2.001])
    pts = np.column_stack([axis, np.zeros(3), np.zeros(3)])
    lattice = PolyData(
        vertices=pts,
        connectivity=structured_cells(2, 0, 0)[0],
        offsets=np.arange(3) * 2,
        element_types=np.full(2, ELEMENT_TYPES["line"], dtype=np.uint8),
    )

    # At x = 1e6 a relative 1e-9 is half a millimetre of slack per plane, so
    # this passed and came back regularised with its last plane moved.
    with pytest.raises(CodecError, match="not evenly"):
        write(lattice, tmp_path / "far.vti", binary=False)


def test_a_bare_grid_is_readable_by_the_codec_that_wrote_it(tmp_path) -> None:
    """An ImageData describes its points rather than spelling them."""
    axis = np.arange(8.0)
    pts = np.array([[i, j, k] for k in axis for j in axis for i in axis])
    grid = PolyData(
        vertices=pts,
        connectivity=structured_cells(7, 7, 7)[0],
        offsets=np.arange(344) * 8,
        element_types=np.full(343, ELEMENT_TYPES["hexahedron"], dtype=np.uint8),
    )
    path = tmp_path / "bare.vti"
    write(grid, path, binary=False)

    # 512 points in a 237-byte file is the format working, not a corrupt
    # file: the header check weighed the count against bytes that are only
    # there when a format spells its points.
    assert np.array_equal(read(path).vertices, pts)


def test_the_grid_a_piece_belongs_to_travels_with_it(tmp_path) -> None:
    """WholeExtent is the one thing a .pvti assembling the piece reads."""
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 9 0 4 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="6 9 2 4 0 0"/>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "piece.vti"
    path.write_text(xml)
    piece = read(path)
    assert piece.global_attrs["vti_whole_extent"] == [0, 9, 0, 4, 0, 0]

    out = tmp_path / "back.vti"
    write(piece, out, binary=False)
    body = out.read_text()
    assert 'WholeExtent="0 9 0 4 0 0"' in body
    assert 'Extent="6 9 2 4 0 0"' in body


def test_a_moved_piece_drops_the_grid_it_no_longer_stands_in(tmp_path) -> None:
    """A re-derived extent is zero-based and says nothing about the old grid."""
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 9 0 4 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="6 9 2 4 0 0"/>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "piece.vti"
    path.write_text(xml)
    moved = read(path)
    moved = replace(moved, vertices=moved.vertices + 100.0)

    out = tmp_path / "moved.vti"
    write(moved, out, binary=False)
    assert 'WholeExtent="0 3 0 2 0 0"' in out.read_text()


def test_a_malformed_whole_extent_names_itself(tmp_path) -> None:
    """Unpacked straight into ints it failed with a literal and no file."""
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="nonsense" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="0 1 0 1 0 1"/>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path = tmp_path / "bad.vti"
    path.write_text(xml)
    with pytest.raises(CodecError, match="WholeExtent"):
        read(path)


def test_a_tiny_file_cannot_ask_for_a_grid_too_big_to_hold(tmp_path) -> None:
    """An ImageData spells its whole geometry in six indices and no points.

    Every other format has to spend bytes on a point before the reader
    allocates one, which is what the file-size heuristic weighs. This one
    spends none, so a 250-byte file declared a hundred million planes on the
    x axis and was expanded into the 2.4 GB of vertices they come to.
    """
    path = tmp_path / "bomb.vti"
    extent = "0 100000000 0 0 0 0"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="{extent}"/>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    assert path.stat().st_size < 1000

    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        read(path)


def test_an_image_of_a_size_worth_reading_is_still_read(tmp_path) -> None:
    """The cap is a bound on the absurd, not on the format working normally."""
    path = tmp_path / "big.vti"
    extent = "0 63 0 63 0 63"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="{extent}"/>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )

    poly = read(path)

    assert len(poly.vertices) == 64**3
    validate(poly)


def test_an_implied_grid_is_built_without_copies_of_itself(tmp_path) -> None:
    """A header-only grid allocates its mesh and little else.

    The vertices went through a ``meshgrid`` and a ``column_stack``: three
    full copies of every coordinate on the way to the one kept, about 80
    bytes a point for a run of lines. Under the cap, a 162-byte file asked
    for 67 million points and peaked at 5.4 GB.
    """
    n = 250_000
    path = tmp_path / "line.vti"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="0 0 0 {n - 1} 0 0"/>\n'
        "  </ImageData>\n"
        "</VTKFile>\n"
    )

    tracemalloc.start()
    try:
        poly = read(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert len(poly.vertices) == n
    # Kept: 24 bytes of vertices, 8 of connectivity, 4 of offsets, 1 of
    # type; the int64 cell origins (8) are the transient on top.
    assert peak < 56 * n


def test_an_implied_grid_runs_x_fastest() -> None:
    """The lattice comes out in the order a meshgrid expansion gave it."""
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([-2.0, 5.0], dtype=np.float32)
    z = np.array([7.0, 8.0, 9.0, 10.0])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    expected = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    out = lattice_vertices(x, y, z)

    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, expected)


def test_a_rotated_lattice_is_the_rotation_of_the_lattice() -> None:
    """Built a column at a time, it agrees with the matrix product it replaces."""
    x = np.array([0.0, 0.5, 2.0])
    y = np.array([-1.0, 4.0], dtype=np.float32)
    z = np.array([3.0, 3.5, 9.0, 10.0])
    theta = 0.3
    direction = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    out = lattice_vertices(x, y, z, direction=direction)

    assert out.dtype == np.float64
    np.testing.assert_allclose(
        out, lattice_vertices(x, y, z) @ direction.T, rtol=0, atol=1e-12
    )


def test_structured_offsets_widen_once_the_connectivity_outgrows_an_int32(
    monkeypatch,
) -> None:
    """The last offset is the connectivity length, which wraps an int32 first.

    Lowered so a grid of a few cells crosses it: 300 million points of
    hexahedra hold 2.4 billion corners, and int32 offsets wrapped negative.
    """
    assert structured_offsets(3, 8).dtype == np.int32
    monkeypatch.setattr(_vtk_xml, "_INT32_MAX", 23)

    offsets = structured_offsets(3, 8)

    assert offsets.dtype == np.int64
    np.testing.assert_array_equal(offsets, [0, 8, 16, 24])


def test_a_vti_past_the_int32_offsets_reads_them_wide(monkeypatch, tmp_path) -> None:
    """The reader takes the offsets' width from the connectivity length."""
    monkeypatch.setattr(_vtk_xml, "_INT32_MAX", 20)
    path = tmp_path / "wide.vti"
    path.write_text(_vti_text(extent="0 3 0 1 0 1"))

    poly = read(path)

    assert poly.offsets.dtype == np.int64
    np.testing.assert_array_equal(poly.offsets, np.arange(4) * 8)
    validate(poly)


def _vti_text(*, extent: str, piece: bool = True, data: str = "") -> str:
    body = (
        f'<Piece Extent="{extent}"><PointData>{data}</PointData></Piece>'
        if piece
        else ""
    )
    return (
        '<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0"'
        ' byte_order="LittleEndian">'
        f'<ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">'
        f"{body}</ImageData></VTKFile>"
    )


def test_an_axis_ending_before_it_starts_builds_no_coordinates(tmp_path) -> None:
    """The grid such an axis empties still had the other axes' coordinates
    built: a 180-byte file spelling 0..2666666666 along x took 21 GB."""
    path = tmp_path / "flat.vti"
    path.write_text(_vti_text(extent="0 2666666666 0 2 8 0"))
    poly = read(path)
    assert poly.vertices.shape == (0, 3)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_vti_text(extent="0 1 0 1 0 1", piece=False), "No <Piece>"),
        (
            _vti_text(
                extent="0 1 0 1 0 1",
                data='<DataArray type="Float32" Name="s" format="binary">AAAAA</DataArray>',
            ),
            "not valid base64",
        ),
    ],
)
def test_a_malformed_image_is_a_codec_error(tmp_path, text: str, match: str) -> None:
    """A missing piece raised ValueError and bad base64 binascii.Error."""
    path = tmp_path / "bad.vti"
    path.write_text(text)
    with pytest.raises(CodecError, match=match):
        read(path)


def test_bytes_that_are_not_utf8_do_not_escape_the_xml_parse(tmp_path) -> None:
    """A file keeping its arrays inline was decoded as strict UTF-8, so a
    stray byte raised UnicodeDecodeError; the appended path already replaced
    it."""
    path = tmp_path / "odd.vti"
    text = _vti_text(extent="0 1 0 1 0 1").replace(
        'Spacing="1 1 1"', 'Spacing="1 1 1" Note="\xff"'
    )
    path.write_bytes(text.encode("latin-1"))
    with pytest.warns(UserWarning, match="read as Latin-1"):
        assert read(path).vertices.shape == (8, 3)


def _offset_image() -> ImageData:
    """A grid that starts away from zero, with every kind of array."""
    return ImageData(
        extent=(1, 3, 2, 3, 0, 1),
        origin=(10.0, 20.0, 30.0),
        spacing=(0.5, 2.0, 4.0),
        point_attrs={
            "p": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            "v": np.arange(36, dtype=np.int16).reshape(2, 2, 3, 3),
        },
        cell_attrs={"c": np.arange(2, dtype=np.int32).reshape(1, 1, 2) + 5},
        global_attrs={
            "title": "grid",
            "gnum": np.array([42.0]),
            "vti_whole_extent": [0, 9, 0, 4, 0, 1],
        },
    )


def test_read_image_is_the_grid_read_expands(tmp_path) -> None:
    path = tmp_path / "g.vti"
    write_image(_offset_image(), path)
    image = read_image(path)
    full = read(path)
    expanded = image.to_polydata()
    np.testing.assert_array_equal(expanded.vertices, full.vertices)
    np.testing.assert_array_equal(expanded.connectivity, full.connectivity)
    np.testing.assert_array_equal(expanded.element_types, full.element_types)
    assert expanded.vertex_attrs.keys() == full.vertex_attrs.keys()
    for name, values in full.vertex_attrs.items():
        np.testing.assert_array_equal(expanded.vertex_attrs[name], values)
    for name, values in full.element_attrs.items():
        np.testing.assert_array_equal(expanded.element_attrs[name], values)
    expanded.global_attrs.update(
        vti_origin=[10.0, 20.0, 30.0],
        vti_spacing=[0.5, 2.0, 4.0],
        vti_extent=[1, 3, 2, 3, 0, 1],
    )
    assert expanded.global_attrs.keys() == full.global_attrs.keys()


def test_write_image_keeps_extent_origin_spacing_and_arrays(tmp_path) -> None:
    path = tmp_path / "g.vti"
    image = _offset_image()
    write_image(image, path, binary=False)
    back = read_image(path)
    assert back.extent == image.extent
    assert back.origin == image.origin
    assert back.spacing == image.spacing
    assert back.direction is None
    for name, values in image.point_attrs.items():
        np.testing.assert_array_equal(back.point_attrs[name], values)
    np.testing.assert_array_equal(back.cell_attrs["c"], image.cell_attrs["c"])
    assert back.global_attrs["title"] == "grid"


def test_a_wider_whole_extent_survives_the_image_path(tmp_path) -> None:
    """WholeExtent names the grid a piece belongs to; read_image then
    write_image must not narrow it to the piece."""
    path = tmp_path / "piece.vti"
    _write_piece(path, whole="0 9 0 4 0 1", extent="1 3 2 3 0 1")
    image = read_image(path)
    assert image.global_attrs["vti_whole_extent"] == [0, 9, 0, 4, 0, 1]
    again = tmp_path / "again.vti"
    write_image(image, again)
    assert 'WholeExtent="0 9 0 4 0 1"' in again.read_text(encoding="utf-8")
    assert read_image(again) == image
    assert read(again).global_attrs["vti_whole_extent"] == [0, 9, 0, 4, 0, 1]


def test_a_whole_extent_equal_to_the_piece_is_not_kept_on_the_image(tmp_path) -> None:
    path = tmp_path / "solo.vti"
    write_image(replace(_offset_image(), global_attrs={"title": "grid"}), path)
    assert 'WholeExtent="1 3 2 3 0 1"' in path.read_text(encoding="utf-8")
    assert "vti_whole_extent" not in read_image(path).global_attrs


def test_write_image_never_builds_the_mesh(tmp_path, monkeypatch) -> None:
    """A volume's mesh is 3 coordinates and a cell per sample; the file needs
    neither, and building one only to take it apart is how it was done."""
    image = _offset_image()
    path = tmp_path / "g.vti"
    expected = tmp_path / "mesh.vti"
    mesh = image.to_polydata()
    mesh.global_attrs.update(
        vti_origin=list(image.origin),
        vti_spacing=list(image.spacing),
        vti_extent=list(image.extent),
    )
    write(mesh, expected, binary=False)

    def _no_mesh(self):
        raise AssertionError("write_image built the mesh")

    monkeypatch.setattr(ImageData, "to_polydata", _no_mesh)
    write_image(image, path, binary=False)
    assert path.read_bytes() == expected.read_bytes()


def test_a_grid_of_no_points_is_refused_by_name(tmp_path) -> None:
    image = ImageData(extent=(0, -1, 0, 1, 0, 0))
    with pytest.raises(CodecError, match="holds no point, and an ImageData file"):
        write_image(image, tmp_path / "empty.vti")
    assert not (tmp_path / "empty.vti").exists()


def test_an_array_not_laid_out_along_the_extent_is_a_codec_error(tmp_path) -> None:
    image = ImageData(extent=(0, 2, 0, 1, 0, 0), point_attrs={"p": np.zeros((1, 3, 2))})
    with pytest.raises(CodecError, match="PointData array 'p'") as caught:
        write_image(image, tmp_path / "bad.vti")
    assert "ImageData:" not in str(caught.value)


def test_a_rotation_is_spelled_as_direction_and_read_back(tmp_path) -> None:
    swap = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    image = ImageData(extent=(0, 1, 0, 1, 0, 0), direction=swap)
    path = tmp_path / "rot.vti"
    write_image(image, path, binary=False)
    assert 'Direction="0 0 1 0 1 0 1 0 0"' in path.read_text(encoding="utf-8")
    assert read_image(path) == image


def test_an_identity_direction_is_not_spelled(tmp_path) -> None:
    path = tmp_path / "id.vti"
    write_image(
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.eye(3)), path, binary=False
    )
    assert "Direction" not in path.read_text(encoding="utf-8")
    assert read_image(path).direction is None


def test_a_flat_nine_number_direction_is_accepted(tmp_path) -> None:
    swap = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    path = tmp_path / "flat.vti"
    write_image(ImageData(extent=(0, 1, 0, 1, 0, 0), direction=swap.ravel()), path)
    assert read_image(path).direction.tolist() == swap.tolist()
    identity = tmp_path / "ident.vti"
    write_image(
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.eye(3).ravel()), identity
    )
    assert read_image(identity).direction is None


def test_a_direction_that_is_no_matrix_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="holds 4 numbers"):
        ImageData(extent=(0, 1, 0, 1, 0, 0), direction=np.zeros(4))


def test_read_keeps_the_mesh_axis_aligned_whatever_direction_says(tmp_path) -> None:
    """read() has never applied Direction; the rotation is for read_image."""
    plain = tmp_path / "plain.vti"
    rotated = tmp_path / "rotated.vti"
    _write_piece(plain, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1")
    _write_piece(
        rotated,
        whole="0 1 0 1 0 1",
        extent="0 1 0 1 0 1",
        direction="0 1 0 1 0 0 0 0 1",
    )
    assert read_image(rotated).direction.tolist() == [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert read(rotated).vertices.tolist() == read(plain).vertices.tolist()


def test_a_malformed_direction_warns_and_reads_the_identity(tmp_path) -> None:
    path = tmp_path / "short.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1", direction="1 0 0")
    with pytest.warns(
        UserWarning, match="Direction spells 3 numbers where nine belong"
    ):
        assert read_image(path).direction is None


_BLAME_CASES = [
    pytest.param({"origin": "1 2"}, "Origin spells 2 numbers", id="origin"),
    pytest.param(
        {
            "arrays": '<DataArray type="Float64" Name="p" NumberOfComponents="x" format="ascii">0 1 2 3</DataArray>'
        },
        "declares NumberOfComponents='x'",
        id="components",
    ),
    pytest.param(
        {"arrays": '<DataArray type="String" Name="s" format="ascii">a b</DataArray>'},
        "has type 'String'",
        id="string",
    ),
    pytest.param(
        {
            "arrays": '<DataArray type="Int32" Name="i" format="ascii">0 1.5 2 3</DataArray>'
        },
        "cannot hold",
        id="wrap",
    ),
    pytest.param(
        {
            "fields": '<FieldData><DataArray type="Float64" Name="f" NumberOfComponents="x" NumberOfTuples="1" format="ascii">1 2</DataArray></FieldData>'
        },
        "declares NumberOfComponents='x'",
        id="field-components",
    ),
]


@pytest.mark.parametrize("entry", [polyxios.read_image, polyxios.read])
@pytest.mark.parametrize(("spelling", "match"), _BLAME_CASES)
def test_a_warning_from_the_reader_is_blamed_on_the_caller(
    tmp_path, entry, spelling, match
) -> None:
    """The public readers are what a caller reaches; the file is theirs."""
    path = tmp_path / "bad.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1", **spelling)
    with pytest.warns(UserWarning) as caught:
        entry(path)
    blamed = [w for w in caught if match in str(w.message)]
    assert blamed and all(w.filename == __file__ for w in blamed)


def test_read_image_refuses_options_it_does_not_take(tmp_path) -> None:
    path = tmp_path / "g.vti"
    write_image(_offset_image(), path)
    with pytest.warns(UserWarning, match=r"vti read: unrecognized options \{'bogus'\}"):
        read_image(path, bogus=1)


def test_write_image_refuses_options_it_does_not_take(tmp_path) -> None:
    with pytest.warns(
        UserWarning, match=r"vti write: unrecognized options \{'bogus'\}"
    ):
        write_image(_offset_image(), tmp_path / "g.vti", bogus=1)


def test_binary_is_a_keyword_of_write_image(tmp_path) -> None:
    path = tmp_path / "ascii.vti"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_image(_offset_image(), path, binary=False)
    assert "<DataArray" in path.read_text(encoding="utf-8")
    assert 'format="ascii"' in path.read_text(encoding="utf-8")


def _write_piece(
    path,
    *,
    whole: str,
    extent: str,
    origin: str = "0 0 0",
    direction: str | None = None,
    arrays: str = "",
    fields: str = "",
) -> None:
    """A one-piece .vti with the given WholeExtent, Extent, optional rotation and
    point arrays, each spelled as its ``<DataArray>`` element."""
    spelled = f' Direction="{direction}"' if direction is not None else ""
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{whole}" Origin="{origin}" Spacing="1 1 1"'
        f"{spelled}>\n"
        f"{fields}"
        f'    <Piece Extent="{extent}">\n'
        f"      <PointData>{arrays}</PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "direction", ["1 0 0", "a b c d e f g h i"], ids=["short", "text"]
)
def test_read_ignores_a_malformed_direction_silently(tmp_path, direction) -> None:
    """The mesh never uses Direction, so read() does not parse it and does not warn
    about it; read_image, which keeps it, does."""
    path = tmp_path / "bad.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1", direction=direction)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read(path).vertices.shape == (8, 3)
    with pytest.warns(UserWarning, match="Direction"):
        assert read_image(path).direction is None


def test_a_tensor_array_is_written_with_its_components_flat(tmp_path) -> None:
    """The .vti counterpart of the .vtkhdf test: (nz, ny, nx, 3, 3) reads back with
    its nine components as one axis, as the lattice's components always do."""
    tensor = np.arange(2 * 2 * 3 * 9, dtype=np.float64).reshape(2, 2, 3, 3, 3)
    path = tmp_path / "t.vti"
    write_image(ImageData(extent=(0, 2, 0, 1, 0, 1), point_attrs={"t": tensor}), path)
    back = read_image(path).point_attrs["t"]
    assert back.shape == (2, 2, 3, 9)
    np.testing.assert_array_equal(back, tensor.reshape(2, 2, 3, 9))


def test_read_keeps_the_field_data_order_with_a_grid_key_in_it(tmp_path) -> None:
    """A FieldData entry spelled like a grid key keeps its place; the grid keys the
    mesh reader adds follow the field data, as they always have."""
    path = tmp_path / "order.vti"
    _write_piece(
        path,
        whole="0 9 0 1 0 1",
        extent="0 1 0 1 0 1",
        fields=(
            "<FieldData>"
            '<DataArray type="Float64" Name="title" NumberOfComponents="1" format="ascii">3</DataArray>'
            '<DataArray type="Float64" Name="vti_origin" NumberOfComponents="1" format="ascii">7 8 9</DataArray>'
            '<DataArray type="Float64" Name="zz" NumberOfComponents="1" format="ascii">4</DataArray>'
            "</FieldData>"
        ),
    )
    assert list(read(path).global_attrs) == [
        "title",
        "vti_origin",
        "zz",
        "vti_spacing",
        "vti_extent",
        "vti_whole_extent",
    ]


def test_a_bare_direction_is_refused_not_broadcast(tmp_path) -> None:
    """Direction="2" is one number where nine belong: the identity, with a warning,
    rather than the same number in all nine entries."""
    path = tmp_path / "bare.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1", direction="2")
    with pytest.warns(
        UserWarning, match="Direction spells 1 numbers where nine belong"
    ):
        assert read_image(path).direction is None


def test_a_short_origin_keeps_the_axis_wording_the_mesh_reader_always_had(
    tmp_path,
) -> None:
    """The mesh reader's text for a short Origin is unchanged from the release that
    first shipped it: it names the axes, not the entries."""
    path = tmp_path / "short.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1", origin="1 2")
    with pytest.warns(UserWarning) as caught:
        read(path)
    assert [str(w.message) for w in caught] == [
        ".vti: Origin spells 2 numbers where three belong; the axes it leaves out "
        "are given 0.0, which the file never said."
    ]


def test_a_field_name_xml_cannot_hold_is_blamed_on_the_entry_point(tmp_path) -> None:
    """A mesh written by polyxios.write is blamed on polyxios' own entry point, as it
    always was; an image written by polyxios.write_image is blamed on the caller."""
    path = tmp_path / "solo.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1")
    mesh = read(path)
    mesh.global_attrs["g\x01"] = np.array([1.0])
    with pytest.warns(UserWarning, match="field array name") as caught:
        polyxios.write(mesh, tmp_path / "mesh.vti")
    assert {w.filename for w in caught} == {polyxios.__file__}

    image = read_image(path)
    image.global_attrs["g\x01"] = np.array([1.0])
    with pytest.warns(UserWarning, match="field array name") as caught:
        polyxios.write_image(image, tmp_path / "image.vti")
    assert {w.filename for w in caught} == {__file__}


def test_mesh_and_image_writes_warn_in_the_order_the_metadata_is_read(tmp_path) -> None:
    """The dropped globals come first, then the field names XML cannot hold, then the
    point names: the order a mesh has always been written in, and the one an image
    now shares."""
    kinds = ("global_attrs", "field array name", "point array name")
    path = tmp_path / "solo.vti"
    _write_piece(path, whole="0 1 0 1 0 1", extent="0 1 0 1 0 1")

    mesh = read(path)
    mesh.vertex_attrs['a<b"c\x01'] = np.zeros(len(mesh.vertices))
    mesh.global_attrs["obj"] = object()
    mesh.global_attrs['g"\x01'] = np.array([1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polyxios.write(mesh, tmp_path / "mesh.vti")
    assert [k for w in caught for k in kinds if k in str(w.message)] == list(kinds)

    image = read_image(path)
    image.point_attrs['a<b"c\x01'] = np.zeros(image.dimensions[::-1])
    image.global_attrs["obj"] = object()
    image.global_attrs['g"\x01'] = np.array([1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polyxios.write_image(image, tmp_path / "image.vti")
    assert [k for w in caught for k in kinds if k in str(w.message)] == list(kinds)
    assert {w.filename for w in caught} == {__file__}


def test_read_image_keeps_the_grid_keys_out_of_its_field_data(tmp_path) -> None:
    """The mesh path lets the grid keys win over a field entry of the same name; the
    image must do the same, so its metadata is the other field data plus the whole
    extent when that is wider than the piece - not a field entry spelled like a grid
    key."""
    path = tmp_path / "order.vti"
    _write_piece(
        path,
        whole="0 9 0 1 0 1",
        extent="0 1 0 1 0 1",
        fields=(
            "<FieldData>"
            '<DataArray type="Float64" Name="title" NumberOfComponents="1" format="ascii">3</DataArray>'
            '<DataArray type="Float64" Name="vti_origin" NumberOfComponents="1" format="ascii">7 8 9</DataArray>'
            '<DataArray type="Float64" Name="zz" NumberOfComponents="1" format="ascii">4</DataArray>'
            '<DataArray type="Float64" Name="vti_whole_extent" NumberOfComponents="1" format="ascii">0 5 0 5 0 1</DataArray>'
            "</FieldData>"
        ),
    )
    image = read_image(path)
    assert set(image.global_attrs) == {"title", "zz", "vti_whole_extent"}
    assert image.global_attrs["vti_whole_extent"] == [0, 9, 0, 1, 0, 1]
    mesh = read(path)
    assert mesh.global_attrs["vti_origin"] == [0.0, 0.0, 0.0]


def test_a_vti_extent_past_int64_reads_as_a_mesh_and_refuses_as_an_image(
    tmp_path,
) -> None:
    """A mesh of one vertex read from such an extent is what read() always gave; an
    ImageData cannot hold the extent, so read_image refuses it as a CodecError."""
    path = tmp_path / "huge.vti"
    huge = "9223372036854775808 9223372036854775808 0 0 0 0"
    _write_piece(path, whole=huge, extent=huge)
    assert read(path).vertices.tolist() == [[9223372036854775808.0, 0.0, 0.0]]
    with pytest.raises(CodecError, match="outside the 64-bit"):
        read_image(path)


def test_a_bare_grid_past_the_mesh_cap_reads_compact(tmp_path) -> None:
    """A bare grid of more points than a mesh may hold builds no vertex, so read_image
    reads it; read() still refuses it as the mesh cap always has, and so does
    to_polydata."""
    n = 465
    image = ImageData(extent=(0, n - 1, 0, n - 1, 0, n - 1))
    path = tmp_path / "bare.vti"
    polyxios.write_image(image, path)
    assert read_image(path).dimensions == (n, n, n)
    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        read(path)
    with pytest.raises(ValidationError, match="MAX_IMPLIED_VERTICES"):
        image.to_polydata()


def test_a_grid_past_the_point_cap_is_refused_before_it_is_written(tmp_path) -> None:
    """A reader refuses a compact image past MAX_SAFE_VERTICES, so the writer refuses it
    first and leaves no file that cannot be read back."""
    path = tmp_path / "huge.vti"
    with pytest.raises(ValidationError, match="MAX_SAFE_VERTICES"):
        polyxios.write_image(ImageData(extent=(0, 999, 0, 999, 0, 999)), path)
    assert not path.exists()


@pytest.mark.parametrize("first", [2**53, 2**63 - 2])
def test_a_wide_extent_round_trips_exactly(tmp_path, first) -> None:
    """A WholeExtent past 2**53 is written and read back as the integers it is, so the
    two-point grid there comes back as the two points it was."""
    image = ImageData(
        extent=(first, first + 1, 0, 0, 0, 0),
        point_attrs={"a": np.array([[[1.0, 2.0]]])},
    )
    path = tmp_path / "wide.vti"
    polyxios.write_image(image, path)
    back = read_image(path)
    assert back == image
    assert back.extent == (first, first + 1, 0, 0, 0, 0)
