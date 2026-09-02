from __future__ import annotations

from dataclasses import replace
import tempfile
import warnings

import numpy as np
import pytest

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._vti import read, write
from polyxios.codecs._vtk_xml import structured_cells
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
