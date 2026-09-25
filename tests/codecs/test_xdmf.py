"""XDMF: XML light data over inline, HDF5 or binary heavy data.

The inline (``Format="XML"``) tests run everywhere; the HDF5 ones skip
without h5py, and one test stands in for the missing package to check the
refusal names the extra that installs it.
"""

from __future__ import annotations

import io
import mmap
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._optpkg import TripWire
from polyxios.codecs import _xdmf
from polyxios.codecs._xdmf import read, read_time_series, write, write_time_series
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    UnknownElementTypeError,
    UnsupportedFormatError,
)

_TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)


def _two_triangles(**kwargs):
    return make_polydata(
        _TRI, [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))], **kwargs
    )


def _mixed():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [2, 0, 0]],
        dtype=np.float64,
    )
    return make_polydata(
        verts,
        [
            ("line", np.array([[0, 1], [1, 5]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("quad", np.array([[0, 1, 3, 2]])),
            ("tetra", np.array([[0, 1, 2, 4]])),
        ],
        vertex_attrs={"s": np.arange(6.0)},
        element_attrs={"e": np.arange(5.0)},
        element_tags={"lines": np.array([0, 1]), "solid": np.array([4])},
        vertex_tags={"base": np.array([0, 1, 2, 3])},
    )


def _xml(path: Path) -> ET.Element:
    return ET.fromstring(path.read_bytes())


def _doc(body: str) -> str:
    return (
        '<?xml version="1.0"?>\n'
        '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n'
        f"<Domain>\n{body}\n</Domain>\n</Xdmf>\n"
    )


_SQUARE_GRID = """
<Grid Name="square" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>
  </Geometry>
  <Topology TopologyType="Triangle" NumberOfElements="2">
    <DataItem Dimensions="2 3" NumberType="Int" Format="XML">
      0 1 2  1 3 2
    </DataItem>
  </Topology>
  <Attribute Name="height" Center="Node" AttributeType="Scalar">
    <DataItem Dimensions="4" NumberType="Float" Format="XML">1 2 3 4</DataItem>
  </Attribute>
  <Attribute Name="area" Center="Cell" AttributeType="Scalar">
    <DataItem Dimensions="2" NumberType="Float" Format="XML">0.5 0.5</DataItem>
  </Attribute>
</Grid>
"""


def _write_doc(tmp_path: Path, body: str, name: str = "m.xdmf") -> Path:
    path = tmp_path / name
    path.write_text(_doc(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reading inline data
# ---------------------------------------------------------------------------


def test_a_single_type_grid_reads_with_its_attributes(tmp_path: Path) -> None:
    poly = read(_write_doc(tmp_path, _SQUARE_GRID))
    np.testing.assert_array_equal(poly.vertices, _TRI)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])
    assert poly.element_types.tolist() == [5, 5]
    np.testing.assert_array_equal(poly.vertex_attrs["height"], [1, 2, 3, 4])
    np.testing.assert_array_equal(poly.element_attrs["area"], [0.5, 0.5])
    assert poly.global_attrs == {}


def test_the_older_type_attribute_spellings_are_accepted(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace("TopologyType=", "Type=").replace(
        'GeometryType="XYZ"', 'Type="XYZ"'
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.element_types.tolist() == [5, 5]


def test_xmf_is_the_same_codec(tmp_path: Path) -> None:
    poly = polyxios.read(_write_doc(tmp_path, _SQUARE_GRID, "m.xmf"))
    assert len(poly.element_types) == 2


def test_an_inline_file_reads_from_a_buffer(tmp_path: Path) -> None:
    buffer = io.BytesIO(_doc(_SQUARE_GRID).encode())
    poly = polyxios.read(buffer, fmt="xdmf")
    assert len(poly.element_types) == 2


def test_a_planar_geometry_is_padded_and_flagged(tmp_path: Path) -> None:
    body = (
        _SQUARE_GRID.replace('GeometryType="XYZ"', 'GeometryType="XY"')
        .replace('Dimensions="4 3"', 'Dimensions="4 2"')
        .replace("0 0 0  1 0 0  0 1 0  1 1 0", "0 0  1 0  0 1  1 1")
    )
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertices, _TRI)
    assert poly.global_attrs == {"was_2d": True}


def test_a_split_axis_geometry_is_stacked(tmp_path: Path) -> None:
    body = """
<Grid Name="g" GridType="Uniform">
  <Geometry GeometryType="X_Y_Z">
    <DataItem Dimensions="3" Format="XML">0 1 0</DataItem>
    <DataItem Dimensions="3" Format="XML">0 0 1</DataItem>
    <DataItem Dimensions="3" Format="XML">5 5 5</DataItem>
  </Geometry>
  <Topology TopologyType="Triangle" NumberOfElements="1">
    <DataItem Dimensions="1 3" NumberType="Int" Format="XML">0 1 2</DataItem>
  </Topology>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertices, [[0, 0, 5], [1, 0, 5], [0, 1, 5]])


@pytest.mark.parametrize(
    ("xdmf_type", "nodes", "ptype"),
    [
        ("Polyvertex", 1, "vertex"),
        ("Polyline", 2, "line"),
        ("Quadrilateral", 4, "quad"),
        ("Tetrahedron", 4, "tetra"),
        ("Pyramid", 5, "pyramid"),
        ("Wedge", 6, "wedge"),
        ("Hexahedron", 8, "hexahedron"),
        ("Edge_3", 3, "quadratic_edge"),
        ("Tri_6", 6, "quadratic_triangle"),
        ("Quadrilateral_8", 8, "quadratic_quad"),
        ("Quadrilateral_9", 9, "biquadratic_quad"),
        ("Tetrahedron_10", 10, "quadratic_tetra"),
        ("Pyramid_13", 13, "quadratic_pyramid"),
        ("Wedge_15", 15, "quadratic_wedge"),
        ("Wedge_18", 18, "biquadratic_quadratic_wedge"),
        ("Hexahedron_20", 20, "quadratic_hexahedron"),
        ("Hexahedron_24", 24, "biquadratic_quadratic_hexahedron"),
        ("Hexahedron_27", 27, "triquadratic_hexahedron"),
        ("Hexahedron_64", 64, "lagrange_hexahedron"),
    ],
)
def test_every_named_topology_type_reads_as_its_element(
    tmp_path: Path, xdmf_type: str, nodes: int, ptype: str
) -> None:
    ids = " ".join(str(i) for i in range(nodes))
    coords = "\n".join(f"{i} 0 0" for i in range(nodes))
    body = f"""
<Grid Name="g" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="{nodes} 3" Format="XML">{coords}</DataItem>
  </Geometry>
  <Topology TopologyType="{xdmf_type}" NumberOfElements="1">
    <DataItem Dimensions="1 {nodes}" NumberType="Int" Format="XML">{ids}</DataItem>
  </Topology>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    assert poly.element_types.tolist() == [polyxios._element_types.ELEMENT_TYPES[ptype]]
    np.testing.assert_array_equal(poly.connectivity, np.arange(nodes))


def test_a_polygon_topology_takes_its_node_count_from_the_attribute(
    tmp_path: Path,
) -> None:
    body = """
<Grid Name="g" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="5 3" Format="XML">0 0 0 1 0 0 1 1 0 0 1 0 2 2 0</DataItem>
  </Geometry>
  <Topology TopologyType="Polygon" NodesPerElement="5" NumberOfElements="1">
    <DataItem Dimensions="5" NumberType="Int" Format="XML">0 1 2 3 4</DataItem>
  </Topology>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    assert poly.element_types.tolist() == [
        polyxios._element_types.ELEMENT_TYPES["polygon"]
    ]
    assert poly.offsets.tolist() == [0, 5]


def test_a_polygon_topology_without_a_node_count_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace('TopologyType="Triangle"', 'TopologyType="Polygon"')
    with pytest.raises(CodecError, match="NodesPerElement"):
        read(_write_doc(tmp_path, body))


def test_a_mixed_topology_counts_the_nodes_of_its_free_size_cells(
    tmp_path: Path,
) -> None:
    """Polyvertex, polyline and polygon carry their node count after the
    type code, the way the Xdmf library and VTK both spell them."""
    body = """
<Grid Name="g" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="5 3" Format="XML">0 0 0 1 0 0 1 1 0 0 1 0 2 2 0</DataItem>
  </Geometry>
  <Topology TopologyType="Mixed" NumberOfElements="4">
    <DataItem Dimensions="18" NumberType="Int" Format="XML">
      1 1 4
      2 3 0 1 2
      3 4 0 1 2 3
      4 0 1 2
    </DataItem>
  </Topology>
  <Attribute Name="c" Center="Cell">
    <DataItem Dimensions="4" Format="XML">10 11 12 13</DataItem>
  </Attribute>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    names = [polyxios._element_types.ELEMENT_TYPES_INV[t] for t in poly.element_types]
    assert names == ["vertex", "poly_line", "polygon", "triangle"]
    assert poly.offsets.tolist() == [0, 1, 4, 8, 11]
    np.testing.assert_array_equal(poly.element_attrs["c"], [10, 11, 12, 13])


def test_a_polyhedron_in_a_mixed_topology_is_dropped_and_the_cell_data_cut(
    tmp_path: Path,
) -> None:
    body = """
<Grid Name="g" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="4 3" Format="XML">0 0 0 1 0 0 0 1 0 0 0 1</DataItem>
  </Geometry>
  <Topology TopologyType="Mixed" NumberOfElements="3">
    <DataItem Dimensions="26" NumberType="Int" Format="XML">
      4 0 1 2
      16 4  3 0 1 2  3 0 1 3  3 1 2 3  3 0 2 3
      4 1 2 3
    </DataItem>
  </Topology>
  <Attribute Name="c" Center="Cell">
    <DataItem Dimensions="3" Format="XML">10 11 12</DataItem>
  </Attribute>
  <Set Name="last" SetType="Cell">
    <DataItem Dimensions="2" NumberType="Int" Format="XML">1 2</DataItem>
  </Set>
</Grid>
"""
    with pytest.warns(UserWarning, match="1 polyhedron cell"):
        poly = read(_write_doc(tmp_path, body))
    assert poly.element_types.tolist() == [5, 5]
    np.testing.assert_array_equal(poly.element_attrs["c"], [10, 12])
    np.testing.assert_array_equal(poly.element_tags["last"], [1])
    # A flat block that is a vector per cell is shaped before it is cut, the
    # way it is for a mesh with nothing dropped.
    flat = body.replace(
        '<DataItem Dimensions="3" Format="XML">10 11 12</DataItem>',
        '<DataItem Dimensions="6" Format="XML">10 1 11 2 12 3</DataItem>',
    )
    with pytest.warns(UserWarning, match="1 polyhedron cell"):
        poly = read(_write_doc(tmp_path, flat))
    np.testing.assert_array_equal(poly.element_attrs["c"], [[10, 1], [12, 3]])


def test_a_mixed_stream_spelled_as_floats_reads_and_a_fraction_is_refused(
    tmp_path: Path,
) -> None:
    body = _SQUARE_GRID.replace(
        'TopologyType="Triangle" NumberOfElements="2"',
        'TopologyType="Mixed" NumberOfElements="2"',
    ).replace(
        'Dimensions="2 3" NumberType="Int" Format="XML">\n      0 1 2  1 3 2',
        'Dimensions="8" NumberType="Float" Format="XML">\n      4 0 1 2  4 1 3 2',
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.connectivity.dtype.kind == "i"
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])
    with pytest.raises(CodecError, match="non-integer indices"):
        read(_write_doc(tmp_path, body.replace("4 1 3 2", "4 1.5 3 2")))


def test_an_integer_block_spelled_with_a_decimal_point_reads(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace("0 1 2  1 3 2", "0.0 1.0 2.0  1.0 3.0 2.0")
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])
    assert poly.connectivity.dtype.kind == "i"


def test_a_set_of_float_ids_is_read_whole_or_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        "</Grid>",
        """  <Set Name="top" SetType="Node">
    <DataItem Dimensions="2" NumberType="Float" Format="XML">2 3</DataItem>
  </Set>
</Grid>""",
    )
    np.testing.assert_array_equal(
        read(_write_doc(tmp_path, body)).vertex_tags["top"], [2, 3]
    )
    with pytest.raises(CodecError, match="set 'top' holds non-integer"):
        read(_write_doc(tmp_path, body.replace(">2 3<", ">2 3.5<")))


def test_an_unknown_mixed_code_is_named_not_a_key_error(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        'TopologyType="Triangle" NumberOfElements="2"',
        'TopologyType="Mixed" NumberOfElements="1"',
    ).replace(
        'Dimensions="2 3" NumberType="Int" Format="XML">\n      0 1 2  1 3 2',
        'Dimensions="4" NumberType="Int" Format="XML">\n      99 0 1 2',
    )
    with pytest.raises(UnknownElementTypeError, match="99"):
        read(_write_doc(tmp_path, body))


def test_a_truncated_mixed_stream_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        'TopologyType="Triangle" NumberOfElements="2"',
        'TopologyType="Mixed" NumberOfElements="1"',
    ).replace(
        'Dimensions="2 3" NumberType="Int" Format="XML">\n      0 1 2  1 3 2',
        'Dimensions="3" NumberType="Int" Format="XML">\n      4 0 1',
    )
    with pytest.raises(CodecError, match="ends inside a cell"):
        read(_write_doc(tmp_path, body))


def test_an_index_past_the_points_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace("0 1 2  1 3 2", "0 1 2  1 9 2")
    with pytest.raises(CodecError, match="indexes point 9"):
        read(_write_doc(tmp_path, body))


def test_a_data_item_short_of_its_dimensions_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace('Dimensions="4 3"', 'Dimensions="40 3"')
    with pytest.raises(CodecError, match="120 values"):
        read(_write_doc(tmp_path, body))


def test_a_value_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace("0 1 2  1 3 2", "0 1 2  1 x 2")
    with pytest.raises(CodecError, match="not a Int"):
        read(_write_doc(tmp_path, body))


def test_a_file_that_is_not_xdmf_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.xdmf"
    path.write_text('<?xml version="1.0"?><VTKFile/>')
    with pytest.raises(CodecError, match="not an XDMF file"):
        read(path)
    path.write_text("<Xdmf><Domain>")
    with pytest.raises(CodecError, match="not well-formed"):
        read(path)


def test_lazy_refuses_inline_values(tmp_path: Path) -> None:
    """Text has to be parsed before it holds numbers; there is nothing to map."""
    with pytest.raises(LazyReadError, match="inline"):
        read(_write_doc(tmp_path, _SQUARE_GRID), lazy=True)


def _mapped(arr: np.ndarray) -> bool:
    base = arr
    while isinstance(base, np.ndarray):
        base = base.base
    if isinstance(base, memoryview):
        base = base.obj
    return isinstance(base, mmap.mmap)


def _assert_lazy_views(back, poly) -> None:
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.vertex_attrs["s"], poly.vertex_attrs["s"])
    for arr in (back.vertices, back.connectivity, back.vertex_attrs["s"]):
        assert _mapped(arr)
        assert not arr.flags.writeable
    assert back.connectivity.dtype == back.offsets.dtype


def test_lazy_views_a_binary_sidecar(tmp_path: Path) -> None:
    """A Binary DataItem is a run of values at an offset in a raw file: the
    file is mapped and each array is a view of it."""
    poly = _two_triangles(vertex_attrs={"s": np.arange(4.0)})
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="binary")
    _assert_lazy_views(read(path, lazy=True), poly)


def test_lazy_views_a_contiguous_hdf5_dataset(tmp_path: Path) -> None:
    """A dataset stored contiguously without a filter sits at an offset HDF5
    can name, so the .h5 file is mapped and the dataset viewed in place."""
    pytest.importorskip("h5py")
    poly = _two_triangles(vertex_attrs={"s": np.arange(4.0)})
    path = tmp_path / "m.xdmf"
    write(poly, path)
    _assert_lazy_views(read(path, lazy=True), poly)


def test_lazy_refuses_a_compressed_hdf5_dataset(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    path = tmp_path / "m.xdmf"
    write(_two_triangles(), path, compression="gzip")
    with pytest.raises(LazyReadError, match="chunked|compressed"):
        read(path, lazy=True)
    np.testing.assert_array_equal(
        read(path).connectivity, _two_triangles().connectivity
    )


def test_an_unknown_read_option_is_warned_about(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        read(_write_doc(tmp_path, _SQUARE_GRID), bogus=1)


# ---------------------------------------------------------------------------
# Sets, information, grid attributes, time
# ---------------------------------------------------------------------------


def test_sets_become_tags_and_information_becomes_text(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        "</Grid>",
        """
  <Set Name="left" SetType="Node">
    <DataItem Dimensions="2" NumberType="Int" Format="XML">0 2</DataItem>
  </Set>
  <Set Name="first" SetType="Cell">
    <DataItem Dimensions="1" NumberType="Int" Format="XML">0</DataItem>
  </Set>
  <Set Name="edge" SetType="Face">
    <DataItem Dimensions="2" NumberType="Int" Format="XML">0 1</DataItem>
  </Set>
  <Information Name="solver" Value="fem"/>
  <Information Name="solver" Value="v2"/>
  <Attribute Name="gnum" Center="Grid">
    <DataItem Dimensions="1" NumberType="Int" Format="XML">42</DataItem>
  </Attribute>
  <Time Value="0.25"/>
</Grid>""",
    )
    with pytest.warns(UserWarning, match="'edge' is a Face set"):
        poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertex_tags["left"], [0, 2])
    np.testing.assert_array_equal(poly.element_tags["first"], [0])
    assert poly.global_attrs["solver"] == ["fem", "v2"]
    np.testing.assert_array_equal(poly.global_attrs["gnum"], [42])
    assert poly.global_attrs["time"] == 0.25


def test_an_information_named_time_does_not_bury_the_time_value(
    tmp_path: Path,
) -> None:
    """The <Time> value is the time; an <Information Name="time"> beside it
    would otherwise list the two together and read as no time at all."""
    body = _SQUARE_GRID.replace(
        "</Grid>",
        """
  <Time Value="0.25"/>
  <Information Name="time" Value="run-A"/>
</Grid>""",
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.global_attrs["time"] == 0.25


def test_an_inline_integer_past_its_precision_is_refused(tmp_path: Path) -> None:
    """A token too wide for the declared width must not wrap or saturate."""
    body = _SQUARE_GRID.replace("0 1 2  1 3 2", "0 1 2  1 99999999999 2")
    with pytest.raises(CodecError, match="not a Int of 4 byte"):
        read(_write_doc(tmp_path, body))
    body = _SQUARE_GRID.replace(
        'Dimensions="2" NumberType="Float" Format="XML">0.5 0.5',
        'Dimensions="2" NumberType="UInt" Precision="1" Format="XML">-1 0',
    )
    with pytest.raises(CodecError, match="not a UInt of 1 byte"):
        read(_write_doc(tmp_path, body))


def test_an_attribute_that_is_neither_per_point_nor_per_cell_is_skipped(
    tmp_path: Path,
) -> None:
    body = _SQUARE_GRID.replace(
        "</Grid>",
        """
  <Attribute Name="u" Center="Other" ItemType="FiniteElementFunction"
             ElementFamily="DG" ElementDegree="1" ElementCell="triangle">
    <DataItem Dimensions="2 3" NumberType="UInt" Format="XML">0 1 2 3 4 5</DataItem>
    <DataItem Dimensions="6" Format="XML">0 0 0 0 0 0</DataItem>
  </Attribute>
  <Attribute Name="short" Center="Node">
    <DataItem Dimensions="3" Format="XML">1 2 3</DataItem>
  </Attribute>
</Grid>""",
    )
    with pytest.warns(UserWarning, match="finite element function"):
        with pytest.warns(
            UserWarning, match="'short' does not hold one value per point"
        ):
            poly = read(_write_doc(tmp_path, body))
    assert set(poly.vertex_attrs) == {"height"}


def test_a_vector_and_a_tensor_keep_their_shape(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        "</Grid>",
        """
  <Attribute Name="v" Center="Node" AttributeType="Vector">
    <DataItem Dimensions="4 3" Format="XML">
      0 0 0 1 1 1 2 2 2 3 3 3
    </DataItem>
  </Attribute>
  <Attribute Name="t" Center="Cell" AttributeType="Tensor">
    <DataItem Dimensions="2 3 3" Format="XML">
      0 1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16 17
    </DataItem>
  </Attribute>
</Grid>""",
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.vertex_attrs["v"].shape == (4, 3)
    assert poly.element_attrs["t"].shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# References, includes, several grids
# ---------------------------------------------------------------------------


_TWO_GRIDS = """
<DataItem Name="Point Data" Format="XML" Dimensions="6 3">
  0.0 1.0 0.0
  1.0 0.0 0.0
  2.0 0.0 0.0
  2.0 2.0 0.0
  1.0 2.0 0.0
  0.0 3.0 0.0
</DataItem>
<Grid Name="One Quadrilateral">
  <Topology Type="Quadrilateral" NumberOfElements="1">
    <DataItem Format="XML" DataType="Int" Dimensions="1 4">1 2 3 4</DataItem>
  </Topology>
  <Geometry GeometryType="XYZ">
    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="Point Data"]</DataItem>
  </Geometry>
</Grid>
<Grid Name="Two Triangles">
  <Topology Type="Triangle" NumberOfElements="2">
    <DataItem Format="XML" DataType="Int" Dimensions="2 3">0 1 4  0 4 5</DataItem>
  </Topology>
  <Geometry GeometryType="XYZ">
    <DataItem Reference="XML">/Xdmf/Domain/DataItem[@Name="Point Data"]</DataItem>
  </Geometry>
</Grid>
"""


def test_issue_568_several_grids_merge_into_one_mesh_tagged_by_grid(
    tmp_path: Path,
) -> None:
    """Two uniform grids sharing one point block through a Reference: the
    reader that only took one grid refused this file, and the one that
    only took a DataItem with Dimensions could not follow the reference."""
    poly = read(_write_doc(tmp_path, _TWO_GRIDS))
    assert poly.vertices.shape == (12, 3)
    assert poly.element_types.tolist() == [9, 5, 5]
    np.testing.assert_array_equal(poly.connectivity, [1, 2, 3, 4, 6, 7, 10, 6, 10, 11])
    np.testing.assert_array_equal(poly.element_tags["One Quadrilateral"], [0])
    np.testing.assert_array_equal(poly.element_tags["Two Triangles"], [1, 2])


def test_a_reference_to_nothing_is_named(tmp_path: Path) -> None:
    body = _TWO_GRIDS.replace('DataItem[@Name="Point Data"]', 'DataItem[@Name="Nope"]')
    with pytest.raises(CodecError, match="names nothing"):
        read(_write_doc(tmp_path, body))


def test_a_grid_whose_name_is_taken_by_a_set_gets_a_free_one(tmp_path: Path) -> None:
    """A grid named ``grid_0`` with a set of that name too: the fallback
    name is taken as well, so the tag moves on to one that is free."""
    body = _TWO_GRIDS.replace(
        '<Grid Name="One Quadrilateral">', '<Grid Name="grid_0">'
    ).replace(
        "</Geometry>\n</Grid>\n",
        '</Geometry>\n<Set Name="grid_0" SetType="Cell">'
        '<DataItem Dimensions="1">0</DataItem></Set>\n</Grid>\n',
        1,
    )
    poly = read(_write_doc(tmp_path, body))
    assert set(poly.element_tags) == {"grid_0", "grid_0_1", "Two Triangles"}
    np.testing.assert_array_equal(poly.element_tags["grid_0"], [0])
    np.testing.assert_array_equal(poly.element_tags["grid_0_1"], [0])


def test_a_reference_chain_is_followed_to_its_end(tmp_path: Path) -> None:
    body = _TWO_GRIDS.replace(
        '<Grid Name="One Quadrilateral">',
        '<DataItem Name="Hop" Reference="XML">'
        '/Xdmf/Domain/DataItem[@Name="Point Data"]</DataItem>\n'
        '<Grid Name="One Quadrilateral">',
    ).replace(
        '/Xdmf/Domain/DataItem[@Name="Point Data"]</DataItem>\n  </Geometry>\n</Grid>\n'
        '<Grid Name="Two Triangles">',
        '/Xdmf/Domain/DataItem[@Name="Hop"]</DataItem>\n  </Geometry>\n</Grid>\n'
        '<Grid Name="Two Triangles">',
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.vertices.shape == (12, 3)


def test_a_reference_cycle_is_refused_not_followed_forever(tmp_path: Path) -> None:
    body = _TWO_GRIDS.replace(
        '<Grid Name="One Quadrilateral">',
        '<DataItem Name="A" Reference="XML">/Xdmf/Domain/DataItem[@Name="B"]</DataItem>\n'
        '<DataItem Name="B" Reference="XML">/Xdmf/Domain/DataItem[@Name="A"]</DataItem>\n'
        '<Grid Name="One Quadrilateral">',
    ).replace('DataItem[@Name="Point Data"]', 'DataItem[@Name="A"]', 1)
    with pytest.raises(CodecError, match="leads back to itself"):
        read(_write_doc(tmp_path, body))


def test_a_spatial_collection_is_read_flat(tmp_path: Path) -> None:
    body = (
        '<Grid Name="all" GridType="Collection" CollectionType="Spatial">'
        + _SQUARE_GRID
        + _SQUARE_GRID.replace('Name="square"', 'Name="square2"')
        + "</Grid>"
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.vertices.shape == (8, 3)
    assert set(poly.element_tags) == {"square", "square2"}
    np.testing.assert_array_equal(poly.vertex_attrs["height"], [1, 2, 3, 4] * 2)


def test_an_include_pointer_lends_a_mesh_to_another_grid(tmp_path: Path) -> None:
    body = (
        _SQUARE_GRID
        + """
<Grid Name="borrower" GridType="Uniform">
  <xi:include xpointer="xpointer(//Grid[@Name=&quot;square&quot;]/*[self::Topology or self::Geometry])"/>
  <Attribute Name="other" Center="Node">
    <DataItem Dimensions="4" Format="XML">4 3 2 1</DataItem>
  </Attribute>
</Grid>
"""
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.vertices.shape == (8, 3)
    assert set(poly.element_tags) == {"square", "borrower"}


def test_an_include_that_names_nothing_is_skipped_and_the_grid_refused(
    tmp_path: Path,
) -> None:
    body = """
<Grid Name="borrower" GridType="Uniform">
  <xi:include xpointer="xpointer(//Grid[@Name=&quot;nope&quot;]/*[self::Topology or self::Geometry])"/>
</Grid>
"""
    with pytest.warns(UserWarning, match="names nothing in the file"):
        with pytest.raises(CodecError, match="has no <Topology>"):
            read(_write_doc(tmp_path, body))


def test_an_include_of_another_document_is_not_followed(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace("</Grid>", '<xi:include href="other.xdmf"/></Grid>')
    with pytest.warns(UserWarning, match="includes another document"):
        poly = read(_write_doc(tmp_path, body))
    assert len(poly.element_types) == 2


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


_SERIES = """
<Grid CollectionType="Temporal" GridType="Collection" Name="TimeSeries">
  <Grid GridType="Uniform" Name="grid0">
    <Time Value="0"/>
    <Attribute Name="u" Center="Node">
      <DataItem Dimensions="4" Format="XML">0 0 0 0</DataItem>
    </Attribute>
    <Topology NodesPerElement="3" NumberOfElements="2" TopologyType="Triangle">
      <DataItem Dimensions="2 3" Format="XML" NumberType="UInt">0 1 2 1 2 3</DataItem>
    </Topology>
    <Geometry GeometryType="XY">
      <DataItem Dimensions="4 2" Format="XML">0 0 1 0 0 1 1 1</DataItem>
    </Geometry>
  </Grid>
  <Grid GridType="Uniform" Name="grid1">
    <Time Value="0.001"/>
    <Attribute Name="u" Center="Node">
      <DataItem Dimensions="4" Format="XML">1 1 1 1</DataItem>
    </Attribute>
    <xi:include xpointer="xpointer(//Grid[@Name=&#34;TimeSeries&#34;]/Grid[@Name=&#34;grid0&#34;]/*[self::Topology or self::Geometry])"/>
  </Grid>
</Grid>
"""


def test_issue_461_a_temporal_collection_reads_at_a_step(tmp_path: Path) -> None:
    """The FEniCS layout: every step after the first borrows the mesh of the
    first through an include pointer. The reader that met 'Grid' inside
    'Grid' as an unknown section refused it whole."""
    path = _write_doc(tmp_path, _SERIES)
    first = read(path)
    assert first.global_attrs["time"] == 0.0
    np.testing.assert_array_equal(first.vertex_attrs["u"], [0, 0, 0, 0])
    second = read(path, step=1)
    assert second.global_attrs["time"] == 0.001
    np.testing.assert_array_equal(second.vertex_attrs["u"], [1, 1, 1, 1])
    np.testing.assert_array_equal(second.connectivity, first.connectivity)
    assert read(path, step=-1).global_attrs["time"] == 0.001


def test_a_step_the_series_lacks_is_refused_by_count(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="holds 2 step"):
        read(_write_doc(tmp_path, _SERIES), step=5)


def test_a_step_asked_of_a_file_with_no_series_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="no temporal collection"):
        read(_write_doc(tmp_path, _SQUARE_GRID), step=0)


def test_the_whole_series_reads_one_mesh_per_step(tmp_path: Path) -> None:
    steps = read_time_series(_write_doc(tmp_path, _SERIES))
    assert [s.global_attrs["time"] for s in steps] == [0.0, 0.001]
    single = read_time_series(_write_doc(tmp_path, _SQUARE_GRID, "s.xdmf"))
    assert len(single) == 1 and "time" not in single[0].global_attrs


def test_a_collection_time_list_names_each_step(tmp_path: Path) -> None:
    body = (
        _SERIES.replace('<Time Value="0"/>', "")
        .replace('<Time Value="0.001"/>', "")
        .replace(
            'Name="TimeSeries">',
            'Name="TimeSeries">\n  <Time TimeType="List"><DataItem Dimensions="2" Format="XML">3 4</DataItem></Time>',
        )
    )
    steps = read_time_series(_write_doc(tmp_path, body))
    assert [s.global_attrs["time"] for s in steps] == [3.0, 4.0]


def test_a_collection_time_range_spreads_over_the_steps(tmp_path: Path) -> None:
    body = (
        _SERIES.replace('<Time Value="0"/>', "")
        .replace('<Time Value="0.001"/>', "")
        .replace(
            'Name="TimeSeries">',
            'Name="TimeSeries">\n  <Time TimeType="Range"><DataItem Dimensions="2" Format="XML">10 20</DataItem></Time>',
        )
    )
    steps = read_time_series(_write_doc(tmp_path, body))
    assert [s.global_attrs["time"] for s in steps] == [10.0, 20.0]


def test_a_positional_include_pointer_finds_the_step(tmp_path: Path) -> None:
    """The other common spelling of the shared-mesh pointer counts the step
    rather than naming it: ``Grid[1]`` is the first grid of the collection."""
    body = _SERIES.replace("Grid[@Name=&#34;grid0&#34;]", "Grid[1]")
    second = read(_write_doc(tmp_path, body), step=1)
    assert second.vertices.shape == (4, 3)
    np.testing.assert_array_equal(second.vertex_attrs["u"], [1, 1, 1, 1])
    absolute = _SERIES.replace(
        "//Grid[@Name=&#34;TimeSeries&#34;]",
        "/Xdmf/Domain/Grid[@Name=&#34;TimeSeries&#34;]",
    )
    assert read(_write_doc(tmp_path, absolute), step=1).vertices.shape == (4, 3)


def test_a_positional_pointer_past_the_collection_names_nothing(
    tmp_path: Path,
) -> None:
    body = _SERIES.replace("Grid[@Name=&#34;grid0&#34;]", "Grid[7]")
    with pytest.warns(UserWarning, match="names nothing"):
        with pytest.raises(CodecError, match="has no <Topology>"):
            read(_write_doc(tmp_path, body), step=1)


def test_the_series_reads_an_included_array_once_and_copies_it_out(
    tmp_path: Path, monkeypatch
) -> None:
    """Every step includes the first's geometry; the series decodes that
    DataItem once, and the steps do not share the array it hands back."""
    decoded: list[str] = []
    original = _xdmf._decode_item

    def counting(item, ctx):
        decoded.append(item.text or "")
        return original(item, ctx)

    monkeypatch.setattr(_xdmf, "_decode_item", counting)
    steps = read_time_series(_write_doc(tmp_path, _SERIES))
    geometry = "0 0 1 0 0 1 1 1"
    assert sum(text.strip() == geometry for text in decoded) == 1
    steps[0].vertices[0, 0] = 99.0
    assert steps[1].vertices[0, 0] == 0.0


def test_issue_1399_a_mesh_grid_beside_the_series_is_not_a_second_mesh(
    tmp_path: Path,
) -> None:
    """A writer that keeps the static mesh in a grid beside the temporal
    collection, every step including it, describes one mesh: reading the
    holder as a grid of its own draws it twice."""
    body = """
<Grid Name="TimeSeries_meshio" GridType="Collection" CollectionType="Temporal">
  <Grid>
    <xi:include xpointer="xpointer(//Grid[@Name=&quot;mesh&quot;]/*[self::Topology or self::Geometry])"/>
    <Time Value="0.0"/>
    <Attribute Name="d" Center="Node">
      <DataItem Dimensions="4" Format="XML">0 0 0 0</DataItem>
    </Attribute>
  </Grid>
  <Grid>
    <xi:include xpointer="xpointer(//Grid[@Name=&quot;mesh&quot;]/*[self::Topology or self::Geometry])"/>
    <Time Value="0.1"/>
    <Attribute Name="d" Center="Node">
      <DataItem Dimensions="4" Format="XML">1 1 1 1</DataItem>
    </Attribute>
  </Grid>
</Grid>
<Grid Name="mesh" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="4 3" Format="XML">0 0 0 1 0 0 0 1 0 1 1 0</DataItem>
  </Geometry>
  <Topology TopologyType="Triangle" NumberOfElements="2">
    <DataItem Dimensions="2 3" NumberType="Int" Format="XML">0 1 2 1 3 2</DataItem>
  </Topology>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body), step=1)
    assert poly.vertices.shape == (4, 3)
    assert poly.element_tags == {}
    np.testing.assert_array_equal(poly.vertex_attrs["d"], [1, 1, 1, 1])


# ---------------------------------------------------------------------------
# Structured topologies
# ---------------------------------------------------------------------------


def test_a_corectilinear_lattice_expands_to_hexahedra(tmp_path: Path) -> None:
    body = """
<Grid Name="lattice" GridType="Uniform">
  <Topology TopologyType="3DCoRectMesh" Dimensions="2 3 4"/>
  <Geometry GeometryType="ORIGIN_DXDYDZ">
    <DataItem Dimensions="3" Format="XML">10 20 30</DataItem>
    <DataItem Dimensions="3" Format="XML">1 2 3</DataItem>
  </Geometry>
  <Attribute Name="c" Center="Cell">
    <DataItem Dimensions="1 2 3" Format="XML">0 1 2 3 4 5</DataItem>
  </Attribute>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    assert poly.vertices.shape == (24, 3)
    # Slowest axis first: the file's "2 3 4" is z, y, x, and so are the
    # origin and the spacing.
    np.testing.assert_array_equal(poly.vertices[0], [30, 20, 10])
    np.testing.assert_array_equal(poly.vertices[1], [33, 20, 10])
    np.testing.assert_array_equal(poly.vertices[4], [30, 22, 10])
    np.testing.assert_array_equal(poly.vertices[12], [30, 20, 11])
    assert poly.element_types.tolist() == [12] * 6
    np.testing.assert_array_equal(poly.element_attrs["c"], np.arange(6))


def test_a_rectilinear_lattice_takes_its_axes(tmp_path: Path) -> None:
    body = """
<Grid Name="lattice" GridType="Uniform">
  <Topology TopologyType="2DRectMesh" Dimensions="2 3"/>
  <Geometry GeometryType="VXVY">
    <DataItem Dimensions="3" Format="XML">0 1 5</DataItem>
    <DataItem Dimensions="2" Format="XML">0 7</DataItem>
  </Geometry>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(
        poly.vertices,
        [[0, 0, 0], [1, 0, 0], [5, 0, 0], [0, 7, 0], [1, 7, 0], [5, 7, 0]],
    )
    assert poly.element_types.tolist() == [9, 9]
    assert poly.global_attrs == {"was_2d": True}


def test_a_curvilinear_lattice_takes_its_points(tmp_path: Path) -> None:
    body = """
<Grid Name="lattice" GridType="Uniform">
  <Topology TopologyType="2DSMesh" Dimensions="2 2"/>
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="4 3" Format="XML">0 0 0 1 0 0 0 1 0 1 1 0</DataItem>
  </Geometry>
</Grid>
"""
    poly = read(_write_doc(tmp_path, body))
    assert poly.element_types.tolist() == [9]
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 3, 2])


def test_a_lattice_whose_points_do_not_match_is_refused(tmp_path: Path) -> None:
    body = """
<Grid Name="lattice" GridType="Uniform">
  <Topology TopologyType="2DSMesh" Dimensions="2 3"/>
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="4 3" Format="XML">0 0 0 1 0 0 0 1 0 1 1 0</DataItem>
  </Geometry>
</Grid>
"""
    with pytest.raises(CodecError, match="has 6 points and the geometry holds 4"):
        read(_write_doc(tmp_path, body))


# ---------------------------------------------------------------------------
# HyperSlab, binary and HDF5 heavy data
# ---------------------------------------------------------------------------


def test_a_hyperslab_selects_from_its_source(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4" NumberType="Float" Format="XML">1 2 3 4</DataItem>""",
        """<DataItem ItemType="HyperSlab" Dimensions="4">
      <DataItem Dimensions="3 1" Format="XML">1 2 4</DataItem>
      <DataItem Dimensions="8" Format="XML">0 1 0 2 0 3 0 4</DataItem>
    </DataItem>""",
    )
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertex_attrs["height"], [1, 2, 3, 4])


def test_a_hyperslab_past_its_source_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4" NumberType="Float" Format="XML">1 2 3 4</DataItem>""",
        """<DataItem ItemType="HyperSlab" Dimensions="4">
      <DataItem Dimensions="3 1" Format="XML">2 2 4</DataItem>
      <DataItem Dimensions="8" Format="XML">0 1 0 2 0 3 0 4</DataItem>
    </DataItem>""",
    )
    with pytest.raises(CodecError, match="reaches past its source"):
        read(_write_doc(tmp_path, body))
    short = body.replace('<DataItem Dimensions="3 1" Format="XML">2 2 4</DataItem>', "")
    with pytest.raises(CodecError, match="holds 1 DataItem"):
        read(_write_doc(tmp_path, short))


def test_a_hyperslab_of_an_hdf_dataset_reads_only_its_slab(
    tmp_path: Path, monkeypatch
) -> None:
    """A step cut out of one big dataset is sliced by h5py, so the dataset is
    never read whole; the values are the same either way."""
    h5py = pytest.importorskip("h5py")

    def whole_read(*args):
        raise AssertionError("the dataset was read whole")

    monkeypatch.setattr(_xdmf, "_hdf_values", whole_read)
    with h5py.File(tmp_path / "m.h5", "w") as f:
        f["/u"] = np.arange(20.0).reshape(5, 4)
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4" NumberType="Float" Format="XML">1 2 3 4</DataItem>""",
        """<DataItem ItemType="HyperSlab" Dimensions="4">
      <DataItem Dimensions="3 2" Format="XML">2 0 1 1 1 4</DataItem>
      <DataItem Dimensions="5 4" Format="HDF">m.h5:/u</DataItem>
    </DataItem>""",
    ).replace(
        """<DataItem Dimensions="2" NumberType="Float" Format="XML">0.5 0.5</DataItem>""",
        """<DataItem ItemType="HyperSlab" Dimensions="2">
      <DataItem Dimensions="3 2" Format="XML">0 1 2 2 2 1</DataItem>
      <DataItem Dimensions="5 4" Format="HDF">m.h5:/u</DataItem>
    </DataItem>""",
    )
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertex_attrs["height"], [8, 9, 10, 11])
    np.testing.assert_array_equal(poly.element_attrs["area"], [1, 9])


def test_binary_heavy_data_is_read_at_its_seek(tmp_path: Path) -> None:
    coords = np.arange(12, dtype=">f8")
    (tmp_path / "m.bin").write_bytes(b"xx" + coords.tobytes())
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="Binary" Endian="Big" Seek="2">m.bin</DataItem>""",
    )
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertices, np.arange(12.0).reshape(4, 3))


def test_a_binary_file_too_short_for_its_item_is_refused(tmp_path: Path) -> None:
    (tmp_path / "m.bin").write_bytes(b"\0" * 8)
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="Binary">m.bin</DataItem>""",
    )
    with pytest.raises(CodecError, match="holds 8 bytes"):
        read(_write_doc(tmp_path, body))


def test_a_sidecar_outside_the_directory_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="Binary">../m.bin</DataItem>""",
    )
    with pytest.raises(CodecError, match="outside the file's own directory"):
        read(_write_doc(tmp_path, body))


def test_a_sidecar_needs_a_path(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="Binary">m.bin</DataItem>""",
    )
    with pytest.raises(CodecError, match="Pass a path instead"):
        polyxios.read(io.BytesIO(_doc(body).encode()), fmt="xdmf")


def test_hdf_heavy_data_reads_through_h5py(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "m.h5", "w") as f:
        f["/geo/points"] = np.arange(12.0).reshape(4, 3)
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="HDF">m.h5:/geo/points</DataItem>""",
    )
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertices, np.arange(12.0).reshape(4, 3))


def test_a_big_endian_hdf_dataset_comes_back_in_native_order(tmp_path: Path) -> None:
    """h5py hands a dataset back in its stored byte order; the mesh must not
    carry a swapped block on, whole or through a HyperSlab."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "m.h5", "w") as f:
        f["/geo/points"] = np.arange(12.0).reshape(4, 3).astype(">f8")
        f["/u"] = np.arange(8.0).reshape(2, 4).astype(">f4")
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="HDF">m.h5:/geo/points</DataItem>""",
    ).replace(
        "</Grid>",
        '<Attribute Name="u" Center="Node"><DataItem ItemType="HyperSlab"'
        ' Dimensions="4"><DataItem Dimensions="3 2">1 0 1 1 1 4</DataItem>'
        '<DataItem Dimensions="2 4" Format="HDF">m.h5:/u</DataItem>'
        "</DataItem></Attribute></Grid>",
    )
    poly = read(_write_doc(tmp_path, body))
    assert poly.vertices.dtype.isnative
    assert poly.vertex_attrs["u"].dtype.isnative
    np.testing.assert_array_equal(poly.vertex_attrs["u"], [4, 5, 6, 7])


def test_a_colon_in_an_hdf_dataset_path_stays_with_the_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "m.h5", "w") as f:
        f["/geo:points"] = np.arange(12.0).reshape(4, 3)
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" Format="HDF">m.h5:/geo:points</DataItem>""",
    )
    poly = read(_write_doc(tmp_path, body))
    np.testing.assert_array_equal(poly.vertices, np.arange(12.0).reshape(4, 3))


def test_a_grid_attribute_named_time_does_not_pose_as_the_time(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        "</Grid>",
        '<Attribute Name="time" Center="Grid"><DataItem Dimensions="1">5</DataItem>'
        '</Attribute><Time Value="2.5"/></Grid>',
    )
    with pytest.warns(UserWarning, match="grid attribute 'time'"):
        poly = read(_write_doc(tmp_path, body))
    assert poly.global_attrs["time"] == 2.5


def test_a_missing_hdf_dataset_is_named(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "m.h5", "w") as f:
        f["/geo/points"] = np.arange(12.0).reshape(4, 3)
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="HDF">m.h5:/nope</DataItem>""",
    )
    with pytest.raises(CodecError, match="no dataset '/nope'"):
        read(_write_doc(tmp_path, body))


def test_without_h5py_an_hdf_reference_names_the_extra(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(_xdmf, "_h5py", lambda: (TripWire("no h5py"), False))
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="HDF">m.h5:/points</DataItem>""",
    )
    with pytest.raises(UnsupportedFormatError, match=r"polyxios\[hdf5\]"):
        read(_write_doc(tmp_path, body))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_default_write_is_an_hdf5_sidecar_beside_the_file(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    out = tmp_path / "sub"
    out.mkdir()
    write(_two_triangles(), out / "m.xdmf")
    assert (out / "m.h5").exists()
    assert not (tmp_path / "m.h5").exists()
    root = _xml(out / "m.xdmf")
    item = root.find(".//Geometry/DataItem")
    assert item.get("Format") == "HDF"
    assert item.text.strip() == "m.h5:/mesh/geometry"
    back = read(out / "m.xdmf")
    np.testing.assert_array_equal(back.vertices, _TRI)


def test_issue_1518_the_sidecar_of_a_time_series_sits_beside_it(tmp_path: Path) -> None:
    """The heavy file of a series written to 'output/file.xdmf' belongs in
    'output/', not the working directory."""
    pytest.importorskip("h5py")
    out = tmp_path / "output"
    out.mkdir()
    mesh = _two_triangles()
    write_time_series([(0.0, mesh), (1.0, mesh)], out / "file.xdmf")
    assert (out / "file.h5").exists()
    assert not (tmp_path / "file.h5").exists()


def test_a_single_type_mesh_writes_a_named_topology(tmp_path: Path) -> None:
    path = tmp_path / "m.xdmf"
    write(_two_triangles(), path, data_format="xml")
    topology = _xml(path).find(".//Topology")
    assert topology.get("TopologyType") == "Triangle"
    assert topology.get("NumberOfElements") == "2"
    assert topology.find("DataItem").get("Dimensions") == "2 3"


def test_issue_1462_lines_beside_other_cells_write_as_a_mixed_topology(
    tmp_path: Path,
) -> None:
    """Three lines, two triangles and a quad: the writer that resized its
    line block in place raised a broadcast error here."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], dtype=float
    )
    poly = make_polydata(
        verts,
        [
            ("line", np.array([[0, 1], [1, 2], [2, 3]])),
            ("triangle", np.array([[0, 1, 2], [1, 3, 2]])),
            ("quad", np.array([[1, 4, 5, 3]])),
        ],
    )
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    topology = _xml(path).find(".//Topology")
    assert topology.get("TopologyType") == "Mixed"
    stream = [int(t) for t in topology.find("DataItem").text.split()]
    assert stream == [
        2,
        2,
        0,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        3,
        4,
        0,
        1,
        2,
        4,
        1,
        3,
        2,
        5,
        1,
        4,
        5,
        3,
    ]
    back = read(path)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)


def test_issue_1251_a_series_over_lines_alone_reads_back(tmp_path: Path) -> None:
    """A graph - three points, three edges - with point data over time. The
    file the reference implementation wrote for this opened in no viewer."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    poly = make_polydata(verts, [("line", np.array([[0, 1], [1, 2], [2, 0]]))])
    path = tmp_path / "graph.xdmf"
    steps = [
        (
            t,
            make_polydata(
                verts,
                [("line", np.array([[0, 1], [1, 2], [2, 0]]))],
                vertex_attrs={"T": np.full(3, float(i))},
            ),
        )
        for i, t in enumerate([0.0, 0.1, 0.21])
    ]
    write_time_series(steps, path, data_format="xml")
    topology = _xml(path).find(".//Topology")
    assert topology.get("TopologyType") == "Polyline"
    assert topology.get("NodesPerElement") == "2"
    back = read_time_series(path)
    assert [b.global_attrs["time"] for b in back] == [0.0, 0.1, 0.21]
    np.testing.assert_array_equal(back[2].vertex_attrs["T"], [2, 2, 2])
    np.testing.assert_array_equal(back[2].connectivity, poly.connectivity)


def test_a_mixed_mesh_round_trips_with_its_data_and_tags(tmp_path: Path) -> None:
    poly = _mixed()
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    back = read(path)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.offsets, poly.offsets)
    np.testing.assert_array_equal(back.vertex_attrs["s"], poly.vertex_attrs["s"])
    np.testing.assert_array_equal(back.element_attrs["e"], poly.element_attrs["e"])
    np.testing.assert_array_equal(back.element_tags["lines"], [0, 1])
    np.testing.assert_array_equal(back.element_tags["solid"], [4])
    np.testing.assert_array_equal(back.vertex_tags["base"], [0, 1, 2, 3])


def test_a_pixel_and_a_voxel_go_out_in_xdmf_corner_order(tmp_path: Path) -> None:
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    poly = make_polydata(
        verts,
        [
            ("pixel", np.array([[0, 1, 2, 3]])),
            ("voxel", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        ],
    )
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    stream = [int(t) for t in _xml(path).find(".//Topology/DataItem").text.split()]
    assert stream == [5, 0, 1, 3, 2, 9, 0, 1, 3, 2, 4, 5, 7, 6]
    back = read(path)
    assert back.element_types.tolist() == [9, 12]


def test_an_element_type_xdmf_cannot_hold_is_dropped_with_its_data(
    tmp_path: Path,
) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    poly = make_polydata(
        verts,
        [
            ("triangle_strip", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
        element_attrs={"e": np.array([7.0, 8.0])},
        element_tags={"g": np.array([0, 1])},
    )
    path = tmp_path / "m.xdmf"
    with pytest.warns(UserWarning, match=r"\['triangle_strip'\] have no XDMF topology"):
        write(poly, path, data_format="xml")
    back = read(path)
    assert back.element_types.tolist() == [5]
    np.testing.assert_array_equal(back.element_attrs["e"], [8.0])
    np.testing.assert_array_equal(back.element_tags["g"], [0])


def test_a_mesh_with_no_elements_writes_and_reads(tmp_path: Path) -> None:
    poly = make_polydata(_TRI, [])
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    back = read(path)
    assert back.vertices.shape == (4, 3)
    assert len(back.element_types) == 0


def test_a_planar_mesh_from_a_2d_file_writes_xy(tmp_path: Path) -> None:
    poly = _two_triangles(global_attrs={"was_2d": True})
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    geometry = _xml(path).find(".//Geometry")
    assert geometry.get("GeometryType") == "XY"
    assert geometry.find("DataItem").get("Dimensions") == "4 2"
    assert read(path).global_attrs == {"was_2d": True}


def test_globals_travel_as_information_and_grid_attributes(tmp_path: Path) -> None:
    poly = _two_triangles(
        global_attrs={
            "solver": "fem",
            "tags": ["a", "b"],
            "gnum": 42,
            "k": [1.5, 2.5],
            "time": 3.0,
        }
    )
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    root = _xml(path)
    assert [i.get("Value") for i in root.iter("Information")] == ["fem", "a", "b"]
    assert root.find(".//Time").get("Value") == "3.0"
    back = read(path)
    assert back.global_attrs["solver"] == "fem"
    assert back.global_attrs["tags"] == ["a", "b"]
    np.testing.assert_array_equal(back.global_attrs["gnum"], [42])
    np.testing.assert_array_equal(back.global_attrs["k"], [1.5, 2.5])
    assert back.global_attrs["time"] == 3.0


def test_a_time_option_wins_over_the_global(tmp_path: Path) -> None:
    poly = _two_triangles(global_attrs={"time": 3.0})
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml", time=7)
    assert read(path).global_attrs["time"] == 7.0
    with pytest.raises(CodecError, match="time='soon' is not a number"):
        write(poly, path, data_format="xml", time="soon")


def test_an_attribute_that_holds_no_numbers_is_warned_about(tmp_path: Path) -> None:
    poly = _two_triangles(
        vertex_attrs={
            "names": np.array(["a", "b", "c", "d"]),
            "ok": np.ones(4, dtype=bool),
        },
        element_attrs={"short": np.array([1.0])},
    )
    path = tmp_path / "m.xdmf"
    with pytest.warns(
        UserWarning, match=r"vertex attribute\(s\) \['names'\] hold no numbers"
    ):
        with pytest.warns(
            UserWarning,
            match=r"element attribute\(s\) \['short'\] are not one value per element",
        ):
            write(poly, path, data_format="xml")
    back = read(path)
    assert set(back.vertex_attrs) == {"ok"}
    assert back.vertex_attrs["ok"].dtype == np.uint8
    assert back.element_attrs == {}


def test_a_tag_naming_a_stale_index_is_trimmed_with_a_warning(tmp_path: Path) -> None:
    poly = _two_triangles(element_tags={"g": np.array([0, 9])})
    path = tmp_path / "m.xdmf"
    with pytest.warns(
        UserWarning, match=r"element tag group\(s\) \['g'\] name members"
    ):
        write(poly, path, data_format="xml")
    np.testing.assert_array_equal(read(path).element_tags["g"], [0])


def test_a_vector_and_a_tensor_are_typed_on_the_way_out(tmp_path: Path) -> None:
    poly = _two_triangles(
        vertex_attrs={
            "v": np.zeros((4, 3)),
            "t": np.zeros((4, 3, 3)),
            "m": np.zeros((4, 5)),
        },
    )
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    kinds = {
        a.get("Name"): a.get("AttributeType") for a in _xml(path).iter("Attribute")
    }
    assert kinds == {"v": "Vector", "t": "Tensor", "m": "Matrix"}
    back = read(path)
    assert back.vertex_attrs["t"].shape == (4, 3, 3)
    assert back.vertex_attrs["m"].shape == (4, 5)


def test_a_bad_data_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="none of 'hdf', 'xml' or 'binary'"):
        write(_two_triangles(), tmp_path / "m.xdmf", data_format="ascii")


def test_a_sidecar_format_refuses_a_buffer_and_a_gzip_path(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="file object is not enough"):
        write(_two_triangles(), io.BytesIO(), data_format="binary")
    with pytest.raises(CodecError, match="gzip is not handled"):
        write(_two_triangles(), tmp_path / "m.xdmf.gz", data_format="binary")


def test_the_inline_format_writes_to_a_buffer_and_gzip(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    polyxios.write(_two_triangles(), buffer, fmt="xdmf", data_format="xml")
    assert b"<Xdmf" in buffer.getvalue()
    path = tmp_path / "m.xdmf.gz"
    polyxios.write(_two_triangles(), path, data_format="xml")
    assert polyxios.read(path).vertices.shape == (4, 3)


def test_binary_format_writes_one_sidecar_with_seeks(tmp_path: Path) -> None:
    poly = _mixed()
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="binary")
    items = list(_xml(path).iter("DataItem"))
    assert all(item.get("Format") == "Binary" for item in items)
    assert all(item.text.strip() == "m.bin" for item in items)
    seeks = [int(item.get("Seek")) for item in items]
    assert seeks == sorted(seeks) and seeks[0] == 0 and len(set(seeks)) == len(seeks)
    back = read(path)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    np.testing.assert_array_equal(back.vertex_attrs["s"], poly.vertex_attrs["s"])


def test_without_h5py_the_default_write_names_the_extra_and_the_way_round(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(_xdmf, "_h5py", lambda: (TripWire("no h5py"), False))
    with pytest.raises(
        UnsupportedFormatError, match=r'polyxios\[hdf5\].*data_format="xml"'
    ):
        write(_two_triangles(), tmp_path / "m.xdmf")
    assert not (tmp_path / "m.xdmf").exists()


def test_hdf_datasets_are_named_after_their_arrays(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    poly = _mixed()
    poly = make_polydata(
        poly.vertices,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"a/b": np.arange(6.0), "a_b": np.arange(6.0)},
    )
    path = tmp_path / "m.xdmf"
    write(poly, path, compression="gzip", compression_opts=4)
    with h5py.File(tmp_path / "m.h5") as f:
        names = []
        f.visit(names.append)
        assert "mesh/node/a_b" in names and "mesh/node/a_b_1" in names
        assert f["mesh/geometry"].compression == "gzip"
    back = read(path)
    assert set(back.vertex_attrs) == {"a/b", "a_b"}


def test_an_unknown_write_option_is_warned_about(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_two_triangles(), tmp_path / "m.xdmf", data_format="xml", bogus=1)


def test_inline_values_round_trip_exactly_and_are_spelled_short(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    poly = make_polydata(
        rng.random((4, 3)),
        [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))],
        vertex_attrs={
            "f32": rng.random(4).astype(np.float32),
            "wide": np.array([[0.1, 1e300], [np.nan, -np.inf], [1, 2], [3, 4]]),
            "u8": np.array([255, 0, 7, 9], dtype=np.uint8),
        },
    )
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    text = path.read_text()
    assert "0.1 1e+300" in text
    assert "0.10000000000000001" not in text
    back = read(path)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    for name, held in poly.vertex_attrs.items():
        np.testing.assert_array_equal(back.vertex_attrs[name], held)
        assert back.vertex_attrs[name].dtype == held.dtype


# ---------------------------------------------------------------------------
# Writing a time series
# ---------------------------------------------------------------------------


def test_a_series_writes_the_mesh_once_and_includes_it_after(tmp_path: Path) -> None:
    verts = _TRI
    cells = [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]
    steps = (
        (0.5 * i, make_polydata(verts, cells, vertex_attrs={"u": np.full(4, float(i))}))
        for i in range(3)
    )
    path = tmp_path / "s.xdmf"
    write_time_series(steps, path, data_format="xml")
    root = _xml(path)
    collection = root.find("Domain/Grid")
    assert collection.get("CollectionType") == "Temporal"
    grids = collection.findall("Grid")
    assert len(grids) == 3
    assert len(root.findall(".//Topology")) == 1
    assert len(root.findall(".//Geometry")) == 1
    include = grids[1].find("{http://www.w3.org/2001/XInclude}include")
    assert 'Grid[@Name="step0"]' in include.get("xpointer")
    assert "self::Topology or self::Geometry" in include.get("xpointer")
    back = read_time_series(path)
    assert [b.global_attrs["time"] for b in back] == [0.0, 0.5, 1.0]
    np.testing.assert_array_equal(back[2].vertex_attrs["u"], [2, 2, 2, 2])
    np.testing.assert_array_equal(back[2].connectivity, back[0].connectivity)


def test_a_time_option_on_a_series_is_warned_about_not_swallowed(
    tmp_path: Path,
) -> None:
    steps = [(0.0, _two_triangles()), (1.0, _two_triangles())]
    with pytest.warns(UserWarning, match=r"unrecognized options \{'time'\}"):
        write_time_series(steps, tmp_path / "s.xdmf", data_format="xml", time=5.0)
    back = read_time_series(tmp_path / "s.xdmf")
    assert [b.global_attrs["time"] for b in back] == [0.0, 1.0]


def test_a_moving_mesh_writes_its_own_geometry_per_step(tmp_path: Path) -> None:
    cells = [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]
    steps = [(0.0, make_polydata(_TRI, cells)), (1.0, make_polydata(_TRI + 1.0, cells))]
    path = tmp_path / "s.xdmf"
    write_time_series(steps, path, data_format="xml")
    root = _xml(path)
    assert len(root.findall(".//Geometry")) == 2
    assert len(root.findall(".//Topology")) == 1
    back = read_time_series(path)
    np.testing.assert_array_equal(back[1].vertices, _TRI + 1.0)
    np.testing.assert_array_equal(back[1].connectivity, back[0].connectivity)


def test_a_step_with_other_elements_is_refused(tmp_path: Path) -> None:
    steps = [
        (0.0, make_polydata(_TRI, [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))])),
        (1.0, make_polydata(_TRI, [("triangle", np.array([[0, 1, 2]]))])),
    ]
    with pytest.raises(CodecError, match="step 1 holds different elements"):
        write_time_series(steps, tmp_path / "s.xdmf", data_format="xml")


def test_an_empty_series_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="at least one step"):
        write_time_series([], tmp_path / "s.xdmf", data_format="xml")


def test_an_empty_series_leaves_no_sidecar_behind(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="at least one step"):
        write_time_series([], tmp_path / "s.xdmf", data_format="binary")
    assert not (tmp_path / "s.bin").exists()
    assert not (tmp_path / "s.xdmf").exists()


def test_a_series_warns_once_for_the_elements_it_drops(tmp_path: Path) -> None:
    """Every step holds the first's elements, so the elements XDMF cannot
    hold are named once, not once per step."""
    cells = [
        ("triangle_strip", np.array([[0, 1, 2, 3]])),
        ("triangle", np.array([[0, 1, 2]])),
    ]
    steps = [(float(k), make_polydata(_TRI, cells)) for k in range(4)]
    with pytest.warns(UserWarning) as caught:
        write_time_series(steps, tmp_path / "s.xdmf", data_format="xml")
    dropped = [w for w in caught if "have no XDMF topology" in str(w.message)]
    assert len(dropped) == 1
    back = read_time_series(tmp_path / "s.xdmf")
    assert [b.element_types.tolist() for b in back] == [[5]] * 4


# ---------------------------------------------------------------------------
# Malformed streams, sidecars and names
# ---------------------------------------------------------------------------


_TET_POINTS = """
<Grid Name="g" GridType="Uniform">
  <Geometry GeometryType="XYZ">
    <DataItem Dimensions="4 3" Format="XML">0 0 0 1 0 0 0 1 0 0 0 1</DataItem>
  </Geometry>
  <Topology TopologyType="Mixed" NumberOfElements="1">
    <DataItem Dimensions="{n}" NumberType="Int" Precision="8" Format="XML">
      {stream}
    </DataItem>
  </Topology>
</Grid>
"""


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        ("16 4000000000000 -1 0", "declares 4000000000000 faces"),
        ("16 -1 3 0 1 2", "declares -1 faces"),
        ("16 2 3 0 1 2 -1 4", "face in the mixed topology declares -1 nodes"),
        ("16 2 3 0 1 2 0 4", "face in the mixed topology declares 0 nodes"),
        ("16 3 3 0 1 2 3 0 1 3", "ends inside a cell"),
    ],
)
def test_a_malformed_polyhedron_is_refused_not_spun_on(
    tmp_path: Path, stream: str, message: str
) -> None:
    """A face count that cannot fit, or a face of no nodes, used to leave the
    walk in place; a count of 4e12 faces of -1 nodes ran for as long as it
    was let. Each is a malformed stream and is refused at once."""
    body = _TET_POINTS.format(n=len(stream.split()), stream=stream)
    with pytest.raises(CodecError, match=message):
        read(_write_doc(tmp_path, body))


def test_a_topology_that_holds_more_cells_than_declared_warns(
    tmp_path: Path,
) -> None:
    body = _SQUARE_GRID.replace('NumberOfElements="2"', 'NumberOfElements="1"')
    with pytest.warns(UserWarning, match="declares 1 cells and holds 2"):
        poly = read(_write_doc(tmp_path, body))
    assert len(poly.element_types) == 1


def test_a_sidecar_that_is_a_symlink_beside_the_file_is_read(tmp_path: Path) -> None:
    """The cluster layout: the XML in the run directory, the bytes on bulk
    storage, a link between them. The check is on the name the XML spells,
    not on where the link leads."""
    store = tmp_path / "store"
    store.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    write(_two_triangles(), store / "m.xdmf", data_format="binary")
    try:
        (run / "m.bin").symlink_to(store / "m.bin")
    except OSError as exc:
        pytest.skip(f"symlinks are not available here: {exc}")
    (run / "m.xdmf").write_bytes((store / "m.xdmf").read_bytes())
    assert read(run / "m.xdmf").element_types.tolist() == [5, 5]


@pytest.mark.parametrize("reference", ["../m.bin", "sub/../../m.bin", "/etc/m.bin"])
def test_a_spelled_path_that_leaves_the_directory_is_refused(
    tmp_path: Path, reference: str
) -> None:
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="XML">
      0 0 0  1 0 0  0 1 0  1 1 0
    </DataItem>""",
        f"""<DataItem Dimensions="4 3" NumberType="Float" Precision="8" Format="Binary">{reference}</DataItem>""",
    )
    with pytest.raises(CodecError, match="outside the file's own directory"):
        read(_write_doc(tmp_path, body))


def test_an_hdf_item_is_not_refused_for_a_number_type_it_never_reads(
    tmp_path: Path,
) -> None:
    pytest.importorskip("h5py")
    path = tmp_path / "m.xdmf"
    write(_two_triangles(), path)
    text = path.read_text(encoding="utf-8")
    assert 'Precision="8"' in text
    path.write_text(text.replace('Precision="8"', 'Precision="16"'), encoding="utf-8")
    assert read(path).element_types.tolist() == [5, 5]


def test_a_hyperslab_selection_with_a_fraction_is_refused(tmp_path: Path) -> None:
    body = _SQUARE_GRID.replace(
        """<DataItem Dimensions="4" NumberType="Float" Format="XML">1 2 3 4</DataItem>""",
        """<DataItem ItemType="HyperSlab" Dimensions="4">
      <DataItem Dimensions="3" Format="XML">0.5 1 4</DataItem>
      <DataItem Dimensions="8" Format="XML">1 2 3 4 5 6 7 8</DataItem>
    </DataItem>""",
    )
    with pytest.raises(CodecError, match="HyperSlab selection holds non-integer"):
        read(_write_doc(tmp_path, body))


def test_only_the_first_temporal_collection_is_read(tmp_path: Path) -> None:
    """Two series in one file would be read at one step, which the second
    may not have; the first is the series, the second is named and left."""
    second = _SERIES.replace('Name="TimeSeries"', 'Name="Other"').replace(
        'Name="grid0"', 'Name="o0"'
    )
    second = (
        second[: second.index('<Grid GridType="Uniform" Name="grid1">')] + "</Grid>"
    )
    path = _write_doc(tmp_path, _SERIES + second)
    with pytest.warns(UserWarning, match="only the first is read, and 'Other'"):
        steps = read_time_series(path)
    assert [s.global_attrs["time"] for s in steps] == [0.0, 0.001]
    assert all(len(s.element_types) == 2 for s in steps)


def test_a_failed_write_leaves_no_sidecar_behind(tmp_path: Path, monkeypatch) -> None:
    """The sidecar is moved into place once the arrays are all in it, so a
    write that fails halfway leaves neither a truncated sidecar nor a stale
    one from before beside a missing XML file."""
    write(_two_triangles(), tmp_path / "m.xdmf", data_format="binary")
    before = (tmp_path / "m.bin").read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_xdmf, "_set_lines", boom)
    with pytest.raises(RuntimeError, match="boom"):
        write(_two_triangles(), tmp_path / "m.xdmf", data_format="binary")
    assert (tmp_path / "m.bin").read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))
    pytest.importorskip("h5py")
    with pytest.raises(RuntimeError, match="boom"):
        write(_two_triangles(), tmp_path / "h.xdmf")
    assert not (tmp_path / "h.h5").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_dataset_name_hdf5_reserves_is_still_written(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    poly = _two_triangles(
        vertex_attrs={".": np.arange(4.0), "..": np.ones(4), "a/b": np.zeros(4)}
    )
    path = tmp_path / "m.xdmf"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["."], np.arange(4.0))
    np.testing.assert_array_equal(back.vertex_attrs[".."], np.ones(4))
    np.testing.assert_array_equal(back.vertex_attrs["a/b"], np.zeros(4))


def test_a_text_time_global_round_trips_as_information(tmp_path: Path) -> None:
    poly = _two_triangles(global_attrs={"time": "run-A", "note": "x"})
    path = tmp_path / "m.xdmf"
    write(poly, path, data_format="xml")
    assert read(path).global_attrs == {"time": "run-A", "note": "x"}
    with pytest.warns(UserWarning, match="'run-A', which is not the <Time> written"):
        write(poly, path, data_format="xml", time=2.0)
    back = read(path)
    assert back.global_attrs["time"] == 2.0
    with pytest.warns(UserWarning, match="not a number"):
        write(
            _two_triangles(global_attrs={"time": np.arange(3)}), path, data_format="xml"
        )
    assert "time" not in read(path).global_attrs


def test_h5py_is_not_imported_with_polyxios() -> None:
    code = "import sys, polyxios; print('h5py' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_merged_grids_are_planar_only_when_every_grid_was(tmp_path: Path) -> None:
    """A 3-D grid merged with a planar one used to take the planar flag from
    whichever grid came last, and lose its z on the way back out."""
    flat = """
<Grid Name="flat" GridType="Uniform">
  <Geometry GeometryType="XY">
    <DataItem Dimensions="3 2" Format="XML">0 0 1 0 0 1</DataItem>
  </Geometry>
  <Topology TopologyType="Triangle" NumberOfElements="1">
    <DataItem Dimensions="3" Format="XML">0 1 2</DataItem>
  </Topology>
</Grid>
"""
    tall = flat.replace('Name="flat"', 'Name="tall"').replace(
        """<Geometry GeometryType="XY">
    <DataItem Dimensions="3 2" Format="XML">0 0 1 0 0 1</DataItem>""",
        """<Geometry GeometryType="XYZ">
    <DataItem Dimensions="3 3" Format="XML">0 0 5 1 0 5 0 1 5</DataItem>""",
    )
    poly = read(_write_doc(tmp_path, tall + flat))
    assert "was_2d" not in poly.global_attrs
    out = tmp_path / "out.xdmf"
    write(poly, out, data_format="xml")
    assert read(out).vertices[:, 2].tolist() == [5, 5, 5, 0, 0, 0]
    both_flat = read(_write_doc(tmp_path, flat + flat.replace("flat", "flat2")))
    assert both_flat.global_attrs["was_2d"] is True


def test_a_compression_level_without_a_method_is_refused_up_front(
    tmp_path: Path,
) -> None:
    with pytest.raises(CodecError, match="compression names no method"):
        write(_two_triangles(), tmp_path / "m.xdmf", compression_opts=4)
    assert not (tmp_path / "m.h5").exists()
