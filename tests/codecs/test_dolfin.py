from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._dolfin import read, sniff, write
from polyxios.exceptions import CodecError


def _tet_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def _flat_tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    return make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.xml"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tri_mesh()
    out = tmp_path / "tri.xml"
    write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_file_is_valid_xml(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.xml"
    write(poly, out)
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "dolfin"
    assert root.find("mesh") is not None


def test_bad_xml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<not valid xml <<")
    with pytest.raises(CodecError):
        read(bad)


def test_unsupported_celltype_raises(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="wedge" dim="3">'
        '<vertices size="0"/><cells size="0"/></mesh></dolfin>'
    )
    bad = tmp_path / "wedge.xml"
    bad.write_text(xml)
    with pytest.raises(CodecError):
        read(bad)


def test_missing_vertex_attr_raises(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="triangle" dim="2">'
        '<vertices size="1"><vertex index="0" y="0" z="0"/></vertices>'
        '<cells size="0"/></mesh></dolfin>'
    )
    bad = tmp_path / "no_x.xml"
    bad.write_text(xml)
    with pytest.raises(CodecError, match="missing 'x'"):
        read(bad)


def test_read_2d_mesh_no_z(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="triangle" dim="2">'
        '<vertices size="3">'
        '<vertex index="0" x="0" y="0"/>'
        '<vertex index="1" x="1" y="0"/>'
        '<vertex index="2" x="0" y="1"/>'
        "</vertices>"
        '<cells size="1"><triangle index="0" v0="0" v1="1" v2="2"/></cells>'
        "</mesh></dolfin>"
    )
    f = tmp_path / "2d.xml"
    f.write_text(xml)
    poly = read(f)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.vertices[:, 2], 0.0)


def test_vertex_size_mismatch_raises(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="triangle" dim="3">'
        '<vertices size="5">'
        '<vertex index="0" x="0" y="0" z="0"/>'
        '<vertex index="1" x="1" y="0" z="0"/>'
        "</vertices>"
        '<cells size="0"/></mesh></dolfin>'
    )
    f = tmp_path / "size_mismatch.xml"
    f.write_text(xml)
    with pytest.raises(CodecError, match="size"):
        read(f)


def test_nonsequential_vertex_index_raises(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="triangle" dim="2">'
        '<vertices size="3">'
        '<vertex index="0" x="0" y="0" z="0"/>'
        '<vertex index="7" x="1" y="0" z="0"/>'
        '<vertex index="2" x="0" y="1" z="0"/>'
        "</vertices>"
        '<cells size="0"/></mesh></dolfin>'
    )
    f = tmp_path / "nonseq_vert.xml"
    f.write_text(xml)
    with pytest.raises(CodecError, match="non-sequential"):
        read(f)


def test_nonsequential_cell_index_raises(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="triangle" dim="2">'
        '<vertices size="3">'
        '<vertex index="0" x="0" y="0" z="0"/>'
        '<vertex index="1" x="1" y="0" z="0"/>'
        '<vertex index="2" x="0" y="1" z="0"/>'
        "</vertices>"
        '<cells size="1"><triangle index="5" v0="0" v1="1" v2="2"/></cells>'
        "</mesh></dolfin>"
    )
    f = tmp_path / "nonseq.xml"
    f.write_text(xml)
    with pytest.raises(CodecError, match="non-sequential"):
        read(f)


def test_lazy_warns(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.xml"
    write(poly, out)
    with pytest.warns(UserWarning, match="lazy"):
        read(out, lazy=True)


def test_missing_cell_attr_raises(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?>'
        '<dolfin><mesh celltype="triangle" dim="2">'
        '<vertices size="3">'
        '<vertex index="0" x="0" y="0" z="0"/>'
        '<vertex index="1" x="1" y="0" z="0"/>'
        '<vertex index="2" x="0" y="1" z="0"/>'
        "</vertices>"
        '<cells size="1"><triangle index="0" v0="0" v1="1"/></cells>'
        "</mesh></dolfin>"
    )
    bad = tmp_path / "no_v2.xml"
    bad.write_text(xml)
    with pytest.raises(CodecError, match="missing 'v2'"):
        read(bad)


def test_dim_explicit(tmp_path: Path) -> None:
    poly = _flat_tri_mesh()
    out = tmp_path / "flat.xml"
    write(poly, out, dim=3)
    tree = ET.parse(out)
    assert tree.getroot().find("mesh").get("dim") == "3"


def test_dim_inferred_2d(tmp_path: Path) -> None:
    poly = _flat_tri_mesh()
    out = tmp_path / "flat.xml"
    write(poly, out)
    tree = ET.parse(out)
    assert tree.getroot().find("mesh").get("dim") == "2"


def test_dim_inferred_3d(tmp_path: Path) -> None:
    poly = _tet_mesh()
    out = tmp_path / "tet.xml"
    write(poly, out)
    tree = ET.parse(out)
    assert tree.getroot().find("mesh").get("dim") == "3"


def test_mixed_type_write_warns(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    out = tmp_path / "mixed.xml"
    with pytest.warns(UserWarning, match="skipped"):
        write(poly, out)
    poly2 = read(out)
    assert len(poly2.element_types) == 1


def test_a_gmsh_mesh_with_physical_groups_writes(tmp_path: Path) -> None:
    """A DOLFIN XML holds one cell type; a mixed mesh must not fail the write."""
    import polyxios

    msh = tmp_path / "g.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n2\n2 1 "surface"\n3 2 "volume"\n$EndPhysicalNames\n'
        "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
        "$Elements\n2\n1 2 2 1 1 1 2 3\n2 4 2 2 2 1 2 3 4\n$EndElements\n"
    )
    poly = polyxios.read(msh)
    assert sorted(poly.element_tags) == ["surface", "volume"]

    out = tmp_path / "d.xml"
    # The surface triangle has no place beside the tetrahedron; it is reported
    # rather than turning the write into an error.
    with pytest.warns(UserWarning, match="skipped"):
        write(poly, out)
    back = read(out)
    assert back.vertices.shape == (4, 3)
    assert len(back.element_types) == 1


def test_a_single_type_gmsh_mesh_writes_without_a_word(
    tmp_path: Path,
) -> None:
    import polyxios

    msh = tmp_path / "tri.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n1\n2 1 "surface"\n$EndPhysicalNames\n'
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 1 1 1 2 3\n$EndElements\n"
    )
    out = tmp_path / "tri.xml"
    write(polyxios.read(msh), out)
    assert len(read(out).element_types) == 1


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b'<?xml version="1.0"?>\n<dolfin nsmap="">\n<mesh celltype="interval">', True),
        (
            b"\xef\xbb\xbf<!-- exported -->\n<!DOCTYPE dolfin>\n<dolfin>"
            b"<mesh dim='3' celltype='tetrahedron'>",
            True,
        ),
        (b'<mesh celltype="triangle" dim="2">', True),
        (
            b'<!DOCTYPE dolfin [\n  <!ENTITY v "vertex">\n]>\n<dolfin>'
            b'<mesh celltype="triangle">',
            True,
        ),
        (b'<dolfin>\n  <mesh_function type="uint" dim="2" size="4">', False),
        (b"<dolfin>", False),
        (b'<mesh sharedvertexbuffers="true">\n  <submeshes>', False),
        (b"<!DOCTYPE config [<!ENTITY a 'b'>]><config/>", False),
        (b'<?xml version="1.0"?>\n<config><mesh/></config>', False),
        (b"<meshes>", False),
        (b"", False),
    ],
)
def test_sniff_reads_the_root_element(head, expected):
    assert sniff(head) is expected


def test_sniff_accepts_what_write_produces(tmp_path):
    out = tmp_path / "m.xml"
    write(_tet_mesh(), out)
    assert sniff(out.read_bytes())


def _interval_xml(*, encoding: str = "utf-8", v1: str = "1", index: str = "0") -> bytes:
    return (
        f"<?xml version='1.0' encoding='{encoding}'?>\n"
        '<dolfin><mesh celltype="interval" dim="1">'
        '<vertices size="2"><vertex index="0" x="0" y="0"/>'
        '<vertex index="1" x="1" y="0"/></vertices>'
        f'<cells size="1"><interval index="{index}" v0="0" v1="{v1}"/></cells>'
        "</mesh></dolfin>"
    ).encode()


@pytest.mark.parametrize("encoding", ["utf-9", "utf-16"])
def test_an_encoding_the_parser_cannot_use_is_refused(tmp_path, encoding):
    """expat raised LookupError for an unknown encoding and ValueError for a
    multi-byte one, both past the ParseError the reader caught."""
    out = tmp_path / "m.xml"
    out.write_bytes(_interval_xml(encoding=encoding))
    with pytest.raises(CodecError, match="encoding"):
        read(out)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"index": ",3"}, "cell index ',3' is not an integer"),
        ({"v1": "x"}, "cell 0 v1 'x' is not an integer"),
        ({"v1": "7"}, "v1=7 is not one of the 2 vertices"),
        ({"v1": "-1"}, "v1=-1 is not one of the 2 vertices"),
        ({"v1": "7" * 40}, "is not one of the 2 vertices"),
    ],
)
def test_a_cell_that_names_no_vertex_is_refused(tmp_path, kwargs, match):
    """A cell index past the vertices was handed back in the connectivity, and
    one past int64 overflowed while the array was built."""
    out = tmp_path / "m.xml"
    out.write_bytes(_interval_xml(**kwargs))
    with pytest.raises(CodecError, match=match):
        read(out)


def test_a_coordinate_that_is_not_a_number_is_refused(tmp_path):
    out = tmp_path / "m.xml"
    out.write_bytes(_interval_xml().replace(b' x="1"', b' x="1,5"'))
    with pytest.raises(CodecError, match="vertex 1 x '1,5' is not a number"):
        read(out)


@pytest.mark.parametrize(
    ("dim", "was_2d"),
    [("2", True), (" 2 ", True), ("9" * 5000, True), ("２", True), ("3", False)],
)
def test_a_dim_that_is_not_one_digit_is_read_off_the_vertices(
    tmp_path: Path, dim: str, was_2d: bool
) -> None:
    """Five thousand digits passed isdigit() and int() refused them."""
    f = tmp_path / "dim.xml"
    f.write_text(
        f'<dolfin><mesh celltype="triangle" dim="{dim}"><vertices size="3">'
        '<vertex index="0" x="0" y="0"/><vertex index="1" x="1" y="0"/>'
        '<vertex index="2" x="0" y="1"/></vertices><cells size="1">'
        '<triangle index="0" v0="0" v1="1" v2="2"/></cells></mesh></dolfin>',
        encoding="utf-8",
    )
    assert read(f).global_attrs.get("was_2d", False) is was_2d
