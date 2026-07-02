from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from polyxios import make_polydata
from polyxios.codecs._dolfin import read, write
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
