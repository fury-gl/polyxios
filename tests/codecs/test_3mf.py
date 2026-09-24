"""3MF: a ZIP package around an XML model of triangles.

Files are built with ``zipfile`` and hand-written XML, in the layout the
specification fixes, so the reader is tested against the format and not
against the writer beside it.
"""

from __future__ import annotations

import io
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._types import PolyData
from polyxios.codecs._3mf import _CORE_NS, _MODEL_REL_TYPE, read, write
from polyxios.exceptions import CodecError, LazyReadError

_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
_TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{_MODEL_REL_TYPE}"/>'
    "</Relationships>"
)
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    "</Types>"
)


def _tet_mesh_xml() -> str:
    verts = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in _TET.tolist())
    tris = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in _TET_FACES.tolist()
    )
    return f"<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh>"


def _model(
    resources: str,
    build: str,
    *,
    unit: str = "millimeter",
    metadata: str = "",
    ns: str = _CORE_NS,
) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="{unit}" xml:lang="en-US" xmlns="{ns}">'
        f"{metadata}<resources>{resources}</resources><build>{build}</build></model>"
    )


def _package(
    path: Path,
    model: str,
    *,
    model_part: str = "3D/3dmodel.model",
    rels: str | None = _RELS,
) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        if rels is not None:
            z.writestr("_rels/.rels", rels)
        z.writestr(model_part, model)
    return path


def _tet_file(tmp_path: Path, **kwargs) -> Path:
    resources = f'<object id="1" type="model" name="tet">{_tet_mesh_xml()}</object>'
    return _package(
        tmp_path / "tet.3mf", _model(resources, '<item objectid="1"/>', **kwargs)
    )


def _tet_poly(**kwargs) -> PolyData:
    return make_polydata(_TET, [("triangle", _TET_FACES)], **kwargs)


def _read_quietly(path) -> PolyData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return polyxios.read(path)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_reads_a_single_object(tmp_path: Path) -> None:
    mesh = read(_tet_file(tmp_path))
    np.testing.assert_array_equal(mesh.vertices, _TET)
    np.testing.assert_array_equal(mesh.connectivity.reshape(-1, 3), _TET_FACES)
    assert set(mesh.element_types.tolist()) == {5}
    np.testing.assert_array_equal(mesh.element_tags["tet"], [0, 1, 2, 3])
    assert mesh.global_attrs == {"unit": "millimeter"}
    assert "colors" not in mesh.element_attrs


def test_unit_and_metadata_are_global_attrs(tmp_path: Path) -> None:
    path = _tet_file(
        tmp_path,
        unit="inch",
        metadata='<metadata name="Title">bracket</metadata>'
        '<metadata name="Designer">me</metadata>',
    )
    mesh = read(path)
    assert mesh.global_attrs == {"unit": "inch", "Title": "bracket", "Designer": "me"}


def test_an_unnamed_object_is_tagged_by_its_id(tmp_path: Path) -> None:
    resources = f'<object id="7" type="model">{_tet_mesh_xml()}</object>'
    path = _package(tmp_path / "a.3mf", _model(resources, '<item objectid="7"/>'))
    assert list(read(path).element_tags) == ["object_7"]


def test_a_namespace_prefix_is_read_like_the_default_namespace(tmp_path: Path) -> None:
    mesh_xml = _tet_mesh_xml().replace("<", "<c:").replace("<c:/", "</c:")
    model = (
        '<?xml version="1.0"?>'
        f'<c:model unit="millimeter" xmlns:c="{_CORE_NS}"><c:resources>'
        f'<c:object id="1" type="model" name="p">{mesh_xml}</c:object>'
        '</c:resources><c:build><c:item objectid="1"/></c:build></c:model>'
    )
    mesh = read(_package(tmp_path / "a.3mf", model))
    assert len(mesh.element_types) == 4
    assert list(mesh.element_tags) == ["p"]


def test_a_build_item_transform_is_applied(tmp_path: Path) -> None:
    resources = f'<object id="1" type="model" name="tet">{_tet_mesh_xml()}</object>'
    # Scale x by 2, then translate by (10, 20, 30).
    build = '<item objectid="1" transform="2 0 0 0 1 0 0 0 1 10 20 30"/>'
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, build)))
    expected = _TET * [2, 1, 1] + [10, 20, 30]
    np.testing.assert_allclose(mesh.vertices, expected)


def test_components_compose_their_transforms(tmp_path: Path) -> None:
    resources = (
        f'<object id="1" type="model" name="tet">{_tet_mesh_xml()}</object>'
        '<object id="2" type="model" name="pair"><components>'
        '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0"/>'
        '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 0 5 0"/>'
        "</components></object>"
    )
    build = '<item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 100"/>'
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, build)))
    assert mesh.vertices.shape == (8, 3)
    np.testing.assert_allclose(mesh.vertices[:4], _TET + [5, 0, 100])
    np.testing.assert_allclose(mesh.vertices[4:], _TET + [0, 5, 100])
    # The second placement's triangles index the second copy's vertices.
    np.testing.assert_array_equal(mesh.connectivity[12:].reshape(-1, 3), _TET_FACES + 4)
    # Both placements are the one part, so they share its tag.
    assert list(mesh.element_tags) == ["tet"]
    np.testing.assert_array_equal(mesh.element_tags["tet"], np.arange(8))


def test_a_component_transform_is_applied_before_the_item_transform(
    tmp_path: Path,
) -> None:
    """The component moves in the parent's frame, then the item rotates the parent."""
    resources = (
        f'<object id="1" type="model" name="tet">{_tet_mesh_xml()}</object>'
        '<object id="2" type="model" name="pair"><components>'
        '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0"/>'
        "</components></object>"
    )
    # x maps to +y: the translation along x lands along y, not along x.
    build = '<item objectid="2" transform="0 1 0 -1 0 0 0 0 1 0 0 0"/>'
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, build)))
    rotated = _TET @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
    np.testing.assert_allclose(mesh.vertices, rotated + [0, 5, 0], atol=1e-12)


def test_a_rotation_is_applied_in_row_vector_order(tmp_path: Path) -> None:
    """The matrix rows are the images of the axes: x maps to +y here."""
    resources = f'<object id="1" type="model">{_tet_mesh_xml()}</object>'
    build = '<item objectid="1" transform="0 1 0 -1 0 0 0 0 1 0 0 0"/>'
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, build)))
    np.testing.assert_allclose(mesh.vertices[1], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[2], [-1, 0, 0], atol=1e-12)


def test_a_mirroring_transform_reverses_the_winding(tmp_path: Path) -> None:
    resources = f'<object id="1" type="model">{_tet_mesh_xml()}</object>'
    build = '<item objectid="1" transform="-1 0 0 0 1 0 0 0 1 0 0 0"/>'
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, build)))
    np.testing.assert_array_equal(
        mesh.connectivity.reshape(-1, 3), _TET_FACES[:, [0, 2, 1]]
    )


def test_an_object_not_in_the_build_is_not_read(tmp_path: Path) -> None:
    resources = (
        f'<object id="1" type="model" name="built">{_tet_mesh_xml()}</object>'
        f'<object id="2" type="other" name="spare">{_tet_mesh_xml()}</object>'
    )
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, '<item objectid="1"/>')))
    assert list(mesh.element_tags) == ["built"]
    assert len(mesh.element_types) == 4


def test_base_materials_are_read_as_colours(tmp_path: Path) -> None:
    resources = (
        '<basematerials id="1"><base name="Red" displaycolor="#FF0000"/>'
        '<base name="Glass" displaycolor="#0000FF80"/></basematerials>'
        f'<object id="2" type="model" pid="1" pindex="0">{_tet_mesh_xml()}</object>'
    )
    model = _model(resources, '<item objectid="2"/>').replace(
        '<triangle v1="1" v2="2" v3="3"/>',
        '<triangle v1="1" v2="2" v3="3" pid="1" p1="1"/>',
    )
    mesh = read(_package(tmp_path / "a.3mf", model))
    colors = mesh.element_attrs["colors"]
    assert colors.shape == (4, 4)
    np.testing.assert_allclose(colors[0], [1, 0, 0, 1])
    np.testing.assert_allclose(colors[3], [0, 0, 1, 128 / 255])


def test_a_triangle_p1_without_pid_indexes_the_object_group(tmp_path: Path) -> None:
    """pid and p1 fall back to the object's pid and pindex each on its own."""
    resources = (
        '<basematerials id="1"><base name="Red" displaycolor="#FF0000"/>'
        '<base name="Blue" displaycolor="#0000FF"/></basematerials>'
        f'<object id="2" type="model" pid="1" pindex="0">{_tet_mesh_xml()}</object>'
    )
    model = _model(resources, '<item objectid="2"/>').replace(
        '<triangle v1="1" v2="2" v3="3"/>', '<triangle v1="1" v2="2" v3="3" p1="1"/>'
    )
    colors = read(_package(tmp_path / "a.3mf", model)).element_attrs["colors"]
    np.testing.assert_allclose(colors[0], [1, 0, 0, 1])
    np.testing.assert_allclose(colors[3], [0, 0, 1, 1])


def test_a_colour_group_is_read_as_colours(tmp_path: Path) -> None:
    mat = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    resources = (
        f'<m:colorgroup xmlns:m="{mat}" id="1"><m:color color="#00FF00"/></m:colorgroup>'
        f'<object id="2" type="model">{_tet_mesh_xml()}</object>'
    )
    model = _model(resources, '<item objectid="2"/>').replace(
        '<triangle v1="0" v2="2" v3="1"/>',
        '<triangle v1="0" v2="2" v3="1" pid="1" p1="0"/>',
    )
    mesh = read(_package(tmp_path / "a.3mf", model))
    colors = mesh.element_attrs["colors"]
    np.testing.assert_allclose(colors[0], [0, 1, 0, 1])
    assert np.isnan(colors[1:]).all()


def test_a_texture_property_is_no_colour(tmp_path: Path) -> None:
    mat = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    resources = (
        f'<m:texture2dgroup xmlns:m="{mat}" id="1" texid="9"><m:tex2coord u="0" v="0"/>'
        "</m:texture2dgroup>"
        f'<object id="2" type="model" pid="1" pindex="0">{_tet_mesh_xml()}</object>'
    )
    mesh = read(_package(tmp_path / "a.3mf", _model(resources, '<item objectid="2"/>')))
    assert "colors" not in mesh.element_attrs


def test_the_relationships_name_the_model_part(tmp_path: Path) -> None:
    resources = f'<object id="1" type="model">{_tet_mesh_xml()}</object>'
    rels = _RELS.replace("/3D/3dmodel.model", "/3D/custom.model")
    path = _package(
        tmp_path / "a.3mf",
        _model(resources, '<item objectid="1"/>'),
        model_part="3D/custom.model",
        rels=rels,
    )
    assert len(read(path).element_types) == 4


def test_a_percent_encoded_model_part_name_is_decoded(tmp_path: Path) -> None:
    resources = f'<object id="1" type="model">{_tet_mesh_xml()}</object>'
    rels = _RELS.replace("/3D/3dmodel.model", "/3D/my%20model.model")
    path = _package(
        tmp_path / "a.3mf",
        _model(resources, '<item objectid="1"/>'),
        model_part="3D/my model.model",
        rels=rels,
    )
    assert len(read(path).element_types) == 4


def test_a_package_without_relationships_uses_the_conventional_part(
    tmp_path: Path,
) -> None:
    resources = f'<object id="1" type="model">{_tet_mesh_xml()}</object>'
    path = _package(
        tmp_path / "a.3mf", _model(resources, '<item objectid="1"/>'), rels=None
    )
    assert len(read(path).element_types) == 4


def test_an_empty_build_reads_as_an_empty_mesh(tmp_path: Path) -> None:
    mesh = read(_package(tmp_path / "a.3mf", _model("", "")))
    assert mesh.vertices.shape == (0, 3)
    assert len(mesh.element_types) == 0
    assert mesh.global_attrs == {"unit": "millimeter"}


def test_reads_from_a_buffer(tmp_path: Path) -> None:
    raw = _tet_file(tmp_path).read_bytes()
    mesh = polyxios.read(io.BytesIO(raw), fmt=".3mf")
    assert len(mesh.element_types) == 4


def test_lazy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_tet_file(tmp_path), lazy=True)


@pytest.mark.parametrize(
    ("resources", "build", "message"),
    [
        (
            '<object id="1" type="model"><mesh><vertices><vertex x="0" y="0" z="0"/>'
            '</vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh>'
            "</object>",
            '<item objectid="1"/>',
            "names vertex 2 but holds 1",
        ),
        ("", '<item objectid="4"/>', "names object 4"),
        (
            '<object id="1" type="model"><components><component objectid="1"/>'
            "</components></object>",
            '<item objectid="1"/>',
            "contains itself",
        ),
        (
            '<object id="1" type="model"><components><component objectid="2"/>'
            '</components></object><object id="2" type="model"><components>'
            '<component objectid="1"/></components></object>',
            '<item objectid="1"/>',
            "1 -> 2 -> 1 contains itself",
        ),
        (
            '<object id="1" type="model"><mesh><vertices/><triangles/></mesh></object>',
            '<item objectid="1" transform="1 0 0"/>',
            "12 numbers",
        ),
        (
            '<object id="1" type="model"><mesh><vertices><vertex x="a" y="0" z="0"/>'
            "</vertices><triangles/></mesh></object>",
            '<item objectid="1"/>',
            "not three numbers",
        ),
        (
            '<basematerials id="1"><base name="x" displaycolor="red"/></basematerials>',
            "",
            "not an sRGB colour",
        ),
        (
            '<object id="1" type="model"><mesh><vertices><vertex x="0" y="0" z="0"/>'
            '<vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>'
            '<triangles><triangle v1="0" v2="1" v3="2" pid="9" p1="0"/></triangles>'
            "</mesh></object>",
            '<item objectid="1"/>',
            "property group 9",
        ),
        (
            '<basematerials id="1"><base name="x" displaycolor="#FF0000"/></basematerials>'
            '<object id="2" type="model"><mesh><vertices><vertex x="0" y="0" z="0"/>'
            '<vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>'
            '<triangles><triangle v1="0" v2="1" v3="2" pid="1" p1="3"/></triangles>'
            "</mesh></object>",
            '<item objectid="2"/>',
            "holds 1 entries; p1=3 names none",
        ),
        (
            '<basematerials id="1"><base name="x" displaycolor="#FF0000"/></basematerials>'
            '<object id="2" type="model"><mesh><vertices><vertex x="0" y="0" z="0"/>'
            '<vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>'
            '<triangles><triangle v1="0" v2="1" v3="2" pid="1" p1="-1"/></triangles>'
            "</mesh></object>",
            '<item objectid="2"/>',
            "holds 1 entries; p1=-1 names none",
        ),
        ('<object type="model"/>', "", "an <object> id is missing"),
        (
            '<object id="1" type="model"><mesh><vertices/><triangles/></mesh></object>'
            '<object id="1" type="model"><mesh><vertices/><triangles/></mesh></object>',
            "",
            "gives two resources id 1",
        ),
        (
            '<basematerials id="1"><base name="x" displaycolor="#FF0000"/></basematerials>'
            '<object id="1" type="model"><mesh><vertices/><triangles/></mesh></object>',
            "",
            "gives two resources id 1",
        ),
    ],
)
def test_a_broken_model_is_refused_by_name(
    tmp_path: Path, resources: str, build: str, message: str
) -> None:
    path = _package(tmp_path / "broken.3mf", _model(resources, build))
    with pytest.raises(CodecError, match=message):
        read(path)


def test_a_file_that_is_not_a_zip_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "text.3mf"
    path.write_bytes(b"solid nothing\n")
    with pytest.raises(CodecError, match="not a ZIP"):
        read(path)


def test_a_package_without_a_model_part_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.3mf"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
    with pytest.raises(CodecError, match="no model part"):
        read(path)


def test_a_model_that_is_not_xml_is_refused(tmp_path: Path) -> None:
    path = _package(tmp_path / "bad.3mf", "<model><resources>")
    with pytest.raises(CodecError, match="not well-formed"):
        read(path)


def test_a_model_part_that_is_not_a_model_is_refused(tmp_path: Path) -> None:
    path = _package(tmp_path / "bad.3mf", "<Relationships/>")
    with pytest.raises(CodecError, match="<Relationships> where a <model>"):
        read(path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_round_trip_keeps_geometry_and_order(tmp_path: Path) -> None:
    path = tmp_path / "out.3mf"
    write(_tet_poly(), path)
    back = read(path)
    np.testing.assert_array_equal(back.vertices, _TET)
    np.testing.assert_array_equal(back.connectivity.reshape(-1, 3), _TET_FACES)
    assert list(back.element_tags) == ["object_1"]


def test_the_package_has_the_parts_the_specification_requires(tmp_path: Path) -> None:
    path = tmp_path / "out.3mf"
    write(_tet_poly(), path)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        rels = z.read("_rels/.rels").decode()
        model = z.read("3D/3dmodel.model").decode()
    assert names == {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
    assert _MODEL_REL_TYPE in rels
    assert f'xmlns="{_CORE_NS}"' in model
    assert 'unit="millimeter"' in model


def test_two_writes_of_one_mesh_are_byte_identical(tmp_path: Path) -> None:
    a, b = io.BytesIO(), io.BytesIO()
    write(_tet_poly(), a)
    write(_tet_poly(), b)
    assert a.getvalue() == b.getvalue()


def test_coordinates_survive_to_the_last_digit(tmp_path: Path) -> None:
    verts = np.array([[0.1, 0.2, 0.3], [1 / 3, 2 / 3, 1e-17], [np.pi, -np.e, 1e20]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    path = tmp_path / "out.3mf"
    write(poly, path)
    np.testing.assert_array_equal(read(path).vertices, verts)


def test_quads_are_split_into_triangles(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "out.3mf"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(
        back.connectivity.reshape(-1, 3), [[0, 1, 2], [0, 2, 3]]
    )


def test_polygons_strips_and_pixels_are_split_into_triangles(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("polygon", np.array([[0, 1, 2, 3, 4, 5]])),
            ("triangle_strip", np.array([[0, 1, 5, 4]])),
            ("pixel", np.array([[0, 1, 5, 4]])),
        ],
    )
    path = tmp_path / "out.3mf"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    back = read(path).connectivity.reshape(-1, 3)
    expected = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5],
        [0, 1, 5], [5, 1, 4],
        [0, 1, 4], [0, 4, 5],
    ]  # fmt: skip
    np.testing.assert_array_equal(back, expected)


def test_mixed_element_types_keep_their_order(tmp_path: Path) -> None:
    """A triangle, a quad, a triangle: the quad's two halves sit between them."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 3]])),
            ("quad", np.array([[1, 4, 5, 2]])),
            ("triangle", np.array([[1, 2, 3]])),
        ],
        element_attrs={
            "colors": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        },
    )
    path = tmp_path / "out.3mf"
    write(poly, path)
    back = read(path)
    expected = [[0, 1, 3], [1, 4, 5], [1, 5, 2], [1, 2, 3]]
    np.testing.assert_array_equal(
        verts[back.connectivity.reshape(-1, 3)], verts[expected]
    )
    np.testing.assert_allclose(
        back.element_attrs["colors"][:, :3],
        [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
    )


def test_a_vertex_that_is_not_finite_is_refused(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]])
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    with pytest.raises(CodecError, match="NaN or infinite"):
        write(poly, tmp_path / "out.3mf")


def test_a_quadratic_triangle_keeps_its_corners(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts, [("quadratic_triangle", np.array([[0, 1, 2, 3, 4, 5]]))]
    )
    path = tmp_path / "out.3mf"
    write(poly, path)
    back = read(path)
    assert back.vertices.shape == (3, 3)
    np.testing.assert_array_equal(back.connectivity, [0, 1, 2])


def test_volume_elements_are_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("triangle", _TET_FACES[:1]), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    path = tmp_path / "out.3mf"
    with pytest.warns(UserWarning, match=r"1 tetra element\(s\) .* dropped"):
        write(poly, path)
    assert len(read(path).element_types) == 1


def test_a_mesh_with_no_surface_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_TET, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(CodecError, match="has none: 1 tetra"):
        write(poly, tmp_path / "out.3mf")


def test_an_empty_mesh_writes_and_reads_back_empty(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=np.empty((0, 3)),
        connectivity=np.array([], dtype=np.int32),
        offsets=np.zeros(1, dtype=np.int32),
        element_types=np.array([], dtype=np.uint8),
    )
    path = tmp_path / "out.3mf"
    write(poly, path)
    assert len(read(path).element_types) == 0


def test_only_the_vertices_an_object_uses_are_written(tmp_path: Path) -> None:
    verts = np.vstack([_TET, [[9, 9, 9]]])
    poly = make_polydata(verts, [("triangle", _TET_FACES)])
    path = tmp_path / "out.3mf"
    write(poly, path)
    np.testing.assert_array_equal(read(path).vertices, _TET)


def test_tag_groups_become_named_objects(tmp_path: Path) -> None:
    poly = _tet_poly(
        element_tags={
            "top": np.array([0, 1], dtype=np.int32),
            "bottom": np.array([2, 3], dtype=np.int32),
        }
    )
    path = tmp_path / "out.3mf"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    back = read(path)
    assert list(back.element_tags) == ["top", "bottom"]
    np.testing.assert_array_equal(back.element_tags["top"], [0, 1])
    np.testing.assert_array_equal(back.element_tags["bottom"], [2, 3])
    # Each object carries its own vertex list, so the shared corners repeat.
    assert back.vertices.shape == (8, 3)
    with zipfile.ZipFile(path) as z:
        model = z.read("3D/3dmodel.model").decode()
    assert 'name="top"' in model and 'name="bottom"' in model


def test_an_element_in_two_groups_stays_with_the_first(tmp_path: Path) -> None:
    poly = _tet_poly(
        element_tags={
            "a": np.array([0, 1], dtype=np.int32),
            "b": np.array([1, 2], dtype=np.int32),
        }
    )
    path = tmp_path / "out.3mf"
    with pytest.warns(
        UserWarning, match=r"group\(s\) \['b'\] name elements an earlier"
    ):
        write(poly, path)
    back = read(path)
    assert list(back.element_tags) == ["a", "b", "object_3"]
    assert len(back.element_tags["a"]) == 2
    assert len(back.element_tags["b"]) == 1
    assert len(back.element_tags["object_3"]) == 1


def test_a_group_left_with_nothing_is_not_an_object(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET,
        [("triangle", _TET_FACES[:1]), ("tetra", np.array([[0, 1, 2, 3]]))],
        element_tags={"solid": np.array([1], dtype=np.int32)},
    )
    path = tmp_path / "out.3mf"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write(poly, path)
    assert list(read(path).element_tags) == ["object_1"]


def test_an_object_name_is_escaped(tmp_path: Path) -> None:
    poly = _tet_poly(element_tags={'a<b>&"c"': np.arange(4, dtype=np.int32)})
    path = tmp_path / "out.3mf"
    write(poly, path)
    assert list(read(path).element_tags) == ['a<b>&"c"']


def test_a_name_with_a_control_character_is_refused_on_write(tmp_path: Path) -> None:
    """XML cannot carry it, so the writer says so rather than leave a broken file."""
    poly = _tet_poly(element_tags={"a\x01b": np.arange(4, dtype=np.int32)})
    with pytest.raises(CodecError, match="tag group name 'a\\\\x01b' holds a control"):
        write(poly, tmp_path / "out.3mf")
    poly = _tet_poly(global_attrs={"Title": "a\x00b"})
    with pytest.raises(CodecError, match="Title 'a\\\\x00b' holds a control"):
        write(poly, tmp_path / "out.3mf")


def test_a_tag_named_like_the_untagged_object_is_warned_about(tmp_path: Path) -> None:
    """The rest goes out as object 2, which reads back as 'object_2' too."""
    poly = _tet_poly(element_tags={"object_2": np.array([0, 1], dtype=np.int32)})
    path = tmp_path / "out.3mf"
    with pytest.warns(UserWarning, match="reads back as 'object_2'"):
        write(poly, path)
    np.testing.assert_array_equal(read(path).element_tags["object_2"], [0, 1, 2, 3])


@pytest.mark.parametrize(
    ("colors", "expected"),
    [
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64),
            None,
        ),
        (
            np.array(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]], dtype=np.uint8
            ),
            None,
        ),
        (
            np.array([[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0.5]]),
            "rgba",
        ),
    ],
)
def test_colours_round_trip_through_base_materials(
    tmp_path: Path, colors: np.ndarray, expected: str | None
) -> None:
    poly = _tet_poly(element_attrs={"colors": colors})
    path = tmp_path / "out.3mf"
    write(poly, path)
    back = read(path).element_attrs["colors"]
    if expected == "rgba":
        np.testing.assert_allclose(back, colors, atol=1 / 255)
    else:
        rgb = colors.astype(np.float64) / (255 if colors.dtype.kind == "u" else 1)
        np.testing.assert_allclose(back[:, :3], rgb, atol=1 / 255)
        np.testing.assert_array_equal(back[:, 3], 1.0)
    with zipfile.ZipFile(path) as z:
        model = z.read("3D/3dmodel.model").decode()
    # One base per distinct colour, not one per triangle.
    assert model.count("<base ") == 3


def test_a_triangle_without_a_colour_names_no_material(tmp_path: Path) -> None:
    colors = np.array([[1, 0, 0, 1], [np.nan] * 4, [0, 0, 1, 1], [np.nan] * 4])
    poly = _tet_poly(element_attrs={"colors": colors})
    path = tmp_path / "out.3mf"
    write(poly, path)
    back = read(path).element_attrs["colors"]
    np.testing.assert_allclose(back[[0, 2]], colors[[0, 2]])
    assert np.isnan(back[[1, 3]]).all()


def test_a_colour_that_is_not_a_colour_is_not_written(tmp_path: Path) -> None:
    poly = _tet_poly(element_attrs={"colors": np.arange(4, dtype=np.float64)})
    path = tmp_path / "out.3mf"
    with pytest.warns(UserWarning, match="three or four components"):
        write(poly, path)
    assert "colors" not in read(path).element_attrs


def test_a_split_quad_keeps_its_colour_on_both_halves(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    poly = make_polydata(
        verts,
        [("quad", np.array([[0, 1, 2, 3]]))],
        element_attrs={"colors": np.array([[0, 1, 0]], dtype=np.float64)},
    )
    path = tmp_path / "out.3mf"
    write(poly, path)
    back = read(path).element_attrs["colors"]
    np.testing.assert_allclose(back, [[0, 1, 0, 1], [0, 1, 0, 1]])


def test_the_unit_is_taken_from_the_argument_then_the_mesh(tmp_path: Path) -> None:
    path = tmp_path / "out.3mf"
    write(_tet_poly(global_attrs={"unit": "inch"}), path)
    assert read(path).global_attrs["unit"] == "inch"
    write(_tet_poly(global_attrs={"unit": "inch"}), path, unit="meter")
    assert read(path).global_attrs["unit"] == "meter"


def test_a_unit_the_format_lacks_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="'furlong' is not a 3MF unit"):
        write(_tet_poly(), tmp_path / "out.3mf", unit="furlong")


def test_known_metadata_is_written_and_the_rest_left_out(tmp_path: Path) -> None:
    poly = _tet_poly(
        global_attrs={"Title": "a <part>", "Designer": "me", "gnum": 42, "Rating": 5}
    )
    path = tmp_path / "out.3mf"
    write(poly, path)
    back = read(path).global_attrs
    assert back == {
        "unit": "millimeter",
        "Title": "a <part>",
        "Designer": "me",
        "Rating": "5",
    }


def test_writes_to_a_buffer(tmp_path: Path) -> None:
    buf = io.BytesIO()
    polyxios.write(_tet_poly(), buf, fmt=".3mf")
    assert zipfile.is_zipfile(io.BytesIO(buf.getvalue()))
    buf.seek(0)
    assert len(polyxios.read(buf, fmt=".3mf").element_types) == 4
