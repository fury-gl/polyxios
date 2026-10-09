"""COLLADA: the XML scene format, read and written as a SceneData.

Every file is spelled inline by the builders at the top so a test names
exactly the element it exercises; the reference-implementation checks at
the bottom need pycollada and skip without it.

The two reference-implementation tests silence warnings wholesale: pycollada
0.8 sets an array's shape in place and uses np.matrix, both of which numpy 2.5
warns about on every call.
"""

from __future__ import annotations

import base64
import dataclasses
from datetime import datetime
import io
import json
from pathlib import Path
import re
import tracemalloc
from typing import Any
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import polyxios
from polyxios import PolyData, SceneData, SceneMaterial, SceneNode, make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._scene import SceneImage, SceneTexture
from polyxios.codecs import _collada
from polyxios.codecs._collada import read, read_scene, write, write_scene
from polyxios.codecs._gltf import (
    read_scene as gltf_read_scene,
    write_scene as gltf_write_scene,
)
from polyxios.exceptions import CodecError, LazyReadError
from tests.codecs.test_gltf import _skinned_glb

_NS = "http://www.collada.org/2005/11/COLLADASchema"
_PARAM_TYPE = {"TRANSFORM": "float4x4"}

_TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_QUAD = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def _nums(values) -> str:
    return " ".join(str(v) for v in np.asarray(values).ravel().tolist())


def _source(sid: str, values, params: str, *, kind: str = "float_array") -> str:
    arr = np.asarray(values)
    stride = arr.shape[1] if arr.ndim == 2 else 1
    count = arr.shape[0] if arr.ndim == 2 else len(arr)
    text = _nums(values) if kind == "float_array" else " ".join(values)
    names = "".join(
        f'<param name="{p}" type="{_PARAM_TYPE.get(p, "Name" if kind == "Name_array" else "float")}"/>'
        for p in params.split()
    )
    return (
        f'<source id="{sid}"><{kind} id="{sid}-array" count="{count * stride}">'
        f"{text}</{kind}><technique_common>"
        f'<accessor source="#{sid}-array" count="{count}" stride="{stride}">'
        f"{names}</accessor></technique_common></source>"
    )


def _geometry(gid: str, positions, prims: str, *, sources: str = "", name=None) -> str:
    nm = f' name="{name}"' if name else ""
    return (
        f'<geometry id="{gid}"{nm}><mesh>'
        + _source(f"{gid}-pos", positions, "X Y Z")
        + sources
        + f'<vertices id="{gid}-vtx"><input semantic="POSITION" source="#{gid}-pos"/>'
        "</vertices>" + prims + "</mesh></geometry>"
    )


def _triangles(gid: str, faces, *, material=None, extra_inputs: str = "") -> str:
    faces = np.asarray(faces)
    mat = f' material="{material}"' if material else ""
    return (
        f'<triangles count="{len(faces)}"{mat}>'
        f'<input semantic="VERTEX" source="#{gid}-vtx" offset="0"/>'
        + extra_inputs
        + f"<p>{_nums(faces)}</p></triangles>"
    )


def _scene_with(gid: str, *, node_extra: str = "", bind: str = "") -> str:
    return (
        '<library_visual_scenes><visual_scene id="Scene" name="Scene">'
        f'<node id="n0" name="thing">{node_extra}'
        f'<instance_geometry url="#{gid}">{bind}</instance_geometry>'
        "</node></visual_scene></library_visual_scenes>"
        '<scene><instance_visual_scene url="#Scene"/></scene>'
    )


def _dae(body: str, *, version: str = "1.4.1", asset: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<COLLADA xmlns="{_NS}" version="{version}">'
        f"<asset>{asset}</asset>{body}</COLLADA>\n"
    )


def _tri_dae(**kw) -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    return _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))


def _write(tmp_path: Path, text: str, name: str = "m.dae") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _surface() -> PolyData:
    return make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]]),
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_read_triangles(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _tri_dae()))
    assert len(scene.meshes) == 1
    mesh = scene.meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)
    np.testing.assert_array_equal(mesh.connectivity, [0, 1, 2])
    assert mesh.element_types.tolist() == [ELEMENT_TYPES["triangle"]]
    assert scene.nodes[0].mesh == 0
    assert scene.nodes[0].name == "thing"
    assert scene.scenes == ((0,),)
    assert scene.active_scene == 0
    assert "material" not in mesh.element_attrs


def test_read_polylist_quads_and_polygons(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
    prim = (
        '<polylist count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<vcount>4 5</vcount><p>0 1 2 3 1 4 5 2 3</p></polylist>"
    )
    geo = _geometry("g0", verts, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["quad"],
        ELEMENT_TYPES["polygon"],
    ]
    assert mesh.offsets.tolist() == [0, 4, 9]
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 1, 4, 5, 2, 3]


def test_read_polylist_triangles_are_triangles(tmp_path: Path) -> None:
    prim = (
        '<polylist count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<vcount>3</vcount><p>0 1 2</p></polylist>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [ELEMENT_TYPES["triangle"]]


def test_read_polygons_element_with_holes_warns(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]])
    prim = (
        '<polygons count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2 3</p><ph><p>0 1 4</p><h>1 2 4</h></ph></polygons>"
    )
    geo = _geometry("g0", verts, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="hole"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["quad"],
        ELEMENT_TYPES["triangle"],
    ]
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 0, 1, 4]


def test_read_lines_and_linestrips(tmp_path: Path) -> None:
    prim = (
        '<lines count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 1 2</p></lines>"
        '<linestrips count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2</p></linestrips>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["poly_line"],
    ]
    assert mesh.connectivity.tolist() == [0, 1, 1, 2, 0, 1, 2]


def test_read_tristrips_and_trifans(tmp_path: Path) -> None:
    prim = (
        '<tristrips count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2 3</p></tristrips>"
        '<trifans count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1 2 3</p></trifans>"
    )
    geo = _geometry("g0", _QUAD, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.element_types.tolist() == [
        ELEMENT_TYPES["triangle_strip"],
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["triangle"],
    ]
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 2, 3]


def test_read_empty_trifans(tmp_path: Path) -> None:
    prim = '<trifans count="0"><input semantic="VERTEX" source="#g0-vtx" offset="0"/></trifans>'
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.offsets) == 1 and len(mesh.vertices) == 3


def test_read_multi_offset_inputs_split_corners(tmp_path: Path) -> None:
    normals = np.array([[0.0, 0, 1], [0, 0, -1]])
    uv = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    faces = np.array(
        [[0, 0, 0, 1, 0, 1, 2, 0, 2], [0, 1, 3, 1, 0, 1, 2, 0, 2]], dtype=np.int64
    )
    inputs = (
        '<input semantic="NORMAL" source="#g0-nrm" offset="1"/>'
        '<input semantic="TEXCOORD" source="#g0-uv" offset="2" set="0"/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", faces, extra_inputs=inputs),
        sources=_source("g0-nrm", normals, "X Y Z") + _source("g0-uv", uv, "S T"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # Corner (0, 0, 0) and (0, 1, 3) differ, so vertex 0 is split; the other
    # two corners agree across faces and stay one vertex each.
    assert len(mesh.vertices) == 4
    np.testing.assert_array_equal(mesh.vertices, np.vstack([_TRI, _TRI[:1]]))
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 1, 2]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"], normals[[0, 0, 0, 1]])
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords"], uv[[0, 1, 2, 3]])


@pytest.mark.parametrize("faces", [[[0, 0, 1, 0, 2, 0]], []])
def test_read_split_corners_keep_unreferenced_vertices(
    tmp_path: Path, faces: list
) -> None:
    verts = np.vstack([_TRI, [[5.0, 5, 5]]])
    geo = _geometry(
        "g0",
        verts,
        _triangles(
            "g0",
            np.asarray(faces, dtype=np.int64).reshape(-1, 6),
            extra_inputs='<input semantic="NORMAL" source="#g0-nrm" offset="1"/>',
        ),
        sources=_source("g0-nrm", np.array([[0.0, 0, 1]]), "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.vertices) == 4
    np.testing.assert_array_equal(
        np.sort(mesh.vertices, axis=0), np.sort(verts, axis=0)
    )
    assert mesh.vertex_attrs["normals"].shape == (4, 3)


def test_read_shared_offset_keeps_unreferenced_vertices(tmp_path: Path) -> None:
    verts = np.vstack([_TRI, [[5.0, 5, 5]]])
    normals = np.tile([[0.0, 0, 1]], (4, 1))
    inputs = '<input semantic="NORMAL" source="#g0-nrm" offset="0"/>'
    geo = _geometry(
        "g0",
        verts,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-nrm", normals, "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.vertices) == 4
    assert mesh.vertex_attrs["normals"].shape == (4, 3)


def test_read_inputs_inside_vertices(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (3, 1))
    geo = (
        '<geometry id="g0"><mesh>'
        + _source("g0-pos", _TRI, "X Y Z")
        + _source("g0-nrm", normals, "X Y Z")
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        '<input semantic="NORMAL" source="#g0-nrm"/></vertices>'
        + _triangles("g0", [[0, 1, 2]])
        + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"], normals)


def test_read_texcoord_set_with_spaces_is_a_number(tmp_path: Path) -> None:
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    inputs = (
        '<input semantic="TEXCOORD" source="#g0-uv" offset="0" set="0"/>'
        '<input semantic="TEXCOORD" source="#g0-uv" offset="0" set=" 1 "/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-uv", uv, "S T"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert sorted(mesh.vertex_attrs) == ["texcoords", "texcoords_1"]


@pytest.mark.parametrize("offset", [0, 1])
def test_read_int_array_colors_are_float64_on_every_path(
    tmp_path: Path, offset: int
) -> None:
    inputs = f'<input semantic="COLOR" source="#g0-col" offset="{offset}"/>'
    prims = _triangles("g0", [[0, 1, 2]], extra_inputs=inputs)
    if offset:
        prims = prims.replace("<p>0 1 2</p>", "<p>0 0 1 1 2 2</p>")
    geo = _geometry(
        "g0",
        _TRI,
        prims,
        sources=(
            '<source id="g0-col"><int_array id="g0-col-array" count="9">'
            "1 0 0 0 1 0 0 0 1</int_array><technique_common>"
            '<accessor source="#g0-col-array" count="3" stride="3">'
            '<param name="R" type="int"/><param name="G" type="int"/>'
            '<param name="B" type="int"/></accessor></technique_common></source>'
        ),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    colors = read_scene(_write(tmp_path, text)).meshes[0].vertex_attrs["colors"]
    assert colors.dtype == np.float64
    np.testing.assert_array_equal(colors, np.eye(3))


def test_read_colors_and_texcoord_sets(tmp_path: Path) -> None:
    colors = np.array([[1.0, 0, 0, 1], [0, 1, 0, 0.5], [0, 0, 1, 1]])
    uv1 = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    inputs = (
        '<input semantic="COLOR" source="#g0-col" offset="0" set="0"/>'
        '<input semantic="TEXCOORD" source="#g0-uv1" offset="0" set="1"/>'
        '<input semantic="TEXTANGENT" source="#g0-tan" offset="0"/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-col", colors, "R G B A")
        + _source("g0-uv1", uv1, "S T")
        + _source("g0-tan", np.tile([[1.0, 0, 0]], (3, 1)), "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["colors"], colors)
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords"], uv1)
    assert mesh.vertex_attrs["textangent"].shape == (3, 3)


@pytest.mark.parametrize(
    ("sets", "keys"),
    [
        ("1", ["texcoords"]),
        ("1 2", ["texcoords", "texcoords_1"]),
        ("0 2", ["texcoords", "texcoords_2"]),
    ],
)
def test_read_texcoord_sets_renumbered_from_the_lowest_without_set_0(
    tmp_path: Path, sets: str, keys: list[str]
) -> None:
    uv = np.array([[0.0, 0], [1, 0], [0, 1]])
    inputs = "".join(
        f'<input semantic="TEXCOORD" source="#g0-uv" offset="0" set="{k}"/>'
        for k in sets.split()
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-uv", uv, "S T"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert sorted(mesh.vertex_attrs) == keys


def test_read_uvw_texcoords_with_empty_p_are_uv(tmp_path: Path) -> None:
    uvw = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    uvw_3d = uvw + [0, 0, 0.5]
    inputs = (
        '<input semantic="TEXCOORD" source="#g0-uv" offset="0" set="0"/>'
        '<input semantic="TEXCOORD" source="#g0-uvw" offset="0" set="1"/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-uv", uvw, "S T P") + _source("g0-uvw", uvw_3d, "S T P"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords"], uvw[:, :2])
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords_1"], uvw_3d)


def test_read_accessor_offset_and_unnamed_params(tmp_path: Path) -> None:
    src = (
        '<source id="g0-pos"><float_array id="g0-pos-array" count="13">'
        "9 0 0 0 7 1 0 0 7 0 1 0 7</float_array><technique_common>"
        '<accessor source="#g0-pos-array" count="3" stride="4" offset="1">'
        '<param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/><param type="float"/></accessor>'
        "</technique_common></source>"
    )
    geo = (
        '<geometry id="g0"><mesh>'
        + src
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        "</vertices>" + _triangles("g0", [[0, 1, 2]]) + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)


def test_read_interleaved_accessors_over_one_array(tmp_path: Path) -> None:
    normals = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]])
    rows = np.hstack([_TRI, normals])
    src = (
        f'<float_array id="pn" count="18">{_nums(rows)}</float_array>'
        '<source id="g0-pos"><technique_common>'
        '<accessor source="#pn" count="3" stride="6" offset="0">'
        '<param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/></accessor></technique_common></source>'
        '<source id="g0-nrm"><technique_common>'
        '<accessor source="#pn" count="3" stride="6" offset="3">'
        '<param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/></accessor></technique_common></source>'
    )
    geo = (
        '<geometry id="g0"><mesh>'
        + src
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        '<input semantic="NORMAL" source="#g0-nrm"/></vertices>'
        + _triangles("g0", [[0, 1, 2]])
        + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"], normals)
    short = text.replace('count="18"', 'count="17"').replace(
        " 0.0</float_array>", "</float_array>"
    )
    with pytest.raises(CodecError, match="reads 18 values from an array of 17"):
        read_scene(_write(tmp_path, short))


def test_read_geometry_never_instanced_still_a_mesh(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]])) + _geometry(
        "g1", _QUAD, _triangles("g1", [[0, 1, 2]])
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g1"))
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 2
    assert scene.nodes[0].mesh == 1


def test_read_geometry_without_mesh_warns(tmp_path: Path) -> None:
    geo = '<geometry id="g0"><spline/></geometry>'
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="spline"):
        scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes[0].vertices) == 0


def test_read_vertices_input_of_wrong_length_refused(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (2, 1))
    geo = (
        '<geometry id="g0"><mesh>'
        + _source("g0-pos", _TRI, "X Y Z")
        + _source("g0-nrm", normals, "X Y Z")
        + '<vertices id="g0-vtx"><input semantic="NORMAL" source="#g0-nrm"/>'
        '<input semantic="POSITION" source="#g0-pos"/></vertices>'
        + _triangles("g0", [[0, 1, 2]])
        + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="normals inside <vertices> with 2 rows"):
        read_scene(_write(tmp_path, text))


def test_read_shared_offset_short_source_refused_past_its_end(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (2, 1))
    inputs = '<input semantic="NORMAL" source="#g0-nrm" offset="0"/>'
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-nrm", normals, "X Y Z"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="entry 2 of a source holding 2"):
        read_scene(_write(tmp_path, text))


def test_read_accessor_stride_past_its_named_params(tmp_path: Path) -> None:
    src = (
        '<source id="g0-pos"><float_array id="g0-pos-array" count="12">'
        "0 0 0 7 1 0 0 7 0 1 0 7</float_array><technique_common>"
        '<accessor source="#g0-pos-array" count="3" stride="4">'
        '<param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/></accessor></technique_common></source>'
    )
    geo = (
        '<geometry id="g0"><mesh>'
        + src
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        "</vertices>" + _triangles("g0", [[0, 1, 2]]) + "</mesh></geometry>"
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)


def test_read_block_without_an_input_gets_zeros(tmp_path: Path) -> None:
    normals = np.array([[0.0, 0, 1], [0, 0, -1], [0, 1, 0]])
    inputs = '<input semantic="NORMAL" source="#g0-nrm" offset="1"/>'
    prims = _triangles("g0", [[0, 0, 1, 1, 2, 2]], extra_inputs=inputs) + (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<p>0 1</p></lines>"
    )
    geo = _geometry("g0", _TRI, prims, sources=_source("g0-nrm", normals, "X Y Z"))
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="normals.*some primitive blocks"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.vertices) == 5
    n = mesh.vertex_attrs["normals"]
    assert np.isfinite(n).all()
    np.testing.assert_array_equal(n[:3], normals)
    np.testing.assert_array_equal(n[3:], 0.0)


def test_read_attr_of_two_widths_warns_once(tmp_path: Path) -> None:
    n3 = np.array([[0.0, 0, 1], [0, 0, -1], [0, 1, 0]])
    n2 = np.array([[1.0, 0], [0, 1]])
    tri = _triangles(
        "g0",
        [[0, 0, 1, 1, 2, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-n3" offset="1"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-n2" offset="1"/><p>0 0 1 1</p></lines>'
    )
    sources = _source("g0-n3", n3, "X Y Z") + _source("g0-n2", n2, "X Y")
    geo = _geometry("g0", _TRI, tri + lines, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    messages = [str(w.message) for w in record]
    assert len(messages) == 1 and "two widths" in messages[0]
    assert mesh.vertex_attrs["normals"].shape == (5, 3)


def test_read_attr_of_two_widths_says_what_the_corners_hold(tmp_path: Path) -> None:
    n3 = np.array([[0.0, 0, 1], [0, 0, -1], [0, 1, 0]])
    n2 = np.array([[1.0, 0], [0, 1]])
    tri = _triangles(
        "g0",
        [[0, 0, 1, 1, 2, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-n3" offset="1"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-n2" offset="1"/><p>0 0 1 1</p></lines>'
    )
    sources = _source("g0-n3", n3, "X Y Z") + _source("g0-n2", n2, "X Y")
    geo = _geometry("g0", _TRI, tri + lines, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="keeps the earlier source's value, or zero"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    line_corners = mesh.connectivity[mesh.offsets[1] : mesh.offsets[2]]
    assert not mesh.vertex_attrs["normals"][line_corners].any()


def test_read_symbols_resolved_per_block(tmp_path: Path) -> None:
    prims = (
        _triangles("g0", [[0, 1, 2]] * 3, material="sym")
        + _triangles("g0", [[0, 1, 2]] * 2, material="ghost")
        + _triangles("g0", [[0, 1, 2]] * 2)
    )
    text = _dae(
        f"<library_geometries>{_geometry('g0', _TRI, prims)}</library_geometries>"
        '<library_materials><material id="sym"/></library_materials>'
        + _scene_with("g0")
    )
    path = _write(tmp_path, text)
    with pytest.warns(UserWarning, match=r"\['ghost'\]"):
        mesh = read_scene(path).meshes[0]
    assert mesh.element_attrs["material"].tolist() == [0, 0, 0, -1, -1, -1, -1]
    root = ET.fromstring(path.read_bytes())
    doc = _collada._Doc(
        root=root, name="m.dae", size=len(text), ids=_collada._ids(root)
    )
    geo = root.find(f"{{{_NS}}}library_geometries/{{{_NS}}}geometry")
    assert _collada._read_geometry(doc, geo, 0).symbols == [
        ("sym", 3),
        ("ghost", 2),
        (None, 2),
    ]


def test_read_per_position_source_reindexed_by_a_later_block(tmp_path: Path) -> None:
    normals = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    tri = _triangles(
        "g0",
        [[0, 1, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-nrm" offset="0"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-nrm" offset="1"/><p>0 2 1 2</p></lines>'
    )
    geo = _geometry(
        "g0", _TRI, tri + lines, sources=_source("g0-nrm", normals, "X Y Z")
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # The line corners name normal 2 for positions 0 and 1, so they split.
    assert len(mesh.vertices) == 5
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][:3], normals)
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][3:], normals[[2, 2]])


def test_read_second_per_position_source_of_one_attr_splits_its_corners(
    tmp_path: Path,
) -> None:
    n1 = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    n2 = -n1
    tri = _triangles(
        "g0",
        [[0, 1, 2]],
        extra_inputs='<input semantic="NORMAL" source="#g0-n1" offset="0"/>',
    )
    lines = (
        '<lines count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        '<input semantic="NORMAL" source="#g0-n2" offset="0"/><p>0 1</p></lines>'
    )
    sources = _source("g0-n1", n1, "X Y Z") + _source("g0-n2", n2, "X Y Z")
    geo = _geometry("g0", _TRI, tri + lines, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    # The line names the second source for positions 0 and 1, so those two
    # corners are new vertices carrying its values, not the first source's.
    assert len(mesh.vertices) == 5
    assert mesh.connectivity.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][:3], n1)
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][3:], n2[:2])


def test_read_geometries_sharing_a_source_own_their_memory(tmp_path: Path) -> None:
    normals = np.tile([[0.0, 0, 1]], (3, 1))
    shared = _source("shared-pos", _TRI, "X Y Z") + _source(
        "shared-n", normals, "X Y Z"
    )
    geos = "".join(
        f'<geometry id="{gid}"><mesh><vertices id="{gid}-vtx">'
        '<input semantic="POSITION" source="#shared-pos"/>'
        '<input semantic="NORMAL" source="#shared-n"/></vertices>'
        + _triangles(gid, [[0, 1, 2]])
        + "</mesh></geometry>"
        for gid in ("g0", "g1")
    )
    text = _dae(
        f"<library_geometries>{shared}{geos}</library_geometries>" + _scene_with("g0")
    )
    a, b = read_scene(_write(tmp_path, text)).meshes
    assert not np.shares_memory(a.vertices, b.vertices)
    assert not np.shares_memory(a.vertex_attrs["normals"], b.vertex_attrs["normals"])
    a.vertices[0] = 9.0
    np.testing.assert_array_equal(b.vertices, _TRI)


def test_read_two_sources_over_one_array_own_their_memory(tmp_path: Path) -> None:
    array = f'<float_array id="shared" count="9">{_nums(_TRI)}</float_array>'
    geos = "".join(
        f'<geometry id="{gid}"><mesh><source id="{gid}-pos"><technique_common>'
        '<accessor source="#shared" count="3" stride="3"><param name="X" type="float"/>'
        '<param name="Y" type="float"/><param name="Z" type="float"/></accessor>'
        f'</technique_common></source><vertices id="{gid}-vtx">'
        f'<input semantic="POSITION" source="#{gid}-pos"/></vertices>'
        + _triangles(gid, [[0, 1, 2]])
        + "</mesh></geometry>"
        for gid in ("g0", "g1")
    )
    text = _dae(
        f"<library_geometries>{array}{geos}</library_geometries>" + _scene_with("g0")
    )
    a, b = read_scene(_write(tmp_path, text)).meshes
    assert not np.shares_memory(a.vertices, b.vertices)
    a.vertices[0] = 9.0
    np.testing.assert_array_equal(b.vertices, _TRI)


# ---------------------------------------------------------------------------
# materials, effects, images
# ---------------------------------------------------------------------------


def _effect_dae(technique: str, *, newparams: str = "", images: str = "") -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    bind = (
        "<bind_material><technique_common>"
        '<instance_material symbol="sym" target="#mat0"/>'
        "</technique_common></bind_material>"
    )
    return _dae(
        f"<library_images>{images}</library_images>"
        f'<library_effects><effect id="fx0"><profile_COMMON>{newparams}'
        f'<technique sid="common">{technique}</technique></profile_COMMON>'
        "</effect></library_effects>"
        '<library_materials><material id="mat0" name="Red">'
        '<instance_effect url="#fx0"/></material></library_materials>'
        f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0", bind=bind)
    )


def test_read_material_symbol_bound_twice_keeps_the_first(tmp_path: Path) -> None:
    text = (
        _effect_dae("<phong><diffuse><color>1 0 0 1</color></diffuse></phong>")
        .replace(
            '<instance_material symbol="sym" target="#mat0"/>',
            '<instance_material symbol="sym" target="#mat0"/>'
            '<instance_material symbol="sym" target="#mat1"/>',
        )
        .replace(
            "</library_materials>",
            '<material id="mat1" name="Blue"><instance_effect url="#fx0"/>'
            "</material></library_materials>",
        )
    )
    with pytest.warns(UserWarning, match="'sym' is bound twice"):
        scene = read_scene(_write(tmp_path, text))
    (material,) = scene.meshes[0].element_attrs["material"].tolist()
    assert scene.materials[material].name == "Red"


def test_read_phong_material(tmp_path: Path) -> None:
    technique = (
        "<phong><emission><color>0.1 0.2 0.3 1</color></emission>"
        "<ambient><color>0 0 0 1</color></ambient>"
        "<diffuse><color>1 0 0 1</color></diffuse>"
        "<specular><color>0.5 0.5 0.5 1</color></specular>"
        "<shininess><float>50</float></shininess>"
        "<transparency><float>1</float></transparency></phong>"
    )
    scene = read_scene(_write(tmp_path, _effect_dae(technique)))
    assert len(scene.materials) == 1
    mat = scene.materials[0]
    assert mat.name == "Red"
    assert mat.base_color == (1.0, 0.0, 0.0, 1.0)
    assert mat.emissive == pytest.approx((0.1, 0.2, 0.3))
    assert mat.alpha_mode == "OPAQUE"
    assert mat.metallic == 0.0
    assert mat.extras["shading"] == "phong"
    assert mat.extras["specular"] == (0.5, 0.5, 0.5, 1.0)
    assert mat.extras["shininess"] == 50.0
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]


def test_read_transparency_a_one_becomes_blend(tmp_path: Path) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>0 0 0 1</color></transparent>'
        "<transparency><float>0.25</float></transparency></lambert>"
    )
    mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color == (0.0, 1.0, 0.0, 0.25)
    assert mat.alpha_mode == "BLEND"
    assert mat.extras["shading"] == "lambert"


@pytest.mark.parametrize("amount", ["nan", "inf"])
def test_read_transparency_not_finite_is_opaque(tmp_path: Path, amount: str) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>0 0 0 1</color></transparent>'
        f"<transparency><float>{amount}</float></transparency></lambert>"
    )
    with pytest.warns(UserWarning, match="not finite"):
        mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color == (0.0, 1.0, 0.0, 1.0)
    assert mat.alpha_mode == "OPAQUE"


def test_read_transparency_rgb_zero(tmp_path: Path) -> None:
    technique = (
        "<blinn><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="RGB_ZERO"><color>1 1 1 1</color></transparent>'
        "<transparency><float>1</float></transparency></blinn>"
    )
    mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color[3] == pytest.approx(0.0)
    assert mat.alpha_mode == "BLEND"


def test_read_unknown_opaque_mode_is_a_one_with_a_warning(tmp_path: Path) -> None:
    technique = (
        "<phong><diffuse><color>0 1 0 0.5</color></diffuse>"
        '<transparent opaque="a_one"><color>1 1 1 1</color></transparent>'
        "<transparency><float>1</float></transparency></phong>"
    )
    with pytest.warns(UserWarning, match="opaque='a_one'"):
        mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    # The diffuse alpha is applied once, not squared into 0.25.
    assert mat.base_color[3] == pytest.approx(0.5)
    assert mat.extras["transparent_mode"] == "A_ONE"


def test_read_texture_chain_and_images(tmp_path: Path) -> None:
    newparams = (
        '<newparam sid="surf"><surface type="2D"><init_from>img0</init_from>'
        "</surface></newparam>"
        '<newparam sid="samp"><sampler2D><source>surf</source>'
        "<wrap_s>CLAMP</wrap_s><wrap_t>MIRROR</wrap_t>"
        "<minfilter>LINEAR_MIPMAP_LINEAR</minfilter><magfilter>NEAREST</magfilter>"
        "</sampler2D></newparam>"
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse>'
        '<extra><technique profile="FCOLLADA"><bump>'
        '<texture texture="img1" texcoord="UVMap"/></bump></technique></extra>'
        "</phong>"
    )
    images = (
        '<image id="img0" name="skin"><init_from>textures/skin.png</init_from></image>'
        '<image id="img1"><init_from>bump.jpg</init_from></image>'
    )
    scene = read_scene(
        _write(tmp_path, _effect_dae(technique, newparams=newparams, images=images))
    )
    assert scene.images == (
        SceneImage(uri="textures/skin.png", media_type="image/png", name="skin"),
        SceneImage(uri="bump.jpg", media_type="image/jpeg"),
    )
    mat = scene.materials[0]
    assert mat.base_color_texture == 0
    assert mat.normal_texture == 1
    tex = scene.textures[0]
    assert tex.image == 0
    assert (tex.wrap_s, tex.wrap_t) == (33071, 33648)
    assert (tex.min_filter, tex.mag_filter) == (9987, 9728)
    assert scene.textures[1] == SceneTexture(image=1)


def test_read_image_held_by_the_effect(tmp_path: Path) -> None:
    newparams = (
        '<image id="local"><init_from>wood.png</init_from></image>'
        '<newparam sid="surf"><surface type="2D"><init_from>local</init_from>'
        "</surface></newparam>"
        '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>'
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse></phong>'
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(
            _write(tmp_path, _effect_dae(technique, newparams=newparams))
        )
    assert scene.images == (SceneImage(uri="wood.png", media_type="image/png"),)
    assert scene.materials[0].base_color_texture == 0
    assert scene.textures[0].image == 0


def test_warnings_point_at_the_caller(tmp_path: Path) -> None:
    technique = '<phong><diffuse><texture texture="nowhere" texcoord="UVMap"/></diffuse></phong>'
    with pytest.warns(UserWarning, match="reaches no image") as record:
        read_scene(_write(tmp_path, _effect_dae(technique)))
    assert record[0].filename == __file__


def test_read_flatten_warning_points_at_the_caller(tmp_path: Path) -> None:
    path = _write(tmp_path, _tri_dae())
    for reader in (read, polyxios.read):
        with pytest.warns(UserWarning, match="scene format") as record:
            reader(path)
        assert record[0].filename == __file__


def test_read_opaque_convention_warns_once_per_document(tmp_path: Path) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>'
        "<transparency><float>0</float></transparency></lambert>"
    )
    text = (
        _effect_dae(technique)
        .replace(
            "</library_effects>",
            f'<effect id="fx1"><profile_COMMON><technique sid="c">{technique}'
            "</technique></profile_COMMON></effect></library_effects>",
        )
        .replace(
            "</library_materials>",
            '<material id="mat1"><instance_effect url="#fx1"/></material>'
            "</library_materials>",
        )
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        scene = read_scene(_write(tmp_path, text))
    assert [str(w.message).count("convention for opaque") for w in record] == [1]
    assert [m.alpha_mode for m in scene.materials] == ["OPAQUE", "OPAQUE"]


def test_read_image_forms_1_5(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    images = (
        '<image id="img0"><init_from><ref>a.png</ref></init_from></image>'
        f'<image id="img1"><init_from><hex format="png">{payload.hex()}</hex>'
        "</init_from></image>"
        '<image id="img2"><init_from>data:image/jpeg;base64,'
        f"{base64.b64encode(payload).decode()}</init_from></image>"
    )
    text = _effect_dae(
        "<phong><diffuse><color>1 1 1 1</color></diffuse></phong>", images=images
    ).replace('version="1.4.1"', 'version="1.5.0"')
    scene = read_scene(_write(tmp_path, text))
    assert scene.images[0] == SceneImage(uri="a.png", media_type="image/png")
    assert scene.images[1] == SceneImage(data=payload, media_type="image/png")
    assert scene.images[2] == SceneImage(data=payload, media_type="image/jpeg")


def test_read_data_uri_wrapped_and_with_parameters(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    encoded = base64.b64encode(payload).decode()
    wrapped = "\n      ".join(encoded[i : i + 4] for i in range(0, len(encoded), 4))
    images = (
        f'<image id="img0"><init_from>data:image/png;base64,{wrapped}</init_from></image>'
        f'<image id="img1"><init_from>data:image/png;charset=x;base64,{encoded}'
        "</init_from></image>"
        f'<image id="img2"><init_from>data:;base64,{encoded}</init_from></image>'
    )
    text = _effect_dae(
        "<phong><diffuse><color>1 1 1 1</color></diffuse></phong>", images=images
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.images[0] == SceneImage(data=payload, media_type="image/png")
    assert scene.images[1] == SceneImage(data=payload, media_type="image/png")
    assert scene.images[2] == SceneImage(data=payload, media_type=None)


def test_read_data_uri_base64_marker_any_case_and_percent_encoded(
    tmp_path: Path,
) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    encoded = base64.b64encode(payload).decode()
    images = (
        f'<image id="img0"><init_from>data:image/png;BASE64,{encoded}</init_from>'
        "</image>"
        '<image id="img1"><init_from>DATA:text/plain,a%20b</init_from></image>'
    )
    text = _effect_dae(
        "<phong><diffuse><color>1 1 1 1</color></diffuse></phong>", images=images
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.images[0] == SceneImage(data=payload, media_type="image/png")
    assert scene.images[1] == SceneImage(data=b"a b", media_type="text/plain")


def test_read_image_inline_data_1_4(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfake"
    images = f'<image id="img0"><data>{payload.hex().upper()}</data></image>'
    text = _effect_dae(
        "<phong><diffuse><color>1 1 1 1</color></diffuse></phong>", images=images
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.images[0] == SceneImage(data=payload, media_type="image/png")


def test_read_image_created_by_the_reader_warns(tmp_path: Path) -> None:
    images = '<image id="img0"><create_2d><size_exact width="4" height="4"/></create_2d></image>'
    text = _effect_dae(
        "<phong><diffuse><color>1 1 1 1</color></diffuse></phong>", images=images
    ).replace('version="1.4.1"', 'version="1.5.0"')
    with pytest.warns(UserWarning, match="create_2d"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.images[0] == SceneImage()


def test_read_unbound_symbol_falls_back_to_material_id(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="mat0"))
    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<constant/></technique></profile_COMMON></effect></library_effects>"
        '<library_materials><material id="mat0"><instance_effect url="#fx0"/>'
        "</material></library_materials>"
        f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0")
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]
    assert scene.materials[0].extras["shading"] == "constant"


def test_read_unknown_symbol_is_minus_one(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="nope"))
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="nope"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [-1]


def test_read_second_binding_of_a_shared_mesh_gets_its_own_mesh(
    tmp_path: Path,
) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    mats = "".join(
        f'<material id="mat{i}"><instance_effect url="#fx0"/></material>'
        for i in range(2)
    )

    def bind(i):
        return (
            "<bind_material><technique_common>"
            f'<instance_material symbol="sym" target="#mat{i}"/>'
            "</technique_common></bind_material>"
        )

    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<phong/></technique></profile_COMMON></effect></library_effects>"
        f"<library_materials>{mats}</library_materials>"
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        f'<node id="a"><instance_geometry url="#g0">{bind(0)}</instance_geometry></node>'
        f'<node id="b"><instance_geometry url="#g0">{bind(1)}</instance_geometry></node>'
        "</visual_scene></library_visual_scenes>"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 2
    assert [n.mesh for n in scene.nodes] == [0, 1]
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]
    assert scene.meshes[1].element_attrs["material"].tolist() == [1]
    np.testing.assert_array_equal(scene.meshes[0].vertices, scene.meshes[1].vertices)
    assert scene.meshes[1].vertex_attrs is not scene.meshes[0].vertex_attrs


def _two_instances_dae(material: str, first: str, second: str) -> str:
    """Return one geometry of symbol ``material`` instanced under two bindings."""
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material=material))
    mats = "".join(
        f'<material id="mat{i}"><instance_effect url="#fx0"/></material>'
        for i in range(2)
    )

    def bind(pairs: str) -> str:
        items = "".join(
            f'<instance_material symbol="{sym}" target="#{mat}"/>'
            for sym, mat in (pair.split("=") for pair in pairs.split())
        )
        return f"<bind_material><technique_common>{items}</technique_common></bind_material>"

    return _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<phong/></technique></profile_COMMON></effect></library_effects>"
        f"<library_materials>{mats}</library_materials>"
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        f'<node id="a"><instance_geometry url="#g0">{bind(first)}</instance_geometry></node>'
        f'<node id="b"><instance_geometry url="#g0">{bind(second)}</instance_geometry></node>'
        "</visual_scene></library_visual_scenes>"
    )


def test_read_second_binding_adding_other_symbols_is_silent(tmp_path: Path) -> None:
    text = _two_instances_dae("sym", "sym=mat0", "sym=mat0 other=mat1")
    scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [0]


def test_read_binding_omitting_a_symbol_falls_back_to_its_id_with_warning(
    tmp_path: Path,
) -> None:
    text = _two_instances_dae("mat1", "sym=mat0", "sym=mat0")
    with pytest.warns(UserWarning, match="bound without material symbol"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].element_attrs["material"].tolist() == [1]


def test_read_a_one_white_zero_is_the_opaque_convention(tmp_path: Path) -> None:
    technique = (
        "<lambert><diffuse><color>0 1 0 1</color></diffuse>"
        '<transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>'
        "<transparency><float>0</float></transparency></lambert>"
    )
    with pytest.warns(UserWarning, match="convention for opaque"):
        mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color == (0.0, 1.0, 0.0, 1.0)
    assert mat.alpha_mode == "OPAQUE"


def test_read_1_5_sampler_instance_image(tmp_path: Path) -> None:
    newparams = (
        '<newparam sid="samp"><sampler2D><instance_image url="#img0"/>'
        "<wrap_s>CLAMP</wrap_s></sampler2D></newparam>"
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse></phong>'
    )
    images = '<image id="img0"><init_from><ref>a.png</ref></init_from></image>'
    text = _effect_dae(technique, newparams=newparams, images=images).replace(
        'version="1.4.1"', 'version="1.5.0"'
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, text))
    assert scene.materials[0].base_color_texture == 0
    assert scene.textures[0] == SceneTexture(image=0, wrap_s=33071)


def test_read_empty_surface_does_not_match_an_image_without_id(tmp_path: Path) -> None:
    newparams = (
        '<newparam sid="surf"><surface type="2D"><init_from></init_from>'
        "</surface></newparam>"
        '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>'
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UVMap"/></diffuse></phong>'
    )
    images = "<image><init_from>a.png</init_from></image>"
    with pytest.warns(UserWarning, match="reaches no image"):
        scene = read_scene(
            _write(tmp_path, _effect_dae(technique, newparams=newparams, images=images))
        )
    assert scene.materials[0].base_color_texture is None


def test_read_unbound_and_bound_instances_get_their_own_meshes(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<phong/></technique></profile_COMMON></effect></library_effects>"
        '<library_materials><material id="mat0"><instance_effect url="#fx0"/>'
        "</material></library_materials>"
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_geometry url="#g0"/></node>'
        '<node id="b"><instance_geometry url="#g0"><bind_material>'
        '<technique_common><instance_material symbol="sym" target="#mat0"/>'
        "</technique_common></bind_material></instance_geometry></node>"
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.warns(UserWarning, match="no binding or material id resolves"):
        scene = read_scene(_write(tmp_path, text))
    assert [n.mesh for n in scene.nodes] == [0, 1]
    assert scene.meshes[0].element_attrs["material"].tolist() == [-1]
    assert scene.meshes[1].element_attrs["material"].tolist() == [0]


# ---------------------------------------------------------------------------
# nodes, transforms, scenes
# ---------------------------------------------------------------------------


def test_read_node_transform_composition(tmp_path: Path) -> None:
    node_extra = (
        '<translate sid="location">1 2 3</translate>'
        '<rotate sid="rotationZ">0 0 1 90</rotate>'
        '<scale sid="scale">2 2 2</scale>'
    )
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=node_extra)
    )
    scene = read_scene(_write(tmp_path, text))
    node = scene.nodes[0]
    expected = np.array(
        [[0.0, -2, 0, 1], [2, 0, 0, 2], [0, 0, 2, 3], [0, 0, 0, 1]], dtype=np.float64
    )
    np.testing.assert_allclose(node.matrix, expected, atol=1e-12)
    assert [t["kind"] for t in node.extras["transforms"]] == [
        "translate",
        "rotate",
        "scale",
    ]
    assert node.extras["transforms"][1]["sid"] == "rotationZ"
    assert node.extras["id"] == "n0"
    flat = scene.to_polydata()
    np.testing.assert_allclose(flat.vertices[1], [1, 4, 3], atol=1e-12)


def test_read_matrix_is_row_major(tmp_path: Path) -> None:
    m = np.arange(16, dtype=np.float64).reshape(4, 4)
    m[3] = [0, 0, 0, 1]
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=f'<matrix sid="transform">{_nums(m)}</matrix>')
    )
    node = read_scene(_write(tmp_path, text)).nodes[0]
    np.testing.assert_array_equal(node.matrix, m)


def test_read_lookat_and_skew(tmp_path: Path) -> None:
    node_extra = "<lookat>0 0 5 0 0 0 0 1 0</lookat><skew>10 1 0 0 0 1 0</skew>"
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=node_extra)
    )
    with pytest.warns(UserWarning, match="skew"):
        node = read_scene(_write(tmp_path, text)).nodes[0]
    np.testing.assert_allclose(node.matrix[:3, 3], [0, 0, 5])
    np.testing.assert_allclose(node.matrix[:3, :3], np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "lookat",
    [
        "5 0 0 0 0 0 1 0 0",
        "0 0 5 0 0 0 0 0 1",
        "0 5 0 0 0 0 0 1 0",
        "1 2 3 0 0 0 2 4 6",
    ],
)
def test_read_lookat_with_up_along_the_view_stays_a_rotation(
    tmp_path: Path, lookat: str
) -> None:
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra=f"<lookat>{lookat}</lookat>")
    )
    rot = read_scene(_write(tmp_path, text)).nodes[0].matrix[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0)
    eye = np.array(lookat.split(), dtype=float)[:3]
    np.testing.assert_allclose(rot[:, 2], eye / np.linalg.norm(eye))


def test_read_hierarchy_instance_node_and_multiple_geometries(tmp_path: Path) -> None:
    geos = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]), name="A") + _geometry(
        "g1", _QUAD, _triangles("g1", [[0, 1, 2]]), name="B"
    )
    text = _dae(
        f"<library_geometries>{geos}</library_geometries>"
        '<library_nodes><node id="lib" name="Lib"><translate>0 0 1</translate>'
        '<instance_geometry url="#g1"/></node></library_nodes>'
        '<library_visual_scenes><visual_scene id="S" name="Main">'
        '<node id="root" name="Root" sid="r" type="JOINT">'
        '<node id="kid" name="Kid"><instance_geometry url="#g0"/>'
        '<instance_geometry url="#g1"/></node>'
        '<instance_node url="#lib"/><instance_camera url="#cam"/>'
        "</node></visual_scene>"
        '<visual_scene id="S2" name="Second"><node id="other"/></visual_scene>'
        "</library_visual_scenes>"
        '<scene><instance_visual_scene url="#S2"/></scene>'
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.scenes == ((0,), (4,))
    assert scene.active_scene == 1
    assert scene.name == "Second"
    root, kid, extra, lib, other = scene.nodes
    assert root.name == "Root" and root.mesh is None
    assert root.extras["sid"] == "r" and root.extras["type"] == "JOINT"
    assert root.extras["camera"] == "cam"
    assert root.children == (1, 3)
    assert kid.mesh == 0 and kid.children == (2,)
    assert extra.name == "B" and extra.mesh == 1
    assert lib.name == "Lib" and lib.mesh == 1
    np.testing.assert_array_equal(lib.matrix[:3, 3], [0, 0, 1])
    assert other.name == "" and other.children == ()


def test_read_input_without_semantic_dropped(tmp_path: Path) -> None:
    uv = _source("g0-uv", [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]], "S T")
    prims = _triangles(
        "g0", [[0, 1, 2]], extra_inputs='<input source="#g0-uv" offset="0"/>'
    )
    geo = _geometry("g0", _TRI, prims, sources=uv)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="has no semantic"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].vertex_attrs == {}


def test_write_mesh_name_that_is_not_a_string_spelled_as_one(tmp_path: Path) -> None:
    mesh = dataclasses.replace(_surface(), global_attrs={"mesh_name": 7})
    out = tmp_path / "seven.dae"
    write_scene(SceneData(meshes=(mesh,), nodes=(SceneNode(mesh=0),)), out)
    assert read_scene(out).meshes[0].global_attrs["mesh_name"] == "7"


def test_geometry_name_is_mesh_name_both_ways(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]), name="Hull")
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    scene = read_scene(_write(tmp_path, text))
    assert scene.meshes[0].global_attrs == {"mesh_name": "Hull"}
    out = tmp_path / "named.dae"
    write_scene(scene, out)
    assert '<geometry id="geometry0" name="Hull">' in out.read_text()
    assert read_scene(out).meshes[0].global_attrs["mesh_name"] == "Hull"


def test_read_every_library_of_a_kind(tmp_path: Path) -> None:
    geos = (
        f"<library_geometries>{_geometry('g0', _TRI, _triangles('g0', [[0, 1, 2]]))}"
        "</library_geometries><library_geometries>"
        f"{_geometry('g1', _QUAD, _triangles('g1', [[0, 1, 2]], material='sym'))}"
        "</library_geometries>"
    )
    effects = (
        '<library_effects><effect id="fx"><profile_COMMON><technique sid="c">'
        "<lambert><diffuse><color>1 0 0 1</color></diffuse></lambert>"
        "</technique></profile_COMMON></effect></library_effects>"
    )
    materials = (
        '<library_materials><material id="m0"><instance_effect url="#fx"/>'
        "</material></library_materials><library_materials>"
        '<material id="m1"><instance_effect url="#fx"/></material>'
        "</library_materials>"
    )
    images = (
        '<library_images><image id="i0"><init_from>a.png</init_from></image>'
        '</library_images><library_images><image id="i1"><init_from>b.png'
        "</init_from></image></library_images>"
    )
    bind = (
        '<bind_material><technique_common><instance_material symbol="sym" '
        'target="#m1"/></technique_common></bind_material>'
    )
    scenes = (
        '<library_visual_scenes><visual_scene id="S0"><node id="n0">'
        '<instance_geometry url="#g0"/></node></visual_scene></library_visual_scenes>'
        '<library_visual_scenes><visual_scene id="S1"><node id="n1">'
        '<translate sid="loc">0 0 0</translate>'
        f'<instance_geometry url="#g1">{bind}</instance_geometry></node>'
        "</visual_scene></library_visual_scenes>"
        '<scene><instance_visual_scene url="#S1"/></scene>'
    )
    text = _dae(images + effects + materials + geos + scenes)
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 2
    assert [img.uri for img in scene.images] == ["a.png", "b.png"]
    assert len(scene.materials) == 2
    assert scene.meshes[1].element_attrs["material"].tolist() == [1]
    assert scene.scenes == ((0,), (1,)) and scene.active_scene == 1


def test_read_instance_node_cycle_refused(tmp_path: Path) -> None:
    text = _dae(
        '<library_nodes><node id="a"><instance_node url="#b"/></node>'
        '<node id="b"><instance_node url="#a"/></node></library_nodes>'
        '<library_visual_scenes><visual_scene id="S"><instance_node url="#a"/>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="itself"):
        read_scene(_write(tmp_path, text))


def test_read_node_instancing_another_node_with_its_id_is_not_a_cycle(
    tmp_path: Path,
) -> None:
    text = _dae(
        '<library_nodes><node id="X" name="lib"/></library_nodes>'
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="X" name="top"><instance_node url="#X"/></node>'
        "</visual_scene></library_visual_scenes>"
    )
    scene = read_scene(_write(tmp_path, text))
    top = scene.nodes[scene.scenes[0][0]]
    assert top.name == "top"
    assert [scene.nodes[c].name for c in top.children] == ["lib"]


def test_read_no_visual_scene_synthesises_nodes(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    scene = read_scene(
        _write(tmp_path, _dae(f"<library_geometries>{geo}</library_geometries>"))
    )
    assert scene.nodes == (SceneNode(mesh=0),)
    assert scene.scenes == ((0,),)


def test_read_empty_visual_scenes_with_scene_on_the_second(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S1" name="First"/>'
        '<visual_scene id="S2" name="Second"/></library_visual_scenes>'
        '<scene><instance_visual_scene url="#S2"/></scene>'
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.scenes == ((0,),)
    assert scene.active_scene == 0
    assert scene.name == "Second"
    assert len(scene.to_polydata().vertices) == 3


def _copies_dae() -> str:
    """A two-node library subtree instanced by two scene nodes."""
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    return _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_nodes><node id="lib" name="L"><translate sid="t">1 2 3</translate>'
        '<node id="tip" sid="tip" name="T"><instance_geometry url="#g0"/></node>'
        "</node></library_nodes>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_node url="#lib"/></node>'
        '<node id="b"><instance_node url="#lib"/></node>'
        "</visual_scene></library_visual_scenes>"
    )


def _shape(scene: SceneData, idx: int) -> tuple:
    node = scene.nodes[idx]
    return (
        node.name,
        node.mesh,
        node.matrix.tolist(),
        tuple(_shape(scene, c) for c in node.children),
    )


def test_write_instanced_subtree_round_trips(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _copies_dae()))
    assert len(scene.nodes) == 6
    out = tmp_path / "copies.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    text = out.read_text()
    assert text.count('<instance_node url="#lib"/>') == 2
    assert text.count(' id="tip"') == 1
    lib = ET.fromstring(text).find(f"{{{_NS}}}library_nodes")
    assert [n.get("id") for n in lib] == ["lib"]
    back = read_scene(out)
    assert [_shape(back, r) for r in back.scenes[0]] == [
        _shape(scene, r) for r in scene.scenes[0]
    ]


def test_write_edited_copy_written_in_full(tmp_path: Path) -> None:
    """A copy moved after reading is no longer the node it copied; its
    untouched child still is."""
    scene = read_scene(_write(tmp_path, _copies_dae()))
    copy = scene.nodes[4]
    assert copy.extras["id"] == "lib"
    moved = dataclasses.replace(copy, matrix=copy.matrix @ np.diag([2.0, 2, 2, 1]))
    nodes = scene.nodes[:4] + (moved,) + scene.nodes[5:]
    scene = dataclasses.replace(scene, nodes=nodes)
    out = tmp_path / "edited.dae"
    with pytest.warns(UserWarning, match=r"node\(s\) \[4\] repeat an id"):
        write_scene(scene, out)
    text = out.read_text()
    assert "#lib" not in text
    assert text.count('<instance_node url="#tip"/>') == 2
    back = read_scene(out)
    np.testing.assert_allclose(back.nodes[4].matrix, moved.matrix)


def test_write_copy_as_root_of_a_second_scene(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _copies_dae()))
    scene = dataclasses.replace(scene, scenes=((0, 3), (4,)))
    out = tmp_path / "two.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out)
    assert out.read_text().count('<instance_node url="#lib"/>') == 3
    assert [_shape(back, r) for r in back.scenes[1]] == [
        ("", None, np.eye(4).tolist(), (_shape(scene, 4),))
    ]


def test_read_duplicate_geometry_id_instances_the_first(tmp_path: Path) -> None:
    first = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    second = _geometry("g0", _QUAD, _triangles("g0", [[0, 1, 2], [0, 2, 3]]))
    second = second.replace('id="g0-', 'id="h0-').replace('"#g0-', '"#h0-')
    text = _dae(
        f"<library_geometries>{first}{second}</library_geometries>" + _scene_with("g0")
    )
    scene = read_scene(_write(tmp_path, text))
    assert scene.nodes[0].mesh == 0
    assert len(scene.meshes[0].vertices) == 3


def test_read_deep_node_chain_refused_as_codec_error(tmp_path: Path) -> None:
    depth = 2000
    chain = "<node>" * depth + "</node>" * depth
    text = _dae(
        f'<library_visual_scenes><visual_scene id="S">{chain}</visual_scene>'
        "</library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="deeper"):
        read_scene(_write(tmp_path, text))
    nodes = tuple(SceneNode(children=(i + 1,)) for i in range(depth - 1)) + (
        SceneNode(),
    )
    with pytest.raises(CodecError, match="deeper"):
        write_scene(
            SceneData(meshes=(), nodes=nodes, scenes=((0,),)), tmp_path / "deep.dae"
        )


def test_read_asset(tmp_path: Path) -> None:
    asset = (
        "<contributor><authoring_tool>Blender</authoring_tool></contributor>"
        "<created>2020-01-01T00:00:00</created>"
        '<unit name="centimeter" meter="0.01"/><up_axis>Z_UP</up_axis>'
    )
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    scene = read_scene(
        _write(
            tmp_path,
            _dae(f"<library_geometries>{geo}</library_geometries>", asset=asset),
        )
    )
    assert scene.global_attrs["asset"] == {
        "up_axis": "Z_UP",
        "unit": {"name": "centimeter", "meter": 0.01},
        "authoring_tool": "Blender",
        "created": "2020-01-01T00:00:00",
    }


def test_read_instance_node_of_a_geometry_refused(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="n"><instance_node url="#g0"/></node>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="not a node"):
        read_scene(_write(tmp_path, text))


def test_read_instance_node_fan_out_capped(tmp_path: Path) -> None:
    depth = 20
    lib = "".join(
        f'<node id="n{i}"><instance_node url="#n{i + 1}"/>'
        f'<instance_node url="#n{i + 1}"/></node>'
        for i in range(depth)
    )
    text = _dae(
        f'<library_nodes>{lib}<node id="n{depth}"/></library_nodes>'
        '<library_visual_scenes><visual_scene id="S"><instance_node url="#n0"/>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="expands the node tree past 262144"):
        read_scene(_write(tmp_path, text))


def test_read_external_instances_dropped_with_warning(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_geometry url="#g0"/>'
        '<instance_node url="other.dae#lamp"/></node>'
        '<node id="b"><instance_geometry url="parts.dae#bolt"/></node>'
        '<node id="c"><instance_controller url="rig.dae#skin"/></node>'
        '<instance_node url="lib.dae#root"/>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.warns(UserWarning, match="external document") as record:
        scene = read_scene(_write(tmp_path, text))
    assert sum("external document" in str(w.message) for w in record) == 4
    assert scene.scenes == ((0, 1, 2),)
    assert [n.mesh for n in scene.nodes] == [0, None, None]
    assert scene.nodes[0].children == ()


def test_read_external_effect_and_visual_scene_dropped_with_warning(
    tmp_path: Path,
) -> None:
    text = (
        _effect_dae("<phong/>")
        .replace('<instance_effect url="#fx0"/>', '<instance_effect url="fx.dae#fx"/>')
        .replace('url="#Scene"', 'url="scenes.dae#Main"')
    )
    with pytest.warns(UserWarning, match="external document") as record:
        scene = read_scene(_write(tmp_path, text))
    assert sum("external document" in str(w.message) for w in record) == 2
    assert scene.materials[0].name == "Red"
    assert scene.materials[0].extras == {"shading": "phong"}
    assert scene.active_scene == 0
    assert scene.nodes[0].mesh == 0


# ---------------------------------------------------------------------------
# skins
# ---------------------------------------------------------------------------


def test_read_morph_controller_warns(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_controllers><controller id="m"><morph source="#g0"/></controller>'
        "</library_controllers>"
        '<library_visual_scenes><visual_scene id="S"><node id="n">'
        '<instance_controller url="#m"/></node></visual_scene></library_visual_scenes>'
    )
    with pytest.warns(UserWarning, match="morph"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.nodes[0].mesh == 0
    assert "skins" not in scene.global_attrs


def _skin_controller(
    cid: str,
    names: list[str],
    v: str,
    vcount: str,
    *,
    source: str = "#g0",
    kind: str = "Name_array",
    ibm=None,
    weights=(1.0,),
    bind_shape: str | None = None,
) -> str:
    """A ``<controller>`` whose ``<v>`` pairs index ``names`` and ``weights``."""
    count = len(vcount.split())
    ibm_xml = ""
    ibm_input = ""
    if ibm is not None:
        ibm_xml = _source(f"{cid}-m", np.asarray(ibm).reshape(-1, 16), "TRANSFORM")
        ibm_input = f'<input semantic="INV_BIND_MATRIX" source="#{cid}-m"/>'
    bsm = (
        ""
        if bind_shape is None
        else f"<bind_shape_matrix>{bind_shape}</bind_shape_matrix>"
    )
    return (
        f'<controller id="{cid}"><skin source="{source}">{bsm}'
        + _source(f"{cid}-j", names, "JOINT", kind=kind)
        + ibm_xml
        + _source(f"{cid}-w", list(weights), "WEIGHT")
        + f'<joints><input semantic="JOINT" source="#{cid}-j"/>{ibm_input}</joints>'
        f'<vertex_weights count="{count}">'
        f'<input semantic="JOINT" source="#{cid}-j" offset="0"/>'
        f'<input semantic="WEIGHT" source="#{cid}-w" offset="1"/>'
        f"<vcount>{vcount}</vcount><v>{v}</v></vertex_weights>"
        "</skin></controller>"
    )


def _skin_dae(controllers: str, nodes: str, *, library_nodes: str = "") -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    lib = f"<library_nodes>{library_nodes}</library_nodes>" if library_nodes else ""
    return _dae(
        f"<library_geometries>{geo}</library_geometries>"
        f"<library_controllers>{controllers}</library_controllers>{lib}"
        f'<library_visual_scenes><visual_scene id="S">{nodes}</visual_scene>'
        "</library_visual_scenes>"
    )


def _rig(prefix: str = "") -> str:
    """A hip joint with a knee below it, ids prefixed, sids bare."""
    return (
        f'<node id="{prefix}hip" sid="hip" type="JOINT">'
        f'<node id="{prefix}knee" sid="knee" type="JOINT">'
        "<translate>0 1 0</translate></node></node>"
    )


def _hip_knee(cid: str = "c", **kw) -> str:
    """Vertex 0 split half and half, vertex 1 on the hip, vertex 2 on the knee."""
    return _skin_controller(
        cid, ["hip", "knee"], "0 0 1 0 0 1 1 1", "2 1 1", weights=(0.5, 1.0), **kw
    )


def test_read_skin_attaches_joints_weights_and_a_skin(tmp_path: Path) -> None:
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[1, 1, 3] = -1.0
    text = _skin_dae(
        _hip_knee(ibm=ibm, bind_shape="1 0 0 5 0 1 0 0 0 0 1 0 0 0 0 1"),
        _rig() + '<node id="n"><instance_controller url="#c"><skeleton>#hip</skeleton>'
        "</instance_controller></node>",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, text))
    mesh = scene.meshes[0]
    np.testing.assert_array_equal(mesh.vertices, _TRI)
    np.testing.assert_array_equal(
        mesh.vertex_attrs["joints"], [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
    )
    assert mesh.vertex_attrs["joints"].dtype == np.int32
    np.testing.assert_array_equal(
        mesh.vertex_attrs["weights"], [[0.5, 0.5, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    )
    (skin,) = scene.global_attrs["skins"]
    assert skin["name"] == "c"
    assert skin["joints"] == [0, 1]
    assert skin["skeleton"] == 0
    np.testing.assert_array_equal(skin["inverse_bind_matrices"], ibm)
    assert skin["bind_shape_matrix"][0, 3] == 5.0
    assert scene.nodes[2].extras["skin"] == 0 and scene.nodes[2].mesh == 0


def test_read_skin_without_inverse_bind_matrices_or_bind_shape(tmp_path: Path) -> None:
    text = _skin_dae(
        _hip_knee(), _rig() + '<node id="n"><instance_controller url="#c"/></node>'
    )
    (skin,) = read_scene(_write(tmp_path, text)).global_attrs["skins"]
    np.testing.assert_array_equal(
        skin["inverse_bind_matrices"], np.tile(np.eye(4), (2, 1, 1))
    )
    assert "bind_shape_matrix" not in skin and "skeleton" not in skin
    assert skin["joints"] == [0, 1]


@pytest.mark.parametrize(
    ("kind", "names"),
    [
        ("IDREF_array", ["hipnode", "kneenode"]),
        ("SIDREF_array", ["hipnode/knee", "hip"]),
        ("Name_array", ["hipnode", "kneenode"]),
    ],
)
def test_read_skin_joint_names_by_id_or_address(
    tmp_path: Path, kind: str, names: list[str]
) -> None:
    ctrl = _skin_controller("c", names, "0 0 0 0 0 0", "1 1 1", kind=kind)
    rig = (
        '<node id="hipnode" sid="hip" type="JOINT">'
        '<node id="kneenode" sid="knee" type="JOINT"/></node>'
    )
    scene = read_scene(
        _write(
            tmp_path,
            _skin_dae(
                ctrl, rig + '<node id="n"><instance_controller url="#c"/></node>'
            ),
        )
    )
    expected = [1, 0] if kind == "SIDREF_array" else [0, 1]
    assert scene.global_attrs["skins"][0]["joints"] == expected


def test_read_skin_binds_each_rig_by_its_skeleton(tmp_path: Path) -> None:
    """Two rigs spell the same sids; each instance's <skeleton> picks its own."""
    text = _skin_dae(
        _hip_knee(),
        _rig("a_")
        + _rig("b_")
        + '<node id="n1"><instance_controller url="#c"><skeleton>#b_hip</skeleton>'
        "</instance_controller></node>"
        '<node id="n2"><instance_controller url="#c"><skeleton>#a_hip</skeleton>'
        "</instance_controller></node>",
    )
    scene = read_scene(_write(tmp_path, text))
    skins = scene.global_attrs["skins"]
    assert [s["joints"] for s in skins] == [[2, 3], [0, 1]]
    assert [scene.nodes[i].extras["skin"] for i in (4, 5)] == [0, 1]
    assert scene.nodes[4].mesh == scene.nodes[5].mesh == 0
    assert len(scene.meshes) == 1
    skins[0]["inverse_bind_matrices"][0, 0, 0] = 9.0
    assert skins[1]["inverse_bind_matrices"][0, 0, 0] == 1.0


def test_read_skin_without_skeleton_binds_the_nearest_rig(tmp_path: Path) -> None:
    text = _skin_dae(
        _hip_knee(),
        '<node id="A">' + _rig("a_") + '<node id="na"><instance_controller url="#c"/>'
        '</node></node><node id="B">'
        + _rig("b_")
        + '<node id="nb"><instance_controller url="#c"/></node></node>',
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, text))
    by_node = {
        scene.nodes[i].extras["id"]: scene.global_attrs["skins"][
            scene.nodes[i].extras["skin"]
        ]["joints"]
        for i in range(len(scene.nodes))
        if "skin" in scene.nodes[i].extras
    }
    ids = [n.extras.get("id") for n in scene.nodes]
    assert by_node == {
        "na": [ids.index("a_hip"), ids.index("a_knee")],
        "nb": [ids.index("b_hip"), ids.index("b_knee")],
    }


def test_read_skin_tie_warns_and_binds_the_first(tmp_path: Path) -> None:
    text = _skin_dae(
        _hip_knee(),
        _rig("a_") + _rig("b_") + '<node id="n"><instance_controller url="#c"/></node>',
    )
    with pytest.warns(UserWarning, match=r"joint\(s\) \['hip', 'knee'\] that several"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["skins"][0]["joints"] == [0, 1]


def test_read_skin_unresolved_names_are_minus_one(tmp_path: Path) -> None:
    text = _skin_dae(
        _hip_knee(),
        '<node id="hip" sid="hip"/><node id="n"><instance_controller url="#c">'
        "<skeleton>#nowhere</skeleton></instance_controller></node>",
    )
    with (
        pytest.warns(UserWarning, match=r"<skeleton> root\(s\) \['nowhere'\]"),
        pytest.warns(UserWarning, match=r"joint\(s\) \['knee'\] that no node"),
    ):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["skins"][0]["joints"] == [0, -1]


def test_read_skinned_instance_copies_bind_their_own_rig(tmp_path: Path) -> None:
    character = (
        '<node id="char">' + _rig() + '<node id="body"><instance_controller url="#c">'
        "<skeleton>#hip</skeleton></instance_controller></node></node>"
    )
    text = _skin_dae(
        _hip_knee(),
        '<node id="p1"><instance_node url="#char"/></node>'
        '<node id="p2"><instance_node url="#char"/></node>',
        library_nodes=character,
    )
    scene = read_scene(_write(tmp_path, text))
    skins = scene.global_attrs["skins"]
    assert len(skins) == 2
    holders = [i for i, n in enumerate(scene.nodes) if "skin" in n.extras]
    for i, skin in zip(holders, skins, strict=True):
        assert skin["skeleton"] == skin["joints"][0]
        assert scene.nodes[skin["joints"][0]].extras["id"] == "hip"
        # Each copy's rig sits beside its own body, under its own char copy.
        parent = next(p for p, n in enumerate(scene.nodes) if i in n.children)
        assert skin["joints"][0] in scene.nodes[parent].children
    assert skins[0]["inverse_bind_matrices"] is not skins[1]["inverse_bind_matrices"]


def test_read_geometry_plain_then_skinned_gets_a_skinned_variant(
    tmp_path: Path,
) -> None:
    text = _skin_dae(
        _hip_knee(),
        _rig() + '<node id="plain"><instance_geometry url="#g0"/></node>'
        '<node id="n"><instance_controller url="#c"/></node>',
    )
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 2
    plain, skinned = scene.meshes
    assert "joints" not in plain.vertex_attrs
    assert skinned.vertices is plain.vertices
    np.testing.assert_array_equal(skinned.vertex_attrs["weights"][:, 0], [0.5, 1, 1])
    assert [n.mesh for n in scene.nodes if n.extras.get("id") in ("plain", "n")] == [
        0,
        1,
    ]


def test_read_skinned_geometry_bound_to_two_materials(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="sym"))
    mats = (
        '<library_effects><effect id="e"><profile_COMMON><technique sid="t"><phong/>'
        "</technique></profile_COMMON></effect></library_effects><library_materials>"
        '<material id="m0"><instance_effect url="#e"/></material>'
        '<material id="m1"><instance_effect url="#e"/></material></library_materials>'
    )

    def bound(m: str) -> str:
        return (
            "<bind_material><technique_common>"
            f'<instance_material symbol="sym" target="#{m}"/>'
            "</technique_common></bind_material>"
        )

    text = _dae(
        mats + f"<library_geometries>{geo}</library_geometries>"
        f"<library_controllers>{_hip_knee()}</library_controllers>"
        '<library_visual_scenes><visual_scene id="S">'
        + _rig()
        + f'<node id="a"><instance_controller url="#c">{bound("m0")}'
        "</instance_controller></node>"
        f'<node id="b"><instance_controller url="#c">{bound("m1")}'
        "</instance_controller></node>"
        f'<node id="c"><instance_geometry url="#g0">{bound("m1")}'
        "</instance_geometry></node>"
        "</visual_scene></library_visual_scenes>"
    )
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 3
    a, b, c = (scene.meshes[n.mesh] for n in scene.nodes[2:])
    assert a.element_attrs["material"].tolist() == [0]
    assert b.element_attrs["material"].tolist() == [1]
    assert c.element_attrs["material"].tolist() == [1]
    assert "joints" in a.vertex_attrs and "joints" in b.vertex_attrs
    assert "joints" not in c.vertex_attrs
    assert scene.nodes[2].extras["skin"] == scene.nodes[3].extras["skin"] == 0


def test_read_skin_split_corners_take_their_position_weights(tmp_path: Path) -> None:
    normals = _source("g0-n", [[0, 0, 1], [0, 0, -1]], "X Y Z")
    prims = _triangles(
        "g0",
        [[0, 0, 1, 0, 2, 1], [0, 1, 2, 0, 1, 1]],
        extra_inputs='<input semantic="NORMAL" source="#g0-n" offset="1"/>',
    ).replace('offset="0"/>', 'offset="0"/>', 1)
    geo = _geometry("g0", _TRI, prims, sources=normals)
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        f"<library_controllers>{_hip_knee()}</library_controllers>"
        '<library_visual_scenes><visual_scene id="S">'
        + _rig()
        + '<node id="n"><instance_controller url="#c"/></node>'
        "</visual_scene></library_visual_scenes>"
    )
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert len(mesh.vertices) > 3
    for v, w in zip(mesh.vertices, mesh.vertex_attrs["weights"], strict=True):
        pos = int(np.flatnonzero((_TRI == v).all(axis=1))[0])
        assert w[0] == (0.5 if pos == 0 else 1.0)


def test_read_skin_crowded_and_bind_shape_influences_renormalised(
    tmp_path: Path,
) -> None:
    names = ["a", "b", "c", "d", "e"]
    rig = "".join(f'<node id="{n}" sid="{n}"/>' for n in names)
    # Vertex 0: five influences; vertex 1: half on the bind shape; vertex 2: none.
    ctrl = _skin_controller(
        "c",
        names,
        "0 0 1 1 2 2 3 3 4 4 -1 5 1 5",
        "5 2 0",
        weights=(0.1, 0.2, 0.3, 0.15, 0.25, 0.5),
    )
    text = _skin_dae(ctrl, rig + '<node id="n"><instance_controller url="#c"/></node>')
    with (
        pytest.warns(UserWarning, match="more than four influences"),
        pytest.warns(UserWarning, match=r"weights 1 vertex\(es\) to the bind shape"),
    ):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    j, w = mesh.vertex_attrs["joints"], mesh.vertex_attrs["weights"]
    assert j[0].tolist() == [2, 4, 1, 3]
    np.testing.assert_allclose(w[0], np.array([0.3, 0.25, 0.2, 0.15]) / 0.9)
    assert j[1].tolist() == [1, 0, 0, 0]
    np.testing.assert_allclose(w[1], [1, 0, 0, 0])
    assert w[2].tolist() == [0, 0, 0, 0]


def test_read_skin_short_vertex_weights_warns(tmp_path: Path) -> None:
    ctrl = _skin_controller("c", ["hip"], "0 0", "1")
    text = _skin_dae(
        ctrl,
        '<node id="hip" sid="hip"/><node id="n"><instance_controller url="#c"/></node>',
    )
    with pytest.warns(UserWarning, match="covers 1 of the geometry's 3 vertices"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.vertex_attrs["weights"][:, 0].tolist() == [1, 0, 0]


def test_read_skin_vertex_weights_joint_source_mapped_by_name(tmp_path: Path) -> None:
    ctrl = (
        '<controller id="c"><skin source="#g0">'
        + _source("c-j", ["hip", "knee"], "JOINT", kind="Name_array")
        + _source("c-k", ["knee", "hip"], "JOINT", kind="Name_array")
        + _source("c-w", [1.0], "WEIGHT")
        + '<joints><input semantic="JOINT" source="#c-j"/></joints>'
        '<vertex_weights count="3"><input semantic="JOINT" source="#c-k" offset="0"/>'
        '<input semantic="WEIGHT" source="#c-w" offset="1"/>'
        "<vcount>1 1 1</vcount><v>0 0 1 0 0 0</v></vertex_weights></skin></controller>"
    )
    text = _skin_dae(
        ctrl, _rig() + '<node id="n"><instance_controller url="#c"/></node>'
    )
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert mesh.vertex_attrs["joints"][:, 0].tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    ("ctrl", "match"),
    [
        (
            '<controller id="c"><skin source="#g0"><joints/></skin></controller>',
            "no JOINT input",
        ),
        (_skin_controller("c", ["hip"], "0 0 0 0", "1 1 1"), "sums to 6 indices"),
        (
            _skin_controller("c", ["hip"], "0 0 0 0 0 0", "1 1 1").replace(
                'vertex_weights count="3"', 'vertex_weights count="2"'
            ),
            "count=2 but <vcount> holds 3",
        ),
        (_skin_controller("c", ["hip"], "0 0 0 0 0 0 0 0", "1 1 1 1"), "covers 4"),
        (_skin_controller("c", ["hip"], "0 0 0 0 0 9", "1 1 1"), "past the WEIGHT"),
        (_skin_controller("c", ["hip"], "0 0 0 0 1 0", "1 1 1"), "past the JOINT"),
        (_skin_controller("c", ["hip"], "0 0 0 0 -2 0", "1 1 1"), "only -1"),
        (_skin_controller("c", ["hip"], "0 0 0 0", "1 -1 2"), "below 0"),
        (_skin_controller("c", ["hip"], "0 0", "99 0 0"), "entry of 99"),
        (
            _skin_controller(
                "c", ["hip"], "0 0 0 0 0 0", "1 1 1", ibm=np.eye(4)[None].repeat(2, 0)
            ),
            "1 joints but holds 2",
        ),
        (
            _skin_controller("c", ["hip"], "0 0 0 0 0 0", "1 1 1", bind_shape="1 0 0"),
            "bind_shape_matrix",
        ),
        (
            _skin_controller("c", ["hip"], "0 0 0 0 0 0", "1 1 1").replace(
                '<input semantic="WEIGHT" source="#c-w" offset="1"/>', ""
            ),
            "needs JOINT and WEIGHT",
        ),
    ],
)
def test_read_skin_malformed_is_refused(tmp_path: Path, ctrl: str, match: str) -> None:
    text = _skin_dae(
        ctrl,
        '<node id="hip" sid="hip"/><node id="n"><instance_controller url="#c"/></node>',
    )
    with pytest.raises(CodecError, match=match):
        read_scene(_write(tmp_path, text))


def test_read_skin_inverse_bind_stride_refused(tmp_path: Path) -> None:
    ctrl = (
        _skin_controller("c", ["hip"], "0 0 0 0 0 0", "1 1 1", ibm=np.eye(4))
        .replace(
            'stride="16"><param name="TRANSFORM" type="float4x4"/>',
            'stride="8"><param name="TRANSFORM" type="float4x4"/>',
        )
        .replace('count="1" stride="8"', 'count="2" stride="8"')
    )
    text = _skin_dae(
        ctrl,
        '<node id="hip" sid="hip"/><node id="n"><instance_controller url="#c"/></node>',
    )
    with pytest.raises(CodecError, match="stride 8, not 16"):
        read_scene(_write(tmp_path, text))


def test_read_skin_copies_are_capped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_collada, "_MIN_COPY_CAP", 0)
    names = [f"j{k}" for k in range(200)]
    rig = "".join(f'<node id="j{k}" sid="j{k}"/>' for k in range(200))
    character = (
        f'<node id="char">{rig}<node id="body"><instance_controller url="#c"/>'
        "</node></node>"
    )
    ctrl = _skin_controller(
        "c", names, "0 0 0 0 0 0", "1 1 1", ibm=np.tile(np.eye(4), (200, 1, 1))
    )
    parents = "".join(
        f'<node id="p{k}"><instance_node url="#char"/></node>' for k in range(40)
    )
    text = _skin_dae(ctrl, parents, library_nodes=character)
    with pytest.raises(CodecError, match="instances of skin 'c' need copies past"):
        read_scene(_write(tmp_path, text))


def test_read_skin_bindings_are_capped(tmp_path: Path, monkeypatch) -> None:
    """Without a <skeleton>, each instance in its own place binds afresh."""
    monkeypatch.setattr(_collada, "_MIN_NODE_CAP", 0)
    names = [f"j{k}" for k in range(2000)]
    rig = "".join(f'<node id="j{k}" sid="j{k}"/>' for k in range(2000))
    character = (
        '<node id="char"><node id="body"><instance_controller url="#c"/></node></node>'
    )
    ctrl = _skin_controller("c", names, "0 0 0 0 0 0", "1 1 1")
    parents = "".join(
        f'<node id="p{k}"><instance_node url="#char"/></node>' for k in range(300)
    )
    text = _skin_dae(ctrl, rig + parents, library_nodes=character)
    with pytest.raises(CodecError, match="bind joints past"):
        read_scene(_write(tmp_path, text))


def test_read_skin_shared_skeleton_binds_once(tmp_path: Path, monkeypatch) -> None:
    """The same <skeleton> picks the same joints wherever the instance sits."""
    monkeypatch.setattr(_collada, "_MIN_NODE_CAP", 0)
    names = [f"j{k}" for k in range(2000)]
    rig = (
        '<node id="rig">'
        + "".join(f'<node id="j{k}" sid="j{k}"/>' for k in range(2000))
        + "</node>"
    )
    character = (
        '<node id="char"><node id="body"><instance_controller url="#c">'
        "<skeleton>#rig</skeleton></instance_controller></node></node>"
    )
    ctrl = _skin_controller("c", names, "0 0 0 0 0 0", "1 1 1")
    parents = "".join(
        f'<node id="p{k}"><instance_node url="#char"/></node>' for k in range(300)
    )
    text = _skin_dae(ctrl, rig + parents, library_nodes=character)
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.global_attrs["skins"]) == 1


def test_write_skin_round_trips_a_collada_rig(tmp_path: Path) -> None:
    ibm = np.tile(np.eye(4), (2, 1, 1))
    ibm[1, 1, 3] = -1.0
    text = _skin_dae(
        _hip_knee(ibm=ibm, bind_shape="1 0 0 5 0 1 0 0 0 0 1 0 0 0 0 1"),
        _rig() + '<node id="n"><instance_controller url="#c"><skeleton>#hip</skeleton>'
        "</instance_controller></node>",
    )
    scene = read_scene(_write(tmp_path, text))
    out = tmp_path / "again.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    written = out.read_text()
    assert (
        '<Name_array id="controller0-joints-array" count="2">hip knee</Name_array>'
        in written
    )
    assert "<skeleton>#hip</skeleton>" in written
    back = read_scene(out)
    assert back.global_attrs["skins"][0]["name"] == "c"
    assert SceneData(meshes=back.meshes, nodes=()) == SceneData(
        meshes=scene.meshes, nodes=()
    )
    assert [n.extras.get("skin") for n in back.nodes] == [None, None, 0]
    for key, value in scene.global_attrs["skins"][0].items():
        np.testing.assert_array_equal(back.global_attrs["skins"][0][key], value)


def test_write_skin_from_gltf_round_trips(tmp_path: Path) -> None:
    data, ibm = _skinned_glb()
    glb = tmp_path / "skin.glb"
    glb.write_bytes(data)
    scene = gltf_read_scene(glb)
    out = tmp_path / "skin.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out)
    (skin,) = back.global_attrs["skins"]
    assert skin["joints"] == [1, 2]
    assert skin["skeleton"] == 1
    np.testing.assert_array_equal(skin["inverse_bind_matrices"], ibm)
    assert back.nodes[0].extras["skin"] == 0
    mesh, original = back.meshes[0], scene.meshes[0]
    np.testing.assert_array_equal(
        mesh.vertex_attrs["joints"], original.vertex_attrs["joints"]
    )
    np.testing.assert_allclose(
        mesh.vertex_attrs["weights"], original.vertex_attrs["weights"]
    )
    assert [back.nodes[j].extras.get("type") for j in (1, 2)] == ["JOINT", "JOINT"]
    np.testing.assert_allclose(
        back.to_polydata().vertices, scene.to_polydata().vertices
    )


def test_write_skin_zero_weights_pack_to_the_front(tmp_path: Path) -> None:
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "joints": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], np.uint8),
            "weights": np.array([[0.0, 1, 0, 0], [0.25, 0.75, 0, 0], [0, 0, 0, 0]]),
        },
    )
    scene = _skinned_scene(poly)
    out = tmp_path / "zero.dae"
    write_scene(scene, out)
    mesh = read_scene(out).meshes[0]
    assert mesh.vertex_attrs["joints"].tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert mesh.vertex_attrs["weights"].tolist() == [
        [1, 0, 0, 0],
        [0.25, 0.75, 0, 0],
        [0, 0, 0, 0],
    ]


def _skinned_scene(poly: PolyData, **skin: Any) -> SceneData:
    """A mesh node skinned to a two-joint rig, nodes 1 and 2."""
    return SceneData(
        meshes=(poly,),
        nodes=(
            SceneNode(mesh=0, extras={"skin": 0}),
            SceneNode(name="hip", children=(2,)),
            SceneNode(name="knee"),
        ),
        scenes=((0, 1),),
        global_attrs={"skins": [{"joints": [1, 2], **skin}]},
    )


def _weighted() -> PolyData:
    return make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "joints": np.array([[0, 1, 0, 0]] * 3, np.int32),
            "weights": np.array([[0.5, 0.5, 0, 0]] * 3),
        },
    )


def test_write_skin_names_repeated_sids_by_id(tmp_path: Path) -> None:
    """Two rigs under one root share sids; the joints go out as ids."""
    scene = SceneData(
        meshes=(_weighted(),),
        nodes=(
            SceneNode(mesh=0, extras={"skin": 0}),
            SceneNode(name="root", children=(2, 3)),
            SceneNode(name="a", extras={"sid": "bone"}),
            SceneNode(name="b", extras={"sid": "bone"}),
        ),
        scenes=((0, 1),),
        global_attrs={"skins": [{"joints": [3, 2], "skeleton": 1}]},
    )
    out = tmp_path / "idref.dae"
    write_scene(scene, out)
    assert "<IDREF_array" in out.read_text()
    assert read_scene(out).global_attrs["skins"][0]["joints"] == [3, 2]


def test_write_skin_generated_sid_avoids_a_given_one(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_weighted(),),
        nodes=(
            SceneNode(mesh=0, extras={"skin": 0}),
            SceneNode(name="hip", children=(2,), extras={"sid": "node2"}),
            SceneNode(name="knee"),
        ),
        scenes=((0, 1),),
        global_attrs={"skins": [{"joints": [1, 2]}]},
    )
    out = tmp_path / "sid.dae"
    write_scene(scene, out)
    assert "<Name_array" in out.read_text()
    back = read_scene(out)
    assert back.global_attrs["skins"][0]["joints"] == [1, 2]
    assert back.nodes[1].extras["sid"] == "node2"
    assert back.nodes[2].extras["sid"] not in ("node2", None)


def test_write_skin_on_a_shared_rig_names_joints_by_id(tmp_path: Path) -> None:
    """A rig under two parents is written once and copied; ids reach each copy."""
    scene = SceneData(
        meshes=(_weighted(),),
        nodes=(
            SceneNode(name="p1", children=(2, 4)),
            SceneNode(name="p2", children=(2,)),
            SceneNode(name="hip", children=(3,)),
            SceneNode(name="knee"),
            SceneNode(mesh=0, extras={"skin": 0}),
        ),
        scenes=((0, 1),),
        global_attrs={"skins": [{"joints": [2, 3]}]},
    )
    out = tmp_path / "shared.dae"
    write_scene(scene, out)
    assert "<IDREF_array" in out.read_text()
    back = read_scene(out)
    joints = back.global_attrs["skins"][0]["joints"]
    holder = next(i for i, n in enumerate(back.nodes) if "skin" in n.extras)
    parent = next(p for p, n in enumerate(back.nodes) if holder in n.children)
    assert joints[0] in back.nodes[parent].children


def test_write_skinned_instance_copies_round_trip(tmp_path: Path) -> None:
    character = (
        '<node id="char">' + _rig() + '<node id="body"><instance_controller url="#c">'
        "<skeleton>#hip</skeleton></instance_controller></node></node>"
    )
    text = _skin_dae(
        _hip_knee(),
        '<node id="p1"><instance_node url="#char"/></node>'
        '<node id="p2"><instance_node url="#char"/></node>',
        library_nodes=character,
    )
    scene = read_scene(_write(tmp_path, text))
    out = tmp_path / "copies.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    written = out.read_text()
    assert written.count("<controller ") == 1
    assert written.count('<instance_node url="#char"/>') == 2
    back = read_scene(out)
    assert len(back.global_attrs["skins"]) == 2
    assert [_shape(back, r) for r in back.scenes[0]] == [
        _shape(scene, r) for r in scene.scenes[0]
    ]


def test_write_skinned_copy_whose_skin_moved_elsewhere_is_written_in_full(
    tmp_path: Path,
) -> None:
    character = (
        '<node id="char">' + _rig() + '<node id="body"><instance_controller url="#c">'
        "<skeleton>#hip</skeleton></instance_controller></node></node>"
    )
    text = _skin_dae(
        _hip_knee(),
        '<node id="p1"><instance_node url="#char"/></node>'
        '<node id="p2"><instance_node url="#char"/></node>',
        library_nodes=character,
    )
    scene = read_scene(_write(tmp_path, text))
    skins = scene.global_attrs["skins"]
    skins[1] = {**skins[1], "joints": list(skins[0]["joints"])}
    with pytest.warns(UserWarning, match="generated"):
        write_scene(scene, tmp_path / "moved.dae")


def test_write_two_skins_over_one_mesh_get_a_controller_each(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_weighted(),),
        nodes=(
            SceneNode(mesh=0, extras={"skin": 0}),
            SceneNode(mesh=0, extras={"skin": 1}),
            SceneNode(mesh=0, extras={"skin": 0}),
            SceneNode(name="a"),
            SceneNode(name="b"),
        ),
        scenes=((0, 1, 2, 3, 4),),
        global_attrs={"skins": [{"joints": [3, 4]}, {"joints": [4, 3]}]},
    )
    out = tmp_path / "two.dae"
    write_scene(scene, out)
    assert out.read_text().count("<controller ") == 2
    back = read_scene(out)
    assert [s["joints"] for s in back.global_attrs["skins"]] == [[3, 4], [4, 3]]
    assert [n.extras.get("skin") for n in back.nodes[:3]] == [0, 1, 0]


@pytest.mark.parametrize(
    ("skin", "match"),
    [
        ("nope", "is not a dict"),
        ({}, "has no joints list"),
        ({"joints": [1, 9]}, r"joint\(s\) \[9\] that are not written"),
        ({"joints": [1, 2], "inverseBindMatrices": 0}, "never decoded"),
        ({"joints": [1, 2], "inverse_bind_matrices": np.eye(4)}, "not 2 finite 4x4"),
        (
            {"joints": [1, 2], "inverse_bind_matrices": np.full((2, 4, 4), np.nan)},
            "not 2 finite 4x4",
        ),
        ({"joints": [1, 2], "bind_shape_matrix": [1, 2]}, "bind_shape_matrix"),
        ({"joints": [1, 2], "bind_shape_matrix": "x"}, "bind_shape_matrix"),
        ({"joints": [1, 2], "inverse_bind_matrices": "x"}, "not 2 finite 4x4"),
    ],
)
def test_write_unwritable_skin_warns_and_writes_the_mesh_plain(
    tmp_path: Path, skin: Any, match: str
) -> None:
    scene = dataclasses.replace(
        _skinned_scene(_weighted()), global_attrs={"skins": [skin]}
    )
    out = tmp_path / "bad.dae"
    with (
        pytest.warns(UserWarning, match=match),
        pytest.warns(UserWarning, match=r"\['joints', 'weights'\].*dropped"),
    ):
        write_scene(scene, out)
    assert "<controller" not in out.read_text()
    assert "<instance_geometry" in out.read_text()


def test_write_skin_unknown_keys_dropped_with_warning(tmp_path: Path) -> None:
    scene = _skinned_scene(_weighted(), extras={"x": 1}, name="rig")
    out = tmp_path / "keys.dae"
    with pytest.warns(UserWarning, match=r"keys \['extras'\] have no COLLADA"):
        write_scene(scene, out)
    assert read_scene(out).global_attrs["skins"][0]["name"] == "rig"


def test_write_skin_reference_problems_warn(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_weighted(),),
        nodes=(
            SceneNode(mesh=0, extras={"skin": 5}),
            SceneNode(extras={"skin": 0}),
            SceneNode(name="hip"),
        ),
        scenes=((0, 1, 2),),
        global_attrs={"skins": [{"joints": [2]}, {"joints": [2]}]},
    )
    out = tmp_path / "refs.dae"
    with (
        pytest.warns(UserWarning, match=r"node\(s\) \[0\] name a skin"),
        pytest.warns(UserWarning, match=r"node\(s\) \[1\] name a skin but no mesh"),
        pytest.warns(UserWarning, match=r"skin\(s\) \[1\] deform no written"),
        pytest.warns(UserWarning, match=r"\['joints', 'weights'\].*dropped"),
    ):
        write_scene(scene, out)
    assert "<controller" not in out.read_text()


def test_write_skin_list_must_be_a_list(tmp_path: Path) -> None:
    scene = dataclasses.replace(_skinned_scene(_weighted()), global_attrs={"skins": 3})
    with pytest.raises(CodecError, match="must be a list"):
        write_scene(scene, tmp_path / "x.dae")


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({}, "has no joints and weights"),
        (
            {"joints": np.zeros((3, 4), np.int32), "weights": np.ones((3, 3))},
            "of one shape",
        ),
        (
            {"joints": np.zeros((3, 4)), "weights": np.ones((3, 4))},
            "needs integer joints",
        ),
        (
            {"joints": np.zeros((2, 4), np.int32), "weights": np.ones((2, 4))},
            "one row per vertex",
        ),
    ],
)
def test_write_skinned_mesh_without_usable_weights_is_written_plain(
    tmp_path: Path, attrs: dict, match: str
) -> None:
    poly = make_polydata(
        _TRI, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs=attrs
    )
    out = tmp_path / "plain.dae"
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        write_scene(_skinned_scene(poly), out)
    assert any(re.search(match, str(w.message)) for w in record)
    assert "<controller" not in out.read_text()


def test_write_skin_bad_influences_dropped_with_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "joints": np.array([[0, 1, 0, 0], [7, 0, 0, 0], [0, 0, 0, 0]], np.int32),
            "weights": np.array(
                [[-1.0, 1, 0, 0], [0.5, 0.5, 0, 0], [np.nan, np.inf, 0, 0]]
            ),
        },
    )
    out = tmp_path / "bad.dae"
    with (
        pytest.warns(UserWarning, match=r"3 negative or non-finite weight\(s\)"),
        pytest.warns(UserWarning, match=r"1 influence\(s\) on a joint outside"),
    ):
        write_scene(_skinned_scene(poly), out)
    mesh = read_scene(out).meshes[0]
    assert mesh.vertex_attrs["joints"].tolist() == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert mesh.vertex_attrs["weights"].tolist() == [
        [1, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0, 0, 0, 0],
    ]


# ---------------------------------------------------------------------------
# animations
# ---------------------------------------------------------------------------


def _anim_dae(
    channels: str, *, nested: bool = False, interp=("LINEAR", "LINEAR")
) -> str:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    times = [0.0, 1.0]
    tr = np.array([[0.0, 0, 0], [1, 2, 3]])
    ang = [0.0, 90.0]
    mats = np.tile(np.eye(4).ravel(), (2, 1))
    body = (
        _source("a-t", times, "TIME")
        + _source("a-tr", tr, "X Y Z")
        + _source("a-ang", ang, "ANGLE")
        + _source("a-m", mats, "TRANSFORM")
        + _source("a-i", list(interp), "INTERPOLATION", kind="Name_array")
        + '<sampler id="s-tr"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/>'
        '<input semantic="INTERPOLATION" source="#a-i"/></sampler>'
        '<sampler id="s-ang"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-ang"/></sampler>'
        '<sampler id="s-m"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-m"/></sampler>' + channels
    )
    anim = (
        f'<animation id="anim" name="Walk"><animation id="inner">{body}</animation>'
        "</animation>"
        if nested
        else f'<animation id="anim" name="Walk">{body}</animation>'
    )
    return _dae(
        f"<library_geometries>{geo}</library_geometries>"
        f"<library_animations>{anim}</library_animations>"
        + _scene_with(
            "g0",
            node_extra='<translate sid="location">0 0 0</translate>'
            '<rotate sid="rotationZ">0 0 1 0</rotate>'
            '<node id="n1" sid="child"><matrix sid="transform">'
            "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>",
        )
    )


def test_read_instanced_node_animates_every_copy(tmp_path: Path) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    anim = (
        '<animation id="a">'
        + _source("a-t", [0.0, 1.0], "TIME")
        + _source("a-tr", np.array([[0.0, 0, 0], [1, 2, 3]]), "X Y Z")
        + '<sampler id="s"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/></sampler>'
        '<channel source="#s" target="lib/location"/></animation>'
    )
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_nodes><node id="lib"><translate sid="location">0 0 0</translate>'
        '<instance_geometry url="#g0"/></node></library_nodes>'
        f"<library_animations>{anim}</library_animations>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_node url="#lib"/></node>'
        '<node id="b"><instance_node url="#lib"/></node>'
        "</visual_scene></library_visual_scenes>"
    )
    scene = read_scene(_write(tmp_path, text))
    channels = scene.global_attrs["animations"][0]["channels"]
    assert [c["target"]["node"] for c in channels] == [1, 3]
    assert [c["sampler"] for c in channels] == [0, 0]
    out = tmp_path / "copies.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    # The node goes to <library_nodes> once, both parents instance it from
    # there, and its channel is written once for both.
    text = out.read_text()
    assert text.count('<node id="lib">') == 1
    assert text.count('<instance_node url="#lib"/>') == 2
    assert text.count("<channel ") == 1
    back = read_scene(out)
    assert len(back.nodes) == len(scene.nodes)
    channels = back.global_attrs["animations"][0]["channels"]
    assert [c["target"]["node"] for c in channels] == [1, 3]


def test_read_channels_on_instanced_nodes_capped(tmp_path: Path) -> None:
    """Every channel gets a target per copy of the node it addresses, so the
    targets are capped as the copies are."""
    depth = 10
    lib = "".join(
        f'<node id="n{i}"><instance_node url="#n{i + 1}"/>'
        f'<instance_node url="#n{i + 1}"/></node>'
        for i in range(depth)
    )
    leaf = f'<node id="n{depth}"><translate sid="t">0 0 0</translate></node>'
    body = (
        _source("a-t", [0.0, 1.0], "TIME")
        + _source("a-tr", np.zeros((2, 3)), "X Y Z")
        + '<sampler id="s"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/></sampler>'
        + f'<channel source="#s" target="n{depth}/t"/>'
        * 300
    )
    text = _dae(
        f"<library_nodes>{lib}{leaf}</library_nodes>"
        f'<library_animations><animation id="a">{body}</animation>'
        "</library_animations>"
        '<library_visual_scenes><visual_scene id="S"><instance_node url="#n0"/>'
        "</visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError, match="expand past 262144 targets"):
        read_scene(_write(tmp_path, text))


def test_read_channels_missing_on_instanced_nodes_capped(tmp_path: Path) -> None:
    """A channel that reaches no transform still looks at every copy, so
    the copies it misses on are capped as the targets are."""
    depth = 10
    lib = "".join(
        f'<node id="n{i}"><instance_node url="#n{i + 1}"/>'
        f'<instance_node url="#n{i + 1}"/></node>'
        for i in range(depth)
    )
    leaf = f'<node id="n{depth}"><translate sid="t">0 0 0</translate></node>'
    body = (
        _source("a-t", [0.0, 1.0], "TIME")
        + _source("a-tr", np.zeros((2, 3)), "X Y Z")
        + '<sampler id="s"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/></sampler>'
        + f'<channel source="#s" target="n{depth}/missing"/>'
        * 300
    )
    text = _dae(
        f"<library_nodes>{lib}{leaf}</library_nodes>"
        f'<library_animations><animation id="a">{body}</animation>'
        "</library_animations>"
        '<library_visual_scenes><visual_scene id="S"><instance_node url="#n0"/>'
        "</visual_scene></library_visual_scenes>"
    )
    with (
        warnings.catch_warnings(),
        pytest.raises(CodecError, match="miss their transform on more than 262144"),
    ):
        warnings.simplefilter("ignore")
        read_scene(_write(tmp_path, text))


def test_read_animation_address_takes_the_shallowest_of_many_holders(
    tmp_path: Path,
) -> None:
    """Breadth-first, a later child holding the sid wins over every
    grandchild before it, however many hold it."""
    kids = "".join(
        f'<node id="c{i}"{' sid="k"' if i == 35 else ""}>'
        + ('<translate sid="t">0 0 0</translate>' if i == 35 else "")
        + f'<node id="g{i}" sid="k"><translate sid="t">0 0 0</translate></node>'
        "</node>"
        for i in range(40)
    )
    body = (
        _source("a-t", [0.0, 1.0], "TIME")
        + _source("a-tr", np.zeros((2, 3)), "X Y Z")
        + '<sampler id="s"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/></sampler>'
        '<channel source="#s" target="r/t"/>'
        '<channel source="#s" target="r/k/t"/>'
        '<channel source="#s" target="g3/t"/>'
    )
    text = _dae(
        f'<library_animations><animation id="a">{body}</animation>'
        "</library_animations>"
        f'<library_visual_scenes><visual_scene id="S"><node id="r">{kids}</node>'
        "</visual_scene></library_visual_scenes>"
    )
    scene = read_scene(_write(tmp_path, text))
    targets = [
        c["target"]["node"] for c in scene.global_attrs["animations"][0]["channels"]
    ]
    c35 = next(i for i, n in enumerate(scene.nodes) if n.extras.get("id") == "c35")
    g3 = next(i for i, n in enumerate(scene.nodes) if n.extras.get("id") == "g3")
    assert targets == [c35, c35, g3]


def _copy_of(scene: SceneData) -> dict[int, str]:
    """Map every node under ``A`` or ``B`` to that copy's name."""
    out: dict[int, str] = {}
    for top in scene.scenes[0]:
        stack = [top]
        while stack:
            idx = stack.pop()
            out[idx] = scene.nodes[top].extras["id"]
            stack.extend(scene.nodes[idx].children)
    return out


def test_read_animation_channels(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
        '<channel source="#s-m" target="n0/child/transform"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    anims = scene.global_attrs["animations"]
    assert len(anims) == 1
    anim = anims[0]
    assert anim["name"] == "Walk"
    assert [c["target"] for c in anim["channels"]] == [
        {"node": 0, "path": "translation", "sid": "location", "member": None},
        {"node": 0, "path": "rotation", "sid": "rotationZ", "member": "ANGLE"},
        {"node": 1, "path": "matrix", "sid": "transform", "member": None},
    ]
    assert [c["sampler"] for c in anim["channels"]] == [0, 1, 2]
    s = anim["samplers"]
    np.testing.assert_array_equal(s[0]["times"], [0.0, 1.0])
    np.testing.assert_array_equal(s[0]["values"], [[0, 0, 0], [1, 2, 3]])
    assert s[0]["interpolation"] == "LINEAR"
    np.testing.assert_array_equal(s[1]["values"], [0.0, 90.0])
    assert s[1]["interpolation"] == "LINEAR"
    assert s[2]["values"].shape == (2, 16)


def test_read_animation_samplers_sharing_a_source_own_their_times(
    tmp_path: Path,
) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    s = scene.global_attrs["animations"][0]["samplers"]
    s[0]["times"][0] = 42.0
    assert s[1]["times"][0] == 0.0


def test_read_sampler_shared_by_two_animations_owned_by_each(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    second = (
        '<animation id="again"><channel source="#s-tr" target="n0/location"/>'
        "</animation></library_animations>"
    )
    text = _anim_dae(channels).replace("</library_animations>", second)
    first, other = read_scene(_write(tmp_path, text)).global_attrs["animations"]
    first["samplers"][0]["values"][0, 0] = 99.0
    first["samplers"][0]["times"][0] = 99.0
    assert other["samplers"][0]["values"][0, 0] == 0.0
    assert other["samplers"][0]["times"][0] == 0.0


def test_read_animation_target_sid_found_below_a_child(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-m" target="n0/grandkid/transform"/>'
        '<channel source="#s-m" target="n0/child/grandkid/transform"/>'
    )
    text = _anim_dae(channels).replace(
        '<node id="n1" sid="child"><matrix sid="transform">'
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>",
        '<node id="n1" sid="child"><node id="n2" sid="grandkid">'
        '<matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
        "</node></node>",
    )
    anim = read_scene(_write(tmp_path, text)).global_attrs["animations"][0]
    assert [c["target"]["node"] for c in anim["channels"]] == [2, 2]
    assert [c["target"]["sid"] for c in anim["channels"]] == ["transform"] * 2


def test_read_animation_nested_and_bezier_tangents(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = _anim_dae(channels, nested=True, interp=("BEZIER", "BEZIER")).replace(
        '<input semantic="INTERPOLATION" source="#a-i"/>',
        '<input semantic="INTERPOLATION" source="#a-i"/>'
        '<input semantic="IN_TANGENT" source="#a-tr"/>'
        '<input semantic="OUT_TANGENT" source="#a-tr"/>',
    )
    anim = read_scene(_write(tmp_path, text)).global_attrs["animations"][0]
    assert anim["name"] == "Walk"
    s = anim["samplers"][0]
    assert s["interpolation"] == "BEZIER"
    assert s["in_tangents"].shape == (2, 3)
    assert s["out_tangents"].shape == (2, 3)


def test_read_interpolation_in_lower_case_upper_cased(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = _anim_dae(channels, interp=("bezier", "bezier"))
    scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["animations"][0]["samplers"][0]["interpolation"] == (
        "BEZIER"
    )
    # What the reader hands out, the writer takes.
    write_scene(scene, tmp_path / "again.dae")


def test_read_unknown_interpolation_read_as_linear(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = _anim_dae(channels, interp=("SMOOTH", "SMOOTH"))
    with pytest.warns(UserWarning, match="'SMOOTH'.*read as LINEAR"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["animations"][0]["samplers"][0]["interpolation"] == (
        "LINEAR"
    )
    write_scene(scene, tmp_path / "again.dae")


def test_collada_channels_not_written_into_gltf(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    out = tmp_path / "a.gltf"
    with pytest.warns(UserWarning, match="not written.*'member', 'sid'"):
        polyxios.write_scene(scene, out)
    assert "animations" not in json.loads(out.read_text())


def test_read_animation_unresolved_target_dropped(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="nowhere/location"/>'
        '<channel source="#s-tr" target="n0/nosuchsid"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    )
    with pytest.warns(UserWarning) as rec:
        anim = read_scene(_write(tmp_path, _anim_dae(channels))).global_attrs[
            "animations"
        ][0]
    assert any("nowhere" in str(w.message) for w in rec)
    assert any("nosuchsid" in str(w.message) for w in rec)
    assert len(anim["channels"]) == 1
    assert anim["channels"][0]["sampler"] == 0
    assert len(anim["samplers"]) == 1


def test_read_animation_times_values_mismatch_refused(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = (
        _anim_dae(channels)
        .replace(
            '<float_array id="a-t-array" count="2">0.0 1.0',
            '<float_array id="a-t-array" count="3">0.0 1.0 2.0',
        )
        .replace(
            '<accessor source="#a-t-array" count="2"',
            '<accessor source="#a-t-array" count="3"',
        )
    )
    with pytest.raises(CodecError, match="keys"):
        read_scene(_write(tmp_path, text))


def _with_animations(scene: SceneData, anims: list) -> SceneData:
    return SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        materials=scene.materials,
        textures=scene.textures,
        images=scene.images,
        scenes=scene.scenes,
        name=scene.name,
        global_attrs={"animations": anims},
    )


def _animated_scene(transforms: list, target: dict, values) -> SceneData:
    node = SceneNode(
        name="n",
        mesh=0,
        matrix=np.eye(4),
        extras={"transforms": transforms},
    )
    anim = {
        "name": "a",
        "channels": [{"sampler": 0, "target": {"node": 0, **target}}],
        "samplers": [
            {
                "times": np.array([0.0, 1.0]),
                "values": np.asarray(values, dtype=np.float64),
                "interpolation": "LINEAR",
            }
        ],
    }
    return SceneData(
        meshes=(_surface(),),
        nodes=(node,),
        scenes=((0,),),
        global_attrs={"animations": [anim]},
    )


_TRANSLATE = {"kind": "translate", "sid": "location", "values": [0.0, 0.0, 0.0]}


_ROTATE_X = {"kind": "rotate", "sid": "rotationX", "values": [1.0, 0.0, 0.0, 0.0]}


_ROTATE_Y = {"kind": "rotate", "sid": "rotationY", "values": [0.0, 1.0, 0.0, 0.0]}


def test_write_sidless_rotation_is_not_bound_to_a_rotate(tmp_path: Path) -> None:
    quat = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.7071, 0.7071]]
    scene = _animated_scene([_ROTATE_X, _ROTATE_Y], {"path": "rotation"}, quat)
    out = tmp_path / "quat.dae"
    with pytest.warns(UserWarning, match="no sid and path 'rotation'"):
        write_scene(scene, out)
    assert "<channel" not in out.read_text()


def test_write_sidless_translation_binds_the_single_translate(tmp_path: Path) -> None:
    scene = _animated_scene(
        [_TRANSLATE, _ROTATE_X], {"path": "translation"}, [[0, 0, 0], [1, 2, 3]]
    )
    out = tmp_path / "tr.dae"
    write_scene(scene, out)
    (channel,) = read_scene(out).global_attrs["animations"][0]["channels"]
    assert channel["target"]["sid"] == "location"


def test_write_sidless_translation_with_two_translates_dropped(tmp_path: Path) -> None:
    second = {**_TRANSLATE, "sid": "offset"}
    scene = _animated_scene(
        [_TRANSLATE, second], {"path": "translation"}, [[0, 0, 0], [1, 2, 3]]
    )
    out = tmp_path / "tr2.dae"
    with pytest.warns(UserWarning, match="spells 2 <translate> elements"):
        write_scene(scene, out)
    assert "<channel" not in out.read_text()


def test_write_sidless_translation_of_the_wrong_width_dropped(tmp_path: Path) -> None:
    scene = _animated_scene([_TRANSLATE], {"path": "translation"}, [0.0, 1.0])
    out = tmp_path / "tr1.dae"
    with pytest.warns(UserWarning, match="1 values per key where <translate> holds 3"):
        write_scene(scene, out)
    assert "<channel" not in out.read_text()


def test_write_sidless_cubicspline_channel_dropped(tmp_path: Path) -> None:
    scene = _animated_scene(
        [_TRANSLATE], {"path": "translation"}, [[0, 0, 0], [1, 2, 3]]
    )
    scene.global_attrs["animations"][0]["samplers"][0]["interpolation"] = "CUBICSPLINE"
    out = tmp_path / "cubic.dae"
    with pytest.warns(UserWarning, match="no sid and interpolation 'CUBICSPLINE'"):
        write_scene(scene, out)
    assert "<channel" not in out.read_text()


def _trs(t, q, s) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rot * np.asarray(s, dtype=np.float64)
    out[:3, 3] = t
    return out


_QUARTER_Z = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]


def _gltf_scene(tracks: dict, *, matrix=None) -> SceneData:
    """A one-node scene animated as glTF animates it: no sid, a quaternion rotation."""
    samplers, channels = [], []
    for path, (times, values, interp) in tracks.items():
        channels.append({"sampler": len(samplers), "target": {"node": 0, "path": path}})
        samplers.append(
            {
                "times": np.asarray(times, dtype=np.float64),
                "values": np.asarray(values, dtype=np.float64),
                "interpolation": interp,
            }
        )
    node = SceneNode(mesh=0, matrix=np.eye(4) if matrix is None else matrix)
    return SceneData(
        meshes=(_surface(),),
        nodes=(node,),
        scenes=((0,),),
        global_attrs={"animations": [{"channels": channels, "samplers": samplers}]},
    )


def _baked(scene: SceneData, out: Path) -> dict:
    write_scene(scene, out)
    (anim,) = read_scene(out).global_attrs["animations"]
    (channel,) = anim["channels"]
    assert channel["target"] == {
        "node": 0,
        "path": "matrix",
        "sid": "transform",
        "member": None,
    }
    return anim["samplers"][channel["sampler"]]


def test_write_sidless_trs_channels_bake_into_matrix_keys(tmp_path: Path) -> None:
    scene = _gltf_scene(
        {
            "translation": ([0.0, 1.0], [[0, 0, 0], [1, 2, 3]], "LINEAR"),
            "rotation": ([0.0, 1.0], [[0, 0, 0, 1], _QUARTER_Z], "LINEAR"),
            "scale": ([0.0, 1.0], [[1, 1, 1], [2, 2, 2]], "LINEAR"),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = _baked(scene, tmp_path / "trs.dae")
    assert sampler["interpolation"] == "LINEAR"
    # A quarter turn is split into 15-degree spans.
    np.testing.assert_allclose(sampler["times"], np.linspace(0.0, 1.0, 7))
    keys = sampler["values"].reshape(-1, 4, 4)
    np.testing.assert_allclose(keys[0], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(
        keys[-1], _trs([1, 2, 3], _QUARTER_Z, [2, 2, 2]), atol=1e-12
    )
    half = [0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]
    np.testing.assert_allclose(
        keys[3], _trs([0.5, 1, 1.5], half, [1.5, 1.5, 1.5]), atol=1e-12
    )


def test_write_baked_rotation_holds_the_rest_translation_and_scale(
    tmp_path: Path,
) -> None:
    rest = _trs([5, 0, 0], [0, 0, 0, 1], [-1, 1, 1])
    scene = _gltf_scene(
        {"rotation": ([0.0, 2.0], [[0, 0, 0, 1], _QUARTER_Z], "LINEAR")},
        matrix=rest,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = _baked(scene, tmp_path / "rest.dae")
    keys = sampler["values"].reshape(-1, 4, 4)
    np.testing.assert_allclose(keys[0], rest, atol=1e-12)
    np.testing.assert_allclose(
        keys[-1], _trs([5, 0, 0], _QUARTER_Z, [-1, 1, 1]), atol=1e-12
    )


def test_write_baked_step_channels_stay_step(tmp_path: Path) -> None:
    scene = _gltf_scene(
        {
            "translation": ([0.0, 1.0, 2.0], [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "STEP"),
            "scale": ([0.5], [[3, 3, 3]], "STEP"),
        }
    )
    sampler = _baked(scene, tmp_path / "step.dae")
    assert sampler["interpolation"] == "STEP"
    np.testing.assert_array_equal(sampler["times"], [0.0, 0.5, 1.0, 2.0])
    keys = sampler["values"].reshape(-1, 4, 4)
    np.testing.assert_array_equal(keys[:, 0, 3], [0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(keys[:, 0, 0], [3.0] * 4)


def test_write_baked_step_among_blending_channels_holds_until_its_key(
    tmp_path: Path,
) -> None:
    scene = _gltf_scene(
        {
            "translation": ([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], "STEP"),
            "scale": ([0.0, 2.0], [[1, 1, 1], [3, 3, 3]], "LINEAR"),
        }
    )
    sampler = _baked(scene, tmp_path / "mixed.dae")
    assert sampler["interpolation"] == "LINEAR"
    held = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    np.testing.assert_array_equal(sampler["times"], [0.0, held, 1.0, 2.0])
    assert len(np.unique(sampler["times"].astype(np.float32))) == 4
    keys = sampler["values"].reshape(-1, 4, 4)
    np.testing.assert_array_equal(keys[:, 0, 3], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(keys[:, 1, 1], [1.0, 1.0 + held, 2.0, 3.0])


def test_write_baked_cubicspline_is_split_and_follows_the_spline(
    tmp_path: Path,
) -> None:
    # In-tangent, value, out-tangent per key; flat tangents give smoothstep.
    values = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [4, 0, 0], [0, 0, 0]]
    scene = _gltf_scene({"translation": ([0.0, 1.0], values, "CUBICSPLINE")})
    sampler = _baked(scene, tmp_path / "cubic.dae")
    np.testing.assert_allclose(sampler["times"], [0.0, 0.25, 0.5, 0.75, 1.0])
    u = sampler["times"]
    np.testing.assert_allclose(
        sampler["values"].reshape(-1, 4, 4)[:, 0, 3], 4 * (3 * u**2 - 2 * u**3)
    )


def test_write_baked_cubicspline_rotation_overshoot_is_split_to_the_cap(
    tmp_path: Path,
) -> None:
    ten = [0.0, 0.0, np.sin(np.radians(5.0)), np.cos(np.radians(5.0))]
    # In-tangent, value, out-tangent per key; a steep out-tangent swings the
    # rotation far past both keys inside the span.
    values = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 3, 0], [0, 0, 0, 0], ten, [0, 0, 0, 0]]
    scene = _gltf_scene({"rotation": ([0.0, 1.0], values, "CUBICSPLINE")})
    keys = _baked(scene, tmp_path / "swing.dae")["values"].reshape(-1, 4, 4)[:, :3, :3]
    cos = (np.einsum("kij,kij->k", keys[:-1], keys[1:]) - 1.0) / 2.0
    assert np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max() <= 15.0 + 1e-6


def test_write_bake_drops_a_bad_channel_and_keeps_the_others(tmp_path: Path) -> None:
    scene = _gltf_scene(
        {
            "translation": ([0.0, 1.0], [[0, 0, 0], [1, 2, 3]], "LINEAR"),
            "scale": ([0.0, 1.0], [1.0, 2.0], "LINEAR"),
        }
    )
    scene.global_attrs["animations"][0]["channels"].append(
        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
    )
    with pytest.warns(UserWarning) as record:
        sampler = _baked(scene, tmp_path / "bad.dae")
    messages = " ".join(str(w.message) for w in record)
    assert "channel 1 has no sid and 2 keys of 1 values" in messages
    assert "channel 2 animates the translation of node 0 a second time" in messages
    keys = sampler["values"].reshape(-1, 4, 4)
    np.testing.assert_allclose(keys[-1], _trs([1, 2, 3], [0, 0, 0, 1], [1, 1, 1]))


def test_write_bake_of_a_sheared_node_warns(tmp_path: Path) -> None:
    sheared = np.eye(4)
    sheared[0, 1] = 0.5
    scene = _gltf_scene(
        {"translation": ([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], "LINEAR")},
        matrix=sheared,
    )
    with pytest.warns(UserWarning, match="shear or projection"):
        _baked(scene, tmp_path / "shear.dae")


def test_write_bake_whose_keys_overflow_is_dropped(tmp_path: Path) -> None:
    """Finite key times whose spans overflow when split give no NaN keys."""
    scene = _gltf_scene(
        {
            "rotation": (
                [0.0, 1e308, 1.7e308],
                [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
                "LINEAR",
            )
        }
    )
    out = tmp_path / "overflow.dae"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_scene(scene, out)
    messages = [str(w.message) for w in caught]
    assert all(w.category is UserWarning for w in caught), messages
    assert any("not finite" in m for m in messages), messages
    text = out.read_text()
    assert "<channel " not in text
    assert "NaN" not in text and "INF" not in text
    assert "animations" not in read_scene(out).global_attrs


def test_read_nested_nodes_repeating_an_id_reach_a_transform_once(
    tmp_path: Path,
) -> None:
    """An id two nested nodes repeat reaches the inner one's transform from
    both; the channel animates it once, not twice."""
    text = _anim_dae('<channel source="#s-m" target="n1/transform"/>').replace(
        '<node id="n1" sid="child"><matrix sid="transform">'
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>",
        '<node id="n1" sid="child"><node id="n1" sid="inner"><matrix sid="transform">'
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node></node>",
    )
    anim = read_scene(_write(tmp_path, text)).global_attrs["animations"][0]
    assert [c["target"]["node"] for c in anim["channels"]] == [2]


def test_write_animation_without_channels_warns(tmp_path: Path) -> None:
    scene = _gltf_scene({})
    scene.global_attrs["animations"][0]["name"] = "Idle"
    out = tmp_path / "empty.dae"
    with pytest.warns(UserWarning, match="animation 0 has no channels"):
        write_scene(scene, out)
    assert "<library_animations>" not in out.read_text()


def test_write_bake_dropped_for_overflow_frees_its_address(tmp_path: Path) -> None:
    """A matrix channel turned away by a bake whose keys later overflow is
    written in the bake's place rather than lost with it."""
    scene = _gltf_scene(
        {
            "rotation": (
                [0.0, 1e308, 1.7e308],
                [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
                "LINEAR",
            )
        }
    )
    anim = scene.global_attrs["animations"][0]
    anim["samplers"].append(
        {"times": [0.0, 1.0], "values": np.tile(np.eye(4).ravel(), (2, 1))}
    )
    explicit = {"node": 0, "path": "matrix", "sid": "transform", "member": None}
    anim["channels"].append({"sampler": 1, "target": explicit})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler = _baked(scene, tmp_path / "freed.dae")
    messages = [str(w.message) for w in caught]
    assert any("not finite" in m for m in messages), messages
    assert not any("a second time" in m for m in messages), messages
    np.testing.assert_array_equal(sampler["times"], [0.0, 1.0])


def test_gltf_trs_animation_reaches_collada(tmp_path: Path) -> None:
    scene = _gltf_scene(
        {
            "translation": ([0.0, 1.0], [[0, 0, 0], [1, 2, 3]], "LINEAR"),
            "rotation": ([0.0, 1.0], [[0, 0, 0, 1], _QUARTER_Z], "LINEAR"),
        }
    )
    polyxios.write_scene(scene, tmp_path / "a.glb")
    from_gltf = polyxios.read_scene(tmp_path / "a.glb")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = _baked(from_gltf, tmp_path / "a.dae")
    np.testing.assert_allclose(
        sampler["values"].reshape(-1, 4, 4)[-1],
        _trs([1, 2, 3], _QUARTER_Z, [1, 1, 1]),
        atol=1e-6,
    )


def test_write_animation_roundtrip(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
        '<channel source="#s-m" target="n0/child/transform"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    out = tmp_path / "anim.dae"
    write_scene(scene, out)
    back = read_scene(out)
    a0 = scene.global_attrs["animations"][0]
    a1 = back.global_attrs["animations"][0]
    assert a1["name"] == "Walk"
    assert [c["target"] for c in a1["channels"]] == [
        c["target"] for c in a0["channels"]
    ]
    for s0, s1 in zip(a0["samplers"], a1["samplers"], strict=True):
        np.testing.assert_array_equal(s0["times"], s1["times"])
        np.testing.assert_array_equal(s0["values"], s1["values"])
        assert s0["interpolation"] == s1["interpolation"]
    # The node's own transform elements come back as they were spelled.
    assert [t["sid"] for t in back.nodes[0].extras["transforms"]] == [
        "location",
        "rotationZ",
    ]


def test_write_animation_sources_then_samplers_then_channels(
    tmp_path: Path,
) -> None:
    channels = (
        '<channel source="#s-tr" target="n0/location"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    out = tmp_path / "order.dae"
    write_scene(scene, out)
    anim = ET.parse(out).getroot().find(f"{{{_NS}}}library_animations")[0]
    tags = [child.tag.rsplit("}", 1)[-1] for child in anim]
    assert tags.count("sampler") == 2
    # The schema's sequence: every source, then every sampler, then every channel.
    assert tags == sorted(tags, key=["source", "sampler", "channel"].index)


def test_write_sampler_shared_by_two_channels_written_once(tmp_path: Path) -> None:
    channels = (
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
        '<channel source="#s-ang" target="n0/location.X"/>'
    )
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    assert [c["sampler"] for c in scene.global_attrs["animations"][0]["channels"]] == [
        0,
        0,
    ]
    out = tmp_path / "shared.dae"
    write_scene(scene, out)
    assert out.read_text().count("<sampler ") == 1
    back = read_scene(out).global_attrs["animations"][0]
    assert len(back["samplers"]) == 1
    assert [c["sampler"] for c in back["channels"]] == [0, 0]


_LOC = {"node": 0, "path": "translation", "sid": "location", "member": None}
_MAT = {"node": 1, "path": "matrix", "sid": "transform", "member": None}


@pytest.mark.parametrize(
    ("target", "values", "need"),
    [
        (_LOC, [0.0, 5.0], 3),
        ({**_LOC, "member": "X"}, [[0.0, 0, 0], [1, 2, 3]], 1),
        (
            {"node": 0, "path": "rotation", "sid": "rotationZ", "member": "ANGLE"},
            [[0.0, 0, 0], [1, 2, 3]],
            1,
        ),
        (_MAT, [[0.0, 0, 0], [1, 2, 3]], 16),
        ({**_MAT, "member": "(1)"}, [[0.0, 0, 0], [1, 2, 3]], 4),
        ({**_MAT, "member": "(1)(2)"}, [[0.0, 0, 0], [1, 2, 3]], 1),
    ],
)
def test_write_channel_keys_of_another_width_than_the_target_dropped(
    tmp_path: Path, target: dict, values: list, need: int
) -> None:
    """A whole element takes its own width, a member one value and a matrix
    row four; keys of another width would make a file readers reject."""
    channels = '<channel source="#s-tr" target="n0/location"/>'
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    anim = scene.global_attrs["animations"][0]
    anim["channels"][0]["target"] = dict(target)
    anim["samplers"][0]["values"] = np.array(values)
    out = tmp_path / "width.dae"
    with pytest.warns(UserWarning, match=f"where .* takes {need}; it is dropped"):
        write_scene(scene, out)
    assert "<library_animations>" not in out.read_text()


def test_write_channel_keys_matching_a_matrix_row_kept(tmp_path: Path) -> None:
    channels = '<channel source="#s-m" target="n0/child/transform"/>'
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    anim = scene.global_attrs["animations"][0]
    anim["channels"][0]["target"]["member"] = "(1)"
    sampler = anim["samplers"][0]
    sampler["values"] = np.array([[0.0, 1, 0, 0], [0, 1, 0, 2]])
    out = tmp_path / "row.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out).global_attrs["animations"][0]
    assert back["channels"][0]["target"]["member"] == "(1)"
    np.testing.assert_array_equal(back["samplers"][0]["values"], sampler["values"])


@pytest.mark.parametrize(
    ("target", "source", "need"),
    [
        ("n0/location", "#s-ang", 3),
        ("n0/location.X", "#s-tr", 1),
        ("n0/rotationZ.ANGLE", "#s-tr", 1),
        ("n0/child/transform", "#s-tr", 16),
        ("n0/child/transform(1)", "#s-tr", 4),
        ("n0/child/transform(1)(2)", "#s-tr", 1),
    ],
)
def test_read_channel_keys_of_another_width_than_the_target_dropped(
    tmp_path: Path, target: str, source: str, need: int
) -> None:
    channels = (
        f'<channel source="{source}" target="{target}"/>'
        '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    )
    with pytest.warns(
        UserWarning, match=f"where its target takes {need}; it is dropped"
    ):
        scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    anim = scene.global_attrs["animations"][0]
    assert [c["target"]["sid"] for c in anim["channels"]] == ["rotationZ"]
    assert len(anim["samplers"]) == 1


def test_read_unreached_channels_warned_once(tmp_path: Path) -> None:
    """Material, light and camera channels are common in exported files; one
    warning counts them rather than one per channel."""
    channels = "".join(
        f'<channel source="#s-ang" target="effect{i}/diffuse.R"/>' for i in range(7)
    )
    channels += '<channel source="#s-ang" target="n0/rotationZ.ANGLE"/>'
    with pytest.warns(UserWarning) as rec:
        read_scene(_write(tmp_path, _anim_dae(channels)))
    unreached = [
        str(w.message) for w in rec if "reach no node transform" in str(w.message)
    ]
    assert len(unreached) == 1
    assert "7 animation channel(s)" in unreached[0]
    assert "'effect0/diffuse.R'" in unreached[0] and "and 2 more" in unreached[0]


@pytest.mark.parametrize("interp", ["LIN EAR", "<b>", "lineaire"])
def test_write_interpolation_outside_the_specification_refused(
    tmp_path: Path, interp: str
) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    scene.global_attrs["animations"][0]["samplers"][0]["interpolation"] = interp
    with pytest.raises(CodecError, match=re.escape(repr(interp))):
        write_scene(scene, tmp_path / "bad.dae")


def test_write_interpolation_in_lower_case_upper_cased(tmp_path: Path) -> None:
    """The reader upper-cases an interpolation silently, so the writer does too."""
    channels = '<channel source="#s-tr" target="n0/location"/>'
    scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    scene.global_attrs["animations"][0]["samplers"][0]["interpolation"] = "bezier"
    out = tmp_path / "lower.dae"
    write_scene(scene, out)
    assert "bezier" not in out.read_text()
    back = read_scene(out).global_attrs["animations"][0]["samplers"][0]
    assert back["interpolation"] == "BEZIER"


def test_write_animation_channel_without_target_sid_dropped(tmp_path: Path) -> None:
    scene = _material_scene()
    anims = [
        {
            "name": "A",
            "channels": [
                {
                    "sampler": 0,
                    "target": {"node": 0, "path": "translation", "sid": "location"},
                }
            ],
            "samplers": [
                {
                    "times": np.array([0.0, 1.0]),
                    "values": np.zeros((2, 3)),
                    "interpolation": "LINEAR",
                }
            ],
        }
    ]
    scene = _with_animations(scene, anims)
    with pytest.warns(UserWarning, match="location"):
        write_scene(scene, tmp_path / "a.dae")


def test_write_matrix_path_targets_generated_transform(tmp_path: Path) -> None:
    scene = _material_scene()
    anims = [
        {
            "name": "A",
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "matrix"}}],
            "samplers": [
                {
                    "times": np.array([0.0, 1.0]),
                    "values": np.tile(np.eye(4).ravel(), (2, 1)),
                    "interpolation": "STEP",
                }
            ],
        }
    ]
    scene = _with_animations(scene, anims)
    out = tmp_path / "m.dae"
    write_scene(scene, out)
    back = read_scene(out).global_attrs["animations"][0]
    assert back["channels"][0]["target"] == {
        "node": 1,
        "path": "matrix",
        "sid": "transform",
        "member": None,
    }
    assert back["samplers"][0]["interpolation"] == "STEP"


def test_write_sampler_without_times_or_values_refused(tmp_path: Path) -> None:
    node = SceneNode(
        mesh=0,
        extras={"transforms": [{"kind": "translate", "sid": "t", "values": [0, 0, 0]}]},
    )
    base = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    target = {"node": 0, "path": "translation", "sid": "t"}
    for sampler in ({"values": [[0, 0, 0]]}, {"times": [0.0]}, "not a dict"):
        anims = [
            {"channels": [{"sampler": 0, "target": target}], "samplers": [sampler]}
        ]
        with pytest.raises(CodecError, match="'times' and 'values'"):
            write_scene(_with_animations(base, anims), tmp_path / "s.dae")


def test_write_animation_entries_that_are_not_dicts_refused(tmp_path: Path) -> None:
    base = SceneData(meshes=(_surface(),), nodes=(SceneNode(mesh=0),), scenes=((0,),))
    with pytest.raises(CodecError, match="animation 0 is not a dict"):
        write_scene(_with_animations(base, ["walk"]), tmp_path / "a.dae")
    anims = [{"channels": ["c"], "samplers": []}]
    with pytest.raises(CodecError, match="channel 0 is not a dict"):
        write_scene(_with_animations(base, anims), tmp_path / "a.dae")


def test_write_sampler_matrix_per_key_flattened(tmp_path: Path) -> None:
    scene = _material_scene()
    mats = np.tile(np.eye(4), (2, 1, 1))
    mats[1, :3, 3] = [1, 2, 3]
    anims = [
        {
            "name": "A",
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "matrix"}}],
            "samplers": [
                {
                    "times": np.array([0.0, 1.0]),
                    "values": mats,
                    "in_tangents": mats,
                    "interpolation": "BEZIER",
                }
            ],
        }
    ]
    out = tmp_path / "mats.dae"
    write_scene(_with_animations(scene, anims), out)
    sampler = read_scene(out).global_attrs["animations"][0]["samplers"][0]
    assert sampler["values"].shape == (2, 16)
    np.testing.assert_array_equal(sampler["values"], mats.reshape(2, 16))
    assert sampler["in_tangents"].shape == (2, 16)


def test_write_channel_target_that_is_not_a_dict_dropped(tmp_path: Path) -> None:
    scene = _material_scene()
    anims = [
        {
            "channels": [{"sampler": 0, "target": "node0/transform"}],
            "samplers": [{"times": np.array([0.0]), "values": np.zeros((1, 3))}],
        }
    ]
    with pytest.warns(UserWarning, match="channel 0 names a node or sampler"):
        write_scene(_with_animations(scene, anims), tmp_path / "t.dae")


def test_write_channel_member_must_be_a_name_or_indices(tmp_path: Path) -> None:
    def scene_with(member) -> SceneData:
        return SceneData(
            meshes=(),
            nodes=(SceneNode(),),
            scenes=((0,),),
            global_attrs={
                "animations": [
                    {
                        "channels": [
                            {
                                "sampler": 0,
                                "target": {
                                    "node": 0,
                                    "sid": "transform",
                                    "member": member,
                                },
                            }
                        ],
                        "samplers": [{"times": [0.0, 1.0], "values": [0.0, 1.0]}],
                    }
                ]
            },
        )

    for member in ('X"/><evil a="', 3, "(a)"):
        out = tmp_path / "member.dae"
        with pytest.warns(UserWarning, match="neither a name nor"):
            write_scene(scene_with(member), out)
        assert not any(e.tag.endswith("evil") for e in ET.parse(out).iter())
        assert "<channel" not in out.read_text()
    for member, target in (("X", "transform.X"), ("(0)(3)", "transform(0)(3)")):
        write_scene(scene_with(member), out)
        assert f'target="node0/{target}"' in out.read_text()
        back = read_scene(out)
        assert (
            back.global_attrs["animations"][0]["channels"][0]["target"]["member"]
            == member
        )


def test_read_animation_tangent_count_mismatch_refused(tmp_path: Path) -> None:
    channels = '<channel source="#s-tr" target="n0/location"/>'
    text = _anim_dae(channels, interp=("BEZIER", "BEZIER")).replace(
        '<input semantic="INTERPOLATION" source="#a-i"/>',
        '<input semantic="INTERPOLATION" source="#a-i"/>'
        '<input semantic="IN_TANGENT" source="#a-bad"/>',
        1,
    )
    text = text.replace(
        '<sampler id="s-tr">',
        _source("a-bad", np.zeros((3, 2)), "X Y") + '<sampler id="s-tr">',
    )
    with pytest.raises(CodecError, match="2 keys but 3 in_tangent"):
        read_scene(_write(tmp_path, text))


def test_write_sampler_tangent_count_mismatch_refused(tmp_path: Path) -> None:
    sampler = {
        "times": np.array([0.0, 1.0]),
        "values": np.tile(np.eye(4).ravel(), (2, 1)),
        "interpolation": "BEZIER",
        "out_tangents": np.zeros((1, 2)),
    }
    anims = [
        {
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "matrix"}}],
            "samplers": [sampler],
        }
    ]
    scene = _with_animations(_material_scene(), anims)
    with pytest.raises(CodecError, match="2 times but 1 out_tangent"):
        write_scene(scene, tmp_path / "t.dae")


def test_write_sampler_tangents_not_numbers_refused(tmp_path: Path) -> None:
    sampler = {
        "times": np.array([0.0, 1.0]),
        "values": np.tile(np.eye(4).ravel(), (2, 1)),
        "interpolation": "BEZIER",
        "in_tangents": [["a", "b"], [1.0, 2.0]],
    }
    anims = [
        {
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "matrix"}}],
            "samplers": [sampler],
        }
    ]
    scene = _with_animations(_material_scene(), anims)
    with pytest.raises(CodecError, match="in_tangents that are not numbers"):
        write_scene(scene, tmp_path / "t.dae")


def test_write_baked_step_rotation_among_blending_channels_adds_a_key_per_step(
    tmp_path: Path,
) -> None:
    """A rotation that steps turns at its key alone: splitting the span before
    it would pile keys no float32 tells apart against the step."""
    half_turn = [0.0, 0.0, 1.0, 0.0]
    scene = _gltf_scene(
        {
            "rotation": (
                [0.0, 1.0, 2.0],
                [[0, 0, 0, 1], half_turn, [0, 0, 0, 1]],
                "STEP",
            ),
            "translation": ([0.0, 2.0], [[0, 0, 0], [2, 0, 0]], "LINEAR"),
        }
    )
    sampler = _baked(scene, tmp_path / "step_rotation.dae")
    times = sampler["times"]
    held = [float(np.nextafter(np.float32(t), np.float32(0.0))) for t in (1.0, 2.0)]
    np.testing.assert_array_equal(times, [0.0, held[0], 1.0, held[1], 2.0])
    assert (np.diff(times.astype(np.float32)) > 0).all()
    keys = sampler["values"].reshape(-1, 4, 4)
    np.testing.assert_allclose(keys[:, 0, 0], [1.0, 1.0, -1.0, -1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(keys[:, 0, 3], times)


def test_write_baked_step_beside_cubicspline_keeps_keys_float32_apart(
    tmp_path: Path,
) -> None:
    scene = _gltf_scene(
        {
            "translation": ([0.0, 1.0, 2.0], [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "STEP"),
            "scale": ([0.0, 2.0], np.ones((6, 3)), "CUBICSPLINE"),
        }
    )
    times = _baked(scene, tmp_path / "step_cubic.dae")["times"]
    held = [float(np.nextafter(np.float32(t), np.float32(0.0))) for t in (1.0, 2.0)]
    np.testing.assert_array_equal(
        times, [0.0, 0.25, 0.5, 0.75, held[0], 1.0, 1.25, 1.5, 1.75, held[1], 2.0]
    )
    assert (np.diff(times.astype(np.float32)) > 0).all()


def _translated_scene(sampler: dict, **target) -> SceneData:
    node = SceneNode(
        mesh=0,
        extras={"transforms": [{"kind": "translate", "sid": "t", "values": [0, 0, 0]}]},
    )
    base = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    channel = {
        "sampler": 0,
        "target": {"node": 0, "path": "translation", "sid": "t", **target},
    }
    return _with_animations(base, [{"channels": [channel], "samplers": [sampler]}])


@pytest.mark.parametrize(
    ("sampler", "reason"),
    [
        ({"times": [0.0, np.nan], "values": [[0, 0, 0]] * 2}, "not finite"),
        ({"times": [0.0, 1.0], "values": [[0, 0, 0], [0, np.inf, 0]]}, "not finite"),
        ({"times": [0.0, 1.0, 0.5], "values": [[0, 0, 0]] * 3}, "do not increase"),
        ({"times": [0.0, 0.0], "values": [[0, 0, 0]] * 2}, "do not increase"),
        ({"times": [], "values": []}, "without keys"),
        (
            {
                "times": [0.0, 1.0],
                "values": [[0, 0, 0]] * 2,
                "interpolation": "BEZIER",
                "in_tangents": [[0.0] * 6, [np.nan] * 6],
            },
            "in_tangents that are not finite",
        ),
    ],
)
def test_write_channel_with_unusable_keys_dropped(
    tmp_path: Path, sampler: dict, reason: str
) -> None:
    """The keys a baked channel is dropped for drop one with a sid too."""
    out = tmp_path / "keys.dae"
    with pytest.warns(UserWarning, match=f"channel 0 has .*{reason}; it is dropped"):
        write_scene(_translated_scene(sampler), out)
    text = out.read_text()
    assert "<library_animations>" not in text
    assert "NaN" not in text and "INF" not in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "animations" not in read_scene(out).global_attrs


def test_write_second_channel_on_a_target_dropped(tmp_path: Path) -> None:
    scene = _translated_scene({"times": [0.0, 1.0], "values": [[0, 0, 0], [1, 2, 3]]})
    anim = scene.global_attrs["animations"][0]
    anim["samplers"].append({"times": [0.0, 2.0], "values": [[0, 0, 0], [9, 9, 9]]})
    anim["samplers"].append({"times": [0.0, 2.0], "values": [0.0, 9.0]})
    anim["channels"].append(
        {"sampler": 1, "target": {"node": 0, "path": "translation"}}
    )
    anim["channels"].append(
        {
            "sampler": 2,
            "target": {"node": 0, "path": "translation", "sid": "t", "member": "X"},
        }
    )
    out = tmp_path / "twice.dae"
    with pytest.warns(UserWarning, match=r"channel 1 animates 'node0/t' a second time"):
        write_scene(scene, out)
    back = read_scene(out).global_attrs["animations"][0]
    assert [c["target"]["member"] for c in back["channels"]] == [None, "X"]
    np.testing.assert_array_equal(back["samplers"][0]["times"], [0.0, 1.0])


def test_write_baked_and_matrix_channels_on_one_node_keep_the_first(
    tmp_path: Path,
) -> None:
    scene = _gltf_scene({"translation": ([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], "LINEAR")})
    anim = scene.global_attrs["animations"][0]
    anim["samplers"].append(
        {"times": [0.0, 1.0], "values": np.tile(np.eye(4).ravel(), (2, 1))}
    )
    explicit = {"node": 0, "path": "matrix", "sid": "transform"}
    anim["channels"].append({"sampler": 1, "target": explicit})
    with pytest.warns(
        UserWarning, match="channel 1 animates 'node0/transform' a second"
    ):
        sampler = _baked(scene, tmp_path / "baked_first.dae")
    np.testing.assert_array_equal(sampler["values"][:, 3], [0.0, 1.0])

    anim["channels"].reverse()
    with pytest.warns(
        UserWarning, match="channel 1 animates 'node0/transform' a second"
    ):
        sampler = _baked(scene, tmp_path / "matrix_first.dae")
    np.testing.assert_array_equal(sampler["values"][:, 3], [0.0, 0.0])


def test_write_bake_ids_count_the_nodes_baked(tmp_path: Path) -> None:
    scene = _gltf_scene({"translation": ([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], "LINEAR")})
    scene = dataclasses.replace(scene, nodes=(SceneNode(children=(1,)), scene.nodes[0]))
    anim = scene.global_attrs["animations"][0]
    anim["samplers"].append({"times": [1.0, 0.0], "values": [[1, 1, 1], [2, 2, 2]]})
    anim["channels"] = [
        {"sampler": 1, "target": {"node": 0, "path": "scale"}},
        {"sampler": 0, "target": {"node": 1, "path": "translation"}},
    ]
    out = tmp_path / "ids.dae"
    with pytest.warns(UserWarning, match="channel 0 has no sid and key times"):
        write_scene(scene, out)
    assert re.findall(r'<sampler id="([^"]+)"', out.read_text()) == [
        "animation0-bake0-sampler"
    ]


@pytest.mark.parametrize("name", [None, ""])
def test_write_animation_without_a_name_spells_none(tmp_path: Path, name) -> None:
    scene = _translated_scene({"times": [0.0, 1.0], "values": [[0, 0, 0], [1, 2, 3]]})
    scene.global_attrs["animations"][0]["name"] = name
    out = tmp_path / "unnamed.dae"
    write_scene(scene, out)
    assert '<animation id="animation0">' in out.read_text()
    assert read_scene(out).global_attrs["animations"][0]["name"] == "animation0"


def test_read_animation_transform_sid_found_on_a_descendant(tmp_path: Path) -> None:
    """The last sid of an address is looked up as the ones before it are: on
    the node, then breadth-first below it."""
    channels = (
        '<channel source="#s-m" target="n0/transform"/>'
        '<channel source="#s-tr" target="n0/location"/>'
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = read_scene(_write(tmp_path, _anim_dae(channels)))
    targets = [c["target"] for c in scene.global_attrs["animations"][0]["channels"]]
    assert [(t["node"], t["path"], t["sid"]) for t in targets] == [
        (1, "matrix", "transform"),
        (0, "translation", "location"),
    ]
    out = tmp_path / "below.dae"
    write_scene(scene, out)
    assert 'target="n1/transform"' in out.read_text()
    back = read_scene(out).global_attrs["animations"][0]
    assert [c["target"] for c in back["channels"]] == targets


def test_read_channel_without_source_names_the_file_once(tmp_path: Path) -> None:
    path = _write(tmp_path, _anim_dae('<channel target="n0/location"/>'))
    with pytest.raises(CodecError) as exc:
        read_scene(path)
    assert str(exc.value).count("m.dae") == 1
    assert "channel 'n0/location' has no url" in str(exc.value)


def test_read_sampler_messages_name_the_file_once(tmp_path: Path) -> None:
    text = _anim_dae('<channel source="#s-tr" target="n0/location"/>').replace(
        '<input semantic="INPUT" source="#a-t"/><input semantic="OUTPUT" source="#a-tr"/>',
        '<input semantic="INPUT" source="#a-t"/><input semantic="OUTPUT" source="#gone"/>',
    )
    with pytest.raises(CodecError) as exc:
        read_scene(_write(tmp_path, text))
    assert str(exc.value).count("m.dae") == 1


def test_write_channel_on_a_copy_alone_warns_it_reaches_every_instance(
    tmp_path: Path,
) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    anim = (
        '<animation id="a">'
        + _source("a-t", [0.0, 1.0], "TIME")
        + _source("a-tr", np.array([[0.0, 0, 0], [1, 2, 3]]), "X Y Z")
        + '<sampler id="s"><input semantic="INPUT" source="#a-t"/>'
        '<input semantic="OUTPUT" source="#a-tr"/></sampler>'
        '<channel source="#s" target="lib/location"/></animation>'
    )
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>"
        '<library_nodes><node id="lib"><translate sid="location">0 0 0</translate>'
        '<instance_geometry url="#g0"/></node></library_nodes>'
        f"<library_animations>{anim}</library_animations>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_node url="#lib"/></node>'
        '<node id="b"><instance_node url="#lib"/></node>'
        "</visual_scene></library_visual_scenes>"
    )
    scene = read_scene(_write(tmp_path, text))
    channels = scene.global_attrs["animations"][0]["channels"]
    assert [c["target"]["node"] for c in channels] == [1, 3]
    del channels[0]
    out = tmp_path / "lone.dae"
    with pytest.warns(
        UserWarning, match="animates node 3, a copy of instanced node 1, alone"
    ):
        write_scene(scene, out)
    back = read_scene(out).global_attrs["animations"][0]["channels"]
    assert [c["target"]["node"] for c in back] == [1, 3]


def test_no_animation_key_when_absent(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _tri_dae()))
    assert "animations" not in scene.global_attrs


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_refuse_not_xml(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="well-formed"):
        read_scene(_write(tmp_path, "<COLLADA"))


@pytest.mark.parametrize("encoding", ["shift_jis", "euc-jp", "gb2312"])
def test_read_multibyte_encoding(tmp_path: Path, encoding: str) -> None:
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(
        f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"),
        asset="<contributor><author>山田</author></contributor>",
    )
    text = text.replace('encoding="utf-8"', f'encoding="{encoding}"')
    p = tmp_path / "m.dae"
    p.write_bytes(text.encode(encoding))
    scene = read_scene(p)
    assert scene.global_attrs["asset"]["author"] == "山田"
    assert len(scene.meshes) == 1


def test_refuse_unknown_encoding(tmp_path: Path) -> None:
    text = _tri_dae().replace('encoding="utf-8"', 'encoding="no-such-codec"')
    with pytest.raises(CodecError, match="no-such-codec"):
        read_scene(_write(tmp_path, text))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_refuse_entity_declaration(tmp_path: Path, encoding: str) -> None:
    text = _tri_dae().replace(
        "\n<COLLADA", '\n<!DOCTYPE COLLADA [<!ENTITY a "aaaa">]>\n<COLLADA'
    )
    text = text.replace('encoding="utf-8"', f'encoding="{encoding}"')
    p = tmp_path / "m.dae"
    p.write_bytes(text.encode(encoding))
    with pytest.raises(CodecError, match="entity"):
        read_scene(p)


@pytest.mark.parametrize("encoding", ["utf-7", "utf-32-le"])
def test_refuse_entity_declaration_hidden_by_transcoding(
    tmp_path: Path, encoding: str
) -> None:
    text = _tri_dae().replace(
        "\n<COLLADA", '\n<!DOCTYPE COLLADA [<!ENTITY a "aaaa">]>\n<COLLADA'
    )
    head, body = text.split("?>", 1)
    head = head.replace('encoding="utf-8"', f'encoding="{encoding}"') + "?>"
    p = tmp_path / "m.dae"
    p.write_bytes(head.encode("ascii") + body.encode(encoding))
    with pytest.raises(CodecError, match="entity"):
        read_scene(p)


def test_refuse_wrong_root(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="<COLLADA>"):
        read_scene(_write(tmp_path, '<?xml version="1.0"?><model/>'))


def test_refuse_unknown_version(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="version"):
        read_scene(
            _write(tmp_path, _tri_dae().replace('version="1.4.1"', 'version="2.0"'))
        )


@pytest.mark.parametrize("version", ["1.40", "1.55", "1.4a"])
def test_refuse_version_that_only_starts_like_a_known_one(
    tmp_path: Path, version: str
) -> None:
    text = _tri_dae().replace('version="1.4.1"', f'version="{version}"')
    with pytest.raises(CodecError, match="version"):
        read_scene(_write(tmp_path, text))


@pytest.mark.parametrize("version", ["1.4", "1.4.0", "1.5.0"])
def test_read_known_versions(tmp_path: Path, version: str) -> None:
    text = _tri_dae().replace('version="1.4.1"', f'version="{version}"')
    assert len(read_scene(_write(tmp_path, text)).meshes) == 1


def test_refuse_index_too_large_for_int64(tmp_path: Path) -> None:
    text = _tri_dae().replace("<p>0 1 2</p>", "<p>0 1 99999999999999999999</p>")
    with pytest.raises(CodecError, match="not integers"):
        read_scene(_write(tmp_path, text))


def test_refuse_int_array_value_too_large_for_int64(tmp_path: Path) -> None:
    text = (
        _tri_dae()
        .replace("<float_array", "<int_array")
        .replace("</float_array>", "</int_array>")
        .replace(">0.0 0.0 0.0 1.0", ">0 0 0 99999999999999999999")
    )
    assert "99999999999999999999" in text
    with pytest.raises(CodecError, match="not integers"):
        read_scene(_write(tmp_path, text))


def test_refuse_float_array_count_mismatch(tmp_path: Path) -> None:
    text = _tri_dae().replace('count="9"', 'count="12"')
    with pytest.raises(CodecError, match="12"):
        read_scene(_write(tmp_path, text))


def test_refuse_accessor_past_array(tmp_path: Path) -> None:
    text = _tri_dae().replace('count="3" stride="3"', 'count="4" stride="3"')
    with pytest.raises(CodecError, match="accessor"):
        read_scene(_write(tmp_path, text))


def test_refuse_index_past_source(tmp_path: Path) -> None:
    text = _tri_dae().replace("<p>0 1 2</p>", "<p>0 1 3</p>")
    with pytest.raises(CodecError, match="3"):
        read_scene(_write(tmp_path, text))


def test_refuse_p_length_mismatch(tmp_path: Path) -> None:
    text = _tri_dae().replace("<p>0 1 2</p>", "<p>0 1 2 0</p>")
    with pytest.raises(CodecError, match="triangles"):
        read_scene(_write(tmp_path, text))


def test_refuse_vcount_mismatch(tmp_path: Path) -> None:
    prim = (
        '<polylist count="2"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        "<vcount>3</vcount><p>0 1 2</p></polylist>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="vcount"):
        read_scene(_write(tmp_path, text))


def test_refuse_accessor_stride_past_its_array_before_allocating(
    tmp_path: Path,
) -> None:
    src = (
        '<source id="g0-pos"><float_array id="g0-pos-array" count="3">0 0 0'
        '</float_array><technique_common><accessor source="#g0-pos-array" '
        'count="1" stride="400000000"/></technique_common></source>'
    )
    text = _dae(
        '<library_geometries><geometry id="g0"><mesh>'
        + src
        + '<vertices id="g0-vtx"><input semantic="POSITION" source="#g0-pos"/>'
        "</vertices></mesh></geometry></library_geometries>"
    )
    path = _write(tmp_path, text)
    tracemalloc.start()
    try:
        with pytest.raises(CodecError, match="reads 400000000 values"):
            read_scene(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64 << 20


def test_refuse_polylist_vcount_whose_sum_wraps(tmp_path: Path) -> None:
    big = " ".join([str(1 << 62)] * 4)
    prim = (
        '<polylist count="4"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        f"<vcount>{big}</vcount><p></p></polylist>"
    )
    geo = _geometry("g0", _TRI, prim)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.raises(CodecError, match="more than its <p> of 0 indices"):
        read_scene(_write(tmp_path, text))


def test_refuse_dangling_url(tmp_path: Path) -> None:
    text = _tri_dae().replace('url="#g0"', 'url="#g9"')
    with pytest.raises(CodecError, match="g9"):
        read_scene(_write(tmp_path, text))


def test_refuse_bad_matrix(tmp_path: Path) -> None:
    text = _dae(
        "<library_geometries>"
        + _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
        + "</library_geometries>"
        + _scene_with("g0", node_extra="<matrix>1 2 3</matrix>")
    )
    with pytest.raises(CodecError, match="16"):
        read_scene(_write(tmp_path, text))


def test_read_normal_sets_kept_apart(tmp_path: Path) -> None:
    sources = _source("g0-n0", np.tile([0.0, 0, 1], (3, 1)), "X Y Z") + _source(
        "g0-n1", np.tile([1.0, 0, 0], (3, 1)), "X Y Z"
    )
    inputs = (
        '<input semantic="NORMAL" source="#g0-n0" offset="1" set="0"/>'
        '<input semantic="NORMAL" source="#g0-n1" offset="2" set="1"/>'
    )
    prim = (
        '<triangles count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        f"{inputs}<p>0 0 0 1 1 1 2 2 2</p></triangles>"
    )
    geo = _geometry("g0", _TRI, prim, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(
        mesh.vertex_attrs["normals"], np.tile([0, 0, 1], (3, 1))
    )
    np.testing.assert_array_equal(
        mesh.vertex_attrs["normals_1"], np.tile([1, 0, 0], (3, 1))
    )


def test_read_attribute_given_twice_in_a_block_keeps_the_first(
    tmp_path: Path,
) -> None:
    sources = _source("g0-t0", np.zeros((3, 2)), "S T") + _source(
        "g0-t1", np.ones((3, 2)), "S T"
    )
    inputs = (
        '<input semantic="TEXCOORD" source="#g0-t0" offset="1"/>'
        '<input semantic="TEXCOORD" source="#g0-t1" offset="2" set="0"/>'
    )
    prim = (
        '<triangles count="1"><input semantic="VERTEX" source="#g0-vtx" offset="0"/>'
        f"{inputs}<p>0 0 0 1 1 1 2 2 2</p></triangles>"
    )
    geo = _geometry("g0", _TRI, prim, sources=sources)
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    with pytest.warns(UserWarning, match="gives texcoords twice"):
        mesh = read_scene(_write(tmp_path, text)).meshes[0]
    np.testing.assert_array_equal(mesh.vertex_attrs["texcoords"], np.zeros((3, 2)))


def test_refuse_not_numbers(tmp_path: Path) -> None:
    text = _tri_dae().replace("0.0 0.0 0.0 1.0", "0.0 x 0.0 1.0")
    with pytest.raises(CodecError, match="not numbers"):
        read_scene(_write(tmp_path, text))


def test_read_unit_with_decimal_comma_warns(tmp_path: Path) -> None:
    text = _tri_dae().replace(
        "<asset></asset>", '<asset><unit name="centimeter" meter="0,01"/></asset>'
    )
    with pytest.warns(UserWarning, match="decimal comma"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["asset"]["unit"] == {"name": "centimeter", "meter": 0.01}


def test_write_mixed_surface_keeps_element_order(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[1, 2, 4]])),
        ],
    )
    out = tmp_path / "mixed.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    assert out.read_text().count("<polylist") == 1
    assert "<triangles" not in out.read_text()


@pytest.mark.parametrize("meter", ["0", "-2", "nan", "inf", "abc"])
def test_read_unit_meter_not_finite_positive_read_as_one(
    tmp_path: Path, meter: str
) -> None:
    text = _tri_dae().replace(
        "<asset></asset>", f'<asset><unit name="cm" meter="{meter}"/></asset>'
    )
    with pytest.warns(UserWarning, match="finite positive"):
        scene = read_scene(_write(tmp_path, text))
    assert scene.global_attrs["asset"]["unit"] == {"name": "cm", "meter": 1.0}
    # What the reader hands out, the writer takes.
    write_scene(scene, tmp_path / "again.dae")


def test_read_unknown_up_axis_dropped_with_a_warning(tmp_path: Path) -> None:
    text = _tri_dae().replace(
        "<asset></asset>",
        '<asset><up_axis>W_UP</up_axis><unit name="cm" meter="0.01"/></asset>',
    )
    with pytest.warns(UserWarning, match="W_UP"):
        scene = read_scene(_write(tmp_path, text))
    assert "up_axis" not in scene.global_attrs["asset"]
    assert scene.global_attrs["asset"]["unit"] == {"name": "cm", "meter": 0.01}
    assert len(scene.meshes[0].vertices) == 3


def test_lazy_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_write(tmp_path, _tri_dae()), lazy=True)


def test_read_flattens_with_warning(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="read_scene"):
        poly = read(_write(tmp_path, _tri_dae()))
    np.testing.assert_array_equal(poly.vertices, _TRI)


def test_read_from_buffer(tmp_path: Path) -> None:
    scene = read_scene(io.BytesIO(_tri_dae().encode()))
    assert len(scene.meshes) == 1


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def _material_scene() -> SceneData:
    mesh = _surface()
    mesh = make_polydata(
        mesh.vertices,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={
            "normals": np.tile([[0.0, 0, 1]], (5, 1)),
            "texcoords": np.zeros((5, 2)),
            "colors": np.full((5, 3), 0.5),
        },
        element_attrs={"material": np.array([0, 1], dtype=np.int32)},
    )
    child = SceneNode(name="Child", mesh=0, matrix=np.diag([2.0, 2.0, 2.0, 1.0]))
    root_m = np.eye(4)
    root_m[:3, 3] = [1, 2, 3]
    root = SceneNode(name="Root", children=(1,), matrix=root_m)
    return SceneData(
        meshes=(mesh,),
        nodes=(root, child),
        materials=(
            SceneMaterial(
                name="Red",
                base_color=(1.0, 1.0, 1.0, 0.5),
                emissive=(0.1, 0.1, 0.1),
                alpha_mode="BLEND",
                double_sided=True,
                base_color_texture=0,
            ),
            SceneMaterial(
                name="Green", base_color=(0.0, 1.0, 0.0, 1.0), normal_texture=1
            ),
        ),
        textures=(SceneTexture(image=0, wrap_s=33071), SceneTexture(image=1)),
        images=(
            SceneImage(uri="tex.png", media_type="image/png", name="tex"),
            SceneImage(data=b"\x89PNGxx", media_type="image/png"),
        ),
        scenes=((0,),),
        name="Main",
        global_attrs={
            "asset": {"up_axis": "Z_UP", "unit": {"name": "cm", "meter": 0.01}}
        },
    )


def test_write_scene_roundtrip(tmp_path: Path) -> None:
    scene = _material_scene()
    out = tmp_path / "out.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert back.name == "Main"
    assert back.scenes == ((0,),)
    assert len(back.nodes) == 2
    np.testing.assert_array_equal(back.nodes[0].matrix, scene.nodes[0].matrix)
    np.testing.assert_array_equal(back.nodes[1].matrix, scene.nodes[1].matrix)
    assert back.nodes[0].children == (1,)
    assert back.nodes[1].mesh == 0
    mesh = back.meshes[0]
    np.testing.assert_array_equal(mesh.vertices, scene.meshes[0].vertices)
    np.testing.assert_array_equal(mesh.connectivity, scene.meshes[0].connectivity)
    np.testing.assert_array_equal(mesh.element_types, scene.meshes[0].element_types)
    for key in ("normals", "texcoords", "colors"):
        np.testing.assert_array_equal(
            mesh.vertex_attrs[key], scene.meshes[0].vertex_attrs[key]
        )
    assert mesh.element_attrs["material"].tolist() == [0, 1]
    red, green = back.materials
    assert red.name == "Red" and red.base_color == (1.0, 1.0, 1.0, 0.5)
    assert red.alpha_mode == "BLEND" and red.double_sided
    assert red.emissive == pytest.approx((0.1, 0.1, 0.1))
    assert red.base_color_texture == 0 and green.normal_texture == 1
    assert back.textures[0].wrap_s == 33071
    assert back.images[0] == scene.images[0]
    assert back.images[1] == scene.images[1]
    assert back.global_attrs["asset"]["up_axis"] == "Z_UP"
    assert back.global_attrs["asset"]["unit"] == {"name": "cm", "meter": 0.01}


def test_write_is_deterministic_and_buffer_equal(tmp_path: Path) -> None:
    scene = _material_scene()
    out = tmp_path / "a.dae"
    write_scene(scene, out)
    buf = io.BytesIO()
    write_scene(scene, buf)
    assert out.read_bytes() == buf.getvalue()
    write_scene(scene, tmp_path / "b.dae")
    assert out.read_bytes() == (tmp_path / "b.dae").read_bytes()


def test_write_flat_polydata(tmp_path: Path) -> None:
    poly = make_polydata(
        _surface().vertices,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
        element_attrs={"material": np.array([2, -1], dtype=np.int32)},
    )
    out = tmp_path / "flat.dae"
    write(poly, out)
    back = read_scene(out)
    assert len(back.meshes) == 1 and len(back.nodes) == 1
    assert back.meshes[0].element_attrs["material"].tolist() == [0, -1]
    assert back.materials[0].name == "material_2"
    with pytest.warns(UserWarning, match="read_scene"):
        flat = read(out)
    np.testing.assert_array_equal(flat.vertices, poly.vertices)


def test_write_lines_strips_and_points(tmp_path: Path) -> None:
    poly = make_polydata(
        _QUAD,
        [
            ("line", np.array([[0, 1]])),
            ("poly_line", np.array([[0, 1, 2, 3]])),
            ("triangle_strip", np.array([[0, 1, 3, 2]])),
            ("vertex", np.array([[0]])),
        ],
    )
    out = tmp_path / "l.dae"
    with pytest.warns(UserWarning, match="vertex"):
        write(poly, out)
    back = read_scene(out).meshes[0]
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["poly_line"],
        ELEMENT_TYPES["triangle_strip"],
    ]
    assert back.connectivity.tolist() == [0, 1, 0, 1, 2, 3, 0, 1, 3, 2]


def test_write_volume_elements_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [("tetra", np.array([[0, 1, 2, 3]])), ("triangle", np.array([[0, 1, 2]]))],
    )
    out = tmp_path / "v.dae"
    with pytest.warns(UserWarning, match="tetra"):
        write(poly, out)
    assert len(read_scene(out).meshes[0].element_types) == 1


def test_write_nothing_writable_keeps_the_positions(tmp_path: Path) -> None:
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for kind, cells in (("tetra", [[0, 1, 2, 3]]), ("vertex", [[0], [1], [2], [3]])):
        poly = make_polydata(verts, [(kind, np.array(cells))])
        out = tmp_path / f"{kind}.dae"
        with pytest.warns(UserWarning, match=rf"\['{kind}'\].*leaving the positions"):
            write(poly, out)
        back = read_scene(out).meshes[0]
        np.testing.assert_array_equal(back.vertices, verts)
        assert len(back.element_types) == 0


def test_write_other_vertex_attrs_dropped_with_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _TRI, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs={"temp": np.ones(3)}
    )
    with pytest.warns(UserWarning, match="temp"):
        write(poly, tmp_path / "t.dae")


def test_write_unskinned_joints_and_weights_are_reported_dropped(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "joints": np.zeros((3, 4), dtype=np.int32),
            "weights": np.ones((3, 4)),
        },
    )
    with pytest.warns(UserWarning, match=r"\['joints', 'weights'\].*dropped"):
        write(poly, tmp_path / "j.dae")


def test_write_images_without_type_or_uri_round_trip(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\nxx"
    scene = SceneData(
        meshes=(),
        nodes=(),
        images=(
            SceneImage(data=png),
            SceneImage(data=b"\xff\xd8\xffxx"),
            SceneImage(data=b"RIFF....WEBPxx"),
            SceneImage(data=b"zzz"),
            SceneImage(name="empty"),
        ),
    )
    out = tmp_path / "img.dae"
    write_scene(scene, out)
    back = read_scene(out).images
    assert back[0] == SceneImage(data=png, media_type="image/png")
    assert back[1].media_type == "image/jpeg"
    assert back[2].media_type == "image/webp"
    assert back[3] == SceneImage(data=b"zzz")
    assert back[4] == SceneImage(name="empty")


def test_write_out_of_range_active_scene_keeps_the_name(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        active_scene=5,
        name="Main",
    )
    out = tmp_path / "active.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert back.name == "Main"
    assert back.active_scene == 0


def test_write_texcoord_set_that_is_not_a_number_gets_a_free_one(
    tmp_path: Path,
) -> None:
    uv = np.zeros((3, 2))
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "texcoords_uv2": uv + 2,
            "texcoords_1": uv + 1,
            "texcoords": uv,
            "texcoords_0": uv + 3,
        },
    )
    out = tmp_path / "sets.dae"
    with pytest.warns(
        UserWarning, match=r"texcoords_uv2 as set 2.*texcoords_0 as set 3"
    ):
        write(poly, out)
    inputs = re.findall(
        r'semantic="TEXCOORD" source="#([^"]*)"[^>]*set="([^"]*)"', out.read_text()
    )
    assert inputs == [
        ("geometry0-texcoords_2", "2"),
        ("geometry0-texcoords_1", "1"),
        ("geometry0-texcoords", "0"),
        ("geometry0-texcoords_3", "3"),
    ]
    back = read_scene(out).meshes[0].vertex_attrs
    assert sorted(back) == ["texcoords", "texcoords_1", "texcoords_2", "texcoords_3"]
    np.testing.assert_array_equal(back["texcoords_2"], uv + 2)
    np.testing.assert_array_equal(back["texcoords_3"], uv + 3)


def test_write_texture_shared_by_two_slots_declares_one_sampler(tmp_path: Path) -> None:
    scene = _material_scene()
    shared = SceneMaterial(
        name="Glow", base_color_texture=0, emissive=(1.0, 1.0, 1.0), emissive_texture=0
    )
    scene = SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        materials=(shared, scene.materials[1]),
        textures=scene.textures,
        images=scene.images,
        scenes=scene.scenes,
    )
    out = tmp_path / "shared.dae"
    write_scene(scene, out)
    text = out.read_text()
    assert text.count('<newparam sid="effect0-surface0">') == 1
    assert text.count('<newparam sid="effect0-sampler0">') == 1
    back = read_scene(out).materials[0]
    assert back.base_color_texture == 0 and back.emissive_texture == 0


def test_write_fully_transparent_material_roundtrips(tmp_path: Path) -> None:
    scene = _material_scene()
    clear = SceneMaterial(
        name="Clear", base_color=(1.0, 0.0, 0.0, 0.0), alpha_mode="BLEND"
    )
    scene = SceneData(
        meshes=scene.meshes,
        nodes=scene.nodes,
        materials=(clear, SceneMaterial(name="Green")),
        scenes=scene.scenes,
    )
    out = tmp_path / "clear.dae"
    write_scene(scene, out)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_scene(out).materials[0]
    assert back.base_color == (1.0, 0.0, 0.0, 0.0)
    assert back.alpha_mode == "BLEND"


def test_write_non_finite_values_in_xml_lexical_form(tmp_path: Path) -> None:
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={
            "normals": np.array([[np.nan, 0, 1], [np.inf, 0, 1], [-np.inf, 0, 1]])
        },
    )
    out = tmp_path / "n.dae"
    write(poly, out)
    assert ">NaN 0.0 1.0 INF 0.0 1.0 -INF 0.0 1.0<" in out.read_text()
    back = read_scene(out).meshes[0].vertex_attrs["normals"]
    assert np.isnan(back[0, 0]) and back[1, 0] == np.inf and back[2, 0] == -np.inf


def test_write_groups_elements_by_block_and_material(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)
    poly = make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[3, 4]])),
            ("triangle", np.array([[1, 2, 4]])),
            ("polygon", np.array([[0, 1, 2, 3, 4]])),
        ],
        element_attrs={"material": np.array([0, 1, 0, 1, 0], dtype=np.int32)},
    )
    out = tmp_path / "grouped.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    # Blocks appear in first-occurrence order of (block, material); within one
    # block the elements keep their order.
    assert back.element_attrs["material"].tolist() == [0, 0, 1, 1, 0]
    assert back.offsets.tolist() == [0, 3, 8, 12, 15, 17]
    assert back.connectivity.tolist() == [
        *[0, 1, 4],
        *[0, 1, 2, 3, 4],
        *[0, 1, 2, 3],
        *[1, 2, 4],
        *[3, 4],
    ]


def test_gather_runs() -> None:
    offs = np.array([0, 3, 7, 7, 9])
    got = _collada._gather(offs[[1, 3, 0]], offs[[2, 4, 1]] - offs[[1, 3, 0]])
    assert got.tolist() == [3, 4, 5, 6, 7, 8, 0, 1, 2]
    assert _collada._gather(np.zeros(0, int), np.zeros(0, int)).tolist() == []


def test_write_material_column_regrouped_by_material(tmp_path: Path) -> None:
    poly = make_polydata(
        _QUAD,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3], [0, 1, 3]]))],
        element_attrs={"material": np.array([0, 1, 0, 1], dtype=np.int32)},
    )
    out = tmp_path / "alt.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    assert back.element_attrs["material"].tolist() == [0, 0, 1, 1]
    assert back.connectivity.tolist() == [0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3]


def _transformed_scene(transforms: list) -> SceneData:
    node = SceneNode(
        name="n",
        mesh=0,
        matrix=np.eye(4),
        extras={"transforms": transforms},
    )
    return SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))


def test_write_transform_sid_with_a_dot_warns_and_is_dropped(tmp_path: Path) -> None:
    rot = {"kind": "rotate", "sid": "rot.Z", "values": [0.0, 0.0, 1.0, 0.0]}
    scene = _transformed_scene([rot])
    out = tmp_path / "dot.dae"
    with pytest.warns(UserWarning, match=r"\(0, 'rot\.Z'\)"):
        write_scene(scene, out)
    assert "rot.Z" not in out.read_text()
    (back,) = read_scene(out).nodes[0].extras["transforms"]
    assert back["sid"] is None


def test_write_transform_sid_with_a_dash_round_trips(tmp_path: Path) -> None:
    rot = {"kind": "rotate", "sid": "rot-Z", "values": [0.0, 0.0, 1.0, 0.0]}
    out = tmp_path / "dash.dae"
    write_scene(_transformed_scene([rot]), out)
    (back,) = read_scene(out).nodes[0].extras["transforms"]
    assert back["sid"] == "rot-Z"


def test_write_spelled_transform_values_written_as_numbers(tmp_path: Path) -> None:
    move = {"kind": "translate", "sid": "t", "values": [True, False, True]}
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 0.0, 1.0]
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0, matrix=matrix, extras={"transforms": [move]}),),
        scenes=((0,),),
    )
    out = tmp_path / "bools.dae"
    write_scene(scene, out)
    assert '<translate sid="t">1.0 0.0 1.0</translate>' in out.read_text()
    np.testing.assert_array_equal(read_scene(out).nodes[0].matrix, matrix)


def test_write_tags_and_globals_dropped_with_a_warning(tmp_path: Path) -> None:
    mesh = dataclasses.replace(
        _surface(),
        vertex_tags={"vgroup": np.array([0, 1], dtype=np.int32)},
        global_attrs={"mesh_name": "m", "units": "mm"},
    )
    scene = SceneData(
        meshes=(mesh,),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={"asset": {"up_axis": "Z_UP"}, "source": "cad"},
    )
    with pytest.warns(UserWarning) as record:
        write_scene(scene, tmp_path / "dropped.dae")
    messages = " ".join(str(w.message) for w in record)
    assert "tag group(s) ['vgroup'] of mesh 0" in messages
    assert "global_attrs ['units'] of mesh 0" in messages
    assert "global_attrs ['source'] of the scene" in messages


def test_write_polydata_asset_is_not_reported_dropped(tmp_path: Path) -> None:
    poly = dataclasses.replace(_surface(), global_attrs={"asset": {"up_axis": "Z_UP"}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "asset.dae")
    assert read_scene(tmp_path / "asset.dae").global_attrs["asset"]["up_axis"] == "Z_UP"


def test_write_interleaved_kinds_read_back_grouped_by_block(tmp_path: Path) -> None:
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    poly = make_polydata(
        verts,
        [
            ("line", np.array([[0, 1]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("line", np.array([[2, 3]])),
        ],
    )
    out = tmp_path / "kinds.dae"
    write(poly, out)
    with pytest.warns(UserWarning, match="read_scene"):
        back = read(out)
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["triangle"],
    ]


def test_write_node_sid_only_when_the_node_has_one(tmp_path: Path) -> None:
    out = tmp_path / "sid.dae"
    write_scene(_material_scene(), out)
    assert all(n.get("sid") is None for n in ET.parse(out).iter(f"{{{_NS}}}node"))
    assert all("sid" not in n.extras for n in read_scene(out).nodes)


def test_write_interleaved_materials_read_back_grouped(tmp_path: Path) -> None:
    poly = make_polydata(
        _QUAD,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]]))],
        element_attrs={"material": np.array([0, 1, 0], dtype=np.int32)},
    )
    out = tmp_path / "grouped.dae"
    write(poly, out)
    back = read_scene(out).meshes[0]
    assert back.element_attrs["material"].tolist() == [0, 0, 1]
    assert back.connectivity.tolist() == [0, 1, 2, 1, 2, 3, 0, 2, 3]


def test_write_nodes_outside_every_scene_are_not_written(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(name="kept", mesh=0), SceneNode(name="orphan", mesh=0)),
        scenes=((0,),),
    )
    out = tmp_path / "orphan.dae"
    with pytest.warns(UserWarning, match=r"1 node\(s\) are in no scene"):
        write_scene(scene, out)
    assert "orphan" not in out.read_text()
    assert [n.name for n in read_scene(out).nodes] == ["kept"]


def test_write_duplicate_node_ids_keep_the_first(tmp_path: Path) -> None:
    nodes = tuple(
        SceneNode(name=f"n{i}", mesh=0, extras={"id": "dup"}) for i in range(2)
    ) + (SceneNode(name="x", mesh=0, extras={"id": "x"}),)
    scene = SceneData(meshes=(_surface(),), nodes=nodes, scenes=((0, 1, 2),))
    out = tmp_path / "dup.dae"
    with pytest.warns(UserWarning, match=r"node\(s\) \[1\] repeat an id"):
        write_scene(scene, out)
    back = read_scene(out)
    assert [n.extras["id"] for n in back.nodes] == ["dup", "node1", "x"]


def test_write_node_id_with_a_trailing_newline_is_not_an_xml_name(
    tmp_path: Path,
) -> None:
    node = SceneNode(mesh=0, extras={"id": "foo\n"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "nl.dae"
    with pytest.warns(UserWarning, match="not an XML name"):
        write_scene(scene, out)
    assert 'id="foo' not in out.read_text(encoding="utf-8")
    assert read_scene(out).nodes[0].extras["id"] == "node0"


@pytest.mark.parametrize("given", ["\u00b2a", "\u0301a", "a\u00b2", "1a"])
def test_write_node_id_outside_xml_name_characters_is_replaced(
    tmp_path: Path, given: str
) -> None:
    node = SceneNode(mesh=0, extras={"id": given})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "name.dae"
    with pytest.warns(UserWarning, match="not an XML name"):
        write_scene(scene, out)
    assert f'"{given}' not in out.read_text(encoding="utf-8")
    assert read_scene(out).nodes[0].extras["id"] == "node0"


def test_write_node_id_starting_with_a_non_ascii_letter_is_kept(tmp_path: Path) -> None:
    node = SceneNode(mesh=0, extras={"id": "élan", "sid": "ré"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "unicode.dae"
    write_scene(scene, out)
    back = read_scene(out).nodes[0]
    assert back.extras["id"] == "élan"
    assert back.extras["sid"] == "ré"


def test_write_names_with_line_breaks_round_trip(tmp_path: Path) -> None:
    mesh = dataclasses.replace(_surface(), global_attrs={"mesh_name": "a\tb"})
    node = SceneNode(name="line\none", mesh=0)
    scene = SceneData(meshes=(mesh,), nodes=(node,), scenes=((0,),), name="x\r\ny")
    out = tmp_path / "names.dae"
    write_scene(scene, out)
    back = read_scene(out)
    assert back.nodes[0].name == "line\none"
    assert back.meshes[0].global_attrs["mesh_name"] == "a\tb"
    assert back.name == "x\r\ny"


def test_write_node_id_shaped_like_a_generated_sub_id_keeps_ids_unique(
    tmp_path: Path,
) -> None:
    node = SceneNode(mesh=0, extras={"id": "geometry0-positions"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "ids.dae"
    write_scene(scene, out)
    ids = [e.get("id") for e in ET.fromstring(out.read_bytes()).iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    back = read_scene(out)
    assert back.nodes[0].extras["id"] == "geometry0-positions"
    assert len(back.meshes[0].vertices) == 5


def test_write_transform_sid_that_is_not_a_name_dropped(tmp_path: Path) -> None:
    m = np.eye(4)
    m[0, 3] = 2.0
    node = SceneNode(
        mesh=0,
        matrix=m,
        extras={
            "transforms": [
                {"kind": "translate", "sid": "my loc", "values": [2.0, 0.0, 0.0]}
            ]
        },
    )
    anims = [
        {
            "name": "A",
            "channels": [
                {
                    "sampler": 0,
                    "target": {"node": 0, "path": "translation", "sid": "my loc"},
                }
            ],
            "samplers": [{"times": np.array([0.0]), "values": np.zeros((1, 3))}],
        }
    ]
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(node,),
        scenes=((0,),),
        global_attrs={"animations": anims},
    )
    out = tmp_path / "sid.dae"
    with pytest.warns(UserWarning) as record:
        write_scene(scene, out)
    messages = " | ".join(str(w.message) for w in record)
    assert "not names a channel target" in messages and "my loc" in messages
    assert "<translate>2.0 0.0 0.0</translate>" in out.read_text()
    back = read_scene(out)
    np.testing.assert_array_equal(back.nodes[0].matrix, m)
    assert "animations" not in back.global_attrs


def test_write_edited_node_does_not_warn_about_an_unused_sid(tmp_path: Path) -> None:
    m = np.eye(4)
    m[0, 3] = 5.0
    node = SceneNode(
        mesh=0,
        matrix=m,
        extras={
            "transforms": [
                {"kind": "translate", "sid": "my loc", "values": [2.0, 0.0, 0.0]}
            ]
        },
    )
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "edited.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    assert '<matrix sid="transform">' in out.read_text()


def test_write_edited_matrix_wins_over_spelled_transforms(tmp_path: Path) -> None:
    scene = read_scene(_write(tmp_path, _anim_dae("")))
    node = scene.nodes[0]
    m = np.eye(4)
    m[0, 3] = 7.0
    edited = SceneNode(
        name=node.name,
        mesh=node.mesh,
        children=node.children,
        matrix=m,
        extras=node.extras,
    )
    scene = SceneData(
        meshes=scene.meshes, nodes=(edited, scene.nodes[1]), scenes=scene.scenes
    )
    out = tmp_path / "e.dae"
    write_scene(scene, out)
    back = read_scene(out)
    np.testing.assert_array_equal(back.nodes[0].matrix, m)


def test_write_unit_that_is_not_a_dict_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="unit must be a dict"):
        write(_surface(), tmp_path / "u.dae", unit="cm")
    with pytest.raises(CodecError, match="meter must be a number"):
        write(_surface(), tmp_path / "u.dae", unit={"name": "cm", "meter": "abc"})
    poly = dataclasses.replace(_surface(), global_attrs={"asset": "Z_UP"})
    with pytest.raises(CodecError, match=r"asset'\] must be a dict"):
        write(poly, tmp_path / "u.dae")


def test_write_material_index_past_materials_refused(tmp_path: Path) -> None:
    scene = _material_scene()
    scene = SceneData(meshes=scene.meshes, nodes=scene.nodes, scenes=scene.scenes)
    with pytest.raises(CodecError, match="material"):
        write_scene(scene, tmp_path / "m.dae")


def test_write_node_cycle_refused(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(children=(1,)), SceneNode(children=(0,), mesh=0)),
        scenes=((0,),),
    )
    with pytest.raises(CodecError, match="cycle"):
        write_scene(scene, tmp_path / "c.dae")


def test_write_bad_name_refused(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),), nodes=(SceneNode(mesh=0, name="a\x00b"),), scenes=((0,),)
    )
    with pytest.raises(CodecError, match="name"):
        write_scene(scene, tmp_path / "n.dae")


def test_write_up_axis_option(tmp_path: Path) -> None:
    out = tmp_path / "z.dae"
    write(_surface(), out, up_axis="Z_UP")
    assert read_scene(out).global_attrs["asset"]["up_axis"] == "Z_UP"
    with pytest.raises(CodecError, match="up_axis"):
        write(_surface(), out, up_axis="Q_UP")


def test_write_default_asset(tmp_path: Path) -> None:
    out = tmp_path / "d.dae"
    write(_surface(), out)
    asset = read_scene(out).global_attrs["asset"]
    assert asset["up_axis"] == "Y_UP"
    assert asset["unit"] == {"name": "meter", "meter": 1.0}
    assert asset["authoring_tool"].startswith("polyxios")


def test_read_asset_author_and_copyright(tmp_path: Path) -> None:
    asset = (
        "<contributor><author>Ada</author><authoring_tool>Maya</authoring_tool>"
        "<copyright>(c) Ada</copyright></contributor>"
    )
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]]))
    text = _dae(f"<library_geometries>{geo}</library_geometries>", asset=asset)
    got = read_scene(_write(tmp_path, text)).global_attrs["asset"]
    assert (got["author"], got["copyright"]) == ("Ada", "(c) Ada")


def test_write_asset_author_and_copyright_round_trip(tmp_path: Path) -> None:
    poly = dataclasses.replace(
        _surface(), global_attrs={"asset": {"author": "Ada", "copyright": "(c) <Ada>"}}
    )
    out = tmp_path / "c.dae"
    write(poly, out)
    asset = read_scene(out).global_attrs["asset"]
    assert (asset["author"], asset["copyright"]) == ("Ada", "(c) <Ada>")
    assert asset["authoring_tool"].startswith("polyxios")


def test_gltf_copyright_survives_a_collada_round_trip(tmp_path: Path) -> None:
    """glTF's own version and generator are replaced silently."""
    glb = tmp_path / "a.glb"
    gltf_write_scene(
        SceneData(
            meshes=(_surface(),),
            nodes=(SceneNode(mesh=0),),
            scenes=((0,),),
            global_attrs={"asset": {"copyright": "(c) Ada"}},
        ),
        glb,
    )
    dae = tmp_path / "b.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(gltf_read_scene(glb), dae)
    gltf_write_scene(read_scene(dae), tmp_path / "c.glb")
    assert (
        gltf_read_scene(tmp_path / "c.glb").global_attrs["asset"]["copyright"]
        == "(c) Ada"
    )


def test_gltf_pbr_material_survives_a_collada_round_trip(tmp_path: Path) -> None:
    pbr = SceneMaterial(
        name="pbr",
        base_color=(0.5, 0.6, 0.7, 1.0),
        metallic=0.2,
        roughness=0.8,
        alpha_mode="MASK",
        alpha_cutoff=0.25,
        base_color_texture=0,
        metallic_roughness_texture=0,
        occlusion_texture=0,
    )
    glb = tmp_path / "a.glb"
    gltf_write_scene(
        SceneData(
            meshes=(
                dataclasses.replace(
                    _surface(),
                    vertex_attrs={"texcoords": np.zeros((5, 2))},
                    element_attrs={"material": np.zeros(2, dtype=np.int32)},
                ),
            ),
            nodes=(SceneNode(mesh=0),),
            materials=(pbr,),
            textures=(SceneTexture(image=0),),
            images=(SceneImage(data=b"\x89PNGxx", media_type="image/png"),),
            scenes=((0,),),
        ),
        glb,
    )
    dae = tmp_path / "b.dae"
    write_scene(gltf_read_scene(glb), dae)
    gltf_write_scene(read_scene(dae), tmp_path / "c.glb")
    back = gltf_read_scene(tmp_path / "c.glb").materials[0]
    assert dataclasses.replace(back, extras={}) == pbr


def test_write_asset_key_without_a_counterpart_warns(tmp_path: Path) -> None:
    poly = dataclasses.replace(
        _surface(), global_attrs={"asset": {"minVersion": "2.0", "copyright": "x"}}
    )
    with pytest.warns(UserWarning, match=r"asset key\(s\) \['minVersion'\]"):
        write(poly, tmp_path / "k.dae")


def test_write_explicit_empty_unit_does_not_fall_back_to_the_asset(
    tmp_path: Path,
) -> None:
    poly = dataclasses.replace(
        _surface(), global_attrs={"asset": {"unit": {"name": "cm", "meter": 0.01}}}
    )
    out = tmp_path / "u.dae"
    write(poly, out, unit={})
    assert read_scene(out).global_attrs["asset"]["unit"] == {
        "name": "meter",
        "meter": 1.0,
    }
    write(poly, out)
    assert read_scene(out).global_attrs["asset"]["unit"] == {
        "name": "cm",
        "meter": 0.01,
    }


def test_write_camera_or_light_instance_warns(tmp_path: Path) -> None:
    scene = _material_scene()
    nodes = list(scene.nodes)
    nodes[0] = dataclasses.replace(nodes[0], extras={"camera": "cam", "light": "sun"})
    scene = dataclasses.replace(scene, nodes=tuple(nodes))
    out = tmp_path / "cam.dae"
    with pytest.warns(UserWarning, match="camera and a light"):
        write_scene(scene, out)
    assert "instance_camera" not in out.read_text()
    assert "camera" not in read_scene(out).nodes[0].extras


def test_registry_and_public_api(tmp_path: Path) -> None:
    assert ".dae" in polyxios.supported_extensions()
    out = tmp_path / "api.dae"
    polyxios.write_scene(_material_scene(), out)
    scene = polyxios.read_scene(out)
    assert len(scene.materials) == 2
    with pytest.warns(UserWarning, match="read_scene"):
        polyxios.read(out)


def test_xml_forbidden_noncharacters() -> None:
    assert _collada._XML_FORBIDDEN.search("a\ufffeb")
    assert _collada._XML_FORBIDDEN.search("\uffff")
    assert not _collada._XML_FORBIDDEN.search("plain \u00e9 name")


def test_module_constants() -> None:
    assert _collada.EXTENSION == ".dae"
    assert _collada.LABEL == "COLLADA"


def test_write_node_with_two_parents_is_written_once(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(
            SceneNode(name="a", children=(2, 3)),
            SceneNode(name="b", children=(2,)),
            SceneNode(name="leaf", mesh=0, extras={"id": "leaf"}),
            SceneNode(name="c"),
        ),
        scenes=((0, 1), (1,)),
    )
    out = tmp_path / "dag.dae"
    write_scene(scene, out)
    ids = [e.get("id") for e in ET.parse(out).iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    text = out.read_text()
    assert text.count("<instance_node") == 4
    lib = ET.fromstring(text).find(f"{{{_NS}}}library_nodes")
    assert sorted(n.get("name") for n in lib) == ["b", "leaf"]
    back = read_scene(out)
    names = [n.name for n in back.nodes]
    a = names.index("a")
    assert [names[c] for c in back.nodes[a].children] == ["leaf", "c"]
    for roots in back.scenes:
        wrapper = back.nodes[roots[-1]]
        assert [names[c] for c in wrapper.children] == ["b"]
        b = back.nodes[wrapper.children[0]]
        assert [names[c] for c in b.children] == ["leaf"]
    assert [n.mesh for n in back.nodes].count(0) == 3


def test_write_textured_material_binds_its_texcoord_set(tmp_path: Path) -> None:
    out = tmp_path / "uv.dae"
    write_scene(_material_scene(), out)
    root = ET.parse(out).getroot()
    bound = {
        im.get("symbol"): [
            (b.get("semantic"), b.get("input_semantic"), b.get("input_set"))
            for b in im
            if b.tag.endswith("bind_vertex_input")
        ]
        for im in root.iter(f"{{{_NS}}}instance_material")
    }
    assert bound == {
        "material0": [("UVMap", "TEXCOORD", "0")],
        "material1": [("UVMap", "TEXCOORD", "0")],
    }
    plain = dataclasses.replace(
        _material_scene(),
        materials=(SceneMaterial(name="Plain"), SceneMaterial(name="Plain2")),
    )
    write_scene(plain, out)
    assert "bind_vertex_input" not in out.read_text()


def test_write_textured_material_binds_the_lowest_written_set(
    tmp_path: Path,
) -> None:
    base = _material_scene()
    attrs = dict(base.meshes[0].vertex_attrs)
    attrs["texcoords_2"] = attrs.pop("texcoords")
    scene = dataclasses.replace(
        base, meshes=(dataclasses.replace(base.meshes[0], vertex_attrs=attrs),)
    )
    out = tmp_path / "uv2.dae"
    write_scene(scene, out)
    sets = {
        b.get("input_set") for b in ET.parse(out).iter(f"{{{_NS}}}bind_vertex_input")
    }
    assert sets == {"2"}
    no_uv = dict(attrs)
    del no_uv["texcoords_2"]
    scene = dataclasses.replace(
        base, meshes=(dataclasses.replace(base.meshes[0], vertex_attrs=no_uv),)
    )
    write_scene(scene, out)
    assert "bind_vertex_input" not in out.read_text()


def test_write_base_colour_beside_its_texture_round_trips(tmp_path: Path) -> None:
    base = _material_scene()
    tinted = dataclasses.replace(base.materials[0], base_color=(1.0, 0.0, 0.0, 0.5))
    constant = dataclasses.replace(
        base.materials[1],
        base_color=(0.0, 1.0, 0.0, 1.0),
        extras={"shading": "constant"},
    )
    scene = dataclasses.replace(base, materials=(tinted, constant))
    out = tmp_path / "tint.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out).materials
    assert back[0].base_color == (1.0, 0.0, 0.0, 0.5)
    assert back[0].base_color_texture == 0
    assert back[1].base_color == (0.0, 1.0, 0.0, 1.0)
    diffuse = ET.parse(out).find(f".//{{{_NS}}}diffuse")
    assert diffuse.find(f"{{{_NS}}}texture") is not None
    assert diffuse.find(f"{{{_NS}}}color") is None


def test_write_constant_material_drops_its_base_colour_texture(
    tmp_path: Path,
) -> None:
    base = _material_scene()
    constant = dataclasses.replace(base.materials[0], extras={"shading": "constant"})
    scene = dataclasses.replace(base, materials=(constant, base.materials[1]))
    with pytest.warns(UserWarning, match=r"constant shading, which has no diffuse"):
        write_scene(scene, tmp_path / "constant.dae")


def test_write_constant_material_drops_the_tint_of_its_texture(
    tmp_path: Path,
) -> None:
    """The tint scales the dropped texture, so it is no flat colour either."""
    base = _material_scene()
    constant = dataclasses.replace(
        base.materials[0],
        base_color=(1.0, 0.0, 0.0, 1.0),
        extras={"shading": "constant"},
    )
    scene = dataclasses.replace(base, materials=(constant, base.materials[1]))
    out = tmp_path / "constant.dae"
    with pytest.warns(UserWarning, match=r"base colour texture and its tint"):
        write_scene(scene, out)
    assert "<base_color>" not in out.read_text()
    back = read_scene(out).materials[0]
    assert back.base_color_texture is None
    assert back.base_color == (1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("shading", "lost"),
    [
        ("constant", "['ambient', 'specular', 'shininess']"),
        ("lambert", "['specular', 'shininess']"),
    ],
)
def test_write_terms_the_shading_lacks_warn(
    tmp_path: Path, shading: str, lost: str
) -> None:
    extras = {
        "shading": shading,
        "ambient": (0.1, 0.1, 0.1, 1.0),
        "specular": (0.5, 0.5, 0.5, 1.0),
        "shininess": 20.0,
    }
    mat = SceneMaterial(name="m", metallic=0.0, extras=extras)
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        materials=(mat,),
    )
    with pytest.warns(UserWarning, match=re.escape(f"has {lost}, which {shading}")):
        write_scene(scene, tmp_path / "terms.dae")


def test_write_unknown_shading_warns(tmp_path: Path) -> None:
    mat = SceneMaterial(name="m", metallic=0.0, extras={"shading": "toon"})
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        materials=(mat,),
    )
    out = tmp_path / "toon.dae"
    with pytest.warns(UserWarning, match="shading 'toon'.*written as phong"):
        write_scene(scene, out)
    assert read_scene(out).materials[0].extras["shading"] == "phong"


def test_write_element_attrs_other_than_material_warn(tmp_path: Path) -> None:
    poly = dataclasses.replace(
        _surface(),
        element_attrs={
            "material": np.array([0, 0], dtype=np.int32),
            "region": np.array([1.0, 2.0]),
        },
    )
    out = tmp_path / "ea.dae"
    with pytest.warns(UserWarning, match=r"element attribute\(s\) \['region'\]"):
        write(poly, out)
    assert "region" not in out.read_text()


@pytest.mark.parametrize("meter", [float("nan"), float("inf"), 0.0, -1.0])
def test_write_unit_meter_must_be_finite_and_positive(
    tmp_path: Path, meter: float
) -> None:
    with pytest.raises(CodecError, match="finite positive"):
        write(_surface(), tmp_path / "u.dae", unit={"name": "m", "meter": meter})


def test_write_integer_colours_scaled_to_unit_floats(tmp_path: Path) -> None:
    for dtype, top in ((np.uint8, 255), (np.uint16, 65535), (np.int32, 255)):
        colors = np.array([[top, 0, 0], [0, top, 0], [0, 0, top]] + [[top] * 3] * 2)
        poly = dataclasses.replace(
            _surface(), vertex_attrs={"colors": colors.astype(dtype)}
        )
        out = tmp_path / "c.dae"
        write(poly, out)
        back = read_scene(out).meshes[0].vertex_attrs["colors"]
        np.testing.assert_array_equal(back, colors / top)


def test_write_signed_colours_past_255_warn(tmp_path: Path) -> None:
    colors = np.full((5, 3), 1000, dtype=np.int16)
    poly = dataclasses.replace(_surface(), vertex_attrs={"colors": colors})
    with pytest.warns(UserWarning, match="outside 0..255"):
        write(poly, tmp_path / "c.dae")


def test_write_node_id_and_sid_not_xml_names_warn(tmp_path: Path) -> None:
    node = SceneNode(mesh=0, extras={"id": "bad id", "sid": "1x"})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    with (
        pytest.warns(UserWarning, match=r"id of node\(s\) \[0\].*generated id"),
        pytest.warns(UserWarning, match=r"sid of node\(s\) \[0\].*without a sid"),
    ):
        write_scene(scene, tmp_path / "ids.dae")
    back = read_scene(tmp_path / "ids.dae").nodes[0].extras
    assert back["id"] != "bad id"
    assert "sid" not in back


def test_write_material_column_of_another_length_refused(tmp_path: Path) -> None:
    poly = _surface()
    bad = dataclasses.replace(
        poly, element_attrs={"material": np.zeros(len(poly.element_types) + 1)}
    )
    with pytest.raises(CodecError, match="material entries for"):
        write(bad, tmp_path / "m.dae")


def _one(mesh: PolyData, **kw) -> SceneData:
    """A scene of ``mesh`` under one root node."""
    return SceneData(meshes=(mesh,), nodes=(SceneNode(mesh=0),), scenes=((0,),), **kw)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("element_types", np.array([300, 5]), "element_types"),
        ("element_types", np.array([5.0, 9.0]), "element_types"),
        ("vertices", np.zeros((5, 2)), "vertices"),
        ("offsets", np.array([0.0, 3, 7]), "offsets"),
    ],
)
def test_write_malformed_mesh_arrays_refused(
    tmp_path: Path, field: str, value, match: str
) -> None:
    mesh = dataclasses.replace(_surface(), **{field: value})
    with pytest.raises(CodecError, match=match):
        write_scene(_one(mesh), tmp_path / "m.dae")


def test_write_uint64_offsets(tmp_path: Path) -> None:
    mesh = _surface()
    mesh = dataclasses.replace(mesh, offsets=mesh.offsets.astype(np.uint64))
    write_scene(_one(mesh), tmp_path / "m.dae")
    back = read_scene(tmp_path / "m.dae").meshes[0]
    np.testing.assert_array_equal(back.connectivity, _surface().connectivity)


def test_write_node_matrix_not_4x4_refused(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),), nodes=(SceneNode(mesh=0, matrix=np.eye(3)),)
    )
    with pytest.raises(CodecError, match="4x4"):
        write_scene(scene, tmp_path / "m.dae")


def test_write_rgb_base_color_is_opaque(tmp_path: Path) -> None:
    mesh = dataclasses.replace(
        _surface(), element_attrs={"material": np.array([0, 0], dtype=np.int32)}
    )
    scene = _one(mesh, materials=(SceneMaterial(base_color=(1.0, 0.0, 0.0)),))
    write_scene(scene, tmp_path / "m.dae")
    mat = read_scene(tmp_path / "m.dae").materials[0]
    assert mat.base_color == (1.0, 0.0, 0.0, 1.0)
    assert mat.alpha_mode == "OPAQUE"


def test_write_non_finite_alpha_spelled_as_xs_float(tmp_path: Path) -> None:
    mesh = dataclasses.replace(
        _surface(), element_attrs={"material": np.array([0, 0], dtype=np.int32)}
    )
    scene = _one(
        mesh,
        materials=(
            SceneMaterial(base_color=(1.0, 1.0, 1.0, np.inf), alpha_mode="BLEND"),
        ),
    )
    write_scene(scene, tmp_path / "m.dae")
    text = (tmp_path / "m.dae").read_text(encoding="utf-8")
    assert "<color>1 1 1 INF</color>" in text


def test_write_complex_vertex_attribute_dropped(tmp_path: Path) -> None:
    mesh = _surface()
    mesh = dataclasses.replace(mesh, vertex_attrs={"normals": np.zeros((5, 3)) + 1j})
    with pytest.warns(UserWarning, match="normals"):
        write_scene(_one(mesh), tmp_path / "m.dae")
    assert "normals" not in read_scene(tmp_path / "m.dae").meshes[0].vertex_attrs


def test_write_carriage_return_in_text_roundtrips(tmp_path: Path) -> None:
    scene = _one(_surface(), global_attrs={"asset": {"author": "a\r\nb"}})
    write_scene(scene, tmp_path / "m.dae")
    assert read_scene(tmp_path / "m.dae").global_attrs["asset"]["author"] == "a\r\nb"


def test_write_small_polygons_read_back_as_triangle_and_quad(tmp_path: Path) -> None:
    mesh = make_polydata(
        _QUAD,
        [("polygon", np.array([0, 1, 2])), ("polygon", np.array([0, 1, 2, 3]))],
    )
    write_scene(_one(mesh), tmp_path / "m.dae")
    back = read_scene(tmp_path / "m.dae").meshes[0]
    assert back.element_types.tolist() == [
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["quad"],
    ]


def test_write_unknown_extras_reported_dropped(tmp_path: Path) -> None:
    mesh = dataclasses.replace(
        _surface(), element_attrs={"material": np.array([0, 0], dtype=np.int32)}
    )
    scene = _one(mesh, materials=(SceneMaterial(extras={"gloss": 1.0}),))
    scene = dataclasses.replace(
        scene, nodes=(SceneNode(mesh=0, extras={"tag": "x", "id": "n"}),)
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        write_scene(scene, tmp_path / "m.dae")
    messages = [str(w.message) for w in record]
    assert any("node extras ['tag']" in m for m in messages)
    assert any("material extras ['gloss']" in m for m in messages)


def test_write_repeated_transform_sid_written_once(tmp_path: Path) -> None:
    move = {"kind": "translate", "values": [1.0, 0.0, 0.0]}
    matrix = np.eye(4)
    matrix[0, 3] = 2.0
    node = SceneNode(
        mesh=0,
        matrix=matrix,
        extras={"transforms": [{**move, "sid": "t"}, {**move, "sid": "t"}]},
    )
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "twice.dae"
    with pytest.warns(UserWarning, match="repeats an earlier one"):
        write_scene(scene, out)
    assert out.read_text().count('sid="t"') == 1
    back = read_scene(out).nodes[0]
    np.testing.assert_array_equal(back.matrix, matrix)
    assert [t["sid"] for t in back.extras["transforms"]] == ["t", None]


def test_write_invalid_transform_sid_warns_once(tmp_path: Path) -> None:
    rot = {"kind": "rotate", "sid": "rot.Z", "values": [0.0, 0.0, 1.0, 0.0]}
    node = SceneNode(mesh=0, extras={"transforms": [rot]})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    with pytest.warns(UserWarning) as record:
        write_scene(scene, tmp_path / "once.dae")
    assert sum("sid" in str(w.message) for w in record) == 1


def test_write_edited_matrix_of_a_far_node_not_reverted(tmp_path: Path) -> None:
    move = {"kind": "translate", "sid": "location", "values": [1e4, 0.0, 0.0]}
    matrix = np.eye(4)
    matrix[0, 3] = 1e4 + 0.05
    node = SceneNode(mesh=0, matrix=matrix, extras={"transforms": [move]})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "far.dae"
    write_scene(scene, out)
    np.testing.assert_array_equal(read_scene(out).nodes[0].matrix, matrix)


def test_write_pbr_terms_round_trip(tmp_path: Path) -> None:
    pbr = SceneMaterial(
        name="pbr",
        metallic=0.25,
        roughness=0.5,
        alpha_mode="MASK",
        alpha_cutoff=0.3,
        occlusion_texture=0,
        metallic_roughness_texture=1,
    )
    scene = dataclasses.replace(_material_scene(), materials=(pbr, SceneMaterial()))
    out = tmp_path / "pbr.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
        back = read_scene(out)
    mat = back.materials[0]
    image = {
        t: back.textures[getattr(mat, t)].image
        for t in ("occlusion_texture", "metallic_roughness_texture")
    }
    assert image == {"occlusion_texture": 0, "metallic_roughness_texture": 1}
    expected_slots = {
        "occlusion_texture": 0,
        "metallic_roughness_texture": 1,
        "extras": {},
    }
    assert dataclasses.replace(mat, **expected_slots) == pbr
    assert back.materials[1].metallic == 1.0
    techniques = [t.get("profile") for t in ET.parse(out).iter(f"{{{_NS}}}technique")]
    assert techniques.count("polyxios") == 2


def test_write_material_at_the_common_profile_defaults_has_no_extra(
    tmp_path: Path,
) -> None:
    plain = SceneMaterial(name="plain", metallic=0.0, base_color=(0.2, 0.4, 0.6, 1.0))
    scene = dataclasses.replace(_material_scene(), materials=(plain, plain))
    out = tmp_path / "plain.dae"
    write_scene(scene, out)
    assert 'profile="polyxios"' not in out.read_text()
    assert read_scene(out).materials[0].base_color == (0.2, 0.4, 0.6, 1.0)


def test_write_opaque_mode_beside_an_alpha_round_trips(tmp_path: Path) -> None:
    """glTF ignores the alpha of an OPAQUE material; the mode is kept."""
    opaque = SceneMaterial(base_color=(1.0, 1.0, 1.0, 0.5), alpha_mode="OPAQUE")
    scene = dataclasses.replace(_material_scene(), materials=(opaque, opaque))
    out = tmp_path / "opaque.dae"
    write_scene(scene, out)
    back = read_scene(out).materials[0]
    assert (back.alpha_mode, back.base_color[3]) == ("OPAQUE", 0.5)


def test_write_pbr_term_not_finite_refused(tmp_path: Path) -> None:
    scene = dataclasses.replace(
        _material_scene(),
        materials=(SceneMaterial(roughness=float("nan")), SceneMaterial()),
    )
    with pytest.raises(CodecError, match="roughness must be a finite number"):
        write_scene(scene, tmp_path / "nan.dae")


def test_write_unknown_alpha_mode_dropped_with_warning(tmp_path: Path) -> None:
    scene = dataclasses.replace(
        _material_scene(),
        materials=(SceneMaterial(alpha_mode="CUTOUT"), SceneMaterial()),
    )
    out = tmp_path / "mode.dae"
    with pytest.warns(UserWarning, match="alpha_mode 'CUTOUT'"):
        write_scene(scene, out)
    assert read_scene(out).materials[0].alpha_mode == "OPAQUE"


@pytest.mark.parametrize(
    ("term", "match"),
    [
        ("<alpha_mode>CUTOUT</alpha_mode>", "alpha_mode 'CUTOUT'"),
        ("<metallic><float>NaN</float></metallic>", "metallic is not finite"),
        (
            "<base_color><color>INF 0 0 1</color></base_color>",
            "base_color is not finite",
        ),
        ("<metallic><float>abc</float></metallic>", "metallic is not numbers"),
        ("<roughness><float>1 2</float></roughness>", "roughness needs 1 number,"),
        (
            "<base_color><color>1 0</color></base_color>",
            "base_color needs 3 or 4 numbers",
        ),
        ("<emissive><color>x 0 0</color></emissive>", "emissive is not numbers"),
        ("<metallic>0.3</metallic>", "metallic holds no <float>"),
        ("<base_color>1 0 0</base_color>", "base_color holds no <color>"),
    ],
)
def test_read_bad_pbr_term_ignored_with_warning(
    tmp_path: Path, term: str, match: str
) -> None:
    base = _material_scene()
    red = dataclasses.replace(base.materials[0], metallic=0.0)
    out = tmp_path / "bad.dae"
    write_scene(dataclasses.replace(base, materials=(red, red)), out)
    assert 'profile="polyxios"' not in out.read_text()
    text = out.read_text().replace(
        "</technique>\n",
        f'<extra><technique profile="polyxios">{term}'
        "</technique></extra></technique>\n",
        1,
    )
    out.write_text(text)
    with pytest.warns(UserWarning, match=match):
        back = read_scene(out).materials[0]
    assert back.alpha_mode == "BLEND"
    assert back.base_color[:3] == (1.0, 1.0, 1.0)


def test_read_pbr_tint_of_an_unresolved_texture_ignored(tmp_path: Path) -> None:
    """A tint scales its texture; without the texture it is not a colour."""
    base = _material_scene()
    lit = dataclasses.replace(
        base.materials[0],
        base_color=(1.0, 0.0, 0.0, 0.5),
        emissive=(0.5, 0.5, 0.5),
        emissive_texture=0,
    )
    out = tmp_path / "lost.dae"
    write_scene(dataclasses.replace(base, materials=(lit, base.materials[1])), out)
    assert "<emissive>" in out.read_text()
    text = re.sub(
        r'(<surface type="2D"><init_from>)[^<]*', r"\1missing", out.read_text()
    )
    out.write_text(text)
    with pytest.warns(UserWarning, match="reaches no image"):
        back = read_scene(out).materials[0]
    assert back.base_color_texture is None
    assert back.emissive_texture is None
    assert back.base_color == (1.0, 1.0, 1.0, 0.5)
    assert back.emissive == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "where",
    [
        "</profile_COMMON>",
        "</effect>",
    ],
)
def test_read_pbr_technique_outside_the_common_technique_ignored(
    tmp_path: Path, where: str
) -> None:
    """Only the common technique's ``<extra>`` is where a write puts the terms."""
    base = _material_scene()
    red = dataclasses.replace(base.materials[0], metallic=0.0)
    out = tmp_path / "elsewhere.dae"
    write_scene(dataclasses.replace(base, materials=(red, red)), out)
    term = (
        '<extra><technique profile="polyxios">'
        "<metallic><float>0.5</float></metallic></technique></extra>"
    )
    if where == "</effect>":
        text = out.read_text().replace(where, term + where, 1)
    else:
        text = out.read_text().replace(
            where,
            f'<extra><technique profile="other">{term}</technique></extra>{where}',
            1,
        )
    out.write_text(text)
    assert read_scene(out).materials[0].metallic == 0.0


def test_write_alpha_cutoff_only_for_mask(tmp_path: Path) -> None:
    """glTF uses ``alpha_cutoff`` under ``MASK`` alone, and so does a write."""
    blend = SceneMaterial(metallic=0.0, alpha_mode="BLEND", base_color=(1, 1, 1, 0.5))
    unused = (
        dataclasses.replace(blend, alpha_cutoff=float("nan")),
        dataclasses.replace(blend, alpha_cutoff=0.3),
    )
    scene = dataclasses.replace(_material_scene(), materials=unused)
    out = tmp_path / "cutoff.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    assert 'profile="polyxios"' not in out.read_text()
    masked = dataclasses.replace(unused[0], alpha_mode="MASK")
    scene = dataclasses.replace(scene, materials=(masked, masked))
    with pytest.raises(CodecError, match="alpha_cutoff must be a finite number"):
        write_scene(scene, out)


@pytest.mark.parametrize(
    "material",
    [
        SceneMaterial(metallic=np.array([0.5])),
        SceneMaterial(metallic=0.0, extras={"shininess": np.array([2.0])}),
    ],
)
def test_write_array_material_scalar_refused(
    tmp_path: Path, material: SceneMaterial
) -> None:
    scene = dataclasses.replace(_material_scene(), materials=(material, material))
    with pytest.raises(CodecError, match="must be a finite number"):
        write_scene(scene, tmp_path / "array.dae")


@pytest.mark.parametrize(
    "material",
    [
        SceneMaterial(metallic="0.5"),
        SceneMaterial(metallic=b"0.5"),
        SceneMaterial(roughness=True),
        SceneMaterial(roughness=np.bool_(False)),
        SceneMaterial(metallic=np.str_("0.5")),
        SceneMaterial(metallic=0.0, extras={"shininess": "2"}),
        SceneMaterial(metallic=10**400),
    ],
)
def test_write_material_scalar_string_or_bool_refused(
    tmp_path: Path, material: SceneMaterial
) -> None:
    """``float`` parses a string and a boolean; neither is a number."""
    scene = dataclasses.replace(_material_scene(), materials=(material, material))
    with pytest.raises(CodecError, match="must be a finite number"):
        write_scene(scene, tmp_path / "typed.dae")


@pytest.mark.parametrize(
    "mode",
    ["", "<alpha_mode>BLEND</alpha_mode>", "<alpha_mode>CUTOUT</alpha_mode>"],
)
def test_read_alpha_cutoff_outside_mask_ignored(tmp_path: Path, mode: str) -> None:
    """glTF reads ``alpha_cutoff`` under ``MASK`` alone, and a write puts it there."""
    base = _material_scene()
    red = dataclasses.replace(base.materials[0], metallic=0.0)
    out = tmp_path / "cutoff.dae"
    write_scene(dataclasses.replace(base, materials=(red, red)), out)
    text = out.read_text().replace(
        "</technique>\n",
        '<extra><technique profile="polyxios">'
        f"<alpha_cutoff><float>0.25</float></alpha_cutoff>{mode}"
        "</technique></extra></technique>\n",
        1,
    )
    out.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = read_scene(out).materials[0]
    assert back.alpha_mode == "BLEND"
    assert back.alpha_cutoff == 0.5
    masked = text.replace(
        f"{mode}</technique>", "<alpha_mode>MASK</alpha_mode></technique>", 1
    )
    out.write_text(masked)
    back = read_scene(out).materials[0]
    assert (back.alpha_mode, back.alpha_cutoff) == ("MASK", 0.25)


def test_read_pbr_extra_of_a_technique_without_shading(tmp_path: Path) -> None:
    """A technique naming no shading still holds the terms its extra spells."""
    base = _material_scene()
    pbr = dataclasses.replace(base.materials[0], metallic=0.5, roughness=0.25)
    out = tmp_path / "bare.dae"
    write_scene(dataclasses.replace(base, materials=(pbr, pbr)), out)
    text = re.sub(r"<phong>.*?</phong>", "", out.read_text(), count=1, flags=re.S)
    out.write_text(text)
    back = read_scene(out).materials[0]
    assert (back.metallic, back.roughness) == (0.5, 0.25)
    assert back.extras["shading"] == "phong"


@pytest.mark.parametrize("key", ["base_color", "emissive"])
def test_read_pbr_tint_without_a_texture_warns(tmp_path: Path, key: str) -> None:
    """A tint beside no texture is not applied, and the drop is reported."""
    base = _material_scene()
    plain = dataclasses.replace(base.materials[1], metallic=0.0)
    out = tmp_path / "bare.dae"
    write_scene(dataclasses.replace(base, materials=(plain, plain)), out)
    assert 'profile="polyxios"' not in out.read_text()
    text = out.read_text().replace(
        "</technique>\n",
        f'<extra><technique profile="polyxios"><{key}><color>0 0 1</color></{key}>'
        "</technique></extra></technique>\n",
        1,
    )
    out.write_text(text)
    with pytest.warns(UserWarning, match=f"{key} tints a texture the effect lacks"):
        back, other = read_scene(out).materials
    assert back == other


def test_write_every_attribute_the_reader_names_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    keys = (
        "normals",
        "normals_1",
        "textangent",
        "texbinormal_2",
        "tangent",
        "binormal",
    )
    attrs = {k: rng.random((3, 3)) for k in keys}
    poly = make_polydata(
        _TRI, [("triangle", np.array([[0, 1, 2]]))], vertex_attrs=attrs
    )
    out = tmp_path / "sets.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(SceneData(meshes=(poly,), nodes=(SceneNode(mesh=0),)), out)
    back = read_scene(out).meshes[0].vertex_attrs
    assert sorted(back) == sorted(keys)
    for k in keys:
        np.testing.assert_array_equal(back[k], attrs[k])


def test_write_normal_set_that_is_not_a_number_gets_a_free_one(
    tmp_path: Path,
) -> None:
    n = np.tile([0.0, 0, 1], (3, 1))
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"normals": n, "normals_smooth": -n},
    )
    out = tmp_path / "n.dae"
    with pytest.warns(UserWarning, match="normals_smooth as set 1"):
        write(poly, out)
    np.testing.assert_array_equal(
        read_scene(out).meshes[0].vertex_attrs["normals_1"], -n
    )


def test_write_image_without_uri_or_data_still_has_init_from(tmp_path: Path) -> None:
    scene = SceneData(meshes=(), nodes=(), images=(SceneImage(name="empty"),))
    out = tmp_path / "img.dae"
    write_scene(scene, out)
    image = next(ET.parse(out).iter(f"{{{_NS}}}image"))
    assert image.find(f"{{{_NS}}}init_from") is not None
    assert read_scene(out).images == (SceneImage(name="empty"),)


def test_read_second_camera_of_a_node_keeps_the_first(tmp_path: Path) -> None:
    text = _dae(
        '<library_visual_scenes><visual_scene id="S"><node id="n0">'
        '<instance_camera url="#cam0"/><instance_camera url="#cam1"/>'
        '<instance_light url="#sun"/></node></visual_scene>'
        "</library_visual_scenes>"
    )
    with pytest.warns(UserWarning, match="second camera, '#cam1'"):
        node = read_scene(_write(tmp_path, text)).nodes[0]
    assert node.extras["camera"] == "cam0"
    assert node.extras["light"] == "sun"


def test_write_set_suffix_the_reader_would_not_spell_warns(tmp_path: Path) -> None:
    uv = np.zeros((3, 2))
    poly = make_polydata(
        _TRI,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"texcoords_0": uv, "texcoords_01": uv + 1, "texcoords_²": uv + 2},
    )
    out = tmp_path / "suffix.dae"
    with pytest.warns(
        UserWarning,
        match=r"texcoords_0 as set 0.*texcoords_01 as set 1.*texcoords_² as set 2",
    ):
        write(poly, out)
    back = read_scene(out).meshes[0].vertex_attrs
    np.testing.assert_array_equal(back["texcoords"], uv)
    np.testing.assert_array_equal(back["texcoords_2"], uv + 2)


def test_read_set_with_leading_zeros_is_the_plain_number(tmp_path: Path) -> None:
    uv = np.full((3, 2), 0.5)
    inputs = (
        '<input semantic="TEXCOORD" source="#g0-uv" offset="0" set="00"/>'
        '<input semantic="TEXCOORD" source="#g0-uv" offset="0" set="01"/>'
    )
    geo = _geometry(
        "g0",
        _TRI,
        _triangles("g0", [[0, 1, 2]], extra_inputs=inputs),
        sources=_source("g0-uv", uv, "S T"),
    )
    text = _dae(f"<library_geometries>{geo}</library_geometries>" + _scene_with("g0"))
    mesh = read_scene(_write(tmp_path, text)).meshes[0]
    assert sorted(mesh.vertex_attrs) == ["texcoords", "texcoords_1"]


def test_read_emission_texture_has_a_white_emissive_factor(tmp_path: Path) -> None:
    technique = (
        '<phong><emission><texture texture="img0" texcoord="UVMap"/></emission></phong>'
    )
    images = '<image id="img0"><init_from>glow.png</init_from></image>'
    mat = read_scene(_write(tmp_path, _effect_dae(technique, images=images))).materials[
        0
    ]
    assert mat.emissive_texture == 0
    assert mat.emissive == (1.0, 1.0, 1.0)


def test_write_emissive_texture_under_a_black_factor_is_dropped(
    tmp_path: Path,
) -> None:
    scene = _material_scene()
    scene = dataclasses.replace(
        scene,
        materials=(SceneMaterial(emissive_texture=0), *scene.materials[1:]),
    )
    out = tmp_path / "off.dae"
    with pytest.warns(UserWarning, match="black emissive factor"):
        write_scene(scene, out)
    assert "effect0-sampler" not in out.read_text().split("</effect>")[0]
    back = read_scene(out).materials[0]
    assert back.emissive_texture is None
    assert back.emissive == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "column", [np.array([0.7, 1.2]), np.array([np.nan, 1.0]), np.array(["a", "b"])]
)
def test_write_material_column_of_non_whole_numbers_refused(
    tmp_path: Path, column: np.ndarray
) -> None:
    scene = _material_scene()
    mesh = dataclasses.replace(scene.meshes[0], element_attrs={"material": column})
    scene = dataclasses.replace(scene, meshes=(mesh,))
    with pytest.raises(CodecError, match="not whole numbers"):
        write_scene(scene, tmp_path / "m.dae")


@pytest.mark.parametrize("column", [np.array([0.5, 1.5]), np.array([np.nan, 0.0])])
def test_write_polydata_material_column_of_non_whole_numbers_refused(
    tmp_path: Path, column: np.ndarray
) -> None:
    poly = dataclasses.replace(_surface(), element_attrs={"material": column})
    with pytest.raises(CodecError, match="not whole numbers"):
        write(poly, tmp_path / "m.dae")


def test_write_material_column_of_whole_floats_accepted(tmp_path: Path) -> None:
    scene = _material_scene()
    mesh = dataclasses.replace(
        scene.meshes[0], element_attrs={"material": np.array([1.0, 0.0])}
    )
    scene = dataclasses.replace(scene, meshes=(mesh,))
    out = tmp_path / "m.dae"
    write_scene(scene, out)
    back = read_scene(out)
    names = [back.materials[m].name for m in back.meshes[0].element_attrs["material"]]
    assert names == ["Green", "Red"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("specular", 5),
        ("ambient", (1.0, 2.0)),
        ("reflective", ("a", "b", "c")),
        ("shininess", "x"),
        ("reflectivity", float("nan")),
        ("index_of_refraction", [1.0, 2.0]),
    ],
)
def test_write_material_extras_of_the_wrong_shape_refused(
    tmp_path: Path, key: str, value
) -> None:
    scene = _material_scene()
    mat = dataclasses.replace(scene.materials[1], extras={key: value})
    scene = dataclasses.replace(scene, materials=(scene.materials[0], mat))
    with pytest.raises(CodecError, match=rf"extras\['{key}'\]"):
        write_scene(scene, tmp_path / "m.dae")


def test_write_node_name_that_is_not_a_string_is_spelled(tmp_path: Path) -> None:
    scene = _material_scene()
    root = dataclasses.replace(scene.nodes[0], name=5)
    scene = dataclasses.replace(scene, nodes=(root, scene.nodes[1]))
    out = tmp_path / "m.dae"
    write_scene(scene, out)
    assert read_scene(out).nodes[0].name == "5"


def test_write_points_only_mesh_keeps_its_vertex_attributes(tmp_path: Path) -> None:
    normals = np.eye(3)
    poly = make_polydata(
        _TRI,
        [("vertex", np.array([[0], [1], [2]]))],
        vertex_attrs={"normals": normals, "texcoords_1": np.zeros((3, 2))},
    )
    out = tmp_path / "pts.dae"
    with pytest.warns(UserWarning) as record:
        write(poly, out)
    messages = " ".join(str(w.message) for w in record)
    assert "leaving the positions" in messages
    assert "['texcoords_1'] of mesh 0 have a set other than 0" in messages
    back = read_scene(out).meshes[0]
    np.testing.assert_array_equal(back.vertices, _TRI)
    np.testing.assert_array_equal(back.vertex_attrs["normals"], normals)
    assert "texcoords_1" not in back.vertex_attrs


def test_write_emissive_tint_on_a_texture_round_trips(tmp_path: Path) -> None:
    scene = dataclasses.replace(
        _material_scene(),
        materials=(
            SceneMaterial(emissive=(0.5, 0.2, 0.1), emissive_texture=0),
            SceneMaterial(),
        ),
    )
    out = tmp_path / "glow.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    back = read_scene(out).materials[0]
    assert back.emissive_texture == 0 and back.emissive == (0.5, 0.2, 0.1)


# ---------------------------------------------------------------------------
# reference implementation
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
def test_reference_reads_our_file(tmp_path: Path) -> None:
    collada = pytest.importorskip("collada")
    out = tmp_path / "ref.dae"
    write_scene(_material_scene(), out)
    doc = collada.Collada(str(out))
    assert len(doc.geometries) == 1
    prims = list(doc.geometries[0].primitives)
    # The triangle joins the quad's polylist form; one block per material.
    assert [type(p).__name__ for p in prims] == ["Polylist", "Polylist"]
    assert [p.material for p in prims] == ["material0", "material1"]
    tri, quad = prims
    assert tri.vcounts.tolist() == [3] and quad.vcounts.tolist() == [4]
    np.testing.assert_array_equal(tri.vertex_index, [0, 1, 4])
    np.testing.assert_array_equal(
        quad.triangleset().vertex_index, [[0, 1, 2], [0, 2, 3]]
    )
    assert tri.normal is not None and tri.texcoordset
    assert len(doc.materials) == 2
    assert doc.materials[0].effect.diffuse is not None
    assert len(doc.scene.nodes) == 1
    geoms = list(doc.scene.objects("geometry"))
    assert len(geoms) == 1
    world = np.asarray(list(geoms[0].primitives())[0].vertex)
    expected = _surface().vertices * 2 + [1, 2, 3]
    np.testing.assert_allclose(sorted(world.tolist()), sorted(expected.tolist()))


@pytest.mark.filterwarnings("ignore")
def test_reference_reads_our_shared_nodes(tmp_path: Path) -> None:
    """A node under two parents is written in <library_nodes>, the only
    place that reader resolves an <instance_node>."""
    collada = pytest.importorskip("collada")
    scene = read_scene(_write(tmp_path, _copies_dae()))
    out = tmp_path / "shared.dae"
    write_scene(scene, out)
    doc = collada.Collada(str(out))
    assert len(list(doc.scene.objects("geometry"))) == 2


@pytest.mark.filterwarnings("ignore")
def test_reference_reads_our_skin(tmp_path: Path) -> None:
    collada = pytest.importorskip("collada")
    data, ibm = _skinned_glb()
    glb = tmp_path / "skin.glb"
    glb.write_bytes(data)
    out = tmp_path / "skin.dae"
    write_scene(gltf_read_scene(glb), out)
    doc = collada.Collada(str(out))
    (skin,) = doc.controllers
    names = [str(n) for n in skin.weight_joints]
    assert names == ["node1", "node2"]
    np.testing.assert_allclose(skin.joint_matrices["node2"], ibm[1])
    assert skin.vcounts.tolist() == [2, 2, 2]
    np.testing.assert_allclose(skin.weights.data.ravel(), [0.5] * 6)
    assert [type(o).__name__ for o in doc.scene.objects("controller")] == ["BoundSkin"]


@pytest.mark.filterwarnings("ignore")
def test_reference_file_reads_here(tmp_path: Path) -> None:
    collada = pytest.importorskip("collada")

    doc = collada.Collada()
    effect = collada.material.Effect(
        "effect0", [], "phong", diffuse=(0.2, 0.4, 0.6), specular=(0, 1, 0)
    )
    mat = collada.material.Material("material0", "mymaterial", effect)
    doc.effects.append(effect)
    doc.materials.append(mat)
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float32)
    vert_src = collada.source.FloatSource("verts", verts.ravel(), ("X", "Y", "Z"))
    norm_src = collada.source.FloatSource("norms", normals.ravel(), ("X", "Y", "Z"))
    geom = collada.geometry.Geometry(doc, "geometry0", "square", [vert_src, norm_src])
    inputs = collada.source.InputList()
    inputs.addInput(0, "VERTEX", "#verts")
    inputs.addInput(1, "NORMAL", "#norms")
    idx = np.array([0, 0, 1, 0, 2, 0, 0, 1, 2, 1, 3, 1])
    triset = geom.createTriangleSet(idx, inputs, "materialref")
    geom.primitives.append(triset)
    doc.geometries.append(geom)
    matnode = collada.scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = collada.scene.GeometryNode(geom, [matnode])
    node = collada.scene.Node(
        "node0",
        children=[geomnode],
        transforms=[collada.scene.TranslateTransform(0, 0, 9)],
    )
    myscene = collada.scene.Scene("myscene", [node])
    doc.scenes.append(myscene)
    doc.scene = myscene
    path = tmp_path / "pyc.dae"
    with open(path, "wb") as fh:
        doc.write(fh)

    got = read_scene(path)
    assert got.materials[0].base_color == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert got.materials[0].extras["specular"] == pytest.approx((0.0, 1.0, 0.0, 1.0))
    mesh = got.meshes[0]
    assert len(mesh.vertices) == 6
    assert mesh.element_types.tolist() == [ELEMENT_TYPES["triangle"]] * 2
    np.testing.assert_array_equal(mesh.vertex_attrs["normals"][[0, 3]], normals)
    assert mesh.element_attrs["material"].tolist() == [0, 0]
    np.testing.assert_array_equal(got.nodes[0].matrix[:3, 3], [0, 0, 9])
    flat = got.to_polydata()
    assert flat.vertices[:, 2].tolist() == [9.0] * 6


def test_read_texture_ignores_another_profiles_param_of_the_same_sid(
    tmp_path: Path,
) -> None:
    common = (
        '<newparam sid="surf"><surface type="2D"><init_from>img0</init_from>'
        "</surface></newparam>"
        '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>'
    )
    glsl = (
        '<profile_GLSL><newparam sid="surf"><surface type="2D">'
        "<init_from>img1</init_from></surface></newparam></profile_GLSL>"
    )
    technique = (
        '<phong><diffuse><texture texture="samp" texcoord="UV"/></diffuse></phong>'
    )
    images = (
        '<image id="img0"><init_from>common.png</init_from></image>'
        '<image id="img1"><init_from>glsl.png</init_from></image>'
    )
    text = _effect_dae(technique, newparams=common, images=images).replace(
        "</profile_COMMON>", "</profile_COMMON>" + glsl
    )
    scene = read_scene(_write(tmp_path, text))
    tex = scene.textures[scene.materials[0].base_color_texture]
    assert scene.images[tex.image].uri == "common.png"


def test_read_bool_array_source_is_named_in_the_error(tmp_path: Path) -> None:
    text = _tri_dae().replace(
        _source("g0-pos", _TRI, "X Y Z"),
        '<source id="g0-pos"><bool_array id="b" count="9">'
        + " ".join(["true"] * 9)
        + '</bool_array><technique_common><accessor source="#b" count="3" '
        'stride="3"><param name="X" type="bool"/><param name="Y" type="bool"/>'
        '<param name="Z" type="bool"/></accessor></technique_common></source>',
    )
    with pytest.raises(CodecError, match="bool_array"):
        read_scene(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("conn", "offsets", "kind", "match"),
    [
        ([0, 1, 2], [0, 3], "line", "a line, has 3 vertices"),
        ([0, 1, 3, 2], [0, 4], "triangle", "a triangle, has 4 vertices"),
        ([0, 1], [0, 2], "polygon", "at least 3"),
        ([0, 1], [0, 2], "triangle_strip", "at least 3"),
        ([0, 1, 7], [0, 3], "triangle", "outside 0..4"),
        ([0, 1, -1], [0, 3], "triangle", "outside 0..4"),
        ([0, 1, 2], [0], "triangle", "one per element"),
        ([0, 1, 2], [3, 0], "triangle", "rise from 0"),
        ([0, 1, 2, 3], [0, 3], "triangle", "rise from 0"),
    ],
)
def test_write_refuses_connectivity_a_reader_could_not_read_back(
    tmp_path: Path, conn: list, offsets: list, kind: str, match: str
) -> None:
    mesh = PolyData(
        vertices=np.zeros((5, 3)),
        connectivity=np.array(conn, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.int64),
        element_types=np.full(
            max(len(offsets) - 1, 1), ELEMENT_TYPES[kind], dtype=np.uint8
        ),
    )
    scene = SceneData(meshes=(mesh,), nodes=(SceneNode(mesh=0),), scenes=((0,),))
    with pytest.raises(CodecError, match=re.escape(match)):
        write_scene(scene, tmp_path / "bad.dae")


def test_write_refuses_a_text_handle_in_another_encoding(tmp_path: Path) -> None:
    poly = dataclasses.replace(_surface(), global_attrs={"mesh_name": "café"})
    with (
        open(tmp_path / "latin.dae", "w", encoding="latin-1") as fh,
        pytest.raises(CodecError, match="encoding='utf-8'"),
    ):
        write(poly, fh)


def test_write_ascii_document_to_a_latin_1_handle_reads_back(tmp_path: Path) -> None:
    out = tmp_path / "ascii.dae"
    with open(out, "w", encoding="latin-1") as fh:
        write(_surface(), fh)
    with pytest.warns(UserWarning, match="flattens"):
        back = read(out)
    assert len(back.element_types) == 2


def test_write_stale_transforms_of_a_tiny_node_lose_to_its_matrix(
    tmp_path: Path,
) -> None:
    stale = [{"kind": "scale", "sid": "s", "values": [1e-12, 1e-12, 1e-12]}]
    matrix = np.diag([1e-10, 1e-10, 1e-10, 1.0])
    node = SceneNode(mesh=0, matrix=matrix, extras={"transforms": stale})
    scene = SceneData(meshes=(_surface(),), nodes=(node,), scenes=((0,),))
    out = tmp_path / "tiny.dae"
    write_scene(scene, out)
    np.testing.assert_array_equal(read_scene(out).nodes[0].matrix, matrix)


def test_refuse_instance_without_url_names_the_document_once(tmp_path: Path) -> None:
    text = _dae(
        '<library_visual_scenes><visual_scene id="S"><node id="n">'
        "<instance_geometry/></node></visual_scene></library_visual_scenes>"
    )
    with pytest.raises(CodecError) as info:
        read_scene(_write(tmp_path, text))
    assert str(info.value) == "'m.dae': node 'n' instance_geometry has no url."


@pytest.mark.parametrize("value", ["not a date", "2024-01-02", 20240102])
def test_write_asset_date_that_is_not_a_date_time_is_the_epoch(
    tmp_path: Path, value: Any
) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={"asset": {"created": value}},
    )
    out = tmp_path / "date.dae"
    with pytest.warns(UserWarning, match="not an ISO 8601 date and time"):
        write_scene(scene, out)
    assert read_scene(out).global_attrs["asset"]["created"] == "1970-01-01T00:00:00Z"


def test_write_asset_date_time_is_kept(tmp_path: Path) -> None:
    stamp = "2024-01-02T03:04:05Z"
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={"asset": {"created": stamp, "modified": stamp}},
    )
    out = tmp_path / "date.dae"
    write_scene(scene, out)
    asset = read_scene(out).global_attrs["asset"]
    assert asset["created"] == asset["modified"] == stamp


def test_write_unit_name_that_is_not_a_name_token_is_omitted(tmp_path: Path) -> None:
    out = tmp_path / "unit.dae"
    with pytest.warns(UserWarning, match="not an XML name token"):
        write(_surface(), out, unit={"name": "centi meter", "meter": 0.01})
    with pytest.warns(UserWarning, match="flattens"):
        read(out)
    unit = read_scene(out).global_attrs["asset"]["unit"]
    assert unit == {"name": "meter", "meter": 0.01}


def _random_mesh(rng: np.random.Generator, n_materials: int) -> PolyData:
    n_verts = int(rng.integers(8, 20))
    verts = rng.normal(size=(n_verts, 3))
    groups = []
    for kind, low, high in (
        ("triangle", 3, 3),
        ("quad", 4, 4),
        ("polygon", 5, 7),
        ("line", 2, 2),
        ("poly_line", 3, 5),
        ("triangle_strip", 4, 6),
    ):
        for _ in range(int(rng.integers(0, 3))):
            size = int(rng.integers(low, high + 1))
            corners = rng.choice(n_verts, size=size, replace=False)
            groups.append((kind, corners[None, :]))
    if not groups:
        groups.append(("triangle", np.array([[0, 1, 2]])))
    poly = make_polydata(verts, groups)
    attrs = {}
    if rng.random() < 0.7:
        attrs["normals"] = rng.normal(size=(n_verts, 3))
    if rng.random() < 0.7:
        attrs["texcoords"] = rng.random((n_verts, 2))
    if rng.random() < 0.5:
        attrs["texcoords_1"] = rng.random((n_verts, 2))
    if rng.random() < 0.5:
        attrs["colors"] = rng.random((n_verts, 4))
    element_attrs = {}
    if n_materials:
        element_attrs["material"] = rng.integers(
            -1, n_materials, len(poly.element_types)
        ).astype(np.int32)
    return dataclasses.replace(poly, vertex_attrs=attrs, element_attrs=element_attrs)


def _random_scene(seed: int) -> SceneData:
    rng = np.random.default_rng(seed)
    n_materials = int(rng.integers(0, 3))
    materials = tuple(
        SceneMaterial(name=f"m{i}", base_color=(*rng.random(3), 1.0), metallic=0.0)
        for i in range(n_materials)
    )
    meshes = tuple(_random_mesh(rng, n_materials) for _ in range(rng.integers(1, 4)))
    n_nodes = int(rng.integers(1, 7))
    nodes = []
    for i in range(n_nodes):
        m = np.eye(4)
        m[:3, 3] = rng.normal(size=3)
        m[:3, :3] = np.diag(rng.uniform(0.5, 2.0, 3))
        children = tuple(
            int(c)
            for c in rng.choice(
                np.arange(i + 1, n_nodes),
                size=min(n_nodes - i - 1, int(rng.integers(0, 3))),
                replace=False,
            )
        )
        mesh = int(rng.integers(0, len(meshes))) if rng.random() < 0.8 else None
        nodes.append(SceneNode(name=f"n{i}", mesh=mesh, matrix=m, children=children))
    reached = {c for node in nodes for c in node.children}
    roots = tuple(i for i in range(n_nodes) if i not in reached)
    return SceneData(
        meshes=meshes, nodes=tuple(nodes), materials=materials, scenes=(roots,)
    )


def _mesh_elements(mesh: PolyData, world: np.ndarray) -> list:
    """Every element of a mesh as (type, material, corners), corners in world space."""
    out = []
    material = mesh.element_attrs.get("material")
    pts = mesh.vertices @ world[:3, :3].T + world[:3, 3]
    for k, code in enumerate(mesh.element_types):
        ids = mesh.connectivity[mesh.offsets[k] : mesh.offsets[k + 1]]
        corners = tuple(
            tuple(
                np.round(
                    np.concatenate(
                        [pts[v]]
                        + [mesh.vertex_attrs[a][v] for a in sorted(mesh.vertex_attrs)]
                    ),
                    9,
                )
            )
            for v in ids
        )
        mat = -1 if material is None else int(material[k])
        out.append((int(code), mat, corners))
    return sorted(out)


def _elements(scene: SceneData) -> list:
    """Every element of every mesh a node reaches, then of every mesh alone."""
    out = []

    def walk(idx: int, parent: np.ndarray) -> None:
        node = scene.nodes[idx]
        world = parent @ node.matrix
        if node.mesh is not None:
            out.extend(_mesh_elements(scene.meshes[node.mesh], world))
        for child in node.children:
            walk(child, world)

    for root in scene.scenes[scene.active_scene]:
        walk(root, np.eye(4))
    meshes = sorted(_mesh_elements(m, np.eye(4)) for m in scene.meshes)
    return [sorted(out), meshes]


@pytest.mark.parametrize("seed", range(40))
def test_random_scene_round_trips(tmp_path: Path, seed: int) -> None:
    scene = _random_scene(seed)
    out = tmp_path / "random.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
        back = read_scene(out)
    assert _elements(back) == _elements(scene)
    assert [m.base_color for m in back.materials] == [
        m.base_color for m in scene.materials
    ]


def test_write_copy_of_a_copy_is_instanced_from_the_original(tmp_path: Path) -> None:
    """A subtree copying a copy resolves to the node both copy, so its id
    is written once."""
    nodes = (
        SceneNode(name="root", children=(1, 2, 3)),
        SceneNode(name="S", mesh=0, extras={"id": "s"}),
        SceneNode(name="P", children=(4,), extras={"id": "p"}),
        SceneNode(name="P", children=(5,), extras={"id": "p"}),
        SceneNode(name="S", mesh=0, extras={"id": "s"}),
        SceneNode(name="S", mesh=0, extras={"id": "s"}),
    )
    scene = SceneData(meshes=(_surface(),), nodes=nodes, scenes=((0,), (5,)))
    out = tmp_path / "chain.dae"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_scene(scene, out)
    ids = [e.get("id") for e in ET.parse(out).iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    # The first write wraps the second scene's shared root, which reads
    # back as a node of its own; from then on a write is stable.
    second, third = tmp_path / "second.dae", tmp_path / "third.dae"
    write_scene(read_scene(out), second)
    write_scene(read_scene(second), third)
    assert third.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("root", [5, -1])
def test_write_scene_root_out_of_range_refused(tmp_path: Path, root: int) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0), SceneNode(mesh=0)),
        scenes=((root,),),
    )
    with (
        pytest.warns(UserWarning, match="in no scene"),
        pytest.raises(CodecError, match=f"root node {root}"),
    ):
        write_scene(scene, tmp_path / "root.dae")


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "cp500"])
def test_write_refuses_a_text_handle_not_spelling_ascii_as_ascii(
    tmp_path: Path, encoding: str
) -> None:
    with (
        open(tmp_path / "wide.dae", "w", encoding=encoding) as fh,
        pytest.raises(CodecError, match="encoding='utf-8'"),
    ):
        write(_surface(), fh)


def test_write_utf_8_sig_handle_reads_back(tmp_path: Path) -> None:
    poly = dataclasses.replace(_surface(), global_attrs={"mesh_name": "café"})
    out = tmp_path / "bom.dae"
    with open(out, "w", encoding="utf-8-sig") as fh:
        write(poly, fh)
    assert read_scene(out).meshes[0].global_attrs["mesh_name"] == "café"


def test_write_unsigned_offsets_that_fall_refused(tmp_path: Path) -> None:
    mesh = PolyData(
        vertices=np.zeros((4, 3)),
        connectivity=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        offsets=np.array([0, 6, 3, 6], dtype=np.uint32),
        element_types=np.full(3, ELEMENT_TYPES["triangle"], dtype=np.uint8),
    )
    scene = SceneData(meshes=(mesh,), nodes=(SceneNode(mesh=0),), scenes=((0,),))
    with pytest.raises(CodecError, match="rise from 0"):
        write_scene(scene, tmp_path / "u.dae")


@pytest.mark.parametrize(
    "value",
    [
        "20200101T000000",
        "2020-01-01T00",
        "2020-W01-1T00:00",
        "2020-01-01T00:00:00,5",
        "2020-01-01T00:00:00+0100",
    ],
)
def test_write_iso_date_outside_xs_date_time_is_the_epoch(
    tmp_path: Path, value: str
) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={"asset": {"created": value}},
    )
    with pytest.warns(UserWarning, match="not an ISO 8601"):
        write_scene(scene, tmp_path / "d.dae")


def test_write_datetime_value_is_written_as_iso(tmp_path: Path) -> None:
    scene = SceneData(
        meshes=(_surface(),),
        nodes=(SceneNode(mesh=0),),
        scenes=((0,),),
        global_attrs={"asset": {"created": datetime(2020, 1, 1, 12)}},
    )
    out = tmp_path / "d.dae"
    write_scene(scene, out)
    assert read_scene(out).global_attrs["asset"]["created"] == "2020-01-01T12:00:00"


def test_read_diffuse_alpha_not_finite_is_opaque(tmp_path: Path) -> None:
    technique = "<lambert><diffuse><color>1 1 1 NaN</color></diffuse></lambert>"
    with pytest.warns(UserWarning, match="not finite"):
        mat = read_scene(_write(tmp_path, _effect_dae(technique))).materials[0]
    assert mat.base_color == (1.0, 1.0, 1.0, 1.0)
    assert mat.alpha_mode == "OPAQUE"


def test_read_unbound_and_id_bound_instances_share_one_mesh(tmp_path: Path) -> None:
    """Two bindings resolving every symbol to the same material are one mesh."""
    geo = _geometry("g0", _TRI, _triangles("g0", [[0, 1, 2]], material="mat0"))
    text = _dae(
        '<library_effects><effect id="fx0"><profile_COMMON><technique sid="c">'
        "<phong/></technique></profile_COMMON></effect></library_effects>"
        '<library_materials><material id="mat0"><instance_effect url="#fx0"/>'
        "</material></library_materials>"
        f"<library_geometries>{geo}</library_geometries>"
        '<library_visual_scenes><visual_scene id="S">'
        '<node id="a"><instance_geometry url="#g0"/></node>'
        '<node id="b"><instance_geometry url="#g0"><bind_material>'
        '<technique_common><instance_material symbol="mat0" target="#mat0"/>'
        "</technique_common></bind_material></instance_geometry></node>"
        "</visual_scene></library_visual_scenes>"
    )
    scene = read_scene(_write(tmp_path, text))
    assert len(scene.meshes) == 1
    assert [n.mesh for n in scene.nodes] == [0, 0]
