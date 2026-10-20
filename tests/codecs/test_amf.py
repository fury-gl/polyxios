"""AMF: an XML document of objects, volumes and constellations, plain or zipped.

Documents are hand-written XML in the layout the specification gives, so the
reader is tested against the format and not against the writer beside it.
"""

from __future__ import annotations

import io
from pathlib import Path
import struct
import warnings
import zipfile

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios.codecs import _3mf, _amf as _amf_module
from polyxios.codecs._amf import read, write
from polyxios.exceptions import CodecError, LazyReadError

_TET = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
_TET_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def _color(r, g, b, a=None) -> str:
    alpha = "" if a is None else f"<a>{a}</a>"
    return f"<color><r>{r}</r><g>{g}</g><b>{b}</b>{alpha}</color>"


def _vertex(p, extra: str = "") -> str:
    x, y, z = p
    return f"<vertex><coordinates><x>{x}</x><y>{y}</y><z>{z}</z></coordinates>{extra}</vertex>"


def _triangle(t, extra: str = "") -> str:
    a, b, c = t
    return f"<triangle>{extra}<v1>{a}</v1><v2>{b}</v2><v3>{c}</v3></triangle>"


def _volume(faces=_TET_FACES, *, attrs: str = "", inner: str = "") -> str:
    return f"<volume{attrs}>{inner}{''.join(_triangle(f) for f in faces)}</volume>"


def _object(
    oid: str = "1",
    *,
    name: str | None = "tet",
    vertices=_TET,
    volumes: str | None = None,
    inner: str = "",
) -> str:
    meta = f'<metadata type="name">{name}</metadata>' if name is not None else ""
    verts = "".join(_vertex(p) for p in vertices)
    vols = _volume() if volumes is None else volumes
    return (
        f'<object id="{oid}">{meta}{inner}<mesh><vertices>{verts}</vertices>'
        f"{vols}</mesh></object>"
    )


def _amf(body: str, *, unit: str = "millimeter") -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><amf unit="{unit}">{body}</amf>'


def _file(tmp_path: Path, text: str | bytes, name: str = "a.amf") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode() if isinstance(text, str) else text)
    return path


def _zipped(text: str, member: str = "a.amf") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, text)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_reads_an_object(tmp_path: Path) -> None:
    mesh = read(_file(tmp_path, _amf(_object())))
    np.testing.assert_array_equal(mesh.vertices, _TET)
    np.testing.assert_array_equal(mesh.connectivity.reshape(-1, 3), _TET_FACES)
    assert mesh.element_tags["tet"].tolist() == [0, 1, 2, 3]
    assert mesh.global_attrs == {"unit": "millimeter"}
    assert not mesh.element_attrs
    assert not mesh.vertex_attrs


def test_an_unnamed_object_is_tagged_by_its_id(tmp_path: Path) -> None:
    mesh = read(_file(tmp_path, _amf(_object("7", name=None))))
    assert list(mesh.element_tags) == ["object_7"]


def test_every_volume_of_an_object_is_read(tmp_path: Path) -> None:
    volumes = _volume(_TET_FACES[:2]) + _volume(_TET_FACES[2:])
    mesh = read(_file(tmp_path, _amf(_object(volumes=volumes))))
    np.testing.assert_array_equal(mesh.connectivity.reshape(-1, 3), _TET_FACES)


def test_two_objects_keep_their_own_vertices(tmp_path: Path) -> None:
    body = _object("1", name="a") + _object("2", name="b")
    mesh = read(_file(tmp_path, _amf(body)))
    assert len(mesh.vertices) == 8
    assert mesh.element_tags["b"].tolist() == [4, 5, 6, 7]
    assert mesh.connectivity.reshape(-1, 3)[4:].min() == 4


def test_unit_and_metadata_are_global_attrs(tmp_path: Path) -> None:
    body = '<metadata type="author">me</metadata>' + _object()
    mesh = read(_file(tmp_path, _amf(body, unit="inch")))
    assert mesh.global_attrs == {"unit": "inch", "author": "me"}


def test_both_spellings_of_micrometres_are_read_quietly(tmp_path: Path) -> None:
    for unit in ("micron", "micrometer"):
        assert (
            read(_file(tmp_path, _amf(_object(), unit=unit))).global_attrs["unit"]
            == unit
        )


def test_an_unknown_unit_is_kept_with_a_warning(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unit 'cubit' is not one AMF defines"):
        mesh = read(_file(tmp_path, _amf(_object(), unit="cubit")))
    assert mesh.global_attrs["unit"] == "cubit"


def test_the_most_specific_colour_wins(tmp_path: Path) -> None:
    material = f'<material id="5">{_color(0, 0, 1)}</material>'
    faces = "".join(
        _triangle(f, _color(1, 1, 1, 0.5) if k == 0 else "")
        for k, f in enumerate(_TET_FACES)
    )
    volumes = (
        f'<volume materialid="5">{faces}</volume>'
        f"<volume>{_color(0, 1, 0)}{_triangle(_TET_FACES[0])}</volume>"
    )
    body = (
        material
        + _object(volumes=volumes)
        + _object(
            "2", name="m", volumes=_volume(_TET_FACES[:1], attrs=' materialid="5"')
        )
    )
    body_obj = _object(
        "3",
        name="o",
        inner=_color(1, 0, 0),
        volumes=_volume(_TET_FACES[:1], attrs=' materialid="5"'),
    )
    mesh = read(_file(tmp_path, _amf(body + body_obj)))
    colors = mesh.element_attrs["colors"]
    np.testing.assert_array_equal(colors[0], [1, 1, 1, 0.5])
    np.testing.assert_array_equal(colors[1], [0, 0, 1, 1])
    np.testing.assert_array_equal(colors[4], [0, 1, 0, 1])
    np.testing.assert_array_equal(colors[5], [0, 0, 1, 1])
    np.testing.assert_array_equal(colors[6], [1, 0, 0, 1])


def test_a_triangle_without_a_colour_is_nan(tmp_path: Path) -> None:
    faces = _triangle(_TET_FACES[0], _color(1, 0, 0)) + _triangle(_TET_FACES[1])
    mesh = read(_file(tmp_path, _amf(_object(volumes=f"<volume>{faces}</volume>"))))
    assert np.isnan(mesh.element_attrs["colors"][1]).all()


def test_a_colour_formula_reads_as_nan_with_a_warning(tmp_path: Path) -> None:
    faces = _triangle(_TET_FACES[0], _color("x*0.5", 0, 0))
    with pytest.warns(UserWarning, match=r"1 colour channel\(s\) are formulas"):
        mesh = read(_file(tmp_path, _amf(_object(volumes=f"<volume>{faces}</volume>"))))
    assert "colors" not in mesh.element_attrs


def test_vertex_colours_and_normals(tmp_path: Path) -> None:
    verts = [
        _vertex(
            _TET[0], _color(1, 0, 0) + "<normal><nx>0</nx><ny>0</ny><nz>1</nz></normal>"
        ),
        *(_vertex(p) for p in _TET[1:3]),
        _vertex(
            _TET[3],
            "<normal><nx>1</nx><ny>0</ny><nz>0</nz></normal>" + _color(0, 1, 0, 0.25),
        ),
    ]
    body = f'<object id="1"><mesh><vertices>{"".join(verts)}</vertices>{_volume()}</mesh></object>'
    mesh = read(_file(tmp_path, _amf(body)))
    colors, normals = mesh.vertex_attrs["colors"], mesh.vertex_attrs["normals"]
    np.testing.assert_array_equal(colors[[0, 3]], [[1, 0, 0, 1], [0, 1, 0, 0.25]])
    assert np.isnan(colors[1:3]).all()
    np.testing.assert_array_equal(normals[[0, 3]], [[0, 0, 1], [1, 0, 0]])


def test_a_material_the_document_lacks_is_warned_about(tmp_path: Path) -> None:
    volumes = _volume(attrs=' materialid="9"')
    with pytest.warns(UserWarning, match=r"material\(s\) \['9'\] are named"):
        mesh = read(_file(tmp_path, _amf(_object(volumes=volumes))))
    assert len(mesh.element_types) == len(_TET_FACES)
    assert "colors" not in mesh.element_attrs


def test_the_void_material_needs_no_definition(tmp_path: Path) -> None:
    volumes = _volume(attrs=' materialid="0"')
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = read(_file(tmp_path, _amf(_object(volumes=volumes))))
    assert len(mesh.element_types) == len(_TET_FACES)


def test_edges_and_texmaps_are_skipped_with_a_warning(tmp_path: Path) -> None:
    faces = _triangle(_TET_FACES[0], "<texmap rtexid='1'><utex1>0</utex1></texmap>")
    edges = "<edges><edge><v1>0</v1><dx1>0</dx1></edge></edges>"
    volumes = f"{edges}<volume>{faces}</volume>"
    with (
        pytest.warns(UserWarning, match="<edges> is not read; curved triangles"),
        pytest.warns(UserWarning, match="<texmap> is not read; texture coordinates"),
    ):
        mesh = read(_file(tmp_path, _amf(_object(volumes=volumes))))
    assert len(mesh.element_types) == 1


# ---------------------------------------------------------------------------
# Constellations
# ---------------------------------------------------------------------------


def _instance(oid: str, **numbers: float) -> str:
    inner = "".join(f"<{k}>{v}</{k}>" for k, v in numbers.items())
    return f'<instance objectid="{oid}">{inner}</instance>'


def test_a_constellation_places_its_instances(tmp_path: Path) -> None:
    body = _object() + (
        '<constellation id="10">'
        + _instance("1", deltax=10)
        + _instance("1", rz=90, deltay=5)
        + "</constellation>"
    )
    mesh = read(_file(tmp_path, _amf(body)))
    assert len(mesh.vertices) == 8
    np.testing.assert_allclose(mesh.vertices[:4], np.array(_TET) + [10, 0, 0])
    # 90 degrees about z takes x to y.
    np.testing.assert_allclose(mesh.vertices[5], [0, 6, 0], atol=1e-12)


def test_rotations_turn_about_x_then_y_then_z(tmp_path: Path) -> None:
    body = (
        _object()
        + '<constellation id="10">'
        + _instance("1", rx=90, rz=90)
        + "</constellation>"
    )
    mesh = read(_file(tmp_path, _amf(body)))
    # y -> z under rx, then z stays under rz; x -> y under rz.
    np.testing.assert_allclose(mesh.vertices[2], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[1], [0, 1, 0], atol=1e-12)


def test_nested_constellations_compose(tmp_path: Path) -> None:
    body = _object() + (
        '<constellation id="10">' + _instance("1", deltax=1) + "</constellation>"
        '<constellation id="11">' + _instance("10", deltaz=2) + "</constellation>"
    )
    mesh = read(_file(tmp_path, _amf(body)))
    np.testing.assert_allclose(mesh.vertices, np.array(_TET) + [1, 0, 2])


def test_an_object_no_constellation_names_is_placed_as_it_stands(
    tmp_path: Path,
) -> None:
    body = (
        _object("1", name="a")
        + _object("2", name="b")
        + ('<constellation id="10">' + _instance("1", deltax=5) + "</constellation>")
    )
    mesh = read(_file(tmp_path, _amf(body)))
    np.testing.assert_allclose(mesh.vertices[4:], _TET)
    assert list(mesh.element_tags) == ["a", "b"]


def test_a_constellation_containing_itself_is_refused(tmp_path: Path) -> None:
    body = _object() + (
        '<constellation id="10">' + _instance("11") + "</constellation>"
        '<constellation id="11">' + _instance("10") + "</constellation>"
        '<constellation id="12">' + _instance("10") + "</constellation>"
    )
    with pytest.raises(CodecError, match="contains itself"):
        read(_file(tmp_path, _amf(body)))


def test_constellations_all_in_a_cycle_are_refused(tmp_path: Path) -> None:
    body = _object("5") + (
        '<constellation id="10">'
        + _instance("11")
        + _instance("5")
        + "</constellation>"
        '<constellation id="11">' + _instance("10") + "</constellation>"
    )
    with pytest.raises(
        CodecError, match="constellation 10 -> 11 -> 10 contains itself"
    ):
        read(_file(tmp_path, _amf(body)))


def test_constellations_placing_past_the_document_size_are_refused(
    tmp_path: Path,
) -> None:
    # Each constellation places the one before it twice: 2**40 tetrahedra.
    body = _object("1") + '<constellation id="c0">' + _instance("1") * 2
    body += "</constellation>"
    for k in range(1, 40):
        body += f'<constellation id="c{k}">' + _instance(f"c{k - 1}") * 2
        body += "</constellation>"
    with pytest.raises(CodecError, match="place more than"):
        read(_file(tmp_path, _amf(body)))


def test_a_doubling_chain_is_refused_at_the_first_constellation_past_the_budget(
    tmp_path: Path, monkeypatch
) -> None:
    # A tetrahedron costs 24; c17 places 2**18 of them, past a 2**22 budget.
    # Summing the whole chain first would carry integers of 4000 bits.
    monkeypatch.setattr(_amf_module, "_EXPANSION_PER_BYTE", 0)
    monkeypatch.setattr(_amf_module, "_EXPANSION_FLOOR", 1 << 22)
    body = _object("1") + '<constellation id="c0">' + _instance("1") * 2
    body += "</constellation>"
    for k in range(1, 4000):
        body += f'<constellation id="c{k}">' + _instance(f"c{k - 1}") * 2
        body += "</constellation>"
    with pytest.raises(CodecError, match="constellation c17 and those in it place"):
        read(_file(tmp_path, _amf(body)))


def test_a_doubling_chain_over_one_vertex_is_refused(tmp_path: Path) -> None:
    # A million placements of a single vertex: each placement costs work.
    body = _object("1", vertices=_TET[:1], volumes="")
    body += '<constellation id="c0">' + _instance("1") * 2 + "</constellation>"
    for k in range(1, 20):
        body += f'<constellation id="c{k}">' + _instance(f"c{k - 1}") * 2
        body += "</constellation>"
    with pytest.raises(CodecError, match="each placement counted as 16 more"):
        read(_file(tmp_path, _amf(body)))


def test_placements_keep_their_order_across_objects(tmp_path: Path) -> None:
    body = _object("1", name="a") + _object("2", name="b", volumes=_volume([(0, 1, 2)]))
    body += (
        '<constellation id="10">'
        + _instance("1", deltax=1)
        + _instance("2", deltax=2)
        + _instance("1", deltax=3, rz=90)
        + "</constellation>"
    )
    mesh = read(_file(tmp_path, _amf(body)))
    tet = np.array(_TET, float)
    turned = tet[:, [1, 0, 2]] * [-1, 1, 1]
    np.testing.assert_allclose(
        mesh.vertices,
        np.vstack([tet + [1, 0, 0], tet + [2, 0, 0], turned + [3, 0, 0]]),
        atol=1e-12,
    )
    assert {k: v.tolist() for k, v in mesh.element_tags.items()} == {
        "a": [0, 1, 2, 3, 5, 6, 7, 8],
        "b": [4],
    }
    assert mesh.connectivity.reshape(-1, 3)[4].tolist() == [4, 5, 6]
    assert mesh.connectivity.reshape(-1, 3)[5].tolist() == [8, 10, 9]


def test_a_long_chain_of_constellations_is_read(tmp_path: Path) -> None:
    body = _object("1") + '<constellation id="c0">' + _instance("1", deltax=1)
    body += "</constellation>"
    for k in range(1, 3000):
        body += f'<constellation id="c{k}">' + _instance(f"c{k - 1}", deltax=1)
        body += "</constellation>"
    mesh = read(_file(tmp_path, _amf(body)))
    np.testing.assert_allclose(mesh.vertices, np.array(_TET) + [3000, 0, 0])


def test_the_unit_attribute_wins_over_unit_metadata(tmp_path: Path) -> None:
    body = '<metadata type="unit">parsec</metadata>' + _object()
    mesh = read(_file(tmp_path, _amf(body, unit="inch")))
    assert mesh.global_attrs["unit"] == "inch"


def test_the_children_of_a_skipped_element_are_not_read(tmp_path: Path) -> None:
    # Whatever an extension nests in an element the reader passes over is
    # passed over with it, however much it looks like a triangle.
    inner = "<edges>" + _triangle((0, 1, 2)) + "</edges>"
    body = _object(volumes=_volume(inner=inner))
    with pytest.warns(UserWarning, match="<edges> is not read"):
        mesh = read(_file(tmp_path, _amf(body)))
    assert len(mesh.element_types) == len(_TET_FACES)


def test_an_instance_of_a_missing_object_is_refused(tmp_path: Path) -> None:
    body = _object() + '<constellation id="10">' + _instance("4") + "</constellation>"
    with pytest.raises(CodecError, match="names object 4"):
        read(_file(tmp_path, _amf(body)))


# ---------------------------------------------------------------------------
# Packaging and refusals
# ---------------------------------------------------------------------------


def test_a_zipped_document_is_read(tmp_path: Path) -> None:
    path = _file(tmp_path, _zipped(_amf(_object()), member="model.amf"))
    assert len(read(path).element_types) == 4


def test_a_zip_holding_a_lone_member_of_another_name_is_read(tmp_path: Path) -> None:
    path = _file(tmp_path, _zipped(_amf(_object()), member="model.xml"))
    assert len(read(path).element_types) == 4


def test_a_zip_holding_two_documents_reads_the_first_with_a_warning(
    tmp_path: Path,
) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.amf", _amf(_object()))
        z.writestr("b.amf", _amf(_object() + _object("2")))
    with pytest.warns(UserWarning, match="holds 2 .amf documents; only 'a.amf'"):
        mesh = read(_file(tmp_path, buf.getvalue()))
    assert len(mesh.element_types) == 4


def test_a_zip_holding_no_document_is_refused(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
        z.writestr("b.txt", "y")
    with pytest.raises(CodecError, match="holding no .amf document"):
        read(_file(tmp_path, buf.getvalue()))


def _patched_zip(raw: bytes, *, flag: int = 0, method: int | None = None) -> bytes:
    """Set a general-purpose flag or the compression method in both ZIP headers."""
    out = bytearray(raw)
    for sig, at in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        i = out.find(sig)
        out[i + at] |= flag
        if method is not None:
            out[i + at + 2 : i + at + 4] = method.to_bytes(2, "little")
    return bytes(out)


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"flag": 1}, "encrypted"), ({"method": 97}, "compression")]
)
def test_a_member_zipfile_cannot_open_is_refused(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    raw = _patched_zip(_zipped(_amf(_object())), **kwargs)
    with pytest.raises(CodecError, match=match):
        read(_file(tmp_path, raw))


def _broken_zip(raw: bytes, kind: str) -> bytes:
    """Break an archive the way zipfile reports with other than BadZipFile."""
    out = bytearray(raw)
    if kind == "seek":
        # A directory offset past where it stands sends a seek before byte 0.
        at = out.rfind(b"PK\x05\x06")
        out[at + 16 : at + 20] = struct.pack("<I", 0x7FFFFFFF)
        return bytes(out)
    at = out.find(b"PK\x01\x02")
    if kind == "version":
        out[at + 6 : at + 8] = struct.pack("<H", 0xFF)
    else:
        # The UTF-8 flag over a name that is not UTF-8.
        out[at + 9] |= 0x08
        out[at + 46] = 0xFF
    return bytes(out)


@pytest.mark.parametrize("kind", ["version", "name", "seek"])
def test_a_zip_zipfile_rejects_oddly_is_refused(tmp_path: Path, kind: str) -> None:
    raw = _broken_zip(_zipped(_amf(_object())), kind)
    with pytest.raises(CodecError, match="not a readable ZIP"):
        read(_file(tmp_path, raw))


def test_a_zip_bomb_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_3mf, "_INFLATE_FLOOR", 1 << 16)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.amf", _amf(_object()) + " " * (1 << 20))
    with pytest.raises(CodecError, match="refused as a ZIP bomb"):
        read(_file(tmp_path, buf.getvalue()))


def test_a_broken_zip_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="not a readable ZIP"):
        read(_file(tmp_path, b"PK\x03\x04" + b"\x00" * 40))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("<amf><object id='1'>", "not well-formed XML"),
        ('<?xml version="1.0" encoding="UTF-9"?><amf/>', "unknown encoding"),
        ("<model/>", "root is <model>, not <amf>"),
        (_amf(_object(volumes=_volume([(0, 1, 4)]))), "names vertex 4 but holds 4"),
        (_amf(_object(volumes=_volume([(0, -1, 2)]))), "names vertex -1"),
        (_amf(_object(vertices=[(0, 0, "a")])), "vertex coordinate is not a number"),
        (_amf(_object(vertices=[(0, 0, "inf")])), "is not finite"),
        (
            _amf(
                _object(
                    volumes="<volume><triangle><v1>0</v1><v2>1</v2></triangle></volume>"
                )
            ),
            "not three vertex indices",
        ),
        (_amf(_object(volumes=_volume([("a", 1, 2)]))), "not an integer"),
        (_amf(_object("1") + _object("1")), "two objects id 1"),
        (
            _amf(
                '<object id="1"><mesh><vertices><vertex><coordinates><x>0</x></coordinates></vertex></vertices></mesh></object>'
            ),
            "not three coordinates",
        ),
        (
            '<?xml version="1.0"?><!DOCTYPE amf [<!ENTITY a "b">]><amf/>',
            "declares an XML entity",
        ),
        (
            _amf(_object(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])).replace(
                "</vertices>", f"<vertex>{_color(1, 0, 0)}</vertex></vertices>"
            ),
            "a vertex that is not one set of coordinates",
        ),
        (
            _amf(
                _object(
                    volumes=f"<volume><triangle>{_color(1, 0, 0)}</triangle></volume>"
                )
            ),
            "not three vertex indices",
        ),
        (
            _amf(
                _object(
                    volumes="<volume>"
                    + _triangle((0, 1, 2)).replace(
                        "</triangle>", "<v1>0</v1><v2>1</v2><v3>2</v3></triangle>"
                    )
                    + "</volume>"
                )
            ),
            "gives <v1> twice",
        ),
        (
            _amf(
                _object(
                    volumes="<volume><triangle><v1>0</v1><v1>1</v1><v1>2</v1></triangle></volume>"
                )
            ),
            "gives <v1> twice",
        ),
        (
            _amf(_object(vertices=[(0, 0, 0)])).replace("<y>0</y>", "<x>0</x>"),
            "gives <x> twice",
        ),
    ],
)
def test_a_malformed_document_is_refused(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(CodecError, match=match):
        read(_file(tmp_path, text))


def test_lazy_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LazyReadError):
        read(_file(tmp_path, _amf(_object())), lazy=True)


def test_an_empty_document_reads_as_an_empty_mesh(tmp_path: Path) -> None:
    mesh = read(_file(tmp_path, _amf("")))
    assert mesh.vertices.shape == (0, 3)
    assert len(mesh.element_types) == 0


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)


def _round_trip(poly, tmp_path: Path, **kwargs):
    path = tmp_path / "out.amf"
    write(poly, path, **kwargs)
    return read(path)


def test_tags_become_objects_and_quads_split(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]])), ("quad", np.array([[0, 1, 2, 3]]))],
        element_tags={"a": np.array([1])},
    )
    back = _round_trip(poly, tmp_path)
    assert len(back.element_types) == 3
    assert {k: v.tolist() for k, v in back.element_tags.items()} == {
        "a": [0, 1],
        "object_2": [2],
    }


def test_colours_and_vertex_attributes_round_trip(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4], [0, 4, 2]]))],
        element_attrs={"colors": np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)},
        vertex_attrs={
            "colors": np.array([[0.5, 0.5, 0.5, 1]] * 5),
            "normals": np.tile([0.0, 0.0, 1.0], (5, 1)),
        },
    )
    back = _round_trip(poly, tmp_path)
    np.testing.assert_array_equal(
        back.element_attrs["colors"], [[1, 0, 0, 1], [0, 0, 1, 1]]
    )
    np.testing.assert_array_equal(
        back.vertex_attrs["normals"], np.tile([0, 0, 1.0], (4, 1))
    )
    np.testing.assert_array_equal(back.vertex_attrs["colors"], [[0.5, 0.5, 0.5, 1]] * 4)


def test_a_vertex_column_of_the_wrong_shape_is_not_written(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]]))],
        vertex_attrs={"normals": np.zeros((5, 2))},
    )
    with pytest.warns(
        UserWarning, match=r"vertex_attrs\['normals'\] is not 3 components"
    ):
        back = _round_trip(poly, tmp_path)
    assert not back.vertex_attrs


def test_compress_writes_a_zip_with_a_fixed_timestamp(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("triangle", np.array([[0, 1, 4]]))])
    first, second = io.BytesIO(), io.BytesIO()
    write(poly, first, compress=True)
    write(poly, second, compress=True)
    assert first.getvalue() == second.getvalue()
    assert first.getvalue().startswith(b"PK\x03\x04")
    assert len(read(io.BytesIO(first.getvalue())).element_types) == 1


def test_compress_to_a_buffer_names_its_member_model(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("triangle", np.array([[0, 1, 2]]))])
    buf = io.BytesIO()
    write(poly, buf, compress=True)
    assert zipfile.ZipFile(io.BytesIO(buf.getvalue())).namelist() == ["model.amf"]


def test_names_and_metadata_keep_their_spaces(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={" walls ": np.array([0])},
        global_attrs={"description": " two  words "},
    )
    back = _round_trip(poly, tmp_path)
    assert list(back.element_tags) == [" walls "]
    assert back.global_attrs["description"] == " two  words "


def test_the_unit_and_metadata_are_written(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]]))],
        global_attrs={"unit": "foot", "author": "me", "gnum": 42},
    )
    back = _round_trip(poly, tmp_path)
    assert back.global_attrs == {"unit": "feet", "author": "me"}


def test_an_explicit_unit_the_format_lacks_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("triangle", np.array([[0, 1, 4]]))])
    with pytest.raises(CodecError, match="not an AMF unit"):
        write(poly, tmp_path / "out.amf", unit="cubit")


def test_a_mesh_unit_the_format_lacks_is_written_as_millimetres(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("triangle", np.array([[0, 1, 4]]))], global_attrs={"unit": "parsec"}
    )
    with pytest.warns(UserWarning, match="written as 'millimeter'"):
        back = _round_trip(poly, tmp_path)
    assert back.global_attrs["unit"] == "millimeter"


def test_a_vertex_that_is_not_finite_is_refused(tmp_path: Path) -> None:
    verts = _SQUARE.copy()
    verts[4, 2] = np.inf
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 4]]))])
    with pytest.raises(CodecError, match="NaN or infinite"):
        write(poly, tmp_path / "out.amf")


def test_a_mesh_without_surfaces_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(CodecError, match="AMF holds triangles only"):
        write(poly, tmp_path / "out.amf")


def test_an_empty_tag_group_name_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4], [0, 1, 2]]))],
        element_tags={"": np.array([0]), "b": np.array([1])},
    )
    tags = _round_trip(poly, tmp_path).element_tags
    assert {k: v.tolist() for k, v in tags.items()} == {"": [0], "b": [1]}


def test_through_the_registry(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("triangle", np.array([[0, 1, 4]]))])
    path = tmp_path / "reg.amf"
    polyxios.write(poly, path)
    assert len(polyxios.read(path).element_types) == 1


def test_a_doubling_chain_of_a_many_triangle_object_is_refused(
    tmp_path: Path,
) -> None:
    # Three vertices and 2000 triangles, placed 4096 times: the triangles count.
    faces = [(0, 1, 2)] * 2000
    body = _object("1", vertices=_TET[:3], volumes=_volume(faces))
    body += '<constellation id="c0">' + _instance("1") * 2 + "</constellation>"
    for k in range(1, 12):
        body += f'<constellation id="c{k}">' + _instance(f"c{k - 1}") * 2
        body += "</constellation>"
    with pytest.raises(CodecError, match="vertices and triangles"):
        read(_file(tmp_path, _amf(body)))


@pytest.mark.parametrize("empty_id", ["1", "9"])
def test_a_constellation_sharing_an_object_id_does_not_hide_its_cost(
    tmp_path: Path, monkeypatch, empty_id: str
) -> None:
    # 200 placements of 100 triangles against a budget of 10000: refused
    # whether or not an empty constellation takes the object's id.
    monkeypatch.setattr(_amf_module, "_EXPANSION_PER_BYTE", 0)
    monkeypatch.setattr(_amf_module, "_EXPANSION_FLOOR", 10_000)
    body = _object("1", vertices=_TET[:3], volumes=_volume([(0, 1, 2)] * 100))
    body += f'<constellation id="{empty_id}"></constellation>'
    body += '<constellation id="2">' + _instance("1") * 200 + "</constellation>"
    with pytest.raises(CodecError, match="place more than 10000"):
        read(_file(tmp_path, _amf(body)))


def test_objects_without_vertex_colours_or_normals_allocate_none(
    tmp_path: Path,
) -> None:
    target = _amf_module._AmfTarget("a.amf")
    parser = _amf_module.ET.XMLParser(target=target)
    parser.feed(_amf(_object()).encode())
    parser.close()
    mesh = _amf_module._mesh_of(
        target.objects["1"], target.materials, "a.amf", missing=set()
    )
    assert mesh[3] is None
    assert mesh[4] is None
    assert read(_file(tmp_path, _amf(_object()))).vertex_attrs == {}


def test_normals_of_one_object_leave_the_other_nan(tmp_path: Path) -> None:
    normal = "<normal><nx>0</nx><ny>0</ny><nz>1</nz></normal>"
    body = _object("1", name="a").replace(
        "</coordinates></vertex>", f"</coordinates>{normal}</vertex>", 1
    )
    body += _object("2", name="b")
    mesh = read(_file(tmp_path, _amf(body)))
    normals = mesh.vertex_attrs["normals"]
    assert "colors" not in mesh.vertex_attrs
    np.testing.assert_array_equal(normals[0], [0, 0, 1])
    assert len(normals) == 8
    assert np.isnan(normals[1:]).all()


def test_bool_global_attrs_are_not_written_as_metadata(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET, [("triangle", _TET_FACES)], global_attrs={"name": True, "author": "me"}
    )
    path = tmp_path / "b.amf"
    write(poly, path)
    text = path.read_text()
    assert "True" not in text
    assert read(path).global_attrs == {"unit": "millimeter", "author": "me"}


def _doubling_chain(depth: int, *, then: int = 0) -> str:
    """One vertex placed ``2**depth`` times, then through ``then`` more links."""
    body = _object("1", vertices=_TET[:1], volumes="")
    body += '<constellation id="c0">' + _instance("1") * 2 + "</constellation>"
    for k in range(1, depth):
        body += f'<constellation id="c{k}">' + _instance(f"c{k - 1}") * 2
        body += "</constellation>"
    last = f"c{depth - 1}"
    for k in range(then):
        body += f'<constellation id="l{k}">' + _instance(last) + "</constellation>"
        last = f"l{k}"
    return body


def test_placements_held_by_inner_constellations_count(tmp_path: Path) -> None:
    # 65536 placements within the budget, then carried through four
    # single-instance links: each link holds its own stack of all of them.
    assert len(read(_file(tmp_path, _amf(_doubling_chain(16)))).vertices) == 1 << 16
    with pytest.raises(CodecError, match="place more than 4194304"):
        read(_file(tmp_path, _amf(_doubling_chain(16, then=4)), name="b.amf"))


def test_a_zipped_document_is_budgeted_by_its_size_on_disk(tmp_path: Path) -> None:
    # Two MiB of whitespace deflate to a few KiB: they buy no placements.
    text = _amf(" " * (1 << 21) + _doubling_chain(18))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.amf", text)
    assert len(buf.getvalue()) < 1 << 16
    with pytest.raises(CodecError, match="place more than 4194304"):
        read(_file(tmp_path, buf.getvalue()))


def test_a_vertex_index_past_int64_is_refused(tmp_path: Path) -> None:
    body = _object(volumes=_volume([(0, 1, 99999999999999999999)]))
    with pytest.raises(CodecError, match="names vertex '99999999999999999999'"):
        read(_file(tmp_path, _amf(body)))


def test_constellations_placing_a_vertex_past_the_float_range_are_refused(
    tmp_path: Path,
) -> None:
    body = _object("1") + '<constellation id="c0">'
    body += _instance("1", deltax=1e308) + "</constellation>"
    body += '<constellation id="c1">' + _instance("c0", deltax=1e308)
    body += "</constellation>"
    with pytest.raises(CodecError, match="past the float range"):
        read(_file(tmp_path, _amf(body)))


def test_an_indented_name_and_metadata_drop_their_layout(tmp_path: Path) -> None:
    body = (
        '\n  <metadata type="description">\n    two  words\n  </metadata>\n'
        + _object(name="\n      Part A\n    ")
    )
    mesh = read(_file(tmp_path, _amf(body)))
    assert list(mesh.element_tags) == ["Part A"]
    assert mesh.global_attrs["description"] == "two  words"


def test_a_name_whose_edge_holds_a_line_break_is_written_with_a_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2]]))],
        element_tags={"walls\n": np.array([0])},
    )
    with pytest.warns(UserWarning, match="white space holding a line break"):
        back = _round_trip(poly, tmp_path)
    assert list(back.element_tags) == ["walls"]


@pytest.mark.parametrize(
    ("unit", "factor"), [("centimeter", 10.0), ("yard", 914.4), ("kilometer", 1e6)]
)
def test_a_length_amf_lacks_is_scaled_to_millimetres(
    tmp_path: Path, unit: str, factor: float
) -> None:
    poly = make_polydata(
        _SQUARE, [("triangle", np.array([[0, 1, 4]]))], global_attrs={"unit": unit}
    )
    with pytest.warns(UserWarning, match="scaled by"):
        back = _round_trip(poly, tmp_path)
    assert back.global_attrs["unit"] == "millimeter"
    np.testing.assert_allclose(back.vertices, _SQUARE[[0, 1, 4]] * factor)


def test_a_unit_that_is_no_length_keeps_the_coordinates(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("triangle", np.array([[0, 1, 4]]))], global_attrs={"unit": "cubit"}
    )
    with pytest.warns(UserWarning, match="kept as they are"):
        back = _round_trip(poly, tmp_path)
    np.testing.assert_array_equal(back.vertices, _SQUARE[[0, 1, 4]])


def test_a_root_that_is_not_amf_is_refused_before_the_rest(tmp_path: Path) -> None:
    # The rest is not even well-formed: the root alone decides.
    with pytest.raises(CodecError, match="the document root is <model>"):
        read(_file(tmp_path, "<model><a></b></model>"))


def test_colours_outside_0_1_are_clipped_with_a_warning(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 4]]))],
        element_attrs={"colors": np.array([[2.0, -1.0, 0.5, 1.0]])},
        vertex_attrs={"colors": np.tile([[1.5, 0.0, 0.0, 1.0]], (len(_SQUARE), 1))},
    )
    with pytest.warns(UserWarning, match="clipped") as caught:
        back = _round_trip(poly, tmp_path)
    assert sum("clipped" in str(w.message) for w in caught) == 2
    assert back.element_attrs["colors"].tolist() == [[1.0, 0.0, 0.5, 1.0]]
    assert back.vertex_attrs["colors"][:, 0].tolist() == [1.0] * 3


def test_write_warnings_point_at_the_caller(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE, [("triangle", np.array([[0, 1, 4]]))], global_attrs={"unit": "cubit"}
    )
    with pytest.warns(UserWarning, match="not an AMF unit") as caught:
        polyxios.write(poly, tmp_path / "out.amf")
    assert {w.filename for w in caught} == {__file__}
