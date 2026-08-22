"""The 2-D policy, held to across every format that can spell two dimensions.

``polyxios/_dimension.py`` states the rule: a mesh always carries three
coordinate columns, a file that declared two is padded with ``z=0`` and sets
``global_attrs["was_2d"]``, and a writer that can spell two columns puts them
back. The table below is one 2-D and one 3-D fixture per such format, so a
codec cannot drift off the policy without a failing test naming it.
"""

from __future__ import annotations

import dataclasses
import struct
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._dimension import (
    WAS_2D_KEY,
    mark_2d,
    output_dimension,
    pad_to_3d,
    was_2d,
)
from polyxios.exceptions import CodecError

# ---------------------------------------------------------------------------
# The helpers themselves
# ---------------------------------------------------------------------------


def test_mark_2d_speaks_only_for_a_two_dimensional_file() -> None:
    """The key's absence means 'not known to be 2-D', so a 3-D file sets none."""
    assert mark_2d(2) == {WAS_2D_KEY: True}
    assert mark_2d(3) == {}


@pytest.mark.parametrize("dim", [2, 3])
def test_pad_to_3d_widens_without_touching_the_columns_it_keeps(dim: int) -> None:
    block = np.arange(3 * dim, dtype=np.float64).reshape(3, dim)

    out = pad_to_3d(block, dim)

    assert out.shape == (3, 3)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out[:, :dim], block)
    np.testing.assert_array_equal(out[:, dim:], 0.0)


def test_pad_to_3d_reads_only_the_columns_the_dimension_names() -> None:
    """A record array carrying a reference beside the coordinates is handed
    over whole, so the columns past the dimension have to be ignored."""
    block = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 98.0]])

    np.testing.assert_array_equal(pad_to_3d(block, 2)[:, 2], 0.0)


@pytest.mark.parametrize(
    ("values", "dim"),
    [
        (np.zeros((2, 3)), 4),
        (np.zeros((2, 1)), 2),
        (np.zeros(3), 3),
    ],
)
def test_pad_to_3d_refuses_a_block_it_cannot_widen(values, dim: int) -> None:
    """A codec error, not a ValueError: the block came off a file, and that is
    what every other malformed-input path in a codec raises."""
    with pytest.raises(CodecError):
        pad_to_3d(values, dim)


def _flat(**global_attrs) -> polyxios.PolyData:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return make_polydata(
        verts, [("triangle", np.array([[0, 1, 2]]))], global_attrs=global_attrs
    )


def _lift(poly: polyxios.PolyData) -> polyxios.PolyData:
    return dataclasses.replace(poly, vertices=poly.vertices + np.array([0.0, 0.0, 1.0]))


def test_a_flagged_flat_mesh_is_written_in_two_dimensions() -> None:
    poly = _flat(**{WAS_2D_KEY: True})

    assert was_2d(poly)
    assert output_dimension(poly, fmt=".x") == 2
    assert output_dimension(poly, fmt=".x", flat_default=2) == 2


def test_an_unflagged_flat_mesh_takes_the_formats_own_default() -> None:
    poly = _flat()

    assert not was_2d(poly)
    assert output_dimension(poly, fmt=".x") == 3
    assert output_dimension(poly, fmt=".x", flat_default=2) == 2


def test_coordinates_outrank_the_flag_and_the_writer_says_so() -> None:
    """A z that reached the mesh after the read is data; the flag is what gives."""
    lifted = _lift(_flat(**{WAS_2D_KEY: True}))

    with pytest.warns(UserWarning, match=r"\.x: .*now carry a third coordinate"):
        assert output_dimension(lifted, fmt=".x") == 3


def test_a_lifted_unflagged_mesh_is_three_dimensional_without_a_word() -> None:
    lifted = _lift(_flat())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert output_dimension(lifted, fmt=".x", flat_default=2) == 3


def test_a_codec_may_supply_its_own_flatness_test() -> None:
    """WKT reads only the vertices an element reaches, so it decides flatness
    itself and hands the answer in rather than having it recomputed."""
    assert output_dimension(_flat(), fmt=".x", flat=False, flat_default=2) == 3


def test_a_two_column_mesh_is_taken_at_its_word() -> None:
    """Not what a PolyData holds, but a release of the meshb reader handed one
    back: the columns it has are the whole dimension, flag or no flag."""
    narrow = dataclasses.replace(_flat(), vertices=np.zeros((3, 2)))

    assert output_dimension(narrow, fmt=".x") == 2
    assert output_dimension(dataclasses.replace(narrow, global_attrs={}), fmt=".x") == 2


@pytest.mark.parametrize("shape", [(3,), (3, 1), (2, 3, 3)])
def test_a_block_that_is_not_coordinates_is_named_rather_than_indexed(shape) -> None:
    """The z column is read to test flatness; a mesh that has none used to
    leave an IndexError from inside the helper."""
    poly = dataclasses.replace(_flat(), vertices=np.zeros(shape))

    with pytest.raises(CodecError, match=r"\(n, 2\) or wider"):
        output_dimension(poly, fmt=".x")


# ---------------------------------------------------------------------------
# The formats
# ---------------------------------------------------------------------------

# The corners of one triangle, spelled the way each format spells them.
_CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def _rows(dim: int, *, sep: str = " ", index: bool = False, trailing: str = "") -> str:
    """Return one coordinate row per corner, ``dim`` columns wide."""
    out = []
    for i, (x, y) in enumerate(_CORNERS):
        cells = [f"{v:g}" for v in ((x, y) if dim == 2 else (x, y, 0.0))]
        if index:
            cells.insert(0, str(i + 1))
        out.append(sep.join(cells) + trailing)
    return "\n".join(out)


def _dolfin_rows(dim: int) -> str:
    return "\n".join(
        f'      <vertex index="{i}" x="{x}" y="{y}"'
        + ("" if dim == 2 else ' z="0"')
        + "/>"
        for i, (x, y) in enumerate(_CORNERS)
    )


def _meshb_bytes(dim: int) -> bytes:
    """A one-triangle ``.meshb`` of the given Dimension."""
    out = struct.pack("<iiii", 1, 2, 3, dim)
    out += struct.pack("<ii", 4, len(_CORNERS))
    for x, y in _CORNERS:
        coords = (x, y) if dim == 2 else (x, y, 0.0)
        out += struct.pack(f"<{dim}di", *coords, 0)
    out += struct.pack("<ii", 6, 1) + struct.pack("<4i", 1, 2, 3, 0)
    return out + struct.pack("<i", 54)


_MEDIT = """MeshVersionFormatted 2
Dimension {dim}
Vertices
3
{verts}
Triangles
1
1 2 3 0
End
"""

_SU2 = """NDIME= {dim}
NPOIN= 3
{verts}
NELEM= 1
5 0 1 2 0
NMARK= 0
"""

_VOL = """mesh3d

dimension
{dim}

geomtype
0

surfaceelements
1
1 1 0 0 3 1 2 3

points
3
{verts}

endmesh
"""

_MFEM = """MFEM mesh v1.0

dimension
2

elements
1
1 2 0 1 2

boundary
0

vertices
3
{dim}
{verts}
"""

_DOLFIN = """<?xml version="1.0"?>
<dolfin>
  <mesh celltype="triangle" dim="{dim}">
    <vertices size="3">
{verts}
    </vertices>
    <cells size="1">
      <triangle index="0" v0="0" v1="1" v2="2"/>
    </cells>
  </mesh>
</dolfin>
"""

_TEC = """TITLE = "t"
VARIABLES = {vars}
ZONE T="z", N=3, E=1, F=FEPOINT, ET=TRIANGLE
{verts}
1 2 3
"""

_INP = """*Heading
** a plane, or not
*Node
{verts}
*Element, type=CPS3
1, 1, 2, 3
"""

_NODE = """3 {dim} 0 0
{verts}
"""
# One tetrahedron would need a fourth node; the pair exists so the reader does
# not warn about a .node file standing alone.
_ELE = "0 4 0\n"


@dataclasses.dataclass(frozen=True)
class Fixture:
    """One format's pair of fixtures and what its writer can spell.

    Attributes
    ----------
    ext
        The extension the fixture is written under and read back through.
    flat, solid
        The 2-D and the 3-D file, as text or as bytes.
    fmt
        What ``write`` is told, when the extension alone names another codec.
    sibling
        A second file the reader needs beside the first, as
        ``(suffix, text)`` - TetGen's ``.node``/``.ele`` pair.
    restores
        Whether the writer can spell two columns. False for a format whose
        only spelling is three: it still records the flag on the way in, so a
        mesh read from it and written elsewhere stays two-dimensional.
    """

    ext: str
    flat: str | bytes
    solid: str | bytes
    fmt: str | None = None
    sibling: tuple[str, str] | None = None
    restores: bool = True


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        ".medit",
        _MEDIT.format(dim=2, verts=_rows(2, trailing=" 0")),
        _MEDIT.format(dim=3, verts=_rows(3, trailing=" 0")),
        fmt=".medit",
    ),
    Fixture(".meshb", _meshb_bytes(2), _meshb_bytes(3)),
    Fixture(
        ".su2", _SU2.format(dim=2, verts=_rows(2)), _SU2.format(dim=3, verts=_rows(3))
    ),
    Fixture(
        ".vol",
        _VOL.format(dim=2, verts=_rows(2)),
        _VOL.format(dim=3, verts=_rows(3)),
        restores=False,
    ),
    Fixture(
        ".mesh",
        _MFEM.format(dim=2, verts=_rows(2)),
        _MFEM.format(dim=3, verts=_rows(3)),
    ),
    Fixture(
        ".xml",
        _DOLFIN.format(dim=2, verts=_dolfin_rows(2)),
        _DOLFIN.format(dim=3, verts=_dolfin_rows(3)),
    ),
    Fixture(
        ".tec",
        _TEC.format(vars='"X", "Y"', verts=_rows(2)),
        _TEC.format(vars='"X", "Y", "Z"', verts=_rows(3)),
    ),
    Fixture(
        ".inp",
        _INP.format(verts=_rows(2, sep=", ", index=True)),
        _INP.format(verts=_rows(3, sep=", ", index=True)),
    ),
    Fixture(
        ".node",
        _NODE.format(dim=2, verts=_rows(2, index=True)),
        _NODE.format(dim=3, verts=_rows(3, index=True)),
        sibling=(".ele", _ELE),
    ),
    Fixture(
        ".wkt",
        "POLYGON ((0 0, 1 0, 0 1, 0 0))",
        "POLYGON Z ((0 0 0, 1 0 0, 0 1 0, 0 0 0))",
    ),
)

_IDS = tuple(f.ext for f in FIXTURES)
_RESTORING = tuple(f for f in FIXTURES if f.restores)
_RESTORING_IDS = tuple(f.ext for f in _RESTORING)


def _source(tmp_path, fixture: Fixture, body: str | bytes):
    """Lay the fixture down on disk and return the path to read."""
    path = tmp_path / f"mesh{fixture.ext}"
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    if fixture.sibling is not None:
        suffix, text = fixture.sibling
        path.with_suffix(suffix).write_text(text, encoding="utf-8")
    return path


def _write(poly, path, fixture: Fixture) -> None:
    if fixture.fmt is None:
        polyxios.write(poly, path)
    else:
        polyxios.write(poly, path, fmt=fixture.fmt)


@pytest.mark.parametrize("fixture", FIXTURES, ids=_IDS)
def test_a_two_dimensional_file_reads_with_three_columns(tmp_path, fixture) -> None:
    poly = polyxios.read(_source(tmp_path, fixture, fixture.flat))

    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.vertices[:, 2], 0.0)
    assert poly.global_attrs.get(WAS_2D_KEY) is True


@pytest.mark.parametrize("fixture", FIXTURES, ids=_IDS)
def test_a_three_dimensional_file_is_not_flagged(tmp_path, fixture) -> None:
    poly = polyxios.read(_source(tmp_path, fixture, fixture.solid))

    assert poly.vertices.shape == (3, 3)
    assert WAS_2D_KEY not in poly.global_attrs


@pytest.mark.parametrize("fixture", _RESTORING, ids=_RESTORING_IDS)
def test_a_two_dimensional_file_is_written_back_as_one(tmp_path, fixture) -> None:
    poly = polyxios.read(_source(tmp_path, fixture, fixture.flat))
    out = tmp_path / f"again{fixture.ext}"

    _write(poly, out, fixture)

    back = polyxios.read(out)
    assert back.global_attrs.get(WAS_2D_KEY) is True
    np.testing.assert_allclose(back.vertices, poly.vertices)


@pytest.mark.parametrize("fixture", _RESTORING, ids=_RESTORING_IDS)
def test_a_lifted_mesh_is_written_in_three_dimensions(tmp_path, fixture) -> None:
    """The flag says 2-D and the coordinates say otherwise; the data wins, and
    the writer says so rather than flattening the mesh in silence."""
    lifted = _lift(polyxios.read(_source(tmp_path, fixture, fixture.flat)))
    out = tmp_path / f"lifted{fixture.ext}"

    # Recorded rather than asserted with pytest.warns: SU2 has a second
    # warning of its own to add here, and the assertion is that the policy's
    # warning is among them, not that it is the only one.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write(lifted, out, fixture)

    assert any("now carry a third coordinate" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]
    np.testing.assert_allclose(polyxios.read(out).vertices, lifted.vertices)


def test_the_flag_travels_between_formats(tmp_path) -> None:
    """The key says something about the mesh, not about the file it came from,
    so a plane read as Netgen - which has no 2-D spelling of its own to write
    back - still lands as an ``NDIME= 2`` SU2 case."""
    src = tmp_path / "mesh.vol"
    src.write_text(_VOL.format(dim=2, verts=_rows(2)), encoding="utf-8")
    out = tmp_path / "mesh.su2"

    with pytest.warns(UserWarning, match="name no boundary element"):
        # The .vol names its one face, and SU2 marks only the elements below
        # its volume cells; the tag is beside the point here.
        polyxios.write(polyxios.read(src), out)

    assert out.read_text(encoding="utf-8").splitlines()[0] == "NDIME= 2"


def test_a_format_that_cannot_spell_two_dimensions_keeps_the_flag(tmp_path) -> None:
    """Netgen writes ``mesh3d`` whatever it is handed, so the round trip
    through it loses the flag - which is the documented limit, not a bug."""
    src = tmp_path / "mesh.vol"
    src.write_text(_VOL.format(dim=2, verts=_rows(2)), encoding="utf-8")
    out = tmp_path / "again.vol"

    polyxios.write(polyxios.read(src), out)

    assert WAS_2D_KEY not in polyxios.read(out).global_attrs


# ---------------------------------------------------------------------------
# Where two columns constrain the rest of the file
# ---------------------------------------------------------------------------


def _flagged(pairs, kind: str, **kw) -> polyxios.PolyData:
    """One flat element of ``kind``, flagged as read from a 2-D file."""
    verts = np.array([[x, y, 0.0] for x, y in pairs])
    return make_polydata(
        verts,
        [(kind, np.array([list(range(len(pairs)))]))],
        global_attrs={WAS_2D_KEY: True},
        **kw,
    )


_SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


@pytest.mark.parametrize(
    ("kind", "card"),
    [
        ("line", "T2D2"),
        ("triangle", "CPS3"),
        ("quad", "CPS4"),
    ],
)
def test_a_two_dimensional_deck_uses_planar_element_cards(
    tmp_path, kind: str, card: str
) -> None:
    """Abaqus takes a node's dimensionality from the element referencing it, so
    two coordinates per node under an S3 shell is a deck no solver loads."""
    n = {"line": 2, "triangle": 3, "quad": 4}[kind]
    out = tmp_path / "mesh.inp"

    polyxios.write(_flagged(_SQUARE[:n], kind), out)

    text = out.read_text(encoding="utf-8")
    assert f"*Element, type={card}" in text
    assert "1, 0, 0" in text and "1, 0, 0, 0" not in text


def test_a_type_with_no_planar_card_keeps_the_deck_three_dimensional(
    tmp_path,
) -> None:
    """A flat tetrahedron is still a solid, and C3D4 holds its nodes in space:
    the node cards keep their third coordinate rather than the deck breaking."""
    out = tmp_path / "mesh.inp"
    corners = (*_SQUARE[:3], (0.0, 0.0))

    polyxios.write(_flagged(corners, "tetra"), out)

    text = out.read_text(encoding="utf-8")
    assert "*Element, type=C3D4" in text
    assert "1, 0, 0, 0" in text


def test_an_override_naming_a_three_dimensional_card_wins(tmp_path) -> None:
    """``element_type=`` is the caller saying what the deck is for; a shell card
    asked for by name keeps its three coordinates per node."""
    out = tmp_path / "mesh.inp"
    poly = _flagged(_SQUARE[:3], "triangle")

    polyxios.write(poly, out, element_type={"triangle": "S3"})
    assert "1, 0, 0, 0" in out.read_text(encoding="utf-8")

    # A planar card with a modifier suffix is still planar.
    polyxios.write(poly, out, element_type={"triangle": "CPE3H"})
    text = out.read_text(encoding="utf-8")
    assert "*Element, type=CPE3H" in text
    assert "1, 0, 0" in text and "1, 0, 0, 0" not in text


_SOLID = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.3, 0.3))


@pytest.mark.parametrize(
    ("ext", "fmt", "needle"),
    [
        (".tec", None, 'VARIABLES = "X", "Y", "Z"'),
        (".mesh", None, "dimension\n3"),
        (".node", None, "4 3 0 0"),
        (".medit", ".medit", "Dimension 3"),
        (".xml", None, 'celltype="tetrahedron" dim="3"'),
    ],
    ids=[".tec", ".mesh", ".node", ".medit", ".xml"],
)
def test_a_flat_solid_keeps_its_third_column(tmp_path, ext, fmt, needle) -> None:
    """A flat tetrahedron is still a solid. Each of these formats declares its
    coordinate count in one place and its node count per element in another,
    so a two-column file of tetrahedra is one no reader loads."""
    out = tmp_path / f"mesh{ext}"
    poly = _flagged(_SOLID, "tetra")

    if fmt is None:
        polyxios.write(poly, out)
    else:
        polyxios.write(poly, out, fmt=fmt)

    assert needle in out.read_text(encoding="utf-8")


def test_a_flat_solid_keeps_its_third_column_in_meshb(tmp_path) -> None:
    """The binary spelling of the same rule: a Tetrahedra section under
    ``Dimension 2`` is a file no reader loads."""
    out = tmp_path / "mesh.meshb"

    polyxios.write(_flagged(_SOLID, "tetra"), out)

    assert struct.unpack_from("<i", out.read_bytes(), 12)[0] == 3


def test_an_explicit_dolfin_dimension_is_the_callers_own_word(tmp_path) -> None:
    """The guard above rewrites what was inferred, not what was asked for."""
    out = tmp_path / "mesh.xml"

    polyxios.write(_flagged(_SOLID, "tetra"), out, dim=2)

    assert 'celltype="tetrahedron" dim="2"' in out.read_text(encoding="utf-8")


def test_an_override_a_planar_card_cannot_hold_is_named(tmp_path) -> None:
    """The card table for a 2-D deck holds only the planar types, and an
    override is what can name one for a type outside it. The mismatch is the
    caller's, and is reported as such rather than as a lookup that missed."""
    verts = np.array(
        [
            (x, y, 0.0)
            for x, y in (
                *_SQUARE,
                (0.5, 0.0),
                (1.0, 0.5),
                (0.5, 1.0),
                (0.0, 0.5),
                (0.5, 0.5),
            )
        ]
    )
    poly = make_polydata(
        verts,
        [("biquadratic_quad", np.arange(9).reshape(1, 9))],
        global_attrs={WAS_2D_KEY: True},
    )

    with pytest.raises(CodecError, match="which is a 8-node 'quadratic_quad' card"):
        polyxios.write(
            poly, tmp_path / "mesh.inp", element_type={"biquadratic_quad": "CPS8"}
        )


def test_a_mesh_with_no_vertex_is_not_called_flat(tmp_path) -> None:
    """A DOLFIN file may leave ``dim`` off, and then the vertices are what says
    how many dimensions it has. An empty mesh has no extent to read one off,
    so it is not flagged - which is what its writer already assumed."""
    src = tmp_path / "empty.xml"
    src.write_text(
        '<?xml version="1.0"?>\n<dolfin>\n  <mesh celltype="triangle">\n'
        '    <vertices size="0"></vertices>\n'
        '    <cells size="0"></cells>\n  </mesh>\n</dolfin>\n',
        encoding="utf-8",
    )

    assert WAS_2D_KEY not in polyxios.read(src).global_attrs


def test_a_tecplot_variable_named_z_keeps_the_zone_three_dimensional(
    tmp_path,
) -> None:
    """Tecplot names the coordinates by position, so "X", "Y", "Z" where the
    third is a solution variable reads back as three coordinates."""
    out = tmp_path / "mesh.tec"
    poly = _flagged(_SQUARE[:3], "triangle", vertex_attrs={"Z": np.arange(3.0)})

    polyxios.write(poly, out)

    back = polyxios.read(out)
    np.testing.assert_array_equal(back.vertices[:, 2], 0.0)
    np.testing.assert_array_equal(back.vertex_attrs["Z"], np.arange(3.0))


def test_a_tecplot_variable_named_otherwise_leaves_the_zone_flat(tmp_path) -> None:
    out = tmp_path / "mesh.tec"
    poly = _flagged(_SQUARE[:3], "triangle", vertex_attrs={"T": np.arange(3.0)})

    polyxios.write(poly, out)

    assert 'VARIABLES = "X", "Y", "T"' in out.read_text(encoding="utf-8")


def test_dolfin_trusts_its_own_declaration_over_the_vertices(tmp_path) -> None:
    """``<mesh dim=>`` is DOLFIN's geometric dimension. An interval mesh
    declares 1 and is not two-dimensional, whatever its vertices omit."""
    src = tmp_path / "mesh.xml"
    src.write_text(
        '<?xml version="1.0"?>\n<dolfin>\n'
        '  <mesh celltype="interval" dim="1">\n'
        '    <vertices size="2">\n'
        '      <vertex index="0" x="0.0" y="0.0"/>\n'
        '      <vertex index="1" x="1.0" y="0.0"/>\n'
        "    </vertices>\n"
        '    <cells size="1">\n'
        '      <interval index="0" v0="0" v1="1"/>\n'
        "    </cells>\n"
        "  </mesh>\n</dolfin>\n",
        encoding="utf-8",
    )

    assert WAS_2D_KEY not in polyxios.read(src).global_attrs


def test_a_two_dimensional_nurbs_mesh_is_flagged_like_any_other(tmp_path) -> None:
    """Control points are coordinates, and the nodes field spells their vdim
    the same way a standard mesh's vertices section does."""
    src = tmp_path / "mesh.mesh"
    src.write_text(
        "MFEM NURBS mesh v1.0\n\ndimension\n2\n\nelements\n0\n\nvertices\n3\n\n"
        "nodes\nFiniteElementSpace\nVDim: 2\n0 0 1 0 0 1\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="NURBS"):
        poly = polyxios.read(src)

    assert poly.vertices.shape == (3, 3)
    assert poly.global_attrs.get(WAS_2D_KEY) is True
