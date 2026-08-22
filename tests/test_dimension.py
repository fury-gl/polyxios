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
    with pytest.raises(ValueError):
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
