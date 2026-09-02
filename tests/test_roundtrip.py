"""Cross-codec round-trip matrix.

Every writable codec is handed a canonical mesh, writes it, reads it back, and
is checked against a declared record of what it keeps and what it loses. The
declaration is the point of the file: a format that cannot hold element
attributes is not a failing test, it is a documented limit, and writing the
limit down is what turns an unnoticed regression into a failing assertion.

The table is exhaustive by construction - ``test_every_extension_is_accounted_for``
fails when a codec joins the registry without an entry here - so a new format
cannot land without saying what survives a round trip through it.

Warnings are errors for this suite (``filterwarnings = ["error"]``), so a codec
that warns has to declare the warning too, and one that gains an undeclared
warning fails.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, UnsupportedFormatError

# ---------------------------------------------------------------------------
# Canonical meshes
# ---------------------------------------------------------------------------
# Every vertex is referenced by an element, so a codec dropping orphan vertices
# cannot be mistaken for one corrupting the geometry. Attribute names are the
# same across the meshes, which is what lets one table describe them all.


def _vertex_attrs(n: int) -> dict[str, np.ndarray]:
    return {
        "scalar": np.arange(n, dtype=np.float64),
        "vector": np.arange(3 * n, dtype=np.float64).reshape(n, 3),
    }


def _element_attrs(n: int) -> dict[str, np.ndarray]:
    return {
        "eint": np.arange(10, 10 + n, dtype=np.int32),
        "efloat": np.arange(n, dtype=np.float64) + 0.5,
    }


def _mixed() -> PolyData:
    """Two triangles, a quad and a tetrahedron over seven vertices."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 1.5, 0.5],
        ]
    )
    return make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 2], [4, 5, 6]])),
            ("quad", np.array([[0, 1, 2, 3]])),
            ("tetra", np.array([[0, 1, 3, 4]])),
        ],
        vertex_attrs=_vertex_attrs(7),
        element_attrs=_element_attrs(4),
        vertex_tags={"vgroup": np.array([0, 1, 2], dtype=np.int32)},
        # Element 1 sits in both groups: a format storing one label per element
        # has to say so rather than lose the second silently.
        element_tags={
            "a": np.array([0, 1], dtype=np.int32),
            "b": np.array([1, 3], dtype=np.int32),
        },
        global_attrs={"gnum": 42},
    )


def _surface() -> PolyData:
    """A triangle and a quad; the shape a surface-only format can hold."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    return make_polydata(
        verts,
        [
            ("triangle", np.array([[0, 1, 4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
        ],
        vertex_attrs=_vertex_attrs(5),
        element_attrs=_element_attrs(2),
        vertex_tags={"vgroup": np.array([0, 1], dtype=np.int32)},
        element_tags={
            "a": np.array([0], dtype=np.int32),
            "b": np.array([0, 1], dtype=np.int32),
        },
        global_attrs={"gnum": 42},
    )


def _volume() -> PolyData:
    """Two tetrahedra; the shape a single-element-type volume format can hold."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return make_polydata(
        verts,
        [("tetra", np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))],
        vertex_attrs=_vertex_attrs(5),
        element_attrs=_element_attrs(2),
        vertex_tags={"vgroup": np.array([0, 1], dtype=np.int32)},
        element_tags={
            "a": np.array([0], dtype=np.int32),
            "b": np.array([0, 1], dtype=np.int32),
        },
        global_attrs={"gnum": 42},
    )


def _structured() -> PolyData:
    """A 2x2x2 grid of hexahedra: what .vti, .vts and .vtr are shaped for."""
    axis = np.arange(3.0)
    points = [[i, j, k] for k in axis for j in axis for i in axis]

    def node(i: int, j: int, k: int) -> int:
        return k * 9 + j * 3 + i

    conn = [
        [
            node(i, j, k),
            node(i + 1, j, k),
            node(i + 1, j + 1, k),
            node(i, j + 1, k),
            node(i, j, k + 1),
            node(i + 1, j, k + 1),
            node(i + 1, j + 1, k + 1),
            node(i, j + 1, k + 1),
        ]
        for k in range(2)
        for j in range(2)
        for i in range(2)
    ]
    return make_polydata(
        np.array(points),
        [("hexahedron", np.array(conn))],
        vertex_attrs=_vertex_attrs(27),
        element_attrs=_element_attrs(8),
        vertex_tags={"vgroup": np.array([0, 1], dtype=np.int32)},
        element_tags={
            "a": np.array([0], dtype=np.int32),
            "b": np.array([0, 1], dtype=np.int32),
        },
        global_attrs={"gnum": 42},
    )


def _points() -> PolyData:
    """Three vertex elements; a point cloud carries no faces to lose."""
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return make_polydata(
        verts,
        [("vertex", np.array([[0], [1], [2]]))],
        vertex_attrs=_vertex_attrs(3),
        global_attrs={"gnum": 42},
    )


CANONICAL = {
    "mixed": _mixed,
    "surface": _surface,
    "volume": _volume,
    "structured": _structured,
    "points": _points,
}


# ---------------------------------------------------------------------------
# What each format keeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cap:
    """One format's declared round-trip behaviour.

    The attribute, tag and global-attribute fields are the *exact* names read
    back, invented ones included, so a codec that starts keeping - or starts
    inventing - a field forces this table to be updated rather than passing
    unnoticed.

    Attributes
    ----------
    mesh
        Key into CANONICAL: the mesh this format is fed.
    n_vertices, n_elements
        Counts read back. None means unchanged from what was written.
    geometry
        Whether element_types and connectivity come back identical.
    coords
        Whether the vertex coordinates come back identical to ``rtol``.
    vertex_attrs, element_attrs, vertex_tags, element_tags, global_attrs
        The exact keys present after the round trip.
    attr_values
        Whether a surviving attribute comes back with its original shape and
        numbers. False points at a codec that keeps the name but not the data.
    tag_membership
        Whether a surviving tag group comes back naming the same elements.
    warns
        One regex per warning the round trip must emit, in order.
    rtol
        Relative tolerance for the coordinate comparison. ASCII writers
        default to a ``.10g`` field.
    note
        Why anything above is lossy. Required whenever something is lost.
    """

    mesh: str
    n_vertices: int | None = None
    n_elements: int | None = None
    geometry: bool = True
    coords: bool = True
    vertex_attrs: tuple[str, ...] = ()
    element_attrs: tuple[str, ...] = ()
    vertex_tags: tuple[str, ...] = ()
    element_tags: tuple[str, ...] = ()
    global_attrs: tuple[str, ...] = ()
    attr_values: bool = True
    tag_membership: bool = True
    warns: tuple[str, ...] = ()
    rtol: float = 1e-9
    note: str = ""


# Keyed by each codec's canonical EXTENSION. Aliases resolve to the same codec
# and are covered by tests/test_registry.py, so they are listed in _ALIASES
# rather than round-tripped twice.
CAPABILITIES: dict[str, Cap] = {
    ".avs": Cap(
        "mixed",
        element_attrs=("mat_id",),
        note="UCD stores one material id per element and no other data.",
    ),
    ".bdf": Cap(
        "mixed",
        element_attrs=("pid",),
        element_tags=("pid_1",),
        note="A bulk deck carries property ids, not named groups or fields.",
    ),
    ".ele": Cap(
        "volume",
        vertex_attrs=("attr_0",),
        element_attrs=("region",),
        vertex_tags=("boundary_1",),
        warns=(
            r"element attribute 'efloat' was written as the \.ele region",
            r"element attribute\(s\) \['eint'\] have no column",
            r"vertex attribute\(s\) \['vector'\] are not one scalar per vertex",
        ),
        note="TetGen holds one region attribute per element and one per node.",
    ),
    ".f3grid": Cap(
        "mixed",
        geometry=False,
        element_tags=("a", "b"),
        note="FLAC3D groups zones before faces, so element order changes.",
    ),
    ".inp": Cap(
        "mixed",
        vertex_tags=("vgroup",),
        element_tags=("a", "b"),
        note="Abaqus node and element sets carry the tags; attributes have no"
        " card in a mesh deck.",
    ),
    ".mdpa": Cap(
        "mixed",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        vertex_tags=("vgroup",),
        element_tags=("a", "b"),
        global_attrs=("gnum",),
    ),
    ".medit": Cap(
        "mixed",
        warns=(r"element tag group\(s\) \['a', 'b'\] are not named 'ref_<n>'",),
        note="A Medit record carries a reference number, not a name, so only"
        " groups already called 'ref_<n>' survive; the writer warns rather"
        " than numbering the rest itself. A 3-D file leaves global_attrs"
        " empty; only a 2-D one sets 'was_2d'.",
    ),
    ".mesh": Cap("mixed", note="MFEM stores geometry only."),
    ".meshb": Cap(
        "mixed",
        warns=(r"element tag group\(s\) \['a', 'b'\] are not named 'ref_<n>'",),
        note="A Medit record carries a reference number, not a name, so only"
        " groups already called 'ref_<n>' survive; the writer warns rather"
        " than numbering the rest itself.",
    ),
    ".msh": Cap(
        "mixed",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint", "phys_tag"),
        element_tags=("a",),
        warns=(r"element tag group\(s\) \['b'\] were not written",),
        note="A Gmsh element carries one physical tag, so overlaps cannot both"
        " survive; the writer warns rather than dropping silently.",
    ),
    ".obj": Cap(
        "surface",
        element_tags=("a", "b"),
        note="OBJ groups survive; attributes have no place in the format.",
    ),
    ".off": Cap("surface", note="OFF stores geometry only."),
    ".ply": Cap(
        "surface",
        vertex_attrs=("scalar", "vector_0", "vector_1", "vector_2"),
        element_attrs=("efloat", "eint"),
        note="PLY properties are scalar, so a vector splits into one per"
        " component and cannot be rebuilt on the way back.",
    ),
    ".splat": Cap(
        "points",
        n_elements=0,
        geometry=False,
        vertex_attrs=(
            "color_b",
            "color_g",
            "color_r",
            "opacity",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "scale_0",
            "scale_1",
            "scale_2",
        ),
        note="A splat file is a point cloud with a fixed per-point record: it"
        " has no cells and no room for a caller's own attributes.",
    ),
    ".stl": Cap(
        "surface",
        n_vertices=3,
        n_elements=1,
        geometry=False,
        coords=False,
        element_attrs=("normals",),
        note="STL is triangles only, stored per facet: the quad is dropped and"
        " the surviving triangle's vertices are not shared.",
    ),
    ".su2": Cap(
        "mixed",
        geometry=False,
        element_tags=("a", "unnamed"),
        warns=(
            r"element\(s\) belong to more than one tag",
            r"tag\(s\) \['b'\] also name 1 element\(s\) that are not boundary",
        ),
        note="SU2 splits volume cells from boundary markers, which reorders the"
        " mesh, and marks only elements one dimension below the volume.",
    ),
    ".tec": Cap(
        "volume",
        vertex_attrs=("scalar",),
        element_attrs=("efloat", "eint"),
        global_attrs=("tecplot_title", "tecplot_zone_title"),
        warns=(r"only named 1-D numeric vertex_attrs can be written",),
        note="A Tecplot FE zone is single-type: vectors have no column, while"
        " element attributes travel as VARLOCATION cell-centred variables. Every"
        " zone carries a title, so the caller's global_attrs are replaced by the"
        " file and zone names.",
    ),
    ".ugrid": Cap(
        "mixed",
        element_tags=("boundary_1", "boundary_2"),
        warns=(
            r"boundary face\(s\) belong to more than one tag",
            r"element tag\(s\) name 1 element\(s\) that are not boundary faces",
        ),
        note="A UGRID face carries one surface id, and only triangles and quads"
        " carry one at all.",
    ),
    ".vol": Cap(
        "mixed",
        element_tags=("a", "b", "bc_2"),
        tag_membership=False,
        warns=(r"element\(s\) belong to more than one tag",),
        note="A Netgen element carries one index per codimension: an element in"
        " two groups keeps the first, so 'b' comes back without it, and an"
        " untagged face picks up a generated 'bc_<n>' name.",
    ),
    ".vti": Cap(
        "structured",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        global_attrs=(
            "vti_extent",
            "vti_origin",
            "vti_spacing",
            "vti_whole_extent",
        ),
        attr_values=False,
        note="Grid metadata replaces the caller's global_attrs, tags are lost"
        " (see test_the_vtk_family_keeps_tags_it_has_room_for), and a vector"
        " attribute comes back flattened (see the xfail below).",
    ),
    ".vtk": Cap(
        "mixed",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        note="Legacy VTK keeps point and cell data; tags and field data do not"
        " survive (see the xfails below).",
    ),
    ".vtp": Cap(
        "surface",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        note="As .vtk, restricted to surface cells.",
    ),
    ".vtr": Cap(
        "structured",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        global_attrs=("vtr_extents", "vtr_whole_extent"),
        attr_values=False,
        note="As .vti, with rectilinear axis metadata.",
    ),
    ".vts": Cap(
        "structured",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        global_attrs=("vts_extent", "vts_whole_extent"),
        attr_values=False,
        note="As .vti, with curvilinear extent metadata.",
    ),
    ".vtu": Cap(
        "mixed",
        vertex_attrs=("scalar", "vector"),
        element_attrs=("efloat", "eint"),
        note="As .vtk; the missing <FieldData> is meshio #1546.",
    ),
    ".wkt": Cap(
        "surface",
        geometry=False,
        coords=False,
        element_attrs=("wkt_polygon_id", "wkt_ring"),
        global_attrs=("was_2d",),
        note="WKT has geometries, not elements: both faces come back as"
        " polygons and the ring bookkeeping replaces the caller's data. The"
        " canonical surface is flat, so it is written without a Z suffix and"
        " reads back flagged two-dimensional.",
    ),
    ".xml": Cap("volume", note="DOLFIN XML stores a single-type mesh only."),
}

# Same codec under another name; tests/test_registry.py covers the aliasing.
_ALIASES: frozenset[str] = frozenset({".nas", ".fem", ".node"})

# Registered so the error names the format, never to be written. Each is
# asserted below, so an entry cannot be parked here to escape the matrix.
_NOT_WRITABLE: dict[str, type[Exception]] = {
    ".pvti": NotImplementedError,
    ".pvtp": NotImplementedError,
    ".pvtr": NotImplementedError,
    ".pvts": NotImplementedError,
    ".pvtu": NotImplementedError,
    ".vtm": NotImplementedError,
    # Binary Tecplot: the ASCII writer refuses rather than mislabel its output.
    ".plt": CodecError,
    # Shared with Nastran and others: an output file has nothing to sniff.
    ".dat": UnsupportedFormatError,
}


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def _round_trip(poly: PolyData, path, cap: Cap) -> PolyData:
    """Write, read back, and check the warnings against the declaration.

    ``pytest.warns`` cannot express "these warnings and no others", which is
    the assertion the matrix needs: an undeclared warning is a change in codec
    behaviour and has to fail.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        polyxios.write(poly, path)
        back = polyxios.read(path)

    messages = [str(record.message) for record in caught]
    assert len(messages) == len(cap.warns), (
        f"expected {len(cap.warns)} warning(s), got {messages}"
    )
    for message, pattern in zip(messages, cap.warns, strict=True):
        assert re.search(pattern, message), f"{pattern!r} does not match {message!r}"
    return back


@pytest.mark.parametrize("ext", sorted(CAPABILITIES))
def test_round_trip_matches_the_declared_capabilities(tmp_path, ext: str) -> None:
    cap = CAPABILITIES[ext]
    poly = CANONICAL[cap.mesh]()
    back = _round_trip(poly, tmp_path / f"mesh{ext}", cap)

    expected_verts = (
        poly.vertices.shape[0] if cap.n_vertices is None else cap.n_vertices
    )
    expected_elems = (
        len(poly.element_types) if cap.n_elements is None else cap.n_elements
    )
    assert back.vertices.shape == (expected_verts, 3)
    assert len(back.element_types) == expected_elems

    if cap.coords:
        np.testing.assert_allclose(back.vertices, poly.vertices, rtol=cap.rtol)
    if cap.geometry:
        np.testing.assert_array_equal(back.element_types, poly.element_types)
        np.testing.assert_array_equal(back.connectivity, poly.connectivity)
        np.testing.assert_array_equal(back.offsets, poly.offsets)

    assert tuple(sorted(back.vertex_attrs)) == tuple(sorted(cap.vertex_attrs))
    assert tuple(sorted(back.element_attrs)) == tuple(sorted(cap.element_attrs))
    assert tuple(sorted(back.vertex_tags)) == tuple(sorted(cap.vertex_tags))
    assert tuple(sorted(back.element_tags)) == tuple(sorted(cap.element_tags))
    assert tuple(sorted(back.global_attrs)) == tuple(sorted(cap.global_attrs))


@pytest.mark.parametrize("ext", sorted(CAPABILITIES))
def test_surviving_attribute_values_are_unchanged(tmp_path, ext: str) -> None:
    """Keeping a name is not enough; the numbers have to come back too."""
    cap = CAPABILITIES[ext]
    if not cap.geometry:
        pytest.skip(
            f"{ext} reorders or reshapes the mesh, so an index-wise comparison"
            " means nothing; the per-codec tests carry the value checks"
        )
    poly = CANONICAL[cap.mesh]()
    back = _round_trip(poly, tmp_path / f"mesh{ext}", cap)

    if cap.attr_values:
        for name in cap.vertex_attrs:
            if name in poly.vertex_attrs:
                np.testing.assert_allclose(
                    back.vertex_attrs[name], poly.vertex_attrs[name], rtol=cap.rtol
                )
        for name in cap.element_attrs:
            if name in poly.element_attrs:
                np.testing.assert_allclose(
                    back.element_attrs[name], poly.element_attrs[name], rtol=cap.rtol
                )
    if cap.tag_membership:
        for name in cap.element_tags:
            if name in poly.element_tags:
                np.testing.assert_array_equal(
                    np.sort(back.element_tags[name]), np.sort(poly.element_tags[name])
                )


def test_every_extension_is_accounted_for() -> None:
    """A new codec cannot land without declaring its round-trip behaviour."""
    declared = frozenset(CAPABILITIES) | _ALIASES | frozenset(_NOT_WRITABLE)
    assert declared == frozenset(polyxios.supported_extensions())


def test_every_lossy_entry_explains_itself() -> None:
    """A loss without a reason is indistinguishable from a bug."""
    for ext, cap in CAPABILITIES.items():
        poly = CANONICAL[cap.mesh]()
        lossy = (
            cap.n_vertices is not None
            or cap.n_elements is not None
            or not cap.geometry
            or not cap.coords
            or set(cap.vertex_attrs) != set(poly.vertex_attrs)
            or set(cap.element_attrs) != set(poly.element_attrs)
            or set(cap.vertex_tags) != set(poly.vertex_tags)
            or set(cap.element_tags) != set(poly.element_tags)
            or set(cap.global_attrs) != set(poly.global_attrs)
        )
        assert not lossy or cap.note, f"{ext} loses data without saying why"


@pytest.mark.parametrize("ext", sorted(_NOT_WRITABLE))
def test_an_unwritable_extension_refuses_to_be_written(tmp_path, ext: str) -> None:
    """The formats excluded from the matrix are excluded for a stated reason."""
    with pytest.raises(_NOT_WRITABLE[ext]):
        polyxios.write(_surface(), tmp_path / f"mesh{ext}")


# ---------------------------------------------------------------------------
# What the matrix turned up. Each is xfail(strict=True) so the fix fails the
# suite rather than passing unnoticed, and each carries the parity-plan item
# it belongs to.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="P2.5 / meshio #1546: no <FieldData>")
@pytest.mark.parametrize("ext", [".vtu", ".vtk", ".vtp"])
def test_the_vtk_family_keeps_global_attrs(tmp_path, ext: str) -> None:
    poly = _surface()
    path = tmp_path / f"mesh{ext}"
    polyxios.write(poly, path)
    assert polyxios.read(path).global_attrs["gnum"] == 42


@pytest.mark.xfail(strict=True, reason="P2.6: tags have no VTK array yet")
@pytest.mark.parametrize("ext", [".vtu", ".vtk", ".vtp"])
def test_the_vtk_family_keeps_tags_it_has_room_for(tmp_path, ext: str) -> None:
    """PointData and CellData can hold one int array per tag group."""
    poly = _surface()
    path = tmp_path / f"mesh{ext}"
    polyxios.write(poly, path)
    back = polyxios.read(path)
    assert set(back.element_tags) == {"a", "b"}
    assert set(back.vertex_tags) == {"vgroup"}


@pytest.mark.parametrize("ext", [".vti", ".vts", ".vtr"])
def test_a_structured_reader_keeps_a_vector_attribute_2d(tmp_path, ext: str) -> None:
    """A component count declared on the way out is honoured on the way back,
    the way the .vtu and .vtk of the same XML family already did."""
    poly = _structured()
    path = tmp_path / f"mesh{ext}"
    polyxios.write(poly, path)
    assert polyxios.read(path).vertex_attrs["vector"].shape == (27, 3)


@pytest.mark.parametrize("ext", [".vti", ".vts", ".vtr"])
def test_a_structured_writer_rejects_an_unstructured_mesh(tmp_path, ext: str) -> None:
    """.vts wrote a file its own reader could not parse; .vti and .vtr invented
    a grid. Either way the caller learned nothing at write time."""
    with pytest.raises(CodecError):
        polyxios.write(_mixed(), tmp_path / f"mesh{ext}")


@pytest.mark.parametrize("ext", [".vti", ".vtr"])
def test_a_lattice_writer_rejects_a_grid_in_the_wrong_order(tmp_path, ext: str) -> None:
    """The same points, z varying fastest: neither format writes its points,
    so every attribute would move to the point mirrored through the diagonal,
    and the file would not say so."""
    poly = _structured()
    mirrored = replace(poly, vertices=np.ascontiguousarray(poly.vertices[:, ::-1]))
    with pytest.raises(CodecError):
        polyxios.write(mirrored, tmp_path / f"mesh{ext}")


def test_a_structured_grid_holds_points_no_lattice_could(tmp_path) -> None:
    """A StructuredGrid writes its points, and holding the ones a lattice
    cannot - a warped block, a cylindrical shell, an aerofoil O-grid - is why
    the format exists next to .vti. Its extent is the cells' shape, not the
    coordinates', so a warp does not put the mesh out of its reach."""
    poly = _structured()
    warped = replace(
        poly,
        vertices=poly.vertices
        + np.column_stack(
            [
                0.3 * np.sin(poly.vertices[:, 2]),
                0.2 * poly.vertices[:, 0] ** 2,
                np.zeros(len(poly.vertices)),
            ]
        ),
    )
    path = tmp_path / "warped.vts"
    polyxios.write(warped, path)
    back = polyxios.read(path)
    np.testing.assert_allclose(back.vertices, warped.vertices)
    np.testing.assert_array_equal(back.connectivity, warped.connectivity)
    np.testing.assert_array_equal(back.element_types, warped.element_types)
    np.testing.assert_allclose(
        back.vertex_attrs["scalar"], warped.vertex_attrs["scalar"]
    )

    # The same mesh is out of .vti's and .vtr's reach, and they say so.
    for ext in (".vti", ".vtr"):
        with pytest.raises(CodecError):
            polyxios.write(warped, tmp_path / f"warped{ext}")


@pytest.mark.parametrize("ext", [".vti", ".vts", ".vtr"])
def test_a_structured_writer_rejects_cells_the_grid_would_not_read_back(
    tmp_path, ext: str
) -> None:
    """None of the three writes its connectivity, so cells that are not the
    grid's own are dropped on the way back in and their data with them. The
    points here are a grid; the two tetrahedra over them are not its cells."""
    poly = _structured()
    tets = replace(
        poly,
        connectivity=np.array([0, 1, 3, 9, 1, 2, 4, 10]),
        offsets=np.array([0, 4, 8]),
        element_types=np.full(2, ELEMENT_TYPES["tetra"], dtype=np.uint8),
        element_attrs={"c": np.array([1.0, 2.0])},
        element_tags={},
    )
    with pytest.raises(CodecError):
        polyxios.write(tets, tmp_path / f"mesh{ext}")


@pytest.mark.xfail(strict=True, reason="a tetrahedron is written as a quad")
@pytest.mark.parametrize("ext", [".obj", ".ply", ".vtp"])
def test_a_surface_writer_does_not_turn_a_tetrahedron_into_a_quad(
    tmp_path, ext: str
) -> None:
    """A surface format has no cell for a tet. Dropping it with a warning is
    honest; writing its four nodes as a quadrilateral is corrupt data."""
    from polyxios._element_types import ELEMENT_TYPES

    poly = make_polydata(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    path = tmp_path / f"mesh{ext}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polyxios.write(poly, path)
        back = polyxios.read(path)
    assert ELEMENT_TYPES["quad"] not in set(back.element_types.tolist())
