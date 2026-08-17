from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polyxios import make_polydata, read as api_read, write as api_write
from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.codecs._tecplot import read, write
from polyxios.exceptions import CodecError, UnsupportedFormatError

_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _tet_mesh():
    return make_polydata(_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])


def _tri_mesh():
    return make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])


def _write_tec(tmp_path: Path, text: str, name: str = "m.tec") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_roundtrip_tetra(tmp_path: Path) -> None:
    poly = _tet_mesh()
    tmp = tmp_path / "t.tec"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_array_equal(poly2.offsets, poly.offsets)


def test_roundtrip_triangles(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "t.tec"
    write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 2
    np.testing.assert_allclose(poly2.vertices, poly.vertices)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


@pytest.mark.parametrize(
    ("kind", "conn"),
    [
        ("quad", np.array([[0, 1, 2, 3]])),
        ("hexahedron", np.arange(8).reshape(1, 8)),
    ],
)
def test_roundtrip_quad_and_hex(tmp_path: Path, kind: str, conn: np.ndarray) -> None:
    verts = np.arange(24, dtype=np.float64).reshape(8, 3)
    poly = make_polydata(verts, [(kind, conn)])
    tmp = tmp_path / "t.tec"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)
    np.testing.assert_allclose(poly2.vertices, verts)


def test_registered_for_tec_extension(tmp_path: Path) -> None:
    poly = _tri_mesh()
    tmp = tmp_path / "t.tec"
    api_write(poly, tmp)
    poly2 = api_read(tmp)
    np.testing.assert_array_equal(poly2.connectivity, poly.connectivity)


def test_dat_stays_unclaimed_and_reads_through_the_fmt_override(
    tmp_path: Path,
) -> None:
    """'.dat' is shared with LS-DYNA and Nastran, so it resolves to no codec."""
    poly = _tri_mesh()
    path = tmp_path / "mesh.dat"
    api_write(poly, path, fmt=".tec")
    with pytest.raises(UnsupportedFormatError, match=r"\.dat"):
        api_read(path)
    np.testing.assert_array_equal(
        api_read(path, fmt=".tec").connectivity, poly.connectivity
    )


def test_file_has_zone_header(tmp_path: Path) -> None:
    tmp = tmp_path / "t.tec"
    write(_tet_mesh(), tmp)
    text = tmp.read_text(encoding="utf-8")
    assert "ZONE" in text
    assert "ET=TETRAHEDRON" in text


def test_missing_zone_raises(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, 'TITLE = "bad"\nVARIABLES = "X" "Y" "Z"\n')
    with pytest.raises(CodecError, match="missing ZONE"):
        read(path)


def test_unsupported_et_raises(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'TITLE="t"\nVARIABLES="X","Y","Z"\nZONE N=3, E=1, F=FEPOINT, ET=LINESEG\n',
    )
    with pytest.raises(CodecError, match="unsupported element type"):
        read(path)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        'ZONE T="z", ET=TRIANGLE, N=3, E=1, F=FEPOINT',
        'ZONE T="z", E=1, N=3, F=FEPOINT, ET=TRIANGLE',
        "zone n=3, e=1, f=fepoint, et=triangle",
        'ZONE T="z", NODES=3, ELEMENTS=1, DATAPACKING=POINT, ZONETYPE=FETRIANGLE',
        'ZONE T="z"\n N=3, E=1\n F=FEPOINT, ET=TRIANGLE',
    ],
    ids=["et-first", "e-before-n", "lowercase", "modern", "continuation"],
)
def test_header_key_order_and_spellings(tmp_path: Path, header: str) -> None:
    path = _write_tec(
        tmp_path,
        f'VARIABLES="X","Y","Z"\n{header}\n0 0 0\n1 0 0\n0 1 0\n1 2 3\n',
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_quoted_title_is_not_mined_for_counts(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES="X","Y","Z"\n'
        'ZONE T="grid N=99 E=99", N=3, E=1, F=FEPOINT, ET=TRIANGLE\n'
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    assert read(path).vertices.shape == (3, 3)


def test_missing_counts_raise(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, "ZONE F=FEPOINT, ET=TRIANGLE\n0 0 0\n")
    with pytest.raises(CodecError, match="missing N= or E="):
        read(path)


def test_missing_element_type_raises(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, "ZONE N=3, E=1, F=FEPOINT\n0 0 0\n")
    with pytest.raises(CodecError, match="missing ET="):
        read(path)


def test_unsupported_packing_rejected(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, "ZONE N=3, E=1, F=WHAT, ET=TRIANGLE\n0 0 0\n")
    with pytest.raises(CodecError, match="unsupported data packing"):
        read(path)


def test_header_key_per_line(tmp_path: Path) -> None:
    """Tecplot writes one key per line; ZONETYPE must not read as a new ZONE."""
    path = _write_tec(
        tmp_path,
        'TITLE = "grid"\n'
        'VARIABLES = "X" "Y" "Z"\n'
        'ZONE T = "zone 1"\n'
        " Nodes = 3\n"
        " Elements = 1\n"
        " DATAPACKING = POINT\n"
        " ZONETYPE = FETRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_both_element_type_spellings_agreeing(tmp_path: Path) -> None:
    """Older writers emit ET= and ZONETYPE= at once; agreeing ones are fine."""
    path = _write_tec(
        tmp_path,
        'VARIABLES="X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE, ZONETYPE=FETRIANGLE,"
        " DATAPACKING=POINT\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    np.testing.assert_array_equal(read(path).connectivity, [0, 1, 2])


def test_disagreeing_element_type_spellings_rejected(tmp_path: Path) -> None:
    """Picking one at random would read the connectivity at the wrong width."""
    path = _write_tec(
        tmp_path,
        'VARIABLES="X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE, ZONETYPE=FEQUADRILATERAL\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="different element types"):
        read(path)


def test_disagreeing_packing_spellings_rejected(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES="X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, DATAPACKING=BLOCK, ET=TRIANGLE\n"
        "0 1 0\n0 0 1\n0 0 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="different data packings"):
        read(path)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ("N=3, NODES=9, E=1", "different node counts"),
        ("N=3, E=1, ELEMENTS=5", "different element counts"),
    ],
    ids=["nodes", "elements"],
)
def test_disagreeing_count_spellings_rejected(
    tmp_path: Path, header: str, match: str
) -> None:
    path = _write_tec(
        tmp_path,
        f'VARIABLES="X","Y","Z"\nZONE {header}, F=FEPOINT, ET=TRIANGLE\n'
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match=match):
        read(path)


@pytest.mark.parametrize(
    ("feature", "match"),
    [
        ("VARLOCATION=([4]=CELLCENTERED)", "CELLCENTERED"),
        ("VARSHARELIST=([1-3]=1)", "VARSHARELIST"),
        ("PASSIVEVARLIST=[4]", "PASSIVEVARLIST"),
        ("CONNECTIVITYSHAREZONE=1", "CONNECTIVITYSHAREZONE"),
    ],
    ids=["cellcentered", "varshare", "passive", "connshare"],
)
def test_zone_features_that_move_data_are_rejected(
    tmp_path: Path, feature: str, match: str
) -> None:
    """A cell-centred run holds E values, not N — reading it as nodal is wrong."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z","P"\n'
        f"ZONE N=3, E=3, F=FEBLOCK, ET=TRIANGLE, {feature}\n"
        "0 1 0\n0 0 1\n0 0 0\n7 8 9\n1 2 3\n1 2 3\n1 2 3\n",
    )
    with pytest.raises(CodecError, match=match):
        read(path)


def test_nodal_varlocation_still_reads(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z","P"\n'
        "ZONE N=3, E=1, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-4]=NODAL)\n"
        "0 1 0\n0 0 1\n0 0 0\n7 8 9\n1 2 3\n",
    )
    np.testing.assert_allclose(read(path).vertex_attrs["P"], [7, 8, 9])


# ---------------------------------------------------------------------------
# BLOCK packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("packing", ["F=FEBLOCK", "DATAPACKING=BLOCK"])
def test_block_packing_reads(tmp_path: Path, packing: str) -> None:
    """Under BLOCK packing each variable is one run over every node."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z"\n'
        f"ZONE N=4, E=2, {packing}, ET=TRIANGLE\n"
        "0 1 0 1\n"  # X for all four nodes
        "0 0 1 1\n"  # Y
        "0 0 0 0\n"  # Z
        "1 2 3\n2 4 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(
        poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    )
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])


def test_block_packing_run_wraps_across_lines(tmp_path: Path) -> None:
    """A variable's run may be broken over as many lines as the writer likes."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z"\n'
        "ZONE N=4, E=2, F=FEBLOCK, ET=TRIANGLE\n"
        "0 1\n0 1\n0 0 1 1\n0 0\n0 0\n1 2 3 2 4 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(
        poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    )
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2, 1, 3, 2])


def test_block_packing_keeps_solution_variables(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z", "P"\n'
        "ZONE N=3, E=1, F=FEBLOCK, ET=TRIANGLE\n"
        "0 1 0\n0 0 1\n0 0 0\n7 8 9\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertex_attrs["P"], [7, 8, 9])


def test_block_packing_truncation_raises(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z"\n'
        "ZONE N=4, E=2, F=FEBLOCK, ET=TRIANGLE\n"
        "0 1 0 1\n0 0 1 1\n0 0 0 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="truncated"):
        read(path)


def test_block_packing_recovers_variable_count_without_a_declaration(
    tmp_path: Path,
) -> None:
    path = _write_tec(
        tmp_path,
        "ZONE N=3, E=1, F=FEBLOCK, ET=TRIANGLE\n0 1 0\n0 0 1\n0 0 0\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_block_packing_rejects_an_unrecoverable_variable_count(
    tmp_path: Path,
) -> None:
    path = _write_tec(
        tmp_path,
        "ZONE N=3, E=1, F=FEBLOCK, ET=TRIANGLE\n0 1 0\n0 0 1\n0 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="not a multiple of N"):
        read(path)


def test_negative_count_rejected(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, "ZONE N=-3, E=1, F=FEPOINT, ET=TRIANGLE\n")
    with pytest.raises(CodecError, match="negative node count"):
        read(path)


def test_absurd_count_rejected_before_allocating(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path, "ZONE N=999999999999, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\n1 2 3\n"
    )
    with pytest.raises(CodecError, match="safety cap"):
        read(path)


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def test_truncated_file_raises(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, "ZONE N=5, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\n1 0 0\n")
    with pytest.raises(CodecError, match="truncated"):
        read(path)


def test_malformed_node_line_raises(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        "ZONE N=2, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\nfoo bar baz\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="malformed node line"):
        read(path)


def test_short_element_line_raises(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\n1 0 0\n0 1 0\n1 2\n",
    )
    with pytest.raises(CodecError, match="expected 3 for ET=TRIANGLE"):
        read(path)


@pytest.mark.parametrize("bad", ["0 1 2", "1 2 9"])
def test_out_of_range_node_reference_raises(tmp_path: Path, bad: str) -> None:
    path = _write_tec(
        tmp_path,
        f"ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\n1 0 0\n0 1 0\n{bad}\n",
    )
    with pytest.raises(CodecError, match="outside 1..3"):
        read(path)


def test_record_inside_zone_data_raises(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        "ZONE N=4, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n"
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 1\n1 0 1\n0 1 1\n1 2 3\n",
    )
    # The second ZONE is seen, and warned about, before its data is read.
    with pytest.warns(UserWarning, match="more than one zone"):
        with pytest.raises(CodecError, match="new record starts inside"):
            read(path)


def test_extra_zone_warns_and_reads_the_first(tmp_path: Path) -> None:
    zone = "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\n1 0 0\n0 1 0\n1 2 3\n"
    path = _write_tec(tmp_path, 'VARIABLES="X","Y","Z"\n' + zone + zone)
    with pytest.warns(UserWarning, match="more than one zone"):
        poly = read(path)
    assert len(poly.element_types) == 1


def test_comments_and_blank_lines_ignored(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES="X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "# a comment\n\n0 0 0\n1 0 0\n\n0 1 0\n# another\n1 2 3\n",
    )
    assert read(path).vertices.shape == (3, 3)


def test_lazy_warns(tmp_path: Path) -> None:
    tmp = tmp_path / "t.tec"
    write(_tri_mesh(), tmp)
    with pytest.warns(UserWarning, match="lazy=True is not supported"):
        read(tmp, lazy=True)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def test_two_dimensional_zone_pads_z(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0\n1 0\n0 1\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_two_dimensional_zone_keeps_solution_variable(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "P"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 7\n1 0 8\n0 1 9\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices[:, 2], 0.0)
    np.testing.assert_allclose(poly.vertex_attrs["P"], [7, 8, 9])


def test_solution_variables_roundtrip(tmp_path: Path) -> None:
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"P": np.array([1.5, 2.5, 3.5, 4.5])},
    )
    tmp = tmp_path / "t.tec"
    write(poly, tmp)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertex_attrs["P"], [1.5, 2.5, 3.5, 4.5])


def test_node_line_missing_declared_variable_raises(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z", "P"\n'
        "ZONE N=1, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 1 1\n",
    )
    with pytest.raises(CodecError, match="expected 4"):
        read(path)


def test_point_packing_recovers_variables_from_a_node_record(tmp_path: Path) -> None:
    """An undeclared solution variable is kept rather than silently dropped."""
    path = _write_tec(
        tmp_path,
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0 7\n1 0 0 8\n0 1 0 9\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(poly.vertex_attrs["var4"], [7, 8, 9])


def test_point_record_wrapped_across_lines(tmp_path: Path) -> None:
    """Writers wrap a long node record; the record, not the line, is the unit."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z", "A", "B"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 4\n1 0 0\n2 5\n0 1 0\n3 6\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(poly.vertex_attrs["A"], [1, 2, 3])
    np.testing.assert_allclose(poly.vertex_attrs["B"], [4, 5, 6])
    np.testing.assert_array_equal(poly.connectivity, [0, 1, 2])


def test_element_line_with_extra_nodes_rejected(tmp_path: Path) -> None:
    """A fourth node under ET=TRIANGLE is a quad, not a triangle to truncate."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3 3\n",
    )
    with pytest.raises(CodecError, match="expected 3 for ET=TRIANGLE"):
        read(path)


def test_error_names_the_file_line(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES="X","Y","Z"\n\n# a comment\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\nfoo bar baz\n0 1 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="malformed node line 6"):
        read(path)


def test_duplicate_variable_names_keep_both_columns(tmp_path: Path) -> None:
    """A repeated name would otherwise collapse in the vertex_attrs dict."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z","P","P"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0 1 10\n1 0 0 2 20\n0 1 0 3 30\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertex_attrs["P"], [1, 2, 3])
    np.testing.assert_allclose(poly.vertex_attrs["P_2"], [10, 20, 30])


def test_second_variable_not_named_y_warns(tmp_path: Path) -> None:
    """The names say the second variable is not a coordinate; say so out loud."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Density"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0\n1 5\n0 6\n1 2 3\n",
    )
    with pytest.warns(UserWarning, match="not 'Y'"):
        poly = read(path)
    assert poly.vertices.shape == (3, 3)


def test_multiline_variables_block(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y"\n"Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    assert not poly.vertex_attrs


def test_quoted_variable_name_with_spaces_is_one_variable(tmp_path: Path) -> None:
    """Names carrying units are common; splitting one on its space miscounts."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z", "Pressure (Pa)"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0 7\n1 0 0 8\n0 1 0 9\n1 2 3\n",
    )
    poly = read(path)
    np.testing.assert_allclose(poly.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(poly.vertex_attrs["Pressure (Pa)"], [7, 8, 9])


def test_single_variable_rejected(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X"\nZONE N=1, E=1, F=FEPOINT, ET=TRIANGLE\n0\n1 1 1\n',
    )
    with pytest.raises(CodecError, match="at least 2 coordinate variables"):
        read(path)


def test_byte_order_mark_does_not_hide_the_header(tmp_path: Path) -> None:
    path = tmp_path / "bom.tec"
    path.write_text(
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
        encoding="utf-8-sig",
    )
    assert read(path).vertices.shape == (3, 3)


def test_empty_zone_reads_as_an_empty_mesh(tmp_path: Path) -> None:
    path = _write_tec(tmp_path, "ZONE N=0, E=0, F=FEPOINT, ET=TRIANGLE\n")
    poly = read(path)
    assert poly.vertices.shape == (0, 3)
    assert poly.connectivity.shape == (0,)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_rejects_unsupported_only_mesh(tmp_path: Path) -> None:
    poly = make_polydata(_VERTS, [("line", np.array([[0, 1]]))])
    with pytest.raises(CodecError, match="no supported element type"):
        write(poly, tmp_path / "t.tec")


def test_write_warns_on_dropped_element_types(tmp_path: Path) -> None:
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]])), ("tetra", np.array([[0, 1, 2, 3]]))],
    )
    tmp = tmp_path / "t.tec"
    with pytest.warns(UserWarning, match="skipped"):
        write(poly, tmp)
    poly2 = read(tmp)
    assert len(poly2.element_types) == 1


def test_write_warns_on_unknown_options(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_tri_mesh(), tmp_path / "t.tec", bogus=1)


def test_write_skips_non_scalar_vertex_attrs(tmp_path: Path) -> None:
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"rgb": np.zeros((4, 3), dtype=np.float64)},
    )
    tmp = tmp_path / "t.tec"
    with pytest.warns(UserWarning, match="1-D numeric vertex_attrs"):
        write(poly, tmp)
    assert not read(tmp).vertex_attrs


def test_write_skips_a_nameless_vertex_attr(tmp_path: Path) -> None:
    """A nameless variable reads back as no variable, dropping its column."""
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"  ": np.array([1.0, 2.0, 3.0, 4.0])},
    )
    tmp = tmp_path / "t.tec"
    with pytest.warns(UserWarning, match="only named 1-D numeric"):
        write(poly, tmp)
    assert not read(tmp).vertex_attrs


def test_attr_named_like_a_coordinate_keeps_its_name(tmp_path: Path) -> None:
    """Only solution variables are renamed, so 'X' stays 'X' on the way back."""
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"X": np.array([1.0, 2.0, 3.0, 4.0])},
    )
    tmp = tmp_path / "t.tec"
    write(poly, tmp)
    np.testing.assert_allclose(read(tmp).vertex_attrs["X"], [1.0, 2.0, 3.0, 4.0])


def test_write_quotes_a_variable_name_the_reader_can_take_back(
    tmp_path: Path,
) -> None:
    """An unescaped quote would split one variable into two on the way back."""
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={'a "quoted" name': np.array([1.0, 2.0, 3.0, 4.0])},
    )
    tmp = tmp_path / "t.tec"
    with pytest.warns(UserWarning, match="double quote"):
        write(poly, tmp)
    np.testing.assert_allclose(
        read(tmp).vertex_attrs["a 'quoted' name"], [1.0, 2.0, 3.0, 4.0]
    )


@pytest.mark.parametrize("bad", [7, -1])
def test_write_rejects_out_of_range_connectivity(tmp_path: Path, bad: int) -> None:
    """A dangling reference would be written as a node the zone never declares."""
    poly = PolyData(
        vertices=_VERTS[:3],
        connectivity=np.array([0, 1, bad], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="outside 0..2"):
        write(poly, tmp_path / "t.tec")


def test_write_rejects_a_node_count_that_contradicts_the_type(tmp_path: Path) -> None:
    poly = PolyData(
        vertices=_VERTS,
        connectivity=np.array([0, 1, 2, 3], dtype=np.int32),
        offsets=np.array([0, 4], dtype=np.int32),
        element_types=np.array([ELEMENT_TYPES["triangle"]], dtype=np.uint8),
    )
    with pytest.raises(CodecError, match="expected 3"):
        write(poly, tmp_path / "t.tec")


def test_write_folds_a_line_break_in_a_variable_name(tmp_path: Path) -> None:
    """A raw newline would split VARIABLES in two and could forge a ZONE line."""
    poly = make_polydata(
        _VERTS,
        [("triangle", np.array([[0, 1, 2]]))],
        vertex_attrs={"a\nZONE N=9": np.array([1.0, 2.0, 3.0, 4.0])},
    )
    tmp = tmp_path / "t.tec"
    with pytest.warns(UserWarning, match="line break"):
        write(poly, tmp)
    header = tmp.read_text(encoding="utf-8").splitlines()[1]
    assert header == 'VARIABLES = "X", "Y", "Z", "a ZONE N=9"'
    np.testing.assert_allclose(
        read(tmp).vertex_attrs["a ZONE N=9"], [1.0, 2.0, 3.0, 4.0]
    )


# ---------------------------------------------------------------------------
# Header and data hygiene
# ---------------------------------------------------------------------------


def test_comment_inside_the_variables_list_is_not_a_variable(tmp_path: Path) -> None:
    """'#' is a comment everywhere else in the file; the list is no exception."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y",\n# the coordinates\n"Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    assert not poly.vertex_attrs


def test_blank_line_inside_the_variables_list_keeps_the_rest(tmp_path: Path) -> None:
    """Stopping at the blank line would drop 'T' and the column it stands for."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z",\n\n"T"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0 1\n1 0 0 2\n0 1 0 3\n1 2 3\n",
    )
    np.testing.assert_allclose(read(path).vertex_attrs["T"], [1, 2, 3])


def test_variables_list_stops_at_the_next_keyed_record(tmp_path: Path) -> None:
    """A 'KEY =' line is the next record, not four more variable names."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X", "Y", "Z"\nFILETYPE = FULL\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    poly = read(path)
    assert poly.vertices.shape == (3, 3)
    assert not poly.vertex_attrs


def test_cellcentered_in_a_title_is_not_a_cellcentered_variable(
    tmp_path: Path,
) -> None:
    """Only the VARLOCATION clause moves data off the nodes, not the word."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z"\n'
        'ZONE T="cellcentered study", N=3, E=1, F=FEPOINT, ET=TRIANGLE\n'
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    assert read(path).vertices.shape == (3, 3)


def test_one_field_declared_twice_with_two_values_rejected(tmp_path: Path) -> None:
    """Taking the last would read three nodes as ninety-nine, or the reverse."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z"\n'
        "ZONE N=3, N=99, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    with pytest.raises(CodecError, match="declares N= twice"):
        read(path)


def test_one_field_declared_twice_agreeing_still_reads(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE, N=3\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    assert read(path).vertices.shape == (3, 3)


def test_point_records_packed_several_to_a_line_recover_their_width(
    tmp_path: Path,
) -> None:
    """The first line is two records, so its token count is not the variable count."""
    path = _write_tec(
        tmp_path,
        "ZONE N=4, E=1, F=FEPOINT, ET=TRIANGLE\n0 0 0 1 0 0\n0 1 0 1 1 0\n1 2 3\n",
    )
    np.testing.assert_allclose(
        read(path).vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    )


def test_node_record_wider_than_declared_warns(tmp_path: Path) -> None:
    """A dropped column is data leaving the file; it does not leave quietly."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0 9\n1 0 0 8\n0 1 0 7\n1 2 3\n",
    )
    with pytest.warns(UserWarning, match="more than the 3 declared values"):
        assert read(path).vertices.shape == (3, 3)


def test_block_values_past_the_declared_counts_warn(tmp_path: Path) -> None:
    path = _write_tec(
        tmp_path,
        'VARIABLES = "X","Y","Z"\n'
        "ZONE N=3, E=1, DATAPACKING=BLOCK, ZONETYPE=FETRIANGLE\n"
        "0 1 0\n0 0 1\n0 0 0\n1 2 3\n99 99\n",
    )
    with pytest.warns(UserWarning, match="past the declared"):
        assert read(path).vertices.shape == (3, 3)


def test_first_variable_not_named_x_warns(tmp_path: Path) -> None:
    """The fallback reads three unnamed columns as coordinates; say so out loud."""
    path = _write_tec(
        tmp_path,
        'VARIABLES = "A", "B", "C"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n",
    )
    with pytest.warns(UserWarning, match="not 'X'"):
        assert read(path).vertices.shape == (3, 3)


def test_non_utf8_file_stays_inside_a_codec_error(tmp_path: Path) -> None:
    """A latin-1 title must not leak UnicodeDecodeError past the codec."""
    path = tmp_path / "m.tec"
    path.write_bytes(
        'TITLE = "grid at 20\xb0C"\nVARIABLES = "X","Y","Z"\n'
        "ZONE N=3, E=1, F=FEPOINT, ET=TRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n".encode("latin-1")
    )
    assert read(path).vertices.shape == (3, 3)


# ---------------------------------------------------------------------------
# meshio interoperability
# ---------------------------------------------------------------------------

_MESHIO_CASES = {
    "triangle": (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64),
        np.array([[0, 1, 2], [1, 3, 2]]),
    ),
    "quad": (
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64),
        np.array([[0, 1, 2, 3]]),
    ),
    "tetra": (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 2, 3]]),
    ),
    "hexahedron": (
        np.arange(24, dtype=np.float64).reshape(8, 3),
        np.arange(8).reshape(1, 8),
    ),
}


@pytest.mark.parametrize("kind", sorted(_MESHIO_CASES))
def test_reads_what_meshio_writes(tmp_path: Path, kind: str) -> None:
    """meshio's Tecplot writer emits BLOCK packing, the common real-world one."""
    meshio = pytest.importorskip("meshio")
    verts, cells = _MESHIO_CASES[kind]
    out = tmp_path / f"{kind}.dat"
    meshio.Mesh(verts, [(kind, cells)]).write(out, file_format="tecplot")

    poly = read(out)
    np.testing.assert_allclose(poly.vertices, verts)
    np.testing.assert_array_equal(poly.connectivity.reshape(cells.shape), cells)


@pytest.mark.parametrize("kind", sorted(_MESHIO_CASES))
def test_meshio_reads_what_polyxios_writes(tmp_path: Path, kind: str) -> None:
    meshio = pytest.importorskip("meshio")
    verts, cells = _MESHIO_CASES[kind]
    out = tmp_path / f"{kind}.dat"
    write(make_polydata(verts, [(kind, cells)]), out)

    mesh = meshio.read(out, file_format="tecplot")
    np.testing.assert_allclose(mesh.points[:, :3], verts)
    assert mesh.cells[0].type == kind
    np.testing.assert_array_equal(mesh.cells[0].data, cells)


def test_meshio_point_data_survives_the_read(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    verts, cells = _MESHIO_CASES["triangle"]
    out = tmp_path / "attrs.dat"
    meshio.Mesh(
        verts,
        [("triangle", cells)],
        point_data={"P": np.array([1.0, 2.0, 3.0, 4.0])},
    ).write(out, file_format="tecplot")

    np.testing.assert_allclose(read(out).vertex_attrs["P"], [1.0, 2.0, 3.0, 4.0])
