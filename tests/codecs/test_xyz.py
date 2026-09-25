from __future__ import annotations

import io
import warnings

import numpy as np
import pytest

import polyxios
from polyxios._types import PolyData, make_polydata
from polyxios.exceptions import CodecError, LazyReadError

_PTX = """\
2
1
0 0 0
1 0 0
0 1 0
0 0 1
1 0 0 0
0 1 0 0
0 0 1 0
10 0 0 1
1 2 3 0.5 255 0 0
0 0 0 0.5 0 0 0
1
1
0 0 0
1 0 0
0 1 0
0 0 1
0 -1 0 0
1 0 0 0
0 0 1 0
0 0 0 1
1 0 0 0.7 0 255 0
"""


def _read(text: str, ext: str = ".xyz", **opts) -> PolyData:
    return polyxios.read(io.StringIO(text), fmt=ext, **opts)


def _write(poly: PolyData, ext: str = ".xyz", **opts) -> str:
    """Write to a nameless buffer, which needs the variant said out loud."""
    buf = io.StringIO()
    polyxios.write(poly, buf, fmt=ext, variant=ext.lstrip("."), **opts)
    return buf.getvalue()


def _cloud(**attrs) -> PolyData:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    return make_polydata(verts, [("vertex", np.arange(3)[:, None])], vertex_attrs=attrs)


# ---------------------------------------------------------------------------
# Columns are named by count and kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("1 2 3", {}),
        ("1 2 3 0.5", {"intensity": [0.5]}),
        ("1 2 3 255 0 128", {"colors": [[1.0, 0.0, 128 / 255]]}),
        ("1 2 3 0.1 0.2 0.3", {"normals": [[0.1, 0.2, 0.3]]}),
        (
            "1 2 3 0.5 255 0 128",
            {"intensity": [0.5], "colors": [[1.0, 0.0, 128 / 255]]},
        ),
        ("1 2 3 0.5 0.1 0.2 0.3", {"intensity": [0.5], "normals": [[0.1, 0.2, 0.3]]}),
        (
            "1 2 3 255 0 128 0.1 0.2 0.3",
            {"colors": [[1.0, 0.0, 128 / 255]], "normals": [[0.1, 0.2, 0.3]]},
        ),
        (
            "1 2 3 0.1 0.2 0.3 255 0 128",
            {"normals": [[0.1, 0.2, 0.3]], "colors": [[1.0, 0.0, 128 / 255]]},
        ),
        (
            "1 2 3 0.5 255 0 128 0.1 0.2 0.3",
            {
                "intensity": [0.5],
                "colors": [[1.0, 0.0, 128 / 255]],
                "normals": [[0.1, 0.2, 0.3]],
            },
        ),
        ("1 2 3 4 5", {"extra": [[4.0, 5.0]]}),
        ("1 2 3 4 5 6 7 8", {"extra": [[4.0, 5.0, 6.0, 7.0, 8.0]]}),
        ("1 2 3 0.1 0.2 0.3 0.4 0.5 0.6", {"extra": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}),
    ],
)
def test_columns_past_the_coordinates_are_named_by_count_and_kind(
    row, expected
) -> None:
    back = _read(row + "\n")
    np.testing.assert_array_equal(back.vertices, [[1, 2, 3]])
    assert back.element_types.size == 0
    assert set(back.vertex_attrs) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(back.vertex_attrs[key], value)


def test_comments_blank_lines_and_a_column_header_are_skipped() -> None:
    text = "# scanner export\n\nX Y Z R G B\n0 0 0 255 0 0\n// mid\n1 1 1 0 0 255\n\n"
    back = _read(text)
    assert back.vertices.shape == (2, 3)
    np.testing.assert_allclose(back.vertex_attrs["colors"][1], [0, 0, 1])


def test_a_row_with_a_trailing_comment_is_refused_not_taken_for_a_header() -> None:
    """A row that starts with a number is data, whatever follows it; only
    a first line whose first token is not a number is a column header."""
    with pytest.raises(CodecError, match="not a number"):
        _read("1 2 3 # first\n4 5 6\n")
    with pytest.raises(CodecError, match="not a number"):
        _read("X Y Z\n1 2 3 # first\n4 5 6\n")


def test_three_columns_of_only_zeros_and_ones_are_normals_not_colours() -> None:
    """A flat scan carries ``0 0 1`` on every point; a colour whose every
    byte is 0 or 1 is all but black, so the tie goes to the normal."""
    back = _read("0 0 0 0 0 1\n1 0 0 0 0 1\n")
    assert set(back.vertex_attrs) == {"normals"}
    np.testing.assert_array_equal(back.vertex_attrs["normals"], [[0, 0, 1]] * 2)
    back = _read("0 0 0 0.5 0 0 1\n1 0 0 0.5 1 0 0\n")
    assert set(back.vertex_attrs) == {"intensity", "normals"}
    back = _read("0 0 0 0 0 2\n1 0 0 0 0 1\n")
    assert set(back.vertex_attrs) == {"colors"}


def test_commas_and_tabs_are_not_separators_but_tabs_are_whitespace() -> None:
    assert _read("0\t0\t0\n1\t0\t0\n").vertices.shape == (2, 3)
    with pytest.raises(CodecError, match="not a number"):
        _read("0,0,0\n")


def test_an_empty_file_is_an_empty_cloud() -> None:
    assert _read("").vertices.shape == (0, 3)
    assert _read("# nothing\n").vertices.shape == (0, 3)


def test_a_pts_count_line_is_honoured_and_checked() -> None:
    back = _read("2\n0 0 0 1 255 0 0\n1 0 0 2 0 255 0\n", ".pts")
    assert back.vertices.shape == (2, 3)
    np.testing.assert_array_equal(back.vertex_attrs["intensity"], [1, 2])
    with pytest.raises(CodecError, match="count line says 3 points, the file holds 2"):
        _read("3\n0 0 0\n1 0 0\n", ".pts")


def test_a_count_line_over_too_few_columns_names_the_columns() -> None:
    """``1\\n2\\n3`` is a one-column file whatever its first line looks
    like; the fault to name is the missing columns, not a count."""
    with pytest.raises(CodecError, match="1 columns; a point needs x, y and z"):
        _read("1\n2\n3\n", ".pts")
    with pytest.raises(CodecError, match="2 columns; a point needs x, y and z"):
        _read("3\n1 2\n3 4\n", ".pts")


def test_a_pts_of_no_points_is_a_count_line_alone_and_reads_back_empty() -> None:
    assert _write(make_polydata(np.empty((0, 3)), []), ".pts") == "0\n"
    assert _read("0\n", ".pts").vertices.shape == (0, 3)
    assert _read("0\nX Y Z\n", ".pts").vertices.shape == (0, 3)


def test_a_byte_order_mark_before_the_first_row_is_skipped() -> None:
    """Windows exporters put a UTF-8 BOM first; it is not a column."""
    assert _read("\ufeff1 2 3\n").vertices.tolist() == [[1, 2, 3]]
    buf = io.BytesIO("\ufeff4 5 6\n".encode("utf-8"))
    assert polyxios.read(buf, fmt=".xyz").vertices.tolist() == [[4, 5, 6]]


def test_a_count_no_file_can_hold_is_refused() -> None:
    with pytest.raises(CodecError, match=str(2**62)):
        _read(f"{2**62}\n0 0 0\n", ".pts")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("0 0\n", "a point needs x, y and z"),
        ("0 0 0\n1 1\n", "rows are not all 3 columns wide: row 2 has 2"),
        ("0 0 0\n1 a 1\n", "not a number"),
        ("1 2 3\n4 5 6 7 8 9\n", "rows are not all 3 columns wide: row 2 has 6"),
        ("1 2 3\n\n4 5\n6 7 8 9\n", "rows are not all 3 columns wide: row 2 has 2"),
        ("1 2 3 4\n5 6 7 8\n9 10\n11 12", "not all 4 columns wide: row 3 has 2"),
    ],
)
def test_a_malformed_table_names_what_is_wrong(text, message) -> None:
    """A total that divides by the first row's width proves nothing: one
    short row and one long one add up too, so every row is counted."""
    with pytest.raises(CodecError, match=message):
        _read(text)
    with pytest.raises(CodecError, match=message):
        _read(f"{text.count(chr(10)) or 1}\n{text}", ".pts")


def test_lazy_is_refused_for_text() -> None:
    with pytest.raises(LazyReadError, match="ASCII point clouds"):
        _read("0 0 0\n", lazy=True)


# ---------------------------------------------------------------------------
# PTX scans
# ---------------------------------------------------------------------------


def test_ptx_scans_are_transformed_tagged_and_their_invalid_points_dropped() -> None:
    with pytest.warns(UserWarning, match="1 invalid PTX points"):
        back = _read(_PTX, ".ptx")
    # Row-vector convention: the last row of the matrix is the translation.
    np.testing.assert_allclose(back.vertices, [[11, 2, 3], [0, -1, 0]])
    np.testing.assert_allclose(back.vertex_attrs["intensity"], [0.5, 0.7])
    np.testing.assert_allclose(back.vertex_attrs["colors"], [[1, 0, 0], [0, 1, 0]])
    assert {k: v.tolist() for k, v in back.vertex_tags.items()} == {
        "scan_0": [0],
        "scan_1": [1],
    }
    assert back.global_attrs["ptx_transforms"].shape == (2, 4, 4)
    np.testing.assert_array_equal(back.global_attrs["ptx_dimensions"], [[1, 2], [1, 1]])


def test_a_single_ptx_scan_carries_no_tags() -> None:
    one = _PTX.split("1\n1\n0 0 0\n1 0 0")[0].replace(
        "0 0 0 0.5 0 0 0\n", "4 5 6 0.1 0 0 0\n"
    )
    back = _read(one, ".ptx")
    assert back.vertex_tags == {}
    assert back.vertices.shape == (2, 3)


def test_a_ptx_without_colours_reads_intensity_alone() -> None:
    text = "1\n1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n1 2 3 0.25\n"
    back = _read(text, ".ptx")
    assert set(back.vertex_attrs) == {"intensity"}


def test_ptx_scans_that_disagree_on_colour_keep_none_with_a_warning() -> None:
    text = _PTX.replace("1 0 0 0.7 0 255 0\n", "1 0 0 0.7\n")
    with pytest.warns(UserWarning, match="only some PTX scans carry colours"):
        with pytest.warns(UserWarning, match="invalid PTX points"):
            back = _read(text, ".ptx")
    assert "colors" not in back.vertex_attrs


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("1 0 0 0.7 0 255 0\n", ""), "declares 1 points, the file holds fewer"),
        (("0 0 0 0.5 0 0 0\n", ""), "PTX scan declares 2 points, holds 1"),
        (("2\n1\n0 0 0", "9\n1\n0 0 0"), "PTX scan declares 9 points, holds 2"),
        (("1 0 0 0.7 0 255 0\n", "1 0 0 0.7 0\n"), "4 or 7 columns, found 5"),
        (("1 0 0 0.7 0 255 0\n", "1 0 0 0.7 0 255 0\nstray\n"), "not a scan header"),
        (("2\n1\n0 0 0", "-2\n-1\n0 0 0"), "PTX scan of -2 x -1 points"),
    ],
)
def test_a_malformed_ptx_names_what_is_wrong(edit, message) -> None:
    with pytest.raises(CodecError, match=message):
        _read(_PTX.replace(*edit), ".ptx")


def test_a_blank_line_inside_a_ptx_scan_is_named_as_such() -> None:
    text = _PTX.replace("1 2 3 0.5 255 0 0\n", "1 2 3 0.5 255 0 0\n\n")
    with pytest.raises(CodecError, match="PTX scan of 2 points holds a blank line"):
        _read(text, ".ptx")


def test_an_empty_ptx_scan_reads_back_and_does_not_veto_colours() -> None:
    empty = "0\n1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    back = _read(empty, ".ptx")
    assert back.vertices.shape == (0, 3)
    assert back.vertex_attrs["intensity"].shape == (0,)
    assert back.global_attrs["ptx_transforms"].shape == (1, 4, 4)
    with pytest.warns(UserWarning, match="invalid PTX points"):
        back = _read(empty + _PTX, ".ptx")
    np.testing.assert_allclose(back.vertex_attrs["colors"], [[1, 0, 0], [0, 1, 0]])


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_spells_intensity_colours_then_normals() -> None:
    poly = _cloud(
        intensity=np.array([0.1, 0.2, 0.3]),
        colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
    )
    assert _write(poly).splitlines()[0] == "0 0 0 0.1 255 0 0 0 0 1"
    assert _write(poly, ".pts").splitlines()[:2] == ["3", "0 0 0 0.1 255 0 0 0 0 1"]
    back = polyxios.read(io.StringIO(_write(poly)), fmt=".xyz")
    for key, value in poly.vertex_attrs.items():
        np.testing.assert_allclose(back.vertex_attrs[key], value)


def test_write_ptx_is_one_scan_in_a_row_with_the_identity_pose() -> None:
    poly = _cloud(colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
    lines = _write(poly, ".ptx").splitlines()
    assert lines[:10] == [
        "3",
        "1",
        "0 0 0",
        "1 0 0",
        "0 1 0",
        "0 0 1",
        "1 0 0 0",
        "0 1 0 0",
        "0 0 1 0",
        "0 0 0 1",
    ]
    assert lines[10] == "0 0 0 1 255 0 0"
    back = polyxios.read(io.StringIO("\n".join(lines) + "\n"), fmt=".ptx")
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.vertex_attrs["intensity"], [1, 1, 1])


def test_write_ptx_of_no_points_reads_back_empty() -> None:
    poly = make_polydata(np.empty((0, 3)), [])
    back = polyxios.read(io.StringIO(_write(poly, ".ptx")), fmt=".ptx")
    assert back.vertices.shape == (0, 3)


def test_a_nan_colour_writes_as_black_without_a_warning() -> None:
    poly = _cloud(colors=np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = _write(poly)
    assert text.splitlines()[0] == "0 0 0 0 0 0"


def test_a_four_channel_colour_drops_its_alpha_with_a_warning() -> None:
    poly = _cloud(colors=np.array([[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, 0.0]]))
    with pytest.warns(UserWarning, match="alpha channel is dropped"):
        text = _write(poly)
    assert text.splitlines()[0] == "0 0 0 255 0 0"


def test_a_bad_float_fmt_is_refused_as_a_codec_error() -> None:
    with pytest.raises(CodecError, match="float_fmt 'zz' is not usable"):
        _write(_cloud(), float_fmt="zz")


def test_a_ptx_row_count_that_is_not_an_integer_is_a_codec_error() -> None:
    with pytest.raises(CodecError):
        _read(_PTX.replace("2\n1\n0 0 0", "2\n1.0\n0 0 0", 1), ".ptx")


def test_write_ptx_drops_normals_with_a_warning() -> None:
    poly = _cloud(normals=np.tile([0.0, 0.0, 1.0], (3, 1)))
    with pytest.warns(UserWarning, match="no column for vertex attributes normals"):
        _write(poly, ".ptx")


def test_a_nameless_buffer_takes_the_variant_by_name() -> None:
    poly = _cloud()
    buf = io.StringIO()
    polyxios.write(poly, buf, fmt=".xyz", variant="pts")
    assert buf.getvalue().splitlines()[0] == "3"
    with pytest.raises(ValueError, match="variant must be one of"):
        polyxios.write(poly, io.StringIO(), fmt=".xyz", variant="las")


def test_attributes_without_a_column_are_dropped_with_a_warning() -> None:
    poly = _cloud(
        scalar=np.arange(3.0), colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    )
    with pytest.warns(UserWarning, match="no column for vertex attributes scalar"):
        text = _write(poly)
    assert text.splitlines()[0] == "0 0 0 255 0 0"


def test_elements_that_are_not_points_are_dropped_with_a_warning() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    poly = make_polydata(verts, [("triangle", np.array([[0, 1, 2]]))])
    with pytest.warns(UserWarning, match="points only; 1 elements dropped"):
        text = _write(poly)
    assert len(text.splitlines()) == 3


def test_float_fmt_controls_the_spelling() -> None:
    poly = make_polydata(np.array([[1 / 3, 0, 0]]), [("vertex", np.array([[0]]))])
    assert _write(poly, float_fmt=".3f").strip() == "0.333 0.000 0.000"


def test_a_suffix_that_names_no_variant_writes_xyz_columns(tmp_path) -> None:
    poly = _cloud(intensity=np.array([0.1, 0.2, 0.3]))
    path = tmp_path / "cloud.txt"
    polyxios.write(poly, path, fmt=".xyz")
    assert path.read_text().splitlines()[0] == "0 0 0 0.1"
    back = polyxios.read(path, fmt=".xyz")
    np.testing.assert_array_equal(back.vertices, poly.vertices)


def test_a_written_file_reads_from_a_path_and_a_gzipped_one(tmp_path) -> None:
    poly = _cloud(intensity=np.array([0.1, 0.2, 0.3]))
    for name in ("cloud.pts", "cloud.pts.gz", "cloud.xyz"):
        path = tmp_path / name
        polyxios.write(poly, path)
        back = polyxios.read(path)
        np.testing.assert_array_equal(back.vertices, poly.vertices)
        np.testing.assert_allclose(back.vertex_attrs["intensity"], [0.1, 0.2, 0.3])


def test_integer_colours_count_0_to_255_like_every_other_codec() -> None:
    poly = _cloud(colors=np.array([[128, 0, 0], [0, 255, 0], [0, 0, 64]], np.uint8))
    assert _write(poly).splitlines()[0] == "0 0 0 128 0 0"


def test_a_single_column_intensity_is_written_like_a_flat_one() -> None:
    poly = _cloud(intensity=np.array([[0.1], [0.2], [0.3]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = _write(poly)
    assert text.splitlines()[0] == "0 0 0 0.1"


def test_warnings_point_at_the_caller() -> None:
    with pytest.warns(UserWarning, match="invalid PTX points") as caught:
        _read(_PTX, ".ptx")
    assert {w.filename for w in caught} == {__file__}
    poly = _cloud(scalar=np.arange(3.0))
    with pytest.warns(UserWarning, match="no column") as caught:
        _write(poly)
    assert {w.filename for w in caught} == {__file__}
