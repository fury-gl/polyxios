from __future__ import annotations

from dataclasses import replace
import gzip
import io
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from polyxios import make_polydata, read as api_read, write as api_write
from polyxios.codecs._svg import read, write
from polyxios.exceptions import CodecError, UnsupportedFormatError

_NS = "{http://www.w3.org/2000/svg}"

_SQUARE = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=np.float64)


def _square():
    return make_polydata(
        _SQUARE,
        [("triangle", np.array([[0, 1, 2]])), ("quad", np.array([[0, 1, 2, 3]]))],
    )


def _svg(path: Path) -> ET.Element:
    return ET.fromstring(path.read_bytes())


def _paths(root: ET.Element) -> dict[str, str]:
    return {p.get("class"): p.get("d") for p in root.iter(f"{_NS}path")}


def _write(poly, tmp_path: Path, **opts) -> ET.Element:
    path = tmp_path / "m.svg"
    write(poly, path, **opts)
    return _svg(path)


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------


def test_a_face_is_a_closed_path_through_its_corners(tmp_path: Path) -> None:
    root = _write(_square(), tmp_path)
    paths = _paths(root)
    # SVG's y runs downward, so the top edge of the mesh lands on y=0 and the
    # bottom edge on y=1: the picture is the right way up.
    assert paths["triangle"] == "M 0 1 L 2 1 L 2 0 Z"
    assert paths["quad"] == "M 0 1 L 2 1 L 2 0 L 0 0 Z"


def test_the_view_box_is_the_extent_padded_by_half_a_stroke(tmp_path: Path) -> None:
    root = _write(_square(), tmp_path)
    group = root.find(f"{_NS}g")
    # A hundredth of the shorter side.
    assert group.get("stroke-width") == "0.01"
    assert group.get("fill") == "none"
    assert root.get("viewBox") == "-0.005 -0.005 2.01 1.01"
    assert root.get("width") is None
    assert root.get("height") is None


def test_the_mesh_is_shifted_to_the_origin_not_the_view_box(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE + [10.0, -5.0, 0.0], [("quad", np.array([[0, 1, 2, 3]]))]
    )
    root = _write(poly, tmp_path)
    assert _paths(root)["quad"] == "M 0 1 L 2 1 L 2 0 L 0 0 Z"
    assert root.get("viewBox") == "-0.005 -0.005 2.01 1.01"


def test_one_path_per_type_in_mesh_order_with_elements_as_subpaths(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _SQUARE,
        [
            ("quad", np.array([[0, 1, 2, 3]])),
            ("line", np.array([[0, 2], [1, 3]])),
            ("triangle", np.array([[0, 1, 2]])),
        ],
    )
    root = _write(poly, tmp_path)
    classes = [p.get("class") for p in root.iter(f"{_NS}path")]
    assert classes == ["quad", "line", "triangle"]
    assert _paths(root)["line"] == "M 0 1 L 2 0 M 2 1 L 0 0"


def test_a_line_is_open_and_a_polygon_closed(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [
            ("poly_line", np.array([0, 1, 2]).reshape(1, 3)),
            ("polygon", np.array([0, 1, 2, 3]).reshape(1, 4)),
        ],
    )
    paths = _paths(_write(poly, tmp_path))
    assert paths["poly_line"] == "M 0 1 L 2 1 L 2 0"
    assert paths["polygon"] == "M 0 1 L 2 1 L 2 0 L 0 0 Z"


def test_a_pixel_is_drawn_as_a_ring_not_in_node_order(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("pixel", np.array([[0, 1, 3, 2]]))])
    assert _paths(_write(poly, tmp_path))["pixel"] == "M 0 1 L 2 1 L 2 0 L 0 0 Z"


def test_a_triangle_strip_is_one_ring_per_triangle(tmp_path: Path) -> None:
    poly = make_polydata(_SQUARE, [("triangle_strip", np.array([[0, 1, 3, 2]]))])
    assert (
        _paths(_write(poly, tmp_path))["triangle_strip"]
        == "M 0 1 L 2 1 L 0 0 Z M 2 1 L 0 0 L 2 0 Z"
    )


def test_a_higher_order_element_is_drawn_through_its_corners(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0.3, 0], [1.2, 1.2, 0], [0.3, 1, 0]],
        dtype=np.float64,
    )
    poly = make_polydata(
        verts,
        [
            ("quadratic_triangle", np.array([[0, 1, 2, 3, 4, 5]])),
            ("quadratic_edge", np.array([[0, 1, 3]])),
        ],
    )
    paths = _paths(_write(poly, tmp_path))
    assert paths["quadratic_triangle"] == "M 0 2 L 2 2 L 0 0 Z"
    assert paths["quadratic_edge"] == "M 0 2 L 2 2"


def test_a_quadratic_polygon_rings_its_first_half(tmp_path: Path) -> None:
    verts = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [2, 0.5, 0],
            [1, 1, 0],
            [0, 0.5, 0],
        ],
        dtype=np.float64,
    )
    poly = make_polydata(verts, [("quadratic_polygon", np.arange(8).reshape(1, 8))])
    assert (
        _paths(_write(poly, tmp_path))["quadratic_polygon"]
        == "M 0 1 L 2 1 L 2 0 L 0 0 Z"
    )


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_width_sets_the_rendered_size_and_keeps_the_aspect(tmp_path: Path) -> None:
    root = _write(_square(), tmp_path, width=402)
    assert root.get("width") == "402"
    assert root.get("height") == "202"
    # The coordinates are untouched: the view box does the scaling.
    assert root.get("viewBox") == "-0.005 -0.005 2.01 1.01"


def test_stroke_width_is_taken_as_given(tmp_path: Path) -> None:
    root = _write(_square(), tmp_path, stroke_width=0.5)
    assert root.find(f"{_NS}g").get("stroke-width") == "0.5"
    assert root.get("viewBox") == "-0.25 -0.25 2.5 1.5"


@pytest.mark.parametrize("option", ["stroke_width", "width"])
@pytest.mark.parametrize("value", [0, -1.0, "wide", float("nan"), True, np.True_])
def test_a_size_that_is_not_a_positive_number_is_refused(
    tmp_path: Path, option: str, value
) -> None:
    with pytest.raises(CodecError, match=f"{option} must be a positive number"):
        write(_square(), tmp_path / "m.svg", **{option: value})


def test_float_fmt_is_applied_everywhere(tmp_path: Path) -> None:
    root = _write(_square(), tmp_path, float_fmt=".2f", width=100)
    assert _paths(root)["triangle"] == "M 0.00 1.00 L 2.00 1.00 L 2.00 0.00 Z"
    assert root.get("viewBox") == "-0.01 -0.01 2.01 1.01"
    assert root.get("height") == "50.25"


def test_an_unusable_float_fmt_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="float_fmt"):
        write(_square(), tmp_path / "m.svg", float_fmt="nope")


@pytest.mark.parametrize("float_fmt", ['"<8', "<<8", "&^6g", ">=12.3f"])
def test_a_float_fmt_padding_with_an_xml_delimiter_is_refused(
    tmp_path: Path, float_fmt: str
) -> None:
    # A fill character lands inside d="..." and viewBox="..."; one that ends
    # the attribute early would leave a file no parser accepts.
    with pytest.raises(CodecError, match="which no XML attribute may hold"):
        write(_square(), tmp_path / "m.svg", float_fmt=float_fmt)


def test_a_harmless_padding_is_kept(tmp_path: Path) -> None:
    root = _write(_square(), tmp_path, float_fmt="*>4g")
    assert _paths(root)["triangle"] == "M ***0 ***1 L ***2 ***1 L ***2 ***0 Z"


def test_plane_picks_the_projection(tmp_path: Path) -> None:
    verts = np.array([[0, 5, 0], [2, 5, 0], [2, 5, 1], [0, 5, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])
    assert (
        _paths(_write(poly, tmp_path, plane="xz"))["quad"]
        == "M 0 1 L 2 1 L 2 0 L 0 0 Z"
    )


def test_an_unknown_plane_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CodecError, match="plane 'zx' is not one of xy, xz, yz"):
        write(_square(), tmp_path / "m.svg", plane="zx")
    with pytest.raises(CodecError, match="plane 3 is not one of"):
        write(_square(), tmp_path / "m.svg", plane=3)


def test_an_unrecognized_option_warns_and_is_ignored(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options {'binary'}"):
        write(_square(), tmp_path / "m.svg", binary=True)


def test_every_warning_points_at_the_caller(tmp_path: Path) -> None:
    verts = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0.5], [0, 1, 0], [1, 0.5, 1]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 4]])),
            ("poly_line", np.array([0]).reshape(1, 1)),
            ("quad", np.array([[0, 1, 2, 3]])),
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        api_write(poly, tmp_path / "m.svg", binary=True)
    assert len(caught) == 4
    assert {w.filename for w in caught} == {__file__}


# ---------------------------------------------------------------------------
# What is dropped, and said so
# ---------------------------------------------------------------------------


def test_a_mesh_off_the_plane_is_projected_with_a_warning(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0.5], [0, 1, 0.5]], dtype=np.float64)
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])
    with pytest.warns(UserWarning, match="not flat .* z runs from 0 to 0.5"):
        root = _write(poly, tmp_path)
    assert _paths(root)["quad"] == "M 0 1 L 2 1 L 2 0 L 0 0 Z"


def test_a_plane_off_the_origin_is_still_flat(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE + [0.0, 0.0, 7.0], [("quad", np.array([[0, 1, 2, 3]]))]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _write(poly, tmp_path)


def test_a_vertex_and_a_solid_cell_are_skipped_one_warning_per_type(
    tmp_path: Path,
) -> None:
    verts = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0.5, 1]], dtype=np.float64
    )
    poly = make_polydata(
        verts,
        [
            ("tetra", np.array([[0, 1, 2, 4], [0, 2, 3, 4]])),
            ("vertex", np.array([[4]])),
            ("quad", np.array([[0, 1, 2, 3]])),
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root = _write(poly, tmp_path)
    messages = sorted(str(w.message) for w in caught)
    assert len(messages) == 3
    assert any("2 tetra element(s) skipped" in m for m in messages)
    assert any("1 vertex element(s) skipped" in m for m in messages)
    assert any("not flat" in m for m in messages)
    assert list(_paths(root)) == ["quad"]


def test_a_free_size_element_too_small_to_draw_is_skipped(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [
            ("polygon", np.array([0, 1, 2, 3]).reshape(1, 4)),
            ("poly_line", np.array([0]).reshape(1, 1)),
            ("triangle_strip", np.array([0, 1]).reshape(1, 2)),
            ("quadratic_polygon", np.array([0, 1, 2]).reshape(1, 3)),
        ],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root = _write(poly, tmp_path)
    messages = sorted(str(w.message) for w in caught)
    assert len(messages) == 3
    assert any("1 poly_line element(s) with fewer than 2 nodes" in m for m in messages)
    assert any(
        "1 triangle_strip element(s) with fewer than 3 nodes" in m for m in messages
    )
    assert any(
        "1 quadratic_polygon element(s) with fewer than 4 nodes" in m for m in messages
    )
    assert list(_paths(root)) == ["polygon"]


def test_a_non_finite_vertex_is_refused(tmp_path: Path) -> None:
    # It would poison the extent, and with it the viewBox of the whole picture.
    verts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, np.nan], [0, 1, 0]])
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(CodecError, match="'m.svg': a vertex coordinate is not finite"):
        write(poly, tmp_path / "m.svg")
    verts[2, 2] = np.inf
    poly = make_polydata(verts, [("quad", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(CodecError, match="not finite"):
        write(poly, tmp_path / "m.svg")


def test_attributes_and_tags_have_no_spelling(tmp_path: Path) -> None:
    poly = make_polydata(
        _SQUARE,
        [("quad", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"t": np.arange(4.0)},
        element_attrs={"e": np.arange(1.0)},
        element_tags={"g": np.array([0], dtype=np.int32)},
        global_attrs={"title": "<script>"},
    )
    path = tmp_path / "m.svg"
    write(poly, path)
    text = path.read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "title" not in text
    ET.fromstring(text)


def test_an_element_past_the_vertex_table_is_refused(tmp_path: Path) -> None:
    poly = _square()
    bad = np.array(poly.connectivity)
    bad[0] = 9
    with pytest.raises(CodecError, match="outside 0..3"):
        write(replace(poly, connectivity=bad), tmp_path / "m.svg")


# ---------------------------------------------------------------------------
# Degenerate pictures
# ---------------------------------------------------------------------------


def test_an_empty_mesh_is_an_empty_picture(tmp_path: Path) -> None:
    poly = make_polydata(np.zeros((0, 3)), [])
    root = _write(poly, tmp_path)
    assert root.get("viewBox") == "-0.5 -0.5 1 1"
    assert _paths(root) == {}


def test_a_flat_line_still_has_a_view_box_with_height(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float64)
    poly = make_polydata(verts, [("line", np.array([[0, 1]]))])
    root = _write(poly, tmp_path)
    # The stroke falls back to the longer side when the shorter is flat, and
    # the padding gives the picture the height the mesh does not have.
    assert root.find(f"{_NS}g").get("stroke-width") == "0.04"
    assert root.get("viewBox") == "-0.02 -0.02 4.04 0.04"


def test_a_single_vertex_takes_the_unit_stroke(tmp_path: Path) -> None:
    poly = make_polydata(np.array([[1.0, 2.0, 3.0]]), [("vertex", np.array([[0]]))])
    with pytest.warns(UserWarning, match="1 vertex element\\(s\\) skipped"):
        root = _write(poly, tmp_path)
    assert root.find(f"{_NS}g").get("stroke-width") == "1"
    assert root.get("viewBox") == "-0.5 -0.5 1 1"


# ---------------------------------------------------------------------------
# Destinations, and the one direction there is
# ---------------------------------------------------------------------------


def test_a_buffer_and_a_gzip_path_are_written(tmp_path: Path) -> None:
    buf = io.BytesIO()
    api_write(_square(), buf, fmt="svg")
    ET.fromstring(buf.getvalue())

    path = tmp_path / "m.svg.gz"
    api_write(_square(), path)
    with gzip.open(path, "rb") as fh:
        assert ET.fromstring(fh.read()).tag == f"{_NS}svg"


def test_a_text_handle_is_written_as_text(tmp_path: Path) -> None:
    buf = io.StringIO()
    write(_square(), buf)
    assert buf.getvalue().startswith('<?xml version="1.0"')


def test_reading_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "m.svg"
    api_write(_square(), path)
    with pytest.raises(UnsupportedFormatError, match="'m.svg': SVG is write-only"):
        api_read(path)
    with pytest.raises(UnsupportedFormatError, match="write-only"):
        read(path, lazy=True)
    # An option meant for a reader is refused for the same reason, not with a
    # TypeError about an unexpected keyword.
    with pytest.raises(UnsupportedFormatError, match="write-only"):
        api_read(path, some_reader_option=1)
