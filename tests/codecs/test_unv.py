from __future__ import annotations

import dataclasses
import io
from pathlib import Path
import warnings

import numpy as np
import pytest

from polyxios import make_polydata, read as api_read, write as api_write
from polyxios._element_types import ELEMENT_TYPES, NODES_PER_ELEMENT
from polyxios.codecs._unv import _WRITE_DESCRIPTOR, read, write
from polyxios.exceptions import CodecError

_TET_VERTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def _nodes(labels, coords) -> str:
    """A 2411 dataset in the layout a reference writer spells."""
    lines = ["    -1", "  2411"]
    for label, (x, y, z) in zip(labels, coords):
        lines.append(f"{label:10d}{1:10d}{1:10d}{11:10d}")
        lines.append(f"{x:25.16E}{y:25.16E}{z:25.16E}".replace("E", "D"))
    lines.append("    -1")
    return "\n".join(lines) + "\n"


def _elements(*records: str) -> str:
    return "    -1\n  2412\n" + "".join(records) + "    -1\n"


def _element(label, descriptor, nodes, *, beam=False) -> str:
    out = f"{label:10d}{descriptor:10d}{1:10d}{1:10d}{7:10d}{len(nodes):10d}\n"
    if beam:
        out += f"{0:10d}{0:10d}{0:10d}\n"
    for k in range(0, len(nodes), 8):
        out += "".join(f"{v:10d}" for v in nodes[k : k + 8]) + "\n"
    return out


def _group(number, name, members, dataset=2477) -> str:
    lines = ["    -1", f"{dataset:6d}"]
    lines.append(f"{number:10d}" + f"{0:10d}" * 6 + f"{len(members):10d}")
    lines.append(name)
    lines.extend(
        "".join(f"{t:10d}{v:10d}{0:10d}{0:10d}" for t, v in members[k : k + 2])
        for k in range(0, len(members), 2)
    )
    lines.append("    -1")
    return "\n".join(lines) + "\n"


def _write(tmp_path: Path, text: str, name: str = "m.unv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


_TET_FILE = _nodes([1, 2, 3, 4], _TET_VERTS) + _elements(_element(1, 111, [1, 2, 3, 4]))


def _mixed_mesh():
    return make_polydata(
        _TET_VERTS,
        [
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("triangle", np.array([[0, 1, 2], [0, 1, 3]])),
            ("line", np.array([[0, 3]])),
        ],
        vertex_tags={"top": np.array([3, 0], dtype=np.int32)},
        element_tags={
            "solid": np.array([0], dtype=np.int32),
            "skin": np.array([1, 2], dtype=np.int32),
        },
    )


# --- reading ---------------------------------------------------------------


def test_minimal_file(tmp_path: Path) -> None:
    poly = read(_write(tmp_path, _TET_FILE))
    np.testing.assert_array_equal(poly.vertices, _TET_VERTS)
    assert poly.element_types.tolist() == [ELEMENT_TYPES["tetra"]]
    assert poly.connectivity.tolist() == [0, 1, 2, 3]
    assert "original_ids" not in poly.vertex_attrs
    assert "original_ids" not in poly.element_attrs


def test_free_labels_are_renumbered_and_remembered(tmp_path: Path) -> None:
    text = _nodes([40, 10, 30, 20], _TET_VERTS) + _elements(
        _element(7, 111, [40, 10, 30, 20])
    )
    poly = read(_write(tmp_path, text))
    assert poly.connectivity.tolist() == [0, 1, 2, 3]
    assert poly.vertex_attrs["original_ids"].tolist() == [40, 10, 30, 20]
    assert poly.element_attrs["original_ids"].tolist() == [7]


def test_a_fortran_d_exponent_and_a_bare_exponent_read(tmp_path: Path) -> None:
    text = (
        "    -1\n  2411\n         1         1         1        11\n"
        "   1.5000000000000000D+00   2.0000000000000000d-01   1.00000-100\n"
        "    -1\n"
    )
    poly = read(_write(tmp_path, text))
    np.testing.assert_array_equal(poly.vertices, [[1.5, 0.2, 1e-100]])


def test_single_precision_nodes_read(tmp_path: Path) -> None:
    text = (
        "    -1\n    15\n"
        "         1         0         0        11  1.00000E+00  2.00000E+00  3.00000E+00\n"
        "         2         0         0        11 -1.00000E+00  0.00000E+00  5.00000E-01\n"
        "    -1\n"
    )
    poly = read(_write(tmp_path, text))
    np.testing.assert_array_equal(poly.vertices, [[1, 2, 3], [-1, 0, 0.5]])


def test_a_line_padded_to_eighty_columns_is_still_a_delimiter(tmp_path: Path) -> None:
    text = "\n".join(line.ljust(80) for line in _TET_FILE.splitlines()) + "\n"
    assert read(_write(tmp_path, text)).element_types.size == 1


def test_a_beam_skips_its_orientation_record(tmp_path: Path) -> None:
    text = _nodes([1, 2, 3, 4], _TET_VERTS) + _elements(
        _element(1, 21, [1, 2], beam=True),
        _element(2, 11, [2, 3], beam=True),
        _element(3, 41, [1, 2, 3]),
    )
    poly = read(_write(tmp_path, text))
    assert poly.element_types.tolist() == [
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["triangle"],
    ]
    assert poly.connectivity.tolist() == [0, 1, 1, 2, 0, 1, 2]


@pytest.mark.parametrize(
    ("descriptor", "count", "name"),
    [
        (41, 3, "triangle"),
        (51, 3, "triangle"),
        (61, 3, "triangle"),
        (74, 3, "triangle"),
        (81, 3, "triangle"),
        (91, 3, "triangle"),
        (44, 4, "quad"),
        (71, 4, "quad"),
        (94, 4, "quad"),
        (42, 6, "quadratic_triangle"),
        (95, 8, "quadratic_quad"),
        (111, 4, "tetra"),
        (118, 10, "quadratic_tetra"),
        (112, 6, "wedge"),
        (113, 15, "quadratic_wedge"),
        (115, 8, "hexahedron"),
        (116, 20, "quadratic_hexahedron"),
        (161, 1, "vertex"),
    ],
)
def test_a_descriptor_reads_as_its_geometry(
    tmp_path: Path, descriptor: int, count: int, name: str
) -> None:
    coords = np.arange(3.0 * count).reshape(count, 3)
    labels = list(range(1, count + 1))
    text = _nodes(labels, coords) + _elements(_element(1, descriptor, labels))
    poly = read(_write(tmp_path, text))
    assert poly.element_types.tolist() == [ELEMENT_TYPES[name]]


# The node labels a reference mesher spelled, in its own file, for one
# element of each second order type whose nodes it numbered 0.. in its own
# node order; the expected list is that element in polyxios order, checked
# against the same mesher's own native file.
_SECOND_ORDER = [
    (24, [0, 2, 1], "quadratic_edge", [0, 1, 2]),
    (92, [0, 3, 1, 4, 2, 5], "quadratic_triangle", [0, 1, 2, 3, 4, 5]),
    (95, [0, 4, 1, 5, 2, 6, 3, 7], "quadratic_quad", [0, 1, 2, 3, 4, 5, 6, 7]),
    (
        118,
        [0, 4, 1, 5, 2, 6, 7, 9, 8, 3],
        "quadratic_tetra",
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
    ),
    (
        116,
        [0, 8, 1, 11, 2, 13, 3, 9, 10, 12, 14, 15, 4, 16, 5, 18, 6, 19, 7, 17],
        "quadratic_hexahedron",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15],
    ),
    (
        113,
        [0, 6, 1, 9, 2, 7, 8, 10, 11, 3, 12, 4, 14, 5, 13],
        "quadratic_wedge",
        [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11],
    ),
]


@pytest.mark.parametrize(("descriptor", "spelled", "name", "expected"), _SECOND_ORDER)
def test_second_order_nodes_are_put_in_polyxios_order(
    tmp_path: Path, descriptor, spelled, name, expected
) -> None:
    count = len(spelled)
    coords = np.arange(3.0 * count).reshape(count, 3)
    labels = [100 + k for k in range(count)]
    text = _nodes(labels, coords) + _elements(
        _element(500, descriptor, [100 + k for k in spelled], beam=descriptor == 24)
    )
    poly = read(_write(tmp_path, text))
    assert poly.element_types.tolist() == [ELEMENT_TYPES[name]]
    assert poly.connectivity.tolist() == expected


@pytest.mark.parametrize(("descriptor", "spelled", "name", "expected"), _SECOND_ORDER)
def test_second_order_nodes_are_written_in_the_format_order(
    tmp_path: Path, descriptor, spelled, name, expected
) -> None:
    count = len(spelled)
    poly = make_polydata(
        np.arange(3.0 * count).reshape(count, 3), [(name, np.array([expected]))]
    )
    path = tmp_path / "out.unv"
    write(poly, path)
    text = path.read_text().split("  2412\n", 1)[1].split("    -1", 1)[0]
    rows = text.splitlines()
    assert int(rows[0].split()[1]) == descriptor
    body = rows[2:] if descriptor == 24 else rows[1:]
    written = [int(v) - 1 for row in body for v in row.split()]
    assert written == spelled


def test_an_element_spanning_lines_is_gathered(tmp_path: Path) -> None:
    labels = list(range(1, 21))
    text = _nodes(labels, np.zeros((20, 3))) + _elements(_element(1, 116, labels))
    assert read(_write(tmp_path, text)).element_types.size == 1


def test_groups_become_vertex_and_element_tags(tmp_path: Path) -> None:
    text = (
        _nodes([1, 2, 3, 4], _TET_VERTS)
        + _elements(
            _element(10, 111, [1, 2, 3, 4]),
            _element(11, 91, [1, 2, 3]),
        )
        + _group(1, "wall", [(7, 3), (7, 1), (8, 11)])
        + _group(2, "solid", [(8, 10)], dataset=2467)
    )
    poly = read(_write(tmp_path, text))
    assert poly.vertex_tags["wall"].tolist() == [2, 0]
    assert poly.element_tags["wall"].tolist() == [1]
    assert poly.element_tags["solid"].tolist() == [0]
    assert "solid" not in poly.vertex_tags


def test_two_groups_of_one_name_add_to_one_tag(tmp_path: Path) -> None:
    text = (
        _TET_FILE
        + _group(1, "g", [(7, 1)])
        + _group(2, "g", [(7, 2), (7, 1)], dataset=2467)
    )
    assert read(_write(tmp_path, text)).vertex_tags["g"].tolist() == [0, 1]


def test_an_empty_group_reads_as_an_empty_element_tag(tmp_path: Path) -> None:
    poly = read(_write(tmp_path, _TET_FILE + _group(1, "none", [])))
    assert poly.element_tags["none"].tolist() == []


def test_a_group_member_naming_nothing_is_dropped_with_a_warning(
    tmp_path: Path,
) -> None:
    text = _TET_FILE + _group(1, "g", [(7, 1), (7, 99), (8, 42)])
    with pytest.warns(UserWarning, match=r"1 node\(s\) and 1 element\(s\)"):
        poly = read(_write(tmp_path, text))
    assert poly.vertex_tags["g"].tolist() == [0]
    assert "g" not in poly.element_tags


def test_a_group_member_of_another_entity_type_is_passed_over(
    tmp_path: Path,
) -> None:
    text = _TET_FILE + _group(1, "g", [(7, 1), (3, 1)])
    with pytest.warns(UserWarning, match=r"entity type\(s\) \[3\]"):
        poly = read(_write(tmp_path, text))
    assert poly.vertex_tags["g"].tolist() == [0]


def test_a_group_with_no_name_is_named_and_warned_about(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="carry no name"):
        poly = read(_write(tmp_path, _TET_FILE + _group(5, "  ", [(7, 1)])))
    assert "group_5" in poly.vertex_tags


def _analysis(name, location, width, records, *, data_type=4, header_lines=2) -> str:
    lines = ["    -1", "  2414", "         1", name, f"{location:10d}"]
    lines += ["NONE"] * 5
    lines.append(f"{1:10d}{1:10d}{1:10d}{94:10d}{data_type:10d}{width:10d}")
    lines.append(f"{0:10d}" * 8)
    lines.append(f"{0:10d}" * 2)
    lines += [" 0.00000E+00" * (12 // header_lines)] * header_lines
    lines += records
    lines.append("    -1")
    return "\n".join(lines) + "\n"


def test_unnamed_groups_of_one_number_in_two_datasets_stay_apart(
    tmp_path: Path,
) -> None:
    text = _TET_FILE + _group(5, "", [(7, 1)]) + _group(5, "", [(7, 2)], dataset=2467)
    with pytest.warns(UserWarning, match="carry no name"):
        poly = read(_write(tmp_path, text))
    assert poly.vertex_tags["group_5"].tolist() == [0]
    assert poly.vertex_tags["group_5_2"].tolist() == [1]


def test_an_unnamed_group_does_not_take_a_later_group_s_name(
    tmp_path: Path,
) -> None:
    text = _TET_FILE + _group(5, "", [(7, 1)]) + _group(6, "group_5", [(7, 2)])
    with pytest.warns(UserWarning, match="carry no name"):
        poly = read(_write(tmp_path, text))
    assert poly.vertex_tags["group_5"].tolist() == [1]
    assert poly.vertex_tags["group_5_2"].tolist() == [0]
    assert list(poly.vertex_tags) == ["group_5_2", "group_5"]


def test_an_empty_unnamed_group_keeps_its_element_tag(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="carry no name"):
        poly = read(_write(tmp_path, _TET_FILE + _group(3, "", [])))
    assert poly.element_tags["group_3"].size == 0


def test_a_cut_unnamed_group_is_named_in_the_error(tmp_path: Path) -> None:
    text = _TET_FILE + _group(5, "", [(7, 1), (7, 2), (7, 3)])
    text = text.replace("         7         3         0         0", "")
    with pytest.raises(CodecError, match="group 'group_5'"):
        read(_write(tmp_path, text))


def test_analysis_data_at_nodes_reads_as_a_vertex_attribute(tmp_path: Path) -> None:
    text = _TET_FILE + _analysis(
        "TEMP",
        1,
        1,
        ["         1", "  1.50000E+00", "         3", "  3.50000E+00"],
    )
    poly = read(_write(tmp_path, text))
    np.testing.assert_array_equal(poly.vertex_attrs["TEMP"], [1.5, np.nan, 3.5, np.nan])


def test_analysis_vector_data_reads_one_row_per_node(tmp_path: Path) -> None:
    records = []
    for k in range(1, 5):
        records += [f"{k:10d}", f"{k:13.5E}{k + 0.5:13.5E}{-k:13.5E}"]
    poly = read(_write(tmp_path, _TET_FILE + _analysis("DISP", 1, 3, records)))
    assert poly.vertex_attrs["DISP"].shape == (4, 3)
    np.testing.assert_array_equal(poly.vertex_attrs["DISP"][2], [3, 3.5, -3])


def test_double_precision_reals_over_more_lines_are_counted_by_value(
    tmp_path: Path,
) -> None:
    records = ["         1", "  2.0000000000000000D+00"]
    text = _TET_FILE + _analysis("P", 1, 1, records, header_lines=4)
    assert read(_write(tmp_path, text)).vertex_attrs["P"][0] == 2.0


def test_analysis_data_on_elements_reads_as_an_element_attribute(
    tmp_path: Path,
) -> None:
    records = ["         1         2", "  1.00000E+00  2.00000E+00"]
    poly = read(_write(tmp_path, _TET_FILE + _analysis("S", 2, 2, records)))
    np.testing.assert_array_equal(poly.element_attrs["S"], [[1.0, 2.0]])


def test_element_data_of_varying_width_is_skipped_with_a_warning(
    tmp_path: Path,
) -> None:
    text = (
        _nodes([1, 2, 3, 4], _TET_VERTS)
        + _elements(_element(1, 111, [1, 2, 3, 4]), _element(2, 91, [1, 2, 3]))
        + _analysis(
            "L",
            2,
            1,
            ["         1         1", "  1.0E+00", "         2         2", " 1.0 2.0"],
        )
    )
    with pytest.warns(UserWarning, match="change their value count"):
        poly = read(_write(tmp_path, text))
    assert "L" not in poly.element_attrs


def test_complex_and_integer_data_keep_their_kind(tmp_path: Path) -> None:
    complex_records = []
    int_records = []
    for k in range(1, 5):
        complex_records += [f"{k:10d}", f"{k:13.5E}{-k:13.5E}"]
        int_records += [f"{k:10d}", f"{10 * k:10d}"]
    text = (
        _TET_FILE
        + _analysis("C", 1, 1, complex_records, data_type=6)
        + _analysis("I", 1, 1, int_records, data_type=1)
    )
    poly = read(_write(tmp_path, text))
    np.testing.assert_array_equal(
        poly.vertex_attrs["C"], [1 - 1j, 2 - 2j, 3 - 3j, 4 - 4j]
    )
    assert poly.vertex_attrs["I"].dtype == np.int64
    assert poly.vertex_attrs["I"].tolist() == [10, 20, 30, 40]


def test_repeated_result_names_are_kept_apart(tmp_path: Path) -> None:
    records = [f"{k:10d}\n  1.00000E+00" for k in range(1, 5)]
    text = (
        _TET_FILE
        + _analysis("T", 1, 1, records)
        + _analysis("T", 1, 1, records)
        + _analysis("original_ids", 1, 1, records)
    )
    poly = read(_write(tmp_path, text))
    # The id table's key is kept for ids, so a result named after it is not
    # taken for one on the next write.
    assert sorted(poly.vertex_attrs) == ["T", "T_2", "original_ids_2"]


def test_a_complex_value_with_one_nan_part_is_written(tmp_path: Path) -> None:
    poly = dataclasses.replace(
        _mixed_mesh(),
        vertex_attrs={
            "z": np.array([complex(np.nan, 1.0), 2.0, complex(np.nan, np.nan), 4.0])
        },
    )
    path = tmp_path / "m.unv"
    write(poly, path)
    back = read(path).vertex_attrs["z"]
    assert back[0].imag == 1.0 and np.isnan(back[0].real)
    assert np.isnan(back[2].real) and np.isnan(back[2].imag)


def test_complex_element_data_counts_complex_values(tmp_path: Path) -> None:
    records = ["         1         2", "  1.0E+00  2.0E+00  3.0E+00  4.0E+00"]
    text = _TET_FILE + _analysis("Z", 2, 2, records, data_type=6)
    poly = read(_write(tmp_path, text))
    np.testing.assert_array_equal(poly.element_attrs["Z"], [[1 + 2j, 3 + 4j]])


def test_analysis_specific_reals_past_twelve_are_refused(tmp_path: Path) -> None:
    text = _TET_FILE + _analysis("T", 1, 1, ["         1", "  1.0"])
    text = text.replace(" 0.00000E+00\n", " 0.00000E+00 0.00000E+00\n", 1)
    with pytest.raises(CodecError, match="13 analysis-specific reals"):
        read(_write(tmp_path, text))


def test_data_at_nodes_on_elements_is_skipped_with_a_warning(tmp_path: Path) -> None:
    text = _TET_FILE + _analysis(
        "N", 3, 1, ["         1         1         4         1"]
    )
    with pytest.warns(UserWarning, match="2414 location 3"):
        poly = read(_write(tmp_path, text))
    assert "N" not in poly.vertex_attrs and "N" not in poly.element_attrs


def test_dataset_55_reads_as_a_vertex_attribute(tmp_path: Path) -> None:
    text = _TET_FILE + (
        "    -1\n    55\nmode shape\nNONE\nNONE\nNONE\nNONE\n"
        "         1         2         2         8         2         3\n"
        "         2         4         1         1\n"
        "  1.00000e+01  1.00000e+00  0.00000e+00  0.00000e+00\n"
        "         2\n  1.00000e+00  5.00000e+00  9.00000e+00\n"
        "         4\n  4.00000e+00  8.00000e+00  1.20000e+01\n"
        "    -1\n"
    )
    poly = read(_write(tmp_path, text))
    np.testing.assert_array_equal(poly.vertex_attrs["mode shape"][1], [1, 5, 9])
    assert np.isnan(poly.vertex_attrs["mode shape"][0]).all()


def test_a_result_naming_an_unknown_entity_drops_that_record(tmp_path: Path) -> None:
    records = ["         9", "  1.00000E+00", "         1", "  2.00000E+00"]
    with pytest.warns(UserWarning, match="1 entity label"):
        poly = read(_write(tmp_path, _TET_FILE + _analysis("T", 1, 1, records)))
    assert poly.vertex_attrs["T"][0] == 2.0


def test_datasets_this_codec_does_not_read_are_passed_over(tmp_path: Path) -> None:
    text = (
        "    -1\n   151\nmodel\nNONE\nNONE\nNONE\nNONE\nNONE\nNONE\n    -1\n"
        "    -1\n   164\n         1  SI: Meter (newton)         2\n"
        "  1.0000000000000000E+00  1.0000000000000000E+00  1.0000000000000000E+00\n"
        "  2.7314999999999998E+02\n    -1\n" + _TET_FILE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read(_write(tmp_path, text)).element_types.size == 1


def test_a_data_dataset_this_codec_does_not_read_is_warned_about(
    tmp_path: Path,
) -> None:
    text = _TET_FILE + "    -1\n    57\nwhatever\n    -1\n"
    with pytest.warns(UserWarning, match="57 \\(data at nodes on elements\\)"):
        read(_write(tmp_path, text))


def test_an_unknown_descriptor_is_skipped_with_a_warning(tmp_path: Path) -> None:
    text = _nodes([1, 2, 3, 4], _TET_VERTS) + _elements(
        _element(1, 136, [1, 2]),
        _element(2, 93, [1, 2, 3, 4, 1, 2, 3, 4, 1]),
        _element(3, 111, [1, 2, 3, 4]),
    )
    with pytest.warns(UserWarning, match=r"136 \(2 nodes\) x1, 93 \(9 nodes\) x1"):
        poly = read(_write(tmp_path, text))
    assert poly.element_types.tolist() == [ELEMENT_TYPES["tetra"]]


def test_members_of_a_skipped_element_are_told_apart(tmp_path: Path) -> None:
    text = (
        _nodes([1, 2, 3, 4], _TET_VERTS)
        + _elements(_element(1, 111, [1, 2, 3, 4]), _element(2, 136, [1, 2]))
        + _group(1, "g", [(8, 1), (8, 2)])
        + _analysis(
            "S",
            2,
            1,
            ["         1         1", "  1.0", "         2         1", "  2.0"],
        )
    )
    with pytest.warns(UserWarning) as record:
        poly = read(_write(tmp_path, text))
    messages = " ".join(str(w.message) for w in record)
    assert (
        "1 group member(s) and 1 result record(s) name an element skipped" in messages
    )
    assert "declares no entity for" not in messages
    assert "declares no node or element" not in messages
    assert poly.element_tags["g"].tolist() == [0]
    assert poly.element_attrs["S"].tolist() == [1.0]


def test_labels_and_integers_past_2_53_are_kept_exact(tmp_path: Path) -> None:
    big = 2**53 + 1
    text = (
        _nodes([big, 2, 3, 4], _TET_VERTS)
        + _elements(_element(1, 111, [big, 2, 3, 4]))
        + _analysis(
            "I",
            1,
            1,
            [f"{label:20d}\n{big:20d}" for label in (big, 2, 3, 4)],
            data_type=1,
        )
    )
    poly = read(_write(tmp_path, text))
    assert poly.vertex_attrs["I"].dtype == np.int64
    assert poly.vertex_attrs["I"].tolist() == [big] * 4


def test_an_integer_result_past_2_53_widened_to_float_is_warned_about(
    tmp_path: Path,
) -> None:
    big = 2**53 + 1
    records = [f"{k:10d}\n{big:20d}" for k in (1, 2, 3)]
    text = _TET_FILE + _analysis("I", 1, 1, records, data_type=1)
    with pytest.warns(UserWarning, match=r"\['I'\] leave an entity without data"):
        poly = read(_write(tmp_path, text))
    assert poly.vertex_attrs["I"].dtype == np.float64


def test_an_integer_result_widened_to_float_within_2_53_is_silent(
    tmp_path: Path,
) -> None:
    records = [f"{k:10d}\n{2**53:20d}" for k in (1, 2, 3)]
    text = _TET_FILE + _analysis("I", 1, 1, records, data_type=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = read(_write(tmp_path, text))
    assert poly.vertex_attrs["I"][:3].tolist() == [2.0**53] * 3


def test_a_skipped_element_s_label_repeated_by_a_kept_one_is_refused(
    tmp_path: Path,
) -> None:
    text = (
        _nodes([1, 2, 3, 4], _TET_VERTS)
        + _elements(_element(1, 136, [1, 2]), _element(1, 111, [1, 2, 3, 4]))
        + _group(1, "springs", [(8, 1)])
    )
    with pytest.raises(
        CodecError, match="element label 1 at line 16 repeats the one at line 14"
    ):
        read(_write(tmp_path, text))


def test_a_repeat_among_skipped_elements_names_lines_in_file_order(
    tmp_path: Path,
) -> None:
    text = _nodes([1, 2, 3, 4], _TET_VERTS) + _elements(
        _element(5, 111, [1, 2, 3, 4]),
        _element(5, 136, [1, 2]),
    )
    with pytest.raises(
        CodecError, match="label 5 at line 16 repeats the one at line 14"
    ):
        read(_write(tmp_path, text))


@pytest.mark.parametrize("dataset", [2435, 2452, 2467, 2477])
def test_every_permanent_group_dataset_reads(tmp_path: Path, dataset: int) -> None:
    text = _TET_FILE + _group(1, "g", [(7, 2), (8, 1)], dataset=dataset)
    poly = read(_write(tmp_path, text))
    assert poly.vertex_tags["g"].tolist() == [1]
    assert poly.element_tags["g"].tolist() == [0]


def test_a_result_naming_an_entity_twice_keeps_the_last_record(
    tmp_path: Path,
) -> None:
    records = [f"{k:10d}\n{10 * k:10d}" for k in (1, 2, 3, 4)] + [
        "         1\n        99"
    ]
    text = _TET_FILE + _analysis("I", 1, 1, records, data_type=1)
    with pytest.warns(UserWarning, match=r"\['I'\] name one entity in more than one"):
        poly = read(_write(tmp_path, text))
    assert poly.vertex_attrs["I"].dtype == np.int64
    assert poly.vertex_attrs["I"].tolist() == [99, 20, 30, 40]


def test_a_bad_node_label_names_its_own_line(tmp_path: Path) -> None:
    text = _nodes(range(1, 11), np.zeros((10, 3))) + (
        "    -1\n  2412\n         1       115         1         1         7"
        "         8\n         1         2         3         4         5         6"
        "         7\n        x8\n    -1\n"
    )
    with pytest.raises(CodecError, match="'x8' on line 28"):
        read(_write(tmp_path, text))


def test_lazy_warns_and_loads_eagerly(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="lazy=True is not supported"):
        poly = read(_write(tmp_path, _TET_FILE), lazy=True)
    assert poly.element_types.size == 1


def test_a_buffer_and_gzip_read_and_write(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    buf = io.BytesIO()
    api_write(poly, buf, fmt=".unv")
    buf.seek(0)
    assert api_read(buf, fmt=".unv").element_types.size == 4
    api_write(poly, tmp_path / "m.unv.gz")
    assert api_read(tmp_path / "m.unv.gz").element_types.size == 4


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "no dataset found"),
        ("hello\n", "outside every dataset"),
        ("    -1\n  2411\n         1         1         1        11\n", "no closing -1"),
        ("    -1\n    -1\n", "holds no dataset number"),
        ("    -1\n 2411b     1     1\n    -1\n", "is binary"),
        ("    -1\n  24x1\n    -1\n", "should hold a dataset number"),
        (_TET_FILE + "trailing\n", "outside every dataset"),
        (
            "    -1\n  2411\n         1         1         1        11\n    -1\n",
            "every node is two records",
        ),
        (
            "    -1\n  2411\n         1         1         1        11\n"
            "  1.0  2.0\n    -1\n",
            r"carries 2 value\(s\), expected 3",
        ),
        (
            "    -1\n  2411\n         1         1         1        11\n"
            "  1.0  2.0  abc\n    -1\n",
            "malformed coordinate 'abc'",
        ),
        (
            "    -1\n  2411\n         0         1         1        11\n"
            "  1.0  2.0  3.0\n    -1\n",
            "node label 0 at line 3 is not positive",
        ),
        (
            "    -1\n  2411\n       1.5         1         1        11\n"
            "  1.0  2.0  3.0\n    -1\n",
            "malformed node label '1.5'",
        ),
        (
            _nodes([1, 1], np.zeros((2, 3))),
            "node label 1 at line 5 repeats the one at line 3",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + _elements(_element(1, 111, [1, 2, 3, 4]), _element(1, 111, [1, 2, 3, 4])),
            "element label 1 at line 16 repeats the one at line 14",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + _elements(_element(1, 111, [1, 2, 3, 9])),
            "names node 9, which the file never declares",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + "    -1\n  2412\n         1       111         1         1\n    -1\n",
            r"carries 4 field\(s\), expected 6",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + "    -1\n  2412\n         1       111         1         1"
            "         7         4\n         1         2\n    -1\n",
            r"lists 2 of its 4 node\(s\); the dataset is cut short",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + "    -1\n  2412\n         1       111         1         1"
            "         7         4\n         1         2\n         3         4"
            "         1\n    -1\n",
            r"lists 5 node\(s\) where it declares 4",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + "    -1\n  2412\n         1        21         1         1"
            "         7         2\n    -1\n",
            "has no orientation record",
        ),
        (
            _nodes([1, 2, 3, 4], _TET_VERTS)
            + _elements(_element(0, 111, [1, 2, 3, 4])),
            "element label 0",
        ),
        (
            _TET_FILE + "    -1\n  2477\n         1         0\n    -1\n",
            "has no name record",
        ),
        (
            _TET_FILE + "    -1\n  2477\n         5\n    -1\n",
            "should end in a member count",
        ),
        (
            _TET_FILE + "    -1\n  2477\n         1         0         3\ng\n"
            "         7         1         0         0         7         2\n"
            "         0         0\n    -1\n",
            r"lists 2 of its 3 member\(s\)",
        ),
        (
            _TET_FILE + "    -1\n  2414\n         1\nT\n    -1\n",
            "cut short in its header",
        ),
        (
            _TET_FILE + _analysis("T", 1, 1, ["         1", "  1.0  2.0"]),
            "not a whole number of 2-value records",
        ),
        (
            _TET_FILE + _analysis("T", 1, 1, ["       1.5", "  1.0"]),
            "not a whole number",
        ),
        (
            _TET_FILE + _analysis("T", 1, 1, ["       1_0", "  1.0"]),
            "malformed entity label '1_0'",
        ),
        (
            _TET_FILE
            + _analysis("I", 1, 1, ["         1", "\uff11\uff12"], data_type=1),
            "malformed result value '\uff11\uff12'",
        ),
        (
            _TET_FILE + _analysis("T", 1, 1, ["         1", "  1_0.5"]),
            "malformed result value '1_0.5'",
        ),
        (
            _nodes([1], [[0.0, 0.0, 0.0]]).replace("0.0000000000000000D+00", "1_0", 1),
            "malformed coordinate '1_0'",
        ),
    ],
)
def test_a_malformed_file_is_refused(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(CodecError, match=match):
        read(_write(tmp_path, text))


def test_a_label_wider_than_64_bits_is_refused(tmp_path: Path) -> None:
    text = (
        "    -1\n  2411\n 99999999999999999999         1         1        11\n"
        "  1.0  2.0  3.0\n    -1\n"
    )
    with pytest.raises(CodecError, match="wider than 64 bits"):
        read(_write(tmp_path, text))


def test_an_absurd_element_node_count_is_refused_before_gathering(
    tmp_path: Path,
) -> None:
    text = _nodes([1, 2, 3, 4], _TET_VERTS) + (
        f"    -1\n  2412\n         1       111         1         1         7"
        f" {2**62}\n         1         2         3         4\n    -1\n"
    )
    with pytest.raises(CodecError, match=str(2**62)):
        read(_write(tmp_path, text))


# --- writing ---------------------------------------------------------------


def test_roundtrip_keeps_element_order_tags_and_coordinates(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    path = tmp_path / "m.unv"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.vertices, poly.vertices)
    np.testing.assert_array_equal(back.element_types, poly.element_types)
    np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    assert back.vertex_tags["top"].tolist() == [3, 0]
    assert back.element_tags["solid"].tolist() == [0]
    assert back.element_tags["skin"].tolist() == [1, 2]


def test_coordinates_survive_exactly(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    verts = rng.normal(size=(50, 3)) * 10.0 ** rng.integers(-300, 300, size=(50, 3))
    poly = make_polydata(verts, [("vertex", np.arange(50).reshape(-1, 1))])
    path = tmp_path / "m.unv"
    write(poly, path)
    np.testing.assert_array_equal(read(path).vertices, verts)


@pytest.mark.parametrize(
    "name", sorted(name for name, count in NODES_PER_ELEMENT.items() if count > 0)
)
def test_every_type_either_round_trips_or_is_dropped(tmp_path: Path, name) -> None:
    count = NODES_PER_ELEMENT[name]
    poly = make_polydata(
        np.arange(3.0 * count).reshape(count, 3),
        [(name, np.arange(count).reshape(1, count))],
    )
    path = tmp_path / "m.unv"
    if name in _WRITE_DESCRIPTOR:
        write(poly, path)
        back = read(path)
        np.testing.assert_array_equal(back.element_types, poly.element_types)
        np.testing.assert_array_equal(back.connectivity, poly.connectivity)
    else:
        with pytest.warns(UserWarning, match="have no universal file descriptor"):
            write(poly, path)
        assert read(path).element_types.size == 0


def test_remembered_ids_are_written_back(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"original_ids": np.array([40, 10, 30, 20])},
        element_attrs={"original_ids": np.array([77])},
    )
    path = tmp_path / "m.unv"
    write(poly, path)
    back = read(path)
    assert back.vertex_attrs["original_ids"].tolist() == [40, 10, 30, 20]
    assert back.element_attrs["original_ids"].tolist() == [77]
    assert "original_ids" not in path.read_text()


def test_ids_too_wide_for_a_label_field_are_renumbered(tmp_path: Path) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        vertex_attrs={"original_ids": np.array([1, 2, 1_500_000_000, 4])},
        element_attrs={"original_ids": np.array([2_000_000_000])},
    )
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match="999999999"):
        write(poly, path)
    back = read(path)
    assert back.element_types.tolist() == [ELEMENT_TYPES["tetra"]]
    assert "original_ids" not in back.vertex_attrs


def test_an_empty_vertex_tag_is_not_written_as_an_element_group(
    tmp_path: Path,
) -> None:
    poly = _mixed_mesh()
    poly.vertex_tags["none"] = np.array([], dtype=np.int32)
    poly.vertex_tags["outside"] = np.array([9], dtype=np.int32)
    poly.element_tags["empty"] = np.array([], dtype=np.int32)
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    messages = " ".join(str(w.message) for w in record)
    assert "['none', 'outside'] hold no node" in messages
    back = read(path)
    assert "none" not in back.element_tags and "outside" not in back.element_tags
    assert back.element_tags["empty"].tolist() == []


def test_an_empty_vertex_tag_sharing_an_element_group_is_silent(
    tmp_path: Path,
) -> None:
    poly = _mixed_mesh()
    poly.vertex_tags["solid"] = np.array([], dtype=np.int32)
    path = tmp_path / "m.unv"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    assert read(path).element_tags["solid"].tolist() == [0]


def test_an_unwritten_vertex_tag_neither_renames_nor_takes_a_name(
    tmp_path: Path,
) -> None:
    poly = dataclasses.replace(
        _mixed_mesh(),
        vertex_tags={" g ": np.array([], dtype=np.int32)},
        element_tags={"g ": np.array([0], dtype=np.int32)},
    )
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    messages = " ".join(str(w.message) for w in record)
    assert "[' g '] hold no node" in messages
    assert "'g ' as 'g'" in messages
    assert "' g ' as" not in messages
    assert list(read(path).element_tags) == ["g"]


def test_attributes_round_trip_as_analysis_data(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.vertex_attrs["temp"] = np.array([1.0, np.nan, 3.0, 4.0])
    poly.vertex_attrs["disp"] = np.arange(12.0).reshape(4, 3)
    poly.vertex_attrs["wave"] = np.array([1 + 2j, 3j, -1, 0])
    poly.element_attrs["stress"] = np.arange(4, dtype=np.int32)
    path = tmp_path / "m.unv"
    write(poly, path)
    back = read(path)
    np.testing.assert_array_equal(back.vertex_attrs["temp"], poly.vertex_attrs["temp"])
    np.testing.assert_array_equal(back.vertex_attrs["disp"], poly.vertex_attrs["disp"])
    np.testing.assert_array_equal(back.vertex_attrs["wave"], poly.vertex_attrs["wave"])
    np.testing.assert_array_equal(back.element_attrs["stress"], [0.0, 1.0, 2.0, 3.0])
    # A NaN node carries no record: that is how the format says "no data".
    temp = path.read_text().split("\ntemp\n", 1)[1].split("    -1", 1)[0]
    assert "\n         1\n" in temp and "\n         2\n" not in temp


def test_a_non_numeric_attribute_is_dropped_with_a_warning(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.vertex_attrs["label"] = np.array(["a", "b", "c", "d"])
    with pytest.warns(UserWarning, match=r"\['label'\] are not numeric"):
        write(poly, tmp_path / "m.unv")


def test_an_unwritten_element_leaves_its_group_and_attribute_rows(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        np.zeros((5, 3)),
        [("pyramid", np.array([[0, 1, 2, 3, 4]])), ("triangle", np.array([[0, 1, 2]]))],
        element_attrs={"e": np.array([1.0, 2.0])},
        element_tags={"all": np.array([0, 1], dtype=np.int32)},
    )
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    messages = " ".join(str(w.message) for w in record)
    assert "['pyramid'] have no universal file descriptor" in messages
    assert "1 tag member(s) name an element that was not written" in messages
    back = read(path)
    assert back.element_tags["all"].tolist() == [0]
    assert back.element_attrs["e"].tolist() == [2.0]


def test_a_wrong_node_count_is_dropped(tmp_path: Path) -> None:
    poly = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    poly = dataclasses.replace(
        poly, offsets=np.array([0, 3]), connectivity=poly.connectivity[:3]
    )
    with pytest.warns(UserWarning, match="does not match their type"):
        write(poly, tmp_path / "m.unv")


def test_a_dropped_element_s_attribute_rows_are_named_in_the_warning(
    tmp_path: Path,
) -> None:
    poly = make_polydata(
        _TET_VERTS,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("pyramid", np.array([[0, 1, 2, 3, 0]])),
        ],
    )
    poly.element_attrs["e"] = np.array([1.0, 2.0])
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match="dropped, with their element attribute rows"):
        write(poly, path)
    assert read(path).element_attrs["e"].tolist() == [1.0]


def test_nul_padding_is_not_part_of_a_name(tmp_path: Path) -> None:
    text = (
        _TET_FILE
        + _group(1, "skin\x00\x00\x00", [(7, 1)])
        + _analysis("temp\x00\x00", 1, 1, [f"{k:10d}\n{k:25.16E}" for k in range(1, 5)])
    )
    poly = read(_write(tmp_path, text))
    assert list(poly.vertex_tags) == ["skin"]
    assert "temp" in poly.vertex_attrs


def test_a_vertex_outside_the_mesh_is_refused(tmp_path: Path) -> None:
    poly = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    poly.connectivity[0] = 9
    with pytest.raises(CodecError, match="outside 0..3"):
        write(poly, tmp_path / "m.unv")


def test_a_tag_member_outside_the_mesh_is_dropped(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.vertex_tags["top"] = np.array([0, 99], dtype=np.int32)
    with pytest.warns(UserWarning, match="index an entity the mesh does not hold"):
        write(poly, tmp_path / "m.unv")


def test_a_name_on_two_lines_is_joined_with_a_warning(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.element_tags["two\nlines"] = poly.element_tags.pop("skin")
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match=r"'two\\nlines' as 'two lines'"):
        write(poly, path)
    assert "two lines" in read(path).element_tags


def test_a_name_the_reader_would_misread_is_renamed(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.element_tags["-1"] = poly.element_tags.pop("skin")
    poly.vertex_attrs["-1"] = np.arange(4.0)
    poly.vertex_attrs["none"] = np.arange(4.0)
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    message = " ".join(str(w.message) for w in record)
    assert "'-1' as '_-1'" in message and "'none' as '_none'" in message
    back = read(path)
    assert back.element_tags["_-1"].tolist() == [1, 2]
    assert sorted(back.vertex_attrs) == ["_-1", "_none"]


@pytest.mark.parametrize("brk", ["\v", "\f", "\x1c", "\x85", " "])
def test_every_line_break_the_reader_splits_at_is_joined(
    tmp_path: Path, brk: str
) -> None:
    poly = _mixed_mesh()
    poly.element_tags[f"a{brk}b"] = poly.element_tags.pop("skin")
    poly.vertex_attrs[f"t{brk}u"] = np.arange(4.0)
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match="as 'a b'"):
        write(poly, path)
    back = read(path)
    assert "a b" in back.element_tags and "t u" in back.vertex_attrs


def test_outer_blanks_are_warned_about_as_what_they_are(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.element_tags[" padded "] = poly.element_tags.pop("skin")
    poly.element_tags[""] = poly.element_tags.pop("solid")
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    message = " ".join(str(w.message) for w in record)
    assert "' padded ' as 'padded'" in message and "'' as 'unnamed'" in message
    assert "fit on one line" not in message


def test_two_tags_cleaning_to_one_name_stay_apart(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.element_tags["a\nb"] = poly.element_tags.pop("skin")
    poly.element_tags["a b"] = poly.element_tags.pop("solid")
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match=r"'a b' as 'a b_2'"):
        write(poly, path)
    back = read(path)
    assert back.element_tags["a b"].tolist() == [1, 2]
    assert back.element_tags["a b_2"].tolist() == [0]


def test_two_attributes_cleaning_to_one_name_stay_apart(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    n = len(poly.vertices)
    poly = dataclasses.replace(
        poly,
        vertex_attrs={"a": np.ones(n), "a ": np.full(n, 2.0)},
        element_attrs={"a": np.ones(len(poly.element_types))},
    )
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match=r"'a ' as 'a_2'"):
        write(poly, path)
    back = read(path)
    assert back.vertex_attrs["a"][0] == 1.0 and back.vertex_attrs["a_2"][0] == 2.0
    assert "a" in back.element_attrs and "a_2" not in back.element_attrs


def test_a_tag_and_an_attribute_renamed_apart_are_both_reported(
    tmp_path: Path,
) -> None:
    poly = _mixed_mesh()
    tags = dict(poly.element_tags)
    tags["x"] = tags.pop("skin")
    tags[" x"] = tags.pop("solid")
    poly = dataclasses.replace(
        poly,
        element_tags=tags,
        vertex_attrs={" x": np.ones(len(poly.vertices))},
    )
    with pytest.warns(UserWarning) as record:
        write(poly, tmp_path / "m.unv")
    message = " ".join(str(w.message) for w in record)
    assert "' x' as 'x_2'" in message and "' x' as 'x'" in message


def test_a_name_that_is_not_a_string_is_reported(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.element_tags[1] = poly.element_tags.pop("skin")
    poly = dataclasses.replace(poly, vertex_attrs={2: np.ones(len(poly.vertices))})
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning) as record:
        write(poly, path)
    message = " ".join(str(w.message) for w in record)
    assert "1 as '1'" in message and "2 as '2'" in message
    back = read(path)
    assert "1" in back.element_tags and "2" in back.vertex_attrs


def test_a_group_name_past_forty_columns_is_written_whole(tmp_path: Path) -> None:
    long_name = "g" * 40 + "_tail"
    poly = _mixed_mesh()
    poly.element_tags[long_name] = poly.element_tags.pop("skin")
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match="past the 40 columns") as record:
        write(poly, path)
    assert "two groups one name" not in str(record[0].message)
    assert read(path).element_tags[long_name].tolist() == [1, 2]


def test_long_group_names_alike_in_forty_columns_are_flagged(
    tmp_path: Path,
) -> None:
    poly = _mixed_mesh()
    poly.element_tags["g" * 40 + "_a"] = poly.element_tags.pop("skin")
    poly.element_tags["g" * 40 + "_b"] = poly.element_tags.pop("solid")
    with pytest.warns(UserWarning, match="two groups one name"):
        write(poly, tmp_path / "m.unv")


def test_a_group_name_of_forty_columns_is_silent(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly.element_tags["g" * 40] = poly.element_tags.pop("skin")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, tmp_path / "m.unv")


def test_an_attribute_cleaning_to_the_id_key_is_renamed(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    poly = dataclasses.replace(
        poly, vertex_attrs={"original_ids ": np.ones(len(poly.vertices))}
    )
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match=r"'original_ids ' as 'original_ids_2'"):
        write(poly, path)
    assert read(path).vertex_attrs["original_ids_2"][0] == 1.0


def test_an_attribute_of_no_entities_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(
        np.empty((0, 3)),
        [],
        vertex_attrs={"t": np.empty(0), "v": np.empty((0, 3))},
        element_attrs={"e": np.empty(0)},
    )
    path = tmp_path / "m.unv"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(poly, path)
    back = read(path)
    assert back.vertex_attrs["t"].shape == (0,)
    assert back.vertex_attrs["v"].shape == (0, 3)
    assert back.element_attrs["e"].shape == (0,)


def test_a_non_finite_coordinate_is_written_with_a_warning(tmp_path: Path) -> None:
    verts = _TET_VERTS.copy()
    verts[1, 2] = np.inf
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    path = tmp_path / "m.unv"
    with pytest.warns(UserWarning, match="non-finite"):
        write(poly, path)
    assert read(path).vertices[1, 2] == np.inf


def test_a_planar_mesh_is_lifted_with_a_zero_z(tmp_path: Path) -> None:
    poly = make_polydata(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
    )
    path = tmp_path / "m.unv"
    write(poly, path)
    np.testing.assert_array_equal(read(path).vertices[:, 2], 0.0)


def test_vertices_of_another_width_are_refused(tmp_path: Path) -> None:
    poly = make_polydata(_TET_VERTS, [("tetra", np.array([[0, 1, 2, 3]]))])
    poly = dataclasses.replace(poly, vertices=np.zeros((4, 4)))
    with pytest.raises(CodecError, match=r"expected \(n, 2\) or \(n, 3\)"):
        write(poly, tmp_path / "m.unv")


def test_unknown_write_options_warn(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unrecognized options"):
        write(_mixed_mesh(), tmp_path / "m.unv", precision=3)


def test_an_empty_mesh_round_trips(tmp_path: Path) -> None:
    poly = make_polydata(np.empty((0, 3)), [])
    path = tmp_path / "m.unv"
    write(poly, path)
    back = read(path)
    assert back.vertices.shape[0] == 0 and back.element_types.size == 0


def test_the_written_file_is_the_reference_layout(tmp_path: Path) -> None:
    path = tmp_path / "m.unv"
    write(_mixed_mesh(), path)
    lines = path.read_text().splitlines()
    assert lines[:4] == [
        "    -1",
        "  2411",
        "         1         1         1        11",
        "   0.0000000000000000D+00   0.0000000000000000D+00   0.0000000000000000D+00",
    ]
    i = lines.index("  2412")
    assert lines[i + 1 : i + 3] == [
        "         1       111         1         1         7         4",
        "         1         2         3         4",
    ]
    assert "  2477" in lines


def test_pyuff_reads_the_writer_output(tmp_path: Path) -> None:
    pyuff = pytest.importorskip("pyuff")
    poly = _mixed_mesh()
    poly.vertex_attrs["disp"] = np.arange(12.0).reshape(4, 3)
    poly.element_attrs["stress"] = np.array([0.5, 1.5, 2.5, 3.5])
    path = tmp_path / "m.unv"
    write(poly, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sets = pyuff.UFF(str(path)).read_sets()
    by_type = {}
    for dataset in sets:
        by_type.setdefault(dataset["type"], []).append(dataset)
    nodes = by_type[2411][0]
    np.testing.assert_array_equal(nodes["node_nums"], [1, 2, 3, 4])
    np.testing.assert_array_equal(
        np.column_stack([nodes["x"], nodes["y"], nodes["z"]]), _TET_VERTS
    )
    assert sorted(k for k in by_type[2412][0] if isinstance(k, int)) == [21, 91, 111]
    disp, stress = by_type[2414]
    np.testing.assert_array_equal(disp["data_at_node"], poly.vertex_attrs["disp"])
    np.testing.assert_array_equal(
        np.ravel(stress["data_at_element"]), poly.element_attrs["stress"]
    )


def test_integer_attributes_round_trip_exactly(tmp_path: Path) -> None:
    poly = _mixed_mesh()
    big = 2**62 + 1
    poly.vertex_attrs["id"] = np.array([big, -big, 7, 0], dtype=np.int64)
    poly.vertex_attrs["pair"] = np.arange(8, dtype=np.int32).reshape(4, 2)
    poly.vertex_attrs["flag"] = np.array([True, False, True, True])
    poly.element_attrs["mat"] = np.array([1, 2, 3, 4], dtype=np.uint16)
    poly.element_attrs["huge"] = np.array([2**64 - 1, 0, 1, 2], dtype=np.uint64)
    path = tmp_path / "m.unv"
    write(poly, path)
    back = read(path)
    for key in ("id", "pair", "flag"):
        assert back.vertex_attrs[key].dtype == np.int64
        np.testing.assert_array_equal(back.vertex_attrs[key], poly.vertex_attrs[key])
    assert back.element_attrs["mat"].dtype == np.int64
    np.testing.assert_array_equal(back.element_attrs["mat"], [1, 2, 3, 4])
    assert back.element_attrs["huge"].dtype == np.float64
    np.testing.assert_array_equal(back.element_attrs["huge"], [2.0**64, 0, 1, 2])


def test_pyuff_reads_integer_attributes(tmp_path: Path) -> None:
    pyuff = pytest.importorskip("pyuff")
    poly = _mixed_mesh()
    poly.vertex_attrs["id"] = np.array([10, -20, 300, 4000], dtype=np.int64)
    path = tmp_path / "m.unv"
    write(poly, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sets = pyuff.UFF(str(path)).read_sets()
    (result,) = [s for s in sets if s["type"] == 2414]
    np.testing.assert_array_equal(
        np.ravel(result["data_at_node"]), poly.vertex_attrs["id"]
    )


@pytest.mark.parametrize(
    ("width", "characteristic", "result_type"),
    [(1, 1, 94), (3, 2, 95), (6, 0, 93), (2, 0, 93)],
)
def test_an_attribute_is_marked_scalar_vector_or_unknown(
    tmp_path: Path, width: int, characteristic: int, result_type: int
) -> None:
    poly = _mixed_mesh()
    poly.vertex_attrs["sym"] = np.arange(4.0 * width).reshape(4, width)
    path = tmp_path / "m.unv"
    write(poly, path)
    lines = path.read_text().splitlines()
    start = lines.index("  2414")
    header = [int(v) for v in lines[start + 9].split()]
    assert header[2:] == [characteristic, result_type, 4, width]
    back = read(path).vertex_attrs["sym"]
    np.testing.assert_array_equal(back.reshape(4, width), poly.vertex_attrs["sym"])


def test_a_malformed_dataset_55_names_the_line_its_data_starts(
    tmp_path: Path,
) -> None:
    text = _TET_FILE + (
        "    -1\n    55\nmode shape\nNONE\nNONE\nNONE\nNONE\n"
        "         1         2         2         8         2         3\n"
        "         2         4         1         1\n"
        "  1.00000e+01  1.00000e+00  0.00000e+00  0.00000e+00\n"
        "         2\n  1.00000e+00  5.00000e+00  9.00000e+00\n"
        "         4\n  4.00000e+00  8.00000e+00\n"
        "    -1\n"
    )
    assert text.splitlines()[26].strip() == "2"
    with pytest.raises(CodecError, match="'mode shape' at line 27 holds 7 value"):
        read(_write(tmp_path, text))
