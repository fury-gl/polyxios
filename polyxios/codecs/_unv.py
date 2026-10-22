"""I-DEAS universal file codec (ASCII ``.unv``) - read + write.

A universal file is a run of datasets, each opened and closed by a line
holding ``-1`` in its first six columns, the opening one followed by the
dataset number. The mesh lives in three of them: 2411 lists nodes as a label
record and a coordinate record, 2412 lists elements as a label record naming
an FE descriptor and a node count, then the node labels, and 2467 / 2477
name permanent groups of nodes and elements. Results are 55 (data at nodes)
and 2414 (analysis data at nodes or on elements). Every other dataset -
header, units, coordinate systems, function data - is passed over.

Nodes and elements are numbered freely, so both are renumbered on the way in
and the file's own labels are kept under ``original_ids`` - see
:mod:`polyxios._ids`. A group becomes a vertex tag for the nodes it names and
an element tag for the elements, both under the group's name. A result
becomes a vertex or element attribute named after its dataset.

The FE descriptor is a solver class rather than a geometry - a thin shell, a
membrane and a plane-stress triangle are all triangles - so a descriptor is
read as its family and the node count picks the geometry inside it. Second
order elements list their nodes around the element, corner and mid-edge
alternating, where polyxios lists corners first; the reader and writer
permute between the two, and the permutations agree with a reference
implementation's.
"""

from collections.abc import Container
import re
from typing import Any, NamedTuple
import warnings

import numpy as np

from polyxios._dimension import pad_to_3d
from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    MAX_SAFE_CONN,
    MAX_SAFE_ELEMENTS,
    MAX_SAFE_VERTICES,
    NODES_PER_ELEMENT,
)
from polyxios._ids import IDS_KEY, ids_for_write, record_ids
from polyxios._io import Source, read_text, write_text
from polyxios._tags import member_values
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".unv"

_NODES_DOUBLE: frozenset[int] = frozenset({2411, 781})
_NODES_SINGLE: int = 15
_ELEMENTS: int = 2412
# The permanent group datasets. All four spell an entity as four integers -
# type, label and two leaf ids - after a header ending in the member count.
_GROUPS: frozenset[int] = frozenset({2467, 2477, 2435, 2452})
_DATA_AT_NODES: int = 55
_ANALYSIS_DATA: int = 2414
# Datasets that carry mesh or result data this codec does not read, and so
# are worth a warning rather than the silence every other skipped one gets.
_UNREAD_DATA: dict[int, str] = {
    56: "data at elements",
    57: "data at nodes on elements",
    780: "elements in the pre-2412 layout",
    2417: "groups in the pre-2435 layout",
    2429: "groups in the pre-2435 layout",
    2430: "groups in the pre-2435 layout",
    2432: "groups in the pre-2435 layout",
}

# Group entity type codes this codec maps; the rest name coordinate systems,
# loads and the like, which a mesh has nothing to hold them in.
_GROUP_NODE: int = 7
_GROUP_ELEMENT: int = 8

# FE descriptor -> family. The node count picks the geometry in the family,
# which is how one table covers the plane stress, plane strain, plate,
# membrane, axisymmetric and shell variants of every surface element.
_BEAM_FAMILY: str = "edge"
_FAMILY: dict[int, str] = {
    **dict.fromkeys((11, 21, 22, 23, 24, 31, 32), _BEAM_FAMILY),
    **dict.fromkeys(
        (41, 42, 44, 45, 51, 52, 54, 55, 61, 62, 64, 65, 71, 72, 74, 75)
        + (81, 82, 84, 85, 91, 92, 94, 95),
        "surface",
    ),
    111: "tetra",
    118: "tetra",
    112: "wedge",
    113: "wedge",
    115: "hexa",
    116: "hexa",
    161: "point",
}
_GEOMETRY: dict[tuple[str, int], str] = {
    ("edge", 2): "line",
    ("edge", 3): "quadratic_edge",
    ("surface", 3): "triangle",
    ("surface", 6): "quadratic_triangle",
    ("surface", 4): "quad",
    ("surface", 8): "quadratic_quad",
    ("tetra", 4): "tetra",
    ("tetra", 10): "quadratic_tetra",
    ("wedge", 6): "wedge",
    ("wedge", 15): "quadratic_wedge",
    ("hexa", 8): "hexahedron",
    ("hexa", 20): "quadratic_hexahedron",
    ("point", 1): "vertex",
}
# A beam descriptor's element is followed by an orientation record before its
# node labels. The range is the one-dimensional block of the descriptor table.
_BEAM_RANGE: range = range(11, 33)

# Geometry -> the descriptor written: the thin shell for a surface and the
# solid for a volume, which is what every reader of the format maps.
_WRITE_DESCRIPTOR: dict[str, int] = {
    "vertex": 161,
    "line": 21,
    "quadratic_edge": 24,
    "triangle": 91,
    "quadratic_triangle": 92,
    "quad": 94,
    "quadratic_quad": 95,
    "tetra": 111,
    "quadratic_tetra": 118,
    "wedge": 112,
    "quadratic_wedge": 113,
    "hexahedron": 115,
    "quadratic_hexahedron": 116,
}

# polyxios position i holds the universal file's node PERM[i]: corners first
# in polyxios, corner and mid-edge alternating around each face here.
_PERM: dict[str, tuple[int, ...]] = {
    "quadratic_edge": (0, 2, 1),
    "quadratic_triangle": (0, 2, 4, 1, 3, 5),
    "quadratic_quad": (0, 2, 4, 6, 1, 3, 5, 7),
    "quadratic_tetra": (0, 2, 4, 9, 1, 3, 5, 6, 7, 8),
    "quadratic_wedge": (0, 2, 4, 9, 11, 13, 1, 3, 5, 10, 12, 14, 6, 7, 8),
    "quadratic_hexahedron": (
        (0, 2, 4, 6, 12, 14, 16, 18, 1, 3, 5, 7, 13, 15, 17, 19) + (8, 9, 10, 11)
    ),
}
_INV_PERM: dict[str, np.ndarray] = {
    name: np.argsort(np.asarray(perm)) for name, perm in _PERM.items()
}

# Result data type codes: integer, single and double real, single and double
# complex. Complex values are spelled as a real and an imaginary part.
_INTEGER: int = 1
_REAL: frozenset[int] = frozenset({2, 4})
_COMPLEX: frozenset[int] = frozenset({5, 6})
_DATA_TYPES: frozenset[int] = frozenset({_INTEGER}) | _REAL | _COMPLEX
_AT_NODES: int = 1
_ON_ELEMENTS: int = 2

# Data characteristic and result type written for a value count, the unknown
# kinds the format reserves for data it cannot name. Six values are left
# unknown: the 6 DOF characteristic would show a tensor as a rotation.
_CHARACTERISTIC: dict[int, int] = {1: 1, 3: 2}
_RESULT_TYPE: dict[int, int] = {1: 94, 3: 95}
_UNKNOWN_RESULT: int = 93

# What an integer field is spelled as. ``int`` alone also takes ``1_0`` and
# the digits of other scripts, neither of which a universal file holds.
_INT_RE = re.compile(r"[-+]?[0-9]+")
# A line of integer fields every one of which int64 holds, eighteen digits
# being the most that always fit; a line this does not match is converted
# field by field, which names the field at fault.
_INT_LINE_RE = re.compile(r"\s*[-+]?[0-9]{1,18}(?:\s+[-+]?[0-9]{1,18})*\s*")
# A Fortran real whose three-digit exponent pushed the ``E`` out of its
# field: ``1.00000-100``.
_BARE_EXPONENT_RE = re.compile(r"([0-9.])([-+][0-9]+)$")
_INT64 = np.iinfo(np.int64)
_EXACT_IN_DOUBLE: int = 2**53
_FORTRAN_EXPONENT = str.maketrans("Dd", "Ee")
# Labels are I10 fields with no separator, so a tenth digit would run into
# the field before it.
_LABEL_LIMIT: int = 999_999_999

# Node labels per line of an element record, and values per line of a
# double precision result record, as the format's own layout spells them.
_LABELS_PER_LINE: int = 8
_VALUES_PER_LINE: int = 3
_INTEGERS_PER_LINE: int = 6
_INTEGER_WIDTH: int = 10
_GROUP_MEMBERS_PER_LINE: int = 2
# A group name is a 20A2 record: forty columns to a reader that holds to
# the layout, while gmsh and this codec read the whole line.
_GROUP_NAME_WIDTH: int = 40

# Nodes per polyxios element code for the geometries this codec writes, -1
# for the rest, so a mesh's whole element table is checked in one indexing.
_WRITE_WIDTH: np.ndarray = np.full(256, -1, dtype=np.int64)
for _name in _WRITE_DESCRIPTOR:
    _WRITE_WIDTH[ELEMENT_TYPES[_name]] = NODES_PER_ELEMENT[_name]
del _name


def _is_delimiter(line: str) -> bool:
    """Whether a line is the ``-1`` that opens or closes a dataset.

    The delimiter is an ``I6`` field, so it ends by column six; a ``-1``
    further right is an integer field of a record, such as an ``I10``.
    Trailing blanks are allowed, since some writers pad every line to 80
    columns.
    """
    stripped = line.rstrip()
    return len(stripped) <= 6 and stripped.lstrip() == "-1"


def _datasets(lines: list[str]) -> list[tuple[int, int, int, list[str]]]:
    """Split the file into ``(number, line no, first body line no, body)``.

    Line numbers are one-based, so an error names the line of the file.
    """
    out: list[tuple[int, int, int, list[str]]] = []
    delims = [k for k, line in enumerate(lines) if _is_delimiter(line)]
    pos = 0
    i = 0
    while i < len(delims):
        opening = delims[i]
        for k in range(pos, opening):
            if lines[k].strip():
                raise CodecError(
                    f"{EXTENSION}: line {k + 1} lies outside every dataset:"
                    f" {lines[k]!r}."
                )
        if i + 1 == len(delims):
            raise CodecError(
                f"{EXTENSION}: the dataset opened at line {opening + 1} has no"
                " closing -1; the file is cut short."
            )
        closing = delims[i + 1]
        if closing == opening + 1:
            raise CodecError(
                f"{EXTENSION}: the dataset opened at line {opening + 1} holds no"
                " dataset number."
            )
        head = lines[opening + 1].split()
        token = head[0] if head else ""
        if token[-1:] in ("b", "B") and _INT_RE.fullmatch(token[:-1]):
            raise CodecError(
                f"{EXTENSION}: dataset {token[:-1]} at line {opening + 2} is"
                " binary; only ASCII datasets are read."
            )
        if not _INT_RE.fullmatch(token):
            raise CodecError(
                f"{EXTENSION}: line {opening + 2} should hold a dataset number,"
                f" not {lines[opening + 1]!r}."
            )
        body = lines[opening + 2 : closing]
        out.append((int(token), opening + 2, opening + 3, body))
        pos = closing + 1
        i += 2
    for k in range(pos, len(lines)):
        if lines[k].strip():
            raise CodecError(
                f"{EXTENSION}: line {k + 1} lies outside every dataset: {lines[k]!r}."
            )
    return out


def _record_name(line: str) -> str:
    """Return the name a text record spells, NUL padding and blanks stripped.

    A writer in C may pad a fixed-width name record with NUL rather than
    the blanks Fortran pads it with; neither is part of the name.
    """
    return line.replace("\x00", " ").strip()


def _int(tok: str, no: int, what: str) -> int:
    """Convert one integer field, refusing what int64 cannot hold."""
    if not _INT_RE.fullmatch(tok):
        raise CodecError(f"{EXTENSION}: malformed {what} {tok!r} on line {no}.")
    value = int(tok)
    if not _INT64.min <= value <= _INT64.max:
        raise CodecError(
            f"{EXTENSION}: {what} {tok!r} on line {no} is wider than 64 bits."
        )
    return value


def _ints(line: str, no: int, what: str) -> list[int]:
    """Convert every field of an integer record."""
    return [_int(tok, no, what) for tok in line.split()]


def _float(tok: str) -> float:
    """Convert one real, accepting the Fortran ``D`` and a bare exponent."""
    spelled = tok.translate(_FORTRAN_EXPONENT)
    try:
        return float(spelled)
    except ValueError:
        return float(_BARE_EXPONENT_RE.sub(r"\1E\2", spelled))


def _check_spelling(tokens: list[str], first_no: int, what: str) -> None:
    """Refuse a field spelled with ``_`` or a non-ASCII character.

    numpy and Python take ``1_0`` and the digits of other scripts as
    numbers, which a universal file never spells, so they are refused as
    :func:`_int` refuses them, with the whole run checked in one pass.
    """
    text = "".join(tokens)
    if text.isascii() and "_" not in text:
        return
    for tok in tokens:
        if not tok.isascii() or "_" in tok:
            raise CodecError(
                f"{EXTENSION}: malformed {what} {tok!r} in the record starting"
                f" at line {first_no}."
            )


def _floats(tokens: list[str], first_no: int, what: str) -> np.ndarray:
    """Convert a run of reals in one numpy pass, one at a time on failure.

    ``first_no`` is the line the run starts on; the slow path is the only
    one that can say which token is wrong, and it names that one. A run
    spelled with the Fortran ``D`` exponent, as this codec writes
    coordinates, is translated and converted in one pass before that.
    """
    _check_spelling(tokens, first_no, what)
    try:
        return np.array(tokens, dtype=np.float64)
    except ValueError:
        pass
    try:
        return np.array(
            " ".join(tokens).translate(_FORTRAN_EXPONENT).split(), dtype=np.float64
        )
    except ValueError:
        pass
    out = np.empty(len(tokens), dtype=np.float64)
    for k, tok in enumerate(tokens):
        try:
            out[k] = _float(tok)
        except ValueError as exc:
            raise CodecError(
                f"{EXTENSION}: malformed {what} {tok!r} in the record starting"
                f" at line {first_no}."
            ) from exc
    return out


def _whole(tokens: list[str], first_no: int, what: str, name: str) -> np.ndarray:
    """Convert a run of integer fields to int64 without passing through float.

    A field spelled as a real is still taken when it holds a whole number,
    as long as a double holds it exactly.
    """
    _check_spelling(tokens, first_no, what)
    try:
        return np.array(tokens, dtype=np.int64)
    except (ValueError, OverflowError):
        pass
    values = _floats(tokens, first_no, what)
    exact = np.isfinite(values) & (np.abs(values) <= _EXACT_IN_DOUBLE)
    whole = np.where(exact, values, 0).astype(np.int64)
    if not (exact.all() and np.array_equal(whole, values)):
        raise CodecError(
            f"{EXTENSION}: result {name!r} at line {first_no} holds a {what}"
            " that is not a whole number."
        )
    return whole


def _sorted_ids(ids: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted ids, their positions)`` for :func:`_index_of`."""
    arr = np.asarray(ids, dtype=np.int64)
    order = np.argsort(arr, kind="stable")
    return arr[order], order


def _index_of(table: tuple[np.ndarray, np.ndarray], wanted: np.ndarray) -> np.ndarray:
    """Return the position of each ``wanted`` label in the table, -1 if absent."""
    sorted_ids, order = table
    if sorted_ids.size == 0:
        return np.full(wanted.shape, -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(sorted_ids, wanted), sorted_ids.size - 1)
    return np.where(sorted_ids[pos] == wanted, order[pos], -1)


def _first_repeat(table: tuple[np.ndarray, np.ndarray]) -> tuple[int, int] | None:
    """Return ``(first position, repeat position)`` of the earliest repeat."""
    sorted_ids, order = table
    hits = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    if hits.size == 0:
        return None
    j = hits[np.argmin(order[hits + 1])]
    return int(order[j]), int(order[j + 1])


class _Result(NamedTuple):
    """One result dataset, its labels still the file's own."""

    kind: str
    name: str
    labels: np.ndarray
    values: np.ndarray


class _Reader:
    """The state one pass over the datasets accumulates.

    Labels are resolved only once every dataset has been seen: a group or a
    result may name a node the file declares later.
    """

    def __init__(self) -> None:
        self.node_ids: list[np.ndarray] = []
        self.node_xyz: list[np.ndarray] = []
        self.node_line_nos: list[np.ndarray] = []
        self.elem_ids: list[int] = []
        self.elem_line_nos: list[int] = []
        # Node labels as spelled, every line already checked to hold int64
        # fields, so the whole table converts in one pass at the end.
        self.elem_nodes: list[str] = []
        self.elem_widths: list[int] = []
        self.elem_codes: list[int] = []
        self.node_groups: dict[str, list[int]] = {}
        self.elem_groups: dict[str, list[int]] = {}
        self.results: list[_Result] = []
        self.skipped_types: dict[str, int] = {}
        self.unread: dict[str, int] = {}
        # Placeholder keys of the groups with no name, with their numbers;
        # named only once every group is seen, so a later group spelling
        # ``group_<n>`` cannot fold into one of them.
        self.unnamed_groups: list[tuple[str, int]] = []
        self.empty_groups: set[str] = set()
        self.uneven_results: list[str] = []
        self.unknown_types: dict[int, int] = {}
        self.skipped_elem_ids: list[int] = []
        self.skipped_elem_line_nos: list[int] = []
        self.repeated_results: list[str] = []
        self.inexact_results: list[str] = []

    def read_nodes(self, body: list[str], first: int) -> None:
        if len(body) % 2:
            raise CodecError(
                f"{EXTENSION}: the node dataset starting at line {first} holds"
                f" {len(body)} lines; every node is two records."
            )
        labels = body[0::2]
        coords = body[1::2]
        ids = np.empty(len(labels), dtype=np.int64)
        for k, line in enumerate(labels):
            head = line.split(None, 1)
            if not head:
                raise CodecError(
                    f"{EXTENSION}: node record at line {first + 2 * k} is empty."
                )
            ids[k] = _int(head[0], first + 2 * k, "node label")
        rows = [line.split() for line in coords]
        for k, row in enumerate(rows):
            if len(row) != 3:
                raise CodecError(
                    f"{EXTENSION}: node coordinate record at line"
                    f" {first + 2 * k + 1} carries {len(row)} value(s),"
                    " expected 3."
                )
        flat = [tok for row in rows for tok in row]
        xyz = _floats(flat, first, "coordinate").reshape(-1, 3)
        self._add_nodes(ids, xyz, first + 2 * np.arange(len(labels)))

    def read_single_nodes(self, body: list[str], first: int) -> None:
        ids = np.empty(len(body), dtype=np.int64)
        flat: list[str] = []
        for k, line in enumerate(body):
            row = line.split()
            if len(row) != 7:
                raise CodecError(
                    f"{EXTENSION}: node record at line {first + k} carries"
                    f" {len(row)} value(s), expected 7."
                )
            ids[k] = _int(row[0], first + k, "node label")
            flat.extend(row[4:])
        xyz = _floats(flat, first, "coordinate").reshape(-1, 3)
        self._add_nodes(ids, xyz, first + np.arange(len(body)))

    def _add_nodes(
        self, ids: np.ndarray, xyz: np.ndarray, line_nos: np.ndarray
    ) -> None:
        bad = np.flatnonzero(ids < 1)
        if bad.size:
            k = int(bad[0])
            raise CodecError(
                f"{EXTENSION}: node label {int(ids[k])} at line {int(line_nos[k])}"
                " is not positive; the format numbers from 1."
            )
        self.node_ids.append(ids)
        self.node_xyz.append(xyz)
        self.node_line_nos.append(line_nos)

    def read_elements(self, body: list[str], first: int) -> None:
        k = 0
        n = len(body)
        while k < n:
            no = first + k
            line = body[k]
            if _INT_LINE_RE.fullmatch(line):
                rec = [int(tok) for tok in line.split()]
            else:
                rec = _ints(line, no, "element field")
            if len(rec) != 6:
                raise CodecError(
                    f"{EXTENSION}: element record at line {no} carries"
                    f" {len(rec)} field(s), expected 6."
                )
            label, descriptor, _, _, _, count = rec
            if label < 1:
                raise CodecError(
                    f"{EXTENSION}: element label {label} at line {no} is not"
                    " positive; the format numbers from 1."
                )
            k += 1
            if descriptor in _BEAM_RANGE:
                if k >= n:
                    raise CodecError(
                        f"{EXTENSION}: beam element {label} at line {no} has no"
                        " orientation record; the dataset is cut short."
                    )
                k += 1
            # Each line holds at most eight labels, so a count past what the
            # lines left could spell is refused before anything is gathered.
            if count < 1 or count > _LABELS_PER_LINE * (n - k):
                raise CodecError(
                    f"{EXTENSION}: element {label} at line {no} declares"
                    f" {count} node(s), which the dataset cannot hold."
                )
            nodes: list[str] = []
            node_start = k
            while len(nodes) < count:
                if k >= n:
                    raise CodecError(
                        f"{EXTENSION}: element {label} at line {no} lists"
                        f" {len(nodes)} of its {count} node(s); the dataset"
                        " is cut short."
                    )
                nodes.extend(body[k].split())
                k += 1
            if len(nodes) != count:
                raise CodecError(
                    f"{EXTENSION}: element {label} at line {no} lists"
                    f" {len(nodes)} node(s) where it declares {count}."
                )
            family = _FAMILY.get(descriptor)
            name = None if family is None else _GEOMETRY.get((family, count))
            if name is None:
                key = f"{descriptor} ({count} nodes)"
                self.skipped_types[key] = self.skipped_types.get(key, 0) + 1
                self.skipped_elem_ids.append(label)
                self.skipped_elem_line_nos.append(no)
                continue
            for j in range(node_start, k):
                if not _INT_LINE_RE.fullmatch(body[j]):
                    _ints(body[j], first + j, "node label")
            perm = _PERM.get(name)
            labels = nodes if perm is None else [nodes[p] for p in perm]
            self.elem_ids.append(label)
            self.elem_line_nos.append(no)
            self.elem_nodes.extend(labels)
            self.elem_widths.append(count)
            self.elem_codes.append(ELEMENT_TYPES[name])

    def read_group(self, number: int, body: list[str], first: int) -> None:
        k = 0
        n = len(body)
        while k < n:
            no = first + k
            rec = _ints(body[k], no, "group field")
            if len(rec) < 2:
                raise CodecError(
                    f"{EXTENSION}: group record at line {no} carries"
                    f" {len(rec)} field(s); it should end in a member count."
                )
            count = rec[-1]
            if k + 1 >= n:
                raise CodecError(f"{EXTENSION}: group at line {no} has no name record.")
            # Two members to a line, so a count past what the lines left
            # could spell is refused before anything is gathered.
            if count < 0 or count > _GROUP_MEMBERS_PER_LINE * (n - k - 2):
                raise CodecError(
                    f"{EXTENSION}: group at line {no} declares {count}"
                    " member(s), which the dataset cannot hold."
                )
            name = _record_name(body[k + 1])
            shown = name or f"group_{rec[0]}"
            if not name:
                # A spelled name is stripped, so one opening with a blank is
                # never a file's own.
                name = f" {len(self.unnamed_groups)}"
                self.unnamed_groups.append((name, rec[0]))
            k += 2
            fields: list[int] = []
            while len(fields) < 4 * count:
                if k >= n:
                    raise CodecError(
                        f"{EXTENSION}: group {shown!r} at line {no} lists"
                        f" {len(fields) // 4} of its {count} member(s); the"
                        " dataset is cut short."
                    )
                fields.extend(_ints(body[k], first + k, "group member"))
                k += 1
            if len(fields) != 4 * count:
                raise CodecError(
                    f"{EXTENSION}: group {shown!r} at line {no} spells"
                    f" {len(fields)} member field(s) where it declares {count}"
                    " member(s) of four."
                )
            if not count:
                self.empty_groups.add(name)
            members = np.asarray(fields, dtype=np.int64).reshape(-1, 4)
            kinds = members[:, 0]
            nodes = self.node_groups.setdefault(name, [])
            nodes.extend(members[kinds == _GROUP_NODE, 1].tolist())
            elems = self.elem_groups.setdefault(name, [])
            elems.extend(members[kinds == _GROUP_ELEMENT, 1].tolist())
            for kind in np.unique(
                kinds[(kinds != _GROUP_NODE) & (kinds != _GROUP_ELEMENT)]
            ):
                self.unknown_types[int(kind)] = self.unknown_types.get(int(kind), 0) + 1

    def read_data_at_nodes(self, body: list[str], first: int) -> None:
        if len(body) < 7:
            raise CodecError(
                f"{EXTENSION}: the data at nodes dataset starting at line"
                f" {first} is cut short in its header."
            )
        name = _record_name(body[0])
        if not name or name.upper() == "NONE":
            name = "data_at_nodes"
        rec = _ints(body[5], first + 5, "data header field")
        if len(rec) != 6:
            raise CodecError(
                f"{EXTENSION}: data header at line {first + 5} carries"
                f" {len(rec)} field(s), expected 6."
            )
        data_type, width = rec[4], rec[5]
        tokens = " ".join(body[6:]).split()
        if len(tokens) < 2:
            raise CodecError(
                f"{EXTENSION}: data at nodes at line {first + 6} names no"
                " analysis-specific value counts."
            )
        n_int = _int(tokens[0], first + 6, "integer value count")
        n_real = _int(tokens[1], first + 6, "real value count")
        skip = 2 + n_int + n_real
        if n_int < 0 or n_real < 0 or skip > len(tokens):
            raise CodecError(
                f"{EXTENSION}: data at nodes at line {first + 6} declares"
                f" {n_int} integer and {n_real} real header value(s), which"
                " the dataset cannot hold."
            )
        # The line the first data record opens, past the header tokens.
        ends = np.cumsum([len(line.split()) for line in body[6:]])
        start = first + 6 + int(np.searchsorted(ends, skip, side="right"))
        self._add_result(
            "vertex", name, data_type, width, tokens[skip:], start, label_fields=1
        )

    def read_analysis_data(self, body: list[str], first: int) -> None:
        if len(body) < 11:
            raise CodecError(
                f"{EXTENSION}: the analysis dataset starting at line {first} is"
                " cut short in its header."
            )
        label = _ints(body[0], first, "dataset label")
        name = _record_name(body[1])
        if not name or name.upper() == "NONE":
            name = _record_name(body[3])
        if not name or name.upper() == "NONE":
            name = f"result_{label[0] if label else 0}"
        location = _ints(body[2], first + 2, "dataset location")
        rec = _ints(body[8], first + 8, "data header field")
        if len(rec) != 6 or len(location) != 1:
            raise CodecError(
                f"{EXTENSION}: the analysis dataset at line {first} has a"
                " malformed location or data header record."
            )
        data_type, width = rec[4], rec[5]
        # Records 12 and 13 are twelve reals, two lines of six in single
        # precision and more lines in double, so they are counted by value.
        k = 11
        reals = 0
        while reals < 12:
            if k >= len(body):
                raise CodecError(
                    f"{EXTENSION}: the analysis dataset at line {first} ends"
                    " inside its twelve analysis-specific reals."
                )
            reals += len(body[k].split())
            k += 1
        if reals != 12:
            raise CodecError(
                f"{EXTENSION}: the analysis dataset at line {first} spells"
                f" {reals} analysis-specific reals where the format has twelve."
            )
        if location[0] == _AT_NODES:
            self._add_result(
                "vertex",
                name,
                data_type,
                width,
                " ".join(body[k:]).split(),
                first + k,
                label_fields=1,
            )
        elif location[0] == _ON_ELEMENTS:
            self._add_result(
                "element",
                name,
                data_type,
                width,
                " ".join(body[k:]).split(),
                first + k,
                label_fields=2,
            )
        else:
            key = f"2414 location {location[0]}"
            self.unread[key] = self.unread.get(key, 0) + 1

    def _add_result(
        self,
        kind: str,
        name: str,
        data_type: int,
        width: int,
        tokens: list[str],
        first: int,
        *,
        label_fields: int,
    ) -> None:
        """Turn a result's data records into labels and a value array.

        A record is a label - and on elements a value count - followed by
        the values, so a run where every count agrees is one reshape; one
        that varies is walked record by record.
        """
        if data_type not in _DATA_TYPES:
            key = f"result data type {data_type}"
            self.unread[key] = self.unread.get(key, 0) + 1
            return
        parts = 2 if data_type in _COMPLEX else 1
        if label_fields == 2 and len(tokens) > 1:
            # On elements each record says how many values it holds - several
            # layers through a shell's thickness, say - so the first record's
            # count stands in for the header's per-component one.
            width = _int(tokens[1], first, "element value count")
        if width < 1 or (tokens and width * parts > len(tokens)):
            raise CodecError(
                f"{EXTENSION}: result {name!r} at line {first} declares {width}"
                " value(s) per entity, which the dataset cannot hold."
            )
        stride = label_fields + width * parts
        uniform = len(tokens) % stride == 0
        if uniform and label_fields == 2:
            counts = _whole(tokens[1::stride], first, "value count", name)
            uniform = bool(np.all(counts == width))
        if not uniform:
            if label_fields == 1:
                raise CodecError(
                    f"{EXTENSION}: result {name!r} at line {first} holds"
                    f" {len(tokens)} value(s), not a whole number of"
                    f" {stride}-value records."
                )
            self.uneven_results.append(name)
            return
        # Labels and integer values are parsed as integers: through a double
        # they would round past 2**53.
        labels = _whole(tokens[0::stride], first, "entity label", name)
        if data_type == _INTEGER:
            rows = _whole(tokens, first, "result value", name).reshape(-1, stride)
            data = rows[:, label_fields:]
        else:
            rows = _floats(tokens, first, "result value").reshape(-1, stride)
            data = rows[:, label_fields:]
            if parts == 2:
                data = data[:, 0::2] + 1j * data[:, 1::2]
        self.results.append(_Result(kind, name, labels, data))


def _resolve_groups(
    groups: dict[str, list[int]],
    table: tuple[np.ndarray, np.ndarray],
    *,
    empty: frozenset[str] = frozenset(),
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Turn groups of file labels into tags of mesh indices.

    A group naming no entity of this kind is not a tag of it: a group of
    elements alone has no vertex tag. The names in ``empty`` are groups
    declaring no member at all, which are kept as empty tags of this kind.
    The labels naming no entity of the table are handed back, so the
    caller can say why each was lost.
    """
    tags: dict[str, np.ndarray] = {}
    lost: list[np.ndarray] = []
    for name, members in groups.items():
        labels = np.asarray(members, dtype=np.int64)
        found = _index_of(table, labels)
        kept = found[found >= 0]
        lost.append(labels[found < 0])
        if not kept.size and name not in empty:
            continue
        _, first = np.unique(kept, return_index=True)
        tags[name] = kept[np.sort(first)].astype(np.int32, copy=False)
    return tags, np.concatenate(lost) if lost else np.empty(0, np.int64)


def _name_unnamed_groups(
    groups: dict[str, list[int]],
    unnamed: list[tuple[str, int]],
) -> dict[str, str]:
    """Map each unnamed group's placeholder key to ``group_<number>``.

    Group numbers are per dataset, so two unnamed groups can share one;
    each takes ``_2``, ``_3`` past any name a group of the file spells.
    """
    placeholders = {key for key, _ in unnamed}
    taken = {name for name in groups if name not in placeholders}
    names: dict[str, str] = {}
    for key, number in unnamed:
        names[key] = _unique_key(f"group_{number}", taken)
        taken.add(names[key])
    return names


def _renamed(
    groups: dict[str, list[int]], names: dict[str, str]
) -> dict[str, list[int]]:
    """Return ``groups`` in their order, with the keys in ``names`` renamed."""
    return {names.get(key, key): members for key, members in groups.items()}


def _unique_key(name: str, taken: Container[str]) -> str:
    """Return ``name``, suffixed with ``_<n>`` until ``taken`` holds no such key."""
    candidate, k = name, 1
    while candidate in taken:
        k += 1
        candidate = f"{name}_{k}"
    return candidate


def _attach_results(
    results: list[_Result],
    kind: str,
    table: tuple[np.ndarray, np.ndarray],
    count: int,
    attrs: dict[str, np.ndarray],
    repeated: list[str],
    inexact: list[str],
) -> np.ndarray:
    """Place each result of one kind as an attribute, returning labels lost.

    An entity no record names has no data, which the format distinguishes
    from a zero: it reads as NaN, and an integer result missing one is
    widened to float to hold it, its name added to ``inexact`` when a value
    is past the 2**53 a double holds exactly. An entity named by more than
    one record takes the last, and the result's name is added to
    ``repeated``.
    """
    lost: list[np.ndarray] = []
    for result in results:
        if result.kind != kind:
            continue
        idx = _index_of(table, result.labels)
        hit = idx >= 0
        lost.append(result.labels[~hit])
        data = result.values[hit]
        idx = idx[hit]
        targets, last = np.unique(idx[::-1], return_index=True)
        if targets.size < idx.size:
            repeated.append(result.name)
            keep = np.sort(idx.size - 1 - last)
            idx, data = idx[keep], data[keep]
        if targets.size == count:
            out = np.empty((count, data.shape[1]), dtype=data.dtype)
        elif data.dtype.kind == "c":
            out = np.full(
                (count, data.shape[1]), complex(np.nan, np.nan), dtype=data.dtype
            )
        else:
            out = np.full((count, data.shape[1]), np.nan, dtype=np.float64)
            if data.dtype.kind == "i" and (
                (data > _EXACT_IN_DOUBLE).any() or (data < -_EXACT_IN_DOUBLE).any()
            ):
                inexact.append(result.name)
        out[idx] = data
        # The id table's key is never a result's, even when the labels were
        # dense and no table was kept: the next write would take it for ids.
        key = _unique_key(result.name, attrs.keys() | {IDS_KEY})
        attrs[key] = out[:, 0] if out.shape[1] == 1 else out
    return np.concatenate(lost) if lost else np.empty(0, np.int64)


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an I-DEAS universal file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.unv`` file, or an open file object.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Nodes in file order from datasets 2411, 781 and 15, and elements in
        file order from dataset 2412, both renumbered from zero. The labels
        the file spelled are kept under ``vertex_attrs["original_ids"]`` /
        ``element_attrs["original_ids"]`` when they say something the index
        does not. A second order element's nodes are put in polyxios order.

    Raises
    ------
    CodecError
        On a file holding no dataset, a line outside every dataset, a
        dataset with no closing ``-1`` or no number, a binary dataset, a
        node record pair cut in half, a coordinate record without three
        values, an element record without six fields, an element or a group
        declaring more nodes or members than its dataset holds, a label
        spelled twice, not positive or wider than 64 bits, an element naming
        a node the file never declares, a result declaring more values than
        its dataset holds or not a whole number of records, a 2414 header
        spelling other than twelve analysis-specific reals, a malformed
        number, or a count past the safety caps.

    Notes
    -----
    A group from datasets 2467, 2477, 2435 or 2452 becomes a
    ``vertex_tags`` entry for the nodes it names and an ``element_tags``
    entry for the elements, both under its name; two groups of one name add
    to one tag. A group or result name is read with its blanks and any NUL
    padding stripped. A group with no name is called ``group_<number>``, with
    ``_2``, ``_3`` added when another group of the file, before or after
    it, takes that name, and with a warning; a member naming no node or
    element the file declares is dropped, with a warning. Members of any
    other entity type are passed over with a warning.

    Dataset 55 and dataset 2414 at nodes or on elements become vertex and
    element attributes named after the dataset - the 2414 name record, else
    its first ID line, the first ID line for 55 - with ``_2``, ``_3`` added
    to a name already taken, so successive time steps of one result each
    keep theirs; ``original_ids`` is never a result's name, as it holds the
    file's labels. One value per entity reads as a column, several as one
    row each; complex data reads as complex, integer data as int64. An
    entity no record names reads as NaN, both parts NaN for complex data,
    an integer result missing one as float - with a warning when one of
    its values is past 2**53, which a float rounds; an entity named by
    more than one record of a result takes the last, with a warning. 2414
    element data whose value count changes between elements, 2414 data at
    nodes on elements or at points, and datasets 56, 57, 780, 2417, 2429,
    2430 and 2432 are skipped with a warning.

    An element whose FE descriptor and node count name no geometry - a
    spring, a cubic shell, a rigid element - is skipped with a warning, and
    a group member or result record naming it is left out with it. Node
    coordinates are read as they are written, whatever coordinate system a
    node names: the file's coordinate system datasets are not applied.
    """
    if lazy:
        warnings.warn(
            f"{EXTENSION}: lazy=True is not supported; loading eagerly.",
            stacklevel=2,
        )

    # 'utf-8-sig' so a byte-order mark cannot hide the first delimiter;
    # errors="replace" keeps a file written in another 8-bit encoding
    # readable, since only a group or result name can be hurt.
    text = read_text(path, encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    datasets = _datasets(lines)
    if not datasets:
        raise CodecError(f"{EXTENSION}: no dataset found; not a universal file.")

    state = _Reader()
    for number, _, first, body in datasets:
        if number in _NODES_DOUBLE:
            state.read_nodes(body, first)
        elif number == _NODES_SINGLE:
            state.read_single_nodes(body, first)
        elif number == _ELEMENTS:
            state.read_elements(body, first)
        elif number in _GROUPS:
            state.read_group(number, body, first)
        elif number == _DATA_AT_NODES:
            state.read_data_at_nodes(body, first)
        elif number == _ANALYSIS_DATA:
            state.read_analysis_data(body, first)
        elif number in _UNREAD_DATA:
            key = f"{number} ({_UNREAD_DATA[number]})"
            state.unread[key] = state.unread.get(key, 0) + 1

    node_ids = (
        np.concatenate(state.node_ids) if state.node_ids else np.empty(0, np.int64)
    )
    xyz = (
        np.concatenate(state.node_xyz)
        if state.node_xyz
        else np.empty((0, 3), dtype=np.float64)
    )
    node_line_nos = (
        np.concatenate(state.node_line_nos)
        if state.node_line_nos
        else np.empty(0, np.int64)
    )
    n_verts = node_ids.size
    n_elems = len(state.elem_ids)
    if n_verts > MAX_SAFE_VERTICES:
        raise CodecError(
            f"{EXTENSION}: node count {n_verts} exceeds the safety cap"
            f" {MAX_SAFE_VERTICES}."
        )
    if n_elems > MAX_SAFE_ELEMENTS:
        raise CodecError(
            f"{EXTENSION}: element count {n_elems} exceeds the safety cap"
            f" {MAX_SAFE_ELEMENTS}."
        )
    node_table = _sorted_ids(node_ids)
    elem_table = _sorted_ids(state.elem_ids)
    declared = (elem_table, state.elem_line_nos, state.elem_ids)
    if state.skipped_elem_ids:
        # A label is spelled once whether or not its element is kept: a
        # group naming a label a skipped element shares would land on the
        # kept one.
        all_line_nos = np.array(
            state.elem_line_nos + state.skipped_elem_line_nos, dtype=np.int64
        )
        in_file_order = np.argsort(all_line_nos, kind="stable")
        all_ids = np.array(state.elem_ids + state.skipped_elem_ids, dtype=np.int64)
        all_ids = all_ids[in_file_order]
        declared = (
            _sorted_ids(all_ids),
            all_line_nos[in_file_order].tolist(),
            all_ids.tolist(),
        )
    for what, table, line_nos, ids in (
        ("node", node_table, node_line_nos.tolist(), node_ids.tolist()),
        ("element", *declared),
    ):
        repeat = _first_repeat(table)
        if repeat is not None:
            first_at, again = repeat
            raise CodecError(
                f"{EXTENSION}: {what} label {ids[again]} at line"
                f" {line_nos[again]} repeats the one at line {line_nos[first_at]}."
            )

    total = len(state.elem_nodes)
    if total > MAX_SAFE_CONN:
        raise CodecError(
            f"{EXTENSION}: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
        )
    idx_dtype = np.int64 if total > np.iinfo(np.int32).max else np.int32
    offsets = np.zeros(n_elems + 1, dtype=idx_dtype)
    if n_elems:
        offsets[1:] = np.cumsum(state.elem_widths, dtype=np.int64)
    referenced = np.asarray(state.elem_nodes, dtype=np.int64)
    connectivity = _index_of(node_table, referenced)
    missing = np.flatnonzero(connectivity < 0)
    if missing.size:
        k = int(np.searchsorted(offsets, missing[0], side="right")) - 1
        raise CodecError(
            f"{EXTENSION}: element {state.elem_ids[k]} at line"
            f" {state.elem_line_nos[k]} names node {int(referenced[missing[0]])},"
            " which the file never declares."
        )
    connectivity = connectivity.astype(idx_dtype, copy=False)

    group_names = _name_unnamed_groups(state.node_groups, state.unnamed_groups)
    vertex_tags, lost_nodes = _resolve_groups(
        _renamed(state.node_groups, group_names), node_table
    )
    element_tags, lost_elems = _resolve_groups(
        _renamed(state.elem_groups, group_names),
        elem_table,
        empty=frozenset(group_names.get(key, key) for key in state.empty_groups),
    )

    vertex_attrs: dict[str, np.ndarray] = dict(record_ids(node_ids, count=n_verts))
    element_attrs: dict[str, np.ndarray] = dict(
        record_ids(state.elem_ids, count=n_elems)
    )
    lost_node_data = _attach_results(
        state.results,
        "vertex",
        node_table,
        n_verts,
        vertex_attrs,
        state.repeated_results,
        state.inexact_results,
    )
    lost_elem_data = _attach_results(
        state.results,
        "element",
        elem_table,
        n_elems,
        element_attrs,
        state.repeated_results,
        state.inexact_results,
    )
    # An element skipped for its descriptor is one the file does declare;
    # a group or result naming it is told apart from one naming nothing.
    skipped_table = _sorted_ids(state.skipped_elem_ids)
    skipped_members = int((_index_of(skipped_table, lost_elems) >= 0).sum())
    skipped_records = int((_index_of(skipped_table, lost_elem_data) >= 0).sum())
    n_lost_nodes = lost_nodes.size
    n_lost_elems = lost_elems.size - skipped_members
    lost_data = lost_node_data.size + lost_elem_data.size - skipped_records

    if state.skipped_types:
        listed = ", ".join(
            f"{key} x{count}" for key, count in sorted(state.skipped_types.items())
        )
        warnings.warn(
            f"{EXTENSION}: element descriptor(s) with no geometry in this codec"
            f" were skipped: {listed}.",
            stacklevel=2,
        )
    if state.unread:
        listed = ", ".join(
            f"{key} x{count}" for key, count in sorted(state.unread.items())
        )
        warnings.warn(
            f"{EXTENSION}: dataset(s) this codec does not read were skipped: {listed}.",
            stacklevel=2,
        )
    if state.uneven_results:
        warnings.warn(
            f"{EXTENSION}: element result(s) {state.uneven_results} change their"
            " value count between elements; skipped.",
            stacklevel=2,
        )
    if state.unnamed_groups:
        warnings.warn(
            f"{EXTENSION}: {len(state.unnamed_groups)} group(s) carry no name; named"
            " them group_<number>.",
            stacklevel=2,
        )
    if state.unknown_types:
        listed = sorted(state.unknown_types)
        warnings.warn(
            f"{EXTENSION}: group members of entity type(s) {listed} are neither"
            " nodes nor elements; passed over.",
            stacklevel=2,
        )
    if n_lost_nodes or n_lost_elems:
        warnings.warn(
            f"{EXTENSION}: groups name {n_lost_nodes} node(s) and {n_lost_elems}"
            " element(s) the file declares no entity for; those members were"
            " dropped.",
            stacklevel=2,
        )
    if skipped_members or skipped_records:
        warnings.warn(
            f"{EXTENSION}: {skipped_members} group member(s) and"
            f" {skipped_records} result record(s) name an element skipped for"
            " its descriptor; left out with it.",
            stacklevel=2,
        )
    if state.repeated_results:
        warnings.warn(
            f"{EXTENSION}: result(s) {state.repeated_results} name one entity in"
            " more than one record; the last record of each is kept.",
            stacklevel=2,
        )
    if state.inexact_results:
        warnings.warn(
            f"{EXTENSION}: integer result(s) {state.inexact_results} leave an"
            " entity without data, so read as float with NaN for it, and hold"
            " a value past 2**53, which a double rounds.",
            stacklevel=2,
        )
    if lost_data:
        warnings.warn(
            f"{EXTENSION}: results name {lost_data} entity label(s) the file"
            " declares no node or element for; those records were dropped.",
            stacklevel=2,
        )

    return PolyData(
        vertices=xyz,
        connectivity=connectivity,
        offsets=offsets,
        element_types=np.array(state.elem_codes, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
    )


def _real(value: float) -> str:
    """Spell one double in a ``D25.16`` field."""
    return f"{value:25.16E}".replace("E", "D")


def _value_lines(values: list[float]) -> list[str]:
    """Spell a run of result doubles three to a line.

    With an ``E`` exponent rather than the ``D`` a coordinate takes: readers
    of result data written outside Fortran parse the values with the host
    language's own float conversion, which knows no ``D``.
    """
    spelled = [f"{v:25.16E}" for v in values]
    return [
        "".join(spelled[k : k + _VALUES_PER_LINE])
        for k in range(0, len(spelled), _VALUES_PER_LINE)
    ]


def _integer_lines(values: list[int], *, width: int) -> list[str]:
    """Spell a run of result integers six to a line, ``width`` columns each."""
    spelled = [f"{v:{width}d}" for v in values]
    return [
        "".join(spelled[k : k + _INTEGERS_PER_LINE])
        for k in range(0, len(spelled), _INTEGERS_PER_LINE)
    ]


def _clean_name(name: object, *, result: bool) -> str:
    """Return a name its record line spells and a reader takes back as it is.

    Parameters
    ----------
    name
        The tag or attribute name.
    result
        Whether the name is a result's, which a reader takes for no name at
        all when it reads ``NONE``.

    Returns
    -------
    str
        The name joined onto one line and stripped, ``unnamed`` when nothing
        is left, and prefixed with ``_`` when the line would otherwise read
        as a dataset delimiter or, for a result, as ``NONE``.
    """
    text = str(name)
    # Every break the reader splits lines at, form feed and U+2028 among
    # them, not only CR and LF.
    clean = " ".join(text.splitlines()).strip()
    if not clean:
        clean = "unnamed"
    if _is_delimiter(clean) or (result and clean.upper() == "NONE"):
        clean = "_" + clean
    return clean


def _result_columns(name: str, arr: object, count: int) -> np.ndarray | None:
    """Return an attribute as an ``(count, k)`` numeric array, or None.

    Integers stay int64 so they are written as integer data, unless int64
    cannot hold them, which leaves a uint64 past its range to doubles.
    """
    values = np.asarray(arr)
    if values.dtype.kind not in "biufc" or values.ndim < 1 or len(values) != count:
        return None
    values = values.reshape(count, int(np.prod(values.shape[1:])))
    if values.shape[1] == 0:
        return None
    if values.dtype.kind == "c":
        return values.astype(np.complex128)
    if values.dtype.kind in "bi" or (
        values.dtype.kind == "u" and (not values.size or values.max() <= _INT64.max)
    ):
        return values.astype(np.int64)
    return values.astype(np.float64)


def _result_dataset(
    label: int,
    name: str,
    location: int,
    ids: np.ndarray,
    values: np.ndarray,
) -> list[str]:
    """Spell one attribute as a 2414 analysis dataset.

    Integers are written as integer data, in ``I10`` fields widened to keep
    a blank before every value; everything else in double precision, real
    or complex. An entity whose values are all NaN is left out: the
    format's way to say an entity has no data is to name no record for it.
    """
    width = values.shape[1]
    kind = values.dtype.kind
    data_type = 6 if kind == "c" else _INTEGER if kind == "i" else 4
    lines = [
        "    -1",
        f"{_ANALYSIS_DATA:6d}",
        f"{label:10d}",
        name,
        f"{location:10d}",
        name,
        "NONE",
        "NONE",
        "NONE",
        "NONE",
        f"{0:10d}{0:10d}{_CHARACTERISTIC.get(width, 0):10d}"
        f"{_RESULT_TYPE.get(width, _UNKNOWN_RESULT):10d}"
        f"{data_type:10d}{width:10d}",
        f"{0:10d}" * 8,
        f"{0:10d}" * 2,
        f"{0.0:13.5E}" * 6,
        f"{0.0:13.5E}" * 6,
    ]
    if kind == "i":
        present = np.ones(values.shape[0], dtype=bool)
        flat = values
        widest = len(str(int(values.min()))) if values.size else 0
        widest = max(widest, len(str(int(values.max()))) if values.size else 0)
        field = max(_INTEGER_WIDTH, widest + 1)
    else:
        # numpy calls a complex value NaN when either part is, but a value
        # with one finite part still carries data.
        missing = (
            np.isnan(values.real) & np.isnan(values.imag)
            if kind == "c"
            else np.isnan(values)
        )
        present = ~missing.all(axis=1)
        if kind == "c":
            flat = np.empty((values.shape[0], 2 * width), dtype=np.float64)
            flat[:, 0::2] = values.real
            flat[:, 1::2] = values.imag
        else:
            flat = values
    rows = flat.tolist()
    for i in np.flatnonzero(present).tolist():
        if location == _AT_NODES:
            lines.append(f"{int(ids[i]):10d}")
        else:
            lines.append(f"{int(ids[i]):10d}{width:10d}")
        if kind == "i":
            lines.extend(_integer_lines(rows[i], width=field))
        else:
            lines.extend(_value_lines(rows[i]))
    lines.append("    -1")
    return lines


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to an I-DEAS universal file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.unv`` path, or an open file object.
    **opts
        None are recognised; any given is warned about and ignored.

    Raises
    ------
    CodecError
        If an element references a vertex that does not exist, or the
        vertices carry other than two or three columns.

    Notes
    -----
    The file holds a 2411 node dataset in double precision, a 2412 element
    dataset in the mesh's own element order, a 2477 group per tag name -
    holding the nodes of the vertex tag and the elements of the element tag
    of that name - and a 2414 analysis dataset per attribute. Nodes and
    elements are numbered by ``original_ids`` when the mesh remembers a
    numbering it can still spell - positive, unique and at most nine digits,
    the widest a label field holds - and from one otherwise. A surface is
    written with the thin shell descriptors, a volume with the solid ones, a
    line with the linear beam and a vertex with the lumped mass; a second
    order element's nodes are put in the format's order. A vertex tag left
    with no node is not written, with a warning: a group holding no member
    reads back as an element group.

    An attribute is written at nodes or on elements with one value per
    column - several columns of a wider array, flattened - and is marked a
    scalar for one column, a vector for three and of unknown kind for any
    other count. An integer or
    boolean attribute is written as integer data and reads back as int64,
    exactly: in ``I10`` fields, widened past ten columns when a value needs
    more, which a reader holding to fixed columns then misreads. A uint64
    attribute past what int64 holds, and every real one, is written in
    double precision, and a complex one as complex doubles. An entity whose
    values are all NaN - both parts, for a complex value - gets no record,
    which is how the format says it has no data, and a NaN among other
    values is spelled ``NAN``. An attribute that is not numeric, or not one
    row per entity, is dropped with a warning.

    An element of a geometry the format has no descriptor for, or whose node
    count does not match its type, is dropped with a warning, and a group
    naming it leaves it out, as its element attribute rows do. A node with
    a non-finite coordinate is written as it is, with a warning. A tag
    member indexing outside the mesh is
    dropped, with a warning. A name is written so a reader takes it back as
    it is, with a warning when it changes: one spanning lines is joined
    onto one, outer blanks are stripped, a blank one becomes ``unnamed``,
    one reading as the ``-1`` delimiter - or for an attribute as ``NONE``,
    which reads as no name - takes a leading ``_``, and a tag name another
    tag's already cleaned to takes ``_2``, ``_3``, as does an attribute
    name another attribute of its kind already cleaned to, or one cleaning
    to ``original_ids``. A group name past the forty columns of its record
    is written whole, with a warning, since a reader that holds to the
    record cuts it. ``global_attrs`` are not written.
    """
    if opts:
        warnings.warn(
            f"{EXTENSION} write: unrecognized options {set(opts)}; ignored.",
            stacklevel=2,
        )

    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise CodecError(
            f"{EXTENSION}: vertices have shape {vertices.shape}, expected (n, 2)"
            " or (n, 3)."
        )
    if vertices.shape[1] == 2:
        vertices = pad_to_3d(vertices, 2)
    n_verts = vertices.shape[0]
    n_elems = len(poly.element_types)
    offsets = np.asarray(poly.offsets, dtype=np.int64)
    conn = np.asarray(poly.connectivity, dtype=np.int64)
    if conn.size:
        lo, hi = int(conn.min()), int(conn.max())
        if lo < 0 or hi >= n_verts:
            where = f"0..{n_verts - 1}" if n_verts else "an empty vertex array"
            raise CodecError(
                f"{EXTENSION}: an element references vertex {lo}..{hi},"
                f" outside {where}."
            )
    non_finite = int((~np.isfinite(vertices)).any(axis=1).sum()) if n_verts else 0

    codes = np.asarray(poly.element_types, dtype=np.int64)
    in_table = (codes >= 0) & (codes < _WRITE_WIDTH.size)
    expected = np.full(n_elems, -1, dtype=np.int64)
    expected[in_table] = _WRITE_WIDTH[codes[in_table]]
    spellable = expected >= 0
    written = spellable & (np.diff(offsets) == expected)
    malformed = int(spellable.sum() - written.sum())
    unspellable = {
        ELEMENT_TYPES_INV.get(code, "") or f"code {code}"
        for code in np.unique(codes[~spellable]).tolist()
    }

    node_ids = ids_for_write(
        poly, kind="vertex", count=n_verts, fmt=EXTENSION, limit=_LABEL_LIMIT
    )
    dense = np.zeros(n_elems, dtype=np.int64)
    dense[written] = np.arange(1, int(written.sum()) + 1)
    elem_ids = ids_for_write(
        poly,
        kind="element",
        count=n_elems,
        fmt=EXTENSION,
        default=dense,
        limit=_LABEL_LIMIT,
    )

    lines: list[str] = ["    -1", "  2411"]
    for nid, (x, y, z) in zip(node_ids.tolist(), vertices.tolist()):
        lines.append(f"{nid:10d}{1:10d}{1:10d}{11:10d}")
        lines.append(_real(x) + _real(y) + _real(z))
    lines.append("    -1")

    lines.extend(["    -1", "  2412"])
    spelled_conn = node_ids[conn]
    starts = offsets.tolist()
    codes_list = codes.tolist()
    elem_ids_list = elem_ids.tolist()
    for i in np.flatnonzero(written).tolist():
        name = ELEMENT_TYPES_INV[codes_list[i]]
        descriptor = _WRITE_DESCRIPTOR[name]
        nodes = spelled_conn[starts[i] : starts[i + 1]]
        inv = _INV_PERM.get(name)
        if inv is not None:
            nodes = nodes[inv]
        labels = nodes.tolist()
        lines.append(
            f"{elem_ids_list[i]:10d}{descriptor:10d}{1:10d}{1:10d}{7:10d}"
            f"{len(labels):10d}"
        )
        if descriptor in _BEAM_RANGE:
            lines.append(f"{0:10d}" * 3)
        lines.extend(
            "".join(f"{v:10d}" for v in labels[k : k + _LABELS_PER_LINE])
            for k in range(0, len(labels), _LABELS_PER_LINE)
        )
    lines.append("    -1")

    # Keyed by the pair, as a dict for its order: a tag and an attribute of
    # one name can be written as two different names.
    renamed: dict[tuple[Any, str], None] = {}
    out_of_range = 0
    unwritten = 0
    groups: dict[str, list[tuple[int, int]]] = {}
    # A vertex and an element tag of one name share a group; two names
    # cleaning to one string do not.
    group_names: dict[Any, str] = {}
    element_tags = poly.element_tags or {}
    unwritten_tags: list[str] = []
    for kind, tags, ids, count in (
        (_GROUP_NODE, poly.vertex_tags or {}, node_ids, n_verts),
        (_GROUP_ELEMENT, element_tags, elem_ids, n_elems),
    ):
        for name, members in tags.items():
            values = member_values(members)
            _, first = np.unique(values, return_index=True)
            idx = values[np.sort(first)]
            kept = idx[(idx >= 0) & (idx < count)]
            out_of_range += idx.size - kept.size
            if kind == _GROUP_ELEMENT:
                live = written[kept]
                unwritten += int(kept.size - live.sum())
                kept = kept[live]
            elif not kept.size and name not in element_tags:
                # A group says the kind of each member, so one with none
                # reads back as an element group: an empty vertex tag has
                # no group of its own, nor a name it keeps from another.
                unwritten_tags.append(str(name))
                continue
            clean = group_names.get(name)
            if clean is None:
                clean = _unique_key(_clean_name(name, result=False), groups)
                if clean != name:
                    renamed[name, clean] = None
                group_names[name] = clean
                groups[clean] = []
            groups[clean].extend((kind, int(v)) for v in ids[kept].tolist())
    long_names = [name for name in groups if len(name) > _GROUP_NAME_WIDTH]
    if groups:
        lines.extend(["    -1", "  2477"])
        for number, (name, members) in enumerate(groups.items(), start=1):
            lines.append(f"{number:10d}" + f"{0:10d}" * 6 + f"{len(members):10d}")
            lines.append(name)
            lines.extend(
                "".join(
                    f"{kind:10d}{label:10d}{0:10d}{0:10d}"
                    for kind, label in members[k : k + _GROUP_MEMBERS_PER_LINE]
                )
                for k in range(0, len(members), _GROUP_MEMBERS_PER_LINE)
            )
        lines.append("    -1")

    dropped_attrs: list[str] = []
    label = 0
    written_ids = elem_ids[written]
    for location, attrs, ids, count, keep in (
        (_AT_NODES, poly.vertex_attrs or {}, node_ids, n_verts, None),
        (_ON_ELEMENTS, poly.element_attrs or {}, written_ids, n_elems, written),
    ):
        # A reader keeps one name per kind and never a result under the id
        # table's key, so two attributes cleaning to one name are told apart
        # here, as tags are, rather than renamed on the way back in.
        taken = {IDS_KEY}
        for name, arr in attrs.items():
            if name == IDS_KEY:
                continue
            values = _result_columns(name, arr, count)
            if values is None:
                dropped_attrs.append(name)
                continue
            if keep is not None:
                values = values[keep]
            clean = _unique_key(_clean_name(name, result=True), taken)
            taken.add(clean)
            if clean != name:
                renamed[name, clean] = None
            label += 1
            lines.extend(
                _result_dataset(
                    label,
                    clean,
                    location,
                    ids,
                    values,
                )
            )

    if non_finite:
        warnings.warn(
            f"{EXTENSION}: {non_finite} node(s) have non-finite coordinates;"
            " written as they are.",
            stacklevel=2,
        )
    if unspellable:
        warnings.warn(
            f"{EXTENSION}: element type(s) {sorted(unspellable)} have no"
            " universal file descriptor; those elements were dropped, with"
            " their element attribute rows, and a group naming one leaves it"
            " out.",
            stacklevel=2,
        )
    if malformed:
        warnings.warn(
            f"{EXTENSION}: {malformed} element(s) carry a node count that does"
            " not match their type; dropped, with their element attribute rows.",
            stacklevel=2,
        )
    if out_of_range:
        warnings.warn(
            f"{EXTENSION}: {out_of_range} tag member(s) index an entity the"
            " mesh does not hold; dropped.",
            stacklevel=2,
        )
    if unwritten:
        warnings.warn(
            f"{EXTENSION}: {unwritten} tag member(s) name an element that was"
            " not written; left out of their group.",
            stacklevel=2,
        )
    if unwritten_tags:
        warnings.warn(
            f"{EXTENSION}: vertex tag(s) {sorted(unwritten_tags)}"
            " hold no node a group can name; not written.",
            stacklevel=2,
        )
    if long_names:
        cut = [name[:_GROUP_NAME_WIDTH] for name in groups]
        collide = len(set(cut)) < len(cut)
        warnings.warn(
            f"{EXTENSION}: group name(s) {long_names} run past the"
            f" {_GROUP_NAME_WIDTH} columns of the format's name record; written"
            " whole, and a reader that holds to the record cuts them"
            + (", which leaves two groups one name." if collide else "."),
            stacklevel=2,
        )
    if dropped_attrs:
        warnings.warn(
            f"{EXTENSION}: attribute(s) {sorted(dropped_attrs)} are not numeric"
            " with one row per entity; dropped.",
            stacklevel=2,
        )
    if renamed:
        listed = ", ".join(f"{old!r} as {new!r}" for old, new in renamed)
        warnings.warn(
            f"{EXTENSION}: a name is one line, not blank, not -1 and, for a"
            f" result, not NONE, and no two groups or results of a kind share"
            f" one; wrote {listed}.",
            stacklevel=2,
        )

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
