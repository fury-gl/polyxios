"""PERMAS codec (ASCII ``.dato`` / ``.post``) - read + write.

A PERMAS deck is a sequence of ``$KEYWORD`` records, each followed by the
data lines it governs until the next ``$``. The structure this codec reads
lives in four of them: ``$COOR`` lists nodes as ``id x y z``, ``$ELEMENT
TYPE = <type>`` lists cells as ``id <nodes>``, and ``$NSET`` / ``$ESET`` name
groups of node and element ids. A ``!`` opens a comment and a ``&`` in the
first column continues the line before it. Every other record - the
``$ENTER COMPONENT`` / ``$STRUCTURE`` framing, ``$SYSTEM``, ``$LOADING``,
``$MATERIAL`` and the rest of a solver deck - is passed over.

Nodes and elements are numbered freely, so both are renumbered on the way in
and the file's own ids are kept under ``original_ids`` - see
:mod:`polyxios._ids`. Each set becomes a tag named after it, as does the
``NSET =`` a ``$COOR`` record names and the ``ESET =`` an ``$ELEMENT`` record
names. Nothing in the structure section carries per-entity data, so
attributes have no home here and are dropped on write.

The PERMAS element type is a solver class rather than a geometry - ``QUAD4``,
``SHELL4`` and ``LOADA4`` are all four-node quadrilaterals - so every class the
table knows reads as its geometry, and the writer spells the geometry with one
canonical class unless ``element_type=`` chooses another. Node order for the
higher-order types follows a reference implementation, which agrees with
polyxios's own; no permutation is applied.
"""

import re
from typing import Any
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
from polyxios._ids import ids_for_write, record_ids
from polyxios._io import Source, read_text, write_text
from polyxios._tags import member_values
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

# ``.dato`` is the model a solver run is fed and ``.post`` is what it writes
# back, structure included; the writer produces a model, so that one is
# canonical. ``.dat`` is used for the same deck and by several unrelated
# formats besides, so it is competed for by content instead.
EXTENSION: str = ".dato"
EXTENSIONS: tuple[str, ...] = (".dato", ".post")
SNIFF_EXTENSIONS: tuple[str, ...] = (".dat",)
# A deck opens with a ``$`` record spelling a PERMAS keyword, which nothing
# else sharing ``.dat`` does, so the narrow test sorts ahead of the broad ones.
SNIFF_PRIORITY: int = -1

#: Where a reader keeps the ``$ENTER COMPONENT NAME =`` a file spelled, when
#: it is not the default a writer would spell anyway.
COMPONENT_KEY: str = "permas_component"

#: The component name written when the mesh remembers none.
DEFAULT_COMPONENT: str = "DFLT_COMP"

# PERMAS element class -> polyxios geometry. Several classes share a geometry
# and differ only in what the solver does with them, which is nothing a mesh
# carries. Node order follows a reference implementation for every type.
_PERMAS_TO_POLYXIOS: dict[str, str] = {
    "PLOT1": "vertex",
    "PLOTL2": "line",
    "FLA2": "line",
    "BECOS": "line",
    "BECOC": "line",
    "BETAC": "line",
    "BECOP": "line",
    "BETOP": "line",
    "BEAM2": "line",
    "FSCPIPE2": "line",
    "PLOTL3": "quadratic_edge",
    "FLA3": "quadratic_edge",
    "PLOTA3": "triangle",
    "SHELL3": "triangle",
    "TRIA3": "triangle",
    "TRIA3K": "triangle",
    "TRIA3S": "triangle",
    "TRIMS3": "triangle",
    "LOADA6": "quadratic_triangle",
    "TRIMS6": "quadratic_triangle",
    "PLOTA4": "quad",
    "LOADA4": "quad",
    "QUAD4": "quad",
    "QUAD4S": "quad",
    "QUAMS4": "quad",
    "SHELL4": "quad",
    "PLOTA8": "quadratic_quad",
    "LOADA8": "quadratic_quad",
    "QUAMS8": "quadratic_quad",
    "PLOTA9": "biquadratic_quad",
    "LOADA9": "biquadratic_quad",
    "QUAMS9": "biquadratic_quad",
    "TET4": "tetra",
    "TET10": "quadratic_tetra",
    "HEXE8": "hexahedron",
    "HEXFO8": "hexahedron",
    "HEXE20": "quadratic_hexahedron",
    "HEXE27": "triquadratic_hexahedron",
    "PYRA5": "pyramid",
    "PENTA6": "wedge",
    "PENTA15": "quadratic_wedge",
}

# The class each geometry is written as when the caller names none: the plain
# structural element of that shape, which every PERMAS reader accepts.
_WRITE_TYPE: dict[str, str] = {
    "vertex": "PLOT1",
    "line": "PLOTL2",
    "quadratic_edge": "PLOTL3",
    "triangle": "TRIA3",
    "quadratic_triangle": "TRIMS6",
    "quad": "QUAD4",
    "quadratic_quad": "QUAMS8",
    "biquadratic_quad": "QUAMS9",
    "tetra": "TET4",
    "quadratic_tetra": "TET10",
    "hexahedron": "HEXE8",
    "quadratic_hexahedron": "HEXE20",
    "triquadratic_hexahedron": "HEXE27",
    "pyramid": "PYRA5",
    "wedge": "PENTA6",
    "quadratic_wedge": "PENTA15",
}

# Records this codec reads; every other ``$`` record is passed over.
_ENTER: str = "ENTER COMPONENT"
_FIN: str = "FIN"
_DATA_RECORDS: frozenset[str] = frozenset({"COOR", "ELEMENT", "NSET", "ESET"})

# What a data value opens with. A bare flag word a record carries -
# ``DOFTYPE = DISP MATH``, ``$COOR CART`` - never does, so a token that does
# is a data line that a ``&`` glued onto the record above it.
_DATA_START_RE = re.compile(r"[-+.\d]")

# The one keyword this codec reads that is spelled in two words; every other
# ``$`` record it acts on is named by its first word alone.
_TWO_WORD_KEYWORDS: frozenset[str] = frozenset({"ENTER"})

# The ``$COOR`` flags naming a coordinate system other than Cartesian. Their
# rows are (r, phi, z) / (r, phi, theta) in the solver's own angle convention,
# which this codec does not convert, so such a block is refused rather than
# read as x y z.
_NON_CARTESIAN: frozenset[str] = frozenset({"CYL", "SPH"})

# The bare flag words a data record may carry. A Nastran banner spelled flush
# against its ``$`` - ``$ELEMENT PROPERTIES``, ``$COOR SYSTEMS`` - parses to
# the same keyword with a word the format never puts there, so the sniffer
# lets a record through only when every flag is one the record can hold.
_SNIFF_FLAGS: dict[str, frozenset[str]] = {"COOR": frozenset({"CART", *_NON_CARTESIAN})}

# What an id or a set member is spelled as. ``int`` alone also takes ``1_0``
# and the digits of other scripts, neither of which a deck holds.
_INT_RE = re.compile(r"[-+]?[0-9]+")
_INT64 = np.iinfo(np.int64)

# Nodes per polyxios element code for the geometries this codec writes, -1
# for the rest, so a mesh's whole element table is checked in one indexing.
_WRITE_WIDTH: np.ndarray = np.full(256, -1, dtype=np.int64)
for _name in _WRITE_TYPE:
    _WRITE_WIDTH[ELEMENT_TYPES[_name]] = NODES_PER_ELEMENT[_name]
del _name

# The characters a name cannot carry on a record: whitespace and ``=`` split
# it into parameters, ``!`` comments out its tail, ``$`` and ``&`` open a
# record and a continuation.
_UNSAFE_NAME_RE = re.compile(r"[\s=!$&]+")

# Members per line when a set is written. Any number is legal; eight keeps a
# line under the width the format's own tools wrap at.
_SET_MEMBERS_PER_LINE: int = 8


def _scan(text: str) -> list[tuple[int, str]]:
    """Return the file's meaningful lines as ``(line number, text)`` pairs.

    A ``!`` comments out the rest of its line, a blank line carries nothing,
    and a line opening with ``&`` continues the record before it - so all
    three are folded away here, and a reader sees one record per entry. The
    original line number rides along so an error can name the line of the
    file, not of the list.
    """
    records: list[tuple[int, str]] = []
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if "!" in line:
            line = line.split("!", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("&"):
            if not records:
                raise CodecError(
                    f"{EXTENSION}: line {no} continues a record, but no record"
                    " precedes it."
                )
            prev_no, prev = records[-1]
            records[-1] = (prev_no, f"{prev} {line[1:].strip()}")
            continue
        records.append((no, line))
    return records


def _parse_keyword(
    line: str,
) -> tuple[str, dict[str, str], frozenset[str], list[str]]:
    """Split a ``$`` record into keyword, parameters, flags and data.

    The keyword is the record's first word, upper-cased - two for the
    ``$ENTER <section>`` framing - so ``$ELEMENT TYPE = HEXE8`` is
    ``ELEMENT`` and ``$COOR CART`` is still ``COOR``. Spacing around ``=`` is
    free in the format and is normalised away into the ``NAME = value``
    parameters. A bare word anywhere else on the record is a flag - the
    ``CART`` of ``$COOR CART``, the ``MATH`` of ``DOFTYPE = DISP MATH`` - and
    comes back upper-cased in the third item. The first token that opens like
    a number, wherever it sits, starts the data a ``&`` folded onto the
    record, which comes back whole as the last item: a bare ``$COOR``
    continued by ``& 1 0 0 0`` reads as ``COOR`` with that node line as its
    data.
    """
    body = re.sub(r"\s*=\s*", "=", line[1:].strip())
    words: list[str] = []
    params: dict[str, str] = {}
    data: list[str] = []
    for tok in body.split():
        if data:
            data.append(tok)
        elif "=" in tok:
            key, _, value = tok.partition("=")
            params[key.upper()] = value
        elif _DATA_START_RE.match(tok):
            data.append(tok)
        else:
            words.append(tok.upper())
    take = 2 if words and words[0] in _TWO_WORD_KEYWORDS else 1
    return " ".join(words[:take]), params, frozenset(words[take:]), data


def _sorted_ids(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted ids, their positions)`` for :func:`_index_of`."""
    arr = np.asarray(ids, dtype=np.int64)
    order = np.argsort(arr, kind="stable")
    return arr[order], order


def _index_of(table: tuple[np.ndarray, np.ndarray], wanted: np.ndarray) -> np.ndarray:
    """Return the position of each ``wanted`` id in the table, -1 if absent.

    A binary search over the sorted ids, so a whole connectivity resolves in
    one vectorised pass rather than a dictionary lookup per node reference.
    The ids are unique - a repeat was refused when read - so a hit is exact.
    """
    sorted_ids, order = table
    if sorted_ids.size == 0:
        return np.full(wanted.shape, -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(sorted_ids, wanted), sorted_ids.size - 1)
    return np.where(sorted_ids[pos] == wanted, order[pos], -1)


def _next_record(records: list[tuple[int, str]], start: int) -> int:
    """Return the index of the next ``$`` record at or after ``start``."""
    for i in range(start, len(records)):
        if records[i][1].startswith("$"):
            return i
    return len(records)


def _parse_int(tok: str, no: int, what: str) -> int:
    """Convert one id or set member, refusing what int64 cannot hold."""
    if not _INT_RE.fullmatch(tok):
        raise CodecError(f"{EXTENSION}: malformed {what} {tok!r} on line {no}.")
    value = int(tok)
    if not _INT64.min <= value <= _INT64.max:
        raise CodecError(
            f"{EXTENSION}: {what} {tok!r} on line {no} is wider than 64 bits."
        )
    return value


def _parse_ints(line: str, toks: list[str], no: int, what: str) -> list[int]:
    """Convert a run of integer tokens, the plain way when the line allows it.

    ``int`` is the fast path, but it also accepts ``1_0`` and the digits of
    other scripts, so a line carrying either goes token by token through
    :func:`_parse_int` instead, which is also where a malformed token gets
    named. The width check waits for the numpy conversion downstream, where
    an overflow is caught once for the whole array rather than tested here
    on every value.
    """
    if line.isascii() and "_" not in line:
        try:
            return list(map(int, toks))
        except ValueError:
            pass
    return [_parse_int(tok, no, what) for tok in toks]


def _parse_reals(toks: list[str], no: int, line: str) -> tuple[float, float, float]:
    """Convert three coordinate tokens, accepting the Fortran ``D`` exponent.

    PERMAS is Fortran-hosted and a ``.post`` it writes may spell a double as
    ``1.0D+00``, which ``float`` refuses. This is the slow path a line takes
    only after the plain conversion failed, so the common deck pays nothing.
    """
    try:
        x, y, z = (float(tok.replace("D", "E").replace("d", "e")) for tok in toks)
    except ValueError as exc:
        raise CodecError(f"{EXTENSION}: malformed node line {no}: {line!r}.") from exc
    return x, y, z


def _first_repeat(table: tuple[np.ndarray, np.ndarray]) -> tuple[int, int] | None:
    """Return ``(first position, repeat position)`` of the earliest repeated id.

    Read off the sorted table in one vectorised pass rather than a dictionary
    probe per line while reading. The sort is stable, so of two equal ids the
    lower position sorts first, and the repeat reported is the one the file
    reaches first.
    """
    sorted_ids, order = table
    hits = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    if hits.size == 0:
        return None
    j = hits[np.argmin(order[hits + 1])]
    return int(order[j]), int(order[j + 1])


class _Reader:
    """The state one pass over a deck accumulates.

    Every ``$COOR`` and ``$ELEMENT`` block appends to the same lists, in file
    order, and the ids they spell are resolved only once the whole deck has
    been seen: a set may name a node the deck declares later, and an element
    block may precede the coordinates in a hand-written file.
    """

    def __init__(self) -> None:
        self.node_ids: list[int] = []
        self.coords: list[tuple[float, float, float]] = []
        self.node_line_nos: list[int] = []
        self.elem_ids: list[int] = []
        self.elem_line_nos: list[int] = []
        self.elem_nodes: list[int] = []
        self.elem_widths: list[int] = []
        self.elem_codes: list[int] = []
        self.nsets: dict[str, list[int]] = {}
        self.esets: dict[str, list[int]] = {}
        self.component: str | None = None
        self.skipped_types: dict[str, int] = {}
        self.overlong: int = 0
        self.first_overlong: int = 0
        self.unnamed_sets: int = 0

    def read_nodes(self, body: list[tuple[int, str]], params: dict[str, str]) -> None:
        first = len(self.node_ids)
        for no, line in body:
            parts = line.split()
            nid = _parse_int(parts[0], no, "node id")
            if nid < 1:
                raise CodecError(
                    f"{EXTENSION}: node id {nid} on line {no} is not positive;"
                    " the format numbers from 1."
                )
            if len(parts) < 4:
                raise CodecError(
                    f"{EXTENSION}: node line {no} carries {len(parts) - 1}"
                    " coordinate(s), expected 3."
                )
            if len(parts) > 4:
                if not self.overlong:
                    self.first_overlong = no
                self.overlong += 1
            try:
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                xyz = _parse_reals(parts[1:4], no, line)
            self.node_line_nos.append(no)
            self.node_ids.append(nid)
            self.coords.append(xyz)
        name = params.get("NSET")
        if name:
            self.nsets.setdefault(name, []).extend(self.node_ids[first:])

    def read_elements(
        self, body: list[tuple[int, str]], params: dict[str, str], no: int
    ) -> None:
        klass = params.get("TYPE", "").upper()
        if not klass:
            raise CodecError(
                f"{EXTENSION}: $ELEMENT record on line {no} names no TYPE."
            )
        name = _PERMAS_TO_POLYXIOS.get(klass)
        if name is None:
            if body:
                self.skipped_types[klass] = self.skipped_types.get(klass, 0) + len(body)
            return
        width = NODES_PER_ELEMENT[name]
        code = ELEMENT_TYPES[name]
        first = len(self.elem_ids)
        for line_no, line in body:
            parts = line.split()
            eid = _parse_int(parts[0], line_no, "element id")
            if eid < 1:
                raise CodecError(
                    f"{EXTENSION}: element id {eid} on line {line_no} is not"
                    " positive; the format numbers from 1."
                )
            if len(parts) - 1 < width:
                raise CodecError(
                    f"{EXTENSION}: element line {line_no} carries"
                    f" {len(parts) - 1} node(s), expected {width} for {klass}."
                )
            if len(parts) - 1 > width:
                if not self.overlong:
                    self.first_overlong = line_no
                self.overlong += 1
            nodes = _parse_ints(line, parts[1 : 1 + width], line_no, "node reference")
            self.elem_line_nos.append(line_no)
            self.elem_ids.append(eid)
            self.elem_nodes.extend(nodes)
            self.elem_widths.append(width)
            self.elem_codes.append(code)
        eset = params.get("ESET")
        if eset:
            self.esets.setdefault(eset, []).extend(self.elem_ids[first:])

    def read_set(
        self,
        body: list[tuple[int, str]],
        params: dict[str, str],
        sets: dict[str, list[int]],
        kind: str,
    ) -> None:
        name = params.get("NAME", "")
        if not name:
            self.unnamed_sets += 1
            name = f"{kind}_{self.unnamed_sets}"
        members = sets.setdefault(name, [])
        for no, line in body:
            members.extend(_parse_ints(line, line.split(), no, f"{kind} member"))


def _resolve_tags(
    sets: dict[str, list[int]],
    table: tuple[np.ndarray, np.ndarray],
    dtype: type,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Turn sets of file ids into tags of mesh indices, counting the unknown.

    A member the deck declares no entity for is dropped and counted: a set
    naming an element of a class this codec skipped is the usual way to get
    one, and the mesh is whole without it.
    """
    tags: dict[str, np.ndarray] = {}
    dropped: dict[str, int] = {}
    for name, members in sets.items():
        try:
            wanted = np.asarray(members, dtype=np.int64)
        except OverflowError as exc:
            wide = next(m for m in members if not _INT64.min <= m <= _INT64.max)
            raise CodecError(
                f"{EXTENSION}: set {name!r} names {wide}, which is wider than 64 bits."
            ) from exc
        found = _index_of(table, wanted)
        kept = found[found >= 0]
        lost = found.size - kept.size
        if lost:
            dropped[name] = lost
        # A set that repeats a member names it once, and the file's order is
        # the only order the members have, so the first spelling of each wins.
        _, first = np.unique(kept, return_index=True)
        tags[name] = kept[np.sort(first)].astype(dtype, copy=False)
    return tags, dropped


def sniff(head: bytes) -> bool:
    """Report whether a file's opening bytes look like a PERMAS deck.

    Parameters
    ----------
    head
        The file's first bytes, as handed over by the registry.

    Returns
    -------
    bool
        True when the first record is a ``$`` keyword this codec knows the
        deck to open with: an ``$ENTER <section>`` framing - component,
        material, system - a bare ``$STRUCTURE``, or one of the records it
        reads, spelled the way the format spells it and followed by a data
        line or another record.

    Notes
    -----
    Used to resolve ``.dat``, which several unrelated formats share. Comment
    and blank lines are stepped over. A Nastran deck also opens with ``$``
    lines, as comments, and those usually put a space after the ``$``; a
    PERMAS keyword sits flush against it, so ``$ STRUCTURE`` is not claimed.
    A banner spelled flush - ``$ELEMENT PROPERTIES``, ``$COOR SYSTEMS`` -
    parses to a record with a word the format never puts there, or one with
    a bulk data card where its data should be, and is not claimed either.
    """
    # 'utf-8-sig' as in :func:`read`, so a byte-order mark cannot hide the
    # ``$`` of the first record.
    text = head.decode("utf-8-sig", errors="replace")
    opening: list[str] = []
    for line in text.splitlines():
        stripped = line.split("!", 1)[0].strip()
        if stripped:
            opening.append(stripped)
            if len(opening) == 2:
                break
    if not opening:
        return False
    first = opening[0]
    # A keyword sits flush against its ``$``; a Nastran comment such as
    # ``$ STRUCTURE`` puts a space there, and is not a deck.
    if not first.startswith("$") or not first[1:2].strip():
        return False
    keyword, params, flags, data = _parse_keyword(first)
    if keyword.partition(" ")[0] == "ENTER":
        return True
    if keyword == "STRUCTURE":
        return not (params or flags or data)
    if keyword not in _DATA_RECORDS:
        return False
    if flags - _SNIFF_FLAGS.get(keyword, frozenset()):
        return False
    if keyword == "ELEMENT" and "TYPE" not in params:
        return False
    if data or len(opening) < 2:
        return True
    # What follows a data record is its data, a continuation of it, or the
    # next record; a bulk data card is none of those.
    following = opening[1]
    return (
        following.startswith(("$", "&")) or _DATA_START_RE.match(following) is not None
    )


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a PERMAS deck and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.dato`` / ``.post`` file, or an open file object.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Nodes in ``$COOR`` order and elements in ``$ELEMENT`` order, both
        renumbered from zero. The ids the file spelled are kept under
        ``vertex_attrs["original_ids"]`` / ``element_attrs["original_ids"]``
        when they say something the index does not.

    Raises
    ------
    CodecError
        On a node or element id spelled twice, not positive or wider than 64
        bits, a node line without three coordinates, an element line with
        fewer nodes than its class holds, an element naming a node the deck
        never declares, a malformed number,
        a continuation with nothing to continue, a data line outside every
        record, a file holding no ``$`` record at all, a count past the
        safety caps, or a ``$COOR`` block flagged ``CYL`` or ``SPH``, whose
        rows are not the Cartesian x y z this codec reads.

    Notes
    -----
    Each ``$NSET`` becomes a ``vertex_tags`` entry and each ``$ESET`` an
    ``element_tags`` entry, as do the ``NSET =`` of a ``$COOR`` record and the
    ``ESET =`` of an ``$ELEMENT`` record. Two records naming one set add to
    it. A set naming an id the deck declares no entity for drops that member
    with a warning, and a set with no ``NAME =`` is called ``nset_<n>`` /
    ``eset_<n>``, also with a warning.

    An ``$ELEMENT`` block of a class the table does not know - a spring, a
    mass, a rigid body - is skipped with a warning; an element line carrying
    more nodes than its class holds has the extra ignored, with a warning. A
    deck holding a second ``$ENTER COMPONENT`` is read down to the end of the
    first, with a warning. The name of the component read is kept under
    ``global_attrs["permas_component"]`` unless it is ``DFLT_COMP``, the one
    a writer spells on its own.
    """
    if lazy:
        warnings.warn(
            f"{EXTENSION}: lazy=True is not supported; loading eagerly.",
            stacklevel=2,
        )

    # 'utf-8-sig' so a byte-order mark cannot glue itself to the first
    # record; errors="replace" keeps a deck written in another 8-bit encoding
    # inside a CodecError, since only a set name can be hurt.
    text = read_text(path, encoding="utf-8-sig", errors="replace")
    records = _scan(text)
    if not records:
        raise CodecError(f"{EXTENSION}: no $ record found; not a PERMAS deck.")
    state = _Reader()

    i = 0
    components = 0
    while i < len(records):
        no, line = records[i]
        if not line.startswith("$"):
            raise CodecError(
                f"{EXTENSION}: line {no} carries data outside any $ record: {line!r}."
            )
        keyword, params, flags, data = _parse_keyword(line)
        if keyword == _FIN:
            break
        end = _next_record(records, i + 1)
        body = records[i + 1 : end]
        # A ``&`` folds its line onto the record above, keyword records
        # included, so ``$NSET NAME = A`` + ``& 1 2 3`` arrives as one line.
        if data and keyword in _DATA_RECORDS:
            body = [(no, " ".join(data)), *body]
        if keyword == _ENTER:
            components += 1
            if components > 1:
                warnings.warn(
                    f"{EXTENSION}: the deck holds a second $ENTER COMPONENT at"
                    f" line {no}; only the first component was read.",
                    stacklevel=2,
                )
                break
            state.component = params.get("NAME") or None
        elif keyword == "COOR":
            system = flags & _NON_CARTESIAN
            if system:
                raise CodecError(
                    f"{EXTENSION}: $COOR on line {no} is flagged"
                    f" {next(iter(system))}; only Cartesian coordinates are"
                    " read."
                )
            state.read_nodes(body, params)
        elif keyword == "ELEMENT":
            state.read_elements(body, params, no)
        elif keyword == "NSET":
            state.read_set(body, params, state.nsets, "nset")
        elif keyword == "ESET":
            state.read_set(body, params, state.esets, "eset")
        i = end

    n_verts = len(state.node_ids)
    n_elems = len(state.elem_ids)
    node_table = _sorted_ids(state.node_ids)
    elem_table = _sorted_ids(state.elem_ids)
    for what, table, line_nos, ids in (
        ("node", node_table, state.node_line_nos, state.node_ids),
        ("element", elem_table, state.elem_line_nos, state.elem_ids),
    ):
        repeat = _first_repeat(table)
        if repeat is not None:
            first, again = repeat
            raise CodecError(
                f"{EXTENSION}: {what} id {ids[again]} on line {line_nos[again]}"
                f" repeats the one on line {line_nos[first]}."
            )
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

    total = len(state.elem_nodes)
    if total > MAX_SAFE_CONN:
        raise CodecError(
            f"{EXTENSION}: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
        )
    idx_dtype = np.int64 if total > np.iinfo(np.int32).max else np.int32
    offsets = np.zeros(n_elems + 1, dtype=idx_dtype)
    if n_elems:
        offsets[1:] = np.cumsum(state.elem_widths, dtype=np.int64)

    try:
        referenced = np.asarray(state.elem_nodes, dtype=np.int64)
    except OverflowError as exc:
        j, wide = next(
            (j, v)
            for j, v in enumerate(state.elem_nodes)
            if not _INT64.min <= v <= _INT64.max
        )
        k = int(np.searchsorted(offsets, j, side="right")) - 1
        raise CodecError(
            f"{EXTENSION}: element {state.elem_ids[k]} on line"
            f" {state.elem_line_nos[k]} references node {wide}, which is wider"
            " than 64 bits."
        ) from exc
    connectivity = _index_of(node_table, referenced)
    missing = np.flatnonzero(connectivity < 0)
    if missing.size:
        k = int(np.searchsorted(offsets, missing[0], side="right")) - 1
        raise CodecError(
            f"{EXTENSION}: element {state.elem_ids[k]} on line"
            f" {state.elem_line_nos[k]} references node"
            f" {int(referenced[missing[0]])}, which the deck never declares."
        )
    connectivity = connectivity.astype(idx_dtype, copy=False)

    vertex_tags, lost_nodes = _resolve_tags(state.nsets, node_table, np.int32)
    element_tags, lost_elems = _resolve_tags(state.esets, elem_table, np.int32)

    if state.overlong:
        warnings.warn(
            f"{EXTENSION}: {state.overlong} line(s) carry more values than"
            f" their record needs, first at line {state.first_overlong}; the"
            " extra values were ignored.",
            stacklevel=2,
        )
    if state.skipped_types:
        listed = ", ".join(
            f"{klass} ({count})" for klass, count in sorted(state.skipped_types.items())
        )
        warnings.warn(
            f"{EXTENSION}: element class(es) with no geometry in this codec"
            f" were skipped: {listed}.",
            stacklevel=2,
        )
    if state.unnamed_sets:
        warnings.warn(
            f"{EXTENSION}: {state.unnamed_sets} set(s) carry no NAME =; named"
            " them nset_<n> / eset_<n>.",
            stacklevel=2,
        )
    for kind, lost in (("node", lost_nodes), ("element", lost_elems)):
        if lost:
            warnings.warn(
                f"{EXTENSION}: {kind} set(s) {sorted(lost)} name"
                f" {sum(lost.values())} id(s) the deck declares no {kind} for;"
                " those members were dropped.",
                stacklevel=2,
            )

    vertex_attrs = dict(record_ids(state.node_ids, count=n_verts))
    element_attrs = dict(record_ids(state.elem_ids, count=n_elems))
    global_attrs: dict[str, Any] = {}
    if state.component and state.component != DEFAULT_COMPONENT:
        global_attrs[COMPONENT_KEY] = state.component

    return PolyData(
        vertices=np.array(state.coords, dtype=np.float64).reshape(n_verts, 3),
        connectivity=connectivity,
        offsets=offsets,
        element_types=np.array(state.elem_codes, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _safe_name(name: str, used: set[str], renamed: list[tuple[str, str]]) -> str:
    """Return a name that survives its own record, unique against ``used``."""
    safe = _UNSAFE_NAME_RE.sub("_", str(name).strip()) or "unnamed"
    candidate, k = safe, 1
    while candidate in used:
        k += 1
        candidate = f"{safe}_{k}"
    used.add(candidate)
    if candidate != str(name):
        renamed.append((str(name), candidate))
    return candidate


def _check_overrides(overrides: object) -> dict[str, str]:
    """Validate an ``element_type=`` mapping and return it upper-cased.

    Raises
    ------
    CodecError
        If the option is not a mapping of strings, names a geometry this
        codec does not write, or gives one a class the table knows to hold a
        different geometry - a deck spelling four nodes under ``HEXE8`` is
        one no reader loads. A class outside the table is the caller's own
        business, since they may be writing for a solver build this codec
        does not read back, but it still has to be a bare word.
    """
    if not isinstance(overrides, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()
    ):
        raise CodecError(
            f"{EXTENSION}: element_type= takes a mapping from polyxios element"
            " name to PERMAS element class, such as {'quad': 'SHELL4'}."
        )
    checked: dict[str, str] = {}
    for name, klass in overrides.items():
        if name not in _WRITE_TYPE:
            raise CodecError(
                f"{EXTENSION}: element_type= names {name!r}, which this codec"
                f" does not write; it writes {sorted(_WRITE_TYPE)}."
            )
        word = klass.strip().upper()
        if not word or _UNSAFE_NAME_RE.search(word):
            raise CodecError(
                f"{EXTENSION}: element_type= gives {name!r} the class {klass!r},"
                " which is not a bare class name."
            )
        known = _PERMAS_TO_POLYXIOS.get(word)
        if known is not None and known != name:
            raise CodecError(
                f"{EXTENSION}: element_type= writes {name!r} as {word!r}, which"
                f" is a {NODES_PER_ELEMENT[known]}-node {known!r} class."
            )
        checked[name] = word
    return checked


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a PERMAS deck.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.dato`` / ``.post`` path, or an open file object.
    **opts
        ``element_type``: a mapping from polyxios element name to the PERMAS
        element class to spell it with, such as ``{"quad": "SHELL4"}``, for
        the geometries the caller wants written as something other than the
        codec's own choice. Any other option is warned about and ignored.

    Raises
    ------
    CodecError
        If ``element_type`` is not a mapping of strings, names a geometry
        this codec does not write, or gives one a class known to hold a
        different geometry. Also if an element references a vertex that does
        not exist, or the vertices carry other than two or three columns.

    Notes
    -----
    The deck is one component holding one ``$STRUCTURE``: a ``$COOR`` block,
    one ``$ELEMENT TYPE =`` block per run of elements of one geometry - so the
    mesh's own element order is kept - then a ``$NSET`` per ``vertex_tags``
    entry and an ``$ESET`` per ``element_tags`` entry. Nodes and elements are
    numbered by ``original_ids`` when the mesh remembers a numbering it can
    still spell, and from one otherwise.

    An element of a geometry PERMAS has no class for, or whose node count does
    not match its type, is dropped with a warning, and a set naming it leaves
    it out. A node with a non-finite coordinate is written as it is, with a
    warning, since the solver will not load it. A tag naming an index outside the mesh drops that member. A name
    that cannot sit on a record - one carrying whitespace, ``=``, ``!``,
    ``$`` or ``&`` - has those replaced by ``_``, and a repeat after that gets
    a ``_<n>`` suffix, each with a warning. The component is named by
    ``global_attrs["permas_component"]`` when present and ``DFLT_COMP``
    otherwise. Nothing in the structure section carries per-entity data, so
    ``vertex_attrs``, ``element_attrs`` and the rest of ``global_attrs`` are
    not written.
    """
    chosen = opts.pop("element_type", None)
    overrides = _check_overrides({} if chosen is None else chosen)
    if opts:
        warnings.warn(
            f"{EXTENSION} write: unrecognized options {set(opts)}; ignored.",
            stacklevel=2,
        )

    # A $COOR line holds exactly three coordinates: a planar mesh is lifted
    # with a zero z, and any other width would put rows of that width on disk.
    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise CodecError(
            f"{EXTENSION}: vertices have shape {vertices.shape}, expected (n, 3)."
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

    # A code outside the table is unspellable; the mask keeps a hand-built
    # element_types of a wider dtype from indexing past it.
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
    kept_names = {
        ELEMENT_TYPES_INV[code] for code in np.unique(codes[written]).tolist()
    }
    # Judged against every element of the geometry, written or not: a class
    # chosen for a geometry the mesh holds only in malformed elements was not
    # ignored, those elements were dropped, and that is warned about below.
    present = {ELEMENT_TYPES_INV[code] for code in np.unique(codes[spellable]).tolist()}
    unused = sorted(set(overrides) - present)

    node_ids = ids_for_write(poly, kind="vertex", count=n_verts, fmt=EXTENSION)
    # A dropped element takes no id, so an unremembered mesh still lands
    # numbered densely over the elements the deck holds.
    dense = np.zeros(n_elems, dtype=np.int64)
    dense[written] = np.arange(1, int(written.sum()) + 1)
    elem_ids = ids_for_write(
        poly, kind="element", count=n_elems, fmt=EXTENSION, default=dense
    )

    used: set[str] = set()
    renamed: list[tuple[str, str]] = []
    component = _safe_name(
        (poly.global_attrs or {}).get(COMPONENT_KEY, DEFAULT_COMPONENT), used, renamed
    )
    node_ids_list = node_ids.tolist()
    lines: list[str] = [
        f"$ENTER COMPONENT NAME = {component} DOFTYPE = DISP",
        "$STRUCTURE",
        "$COOR",
    ]
    lines.extend(
        f"    {nid} {x!r} {y!r} {z!r}"
        for nid, (x, y, z) in zip(node_ids_list, vertices.tolist())
    )

    # The class of each element, resolved once per code rather than per
    # element, and every node reference spelled as its id in one indexing.
    klass_of = {
        code: overrides.get(name, _WRITE_TYPE[name])
        for code, name in ((ELEMENT_TYPES[n], n) for n in kept_names)
    }
    spelled_conn = node_ids[conn].tolist()
    starts = offsets.tolist()
    codes_list = codes.tolist()
    elem_ids_list = elem_ids.tolist()
    current: str | None = None
    for i in np.flatnonzero(written).tolist():
        klass = klass_of[codes_list[i]]
        if klass != current:
            lines.append(f"$ELEMENT TYPE = {klass}")
            current = klass
        spelled = " ".join(map(str, spelled_conn[starts[i] : starts[i + 1]]))
        lines.append(f"    {elem_ids_list[i]} {spelled}")

    out_of_range = 0
    unwritten = 0
    set_names: set[str] = set()
    for keyword, tags, ids, count in (
        ("NSET", poly.vertex_tags or {}, node_ids, n_verts),
        ("ESET", poly.element_tags or {}, elem_ids, n_elems),
    ):
        # One namespace per set kind: PERMAS keeps node and element sets apart,
        # so a tag name used for both is written under both.
        set_names.clear()
        for name, members in tags.items():
            # A member spelled twice is written once, at its first spelling,
            # which is the order a reader hands back.
            values = member_values(members)
            _, first = np.unique(values, return_index=True)
            idx = values[np.sort(first)]
            kept = idx[(idx >= 0) & (idx < count)]
            out_of_range += idx.size - kept.size
            spelled_ids = ids[kept]
            if keyword == "ESET":
                live = written[kept]
                unwritten += int(kept.size - live.sum())
                spelled_ids = spelled_ids[live]
            lines.append(f"${keyword} NAME = {_safe_name(name, set_names, renamed)}")
            values = spelled_ids.tolist()
            for start in range(0, len(values), _SET_MEMBERS_PER_LINE):
                chunk = values[start : start + _SET_MEMBERS_PER_LINE]
                lines.append("    " + " ".join(str(v) for v in chunk))

    lines.extend(["$END STRUCTURE", "$EXIT COMPONENT", "$FIN"])

    if non_finite:
        warnings.warn(
            f"{EXTENSION}: {non_finite} node(s) have non-finite coordinates;"
            " PERMAS will not load the file.",
            stacklevel=2,
        )
    if unspellable:
        warnings.warn(
            f"{EXTENSION}: element type(s) {sorted(unspellable)} have no PERMAS"
            " class; those elements were dropped, and a set naming one leaves"
            " it out.",
            stacklevel=2,
        )
    if malformed:
        warnings.warn(
            f"{EXTENSION}: {malformed} element(s) carry a node count that does"
            " not match their type; dropped.",
            stacklevel=2,
        )
    if unused:
        warnings.warn(
            f"{EXTENSION}: element_type= names {unused}, which this mesh has"
            " no element of; ignored.",
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
            " not written; left out of their set.",
            stacklevel=2,
        )
    if renamed:
        warnings.warn(
            f"{EXTENSION}: a name cannot carry whitespace, '=', '!', '$', '&'"
            f" or a repeat on a record; wrote {sorted(renamed)}.",
            stacklevel=2,
        )

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
