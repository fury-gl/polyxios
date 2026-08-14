"""Tecplot ASCII codec (finite-element zone) — read + write.

Reads a single finite-element zone under either POINT or BLOCK packing, in
both the classic ``F=FEPOINT, ET=TRIANGLE`` spelling and the modern
``DATAPACKING=POINT, ZONETYPE=FETRIANGLE`` one. Variables past the node
coordinates travel as ``vertex_attrs``. Zones are written back under POINT
packing.

Registered for ``.tec``. Tecplot's other common spelling, ``.dat``, is
deliberately not claimed — LS-DYNA, Nastran and plain ASCII tables all use it
too, so a file named that way reads through ``read(path, fmt=".tec")``.
"""

from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    MAX_SAFE_CONN,
    MAX_SAFE_ELEMENTS,
    MAX_SAFE_VERTICES,
)
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".tec"

# Classic ``ET=`` spellings and the modern ``ZONETYPE=`` ones both land here;
# a file may use either, and older writers emit both at once.
_ET_TO_POLYXIOS: dict[str, tuple[str, int]] = {
    "TRIANGLE": ("triangle", 3),
    "QUADRILATERAL": ("quad", 4),
    "TETRAHEDRON": ("tetra", 4),
    "BRICK": ("hexahedron", 8),
    "FETRIANGLE": ("triangle", 3),
    "FEQUADRILATERAL": ("quad", 4),
    "FETETRAHEDRON": ("tetra", 4),
    "FEBRICK": ("hexahedron", 8),
}
_POLYXIOS_TO_ET: dict[str, str] = {
    "triangle": "TRIANGLE",
    "quad": "QUADRILATERAL",
    "tetra": "TETRAHEDRON",
    "hexahedron": "BRICK",
}

# POINT packing writes one record per node; BLOCK packing writes each variable
# as its own run over every node. The two need different readers, and reading
# one as the other yields silent garbage, so the spelling is never guessed.
_POINT_PACKING: frozenset[str] = frozenset({"FEPOINT", "POINT"})
_BLOCK_PACKING: frozenset[str] = frozenset({"FEBLOCK", "BLOCK"})

# ``KEY = VALUE`` with the value either quoted or a bare token. Consuming a
# quoted value whole is what keeps a title such as T="N=5 grid" from being
# mistaken for a node count.
_KV_RE = re.compile(r"([A-Za-z_]+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s,]+)")

# Lines that open a new record rather than continue the ZONE header. The word
# boundary is what keeps a ``ZONETYPE=FETRIANGLE`` continuation line — the
# spelling Tecplot itself writes, one key per line — from reading as a new ZONE.
_SECTION_RE = re.compile(
    r"(?:ZONE|TITLE|VARIABLES|TEXT|GEOMETRY|CUSTOMLABELS|DATASETAUXDATA)\b"
)
_ZONE_RE = re.compile(r"ZONE\b")

# A ``KEY =`` line ends a VARIABLES list: names are bare or quoted tokens, so
# anything spelling a key is the next record rather than another name.
_HEADER_LINE_RE = re.compile(r"[A-Za-z_]+\s*=")

# ``VARLOCATION=([4]=CELLCENTERED)`` moves a variable off the nodes. Reading
# the clause rather than scanning the header for the bare word is what keeps a
# zone titled ``T="cellcentered run"`` from being refused for its own title.
_VARLOCATION_RE = re.compile(r"VARLOCATION\s*=\s*(\([^)]*\)|[A-Za-z]+)", re.IGNORECASE)

# One variable name: quoted whole, or a bare token. Keeping a quoted name whole
# is what stops ``"Pressure (Pa)"`` from counting as two variables.
_VAR_RE = re.compile(r"\"([^\"]*)\"|'([^']*)'|([^\s,]+)")

# Zone features that move data out of the plain node-then-connectivity layout
# this codec reads. Honouring them means reading a different file entirely, so
# they are refused rather than parsed into silently wrong values.
_UNSUPPORTED_KEYS: tuple[str, ...] = (
    "VARSHARELIST",
    "PASSIVEVARLIST",
    "CONNECTIVITYSHAREZONE",
)

# Keys that decide where the zone's data sits. A header spelling one of them
# twice with two different values cannot be read either way, so the
# disagreement is refused rather than settled by whichever came last.
_CRITICAL_KEYS: frozenset[str] = frozenset(
    {"N", "NODES", "E", "ELEMENTS", "ET", "ZONETYPE", "F", "DATAPACKING"}
)


def _safe_variable_name(name: str) -> str:
    """Return a variable name that survives a quoted Tecplot header.

    Tecplot has no escape inside a quoted name: a double quote closes the
    string early and splits one variable into two, and a line break splits the
    ``VARIABLES`` record itself — a name holding ``"\\nZONE"`` would open a
    second zone header. Both are folded into characters a header can carry.
    """
    folded = name.replace('"', "'")
    return "".join(" " if ch < " " or ch == "\x7f" else ch for ch in folded)


def _checked_count(value: str, cap: int, what: str) -> int:
    """Parse a declared count, rejecting negatives and absurd values."""
    try:
        count = int(value)
    except ValueError as exc:
        raise CodecError(f".tec: non-integer {what} count {value!r}.") from exc
    if count < 0:
        raise CodecError(f".tec: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".tec: {what} count {count} exceeds the safety cap {cap}.")
    return count


def _is_section(line: str) -> bool:
    """Return whether a stripped line opens a new record."""
    return _SECTION_RE.match(line.upper()) is not None


def _is_zone(line: str) -> bool:
    """Return whether a stripped line opens a ZONE record."""
    return _ZONE_RE.match(line.upper()) is not None


def _parse_kv(text: str) -> dict[str, str]:
    """Return the ``KEY=VALUE`` pairs of a header, upper-cased and unquoted."""
    pairs: dict[str, str] = {}
    for key, raw in _KV_RE.findall(text):
        value = raw
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        upper = key.upper()
        previous = pairs.get(upper)
        if (
            previous is not None
            and upper in _CRITICAL_KEYS
            and previous.strip().upper() != value.strip().upper()
        ):
            raise CodecError(
                f".tec: ZONE header declares {upper}= twice, as {previous!r}"
                f" and {value!r}."
            )
        pairs[upper] = value
    return pairs


def _parse_variables(lines: list[str], zone_idx: int) -> list[str]:
    """Collect the variable names declared before the zone, in file order."""
    names: list[str] = []
    for i, ln in enumerate(lines[:zone_idx]):
        stripped = ln.strip()
        if not stripped.upper().startswith("VARIABLES"):
            continue
        body = stripped.split("=", 1)[1] if "=" in stripped else ""
        # The list may spill onto following lines, which carry no keyword.
        j = i + 1
        while j < zone_idx:
            nxt = lines[j].strip()
            # A comment or a blank line punctuates a long list rather than
            # ending it: folding one in would count '#' as a variable name,
            # and stopping at one would drop every name past it along with
            # the column each of those names stands for.
            if not nxt or nxt.startswith("#"):
                j += 1
                continue
            if _is_section(nxt) or _HEADER_LINE_RE.match(nxt):
                break
            body += " " + nxt
            j += 1
        names = [
            double or single or bare
            for double, single, bare in _VAR_RE.findall(body)
            if double or single or bare
        ]
    return names


def _spatial_dim(var_names: list[str], n_vars: int) -> int:
    """Return how many leading variables carry the node coordinates.

    Tecplot has no marker for this: the convention is that the coordinates come
    first and are named X/Y/Z. When the names say so we trust them, so a 2-D
    zone with a solution variable does not read that variable as a Z; otherwise
    we fall back to the first three columns.
    """
    if var_names:
        matched = 0
        for name, axis in zip(var_names, ("X", "Y", "Z")):
            if name.upper() != axis:
                break
            matched += 1
        if matched >= 2:
            return matched
        if n_vars >= 2:
            # The names themselves say a leading variable is not a coordinate,
            # yet the fallback is about to read it as one. Warn wherever the
            # disagreement starts, not only when it starts at the second name.
            warnings.warn(
                f".tec: variable {matched + 1} is named {var_names[matched]!r}, not"
                f" {'XYZ'[matched]!r}; reading the first variables as coordinates"
                " anyway.",
                stacklevel=2,
            )
    return min(3, n_vars)


def _variable_names(var_names: list[str], n_vars: int) -> list[str]:
    """Return ``n_vars`` names, inventing the ones the file left undeclared."""
    if len(var_names) >= n_vars:
        return var_names[:n_vars]
    return [*var_names, *(f"var{k + 1}" for k in range(len(var_names), n_vars))]


def _unique_names(names: list[str]) -> list[str]:
    """Return the solution-variable names made unique.

    They land in a ``vertex_attrs`` dict, so a file that declares the same name
    twice would otherwise lose a column to the collision. Only the solution
    variables are renamed; the coordinates have already been taken out, so an
    attribute named ``"X"`` keeps its name.
    """
    unique: list[str] = []
    used: set[str] = set()
    for name in names:
        candidate, k = name, 1
        while candidate in used:
            k += 1
            candidate = f"{name}_{k}"
        used.add(candidate)
        unique.append(candidate)
    return unique


def _value_count(
    var_names: list[str],
    zone_lines: list[tuple[int, str]],
    n_nodes: int,
    n_conn: int,
    *,
    block: bool,
) -> int:
    """Return how many values each node carries.

    ``VARIABLES`` settles it whenever the file declares it. A zone that omits
    it still has to be read, and guessing three would silently drop a solution
    variable, so the count is recovered from the data instead: from a node's
    own record under POINT packing, and from the size of the value run under
    BLOCK packing.
    """
    if var_names:
        return len(var_names)
    if n_nodes == 0:
        return 3
    total = sum(len(ln.split()) for _, ln in zone_lines)
    if not block:
        if not zone_lines:
            return 3
        first = len(zone_lines[0][1].split())
        if n_nodes * first + n_conn == total:
            return first
        # The first line is not one whole record: it either wraps over the
        # next line or packs several records. The token total still settles
        # the count whenever it divides, so prefer it over a line that has
        # already been shown not to be a record.
        recovered = total - n_conn
        if recovered > 0 and recovered % n_nodes == 0:
            return recovered // n_nodes
        return first
    n_values = total - n_conn
    if n_values <= 0 or n_values % n_nodes:
        raise CodecError(
            ".tec: zone declares no VARIABLES and its value count is not a"
            " multiple of N, so the variables per node cannot be recovered."
        )
    return n_values // n_nodes


def _point_records(
    zone_lines: list[tuple[int, str]],
    n_nodes: int,
    n_vars: int,
    n_elems: int,
    n_per_elem: int,
) -> list[tuple[int, list[str]]]:
    """Return the zone's records as ``(file line, tokens)`` pairs.

    POINT packing normally puts one record on one line, and that layout is kept
    whenever it fits. Writers do wrap a long record over several lines, though,
    so a zone whose lines do not fit but whose token count matches the declared
    counts exactly is regrouped record by record instead of being refused.
    """
    split = [(no, txt.split()) for no, txt in zone_lines]
    fits = (
        len(split) >= n_nodes + n_elems
        and all(len(toks) >= n_vars for _, toks in split[:n_nodes])
        and all(
            len(toks) >= n_per_elem for _, toks in split[n_nodes : n_nodes + n_elems]
        )
    )
    if fits:
        return split

    stream = [(no, tok) for no, toks in split for tok in toks]
    if len(stream) != n_nodes * n_vars + n_elems * n_per_elem:
        # Not a wrapped file either; hand the lines back so the strict readers
        # below report the shortfall against the record they were reading.
        return split

    records: list[tuple[int, list[str]]] = []
    pos = 0
    for width in (*([n_vars] * n_nodes), *([n_per_elem] * n_elems)):
        chunk = stream[pos : pos + width]
        pos += width
        records.append((chunk[0][0], [tok for _, tok in chunk]))
    return records


def _read_point(
    zone_lines: list[tuple[int, str]],
    n_nodes: int,
    n_vars: int,
    n_elems: int,
    n_per_elem: int,
    et: str,
    *,
    cut: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a POINT-packed zone, one record per node and per element."""
    records = _point_records(zone_lines, n_nodes, n_vars, n_elems, n_per_elem)
    if len(records) < n_nodes + n_elems:
        _raise_short(cut, len(records), n_nodes + n_elems, "node and element lines")
    if len(records) > n_nodes + n_elems:
        # The counts the header declares are what the reader trusts, so the
        # tail is dropped — but a zone holding more than it declares is
        # already inconsistent, and reading it silently hides that.
        warnings.warn(
            f".tec: zone holds {len(records) - n_nodes - n_elems} record(s) past"
            " the declared N=/E= counts; ignored.",
            stacklevel=2,
        )

    # A record wider than the zone declares is about to lose its tail to the
    # slice below; that is a column of data leaving, so it is not done quietly.
    wide = [no for no, parts in records[:n_nodes] if len(parts) > n_vars]
    if wide:
        warnings.warn(
            f".tec: {len(wide)} node record(s) carry more than the {n_vars}"
            f" declared values, first at line {wide[0]}; the extra values are"
            " dropped.",
            stacklevel=2,
        )

    values = np.zeros((n_nodes, n_vars), dtype=np.float64)
    for i, (no, parts) in enumerate(records[:n_nodes]):
        if len(parts) < n_vars:
            raise CodecError(
                f".tec: node line {no} has {len(parts)} values,"
                f" expected {n_vars}: {' '.join(parts)!r}."
            )
        try:
            values[i] = [float(tok) for tok in parts[:n_vars]]
        except ValueError as exc:
            raise CodecError(
                f".tec: malformed node line {no}: {' '.join(parts)!r}."
            ) from exc

    nodes = np.empty(n_elems * n_per_elem, dtype=np.int64)
    for i, (no, parts) in enumerate(records[n_nodes : n_nodes + n_elems]):
        if len(parts) != n_per_elem:
            # Too few nodes truncates the element; too many means the line
            # holds a different element type than the zone declares.
            raise CodecError(
                f".tec: element line {no} has {len(parts)} nodes,"
                f" expected {n_per_elem} for ET={et}: {' '.join(parts)!r}."
            )
        try:
            nodes[i * n_per_elem : (i + 1) * n_per_elem] = [
                int(tok) for tok in parts[:n_per_elem]
            ]
        except ValueError as exc:
            raise CodecError(
                f".tec: malformed element line {no}: {' '.join(parts)!r}."
            ) from exc
    return values, nodes


def _read_block(
    zone_lines: list[tuple[int, str]],
    n_nodes: int,
    n_vars: int,
    n_conn: int,
    *,
    cut: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a BLOCK-packed zone, whose values run one variable at a time.

    BLOCK packing wraps a variable's run across as many lines as it likes, so
    the data is consumed as a token stream rather than line by line.
    """
    tokens = [tok for _, ln in zone_lines for tok in ln.split()]
    n_values = n_nodes * n_vars
    if len(tokens) < n_values + n_conn:
        _raise_short(cut, len(tokens), n_values + n_conn, "values")
    if len(tokens) > n_values + n_conn:
        # The counts the header declares are what the reader trusts, so the
        # tail is dropped — but a zone holding more than it declares is
        # already inconsistent, and reading it silently hides that.
        warnings.warn(
            f".tec: zone holds {len(tokens) - n_values - n_conn} value(s) past"
            " the declared N=/E= counts; ignored.",
            stacklevel=2,
        )
    try:
        grid = np.array(tokens[:n_values], dtype=np.float64).reshape(n_vars, n_nodes)
    except ValueError as exc:
        raise CodecError(f".tec: malformed value in the zone's data: {exc}.") from exc
    try:
        nodes = np.array(tokens[n_values : n_values + n_conn], dtype=np.int64)
    except ValueError as exc:
        raise CodecError(f".tec: malformed node reference: {exc}.") from exc
    return np.ascontiguousarray(grid.T), nodes


def _raise_short(cut: bool, got: int, want: int, unit: str) -> None:
    """Report a zone that ran out of data, naming the reason when it is known."""
    if cut:
        raise CodecError(
            f".tec: a new record starts inside the zone's data — got {got} {unit},"
            f" the declared N=/E= counts need {want}."
        )
    raise CodecError(f".tec: file truncated — expected {want} {unit}, got {got}.")


def _reject_unsupported_zone(hdr: dict[str, str], zone_hdr: str) -> None:
    """Refuse a zone whose data does not sit where this codec looks for it."""
    for key in _UNSUPPORTED_KEYS:
        if key in hdr:
            raise CodecError(
                f".tec: zone uses {key}=, which moves data out of the zone;"
                " this codec cannot read it."
            )
    location = _VARLOCATION_RE.search(zone_hdr)
    if location is not None and "CELLCENTERED" in location.group(1).upper():
        # A cell-centred variable holds E values, not N. Reading it as nodal
        # would land element data on the nodes, and when E equals N it would
        # do so without a single count disagreeing.
        raise CodecError(
            ".tec: zone declares a CELLCENTERED variable; only node-centred"
            " variables can be read."
        )


def _resolve_count(hdr: dict[str, str], keys: tuple[str, str], what: str) -> str | None:
    """Return a count declared under either spelling, refusing disagreement."""
    values = [hdr[key] for key in keys if hdr.get(key) is not None]
    if not values:
        return None
    if len({value.strip() for value in values}) > 1:
        raise CodecError(
            f".tec: {keys[0]}={values[0]!r} and {keys[1]}={values[1]!r} declare"
            f" different {what} counts."
        )
    return values[0]


def _resolve_element_type(hdr: dict[str, str]) -> str:
    """Return the zone's element type, refusing ``ET``/``ZONETYPE`` disagreement."""
    spellings = [
        (key, hdr[key]) for key in ("ET", "ZONETYPE") if hdr.get(key) is not None
    ]
    if not spellings:
        raise CodecError(".tec: ZONE header missing ET= / ZONETYPE= field.")
    for _, raw in spellings:
        if raw.upper() not in _ET_TO_POLYXIOS:
            raise CodecError(f".tec: unsupported element type {raw!r}.")
    # Older writers emit both spellings at once; they have to agree, since
    # picking one at random would read the connectivity with the wrong width.
    types = {_ET_TO_POLYXIOS[raw.upper()] for _, raw in spellings}
    if len(types) > 1:
        raise CodecError(
            f".tec: ET={spellings[0][1]!r} and ZONETYPE={spellings[1][1]!r}"
            " name different element types."
        )
    return spellings[0][1].upper()


def _resolve_packing(hdr: dict[str, str]) -> bool:
    """Return whether the zone is BLOCK-packed, refusing a disagreeing header."""
    spellings = [
        (key, hdr[key]) for key in ("F", "DATAPACKING") if hdr.get(key) is not None
    ]
    if not spellings:
        return False
    packings = set()
    for _, raw in spellings:
        packing = raw.upper()
        if packing in _BLOCK_PACKING:
            packings.add(True)
        elif packing in _POINT_PACKING:
            packings.add(False)
        else:
            raise CodecError(f".tec: unsupported data packing {raw!r}.")
    if len(packings) > 1:
        raise CodecError(
            f".tec: F={spellings[0][1]!r} and DATAPACKING={spellings[1][1]!r}"
            " name different data packings."
        )
    return packings.pop()


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a Tecplot ASCII file (finite-element zone).

    Parameters
    ----------
    path
        Path to the .tec file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        On a missing or malformed ZONE header, two header spellings of the same
        field disagreeing, one field declared twice with two values, a zone
        that shares or cell-centres its data, an unsupported element type or
        data packing, a truncated file, or an out-of-range node reference.

    Notes
    -----
    Only the first zone of a multi-zone file is read; the rest are skipped with
    a warning. Variables beyond the node coordinates become ``vertex_attrs``,
    under names made unique when the file declares one twice. POINT records
    wrapped over several lines are joined back together.
    """
    if lazy:
        warnings.warn(
            ".tec: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    # 'utf-8-sig' so a byte-order mark left by a Windows pre-processor does not
    # glue itself to the first keyword and hide the header. errors="replace"
    # keeps a file written in some other 8-bit encoding inside a CodecError —
    # its numbers are ASCII either way, and only a title or comment is hurt.
    lines = Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines()

    zone_idx = -1
    zone_hdr = ""
    data_start = 0
    for i, ln in enumerate(lines):
        if not _is_zone(ln.strip()):
            continue
        zone_idx = i
        zone_hdr = ln
        # The header may spill onto following lines; each carries key=value
        # pairs, so a line that does not look like one ends the header.
        j = i + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or stripped.startswith("#"):
                j += 1
                continue
            if (
                _is_section(stripped)
                or not stripped[0].isalpha()
                or "=" not in stripped
            ):
                break
            zone_hdr += " " + stripped
            j += 1
        data_start = j
        break

    if zone_idx < 0:
        raise CodecError(".tec: missing ZONE header.")

    hdr = _parse_kv(zone_hdr)
    _reject_unsupported_zone(hdr, zone_hdr)

    et = _resolve_element_type(hdr)
    is_block = _resolve_packing(hdr)

    n_raw = _resolve_count(hdr, ("N", "NODES"), "node")
    e_raw = _resolve_count(hdr, ("E", "ELEMENTS"), "element")
    if n_raw is None or e_raw is None:
        raise CodecError(".tec: ZONE header missing N= or E= field.")
    n_nodes = _checked_count(n_raw, MAX_SAFE_VERTICES, "node")
    n_elems = _checked_count(e_raw, MAX_SAFE_ELEMENTS, "element")

    elem_name, n_per_elem = _ET_TO_POLYXIOS[et]
    elem_code = ELEMENT_TYPES[elem_name]
    if n_elems * n_per_elem > MAX_SAFE_CONN:
        raise CodecError(f".tec: connectivity exceeds the safety cap {MAX_SAFE_CONN}.")

    # Line numbers are carried alongside the text so an error can name the line
    # of the file the reader choked on rather than its index within the zone.
    data_lines = [
        (data_start + k + 1, ln)
        for k, ln in enumerate(lines[data_start:])
        if ln.strip() and not ln.strip().startswith("#")
    ]

    # The zone's data runs until the next record opens; everything past that
    # belongs to a zone or annotation this codec does not read.
    end = next(
        (k for k, (_, ln) in enumerate(data_lines) if _is_section(ln.strip())),
        len(data_lines),
    )
    zone_lines = data_lines[:end]
    cut = end < len(data_lines)
    if cut and _is_zone(data_lines[end][1].strip()):
        warnings.warn(
            ".tec: file holds more than one zone; only the first is read.",
            stacklevel=2,
        )

    n_conn = n_elems * n_per_elem
    var_names = _parse_variables(lines, zone_idx)
    n_vars = _value_count(var_names, zone_lines, n_nodes, n_conn, block=is_block)
    n_spatial = _spatial_dim(var_names, n_vars)
    if n_spatial < 2:
        raise CodecError(
            f".tec: need at least 2 coordinate variables, the zone declares {n_vars}."
        )
    names = _variable_names(var_names, n_vars)

    if is_block:
        values, nodes = _read_block(zone_lines, n_nodes, n_vars, n_conn, cut=cut)
    else:
        values, nodes = _read_point(
            zone_lines, n_nodes, n_vars, n_elems, n_per_elem, et, cut=cut
        )

    if nodes.size and (int(nodes.min()) < 1 or int(nodes.max()) > n_nodes):
        # Tecplot node references are 1-based; 0 or an overshoot would wrap to
        # a valid-looking index once shifted, so reject them here.
        raise CodecError(
            f".tec: element references node {int(nodes.min())}..{int(nodes.max())},"
            f" outside 1..{n_nodes}."
        )

    coords = np.zeros((n_nodes, 3), dtype=np.float64)
    coords[:, :n_spatial] = values[:, :n_spatial]
    vertex_attrs = {
        name: np.ascontiguousarray(values[:, n_spatial + k])
        for k, name in enumerate(_unique_names(names[n_spatial:]))
    }

    conn = (nodes - 1).astype(np.int32)

    return PolyData(
        vertices=coords,
        connectivity=conn,
        offsets=np.arange(n_elems + 1, dtype=np.int32) * n_per_elem,
        element_types=np.full(n_elems, elem_code, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Write PolyData to Tecplot ASCII format (FEPOINT).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .tec path.
    **opts
        None are recognized; any that are passed are warned about and ignored.

    Raises
    ------
    CodecError
        If no supported element type is found — which a mesh carrying no
        elements at all also is, since an FE zone has no cells to hold its
        nodes — or an element's node count does not match its type, or an
        element references a vertex that does not exist.

    Notes
    -----
    A Tecplot FE zone holds a single element type, so only the first supported
    type encountered is written; every other element is skipped with a warning.
    Named one-dimensional numeric ``vertex_attrs`` are written as extra
    variables; the rest are skipped with a warning, as is the double quote a
    variable name cannot carry. ``element_attrs`` have no place in a
    node-centred zone and are dropped.
    """
    if opts:
        warnings.warn(
            f".tec write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    # A Tecplot FE zone is single-type: the first supported type wins and the
    # rest of the mesh cannot travel with it.
    et_name: str | None = None
    elem_indices: list[int] = []
    skipped: set[str] = set()
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if name in _POLYXIOS_TO_ET and (et_name is None or name == et_name):
            et_name = name
            elem_indices.append(i)
        else:
            skipped.add(name or f"code {int(poly.element_types[i])}")

    if et_name is None:
        raise CodecError(
            ".tec: no supported element type (need triangle/quad/tetra/hex)."
        )
    if skipped:
        warnings.warn(
            f".tec: a zone holds one element type; wrote {et_name!r} and skipped"
            f" {sorted(skipped)}.",
            stacklevel=2,
        )

    et_str = _POLYXIOS_TO_ET[et_name]

    attr_names: list[str] = []
    attr_arrays: list[np.ndarray] = []
    bad_attrs: set[str] = set()
    for name, arr in (poly.vertex_attrs or {}).items():
        # A nameless variable reads back as no variable at all, taking its
        # column with it, so it is dropped here rather than silently on load.
        if (
            name.strip()
            and arr.ndim == 1
            and arr.shape[0] == n_verts
            and arr.dtype.kind in "fiub"
        ):
            attr_names.append(name)
            attr_arrays.append(arr)
        else:
            bad_attrs.add(name)
    if bad_attrs:
        warnings.warn(
            f".tec: only named 1-D numeric vertex_attrs can be written;"
            f" skipped {sorted(bad_attrs)}.",
            stacklevel=2,
        )

    quoted = [_safe_variable_name(name) for name in attr_names]
    if quoted != attr_names:
        warnings.warn(
            ".tec: a variable name cannot hold a double quote or a line break;"
            f" wrote {sorted(set(quoted) - set(attr_names))} instead.",
            stacklevel=2,
        )
    variables = ", ".join(f'"{n}"' for n in ("X", "Y", "Z", *quoted))
    lines: list[str] = [
        'TITLE = "polyxios mesh"',
        f"VARIABLES = {variables}",
        f'ZONE T="Zone 1", N={n_verts}, E={len(elem_indices)}, F=FEPOINT, ET={et_str}',
    ]
    columns = [poly.vertices[:, 0], poly.vertices[:, 1], poly.vertices[:, 2]]
    columns.extend(attr_arrays)
    lines.extend(" ".join(f"{col[i]:.10g}" for col in columns) for i in range(n_verts))

    n_per_elem = _ET_TO_POLYXIOS[et_str][1]
    for ei in elem_indices:
        s, e = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
        if e - s != n_per_elem:
            raise CodecError(
                f".tec: element {ei} has {e - s} nodes, expected {n_per_elem}"
                f" for {et_name!r}."
            )
        nodes = [int(poly.connectivity[s + j]) for j in range(e - s)]
        for nid in nodes:
            # A reference past the vertex array would be written as a node
            # number the zone never declares — a file no reader can load.
            if not 0 <= nid < n_verts:
                raise CodecError(
                    f".tec: element {ei} references vertex {nid},"
                    f" outside 0..{n_verts - 1}."
                )
        lines.append(" ".join(str(nid + 1) for nid in nodes))
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
