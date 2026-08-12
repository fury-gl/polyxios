"""Nastran .bdf bulk data codec — read + write.

Reads free-field (comma separated), small-field (8-column) and large-field
(16-column) bulk data cards, including continuation lines. Writes free-field
cards, or large-field ``GRID*`` cards on request.

Registered for ``.bdf``, ``.nas`` and ``.fem``. A deck named ``.dat`` reads
through ``read(path, fmt=".bdf")``; see ``EXTENSIONS`` for why that one is
not claimed outright.
"""

import bisect
from collections.abc import Iterator
import math
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".bdf"

# A bulk data deck ships under several names: '.bdf' is the canonical one,
# '.nas' is the Nastran input spelling and '.fem' is what Altair OptiStruct
# writes. '.dat' is deliberately absent — LS-DYNA, Tecplot and plain ASCII
# tables all claim it, so binding it here would hand every '.dat' in the
# world to this codec and foreclose the extension for the others. Read one
# with ``read(path, fmt=".bdf")``.
EXTENSIONS: tuple[str, ...] = (".bdf", ".nas", ".fem")

# Bulk data is ASCII, but comments and INCLUDE paths written by a
# pre-processor need not be, and a stray byte there must not sink the read.
_READ_ENCODING: str = "utf-8-sig"
_WRITE_ENCODING: str = "utf-8"

# Nastran bulk card → (polyxios name, n_nodes). Higher-order and revised
# variants share the shape of their linear counterpart: only the corner grid
# points (the leading n_nodes fields) are kept.
_CARD_TO_POLYXIOS: dict[str, tuple[str, int]] = {
    "CTRIA3": ("triangle", 3),
    "CTRIA6": ("triangle", 3),
    "CTRIAR": ("triangle", 3),
    "CQUAD4": ("quad", 4),
    "CQUAD8": ("quad", 4),
    "CQUADR": ("quad", 4),
    "CQUAD": ("quad", 4),
    "CTETRA": ("tetra", 4),
    "CPYRAM": ("pyramid", 5),
    "CPYRA": ("pyramid", 5),
    "CPENTA": ("wedge", 6),
    "CHEXA": ("hexahedron", 8),
}

# Write mapping: one canonical card per polyxios element type.
_POLYXIOS_TO_CARD: dict[str, str] = {
    "triangle": "CTRIA3",
    "quad": "CQUAD4",
    "tetra": "CTETRA",
    "pyramid": "CPYRAM",
    "wedge": "CPENTA",
    "hexahedron": "CHEXA",
}

# Cards starting with 'C' that are not element connectivity cards; used to
# keep the "skipped element card" warning free of false positives.
_NON_ELEMENT_C_CARDS: frozenset[str] = frozenset(
    {
        "CAERO1",
        "CAERO2",
        "CAERO3",
        "CAERO4",
        "CAERO5",
        "CBARAO",
        "CEND",
        "CGEN",
        "CLOAD",
        "CMFREE",
        "CONV",
        "CONVM",
        "CORD1C",
        "CORD1R",
        "CORD1S",
        "CORD2C",
        "CORD2R",
        "CORD2S",
        "CORD3G",
        "CSET",
        "CSET1",
        "CSSCHD",
        "CSUPER",
        "CSUPEXT",
        "CYAX",
        "CYJOIN",
        "CYSUP",
        "CYSYM",
    }
)

# A physical card line holds 10 fields; the 10th is the continuation marker,
# so only 9 (name + 8 data) carry payload. Small field = 8 columns each,
# large field = 16 columns for the 4 data fields.
_FIELDS_PER_LINE: int = 9
_PAYLOAD_END: int = 72
_LARGE_WIDTH: int = 16

# A bulk data record is read from an 80 column line whatever the field
# format, so a free-field card carrying long reals breaks on width too.
_LINE_LIMIT: int = 80

# Nastran keeps only the first eight characters of a free-field entry, the
# same precision a small-field card carries.
_FREE_SIGNIFICANT: int = 8

# Enough digits to round-trip any double, the starting point when trimming
# a real down to a fixed field width.
_MAX_DIGITS: int = 17

# A solver truncating a free-field entry to its first eight characters only
# rounds it when what falls off is fractional digits; cutting an exponent or
# an integer digit moves the value by orders of magnitude instead. Below one,
# an exact fixed-point spelling always truncates safely, so it is preferred
# over the exponent form up to this many decimals — past that the field grows
# unreasonable and the exponent form is kept, with a warning.
_MAX_FIXED_DECIMALS: int = 30

# Ids travel through int32 arrays, so anything wider is refused rather than
# wrapped around silently.
_INT32_MAX: int = 2**31 - 1

# Property ids have no name in the deck, so they travel as element_attrs
# ["pid"] plus one element_tags entry per distinct id.
_PID_TAG_PREFIX: str = "pid_"
_PID_TAG_RE = re.compile(rf"^{_PID_TAG_PREFIX}(\d+)$", re.IGNORECASE)
_DEFAULT_PID: int = 1


def _is_continuation(line: str) -> bool:
    """Tell whether a physical line continues the preceding card.

    A continuation carries a blank first field or one starting with ``+``
    or ``*``.

    Parameters
    ----------
    line
        Raw physical line, comments already stripped.

    Returns
    -------
    bool
    """
    first = line.split(",", 1)[0].strip() if "," in line else line[:8].strip()
    return not first or first[0] in "+*"


def _names_a_card(field: str) -> bool:
    """Tell whether a ``+`` or ``*`` field could name a continuation.

    A marker name is an unsigned integer (``+11``, ``*2``) or plain text
    (``+ABC``); a real is not, since no marker name carries a decimal point
    or an exponent. Only :func:`_is_marker` rules out the text case, so this
    is the integer half of the question and expects a field it already
    cleared.

    Parameters
    ----------
    field
        Stripped field text, sigil included.

    Returns
    -------
    bool
    """
    body = field[1:]
    if "_" in body:
        return False
    try:
        int(body)
    except ValueError:
        return False
    return True


def _is_marker(field: str) -> bool:
    """Tell whether a trailing field is a continuation marker, not data.

    Used for lines that stop short of the tenth field, where position alone
    cannot decide. Free-field reals may carry an explicit plus sign
    (``+3.``), so a field only counts as a marker when what follows the
    ``+`` or ``*`` is not a number.

    Parameters
    ----------
    field
        Stripped field text.

    Returns
    -------
    bool
    """
    if field[:1] not in ("+", "*"):
        return False
    if len(field) == 1:
        return True
    try:
        _parse_real(field[1:])
    except ValueError:
        return True
    return False


def _split_card_line(line: str, *, width: int | None = None) -> tuple[list[str], bool]:
    """Split one physical line into stripped Nastran fields.

    Detects the field format from the line itself: a comma makes it free
    field, a ``*`` in the name field makes it large field (16 columns),
    otherwise small field (8 columns). The trailing continuation marker is
    dropped; blank fields are kept, since they hold a card position.

    The tenth field of a line is the continuation field by definition, so a
    free-field line is cut there whatever it holds — an unnamed marker is
    blank and a named one is often numeric (``+11``). Fixed-field markers
    sit past column 72, which the payload slice drops: a line reaching that
    far has its marker there, so its last payload field is data whatever it
    looks like. Only a line stopping short of column 72 leaves the question
    open, and there :func:`_is_marker` tells a marker from data. A ``+``
    prefixed integer is ambiguous either way, so it is reported back rather
    than dropped, and the next line decides. A ``+`` prefixed real (``+1.5``)
    is not: a continuation marker names a card, and no marker name carries a
    decimal point or an exponent, so it stays data.

    Parameters
    ----------
    line
        Raw physical line, comments already stripped.
    width
        Field width inherited from the card being continued, used only when
        the line's marker field is blank — a large-field continuation may
        leave it blank, and nothing in the line itself then tells the
        16-column layout from the 8-column one. A marker of its own settles
        it: ``*`` means large field, ``+`` means small, whatever the card
        being continued used.

    Returns
    -------
    list of str
        Fields, index 0 being the card name (or the continuation marker).
    bool
        Whether the last field may still be a continuation marker rather
        than data; only a line continuing this card settles it.

    Raises
    ------
    CodecError
        If a free-field line carries data past its tenth field.
    """
    if "," in line:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) > _FIELDS_PER_LINE:
            if any(fields[_FIELDS_PER_LINE + 1 :]):
                raise CodecError(
                    f".bdf: free-field line holds more than {_FIELDS_PER_LINE + 1}"
                    f" fields: {line.strip()!r}"
                )
            return fields[:_FIELDS_PER_LINE], False
    else:
        name = line[:8].strip()
        if name.endswith("*") or name.startswith("*"):
            width = 16
        elif name.startswith("+") or width is None:
            width = 8
        fields = [name]
        fields += [
            line[j : j + width].strip()
            for j in range(8, min(len(line), _PAYLOAD_END), width)
        ]
        # The card reaches into the continuation field, so whatever marker it
        # carries sits there, past the payload the slice above kept. The last
        # payload field is then data by position, and a '+3' spelling of an
        # id must not be mistaken for a marker and dropped.
        if len(line.rstrip()) > _PAYLOAD_END:
            return fields, False

    if len(fields) > 1 and fields[-1][:1] in ("+", "*"):
        if _is_marker(fields[-1]):
            return fields[:-1], False
        if not _names_a_card(fields[-1]):
            return fields, False
        return fields, True
    return fields, False


def _bulk_cards(text: str) -> Iterator[list[str]]:
    """Yield the logical bulk data cards of a .bdf file body.

    Comments are stripped, the executive and case control sections are
    skipped, parsing stops at ``ENDDATA`` and continuation lines are merged
    into the card they continue.

    Parameters
    ----------
    text
        Full file contents.

    Yields
    ------
    list of str
        One field list per logical card.

    Warns
    -----
    UserWarning
        If a continuation line precedes any card, having nothing to continue.

    Raises
    ------
    CodecError
        If a free-field line carries data past its tenth field.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        # A tab advances to the next 8-column field boundary.
        line = raw.split("$")[0].rstrip().expandtabs(8)
        if line.strip():
            lines.append(line)

    start = 0
    for i, line in enumerate(lines):
        # The delimiter is free format, so the two words may be spaced apart
        # by any run of blanks.
        if " ".join(line.split()).upper().startswith("BEGIN BULK"):
            start = i + 1
            break

    current: list[str] | None = None
    pending = False
    width = 8
    orphans = 0
    for line in lines[start:]:
        if line.strip().upper().startswith("ENDDATA"):
            break
        continues = _is_continuation(line)
        fields, pending_here = _split_card_line(
            line, width=width if continues else None
        )
        if continues:
            if current is None:
                # A continuation with no card ahead of it has nothing to
                # continue; its fields are dropped, so say so rather than
                # lose them quietly.
                orphans += 1
            else:
                # A numeric named marker ('+11') survives _split_card_line,
                # since nothing in that line tells it from a signed integer
                # field. A line continuing the card settles it — Nastran
                # does not require the two markers to match, so the marker
                # this line carries says nothing about it.
                if pending:
                    current.pop()
                current.extend(fields[1:])
                pending = pending_here
            # A marked continuation sets the width the next blank-marked one
            # inherits, so a card may switch format part way through.
            if fields[0][:1] in ("+", "*"):
                width = 16 if fields[0][:1] == "*" else 8
            continue
        if current is not None:
            yield current
        current = fields
        pending = pending_here
        # A '*' leading the first field makes the line a continuation, so a
        # card name can only carry its large-field star at the end.
        width = 16 if fields[0].endswith("*") else 8
    if current is not None:
        yield current
    if orphans:
        warnings.warn(
            f".bdf: {orphans} continuation line(s) precede any card and were dropped",
            stacklevel=3,
        )


def _field(fields: list[str], index: int) -> str:
    """Return a card field, or an empty string when it is absent."""
    return fields[index] if index < len(fields) else ""


def _parse_real(text: str) -> float:
    """Convert Nastran real notation to a float.

    Accepts the implicit-exponent shorthand (``1.5+3``, ``-2.1-4``) and the
    double-precision ``D`` exponent (``1.5D+3``) on top of plain floats.
    Nastran has no notation for infinities or NaN, nor for the digit
    grouping Python's own literals allow (``1_000``), so those are
    rejected.

    Parameters
    ----------
    text
        Stripped field text.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the text is not a Nastran real.
    """
    if "_" in text:
        raise ValueError(f"not a Nastran real: {text!r}")
    try:
        number = float(text)
    except ValueError:
        candidate = text.replace("D", "E").replace("d", "E")
        for i in range(1, len(candidate)):
            if candidate[i] in "+-" and candidate[i - 1] not in "eE+-":
                candidate = f"{candidate[:i]}E{candidate[i:]}"
                break
        number = float(candidate)

    if not math.isfinite(number):
        raise ValueError(f"not a Nastran real: {text!r}")
    return number


def _to_float(value: str, *, ctx: str) -> float:
    """Parse a Nastran real field.

    Blank means zero. Accepts the implicit-exponent shorthand (``1.5+3``,
    ``-2.1-4``) and the double-precision ``D`` exponent (``1.5D+3``).

    Parameters
    ----------
    value
        Raw field text.
    ctx
        Human readable field description, used in the error message.

    Returns
    -------
    float

    Raises
    ------
    CodecError
        If the field is not a valid Nastran real.
    """
    text = value.strip()
    if not text:
        return 0.0
    try:
        return _parse_real(text)
    except ValueError as exc:
        raise CodecError(f".bdf: {ctx} is not a valid real: {value!r}") from exc


def _to_int(value: str, *, ctx: str) -> int:
    """Parse a Nastran integer field.

    Nastran has no digit grouping, so the underscores Python's own literals
    allow (``1_000``) are rejected rather than quietly ignored.

    Parameters
    ----------
    value
        Raw field text.
    ctx
        Human readable field description, used in the error message.

    Returns
    -------
    int

    Raises
    ------
    CodecError
        If the field is not a valid integer.
    """
    text = value.strip()
    if "_" in text:
        raise CodecError(f".bdf: {ctx} is not a valid integer: {value!r}")
    try:
        number = int(text)
    except ValueError as exc:
        raise CodecError(f".bdf: {ctx} is not a valid integer: {value!r}") from exc
    if abs(number) > _INT32_MAX:
        raise CodecError(f".bdf: {ctx} does not fit a 32-bit integer: {value!r}")
    return number


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a Nastran .bdf file.

    Free-field, small-field and large-field cards are supported, in any
    order: elements keep their raw grid ids until the whole deck is read,
    so they may reference grid points defined further down the file.

    Parameters
    ----------
    path
        Path to the bulk data file (``.bdf``, ``.nas`` or ``.fem``). A deck
        named ``.dat`` needs ``polyxios.read(path, fmt=".bdf")``, that
        extension being too widely shared to bind to one codec.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Empty when the deck holds no GRID and no element cards. Property
        ids land in ``element_attrs["pid"]`` and each distinct one also
        becomes an ``element_tags`` entry named ``pid_<id>``.

    Raises
    ------
    CodecError
        If a card is malformed, or if an element references an undefined
        grid point.

    Notes
    -----
    Grid points defined in a local coordinate system (non-zero ``CP``) are
    read as-is and a warning is emitted; no frame transform is applied.
    Mid-side nodes of higher-order cards (e.g. CTETRA with 10 grid points)
    are dropped, keeping the corner nodes only. Element cards whose shape
    has no polyxios counterpart (CBAR, CROD, ...) are skipped with a
    warning, as is a continuation line with no card ahead of it to
    continue. ``INCLUDE`` statements are not followed. A blank property id
    reads as 1; PSHELL, PSOLID and the other property cards are not parsed,
    so the ids carry no name or material.
    """
    if lazy:
        warnings.warn(
            ".bdf: lazy=True ignored; ASCII format always loads eagerly.",
            stacklevel=2,
        )

    node_map: dict[int, int] = {}
    coords: list[float] = []
    grid_ids: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    pid_list: list[int] = []
    # Card name and element id as the deck spells them, kept only so a
    # dangling grid reference can be pointed at the card that made it.
    elem_sources: list[tuple[str, str]] = []
    local_frames = 0
    duplicates = 0
    includes = 0
    skipped: dict[str, int] = {}

    text = Path(path).read_text(encoding=_READ_ENCODING, errors="replace")
    for fields in _bulk_cards(text):
        # The large-field star only has to sit somewhere in the eight column
        # name field, so 'GRID   *' names the same card as 'GRID*'. Strip the
        # star wherever it landed, then the padding it left behind.
        card = fields[0].replace("*", "").strip().upper()

        if card == "GRID":
            node_id = _to_int(_field(fields, 1), ctx="GRID id")
            cp = _field(fields, 2)
            if cp and _to_int(cp, ctx=f"GRID {node_id} coordinate system") != 0:
                local_frames += 1
            xyz = [
                _to_float(_field(fields, 3 + j), ctx=f"GRID {node_id} coordinate")
                for j in range(3)
            ]
            existing = node_map.get(node_id)
            if existing is not None:
                duplicates += 1
                coords[3 * existing : 3 * existing + 3] = xyz
                continue
            node_map[node_id] = len(coords) // 3
            coords.extend(xyz)
            continue

        # A comma in the quoted path makes the line look free field, so the
        # name field arrives glued to the head of the path ("INCLUDE 'PART").
        # Match on the prefix, or a skipped INCLUDE goes unreported.
        if card.startswith("INCLUDE"):
            includes += 1
            continue

        info = _CARD_TO_POLYXIOS.get(card)
        if info is None:
            if card.startswith("C") and card not in _NON_ELEMENT_C_CARDS:
                skipped[card] = skipped.get(card, 0) + 1
            continue

        elem_name, n_nodes = info
        eid = _field(fields, 1) or "?"
        node_fields = [_field(fields, 3 + j) for j in range(n_nodes)]
        if any(not f for f in node_fields):
            raise CodecError(
                f".bdf: {card} {eid} is missing grid point fields; {n_nodes}"
                " are required"
            )
        grid_ids.extend(_to_int(f, ctx=f"{card} {eid} grid point") for f in node_fields)
        offsets_list.append(offsets_list[-1] + n_nodes)
        types_list.append(ELEMENT_TYPES[elem_name])
        elem_sources.append((card, eid))
        pid = _field(fields, 2)
        pid_list.append(
            _to_int(pid, ctx=f"{card} {eid} property id") if pid else _DEFAULT_PID
        )

    if duplicates:
        warnings.warn(
            f".bdf: {duplicates} duplicate GRID id(s); kept the last definition",
            stacklevel=2,
        )
    if local_frames:
        warnings.warn(
            f".bdf: {local_frames} grid point(s) reference a local coordinate"
            " system (CP); coordinates are read without transforming them",
            stacklevel=2,
        )
    if includes:
        warnings.warn(
            f".bdf: {includes} INCLUDE statement(s) ignored; the referenced"
            " files were not read",
            stacklevel=2,
        )
    if skipped:
        names = ", ".join(f"{name} ({n})" for name, n in sorted(skipped.items()))
        warnings.warn(
            f".bdf: skipped {sum(skipped.values())} element card(s) of"
            f" unsupported type: {names}",
            stacklevel=2,
        )

    try:
        connectivity = np.array(
            [node_map[grid_id] for grid_id in grid_ids], dtype=np.int32
        )
    except KeyError as exc:
        missing = exc.args[0]
        index = bisect.bisect_right(offsets_list, grid_ids.index(missing)) - 1
        card, eid = elem_sources[index]
        raise CodecError(
            f".bdf: {card} {eid} references undefined GRID {missing}"
        ) from exc

    n_verts = len(coords) // 3
    vertices = np.array(coords, dtype=np.float64).reshape(n_verts, 3)

    element_attrs: dict[str, np.ndarray] = {}
    element_tags: dict[str, np.ndarray] = {}
    if pid_list:
        pids = np.array(pid_list, dtype=np.int32)
        element_attrs["pid"] = pids
        # One flatnonzero per distinct id rescans the whole array each time,
        # which a deck carrying thousands of properties feels; a single
        # stable sort groups every id in one pass, ids ascending and members
        # ascending inside a group, as the scan produced them.
        order = np.argsort(pids, kind="stable").astype(np.int32)
        ranked = pids[order]
        starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
        element_tags = {
            f"{_PID_TAG_PREFIX}{int(pid)}": members
            for pid, members in zip(ranked[starts], np.split(order, starts[1:]))
        }

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_attrs=element_attrs,
        element_tags=element_tags,
    )


def _fmt_real(value: float, *, width: int | None = None) -> str:
    """Format a float as a Nastran real field.

    Uses the shortest representation that round-trips the double exactly,
    so coordinates survive a write/read cycle bit for bit. Nastran real
    fields must carry a decimal point, so integral values and exponent
    forms are padded with one. When ``width`` is given and that exact form
    is too long, significant digits are dropped until it fits, which costs
    precision.

    Parameters
    ----------
    value
        Value to format.
    width
        Field width the text has to fit in, or None for no limit.

    Returns
    -------
    str

    Raises
    ------
    CodecError
        If the value is not finite, or cannot be written in ``width``
        characters at all.
    """
    number = float(value)
    if not math.isfinite(number):
        raise CodecError(f".bdf: cannot write non-finite coordinate {number!r}")
    mantissa, sep, exponent = repr(number).partition("e")
    if "." not in mantissa:
        mantissa += "."
    # Bulk data spells an exponent 'E' or 'D'; Python's repr spells it 'e',
    # which a strict reader rejects.
    text = f"{mantissa}{sep.upper()}{exponent}"

    if width is None or len(text) <= width:
        return text

    # %G picks the shorter of the fixed and exponent forms and drops
    # trailing zeros, so it fits more significant digits in the field than
    # a plain exponent format would.
    for digits in range(_MAX_DIGITS, 0, -1):
        mantissa, sep, exponent = f"{number:.{digits}G}".partition("E")
        if "." not in mantissa:
            mantissa += "."
        candidate = f"{mantissa}{sep}{exponent}"
        # Dropping digits rounds the mantissa, and rounding up at the top of
        # the double range steps past it: 1.7976931348623157E+308 shortens to
        # '1.797693135E+308', which fits sixteen characters and reads back as
        # infinity. Keep only a candidate that still names a finite value, so
        # the search falls through to the next shorter form instead.
        if len(candidate) <= width and math.isfinite(float(candidate)):
            return candidate
    raise CodecError(f".bdf: cannot write {number!r} in a {width}-character field")


def _survives_truncation(text: str) -> bool:
    """Tell whether the first eight characters still name the same value.

    A solver keeps only that much of a free-field entry. Dropping the tail
    of a plain decimal drops fractional digits, which merely rounds the
    value; cutting an exponent (``1.2345678901234E-05`` → ``1.234567``) or
    an integer digit (``123456789012345.6`` → ``12345678``) moves it by
    orders of magnitude.

    Parameters
    ----------
    text
        Field text as it would be written.

    Returns
    -------
    bool
    """
    if len(text) <= _FREE_SIGNIFICANT:
        return True
    if "E" in text.upper():
        return False
    return len(text.partition(".")[0]) <= _FREE_SIGNIFICANT


def _exact_fixed_point(number: float, text: str) -> str | None:
    """Spell a magnitude below one in fixed point, exactly, or give up.

    The exponent form already says how many decimals that takes: the digits
    behind its point, shifted down by the exponent. Searching for the count
    instead would cost one format call per candidate, on every coordinate of
    a mesh whose whole scale sits below one.

    Parameters
    ----------
    number
        Value to spell; only magnitudes below one are worth asking about,
        every other fixed-point form being longer than the exponent one.
    text
        The value's exponent form, as :func:`_fmt_real` gives it.

    Returns
    -------
    str or None
        Shortest fixed-point text that reads back as ``number``, or None
        when it would run past :data:`_MAX_FIXED_DECIMALS` decimals.
    """
    mantissa, _, exponent = text.partition("E")
    if not exponent:
        return None
    decimals = len(mantissa.partition(".")[2]) - int(exponent)
    if not 0 < decimals <= _MAX_FIXED_DECIMALS:
        return None
    fixed = f"{number:.{decimals}f}"
    return fixed if float(fixed) == number else None


def _free_field_real(value: float) -> str:
    """Format a float as a free-field real a solver cannot misread.

    The exact spelling wins whenever an eight character truncation leaves
    it recognisable. When it does not — an exponent form below one, say —
    an exact fixed-point spelling is tried instead: it is longer, but what
    a solver cuts off it is fractional digits rather than the exponent.

    Parameters
    ----------
    value
        Value to format.

    Returns
    -------
    str

    Raises
    ------
    CodecError
        If the value is not finite.
    """
    text = _fmt_real(value)
    number = float(value)
    if _survives_truncation(text) or not 0.0 < abs(number) < 1.0:
        return text
    fixed = _exact_fixed_point(number, text)
    if fixed is not None and _survives_truncation(fixed):
        return fixed
    return text


def _card_lines(fields: list[str]) -> list[str]:
    """Render a free-field card as physical lines with continuations.

    A line ends at the ninth field, the tenth being the continuation field,
    or at column 80, the width of a bulk data record — a long real reaches
    the second bound well before the first.

    Parameters
    ----------
    fields
        Card name followed by its data fields.

    Returns
    -------
    list of str
        One entry per physical line.
    """
    lines: list[str] = []
    rest = list(fields)
    while rest:
        # The opening line leads with the card name, the rest with the marker
        # tying them to it.
        head = ["+"] if lines else []
        chunk: list[str] = []
        while rest and len(head) + len(chunk) < _FIELDS_PER_LINE:
            # A continuation costs a ',+' this line may still have to pay, so
            # budget for it whether or not more fields follow. The first field
            # of a line goes in unweighed, or an over-wide one would stall.
            width = len(",".join([*head, *chunk, rest[0]])) + 2
            if chunk and width > _LINE_LIMIT:
                break
            chunk.append(rest.pop(0))
        lines.append(",".join([*head, *chunk, *(["+"] if rest else [])]))
    return lines


def _grid_lines(
    index: int, xyz: np.ndarray, *, large: bool
) -> tuple[list[str], int, int]:
    """Render one GRID card, free field or large field.

    Parameters
    ----------
    index
        Zero-based vertex index; the grid id is ``index + 1``.
    xyz
        The three coordinates of the vertex.
    large
        Write a 16-column ``GRID*`` card instead of a free-field one.

    Returns
    -------
    list of str
        Physical lines of the card.
    int
        Number of coordinates the field width cannot hold exactly.
    int
        Number of coordinates no free-field spelling keeps recognisable
        under a solver's eight character truncation; always 0 in large
        field, whose sixteen characters the text already fits.
    """
    if not large:
        texts = [_free_field_real(value) for value in xyz]
        # Free field keeps only the first eight characters of a field, so a
        # longer one reads back exactly here but rounded in a solver — or,
        # when even the fixed-point spelling stays out of reach, wrong.
        lost = sum(len(text) > _FREE_SIGNIFICANT for text in texts)
        misread = sum(not _survives_truncation(text) for text in texts)
        return _card_lines(["GRID", str(index + 1), "", *texts]), lost, misread

    texts = [_fmt_real(value, width=_LARGE_WIDTH) for value in xyz]
    lost = sum(float(text) != float(value) for text, value in zip(texts, xyz))

    ids = "".join(f"{text:<{_LARGE_WIDTH}}" for text in (str(index + 1), ""))
    body = "".join(f"{text:<{_LARGE_WIDTH}}" for text in texts[:2])
    return [f"{'GRID*':<8}{ids}{body}".rstrip(), f"{'*':<8}{texts[2]}"], lost, 0


def _stored_pids(poly: PolyData, n_elems: int) -> np.ndarray | None:
    """Return ``element_attrs["pid"]`` as int32, or None when unusable.

    A wrong length or a non-integral payload is reported and dropped rather
    than written out as a bogus property id.

    Parameters
    ----------
    poly
        PolyData being written.
    n_elems
        Number of elements in the mesh.

    Returns
    -------
    numpy.ndarray or None
    """
    stored = poly.element_attrs.get("pid")
    if stored is None:
        return None

    values = np.asarray(stored)
    if values.shape != (n_elems,):
        warnings.warn(
            f".bdf: element_attrs['pid'] has shape {values.shape}, expected"
            f" ({n_elems},), and was ignored; property ids were"
            " derived from element_tags instead.",
            stacklevel=4,
        )
        return None

    try:
        # A NaN or out of range float raises the invalid/overflow flag rather
        # than an exception, and numpy turns that into a RuntimeWarning a
        # warnings filter may promote to an error. The value it does produce
        # fails the equality check below, which is the answer wanted here.
        with np.errstate(invalid="ignore", over="ignore"):
            pids = values.astype(np.int32)
    except (TypeError, ValueError):
        pids = None
    if pids is None or not np.array_equal(pids, values):
        warnings.warn(
            ".bdf: element_attrs['pid'] is not integral, or does not fit a"
            " 32-bit integer, and was ignored; property ids were derived"
            " from element_tags instead.",
            stacklevel=4,
        )
        return None
    return pids


def _element_pids(poly: PolyData) -> np.ndarray:
    """Resolve the Nastran property id of every element.

    ``element_attrs["pid"]`` wins when it is usable; otherwise ids come from
    the ``pid_<id>`` entries of ``element_tags``, first tag winning for an
    element that belongs to several. Anything left over, and any id Nastran
    would reject, falls back to 1.

    Parameters
    ----------
    poly
        PolyData being written.

    Returns
    -------
    numpy.ndarray
        Property ids, dtype int32, length n_elements.
    """
    n_elems = len(poly.element_types)
    pids = _stored_pids(poly, n_elems)

    if pids is None:
        pids = np.full(n_elems, _DEFAULT_PID, dtype=np.int32)
        assigned = np.zeros(n_elems, dtype=bool)
        for name, members in poly.element_tags.items():
            match = _PID_TAG_RE.match(name)
            if match is None or int(match.group(1)) > _INT32_MAX:
                continue
            idx = np.asarray(members, dtype=np.int64).ravel()
            idx = idx[(idx >= 0) & (idx < n_elems)]
            idx = idx[~assigned[idx]]
            pids[idx] = int(match.group(1))
            assigned[idx] = True

    invalid = pids < 1
    if invalid.any():
        warnings.warn(
            f".bdf: {int(invalid.sum())} element(s) carry a property id below"
            f" 1, which Nastran rejects; wrote {_DEFAULT_PID} instead.",
            stacklevel=3,
        )
        pids = np.where(invalid, _DEFAULT_PID, pids)
    return pids


def write(
    poly: PolyData,
    path: Path | str,
    *,
    field_format: str = "free",
    **opts: Any,
) -> None:
    """Write PolyData to a Nastran .bdf deck.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output path. The deck is the same whichever of ``.bdf``, ``.nas``
        and ``.fem`` names it; ``.bdf`` is the canonical spelling.
    field_format
        ``"free"`` writes comma separated cards, ``"large"`` writes
        16-column ``GRID*`` cards. Element cards stay free field either
        way, their fields being short integers.
    **opts
        Ignored; accepted for signature compatibility.

    Raises
    ------
    CodecError
        If ``field_format`` is unknown, ``vertices`` is not shaped
        ``(n_vertices, 3)``, ``offsets`` does not hold one entry per element
        plus a closing one, ``offsets`` ends past the end of
        ``connectivity``, an element type id is unknown, has no Nastran
        write mapping, carries an unexpected number of grid points, or
        references a vertex outside the vertex array.

    Notes
    -----
    Elements keep their input order, and grid and element ids are
    renumbered from 1 in array order: the ids a deck carried before a read
    are not kept, so a read/write cycle preserves the mesh but not its
    numbering. Property ids come from ``element_attrs["pid"]``, or failing
    that from ``element_tags`` entries named ``pid_<id>``, and default to
    1; no property or material cards are emitted, so the ids stay
    undefined in the deck.

    Coordinates are written with the shortest text that round-trips the
    double exactly, which free field has the room for: a deck written that
    way and read back by polyxios is bit-for-bit identical. Nastran itself
    keeps only the first eight characters of a free-field entry, so values
    needing more are read rounded by a solver and a warning says so. A
    magnitude below one whose short spelling carries an exponent is written
    in fixed point instead, longer but truncating to the right magnitude
    rather than to a stray mantissa; the few values no spelling saves —
    very large or very small exponents — get a warning of their own.
    ``field_format="large"`` raises the solver's budget to sixteen
    characters, which holds ten to fourteen significant digits depending on
    the sign and the exponent, but caps the written text at sixteen too:
    a value needing more is rounded on the way out, with a warning, and
    then no longer round-trips exactly.
    """
    if field_format not in ("free", "large"):
        raise CodecError(
            f".bdf: unknown field_format {field_format!r}; use 'free' or 'large'"
        )
    large = field_format == "large"

    vertices = np.asarray(poly.vertices)
    # A GRID card carries exactly three coordinates, so a row holding fewer
    # writes a card short of one — silently in free field, and as a bare
    # IndexError in large field. An empty mesh emits no GRID at all, so its
    # shape is nobody's business.
    if vertices.size and (vertices.ndim != 2 or vertices.shape[1] != 3):
        raise CodecError(
            f".bdf: vertices has shape {vertices.shape}, expected (n_vertices, 3)"
        )

    n_verts = len(vertices)
    n_elems = len(poly.element_types)
    connectivity = np.asarray(poly.connectivity)
    if connectivity.size and (
        int(connectivity.min()) < 0 or int(connectivity.max()) >= n_verts
    ):
        raise CodecError(
            ".bdf: connectivity references a vertex outside the vertex array"
            f" of length {n_verts}"
        )
    # One offset per element plus the closing one; without this the element
    # loop below walks off the end with a bare IndexError.
    if len(poly.offsets) != n_elems + 1:
        raise CodecError(
            f".bdf: offsets has length {len(poly.offsets)}, expected"
            f" {n_elems + 1} for {n_elems} element(s)"
        )
    # The closing offset says where the connectivity array ends. A shorter
    # one hands the element loop a truncated slice, and the grid point count
    # is checked against the offsets rather than the slice, so the card goes
    # out missing a grid point instead of the write failing.
    if int(poly.offsets[-1]) > connectivity.size:
        raise CodecError(
            f".bdf: offsets end at {int(poly.offsets[-1])} but connectivity holds"
            f" {connectivity.size} entries"
        )

    pids = _element_pids(poly)
    lines: list[str] = ["$ Nastran BDF exported by polyxios", "BEGIN BULK"]

    lost = 0
    misread = 0
    for i, vertex in enumerate(vertices):
        card_lines, lost_here, misread_here = _grid_lines(i, vertex, large=large)
        lines.extend(card_lines)
        lost += lost_here
        misread += misread_here

    if lost and large:
        warnings.warn(
            f".bdf: {lost} coordinate(s) do not fit a {_LARGE_WIDTH}-character"
            " large field and were rounded.",
            stacklevel=2,
        )
    elif lost:
        warnings.warn(
            f".bdf: {lost} coordinate(s) need more than {_FREE_SIGNIFICANT}"
            " characters; Nastran keeps only the first"
            f" {_FREE_SIGNIFICANT} of a free-field entry, so a solver reads"
            " them rounded. Pass field_format='large' to keep more digits.",
            stacklevel=2,
        )
    if misread:
        warnings.warn(
            f".bdf: {misread} coordinate(s) have no free-field spelling that"
            f" survives being cut to {_FREE_SIGNIFICANT} characters; a solver"
            " loses their exponent and reads a different magnitude, not a"
            " rounded value. Pass field_format='large'.",
            stacklevel=2,
        )

    for ei in range(n_elems):
        type_id = int(poly.element_types[ei])
        name = ELEMENT_TYPES_INV.get(type_id)
        if name is None:
            raise CodecError(f".bdf: unknown element type id {type_id}")
        card = _POLYXIOS_TO_CARD.get(name)
        if card is None:
            raise CodecError(f".bdf: no write mapping for element type '{name}'")

        start, end = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
        expected = _CARD_TO_POLYXIOS[card][1]
        if end - start != expected:
            raise CodecError(
                f".bdf: element {ei} of type '{name}' has {end - start} grid"
                f" points, expected {expected}"
            )
        nodes = [str(node + 1) for node in connectivity[start:end].tolist()]
        lines.extend(_card_lines([card, str(ei + 1), str(int(pids[ei])), *nodes]))

    lines.append("ENDDATA")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding=_WRITE_ENCODING)
