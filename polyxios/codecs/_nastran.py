"""Nastran .bdf bulk data codec - read + write.

Reads free-field (comma separated), small-field (8-column) and large-field
(16-column) bulk data cards, including continuation lines. Writes free-field
cards, or large-field ``GRID*`` cards on request.

Registered for ``.bdf``, ``.nas`` and ``.fem``. A deck named ``.dat`` is
resolved through the registry's sniff hook - see ``SNIFF_EXTENSIONS`` - since
that extension belongs to no one format; ``read(path, fmt=".bdf")`` still
forces the issue.
"""

import bisect
from collections.abc import Iterator
import math
import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, read_text, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".bdf"

# A bulk data deck ships under several names: '.bdf' is the canonical one,
# '.nas' is the Nastran input spelling and '.fem' is what Altair OptiStruct
# writes. '.dat' is deliberately absent here - LS-DYNA, Tecplot and plain
# ASCII tables all claim it, so binding it outright would hand every '.dat'
# in the world to this codec and foreclose the extension for the others. It
# is competed for by content instead; see SNIFF_EXTENSIONS.
EXTENSIONS: tuple[str, ...] = (".bdf", ".nas", ".fem")

SNIFF_EXTENSIONS: tuple[str, ...] = (".dat",)
# Behind Tecplot's: a Tecplot file is known by its very first line, while a
# deck's first real card may sit under a long banner of '$' comments, so the
# narrower test should have its say first.
SNIFF_PRIORITY: int = 50

# Statements that open a deck and belong to no other format sharing '.dat'.
# GRID is the bulk-data workhorse ('GRID*' being its large-field spelling);
# the executive and case-control keywords cover a deck whose geometry sits
# behind BEGIN BULK. A GRID card must be followed by its integer id, which is
# what keeps a table headed 'GRID POINTS OF THE MODEL' out of this codec.
# SOL takes a solution number or a solution name, never a float, so the
# lookahead is what keeps a numeric table whose first row reads 'SOL 1.0 2.0'
# out of this codec; a deck spelling 'SOL STATIC' still carries CEND or
# BEGIN BULK further down the window.
_SNIFF_RE: re.Pattern[str] = re.compile(
    r"(GRID\*?[\s,]+\d|CEND\b|BEGIN\s+BULK\b"
    r"|SOL\s+(?:\d+(?![\d.])|[A-Z])|NASTRAN\s)",
    re.IGNORECASE,
)

# Bulk data is ASCII, but comments and INCLUDE paths written by a
# pre-processor need not be, and a stray byte there must not sink the read.
_READ_ENCODING: str = "utf-8-sig"
_WRITE_ENCODING: str = "utf-8"

# Nastran bulk card → the shapes it can hold, smallest first. A card name
# does not say its order: CTETRA is a 4-node tetrahedron with four grid point
# fields and a 10-node one with ten, and Nastran lets any mid-side field be
# left blank. The reader counts the grid points the card actually carries and
# takes the largest shape that fits, so a linear card is never read as a
# truncated quadratic one and a quadratic one never loses its mid-side nodes.
_CARD_SHAPES: dict[str, tuple[tuple[int, str], ...]] = {
    # shells and plane elements
    "CTRIA3": ((3, "triangle"),),
    "CTRIAR": ((3, "triangle"),),
    "CTRAX3": ((3, "triangle"),),
    "CTRIA6": ((3, "triangle"), (6, "quadratic_triangle")),
    "CTRAX6": ((3, "triangle"), (6, "quadratic_triangle")),
    "CTRIAX6": ((3, "triangle"), (6, "quadratic_triangle")),
    "CQUAD4": ((4, "quad"),),
    "CQUADR": ((4, "quad"),),
    "CQUADX4": ((4, "quad"),),
    "CSHEAR": ((4, "quad"),),
    "CQUAD8": ((4, "quad"), (8, "quadratic_quad")),
    "CQUADX8": ((4, "quad"), (8, "quadratic_quad")),
    "CQUAD": ((4, "quad"), (8, "quadratic_quad"), (9, "biquadratic_quad")),
    # solids
    "CTETRA": ((4, "tetra"), (10, "quadratic_tetra")),
    "CPYRAM": ((5, "pyramid"), (13, "quadratic_pyramid")),
    "CPYRA": ((5, "pyramid"), (13, "quadratic_pyramid")),
    "CPENTA": ((6, "wedge"), (15, "quadratic_wedge")),
    "CHEXA": ((8, "hexahedron"), (20, "quadratic_hexahedron")),
    # one-dimensional elements: a deck of beams and rods is still a mesh
    "CBAR": ((2, "line"),),
    "CBEAM": ((2, "line"),),
    "CBEND": ((2, "line"),),
    "CBUSH": ((2, "line"),),
    "CBUSH1D": ((2, "line"),),
    "CGAP": ((2, "line"),),
    "CROD": ((2, "line"),),
    "CONROD": ((2, "line"),),
    "CTUBE": ((2, "line"),),
    "CVISC": ((2, "line"),),
}

# Node ordering. Nastran and VTK number the corner grid points alike, and the
# mid-side nodes of every card but one as well. CPENTA runs the bottom ring,
# then the three vertical edges, then the top ring; VTK runs the bottom ring,
# the top ring, and the verticals last. Entry ``i`` holds the Nastran position
# of the node that belongs at VTK position ``i``.
_READ_ORDER: dict[str, tuple[int, ...]] = {
    "quadratic_wedge": (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11),
}
_WRITE_ORDER: dict[str, tuple[int, ...]] = {
    name: tuple(order.index(i) for i in range(len(order)))
    for name, order in _READ_ORDER.items()
}

# CONROD names a material where every other element card names a property, so
# its grid points sit one field earlier than the rest.
_GRID_FIELD_START: dict[str, int] = {"CONROD": 2}
_DEFAULT_GRID_FIELD: int = 3

# Where a shell card carries its offset, ZOFFS: after the grid points and the
# material orientation angle. Nastran measures a shell's mid-surface from it,
# so dropping it moves the geometry the deck describes.
_ZOFFS_FIELD: dict[str, int] = {
    "CTRIA3": 7,
    "CTRIAR": 7,
    "CQUAD4": 8,
    "CQUADR": 8,
}
_ZOFFS_KEY: str = "zoffs"

# Write mapping: one canonical card per polyxios element type. Cards that
# carry their whole geometry in grid point fields are preferred, so a written
# deck needs no orientation vector to be valid.
_POLYXIOS_TO_CARD: dict[str, str] = {
    "line": "CROD",
    "triangle": "CTRIA3",
    "quadratic_triangle": "CTRIA6",
    "quad": "CQUAD4",
    "quadratic_quad": "CQUAD8",
    "biquadratic_quad": "CQUAD",
    "tetra": "CTETRA",
    "quadratic_tetra": "CTETRA",
    "pyramid": "CPYRAM",
    "quadratic_pyramid": "CPYRAM",
    "wedge": "CPENTA",
    "quadratic_wedge": "CPENTA",
    "hexahedron": "CHEXA",
    "quadratic_hexahedron": "CHEXA",
}

# Node count per polyxios type, from the shapes the cards declare.
_NODE_COUNT: dict[str, int] = {
    name: n for shapes in _CARD_SHAPES.values() for n, name in shapes
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
# over the exponent form up to this many decimals - past that the field grows
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
    free-field line is cut there whatever it holds - an unnamed marker is
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
        the line's marker field is blank - a large-field continuation may
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
                # field. A line continuing the card settles it - Nastran
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


def sniff(head: bytes) -> bool:
    """Report whether a file's opening bytes look like a Nastran deck.

    Parameters
    ----------
    head
        The file's first bytes, as handed over by the registry.

    Returns
    -------
    bool
        True when a bulk data or executive statement appears before the end
        of the sniffed window.

    Notes
    -----
    Used to resolve ``.dat``, which several unrelated formats share. ``$``
    comment lines and blanks are stepped over, since a deck often opens with
    a banner of them; a file that is nothing but comments answers False, the
    ambiguity being more useful to the caller than a wrong guess.
    """
    text = head.decode(_READ_ENCODING, errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        if _SNIFF_RE.match(stripped):
            return True
    return False


def _resolve_shape(
    fields: list[str],
    shapes: tuple[tuple[int, str], ...],
    first: int,
    card: str,
    eid: str,
) -> tuple[str, int]:
    """Return the element a card holds, from the grid points it carries.

    Parameters
    ----------
    fields
        The card's fields, continuation lines already joined.
    shapes
        The shapes this card name can hold, smallest first.
    first
        Index of the card's first grid point field.
    card, eid
        Card name and element id, named in the error when the card is short.

    Returns
    -------
    tuple of (str, int)
        The polyxios element name and its grid point count.

    Raises
    ------
    CodecError
        When the card carries fewer grid points than its smallest shape needs.

    Notes
    -----
    Counting stops at the first blank field: Nastran lets a mid-side grid
    point be left out, and that card is a linear element, not a quadratic one
    with a hole in it. The count is capped at the largest shape, so the
    material angle and offset a shell card carries past its grid points are
    never mistaken for grid points of their own.
    """
    largest = shapes[-1][0]
    present = 0
    while present < largest and _field(fields, first + present):
        present += 1
    for n_nodes, name in reversed(shapes):
        if present >= n_nodes:
            return name, n_nodes
    raise CodecError(
        f".bdf: {card} {eid} carries {present} grid point field(s);"
        f" {shapes[0][0]} are required"
    )


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a Nastran .bdf file.

    Free-field, small-field and large-field cards are supported, in any
    order: elements keep their raw grid ids until the whole deck is read,
    so they may reference grid points defined further down the file.

    Parameters
    ----------
    path
        Path to the bulk data file (``.bdf``, ``.nas`` or ``.fem``). A deck
        named ``.dat`` is recognised by its content; ``fmt=".bdf"`` forces
        the issue when the content is not recognisable.
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
    A card name does not say the element's order, so the shape is taken from
    the grid points the card actually carries: a CTETRA with ten becomes a
    ``quadratic_tetra`` and one with four a ``tetra``. Counting stops at the
    first blank field, since Nastran allows a mid-side grid point to be left
    out and that card is a linear element. CPENTA's mid-side nodes are
    permuted into VTK order; every other card already agrees. A shell's
    ``ZOFFS`` lands in ``element_attrs["zoffs"]`` when any card carries one.
    An element card outside the table is skipped with a warning, as is a
    continuation line with no card ahead of it to continue. ``INCLUDE``
    statements are not followed. A blank property id reads as 1; PSHELL,
    PSOLID and the other property cards are not parsed, so the ids carry no
    name or material.
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
    zoffs_list: list[float] = []
    # Card name and element id as the deck spells them, kept only so a
    # dangling grid reference can be pointed at the card that made it.
    elem_sources: list[tuple[str, str]] = []
    local_frames = 0
    duplicates = 0
    includes = 0
    skipped: dict[str, int] = {}

    text = read_text(path, encoding=_READ_ENCODING, errors="replace")
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

        shapes = _CARD_SHAPES.get(card)
        if shapes is None:
            if card.startswith("C") and card not in _NON_ELEMENT_C_CARDS:
                skipped[card] = skipped.get(card, 0) + 1
            continue

        eid = _field(fields, 1) or "?"
        first = _GRID_FIELD_START.get(card, _DEFAULT_GRID_FIELD)
        elem_name, n_nodes = _resolve_shape(fields, shapes, first, card, eid)
        node_fields = [_field(fields, first + j) for j in range(n_nodes)]
        nodes = [_to_int(f, ctx=f"{card} {eid} grid point") for f in node_fields]
        order = _READ_ORDER.get(elem_name)
        if order is not None:
            nodes = [nodes[k] for k in order]
        grid_ids.extend(nodes)
        offsets_list.append(offsets_list[-1] + n_nodes)
        types_list.append(ELEMENT_TYPES[elem_name])
        elem_sources.append((card, eid))
        pid = _field(fields, 2) if card not in _GRID_FIELD_START else ""
        pid_list.append(
            _to_int(pid, ctx=f"{card} {eid} property id") if pid else _DEFAULT_PID
        )
        zoffs_field = _ZOFFS_FIELD.get(card)
        zoffs_list.append(
            _to_float(_field(fields, zoffs_field), ctx=f"{card} {eid} ZOFFS")
            if zoffs_field is not None and _field(fields, zoffs_field)
            else 0.0
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

    # An offset of zero is what a shell has when the deck says nothing, so an
    # all-zero column is an attribute invented for every mesh rather than data
    # the file carried.
    if any(zoffs_list):
        element_attrs[_ZOFFS_KEY] = np.array(zoffs_list, dtype=np.float64)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_attrs=element_attrs,
        element_tags=element_tags,
    )


def _element_zoffs(poly: PolyData, n_elems: int) -> np.ndarray | None:
    """Return the shell offsets to write, or None when there are none to write.

    Parameters
    ----------
    poly
        The mesh being written.
    n_elems
        How many elements it holds.

    Returns
    -------
    numpy.ndarray or None
        One float per element, or None when the attribute is absent or does
        not describe this mesh.
    """
    stored = (poly.element_attrs or {}).get(_ZOFFS_KEY)
    if stored is None:
        return None
    values = np.asarray(stored)
    if (
        values.ndim != 1
        or values.shape[0] != n_elems
        or values.dtype.kind not in "fiub"
    ):
        warnings.warn(
            f".bdf: element_attrs['{_ZOFFS_KEY}'] is not one number per element;"
            " the shell offsets were not written.",
            stacklevel=3,
        )
        return None
    return values.astype(np.float64, copy=False)


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

    # A whole sweep asking for the value back unchanged comes first, because
    # more significant digits is not the same as closer: dropping a digit
    # can free the column that lets an exact spelling in, and a mantissa
    # stepped toward zero keeps its digit count while losing the value
    # ('10000000.' steps to '9999999.', where '1.E+07' is exact and fits).
    # The second sweep takes the closest spelling that fits, which is what a
    # value no field can hold exactly - the top of the double range - needs.
    #
    # ``repr`` is the shortest decimal that reads back as this double, so
    # rounding to fewer significant digits than it carries lands on a
    # different value whatever form the result is spelled in. The exact
    # sweep stops there rather than spelling and parsing candidates that
    # cannot qualify: that floor is what keeps a large-field write from
    # paying a hundred parses per coordinate.
    exact_digits = _shortest_exact_digits(mantissa)
    # A field carries its significant digits, a mandatory decimal point and
    # a sign if the value has one, so a precision past that cannot be spelled
    # here however the exponent is written. Starting the sweep at the widest
    # precision the field could hold skips the ones it never could.
    ceiling = min(_MAX_DIGITS, width - 1 - (number < 0))
    spellings: dict[int, tuple[list[str], list[tuple[str, str, str]]]] = {}
    for exact_only in (True, False):
        floor = exact_digits if exact_only else 1
        for digits in range(ceiling, floor - 1, -1):
            entry = spellings.get(digits)
            if entry is None:
                entry = spellings[digits] = _candidates(number, digits)
            rounded, split = entry
            # Stepping borrows out of the leading digit at most, so a
            # stepped spelling is never more than one character shorter
            # than the rounded one it came from. When even that cannot fit,
            # the whole precision is out of reach and building the stepped
            # forms would only spell candidates to throw away.
            if min(len(candidate) for candidate in rounded) > width + 1:
                continue
            for candidate in rounded:
                if len(candidate) > width:
                    continue
                read = _read_back(candidate)
                if read is None or (exact_only and read != number):
                    continue
                return candidate
            for candidate in _stepped(split):
                if len(candidate) > width:
                    continue
                read = _read_back(candidate)
                if read is None or (exact_only and read != number):
                    continue
                return candidate
    raise CodecError(f".bdf: cannot write {number!r} in a {width}-character field")


def _shortest_exact_digits(mantissa: str) -> int:
    """Count the significant digits the shortest exact spelling carries.

    Rounding a double to this many significant digits reads back as the
    same double, and rounding to fewer does not, so it is the floor of any
    search for an exact spelling.

    Parameters
    ----------
    mantissa
        The mantissa of ``repr(number)``, which is the shortest decimal
        that reads back as ``number``.

    Returns
    -------
    int
        Digits that carry value, zeros on either end excluded; at least 1,
        which is what a zero mantissa needs.
    """
    return len(mantissa.lstrip("-+").replace(".", "").strip("0")) or 1


def _candidates(
    number: float, digits: int
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Spell a value at one precision, richest mantissa and longest first.

    Both formats of the same rounded value are offered: %G picks the shorter
    of the fixed and exponent forms and drops trailing zeros, while %E is
    always an exponent form - which is what saves a small magnitude, where
    the fixed form spends the field on leading zeros ('-0.00012' against
    '-1.235-4').

    Parameters
    ----------
    number
        Value to spell.
    digits
        Significant digits to round to.

    Returns
    -------
    tuple[list of str, list of tuple[str, str, str]]
        The rounded spellings, and the mantissa/separator/exponent triples
        they came from, in the same order. A caller takes the first
        spelling that fits its field and reads back the way it wants, and
        turns to ``_stepped`` on the triples only when none of them did:
        stepping is the expensive half, and a value the rounded forms
        already cover never needs it.
    """
    # %G drops trailing zeros, so at the same precision it can carry one
    # significant digit less than %E; the richer mantissa is offered first,
    # and the shorter one only when it is the one that fits.
    split = sorted(
        (
            _split_mantissa(f"{number:.{digits}G}"),
            _split_mantissa(f"{number:.{digits - 1}E}"),
        ),
        key=lambda parts: -_significant(parts[0]),
    )
    return [form for parts in split for form in _spellings(*parts)], split


def _stepped(split: list[tuple[str, str, str]]) -> list[str]:
    """Spell each mantissa one unit in the last place closer to zero.

    Parameters
    ----------
    split
        Mantissa, separator and exponent triples, as ``_candidates`` sorts
        them.

    Returns
    -------
    list of str
        The stepped spellings, in the order of the triples they came from;
        a mantissa with nothing left to step down to contributes none.
    """
    return [
        form
        for mantissa, sep, exponent in split
        if (down := _toward_zero(mantissa)) is not None
        for form in _spellings(down, sep, exponent)
    ]


def _significant(mantissa: str) -> int:
    """Count the significant digits a mantissa carries.

    Parameters
    ----------
    mantissa
        A mantissa as a float format spells it, sign and decimal point
        included.

    Returns
    -------
    int
        Digits that carry value; a leading zero before the point does not.
    """
    digits = mantissa.lstrip("-+").replace(".", "").lstrip("0")
    return len(digits)


def _split_mantissa(spelled: str) -> tuple[str, str, str]:
    """Split a formatted float into mantissa, 'E' and exponent.

    Parameters
    ----------
    spelled
        Output of a ``%G`` or ``%E`` format.

    Returns
    -------
    tuple[str, str, str]
        The mantissa with a decimal point, ``'E'`` when there is an
        exponent, and the exponent digits with their sign.
    """
    mantissa, sep, exponent = spelled.partition("E")
    if "." not in mantissa:
        mantissa += "."
    return mantissa, sep, exponent


def _read_back(text: str) -> float | None:
    """Read a written field back the way the parser will.

    Rounding a mantissa up at the top of the double range steps off it, and
    the reader answers that with a ValueError rather than an infinity, so
    both ways of failing are one question here.

    Parameters
    ----------
    text
        Candidate field text.

    Returns
    -------
    float or None
        The value the field names, or None when it names no finite one.
    """
    try:
        return _parse_real(text)
    except ValueError:
        return None


def _spellings(mantissa: str, sep: str, exponent: str) -> list[str]:
    """Spell one rounded mantissa the ways bulk data allows, longest first.

    Bulk data lets the ``E`` go when the exponent carries its own sign, so
    ``1.234-10`` is ``1.234E-10`` in one character less - which is two more
    significant digits in an eight-column field, and the difference between
    writing a value and refusing it.

    Parameters
    ----------
    mantissa
        The rounded mantissa, decimal point included.
    sep
        ``'E'`` when the format produced an exponent, empty otherwise.
    exponent
        The exponent digits with their sign, or empty.

    Returns
    -------
    list of str
        Candidate spellings of the same value, the longest first. A caller
        keeps the first that fits its field and still reads back finite.
    """
    if not sep:
        return [mantissa, *_dot_form(mantissa)]

    # An exponent with no sign is positive; the shorthand needs the sign to
    # mark where the mantissa ends, so it is spelled back in.
    signed = exponent if exponent[:1] in "+-" else f"+{exponent}"
    # %E pads the exponent to two digits; nothing reads '+07' differently
    # from '+7', and the column the padding costs is a significant digit.
    trimmed = f"{signed[0]}{signed[1:].lstrip('0') or '0'}"
    tails = [f"{sep}{exponent}", signed]
    if trimmed != signed:
        tails.append(trimmed)
    heads = [mantissa, *_dot_form(mantissa)]
    # Longest first, and the leading-zero spelling ahead of the dot one at
    # the same length: a caller keeps the first that fits, so the plain form
    # wins whenever the column is there for it.
    return [f"{head}{tail}" for tail in tails for head in heads]


def _dot_form(mantissa: str) -> list[str]:
    """Spell a mantissa below one without its leading zero, if it has one.

    Bulk data reads ``.5`` as ``0.5``, and the column the zero costs is a
    significant digit - which in an eight-character field is the difference
    between ``0.333333`` and ``.3333333``.

    Parameters
    ----------
    mantissa
        A mantissa as a float format spells it, sign and decimal point
        included.

    Returns
    -------
    list of str
        The shortened spelling, or nothing when the mantissa has no leading
        zero to drop or nothing left after the point - ``.`` names no value.
    """
    sign = "-" if mantissa.startswith("-") else ""
    body = mantissa[len(sign) :]
    if not body.startswith("0.") or len(body) == 2:
        return []
    return [f"{sign}{body[1:]}"]


def _toward_zero(mantissa: str) -> str | None:
    """Step a mantissa one unit in the last place closer to zero.

    Rounding to nearest is what overflows at the top of the double range,
    and the step down is the only spelling of the same length that stays on
    it. The last place is a decimal digit, so this borrows across the point
    the way subtraction does: ``1.80`` becomes ``1.79``.

    Parameters
    ----------
    mantissa
        A mantissa as a float format spells it, sign and decimal point
        included.

    Returns
    -------
    str or None
        The stepped mantissa, keeping its sign, its point and its digit
        count; None when every digit is already zero and there is nothing
        left to step down to.
    """
    sign = "-" if mantissa.startswith("-") else ""
    body = mantissa.lstrip("-+")
    head, point, tail = body.partition(".")
    digits = f"{head}{tail}"
    if not digits.isdigit() or int(digits) == 0:
        return None

    stepped = f"{int(digits) - 1:0{len(digits)}d}"
    # Borrowing out of the leading digit pads it with a zero ('10.0' steps to
    # '09.9'); a field is too narrow to spend a column on a digit that says
    # nothing, and a solver reads the same value either way.
    whole = stepped[: len(head)].lstrip("0") or "0"
    return f"{sign}{whole}{point}{stepped[len(head) :]}"


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
    it recognisable. When it does not - an exponent form below one, say -
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
    or at column 80, the width of a bulk data record - a long real reaches
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
        # longer one reads back exactly here but rounded in a solver - or,
        # when even the fixed-point spelling stays out of reach, wrong.
        lost = sum(len(text) > _FREE_SIGNIFICANT for text in texts)
        misread = sum(not _survives_truncation(text) for text in texts)
        return _card_lines(["GRID", str(index + 1), "", *texts]), lost, misread

    texts = [_fmt_real(value, width=_LARGE_WIDTH) for value in xyz]
    # The text may be spelled in the implicit-exponent shorthand, which
    # float() does not read; ask the parser that wrote the rules instead.
    lost = sum(_parse_real(text) != float(value) for text, value in zip(texts, xyz))

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
    path: Source,
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
    rather than to a stray mantissa; the few values no spelling saves -
    very large or very small exponents - get a warning of their own.
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
    # writes a card short of one - silently in free field, and as a bare
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
    zoffs = _element_zoffs(poly, n_elems)
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
        expected = _NODE_COUNT[name]
        if end - start != expected:
            raise CodecError(
                f".bdf: element {ei} of type '{name}' has {end - start} grid"
                f" points, expected {expected}"
            )
        indices = connectivity[start:end].tolist()
        order = _WRITE_ORDER.get(name)
        if order is not None:
            indices = [indices[k] for k in order]
        nodes = [str(node + 1) for node in indices]
        card_fields = [card, str(ei + 1), str(int(pids[ei])), *nodes]
        offset_at = _ZOFFS_FIELD.get(card)
        if offset_at is not None and zoffs is not None and zoffs[ei]:
            # ZOFFS sits past the material orientation angle, which stays
            # blank: a deck that omits it takes the property's own value.
            card_fields.extend([""] * (offset_at - len(card_fields) + 1))
            card_fields[offset_at] = _fmt_real(float(zoffs[ei]))
        lines.extend(_card_lines(card_fields))

    lines.append("ENDDATA")
    lines.append("")
    write_text(path, "\n".join(lines), encoding=_WRITE_ENCODING)
