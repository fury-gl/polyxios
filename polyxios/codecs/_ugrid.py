"""UGRID mesh codec (AFLR ASCII ``.ugrid``) - read + write.

A UGRID file opens with one header line of seven counts and then lists every
section back to back, whitespace-separated::

    <n_nodes> <n_tri> <n_quad> <n_tet> <n_pyramid> <n_prism> <n_hex>
    <node coordinates>
    <triangle connectivity>
    <quad connectivity>
    <triangle surface IDs>
    <quad surface IDs>
    <tetrahedron connectivity>
    <pyramid connectivity>
    <prism connectivity>
    <hexahedron connectivity>

Nothing in the body says which section it belongs to, so the header counts are
the only thing separating them and every one of them has to be walked in this
order. The surface IDs in particular sit *between* the boundary faces and the
volume cells, not at the end of the file: reading them anywhere else shifts
every volume element that follows.

Node references are 1-based. Prism and hexahedron node order matches polyxios's
(and VTK's), but the pyramid's does not, so that one section is permuted in
both directions.

Only the ASCII flavour is handled here. The same format has binary variants
spelled by an infix suffix - ``mesh.b8.ugrid``, ``mesh.lb8.ugrid``,
``mesh.r8.ugrid`` - which this codec refuses rather than parsing as text: by
name where the name carries the infix, and by the NUL in the opening record
where it does not.
"""

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
from polyxios._io import Source, read_bytes, source_name, strip_gzip, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".ugrid"

# Matches the other ASCII writers (.obj, .avs, .su2, .tetgen); pass float_fmt
# to widen it.
_DEFAULT_FLOAT_FMT: str = ".10g"

# The file's sections, in the order the format lists them: (polyxios type name,
# nodes per element). The first two are the boundary faces that carry a surface
# ID; the rest are the volume cells that do not.
_SECTIONS: tuple[tuple[str, int], ...] = (
    ("triangle", 3),
    ("quad", 4),
    ("tetra", 4),
    ("pyramid", 5),
    ("wedge", 6),
    ("hexahedron", 8),
)
_N_SURFACE_SECTIONS: int = 2

_TYPE_CODES: dict[str, int] = {name: int(ELEMENT_TYPES[name]) for name, _ in _SECTIONS}

# UGRID spells a pyramid with its apex third and the base quad around it, which
# is not the base-then-apex order polyxios and VTK use. The two permutations are
# inverses of one another; applying either one alone would bend the element in
# silence, since a pyramid with its nodes shuffled is still five valid nodes.
_PYRAMID_TO_POLYXIOS: list[int] = [1, 0, 3, 4, 2]
_PYRAMID_FROM_POLYXIOS: list[int] = [1, 0, 4, 2, 3]

# A surface ID is an integer, and ``boundary_<n>`` is how reading spells the
# faces carrying it; writing reads the number back out of the name so a round
# trip keeps the ID it started with.
_BOUNDARY_RE = re.compile(r"^boundary_(-?\d+)$")

_DIGITS_RE = re.compile(r"(\d+)")

# The body is parsed as float64 in one block, which is exact below 2**53 and
# rounds at or past it. A node reference or a surface ID that rounds is a
# different number than the file holds, so both are refused rather than
# trusted. The bound is exclusive: 2**53 itself survives the trip, but 2**53+1
# lands on it, so admitting either one would let the two come back as the same
# number.
_MAX_EXACT_INT: int = 2**53

# The infixes the binary flavours of the format are spelled with. They share the
# .ugrid extension, so the name is the only thing that tells them apart before
# the first token fails to parse.
_BINARY_INFIXES: frozenset[str] = frozenset(
    {"b8", "b8l", "b4", "lb8", "lb8l", "lb4", "r8", "r4", "lr8", "lr4"}
)

# How much of a file's head is looked at to tell binary from text. The seven
# counts a UGRID file opens with are well inside it under either flavour.
_SNIFF_BYTES: int = 4096


def _natural_key(name: str) -> tuple[tuple[int, Any], ...]:
    """Return a sort key ordering embedded numbers by value, not by spelling.

    Tag names are what fix the order surface IDs are handed out in, and the
    names reading gives back are numbered - ``boundary_1``, ``boundary_2``, ….
    Plain lexicographic order puts ``boundary_10`` between ``boundary_1`` and
    ``boundary_2``, which would hand an unnumbered tag an ID out of turn.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in _DIGITS_RE.split(name)
    )


def _reject_binary(path: Source) -> None:
    """Refuse a binary UGRID variant by name before parsing it as text.

    The binary flavours share the ``.ugrid`` extension and are told apart only
    by an infix suffix. Left to the tokenizer they surface as a non-numeric
    value somewhere in the middle of the file, which names the wrong fault.

    Only a real infix counts: ``b4.ugrid`` is a mesh named ``b4``, not a
    stem-less binary file, and refusing it would lock a caller out of a file
    that is perfectly readable. A binary file spelled that way is caught by
    :func:`_reject_binary_bytes` instead, which does not have to guess.

    A ``.gz`` on the end names the compression rather than the format, so it
    comes off before the infix is looked for: ``mesh.lb8.ugrid.gz`` is the
    same binary variant as ``mesh.lb8.ugrid``.
    """
    parts = strip_gzip(source_name(path).lower()).split(".")
    if len(parts) > 2 and parts[-2] in _BINARY_INFIXES:
        raise CodecError(
            f".ugrid: '{source_name(path)}' names the binary '{parts[-2]}' UGRID variant;"
            " only the ASCII flavour is supported."
        )


def _reject_binary_bytes(path: Source, head: bytes) -> None:
    """Refuse a file whose opening bytes are not text.

    The name check only catches the ``mesh.lb8.ugrid`` spelling. A binary UGRID
    file opens with seven raw integers whose high bytes are NUL, and no ASCII
    one holds a NUL anywhere, so the first record settles by itself what the
    name could only guess at.
    """
    if b"\x00" in head:
        raise CodecError(
            f".ugrid: '{source_name(path)}' opens with binary data; only the ASCII"
            " UGRID flavour is supported, not the b8/lb8/r8 variants."
        )


def _tokenize(path: Source) -> list[str]:
    """Return the file's whitespace-separated tokens.

    The format defines no comment character, so nothing is stripped: a ``#``
    inside a UGRID file is a corrupt token, and dropping the rest of its line
    would shift every value after it instead of reporting it.
    """
    try:
        raw = read_bytes(path)
    except OSError as exc:
        raise CodecError(f".ugrid: cannot read '{source_name(path)}': {exc}") from exc
    _reject_binary_bytes(path, raw[:_SNIFF_BYTES])
    return raw.decode("utf-8-sig", errors="replace").split()


def _checked_count(token: str, cap: int, what: str) -> int:
    """Parse a declared count, rejecting negatives and absurd values.

    A corrupt header is the cheapest way to make a reader allocate a mesh the
    machine cannot hold, so the declared size is checked before it is trusted.
    """
    try:
        count = int(token)
    except ValueError as exc:
        raise CodecError(f".ugrid: non-integer {what} count {token!r}.") from exc
    if count < 0:
        raise CodecError(f".ugrid: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".ugrid: {what} count {count} exceeds the safety cap {cap}.")
    return count


def _body(tokens: list[str], needed: int) -> np.ndarray:
    """Return the ``needed`` values after the header as one float64 array.

    Read in one block rather than section by section: every section is fixed
    width, so a file short of a single value has already lost the boundary
    between two of them and reading on would pull one section's nodes into the
    one before it.
    """
    # Counted rather than sliced: ``tokens[7:]`` would copy the whole body only
    # to copy the wanted part out of it again.
    available = len(tokens) - 7
    if available < needed:
        raise CodecError(
            f".ugrid: file truncated - the header declares {needed} body"
            f" value(s), found {available}."
        )
    if available > needed:
        # Trailing values are ones this reader will never see; going quiet would
        # report a smaller file than the one on disk.
        warnings.warn(
            f".ugrid: {available - needed} value(s) follow the sections the"
            " header declares; ignored.",
            stacklevel=3,
        )
    try:
        return np.array(tokens[7 : 7 + needed], dtype=np.float64)
    except ValueError as exc:
        raise CodecError(".ugrid: non-numeric value in the file body.") from exc


def _as_ints(values: np.ndarray, what: str) -> np.ndarray:
    """Return a body slice as int64, refusing values float64 cannot hold.

    The body is parsed as float64, which is exact below 2**53 and silently
    rounds at or past it. A node reference that rounds resolves to some other
    vertex, so it is refused rather than trusted. The check is on 2**53 itself
    rather than past it: 2**53+1 has already become 2**53 by the time it gets
    here, so a bound that let 2**53 through would let its neighbour through
    wearing the same value.
    """
    if values.size:
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) >= _MAX_EXACT_INT:
            raise CodecError(f".ugrid: {what} out of range.")
        if not np.array_equal(values, np.rint(values)):
            raise CodecError(f".ugrid: non-integer {what}.")
    return values.astype(np.int64)


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an AFLR UGRID ASCII ``.ugrid`` file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.ugrid`` file.
    lazy
        Ignored; a true value is warned about (ASCII format, always loads
        eagerly).

    Returns
    -------
    PolyData
        Elements in the order the format lists them: triangles, quads, then
        tetrahedra, pyramids, prisms and hexahedra.

    Raises
    ------
    CodecError
        On a file naming a binary UGRID variant or opening with binary data, a
        header short of its seven counts, a negative or absurd count, a
        truncated or non-numeric body, a non-integer node reference or surface
        ID, or a node reference outside the node array the header declares.

    Notes
    -----
    Node references are 1-based in the file and 0-based in the PolyData. The
    pyramid section is permuted into polyxios's node order; prisms (``wedge``)
    and hexahedra already match and are read as they lie.

    Each distinct non-zero surface ID becomes an ``element_tags`` entry named
    ``boundary_<id>`` over the boundary faces carrying it; ID 0 means unmarked
    and is dropped. Values past the sections the header declares are ignored
    with a warning.
    """
    if lazy:
        warnings.warn(
            ".ugrid: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    _reject_binary(path)
    tokens = _tokenize(path)
    if len(tokens) < 7:
        raise CodecError(
            f".ugrid: header declares {len(tokens)} count(s), expected 7"
            " (nodes, triangles, quads, tetrahedra, pyramids, prisms, hexahedra)."
        )

    n_nodes = _checked_count(tokens[0], MAX_SAFE_VERTICES, "node")
    counts = [
        _checked_count(tokens[i + 1], MAX_SAFE_ELEMENTS, f"{name} element")
        for i, (name, _width) in enumerate(_SECTIONS)
    ]

    # Each section is capped on its own, so only their sum can walk past the
    # caps - which is what keeps the offsets array and its dtype honest.
    total_conn = sum(count * width for count, (_name, width) in zip(counts, _SECTIONS))
    if total_conn > MAX_SAFE_CONN:
        raise CodecError(
            f".ugrid: connectivity {total_conn} exceeds the safety cap {MAX_SAFE_CONN}."
        )
    n_elems = sum(counts)
    if n_elems > MAX_SAFE_ELEMENTS:
        raise CodecError(
            f".ugrid: element count {n_elems} exceeds the safety cap"
            f" {MAX_SAFE_ELEMENTS}."
        )

    n_surf = sum(counts[:_N_SURFACE_SECTIONS])
    values = _body(tokens, 3 * n_nodes + total_conn + n_surf)

    pos = 3 * n_nodes
    vertices = values[:pos].reshape(n_nodes, 3).copy()

    # The surface IDs sit between the boundary faces and the volume cells, so
    # the sections are walked in file order with the IDs read in their place.
    blocks: list[np.ndarray] = []
    surface_ids: list[np.ndarray] = []
    for i, ((name, width), count) in enumerate(zip(_SECTIONS, counts)):
        if i == _N_SURFACE_SECTIONS:
            for surf_count in counts[:_N_SURFACE_SECTIONS]:
                surface_ids.append(
                    _as_ints(values[pos : pos + surf_count], "surface ID")
                )
                pos += surf_count
        size = count * width
        refs = _as_ints(values[pos : pos + size], "node reference").reshape(
            count, width
        )
        pos += size
        if name == "pyramid" and count:
            refs = refs[:, _PYRAMID_TO_POLYXIOS]
        blocks.append(refs)

    conn = (
        np.concatenate([refs.reshape(-1) for refs in blocks])
        if total_conn
        else np.zeros(0, dtype=np.int64)
    )
    if conn.size:
        lo, hi = int(conn.min()), int(conn.max())
        if lo < 1 or hi > n_nodes:
            # UGRID counts nodes from 1, so 0 is as much out of range as a
            # reference past the end; both would index the wrong vertex.
            where = f"1..{n_nodes}" if n_nodes else "an empty node array"
            raise CodecError(
                f".ugrid: element references node {lo}..{hi}, outside {where}."
            )
    conn -= 1

    # The offsets are a cumulative sum MAX_SAFE_CONN lets past 2**31, where
    # int32 would overflow; widen the pair together so the CSR dtypes match.
    idx_dtype = np.int64 if total_conn > np.iinfo(np.int32).max else np.int32
    # Filled a section at a time rather than by summing a per-element width
    # array: every element in one section is the same width, so its offsets are
    # an arange. The width array and its cumulative sum would each be as long as
    # the mesh and both alive at once, which is what costs on a large file.
    offsets = np.zeros(n_elems + 1, dtype=idx_dtype)
    at = 0
    base = 0
    for count, (_name, width) in zip(counts, _SECTIONS):
        if count:
            # Stepped in int64 whatever the offsets hold: the stop is one past
            # the last offset, which for an int32 mesh sits at the very top of
            # the range. The values themselves are inside it, so the assignment
            # narrows them without loss.
            offsets[at + 1 : at + 1 + count] = np.arange(
                base + width, base + width * count + 1, width, dtype=np.int64
            )
            at += count
            base += width * count
    element_types = np.concatenate(
        [
            np.full(count, _TYPE_CODES[name], dtype=np.uint8)
            for count, (name, _width) in zip(counts, _SECTIONS)
        ]
    )

    # One tag per distinct ID, not one tag over every marked face: the IDs are
    # what tell two boundary patches apart, and a single tag would fuse them
    # into an unrecoverable whole. Surface elements lead the element array, so
    # a position in the concatenated IDs is already the element's index.
    element_tags: dict[str, np.ndarray] = {}
    ids = np.concatenate(surface_ids) if surface_ids else np.zeros(0, dtype=np.int64)
    # Grouped by one stable sort rather than a pass over every ID per distinct
    # value: a mesh carrying many boundary patches would otherwise walk the
    # whole array once for each of them. The sort is stable and the marked
    # positions go in ascending, so each group comes out in element order.
    marked = np.flatnonzero(ids)
    if marked.size:
        order = marked[np.argsort(ids[marked], kind="stable")]
        ranked = ids[order]
        starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
        stops = np.append(starts[1:], ranked.size)
        for start, stop in zip(starts, stops):
            element_tags[f"boundary_{int(ranked[start])}"] = order[start:stop].astype(
                np.int32
            )

    return PolyData(
        vertices=vertices,
        connectivity=conn.astype(idx_dtype),
        offsets=offsets,
        element_types=element_types,
        element_tags=element_tags,
    )


def _group_elements(poly: PolyData) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sort the mesh's elements into the sections a UGRID file holds.

    Returns
    -------
    dict
        Section name to the element indices it takes, in mesh order.
    numpy.ndarray
        The mesh's offsets as int64, handed back so the caller gathering the
        connectivity does not widen the same array a second time.

    Raises
    ------
    CodecError
        If the offsets array does not close over the elements, or an element's
        node count does not match its type - the format records no per-element
        width, so a wrong one silently steals the next element's nodes.
    """
    types = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets).astype(np.int64)
    if offsets.size != types.shape[0] + 1:
        # An offsets array one short of the elements it indexes would be read
        # off its end while gathering the last one; saying so beats the
        # IndexError that gathering would raise from inside a helper.
        raise CodecError(
            f".ugrid: offsets carry {offsets.size} entry/entries for"
            f" {types.shape[0]} element(s), expected {types.shape[0] + 1}."
        )
    widths = np.diff(offsets)
    groups: dict[str, np.ndarray] = {}
    known = np.zeros(types.shape[0], dtype=bool)

    for name, width in _SECTIONS:
        members = np.flatnonzero(types == _TYPE_CODES[name])
        known[members] = True
        bad = np.flatnonzero(widths[members] != width)
        if bad.size:
            raise CodecError(
                f".ugrid: element {int(members[bad[0]])} is a {name} carrying"
                f" {int(widths[members[bad[0]]])} node(s), expected {width}."
            )
        groups[name] = members

    if not np.all(known):
        skipped = {
            ELEMENT_TYPES_INV.get(int(code), f"code {int(code)}")
            for code in np.unique(types[~known])
        }
        warnings.warn(
            f".ugrid: unsupported element type(s) {sorted(skipped)}; skipped."
            " UGRID knows triangle, quad, tetra, pyramid, wedge and hexahedron.",
            stacklevel=3,
        )
    return groups, offsets


def _surface_ids(poly: PolyData, surface: np.ndarray) -> np.ndarray:
    """Fold ``element_tags`` into one surface ID per boundary face.

    A UGRID face carries a single ID, so a face in two tags keeps the first
    that claims it in name order. ``boundary_<n>`` recovers the number reading
    gave it; any other name is numbered from 1 upward so the tag still travels.
    A face no tag claims takes the first ID no tag is using - 1 on a mesh with
    no tags at all, which is what a UGRID file carrying no boundary information
    holds. Writing 0 there would leave a file solvers reject, and reusing a
    number a tag already holds would fuse two boundary patches into one.

    Only a tag that writes at least one face takes an ID at all: one whose every
    member was dropped, or was already claimed, puts nothing in the file, and
    reserving a number for it would push every tag after it up by one.
    """
    n_elems = len(poly.element_types)
    # Position of each element inside the boundary-face run, which is where its
    # ID goes; -1 for everything that is not a boundary face.
    slot = np.full(n_elems, -1, dtype=np.int64)
    slot[surface] = np.arange(surface.size)

    ids = np.zeros(surface.size, dtype=np.int64)
    claimed = np.zeros(surface.size, dtype=bool)
    shared = 0
    out_of_range = 0
    not_surface = 0
    collided = 0
    used: set[int] = set()
    named: list[tuple[str, int]] = []

    for name in sorted(poly.element_tags or {}, key=lambda n: _natural_key(str(n))):
        match = _BOUNDARY_RE.match(str(name))
        # ID 0 is the format's word for unmarked, so a tag literally named
        # ``boundary_0`` cannot keep its number without vanishing; it takes a
        # fresh one, the same way an unnumbered name does. A number past what
        # reading resolves exactly goes the same way: keeping it would write an
        # ID that comes back as a different one, or as the error refusing it.
        value = int(match.group(1)) if match is not None else 0
        if abs(value) >= _MAX_EXACT_INT:
            value = 0
        named.append((name, value))

    # Walked twice: whether a tag writes any face at all decides whether it
    # takes an ID, and every ID already spoken for has to be known in full
    # before the first unnumbered tag is handed one.
    writing: list[tuple[int, np.ndarray]] = []
    for name, value in named:
        idx = np.unique(np.asarray(poly.element_tags[name]).ravel().astype(np.int64))
        inside = idx[(idx >= 0) & (idx < n_elems)]
        out_of_range += idx.size - inside.size
        slots = slot[inside]
        on_surface = slots[slots >= 0]
        not_surface += slots.size - on_surface.size
        fresh = on_surface[~claimed[on_surface]]
        shared += on_surface.size - fresh.size
        if fresh.size == 0:
            continue
        if value in used:
            # Two names spelling the same ID - ``boundary_3`` and
            # ``boundary_03`` - would land on one another and come back as a
            # single fused patch. The later one is renumbered instead, since a
            # tag under a different number still tells the two apart.
            collided += 1
            value = 0
        claimed[fresh] = True
        writing.append((value, fresh))
        if value:
            used.add(value)

    nxt = 1
    for value, fresh in writing:
        if value == 0:
            while nxt in used:
                nxt += 1
            value = nxt
            used.add(value)
        ids[fresh] = value

    if collided:
        warnings.warn(
            f".ugrid: {collided} element tag(s) name a surface ID another tag"
            " already holds; a shared ID would come back as one fused patch, so"
            " those tags were renumbered.",
            stacklevel=3,
        )
    if shared:
        warnings.warn(
            f".ugrid: {shared} boundary face(s) belong to more than one tag; a"
            " UGRID face carries one surface ID, so each kept the first that"
            " claims it.",
            stacklevel=3,
        )
    if not_surface:
        warnings.warn(
            f".ugrid: element tag(s) name {not_surface} element(s) that are not"
            " boundary faces; UGRID marks triangles and quads only, so those"
            " members were dropped.",
            stacklevel=3,
        )
    if out_of_range:
        warnings.warn(
            f".ugrid: element tag(s) name {out_of_range} index(es) outside the"
            f" mesh's {n_elems} element(s); those members were dropped.",
            stacklevel=3,
        )

    # A face no tag reached still has to carry an ID. It takes a free one out of
    # the same allocator the tags drew from: handing it 1 outright would put it
    # under whichever tag already holds 1, and the two would come back fused.
    if not np.all(claimed):
        while nxt in used:
            nxt += 1
        ids[~claimed] = nxt
    return ids


def _float_fmt(opts: dict[str, Any]) -> str:
    """Return the coordinate format specifier, refusing an unusable one."""
    fmt = opts.pop("float_fmt", _DEFAULT_FLOAT_FMT)
    if opts:
        warnings.warn(
            f".ugrid write: unrecognized options {set(opts)}; ignored.", stacklevel=3
        )
    try:
        probe = format(-1234.5, fmt)
    except (TypeError, ValueError) as exc:
        raise CodecError(f".ugrid: float_fmt {fmt!r} is not usable.") from exc
    try:
        # A grouping or percent format is a legal specifier that writes tokens
        # like ``-1,234.50``; the file would land on disk and only fail on the
        # way back in, so it is refused here rather than by the reader.
        float(probe)
    except ValueError as exc:
        raise CodecError(
            f".ugrid: float_fmt {fmt!r} writes {probe!r}, which is not a number"
            " the reader can parse back."
        ) from exc
    return fmt


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to an AFLR UGRID ASCII ``.ugrid`` file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.ugrid`` path.
    **opts
        ``float_fmt`` overrides the ASCII coordinate format specifier. Any
        other option is warned about and ignored.

    Raises
    ------
    CodecError
        If ``path`` names a binary UGRID variant, ``float_fmt`` is not a usable
        format specifier or writes a token that is not a number, the vertices
        are not an ``(n, 3)`` array, the offsets do not close over the elements
        or over the connectivity, an element's node count does not match its
        type, or an element references a vertex that does not exist.

    Notes
    -----
    Elements are written section by section in the order the format fixes -
    triangles, quads, tetrahedra, pyramids, prisms, hexahedra - so an element's
    index changes unless the mesh already lay in that order. Node references
    are written 1-based, and the pyramid section is permuted out of polyxios's
    node order into UGRID's.

    Elements of a type the format cannot spell are skipped with a warning.
    ``element_tags`` fold into one surface ID per boundary face: a
    ``boundary_<n>`` name keeps its number, ``boundary_0`` and any other name
    are numbered from 1 instead, since ID 0 is the format's word for unmarked,
    and a face in two tags keeps the first that claims it in name order. Two
    names spelling one ID would come back fused, so the later is renumbered
    with a warning. Tag members that are not boundary faces, or that fall
    outside the element array, are dropped with a warning, and a tag left with
    no face to write takes no ID at all. A boundary face no tag claims is
    written with the first ID no tag is using - 1 on a mesh carrying no tags -
    and reads back tagged with it.

    The format carries no fields, so ``vertex_attrs``, ``element_attrs``,
    ``vertex_tags`` and ``global_attrs`` are not written.
    """
    _reject_binary(path)

    fmt = _float_fmt(dict(opts))

    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        # The coordinates are written a row at a time, so an array of any other
        # width would put rows of that width on disk under a header counting
        # threes - a file that only fails on the way back in.
        raise CodecError(
            f".ugrid: vertices have shape {vertices.shape}, expected (n, 3)."
        )

    groups, offsets = _group_elements(poly)
    counts = [groups[name].size for name, _width in _SECTIONS]
    n_verts = vertices.shape[0]

    connectivity = np.asarray(poly.connectivity)
    n_conn = connectivity.shape[0] if connectivity.ndim else 0
    rows: list[np.ndarray] = []
    for name, width in _SECTIONS:
        members = groups[name]
        if members.size == 0:
            rows.append(np.zeros((0, width), dtype=np.int64))
            continue
        starts = offsets[members]
        # The gather below reads ``width`` values from each start. A start past
        # the end of the connectivity would raise a bare IndexError from inside
        # the gather, and a negative one is worse: numpy counts it from the end
        # and writes some other element's nodes without a word.
        astray = np.flatnonzero((starts < 0) | (starts + width > n_conn))
        if astray.size:
            first = int(astray[0])
            raise CodecError(
                f".ugrid: element {int(members[first])} starts at offset"
                f" {int(starts[first])}, so its {width} node(s) fall outside the"
                f" connectivity's {n_conn} value(s)."
            )
        refs = connectivity[starts[:, None] + np.arange(width)].astype(np.int64)
        lo, hi = int(refs.min()), int(refs.max())
        if lo < 0 or hi >= n_verts:
            where = f"0..{n_verts - 1}" if n_verts else "an empty vertex array"
            raise CodecError(
                f".ugrid: element references vertex {lo}..{hi}, outside {where}."
            )
        if name == "pyramid":
            refs = refs[:, _PYRAMID_FROM_POLYXIOS]
        rows.append(refs + 1)

    # Folded even when the mesh has no boundary face at all: that is the case
    # where every tag it carries is about to be dropped, and the warning saying
    # so is the only word the caller gets.
    surface = np.concatenate(
        [groups[name] for name, _w in _SECTIONS[:_N_SURFACE_SECTIONS]]
    )
    ids = _surface_ids(poly, surface)

    lines: list[str] = [" ".join(str(c) for c in (n_verts, *counts))]
    # One crossing into Python per array rather than one per number: indexing a
    # numpy array a scalar at a time is what costs on a mesh of any size.
    lines.extend(" ".join(format(c, fmt) for c in xyz) for xyz in vertices.tolist())
    for i, block in enumerate(rows):
        if i == _N_SURFACE_SECTIONS:
            lines.extend(str(v) for v in ids.tolist())
        lines.extend(" ".join(str(v) for v in row) for row in block.tolist())

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
