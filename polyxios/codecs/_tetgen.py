"""TetGen mesh codec (ASCII ``.node`` + ``.ele`` pair) — read + write.

A TetGen mesh is split across two files that share a stem: ``.node`` holds the
coordinates and ``.ele`` holds the tetrahedra that index them. Neither is
readable alone as a mesh, so both are found from whichever one the caller
names, and a write emits the pair.

Both files open with a header line and then list one fixed-width record per
row, each row opening with the item's own number. That number is data, not a
position: TetGen's ``-z`` switch renumbers from zero, a ``.node`` written by
hand may skip a number, and the two files number independently. So the row
numbers of the ``.node`` file — not the ``.ele`` file's — are what element node
references are resolved against.

The two headers are::

    <n_points>  <dimension>  <n_attributes>  <boundary marker: 0 or 1>
    <n_tets>    <nodes per tet>  <n_attributes>

Every column the header declares has to be counted even when its content is
dropped, since a node attribute read as a coordinate shifts the whole file.
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

EXTENSION: str = ".ele"
# The pair is one format, and a caller holding the .node half should not have to
# know that the .ele half is the one this codec is filed under.
EXTENSIONS: tuple[str, ...] = (".ele", ".node")

# Matches the other ASCII writers (.obj, .avs, .su2); pass float_fmt to widen it.
_DEFAULT_FLOAT_FMT: str = ".10g"

_TETRA_CODE: int = int(ELEMENT_TYPES["tetra"])

# An attribute count is a column count, not a row count, and the row-width check
# only bounds it when the file declares rows: a header of ``0 3 <huge> 0`` names
# no rows at all, so nothing downstream stops it from naming a billion columns.
# Real TetGen files carry a handful, so a column cap of its own is what keeps a
# one-line file from allocating an array per declared column.
_MAX_ATTR_COLUMNS: int = 4096

# A boundary marker is an integer, and ``boundary_<n>`` is how reading spells
# the vertices carrying it; writing reads the number back out of the name so a
# round trip keeps the marker it started with.
_BOUNDARY_RE = re.compile(r"^boundary_(-?\d+)$")

_DIGITS_RE = re.compile(r"(\d+)")

# A body is parsed as float64 in one block, which is exact to 2**53 and rounds
# past it. That bounds reading, and writing with it too is what keeps the pair
# this codec emits one it can read back: a marker wider than this would come
# home as some other number, or not at all.
_MAX_EXACT_INT: int = 2**53


def _natural_key(name: str) -> tuple[tuple[int, Any], ...]:
    """Return a sort key ordering embedded numbers by value, not by spelling.

    Names are what fix the column order of a written file, and the names
    reading hands back are numbered — ``attr_0``, ``attr_1``, …. Plain
    lexicographic order puts ``attr_10`` between ``attr_1`` and ``attr_2``, so
    a mesh with ten or more attributes would come back with its columns
    shuffled. Splitting on digit runs keeps every other name where sorting
    already put it.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in _DIGITS_RE.split(name)
    )


def _sibling(path: Path, suffix: str) -> Path:
    """Return ``path`` renamed to ``suffix``, stripping a ``.ele``/``.node`` one.

    ``with_suffix`` alone would turn ``mesh.node`` into itself when asked for
    ``.node``, so a write handed the wrong half of the pair would send both
    files to one name and keep only the second.
    """
    if path.suffix.lower() in (".ele", ".node"):
        return path.with_suffix(suffix)
    return path.with_name(path.name + suffix)


def _pair_paths(path: Path) -> tuple[Path, Path]:
    """Return the ``(.node, .ele)`` paths a write to ``path`` should produce.

    The half the caller named is kept exactly as spelled and the sibling copies
    its case: a path of ``MESH.ELE`` writes ``MESH.NODE`` beside itself. Folding
    the name to lower case instead would leave the file the caller asked for
    uncreated on a case-sensitive filesystem.
    """
    suffix = path.suffix
    if suffix.lower() == ".node":
        return path, path.with_suffix(".ELE" if suffix.isupper() else ".ele")
    if suffix.lower() == ".ele":
        return path.with_suffix(".NODE" if suffix.isupper() else ".node"), path
    return path.with_name(path.name + ".node"), path.with_name(path.name + ".ele")


def _exists(path: Path) -> bool:
    """Return whether ``path`` is on disk, as a CodecError when that is unknown.

    ``Path.exists`` answers False for a name that is absent but propagates the
    OSError for one it is not allowed to look at, so a directory the process
    cannot search would otherwise raise a bare ``PermissionError`` out of a
    codec whose contract is ``CodecError``. Answering False instead would be
    worse than either: a file that is on disk would be reported missing, and
    the caller would go looking for a path that is already there.
    """
    try:
        return path.exists()
    except OSError as exc:
        raise CodecError(f".tetgen: cannot check for '{path}': {exc}") from exc


def _find_half(path: Path, suffix: str) -> Path:
    """Return the sibling to read, preferring one that is actually there.

    A pair written ``MESH.ELE``/``MESH.NODE`` names its halves in upper case,
    and on a case-sensitive filesystem the lower-cased sibling is a file that
    does not exist. Both conventional spellings are tried, then the directory
    is listed for one that differs only in case, so a ``Mesh.Node`` written by
    a tool that title-cases its output is still found. The lower-cased name is
    what comes back when nothing matches, so the error names the usual one.
    """
    lower = _sibling(path, suffix)
    if _exists(lower):
        return lower
    upper = _sibling(path, suffix.upper())
    if _exists(upper):
        return upper
    # Listing is the only way to reach a half spelled ``Mesh.Node``, and it is
    # paid for only once both conventional spellings have already missed.
    wanted = lower.name.lower()
    try:
        for entry in lower.parent.iterdir():
            if entry.name.lower() == wanted:
                return entry
    except OSError:
        # A directory that stats but does not list leaves the case-folded name
        # unknowable. Falling through reports the half as missing, which is the
        # honest answer when the only spelling that could match cannot be seen.
        pass
    return lower


def _tokenize(path: Path) -> list[str]:
    """Return the file's whitespace-separated tokens, ``#`` comments dropped."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise CodecError(f".tetgen: cannot read '{path.name}': {exc}") from exc
    tokens: list[str] = []
    for line in text.splitlines():
        tokens.extend(line.split("#", 1)[0].split())
    return tokens


def _checked_count(token: str, cap: int, what: str, where: str) -> int:
    """Parse a declared count, rejecting negatives and absurd values.

    A corrupt header is the cheapest way to make a reader allocate a mesh the
    machine cannot hold, so the declared size is checked before it is trusted.
    """
    try:
        count = int(token)
    except ValueError as exc:
        raise CodecError(
            f".tetgen: non-integer {what} count {token!r} in {where}."
        ) from exc
    if count < 0:
        raise CodecError(f".tetgen: negative {what} count {count} in {where}.")
    if count > cap:
        raise CodecError(
            f".tetgen: {what} count {count} in {where} exceeds the safety cap {cap}."
        )
    return count


def _small_int(token: str, what: str, where: str) -> int:
    """Parse a header field whose legal values are a short fixed set.

    Kept apart from ``_checked_count`` so a dimension of 4 is reported as the
    dimension it is rather than as a count over some cap it never was.
    """
    try:
        return int(token)
    except ValueError as exc:
        raise CodecError(f".tetgen: non-integer {what} {token!r} in {where}.") from exc


def _as_ints(column: np.ndarray, what: str, where: str) -> np.ndarray:
    """Return a body column as int64, refusing values float64 cannot hold.

    The body is parsed as float64 in one block, which is exact up to 2**53 and
    silently rounds past it. A node number that rounds is a reference resolving
    to some other vertex, so it is refused rather than trusted.
    """
    if column.size:
        if not np.all(np.isfinite(column)) or np.max(np.abs(column)) > _MAX_EXACT_INT:
            raise CodecError(f".tetgen: {what} out of range in {where}.")
        if not np.array_equal(column, np.rint(column)):
            raise CodecError(f".tetgen: non-integer {what} in {where}.")
    return column.astype(np.int64)


def _records(
    tokens: list[str], start: int, n_rows: int, n_cols: int, what: str, where: str
) -> np.ndarray:
    """Return the body as an ``(n_rows, n_cols)`` float array.

    Both files list fixed-width rows, so the body is read in one block rather
    than field by field; a row short of its width is what a header describing
    columns the file does not carry looks like, and reading on from there would
    pull the next row's leading number in as this row's last value.
    """
    # Counted rather than sliced: ``tokens[start:]`` would copy the whole body
    # only to copy the part that is wanted out of it again, which on a mesh of
    # any size is two throwaway lists the length of the file.
    available = len(tokens) - start
    needed = n_rows * n_cols
    if available < needed:
        raise CodecError(
            f".tetgen: {where} truncated — {n_rows} {what} of {n_cols} column(s)"
            f" need {needed} value(s), found {available}."
        )
    if available > needed:
        # Trailing values are ones this reader will never see; going quiet
        # would report a smaller file than the one on disk.
        warnings.warn(
            f".tetgen: {where} carries {available - needed} value(s) past the"
            f" {n_rows} {what} its header declares; ignored.",
            stacklevel=4,
        )
    try:
        return np.array(tokens[start : start + needed], dtype=np.float64).reshape(
            n_rows, n_cols
        )
    except ValueError as exc:
        raise CodecError(f".tetgen: non-numeric value in {where}.") from exc


def _read_node(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Read a ``.node`` file.

    Returns
    -------
    tuple
        ``(vertices, node ids, vertex attributes, boundary markers)``. The ids
        are what element node references resolve against, and the markers are
        an all-zero array when the file declares none.
    """
    tokens = _tokenize(path)
    if len(tokens) < 4:
        raise CodecError(f".tetgen: malformed .node header in '{path.name}'.")

    where = f"'{path.name}'"
    n_pts = _checked_count(tokens[0], MAX_SAFE_VERTICES, "node", where)
    dim = _small_int(tokens[1], ".node dimension", where)
    n_attrs = _checked_count(tokens[2], _MAX_ATTR_COLUMNS, "node attribute", where)
    if dim not in (2, 3):
        raise CodecError(f".tetgen: .node dimension {dim} in {where}, expected 2 or 3.")
    # The format has room for one marker column and no more, so any non-zero
    # flag means the same single column; a negative one means none.
    has_markers = int(_small_int(tokens[3], "boundary marker flag", where) > 0)

    n_cols = 1 + dim + n_attrs + has_markers
    rows = _records(tokens, 4, n_pts, n_cols, "node(s)", where)

    vertices = np.zeros((n_pts, 3), dtype=np.float64)
    vertices[:, :dim] = rows[:, 1 : 1 + dim]
    ids = _as_ints(rows[:, 0], "node number", where)

    # TetGen gives its node attributes no names, so they are numbered; dropping
    # them would lose the only per-vertex field the format carries.
    attrs = {
        f"attr_{k}": np.ascontiguousarray(rows[:, 1 + dim + k]) for k in range(n_attrs)
    }
    markers = (
        _as_ints(rows[:, -1], "boundary marker", where)
        if has_markers
        else np.zeros(n_pts, dtype=np.int64)
    )
    return vertices, ids, attrs, markers


def _resolve_ids(refs: np.ndarray, ids: np.ndarray, n_pts: int) -> np.ndarray:
    """Map ``.ele`` node references onto rows of the ``.node`` file.

    TetGen numbers from 0 or from 1 depending on the switch it ran under, and
    the number is the file's own — a hand-written ``.node`` may skip one. The
    contiguous case is the overwhelming one and is a subtraction; anything else
    goes through a sorted lookup rather than being assumed away.

    Raises
    ------
    CodecError
        If a reference names a node the ``.node`` file does not list.
    """
    if refs.size == 0:
        return np.zeros(0, dtype=np.int64)
    if n_pts == 0:
        raise CodecError(
            ".tetgen: the .ele file lists elements but the .node file is empty."
        )

    # The endpoints settle it for free whenever the run is gapped, which is the
    # only case that would otherwise pay for an arange it goes on to discard.
    first = int(ids[0])
    contiguous = int(ids[-1]) - first == n_pts - 1 and np.array_equal(
        ids, np.arange(first, first + n_pts, dtype=np.int64)
    )
    if contiguous:
        mapped = refs - first
        # The offending number, not the span every reference covers: a mesh
        # whose references are sound but for one names the one, and the sorted
        # branch below already reports it that way.
        outside = (mapped < 0) | (mapped >= n_pts)
        if outside.any():
            raise CodecError(
                f".tetgen: element references node {int(refs[outside][0])},"
                f" outside the {first}..{first + n_pts - 1} the .node file lists."
            )
        return mapped

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        # Two rows under one number make every reference to it ambiguous; the
        # first is kept because argsort is stable, and a silent pick would be
        # a wrong vertex rather than a reported one.
        warnings.warn(
            ".tetgen: the .node file numbers two or more nodes alike; references"
            " to a repeated number resolve to the first of them.",
            stacklevel=3,
        )
    pos = np.searchsorted(sorted_ids, refs)
    clipped = np.clip(pos, 0, n_pts - 1)
    if not np.all(sorted_ids[clipped] == refs):
        missing = refs[sorted_ids[clipped] != refs]
        raise CodecError(
            f".tetgen: element references node {int(missing[0])}, which the"
            " .node file does not list."
        )
    return order[clipped].astype(np.int64)


def _read_ele(path: Path) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
    """Read an ``.ele`` file.

    Returns
    -------
    tuple
        ``(node references as (n_tets, corners), nodes per tet as declared,
        element attributes)``.
    """
    tokens = _tokenize(path)
    if len(tokens) < 3:
        raise CodecError(f".tetgen: malformed .ele header in '{path.name}'.")

    where = f"'{path.name}'"
    n_tets = _checked_count(tokens[0], MAX_SAFE_ELEMENTS, "element", where)
    n_nodes = _small_int(tokens[1], "nodes-per-element", where)
    n_attrs = _checked_count(tokens[2], _MAX_ATTR_COLUMNS, "element attribute", where)
    if n_nodes not in (4, 10):
        raise CodecError(
            f".tetgen: .ele declares {n_nodes} node(s) per element in {where},"
            " expected 4 (linear) or 10 (quadratic)."
        )
    if n_tets * 4 > MAX_SAFE_CONN:
        raise CodecError(
            f".tetgen: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
        )

    rows = _records(tokens, 3, n_tets, 1 + n_nodes + n_attrs, "element(s)", where)
    refs = _as_ints(rows[:, 1:5].reshape(-1), "node reference", where).reshape(-1, 4)

    # A quadratic tetrahedron's six extra nodes sit on its edges, but TetGen's
    # order for them is not the one polyxios's quadratic_tetra is spelled in,
    # and a wrong permutation is a silently bent element. The corners are
    # unambiguous, so the element is linearized and the loss is reported.
    if n_nodes == 10 and n_tets:
        warnings.warn(
            f".tetgen: {where} holds 10-node quadratic tetrahedra; the six"
            " mid-edge nodes were dropped and the elements read as linear.",
            stacklevel=3,
        )

    # Attributes start after the element number and every node the header
    # declares — 5 for a linear tet but 11 for a quadratic one. Fixing the
    # offset at 5 would read a mid-edge node number as the region and never
    # fail, since those columns exist either way.
    first_attr = 1 + n_nodes
    if n_attrs == 1:
        attrs = {"region": np.ascontiguousarray(rows[:, first_attr])}
    else:
        attrs = {
            f"region_{k}": np.ascontiguousarray(rows[:, first_attr + k])
            for k in range(n_attrs)
        }
    return refs, n_nodes, attrs


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a TetGen ``.node`` + ``.ele`` pair and return a PolyData.

    Parameters
    ----------
    path
        Path to either half of the pair; the other is found beside it under the
        same stem.
    lazy
        Ignored; a true value is warned about (ASCII format, always loads
        eagerly).

    Returns
    -------
    PolyData
        Tetrahedra from the ``.ele`` file over the vertices of the ``.node``
        file. A ``.node`` file with no ``.ele`` beside it reads as the point
        cloud it is.

    Raises
    ------
    CodecError
        On a missing ``.node`` file, a malformed, negative or absurd header, a
        dimension other than 2 or 3, an element width other than 4 or 10, a
        truncated or non-numeric body, or a node reference the ``.node`` file
        does not list.

    Notes
    -----
    Node numbers are the ``.node`` file's own, so a mesh numbered from 0, from
    1, or with gaps resolves the same way; the ``.ele`` file's element numbers
    are never used to shift them. A 2-D ``.node`` file is padded to ``z=0``.

    Node attributes become ``vertex_attrs`` named ``attr_<k>`` — TetGen gives
    them no names — and element attributes become ``element_attrs``, named
    ``region`` when the file declares one and ``region_<k>`` when it declares
    several. Each distinct non-zero boundary marker becomes a ``vertex_tags``
    entry named ``boundary_<marker>``; marker 0 means unmarked and is dropped.

    10-node quadratic tetrahedra are read as linear ones: TetGen's mid-edge
    order is not polyxios's, and a wrong permutation would bend the element in
    silence. The loss is warned about, as are values past the count a header
    declares and a ``.node`` file numbering two nodes alike.
    """
    if lazy:
        warnings.warn(
            ".tetgen: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    given = Path(path)
    node_path = _find_half(given, ".node")
    ele_path = _find_half(given, ".ele")
    if not _exists(node_path):
        raise CodecError(f".tetgen: .node file not found: '{node_path}'.")

    vertices, ids, vertex_attrs, markers = _read_node(node_path)
    n_pts = vertices.shape[0]

    if _exists(ele_path):
        refs, _n_nodes, element_attrs = _read_ele(ele_path)
        n_tets = refs.shape[0]
        flat = _resolve_ids(refs.reshape(-1), ids, n_pts)
    else:
        if given.suffix.lower() == ".ele":
            # The caller named this file. Handing back a point cloud instead
            # would answer a question they did not ask.
            raise CodecError(f".tetgen: .ele file not found: '{ele_path}'.")
        # A .node file stands on its own as a point set, and refusing to read
        # one the caller named outright would be worse than returning it.
        warnings.warn(
            f".tetgen: no .ele file beside '{node_path.name}'; read the .node"
            " file as a point cloud.",
            stacklevel=2,
        )
        n_tets = 0
        flat = np.zeros(0, dtype=np.int64)
        element_attrs = {}

    # The offsets run to n_tets * 4, which outgrows int32 well before the
    # vertex count does, so the widest of the two is what picks the dtype.
    widest = max(n_pts, n_tets * 4)
    idx_dtype = np.int64 if widest > np.iinfo(np.int32).max else np.int32
    connectivity = flat.astype(idx_dtype)
    offsets = np.arange(0, n_tets * 4 + 1, 4, dtype=idx_dtype)
    element_types = np.full(n_tets, _TETRA_CODE, dtype=np.uint8)

    # One tag per distinct marker, not one tag over every marked vertex: the
    # markers are what tell two boundaries apart, and a single tag would fuse
    # them into an unrecoverable whole.
    vertex_tags: dict[str, np.ndarray] = {}
    for value in np.unique(markers[markers != 0]):
        members = np.flatnonzero(markers == value)
        vertex_tags[f"boundary_{int(value)}"] = members.astype(np.int32)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
    )


def _tetra_elements(poly: PolyData) -> np.ndarray:
    """Return the indices of the tetrahedra, warning about what is left out.

    Raises
    ------
    CodecError
        If a tetrahedron does not carry exactly four nodes — TetGen records no
        per-element width, so a wrong one would read back as a different mesh.
    """
    types = np.asarray(poly.element_types)
    is_tet = types == _TETRA_CODE
    if not np.all(is_tet):
        skipped = {
            ELEMENT_TYPES_INV.get(int(code), f"code {int(code)}")
            for code in np.unique(types[~is_tet])
        }
        warnings.warn(
            f".tetgen: unsupported element type(s) {sorted(skipped)}; skipped."
            " A TetGen .ele file holds tetrahedra only.",
            stacklevel=3,
        )

    keep = np.flatnonzero(is_tet)
    widths = np.diff(np.asarray(poly.offsets))[keep]
    bad = np.flatnonzero(widths != 4)
    if bad.size:
        raise CodecError(
            f".tetgen: element {int(keep[bad[0]])} is a tetra carrying"
            f" {int(widths[bad[0]])} node(s), expected 4."
        )
    return keep


def _tetra_refs(poly: PolyData, keep: np.ndarray) -> np.ndarray:
    """Gather the node references of ``keep`` as an ``(n_tets, 4)`` array."""
    if keep.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    starts = np.asarray(poly.offsets)[keep].astype(np.int64)
    return np.asarray(poly.connectivity)[starts[:, None] + np.arange(4)]


def _boundary_markers(poly: PolyData) -> np.ndarray:
    """Fold ``vertex_tags`` into one integer marker per vertex.

    TetGen gives a node a single marker, so a vertex in two tags keeps the
    first that claims it. ``boundary_<n>`` recovers the number reading gave it;
    any other name is numbered from 1 upward so the tag still travels.
    """
    n_verts = poly.vertices.shape[0]
    markers = np.zeros(n_verts, dtype=np.int64)
    if not poly.vertex_tags:
        return markers

    claimed = np.zeros(n_verts, dtype=bool)
    shared = 0
    out_of_range = 0
    used: set[int] = set()
    named: list[tuple[str, int]] = []

    for name in sorted(poly.vertex_tags, key=lambda n: _natural_key(str(n))):
        match = _BOUNDARY_RE.match(str(name))
        # Marker 0 is TetGen's word for unmarked, so a tag literally named
        # ``boundary_0`` cannot keep its number without vanishing; it takes a
        # fresh one instead, the same way an unnumbered name does. A number
        # past what reading resolves exactly goes the same way: keeping it
        # would write a marker that comes back as a different one, or as the
        # error that refuses it.
        value = int(match.group(1)) if match is not None else 0
        if abs(value) > _MAX_EXACT_INT:
            value = 0
        named.append((name, value))
        if value:
            used.add(value)

    nxt = 1
    for name, value in named:
        if value == 0:
            while nxt in used:
                nxt += 1
            value = nxt
            used.add(value)
        idx = np.unique(np.asarray(poly.vertex_tags[name]).ravel().astype(np.int64))
        inside = idx[(idx >= 0) & (idx < n_verts)]
        out_of_range += idx.size - inside.size
        fresh = inside[~claimed[inside]]
        shared += inside.size - fresh.size
        markers[fresh] = value
        claimed[fresh] = True

    if shared:
        warnings.warn(
            f".tetgen: {shared} vertex/vertices belong to more than one tag; a"
            " TetGen node carries one marker, so each kept the first that"
            " claims it.",
            stacklevel=3,
        )
    if out_of_range:
        warnings.warn(
            f".tetgen: vertex tag(s) name {out_of_range} index(es) outside the"
            f" mesh's {n_verts} vertex/vertices; those members were dropped.",
            stacklevel=3,
        )
    return markers


def _node_attrs(poly: PolyData) -> list[np.ndarray]:
    """Return the vertex attributes a ``.node`` file can carry, in name order.

    A node attribute is one number per node, so a multi-component array — a
    normal, a colour — has no column to go in and is left out with a word.

    No more columns are written than reading admits. Going past that cap would
    put a ``.node`` on disk that this codec refuses on the way back in, which
    is a worse answer than a warning and a file that reads.
    """
    n_verts = poly.vertices.shape[0]
    columns: list[np.ndarray] = []
    kept: list[str] = []
    skipped: list[str] = []
    for name in sorted(poly.vertex_attrs, key=lambda n: _natural_key(str(n))):
        arr = np.asarray(poly.vertex_attrs[name])
        if arr.ndim != 1 or arr.shape[0] != n_verts or arr.dtype.kind not in "fiub":
            skipped.append(str(name))
            continue
        columns.append(arr.astype(np.float64, copy=False))
        kept.append(str(name))
    if skipped:
        warnings.warn(
            f".tetgen: vertex attribute(s) {sorted(skipped)} are not one scalar"
            " per vertex and have no column in a .node file; skipped.",
            stacklevel=3,
        )
    if len(columns) > _MAX_ATTR_COLUMNS:
        over = kept[_MAX_ATTR_COLUMNS:]
        warnings.warn(
            f".tetgen: {len(over)} vertex attribute(s) past the"
            f" {_MAX_ATTR_COLUMNS}-column cap a .node file is read back under"
            f" were skipped, the last kept being {kept[_MAX_ATTR_COLUMNS - 1]!r}.",
            stacklevel=3,
        )
        columns = columns[:_MAX_ATTR_COLUMNS]
    return columns


def _region_column(poly: PolyData, keep: np.ndarray) -> np.ndarray | None:
    """Return the per-element region attribute, or None when there is none.

    A ``.ele`` file carries one attribute per element in practice, so the one
    named ``region`` — or ``region_0``, which is what reading a multi-attribute
    file leaves behind — is the one written, and the rest are named as lost.
    A mesh from another format names its one attribute whatever that format
    called it, ``material`` or ``zone``; dropping it for want of the spelling
    this codec reads back would lose the only per-element field the file can
    hold, so the first by name takes the column instead.
    """
    n_elems = len(poly.element_types)
    chosen: str | None = next(
        (name for name in ("region", "region_0") if name in poly.element_attrs), None
    )
    if chosen is None and poly.element_attrs:
        chosen = sorted(poly.element_attrs, key=lambda n: _natural_key(str(n)))[0]
        warnings.warn(
            f".tetgen: element attribute {chosen!r} was written as the .ele"
            " region attribute; the format stores no attribute names, so it"
            " reads back as 'region'.",
            stacklevel=3,
        )
    dropped = sorted(set(poly.element_attrs) - {chosen})
    if dropped:
        warnings.warn(
            f".tetgen: element attribute(s) {dropped} have no column in a .ele"
            " file, which carries one region attribute; skipped.",
            stacklevel=3,
        )
    if chosen is None:
        return None

    arr = np.asarray(poly.element_attrs[chosen])
    if arr.ndim != 1 or arr.shape[0] != n_elems or arr.dtype.kind not in "fiub":
        warnings.warn(
            f".tetgen: element attribute {chosen!r} is not one scalar per"
            " element; no region attribute was written.",
            stacklevel=3,
        )
        return None
    return arr.astype(np.float64, copy=False)[keep]


def _write_node(
    path: Path,
    vertices: np.ndarray,
    attrs: list[np.ndarray],
    markers: np.ndarray,
    float_fmt: str,
) -> None:
    n_verts = vertices.shape[0]
    has_markers = int(bool(np.any(markers)))
    lines = [f"{n_verts} 3 {len(attrs)} {has_markers}"]
    # One crossing into Python per array rather than one per number: indexing a
    # numpy array a scalar at a time is what costs on a mesh of any size.
    coords = np.asarray(vertices, dtype=np.float64).tolist()
    attr_rows = np.column_stack(attrs).tolist() if attrs else [()] * n_verts
    marker_list = markers.tolist() if has_markers else ()
    for i, xyz in enumerate(coords):
        parts = [str(i + 1)]
        parts.extend(format(c, float_fmt) for c in xyz)
        parts.extend(format(a, float_fmt) for a in attr_rows[i])
        if has_markers:
            parts.append(str(int(marker_list[i])))
        lines.append("\t".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ele(
    path: Path,
    refs: np.ndarray,
    regions: np.ndarray | None,
    float_fmt: str,
) -> None:
    lines = [f"{refs.shape[0]} 4 {0 if regions is None else 1}"]
    corners = np.asarray(refs, dtype=np.int64).tolist()
    region_list = regions.tolist() if regions is not None else ()
    for seq, nodes in enumerate(corners):
        parts = [str(seq + 1)]
        parts.extend(str(v + 1) for v in nodes)
        if regions is not None:
            parts.append(format(region_list[seq], float_fmt))
        lines.append("\t".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a TetGen ``.node`` + ``.ele`` pair.

    Parameters
    ----------
    poly
        PolyData to write. Only tetrahedra reach the ``.ele`` file.
    path
        Output path; the pair is written to its stem with ``.node`` and
        ``.ele`` appended, whichever of the two the path names. A named half
        keeps the case it was spelled in and its sibling copies it.
    **opts
        ``float_fmt`` overrides the ASCII coordinate format specifier. Any
        other option is warned about and ignored.

    Raises
    ------
    CodecError
        If ``float_fmt`` is not a usable format specifier or writes a token
        that is not a number, a tetrahedron does not carry four nodes, or an
        element references a vertex that does not exist.

    Notes
    -----
    Both files are written numbered from 1, which is TetGen's default. Every
    vertex is written whether an element reaches it or not, so node references
    survive the trip unshifted.

    Non-tetrahedral elements have no place in a ``.ele`` file and are skipped
    with a warning. ``vertex_tags`` fold into one boundary marker per node —
    a ``boundary_<n>`` name keeps its number, ``boundary_0`` and any other
    name are numbered from 1 instead, since marker 0 is TetGen's word for
    unmarked, and a vertex in two tags keeps the first that claims it in name
    order. Scalar ``vertex_attrs`` become node attributes in name order and
    read back named ``attr_<k>``, since the format stores no names; a number
    inside a name orders by value, so ``attr_10`` follows ``attr_9`` rather
    than ``attr_1``. Anything wider than one column per vertex is skipped with
    a warning, as is anything past the column cap reading admits, so the pair
    that lands on disk is always one this codec reads back. One scalar
    ``element_attrs`` entry becomes the element attribute — ``region`` when the
    mesh has it, else the first name, warned about because it reads back as
    ``region``. ``element_tags`` and ``global_attrs`` have nowhere to go and
    are not written.
    """
    float_fmt = opts.pop("float_fmt", _DEFAULT_FLOAT_FMT)
    if opts:
        warnings.warn(
            f".tetgen write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )
    try:
        probe = format(-1234.5, float_fmt)
    except (TypeError, ValueError) as exc:
        raise CodecError(f".tetgen: float_fmt {float_fmt!r} is not usable.") from exc
    try:
        # A grouping or percent format is a legal specifier that writes tokens
        # like ``-1,234.50``; the pair would land on disk and only fail on the
        # way back in, so it is refused here rather than by the reader.
        float(probe)
    except ValueError as exc:
        raise CodecError(
            f".tetgen: float_fmt {float_fmt!r} writes {probe!r}, which is not a"
            " number the reader can parse back."
        ) from exc

    given = Path(path)
    node_path, ele_path = _pair_paths(given)

    keep = _tetra_elements(poly)
    refs = _tetra_refs(poly, keep)
    n_verts = poly.vertices.shape[0]
    if refs.size:
        lo, hi = int(refs.min()), int(refs.max())
        if lo < 0 or hi >= n_verts:
            where = f"0..{n_verts - 1}" if n_verts else "an empty vertex array"
            raise CodecError(
                f".tetgen: element references vertex {lo}..{hi}, outside {where}."
            )

    regions = _region_column(poly, keep)
    # Every check the mesh can fail runs above, so a rejected mesh does not
    # leave a .node file behind with no .ele to go with it. An OSError from the
    # filesystem itself can still land between the two writes.
    _write_node(
        node_path, poly.vertices, _node_attrs(poly), _boundary_markers(poly), float_fmt
    )
    _write_ele(ele_path, refs, regions, float_fmt)
