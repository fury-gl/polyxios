"""ASCII point-cloud codec - ``.xyz`` columns, Leica ``.pts`` and ``.ptx``."""

from __future__ import annotations

import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import Source, format_suffix, read_text, source_name, write_text
from polyxios._types import PolyData
from polyxios.codecs._pointcloud import (
    BYTE_SCALE,
    check_float_fmt,
    color_bytes,
    point_cloud,
)
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".xyz"
EXTENSIONS: tuple[str, ...] = (".xyz", ".pts", ".ptx")

_XYZ: str = "xyz"
_PTS: str = "pts"
_PTX: str = "ptx"
_VARIANTS: frozenset[str] = frozenset({_XYZ, _PTS, _PTX})
_DEFAULT_FLOAT_FMT: str = ".10g"

_INTENSITY: str = "intensity"
_COLORS: str = "colors"
_NORMALS: str = "normals"
_EXTRA: str = "extra"

_SCAN_TAG: str = "scan_{}"
_TRANSFORMS_KEY: str = "ptx_transforms"
_DIMENSIONS_KEY: str = "ptx_dimensions"

# A PTX scan starts with ten header lines: the grid's columns and rows, the
# scanner position, its three axes, and a 4 x 4 row-vector transform.
_PTX_HEADER_SHAPE: tuple[int, ...] = (1, 1, 3, 3, 3, 3, 4, 4, 4, 4)
_PTX_HEADER_LINES: int = len(_PTX_HEADER_SHAPE)
_PTX_INVALID: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)
_PTX_DEFAULT_INTENSITY: float = 1.0

_VERTEX_CODE: int = ELEMENT_TYPES["vertex"]
_COMMENT_MARKS: tuple[str, ...] = ("#", "//")
_NON_BLANK: re.Pattern[str] = re.compile(r"\S")
_NEWLINE: int = ord("\n")
_BLANK_BYTE: np.ndarray = np.zeros(256, dtype=bool)
_BLANK_BYTE[[9, 10, 11, 12, 13, 32]] = True
_BOM: str = "\ufeff"


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an ASCII point cloud and return a PolyData of points.

    Parameters
    ----------
    path
        Path or file object of the ``.xyz``, ``.pts`` or ``.ptx`` file. The
        layout is told from the content, not the name: a leading line of
        one integer is a point count, a ``.ptx`` scan header is ten lines
        of a fixed shape, and everything else is bare columns.
    lazy
        Not supported - raises ``LazyReadError``; a number spelled as text
        has to be parsed before it is a number.

    Returns
    -------
    PolyData
        Points only, no elements. Three columns are the coordinates; what
        follows is read by count and by kind. With 1, 4 or 7 more columns
        the first is ``vertex_attrs["intensity"]``. Three (left) are
        ``vertex_attrs["colors"]`` (floats in 0..1) when they are integers
        in 0..255 with one above 1, else ``vertex_attrs["normals"]``; six
        are ``colors`` and ``normals`` in whichever order the colour test
        tells, or, when neither half is a colour, one six-wide
        ``vertex_attrs["extra"]``. Any other width (2, 5, 8 or more) is
        kept whole as ``extra``, intensity and all. A ``.ptx``
        file's scans are each transformed by their own matrix into one
        frame, the invalid points (``0 0 0 0.5``) dropped with a warning,
        every scan tagged ``scan_<k>`` in ``vertex_tags`` when there are
        several, and the matrices and grid sizes kept in ``global_attrs``
        as ``ptx_transforms`` (``(scans, 4, 4)``) and ``ptx_dimensions``
        (``(scans, 2)``, one ``(rows, cols)`` pair per scan, the reverse of
        the header's ``cols`` then ``rows`` lines).

    Raises
    ------
    LazyReadError
        Always, if ``lazy`` is set.
    CodecError
        On a row that is not numbers, rows of unequal width, fewer than
        three columns, or a count that the file does not hold.
    """
    if lazy:
        raise LazyReadError(
            f"{source_name(path)}: ASCII point clouds do not support lazy reads."
        )
    name = source_name(path)
    text = read_text(path, errors="replace").removeprefix(_BOM)
    start = _first_data_offset(text)
    if start is None:
        return _cloud(np.empty((0, 3)), {}, {}, {})
    head_lines = _lines_from(text, start, _PTX_HEADER_LINES)
    head = head_lines[0].split()
    if len(head) == 1 and _is_int(head[0]):
        if _looks_like_ptx(head_lines, 0):
            lines = text.split("\n")
            return _read_ptx(lines, text.count("\n", 0, start), name=name)
        count = int(head[0])
        table = _table(text, start + len(head_lines[0]) + 1, name=name)
        # Fewer than three columns is the fault to name whatever the
        # count says; _split_columns reports it below.
        if table.shape[1] >= 3 and table.shape[0] != count:
            raise CodecError(
                f"'{name}': the count line says {count} points, the file holds "
                f"{table.shape[0]}."
            )
    else:
        table = _table(text, start, name=name)
    verts, attrs = _split_columns(table, name=name)
    return _cloud(verts, attrs, {}, {})


def _first_data_offset(text: str) -> int | None:
    """The offset of the first line that is neither blank nor a comment."""
    pos = 0
    n = len(text)
    while pos < n:
        end = text.find("\n", pos)
        if end < 0:
            end = n
        line = text[pos:end]
        if line.strip() and not line.lstrip().startswith(_COMMENT_MARKS):
            return pos
        pos = end + 1
    return None


def _lines_from(text: str, start: int, count: int) -> list[str]:
    """Up to ``count`` lines from ``start``, without splitting the rest."""
    out: list[str] = []
    pos = start
    n = len(text)
    while pos < n and len(out) < count:
        end = text.find("\n", pos)
        if end < 0:
            end = n
        out.append(text[pos:end])
        pos = end + 1
    return out


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_ptx(lines: list[str], first: int) -> bool:
    """Whether ten lines from ``first`` have a PTX scan header's shape."""
    if first + _PTX_HEADER_LINES > len(lines):
        return False
    for k, width in enumerate(_PTX_HEADER_SHAPE):
        tokens = lines[first + k].split()
        # The first two lines are the grid's column and row counts.
        accept = _is_int if k < 2 else _is_number
        if len(tokens) != width or not all(accept(t) for t in tokens):
            return False
    return True


def _table(text: str, start: int, *, name: str) -> np.ndarray:
    """Parse the rows from ``start`` on as one numeric table.

    Comment lines and blank lines are dropped; a first row whose first
    token is not a number is taken for a column header and dropped too.
    The body goes to numpy in one pass; the lines are only split apart
    when a comment mark is in them, so a file without any is never held
    twice.
    """
    body = text[start:]
    if any(mark in body for mark in _COMMENT_MARKS):
        body = "\n".join(
            line
            for line in body.split("\n")
            if line.strip() and not line.lstrip().startswith(_COMMENT_MARKS)
        )
    head_line, after = _first_line(body)
    head = head_line.split()
    if head and not _is_number(head[0]) and any(c.isalpha() for c in head_line):
        # Column labels, "X Y Z R G B". A row with a stray token after its
        # numbers still starts with a number and is refused below, and one
        # with the wrong separator has no letters and is refused too.
        body = body[after:]
        head_line, _ = _first_line(body)
        head = head_line.split()
    if not head:
        return np.empty((0, 3))
    width = len(head)
    with warnings.catch_warnings():
        # A stray token makes numpy warn that it stopped early; that is a
        # malformed row and reported as one.
        warnings.simplefilter("error", DeprecationWarning)
        try:
            values = np.fromstring(body, dtype=np.float64, sep=" ")
        except (DeprecationWarning, ValueError) as exc:
            raise CodecError(
                f"'{name}': a row holds something that is not a number."
            ) from exc
    _check_row_widths(body, width, name=name)
    return values.reshape(-1, width)


def _check_row_widths(body: str, width: int, *, name: str) -> None:
    """Refuse a body whose non-blank rows are not all ``width`` tokens.

    Parameters
    ----------
    body
        The rows, already parsed as numbers without complaint.
    width
        The first row's token count.
    name
        The source, for messages.

    Raises
    ------
    CodecError
        Naming the first row of another width.

    Notes
    -----
    ``np.fromstring`` returns one flat array and cannot say where a line
    ended, and a total that divides by ``width`` proves nothing when one
    row is short and another long. One numpy pass over the bytes finds the
    token starts and the newlines, and the count between two newlines is a
    row's width; it costs less than the parse itself.
    """
    buf = np.frombuffer(body.encode("ascii", "replace"), dtype=np.uint8)
    if buf.size == 0:
        return
    blank = _BLANK_BYTE[buf]
    starts = np.flatnonzero(blank[:-1] & ~blank[1:]) + 1
    if not blank[0]:
        starts = np.concatenate(([0], starts))
    ends = np.searchsorted(starts, np.flatnonzero(buf == _NEWLINE))
    per_row = np.diff(np.concatenate(([0], ends, [starts.size])))
    per_row = per_row[per_row != 0]
    off = np.flatnonzero(per_row != width)
    if off.size:
        k = int(off[0])
        raise CodecError(
            f"'{name}': rows are not all {width} columns wide: row {k + 1} has "
            f"{int(per_row[k])}."
        )


def _first_line(body: str) -> tuple[str, int]:
    """The first non-blank line of ``body`` and the offset just past it."""
    hit = _NON_BLANK.search(body)
    if hit is None:
        return "", len(body)
    begin = body.rfind("\n", 0, hit.start()) + 1
    end = body.find("\n", hit.start())
    if end < 0:
        end = len(body)
    return body[begin:end], end + 1


def _colourish(cols: np.ndarray) -> bool:
    """Whether three columns are whole numbers in 0..255, as colours are.

    Columns of nothing but 0 and 1 are not: an axis-aligned normal field
    (``0 0 1`` on every point of a flat scan) spells exactly that, while a
    colour image whose every byte is 0 or 1 is all but black.
    """
    if cols.shape[1] != 3 or cols.shape[0] == 0:
        return False
    finite = np.isfinite(cols).all()
    return bool(
        finite
        and np.all(np.mod(cols, 1) == 0)
        and cols.min() >= 0
        and 1 < cols.max() <= BYTE_SCALE
    )


def _split_columns(
    table: np.ndarray, *, name: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Name the columns past the coordinates by their count and kind."""
    width = table.shape[1]
    if width < 3:
        raise CodecError(f"'{name}': {width} columns; a point needs x, y and z.")
    verts = np.ascontiguousarray(table[:, :3])
    rest = table[:, 3:]
    attrs: dict[str, np.ndarray] = {}
    k = rest.shape[1]
    if k == 0:
        return verts, attrs
    if k in (1, 4, 7):
        attrs[_INTENSITY] = np.ascontiguousarray(rest[:, 0])
        rest = rest[:, 1:]
        k -= 1
    if k == 3:
        if _colourish(rest):
            attrs[_COLORS] = rest / BYTE_SCALE
        else:
            attrs[_NORMALS] = np.ascontiguousarray(rest)
    elif k == 6:
        a, b = rest[:, :3], rest[:, 3:]
        if _colourish(a):
            attrs[_COLORS] = a / BYTE_SCALE
            attrs[_NORMALS] = np.ascontiguousarray(b)
        elif _colourish(b):
            attrs[_NORMALS] = np.ascontiguousarray(a)
            attrs[_COLORS] = b / BYTE_SCALE
        else:
            attrs[_EXTRA] = np.ascontiguousarray(rest)
    elif k:
        attrs[_EXTRA] = np.ascontiguousarray(table[:, 3:])
    return verts, attrs


def _refuse_short_scan(lines: list[str], start: int, count: int, *, name: str) -> None:
    """Raise when the next scan's header begins inside this scan's rows.

    Parameters
    ----------
    lines
        The whole file, one entry per line.
    start
        The line the scan's rows begin at.
    count
        The points the scan declares.
    name
        The source, for messages.

    Raises
    ------
    CodecError
        Naming the points declared and the rows actually held, when a
        header sits within the ``count`` lines from ``start``. A scan that
        is short by any other cause returns and leaves the caller's
        message to stand.

    Notes
    -----
    A header line holds a single token where a point holds four or seven,
    so a data row never passes for one. The check runs only on a scan
    already found faulty, so a well-formed file pays nothing for it.
    """
    header = next((k for k in range(count) if _looks_like_ptx(lines, start + k)), None)
    if header is not None:
        raise CodecError(f"'{name}': PTX scan declares {count} points, holds {header}.")


def _read_ptx(lines: list[str], first: int, *, name: str) -> PolyData:
    pos = first
    chunks: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    colours: list[np.ndarray | None] = []
    transforms: list[np.ndarray] = []
    dims: list[tuple[int, int]] = []
    dropped = 0
    n_lines = len(lines)
    while pos < n_lines and _looks_like_ptx(lines, pos):
        cols = int(lines[pos].split()[0])
        rows = int(lines[pos + 1].split()[0])
        matrix = np.array(
            [[float(t) for t in lines[pos + 6 + r].split()] for r in range(4)],
            dtype=np.float64,
        )
        if rows < 0 or cols < 0:
            raise CodecError(f"'{name}': PTX scan of {cols} x {rows} points.")
        count = rows * cols
        start = pos + _PTX_HEADER_LINES
        body = lines[start : start + count]
        blank = next((k for k, line in enumerate(body) if not line.strip()), None)
        if len(body) < count or blank is not None:
            tail = lines[start + (len(body) if blank is None else blank) :]
            if any(line.strip() for line in tail):
                raise CodecError(
                    f"'{name}': PTX scan of {count} points holds a blank line."
                )
            raise CodecError(
                f"'{name}': PTX scan declares {count} points, the file holds fewer."
            )
        try:
            table = _table("\n".join(body), 0, name=name) if count else np.empty((0, 4))
        except CodecError:
            _refuse_short_scan(lines, start, count, name=name)
            raise
        if table.shape[0] != count or table.shape[1] not in (4, 7):
            _refuse_short_scan(lines, start, count, name=name)
        if table.shape[0] != count:
            raise CodecError(
                f"'{name}': PTX scan declares {count} points, found {table.shape[0]}."
            )
        if table.shape[1] not in (4, 7):
            raise CodecError(
                f"'{name}': a PTX point is 4 or 7 columns, found {table.shape[1]}."
            )
        valid = ~np.all(table[:, :4] == _PTX_INVALID, axis=1)
        dropped += int(count - valid.sum())
        table = table[valid]
        xyz = table[:, :3]
        homogeneous = np.column_stack([xyz, np.ones(xyz.shape[0])]) @ matrix
        chunks.append(homogeneous[:, :3])
        intensities.append(table[:, 3])
        if count:
            # An empty scan has no say in whether the file carries colours.
            colours.append(table[:, 4:7] / BYTE_SCALE if table.shape[1] == 7 else None)
        transforms.append(matrix)
        dims.append((rows, cols))
        pos = start + count
        while pos < n_lines and not lines[pos].strip():
            pos += 1
    if pos < n_lines and any(line.strip() for line in lines[pos:]):
        raise CodecError(
            f"'{name}': text after the last PTX scan is not a scan header."
        )
    if dropped:
        warnings.warn(
            f"'{name}': {dropped} invalid PTX points (0 0 0 0.5) dropped.", stacklevel=4
        )
    verts = np.concatenate(chunks) if chunks else np.empty((0, 3))
    attrs: dict[str, np.ndarray] = {_INTENSITY: np.concatenate(intensities)}
    present = [c for c in colours if c is not None]
    if present and len(present) == len(colours):
        attrs[_COLORS] = np.concatenate(present)
    elif present:
        warnings.warn(
            f"'{name}': only some PTX scans carry colours; none kept.", stacklevel=4
        )
    tags: dict[str, np.ndarray] = {}
    if len(chunks) > 1:
        offset = 0
        for k, chunk in enumerate(chunks):
            tags[_SCAN_TAG.format(k)] = np.arange(offset, offset + chunk.shape[0])
            offset += chunk.shape[0]
    globals_: dict[str, Any] = {
        _TRANSFORMS_KEY: np.stack(transforms),
        _DIMENSIONS_KEY: np.array(dims, dtype=np.int64),
    }
    return _cloud(verts, attrs, tags, globals_)


def _cloud(
    vertices: np.ndarray,
    vertex_attrs: dict[str, np.ndarray],
    vertex_tags: dict[str, np.ndarray],
    global_attrs: dict[str, Any],
) -> PolyData:
    return point_cloud(
        vertices=np.ascontiguousarray(vertices, dtype=np.float64),
        vertex_attrs=vertex_attrs,
        vertex_tags=vertex_tags,
        global_attrs=global_attrs,
    )


def write(
    poly: PolyData,
    path: Source,
    *,
    variant: str | None = None,
    float_fmt: str = _DEFAULT_FLOAT_FMT,
    **opts: Any,
) -> None:
    """Serialise a PolyData's vertices as ASCII columns.

    Parameters
    ----------
    poly
        The mesh; its vertices are the points. Elements are not part of
        the format: any that are not single vertices are dropped with a
        warning.
    path
        Destination path or file object.
    variant
        ``"xyz"``, ``"pts"`` or ``"ptx"``; by default the one the
        destination's suffix names, ``xyz`` for any other suffix or a
        nameless buffer.
    float_fmt
        Format spec for every number that is not a colour byte.

    Notes
    -----
    The columns are ``x y z``, then ``intensity`` when
    ``vertex_attrs`` holds it, then ``colors`` as three integers in
    0..255 (from floats in 0..1, or integers already in that range), then
    ``normals``. ``pts`` prefixes the point count on a line
    of its own. ``ptx`` writes one scan of the points in a single row with
    the identity transform, ``x y z intensity [r g b]``, an intensity of 1
    where the mesh has none, and no normals. Every other vertex attribute
    is dropped with a warning naming it, and no global attribute has a
    place in any of the three layouts.
    """
    if variant is None:
        suffix = format_suffix(path).lstrip(".")
        variant = suffix if suffix in _VARIANTS else _XYZ
    if variant not in _VARIANTS:
        raise ValueError(
            f"variant must be one of {sorted(_VARIANTS)}, not {variant!r}."
        )
    check_float_fmt(float_fmt, fmt=f".{variant}")
    n = poly.vertices.shape[0]
    non_vertex = int(np.count_nonzero(poly.element_types != _VERTEX_CODE))
    if non_vertex:
        warnings.warn(
            f".{variant} holds points only; {non_vertex} elements dropped.",
            stacklevel=3,
        )
    attrs = poly.vertex_attrs
    columns: list[tuple[np.ndarray, bool]] = [(poly.vertices, False)]
    kept: set[str] = set()
    intensity = np.asarray(attrs[_INTENSITY]) if _INTENSITY in attrs else None
    if intensity is not None and intensity.ndim == 2 and intensity.shape[1] == 1:
        intensity = intensity[:, 0]
    if intensity is not None and intensity.ndim == 1:
        columns.append((intensity.astype(np.float64)[:, None], False))
        kept.add(_INTENSITY)
    elif variant == _PTX:
        columns.append((np.full((n, 1), _PTX_DEFAULT_INTENSITY), False))
    colors = np.asarray(attrs[_COLORS]) if _COLORS in attrs else None
    if colors is not None and colors.ndim == 2 and colors.shape[1] in (3, 4):
        if colors.shape[1] == 4:
            warnings.warn(
                f".{variant} colours are r g b; the alpha channel is dropped.",
                stacklevel=3,
            )
        columns.append((color_bytes(colors[:, :3]), True))
        kept.add(_COLORS)
    normals = np.asarray(attrs[_NORMALS]) if _NORMALS in attrs else None
    if (
        variant != _PTX
        and normals is not None
        and normals.ndim == 2
        and normals.shape[1] == 3
    ):
        columns.append((normals.astype(np.float64), False))
        kept.add(_NORMALS)
    dropped = sorted(set(attrs) - kept)
    if dropped:
        warnings.warn(
            f".{variant} has no column for vertex attributes {', '.join(dropped)};"
            " dropped.",
            stacklevel=3,
        )
    rows = _spell(columns, n, float_fmt=float_fmt)
    if variant == _PTS:
        head = f"{n}\n"
    elif variant == _PTX:
        head = (
            f"{n}\n1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        )
    else:
        head = ""
    write_text(path, head + rows)


def _spell(columns: list[tuple[np.ndarray, bool]], n: int, *, float_fmt: str) -> str:
    """Spell the columns one point per line, integers as such."""
    for arr, _ in columns:
        if arr.shape[0] != n:
            raise CodecError(
                f"a vertex attribute has {arr.shape[0]} rows for {n} points."
            )
    if n == 0:
        return ""
    rows: list[list[str]] = [[] for _ in range(n)]
    for arr, as_int in columns:
        flat = arr.reshape(n, -1)
        width = flat.shape[1]
        if as_int:
            tokens = [str(v) for v in flat.astype(np.int64).ravel()]
        else:
            tokens = [format(v, float_fmt) for v in flat.ravel()]
        for r, row in enumerate(rows):
            row.extend(tokens[r * width : (r + 1) * width])
    return "\n".join(" ".join(row) for row in rows) + "\n"
