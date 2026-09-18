"""SVG codec - write-only, a mesh drawn as a two-dimensional picture."""

from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, source_name, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, UnsupportedFormatError

EXTENSION: str = ".svg"

# Coordinates go out in the mesh's own units, so a fixed number of decimals
# would flatten a millimetre-scale mesh to a grid; significant digits do not.
_DEFAULT_FLOAT_FMT: str = ".6g"

# A format spec's fill character may be any character at all, including one
# that ends an XML attribute early. Every number lands in an attribute, so a
# spec that pads with one of these is refused up front.
_UNSAFE_IN_ATTRIBUTE: frozenset[str] = frozenset('"<>&')

# The default stroke as a fraction of the picture's shorter side, which keeps
# the line weight the same at every scale a mesh might be drawn at.
_STROKE_FRACTION: float = 0.01

# What a stroke falls back to when the picture has no extent to take a
# fraction of - one vertex, or none.
_UNIT_STROKE: float = 1.0

# The two coordinate columns each projection keeps, and the one it drops.
_PLANES: dict[str, tuple[int, int, int]] = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}
_AXES: str = "xyz"

# A dropped coordinate spanning less than this fraction of the picture's
# longer side is flat enough: the tolerance is relative, so a plane at
# z=1e-12 in a unit mesh warns no more than one at z=0.
_FLAT_RTOL: float = 1e-9

# The corner nodes each drawable type is outlined through, and whether the
# outline closes. A closed outline is a face; an open one a curve. A
# higher-order element is drawn through its corners, which VTK lists first,
# the mid-side and interior nodes being no part of the silhouette. None means
# every node, in order, for the types whose size the element itself sets.
_CORNERS: dict[int, tuple[tuple[int, ...] | None, bool]] = {
    ELEMENT_TYPES["line"]: ((0, 1), False),
    ELEMENT_TYPES["poly_line"]: (None, False),
    ELEMENT_TYPES["triangle"]: ((0, 1, 2), True),
    ELEMENT_TYPES["polygon"]: (None, True),
    # VTK pixel nodes run row by row, so the ring passes through them as
    # 0 1 3 2 rather than in index order.
    ELEMENT_TYPES["pixel"]: ((0, 1, 3, 2), True),
    ELEMENT_TYPES["quad"]: ((0, 1, 2, 3), True),
    ELEMENT_TYPES["quadratic_edge"]: ((0, 1), False),
    ELEMENT_TYPES["cubic_line"]: ((0, 1), False),
    ELEMENT_TYPES["quadratic_triangle"]: ((0, 1, 2), True),
    ELEMENT_TYPES["biquadratic_triangle"]: ((0, 1, 2), True),
    ELEMENT_TYPES["quadratic_quad"]: ((0, 1, 2, 3), True),
    ELEMENT_TYPES["biquadratic_quad"]: ((0, 1, 2, 3), True),
    ELEMENT_TYPES["quadratic_linear_quad"]: ((0, 1, 2, 3), True),
    ELEMENT_TYPES["lagrange_curve"]: ((0, 1), False),
    ELEMENT_TYPES["lagrange_triangle"]: ((0, 1, 2), True),
    ELEMENT_TYPES["lagrange_quadrilateral"]: ((0, 1, 2, 3), True),
    ELEMENT_TYPES["bezier_curve"]: ((0, 1), False),
    ELEMENT_TYPES["bezier_triangle"]: ((0, 1, 2), True),
}

# A strip of n nodes is n - 2 triangles, one closed ring each; a quadratic
# polygon lists its corners first and its mid-sides after, so the ring is the
# first half. Neither fits the corner table's fixed spelling.
_TRIANGLE_STRIP: int = ELEMENT_TYPES["triangle_strip"]
_QUADRATIC_POLYGON: int = ELEMENT_TYPES["quadratic_polygon"]

_DRAWABLE: frozenset[int] = frozenset(_CORNERS) | {_TRIANGLE_STRIP, _QUADRATIC_POLYGON}

# Fewer nodes than this cannot be joined by a segment: two for a free-size
# type, three for a strip's first triangle, and four for a quadratic polygon,
# whose first two corners come with their two mid-sides.
_MIN_DRAWN_NODES: int = 2
_MIN_STRIP_NODES: int = 3
_MIN_QUADRATIC_POLYGON_NODES: int = 4

_XMLNS: str = "http://www.w3.org/2000/svg"


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Raise UnsupportedFormatError - SVG is written, never read.

    Parameters
    ----------
    path
        Path to the .svg file.
    lazy
        Ignored; the error is raised immediately.
    **opts
        Ignored for the same reason.

    Raises
    ------
    UnsupportedFormatError
        Always. An SVG is a drawing of a mesh's projection: the third
        coordinate, the element types and every attribute are gone from it,
        so there is no mesh in the file to read back.
    """
    raise UnsupportedFormatError(
        f"'{source_name(path)}': SVG is write-only. The file is a drawing of "
        "the mesh projected onto a plane, with no third coordinate, element "
        "type or attribute in it to read a mesh back from. Keep the mesh in a "
        "mesh format and write the picture from that."
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Draw the mesh as an SVG picture.

    Parameters
    ----------
    poly
        PolyData to draw.
    path
        Output file path or open file object.
    **opts
        ``plane`` picks the projection: ``"xy"``, ``"xz"`` or ``"yz"``, the
        first axis running rightward and the second upward. ``float_fmt``
        is the format specifier for every coordinate. ``stroke_width``
        sets the line weight in the mesh's own units. ``width`` sets the
        picture's rendered width in CSS pixels, the height following from
        the mesh's aspect ratio; without it the picture takes the size of
        whatever displays it.

    Raises
    ------
    CodecError
        If ``plane`` names no projection, ``float_fmt`` is not a usable
        format specifier or pads with a character no XML attribute may
        hold, ``stroke_width`` or ``width`` is not a positive number, a
        vertex coordinate is not finite, or an element references a vertex
        the mesh does not have.

    Notes
    -----
    Coordinates are written in the mesh's own units, shifted so the picture's
    top-left corner is the origin: SVG's y axis runs downward, so the
    upward axis is mirrored to keep the drawing the right way up. The
    ``viewBox`` names the mesh's extent, padded by half a stroke so an
    outline on the boundary is not clipped.

    Lines and faces are drawn: a line, polyline, triangle, quadrilateral,
    pixel, polygon or triangle strip, and the higher-order kinds of each
    through their corner nodes. Every other element type - a vertex, which
    has no outline, and a solid cell, which has no place in the plane - is
    skipped with a warning, one per type; so is an element of a free-size
    type holding too few nodes to outline. Each drawn type becomes one
    ``<path>`` carrying the type's name as its ``class``, with the elements
    as subpaths in mesh order, so a stylesheet can colour triangles apart
    from quadrilaterals. Attributes, tags and global attributes have no
    spelling in a picture and are not written.

    A coordinate the projection drops is expected to be constant across the
    mesh; when it is not, the picture is still written and a warning says
    that it is a projection.
    """
    plane = opts.pop("plane", "xy")
    float_fmt = opts.pop("float_fmt", _DEFAULT_FLOAT_FMT)
    stroke_width = opts.pop("stroke_width", None)
    width = opts.pop("width", None)
    if opts:
        warnings.warn(
            f".svg write: unrecognized options {set(opts)}; ignored.",
            stacklevel=3,  # user -> polyxios.write -> here
        )
    columns = _PLANES.get(plane if isinstance(plane, str) else None)
    if columns is None:
        raise CodecError(
            f".svg: plane {plane!r} is not one of {', '.join(sorted(_PLANES))}."
        )
    _check_float_fmt(float_fmt)
    stroke_width = _positive_or_none("stroke_width", stroke_width)
    width = _positive_or_none("width", width)

    name = source_name(path)
    right, up, dropped = columns
    verts = np.asarray(poly.vertices, dtype=np.float64)
    if not np.isfinite(verts).all():
        raise CodecError(
            f"'{name}': a vertex coordinate is not finite; a picture has no "
            "place for it and its extent would be undefined."
        )
    x, y, extent = _project(verts, right, up)
    _warn_if_not_flat(verts, dropped, extent, name)

    if stroke_width is None:
        stroke_width = _default_stroke(extent)
    pad = stroke_width / 2.0

    points = [
        f"{format(px, float_fmt)} {format(py, float_fmt)}"
        for px, py in zip(x.tolist(), y.tolist(), strict=True)
    ]
    paths = _paths(poly, points, name)

    view_box = " ".join(
        format(v, float_fmt)
        for v in (-pad, -pad, extent[0] + 2.0 * pad, extent[1] + 2.0 * pad)
    )
    size = ""
    if width is not None:
        height = width * (extent[1] + 2.0 * pad) / (extent[0] + 2.0 * pad)
        size = (
            f' width="{format(width, float_fmt)}" height="{format(height, float_fmt)}"'
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="{_XMLNS}" version="1.1" viewBox="{view_box}"{size}>',
        f'<g fill="none" stroke="black" stroke-width="{format(stroke_width, float_fmt)}"'
        ' stroke-linejoin="round" stroke-linecap="round">',
    ]
    lines.extend(f'<path class="{type_name}" d="{d}"/>' for type_name, d in paths)
    lines.append("</g>")
    lines.append("</svg>")
    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def _check_float_fmt(float_fmt: Any) -> None:
    """Raise unless ``float_fmt`` formats a float into text safe in an attribute.

    Zero is the shortest rendering under every float presentation type, so
    a fill wide enough to show on any coordinate shows on it; probing that
    one value is enough to catch a fill character XML would choke on.
    """
    try:
        sample = format(0.0, float_fmt)
    except (TypeError, ValueError) as exc:
        raise CodecError(f".svg: float_fmt {float_fmt!r} is not usable.") from exc
    unsafe = _UNSAFE_IN_ATTRIBUTE.intersection(sample)
    if unsafe:
        raise CodecError(
            f".svg: float_fmt {float_fmt!r} pads numbers with "
            f"{''.join(sorted(unsafe))!r}, which no XML attribute may hold."
        )


def _positive_or_none(option: str, value: Any) -> float | None:
    """Return ``value`` as a float, or raise if it is not a positive number."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise CodecError(f".svg: {option} must be a positive number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CodecError(
            f".svg: {option} must be a positive number, got {value!r}."
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise CodecError(f".svg: {option} must be a positive number, got {value!r}.")
    return number


def _project(
    verts: np.ndarray, right: int, up: int
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Return picture coordinates and the picture's extent.

    The rightward axis is shifted to start at zero; the upward one is
    mirrored, since SVG's y grows downward, and shifted the same way.
    """
    if verts.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, (0.0, 0.0)
    x = verts[:, right]
    y = verts[:, up]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    return x - x_min, y_max - y, (x_max - x_min, y_max - y_min)


def _warn_if_not_flat(
    verts: np.ndarray, dropped: int, extent: tuple[float, float], name: str
) -> None:
    if verts.shape[0] == 0:
        return
    column = verts[:, dropped]
    low, high = float(column.min()), float(column.max())
    span = high - low
    if span <= _FLAT_RTOL * max(extent[0], extent[1], span):
        return
    warnings.warn(
        f"'{name}': the mesh is not flat in the picture's plane - {_AXES[dropped]} "
        f"runs from {low:g} to {high:g} - so the drawing is its projection.",
        stacklevel=4,  # user -> polyxios.write -> write -> here
    )


def _default_stroke(extent: tuple[float, float]) -> float:
    """A hundredth of the shorter side, of the longer when one side is flat."""
    sides = [side for side in extent if side > 0.0]
    if not sides:
        return _UNIT_STROKE
    return min(sides) * _STROKE_FRACTION


def _paths(poly: PolyData, points: list[str], name: str) -> list[tuple[str, str]]:
    """Return one ``(type name, path data)`` per drawn type, in mesh order.

    ``points`` holds each vertex already spelled as ``"x y"``, so an element
    costs one list slice and one join; the connectivity and offsets are
    turned into lists once, since a numpy slice per element would dominate
    a large mesh.
    """
    types = np.asarray(poly.element_types)
    if types.size == 0:
        return []
    conn = np.asarray(poly.connectivity)
    n_verts = len(points)
    if conn.size and (int(conn.min()) < 0 or int(conn.max()) >= n_verts):
        raise CodecError(
            f"'{name}': an element references a vertex outside 0..{n_verts - 1}."
        )
    offsets = np.asarray(poly.offsets).tolist()
    conn_list = conn.tolist()

    codes, first = np.unique(types, return_index=True)
    out: list[tuple[str, str]] = []
    for code in codes[np.argsort(first)].tolist():
        members = np.flatnonzero(types == code).tolist()
        type_name = ELEMENT_TYPES_INV.get(code, f"type {code}")
        if code not in _DRAWABLE:
            warnings.warn(
                f"'{name}': SVG draws lines and faces only; {len(members)} "
                f"{type_name} element(s) skipped.",
                stacklevel=4,  # user -> polyxios.write -> write -> here
            )
            continue
        needed = _nodes_needed(code)
        subpaths: list[str] = []
        too_small = 0
        for i in members:
            nodes = conn_list[offsets[i] : offsets[i + 1]]
            if len(nodes) < needed:
                too_small += 1
                continue
            subpaths.extend(_rings(code, nodes, points))
        if too_small:
            warnings.warn(
                f"'{name}': {too_small} {type_name} element(s) with fewer than "
                f"{needed} nodes skipped; there is no outline to draw through them.",
                stacklevel=4,  # user -> polyxios.write -> write -> here
            )
        if subpaths:
            out.append((type_name, " ".join(subpaths)))
    return out


def _nodes_needed(code: int) -> int:
    """How many nodes an element of this type has to hold to be drawn."""
    if code == _TRIANGLE_STRIP:
        return _MIN_STRIP_NODES
    if code == _QUADRATIC_POLYGON:
        return _MIN_QUADRATIC_POLYGON_NODES
    corners, _ = _CORNERS[code]
    return _MIN_DRAWN_NODES if corners is None else max(corners) + 1


def _rings(code: int, nodes: list[int], points: list[str]) -> list[str]:
    """Spell one element as its subpaths."""
    if code == _TRIANGLE_STRIP:
        return [
            _ring(nodes[i : i + 3], points, closed=True) for i in range(len(nodes) - 2)
        ]
    if code == _QUADRATIC_POLYGON:
        return [_ring(nodes[: len(nodes) // 2], points, closed=True)]
    corners, closed = _CORNERS[code]
    ring = nodes if corners is None else [nodes[c] for c in corners]
    return [_ring(ring, points, closed=closed)]


def _ring(nodes: list[int], points: list[str], *, closed: bool) -> str:
    path = " L ".join([points[n] for n in nodes])
    return f"M {path} Z" if closed else f"M {path}"
