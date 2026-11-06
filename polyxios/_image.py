"""Uniform grids as data: :class:`ImageData`, and the arrays that become one.

An ImageData is VTK's model of a regular lattice: an extent of point
indices, an origin, a spacing, optionally a rotation, and arrays laid over
the points and the cells. It is the compact form of everything a raster
file or a ``.vti`` holds - ``n`` pixels cost ``n`` samples, where the
:class:`~polyxios.PolyData` they expand into costs three coordinates and a
cell for each of them. :meth:`ImageData.to_polydata` and
:meth:`ImageData.from_polydata` go between the two.
"""

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._scene import _same_value
from polyxios._tags import tags_from_masks, with_tag_masks
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.codecs._vtk_xml import (
    grid_axes,
    lattice_vertices,
    require_grid_order,
    require_structured_cells,
    structured_cell_shape,
    structured_cells,
    structured_offsets,
    structured_spans_from_cells,
    uniform_steps,
)
from polyxios.exceptions import CodecError

# Names the pixel values travel under, the ones napari, PCD and XYZ use.
COLORS: str = "colors"
INTENSITY: str = "intensity"
ALPHA: str = "alpha"

_MODES: dict[int, str] = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}


@dataclass(frozen=True, slots=True, eq=False)
class ImageData:
    """A uniform grid of points and cells carrying arrays.

    Parameters
    ----------
    extent
        VTK's ``(i0, i1, j0, j1, k0, k1)``: the first and last point index
        along each axis, inclusive. An axis whose end is before its start
        holds no points.
    origin
        Position of point ``(0, 0, 0)``.
    spacing
        Step between points along each axis; may be negative.
    direction
        ``(3, 3)`` matrix whose columns are the directions the axes run
        along, or None for the identity.
    point_attrs
        Arrays over the points, shaped ``(nz, ny, nx, *components)``: z
        slowest, x fastest, as in VTKHDF.
    cell_attrs
        Arrays over the cells, shaped ``(max(cz, 1), max(cy, 1),
        max(cx, 1), *components)`` where ``cx, cy, cz`` are the spans
        ``i1 - i0`` and so on.
    global_attrs
        Whole-image metadata.
    """

    extent: tuple[int, int, int, int, int, int]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    direction: np.ndarray | None = None
    point_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    cell_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    global_attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Points along ``(x, y, z)``."""
        e = self.extent
        return (
            max(e[1] - e[0] + 1, 0),
            max(e[3] - e[2] + 1, 0),
            max(e[5] - e[4] + 1, 0),
        )

    @property
    def spans(self) -> tuple[int, int, int]:
        """Cells along ``(x, y, z)``: ``i1 - i0``, which is -1 on an empty axis."""
        e = self.extent
        return e[1] - e[0], e[3] - e[2], e[5] - e[4]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageData):
            return NotImplemented
        if (
            tuple(self.extent) != tuple(other.extent)
            or tuple(self.origin) != tuple(other.origin)
            or tuple(self.spacing) != tuple(other.spacing)
        ):
            return False
        if (self.direction is None) != (other.direction is None):
            return False
        if self.direction is not None and not np.array_equal(
            self.direction, other.direction
        ):
            return False
        return (
            _same_value(self.point_attrs, other.point_attrs)
            and _same_value(self.cell_attrs, other.cell_attrs)
            and _same_value(self.global_attrs, other.global_attrs)
        )

    def to_polydata(self) -> PolyData:
        """Expand the grid into a :class:`~polyxios.PolyData` lattice.

        Returns
        -------
        PolyData
            One vertex per point and one cell per grid cell - quads for a
            flat image, hexahedra for a volume - with the arrays flattened to
            one row per vertex or cell.

        Raises
        ------
        ValueError
            If an attribute's leading shape is not the one the extent needs.
        """
        spans = self.spans
        n_cells, per_cell, kind = structured_cell_shape(*spans)
        nx, ny, nz = self.dimensions
        n_verts = nx * ny * nz
        if n_verts == 0:
            vertices = np.empty((0, 3), dtype=np.float64)
        else:
            e = self.extent
            if self.direction is None:
                vertices = lattice_vertices(
                    *[
                        self.origin[a]
                        + np.arange(e[2 * a], e[2 * a + 1] + 1) * self.spacing[a]
                        for a in range(3)
                    ]
                )
            else:
                vertices = lattice_vertices(
                    *[
                        np.arange(e[2 * a], e[2 * a + 1] + 1) * self.spacing[a]
                        for a in range(3)
                    ],
                    direction=self.direction,
                )
                vertices += np.asarray(self.origin, dtype=np.float64)

        connectivity, _, _ = structured_cells(*spans)
        offsets = structured_offsets(n_cells, per_cell)
        element_types = np.full(n_cells, ELEMENT_TYPES[kind], dtype=np.uint8)

        point_lead = (nz, ny, nx)
        cell_lead = tuple(max(s, 1) for s in (spans[2], spans[1], spans[0]))
        vertex_attrs, vertex_tags = tags_from_masks(
            _flatten(self.point_attrs, point_lead, n_verts, "point")
        )
        element_attrs, element_tags = tags_from_masks(
            _flatten(self.cell_attrs, cell_lead, n_cells, "cell")
        )
        return PolyData(
            vertices=vertices,
            connectivity=connectivity,
            offsets=offsets,
            element_types=element_types,
            vertex_attrs=vertex_attrs,
            element_attrs=element_attrs,
            vertex_tags=vertex_tags,
            element_tags=element_tags,
            global_attrs=dict(self.global_attrs),
        )

    @classmethod
    def from_polydata(cls, poly: PolyData, *, fmt: str) -> Self:
        """Recover the grid a lattice mesh is the expansion of.

        Parameters
        ----------
        poly
            A mesh whose cells are a structured grid and whose vertices are
            that grid expanded, x fastest and z slowest.
        fmt
            Name the error messages carry, such as ``'.png'``.

        Returns
        -------
        ImageData
            The grid, zero-based, with the origin and the step measured off
            the vertices and no rotation.

        Raises
        ------
        CodecError
            If the mesh is not a uniform grid.
        """
        spans = structured_spans_from_cells(
            poly.vertices, poly.connectivity, poly.element_types, fmt=fmt
        )
        require_structured_cells(poly.connectivity, poly.element_types, spans, fmt=fmt)
        axes = grid_axes(poly.vertices, spans)
        require_grid_order(poly.vertices, axes, fmt=fmt)
        spacing = uniform_steps(axes, [1.0, 1.0, 1.0], fmt=fmt)
        origin = [float(u[0]) if len(u) else 0.0 for u in axes]
        # An axis with no vertex ends at -1, which is how VTK spells it.
        extent = tuple(bound for span in spans for bound in (0, span))
        image = cls(
            extent=extent,  # type: ignore[arg-type]
            origin=tuple(origin),  # type: ignore[arg-type]
            spacing=tuple(spacing),  # type: ignore[arg-type]
        )
        nx, ny, nz = image.dimensions
        n_verts = len(poly.vertices)
        n_cells = len(poly.element_types)

        points = with_tag_masks(
            poly.vertex_attrs, poly.vertex_tags, n_verts, fmt=fmt, kind="point"
        )
        point_lead = (nz, ny, nx)
        cell_lead = tuple(max(s, 1) for s in (spans[2], spans[1], spans[0]))
        cells: dict[str, np.ndarray] = {}
        if n_cells > 0:
            cells = {
                name: arr.reshape(cell_lead + arr.shape[1:])
                for name, arr in with_tag_masks(
                    poly.element_attrs,
                    poly.element_tags,
                    n_cells,
                    fmt=fmt,
                    kind="cell",
                ).items()
            }
        # Imported here: the codec imports this module for the type.
        from polyxios.codecs._vti import RESERVED_GLOBALS

        spelled = RESERVED_GLOBALS | {"vtkhdf_direction", "vtkhdf_type"}
        return cls(
            extent=extent,  # type: ignore[arg-type]
            origin=image.origin,
            spacing=image.spacing,
            point_attrs={
                name: arr.reshape(point_lead + arr.shape[1:])
                for name, arr in points.items()
            },
            cell_attrs=cells,
            global_attrs={
                k: v for k, v in (poly.global_attrs or {}).items() if k not in spelled
            },
        )


def _flatten(
    attrs: dict[str, np.ndarray], lead: tuple[int, ...], n: int, kind: str
) -> dict[str, np.ndarray]:
    """Fold grid-shaped arrays into one row per point or cell."""
    out: dict[str, np.ndarray] = {}
    for name, arr in attrs.items():
        arr = np.asarray(arr)
        if arr.shape[:3] != lead or (n == 0 and arr.size):
            raise ValueError(
                f"ImageData: {kind} attribute '{name}' has shape {arr.shape}, "
                f"but this extent holds {n} {kind}s laid out {lead}."
            )
        out[name] = arr.reshape((n,) + arr.shape[3:])
    return out


def image_to_array(
    image: ImageData | PolyData, *, name: str | None = None, origin: str = "upper"
) -> np.ndarray:
    """Return the pixel array an image holds.

    Parameters
    ----------
    image
        An :class:`ImageData`, or a lattice :class:`~polyxios.PolyData` such
        as ``read()`` gives for a raster file.
    name
        Point attribute to return. By default ``'colors'``, else
        ``'intensity'`` (with ``'alpha'`` stacked on a last axis when there
        is one), else the only point attribute.
    origin
        ``'upper'`` puts row 0 at the top of the picture, the way image
        files and arrays order them; ``'lower'`` keeps VTK's order, with y
        up.

    Returns
    -------
    numpy.ndarray
        ``(H, W[, C])`` for a flat image, ``(D, H, W[, C])`` for a volume.

    Raises
    ------
    ValueError
        If ``name`` or ``origin`` is not usable, or the choice of attribute
        is ambiguous.
    """
    if origin not in ("upper", "lower"):
        raise ValueError(
            f"image_to_array: origin is 'upper' or 'lower', not {origin!r}."
        )
    if isinstance(image, PolyData):
        image = ImageData.from_polydata(image, fmt="image_to_array")
    attrs = image.point_attrs
    if name is not None:
        if name not in attrs:
            raise ValueError(
                f"image_to_array: no point attribute {name!r}; "
                f"the image holds {sorted(attrs)}."
            )
        arr = np.asarray(attrs[name])
    elif COLORS in attrs:
        arr = np.asarray(attrs[COLORS])
    elif INTENSITY in attrs:
        arr = np.asarray(attrs[INTENSITY])
        if ALPHA in attrs:
            channels = [
                a.reshape(a.shape[:3] + (-1,)) for a in (arr, np.asarray(attrs[ALPHA]))
            ]
            arr = np.concatenate(channels, axis=-1)
    elif len(attrs) == 1:
        arr = np.asarray(next(iter(attrs.values())))
    elif attrs:
        raise ValueError(f"image_to_array: name one of {sorted(attrs)}.")
    else:
        raise ValueError("image_to_array: the image holds no point attributes.")
    if origin == "upper":
        arr = arr[:, ::-1]
    if arr.shape[0] == 1:
        arr = arr[0]
    return np.ascontiguousarray(arr)


def array_to_image(
    pixels: np.ndarray,
    *,
    origin: str = "upper",
    global_attrs: dict | None = None,
) -> ImageData:
    """Wrap a pixel array as a flat :class:`ImageData`.

    Parameters
    ----------
    pixels
        ``(H, W)`` or ``(H, W, C)`` with ``C`` in 1-4. One channel or two
        dimensions become ``'intensity'``, two channels ``'intensity'`` and
        ``'alpha'``, three or four ``'colors'``.
    origin
        ``'upper'`` if row 0 is the top of the picture, as in an image file;
        ``'lower'`` if it is the lowest y, VTK's order.
    global_attrs
        Whole-image metadata to carry.

    Returns
    -------
    ImageData
        One z plane, ``extent = (0, W-1, 0, H-1, 0, 0)``. The dtype is kept
        and the data copied.

    Raises
    ------
    ValueError
        If the array has another shape, or ``origin`` is not recognised.
    """
    if origin not in ("upper", "lower"):
        raise ValueError(
            f"array_to_image: origin is 'upper' or 'lower', not {origin!r}."
        )
    arr = np.asarray(pixels)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in _MODES):
        raise ValueError(
            "array_to_image: expects (H, W) or (H, W, C) with C in 1-4, "
            f"got shape {arr.shape}."
        )
    height, width = arr.shape[:2]
    if origin == "upper":
        arr = arr[::-1]
    arr = arr[np.newaxis].copy()
    channels = 1 if arr.ndim == 3 else arr.shape[3]
    if channels == 1:
        attrs = {INTENSITY: arr.reshape(arr.shape[:3])}
    elif channels == 2:
        attrs = {INTENSITY: arr[..., 0].copy(), ALPHA: arr[..., 1].copy()}
    else:
        attrs = {COLORS: arr}
    return ImageData(
        extent=(0, width - 1, 0, height - 1, 0, 0),
        point_attrs=attrs,
        global_attrs=dict(global_attrs or {}),
    )


def _int_range(arr: np.ndarray) -> tuple[Any, Any]:
    return (arr.min(), arr.max()) if arr.size else (0, 0)


def _samples(arr: np.ndarray, *, fmt: str, dtypes: frozenset[str]) -> np.ndarray:
    """Convert pixel values to the unsigned integers a file holds."""
    kind = arr.dtype.kind
    wide = "uint16" in dtypes
    if arr.dtype == np.uint8 or (arr.dtype == np.uint16 and wide):
        return arr
    if kind == "f":
        if arr.size and (
            not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 1.0
        ):
            raise CodecError(
                f"{fmt}: {arr.dtype} pixels must lie in [0, 1] to be scaled "
                f"to 0-255; got {arr.min()}..{arr.max()}."
            )
        return np.rint(arr * 255.0).astype(np.uint8)
    if kind in "iu":
        lo, hi = _int_range(arr)
        if lo >= 0 and hi <= 255:
            return arr.astype(np.uint8)
        if wide and lo >= 0 and hi <= 65535:
            return arr.astype(np.uint16)
        limit = "0-65535" if wide else "0-255"
        raise CodecError(
            f"{fmt}: {arr.dtype} pixels spanning {lo}..{hi} do not fit "
            f"{limit}; convert them to uint8 first."
        )
    raise CodecError(f"{fmt}: {arr.dtype} pixels cannot be written; use uint8.")


def raster_pixels(
    image: ImageData, *, fmt: str, dtypes: frozenset[str]
) -> tuple[np.ndarray, str]:
    """Return the pixels of a flat image as an image file holds them.

    Parameters
    ----------
    image
        The image to write.
    fmt
        Name the messages carry.
    dtypes
        The sample types the file can hold, from ``{'uint8', 'uint16'}``.

    Returns
    -------
    tuple[numpy.ndarray, str]
        ``(H, W[, C])`` C-contiguous pixels, top row first, and the mode:
        ``'L'``, ``'LA'``, ``'RGB'`` or ``'RGBA'``.

    Raises
    ------
    CodecError
        If the image is not one flat unrotated layer, holds no pixel
        attribute, or its values do not fit the file's samples.

    Warns
    -----
    UserWarning
        Once, naming what the image holds that the file has no place for.
    """
    nx, ny, nz = image.dimensions
    if nz > 1:
        raise CodecError(
            f"{fmt}: an image file holds one flat layer; this ImageData has "
            f"{nz} planes along z."
        )
    if image.direction is not None and not np.array_equal(image.direction, np.eye(3)):
        raise CodecError(
            f"{fmt}: an image file has no rotation; this ImageData is rotated."
        )
    if nx == 0 or ny == 0 or nz == 0:
        raise CodecError(f"{fmt}: the image holds no pixels.")
    attrs = image.point_attrs
    colors = attrs.get(COLORS)
    intensity = attrs.get(INTENSITY)
    used: set[str]
    if colors is not None and colors.ndim == 4 and colors.shape[3] in (3, 4):
        plane = np.asarray(colors)[0]
        used = {COLORS}
    elif intensity is not None and intensity.ndim in (3, 4):
        plane = np.asarray(intensity)[0].reshape(ny, nx, -1)
        used = {INTENSITY}
        alpha = attrs.get(ALPHA)
        if alpha is not None and alpha.shape[:3] == intensity.shape[:3]:
            plane = np.concatenate(
                [plane, np.asarray(alpha)[0].reshape(ny, nx, -1)], axis=-1
            )
            used.add(ALPHA)
        if plane.shape[2] not in (1, 2):
            raise CodecError(
                f"{fmt}: the 'intensity' attribute is {plane.shape[2]} wide; "
                "a gray image holds one sample per pixel."
            )
    else:
        raise CodecError(
            f"{fmt}: needs a 'colors' (3 or 4 wide) or 'intensity' point "
            f"attribute; it holds {sorted(attrs)}."
        )
    parts: list[str] = []
    others = sorted(set(attrs) - used)
    if others:
        parts.append(f"point attributes {others}")
    if image.cell_attrs:
        parts.append(f"cell attributes {sorted(image.cell_attrs)}")
    if any(v != 0 for v in image.origin):
        parts.append("origin")
    if any(abs(v) != 1 for v in image.spacing[:2]):
        parts.append("spacing")
    if parts:
        warn_caller(
            f"{fmt}: {', '.join(parts)} have no place in the file and are left out."
        )
    plane = _samples(plane, fmt=fmt, dtypes=dtypes)
    # Row 0 of the file is the top, which is the highest y unless the
    # spacing runs y downwards.
    if image.spacing[1] >= 0:
        plane = plane[::-1]
    mode = _MODES[plane.shape[2]]
    if plane.shape[2] == 1 and mode == "L":
        plane = plane[:, :, 0]
    return np.ascontiguousarray(plane), mode


def warn_left_out(global_attrs: dict, *, fmt: str, written: frozenset[str]) -> None:
    """Warn about the ``global_attrs`` keys a file has no place for.

    Parameters
    ----------
    global_attrs
        The image's whole-image metadata.
    fmt
        Name the message carries.
    written
        The keys the writer spells.
    """
    extra = sorted(k for k in global_attrs if k not in written)
    if extra:
        warn_caller(
            f"{fmt}: global_attrs {extra} have no place in the file and are left out."
        )
