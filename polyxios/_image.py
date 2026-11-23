"""Uniform grids as data: :class:`ImageData`, and the arrays that become one.

An ImageData is VTK's model of a regular lattice: an extent of point
indices, an origin, a spacing, optionally a rotation, and arrays laid over
the points and the cells. It is the compact form of everything a raster
file or a ``.vti`` holds - ``n`` pixels cost ``n`` samples, where the
:class:`~polyxios.PolyData` they expand into costs three coordinates and a
cell for each of them. :meth:`ImageData.to_polydata` and
:meth:`ImageData.from_polydata` go between the two.

:func:`raster_pixels` and :func:`warn_left_out` are the shared half of the
raster codecs; the PNG writer calls both.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import numbers
from typing import Any, Self

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, MAX_IMPLIED_VERTICES
from polyxios._scene import _same_value
from polyxios._tags import tags_from_masks, with_tag_masks
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.codecs._vtk_xml import (
    extent_points,
    extent_spans,
    grid_axes,
    lattice_vertices,
    require_grid_order,
    require_structured_cells,
    structured_cell_shape,
    structured_cells,
    structured_cells_fit,
    structured_offsets,
    structured_spans_from_cells,
    vertices_match_axes,
    whole_extent,
)
from polyxios.exceptions import CodecError, ValidationError
from polyxios.validate import validate_header

# Names the pixel values travel under, the ones napari, PCD and XYZ use.
COLORS: str = "colors"
INTENSITY: str = "intensity"
ALPHA: str = "alpha"

_MODES: dict[int, str] = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}

# The two global_attrs keys the VTKHDF codec spells as the file's own structure:
# the VTK type a file holds, and the rotation a reader hands back as ``direction``.
# Defined here, once, so the codec and every reader that names them agree.
TYPE_KEY: str = "vtkhdf_type"
DIRECTION_KEY: str = "vtkhdf_direction"

_IDENTITY: np.ndarray = np.eye(3).ravel()


def _rotation_values(direction: np.ndarray | None) -> np.ndarray:
    """The nine numbers a direction spells, row-major; the identity's for None."""
    if direction is None:
        return _IDENTITY
    return np.asarray(direction, dtype=np.float64).ravel()


_INT64 = np.iinfo(np.int64)


def _refuse_complex(value: Any, what: str) -> Any:
    """A value that is a complex number where a real one is meant, refused. NumPy's
    complex values are numbers.Complex and not numbers.Real; float() would keep the
    real part and drop the rest without a word."""
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        raise ValueError(
            f"ImageData: {what} holds {value!r}, which is complex, not a real number."
        )
    return value


def _whole_extent_of(extent: Any) -> tuple[int, ...]:
    if isinstance(extent, (str, bytes)):
        raise ValueError(
            f"ImageData: extent is not a run of six numbers ({extent!r} is text)."
        )
    try:
        values = tuple(extent)
    except TypeError as exc:
        raise ValueError("ImageData: extent is not a run of six numbers.") from exc
    if len(values) != 6:
        raise ValueError(
            f"ImageData: extent holds {len(values)} indices; an extent holds six."
        )
    out: list[int] = []
    for value in values:
        if isinstance(value, (str, bytes)):
            raise ValueError(
                f"ImageData: extent {values} holds {value!r}, which is text, not a number."
            )
        value = _refuse_complex(value, f"extent {values}")
        if isinstance(value, numbers.Integral):
            whole = int(value)  # exact: a float would round one past 2**53
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ImageData: extent {values} holds {value!r}, which is not a number."
                ) from exc
            if not number.is_integer():
                raise ValueError(
                    f"ImageData: extent {values} holds {value!r}, which is not a whole "
                    "number."
                )
            whole = int(number)
        if not _INT64.min <= whole <= _INT64.max:
            raise ValueError(
                f"ImageData: extent {values} holds {value!r}, which is outside the "
                "64-bit integers a file spells."
            )
        out.append(whole)
    return tuple(out)


def _as_array(value: Any, attr: str, key: str) -> np.ndarray:
    """One attribute as an array; a value NumPy cannot make one of is refused by name,
    where it is given, rather than by NumPy's own message."""
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ImageData: {attr} '{key}' is not an array ({exc}).") from exc


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    """An ImageData attribute map as a dict: None is no attributes, and anything that
    is not a mapping is refused where it is given rather than in a writer."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"ImageData: {name} is a {type(value).__name__}, not a mapping."
        )
    return dict(value)


def _run_of_numbers(values: Any, name: str) -> tuple[Any, ...]:
    """The items of a run of numbers, text refused: a string parses as a number, and
    iterates as characters, and neither is a number a grid can mean."""
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"ImageData: {name} is not a run of numbers ({values!r} is text)."
        )
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"ImageData: {name} is not a run of numbers ({exc}).") from exc
    for item in items:
        if isinstance(item, (str, bytes)):
            raise ValueError(
                f"ImageData: {name} is not a run of numbers ({item!r} is text)."
            )
        _refuse_complex(item, name)
    return items


def _float_triple(values: Any, *, name: str) -> tuple[float, float, float]:
    items = _run_of_numbers(values, name)
    try:
        out = tuple(float(v) for v in items)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ImageData: {name} is not a run of numbers ({exc}).") from exc
    if len(out) != 3:
        raise ValueError(
            f"ImageData: {name} holds {len(out)} numbers; it holds three, one per axis."
        )
    return out


def _rotation_matrix(direction: Any) -> np.ndarray | None:
    """The ``(3, 3)`` float matrix a rotation spells, or None for no rotation.

    Text and complex numbers are refused: a string parses as a number, and a complex
    value would be cast to its real part without a word.
    """
    if direction is None:
        return None
    try:
        arr = np.asarray(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ImageData: direction is not a run of numbers ({exc})."
        ) from exc
    if arr.dtype.kind in "US":
        raise ValueError(
            "ImageData: direction is not a run of numbers (text is not a number)."
        )
    if arr.dtype.kind == "c":
        raise ValueError(
            "ImageData: direction holds complex numbers; a rotation is real."
        )
    if arr.dtype.kind == "O":
        # An object array holds whatever it was given. Each item meets the rule
        # origin and spacing meet: text and complex are refused, and so is anything
        # that is not a real number, such as None; a NaN is a real number, as it is
        # for origin and spacing.
        for item in _run_of_numbers(arr.ravel(), "direction"):
            if not isinstance(item, numbers.Real):
                raise ValueError(
                    f"ImageData: direction is not a run of numbers ({item!r} is not a "
                    "number)."
                )
    try:
        values = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ImageData: direction is not a run of numbers ({exc})."
        ) from exc
    if values.size != 9:
        raise ValueError(
            f"ImageData: direction holds {values.size} numbers; a rotation is a "
            "(3, 3) matrix of nine."
        )
    return values.reshape(3, 3)


def stored_rotation(value: Any, *, fmt: str) -> np.ndarray | None:
    """The rotation a mesh's stored ``vtkhdf_direction`` spells, or None for none.

    A stored value that is not a rotation - text, complex, or not nine numbers - is a
    :class:`~polyxios.exceptions.CodecError` naming ``fmt``, not NumPy's own message.
    """
    try:
        matrix = _rotation_matrix(value)
    except ValueError as exc:
        raise CodecError(
            f"{fmt}: global 'vtkhdf_direction' holds {value!r}, which is not a "
            f"rotation ({exc})."
        ) from exc
    return None if matrix is None or is_identity(matrix) else matrix


@dataclass(frozen=True, slots=True, eq=False)
class ImageData:
    """A uniform grid of points and cells carrying arrays.

    Parameters
    ----------
    extent
        VTK's ``(i0, i1, j0, j1, k0, k1)``: the first and last point index
        along each axis, inclusive. An axis whose end is before its start
        holds no points. Six whole numbers; a float holding one is taken as
        that integer, and any other is refused.
    origin
        Position of point ``(0, 0, 0)``: three numbers.
    spacing
        Step between points along each axis, three numbers; may be negative.
    direction
        ``(3, 3)`` matrix whose columns are the directions the axes run
        along, or None for the identity. A flat run of nine numbers is the
        same matrix, row-major, and is stored as ``(3, 3)``.
    point_attrs
        Arrays over the points, shaped ``(nz, ny, nx, *components)``: z
        slowest, x fastest, as in VTKHDF. A trailing axis of size one is not
        kept: an array shaped ``(nz, ny, nx, 1)`` reads back as
        ``(nz, ny, nx)``, in ``.vti`` and ``.vtkhdf`` alike.
    cell_attrs
        Arrays over the cells, shaped ``(max(cz, 1), max(cy, 1),
        max(cx, 1), *components)`` where ``cx, cy, cz`` are the spans
        ``i1 - i0`` and so on. An extent that spans no cell holds no cell
        array. As for ``point_attrs``, a trailing axis of size one reads back
        collapsed.
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "extent", _whole_extent_of(self.extent))
        object.__setattr__(self, "origin", _float_triple(self.origin, name="origin"))
        object.__setattr__(self, "spacing", _float_triple(self.spacing, name="spacing"))
        object.__setattr__(self, "direction", _rotation_matrix(self.direction))
        # Equality compares values, so a list and an array of the same numbers
        # must be the same array here: ndarrays are kept without a copy.
        for attr in ("point_attrs", "cell_attrs"):
            held = _as_mapping(getattr(self, attr), attr)
            object.__setattr__(
                self,
                attr,
                {k: _as_array(v, attr, k) for k, v in held.items()},
            )
        object.__setattr__(
            self, "global_attrs", _as_mapping(self.global_attrs, "global_attrs")
        )

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
        """Cells along ``(x, y, z)``: ``i1 - i0``, which is negative on an empty axis."""
        e = self.extent
        return e[1] - e[0], e[3] - e[2], e[5] - e[4]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageData):
            return NotImplemented
        if (
            self.extent != other.extent
            or self.origin != other.origin
            or self.spacing != other.spacing
        ):
            return False
        # An explicit identity and no rotation at all are the same grid.
        if not np.array_equal(
            _rotation_values(self.direction), _rotation_values(other.direction)
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
            one row per vertex or cell. The arrays are reshapes of this image's
            arrays: views wherever NumPy can make one, so writing to a view
            writes to the image.

        Raises
        ------
        ValueError
            If an attribute's leading shape is not the one the extent needs.
        ValidationError
            If the grid expands to more points than ``MAX_IMPLIED_VERTICES``, the
            cap a mesh is held to; a grid that size is read compact instead.
        """
        return lattice_mesh(
            self.extent,
            self.origin,
            self.spacing,
            self.direction,
            self.point_attrs,
            self.cell_attrs,
            self.global_attrs,
        )

    @classmethod
    def from_polydata(
        cls, poly: PolyData, *, fmt: str = "ImageData.from_polydata"
    ) -> Self:
        """Recover the grid a lattice mesh is the expansion of.

        Parameters
        ----------
        poly
            A mesh whose cells are a structured grid and whose vertices are
            that grid expanded, x fastest and z slowest. The ``vti_*`` keys a
            ``.vti`` read left in ``global_attrs`` say where the grid stood, and
            are used as the ``.vti`` writer uses them. A rotation a VTKHDF
            image was read with (``vtkhdf_direction``) is not applied to the
            grid, so such a mesh is refused: read the file with
            :func:`polyxios.read_image` instead.
        fmt
            Name the error messages carry, such as ``'.png'``.

        Returns
        -------
        ImageData
            The grid. Where the mesh is still the grid its ``vti_extent``
            describes, that extent, origin and step are kept; otherwise the
            grid is zero-based, with the origin and the step measured off
            the vertices and no rotation. ``vti_whole_extent`` is kept only when it
            is wider than that extent, and ``vti_origin``, ``vti_spacing`` and
            ``vti_extent`` are never field data.

        Raises
        ------
        TypeError
            If ``poly`` is not a :class:`~polyxios.PolyData`.
        CodecError
            If the mesh is not a uniform grid, it is a rotated VTKHDF image, or an
            attribute does not hold one row per point or cell of the grid.
        """
        if not isinstance(poly, PolyData):
            raise TypeError(f"{fmt} takes a PolyData, got {type(poly).__name__}.")
        stored = stored_rotation((poly.global_attrs or {}).get(DIRECTION_KEY), fmt=fmt)
        if stored is not None:
            raise CodecError(
                f"{fmt}: the mesh is a rotated VTKHDF image, and a mesh holds no "
                "rotation to give back; read the file with polyxios.read_image()."
            )
        extent, origin, spacing, kept = grid_of(poly, fmt=fmt)
        check_mesh_rows(poly, fmt=fmt)

        spans = (extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
        nx, ny, nz = (s + 1 for s in spans)
        ga = poly.global_attrs or {}
        n_verts = len(poly.vertices)
        n_cells = len(poly.element_types)

        points = with_tag_masks(
            {k: np.asarray(v) for k, v in poly.vertex_attrs.items()},
            poly.vertex_tags,
            n_verts,
            fmt=fmt,
            kind="point",
            verb="read",
        )
        point_lead = (nz, ny, nx)
        cell_lead = tuple(max(s, 1) for s in (spans[2], spans[1], spans[0]))
        cell_columns = with_tag_masks(
            {k: np.asarray(v) for k, v in poly.element_attrs.items()},
            poly.element_tags,
            n_cells,
            fmt=fmt,
            kind="cell",
            verb="read",
        )
        # With no cell there is nothing to reshape into: the columns are read for
        # the warnings their groups raise, and no cell array is kept.
        cells: dict[str, np.ndarray] = {}
        if n_cells > 0:
            cells = {
                name: arr.reshape(cell_lead + arr.shape[1:])
                for name, arr in cell_columns.items()
            }
        spelled = IMAGE_RESERVED_GLOBALS
        global_attrs = {k: v for k, v in ga.items() if k not in spelled}
        whole = wider_whole_extent(ga.get("vti_whole_extent"), extent) if kept else None
        if whole is not None:
            global_attrs["vti_whole_extent"] = whole
        try:
            return cls(
                extent=extent,  # type: ignore[arg-type]
                origin=tuple(origin),  # type: ignore[arg-type]
                spacing=tuple(spacing),  # type: ignore[arg-type]
                point_attrs={
                    name: arr.reshape(point_lead + arr.shape[1:])
                    for name, arr in points.items()
                },
                cell_attrs=cells,
                global_attrs=global_attrs,
            )
        except ValueError as exc:
            raise CodecError(f"{fmt}: {exc}") from exc


def lattice_mesh(
    extent: Sequence[int],
    origin: Sequence[float],
    spacing: Sequence[float],
    direction: np.ndarray | None,
    point_attrs: dict[str, np.ndarray],
    cell_attrs: dict[str, np.ndarray],
    global_attrs: dict[str, Any],
) -> PolyData:
    """Expand a grid's extent, origin, step and rotation into its lattice mesh.

    :meth:`ImageData.to_polydata` is this on an image's parts. A mesh read from a
    file takes this route directly, since an :class:`ImageData` refuses an extent a
    64-bit integer cannot hold, which a one-vertex mesh read from the same file has
    always had.
    """
    spans = (extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4])
    nx, ny, nz = (max(s + 1, 0) for s in spans)
    n_verts = nx * ny * nz
    if n_verts > MAX_IMPLIED_VERTICES:
        raise ValidationError(
            f"ImageData expands to {n_verts} points, over MAX_IMPLIED_VERTICES="
            f"{MAX_IMPLIED_VERTICES}; a mesh that size is not built. Read the grid "
            "with polyxios.read_image() instead."
        )
    if n_verts == 0:
        vertices = np.empty((0, 3), dtype=np.float64)
    elif direction is None:
        vertices = lattice_vertices(
            *[
                origin[a] + np.arange(extent[2 * a], extent[2 * a + 1] + 1) * spacing[a]
                for a in range(3)
            ]
        )
    else:
        vertices = lattice_vertices(
            *[
                np.arange(extent[2 * a], extent[2 * a + 1] + 1) * spacing[a]
                for a in range(3)
            ],
            direction=direction,
        )
        vertices += np.asarray(origin, dtype=np.float64)

    connectivity, _, _ = structured_cells(*spans)
    n_cells, per_cell, kind = structured_cell_shape(*spans)
    offsets = structured_offsets(n_cells, per_cell)
    element_types = np.full(n_cells, ELEMENT_TYPES[kind], dtype=np.uint8)

    point_lead = (nz, ny, nx)
    cell_lead = tuple(max(s, 1) for s in (spans[2], spans[1], spans[0]))
    vertex_attrs, vertex_tags = tags_from_masks(
        flatten_rows(point_attrs, point_lead, n_verts, "point")
    )
    element_attrs, element_tags = tags_from_masks(
        flatten_rows(cell_attrs, cell_lead, n_cells, "cell")
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
        global_attrs=dict(global_attrs),
    )


def check_readable(image: ImageData) -> None:
    """Refuse an image its readers would refuse, before a writer writes it.

    A compact read builds no vertex, so only the hard caps on the grid's counts
    apply to it; those are the caps checked here, so no file is written that
    cannot be read back.
    """
    nx, ny, nz = image.dimensions
    n_cells, per_cell, _ = structured_cell_shape(*image.spans)
    validate_header(
        nx * ny * nz,
        n_cells,
        n_cells * per_cell,
        0,
        compressed=True,
        spells_vertices=False,
        spells_connectivity=False,
        builds_vertices=False,
    )


def build_image(name: str, **parts: Any) -> ImageData:
    """Make the ImageData a file's parts describe; one the file cannot be is refused
    as a :class:`~polyxios.exceptions.CodecError` naming the file."""
    try:
        return ImageData(**parts)
    except ValueError as exc:
        raise CodecError(f"{name}: {exc}") from exc


def _row_count(arr: Any) -> int | None:
    """The rows an attribute holds, or None for one with no axis to count them on."""
    shape = np.shape(arr)
    return shape[0] if shape else None


def check_mesh_rows(poly: PolyData, *, fmt: str) -> None:
    """Refuse a mesh attribute that does not hold one row per point or cell, naming
    it, before any of them is reshaped into a grid. A value with no rows at all - a
    scalar, say - is refused the same way."""
    n_verts = len(poly.vertices)
    for name, arr in poly.vertex_attrs.items():
        rows = _row_count(arr)
        if rows != n_verts:
            held = "no rows" if rows is None else f"{rows} rows"
            raise CodecError(
                f"{fmt}: point attribute '{name}' holds {held}; "
                f"the grid has {n_verts} points."
            )
    n_cells = len(poly.element_types)
    for name, arr in poly.element_attrs.items():
        rows = _row_count(arr)
        if rows != n_cells:
            held = "no rows" if rows is None else f"{rows} rows"
            raise CodecError(
                f"{fmt}: cell attribute '{name}' holds {held}; "
                f"the grid has {n_cells} cells."
            )


def _laid_out(
    arr: np.ndarray, lead: tuple[int, ...], n: int, label: str, kind: str
) -> np.ndarray:
    """One array as one row per entity, or a ValueError if it lies off the extent.

    The one rule both the mesh and the file writers apply to an array laid out
    along the grid: its leading axes are ``lead``, and an extent with no entity
    holds no array with values.
    """
    if arr.shape[:3] != lead or (n == 0 and arr.size):
        held = f"holds no {kind}" if n == 0 else f"holds {n} {kind}s laid out {lead}"
        raise ValueError(f"{label} has shape {arr.shape}, but this extent {held}.")
    return arr.reshape((n,) + arr.shape[3:])


def flatten_rows(
    attrs: dict[str, Any], lead: tuple[int, ...], n: int, kind: str
) -> dict[str, np.ndarray]:
    """Fold grid-shaped arrays into one row per point or cell, or refuse them.

    Parameters
    ----------
    attrs
        The arrays as an ImageData holds them, each laid out ``lead`` first.
    lead
        The leading shape every array must have: ``(nz, ny, nx)`` for points.
    n
        How many rows the extent holds.
    kind
        ``'point'`` or ``'cell'``, for the message.

    Returns
    -------
    dict of str to numpy.ndarray
        ``(n, *components)`` views of the arrays, in the order they were held.

    Raises
    ------
    ValueError
        If an array is not laid out along the extent.
    """
    return {
        name: _laid_out(np.asarray(arr), lead, n, f"{kind} attribute '{name}'", kind)
        for name, arr in attrs.items()
    }


def vtk_lattice(lead: tuple[int, ...], points: tuple[int, ...]) -> tuple[int, ...]:
    """The shape VTKHDF stores an ImageData array in, given its own lattice.

    VTK's writer leaves out each axis that holds one point. So ``lead`` - the
    array's ``(nz, ny, nx)``-shaped lattice, the points' for a point array and the
    cells' for a cell array - keeps only the axes along which ``points`` counts
    more than one. An array over a single point is one row, which VTK spells
    ``(1,)``.
    """
    return tuple(c for c, p in zip(lead, points) if p > 1) or (1,)


def grid_rows(
    attrs: dict[str, Any], lead: tuple[int, ...], n: int, *, fmt: str, group: str
) -> dict[str, np.ndarray]:
    """Lay the arrays of one ImageData group out one row per point or cell.

    The rule the file writers share: an array that is not numeric, has no
    component, or sits on an extent with no entity to hold it, is dropped by
    name with one warning; a numeric array not laid out along the extent is
    refused.

    Parameters
    ----------
    attrs
        The arrays as the ImageData holds them.
    lead
        The leading shape every array must have: ``(nz, ny, nx)`` for points.
    n
        How many rows the extent holds.
    fmt
        The format's extension, for the messages.
    group
        ``PointData`` or ``CellData``, for the messages.

    Returns
    -------
    dict of str to numpy.ndarray
        The kept arrays as ``(n, *components)``, in the order they were held.

    Raises
    ------
    CodecError
        If a numeric array is not laid out along the extent.

    Warns
    -----
    UserWarning
        Once, naming the arrays dropped, blamed on the caller.
    """
    kind = "point" if group == "PointData" else "cell"
    out: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for key, arr in attrs.items():
        arr = np.asarray(arr)
        if arr.dtype.kind not in "biuf" or n == 0 or 0 in arr.shape[3:]:
            dropped.append(str(key))
            continue
        try:
            out[str(key)] = _laid_out(arr, lead, n, f"{group} array '{key}'", kind)
        except ValueError as exc:
            raise CodecError(f"{fmt} write: {exc}") from exc
    if dropped:
        warn_caller(
            f"{fmt} write: {group} array(s) {dropped} are not numeric, hold no"
            " component, or the extent spans nothing to hold them; dropped."
        )
    return out


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
        attribute, a pixel attribute is not laid out along the extent, or
        its values do not fit the file's samples.

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
    if not is_identity(image.direction):
        raise CodecError(
            f"{fmt}: an image file has no rotation; this ImageData is rotated."
        )
    if nx == 0 or ny == 0 or nz == 0:
        raise CodecError(f"{fmt}: the image holds no pixels.")
    attrs = image.point_attrs
    colors = attrs.get(COLORS)
    intensity = attrs.get(INTENSITY)
    for name, arr in ((COLORS, colors), (INTENSITY, intensity)):
        if arr is not None and np.shape(arr)[:3] != (nz, ny, nx):
            raise CodecError(
                f"{fmt}: point attribute '{name}' is shaped {np.shape(arr)}, and "
                f"the extent lays its arrays out {(nz, ny, nx)}."
            )
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
    # Row 0 of the file is the top, which is the highest y unless the y step
    # is negative.
    if image.spacing[1] >= 0:
        plane = plane[::-1]
    mode = _MODES[plane.shape[2]]
    if mode == "L":
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


# ---------------------------------------------------------------------------
# The grid a lattice file spells, for the codecs that write one
# ---------------------------------------------------------------------------

# What the .vti codec spells from the grid itself, so it never travels as field
# data: the extent, the origin and step, and the whole grid a piece belongs to.
GRID_GLOBALS: frozenset[str] = frozenset(
    {"vti_origin", "vti_spacing", "vti_extent", "vti_whole_extent"}
)

# What the .vtkhdf codec spells as the file's own structure: the VTK type it
# holds, and the rotation a reader hands back as ``direction``.
STRUCTURE_GLOBALS: frozenset[str] = frozenset({TYPE_KEY, DIRECTION_KEY})

# The keys no image writer writes as field data, whichever codec it is.
IMAGE_RESERVED_GLOBALS: frozenset[str] = GRID_GLOBALS | STRUCTURE_GLOBALS


def is_identity(direction: Any) -> bool:
    """Say whether a rotation is no rotation: None, or the identity either way.

    The identity is spelled as a ``(3, 3)`` matrix or as the nine numbers of a
    flat one; both mean no rotation.
    """
    return bool(np.array_equal(_rotation_values(direction), _IDENTITY))


def triple_of(values: Any, *, default: float, what: str, fmt: str) -> list[float]:
    """Coerce a stored origin or spacing into exactly three floats.

    Parameters
    ----------
    values
        What the file or ``global_attrs`` held, or None when it held nothing.
        A bare number is read as the value of every axis, which is what a
        caller writing ``vti_spacing=0.5`` means by it.
    default
        The value each missing axis takes.
    what
        Name of the thing being read, for the warnings.
    fmt
        The format the messages name, such as ``'.vti'`` or the call that asked.

    Returns
    -------
    list of float
        Three floats. A shorter sequence is filled out with ``default`` rather
        than left short: an ``Origin`` of two numbers is not an origin, and
        zipping against it drops the third axis silently. A longer one keeps its
        first three - a mesh has three axes, and a fourth number describes
        nothing this codec can write.

    Warns
    -----
    UserWarning
        If the values are not numbers at all, or if there are any number of
        them but three. The default is taken in the first case; the warnings
        are blamed on the first frame outside polyxios.
    """
    if values is None:
        return [default] * 3
    try:
        if np.ndim(values) == 0:
            # A bare number, or the zero-dimensional array one becomes when it
            # travels through NumPy: neither can be sliced.
            return [float(values)] * 3
        out = [float(v) for v in values[:3]]
        n_values = len(values)
    except (TypeError, ValueError):
        # Not numbers and not one: a string, a run of names, something with no
        # length. Asked in a try rather than in an isinstance ladder because the
        # happy path pays nothing for it.
        warn_caller(
            f"{fmt}: {what} holds {values!r}, which is not a number or a run of"
            " them; the default is used instead."
        )
        return [default] * 3
    if n_values > 3:
        warn_caller(
            f"{fmt}: {what} spells {n_values} numbers where three belong; the rest"
            " name no axis and are dropped."
        )
    elif n_values < 3:
        warn_caller(
            f"{fmt}: {what} spells {n_values} numbers where three belong; the axes"
            f" it leaves out are given {default}, which the file never said."
        )
    out.extend([default] * (3 - len(out)))
    return out


def metadata_axes(
    origin: list[float], spacing: list[float], extent: Sequence[int]
) -> list[np.ndarray]:
    """Expand an origin, a step and an extent into the points they spell.

    The coordinates are built exactly as the .vti reader builds them, so a file
    that has not been touched compares equal to itself.
    """
    return [
        origin[axis]
        + np.arange(extent[2 * axis], extent[2 * axis + 1] + 1) * spacing[axis]
        for axis in range(3)
    ]


def wider_whole_extent(stored: Any, extent: Sequence[int]) -> list[int] | None:
    """The stored ``WholeExtent`` when it names a grid wider than the piece.

    An ImageData keeps its whole extent only when that grid contains the piece and
    is larger than it: a file whose ``WholeExtent`` is its ``Extent`` - the common
    case - carries nothing a grid of its own would not, and an image read from it
    should compare equal to the one written. A stored grid that does not contain
    the piece is not kept either.
    """
    whole = whole_extent(stored, extent)
    return None if whole == list(extent) else whole


def grid_of(
    poly: PolyData, *, fmt: str
) -> tuple[list[int], list[float], list[float], bool]:
    """The extent, origin and step a mesh's grid is written with.

    The extent, origin and step a file was read with travel with the mesh, so
    a grid that did not begin at zero - or that runs down an axis - goes back
    exactly where it stood. They are checked against the mesh rather than
    trusted, the way the entity ids are: a transform since the read leaves them
    describing the grid the mesh used to be. Pruning cells leaves the extent
    counting points that are not in the file, moving the mesh leaves the origin
    behind, scaling it leaves the step behind - and the last two write a
    well-formed file holding a grid somewhere else. Cheapest question first: the
    point count is three integers, the points are compared against broadcast
    views that allocate nothing the size of the mesh, and only a mesh that has
    answered both is walked cell by cell.

    Parameters
    ----------
    poly
        The mesh whose grid is wanted.
    fmt
        The format the messages name.

    Returns
    -------
    tuple
        ``extent``, ``origin`` and ``spacing``, and whether the stored extent
        was kept. Kept means the mesh is still the grid the stored metadata
        describes, so the stored origin, step and extent are those returned;
        otherwise the grid is re-derived from the cells, zero-based.

    Raises
    ------
    CodecError
        If the cells are not a uniform structured grid.
    """
    ga = poly.global_attrs or {}
    origin = triple_of(ga.get("vti_origin"), default=0.0, what="vti_origin", fmt=fmt)
    spacing = triple_of(ga.get("vti_spacing"), default=1.0, what="vti_spacing", fmt=fmt)
    stored = ga.get("vti_extent")
    spans = extent_spans(stored) if stored is not None else None
    stored_extent = [int(v) for v in stored] if spans is not None else None
    if (
        stored_extent is not None
        and spans is not None
        and extent_points(spans) == len(poly.vertices)
        and vertices_match_axes(
            poly.vertices, metadata_axes(origin, spacing, stored_extent)
        )
        and structured_cells_fit(poly.connectivity, poly.element_types, spans)
    ):
        return stored_extent, origin, spacing, True
    # What the mesh is now, read off the cells: they are the grid whatever the
    # coordinates do, and the extent has to be theirs or the cells the reader
    # rebuilds are not the ones written.
    spans = structured_spans_from_cells(
        poly.vertices, poly.connectivity, poly.element_types, fmt=fmt
    )
    require_structured_cells(poly.connectivity, poly.element_types, spans, fmt=fmt)
    axes = grid_axes(poly.vertices, spans)
    require_grid_order(poly.vertices, axes, fmt=fmt)
    spacing = uniform_steps(axes, spacing, fmt=fmt)
    origin = [float(u[0]) if len(u) else o for u, o in zip(axes, origin, strict=True)]
    # An axis with no vertex at all ends at -1, which is how VTK spells an empty
    # extent and how the .vts and .vtr writers here spell one; a 0 would declare
    # the one point an empty mesh does not hold.
    extent = [bound for span in spans for bound in (0, span)]
    return extent, origin, spacing, False


def _evenly_spaced(u: np.ndarray, step: float) -> bool:
    """Say whether one step reproduces every plane on an axis.

    The slack is measured against the step and against what a double holds at
    this magnitude - never as a fraction of the coordinate. A relative
    tolerance is a fraction of where the axis sits rather than of how far
    apart its planes are, so an axis at ``x = 1e6`` was allowed half a
    millimetre of drift per plane: a visibly uneven lattice passed, and came
    back regularised with every plane past the second moved. The two bounds
    answer the two ways a plane can miss: by a fraction of the step, and by
    the last few bits of a coordinate that has no room to hold it.

    Parameters
    ----------
    u
        The axis's coordinates, at least two of them.
    step
        The single step being proposed for it.

    Returns
    -------
    bool
        True when ``u[0] + i * step`` is every plane the axis holds.

    Examples
    --------
    >>> _evenly_spaced(np.array([0.0, 1.0, 2.0]), 1.0)
    True
    >>> _evenly_spaced(np.array([1e6, 1e6 + 1.0, 1e6 + 2.001]), 1.0005)
    False
    """
    rebuilt = float(u[0]) + np.arange(len(u)) * step
    # The axis is monotone in the step, so its widest value is an endpoint and
    # the bound costs two lookups rather than a pass over the whole axis.
    magnitude = max(abs(float(rebuilt[0])), abs(float(rebuilt[-1])))
    atol = max(abs(step) * 1e-9, 8.0 * float(np.spacing(magnitude)))
    return bool(np.allclose(u, rebuilt, rtol=0.0, atol=atol))


def uniform_steps(
    axes: Sequence[np.ndarray], spacing: list[float], *, fmt: str
) -> list[float]:
    """Measure the one step each axis of an ImageData is allowed.

    An ImageData holds no coordinates: it holds a step, and every plane on an
    axis is that step from the last. A lattice whose planes are not evenly
    spaced is a RectilinearGrid, and writing its first step as though it were
    the only one moves every point after the second - silently, since the
    file that comes back is a well-formed grid of the right size.

    Parameters
    ----------
    axes
        The x, y and z coordinates, as :func:`grid_axes` gives them.
    spacing
        The step to keep on an axis with no second plane to measure against.
    fmt
        Extension the message names.

    Returns
    -------
    list of float
        The step on each axis.

    Raises
    ------
    CodecError
        If an axis is not evenly spaced.
    """
    steps: list[float] = []
    for axis, (u, fallback) in enumerate(zip(axes, spacing, strict=True)):
        if len(u) < 2:
            # One plane, or none: there is no step to measure, and VTK reads
            # the same grid back whatever is written. Keep what came with the
            # mesh rather than inventing a number for it.
            steps.append(float(fallback))
            continue
        step = (float(u[-1]) - float(u[0])) / (len(u) - 1)
        if len(u) > 2 and not _evenly_spaced(u, step):
            raise CodecError(
                f"{fmt}: the planes on the {'xyz'[axis]} axis are not evenly "
                "spaced, and an ImageData holds one step per axis rather than "
                "a coordinate for each plane. Every plane past the second "
                "would move - write a .vtr instead, which spells its "
                "coordinates out."
            )
        steps.append(step)
    return steps
