"""One rule for meshes that came out of a two-dimensional file.

A ``PolyData`` holds three coordinate columns, always. A format that spells
two - a bamg ``.mesh``, an ``NDIME= 2`` SU2 case, a 2-D MFEM mesh - is padded
with ``z=0`` on the way in rather than kept narrow, so that every consumer can
index ``vertices[:, 2]`` without first asking what the file said. Padding on
its own is lossy in one direction: nothing downstream can tell a plane that
was written in two dimensions from one that was written in three and happens
to sit at ``z=0``, so a round trip through the same format would widen the
file. The reader therefore records the fact in
``global_attrs["was_2d"] = True`` and the writer reads it back.

The key is deliberately not format-prefixed the way ``tecplot_title`` or
``vtr_whole_extent`` are: it says something about the mesh, not about the file
it came from, so a 2-D SU2 mesh written out as MFEM stays two-dimensional.

Three rules, and every 2-D-capable codec follows them:

1. **Read pads.** Vertices are ``(n, 3)`` float64 whatever the file declared.
2. **Read remembers.** A file that declared two dimensions sets
   ``global_attrs["was_2d"] = True``. A three-dimensional file sets nothing:
   the key's absence means "not known to be two-dimensional", not "3-D".
3. **Write asks.** ``output_dimension`` decides how many columns go out.
   The flag wins when the mesh is still flat; coordinates win when it is not,
   since a third coordinate that reached the mesh after the read is data, and
   dropping it silently would be the worse loss.

A format with no two-dimensional spelling at all - Netgen writes ``mesh3d``,
OBJ and the VTK family always carry three - keeps writing three columns and
ignores the flag. It is recorded on the way in all the same, so that a mesh
read from a 2-D ``.vol`` and written as ``.su2`` still lands as ``NDIME= 2``.

Two columns sometimes constrain the rest of the file, and a writer keeps the
two in step rather than emitting one no reader loads. A flat mesh of solid
cells - a tetrahedron is one however flat it lies - keeps its third column
wherever the node count per element is declared apart from the coordinate
count, which :func:`has_solid_cells` is the test for; Abaqus goes further and
picks its element cards to match, a node's dimensionality there coming from
the element that references it.
"""

from typing import TYPE_CHECKING, Final
import warnings

import numpy as np

from polyxios._element_types import TOPOLOGICAL_DIMENSION
from polyxios.exceptions import CodecError

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from polyxios._types import PolyData

__all__ = [
    "WAS_2D_KEY",
    "has_solid_cells",
    "mark_2d",
    "output_dimension",
    "pad_to_3d",
    "was_2d",
]

#: Where a reader records that its file declared two dimensions.
WAS_2D_KEY: Final[str] = "was_2d"

# One entry per element type code, counting the three-dimensional ones. A
# bincount over the uint8 type column indexes straight into it, so the test
# below is one linear sweep with no array the length of the mesh behind it.
_SOLID_CODES: Final[np.ndarray] = np.array(
    sorted(code for code, dim in TOPOLOGICAL_DIMENSION.items() if dim == 3),
    dtype=np.intp,
)


def has_solid_cells(poly: "PolyData") -> bool:
    """Report whether the mesh holds a cell that cannot lie in a plane.

    Parameters
    ----------
    poly
        The mesh being written.

    Returns
    -------
    bool
        True when any element is three-dimensional. A tetrahedron is one
        however flat it lies, so a format that reads its node count per
        element from a keyword and its coordinate count from a header has to
        keep writing three columns for it whatever ``was_2d`` says.

    Notes
    -----
    Every element the mesh holds is counted, not only the ones a given writer
    has a spelling for. A codec that would skip a solid it cannot write then
    keeps a third column it did not need, which costs a column of zeros and
    never a coordinate.

    Examples
    --------
    >>> from polyxios import make_polydata
    >>> import numpy as np
    >>> flat = make_polydata(np.zeros((3, 3)), [("triangle", np.array([[0, 1, 2]]))])
    >>> has_solid_cells(flat)
    False
    """
    codes = np.asarray(poly.element_types)
    if codes.size == 0:
        return False
    counts = np.bincount(codes.ravel(), minlength=int(_SOLID_CODES[-1]) + 1)
    return bool(counts[_SOLID_CODES].any())


def mark_2d(dim: int) -> dict[str, bool]:
    """Return the ``global_attrs`` entries a file of this dimension needs.

    Parameters
    ----------
    dim
        The dimension the file declared, 2 or 3.

    Returns
    -------
    dict
        ``{"was_2d": True}`` for a two-dimensional file, an empty dict
        otherwise, so a reader can splat it into the mapping it is building
        without a branch of its own.

    Examples
    --------
    >>> mark_2d(2)
    {'was_2d': True}
    >>> mark_2d(3)
    {}
    """
    return {WAS_2D_KEY: True} if dim == 2 else {}


def was_2d(poly: "PolyData") -> bool:
    """Report whether a mesh is known to have come from a 2-D file.

    Parameters
    ----------
    poly
        The mesh to ask. Anything carrying a ``global_attrs`` mapping.

    Returns
    -------
    bool
        True only when the flag is present and truthy. A mesh that never
        went through a reader, or came from a three-dimensional file,
        answers False.
    """
    return bool((poly.global_attrs or {}).get(WAS_2D_KEY))


def pad_to_3d(values: np.ndarray, dim: int) -> np.ndarray:
    """Widen a coordinate block of ``dim`` columns to three.

    Parameters
    ----------
    values
        Coordinates, shape ``(n, dim)`` or wider; only the first ``dim``
        columns are read, so a record array holding a reference alongside
        the coordinates can be handed over whole.
    dim
        How many columns carry coordinates, 2 or 3.

    Returns
    -------
    numpy.ndarray
        A new ``(n, 3)`` float64 array, zero past ``dim``.

    Raises
    ------
    CodecError
        If ``dim`` is not 2 or 3, or the block has fewer columns than that.
        A codec error rather than a ValueError because the block comes off a
        file: every other malformed-input path in a codec raises this.
    """
    if dim not in (2, 3):
        raise CodecError(f"dimension must be 2 or 3, got {dim}.")
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] < dim:
        raise CodecError(
            f"expected an (n, {dim}) coordinate block, got {values.shape}."
        )
    out = np.zeros((values.shape[0], 3), dtype=np.float64)
    out[:, :dim] = values[:, :dim]
    return out


def output_dimension(
    poly: "PolyData",
    *,
    fmt: str,
    flat_default: int = 3,
    flat: bool | None = None,
    stacklevel: int = 3,
) -> int:
    """Return how many coordinate columns to write, 2 or 3.

    Parameters
    ----------
    poly
        The mesh being written.
    fmt
        The format's own name, ``".su2"`` and the like, so the warning a
        lifted mesh raises names the file it is about to land in.
    flat_default
        What a flat mesh that carries no ``was_2d`` flag is written as. 2 for
        a format whose writer already inferred the dimension from the
        coordinates - dropping that inference would turn a hand-built plane
        into a file the target solver refuses - and 3 for one that has always
        written three columns.
    flat
        Whether the mesh has stayed in the plane, for a codec whose own test
        is narrower than "no vertex carries a z" - WKT reads only the
        vertices an element reaches, and ignores a non-finite one. None asks
        the coordinates directly.
    stacklevel
        Passed to :func:`warnings.warn`. The default points at the caller of
        the codec's own ``write``, which is two frames above this helper.

    Returns
    -------
    int
        2 or 3. A mesh whose vertices carry only two columns answers 2
        whatever the flag says: the third coordinate is not there to write.

    Raises
    ------
    CodecError
        If the vertices are not a block of at least two coordinate columns.

    Warns
    -----
    UserWarning
        When the mesh is flagged two-dimensional but its vertices have since
        left the plane. The third coordinate is data and is written; the flag
        is what gives way.
    """
    vertices = poly.vertices
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise CodecError(
            f"expected an (n, 2) or wider coordinate block, got {vertices.shape}."
        )
    if vertices.shape[1] < 3:
        # Narrower than a PolyData holds - a caller built it by hand, or it
        # came from a release of the meshb reader that handed back (n, dim).
        # The columns it has are the whole dimension; there is no flag to ask.
        return 2
    if flat is None:
        # ``any`` on the z column alone: one pass, and an empty mesh answers
        # False without a length test of its own. A NaN z counts as non-flat -
        # it is a coordinate the mesh carries, and dropping the column would
        # lose it. A codec that reads its z otherwise - WKT ignores a
        # non-finite one - passes ``flat`` in rather than having it recomputed.
        flat = not vertices[:, 2].any()
    flagged = was_2d(poly)
    if not flat:
        if flagged:
            # Not an error: a transform lifting a plane out of its plane is a
            # normal thing to do, and the flag simply stopped being true.
            warnings.warn(
                f"{fmt}: the mesh was read from a two-dimensional file but its"
                " vertices now carry a third coordinate; it was written in"
                " three dimensions.",
                stacklevel=stacklevel,
            )
        return 3
    return 2 if flagged else flat_default
