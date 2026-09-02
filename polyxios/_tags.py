"""Turning a column of integer labels into named tag groups, and back.

Several formats label an element with a bare number - a Medit reference, a
Nastran property id - and a number that stays a column of integers does not
survive a conversion the way a named group does. The two directions live here
so the codecs that share the convention cannot drift apart on it.
"""

import re
from typing import NamedTuple
import warnings

import numpy as np

__all__ = [
    "MASK_PREFIX",
    "TagValues",
    "group_by_value",
    "integer_column",
    "mask_arrays",
    "member_indices",
    "member_values",
    "members_array",
    "tags_from_masks",
    "values_from_tags",
    "with_tag_masks",
]

# What a tag group is called when it travels as an ordinary data array. A
# format with a general attribute channel and no set of its own - the VTK
# family - can carry a group as one column of ones and zeros, and the only
# thing that tells such a column from an attribute is its name. The prefix is
# spelled out rather than short, because an attribute genuinely called this is
# read back as the group it claims to be.
MASK_PREFIX: str = "polyxios_tag_"


class TagValues(NamedTuple):
    """The labels a mesh's tag groups spell, and what could not be written.

    Attributes
    ----------
    values
        One label per element, zero where no group names one.
    unnamed
        The groups whose names spell no number, so a caller can report them.
    named
        Whether any group named a label that reached an element. A group
        whose members index nothing counts for no more here than one that
        was never there: writing its column would relabel every element 0.
    unusable
        The groups that reach no element of this mesh. Nothing checks a tag
        group on the way in, so a group may hold floats, which index nothing
        and are refused outright rather than rounded - rounding an index
        moves a label onto another element - or hold indices past the end of
        a mesh it was not built for.
    oversized
        The groups naming a number the column cannot hold. A name is free
        text, so it may spell one wider than the format's own field;
        assigning it raises an OverflowError from numpy, which says nothing
        about which group is at fault, so it is reported here instead.
    contested
        The groups that took an element another group had already labelled,
        so a caller can report the label that did not survive. A record
        carries one reference and the groups are walked in the order the mesh
        holds them, so the last to name an element is the one that keeps it -
        an order the caller never chose, which is why it is worth saying.

    Notes
    -----
    A named tuple rather than a bare one: six fields, four of them sets of
    names, is more than a positional unpack at a call site can be read back
    against.
    """

    values: np.ndarray
    unnamed: set[str]
    named: bool
    unusable: set[str]
    oversized: set[str]
    contested: set[str]


def group_by_value(values: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Return one tag group per distinct value, members ascending.

    Parameters
    ----------
    values
        One integer label per element.
    prefix
        Spelled ahead of the value in each group's name.

    Returns
    -------
    dict of str to numpy.ndarray
        ``prefix + value`` to the element indices carrying it, values
        ascending and members ascending inside a group.

    Notes
    -----
    One ``flatnonzero`` per distinct value rescans the whole column each time,
    which a mesh carrying thousands of labels feels; a single stable sort
    groups every value in one pass.
    """
    if values.size == 0:
        return {}
    order = np.argsort(values, kind="stable").astype(np.int32)
    ranked = values[order]
    starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
    return {
        f"{prefix}{int(value)}": members
        for value, members in zip(ranked[starts], np.split(order, starts[1:]))
    }


def members_array(members: object) -> np.ndarray | None:
    """Return a tag group's members as one array, or None when they are not one.

    Parameters
    ----------
    members
        A tag group's members, as the mesh carries them.

    Returns
    -------
    numpy.ndarray or None
        The members as an array, or None when numpy can build no array of
        them at all.

    Notes
    -----
    Nothing checks a tag group on the way in, so a group may hold a ragged
    list of lists, a dict, or an object with no array protocol. numpy refuses
    each of those rather than inventing a shape for them, and a bare
    ``ValueError`` out of the middle of a write names neither the group nor
    the file. Answering None is what lets every caller report the group the
    way it reports one holding floats or a stale index.

    Examples
    --------
    >>> members_array([0, 2]).tolist()
    [0, 2]
    >>> members_array([[0, 1], [2]]) is None
    True
    """
    try:
        return np.asarray(members)
    except (TypeError, ValueError):
        return None


def member_indices(members: object, n_elems: int) -> np.ndarray:
    """Return the element indices a tag group names, dropping what indexes none.

    Parameters
    ----------
    members
        A tag group's members, as the mesh carries them.
    n_elems
        How many elements the mesh holds.

    Returns
    -------
    numpy.ndarray
        The members that are indices into this mesh, as int64. Empty when the
        group holds no usable index at all.

    Notes
    -----
    Nothing checks a tag group on the way in, so a group may hold floats, an
    extra dimension, an index past the end of a mesh it was not built for, or
    a ragged list numpy builds no array of at all. Each of those indexes
    nothing: a float column is refused whole rather than rounded - rounding an
    index moves a label onto another element - and a stale index reaches an
    element that is not the one it named. Dropping them is what lets a writer
    report the loss instead of raising whatever the first stray value happens
    to raise.
    """
    held = members_array(members)
    if held is None:
        return np.empty(0, dtype=np.int64)
    picked = held.ravel()
    if picked.dtype.kind not in "iub":
        return np.empty(0, dtype=np.int64)
    picked = picked.astype(np.int64, copy=False)
    return picked[(picked >= 0) & (picked < n_elems)]


def member_values(members: object) -> np.ndarray:
    """Return a tag group's members as whole numbers, whatever they came as.

    Parameters
    ----------
    members
        A tag group's members, as the mesh carries them.

    Returns
    -------
    numpy.ndarray
        The members as int64, flat and in the order the group holds them.
        Empty when the group holds no numbers at all.

    Notes
    -----
    The unbounded twin of :func:`member_indices`, for the writers that count
    the members reaching no entity of the mesh and so need to see them. A
    float is truncated rather than refused, which is what every caller here
    did with its own ``astype`` before there was one place to do it; what
    they did not do is survive a group holding a ragged list, or names, which
    index nothing and answer empty rather than raising out of the middle of a
    write.

    Examples
    --------
    >>> member_values([2, 0]).tolist()
    [2, 0]
    >>> member_values([[0, 1], [2]]).tolist()
    []
    """
    held = members_array(members)
    if held is None or held.dtype.kind not in "biuf":
        return np.empty(0, dtype=np.int64)
    return held.ravel().astype(np.int64)


def integer_column(stored: object, n_elems: int) -> np.ndarray | None:
    """Return a stored attribute if it is one integer per element, else None.

    Parameters
    ----------
    stored
        The attribute as the mesh carries it.
    n_elems
        How many elements the mesh holds.

    Returns
    -------
    numpy.ndarray or None
        The attribute, or None when it does not describe this mesh. A float
        column is refused rather than truncated: a label is a number the file
        names exactly, and rounding one relabels the elements.
    """
    values = members_array(stored)
    if (
        values is None
        or values.ndim != 1
        or values.shape[0] != n_elems
        or values.dtype.kind not in "iub"
    ):
        return None
    return values


def values_from_tags(
    tags: dict[str, np.ndarray] | None,
    prefix: str,
    n_elems: int,
    *,
    dtype: np.dtype | type,
) -> TagValues:
    """Return the labels the ``prefix<n>`` tag groups spell.

    Parameters
    ----------
    tags
        The mesh's element tag groups.
    prefix
        The naming convention a group has to follow to be a label.
    n_elems
        How many elements the mesh holds.
    dtype
        Type of the returned column, whose width is what a label has to fit
        in to be written at all.

    Returns
    -------
    TagValues
        The label column and the groups that could not be written, each named
        on the tuple rather than left to a positional unpack.

    Notes
    -----
    Which element each group reached is tracked in one boolean column rather
    than by intersecting the groups against each other, so the whole walk
    stays linear in the members the mesh holds however many groups it spreads
    them over.
    """
    values = np.zeros(n_elems, dtype=dtype)
    limits = np.iinfo(values.dtype)
    pattern = re.compile(rf"{re.escape(prefix)}(-?\d+)")
    claimed = np.zeros(n_elems, dtype=bool)
    named = False
    unnamed: set[str] = set()
    unusable: set[str] = set()
    oversized: set[str] = set()
    contested: set[str] = set()
    for name, members in (tags or {}).items():
        match = pattern.fullmatch(name)
        if match is None:
            unnamed.add(name)
            continue
        held = members_array(members)
        if held is None or held.ravel().dtype.kind not in "iub":
            unusable.add(name)
            continue
        label = int(match.group(1))
        if not limits.min <= label <= limits.max:
            oversized.add(name)
            continue
        picked = member_indices(held, n_elems)
        if not picked.size:
            # Every member indexed past the end of this mesh, so the label
            # reaches nothing. Counting it as a label named would write a
            # column of zeros over a mesh no group actually labelled.
            unusable.add(name)
            continue
        if claimed[picked].any():
            contested.add(name)
        claimed[picked] = True
        values[picked] = label
        named = True
    return TagValues(values, unnamed, named, unusable, oversized, contested)


def mask_arrays(
    tags: dict[str, np.ndarray] | None,
    n_items: int,
    *,
    fmt: str,
    kind: str,
    stacklevel: int = 3,
) -> dict[str, np.ndarray]:
    """Return one membership column per tag group, named for it.

    Parameters
    ----------
    tags
        The mesh's tag groups.
    n_items
        How many vertices or elements the mesh holds.
    fmt
        The format's extension, for the warning naming what was dropped.
    kind
        ``point`` or ``cell``, named in the same warning.
    stacklevel
        How far above this frame the warnings should point. The default
        answers a codec's ``write``, which is where this is called from; a
        wrapper adds its own frame to it.

    Returns
    -------
    dict of str to numpy.ndarray
        One ``uint8`` column of ones and zeros per group, keyed by the group's
        name behind :data:`MASK_PREFIX`. Empty when there are no groups.

    Warns
    -----
    UserWarning
        Once, naming every group with a member this mesh has no entity for,
        and once for the groups no column can be named for: the column's name
        is the only thing that says it is a group rather than an attribute,
        so a group whose name is blank - or is not text at all - reads back
        as neither.

    Notes
    -----
    A column rather than a list of indices, because the channel it travels in
    holds one value per entity and nothing else. An element in two groups is
    named by both columns, which is what a single label per element - the way
    Medit, Gmsh and Netgen have to spell it - cannot say.

    Nothing checks a tag group on the way in, so a group may hold floats -
    refused whole rather than rounded, since rounding an index moves a label
    onto another entity - an index past the end of a mesh it was not built
    for, or a ragged list numpy builds no array of at all. Each leaves the
    column short of the members the group named, which is a loss the
    ``*Elset`` writers already report and this one reports the same way.

    One column per group is one value per entity per group, so what this
    costs a file is the product and not the sum: a mesh of a million cells
    carrying two hundred groups writes two hundred million values, where the
    members themselves are a million at most. That is the price of saying
    what a single label per element cannot, and it is worth knowing before
    converting a ``.mesh`` or a Nastran deck, whose readers open one group
    per distinct reference or property id.

    Examples
    --------
    >>> mask_arrays({"a": np.array([0, 2])}, 3, fmt=".vtu", kind="cell")
    {'polyxios_tag_a': array([1, 0, 1], dtype=uint8)}
    """
    columns: dict[str, np.ndarray] = {}
    unreachable: list[str] = []
    unnamed: list[str] = []
    for name, members in (tags or {}).items():
        if not isinstance(name, str) or not name:
            unnamed.append(name)
            continue
        # ``held`` rather than ``members`` into member_indices: the column is
        # the expensive half of this loop and the conversion is not worth
        # doing twice for the size check below.
        held = members_array(members)
        column = np.zeros(n_items, dtype=np.uint8)
        if held is None:
            unreachable.append(name)
        else:
            picked = member_indices(held, n_items)
            column[picked] = 1
            if picked.size != held.size:
                unreachable.append(name)
        columns[f"{MASK_PREFIX}{name}"] = column
    if unnamed:
        warnings.warn(
            f"{fmt} write: {kind} tag group(s) {sorted(unnamed, key=repr)} have"
            " no name a data array can carry; dropped.",
            UserWarning,
            stacklevel=stacklevel,
        )
    if unreachable:
        warnings.warn(
            f"{fmt} write: {kind} tag group(s) {sorted(unreachable)} name"
            f" members that index no {kind} of this mesh; those members were"
            " dropped.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return columns


def with_tag_masks(
    attrs: dict[str, np.ndarray],
    tags: dict[str, np.ndarray] | None,
    n_items: int,
    *,
    fmt: str,
    kind: str,
) -> dict[str, np.ndarray]:
    """Return the arrays a data section writes: the attributes and the groups.

    Parameters
    ----------
    attrs
        The mesh's ``vertex_attrs`` or ``element_attrs``.
    tags
        Its tag groups of the same kind.
    n_items
        How many vertices or elements the mesh holds.
    fmt
        The format's extension, for the warnings naming what was dropped.
    kind
        ``point`` or ``cell``, named in the same warnings.

    Returns
    -------
    dict of str to numpy.ndarray
        The attributes, with one membership column per group added.

    Warns
    -----
    UserWarning
        Once, naming every attribute a group's column takes the name of, plus
        whatever :func:`mask_arrays` reports of the groups themselves.

    Notes
    -----
    An attribute already called what a group's column is called is the one
    collision the section cannot spell twice, and the group wins it: the file
    has one array of that name, and a reader takes it for the group whichever
    of the two wrote it - so writing the attribute would lose the group *and*
    read the attribute back as a group it never was. The attribute is lost
    either way, which is why it is worth a word.

    Examples
    --------
    >>> attrs = {"s": np.zeros(3)}
    >>> sorted(with_tag_masks(attrs, {"a": np.array([0])}, 3, fmt=".vtu", kind="cell"))
    ['polyxios_tag_a', 's']
    """
    masks = mask_arrays(tags, n_items, fmt=fmt, kind=kind, stacklevel=4)
    if not masks:
        return dict(attrs)
    shadowed = masks.keys() & attrs.keys()
    if shadowed:
        warnings.warn(
            f"{fmt} write: {kind} attribute(s) {sorted(shadowed)} are named"
            f" what a tag group's column is named; the group is written and"
            " the attribute is dropped.",
            UserWarning,
            stacklevel=3,
        )
    return attrs | masks


def tags_from_masks(
    attrs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split the membership columns out of a block of attributes.

    Parameters
    ----------
    attrs
        The arrays a reader took off one data section.

    Returns
    -------
    tuple[dict of str to numpy.ndarray, dict of str to numpy.ndarray]
        The attributes that are attributes, and the tag groups the rest spell.

    Notes
    -----
    A column that carries whole numbers reads as membership wherever it is not
    zero; one that carries anything else - a float attribute a writer happened
    to name this way, a vector - is left an attribute, since a group whose
    members were rounded into place names the wrong entities.

    A column named the bare prefix and nothing else is left an attribute too:
    the part after the prefix is the group's name, and there is no group
    called nothing. :func:`mask_arrays` refuses to write one, so a file
    holding it was written by something else.

    Examples
    --------
    >>> attrs = {"polyxios_tag_a": np.array([1, 0, 1]), "s": np.arange(3.0)}
    >>> kept, tags = tags_from_masks(attrs)
    >>> sorted(kept), tags["a"].tolist()
    (['s'], [0, 2])
    """
    kept: dict[str, np.ndarray] = {}
    tags: dict[str, np.ndarray] = {}
    for name, values in attrs.items():
        group = name[len(MASK_PREFIX) :] if name.startswith(MASK_PREFIX) else ""
        if not group or not _is_membership(values):
            kept[name] = values
            continue
        tags[group] = np.flatnonzero(values).astype(np.int32)
    return kept, tags


def _is_membership(values: object) -> bool:
    """Say whether a column can be read as one entity's membership per row.

    Parameters
    ----------
    values
        The array a reader took off a data section.

    Returns
    -------
    bool
        True for one whole number per entity. A legacy VTK data array is a
        double whatever it holds, so a column of ones and zeros written as
        one still reads as membership; a column carrying anything else does
        not, since a group whose members were rounded into place names the
        wrong entities.
    """
    column = np.asarray(values)
    if column.ndim != 1:
        return False
    if column.dtype.kind in "iub":
        return True
    return bool(
        column.dtype.kind == "f"
        and np.isfinite(column).all()
        and np.array_equal(column, np.trunc(column))
    )
