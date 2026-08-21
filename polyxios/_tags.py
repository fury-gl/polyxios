"""Turning a column of integer labels into named tag groups, and back.

Several formats label an element with a bare number - a Medit reference, a
Nastran property id - and a number that stays a column of integers does not
survive a conversion the way a named group does. The two directions live here
so the codecs that share the convention cannot drift apart on it.
"""

import re

import numpy as np

__all__ = [
    "group_by_value",
    "integer_column",
    "member_indices",
    "values_from_tags",
]


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
    extra dimension, or an index past the end of a mesh it was not built for.
    Each of those indexes nothing: a float column is refused whole rather than
    rounded - rounding an index moves a label onto another element - and a
    stale index reaches an element that is not the one it named. Dropping them
    is what lets a writer report the loss instead of raising whatever the
    first stray value happens to raise.
    """
    picked = np.asarray(members).ravel()
    if picked.dtype.kind not in "iub":
        return np.empty(0, dtype=np.int64)
    picked = picked.astype(np.int64, copy=False)
    return picked[(picked >= 0) & (picked < n_elems)]


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
    values = np.asarray(stored)
    if values.ndim != 1 or values.shape[0] != n_elems or values.dtype.kind not in "iub":
        return None
    return values


def values_from_tags(
    tags: dict[str, np.ndarray] | None,
    prefix: str,
    n_elems: int,
    *,
    dtype: np.dtype | type,
) -> tuple[np.ndarray, set[str], bool, set[str], set[str]]:
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
    numpy.ndarray
        One label per element, zero where no group names one.
    set of str
        The groups whose names spell no number, so a caller can report them.
    bool
        Whether any group did name one.
    set of str
        The groups whose members are not element indices, so a caller can
        report those too. Nothing checks a tag group's dtype on the way in,
        and a float one indexes nothing: it is refused outright rather than
        rounded, which is the right answer given rounding an index moves a
        label onto another element.
    set of str
        The groups naming a number the column cannot hold. A name is free
        text, so it may spell one wider than the format's own field; assigning
        it raises an OverflowError from numpy, which says nothing about which
        group is at fault, so it is reported here instead.
    """
    values = np.zeros(n_elems, dtype=dtype)
    limits = np.iinfo(values.dtype)
    pattern = re.compile(rf"{re.escape(prefix)}(-?\d+)")
    named = False
    unnamed: set[str] = set()
    unusable: set[str] = set()
    oversized: set[str] = set()
    for name, members in (tags or {}).items():
        match = pattern.fullmatch(name)
        if match is None:
            unnamed.add(name)
            continue
        if np.asarray(members).ravel().dtype.kind not in "iub":
            unusable.add(name)
            continue
        label = int(match.group(1))
        if not limits.min <= label <= limits.max:
            oversized.add(name)
            continue
        values[member_indices(members, n_elems)] = label
        named = True
    return values, unnamed, named, unusable, oversized
