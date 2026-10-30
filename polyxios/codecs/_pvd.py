"""PVD codec - ParaView's collection file, an index of datasets over time.

A ``.pvd`` file holds no geometry. It is a short XML document - a
``<VTKFile type="Collection">`` root with a ``<Collection>`` of ``<DataSet>``
elements - and each ``DataSet`` names another file, its ``timestep``, and
the ``part`` and ``group`` it belongs to when a step is split across
several. The files it names are whatever VTK writes: ``.vtu``, ``.vtp``,
``.vti``, a parallel ``.pvtu`` index, and they are found beside the index
by the relative path it spells.

:func:`read` hands back one mesh, always: every dataset at one time step
merged, the first step unless ``step=`` says otherwise. The whole series is
read by :func:`read_time_series` and written by :func:`write_time_series`,
both reached through :mod:`polyxios.helper`; :func:`write` is the one-step
series, its dataset written beside the index.
"""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import math
import operator
from pathlib import Path
from typing import Any
import warnings
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

import numpy as np

import polyxios
from polyxios import transforms
from polyxios._io import Source, require_path, source_name
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".pvd"
LABEL: str = "PVD"

_TIME_KEY: str = "time"
_DEFAULT_FORMAT: str = ".vtu"
# The index's own extension is never a dataset's: a ``.pvd`` naming a
# ``.pvd`` is a chain this reader does not follow.
_NOT_A_DATASET: frozenset[str] = frozenset({EXTENSION})


# =============================================================================
# Reading
# =============================================================================


def read(path: Source, *, lazy: bool = False, **opts: Any) -> PolyData:
    """Read a PVD collection at one time step and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.pvd`` file. A file object is refused: the datasets
        the index names are found beside it, and a handle has no beside.
    lazy
        Passed to the reader of each dataset. A step held in one dataset
        comes back as that reader hands it out - views of a ``.vtu`` with a
        raw appended section, say - while a step split across several is
        merged, which copies. A dataset that is itself an index - a
        ``.pvtu`` naming its pieces - is merged the same way, through
        :func:`polyxios.helper.read_multiblock`, which reads eagerly.
    **opts
        ``step`` picks the time step, counted from zero and negative from
        the end the way a list is; the first is read without it. Every
        dataset at that step is merged into the one mesh and, when there
        are several, tagged the way ParaView groups them: the datasets of
        one ``group`` share its tag, which names every element of every
        one of them, and a dataset naming no group is ``part_<n>``, ``n``
        its place among the step's datasets counted from zero. A name
        already in use - a tag a dataset carries, or another group's - is
        left to it, and the collection's gets a number appended.

    Returns
    -------
    PolyData
        The mesh, its step's ``timestep`` under ``global_attrs["time"]``.

    Raises
    ------
    CodecError
        If the file is not a collection, names no dataset, names one outside
        its own directory, or ``step`` is not a whole number or names a step
        it does not have; and whatever the dataset's own reader raises.
    LazyReadError
        If ``lazy`` is set and a dataset's codec cannot map its file - a
        ``.vtkhdf``, whose arrays h5py decodes whole. ``lazy`` is handed
        on as asked, not dropped: a caller that wants views over a
        collection of such files reads it eagerly instead.

    Warns
    -----
    UserWarning
        When two datasets at one step spell a ``global_attrs`` key
        differently; the later dataset's value is the one kept.
    """
    step = opts.pop("step", None)
    _warn_unknown_opts(opts, "read")
    index = _index_path(path, reading=True)
    times, entries = _entries(index)
    which = _step_index(step, len(times), index)
    return _read_at(index, times[which], entries, lazy=lazy)


def read_time_series(path: Source, **opts: Any) -> list[PolyData]:
    """Read every time step of a collection, one PolyData each.

    Parameters
    ----------
    path
        Path to the ``.pvd`` file.
    **opts
        None are taken; any given is warned about and ignored.

    Returns
    -------
    list of PolyData
        One mesh per time step, in ascending time, each carrying its
        ``timestep`` under ``global_attrs["time"]``.
    """
    _warn_unknown_opts(opts, "read")
    index = _index_path(path, reading=True)
    times, entries = _entries(index)
    return [_read_at(index, time, entries, lazy=False) for time in times]


def _warn_unknown_opts(opts: dict[str, Any], what: str) -> None:
    if opts:
        warnings.warn(
            f"{EXTENSION} {what}: unrecognized options {set(opts)}; ignored.",
            UserWarning,
            stacklevel=3,
        )


def _index_path(path: Source, *, reading: bool) -> Path:
    verb = "found" if reading else "written"
    return require_path(
        path,
        fmt=EXTENSION,
        reason=f"the datasets the index names are {verb} beside it",
        reading=reading,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Entry:
    time: float
    group: str
    file: Path


def _entries(index: Path) -> tuple[list[float], list[_Entry]]:
    """Every dataset the index names, and the ascending times they cover."""
    try:
        root = ET.parse(index).getroot()
    except (ET.ParseError, LookupError, ValueError) as exc:
        raise CodecError(f"'{index}': not a PVD file ({exc}).") from None
    except OSError as exc:
        raise CodecError(f"'{index}': {exc.strerror or exc}.") from None
    if root.tag != "VTKFile" or root.get("type") != "Collection":
        raise CodecError(
            f"'{index}': not a PVD file; the root is not <VTKFile type=\"Collection\">."
        )
    collection = root.find("Collection")
    if collection is None:
        raise CodecError(f"'{index}': the file holds no <Collection>.")
    home = index.resolve().parent
    entries: list[_Entry] = []
    for k, node in enumerate(collection.findall("DataSet")):
        spelled = node.get("file")
        if not spelled:
            raise CodecError(f"'{index}': DataSet {k} names no file.")
        # An index written on Windows spells its subdirectories with a
        # backslash, which POSIX takes for one long filename.
        target = (home / spelled.replace("\\", "/")).resolve()
        if not target.is_relative_to(home):
            # The same line the multi-block helpers draw: an index is data,
            # and one naming a file outside its own directory asks for a
            # file the caller did not.
            raise CodecError(
                f"'{index}': DataSet {k} names '{spelled}', which is outside"
                " the index's own directory."
            )
        if target.suffix.lower() in _NOT_A_DATASET:
            raise CodecError(
                f"'{index}': DataSet {k} names '{spelled}', another collection;"
                " a collection of collections is not read."
            )
        # ParaView reads ``part`` as an integer and refuses one that is not;
        # the tag it gets here is its place at its step, so several datasets
        # all spelling part="0" still read apart.
        _require_whole(node.get("part"), index, k, "part")
        entries.append(
            _Entry(
                time=_number(node.get("timestep"), index, k, "timestep", 0.0),
                group=node.get("group") or "",
                file=target,
            )
        )
    if not entries:
        raise CodecError(f"'{index}': the collection names no dataset.")
    times = sorted({entry.time for entry in entries})
    return times, entries


def _number(text: str | None, index: Path, k: int, what: str, default: float) -> float:
    if text is None or not text.strip():
        return default
    try:
        value = float(text)
    except ValueError:
        raise CodecError(
            f"'{index}': DataSet {k} spells {what}={text!r}, which is not a number."
        ) from None
    if not math.isfinite(value):
        raise CodecError(
            f"'{index}': DataSet {k} spells {what}={text.strip()}, which is not"
            " a finite number."
        )
    return value


def _require_whole(text: str | None, index: Path, k: int, what: str) -> None:
    if text is None or not text.strip():
        return
    try:
        int(text)
    except ValueError:
        raise CodecError(
            f"'{index}': DataSet {k} spells {what}={text!r}, which is not a whole"
            " number."
        ) from None


def _step_index(step: Any, n_steps: int, index: Path) -> int:
    if step is None:
        return 0
    try:
        which = operator.index(step)
    except TypeError:
        raise CodecError(
            f"'{index}': step={step!r} is not a whole number; a step is counted"
            " from zero, and negative from the end."
        ) from None
    if which < 0:
        which += n_steps
    if not 0 <= which < n_steps:
        raise CodecError(
            f"'{index}': step={step} is out of range; the collection holds"
            f" {n_steps} time step(s)."
        )
    return which


def _read_at(
    index: Path, time: float, entries: list[_Entry], *, lazy: bool
) -> PolyData:
    chosen = [entry for entry in entries if entry.time == time]
    meshes = [_read_dataset(entry, lazy=lazy) for entry in chosen]
    if len(meshes) == 1:
        poly = meshes[0]
        return dataclasses.replace(
            poly, global_attrs={**poly.global_attrs, _TIME_KEY: time}
        )
    tagged: list[PolyData] = []
    merged_globals: dict[str, Any] = {}
    labels = _labels(chosen, meshes)
    for entry, poly, label in zip(chosen, meshes, labels, strict=True):
        tags = {
            **poly.element_tags,
            label: np.arange(len(poly.element_types), dtype=np.int32),
        }
        for key, value in poly.global_attrs.items():
            if key in merged_globals and _differs(merged_globals[key], value):
                warnings.warn(
                    f"{EXTENSION}: the datasets at time {time} spell global"
                    f" '{key}' differently; '{entry.file.name}' has the last"
                    " word.",
                    UserWarning,
                    stacklevel=4,
                )
            merged_globals[key] = value
        tagged.append(dataclasses.replace(poly, element_tags=tags, global_attrs={}))
    merged = transforms.merge(*tagged)
    merged_globals[_TIME_KEY] = time
    return dataclasses.replace(merged, global_attrs=merged_globals)


def _labels(chosen: list[_Entry], meshes: list[PolyData]) -> list[str]:
    """The tag each dataset's elements get, the datasets of one group sharing it.

    A dataset naming no group is ``part_<n>``, ``n`` its place among the
    step's datasets. Merging unions same-named tags, so one label per group
    is what makes the group one tag. A name some dataset already tags its
    elements with is left to it: the collection's gets a number appended.
    """
    taken = {name for poly in meshes for name in poly.element_tags}
    by_key: dict[str | int, str] = {}
    labels: list[str] = []
    for n, entry in enumerate(chosen):
        key: str | int = entry.group or n
        if key not in by_key:
            wanted = entry.group or f"part_{n}"
            label = wanted
            counter = 0
            while label in taken:
                counter += 1
                label = f"{wanted}_{counter}"
            taken.add(label)
            by_key[key] = label
        labels.append(by_key[key])
    return labels


def _differs(before: Any, after: Any) -> bool:
    try:
        return not np.array_equal(np.asarray(before), np.asarray(after))
    except (TypeError, ValueError):
        return before != after


def _read_dataset(entry: _Entry, *, lazy: bool) -> PolyData:
    # helper imports this codec through the series dispatch, so it is
    # reached at call time rather than at import.
    from polyxios import helper

    if entry.file.suffix.lower() in helper.META_SUFFIXES:
        return helper.read_multiblock(entry.file)
    return polyxios.read(entry.file, lazy=lazy)


# =============================================================================
# Writing
# =============================================================================


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as a one-step collection, its dataset beside the index.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.pvd`` path. A file object is refused: the dataset is
        written beside the index, and a handle has no beside.
    **opts
        ``format`` names the dataset's extension, ``.vtu`` without it; any
        extension polyxios writes will do, ``.vtp`` for a surface or
        ``.vti`` for a lattice. ``time`` is the step's ``timestep``;
        without it, a number under ``global_attrs["time"]`` is, and zero
        otherwise. The index spells the time, so ``global_attrs["time"]``
        is never written into the dataset: a number there is left to the
        index, a text dropped with a warning, since the step's time reads
        back over it. Every other option goes to the dataset's writer -
        ``binary``, ``appended`` for a ``.vtu``.

    Raises
    ------
    CodecError
        If ``format`` is the collection's own extension, the time is not a
        finite number, or the dataset's writer refuses the mesh.
    UnsupportedFormatError
        If ``format`` names an extension polyxios does not write.

    Warns
    -----
    UserWarning
        For a text ``global_attrs["time"]``, dropped; and whatever the
        dataset's writer warns about.
    """
    fmt = _format_of(opts, source_name(path))
    time = opts.pop("time", None)
    index = _index_path(path, reading=False)
    time = _time_value(_time_of(poly, time, 0.0), index=index, k=0)
    _write_index(index, [(time, _write_dataset(index, 0, poly, fmt=fmt, opts=opts))])


def write_time_series(
    steps: Iterable[tuple[float, PolyData]], path: Source, **opts: Any
) -> None:
    """Write a sequence of meshes as one collection, a dataset per step.

    Parameters
    ----------
    steps
        ``(time, mesh)`` pairs in time order. Any iterable will do, a
        generator included: each step is written as it arrives. The meshes
        need not share their elements - every step is a file of its own.
    path
        Output ``.pvd`` path.
    **opts
        As :func:`write`, ``time`` excepted: each step carries its own, and
        a ``global_attrs["time"]`` is left out of every dataset the same
        way.

    Raises
    ------
    CodecError
        If ``steps`` is empty, a step's time is not a finite number, or two
        steps spell the same time. The datasets written before the refusal stay
        beside the index's path, which is not written; a shorter series
        written over a longer one leaves the earlier run's later datasets
        beside it too. A read follows the index, so neither is read.
    UnsupportedFormatError
        If ``format`` names an extension polyxios does not write.
    """
    fmt = _format_of(opts, source_name(path))
    index = _index_path(path, reading=False)
    written: list[tuple[float, str]] = []
    seen: set[float] = set()
    for k, (time, poly) in enumerate(steps):
        time = _time_value(time, index=index, k=k)
        if time in seen:
            raise CodecError(
                f"'{index}': two steps are at time {time}; a collection reads"
                " every dataset at one time as parts of one mesh."
            )
        seen.add(time)
        written.append((time, _write_dataset(index, k, poly, fmt=fmt, opts=opts)))
    if not written:
        raise CodecError(f"'{index}': a time series needs at least one step.")
    _write_index(index, written)


def _format_of(opts: dict[str, Any], name: str) -> str:
    fmt = str(opts.pop("format", _DEFAULT_FORMAT)).strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    if fmt in _NOT_A_DATASET:
        raise CodecError(
            f"'{name}': format={fmt!r} names another collection; a dataset is"
            " a mesh file such as .vtu or .vtp."
        )
    return fmt


def _time_of(poly: PolyData, time: Any, default: float) -> Any:
    if time is not None:
        return time
    held = _held_time(poly)
    return default if held is None else held


def _held_time(poly: PolyData) -> float | None:
    """The number under ``global_attrs["time"]``, or None when there is none."""
    held = (poly.global_attrs or {}).get(_TIME_KEY)
    if isinstance(held, bool) or not isinstance(
        held, (int, float, np.integer, np.floating)
    ):
        return None
    return float(held)


def _time_value(time: Any, *, index: Path, k: int) -> float:
    """A step's time as the index will spell it; refused unless finite, as on read."""
    try:
        value = float(time)
    except (TypeError, ValueError):
        raise CodecError(
            f"'{index}': step {k} is at time {time!r}, which is not a number."
        ) from None
    if not math.isfinite(value):
        raise CodecError(
            f"'{index}': step {k} is at time {value}, which no collection can spell."
        )
    return value


def _write_dataset(
    index: Path, k: int, poly: PolyData, *, fmt: str, opts: dict[str, Any]
) -> str:
    """Write one step's dataset beside the index; return the name spelled.

    The index spells the time, so ``global_attrs["time"]`` is left out of
    the dataset: written there, it would read back under the index's value.
    A number is the time itself, or the one ``time=`` replaced, and goes
    silently; a text is nothing of the kind, and is warned about.
    """
    target = index.with_name(f"{index.stem}_{k}{fmt}")
    if _TIME_KEY in (poly.global_attrs or {}):
        held = dict(poly.global_attrs)
        value = held.pop(_TIME_KEY)
        if _held_time(poly) is None:
            warnings.warn(
                f"{EXTENSION} write: global '{_TIME_KEY}' holds {value!r}, and"
                f" the index spells step {k}'s time, which reads back over it;"
                " dropped.",
                UserWarning,
                stacklevel=4,
            )
        poly = dataclasses.replace(poly, global_attrs=held)
    polyxios.write(poly, target, **opts)
    return target.name


def _write_index(index: Path, datasets: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time, name in datasets:
        lines.append(
            f'    <DataSet timestep="{time!r}" group="" part="0"'
            f" file={quoteattr(name)}/>"
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    index.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "EXTENSION",
    "LABEL",
    "read",
    "read_time_series",
    "write",
    "write_time_series",
]
