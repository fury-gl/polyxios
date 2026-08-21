"""Abaqus ``.inp`` ASCII codec - read + write.

An Abaqus deck is a sequence of keyword cards, each followed by its data
lines. This codec reads the mesh cards - ``*NODE``, ``*ELEMENT``, ``*NSET``,
``*ELSET``, ``*SYSTEM`` and ``*INCLUDE`` - and ignores the solver cards
around them, since a step definition says nothing about the geometry.

Node and element sets become ``vertex_tags`` and ``element_tags``. A deck
split over several files with ``*INCLUDE`` is read as one mesh, with the
included paths resolved against the parent file and refused when they leave
its directory - an ``.inp`` is untrusted input like any other file.
"""

import os
from pathlib import Path
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
from polyxios._io import Source, is_buffer, read_text, source_name, write_text
from polyxios._tags import member_indices
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".inp"

# A deck that includes itself is legal text and reads until memory runs out,
# so the nesting is capped. Real decks nest two or three deep.
_MAX_INCLUDE_DEPTH: int = 8

# Characters a set name cannot carry: each of them ends the card early.
_NAME_UNSAFE: frozenset[str] = frozenset(",=*")

# Abaqus element cards, keyed by the base card with its modifier suffixes
# removed. The suffixes - R (reduced integration), H (hybrid), I
# (incompatible modes), M (modified), T (coupled temperature), S/W (shell
# variants), P (pore pressure), and the 5/6 degree-of-freedom shell numbers -
# change the solver's behaviour, never the node count or their order, so
# CPS8R holds the same eight nodes as CPS8.
_INP_BASE_TYPES: dict[str, tuple[str, int]] = {
    # trusses and beams
    "T2D2": ("line", 2),
    "T2D3": ("quadratic_edge", 3),
    "T3D2": ("line", 2),
    "T3D3": ("quadratic_edge", 3),
    "B21": ("line", 2),
    "B22": ("quadratic_edge", 3),
    "B31": ("line", 2),
    "B32": ("quadratic_edge", 3),
    "B33": ("quadratic_edge", 3),
    "DC1D2": ("line", 2),
    "DC1D3": ("quadratic_edge", 3),
    # plane stress / plane strain / axisymmetric / generalised plane strain
    "CPS3": ("triangle", 3),
    "CPE3": ("triangle", 3),
    "CPEG3": ("triangle", 3),
    "CAX3": ("triangle", 3),
    "CGAX3": ("triangle", 3),
    "CPS4": ("quad", 4),
    "CPE4": ("quad", 4),
    "CPEG4": ("quad", 4),
    "CAX4": ("quad", 4),
    "CGAX4": ("quad", 4),
    "CPS6": ("quadratic_triangle", 6),
    "CPE6": ("quadratic_triangle", 6),
    "CPEG6": ("quadratic_triangle", 6),
    "CAX6": ("quadratic_triangle", 6),
    "CGAX6": ("quadratic_triangle", 6),
    "CPS8": ("quadratic_quad", 8),
    "CPE8": ("quadratic_quad", 8),
    "CPEG8": ("quadratic_quad", 8),
    "CAX8": ("quadratic_quad", 8),
    "CGAX8": ("quadratic_quad", 8),
    # shells, membranes and rigid surfaces
    "S3": ("triangle", 3),
    "STRI3": ("triangle", 3),
    "S4": ("quad", 4),
    "S6": ("quadratic_triangle", 6),
    "STRI65": ("quadratic_triangle", 6),
    "S8": ("quadratic_quad", 8),
    "S9": ("biquadratic_quad", 9),
    "SC6": ("wedge", 6),
    "SC8": ("hexahedron", 8),
    "R3D3": ("triangle", 3),
    "R3D4": ("quad", 4),
    "M3D3": ("triangle", 3),
    "M3D4": ("quad", 4),
    "M3D6": ("quadratic_triangle", 6),
    "M3D8": ("quadratic_quad", 8),
    "M3D9": ("biquadratic_quad", 9),
    # solids
    "C3D4": ("tetra", 4),
    "C3D5": ("pyramid", 5),
    "C3D6": ("wedge", 6),
    "C3D8": ("hexahedron", 8),
    "C3D10": ("quadratic_tetra", 10),
    "C3D15": ("quadratic_wedge", 15),
    "C3D20": ("quadratic_hexahedron", 20),
    # heat transfer and cohesive
    "DC2D3": ("triangle", 3),
    "DC2D4": ("quad", 4),
    "DC2D6": ("quadratic_triangle", 6),
    "DC2D8": ("quadratic_quad", 8),
    "DC3D4": ("tetra", 4),
    "DC3D6": ("wedge", 6),
    "DC3D8": ("hexahedron", 8),
    "DC3D10": ("quadratic_tetra", 10),
    "DC3D15": ("quadratic_wedge", 15),
    "DC3D20": ("quadratic_hexahedron", 20),
    "COH2D4": ("quad", 4),
    "COH3D6": ("wedge", 6),
    "COH3D8": ("hexahedron", 8),
}

# The card written for each polyxios type. Cards other readers know are
# preferred over the ones this codec would pick for itself, so an exported
# deck loads elsewhere; ``element_type=`` overrides any of them.
_POLYXIOS_TO_INP: dict[str, str] = {
    "line": "T3D2",
    "quadratic_edge": "T3D3",
    "triangle": "S3",
    "quadratic_triangle": "STRI65",
    "quad": "S4",
    "quadratic_quad": "S8R",
    "biquadratic_quad": "S9R5",
    "tetra": "C3D4",
    "quadratic_tetra": "C3D10",
    "pyramid": "C3D5",
    "wedge": "C3D6",
    "quadratic_wedge": "C3D15",
    "hexahedron": "C3D8",
    "quadratic_hexahedron": "C3D20",
}


def _inp_type_info(type_str: str) -> tuple[str, int] | None:
    """Return the polyxios type a card names, ignoring its modifier suffixes.

    Parameters
    ----------
    type_str
        The ``TYPE=`` value of an ``*ELEMENT`` card.

    Returns
    -------
    tuple of (str, int) or None
        The polyxios element name and its node count, or None when no known
        card is a prefix of this one.

    Notes
    -----
    The lookup takes the longest prefix that ends in a digit - the node count
    is the last thing a card name spells before its modifiers - so ``CPS8R``
    and ``CPS8`` resolve alike while ``C3D20RH`` does not fall back to
    ``C3D2``.
    """
    up = type_str.upper().strip()
    for end in range(len(up), 0, -1):
        if not up[end - 1].isdigit():
            continue
        info = _INP_BASE_TYPES.get(up[:end])
        if info is not None:
            return info
    return _INP_BASE_TYPES.get(up)


def _parse_params(card: str) -> dict[str, str]:
    """Return a keyword card's parameters, upper-cased keys, values as written."""
    params: dict[str, str] = {}
    for part in card.split(",")[1:]:
        key, sep, value = part.partition("=")
        params[key.strip().upper()] = value.strip() if sep else ""
    return params


def _include_path(card: str, base: Path | None, name: str, root: Path | None) -> Path:
    """Return the file an ``*INCLUDE`` card names, refusing an escaping path.

    Parameters
    ----------
    card
        The ``*INCLUDE`` card, whitespace stripped.
    base
        Directory the card's path is relative to, which is the directory of
        the file the card sits in. None for a deck read from a buffer, which
        has no directory for a relative path to be relative to.
    name
        The file the card sits in, named in the error.
    root
        Directory the whole read is confined to, which is the top deck's own.
        None alongside a None ``base``.

    Returns
    -------
    pathlib.Path
        The resolved file.

    Raises
    ------
    CodecError
        On a card naming no file, a deck read from a buffer, an absolute
        path, a path resolving outside ``root``, or a file that is not there.

    Notes
    -----
    The boundary is the top deck's directory and stays there as the includes
    nest, rather than narrowing to each included file's own: a deck laid out
    as ``main.inp`` including ``parts/a.inp`` including ``../common/nodes.inp``
    never leaves the directory it was read from, and a boundary that followed
    the nesting would refuse the third file for stepping out of ``parts``.
    """
    params = _parse_params(card)
    target = params.get("INPUT") or params.get("FILE") or ""
    target = target.strip().strip("'\"")
    if not target:
        raise CodecError(f".inp: '{name}' has an *INCLUDE card with no INPUT=.")
    if base is None or root is None:
        raise CodecError(
            f".inp: '{name}' uses *INCLUDE, which names a file relative to the"
            " deck's own directory; read it from a path rather than a buffer."
        )
    if Path(target).is_absolute():
        raise CodecError(
            f".inp: *INCLUDE names the absolute path '{target}'; only paths"
            " inside the deck's own directory are read."
        )
    resolved = (base / target).resolve()
    if not resolved.is_relative_to(root):
        raise CodecError(
            f".inp: *INCLUDE path '{target}' resolves outside the deck's"
            f" directory '{root}'; refusing to read it."
        )
    if not resolved.is_file():
        raise CodecError(f".inp: *INCLUDE names '{target}', which does not exist.")
    return resolved


def _expand_includes(
    text: str,
    base: Path | None,
    name: str,
    depth: int,
    seen: tuple[Path, ...],
    root: Path | None,
) -> list[str]:
    """Return the deck's lines with every ``*INCLUDE`` replaced by its file."""
    out: list[str] = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped.upper().startswith("*INCLUDE"):
            out.append(ln)
            continue
        if depth >= _MAX_INCLUDE_DEPTH:
            raise CodecError(
                f".inp: *INCLUDE nests deeper than {_MAX_INCLUDE_DEPTH} files;"
                " a deck that includes itself would never finish."
            )
        target = _include_path(stripped, base, name, root)
        if target in seen:
            raise CodecError(
                f".inp: *INCLUDE nests '{target.name}' inside itself; a deck"
                " that includes itself would never finish."
            )
        out.extend(
            _expand_includes(
                target.read_text(encoding="utf-8", errors="replace"),
                target.parent,
                target.name,
                depth + 1,
                (*seen, target),
                root,
            )
        )
    return out


def _logical_lines(raw_lines: list[str]) -> list[str]:
    """Strip comments and join the continuation lines a data line may spill onto."""
    raw = [
        ln.split("**")[0].rstrip()
        for ln in raw_lines
        if not ln.strip().startswith("**")
    ]

    lines: list[str] = []
    buf = ""
    for ln in raw:
        stripped = ln.strip()
        if not stripped:
            if buf:
                lines.append(buf)
                buf = ""
            continue
        if stripped.startswith("*"):
            if buf:
                lines.append(buf)
                buf = ""
            lines.append(ln)
        elif stripped.endswith(",") or buf:
            buf += stripped
            if not stripped.endswith(","):
                lines.append(buf)
                buf = ""
        else:
            lines.append(ln)
    if buf:
        lines.append(buf)
    return lines


def _parse_system(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the origin and rotation a ``*SYSTEM`` card's data lines define.

    Parameters
    ----------
    rows
        The card's data lines, already split on commas.

    Returns
    -------
    tuple of numpy.ndarray or None
        Origin and a 3x3 rotation whose columns are the local axes, or None
        for a card with no data lines, which restores the global system.
    """
    if not rows:
        return None
    try:
        first = [float(tok) for tok in rows[0] if tok]
        second = [float(tok) for tok in rows[1] if tok] if len(rows) > 1 else []
    except ValueError as exc:
        raise CodecError(f".inp: malformed *SYSTEM data line: {exc}.") from exc
    if len(first) < 3:
        raise CodecError(
            f".inp: *SYSTEM needs at least an origin of 3 values, got {len(first)}."
        )
    origin = np.array(first[:3], dtype=np.float64)
    if len(first) < 6:
        # An origin on its own is a pure shift; the axes stay global.
        return origin, np.eye(3)

    x_axis = np.array(first[3:6], dtype=np.float64) - origin
    norm = float(np.linalg.norm(x_axis))
    if norm == 0.0:
        raise CodecError(".inp: *SYSTEM local X axis has zero length.")
    x_axis /= norm

    if len(second) >= 3:
        plane = np.array(second[:3], dtype=np.float64) - origin
    else:
        # No second point: any direction not along X gives the same result
        # for a pure rotation about X, which is all the card can mean here.
        plane = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(plane, x_axis))) > 1.0 - 1e-12:
            plane = np.array([0.0, 1.0, 0.0])
    y_axis = plane - float(np.dot(plane, x_axis)) * x_axis
    norm = float(np.linalg.norm(y_axis))
    if norm == 0.0:
        raise CodecError(
            ".inp: *SYSTEM second point lies on the local X axis, so it does"
            " not define a plane."
        )
    y_axis /= norm
    z_axis = np.cross(x_axis, y_axis)
    return origin, np.column_stack((x_axis, y_axis, z_axis))


def _set_indices(
    rows: list[list[str]],
    id_map: dict[int, int],
    known: dict[str, np.ndarray],
    folded: dict[str, str],
    *,
    generate: bool,
    name: str,
    what: str,
) -> list[int]:
    """Return the indices a ``*NSET`` / ``*ELSET`` body names.

    Parameters
    ----------
    rows
        The card's data lines, already split on commas.
    id_map
        The deck's node or element ids, to the indices they were read into.
    known
        The sets declared so far, which a body may name instead of ids.
    folded
        Upper-cased set name to the spelling ``known`` holds it under: Abaqus
        matches a set name without regard to case, so a body naming ``TOP``
        reaches the set the deck declared as ``Top``.
    generate
        Whether the card carried ``GENERATE``, making each row a range.
    name, what
        The set's name and kind, named in the warning when an entry is
        missing.

    Returns
    -------
    list of int
        The indices the body names, ascending and without repeats.

    Notes
    -----
    An id the deck never defined is reported rather than dropped in silence,
    since a set standing for a boundary condition matters.

    A ``GENERATE`` range wider than the deck has ids is resolved by walking
    the ids rather than the range: the card names two numbers and nothing
    bounds their distance, so the range itself is not a thing to iterate.
    """
    indices: list[int] = []
    n_missing = 0
    first_missing: str | None = None

    def miss(what_missed: str) -> None:
        """Count an entry the deck never defined, keeping the first name."""
        nonlocal n_missing, first_missing
        n_missing += 1
        if first_missing is None:
            first_missing = what_missed

    if generate:
        for row in rows:
            values = [tok for tok in row if tok]
            if len(values) < 2:
                continue
            try:
                first, last = int(values[0]), int(values[1])
                step = int(values[2]) if len(values) > 2 else 1
            except ValueError:
                miss(", ".join(values))
                continue
            if step <= 0:
                raise CodecError(
                    f".inp: {what} set '{name}' generates with a step of {step}."
                )
            if last < first:
                continue
            span = (last - first) // step + 1
            if span > len(id_map):
                # A range wider than the deck has ids is walked from the ids
                # instead: 'GENERATE 1, 999999999' is a legal card, and a loop
                # over the range it spells would run until the memory ran out.
                hits = 0
                for ident, index in id_map.items():
                    if first <= ident <= last and (ident - first) % step == 0:
                        indices.append(index)
                        hits += 1
                if hits < span:
                    n_missing += span - hits
                    if first_missing is None:
                        # Bounded by the ids the deck holds: the walk stops at
                        # the first gap, and every step before it was a hit.
                        first_missing = str(
                            next(
                                ident
                                for ident in range(first, last + 1, step)
                                if ident not in id_map
                            )
                        )
                continue
            for ident in range(first, last + 1, step):
                index = id_map.get(ident)
                if index is None:
                    miss(str(ident))
                else:
                    indices.append(index)
    else:
        for row in rows:
            for tok in row:
                if not tok:
                    continue
                try:
                    ident = int(tok)
                except ValueError:
                    # A set body may name sets declared earlier, in any case.
                    held = folded.get(tok.upper())
                    if held is None:
                        miss(tok)
                    else:
                        indices.extend(int(k) for k in known[held])
                    continue
                index = id_map.get(ident)
                if index is None:
                    miss(str(ident))
                else:
                    indices.append(index)

    if n_missing:
        warnings.warn(
            f".inp: {what} set '{name}' names {n_missing} entry(ies) the deck"
            f" never defines, first {first_missing!r}; they are dropped.",
            stacklevel=2,
        )
    return sorted(set(indices))


def _store_set(
    tags: dict[str, np.ndarray],
    folded: dict[str, str],
    name: str,
    indices: list[int],
    what: str,
) -> None:
    """Record a set under its name, merging a name the deck reuses.

    Parameters
    ----------
    tags
        Where the sets are collected.
    folded
        Upper-cased name to the spelling ``tags`` holds it under, carried
        along so a deck reusing a name in another case adds to the set it
        already declared rather than opening a second one beside it.
    name
        The set's name as the card spells it.
    indices
        The elements or nodes it holds.
    what
        ``node`` or ``element``, named in the warning for an empty set.
    """
    if not indices:
        warnings.warn(
            f".inp: {what} set '{name}' is empty; it is dropped.", stacklevel=2
        )
        return
    array = np.unique(np.array(indices, dtype=np.int32))
    key = folded.setdefault(name.upper(), name)
    held = tags.get(key)
    if held is not None:
        array = np.unique(np.concatenate([held, array]))
    tags[key] = array.astype(np.int32, copy=False)


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an Abaqus ``.inp`` file.

    Parameters
    ----------
    path
        Path to the ``.inp`` file, or a file object - a buffer holds no
        directory, so a deck read from one cannot use ``*INCLUDE``.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        If no ``*Node`` section is found, an ``*INCLUDE`` card names a
        missing file or one outside the deck's directory or nests too deeply,
        a ``*SYSTEM`` card is malformed, an element row is short or names an
        undefined node, or a count exceeds the safety caps.

    Notes
    -----
    Several ``*NODE`` blocks accumulate, and a repeated node id restates that
    node rather than adding another. ``NSET=`` / ``ELSET=`` on a block, and
    standalone ``*NSET`` / ``*ELSET`` cards with or without ``GENERATE``,
    become ``vertex_tags`` and ``element_tags``, and a set carrying
    ``INSTANCE=`` is numbered by that instance rather than by whatever
    numbering is in force where the card sits, which is where an assembly
    keeps its sets. ``*SYSTEM`` transforms the node blocks that follow it,
    until the next ``*SYSTEM``. Every ``*INSTANCE`` is merged into one mesh,
    each under its own node numbering and tagged by its instance name; one
    that only places its part shares the part's numbering. Solver cards are
    ignored.
    """
    if lazy:
        warnings.warn(
            ".inp: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    text = read_text(path, errors="replace")
    base = None if is_buffer(path) else Path(os.fspath(path)).parent  # type: ignore[arg-type]
    # The boundary every *INCLUDE has to stay inside, fixed here so the
    # nesting cannot narrow it as it descends into subdirectories.
    root = None if base is None else base.resolve()
    lines = _logical_lines(_expand_includes(text, base, source_name(path), 0, (), root))

    node_map: dict[int, int] = {}
    elem_map: dict[int, int] = {}
    coords: list[float] = []
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    vertex_tags: dict[str, np.ndarray] = {}
    element_tags: dict[str, np.ndarray] = {}
    # Upper-cased set name to the spelling the tags hold it under; Abaqus
    # matches a set name without regard to case and so must this.
    vertex_folded: dict[str, str] = {}
    element_folded: dict[str, str] = {}

    # The mesh cards carry state across their data lines, so the loop tracks
    # which card it is inside and what that card asked for.
    mode: str | None = None
    elem_info: tuple[str, int] | None = None
    block_nodes: list[int] = []
    block_elems: list[int] = []
    block_set: str | None = None
    set_name: str | None = None
    set_rows: list[list[str]] = []
    set_generate = False
    system: tuple[np.ndarray, np.ndarray] | None = None
    system_rows: list[list[str]] = []
    instance: str | None = None
    instance_part: str | None = None
    instance_nodes: list[int] = []
    instance_elems: list[int] = []
    # Upper-cased instance name to the node and element ids it numbered, kept
    # past its *End Instance so an assembly-level set naming INSTANCE= still
    # reaches them: a deck keeps almost all of its sets out there.
    instance_ids: dict[str, tuple[dict[int, int], dict[int, int]]] = {}
    # The node and element numbering in force outside the part being read,
    # held while it is open so the deck's own numbering survives it.
    outer_maps: tuple[dict[int, int], dict[int, int]] | None = None
    set_instance: str | None = None
    unknown_types: set[str] = set()
    unknown_instances: set[str] = set()
    repeated_elems = 0

    def close_block() -> None:
        """Record whatever the card that is ending asked to be recorded."""
        nonlocal system
        if mode == "system":
            system = _parse_system(system_rows)
        elif mode in ("nset", "elset") and set_name is not None:
            # A set naming INSTANCE= is numbered by that instance, not by
            # whatever the reader happens to be inside now.
            held = None
            if set_instance is not None:
                held = instance_ids.get(set_instance.upper())
                if held is None:
                    unknown_instances.add(set_instance)
            set_nodes, set_elems = held if held is not None else (node_map, elem_map)
            if mode == "nset":
                _store_set(
                    vertex_tags,
                    vertex_folded,
                    set_name,
                    _set_indices(
                        set_rows,
                        set_nodes,
                        vertex_tags,
                        vertex_folded,
                        generate=set_generate,
                        name=set_name,
                        what="node",
                    ),
                    "node",
                )
            else:
                _store_set(
                    element_tags,
                    element_folded,
                    set_name,
                    _set_indices(
                        set_rows,
                        set_elems,
                        element_tags,
                        element_folded,
                        generate=set_generate,
                        name=set_name,
                        what="element",
                    ),
                    "element",
                )
        elif mode == "node" and block_set:
            _store_set(vertex_tags, vertex_folded, block_set, block_nodes, "node")
        elif mode == "element" and block_set:
            _store_set(element_tags, element_folded, block_set, block_elems, "element")

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue

        if stripped.startswith("*"):
            close_block()
            mode = None
            block_set = None
            block_nodes = []
            block_elems = []
            set_rows = []
            set_name = None
            set_instance = None
            set_generate = False
            system_rows = []

            # Whitespace inside a keyword is collapsed, so '*End  Instance'
            # is the card '*End Instance' spelled loosely, not another one.
            keyword = " ".join(stripped.split(",")[0].split()).lstrip("*").upper()
            params = _parse_params(stripped)
            if keyword == "NODE":
                mode = "node"
                block_set = params.get("NSET") or None
            elif keyword == "ELEMENT":
                mode = "element"
                block_set = params.get("ELSET") or None
                type_str = params.get("TYPE", "")
                elem_info = _inp_type_info(type_str) if type_str else None
                if elem_info is None:
                    unknown_types.add(type_str or "<missing TYPE=>")
            elif keyword == "NSET":
                mode = "nset"
                set_name = params.get("NSET") or None
                set_instance = params.get("INSTANCE") or None
                set_generate = "GENERATE" in params
            elif keyword == "ELSET":
                mode = "elset"
                set_name = params.get("ELSET") or None
                set_instance = params.get("INSTANCE") or None
                set_generate = "GENERATE" in params
            elif keyword == "SYSTEM":
                mode = "system"
            elif keyword in ("INSTANCE", "PART"):
                # An instance numbers its own nodes from 1, so it reads
                # against a fresh map and its elements are tagged by name.
                # The numbering in force outside is set aside rather than
                # dropped: a deck may define nodes at the top level too, and
                # a set out there still has to reach them once the part
                # closes.
                if instance is None:
                    outer_maps = (node_map, elem_map)
                instance = params.get("NAME") or keyword.lower()
                instance_part = params.get("PART") or None
                node_map = {}
                elem_map = {}
                instance_nodes = []
                instance_elems = []
            elif keyword in ("END INSTANCE", "END PART"):
                if instance is not None:
                    if instance_nodes:
                        _store_set(
                            vertex_tags, vertex_folded, instance, instance_nodes, "node"
                        )
                    if instance_elems:
                        _store_set(
                            element_tags,
                            element_folded,
                            instance,
                            instance_elems,
                            "element",
                        )
                    held_nodes, held_elems = node_map, elem_map
                    if not held_nodes and not held_elems and instance_part:
                        # An instance that only places its part carries no
                        # nodes of its own, so a set naming the instance
                        # means the part's.
                        held_nodes, held_elems = instance_ids.get(
                            instance_part.upper(), ({}, {})
                        )
                    instance_ids[instance.upper()] = (held_nodes, held_elems)
                instance = None
                instance_part = None
                # Back to the numbering the deck was under before the part
                # opened. A stray '*End Part' that closes nothing leaves it
                # alone rather than wiping it.
                if outer_maps is not None:
                    node_map, elem_map = outer_maps
                    outer_maps = None
            continue

        if mode == "node":
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) < 3:
                continue
            try:
                node_id = int(parts[0])
                xyz = [float(parts[1]), float(parts[2])]
                xyz.append(float(parts[3]) if len(parts) >= 4 and parts[3] else 0.0)
            except ValueError as exc:
                raise CodecError(f".inp: malformed node line {stripped!r}.") from exc
            point = np.array(xyz, dtype=np.float64)
            if system is not None:
                origin, rotation = system
                point = origin + rotation @ point
            existing = node_map.get(node_id)
            if existing is None:
                index = len(coords) // 3
                if index >= MAX_SAFE_VERTICES:
                    raise CodecError(
                        f".inp: node count exceeds the safety cap {MAX_SAFE_VERTICES}."
                    )
                node_map[node_id] = index
                coords.extend(point.tolist())
            else:
                # A repeated id restates the same node; appending would leave
                # a vertex nothing references.
                index = existing
                coords[3 * index : 3 * index + 3] = point.tolist()
            block_nodes.append(index)
            if instance is not None:
                instance_nodes.append(index)

        elif mode == "element":
            if elem_info is None:
                continue
            elem_name, n_nodes = elem_info
            parts = [p.strip() for p in stripped.split(",")]
            try:
                elem_id = int(parts[0])
                nodes = [node_map[int(parts[1 + j])] for j in range(n_nodes)]
            except IndexError as exc:
                raise CodecError(
                    f".inp: element row has fewer than {n_nodes} node refs"
                ) from exc
            except KeyError as exc:
                raise CodecError(
                    f".inp: element refs undefined node {exc.args[0]}"
                ) from exc
            except ValueError as exc:
                raise CodecError(f".inp: malformed element line {stripped!r}.") from exc
            index = len(types_list)
            if index >= MAX_SAFE_ELEMENTS or len(conn_list) >= MAX_SAFE_CONN:
                raise CodecError(".inp: element count exceeds the safety caps.")
            if elem_id in elem_map:
                # Both elements are kept - they are two cells - but only the
                # last answers to the id, so a set naming it reaches one.
                repeated_elems += 1
            elem_map[elem_id] = index
            conn_list.extend(nodes)
            offsets_list.append(offsets_list[-1] + n_nodes)
            types_list.append(ELEMENT_TYPES[elem_name])
            block_elems.append(index)
            if instance is not None:
                instance_elems.append(index)

        elif mode in ("nset", "elset"):
            set_rows.append([p.strip() for p in stripped.split(",")])
        elif mode == "system":
            system_rows.append([p.strip() for p in stripped.split(",")])

    close_block()

    if unknown_types:
        warnings.warn(
            f".inp: unrecognised element type(s) {sorted(unknown_types)};"
            " their blocks are skipped.",
            stacklevel=2,
        )
    if unknown_instances:
        warnings.warn(
            f".inp: set(s) name the instance(s) {sorted(unknown_instances)},"
            " which the deck never defines; they were resolved against the"
            " numbering in force instead.",
            stacklevel=2,
        )
    if repeated_elems:
        warnings.warn(
            f".inp: {repeated_elems} element id(s) are defined twice; a set"
            " naming one reaches the last definition.",
            stacklevel=2,
        )

    if not coords:
        raise CodecError(".inp: no *Node section found.")

    n_verts = len(coords) // 3
    vertices = np.array(coords, dtype=np.float64).reshape(n_verts, 3)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_tags=vertex_tags,
        element_tags=element_tags,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write PolyData to Abaqus ``.inp`` ASCII format.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.inp`` path.
    **opts
        ``element_type``: a mapping from polyxios element name to the Abaqus
        card to write for it, so a deck can ask for ``C3D8R`` where the mesh
        only says ``hexahedron``. A card this codec knows has to be one that
        holds that element; one it does not know is written as asked. Any
        other option is warned about and ignored.

    Raises
    ------
    CodecError
        If an element type id is unknown, or has no Abaqus write mapping, or
        ``element_type`` is not a mapping of strings, names a card that holds
        another element, or spells two element types with one card.

    Notes
    -----
    ``vertex_tags`` and ``element_tags`` are written as ``*Nset`` and
    ``*Elset`` cards, so a boundary named on the way in is still named on the
    way out. Elements are grouped by type, one ``*Element`` card per group.
    """
    overrides = opts.pop("element_type", None) or {}
    if opts:
        warnings.warn(
            f".inp write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )
    if not isinstance(overrides, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()
    ):
        raise CodecError(
            ".inp: element_type= takes a mapping from polyxios element name to"
            " Abaqus card, such as {'hexahedron': 'C3D8R'}."
        )

    n_elems = len(poly.element_types)

    groups: dict[str, list[int]] = {}
    used: set[str] = set()
    # Which element type each card was given to, so one card asked to hold two
    # is caught before a block of rows of two lengths is written under it.
    holder_of: dict[str, str] = {}
    for i in range(n_elems):
        type_id = int(poly.element_types[i])
        name = ELEMENT_TYPES_INV.get(type_id)
        if name is None:
            raise CodecError(f".inp: unknown element type id {type_id}")
        if name not in _POLYXIOS_TO_INP:
            raise CodecError(f".inp: no write mapping for element type '{name}'")
        inp_type = overrides.get(name, _POLYXIOS_TO_INP[name])
        if name not in used:
            used.add(name)
            _check_override(name, inp_type)
            held = holder_of.setdefault(inp_type, name)
            if held != name:
                raise CodecError(
                    f".inp: element_type= writes both '{held}' and '{name}' as"
                    f" '{inp_type}'; one card cannot hold two element types."
                )
        groups.setdefault(inp_type, []).append(i)

    unused = sorted(set(overrides) - used)
    if unused:
        # An override the mesh has no element for silently writes the deck
        # the caller did not ask for, usually because the name is misspelt.
        warnings.warn(
            f".inp: element_type= names {unused}, which this mesh has no"
            " elements of; ignored.",
            stacklevel=2,
        )

    n_verts = poly.vertices.shape[0]
    lines: list[str] = [
        "*Heading",
        "** exported by polyxios",
        "*Node",
    ]
    lines.extend(
        f"{i + 1}, {v[0]:.10g}, {v[1]:.10g}, {v[2]:.10g}"
        for i, v in enumerate(poly.vertices)
    )

    # Element ids are handed out as the groups are written, so a set naming an
    # element has to be resolved against the numbering, not the mesh order. A
    # column rather than a dictionary: a set resolves its whole membership in
    # one gather, and 0 marks an element that never reached the file.
    elem_ids = np.zeros(n_elems, dtype=np.int64)
    elem_id = 0
    for inp_type, indices in groups.items():
        lines.append(f"*Element, type={inp_type}")
        for ei in indices:
            elem_id += 1
            elem_ids[ei] = elem_id
            s, e = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
            node_str = ", ".join(
                str(poly.connectivity[s + j] + 1) for j in range(e - s)
            )
            lines.append(f"{elem_id}, {node_str}")

    lines.extend(_set_cards(poly.vertex_tags, "Nset", "nset", None, n_verts, "node"))
    lines.extend(
        _set_cards(poly.element_tags, "Elset", "elset", elem_ids, n_elems, "element")
    )

    lines.append("")
    write_text(path, "\n".join(lines))


def _check_override(name: str, card: str) -> None:
    """Refuse a card that cannot hold the element it was asked to spell.

    Parameters
    ----------
    name
        The polyxios element type being written.
    card
        The Abaqus card ``element_type=`` asked for, or the codec's own.

    Raises
    ------
    CodecError
        When the card is not a bare word, or when it is a card this codec
        knows and that card holds a different number of nodes - a deck
        spelling eight nodes under a twenty-node card is one no reader loads.
    """
    if not card.strip() or any(ch in card for ch in ", \t\n*="):
        raise CodecError(
            f".inp: element_type= gives '{name}' the card {card!r}, which is"
            " not a bare card name."
        )
    known = _inp_type_info(card)
    if known is None:
        # A card outside the table is the caller's own business: they may be
        # writing for a solver this codec does not read back.
        return
    if known[0] != name:
        raise CodecError(
            f".inp: element_type= writes '{name}' as '{card}', which is a"
            f" {known[1]}-node '{known[0]}' card."
        )


def _safe_set_name(name: str) -> str:
    """Return a set name a card can carry, or an empty string for none.

    Parameters
    ----------
    name
        The tag group's name.

    Returns
    -------
    str
        The name with every character that would end the card folded to an
        underscore, or ``""`` when nothing is left.

    Notes
    -----
    A comma opens the next parameter, an ``=`` splits the one being written,
    a ``*`` opens a keyword card and a line break ends the card outright, so
    a name carrying any of them writes a deck no reader parses.
    """
    folded = "".join("_" if ch in _NAME_UNSAFE or ch < " " else ch for ch in name)
    return folded.strip()


def _set_cards(
    tags: dict[str, np.ndarray] | None,
    keyword: str,
    param: str,
    ident: np.ndarray | None,
    count: int,
    what: str,
) -> list[str]:
    """Return the ``*Nset`` / ``*Elset`` cards a tag dictionary spells.

    Parameters
    ----------
    tags
        Named index arrays.
    keyword
        Card keyword to write, ``Nset`` or ``Elset``.
    param
        Parameter naming the set on that card.
    ident
        The 1-based id written for each index, 0 for an index that never
        reached the file; None when the ids are the indices themselves, as a
        node's is.
    count
        How many nodes or elements the mesh holds, which bounds an index.
    what
        ``node`` or ``element``, named in the warnings.

    Returns
    -------
    list of str
        One card per tag, its ids wrapped 16 to a line as Abaqus expects.

    Notes
    -----
    Nothing checks a tag group on the way in, so a group may hold floats or an
    index past the end of a mesh it was not built for. Either would spell an
    id no ``*Node`` card defines and a deck Abaqus refuses to load, so they
    are dropped and reported rather than written; a float column is refused
    whole rather than rounded, since rounding an index moves a label onto
    another entity.
    """
    lines: list[str] = []
    unreachable: set[str] = set()
    unwritten: set[str] = set()
    nameless: set[str] = set()
    for name, members in (tags or {}).items():
        picked = member_indices(members, count)
        if picked.size != np.asarray(members).size:
            unreachable.add(name)
        ids = picked + 1 if ident is None else ident[picked]
        kept = np.unique(ids[ids > 0])
        if kept.size != ids.size:
            unwritten.add(name)
        if not kept.size:
            continue
        safe = _safe_set_name(name)
        if not safe:
            nameless.add(name)
            continue
        lines.append(f"*{keyword}, {param}={safe}")
        text = [str(i) for i in kept.tolist()]
        lines.extend(
            ", ".join(text[start : start + 16]) for start in range(0, len(text), 16)
        )
    if unreachable:
        warnings.warn(
            f".inp: {what} tag group(s) {sorted(unreachable)} name members that"
            f" index no {what} of this mesh; those members were dropped.",
            stacklevel=3,
        )
    if unwritten:
        warnings.warn(
            f".inp: {what} tag group(s) {sorted(unwritten)} name {what}(s) that"
            " were not written, so no card defines them; those members were"
            " dropped.",
            stacklevel=3,
        )
    if nameless:
        warnings.warn(
            f".inp: {what} tag group(s) {sorted(nameless)} spell no name a"
            " card can carry; they were not written.",
            stacklevel=3,
        )
    return lines
