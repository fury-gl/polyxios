.. _format-permas:

PERMAS
======

.. rst-class:: px-badges

``.dato`` ``.post`` ``.dat`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A PERMAS deck is plain ASCII made of ``$KEYWORD`` records, each followed by the data lines it governs until the next ``$``. A ``!`` opens a comment, and a line whose first column is ``&`` continues the line before it, a data line or a ``$`` record alike. The model lives inside ``$ENTER COMPONENT`` … ``$EXIT COMPONENT``, whose ``$STRUCTURE`` section carries the mesh: ``$COOR`` lists nodes as ``id x y z``, ``$ELEMENT TYPE = <class>`` lists cells as ``id <nodes>``, and ``$NSET NAME = <name>`` / ``$ESET NAME = <name>`` name groups of node and element ids. A ``$COOR`` record may carry ``NSET =`` and an ``$ELEMENT`` record ``ESET =``, which put the entities they list into that set. Nodes and elements are numbered freely. ``$FIN`` closes the deck.

The element class is a solver element rather than a geometry: ``QUAD4``, ``SHELL4``, ``LOADA4`` and ``PLOTA4`` are all four-node quadrilaterals that differ in what the solver does with them. ``.dato`` is the model a run is fed, ``.post`` is what a run writes back with the structure section repeated, and ``.dat`` is used for the same deck.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - $KEYWORD records, each running to the next $
   * - node record
     - $COOR [NSET = name], then id x y z
   * - cell record
     - $ELEMENT TYPE = class [ESET = name], then id <nodes>
   * - indices
     - free numbering, 1-based by convention
   * - groups
     - $NSET NAME = …, $ESET NAME = …
   * - comments
     - ``!`` to end of line
   * - continuation
     - ``&`` in the first column

.. rst-class:: px-speclink

`PERMAS by INTES ↗ <https://www.intes.de/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.dato")
    mesh.vertex_tags                         # one entry per $NSET, plus a $COOR NSET =
    mesh.element_tags                        # one entry per $ESET, plus an $ELEMENT ESET =
    mesh.vertex_attrs["original_ids"]        # the node ids the deck spelled, when not 1..n
    mesh.global_attrs["permas_component"]    # the component name, when not DFLT_COMP

Writing
-------

.. code-block:: python

    px.write(mesh, "out.dato")
    px.write(mesh, "out.dato", element_type={"quad": "SHELL4", "tetra": "TET4"})

``element_type`` maps a polyxios element name to the PERMAS class to spell it with. Without it every geometry goes out as its plain structural class: ``TRIA3``, ``QUAD4``, ``TET4``, ``HEXE8``, ``PENTA6``, ``PYRA5`` and their higher-order counterparts. A class the codec knows to hold a different geometry is refused. Any other option is warned about and ignored.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Every class in the table reads as its geometry, so ``SHELL4`` and ``LOADA4`` both come back as a quad and, written again, both land as ``QUAD4``. The solver class is not remembered; pass ``element_type=`` to choose one on the way out. A block of a class outside the table - a spring, a mass, a rigid body - is skipped with a warning, and a set naming one of its elements drops that member.
- Node order for the higher-order classes follows a reference implementation, which agrees with polyxios's own for every type; no permutation is applied.
- Elements keep their file order, and the writer keeps the mesh's: it opens a new ``$ELEMENT TYPE =`` block wherever the geometry changes rather than sorting the mesh by type.
- Nodes and elements are renumbered from zero on the way in, and the deck's own ids are kept under ``original_ids`` when they say something the index does not - a file numbering ``1..n`` in order records nothing. The writer spells them back when it still can, and numbers from one otherwise. An id spelled twice, not positive or wider than 64 bits, or an element naming a node the deck never declares, is refused; an integer is digits only, so ``1_0`` is not one. A coordinate may carry a Fortran ``D`` exponent (``1.0D+00``). Only Cartesian coordinates are read: a ``$COOR`` block flagged ``CYL`` or ``SPH`` holds radii and angles in the solver's own convention and is refused rather than read as x y z. A mesh whose vertices carry two columns is written with a zero z; any other width is refused.
- A set may be spelled over several records, which add to it, and a member it repeats names its entity once. Members are written in the order the tag holds them, a repeat at its first spelling, so a set reads back as it was. A set with no ``NAME =`` is called ``nset_<n>`` / ``eset_<n>`` with a warning. Node and element sets are separate namespaces, so a tag name used for both is written under both.
- The ``$ENTER COMPONENT NAME =`` is kept under ``global_attrs["permas_component"]`` unless it is ``DFLT_COMP``, the name a writer spells on its own. A deck holding a second component is read down to the end of the first, with a warning.
- Nothing in the structure section carries per-entity data, so ``vertex_attrs``, ``element_attrs`` and the rest of ``global_attrs`` are not written. Every other record of a solver deck - ``$SYSTEM``, ``$MATERIAL``, ``$LOADING``, ``$SITUATION`` and the like - is passed over on read, as is a bare flag word on a record this codec does read. A file holding no ``$`` record at all is refused.
- A name that cannot sit on a record - one carrying whitespace, ``=``, ``!``, ``$`` or ``&`` - has those replaced by ``_``, and a repeat after that gets a ``_<n>`` suffix, each with a warning. An element of a geometry PERMAS has no class for, or whose node count does not match its type, is dropped with a warning.
- ``.dat`` is shared with Nastran and Tecplot and resolved by content: a file whose first record is an ``$ENTER`` section or a PERMAS structure keyword reads here. Writing to ``.dat`` needs ``fmt="dato"``.
- ``lazy=True`` warns and loads eagerly; the format is ASCII.

.. seealso::

   :doc:`index` - the full format table.
