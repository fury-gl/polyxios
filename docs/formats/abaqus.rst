.. _format-abaqus:

Abaqus
======

.. rst-class:: px-badges

``.inp`` ``read + write`` ``eager``

Summary of the specification
----------------------------

An Abaqus input deck is a keyword-driven text file. Lines beginning with ``*`` introduce a keyword with comma-separated parameters; the data lines that follow belong to it until the next keyword. A mesh needs only two: ``*NODE``, listing an id and its coordinates per line, and ``*ELEMENT, TYPE=...``, listing an element id and its node ids. Named collections are declared with ``*NSET`` and ``*ELSET``, and ``**`` starts a comment.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - keywords
     - ``*NODE``, ``*ELEMENT``, ``*NSET``, ``*ELSET``, ``*PART``, ``*INCLUDE``
   * - node line
     - id, x, y, z
   * - element line
     - id, n1, n2, ... (count implied by TYPE)
   * - element types
     - C3D4, C3D8, CPS3, S4R, ... mapped to polyxios element types
   * - comments
     - lines starting with ``**``; keywords are case-insensitive

.. rst-class:: px-speclink

`Read the full Abaqus specification ↗ <https://help.3ds.com/2023/english/dssimulia_established/SIMACAEKEYRefMap/simakey-c-gen.htm>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.inp")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.inp")

``element_type=`` maps a polyxios element name to the Abaqus card written for
it, so a deck can ask for reduced integration where the mesh only says
``hexahedron``:

.. code-block:: python

    px.write(mesh, "out.inp", element_type={"hexahedron": "C3D8R"})

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Node ids need not be contiguous or sorted; they are remapped to a dense 0-based index. Several ``*NODE`` blocks accumulate, and a repeated id restates that node rather than adding another.
- ``*NSET`` and ``*ELSET`` names become vertex and element tags, whether declared on the block itself or as a standalone card, with or without ``GENERATE``; an entity in several sets stays in all of them. Abaqus matches a set name without regard to case, so a body naming ``TOP`` reaches the set declared as ``Top``, and the same name in two cases is one set.
- Element cards are matched on their base name, so the modifier suffixes - ``R`` reduced integration, ``H`` hybrid, ``I``, ``M``, ``T``, ``P``, and the shell degree-of-freedom numbers - resolve to the same element: ``CPS8R`` reads as ``CPS8``.
- ``*SYSTEM`` transforms every node block that follows it until the next ``*SYSTEM``, which with no data lines restores the global system.
- ``*INCLUDE`` is resolved against the including file's directory. A path that leaves that directory, or nesting deeper than eight files, is refused - an input deck is untrusted input. A deck read from a buffer has no directory, so it cannot use ``*INCLUDE``.
- Every ``*PART`` / ``*INSTANCE`` is merged into one mesh, each read under its own node numbering and tagged by its name. An instance that only places its part carries no nodes of its own and shares the part's numbering.
- A set carrying ``INSTANCE=`` is numbered by that instance rather than by whatever numbering is in force where the card sits, which is what lets an assembly keep its sets outside the instance they name. One naming an instance the deck never defines is reported.
- A ``GENERATE`` range wider than the deck has ids is resolved by walking the ids rather than the range: the card names two numbers and nothing bounds their distance.
- A ``*PART`` sets the deck's own numbering aside rather than replacing it, so a set out past ``*End Part`` still reaches the nodes the deck defined before it.
- On write, a tag member that indexes no node or element of the mesh is dropped and reported: a set naming an id no card defines is a deck Abaqus refuses to load. A float column is refused whole rather than rounded, since rounding an index moves a label onto another entity.
- Analysis keywords (steps, materials, boundary conditions) are skipped rather than treated as errors, and an unrecognised element card is warned about and skipped rather than failing the read.
- A node card spells the third coordinate only in a 3-D model, so a deck where none does is a plane: the vertices are padded with a zero z and ``global_attrs["was_2d"]`` records the fact, which is what writes the cards back out two columns wide. A mesh whose vertices have since left the plane is written in three with a warning.

.. seealso::

   :doc:`index` - the full format table.
