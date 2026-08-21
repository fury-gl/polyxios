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
- ``*NSET`` and ``*ELSET`` names become vertex and element tags, whether declared on the block itself or as a standalone card, with or without ``GENERATE``; an entity in several sets stays in all of them.
- Element cards are matched on their base name, so the modifier suffixes - ``R`` reduced integration, ``H`` hybrid, ``I``, ``M``, ``T``, ``P``, and the shell degree-of-freedom numbers - resolve to the same element: ``CPS8R`` reads as ``CPS8``.
- ``*SYSTEM`` transforms every node block that follows it until the next ``*SYSTEM``, which with no data lines restores the global system.
- ``*INCLUDE`` is resolved against the including file's directory. A path that leaves that directory, or nesting deeper than eight files, is refused - an input deck is untrusted input. A deck read from a buffer has no directory, so it cannot use ``*INCLUDE``.
- Every ``*PART`` / ``*INSTANCE`` is merged into one mesh, each read under its own node numbering and tagged by its name.
- Analysis keywords (steps, materials, boundary conditions) are skipped rather than treated as errors, and an unrecognised element card is warned about and skipped rather than failing the read.

.. seealso::

   :doc:`index` - the full format table.
