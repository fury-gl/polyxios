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

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Node ids need not be contiguous or sorted; they are remapped to a dense 0-based index and the original ids are kept as vertex tags.
- ``*NSET`` and ``*ELSET`` names become vertex and element tags, and an entity in several sets stays in all of them.
- Analysis keywords (steps, materials, boundary conditions) are skipped rather than treated as errors.

.. seealso::

   :doc:`index` — the full format table.
