.. _format-flac3d:

FLAC3D
======

.. rst-class:: px-badges

``.f3grid`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.f3grid`` is Itasca's ASCII grid exchange file. Records are keyword-led: ``G`` (or ``GRIDPOINT``) lines declare a gridpoint id and its coordinates, ``Z`` lines declare a zone with a shape keyword (``B8``, ``W6``, ``P5``, ``T4``) and its gridpoint ids, and ``F`` lines declare surface faces. ``ZGROUP`` and ``FGROUP`` records name a group and list the zone or face ids that belong to it, which is how regions and boundaries are carried.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - gridpoints
     - G <id> <x> <y> <z>
   * - zones
     - Z <shape> <id> <gp ids...> — B8 hex, W6 wedge, P5 pyramid, T4 tet
   * - faces
     - F <shape> <id> <gp ids...>
   * - groups
     - ZGROUP / FGROUP <name> followed by member ids
   * - comments
     - lines starting with *

.. rst-class:: px-speclink

`Read the full FLAC3D specification ↗ <https://docs.itascacg.com/flac3d700/common/docproject/source/manual/program_guide/mechanics/otherinputs/gridformat.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.f3grid")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.f3grid")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Both zones and faces are read, so a file's volume mesh and its boundary surface arrive as separate element groups.
- ``ZGROUP`` and ``FGROUP`` names become element tags; a zone in several groups stays in all of them.
- Gridpoint ids are sparse in practice and are remapped to a dense 0-based index on read.

.. seealso::

   :doc:`index` — the full format table.
