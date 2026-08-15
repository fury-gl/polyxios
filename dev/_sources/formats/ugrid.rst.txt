.. _format-ugrid:

UGRID (AFLR)
============

.. rst-class:: px-badges

``.ugrid`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A UGRID file opens with one header line of seven counts — nodes, triangles, quads, tetrahedra, pyramids, prisms, hexahedra — and then lists every section back to back, whitespace-separated. Nothing in the body says which section it belongs to, so the header counts are the only thing separating them and each has to be walked in the fixed order. The surface IDs in particular sit *between* the boundary faces and the volume cells rather than at the end of the file: reading them anywhere else shifts every volume element that follows. Node references are 1-based. The same format has binary variants spelled by an infix suffix — ``mesh.b8.ugrid``, ``mesh.lb8.ugrid``, ``mesh.r8.ugrid``.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - n_nodes n_tri n_quad n_tet n_pyramid n_prism n_hex
   * - body order
     - coordinates, tri, quad, tri IDs, quad IDs, tet, pyramid, prism, hex
   * - indices
     - 1-based
   * - surface IDs
     - one per boundary face, 0 means unmarked
   * - binary variants
     - .b8.ugrid, .lb8.ugrid, .r8.ugrid (not handled here)

.. rst-class:: px-speclink

`Read the full UGRID specification ↗ <https://www.simcenter.msstate.edu/software/documentation/ugrid/ugrid.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("wing.ugrid")
    mesh.element_tags      # boundary_<id>, one per distinct non-zero surface ID

Writing
-------

.. code-block:: python

    px.write(mesh, "out.ugrid")
    px.write(mesh, "out.ugrid", float_fmt=".17g")   # bit-exact coordinates

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: px-spec-table

   * - option
     - meaning
   * - ``float_fmt``
     - ASCII coordinate format specifier. Defaults to ``.10g``; pass ``.17g`` for a bit-exact round trip.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Binary variants are refused rather than parsed as text — by name where the name carries the infix, and by the NUL in the opening record where it does not.
- Elements are written in the order the format fixes — triangles, quads, tetrahedra, pyramids, prisms, hexahedra — so an element's index changes unless the mesh already lay in that order.
- The pyramid section is permuted in both directions; prisms (``wedge``) and hexahedra already match polyxios's node order and are read as they lie.
- ``boundary_<n>`` tag names keep their number. ``boundary_0`` and any other name are numbered from 1 instead, because ID 0 is the format's word for unmarked.
- Two tag names that would spell one ID would come back fused, so the later one is renumbered with a warning.
- ``lazy=True`` warns and loads eagerly; the ASCII flavour has no seekable structure.

.. seealso::

   :doc:`index` — the full format table.
