.. _format-su2:

SU2
===

.. rst-class:: px-badges

``.su2`` ``read + write`` ``eager``

Summary of the specification
----------------------------

The SU2 native mesh is plain ASCII, keyed by uppercase tokens on their own lines. ``NDIME=`` gives the dimension, ``NELEM=`` precedes the volume element list - one line per element: a VTK element type code, its node indices, then the element index - and ``NPOIN=`` precedes the coordinates. Boundaries are declared per marker: ``NMARK=`` counts them, then each ``MARKER_TAG=`` names one and ``MARKER_ELEMS=`` lists its surface elements in the same code-then-nodes form.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - dimension
     - NDIME= 2 or 3
   * - elements
     - NELEM= then <vtk_type> <nodes...> <index>
   * - type codes
     - 3 line, 5 triangle, 9 quad, 10 tetra, 12 hexa, 13 prism, 14 pyramid
   * - points
     - NPOIN= then coordinates, one point per line
   * - boundaries
     - NMARK=, MARKER_TAG=<name>, MARKER_ELEMS=<n>

.. rst-class:: px-speclink

`Read the full SU2 specification ↗ <https://su2code.github.io/docs_v7/Mesh-File/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.su2")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.su2")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Boundary marker names become element tags, so named inlets, outlets and walls survive a round trip.
- A 2-D file's coordinates are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact. ``NDIME`` follows it: a flagged flat mesh, or any mesh with no z extent, goes out as 2. A mesh whose vertices have since left it is written in three with a warning.
- The trailing per-element index is read but not trusted for ordering; elements keep file order.

.. seealso::

   :doc:`index` - the full format table.
