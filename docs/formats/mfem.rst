.. _format-mfem:

MFEM mesh
=========

.. rst-class:: px-badges

``.mesh`` ``read + write`` ``eager``

Summary of the specification
----------------------------

An MFEM mesh file opens with a ``MFEM mesh v1.0`` version line and is then a sequence of named sections, each a keyword on its own line followed by a count and that many records. ``dimension`` fixes whether coordinates are written with two or three components. ``elements`` lists one element per line as an attribute, a geometry type code and its 0-based node indices; the codes run ``0`` point, ``1`` segment, ``2`` triangle, ``3`` quadrilateral, ``4`` tetrahedron, ``5`` hexahedron, ``6`` wedge, ``7`` pyramid. ``boundary`` uses the same record shape for the boundary elements, and ``vertices`` gives a count, a dimension and then one coordinate line per vertex. The format also has INLINE and NURBS variants that carry a parametric recipe rather than an explicit mesh.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - MFEM mesh v1.0
   * - sections
     - dimension, elements, boundary, vertices
   * - element record
     - <attribute> <geometry code> <node indices>
   * - geometry codes
     - 0 point, 1 segment, 2 triangle, 3 quad, 4 tet, 5 hex, 6 wedge, 7 pyramid
   * - indices
     - 0-based
   * - variants
     - INLINE (parametric recipe), NURBS (spline patches)

.. rst-class:: px-speclink

`Read the full MFEM mesh specification ↗ <https://mfem.org/mesh-format-v1.0/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("beam.mesh")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.mesh")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- INLINE and NURBS meshes raise :class:`~polyxios.exceptions.UnsupportedFormatError`; they store a recipe rather than an explicit mesh.
- A 2-D file's coordinates are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact. The written ``dimension`` follows it, and a mesh whose ``z`` column is entirely zero goes out as 2-D whether it was flagged or not. A flagged mesh whose vertices have since left the plane is written in three with a warning, and a flat mesh of solid cells keeps three columns whatever the flag says - MFEM reads exactly as many coordinates per vertex as the block declares, and a tetrahedron of two-coordinate vertices is not a cell.
- Coordinates are written with ``.10g``, which does not name a float64 exactly - a round trip differs in the last few digits.
- MFEM names eight geometries and no higher-order one. An element it has no geometry for - a ``quadratic_tetra``, a ``polygon`` - is skipped on write with a warning naming its type, and the declared element count drops with it. Writing one under another geometry's code would leave its extra nodes to be read as the record that follows, which costs every element after it rather than the one that did not fit.

.. seealso::

   :doc:`index` - the full format table.
