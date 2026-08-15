.. _format-meshb:

Medit binary
============

.. rst-class:: px-badges

``.meshb`` ``read + write`` ``lazy: binary only``

Summary of the specification
----------------------------

``.meshb`` is the binary form of the INRIA Medit mesh format. The file is a stream of keyword-indexed fields: an integer keyword code, the byte position of the next field, then that field's data. Codes identify the version and dimension first, then typed entity blocks — ``Vertices``, ``Edges``, ``Triangles``, ``Quadrilaterals``, ``Tetrahedra``, ``Hexahedra`` — each a count followed by fixed-width records of node indices plus a trailing reference (tag) integer. The version code fixes whether floats are 32- or 64-bit.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - layout
     - keyword code, next-field offset, then the field payload
   * - header codes
     - MeshVersionFormatted, Dimension
   * - entity blocks
     - Vertices, Edges, Triangles, Quadrilaterals, Tetrahedra, Hexahedra
   * - record shape
     - node indices (1-based) + one reference integer
   * - float width
     - set by the version code: 32-bit or 64-bit
   * - ascii sibling
     - .mesh — the same keywords in text form

.. rst-class:: px-speclink

`Read the full Medit binary specification ↗ <https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.meshb")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Binary bodies can be memory-mapped instead of loaded:

.. code-block:: python

    mesh = px.read("big.meshb", lazy=True)

Writing
-------

.. code-block:: python

    px.write(mesh, "out.meshb")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The trailing reference integer on each entity becomes an element tag, which is how Medit files carry surface and region labels.
- Field offsets are validated against the file size before allocation, so a truncated or hostile file raises instead of over-allocating.
- Binary bodies can be memory-mapped with ``lazy=True``.

.. seealso::

   :doc:`index` — the full format table.
