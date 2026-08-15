.. _format-stl:

STL
===

.. rst-class:: px-badges

``.stl`` ``read + write`` ``lazy: binary only``

Summary of the specification
----------------------------

STL describes a solid as an unordered set of triangles — a triangle soup with no vertex sharing and no topology. Each facet carries its own normal and its three corner points. The ASCII flavour spells this out with ``solid`` / ``facet normal`` / ``outer loop`` / ``vertex`` keywords; the binary flavour is an 80-byte free-text header, a uint32 triangle count, then a fixed 50-byte record per facet (twelve float32 values plus a two-byte attribute field).

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - ascii form
     - solid <name> ... facet normal / outer loop / vertex x3 / endloop / endfacet
   * - binary form
     - 80-byte header, uint32 count, then 50 bytes per facet
   * - binary record
     - 3 float32 normal + 9 float32 vertices + uint16 attribute byte count
   * - byte order
     - little-endian
   * - topology
     - none — vertices are repeated per triangle

.. rst-class:: px-speclink

`Read the full STL specification ↗ <https://www.fabbers.com/tech/STL_Format>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.stl")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Binary bodies can be memory-mapped instead of loaded:

.. code-block:: python

    mesh = px.read("big.stl", lazy=True)

Writing
-------

.. code-block:: python

    px.write(mesh, "out.stl")

Format-specific options:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``binary``
     - ``False``
     - Write the 50-byte-per-facet binary form instead of ASCII.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Vertices are deduplicated on read so the mesh has shared topology — except in binary lazy mode, which returns them as-is, three per triangle, to avoid a second pass over the data.
- Facet normals are read but not trusted for orientation; they are kept as element attributes.
- The declared triangle count is validated against the real file size before memory is allocated.

.. seealso::

   :doc:`index` — the full format table.
