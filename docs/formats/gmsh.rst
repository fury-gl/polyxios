.. _format-gmsh:

Gmsh
====

.. rst-class:: px-badges

``.msh`` ``read + write (v2)`` ``eager``

Summary of the specification
----------------------------

A Gmsh ``.msh`` file is a sequence of ``$Section`` / ``$EndSection`` blocks. ``$MeshFormat`` states the version, whether the file is ASCII or binary, and the size of a float. ``$Nodes`` lists node tags with coordinates, and ``$Elements`` lists elements with an integer type code and their node tags. ``$PhysicalNames`` maps physical group ids to names. Version 4.1 restructures nodes and elements into per-entity blocks with a leading count line, which is why the two revisions must be parsed differently.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - sections
     - $MeshFormat, $PhysicalNames, $Nodes, $Elements, $Periodic
   * - v2 nodes
     - tag x y z, one per line
   * - v4.1 nodes
     - entity blocks: numBlocks, then per-block tags and coordinates
   * - element codes
     - integer type ids (1 = 2-node line, 2 = triangle, 4 = tetrahedron, ...)
   * - groups
     - $PhysicalNames: dimension, tag, "name"

.. rst-class:: px-speclink

`Read the full Gmsh specification ↗ <https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.msh")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.msh")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- ASCII v2 and v4.1 are both read; writing emits v2, the revision every consumer still understands.
- Physical group names become element tags, so named boundaries and volumes survive the read.
- Node tags need not be contiguous; they are remapped to dense indices.
- ``$NodeData`` and ``$ElementData`` sections become ``vertex_attrs`` and ``element_attrs``, at any component count the file declares, and are written back. A field is scattered by the tag each row names, so one covering part of the mesh lands where it belongs and the rest stays ``NaN``. A row naming an entity the mesh does not hold - an element of a type this codec skipped, say - costs that row rather than the whole field.
- A data section declares 1, 3 or 9 components and Gmsh refuses any other count, so a field of another width is padded out to the next of the three with zero columns and the padding is reported. The reader stays lenient and takes whatever width a file declares.
- Gmsh numbers the mid-edge and face nodes of the higher-order elements by its own edge and face tables, which are not VTK's; they are permuted on the way in and back on the way out. Gmsh's 18-node prism becomes a ``biquadratic_quadratic_wedge``.
- Gmsh type 14, the 14-node pyramid, has no VTK equivalent and is skipped with a warning rather than reshaped into something else.

.. seealso::

   :doc:`index` - the full format table.
