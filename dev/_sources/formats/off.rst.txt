.. _format-off:

OFF
===

.. rst-class:: px-badges

``.off`` ``read + write`` ``eager``

Summary of the specification
----------------------------

The Object File Format opens with an ``OFF`` magic line, optionally prefixed by variant letters, then a counts line giving the number of vertices, faces and edges (the edge count is conventionally ignored). Vertex coordinates follow one per line, then one line per face: a vertex count followed by that many 0-based indices, optionally trailed by per-face colour components. The ``ST``, ``C``, ``N`` and ``4`` prefixes declare that vertices additionally carry texture coordinates, colour, normals or a fourth homogeneous component, in that fixed order.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - magic
     - [ST][C][N][4][n]OFF, optionally followed by BINARY
   * - counts line
     - nvertices nfaces nedges (edges ignored)
   * - faces
     - <k> i0 i1 ... ik-1, 0-based, optional trailing colour
   * - variants
     - ST = texture coords, C = colour, N = normals, 4 = homogeneous
   * - binary form
     - big-endian, IEEE floats and 32-bit ints

.. rst-class:: px-speclink

`Read the full OFF specification ↗ <https://segeval.cs.princeton.edu/public/off_format.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.off")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.off")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- ASCII and big-endian binary OFF are both read; writing emits ASCII.
- ``ST`` / ``C`` / ``N`` variant data is mapped onto vertex and face attributes instead of being discarded.
- The edge count on the counts line is ignored, as the format intends — no error is raised when it disagrees with the face list.

.. seealso::

   :doc:`index` — the full format table.
