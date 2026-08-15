.. _format-obj:

Wavefront OBJ
=============

.. rst-class:: px-badges

``.obj`` ``read + write`` ``eager``

Summary of the specification
----------------------------

OBJ is a line-oriented ASCII format where each line is a keyword followed by whitespace-separated values: ``v`` for a geometric vertex, ``vn`` for a normal, ``vt`` for a texture coordinate, and ``f`` for a face. Face vertices are given as ``v``, ``v/vt``, ``v//vn`` or ``v/vt/vn`` triples, indices are **1-based**, and a negative index counts backwards from the most recently declared vertex. Faces may have any number of vertices, and material and grouping keywords (``usemtl``, ``g``, ``o``, ``s``) may appear anywhere.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - records
     - v, vn, vt, f, l, p, g, o, usemtl, mtllib
   * - indexing
     - 1-based; negative values are relative to the current vertex count
   * - face syntax
     - v | v/vt | v//vn | v/vt/vn
   * - polygons
     - arbitrary vertex count per face
   * - companion file
     - materials in a separate .mtl referenced by mtllib

.. rst-class:: px-speclink

`Read the full Wavefront OBJ specification ↗ <https://paulbourke.net/dataformats/obj/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.obj")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.obj")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Negative (relative) indices are resolved against the vertex count at the point the face appears, not the final count.
- Faces with more than four vertices are kept as polygons rather than being silently triangulated.
- Material and group keywords are read as element tags; ``.mtl`` files are not parsed.

.. seealso::

   :doc:`index` — the full format table.
