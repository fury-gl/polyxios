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

A textured mesh keeps both sides of a seam when the vertices are split:

.. code-block:: python

    mesh = px.read("model.obj", split_seams=True)
    mesh.vertex_attrs["texcoords"]   # one uv per vertex, seams included

Writing
-------

.. code-block:: python

    px.write(mesh, "out.obj")

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``split_seams``
     - ``False``
     - On read: copy a vertex once per distinct texture coordinate and normal
       its corners name, so a seam keeps both uvs and a hard edge both
       normals. Adds vertices; removes none.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Negative (relative) indices are resolved against the vertex count at the point the face appears, not the final count.
- An index naming a record the file has not declared raises ``CodecError`` naming the line, rather than wrapping around into another vertex.
- Faces with more than four vertices are kept as polygons rather than being silently triangulated.
- Material and group keywords are read as element tags; ``.mtl`` files are not parsed.
- ``vt`` and ``vn`` are indexed per face corner, so a file may hold more of either than it holds vertices. polyxios stores one value per vertex in ``vertex_attrs['texcoords']`` and ``vertex_attrs['normals']``: a corner assigns to its vertex, and a vertex given two different values keeps the last and warns. Records nothing indexes are kept only when there is exactly one per vertex.
- ``split_seams=True`` keeps what that fold drops. A vertex is copied once per distinct pairing its corners name, the first corner keeping the index it had and each later one taking a fresh index off the end, so a vertex no face names stays where it was and a file whose corners agree reads exactly as it does without the option. The faces are untouched - only the indices inside them move - so the element count, the element order, ``element_attrs`` and ``element_tags`` are the same either way. Records nothing indexes are still counted against the ``v`` records the file wrote rather than the vertices the split left, and a copy takes what the vertex it came from takes, so a split neither drops such an attribute nor invents one out of a count that happens to match.
- A split mesh writes back with the copied positions repeated as their own ``v`` records, since the writer emits one ``vt``/``vn`` per vertex in vertex order. Do not run ``transforms.merge_duplicate_vertices`` on one: it welds the copies back by position and keeps a single survivor's uv, undoing the split.
- Records that cannot be lined up with the vertices leave the attribute out entirely, on write as well as on read: an attribute that does not hold one row per vertex is warned about and left out, rather than written as faces indexing records that are not in the file. A vertex no face names carries NaN in the array and is written back as zero, since ``vt nan nan`` is not a record another OBJ reader takes.
- A face belonging to no group is written after a bare ``g``, so it does not inherit the group of the face above it; a bare ``g`` on read clears the active groups rather than inventing a ``default`` tag.
- A ``v``, ``vn`` or ``vt`` record that does not carry the components its directive needs, or carries something that is not a number, raises ``CodecError`` naming the line. A ``vt`` may carry a third component - the depth of a volumetric texture - and ``texcoords`` keeps the two a surface uses.
- An ``f`` record is a flat ring of vertices and OBJ spells no other shape, so an element that is not one - a ``tetra``, a ``line`` - keeps its vertices and loses the type it was: it comes back a triangle at three vertices, a quad at four and a polygon otherwise. The elements are still written, and the types they lose are named in a warning.
- ``l`` and ``p`` records name geometry this codec has no element for. They are dropped, and the read says how many of each the file held rather than leaving the loss to be found later.
- ``element_attrs['material']`` is written as ``usemtl`` only when it holds one value per face; a shorter attribute is warned about and left out, the way an ill-fitting vertex attribute is.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`; the records are text and a face may name a vertex by a relative index, so nothing can be located without a parse.

.. seealso::

   :doc:`index` - the full format table.
