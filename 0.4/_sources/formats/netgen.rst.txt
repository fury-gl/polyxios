.. _format-netgen:

Netgen
======

.. rst-class:: px-badges

``.vol`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A Netgen ``.vol`` file opens with the word ``mesh3d`` and is then a sequence of named sections, each one a keyword line, a count line, and that many records. The leading fields differ per section and are what say which element an index belongs to: a surface element spells ``surfnr bcnr domin domout np`` before its nodes, while a volume element spells ``matnr np``, so the node list starts at a different column in each. ``np`` is stated per record rather than fixed per section, because one section holds every element of its dimension whatever its order - a six-node triangle can sit next to a three-node one. The ``…gi`` and ``…uv`` spellings of the element sections carry extra geometry or parameter values after the nodes. ``materials``, ``bcnames``, ``cd2names`` and ``cd3names`` map the numeric indices back to names, and the file closes with ``endmesh``.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - mesh3d, then dimension and geomtype
   * - sections
     - points, edgesegments, surfaceelements, volumeelements, plus …gi / …uv spellings
   * - surface record
     - surfnr bcnr domin domout np <nodes>
   * - volume record
     - matnr np <nodes>
   * - indices
     - 1-based
   * - name sections
     - materials, bcnames, cd2names, cd3names
   * - terminator
     - endmesh

.. rst-class:: px-speclink

`Netgen / NGSolve project ↗ <https://github.com/NGSolve/netgen>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("cube.vol")
    mesh.element_tags      # bcnr / matnr indices, by name where the file names them

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vol")
    px.write(mesh, "out.vol", float_fmt=".17g")   # bit-exact coordinates

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: px-spec-table

   * - option
     - meaning
   * - ``float_fmt``
     - ASCII coordinate format specifier. Defaults to ``.10g``, which does not name a float64 exactly; pass ``.17g`` for a bit-exact round trip, at the cost of file size.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Elements are written section by section in the order the format fixes - faces, cells, edges, then point elements - so an element's index changes unless the mesh already lay in that order.
- Node references are 1-based in the file, 0-based in the :class:`~polyxios.PolyData`; every type whose node order differs from polyxios's is permuted both ways.
- Sections polyxios carries no home for - ``facedescriptors``, ``identifications``, ``face_colours``, the ``singular_*`` family - are stepped over by name, and a non-empty one says how many records it dropped rather than reading as a section nobody recognised.
- Index ``0`` means "unset" and is dropped rather than becoming a tag. A ``bc_<n>``-style tag name keeps its number, unless that number is negative or past ``int64``, which the format cannot spell.
- A tag holding elements of more than one dimension is written into each dimension's section, and comes back split one tag per dimension, because the sections number their indices independently.
- A ``dimension 2`` file's points are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact. Netgen writes ``mesh3d`` whatever it is handed, so the flag does not come back through a ``.vol`` round trip - but it does carry the plane into a format that can spell one, so a 2-D ``.vol`` written as SU2 lands as ``NDIME= 2``.
- Sections polyxios has no home for - ``identifications``, ``face_colours``, the ``singular_*`` family - are stepped over, with a warning when one is not empty.
- ``lazy=True`` warns and loads eagerly; the format is ASCII.

.. seealso::

   :doc:`index` - the full format table.
