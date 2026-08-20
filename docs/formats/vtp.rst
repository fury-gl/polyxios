.. _format-vtp:

VTK PolyData
============

.. rst-class:: px-badges

``.vtp`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.vtp`` is the VTK XML format for surface and curve geometry. A ``<Piece>`` declares its point and cell counts as attributes, then carries ``<Points>`` plus up to four cell containers - ``<Verts>``, ``<Lines>``, ``<Strips>`` and ``<Polys>`` - each expressed as a ``connectivity`` array and an ``offsets`` array rather than the legacy size-prefixed lists. Attributes travel in ``<PointData>`` and ``<CellData>`` with the same inline, base64 or appended storage choices as the other XML formats.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <VTKFile type="PolyData">
   * - cell containers
     - Verts, Lines, Strips, Polys
   * - connectivity
     - paired connectivity + offsets DataArrays (0-based)
   * - storage
     - ascii, base64 binary, or appended raw block
   * - attributes
     - PointData / CellData, one named DataArray each

.. rst-class:: px-speclink

`Read the full VTK PolyData specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.vtp")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtp")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Triangle strips are expanded into individual triangles on read; writing emits polygons rather than re-striping.
- Each cell container becomes its own element group, so lines and polygons in one file stay distinguishable.
- A piece that declares points and does not deliver them raises :class:`~polyxios.exceptions.CodecError`; its cells would index points that are not there, and every later piece would be shifted by the count that never arrived.
- A point or cell array carried by only some of the pieces is dropped with a warning: joined short, its rows would sit against the wrong points from the second piece on.
- A ``Points`` array of a type that holds no numbers - ``type="String"``, or any type this reader does not know - raises :class:`~polyxios.exceptions.CodecError` naming the type.
- Attributes are written in the type their array is held in, so an integer identifier keeps every digit rather than being rounded through a double.

.. seealso::

   :doc:`index` - the full format table.
