.. _format-vtk:

VTK Legacy
==========

.. rst-class:: px-badges

``.vtk`` ``read + write`` ``lazy: binary only``

Summary of the specification
----------------------------

The legacy VTK format is a single-dataset serial file with a five-line ASCII preamble - version banner, title, ``ASCII`` or ``BINARY`` data mode, and a ``DATASET`` keyword naming the geometry type. What follows depends on that type: an unstructured grid lists ``POINTS``, then ``CELLS`` as connectivity lists prefixed by their point count, then a parallel ``CELL_TYPES`` array of integer type codes. Point and cell attributes arrive afterwards in ``POINT_DATA`` / ``CELL_DATA`` sections as named ``SCALARS``, ``VECTORS`` or ``FIELD`` arrays. In ``BINARY`` mode the arrays are raw big-endian values packed straight after their declaration line.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - banner
     - # vtk DataFile Version x.y
   * - data mode
     - ASCII or BINARY (binary payload is big-endian)
   * - dataset types
     - STRUCTURED_POINTS, STRUCTURED_GRID, RECTILINEAR_GRID, POLYDATA, UNSTRUCTURED_GRID, FIELD
   * - connectivity
     - CELLS <n> <size> followed by CELL_TYPES <n> integer codes
   * - attributes
     - POINT_DATA / CELL_DATA with SCALARS, COLOR_SCALARS, VECTORS, NORMALS, TENSORS, TEXTURE_COORDINATES, FIELD

.. rst-class:: px-speclink

`Read the full VTK Legacy specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.vtk")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Binary bodies can be memory-mapped instead of loaded:

.. code-block:: python

    mesh = px.read("big.vtk", lazy=True)

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtk")

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
     - Write a BINARY body instead of ASCII.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A tag group travels as one ``POINT_DATA`` or ``CELL_DATA`` column of ones and zeros named ``polyxios_tag_<group>``, so an element in two groups is named by both columns. A legacy header names its array in a whitespace-separated field and nothing in the format escapes one, so a group whose name holds whitespace is dropped with a warning rather than written as a name and a stray token.
- A ``FIELD FieldData`` block between the ``DATASET`` line and the geometry belongs to the mesh rather than to its points or cells, and is read into ``global_attrs``; ``write`` puts one back there. Unlike a point or cell array, which is written as a double, a field array keeps the type it is held in, so an integer comes home an integer. A ``FIELD`` inside a ``POINT_DATA`` or ``CELL_DATA`` section still names arrays over the points or the cells, and is read as attributes. A ``STRUCTURED_POINTS``, ``RECTILINEAR_GRID`` or ``STRUCTURED_GRID`` file reads a dataset-level block too, which is where VTK's own writer puts a time value.
- A value no numeric array holds - a string, a mapping - is dropped with a warning naming the key; the XML family's ``<FieldData>`` holds text and this block does not. The ``vtk_*`` grid entries a structured read recorded do travel in the block: this writer spells an ``UNSTRUCTURED_GRID`` and rebuilds no grid, so holding them back would drop them without a word. A structured read takes the grid it rebuilt over anything a field block names, so carrying them costs the next read nothing.
- ``POINTS`` and a v4.2 ``CELLS`` block are a run of numbers the header counts, not a line apiece. One row to a vertex or a cell is what VTK's own writer emits and what this reads first; a block wrapped some other way is read again as that run, so a file no reader of rows could take is read as the mesh it holds.
- Binary files can be memory-mapped with ``lazy=True``; ASCII files must be parsed end to end before any value is available.
- Cell type codes are mapped to polyxios element types, so a file mixing triangles, quads and tetrahedra keeps every group separate.
- Point and cell data arrays are carried through as named vertex and element attributes rather than being dropped on read.
- ``SCALARS``, ``VECTORS``, ``NORMALS``, ``TENSORS``, ``COLOR_SCALARS``, ``TEXTURE_COORDINATES`` and ``FIELD`` sections are all read, in every dataset type - unstructured, polydata, structured points, structured grid and rectilinear grid alike. A ``LOOKUP_TABLE`` definition is a palette rather than a value per point, so it becomes no attribute, but it is counted past so the arrays after it are still found. A keyword outside that set stops the scan, and says so. ``COLOR_SCALARS`` is the one attribute whose type its own line does not name: one unsigned char per component in a binary file, a float in 0..1 in an ASCII one. The byte is scaled onto 0..1, so the same colour reads back the same from either flavour.
- An attribute section that declares more values than the file holds raises ``CodecError`` naming the array, rather than an ``IndexError`` naming nothing in ASCII or a reshape failure naming nothing in binary.
- ``STRUCTURED_POINTS`` keeps ``DIMENSIONS``, ``ORIGIN`` and ``SPACING`` in ``global_attrs`` (``vtk_dimensions``, ``vtk_origin``, ``vtk_spacing``); ``STRUCTURED_GRID`` and ``RECTILINEAR_GRID`` keep ``DIMENSIONS``. The points are expanded into an explicit array, so without those the grid behind them would be lost.
- Those ``vtk_*`` entries are read-only: ``write`` always emits an ``UNSTRUCTURED_GRID`` and does not consume them.
- ``CELL_DATA`` is read from the structured datasets as well as the unstructured ones. An array whose declared length matches neither the points nor the cells of the grid the header describes is dropped with a warning naming it, rather than reaching ``PolyData`` as a validation error about lengths.
- A structured grid extends along whichever axes its ``DIMENSIONS`` declare: ``3 1 3`` is a sheet of quads in the x-z plane, not a run of lines, and a column along ``y`` or ``z`` is indexed with its own stride.
- A ``DIMENSIONS`` of one is an axis the grid does not extend along; one of zero - or the negative a malformed header spells - is an axis with no point on it at all, and one such axis empties the grid, since the point count is a product. It holds no cells either: counting the other two axes' cells handed back quadrilaterals whose corners named points the file never laid out.
- VTK 5.1 cells - the default since VTK 9.0 - are read wherever they appear: ``CELLS`` in an unstructured grid and ``POLYGONS``, ``LINES``, ``VERTICES`` or ``TRIANGLE_STRIPS`` in polydata. The two numbers on such a line are the length of the ``OFFSETS`` array and the length of ``CONNECTIVITY``, so the mesh holds one cell fewer than the first of them; the offsets are counted up to the ``CONNECTIVITY`` keyword, so a file spelling that line either way is read. ``write(..., vtk_version="5.1")`` declares the offsets length, which is what VTK's own reader expects.
- A ``METADATA`` block - component names and information keys, written after every array by VTK 4.2 and later - is stepped over rather than read as an array. It is text even in a binary file, and it appears between the entries of a ``FIELD`` block as well as after a section.
- A ``METADATA`` block also sits inside a v5.1 ``CELLS`` section, between its offsets and its connectivity, and is stepped over there too.
- A ``DATASET FIELD`` file carries field arrays and no geometry. It reads as an empty :class:`~polyxios.PolyData` whose ``global_attrs`` hold the arrays, with a warning saying so.
- Which cell spelling a v5.1 ``CELLS`` section uses is decided by what follows the header, not by the version in the first line, so a file declaring a version this reader has never heard of is still read by what it holds.
- A ``RECTILINEAR_GRID`` takes its grid from its coordinate arrays: the points are their outer product, so a ``DIMENSIONS`` header that disagrees with them is warned about and ignored.
- A ``STRUCTURED_GRID`` carries an explicit ``POINTS`` array, which its ``DIMENSIONS`` cannot be reconciled against the way a rectilinear grid's coordinates can. When the two disagree the points are handed back without cells, with a warning naming both counts: the cells the header describes would index points the file does not hold.
- An attribute section is read by the count its own header declares, which is the only thing that says where one array ends and the next begins. An array that then covers no point or cell of the mesh is dropped with a warning naming it.
- The ``LOOKUP_TABLE`` line after a ``SCALARS`` section is optional, and a binary file without one is read as such rather than losing the head of its payload.
- A header missing a field, or spelling a count as something that is not a number, raises ``CodecError`` naming the line it is on - the byte offset, in a binary file. This covers the geometry headers - ``POINTS``, ``CELLS``, ``CELL_TYPES``, ``DIMENSIONS``, ``ORIGIN``, ``SPACING``, the coordinate arrays - as well as the attribute ones.
- A binary block is read as the type its header names, ``POINTS`` included: an integer point array is not read at the width of a float. A type name with no numpy equivalent raises ``CodecError`` naming it rather than being guessed at, since a guessed width reads numbers the file never held. An ASCII payload is text whatever its header calls it, so it is read either way.

.. seealso::

   :doc:`index` - the full format table.
