.. _format-stl:

STL
===

.. rst-class:: px-badges

``.stl`` ``read + write`` ``lazy: binary only``

Summary of the specification
----------------------------

STL describes a solid as an unordered set of triangles - a triangle soup with no vertex sharing and no topology. Each facet carries its own normal and its three corner points. The ASCII flavour spells this out with ``solid`` / ``facet normal`` / ``outer loop`` / ``vertex`` keywords; the binary flavour is an 80-byte free-text header, a uint32 triangle count, then a fixed 50-byte record per facet (twelve float32 values plus a two-byte attribute field).

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
     - none - vertices are repeated per triangle

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

- Vertices are deduplicated on read so the mesh has shared topology - except in binary lazy mode, which returns them as-is, three per triangle, to avoid a second pass over the data.
- Facet normals are read but not trusted for orientation; they are kept as element attributes.
- The declared triangle count is validated against the real file size before memory is allocated.
- A binary facet's attribute word carries its colour, five bits per channel - but the two conventions in the wild disagree on both the channel order and the top bit. VisCAM and SolidView run blue in bits 0-4 and red in bits 10-14, and set the top bit to say the word holds a colour; Materialise Magics runs red low and blue high, and *clears* the top bit to say the facet owns its colour. Magics writes a ``COLOR=`` record in the 80-byte header, which is the only thing in the file that tells the two apart, so a header carrying one is read the Magics way and every other header the VisCAM way. VisCAM is also what polyxios writes. A zero word is no colour under either convention: read the Magics way to the letter it claims a facet colour of black, but it is what a writer that coloured nothing leaves behind, and an untouched file is not a black one.
- Facets that claim a colour land in ``element_attrs["colors"]`` as RGB in 0..1; the ones that do not stay ``NaN``, so an uncoloured facet in a coloured file is not read as black. A file where no facet claims a colour grows no attribute at all. On write a floating point column is taken as 0..1 and an integer one as 0..255, the way every image format counts.
- Colours are written back on a binary write. ASCII STL has no field for one, so an ASCII write reports that they were dropped rather than losing them quietly.

.. seealso::

   :doc:`index` - the full format table.
