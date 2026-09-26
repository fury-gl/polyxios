.. _format-xyz:

ASCII point clouds: XYZ, PTS, PTX
=================================

.. rst-class:: px-badges

``.xyz`` ``.pts`` ``.ptx`` ``read + write`` ``lazy: no``

Summary of the specification
----------------------------

The plainest point-cloud files are columns of numbers, one point per line, and nearly every scanner, photogrammetry package and GIS tool exports one. ``.xyz`` has no header at all: three coordinates, then whatever the exporter added - an intensity, a colour as three integers in 0..255, a normal. Leica's ``.pts`` is the same with the point count on a first line and ``x y z intensity r g b`` rows, the intensity in -2048..2047. Leica's ``.ptx`` holds one or more scans, each a ten-line header - the grid's columns and rows, the scanner position and axes, and a 4 x 4 transform in row-vector convention - followed by ``rows x cols`` points of ``x y z intensity`` or ``x y z intensity r g b``, the intensity in 0..1, with ``0 0 0 0.5`` marking a grid cell the scanner did not hit.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - ``.xyz``
     - ``x y z [i] [r g b | nx ny nz] [nx ny nz]`` per line; ``#`` and ``//`` comments; an optional column-label line
   * - ``.pts``
     - point count on the first line, then ``x y z i r g b`` (or any ``.xyz`` row)
   * - ``.ptx``
     - per scan: ``cols``, ``rows``, position, three axes, four rows of a 4 x 4 transform, then ``rows x cols`` lines of ``x y z i [r g b]``
   * - separators
     - whitespace; tabs count, commas do not
   * - connectivity
     - none

.. rst-class:: px-speclink

`PTX reference ↗ <https://paulbourke.net/dataformats/ptx/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    cloud = px.read("scan.xyz")
    cloud.vertices                    # (n, 3)
    cloud.vertex_attrs["intensity"]   # when a fourth column is there
    cloud.vertex_attrs["colors"]      # (n, 3) floats in 0..1, from integers in 0..255
    cloud.vertex_attrs["normals"]     # (n, 3), from three columns that are not colours

    station = px.read("station.ptx")
    station.vertex_tags               # scan_0, scan_1, ... when the file holds several
    station.global_attrs["ptx_transforms"]   # (scans, 4, 4), already applied

The layout is told from the content, not the extension: a first line of one integer is a count, a ``.ptx`` scan header has a fixed shape, and everything else is bare columns. Columns past the coordinates are named by count and kind: with 1, 4 or 7 of them the first is ``intensity``; three (left) are ``colors`` when they are whole numbers in 0..255 with at least one above 1 and ``normals`` otherwise; six are ``colors`` and ``normals`` in whichever order the colour test tells, or one six-wide ``extra`` when neither half is a colour; any other width (2, 5, 8 or more) is kept whole as ``extra``, intensity and all. A ``.ptx`` file's scans are transformed by their own matrices into one frame, the invalid points dropped with a warning, each scan tagged when there are several, and the matrices and grid sizes kept in ``global_attrs`` as ``ptx_transforms`` (``(scans, 4, 4)``) and ``ptx_dimensions`` (``(scans, 2)``, a ``(rows, cols)`` pair per scan - the reverse of the header's ``cols`` then ``rows`` lines).

Writing
-------

.. code-block:: python

    px.write(cloud, "out.xyz")                     # columns as the mesh has them
    px.write(cloud, "out.pts")                     # count line first
    px.write(cloud, "out.ptx")                     # one scan, identity pose
    px.write(cloud, buffer, fmt=".xyz", variant="pts", float_fmt=".6f")

The columns are ``x y z``, then ``intensity`` when ``vertex_attrs`` holds it, then ``colors`` as three integers in 0..255 (from floats in 0..1, or integers already in that range), then ``normals``. ``.pts`` prefixes the count, ``.ptx`` writes one scan of all the points in a single row with the identity transform and an intensity of 1 where the mesh has none, and no normals. A nameless buffer, or a name with any other suffix, writes ``xyz`` columns unless ``variant=`` says otherwise.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Six columns are ambiguous by design - ``x y z r g b`` or ``x y z nx ny nz`` - and are told apart by whether the last three are whole numbers in 0..255. Columns of nothing but 0 and 1 go to the normal, since a flat scan spells ``0 0 1`` on every point and a colour that dark is all but black; a normal such as ``0 0 255`` still reads as a colour. Write it through a format with a header if that matters.
- Any vertex attribute other than ``intensity``, ``colors`` and ``normals`` has no column and is dropped with a warning naming it; no ``global_attrs`` entry has a line in any of the three layouts.
- Elements are not part of the format: a mesh's faces are dropped with a warning, and every file reads back with no elements at all.
- A ``.ptx`` point at exactly the origin with an intensity of exactly 0.5 is the format's invalid marker and is dropped on read, so it cannot be written and read back.
- A UTF-8 byte order mark before the first row is skipped. A row that starts with a number is data whatever follows it, so a trailing comment on a row is refused rather than the row taken for a column header.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`: a number spelled as text has to be parsed before it is a number. The body is handed to numpy in one pass, and only split into lines when a comment mark is in it, so a file without one is never held twice.

.. seealso::

   :doc:`pcd` - the binary point-cloud format PCL, ROS and Open3D exchange.
   :doc:`index` - the full format table.
