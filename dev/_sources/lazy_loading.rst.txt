Lazy loading
============

.. meta::
   :description: Read large mesh files in Python without copying them. polyxios maps a binary VTK, VTU, VTP, XDMF, PLY, PCD, Medit or splat file with lazy=True and hands back arrays that view the file's own bytes.


For a large binary mesh, pass ``lazy=True``. polyxios maps the file and,
wherever the file stores an array as one run of bytes in the shape the array
needs, hands back a view of the mapping instead of a copy: nothing is
decoded, the pages come in as they are touched, and the process holds the
mesh once - in the page cache - rather than once there and again in its own
memory.

.. code-block:: python

    mesh = px.read("huge.vtu", lazy=True)

    mesh.vertices.flags.writeable   # False: these are the file's bytes
    mesh.vertices.dtype             # whatever the file holds - float32 stays float32
    mesh.vertices[0]                # the first page comes in here

    coords = np.asarray(mesh.vertices, dtype=np.float64)   # a copy, when one is wanted

What comes back
---------------

A lazy array is read-only, and carries the dtype and byte order the file
holds: a ``Float32`` block stays ``float32``, a big-endian legacy VTK block
stays big-endian. An eager read converts to float64 coordinates and native
integer indices; a lazy read leaves that to you, since the conversion is the
copy you declined. ``np.asarray(arr, dtype=...)`` makes one when you need it,
and every NumPy operation reads a swapped or read-only array as it is.

Only an array the file stores in the shape :class:`~polyxios.PolyData` needs
can be a view. That rules out anything encoded (ASCII, base64, zlib), any
record that interleaves a count or a reference with the values, and any
index that counts from one. Where a format stores some arrays that way and
not others, the ones it does are views and the rest are built in memory -
the offsets of a uniform topology, the element types, a padded third
coordinate. The per-format table below says which is which.

Which formats
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 24 54
   :class: px-spec-table

   * - format
     - lazy arrays
     - notes
   * - ``.vtu`` ``.vtp``
     - vertices, connectivity, attributes
     - Raw appended section only, the layout VTK writes by default and
       ``write(..., appended=True)`` writes. Inline, base64 or compressed
       arrays raise. Offsets and element types are derived. Several pieces
       are joined by copying.
   * - ``.vtk``
     - vertices, attributes; cells in v5.1
     - Binary ``UNSTRUCTURED_GRID``. The v5.1 layout stores offsets and
       connectivity as two blocks and both are views; a v4.2 ``CELLS``
       block interleaves counts with indices and is decoded. Big-endian.
   * - ``.xdmf``
     - every array the sidecar stores
     - A ``Binary`` DataItem, or an HDF5 dataset stored contiguously
       without a filter, which is what ``write`` emits unless asked for
       ``compression``. Chunked, compressed or inline values raise.
   * - ``.splat``
     - everything
     - 32-byte records: float32 positions and every attribute are strided
       views. Nothing is decoded.
   * - ``.pcd``
     - vertices, every field
     - ``DATA binary`` only. Coordinates and normals are one strided view
       each when their three fields sit side by side in one type; split
       normals come back as three plain fields, split coordinates raise.
       Every other field is a view of its own column. A packed ``rgb`` is decoded
       into ``colors``. ``ascii`` and ``binary_compressed`` raise.
   * - ``.ply``
     - vertices, vertex properties
     - Binary only. Coordinates are one strided view when ``x``, ``y``,
       ``z`` sit side by side in one type. A face list is prefixed by its
       count, so faces are decoded.
   * - ``.meshb``
     - vertices
     - Each record is coordinates then a reference, so the coordinates are
       one strided view. Elements number vertices from one and are decoded.
   * - ``.stl``
     - none
     - A 50-byte record cannot be viewed as ``(n, 3)`` coordinates. Binary
       STL's ``lazy=True`` skips vertex deduplication instead, returning
       three vertices per triangle, and copies what it reads.
   * - ``.vti`` ``.vtr`` ``.vts`` ``.3mf`` ``.glb`` ``.obj`` ``.wkt``, HDF5
       formats, Exodus
     - none
     - ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`
       naming why: implicit points, a ZIP member, a scene, text, or a
       library that reads the arrays whole.
   * - the text formats
     - none
     - ``lazy=True`` warns and loads eagerly; a number spelled as text has
       to be parsed before it is a number.

What a mapping needs
--------------------

``mmap`` maps a file descriptor from byte zero, so a lazy read needs a real,
uncompressed file standing at its start: an ``io.BytesIO``, a handle
part-way into a file, or a gzipped one raises
:class:`~polyxios.exceptions.LazyReadError` naming the reason rather than
quietly loading eagerly. Binary STL's lazy mode only skips work, so it takes
a buffer or a compressed file like any other read.

The mapping lives as long as any array viewing it does, and goes when the
last one does. Nothing is closed behind your back and nothing needs closing.
A mapping does keep its own descriptor open for that long, so a thousand
lazy meshes held at once are a thousand open files; drop the arrays to
release them.

.. seealso::

   Each page under :doc:`formats/index` carries a badge saying what its
   codec maps, and the table on the :doc:`front page <index>` has a
   ``lazy`` column.
