Usage
=====

.. meta::
   :description: Read and write 3D meshes in Python with polyxios.read() and polyxios.write(). Covers basic I/O, format detection by extension, and format-specific write options.


Basic I/O
---------

.. code-block:: python

    import polyxios as px

    # Read any supported format
    mesh = px.read("brain.vtk")

    # Inspect
    print(mesh.vertices.shape)      # (n_verts, 3)
    print(len(mesh.element_types))  # number of elements

    # Write to a different format
    px.write(mesh, "brain.ply")
    px.write(mesh, "brain.vtp")

Files, buffers and streams
--------------------------

Anything with a ``read`` or a ``write`` works where a path does, so a mesh
never has to touch disk:

.. code-block:: python

    import io

    buf = io.BytesIO()
    px.write(mesh, buf, fmt=".ply")     # fmt= names the format

    buf.seek(0)
    same = px.read(buf, fmt=".ply")

    with open("brain.vtk", "rb") as fh:
        mesh = px.read(fh)              # a named handle needs no fmt=

Two rules hold everywhere:

* a handle polyxios is given is read or written **where it stands**, and is
  never closed - the caller keeps control of its own file;
* a buffer with no file name has no extension to infer a format from, so
  ``fmt=`` is required for one; ``open()`` gives a handle a name, and that
  is enough.

Reading lazily needs a real file behind the handle, since ``mmap`` maps a
file descriptor: an ``io.BytesIO`` raises ``LazyReadError`` rather than
quietly loading eagerly. TetGen is the one format a buffer cannot carry -
a mesh is a ``.node`` and an ``.ele`` file found beside each other by name.

Compressed files
----------------

gzip is handled by the same layer, for every format at once:

.. code-block:: python

    mesh = px.read("brain.vol.gz")      # decompressed on the way in
    px.write(mesh, "brain.vtk.gz")      # compressed on the way out

Reading looks at the **content**: a file compressed without being renamed
reads just as well as one ending in ``.gz``, and a ``.gz`` name over plain
bytes is read as the plain file it is. Writing looks at the **name**, since
an output file has no content to inspect yet - a destination ending ``.gz``
is compressed, and nothing else is.

The compressed output carries no timestamp and no embedded file name, so the
same mesh always produces the same bytes. Lazy reads are the one thing gzip
takes away: ``mmap`` maps a file as it is stored, so a compressed file raises
``LazyReadError`` rather than handing back compressed bytes.

Format-specific options
-----------------------

.. code-block:: python

    px.write(mesh, "brain.vtk", binary=True)
    px.write(mesh, "brain.ply", binary=True, endian="little")

Every codec's own options are listed on its page under
:doc:`formats/index`.

Where to go next
----------------

* :doc:`formats/index` - the twenty-five supported formats, one page each
* :doc:`lazy_loading` - reading files larger than RAM
* :doc:`transforms` - filtering, cleaning and merging meshes
* :doc:`cli` - the ``pxios`` command line
* :doc:`plugins` - teaching polyxios a new format
