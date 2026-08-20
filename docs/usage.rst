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

A source has to offer a binary ``read`` and a destination a binary
``write``; everything else a codec reaches for - ``readline``, ``seek``,
``seekable`` - is supplied for a handle that lacks it. A handle that cannot
seek is still read, except by the formats that need to measure a file before
parsing it and by the extensions several formats share, both of which say so
rather than guess.

Reading lazily needs a real file behind the handle wherever the arrays that
come back view the file itself - binary PLY and binary VTK - since ``mmap``
maps a file descriptor: an ``io.BytesIO`` raises ``LazyReadError`` rather
than quietly loading eagerly. It also needs the handle to stand at the start
of that file - a mapping addresses a file from byte zero - so a handle
part-way into one is refused for a lazy read and read eagerly without
complaint. A format whose lazy mode only skips work, such as binary STL
skipping vertex deduplication, copies what it reads and so takes a buffer
like any other read. TetGen is the one format a buffer cannot carry - a mesh
is a ``.node`` and an ``.ele`` file found beside each other by name.

Text formats are written with ``\n`` line endings on every platform, so the
bytes a path receives and the bytes a buffer receives are the same ones.

Compressed files
----------------

gzip is handled by the same layer, for every format at once:

.. code-block:: python

    mesh = px.read("brain.vol.gz")      # decompressed on the way in
    px.write(mesh, "brain.vtk.gz")      # compressed on the way out

Reading looks at the **content**: a file compressed without being renamed
reads just as well as one ending in ``.gz``, and a ``.gz`` name over plain
bytes is read as the plain file it is. The whole four-byte gzip header is
what decides, not the two magic bytes alone - those open one file in every
65536 by chance, and inside a headerless binary format they are an ordinary
coordinate's low bytes. Writing looks at the **name**, since
an output file has no content to inspect yet - a destination ending ``.gz``
is compressed, and nothing else is. A nameless buffer has no name to end in
``.gz``, so ``fmt=`` says it there instead:

.. code-block:: python

    buf = io.BytesIO()
    px.write(mesh, buf, fmt=".obj.gz")  # compressed into the buffer

``.gz`` names the compression rather than the format wherever it appears, in
a file name and in ``fmt=`` alike, so ``".obj.gz"`` picks the same codec
``brain.obj.gz`` does.

A member does not have to run to the end of what it sits in. Reading stops
where the member stops, so a mesh compressed into the middle of an archive
is read without the archive's own bytes after it turning into an error;
members written back to back are still read as the one stream they spell.

The compressed output carries no timestamp and no embedded file name, so the
same mesh always produces the same bytes. Lazy reads are the one thing gzip
takes away, in the formats that map: ``mmap`` maps a file as it is stored, so
a format whose lazy read hands back arrays viewing the mapping raises
``LazyReadError`` over a compressed file rather than handing back compressed
bytes. A lazy read that copies what it reads - binary STL's, which skips
vertex deduplication and nothing else - takes a compressed file like any
other read.

TetGen is outside all of this for the same reason it cannot take a buffer: it
opens its own sibling files rather than going through the layer that unwraps
gzip. A ``.node.gz`` is refused with a message saying so, rather than read as
text or written uncompressed under a name promising otherwise.

A compressed handle is the one exception to "left where the codec left it":
a decompressor reads ahead, so where it stops says nothing about how much of
the mesh was consumed. The handle is put back at the front of the member
instead, which is the only position over compressed bytes that means
anything to whatever reads next.

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
