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

Entity numbering
----------------

A :class:`~polyxios.PolyData` indexes its vertices and elements densely from
zero, always. Most formats agree, but a hand-edited Abaqus deck, a Nastran bulk
data file, a Gmsh ``.msh``, a FLAC3D grid and a Kratos ``.mdpa`` all let their
author number entities freely - neither dense nor zero-based, and in any order.
Reading such a file has to renumber, and renumbering on its own is lossy: the
ids are how the author's other files talk about this mesh, a load case naming
``GRID 7000001`` or a report keyed on element ``4001``.

So the reader records what the file said and the writer puts it back:

.. code-block:: python

    mesh = px.read("model.bdf")
    mesh.vertex_attrs["original_ids"]    # the GRID ids the deck spelled
    mesh.element_attrs["original_ids"]   # the element ids

    px.write(mesh, "model.msh")          # written back under those numbers

The key is deliberately not format-prefixed: an id says something about the
entity, not about the file it came from, so a mesh read from a ``.bdf`` and
written as ``.msh`` keeps its numbering. Three rules hold across every codec
that carries ids:

* **Read renumbers.** Connectivity, tags and attributes all speak dense
  zero-based indices whatever the file spelled.
* **Read remembers only what the index does not already say.** A file
  numbering ``1..n`` in order records nothing at all - the writer's own
  renumbering reproduces it exactly. The key's absence means "no numbering
  worth keeping", not "numbered from one".
* **Write asks.** Stored ids win when they can still be written; otherwise the
  writer renumbers from one, with a warning.

"Can still be written" is checked at the point of writing rather than trusted,
because a transform moves a mesh out from under its ids without either one
being wrong: ``merge`` concatenates two meshes that each numbered from one, so
the ids collide, and splitting a quad leaves two triangles carrying one id.
Either would spell a file with two entities answering to one number, which no
solver loads, so the ids have to be positive, unique and one per entity to
survive.

A format that numbers densely by construction - Medit, OFF, STL, the VTK
family - has no id of its own to record, and ignores the key on write. Whether
it *carries* one is a separate question, and comes down to whether it has a
general attribute channel: PLY and the VTK family write ``original_ids`` back
out as an ordinary data array, so a mesh read from a ``.bdf``, welded, and
written as ``.vtu`` keeps the ids of the vertices that survived. Tecplot has
the channel on one side only - its variables are per-vertex, so vertex ids
travel and element ids are dropped with the rest of the element attributes.
Medit, OFF, STL and OBJ have no such channel at all and drop the key with
every other attribute, silently. Where the channel is float - a legacy VTK
data array, a Tecplot variable, a PLY face property - the ids come back as
whole float64 values rather than int64.

Two-dimensional meshes
----------------------

A :class:`~polyxios.PolyData` holds three coordinate columns, always. A format
that spells two - a bamg ``.mesh``, an ``NDIME= 2`` SU2 case, a 2-D MFEM mesh -
is padded with ``z=0`` on the way in rather than kept narrow, so every consumer
can index ``vertices[:, 2]`` without first asking what the file said.

Padding alone is lossy in one direction: nothing downstream could tell a plane
written in two dimensions from one written in three that happens to sit at
``z=0``, so a round trip would widen the file. The reader records the fact
instead:

.. code-block:: python

    mesh = px.read("plate.su2")          # NDIME= 2
    mesh.vertices.shape                  # (n, 3), the z column all zeros
    mesh.global_attrs["was_2d"]          # True

    px.write(mesh, "plate.mesh")         # an MFEM mesh of two columns

Three rules, and every 2-D-capable codec follows them:

1. **Read pads.** Vertices are ``(n, 3)`` float64 whatever the file declared.
2. **Read remembers.** A file that declared two dimensions sets
   ``global_attrs["was_2d"] = True``. A three-dimensional file sets nothing:
   the key's absence means "not known to be two-dimensional", not "3-D".
3. **Write asks.** The flag decides how many columns go out - while the mesh
   is still flat. Coordinates outrank it: a third coordinate that reached the
   mesh after the read is data, so a mesh that has left the plane is written
   in three dimensions with a warning rather than flattened in silence.

The key is deliberately not format-prefixed the way ``tecplot_title`` is: it
says something about the mesh, not about the file it came from. A plane read
from a 2-D Netgen ``.vol`` - which has no two-dimensional spelling of its own
to write back - still lands as an ``NDIME= 2`` SU2 case.

Medit, Medit binary, SU2, Netgen, TetGen, MFEM, DOLFIN, Tecplot, Abaqus and
WKT all record the flag on the way in; every one of those but Netgen restores
it on the way out. A format with no two-dimensional spelling at all - OBJ, the
VTK family - keeps writing three columns and ignores the flag.

Two columns sometimes constrain the rest of the file, and the writer keeps it
consistent rather than emitting one no reader loads: an Abaqus deck of
two-column node cards is written under the planar element cards (``CPS3``,
``CPS4``, ``T2D2``), since Abaqus takes a node's dimensionality from the
element referencing it, and a flat mesh of solid cells - a tetrahedron is one
however flat it lies - keeps its third column in every format that reads the
node count per element from a separate number.

Several meshes in one file
--------------------------

:func:`polyxios.read` hands back one :class:`~polyxios.PolyData`, always. A
function that sometimes returns a sequence puts the branch in every caller, so
a file that holds several meshes is treated as what it is - an index of
sub-files, holding no geometry of its own - and reading one raises
:class:`~polyxios.exceptions.UnsupportedFormatError` naming the format.

The several live in :mod:`polyxios.helper`:

.. code-block:: python

    from polyxios import helper

    whole = helper.read_multiblock("case.pvtu")   # every piece, merged
    blocks = helper.read_blocks("case.vtm")       # one PolyData per sub-file

Both read ``.vtm``, ``.pvtu``, ``.pvtp``, ``.pvtr``, ``.pvts`` and ``.pvti``,
as well as a ``.vtp`` holding a ``<vtkMultiBlockDataSet>``. An index naming
another index is followed and read flat; a sub-file that is missing or
unreadable is skipped with a line on the ``polyxios`` logger, so a partially
downloaded companion directory still loads; and a reference resolving outside
the index file's own directory raises ``PermissionError`` rather than reading
it.

``helper.read_blocks`` is the one to reach for when the blocks mean different
things - a fluid domain and a solid one, one part per block - since
:func:`~polyxios.transforms.merge` joins them into a mesh that no longer says
which was which.

Where to go next
----------------

* :doc:`formats/index` - the twenty-seven supported formats, one page each
* :doc:`lazy_loading` - reading files larger than RAM
* :doc:`transforms` - filtering, cleaning and merging meshes
* :doc:`cli` - the ``pxios`` command line
* :doc:`plugins` - teaching polyxios a new format
