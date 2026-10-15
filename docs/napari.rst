napari plugin
=============

.. meta::
   :description: Open and save meshes and point clouds in napari with polyxios - every format polyxios reads, as Surface and Points layers, with no extra package to install.


polyxios is a `napari <https://napari.org>`_ plugin as it is installed: no
separate package, and no napari dependency for anyone who does not use it.

.. code-block:: bash

    pip install polyxios "napari[all]"
    napari mesh.vtu cloud.las

Every extension polyxios reads opens by drag-and-drop, *File > Open* or
``viewer.open``, gzipped or not. Extensions other software uses too -
``.dat``, ``.e``, ``.plt``, ``.pts``, ``.vol``, ``.xml``, ``.xyz`` - are taken
only when the file opens the way a format polyxios reads does, so a settings
``.xml``, a binary Tecplot ``.plt``, a molecule ``.xyz`` or a face-landmark
``.pts`` is left to the next plugin rather than failing here, and so is a
file that starts like gzip and is not. ``.svg``
is not offered: polyxios writes SVG pictures but cannot read a mesh back from
one.

What each file becomes
----------------------

- A mesh with surface cells opens as a **Surface** layer. Quads, pixels,
  strips and polygons are triangulated, and the layer's ``values`` - what
  colours it - is the first floating vertex attribute that is not a normal
  or texture coordinate component (``nx``, ``s``, ``u``, ...), or failing one
  the first integer attribute that is not an index (``original_ids``,
  ``*_index``). Polygons are fanned from their first corner, so a concave
  one draws wrong. Lines and vertex cells next to the surface are not drawn,
  and opening the file warns about them.
- A volume mesh - tetrahedra, hexahedra, wedges, pyramids, prisms and voxels,
  linear or quadratic - opens as the surface that bounds it. A boundary
  patch the file lists next to the volume, as a gmsh or Abaqus mesh often
  does, is drawn once rather than twice. A VTK image (``.vti``) is a grid of
  voxels to polyxios, so it opens the same way, as the skin of the grid; to
  see it as an image, open it with an image reader instead. A structured
  grid finds that skin from its extent, at the cost of its surface rather
  than of every cell.
- Polyhedra, Lagrange, Bézier and other higher-order cells, quadratic
  polygons and quadratic-linear quads are not drawn. A file holding only
  those opens as a **Points** layer, with a warning, and keeps its cells for
  saving.
- A ParaView index (``.vtm``, ``.pvtu``, ``.pvtp``, ``.pvtr``, ``.pvts``,
  ``.pvti``), or a ``.vtp`` holding a multi-block set, opens as its pieces
  merged into one layer, as :func:`polyxios.helper.read_multiblock` merges
  them. A gzipped index is left to other plugins, and a gzipped multi-block
  ``.vtp`` fails with the VTP reader's error: an index is read as plain XML
  only.
- A file with no surface or volume cells - a point cloud from PCD, LAS,
  ``.xyz`` or ``.splat``, or a line set - opens as a **Points** layer. Each
  scalar vertex attribute (``intensity``, ``classification``, ...) becomes a
  layer feature. Points are sized to a fraction of the cloud's extent, so a
  survey in metres and a scan in millimetres both show as points rather than
  specks or blobs.
- Vertex colours show on either layer: a ``colors`` attribute, or the
  separate ``red green blue`` (and ``alpha``) channels a PLY file spells them
  as. 8- and 16-bit channels are scaled by their type's full range. Nothing
  else is taken for a colour, however it is shaped - a normal is three
  numbers per vertex too.

Lines are not drawn as paths yet, but they are not lost either: see
`Saving`_.

Axis order
----------

napari orders coordinates the way it orders image axes, slowest first.
polyxios reverses each vertex ``(x, y, z)`` into ``(z, y, x)``, which puts a
mesh on top of the volume it was segmented from. Each layer records the order
it was opened in, under ``layer.metadata["polyxios"]["axis_order"]``, and
saving undoes exactly that order. Reversing the columns mirrors the mesh, so
each triangle's vertex order is reversed with them and the surface still
faces outward when napari draws it; saving reverses it back. A layer drawn in
napari is taken as ``(z, y, x)``. Any other value under that key is refused
on save rather than guessed at.

To keep the columns as stored, build the layer from Python with
:func:`polyxios.napari.layer_data`:

.. code-block:: python

    import napari
    import polyxios
    from polyxios.napari import layer_data

    viewer = napari.Viewer()
    poly = polyxios.read("part.inp")
    for data, kwargs, layer_type in layer_data(poly, name="part", axis_order="xyz"):
        viewer.add_layer(napari.layers.Layer.create(data, kwargs, layer_type))

The plugin offers one reader, not one per axis order: napari lists readers
by plugin, so a second one would never be offered in *Open with plugin*, and
which of the two opened a file would be left to chance.

Saving
------

*File > Save Selected Layer* writes a Surface layer to
``.ply .stl .obj .off .vtk .vtp .vtu .vtkhdf .glb .gltf .3mf .dae .msh .inp
.med .mesh .meshb .exo``, and a Points layer to
``.ply .pcd .las .laz .xyz .pts .vtk .vtp .vtu``. ``.mesh`` is written as
MFEM, as :func:`polyxios.write` writes it; ``.meshb`` is Medit.

What the layer cannot show is kept in its metadata and written back: the
other vertex attributes (normals, texture coordinates, ...), the vertex tags,
the lines of a line set, and the mesh-level attributes such as a LAS file's
coordinate system. A Surface's ``values`` are written under the name they were
read from, a time series of shape (T, n) as one attribute of shape (n, T).
napari gives a Surface opened without values a field of ones; that field is
not written, while values set on such a layer are written as ``values``; a
Points layer's numeric features are written as vertex attributes, and a text
feature is dropped with a warning, and one sharing a name with the file's
colours or a kept attribute is written under a free name (``red_1``), with a
warning; colours are written under the name and in
the range they were read in, and only when the file had them. An alpha set in
napari on colours read without one is written as a fourth ``colors`` column
or a PLY ``alpha`` channel; where the file's colour channels have no place for
it, the writer warns.

The writers save the layer's data coordinates. Its ``scale``, ``translate``
and other transforms are napari's view of the layer, not part of the mesh,
and are not applied: a mesh shown at a volume's voxel spacing through
``scale`` is saved at its own coordinates, as it was read.

That holds while the layer's points are the file's points. Moving a point
keeps what was kept for it, except its normals (``normals``, ``nx``, ...),
which the move leaves pointing the way the old surface did: those are
dropped, with a warning. Once points are added, deleted or shift to another
row, what was kept no longer lines up; the writer says so in a warning and
writes the layer as it stands. That includes the attributes that lay the rows
out in a grid - an organised PCD's ``pcd_width`` and ``pcd_height``, a PTX
file's scans - which are dropped with the rest. A point deleted and another
added where it was cannot be told from a move.

A ``.gltf`` is saved with its ``.bin`` beside it, and the writer reports
both files.

A Surface layer is triangles. Saving one warns when the file it came from had
more than that - quads, volume cells, lines, per-element data - since what is
written is the triangulated skin, not the original cells. The extent of a
``.vti``, ``.vtr`` or ``.vts`` grid, and the ``ImageData`` type of a
``.vtkhdf`` image, are not written either: no grid describes its skin.

Each writer saves one layer. With a Surface and a Points layer both selected,
napari finds no writer for the pair: save them one at a time.

A format that cannot hold an attribute warns and drops it, as it does for any
write. Formats that need an optional package (``.vtkhdf``, ``.med``, ``.exo``,
``.laz``) say which one when it is missing.
