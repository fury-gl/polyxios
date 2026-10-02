.. _formats:

Supported formats
=================

.. meta::
   :description: The forty-four 3D mesh and geometry formats polyxios reads and writes, across fifty-seven extensions - VTK, VTKHDF, OBJ, PLY, STL, glTF, COLLADA, LAS, PCD, XYZ, XDMF, MED, CGNS, Gmsh, ANSYS Fluent, Abaqus, Nastran, Netgen, Kratos MDPA, PERMAS, SVG and more, one reference page each.


polyxios ships forty-four codecs across fifty-seven extensions - the fifty-six
below plus ``.plt``, which is recognised but not read. Each is registered by
extension, so :func:`polyxios.read` picks the right reader from the filename - pass
``fmt=`` to override it.

Every page below summarises the format's specification, notes where polyxios extends or
deviates from it, and links to the authoritative document.

.. raw:: html
   :file: ../_includes/formats_grid.html

Parallel and multi-block meta-files
-----------------------------------

``.vtm``, ``.pvtu``, ``.pvts``, ``.pvti``, ``.pvtp`` and ``.pvtr`` are registered too,
but they hold no geometry - only references to sub-files. (A ``.pvd`` names
sub-files too, but over time, and :doc:`reads as a time series <pvd>`.) :func:`polyxios.read`
hands back one mesh, always, so reading an index raises
:class:`~polyxios.exceptions.UnsupportedFormatError` rather than failing with a
parse error further in. The several live in the helper instead:

.. code-block:: python

    from polyxios import helper, transforms

    whole = helper.read_multiblock("case.pvtu")   # every piece, merged
    blocks = helper.read_blocks("case.vtm")       # one PolyData per sub-file

Both follow an index that names another index, skip a sub-file that is missing
or unreadable with a line on the ``polyxios`` logger, and refuse a reference
resolving outside the index file's own directory.
``examples/read_parallel_vtk.py`` walks through what they do. Writing an index
file is not supported.

.. toctree::
   :hidden:
   :maxdepth: 1

   vtk
   vtr
   vtp
   obj
   ply
   stl
   3mf
   off
   abaqus
   avs
   meshb
   medit
   dolfin
   flac3d
   gltf
   collada
   gmsh
   nastran
   tecplot
   su2
   tetgen
   wkt
   vtu
   vts
   vti
   vtkhdf
   pvd
   mfem
   netgen
   ugrid
   splat
   pcd
   las
   xyz
   mdpa
   permas
   fluent
   svg
   xdmf
   med
   cgns
   h5m
   hmf
   exodus
