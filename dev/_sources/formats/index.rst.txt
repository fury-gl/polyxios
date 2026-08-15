.. _formats:

Supported formats
=================

polyxios ships twenty-five codecs across thirty-four extensions. Each is registered by
extension, so :func:`polyxios.read` picks the right reader from the filename — pass
``fmt=`` to override it.

Every page below summarises the format's specification, notes where polyxios extends or
deviates from it, and links to the authoritative document.

.. raw:: html
   :file: ../_includes/formats_grid.html

Parallel and multi-block meta-files
-----------------------------------

``.vtm``, ``.pvtu``, ``.pvts``, ``.pvti``, ``.pvtp`` and ``.pvtr`` are registered too,
but they hold no geometry — only references to sub-files. Reading one raises
:class:`~polyxios.exceptions.UnsupportedFormatError` with a pointer to
``examples/read_parallel_vtk.py``, rather than failing with a parse error further in.
Writing them is not supported.

.. toctree::
   :hidden:
   :maxdepth: 1

   vtk
   vtr
   vtp
   obj
   ply
   stl
   off
   abaqus
   avs
   meshb
   dolfin
   flac3d
   gmsh
   nastran
   tecplot
   su2
   tetgen
   wkt
   mfem
   netgen
   ugrid
   splat
   vtu
   vts
   vti
