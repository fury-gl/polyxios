.. _formats:

Supported formats
=================

polyxios ships eighteen codecs. Each is registered by extension, so :func:`polyxios.read`
picks the right reader from the filename — pass ``fmt=`` to override it.

Every page below summarises the format's specification, notes where polyxios extends or
deviates from it, and links to the authoritative document.

.. raw:: html
   :file: ../_includes/formats_grid.html

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
