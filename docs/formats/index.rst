.. _formats:

Supported formats
=================

polyxios ships eighteen codecs. Each is registered by extension, so :func:`polyxios.read`
picks the right reader from the filename — pass ``fmt=`` to override it.

Every page below summarises the format's specification, notes where polyxios extends or
deviates from it, and links to the authoritative document.

.. list-table::
   :header-rows: 1
   :widths: 30 20 8 16 18
   :class: px-format-table

   * - Format
     - Extension
     - Read
     - Write
     - Lazy
   * - :doc:`VTK Legacy <vtk>`
     - ``.vtk``
     - [x]
     - [x]
     - mmap: binary
   * - :doc:`VTK RectilinearGrid <vtr>`
     - ``.vtr``
     - [x]
     - [x]
     - - - - -
   * - :doc:`VTK PolyData <vtp>`
     - ``.vtp``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Wavefront OBJ <obj>`
     - ``.obj``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Stanford PLY <ply>`
     - ``.ply``
     - [x]
     - [x]
     - mmap: binary
   * - :doc:`STL <stl>`
     - ``.stl``
     - [x]
     - [x]
     - mmap: binary
   * - :doc:`OFF <off>`
     - ``.off``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Abaqus <abaqus>`
     - ``.inp``
     - [x]
     - [x]
     - - - - -
   * - :doc:`AVS-UCD <avs>`
     - ``.avs``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Medit binary <meshb>`
     - ``.meshb``
     - [x]
     - [x]
     - mmap: binary
   * - :doc:`DOLFIN / FEniCS XML <dolfin>`
     - ``.xml``
     - [x]
     - [x]
     - - - - -
   * - :doc:`FLAC3D <flac3d>`
     - ``.f3grid``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Gmsh <gmsh>`
     - ``.msh``
     - [x]
     - [x] v2 only
     - - - - -
   * - :doc:`Nastran <nastran>`
     - ``.bdf .nas .fem``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Tecplot ASCII <tecplot>`
     - ``.tec``
     - [x]
     - [x]
     - - - - -
   * - :doc:`SU2 <su2>`
     - ``.su2``
     - [x]
     - [x]
     - - - - -
   * - :doc:`TetGen <tetgen>`
     - ``.ele + .node``
     - [x]
     - [x]
     - - - - -
   * - :doc:`Well-Known Text <wkt>`
     - ``.wkt``
     - [x]
     - [x]
     - - - - -

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
