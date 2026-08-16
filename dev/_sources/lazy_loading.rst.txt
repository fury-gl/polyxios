Lazy loading
============

.. meta::
   :description: Read mesh files larger than RAM in Python. polyxios memory-maps binary VTK, PLY, STL and Medit files with lazy=True, so vertex and face arrays are paged in on demand.


For large meshes (gigabytes of binary data), pass ``lazy=True``. polyxios
memory-maps the file and only loads the pages you actually touch - the rest
stays on disk until needed.

.. code-block:: python

    # File is opened but data is not loaded into RAM yet
    mesh = px.read("huge_brain.vtk", lazy=True)

    # Only the vertices are pulled from disk here
    first_vertex = mesh.vertices[0]

    # Element connectivity is still on disk until you access it

Lazy loading is supported for binary ``.vtk``, ``.ply``, ``.stl`` and
``.meshb`` files. ASCII formats load eagerly (the whole file must be parsed to
extract values). Binary STL lazy mode skips vertex deduplication - vertices are
returned as-is (3 per triangle), avoiding the extra pass over the data.

.. seealso::

   :doc:`formats/index` - the per-format table records which formats support
   lazy loading.
