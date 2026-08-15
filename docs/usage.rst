Usage
=====

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

Format-specific options
-----------------------

.. code-block:: python

    px.write(mesh, "brain.vtk", binary=True)
    px.write(mesh, "brain.ply", binary=True, endian="little")

Every codec's own options are listed on its page under
:doc:`formats/index`.

Where to go next
----------------

* :doc:`formats/index` - the eighteen supported formats, one page each
* :doc:`lazy_loading` - reading files larger than RAM
* :doc:`transforms` - filtering, cleaning and merging meshes
* :doc:`cli` - the ``pxios`` command line
* :doc:`plugins` - teaching polyxios a new format
