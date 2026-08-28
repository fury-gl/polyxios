.. _format-tetgen:

TetGen
======

.. rst-class:: px-badges

``.ele + .node`` ``read + write`` ``eager``

Summary of the specification
----------------------------

TetGen splits a mesh across sidecar files that share a basename. ``.node`` starts with a counts line - number of points, dimension, number of attributes, boundary-marker flag - then one line per point: an index, its coordinates, its attributes, and its marker if the flag is set. ``.ele`` starts with its own counts line - number of tetrahedra, nodes per tetrahedron, region-attribute flag - then one line per element: an index and its node indices, with an optional trailing region attribute. Indices may be 0- or 1-based; the first index in the file decides.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - .node header
     - npoints dim nattrs boundary_marker_flag
   * - .node line
     - index x y z [attrs...] [marker]
   * - .ele header
     - ntetrahedra nodes_per_tet region_attr_flag
   * - .ele line
     - index n1 n2 n3 n4 [region]
   * - index base
     - 0 or 1, inferred from the first index
   * - companions
     - .face, .edge, .neigh use the same convention

.. rst-class:: px-speclink

`Read the full TetGen specification ↗ <https://wias-berlin.de/software/tetgen/fformats.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.ele")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.ele")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Pass either path - ``px.read("bar.ele")`` or ``px.read("bar.node")`` - and the paired file is located by basename.
- Boundary markers become vertex tags and region attributes become element attributes, so both labelling schemes are kept. Those attributes are REALs in the format and are read as float64, so a fractional region value keeps its fraction.
- Both 0- and 1-based indexing are handled; the base is inferred rather than assumed.
- A ``.node`` file declaring 2 dimensions is padded with a zero z, so the vertex array is always ``(n, 3)``, and ``global_attrs["was_2d"]`` records the fact so the header goes back out declaring 2. A mesh whose vertices have since left the plane is written in three with a warning, and a mesh with tetrahedra to write keeps three whatever the flag says: the ``.ele`` file declares four nodes per element, and TetGen's own 2-D mode fills the plane with triangles instead.

.. seealso::

   :doc:`index` - the full format table.
