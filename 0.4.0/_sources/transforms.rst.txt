Transforms
==========

.. meta::
   :description: Filter, clean and merge meshes in polyxios: drop orphan vertices, weld coincident ones, triangulate surfaces, extract the boundary of a volume and combine datasets.


Every transform takes a :class:`~polyxios.PolyData` and returns a new one -
nothing is modified in place - so they compose freely.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Transform
     - What it does
   * - ``pipeline(*fns)``
     - Compose transforms left to right into one callable.
   * - ``merge(*polys)``
     - Concatenate several meshes into one, offsetting the indices.
   * - ``filter_element_type(poly, keep=...)``
     - Keep only the named element types.
   * - ``remove_orphan_vertices(poly)``
     - Drop vertices no element references and remap the indices.
   * - ``reindex(poly)``
     - Alias of ``remove_orphan_vertices``.
   * - ``merge_duplicate_vertices(poly, tol=...)``
     - Weld coincident vertices into one.
   * - ``triangulate(poly)``
     - Split every surface element into triangles.
   * - ``extract_surface(poly)``
     - Return the boundary faces of a volumetric mesh.
   * - ``vertex_colors(poly)``
     - Read per-vertex RGB out of the vertex attributes, or ``None``.

Composing and merging
---------------------

.. code-block:: python

    from functools import partial

    from polyxios.transforms import (
        pipeline,
        merge,
        filter_element_type,
        remove_orphan_vertices,
    )

    # Compose transforms into a single function
    clean = pipeline(
        partial(filter_element_type, keep="triangle"),
        remove_orphan_vertices,
    )
    result = clean(mesh)

    # Merge two meshes into one
    combined = merge(mesh_a, mesh_b)

Cleaning
--------

``remove_orphan_vertices`` drops the vertices nothing references;
``merge_duplicate_vertices`` welds the ones that sit on top of each other -
the equivalent of ParaView's "Clean to Grid", and what turns the facet soup
STL hands back into a surface again:

.. code-block:: python

    from polyxios.transforms import merge_duplicate_vertices

    welded = merge_duplicate_vertices(mesh)             # exact matches only
    snapped = merge_duplicate_vertices(mesh, tol=1e-6)  # snap to a 1e-6 grid

``tol`` snaps each coordinate to a grid of that step before comparing, so
two points merge when they land in the same cell. The survivor of a group is
its lowest original index and keeps its own coordinates and attributes: the
result does not depend on which duplicate the file listed first, and a
tolerance never moves a point. Welding is not culling - a vertex no element
references is kept, so compose with ``remove_orphan_vertices`` to drop those
too.

Surfaces
--------

``triangulate`` splits quads and pixels into two triangles and fan-triangulates
polygons and triangle strips; quadratic surface elements are linearised to
their corner nodes first, and lines and volumes are dropped.

``extract_surface`` returns the boundary of a volumetric mesh - the faces
shared by exactly one element. Surface elements already in the mesh are
ignored, so a tetrahedral mesh that also carries its boundary triangles is
not counted twice. Vertices are preserved unchanged, so follow it with
``remove_orphan_vertices`` to compact them:

.. code-block:: python

    from polyxios.transforms import extract_surface, remove_orphan_vertices

    boundary = remove_orphan_vertices(extract_surface(volume_mesh))

Colours
-------

``vertex_colors`` returns an ``(n_verts, 3)`` float32 array in ``[0, 1]``, or
``None`` when no vertex attribute looks like a colour. An attribute named for
colour wins outright; otherwise the first ``(n, >= 3)`` attribute is taken,
skipping any carrying a negative value - a normal is the same shape as an RGB
triple, so shape alone would hand back surface directions as colours.

Full signatures and parameters are in the :doc:`API reference <api/index>`.
