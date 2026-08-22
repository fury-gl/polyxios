Transforms
==========

.. meta::
   :description: Filter, clean and merge meshes in polyxios: drop degenerate elements, weld duplicate vertices, extract surfaces and combine datasets.


.. code-block:: python

    from polyxios.transforms import (
        pipeline,
        merge,
        merge_duplicate_vertices,
        filter_element_type,
        remove_orphan_vertices,
    )

    # Compose transforms into a single function
    clean = pipeline(
        filter_element_type(keep="triangle"),
        remove_orphan_vertices,
    )
    result = clean(mesh)

    # Weld coincident vertices - the STL facet soup back into a surface
    welded = merge_duplicate_vertices(mesh)
    snapped = merge_duplicate_vertices(mesh, tol=1e-6)

    # Merge two meshes into one
    combined = merge(mesh_a, mesh_b)
