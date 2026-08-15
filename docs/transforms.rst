Transforms
==========

.. code-block:: python

    from polyxios.transforms import (
        pipeline,
        merge,
        filter_element_type,
        remove_orphan_vertices,
    )

    # Compose transforms into a single function
    clean = pipeline(
        filter_element_type(keep="triangle"),
        remove_orphan_vertices,
    )
    result = clean(mesh)

    # Merge two meshes into one
    combined = merge(mesh_a, mesh_b)
