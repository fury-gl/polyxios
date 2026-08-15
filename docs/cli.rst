Command line interface (pxios)
==============================

polyxios comes with a command-line interface ``pxios`` to quickly fetch, list,
convert, and visualize 3D models.

``--verbose`` can be given on either side of the subcommand (e.g.
``pxios --verbose fetch bunny.obj`` or ``pxios fetch bunny.obj --verbose``) to
print debug logs and full tracebacks when a command fails.

Subcommands
-----------

*   ``pxios list``: Lists all available remote or cached files, or registered
    formats. The three listing modes below are mutually exclusive.

    *   ``--local``: Lists locally cached files (can filter by optional
        extension argument, e.g. ``pxios list obj --local``).
    *   ``--extensions`` / ``--formats``: Lists all formats and extensions
        available in the remote catalog.
    *   ``--codecs``: Lists all formats supported by polyxios codecs.

*   ``pxios fetch <filename|extension>``: Downloads and caches a single model
    file (e.g., ``bunny.obj``) or every model catalogued for an extension
    (e.g., ``obj`` or ``.obj``).
*   ``pxios convert <input_file> <output_file>``: Converts a model file from
    one format to another directly in a single process.
*   ``pxios viz <filename>``: Visualizes a local or cached model file using the
    `FURY <https://fury.gl>`__ library.

    *   ``--lines``: Render line elements using ``actor.line`` instead of
        rendering as a surface/point cloud.
    *   ``--points``: Render strictly as a point cloud.

Example commands
----------------

.. code-block:: bash

    # List all fetchable remote models
    pxios list

    # Fetch a single model
    pxios fetch bunny.obj

    # Fetch every model catalogued for an extension
    pxios fetch vtk

    # Convert a mesh file
    pxios convert bunny.obj bunny.vtk

    # Visualize a model
    pxios viz bunny.obj
