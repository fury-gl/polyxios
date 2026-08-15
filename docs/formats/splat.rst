.. _format-splat:

Gaussian splat
==============

.. rst-class:: px-badges

``.splat`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.splat`` is the 32-byte-per-Gaussian binary layout used by the WebGL Gaussian Splat Viewer and compatible tools. It has no header: the file is a flat array of fixed-width little-endian records, and the splat count is the file size divided by 32. Each record is three ``float32`` positions, three ``float32`` scales, four ``uint8`` colour and opacity bytes, and four ``uint8`` packed rotation quaternion components — 32 bytes exactly. The format carries a point cloud only; there is no connectivity of any kind.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - none; count = file size / 32
   * - record size
     - 32 bytes, little-endian
   * - position
     - x, y, z as float32
   * - scale
     - scale_0, scale_1, scale_2 as float32
   * - colour
     - color_r, color_g, color_b, opacity as uint8
   * - rotation
     - rot_0..rot_3 as uint8, packed quaternion
   * - connectivity
     - none

.. rst-class:: px-speclink

`Reference implementation ↗ <https://github.com/antimatter15/splat>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    cloud = px.read("scene.splat")
    cloud.vertices               # (n, 3)
    cloud.vertex_attrs["opacity"]
    len(cloud.element_types)     # 0 - the format carries no elements

Writing
-------

.. code-block:: python

    px.write(cloud, "out.splat")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The eleven non-positional fields are read into and written from ``vertex_attrs`` under their record names: ``scale_0..2``, ``color_r/g/b``, ``opacity``, ``rot_0..3``.
- A missing ``vertex_attrs`` entry is written as zeros rather than raising, so a plain point cloud can be written out as a ``.splat``.
- ``element_types`` is always empty. Anything expecting faces will find none.
- ``lazy=True`` is ignored; the file is a flat binary array with no seekable structure to defer.

.. seealso::

   :doc:`index` — the full format table.
