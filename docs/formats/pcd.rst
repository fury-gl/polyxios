.. _format-pcd:

PCD (Point Cloud Library)
=========================

.. rst-class:: px-badges

``.pcd`` ``read + write`` ``lazy: binary``

Summary of the specification
----------------------------

PCD is the Point Cloud Library's own file format and, through PCL, ROS and Open3D, the one most robotics and scanning pipelines exchange clouds in. A short ASCII header names the fields of one point record - ``FIELDS``, their ``SIZE`` in bytes, ``TYPE`` (``I`` signed, ``U`` unsigned, ``F`` float) and ``COUNT`` of components - and the cloud's ``WIDTH`` and ``HEIGHT``, which are a grid for an organised cloud from a depth camera and ``n`` by ``1`` otherwise. ``VIEWPOINT`` is the sensor pose as a translation and a quaternion. ``DATA`` picks the body: ``ascii`` rows, ``binary`` records, or ``binary_compressed``, where the fields are laid out one after another and packed with LZF. There is no connectivity of any kind.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - ``VERSION 0.7``, ``FIELDS``, ``SIZE``, ``TYPE``, ``COUNT``, ``WIDTH``, ``HEIGHT``, ``VIEWPOINT``, ``POINTS``, ``DATA``
   * - field types
     - ``I`` 1/2/4/8, ``U`` 1/2/4/8, ``F`` 4/8; ``COUNT`` above one is a per-point vector
   * - bodies
     - ``ascii`` one point per line; ``binary`` little-endian records; ``binary_compressed`` two ``uint32`` sizes then an LZF block of the fields laid out field by field
   * - colour
     - one ``rgb`` field packed ``0x00RRGGBB`` in a float's bits (``rgba`` in a ``U 4``)
   * - padding
     - a field named ``_`` holds bytes nothing names
   * - connectivity
     - none

.. rst-class:: px-speclink

`Full specification ↗ <https://pointclouds.org/documentation/tutorials/pcd_file_format.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    cloud = px.read("scan.pcd")
    cloud.vertices                     # (n, 3) float64, NaN points kept
    cloud.vertex_attrs["colors"]       # (n, 3) floats in 0..1, from a packed rgb
    cloud.vertex_attrs["normals"]      # (n, 3), from normal_x normal_y normal_z
    cloud.vertex_attrs["intensity"]    # every other field under its own name
    cloud.global_attrs.get("pcd_width")  # with pcd_height, for an organised cloud

``x``, ``y`` and ``z`` are the vertices. ``normal_x/y/z`` fold into ``normals`` and a packed ``rgb`` or ``rgba`` - one scalar of four bytes - into ``colors``; every other field, an ``rgb`` of any other width included, is a vertex attribute of its own name and dtype, two-dimensional when its ``COUNT`` is above one. In an ``ascii`` body an integer field's ``nan`` reads as 0, as PCL reads it, and a fraction float64 can see is refused; a name ``FIELDS`` spells twice is read with its position appended, with a warning. A ``HEIGHT`` above one keeps ``pcd_width`` and ``pcd_height`` in ``global_attrs``, a ``VIEWPOINT`` other than the identity ``pcd_viewpoint``. Padding fields are skipped.

Writing
-------

.. code-block:: python

    px.write(cloud, "out.pcd")                                    # DATA binary
    px.write(cloud, "out.pcd", data_format="binary_compressed")   # LZF, the way PCL packs it
    px.write(cloud, "out.pcd", data_format="ascii", float_fmt=".8g")
    px.write(cloud, "out.pcd", double=True)                       # F 8 coordinates

The vertices go out as ``x y z``, ``vertex_attrs["normals"]`` as ``normal_x normal_y normal_z``, ``vertex_attrs["colors"]`` (three or four wide; floats in 0..1, or integers already in 0..255) as one packed ``rgb`` or ``rgba``, and every other numeric vertex attribute as a field of its own name at its own dtype, a two-dimensional one with its width as ``COUNT`` (so a single column reads back one-dimensional). An attribute whose name a folded field already took - ``rgb``, ``normal_x`` - is dropped with a warning, as is one whose name is not a bare ASCII token or is the padding name ``_``. ``pcd_width`` and ``pcd_height`` in ``global_attrs`` write an organised cloud when they are whole numbers whose product is the point count, ``pcd_viewpoint`` the ``VIEWPOINT`` line. Coordinates and normals are ``F 4`` unless ``double=True``, which in an ``ascii`` body also spells every float at ``.17g`` rather than ``.10g`` unless ``float_fmt`` is given. A one-column ``rgb`` or ``rgba`` attribute of four bytes, which every reader would take for a packed colour, is written as eight with a warning so it reads back as its values.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Elements are not part of the format: a mesh's faces are dropped with a warning, and every file reads back with no elements at all.
- PCL spells a packed ``rgb`` in an ``ascii`` body as the integer's decimal value (unsigned today, signed in older releases) and in a ``binary`` one as the bits of a float; PCL before 1.8 and a few other writers spell that float itself in ``ascii`` (``4.2108e+06`` in the PCL tutorial cloud). polyxios reads a bare integer token as the decimal value and any other spelling as the float whose bits are the colour, and writes PCL's, so a colour survives every layout.
- ``binary_compressed`` needs LZF, which polyxios ships: a compiled module with a pure-Python twin behind it, so the layout reads without any dependency and at full speed on a built wheel.
- A vertex attribute that is not numeric, or is more than two-dimensional, has no PCD type and is dropped with a warning naming it. No other ``global_attrs`` entry has a line in the header.
- ``F 4`` is float32: a coordinate past about 3.4e38 becomes ``inf`` - in a ``binary`` or ``binary_compressed`` body on write, in an ``ascii`` body (which spells the source value) on read - with numpy's overflow ``RuntimeWarning`` as the only notice. ``double=True`` keeps it.
- ``lazy=True`` maps a ``DATA binary`` file and hands back views of it, read-only and in the file's own dtypes: the coordinates are one strided view when ``x``, ``y``, ``z`` sit side by side in one type (a file where they are not raises), the normals likewise when they are and three plain attributes when not, every other field a view of its own column. A packed colour is decoded into ``colors`` either way. An ``ascii`` or ``binary_compressed`` file raises :class:`~polyxios.exceptions.LazyReadError`, as does an in-memory buffer.
- ``POINTS`` is checked against the file size before anything is allocated; a header promising more points than the file holds is refused with the number it named. A ``binary_compressed`` block claiming to expand more than 88-fold, past what LZF can spell, is refused before the expansion is allocated.

.. seealso::

   :doc:`xyz` - the ASCII column formats scanners export.
   :doc:`index` - the full format table.
