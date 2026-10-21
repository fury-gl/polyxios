.. _format-spz:

SPZ (compressed Gaussian splats)
================================

.. rst-class:: px-badges

``.spz`` ``read + write`` ``eager``

Summary of the specification
----------------------------

SPZ is Niantic's compressed format for 3D Gaussian splats, about a tenth the size of the equivalent PLY. Each Gaussian is packed for size: its position as three 24-bit fixed-point integers with a fractional-bit count from the header, its opacity, three colour components and three log-scales as one byte each, its rotation quaternion as three bytes (version 2) or as its smallest three components in four bytes (version 3 on), and its higher spherical-harmonic coefficients as one byte each. The attributes are stored one array after another, not one record per Gaussian.

Versions 1 to 3 are a 16-byte header (``NGSP`` magic, version, point count, harmonic degree, fractional bits, flags) followed by the arrays, the whole compressed with gzip. Version 4 is a 32-byte plaintext header, a table of contents and each array compressed on its own with zstd. A flag marks a splat trained with antialiasing, another vendor extension records.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - compression
     - gzip (v1-3) or zstd per stream (v4)
   * - header
     - ``NGSP``, version, count, SH degree, fractional bits, flags
   * - position
     - 24-bit signed fixed point (v1: float16)
   * - opacity, colour, scale
     - one byte per component: sigmoid, DC colour, log scale
   * - rotation
     - x, y, z bytes (v1-2), smallest three in 32 bits (v3-4)
   * - harmonics
     - degree 0-4, one byte each, colour channel fastest
   * - elements
     - none

.. rst-class:: px-speclink

`SPZ reference implementation ↗ <https://github.com/nianticlabs/spz>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    cloud = px.read("scene.spz")
    cloud.vertex_attrs["opacity"]          # a logit, as in a splat PLY
    cloud.vertex_attrs["rot_0"]            # w; rot_1..3 are x, y, z
    cloud.vertex_attrs["f_rest_0"]         # the first higher harmonic, red
    cloud.global_attrs["spz_antialiased"]  # True when the flag is set
    px.write(cloud, "scene.ply")           # a splat PLY other tools load

Writing
-------

.. code-block:: python

    px.write(cloud, "out.spz")                      # version 3, gzip
    px.write(cloud, "out.spz", version=4)           # zstd streams
    px.write(cloud, "out.spz", fractional_bits=10)  # fixed-point precision

``version`` is 2, 3 or 4, 3 by default: the gzip layout with the finer smallest-three rotations, which needs nothing past the standard library. 2 stores rotations in three bytes, for readers older than version 3. 4 is the reference library's current default and needs ``compression.zstd`` (Python 3.14) or ``pip install "polyxios[zstd]"``. ``fractional_bits`` sets the fixed-point precision; without it twelve are used, as the reference writer does, or fewer when a position falls outside the ``[-2048, 2048)`` twelve leave. A cloud of no point is written with a warning: polyxios reads it back, the reference reader refuses it. So is a version 4 file holding more than 1024 times its size in positions, as a cloud of many repeated points compresses: the reference reader refuses it too, and ``version=3`` has no such bound. Any other option is warned about and ignored.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The Gaussians are decoded into the attributes a 3D Gaussian splatting PLY carries - ``f_dc_0..2``, ``f_rest_<k>`` with the colour channel slower varying, ``opacity``, ``scale_0..2``, ``rot_0..3`` as ``w, x, y, z`` - so an SPZ read here writes to ``.ply`` as a splat PLY, and one read from a splat PLY writes to ``.spz``. Values are float32, decoded as the reference decodes them: on the reference library's sample scenes every attribute matches it bit for bit except, now and then, the last bit of the quaternion component derived from the other three, which depends on whether the reference build fuses multiply-adds.
- Writing quantises as the reference writer does - the bytes of a version 2 or 3 payload and the whole of a version 4 file come out identical to its own - so precision is lost on the way in that no read gives back. A rotation re-encoded can come back as its negative, the same rotation.
- An opacity stored as 0 or 255 reads as an infinite logit, as it does in the reference.
- A cloud lacking a splat attribute is written with zeros for it - a grey, half-opaque, unit-scale Gaussian - and an identity rotation, with a warning; a rotation of zero length or not finite is written as the identity, with a warning. The ``f_rest_<k>`` run is split into three colour channels of a third of it each, as a splat PLY lays them out and the reference PLY reader splits them; the harmonic degree written is the highest whose coefficients fit in one channel, and each channel's coefficients past it, every other vertex attribute, the elements and the tags are dropped with a warning.
- Positions are kept as the file spells them; no axis convention is converted. A position that is not finite, or that reaches past what 24 bits hold, is refused on write.
- Vendor extension records are skipped with a warning: they follow the arrays in versions 1 to 3 and sit between the header and the table of contents in version 4. Other bytes following the arrays of a version 1 to 3 file, or lying before the table of contents of a version 4 file with no extension flag, are skipped with a warning; in version 4 nothing may follow the last stream, as in the reference reader. Skipped bytes are counted as they are read and never held, so a gzip stream that inflates to far more than the splat needs costs time, not memory. A file without the ``NGSP`` magic, of a version past 4, of a harmonic degree past 4, or shorter than its point count needs is refused; a version 4 file whose table of contents does not match its point count or its length is refused before any stream is decompressed, a stream longer or shorter than zstd ever makes one of its declared size is refused, and a stream that does not decompress to exactly its declared size is refused. The point count is held to what zstd can pack into each stream, not to the reference reader's tighter 1024:1, so a file of many repeated points it refuses reads here.
- ``lazy=True`` raises ``LazyReadError``: every attribute is quantised and compressed.
