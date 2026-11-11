.. _format-png:

PNG
===

.. rst-class:: px-badges

``.png`` ``read + write`` ``eager``

Summary of the specification
----------------------------

PNG is a lossless raster format: an 8-byte signature, then a run of chunks, each a big-endian length, a four-letter type, the data and a CRC-32 of the type and data. ``IHDR`` gives the width, height, bit depth (1, 2, 4, 8 or 16), color type (gray, RGB, palette, gray with alpha, RGB with alpha) and whether the rows are interlaced; the pixels sit in one or more consecutive ``IDAT`` chunks as a single zlib stream of rows, each led by a filter byte (none, sub, up, average, Paeth) that predicts the row from its neighbors, and ``IEND`` ends the file. Interlaced files hold seven Adam7 passes, each a smaller image. A ``PLTE`` chunk, critical for a palette image, carries the palette; ancillary chunks carry transparency (``tRNS``), resolution (``pHYs``), gamma (``gAMA``), an ICC profile (``iCCP``), Exif and text.

polyxios decodes and encodes the format with NumPy and the standard library's ``zlib``; there is no Pillow dependency.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - signature
     - ``89 50 4E 47 0D 0A 1A 0A``
   * - chunks
     - length, type, data, CRC-32; ``IHDR`` first, ``IEND`` last
   * - color types
     - 0 gray, 2 RGB, 3 palette, 4 gray + alpha, 6 RGB + alpha
   * - depths
     - 1, 2, 4, 8, 16 bits (which a color type allows varies)
   * - pixels
     - one zlib stream of filtered rows; Adam7 interlacing optional
   * - elements
     - a lattice of quads, one vertex per pixel (``read()``)

.. rst-class:: px-speclink

`Read the full PNG specification ↗ <https://www.w3.org/TR/png-3/>`__

Reading
-------

An image is a grid of samples, not a mesh. :func:`polyxios.read_image` returns the compact :class:`~polyxios.ImageData`, one sample per pixel; :func:`polyxios.read` expands it into a :class:`~polyxios.PolyData` lattice with a vertex per pixel and a quad between four of them, and warns to say so.

.. code-block:: python

    import polyxios as px
    from polyxios import helper

    image = px.read_image("photo.png")
    image.point_attrs["colors"]               # (1, H, W, 3) uint8, y up
    pixels = helper.image_to_array(image)     # (H, W, 3), row 0 on top
    image.global_attrs["dpi"]                 # when the file has pHYs in meters

    mesh = px.read("photo.png")               # warns: use read_image()

Samples come back as ``uint8``, or ``uint16`` for a 16-bit file, under ``'colors'`` (RGB or RGBA) or ``'intensity'`` (gray, with ``'alpha'`` beside it for gray + alpha). Gray at 1, 2 and 4 bits is scaled to 0-255. A palette is applied: the result is RGB, or RGBA when ``tRNS`` names alpha entries. A ``tRNS`` key on a gray or RGB file adds an alpha channel.

Writing
-------

.. code-block:: python

    px.write_image(helper.array_to_image(pixels), "out.png")
    px.write_image(image, "out.png", compress_level=9)

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: px-spec-table

   * - option
     - meaning
   * - ``compress_level``
     - zlib level for the image data and the ICC profile, 0 (none) to 9 (smallest); 6 by default.

Each row takes whichever of the five filters leaves the smallest sum of absolute residuals. The mode follows the image: gray, gray + alpha, RGB or RGBA, at 8 or 16 bits by dtype. Floats in [0, 1] are scaled to 0-255 and integers within 0-255 or 0-65535 are narrowed; anything else (NaN, out of range, ``bool``) is refused. :func:`polyxios.write` takes a lattice mesh and recovers the grid first.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- ``global_attrs`` holds ``dpi`` (a pair, from ``pHYs``), ``gamma`` (from ``gAMA``), ``icc_profile`` (bytes), ``exif`` (bytes) and ``text`` (a dict of ``tEXt``, ``zTXt`` and ``iTXt``, the first of a repeated key kept, with a warning). The writer spells these five and warns about every other ``global_attrs`` key, about extra point attributes, cell attributes and about a non-zero origin or an x or y step whose size is not 1, which a file has no place for. A negative y step flips the rows, as ``helper.image_to_array`` does, and a negative x step is not mirrored.
- Rows are top-first in the file and bottom-first in an ``ImageData`` (VTK's y axis points up), so ``read_image`` flips and the writer flips back. ``helper.image_to_array`` flips once more by default; ``origin="lower"`` keeps VTK's order.
- ``text`` goes out as ``tEXt`` when it is Latin-1 and as an uncompressed ``iTXt`` otherwise.
- ``exif`` is written without a leading ``Exif`` and two NUL bytes, the JPEG ID code that Pillow and cameras put in front of it: an ``eXIf`` chunk holds the TIFF data alone.
- No interlaced and no palette output: an Adam7 or palette file is read, and written back as a plain RGB(A) or gray one.
- A truncated or corrupt file is refused naming itself: a bad CRC, a missing ``IEND``, a filter byte above 4 (naming the Adam7 pass when interlaced), ``IDAT`` chunks that are not consecutive, an unknown critical chunk, a gray or RGB ``tRNS`` of the wrong length or a palette ``tRNS`` longer than the palette. A ``tRNS`` on a color type that already has alpha is ignored. A palette index past the palette reads as opaque black, as the specification requires; a malformed ``pHYs``, ``gAMA``, ``iCCP`` or text chunk, or a ``tRNS`` after ``IDAT``, warns and is ignored. Unknown ancillary chunks (APNG's ``acTL``, ``fcTL`` and ``fdAT`` among them) are ignored, so an animated PNG reads as its default image. Bytes after ``IEND`` are ignored; ``IDAT`` data in excess of the image warns.
- ``read_image`` holds the width and height to the same hard caps as the ``.vti`` image reader, before the pixels are allocated; ``read()`` also applies the cap on expanded vertices.
- ``read(..., lazy=True)`` raises :class:`~polyxios.exceptions.LazyReadError`: the pixels are compressed. ``read_image`` warns about, and ignores, any option; ``write_image`` takes ``compress_level`` alone.

.. seealso::

   :doc:`index` - the full format table.
   :doc:`vti` - the VTK image a PNG converts to and from without expanding.
