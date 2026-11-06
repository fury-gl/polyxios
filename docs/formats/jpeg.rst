.. _format-jpeg:

JPEG
====

.. rst-class:: px-badges

``.jpg`` ``.jpeg`` ``.jpe`` ``.jfif`` ``read + write`` ``eager``

Summary of the specification
----------------------------

JPEG (ITU T.81) is a family of image codings sharing one file structure: marker segments (``FF`` and a code), each with a length, ahead of entropy-coded scans. The common lossy coding splits the image into 8x8 blocks of each component, takes their discrete cosine transform, divides by a quantization table, and Huffman-codes the result. Color images are usually stored as YCbCr with the chroma planes subsampled. A *baseline* file holds one scan; a *progressive* one holds several, building the coefficients up in spectral bands and bit planes. A separate *lossless* process (``SOF3``) skips the transform: each sample is predicted from its neighbors and the difference is Huffman-coded.

polyxios decodes baseline, extended-sequential and progressive Huffman files at 8 bits, and lossless Huffman files at 2 to 16 bits, and writes baseline lossy and lossless files, with NumPy alone.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - markers
     - ``SOI``, segments (``APPn``, ``DQT``, ``DHT``, ``DRI``, ``SOFn``, ``SOS``, ``COM``), ``EOI``
   * - lossy
     - ``SOF0``, ``SOF1`` and ``SOF2`` (progressive), 8 bits, Huffman
   * - lossless
     - ``SOF3``, 2-16 bits, predictors 1-7, Huffman
   * - components
     - 1 (gray) or 3 (YCbCr, or RGB when the file says so)
   * - subsampling
     - 4:4:4, 4:2:2, 4:2:0 on write; any factors 1-4 on read
   * - elements
     - a lattice of quads, one vertex per pixel (``read()``)

.. rst-class:: px-speclink

`ITU T.81 ↗ <https://www.w3.org/Graphics/JPEG/itu-t81.pdf>`__

Reading
-------

.. code-block:: python

    import polyxios as px
    from polyxios import helper

    image = px.read_image("photo.jpg")
    pixels = helper.image_to_array(image)    # (H, W, 3) uint8, row 0 on top
    image.global_attrs["exif"]               # bytes, without the "Exif" prefix

    mesh = px.read("photo.jpg")              # warns: use read_image()

The result is ``'intensity'`` for one component and ``'colors'`` for three. A three-component file is YCbCr unless an Adobe ``APP14`` segment says transform 0, or the component ids spell ``R``, ``G``, ``B``. Chroma is upsampled the way libjpeg's default "fancy" upsampling does for 2:1 and 2x2:1 ratios, and replicated for others; a decode agrees with Pillow's to within a few levels (never bit for bit, the IDCT is exact floating point rather than libjpeg's integer one). A lossless file is returned as stored: ``uint8`` up to 8 bits, ``uint16`` above, not rescaled, with no color transform. EXIF orientation is not applied.

Writing
-------

.. code-block:: python

    px.write_image(image, "out.jpg")                              # quality 75, 4:2:0
    px.write_image(image, "out.jpg", quality=95, subsampling="4:4:4")
    px.write_image(image, "out.jpg", lossless=True)               # SOF3, exact

``quality`` is 1-100 and defaults to 75; its scaling of the Annex K tables is libjpeg's. ``subsampling`` is ``"4:4:4"``, ``"4:2:2"`` or ``"4:2:0"`` (the default). ``lossless=True`` writes predictor 1 at the dtype's own precision (8 or 16 bits) with no color transform, so ``read_image`` gives the array back exactly; ``quality`` and ``subsampling`` are refused beside it. An image with alpha is refused: JPEG holds none.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- ``global_attrs`` holds ``dpi`` (from ``JFIF`` density), ``exif``, ``icc_profile`` and ``comment``; the writer spells these four and warns about every other key. A lossless RGB file carries no ``JFIF`` segment, since JFIF implies YCbCr, so its ``dpi`` is reported as left out.
- Refused with a message naming the variant: arithmetic-coded and hierarchical files, 12-bit lossy files, CMYK and YCCK, subsampled lossless files, a second frame, a zero height (``DNL``), a progressive AC scan of several components, a missing ``EOI``, an undefined table.
- Reading does not record how a file was encoded: a lossless file written back is lossy unless ``lossless=True`` is passed, and ``pxios convert`` always writes lossy JPEG.
- The width and height are checked against the same bound on implied vertices the ``.vti`` reader uses before any coefficient is allocated.
- ``lazy=True`` raises ``LazyReadError``: the pixels are compressed.
