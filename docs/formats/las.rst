.. _format-las:

LAS / LAZ (ASPRS LiDAR)
=======================

.. rst-class:: px-badges

``.las`` ``.laz`` ``read + write`` ``lazy: .las``

Summary of the specification
----------------------------

LAS is the ASPRS exchange format for LiDAR point clouds and the one every airborne, mobile and terrestrial scanning pipeline can produce or consume; LAZ is the same file with its point records compressed by LASzip. A 227-byte public header (235 in 1.3, 375 in 1.4) names the version, the point data record format, the record length, the point count, the ``X Y Z`` scale factors and offsets that turn the stored ``int32`` coordinates into coordinates, and a bounding box. Variable length records (VLRs) follow it: the coordinate reference system as GeoTIFF keys or, from 1.4, OGC WKT; an "extra bytes" descriptor naming the fields a writer appended to each record; the LASzip parameters of a ``.laz``. Then the point records, one fixed-size struct each, in one of eleven formats: 0 to 5 with 3-bit return counts and a 5-bit classification, 6 to 10 with 4 bits and a full byte, ``gps_time``, ``red green blue``, ``nir`` and a wave packet added by format. LAS 1.4 puts extended VLRs after the points.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - ``LASF``, version 1.0-1.4, point format 0-10, record length, point count (64-bit in 1.4), scale and offset per axis, bounding box
   * - coordinates
     - ``int32`` ``X Y Z``; ``x = X * scale + offset``
   * - point formats
     - 0 base · 1 +gps · 2 +rgb · 3 +gps+rgb · 4 +gps+wave · 5 +gps+rgb+wave · 6 new base +gps · 7 +rgb · 8 +rgb+nir · 9 +wave · 10 +rgb+nir+wave
   * - extra bytes
     - bytes past the format's record, described by the ``LASF_Spec`` / 4 VLR: name, type, scale, offset
   * - CRS
     - ``LASF_Projection`` VLRs 34735/34736/34737 (GeoTIFF keys) or 2112 (OGC WKT)
   * - compression
     - ``.laz``: point format bit 7 set, a ``laszip encoded`` VLR, LASzip chunks
   * - connectivity
     - none

.. rst-class:: px-speclink

`Full specification ↗ <https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    cloud = px.read("tile.las")                  # or tile.laz, with polyxios[laz]
    cloud.vertices                               # (n, 3) float64, scaled and offset
    cloud.vertex_attrs["intensity"]              # uint16, as the file holds it
    cloud.vertex_attrs["classification"]         # uint8
    cloud.vertex_attrs["return_number"]          # uint8, from the packed bits
    cloud.vertex_attrs["gps_time"]               # float64, formats 1, 3-10
    cloud.vertex_attrs["colors"]                 # (n, 3) floats in 0..1, formats 2, 3, 5, 7, 8, 10
    cloud.global_attrs["las_scale"]              # (3,) with las_offset, las_version, las_point_format
    cloud.global_attrs.get("crs_wkt")            # from an OGC WKT VLR
    cloud.global_attrs.get("las_geokeys")        # {key id: value} from the GeoTIFF VLRs

The stored ``X Y Z`` become the vertices through the header's scale and offset. Every field of the point record is a vertex attribute in the file's own dtype: ``intensity``, ``user_data``, ``point_source_id``, ``gps_time``, ``nir``, and each bit of the packed bytes as a ``uint8`` of its own - ``return_number``, ``number_of_returns``, ``scan_direction_flag``, ``edge_of_flight_line``, ``classification``, ``synthetic``, ``key_point``, ``withheld``, plus ``overlap`` and ``scanner_channel`` from format 6. The scan angle keeps the format's spelling: ``scan_angle_rank`` in whole degrees (``int8``) in formats 0 to 5, ``scan_angle`` in 0.006-degree steps (``int16``) from 6. ``red green blue`` fold into ``colors`` as floats in 0..1 over the 16-bit range. The wave packet of formats 4, 5, 9 and 10 reads as ``wavepacket_index``, ``wavepacket_offset``, ``wavepacket_size``, ``return_point_wave_location``, ``x_t``, ``y_t`` and ``z_t``; the waveform samples themselves, in an EVLR or a ``.wdp`` sidecar, are not read.

Extra bytes described by their VLR are attributes of their own name and dtype, two- or three-wide when the descriptor says so, scaled and offset when it says so; bytes past the format's record that no descriptor names are one ``extra_bytes`` block of ``uint8``. A descriptor whose name is already taken is read with a number appended, with a warning; one with a blank name is read as ``extra_<i>``, ``i`` its position among the described fields.

``global_attrs`` carries ``las_version`` (``"1.2"``), ``las_point_format``, ``las_scale`` and ``las_offset``, the ``las_system_identifier`` and ``las_generating_software`` strings when set, ``las_file_source_id``, ``las_global_encoding`` and ``las_creation_date`` (``(year, day of year)``) when nonzero; an OGC WKT VLR as ``crs_wkt``, the GeoTIFF key directory resolved through its doubles and ASCII twins as ``las_geokeys``; and every other VLR or EVLR, raw, in ``las_vlrs`` as ``{"user_id", "record_id", "description", "data"}``.

Writing
-------

.. code-block:: python

    px.write(cloud, "out.las")                                   # format from the attributes present
    px.write(cloud, "out.laz")                                   # LASzip, needs polyxios[laz]
    px.write(cloud, "out.las", point_format=6, version="1.4")
    px.write(cloud, "out.las", scale=0.01, offset=(400000.0, 5000000.0, 0.0))
    px.write(px.read("tile.las", lazy=True), "copy.las")          # stored ints pass through, lossless

The point format is ``las_point_format`` from ``global_attrs`` when there is one, else the smallest that holds the attributes present: ``gps_time`` asks for 1, ``colors`` for 2, both for 3, a wave packet for 4 or 5, and ``nir``, ``scan_angle``, ``overlap`` or ``scanner_channel`` for 6 to 10; ``point_format=`` overrides it, and an attribute the chosen format has no field for is dropped with a warning. The version is ``las_version`` raised to what the format and the count need, else the lowest they allow (1.2, 1.3 for a wave packet, 1.4 from format 6 or past 2\ :sup:`32` points); ``version=`` overrides it, 1.0 and 1.1 are written only when it asks for them, and a format the given version predates is refused. The scale is ``las_scale`` or 0.001 per axis, the offset ``las_offset`` or zero when every coordinate then fits an ``int32`` and the rounded lower corner of the bounding box when not; a coordinate that still does not fit, or a NaN, is refused. Vertices of an integer dtype whose ``global_attrs`` carry both ``las_scale`` and ``las_offset`` - what a lazy read hands back - are the stored ``X Y Z`` themselves: they go out untouched at that scale and offset, so a lazily read file is written back bit for bit, and are rescaled through ``int * las_scale + las_offset`` when ``scale=`` or ``offset=`` asks for another.

Every standard attribute goes back into its field, ``return_number`` and ``number_of_returns`` defaulting to 1 and the rest to 0, a float rounded to the nearest integer; a value its field cannot hold - an ``intensity`` past 65535, a ``return_number`` of 8 in formats 0 to 5, a ``classification`` above 31 there - is refused, as is an attribute that is not one row per vertex. A ``scan_angle_rank`` written into a format from 6 on, or a ``scan_angle`` into one before, is converted between degrees and 0.006-degree steps. ``colors`` as floats in 0..1 scale to 16 bits, as ``uint16`` are written as they are, and as any other integer count 0..255 spread over the 16. Every other numeric vertex attribute of one to three columns and a name of 1 to 32 printable ASCII bytes (trailing whitespace stripped, as a reader strips it) goes out as an extra bytes field of its own name and dtype, with the descriptor VLR to match, and an ``extra_bytes`` block of ``uint8`` columns goes back undescribed; one that does not fit, or is named like a raw dimension (``X``, ``red``, ``flags``, ...), is dropped with a warning. ``crs_wkt`` writes the OGC WKT VLR and sets the header's WKT bit, which is cleared when there is no ``crs_wkt``; ``las_geokeys`` writes the three GeoTIFF VLRs, a key or value it cannot spell dropped with a warning, and ``las_vlrs`` its entries as they are - one over 65535 bytes as an EVLR in 1.4, dropped with a warning before, as is one the writer builds itself. The generating software defaults to ``polyxios``, and it and the system identifier are cut to their 32 ASCII bytes, or at a NUL, with a warning; the creation date is left blank unless ``las_creation_date`` is given.

A destination named ``.laz`` is compressed and any other is not; ``compress=`` overrides the name, and a nameless buffer is written plain unless it is passed.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Elements are not part of the format: a mesh's faces are dropped with a warning, and every file reads back with no elements at all.
- Which of ``.las`` and ``.laz`` a file is comes from its header, not its name: a compressed file under either name is decompressed, needing lazrs (``pip install "polyxios[laz]"``), and a plain one under ``.laz`` reads as it is. A ``.laz`` cannot be read lazily.
- Coordinates are quantised on write: at the default scale of 0.001 a coordinate keeps a millimetre. Pass ``scale=`` for finer, or read back ``las_scale`` from a file to write it at its own.
- ``lazy=True`` maps an uncompressed file and hands back views of it, read-only and in the file's own dtypes: the vertices are the ``(n, 3)`` ``int32`` the file stores, for you to scale by ``las_scale`` and shift by ``las_offset``, ``colors`` the ``uint16`` triples, every other field and extra byte a view of its column. The bit fields are decoded into small arrays of their own, and an extra byte carrying a scale or offset is left raw with a warning naming it.
- Every header field the reader relies on is checked before anything is allocated: the signature, version and point format, a header shorter than its version needs, an offset to point data inside the VLRs, a record shorter than its format, a count of points past the end of the file or, in a ``.laz``, past what its chunk table can hold, an extra bytes VLR describing more bytes than the record holds, an EVLR placed outside the file. A ``.laz`` is then expanded one chunk at a time, so a count the chunk table allows but the stream does not hold is refused having cost one chunk of memory, not the records it claimed.
- The waveform samples of formats 4, 5, 9 and 10 are not read or written; the per-point wave packet fields are.
- LAS 1.4 asks a file in formats 6 to 10 to carry its CRS as OGC WKT with the header's WKT bit set. The writer sets the bit and writes the VLR only when ``crs_wkt`` is given; a cloud without one goes out in those formats with neither, which a reference implementation reads as it does every file polyxios writes: the test suite checks, when that implementation is installed, that it reads back the coordinates, colours, ``gps_time``, extra bytes, WKT and GeoTIFF keys of a written ``.las`` and ``.laz``, and that polyxios reads what it writes.

.. seealso::

   :doc:`pcd` - the Point Cloud Library's own format.
   :doc:`xyz` - the ASCII column formats scanners export.
   :doc:`index` - the full format table.
