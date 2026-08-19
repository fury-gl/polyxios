.. _changes:

=========
Changelog
=========

.. _changes_0.4.0:

0.4.0 (upcoming)
----------------

New features
~~~~~~~~~~~~

- ``read()`` and ``write()`` now take an open binary file object wherever
  they take a path, so a mesh round-trips through ``io.BytesIO``, a socket
  or a file inside an archive without touching disk. A handle polyxios was
  given is read or written where it stands and is never closed. A buffer
  with no file name cannot have its format inferred, so ``fmt=`` is
  required there; a handle from ``open()`` carries its own extension and
  does not need it. TetGen is the exception: a ``.node``/``.ele`` pair is
  two files and still needs a path.
- Lazy reads over a file object work when the handle is backed by a real
  file and stands at its start - mmap needs a descriptor and addresses a
  file from byte zero - and raise ``LazyReadError`` naming the reason for an
  in-memory buffer or a handle part-way into a file. Only the formats whose
  lazy read hands back arrays viewing the mapping ask for that. Binary STL's
  lazy mode copies what it reads - it skips vertex deduplication and nothing
  else - so it takes a buffer like any other read rather than sending the
  caller to an eager read that would merge the vertices it was asked to keep.
- gzip is now transparent for every format: a file opening with the gzip
  magic is decompressed on the way in, whatever it is named, and a
  destination named ``.gz`` is compressed on the way out. ``.vol.gz`` is
  read as Netgen rather than refused, ``.gz`` names the compression rather
  than the format when a codec is chosen - in ``fmt=`` as well as in a file
  name - and the compressed output is byte-reproducible (no timestamp, no
  embedded name). ``fmt=".obj.gz"`` is how a nameless buffer asks for
  compression, since it has no name to end in ``.gz``. A compressed member
  need not run to the end of what it sits in, so a mesh gzipped into the
  middle of an archive reads without the bytes after it becoming an error,
  and a file holding several members back to back - what ``cat a.gz b.gz``
  leaves - is read and measured as the whole it decompresses to. A path and
  a file object go through the same reader, so one compressed file reads the
  same way and fails the same way whichever it was handed over as. A
  destination that compresses on its own, such as a handle from
  ``gzip.open()``, is written to as it is rather than compressed a second
  time. Lazy reads still need an uncompressed file and say so, and TetGen -
  which opens its own sibling files rather than going through this layer -
  refuses a compressed file rather than parsing it as text, by its content on
  the way in and by its name on the way out, and for whichever half of the
  pair carries it rather than only for the one the caller named. Whether a
  source is compressed is decided by the whole four-byte gzip header rather
  than by the two magic bytes alone: those open one file in every 65536 by
  chance,
  and in a headerless binary format they are an ordinary coordinate's low
  mantissa bytes - a ``.splat`` whose first x is 10.658965 opens with exactly
  them and is a mesh, not an archive.
- The formats that check a header against the size of the file it came from
  now take that size from the read they were already making rather than
  measuring the source separately. Measuring a compressed one cost a whole
  decompression pass that the read then repeated, and a stream that cannot
  seek could not be measured at all, so legacy VTK, the VTK XML formats and
  ``.splat`` read one pass faster and from more kinds of source than before.
- Extensions several unrelated formats share are now resolved by looking
  inside the file. A codec declares ``SNIFF_EXTENSIONS``, a
  ``sniff(head) -> bool`` test and a ``SNIFF_PRIORITY``; the contested
  extension resolves to a dispatcher that delegates to the first codec
  recognising the opening bytes, and names the candidates when none does.
  Writing to such an extension raises, an output file having no content to
  inspect.
- ``.dat`` is the first user of this: a Tecplot header resolves to the
  Tecplot codec, a bulk data card to the Nastran one. It no longer needs
  ``fmt=``, which still works for the cases sniffing cannot settle.
- ``.plt`` is registered to the Tecplot codec so a binary Tecplot file is
  told what is wrong instead of resolving nowhere. Reading it is not
  supported.

Behaviour changes
~~~~~~~~~~~~~~~~~

- Text formats are written with ``\n`` line endings on every platform.
  Writing used to go through ``Path.write_text``, which turned them into
  ``\r\n`` on Windows; every write now goes through the same binary path a
  buffer does, so the bytes a path receives and the bytes a ``BytesIO``
  receives are the same ones. Every reader already accepted either ending.

Bug fixes
~~~~~~~~~

- A broken binary file now reports what is wrong with it rather than
  ``BufferError: cannot close exported pointers exist``. The formats that
  parse over a mapping - ``.ply``, legacy ``.vtk``, ``.meshb`` - build their
  arrays as views of it, and a parse that gives up half-way leaves one of
  those alive in the traceback carrying the failure; unmapping underneath it
  then raised, and that error replaced the codec's own. The mapping is left
  to the last view of it instead, so a corrupt file says the same thing read
  from a path as it does read from a buffer, where there was never a mapping
  to unmap.
- A source that answers a read with less than it was asked for is now read to
  the end of the request. A bare ``read`` is allowed to come up short - a
  socket hands back what has arrived, not what was wanted - and the wrapper
  put in front of a duck-typed handle passed that straight through, so a
  codec asking for n bytes could silently get fewer and parse the gap as
  data. A handle from ``open()`` never came up short, so no codec guarded
  against it. Nothing is read past what was asked for, so a handle the caller
  shares still ends up where the codec's reading left it.

Tests
~~~~~

- A cross-codec round-trip matrix (``tests/test_roundtrip.py``) writes and
  re-reads five canonical meshes through every writable codec, checked
  against a table declaring exactly what each format keeps. A new codec
  cannot join the registry without an entry.

.. _changes_0.3.0:

0.3.0 (2026-08-18)
------------------

Fifteen new mesh formats, a real command line interface and a versioned
documentation site.

New features
~~~~~~~~~~~~

- Fifteen new codecs, taking the registry from 16 to 34 recognised
  extensions:

  - Abaqus ``.inp`` (C3D4 / C3D8 / S3 / S4 and friends).
  - AVS-UCD ``.avs``, preserving ``mat_id``.
  - DOLFIN / FEniCS XML ``.xml``.
  - FLAC3D ``.f3grid`` (zones and faces).
  - Gmsh ``.msh`` - ASCII v2 read/write, v4.1 read, physical groups kept.
  - Medit binary ``.meshb`` (GmfLib v1 and v2, zero-copy mmap).
  - Nastran ``.bdf``, also registered for ``.nas`` and ``.fem``, reading
    free, small and large field formats.
  - Netgen ``.vol`` (points, edges, faces, cells, tags).
  - OFF ``.off`` (ASCII and binary, colours, normals, texture coordinates).
  - STL ``.stl`` (binary and ASCII, lazy loading for binary).
  - SU2 ``.su2`` (VTK element codes, boundary markers).
  - Tecplot ``.tec`` (ASCII FE zone, POINT and BLOCK ordering).
  - TetGen ``.node`` + ``.ele`` pairs (markers, regions).
  - UGRID ``.ugrid`` (AFLR ASCII, boundary tags).
  - WKT ``.wkt`` (Well-Known Text).

- ``pxios`` command line interface with ``fetch``, ``list``, ``convert`` and
  ``viz`` subcommands. Visualization requires the optional ``viz`` extra
  (``pip install polyxios[viz]``).
- ``polyxios.read_polydata`` and ``polyxios.visualize_mesh`` helpers, plus
  ``polyxios.supported_extensions`` to introspect the codec registry.
- Versioned documentation at https://polyxios.org, with a version switcher,
  a credits page generated from the git history, and a page per format.

Changes
~~~~~~~

- The fetcher now resolves assets through a remote ``models.json`` catalog and
  downloads files individually, instead of pinned per-format release zips.
  Downloads are checksum-verified and the result is cached across runs; set
  ``POLYXIOS_MODELS_URL`` to pin the catalog to an immutable URL.
- ``pxios fetch`` gained ``--verbose``, requires a ``sha256`` for every asset
  and resolves destination paths defensively.
- ASCII writing is vectorised, and the docs build now treats Sphinx warnings
  as errors.

Bug fixes
~~~~~~~~~

- Fetcher: rejected zip-slip paths, non-HTTPS URLs and stale cache entries,
  added timeouts, retried dropped transfers, and stopped ``example --list``
  from downloading the whole pack.
- STL: fixed a file-descriptor leak, ASCII decoding, lazy-format detection,
  trailing-data handling and zero normals on degenerate triangles.
- WKT: fixed ring grouping, decoding and degenerate rings, and hardened
  parsing and hole encoding.
- Gmsh: corrected the element type codes.
- Nastran: fixed marker parsing, exponent handling and write precision.
- ``resolve()`` now accepts a dotless ``fmt`` override.
- Fixed multi-channel vertex attributes in ``merge`` and ``vertex_colors``,
  offset validation and ASCII newlines in the VTK XML path, and restored
  multiblock partial loading and visualization colours.


GitHub stats for 2026/06/25 - 2026/08/18 (tag: v0.2.0)

These lists are automatically generated and may be incomplete or contain duplicates.

The following 3 authors contributed 97 commits.

* Praneeth Shetty
* Serge Koudoro
* dependabot[bot]


We closed a total of 29 issues, 29 pull requests and 0 regular issues.

Pull Requests (29):

* :ghpull:`43`: CI: publish versioned docs from the tag push
* :ghpull:`42`: NF: warnings as error
* :ghpull:`41`: MNT: update pre-commit hooks
* :ghpull:`40`: DOC: SEO metadata, em dash removal, copy icon, mobile fixes
* :ghpull:`39`: NF: new polyxios website
* :ghpull:`38`: NF: add Netgen .vol codec
* :ghpull:`37`: NF: add UGRID ASCII .ugrid codec (AFLR format, boundary tags)
* :ghpull:`36`: NF: add TetGen codec
* :ghpull:`35`: NF: add SU2 codec
* :ghpull:`34`: NF: add Tecplot ASCII .tec codec (FE zone, POINT + BLOCK)
* :ghpull:`33`: NF: add OFF codec (ASCII + binary, colours/normals/texcoords)
* :ghpull:`32`: NF: add Nastran .bdf codec (free/small/large field read)
* :ghpull:`23`: NF: Add WKT (Well-Known Text) codec
* :ghpull:`31`: NF:  Adding gmsh codec
* :ghpull:`21`: NF: add FLAC3D .f3grid codec (T4/P5/W6/B8 zone types)
* :ghpull:`20`: NF: Adding CLI support `pxios`
* :ghpull:`29`: MNT: update pre-commit hooks
* :ghpull:`28`: MNT: Bump pypa/cibuildwheel from 4.1.1 to 4.2.0 in the actions group
* :ghpull:`26`: MNT: update pre-commit hooks
* :ghpull:`27`: MNT: Bump pypa/cibuildwheel from 4.1.0 to 4.1.1 in the actions group
* :ghpull:`25`: MNT: Bump actions/setup-python from 6 to 7 in the actions group
* :ghpull:`24`: MNT: update pre-commit hooks
* :ghpull:`22`: MNT: update pre-commit hooks
* :ghpull:`19`: Feat/dolfin codec
* :ghpull:`18`: NF: add Medit .meshb codec (GmFlib binary, v1/v2, lazy mmap support)
* :ghpull:`17`: NF: add AVS-UCD .avs codec (ASCII, mat_id preserved)
* :ghpull:`16`: NF: add Abaqus .inp codec
* :ghpull:`15`: MNT: update pre-commit hooks
* :ghpull:`14`: NF: add STL codec with binary/ASCII read-write and lazy binary support

Issues (0):


.. _changes_0.2.0:

0.2.0 (2026-06-25)
------------------

First public release of **polyxios**.

New features
~~~~~~~~~~~~

- Plugin-based codec registry via Python entry points - third-party packages
  can register mesh formats without patching polyxios.
- VTK legacy (``.vtk``) and XML (``.vtu``, ``.vtp``) reader/writer with
  ASCII and binary (raw + appended) encoding.
- VTR appended format support.
- MFEM mesh codec (``.mesh``).
- MEDIT mesh codec (``.mesh``).
- ``polyxios convert`` and ``polyxios visualize-mesh`` CLI commands.
- Lazy / memory-mapped loading for binary formats (``read(..., lazy=True)``).
- ``polyxios.__version__`` exposes the full version string including the git
  commit hash for development builds (e.g. ``0.1.0.dev0+git20260623.101006a``).

GitHub stats for 2026/05/26 - 2026/06/25 (tag: None)

These lists are automatically generated and may be incomplete or contain duplicates.

The following 4 authors contributed 47 commits.

* Maharshi Gor
* Praneeth Shetty
* Serge Koudoro
* skoudoro


We closed a total of 12 issues, 12 pull requests and 0 regular issues.

Pull Requests (12):

* :ghpull:`12`: NF: add MFEM mesh codec (.mesh)
* :ghpull:`9`: NF: Handle the other vtk formats
* :ghpull:`11`: MNT: update pre-commit hooks
* :ghpull:`10`: BF/NF: fix PLY binary reader + add SPLAT codec and compressed 3DGS support
* :ghpull:`8`: BF: handling VTK files improvements
* :ghpull:`4`: Fix: vtk codec to read ascii polydata and support v1.0
* :ghpull:`7`: CI: Avoid cron job on fork
* :ghpull:`6`: MNT: update pre-commit hooks
* :ghpull:`3`: NF: Adding Data Fetcher
* :ghpull:`5`: MNT: update pre-commit hooks
* :ghpull:`2`: DOC: Replace arrow and em-dashes with dash
* :ghpull:`1`: NF: initial framework from polyxios

Issues (0):
