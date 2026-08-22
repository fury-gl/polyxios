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
- ``PolyData.topological_dimension`` reports the highest dimension the mesh
  actually holds - 0 for points, 1 for lines, 2 for surfaces, 3 for volumes -
  taking the maximum over the element types present, so a tetrahedral mesh
  that also carries its boundary triangles still reads as 3-D. It is the
  dimension of the elements, not of the space they sit in: a triangle mesh
  embedded in 3-D is 2-D. An empty mesh is 0.
- ``transforms.merge_duplicate_vertices`` welds coincident vertices into
  one, the equivalent of ParaView's "Clean to Grid". Formats that write a
  corner per element - STL above all - hand back a soup of unconnected
  vertices, and welding is what turns it back into a surface. ``tol=``
  snaps coordinates to a grid of that step before comparing; the default
  welds only exactly equal ones, and a ``tol`` so small that snapping
  overflows to infinity is refused rather than welding every point it
  overflowed. The survivor of each group is its lowest original index and
  keeps its own coordinates and attributes, so the result does not depend
  on which duplicate the file listed first and a tolerance never moves a
  point. Welding is not culling: a vertex no
  element references is kept - compose with ``remove_orphan_vertices`` to
  drop those too.
- Two-dimensional files now follow one rule across every format that can
  spell one. A mesh always carries three coordinate columns, a file that
  declared two is padded with ``z=0``, and the fact is recorded in
  ``global_attrs["was_2d"]`` so a writer can drop the column again. It says
  something about the mesh rather than about the file it came from, so a
  plane read as a 2-D Netgen ``.vol`` - which has no two-dimensional
  spelling of its own to write back - still lands as an ``NDIME= 2`` SU2
  case. Medit, Medit binary, SU2, Netgen, TetGen, MFEM, DOLFIN, Tecplot,
  Abaqus and WKT all record it on the way in, and every one of those but
  Netgen restores it on the way out. Coordinates outrank the flag: a mesh
  flagged two-dimensional whose vertices have since left the plane is
  written in three, with a warning, rather than being flattened in silence.

Behaviour changes
~~~~~~~~~~~~~~~~~

- Medit ``.mesh`` records a two-dimensional file in
  ``global_attrs["was_2d"]`` rather than keeping the file's own number in
  ``global_attrs["medit_dimension"]``, which is now the shared spelling
  every format uses. A three-dimensional file sets nothing, where it used to
  set ``medit_dimension: 3``.
- Formats that had always written three coordinate columns now write two for
  a mesh that came from a two-dimensional file and has stayed flat: TetGen's
  ``.node`` header declares 2, a Tecplot zone declares ``X`` and ``Y``
  alone, an Abaqus node card carries two coordinates, and a ``.meshb``
  header declares ``Dimension 2``.

- Text formats are written with ``\n`` line endings on every platform.
  Writing used to go through ``Path.write_text``, which turned them into
  ``\r\n`` on Windows; every write now goes through the same binary path a
  buffer does, so the bytes a path receives and the bytes a ``BytesIO``
  receives are the same ones. Every reader already accepted either ending.

Bug fixes
~~~~~~~~~

- ``.meshb`` hands back vertices with three columns. A file declaring
  ``Dimension 2`` was read into an ``(n, 2)`` array, which is not what a
  ``PolyData`` holds: every consumer indexing ``vertices[:, 2]`` - the
  transforms, the other writers, ``faces`` - raised on it. The coordinates
  are padded with ``z=0`` like every other codec's, and the write side takes
  its ``Dimension`` from the mesh rather than from the width of the array.
- A tag group naming a vertex or an element that this mesh does not have no
  longer moves the label onto one that it does. ``remove_orphan_vertices``,
  ``merge_duplicate_vertices`` and ``filter_element_type`` carried a group
  through their index remap after dropping only the members past the end, so
  a negative index reached the array from the far end and landed on a real
  item, and a group holding floats - which nothing stops a reader from
  building - raised ``IndexError`` from inside numpy instead. All three now
  go through the same member check the writers use: a member that indexes
  nothing in this mesh is dropped, and what is left is remapped.
- The documented ``pipeline(filter_element_type(keep="triangle"), ...)``
  raised ``TypeError``: ``filter_element_type`` takes the mesh first and does
  not curry. The example now composes it with ``functools.partial``.
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
- OBJ face indices are resolved the way the format defines them: a negative
  index counts back from what has been declared so far, and an index naming
  a record the file does not have raises ``CodecError`` naming the line
  instead of wrapping around into a different vertex.
- OBJ ``vt`` records are read. They are indexed per face corner, so a file
  may hold more of them than it holds vertices; each corner assigns to its
  vertex, a vertex given two different values keeps the last and warns, and
  records nothing indexes are kept only when there is one per vertex.
  ``vt`` and ``vn`` are written back, so texture coordinates and normals
  survive a round trip.
- OBJ writes a bare ``g`` before a face that belongs to no group, so it no
  longer inherits the group of the face above it, and a bare ``g`` on read
  clears the active groups rather than inventing a ``default`` tag.
- An OBJ file whose ``vn`` or ``vt`` records cannot be lined up with its
  vertices now leaves the attribute out. The fold that gives up returned
  None, and only the ``vt`` path checked for it, so ``vertex_attrs`` could
  hand back a None where an array belongs and the writer raised
  ``TypeError`` on it.
- OBJ writes a number where a vertex has no record. A vertex no face names
  carries NaN out of the reader, and ``vt nan nan`` is not a record another
  reader takes; the row nothing indexes is written as zero. A ``vt`` array
  narrower than two columns is padded rather than indexed past its end.
- OBJ leaves out a ``vn`` or ``vt`` attribute that does not hold one row
  per vertex, with a warning naming its shape. Only the column count was
  checked, so an attribute with fewer rows than the mesh wrote faces
  indexing ``vt`` records that were not in the file - which no reader can
  take, this one included: a three-vertex mesh carrying one texture
  coordinate wrote a file that read back as ``CodecError``. A
  one-dimensional attribute is now read as one value per vertex rather
  than as a single row.
- Legacy ``.vtk`` files now read ``COLOR_SCALARS`` and ``NORMALS``, in both
  the ASCII and the binary flavour. Both used to stop the attribute scan,
  dropping the array and everything after it without a word. A binary
  ``COLOR_SCALARS`` component is an unsigned char standing for the 0..1
  float an ASCII file writes, and is scaled onto that range, so the same
  colour reads back the same from either flavour.
- Legacy ``.vtk`` reads ``TEXTURE_COORDINATES``, and steps over a
  ``LOOKUP_TABLE`` definition rather than stopping at it. Those were the
  last two attribute keywords the format defines that the scan did not
  know, and in a binary file an unknown keyword ends the scan: a file
  carrying either lost every array after it. A palette is not a value per
  point, so a lookup table becomes no attribute - it is only counted past.
- A legacy ``.vtk`` attribute keyword the reader does not know now warns
  that it and everything after it in that section are being dropped. Its
  payload is binary of unknown length, so the scan still cannot go on, but
  a short read is no longer a silent one.
- ``NORMALS`` and ``COLOR_SCALARS`` are read from ``STRUCTURED_POINTS``,
  ``STRUCTURED_GRID`` and ``RECTILINEAR_GRID`` files too. Those three
  datasets scan their attributes themselves rather than through the shared
  parser, and knew only ``SCALARS``, ``VECTORS`` and ``FIELD``.
- A binary legacy ``.vtk`` attribute that runs past the end of the file now
  raises ``CodecError`` naming the array and the bytes that are missing.
  The slice came up short in silence and the reshape after it failed with
  a ``ValueError`` naming neither the array nor the file.
- A legacy ``.vtk`` attribute section that declares more values than the
  file holds raises ``CodecError`` naming the array and the count, rather
  than running off the end of the line list with an ``IndexError`` that
  names nothing. This covers ``SCALARS``, ``COLOR_SCALARS``, ``VECTORS``,
  ``NORMALS``, ``TENSORS`` and ``FIELD``.
- A Nastran real field is now written in whatever spelling fits it. Bulk
  data allows the exponent's ``E`` to be dropped when its sign is there
  (``1.234-10``), and nothing reads ``+07`` differently from ``+7``, so the
  writer offers both - three columns back on the explicit form, which is
  three more significant digits in an eight-column field. A value at the
  top of the double range is stepped one digit toward zero rather than
  refused, since rounding it to nearest overflows to infinity. No finite
  coordinate is refused by a field any more.
- A Nastran real field prefers a spelling that reads back as the value it
  was given. The search spent precision one digit at a time and took the
  first form that fit, so a mantissa stepped toward zero could win a field
  an exact form two digits shorter would also have fit: ``1e7`` went out as
  ``9999999.`` where ``1.E+07`` was available, and ``1e15`` as
  ``999999999999999.`` in a large field. Exact spellings are now swept for
  first, at every precision, and only a value no field can hold exactly -
  the top of the double range - falls through to the closest one.
- The XML writers declare ``version="1.0"`` in the ``<VTKFile>`` header
  rather than ``version="0.1"``, which no VTK release ever defined.
  Reading a file that declares ``0.1`` is unchanged.
- A ``<DataArray>`` polyxios cannot decode - a ``type="String"`` label
  array, or any type it does not know - is skipped with a warning naming
  it, and the arrays around it are still read. It used to vanish without
  a word.
- A legacy ``STRUCTURED_POINTS`` file keeps its ``DIMENSIONS``, ``ORIGIN``
  and ``SPACING`` in ``global_attrs`` as ``vtk_dimensions`` /
  ``vtk_origin`` / ``vtk_spacing``, and ``STRUCTURED_GRID`` /
  ``RECTILINEAR_GRID`` keep their ``DIMENSIONS``. Expanding the header into
  a point array used to throw the grid away.
- A ``.vtu`` or ``.vtp`` ``Piece`` that declares points and does not
  deliver them raises ``CodecError``, whether its ``<Points>`` element is
  short or missing altogether. It used to be skipped, leaving the piece's
  cells indexing points that are not there and every later piece shifted
  by the count that never arrived.
- A point or cell array that covers only some of a multi-piece ``.vtu`` or
  ``.vtp`` file is dropped with a warning naming it. Joining the pieces
  that carried it gave an array shorter than the mesh, whose rows then sat
  against the wrong points from the second piece on. An array the pieces
  shape differently - one calling it scalar and the next a vector - is
  dropped the same way, with its shapes named, rather than raising a bare
  ``ValueError`` from ``numpy.concatenate``.
- A ``CELL_DATA`` section is read from ``STRUCTURED_POINTS``,
  ``STRUCTURED_GRID`` and ``RECTILINEAR_GRID`` files. Those three walk
  their own attributes, and the chain that did it asked only about points,
  so every cell array fell past it in silence - the cell scalars in VTK's
  own ``SampleStructGrid.vtk`` among them. The three now share one scanner,
  which also gives them ``TEXTURE_COORDINATES`` and ``TENSORS``. An array
  whose declared length matches neither the points nor the cells is dropped
  with a warning rather than reaching ``PolyData`` as a validation error
  about lengths.
- A structured legacy ``.vtk`` grid extends along whichever axes it
  declares. A ``DIMENSIONS 3 1 3`` sheet was read as two lines over the
  first two points instead of four quads over all nine, and a column along
  ``y`` or ``z`` was read with the stride of a row along ``x``. Only grids
  flat in ``z`` and rows along ``x`` came out right.
- An attribute keyword one of the structured readers does not handle now
  warns that its array is being dropped, the way the shared parser's binary
  scan already did. Skipping the line does not step over the payload, so in
  a binary file the scan carries on inside it.
- An unknown attribute keyword in an ASCII legacy ``.vtk`` file warns as
  well. Only the binary scan said anything; ASCII skipped the keyword and
  its value lines without a word.
- A legacy ``.vtk`` attribute section whose values run into the next
  header raises ``CodecError`` naming the array, the line and what it
  holds, rather than ``float()`` answering with a bare ``ValueError`` that
  names neither the array nor the file.
- A binary attribute in a structured legacy ``.vtk`` file is checked
  against the end of the file before it is sliced, which the shared parser
  already did. A truncated file gave a nameless reshape ``ValueError``.
- A ``.vtu`` or ``.vtp`` ``Points`` array whose size is not a whole number
  of tuples per point raises ``CodecError`` naming the Piece. Only a short
  array was caught, so ten values for three points reached ``reshape`` and
  came back as a ``ValueError`` naming neither the file nor the Piece.
- A ``v``, ``vn`` or ``vt`` record in an OBJ file that does not carry the
  components its directive needs, or carries something that is not a
  number, raises ``CodecError`` naming the line. Both used to reach the
  caller as a bare ``IndexError`` or ``ValueError``. A ``vt`` carrying a
  third component - the depth of a volumetric texture - keeps the two a
  surface uses rather than making the records ragged.
- Writing an OBJ ``vertex_attrs`` entry that holds no numbers - a label per
  vertex, say - drops it with a warning instead of raising out of
  ``numpy.asarray``.
- Legacy ``.vtk`` files written by VTK 5.1 - the default since VTK 9.0 -
  are read. The two numbers on a v5.1 ``CELLS`` line are the length of the
  ``OFFSETS`` array and the length of ``CONNECTIVITY``, not the cell count;
  reading the first as a cell count ran the offsets into the
  ``CONNECTIVITY`` keyword and answered with a bare ``ValueError``. The
  offsets are now counted up to that keyword, so files spelling the line
  either way are read. ``POLYDATA`` gained the layout altogether: its
  ``POLYGONS``, ``LINES``, ``VERTICES`` and ``TRIANGLE_STRIPS`` sections
  knew only the v4.2 form, so no polydata file VTK 9 writes could be read
  at all.
- Writing a v5.1 legacy ``.vtk`` file declares the length of its
  ``OFFSETS`` array on the ``CELLS`` line. It declared the cell count,
  which VTK's own reader takes literally: it stopped with "Error reading
  cell array connectivity header" and returned a mesh with no cells.
  Files polyxios wrote before this still read.
- A ``METADATA`` block is stepped over. Every VTK writer since 4.2 puts one
  after each array, and it is text even in a binary file. In ASCII it
  warned twice about keywords it named as dropped; in binary it ended the
  attribute scan, so a file with two arrays came back with one. Inside a
  ``FIELD`` block it was read as an array header, which took the array
  after it with it.
- A binary ``STRUCTURED_GRID`` keeps the section that follows its points.
  The line cursor was stepped past the payload and then once more over the
  newline that ended it, so whichever section came next - a ``CELL_DATA``
  written before ``POINT_DATA``, as VTK writes it - was skipped.
- A ``RECTILINEAR_GRID`` follows its coordinate arrays rather than its
  ``DIMENSIONS`` header. The points are the outer product of the three
  arrays, so a header that disagreed with them generated cells indexing
  points that do not exist and dropped attributes that covered every point
  there is. The header is now checked against them, warned about when it
  differs, and the arrays win.
- ``.vti``, ``.vts`` and ``.vtr`` honour ``NumberOfComponents`` when
  reading an attribute. A three-component array on 27 points came back as
  81 rows, which belongs to no mesh and fails ``validate``; ``.vtu`` and
  ``.vtp`` of the same family already cut it into tuples.
- ``.vtr`` declares the component count of the attributes it writes, so a
  vector survives a round trip, and writes each array in the type its
  ``<DataArray>`` declares. Everything was cast to float64 under whatever
  header the original dtype produced, so an ``Int32`` attribute read back
  as the bit pattern of a double.
- A ``.vtu`` or ``.vtp`` ``Points`` array of a type that holds no numbers -
  ``type="String"``, or any type this reader does not know - raises
  ``CodecError`` naming the type. It warned that the array was skipped and
  then blamed the point count for the zero values that left.
- A ``.vtr`` attribute of a dtype no VTK type names - a boolean mask, a
  float16 - is written as the ``Float64`` its header declares. Only the
  header fell back; the bytes stayed the dtype's own, so eight booleans
  went out as eight bytes under a ``Float64`` header and read back as one
  garbage double.
- Every XML writer declares the whole width of a tuple, not the second
  dimension of the array holding it. A ``(n, 3, 3)`` tensor attribute -
  what a legacy ``TENSORS`` section reads back as - was declared as one
  component in ``.vti``, ``.vts``, ``.vtp`` and ``.vtu`` and as three in
  ``.vtr``, so nine times the rows came back and belonged to no mesh.
- The XML writers cast to little endian before taking the bytes, which is
  what the ``byte_order`` they all declare says those bytes are. On a
  big-endian machine every binary array went out reversed.
- ``.vti``, ``.vts`` and ``.vtr`` drop an attribute whose rows cannot be
  matched to the mesh, with a warning naming it, as ``.vtu`` and ``.vtp``
  already did. It used to reach ``PolyData`` and fail ``validate`` with a
  message about lengths rather than about the file.
- A binary legacy ``.vtk`` file holds its ``TENSORS`` as binary. Both
  tensor branches of the writer - the 3x3 one and the Voigt 6-component
  one - spelled their numbers whatever was asked for, so a binary file
  carrying either had a run of ASCII in the middle of it and came back as
  a ``CodecError`` about a short block.
- The ASCII writers spell a value as the shortest decimal that reads back
  as it, rather than to ten significant digits. A legacy ``.vtk``,
  ``.vti``, ``.vts``, ``.vtp`` or ``.vtr`` file declaring ``double`` or
  ``Float64`` held seven digits fewer than that, so coordinates came back
  about 1e-10 off what was written; they are now exact.
- A ``METADATA`` block written without its blank terminator ends at the
  geometry keyword after it as well as at an attribute one. Left open at
  the end of a ``POINT_DATA`` section it swallowed the ``CELLS`` that
  followed and every line to the end of the file - the failure the
  terminator search was added to prevent.
- A malformed attribute section in a structured legacy ``.vtk`` file costs
  the section rather than the mesh. A count running past the end of the
  file raised ``CodecError`` out of a read whose geometry was already
  whole, and those sections were skipped entirely before they were read at
  all, so files that used to load stopped loading.
- A ``.vtu`` or ``.vtp`` ``Piece`` that cannot be read is named by its
  index. A file of forty pieces gave no way to find the one at fault.
- An OBJ vertex named twice with different values is warned about whichever
  component they differ in. The check asked whether the first component had
  been written yet, so a record whose first component is NaN hid every
  conflict on that vertex.
- Writing an OBJ ``vn`` or ``vt`` attribute wider than the record says how
  many components are being left out, rather than truncating in silence.
- A legacy ``STRUCTURED_GRID`` whose ``DIMENSIONS`` does not cover its
  ``POINTS`` array hands the points back without cells, warning about the
  two counts. The cells are strides through the layout the header
  describes, so a header naming more points than the file delivered
  generated connectivity indexing points that are not there: the read
  returned a ``PolyData`` that fails ``validate``. ``RECTILINEAR_GRID``
  already reconciled the two.
- An attribute section is read by the count its own header declares rather
  than by the count of the mesh. The two agree in a well-formed file, and
  where they do not the section's is the only number that says where one
  array ends and the next begins - reading by the mesh's walked an array
  straight into the header after it. An array that then covers no point or
  cell of the mesh is dropped with a warning naming it, as the structured
  readers already did.
- A binary legacy ``.vtk`` ``SCALARS`` section without a ``LOOKUP_TABLE``
  line - which the format leaves optional - reads its own values. The line
  was consumed unconditionally, so the payload up to its first ``0x0a``
  byte was swallowed, and a payload holding none rewound the scan to the
  top of the file.
- A binary legacy ``.vtk`` ``POINTS``, ``CELLS`` or ``CELL_TYPES`` block is
  checked against the end of the file before it is sliced, the way the
  attribute blocks already were. The whole-file bound the header check
  applies clears a block that still runs off the end - a file with a long
  comment header, say - and the reshape after the short slice failed with a
  ``ValueError`` naming neither the array nor the file.
- A legacy ``.vtk`` attribute header missing a field, or spelling a count
  as something that is not a number, names the file and the line it is on.
  ``SCALARS`` with no array name reached the caller as ``IndexError: list
  index out of range`` and ``SCALARS s float x`` as a bare ``ValueError``,
  in the ASCII scan, the binary scan and the structured one alike. A binary
  file has no line to name, so the byte offset stands in for one.
- ``.vti``, ``.vts``, ``.vtp`` and ``.vtu`` write each attribute in the
  type the array is held in, as ``.vtr`` does. Everything was cast to a
  double under a ``Float64`` header, so an ``int64`` identifier past 2**53
  came back a different number.
- The ASCII body of a ``<DataArray>`` is parsed into the type the element
  declares rather than through ``float()`` first. An ``Int64`` array was
  rounded to a double before the declared type ever saw it, which the top
  of the integer range does not survive.
- A binary legacy ``.vtk`` block is read as the type its header names. A
  ``POINTS n int`` was read at four bytes a float, so an integer point
  array came back as coordinates the file never held, and a type name the
  reader has no numpy equivalent for - ``bit``, or a misspelling - was
  guessed at rather than refused. Every binary header now resolves its type
  the same way and raises ``CodecError`` naming it when it cannot; the
  names VTK writes for 64-bit and signed-char arrays were missing from the
  table and are there now. An ASCII payload is unaffected: its values are
  text whatever the header calls them.
- A legacy ``.vtk`` geometry or section header spelling a count as
  something that is not a number, or leaving it out, names the file and the
  line it is on. ``POINTS``, ``CELLS``, ``CELL_TYPES``, ``POINT_DATA``,
  ``CELL_DATA``, ``DIMENSIONS``, ``ORIGIN``, ``SPACING`` and the coordinate
  arrays reached the caller as a bare ``ValueError`` or ``IndexError``,
  which the attribute headers had already stopped doing.
- A ``CELL_DATA`` array in a legacy ``STRUCTURED_GRID`` is measured against
  the cells the mesh ends up with rather than the cells ``DIMENSIONS``
  describes. A header its ``POINTS`` array does not cover leaves the mesh
  with no cells, and the array was kept against the header's count, so the
  read returned a ``PolyData`` that fails ``validate``.
- An OBJ face index spelled with a superscript digit raises ``CodecError``
  naming the line. ``str.isdigit`` admits ``²`` and ``int()`` then refuses
  it, so the fast path let a bare ``ValueError`` out; the test is now
  ``str.isdecimal``, which still admits the non-Latin digits ``int()``
  reads.
- A bare ``mtllib`` or ``o`` directive in an OBJ file names nothing rather
  than the empty string, which used to be written back as a directive with
  nothing after it.
- Writing an OBJ ``element_attrs['material']`` that does not cover the
  faces drops it with a warning naming its length, the way the vertex
  attributes already were. Indexed per face it ran off the end partway
  through, leaving a half-written file and an ``IndexError`` naming an
  axis.
- A ``.vtu`` or ``.vtp`` ``Piece`` whose ``NumberOfPoints`` is not a count,
  and a ``.vti``, ``.vts`` or ``.vtr`` ``Extent`` that is not six whole
  numbers, raise ``CodecError`` naming the file. They reached the caller as
  a bare ``ValueError`` about ``int()`` or about unpacking.
- A ``<DataArray>`` whose ``NumberOfComponents`` is not a count is read
  flat with a warning naming it, rather than raising a bare ``ValueError``
  from ``int()``.
- A legacy ``.vtk`` file whose header declares ``DATASET FIELD`` is read.
  The dispatch asked what the line starts with while the line still
  carried its ``DATASET`` keyword, so the branch never ran and every field
  data file was refused by the one below it.
- A ``METADATA`` block inside a v5.1 ``CELLS`` section is stepped over. VTK
  follows a cell array with one, so a block sat between the offsets and the
  connectivity of files every release since 9.0 writes; read as offsets it
  raised ``CodecError`` about a line of words where numbers belong.
- A v5.1 ``CELLS`` section is found by what follows the header rather than
  by the version in the first line. Versions were compared as strings, so a
  file declaring ``10.0`` sorted below ``5.1`` and its offsets would have
  been read as a v4.2 cell stream. The binary scan already asked this way.
- A ``.vti``, ``.vts`` or ``.vtr`` extent flat along an axis - an image one
  voxel deep - is a sheet of quads, or a run of lines when it is flat along
  two. All three read it as a grid of no cells, which left every
  ``CellData`` array belonging to nothing.
- An ASCII ``<DataArray>`` holding a value its declared type is too narrow
  for wraps with a warning naming the array, the way a C reader wraps it,
  rather than escaping as a bare ``OverflowError`` from numpy. A token that
  names no number at all raises ``CodecError`` naming the array.
- Writing a point or cell attribute of a kind no ``<DataArray>`` can hold -
  a label per vertex, say - raises ``CodecError`` naming the array and its
  dtype, rather than a ``ValueError`` about one element.
- The warnings the ``.vtk``, XML and OBJ codecs raise are blamed on the code
  that asked for the file. Every one of them pointed a frame short, at
  ``polyxios.read`` or ``polyxios.write`` itself, which tells a caller
  nothing about which of their own calls found the file.

Optimizations
~~~~~~~~~~~~~

- Reading an OBJ file resolves a face corner inline when it names a plain
  index inside what has been declared, which is what nearly every corner
  does; the rest still go the long way round, where the message naming the
  line lives. A mesh of any size has millions of corners, and the call this
  saves is most of what checking them cost. A ``v``, ``vn`` or ``vt`` record
  carrying exactly the components its directive spells skips the padding and
  the slice that feeds it.
- A ``float32`` attribute is spelled at its own width in an ASCII
  ``<DataArray>``. Widened to a double first, ``0.1`` went out as
  ``0.10000000149011612`` - seventeen digits of a value carrying seven -
  which is nearly twice the file for the same numbers.
- Writing a Nastran large-field deck is roughly ten times faster. Sweeping
  for an exact spelling asked every precision from seventeen digits down,
  spelling and parsing candidates at each; rounding to fewer significant
  digits than ``repr`` carries cannot be exact whatever form it takes, so
  the sweep now stops there. Neither sweep spells a candidate at a
  precision the field could not hold in the first place, and the stepped
  mantissas - the expensive half - are built only when the rounded ones
  have all missed. Twenty thousand ``GRID*`` cards went from 10.3 s to
  1.0 s, and every value is written exactly as before.
- Reading an OBJ file no longer names its source once per line. The name
  costs a path walk and was only ever used to spell an error message.
- A Nastran real field below one drops its leading zero when the column it
  costs is a significant digit. Bulk data reads ``.5`` as ``0.5``, so a
  third in an eight-column field goes out as ``.3333333`` rather than
  ``0.333333``.
- Writing an OBJ file looks its material attribute up once rather than once
  per face.
- Stepping past a binary payload in a structured legacy ``.vtk`` file is a
  binary search over the line offsets rather than a walk. A binary payload
  carries a newline every few values, so the lines it is cut into number in
  the thousands for a grid of any size; a 5 MB ``STRUCTURED_POINTS`` file
  reads about a fifth faster.
- ``.vti``, ``.vts`` and ``.vtr`` build their hexahedra with array
  arithmetic instead of a Python loop per cell. The corners of a cell are
  strides from its own origin, so the whole connectivity is eight adds over
  the origins; a 40x40x40 grid builds about nine times faster.
- The structured legacy ``.vtk`` readers cut their file into lines in one
  pass rather than a ``find`` per line, and they all do it in the same
  place. The lines are byte-identical to what the walk produced.
- Folding OBJ ``vt`` and ``vn`` records onto their vertices is one pass over
  the corners rather than a numpy row assignment per corner.
- A legacy ``.vtk`` ASCII payload is spelled and written in one pass instead
  of a formatted write per point, which was a syscall per line.

Tests
~~~~~

- Regression tests for the dtype a ``.vtr`` header declares against the
  bytes under it, the component count of a tensor attribute in every XML
  writer, a binary legacy ``TENSORS`` section, an unterminated ``METADATA``
  block followed by geometry, an attribute section declaring more than its
  file holds, an OBJ conflict hiding behind a NaN first component, and the
  ``Piece`` index in a ``.vtu`` error.
- ``test_a_double_section_holds_every_digit_of_a_double`` writes a
  coordinate no ten-digit spelling can hold and asks for it back unchanged.

- ``test_a_power_of_ten_is_written_exactly`` spells its powers as literals
  rather than computing them with ``**``. ``pow`` is not correctly rounded
  everywhere - glibc and MSVC answer ``10.0 ** 23`` with the double one unit
  above ``1e23``, macOS with ``1e23`` itself - and that neighbour needs
  seventeen significant digits, which no eight or sixteen character field
  can hold. The test asked the writer for a field wider than the format has,
  and failed on Linux and Windows only.
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
