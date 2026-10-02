.. _format-collada:

COLLADA
=======

.. rst-class:: px-badges

``.dae`` ``read + write`` ``eager`` ``scene: read_scene + write_scene``

Summary of the specification
-----------------------------

COLLADA is the Khronos Group's XML interchange format for 3D scenes, the file every modelling tool of the 2000s learned to export and the one glTF was designed to replace. A document is a set of libraries tied together by ``#id`` references: geometries, effects and the materials that instance them, images, controllers (skins and morphs), animations and visual scenes, with a ``<scene>`` naming the visual scene to show. A ``<geometry>`` holds a ``<mesh>`` of ``<source>`` arrays, each read through an ``<accessor>`` that names its stride and parameters, a ``<vertices>`` element binding the position source, and primitive blocks - ``<triangles>``, ``<polylist>``, ``<polygons>``, ``<lines>``, ``<linestrips>``, ``<tristrips>``, ``<trifans>`` - whose ``<p>`` index tuples name one entry of every ``<input>`` per corner, so a corner carries its own normal and texture coordinate. Nodes hold transforms as ``<matrix>``, ``<translate>``, ``<rotate>``, ``<scale>`` and ``<lookat>`` elements applied in document order, each addressable by ``sid`` so an ``<animation>`` channel can target it. Effects describe materials in fixed-function terms - phong, lambert, blinn or constant shading with emission, diffuse, specular and transparency channels that are colours or textures reached through a sampler and a surface.

Specification at a glance
--------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - document
     - UTF-8 XML, root ``<COLLADA version="1.4.1">`` (or 1.5.0) in the ``collada.org/2005/11/COLLADASchema`` namespace
   * - source
     - ``<float_array count>`` (or ``Name_array``) plus ``<accessor count stride offset>`` with one ``<param name type>`` per used component
   * - mesh
     - ``<vertices>`` binding ``POSITION``; primitive blocks with ``<input semantic offset set>`` and ``<p>`` index tuples, ``<vcount>`` for a polylist
   * - primitive blocks
     - ``triangles``, ``polylist``, ``polygons`` (with ``<ph>`` holes), ``lines``, ``linestrips``, ``tristrips``, ``trifans``
   * - node transforms
     - ``<matrix>`` (16, row-major), ``<translate>`` (3), ``<rotate>`` (axis + degrees), ``<scale>`` (3), ``<lookat>`` (9), ``<skew>``; composed in document order
   * - instancing
     - ``<instance_geometry>`` with ``<bind_material>``, ``<instance_controller>`` with ``<skeleton>``, ``<instance_node>``, cameras and lights
   * - effect
     - ``<profile_COMMON>`` with ``<newparam>`` surfaces and samplers, a ``<technique>`` of ``phong`` / ``lambert`` / ``blinn`` / ``constant``
   * - skin
     - ``<bind_shape_matrix>``, ``<joints>`` (``JOINT`` names, ``INV_BIND_MATRIX``), ``<vertex_weights>`` with ``<vcount>`` and ``<v>`` joint/weight pairs
   * - animation
     - ``<sampler>`` of ``INPUT`` times, ``OUTPUT`` values, ``INTERPOLATION`` names and tangents; ``<channel target="node/sid.MEMBER">``
   * - asset
     - ``<up_axis>`` X_UP / Y_UP / Z_UP, ``<unit name meter>``, contributor, created and modified

.. rst-class:: px-speclink

`Read the full COLLADA 1.4.1 specification ↗ <https://www.khronos.org/files/collada_spec_1_4.pdf>`__

Reading
-------

``read_scene`` returns a :class:`~polyxios.SceneData` with one mesh per ``<geometry>``, in library order:

.. code-block:: python

    import polyxios as px

    scene = px.read_scene("robot.dae")
    scene.meshes       # tuple of PolyData, one per <geometry>
    scene.nodes        # tuple of SceneNode: local matrix, mesh index, children
    scene.scenes       # root node indices of each <visual_scene>
    scene.materials    # tuple of SceneMaterial built from the effects
    scene.textures     # sampler settings, one per distinct (image, wrap, filter)
    scene.images       # <library_images>: a uri, or bytes from <hex> / data: URIs

A corner is one index per ``<input>``; corners that name the same position but a different normal, texture coordinate or colour become separate vertices, in first-occurrence order, the way an OBJ reader splits ``v/vt/vn`` triples. Every position of ``<vertices>`` is a vertex, whether or not a corner names it. A block whose inputs all share one offset is read as values per position; a later block naming another source for the same attribute splits its corners like any other disagreeing input. When one block carries an input another block of the same mesh lacks (lines beside textured triangles, say), the corners without it get zeros, with a warning. ``NORMAL`` set 0 lands in ``normals`` and set *n* in ``normals_n``, ``TEXCOORD`` set 0 in ``texcoords`` and set *n* in ``texcoords_n`` (a mesh with no set 0, as 3ds Max numbers its first map channel 1, has its sets renumbered down from the lowest, so its first is ``texcoords``; a ``P`` column that is zero throughout is dropped), ``COLOR`` in ``colors`` (3 or 4 columns as the file has them); any other semantic keeps its lowercased name, and a set spelled with leading zeros is read as its number; a block giving one attribute twice (a ``TEXCOORD`` with no set beside one of set 0, say) keeps the first with a warning. Every primitive block maps to the VTK element it is: triangles, quads and polygons (a polylist entry by its size), lines, ``poly_line`` for a line strip, ``triangle_strip`` for a triangle strip, and a fan is expanded into triangles. A block's ``material`` symbol is resolved through the instancing node's ``<bind_material>``, or straight to a material of that id (with a warning when the instance binds other symbols but not this one), into ``element_attrs["material"]`` (``-1`` where none). A geometry whose instances resolve its symbols to different materials - two nodes binding one symbol differently, or one binding it and one leaving it to the material of that id - gets a mesh per resolution: the first keeps the geometry's place in ``meshes``, each other is appended after the last geometry, sharing the first's arrays (PolyData is immutable) under its own ``material`` column, and those columns may hold sixteen bytes per byte of document (and never less than 256 MiB) before it is refused. A ``<geometry name>`` is the mesh's ``global_attrs["mesh_name"]``, the key glTF and MED use, and every ``<library_*>`` of a kind is read, not only the first.

Nodes keep their ``id``, ``sid`` and ``JOINT`` type in ``extras``, along with the transform elements they were spelled with (``extras["transforms"]``, a list of ``{"kind", "sid", "values"}``), so a node writes back spelled as it was read. A node instancing several geometries gets a child node per extra one; an ``<instance_node>`` is copied into place, and a document whose instances would expand the tree past one node per eight bytes (and past 262144 nodes) is refused rather than grown exponentially. Skins are not read yet: an ``<instance_controller>`` reads as its geometry in the bind shape, without ``joints`` or ``weights``, with one warning counting the skinned instances. Animations land under ``global_attrs["animations"]`` in the same shape glTF uses: each named by its ``name``, else its ``id`` (so an unnamed animation a write gave a generated id reads back under that id), with ``channels`` (``sampler`` index and a ``target`` of node index, ``path``, the transform's ``sid`` and an optional ``member`` such as ``ANGLE``) and ``samplers`` (``times``, ``values``, the COLLADA ``interpolation`` name - upper-cased, and ``LINEAR`` with a warning when the specification does not name it - and Bezier ``in_tangents`` / ``out_tangents`` when present). A channel target is an ``id/sid`` address: each sid of the address is looked up breadth-first below the node before it (the transform's on that node first, then on the nodes below it), a channel whose address reaches no transform is dropped with a warning, and one addressing a node that ``<instance_node>`` copied gets a target per copy, capped like the copies themselves. The shape is glTF's but the meaning is COLLADA's - a ``rotation`` channel animates one ``<rotate>`` element, often only its ``ANGLE`` in degrees - so a glTF write skips these channels with a warning rather than writing them as quaternions. The exception is a ``matrix`` channel of ``LINEAR`` or ``STEP`` keys on a node spelled with that one ``<matrix>`` alone, which is the node's whole local transform: a glTF write splits each key into translation, rotation and scale channels, so a glTF animation baked on a COLLADA write (see below) comes back as one, holding its values at its key times, with the extra keys the bake added. The ``<asset>`` up axis, unit, timestamps and its contributor's author, authoring tool and copyright are ``global_attrs["asset"]``; the axis is recorded, never applied. A write keeps the author and copyright (a glTF asset's copyright included), names polyxios as the authoring tool, and drops with a warning any other key but another format's own ``version`` and ``generator``.

Flatten to a single :class:`~polyxios.PolyData`, or let :func:`~polyxios.read` do it with a warning:

.. code-block:: python

    mesh = scene.to_polydata()    # applies every node's world transform
    mesh = px.read("robot.dae")   # the same, with a UserWarning

Writing
-------

``write_scene`` spells a :class:`~polyxios.SceneData` back in COLLADA 1.4.1 syntax:

.. code-block:: python

    px.write_scene(scene, "out.dae")
    px.write(mesh, "out.dae")     # a one-node scene; material values become material_<n>

Each mesh becomes a ``<geometry>``, named by its ``global_attrs["mesh_name"]``, with its positions, ``normals``, ``texcoords``, ``colors``, ``tangent``, ``binormal``, ``textangent`` and ``texbinormal`` (each with its ``_n`` set suffix, so every attribute the reader names comes back) as sources sharing one index (integer colours scaled to 0..1 by their dtype's maximum, 255 for a signed dtype, with a warning when a signed one holds values outside 0..255); element attributes other than ``material``, tag groups, ``joints`` and ``weights``, and ``global_attrs`` other than ``mesh_name`` (or, on the scene, ``asset``) are dropped with a warning, the scene's ``skins`` included; a set suffix the reader would not spell back - not a number, ``_0``, a leading zero, or one another key of the mesh already spells - is written under the lowest free set number with a warning and reads back under that number. Triangles go in a ``<triangles>`` block, or into the ``<polylist>`` with the quads and polygons when the mesh has any; lines, line strips and triangle strips get their own blocks, and points and volume cells are skipped with a warning - a point cloud keeps its positions as a ``<mesh>`` without primitives, which is valid COLLADA and reads back as a mesh of no elements, its set-0 vertex attributes written inside ``<vertices>`` (whose inputs take no set, so another set is dropped with a warning). Elements are grouped by primitive block and material, each group its own block, in first-occurrence order: element order survives within one block, so a surface whose ``material`` column alternates reads back regrouped by material (``[0, 1, 0, 1]`` becomes ``[0, 0, 1, 1]``), and lines and triangles given as ``[line, triangle, line]`` read back as ``[line, line, triangle]``. A ``<polylist>`` keeps only each element's vertex count, so a polygon of three or four vertices reads back as a triangle or a quad. Connectivity is checked before anything is written: offsets that do not rise from 0 to the connectivity's length, an index naming a vertex the mesh lacks, or an element whose vertex count its type does not allow (a line two, a triangle three, a quad four, a polygon or triangle strip at least three, a line strip at least two) is refused rather than written into a file no reader could read back. Node and material ``extras`` keys COLLADA has no place for are dropped with a warning. A ``material`` column must hold whole numbers (an integer dtype, or floats that are all whole and finite); anything else is refused, by ``write`` as by ``write_scene``. Materials become ``<effect>`` entries in the shading their ``extras["shading"]`` names (phong by default) with the base colour as diffuse, ``emissive`` as emission, alpha below one as the alpha of an ``A_ONE`` ``<transparent>`` colour, a base colour texture as the diffuse texture, bound to the lowest ``TEXCOORD`` set the mesh writes by a ``<bind_vertex_input>`` (none when it writes no texture coordinates), a normal texture as an FCOLLADA ``<bump>``, and ``double_sided`` as the GOOGLEEARTH extra; a metallic-roughness or occlusion texture has no slot in the common profile and is dropped with a warning. Images are written by uri, or as a ``data:`` URI when they hold bytes. A node is spelled with the transform elements in ``extras["transforms"]`` when they still compose to its matrix (to a tolerance relative to the matrix's own magnitude), else as one ``<matrix sid="transform">``. A node reached more than once - by a second parent, or as the root of a second scene - is written once in ``<library_nodes>``, the only place some readers resolve an ``<instance_node>``, and every parent instances it from there: listed before that parent's child nodes, as the schema orders them, so it reads back first among its siblings, and a scene root reached so is held by a wrapper node, since a visual scene holds only ``<node>`` elements. A node read as a copy of an ``<instance_node>`` - holding the ``id`` of a node written before it and matching its subtree node for node - is instanced from that node again, so a document's instancing survives a round trip, while a copy edited after reading is written in full under a generated id, with a warning. An animation channel targets the transform element whose ``sid`` the written node has, and is dropped with a warning otherwise or when its ``member`` is neither a name nor ``(i)(j)`` indices. A channel whose sampler has no keys, keys or tangents that are not finite, or key times that do not increase is dropped with a warning, as is a second channel on a target the animation already animates. A channel without a ``sid`` (one read from another format) on a node written as one ``<matrix>`` - glTF's ``translation``, ``rotation`` and ``scale`` - is baked: an animation's channels of one node become one channel of matrix keys targeting ``node/transform``, at every channel's key times, with the node's rest translation, rotation or scale where no channel animates it (a matrix holding a shear or projection loses it, with a warning); rotations are slerped and ``CUBICSPLINE`` follows glTF's Hermite spline, and since COLLADA blends a matrix element by element, which shrinks a turning node between keys, extra keys split a span turning more than 15 degrees (a rotation that steps turns at its keys alone and is not split) and each ``CUBICSPLINE`` span in four, and a ``STEP`` channel among blending ones gets a key just before each of its steps, one float32 step before so a reader parsing times as float32 keeps them apart; when every channel is ``STEP`` the sampler is too. A channel to bake whose keys are not its path's width, not finite or not increasing, or that animates a path of the node a second time in one animation, is dropped with a warning. On a node spelled with its own transform elements, a channel without a ``sid`` binds by its path only when the node spells exactly one element of that kind, each key holds that element's whole value and its sampler interpolates in a way COLLADA names (glTF's ``CUBICSPLINE`` does not); a ``rotation`` path never binds that way, since without a ``sid`` it is a quaternion and a ``<rotate>`` an axis and an angle. A channel naming a node no scene reaches is dropped with a warning, and one on a node read as an ``<instance_node>`` copy is written once for the node, with a warning when only a copy holds it, since it then reaches every instance. Only nodes a ``<visual_scene>`` reaches are written: a node in no scene is dropped with a warning. A node ``id`` or ``sid`` that is not an XML name, or an ``id`` an earlier node already holds, is dropped with a warning, the node taking a generated id.

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``up_axis``
     - ``None``
     - ``"X_UP"``, ``"Y_UP"`` or ``"Z_UP"``; overrides ``global_attrs["asset"]["up_axis"]``, which itself defaults to ``Y_UP``.
   * - ``unit``
     - ``None``
     - ``{"name": ..., "meter": ...}``; overrides the asset's unit, which defaults to one metre.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A document declaring an XML entity is refused, whatever the expat Python links against would do with it; one in an encoding expat cannot read itself (Shift_JIS, EUC-JP, ...) is transcoded first, and an encoding Python does not know is refused.
- Every count is checked before anything is allocated: a ``float_array`` whose ``count`` disagrees with its content, an accessor reading past its array, a ``<p>`` whose length disagrees with the block's ``count`` or ``<vcount>``, an index past its source, a ``#url`` the document does not hold, a transform with the wrong number of values and a node instancing itself are each refused by name.
- The writer uses fixed timestamps (``1970-01-01T00:00:00Z`` unless the asset carries its own ``xs:dateTime`` - ``2024-01-02T03:04:05Z``, or a ``datetime``; any other value is replaced by the epoch with a warning), so two writes of one scene are byte-identical.
- The document is UTF-8 and says so: a text handle opened with another encoding is refused, unless that encoding spells ASCII as ASCII (Latin-1, cp1252, ...) and the document holds nothing else. Open it with ``encoding="utf-8"``, in binary mode, or pass a path.
- A name holding a line break or tab is written as a character reference, since a parser folds the literal character to a space, so it reads back unchanged; a node ``id`` that is not an XML name is replaced by a generated one, a node ``sid`` is written only when the node has a valid one, a transform ``sid`` that a channel target cannot end in (a letter or ``_``, then letters, digits, ``_`` or ``-``: a ``.`` would start the member) or that repeats an earlier one of the node is written without one, with a warning.
- A diffuse texture replaces the diffuse colour, as the format has room for one or the other: a textured material reads back with a white base colour and its alpha, and a tint on the texture is dropped with a warning (an emission texture likewise reads back with a white ``emissive``, the factor that leaves it as drawn, and a tint on it is dropped with a warning; under a black ``emissive``, which switches it off, the texture itself is dropped with a warning rather than written to glow), as is the base colour of a ``constant`` effect, which has no diffuse. Metallic and roughness have no COLLADA spelling and are not written; every material reads back with ``metallic=0``.
- Transparency follows the ``<transparent opaque="...">`` mode (``A_ONE``, ``A_ZERO``, ``RGB_ZERO``, ``RGB_ONE``; any other spelling is read as ``A_ONE`` with a warning) times ``<transparency>``; an alpha below one reads as ``alpha_mode="BLEND"``. The one exception is ``A_ONE`` with a white transparent colour and ``<transparency>`` 0, which SketchUp, Google Earth and Blender before 2.8 write for an *opaque* material: read literally it would be invisible, so it is read as opaque with a warning (once per document, as a SketchUp export has hundreds), as other readers do. COLLADA has no alpha mask: ``alpha_mode`` is derived from the alpha on read, and ``MASK`` with its ``alpha_cutoff`` is not written.
- Non-finite values are written in XML's own spelling (``NaN``, ``INF``, ``-INF``) and read back as such.
- An image ``uri`` is kept as the file spells it, percent-encoding included (``tex%20ture.png``), as glTF's is: decode it before opening a file.
- A 1.5 sampler naming its image with ``<instance_image>`` resolves like a 1.4 sampler-and-surface chain. A 1.4 ``<image>`` holding its pixels in ``<data>`` reads like a 1.5 ``<hex>``, its media type sniffed from the bytes; a ``data:`` URI's ``base64`` marker is matched in any case, and one without it is percent-decoded; a 1.5 ``<create_2d>``, ``<create_3d>`` or ``<create_cube>`` image is read empty with a warning.
- Polygons with holes (``<ph>``) keep their outer ring and drop the holes with a warning; ``<skew>`` transforms are ignored with a warning; a morph controller reads as its base geometry; a ``<transparency>`` or diffuse alpha that is not finite is read as opaque with a warning; an ``<input>`` without a ``semantic`` names no attribute and is dropped with a warning; an ``<up_axis>`` outside ``X_UP`` / ``Y_UP`` / ``Z_UP`` is dropped with a warning.
- An ``<instance_camera>`` or ``<instance_light>`` is read as the id it names, under the node's ``extras["camera"]`` / ``extras["light"]``; a node's second one of a kind is dropped with a warning, and the writer carries neither and warns when it drops one.
- A file whose numbers use a decimal comma is refused as "not numbers", except the asset's ``meter``, which is read with a warning since one exporter spells only that one in the machine's locale; a ``meter`` that is zero, negative or not finite is read as 1 with a warning, since the writer refuses it.
- ``.zae`` (a zipped ``.dae``) and external ``file.dae#id`` references are not read: an ``<instance_node>``, ``<instance_geometry>``, ``<instance_controller>``, ``<instance_effect>`` (the material reads as a bare phong one) or ``<instance_visual_scene>`` (the first visual scene is active) naming another document is dropped with a warning, any other external reference is refused. An ``<image>`` held inside an effect is read after those of ``<library_images>``.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`: the geometry is XML text.

Known limitations
-----------------

.. rst-class:: px-quirks

- Skins are neither read nor written yet (see above); a skinned mesh reads in its bind shape.
- ``<library_animation_clips>`` is not read: every ``<animation>`` of the document is an animation of the scene, whatever clip groups them.
- A glTF animation survives a round trip through COLLADA as matrix keys: values and key times are kept, the interpolation becomes ``LINEAR`` (or ``STEP`` when every channel of the node steps) over the denser keys the bake adds, and translation, rotation and scale channels come back for every animated node, whichever of the three the original animated.
- Ids follow the XML 1.0 fifth edition's name characters. Validators built on XML Schema 1.0, libxml2 among them, still apply the older, narrower classes, so an id starting with a letter outside them (Glagolitic, a supplementary-plane script) is valid XML that such a validator rejects.
- COLLADA 1.4.1 types every ``name`` attribute as an ``xs:NCName``, so a name holding a space or starting with a digit makes a document a validating parser rejects. Most readers accept it, and it is written as given.
- ``write`` names materials ``material_<value>`` after the PolyData's ``material`` values, but a read numbers materials from 0 in document order, so ``[7]`` reads back as ``[0]`` with material ``material_7``.
- An attribute only a primitive block of ``count="0"`` names becomes a zero column on read, without the warning a partial attribute gets.
- External references (``file.dae#id``) and ``.zae`` archives are not followed.

.. seealso::

   :doc:`gltf` - the format's successor, sharing the SceneData model.
   :doc:`index` - the full format table.
