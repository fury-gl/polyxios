.. _format-gltf:

glTF
====

.. rst-class:: px-badges

``.gltf`` ``.glb`` ``read + write`` ``eager`` ``scene: read_scene + write_scene``

Summary of the specification
-----------------------------

glTF 2.0 is a JSON-based container for real-time 3D assets published by the Khronos Group. A ``.gltf`` file is a UTF-8 JSON document that references external binary buffers (``.bin``) and image files; a ``.glb`` file packs the JSON and an optional binary buffer into a single binary container whose chunks are each 4-byte aligned. The scene graph is a forest of nodes, each holding a local transform (either a 4×4 column-major matrix or explicit translation/rotation/scale), optional mesh and camera references, and a list of child indices. Meshes are composed of one or more primitives, each of which is a typed accessor view into the binary buffer together with an optional material index. Materials follow a PBR metallic-roughness model with base-colour texture, metallic-roughness texture, normal map, occlusion texture, and emissive factor. Animations describe time-sampled keyframe channels targeting node TRS properties or morph-target weights.

Specification at a glance
--------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - container forms
     - ``.gltf`` (JSON + external ``.bin``) or ``.glb`` (self-contained binary)
   * - GLB magic
     - ``glTF`` (bytes 0–3), version uint32 = 2, total-length uint32
   * - GLB chunks
     - JSON chunk (type ``0x4E4F534A``), then optional BIN chunk (type ``0x004E4942``); all 4-byte aligned
   * - scene graph
     - ``scenes`` array of root-node-index lists; ``scene`` selects the default
   * - node transform
     - 4×4 column-major ``matrix``, or ``translation`` / ``rotation`` (xyzw quaternion) / ``scale``
   * - bufferViews
     - byte range inside a buffer; ``target`` 34962 (ARRAY_BUFFER) or 34963 (ELEMENT_ARRAY_BUFFER) for mesh data; omitted for animation and image data
   * - accessors
     - typed view into a bufferView: ``componentType``, ``type`` (SCALAR/VEC2/VEC3/VEC4/MAT*), ``count``
   * - primitive modes
     - 0 POINTS, 1 LINES, 3 LINE_STRIP, 4 TRIANGLES, 5 TRIANGLE_STRIP
   * - PBR material
     - ``pbrMetallicRoughness``: ``baseColorFactor``, ``baseColorTexture``, ``metallicFactor``, ``roughnessFactor``, ``metallicRoughnessTexture``; plus ``normalTexture``, ``occlusionTexture``, ``emissiveFactor``
   * - data URI buffers
     - ``data:application/octet-stream;base64,<b64>`` inline in the JSON
   * - animation
     - sampler input (time) and output (values) accessors, LINEAR/STEP/CUBICSPLINE interpolation

.. rst-class:: px-speclink

`Read the full glTF 2.0 specification ↗ <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>`__

Reading
-------

``read_scene`` returns a :class:`~polyxios.SceneData` preserving the full scene graph:

.. code-block:: python

    import polyxios as px

    scene = px.read_scene("robot.glb")
    scene.nodes        # tuple of SceneNode (matrix, mesh index, children)
    scene.meshes       # tuple of PolyData (one per glTF mesh, primitives merged)
    scene.materials    # tuple of SceneMaterial (PBR metallic-roughness)
    scene.textures     # tuple of SceneTexture (image index + sampler settings)
    scene.images       # tuple of SceneImage (URI or embedded bytes)

Flatten to a single :class:`~polyxios.PolyData` applying all node transforms:

.. code-block:: python

    mesh = scene.to_polydata()  # applies world transforms, merges all meshes
    mesh.vertices               # (n, 3)
    mesh.element_types          # element groups from all nodes

:func:`~polyxios.read` flattens automatically but warns that scene data is discarded:

.. code-block:: python

    mesh = px.read("robot.glb")   # issues UserWarning; use read_scene to suppress

Writing
-------

``write_scene`` round-trips a :class:`~polyxios.SceneData`:

.. code-block:: python

    px.write_scene(scene, "out.glb")   # GLB (self-contained)
    px.write_scene(scene, "out.gltf")  # glTF JSON + out.bin (separate buffer)

``write`` serialises a flat :class:`~polyxios.PolyData` as a single-node scene:

.. code-block:: python

    px.write(mesh, "out.glb")
    px.write(mesh, "out.gltf")

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``binary``
     - ``None``
     - ``True`` → GLB output; ``False`` → ``.gltf`` + ``.bin``.  When ``None``, inferred from the path suffix (``.glb`` → True, ``.gltf`` → False).

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- GLB is detected by the ``glTF`` magic bytes, so a file renamed to ``.gltf`` but carrying a binary GLB header still reads correctly.
- Multiple glTF primitives inside one mesh are merged into a single :class:`~polyxios.PolyData` on read; ``element_attrs["material"]`` records per-element material indices.
- Node world transforms are applied during :meth:`~polyxios.SceneData.to_polydata`: vertices are multiplied by the world matrix, normals by the inverse-transpose of the linear 3×3 part of the node matrix, and tangents by the linear 3×3 part directly (covariant transform).
- Animation data is preserved in ``global_attrs["animations"]`` as decoded numpy arrays and re-encoded on ``write_scene``; a malformed animation, channel or sampler, or a sampler whose ``input`` or ``output`` is not the index of an accessor, raises :class:`~polyxios.exceptions.CodecError` on read.  Animated nodes are forced to TRS form (never matrix) as the glTF spec requires; a rest matrix that mirrors carries the reflection as a negative x scale, as a split key does, and one holding a shear or projection loses it with a warning. A channel whose meaning glTF lacks, targeting a node the scene does not have, whose sampler is not an object, holds no key, NaN, Inf or a value float32 cannot hold, times that are not strictly increasing, a rotation of zero length, or values that are not three per key for a translation or scale and four for a rotation, or animating a node path an earlier channel of its animation already animates, is skipped with a warning; a COLLADA channel (one naming a transform element's ``sid``, or a ``matrix`` channel on a node read without a spelling) is rebuilt as translation, rotation and scale: each node's channels are applied to the elements of its ``extras["transforms"]`` (a ``<matrix>`` member ``(i)(j)`` being its cell in column ``i``, row ``j``, and ``(i)`` its column ``i``, as other readers take them) at the union of their key times, the elements composed and each key split into the three, a key holding a shear or projection losing it with a warning; a node whose composed keys, or the translation or scale split from them, hold a value float32 cannot is not rebuilt, its three channels dropped together with a warning. COLLADA plays every animation at once, and some exporters write one per curve, so a node's channels in several animations are rebuilt together, in the first of them; an animation left with no channel is not written, so its name is lost and the animations after it move up one index. Animations that animate the same values cannot play together: each joins the first clip whose values it does not touch, the others stay clips of their own, and a warning says so when a clip holds at rest values only another animates. ``LINEAR`` and ``STEP`` keys are kept, a ``STEP`` channel beside blending ones gets a key one float32 step before each of its steps, ``BEZIER`` keys are sampled four times per span along their curve (a tangent of two values per key component pairs a time and a value per component, ``X`` time, ``X`` value, ``Y`` time and so on, as the OpenCOLLADA 3ds Max and Maya exporters write a whole element's and Blender reads it; a tangent of one more value than the key is one time, then a value per component, the layout the COLLADA 1.4.1 specification gives; and a tangent of one value per component is the control point's value, a third of the way into the span, the specification's 1-D form; a ``BEZIER`` tangent is a control point, never a slope), and every blending span is split until the translation and scale stray from the straight line between two keys by at most a thousandth of their range over the node's keys and the middles of its spans, as a curve does or a translation a ``<rotate>`` before it swings on an arc, and a span whose rotation turns more than 15 degrees through its middle is split until none does, since glTF slerps where COLLADA blends the elements (a whole ``<matrix>`` alone is blended as its decomposed keys, as glTF does, and is not split; a ``LINEAR`` whole ``<matrix>`` beside other elements is blended that way too, while a ``BEZIER`` one is followed element by element along its curve, as COLLADA blends it, so the shrink of a node turning between its keys is kept, and its scale's range takes that shrink in); the keys these splits add are capped at 262144 per write, with a warning, and a warning also says when spans still turn or stray that far after eight rounds of splitting; the channels come out ``STEP`` when every one of the node's steps, else ``LINEAR``. A COLLADA channel is skipped with a warning when its node spells no element of its ``sid`` and kind, when its ``member`` names no part of the element, when its keys are not as wide as the part it sets, when its interpolation is ``HERMITE``, ``CARDINAL`` or ``BSPLINE``, when a ``BEZIER`` sampler lacks tangents, when the node's matrix no longer composes from its elements to about six significant digits of its rotation and scale and, apart, of its translation (it was edited after reading), or when an earlier channel of the animation sets the same values; all of a node's in one animation are skipped with a warning naming that animation when another channel of it already animates the node's translation, rotation or scale, the node's channels in the other animations rebuilt without them; keys that change handedness from one to the next warn, since the rotation between them swings through the reflection, and the span holding the change is only halved per splitting pass, never split after that swing, its ``<rotate>`` ANGLE tracks still splitting it. Key times that float32 would collapse onto their neighbour are nudged down to stay strictly increasing, on every channel, or up when down would make the first time negative. A sampler several channels share is written once, as is a times array several samplers share. The values of a ``pointer`` channel, one row per key, are written as the accessor type as wide as a row (scalar, ``VEC2`` to ``VEC4``, ``MAT3`` or ``MAT4``); any other shape is skipped with a warning.
- Skins are kept in ``global_attrs["skins"]`` as their glTF dicts, each with its ``inverseBindMatrices`` accessor also decoded into ``inverse_bind_matrices``, ``(len(joints), 4, 4)`` row-major float64; a skin with no joints, or whose accessor is not float MAT4, has no ``bufferView``, cannot be read or holds fewer matrices than joints, is left undecoded with a warning. ``write_scene`` writes each skin back with its ``name``, ``joints``, ``skeleton``, ``extras`` and its ``inverse_bind_matrices`` as a float MAT4 accessor (none when the skin has no matrices, glTF's identity), and a node's ``extras["skin"]`` as its ``skin``. A ``bind_shape_matrix``, which glTF lacks, is folded into the inverse bind matrices. A skin whose joints are not distinct node indices under one root, whose matrices are not one finite 4x4 per joint with a last row of ``[0, 0, 0, 1]`` once the bind shape is folded in, or whose ``inverseBindMatrices`` was never decoded is left out with a warning; so is a skin's ``extensions``, since the writer keeps no ``extensionsUsed``, and a node's ``skin`` when it names no written skin, has no mesh, or its mesh has no ``joints`` and ``weights`` or a joint index past the skin's joints. ``joints`` and ``weights`` go out as a pair of ``VEC4``; uint8 and uint16 weights are taken as normalised and written as floats. Joints that are not whole numbers in 0..65535, or influences that are not four per vertex, are left out with their weights and a warning. ``joints_<n>`` and ``weights_<n>`` go out as ``JOINTS_<n>`` and ``WEIGHTS_<n>``, as a read names them.
- On write, a mesh's ``global_attrs["mesh_name"]`` becomes its ``name`` and a scene's ``global_attrs["extras"]`` the file's ``extras``. Vertex attributes glTF has no semantic for, element attributes other than ``material``, tags, other globals and ``extensions`` (the writer keeps no ``extensionsUsed``) are left out with a warning. ``tangents`` that are not 4 numbers per vertex, the ``VEC4`` a ``TANGENT`` is, are left out with a warning. A mesh without such ``tangents`` but with the tangent frame a COLLADA read gives writes it as ``TANGENT``: the tangent, scaled to unit length, as xyz and, as ``w``, the side of ``cross(normals, tangent)`` the bitangent lies on. The frame taken is the first of ``textangent`` and ``texbinormal``, then ``tangent`` and ``binormal``, that holds both of 3 columns, else the first tangent alone, whose handedness is taken as +1 with a warning, as is that of a row whose bitangent lies on neither side of ``cross(normals, tangent)`` beyond rounding; only set 0 is looked at, so a ``textangent_1`` or ``tangent_1`` is left out like any other attribute glTF has no semantic for. A COLLADA ``tangent`` names a geometric tangent, a ``textangent`` the texture-space one glTF means, and readers fill one tangent slot from either, so both are taken, a ``textangent`` frame first. Since glTF clients ignore a ``TANGENT`` without a ``NORMAL`` and require it of unit length, the frame is left out with a warning when the mesh has no ``normals`` of 3 columns or a tangent row of zero or non-finite length.
- Textures with ``image: -1`` are placeholder entries that preserve index alignment for ``KHR_texture_basisu`` / ``EXT_texture_webp`` extensions; they are read and written without a ``source`` key.
- Writing uses ``allow_nan=False`` in the JSON serialiser; a :class:`~polyxios.exceptions.CodecError` is raised if any floating-point value is NaN or Inf.
- External ``.bin`` buffers require a filesystem path; writing ``.gltf`` to a stream raises :class:`~polyxios.exceptions.CodecError`. Use ``binary=True`` for a self-contained GLB instead.
- TRS decomposition for animated-node export attempts scipy's ``Rotation.from_matrix`` and falls back to a pure-numpy Shepherd quaternion extraction if scipy is unavailable.

.. seealso::

   :doc:`index` - the full format table.
