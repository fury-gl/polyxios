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
- Animation data is preserved in ``global_attrs["animations"]`` as decoded numpy arrays and re-encoded on ``write_scene``; a malformed animation, channel or sampler, or a sampler whose ``input`` or ``output`` is not the index of an accessor, raises :class:`~polyxios.exceptions.CodecError` on read.  Animated nodes are forced to TRS form (never matrix) as the glTF spec requires; a rest matrix that mirrors carries the reflection as a negative x scale, as a split key does, and one holding a shear or projection loses it with a warning. A channel whose meaning glTF lacks, targeting a node the scene does not have, whose sampler is not an object, holds no key, NaN, Inf or a value float32 cannot hold, times that are not strictly increasing, a rotation of zero length, or values that are not three per key for a translation or scale and four for a rotation, or animating a node path an earlier channel of its animation already animates, is skipped with a warning; a ``matrix`` channel of ``LINEAR`` or ``STEP`` keys animating a node's whole transform is split into translation, rotation and scale channels, a key holding a shear or projection losing it with a warning, and is skipped with a warning when another channel of its animation already animates one of those paths; keys that change handedness from one to the next warn, since the rotation between them swings through the reflection. Key times that float32 would collapse onto their neighbour are nudged down to stay strictly increasing, on every channel, or up when down would make the first time negative. A sampler several channels share is written once, as is a times array several samplers share. The values of a ``pointer`` channel, one row per key, are written as the accessor type as wide as a row (scalar, ``VEC2`` to ``VEC4``, ``MAT3`` or ``MAT4``); any other shape is skipped with a warning.
- Skins are kept in ``global_attrs["skins"]`` as their glTF dicts, each with its ``inverseBindMatrices`` accessor also decoded into ``inverse_bind_matrices``, ``(len(joints), 4, 4)`` row-major float64; a skin with no joints, or whose accessor is not float MAT4, has no ``bufferView``, cannot be read or holds fewer matrices than joints, is left undecoded with a warning. ``write_scene`` writes each skin back with its ``name``, ``joints``, ``skeleton``, ``extras`` and its ``inverse_bind_matrices`` as a float MAT4 accessor (none when the skin has no matrices, glTF's identity), and a node's ``extras["skin"]`` as its ``skin``. A ``bind_shape_matrix``, which glTF lacks, is folded into the inverse bind matrices. A skin whose joints are not distinct node indices under one root, whose matrices are not one finite 4x4 per joint with a last row of ``[0, 0, 0, 1]`` once the bind shape is folded in, or whose ``inverseBindMatrices`` was never decoded is left out with a warning; so is a skin's ``extensions``, since the writer keeps no ``extensionsUsed``, and a node's ``skin`` when it names no written skin, has no mesh, or its mesh has no ``joints`` and ``weights`` or a joint index past the skin's joints. ``joints`` and ``weights`` go out as a pair of ``VEC4``; uint8 and uint16 weights are taken as normalised and written as floats. Joints that are not whole numbers in 0..65535, or influences that are not four per vertex, are left out with their weights and a warning. ``joints_<n>`` and ``weights_<n>`` go out as ``JOINTS_<n>`` and ``WEIGHTS_<n>``, as a read names them.
- On write, a mesh's ``global_attrs["mesh_name"]`` becomes its ``name`` and a scene's ``global_attrs["extras"]`` the file's ``extras``. Vertex attributes glTF has no semantic for, element attributes other than ``material``, tags, other globals and ``extensions`` (the writer keeps no ``extensionsUsed``) are left out with a warning.
- Textures with ``image: -1`` are placeholder entries that preserve index alignment for ``KHR_texture_basisu`` / ``EXT_texture_webp`` extensions; they are read and written without a ``source`` key.
- Writing uses ``allow_nan=False`` in the JSON serialiser; a :class:`~polyxios.exceptions.CodecError` is raised if any floating-point value is NaN or Inf.
- External ``.bin`` buffers require a filesystem path; writing ``.gltf`` to a stream raises :class:`~polyxios.exceptions.CodecError`. Use ``binary=True`` for a self-contained GLB instead.
- TRS decomposition for animated-node export attempts scipy's ``Rotation.from_matrix`` and falls back to a pure-numpy Shepherd quaternion extraction if scipy is unavailable.

.. seealso::

   :doc:`index` - the full format table.
