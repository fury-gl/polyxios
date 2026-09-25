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
- Animation data is preserved in ``global_attrs["animations"]`` as decoded numpy arrays and re-encoded on ``write_scene``.  Animated nodes are forced to TRS form (never matrix) as the glTF spec requires.
- Textures with ``image: -1`` are placeholder entries that preserve index alignment for ``KHR_texture_basisu`` / ``EXT_texture_webp`` extensions; they are read and written without a ``source`` key.
- Writing uses ``allow_nan=False`` in the JSON serialiser; a :class:`~polyxios.exceptions.CodecError` is raised if any floating-point value is NaN or Inf.
- External ``.bin`` buffers require a filesystem path; writing ``.gltf`` to a stream raises :class:`~polyxios.exceptions.CodecError`. Use ``binary=True`` for a self-contained GLB instead.
- TRS decomposition for animated-node export attempts scipy's ``Rotation.from_matrix`` and falls back to a pure-numpy Shepherd quaternion extraction if scipy is unavailable.

.. seealso::

   :doc:`index` - the full format table.
