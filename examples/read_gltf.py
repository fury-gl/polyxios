"""Read a glTF/GLB file, print a scene summary, and visualize with optional animation.

When no file is given, ``Fox.glb`` is fetched automatically from the
polyxios-data release and cached under ``~/.polyxios/glTF/``.

Usage
-----
    python examples/read_gltf.py                          # fetch & show Fox.glb
    python examples/read_gltf.py Fox.glb                  # same, explicit
    python examples/read_gltf.py AnimatedCube.gltf        # fetch .gltf + companions
    python examples/read_gltf.py path/to/local.glb        # local file
    python examples/read_gltf.py --list                   # show cached glTF files
    python examples/read_gltf.py Fox.glb --no-viz         # summary only
    python examples/read_gltf.py Fox.glb --no-anim        # static rest pose
    python examples/read_gltf.py Fox.glb --lines          # wireframe
    python examples/read_gltf.py Fox.glb --points         # point cloud

Visualization requires FURY 2.0 (``pip install polyxios[viz]``).

Color resolution priority:
  1. Vertex colors (``COLOR_0``) stored in the glTF mesh.
  2. ``baseColorTexture`` sampled at vertex UV coordinates (``TEXCOORD_0``).
  3. ``baseColorFactor`` from the PBR material, broadcast to all vertices.
  4. Default grey when no color information is available.

When animations are present the file's keyframes are fed into
``fury.motion.Timeline`` and played back with a GUI playback panel.
Pass ``--no-anim`` to render the static rest pose instead.
"""

import argparse
import dataclasses
import io
from pathlib import Path
import sys
import warnings

import numpy as np

import polyxios
from polyxios._scene import SceneData, SceneImage
from polyxios.fetcher import fetch, get_cached_files

_DEFAULT_FILE = "Fox.glb"


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _load_image_rgba(img: SceneImage, base_dir: Path | None) -> np.ndarray | None:
    """Load a SceneImage as an (H, W, 4) float32 RGBA array, or None on failure."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None
    try:
        if img.data is not None:
            pil = PILImage.open(io.BytesIO(img.data)).convert("RGBA")
        elif img.uri is not None and not img.uri.startswith("data:"):
            src = (base_dir / img.uri) if base_dir else Path(img.uri)
            pil = PILImage.open(src).convert("RGBA")
        else:
            return None
        return np.array(pil, dtype=np.float32) / 255.0
    except Exception:
        return None


def _sample_texture(img_rgba: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Nearest-neighbour texture sample at per-vertex UV coordinates.

    Parameters
    ----------
    img_rgba
        (H, W, 4) float32 RGBA image.
    uvs
        (N, 2) float64 UV coordinates.

    Returns
    -------
    numpy.ndarray
        (N, 3) float32 RGB sampled colors.
    """
    h, w = img_rgba.shape[:2]
    px = np.clip((uvs[:, 0] % 1.0 * w).astype(np.int32), 0, w - 1)
    py = np.clip((uvs[:, 1] % 1.0 * h).astype(np.int32), 0, h - 1)
    return img_rgba[py, px, :3]


def _resolve_colors(
    poly: polyxios.PolyData,
    scene: SceneData,
    source_path: str,
) -> np.ndarray | None:
    """Return per-vertex (N, 3) float32 RGB in [0, 1], or None."""
    n_verts = poly.vertices.shape[0]

    if "colors" in poly.vertex_attrs:
        raw = poly.vertex_attrs["colors"].astype(np.float32)
        rgb = raw[:, :3]
        return rgb / 255.0 if rgb.max() > 1.0 else rgb

    if not scene.materials:
        return None

    mat_col = poly.element_attrs.get("material")
    if mat_col is None:
        return None

    base_dir = Path(source_path).parent if not source_path.startswith("data:") else None
    vert_colors = np.full((n_verts, 3), 0.7, dtype=np.float32)
    image_cache: dict[int, np.ndarray | None] = {}
    uvs = poly.vertex_attrs.get("texcoords")

    for elem_i in range(len(poly.element_types)):
        mat_idx = int(mat_col[elem_i])
        if mat_idx < 0 or mat_idx >= len(scene.materials):
            continue
        mat = scene.materials[mat_idx]
        start = int(poly.offsets[elem_i])
        end = int(poly.offsets[elem_i + 1])
        vis = poly.connectivity[start:end].tolist()

        tex_idx = mat.base_color_texture
        if tex_idx is not None and uvs is not None and tex_idx < len(scene.textures):
            img_idx = scene.textures[tex_idx].image
            if img_idx not in image_cache:
                image_cache[img_idx] = (
                    _load_image_rgba(scene.images[img_idx], base_dir)
                    if img_idx < len(scene.images)
                    else None
                )
            img_rgba = image_cache[img_idx]
            if img_rgba is not None:
                factor = np.array(mat.base_color[:3], dtype=np.float32)
                sampled = _sample_texture(img_rgba, uvs[vis]) * factor
                for local_i, vi in enumerate(vis):
                    vert_colors[vi] = sampled[local_i]
                continue

        r, g, b, _ = mat.base_color
        for vi in vis:
            vert_colors[vi] = (r, g, b)

    return vert_colors


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(path: str, scene: SceneData) -> None:
    """Print a human-readable summary of the scene."""
    anims = scene.global_attrs.get("animations", [])
    print(f"File:       {path}")
    print(f"Meshes:     {len(scene.meshes)}")
    print(f"Nodes:      {len(scene.nodes)}")
    print(f"Materials:  {len(scene.materials)}")
    print(f"Textures:   {len(scene.textures)}")
    print(f"Images:     {len(scene.images)}")
    print(f"Scenes:     {len(scene.scenes)}")
    print(f"Animations: {len(anims)}")
    print()

    for i, mesh in enumerate(scene.meshes):
        name = mesh.global_attrs.get("mesh_name", f"mesh_{i}")
        print(
            f"  Mesh {i} ({name!r}): "
            f"{mesh.vertices.shape[0]} vertices, "
            f"{len(mesh.element_types)} elements"
        )
        if mesh.vertex_attrs:
            print(f"    vertex_attrs:  {sorted(mesh.vertex_attrs)}")
        if mesh.element_attrs:
            print(f"    element_attrs: {sorted(mesh.element_attrs)}")

    if scene.materials:
        print()
        for i, mat in enumerate(scene.materials):
            r, g, b, a = mat.base_color
            has_tex = mat.base_color_texture is not None
            tex_note = f" + texture[{mat.base_color_texture}]" if has_tex else ""
            print(
                f"  Material {i}: {mat.name!r}  "
                f"base_color=({r:.2f},{g:.2f},{b:.2f}){tex_note}  "
                f"metallic={mat.metallic:.2f}  roughness={mat.roughness:.2f}"
            )

    if anims:
        print()
        for i, anim in enumerate(anims):
            n_ch = len(anim.get("channels", []))
            print(f"  Animation {i}: {anim.get('name', '?')!r}  channels={n_ch}")


# ---------------------------------------------------------------------------
# Actor builder
# ---------------------------------------------------------------------------


def _make_actor(
    poly: polyxios.PolyData,
    colors: np.ndarray | None,
    *,
    lines: bool,
    points: bool,
) -> object:
    """Return a single FURY actor for the given PolyData."""
    from fury import actor as fury_actor

    default_color = (0.7, 0.7, 0.7)

    if points or poly.vertices.shape[0] == 0:
        return fury_actor.point(
            poly.vertices.astype(np.float32),
            colors=colors if colors is not None else default_color,
        )

    faces = poly.faces
    if faces is None or len(faces) == 0:
        return fury_actor.point(
            poly.vertices.astype(np.float32),
            colors=colors if colors is not None else default_color,
        )

    return fury_actor.surface(
        poly.vertices.astype(np.float32),
        faces,
        colors=colors,
    )


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------


def _build_timeline(
    scene: SceneData,
    node_actors: dict[int, object],
) -> object | None:
    """Build a fury.motion.Timeline from decoded glTF animation data.

    Parameters
    ----------
    scene
        SceneData whose ``global_attrs["animations"]`` has decoded
        ``"times"`` and ``"values"`` ndarrays per sampler.
    node_actors
        Maps node index → FURY actor.

    Returns
    -------
    Timeline or None
        None when no animated nodes have actors.
    """
    from fury.motion import (
        Animation,
        Timeline,
        linear_interpolator,
        slerp,
        step_interpolator,
    )

    anims_data = scene.global_attrs.get("animations", [])
    if not anims_data:
        return None

    interp_fns: dict[str, dict[str, object]] = {
        "LINEAR": {
            "position": linear_interpolator,
            "scale": linear_interpolator,
            "rotation": slerp,
        },
        "STEP": {
            "position": step_interpolator,
            "scale": step_interpolator,
            "rotation": step_interpolator,
        },
    }

    timeline = Timeline(playback_panel=True)
    any_added = False

    for anim_data in anims_data:
        samplers = anim_data["samplers"]
        for ch in anim_data.get("channels", []):
            target = ch.get("target", {})
            node_idx = target.get("node")
            path = target.get("path", "")
            sampler_idx = ch.get("sampler", 0)

            if node_idx not in node_actors or sampler_idx >= len(samplers):
                continue

            sampler = samplers[sampler_idx]
            times: np.ndarray = sampler["times"]
            values: np.ndarray = sampler["values"]
            interp_name: str = sampler.get("interpolation", "LINEAR")

            if interp_name == "CUBICSPLINE":
                n = len(times)
                values = values.reshape(n, 3, -1)[:, 1, :].squeeze()
                interp_name = "LINEAR"

            fns = interp_fns.get(interp_name, interp_fns["LINEAR"])
            anim = Animation(actors=node_actors[node_idx])

            if path == "translation":
                for t, v in zip(times, values):
                    anim.set_position(float(t), v)
                anim.set_position_interpolator(fns["position"])
            elif path == "rotation":
                for t, v in zip(times, values):
                    anim.set_rotation(float(t), v)
                anim.set_rotation_interpolator(fns["rotation"])
            elif path == "scale":
                for t, v in zip(times, values):
                    anim.set_scale(float(t), v)
                anim.set_scale_interpolator(fns["scale"])
            else:
                continue

            timeline.add_animation(anim)
            any_added = True

    return timeline if any_added else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch (if needed), summarize, and visualize a glTF/GLB file."""
    parser = argparse.ArgumentParser(
        description=(
            "Read a glTF/GLB file, print its scene summary, and visualize it.\n\n"
            f"When no file is given, '{_DEFAULT_FILE}' is fetched automatically\n"
            "from the polyxios-data release and cached under ~/.polyxios/glTF/.\n\n"
            "Available files in the glTF release:\n"
            "  GLB: BoxVertexColors  MetalRoughSpheresNoTextures  AnimatedMorphCube\n"
            "       Avocado  WaterBottle  BoxAnimated  RiggedSimple  Fox\n"
            "  glTF: Triangle  TriangleWithoutIndices  SimpleSkin  AnimatedCube"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help=(
            f"Filename to fetch and visualize (e.g. 'Fox.glb', 'AnimatedCube.gltf'), "
            f"or a local path. Omit to use '{_DEFAULT_FILE}'."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List locally cached glTF files and exit.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Print summary only; skip visualization.",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Ignore animation data; render static rest pose.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Render surface wireframe instead of solid faces.",
    )
    parser.add_argument(
        "--points",
        action="store_true",
        help="Render as point cloud.",
    )
    args = parser.parse_args()

    if args.list:
        cached = get_cached_files("glb") + get_cached_files("gltf")
        if not cached:
            print(
                "No glTF files cached locally.\n"
                "Run without --list to fetch Fox.glb automatically."
            )
        else:
            print("Cached glTF files:")
            for p in sorted(cached):
                print(f"  {p}")
        sys.exit(0)

    # Resolve path — fetch from release if not a local file.
    filename = args.filename or _DEFAULT_FILE
    p = Path(filename)
    if p.exists() or p.is_absolute():
        path = str(p)
    else:
        path = fetch(filename)

    scene = polyxios.read_scene(path)
    _print_summary(path, scene)

    if args.no_viz:
        return

    try:
        from fury import window
    except ImportError:
        print("FURY not installed.  Run: pip install polyxios[viz]", file=sys.stderr)
        sys.exit(1)

    anims_data = scene.global_attrs.get("animations", [])
    use_anim = bool(anims_data) and not args.no_anim

    if not use_anim:
        # Static: flatten entire scene into one actor.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat_poly = polyxios.read(path)

        colors = _resolve_colors(flat_poly, scene, path)
        if colors is not None:
            flat_poly = dataclasses.replace(
                flat_poly,
                vertex_attrs={**flat_poly.vertex_attrs, "colors": colors},
            )

        act = _make_actor(flat_poly, colors, lines=args.lines, points=args.points)
        window.show(act, title=f"polyxios — {Path(path).name}")
        return

    # Animated: one actor per node so the Timeline moves each independently.
    node_actors: dict[int, object] = {}
    for node_idx, node in enumerate(scene.nodes):
        if node.mesh is None:
            continue
        poly = scene.meshes[node.mesh]
        if poly.vertices.shape[0] == 0:
            continue
        colors = _resolve_colors(poly, scene, path)
        node_actors[node_idx] = _make_actor(
            poly, colors, lines=args.lines, points=args.points
        )

    if not node_actors:
        print("No renderable geometry found.")
        return

    timeline = _build_timeline(scene, node_actors)
    if timeline is None:
        print("No animated nodes found; falling back to static render.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat_poly = polyxios.read(path)
        colors = _resolve_colors(flat_poly, scene, path)
        act = _make_actor(flat_poly, colors, lines=args.lines, points=args.points)
        window.show(act, title=f"polyxios — {Path(path).name}")
        return

    print(
        f"Playing {len(anims_data)} animation(s) — use the playback panel to control."
    )

    fury_scene = window.Scene()
    show_m = window.ShowManager(
        scene=fury_scene,
        title=f"polyxios — {Path(path).name}",
        size=(1200, 900),
    )
    show_m.add_animation(timeline)
    show_m.start()


if __name__ == "__main__":
    main()
