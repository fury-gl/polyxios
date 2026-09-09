# polyxios

**Fast, clean mesh I/O for Python.** Read and write 3D mesh files in one line - no hidden surprises, no silent data corruption.

---

## Install

```bash
pip install polyxios
```

or, from conda-forge:

```bash
conda install -c conda-forge polyxios
```

---

## Usage

```python
import polyxios as px

# Read any supported format
mesh = px.read("brain.vtk")

# Inspect
print(mesh.vertices.shape)  # (n_verts, 3)
print(len(mesh.element_types))  # number of elements
print(mesh.topological_dimension)  # 0 points, 1 lines, 2 surfaces, 3 volumes

# Write to a different format
px.write(mesh, "brain.ply")
px.write(mesh, "brain.vtp")
```

Need binary output or format-specific options?

```python
px.write(mesh, "brain.vtk", binary=True)
px.write(mesh, "brain.ply", binary=True, endian="little")
```

---

## Files, buffers and streams

Anything with a `read` or a `write` works where a path does, so a mesh never
has to touch disk:

```python
import io

buf = io.BytesIO()
px.write(mesh, buf, fmt=".ply")  # fmt= names the format

buf.seek(0)
same = px.read(buf, fmt=".ply")

with open("brain.vtk", "rb") as fh:
    mesh = px.read(fh)  # a named handle needs no fmt=
```

A handle polyxios is given is read or written where it stands and is never
closed - the caller keeps control of its own file. A buffer with no file name
has no extension to infer a format from, so `fmt=` is required there; `open()`
gives a handle a name, and that is enough. TetGen is the one format a buffer
cannot carry: a mesh is a `.node` and an `.ele` file found beside each other.

## Compressed files

gzip is transparent for every format at once:

```python
mesh = px.read("brain.vol.gz")  # decompressed on the way in
px.write(mesh, "brain.vtk.gz")  # compressed on the way out
px.write(mesh, buf, fmt=".obj.gz")  # a buffer says it with fmt=
```

Reading looks at the content, so a file compressed without being renamed reads
just as well as one ending in `.gz`. Writing looks at the name, an output file
having no content to inspect yet. The compressed output carries no timestamp
and no embedded name, so the same mesh always produces the same bytes.

---

## Command Line Interface (pxios)

polyxios comes with a command-line interface `pxios` to quickly fetch, list, convert, and visualize 3D models.

### Subcommands

`--verbose` can be given on either side of the subcommand (e.g. `pxios --verbose fetch bunny.obj` or `pxios fetch bunny.obj --verbose`) to print debug logs and full tracebacks when a command fails.

*   **`pxios list`**: Lists all available remote or cached files, or registered formats. The three listing modes below are mutually exclusive.
    *   `--local`: Lists locally cached files (can filter by optional extension argument, e.g. `pxios list obj --local`).
    *   `--extensions` / `--formats`: Lists all formats and extensions available in the remote catalog.
    *   `--codecs`: Lists all formats supported by polyxios codecs.
*   **`pxios fetch <filename|extension>`**: Downloads and caches a single model file (e.g., `bunny.obj`) or every model catalogued for an extension (e.g., `obj` or `.obj`).
*   **`pxios convert <input_file> <output_file>`**: Converts a model file from one format to another directly in a single process.
*   **`pxios viz <filename>`**: Visualizes a local or cached model file using the [FURY](https://fury.gl) library.
    *   `--lines`: Render line elements using `actor.line` instead of rendering as a surface/point cloud.
    *   `--points`: Render strictly as a point cloud.

```bash
# List all fetchable remote models
pxios list

# Fetch a single model
pxios fetch bunny.obj

# Fetch every model catalogued for an extension
pxios fetch vtk

# Convert a mesh file
pxios convert bunny.obj bunny.vtk

# Visualize a model
pxios viz bunny.obj
```

---

## Lazy loading - work with large files without filling RAM

For large meshes (gigabytes of binary data), pass `lazy=True`. polyxios
memory-maps the file and only loads the pages you actually touch - the rest
stays on disk until needed.

```python
# File is opened but data is not loaded into RAM yet
mesh = px.read("huge_brain.vtk", lazy=True)

# Only the vertices are pulled from disk here
first_vertex = mesh.vertices[0]

# Element connectivity is still on disk until you access it
```

`lazy=True` is honoured for binary `.vtk`, `.ply` and `.stl` files. ASCII
formats load eagerly (the whole file must be parsed to extract values). Binary
STL lazy mode skips vertex deduplication - vertices are returned as-is (3 per
triangle), avoiding the extra pass over the data. `.meshb` needs no flag: a
path is always memory-mapped, so `lazy=True` there warns and changes nothing.

`mmap` maps a file descriptor from byte zero, so the formats whose lazy read
hands back arrays viewing the mapping need a real, uncompressed file standing
at its start: an `io.BytesIO`, a handle part-way into a file, or a gzipped one
raises `LazyReadError` naming the reason rather than quietly loading eagerly.
Binary STL's lazy mode only skips work, so it takes a buffer or a compressed
file like any other read.

---

## Supported formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| VTK Legacy | `.vtk` | ✓ | ✓ | lazy: binary |
| VTK RectilinearGrid | `.vtr` | ✓ | ✓ | per-axis coordinate arrays, appended or inline base64 |
| VTK PolyData | `.vtp` | ✓ | ✓ | points, lines, polygons, strips |
| Wavefront OBJ | `.obj` | ✓ | ✓ | `vt`/`vn` round trip, groups → element tags |
| Stanford PLY | `.ply` | ✓ | ✓ | lazy: binary |
| STL | `.stl` | ✓ | ✓ | lazy: binary, which skips vertex deduplication |
| OFF | `.off` | ✓ | ✓ | ASCII + big-endian binary, `ST`/`C`/`N` variants → vertex/face attrs |
| Abaqus | `.inp` | ✓ | ✓ | `*NSET`/`*ELSET` → tags, planar cards for a 2-D deck |
| AVS-UCD | `.avs` | ✓ | ✓ | node/cell/model data → attrs |
| Medit binary | `.meshb` | ✓ | ✓ | a path is always mmapped; no `lazy=` needed |
| Medit ASCII | `.mesh`* `.medit` | ✓ | ✓ | reference integers → tags; write with `fmt=".medit"` |
| DOLFIN / FEniCS XML | `.xml` | ✓ | ✓ | interval/triangle/tetrahedron meshes |
| FLAC3D | `.f3grid` | ✓ | ✓ | zones + faces, groups → element tags |
| Gmsh | `.msh` | ✓ | ✓ (v2) | ASCII v2 + v4.1, physical groups → element tags |
| Nastran | `.bdf` `.nas` `.fem` `.dat`* | ✓ | ✓ | free/small/large field read, free-field write with large-field `GRID` on request |
| Tecplot ASCII | `.tec` `.dat`* | ✓ | ✓ | FE zone, POINT + BLOCK packing, solution variables → vertex attrs; binary `.plt` is recognised but not read |
| SU2 | `.su2` | ✓ | ✓ | ASCII, VTK element codes, boundary markers → element tags |
| TetGen | `.ele`+`.node` | ✓ | ✓ | paired files, 1-/0-based indices, boundary markers → vertex tags, region attrs |
| Well-Known Text | `.wkt` | ✓ | ✓ | 2D padded to z=0, holes → element attrs, EWKT SRID dropped |
| VTK UnstructuredGrid | `.vtu` | ✓ | ✓ | arbitrary cell-type mix |
| VTK StructuredGrid | `.vts` | ✓ | ✓ | curvilinear grid, cells implied by the extent (hexahedra, or quads when flat) |
| VTK ImageData | `.vti` | ✓ | ✓ | origin/spacing/extent only, no coordinate array |
| MFEM mesh | `.mesh`* | ✓ | ✓ | geometry type codes; INLINE is materialised, NURBS reads back control points |
| Netgen | `.vol` | ✓ | ✓ | ASCII, points/edges/faces/cells incl. quadratic, `bcnr`/`matnr` + names → element tags |
| UGRID (AFLR) | `.ugrid` | ✓ | ✓ | ASCII, tri/quad surface + tet/pyramid/prism/hex volume, boundary tags → element tags |
| Gaussian splat | `.splat` | ✓ | ✓ | headerless 32-byte records, points only |
| Kratos MDPA | `.mdpa` | ✓ | ✓ | ASCII, sub model parts → tags, nodal/elemental data → attrs, conditions read as elements |

\* `.dat` belongs to no single format, so it is resolved by content: a Tecplot header lands
in the Tecplot codec, a bulk data card in the Nastran one, and anything else reports both
candidates. Writing to `.dat` needs an explicit `fmt=`. `.mesh` is MFEM's own extension and
Medit ASCII shares it: a file opening with `MeshVersionFormatted` reads as Medit, one opening
with `MFEM mesh` reads as MFEM, and a bare write goes to MFEM.

`.vtm`, `.pvtu`, `.pvts`, `.pvti`, `.pvtp` and `.pvtr` are registered too, but they hold no
geometry - only references to sub-files. Reading one raises `UnsupportedFormatError` pointing
at `examples/read_parallel_vtk.py` rather than failing with a parse error further in; writing
them is not supported.

**27 formats supported** across the 31 extensions in the table, plus `.plt`, which
is recognised but not read - more coming via the plugin system.

---

## Transforms

Every transform takes a `PolyData` and returns a new one - nothing is modified
in place - so they compose freely.

```python
from functools import partial

from polyxios.transforms import (
    pipeline,
    merge,
    merge_duplicate_vertices,
    filter_element_type,
    remove_orphan_vertices,
)

# Compose transforms into a single function
clean = pipeline(
    partial(filter_element_type, keep="triangle"),
    remove_orphan_vertices,
)
result = clean(mesh)

# Weld coincident vertices - the STL facet soup back into a surface
welded = merge_duplicate_vertices(mesh)
snapped = merge_duplicate_vertices(mesh, tol=1e-6)

# Merge two meshes into one
combined = merge(mesh_a, mesh_b)
```

| Transform | What it does |
|-----------|--------------|
| `pipeline(*fns)` | Compose transforms left to right into one callable |
| `merge(*polys)` | Concatenate several meshes into one, offsetting the indices |
| `filter_element_type(poly, keep=...)` | Keep only the named element types |
| `remove_orphan_vertices(poly)` | Drop vertices no element references, remap indices |
| `reindex(poly)` | Alias of `remove_orphan_vertices` |
| `merge_duplicate_vertices(poly, tol=...)` | Weld coincident vertices into one |
| `triangulate(poly)` | Split every surface element into triangles |
| `extract_surface(poly)` | Return the boundary faces of a volumetric mesh |
| `vertex_colors(poly)` | Per-vertex RGB out of the vertex attributes, or `None` |

---

## Add your own format

Any third-party package can teach polyxios to read and write a new format -
no fork required, no pull request needed.

**Step 1 - write a codec** (two functions, nothing more):

```python
# mypackage/abc_codec.py
from polyxios._registry import Codec
from polyxios._types import PolyData


def read(path, *, lazy=False) -> PolyData: ...


def write(poly: PolyData, path, **opts) -> None: ...


def register():
    return ".abc", Codec(read, write)
```

**Step 2 - declare an entry point** in your `pyproject.toml`:

```toml
[project.entry-points."polyxios.codecs"]
abc = "mypackage.abc_codec:register"
```

After `pip install mypackage`, polyxios picks up `.abc` automatically -
no configuration, no restart needed:

```python
mesh = px.read("model.abc")  # works out of the box
```

---

## Contributing / Development

Clone the repo, then use [spin](https://github.com/scientific-python/spin) to
manage the development workflow:

```bash
pip install spin
spin setup       # add upstream remote + install dev deps (libomp on macOS)
spin install     # build Cython extensions and install
spin install -e  # editable install (source changes reflected immediately)
```

| Command | Description |
|---------|-------------|
| `spin setup` | First-time setup: upstream remote, dev deps, OpenMP on macOS |
| `spin build` | Build with Meson/ninja |
| `spin install` | Regular install (compiled) |
| `spin install -e` | Editable install for development |
| `spin test` | Run the full test suite |
| `spin test -k <pattern>` | Run tests matching a name pattern |
| `spin lint` | ruff linter + formatter check + codespell |
| `spin lint --fix` | Auto-fix lint and formatting issues |
| `spin docs` | Build Sphinx documentation |
| `spin docs --clean` | Wipe `_build/` before building |
| `spin docs --open` | Build and open docs in the browser |
| `spin clean` | Remove build artifacts and `__pycache__` |
| `spin release <version>` | Cut a release: bump version, tag, push, start next dev cycle |

See [`docs/contributing.rst`](docs/contributing.rst) for commit message
conventions and the full contributor guide.
For the full release workflow see [`docs/development.rst`](docs/development.rst).

---

## Why polyxios?

- **No silent data corruption** - large mesh indices raise an error instead of truncating
- **All element groups preserved** - a face belonging to multiple tags stays in all of them
- **Safe on untrusted files** - header counts validated before any memory allocation
- **Memory-efficient** - lazy mmap loading for large binary files
- **Paths, buffers and gzip alike** - one API over files, streams and `.gz`
- **Works without a compiler** - pure Python fallbacks included; Cython hot-paths optional

---

## License

See [LICENSE](LICENSE).
