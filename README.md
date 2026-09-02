# polyxios

**Fast, clean mesh I/O for Python.** Read and write 3D mesh files in one line - no hidden surprises, no silent data corruption.

---

## Install

```bash
pip install polyxios
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

```python
from functools import partial

from polyxios.transforms import (
    pipeline,
    merge,
    filter_element_type,
    remove_orphan_vertices,
)

# Compose transforms into a single function
clean = pipeline(
    partial(filter_element_type, keep="triangle"),
    remove_orphan_vertices,
)
result = clean(mesh)

# Merge two meshes into one
combined = merge(mesh_a, mesh_b)
```

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
- **Works without a compiler** - pure Python fallbacks included; Cython hot-paths optional

---

## License

See [LICENSE](LICENSE).
