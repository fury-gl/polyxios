# polyxios

**Fast, clean mesh I/O for Python.** Read and write 3D mesh files in one line - no hidden surprises, no silent data corruption.

[![PyPI](https://img.shields.io/pypi/v/polyxios.svg)](https://pypi.org/project/polyxios/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/polyxios.svg)](https://anaconda.org/conda-forge/polyxios)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/polyxios/)
[![License](https://img.shields.io/github/license/fury-gl/polyxios.svg)](https://github.com/fury-gl/polyxios/blob/master/LICENSE)
[![Tests](https://github.com/fury-gl/polyxios/actions/workflows/test.yml/badge.svg)](https://github.com/fury-gl/polyxios/actions/workflows/test.yml)

**Documentation: [polyxios.org](https://polyxios.org)**

- [Installation](https://polyxios.org/stable/installation.html)
- [User guide](https://polyxios.org/stable/usage.html)
- [Supported formats](https://polyxios.org/stable/formats/index.html)
- [API reference](https://polyxios.org/stable/api/index.html)
- [Changelog](https://polyxios.org/stable/changelog.html)
- [Issue tracker](https://github.com/fury-gl/polyxios/issues)

---

## Install

```bash
pip install polyxios
```

or, from conda-forge:

```bash
conda install -c conda-forge polyxios
```

The XDMF codec keeps its arrays in an HDF5 file by default, and MED, CGNS,
H5M and HMF are HDF5 files outright; all need
[h5py](https://www.h5py.org/) - an optional extra:

```bash
pip install "polyxios[hdf5]"
```

Exodus II is a netCDF file and needs [netCDF4](https://unidata.github.io/netcdf4-python/)
the same way - `pip install "polyxios[netcdf]"` - and a LAZ-compressed LAS file needs
[lazrs](https://github.com/laz-rs/laz-rs-python) - `pip install "polyxios[laz]"`.

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

Scene-aware formats preserve hierarchy, materials, and textures:

```python
# Read the full scene graph
scene = px.read_scene("robot.glb")
print(len(scene.nodes), "nodes,", len(scene.meshes), "meshes")

# Flatten to a single PolyData (applies transforms, merges meshes)
mesh = scene.to_polydata()

# read() on a glTF file flattens automatically (issues a warning)
mesh = px.read("robot.glb")
```

---

## Files, buffers and streams

Full details in the [user guide](https://polyxios.org/stable/usage.html).

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
The [CLI reference](https://polyxios.org/stable/cli.html) documents every flag.

### Subcommands

`--verbose` can be given on either side of the subcommand (e.g. `pxios --verbose fetch stanford-bunny.obj` or `pxios fetch stanford-bunny.obj --verbose`) to print debug logs and full tracebacks when a command fails. `pxios --version` prints the installed version and exits.

*   **`pxios list`**: Lists all available remote or cached files, or registered formats. The three listing modes below are mutually exclusive.
    *   `--local`: Lists locally cached files (can filter by optional extension argument, e.g. `pxios list obj --local`).
    *   `--extensions` / `--formats`: Lists all formats and extensions available in the remote catalog.
    *   `--codecs`: Lists all formats supported by polyxios codecs.
*   **`pxios fetch <filename|extension>`**: Downloads and caches a single model file (e.g., `stanford-bunny.obj`) or every model catalogued for an extension (e.g., `obj` or `.obj`).
    *   `--overwrite`: Download again over a file already in the cache.
*   **`pxios convert <input_file> <output_file>`**: Converts a model file from one format to another directly in a single process.
    *   `--force` / `-f`: Overwrite an existing output file.
*   **`pxios viz <filename>`**: Visualizes a local or cached model file using the [FURY](https://fury.gl) library.
    *   `--lines`: Render line elements using `actor.line` instead of rendering as a surface/point cloud.
    *   `--points`: Render strictly as a point cloud.

```bash
# List all fetchable remote models
pxios list

# Fetch a single model
pxios fetch stanford-bunny.obj

# Fetch every model catalogued for an extension
pxios fetch vtk

# Convert a mesh file
pxios convert stanford-bunny.obj bunny.vtk

# Visualize a model
pxios viz stanford-bunny.obj
```

---

## Lazy loading - read large files without copying them

See the [lazy loading guide](https://polyxios.org/stable/lazy_loading.html) for the full picture.

For a large binary mesh, pass `lazy=True`. polyxios maps the file and,
wherever the file stores an array as one run of bytes in the shape the array
needs, hands back a read-only view of the mapping in the file's own dtype -
nothing decoded, nothing copied, pages coming in as they are touched.

```python
mesh = px.read("huge.vtu", lazy=True)

mesh.vertices.flags.writeable  # False: these are the file's bytes
mesh.vertices.dtype  # whatever the file holds - float32 stays float32
mesh.vertices[0]  # the first page comes in here
```

Which arrays can be views depends on how the format lays them out. `.vtu`
and `.vtp` with a raw appended section (VTK's default output, and
`write(..., appended=True)`), `.xdmf` over a binary or contiguous-HDF5
sidecar, `.splat`, and binary `.vtk` in the v5.1 layout map their vertices,
connectivity and attributes; binary `.ply`, `.meshb` and v4.2 `.vtk` map
their vertices and decode their cells, whose on-disk form interleaves counts
or references with the indices. Binary STL's `lazy=True` skips vertex
deduplication instead - three vertices per triangle, copied - because a
50-byte STL record cannot be viewed as coordinates. Everything encoded
(ASCII, base64, zlib, ZIP) raises `LazyReadError` or, for some text formats,
warns and loads eagerly.

`mmap` maps a file descriptor from byte zero, so a lazy read needs a real,
uncompressed file standing at its start: an `io.BytesIO`, a handle part-way
into a file, or a gzipped one raises `LazyReadError` naming the reason rather
than quietly loading eagerly.

---

## Supported formats

Each format has its own page at [polyxios.org/stable/formats](https://polyxios.org/stable/formats/index.html)
describing what is read, what is written and what is dropped.

### Scene & animation

Formats that carry a full scene graph — node hierarchy, materials, textures, and animations.
Use `read_scene` / `write_scene` to preserve the structure; `read` flattens to a single mesh.

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| glTF 2.0 | `.gltf` `.glb` | ✓ | ✓ | `read_scene` returns full hierarchy, PBR materials, animations; `read()` flattens with a warning |

### Surface, point & interchange

Primarily surface meshes, point clouds, and widely used interchange formats.

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| VTK Legacy | `.vtk` | ✓ | ✓ | lazy: binary v5.1 zero-copy, v4.2 all but cells |
| VTK PolyData | `.vtp` | ✓ | ✓ | points, lines, polygons, strips; lazy: raw appended, `appended=True` writes it |
| Wavefront OBJ | `.obj` | ✓ | ✓ | `vt`/`vn` round trip, groups → element tags |
| Stanford PLY | `.ply` | ✓ | ✓ | lazy: binary vertices; faces decoded |
| STL | `.stl` | ✓ | ✓ | lazy: binary, which skips vertex deduplication and copies |
| 3MF | `.3mf` | ✓ | ✓ | objects → element tags, assemblies placed by transform, materials → `colors`, `unit` in `global_attrs` |
| OFF | `.off` | ✓ | ✓ | ASCII + big-endian binary, `ST`/`C`/`N` variants → vertex/face attrs |
| AVS-UCD | `.avs` | ✓ | ✓ | node/cell/model data → attrs |
| Medit binary | `.meshb` | ✓ | ✓ | lazy: vertices; elements decoded |
| Medit ASCII | `.mesh`* `.medit` | ✓ | ✓ | reference integers → tags; write with `fmt=".medit"` |
| Well-Known Text | `.wkt` | ✓ | ✓ | 2D padded to z=0, holes → element attrs, EWKT SRID dropped |
| Gaussian splat | `.splat` | ✓ | ✓ | headerless 32-byte records, points only; lazy: zero-copy |
| PCD (Point Cloud Library) | `.pcd` | ✓ | ✓ | ascii, binary and LZF `binary_compressed`; fields → vertex attrs, packed `rgb` → `colors`, organised grid and viewpoint kept; lazy: binary views |
| LAS / LAZ (LiDAR) | `.las` `.laz` | ✓ | ✓ | 1.0-1.4, point formats 0-10; scaled ints → float64, every field and flag → vertex attrs, `red green blue` → `colors`, extra bytes by name, WKT / GeoTIFF CRS → `crs_wkt` / `las_geokeys`; LAZ via `polyxios[laz]`; lazy: `.las` views |
| ASCII point cloud | `.xyz` `.pts` `.ptx` | ✓ | ✓ | columns named by count and kind (intensity, colours, normals); `.pts` count line; `.ptx` scans transformed by their pose and tagged |

### Volume, grid & simulation

Volumetric meshes, structured grids, and FEM/CFD simulation formats.

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| VTK RectilinearGrid | `.vtr` | ✓ | ✓ | per-axis coordinate arrays, appended or inline base64 |
| VTK StructuredGrid | `.vts` | ✓ | ✓ | curvilinear grid, cells implied by the extent (hexahedra, or quads when flat) |
| VTK ImageData | `.vti` | ✓ | ✓ | origin/spacing/extent only, no coordinate array |
| VTK UnstructuredGrid | `.vtu` | ✓ | ✓ | arbitrary cell-type mix |
| MFEM mesh | `.mesh`* | ✓ | ✓ | geometry type codes; INLINE is materialised, NURBS reads back control points |
| Netgen | `.vol` | ✓ | ✓ | ASCII, points/edges/faces/cells incl. quadratic, `bcnr`/`matnr` + names → element tags |
| UGRID (AFLR) | `.ugrid` | ✓ | ✓ | ASCII, tri/quad surface + tet/pyramid/prism/hex volume, boundary tags → element tags |
| DOLFIN / FEniCS XML | `.xml` | ✓ | ✓ | interval/triangle/tetrahedron meshes |
| Abaqus | `.inp` | ✓ | ✓ | `*NSET`/`*ELSET` → tags, planar cards for a 2-D deck |
| FLAC3D | `.f3grid` | ✓ | ✓ | zones + faces, groups → element tags |
| Gmsh | `.msh`* | ✓ | ✓ (v2) | ASCII v2 + v4.1, physical groups → element tags |
| Nastran | `.bdf` `.nas` `.fem` `.dat`* | ✓ | ✓ | free/small/large field read, free-field write with large-field `GRID` on request |
| Tecplot ASCII | `.tec` `.dat`* | ✓ | ✓ | FE zone, POINT + BLOCK packing, solution variables → vertex attrs; binary `.plt` is recognised but not read |
| SU2 | `.su2` | ✓ | ✓ | ASCII, VTK element codes, boundary markers → element tags |
| TetGen | `.ele`+`.node` | ✓ | ✓ | paired files, 1-/0-based indices, boundary markers → vertex tags, region attrs |
| Kratos MDPA | `.mdpa` | ✓ | ✓ | ASCII, sub model parts → tags, nodal/elemental data → attrs, conditions read as elements |
| PERMAS | `.dato` `.post` `.dat`* | ✓ | ✓ | ASCII, `$NSET`/`$ESET` → tags, free numbering → `original_ids`, `element_type=` picks the solver class |
| ANSYS Fluent | `.msh`* `.fluent` | ✓ | ✓ | ASCII + binary sections, cells assembled from faces, zones → element tags, boundary faces read as elements; write with `fmt="fluent"` |
| SVG | `.svg` | – | ✓ | a picture of the mesh projected onto a plane, one `<path>` per element type; `plane=`, `width=`, `stroke_width=` |
| XDMF | `.xdmf` `.xmf` | ✓ | ✓ | XML light data over HDF5 (`pip install "polyxios[hdf5]"`), binary or inline arrays; mixed topologies, lattices, `<Set>` → tags, time series via `helper.read_time_series` / `write_time_series`, `step=` on read |
| MED (Salome) | `.med` | ✓ | ✓ | HDF5 (`polyxios[hdf5]`); families and groups → tags, fields on nodes, cells and Gauss points → attrs, several meshes merged or `mesh=` picks one, `step=` on read |
| CGNS | `.cgns` | ✓ | ✓ | HDF5 (`polyxios[hdf5]`); every zone read and tagged or `zone=` picks one, MIXED and NGON sections, structured zones expanded, `ZoneBC` → tags, `FlowSolution` → attrs |
| H5M (MOAB) | `.h5m` | ✓ | ✓ | HDF5 (`polyxios[hdf5]`); dense tags → attrs, meshsets → tags named by `NAME` or their material / boundary number, `GLOBAL_ID` → `original_ids` |
| HMF | `.hmf` | ✓ | ✓ | HDF5 (`polyxios[hdf5]`); XDMF's model in one file, typed topologies, node and cell attributes, sets and mesh-wide values |
| Exodus II | `.e` `.exo` `.ex2` | ✓ | ✓ | netCDF (`polyxios[netcdf]`); blocks, node sets and element sets → tags, side sets → faces with `face_parent`, nodal / element / global variables → attrs, `step=` on read |

\* `.dat` belongs to no single format, so it is resolved by content: a Tecplot header lands
in the Tecplot codec, a bulk data card in the Nastran one, a `$` keyword record in the PERMAS
one, and anything else reports the candidates. Writing to `.dat` needs an explicit `fmt=`.
`.mesh` is MFEM's own extension and Medit ASCII shares it: a file opening with
`MeshVersionFormatted` reads as Medit, one opening with `MFEM mesh` reads as MFEM, and a bare
write goes to MFEM. `.msh` is Gmsh's and ANSYS Fluent shares it the same way: `$MeshFormat`
reads as Gmsh, a parenthesised section reads as Fluent, and a bare write goes to Gmsh.

`.vtm`, `.pvtu`, `.pvts`, `.pvti`, `.pvtp` and `.pvtr` are registered too, but they hold no
geometry - only references to sub-files. Reading one raises `UnsupportedFormatError` rather
than failing with a parse error further in; the several live in the helper:

```python
from polyxios import helper

whole = helper.read_multiblock("case.pvtu")  # every piece, merged
blocks = helper.read_blocks("case.vtm")  # one PolyData per sub-file
```

`examples/read_parallel_vtk.py` walks through what they do. Writing an index file is not
supported.

**41 formats supported** across the 53 extensions in the tables, plus `.plt`, which
is recognised but not read - more coming via the plugin system.

---

## Transforms

The [transforms reference](https://polyxios.org/stable/transforms.html) lists every transform and its options.

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
no fork required, no pull request needed. The [plugin guide](https://polyxios.org/stable/plugins.html)
walks through the whole process.

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

See the [contributor guide](https://polyxios.org/stable/contributing.html) for commit message
conventions and the review process.
For the full release workflow see the [development guide](https://polyxios.org/stable/development.html).

---

## Why polyxios?

- **No silent data corruption** - large mesh indices raise an error instead of truncating
- **All element groups preserved** - a face belonging to multiple tags stays in all of them
- **Safe on untrusted files** - header counts validated before any memory allocation
- **Memory-efficient** - `lazy=True` maps a binary file and hands back views of it where the layout allows, copies nothing where it does not, and says which
- **Paths, buffers and gzip alike** - one API over files, streams and `.gz`
- **Works without a compiler** - pure Python fallbacks included; Cython hot-paths optional

---

## Links

- Website and documentation: [polyxios.org](https://polyxios.org)
- Source code: [github.com/fury-gl/polyxios](https://github.com/fury-gl/polyxios)
- Bug reports and feature requests: [issue tracker](https://github.com/fury-gl/polyxios/issues)
- Releases: [PyPI](https://pypi.org/project/polyxios/) and [conda-forge](https://anaconda.org/conda-forge/polyxios)
- Changelog: [polyxios.org/stable/changelog.html](https://polyxios.org/stable/changelog.html)

---

## License

See [LICENSE](LICENSE).
