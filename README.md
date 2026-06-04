# polyxios

**Fast, clean mesh I/O for Python.** Read and write 3D mesh files in one line — no hidden surprises, no silent data corruption.

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
print(mesh.vertices.shape)      # (n_verts, 3)
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

## Lazy loading — work with large files without filling RAM

For large meshes (gigabytes of binary data), pass `lazy=True`. polyxios
memory-maps the file and only loads the pages you actually touch — the rest
stays on disk until needed.

```python
# File is opened but data is not loaded into RAM yet
mesh = px.read("huge_brain.vtk", lazy=True)

# Only the vertices are pulled from disk here
first_vertex = mesh.vertices[0]

# Element connectivity is still on disk until you access it
```

Lazy loading is supported for binary `.vtk` and `.ply` files. ASCII formats
load eagerly (the whole file must be parsed to extract values).

---

## Supported formats

| Format | Extension | Read | Write | Lazy load |
|--------|-----------|------|-------|-----------|
| VTK Legacy | `.vtk` | ✓ | ✓ | binary only |
| VTK RectilinearGrid | `.vtr` | ✓ | ✓ | — |
| VTK PolyData | `.vtp` | ✓ | ✓ | — |
| Wavefront OBJ | `.obj` | ✓ | ✓ | — |
| Stanford PLY | `.ply` | ✓ | ✓ | binary only |

**5 formats supported** — more coming via the plugin system.

---

## Transforms

```python
from polyxios.transforms import pipeline, merge, filter_element_type, remove_orphan_vertices

# Compose transforms into a single function
clean = pipeline(
    filter_element_type(keep="triangle"),
    remove_orphan_vertices,
)
result = clean(mesh)

# Merge two meshes into one
combined = merge(mesh_a, mesh_b)
```

---

## Add your own format

Any third-party package can teach polyxios to read and write a new format —
no fork required, no pull request needed.

**Step 1 — write a codec** (two functions, nothing more):

```python
# mypackage/stl_codec.py
from polyxios._registry import Codec
from polyxios._types import PolyData

def read(path, *, lazy=False) -> PolyData:
    ...

def write(poly: PolyData, path, **opts) -> None:
    ...

def register():
    return ".stl", Codec(read, write)
```

**Step 2 — declare an entry point** in your `pyproject.toml`:

```toml
[project.entry-points."polyxios.codecs"]
stl = "mypackage.stl_codec:register"
```

After `pip install mypackage`, polyxios picks up `.stl` automatically —
no configuration, no restart needed:

```python
mesh = px.read("model.stl")   # works out of the box
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

See [`docs/contributing.rst`](docs/contributing.rst) for commit message
conventions and the full contributor guide.

---

## Why polyxios?

- **No silent data corruption** — large mesh indices raise an error instead of truncating
- **All element groups preserved** — a face belonging to multiple tags stays in all of them
- **Safe on untrusted files** — header counts validated before any memory allocation
- **Memory-efficient** — lazy mmap loading for large binary files
- **Works without a compiler** — pure Python fallbacks included; Cython hot-paths optional

---

## License

See [LICENSE](LICENSE).
