# Read benchmarks

`compare.py` measures how long polyxios takes to read a large file, and how
much memory it peaks at, next to the other Python libraries that read the
same format.

## Running it

Use an environment of its own, so that the other readers never become a
dependency of polyxios:

```bash
python -m venv bench && . bench/bin/activate
pip install . -r benchmarks/requirements.txt   # polyxios as built from this checkout

python benchmarks/compare.py make /tmp/polyxios-bench          # ~2.5 GB of files
python benchmarks/compare.py run /tmp/polyxios-bench -o results.json
python benchmarks/compare.py table results.json                # reStructuredText tables
```

`run --only stl las` limits the run to some datasets, `--repeat` sets the
number of fresh processes per reading (3 by default) and `--timeout` the
seconds before one is abandoned. A reader that is not installed is skipped.
It needs the `resource` module, so Linux or macOS.

## What is measured

| Dataset | File | Readers |
|---|---|---|
| `stl` | binary STL, 20M triangles over a 10M-vertex height field | polyxios, meshio, trimesh, PyVista, Open3D |
| `ply` | binary PLY, the same surface with shared vertices | same |
| `vtu` | VTU with raw appended arrays, 10.2M points / 10M hexahedra | polyxios, meshio, PyVista |
| `ply-points` | binary PLY, 10M points, no faces | polyxios, meshio, trimesh, PyVista, Open3D |
| `las` | LAS 1.2, point format 0, 10M points | polyxios, laspy |
| `obj` | ASCII OBJ, 1M vertices / 2M triangles | polyxios, meshio, trimesh, PyVista, Open3D |

polyxios is also run with `lazy=True` on every binary dataset.

- **One process per reading.** Each `read_*.py` script imports only its own
  library, reads the file once and prints one JSON line. Nothing one
  library allocates or caches is seen by another, and the peak resident set
  size (`ru_maxrss`) is that reader's own.
- **Warm page cache.** Each file is read once before its readers run, so no
  reader pays for the disk while the others do not.
- **What the clock covers.** The read call and one pass summing the x
  coordinates, which makes a lazy reader fetch them. The library import is
  outside the clock; its memory is reported as `import_rss_mb` and is part
  of the peak.
- **Median of `--repeat` runs**, for time and for memory.

## Reading the numbers

The readers do not all hand back the same thing, and the point and cell
counts in the output show where they differ:

- An STL lists three corners per triangle. polyxios, meshio and PyVista
  merge the repeated corners into shared vertices (10M here); trimesh with
  `process=False` and a lazy polyxios read keep all 60M; Open3D merges some.
  Merging is work the others skip, so compare like with like.
- trimesh is called with `process=False`, which skips its own vertex merge
  and degenerate-face cleanup, the closest match to what the other readers
  do.
- PyVista reports a point cloud as one vertex cell per point.
- laspy and polyxios both return coordinates with the scale and offset
  applied, as float64. laspy decodes the other point fields only when they
  are accessed; polyxios decodes them all on read.
