"""Compare polyxios' read time and peak memory with other Python readers.

Usage::

    python compare.py make DIR              # write the test files (~3 GB)
    python compare.py run DIR -o out.json   # time every reader on them
    python compare.py table out.json        # print the results as RST

Every read runs in a fresh Python process that imports only the library it
measures (one ``read_*.py`` script each), ``--repeat`` times; the median
time and the median peak memory are kept. A library that is not installed
is skipped, so the comparison runs with whatever subset is at hand.
Needs ``resource``, hence Linux or macOS. See ``docs/benchmarks.rst``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import numpy as np

import polyxios

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True, kw_only=True)
class Reader:
    """One way of reading a file: a script, its flags, a label."""

    label: str
    script: str
    distribution: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Dataset:
    """A test file and the readers that can read it."""

    key: str
    file: str
    title: str
    readers: tuple[Reader, ...]


POLYXIOS = Reader(label="polyxios", script="read_polyxios.py", distribution="polyxios")
POLYXIOS_LAZY = Reader(
    label="polyxios (lazy)",
    script="read_polyxios.py",
    distribution="polyxios",
    flags=("--lazy",),
)
MESHIO = Reader(label="meshio", script="read_meshio.py", distribution="meshio")
TRIMESH = Reader(label="trimesh", script="read_trimesh.py", distribution="trimesh")
PYVISTA = Reader(label="PyVista", script="read_pyvista.py", distribution="pyvista")
OPEN3D = Reader(label="Open3D", script="read_open3d.py", distribution="open3d")
OPEN3D_POINTS = Reader(
    label="Open3D", script="read_open3d.py", distribution="open3d", flags=("--points",)
)
LASPY = Reader(label="laspy", script="read_laspy.py", distribution="laspy")

SURFACE_READERS = (POLYXIOS, POLYXIOS_LAZY, MESHIO, TRIMESH, PYVISTA, OPEN3D)

DATASETS = (
    Dataset(
        key="stl",
        file="surface.stl",
        title="Binary STL, 20M triangles",
        readers=SURFACE_READERS,
    ),
    Dataset(
        key="ply",
        file="surface.ply",
        title="Binary PLY, 10M vertices / 20M triangles",
        readers=SURFACE_READERS,
    ),
    Dataset(
        key="vtu",
        file="volume.vtu",
        title="VTU (raw appended), 10.2M points / 10M hexahedra",
        readers=(POLYXIOS, POLYXIOS_LAZY, MESHIO, PYVISTA),
    ),
    Dataset(
        key="ply-points",
        file="cloud.ply",
        title="Binary PLY point cloud, 10M points",
        readers=(POLYXIOS, POLYXIOS_LAZY, MESHIO, TRIMESH, PYVISTA, OPEN3D_POINTS),
    ),
    Dataset(
        key="las",
        file="cloud.las",
        title="LAS 1.2 point format 0, 10M points",
        readers=(POLYXIOS, POLYXIOS_LAZY, LASPY),
    ),
    Dataset(
        key="obj",
        file="surface.obj",
        title="ASCII OBJ, 1M vertices / 2M triangles",
        readers=(POLYXIOS, MESHIO, TRIMESH, PYVISTA, OPEN3D),
    ),
)


def surface(*, n: int) -> polyxios.PolyData:
    """Build an ``n`` by ``n`` height field split into triangles.

    Parameters
    ----------
    n
        Vertices along each side.

    Returns
    -------
    PolyData
        ``n**2`` vertices and ``2 * (n - 1)**2`` triangles.
    """
    y, x = np.divmod(np.arange(n * n, dtype=np.float64), n)
    z = 0.1 * np.sin(x / 50) * np.cos(y / 50)
    i = np.arange(n - 1)
    a = (i[:, None] * n + i[None, :]).ravel()
    b, c, d = a + 1, a + n, a + n + 1
    tris = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return polyxios.make_polydata(np.column_stack([x, y, z]), [("triangle", tris)])


def volume(*, m: int) -> polyxios.PolyData:
    """Build an ``m`` cubed block of hexahedra.

    Parameters
    ----------
    m
        Cells along each side.

    Returns
    -------
    PolyData
        ``(m + 1)**3`` points and ``m**3`` hexahedra, corners in VTK order.
    """
    p = m + 1
    k, rest = np.divmod(np.arange(p**3), p * p)
    j, i = np.divmod(rest, p)
    points = np.column_stack([i, j, k]).astype(np.float64)
    c = np.arange(m)
    origins = (
        c[:, None, None] * p * p + c[None, :, None] * p + c[None, None, :]
    ).ravel()
    steps = (0, 1, p + 1, p, p * p, p * p + 1, p * p + p + 1, p * p + p)
    hexes = origins[:, None] + np.array(steps)
    return polyxios.make_polydata(points, [("hexahedron", hexes)])


def cloud(*, n: int) -> polyxios.PolyData:
    """Build ``n`` points scattered in a 100 m cube, as a scan would.

    Parameters
    ----------
    n
        Points.

    Returns
    -------
    PolyData
        The points, with no elements.
    """
    rng = np.random.default_rng(0)
    return polyxios.make_polydata(rng.uniform(0, 100, size=(n, 3)), [])


def make(*, directory: Path) -> None:
    """Write every dataset that is not already in ``directory``.

    Parameters
    ----------
    directory
        Where the files go; created when missing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    writers = {
        "surface.stl": lambda: polyxios.write(
            surface(n=3163), directory / "surface.stl"
        ),
        "surface.ply": lambda: polyxios.write(
            surface(n=3163), directory / "surface.ply"
        ),
        "volume.vtu": lambda: polyxios.write(
            volume(m=216), directory / "volume.vtu", appended=True
        ),
        "cloud.ply": lambda: polyxios.write(
            cloud(n=10_000_000), directory / "cloud.ply"
        ),
        "cloud.las": lambda: polyxios.write(
            cloud(n=10_000_000), directory / "cloud.las"
        ),
        "surface.obj": lambda: polyxios.write(
            surface(n=1000), directory / "surface.obj"
        ),
    }
    for name, write in writers.items():
        target = directory / name
        if target.exists():
            print(f"{name}: kept")
            continue
        write()
        print(f"{name}: {target.stat().st_size / 2**20:,.0f} MiB")


def warm(*, path: Path) -> None:
    """Read ``path`` once so every reader finds it in the page cache.

    Without it the first reader of each file pays for the disk and the
    others do not, and the first reader is always polyxios.

    Parameters
    ----------
    path
        The file to read.
    """
    with path.open("rb") as fh:
        while fh.read(1 << 24):
            pass


def read_once(*, reader: Reader, path: Path, timeout: float) -> dict:
    """Run one reader on one file in a fresh process.

    Parameters
    ----------
    reader
        The reader to run.
    path
        The file to read.
    timeout
        Seconds before the read is abandoned.

    Returns
    -------
    dict
        The JSON line the reader printed, or ``{"error": ...}``.
    """
    command = [sys.executable, str(HERE / reader.script), str(path), *reader.flags]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    if done.returncode:
        last = (done.stderr.strip().splitlines() or ["no output"])[-1]
        return {"error": last}
    return json.loads(done.stdout.strip().splitlines()[-1])


def installed(*, distribution: str) -> str | None:
    """Return the installed version of ``distribution``, or None.

    Parameters
    ----------
    distribution
        The name pip knows it by.

    Returns
    -------
    str or None
        Its version.
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(*, directory: Path, repeat: int, timeout: float, only: list[str]) -> dict:
    """Time every reader on every dataset and return the medians.

    Parameters
    ----------
    directory
        Where :func:`make` wrote the files.
    repeat
        Fresh processes per reader and file.
    timeout
        Seconds before one read is abandoned.
    only
        Dataset keys to run; all of them when empty.

    Returns
    -------
    dict
        Machine, versions and one entry per dataset and reader.
    """
    versions = {}
    results = []
    for dataset in DATASETS:
        if only and dataset.key not in only:
            continue
        path = directory / dataset.file
        warm(path=path)
        for reader in dataset.readers:
            version = installed(distribution=reader.distribution)
            if version is None:
                print(f"{dataset.key:<11} {reader.label:<16} not installed")
                continue
            versions[reader.distribution] = version
            runs = [read_once(reader=reader, path=path, timeout=timeout)]
            if "error" not in runs[0]:
                runs += [
                    read_once(reader=reader, path=path, timeout=timeout)
                    for _ in range(repeat - 1)
                ]
            entry = {"dataset": dataset.key, "reader": reader.label}
            if "error" in runs[0]:
                entry["error"] = runs[0]["error"]
                print(f"{dataset.key:<11} {reader.label:<16} {entry['error']}")
            else:
                for field in ("seconds", "peak_rss_mb", "import_rss_mb"):
                    entry[field] = statistics.median(r[field] for r in runs)
                entry["n_points"] = runs[0]["n_points"]
                entry["n_cells"] = runs[0]["n_cells"]
                print(
                    f"{dataset.key:<11} {reader.label:<16} {entry['seconds']:8.2f} s"
                    f" {entry['peak_rss_mb']:8,.0f} MiB"
                    f"  ({entry['n_points']:,} points, {entry['n_cells']:,} cells)"
                )
            results.append(entry)
    return {
        "date": dt.date.today().isoformat(),
        "machine": f"{platform.system()} {platform.machine()}, {platform.processor()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeat": repeat,
        "versions": versions,
        "files": {
            d.key: (directory / d.file).stat().st_size
            for d in DATASETS
            if (directory / d.file).exists()
        },
        "results": results,
    }


def table(*, results: dict) -> str:
    """Render the results as one reStructuredText table per dataset.

    Parameters
    ----------
    results
        What :func:`run` returned.

    Returns
    -------
    str
        The tables. Times and memory are relative to the eager polyxios
        read of the same file, so each row says how far it is from it.
    """
    out = []
    for dataset in DATASETS:
        rows = [r for r in results["results"] if r["dataset"] == dataset.key]
        if not rows:
            continue
        base = next(
            (r for r in rows if r["reader"] == "polyxios" and "error" not in r), None
        )
        size = results["files"].get(dataset.key, 0) / 2**20
        out += [
            f".. list-table:: {dataset.title} ({size:,.0f} MiB)",
            "   :header-rows: 1",
            "   :widths: 30 20 20 30",
            "",
            "   * - Reader",
            "     - Read time",
            "     - Peak memory",
            "     - Points / cells read",
        ]
        for r in sorted(rows, key=lambda r: r.get("seconds", float("inf"))):
            version = results["versions"].get(_distribution(r["reader"]), "")
            out.append(f"   * - {r['reader']} {version}")
            if "error" in r:
                out += ["     - failed", "     - ", f"     - {r['error']}"]
                continue
            ratio = ""
            if base is not None and r is not base:
                ratio = f" ({r['seconds'] / base['seconds']:.1f}x)"
            out += [
                f"     - {r['seconds']:.2f} s{ratio}",
                f"     - {r['peak_rss_mb'] / 1024:.2f} GiB",
                f"     - {r['n_points']:,} / {r['n_cells']:,}",
            ]
        out.append("")
    return "\n".join(out)


def _distribution(label: str) -> str:
    """Return the distribution behind a reader label."""
    readers = {r.label: r.distribution for d in DATASETS for r in d.readers}
    return readers.get(label, "")


def main() -> None:
    """Parse the command line and run one of make, run or table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make_cmd = sub.add_parser("make", help="write the test files")
    make_cmd.add_argument("directory", type=Path)
    run_cmd = sub.add_parser("run", help="time every reader")
    run_cmd.add_argument("directory", type=Path)
    run_cmd.add_argument("-o", "--output", type=Path, required=True)
    run_cmd.add_argument("--repeat", type=int, default=3)
    run_cmd.add_argument("--timeout", type=float, default=900)
    run_cmd.add_argument("--only", nargs="*", default=[])
    table_cmd = sub.add_parser("table", help="print results as RST")
    table_cmd.add_argument("results", type=Path)
    args = parser.parse_args()

    if args.command == "make":
        make(directory=args.directory)
    elif args.command == "run":
        results = run(
            directory=args.directory,
            repeat=args.repeat,
            timeout=args.timeout,
            only=args.only,
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    else:
        print(table(results=json.loads(args.results.read_text())))


if __name__ == "__main__":
    main()
