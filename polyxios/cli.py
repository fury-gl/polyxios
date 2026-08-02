import argparse
import logging
import os
from pathlib import Path
import sys
import time

import polyxios
from polyxios.fetcher import POLYXIOS_HOME, fetch, fetch_by_extension
from polyxios.helper import read_polydata
import polyxios.transforms as transforms

logger = logging.getLogger("polyxios.cli")


class _Formatter(logging.Formatter):
    """Custom log formatter to print raw messages for INFO level, and prefixed levels for warning/error."""

    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"


def _setup_logging():
    """Configure stream handler for CLI logging to stdout."""
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Formatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_available_extensions() -> list[str]:
    """Return the list of all registered codec extensions (without dots).

    Returns
    -------
    list of str
        Supported file extensions like ['obj', 'ply', 'vtk', ...].
    """
    return sorted(ext.lstrip(".") for ext in polyxios._REGISTRY.keys())


def cmd_fetch(args) -> int:
    """Fetch and cache a model file from the remote data repository.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing `filename` and `overwrite`.

    Returns
    -------
    int
        Exit status code (0 for success, 1 for failure).
    """
    try:
        logger.info(f"Fetching '{args.filename}'...")
        _, ext = os.path.splitext(args.filename.lower())
        if not ext:
            ext_clean = args.filename.lower().lstrip(".")
            fetch_by_extension(ext_clean, overwrite=args.overwrite)
            from polyxios.fetcher import get_package_name

            package = get_package_name(ext_clean)
            path = os.path.join(POLYXIOS_HOME, package)
            logger.info(f"Successfully fetched package to: {path}")
        else:
            path = fetch(args.filename, overwrite=args.overwrite)
            logger.info(f"Successfully fetched to: {path}")
        return 0
    except Exception as e:
        logger.error(f"Error fetching model: {e}")
        return 1


def cmd_convert(args) -> int:
    """Convert a model file from one format to another and log performance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing `input_file` and `output_file`.

    Returns
    -------
    int
        Exit status code (0 for success, 1 for failure).
    """
    try:
        logger.info(f"Reading '{args.input_file}'...")
        start_read_time = time.perf_counter()
        polydata = read_polydata(args.input_file)
        elapsed = time.perf_counter() - start_read_time
        logger.info(f"Read in {elapsed:.4f} seconds")

        logger.info(f"Writing to '{args.output_file}'...")
        start_write_time = time.perf_counter()
        polyxios.write(polydata, args.output_file)
        elapsed = time.perf_counter() - start_write_time
        logger.info(f"Wrote in {elapsed:.4f} seconds")

        logger.info("Conversion successful.")
        return 0
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        return 1


def cmd_viz(args) -> int:
    """Visualize a 3D model using the FURY renderer library.

    Parses and renders the geometry using suitable actors (surface, line,
    or point cloud).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing visualization and filtering options.

    Returns
    -------
    int
        Exit status code (0 for success, 1 for failure).
    """
    try:
        if not args.filename:
            logger.error("Error: filename is required for visualization.")
            return 1

        filename = args.filename
        p = Path(filename)
        if p.exists() or p.parent != Path("."):
            path = str(p.resolve())
        else:
            try:
                path = fetch(filename)
            except Exception as e:
                logger.error(f"Could not fetch '{filename}': {e}")
                return 1

        logger.info(f"Reading {path} ...")
        start_time = time.perf_counter()
        polydata = read_polydata(path)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Loaded in {elapsed:.4f} seconds")

        try:
            from fury import actor, window
        except ImportError:
            logger.error(
                "FURY is not installed. Please install it to visualize models:\n"
                "  pip install fury"
            )
            return 1

        logger.info(
            f"  {len(polydata.vertices)} vertices | "
            f"{len(polydata.element_types)} elements | "
            f"vertex attrs: {list(polydata.vertex_attrs) or 'none'}"
        )

        if len(polydata.vertices) == 0:
            logger.info("  No geometry (FIELD data) - skipping window.")
            return 0

        actors = []
        if args.points:
            logger.info("  Rendering strictly as point cloud per --points request.")
            actors.append(actor.point(polydata.vertices, colors=(0.9, 0.9, 0.9)))
        elif args.lines:
            lines_list = polydata.lines
            if not lines_list or len(lines_list) == 0:
                import numpy as np

                lines_list = [np.arange(len(polydata.vertices), dtype=np.int32)]
            lines_coords = [
                polydata.vertices[idx].astype("float64") for idx in lines_list
            ]
            logger.info(
                f"  Rendering {len(lines_coords)} line segment(s) with actor.line."
            )
            actors.append(actor.line(lines_coords, colors=(0.2, 0.8, 0.2)))
        else:
            faces = polydata.faces
            if faces is None:
                surface = transforms.extract_surface(polydata)
                faces = surface.faces
            if faces is not None and len(faces) > 0:
                colors = transforms.vertex_colors(polydata)
                actors.append(
                    actor.surface(
                        vertices=polydata.vertices,
                        faces=faces,
                        colors=colors if colors is not None else (0.8, 0.7, 0.6),
                    )
                )
            else:
                logger.info("  No renderable geometry - rendering as point cloud.")
                actors.append(actor.point(polydata.vertices, colors=(0.9, 0.9, 0.9)))

        window.show(actors)
        return 0
    except Exception as e:
        logger.error(f"Error visualizing model: {e}")
        return 1


def cmd_list(args) -> int:
    """List all available remote or cached files grouped by package.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    int
        Exit status code (0 for success, 1 for failure).
    """
    if args.codecs:
        logger.info("File formats supported by polyxios codecs:")
        import polyxios
        from polyxios.fetcher import _EXT_TO_PACKAGE

        all_exts = sorted(ext.lstrip(".") for ext in polyxios._REGISTRY.keys())
        pkg_to_exts = {}
        for ext_name in all_exts:
            pkg = _EXT_TO_PACKAGE.get(ext_name, ext_name)
            pkg_to_exts.setdefault(pkg, []).append(f".{ext_name}")

        for pkg, exts in sorted(pkg_to_exts.items()):
            logger.info(f"  {pkg} ({', '.join(exts)})")
        return 0

    if args.extensions:
        logger.info("File formats available in remote catalog:")
        from polyxios.fetcher import _EXT_TO_PACKAGE, _load_models_catalog

        package_to_exts = {}
        try:
            catalog = _load_models_catalog()
            ext_to_package = catalog.get("ext_to_package", {})
            for ext_name, pkg in ext_to_package.items():
                package_to_exts.setdefault(pkg, []).append(ext_name.lstrip("."))

            catalog_formats = catalog.get("formats", {})
            for pkg in catalog_formats.keys():
                if pkg not in package_to_exts:
                    package_to_exts[pkg] = [pkg]
        except Exception:
            pass

        if not package_to_exts:
            for ext_name, pkg in _EXT_TO_PACKAGE.items():
                package_to_exts.setdefault(pkg, []).append(ext_name.lstrip("."))

        for pkg, exts in sorted(package_to_exts.items()):
            exts_fmt = [f".{e}" for e in sorted(exts)]
            logger.info(f"  {pkg} ({', '.join(exts_fmt)})")
        return 0

    if args.local:
        if args.ext:
            ext_clean = args.ext.lower().lstrip(".")
            from polyxios.fetcher import get_package_name

            package = get_package_name(ext_clean)
            pkg_dir = os.path.join(POLYXIOS_HOME, package)
            found_any = False
            if os.path.exists(pkg_dir) and os.path.isdir(pkg_dir):
                files = sorted(
                    f
                    for f in os.listdir(pkg_dir)
                    if os.path.isfile(os.path.join(pkg_dir, f))
                    and not f.startswith(".")
                    and f.lower().endswith(f".{ext_clean}")
                )
                if files:
                    logger.info(f"Cached .{ext_clean} files:")
                    for f in files:
                        logger.info(f"  {os.path.join(pkg_dir, f)}")
                    found_any = True
            if not found_any:
                logger.info(f"No local .{ext_clean} files cached.")
        else:
            logger.info("Cached files:")
            found_any = False
            if os.path.exists(POLYXIOS_HOME):
                for pkg in sorted(os.listdir(POLYXIOS_HOME)):
                    pkg_dir = os.path.join(POLYXIOS_HOME, pkg)
                    if os.path.isdir(pkg_dir):
                        files = sorted(
                            f
                            for f in os.listdir(pkg_dir)
                            if os.path.isfile(os.path.join(pkg_dir, f))
                            and not f.startswith(".")
                        )
                        if files:
                            logger.info(f"[{pkg}]")
                            for f in files:
                                logger.info(f"  {os.path.join(pkg_dir, f)}")
                            found_any = True
            if not found_any:
                logger.info("No cached files found.")
        return 0

    from polyxios.fetcher import get_fetchable_files

    try:
        files_dict = get_fetchable_files()
    except Exception as e:
        logger.error(f"Error fetching catalog: {e}")
        return 1

    if args.ext:
        ext_clean = args.ext.lower().lstrip(".")
        if ext_clean not in files_dict:
            logger.error(f"No package/extension found matching '{args.ext}'.")
            return 1
        logger.info(f"Available files for fetch ({ext_clean}):")
        logger.info(f"\n[{ext_clean}]")
        for f in sorted(files_dict[ext_clean]):
            if "/" in f or "\\" in f:
                continue
            logger.info(f"  {f}")
    else:
        logger.info("Available files for fetch:")
        for pkg, files in sorted(files_dict.items()):
            logger.info(f"\n[{pkg}]")
            for f in sorted(files):
                if "/" in f or "\\" in f:
                    continue
                logger.info(f"  {f}")
    return 0


def main():
    """Main CLI entry point for pxios. Parses arguments and routes commands."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Polyxios CLI (pxios): Fetch, convert, and visualize 3D models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch a model file")
    fetch_parser.add_argument(
        "filename",
        help="Name of the model file to fetch (e.g. armadillo.obj) or extension package (e.g. obj)",
    )
    fetch_parser.add_argument(
        "--overwrite", action="store_true", help="Force overwrite existing cached file"
    )

    convert_parser = subparsers.add_parser(
        "convert", help="Convert a model file to another format"
    )
    convert_parser.add_argument("input_file", help="Path to the input model file")
    convert_parser.add_argument("output_file", help="Path to the output model file")

    viz_parser = subparsers.add_parser(
        "viz",
        help="Visualize a model file via polyxios + FURY.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    viz_parser.add_argument(
        "filename",
        help=(
            "Filename to fetch and visualize (e.g. 'mesh.vtk', 'bunny.obj'), or a "
            "local path (relative or absolute)."
        ),
    )
    viz_parser.add_argument(
        "--lines",
        action="store_true",
        help="Render line/poly_line elements with actor.line instead of surface.",
    )
    viz_parser.add_argument(
        "--points",
        action="store_true",
        help="Render strictly as a point cloud.",
    )

    list_parser = subparsers.add_parser(
        "list", help="List all available remote or cached files"
    )
    list_parser.add_argument(
        "ext",
        nargs="?",
        help="Optional extension name to filter the listed files (e.g. 'obj', 'vtk')",
    )
    list_parser.add_argument(
        "--local",
        action="store_true",
        help="List locally cached files instead of remote files",
    )
    list_parser.add_argument(
        "--extensions",
        "--formats",
        dest="extensions",
        action="store_true",
        help="List all formats and extensions available in the remote catalog",
    )
    list_parser.add_argument(
        "--codecs",
        action="store_true",
        help="List all formats supported by active local codecs",
    )

    args = parser.parse_args()

    if args.command == "fetch":
        sys.exit(cmd_fetch(args))
    elif args.command == "convert":
        sys.exit(cmd_convert(args))
    elif args.command == "viz":
        sys.exit(cmd_viz(args))
    elif args.command == "list":
        sys.exit(cmd_list(args))


if __name__ == "__main__":
    main()
