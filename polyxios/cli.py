import argparse
import logging
import os
import sys
import time

import polyxios

# The fetcher is used through its module rather than by importing its names, so
# that POLYXIOS_HOME and the catalog loader resolve to the live module state
# (tests and callers only ever have to patch one place).
from polyxios import fetcher
from polyxios.helper import read_polydata, resolve_path, visualize_mesh

logger = logging.getLogger("polyxios")


class _Formatter(logging.Formatter):
    """Custom log formatter to print raw messages for INFO level, and prefixed levels for warning/error."""

    def format(self, record):
        if record.levelno == logging.INFO:
            message = record.getMessage()
        else:
            message = f"{record.levelname}: {record.getMessage()}"
        # The base implementation appends the traceback; this one replaces it
        # wholesale, so exc_info has to be honoured explicitly or --verbose
        # would report no traceback at all.
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return message


def _setup_logging(*, verbose: bool = False):
    """Configure stream handlers for CLI logging: stdout for INFO, stderr for WARNING+.

    Parameters
    ----------
    verbose : bool, optional
        Emit DEBUG records and attach tracebacks to reported failures.
    """
    logger.handlers.clear()

    class InfoFilter(logging.Filter):
        def filter(self, record):
            return record.levelno <= logging.INFO

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(_Formatter())
    stdout_handler.addFilter(InfoFilter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_Formatter())
    stderr_handler.setLevel(logging.WARNING)
    logger.addHandler(stderr_handler)

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False


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
        _, ext = os.path.splitext(args.filename.lower())
        if not ext:
            ext_clean = args.filename.lower().lstrip(".")
            paths = fetcher.fetch_by_extension(ext_clean, overwrite=args.overwrite)
            logger.info(f"Successfully fetched {len(paths)} file(s):")
            for path in paths:
                logger.info(f"  {path}")
        else:
            path = fetcher.fetch(args.filename, overwrite=args.overwrite)
            logger.info(f"Successfully fetched to: {path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to fetch model: {e}", exc_info=args.verbose)
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
    if os.path.exists(args.output_file) and not args.force:
        logger.error(
            f"Output file '{args.output_file}' already exists. Use --force to overwrite."
        )
        return 1
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
        logger.error(f"Failed to convert model: {e}", exc_info=args.verbose)
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
        path = resolve_path(args.filename)

        logger.info(f"Reading {path} ...")
        start_time = time.perf_counter()
        polydata = read_polydata(path)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Loaded in {elapsed:.4f} seconds")

        logger.info(
            f"  {len(polydata.vertices)} vertices | "
            f"{len(polydata.element_types)} elements | "
            f"vertex attrs: {list(polydata.vertex_attrs) or 'none'}"
        )

        if len(polydata.vertices) == 0:
            logger.info("  No geometry (FIELD data) - skipping window.")
            return 0

        visualize_mesh(polydata, lines=args.lines, points=args.points)
        return 0
    except ImportError as e:
        logger.error(str(e), exc_info=args.verbose)
        return 1
    except Exception as e:
        logger.error(f"Failed to visualize model: {e}", exc_info=args.verbose)
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

        pkg_to_exts = {}
        for ext in polyxios.supported_extensions():
            ext_name = ext.lstrip(".")
            # An extension several formats share resolves to a dispatcher that
            # knows its candidates; grouping it under one of them would claim
            # an ownership no codec has. Everything else groups by package.
            candidates = getattr(polyxios._REGISTRY.get(ext), "candidates", ())
            pkg = (
                "/".join(candidates)
                if candidates
                else fetcher._EXT_TO_PACKAGE.get(ext_name, ext_name)
            )
            pkg_to_exts.setdefault(pkg, []).append(f".{ext_name}")

        for pkg, exts in sorted(pkg_to_exts.items()):
            logger.info(f"  {pkg} ({', '.join(exts)})")
        return 0

    if args.extensions:
        logger.info("File formats available in remote catalog:")

        package_to_exts = {}
        catalog_failed = False
        try:
            catalog = fetcher._load_models_catalog()
            ext_to_package = dict(fetcher._EXT_TO_PACKAGE)
            ext_to_package.update(catalog.get("ext_to_package", {}))
            for ext_name, pkg in ext_to_package.items():
                package_to_exts.setdefault(pkg, []).append(ext_name.lstrip("."))

            # A package the catalog does not map to any extension is usually
            # named after its own extension, but only a registered codec proves
            # it. Anything else is listed without a fabricated extension.
            codec_exts = {e.lstrip(".") for e in polyxios.supported_extensions()}
            for pkg in catalog.get("formats", {}):
                if pkg not in package_to_exts:
                    package_to_exts[pkg] = [pkg] if pkg in codec_exts else []
        except Exception as e:
            catalog_failed = True
            logger.warning(
                f"Catalog retrieval failed ({e}); listing built-in formats only.",
                exc_info=args.verbose,
            )

        if not package_to_exts:
            for ext_name, pkg in fetcher._EXT_TO_PACKAGE.items():
                package_to_exts.setdefault(pkg, []).append(ext_name.lstrip("."))

        for pkg, exts in sorted(package_to_exts.items()):
            if not exts:
                logger.info(f"  {pkg}")
                continue
            exts_fmt = [f".{e}" for e in sorted(exts)]
            logger.info(f"  {pkg} ({', '.join(exts_fmt)})")
        return 1 if catalog_failed else 0

    if args.local:
        if args.ext:
            ext_clean = args.ext.lower().lstrip(".")

            package = fetcher.get_package_name(ext_clean)
            pkg_dir = os.path.join(fetcher.POLYXIOS_HOME, package)
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
            if os.path.exists(fetcher.POLYXIOS_HOME):
                for pkg in sorted(os.listdir(fetcher.POLYXIOS_HOME)):
                    pkg_dir = os.path.join(fetcher.POLYXIOS_HOME, pkg)
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

    try:
        files_dict = fetcher.get_fetchable_files()
    except Exception as e:
        logger.error(f"Failed to retrieve catalog: {e}", exc_info=args.verbose)
        return 1

    def _top_level(files):
        """Keep the entries that name a file directly in the package folder."""
        return sorted(f for f in files if "/" not in f and "\\" not in f)

    if args.ext:
        ext_clean = args.ext.lower().lstrip(".")
        if ext_clean not in files_dict:
            logger.error(f"No package/extension found matching '{args.ext}'.")
            return 1
        logger.info(f"Available files for fetch ({ext_clean}):")
        files = _top_level(files_dict[ext_clean])
        for f in files:
            logger.info(f"  {f}")
        if not files:
            logger.info("  (none)")
    else:
        logger.info("Available files for fetch:")
        for pkg, files in sorted(files_dict.items()):
            visible = _top_level(files)
            if not visible:
                continue
            logger.info(f"\n[{pkg}]")
            for f in visible:
                logger.info(f"  {f}")
    return 0


def main():
    """Main CLI entry point for pxios. Parses arguments and routes commands."""
    # Shared so that --verbose is accepted on either side of the subcommand.
    # SUPPRESS keeps an omitted flag out of the sub-namespace, which would
    # otherwise clobber a --verbose given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show debug logs and full tracebacks on failure",
    )

    parser = argparse.ArgumentParser(
        description="Polyxios CLI (pxios): Fetch, convert, and visualize 3D models.",
        parents=[common],
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"polyxios {polyxios.__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch a model file", parents=[common]
    )
    fetch_parser.add_argument(
        "filename",
        help="Name of the model file to fetch (e.g. armadillo.obj) or extension package (e.g. obj)",
    )
    fetch_parser.add_argument(
        "--overwrite", action="store_true", help="Force overwrite existing cached file"
    )
    fetch_parser.set_defaults(func=cmd_fetch)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert a model file to another format", parents=[common]
    )
    convert_parser.add_argument("input_file", help="Path to the input model file")
    convert_parser.add_argument("output_file", help="Path to the output model file")
    convert_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite existing output file",
    )
    convert_parser.set_defaults(func=cmd_convert)

    viz_parser = subparsers.add_parser(
        "viz",
        help="Visualize a model file via polyxios + FURY.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    viz_parser.add_argument(
        "filename",
        help=(
            "Filename to fetch and visualize (e.g. 'mesh.vtk', 'bunny.obj'), or a "
            "local path (relative or absolute)."
        ),
    )
    viz_group = viz_parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--lines",
        action="store_true",
        help="Render line/poly_line elements or surface wireframe instead of solid surface.",
    )
    viz_group.add_argument(
        "--points",
        action="store_true",
        help="Render strictly as a point cloud.",
    )
    viz_parser.set_defaults(func=cmd_viz)

    list_parser = subparsers.add_parser(
        "list", help="List all available remote or cached files", parents=[common]
    )
    list_parser.add_argument(
        "ext",
        nargs="?",
        help="Optional extension name to filter the listed files (e.g. 'obj', 'vtk')",
    )
    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--local",
        action="store_true",
        help="List locally cached files instead of remote files",
    )
    list_group.add_argument(
        "--extensions",
        "--formats",
        dest="extensions",
        action="store_true",
        help="List all formats and extensions available in the remote catalog",
    )
    list_group.add_argument(
        "--codecs",
        action="store_true",
        help="List all formats supported by active local codecs",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    # SUPPRESS leaves the attribute out entirely when the flag is never given.
    args.verbose = getattr(args, "verbose", False)
    _setup_logging(verbose=args.verbose)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
