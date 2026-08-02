import hashlib
import json
import os
from os.path import expanduser, join as pjoin
import sys
import urllib.error
import urllib.request

from polyxios.exceptions import FetcherError

POLYXIOS_HOME = os.getenv("POLYXIOS_HOME", pjoin(expanduser("~"), ".polyxios"))

_MODELS_URL = (
    "https://raw.githubusercontent.com/fury-gl/polyxios-data/master/models.json"
)

_EXT_TO_PACKAGE = {
    "inp": "abaqus",
    "fem": "nastran",
    "vol": "netgen",
    "f3grid": "flac3d",
    "tec": "tecplot",
    "meshb": "medit",
    "xml": "dolfin",
    "msh": "gmsh",
}


def get_package_name(ext: str) -> str:
    """Resolve a file extension to its corresponding package name.

    Parameters
    ----------
    ext : str
        The file extension (e.g., '.inp' or 'inp').

    Returns
    -------
    str
        The resolved package name.
    """
    ext_clean = ext.lower().lstrip(".")
    try:
        catalog = _load_models_catalog()
        ext_to_package = catalog.get("ext_to_package", {})
        if ext_clean in ext_to_package:
            return ext_to_package[ext_clean]
    except Exception:
        pass
    return _EXT_TO_PACKAGE.get(ext_clean, ext_clean)


def _verify_sha256(filepath: str, expected_sha: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_sha.lower()


def _show_progress(filename: str, downloaded: int, total: int) -> None:
    """Standard-library progress bar emulation using terminal carriage returns."""
    if total <= 0:
        sys.stdout.write(f"\rFetching {filename}: {downloaded / (1024 * 1024):.2f} MB")
    else:
        percent = (downloaded / total) * 100
        bar_length = 30
        filled = int(bar_length * downloaded // total)
        bar = "#" * filled + "-" * (bar_length - filled)
        sys.stdout.write(f"\rFetching {filename}: [{bar}] {percent:.1f}%")
    sys.stdout.flush()


def _load_models_catalog() -> dict:
    """Load the models catalog from raw GitHub or local cache."""
    local_path = pjoin(POLYXIOS_HOME, "models.json")

    # Try to download and cache the latest catalog
    try:
        req = urllib.request.Request(
            _MODELS_URL, headers={"User-Agent": "polyxios-fetcher"}
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, dict) and "formats" in data:
                os.makedirs(POLYXIOS_HOME, exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return data
    except Exception:
        pass

    # If download fails, check if we have a locally cached models.json
    if os.path.exists(local_path):
        try:
            with open(local_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "formats" in data:
                    return data
                else:
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
        except Exception:
            try:
                os.remove(local_path)
            except Exception:
                pass

    raise FetcherError(
        "Could not retrieve models catalog from remote repository or local cache."
    )


def get_fetchable_files() -> dict[str, list[str]]:
    """Return a dictionary of all fetchable packages and their available files.

    Returns
    -------
    dict of str to list of str
        Mapping of package/extension name to list of available filenames.

    Raises
    ------
    FetcherError
        If the catalog could not be retrieved from both remote and cache.
    """
    catalog = _load_models_catalog()
    formats = catalog.get("formats", {})
    return {fmt: sorted(files.keys()) for fmt, files in formats.items()}


def fetch(filename: str, overwrite: bool = False) -> str:
    """Resolve, download, and track local path for any Polyxios test asset.

    Parameters
    ----------
    filename : str
        The name of the file to fetch (e.g., 'stanford-bunny.obj').
    overwrite : bool, optional
        Force re-download of the asset even if it exists locally.

    Returns
    -------
    str
        The absolute local path to the fetched file.

    Raises
    ------
    FetcherError
        If filename is invalid or is not found in the package.
    """
    filename_lower = filename.lower()

    _, ext = os.path.splitext(filename_lower)
    if not ext:
        raise FetcherError(
            f"Cannot resolve target folder: filename '{filename}' has no extension."
        )
    ext_clean = ext[1:]

    package = get_package_name(ext_clean)
    catalog = _load_models_catalog()

    file_info = catalog.get("formats", {}).get(package, {}).get(filename)
    if not file_info:
        raise FetcherError(
            f"Asset '{filename}' was not found in the catalog under package '{package}'."
        )

    target_dir = pjoin(POLYXIOS_HOME, package)
    target_path = pjoin(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        if _verify_sha256(target_path, file_info["sha256"]):
            return target_path

    os.makedirs(target_dir, exist_ok=True)
    temp_path = pjoin(target_dir, f".temp_{filename}")

    try:
        sys.stdout.write(
            f"Fetching '{filename}'... Please wait, it might take some time to download. "
            "Do not close or cancel this process.\n"
        )
        sys.stdout.flush()
        req = urllib.request.Request(
            file_info["url"], headers={"User-Agent": "polyxios-fetcher"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _show_progress(filename, downloaded, total_size)

            sys.stdout.write("\n")
            sys.stdout.flush()

        if not _verify_sha256(temp_path, file_info["sha256"]):
            raise FetcherError(
                f"Integrity verification failed for {filename}. Checksum mismatch."
            )

        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)

    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        sys.stdout.flush()
        if e.code == 404:
            raise FetcherError(
                f"Asset file '{filename}' was not found on remote server."
            ) from e
        raise FetcherError(
            f"HTTP error occurred while downloading asset: {e.code} {e.reason}"
        ) from e
    except Exception as e:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise FetcherError(f"Failed to download asset '{filename}': {e}") from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return target_path


def fetch_by_extension(ext: str, overwrite: bool = False) -> list[str]:
    """
    Discover and download all remote assets matching a specific file extension.

    Parameters
    ----------
    ext : str
        The extension to query (e.g., '.obj' or 'obj').
    overwrite : bool, optional
        Force re-download of all discovered assets.

    Returns
    -------
    list of str
        The absolute local paths to all fetched files.
    """
    ext_clean = ext.lower().lstrip(".")
    if not ext_clean:
        raise FetcherError("Invalid extension format provided.")

    package = get_package_name(ext_clean)
    catalog = _load_models_catalog()

    format_files = catalog.get("formats", {}).get(package, {})
    if not format_files:
        raise FetcherError(f"No assets found for extension/format '{ext_clean}'.")

    local_files = []
    for filename in sorted(format_files.keys()):
        local_path = fetch(filename, overwrite=overwrite)
        local_files.append(local_path)

    return local_files
