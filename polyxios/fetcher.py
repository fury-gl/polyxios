import hashlib
import json
import logging
import os
from os.path import expanduser, join as pjoin
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import zipfile

from polyxios.exceptions import FetcherError

logger = logging.getLogger("polyxios.fetcher")

POLYXIOS_HOME = os.getenv("POLYXIOS_HOME", pjoin(expanduser("~"), ".polyxios"))

# The catalog is the trust root for every download: asset checksums are read
# from it rather than pinned in this file. It is served from a mutable branch so
# newly published assets are picked up automatically; set POLYXIOS_MODELS_URL to
# an immutable raw URL (release tag or commit sha) to pin it.
_MODELS_URL = os.getenv(
    "POLYXIOS_MODELS_URL",
    "https://raw.githubusercontent.com/fury-gl/polyxios-data/master/models.json",
)

# The catalog is a few kilobytes, so a short timeout keeps offline use snappy.
# Assets can be hundreds of megabytes: the timeout applies to each socket
# operation, so it must tolerate slow links rather than the whole transfer.
_CATALOG_TIMEOUT = 5
_DOWNLOAD_TIMEOUT = 30

_EXT_TO_PACKAGE: dict[str, str] = {
    "inp": "abaqus",
    "fem": "nastran",
    "vol": "netgen",
    "f3grid": "flac3d",
    "tec": "tecplot",
    "meshb": "medit",
    "xml": "dolfin",
    "msh": "gmsh",
}


_catalog_cache = None

# (models.json path, mtime) -> ext_to_package mapping, so repeated extension
# lookups do not re-read and re-parse the catalog from disk every call.
_ext_map_cache: tuple[tuple[str, float], dict[str, str]] | None = None


def _load_ext_to_package() -> dict[str, str]:
    """Return the catalog's extension to package mapping, without any download.

    Returns
    -------
    dict of str to str
        Mapping read from the in-memory catalog when one is loaded, otherwise
        from the cached ``models.json``. Empty when neither is available.
    """
    global _ext_map_cache

    if _catalog_cache is not None:
        return _catalog_cache.get("ext_to_package", {})

    local_path = pjoin(POLYXIOS_HOME, "models.json")
    try:
        key = (local_path, os.path.getmtime(local_path))
    except OSError:
        return {}

    if _ext_map_cache is not None and _ext_map_cache[0] == key:
        return _ext_map_cache[1]

    try:
        with open(local_path, encoding="utf-8") as f:
            mapping = json.load(f).get("ext_to_package", {})
    except (OSError, ValueError) as e:
        logger.debug(f"Failed to read ext_to_package from cache: {e}", exc_info=True)
        return {}

    _ext_map_cache = (key, mapping)
    return mapping


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
    catalog_map = _load_ext_to_package()
    if ext_clean in catalog_map:
        return catalog_map[ext_clean]
    return _EXT_TO_PACKAGE.get(ext_clean, ext_clean)


def _verify_sha256(filepath: str, expected_sha: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_sha.lower()


_verified_cache = None


def _get_verified_cache() -> dict:
    global _verified_cache
    if _verified_cache is not None:
        return _verified_cache

    cache_path = pjoin(POLYXIOS_HOME, ".models_cache")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                _verified_cache = json.load(f)
                if isinstance(_verified_cache, dict):
                    return _verified_cache
        except (OSError, ValueError) as e:
            logger.debug(f"Failed to read verification cache: {e}", exc_info=True)
    _verified_cache = {}
    return _verified_cache


def _save_verified_cache(cache: dict) -> None:
    cache_path = pjoin(POLYXIOS_HOME, ".models_cache")
    try:
        os.makedirs(POLYXIOS_HOME, exist_ok=True)
        temp_cache_path = cache_path + f".{os.getpid()}.tmp"
        with open(temp_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_cache_path, cache_path)
    except OSError as e:
        logger.debug(f"Failed to save verification cache: {e}", exc_info=True)


def _verify_and_cache(filepath: str, expected_sha: str) -> bool:
    if not os.path.exists(filepath):
        return False

    try:
        stat = os.stat(filepath)
        mtime = stat.st_mtime
        size = stat.st_size
    except OSError as e:
        logger.debug(f"Failed to stat '{filepath}': {e}", exc_info=True)
        return False

    cache = _get_verified_cache()
    key = os.path.abspath(filepath)
    entry = cache.get(key)

    if (
        entry
        and entry.get("mtime") == mtime
        and entry.get("size") == size
        and entry.get("sha256") == expected_sha
    ):
        return True

    if _verify_sha256(filepath, expected_sha):
        cache[key] = {"mtime": mtime, "size": size, "sha256": expected_sha}
        _save_verified_cache(cache)
        return True

    return False


def _safe_extract_zip(zip_path: str, target_dir: str) -> None:
    """Extract an archive into a directory, rejecting members that escape it.

    Parameters
    ----------
    zip_path : str
        Path to the archive to extract.
    target_dir : str
        Directory the archive must be extracted into.

    Raises
    ------
    FetcherError
        If any member resolves outside `target_dir` (zip slip).
    """
    dest = Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            resolved = (dest / member).resolve()
            if resolved != dest and not resolved.is_relative_to(dest):
                raise FetcherError(
                    f"Refusing to extract '{member}': it resolves outside "
                    f"'{target_dir}'."
                )
        zip_ref.extractall(target_dir)


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
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    local_path = pjoin(POLYXIOS_HOME, "models.json")

    use_cached = False
    if os.path.exists(local_path):
        try:
            if time.time() - os.path.getmtime(local_path) < 86400:
                use_cached = True
        except Exception:
            pass

    if use_cached:
        try:
            with open(local_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "formats" in data:
                    _catalog_cache = data
                    return data
        except Exception as e:
            logger.debug(f"Failed to read fresh local models.json: {e}", exc_info=True)

    try:
        req = urllib.request.Request(
            _MODELS_URL, headers={"User-Agent": "polyxios-fetcher"}
        )
        with urllib.request.urlopen(req, timeout=_CATALOG_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, dict) and "formats" in data:
                os.makedirs(POLYXIOS_HOME, exist_ok=True)
                temp_local_path = local_path + f".{os.getpid()}.tmp"
                with open(temp_local_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(temp_local_path, local_path)
                _catalog_cache = data
                return data
    except Exception as e:
        logger.debug(
            f"Catalog download failed, falling back to local cache: {e}", exc_info=True
        )

    if os.path.exists(local_path):
        try:
            with open(local_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "formats" in data:
                    try:
                        os.utime(local_path, None)
                    except Exception:
                        pass
                    _catalog_cache = data
                    return data
                else:
                    _catalog_cache = None
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Failed to read local models.json cache: {e}", exc_info=True)
            _catalog_cache = None
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


def fetch(filename: str, overwrite: bool = False, *, package: str | None = None) -> str:
    """Resolve, download, and track local path for any Polyxios test asset.

    Parameters
    ----------
    filename : str
        The name of the file to fetch (e.g., 'stanford-bunny.obj').
    overwrite : bool, optional
        Force re-download of the asset even if it exists locally.
    package : str, optional
        Manually override the target catalog package name.

    Returns
    -------
    str
        The absolute local path to the fetched file.

    Raises
    -------
    FetcherError
        If filename is invalid or is not found in the package.
    """
    filename_lower = filename.lower()

    if package is None:
        _, ext = os.path.splitext(filename_lower)
        if not ext:
            raise FetcherError(
                f"Cannot resolve target folder: filename '{filename}' has no extension."
            )
        ext_clean = ext[1:]
        package = get_package_name(ext_clean)

    target_dir = pjoin(POLYXIOS_HOME, package)
    target_path = pjoin(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        # The cached copy is only reusable when its checksum still matches the
        # one the catalog advertises today; a re-released asset must be
        # re-downloaded even though the local file is intact.
        try:
            catalog = _load_models_catalog()
            formats_dict = catalog.get("formats", {}).get(package, {})
            match_filename = None
            for k in formats_dict.keys():
                if k.lower() == filename_lower:
                    match_filename = k
                    break
            if match_filename:
                file_info = formats_dict[match_filename]
                if (
                    "size_bytes" not in file_info
                    or os.path.getsize(target_path) == file_info["size_bytes"]
                ):
                    if _verify_and_cache(target_path, file_info["sha256"]):
                        base_name, _ = os.path.splitext(match_filename)
                        zip_filename = f"{base_name}.zip"
                        has_zip = any(
                            k.lower() == zip_filename.lower()
                            for k in formats_dict.keys()
                        )
                        if has_zip:
                            extracted_dir = pjoin(target_dir, base_name)
                            extracted_flag = pjoin(
                                target_dir, f".extracted_{zip_filename}"
                            )
                            if os.path.exists(extracted_dir) and os.path.exists(
                                extracted_flag
                            ):
                                return target_path
                        else:
                            return target_path
        except FetcherError as e:
            logger.debug(f"Catalog unavailable while reusing cache: {e}", exc_info=True)
        except (OSError, KeyError) as e:
            logger.debug(f"Cannot reuse cached '{filename}': {e}", exc_info=True)

    try:
        catalog = _load_models_catalog()
    except Exception:
        if os.path.exists(target_path):
            return target_path
        raise

    formats_dict = catalog.get("formats", {}).get(package, {})
    match_filename = None
    for k in formats_dict.keys():
        if k.lower() == filename_lower:
            match_filename = k
            break

    if not match_filename:
        raise FetcherError(
            f"Asset '{filename}' was not found in the catalog under package '{package}'."
        )

    file_info = formats_dict[match_filename]
    target_path = pjoin(target_dir, match_filename)

    url = file_info.get("url", "")
    if not url.lower().startswith("https://"):
        raise FetcherError(
            f"Refusing to download '{match_filename}': catalog URL is not https."
        )

    os.makedirs(target_dir, exist_ok=True)
    temp_path = pjoin(target_dir, f".temp_{os.getpid()}_{match_filename}")

    try:
        logger.info(
            f"Fetching '{match_filename}'... Please wait, it might take some time "
            "to download. Do not close or cancel this process."
        )
        req = urllib.request.Request(url, headers={"User-Agent": "polyxios-fetcher"})
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _show_progress(match_filename, downloaded, total_size)

            sys.stdout.write("\n")
            sys.stdout.flush()

        if not _verify_sha256(temp_path, file_info["sha256"]):
            raise FetcherError(
                f"Integrity verification failed for {match_filename}. Checksum mismatch."
            )

        os.replace(temp_path, target_path)

        try:
            stat = os.stat(target_path)
            cache = _get_verified_cache()
            key = os.path.abspath(target_path)
            cache[key] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "sha256": file_info["sha256"],
            }
            _save_verified_cache(cache)
        except OSError as e:
            logger.debug(f"Failed to record '{target_path}' as verified: {e}")

    except FetcherError:
        raise
    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        sys.stdout.flush()
        if e.code == 404:
            raise FetcherError(
                f"Asset file '{match_filename}' was not found on remote server."
            ) from e
        raise FetcherError(
            f"HTTP error occurred while downloading asset: {e.code} {e.reason}"
        ) from e
    except Exception as e:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise FetcherError(f"Failed to download asset '{match_filename}': {e}") from e
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.debug(f"Failed to remove '{temp_path}': {e}", exc_info=True)

    if not match_filename.lower().endswith(".zip"):
        base_name, _ = os.path.splitext(match_filename)
        zip_filename = f"{base_name}.zip"

        formats_dict = catalog.get("formats", {}).get(package, {})
        has_zip = False
        for k in formats_dict.keys():
            if k.lower() == zip_filename.lower():
                has_zip = True
                zip_filename = k
                break

        if has_zip:
            zip_local_path = fetch(zip_filename, overwrite=overwrite, package=package)
            try:
                _safe_extract_zip(zip_local_path, target_dir)
                extracted_flag = pjoin(target_dir, f".extracted_{zip_filename}")
                with open(extracted_flag, "w", encoding="utf-8") as f:
                    f.write("ok")
            except FetcherError:
                raise
            except Exception as e:
                raise FetcherError(
                    f"Failed to extract zip file '{zip_filename}': {e}"
                ) from e

    return target_path


def _select_by_extension(paths: list[str], ext_clean: str) -> list[str]:
    """Keep the paths carrying an extension, or every non-archive path.

    Parameters
    ----------
    paths : list of str
        Candidate paths.
    ext_clean : str
        Extension to keep, without leading dot.

    Returns
    -------
    list of str
        Paths ending with `ext_clean`, or, when none do (the caller passed a
        package name such as 'gmsh' rather than an extension), every path that
        is not a zip companion.
    """
    matching = [p for p in paths if p.lower().endswith(f".{ext_clean}")]
    if matching:
        return matching
    return [p for p in paths if not p.lower().endswith(".zip")]


def get_cached_files(ext: str) -> list[str]:
    """List the already downloaded assets for an extension, without any network.

    Parameters
    ----------
    ext : str
        The extension to query (e.g., '.vtp' or 'vtp').

    Returns
    -------
    list of str
        The absolute local paths of the cached files, sorted.
    """
    ext_clean = ext.lower().lstrip(".")
    if not ext_clean:
        raise FetcherError("Invalid extension format provided.")

    target_dir = pjoin(POLYXIOS_HOME, get_package_name(ext_clean))
    if not os.path.isdir(target_dir):
        return []

    cached = [
        pjoin(target_dir, entry)
        for entry in sorted(os.listdir(target_dir))
        if not entry.startswith(".") and os.path.isfile(pjoin(target_dir, entry))
    ]
    return _select_by_extension(cached, ext_clean)


def fetch_by_extension(ext: str, overwrite: bool = False) -> list[str]:
    """Discover and download all remote assets matching a specific file extension.

    Every asset the catalog lists for the extension is downloaded unless an
    up-to-date copy is already cached. Use `get_cached_files` to list what is
    already on disk without contacting the network.

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

    total_size = sum(info.get("size_bytes", 0) for info in format_files.values())
    logger.info(
        f"Starting download of {len(format_files)} files for extension '{ext_clean}' "
        f"(Total size: {total_size / (1024 * 1024):.2f} MB)..."
    )

    local_files = []
    for filename in sorted(format_files.keys()):
        local_path = fetch(filename, overwrite=overwrite)
        local_files.append(local_path)

    return _select_by_extension(local_files, ext_clean)
