import hashlib
import json
import os
from os.path import expanduser, join as pjoin
import sys
import urllib.error
import urllib.request
import zipfile

from polyxios.exceptions import FetcherError

POLYXIOS_HOME = os.getenv("POLYXIOS_HOME", pjoin(expanduser("~"), ".polyxios"))

_GITHUB_BASE = "https://github.com/fury-gl/polyxios-data/releases/download"

_RELEASE_TAG = "latest"

_PACKAGES: dict[str, str] = {
    "mesh": "a17f51d70cb01e8e498844f7b74859183281e3b47d5805bcc9e4b661276f2927",
    "msh": "63dd184754b3500fe2bf0df51dbb8ab9bc06ec51dc240e96f26741358d9c1d94",
    "obj": "f98bdd9057111414236a9c272942b97337906fb44d59dc19c2f9d277ea7b42e2",
    "ply": "4919cc1e34c43ebd70453c011a7c67ffd92f5a7dc76bfa1163c588370e9385c9",
    "vtk": "76a292cfe4b1c5d5c804095592996c90900e9179c9293e361d36b1dd18d28151",
    "vtp": "a793c7c8662ca4af5ee86f968739a44f104a329299b4a084f01878519bb7f04a",
    "vtr": "c69b2c00b65cd2f34a23f92f03eed82126d868440741a6da60c43e64635928e9",
    "vtu": "562137f76553b18244985a2ffe614824d99c5f98f2160b4f08e48ff1647f89ab",
    "abaqus": "897553bfe1ea300c31c7b8b604e3187e9752c5d34f044ab8c9f6c605cf27c081",
    "avs": "cf23bb87fa00b31582fc41efb8da2ac10666f57d793abe041e576ce190db14be",
    "dolfin": "0098a571ee4faafed4147d572ba213d238e6c989f15dc514a836736599798bd3",
    "flac3d": "5782bcb83c73058b653ca6d6daa54b6a3cc8bede582827594af9e73c24fd741b",
    "gmsh": "e1902080a6af74a4ea0f3fb22e761bcd2316f85d7fde4cf21006367dacd763fb",
    "mdpa": "5aa8c1f3e1e911454f5329663497b338d6232339b6e6fdef15f20c01008c3243",
    "medit": "a98f5eeece9540f7941d21aece8ca865277b69652e2cf5a766edec20ebc725fe",
    "nastran": "929f3d5ddccfd6ae638e8b9176ebda7f5ce026b3025efb2c5ae4b4dd86719fc0",
    "netgen": "9252623a4e19e7176537c3875ed7bd96258e876d40e0026961bd79a6a01df207",
    "off": "593f9bd5e9739126093071773e3619c33bef158bf82ed34c0d2d2f24a2925429",
    "stl": "940b3e82b82917c896da2f84b207ceb022215a74e02847b015c178525c9b4fc5",
    "su2": "8b7450c685f0d91156069052bc41ddd7fe44912c525edcffd62b9e369302e979",
    "tecplot": "bc4164ae4a7d965ac11d7515b358b7fca718ecb2656f4cbecc088195ea0a0c0e",
    "tetgen": "ecba8a7e33c6d31b9bfca99e74755f4c585d7a0d069a6db8b1f0f8b34a2aa283",
    "ugrid": "cc71457956f8a37b9146b845751cc831a68c11f922e85464322e6f0795cb594b",
    "wkt": "3663f114a1526cedbf0500f94598cd0e9cadd76270a8236e73c40f69b8b44d80",
}

# File extension → package name (only needed when they differ)
_EXT_TO_PACKAGE: dict[str, str] = {
    "inp": "abaqus",
    "fem": "nastran",
    "vol": "netgen",
    "f3grid": "flac3d",
    "tec": "tecplot",
    "meshb": "medit",
    "xml": "dolfin",
}


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


def _download_and_extract_zip(subfolder: str) -> None:
    expected_sha = _PACKAGES.get(subfolder)
    if not expected_sha:
        raise FetcherError(
            f"Extension format classification '{subfolder}' is not an official release package."
        )

    target_dir = pjoin(POLYXIOS_HOME, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    zip_filename = f"{subfolder}.zip"
    zip_url = f"{_GITHUB_BASE}/{_RELEASE_TAG}/{zip_filename}"
    temp_zip_path = pjoin(target_dir, f".temp_{subfolder}.zip")

    try:
        req = urllib.request.Request(
            zip_url, headers={"User-Agent": "polyxios-fetcher"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(temp_zip_path, "wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _show_progress(zip_filename, downloaded, total_size)

            sys.stdout.write("\n")
            sys.stdout.flush()

        if not _verify_sha256(temp_zip_path, expected_sha):
            raise FetcherError(
                f"Integrity verification failed for {zip_filename}. Checksum mismatch."
            )

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        # Write metadata tag file upon successful extraction to validate cache
        tag_file = pjoin(target_dir, ".tag")
        with open(tag_file, "w") as f:
            f.write(f"{_RELEASE_TAG}:{expected_sha}")

    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        sys.stdout.flush()
        if e.code == 404:
            raise FetcherError(
                f"Release package '{zip_filename}' was not found on remote server."
            ) from e
        raise FetcherError(
            f"HTTP error occurred while downloading package: {e.code} {e.reason}"
        ) from e
    except Exception as e:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise FetcherError(
            f"Failed to synchronize asset package '{zip_filename}': {e}"
        ) from e
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)


def _is_cache_valid(package: str) -> bool:
    """Check if the cached package matches the currently expected tag/checksum."""
    target_dir = pjoin(POLYXIOS_HOME, package)
    tag_file = pjoin(target_dir, ".tag")
    expected_sha = _PACKAGES.get(package)
    if not expected_sha:
        return False
    expected_tag_content = f"{_RELEASE_TAG}:{expected_sha}"
    if os.path.exists(tag_file):
        try:
            with open(tag_file) as f:
                tag_content = f.read().strip()
            return tag_content == expected_tag_content
        except Exception:
            pass
    return False


_MODELS_URL = f"{_GITHUB_BASE}/{_RELEASE_TAG}/models.json"


def get_fetchable_files() -> dict[str, list[str]]:
    """Return a dictionary of all fetchable packages and their available files.

    Attempts to download the latest model catalog from the remote release
    and cache it locally. Falls back to the locally cached catalog if offline.

    Returns
    -------
    dict of str to list of str
        Mapping of package/extension name to list of available filenames.

    Raises
    ------
    FetcherError
        If the catalog could not be retrieved from both remote and cache.
    """
    local_path = pjoin(POLYXIOS_HOME, "models.json")

    # Try to download and cache the latest catalog
    try:
        req = urllib.request.Request(
            _MODELS_URL, headers={"User-Agent": "polyxios-fetcher"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            if isinstance(data, dict):
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
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    raise FetcherError(
        "Could not retrieve models catalog from remote release or local cache."
    )


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
    package = _EXT_TO_PACKAGE.get(ext_clean, ext_clean)

    target_dir = pjoin(POLYXIOS_HOME, package)
    target_path = pjoin(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        return target_path

    if _is_cache_valid(package) and not overwrite:
        raise FetcherError(
            f"Asset '{filename}' was not found in the cached '{package}' package."
        )

    _download_and_extract_zip(package)

    if not os.path.exists(target_path):
        raise FetcherError(
            f"Asset '{filename}' was not found in the extracted '{package}.zip' package."
        )

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

    package = _EXT_TO_PACKAGE.get(ext_clean, ext_clean)
    target_dir = pjoin(POLYXIOS_HOME, package)

    if not _is_cache_valid(package) or overwrite:
        _download_and_extract_zip(package)

    local_files = []
    if os.path.exists(target_dir):
        for entry in os.listdir(target_dir):
            full_path = pjoin(target_dir, entry)
            if os.path.isfile(full_path) and entry.lower().endswith(f".{ext_clean}"):
                local_files.append(full_path)

    return sorted(local_files)
