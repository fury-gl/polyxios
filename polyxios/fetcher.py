import hashlib
import os
from os.path import expanduser, join as pjoin
import sys
import urllib.error
import urllib.request
import zipfile

from polyxios.exceptions import FetcherError

POLYXIOS_HOME = os.getenv("POLYXIOS_HOME", pjoin(expanduser("~"), ".polyxios"))

RELEASE_BASE_URL = "https://github.com/fury-gl/polyxios-data/releases/download/v0.1.0"

ZIP_SHAS = {
    "mesh": "9d90a3c8c642674b3c567045b6dd8feb0fe0135a1c4d7b4e757aca52f939c40f",
    "msh": "63dd184754b3500fe2bf0df51dbb8ab9bc06ec51dc240e96f26741358d9c1d94",
    "obj": "30660894f05786e369f557d9137f779ddf65c5f1a7dd753de1854caa6444f2c4",
    "ply": "b867443f52cf794d2467ab2ba58aaa5763fdabf321c9fe1a1f221b2179d2e9ed",
    "vtk": "0ae5335020cfc8b520d90fcb5b7898a7f377520b4f6db672ba6a20770e7c7dde",
    "vtp": "6dd8f15e4ae8e387925b855ace6adf94998bf47959de8374df76e155fb3fc67b",
    "vtr": "c69b2c00b65cd2f34a23f92f03eed82126d868440741a6da60c43e64635928e9",
    "vtu": "245004ae8dea5303b18359d416481b8eb1df16687bc7d165c5ee79cad7b695c5",
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
    expected_sha = ZIP_SHAS.get(subfolder)
    if not expected_sha:
        raise FetcherError(
            f"Extension format classification '{subfolder}' is not an official release package."
        )

    target_dir = pjoin(POLYXIOS_HOME, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    zip_filename = f"{subfolder}.zip"
    zip_url = f"{RELEASE_BASE_URL}/{zip_filename}"
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


def fetch(filename: str, overwrite: bool = False) -> str:
    """
    Resolve, download, and track local path for any Polyxios test asset.

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
    """
    filename_lower = filename.lower()

    _, ext = os.path.splitext(filename_lower)
    if not ext:
        raise FetcherError(
            f"Cannot resolve target folder: filename '{filename}' has no extension."
        )
    subfolder = ext[1:]

    target_dir = pjoin(POLYXIOS_HOME, subfolder)
    target_path = pjoin(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        return target_path

    _download_and_extract_zip(subfolder)

    if not os.path.exists(target_path):
        raise FetcherError(
            f"Asset '{filename}' was not found in the extracted '{subfolder}.zip' package."
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

    target_dir = pjoin(POLYXIOS_HOME, ext_clean)
    is_empty = not os.path.exists(target_dir) or not os.listdir(target_dir)

    if is_empty or overwrite:
        _download_and_extract_zip(ext_clean)

    local_files = []
    if os.path.exists(target_dir):
        for entry in os.listdir(target_dir):
            full_path = pjoin(target_dir, entry)
            if os.path.isfile(full_path) and entry.lower().endswith(f".{ext_clean}"):
                local_files.append(full_path)

    return sorted(local_files)
