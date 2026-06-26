import hashlib
import os
from os.path import expanduser, join as pjoin
import sys
import urllib.error
import urllib.request
import zipfile

from polyxios.exceptions import FetcherError

POLYXIOS_HOME = os.getenv("POLYXIOS_HOME", pjoin(expanduser("~"), ".polyxios"))

_GITHUB_BASE = "https://github.com/fury-gl/polyxios-data/releases/download"

# (release_tag, sha256) per package name
_PACKAGES: dict[str, tuple[str, str]] = {
    # v0.1.0 — large curated collections
    "mesh": (
        "v0.1.0",
        "9d90a3c8c642674b3c567045b6dd8feb0fe0135a1c4d7b4e757aca52f939c40f",
    ),
    "msh": (
        "v0.1.0",
        "63dd184754b3500fe2bf0df51dbb8ab9bc06ec51dc240e96f26741358d9c1d94",
    ),
    "obj": (
        "v0.1.0",
        "30660894f05786e369f557d9137f779ddf65c5f1a7dd753de1854caa6444f2c4",
    ),
    "ply": (
        "v0.1.0",
        "b867443f52cf794d2467ab2ba58aaa5763fdabf321c9fe1a1f221b2179d2e9ed",
    ),
    "vtk": (
        "v0.1.0",
        "0ae5335020cfc8b520d90fcb5b7898a7f377520b4f6db672ba6a20770e7c7dde",
    ),
    "vtp": (
        "v0.1.0",
        "6dd8f15e4ae8e387925b855ace6adf94998bf47959de8374df76e155fb3fc67b",
    ),
    "vtr": (
        "v0.1.0",
        "c69b2c00b65cd2f34a23f92f03eed82126d868440741a6da60c43e64635928e9",
    ),
    "vtu": (
        "v0.1.0",
        "245004ae8dea5303b18359d416481b8eb1df16687bc7d165c5ee79cad7b695c5",
    ),
    # v0.2.0 — small curated test files per codec
    "abaqus": (
        "v0.2.0",
        "2d59d621c1d9f98a86f708f6a1458dabba995bb14a0acb44b2f45bb52f95a69d",
    ),
    "avs": (
        "v0.2.0",
        "ca78183ba5a1a1344dfc782f1fc5f5cda5d7ab13ae03ac61281974df31f84d48",
    ),
    "dolfin": (
        "v0.2.0",
        "935a44466aa1e588c064d8a1e38e9bdcf4a025a606d664e0fdc1f893c747cf87",
    ),
    "flac3d": (
        "v0.2.0",
        "5a22537a6d432d745b0e5f1d845aa5b376a50cc4467b36af5dbe7157e1040f0b",
    ),
    "gmsh": (
        "v0.2.0",
        "f28886206e9494ef8a705420fc411962bfaa5cc5fca6b21005e324a426eb5b51",
    ),
    "mdpa": (
        "v0.2.0",
        "8d1c9eb5533b52bfcc99bfebd7173864cc0a2c9fc625f669a659ca96586a764a",
    ),
    "medit": (
        "v0.2.0",
        "ced4fadccf6b55cc4706059e36a966397fc8fa6403a2b25f5dbfd36ffaf2e5c3",
    ),
    "nastran": (
        "v0.2.0",
        "645d77eb32422601536a9bbeb3748050a138508e1074b29e5d5710e069d904ef",
    ),
    "netgen": (
        "v0.2.0",
        "89dd9e2465896598ea592e66d087253f7760fa258bf34f76a3958144260f96b4",
    ),
    "off": (
        "v0.2.0",
        "702c23dc40f593970d7af0f240f2f47fd97a4e64d719202c6a34917e33b12f37",
    ),
    "stl": (
        "v0.2.0",
        "046830521708462449cdaf420f515a5fdb4bb8f68a71c01ecfb4831e863b4152",
    ),
    "su2": (
        "v0.2.0",
        "8bb72ab83ac97b0e63e6ebdb1813f036a22f293fdac12775eef1b035ec8e3b61",
    ),
    "tecplot": (
        "v0.2.0",
        "34ef9c1890f170e6e937cd5d970fce3f90c4f85de375bd1b1909881fe0d28634",
    ),
    "tetgen": (
        "v0.2.0",
        "fcac38d7f8898e2af1478b9a2acd76097494598073dc648d46762cf5fdd861b5",
    ),
    "ugrid": (
        "v0.2.0",
        "83b3e87077455f4038b5112e8917c590520b0ed1c01935b3bf106e2c71e0a026",
    ),
    "wkt": (
        "v0.2.0",
        "02be71804c9a3cf39a558dd852e1a55e9b7f8fcd3c30263c4a5cb67efca16803",
    ),
}

# File extension → package name (only needed when they differ)
_EXT_TO_PACKAGE: dict[str, str] = {
    "inp": "abaqus",
    "fem": "nastran",
    "vol": "netgen",
    "f3grid": "flac3d",
    "tec": "tecplot",
    "mesh": "medit",
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
    pkg = _PACKAGES.get(subfolder)
    if not pkg:
        raise FetcherError(
            f"Extension format classification '{subfolder}' is not an official release package."
        )
    release_tag, expected_sha = pkg

    target_dir = pjoin(POLYXIOS_HOME, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    zip_filename = f"{subfolder}.zip"
    zip_url = f"{_GITHUB_BASE}/{release_tag}/{zip_filename}"
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
    ext_clean = ext[1:]
    package = _EXT_TO_PACKAGE.get(ext_clean, ext_clean)

    target_dir = pjoin(POLYXIOS_HOME, package)
    target_path = pjoin(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        return target_path

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
    is_empty = not os.path.exists(target_dir) or not os.listdir(target_dir)

    if is_empty or overwrite:
        _download_and_extract_zip(package)

    local_files = []
    if os.path.exists(target_dir):
        for entry in os.listdir(target_dir):
            full_path = pjoin(target_dir, entry)
            if os.path.isfile(full_path) and entry.lower().endswith(f".{ext_clean}"):
                local_files.append(full_path)

    return sorted(local_files)
