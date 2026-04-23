"""Transparent download and caching for remote data assets."""

import hashlib
import sys
import urllib.request
from pathlib import Path

from ._registry import CACHE_DIR_NAME, REGISTRY

__all__ = ["resolve_data_path"]


def resolve_data_path(filename: str) -> Path:
    """Resolve a remote data asset to a local file path.

    Resolution order:

    1. ``~/.cache/<CACHE_DIR_NAME>/<filename>``  — previously downloaded
    2. Download from registry URL → save to cache

    Parameters
    ----------
    filename : str
        Filename as registered in ``REGISTRY``.

    Returns
    -------
    Path
        Absolute path to the resolved local file.

    Raises
    ------
    FileNotFoundError
        If the file is not cached and not in the registry (or the
        registry entry has no URL).
    """
    cache_dir = Path.home() / ".cache" / CACHE_DIR_NAME
    cached = cache_dir / filename
    if cached.exists():
        return cached

    if filename not in REGISTRY:
        raise FileNotFoundError(
            f"'{filename}' is not in the REGISTRY and not cached at {cached}."
        )

    entry = REGISTRY[filename]
    url = entry.get("url")
    if not url:
        raise FileNotFoundError(
            f"'{filename}' has no download URL in the REGISTRY."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    _download(url, cached)
    _verify_sha256(cached, entry.get("sha256"))
    return cached


def _download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with progress."""
    print(f"Downloading {dest.name} …", file=sys.stderr, flush=True)
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        received = 0
        with open(dest, "wb") as fout:
            while True:
                chunk = resp.read(1 << 16)  # 64 KiB
                if not chunk:
                    break
                fout.write(chunk)
                received += len(chunk)
                if total:
                    pct = received * 100 // total
                    print(
                        f"\r  {received / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
    if total:
        print(file=sys.stderr)


def _verify_sha256(path: Path, expected: str | None) -> None:
    """Check SHA-256; delete file and raise on mismatch."""
    if not expected:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: "
            f"expected {expected[:16]}…, got {actual[:16]}…"
        )
