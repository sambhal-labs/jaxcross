#!/usr/bin/env python3
"""Fetch NASA C-MAPSS turbofan engine degradation dataset.

Downloads the official NASA PHM Society dataset (all 4 sub-datasets FD001-FD004)
from the public S3 mirror and caches locally. No API key required.

Output files (tab/space-separated, one row per (engine, cycle)):
  examples/c_mapss/results/raw/train_FD00{1-4}.txt
  examples/c_mapss/results/raw/test_FD00{1-4}.txt
  examples/c_mapss/results/raw/RUL_FD00{1-4}.txt

Usage:
    uv run python examples/c_mapss/fetch_cmapss.py
"""

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Public S3 mirror of the NASA PHM Society C-MAPSS dataset.
# Original source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
PRIMARY_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
# Fallback: widely used research mirror
FALLBACK_URL = "https://github.com/LahiruJayasinghe/RUL-Net/raw/master/CMAPSSData.zip"

OUT_DIR = Path("examples/c_mapss/results/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = {
    f"{prefix}_FD00{i}.txt" for prefix in ("train", "test", "RUL") for i in range(1, 5)
}


def _already_cached() -> bool:
    present = {p.name for p in OUT_DIR.glob("*.txt")}
    return REQUIRED_FILES.issubset(present)


def _download(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jaxcross-cmapss-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_zip(blob: bytes) -> int:
    """Extract FD00{1-4} train/test/RUL text files from a zip blob.

    The official NASA S3 distribution wraps the data inside a nested
    ``CMAPSSData.zip``; we recurse into any inner .zip members to find
    the txt files.
    """
    n_extracted = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base in REQUIRED_FILES:
                with zf.open(name) as src, open(OUT_DIR / base, "wb") as dst:
                    dst.write(src.read())
                n_extracted += 1
            elif name.lower().endswith(".zip"):
                with zf.open(name) as inner:
                    n_extracted += _extract_zip(inner.read())
    return n_extracted


def main() -> int:
    if _already_cached():
        print(f"All C-MAPSS files already present in {OUT_DIR}")
        for name in sorted(REQUIRED_FILES):
            size_kb = (OUT_DIR / name).stat().st_size / 1024
            print(f"  {name}  ({size_kb:.0f} KB)")
        return 0

    for url in (PRIMARY_URL, FALLBACK_URL):
        print(f"Downloading C-MAPSS from {url} ...")
        t0 = time.time()
        try:
            blob = _download(url)
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue

        print(f"  Downloaded {len(blob) / 1024 / 1024:.1f} MB in {time.time() - t0:.0f}s")

        try:
            n = _extract_zip(blob)
        except zipfile.BadZipFile as exc:
            print(f"  Not a valid zip: {exc}")
            continue

        if n == len(REQUIRED_FILES):
            print(f"  Extracted {n} files to {OUT_DIR}")
            return 0
        print(f"  Only found {n}/{len(REQUIRED_FILES)} required files; trying next source")

    print(
        "\nCould not fetch C-MAPSS automatically. Manual fallback:",
        "  1. Download 'Turbofan Engine Degradation Simulation Data Set' from",
        "     https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data",
        f"  2. Unzip it and place the 12 txt files into: {OUT_DIR}",
        sep="\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
