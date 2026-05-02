"""Fetch NTAD North American Roads dataset, Texas filter, via ArcGIS REST.

Source dataset: BTS NTAD North American Roads (2020 release, AGOL-hosted).
ArcGIS REST endpoint with `maxRecordCount=2000` per page; ~20 pages for the
~39K Texas segments.

Filter: `COUNTRY = 2 AND JURISCODE = '02_48'` (US Texas).

Output:
  results/raw/page_NNNN.json    paginated feature collections
  results/raw/all_features.json merged feature list (one entry per segment)

Run:
    uv run python examples/ntad_roads/fetch_ntad.py
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path("examples/ntad_roads/results/raw")
BASE_URL = (
    "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/"
    "NTAD_North_American_Roads/FeatureServer/0/query"
)
PAGE_LIMIT = 2000


def fetch_page(offset: int) -> dict:
    params = {
        "where": "COUNTRY=2 AND JURISCODE='02_48'",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "json",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE_LIMIT),
        "orderByFields": "OBJECTID",
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"NTAD Roads (Texas) → {OUT_DIR}/")

    all_features: list[dict] = []
    for page_idx in range(args.max_pages):
        page_path = OUT_DIR / f"page_{page_idx:04d}.json"
        if page_path.exists() and not args.force:
            cached = json.loads(page_path.read_text())
            n = len(cached.get("features", []))
            print(f"  page {page_idx:04d}: cached ({n:,} features)")
            all_features.extend(cached.get("features", []))
            if n < PAGE_LIMIT:
                print("    last page reached on cached read")
                break
            continue

        offset = page_idx * PAGE_LIMIT
        print(f"  page {page_idx:04d}: fetching offset={offset}…")
        page = fetch_page(offset)
        page_path.write_text(json.dumps(page))
        n = len(page.get("features", []))
        print(f"    wrote {page_path} ({n:,} features)")
        all_features.extend(page.get("features", []))
        if n < PAGE_LIMIT:
            print("    last page reached")
            break

    # Save merged
    merged_path = OUT_DIR / "all_features.json"
    merged_path.write_text(json.dumps(all_features))
    print(f"\nSaved {merged_path} ({len(all_features):,} total features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
