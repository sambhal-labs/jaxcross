#!/usr/bin/env python3
"""
Fetch summary data for ALL Materials Project materials and cache to Parquet.
Uses direct REST API with pagination to stay within 3.8GB RAM.
Bypasses mp-api client which loads all pydantic objects into memory.

Output: examples/materials_project/results/mp_all_summary_cache.parquet

Usage: uv run python examples/materials_project/fetch_mp_data.py
"""

import gc
import time
from pathlib import Path

import pandas as pd
import requests

MP_API_KEY = "yGA3US2qaVGQG51xrjLTDidG80JqvG5e"
CACHE_DIR = Path("examples/materials_project/results")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = CACHE_DIR / "mp_all_summary_cache.parquet"

if OUTPUT.exists():
    df = pd.read_parquet(OUTPUT)
    print(f"Cache already exists: {OUTPUT} ({len(df)} materials)")
    raise SystemExit(0)

# Direct REST API — fetch raw JSON, skip pydantic deserialization
BASE_URL = "https://api.materialsproject.org/materials/summary/"
HEADERS = {"X-API-KEY": MP_API_KEY}
FIELDS = (
    "material_id,formula_pretty,band_gap,formation_energy_per_atom,"
    "energy_above_hull,density,volume,nsites,nelements,"
    "is_stable,is_metal,total_magnetization,ordering"
)
LIMIT = 1000  # docs per page

print("Fetching ALL materials from Materials Project REST API...")
print(f"  Fields: {FIELDS}")
t0 = time.time()

all_records = []
skip = 0
total_fetched = 0

while True:
    params = {
        "_fields": FIELDS,
        "_skip": skip,
        "_limit": LIMIT,
    }

    resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    docs = data.get("data", [])
    if not docs:
        break

    for doc in docs:
        rec = {}
        for key in [
            "material_id",
            "formula_pretty",
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "volume",
            "nsites",
            "nelements",
            "is_stable",
            "is_metal",
            "total_magnetization",
            "ordering",
        ]:
            rec[key] = doc.get(key)
        all_records.append(rec)

    total_fetched += len(docs)
    skip += LIMIT

    if total_fetched % 10000 == 0 or len(docs) < LIMIT:
        elapsed = time.time() - t0
        mem_mb = len(all_records) * 200 / 1024 / 1024  # rough estimate
        print(f"  {total_fetched} materials fetched ({elapsed:.0f}s, ~{mem_mb:.0f}MB records)")
        gc.collect()

    if len(docs) < LIMIT:
        break

print(f"\n  Total: {total_fetched} materials ({time.time() - t0:.0f}s)")

df = pd.DataFrame(all_records)
del all_records
gc.collect()

df.to_parquet(OUTPUT, index=False)
elapsed = time.time() - t0
print(f"Done: {len(df)} materials cached to {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Time: {elapsed:.0f}s")
