"""Build NTAD Texas road analytic matrix from raw ArcGIS feature dump.

Raw-column philosophy (per user direction): keep all useful numeric / categorical
attributes from the NTAD North American Roads dataset, drop only the
mechanically-unencodable / constant fields.

Dropped fields:
  OBJECTID, ID, LINKID         per-row IDs (not features)
  JURISCODE, JURISNAME         constants ('02_48', 'Texas')
  COUNTRY                      constant (2 = US)
  DIR                          constant (0 — directionality not modeled in NTAD 2020)
  SURFACE                      constant ('Paved' — NTAD only includes paved roads)
  ROADNAME                     free-form text (high-cardinality)

ROADNUM is parsed into a `road_system` categorical bin:
  I = Interstate, U = US Highway, S = State Highway, C = County Road,
  FM = Farm-to-Market, Other (everything else including unsigned)

Centroid latitude / longitude derived from the polyline geometry's path
coordinates (mean of vertex positions).

Output schema: **12 mixed-type columns**:
  CONTINUOUS (6)   length, shape_length, lanes, speedlim, centroid_latitude,
                   centroid_longitude
  CATEGORICAL (3)  class (6 levels), nhs (8 levels), road_system (6 levels)
  BINARY (3)       admin_is_state, border, is_interstate (derived)

Outputs (results/preprocessed/):
  train_data.npy        (n_rows, 12) float32
  segment_ids.npy       OBJECTID values, fixed-width <U10
  column_info.json      column metadata + transforms + bin maps
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAW_DIR = Path("examples/ntad_roads/results/raw")
OUT_DIR = Path("examples/ntad_roads/results/preprocessed")

ROAD_SYS_PATTERNS = {
    "I": "Interstate",
    "U": "US",
    "S": "State",
    "C": "County",
    "FM": "FM",
}


def _classify_road(roadnum: str | None) -> int:
    """Map ROADNUM string to a road_system bin index (0-5)."""
    if not roadnum or not roadnum.strip():
        return 5  # Other
    s = roadnum.strip().upper()
    if s.startswith("FM"):
        return 4
    c = s[0]
    return {"I": 0, "U": 1, "S": 2, "C": 3}.get(c, 5)


def _centroid(paths: list[list[list[float]]]) -> tuple[float, float]:
    """Polyline centroid = mean of vertex coordinates across all paths."""
    if not paths or not paths[0]:
        return float("nan"), float("nan")
    pts = [pt for path in paths for pt in path]
    if not pts:
        return float("nan"), float("nan")
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return float(sum(lats) / len(lats)), float(sum(lons) / len(lons))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw features…")
    data = json.loads((RAW_DIR / "all_features.json").read_text())
    n = len(data)
    print(f"  {n:,} segments")

    # Pre-pass: NHS distinct levels for encoding
    nhs_codes = sorted(
        {f["attributes"].get("NHS") for f in data if f["attributes"].get("NHS") is not None}
    )
    nhs_to_bin = {c: i for i, c in enumerate(nhs_codes)}
    print(f"  NHS distinct levels ({len(nhs_codes)}): {nhs_codes}")

    # Schema: 12 cols
    n_cols = 12
    train = np.full((n, n_cols), np.nan, dtype=np.float32)
    segment_ids = np.empty(n, dtype="<U10")

    # Per-row fill
    for i, f in enumerate(data):
        a = f.get("attributes", {})
        g = f.get("geometry", {})
        segment_ids[i] = str(a.get("OBJECTID") or "")[:10]

        # 0: length (continuous, km probably)
        train[i, 0] = float(a["LENGTH"]) if a.get("LENGTH") is not None else np.nan
        # 1: shape_length (continuous, geographic units)
        train[i, 1] = float(a["Shape__Length"]) if a.get("Shape__Length") is not None else np.nan
        # 2: lanes (continuous, 2-12)
        train[i, 2] = float(a["LANES"]) if a.get("LANES") is not None else np.nan
        # 3: speedlim (continuous, kph)
        train[i, 3] = float(a["SPEEDLIM"]) if a.get("SPEEDLIM") is not None else np.nan
        # 4-5: centroid lat / lon
        lat, lon = _centroid(g.get("paths", []))
        train[i, 4] = lat
        train[i, 5] = lon
        # 6: class (categorical, 1-6 → 0-5 zero-indexed)
        cls = a.get("CLASS")
        train[i, 6] = float(cls - 1) if cls is not None else np.nan
        # 7: nhs (categorical, encoded)
        nhs = a.get("NHS")
        train[i, 7] = float(nhs_to_bin[nhs]) if nhs in nhs_to_bin else np.nan
        # 8: road_system (categorical, 0-5)
        train[i, 8] = float(_classify_road(a.get("ROADNUM")))
        # 9: admin_is_state (binary)
        admin = a.get("ADMIN")
        train[i, 9] = 1.0 if admin == "State" else (0.0 if admin == "Municipal" else np.nan)
        # 10: border (binary, 0/2 → 0/1)
        b = a.get("BORDER")
        train[i, 10] = 1.0 if b == 2 else (0.0 if b == 0 else np.nan)
        # 11: is_interstate (derived binary)
        train[i, 11] = 1.0 if int(train[i, 8]) == 0 else 0.0

    # ── Diagnostics ──
    nan_frac = float(np.isnan(train).mean())
    print(f"\nMatrix: {train.shape}, NaN fraction {nan_frac:.4f}")
    print(f"is_interstate prevalence: {float(np.nanmean(train[:, 11])):.4f}")
    print(f"admin_is_state prevalence: {float(np.nanmean(train[:, 9])):.4f}")
    print(f"border prevalence: {float(np.nanmean(train[:, 10])):.4f}")
    # Categorical max-value check (must be < max_categories=16)
    print("\nCategorical / binary column ranges (must be <= 15):")
    for j in [6, 7, 8, 9, 10, 11]:
        col = train[:, j]
        clean = col[~np.isnan(col)]
        if len(clean):
            print(
                f"  col {j}: min={int(clean.min())} max={int(clean.max())} n_unique={len(np.unique(clean))}"
            )

    # ── Save ──
    np.save(OUT_DIR / "train_data.npy", train)
    np.save(OUT_DIR / "segment_ids.npy", segment_ids)
    info = {
        "n_rows": n,
        "n_cols": n_cols,
        "release_year": 2020,
        "state_filter": "Texas (COUNTRY=2 AND JURISCODE='02_48')",
        "columns": [
            {
                "index": 0,
                "name": "length",
                "type": "CONTINUOUS",
                "transform": "passthrough",
                "ntad_field": "LENGTH",
            },
            {
                "index": 1,
                "name": "shape_length",
                "type": "CONTINUOUS",
                "transform": "passthrough",
                "ntad_field": "Shape__Length",
            },
            {
                "index": 2,
                "name": "lanes",
                "type": "CONTINUOUS",
                "transform": "passthrough",
                "ntad_field": "LANES",
            },
            {
                "index": 3,
                "name": "speedlim",
                "type": "CONTINUOUS",
                "transform": "passthrough",
                "ntad_field": "SPEEDLIM",
            },
            {
                "index": 4,
                "name": "centroid_latitude",
                "type": "CONTINUOUS",
                "transform": "geometry_centroid",
                "ntad_field": "geometry.paths",
            },
            {
                "index": 5,
                "name": "centroid_longitude",
                "type": "CONTINUOUS",
                "transform": "geometry_centroid",
                "ntad_field": "geometry.paths",
            },
            {
                "index": 6,
                "name": "class",
                "type": "CATEGORICAL",
                "transform": "1to6_minus1",
                "ntad_field": "CLASS",
                "n_levels": 6,
            },
            {
                "index": 7,
                "name": "nhs",
                "type": "CATEGORICAL",
                "transform": "code_to_bin",
                "ntad_field": "NHS",
                "bin_map": {str(k): v for k, v in nhs_to_bin.items()},
            },
            {
                "index": 8,
                "name": "road_system",
                "type": "CATEGORICAL",
                "transform": "roadnum_prefix",
                "ntad_field": "ROADNUM",
                "bin_map": {"I": 0, "U": 1, "S": 2, "C": 3, "FM": 4, "Other": 5},
            },
            {
                "index": 9,
                "name": "admin_is_state",
                "type": "BINARY",
                "transform": "ADMIN == 'State'",
                "ntad_field": "ADMIN",
            },
            {
                "index": 10,
                "name": "border",
                "type": "BINARY",
                "transform": "BORDER == 2",
                "ntad_field": "BORDER",
            },
            {
                "index": 11,
                "name": "is_interstate",
                "type": "BINARY",
                "transform": "derived",
                "ntad_field": "ROADNUM",
                "derivation": "ROADNUM starts with 'I'",
            },
        ],
    }
    (OUT_DIR / "column_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  train_data.npy   ({train.shape}, {train.nbytes / 1024:.0f} KB)")
    print(f"  segment_ids.npy  ({n} segments)")
    print("  column_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
