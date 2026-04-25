#!/usr/bin/env python3
"""Fetch NHANES 2017-2018 SAS XPT tables for the jaxcross structure-discovery
demo.

NHANES (the National Health and Nutrition Examination Survey, run by CDC's
National Center for Health Statistics) is the canonical large mixed-type
clinical-feature dataset that's 100% public, no authorization required.
We pull 12 topic tables from the 2017-2018 cycle (~10 K participants,
joins on the SEQN respondent id):

    DEMO_J     — demographics: age, sex, race, education, household income
    BMX_J      — body measurements: BMI, weight, height, waist
    BPX_J      — blood pressure (systolic, diastolic, pulse)
    BIOPRO_J   — standard biochemistry panel (24 lab values: creatinine,
                 glucose, AST, ALT, albumin, electrolytes, etc.)
    CBC_J      — complete blood count (WBC, RBC, Hgb, platelets, indices)
    GHB_J      — glycohemoglobin (HbA1c)
    TCHOL_J    — total cholesterol
    HDL_J      — HDL cholesterol
    TRIGLY_J   — triglycerides + LDL
    DIQ_J      — diabetes self-report
    BPQ_J      — blood-pressure / hypertension self-report
    MCQ_J      — medical conditions (CHD, CHF, stroke, cancer, etc.)

Outputs (results/raw/):
    {table}_J.xpt           raw SAS Transport file per table
    column_metadata.json    column labels + value formats from the SAS metadata
                            (handy for the discovery writeup)

Usage:
    uv run python examples/nhanes_clinical/fetch_nhanes.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pyreadstat

# NHANES 2017-2018 cycle. Suffix _J is the cycle identifier.
CYCLE = "2017"  # NHANES URL convention: cycle directory uses the start year only
TABLES = [
    "DEMO_J",
    "BMX_J",
    "BPX_J",
    "BIOPRO_J",
    "CBC_J",
    "GHB_J",
    "TCHOL_J",
    "HDL_J",
    "TRIGLY_J",
    "DIQ_J",
    "BPQ_J",
    "MCQ_J",
]
BASE_URL = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{CYCLE}/DataFiles"
OUT_DIR = Path("examples/nhanes_clinical/results/raw")


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "jaxcross-nhanes-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        out.write(resp.read())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict] = {}

    for table in TABLES:
        dest = OUT_DIR / f"{table}.xpt"
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"[cached] {table} ({dest.stat().st_size / 1024:.0f} KB)")
        else:
            url = f"{BASE_URL}/{table}.XPT"
            print(f"[download] {url} -> {dest}")
            t0 = time.time()
            try:
                _download(url, dest)
            except Exception as exc:
                print(f"  Failed: {exc}", file=sys.stderr)
                return 1
            print(f"  Wrote {dest.stat().st_size / 1024:.0f} KB in {time.time() - t0:.1f}s")

        # Read SAS metadata (column labels, value formats) — useful for the
        # discovery writeup so we can label cluster members with human
        # text (e.g. "LBXGH" -> "Glycohemoglobin (%) - HbA1c").
        try:
            _, meta = pyreadstat.read_xport(str(dest), metadataonly=True)
            metadata[table] = {
                "n_columns": len(meta.column_names),
                "n_rows": meta.number_rows or 0,
                "labels": dict(meta.column_names_to_labels),
            }
        except Exception as exc:
            print(f"  Could not read SAS metadata: {exc}", file=sys.stderr)
            metadata[table] = {"error": str(exc)}

    (OUT_DIR / "column_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n[ok] {len(TABLES)} tables fetched + metadata extracted to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
