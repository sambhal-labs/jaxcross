#!/usr/bin/env python3
"""Generate the supplementary publication figures used in the arXiv papers
+ blog posts. Reads the CSV/JSON artifacts already on disk; no GPU needed.

Outputs (results/paper_figures/):
    fig_holdout_coverage.png        — 50/90/95 % empirical CI coverage per
                                      biomarker, with nominal target lines
    fig_in_vs_holdout.png           — side-by-side AUC + 90 % CI coverage,
                                      in-sample vs held-out
    fig_per_cycle_n.png             — per-cycle NHANES cohort size:
                                      our 9,254 vs literature multi-cycle pools

Usage:
    uv run python examples/nhanes_clinical/make_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DISCOVERY_WARM = Path("examples/nhanes_clinical/results/discovery_warm")
DISCOVERY_HOLDOUT = Path("examples/nhanes_clinical/results/discovery_holdout")
OUT_DIR = Path("examples/nhanes_clinical/results/paper_figures")


def fig_holdout_coverage() -> None:
    """Bar chart: 50/90/95% empirical coverage per biomarker, with nominal lines."""
    df = pl.read_csv(DISCOVERY_HOLDOUT / "holdout_ci_coverage.csv")
    # Filter rows with numeric ci_level (string "MAE" rows are interleaved)
    df = df.filter(pl.col("ci_level").is_in(["0.5", "0.9", "0.95"]))
    cols = sorted(df["column"].unique().to_list())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(len(cols))
    for i, level in enumerate([0.50, 0.90, 0.95]):
        cov = []
        for c in cols:
            r = df.filter((pl.col("column") == c) & (pl.col("ci_level") == str(level)))
            cov.append(float(r["empirical_coverage"][0]))
        ax.bar(
            x + (i - 1) * width,
            cov,
            width,
            label=f"{int(level * 100)} % CI",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axhline(level, color=f"C{i}", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Held-out credible-interval coverage on 1,432 masked biomarker cells\n"
        "(dotted lines = nominal target)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    # Cell-weighted aggregate annotation
    agg = json.loads((DISCOVERY_HOLDOUT / "holdout_summary.json").read_text())
    cov90 = agg["ci_coverage_aggregate"]["0.9"]
    cov95 = agg["ci_coverage_aggregate"]["0.95"]
    ax.text(
        0.02,
        0.96,
        f"Cell-weighted aggregate:  90 % CI cov = {cov90:.1%},  95 % CI cov = {cov95:.1%}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_holdout_coverage.png", dpi=120)
    plt.close()


def fig_in_vs_holdout() -> None:
    """Side-by-side: in-sample vs held-out for AUC, 90% CI cov, 95% CI cov."""
    in_class = pl.read_csv(DISCOVERY_WARM / "classification_metrics.csv")
    in_cov = pl.read_csv(DISCOVERY_WARM / "ci_coverage.csv")
    out_class = json.loads(
        (DISCOVERY_HOLDOUT / "holdout_classification_bootstrap.json").read_text()
    )
    out_cov_agg = json.loads((DISCOVERY_HOLDOUT / "holdout_summary.json").read_text())[
        "ci_coverage_aggregate"
    ]

    in_diq = in_class.filter(pl.col("label") == "DIQ010")
    in_auc = float(in_diq["auc"][0])

    in_90 = float(in_cov.filter(pl.col("ci_level") == 0.9)["empirical_coverage"].mean())
    in_95 = float(in_cov.filter(pl.col("ci_level") == 0.95)["empirical_coverage"].mean())
    in_50 = float(in_cov.filter(pl.col("ci_level") == 0.5)["empirical_coverage"].mean())
    out_auc_pt = out_class["auc_point"]
    out_auc_ci = out_class["auc_95ci"]
    out_50 = out_cov_agg["0.5"]
    out_90 = out_cov_agg["0.9"]
    out_95 = out_cov_agg["0.95"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: AUC
    ax = axes[0]
    bars = ax.bar(
        ["In-sample (n=8709)", "Held-out (n=1742)"],
        [in_auc, out_auc_pt],
        color=["steelblue", "firebrick"],
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    ax.errorbar(
        [1],
        [out_auc_pt],
        yerr=[[out_auc_pt - out_auc_ci[0]], [out_auc_ci[1] - out_auc_pt]],
        fmt="none",
        ecolor="black",
        capsize=6,
        elinewidth=1.2,
    )
    for bar, val in zip(bars, [in_auc, out_auc_pt], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    # Comparison lines
    ax.axhline(0.817, color="grey", linestyle="--", alpha=0.7)
    ax.text(
        1.5,
        0.823,
        "Mehrabkhani 2025 lifestyle (0.817)",
        color="grey",
        fontsize=8,
        ha="right",
    )
    ax.axhline(0.957, color="grey", linestyle=":", alpha=0.7)
    ax.text(
        1.5,
        0.909,
        "Dinh 2019 with-labs (0.957)",
        color="grey",
        fontsize=8,
        ha="right",
    )
    ax.set_ylabel("DIQ010 AUC")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Diabetes classification AUC", fontsize=11)

    # Panel 2: CI coverage
    ax = axes[1]
    levels = ["50 %", "90 %", "95 %"]
    in_cov_vals = [in_50, in_90, in_95]
    out_cov_vals = [out_50, out_90, out_95]
    nominal = [0.50, 0.90, 0.95]
    x = np.arange(len(levels))
    w = 0.3
    ax.bar(
        x - w / 2,
        in_cov_vals,
        w,
        label="In-sample",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        out_cov_vals,
        w,
        label="Held-out",
        color="firebrick",
        edgecolor="black",
        linewidth=0.5,
    )
    for xi, n in zip(x, nominal, strict=True):
        ax.hlines(n, xi - w * 1.1, xi + w * 1.1, colors="black", linestyles="--", linewidth=1.0)
    for xi, val in zip(x - w / 2, in_cov_vals, strict=True):
        ax.text(xi, val + 0.018, f"{val:.1%}", ha="center", fontsize=9)
    for xi, val in zip(x + w / 2, out_cov_vals, strict=True):
        ax.text(xi, val + 0.018, f"{val:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("CI coverage (mean across 6 biomarkers)\nDotted = nominal", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)

    plt.suptitle(
        "In-sample vs held-out: classification AUC and CI calibration",
        fontsize=12,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_in_vs_holdout.png", dpi=120)
    plt.close()


def fig_per_cycle_n() -> None:
    """Bar chart: per-cycle cohort size — ours vs literature pools."""
    studies = [
        (
            "Liu et al. 2023\n(Arch Med Sci, 2013–2018 high-risk, 3 cyc, n=2,355)",
            2355 / 3,
            "XGBoost",
        ),
        ("Dinh 2019\n(NHANES 1999–2014, 8 cycles, n≈21k)", 21000 / 8, "XGBoost ensemble"),
        (
            "Long et al. 2024\n(Nature Cardiovasc Res, 1988–2018, 15 cyc, n≈50k)",
            50000 / 15,
            "k-prototypes / GMM",
        ),
        (
            "Mehrabkhani 2025\n(NHANES 2007–2018 lifestyle, 6 cyc, n=29,509)",
            29509 / 6,
            "XGBoost (lifestyle)",
        ),
        ("Ours\n(NHANES 2017–2018, 1 cycle, n=9,254)", 9254, "jaxcross (Bayesian)"),
    ]
    studies.sort(key=lambda x: x[1])
    names = [s[0] for s in studies]
    ns = [s[1] for s in studies]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#888"] * (len(studies) - 1) + ["firebrick"]
    bars = ax.barh(np.arange(len(names)), ns, color=colors, edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars, ns, strict=True):
        ax.text(
            bar.get_width() + 200,
            bar.get_y() + bar.get_height() / 2,
            f"{int(n):,}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Average n per cycle")
    ax.set_title(
        "Per-cycle cohort size: NHANES diabetes / phenotyping literature\n"
        "Pooling cycles trades sample size for assay-drift + non-stationarity confounds",
        fontsize=11,
    )
    ax.set_xlim(0, max(ns) * 1.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_per_cycle_n.png", dpi=120)
    plt.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_holdout_coverage()
    print("  fig_holdout_coverage.png written")
    fig_in_vs_holdout()
    print("  fig_in_vs_holdout.png written")
    fig_per_cycle_n()
    print("  fig_per_cycle_n.png written")
    print(f"\nAll figures in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
