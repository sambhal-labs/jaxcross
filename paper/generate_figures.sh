#!/bin/bash
# Generate all paper figures from benchmark outputs.
#
# Prerequisites:
#   1. Run benchmarks on Kaggle/Colab GPU and download results to benchmarks/results/
#   2. Run scalability benchmark: uv run python benchmarks/scalability_benchmark.py
#
# This script collects outputs into paper/figures/ for LaTeX inclusion.

set -e

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
FIGURES_DIR="${PAPER_DIR}/figures"
RESULTS_DIR="${PAPER_DIR}/../benchmarks/results"

mkdir -p "${FIGURES_DIR}"

echo "=== Collecting paper figures ==="

# --- Synthetic benchmark ---
SYNTHETIC_DIR=$(ls -td "${RESULTS_DIR}/synthetic/"*/ 2>/dev/null | head -1)
if [ -n "${SYNTHETIC_DIR}" ]; then
    echo "Synthetic results: ${SYNTHETIC_DIR}"
    cp -v "${SYNTHETIC_DIR}/convergence.png" "${FIGURES_DIR}/synthetic_convergence.png" 2>/dev/null || echo "  [MISSING] convergence.png"
    cp -v "${SYNTHETIC_DIR}/z_matrix.png" "${FIGURES_DIR}/synthetic_z_matrix.png" 2>/dev/null || echo "  [MISSING] z_matrix.png"
    cp -v "${SYNTHETIC_DIR}/cluster_recovery.png" "${FIGURES_DIR}/synthetic_cluster_recovery.png" 2>/dev/null || echo "  [MISSING] cluster_recovery.png"
else
    echo "[MISSING] No synthetic benchmark results found."
    echo "  Run: uv run python benchmarks/paper_synthetic_benchmark.py"
fi

# --- MNIST benchmark ---
MNIST_DIR="${RESULTS_DIR}/mnist"
if [ -d "${MNIST_DIR}" ]; then
    echo "MNIST results: ${MNIST_DIR}"
    for f in convergence z_matrix pixel_dependence_map contingency classification_roc inpainting; do
        cp -v "${MNIST_DIR}/${f}.png" "${FIGURES_DIR}/mnist_${f}.png" 2>/dev/null || echo "  [MISSING] ${f}.png"
    done
else
    echo "[MISSING] No MNIST benchmark results found."
    echo "  Run mnist_paper_colab.ipynb on Kaggle and download results/"
fi

# --- WDI benchmark ---
WDI_DIR="${RESULTS_DIR}/wdi"
if [ -d "${WDI_DIR}" ]; then
    echo "WDI results: ${WDI_DIR}"
    for f in convergence z_matrix; do
        cp -v "${WDI_DIR}/${f}.png" "${FIGURES_DIR}/wdi_${f}.png" 2>/dev/null || echo "  [MISSING] ${f}.png"
    done
else
    echo "[MISSING] No WDI benchmark results found."
    echo "  Run wdi_macroeconomic_benchmark.ipynb on Kaggle and download results/"
fi

# --- Scalability benchmark ---
SCALE_DIR="${RESULTS_DIR}/scalability"
if [ -d "${SCALE_DIR}" ]; then
    echo "Scalability results: ${SCALE_DIR}"
    cp -v "${SCALE_DIR}/scalability.png" "${FIGURES_DIR}/scalability.png" 2>/dev/null || echo "  [MISSING] scalability.png"
    cp -v "${SCALE_DIR}/scalability_rows.png" "${FIGURES_DIR}/scalability_rows.png" 2>/dev/null || echo "  [MISSING] scalability_rows.png"
    cp -v "${SCALE_DIR}/scalability_cols.png" "${FIGURES_DIR}/scalability_cols.png" 2>/dev/null || echo "  [MISSING] scalability_cols.png"
else
    echo "[MISSING] No scalability benchmark results found."
    echo "  Run: uv run python benchmarks/scalability_benchmark.py"
fi

# --- Architecture diagrams from docs ---
DIAGRAMS_DIR="${PAPER_DIR}/../docs/diagrams"
if [ -d "${DIAGRAMS_DIR}" ]; then
    echo "Architecture diagrams: ${DIAGRAMS_DIR}"
    cp -v "${DIAGRAMS_DIR}/two-level-dp.svg" "${FIGURES_DIR}/two_level_dp.svg" 2>/dev/null || echo "  [MISSING] two-level-dp.svg"
    cp -v "${DIAGRAMS_DIR}/architecture-pipeline.svg" "${FIGURES_DIR}/architecture_pipeline.svg" 2>/dev/null || echo "  [MISSING] architecture-pipeline.svg"
    cp -v "${DIAGRAMS_DIR}/module-architecture.svg" "${FIGURES_DIR}/module_architecture.svg" 2>/dev/null || echo "  [MISSING] module-architecture.svg"
fi

echo ""
echo "=== Figure collection complete ==="
echo "Figures in: ${FIGURES_DIR}/"
ls -la "${FIGURES_DIR}/" 2>/dev/null || echo "  (empty)"
echo ""
echo "=== Missing figures checklist ==="
echo "To complete all figures, you need to run:"
echo "  1. uv run python benchmarks/paper_synthetic_benchmark.py  (local or GPU)"
echo "  2. benchmarks/mnist_paper_colab.ipynb on Kaggle P100"
echo "  3. benchmarks/wdi_macroeconomic_benchmark.ipynb on Kaggle P100"
echo "  4. uv run python benchmarks/scalability_benchmark.py  (GPU required)"
echo "  5. uv run python benchmarks/jit_benchmark.py  (for Table 2 numbers)"
echo ""
echo "Then re-run this script to collect all figures."
