#!/bin/bash
# Full Pipeline Benchmark: PointCloud → RRT* → B-Spline → L-BFGS
# Usage: ./build_and_run.sh [trials=20] [time_budget=0.5]

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$BENCHMARK_DIR/build"
TRIALS="${1:-2}"
TIME_BUDGET="${2:-0.5}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   Full Pipeline Benchmark                                           ║"
echo "║   PointCloud → RRT* → B-Spline → L-BFGS → Trajectory                ║"
echo "║   Comparing: Classical vs Enhanced Informed RRT*                     ║"
echo "║   Trials: $TRIALS  |  Time budget: ${TIME_BUDGET}s  |  Scenarios: 8             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Build
echo "--- Building (Release + -O3 -march=native) ---"
# Avoid stale CMake cache / stale binaries from old paths.
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
echo ""

# Run benchmark
echo "--- Running Benchmark ---"
./full_pipeline_benchmark "$TRIALS" "$TIME_BUDGET"
echo ""

# Plot results
echo "--- Generating Plots ---"
cd "$BENCHMARK_DIR"
PLOT_CSV="$BENCHMARK_DIR/pipeline_benchmark_results.csv"
# Remove stale top-level CSV so we always plot the newest output from this run.
rm -f "$PLOT_CSV"

if [ -f "$PLOT_CSV" ]; then
    python3 plot_results.py "$PLOT_CSV" loads
else
    echo "WARNING: $PLOT_CSV not found in current dir, checking build dir..."
    if [ -f "$BUILD_DIR/pipeline_benchmark_results.csv" ]; then
        python3 plot_results.py "$BUILD_DIR/pipeline_benchmark_results.csv" loads
    else
        echo "ERROR: CSV not found."
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Done.                                                             ║"
echo "║  CSV:   $BENCHMARK_DIR/pipeline_benchmark_results.csv"
echo "║  Plots: $BENCHMARK_DIR/loads/"
echo "╚══════════════════════════════════════════════════════════════════════╝"
