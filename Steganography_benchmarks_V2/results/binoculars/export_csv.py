#!/usr/bin/env python3
"""Export binoculars summary.csv (analysis columns only)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _export_lib import export_benchmark_csv  # noqa: E402

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    export_benchmark_csv(
        runs_dir=HERE / "runs",
        out_csv=HERE / "summary.csv",
        results_filename="binoculars_results.json",
        benchmark="binoculars",
    )
