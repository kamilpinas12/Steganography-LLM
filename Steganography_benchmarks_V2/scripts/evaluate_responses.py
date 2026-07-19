#!/usr/bin/env python3
"""Evaluate raw benchmark runs — routes by manifest.test."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

from common import EVAL_DIR_NAME, RUNS_ROOT, SUMMARY_COLUMNS, SUMMARY_CSV
from eval_handlers import (
    evaluate_binoculars,
    evaluate_capacity,
    evaluate_humaneval,
    evaluate_perplexity,
)
from raw_store import load_manifest, load_raw_records, save_json, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2: evaluate raw JSONL benchmark runs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--platform", choices=("colab", "kaggle", "local"), default="local")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_summary_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(SUMMARY_COLUMNS)


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in SUMMARY_COLUMNS])
        f.flush()
        os.fsync(f.fileno())


def build_summary_row(manifest: dict[str, Any], run_dir: Path, results: dict[str, Any]) -> dict[str, Any]:
    capacity = results.get("capacity", {})
    return {
        "Timestamp": utc_now_iso(),
        "Run_Dir": str(run_dir),
        "Test": manifest.get("test", ""),
        "Model_Key": manifest.get("model_key", ""),
        "Model_ID": manifest.get("model_id", ""),
        "Threshold": manifest.get("threshold", ""),
        "Top_N": manifest.get("top_n", ""),
        "Pass@1": results.get("pass_at_1", ""),
        "Perplexity": results.get("perplexity", ""),
        "Baseline_Perplexity": results.get("baseline_perplexity", ""),
        "Perplexity_Delta": results.get("perplexity_delta", ""),
        "Avg_Pool_Size": capacity.get("avg_pool_size", ""),
        "Avg_Pool_Size_Stego_Only": capacity.get("avg_pool_size_stego_only", ""),
        "Avg_BPT": capacity.get("avg_bits_per_token", ""),
        "Embedding_Rate": capacity.get("embedding_rate", ""),
        "Total_Generation_Steps": capacity.get("total_steps", ""),
        "Stego_Steps": capacity.get("stego_applied_steps", ""),
        "Stego_Activation_Ratio": capacity.get("stego_activation_ratio", ""),
        "Natural_Fallback_Steps": capacity.get("natural_fallback_steps", ""),
        "Binoculars_Score": results.get("binoculars_score", ""),
        "Baseline_Binoculars_Score": results.get("baseline_binoculars_score", ""),
        "Binoculars_Score_Delta": results.get("binoculars_score_delta", ""),
        "AI_Detection_Rate": results.get("ai_detection_rate", ""),
        "Baseline_AI_Detection_Rate": results.get("baseline_ai_detection_rate", ""),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    eval_dir = (args.eval_dir or run_dir / EVAL_DIR_NAME).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_dir)
    records = load_raw_records(run_dir)
    test = manifest.get("test", "humaneval")
    platform = manifest.get("platform", args.platform)

    if test == "humaneval":
        results = evaluate_humaneval(
            run_dir,
            eval_dir,
            manifest,
            records,
            timeout=args.timeout,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        result_file = "humaneval_results.json"
    elif test == "capacity":
        results = evaluate_capacity(eval_dir, records, dry_run=args.dry_run)
        result_file = "capacity_results.json"
    elif test == "perplexity":
        results = evaluate_perplexity(
            eval_dir,
            manifest,
            records,
            platform=platform,
            dry_run=args.dry_run,
        )
        result_file = "perplexity_results.json"
    elif test == "binoculars":
        results = evaluate_binoculars(
            eval_dir,
            manifest,
            records,
            platform=platform,
            dry_run=args.dry_run,
        )
        result_file = "binoculars_results.json"
    else:
        print(f"Unknown test in manifest: {test!r}", file=sys.stderr)
        return 1

    if not results.get("dry_run"):
        save_json(
            eval_dir / "manifest.json",
            {
                "phase": "evaluation",
                "status": "completed",
                "evaluated_at": utc_now_iso(),
                "source_run": str(run_dir),
                "test": test,
                "primary_result_file": result_file,
            },
        )
        ensure_summary_csv(SUMMARY_CSV)
        append_summary_row(SUMMARY_CSV, build_summary_row(manifest, run_dir, results))

    print(f"\nEvaluation saved to: {eval_dir}")
    if not args.dry_run:
        print(f"Summary CSV: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
