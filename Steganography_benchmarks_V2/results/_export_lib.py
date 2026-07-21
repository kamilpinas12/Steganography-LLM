"""Build analysis CSVs from run manifests + evaluation JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Analysis-oriented columns (no timestamps / platform / inferred / etc.).
SCHEMAS: dict[str, list[str]] = {
    "humaneval": [
        "run_dir",
        "model",
        "model_id",
        "threshold",
        "top_n",
        "seed",
        "max_new_tokens",
        "pass_at_1",
        "passed_count",
        "failed_count",
        "total_count",
        "bits_per_token",
        "stego_activation_ratio",
        "avg_pool_size",
        "avg_pool_size_stego_only",
        "total_steps",
        "stego_applied_steps",
    ],
    "capacity": [
        "run_dir",
        "model",
        "model_id",
        "threshold",
        "top_n",
        "seed",
        "max_new_tokens",
        "bits_per_token",
        "stego_activation_ratio",
        "avg_pool_size",
        "avg_pool_size_stego_only",
        "total_steps",
        "stego_applied_steps",
        "natural_fallback_steps",
    ],
    "perplexity": [
        "run_dir",
        "model",
        "model_id",
        "threshold",
        "top_n",
        "seed",
        "max_new_tokens",
        "baseline_perplexity",
        "perplexity",
        "perplexity_delta",
        "bits_per_token",
        "stego_activation_ratio",
        "avg_pool_size",
        "avg_pool_size_stego_only",
        "total_steps",
        "stego_applied_steps",
    ],
    "binoculars": [
        "run_dir",
        "model",
        "model_id",
        "threshold",
        "top_n",
        "seed",
        "max_new_tokens",
        "baseline_binoculars_score",
        "binoculars_score",
        "binoculars_score_delta",
        "baseline_ai_detection_rate",
        "ai_detection_rate",
        "binoculars_threshold",
        "bits_per_token",
        "stego_activation_ratio",
        "avg_pool_size",
        "avg_pool_size_stego_only",
        "total_steps",
        "stego_applied_steps",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _capacity_fields(capacity: dict[str, Any] | None) -> dict[str, Any]:
    cap = capacity or {}
    ratio = cap.get("stego_activation_ratio")
    if ratio is None:
        total = cap.get("total_steps") or 0
        stego = cap.get("stego_applied_steps") or 0
        ratio = (stego / total) if total else ""
    return {
        "bits_per_token": _num(cap.get("avg_bits_per_token")),
        "stego_activation_ratio": _num(ratio),
        "avg_pool_size": _num(cap.get("avg_pool_size")),
        "avg_pool_size_stego_only": _num(cap.get("avg_pool_size_stego_only")),
        "total_steps": _num(cap.get("total_steps")),
        "stego_applied_steps": _num(cap.get("stego_applied_steps")),
        "natural_fallback_steps": _num(cap.get("natural_fallback_steps")),
    }


def collect_run_row(run_dir: Path, results_name: str, benchmark: str) -> dict[str, Any] | None:
    results_path = run_dir / "evaluation" / results_name
    if not results_path.is_file():
        return None

    manifest: dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)

    results = load_json(results_path)
    cap = _capacity_fields(results.get("capacity") if isinstance(results.get("capacity"), dict) else {})

    row: dict[str, Any] = {
        "run_dir": run_dir.name,
        "model": manifest.get("model_key") or "",
        "model_id": manifest.get("model_id") or "",
        "threshold": _num(manifest.get("threshold")),
        "top_n": _num(manifest.get("top_n")),
        "seed": _num(manifest.get("seed")),
        "max_new_tokens": _num(manifest.get("max_new_tokens")),
        **cap,
    }

    if benchmark == "humaneval":
        row["pass_at_1"] = _num(results.get("pass_at_1"))
        row["passed_count"] = _num(results.get("passed_count"))
        row["failed_count"] = _num(results.get("failed_count"))
        row["total_count"] = _num(results.get("total_count"))
    elif benchmark == "perplexity":
        row["baseline_perplexity"] = _num(results.get("baseline_perplexity"))
        row["perplexity"] = _num(results.get("perplexity"))
        row["perplexity_delta"] = _num(results.get("perplexity_delta"))
    elif benchmark == "binoculars":
        row["baseline_binoculars_score"] = _num(results.get("baseline_binoculars_score"))
        row["binoculars_score"] = _num(results.get("binoculars_score"))
        row["binoculars_score_delta"] = _num(results.get("binoculars_score_delta"))
        row["baseline_ai_detection_rate"] = _num(results.get("baseline_ai_detection_rate"))
        row["ai_detection_rate"] = _num(results.get("ai_detection_rate"))
        row["binoculars_threshold"] = _num(results.get("binoculars_threshold"))

    columns = SCHEMAS[benchmark]
    return {col: row.get(col, "") for col in columns}


def export_benchmark_csv(
    *,
    runs_dir: Path,
    out_csv: Path,
    results_filename: str,
    benchmark: str,
) -> int:
    columns = SCHEMAS[benchmark]
    rows: list[dict[str, Any]] = []
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            row = collect_run_row(run_dir, results_filename, benchmark)
            if row:
                rows.append(row)

    rows.sort(key=lambda r: (str(r.get("model", "")), float(r.get("threshold") or 0)))

    if not rows:
        out_csv.write_text("", encoding="utf-8")
        print(f"No runs with {results_filename} in {runs_dir} → empty {out_csv}")
        return 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows, {len(columns)} columns → {out_csv}")
    return len(rows)
