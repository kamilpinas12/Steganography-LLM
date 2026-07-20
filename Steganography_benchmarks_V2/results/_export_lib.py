"""Flatten JSON → CSV row (one run = one row)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Lists / large structures — stay in the run JSON, not in CSV.
SKIP_KEYS = {
    "completed_task_ids",
    "humaneval_tasks",
    "task_results",
    "samples",
    "tasks",
}


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_mapping(data: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to prefix_key; skip lists."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in SKIP_KEYS:
            continue
        col = f"{prefix}{key}" if prefix else str(key)
        if _is_scalar(value):
            out[col] = value
        elif isinstance(value, dict):
            out.update(flatten_mapping(value, prefix=f"{col}_"))
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run_row(run_dir: Path, results_name: str) -> dict[str, Any] | None:
    """Manifest + evaluation/<results_name> → one flattened row (no prefixes)."""
    results_path = run_dir / "evaluation" / results_name
    if not results_path.is_file():
        return None

    row: dict[str, Any] = {"run_dir": run_dir.name}

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        row.update(flatten_mapping(load_json(manifest_path)))

    # Results overwrite any name collisions with the manifest (e.g. test).
    row.update(flatten_mapping(load_json(results_path)))
    return row


def export_benchmark_csv(
    *,
    runs_dir: Path,
    out_csv: Path,
    results_filename: str,
) -> int:
    rows: list[dict[str, Any]] = []
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            row = collect_run_row(run_dir, results_filename)
            if row:
                rows.append(row)

    if not rows:
        out_csv.write_text("", encoding="utf-8")
        print(f"No runs with {results_filename} in {runs_dir} → empty {out_csv}")
        return 0

    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    preferred = [
        "run_dir",
        "test",
        "model_key",
        "model_id",
        "threshold",
        "top_n",
        "platform",
        "seed",
        "max_new_tokens",
    ]
    ordered = [c for c in preferred if c in seen] + [c for c in columns if c not in preferred]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows, {len(ordered)} columns → {out_csv}")
    return len(rows)
