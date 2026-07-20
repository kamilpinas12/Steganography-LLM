#!/usr/bin/env python3
"""Evaluate RAW runs under results/<benchmark>/runs/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent
V2_ROOT = RESULTS_ROOT.parent
SCRIPTS_DIR = V2_ROOT / "scripts"
EVAL_SCRIPT = SCRIPTS_DIR / "evaluate_responses.py"
BENCHMARKS = ("humaneval", "capacity", "perplexity", "binoculars")


def runs_roots() -> list[Path]:
    return [RESULTS_ROOT / name / "runs" for name in BENCHMARKS]


def discover_runs() -> list[Path]:
    runs: set[Path] = set()
    for root in runs_roots():
        if not root.exists():
            continue
        for jsonl in root.rglob("*.jsonl"):
            runs.add(jsonl.parent.resolve())
    return sorted(runs, key=lambda p: (p.name, str(p)))


def run_label(run_dir: Path) -> str:
    for root in runs_roots():
        try:
            rel = run_dir.relative_to(root.resolve())
            bench = root.parent.name
            return f"{bench}/{rel.as_posix()}"
        except ValueError:
            continue
    return run_dir.name


def resolve_run(run_arg: str) -> Path | None:
    candidates = [
        (RESULTS_ROOT / run_arg).resolve(),
        (RESULTS_ROOT / "humaneval" / "runs" / run_arg).resolve(),
        (RESULTS_ROOT / "capacity" / "runs" / run_arg).resolve(),
        (RESULTS_ROOT / "perplexity" / "runs" / run_arg).resolve(),
        (RESULTS_ROOT / "binoculars" / "runs" / run_arg).resolve(),
    ]
    for direct in candidates:
        if direct.is_dir() and list(direct.glob("*.jsonl")):
            return direct

    discovered = discover_runs()
    by_name = [r for r in discovered if r.name == run_arg]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        print(
            "Multiple runs with that name — pass a path (e.g. humaneval/runs/...):",
            file=sys.stderr,
        )
        for r in by_name:
            print(f"  {run_label(r)}", file=sys.stderr)
        return None

    suffix_matches = [r for r in discovered if run_arg in run_label(r)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a run from results/<benchmark>/runs/")
    parser.add_argument("run", nargs="?", help="Run name or path relative to results/")
    parser.add_argument("--list", action="store_true", help="List available runs")
    parser.add_argument("--all", action="store_true", help="Evaluate all runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def eval_cmd(run_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> int:
    args = parse_args()

    if args.list:
        runs = discover_runs()
        if not runs:
            print(f"No runs under {RESULTS_ROOT}/<benchmark>/runs/")
            return 0
        print(f"Runs in {RESULTS_ROOT}:\n")
        for run_dir in runs:
            manifest = run_dir / "manifest.json"
            extra = ""
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                extra = f" [{data.get('test', '?')}]"
            print(f"  {run_label(run_dir)}{extra}")
        return 0

    if args.all:
        runs = discover_runs()
        if not runs:
            print(f"No runs under {RESULTS_ROOT}/<benchmark>/runs/")
            return 0
        failed = 0
        for run_dir in runs:
            print(f"\n{'=' * 60}\n{run_label(run_dir)}\n{'=' * 60}", flush=True)
            rc = subprocess.run(eval_cmd(run_dir, args), cwd=str(SCRIPTS_DIR), check=False).returncode
            if rc != 0:
                failed += 1
        print(f"\nDone: {len(runs) - failed}/{len(runs)} OK, {failed} failed")
        return 1 if failed else 0

    if not args.run:
        print("Pass a run name or use --list", file=sys.stderr)
        return 1

    run_dir = resolve_run(args.run)
    if run_dir is None:
        print(f"Run not found: {args.run!r}", file=sys.stderr)
        print("Use: python run_evaluate.py --list", file=sys.stderr)
        return 1

    print(f"Evaluating: {run_dir}\n", flush=True)
    return subprocess.run(eval_cmd(run_dir, args), cwd=str(SCRIPTS_DIR), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
