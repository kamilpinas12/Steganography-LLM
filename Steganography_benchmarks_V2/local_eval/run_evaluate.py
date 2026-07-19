#!/usr/bin/env python3
"""Lokalna ewaluacja RAW runów — bez GPU, bez torch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RUNS = Path(__file__).resolve().parent / "runs"


def discover_runs(runs_root: Path) -> list[Path]:
    """Find run dirs with raw_responses.jsonl or any *.jsonl."""
    if not runs_root.exists():
        return []
    runs: set[Path] = set()
    for jsonl in runs_root.rglob("*.jsonl"):
        runs.add(jsonl.parent.resolve())
    return sorted(runs, key=lambda p: (p.name, str(p)))


def run_label(run_dir: Path, runs_root: Path) -> str:
    try:
        rel = run_dir.relative_to(runs_root.resolve())
    except ValueError:
        rel = run_dir
    if rel.parent == Path("."):
        return run_dir.name
    return f"{rel.as_posix()}  (folder zip: {rel.parent})"


def resolve_run(run_arg: str, runs_root: Path) -> Path | None:
    direct = (runs_root / run_arg).resolve()
    if direct.is_dir() and list(direct.glob("*.jsonl")):
        return direct

    discovered = discover_runs(runs_root)

    by_name = [r for r in discovered if r.name == run_arg]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        print("Wiele runów o tej nazwie — podaj pełną ścieżkę względem runs/:", file=sys.stderr)
        for r in by_name:
            print(f"  {r.relative_to(runs_root.resolve())}", file=sys.stderr)
        return None

    suffix_matches = [
        r for r in discovered
        if str(r.relative_to(runs_root.resolve())).endswith(run_arg)
        or run_arg in str(r.relative_to(runs_root.resolve()))
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ewaluuj run z local_eval/runs/ (CPU only, no GPU).",
    )
    parser.add_argument(
        "run",
        nargs="?",
        help="Nazwa folderu runu (np. 2026-07-11_..._llama_humaneval_th0_0)",
    )
    parser.add_argument("--list", action="store_true", help="Pokaż dostępne runy")
    parser.add_argument("--all", action="store_true", help="Ewaluuj wszystkie runy w runs/")
    parser.add_argument("--dry-run", action="store_true", help="Tylko ekstrakcja kodu, bez testów")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        runs = discover_runs(LOCAL_RUNS)
        if not runs:
            print(f"Brak runów w {LOCAL_RUNS}")
            return 0
        print(f"Runy w {LOCAL_RUNS}:\n")
        for run_dir in runs:
            manifest = run_dir / "manifest.json"
            status = ""
            test_name = ""
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                test_name = f" [{data.get('test', '?')}]"
                status = (
                    f"  [{data.get('status', '?')}, "
                    f"{data.get('completed_count', '?')}/{data.get('total_tasks', '?')}]"
                )
            print(f"  {run_label(run_dir, LOCAL_RUNS)}{test_name}{status}")
        return 0

    if args.all:
        runs = discover_runs(LOCAL_RUNS)
        if not runs:
            print(f"Brak runów w {LOCAL_RUNS}")
            return 0
        failed = 0
        for run_dir in runs:
            print(f"\n{'='*60}\n{run_dir.name}\n{'='*60}", flush=True)
            cmd = [
                sys.executable,
                str(V2_ROOT / "evaluate_responses.py"),
                "--run-dir", str(run_dir),
                "--workers", str(args.workers),
                "--timeout", str(args.timeout),
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            rc = subprocess.run(cmd, cwd=str(V2_ROOT), check=False).returncode
            if rc != 0:
                failed += 1
        print(f"\nDone: {len(runs) - failed}/{len(runs)} OK, {failed} failed")
        return 1 if failed else 0

    if not args.run:
        print("Podaj nazwę runu lub użyj --list", file=sys.stderr)
        return 1

    run_dir = resolve_run(args.run, LOCAL_RUNS)
    if run_dir is None:
        print(f"Nie znaleziono runu: {args.run!r}", file=sys.stderr)
        print("Użyj: python run_evaluate.py --list", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(V2_ROOT / "evaluate_responses.py"),
        "--run-dir", str(run_dir),
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Ewaluacja (CPU): {run_dir}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(V2_ROOT), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
