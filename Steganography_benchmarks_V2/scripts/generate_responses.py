#!/usr/bin/env python3
"""Generate raw benchmark outputs — incremental JSONL save, no scoring."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from common import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TOP_N,
    EVAL_DIR_NAME,
    MODELS,
    PERPLEXITY_PROMPTS,
    RUNS_ROOT,
    SUMMARY_CSV,
    TESTS,
    create_run_dir,
    load_humaneval_problems,
    resolve_model,
    select_humaneval_problems,
)
from model_runtime import (
    configure_platform,
    enable_offline_mode,
    generate_raw,
    hf_login_if_needed,
    load_model_and_tokenizer,
    release_model,
    set_seed,
)
from prompts import build_chat_prompt, build_humaneval_prompt
from raw_store import RawResponseStore, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2: generate raw benchmark data (save after each sample).",
    )
    parser.add_argument(
        "--test",
        choices=TESTS,
        default="humaneval",
        help="Benchmark: humaneval | perplexity | capacity | binoculars",
    )
    parser.add_argument("--model", default=None, help=f"Model key ({', '.join(MODELS)}) or HF id")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--humaneval-tasks", default=None, help="e.g. '5', '0-10', '0,3,7'")
    parser.add_argument("--output-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None, help="Resume existing run dir")
    parser.add_argument("--platform", choices=("colab", "kaggle"), default="colab")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--eval-after",
        action="store_true",
        help=(
            "After generation: run evaluation in the same process. "
            "perplexity reuses the loaded model; binoculars frees GPU then loads Falcon."
        ),
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-tests", action="store_true")
    return parser.parse_args()


def _generation_target_count(test: str, humaneval_tasks: str | None) -> int:
    if test == "humaneval":
        return len(select_humaneval_problems(load_humaneval_problems(), humaneval_tasks))
    return len(PERPLEXITY_PROMPTS)


def _run_humaneval(
    store: RawResponseStore,
    *,
    model,
    tokenizer,
    model_key: str,
    threshold: float,
    top_n: int,
    max_new_tokens: int,
    seed: int,
    humaneval_tasks: str | None,
    done_ids: set[str],
) -> int:
    problems = select_humaneval_problems(load_humaneval_problems(), humaneval_tasks)
    pending = [p for p in problems if p["task_id"] not in done_ids]
    total_capacity_steps = 0

    for problem in tqdm(pending, desc=f"{model_key} | humaneval | th={threshold}"):
        prompt_text = build_humaneval_prompt(tokenizer, problem["prompt"])
        raw_full, raw_completion, capacity = generate_raw(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=max_new_tokens,
            threshold=threshold,
            top_n=top_n,
            seed=seed,
        )
        record = {
            "task_id": problem["task_id"],
            "task_index": int(problem["task_id"].split("/")[-1]),
            "entry_point": problem["entry_point"],
            "task_prompt": problem["prompt"],
            "prompt_text": prompt_text,
            "raw_full_decoded": raw_full,
            "raw_completion": raw_completion,
            "capacity": capacity.to_dict(),
            "generated_at": utc_now_iso(),
        }
        store.append_raw(record)
        total_capacity_steps += capacity.total_steps
        print(f"  saved {problem['task_id']}", flush=True)
    return total_capacity_steps


def _run_prompt_benchmark(
    store: RawResponseStore,
    *,
    test: str,
    model,
    tokenizer,
    model_key: str,
    threshold: float,
    top_n: int,
    max_new_tokens: int,
    seed: int,
    done_ids: set[str],
) -> int:
    total_capacity_steps = 0
    pending = [
        (idx, prompt)
        for idx, prompt in enumerate(PERPLEXITY_PROMPTS)
        if f"{test}/{idx}" not in done_ids
    ]

    for idx, user_prompt in tqdm(pending, desc=f"{model_key} | {test} | th={threshold}"):
        sample_id = f"{test}/{idx}"
        prompt_text = build_chat_prompt(tokenizer, user_prompt)
        record: dict = {
            "sample_id": sample_id,
            "prompt_index": idx,
            "user_prompt": user_prompt,
            "prompt_text": prompt_text,
            "generated_at": utc_now_iso(),
        }

        if test in ("perplexity", "binoculars"):
            baseline_full, baseline_comp, _ = generate_raw(
                model,
                tokenizer,
                prompt_text,
                max_new_tokens=max_new_tokens,
                threshold=0.0,
                top_n=top_n,
                seed=seed,
            )
            stego_full, stego_comp, capacity = generate_raw(
                model,
                tokenizer,
                prompt_text,
                max_new_tokens=max_new_tokens,
                threshold=threshold,
                top_n=top_n,
                seed=seed,
            )
            record["baseline"] = {
                "raw_full_decoded": baseline_full,
                "raw_completion": baseline_comp,
            }
            record["stego"] = {
                "raw_full_decoded": stego_full,
                "raw_completion": stego_comp,
                "capacity": capacity.to_dict(),
            }
            total_capacity_steps += capacity.total_steps
        else:
            raw_full, raw_completion, capacity = generate_raw(
                model,
                tokenizer,
                prompt_text,
                max_new_tokens=max_new_tokens,
                threshold=threshold,
                top_n=top_n,
                seed=seed,
            )
            record["raw_full_decoded"] = raw_full
            record["raw_completion"] = raw_completion
            record["capacity"] = capacity.to_dict()
            total_capacity_steps += capacity.total_steps

        store.append_raw(record)
        print(f"  saved {sample_id}", flush=True)
    return total_capacity_steps


def _perplexity_eval_path(run_dir: Path) -> Path:
    return run_dir / EVAL_DIR_NAME / "perplexity_results.json"


def _binoculars_eval_path(run_dir: Path) -> Path:
    return run_dir / EVAL_DIR_NAME / "binoculars_results.json"


def _save_evaluation(
    run_dir: Path,
    manifest: dict[str, Any],
    results: dict[str, Any],
    *,
    test: str,
    result_file: str,
) -> None:
    from evaluate_responses import append_summary_row, build_summary_row, ensure_summary_csv
    from raw_store import save_json, utc_now_iso

    eval_dir = run_dir / EVAL_DIR_NAME
    eval_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"\nEvaluation saved to: {eval_dir}", flush=True)
    print(f"Summary CSV: {SUMMARY_CSV}", flush=True)


def _run_perplexity_eval_after(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    model,
    tokenizer,
) -> None:
    from eval_handlers import evaluate_perplexity_loaded
    from raw_store import load_raw_records

    eval_dir = run_dir / EVAL_DIR_NAME
    eval_dir.mkdir(parents=True, exist_ok=True)
    records = load_raw_records(run_dir)
    results = evaluate_perplexity_loaded(
        model,
        tokenizer,
        eval_dir,
        records,
        dry_run=False,
    )
    _save_evaluation(
        run_dir,
        manifest,
        results,
        test="perplexity",
        result_file="perplexity_results.json",
    )


def _run_binoculars_eval_after(run_dir: Path, manifest: dict[str, Any]) -> None:
    import subprocess

    platform = manifest.get("platform", "colab")
    script = Path(__file__).resolve().parent / "evaluate_responses.py"
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run_dir),
        "--platform",
        platform,
    ]
    print(f"Binoculars eval subprocess: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    if args.list_models:
        for key, model_id in MODELS.items():
            print(f"  {key:6} -> {model_id}")
        return 0

    if args.list_tests:
        for test_name in TESTS:
            print(f"  {test_name}")
        return 0

    test = args.test
    humaneval_tasks = args.humaneval_tasks

    if args.run_dir:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            print(f"Run dir does not exist: {run_dir}", file=sys.stderr)
            return 1
        store = RawResponseStore(run_dir)
        done_ids = store.load_existing()
        manifest = store.manifest
        model_key = manifest.get("model_key", "")
        model_id = manifest.get("model_id", "")
        threshold = manifest.get("threshold", args.threshold)
        top_n = manifest.get("top_n", args.top_n)
        max_new_tokens = manifest.get("max_new_tokens", args.max_new_tokens)
        seed = manifest.get("seed", args.seed)
        humaneval_tasks = manifest.get("humaneval_tasks", args.humaneval_tasks)
        platform = manifest.get("platform", args.platform)
        test = manifest.get("test", test)
        if threshold is None or not model_id:
            print("Resume requires manifest with model_id and threshold.", file=sys.stderr)
            return 1
        print(f"Resuming {test} run: {run_dir} ({len(done_ids)} samples saved)")
    else:
        if args.model is None or args.threshold is None:
            print("--model and --threshold are required for a new run.", file=sys.stderr)
            return 1
        model_key, model_id = resolve_model(args.model)
        threshold = args.threshold
        top_n = args.top_n
        max_new_tokens = args.max_new_tokens
        seed = args.seed
        platform = args.platform
        run_dir = create_run_dir(
            args.output_root,
            test,
            model_key,
            threshold,
            humaneval_tasks=humaneval_tasks if test == "humaneval" else None,
        )
        store = RawResponseStore(run_dir)
        store.init_manifest(
            {
                "test": test,
                "model_key": model_key,
                "model_id": model_id,
                "threshold": threshold,
                "top_n": top_n,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "humaneval_tasks": humaneval_tasks,
                "platform": platform,
            },
            total_tasks=_generation_target_count(test, humaneval_tasks),
        )
        done_ids = set()
        print(f"New {test} run: {run_dir}")

    if args.offline:
        enable_offline_mode()
    configure_platform(platform, args.model_cache_dir)
    hf_login_if_needed()

    expected = _generation_target_count(test, humaneval_tasks)
    needs_generation = len(done_ids) < expected
    needs_ppl_eval = (
        args.eval_after
        and test == "perplexity"
        and not _perplexity_eval_path(run_dir).exists()
    )
    needs_bino_eval = (
        args.eval_after
        and test == "binoculars"
        and not _binoculars_eval_path(run_dir).exists()
    )
    needs_eval = needs_ppl_eval or needs_bino_eval

    if not needs_generation and not needs_eval:
        if len(done_ids) >= expected:
            print("All samples already generated.")
            store.mark_completed()
        if args.eval_after and test in ("perplexity", "binoculars"):
            print(f"Evaluation already present: {run_dir / EVAL_DIR_NAME}")
        return 0

    if args.eval_after and test not in ("perplexity", "binoculars"):
        print(
            "--eval-after is supported only for --test perplexity|binoculars.",
            file=sys.stderr,
        )
        return 1

    model = None
    tokenizer = None

    try:
        if needs_generation:
            print(f"Samples: {len(done_ids)} done / {expected} total")
            set_seed(seed)
            model, tokenizer = load_model_and_tokenizer(model_id)

            if test == "humaneval":
                total_capacity_steps = _run_humaneval(
                    store,
                    model=model,
                    tokenizer=tokenizer,
                    model_key=model_key,
                    threshold=threshold,
                    top_n=top_n,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    humaneval_tasks=humaneval_tasks,
                    done_ids=done_ids,
                )
            elif test in ("perplexity", "capacity", "binoculars"):
                total_capacity_steps = _run_prompt_benchmark(
                    store,
                    test=test,
                    model=model,
                    tokenizer=tokenizer,
                    model_key=model_key,
                    threshold=threshold,
                    top_n=top_n,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    done_ids=done_ids,
                )
            else:
                raise ValueError(f"Unsupported test: {test}")

            store.mark_completed({"total_generation_steps": total_capacity_steps})
            print(f"\nGeneration complete: {run_dir}")
            print(f"Raw file: {store.raw_path}")

        if needs_ppl_eval:
            if model is None:
                print("Generation already complete — loading model for perplexity eval...")
                set_seed(seed)
                model, tokenizer = load_model_and_tokenizer(model_id)
            _run_perplexity_eval_after(run_dir, store.manifest, model=model, tokenizer=tokenizer)
        elif needs_bino_eval:
            if model is not None:
                release_model(model, tokenizer)
                model, tokenizer = None, None
                print("Generator released — Binoculars eval in fresh subprocess.", flush=True)
            elif not needs_generation:
                print("Generation already complete — running Binoculars eval subprocess.")
            _run_binoculars_eval_after(run_dir, store.manifest)
        elif not args.eval_after:
            print(f"Next: python scripts/evaluate_responses.py --run-dir {run_dir}")

        if model is not None:
            release_model(model, tokenizer)
        return 0

    except Exception as exc:
        if model is not None:
            release_model(model, tokenizer)
        if needs_generation:
            store.mark_failed(str(exc))
        print(
            f"\nFailed after {store.manifest.get('completed_count', 0)} samples.",
            file=sys.stderr,
        )
        print(f"Raw data preserved in: {store.raw_path}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
