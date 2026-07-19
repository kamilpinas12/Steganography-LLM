"""Evaluation handlers for each V2 benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from code_extract import _strip_to_assistant_reply
from common import merge_capacity_dicts
from raw_store import save_json, utc_now_iso


def _assistant_text(raw_completion: str) -> str:
    return _strip_to_assistant_reply(raw_completion).strip()


def evaluate_humaneval(
    run_dir: Path,
    eval_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    timeout: float,
    workers: int,
    dry_run: bool,
) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from code_extract import extract_completion
    from common import load_humaneval_problems, select_humaneval_problems
    from human_eval.execution import check_correctness

    humaneval_tasks = manifest.get("humaneval_tasks")
    all_problems = load_humaneval_problems()
    problems = select_humaneval_problems(all_problems, humaneval_tasks)
    record_by_id = {row["task_id"]: row for row in records}
    ordered_records = [record_by_id[p["task_id"]] for p in problems if p["task_id"] in record_by_id]
    ordered_problems = [p for p in problems if p["task_id"] in record_by_id]

    extracted_rows = []
    for record in ordered_records:
        raw = record["raw_completion"]
        entry_point = record.get("entry_point")
        extracted = extract_completion(raw, entry_point=entry_point)
        extracted_rows.append(
            {
                "task_id": record["task_id"],
                "entry_point": entry_point,
                "raw_completion": raw,
                "extracted_completion": extracted,
            }
        )
    save_json(eval_dir / "extracted_responses.json", extracted_rows)

    if dry_run:
        return {"dry_run": True, "extracted_count": len(extracted_rows)}

    task_results: list[dict[str, Any]] = []
    order = {p["task_id"]: i for i, p in enumerate(ordered_problems)}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for i, (problem, record) in enumerate(zip(ordered_problems, ordered_records)):
            completion = extract_completion(
                record["raw_completion"],
                entry_point=problem["entry_point"],
            )
            future = executor.submit(check_correctness, problem, completion, timeout, i)
            future_map[future] = (problem["task_id"], completion)
        for future in as_completed(future_map):
            task_id, completion = future_map[future]
            row = future.result()
            task_results.append(
                {
                    "task_id": task_id,
                    "passed": bool(row.get("passed", False)),
                    "result": row.get("result", ""),
                    "completion": completion,
                }
            )
    task_results.sort(key=lambda r: order[r["task_id"]])
    passed_count = sum(1 for r in task_results if r["passed"])
    total = len(task_results)

    capacity = merge_capacity_dicts([r.get("capacity", {}) for r in ordered_records])
    results = {
        "test": "humaneval",
        "evaluated_at": utc_now_iso(),
        "pass_at_1": passed_count / total if total else 0.0,
        "passed_count": passed_count,
        "failed_count": total - passed_count,
        "total_count": total,
        "task_results": task_results,
        "capacity": capacity,
    }
    save_json(eval_dir / "humaneval_results.json", results)
    lines = [f"HumanEval: {passed_count}/{total} passed", ""]
    for row in task_results:
        status = "PASSED" if row["passed"] else "FAILED"
        lines.append(f"[{status}] {row['task_id']}: {row.get('result', '')}")
    (eval_dir / "pass_fail_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n=== HumanEval: {passed_count}/{total} passed ===", flush=True)
    return results


def evaluate_capacity(
    eval_dir: Path,
    records: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    sample_rows = []
    capacity_dicts = []
    for record in records:
        cap = record.get("capacity") or record.get("stego", {}).get("capacity", {})
        capacity_dicts.append(cap)
        sample_rows.append(
            {
                "sample_id": record.get("sample_id"),
                "user_prompt": record.get("user_prompt"),
                "raw_completion": record.get("raw_completion") or record.get("stego", {}).get("raw_completion"),
                "raw_full_decoded": record.get("raw_full_decoded") or record.get("stego", {}).get("raw_full_decoded"),
                "capacity": cap,
            }
        )
    save_json(eval_dir / "samples_detail.json", sample_rows)
    aggregated = merge_capacity_dicts(capacity_dicts)
    results = {
        "test": "capacity",
        "evaluated_at": utc_now_iso(),
        "sample_count": len(records),
        "capacity": aggregated,
        "samples": sample_rows,
    }
    save_json(eval_dir / "capacity_results.json", results)
    if not dry_run:
        print(
            f"\n=== Capacity: avg_pool={aggregated['avg_pool_size']:.2f}, "
            f"BPT={aggregated['avg_bits_per_token']:.3f}, "
            f"embed_rate={aggregated['embedding_rate']:.3f}, "
            f"stego_activation={aggregated['stego_activation_ratio']:.1%} ===",
            flush=True,
        )
    return results


def _perplexity_sample_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_rows = []
    for record in records:
        baseline = record["baseline"]
        stego = record["stego"]
        sample_rows.append(
            {
                "sample_id": record["sample_id"],
                "prompt_index": record["prompt_index"],
                "user_prompt": record["user_prompt"],
                "baseline": baseline,
                "stego": stego,
            }
        )
    return sample_rows


def evaluate_perplexity_loaded(
    model,
    tokenizer,
    eval_dir: Path,
    records: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    from perplexity_metrics import sequence_perplexity

    sample_rows = _perplexity_sample_rows(records)
    save_json(eval_dir / "samples_detail.json", sample_rows)

    if dry_run:
        return {"dry_run": True, "sample_count": len(sample_rows)}

    scored_rows = []
    baseline_ppls: list[float] = []
    stego_ppls: list[float] = []
    for row in sample_rows:
        baseline_ppl = sequence_perplexity(model, tokenizer, row["baseline"]["raw_full_decoded"])
        stego_ppl = sequence_perplexity(model, tokenizer, row["stego"]["raw_full_decoded"])
        baseline_ppls.append(baseline_ppl)
        stego_ppls.append(stego_ppl)
        scored_rows.append(
            {
                **row,
                "baseline_perplexity": baseline_ppl,
                "stego_perplexity": stego_ppl,
                "perplexity_delta": stego_ppl - baseline_ppl,
            }
        )

    mean_baseline = sum(baseline_ppls) / len(baseline_ppls) if baseline_ppls else float("nan")
    mean_stego = sum(stego_ppls) / len(stego_ppls) if stego_ppls else float("nan")
    capacity = merge_capacity_dicts([r["stego"].get("capacity", {}) for r in sample_rows])
    results = {
        "test": "perplexity",
        "evaluated_at": utc_now_iso(),
        "perplexity": mean_stego,
        "baseline_perplexity": mean_baseline,
        "perplexity_delta": mean_stego - mean_baseline,
        "capacity": capacity,
        "samples": scored_rows,
    }
    save_json(eval_dir / "perplexity_results.json", results)
    print(
        f"\n=== Perplexity: baseline={mean_baseline:.2f}, stego={mean_stego:.2f}, "
        f"delta={mean_stego - mean_baseline:+.2f} ===",
        flush=True,
    )
    return results


def evaluate_perplexity(
    eval_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    platform: str,
    dry_run: bool,
) -> dict[str, Any]:
    from model_runtime import configure_platform, hf_login_if_needed, load_model_and_tokenizer, release_model, set_seed

    if dry_run:
        return evaluate_perplexity_loaded(
            None,
            None,
            eval_dir,
            records,
            dry_run=True,
        )

    hf_login_if_needed()
    configure_platform(platform)
    set_seed(int(manifest.get("seed", 1234)))
    model, tokenizer = load_model_and_tokenizer(manifest["model_id"])
    try:
        return evaluate_perplexity_loaded(
            model,
            tokenizer,
            eval_dir,
            records,
            dry_run=False,
        )
    finally:
        release_model(model, tokenizer)


def evaluate_binoculars(
    eval_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    platform: str,
    dry_run: bool,
) -> dict[str, Any]:
    from binoculars_scorer import BinocularsScorer

    sample_rows = []
    for record in records:
        baseline_text = _assistant_text(record["baseline"]["raw_completion"])
        stego_text = _assistant_text(record["stego"]["raw_completion"])
        sample_rows.append(
            {
                "sample_id": record["sample_id"],
                "prompt_index": record["prompt_index"],
                "user_prompt": record["user_prompt"],
                "baseline_generated": baseline_text,
                "stego_generated": stego_text,
                "baseline_raw_completion": record["baseline"]["raw_completion"],
                "stego_raw_completion": record["stego"]["raw_completion"],
                "capacity": record["stego"].get("capacity", {}),
            }
        )
    save_json(eval_dir / "samples_detail.json", sample_rows)

    if dry_run:
        return {"dry_run": True, "sample_count": len(sample_rows)}

    scorer = BinocularsScorer(platform=platform)
    scored_rows = []
    baseline_scores: list[float] = []
    stego_scores: list[float] = []
    for row in sample_rows:
        baseline_score = float(scorer.compute_score(row["baseline_generated"]))
        stego_score = float(scorer.compute_score(row["stego_generated"]))
        baseline_scores.append(baseline_score)
        stego_scores.append(stego_score)
        scored_rows.append(
            {
                **row,
                "baseline_binoculars_score": baseline_score,
                "stego_binoculars_score": stego_score,
                "baseline_prediction": scorer.prediction_label(baseline_score),
                "stego_prediction": scorer.prediction_label(stego_score),
            }
        )

    mean_baseline = sum(baseline_scores) / len(baseline_scores)
    mean_stego = sum(stego_scores) / len(stego_scores)
    baseline_ai_rate = sum(s < scorer.threshold for s in baseline_scores) / len(baseline_scores)
    stego_ai_rate = sum(s < scorer.threshold for s in stego_scores) / len(stego_scores)
    capacity = merge_capacity_dicts([r.get("capacity", {}) for r in sample_rows])
    results = {
        "test": "binoculars",
        "evaluated_at": utc_now_iso(),
        "binoculars_score": mean_stego,
        "baseline_binoculars_score": mean_baseline,
        "binoculars_score_delta": mean_stego - mean_baseline,
        "ai_detection_rate": stego_ai_rate,
        "baseline_ai_detection_rate": baseline_ai_rate,
        "binoculars_threshold": scorer.threshold,
        "capacity": capacity,
        "samples": scored_rows,
    }
    save_json(eval_dir / "binoculars_results.json", results)
    print(
        f"\n=== Binoculars: baseline={mean_baseline:.3f}, stego={mean_stego:.3f}, "
        f"AI rate stego={stego_ai_rate:.1%} ===",
        flush=True,
    )
    return results
