"""HumanEval pass@1 for arbitrary task subsets (no 164-task assert)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from human_eval.execution import check_correctness


def evaluate_pass_at_1_subset(
    problems: list[dict[str, str]],
    predictions: list[str],
    extract_code_fn: Callable[[str], str],
    *,
    timeout: float = 3.0,
    n_workers: int = 4,
) -> dict[str, Any]:
    if len(problems) != len(predictions):
        raise ValueError(
            f"predictions ({len(predictions)}) must match problems ({len(problems)})"
        )

    samples = [
        {
            "task_id": problem["task_id"],
            "completion": extract_code_fn(prediction),
        }
        for problem, prediction in zip(problems, predictions)
    ]

    task_results: list[dict[str, Any]] = []
    order = {problem["task_id"]: idx for idx, problem in enumerate(problems)}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_map = {
            executor.submit(
                check_correctness,
                problem,
                sample["completion"],
                timeout,
                completion_id,
            ): (problem["task_id"], sample["completion"])
            for completion_id, (problem, sample) in enumerate(zip(problems, samples))
        }
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

    task_results.sort(key=lambda row: order[row["task_id"]])
    passed_count = sum(1 for row in task_results if row["passed"])
    total = len(task_results)

    return {
        "pass_at_1": passed_count / total if total else 0.0,
        "task_results": task_results,
        "passed_count": passed_count,
        "failed_count": total - passed_count,
        "total_count": total,
    }
