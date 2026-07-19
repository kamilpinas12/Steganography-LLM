"""Shared constants and helpers for V2 benchmarks."""

from __future__ import annotations

import re
from pathlib import Path

from human_eval.data import read_problems

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent  # Steganography_benchmarks_V2/
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_CSV = RUNS_ROOT / "summary.csv"

MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

TESTS = ("humaneval", "perplexity", "capacity", "binoculars")
DEFAULT_THRESHOLDS = [0.0, 0.01, 0.05, 0.1]
DEFAULT_TOP_N = 15
DEFAULT_SEED = 1234
DEFAULT_MAX_NEW_TOKENS = 512

PERPLEXITY_PROMPTS = [
    "Write a short paragraph explaining how binary search works.",
    "Describe the difference between a stack and a queue in computer science.",
    "Explain what a hash table is and when you would use one.",
    "Write a concise summary of how gradient descent optimizes neural networks.",
]

RAW_RESPONSES_FILE = "raw_responses.jsonl"
MANIFEST_FILE = "manifest.json"
EVAL_DIR_NAME = "evaluation"

SUMMARY_COLUMNS = [
    "Timestamp",
    "Run_Dir",
    "Test",
    "Model_Key",
    "Model_ID",
    "Threshold",
    "Top_N",
    "Pass@1",
    "Perplexity",
    "Baseline_Perplexity",
    "Perplexity_Delta",
    "Avg_Pool_Size",
    "Avg_Pool_Size_Stego_Only",
    "Avg_BPT",
    "Embedding_Rate",
    "Total_Generation_Steps",
    "Stego_Steps",
    "Stego_Activation_Ratio",
    "Natural_Fallback_Steps",
    "Binoculars_Score",
    "Baseline_Binoculars_Score",
    "Binoculars_Score_Delta",
    "AI_Detection_Rate",
    "Baseline_AI_Detection_Rate",
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def resolve_model(model_arg: str) -> tuple[str, str]:
    if model_arg in MODELS:
        return model_arg, MODELS[model_arg]
    return model_arg, model_arg


def record_id(row: dict) -> str:
    return str(row.get("sample_id") or row["task_id"])


def parse_humaneval_task_indices(task_spec: str) -> list[int]:
    indices: set[int] = set()
    for part in task_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())
            if start > end:
                raise ValueError(f"Invalid HumanEval range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if not indices:
        raise ValueError(f"No HumanEval task indices parsed from: {task_spec!r}")
    return sorted(indices)


def load_humaneval_problems() -> list[dict[str, str]]:
    problems = read_problems()
    if len(problems) != 164:
        raise RuntimeError(f"Expected 164 HumanEval tasks, got {len(problems)}")
    return [problems[task_id] for task_id in sorted(problems.keys())]


def select_humaneval_problems(
    problems: list[dict[str, str]],
    task_spec: str | None,
) -> list[dict[str, str]]:
    if not task_spec:
        return problems
    indices = parse_humaneval_task_indices(task_spec)
    total = len(problems)
    for index in indices:
        if index < 0 or index >= total:
            raise ValueError(
                f"HumanEval task index {index} out of range 0..{total - 1} "
                f"(requested: {task_spec!r})"
            )
    return [problems[index] for index in indices]


def create_run_dir(
    output_root: Path,
    test: str,
    model_key: str,
    threshold: float,
    *,
    humaneval_tasks: str | None = None,
) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    threshold_slug = str(threshold).replace(".", "_")
    run_name = f"{timestamp}_{slugify(model_key)}_{test}_th{threshold_slug}"
    if humaneval_tasks:
        run_name += f"_he{slugify(humaneval_tasks.replace(',', '_'))}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def merge_capacity_dicts(capacity_dicts: list[dict]) -> dict:
    empty = {
        "total_steps": 0,
        "stego_applied_steps": 0,
        "natural_fallback_steps": 0,
        "stego_activation_ratio": 0.0,
        "avg_pool_size": 0.0,
        "avg_pool_size_stego_only": 0.0,
        "avg_bits_per_token": 0.0,
        "embedding_rate": 0.0,
    }
    if not capacity_dicts:
        return empty
    total_steps = sum(int(c.get("total_steps", 0)) for c in capacity_dicts)
    stego_steps = sum(int(c.get("stego_applied_steps", 0)) for c in capacity_dicts)
    natural_steps = sum(int(c.get("natural_fallback_steps", 0)) for c in capacity_dicts)
    weighted_pool = sum(
        float(c.get("avg_pool_size", 0.0)) * int(c.get("total_steps", 0)) for c in capacity_dicts
    )
    weighted_pool_stego = sum(
        float(c.get("avg_pool_size_stego_only", 0.0)) * int(c.get("stego_applied_steps", 0))
        for c in capacity_dicts
    )
    weighted_bpt = sum(
        float(c.get("avg_bits_per_token", 0.0)) * int(c.get("total_steps", 0)) for c in capacity_dicts
    )
    weighted_embed = sum(
        float(c.get("embedding_rate", 0.0)) * int(c.get("total_steps", 0)) for c in capacity_dicts
    )
    return {
        "total_steps": total_steps,
        "stego_applied_steps": stego_steps,
        "natural_fallback_steps": natural_steps,
        "stego_activation_ratio": stego_steps / total_steps if total_steps else 0.0,
        "avg_pool_size": weighted_pool / total_steps if total_steps else 0.0,
        "avg_pool_size_stego_only": weighted_pool_stego / stego_steps if stego_steps else 0.0,
        "avg_bits_per_token": weighted_bpt / total_steps if total_steps else 0.0,
        "embedding_rate": weighted_embed / total_steps if total_steps else 0.0,
    }
