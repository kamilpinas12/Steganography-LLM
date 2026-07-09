"""Evaluate text quality under dummy steganography constraints (no secret data hidden)."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dummy_stego_processor import DummyStegoHFLM, make_stego_logits_processor
from helpers import resolve_model_id

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
DEFAULT_CONFIG = BENCHMARKS_DIR / "default.toml"

CODE_EVAL_TASKS = {"humaneval", "mbpp", "mbpp_plus", "multiple-py"}


def prepare_harness_env(harness_cfg: dict) -> bool:
    tasks = harness_cfg.get("tasks", [])
    needs_code_eval = any(task.split(":")[0] in CODE_EVAL_TASKS for task in tasks)
    if not needs_code_eval:
        return False
    if not harness_cfg.get("allow_code_eval", False):
        raise RuntimeError(
            f'Taski {tasks} uruchamiają wygenerowany kod Pythona. '
            'Dodaj do [harness]: allow_code_eval = true w pliku TOML '
            '(lub ustaw HF_ALLOW_CODE_EVAL=1 w shellu).'
        )
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    return True


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_results_dir(config: dict) -> Path:
    output = config.get("output", {})
    rel = output.get("results_dir", "results")
    base = Path(rel)
    return base if base.is_absolute() else REPO_ROOT / base


def create_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_run_metadata(config_path: Path, config: dict, model_id: str, tests: list[str]) -> dict:
    model_key = config.get("model", {}).get("key", "")
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "tests": tests,
        "model_key": model_key,
        "model_id": model_id,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": config,
    }


def write_summary(run_dir: Path, lines: list[str]) -> None:
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model_and_tokenizer(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
    top_n: int,
    threshold: float | None,
    seed: int | None,
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if threshold is not None:
        gen_kwargs["logits_processor"] = make_stego_logits_processor(top_n, threshold, seed=seed)

    output_ids = model.generate(input_ids, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@torch.no_grad()
def sequence_perplexity(model, tokenizer, text: str, device: str) -> float:
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    encodings = {k: v.to(device) for k, v in encodings.items()}
    labels = encodings["input_ids"].clone()
    outputs = model(**encodings, labels=labels)
    return math.exp(outputs.loss.item())


def run_generation_demo(
    model_id: str,
    demo_cfg: dict,
    stego_cfg: dict,
    run_dir: Path,
    summary_lines: list[str],
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = demo_cfg["prompt"]
    max_new_tokens = demo_cfg["max_new_tokens"]
    top_n = stego_cfg["top_n"]
    threshold = stego_cfg["threshold"]
    seed = stego_cfg.get("seed")

    print(f"Model: {model_id}  |  device: {device}")
    model, tokenizer = load_model_and_tokenizer(model_id, device)

    baseline = generate_text(
        model, tokenizer, prompt,
        device=device, max_new_tokens=max_new_tokens,
        top_n=top_n, threshold=None, seed=seed,
    )
    constrained = generate_text(
        model, tokenizer, prompt,
        device=device, max_new_tokens=max_new_tokens,
        top_n=top_n, threshold=threshold, seed=seed,
    )

    baseline_ppl = sequence_perplexity(model, tokenizer, baseline, device)
    constrained_ppl = sequence_perplexity(model, tokenizer, constrained, device)

    result = {
        "prompt": prompt,
        "top_n": top_n,
        "threshold": threshold,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "baseline": {"text": baseline, "perplexity": baseline_ppl},
        "dummy_stego": {"text": constrained, "perplexity": constrained_ppl},
    }
    save_json(run_dir / "generation_demo.json", result)

    print("\n=== Generation demo ===")
    print(f"Prompt: {prompt!r}")
    print(f"top_n={top_n}, threshold={threshold}, seed={seed}\n")
    print("[baseline]", baseline, sep="\n")
    print(f"perplexity: {baseline_ppl:.2f}\n")
    print("[dummy stego]", constrained, sep="\n")
    print(f"perplexity: {constrained_ppl:.2f}")

    summary_lines.extend([
        "",
        "=== demo ===",
        f"Prompt: {prompt!r}",
        f"top_n={top_n}, threshold={threshold}, seed={seed}",
        f"Baseline perplexity: {baseline_ppl:.4f}",
        f"Dummy stego perplexity: {constrained_ppl:.4f}",
        f"Perplexity delta: {constrained_ppl - baseline_ppl:+.4f}",
    ])
    return result


def run_quality_sweep(
    model_id: str,
    sweep_cfg: dict,
    stego_cfg: dict,
    run_dir: Path,
    summary_lines: list[str],
) -> dict:
    """Generate text at multiple thresholds; measure perplexity degradation vs baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = sweep_cfg["prompts"]
    max_new_tokens = sweep_cfg["max_new_tokens"]
    thresholds = stego_cfg["thresholds"]
    top_n = stego_cfg["top_n"]
    seed = stego_cfg.get("seed")

    print(f"Model: {model_id}  |  device: {device}")
    model, tokenizer = load_model_and_tokenizer(model_id, device)

    per_prompt: list[dict] = []
    threshold_totals: dict[str, list[float]] = {str(t): [] for t in thresholds}
    threshold_deltas: dict[str, list[float]] = {str(t): [] for t in thresholds}

    print("\n=== Quality sweep (perplexity vs threshold) ===")
    for prompt in prompts:
        print(f"\n--- prompt: {prompt!r} ---")
        baseline_text = generate_text(
            model, tokenizer, prompt,
            device=device, max_new_tokens=max_new_tokens,
            top_n=top_n, threshold=None, seed=seed,
        )
        baseline_ppl = sequence_perplexity(model, tokenizer, baseline_text, device)
        print(f"baseline ppl: {baseline_ppl:.2f}")

        by_threshold: dict[str, dict] = {}
        for threshold in thresholds:
            text = generate_text(
                model, tokenizer, prompt,
                device=device, max_new_tokens=max_new_tokens,
                top_n=top_n, threshold=threshold, seed=seed,
            )
            ppl = sequence_perplexity(model, tokenizer, text, device)
            delta = ppl - baseline_ppl
            key = str(threshold)
            threshold_totals[key].append(ppl)
            threshold_deltas[key].append(delta)
            by_threshold[key] = {
                "threshold": threshold,
                "text": text,
                "perplexity": ppl,
                "delta_vs_baseline": delta,
                "ratio_vs_baseline": ppl / baseline_ppl if baseline_ppl > 0 else None,
            }
            print(f"  threshold={threshold}: ppl={ppl:.2f}  (delta {delta:+.2f})")

        per_prompt.append({
            "prompt": prompt,
            "baseline": {"text": baseline_text, "perplexity": baseline_ppl},
            "by_threshold": by_threshold,
        })

    aggregate = {}
    for threshold in thresholds:
        key = str(threshold)
        ppls = threshold_totals[key]
        deltas = threshold_deltas[key]
        aggregate[key] = {
            "threshold": threshold,
            "mean_perplexity": sum(ppls) / len(ppls),
            "mean_delta_vs_baseline": sum(deltas) / len(deltas),
            "mean_ratio_vs_baseline": sum(
                per_prompt[i]["by_threshold"][key]["ratio_vs_baseline"]
                for i in range(len(per_prompt))
            ) / len(per_prompt),
        }

    result = {
        "prompts": prompts,
        "max_new_tokens": max_new_tokens,
        "top_n": top_n,
        "seed": seed,
        "thresholds": thresholds,
        "per_prompt": per_prompt,
        "aggregate": aggregate,
    }
    save_json(run_dir / "quality_sweep.json", result)

    summary_lines.extend([
        "",
        "=== quality_sweep (średnia perplexity / delta vs baseline) ===",
        f"Promptów: {len(prompts)}, top_n={top_n}, seed={seed}",
        f"{'threshold':>10} | {'avg_ppl':>8} | {'avg_delta':>10} | {'avg_ratio':>10}",
    ])
    for threshold in thresholds:
        key = str(threshold)
        agg = aggregate[key]
        summary_lines.append(
            f"{threshold:>10} | {agg['mean_perplexity']:>8.2f} | "
            f"{agg['mean_delta_vs_baseline']:>+10.2f} | {agg['mean_ratio_vs_baseline']:>10.2f}x"
        )

    print("\n--- średnia po promptach ---")
    print(f"{'threshold':>10} | {'avg_ppl':>8} | {'avg_delta':>10} | {'avg_ratio':>10}")
    for threshold in thresholds:
        key = str(threshold)
        agg = aggregate[key]
        print(
            f"{threshold:>10} | {agg['mean_perplexity']:>8.2f} | "
            f"{agg['mean_delta_vs_baseline']:>+10.2f} | {agg['mean_ratio_vs_baseline']:>10.2f}x"
        )
    return result


def run_threshold_sweep(
    model_id: str,
    harness_cfg: dict,
    stego_cfg: dict,
    run_dir: Path,
    summary_lines: list[str],
) -> dict:
    if DummyStegoHFLM is None:
        raise RuntimeError("lm-evaluation-harness is not installed. Run: pip install lm-eval")

    from lm_eval import evaluator

    import dummy_stego_processor  # noqa: F401

    prepare_harness_env(harness_cfg)
    confirm_unsafe = bool(harness_cfg.get("allow_code_eval", False))

    thresholds = stego_cfg["thresholds"]
    top_n = stego_cfg["top_n"]
    seed = stego_cfg.get("seed")
    tasks = harness_cfg["tasks"]
    batch_size = harness_cfg.get("batch_size", 1)
    limit = harness_cfg.get("limit")

    harness_dir = run_dir / "harness"
    harness_dir.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    summary_lines.extend([
        "",
        "=== harness ===",
        f"Tasks: {', '.join(tasks)}",
        f"Thresholds: {thresholds}",
    ])

    lm = DummyStegoHFLM(
        pretrained=model_id,
        top_n=top_n,
        threshold=thresholds[0],
        stego_seed=seed,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for threshold in thresholds:
        print(f"\n=== lm-eval | threshold={threshold} ===")
        lm.threshold = threshold
        eval_result = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=batch_size,
            limit=limit,
            confirm_run_unsafe_code=confirm_unsafe,
        )

        threshold_key = str(threshold)
        payload = {
            "threshold": threshold,
            "top_n": top_n,
            "seed": seed,
            "tasks": tasks,
            "batch_size": batch_size,
            "limit": limit,
            "results": eval_result.get("results", {}),
            "configs": eval_result.get("configs", {}),
            "versions": eval_result.get("versions", {}),
            "n-shot": eval_result.get("n-shot", {}),
            "higher_is_better": eval_result.get("higher_is_better", {}),
            "n-samples": eval_result.get("n-samples", {}),
        }
        results[threshold_key] = payload
        save_json(harness_dir / f"threshold_{threshold_key.replace('.', '_')}.json", payload)

        for task_name, task_metrics in payload["results"].items():
            print(f"{task_name}: {json.dumps(task_metrics, indent=2)}")
            summary_lines.append(f"[threshold={threshold}] {task_name}")
            for metric_name, value in sorted(task_metrics.items()):
                summary_lines.append(f"  {metric_name}: {value}")

    save_json(run_dir / "harness_sweep.json", results)
    return results


def harness_available() -> bool:
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        return False


def require_harness() -> None:
    if DummyStegoHFLM is not None:
        return
    if not harness_available():
        raise RuntimeError(
            'Test "harness" wymaga lm-evaluation-harness. Zainstaluj: pip install lm-eval'
        )
    raise RuntimeError(
        'Test "harness" niedostępny — błąd integracji DummyStegoHFLM z lm-eval.'
    )


def run_benchmark(config_path: Path) -> Path:
    config = load_config(config_path)
    run_cfg = config.get("run", {})
    tests = run_cfg.get("tests", ["demo"])
    valid_tests = {"demo", "harness", "quality_sweep"}
    unknown = set(tests) - valid_tests
    if unknown:
        raise ValueError(f"Unknown tests: {unknown}. Use: {valid_tests}")

    if "harness" in tests:
        require_harness()

    model_key = config.get("model", {}).get("key")
    if not model_key:
        raise ValueError("Missing [model].key in config")

    model_id = resolve_model_id(model_key)
    stego_cfg = config.get("stego", {})
    demo_cfg = config.get("demo", {})
    harness_cfg = config.get("harness", {})
    quality_cfg = config.get("quality_sweep", {})

    if "demo" in tests and "threshold" not in stego_cfg:
        raise ValueError('Test "demo" requires [stego].threshold')
    if "harness" in tests and "thresholds" not in stego_cfg:
        raise ValueError('Test "harness" requires [stego].thresholds')
    if "quality_sweep" in tests:
        if "thresholds" not in stego_cfg:
            raise ValueError('Test "quality_sweep" requires [stego].thresholds')
        if "prompts" not in quality_cfg:
            raise ValueError('Test "quality_sweep" requires [quality_sweep].prompts')

    run_dir = create_run_dir(resolve_results_dir(config))
    shutil.copy2(config_path, run_dir / "config.toml")

    summary_lines = [
        f"Run: {run_dir.name}",
        f"Config: {config_path}",
        f"Tests: {', '.join(tests)}",
        f"Model: {model_id}",
    ]
    save_json(
        run_dir / "run_config.json",
        build_run_metadata(config_path, config, model_id, tests),
    )

    print(f"Results dir: {run_dir}")

    if "demo" in tests:
        run_generation_demo(model_id, demo_cfg, stego_cfg, run_dir, summary_lines)

    if "quality_sweep" in tests:
        run_quality_sweep(model_id, quality_cfg, stego_cfg, run_dir, summary_lines)

    if "harness" in tests:
        run_threshold_sweep(model_id, harness_cfg, stego_cfg, run_dir, summary_lines)

    write_summary(run_dir, summary_lines)
    print(f"\nAll artifacts saved under: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dummy-stego benchmarks from a TOML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to benchmark TOML (default: {DEFAULT_CONFIG}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    run_benchmark(config_path)


if __name__ == "__main__":
    main()
