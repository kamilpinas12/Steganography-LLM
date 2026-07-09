from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import transformers
from huggingface_hub import login
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dummy_processor import StegoCapacityStats, make_stego_logits_processor


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"

MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

TESTS = ("humaneval", "perplexity", "capacity", "binoculars")

DEFAULT_THRESHOLDS = [0.0, 0.01, 0.05, 0.1]
DEFAULT_TOP_N = 15
DEFAULT_SEED = 1234

PERPLEXITY_PROMPTS = [
    "Write a short paragraph explaining how binary search works.",
    "Describe the difference between a stack and a queue in computer science.",
    "Explain what a hash table is and when you would use one.",
    "Write a concise summary of how gradient descent optimizes neural networks.",
]

PYTHON_CODE_BLOCK_RE = re.compile(
    r"```python\s*\n?(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
GENERIC_CODE_BLOCK_RE = re.compile(
    r"```(?!python)\s*\n?(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
PYTHON_CODE_BLOCK_OPEN_RE = re.compile(
    r"```python\s*\n?(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)
GENERIC_CODE_BLOCK_OPEN_RE = re.compile(
    r"```(?!python)\s*\n?(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)
TRAILING_FENCE_RE = re.compile(r"\n?```\s*$")

REASONING_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
REASONING_OPEN_TO_FENCE_RE = re.compile(
    r"<think>.*?(?=```)",
    flags=re.DOTALL | re.IGNORECASE,
)
REASONING_OPEN_TO_END_RE = re.compile(
    r"<think>.*",
    flags=re.DOTALL | re.IGNORECASE,
)

CHAT_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|im_start\|>assistant\s*", re.IGNORECASE),
    re.compile(r"<\|redacted_im_end\|>\s*", re.IGNORECASE),
    re.compile(r"<\|eot_id\|>\s*", re.IGNORECASE),
    re.compile(r"<eos>\s*", re.IGNORECASE),
    re.compile(r"<start_of_turn>model\s*", re.IGNORECASE),
)

IMPORT_FALLBACK_RE = re.compile(
    r"(?:^\s*(?:from\s+\S+\s+import|import\s+\S+)|\bdef\s+)",
    flags=re.MULTILINE,
)

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
    "Natural_Fallback_Steps",
    "Binoculars_Score",
    "Baseline_Binoculars_Score",
    "Binoculars_Score_Delta",
    "AI_Detection_Rate",
    "Baseline_AI_Detection_Rate",
]


@dataclass(frozen=True)
class RunConfig:
    test: str
    model_key: str
    model_id: str
    threshold: float
    top_n: int
    max_new_tokens: int
    seed: int
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single steganography benchmark (model + test + threshold).",
    )
    parser.add_argument(
        "--test",
        required=True,
        choices=TESTS,
        help="Benchmark type: humaneval | perplexity | capacity | binoculars",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model key ({', '.join(MODELS)}) or full Hugging Face model id",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        help="Threshold value (e.g. 0.01) or 'all' to run every default threshold",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory for run artifacts",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-tests", action="store_true")
    return parser.parse_args()


def resolve_model(model_arg: str) -> tuple[str, str]:
    if model_arg in MODELS:
        return model_arg, MODELS[model_arg]
    return model_arg, model_arg


def parse_thresholds(threshold_arg: str) -> list[float]:
    if threshold_arg.lower() == "all":
        return list(DEFAULT_THRESHOLDS)
    return [float(threshold_arg)]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hf_login_if_needed() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Logged in to Hugging Face from environment token.")
    else:
        print("HF token not found in env. Gated models may fail to download.")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def create_run_dir(
    output_root: Path,
    test: str,
    model_key: str,
    threshold: float,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    threshold_slug = str(threshold).replace(".", "_")
    run_name = f"{timestamp}_{slugify(model_key)}_{test}_th{threshold_slug}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


# --- Binoculars zero-shot detector (inlined for Colab: no extra file needed) ---

_BINO_CE_LOSS = torch.nn.CrossEntropyLoss(reduction="none")
_BINO_SOFTMAX = torch.nn.Softmax(dim=-1)
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527
BINOCULARS_OBSERVER_MODEL = "tiiuae/falcon-7b"
BINOCULARS_PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"


def _binoculars_perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
) -> np.ndarray:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    token_losses = _BINO_CE_LOSS(shifted_logits.transpose(1, 2), shifted_labels)
    ppl = (token_losses * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    return ppl.to("cpu").float().numpy()


def _binoculars_cross_perplexity(
    observer_logits: torch.Tensor,
    performer_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
) -> np.ndarray:
    vocab_size = observer_logits.shape[-1]
    total_tokens_available = performer_logits.shape[-2]
    p_proba = _BINO_SOFTMAX(observer_logits).view(-1, vocab_size)
    q_scores = performer_logits.view(-1, vocab_size)
    ce = _BINO_CE_LOSS(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    return ((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()


class BinocularsScorer:
    """Zero-shot Binoculars scorer with sequential 4-bit Falcon loading for single-GPU Colab."""

    def __init__(
        self,
        observer_model_id: str = BINOCULARS_OBSERVER_MODEL,
        performer_model_id: str = BINOCULARS_PERFORMER_MODEL,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ) -> None:
        self.observer_model_id = observer_model_id
        self.performer_model_id = performer_model_id
        self.max_token_observed = max_token_observed
        self.threshold = (
            BINOCULARS_FPR_THRESHOLD if mode == "low-fpr" else BINOCULARS_ACCURACY_THRESHOLD
        )
        self.tokenizer = AutoTokenizer.from_pretrained(observer_model_id, trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        return self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )

    def _load_model(self, model_id: str):
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=build_quant_config(),
            device_map="auto",
            trust_remote_code=True,
            token=token,
        )

    def _release_model(self, model) -> None:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    @torch.inference_mode()
    def compute_score(self, input_text: Union[str, list[str]]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        if any(not text.strip() for text in batch):
            raise ValueError("Binoculars cannot score empty text.")

        encodings = self._tokenize(batch)
        observer_model = self._load_model(self.observer_model_id)
        observer_device = observer_model.device
        encodings = {k: v.to(observer_device) for k, v in encodings.items()}
        observer_logits = observer_model(**encodings).logits.detach().cpu()
        self._release_model(observer_model)

        performer_model = self._load_model(self.performer_model_id)
        performer_device = performer_model.device
        encodings_performer = {k: v.to(performer_device) for k, v in encodings.items()}
        performer_logits = performer_model(**encodings_performer).logits.detach().cpu()
        self._release_model(performer_model)

        encodings_cpu = {k: v.cpu() for k, v in encodings.items()}
        ppl = _binoculars_perplexity(encodings_cpu, performer_logits)
        x_ppl = _binoculars_cross_perplexity(
            observer_logits,
            performer_logits,
            encodings_cpu,
            self.tokenizer.pad_token_id,
        )
        scores = (ppl / x_ppl).tolist()
        return scores[0] if isinstance(input_text, str) else scores


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_quant_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def release_model(model, tokenizer) -> None:
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return user_prompt
    return user_prompt


def build_humaneval_prompt(tokenizer, task_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding assistant. "
                "Complete only the function body and return valid Python code."
            ),
        },
        {"role": "user", "content": task_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return task_prompt
    return task_prompt


def extract_generated_part(full_text: str, prompt_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return full_text.strip()


def _strip_reasoning_tags(text: str) -> str:
    cleaned = REASONING_BLOCK_RE.sub("", text)
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = REASONING_OPEN_TO_FENCE_RE.sub("", cleaned, count=1)
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = REASONING_OPEN_TO_END_RE.sub("", cleaned, count=1)
    return cleaned


def _strip_chat_markers(text: str) -> str:
    cleaned = text
    for pattern in CHAT_MARKER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def _extract_markdown_code(text: str) -> str | None:
    for pattern in (PYTHON_CODE_BLOCK_RE, GENERIC_CODE_BLOCK_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    for pattern in (PYTHON_CODE_BLOCK_OPEN_RE, GENERIC_CODE_BLOCK_OPEN_RE):
        match = pattern.search(text)
        if match:
            body = TRAILING_FENCE_RE.sub("", match.group(1).strip())
            return body.strip()

    return None


def _extract_code_fallback(text: str) -> str:
    match = IMPORT_FALLBACK_RE.search(text)
    if match:
        return text[match.start() :].strip()
    return text.strip()


def extract_code_for_eval(text: str) -> str:
    """Normalize model output to raw Python for HumanEval Pass@1."""
    candidate = text.strip()
    if not candidate:
        return candidate

    candidate = _strip_reasoning_tags(candidate)
    candidate = _strip_chat_markers(candidate)

    markdown_code = _extract_markdown_code(candidate)
    if markdown_code is not None:
        return markdown_code

    return _extract_code_fallback(candidate)


def extract_completion(decoded_text: str, prompt_text: str) -> str:
    if decoded_text.startswith(prompt_text):
        candidate = decoded_text[len(prompt_text) :]
    else:
        candidate = decoded_text
    return extract_code_for_eval(candidate)


@torch.no_grad()
def generate_with_capacity(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    threshold: float,
    top_n: int,
    seed: int,
) -> tuple[str, StegoCapacityStats]:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    capacity_stats = StegoCapacityStats()
    if threshold > 0.0:
        logits_processor, stego_processor = make_stego_logits_processor(
            top_n=top_n,
            threshold=threshold,
            seed=seed,
            track_capacity=True,
        )
        generation_kwargs["logits_processor"] = logits_processor
        output_ids = model.generate(**generation_kwargs)
        capacity_stats = stego_processor.capacity_stats
    else:
        output_ids = model.generate(**generation_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded, capacity_stats


@torch.no_grad()
def sequence_perplexity(model, tokenizer, text: str) -> float:
    if not text.strip():
        return float("nan")
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    encodings = {k: v.to(model.device) for k, v in encodings.items()}
    labels = encodings["input_ids"].clone()
    outputs = model(**encodings, labels=labels)
    return math.exp(outputs.loss.item())


def load_humaneval_problems() -> list[dict[str, str]]:
    problems = read_problems()
    if len(problems) != 164:
        raise RuntimeError(f"Expected 164 HumanEval tasks, got {len(problems)}")
    return [problems[task_id] for task_id in sorted(problems.keys())]


def evaluate_pass_at_1(problems: list[dict[str, str]], predictions: list[str]) -> float:
    samples = [
        {
            "task_id": problem["task_id"],
            "completion": extract_code_for_eval(prediction),
        }
        for problem, prediction in zip(problems, predictions)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        sample_path = tmp.name
    try:
        write_jsonl(sample_path, samples)
        scores = evaluate_functional_correctness(
            sample_path,
            k=[1],
            n_workers=4,
            timeout=3.0,
        )
    finally:
        Path(sample_path).unlink(missing_ok=True)
    return float(scores["pass@1"])


def run_humaneval(config: RunConfig) -> dict[str, Any]:
    problems = load_humaneval_problems()
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(config.model_id)

    predictions: list[dict[str, str]] = []
    total_capacity = StegoCapacityStats()

    for problem in tqdm(problems, desc=f"{config.model_key} | humaneval | th={config.threshold}"):
        prompt_text = build_humaneval_prompt(tokenizer, problem["prompt"])
        decoded, capacity_stats = generate_with_capacity(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
            threshold=config.threshold,
            top_n=config.top_n,
            seed=config.seed,
        )
        completion = extract_completion(decoded, prompt_text)
        total_capacity = total_capacity.merge(capacity_stats)
        predictions.append(
            {
                "task_id": problem["task_id"],
                "completion": completion,
                "raw_completion": decoded[len(prompt_text) :] if decoded.startswith(prompt_text) else decoded,
                "capacity": capacity_stats.to_dict(),
            }
        )

    pass_at_1 = evaluate_pass_at_1(problems, [row["completion"] for row in predictions])
    release_model(model, tokenizer)

    return {
        "pass_at_1": pass_at_1,
        "capacity": total_capacity.to_dict(),
        "predictions": predictions,
    }


def run_perplexity(config: RunConfig) -> dict[str, Any]:
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(config.model_id)

    samples: list[dict[str, Any]] = []
    total_capacity = StegoCapacityStats()
    stego_perplexities: list[float] = []
    baseline_perplexities: list[float] = []

    for prompt in tqdm(
        PERPLEXITY_PROMPTS,
        desc=f"{config.model_key} | perplexity | th={config.threshold}",
    ):
        prompt_text = build_chat_prompt(tokenizer, prompt)

        baseline_text, _ = generate_with_capacity(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
            threshold=0.0,
            top_n=config.top_n,
            seed=config.seed,
        )
        stego_text, capacity_stats = generate_with_capacity(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
            threshold=config.threshold,
            top_n=config.top_n,
            seed=config.seed,
        )

        baseline_ppl = sequence_perplexity(model, tokenizer, baseline_text)
        stego_ppl = sequence_perplexity(model, tokenizer, stego_text)
        baseline_perplexities.append(baseline_ppl)
        stego_perplexities.append(stego_ppl)
        total_capacity = total_capacity.merge(capacity_stats)

        samples.append(
            {
                "prompt": prompt,
                "baseline_text": baseline_text,
                "stego_text": stego_text,
                "baseline_perplexity": baseline_ppl,
                "stego_perplexity": stego_ppl,
                "perplexity_delta": stego_ppl - baseline_ppl,
                "capacity": capacity_stats.to_dict(),
            }
        )

    release_model(model, tokenizer)
    mean_baseline = sum(baseline_perplexities) / len(baseline_perplexities)
    mean_stego = sum(stego_perplexities) / len(stego_perplexities)

    return {
        "perplexity": mean_stego,
        "baseline_perplexity": mean_baseline,
        "perplexity_delta": mean_stego - mean_baseline,
        "capacity": total_capacity.to_dict(),
        "samples": samples,
    }


def run_capacity(config: RunConfig) -> dict[str, Any]:
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(config.model_id)

    samples: list[dict[str, Any]] = []
    total_capacity = StegoCapacityStats()

    for prompt in tqdm(
        PERPLEXITY_PROMPTS,
        desc=f"{config.model_key} | capacity | th={config.threshold}",
    ):
        prompt_text = build_chat_prompt(tokenizer, prompt)
        generated_text, capacity_stats = generate_with_capacity(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
            threshold=config.threshold,
            top_n=config.top_n,
            seed=config.seed,
        )
        total_capacity = total_capacity.merge(capacity_stats)
        samples.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "capacity": capacity_stats.to_dict(),
            }
        )

    release_model(model, tokenizer)
    return {
        "capacity": total_capacity.to_dict(),
        "samples": samples,
    }


def run_binoculars(config: RunConfig) -> dict[str, Any]:
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(config.model_id)

    samples: list[dict[str, Any]] = []
    total_capacity = StegoCapacityStats()
    texts_to_score: list[tuple[str, str, str]] = []

    for prompt in tqdm(
        PERPLEXITY_PROMPTS,
        desc=f"{config.model_key} | generate for binoculars | th={config.threshold}",
    ):
        prompt_text = build_chat_prompt(tokenizer, prompt)

        baseline_text, _ = generate_with_capacity(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
            threshold=0.0,
            top_n=config.top_n,
            seed=config.seed,
        )
        stego_text, capacity_stats = generate_with_capacity(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=config.max_new_tokens,
            threshold=config.threshold,
            top_n=config.top_n,
            seed=config.seed,
        )
        total_capacity = total_capacity.merge(capacity_stats)

        baseline_generated = extract_generated_part(baseline_text, prompt_text)
        stego_generated = extract_generated_part(stego_text, prompt_text)
        texts_to_score.append((prompt, baseline_generated, stego_generated))

        samples.append(
            {
                "prompt": prompt,
                "baseline_text": baseline_text,
                "stego_text": stego_text,
                "baseline_generated": baseline_generated,
                "stego_generated": stego_generated,
                "capacity": capacity_stats.to_dict(),
            }
        )

    release_model(model, tokenizer)

    scorer = BinocularsScorer(mode="low-fpr")
    baseline_scores: list[float] = []
    stego_scores: list[float] = []
    baseline_flags: list[bool] = []
    stego_flags: list[bool] = []

    for prompt, baseline_generated, stego_generated in tqdm(
        texts_to_score,
        desc=f"{config.model_key} | binoculars scoring | th={config.threshold}",
    ):
        baseline_score = float(scorer.compute_score(baseline_generated))
        stego_score = float(scorer.compute_score(stego_generated))
        baseline_scores.append(baseline_score)
        stego_scores.append(stego_score)
        baseline_flags.append(baseline_score < scorer.threshold)
        stego_flags.append(stego_score < scorer.threshold)

        for sample in samples:
            if sample["prompt"] == prompt:
                sample["baseline_binoculars_score"] = baseline_score
                sample["stego_binoculars_score"] = stego_score
                sample["baseline_prediction"] = (
                    "Most likely AI-generated"
                    if baseline_score < scorer.threshold
                    else "Most likely human-generated"
                )
                sample["stego_prediction"] = (
                    "Most likely AI-generated"
                    if stego_score < scorer.threshold
                    else "Most likely human-generated"
                )
                break

    mean_baseline = sum(baseline_scores) / len(baseline_scores)
    mean_stego = sum(stego_scores) / len(stego_scores)
    baseline_ai_rate = sum(baseline_flags) / len(baseline_flags)
    stego_ai_rate = sum(stego_flags) / len(stego_flags)

    return {
        "binoculars_score": mean_stego,
        "baseline_binoculars_score": mean_baseline,
        "binoculars_score_delta": mean_stego - mean_baseline,
        "ai_detection_rate": stego_ai_rate,
        "baseline_ai_detection_rate": baseline_ai_rate,
        "binoculars_threshold": scorer.threshold,
        "capacity": total_capacity.to_dict(),
        "samples": samples,
    }


def build_summary_row(config: RunConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    capacity = metrics.get("capacity", {})
    return {
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "Run_Dir": str(config.run_dir),
        "Test": config.test,
        "Model_Key": config.model_key,
        "Model_ID": config.model_id,
        "Threshold": config.threshold,
        "Top_N": config.top_n,
        "Pass@1": metrics.get("pass_at_1", ""),
        "Perplexity": metrics.get("perplexity", ""),
        "Baseline_Perplexity": metrics.get("baseline_perplexity", ""),
        "Perplexity_Delta": metrics.get("perplexity_delta", ""),
        "Avg_Pool_Size": capacity.get("avg_pool_size", ""),
        "Avg_Pool_Size_Stego_Only": capacity.get("avg_pool_size_stego_only", ""),
        "Avg_BPT": capacity.get("avg_bits_per_token", ""),
        "Embedding_Rate": capacity.get("embedding_rate", ""),
        "Total_Generation_Steps": capacity.get("total_steps", ""),
        "Stego_Steps": capacity.get("stego_applied_steps", ""),
        "Natural_Fallback_Steps": capacity.get("natural_fallback_steps", ""),
        "Binoculars_Score": metrics.get("binoculars_score", ""),
        "Baseline_Binoculars_Score": metrics.get("baseline_binoculars_score", ""),
        "Binoculars_Score_Delta": metrics.get("binoculars_score_delta", ""),
        "AI_Detection_Rate": metrics.get("ai_detection_rate", ""),
        "Baseline_AI_Detection_Rate": metrics.get("baseline_ai_detection_rate", ""),
    }


def persist_run(config: RunConfig, metrics: dict[str, Any]) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.run_dir / "run_config.json", asdict(config) | {"run_dir": str(config.run_dir)})
    save_json(config.run_dir / "summary.json", metrics)

    if config.test == "humaneval":
        save_json(config.run_dir / "predictions.json", metrics["predictions"])
    elif config.test == "perplexity":
        save_json(config.run_dir / "perplexity_samples.json", metrics["samples"])
    elif config.test == "capacity":
        save_json(config.run_dir / "capacity_samples.json", metrics["samples"])
    elif config.test == "binoculars":
        save_json(config.run_dir / "binoculars_samples.json", metrics["samples"])

    summary_row = build_summary_row(config, metrics)
    append_summary_row(SUMMARY_CSV, summary_row)

    print(f"\nSaved run artifacts to: {config.run_dir}")
    print(f"Appended summary row to: {SUMMARY_CSV}")
    print(json.dumps(summary_row, indent=2, ensure_ascii=False))


def run_single(config: RunConfig) -> dict[str, Any]:
    if config.test == "humaneval":
        return run_humaneval(config)
    if config.test == "perplexity":
        return run_perplexity(config)
    if config.test == "capacity":
        return run_capacity(config)
    if config.test == "binoculars":
        return run_binoculars(config)
    raise ValueError(f"Unknown test: {config.test}")


def main() -> None:
    args = parse_args()

    if args.list_models:
        for key, model_id in MODELS.items():
            print(f"{key}: {model_id}")
        return

    if args.list_tests:
        for test_name in TESTS:
            print(test_name)
        return

    hf_login_if_needed()
    ensure_summary_csv(SUMMARY_CSV)

    model_key, model_id = resolve_model(args.model)
    thresholds = parse_thresholds(args.threshold)

    for threshold in thresholds:
        run_dir = create_run_dir(args.output_root, args.test, model_key, threshold)
        config = RunConfig(
            test=args.test,
            model_key=model_key,
            model_id=model_id,
            threshold=threshold,
            top_n=args.top_n,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            run_dir=run_dir,
        )

        print(
            f"\n=== Running test={config.test} | model={config.model_key} "
            f"({config.model_id}) | threshold={config.threshold} | top_n={config.top_n} ==="
        )
        metrics = run_single(config)
        persist_run(config, metrics)


if __name__ == "__main__":
    main()
