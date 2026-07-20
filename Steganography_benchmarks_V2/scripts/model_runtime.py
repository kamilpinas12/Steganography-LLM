"""Model loading and generation (no evaluation)."""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dummy_processor import StegoCapacityStats, make_stego_logits_processor

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent  # Steganography_benchmarks_V2/
MODEL_CACHE_ROOT = REPO_ROOT / "models_cache"
OFFLINE_MODE = False
NO_PERSISTENT_CACHE = False


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_offline_mode() -> None:
    global OFFLINE_MODE
    OFFLINE_MODE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Offline mode enabled (local cache only, no Hugging Face).")


def hf_login_if_needed() -> None:
    if OFFLINE_MODE:
        print("Offline mode: skipping Hugging Face login.")
        return
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Logged in to Hugging Face from environment token.")
    else:
        print("HF token not found in env. Gated models may fail to download.")


def configure_model_cache(
    cache_dir: Path | None = None,
    *,
    persist: bool = True,
) -> Path:
    """Colab: persist snapshots on Drive. Kaggle: temp dir only (no persistent save)."""
    global MODEL_CACHE_ROOT, NO_PERSISTENT_CACHE

    if not persist:
        NO_PERSISTENT_CACHE = True
        MODEL_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="stego_v2_hf_nocache_"))
        hf_home = MODEL_CACHE_ROOT / "hf_home"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
        print(f"No persistent model cache — temporary HF dir: {MODEL_CACHE_ROOT}")
        return MODEL_CACHE_ROOT

    NO_PERSISTENT_CACHE = False
    if cache_dir is not None:
        MODEL_CACHE_ROOT = cache_dir
    elif os.getenv("MODEL_CACHE_DIR"):
        MODEL_CACHE_ROOT = Path(os.environ["MODEL_CACHE_DIR"])
    else:
        MODEL_CACHE_ROOT = REPO_ROOT / "models_cache"

    MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    hf_home = MODEL_CACHE_ROOT / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    print(f"Model cache directory: {MODEL_CACHE_ROOT}")
    return MODEL_CACHE_ROOT


def configure_platform(platform: str, model_cache_dir: Path | None = None) -> Path:
    if platform == "kaggle":
        print("Platform=kaggle: always download from HF, no persistent model snapshots.")
        return configure_model_cache(model_cache_dir, persist=False)
    if platform == "colab":
        print("Platform=colab: persistent model cache enabled.")
        return configure_model_cache(model_cache_dir, persist=True)
    raise ValueError(f"Unknown platform: {platform!r} (use 'colab' or 'kaggle')")


def _slugify_model(model_id: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.replace("/", "__")).strip("_")


def _cached_model_dir(model_id: str) -> Path:
    return MODEL_CACHE_ROOT / "models" / _slugify_model(model_id)


def _pretrained_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if OFFLINE_MODE:
        kwargs["local_files_only"] = True
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            kwargs["token"] = token
    return kwargs


def resolve_model_source(model_id: str) -> str:
    if NO_PERSISTENT_CACHE:
        if OFFLINE_MODE:
            raise RuntimeError("Kaggle/no-cache mode cannot be used together with --offline")
        print(f"Loading from Hugging Face Hub (no persistent cache): {model_id}")
        return model_id

    local_dir = _cached_model_dir(model_id)
    if (local_dir / "config.json").exists():
        print(f"Using cached model: {local_dir}")
        return str(local_dir)

    if OFFLINE_MODE:
        raise FileNotFoundError(
            f"Offline mode: model not in cache: {model_id}\nExpected: {local_dir}"
        )

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    print(f"Downloading {model_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"Model saved to cache: {local_dir}")
    return str(local_dir)


def build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_tokenizer(model_id: str):
    source = resolve_model_source(model_id)
    pretrained_kwargs = _pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(source, **pretrained_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=build_quant_config(),
        device_map="auto",
        **pretrained_kwargs,
    )
    model.eval()
    return model, tokenizer


def release_model(model, tokenizer) -> None:
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


@torch.no_grad()
def generate_raw(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    threshold: float,
    top_n: int,
    seed: int,
) -> tuple[str, str, StegoCapacityStats]:
    """Return (full_decoded, raw_completion_after_prompt, capacity_stats)."""
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
    new_token_ids = output_ids[0, input_ids.shape[1] :]
    raw_completion = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decoded, raw_completion, capacity_stats
