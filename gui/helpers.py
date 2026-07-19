import gc
import os
from pathlib import Path
from typing import List, Tuple, TypeAlias

Token: TypeAlias = int

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

AVAILABLE_MODELS: dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "stablelm-1.6b": "stabilityai/stablelm-2-zephyr-1_6b",
    "phi-2": "microsoft/phi-2",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gpt2": "openai-community/gpt2",
    "distilgpt2": "distilgpt2",
}

DEFAULT_MODEL_KEY = "tinyllama"


def resolve_model_id(model_key_or_id: str) -> str:
    return AVAILABLE_MODELS.get(model_key_or_id, model_key_or_id)


def _local_model_path(model_name: str) -> Path:
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return MODELS_DIR / safe_name


def resolve_model_load_path(model_name: str) -> str:
    local_path = _local_model_path(model_name)
    if (local_path / "config.json").exists():
        return str(local_path)
    os.makedirs(local_path, exist_ok=True)
    return model_name


def load_tokenizer(model_key_or_id: str):
    """Load tokenizer only (no LLM weights)."""
    from transformers import AutoTokenizer

    model_name = resolve_model_id(model_key_or_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def release_gpu_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_model_device(model: "ModelWrapper"):
    import torch

    return next(model.model.parameters()).device


class ModelWrapper:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM

        self.model_name = resolve_model_id(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_path = resolve_model_load_path(self.model_name)

        self.tokenizer = load_tokenizer(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()

        if load_path == self.model_name:
            local_path = _local_model_path(self.model_name)
            self.tokenizer.save_pretrained(local_path)
            self.model.save_pretrained(local_path)

    def step(self, context: List[Token], threshold: float) -> List[Tuple[Token, float]]:
        import torch

        input_ids = torch.tensor([context], device=get_model_device(self))
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        mask = probs >= threshold
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return []

        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        result = [(int(token_id), float(probs[token_id].item())) for token_id in indices.tolist()]
        result.sort(key=lambda item: item[1], reverse=True)
        return result


def str2tokens(model: ModelWrapper, string: str) -> List[Token]:
    return model.tokenizer.encode(string, add_special_tokens=True)


def tokens2str(model: ModelWrapper, tokens: List[Token]) -> str:
    return model.tokenizer.decode(tokens, skip_special_tokens=True)
