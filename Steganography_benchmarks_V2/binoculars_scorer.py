"""Binoculars zero-shot AI detector for evaluation phase."""

from __future__ import annotations

import gc
from typing import Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_runtime import configure_platform, resolve_model_source

_CE_LOSS = torch.nn.CrossEntropyLoss(reduction="none")
_SOFTMAX = torch.nn.Softmax(dim=-1)

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527
BINOCULARS_OBSERVER_MODEL = "tiiuae/falcon-7b"
BINOCULARS_PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"

# T4 16GB: fp16 7B nie wchodzi — tylko 4-bit / 8-bit, sekwencyjnie
_MAX_GPU_GB_AFTER_LOAD = 7.5


def _pretrained_kwargs(*, trust_remote_code: bool = False) -> dict:
    import os

    kwargs: dict = {"trust_remote_code": trust_remote_code}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _bnb_8bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_8bit=True)


def _gpu_used_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def _model_device(model) -> torch.device:
    if hasattr(model, "device") and model.device.type != "meta":
        return model.device
    return next(model.parameters()).device


def _binoculars_perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
) -> np.ndarray:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    token_losses = _CE_LOSS(shifted_logits.transpose(1, 2), shifted_labels)
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
    p_proba = _SOFTMAX(observer_logits).view(-1, vocab_size)
    q_scores = performer_logits.view(-1, vocab_size)
    ce = _CE_LOSS(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    return ((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()


class BinocularsScorer:
    """Zero-shot Binoculars scorer with sequential quantized Falcon loading."""

    def __init__(
        self,
        observer_model_id: str = BINOCULARS_OBSERVER_MODEL,
        performer_model_id: str = BINOCULARS_PERFORMER_MODEL,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        *,
        platform: str = "colab",
    ) -> None:
        configure_platform(platform)
        self.observer_model_id = observer_model_id
        self.performer_model_id = performer_model_id
        self.max_token_observed = max_token_observed
        self.threshold = (
            BINOCULARS_FPR_THRESHOLD if mode == "low-fpr" else BINOCULARS_ACCURACY_THRESHOLD
        )
        source = resolve_model_source(observer_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **_pretrained_kwargs())
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
        if not torch.cuda.is_available():
            raise RuntimeError("Binoculars evaluation requires a GPU.")

        source = resolve_model_source(model_id)
        base_kwargs = {
            **_pretrained_kwargs(),
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        strategies: list[tuple[str, BitsAndBytesConfig]] = [
            ("4bit", _bnb_4bit_config()),
            ("8bit", _bnb_8bit_config()),
        ]

        last_error: Exception | None = None
        for label, quant_config in strategies:
            self._free_gpu()
            try:
                print(f"Loading {model_id} ({label})...", flush=True)
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    quantization_config=quant_config,
                    **base_kwargs,
                )
                used = _gpu_used_gb()
                if used > _MAX_GPU_GB_AFTER_LOAD:
                    print(
                        f"{label} looks unquantized ({used:.1f} GB VRAM) — trying next strategy.",
                        flush=True,
                    )
                    self._release_model(model)
                    continue
                print(f"Loaded {model_id} ({label}), VRAM ~{used:.1f} GB", flush=True)
                return model
            except (ValueError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                last_error = exc
                print(f"{label} load failed for {model_id}: {exc}", flush=True)
                self._free_gpu()

        raise RuntimeError(
            f"Could not load {model_id} in 4-bit or 8-bit on GPU "
            f"(last error: {last_error}). Try a GPU with more VRAM or restart runtime."
        ) from last_error

    @staticmethod
    def _free_gpu() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @classmethod
    def _release_model(cls, model) -> None:
        del model
        cls._free_gpu()

    @torch.inference_mode()
    def compute_score(self, input_text: Union[str, list[str]]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        if any(not text.strip() for text in batch):
            raise ValueError("Binoculars cannot score empty text.")

        encodings = self._tokenize(batch)
        encodings_cpu = {k: v.cpu() for k, v in encodings.items()}

        observer_model = self._load_model(self.observer_model_id)
        observer_device = _model_device(observer_model)
        encodings_gpu = {k: v.to(observer_device) for k, v in encodings_cpu.items()}
        observer_logits = observer_model(**encodings_gpu).logits.detach().cpu()
        self._release_model(observer_model)
        del encodings_gpu

        performer_model = self._load_model(self.performer_model_id)
        performer_device = _model_device(performer_model)
        encodings_gpu = {k: v.to(performer_device) for k, v in encodings_cpu.items()}
        performer_logits = performer_model(**encodings_gpu).logits.detach().cpu()
        self._release_model(performer_model)
        del encodings_gpu

        ppl = _binoculars_perplexity(encodings_cpu, performer_logits)
        x_ppl = _binoculars_cross_perplexity(
            observer_logits,
            performer_logits,
            encodings_cpu,
            self.tokenizer.pad_token_id,
        )
        scores = (ppl / x_ppl).tolist()
        return scores[0] if isinstance(input_text, str) else scores

    def prediction_label(self, score: float) -> str:
        return (
            "Most likely AI-generated"
            if score < self.threshold
            else "Most likely human-generated"
        )
