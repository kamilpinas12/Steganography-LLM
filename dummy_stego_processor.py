"""Dummy steganography logits processor — simulates rank-token constraints without hiding data."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

FORCE_LOGIT_BIAS = 1e4


class DummyStegoLogitsProcessor(LogitsProcessor):
    """Restricts generation to top-N tokens above a probability threshold, then forces a uniform random pick.

    Mimics rank-token steganography: the secret bits would choose one of the surviving tokens.
    Here we draw that token uniformly at random. When fewer than two tokens survive the filters,
    logits are left unchanged so decoding stays natural.
    """

    def __init__(self, top_n: int, threshold: float, seed: int | None = None):
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.top_n = top_n
        self.threshold = threshold
        self._seed = seed
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _rng(self, device: torch.device) -> torch.Generator | None:
        if self._seed is None:
            return None
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._seed)
            self._generator_device = device
        return self._generator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab_size]
        probs = torch.softmax(scores, dim=-1)
        k = min(self.top_n, probs.shape[-1])
        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)

        # top-N pool, then probability threshold
        survivor_mask = top_probs >= self.threshold
        survivor_count = survivor_mask.sum(dim=-1)

        # >= 2 survivors: force a uniform random token (stego simulation)
        apply_stego = survivor_count >= 2
        if not apply_stego.any():
            return scores

        rand = torch.rand(
            top_probs.shape,
            device=scores.device,
            dtype=scores.dtype,
            generator=self._rng(scores.device),
        )
        rand = rand.masked_fill(~survivor_mask, -1.0)
        chosen_pos = rand.argmax(dim=-1, keepdim=True)
        chosen_token = top_indices.gather(-1, chosen_pos).squeeze(-1)

        forced_scores = torch.full_like(scores, float("-inf"))
        bias = scores.max(dim=-1, keepdim=True).values + FORCE_LOGIT_BIAS
        forced_scores.scatter_(-1, chosen_token.unsqueeze(-1), bias)

        return torch.where(apply_stego.unsqueeze(-1), forced_scores, scores)


def make_stego_logits_processor(
    top_n: int,
    threshold: float,
    seed: int | None = None,
) -> LogitsProcessorList:
    return LogitsProcessorList([DummyStegoLogitsProcessor(top_n, threshold, seed=seed)])


try:
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM

    @register_model("dummy-stego-hf")
    class DummyStegoHFLM(HFLM):
        """Hugging Face LM wrapper for lm-evaluation-harness with DummyStegoLogitsProcessor injected."""

        def __init__(
            self,
            top_n: int = 15,
            threshold: float = 0.01,
            stego_seed: int | None = 42,
            **kwargs,
        ):
            self.top_n = int(top_n)
            self.threshold = float(threshold)
            self.stego_seed = None if stego_seed is None else int(stego_seed)
            super().__init__(**kwargs)

        def _model_generate(self, context, max_length: int, stop: list[str], **generation_kwargs):
            stego_processor = make_stego_logits_processor(
                self.top_n, self.threshold, seed=self.stego_seed
            )
            existing = generation_kwargs.get("logits_processor")
            if existing is None:
                generation_kwargs["logits_processor"] = stego_processor
            else:
                processors = list(existing) if isinstance(existing, LogitsProcessorList) else [existing]
                generation_kwargs["logits_processor"] = LogitsProcessorList(
                    processors + list(stego_processor)
                )
            return super()._model_generate(context, max_length, stop, **generation_kwargs)

except ImportError:
    DummyStegoHFLM = None  # type: ignore[misc, assignment]
