"""Steganography engine — tokenizer always resident; LLM loaded only on demand."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from helpers import ModelWrapper, Token, get_model_device, load_tokenizer, release_gpu_memory
from llm_steganography import (
    EOM_BYTE,
    _extract_bits_from_token,
    _init_prng,
    _next_token_distribution,
    _pick_token_for_bits,
    _pick_filler_token,
    _safe_tokens,
    bits_to_bytes,
    bytes_to_bits,
)
from shared_config import EOS_THRESHOLD, MAX_RESPONSE_LENGTH, MODEL_KEY, PASSWORD, THRESHOLD, TOP_N

if TYPE_CHECKING:
    ProgressCallback = Callable[[int, int], None]


class StegoEngine:
    def __init__(self) -> None:
        self.model: ModelWrapper | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def tokenizer_ready(self) -> bool:
        return self._tokenizer is not None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def load_tokenizer(self, model_key: str | None = None) -> None:
        key = model_key or MODEL_KEY
        self._tokenizer = load_tokenizer(key)

    def load_model(self, model_key: str | None = None) -> None:
        if self.model is not None:
            return
        key = model_key or MODEL_KEY
        self.model = ModelWrapper(key)
        self._tokenizer = self.model.tokenizer

    def release_model(self) -> None:
        if self.model is None:
            release_gpu_memory()
            return
        if getattr(self.model, "model", None) is not None:
            del self.model.model
        del self.model
        self.model = None
        release_gpu_memory()

    def tokenize(self, text: str) -> list[Token]:
        self._require_tokenizer()
        assert self._tokenizer is not None
        return self._tokenizer.encode(text, add_special_tokens=True)

    def detokenize(self, token_ids: list[Token]) -> str:
        self._require_tokenizer()
        assert self._tokenizer is not None
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode_from_context(
        self,
        context_token_ids: list[Token],
        secret: str,
        *,
        password: str = PASSWORD,
        threshold: float = THRESHOLD,
        eos_threshold: float = EOS_THRESHOLD,
        top_n: int = TOP_N,
        max_steps: int = MAX_RESPONSE_LENGTH,
        on_progress: ProgressCallback | None = None,
    ) -> list[Token]:
        self._require_model()
        model = self.model
        assert model is not None

        _init_prng(password)
        secret_bits = bytes_to_bits(secret.encode("utf-8") + EOM_BYTE)
        total_bits = len(secret_bits)

        context = list(context_token_ids)
        carrier_token_ids: list[Token] = []
        bit_index = 0
        eos_id = model.tokenizer.eos_token_id
        tail_steps = 0
        min_tail_steps = 12

        encoding_done = False

        for _ in range(max_steps):
            top_indices, top_probs, probs = _next_token_distribution(model, context, top_n)
            eos_prob = float(probs[eos_id].item()) if eos_id is not None else 0.0

            safe = _safe_tokens(
                top_indices,
                top_probs,
                threshold,
                strip_eos=not encoding_done,
                eos_token_id=eos_id,
            )

            if bit_index >= len(secret_bits):
                encoding_done = True
                # After payload is done, keep generating likely text until model is ready to end.
                if eos_prob >= eos_threshold:
                    break
                chosen = _pick_filler_token(top_indices, eos_id)
                context.append(chosen)
                carrier_token_ids.append(chosen)
                tail_steps += 1
                token_text = model.tokenizer.decode([chosen], skip_special_tokens=False)
                if chosen == eos_id:
                    break
                if tail_steps >= min_tail_steps and any(p in token_text for p in ".!?"):
                    break
                continue

            if not safe:
                # No token above threshold — advance context with top-1 (no payload bits).
                chosen = _pick_filler_token(top_indices, eos_id)
            else:
                chosen, bit_index = _pick_token_for_bits(safe, secret_bits, bit_index)

            context.append(chosen)
            carrier_token_ids.append(chosen)

            if on_progress is not None:
                on_progress(min(bit_index, total_bits), total_bits)

            if chosen == eos_id and bit_index < len(secret_bits):
                raise RuntimeError("Model produced EOS before secret was fully embedded.")
            if chosen == eos_id:
                break

        if bit_index < len(secret_bits):
            raise RuntimeError(
                f"Could not embed full secret ({bit_index}/{len(secret_bits)} bits) "
                f"within {max_steps} steps. Try increasing MAX_RESPONSE_LENGTH or TOP_N."
            )

        if on_progress is not None:
            on_progress(total_bits, total_bits)

        return carrier_token_ids

    def decode_from_context(
        self,
        context_token_ids: list[Token],
        carrier_token_ids: list[Token],
        *,
        password: str = PASSWORD,
        threshold: float = THRESHOLD,
        top_n: int = TOP_N,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        self._require_model()
        model = self.model
        assert model is not None

        if context_token_ids[-len(carrier_token_ids) :] != carrier_token_ids:
            raise ValueError("carrier_token_ids must be suffix of context_token_ids")

        _init_prng(password)

        prefix_len = len(context_token_ids) - len(carrier_token_ids)
        full_ids = list(context_token_ids)
        decoded_bits: list[int] = []
        eos_id = model.tokenizer.eos_token_id
        total_steps = len(carrier_token_ids)

        device = get_model_device(model)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            all_logits = model.model(input_ids=input_ids).logits

        for step, token_id in enumerate(carrier_token_ids, start=1):
            pos_in_full = prefix_len + step - 1
            logits = all_logits[0, pos_in_full - 1, :]
            probs = torch.softmax(logits, dim=-1)

            k = min(top_n, probs.shape[0])
            top = torch.topk(probs, k)
            top_indices = top.indices.tolist()
            top_probs = top.values.tolist()

            safe = _safe_tokens(
                top_indices,
                top_probs,
                threshold,
                strip_eos=True,
                eos_token_id=eos_id,
            )
            if safe:
                decoded_bits.extend(_extract_bits_from_token(safe, token_id))

            if on_progress is not None:
                on_progress(step, total_steps)

            payload = bits_to_bytes(decoded_bits)
            if payload.endswith(EOM_BYTE):
                return payload[: -len(EOM_BYTE)].decode("utf-8", errors="replace")

        return bits_to_bytes(decoded_bits).decode("utf-8", errors="replace")

    def _require_tokenizer(self) -> None:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("LLM not loaded.")
