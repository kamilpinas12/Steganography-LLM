from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor, LogitsProcessorList


@dataclass
class StegoCapacityStats:
    """Tracks how many candidate tokens were available at each generation step."""

    survivor_counts: list[int] = field(default_factory=list)
    stego_applied_steps: int = 0
    natural_fallback_steps: int = 0

    def record(self, survivor_count: int, stego_applied: bool) -> None:
        self.survivor_counts.append(int(survivor_count))
        if stego_applied:
            self.stego_applied_steps += 1
        else:
            self.natural_fallback_steps += 1

    @property
    def total_steps(self) -> int:
        return len(self.survivor_counts)

    @property
    def avg_pool_size(self) -> float:
        if not self.survivor_counts:
            return 0.0
        return sum(self.survivor_counts) / len(self.survivor_counts)

    @property
    def avg_pool_size_stego_only(self) -> float:
        stego_pools = [
            count for count, applied in zip(self.survivor_counts, self._stego_flags())
            if applied
        ]
        if not stego_pools:
            return 0.0
        return sum(stego_pools) / len(stego_pools)

    def _stego_flags(self) -> list[bool]:
        flags: list[bool] = []
        for count in self.survivor_counts:
            flags.append(count >= 2)
        return flags

    @property
    def avg_bits_per_token(self) -> float:
        """Average embeddable bits per generation step (0 when natural fallback)."""
        if not self.survivor_counts:
            return 0.0
        bits = [math.log2(count) if count >= 2 else 0.0 for count in self.survivor_counts]
        return sum(bits) / len(bits)

    @property
    def embedding_rate(self) -> float:
        """Total embeddable bits divided by total generated tokens."""
        if not self.survivor_counts:
            return 0.0
        total_bits = sum(math.log2(count) if count >= 2 else 0.0 for count in self.survivor_counts)
        return total_bits / len(self.survivor_counts)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_steps": self.total_steps,
            "stego_applied_steps": self.stego_applied_steps,
            "natural_fallback_steps": self.natural_fallback_steps,
            "avg_pool_size": self.avg_pool_size,
            "avg_pool_size_stego_only": self.avg_pool_size_stego_only,
            "avg_bits_per_token": self.avg_bits_per_token,
            "embedding_rate": self.embedding_rate,
        }

    def merge(self, other: StegoCapacityStats) -> StegoCapacityStats:
        merged = StegoCapacityStats()
        merged.survivor_counts = self.survivor_counts + other.survivor_counts
        merged.stego_applied_steps = self.stego_applied_steps + other.stego_applied_steps
        merged.natural_fallback_steps = self.natural_fallback_steps + other.natural_fallback_steps
        return merged


class DummyStegoLogitsProcessor(LogitsProcessor):
    """Simulate stego constraints by forcing uniform choice in filtered pool."""

    def __init__(
        self,
        top_n: int,
        threshold: float,
        seed: int | None = None,
        track_capacity: bool = True,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.top_n = int(top_n)
        self.threshold = float(threshold)
        self.seed = seed
        self.track_capacity = track_capacity
        self.capacity_stats = StegoCapacityStats()
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def reset_capacity_stats(self) -> None:
        self.capacity_stats = StegoCapacityStats()

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        if self.seed is None:
            return None
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
            self._generator_device = device
        return self._generator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        vocab_size = probs.shape[-1]
        k = min(self.top_n, vocab_size)

        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
        survivor_mask = top_probs >= self.threshold
        survivor_count = survivor_mask.sum(dim=-1)
        apply_random = survivor_count >= 2

        if self.track_capacity:
            for row_idx in range(scores.shape[0]):
                count = int(survivor_count[row_idx].item())
                stego_applied = bool(apply_random[row_idx].item())
                self.capacity_stats.record(count, stego_applied)

        if not torch.any(apply_random):
            return scores

        random_scores = torch.rand(
            top_probs.shape,
            device=scores.device,
            generator=self._get_generator(scores.device),
        )
        random_scores = random_scores.masked_fill(~survivor_mask, -1.0)
        chosen_rank = random_scores.argmax(dim=-1, keepdim=True)
        chosen_token = top_indices.gather(dim=-1, index=chosen_rank)

        forced_scores = torch.full_like(scores, float("-inf"))
        forced_scores.scatter_(dim=-1, index=chosen_token, value=float("inf"))

        return torch.where(apply_random.unsqueeze(-1), forced_scores, scores)


def make_stego_logits_processor(
    top_n: int,
    threshold: float,
    seed: int | None = None,
    track_capacity: bool = True,
) -> tuple[LogitsProcessorList, DummyStegoLogitsProcessor]:
    processor = DummyStegoLogitsProcessor(
        top_n=top_n,
        threshold=threshold,
        seed=seed,
        track_capacity=track_capacity,
    )
    return LogitsProcessorList([processor]), processor
