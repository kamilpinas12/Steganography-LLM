"""Perplexity computation for evaluation phase."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def sequence_perplexity(model, tokenizer, text: str) -> float:
    if not text.strip():
        return float("nan")
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    encodings = {k: v.to(model.device) for k, v in encodings.items()}
    labels = encodings["input_ids"].clone()
    outputs = model(**encodings, labels=labels)
    return math.exp(outputs.loss.item())
