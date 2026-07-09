"""Rank-token steganografia LLM — enkoder i dekoder na liście ID tokenów."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch

from helpers import ModelWrapper, Token, get_model_device, str2tokens, tokens2str

EOM_BYTE = b"\x04"


@dataclass
class EncodeResult:
    prompt_token_ids: list[Token]
    carrier_token_ids: list[Token]
    token_ids: list[Token]
    text: str


def bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte_val in data:
        for shift in range(7, -1, -1):
            bits.append((byte_val >> shift) & 1)
    return bits


def bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def int_to_bits(value: int, length: int) -> list[int]:
    return [(value >> i) & 1 for i in range(length - 1, -1, -1)]


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - len(bits) % 8, 8):
        out.append(bits_to_int(bits[i : i + 8]))
    return bytes(out)


def seed_from_password(password: str) -> int:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _init_prng(password: str) -> None:
    seed = seed_from_password(password)
    random.seed(seed)
    torch.manual_seed(seed % (2**31))


def _largest_power_of_2(n: int) -> int:
    if n <= 0:
        return 0
    power = 1
    while power * 2 <= n:
        power *= 2
    return power


def _bits_per_step(num_safe_tokens: int) -> int:
    capacity = _largest_power_of_2(num_safe_tokens)
    if capacity <= 1:
        return 0
    k = 0
    while 2**k < capacity:
        k += 1
    return k


@torch.no_grad()
def _next_token_distribution(
    model: ModelWrapper,
    context: list[Token],
    top_n: int,
) -> tuple[list[Token], list[float], torch.Tensor]:
    device = get_model_device(model)
    input_ids = torch.tensor([context], device=device)
    logits = model.model(input_ids=input_ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    k = min(top_n, probs.shape[0])
    top = torch.topk(probs, k)
    return top.indices.tolist(), top.values.tolist(), probs


def _pick_filler_token(top_indices: list[Token], eos_token_id: int | None) -> Token:
    """Pick a continuation token when none pass threshold; never use EOS as filler."""
    if eos_token_id is not None:
        for token_id in top_indices:
            if token_id != eos_token_id:
                return token_id
    return top_indices[0]


def _safe_tokens(
    top_indices: list[Token],
    top_probs: list[float],
    threshold: float,
    *,
    strip_eos: bool,
    eos_token_id: int | None,
) -> list[Token]:
    safe = [tid for tid, prob in zip(top_indices, top_probs) if prob >= threshold]
    if strip_eos and eos_token_id is not None and eos_token_id in safe:
        safe.remove(eos_token_id)
    return safe


def _pick_token_for_bits(safe_tokens: list[Token], secret_bits: list[int], bit_index: int) -> tuple[Token, int]:
    capacity = _largest_power_of_2(len(safe_tokens))
    k = _bits_per_step(len(safe_tokens))
    if k == 0:
        return safe_tokens[0], bit_index

    chunk = secret_bits[bit_index : bit_index + k]
    bit_index += k
    while len(chunk) < k:
        chunk.append(0)

    pool = safe_tokens[:capacity]
    order = list(range(len(pool)))
    random.shuffle(order)
    shuffled = [pool[i] for i in order]
    return shuffled[bits_to_int(chunk)], bit_index


def _extract_bits_from_token(
    safe_tokens: list[Token],
    token_id: Token,
) -> list[int]:
    capacity = _largest_power_of_2(len(safe_tokens))
    k = _bits_per_step(len(safe_tokens))
    if k == 0 or token_id not in safe_tokens[:capacity]:
        return []

    pool = safe_tokens[:capacity]
    order = list(range(len(pool)))
    random.shuffle(order)
    shuffled = [pool[i] for i in order]
    position = shuffled.index(token_id)
    return int_to_bits(position, k)


def encoder(
    model: ModelWrapper,
    prompt: str,
    secret: bytes,
    password: str,
    *,
    threshold: float = 0.01,
    eos_threshold: float = 0.01,
    top_n: int = 15,
    max_steps: int = 1000,
) -> EncodeResult:
    """Ukryj bajty w wygenerowanym tekście. Na wejściu surowe `bytes` (np. po dekodowaniu hex)."""
    _init_prng(password)

    secret_bits = bytes_to_bits(secret + EOM_BYTE)
    prompt_token_ids = str2tokens(model, prompt)
    context = list(prompt_token_ids)
    carrier_token_ids: list[Token] = []

    bit_index = 0
    encoding_done = False
    eos_id = model.tokenizer.eos_token_id
    tail_steps = 0
    min_tail_steps = 12

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
            chosen = _pick_filler_token(top_indices, eos_id)
        else:
            chosen, bit_index = _pick_token_for_bits(safe, secret_bits, bit_index)

        context.append(chosen)
        carrier_token_ids.append(chosen)

        if chosen == eos_id:
            break

    return EncodeResult(
        prompt_token_ids=prompt_token_ids,
        carrier_token_ids=carrier_token_ids,
        token_ids=context,
        text=tokens2str(model, context),
    )


def decoder(
    model: ModelWrapper,
    context: list[Token],
    carrier_token_ids: list[Token],
    password: str,
    *,
    threshold: float = 0.01,
    top_n: int = 15,
) -> bytes:
    """Odtwórz bajty z listy ID tokenów niosących ukrytą wiadomość."""
    _init_prng(password)

    if len(context) < len(carrier_token_ids):
        raise ValueError("context krótszy niż carrier_token_ids")

    prefix_len = len(context) - len(carrier_token_ids)
    if context[prefix_len:] != carrier_token_ids:
        raise ValueError("carrier_token_ids musi być sufiksem context")

    decoded_bits: list[int] = []
    eos_id = model.tokenizer.eos_token_id

    device = get_model_device(model)
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
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

        payload = bits_to_bytes(decoded_bits)
        if payload.endswith(EOM_BYTE):
            return payload[: -len(EOM_BYTE)]

    return bits_to_bytes(decoded_bits)
