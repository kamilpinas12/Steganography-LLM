import torch
import json
import random
import argparse
import hashlib
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE != "cuda":
    model.to(DEVICE)

model.eval()


def largest_power_of_2(n: int) -> int:
    """Find largest power of 2 <= n."""
    if n <= 0:
        return 0
    power = 1
    while power * 2 <= n:
        power *= 2
    return power


def bits_to_int(bits: list) -> int:
    """Convert list of bits to integer."""
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    return result


def int_to_bits(value: int, length: int) -> list:
    """Convert integer to list of bits with specified length."""
    bits = []
    for i in range(length - 1, -1, -1):
        bits.append((value >> i) & 1)
    return bits


def seed_from_password(password: str) -> int:
    """Create deterministic integer seed from password."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def load_json_config(path: str) -> dict:
    """Load JSON object from file path."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def load_json_if_exists(path: str) -> dict:
    """Load JSON object from file if it exists, otherwise return empty dict."""
    if not os.path.exists(path):
        return {}
    return load_json_config(path)


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for file path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def bits_to_string(bits: list) -> str:
    """Convert list of bits to string. Only complete bytes are converted."""
    result = []
    # Only process complete 8-bit chunks
    for i in range(0, len(bits) - len(bits) % 8, 8):
        byte_bits = bits[i:i+8]
        byte_val = bits_to_int(byte_bits)
        result.append(chr(byte_val))
    return "".join(result)


def decode(input_file: str, password: str, threshold: float = 0.0, top_n: int = 10) -> str:
    """Decode secret message from JSON file."""

    with open(input_file, "r") as f:
        data = json.load(f)

    prompt = data["prompt"]
    token_ids = data["token_ids"]

    print(f"Loaded {input_file}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Total tokens: {len(token_ids)}")

    seed = seed_from_password(password)
    random.seed(seed)
    torch.manual_seed(seed % (2**31))

    prompt_token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    stego_token_ids = token_ids[len(prompt_token_ids):]

    print(f"\nStarting decoding with threshold={threshold}, top_n={top_n}...")
    print(f"Stego tokens to process: {len(stego_token_ids)}")

    full_input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=full_input_ids)
    all_logits = outputs.logits

    print("Single forward pass complete (teacher forcing on full sequence).")

    decoded_bits = []
    eom_reached = False

    for step, target_token_id in enumerate(stego_token_ids, start=1):
        pos_in_full = len(prompt_token_ids) + step - 1
        logits = all_logits[0, pos_in_full - 1, :]
        probs = torch.softmax(logits, dim=-1)

        top_k = torch.topk(probs, min(top_n, len(probs)))
        top_indices = top_k.indices.tolist()
        top_probs = top_k.values.tolist()
        top_prob_by_id = {token_id: prob for token_id, prob in zip(top_indices, top_probs)}

        safe_tokens = []
        safe_probs = []
        for token_id, prob in zip(top_indices, top_probs):
            if prob >= threshold:
                safe_tokens.append(token_id)
                safe_probs.append(prob)

        if not eom_reached:
            if tokenizer.eos_token_id in safe_tokens:
                idx = safe_tokens.index(tokenizer.eos_token_id)
                safe_tokens.pop(idx)
                safe_probs.pop(idx)

        selected_token_raw = tokenizer.convert_ids_to_tokens(target_token_id)
        selected_token_text = tokenizer.decode([target_token_id], skip_special_tokens=False)
        if target_token_id in top_indices:
            top_rank = top_indices.index(target_token_id) + 1
            top_prob = top_prob_by_id[target_token_id]
            print(
                f"Step {step}: selected token {target_token_id} raw={repr(selected_token_raw)} "
                f"text={repr(selected_token_text)} in TOP-{top_n} at rank={top_rank}, prob={top_prob:.6f}"
            )
        else:
            print(
                f"Step {step}: selected token {target_token_id} raw={repr(selected_token_raw)} "
                f"text={repr(selected_token_text)} is outside TOP-{top_n}"
            )

        if len(safe_tokens) == 0:
            print(f"Step {step}: No safe tokens available, stopping.")
            break

        capacity = largest_power_of_2(len(safe_tokens))
        k = 0
        while 2 ** k < capacity:
            k += 1

        safe_tokens = safe_tokens[:capacity]

        shuffled_indices = list(range(len(safe_tokens)))
        random.shuffle(shuffled_indices)
        shuffled_tokens = [safe_tokens[i] for i in shuffled_indices]

        if target_token_id in shuffled_tokens:
            position = shuffled_tokens.index(target_token_id)

            if capacity > 1:
                bits_extracted = int_to_bits(position, k)
                decoded_bits.extend(bits_extracted)

                decoded_string = bits_to_string(decoded_bits)
                if len(decoded_string) > 0 and decoded_string[-1] == "\x04":
                    eom_reached = True
        else:
            print(f"Step {step}: Token {target_token_id} not in shuffled list!")

        token_str = tokenizer.convert_ids_to_tokens(target_token_id)
        if step % 5 == 0 or eom_reached:
            current_decoded = bits_to_string(decoded_bits)
            print(
                f"Step {step}: Token {target_token_id:5d} ({repr(token_str):20s}), "
                f"bits_decoded={len(decoded_bits)}, chars={len(current_decoded)}"
            )

        if eom_reached:
            print(f"Step {step}: EOM marker reached!")
            break

    secret = bits_to_string(decoded_bits)

    if secret.endswith("\x04"):
        secret = secret[:-1]

    print(f"\nDecoding complete!")
    print(f"Decoded bits: {len(decoded_bits)}")
    print(f"Decoded secret: {repr(secret)}")

    return secret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode secret message from JSON file")
    parser.add_argument("input_file", type=str, nargs="?", default=None, help="Input JSON file")
    parser.add_argument("--config", type=str, default="data/config.json", help="Shared JSON config file")
    parser.add_argument("--input", type=str, default="data/input.json", help="Input JSON file")
    parser.add_argument("--password", type=str, default=None, help="Password (if omitted, prompt interactively)")
    parser.add_argument("--decoded-output", type=str, default=None, help="Decoded message output JSON file")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold")
    parser.add_argument("--top-n", type=int, default=None, help="Number of top tokens to consider")
    
    args = parser.parse_args()
    
    config = load_json_if_exists(args.config)
    if config:
        print(f"Loaded config from {args.config}")

    input_data = load_json_if_exists(args.input)
    if input_data:
        print(f"Loaded decoder input from {args.input}")

    input_file = args.input_file
    if input_file is None:
        input_file = input_data.get("message_file")
    if input_file is None:
        input_file = input_data.get("input_file")
    if input_file is None:
        input_file = input_data.get("output")
    if input_file is None:
        input_file = config.get("message_file")
    if input_file is None:
        input_file = config.get("output")
    if input_file is None:
        input_file = "data/message.json"

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input JSON not found: {input_file}. Provide input_file or update {args.input}."
        )

    print(f"Using input file: {input_file}")

    password = args.password if args.password is not None else input_data.get("password")
    if password is None:
        password = input("Enter password: ").strip()

    threshold = args.threshold if args.threshold is not None else config.get("threshold", input_data.get("threshold", 0.01))
    top_n = args.top_n if args.top_n is not None else config.get("top_n", input_data.get("top_n", 15))

    decoded_output_file = args.decoded_output if args.decoded_output is not None else input_data.get("decoded_message_file")
    if decoded_output_file is None:
        decoded_output_file = config.get("decoded_message_file", "data/decoded_message.json")
    
    secret = decode(input_file, password, threshold=threshold, top_n=top_n)
    ensure_parent_dir(decoded_output_file)
    with open(decoded_output_file, "w") as f:
        json.dump({"secret": secret}, f, indent=2)
    print(f"Saved decoded message to {decoded_output_file}")
    print(f"\n{'='*60}")
    print(f"SECRET MESSAGE: {secret}")
    print(f"{'='*60}")
