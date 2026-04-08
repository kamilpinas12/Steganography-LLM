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
TEMPERATURE = 1.0

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


def string_to_bits(s: str) -> list:
    """Convert string to list of bits."""
    bits = []
    for char in s:
        byte_val = ord(char)
        for i in range(8):
            bits.append((byte_val >> (7 - i)) & 1)
    return bits


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


def load_encoder_config(path: str) -> dict:
    """Load encoder settings from JSON config file."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def load_json_if_exists(path: str) -> dict:
    """Load JSON object from file if it exists, otherwise return empty dict."""
    if not os.path.exists(path):
        return {}
    return load_encoder_config(path)


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for file path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def encode(
    prompt: str,
    secret: str,
    password: str,
    threshold: float = 0.0,
    eos_threshold: float = 0.01,
    top_n: int = 10,
    output_file: str = "data/message.json",
):
    """Encode secret message into generated text."""
    
    # Add End-of-Message marker
    secret_with_eom = secret + "\x04"
    secret_bits = string_to_bits(secret_with_eom)
    
    print(f"Secret: {repr(secret)}")
    print(f"Secret with EOM: {repr(secret_with_eom)}")
    print(f"Total bits to encode: {len(secret_bits)}")
    
    # Initialize PRNG with deterministic seed from password
    seed = seed_from_password(password)
    random.seed(seed)
    torch.manual_seed(seed % (2**31))
    
    # Initialize generation
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    all_token_ids = input_ids[0].tolist()
    
    bit_index = 0
    step = 0
    eom_reached = False
    
    print(
        f"\nStarting encoding with threshold={threshold}, eos_threshold={eos_threshold}, top_n={top_n}..."
    )
    
    while True:
        step += 1
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
        # Get top_n tokens
        top_k = torch.topk(probs, min(top_n, len(probs)))
        top_indices = top_k.indices.tolist()
        top_probs = top_k.values.tolist()
        top_prob_by_id = {token_id: prob for token_id, prob in zip(top_indices, top_probs)}
        eos_prob = float(probs[tokenizer.eos_token_id].item()) if tokenizer.eos_token_id is not None else 0.0
        
        # Filter by threshold
        safe_tokens = []
        safe_probs = []
        for token_id, prob in zip(top_indices, top_probs):
            if prob >= threshold:
                safe_tokens.append(token_id)
                safe_probs.append(prob)
        
        # If still encoding and no bits left, mark EOM
        if bit_index >= len(secret_bits):
            eom_reached = True

        # After encoding all bits, keep generating until EOS becomes likely enough.
        if eom_reached and eos_prob >= eos_threshold:
            print(
                f"Step {step}: EOS probability {eos_prob:.6f} >= eos_threshold {eos_threshold:.6f}. Stopping generation."
            )
            break
        
        # Remove EOS token if not at EOM yet
        if not eom_reached:
            if tokenizer.eos_token_id in safe_tokens:
                idx = safe_tokens.index(tokenizer.eos_token_id)
                safe_tokens.pop(idx)
                safe_probs.pop(idx)
        
        if len(safe_tokens) == 0:
            print(f"Step {step}: No safe tokens available, stopping.")
            break
        
        # Determine capacity (largest power of 2)
        capacity = largest_power_of_2(len(safe_tokens))
        # Find k such that 2^k = capacity
        k = 0
        while 2 ** k < capacity:
            k += 1
        
        # Select token based on secret bits
        if bit_index >= len(secret_bits):
            # No more bits to encode; follow the model's most probable continuation.
            chosen_idx = 0
            chosen_token_id = top_indices[0]
            actual_bits = []
        else:
            if capacity == 1:
                # k = 0, no bits encoded
                k = 0
                chosen_idx = 0
                chosen_token_id = safe_tokens[0]
                actual_bits = []
            else:
                # k > 0, extract k bits
                actual_bits = secret_bits[bit_index:bit_index + k]
                bit_index += k
                
                # Pad bits if necessary
                while len(actual_bits) < k:
                    actual_bits.append(0)
                
                # Convert bits to integer
                value = bits_to_int(actual_bits)
                
                # Trim tokens to capacity
                safe_tokens = safe_tokens[:capacity]
                
                # Shuffle using seeded random
                shuffled_indices = list(range(len(safe_tokens)))
                random.shuffle(shuffled_indices)
                shuffled_tokens = [safe_tokens[i] for i in shuffled_indices]
                
                # Select token at index value
                chosen_idx = value
                chosen_token_id = shuffled_tokens[chosen_idx]
        
        token_str = tokenizer.convert_ids_to_tokens(chosen_token_id)
        token_text = tokenizer.decode([chosen_token_id], skip_special_tokens=False)
        
        # Add chosen token to context
        chosen_tensor = torch.tensor([[chosen_token_id]], device=DEVICE)
        input_ids = torch.cat([input_ids, chosen_tensor], dim=1)
        all_token_ids.append(chosen_token_id)
        
        current_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
        chosen_prob = top_prob_by_id.get(chosen_token_id, float(probs[chosen_token_id].item()))
        if chosen_token_id in top_indices:
            top_rank = top_indices.index(chosen_token_id) + 1
            rank_text = f"TOP-{top_n} rank={top_rank}"
        else:
            rank_text = f"outside TOP-{top_n}"
        print(
            f"Step {step}: Selected token {chosen_token_id:5d} raw={repr(token_str)} text={repr(token_text)} "
            f"{rank_text}, prob={chosen_prob:.6f}, bits_encoded={bit_index}/{len(secret_bits)}, "
            f"text_len={len(current_text)}"
        )
        
        # Check if we reached EOM token in generated text
        if chosen_token_id == tokenizer.eos_token_id:
            print(f"Step {step}: EOS token reached, stopping generation.")
            break
        
        if step > 1000:  # Safety limit
            print(f"Step {step}: Max steps reached, stopping.")
            break
    
    # Decode final text
    final_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    
    print(f"\nEncoding complete!")
    print(f"Final text: {repr(final_text)}")
    print(f"Total tokens generated: {len(all_token_ids)}")
    
    # Save to JSON
    output_data = {
        "prompt": prompt,
        "text": final_text,
        "token_ids": all_token_ids,
    }
    
    ensure_parent_dir(output_file)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode secret message into LLM-generated text")
    parser.add_argument("--config", type=str, default="data/config.json", help="Shared JSON config file")
    parser.add_argument("--input", type=str, default="data/input.json", help="Input JSON file")
    parser.add_argument("--prompt", type=str, default=None, help="Starting prompt")
    parser.add_argument("--secret", type=str, default=None, help="Secret message to encode")
    parser.add_argument("--password", type=str, default=None, help="Password for PRNG seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold")
    parser.add_argument("--eos-threshold", type=float, default=None, help="EOS stopping threshold")
    parser.add_argument("--top-n", type=int, default=None, help="Number of top tokens to consider")
    
    args = parser.parse_args()
    
    config = load_json_if_exists(args.config)
    if config:
        print(f"Loaded config from {args.config}")

    input_data = load_json_if_exists(args.input)
    if input_data:
        print(f"Loaded encoder input from {args.input}")

    prompt = args.prompt if args.prompt is not None else input_data.get("prompt")
    secret = args.secret if args.secret is not None else input_data.get("secret")
    password = args.password if args.password is not None else input_data.get("password")
    output_file = args.output if args.output is not None else input_data.get("message_file")
    if output_file is None:
        output_file = input_data.get("output")
    if output_file is None:
        output_file = config.get("message_file")
    if output_file is None:
        output_file = config.get("output", "data/message.json")
    threshold = (
        args.threshold
        if args.threshold is not None
        else config.get("threshold", input_data.get("threshold", 0.01))
    )
    eos_threshold = (
        args.eos_threshold
        if args.eos_threshold is not None
        else config.get("eos_threshold", input_data.get("eos_threshold", 0.01))
    )
    top_n = (
        args.top_n if args.top_n is not None else config.get("top_n", input_data.get("top_n", 15))
    )

    # If required fields are still missing, ask interactively.
    if prompt is None:
        prompt = input("Prompt: ").strip()
    if secret is None:
        secret = input("Secret: ").strip()
    if password is None:
        password = input("Password: ").strip()

    encode(
        prompt=prompt,
        secret=secret,
        password=password,
        threshold=threshold,
        eos_threshold=eos_threshold,
        top_n=top_n,
        output_file=output_file,
    )
