import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOP_N = 10 # Number of top tokens to display for selection
MAX_STEPS = 100
INITIAL_PROMPT = "My favorite programming language is"  



model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Downloading/loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto" if device == "cuda" else None
)
if device != "cuda":
    model.to(device)

model.eval()

prompt = INITIAL_PROMPT

def manual_token_selection(base_prompt: str) -> None:
    input_ids = tokenizer(base_prompt, return_tensors="pt").input_ids.to(device)

    print(f"\nInitial prompt: {base_prompt}")
    print("Enter a token number from the list below (or type 0 to finish).")

    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            top_k = torch.topk(probs, TOP_N)

        print(f"\nStep {step}")
        print(f"{'No':<4} | {'Token':<22} | {'Probability':<16}")
        print("-" * 52)

        top_indices = top_k.indices.tolist()
        for i, (token_id, prob) in enumerate(zip(top_indices, top_k.values.tolist()), start=1):
            # Show raw token as returned by the tokenizer vocabulary (no stripping/replacement)
            token_str = tokenizer.convert_ids_to_tokens(int(token_id))
            # use repr to make leading/trailing spaces visible
            print(f"{i:<4} | {repr(token_str):<30} | {prob * 100:>8.4f}%")

        print("0    | [END]                  | -")
        choice_raw = input("Choice: ").strip()

        if choice_raw == "0":
            print("Stopped by user (end marker).")
            break

        if not choice_raw.isdigit():
            print("Invalid choice. Enter a number from the list.")
            continue

        choice = int(choice_raw)
        if choice < 1 or choice > len(top_indices):
            print("Out of range. Choose a token from 1 to TOP_N.")
            continue

        chosen_token_id = top_indices[choice - 1]
        chosen_token_str = tokenizer.convert_ids_to_tokens(int(chosen_token_id))
        # show chosen token raw and its decoded form (no cleanup) to reveal spaces
        print("Chosen token raw:", repr(chosen_token_str))
        try:
            decoded_chosen = tokenizer.decode([int(chosen_token_id)], clean_up_tokenization_spaces=False)
        except TypeError:
            # some tokenizers don't accept clean_up_tokenization_spaces arg
            decoded_chosen = tokenizer.decode([int(chosen_token_id)])
        print("Chosen token decoded:", repr(decoded_chosen))
        if chosen_token_id == tokenizer.eos_token_id:
            print("EOS token selected. Stopping generation.")
            break

        chosen_tensor = torch.tensor([[chosen_token_id]], device=device)
        input_ids = torch.cat([input_ids, chosen_tensor], dim=1)

        # Show raw token ids and their token strings (no modification)
        ids_list = input_ids[0].tolist()
        token_strs = [tokenizer.convert_ids_to_tokens(int(t)) for t in ids_list]
        print("Current token ids:", ids_list)
        print("Current raw tokens:", token_strs)
        # Also show decoded text for convenience
        current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Current text: {current_text}")

    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\nFinal token ids:", input_ids[0].tolist())
    print("Final raw tokens:", [tokenizer.convert_ids_to_tokens(int(t)) for t in input_ids[0].tolist()])
    print("\nFinal output:")
    print(final_text)


if __name__ == "__main__":
    manual_token_selection(prompt)
