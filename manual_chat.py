import argparse

from helpers import AVAILABLE_MODELS, DEFAULT_MODEL_KEY, ModelWrapper, resolve_model_id, str2tokens, tokens2str

INITIAL_PROMPT = "My favorite programming language is"

TOP_N = 15
THRESHOLD = 0.0
MAX_STEPS = 200


def _token_label(wrapper: ModelWrapper, token_id: int) -> str:
    try:
        decoded = wrapper.tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        decoded = wrapper.tokenizer.decode([token_id])
    return repr(decoded)


def _resolve_model_name(model_key: str | None) -> str:
    if model_key is not None:
        if model_key not in AVAILABLE_MODELS:
            known = ", ".join(sorted(AVAILABLE_MODELS))
            raise ValueError(f"Unknown model '{model_key}'. Available: {known}")
        return resolve_model_id(model_key)

    print("Available models:")
    keys = list(AVAILABLE_MODELS)
    for i, key in enumerate(keys, start=1):
        print(f"  {i}. {key:<16} ({AVAILABLE_MODELS[key]})")

    while True:
        choice = input(f"Choose model [1-{len(keys)}, default={DEFAULT_MODEL_KEY}]: ").strip()
        if not choice:
            return AVAILABLE_MODELS[DEFAULT_MODEL_KEY]
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(keys):
                return AVAILABLE_MODELS[keys[index - 1]]
        if choice in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[choice]
        print("Invalid choice. Enter a number, model key, or press Enter for default.")


def run_chat(model_name: str) -> None:
    print(f"Loading model: {model_name}")
    wrapper = ModelWrapper(model_name)
    context = str2tokens(wrapper, INITIAL_PROMPT)

    print(f"\nInitial prompt: {INITIAL_PROMPT}")
    print(f"Current text: {tokens2str(wrapper, context)}")
    print("Choose the next token by number (0 = stop).\n")

    for step_num in range(1, MAX_STEPS + 1):
        candidates = wrapper.step(context, THRESHOLD)
        if not candidates:
            print("No tokens above threshold. Stopping.")
            break

        display = candidates[:TOP_N]

        print(f"Step {step_num}")
        print(f"{'No':<4} | {'Token ID':<10} | {'Text':<24} | {'Probability':<12}")
        print("-" * 58)
        for i, (token_id, prob) in enumerate(display, start=1):
            print(f"{i:<4} | {token_id:<10} | {_token_label(wrapper, token_id):<24} | {prob * 100:>8.4f}%")
        print(f"0    | {'-':<10} | {'[END]':<24} | {'-':<12}")

        choice_raw = input("Choice: ").strip()
        if choice_raw == "0":
            print("Stopped by user.")
            break

        if not choice_raw.isdigit():
            print("Invalid input. Enter a number from the list.\n")
            continue

        choice = int(choice_raw)
        if choice < 1 or choice > len(display):
            print(f"Out of range. Choose 1–{len(display)} or 0 to stop.\n")
            continue

        chosen_token_id, _ = display[choice - 1]
        context.append(chosen_token_id)

        print(f"Selected: {_token_label(wrapper, chosen_token_id)}")
        print(f"Current text: {tokens2str(wrapper, context)}\n")

        if chosen_token_id == wrapper.tokenizer.eos_token_id:
            print("EOS token selected. Stopping.")
            break

    print("\nFinal output:")
    print(tokens2str(wrapper, context))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual token-by-token chat using ModelWrapper.")
    parser.add_argument(
        "--model",
        choices=sorted(AVAILABLE_MODELS),
        help="Model key from AVAILABLE_MODELS (interactive menu if omitted).",
    )
    args = parser.parse_args()

    model_name = _resolve_model_name(args.model)
    run_chat(model_name)


if __name__ == "__main__":
    main()
