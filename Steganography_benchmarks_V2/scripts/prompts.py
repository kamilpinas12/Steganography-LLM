"""Prompt builders for generation (no post-processing)."""

from __future__ import annotations


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return user_prompt
    return user_prompt


def build_humaneval_prompt(tokenizer, task_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding assistant. "
                "Continue the given function with valid Python only. "
                "Output indented code for the function body. "
                "Do not explain. Do not use markdown. Do not repeat the def line."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{task_prompt}\n"
                "Complete the function body only. Reply with Python code, nothing else."
            ),
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return task_prompt
    return task_prompt
