"""Import ML stack before Qt — avoids PySide6/shiboken vs pandas/six crash."""

from __future__ import annotations

import os


def bootstrap_ml() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
