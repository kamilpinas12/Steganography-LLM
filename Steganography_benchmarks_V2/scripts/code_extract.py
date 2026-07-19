"""Post-processing: extract Python code from raw model output (evaluation phase only)."""

from __future__ import annotations

import re

PYTHON_CODE_BLOCK_RE = re.compile(
    r"```python(?:[ \t]*\r?\n)?(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
GENERIC_CODE_BLOCK_RE = re.compile(
    r"```(?!python)(?:[ \t]*\r?\n)?(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
PYTHON_CODE_BLOCK_OPEN_RE = re.compile(
    r"```python(?:[ \t]*\r?\n)?(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)
GENERIC_CODE_BLOCK_OPEN_RE = re.compile(
    r"```(?!python)(?:[ \t]*\r?\n)?(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)
TRAILING_FENCE_RE = re.compile(r"\r?\n?```[ \t]*$")

REASONING_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)
REASONING_OPEN_TO_FENCE_RE = re.compile(
    r"<think>.*?(?=```)",
    flags=re.DOTALL | re.IGNORECASE,
)
REASONING_OPEN_TO_END_RE = re.compile(
    r"<think>.*",
    flags=re.DOTALL | re.IGNORECASE,
)

CHAT_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|im_start\|>assistant\s*", re.IGNORECASE),
    re.compile(r"<\|redacted_im_end\|>\s*", re.IGNORECASE),
    re.compile(r"<\|eot_id\|>\s*", re.IGNORECASE),
    re.compile(r"<eos>\s*", re.IGNORECASE),
    re.compile(r"<start_of_turn>model\s*", re.IGNORECASE),
)

ASSISTANT_REPLY_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<\|redacted_start_header_id\|>assistant<\|redacted_end_header_id\|>\s*",
        re.IGNORECASE,
    ),
    re.compile(r"<\|im_start\|>assistant\s*", re.IGNORECASE),
    re.compile(r"<start_of_turn>model\s*", re.IGNORECASE),
    re.compile(r"(?:^|\n)assistant\s*\n+", re.IGNORECASE),
)

IMPORT_FALLBACK_RE = re.compile(
    r"(?:^\s*(?:from\s+\S+\s+import|import\s+\S+)|\bdef\s+)",
    flags=re.MULTILINE,
)

PROSE_TAIL_RE = re.compile(
    r"(?:\n\s*\d+\.\s+\*\*|\n\*\*Example|\n\*\*[A-Za-z]|\n-{3,}|\n#{1,3}\s)",
    re.IGNORECASE,
)
DEF_LINE_PROSE_RE = re.compile(
    r"^(\s*def\s+\w+\s*\([^)]*\)\s*:).*$",
    re.IGNORECASE,
)


def extract_generated_part(full_text: str, prompt_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return full_text.strip()


def _trim_code_edges(text: str) -> str:
    return text.strip("\n\r")


def _strip_reasoning_tags(text: str) -> str:
    cleaned = REASONING_BLOCK_RE.sub("", text)
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = REASONING_OPEN_TO_FENCE_RE.sub("", cleaned, count=1)
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = REASONING_OPEN_TO_END_RE.sub("", cleaned, count=1)
    return cleaned


def _strip_to_assistant_reply(text: str) -> str:
    """Keep only text after the last assistant/model turn (Llama plain decode, chat templates)."""
    best_end = -1
    for pattern in ASSISTANT_REPLY_MARKERS:
        for match in pattern.finditer(text):
            if match.end() > best_end:
                best_end = match.end()
    if best_end >= 0:
        return text[best_end:].strip()
    return text


def _strip_chat_markers(text: str) -> str:
    cleaned = text
    for pattern in CHAT_MARKER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return _trim_code_edges(cleaned)


def _extract_markdown_code(text: str) -> str | None:
    for pattern in (PYTHON_CODE_BLOCK_RE, GENERIC_CODE_BLOCK_RE):
        match = pattern.search(text)
        if match:
            return _trim_code_edges(match.group(1))

    for pattern in (PYTHON_CODE_BLOCK_OPEN_RE, GENERIC_CODE_BLOCK_OPEN_RE):
        match = pattern.search(text)
        if match:
            body = TRAILING_FENCE_RE.sub("", match.group(1))
            return _trim_code_edges(body)

    return None


def _extract_code_fallback(text: str, *, entry_point: str | None = None) -> str:
    if entry_point:
        pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if matches:
            return _trim_code_edges(text[matches[-1].start() :])
    match = IMPORT_FALLBACK_RE.search(text)
    if match:
        return _trim_code_edges(text[match.start() :])
    return _trim_code_edges(text)


def _strip_prose_tail(text: str) -> str:
    match = PROSE_TAIL_RE.search(text)
    if match:
        return _trim_code_edges(text[: match.start()])
    return text


def _fix_def_line_prose(text: str) -> str:
    fixed_lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*def\s+", line) and (
            "`" in line or re.search(r"\bdefines\b|\bfunction\b", line, re.IGNORECASE)
        ):
            match = DEF_LINE_PROSE_RE.match(line)
            fixed_lines.append(match.group(1) if match else line.split("`", 1)[0].rstrip())
            continue
        fixed_lines.append(line)
    return _trim_code_edges("\n".join(fixed_lines))


_BODY_STOP_RE = re.compile(
    r"(?:^\s*if __name__\s*==|^\s*```|^\s*\*\*Explanation|\n\s*\*\*)",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_body_garbage(text: str) -> str:
    match = _BODY_STOP_RE.search(text)
    if match:
        return _trim_code_edges(text[: match.start()])
    return text


def _normalize_body_indent(lines: list[str]) -> str:
    """Add 4-space base indent when the model returned a flush-left function body."""
    if not lines:
        return ""
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    if not all(not line.startswith(("    ", "\t")) for line in non_empty):
        # Mixed or already indented — only fix flush-left lines at the start.
        normalized: list[str] = []
        for line in lines:
            if line.strip() and not line.startswith(("    ", "\t")):
                normalized.append("    " + line.lstrip())
            else:
                normalized.append(line)
        return _trim_code_edges("\n".join(normalized))
    normalized = ["    " + line.lstrip() if line.strip() else line for line in lines]
    return _trim_code_edges("\n".join(normalized))


def _extract_indented_python(text: str) -> str | None:
    text = _strip_body_garbage(text)
    lines = text.splitlines()
    has_indented = any(line.startswith(("    ", "\t")) for line in lines if line.strip())
    has_def = bool(re.search(r"^\s*def\s+", text, re.MULTILINE))

    # Gemma-style: function body only, no def line (often flush-left).
    if not has_def:
        body_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if body_lines:
                    body_lines.append(line)
                continue
            if stripped.startswith(("**", "# Explanation")):
                break
            body_lines.append(line)
        if not body_lines:
            return None
        return _normalize_body_indent(body_lines)

    # Models that repeat def: keep legacy indented-block extraction.
    body_lines = []
    started = False
    for line in lines:
        if line.startswith(("    ", "\t")):
            started = True
            body_lines.append(line)
            continue
        if started and not line.strip():
            body_lines.append(line)
            continue
        if started:
            break
        if line.strip().startswith("def "):
            continue
    if body_lines:
        return _trim_code_edges("\n".join(body_lines))
    return None


def _drop_duplicate_def(text: str, entry_point: str | None) -> str:
    if not entry_point:
        return text
    stripped = text.strip()
    if re.fullmatch(rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:", stripped):
        return ""
    pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    tail = text[matches[-1].start() :]
    body = _extract_indented_python(tail)
    return body if body else _trim_code_edges(tail)


def extract_code_for_eval(text: str, *, entry_point: str | None = None) -> str:
    """Normalize raw model output to Python for HumanEval Pass@1."""
    candidate = _trim_code_edges(text)
    if not candidate:
        return candidate

    candidate = _strip_reasoning_tags(candidate)
    candidate = _strip_to_assistant_reply(candidate)
    candidate = _strip_chat_markers(candidate)
    candidate = _strip_prose_tail(candidate)
    candidate = _fix_def_line_prose(candidate)

    markdown_code = _extract_markdown_code(candidate)
    if markdown_code is not None:
        candidate = markdown_code
    else:
        candidate = _extract_code_fallback(candidate, entry_point=entry_point)

    candidate = _strip_prose_tail(candidate)
    candidate = _fix_def_line_prose(candidate)
    candidate = _drop_duplicate_def(candidate, entry_point)

    indented = _extract_indented_python(candidate)
    if indented and (candidate.lstrip().startswith("def ") or entry_point):
        return indented

    return candidate


def extract_completion(
    raw_completion: str,
    *,
    entry_point: str | None = None,
) -> str:
    return extract_code_for_eval(raw_completion, entry_point=entry_point)
