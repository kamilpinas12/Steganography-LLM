# LLM Steganography GUI (Qt for Python)

This document describes the current desktop GUI built with Qt for Python (PySide6) and QML.

## Overview

The app provides:

- Encode flow: hides a secret message in model-generated text.
- Decode flow: recovers the hidden message from saved generated output.
- Shared settings panel: one place for prompt, secret, password, thresholds, and top-n.
- In-app terminal: captures Python stdout/stderr and shows live logs.

The main entry point is `app.py`, and the UI is composed from `AppUI.qml` plus reusable QML components in `components/`.

## Requirements

- Python 3.8+
- Virtual environment recommended

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First run will download the TinyLlama model to the Hugging Face cache.

## Run

```bash
source venv/bin/activate
python app.py
```

The app starts in fullscreen mode.

## Current UI Layout

`AppUI.qml` composes these blocks:

- `RuntimeSettingsPanel.qml`: shared settings used by both encode and decode.
- `EncodePanel.qml`: only action button + generated text result.
- `DecodePanel.qml`: only action button + decoded text result.
- `TerminalPanel.qml`: live terminal output and clear button.

## Shared Settings

All values are entered in one panel and reused by both actions:

- Prompt
- Secret
- Password
- Threshold
- EOS Threshold
- Top N

Current default values in settings UI:

- Prompt: `Steganography `
- Secret: `secret`
- Password: `password`
- Threshold: `0.01`
- EOS Threshold: `0.01`
- Top N: `15`

`Apply Settings` updates runtime values in `PythonBridge`.

## How Data Flows

Encode (`Generate & Encode`):

1. Reads prompt/secret/password from shared settings.
2. Uses runtime threshold, eos_threshold, top_n.
3. Calls `encode(...)` directly in Python.
4. Saves output to `data/message.json`.

Decode (`Decode Secret`):

1. Reads prompt/password from shared settings.
2. Uses runtime threshold and top_n.
3. Calls `decode(...)` on `data/message.json`.
4. Saves recovered secret to `data/decoded_message.json`.
