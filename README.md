# LLM Steganography

Hide a secret in LLM token choices (rank-token steganography): encoder and decoder share a password (PRNG). Only token IDs are sent over the network.

## Layout

| Folder | Description |
|--------|-------------|
| [`gui/`](gui/) | UDP demo: client + server (PyQt6) |
| [`Steganography_benchmarks_V2/`](Steganography_benchmarks_V2/) | Benchmarks (notebooks + `scripts/` + `results/`) |

Hugging Face models are cached under `models/` (gitignored).

## Quick start — GUI

```bash
cd gui
pip install -r requirements.txt
python gui_server.py   # terminal 1
python gui_client.py   # terminal 2
```

## Quick start — benchmarks

See [`Steganography_benchmarks_V2/README.md`](Steganography_benchmarks_V2/README.md).
Results live under `Steganography_benchmarks_V2/results/<benchmark>/`.
