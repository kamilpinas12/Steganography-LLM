# Steganografia w LLM

Ukrywanie sekretu w wyborze tokenów modelu językowego (rank-token steganografia): enkoder i dekoder dzielą hasło (PRNG), a po stronie sieci przesyłane są tylko ID tokenów.

## Struktura

| Folder | Opis |
|--------|------|
| [`gui/`](gui/) | Demo UDP: klient + serwer (PyQt6) |
| [`Steganography_benchmarks_V2/`](Steganography_benchmarks_V2/) | Benchmarki (notebooki + `scripts/` + `results/`) |

Modele Hugging Face cache’owane lokalnie w `models/` (gitignored).

## Szybki start — GUI

```bash
cd gui
pip install -r requirements.txt
python gui_server.py   # terminal 1
python gui_client.py   # terminal 2
```

## Szybki start — benchmarki

Zobacz [`Steganography_benchmarks_V2/README.md`](Steganography_benchmarks_V2/README.md).
Wyniki: `Steganography_benchmarks_V2/results/<benchmark>/`.
