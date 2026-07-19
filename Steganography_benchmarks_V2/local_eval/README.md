# Lokalna ewaluacja

## Setup

```bash
cd Steganography_benchmarks_V2/local_eval
python -m venv .venv && source .venv/bin/activate

# humaneval + capacity (CPU)
pip install -r requirements.txt

# perplexity + binoculars (GPU + torch)
pip install -r requirements-gpu.txt
```

## Workflow

1. Wygeneruj na Kaggle (`Kaggle_Generate.ipynb`, ustaw `TEST`)
2. Pobierz `runs/<nazwa>/` → `local_eval/runs/`
3. `python run_evaluate.py --list`
4. `python run_evaluate.py NAZWA_RUNU`

## Co zapisuje ewaluacja

| Test | Pliki | Zawartość |
|------|-------|-----------|
| humaneval | `extracted_responses.json`, `humaneval_results.json` | raw vs kod, pass/fail per task |
| capacity | `samples_detail.json`, `capacity_results.json` | teksty + capacity per prompt + agregat |
| perplexity | `samples_detail.json`, `perplexity_results.json` | teksty + PPL per prompt + średnie |
| binoculars | `samples_detail.json`, `binoculars_results.json` | teksty + score + etykiety AI/human |

## Wymagania GPU

| Test | GPU |
|------|-----|
| humaneval | nie |
| capacity | nie |
| perplexity | tak (ładuje model z manifestu) |
| binoculars | tak (Falcon 7B observer + performer) |

## Kaggle — wybór testu

W `Kaggle_Generate.ipynb`:

```python
TEST = 'perplexity'   # humaneval | perplexity | capacity | binoculars
THRESHOLD = 0.01
```

Każdy test × threshold = osobny folder w `runs/`.
