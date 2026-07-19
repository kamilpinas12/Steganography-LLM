# Wyniki lokalne

Każdy benchmark ma własny folder: runy + `export_csv.py` → `summary.csv` z **wszystkimi** polami skalarnymi (manifest + wyniki). Nie ma wspólnego CSV dla wszystkich testów.

```
results/
  humaneval/   runs/  export_csv.py  summary.csv
  capacity/    ...
  perplexity/  ...
  binoculars/  ...
  run_evaluate.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # humaneval, capacity
pip install -r requirements-gpu.txt      # + perplexity, binoculars
```

## Workflow

1. Skopiuj folder runu → `results/<benchmark>/runs/`
2. `python run_evaluate.py NAZWA_RUNU` (jeśli brak `evaluation/`)
3. `python <benchmark>/export_csv.py`

Kolumny w CSV: parametry runu + metryki (spłaszczone, bez prefiksów). Listy próbek zostają w JSON.
