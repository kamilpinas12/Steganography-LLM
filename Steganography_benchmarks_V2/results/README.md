# Local results

Each benchmark has its own folder: runs + `export_csv.py` → `summary.csv` with **all** scalar fields (manifest + metrics). There is no shared CSV across test types.

```
results/
  humaneval/   runs/  export_csv.py  summary.csv
  capacity/    ...
  perplexity/  ...
  binoculars/  ...
  run_evaluate.py
```

## Setup

From `Steganography_benchmarks_V2/results/`:

```bash
python -m venv .venv && source .venv/bin/activate

# CPU (humaneval, capacity)
pip install -r requirements.txt

# GPU (perplexity, binoculars) — torch, transformers, bitsandbytes
pip install -r requirements-gpu.txt
```

Or in one step (from `results/`):

```bash
pip install -r ../scripts/requirements.txt
```

## Workflow

1. Copy the run folder → `results/<benchmark>/runs/`
2. `python run_evaluate.py RUN_NAME` (if `evaluation/` is missing)
3. `python <benchmark>/export_csv.py`

CSV keeps analysis columns: model params (`model`, `threshold`, `top_n`, …) + metrics.
Dropped noise: timestamps, platform, status/phase, `inferred`, task-id lists. Full details stay in run JSON.

## Perplexity (missing evaluation)

```bash
export HF_TOKEN="hf_..."   # llama / gemma
cd Steganography_benchmarks_V2/results
source .venv/bin/activate

for run in perplexity/runs/*/; do
  python run_evaluate.py "$(basename "$run")"
done

python perplexity/export_csv.py
```
