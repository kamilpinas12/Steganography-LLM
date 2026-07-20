# Benchmarks V2

```
Steganography_benchmarks_V2/
  *.ipynb          # Colab / Kaggle notebooks
  runs/            # generation output on the cloud platform
  scripts/         # Python pipeline (generate / evaluate)
  results/         # local runs + per-benchmark CSV export
```

## Tests

| Test | Generation | Evaluation |
|------|------------|------------|
| `humaneval` | GPU | CPU |
| `capacity` | GPU | CPU |
| `perplexity` | GPU | GPU |
| `binoculars` | GPU | GPU (Falcon) |

## CLI

```bash
python scripts/generate_responses.py --test capacity --model qwen --threshold 0.01 --platform kaggle
python scripts/evaluate_responses.py --run-dir runs/RUN_NAME
```

## Local results

Copy a run into `results/<benchmark>/runs/`, then:

```bash
cd results
source .venv/bin/activate
pip install -r requirements-gpu.txt   # from results/
python run_evaluate.py --list
python run_evaluate.py RUN_NAME
python humaneval/export_csv.py   # → humaneval/summary.csv
```

Details: [`results/README.md`](results/README.md).
