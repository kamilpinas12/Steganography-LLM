# Benchmarki V2

```
Steganography_benchmarks_V2/
  *.ipynb          # notebooki Colab/Kaggle
  runs/            # wyjście generacji na platformie
  scripts/         # kod Pythona (generate / evaluate)
  results/         # lokalne runy + eksport CSV per benchmark
```

## Testy

| Test | Generacja | Ewaluacja |
|------|-----------|-----------|
| `humaneval` | GPU | CPU |
| `capacity` | GPU | CPU |
| `perplexity` | GPU | GPU |
| `binoculars` | GPU | GPU (Falcon) |

## CLI

```bash
python scripts/generate_responses.py --test capacity --model qwen --threshold 0.01 --platform kaggle
python scripts/evaluate_responses.py --run-dir runs/NAZWA_RUNU
```

## Lokalne wyniki

Skopiuj run do `results/<benchmark>/runs/`, potem:

```bash
cd results
python run_evaluate.py --list
python run_evaluate.py NAZWA_RUNU
python humaneval/export_csv.py   # → humaneval/summary.csv
```

Szczegóły: [`results/README.md`](results/README.md).
