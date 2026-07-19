# Steganography Benchmarks V2

Dwuetapowy pipeline: **generacja RAW (Kaggle/Colab)** → **ewaluacja (lokalnie lub GPU)**.

## Benchmarki

| Test | Generacja (GPU) | Ewaluacja | Zapisuje (RAW) |
|------|-----------------|-----------|----------------|
| **humaneval** | 164 zadań | CPU | `raw_completion`, prompt, capacity per task |
| **perplexity** | 4 prompty × baseline+stego | **GPU** (model benchmarku) | pełne teksty baseline/stego, capacity |
| **capacity** | 4 prompty | **CPU** | teksty + capacity per prompt |
| **binoculars** | 4 prompty × baseline+stego | **GPU** (Falcon 7B) | teksty baseline/stego, capacity |

Thresholdy: **0.0, 0.01, 0.05, 0.1** — osobny run na każdy.

## Generacja

```bash
python generate_responses.py \
  --test perplexity \
  --model llama \
  --threshold 0.01 \
  --platform kaggle
```

Testy: `humaneval` | `perplexity` | `capacity` | `binoculars`

## Ewaluacja

```bash
python evaluate_responses.py --run-dir runs/NAZWA_RUNU
```

Wyniki w `runs/.../evaluation/`:
- `samples_detail.json` — surowe dane per próbka
- `*_results.json` — metryki + pełne sample z score
- `summary.csv` — jedna linia per run

## Struktura runu (perplexity / binoculars)

```json
{
  "sample_id": "perplexity/0",
  "user_prompt": "...",
  "prompt_text": "...",
  "baseline": {
    "raw_full_decoded": "...",
    "raw_completion": "..."
  },
  "stego": {
    "raw_full_decoded": "...",
    "raw_completion": "...",
    "capacity": { "avg_pool_size": 3.2, ... }
  }
}
```

## Lokalna ewaluacja

- **humaneval + capacity** → `local_eval/requirements.txt` (CPU)
- **perplexity + binoculars** → dodatkowo `pip install -r local_eval/requirements-gpu.txt` + GPU

Szczegóły: [`local_eval/README.md`](local_eval/README.md)

V1 (`Steganography_colab/`) bez zmian.
