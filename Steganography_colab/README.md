# Steganography_colab — benchmarki (Colab / Kaggle)

Minimalny pakiet (bez GUI): `run_experiments.py`, `dummy_processor.py`, notebooki.

## Notebooki Kaggle (Input + Working)

| Notebook | Kiedy używać |
|----------|----------------|
| **`Kaggle_Runner.ipynb`** | Pierwsze pobranie modelu, nowy model, potrzebny `HF_TOKEN` |
| **`Kaggle_Runner_Offline.ipynb`** | Kolejne uruchomienia — **zero Hugging Face**, tylko cache |

**Input:** `/kaggle/input/datasets/kamilml/sources/`  
(`run_experiments.py`, `dummy_processor.py`, `requirements.txt`)

**Working:** `/kaggle/working/Steganography_colab/` (wyniki, `models_cache/`)

**HF token:** Kaggle Secrets → `HF_TOKEN`.

## Dwa notebooki Colab

| Notebook | Kiedy używać |
|----------|----------------|
| **`Colab_Runner.ipynb`** | Pierwsze pobranie modelu, nowy model, potrzebny `HF_TOKEN` |
| **`Colab_Runner_Offline.ipynb`** | Kolejne uruchomienia — **zero Hugging Face**, tylko cache na Drive |

## Cache modeli

```
models_cache/models/<nazwa_modelu>/config.json
```

Pierwszy run (online) pobiera i zapisuje. Offline ładuje stamtąd z `local_files_only=True`.

Sprawdź co masz w cache:
```bash
python run_experiments.py --list-cached-models --offline
```

## Offline — co działa / nie działa

**Działa** (gdy model w cache):
- `--test`: `humaneval`, `perplexity`, `capacity`
- `--model`: `qwen`, `llama`, `gemma` (klucze z `run_experiments.py`)
- `--threshold`: `0.0`, `0.01`, `0.05`, `0.1`, `all`
- `--top-n`, `--max-new-tokens`, `--seed`
- `--humaneval-tasks`: `'5'`, `'0-10'`, `'0,3,7'` (tylko z `--test humaneval`)
- `--no-model-cache`: nie zapisuj modelu na dysk (ładuje z HF do katalogu tymczasowego; nie łączyć z `--offline`)

**Działa tylko jeśli pobrano wcześniej (online):**
- `--test binoculars` — wymaga cache dla `tiiuae/falcon-7b` **i** `tiiuae/falcon-7b-instruct`

**Nie działa offline:**
- pobieranie nowych modeli
- `HF_TOKEN` / `login()`
- model spoza cache

## Sekrety HF

- **Colab:** `HF_TOKEN` w Colab Secrets (notebook online)
- **Kaggle:** `HF_TOKEN` w Kaggle Secrets (notebook online)

## Wyniki

`results/<timestamp>_<model>_<test>_thX>/` oraz `results/summary.csv`

HumanEval zapisuje też `humaneval_task_results.json` (pass/fail per zadanie).
