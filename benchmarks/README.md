# Benchmarki dummy-stego

Konfiguracja testów jako pliki TOML. Skrypt `evaluate_performance.py` czyta wybrany plik i zapisuje wyniki do `results/YYYY-MM-DD_HH-MM-SS/`.

## Uruchomienie

```bash
# domyślnie: quality_sweep (widać spadek jakości vs threshold)
python evaluate_performance.py

# ten sam test, jawna nazwa pliku
python evaluate_performance.py --config benchmarks/degradation.toml
```

Wymagania: `torch`, `transformers`. Dla testu `harness` dodatkowo: `pip install lm-eval`.

## Pliki konfiguracyjne

| Plik | Co robi |
|------|---------|
| `default.toml` | **Zalecany** — `quality_sweep` na 4 promptach, sweep po progach |
| `degradation.toml` | To samo co `default.toml` |
| `quick.toml` | Szybki smoke test — jeden prompt, jeden threshold |
| `harness_only.toml` | lm-eval + wikitext (nie pokazuje wpływu stego) |
| `humaneval.toml` | HumanEval — trudny dla małych modeli, często pass@1 = 0 |

## Rodzaje testów (`[run].tests`)

| Test | Opis | Kiedy używać |
|------|------|--------------|
| `quality_sweep` | Generacja na wielu promptach, perplexity vs baseline dla każdego `threshold` | **Główny test degradacji** — działa na TinyLlama |
| `demo` | Jeden prompt, jeden threshold, baseline vs stego | Szybka demonstracja |
| `harness` | Benchmarki lm-eval (wikitext, humaneval, …) | Porównanie z literaturą; wiele tasków **nie** używa procesora stego |

## `[quality_sweep]` — pomiar degradacji

```toml
[run]
tests = ["quality_sweep"]

[stego]
top_n = 15
thresholds = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3]
seed = 42

[quality_sweep]
prompts = [
    "My favorite programming language is",
    "The capital of France is",
]
max_new_tokens = 48
```

Dla każdego promptu: generuje baseline (bez procesora), potem tekst z dummy-stego przy każdym progu. Mierzy **perplexity** wygenerowanego tekstu — im wyższy threshold, tym zwykle gorsza jakość. Wyniki: `quality_sweep.json` + tabela w `summary.txt`.

## Pozostałe sekcje

### `[model]`

| Klucz | Opis |
|-------|------|
| `key` | Klucz z `helpers.AVAILABLE_MODELS` lub surowe ID Hugging Face |

### `[stego]`

| Klucz | Opis |
|-------|------|
| `top_n` | Liczba top tokenów w puli kandydatów |
| `threshold` | Jeden próg — tylko dla testu `demo` |
| `thresholds` | Lista progów — dla `quality_sweep` i `harness` |
| `seed` | Seed losowania w procesorze |

### `[demo]`

| Klucz | Opis |
|-------|------|
| `prompt` | Prompt startowy |
| `max_new_tokens` | Maks. liczba generowanych tokenów |

### `[harness]`

| Klucz | Opis |
|-------|------|
| `tasks` | Taski lm-eval |
| `batch_size` | Batch size (domyślnie 1) |
| `limit` | Opcjonalny limit próbek |
| `allow_code_eval` | `true` — wymagane dla `humaneval` / `mbpp` |

### `[output]`

| Klucz | Opis |
|-------|------|
| `results_dir` | Katalog bazowy na wyniki |

## Wyniki

```
results/2026-07-07_18-30-00/
  config.toml
  run_config.json
  summary.txt
  quality_sweep.json   # gdy tests zawiera "quality_sweep"
  generation_demo.json # gdy tests zawiera "demo"
  harness_sweep.json   # gdy tests zawiera "harness"
```

## Dlaczego nie HumanEval na TinyLlama?

HumanEval wymaga generowania poprawnego kodu Pythona. TinyLlama dostaje **pass@1 = 0%** nawet bez stego — nie widać różnicy między progami. `quality_sweep` mierzy perplexity tekstu, który model faktycznie generuje z procesorem — tu degradacja jest wyraźna.
