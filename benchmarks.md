Ten dokument definiuje niezbędne minimum testów potrzebnych do ewaluacji metody steganograficznej, zgodnie z obietnicami w abstrakcie.

## 1. Wydajność i Pojemność (Capacity)

- **Metryka:** Embedding Rate (ER) / Bits Per Token (BPT) [1].
- **Co bada:** Pojemność kanału steganograficznego.
- **Jak to działa:** Prosty stosunek liczby zakodowanych bitów sekretu do całkowitej liczby wygenerowanych tokenów [2]. 
- **Dlaczego to proste:** Wymaga jedynie zwykłego dzielenia dwóch zmiennych, które już posiadasz w kodzie (długość wiadomości w bitach / długość stego-tekstu w tokenach).

## 2. Spójność i Naturalność Tekstu (Text Coherence)

- **Metryka:** Perplexity (PPL) [3].
- **Co bada:** Płynność i naturalność wygenerowanego tekstu steganograficznego [3]. Wyższy wynik oznacza tekst nienaturalny.
- **Jak to działa:** Wylicza średnią logarytmiczną prawdopodobieństw wygenerowanych słów przy użyciu modelu bazowego.
- **Dlaczego to proste:** W Hugging Face (biblioteka `transformers`) liczenie PPL to przekazanie gotowego tekstu do modelu i odczytanie wartości `loss`.
- *(Opcjonalnie dla Agenta)* **Metryka pomocnicza:** Semantic Similarity (SS) [3]. Użycie biblioteki `sentence-transformers` (np. SBERT) do obliczenia cosinusa odległości między tekstem bez steganografii a stego-tekstem [4]. Zwykłe wywołanie `util.cos_sim()`.



## 3. Zdolność Rozumowania (Model Reasoning)

- **Metryka:** HumanEval (Pass@1).
- **Co bada:** Czy model po narzuceniu restrykcji steganograficznych (np. ucięciu najlepszych tokenów przez `threshold`) wciąż potrafi poprawnie rozwiązywać problemy logiczne i programistyczne.
- **Status:** Zaimplementowane w `run_experiments.py --test humaneval`. Notebook Colab: `colab/Colab_Runner.ipynb`.

## 4. Bezpieczeństwo i Niewykrywalność (Security & Undetectability)

- **Metryka:** Binoculars (Zero-shot detector) [5].
- **Co bada:** Czy zewnętrzny, zautomatyzowany system wykrywania tekstu AI sklasyfikuje Twój stego-tekst jako sztuczny/podejrzany.
- **Status:** Zaimplementowane w `run_experiments.py --test binoculars` (Falcon-7B + Falcon-7B-Instruct, tryb low-fpr).

