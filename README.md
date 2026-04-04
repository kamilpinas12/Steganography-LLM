# Steganografia w LLM - Implementacja

## Opis

System uwzględnia tajne wiadomości w naturalnej tekście generowanego przez model LLM (TinyLlama). Tajne bity są kodowane w wyborze tokenów poprzez manipulację ich pozycją w przetasowanej liście.

## Pliki

- **encoder.py** - Enkoder: generuje zakodowana wiadomosc i zapisuje do `data/message.json`
- **decoder.py** - Dekoder: wczytuje `data/message.json`, odtwarza sekret i zapisuje go do `data/decoded_message.json`
- **data/config.json** - Wspolny config dla enkodera i dekodera (`threshold`, `top_n`, domyslne nazwy plikow)
- **data/input.json** - Wejscie z GUI/uzytkownika (`prompt`, `secret`, `password`, opcjonalnie `message_file`)
- **data/message.json** - Wyjscie enkodera (steganograficzna wiadomosc)
- **data/decoded_message.json** - Wyjscie dekodera (odzyskany sekret)

## Algorytm

### Enkoder
1. Konwertuje sekret na bity (8 bitów na każdy znak), dodaje marker EOM (`\x04`)
2. Inicjalizuje PRNG (generator liczb pseudolosowych) hasłem
3. W każdym kroku generacji:
   - Pobiera top-N tokenów z modelu
   - Filtruje tokeny o prawdopodobieństwie < threshold
   - Znajduje największą potęgę 2 ≤ liczbie opcji (pojemność)
   - Wyciąga k bitów sekretu (gdzie N = 2^k)
   - Tasuje listę tokenów używając PRNG
   - Wybiera token na pozycji odpowiadającej wartości bitów
   - Dodaje token do kontekstu
4. Zapisuje wynik do JSON: prompt, wygenerowany tekst, lista token_ids

### Dekoder
1. Wczytuje JSON i pyta o hasło
2. Inicjalizuje PRNG tym samym hasłem
3. Dla każdego tokenu z token_ids:
   - Generuje te same logity co enkoder (ten sam prompt, hasło, progowe)
   - Znajduje pozycję rzeczywiście wybranego tokenu w przetasowanej liście
   - Wyciąga bity z tej pozycji
   - Akumuluje bity aż do znalezienia markera EOM
4. Wypisuje zdekodowany sekret

## Użycie

### Encoding (generowanie stenografii)

```python
from encoder import encode

secret = "HELLO"
password = "mypassword123"
prompt = "Once upon a time"

encode(
    prompt=prompt,
    secret=secret,
    password=password,
    threshold=0.01,      # Filtruj tokeny o prob. < 0.01
    top_n=15,            # Rozważ top-15 tokenów
    output_file="output.json"
)
```

Lub z command line:
```bash
python encoder.py --prompt "Once upon a time" --secret "HELLO" --password "mypassword123"
```

Lub z plikow konfiguracyjnych (wspolny + input):
```bash
python encoder.py
```

Tryb bez argumentow:
```bash
python encoder.py
```
W tym trybie enkoder domyslnie czyta:
- `data/config.json` (np. `threshold`, `top_n`)
- `data/input.json` (`prompt`, `secret`, `password`, `message_file`)

Mozesz podac wlasne pliki:
```bash
python encoder.py --config moj_config.json --input moj_input.json
```

Jesli brakuje `prompt`, `secret` lub `password`, program dopyta interaktywnie.

Przykladowy [data/config.json](data/config.json):
```json
{
   "threshold": 0.01,
   "top_n": 15,
   "message_file": "data/message.json",
   "decoded_message_file": "data/decoded_message.json"
}
```

Przykladowy [data/input.json](data/input.json):
```json
{
   "prompt": "Once upon a time",
   "secret": "HELLO",
   "password": "mypassword123",
   "message_file": "data/message.json"
}
```

### Decoding (odczytanie sekretu)

```bash
python decoder.py data/message.json
# Pyta: Enter password: mypassword123
# Wypisuje: SECRET MESSAGE: HELLO
```

Tryb bez argumentu pliku:
```bash
python decoder.py
```
W tym trybie dekoder domyslnie czyta:
- `data/config.json` (np. `threshold`, `top_n`)
- `data/input.json` (`password`, opcjonalnie `message_file`)

Mozesz podac wlasne pliki:
```bash
python decoder.py --config moj_config.json --input moj_input.json
```

Przykladowy zapis wyjscia dekodera [data/decoded_message.json](data/decoded_message.json):
```json
{
   "secret": "HELLO"
}
```

## Parametry

- **threshold** (float): Minimum prawdopodobieństwa tokenu do rozpatrywania (domyślnie: 0.01)
- **top_n** (int): Liczba top tokenów do rozpatrywania (domyślnie: 15)
- **password** (str): Hasło używane do inicjalizacji PRNG (obowiązkowe)

## Struktura data/message.json

```json
{
  "prompt": "Once upon a time",
  "text": "Once upon a time there was a wonderful kingdom...",
  "token_ids": [1, 2, 3, 4, ...],
  "secret_length": 5
}
```

## Wymagania

- PyTorch (`torch`)
- Transformers (`transformers`)
- Python 3.8+

Zainstaluj:
```bash
pip install torch transformers
```

## Techniczne detale

### Pojemność (Capacity)
- W każdym kroku enkoder oblicza `capacity = largest_power_of_2(len(safe_tokens))`
- Jeśli capacity = 1: zero bitów w tym kroku
- Jeśli capacity = 2^k: k bitów w tym kroku

### PRNG
- Uzywa deterministycznego seeda z SHA-256 hasla
- Ten sam password zawsze daje ten sam shuffle

### EOM (End of Message)
- Marker `\x04` (ASCII 4) oznacza koniec sekretu
- Dekoder zatrzymuje się w momencie gdy napotka ten znak

## Uwagi

1. **Determinizm**: Dekoder musi użyć dokładnie ten sam model, temperature i threshold co enkoder
2. **Pojemność**: Liczba ukrytych bitów zależy od liczby dostępnych tokenów w każdym kroku
3. **Naturalność**: Tekst powinien wyglądać naturalnie - model generuje go dalej po zakodowaniu wszystkich bitów
4. **Bezpieczeństwo**: Ten system nie jest kryptograficznie bezpieczny - ma na celu naukową demonstrację stenografii w LLM

## TODO

- [x] Implementacja enkoderu
- [x] Implementacja dekodera  
- [x] Algorytm kodowania bitów w tokenach
- [x] Algorytm dekodowania bitów z tokenów
- [x] Obsługa EOM markera
- [x] JSON I/O
- [ ] GUI interfejs
- [ ] Optymalizacja pojemności (lepszy selection algorytm)
- [ ] Wsparcie dla innych modeli
