# GUI (UDP)

Dwa procesy PyQt6: serwer koduje sekret w odpowiedzi LLM, klient dekoduje. Po UDP lecą tylko ID tokenów.

## Uruchomienie

```bash
pip install -r requirements.txt
python gui_server.py
python gui_client.py
```

Wspólne ustawienia (hasło, threshold, model): `shared_config.py`.

## Pliki

- `gui_server.py` / `gui_client.py` — aplikacje
- `stego_service.py` / `llm_steganography.py` — enkoder/dekoder
- `stego_protocol.py` / `stego_workers.py` — UDP + wątki Qt
- `helpers.py` — ładowanie modeli (cache w `../models/`)
