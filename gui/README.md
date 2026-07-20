# GUI (UDP)

Two PyQt6 processes: the server embeds a secret in the LLM reply; the client recovers it. Only token IDs travel over UDP.

## Run

```bash
pip install -r requirements.txt
python gui_server.py
python gui_client.py
```

Shared settings (password, threshold, model): `shared_config.py`.

## Files

- `gui_server.py` / `gui_client.py` — apps
- `stego_service.py` / `llm_steganography.py` — encoder / decoder
- `stego_protocol.py` / `stego_workers.py` — UDP + Qt threads
- `helpers.py` — model loading (cache in `../models/`)
