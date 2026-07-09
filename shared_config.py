"""Shared constants for Client and Server — must stay identical on both apps."""

# --- Steganography & model ---
# Keep in sync with helpers.AVAILABLE_MODELS (no import here — avoids loading transformers before Qt).
MODEL_KEY = "tinyllama"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOP_N = 25
THRESHOLD = 0.1
EOS_THRESHOLD = 0.1
PASSWORD = "stego-demo-password"
MAX_RESPONSE_LENGTH = 256

# --- UDP ---
UDP_HOST = "127.0.0.1"
UDP_SERVER_PORT = 50111
UDP_CLIENT_PORT = 50110
UDP_BUFFER_SIZE = 65535
