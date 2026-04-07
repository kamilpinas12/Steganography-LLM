#!/usr/bin/env python3
"""
PyQt6-based LLM Steganography application with encoder/decoder integration.
"""

import sys
import json
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Slot, Signal, QCoreApplication

# Import encoder and decoder modules directly
from encoder import encode
from decoder import decode


class PythonBridge(QObject):
    """Bridge between QML UI and Python encoder/decoder functions."""
    
    # Signals for async feedback
    textGenerated = Signal(str)
    secretDecoded = Signal(str)
    logOutput = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.ensure_data_dir()
        self.threshold = 0.01
        self.eos_threshold = 0.01
        self.top_n = 15
    
    @staticmethod
    def ensure_data_dir():
        """Ensure data directory exists."""
        Path("data").mkdir(exist_ok=True)
    
    @Slot(str, str, str, result=str)
    def generateText(self, prompt: str, secret: str, password: str) -> str:
        """
        Encode secret message into generated text using encoder.
        
        Args:
            prompt: Starting prompt for text generation
            secret: Secret message to embed
            password: Password for PRNG seed
        
        Returns:
            Generated text or error message
        """
        try:
            if not prompt or not secret or not password:
                return "Error: Prompt, secret, and password are required."
            
            # Call encoder directly
            result = encode(
                prompt=prompt,
                secret=secret,
                password=password,
                threshold=self.threshold,
                eos_threshold=self.eos_threshold,
                top_n=self.top_n,
                output_file="data/message.json"
            )
            
            # Return generated text
            generated_text = result.get("text", "")
            if generated_text:
                return generated_text
            else:
                return "Encoding complete (check data/message.json for output)."
        
        except Exception as e:
            return f"Error: {str(e)}"

    @Slot(str, str, str, result=str)
    def updateSettings(self, threshold: str, eos_threshold: str, top_n: str) -> str:
        """Update runtime encoding/decoding settings from UI values."""
        try:
            parsed_threshold = float(threshold)
            parsed_eos_threshold = float(eos_threshold)
            parsed_top_n = int(top_n)

            if parsed_threshold < 0:
                return "Error: threshold must be >= 0."
            if parsed_eos_threshold < 0:
                return "Error: eos_threshold must be >= 0."
            if parsed_top_n < 1:
                return "Error: top_n must be >= 1."

            self.threshold = parsed_threshold
            self.eos_threshold = parsed_eos_threshold
            self.top_n = parsed_top_n

            return (
                f"Settings updated: threshold={self.threshold}, "
                f"eos_threshold={self.eos_threshold}, top_n={self.top_n}"
            )
        except ValueError:
            return "Error: Invalid settings. Use numeric values for thresholds and integer for top_n."

    @Slot(str, str, result=str)
    def generateSecret(self, prompt: str, password: str) -> str:
        """
        Decode secret message from generated text using decoder.

        Args:
            prompt: Original prompt (for context, if needed)
            password: Password for PRNG seed (must match encoder password)

        Returns:
            Decoded secret message or error message
        """
        try:
            if not password:
                return "Error: Password is required."

            # Check if message.json exists
            if not Path("data/message.json").exists():
                return "Error: No message file found. Generate text first."

            # Call decoder directly
            decoded_secret = decode(
                input_file="data/message.json",
                password=password,
                threshold=self.threshold,
                top_n=self.top_n
            )

            # Save decoded secret to output file
            output_file = "data/decoded_message.json"
            with open(output_file, "w") as f:
                json.dump({"secret": decoded_secret}, f, indent=2)

            return decoded_secret if decoded_secret else "Decoding complete (empty message)."

        except Exception as e:
            return f"Error: {str(e)}"


class StreamRedirector:
    """Redirect Python stream writes to original stream and QML signal."""

    def __init__(self, original_stream, signal):
        self.original_stream = original_stream
        self.signal = signal

    def write(self, message):
        if not message:
            return

        self.original_stream.write(message)
        self.original_stream.flush()
        self.signal.emit(message)
        # Keep UI responsive so logs appear in-app during long operations.
        QCoreApplication.processEvents()

    def flush(self):
        self.original_stream.flush()


def main():
    """Initialize and run the PyQt6 application."""
    app = QGuiApplication(sys.argv)
    
    engine = QQmlApplicationEngine()
    bridge = PythonBridge()

    # Mirror stdout/stderr to the in-app terminal panel.
    bridge._stdout_redirector = StreamRedirector(sys.stdout, bridge.logOutput)
    bridge._stderr_redirector = StreamRedirector(sys.stderr, bridge.logOutput)
    sys.stdout = bridge._stdout_redirector
    sys.stderr = bridge._stderr_redirector
    
    # Register the bridge as a context property
    engine.rootContext().setContextProperty("pythonBridge", bridge)
    
    # Load QML from file
    qml_file = Path(__file__).parent / "AppUI.qml"
    engine.load(str(qml_file))
    
    if not engine.rootObjects():
        print("Error: Failed to load QML file")
        sys.exit(-1)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
