#!/usr/bin/env python3
"""LLM Steganography — Client GUI (UDP + async decode)."""

from __future__ import annotations

import sys

from gui_bootstrap import bootstrap_ml

bootstrap_ml()

from PyQt6.QtCore import QThread, QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui_common import SharedSettingsPanel
from shared_config import UDP_HOST, UDP_SERVER_PORT
from stego_protocol import MSG_PROMPT, MSG_RESPONSE, make_prompt_message
from stego_service import StegoEngine
from stego_workers import (
    DecodeWorker,
    TokenizerLoadWorker,
    UdpListenWorker,
    UdpSendWorker,
    parse_udp_message,
)


class ClientWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LLM Stego — Client")
        self.resize(1000, 700)

        self._engine = StegoEngine()
        self._context_token_ids: list[int] = []
        self._last_carrier_token_ids: list[int] = []
        self._udp_listener: UdpListenWorker | None = None
        self._job_worker: QThread | None = None
        self._tokenizer_worker: TokenizerLoadWorker | None = None

        self._build_ui()
        self._start_udp_listener()
        QTimer.singleShot(0, self._start_tokenizer_load)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self._status_label = QLabel("Starting... You can type in the field below.")
        self._chat_history = QTextEdit()
        self._chat_history.setReadOnly(True)
        self._chat_history.setPlaceholderText("Chat history (read-only)")

        prompt_row = QHBoxLayout()
        self._prompt_input = QLineEdit()
        self._prompt_input.setPlaceholderText(">>> Type your prompt HERE")
        self._prompt_input.setEnabled(True)
        self._prompt_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._send_button = QPushButton("Send")
        self._send_button.setEnabled(False)
        self._send_button.clicked.connect(self._on_send_clicked)
        prompt_row.addWidget(self._prompt_input)
        prompt_row.addWidget(self._send_button)

        left.addWidget(self._status_label)
        left.addWidget(self._chat_history, stretch=1)
        left.addWidget(QLabel("Your prompt (editable):"))
        left.addLayout(prompt_row)

        right = QVBoxLayout()
        right.addWidget(SharedSettingsPanel())

        self._decode_button = QPushButton("Decode")
        self._decode_button.setEnabled(False)
        self._decode_button.clicked.connect(self._on_decode_clicked)

        self._secret_output = QTextEdit()
        self._secret_output.setReadOnly(True)
        self._secret_output.setPlaceholderText("Recovered secret will appear here...")
        self._secret_output.setMaximumHeight(120)

        self._decode_progress = QProgressBar()
        self._decode_progress.setRange(0, 100)
        self._decode_progress.setValue(0)
        self._decode_progress.setFormat("Decode: %p%")

        right.addWidget(QLabel("Recovered secret"))
        right.addWidget(self._secret_output)
        right.addWidget(self._decode_button)
        right.addWidget(self._decode_progress)
        right.addStretch()

        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._prompt_input.setFocus()

    def _start_tokenizer_load(self) -> None:
        self._status_label.setText("Loading tokenizer in background...")
        self._tokenizer_worker = TokenizerLoadWorker(self._engine)
        self._tokenizer_worker.finished_ok.connect(self._on_tokenizer_ready)
        self._tokenizer_worker.failed.connect(self._on_tokenizer_failed)
        self._tokenizer_worker.finished.connect(self._tokenizer_worker.deleteLater)
        self._tokenizer_worker.start()

    def _start_job(self, worker: QThread) -> None:
        self._job_worker = worker
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _start_udp_listener(self) -> None:
        from shared_config import UDP_CLIENT_PORT

        self._udp_listener = UdpListenWorker(UDP_CLIENT_PORT)
        self._udp_listener.message_received.connect(self._on_udp_message)
        self._udp_listener.listen_error.connect(self._on_udp_error)
        self._udp_listener.start()

    def _on_tokenizer_ready(self) -> None:
        self._status_label.setText("Ready. Type prompt below, then Send.")
        self._send_button.setEnabled(True)
        self._prompt_input.setFocus()

    def _on_tokenizer_failed(self, message: str) -> None:
        self._status_label.setText(f"Tokenizer load failed: {message}")
        QMessageBox.critical(self, "Tokenizer error", message)

    def _append_chat(self, role: str, text: str) -> None:
        self._chat_history.append(f"<b>{role}:</b> {text}")

    def _on_send_clicked(self) -> None:
        prompt_text = self._prompt_input.text().strip()
        if not prompt_text:
            return
        if not self._engine.tokenizer_ready:
            QMessageBox.warning(self, "Not ready", "Tokenizer is still loading.")
            return

        try:
            prompt_token_ids = self._engine.tokenize(prompt_text)
        except Exception as exc:
            QMessageBox.critical(self, "Tokenize error", str(exc))
            return

        self._context_token_ids.extend(prompt_token_ids)
        self._append_chat("You", prompt_text)
        self._prompt_input.clear()
        self._send_button.setEnabled(False)

        payload = make_prompt_message(prompt_token_ids)
        worker = UdpSendWorker(payload, (UDP_HOST, UDP_SERVER_PORT))
        worker.finished_ok.connect(lambda: self._on_prompt_sent(prompt_text))
        worker.failed.connect(self._on_send_failed)
        self._start_job(worker)

    def _on_prompt_sent(self, prompt_text: str) -> None:
        self._status_label.setText(f"Prompt sent ({len(prompt_text)} chars). Waiting for server...")
        self._send_button.setEnabled(True)
        self._prompt_input.setFocus()

    def _on_send_failed(self, message: str) -> None:
        QMessageBox.critical(self, "UDP send error", message)
        self._send_button.setEnabled(True)

    def _on_udp_message(self, data: bytes, _addr: tuple) -> None:
        try:
            message = parse_udp_message(data)
        except Exception as exc:
            self._append_chat("System", f"Invalid UDP payload: {exc}")
            return

        if message.msg_type == MSG_RESPONSE:
            self._handle_response(message.token_ids)
        elif message.msg_type == MSG_PROMPT:
            self._append_chat("System", "Unexpected prompt on client port.")
        else:
            self._append_chat("System", message.error or "Unknown message.")

    def _handle_response(self, carrier_token_ids: list[int]) -> None:
        if not carrier_token_ids:
            self._append_chat("System", "Empty response from server.")
            return

        self._context_token_ids.extend(carrier_token_ids)
        self._last_carrier_token_ids = list(carrier_token_ids)
        response_text = self._engine.detokenize(carrier_token_ids)
        self._append_chat("Server", response_text)
        self._decode_button.setEnabled(True)
        self._status_label.setText(
            f"Response received ({len(carrier_token_ids)} tokens). Click Decode."
        )

    def _on_decode_clicked(self) -> None:
        if not self._last_carrier_token_ids:
            QMessageBox.information(self, "Decode", "No server response to decode yet.")
            return

        self._decode_button.setEnabled(False)
        self._decode_progress.setValue(0)
        self._status_label.setText("Decoding secret (loading LLM)...")

        worker = DecodeWorker(
            self._engine,
            self._context_token_ids,
            self._last_carrier_token_ids,
        )
        worker.progress.connect(self._on_decode_progress)
        worker.finished_ok.connect(self._on_decode_finished)
        worker.failed.connect(self._on_decode_failed)
        self._start_job(worker)

    def _on_decode_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._decode_progress.setValue(int(100 * current / total))

    def _on_decode_finished(self, secret: str) -> None:
        self._secret_output.setPlainText(secret)
        self._decode_progress.setValue(100)
        self._decode_button.setEnabled(True)
        self._status_label.setText("Decode complete. LLM unloaded from GPU.")
        self._append_chat("System", f"Secret recovered ({len(secret)} chars).")

    def _on_decode_failed(self, message: str) -> None:
        self._engine.release_model()
        self._decode_button.setEnabled(True)
        self._status_label.setText("Decode failed.")
        QMessageBox.critical(self, "Decode error", message)

    def _on_udp_error(self, message: str) -> None:
        self._status_label.setText(f"UDP error: {message}")

    def closeEvent(self, event) -> None:
        if self._udp_listener is not None:
            self._udp_listener.stop()
            self._udp_listener.wait(2000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = ClientWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
