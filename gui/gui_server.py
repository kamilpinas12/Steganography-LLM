#!/usr/bin/env python3
"""LLM Steganography — Server GUI (UDP + async encode)."""

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
from shared_config import UDP_CLIENT_PORT, UDP_HOST
from stego_protocol import MSG_PROMPT, make_response_message
from stego_service import StegoEngine
from stego_workers import (
    EncodeWorker,
    TokenizerLoadWorker,
    UdpListenWorker,
    UdpSendWorker,
    parse_udp_message,
    server_listen_port,
)


class ServerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LLM Stego — Server")
        self.resize(1000, 700)

        self._engine = StegoEngine()
        self._context_token_ids: list[int] = []
        self._client_addr: tuple[str, int] | None = None
        self._udp_listener: UdpListenWorker | None = None
        self._job_worker = None
        self._tokenizer_worker: TokenizerLoadWorker | None = None

        self._build_ui()
        self._start_udp_listener()
        QTimer.singleShot(0, self._start_tokenizer_load)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self._status_label = QLabel("Starting... You can type the secret below.")
        self._incoming_log = QTextEdit()
        self._incoming_log.setReadOnly(True)
        self._incoming_log.setPlaceholderText("Incoming prompts from client (read-only)")

        self._secret_input = QLineEdit()
        self._secret_input.setPlaceholderText(">>> Type SECRET to hide HERE")
        self._secret_input.setEnabled(True)
        self._secret_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._encode_button = QPushButton("Encode")
        self._encode_button.setEnabled(False)
        self._encode_button.clicked.connect(self._on_encode_clicked)

        self._encode_progress = QProgressBar()
        self._encode_progress.setRange(0, 100)
        self._encode_progress.setValue(0)
        self._encode_progress.setFormat("Encode: %v / %m bits")

        left.addWidget(self._status_label)
        left.addWidget(QLabel("Client prompts (read-only)"))
        left.addWidget(self._incoming_log, stretch=1)
        left.addWidget(QLabel("Secret to hide (editable):"))
        left.addWidget(self._secret_input)
        left.addWidget(self._encode_button)
        left.addWidget(self._encode_progress)

        right = QVBoxLayout()
        right.addWidget(SharedSettingsPanel())
        right.addStretch()

        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._secret_input.setFocus()

    def _start_tokenizer_load(self) -> None:
        self._status_label.setText(f"Loading tokenizer... UDP :{server_listen_port()}")
        self._tokenizer_worker = TokenizerLoadWorker(self._engine)
        self._tokenizer_worker.finished_ok.connect(self._on_tokenizer_ready)
        self._tokenizer_worker.failed.connect(self._on_tokenizer_failed)
        self._tokenizer_worker.finished.connect(self._tokenizer_worker.deleteLater)
        self._tokenizer_worker.start()

    def _start_job(self, worker) -> None:
        self._job_worker = worker
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _start_udp_listener(self) -> None:
        self._udp_listener = UdpListenWorker(server_listen_port())
        self._udp_listener.message_received.connect(self._on_udp_message)
        self._udp_listener.listen_error.connect(self._on_udp_error)
        self._udp_listener.start()

    def _on_tokenizer_ready(self) -> None:
        self._status_label.setText(
            f"Ready. Type secret below. Listening UDP :{server_listen_port()}"
        )
        self._secret_input.setFocus()

    def _on_tokenizer_failed(self, message: str) -> None:
        self._status_label.setText(f"Tokenizer load failed: {message}")
        QMessageBox.critical(self, "Tokenizer error", message)

    def _log(self, text: str) -> None:
        self._incoming_log.append(text)

    def _on_udp_message(self, data: bytes, addr: tuple[str, int]) -> None:
        if not self._engine.tokenizer_ready:
            self._log("Prompt received before tokenizer ready — ignored.")
            return
        self._process_udp_message(data, addr)

    def _process_udp_message(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            message = parse_udp_message(data)
        except Exception as exc:
            self._log(f"[UDP] Invalid payload from {addr}: {exc}")
            return

        if message.msg_type != MSG_PROMPT:
            self._log(f"[UDP] Ignored message type={message.msg_type} from {addr}")
            return

        self._client_addr = addr
        self._context_token_ids.extend(message.token_ids)
        prompt_text = self._engine.detokenize(message.token_ids)
        self._log(f"Client @ {addr[0]}:{addr[1]}")
        self._log(f"Prompt tokens: {len(message.token_ids)} | Context: {len(self._context_token_ids)}")
        self._log(f"Text: {prompt_text}")
        self._log("---")
        self._encode_button.setEnabled(True)
        self._status_label.setText("Prompt received. Enter secret and click Encode.")

    def _on_encode_clicked(self) -> None:
        secret = self._secret_input.text()
        if not secret:
            QMessageBox.warning(self, "Encode", "Enter a secret message first.")
            return
        if not self._context_token_ids:
            QMessageBox.warning(self, "Encode", "No client prompt in context yet.")
            return
        if self._client_addr is None:
            QMessageBox.warning(self, "Encode", "No client address known yet.")
            return

        total_bits = (len(secret.encode("utf-8")) + 1) * 8
        self._encode_progress.setMaximum(max(total_bits, 1))
        self._encode_progress.setValue(0)
        self._encode_button.setEnabled(False)
        self._status_label.setText("Encoding secret (loading LLM into GPU)...")

        worker = EncodeWorker(self._engine, self._context_token_ids, secret)
        worker.progress.connect(self._on_encode_progress)
        worker.finished_ok.connect(self._on_encode_finished)
        worker.failed.connect(self._on_encode_failed)
        self._start_job(worker)

    def _on_encode_progress(self, current: int, total: int) -> None:
        self._encode_progress.setMaximum(max(total, 1))
        self._encode_progress.setValue(min(current, total))

    def _on_encode_finished(self, carrier_token_ids: list[int]) -> None:
        self._context_token_ids.extend(carrier_token_ids)
        response_text = self._engine.detokenize(carrier_token_ids)
        self._log(f"Encoded response ({len(carrier_token_ids)} tokens): {response_text}")

        if self._client_addr is None:
            self._encode_button.setEnabled(True)
            return

        payload = make_response_message(carrier_token_ids)
        target = (UDP_HOST, UDP_CLIENT_PORT)
        sender = UdpSendWorker(payload, target)
        sender.finished_ok.connect(lambda: self._on_response_sent(len(carrier_token_ids), target))
        sender.failed.connect(self._on_send_failed)
        self._start_job(sender)

    def _on_response_sent(self, token_count: int, target: tuple[str, int]) -> None:
        self._encode_progress.setValue(self._encode_progress.maximum())
        self._encode_button.setEnabled(True)
        self._log(f"UDP response sent ({token_count} tokens) -> {target[0]}:{target[1]}")
        self._status_label.setText(
            f"Response sent. LLM unloaded. Shared context = {len(self._context_token_ids)} tokens."
        )

    def _on_encode_failed(self, message: str) -> None:
        self._engine.release_model()
        self._encode_button.setEnabled(True)
        self._status_label.setText("Encode failed.")
        QMessageBox.critical(self, "Encode error", message)

    def _on_send_failed(self, message: str) -> None:
        self._encode_button.setEnabled(True)
        QMessageBox.critical(self, "UDP send error", message)

    def _on_udp_error(self, message: str) -> None:
        self._status_label.setText(f"UDP error: {message}")

    def closeEvent(self, event) -> None:
        if self._udp_listener is not None:
            self._udp_listener.stop()
            self._udp_listener.wait(2000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = ServerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
