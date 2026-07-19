"""Shared GUI widgets for Client and Server."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QFormLayout, QLabel, QVBoxLayout

from shared_config import (
    EOS_THRESHOLD,
    MAX_RESPONSE_LENGTH,
    MODEL_ID,
    MODEL_KEY,
    PASSWORD,
    THRESHOLD,
    TOP_N,
    UDP_CLIENT_PORT,
    UDP_SERVER_PORT,
)


class SharedSettingsPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        title = QLabel("Shared Context (read-only)")
        title.setStyleSheet("font-weight: bold;")
        outer.addWidget(title)

        form_host = QFrame()
        layout = QFormLayout(form_host)
        layout.setContentsMargins(0, 0, 0, 0)

        rows = [
            ("Model key", MODEL_KEY),
            ("Model ID", MODEL_ID),
            ("TOP_N", str(TOP_N)),
            ("THRESHOLD", str(THRESHOLD)),
            ("EOS_THRESHOLD", str(EOS_THRESHOLD)),
            ("PASSWORD", PASSWORD),
            ("MAX_RESPONSE_LENGTH", str(MAX_RESPONSE_LENGTH)),
            ("UDP server port", str(UDP_SERVER_PORT)),
            ("UDP client port", str(UDP_CLIENT_PORT)),
        ]
        for label, value in rows:
            val = QLabel(value)
            val.setWordWrap(True)
            layout.addRow(QLabel(label), val)

        outer.addWidget(form_host)
