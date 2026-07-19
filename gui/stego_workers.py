"""Background workers (QThread) for UDP, tokenizer/model lifecycle, encode and decode."""

from __future__ import annotations

import socket

from PyQt6.QtCore import QThread, pyqtSignal as Signal

from shared_config import UDP_BUFFER_SIZE, UDP_CLIENT_PORT, UDP_HOST, UDP_SERVER_PORT
from stego_protocol import UdpMessage
from stego_service import StegoEngine


class TokenizerLoadWorker(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, engine: StegoEngine) -> None:
        super().__init__()
        self._engine = engine

    def run(self) -> None:
        try:
            self._engine.load_tokenizer()
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class UdpListenWorker(QThread):
    message_received = Signal(bytes, tuple)
    listen_error = Signal(str)

    def __init__(self, bind_port: int) -> None:
        super().__init__()
        self._bind_port = bind_port
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((UDP_HOST, self._bind_port))
            sock.settimeout(0.5)
            while self._running:
                try:
                    data, addr = sock.recvfrom(UDP_BUFFER_SIZE)
                except TimeoutError:
                    continue
                except OSError as exc:
                    if self._running:
                        self.listen_error.emit(str(exc))
                    break
                if data:
                    self.message_received.emit(data, addr)
        finally:
            sock.close()


class EncodeWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        engine: StegoEngine,
        context_token_ids: list[int],
        secret: str,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._context = list(context_token_ids)
        self._secret = secret

    def run(self) -> None:
        try:
            self._engine.load_model()
            carriers = self._engine.encode_from_context(
                self._context,
                self._secret,
                on_progress=lambda cur, total: self.progress.emit(cur, total),
            )
            self._engine.release_model()
            self.finished_ok.emit(carriers)
        except Exception as exc:
            self._engine.release_model()
            self.failed.emit(str(exc))


class DecodeWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        engine: StegoEngine,
        context_token_ids: list[int],
        carrier_token_ids: list[int],
    ) -> None:
        super().__init__()
        self._engine = engine
        self._context = list(context_token_ids)
        self._carriers = list(carrier_token_ids)

    def run(self) -> None:
        try:
            self._engine.load_model()
            secret = self._engine.decode_from_context(
                self._context,
                self._carriers,
                on_progress=lambda cur, total: self.progress.emit(cur, total),
            )
            self._engine.release_model()
            self.finished_ok.emit(secret)
        except Exception as exc:
            self._engine.release_model()
            self.failed.emit(str(exc))


class UdpSendWorker(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, payload: bytes, target_addr: tuple[str, int]) -> None:
        super().__init__()
        self._payload = payload
        self._target_addr = target_addr

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(self._payload, self._target_addr)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            sock.close()


def parse_udp_message(data: bytes) -> UdpMessage:
    return UdpMessage.from_bytes(data)


def server_listen_port() -> int:
    return UDP_SERVER_PORT
