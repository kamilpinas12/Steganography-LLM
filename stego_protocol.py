"""JSON-over-UDP message format (token IDs only, no raw text on the wire)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MSG_PROMPT = "prompt"
MSG_RESPONSE = "response"
MSG_ERROR = "error"


@dataclass(frozen=True)
class UdpMessage:
    msg_type: str
    token_ids: list[int]
    error: str | None = None

    def to_bytes(self) -> bytes:
        payload: dict[str, Any] = {"type": self.msg_type, "token_ids": self.token_ids}
        if self.error is not None:
            payload["error"] = self.error
        return json.dumps(payload).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> UdpMessage:
        obj = json.loads(data.decode("utf-8"))
        return UdpMessage(
            msg_type=str(obj["type"]),
            token_ids=[int(t) for t in obj.get("token_ids", [])],
            error=obj.get("error"),
        )


def make_prompt_message(token_ids: list[int]) -> bytes:
    return UdpMessage(MSG_PROMPT, token_ids).to_bytes()


def make_response_message(token_ids: list[int]) -> bytes:
    return UdpMessage(MSG_RESPONSE, token_ids).to_bytes()


def make_error_message(message: str) -> bytes:
    return UdpMessage(MSG_ERROR, [], error=message).to_bytes()
