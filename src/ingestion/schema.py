"""Unified production-log schema + PII redaction."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def redact_pii(text: str) -> str:
    for label, pattern in _PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{label.upper()}]", text)
    return text


class LogEntry(BaseModel):
    id: str
    user_prompt: str
    system_prompt: str = ""
    model: str
    response: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    user_feedback: str | None = None   # "thumbs_up" | "thumbs_down" | None
    retried: bool = False
    feature: str = "unknown"
    timestamp: str = ""

    def redacted(self) -> "LogEntry":
        return self.model_copy(update={
            "user_prompt": redact_pii(self.user_prompt),
            "response": redact_pii(self.response),
        })


class LogBatch(BaseModel):
    entries: list[LogEntry] = Field(default_factory=list)
