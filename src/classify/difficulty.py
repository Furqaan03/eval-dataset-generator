"""Heuristic difficulty estimator — no LLM call, keeps the eval dataset balanced."""
from __future__ import annotations

import re

from src.ingestion.schema import LogEntry

_ADVERSARIAL = re.compile(r"(ignore (previous|above)|system prompt|jailbreak|pretend you|disregard)", re.I)
_MULTI_PART = re.compile(r"\b(and also|then|additionally|as well as|;)\b", re.I)
_REASONING = re.compile(r"\b(why|how|explain|compare|analyze|justify|derive)\b", re.I)


def estimate_difficulty(entry: LogEntry) -> str:
    """Returns: simple | moderate | hard | adversarial."""
    prompt = entry.user_prompt
    if _ADVERSARIAL.search(prompt):
        return "adversarial"

    signals = 0
    if _MULTI_PART.search(prompt):
        signals += 1
    if _REASONING.search(prompt):
        signals += 1
    if len(prompt.split()) > 60:
        signals += 1

    if signals >= 2:
        return "hard"
    if signals == 1:
        return "moderate"
    return "simple"
