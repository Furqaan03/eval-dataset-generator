"""LLM auto-labeling: generate reference answers + multi-dimensional labels.

Confidence = agreement across two independent labeling runs (per the guide)."""
from __future__ import annotations

import json

from openai import OpenAI

from src.classify.difficulty import estimate_difficulty
from src.ingestion.schema import LogEntry
from src.labeling.dataset import EvalCase

_LABEL_PROMPT = """Given a production LLM interaction, produce an eval label as JSON:
{"reference_answer": "the ideal answer", "expected_behavior": "answer|refuse|clarify",
 "must_contain": ["key assertions the answer must include"],
 "must_not_contain": ["hallucination traps the answer must avoid"],
 "category": "short topic label"}"""


def _label_once(entry: LogEntry, client: OpenAI) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _LABEL_PROMPT},
            {"role": "user", "content": f"User prompt: {entry.user_prompt}\nModel response: {entry.response}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def auto_label(entry: LogEntry, client: OpenAI | None = None) -> EvalCase:
    """Runs the labeler twice; confidence reflects agreement between runs."""
    client = client or OpenAI()
    a = _label_once(entry, client)
    b = _label_once(entry, client)

    # Confidence from agreement on the two most objective fields.
    agree = int(a.get("expected_behavior") == b.get("expected_behavior")) + int(a.get("category") == b.get("category"))
    confidence = 0.5 + 0.25 * agree  # 0.5, 0.75, or 1.0

    return EvalCase(
        user_prompt=entry.user_prompt,
        reference_answer=a.get("reference_answer", ""),
        expected_behavior=a.get("expected_behavior", "answer"),
        must_contain=a.get("must_contain", []),
        must_not_contain=a.get("must_not_contain", []),
        difficulty=estimate_difficulty(entry),
        category=a.get("category", "general"),
        label_confidence=confidence,
        source="auto",
    )
