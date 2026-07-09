"""Eval case model, dedup against existing cases, and confidence-based routing."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_dataset.jsonl"
REVIEW_QUEUE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "review_queue.jsonl"

CONFIDENCE_ROUTING_THRESHOLD = 0.8
DEDUP_SIMILARITY_THRESHOLD = 0.92


class EvalCase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_prompt: str
    reference_answer: str
    expected_behavior: str          # "answer" | "refuse" | "clarify"
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    difficulty: str = "moderate"
    category: str = "general"
    label_confidence: float = 1.0
    source: str = "auto"


def is_duplicate(candidate: EvalCase, existing: list[EvalCase], embedder: Callable[[str], np.ndarray]) -> bool:
    if not existing:
        return False
    cvec = embedder(candidate.user_prompt)
    for e in existing:
        evec = embedder(e.user_prompt)
        denom = np.linalg.norm(cvec) * np.linalg.norm(evec)
        if denom and float(np.dot(cvec, evec) / denom) > DEDUP_SIMILARITY_THRESHOLD:
            return True
    return False


def route_case(case: EvalCase) -> str:
    """High-confidence labels auto-add to the dataset; low-confidence go to review."""
    return "dataset" if case.label_confidence >= CONFIDENCE_ROUTING_THRESHOLD else "review_queue"


def append_case(case: EvalCase, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json() + "\n")


def load_cases(path: Path = DATASET_PATH) -> list[EvalCase]:
    if not path.exists():
        return []
    return [EvalCase(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ingest_candidate(case: EvalCase, embedder: Callable[[str], np.ndarray]) -> str:
    """Full pipeline for one candidate: dedup -> route -> persist. Returns outcome."""
    existing = load_cases()
    if is_duplicate(case, existing, embedder):
        return "skipped_duplicate"
    destination = route_case(case)
    append_case(case, DATASET_PATH if destination == "dataset" else REVIEW_QUEUE_PATH)
    return destination
