"""Eval runner + regression detection + coverage analytics."""
from __future__ import annotations

from dataclasses import dataclass

from src.labeling.dataset import EvalCase


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    category: str
    difficulty: str


def score_response(case: EvalCase, response: str) -> bool:
    """A response passes if it contains all must_contain assertions and none of the
    must_not_contain traps. Cheap, deterministic — no LLM needed for the harness."""
    lower = response.lower()
    if any(token.lower() not in lower for token in case.must_contain):
        return False
    if any(trap.lower() in lower for trap in case.must_not_contain):
        return False
    return True


def run_eval(cases: list[EvalCase], responses: dict[str, str]) -> list[CaseResult]:
    results = []
    for case in cases:
        resp = responses.get(case.id, "")
        results.append(CaseResult(case.id, score_response(case, resp), case.category, case.difficulty))
    return results


def detect_regressions(current: list[CaseResult], previous: list[CaseResult]) -> dict:
    prev_by_id = {r.case_id: r for r in previous}
    new_failures, new_passes = [], []
    for r in current:
        p = prev_by_id.get(r.case_id)
        if p is None:
            continue
        if p.passed and not r.passed:
            new_failures.append(r.case_id)
        elif not p.passed and r.passed:
            new_passes.append(r.case_id)
    return {"new_failures": new_failures, "new_passes": new_passes}


def coverage_report(cases: list[EvalCase]) -> dict:
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    auto = human = 0
    for c in cases:
        by_category[c.category] = by_category.get(c.category, 0) + 1
        by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
        if c.source == "auto":
            auto += 1
        else:
            human += 1
    return {
        "total": len(cases),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "auto_labeled": auto,
        "human_reviewed": human,
    }
