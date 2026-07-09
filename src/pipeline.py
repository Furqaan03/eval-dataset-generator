"""Nightly pipeline entry point: sample logs -> cluster -> auto-label -> dedup -> add.

Wires the stages together. Run via `python -m src.pipeline` or the nightly cron
in docker-compose. Requires OPENAI_API_KEY for the labeling stage."""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from src.classify.cluster import cluster_entries, identify_outliers
from src.ingestion.sampling import SAMPLERS
from src.ingestion.schema import LogBatch
from src.labeling.auto_label import auto_label
from src.labeling.dataset import ingest_candidate
from src.runner.eval_runner import coverage_report
from src.labeling.dataset import load_cases

load_dotenv()


def run(logs_path: str, sample_size: int = 50, strategy: str = "signal_boosted") -> dict:
    import json
    from pathlib import Path

    from src.ingestion.embeddings import openai_embedder  # local import; see note below

    raw = json.loads(Path(logs_path).read_text(encoding="utf-8"))
    batch = LogBatch(**raw)
    entries = [e.redacted() for e in batch.entries]

    sampled = SAMPLERS[strategy](entries, sample_size)
    embedder = openai_embedder()
    clusters = cluster_entries(sampled, embedder)
    priority = identify_outliers(clusters) or sampled

    outcomes = {"dataset": 0, "review_queue": 0, "skipped_duplicate": 0}
    for entry in priority:
        case = auto_label(entry)
        outcome = ingest_candidate(case, embedder)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    return {"processed": len(priority), "outcomes": outcomes, "coverage": coverage_report(load_cases())}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_logs.json"
    print(run(path))
