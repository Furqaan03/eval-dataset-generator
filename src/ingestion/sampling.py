"""Three sampling strategies: uniform, stratified, signal-boosted."""
from __future__ import annotations

import random
from collections import defaultdict

from src.ingestion.schema import LogEntry


def uniform_sample(entries: list[LogEntry], n: int, seed: int = 42) -> list[LogEntry]:
    rng = random.Random(seed)
    return rng.sample(entries, min(n, len(entries)))


def stratified_sample(entries: list[LogEntry], n: int, seed: int = 42) -> list[LogEntry]:
    """Ensures every (feature, model) segment is represented proportionally."""
    rng = random.Random(seed)
    strata: dict[tuple, list[LogEntry]] = defaultdict(list)
    for e in entries:
        strata[(e.feature, e.model)].append(e)

    per_stratum = max(1, n // len(strata)) if strata else 0
    sampled: list[LogEntry] = []
    for group in strata.values():
        sampled.extend(rng.sample(group, min(per_stratum, len(group))))
    return sampled[:n]


def signal_boosted_sample(entries: list[LogEntry], n: int, seed: int = 42) -> list[LogEntry]:
    """Oversamples 'interesting' entries: negative feedback, retries, high latency.
    These are the most valuable eval candidates."""
    rng = random.Random(seed)
    latencies = sorted(e.latency_ms for e in entries)
    p90_latency = latencies[int(len(latencies) * 0.9)] if latencies else float("inf")

    def is_interesting(e: LogEntry) -> bool:
        return e.user_feedback == "thumbs_down" or e.retried or e.latency_ms >= p90_latency

    interesting = [e for e in entries if is_interesting(e)]
    boring = [e for e in entries if not is_interesting(e)]

    # 70% interesting / 30% boring split.
    n_interesting = min(len(interesting), int(n * 0.7))
    n_boring = min(len(boring), n - n_interesting)
    return rng.sample(interesting, n_interesting) + rng.sample(boring, n_boring)


SAMPLERS = {"uniform": uniform_sample, "stratified": stratified_sample, "signal_boosted": signal_boosted_sample}
