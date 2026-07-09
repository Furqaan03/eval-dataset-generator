"""Cluster interactions by semantic similarity (HDBSCAN) + outlier detection.

Embedder injected so the clustering/outlier wiring is testable offline."""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.ingestion.schema import LogEntry


def cluster_entries(
    entries: list[LogEntry],
    embedder: Callable[[str], np.ndarray],
    min_cluster_size: int = 3,
) -> dict[int, list[LogEntry]]:
    """Returns cluster_label -> entries. Label -1 is the noise/outlier cluster
    (novel requests that don't fit existing patterns — prime eval candidates)."""
    if len(entries) < min_cluster_size:
        return {-1: entries}

    import hdbscan

    vectors = np.vstack([embedder(e.user_prompt) for e in entries])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(vectors)

    clusters: dict[int, list[LogEntry]] = {}
    for label, entry in zip(labels, entries):
        clusters.setdefault(int(label), []).append(entry)
    return clusters


def identify_outliers(clusters: dict[int, list[LogEntry]]) -> list[LogEntry]:
    """Outliers = the noise cluster + any interaction the user retried or downvoted."""
    outliers = list(clusters.get(-1, []))
    for label, entries in clusters.items():
        if label == -1:
            continue
        for e in entries:
            if e.retried or e.user_feedback == "thumbs_down":
                outliers.append(e)
    return outliers
