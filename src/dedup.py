"""Dedup. Layer 1: drop uids already in ``papers``. Layer 2: drop near-duplicate
topics (cosine > threshold) against higher-scoring peers + recently-sent papers.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from .models import Paper
from .score import ScoreResult

# A candidate carries its embedding so we can compare topics.
Candidate = tuple[Paper, ScoreResult, np.ndarray]


def filter_unseen(conn: sqlite3.Connection, papers: list[Paper]) -> list[Paper]:
    """Return only papers whose uid is not already stored (also de-dups within-batch)."""
    if not papers:
        return []
    existing = {r["uid"] for r in conn.execute("SELECT uid FROM papers")}
    out: list[Paper] = []
    seen_now: set[str] = set()
    for p in papers:
        if p.uid in existing or p.uid in seen_now:
            continue
        seen_now.add(p.uid)
        out.append(p)
    return out


def drop_near_duplicates(
    candidates: list[Candidate],
    recent: list[tuple[float, np.ndarray]],
    threshold: float,
) -> list[Candidate]:
    """Drop a candidate if it is > ``threshold`` cosine-similar to another item that
    scores equal-or-higher (spec §7). Compares against already-kept candidates and
    ``recent`` (score, embedding) pairs from papers sent in the rolling window.

    Processing highest-score-first means every kept candidate already outranks those
    that follow, so keeping the top of a near-duplicate cluster falls out naturally.
    Embeddings are L2-normalized, so the dot product is cosine.
    """
    kept: list[tuple[float, np.ndarray]] = list(recent)
    out: list[Candidate] = []
    for p, r, e in sorted(candidates, key=lambda t: -t[1].score):
        if any(s >= r.score and float(e @ ke) > threshold for s, ke in kept):
            continue
        out.append((p, r, e))
        kept.append((r.score, e))
    return out
