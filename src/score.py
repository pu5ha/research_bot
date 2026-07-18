"""Taste scoring: max-similarity to positives, penalized by nearest negative.

Everything is L2-normalized, so cosine == dot product. For each paper:
    pos_sim = max_j  paper · positive_j          (nearest favorite)
    neg_sim = max_k  paper · negative_k          (nearest disliked; 0 if none)
    score   = pos_sim - lambda_neg * neg_sim
    nearest = label of the argmax positive       (shown in the ping)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreResult:
    score: float
    nearest: str


def score_matrix(
    emb: np.ndarray,
    pos: np.ndarray,
    pos_labels: list[str],
    neg: np.ndarray,
    lambda_neg: float,
) -> list[ScoreResult]:
    """Score each row of ``emb`` (an ``(n, dim)`` array of paper embeddings)."""
    n = emb.shape[0]
    if n == 0:
        return []

    if pos.shape[0]:
        psim = emb @ pos.T  # (n, P)
        pos_sim = psim.max(axis=1)
        pos_arg = psim.argmax(axis=1)
    else:
        pos_sim = np.zeros(n, dtype=np.float32)
        pos_arg = None

    if neg.shape[0]:
        neg_sim = (emb @ neg.T).max(axis=1)  # (n, N) -> (n,)
    else:
        neg_sim = np.zeros(n, dtype=np.float32)

    scores = pos_sim - lambda_neg * neg_sim
    return [
        ScoreResult(
            score=float(scores[i]),
            nearest=pos_labels[int(pos_arg[i])] if pos_arg is not None else "",
        )
        for i in range(n)
    ]
