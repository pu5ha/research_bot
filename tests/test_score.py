"""M3: taste scoring math (pure numpy, no network)."""

from __future__ import annotations

import numpy as np

from src.score import score_matrix


def test_score_penalizes_nearest_negative() -> None:
    pos = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = ["A", "B"]
    neg = np.array([[0, 0, 1]], dtype=np.float32)
    emb = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)

    res = score_matrix(emb, pos, labels, neg, lambda_neg=0.5)

    # paper 0: on positive A, far from the negative -> 1.0
    assert res[0].nearest == "A"
    assert abs(res[0].score - 1.0) < 1e-6
    # paper 1: no positive overlap, sits on the negative -> 0 - 0.5*1
    assert abs(res[1].score - (-0.5)) < 1e-6


def test_score_with_no_negatives() -> None:
    pos = np.array([[1, 0]], dtype=np.float32)
    neg = np.empty((0, 0), dtype=np.float32)
    emb = np.array([[1, 0]], dtype=np.float32)

    res = score_matrix(emb, pos, ["X"], neg, lambda_neg=0.5)
    assert res[0].nearest == "X"
    assert abs(res[0].score - 1.0) < 1e-6


def test_empty_input_returns_empty() -> None:
    assert score_matrix(np.empty((0, 3), dtype=np.float32), np.empty((0, 0)), [], np.empty((0, 0)), 0.5) == []
