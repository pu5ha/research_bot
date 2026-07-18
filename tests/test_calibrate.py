"""M5: calibrate filtering helpers (pure)."""

from __future__ import annotations

from src.calibrate import _qualifiers, _unique
from src.models import Paper
from src.score import ScoreResult


def _p(uid: str, source: str) -> Paper:
    return Paper(uid=uid, source=source, title="t", abstract="a")


def test_unique_keeps_first_occurrence() -> None:
    papers = [_p("a", "arxiv"), _p("b", "arxiv"), _p("a", "arxiv")]
    assert [p.uid for p in _unique(papers)] == ["a", "b"]


def test_qualifiers_uses_per_source_bars() -> None:
    scored = [
        (_p("a", "arxiv"), ScoreResult(0.62, "x")),   # >= 0.60 -> in
        (_p("b", "arxiv"), ScoreResult(0.58, "x")),   # < 0.60 -> out
        (_p("c", "iacr"), ScoreResult(0.55, "x")),    # >= 0.53 -> in
        (_p("d", "nber"), ScoreResult(0.40, "x")),    # < 0.55 -> out
    ]
    bars = {"arxiv": 0.60, "iacr": 0.53, "nber": 0.55}
    kept = {p.uid for p, _ in _qualifiers(scored, bars)}
    assert kept == {"a", "c"}


def test_qualifiers_unknown_source_excluded() -> None:
    scored = [(_p("z", "mystery"), ScoreResult(0.99, "x"))]
    assert _qualifiers(scored, {"arxiv": 0.6}) == []
