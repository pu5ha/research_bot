"""M6: near-duplicate dedup (§7) and the §8 send-selection logic (pure)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.dedup import drop_near_duplicates
from src.models import Paper
from src.pipeline import plan_from_scored
from src.score import ScoreResult

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _unit(vec: list[float]) -> np.ndarray:
    a = np.array(vec, dtype=np.float32)
    return a / np.linalg.norm(a)


def _cand(uid: str, score: float, vec: list[float], *, source="arxiv", age_h=1.0):
    p = Paper(
        uid=uid,
        source=source,
        title=uid,
        abstract="",
        published=NOW - timedelta(hours=age_h),
    )
    return p, ScoreResult(score, "seed"), _unit(vec)


def test_drop_near_duplicate_keeps_higher_score() -> None:
    a = _cand("a", 0.90, [1, 0, 0])
    b = _cand("b", 0.85, [1, 0.02, 0])  # ~identical topic, lower score -> dropped
    c = _cand("c", 0.80, [0, 1, 0])     # different topic -> kept
    kept = {p.uid for p, _, _ in drop_near_duplicates([a, b, c], [], threshold=0.90)}
    assert kept == {"a", "c"}


def test_drop_near_duplicate_against_recent_sent() -> None:
    recent = [(0.95, _unit([1, 0, 0]))]  # already sent, higher score
    cand = _cand("x", 0.80, [1, 0.01, 0])
    assert drop_near_duplicates([cand], recent, threshold=0.90) == []


def _plan(cands, **kw):
    new = [c[0] for c in cands]
    results = [c[1] for c in cands]
    emb = np.vstack([c[2] for c in cands])
    defaults = dict(
        bars={"arxiv": 0.60, "biorxiv": 0.60},
        dedup_sim=0.90,
        max_age_hours=48,
        max_per_day=3,
        already_sent_today=0,
        now=NOW,
    )
    defaults.update(kw)
    return plan_from_scored(new, results, emb, [], **defaults)


def test_plan_bar_filter_and_cap() -> None:
    cands = [
        _cand("a", 0.90, [1, 0, 0]),
        _cand("b", 0.80, [0, 1, 0]),
        _cand("c", 0.70, [0, 0, 1]),
        _cand("d", 0.50, [1, 1, 1]),  # below bar -> excluded
    ]
    plan = _plan(cands)
    assert plan.qualifiers == 3  # a, b, c pass the 0.60 bar
    assert [p.uid for p, _, _ in plan.to_send] == ["a", "b", "c"]  # capped at 3, score desc


def test_plan_respects_remaining_slots() -> None:
    cands = [_cand("a", 0.90, [1, 0, 0]), _cand("b", 0.80, [0, 1, 0])]
    plan = _plan(cands, already_sent_today=2)  # only 1 slot left
    assert plan.slots == 1
    assert [p.uid for p, _, _ in plan.to_send] == ["a"]


def test_plan_drops_stale_papers() -> None:
    cands = [
        _cand("fresh", 0.90, [1, 0, 0], age_h=1),
        _cand("stale", 0.95, [0, 1, 0], age_h=100),  # older than 48h -> excluded
    ]
    plan = _plan(cands)
    assert [p.uid for p, _, _ in plan.to_send] == ["fresh"]
