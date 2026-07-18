"""`calibrate` — score recent history against the taste profile and print the
bar→papers/day tables used to pick per-source bars. Sends and stores nothing.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict

from .config import Config
from .models import Paper
from .pipeline import embed_papers, fetch_all
from .score import ScoreResult, score_matrix
from .taste import load_taste_vectors

log = logging.getLogger("research-bot")

_SWEEP = [0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86]


def _unique(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    out: list[Paper] = []
    for p in papers:
        if p.uid not in seen:
            seen.add(p.uid)
            out.append(p)
    return out


def run_calibrate(cfg: Config, conn: sqlite3.Connection, days: int) -> None:
    papers = _unique(fetch_all(cfg, days=days))
    if not papers:
        print("\nNo papers fetched — nothing to calibrate.")
        return

    emb = embed_papers(cfg, conn, papers)
    pos, labels, neg = load_taste_vectors(conn)
    if pos.shape[0] == 0:
        print("\nTaste profile is empty — run `refresh-taste` first.")
        return
    results = score_matrix(emb, pos, labels, neg, cfg.lambda_neg)
    scored = list(zip(papers, results))

    _print_volume(scored)
    _print_bar_sweep(scored, days)
    _print_projection(scored, cfg, days)
    _print_would_send(scored, cfg)


def _by_source(scored: list[tuple[Paper, ScoreResult]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for p, r in scored:
        out[p.source].append(r.score)
    return out


def _print_volume(scored: list[tuple[Paper, ScoreResult]]) -> None:
    src = _by_source(scored)
    print(f"\n=== calibrate: {len(scored)} unique papers ===\n")
    print("volume by source:")
    for name in sorted(src, key=lambda k: -len(src[k])):
        print(f"  {name:9} {len(src[name]):6}")


def _print_bar_sweep(scored: list[tuple[Paper, ScoreResult]], days: int) -> None:
    src = _by_source(scored)
    header = "  " + f"{'source':9} {'n':>6}  " + "  ".join(f"{b:>5.2f}" for b in _SWEEP)
    print("\nbar sweep (papers/day at each bar):")
    print(header)
    for name in sorted(src):
        scores = src[name]
        cells = "  ".join(
            f"{sum(s >= b for s in scores) / days:>5.2f}" for b in _SWEEP
        )
        print(f"  {name:9} {len(scores):>6}  {cells}")
    print(f"  (papers/day = count(score >= bar) / {days} days)")


def _qualifiers(
    scored: list[tuple[Paper, ScoreResult]], bars: dict[str, float]
) -> list[tuple[Paper, ScoreResult]]:
    return [
        (p, r) for p, r in scored if r.score >= bars.get(p.source, 1.0)
    ]


def _print_projection(
    scored: list[tuple[Paper, ScoreResult]], cfg: Config, days: int
) -> None:
    src = _by_source(scored)
    quals = _qualifiers(scored, cfg.bars)
    print("\nprojection at current bars:")
    print(f"  {'source':9} {'bar':>5}  {'papers/day':>10}")
    total_per_day = 0.0
    for name in sorted(src):
        bar = cfg.bars.get(name, 1.0)
        n = sum(1 for p, _ in quals if p.source == name)
        per_day = n / days
        total_per_day += per_day
        print(f"  {name:9} {bar:>5.2f}  {per_day:>10.2f}")
    capped = min(total_per_day, float(cfg.max_per_day))
    print(f"  {'TOTAL':9} {'':>5}  {total_per_day:>10.2f}  (cap {cfg.max_per_day}/day → ~{capped:.2f})")

    # Quiet-day estimate: fraction of days with zero qualifiers.
    active_days = {
        p.published.date() for p, _ in quals if p.published is not None
    }
    span = max(days, len(active_days))
    quiet_frac = 1.0 - (len(active_days) / span) if span else 0.0
    print(f"  estimated quiet days: {quiet_frac * 100:.0f}%")


def _print_would_send(
    scored: list[tuple[Paper, ScoreResult]], cfg: Config, limit: int = 40
) -> None:
    quals = sorted(_qualifiers(scored, cfg.bars), key=lambda pr: -pr[1].score)
    print(f"\nwould send at current bars ({len(quals)} papers, top {min(limit, len(quals))}):")
    for p, r in quals[:limit]:
        print(f"  [{r.score:.3f}] {p.source:8} {p.title[:70]}")
        print(f"           ~ {r.nearest}")
    if len(quals) > limit:
        print(f"  … and {len(quals) - limit} more")
