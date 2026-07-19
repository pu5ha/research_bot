"""Fetch → dedup(seen) → embed → score, storing scored papers. Shared by
``run-once`` (M3+) and ``calibrate`` (M5). Sending is layered on top later.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from .config import Config
from .db import blob_to_embedding, insert_paper, now_utc_iso
from .dedup import Candidate, drop_near_duplicates, filter_unseen
from .embed import embed
from .models import Paper
from .score import ScoreResult, score_matrix
from .sources import FETCHERS
from .summarize import summarize_paper
from .taste import load_taste_vectors
from .telegram import send_paper

log = logging.getLogger("research-bot")

_MAX_CHARS = 1500  # truncate title+abstract before embedding (spec §6)


def candidate_text(paper: Paper) -> str:
    return f"{paper.title}. {paper.abstract}"[:_MAX_CHARS]


def _stored_embeddings(conn: sqlite3.Connection, uids: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    chunk = 500  # stay under SQLite's variable limit
    for i in range(0, len(uids), chunk):
        part = uids[i : i + chunk]
        q = f"SELECT uid, embedding FROM papers WHERE uid IN ({','.join('?' * len(part))})"
        for r in conn.execute(q, part):
            if r["embedding"] is not None:
                out[r["uid"]] = blob_to_embedding(r["embedding"])
    return out


def embed_papers(
    cfg: Config, conn: sqlite3.Connection, papers: list[Paper]
) -> np.ndarray:
    """Embed papers, reusing any already-stored vectors so we never re-embed (spec §12).

    Returns an ``(n, dim)`` array row-aligned to ``papers``.
    """
    if not papers:
        return np.empty((0, 0), dtype=np.float32)
    vecs = _stored_embeddings(conn, [p.uid for p in papers])
    misses = [p for p in papers if p.uid not in vecs]
    if misses:
        fresh = embed([candidate_text(p) for p in misses], model=cfg.model)
        for p, v in zip(misses, fresh):
            vecs[p.uid] = v
    return np.vstack([vecs[p.uid] for p in papers])


def fetch_all(cfg: Config, days: int | None = None) -> list[Paper]:
    """Fetch every enabled source; one failing source never aborts the run (spec §12).

    ``days`` overrides every source's window uniformly (used by ``calibrate``);
    when ``None``, each source uses its configured ``windows`` entry (default 2).
    """
    out: list[Paper] = []
    for name, sc in cfg.sources.items():
        if not sc.enabled:
            continue
        fetcher = FETCHERS.get(name)
        if fetcher is None:
            log.info("source %s enabled but not implemented yet; skipping", name)
            continue
        window = days if days is not None else cfg.windows.get(name, 2)
        try:
            papers = fetcher(cfg, window)
            log.info("source %s: fetched %d (last %dd)", name, len(papers), window)
            out.extend(papers)
        except Exception as exc:  # noqa: BLE001 — resilience is the whole point here
            log.warning("source %s failed: %s", name, exc)
    return out


@dataclass
class SendPlan:
    """The outcome of one selection cycle."""

    fetched: int = 0
    new: int = 0
    qualifiers: int = 0  # after bar + age + near-dup dedup
    already_sent_today: int = 0
    slots: int = 0
    # All newly-seen papers (post-embedding), for persistence — everything here gets
    # marked seen; those not in ``to_send`` are dropped by design (spec §8).
    new_scored: list[Candidate] = field(default_factory=list)
    # ``(paper, result, embedding)`` for the papers that would be sent this cycle.
    to_send: list[Candidate] = field(default_factory=list)


def recent_sent_vectors(
    conn: sqlite3.Connection, days: int = 7
) -> list[tuple[float, np.ndarray]]:
    """(score, embedding) for papers sent within the last ``days`` (near-dup window)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT s.score AS score, p.embedding AS embedding "
        "FROM sent s JOIN papers p ON p.uid = s.uid WHERE s.sent_at >= ?",
        (cutoff,),
    ).fetchall()
    return [
        (r["score"] or 0.0, blob_to_embedding(r["embedding"]))
        for r in rows
        if r["embedding"] is not None
    ]


def sent_today(conn: sqlite3.Connection) -> int:
    """Count sends in the current *local* calendar day (the daily cap, spec §8).

    Stored ``sent_at`` is UTC; we compare against local midnight converted to UTC so
    the cap resets cleanly at 00:00 local time. This replaced a rolling-24h window,
    under which a mid-afternoon batch still counted against the next morning's quota.
    """
    local_midnight = (
        datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    cutoff = local_midnight.astimezone(timezone.utc).isoformat(timespec="seconds")
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM sent WHERE sent_at >= ?", (cutoff,)
    ).fetchone()
    return int(row["c"])


def plan_from_scored(
    new: list[Paper],
    results: list[ScoreResult],
    emb: np.ndarray,
    recent: list[tuple[float, np.ndarray]],
    *,
    bars: dict[str, float],
    dedup_sim: float,
    max_age_hours: int,
    max_per_day: int,
    already_sent_today: int,
    now: datetime,
) -> SendPlan:
    """Pure §8 selection: age filter → per-source bar → near-dup dedup → sort → daily cap."""
    age_cutoff = now - timedelta(hours=max_age_hours)
    candidates: list[Candidate] = []
    for i, (p, r) in enumerate(zip(new, results)):
        if p.published is not None and p.published < age_cutoff:
            continue
        if r.score >= bars.get(p.source, 1.0):
            candidates.append((p, r, emb[i]))

    deduped = drop_near_duplicates(candidates, recent, dedup_sim)
    deduped.sort(key=lambda t: -t[1].score)
    slots = max(0, max_per_day - already_sent_today)
    return SendPlan(
        new=len(new),
        qualifiers=len(deduped),
        already_sent_today=already_sent_today,
        slots=slots,
        to_send=deduped[:slots],
    )


def run_cycle(cfg: Config, conn: sqlite3.Connection) -> SendPlan:
    """Fetch → seen-dedup → embed → score → §8 selection. Read-only; no writes.

    Persisting papers and recording sends happen in the send step (M7).
    """
    fetched = fetch_all(cfg)
    new = filter_unseen(conn, fetched)
    log.info("fetched %d, new after seen-dedup %d", len(fetched), len(new))
    if not new:
        already = sent_today(conn)
        return SendPlan(
            fetched=len(fetched),
            already_sent_today=already,
            slots=max(0, cfg.max_per_day - already),
        )

    emb = embed_papers(cfg, conn, new)
    pos, labels, neg = load_taste_vectors(conn)
    if pos.shape[0] == 0:
        log.warning("taste profile is empty — run `refresh-taste` first")
    results = score_matrix(emb, pos, labels, neg, cfg.lambda_neg)

    plan = plan_from_scored(
        new,
        results,
        emb,
        recent_sent_vectors(conn, days=7),
        bars=cfg.bars,
        dedup_sim=cfg.dedup_sim,
        max_age_hours=cfg.max_age_hours,
        max_per_day=cfg.max_per_day,
        already_sent_today=sent_today(conn),
        now=datetime.now(timezone.utc),
    )
    plan.fetched = len(fetched)
    plan.new_scored = [(p, results[i], emb[i]) for i, p in enumerate(new)]
    return plan


def persist_and_send(cfg: Config, conn: sqlite3.Connection, plan: SendPlan) -> int:
    """Mark all new papers seen, then send the selected ones. Returns count sent.

    Idempotency (spec §10): claim each uid in ``sent`` with INSERT OR IGNORE *before*
    the Telegram call, so a crash or retry can never double-ping. A send that fails
    after the claim is a miss, not a duplicate — the priority the spec sets.
    """
    for p, r, e in plan.new_scored:
        insert_paper(conn, p, e, r.score, r.nearest)
    conn.commit()

    sent = 0
    for p, r, _ in plan.to_send:
        cur = conn.execute(
            "INSERT OR IGNORE INTO sent (uid, telegram_message_id, sent_at, score) "
            "VALUES (?, NULL, ?, ?)",
            (p.uid, now_utc_iso(), r.score),
        )
        if cur.rowcount == 0:
            continue  # already sent in a prior run — never ping twice
        conn.commit()
        try:
            message_id = send_paper(cfg, p, summarize_paper(cfg, p))
        except httpx.HTTPError as exc:
            log.warning("send failed for %s: %s (claimed; will not retry)", p.uid, exc)
            message_id = None
        if message_id is not None:
            conn.execute(
                "UPDATE sent SET telegram_message_id = ? WHERE uid = ?",
                (message_id, p.uid),
            )
            conn.commit()
            sent += 1
    return sent


def select_unsent_qualifiers(
    conn: sqlite3.Connection, cfg: Config, *, ignore_age: bool = False
) -> list[Candidate]:
    """Highest-scoring stored papers that clear their per-source bar and were never sent.

    Backs the ``send-now`` admin command: seed or re-kick the feed from papers the
    daily cap left undelivered. ``ignore_age`` relaxes the freshness window for a
    one-off manual send. Returned as ``(paper, result, embedding)`` so it feeds
    straight into :func:`persist_and_send`.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.max_age_hours)
    rows = conn.execute(
        "SELECT p.* FROM papers p LEFT JOIN sent s ON s.uid = p.uid "
        "WHERE s.uid IS NULL ORDER BY p.score DESC"
    ).fetchall()
    out: list[Candidate] = []
    for r in rows:
        score = r["score"] or 0.0
        if score < cfg.bars.get(r["source"], 1.0):
            continue
        published = datetime.fromisoformat(r["published"]) if r["published"] else None
        if not ignore_age and published is not None and published < cutoff:
            continue
        if r["embedding"] is None:
            continue
        paper = Paper(
            uid=r["uid"],
            source=r["source"],
            title=r["title"],
            abstract=r["abstract"] or "",
            authors=json.loads(r["authors"] or "[]"),
            url=r["url"] or "",
            categories=json.loads(r["categories"] or "[]"),
            published=published,
        )
        out.append(
            (paper, ScoreResult(score=score, nearest=r["nearest"] or ""), blob_to_embedding(r["embedding"]))
        )
    return out
