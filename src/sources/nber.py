"""NBER working papers via RSS. Flaky: intermittently 403s or returns 0 entries,
and entries often lack a machine-readable date. Degrades gracefully to ``[]``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import feedparser

from ..config import Config
from ..models import Paper
from .base import get_with_retries

log = logging.getLogger("research-bot")

NAME = "nber"
RSS_URL = "https://www.nber.org/rss/new.xml"
_ATTEMPTS = 5  # spec §12: NBER gets 5 retries


def _clean(text: str) -> str:
    return " ".join(text.split())


def _paper_id(link: str) -> str:
    """``https://www.nber.org/papers/w34123`` -> ``w34123``."""
    return link.rstrip("/").split("/papers/")[-1].split("?")[0]


def parse_feed(text: str, fetch_dt: datetime) -> list[Paper]:
    """Parse NBER RSS. Missing dates fall back to ``fetch_dt`` (spec §4)."""
    feed = feedparser.parse(text)
    papers: list[Paper] = []
    for e in feed.entries:
        published = fetch_dt
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        link = e.get("link", "")
        abstract = e.get("summary", "") or e.get("description", "")
        papers.append(
            Paper(
                uid="nber:" + _paper_id(link),
                source=NAME,
                title=_clean(e.get("title", "")),
                abstract=_clean(abstract),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                url=link,
                categories=[],
                published=published,
            )
        )
    return papers


def fetch_recent(cfg: Config, days: int | None = None, attempts: int = _ATTEMPTS) -> list[Paper]:
    """Try up to ``attempts`` times; never raises. Returns ``[]`` if NBER is down.

    ``days`` optionally filters to recent entries (dateless entries keep the fetch
    date, so they always pass).
    """
    fetch_dt = datetime.now(timezone.utc)
    cutoff = fetch_dt - timedelta(days=days) if days is not None else None
    for attempt in range(1, attempts + 1):
        try:
            resp = get_with_retries(RSS_URL, cfg, retries=1, backoff=0.0)
            papers = parse_feed(resp.text, fetch_dt)
            if cutoff is not None:
                papers = [p for p in papers if p.published is None or p.published >= cutoff]
            if papers:
                return papers
            log.warning("nber: 0 entries (attempt %d/%d)", attempt, attempts)
        except Exception as exc:  # noqa: BLE001 — NBER must never abort the run
            log.warning("nber fetch failed (attempt %d/%d): %s", attempt, attempts, exc)
        if attempt < attempts:
            time.sleep(1)
    log.warning("nber: giving up after %d attempts (source unavailable)", attempts)
    return []
