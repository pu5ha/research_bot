"""IACR Cryptology ePrint via RSS. Full abstracts + dates; refresh ≤ once/day."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import feedparser

from ..config import Config
from ..models import Paper
from .base import get_with_retries

log = logging.getLogger("research-bot")

NAME = "iacr"
RSS_URL = "https://eprint.iacr.org/rss/rss.xml"


def _clean(text: str) -> str:
    return " ".join(text.split())


def _report_id(link: str) -> str:
    """``https://eprint.iacr.org/2026/488`` -> ``2026/488``."""
    return link.rstrip("/").split("eprint.iacr.org/")[-1]


def parse_feed(text: str) -> list[Paper]:
    """Parse the ePrint RSS into Papers (pure; no network)."""
    feed = feedparser.parse(text)
    papers: list[Paper] = []
    for e in feed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        link = e.get("link", "")
        papers.append(
            Paper(
                uid="iacr:" + _report_id(link),
                source=NAME,
                title=_clean(e.get("title", "")),
                abstract=_clean(e.get("summary", "")),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                url=link,
                categories=[t.get("term", "") for t in e.get("tags", [])],
                published=published,
            )
        )
    return papers


def fetch_recent(cfg: Config, days: int = 2) -> list[Paper]:
    """Fetch recent ePrint papers, keeping only the last ``days`` (RSS includes
    revised older papers per spec §4).
    """
    papers = parse_feed(get_with_retries(RSS_URL, cfg).text)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return [p for p in papers if p.published is None or p.published >= cutoff]
