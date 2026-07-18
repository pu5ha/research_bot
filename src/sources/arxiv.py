"""arXiv source via the Atom API. Parse is separated from fetch for offline tests."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser

from ..config import Config
from ..models import Paper
from .base import get_with_retries

log = logging.getLogger("research-bot")

NAME = "arxiv"
API_URL = "https://export.arxiv.org/api/query"  # https + follow_redirects (spec §4)
_PAGE_SIZE = 1000  # arXiv allows up to 1000/page
_MAX_PAGES = 15  # runaway guard (~15k papers)
_SLEEP = 3.0  # ≥3s between requests (spec §4)

_VERSION_RE = re.compile(r"v\d+$")


def _short_id(entry_id: str) -> str:
    """``http://arxiv.org/abs/2501.01234v2`` -> ``2501.01234`` (version stripped)."""
    tail = entry_id.rsplit("/abs/", 1)[-1]
    return _VERSION_RE.sub("", tail)


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse_feed(text: str) -> list[Paper]:
    """Parse an arXiv Atom response into Papers (pure; no network)."""
    feed = feedparser.parse(text)
    papers: list[Paper] = []
    for e in feed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        papers.append(
            Paper(
                uid="arxiv:" + _short_id(e.get("id", "")),
                source=NAME,
                title=_clean(e.get("title", "")),
                abstract=_clean(e.get("summary", "")),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                url=e.get("link", ""),
                categories=[t.get("term", "") for t in e.get("tags", [])],
                published=published,
            )
        )
    return papers


def _search_query(cfg: Config) -> str:
    sc = cfg.sources.get(NAME)
    categories = sc.categories if sc else []
    return " OR ".join(f"cat:{c}" for c in categories) or "cat:cs.LG"


def fetch_recent(cfg: Config, days: int = 2) -> list[Paper]:
    """Fetch papers submitted within the last ``days``, newest first.

    Pages through results (≥3s apart) until the oldest fetched paper predates the
    cutoff, then filters to the window. This matters for daily runs: the firehose
    is ~250–600/day, so a single 100-item page would miss most of a day.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    params_base = {
        "search_query": _search_query(cfg),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": _PAGE_SIZE,
    }

    collected: list[Paper] = []
    for page in range(_MAX_PAGES):
        params = {**params_base, "start": page * _PAGE_SIZE}
        batch = parse_feed(get_with_retries(API_URL, cfg, params=params).text)
        if not batch:
            break
        collected.extend(batch)
        oldest = min((p.published for p in batch if p.published), default=None)
        if len(batch) < _PAGE_SIZE or (oldest is not None and oldest < cutoff):
            break
        time.sleep(_SLEEP)

    return [p for p in collected if p.published is None or p.published >= cutoff]
