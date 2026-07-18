"""bioRxiv + medRxiv via the details JSON API. Paginates 30/page; parse is pure."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..config import Config
from ..models import Paper
from .base import get_with_retries

log = logging.getLogger("research-bot")

API = "https://api.biorxiv.org"
_MAX_PAGES = 60  # runaway guard (~1800 papers); daily windows are far smaller

_BASE_URL = {
    "biorxiv": "https://www.biorxiv.org",
    "medrxiv": "https://www.medrxiv.org",
}


def _clean(text: str) -> str:
    return " ".join(text.split())


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_collection(
    collection: list[dict], server: str, categories: list[str]
) -> list[Paper]:
    """Map API ``collection[]`` items to Papers, filtering by category allow-list.

    Empty ``categories`` means accept all. bioRxiv categories are lowercase strings.
    """
    allowed = {c.lower() for c in categories} if categories else None
    base = _BASE_URL.get(server, _BASE_URL["biorxiv"])
    out: list[Paper] = []
    for item in collection:
        category = (item.get("category") or "").lower()
        if allowed is not None and category not in allowed:
            continue
        doi = item.get("doi", "")
        authors = [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()]
        out.append(
            Paper(
                uid="doi:" + doi,
                source=server,
                title=_clean(item.get("title", "")),
                abstract=_clean(item.get("abstract", "")),
                authors=authors,
                url=f"{base}/content/{doi}",
                categories=[item["category"]] if item.get("category") else [],
                published=_parse_date(item.get("date")),
            )
        )
    return out


def fetch_recent(cfg: Config, server: str, days: int = 2) -> list[Paper]:
    """Fetch the last ``days`` for one server, paging through the cursor."""
    sc = cfg.sources.get(server)
    categories = sc.categories if sc else []
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    papers: list[Paper] = []
    cursor = 0
    for _ in range(_MAX_PAGES):
        url = f"{API}/details/{server}/{start}/{end}/{cursor}/json"
        data = get_with_retries(url, cfg).json()
        msg = (data.get("messages") or [{}])[0]
        total = int(msg.get("total", 0) or 0)
        collection = data.get("collection", [])
        papers.extend(parse_collection(collection, server, categories))
        cursor += len(collection)
        if not collection or cursor >= total:
            break
    return papers
