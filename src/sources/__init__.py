"""Source registry: name -> ``fetch(cfg, days) -> list[Paper]``. New sources register here."""

from __future__ import annotations

from ..config import Config
from ..models import Paper
from . import arxiv, biorxiv, iacr, nber

FETCHERS = {
    "arxiv": lambda cfg, days: arxiv.fetch_recent(cfg, days=days),
    "biorxiv": lambda cfg, days: biorxiv.fetch_recent(cfg, "biorxiv", days=days),
    "medrxiv": lambda cfg, days: biorxiv.fetch_recent(cfg, "medrxiv", days=days),
    "iacr": lambda cfg, days: iacr.fetch_recent(cfg, days=days),
    "nber": lambda cfg, days: nber.fetch_recent(cfg, days=days),
}


def fetch_source(cfg: Config, name: str, days: int) -> list[Paper]:
    """Fetch one source over a ``days`` window. Raises KeyError for unknown names."""
    return FETCHERS[name](cfg, days)
