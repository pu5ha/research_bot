"""M4: source parsers, exercised against saved payloads (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.sources import arxiv, biorxiv, iacr, nber

FIXTURES = Path(__file__).parent / "fixtures"


def test_biorxiv_parse_and_category_filter() -> None:
    data = json.loads((FIXTURES / "biorxiv_details.json").read_text())
    collection = data["collection"]

    # No filter -> all items map through.
    papers = biorxiv.parse_collection(collection, "biorxiv", categories=[])
    assert len(papers) == len(collection)
    p = papers[0]
    assert p.uid.startswith("doi:")
    assert p.source == "biorxiv"
    assert p.url == f"https://www.biorxiv.org/content/{p.uid.removeprefix('doi:')}"
    assert p.authors  # semicolon string was split into a list
    assert isinstance(p.published, datetime)

    # Category allow-list drops non-matching categories (case-insensitive).
    only_neuro = biorxiv.parse_collection(collection, "biorxiv", ["Neuroscience"])
    assert only_neuro
    assert all(c.lower() == "neuroscience" for pp in only_neuro for c in pp.categories)
    assert len(only_neuro) < len(papers)


def test_biorxiv_medrxiv_url_base() -> None:
    data = json.loads((FIXTURES / "biorxiv_details.json").read_text())
    papers = biorxiv.parse_collection(data["collection"], "medrxiv", categories=[])
    assert papers[0].source == "medrxiv"
    assert papers[0].url.startswith("https://www.medrxiv.org/content/")


def test_iacr_parse() -> None:
    papers = iacr.parse_feed((FIXTURES / "iacr_rss.xml").read_text())
    assert len(papers) == 3
    p = papers[0]
    assert p.uid.startswith("iacr:")
    # report id is YYYY/NNN
    assert p.uid.split(":", 1)[1].count("/") == 1
    assert p.source == "iacr"
    assert p.abstract
    assert isinstance(p.published, datetime)


def test_nber_parse_with_date_fallback() -> None:
    fetch_dt = datetime(2026, 7, 18, tzinfo=timezone.utc)
    papers = nber.parse_feed((FIXTURES / "nber_rss.xml").read_text(), fetch_dt)
    assert len(papers) == 2
    assert papers[0].uid == "nber:w34123"
    assert papers[0].published == datetime(2026, 7, 15, tzinfo=timezone.utc)
    # Second item has no pubDate -> falls back to the fetch date.
    assert papers[1].uid == "nber:w34124"
    assert papers[1].published == fetch_dt


def test_arxiv_short_id_strips_version() -> None:
    assert arxiv._short_id("http://arxiv.org/abs/2501.01234v3") == "2501.01234"
    assert arxiv._short_id("http://arxiv.org/abs/cs/0501001v1") == "cs/0501001"
