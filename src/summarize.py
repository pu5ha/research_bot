"""Local-LLM (Ollama) 3-sentence summaries for the papers being sent.

Runs only for the ≤ max_per_day papers actually pinged, so it's cheap. Falls back
to a truncated abstract whenever Ollama is unreachable or returns nothing.
"""

from __future__ import annotations

import logging

import httpx

from .config import Config
from .models import Paper

log = logging.getLogger("research-bot")

_PROMPT = (
    "Summarize this research paper in exactly 3 concise sentences for a technically "
    "literate reader. Focus on what is new and why it matters. Output only the "
    "summary, with no preamble or list formatting.\n\n"
    "Title: {title}\n\nAbstract: {abstract}"
)


def _fallback(paper: Paper) -> str:
    abstract = " ".join((paper.abstract or "").split())
    return (abstract[:300] + "…") if len(abstract) > 300 else abstract


def summarize_paper(cfg: Config, paper: Paper) -> str:
    """Return a 3-sentence summary, or a truncated abstract on any failure."""
    text = paper.abstract or paper.title
    if not cfg.summarizer.enabled or not text:
        return _fallback(paper)

    prompt = _PROMPT.format(title=paper.title, abstract=text)
    try:
        resp = httpx.post(
            f"{cfg.summarizer.url}/api/generate",
            json={
                "model": cfg.summarizer.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=cfg.summarizer.timeout,
        )
        resp.raise_for_status()
        summary = " ".join((resp.json().get("response") or "").split())
        return summary or _fallback(paper)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("summarize failed for %s: %s (using abstract)", paper.uid, exc)
        return _fallback(paper)
