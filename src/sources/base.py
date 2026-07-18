"""Source protocol + a shared, polite HTTP GET with retries and backoff."""

from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

from ..config import Config
from ..models import Paper

log = logging.getLogger("research-bot")


class Source(Protocol):
    NAME: str

    def fetch_recent(self, cfg: Config) -> list[Paper]: ...


def get_with_retries(
    url: str,
    cfg: Config,
    *,
    params: dict | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: float = 30.0,
) -> httpx.Response:
    """GET with a descriptive User-Agent, redirect-following, and exponential backoff.

    ``follow_redirects=True`` is mandatory — arXiv's http endpoint 301s and silently
    returns nothing otherwise (spec §4).
    """
    headers = {"User-Agent": cfg.user_agent}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc
