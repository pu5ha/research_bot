"""Canonical internal shapes shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """A single research item, normalized from any source.

    Every source maps its native payload onto this shape. ``uid`` is the stable
    dedup key, e.g. ``arxiv:2501.01234`` / ``doi:10.1101/...`` / ``iacr:2026/1234``
    / ``nber:w34123``.
    """

    uid: str
    source: str  # "arxiv" | "biorxiv" | "medrxiv" | "iacr" | "nber"
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    url: str = ""
    categories: list[str] = field(default_factory=list)
    published: datetime | None = None  # UTC; None until a source fills it in
