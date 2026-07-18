"""Typed configuration: ``config.yaml`` (non-secret) + ``.env`` (secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Repo root = parent of the ``src`` package directory.
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SourceConfig:
    enabled: bool = False
    categories: list[str] = field(default_factory=list)


@dataclass
class SummarizerConfig:
    enabled: bool = True
    model: str = "llama3.2"
    url: str = "http://localhost:11434"
    timeout: float = 60.0


@dataclass
class Secrets:
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    contact_email: str | None = None


@dataclass
class Config:
    model: str
    max_per_day: int
    max_age_hours: int
    lambda_neg: float
    dedup_sim: float
    bars: dict[str, float]
    windows: dict[str, int]
    sources: dict[str, SourceConfig]
    summarizer: SummarizerConfig
    secrets: Secrets
    db_path: Path

    @property
    def user_agent(self) -> str:
        contact = self.secrets.contact_email or "unknown"
        return f"research-bot/0.1 (contact: {contact})"


def load_config(
    config_path: Path | str | None = None,
    env_path: Path | str | None = None,
) -> Config:
    """Load and type the config. Fails loudly if ``config.yaml`` is missing.

    Missing ``.env`` secrets are treated as ``None`` (M1 needs no secrets).
    """
    cfg_path = Path(config_path) if config_path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config file not found at {cfg_path}. "
            "Copy config.example.yaml to config.yaml and edit it."
        )

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # .env is optional; load_dotenv silently no-ops if the file is absent.
    load_dotenv(Path(env_path) if env_path else ROOT / ".env")

    sources = {
        name: SourceConfig(
            enabled=bool(spec.get("enabled", False)),
            categories=list(spec.get("categories", [])),
        )
        for name, spec in (raw.get("sources") or {}).items()
    }

    summ = raw.get("summarizer") or {}
    summarizer = SummarizerConfig(
        enabled=bool(summ.get("enabled", True)),
        model=str(summ.get("model", "llama3.2")),
        url=str(summ.get("url", "http://localhost:11434")),
        timeout=float(summ.get("timeout", 60.0)),
    )

    secrets = Secrets(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
        contact_email=os.getenv("CONTACT_EMAIL") or None,
    )

    return Config(
        model=str(raw.get("model", "BAAI/bge-small-en-v1.5")),
        max_per_day=int(raw.get("max_per_day", 3)),
        max_age_hours=int(raw.get("max_age_hours", 48)),
        lambda_neg=float(raw.get("lambda_neg", 0.5)),
        dedup_sim=float(raw.get("dedup_sim", 0.90)),
        bars={k: float(v) for k, v in (raw.get("bars") or {}).items()},
        windows={k: int(v) for k, v in (raw.get("windows") or {}).items()},
        sources=sources,
        summarizer=summarizer,
        secrets=secrets,
        db_path=ROOT / "data" / "bot.db",
    )
