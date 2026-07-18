"""Telegram send. ``poll_votes`` (the 👍/👎 loop) arrives in M8."""

from __future__ import annotations

import html
import json
import logging

import httpx

from .config import Config
from .models import Paper

log = logging.getLogger("research-bot")

API = "https://api.telegram.org"


def _esc(text: str) -> str:
    return html.escape(text or "")


def build_message(paper: Paper, summary: str) -> str:
    """The HTML message body: title, source tag, local-LLM summary, link."""
    lines = [f"🔬 <b>{_esc(paper.title)}</b>"]
    if paper.source:
        lines.append(f"<i>{_esc(paper.source)}</i>")
    if summary:
        lines.append(_esc(summary))
    lines.append(_esc(paper.url))
    return "\n".join(lines)


def build_keyboard(uid: str) -> dict:
    """Inline 👍/👎 keyboard; callback_data is parsed back in ``poll-votes`` (M8)."""
    return {
        "inline_keyboard": [
            [
                {"text": "👍", "callback_data": f"up:{uid}"},
                {"text": "👎", "callback_data": f"down:{uid}"},
            ]
        ]
    }


def is_configured(cfg: Config) -> bool:
    return bool(cfg.secrets.telegram_bot_token and cfg.secrets.telegram_chat_id)


def send_paper(cfg: Config, paper: Paper, summary: str) -> int | None:
    """POST one paper to Telegram; return the message_id, or None on a non-ok reply."""
    if not is_configured(cfg):
        raise RuntimeError("Telegram not configured (set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
    payload = {
        "chat_id": cfg.secrets.telegram_chat_id,
        "text": build_message(paper, summary),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": json.dumps(build_keyboard(paper.uid)),
    }
    resp = httpx.post(
        f"{API}/bot{cfg.secrets.telegram_bot_token}/sendMessage",
        data=payload,
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        log.warning("telegram sendMessage not ok: %s", data)
        return None
    return int(data["result"]["message_id"])


def _api(cfg: Config, method: str) -> str:
    return f"{API}/bot{cfg.secrets.telegram_bot_token}/{method}"


def get_updates(cfg: Config, offset: int, timeout: int = 25) -> list[dict]:
    """Long-poll for updates from ``offset``. Returns the raw update list."""
    resp = httpx.get(
        _api(cfg, "getUpdates"),
        params={"offset": offset, "timeout": timeout},
        timeout=timeout + 10,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", []) if data.get("ok") else []


def answer_callback(cfg: Config, callback_query_id: str, text: str = "") -> None:
    """Stop the button's spinner (and optionally show a toast)."""
    httpx.post(
        _api(cfg, "answerCallbackQuery"),
        data={"callback_query_id": callback_query_id, "text": text},
        timeout=30,
        follow_redirects=True,
    )


def mark_message_voted(cfg: Config, chat_id: int, message_id: int, label: str) -> None:
    """Replace the 👍/👎 keyboard with a single confirmation button."""
    keyboard = {"inline_keyboard": [[{"text": label, "callback_data": "noop"}]]}
    httpx.post(
        _api(cfg, "editMessageReplyMarkup"),
        data={
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": json.dumps(keyboard),
        },
        timeout=30,
        follow_redirects=True,
    )
