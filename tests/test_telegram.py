"""M7: Telegram message/keyboard building + persist_and_send (send mocked)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src import pipeline
from src.db import connect, init_schema
from src.models import Paper
from src.pipeline import SendPlan, persist_and_send
from src.score import ScoreResult
from src.telegram import build_keyboard, build_message


def test_build_message_escapes_and_formats() -> None:
    p = Paper(
        uid="arxiv:1",
        source="arxiv",
        title="Attention & <Reasoning>",
        abstract="x",
        authors=["Ada Lovelace", "Alan Turing"],
        url="https://arxiv.org/abs/1",
    )
    msg = build_message(p, "A concise <summary> of the work.")
    assert "<b>Attention &amp; &lt;Reasoning&gt;</b>" in msg
    assert "<i>arxiv</i>" in msg
    assert "A concise &lt;summary&gt; of the work." in msg
    assert "https://arxiv.org/abs/1" in msg
    # Score/nearest/author lines are no longer shown.
    assert "similarity" not in msg
    assert "et al." not in msg


def test_build_keyboard_callback_data() -> None:
    kb = build_keyboard("doi:10.1101/abc")
    buttons = kb["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "up:doi:10.1101/abc"
    assert buttons[1]["callback_data"] == "down:doi:10.1101/abc"


def _cand(uid: str, score: float):
    p = Paper(uid=uid, source="arxiv", title=uid, abstract="", url="u",
              published=datetime(2026, 7, 18, tzinfo=timezone.utc))
    return p, ScoreResult(score, "seed"), np.ones(4, dtype=np.float32) / 2.0


def test_persist_and_send_marks_seen_records_sent_idempotently(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_send(cfg, paper, summary):
        calls.append(paper.uid)
        return 1000 + len(calls)

    monkeypatch.setattr(pipeline, "send_paper", fake_send)
    monkeypatch.setattr(pipeline, "summarize_paper", lambda cfg, paper: "summary")

    to_send = [_cand("arxiv:a", 0.9)]
    new_scored = to_send + [_cand("arxiv:b", 0.5)]  # b is new but below-bar / not sent
    plan = SendPlan(fetched=2, new=2, new_scored=new_scored, to_send=to_send)

    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        sent = persist_and_send(None, conn, plan)  # cfg unused by fake_send
        assert sent == 1
        assert calls == ["arxiv:a"]
        # Both new papers are marked seen; only 'a' is in sent (with a message_id).
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        row = conn.execute("SELECT telegram_message_id FROM sent WHERE uid='arxiv:a'").fetchone()
        assert row[0] == 1001

        # Idempotency: a second identical cycle sends nothing new.
        calls.clear()
        sent2 = persist_and_send(None, conn, plan)
        assert sent2 == 0
        assert calls == []
    finally:
        conn.close()
