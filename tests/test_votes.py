"""M8: callback handling, offset tracking, and vote recording (Telegram mocked)."""

from __future__ import annotations

import numpy as np

from src import votes
from src.db import (
    connect,
    embedding_to_blob,
    get_offset,
    init_schema,
    now_utc_iso,
)
from src.votes import drain_once


def _seed_paper(conn, uid: str) -> None:
    conn.execute(
        "INSERT INTO papers (uid, title, embedding, first_seen) VALUES (?, ?, ?, ?)",
        (uid, uid, embedding_to_blob(np.ones(4, dtype=np.float32) / 2.0), now_utc_iso()),
    )
    conn.commit()


def _mute_telegram(monkeypatch) -> list[tuple[str, str]]:
    """Silence outbound Telegram calls; return a log of (kind, arg) for assertions."""
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(votes.telegram, "answer_callback", lambda cfg, cid, text="": events.append(("answer", cid)))
    monkeypatch.setattr(votes.telegram, "mark_message_voted", lambda cfg, c, m, label: events.append(("edit", label)))
    return events


def _update(update_id: int, uid: str, action: str) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cb{update_id}",
            "data": f"{action}:{uid}",
            "message": {"message_id": update_id, "chat": {"id": 42}},
        },
    }


def test_drain_records_votes_and_advances_offset(tmp_path, monkeypatch) -> None:
    events = _mute_telegram(monkeypatch)
    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        _seed_paper(conn, "arxiv:a")
        _seed_paper(conn, "arxiv:b")
        monkeypatch.setattr(
            votes.telegram,
            "get_updates",
            lambda cfg, offset, timeout=0: [
                _update(10, "arxiv:a", "up"),
                _update(11, "arxiv:b", "down"),
            ],
        )

        n = drain_once(None, conn, timeout=0)
        assert n == 2
        rows = dict(conn.execute("SELECT uid, vote FROM votes").fetchall())
        assert rows == {"arxiv:a": 1, "arxiv:b": -1}
        assert get_offset(conn) == 12  # last update_id (11) + 1
        assert ("edit", "👍 logged") in events
        assert ("edit", "👎 logged") in events
    finally:
        conn.close()


def test_revote_overwrites(tmp_path, monkeypatch) -> None:
    _mute_telegram(monkeypatch)
    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        _seed_paper(conn, "arxiv:a")
        seq = [[_update(10, "arxiv:a", "up")], [_update(11, "arxiv:a", "down")]]
        monkeypatch.setattr(votes.telegram, "get_updates", lambda cfg, offset, timeout=0: seq.pop(0))

        drain_once(None, conn)
        assert conn.execute("SELECT vote FROM votes WHERE uid='arxiv:a'").fetchone()[0] == 1
        drain_once(None, conn)
        assert conn.execute("SELECT vote FROM votes WHERE uid='arxiv:a'").fetchone()[0] == -1
    finally:
        conn.close()


def test_noop_callback_still_answers(tmp_path, monkeypatch) -> None:
    events = _mute_telegram(monkeypatch)
    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        monkeypatch.setattr(
            votes.telegram,
            "get_updates",
            lambda cfg, offset, timeout=0: [
                {"update_id": 5, "callback_query": {"id": "cb5", "data": "noop", "message": {}}}
            ],
        )
        n = drain_once(None, conn)
        assert n == 0  # no vote recorded
        assert ("answer", "cb5") in events  # but spinner was stopped
        assert get_offset(conn) == 6
    finally:
        conn.close()
