"""SQLite schema init, connection, and shared storage helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  uid TEXT PRIMARY KEY, source TEXT, title TEXT, abstract TEXT, authors TEXT,
  url TEXT, categories TEXT, published TEXT, first_seen TEXT,
  embedding BLOB,
  score REAL, nearest TEXT
);
CREATE TABLE IF NOT EXISTS sent (
  uid TEXT PRIMARY KEY, telegram_message_id INTEGER, sent_at TEXT, score REAL
);
CREATE TABLE IF NOT EXISTS votes (
  uid TEXT PRIMARY KEY, vote INTEGER, voted_at TEXT
);
CREATE TABLE IF NOT EXISTS taste (
  id TEXT PRIMARY KEY, kind TEXT, label TEXT, embedding BLOB
);
CREATE TABLE IF NOT EXISTS source_state (
  source TEXT PRIMARY KEY, last_run TEXT, last_cursor TEXT
);
"""


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating parent dir if needed) and return a configured connection."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # run-once (cron) and poll-votes (service) write concurrently; wait for the
    # lock instead of failing with "database is locked".
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def embedding_to_blob(vec: np.ndarray) -> bytes:
    """Serialize an embedding to a float32 blob for BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a float32 blob back into a numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


def record_vote(conn: sqlite3.Connection, uid: str, vote: int) -> None:
    """Upsert a 👍 (+1) / 👎 (-1) vote. Re-voting overwrites the prior row."""
    conn.execute(
        "INSERT OR REPLACE INTO votes (uid, vote, voted_at) VALUES (?, ?, ?)",
        (uid, vote, now_utc_iso()),
    )


def get_offset(conn: sqlite3.Connection) -> int:
    """Last-processed Telegram getUpdates offset (0 if none)."""
    row = conn.execute(
        "SELECT last_cursor FROM source_state WHERE source = 'telegram'"
    ).fetchone()
    return int(row["last_cursor"]) if row and row["last_cursor"] else 0


def set_offset(conn: sqlite3.Connection, offset: int) -> None:
    conn.execute(
        "INSERT INTO source_state (source, last_run, last_cursor) "
        "VALUES ('telegram', ?, ?) "
        "ON CONFLICT(source) DO UPDATE SET last_run=excluded.last_run, "
        "last_cursor=excluded.last_cursor",
        (now_utc_iso(), str(offset)),
    )


def insert_paper(
    conn: sqlite3.Connection,
    paper: Paper,
    embedding: np.ndarray,
    score: float,
    nearest: str,
) -> None:
    """Insert a scored paper (no-op if the uid already exists). Caller commits."""
    conn.execute(
        "INSERT OR IGNORE INTO papers "
        "(uid, source, title, abstract, authors, url, categories, published, "
        " first_seen, embedding, score, nearest) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            paper.uid,
            paper.source,
            paper.title,
            paper.abstract,
            json.dumps(paper.authors),
            paper.url,
            json.dumps(paper.categories),
            paper.published.isoformat() if paper.published else None,
            now_utc_iso(),
            embedding_to_blob(embedding),
            score,
            nearest,
        ),
    )
