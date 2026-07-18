"""M1 storage-layer safety net: schema creation + embedding blob roundtrip."""

from __future__ import annotations

import numpy as np

from src.db import blob_to_embedding, connect, embedding_to_blob, init_schema

EXPECTED_TABLES = {"papers", "sent", "votes", "taste", "source_state"}


def test_init_schema_creates_all_tables(tmp_path) -> None:
    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r["name"] for r in rows}
        assert EXPECTED_TABLES.issubset(names)
    finally:
        conn.close()


def test_embedding_blob_roundtrip_is_exact() -> None:
    vec = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    restored = blob_to_embedding(embedding_to_blob(vec))
    assert np.array_equal(vec, restored)
    assert restored.dtype == np.float32
