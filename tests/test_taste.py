"""M2: seed parsing, seed rebuild, and vote folding into the taste profile."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.db import connect, embedding_to_blob, init_schema, now_utc_iso
from src.taste import (
    fold_votes,
    load_taste_vectors,
    parse_seed_line,
    rebuild_seeds,
    taste_counts,
)


def test_parse_seed_line() -> None:
    assert parse_seed_line("# comment") is None
    assert parse_seed_line("   ") is None
    label, text = parse_seed_line("SAM 2 — segments video objects across frames.")
    assert label == "SAM 2"
    assert text == "SAM 2. segments video objects across frames."
    # No delimiter → the whole line is both label and embedded text.
    assert parse_seed_line("Just a title") == ("Just a title", "Just a title")


def test_rebuild_seeds_stores_unit_norm_positives(tmp_path) -> None:
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(
        "# header\n"
        "DeepSeek-R1 — reasoning via RL alone.\n"
        "AlphaGenome — predicts regulatory effects of DNA variants.\n",
        encoding="utf-8",
    )
    cfg = load_config()  # real config; model is CPU + cached
    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        n = rebuild_seeds(cfg, conn, seeds_path=seeds)
        assert n == 2
        assert taste_counts(conn).get("seed") == 2

        pos, labels, neg = load_taste_vectors(conn)
        assert pos.shape[0] == 2
        assert neg.shape[0] == 0
        assert set(labels) == {"DeepSeek-R1", "AlphaGenome"}
        assert np.allclose(np.linalg.norm(pos, axis=1), 1.0, atol=1e-4)

        # Idempotent: re-running does not duplicate seed rows.
        rebuild_seeds(cfg, conn, seeds_path=seeds)
        assert taste_counts(conn).get("seed") == 2
    finally:
        conn.close()


def test_fold_votes_reflects_thumbs(tmp_path) -> None:
    conn = connect(tmp_path / "bot.db")
    try:
        init_schema(conn)
        vec = np.ones(4, dtype=np.float32) / 2.0  # unit-norm dummy embedding
        conn.execute(
            "INSERT INTO papers (uid, title, embedding, first_seen) VALUES (?, ?, ?, ?)",
            ("arxiv:1", "A liked paper", embedding_to_blob(vec), now_utc_iso()),
        )
        conn.execute(
            "INSERT INTO votes (uid, vote, voted_at) VALUES (?, ?, ?)",
            ("arxiv:1", 1, now_utc_iso()),
        )
        conn.commit()

        pos, neg = fold_votes(conn)
        assert (pos, neg) == (1, 0)
        assert taste_counts(conn).get("pos") == 1

        # Flipping the vote to 👎 moves it to negatives (no duplicate row).
        conn.execute("UPDATE votes SET vote=-1 WHERE uid='arxiv:1'")
        conn.commit()
        pos, neg = fold_votes(conn)
        assert (pos, neg) == (0, 1)
        counts = taste_counts(conn)
        assert counts.get("neg") == 1
        assert counts.get("pos", 0) == 0
    finally:
        conn.close()
