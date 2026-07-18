"""Taste profile: positives (seed anchors + 👍'd papers) and negatives (👎'd papers).

Stored in the ``taste`` table as L2-normalized float32 blobs. ``kind`` is
``'seed'`` | ``'pos'`` | ``'neg'``. Positives = seed ∪ pos; negatives = neg.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np

from .config import ROOT, Config
from .db import blob_to_embedding, embedding_to_blob
from .embed import embed

log = logging.getLogger("research-bot")

SEED_SEP = " — "  # em dash separates the label (title) from the gist
DEFAULT_SEEDS = ROOT / "seeds" / "ground_truth.txt"


def _seed_id(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
    return f"seed:{digest}"


def parse_seed_line(line: str) -> tuple[str, str] | None:
    """Parse one seed line into ``(label, text)``.

    ``label`` is the title (shown in pings); ``text`` is what gets embedded
    (title + gist). Blank lines and ``#`` comments return ``None``.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if SEED_SEP in stripped:
        label, gist = stripped.split(SEED_SEP, 1)
        label = label.strip()
        return label, f"{label}. {gist.strip()}"
    return stripped, stripped


def load_seed_lines(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as fh:
        parsed = [parse_seed_line(ln) for ln in fh]
    return [pair for pair in parsed if pair is not None]


def rebuild_seeds(cfg: Config, conn: sqlite3.Connection, seeds_path: Path | str | None = None) -> int:
    """Embed the seed file and replace all ``kind='seed'`` rows. Returns the count."""
    path = Path(seeds_path) if seeds_path else DEFAULT_SEEDS
    if not path.exists():
        log.warning("no seed file at %s; skipping seed rebuild", path)
        return 0

    pairs = load_seed_lines(path)
    if not pairs:
        log.warning("seed file %s has no usable lines", path)
        return 0

    labels = [lbl for lbl, _ in pairs]
    texts = [txt for _, txt in pairs]
    vecs = embed(texts, model=cfg.model)

    conn.execute("DELETE FROM taste WHERE kind='seed'")
    conn.executemany(
        "INSERT OR REPLACE INTO taste (id, kind, label, embedding) VALUES (?, 'seed', ?, ?)",
        [(_seed_id(lbl), lbl, embedding_to_blob(v)) for lbl, v in zip(labels, vecs)],
    )
    conn.commit()
    log.info("loaded %d seed anchors from %s", len(pairs), path)
    return len(pairs)


def fold_votes(conn: sqlite3.Connection) -> tuple[int, int]:
    """Reflect every vote into the taste table using the paper's stored embedding.

    👍 → ``kind='pos'``, 👎 → ``kind='neg'``. Idempotent: re-voting updates the row.
    Returns ``(positives_from_votes, negatives_from_votes)``. Papers without a
    stored embedding (not yet fetched/scored) are skipped with a warning.
    """
    rows = conn.execute(
        "SELECT v.uid AS uid, v.vote AS vote, p.title AS title, p.embedding AS embedding "
        "FROM votes v JOIN papers p ON p.uid = v.uid"
    ).fetchall()

    pos = neg = 0
    for r in rows:
        if r["embedding"] is None:
            log.warning("vote for %s has no stored embedding; skipping", r["uid"])
            continue
        kind = "pos" if r["vote"] > 0 else "neg"
        conn.execute(
            "INSERT OR REPLACE INTO taste (id, kind, label, embedding) VALUES (?, ?, ?, ?)",
            (f"vote:{r['uid']}", kind, r["title"] or r["uid"], r["embedding"]),
        )
        if kind == "pos":
            pos += 1
        else:
            neg += 1
    conn.commit()
    return pos, neg


def taste_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT kind, COUNT(*) AS c FROM taste GROUP BY kind").fetchall()
    return {r["kind"]: r["c"] for r in rows}


def refresh_taste(cfg: Config, conn: sqlite3.Connection) -> dict[str, int]:
    """Rebuild seeds from the file and fold in any votes. Returns per-kind counts."""
    rebuild_seeds(cfg, conn)
    fold_votes(conn)
    return taste_counts(conn)


def load_taste_vectors(
    conn: sqlite3.Connection,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load the profile for scoring: ``(positives, pos_labels, negatives)``.

    Positives = seed ∪ pos (labels aligned by row). Negatives = neg (unlabeled).
    Empty categories come back as ``(0, dim)`` arrays.
    """
    pos_labels: list[str] = []
    pos_vecs: list[np.ndarray] = []
    neg_vecs: list[np.ndarray] = []
    for r in conn.execute("SELECT kind, label, embedding FROM taste"):
        vec = blob_to_embedding(r["embedding"])
        if r["kind"] in ("seed", "pos"):
            pos_labels.append(r["label"])
            pos_vecs.append(vec)
        else:
            neg_vecs.append(vec)

    pos = np.vstack(pos_vecs) if pos_vecs else np.empty((0, 0), dtype=np.float32)
    neg = np.vstack(neg_vecs) if neg_vecs else np.empty((0, 0), dtype=np.float32)
    return pos, pos_labels, neg
