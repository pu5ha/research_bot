"""M1 acceptance test: related strings embed closer than unrelated ones."""

from __future__ import annotations

import numpy as np

from src.embed import embed


def test_embeddings_are_unit_norm_and_semantically_ordered() -> None:
    texts = [
        "A transformer language model for reasoning and agents.",
        "Large language models improve at multi-step reasoning tasks.",
        "The migratory patterns of Atlantic humpback whales.",
    ]
    vecs = embed(texts)

    # Unit-norm contract: cosine == dot product downstream depends on this.
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    related = float(vecs[0] @ vecs[1])
    unrelated = float(vecs[0] @ vecs[2])
    assert related > unrelated
