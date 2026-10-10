"""Text-quality and meaning-preservation measurements.

- Semantic similarity: cosine similarity of sentence embeddings (SBERT). If the
  embedding model is unavailable, a lexical fallback (token-set Jaccard) is reported
  under a *different* metric name so the two are never mixed.
- Perplexity under an oracle LM that is different from the generator (Kirchenbauer
  et al. use a larger oracle). Lower is more fluent, but repetitive text also scores low,
  so a repetition rate is reported alongside.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from backend.core.config import get_settings
from backend.core.errors import ProviderUnavailable
from backend.core.logging import get_logger

log = get_logger(__name__)
_lock = threading.Lock()
_embedder: Any = None


def _get_embedder(model_name: str | None = None):
    global _embedder
    with _lock:
        if _embedder is None:
            s = get_settings()
            try:
                from sentence_transformers import SentenceTransformer

                _embedder = SentenceTransformer(
                    model_name or s.default_embedding_model, device="cpu", local_files_only=s.hf_offline
                )
            except Exception as exc:  # noqa: BLE001
                raise ProviderUnavailable(f"Embedding model unavailable: {exc}") from exc
        return _embedder


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def semantic_similarity(pairs: Sequence[tuple[str, str]], *, allow_fallback: bool = True) -> dict[str, Any]:
    """Returns {"metric": "sbert_cosine"|"jaccard", "values": [...]}."""
    if not pairs:
        return {"metric": None, "values": []}
    try:
        model = _get_embedder()
        a = model.encode([p[0] for p in pairs], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        b = model.encode([p[1] for p in pairs], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        return {
            "metric": "sbert_cosine",
            "model": get_settings().default_embedding_model,
            "values": [float(x) for x in (a * b).sum(axis=1)],
        }
    except ProviderUnavailable:
        if not allow_fallback:
            raise
        log.warning("embedding model unavailable; using lexical Jaccard similarity")
        return {"metric": "jaccard", "values": [jaccard_similarity(x, y) for x, y in pairs]}


def perplexity(texts: Sequence[str], model_name: str | None = None, max_tokens: int = 512) -> list[float | None]:
    from backend.providers.hf_local import get_lm

    lm = get_lm(model_name or get_settings().default_oracle_model)
    out: list[float | None] = []
    for t in texts:
        st = lm.token_stats(t, max_tokens=max_tokens)
        if st["n_tokens"] < 2:
            out.append(None)
            continue
        out.append(float(math.exp(-float(np.mean(st["logp"])))))
    return out


def repetition_rate(text: str, n: int = 3) -> float | None:
    """Fraction of word n-grams that are repeats (higher = more degenerate text)."""
    w = text.split()
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return None
    return 1 - len(set(grams)) / len(grams)
