from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from sentence_transformers import CrossEncoder

from .retrieve import Hit


@runtime_checkable
class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> object: ...


_reranker: Reranker | None = None


def get_reranker(model_name: str | None = None) -> Reranker:
    global _reranker
    if _reranker is None:
        resolved = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        _reranker = CrossEncoder(resolved)
    return _reranker


def set_reranker(reranker: Reranker | None) -> None:
    global _reranker
    _reranker = reranker


def rerank(
    query: str,
    hits: list[Hit],
    k: int = 5,
    reranker: Reranker | None = None,
) -> list[Hit]:
    if not hits:
        return []
    model = reranker if reranker is not None else get_reranker()
    pairs = [(query, h.text) for h in hits]
    scores = model.predict(pairs)
    for h, s in zip(hits, scores):  # type: ignore[arg-type]
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

