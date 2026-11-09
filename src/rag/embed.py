from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from sentence_transformers import SentenceTransformer


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


_model: Embedder | None = None


def get_embedder(model_name: str | None = None) -> Embedder:
    global _model
    if _model is None:
        resolved = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(resolved)
    return _model


def set_embedder(embedder: Embedder | None) -> None:
    global _model
    _model = embedder


def embed_batch(
    texts: list[str],
    embedder: Embedder | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    model = embedder if embedder is not None else get_embedder()
    embeddings = model.encode(texts, batch_size=100, show_progress_bar=False)
    return [
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        for embedding in embeddings
    ]

