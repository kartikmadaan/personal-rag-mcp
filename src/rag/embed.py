import os
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))


def embed_batch(texts: list[str]) -> list[list[float]]:
    embeddings = _model.encode(texts, batch_size=100, show_progress_bar=False)
    return [embedding.tolist() for embedding in embeddings]
