"""Retrieval and reranking tests (migrated from src/rag/test_retrieval.py)."""

from unittest.mock import patch

from rag.retrieve import Hit, hybrid_search, rrf_fuse


def test_rrf_fuse_combines_dense_and_lexical_rankings() -> None:
    dense_hits = [
        Hit(chunk_id=1, source="examples/rate_limiter.py", text="TokenBucketRateLimiter", score=0.92),
        Hit(chunk_id=2, source="examples/architecture-decisions.md", text="pgvector HNSW", score=0.85),
    ]
    lexical_hits = [
        Hit(chunk_id=2, source="examples/architecture-decisions.md", text="pgvector HNSW", score=0.40),
        Hit(chunk_id=3, source="examples/runbook-local-dev.md", text="docker compose up", score=0.30),
    ]

    fused = rrf_fuse([dense_hits, lexical_hits], k_rrf=60)

    # Chunk 2 appears in both rankings (rank 1 in lexical, rank 2 in dense), so it wins RRF
    assert [h.chunk_id for h in fused] == [2, 1, 3]
    expected_chunk_2_score = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
    assert abs(fused[0].score - expected_chunk_2_score) < 1e-9


def test_hybrid_search_merges_and_truncates_to_k_out() -> None:
    dense_hits = [
        Hit(chunk_id=10, source="a.md", text="alpha", score=0.9),
        Hit(chunk_id=20, source="b.md", text="beta", score=0.8),
    ]
    bm25_hits = [
        Hit(chunk_id=20, source="b.md", text="beta", score=0.5),
        Hit(chunk_id=30, source="c.md", text="gamma", score=0.4),
    ]

    with (
        patch("rag.retrieve.dense_search", return_value=dense_hits) as mock_dense,
        patch("rag.retrieve.bm25_search", return_value=bm25_hits) as mock_bm25,
    ):
        results = hybrid_search("beta query", k_each=5, k_out=2)

    mock_dense.assert_called_once_with("beta query", 5)
    mock_bm25.assert_called_once_with("beta query", 5)
    assert len(results) == 2
    assert results[0].chunk_id == 20


def run_live_smoke(query: str = "How does TokenBucketRateLimiter handle burst capacity?") -> None:
    """Manual CLI smoke test against a running local pgvector instance."""
    from dotenv import load_dotenv

    load_dotenv()

    from rag.reranker import rerank
    from rag.retrieve import bm25_search, dense_search

    print(f"--- Query: {query} ---\n")

    dense_hits = dense_search(query, k=1)
    bm25_hits = bm25_search(query, k=1)

    if dense_hits:
        print(f"[Dense Top Hit] ID: {dense_hits[0].chunk_id} | Snippet: {dense_hits[0].text[:60]}...")
    if bm25_hits:
        print(f"[BM25 Top Hit]  ID: {bm25_hits[0].chunk_id} | Snippet: {bm25_hits[0].text[:60]}...\n")

    hybrid_hits = hybrid_search(query, k_each=20, k_out=10)
    print("--- Stage 1: Top 5 Hybrid (RRF) Results ---")
    for i, h in enumerate(hybrid_hits[:5]):
        print(f"Rank {i + 1} | Chunk ID: {h.chunk_id} | RRF Score: {h.score:.4f} | Snippet: {h.text[:50]}...")

    reranked_hits = rerank(query, list(hybrid_hits), k=5)
    print("\n--- Stage 2: Top 5 Reranked (Cross-Encoder) Results ---")
    for i, h in enumerate(reranked_hits):
        print(f"Rank {i + 1} | Chunk ID: {h.chunk_id} | CE Score: {h.score:.4f} | Snippet: {h.text[:50]}...")


if __name__ == "__main__":
    run_live_smoke()
