"""Optional PostgreSQL + pgvector integration tests.

Run with:
    pytest -m integration
"""

import json
from unittest.mock import patch

import psycopg
import pytest

from rag.retrieve import bm25_search, dense_search, hybrid_search
from rag.vectorstore import _get_database_url, init_db, upsert_chunks


def _db_available() -> bool:
    try:
        with psycopg.connect(_get_database_url(), connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.mark.integration
def test_pgvector_upsert_and_hybrid_retrieval_roundtrip() -> None:
    if not _db_available():
        pytest.skip("PostgreSQL + pgvector is not reachable for integration test")

    init_db()

    vec_a = [1.0] + [0.0] * 383
    vec_b = [0.0, 1.0] + [0.0] * 382

    rows = [
        (
            "examples/rate_limiter.py",
            0,
            "TokenBucketRateLimiter controls burst capacity and refill rate",
            json.dumps({"filetype": ".py"}),
            vec_a,
        ),
        (
            "examples/architecture-decisions.md",
            0,
            "PostgreSQL pgvector HNSW index combined with tsvector GIN search",
            json.dumps({"filetype": ".md"}),
            vec_b,
        ),
    ]
    upsert_chunks(rows)

    with patch("rag.retrieve.embed_batch", return_value=[vec_a]):
        d_hits = dense_search("rate limiter", k=2)
        assert len(d_hits) == 2
        assert d_hits[0].source == "examples/rate_limiter.py"

        l_hits = bm25_search("TokenBucketRateLimiter burst", k=2)
        assert len(l_hits) >= 1
        assert l_hits[0].source == "examples/rate_limiter.py"

        h_hits = hybrid_search("TokenBucketRateLimiter burst", k_each=5, k_out=2)
        assert len(h_hits) >= 1
        assert h_hits[0].source == "examples/rate_limiter.py"
