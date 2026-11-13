"""Optional PostgreSQL + pgvector integration tests.

Run with:
    pytest -m integration
"""

import json
from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest

from rag.ingest import sync_directory
from rag.retrieve import bm25_search, dense_search, hybrid_search
from rag.vectorstore import (
    _get_database_url,
    delete_sources,
    get_indexed_documents,
    init_db,
    upsert_chunks,
)


def _db_available() -> bool:
    try:
        with psycopg.connect(_get_database_url(), connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _reset_tables() -> None:
    with psycopg.connect(_get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks, documents RESTART IDENTITY")
        conn.commit()


@pytest.mark.integration
def test_pgvector_upsert_and_hybrid_retrieval_roundtrip() -> None:
    if not _db_available():
        pytest.skip("PostgreSQL + pgvector is not reachable for integration test")

    init_db()
    _reset_tables()

    vec_a = [1.0] + [0.0] * 383
    vec_b = [0.0, 1.0] + [0.0] * 382

    rows = [
        (
            "examples/rate_limiter.py",
            0,
            "TokenBucketRateLimiter controls burst capacity and refill rate",
            json.dumps({"filetype": ".py", "content_sha256": "sha_a1", "mtime": 100.0}),
            vec_a,
        ),
        (
            "examples/architecture-decisions.md",
            0,
            "PostgreSQL pgvector HNSW index combined with tsvector GIN search",
            json.dumps({"filetype": ".md", "content_sha256": "sha_b1", "mtime": 200.0}),
            vec_b,
        ),
    ]
    res1 = upsert_chunks(rows)
    assert set(res1.inserted) == {"examples/rate_limiter.py", "examples/architecture-decisions.md"}
    assert res1.chunks_upserted == 2

    # Calling init_db() again must be non-destructive (preserves existing chunks & documents)
    init_db()
    docs = get_indexed_documents()
    assert set(docs.keys()) == {"examples/rate_limiter.py", "examples/architecture-decisions.md"}

    # Re-upserting identical rows is idempotent and skips unchanged sources
    res2 = upsert_chunks(rows)
    assert set(res2.skipped) == {"examples/rate_limiter.py", "examples/architecture-decisions.md"}
    assert res2.chunks_upserted == 0

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

    # Deleting a source removes both its chunks and document metadata record
    delete_sources(["examples/rate_limiter.py"])
    assert set(get_indexed_documents().keys()) == {"examples/architecture-decisions.md"}


@pytest.mark.integration
def test_incremental_sync_directory_lifecycle(tmp_path: Path) -> None:
    if not _db_available():
        pytest.skip("PostgreSQL + pgvector is not reachable for integration test")

    init_db()
    _reset_tables()

    doc_a = tmp_path / "alpha.txt"
    doc_b = tmp_path / "beta.txt"
    doc_a.write_text("Alpha first version content.", encoding="utf-8")
    doc_b.write_text("Beta initial content.", encoding="utf-8")

    class Dummy384Embedder:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return [[1.0] + [0.0] * 383 for _ in texts]

    embedder = Dummy384Embedder()

    # 1. Initial sync inserts both files
    r1 = sync_directory(tmp_path, embedder=embedder)
    assert len(r1.inserted) == 2
    assert r1.skipped == []

    # 2. Second sync with no changes skips both files
    r2 = sync_directory(tmp_path, embedder=embedder)
    assert len(r2.skipped) == 2
    assert r2.inserted == []
    assert r2.updated == []

    # 3. Modify alpha.txt and delete beta.txt -> updates alpha, prunes beta
    doc_a.write_text("Alpha updated second version content.", encoding="utf-8")
    doc_b.unlink()
    r3 = sync_directory(tmp_path, embedder=embedder)
    assert r3.updated == [str(doc_a.resolve())]
    assert r3.deleted == [str(doc_b.resolve())]
    assert list(get_indexed_documents().keys()) == [str(doc_a.resolve())]
