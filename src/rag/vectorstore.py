import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
   source TEXT PRIMARY KEY,
   content_sha256 TEXT NOT NULL,
   mtime DOUBLE PRECISION NOT NULL DEFAULT 0,
   chunk_count INT NOT NULL DEFAULT 0,
   updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
   id BIGSERIAL PRIMARY KEY,
   source TEXT NOT NULL,
   chunk_id INT NOT NULL,
   text TEXT NOT NULL,
   metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
   embedding vector(384) NOT NULL,
   tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE UNIQUE INDEX IF NOT EXISTS chunks_source_chunk_id_idx
   ON chunks (source, chunk_id);
CREATE INDEX IF NOT EXISTS chunks_source_idx
   ON chunks (source);
CREATE INDEX IF NOT EXISTS chunks_hnsw
   ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv
   ON chunks USING gin (tsv);
"""

ChunkRow = tuple[str, int, str, Any, Sequence[float]]


@dataclass
class DocumentRecord:
    source: str
    content_sha256: str
    mtime: float
    chunk_count: int
    updated_at: datetime | None = None


@dataclass
class SyncResult:
    inserted: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    chunks_upserted: int = 0


def _get_database_url() -> str:
    raw = os.getenv("DATABASE_URL")
    if raw:
        return raw.replace("+psycopg", "")
    user = os.getenv("POSTGRES_USER", "rag")
    password = os.getenv("POSTGRES_PASSWORD", "rag")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db = os.getenv("POSTGRES_DB", "rag")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@contextmanager
def get_conn() -> Iterator[psycopg.Connection[Any]]:
    conn_str = _get_database_url()
    conn = psycopg.connect(conn_str)

    # We can safely register here because init_db() guarantees
    # the extension exists before any other function uses get_conn()
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    conn_str = _get_database_url()

    # 1. Open a raw connection WITHOUT registering the vector type yet
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # 2. Run the non-destructive DDL to create the extension, tables, and indexes safely
            cur.execute(DDL)
        conn.commit()
        print("Database initialized successfully with pgvector.")


def _parse_metadata_dict(raw_meta: Any) -> dict[str, Any]:
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str) and raw_meta.strip():
        try:
            parsed = json.loads(raw_meta)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _serialize_metadata(raw_meta: Any) -> str:
    if isinstance(raw_meta, str):
        serialized = raw_meta
    else:
        serialized = json.dumps(raw_meta or {})
    return serialized.replace("\\u0000", "").replace("\x00", "")


def _compute_rows_digest(source_rows: list[ChunkRow]) -> tuple[str, float]:
    """Derive (content_sha256, mtime) from chunk rows or their metadata."""
    meta_sha256: str | None = None
    meta_mtime: float | None = None

    for _, _, _, raw_meta, _ in source_rows:
        meta = _parse_metadata_dict(raw_meta)
        if meta_sha256 is None and isinstance(meta.get("content_sha256"), str) and meta["content_sha256"]:
            meta_sha256 = meta["content_sha256"]
        if meta_mtime is None and isinstance(meta.get("mtime"), (int, float)):
            meta_mtime = float(meta["mtime"])
        if meta_sha256 is not None and meta_mtime is not None:
            break

    if meta_sha256 is None or meta_mtime is None:
        source_path = Path(source_rows[0][0])
        if source_path.is_file():
            try:
                if meta_mtime is None:
                    meta_mtime = float(source_path.stat().st_mtime)
                if meta_sha256 is None:
                    h = hashlib.sha256()
                    with source_path.open("rb") as f:
                        for block in iter(lambda: f.read(65536), b""):
                            h.update(block)
                    meta_sha256 = h.hexdigest()
            except OSError:
                pass

    if meta_sha256 is None:
        h = hashlib.sha256()
        for _, chunk_id, text, _, _ in sorted(source_rows, key=lambda r: r[1]):
            h.update(f"{chunk_id}:{text}".encode("utf-8", errors="ignore"))
        meta_sha256 = h.hexdigest()

    return meta_sha256, (meta_mtime if meta_mtime is not None else 0.0)


def get_indexed_documents(
    conn: psycopg.Connection[Any] | None = None,
    source_prefix: str | None = None,
) -> dict[str, DocumentRecord]:
    """Return currently indexed document metadata records keyed by `source`."""
    def _query(c: psycopg.Connection[Any]) -> dict[str, DocumentRecord]:
        with c.cursor() as cur:
            if source_prefix:
                cur.execute(
                    "SELECT source, content_sha256, mtime, chunk_count, updated_at "
                    "FROM documents WHERE source = %s OR source LIKE %s ORDER BY source",
                    (source_prefix, f"{source_prefix.rstrip('/')}/%"),
                )
            else:
                cur.execute(
                    "SELECT source, content_sha256, mtime, chunk_count, updated_at "
                    "FROM documents ORDER BY source"
                )
            records = cur.fetchall()
        return {
            r[0]: DocumentRecord(
                source=r[0],
                content_sha256=r[1],
                mtime=float(r[2]),
                chunk_count=int(r[3]),
                updated_at=r[4],
            )
            for r in records
        }

    if conn is not None:
        return _query(conn)
    with get_conn() as c:
        return _query(c)


def delete_sources(
    sources: Sequence[str],
    conn: psycopg.Connection[Any] | None = None,
) -> int:
    """Atomically delete all chunks and document metadata rows for the given sources."""
    source_list = list(sources)
    if not source_list:
        return 0

    def _delete(c: psycopg.Connection[Any]) -> int:
        with c.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = ANY(%s)", (source_list,))
            cur.execute("DELETE FROM documents WHERE source = ANY(%s)", (source_list,))
        return len(source_list)

    if conn is not None:
        return _delete(conn)

    with get_conn() as c:
        with c.transaction():
            count = _delete(c)
        return count


def upsert_chunks(
    rows: Sequence[ChunkRow],
    *,
    document_meta: dict[str, tuple[str, float]] | None = None,
    deleted_sources: Sequence[str] | None = None,
    force: bool = False,
) -> SyncResult:
    """Incrementally and idempotently sync document chunks inside a DB transaction.

    - Skips sources whose `content_sha256` matches the existing `documents` entry (unless `force=True`).
    - Deletes stale/orphaned chunks for modified sources (when chunk counts shrink or content changes)
      and for explicitly deleted sources before inserting updated chunks.
    - Updates the `documents` metadata table (`source`, `content_sha256`, `mtime`, `chunk_count`, `updated_at`).
    """
    grouped: dict[str, list[ChunkRow]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    result = SyncResult()
    to_delete = list(deleted_sources or [])

    if not grouped and not to_delete:
        return result

    with get_conn() as conn:
        with conn.transaction():
            existing = get_indexed_documents(conn)

            if to_delete:
                delete_sources(to_delete, conn=conn)
                result.deleted.extend(to_delete)

            with conn.cursor() as cur:
                for source, source_rows in grouped.items():
                    if document_meta and source in document_meta:
                        sha256, mtime = document_meta[source]
                    else:
                        sha256, mtime = _compute_rows_digest(source_rows)

                    prev = existing.get(source)
                    if (
                        not force
                        and prev is not None
                        and prev.content_sha256 == sha256
                        and prev.chunk_count == len(source_rows)
                    ):
                        result.skipped.append(source)
                        continue

                    # Remove prior chunks for this source so shrinking chunk counts never leave orphans
                    cur.execute("DELETE FROM chunks WHERE source = %s", (source,))

                    clean_rows = [
                        (
                            src,
                            int(cid),
                            txt.replace("\x00", ""),
                            _serialize_metadata(meta),
                            emb,
                        )
                        for src, cid, txt, meta, emb in source_rows
                    ]
                    cur.executemany(
                        "INSERT INTO chunks (source, chunk_id, text, metadata, embedding) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        clean_rows,
                    )
                    cur.execute(
                        "INSERT INTO documents (source, content_sha256, mtime, chunk_count, updated_at) "
                        "VALUES (%s, %s, %s, %s, NOW()) "
                        "ON CONFLICT (source) DO UPDATE SET "
                        "content_sha256 = EXCLUDED.content_sha256, "
                        "mtime = EXCLUDED.mtime, "
                        "chunk_count = EXCLUDED.chunk_count, "
                        "updated_at = NOW()",
                        (source, sha256, mtime, len(clean_rows)),
                    )

                    result.chunks_upserted += len(clean_rows)
                    if prev is None:
                        result.inserted.append(source)
                    else:
                        result.updated.append(source)

    return result
