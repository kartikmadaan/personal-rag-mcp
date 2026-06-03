import os
from contextlib import contextmanager
import psycopg
from pgvector.psycopg import register_vector


DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
   id BIGSERIAL PRIMARY KEY,
   source TEXT NOT NULL,
   chunk_id INT NOT NULL,
   text TEXT NOT NULL,
   metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
   embedding vector(1536) NOT NULL,
   tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_hnsw
   ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv
   ON chunks USING gin (tsv);
"""




@contextmanager
def get_conn():
   conn = psycopg.connect(os.environ["DATABASE_URL"].replace("+psycopg", ""))
   register_vector(conn)
   try:
       yield conn
   finally:
       conn.close()




def init_db() -> None:
   with get_conn() as conn:
       with conn.cursor() as cur:
           cur.execute(DDL)
       conn.commit()




def upsert_chunks(rows: list[tuple[str, int, str, dict, list[float]]]) -> None:
   with get_conn() as conn:
       with conn.cursor() as cur:
           cur.executemany(
               "INSERT INTO chunks (source, chunk_id, text, metadata, embedding) "
               "VALUES (%s,%s,%s,%s,%s)",
               rows,
           )
       conn.commit()