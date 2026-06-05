import os
from contextlib import contextmanager
import psycopg
from pgvector.psycopg import register_vector

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS chunks CASCADE;
CREATE TABLE IF NOT EXISTS chunks (
   id BIGSERIAL PRIMARY KEY,
   source TEXT NOT NULL,
   chunk_id INT NOT NULL,
   text TEXT NOT NULL,
   metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
   embedding vector(384) NOT NULL,
   tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_hnsw
   ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv
   ON chunks USING gin (tsv);
"""

@contextmanager
def get_conn():
   # Strip the driver string if present
   conn_str = os.environ["DATABASE_URL"].replace("+psycopg", "")
   conn = psycopg.connect(conn_str)
   
   # We can safely register here because init_db() guarantees 
   # the extension exists before any other function uses get_conn()
   register_vector(conn)
   try:
       yield conn
   finally:
       conn.close()

def init_db() -> None:
   conn_str = os.environ["DATABASE_URL"].replace("+psycopg", "")
   
   # 1. Open a raw connection WITHOUT registering the vector type yet
   with psycopg.connect(conn_str) as conn:
       with conn.cursor() as cur:
           # 2. Run the DDL to create the extension and tables safely
           cur.execute(DDL)
       conn.commit()
       print("Database initialized successfully with pgvector.")

def upsert_chunks(rows: list[tuple[str, int, str, dict, list[float]]]) -> None:
   with get_conn() as conn:
       with conn.cursor() as cur:
           cur.executemany(
               "INSERT INTO chunks (source, chunk_id, text, metadata, embedding) "
               "VALUES (%s,%s,%s,%s,%s)",
               rows,
           )
       conn.commit()
