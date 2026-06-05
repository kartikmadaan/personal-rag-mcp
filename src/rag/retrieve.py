from collections import defaultdict
from dataclasses import dataclass
from .vectorstore import get_conn
from .embed import embed_batch




@dataclass
class Hit:
   chunk_id: int
   source: str
   text: str
   score: float




def dense_search(query: str, k: int = 20) -> list[Hit]:
   [q_emb] = embed_batch([query])
   with get_conn() as conn:
       with conn.cursor() as cur:
           cur.execute(
               "SELECT id, source, text, 1 - (embedding <=> %s::vector) AS score "
               "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
               (q_emb, q_emb, k),
           )
           rows = cur.fetchall()
   return [Hit(r[0], r[1], r[2], float(r[3])) for r in rows]




def bm25_search(query: str, k: int = 20) -> list[Hit]:
   with get_conn() as conn:
       with conn.cursor() as cur:
           cur.execute(
               "SELECT id, source, text, ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score "
               "FROM chunks WHERE tsv @@ plainto_tsquery('english', %s) "
               "ORDER BY score DESC LIMIT %s",
               (query, query, k),
           )
           rows = cur.fetchall()
   return [Hit(r[0], r[1], r[2], float(r[3])) for r in rows]




def rrf_fuse(rankings: list[list[Hit]], k_rrf: int = 60) -> list[Hit]:
   """Reciprocal Rank Fusion: score(d) = sum_i 1 / (k_rrf + rank_i(d))."""
   scores: dict[int, float] = defaultdict(float)
   keep: dict[int, Hit] = {}
   for hits in rankings:
       for rank, h in enumerate(hits):
           scores[h.chunk_id] += 1.0 / (k_rrf + rank + 1)
           keep[h.chunk_id] = h
   fused = sorted(keep.values(), key=lambda h: scores[h.chunk_id], reverse=True)
   for h in fused:
       h.score = scores[h.chunk_id]
   return fused




def hybrid_search(query: str, k_each: int = 20, k_out: int = 10) -> list[Hit]:
   dense = dense_search(query, k_each)
   bm25 = bm25_search(query, k_each)
   return rrf_fuse([dense, bm25])[:k_out]