from sentence_transformers import CrossEncoder


_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")




def rerank(query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
   pairs = [(query, h.text) for h in hits]
   scores = _reranker.predict(pairs)
   for h, s in zip(hits, scores):
       h.score = float(s)
   return sorted(hits, key=lambda h: h.score, reverse=True)[:k]