import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. ALWAYS load environment variables before importing your RAG modules
load_dotenv()

# 2. Fix imports: get rerank from reranker.py, others from retrieve.py
from rag.retrieve import hybrid_search, dense_search, bm25_search
from rag.reranker import rerank

# Verify key is loaded
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY still not found in environment!")
    sys.exit(1)

query = "How does Milvus handle vector data management and scalar filtering?"
print(f"--- Query: {query} ---\n")

# Run independent searches
dense_hits = dense_search(query, k=1)
bm25_hits = bm25_search(query, k=1)

if dense_hits:
    print(f"[Dense Top Hit] ID: {dense_hits[0].chunk_id} | Snippet: {dense_hits[0].text[:60]}...")
if bm25_hits:
    print(f"[BM25 Top Hit]  ID: {bm25_hits[0].chunk_id} | Snippet: {bm25_hits[0].text[:60]}...\n")

# Stage 1: Hybrid Search (RRF)
hybrid_hits = hybrid_search(query, k_each=20, k_out=10)
print("--- Stage 1: Top 5 Hybrid (RRF) Results ---")
for i, h in enumerate(hybrid_hits[:5]):
    print(f"Rank {i+1} | Chunk ID: {h.chunk_id} | RRF Score: {h.score:.4f} | Snippet: {h.text[:50]}...")

# Stage 2: Reranker (Cross-Encoder)
# Pass a fresh list copy to preserve hybrid_hits scores if needed
reranked_hits = rerank(query, list(hybrid_hits), k=5)
print("\n--- Stage 2: Top 5 Reranked (Cross-Encoder) Results ---")
for i, h in enumerate(reranked_hits):
    print(f"Rank {i+1} | Chunk ID: {h.chunk_id} | CE Score: {h.score:.4f} | Snippet: {h.text[:50]}...")