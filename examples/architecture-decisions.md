# ADR-001: Hybrid Retrieval on PostgreSQL with pgvector and Full-Text Search

- **Status:** Accepted
- **Owner:** Core Platform Engineering
- **Context:** Local Developer Knowledge Base & MCP Server

## Context and Problem Statement

Developers accumulate engineering context across Markdown design docs, architecture decision records (ADRs), incident runbooks, and source code repositories. Querying this corpus from AI coding assistants (Cursor, Claude Code, Claude Desktop) via the Model Context Protocol (MCP) requires a retrieval engine that handles both:

1. **Conceptual queries** (e.g., *"How do we handle database backpressure during ingestion?"*)
2. **Exact identifier queries** (e.g., *"Where is `TokenBucketRateLimiter` configured or `ERR_CONN_RESET` handled?"*)

Pure vector search struggles with exact symbol names, error codes, and CLI flags, while pure lexical search misses semantically equivalent phrasing.

## Decision

We standardize on a single **PostgreSQL 16 + `pgvector`** instance storing document chunks with both dense vector embeddings and precomputed `tsvector` lexical tokens:

- **Dense Semantic Index:** 384-dimensional embeddings (`all-MiniLM-L6-v2`) indexed with an HNSW graph (`vector_cosine_ops`, `m = 16`, `ef_construction = 64`).
- **Lexical Index:** Generated `tsvector` column (`to_tsvector('english', text)`) indexed with a GIN inverted index.
- **Rank Fusion:** Top candidates from dense cosine similarity and `ts_rank_cd` lexical search are merged in Python using **Reciprocal Rank Fusion (RRF)** with constant $k = 60$:
  $$\text{RRF}(d) = \sum_{r \in \{\text{dense}, \text{lex}\}} \frac{1}{60 + \text{rank}_r(d)}$$
- **Two-Stage Reranking:** The top 20 fused candidates are scored jointly with the query using a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before returning the top $k$ chunks.

## Consequences

### Positive
- **Single operational dependency:** No separate Elasticsearch or dedicated vector database process to manage alongside Postgres.
- **High recall on code and prose:** GIN index catches exact function names and config keys; HNSW captures high-level design concepts.
- **Fully offline execution:** Embedding and cross-encoder reranking run locally on CPU/Metal without external API dependencies.

### Trade-offs
- Cross-encoder reranking adds ~30–80 ms of CPU latency per query over 20 candidate chunks.
- Initial ingestion requires computing embeddings locally, though batching keeps throughput around 150–300 chunks/sec on Apple Silicon.
