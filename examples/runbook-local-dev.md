# Local Development & Operations Runbook

This note documents operational procedures, performance tuning, and troubleshooting steps for running the `personal-rag-mcp` stack locally.

## Database Operations

The local PostgreSQL + `pgvector` container runs on host port **5433** (mapped to container port `5432`) with database name `rag` and user `rag`.

### Inspecting Index Health

Connect via `psql` inside the container to verify chunk counts and index sizes:

```bash
docker compose exec db psql -U rag -d rag -c "
SELECT
    count(*) AS total_chunks,
    count(DISTINCT source) AS indexed_files,
    pg_size_pretty(pg_total_relation_size('chunks')) AS total_size
FROM chunks;
"
```

### Verifying HNSW & GIN Index Usage

When debugging retrieval latency, ensure PostgreSQL uses `chunks_hnsw` for vector queries and `chunks_tsv` for lexical queries. Note that on very small tables (< 100 rows), the Postgres query planner may legitimately choose a sequential scan (`Seq Scan`) because reading a few pages in memory is faster than traversing an index.

## Ollama Local Inference Tuning

For low-latency grounded generation during interactive coding sessions:

1. **Keep model warm in memory:** Set `OLLAMA_KEEP_ALIVE=30m` so `llama3.2:3b` weights stay resident in unified memory/VRAM between MCP tool invocations.
2. **Enable Flash Attention:** Start Ollama with `OLLAMA_FLASH_ATTENTION=1 ollama serve` to reduce prompt processing latency on multi-chunk contexts.
3. **Context window sizing:** Keep retrieved chunk count between `k=4` and `k=6` to avoid prompt cache eviction and minimize time-to-first-token (TTFT).

## Troubleshooting Common Errors

- **`ERR_PG_CONN_REFUSED` (Port 5433):** Verify Docker Desktop is running and start the container via `docker compose up -d`.
- **HTTP 400 on `/query`:** Incoming queries are screened by `src/rag/guards.py` for prompt-injection signatures and length bounds (`MAX_QUERY_LEN = 2000`). Legitimate developer queries containing raw system prompt tags (e.g., `<|im_start|>`) will trigger guard rejection.
