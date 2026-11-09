# personal-rag-mcp

A local-first **developer knowledge base and Model Context Protocol (MCP) server** backed by **PostgreSQL + pgvector**. Index your engineering notes, architecture decision records (ADRs), design docs, PDFs, and local source code, then query them directly from **Cursor**, **Claude Code**, **Claude Desktop**, or over an HTTP API.

Unlike thin vector-only wrappers, `personal-rag-mcp` uses a two-stage **hybrid retrieval** pipeline—combining dense semantic search (`pgvector` HNSW) with lexical full-text search (PostgreSQL `tsvector` / GIN), fused via **Reciprocal Rank Fusion (RRF)** and reranked locally with a cross-encoder—so both conceptual questions and exact symbol/identifier lookups resolve accurately without leaving your machine.

## Features

- **MCP Server (`stdio`)** — Exposes `search_knowledge_base` to MCP-compatible coding clients (Cursor, Claude Code, Claude Desktop).
- **Multi-Format Ingestion** — Loads Markdown (`.md`), plain text/reStructuredText (`.txt`, `.rst`), PDFs (`.pdf`), and source code (`.py`, `.ts`, `.js`, `.go`, `.cpp`, `.h`, `.java`).
- **Token-Aware Chunking** — Splits documents with configurable token windows and overlap via `tiktoken`.
- **Hybrid Retrieval (Dense + Lexical + RRF)** — Merges 384-dim `all-MiniLM-L6-v2` cosine similarity search (HNSW index) and PostgreSQL full-text search (GIN index) using Reciprocal Rank Fusion.
- **Cross-Encoder Reranking** — Re-scores top candidate chunks locally with `cross-encoder/ms-marco-MiniLM-L-6-v2` before context assembly.
- **Grounded Answers with Citations** — Generates answers with inline `[S1]`, `[S2]` source attributions via local **Ollama** models.
- **Prompt-Injection Guardrails** — Screens incoming queries and retrieved context against injection patterns and instruction hijacking.
- **FastAPI Service** — Exposes `POST /query` and `GET /healthz` for HTTP integrations and local scripting.

## Architecture

```
             ┌────────────────────────────────────────────┐
             │  Local Files (Markdown, ADRs, Code, PDFs)  │
             └─────────────────────┬──────────────────────┘
                                   │ scripts/ingest.py
                                   ▼
             ┌────────────────────────────────────────────┐
             │     loaders  →  chunker  →  embedder       │
             └─────────────────────┬──────────────────────┘
                                   │ upsert
                                   ▼
          ┌────────────────────────┴────────────────────────┐
          │                    PostgreSQL                   │
          ▼                                                 ▼
┌────────────────────┐                            ┌────────────────────┐
│ pgvector (HNSW)    │                            │ PostgreSQL FTS     │
│ cosine similarity  │                            │ (tsvector / GIN)   │
└─────────┬──────────┘                            └─────────┬──────────┘
          │                                                 │
          └────────────────────────┬────────────────────────┘
                                   │ Reciprocal Rank Fusion (RRF)
                                   ▼
                     ┌──────────────────────────┐
                     │  Cross-Encoder Reranker  │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │  Injection Guards + LLM  │
                     │     (Local Ollama)       │
                     └─────────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │  MCP Server (stdio)  │      │  FastAPI (/query)    │
        │ Cursor / Claude Code │      │  HTTP JSON Endpoint  │
        └──────────────────────┘      └──────────────────────┘
```

See [`analysis.md`](analysis.md) for a low-level walkthrough of the `pgvector` HNSW graph parameters, `tsvector` GIN index behavior, and two-stage reranking scores.

## Prerequisites

- **Python 3.11+**
- **Docker** (for PostgreSQL 16 + `pgvector`)
- **Ollama** running locally with a chat model pulled (default: `llama3.2:3b`)

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> personal-rag-mcp
cd personal-rag-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

*(Optional)* To install the RAGAS regression evaluation dependencies:

```bash
pip install -e ".[eval]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Runtime environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` | PostgreSQL connection string |
| `POSTGRES_USER` | `rag` | PostgreSQL user (used by `docker compose` and fallback DSN) |
| `POSTGRES_PASSWORD` | `rag` | PostgreSQL password |
| `POSTGRES_DB` | `rag` | PostgreSQL database name |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5433` | Host port mapped to container port `5432` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible local Ollama endpoint |
| `GENERATOR_MODEL` | `llama3.2:3b` | Local Ollama model used for grounded synthesis |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local SentenceTransformer embedding model (384-dim) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local CrossEncoder reranking model |

### 3. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

PostgreSQL listens on **`localhost:5433`** with credentials `rag` / `rag`.

### 4. Ingest sample docs or your own notes

A self-contained [`examples/`](examples/) directory is included with sample architecture decision records, operational runbooks, and Python source code:

```bash
python scripts/ingest.py examples/
```

To index your own notes, docs, or PDFs, drop them into `corpus/` (or pass any directory path):

```bash
python scripts/ingest.py /path/to/your/notes-or-repo
```

### 5. Start Ollama

```bash
ollama pull llama3.2:3b
ollama serve
```

### 6. Query the Knowledge Base

**Via Python:**

```bash
python -c "from rag.pipeline import query; import json; print(json.dumps(query('Why did we choose hybrid RRF retrieval with pgvector?'), indent=2))"
```

**Via HTTP API:**

```bash
uvicorn rag.api:app --reload --port 8088
```

```bash
curl -s -X POST http://localhost:8088/query \
  -H 'content-type: application/json' \
  -d '{"q": "How does TokenBucketRateLimiter handle burst capacity?", "k": 4}' | jq
```

## MCP Client Integration

`src/rag/mcp_server.py` runs an MCP server over `stdio` and exposes the `search_knowledge_base` tool.

### Cursor / Claude Desktop Configuration

A template configuration is provided in [`.cursor/mcp.json`](.cursor/mcp.json). Replace `/absolute/path/to/personal-rag-mcp` with the absolute path to your local clone (returned by `pwd`):

```json
{
  "mcpServers": {
    "personal-rag": {
      "command": "/absolute/path/to/personal-rag-mcp/.venv/bin/python",
      "args": ["-m", "rag.mcp_server"],
      "cwd": "/absolute/path/to/personal-rag-mcp",
      "env": {
        "DATABASE_URL": "postgresql://rag:rag@localhost:5433/rag",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "GENERATOR_MODEL": "llama3.2:3b",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

### Claude Code CLI

```bash
claude mcp add personal-rag -- "$(pwd)/.venv/bin/python" -m rag.mcp_server
```

## HTTP API Reference

### `POST /query`

Request body:

```json
{
  "q": "Where is TokenBucketRateLimiter implemented and how does acquire() work?",
  "k": 5
}
```

Response:

```json
{
  "answer": "TokenBucketRateLimiter is implemented in examples/rate_limiter.py [S1]...",
  "citations": [
    {
      "id": "S1",
      "source": "examples/rate_limiter.py",
      "text": "..."
    }
  ],
  "usage": {
    "prompt_tokens": 640,
    "completion_tokens": 142,
    "total_tokens": 782
  },
  "latency_ms": 980
}
```

Queries matching prompt-injection heuristics or exceeding length bounds return `HTTP 400`.

### `GET /healthz`

Returns `{"ok": true}`.

## Security & Regression Fixtures

- **Input & Context Guards (`src/rag/guards.py`)** — Enforces query length bounds, rejects prompt-injection patterns (instruction overrides, role-switching, fake system markers), and strips control sequences from retrieved context blocks.
- **Automated Tests (`tests/`)** — Run unit tests with `pytest`.
- **Regression Fixtures (`evals/`)** — Contains adversarial prompt-injection test cases (`evals/attacks.jsonl`) and end-to-end retrieval/QA regression fixtures (`evals/qa_set.jsonl`, runnable via `scripts/eval.py` when `.[eval]` is installed).

## Project Layout

```
personal-rag-mcp/
├── .cursor/
│   ├── mcp.json             # Cursor MCP server config template
│   └── rules/search.mdc     # Cursor rule for routing KB queries
├── corpus/                  # Local user documents directory (PDFs gitignored)
├── docker-compose.yml       # PostgreSQL 16 + pgvector on :5433
├── examples/                # Self-contained sample ADRs, runbooks & code
├── evals/                   # Security & retrieval regression fixtures
│   ├── attacks.jsonl        # Prompt-injection regression cases
│   └── qa_set.jsonl         # End-to-end QA benchmark fixtures
├── scripts/
│   ├── ingest.py            # Directory ingestion CLI entrypoint
│   └── eval.py              # Optional RAGAS regression scoring script
├── src/rag/
│   ├── api.py               # FastAPI HTTP service (/query, /healthz)
│   ├── chunker.py           # Token-aware document splitter
│   ├── embed.py             # Local SentenceTransformer embeddings
│   ├── generate.py          # Grounded synthesis with inline citations
│   ├── guards.py            # Prompt-injection & context sanitization guards
│   ├── loaders.py           # Multi-format file loaders (md, code, txt, pdf)
│   ├── mcp_server.py        # Model Context Protocol (MCP) stdio server
│   ├── pipeline.py          # End-to-end retrieval + rerank + generation
│   ├── reranker.py          # Cross-encoder candidate reranking
│   ├── retrieve.py          # Dense HNSW + GIN lexical search + RRF fusion
│   └── vectorstore.py       # pgvector schema, indexes, and batch upsert
├── tests/
│   └── test_retrieval.py    # Retrieval & RRF fusion test suite
├── analysis.md              # Retrieval & index inspection notes
└── pyproject.toml
```

## Roadmap

- [x] MCP server (`search_knowledge_base` stdio tool for Cursor, Claude Code, and Claude Desktop)
- [x] Hybrid retrieval (pgvector HNSW + PostgreSQL GIN full-text search + RRF fusion + Cross-Encoder reranking)
- [x] Regression harness & prompt-injection guardrails
- [ ] Comprehensive unit test suite, lazy model loading, and GitHub Actions CI
- [ ] Incremental content-addressed ingestion (`SHA-256` change detection) & live file watching
- [ ] Structure-aware Markdown & AST code chunking

## License

MIT
