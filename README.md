# personal-rag-mcp

A self-hosted **RAG (Retrieval-Augmented Generation)** pipeline that ingests your documents, indexes them with **hybrid retrieval** (PostgreSQL full-text search + dense embeddings, fused via RRF), reranks candidates with a cross-encoder, generates grounded answers with citations, and exposes the pipeline over a **FastAPI** HTTP API.

The long-term goal is to wrap this as an **MCP (Model Context Protocol)** server so Cursor, Claude Desktop, and other MCP clients can query your knowledge base conversationally. The retrieval and generation stack is in place; MCP integration is planned next.

## Overview

This project is a systems-flavored RAG stack—not a thin wrapper around a single API call. It covers:

- **Ingestion** — load PDFs, Markdown, plain text, and source code from a corpus directory
- **Chunking** — token-aware splitting with overlap (tiktoken)
- **Embedding** — local `all-MiniLM-L6-v2` vectors (384-dim) stored in **pgvector**
- **Hybrid retrieval** — dense cosine search (HNSW index) + BM25-style keyword search (GIN/tsvector), merged with **Reciprocal Rank Fusion (RRF)**
- **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder
- **Generation** — grounded answers with inline citations via **Ollama** (OpenAI-compatible API)
- **Guardrails** — prompt-injection pattern detection on incoming queries
- **HTTP API** — `POST /query` and `GET /healthz`

A sample corpus of 10 ML/AI PDFs is included under `corpus/`. See [`analysis.md`](analysis.md) for a detailed walkthrough of the database schema, indexes, and retrieval behavior.

## Architecture

```
                    ┌──────────────────────────────┐
                    │  corpus/ (PDFs, md, code)    │
                    └──────────────┬───────────────┘
                                   │ ingest
                                   ▼
                    ┌──────────────────────────────┐
                    │  loaders → chunker → embed     │
                    └──────────────┬───────────────┘
                                   │ store
                                   ▼
         ┌─────────────────────────┴─────────────────────────┐
         │                                                   │
         ▼                                                   ▼
┌────────────────────┐                            ┌────────────────────┐
│ pgvector (HNSW)    │                            │ PostgreSQL FTS     │
│ cosine similarity  │                            │ (tsvector / GIN)   │
└─────────┬──────────┘                            └──────────┬─────────┘
          │                                                │
          └──────────────────┬─────────────────────────────┘
                             │ RRF fusion
                             ▼
                  ┌──────────────────────┐
                  │  cross-encoder rerank │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  prompt-injection    │
                  │  guard + Ollama LLM  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  FastAPI  /query     │
                  └──────────────────────┘
```

## Prerequisites

- Python 3.11+
- [Docker](https://docs.docker.com/get-docker/) (for Postgres + pgvector)
- [Ollama](https://ollama.com/) running locally with a chat model pulled (default: `llama3.2:3b`)

## Quick start

### 1. Clone and install

```bash
git clone <repo-url> personal-rag-mcp
cd personal-rag-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

At minimum, set `DATABASE_URL`. Other keys are optional depending on which backends you use:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string (required) |
| `GENERATOR_MODEL` | Ollama model name (default: `llama3.2:3b`) |
| `EMBEDDING_MODEL` | Reserved for OpenAI embeddings (currently unused; local model is default) |
| `OPENAI_API_KEY` | Optional; OpenAI path is commented out in `embed.py` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (optional) |

### 3. Start the database

```bash
docker compose up -d
```

Postgres listens on **localhost:5433** with credentials `rag` / `rag`.

### 4. Ingest the corpus

```bash
python scripts/ingest.py corpus/
```

This creates the `chunks` table (with HNSW + GIN indexes), chunks all supported files, embeds them locally, and upserts rows into pgvector.

### 5. Start Ollama (if not already running)

```bash
ollama pull llama3.2:3b
ollama serve
```

### 6. Run a query

**Python:**

```bash
python -c "from rag.pipeline import query; import json; print(json.dumps(query('what are deep neural networks?'), indent=2))"
```

**HTTP API:**

```bash
uvicorn rag.api:app --reload --port 8088
```

```bash
curl -s -X POST localhost:8088/query \
  -H 'content-type: application/json' \
  -d '{"q":"how does neural network work?","k":4}' | jq
```

**Retrieval-only smoke test:**

```bash
python -m rag.test_retrieval
```

## Evaluation

`scripts/eval.py` runs the pipeline against `evals/qa_set.jsonl` and scores the outputs with [RAGAS](https://docs.ragas.io/). The script first sends each question through `query()`, collecting the generated answer and the retrieved chunk texts. An evaluator LLM then grades those rows on three metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer actually supported by the retrieved context? |
| **Answer relevancy** | Does the answer address the question? |
| **Context precision** | Were the retrieved chunks useful for producing the reference answer? |

Use a local Ollama model for quick iteration, or Gemini if you have an API key and want more reliable structured scoring:

```bash
# local judge (llama3.2:3b via Ollama)
python scripts/eval.py --evaluator local

# Gemini judge — set GEMINI_API_KEY or GOOGLE_API_KEY first
python scripts/eval.py --evaluator gemini

# smoke test on two questions
python scripts/eval.py --evaluator gemini --limit 2
```

On Gemini free tier, keep `--limit` low for first runs. The script uses one concurrent RAGAS worker and a 300s per-job timeout to stay inside rate limits.

The QA set mixes corpus questions (Montúfar, Hornik, OpenVINO papers in `corpus/`) with repo-architecture questions (`chunker.py`, `retrieve.py`, etc.). Only the PDF corpus is ingested today, so architecture questions are expected to miss unless you add source code to the index.

## API

### `POST /query`

```json
{
  "q": "your question",
  "k": 5
}
```

**Response:**

```json
{
  "answer": "...",
  "citations": [{"id": "S1", "source": "corpus/example.pdf"}],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
  "latency_ms": 1234
}
```

Malicious or injection-style queries are rejected with HTTP 400.

### `GET /healthz`

Returns `{"ok": true}`.

## Project layout

```
personal-rag-mcp/
├── corpus/                  # Sample PDFs + .gitkeep
├── docker-compose.yml       # pgvector Postgres on :5433
├── evals/
│   ├── attacks.jsonl        # Prompt-injection test cases
│   └── qa_set.jsonl         # RAGAS evaluation questions
├── scripts/
│   ├── ingest.py            # Corpus ingestion entrypoint
│   └── eval.py              # RAGAS evaluation harness
├── src/rag/
│   ├── api.py               # FastAPI app
│   ├── chunker.py           # Token-aware chunking
│   ├── embed.py             # Local SentenceTransformer embeddings
│   ├── generate.py          # Ollama-backed answer generation
│   ├── guards.py            # Prompt-injection defenses
│   ├── loaders.py           # PDF / md / txt / code loaders
│   ├── mcp_server.py        # MCP stdio server
│   ├── pipeline.py          # End-to-end query()
│   ├── reranker.py          # Cross-encoder reranking
│   ├── retrieve.py          # Dense + BM25 + RRF hybrid search
│   ├── test_retrieval.py    # Retrieval pipeline smoke test
│   └── vectorstore.py       # pgvector schema + upsert
├── analysis.md              # Deep-dive on indexes and retrieval
└── pyproject.toml
```

## Security

Incoming queries pass through `guards.py`, which blocks common prompt-injection patterns (e.g. "ignore previous instructions", fake system tags) and caps query length. Test cases live in `evals/attacks.jsonl`.

Retrieved context is wrapped in `<context>` blocks with explicit system-prompt rules telling the LLM to treat injected instructions as data, not commands.

## Roadmap

- [ ] MCP server (`search_kb` tool for Cursor / Claude Desktop)
- [x] Eval harness (RAGAS faithfulness, answer relevancy, context precision)
- [ ] Optional OpenAI / Anthropic backends for embeddings and generation
- [ ] CI workflow and unit tests

## License

MIT (or your preferred license—add one if publishing publicly).
