"""Unit tests for embedder-aware token sizing, Markdown heading chunking, and AST code chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rag import chunker
from rag.chunker import (
    ChunkConfig,
    WordPieceAlignedEncoding,
    chunk_docs,
    get_chunk_config,
    register_embedder_chunk_config,
)
from rag.loaders import RawDoc, load_file


class FakeEncoding:
    """Deterministic whitespace token encoding for fast, offline unit tests."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class PassThroughWhitespaceEncoding:
    """Deterministic encoding that preserves original whitespace words via integer indices."""

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._to_id: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for tok in text.split():
            if tok not in self._to_id:
                self._to_id[tok] = len(self._vocab)
                self._vocab.append(tok)
            ids.append(self._to_id[tok])
        return ids

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._vocab[i] for i in tokens)


def test_split_respects_target_tokens_and_overlap() -> None:
    fake_enc = PassThroughWhitespaceEncoding()
    text = " ".join(str(i) for i in range(10))  # 10 tokens: 0..9

    with patch("rag.chunker._get_encoding", return_value=fake_enc):
        pieces = chunker._split(text, target_tokens=4, overlap=1)

    # Stride is 4 - 1 = 3: windows at 0..3, 3..6, 6..9, 9..9
    assert pieces == [
        "0 1 2 3",
        "3 4 5 6",
        "6 7 8 9",
        "9",
    ]


def test_split_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than target_tokens"):
        chunker._split("some text", target_tokens=10, overlap=10)


def test_embedder_chunk_configs_fix_minilm_256_token_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default all-MiniLM-L6-v2 config respects the 256-token WordPiece limit and is configurable."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CHUNK_PROSE_TARGET_TOKENS", raising=False)
    monkeypatch.delenv("CHUNK_CODE_TARGET_TOKENS", raising=False)

    default_cfg = get_chunk_config()
    assert default_cfg.embedder_model == "all-MiniLM-L6-v2"
    assert default_cfg.max_seq_tokens == 256
    assert default_cfg.prose_target_tokens <= 256
    assert default_cfg.code_target_tokens <= 256
    assert default_cfg.tokenizer_family == "wordpiece"

    # OpenAI embedder uses larger 8191-token context window with 350/800 targets
    openai_cfg = get_chunk_config("text-embedding-3-small")
    assert openai_cfg.max_seq_tokens == 8191
    assert (openai_cfg.prose_target_tokens, openai_cfg.prose_overlap_tokens) == (350, 60)
    assert (openai_cfg.code_target_tokens, openai_cfg.code_overlap_tokens) == (800, 120)
    assert openai_cfg.tokenizer_family == "cl100k_base"

    # BGE embedder uses 512-token context window
    bge_cfg = get_chunk_config("BAAI/bge-small-en-v1.5")
    assert bge_cfg.max_seq_tokens == 512
    assert bge_cfg.prose_target_tokens == 350
    assert bge_cfg.code_target_tokens == 420

    # Environment variable overrides
    monkeypatch.setenv("CHUNK_PROSE_TARGET_TOKENS", "150")
    monkeypatch.setenv("CHUNK_PROSE_OVERLAP_TOKENS", "25")
    overridden = get_chunk_config("all-MiniLM-L6-v2")
    assert (overridden.prose_target_tokens, overridden.prose_overlap_tokens) == (150, 25)

    # Custom embedder profile registration
    custom_cfg = ChunkConfig(
        embedder_model="custom/code-embed-1k",
        max_seq_tokens=1024,
        prose_target_tokens=400,
        prose_overlap_tokens=50,
        code_target_tokens=512,
        code_overlap_tokens=64,
    )
    register_embedder_chunk_config("custom/code-embed-1k", custom_cfg)
    assert get_chunk_config("code-embed-1k").max_seq_tokens == 1024

    # Validation rejects target_tokens > max_seq_tokens
    with pytest.raises(ValueError, match="exceeds embedder max_seq_tokens"):
        ChunkConfig(
            embedder_model="all-MiniLM-L6-v2",
            max_seq_tokens=256,
            prose_target_tokens=350,
            prose_overlap_tokens=60,
        )


def test_wordpiece_aligned_encoding_lossless_and_splits_punctuation() -> None:
    wp_enc = WordPieceAlignedEncoding()
    sample = "def compute_rate_limit(user_id: str, burst: int = 10) -> tuple[bool, float]:\n    return True, 0.0\n"
    tokens = wp_enc.encode(sample)
    assert wp_enc.decode(tokens) == sample
    # WordPiece splits underscores and punctuation individually, yielding more tokens than raw BPE
    assert len(tokens) >= 25


def test_chunk_docs_uses_code_vs_prose_window_sizes() -> None:
    fake_enc = PassThroughWhitespaceEncoding()
    # 400 tokens: with text-embedding-3-small profile, fits in 1 code chunk (target=800),
    # splits into 2 prose chunks (target=350, overlap=60)
    tokens_400 = " ".join(str(i) for i in range(400))
    docs = [
        RawDoc(text=tokens_400, source="src/app.py", metadata={"filetype": ".py"}),
        RawDoc(text=tokens_400, source="docs/notes.md", metadata={"filetype": ".md"}),
    ]

    with patch("rag.chunker._get_encoding", return_value=fake_enc):
        chunks = list(chunk_docs(docs, embedder_model="text-embedding-3-small"))

    code_chunks = [c for c in chunks if c.source == "src/app.py"]
    prose_chunks = [c for c in chunks if c.source == "docs/notes.md"]

    assert len(code_chunks) == 1
    assert code_chunks[0].chunk_id == 0
    assert code_chunks[0].metadata["filetype"] == ".py"

    assert len(prose_chunks) == 2
    assert [c.chunk_id for c in prose_chunks] == [0, 1]
    assert prose_chunks[1].text.startswith("290 291")

    # With default all-MiniLM-L6-v2 (256 max tokens: code=220, prose=200), both split safely under 256 tokens
    with patch("rag.chunker._get_encoding", return_value=fake_enc):
        minilm_chunks = list(chunk_docs(docs, embedder_model="all-MiniLM-L6-v2"))
    for c in minilm_chunks:
        assert len(fake_enc.encode(c.text)) <= 220


def test_markdown_heading_hierarchy_chunking_preserves_paths_and_ignores_code_fences() -> None:
    md_text = """# System Architecture

High-level overview of the platform.

## Storage Layer

```python
# This comment inside a fenced code block is NOT a heading
def connect_db():
    pass
```

### PostgreSQL & pgvector

Details on HNSW vector indexing.

### Full-Text Search

Details on GIN tsvector indexing.

## Ingestion Pipeline

Details on incremental file syncing.
"""
    doc = RawDoc(text=md_text, source="docs/arch.md", metadata={"filetype": ".md"})
    chunks = list(chunk_docs([doc]))

    assert len(chunks) == 5

    assert chunks[0].metadata["heading_path"] == ["System Architecture"]
    assert chunks[0].metadata["section_heading"] == "System Architecture"
    assert chunks[0].metadata["start_line"] == 1
    assert chunks[0].metadata["end_line"] == 4

    # Code fence `# This comment...` stayed inside Storage Layer instead of starting a new heading
    assert chunks[1].metadata["heading_path"] == ["System Architecture", "Storage Layer"]
    assert "# This comment inside a fenced code block" in chunks[1].text
    assert chunks[1].metadata["start_line"] == 5
    assert chunks[1].metadata["end_line"] == 12

    assert chunks[2].metadata["heading_path"] == [
        "System Architecture",
        "Storage Layer",
        "PostgreSQL & pgvector",
    ]
    assert chunks[2].metadata["section_heading"] == (
        "System Architecture > Storage Layer > PostgreSQL & pgvector"
    )
    assert chunks[2].metadata["start_line"] == 13
    assert chunks[2].metadata["end_line"] == 16

    # Sibling H3 replaces previous H3 on the stack
    assert chunks[3].metadata["heading_path"] == [
        "System Architecture",
        "Storage Layer",
        "Full-Text Search",
    ]
    assert chunks[3].metadata["start_line"] == 17
    assert chunks[3].metadata["end_line"] == 20

    # H2 pops both H3 and previous H2
    assert chunks[4].metadata["heading_path"] == ["System Architecture", "Ingestion Pipeline"]
    assert chunks[4].metadata["start_line"] == 21
    assert chunks[4].metadata["end_line"] == 23


def test_markdown_loader_and_chunker_end_to_end_on_example_file() -> None:
    """Verify load_file() preserves raw Markdown headings so chunk_docs populates heading_path."""
    raw_docs = load_file(Path("examples/architecture-decisions.md"))
    chunks = list(chunk_docs(raw_docs))
    assert len(chunks) >= 4

    paths = [c.metadata["heading_path"] for c in chunks]
    assert [
        "ADR-001: Hybrid Retrieval on PostgreSQL with pgvector and Full-Text Search",
        "Consequences",
        "Positive",
    ] in paths
    assert [
        "ADR-001: Hybrid Retrieval on PostgreSQL with pgvector and Full-Text Search",
        "Consequences",
        "Trade-offs",
    ] in paths


def test_python_ast_code_chunking_extracts_symbols_and_line_ranges() -> None:
    raw_docs = load_file(Path("examples/rate_limiter.py"))
    chunks = list(chunk_docs(raw_docs))

    by_symbol = {c.metadata["symbol_name"]: c for c in chunks}
    assert "RateLimitPolicy" in by_symbol
    assert by_symbol["RateLimitPolicy"].metadata["start_line"] == 10
    assert by_symbol["RateLimitPolicy"].metadata["end_line"] == 20

    # TokenBucketRateLimiter exceeds 220 WordPiece tokens, so it decomposes into class header + methods
    assert "TokenBucketRateLimiter" in by_symbol
    assert "TokenBucketRateLimiter.__init__" in by_symbol
    assert "TokenBucketRateLimiter._refill" in by_symbol
    assert "TokenBucketRateLimiter.acquire" in by_symbol

    acquire_chunk = by_symbol["TokenBucketRateLimiter.acquire"]
    assert acquire_chunk.metadata["start_line"] == 45
    assert acquire_chunk.metadata["end_line"] == 64
    assert "def acquire(" in acquire_chunk.text


def test_treesitter_code_chunking_typescript_go_cpp_and_java() -> None:
    ts_code = """export interface User {
  id: string;
}

export class AuthService {
  login(u: User): boolean {
    return u.id.length > 0;
  }
}

export const verifyToken = (token: string): boolean => {
  return token.startsWith("jwt_");
};
"""
    go_code = """package main

type Server struct {
    Addr string
}

func (s *Server) Start() error {
    return nil
}

func NewServer(addr string) *Server {
    return &Server{Addr: addr}
}
"""
    cpp_code = """class Engine {
public:
    void run() {}
};

int compute_checksum(int value) {
    return value * 31;
}
"""
    java_code = """public class OrderService {
    public void createOrder(String id) {
        System.out.println(id);
    }
}
"""
    docs = [
        RawDoc(text=ts_code, source="src/auth.ts", metadata={"filetype": ".ts"}),
        RawDoc(text=go_code, source="cmd/server.go", metadata={"filetype": ".go"}),
        RawDoc(text=cpp_code, source="src/engine.cpp", metadata={"filetype": ".cpp"}),
        RawDoc(text=java_code, source="src/OrderService.java", metadata={"filetype": ".java"}),
    ]

    chunks = list(chunk_docs(docs))

    ts_chunks = [c for c in chunks if c.source == "src/auth.ts"]
    assert [(c.metadata["symbol_name"], c.metadata["start_line"], c.metadata["end_line"]) for c in ts_chunks] == [
        ("User", 1, 3),
        ("AuthService", 5, 9),
        ("verifyToken", 11, 13),
    ]

    go_chunks = [c for c in chunks if c.source == "cmd/server.go"]
    assert [(c.metadata["symbol_name"], c.metadata["start_line"], c.metadata["end_line"]) for c in go_chunks] == [
        ("<module>", 1, 2),
        ("Server", 3, 5),
        ("Server.Start", 7, 9),
        ("NewServer", 11, 13),
    ]

    cpp_chunks = [c for c in chunks if c.source == "src/engine.cpp"]
    assert [c.metadata["symbol_name"] for c in cpp_chunks if c.metadata["symbol_name"] != "<module>"] == [
        "Engine",
        "compute_checksum",
    ]
    compute_chunk = next(c for c in cpp_chunks if c.metadata["symbol_name"] == "compute_checksum")
    assert (compute_chunk.metadata["start_line"], compute_chunk.metadata["end_line"]) == (6, 8)

    java_chunks = [c for c in chunks if c.source == "src/OrderService.java"]
    assert len(java_chunks) == 1
    assert java_chunks[0].metadata["symbol_name"] == "OrderService"
    assert (java_chunks[0].metadata["start_line"], java_chunks[0].metadata["end_line"]) == (1, 5)
