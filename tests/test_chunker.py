"""Unit tests for token chunking boundaries and overlap logic."""

from unittest.mock import patch

import pytest

from rag import chunker
from rag.loaders import RawDoc


class FakeEncoding:
    """Deterministic whitespace token encoding for fast, offline unit tests."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [int(tok) for tok in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


def test_split_respects_target_tokens_and_overlap() -> None:
    fake_enc = FakeEncoding()
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


def test_chunk_docs_uses_code_vs_prose_window_sizes() -> None:
    fake_enc = FakeEncoding()
    # 400 tokens: fits in 1 code chunk (target=800), splits into 2 prose chunks (target=350, overlap=60)
    tokens_400 = " ".join(str(i) for i in range(400))
    docs = [
        RawDoc(text=tokens_400, source="src/app.py", metadata={"filetype": ".py"}),
        RawDoc(text=tokens_400, source="docs/notes.md", metadata={"filetype": ".md"}),
    ]

    with patch("rag.chunker._get_encoding", return_value=fake_enc):
        chunks = list(chunker.chunk_docs(docs))

    code_chunks = [c for c in chunks if c.source == "src/app.py"]
    prose_chunks = [c for c in chunks if c.source == "docs/notes.md"]

    assert len(code_chunks) == 1
    assert code_chunks[0].chunk_id == 0
    assert code_chunks[0].metadata["filetype"] == ".py"

    assert len(prose_chunks) == 2
    assert [c.chunk_id for c in prose_chunks] == [0, 1]
    # Second prose chunk starts at token 350 - 60 = 290
    assert prose_chunks[1].text.startswith("290 291")
