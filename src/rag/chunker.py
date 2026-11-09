from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken

from .loaders import RawDoc

_enc: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.encoding_for_model("text-embedding-3-small")
    return _enc


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict


def _split(text: str, target_tokens: int, overlap: int) -> list[str]:
    if overlap >= target_tokens:
        raise ValueError("overlap must be smaller than target_tokens")
    enc = _get_encoding()
    tokens = enc.encode(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        out.append(enc.decode(tokens[i : i + target_tokens]))
        i += target_tokens - overlap
    return out


def chunk_docs(docs: Iterable[RawDoc]) -> Iterable[Chunk]:
    for d in docs:
        is_code = d.metadata.get("filetype") in {
            ".py",
            ".go",
            ".cpp",
            ".h",
            ".java",
            ".ts",
            ".js",
        }
        target, overlap = (800, 120) if is_code else (350, 60)
        for i, piece in enumerate(_split(d.text, target, overlap)):
            yield Chunk(text=piece, source=d.source, chunk_id=i, metadata=d.metadata)
