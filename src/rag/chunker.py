from __future__ import annotations

import ast
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import tiktoken
import tree_sitter as ts
import tree_sitter_cpp as tscpp
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts

from .loaders import SUPPORTED_CODE_SUFFIXES, RawDoc


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers supporting lossless encode/decode."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


# WordPiece splits every punctuation character (including underscores) into its own token,
# whereas cl100k_base merges multi-character punctuation sequences like `):\n    ` or `_`.
# Pre-splitting on WordPiece boundaries before encoding into cl100k_base token IDs ensures
# token counts align with BERT/MiniLM WordPiece (preventing silent 256-token truncation)
# while keeping `.decode()` 100% lossless for code indentation and newlines.
_WORDPIECE_SPAN_RE = re.compile(r"[ \t]*[A-Za-z0-9]+|[ \t]*[^\sA-Za-z0-9]|\s+")


class WordPieceAlignedEncoding:
    """Lossless tokenizer whose token granularity matches BERT/MiniLM WordPiece."""

    def __init__(self, base_encoding: tiktoken.Encoding | None = None) -> None:
        self._base = base_encoding or tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        tokens: list[int] = []
        for match in _WORDPIECE_SPAN_RE.finditer(text):
            tokens.extend(self._base.encode(match.group(0)))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        return self._base.decode(tokens)


@dataclass(frozen=True)
class ChunkConfig:
    """Per-embedder token window and overlap configuration."""

    embedder_model: str = "all-MiniLM-L6-v2"
    max_seq_tokens: int = 256
    prose_target_tokens: int = 200
    prose_overlap_tokens: int = 30
    code_target_tokens: int = 220
    code_overlap_tokens: int = 30
    tokenizer_family: str = "wordpiece"  # "wordpiece" | "cl100k_base"

    def __post_init__(self) -> None:
        if self.max_seq_tokens <= 0:
            raise ValueError("max_seq_tokens must be positive")
        if self.prose_target_tokens <= 0 or self.code_target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.prose_target_tokens > self.max_seq_tokens:
            raise ValueError(
                f"prose_target_tokens ({self.prose_target_tokens}) exceeds "
                f"embedder max_seq_tokens ({self.max_seq_tokens})"
            )
        if self.code_target_tokens > self.max_seq_tokens:
            raise ValueError(
                f"code_target_tokens ({self.code_target_tokens}) exceeds "
                f"embedder max_seq_tokens ({self.max_seq_tokens})"
            )
        if not (0 <= self.prose_overlap_tokens < self.prose_target_tokens):
            raise ValueError("prose_overlap_tokens must be in [0, prose_target_tokens)")
        if not (0 <= self.code_overlap_tokens < self.code_target_tokens):
            raise ValueError("code_overlap_tokens must be in [0, code_target_tokens)")


DEFAULT_MINILM_CONFIG = ChunkConfig(
    embedder_model="all-MiniLM-L6-v2",
    max_seq_tokens=256,
    prose_target_tokens=200,
    prose_overlap_tokens=30,
    code_target_tokens=220,
    code_overlap_tokens=30,
    tokenizer_family="wordpiece",
)

EMBEDDER_CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "all-MiniLM-L6-v2": DEFAULT_MINILM_CONFIG,
    "all-MiniLM-L12-v2": replace(DEFAULT_MINILM_CONFIG, embedder_model="all-MiniLM-L12-v2"),
    "multi-qa-MiniLM-L6-cos-v1": replace(
        DEFAULT_MINILM_CONFIG, embedder_model="multi-qa-MiniLM-L6-cos-v1"
    ),
    "all-mpnet-base-v2": ChunkConfig(
        embedder_model="all-mpnet-base-v2",
        max_seq_tokens=384,
        prose_target_tokens=300,
        prose_overlap_tokens=45,
        code_target_tokens=340,
        code_overlap_tokens=50,
        tokenizer_family="wordpiece",
    ),
    "bge-small-en-v1.5": ChunkConfig(
        embedder_model="BAAI/bge-small-en-v1.5",
        max_seq_tokens=512,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=420,
        code_overlap_tokens=60,
        tokenizer_family="wordpiece",
    ),
    "bge-base-en-v1.5": ChunkConfig(
        embedder_model="BAAI/bge-base-en-v1.5",
        max_seq_tokens=512,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=420,
        code_overlap_tokens=60,
        tokenizer_family="wordpiece",
    ),
    "bge-large-en-v1.5": ChunkConfig(
        embedder_model="BAAI/bge-large-en-v1.5",
        max_seq_tokens=512,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=420,
        code_overlap_tokens=60,
        tokenizer_family="wordpiece",
    ),
    "e5-small-v2": ChunkConfig(
        embedder_model="intfloat/e5-small-v2",
        max_seq_tokens=512,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=420,
        code_overlap_tokens=60,
        tokenizer_family="wordpiece",
    ),
    "e5-base-v2": ChunkConfig(
        embedder_model="intfloat/e5-base-v2",
        max_seq_tokens=512,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=420,
        code_overlap_tokens=60,
        tokenizer_family="wordpiece",
    ),
    "nomic-embed-text": ChunkConfig(
        embedder_model="nomic-embed-text",
        max_seq_tokens=2048,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=600,
        code_overlap_tokens=90,
        tokenizer_family="wordpiece",
    ),
    "nomic-embed-text-v1.5": ChunkConfig(
        embedder_model="nomic-ai/nomic-embed-text-v1.5",
        max_seq_tokens=2048,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=600,
        code_overlap_tokens=90,
        tokenizer_family="wordpiece",
    ),
    "text-embedding-3-small": ChunkConfig(
        embedder_model="text-embedding-3-small",
        max_seq_tokens=8191,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=800,
        code_overlap_tokens=120,
        tokenizer_family="cl100k_base",
    ),
    "text-embedding-3-large": ChunkConfig(
        embedder_model="text-embedding-3-large",
        max_seq_tokens=8191,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=800,
        code_overlap_tokens=120,
        tokenizer_family="cl100k_base",
    ),
    "text-embedding-ada-002": ChunkConfig(
        embedder_model="text-embedding-ada-002",
        max_seq_tokens=8191,
        prose_target_tokens=350,
        prose_overlap_tokens=60,
        code_target_tokens=800,
        code_overlap_tokens=120,
        tokenizer_family="cl100k_base",
    ),
}


def register_embedder_chunk_config(model_name: str, config: ChunkConfig) -> None:
    """Register or override a chunking profile for a specific embedding model."""
    EMBEDDER_CHUNK_CONFIGS[model_name] = config
    short_name = model_name.rsplit("/", 1)[-1]
    EMBEDDER_CHUNK_CONFIGS[short_name] = config


def get_chunk_config(embedder_model: str | None = None) -> ChunkConfig:
    """Resolve the ChunkConfig for `embedder_model` (or `EMBEDDING_MODEL` env var) with env overrides."""
    raw_model = embedder_model or os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
    resolved_model = raw_model.strip()
    short_name = resolved_model.rsplit("/", 1)[-1]

    base = (
        EMBEDDER_CHUNK_CONFIGS.get(resolved_model)
        or EMBEDDER_CHUNK_CONFIGS.get(short_name)
        or replace(DEFAULT_MINILM_CONFIG, embedder_model=resolved_model)
    )

    max_seq = int(os.getenv("CHUNK_MAX_SEQ_TOKENS", str(base.max_seq_tokens)))
    prose_target = int(os.getenv("CHUNK_PROSE_TARGET_TOKENS", str(base.prose_target_tokens)))
    prose_overlap = int(os.getenv("CHUNK_PROSE_OVERLAP_TOKENS", str(base.prose_overlap_tokens)))
    code_target = int(os.getenv("CHUNK_CODE_TARGET_TOKENS", str(base.code_target_tokens)))
    code_overlap = int(os.getenv("CHUNK_CODE_OVERLAP_TOKENS", str(base.code_overlap_tokens)))

    return ChunkConfig(
        embedder_model=resolved_model,
        max_seq_tokens=max_seq,
        prose_target_tokens=prose_target,
        prose_overlap_tokens=prose_overlap,
        code_target_tokens=code_target,
        code_overlap_tokens=code_overlap,
        tokenizer_family=base.tokenizer_family,
    )


_cl100k_enc: tiktoken.Encoding | None = None
_wordpiece_enc: WordPieceAlignedEncoding | None = None


def _get_encoding(embedder_model: str | None = None) -> Tokenizer:
    """Return the tokenizer corresponding to the active embedder's tokenizer family."""
    global _cl100k_enc, _wordpiece_enc
    cfg = get_chunk_config(embedder_model)
    if _cl100k_enc is None:
        _cl100k_enc = tiktoken.get_encoding("cl100k_base")
    if cfg.tokenizer_family == "cl100k_base":
        return _cl100k_enc
    if _wordpiece_enc is None:
        _wordpiece_enc = WordPieceAlignedEncoding(_cl100k_enc)
    return _wordpiece_enc


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict[str, Any]


def _resolve_encoding(embedder_model: str | None = None) -> Tokenizer:
    """Call `_get_encoding` cleanly whether or not it has been mocked with a zero-arg callable."""
    if embedder_model is None:
        return _get_encoding()
    try:
        return _get_encoding(embedder_model)
    except TypeError:
        return _get_encoding()


def _split(
    text: str,
    target_tokens: int,
    overlap: int,
    *,
    enc: Tokenizer | None = None,
) -> list[str]:
    if overlap < 0 or overlap >= target_tokens:
        raise ValueError("overlap must be smaller than target_tokens")
    tokenizer = enc if enc is not None else _get_encoding()
    tokens = tokenizer.encode(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        out.append(tokenizer.decode(tokens[i : i + target_tokens]))
        i += target_tokens - overlap
    return out


def _split_lines_with_ranges(
    lines: list[str],
    base_start_line: int,
    target_tokens: int,
    overlap: int,
    enc: Tokenizer,
) -> list[tuple[str, int, int]]:
    """Split a block of lines into (chunk_text, start_line, end_line) windows within `target_tokens`."""
    if overlap < 0 or overlap >= target_tokens:
        raise ValueError("overlap must be smaller than target_tokens")

    block_text = "\n".join(lines)
    if not block_text.strip():
        return []

    total_tokens = enc.encode(block_text)
    if len(total_tokens) <= target_tokens:
        return [(block_text, base_start_line, base_start_line + len(lines) - 1)]

    # Single-line block (or synthetic space-separated tokens on 1 line): split by tokens directly
    if len(lines) == 1:
        pieces = _split(block_text, target_tokens, overlap, enc=enc)
        return [(p, base_start_line, base_start_line) for p in pieces if p.strip()]

    # Multi-line block: accumulate lines into token windows, falling back to token split for single huge lines
    line_token_counts = [max(1, len(enc.encode(line + "\n"))) for line in lines]
    results: list[tuple[str, int, int]] = []
    n = len(lines)
    start_idx = 0

    while start_idx < n:
        cur_tokens = 0
        end_idx = start_idx
        while end_idx < n and (cur_tokens + line_token_counts[end_idx] <= target_tokens or end_idx == start_idx):
            cur_tokens += line_token_counts[end_idx]
            end_idx += 1

        # If a single line itself exceeds target_tokens, split that line via token windowing
        if end_idx == start_idx + 1 and len(enc.encode(lines[start_idx])) > target_tokens:
            line_No = base_start_line + start_idx
            for piece in _split(lines[start_idx], target_tokens, overlap, enc=enc):
                if piece.strip():
                    results.append((piece, line_No, line_No))
            start_idx += 1
            continue

        window_text = "\n".join(lines[start_idx:end_idx])
        # Verify exact token length does not exceed target_tokens; trim trailing line if needed
        while end_idx - start_idx > 1 and len(enc.encode(window_text)) > target_tokens:
            end_idx -= 1
            window_text = "\n".join(lines[start_idx:end_idx])

        if window_text.strip():
            results.append(
                (
                    window_text,
                    base_start_line + start_idx,
                    base_start_line + end_idx - 1,
                )
            )

        if end_idx >= n:
            break

        # Step backwards from end_idx to preserve `overlap` tokens of context
        overlap_tokens = 0
        next_start = end_idx
        while next_start > start_idx + 1 and overlap_tokens + line_token_counts[next_start - 1] <= overlap:
            next_start -= 1
            overlap_tokens += line_token_counts[next_start]

        start_idx = next_start if next_start > start_idx else end_idx

    return results


# ==============================================================================
# Markdown Structure-Aware Heading-Hierarchy Chunking
# ==============================================================================

_ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_H1_RE = re.compile(r"^\s{0,3}=+\s*$")
_SETEXT_H2_RE = re.compile(r"^\s{0,3}-{3,}\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass
class _MarkdownSection:
    heading_path: list[str]
    lines: list[str]
    start_line: int
    end_line: int
    has_body: bool = False


def _parse_markdown_sections(text: str) -> list[_MarkdownSection]:
    """Parse Markdown text into sections annotated with their root-to-leaf heading hierarchy."""
    lines = text.splitlines()
    if not lines:
        return []

    sections: list[_MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    cur_lines: list[str] = []
    cur_start_line = 1
    cur_has_body = False
    in_fence = False
    fence_marker = ""

    def _current_path() -> list[str]:
        return [title for _, title in heading_stack]

    def _flush_section(end_line_num: int) -> None:
        nonlocal cur_lines, cur_has_body
        if not cur_lines:
            return
        if any(line.strip() for line in cur_lines):
            sections.append(
                _MarkdownSection(
                    heading_path=_current_path(),
                    lines=list(cur_lines),
                    start_line=cur_start_line,
                    end_line=end_line_num,
                    has_body=cur_has_body,
                )
            )
        cur_lines = []
        cur_has_body = False

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        line_num = i + 1

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0]
            elif marker[0] == fence_marker:
                in_fence = False
                fence_marker = ""
            cur_lines.append(line)
            if line.strip():
                cur_has_body = True
            i += 1
            continue

        if not in_fence:
            atx_match = _ATX_HEADING_RE.match(line)
            if atx_match:
                _flush_section(line_num - 1)
                level = len(atx_match.group(1))
                title = atx_match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                cur_start_line = line_num
                cur_lines = [line]
                cur_has_body = False
                i += 1
                continue

            # Check Setext heading (non-empty single paragraph line followed by === or ---)
            if (
                i + 1 < n
                and line.strip()
                and not line.lstrip().startswith(("-", "*", "+", ">", "|"))
                and (_SETEXT_H1_RE.match(lines[i + 1]) or _SETEXT_H2_RE.match(lines[i + 1]))
            ):
                _flush_section(line_num - 1)
                level = 1 if _SETEXT_H1_RE.match(lines[i + 1]) else 2
                title = line.strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                cur_start_line = line_num
                cur_lines = [line, lines[i + 1]]
                cur_has_body = False
                i += 2
                continue

        if not cur_lines:
            cur_start_line = line_num
        cur_lines.append(line)
        if line.strip():
            cur_has_body = True
        i += 1

    _flush_section(n)

    # Fold empty parent headings (headings immediately followed by a sub-heading with no body lines)
    # into the following child section so parent headings aren't emitted as standalone 1-line stubs.
    merged: list[_MarkdownSection] = []
    pending_prefix_lines: list[str] = []
    pending_start_line: int | None = None

    for sec in sections:
        if not sec.has_body and sec.heading_path:
            if pending_start_line is None:
                pending_start_line = sec.start_line
            pending_prefix_lines.extend(sec.lines)
        else:
            if pending_prefix_lines:
                sec.lines = pending_prefix_lines + sec.lines
                assert pending_start_line is not None
                sec.start_line = pending_start_line
                pending_prefix_lines = []
                pending_start_line = None
            merged.append(sec)

    if pending_prefix_lines and sections:
        last = sections[-1]
        merged.append(
            _MarkdownSection(
                heading_path=last.heading_path,
                lines=pending_prefix_lines,
                start_line=pending_start_line or last.start_line,
                end_line=last.end_line,
                has_body=True,
            )
        )

    return merged


def _chunk_markdown_doc(
    doc: RawDoc,
    config: ChunkConfig,
    enc: Tokenizer,
) -> list[Chunk]:
    sections = _parse_markdown_sections(doc.text)
    if not sections:
        return []

    # If the Markdown file has no headings at all, split directly while recording empty heading_path
    if len(sections) == 1 and not sections[0].heading_path:
        sec = sections[0]
        windows = _split_lines_with_ranges(
            sec.lines,
            sec.start_line,
            config.prose_target_tokens,
            config.prose_overlap_tokens,
            enc,
        )
        return [
            Chunk(
                text=piece,
                source=doc.source,
                chunk_id=idx,
                metadata={
                    **doc.metadata,
                    "heading_path": [],
                    "section_heading": "",
                    "start_line": s_line,
                    "end_line": e_line,
                },
            )
            for idx, (piece, s_line, e_line) in enumerate(windows)
        ]

    chunks: list[Chunk] = []
    chunk_idx = 0
    for sec in sections:
        windows = _split_lines_with_ranges(
            sec.lines,
            sec.start_line,
            config.prose_target_tokens,
            config.prose_overlap_tokens,
            enc,
        )
        section_heading = " > ".join(sec.heading_path)
        for piece, s_line, e_line in windows:
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    chunk_id=chunk_idx,
                    metadata={
                        **doc.metadata,
                        "heading_path": list(sec.heading_path),
                        "section_heading": section_heading,
                        "start_line": s_line,
                        "end_line": e_line,
                    },
                )
            )
            chunk_idx += 1

    return chunks


# ==============================================================================
# AST & Symbol-Aware Code Chunking (`ast` for Python, `tree-sitter` for others)
# ==============================================================================


@dataclass
class SymbolSpan:
    symbol_name: str
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive
    symbol_kind: str = "symbol"


def _node_start_line_py(node: ast.AST) -> int:
    lineno = getattr(node, "lineno", 1)
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return min([lineno] + [ getattr(d, "lineno", lineno) for d in decorators ])
    return int(lineno)


def _extract_python_symbols(
    text: str,
    lines: list[str],
    target_tokens: int,
    enc: Tokenizer,
) -> list[SymbolSpan] | None:
    """Parse Python source with `ast` into ordered SymbolSpan definitions."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    raw_spans: list[SymbolSpan] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s_line = _node_start_line_py(node)
            e_line = int(getattr(node, "end_lineno", s_line))
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            raw_spans.append(SymbolSpan(node.name, s_line, e_line, kind))
        elif isinstance(node, ast.ClassDef):
            s_line = _node_start_line_py(node)
            e_line = int(getattr(node, "end_lineno", s_line))
            class_text = "\n".join(lines[s_line - 1 : e_line])
            methods = [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            # Decompose oversized classes into class header + individual qualified methods
            if methods and len(enc.encode(class_text)) > target_tokens:
                first_method_start = _node_start_line_py(methods[0])
                if first_method_start > s_line:
                    raw_spans.append(
                        SymbolSpan(node.name, s_line, first_method_start - 1, "class")
                    )
                for m_idx, method in enumerate(methods):
                    m_start = _node_start_line_py(method)
                    m_end = int(getattr(method, "end_lineno", m_start))
                    if m_idx == len(methods) - 1 and m_end < e_line:
                        m_end = e_line
                    m_kind = (
                        "async_method"
                        if isinstance(method, ast.AsyncFunctionDef)
                        else "method"
                    )
                    raw_spans.append(
                        SymbolSpan(f"{node.name}.{method.name}", m_start, m_end, m_kind)
                    )
            else:
                raw_spans.append(SymbolSpan(node.name, s_line, e_line, "class"))

    if not raw_spans:
        return None

    return _fill_inter_symbol_gaps(raw_spans, lines)


_TS_PARSERS: dict[str, ts.Parser] = {}

_TS_LANGUAGE_FACTORIES: dict[str, Callable[[], object]] = {
    ".ts": tsts.language_typescript,
    ".tsx": tsts.language_tsx,
    ".js": tsjs.language,
    ".jsx": tsjs.language,
    ".go": tsgo.language,
    ".cpp": tscpp.language,
    ".h": tscpp.language,
    ".java": tsjava.language,
}


def _get_ts_parser(filetype: str) -> ts.Parser | None:
    factory = _TS_LANGUAGE_FACTORIES.get(filetype)
    if factory is None:
        return None
    if filetype not in _TS_PARSERS:
        lang = ts.Language(factory())
        _TS_PARSERS[filetype] = ts.Parser(lang)
    return _TS_PARSERS[filetype]


def _node_text(node: ts.Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace").strip()


def _unwrap_cpp_declarator_name(node: ts.Node | None) -> str:
    cur = node
    while cur is not None:
        if cur.type in ("identifier", "field_identifier", "qualified_identifier", "type_identifier", "destructor_name"):
            return _node_text(cur)
        inner = cur.child_by_field_name("declarator") or cur.child_by_field_name("name")
        if inner is not None:
            cur = inner
            continue
        # Fallback: look for first identifier child
        found = None
        for child in cur.children:
            if child.type in ("identifier", "field_identifier", "qualified_identifier", "destructor_name"):
                found = child
                break
        if found is not None:
            return _node_text(found)
        break
    return ""


def _extract_go_receiver_type(method_node: ts.Node) -> str:
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return ""
    for child in receiver.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                return _node_text(type_node).lstrip("*").strip()
    return ""


def _extract_ts_symbol_info(
    node: ts.Node,
    prefix: str = "",
) -> tuple[str, str, ts.Node] | None:
    """Return (symbol_name, symbol_kind, target_decl_node) for a tree-sitter AST node."""
    target = node
    if node.type == "export_statement":
        decl = node.child_by_field_name("declaration")
        if decl is not None:
            target = decl
        else:
            for c in node.children:
                if c.is_named and c.type != "decorator":
                    target = c
                    break

    ntype = target.type

    # TypeScript / JavaScript / Java / Go / C++ functions & methods
    if ntype in (
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "method_declaration",
        "constructor_declaration",
    ):
        name = _node_text(target.child_by_field_name("name"))
        if ntype == "method_declaration" and not prefix:
            recv = _extract_go_receiver_type(target)
            if recv and name:
                name = f"{recv}.{name}"
        if name:
            full_name = f"{prefix}.{name}" if prefix else name
            kind = "method" if ("method" in ntype or prefix) else "function"
            return full_name, kind, target

    if ntype == "function_definition":
        name = _unwrap_cpp_declarator_name(target.child_by_field_name("declarator"))
        if name:
            full_name = f"{prefix}.{name}" if prefix else name
            return full_name, ("method" if prefix else "function"), target

    # Classes, structs, interfaces, enums, type aliases
    if ntype in (
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "record_declaration",
        "class_specifier",
        "struct_specifier",
    ):
        name = _node_text(target.child_by_field_name("name"))
        if name:
            full_name = f"{prefix}.{name}" if prefix else name
            return full_name, "class", target

    # Go `type Foo struct` / `type Bar interface`
    if ntype == "type_declaration":
        for child in target.children:
            if child.type in ("type_spec", "type_alias"):
                name = _node_text(child.child_by_field_name("name"))
                if name:
                    full_name = f"{prefix}.{name}" if prefix else name
                    return full_name, "type", target

    # JS/TS top-level `const fn = (...) => ...` or `let service = ...`
    if ntype in ("lexical_declaration", "variable_declaration"):
        for child in target.children:
            if child.type == "variable_declarator":
                name = _node_text(child.child_by_field_name("name"))
                if name:
                    full_name = f"{prefix}.{name}" if prefix else name
                    return full_name, "variable", target

    return None


def _extract_class_method_nodes(decl_node: ts.Node) -> list[ts.Node]:
    body = decl_node.child_by_field_name("body")
    if body is None:
        for c in decl_node.children:
            if c.type in ("class_body", "field_declaration_list", "declaration_list"):
                body = c
                break
    if body is None:
        return []
    return [
        child
        for child in body.children
        if child.type
        in (
            "method_definition",
            "method_declaration",
            "constructor_declaration",
            "function_definition",
            "function_declaration",
        )
    ]


def _extract_treesitter_symbols(
    text: str,
    filetype: str,
    lines: list[str],
    target_tokens: int,
    enc: Tokenizer,
) -> list[SymbolSpan] | None:
    parser = _get_ts_parser(filetype)
    if parser is None:
        return None

    tree = parser.parse(text.encode("utf-8"))
    root = tree.root_node
    raw_spans: list[SymbolSpan] = []

    def _visit_top_nodes(nodes: list[ts.Node], namespace_prefix: str = "") -> None:
        for node in nodes:
            if node.type == "namespace_definition":
                ns_name = _node_text(node.child_by_field_name("name"))
                ns_body = node.child_by_field_name("body")
                next_prefix = f"{namespace_prefix}.{ns_name}" if namespace_prefix and ns_name else (ns_name or namespace_prefix)
                if ns_body is not None:
                    _visit_top_nodes(list(ns_body.children), next_prefix)
                continue

            info = _extract_ts_symbol_info(node, prefix=namespace_prefix)
            if info is None:
                continue

            sym_name, sym_kind, decl_node = info
            s_line = int(node.start_point.row) + 1
            e_line = int(node.end_point.row) + 1
            span_text = "\n".join(lines[s_line - 1 : e_line])

            method_nodes = _extract_class_method_nodes(decl_node)
            if method_nodes and len(enc.encode(span_text)) > target_tokens:
                first_m_start = int(method_nodes[0].start_point.row) + 1
                if first_m_start > s_line:
                    raw_spans.append(SymbolSpan(sym_name, s_line, first_m_start - 1, sym_kind))
                for m_idx, m_node in enumerate(method_nodes):
                    m_info = _extract_ts_symbol_info(m_node, prefix=sym_name)
                    m_name = m_info[0] if m_info is not None else f"{sym_name}.<anonymous>"
                    m_start = int(m_node.start_point.row) + 1
                    m_end = int(m_node.end_point.row) + 1
                    if m_idx == len(method_nodes) - 1 and m_end < e_line:
                        m_end = e_line
                    raw_spans.append(SymbolSpan(m_name, m_start, m_end, "method"))
            else:
                raw_spans.append(SymbolSpan(sym_name, s_line, e_line, sym_kind))

    _visit_top_nodes(list(root.children))
    if not raw_spans:
        return None

    return _fill_inter_symbol_gaps(raw_spans, lines)


def _fill_inter_symbol_gaps(
    symbol_spans: list[SymbolSpan],
    lines: list[str],
) -> list[SymbolSpan]:
    """Attach module-level headers/imports or standalone gap blocks between AST symbols."""
    ordered = sorted(symbol_spans, key=lambda s: (s.start_line, s.end_line))
    total_lines = len(lines)
    completed: list[SymbolSpan] = []
    cursor = 1

    for span in ordered:
        if span.start_line > cursor:
            gap_lines = lines[cursor - 1 : span.start_line - 1]
            if any(line.strip() for line in gap_lines):
                completed.append(
                    SymbolSpan("<module>", cursor, span.start_line - 1, "module")
                )
        completed.append(span)
        cursor = max(cursor, span.end_line + 1)

    if cursor <= total_lines:
        tail_lines = lines[cursor - 1 : total_lines]
        if any(line.strip() for line in tail_lines):
            completed.append(
                SymbolSpan("<module>", cursor, total_lines, "module")
            )

    return completed


def _chunk_code_doc(
    doc: RawDoc,
    config: ChunkConfig,
    enc: Tokenizer,
) -> list[Chunk]:
    lines = doc.text.splitlines()
    if not lines:
        return []

    filetype = str(doc.metadata.get("filetype", "")).lower()
    if filetype == ".py":
        spans = _extract_python_symbols(
            doc.text, lines, config.code_target_tokens, enc
        )
    else:
        spans = _extract_treesitter_symbols(
            doc.text, filetype, lines, config.code_target_tokens, enc
        )

    if not spans:
        spans = [SymbolSpan("<module>", 1, len(lines), "module")]

    chunks: list[Chunk] = []
    chunk_idx = 0
    for span in spans:
        span_lines = lines[span.start_line - 1 : span.end_line]
        windows = _split_lines_with_ranges(
            span_lines,
            span.start_line,
            config.code_target_tokens,
            config.code_overlap_tokens,
            enc,
        )
        for piece, s_line, e_line in windows:
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    chunk_id=chunk_idx,
                    metadata={
                        **doc.metadata,
                        "symbol_name": span.symbol_name,
                        "symbol_kind": span.symbol_kind,
                        "start_line": s_line,
                        "end_line": e_line,
                    },
                )
            )
            chunk_idx += 1

    return chunks


def _chunk_prose_doc(
    doc: RawDoc,
    config: ChunkConfig,
    enc: Tokenizer,
) -> list[Chunk]:
    lines = doc.text.splitlines()
    if not lines:
        return []

    windows = _split_lines_with_ranges(
        lines,
        1,
        config.prose_target_tokens,
        config.prose_overlap_tokens,
        enc,
    )
    return [
        Chunk(
            text=piece,
            source=doc.source,
            chunk_id=idx,
            metadata={
                **doc.metadata,
                "start_line": s_line,
                "end_line": e_line,
            },
        )
        for idx, (piece, s_line, e_line) in enumerate(windows)
    ]


def chunk_docs(
    docs: Iterable[RawDoc],
    *,
    embedder_model: str | None = None,
    config: ChunkConfig | None = None,
) -> Iterable[Chunk]:
    """Chunk documents using structure-aware Markdown heading chunking and AST code chunking."""
    resolved_cfg = config if config is not None else get_chunk_config(embedder_model)
    enc = _resolve_encoding(resolved_cfg.embedder_model)

    for d in docs:
        filetype = str(d.metadata.get("filetype", "")).lower()
        if filetype in SUPPORTED_CODE_SUFFIXES:
            yield from _chunk_code_doc(d, resolved_cfg, enc)
        elif filetype in {".md", ".markdown"}:
            yield from _chunk_markdown_doc(d, resolved_cfg, enc)
        else:
            yield from _chunk_prose_doc(d, resolved_cfg, enc)
