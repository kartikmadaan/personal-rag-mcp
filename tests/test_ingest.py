"""Unit tests for incremental, idempotent ingestion, ignore rules, and live file watching."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from watchfiles import Change

from rag import ingest, loaders, vectorstore


class FakeEncoding:
    """Deterministic whitespace token encoding for offline unit tests."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        return " ".join(f"tok{t}" for t in tokens)


class FakeDBState:
    """In-memory mock of the PostgreSQL `documents` and `chunks` tables."""

    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, str, float, int, None]] = {}
        self.chunks: dict[str, list[tuple[str, int, str, str, list[float]]]] = {}
        self.in_transaction: bool = False
        self.committed_transactions: int = 0


class FakeCursor:
    def __init__(self, state: FakeDBState) -> None:
        self.state = state
        self._results: list[Any] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        norm = " ".join(sql.split())
        if norm.startswith("SELECT source, content_sha256, mtime, chunk_count, updated_at FROM documents"):
            if params and "WHERE source =" in norm:
                exact, prefix_like = params[0], params[1][:-1]
                rows = [
                    rec
                    for src, rec in sorted(self.state.documents.items())
                    if src == exact or src.startswith(prefix_like)
                ]
            else:
                rows = [rec for _, rec in sorted(self.state.documents.items())]
            self._results = rows
        elif norm == "DELETE FROM chunks WHERE source = %s":
            assert params is not None
            self.state.chunks.pop(params[0], None)
        elif norm == "DELETE FROM chunks WHERE source = ANY(%s)":
            assert params is not None
            for src in params[0]:
                self.state.chunks.pop(src, None)
        elif norm == "DELETE FROM documents WHERE source = ANY(%s)":
            assert params is not None
            for src in params[0]:
                self.state.documents.pop(src, None)
        elif norm.startswith("INSERT INTO documents"):
            assert params is not None
            src, sha256, mtime, chunk_count = params
            self.state.documents[src] = (src, sha256, float(mtime), int(chunk_count), None)

    def executemany(self, sql: str, seq_of_params: list[tuple[Any, ...]]) -> None:
        norm = " ".join(sql.split())
        if norm.startswith("INSERT INTO chunks"):
            for src, cid, txt, meta, emb in seq_of_params:
                self.state.chunks.setdefault(src, []).append((src, cid, txt, meta, list(emb)))

    def fetchall(self) -> list[Any]:
        return list(self._results)


class FakeConn:
    def __init__(self, state: FakeDBState) -> None:
        self.state = state

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.state)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.state.in_transaction = True
        try:
            yield
            self.state.committed_transactions += 1
        finally:
            self.state.in_transaction = False


def _make_fake_get_conn(state: FakeDBState):
    @contextmanager
    def _fake_get_conn() -> Iterator[FakeConn]:
        yield FakeConn(state)

    return _fake_get_conn


def test_ddl_is_idempotent_and_adds_documents_table() -> None:
    """Verify destructive DROP TABLE is removed and `documents` metadata schema is present."""
    upper_ddl = vectorstore.DDL.upper()
    assert "DROP TABLE" not in upper_ddl
    assert "CREATE TABLE IF NOT EXISTS DOCUMENTS" in upper_ddl
    assert "CONTENT_SHA256" in upper_ddl
    assert "MTIME" in upper_ddl
    assert "CHUNK_COUNT" in upper_ddl
    assert "UPDATED_AT" in upper_ddl
    assert "CREATE TABLE IF NOT EXISTS CHUNKS" in upper_ddl


def test_upsert_chunks_incremental_skip_update_and_orphan_cleanup() -> None:
    state = FakeDBState()

    with patch("rag.vectorstore.get_conn", _make_fake_get_conn(state)):
        # 1. Initial insert of a 3-chunk document
        initial_rows = [
            ("notes/design.md", 0, "chunk zero", json.dumps({"content_sha256": "hash_v1", "mtime": 10.0}), [0.1, 0.2]),
            ("notes/design.md", 1, "chunk one", json.dumps({"content_sha256": "hash_v1", "mtime": 10.0}), [0.2, 0.3]),
            ("notes/design.md", 2, "chunk two", json.dumps({"content_sha256": "hash_v1", "mtime": 10.0}), [0.3, 0.4]),
        ]
        res1 = vectorstore.upsert_chunks(initial_rows)
        assert res1.inserted == ["notes/design.md"]
        assert res1.updated == []
        assert res1.skipped == []
        assert res1.chunks_upserted == 3
        assert len(state.chunks["notes/design.md"]) == 3
        assert state.documents["notes/design.md"][1] == "hash_v1"
        assert state.documents["notes/design.md"][3] == 3
        assert state.committed_transactions == 1

        # 2. Second upsert with identical content_sha256 is skipped
        res2 = vectorstore.upsert_chunks(initial_rows)
        assert res2.skipped == ["notes/design.md"]
        assert res2.inserted == []
        assert res2.updated == []
        assert res2.chunks_upserted == 0
        assert len(state.chunks["notes/design.md"]) == 3

        # 3. Updating file so it shrinks from 3 chunks to 1 chunk deletes orphaned chunks 1 and 2
        shrunk_rows = [
            ("notes/design.md", 0, "shorter v2", json.dumps({"content_sha256": "hash_v2", "mtime": 20.0}), [0.9, 0.9]),
        ]
        res3 = vectorstore.upsert_chunks(shrunk_rows)
        assert res3.updated == ["notes/design.md"]
        assert res3.chunks_upserted == 1
        assert len(state.chunks["notes/design.md"]) == 1
        assert state.chunks["notes/design.md"][0][2] == "shorter v2"
        assert state.documents["notes/design.md"][1] == "hash_v2"
        assert state.documents["notes/design.md"][3] == 1

        # 4. Explicit deletion removes both chunks and documents metadata record
        res4 = vectorstore.upsert_chunks([], deleted_sources=["notes/design.md"])
        assert res4.deleted == ["notes/design.md"]
        assert "notes/design.md" not in state.chunks
        assert "notes/design.md" not in state.documents


def test_loaders_respects_default_ignores_gitignore_and_ragignore(tmp_path: Path) -> None:
    # Set up workspace with .gitignore and .ragignore
    (tmp_path / ".gitignore").write_text("ignored_dir/\n*.secret.txt\n", encoding="utf-8")
    (tmp_path / ".ragignore").write_text("drafts/\n!drafts/keep.txt\n", encoding="utf-8")

    # Standard files that SHOULD be loaded
    keep_code = tmp_path / "src" / "main.py"
    keep_code.parent.mkdir(parents=True)
    keep_code.write_text("def hello():\n    return 42\n", encoding="utf-8")

    keep_draft = tmp_path / "drafts" / "keep.txt"
    keep_draft.parent.mkdir(parents=True)
    keep_draft.write_text("Important draft to index", encoding="utf-8")

    # Files that MUST be skipped by default rules, .gitignore, or .ragignore
    git_internal = tmp_path / ".git" / "config.txt"
    git_internal.parent.mkdir(parents=True)
    git_internal.write_text("git internal", encoding="utf-8")

    venv_file = tmp_path / ".venv" / "lib" / "pkg.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("venv package", encoding="utf-8")

    node_mod = tmp_path / "node_modules" / "pkg" / "index.js"
    node_mod.parent.mkdir(parents=True)
    node_mod.write_text("console.log(1);", encoding="utf-8")

    gitignored_dir_file = tmp_path / "ignored_dir" / "notes.txt"
    gitignored_dir_file.parent.mkdir(parents=True)
    gitignored_dir_file.write_text("ignored by gitignore dir", encoding="utf-8")

    gitignored_glob_file = tmp_path / "api.secret.txt"
    gitignored_glob_file.write_text("secret key", encoding="utf-8")

    ragignored_file = tmp_path / "drafts" / "skip.txt"
    ragignored_file.write_text("skipped draft", encoding="utf-8")

    discovered = list(loaders.iter_corpus_files(tmp_path))
    rel_paths = {p.relative_to(tmp_path).as_posix() for p in discovered}

    assert rel_paths == {"src/main.py", "drafts/keep.txt"}

    docs = list(loaders.load_corpus(tmp_path))
    assert len(docs) == 2
    for d in docs:
        assert d.content_sha256
        assert d.mtime > 0
        assert d.metadata["content_sha256"] == d.content_sha256
        assert d.metadata["mtime"] == d.mtime


def test_sync_directory_skips_unchanged_reembeds_modified_and_prunes_deleted(tmp_path: Path) -> None:
    state = FakeDBState()
    fake_enc = FakeEncoding()

    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.py"
    file_a.write_text("alpha one two", encoding="utf-8")
    file_b.write_text("def beta(): pass", encoding="utf-8")

    embed_calls: list[list[str]] = []

    class TrackingEmbedder:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            embed_calls.append(list(texts))
            return [[0.5, 0.5] for _ in texts]

    embedder = TrackingEmbedder()

    with (
        patch("rag.ingest.init_db"),
        patch("rag.vectorstore.get_conn", _make_fake_get_conn(state)),
        patch("rag.chunker._get_encoding", return_value=fake_enc),
    ):
        # 1. Initial sync embeds both files
        res1 = ingest.sync_directory(tmp_path, embedder=embedder)
        assert len(res1.inserted) == 2
        assert res1.skipped == []
        assert len(embed_calls) == 2

        # 2. Second sync with no modifications skips both files BEFORE calling embedder
        embed_calls.clear()
        res2 = ingest.sync_directory(tmp_path, embedder=embedder)
        assert len(res2.skipped) == 2
        assert res2.inserted == []
        assert res2.updated == []
        assert embed_calls == []  # No unnecessary re-embedding!

        # 3. Modify file_a and delete file_b -> only file_a is re-embedded, file_b is deleted
        file_a.write_text("alpha updated text with more words", encoding="utf-8")
        file_b.unlink()

        res3 = ingest.sync_directory(tmp_path, embedder=embedder)
        assert res3.updated == [str(file_a.resolve())]
        assert res3.deleted == [str(file_b.resolve())]
        assert len(embed_calls) == 1
        assert str(file_b.resolve()) not in state.documents
        assert str(file_b.resolve()) not in state.chunks


def test_sync_file_events_and_watch_directory_live_updates(tmp_path: Path) -> None:
    state = FakeDBState()
    fake_enc = FakeEncoding()

    note = tmp_path / "note.txt"
    note.write_text("initial live note", encoding="utf-8")
    ignored = tmp_path / ".venv" / "ignored.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("ignored venv code", encoding="utf-8")

    class DummyEmbedder:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    embedder = DummyEmbedder()
    fake_watch_batches = [
        {(Change.added, str(note)), (Change.added, str(ignored))},
        {(Change.modified, str(note))},
        {(Change.deleted, str(note))},
    ]

    def _fake_watch(*args: object, **kwargs: object) -> Iterator[set[tuple[Change, str]]]:
        # Perform initial add
        yield fake_watch_batches[0]
        # Mutate file content before modified event
        note.write_text("modified live note content", encoding="utf-8")
        yield fake_watch_batches[1]
        # Remove file before deleted event
        note.unlink()
        yield fake_watch_batches[2]

    observed_results: list[ingest.SyncResult] = []

    with (
        patch("rag.ingest.init_db"),
        patch("rag.vectorstore.get_conn", _make_fake_get_conn(state)),
        patch("rag.chunker._get_encoding", return_value=fake_enc),
        patch("rag.ingest.watch", side_effect=_fake_watch),
    ):
        stop_event = threading.Event()
        ingest.watch_directory(
            tmp_path,
            initial_sync=False,
            stop_event=stop_event,
            on_sync=observed_results.append,
            embedder=embedder,
        )

    assert len(observed_results) == 3
    # Batch 1: added note.txt (ignored.py in .venv was filtered out)
    assert observed_results[0].inserted == [str(note.resolve())]
    # Batch 2: modified note.txt
    assert observed_results[1].updated == [str(note.resolve())]
    # Batch 3: deleted note.txt
    assert observed_results[2].deleted == [str(note.resolve())]
    assert state.documents == {}
    assert state.chunks == {}
