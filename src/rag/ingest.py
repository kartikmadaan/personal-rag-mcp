from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from watchfiles import Change, watch

from .chunker import chunk_docs
from .embed import Embedder, embed_batch
from .loaders import (
    SUPPORTED_SUFFIXES,
    build_ignore_spec,
    compute_file_sha256,
    is_ignored,
    iter_corpus_files,
    load_file,
)
from .vectorstore import (
    ChunkRow,
    SyncResult,
    delete_sources,
    get_indexed_documents,
    init_db,
    upsert_chunks,
)


def _format_source(file_path: Path, root: Path) -> str:
    """Format `file_path` matching the style (relative vs absolute) of `root`."""
    if root.is_absolute():
        return str(file_path.resolve())
    try:
        rel = file_path.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except ValueError:
        return str(file_path)


def _source_belongs_to_root(source: str, root: Path) -> bool:
    """Return True if an indexed `source` string lives under `root`."""
    root_abs = root.resolve()
    src_path = Path(source)
    try:
        src_abs = src_path.resolve() if src_path.is_absolute() else (Path.cwd() / src_path).resolve()
        if root_abs.is_file():
            return src_abs == root_abs
        return src_abs == root_abs or root_abs in src_abs.parents
    except OSError:
        prefix = str(root).rstrip("/")
        return source == prefix or source.startswith(f"{prefix}/")


def sync_directory(
    root: Path | str,
    *,
    ensure_db: bool = True,
    prune_deleted: bool = True,
    force: bool = False,
    embedder: Embedder | None = None,
) -> SyncResult:
    """Incrementally synchronize a directory or file into PostgreSQL + pgvector.

    - Skips unchanged files BEFORE loading/chunking/embedding when `content_sha256` matches `documents`.
    - Re-embeds and replaces chunks for new or modified files.
    - Prunes orphaned chunks and `documents` records for files deleted from `root`.
    """
    root_path = Path(root)
    if ensure_db:
        init_db()

    existing_docs = get_indexed_documents()

    # Map resolved absolute paths of existing indexed documents under root -> stored source string
    existing_by_abs: dict[Path, str] = {}
    for stored_source in existing_docs:
        if _source_belongs_to_root(stored_source, root_path):
            sp = Path(stored_source)
            abs_p = sp.resolve() if sp.is_absolute() else (Path.cwd() / sp).resolve()
            existing_by_abs[abs_p] = stored_source

    discovered_abs: set[Path] = set()
    skipped_sources: list[str] = []
    changed_files: list[tuple[Path, str, str, float]] = []  # (path, source_key, sha256, mtime)

    for file_path in iter_corpus_files(root_path):
        abs_p = file_path.resolve()
        discovered_abs.add(abs_p)
        source_key = existing_by_abs.get(abs_p, _format_source(file_path, root_path))

        try:
            mtime = float(file_path.stat().st_mtime)
            sha256 = compute_file_sha256(file_path)
        except OSError as e:
            print(f"[skip] {file_path}: {e}")
            continue

        prev = existing_docs.get(source_key)
        if not force and prev is not None and prev.content_sha256 == sha256 and prev.chunk_count > 0:
            skipped_sources.append(source_key)
            continue

        changed_files.append((file_path, source_key, sha256, mtime))

    deleted_sources: list[str] = []
    if prune_deleted and root_path.exists():
        for abs_p, stored_source in existing_by_abs.items():
            if abs_p not in discovered_abs:
                deleted_sources.append(stored_source)

    rows: list[ChunkRow] = []
    doc_meta: dict[str, tuple[str, float]] = {}
    empty_changed_sources: list[str] = []

    for file_path, source_key, sha256, mtime in changed_files:
        try:
            raw_docs = load_file(file_path)
        except Exception as e:
            print(f"[skip] {file_path}: {e}")
            continue

        for d in raw_docs:
            d.source = source_key

        file_chunks = list(chunk_docs(raw_docs))
        if not file_chunks:
            # If an existing tracked file became empty, remove its stale chunks
            if source_key in existing_docs:
                empty_changed_sources.append(source_key)
            continue

        embeddings = embed_batch([c.text for c in file_chunks], embedder=embedder)
        doc_meta[source_key] = (sha256, mtime)
        for c, emb in zip(file_chunks, embeddings):
            rows.append(
                (
                    source_key,
                    c.chunk_id,
                    c.text.replace("\x00", ""),
                    json.dumps(c.metadata).replace("\\u0000", "").replace("\x00", ""),
                    emb,
                )
            )

    all_deleted = deleted_sources + empty_changed_sources
    sync_res = upsert_chunks(
        rows,
        document_meta=doc_meta,
        deleted_sources=all_deleted,
        force=force,
    )
    # Merge pre-skipped files (skipped prior to embedding) into the SyncResult
    sync_res.skipped = skipped_sources + [s for s in sync_res.skipped if s not in skipped_sources]
    return sync_res


def sync_file_events(
    root: Path | str,
    changes: Sequence[tuple[Change, str]],
    *,
    embedder: Embedder | None = None,
) -> SyncResult:
    """Process a batch of (Change, path) file events from `watchfiles` incrementally."""
    root_path = Path(root)
    spec = build_ignore_spec(root_path)
    existing_docs = get_indexed_documents()

    existing_by_abs: dict[Path, str] = {}
    for stored_source in existing_docs:
        sp = Path(stored_source)
        abs_p = sp.resolve() if sp.is_absolute() else (Path.cwd() / sp).resolve()
        existing_by_abs[abs_p] = stored_source

    to_upsert_paths: dict[Path, Path] = {}
    to_delete_sources: set[str] = set()

    for change_type, raw_path in changes:
        event_path = Path(raw_path)
        abs_p = event_path.resolve()

        if change_type == Change.deleted or not event_path.exists():
            to_upsert_paths.pop(abs_p, None)
            stored = existing_by_abs.get(abs_p, _format_source(event_path, root_path))
            if stored in existing_docs:
                to_delete_sources.add(stored)
            continue

        if event_path.is_dir():
            continue
        if event_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if is_ignored(event_path, root_path, spec):
            continue

        to_delete_sources.discard(existing_by_abs.get(abs_p, _format_source(event_path, root_path)))
        to_upsert_paths[abs_p] = event_path

    rows: list[ChunkRow] = []
    doc_meta: dict[str, tuple[str, float]] = {}
    skipped: list[str] = []

    for abs_p, file_path in sorted(to_upsert_paths.items()):
        source_key = existing_by_abs.get(abs_p, _format_source(file_path, root_path))
        try:
            mtime = float(file_path.stat().st_mtime)
            sha256 = compute_file_sha256(file_path)
        except OSError:
            continue

        prev = existing_docs.get(source_key)
        if prev is not None and prev.content_sha256 == sha256 and prev.chunk_count > 0:
            skipped.append(source_key)
            continue

        try:
            raw_docs = load_file(file_path)
        except Exception as e:
            print(f"[skip] {file_path}: {e}")
            continue

        for d in raw_docs:
            d.source = source_key

        file_chunks = list(chunk_docs(raw_docs))
        if not file_chunks:
            if source_key in existing_docs:
                to_delete_sources.add(source_key)
            continue

        embeddings = embed_batch([c.text for c in file_chunks], embedder=embedder)
        doc_meta[source_key] = (sha256, mtime)
        for c, emb in zip(file_chunks, embeddings):
            rows.append(
                (
                    source_key,
                    c.chunk_id,
                    c.text.replace("\x00", ""),
                    json.dumps(c.metadata).replace("\\u0000", "").replace("\x00", ""),
                    emb,
                )
            )

    if not rows and not to_delete_sources:
        return SyncResult(skipped=skipped)

    res = upsert_chunks(
        rows,
        document_meta=doc_meta,
        deleted_sources=sorted(to_delete_sources),
    )
    res.skipped = skipped + [s for s in res.skipped if s not in skipped]
    return res


def watch_directory(
    root: Path | str,
    *,
    initial_sync: bool = True,
    debounce_ms: int = 500,
    stop_event: threading.Event | None = None,
    on_sync: Callable[[SyncResult], Any] | None = None,
    embedder: Embedder | None = None,
) -> None:
    """Continuously watch `root` for file additions, edits, and deletions and sync them to the DB."""
    root_path = Path(root)
    if initial_sync:
        init_res = sync_directory(root_path, ensure_db=True, embedder=embedder)
        if on_sync is not None:
            on_sync(init_res)

    spec = build_ignore_spec(root_path)

    def _watch_filter(change: Change, path_str: str) -> bool:
        p = Path(path_str)
        if p.name in {".gitignore", ".ragignore"}:
            return True
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            return False
        return not is_ignored(p, root_path, spec)

    for changes in watch(
        root_path,
        watch_filter=_watch_filter,
        debounce=debounce_ms,
        stop_event=stop_event,
    ):
        # Rebuild ignore spec if .gitignore or .ragignore changed
        if any(Path(p).name in {".gitignore", ".ragignore"} for _, p in changes):
            spec = build_ignore_spec(root_path)
            res = sync_directory(root_path, ensure_db=False, embedder=embedder)
        else:
            res = sync_file_events(root_path, list(changes), embedder=embedder)

        if on_sync is not None:
            on_sync(res)


__all__ = [
    "SyncResult",
    "delete_sources",
    "sync_directory",
    "sync_file_events",
    "watch_directory",
]
