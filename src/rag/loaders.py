import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pathspec
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

SUPPORTED_CODE_SUFFIXES: set[str] = {
    ".py",
    ".go",
    ".cpp",
    ".h",
    ".java",
    ".ts",
    ".js",
}
SUPPORTED_PROSE_SUFFIXES: set[str] = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
    ".rst",
}
SUPPORTED_SUFFIXES: set[str] = SUPPORTED_CODE_SUFFIXES | SUPPORTED_PROSE_SUFFIXES

DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".nox/",
    "dist/",
    "build/",
    "*.egg-info/",
    ".DS_Store",
    "*.py[cod]",
    "*.so",
    "*.dylib",
]


@dataclass
class RawDoc:
    text: str
    source: str  # path ( matching input root style )
    metadata: dict[str, Any] = field(default_factory=dict)
    content_sha256: str = ""
    mtime: float = 0.0


def compute_file_sha256(path: Path) -> str:
    """Compute the hex SHA-256 digest of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def build_ignore_spec(root: Path) -> pathspec.PathSpec:
    """Build a gitwildmatch PathSpec combining default patterns, .gitignore, and .ragignore."""
    patterns: list[str] = list(DEFAULT_IGNORE_PATTERNS)
    search_root = root if root.is_dir() else root.parent

    # Walk upwards to include parent repository .gitignore/.ragignore if root is a subdirectory
    ancestors: list[Path] = []
    cur = search_root.resolve()
    while True:
        ancestors.append(cur)
        if (cur / ".git").exists() or cur.parent == cur:
            break
        cur = cur.parent

    for directory in reversed(ancestors):
        for ignore_name in (".gitignore", ".ragignore"):
            ignore_file = directory / ignore_name
            if ignore_file.is_file():
                try:
                    lines = ignore_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                    patterns.extend(lines)
                except OSError:
                    pass

    if search_root.is_dir():
        # Also include nested .gitignore / .ragignore files prefixed by their relative directory
        for sub_ignore_name in (".gitignore", ".ragignore"):
            for sub_ignore in search_root.rglob(sub_ignore_name):
                if sub_ignore.parent == search_root:
                    continue
                try:
                    rel_dir = sub_ignore.parent.relative_to(search_root).as_posix()
                except ValueError:
                    continue
                # Skip ignore files that live inside already-ignored directories (e.g. .venv)
                if any(
                    part in {".git", ".venv", "venv", "node_modules", "__pycache__"}
                    for part in sub_ignore.parts
                ):
                    continue
                try:
                    for raw_line in sub_ignore.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("!"):
                            neg = line[1:].lstrip("/")
                            patterns.append(f"!{rel_dir}/{neg}")
                        else:
                            clean = line.lstrip("/")
                            patterns.append(f"{rel_dir}/{clean}")
                except OSError:
                    pass

    return pathspec.PathSpec.from_lines("gitignore", patterns)


def is_ignored(
    path: Path,
    root: Path,
    spec: pathspec.PathSpec | None = None,
) -> bool:
    """Return True if `path` matches default, .gitignore, or .ragignore rules under `root`."""
    if spec is None:
        spec = build_ignore_spec(root)

    base = root if root.is_dir() else root.parent
    try:
        rel = path.relative_to(base).as_posix()
    except ValueError:
        try:
            rel = path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            rel = path.name

    if rel in ("", "."):
        return False

    if path.is_dir():
        return spec.match_file(rel) or spec.match_file(f"{rel}/")
    return spec.match_file(rel)


def iter_corpus_files(root: Path) -> Iterator[Path]:
    """Yield supported, non-ignored files under `root` in deterministic order."""
    if root.is_file():
        spec = build_ignore_spec(root.parent)
        if (
            root.suffix.lower() in SUPPORTED_SUFFIXES
            and not is_ignored(root, root.parent, spec)
        ):
            yield root
        return

    if not root.exists():
        return

    spec = build_ignore_spec(root)
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if is_ignored(p, root, spec):
            continue
        yield p


def load_file(path: Path) -> list[RawDoc]:
    """Load a single file into one or more RawDoc instances with sha256 and mtime metadata."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return []

    stat = path.stat()
    mtime = float(stat.st_mtime)
    sha256 = compute_file_sha256(path)

    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".md", ".markdown"}:
        docs = UnstructuredMarkdownLoader(str(path)).load()
    elif suffix in {".txt", ".rst"} | SUPPORTED_CODE_SUFFIXES:
        docs = TextLoader(str(path), autodetect_encoding=True).load()
    else:
        return []

    return [
        RawDoc(
            text=d.page_content,
            source=str(path),
            metadata={
                **d.metadata,
                "filetype": suffix,
                "mtime": mtime,
                "content_sha256": sha256,
            },
            content_sha256=sha256,
            mtime=mtime,
        )
        for d in docs
    ]


def load_corpus(root: Path) -> Iterator[RawDoc]:
    """Iterate over all supported, non-ignored files in `root` and yield RawDoc items."""
    for p in iter_corpus_files(root):
        try:
            yield from load_file(p)
        except Exception as e:
            print(f"[skip] {p}: {e}")
