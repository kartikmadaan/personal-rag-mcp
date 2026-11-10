import argparse
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

load_dotenv()

from rag.ingest import SyncResult, sync_directory, watch_directory  # noqa: E402


def _print_sync_summary(res: SyncResult) -> None:
    print(
        f"sync: inserted={len(res.inserted)} updated={len(res.updated)} "
        f"skipped={len(res.skipped)} deleted={len(res.deleted)} "
        f"chunks_upserted={res.chunks_upserted}"
    )


def main(
    corpus_dir: str,
    *,
    watch: bool = False,
    force: bool = False,
    no_prune: bool = False,
) -> SyncResult | None:
    root = Path(corpus_dir)
    if watch:
        print(f"Watching {root} for changes (Ctrl+C to stop)...")
        try:
            watch_directory(
                root,
                initial_sync=True,
                on_sync=_print_sync_summary,
            )
        except KeyboardInterrupt:
            print("\nStopped watching.")
        return None

    res = sync_directory(
        root,
        ensure_db=True,
        prune_deleted=not no_prune,
        force=force,
    )
    _print_sync_summary(res)
    print("done")
    return res


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally ingest a directory into PostgreSQL + pgvector."
    )
    parser.add_argument("corpus_dir", help="Path to directory or file to ingest")
    parser.add_argument(
        "-w",
        "--watch",
        action="store_true",
        help="Continuously watch directory for file additions, modifications, and deletions",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-embedding even if content SHA-256 has not changed",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Do not delete indexed sources that are no longer present in corpus_dir",
    )
    args = parser.parse_args(argv)
    main(
        args.corpus_dir,
        watch=args.watch,
        force=args.force,
        no_prune=args.no_prune,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
