import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag.chunker import chunk_docs  # noqa: E402
from rag.embed import embed_batch  # noqa: E402
from rag.loaders import load_corpus  # noqa: E402
from rag.vectorstore import init_db, upsert_chunks  # noqa: E402


def main(corpus_dir: str) -> None:
    init_db()
    chunks = list(chunk_docs(load_corpus(Path(corpus_dir))))
    print(f"chunks: {len(chunks)}")
    embeddings = embed_batch([c.text for c in chunks])

    rows = [
        (
            c.source,
            c.chunk_id,
            c.text.replace("\x00", ""),
            json.dumps(c.metadata).replace("\\u0000", "").replace("\x00", ""),
            e,
        )
        for c, e in zip(chunks, embeddings)
    ]
    upsert_chunks(rows)
    print("done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.ingest <corpus_dir>")
        sys.exit(1)
    main(sys.argv[1])

