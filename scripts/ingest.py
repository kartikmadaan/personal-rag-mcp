import sys
import os
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv() 

from rag.loaders import load_corpus
from rag.chunker import chunk_docs
from rag.embed import embed_batch
from rag.vectorstore import init_db, upsert_chunks

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
           json.dumps(c.metadata).replace("\\u0000", "").replace("\x00", ""), # FIX: Cleans NUL bytes from metadata JSON string
           e
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
