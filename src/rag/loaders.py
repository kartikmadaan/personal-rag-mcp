from pathlib import Path
from typing import Iterator
from dataclasses import dataclass


from langchain_community.document_loaders import (
   PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
)




@dataclass
class RawDoc:
   text: str
   source: str          # absolute path
   metadata: dict       # filetype, mtime, etc.




def load_corpus(root: Path) -> Iterator[RawDoc]:
   for p in root.rglob("*"):
       if p.is_dir():
           continue
       suffix = p.suffix.lower()
       try:
           if suffix == ".pdf":
               docs = PyPDFLoader(str(p)).load()
           elif suffix in {".md", ".markdown"}:
               docs = UnstructuredMarkdownLoader(str(p)).load()
           elif suffix in {".txt", ".rst"}:
               docs = TextLoader(str(p), autodetect_encoding=True).load()
           elif suffix in {".py", ".go", ".cpp", ".h", ".java", ".ts", ".js"}:
               docs = TextLoader(str(p), autodetect_encoding=True).load()
           else:
               continue
           for d in docs:
               yield RawDoc(
                   text=d.page_content,
                   source=str(p),
                   metadata={**d.metadata, "filetype": suffix},
               )
       except Exception as e:
           print(f"[skip] {p}: {e}")