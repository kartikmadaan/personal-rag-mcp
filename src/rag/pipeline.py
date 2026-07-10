import time
from dotenv import load_dotenv

# Load all environment variables from the .env file BEFORE importing submodules
load_dotenv()

from .guards import guard_query
from .retrieve import hybrid_search
from .reranker import rerank
from .generate import generate

def query(q: str, k: int = 5) -> dict:
   t0 = time.perf_counter()
   q = guard_query(q)
   hits = rerank(q, hybrid_search(q, k_each=20, k_out=20), k=k)
   out = generate(q, hits)
   out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
   return out