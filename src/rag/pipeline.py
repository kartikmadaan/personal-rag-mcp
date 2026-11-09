import time
from typing import Any

from dotenv import load_dotenv

# Load all environment variables from the .env file BEFORE importing submodules
load_dotenv()

from .generate import generate  # noqa: E402
from .guards import guard_query  # noqa: E402
from .reranker import rerank  # noqa: E402
from .retrieve import hybrid_search  # noqa: E402


def query(q: str, k: int = 5) -> dict[str, Any]:
    t0 = time.perf_counter()
    q = guard_query(q)
    hits = rerank(q, hybrid_search(q, k_each=20, k_out=20), k=k)
    out = generate(q, hits)
    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return out
