from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .guards import PromptInjectionDetected
from .pipeline import query

app = FastAPI(title="Personal RAG")


class QueryReq(BaseModel):
    q: str
    k: int = 5


@app.post("/query")
def query_endpoint(req: QueryReq) -> dict[str, Any]:
    try:
        return query(req.q, k=req.k)
    except PromptInjectionDetected as e:
        raise HTTPException(status_code=400, detail=f"refused: {e}")


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}
