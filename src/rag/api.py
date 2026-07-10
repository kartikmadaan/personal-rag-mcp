from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .pipeline import query
from .guards import PromptInjectionDetected


app = FastAPI(title="Personal RAG")


class QueryReq(BaseModel):
   q: str
   k: int = 5




@app.post("/query")
def query_endpoint(req: QueryReq):
   try:
       return query(req.q, k=req.k)
   except PromptInjectionDetected as e:
       raise HTTPException(status_code=400, detail=f"refused: {e}")




@app.get("/healthz")
def healthz():
   return {"ok": True}