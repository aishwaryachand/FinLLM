# api.py
# finllm/finllm/api.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any
from finllm.pipeline import rag_pipeline

app = FastAPI(
    title="FinLLM API",
    version="0.1.0",
    description="RAG-based Financial Document Q&A using Hugging Face models only",
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class Source(BaseModel):
    id: int
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_finllm(payload: QueryRequest):
    answer, docs = rag_pipeline(payload.query, top_k=payload.top_k)
    sources = [{"id": d.get("id", -1), "text": d.get("text", "")} for d in docs]
    return QueryResponse(answer=answer, sources=sources)

@app.get("/")
def root():
    return {
        "message": "Welcome to FinLLM. POST /query with {query, top_k} or open /docs for Swagger."
    }

