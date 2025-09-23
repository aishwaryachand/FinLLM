# pipeline.py

from typing import List, Tuple
from finllm.rag.retriever import Retriever
from finllm.rag.generator import Generator

_retriever = None
_generator = None

def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator

def rag_pipeline(query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
    """
    Returns (answer_text, sources_list_of_dicts)
    """
    retriever = _get_retriever()
    generator = _get_generator()

    docs = retriever.search(query, top_k=top_k)
    answer = generator.generate(query, docs)
    return answer, docs
