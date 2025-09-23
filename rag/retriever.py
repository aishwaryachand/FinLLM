# retriever.py
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("FINLLM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
INDEX_PATH = os.getenv("FINLLM_INDEX_PATH", "finllm/data/index/index.faiss")
META_PATH = os.getenv("FINLLM_META_PATH", "finllm/data/index/meta.jsonl")

class Retriever:
    """
    FAISS retriever over chunked documents.
    Default similarity: inner product on normalized embeddings (cosine).
    """
    def __init__(self,
                 index_path: str = INDEX_PATH,
                 meta_path: str = META_PATH,
                 embed_model: str = EMBED_MODEL) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}")

        self.model = SentenceTransformer(embed_model)
        self.index = faiss.read_index(index_path)
        self.meta: List[Dict] = [json.loads(l) for l in open(meta_path, "r", encoding="utf-8")]

    def _encode(self, texts: List[str]) -> np.ndarray:
        X = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return X.astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns a list of top_k metadata dicts (each includes: id, text, and optional fields).
        """
        qvec = self._encode([query])
        _, I = self.index.search(qvec, top_k)
        ids = I[0].tolist()
        results = [self.meta[i] for i in ids if 0 <= i < len(self.meta)]
        return results

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        qvec = self._encode([query])
        D, I = self.index.search(qvec, top_k)
        ids = I[0].tolist()
        scores = D[0].tolist()
        out: List[Tuple[Dict, float]] = []
        for idx, score in zip(ids, scores):
            if 0 <= idx < len(self.meta):
                out.append((self.meta[idx], float(score)))
        return out
