# build_index.py
# finllm/finllm/preprocess/build_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def build_faiss_index(jsonl_file, index_dir="finllm/data/index"):
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    texts, ids = [], []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["id"])
            texts.append(rec["text"])

    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for i, text in zip(ids, texts):
            f.write(json.dumps({"id": i, "text": text}) + "\n")

    print(f"FAISS index built and saved in {index_dir}")

if __name__ == "__main__":
    jsonl_file = "finllm/data/parsed/sample_chunks.jsonl"
    build_faiss_index(jsonl_file)

