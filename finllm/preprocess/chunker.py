# chunker.py

import json

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def save_chunks_to_jsonl(text_file, output_file):
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": chunk}) + "\n")
    return output_file

if __name__ == "__main__":
    input_txt = "finllm/data/parsed/sample.txt"
    output_jsonl = "finllm/data/parsed/sample_chunks.jsonl"
    save_chunks_to_jsonl(input_txt, output_jsonl)
    print(f"✅ Chunks saved to {output_jsonl}")
