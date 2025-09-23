# README.md
# FinLLM – Financial Document Analytics

## Overview
FinLLM is a document analytics system that applies Retrieval-Augmented Generation (RAG) and transformer-based models to financial filings and contracts. The system ingests large, complex documents such as SEC 10-K and 10-Q filings, parses them into manageable sections, and enables question-answering, anomaly detection, and compliance scoring.  

The project is designed to demonstrate both technical and product value: a scalable pipeline for financial document analysis that can reduce manual effort, improve accuracy, and provide transparency in high-stakes decision making.

---

## Objectives
1. Enable natural language Q&A over financial filings and contracts.  
2. Provide structured insights such as risk factors, anomalies, and compliance gaps.  
3. Compare zero-shot prompting, RAG-based answering, and fine-tuned models to show measurable performance gains.  
4. Deliver results through a production-ready API interface.  

---

## Technology Stack
- **Languages**: Python  
- **Libraries & Frameworks**:  
  - Hugging Face Transformers, Sentence-Transformers  
  - FAISS (vector database)  
  - FastAPI (API layer)  
  - pdfminer.six, unstructured (document parsing)  
- **Models**:  
  - `BAAI/bge-small-en-v1.5` for embeddings (lightweight, Mac-friendly)  
  - `microsoft/Phi-3-mini-4k-instruct` for answer generation (inference on Mac)  
  - `ProsusAI/finbert` for financial sentiment (optional analytics)  
  - Optional fine-tuning with QLoRA adapters on `Mistral-7B-Instruct` (trained externally)  

---

## Project Flow
1. **Ingestion**  
   Financial PDFs (e.g., 10-K filings) are parsed into raw text using `pdfminer.six`.  

2. **Preprocessing**  
   - Text is split into overlapping chunks of ~500 tokens.  
   - Chunks are embedded into dense vectors using `bge-small`.  
   - Embeddings and metadata are stored in a FAISS index for retrieval.  

3. **Retrieval-Augmented Generation (RAG)**  
   - A user query is embedded and matched against the FAISS index.  
   - The top-k most relevant chunks are retrieved.  
   - The generator model (`Phi-3-mini`) uses these chunks as context to produce a grounded answer.  

4. **Analytics Layer (extension)**  
   - Sentiment analysis of risk sections (FinBERT).  
   - KPI extraction (Revenue, EPS, COGS) with regex and LLM verification.  
   - Compliance scoring through rubric-based LLM prompts.  

5. **Serving**  
   - A FastAPI service exposes the pipeline via a `/query` endpoint.  
   - Input: JSON payload with query text.  
   - Output: Answer text with supporting source chunks.  

---

## Example Use Case
- **Input query**: "What cybersecurity risks did the company disclose in its most recent filing?"  
- **Process**: System retrieves the relevant Risk Factors section, feeds it into the generator, and outputs an evidence-based answer with chunk citations.  
- **Output**:  
  - Answer: "The filing notes risks related to potential unauthorized access, data breaches, and evolving regulatory requirements."  
  - Sources: IDs of the retrieved chunks.  

---

## How to Run the Application


### Install dependencies  
pip install -r requirements.txt  

### Add a sample PDF  
Place a financial document (e.g., sample.pdf) into:  
finllm/data/raw/  

### Parse and preprocess  
Run the preprocessing steps in order:  
python finllm/ingest/parse_pdf.py  
python finllm/preprocess/chunker.py  
python finllm/preprocess/build_index.py  

### Start the API server  
uvicorn finllm.api:app --reload --port 8000  

### Query the system  
Open your browser at:  
http://127.0.0.1:8000/docs  

Use the Swagger interface to send a POST request to /query with:  
{  
  "query": "What are the key financial risks?",  
  "top_k": 5  
}  

