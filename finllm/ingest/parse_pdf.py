# parse_pdf.py
# finllm/finllm/ingest/parse_pdf.py
from pdfminer.high_level import extract_text
import os

def parse_pdf(input_path, output_path):
    """
    Extract text from a PDF and save to a .txt file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")
    
    text = extract_text(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path

if __name__ == "__main__":
    raw_pdf = "finllm/data/raw/sample.pdf"
    parsed_txt = "finllm/data/parsed/sample.txt"
    parse_pdf(raw_pdf, parsed_txt)
    print(f" Parsed PDF saved to {parsed_txt}")
