#generator.py
import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

GEN_MODEL = os.getenv("FINLLM_GEN_MODEL", "microsoft/Phi-3-mini-4k-instruct")

INSTRUCTION = """You are FinLLM, an assistant answering questions about financial filings and contracts.
Use ONLY the provided context to answer. If the answer cannot be found in the context, say "Insufficient evidence."
Cite the chunk IDs you used at the end as: Sources: [id1, id2, ...].

Question:
{question}

Context:
{context}

Answer:
"""

class Generator:
    def __init__(self, model_name: str = GEN_MODEL) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Mac-friendly: use MPS if available; else CPU. No custom dtype required.
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

    def build_prompt(self, question: str, context: str) -> str:
        return INSTRUCTION.format(question=question.strip(), context=context.strip())

    @torch.inference_mode()
    def generate(self,
                 question: str,
                 docs: List[dict],
                 max_new_tokens: int = 384,
                 temperature: float = 0.2,
                 top_p: float = 0.95) -> str:
        # Concatenate top docs into a single context block.
        context_parts = []
        for d in docs:
            cid = d.get("id", "NA")
            txt = d.get("text", "")
            context_parts.append(f"[id={cid}] {txt}")
        context = "\n\n".join(context_parts)

        prompt = self.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Return only the segment after "Answer:" if present.
        if "Answer:" in text:
            return text.split("Answer:", 1)[1].strip()
        return text.strip()
