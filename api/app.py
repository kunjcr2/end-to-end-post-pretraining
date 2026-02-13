"""
FastAPI + vLLM Inference Server

Serves the fine-tuned StableLM model from HuggingFace using vLLM
for high-performance inference.

Run:
    uvicorn api.app:app --reload --port 8000
"""

import os
from fastapi import FastAPI
from vllm import LLM, SamplingParams
from schema import QueryRequest, QueryResponse

# --- Config -----------------------------------------------------------------
MODEL_ID = os.getenv("HF_MODEL_ID", "kunjcr2/stablelm-1.6b-finetuned-aligned")
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOT_TOKEN = "<|endoftext|>"

# --- App --------------------------------------------------------------------
app = FastAPI(
    title="StableLM Inference API",
    description="vLLM-powered inference for the fine-tuned and aligned StableLM 1.6B model.",
    version="1.0.0",
)

# --- Model (loaded once at startup) ----------------------------------------
llm: LLM | None = None

@app.on_event("startup")
async def load_model():
    global llm
    llm = LLM(model=MODEL_ID, trust_remote_code=True)

# --- Routes -----------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_ID}

@app.post("/query", response_model=QueryResponse)
def query_model(req: QueryRequest):
    """Generate a response from the model."""

    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    # Format prompt with special tokens used during training
    formatted_prompt = f"{USER_TOKEN}\n{req.query}\n{ASSISTANT_TOKEN}\n"

    # TODO: Add tokenization before passing in.
    tokenized_prompt = []

    outputs = llm.generate(tokenized_prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text

    # Strip trailing EOT token if present
    if generated_text.endswith(EOT_TOKEN):
        generated_text = generated_text[: -len(EOT_TOKEN)].strip()

    return QueryResponse(
        query=req.query,
        response=generated_text,
        tokens=len(outputs[0].outputs[0].token_ids),
    )
