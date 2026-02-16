"""
FastAPI + vLLM Inference Server

Serves the fine-tuned & aligned StableLM 1.6B model via vLLM for
high-throughput, low-latency text generation.

The model weights are pulled from HuggingFace at startup:
  https://huggingface.co/kunjcr2/stablelm-1.6b-finetuned-aligned

Run locally:
    uvicorn api.app:app --reload --port 8000

Run via Docker:
    docker build -t stablelm-api -f docker/Dockerfile .
    docker run -p 8000:8000 --gpus all stablelm-api
"""

import os
from fastapi import FastAPI
from vllm import LLM, SamplingParams
from schema import QueryRequest, QueryResponse

# --- Config -----------------------------------------------------------------
# MODEL_ID can be overridden at deploy time with the HF_MODEL_ID env var.
# Default points to the fine-tuned + GRPO-aligned checkpoint on HuggingFace.
MODEL_ID = os.getenv("HF_MODEL_ID", "kunjcr2/stablelm-1.6b-finetuned-aligned")

# Special tokens -- must match what the model was trained with (see config/model.py).
USER_TOKEN = "<" + "|user|" + ">"
ASSISTANT_TOKEN = "<" + "|assistant|" + ">"
EOS_TOKEN = "<" + "|endoftext|" + ">"

# --- vLLM engine (loaded once at startup) ------------------------------------
llm = LLM(model=MODEL_ID, trust_remote_code=True)

# --- FastAPI app -------------------------------------------------------------
app = FastAPI(
    title="StableLM Inference API",
    description="Serves the fine-tuned & GRPO-aligned StableLM 1.6B model.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate", response_model=QueryResponse)
async def generate(request: QueryRequest):
    """
    Generate a response for the given user query.

    The prompt is wrapped with the same special tokens used during
    fine-tuning so the model sees a familiar chat format.
    """
    # Build the chat-formatted prompt
    prompt = f"{USER_TOKEN}\n{request.query}\n{ASSISTANT_TOKEN}\n"

    # Map request params to vLLM SamplingParams
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=[EOS_TOKEN],
    )

    # Run inference
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    num_tokens = len(outputs[0].outputs[0].token_ids)

    return QueryResponse(
        query=request.query,
        response=generated_text,
        tokens=num_tokens,
    )
