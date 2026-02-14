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

# Special tokens — must match what the model was trained with (see config/model.py).
USER_TOKEN = "
