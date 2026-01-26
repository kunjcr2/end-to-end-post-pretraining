"""
Inference utilities for trained models.

This file exists to provide a unified interface for running inference with
any stage of the model (base, SFT, or aligned). It handles model loading,
tokenization, and response generation.

You will implement:
- Model loading utilities for each checkpoint type
- generate_response function with proper chat formatting
- Batch inference support
- Integration with vLLM for production serving
"""

# TODO: Implement inference utilities
# Reference: ipynb/finetune.py (generate_response function)

from flask import Flask, request, jsonify

# need vLLM
from vllm import LLM, SamplingParams

# prompt
# POST /ask
# {
#     "messages": [
#         {
#             "role": "user",
#             "content": "Hello, how are you?"
#         }
#     ]
# }
# Add it to the system prompt

# sampling params
params = SamplingParams()  # add params for infernce

# Load LLM from huggingface
llm = LLM() # Add model from hugginface

# generate output
outputs = llm.generate()