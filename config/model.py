"""
Model configuration for the post-pretraining pipeline.

This file exists to centralize all model paths, HuggingFace model IDs,
and checkpoint paths. Supports environment variable overrides for flexibility.
"""

import os
from typing import List, Dict


class ModelConfig:
    """Configuration for model paths and identifiers."""
    
    # Base model for fine-tuning
    model_base: str = "stabilityai/stablelm-2-1_6b"
    
    # Reward model for GRPO alignment
    rm_model: str = "OpenAssistant/reward-model-deberta-v3-large"
    
    # Dataset identifiers
    ds_sft: str = "HuggingFaceH4/ultrachat_200k"
    ds_grpo: str = "PKU-Alignment/PKU-SafeRLHF"
    
    # Checkpoint paths - can be overridden via environment variables
    sft_checkpoint_path: str = os.getenv(
        "SFT_CHECKPOINT_PATH",
        "/content/drive/MyDrive/checkpoint-2079"
    )
    grpo_checkpoint_path: str = os.getenv(
        "GRPO_CHECKPOINT_PATH",
        "./stablelm-grpo/checkpoint-44"
    )
    
    # Output directories
    sft_output_dir: str = "./stablelm-sft"
    grpo_output_dir: str = "./stablelm-grpo"


class SpecialTokens:
    """Special tokens used for chat format."""
    
    @staticmethod
    def get_special_tokens():
        return {
            "eos_token": "<|endoftext|>",
            "additional_special_tokens": ["<|user|>", "<|assistant|>"]
        }
