"""
Data configuration for dataset processing.

This file exists to define data-related settings including sample limits
and output file paths for processed datasets.
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data processing and limits."""
    
    # SFT data settings
    sft_max_samples: int = 140000
    sft_output_file: str = "ultrachat_sft.json"
    sft_train_split: float = 0.95
    ds_sft: str = "HuggingFaceH4/ultrachat_200k"

    # GRPO data settings
    grpo_max_samples: int = 4000
    grpo_output_file: str = "pku_grpo.json"
    grpo_train_split: float = 0.95
    ds_grpo: str = "PKU-Alignment/PKU-SafeRLHF"
    
    # Tokenization settings
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
