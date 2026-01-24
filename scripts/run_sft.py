"""
CLI script to launch Supervised Fine-Tuning.

This file exists to provide a command-line interface for running SFT training.
It loads configurations and calls the training function from src.train.

Usage:
    python scripts/run_sft.py
    python scripts/run_sft.py --epochs 3 --lr 5e-5
"""

# TODO: Implement CLI with argparse
# - Parse command line arguments to override config values
# - Initialize configs from config/ module
# - Call training function from src.train

import argparse
from config import SFTTrainingConfig, SFTLoRAConfig, DataConfig
from src import train

sft_training_config = SFTTrainingConfig()
sft_lora_config = SFTLoRAConfig()
data_config = DataConfig()

if __name__ == "__main__":
    pass