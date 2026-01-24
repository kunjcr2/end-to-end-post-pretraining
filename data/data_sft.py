"""
Data processing utilities for training datasets.

This file exists to handle all data loading, preprocessing, and formatting
for both SFT and GRPO training stages. It converts raw datasets from
HuggingFace into the format expected by the training scripts.

You will implement:
- process_ultrachat_to_sft: Convert UltraChat to single-turn SFT format
- process_pku_to_grpo: Convert PKU-SafeRLHF to GRPO format
- Tokenization and dataset creation utilities
- Train/validation splitting
"""

# TODO: Implement data processing functions
# Reference: ipynb/finetune.py (process_ultrachat_to_sft)

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

class DataSFT:
    
    """
    DataSFT class for processing and loading SFT datasets.
    """

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_base)
        self.dataset = load_dataset(self.data_config.ds_sft, split="train")

    def process_ultrachat_to_sft(self):
        """
        Convert UltraChat to single-turn SFT format
        """
        processed = []
        output_file = self.data_config.sft_output_file
        split = self.data_config.sft_train_split
        max_samples = self.data_config.sft_max_samples

        for sample in self.dataset:
            messages = sample['messages']
            local_history = ""

            for i in range(0, len(messages)-1, 2):
                if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':

                    # Build prompt with history
                    user_content = messages[i]['content']
                    assistant_content = messages[i+1]['content']

                    full_prompt = f"<|user|>\n{user_content}"

                    processed.append({
                        'text': f"{full_prompt}\n<|assistant|>\n{assistant_content}<|endoftext|>",
                    })

                    # Update local history for this conversation
                    local_history += f"\n<|user|>\n{user_content}\n<|assistant|>\n{assistant_content}"

                    if len(processed) >= max_samples:
                        break

            if len(processed) >= max_samples:
                break

        # Save
        with open(output_file, 'w') as f:
            for item in processed:
                f.write(json.dumps(item) + '\n')

        print(f"Processed {len(processed)} samples to {output_file}.")

        df = pd.read_json(output_file, lines=True)
        ds = self.tokenize_sft(Dataset.from_pandas(df))
        ds = ds.drop(columns=["text"])

        train_ds = ds[:int(len(ds) * split)].shuffle()
        val_ds = ds[int(len(ds) * split):].shuffle()

        return train_ds, val_ds

    def tokenize_sft(self, dataset):
        """
        Tokenize dataset for SFT training.
        """
        dataset = dataset.map(
            lambda x: self.tokenizer(x["text"], 
                padding=self.data_config.padding, 
                truncation=self.data_config.truncation, 
                max_length=self.data_config.max_length
            ),
            num_proc=4
        )
        return dataset

    def _load_from_file(self, output_file):
        """
        Load already processed data from file and return train/val split.
        """
        print(f"Loading cached data from {output_file}...")
        
        split = self.data_config.sft_train_split
    
        df = pd.read_json(output_file, lines=True)
        ds = self.tokenize_sft(Dataset.from_pandas(df))
        ds = ds.remove_columns(["text"])
        
        train_ds = ds.select(range(int(len(ds) * split))).shuffle(seed=42)
        val_ds = ds.select(range(int(len(ds) * split), len(ds))).shuffle(seed=42)
        
        print(f"Loaded {len(ds)} samples from cache.")
        return train_ds, val_ds

    def get_train_dataset(self):
        """
        Get training dataset - loads from cache if output file exists, otherwise processes.
        """
        output_file = self.data_config.sft_output_file
        
        if Path(output_file).exists():
            return self._load_from_file(output_file)
        else:
            print(f"Output file not found. Processing dataset...")
            return self.process_ultrachat_to_sft()