"""
Data processing utilities for training datasets.

This file exists to handle all data loading, preprocessing, and formatting
for GRPO training stages. It converts raw datasets from
HuggingFace into the format expected by the training scripts.

You will implement:
- process_pku_to_grpo: Convert PKU-SafeRLHF to GRPO format
- Tokenization and dataset creation utilities
- Train/validation splitting
"""

##### DONE #####

# TODO: Implement data processing functions
# Reference: ipynb/alignment.py (process_pku_to_grpo)

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from src.tokenizer import Tokenizer

class DataGRPO:
    
    """
    DataGRPO class for processing and loading GRPO datasets.
    """

    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        
        self.tokenizer = Tokenizer(model_config).get_tokenizer()
        self.dataset = load_dataset(self.data_config.ds_grpo, split="train")

    def process_pku_to_grpo(self):
        """
        Convert pku alignment to single-turn grpo format
        """
        processed = []
        output_file = self.data_config.grpo_output_file
        split = self.data_config.grpo_train_split
        max_samples = self.data_config.grpo_max_samples

        for sample in dataset["train"]:
            prompt = sample['prompt'] 
            response_0 = sample['response_0']
            response_1 = sample['response_1']

            processed.append({
                'prompt': f"<|user|>\n{prompt}\n<|assistant|>\n",
                'solution': f"{response_0}<|endoftext|>"
            })
            processed.append({
                'prompt': f"<|user|>\n{prompt}\n<|assistant|>\n",
                'solution': f"{response_1}<|endoftext|>"
            })

            if len(processed) >= max_samples:
                break

        # Save
        with open(output_file, 'w') as f:
            for item in processed:
                f.write(json.dumps(item) + '\n')

        print(f"Processed {len(processed)} samples to {output_file}.")

        df = pd.read_json(output_file, lines=True)
        df.drop(columns=["solution"], inplace=True, axis=1)

        train_df = df[:int(len(df) * 0.95)]
        val_df = df[int(len(df) * 0.95):]

        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)

        return train_ds, val_ds

    def _load_from_file(self, output_file):
        """
        Load already processed data from file and return train/val split.
        """
        print(f"Loading cached data from {output_file}...")
        
        split = self.data_config.grpo_train_split
    
        df = pd.read_json(output_file, lines=True)
        ds = self.tokenize_grpo(Dataset.from_pandas(df))
        ds = ds.remove_columns(["text"])
        
        train_ds = ds.select(range(int(len(ds) * split))).shuffle(seed=42)
        val_ds = ds.select(range(int(len(ds) * split), len(ds))).shuffle(seed=42)
        
        print(f"Loaded {len(ds)} samples from cache.")
        return train_ds, val_ds

    def get_train_dataset(self):
        """
        Get training dataset - loads from cache if output file exists, otherwise processes.
        """
        output_file = self.data_config.grpo_output_file
        
        if Path(output_file).exists():
            return self._load_from_file(output_file)
        else:
            print(f"Output file not found. Processing dataset...")
            return self.process_pku_to_grpo()