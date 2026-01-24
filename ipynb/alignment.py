## Model Alignment
#
# NOTE: This is a REFERENCE notebook file - not for production use.
# For production code, see src/align.py
# Configurations are imported from config/ module.
#

### Import and config

# !pip install trl -q

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
import pandas as pd

import os
import sys
sys.path.insert(0, '..')  # Add parent directory for imports

from config import ModelConfig, GRPOLoRAConfig, GRPOTrainingConfig, DataConfig

# Initialize configs
model_cfg = ModelConfig()
lora_cfg = GRPOLoRAConfig()
training_cfg = GRPOTrainingConfig()
data_cfg = DataConfig()

# Legacy dict format for compatibility (references config classes)
MODEL_CONFIG = {
    "model_base": model_cfg.model_base,
    "rm_model": model_cfg.rm_model,
    "ds_grpo": model_cfg.ds_grpo,
    "output_file": data_cfg.grpo_output_file
}

# GRPO training config from config/training.py
GRPO_CONFIG = training_cfg.to_dict()

### Model prep

from copy import deepcopy # Added import for deepcopy

def prepare():
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_base"], trust_remote_code=True
    ).cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_base"])

    # Add special tokens FIRST to the tokenizer
    tok.add_special_tokens({
        "eos_token": "<|endoftext|>",
        "additional_special_tokens": ["<|user|>", "<|assistant|>"]
    })
    tok.pad_token = tok.eos_token
    # Resize base model embeddings to match the new tokenizer size, BEFORE loading PEFT
    base_model.resize_token_embeddings(len(tok))

    # 1. Load and merge SFT
    checkpoint_path = os.path.join("/content/drive/MyDrive", "checkpoint-2079")
    model_sft = PeftModel.from_pretrained(base_model, checkpoint_path)
    model_sft = model_sft.merge_and_unload()

    # reward model
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIG["rm_model"], trust_remote_code=True
    ).cuda()

    return model_sft, rm_model, tok

model_sft, rm_model, tok = prepare()

### Data and model prep

import json

def process_pku_to_grpo(tokenizer, output_file=MODEL_CONFIG["output_file"], max_samples=200):
    """
    Convert UltraChat multi-turn to single-turn SFT format
    """
    dataset = load_dataset(MODEL_CONFIG["ds_grpo"])
    processed = []

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

# Run it
process_pku_to_grpo(max_samples=4000, tokenizer=tok)

df = pd.read_json(MODEL_CONFIG["output_file"], lines=True)
# df["input_ids"] = df["prompt"].apply(lambda x: tok(x, padding=True)["input_ids"])
# df["attention_mask"] = df["prompt"].apply(lambda x: tok(x, padding=True)["attention_mask"])
df.drop(columns=["solution"], inplace=True, axis=1)

df.head()

train_df = df[:int(len(df) * 0.95)]
val_df = df[int(len(df) * 0.95):]

train_grpo = Dataset.from_pandas(train_df)
val_grpo = Dataset.from_pandas(val_df)

import torch

def rm_reward(prompts, completions, **kwargs):
    combined_texts = [p + c for p, c in zip(prompts, completions)]
    inputs = tok(combined_texts, padding=True, truncation=True,
                 max_length=512, return_tensors="pt").to(rm_model.device)

    with torch.no_grad():
        scores = rm_model(**inputs).logits.squeeze(-1)

    # Handle both single and batch cases
    if scores.dim() == 0:
        return [scores.item()]
    return scores.tolist()

### Training

grpo_trainer = GRPOTrainer(
    model=model_sft,
    reward_funcs=[rm_reward],
    args=GRPOConfig(**GRPO_CONFIG),
    train_dataset=train_grpo,
    eval_dataset=val_grpo,
    peft_config=LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout= 0.05,
        bias= "none",
        task_type= "CAUSAL_LM"
    )
)

grpo_trainer.train()

### Inference

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """
    Generates a response from the given model and tokenizer for a specific prompt.
    """
    # Format the prompt according to the model's expected chat format
    inputs = tokenizer(
        f"<|user|>{prompt}<|endoftext|>", # Adjusted prompt format to be more aligned with training
        return_tensors="pt",
        return_attention_mask=False
    ).to(model.device)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, # Enable sampling for more varied responses
        temperature=0.7, # Control randomness
        top_k=50, # Limit the vocabulary for sampling
        top_p=0.95 # Nucleus sampling
    )

    # Decode and return the generated text
    text = tokenizer.batch_decode(outputs)[0]

    # Extract only the assistant's response
    assistant_start = text.find("<|assistant|>")
    if assistant_start != -1:
        text = text[assistant_start + len("<|assistant|>"):].strip()
    # Remove the eos_token if present
    if text.endswith(tokenizer.eos_token):
        text = text[:-len(tokenizer.eos_token)].strip()

    return text

import os
from peft import PeftModel

print("Loading base model for GRPO checkpoint inference...")
base_model_for_grpo_inference = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["model_base"], trust_remote_code=True).cuda()
base_model_for_grpo_inference.resize_token_embeddings(len(tok))

# Define the path to the GRPO checkpoint (latest epoch saved by GRPO trainer)
grpo_checkpoint_path = os.path.join(GRPO_CONFIG["output_dir"], "checkpoint-44") # Assuming 2 epochs were run and last checkpoint is checkpoint-2

print(f"Loading PEFT adapter for GRPO from: {grpo_checkpoint_path}")
# Load the PEFT adapter onto the base model
model_grpo_inference = PeftModel.from_pretrained(base_model_for_grpo_inference, grpo_checkpoint_path)

# Set the model to evaluation mode
model_grpo_inference.eval()
print("GRPO model loaded successfully for inference.")

sample_prompt_grpo = "<|user|>\nWhat are some common misconceptions about artificial intelligence?\n<|assistant|>\n"

print("\n--- GRPO Model Inference ---")
grpo_response = generate_response(model_grpo_inference, tok, sample_prompt_grpo)
print(grpo_response.replace("\n", "").replace(sample_prompt_grpo, ""))

