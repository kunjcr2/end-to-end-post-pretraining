## Fine-tuning
#
# NOTE: This is a REFERENCE notebook file - not for production use.
# For production code, see src/train.py
# Configurations are imported from config/ module.
#

### Configuration
import sys
sys.path.insert(0, '..')  # Add parent directory for imports

from config import ModelConfig, SFTLoRAConfig, SFTTrainingConfig, DataConfig

# Initialize configs
model_cfg = ModelConfig()
lora_cfg = SFTLoRAConfig()
training_cfg = SFTTrainingConfig()
data_cfg = DataConfig()

# Legacy dict format for compatibility (references config classes)
MODEL = {
    "model_base": model_cfg.model_base,
    "model_sft": "stablelm_sft.safetensor",
    "model_grpo": "stablelm_grpo.safetensor",
    "model_rm": model_cfg.rm_model,
    "ds_sft": model_cfg.ds_sft,
    "ds_processed": data_cfg.sft_output_file,
    "ds_grpo": model_cfg.ds_grpo
}

# LoRA config from config/lora.py
LORA_CONFIG = lora_cfg.to_dict()

# Training args from config/training.py
TRAINING_ARGS = training_cfg.to_dict()

# Data config from config/data.py
DATA_CONFIG = {
    "total_data": data_cfg.sft_max_samples
}

### Imports and model download

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model

import pandas as pd

model_base = AutoModelForCausalLM.from_pretrained(MODEL["model_base"], trust_remote_code=True).cuda()
tok = AutoTokenizer.from_pretrained(MODEL["model_base"])

special_tokens = {
    "eos_token": "<|endoftext|>",
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
}

tok.add_special_tokens(special_tokens)
tok.pad_token = tok.eos_token
model_base.resize_token_embeddings(len(tok))

### Data preprocessing

# preprocess_ultrachat.py
from datasets import load_dataset
import json

def process_ultrachat_to_sft(tokenizer, output_file=MODEL['ds_processed'], max_samples=200):
    """
    Convert UltraChat multi-turn to single-turn SFT format
    """
    dataset = load_dataset(MODEL["ds_sft"], split="train_sft")

    processed = []
    conversation_history = ""

    for sample in dataset:
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

# Run it
process_ultrachat_to_sft(max_samples=140000, tokenizer=tok)

ds = pd.read_json(MODEL["ds_processed"], lines=True)
ds["input_ids"] = ds["text"].apply(lambda x: tok(x, padding=True)["input_ids"])
ds["attention_mask"] = ds["text"].apply(lambda x: tok(x, padding=True)["attention_mask"])
ds = ds.drop(columns=["text"])

ds["input_ids"] = ds["input_ids"].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
ds["attention_mask"] = ds["attention_mask"].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
train_sft = Dataset.from_pandas(ds[:int(len(ds) * 0.95)])
test_sft = Dataset.from_pandas(ds[int(len(ds) * 0.95):])

### Training

lora_config = LoraConfig(
    **LORA_CONFIG
)

model_sft = get_peft_model(model_base, lora_config)

train_args = TrainingArguments(**TRAINING_ARGS)
data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")

trainer = Trainer(
    model=model_sft,
    args=train_args,
    train_dataset=train_sft,
    eval_dataset=test_sft,
    data_collator=data_collator
)

model_sft.enable_input_require_grads()
model_sft.gradient_checkpointing_enable()
output = trainer.train()

### Infernce

import os
from peft import PeftModel

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

# The base model and tokenizer (`model_base` and `tok`) are already loaded.
# We need to re-instantiate the base model to ensure it's clean for loading the adapter.

print("Loading base model for SFT checkpoint...")
base_model_for_sft = AutoModelForCausalLM.from_pretrained(MODEL["model_base"], trust_remote_code=True).cuda()
base_model_for_sft.resize_token_embeddings(len(tok))

# Define the path to the specific checkpoint
checkpoint_dir = TRAINING_ARGS["output_dir"]
checkpoint_path = os.path.join("/content/drive/MyDrive", "checkpoint-2079")

print(f"Loading PEFT adapter from: {checkpoint_path}")
# Load the PEFT adapter onto the base model
model_sft_inference = PeftModel.from_pretrained(base_model_for_sft, checkpoint_path)

# Set the model to evaluation mode
model_sft_inference.eval()
print("SFT model from checkpoint 2079 loaded successfully for inference.")

sample_prompt = "What are some common misconceptions about artificial intelligence?"

# print("\n--- Base Model Inference ---")
# # Assuming `model_base` is still the original, unfine-tuned model
# base_response = generate_response(model_base, tok, sample_prompt)
# print(base_response.replace("\n", "").replace(sample_prompt, ""))

print("\n--- Fine-tuned SFT Model Inference (from checkpoint 2079) ---")
sft_response = generate_response(model_sft_inference, tok, sample_prompt)
print(sft_response.replace("\n", "").replace(sample_prompt, ""))