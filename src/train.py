"""
Combined Training Module for SFT and GRPO Pipeline.

This module provides a unified Train class that executes a complete 
post-pretraining pipeline: SFT → merge adapters → GRPO alignment → save model.

Pipeline:
1. Load base model and tokenizer
2. Train with SFTTrainer using LoRA
3. Merge SFT adapters into base model
4. Apply new LoRA for GRPO
5. Train with GRPOTrainer using reward model
6. Save final aligned model
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, GRPOTrainer, GRPOConfig, SFTConfig

from config import (
    ModelConfig, 
    SFTTrainingConfig, 
    GRPOTrainingConfig, 
    SFTLoRAConfig, 
    GRPOLoRAConfig, 
    DataConfig
)
from config.model import SpecialTokens
from data.data_sft import DataSFT
from data.data_grpo import DataGRPO


class Train:
    """
    Combined training class for SFT and GRPO pipeline.
    
    Executes a complete post-pretraining pipeline:
    SFT training → merge adapters → GRPO alignment → save final model.
    
    Args:
        model_config: Model paths and identifiers (ModelConfig)
        sft_training_config: SFT training hyperparameters (SFTTrainingConfig)
        grpo_training_config: GRPO training hyperparameters (GRPOTrainingConfig)
        sft_lora_config: LoRA config for SFT stage (SFTLoRAConfig)
        grpo_lora_config: LoRA config for GRPO stage (GRPOLoRAConfig)
        data_config: Data processing settings (DataConfig)
        device: Device to use for training (default: "cuda")
    """

    def __init__(
        self,
        model_config: ModelConfig = None,
        sft_training_config: SFTTrainingConfig = None,
        grpo_training_config: GRPOTrainingConfig = None,
        sft_lora_config: SFTLoRAConfig = None,
        grpo_lora_config: GRPOLoRAConfig = None,
        data_config: DataConfig = None,
        device: str = "cuda"
    ):
        # Use defaults if not provided
        self.model_config = model_config or ModelConfig()
        self.sft_training_config = sft_training_config or SFTTrainingConfig()
        self.grpo_training_config = grpo_training_config or GRPOTrainingConfig()
        self.sft_lora_config = sft_lora_config or SFTLoRAConfig()
        self.grpo_lora_config = grpo_lora_config or GRPOLoRAConfig()
        self.data_config = data_config or DataConfig()
        self.device = device

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.reward_model = None
        
        # Load base model and tokenizer
        self._load_base_model()

    def _load_base_model(self):
        """Load base model and tokenizer with special tokens."""
        print(f"Loading base model: {self.model_config.model_base}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_base,
            trust_remote_code=True
        )
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_base)
        
        # Add special tokens
        special_tokens = SpecialTokens.get_special_tokens()
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Resize embeddings to match tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Model loaded with {len(self.tokenizer)} tokens")

    def _get_sft_data(self):
        """Get SFT train and validation datasets."""
        print("Loading SFT datasets...")
        data_sft = DataSFT(self.data_config, self.model_config)
        train_ds, val_ds = data_sft.get_train_dataset()
        print(f"SFT data: {len(train_ds)} train, {len(val_ds)} val samples")
        return train_ds, val_ds

    def _get_grpo_data(self):
        """Get GRPO train and validation datasets."""
        print("Loading GRPO datasets...")
        data_grpo = DataGRPO(self.data_config, self.model_config)
        train_ds, val_ds = data_grpo.get_train_dataset()
        print(f"GRPO data: {len(train_ds)} train, {len(val_ds)} val samples")
        return train_ds, val_ds

    def _train_sft(self):
        """Train model using SFTTrainer with LoRA."""
        print("\n" + "="*50)
        print("STAGE 1: Supervised Fine-Tuning (SFT)")
        print("="*50)

        train_ds, val_ds = self._get_sft_data()

        # Create LoRA config for SFT
        lora_config = LoraConfig(**self.sft_lora_config.to_dict())

        # Create training arguments
        training_args = TrainingArguments(**self.sft_training_config.to_dict())

        # Initialize SFT Trainer
        sft_trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=lora_config,
            tokenizer=self.tokenizer,
        )

        # Train
        print("Starting SFT training...")
        sft_trainer.train()
        
        # Save SFT checkpoint
        sft_output_dir = self.sft_training_config.output_dir
        sft_trainer.save_model(sft_output_dir)
        print(f"SFT model saved to {sft_output_dir}")

        # Update model reference to the trained model
        self.model = sft_trainer.model
        
        return sft_trainer

    def _merge_sft_adapters(self):
        """Merge SFT LoRA adapters into base model."""
        print("\n" + "="*50)
        print("STAGE 2: Merging SFT Adapters")
        print("="*50)

        if isinstance(self.model, PeftModel):
            print("Merging LoRA adapters into base model...")
            self.model = self.model.merge_and_unload()
            print("Adapters merged successfully")
        else:
            print("Model is not a PeftModel, skipping merge")

    def _load_reward_model(self):
        """Load reward model for GRPO training."""
        print(f"Loading reward model: {self.model_config.rm_model}")
        
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.rm_model,
            trust_remote_code=True
        )
        self.reward_model.to(self.device)
        self.reward_model.eval()
        
        print("Reward model loaded")

    def _reward_fn(self, prompts, completions, **kwargs):
        """
        Reward function using the reward model.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            
        Returns:
            List of reward scores
        """
        combined_texts = [p + c for p, c in zip(prompts, completions)]
        inputs = self.tokenizer(
            combined_texts, 
            padding=True, 
            truncation=True,
            max_length=self.data_config.max_length, 
            return_tensors="pt"
        ).to(self.reward_model.device)

        with torch.no_grad():
            scores = self.reward_model(**inputs).logits.squeeze(-1)

        # Handle both single and batch cases
        if scores.dim() == 0:
            return [scores.item()]
        return scores.tolist()

    def _train_grpo(self):
        """Train model using GRPOTrainer with new LoRA."""
        print("\n" + "="*50)
        print("STAGE 3: GRPO Alignment Training")
        print("="*50)

        # Load reward model
        self._load_reward_model()

        train_ds, val_ds = self._get_grpo_data()

        # Create new LoRA config for GRPO
        grpo_lora_config = LoraConfig(**self.grpo_lora_config.to_dict())

        # Create GRPO config
        grpo_config = GRPOConfig(**self.grpo_training_config.to_dict())

        # Initialize GRPO Trainer
        grpo_trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[self._reward_fn],
            args=grpo_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=grpo_lora_config,
        )

        # Train
        print("Starting GRPO training...")
        grpo_trainer.train()

        # Update model reference
        self.model = grpo_trainer.model

        return grpo_trainer

    def run(self):
        """
        Execute the complete training pipeline.
        
        Pipeline: SFT → Merge Adapters → GRPO → Save
        """
        print("\n" + "#"*60)
        print("# STARTING POST-PRETRAINING PIPELINE")
        print("#"*60)

        # Stage 1: SFT Training
        self._train_sft()

        # Stage 2: Merge SFT adapters
        self._merge_sft_adapters()

        # Stage 3: GRPO Training
        self._train_grpo()

        # Stage 4: Save final model
        self.save_model()

        print("\n" + "#"*60)
        print("# PIPELINE COMPLETE")
        print("#"*60)

    def save_model(self, output_dir: str = None):
        """
        Save the final trained model.
        
        Args:
            output_dir: Directory to save model (defaults to grpo_output_dir)
        """
        output_dir = output_dir or self.grpo_training_config.output_dir
        
        print(f"\nSaving final model to {output_dir}...")
        
        if isinstance(self.model, PeftModel):
            # Save PEFT model
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    def load_checkpoint(self, checkpoint_path: str, merge: bool = False):
        """
        Load model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            merge: Whether to merge adapters after loading
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        
        if merge:
            self.model = self.model.merge_and_unload()
            print("Adapters merged")
        
        print("Checkpoint loaded")


# if __name__ == "__main__":
#     # Example usage
#     trainer = Train()
#     trainer.run()