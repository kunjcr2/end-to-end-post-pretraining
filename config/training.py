"""
Training configuration for SFT and GRPO stages.

This file exists to define all training hyperparameters including learning rate,
batch sizes, schedulers, and optimization settings for each training stage.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTTrainingConfig:
    """Training arguments for Supervised Fine-Tuning stage."""
    
    # Output
    output_dir: str = "./sft"
    run_name: str = "stablelm-1.5b-sft-final"
    
    # Training duration
    num_train_epochs: int = 2
    
    # Batch sizes
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Learning rate and scheduler
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimizer
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    
    # Precision and memory
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 50
    eval_accumulation_steps: int = 8
    prediction_loss_only: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    
    # Tracking
    report_to: str = "wandb"
    
    # Misc
    remove_unused_columns: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for TrainingArguments."""
        return {
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optim": self.optim,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "weight_decay": self.weight_decay,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "eval_accumulation_steps": self.eval_accumulation_steps,
            "prediction_loss_only": self.prediction_loss_only,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "report_to": self.report_to,
            "remove_unused_columns": self.remove_unused_columns,
        }


@dataclass
class GRPOTrainingConfig:
    """Training arguments for GRPO alignment stage."""
    
    # Output
    output_dir: str = "./grpo"
    
    # Training duration
    num_train_epochs: int = 2
    max_steps: int = 200
    
    # GRPO specific
    num_generations: int = 8
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 16
    
    # Learning rate and scheduler
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    
    # Optimizer
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    
    # Precision and memory
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: int = 3
    
    # Logging and saving
    logging_steps: int = 1
    save_strategy: str = "epoch"
    
    # Tracking
    report_to: str = "wandb"
    
    # Generation parameters
    temperature: float = 1.0
    top_k: int = 50
    
    def to_dict(self) -> dict:
        """Convert to dictionary for GRPOConfig."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "num_generations": self.num_generations,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optim": self.optim,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "weight_decay": self.weight_decay,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "report_to": self.report_to,
            "temperature": self.temperature,
            "top_k": self.top_k,
        }
