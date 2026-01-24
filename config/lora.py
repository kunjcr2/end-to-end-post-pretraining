"""
LoRA configuration for parameter-efficient fine-tuning.

This file exists to define LoRA hyperparameters for both SFT and GRPO stages.
Different stages use different rank values to balance efficiency and capacity.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SFTLoRAConfig:
    """LoRA configuration for Supervised Fine-Tuning stage.
    
    Uses higher rank (256) for more capacity during instruction tuning.
    """
    r: int = 256
    lora_alpha: int = 512
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for PEFT LoraConfig."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class GRPOLoRAConfig:
    """LoRA configuration for GRPO alignment stage.
    
    Uses lower rank (32) as alignment is a refinement step.
    """
    r: int = 32
    lora_alpha: int = 64
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for PEFT LoraConfig."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }
