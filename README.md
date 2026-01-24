# End-to-End LLM Post-Pretraining Pipeline

A complete, production-style pipeline for LLM post-pretraining covering **Supervised Fine-Tuning (SFT)** and **GRPO Alignment**. This is a learning/portfolio project demonstrating the full workflow from data preprocessing to model alignment.

> **Note**: This is a learning project. The trained models are not production-grade but serve to demonstrate the complete pipeline.

---

## 🔗 Model Checkpoints

**HuggingFace**: *To be added*

---

## 🏗️ Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Base Model    │────▶│   SFT Model     │────▶│  Aligned Model  │
│  (StableLM 1.6B)│     │   (LoRA + SFT)  │     │    (GRPO)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Pre-trained LLM      Instruction-tuned       Safety-aligned
                         on UltraChat           with Reward Model
```

### Stage 1: Supervised Fine-Tuning (SFT)
- Fine-tune the base model on instruction-following data
- Uses **LoRA** (r=256) for parameter-efficient training
- Dataset: UltraChat 200K conversations

### Stage 2: GRPO Alignment
- Align the SFT model using Group Relative Policy Optimization
- Uses a **reward model** to score responses
- Dataset: PKU-SafeRLHF for safety alignment

---

## 🛠️ Tech Stack

| Component | Details |
|-----------|---------|
| **Base Model** | `stabilityai/stablelm-2-1_6b` |
| **Reward Model** | `OpenAssistant/reward-model-deberta-v3-large` |
| **SFT Dataset** | `HuggingFaceH4/ultrachat_200k` |
| **Alignment Dataset** | `PKU-Alignment/PKU-SafeRLHF` |
| **Techniques** | LoRA, PEFT, TRL, GRPO |
| **Hardware** | NVIDIA A100 GPU |
| **Frameworks** | Transformers, TRL, PEFT, PyTorch |

---

## 📁 Project Structure

```
end-to-end-post-pretraining/
├── config/                 # Configuration classes
│   ├── __init__.py
│   ├── model.py           # Model paths and checkpoints
│   ├── lora.py            # LoRA hyperparameters
│   ├── training.py        # Training arguments
│   └── data.py            # Dataset configs
├── src/                   # Core source code
│   ├── __init__.py
│   ├── train.py           # SFT training logic
│   ├── align.py           # GRPO alignment logic
│   ├── inference.py       # Model inference utilities
│   └── data_processing.py # Data loading and preprocessing
├── scripts/               # CLI scripts
│   ├── run_sft.py         # Launch SFT training
│   └── run_grpo.py        # Launch GRPO alignment
├── docker/                # Containerization
│   └── Dockerfile         # Flask API + vLLM setup
├── data/                  # Processed datasets (gitignored)
├── ipynb/                 # Reference notebooks
│   ├── finetune.py        # SFT training reference
│   └── alignment.py       # GRPO training reference
├── requirements.txt
└── .env.example
```

---

## 📊 Sample Outputs

*Sample outputs will be added soon.*

| Prompt | Base Model | SFT Model | Aligned Model |
|--------|------------|-----------|---------------|
| *TBD* | *TBD* | *TBD* | *TBD* |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/end-to-end-post-pretraining.git
cd end-to-end-post-pretraining
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and update paths:

```bash
cp .env.example .env
```

### Training

```bash
# Stage 1: Supervised Fine-Tuning
python scripts/run_sft.py

# Stage 2: GRPO Alignment
python scripts/run_grpo.py
```

### Inference

```python
from src.inference import generate_response

response = generate_response(
    prompt="What are some common misconceptions about AI?",
    model_type="aligned"  # or "base", "sft"
)
print(response)
```

---

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
