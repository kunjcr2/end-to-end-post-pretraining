# End-to-End LLM Post-Pretraining Pipeline

A complete pipeline for LLM post-pretraining covering **Supervised Fine-Tuning (SFT)** and **GRPO Alignment**, built on StableLM 1.6B.

> **Note**: The model has been trained and the weights are hosted on HuggingFace. This repo now serves as the **inference API** and a **reference** for the training process.

---

## Model Weights

**HuggingFace**: [kunjcr2/stablelm-1.6b-finetuned-aligned](https://huggingface.co/kunjcr2/stablelm-1.6b-finetuned-aligned)

---

## Training Pipeline (Completed)

The model was post-pretrained in two stages:

```
+-------------------+     +-------------------+     +-------------------+
|   Base Model      |---->|   SFT Model       |---->|  Aligned Model    |
|  (StableLM 1.6B)  |     |   (LoRA + SFT)    |     |    (GRPO)         |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
   Pre-trained LLM        Instruction-tuned         Safety-aligned
                           on UltraChat            with Reward Model
```

### Stage 1: Supervised Fine-Tuning (SFT)
- Fine-tuned the base model on instruction-following data
- Used **LoRA** (r=256) for parameter-efficient training
- Dataset: UltraChat 200K conversations
- Reference: [`ipynb/finetune.py`](ipynb/finetune.py)

### Stage 2: GRPO Alignment
- Aligned the SFT model using Group Relative Policy Optimization
- Used a **reward model** to score responses
- Dataset: PKU-SafeRLHF for safety alignment
- Reference: [`ipynb/alignment.py`](ipynb/alignment.py)

### Tech Stack Used for Training

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

## Inference API

The repo includes a **FastAPI + vLLM** inference server that loads the model from HuggingFace.

### Project Structure

```
end-to-end-post-pretraining/
├── api/                        # Inference API
│   ├── app.py                  # FastAPI + vLLM server
│   └── schema.py               # Request/response models
├── config/                     # Configuration classes
│   ├── __init__.py              # Package init
│   ├── model.py                # Model paths and checkpoints
│   ├── lora.py                 # LoRA hyperparameters
│   ├── training.py             # Training arguments
│   └── data.py                 # Dataset configs
├── ipynb/                      # Training reference scripts
│   ├── finetune.py             # SFT training reference
│   └── alignment.py            # GRPO alignment reference
├── backend_demo/               # Standalone CRUD demo (see below)
│   ├── app/
│   │   ├── main_orm.py         # FastAPI routes (SQLAlchemy ORM)
│   │   └── main_psycopg2.py    # FastAPI routes (raw psycopg2)
│   ├── database/
│   │   ├── database.py         # Engine, ORM models, CRUD functions
│   │   └── schema.py           # Pydantic V2 request/response schemas
│   ├── utils/
│   │   └── hash.py             # Password hashing utility
│   ├── requirements.txt
│   └── README.md
├── docker/                     # Containerization
│   └── Dockerfile              # FastAPI + vLLM setup
├── data/                       # Data directory
├── requirements.txt
├── .env.example
└── LICENSE
```

### Quick Start

```bash
git clone https://github.com/kunjcr2/end-to-end-post-pretraining.git
cd end-to-end-post-pretraining
pip install -r requirements.txt
```

```bash
# Set your HuggingFace model ID (optional — defaults to the hosted weights)
cp .env.example .env

# Run the inference server
uvicorn api.app:app --reload --port 8000
```

### API Endpoints

**Health Check**
```
GET /health
```

**Generate a Response**
```
POST /generate
Content-Type: application/json

{
    "query": "What are some common misconceptions about AI?",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
}
```

### Docker

```bash
docker build -t stablelm-api -f docker/Dockerfile .
docker run -p 8000:8000 --gpus all stablelm-api
```

---

## Backend Demo

The `backend_demo/` directory contains a standalone **FastAPI + PostgreSQL** CRUD application for **posts** and **users**. It includes two implementations side-by-side:

- **`main_orm.py`** — SQLAlchemy ORM approach (recommended)
- **`main_psycopg2.py`** — raw SQL via psycopg2

### Highlights
- SQLAlchemy ORM with Pydantic V2 (`Post` & `User` models)
- Context-manager based session handling
- Full CRUD operations for posts (Create, Read, Update, Delete)
- User registration with **EmailStr** validation and duplicate-email guard
- Password hashing via `utils/hash.py`

See [`backend_demo/README.md`](backend_demo/README.md) for full endpoint documentation.

**Run the demo:**
```bash
# Requires a .env file in backend_demo/ with PostgreSQL credentials
uvicorn backend_demo.app.main_orm:app --reload
```

---

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
