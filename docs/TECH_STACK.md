# Technical Stack & Dependencies

**Document Version:** 1.0
**Last Updated:** 2025-10-21
**Status:** Active - Implementation Guide

---

## Overview

This document specifies the complete technology stack for the Bangla Continual Learning research project, organized by experimental stage with pinned dependency versions for reproducibility.

**Key Principles:**
- ✅ Reproducibility first: All versions pinned
- ✅ Colab-friendly: Focus on cloud-native tools
- ✅ Open-source only: No proprietary dependencies
- ✅ Modular: Each stage can run independently with core deps only

---

## Core Foundation Stack (All Stages)

### Python & Package Management

```
Python:              3.9 - 3.11 (tested on 3.10)
pip:                 23.0+
conda:               optional (for environment isolation)
```

**Rationale:** Python 3.10 is stable, widely supported, and compatible with all ML libraries. Avoid 3.12+ due to library lag.

---

### Deep Learning Framework

```yaml
PyTorch:
  version: "2.0.1"
  install: "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==0.15.2"
  cuda_version: "11.8"  # or "cu118" in PyTorch naming
  colab_compat: ✅ (included by default)
  rationale: "Stable LTS release; excellent CUDA support; production-ready"

PyTorch Lightning:
  version: "2.0.3"
  install: "pip install pytorch-lightning==2.0.3"
  rationale: "Simplifies training loops; better checkpoint management"
```

---

### Transformer & NLP Stack

```yaml
transformers:
  version: "4.34.0"
  install: "pip install transformers==4.34.0"
  key_models: "IndicBERT, mBERT, XLM-R"
  rationale: "Latest stable with security patches; IndicBERT pre-trained available"

datasets:
  version: "2.14.1"
  install: "pip install datasets==2.14.1"
  rationale: "Easy dataset loading; XNLI-Indic, BLUB benchmarks available"
  usage: |
    from datasets import load_dataset
    xnli_indic = load_dataset("xnli", "hi")  # Example: Hindi XNLI

tokenizers:
  version: "0.14.1"
  install: "pip install tokenizers==0.14.1"
  note: "Auto-installed with transformers; kept for reproducibility"

sentencepiece:
  version: "0.1.99"
  install: "pip install sentencepiece==0.1.99"
  rationale: "IndicBERT uses SentencePiece tokenization"
```

---

### Continual Learning & Adaptation

```yaml
peft:
  version: "0.5.0"
  install: "pip install peft==0.5.0"
  methods_available: ["LoRA", "Prefix Tuning", "Adapters"]
  stage_used: "Stages 2-3 (optional LoRA variant)"
  rationale: "Production-ready PEFT library from HuggingFace; minimal memory overhead"
  example_usage: |
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
```

**Custom Experience Replay:**
- No external dependency; implement as Python class in `src/continual_learning/experience_replay.py`
- Buffer management: dict-based with reservoir sampling for streaming data
- See: Experiment Configuration System (Section 3) for integration

---

### Experiment Tracking & Logging

```yaml
wandb:
  version: "0.15.8"
  install: "pip install wandb==0.15.8"
  setup: |
    wandb login  # Authenticate once
  features:
    - Real-time metric visualization
    - Hyperparameter sweep management
    - Model artifact storage
    - Integration with HuggingFace Trainer
  rationale: "Free tier sufficient; superior UX vs. MLflow; Colab-native"
  colab_usage: |
    import wandb
    wandb.init(project="bangla-cl", config={"lr": 2e-5, "epochs": 3})

tensorboard:
  version: "2.14.0"
  install: "pip install tensorboard==2.14.0"
  rationale: "Backup visualization; auto-logged by PyTorch Lightning"

python-dotenv:
  version: "1.0.0"
  install: "pip install python-dotenv==1.0.0"
  purpose: "Load .env for API keys (W&B, HF tokens)"
```

---

### Data Processing & Scientific Computing

```yaml
pandas:
  version: "2.0.3"
  install: "pip install pandas==2.0.3"
  usage: "Results aggregation, CSV/JSON handling"

numpy:
  version: "1.24.3"
  install: "pip install numpy==1.24.3"
  rationale: "Stable, compatible with scipy/sklearn"

scipy:
  version: "1.11.3"
  install: "pip install scipy==1.11.3"
  stage_used: "Stages 3-4"
  functions_used: |
    - scipy.stats.f_oneway (ANOVA for curriculum comparison)
    - scipy.stats.pearsonr, spearmanr (correlation analysis)
    - scipy.stats.t_test_ind (significance testing)

scikit-learn:
  version: "1.3.2"
  install: "pip install scikit-learn==1.3.2"
  stage_used: "Stages 3-4"
  usage: |
    - Cosine similarity for linguistic distance metrics
    - Preprocessing utilities
```

---

### Visualization Stack

```yaml
matplotlib:
  version: "3.8.0"
  install: "pip install matplotlib==3.8.0"
  usage: "Publication-ready plots; forgetting curves, heatmaps"

seaborn:
  version: "0.13.0"
  install: "pip install seaborn==0.13.0"
  stage_used: "Stages 2-4"
  example: |
    # Forgetting pattern heatmap
    sns.heatmap(forgetting_matrix, cmap="RdYlGn_r", annot=True)

plotly:
  version: "5.17.0"
  install: "pip install plotly==5.17.0"
  usage: "Interactive exploration plots; performance curves"
  rationale: "Better for exploratory analysis; can embed in notebooks"

pillow:
  version: "10.0.1"
  install: "pip install pillow==10.0.1"
  rationale: "Image manipulation for figure composites"
```

---

### Statistical Analysis & Modeling

```yaml
statsmodels:
  version: "0.14.0"
  install: "pip install statsmodels==0.14.0"
  stage_used: "Stages 3-4"
  usage: |
    - OLS regression: Forgetting ~ distance + data_size + task_type
    - Anova_lm for model comparison
    - Post-hoc tests (Tukey HSD)
  rationale: "Comprehensive statistical tools; excellent documentation"

pingouin:
  version: "0.5.3"
  install: "pip install pingouin==0.5.3"
  optional: True
  rationale: "Advanced stats (effect sizes, Bayesian stats); complements statsmodels"
```

---

### Linguistic Analysis (Stage 4 Only)

```yaml
spacy:
  version: "3.7.2"
  install: "pip install spacy==3.7.2"
  models: |
    # Download linguistic models
    python -m spacy download en_core_web_sm  # For reference
    python -m spacy download hi_core_web_sm  # Hindi (if available)
  stage_used: "Stage 4"
  usage: "Morphosyntactic analysis, dependency parsing"

universal-dependencies:
  source: "github.com/UniversalDependencies/UD_*"
  install_note: "Use spacy's UD models or manual UD parsing scripts"
  purpose: "Cross-linguistic morphosyntactic comparison"

nltk:
  version: "3.8.1"
  install: "pip install nltk==3.8.1"
  optional: True
  usage: "Tokenization, POS tagging (backup to spacy)"

regex:
  version: "2023.10.3"
  install: "pip install regex==2023.10.3"
  purpose: "Unicode-aware regex for Bangla/Indic text processing"
```

---

### Testing & Code Quality

```yaml
pytest:
  version: "7.4.2"
  install: "pip install pytest==7.4.2"
  purpose: "Unit tests for data loading, metrics computation"

black:
  version: "23.10.1"
  install: "pip install black==23.10.1"
  purpose: "Code formatting"

flake8:
  version: "6.1.0"
  install: "pip install flake8==6.1.0"
  purpose: "Linting"

mypy:
  version: "1.6.1"
  install: "pip install mypy==1.6.1"
  purpose: "Type checking (optional but recommended)"
```

---

### Development Tools

```yaml
jupyter:
  version: "1.0.0"
  install: "pip install jupyter==1.0.0"
  note: "Pre-installed in Colab"
  usage: "Analysis notebooks, exploratory scripts"

ipython:
  version: "8.17.2"
  install: "pip install ipython==8.17.2"
  rationale: "Enhanced interactive shell"

nbconvert:
  version: "7.14.1"
  install: "pip install nbconvert==7.14.1"
  purpose: "Convert notebooks to scripts/HTML for reproducibility"
```

---

## Stage-Specific Dependencies

### Stage 1: Baseline Infrastructure (Minimal)

```
Core: PyTorch, Transformers, Datasets
Tracking: Weights & Biases
Compute: Google Colab Pro
Visualization: Matplotlib, Seaborn

# Minimal requirements.txt for Stage 1
torch==2.0.1
transformers==4.34.0
datasets==2.14.1
wandb==0.15.8
matplotlib==3.8.0
seaborn==0.13.0
pandas==2.0.3
```

---

### Stage 2: Sequential Continual Learning

```
Adds to Stage 1:
- PyTorch Lightning (for checkpoint management)
- Custom Experience Replay (no external dep)
- Enhanced metrics tracking

# Additional requirements
pytorch-lightning==2.0.3

# Custom code location:
# src/continual_learning/experience_replay.py
# src/training/sequential_trainer.py
```

---

### Stage 3: Curriculum Analysis

```
Adds to Stage 1-2:
- scipy (ANOVA, significance tests)
- statsmodels (regression analysis)
- seaborn (enhanced heatmaps)
- scikit-learn (distance metrics)

# Additional requirements
scipy==1.11.3
scikit-learn==1.3.2
statsmodels==0.14.0
```

---

### Stage 4: Linguistic Feature Analysis

```
Adds to Stages 1-3:
- spaCy (morphosyntactic parsing)
- regex (Unicode text processing)
- Additional statistical tools

# Additional requirements
spacy==3.7.2
regex==2023.10.3
pingouin==0.5.3  # optional
```

---

## Complete requirements.txt

```
# Core ML
torch==2.0.1
torchvision==0.15.2
torchaudio==0.15.2
transformers==4.34.0
datasets==2.14.1
tokenizers==0.14.1
sentencepiece==0.1.99
pytorch-lightning==2.0.3
peft==0.5.0

# Experiment Tracking
wandb==0.15.8
tensorboard==2.14.0
python-dotenv==1.0.0

# Data & Scientific Computing
pandas==2.0.3
numpy==1.24.3
scipy==1.11.3
scikit-learn==1.3.2
statsmodels==0.14.0

# Visualization
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.17.0
pillow==10.0.1

# Linguistic Analysis
spacy==3.7.2
regex==2023.10.3
nltk==3.8.1

# Development & Testing
jupyter==1.0.0
ipython==8.17.2
nbconvert==7.14.1
pytest==7.4.2
black==23.10.1
flake8==6.1.0
mypy==1.6.1
```

---

## Environment Setup Instructions

### Local Machine / Colab Pro

```bash
# 1. Clone repo
git clone <repo-url>
cd BanglaContinualLearning

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download spaCy models (if Stage 4)
python -m spacy download en_core_web_sm

# 6. Set up Weights & Biases
wandb login

# 7. Verify installation
python -c "import torch; import transformers; print('✅ All dependencies installed')"
```

### Colab-Specific Setup

```python
# In first Colab cell:
!pip install -q torch transformers datasets wandb pytorch-lightning
!pip install -q scipy scikit-learn statsmodels
!pip install -q matplotlib seaborn plotly
!pip install -q spacy regex

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!git clone <repo-url> /content/drive/MyDrive/BanglaContinualLearning
%cd /content/drive/MyDrive/BanglaContinualLearning

# Authenticate W&B
import wandb
wandb.login()
```

---

## Version Compatibility Matrix

| Component | Python 3.9 | Python 3.10 | Python 3.11 | Note |
|-----------|-----------|-----------|-----------|------|
| PyTorch 2.0.1 | ✅ | ✅ | ✅ | Recommended |
| Transformers 4.34.0 | ✅ | ✅ | ✅ | - |
| TensorFlow (not used) | - | - | - | PyTorch-only project |
| spaCy 3.7.2 | ✅ | ✅ | ✅ | - |
| CUDA 11.8 | ✅ | ✅ | ✅ | For GPU support |

---

## Model Registry

### Pre-trained Models

| Model | Source | Use Case | Size | Notes |
|-------|--------|----------|------|-------|
| IndicBERT | Facebook/Hugging Face | Primary (all tasks) | 268M | Indic-optimized |
| mBERT | Google | Robustness check | 168M | Multilingual baseline |
| XLM-R | Facebook | Robustness check | 550M | Cross-lingual SOTA |
| IndicRoBERTa | (if available) | Stage 3+ | 355M | Future variant |

### Loading Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Stage 1 baseline
model_name = "google/bert-base-multilingual-cased"  # mBERT
# OR
model_name = "sarvamai/IndicBERT"  # IndicBERT

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Task-dependent
    ignore_mismatched_sizes=True
)
```

---

## Dataset Registry

| Dataset | Tasks | Languages | Source | License |
|---------|-------|-----------|--------|---------|
| XNLI-Indic | NLI | 5 Indic + En | Google Research | CC-BY-4.0 |
| BLUB SA | Sentiment | Indic | IIT Kharagpur | Research Use |
| BLUB NER | NER | Indic | IIT Kharagpur | Research Use |
| BLUB News | Classification | Indic | IIT Kharagpur | Research Use |

### Loading Datasets

```python
from datasets import load_dataset

# XNLI-Indic
xnli_hi = load_dataset("xnli", "hi")  # Hindi
xnli_bn = load_dataset("xnli", "bn")  # Bangla

# BLUB (requires local setup)
# See: https://github.com/aditya-hari/BLUB
sa_dataset = load_from_disk("path/to/blub/sentiment_analysis")
```

---

## Hardware Requirements

### Minimum (Colab Pro)

```
GPU:       1x NVIDIA A100 (40GB) or T4 (16GB) - Colab Pro provides
Memory:    13GB RAM (Colab default)
Storage:   100GB (Google Drive)
Compute:   40-50 GPU hours/month (Colab Pro limit)
```

### Recommended (if using local machine)

```
GPU:       1x RTX 3090 or better (24GB+ VRAM)
CPU:       16+ cores
Memory:    32GB RAM
Storage:   500GB SSD (for model checkpoints)
```

---

## Reproducibility Checklist

- [ ] All versions pinned in `requirements.txt`
- [ ] Python 3.10 or 3.11 confirmed
- [ ] CUDA 11.8 or CUDA-less CPU environment documented
- [ ] Seed values fixed (torch.manual_seed, numpy.random.seed, etc.)
- [ ] Colab Pro subscription active (for consistent GPU allocation)
- [ ] W&B project created and configured
- [ ] Git repository initialized with `.gitignore` for large files
- [ ] Test run of baseline experiment completed successfully
- [ ] Dependency installation documented in README

---

## Troubleshooting Common Issues

### Issue: CUDA Out of Memory
```python
# Solution 1: Reduce batch size
train_batch_size = 8  # Instead of 16/32

# Solution 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use LoRA (Stage 2+)
# Reduces memory from ~11GB to ~3GB
```

### Issue: Dataset Loading Fails
```bash
# Solution: Manually download and cache
huggingface-cli download datasets/xnli --cache-dir ./data/cache
```

### Issue: W&B Authentication
```bash
# Solution: Set API key directly
export WANDB_API_KEY="your-key-here"
# Or in code:
import os
os.environ["WANDB_API_KEY"] = "your-key-here"
```

---

## Future Considerations

- **TensorFlow 2.x support**: If needed later, can parallelize with PyTorch
- **ONNX export**: For model deployment
- **JAX integration**: For advanced differentiation (Stage 4+ probing tasks)
- **Ray Tune**: For hyperparameter search (if scaling beyond Colab)

---

**Last Updated:** 2025-10-21
**Maintained By:** Research Team
**Next Review:** End of Month 1 (after first baseline run)
