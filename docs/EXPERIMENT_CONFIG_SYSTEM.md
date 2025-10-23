# Experiment Configuration System

**Document Version:** 1.0
**Last Updated:** 2025-10-21
**Purpose:** Define standardized YAML configuration format for reproducible, config-driven experiments

---

## Overview

This document specifies the complete experiment configuration system that enables:
- ✅ Single-command experiment execution: `python train.py --config path_a_er_sa.yaml`
- ✅ Full reproducibility: All hyperparameters in version-controlled YAML
- ✅ Systematic comparisons: Easy to run baseline, Path A, Paths B/C with identical settings
- ✅ Hyperparameter tracking: All configs logged to W&B for sweep management

**Key Design Principles:**
1. **Declarative over Imperative**: All experiment parameters in YAML, not scattered in code
2. **Inheritance**: Base configs can be extended/overridden for variants
3. **Validation**: Schema checking before training starts
4. **Audit Trail**: Every config saved with experiment results

---

## Directory Structure

```
experiments/
├── configs/
│   ├── base/
│   │   ├── base.yaml                    # Global defaults
│   │   ├── task_defaults.yaml           # Task-specific defaults
│   │   └── model_defaults.yaml          # Model-specific defaults
│   │
│   ├── stage1_baselines/
│   │   ├── direct_bangla.yaml           # Direct fine-tuning (no transfer)
│   │   ├── pairwise_hi_bn.yaml          # Hindi→Bangla
│   │   ├── pairwise_mr_bn.yaml          # Marathi→Bangla
│   │   ├── pairwise_ta_bn.yaml          # Tamil→Bangla
│   │   ├── pairwise_te_bn.yaml          # Telugu→Bangla
│   │   ├── pairwise_en_bn.yaml          # English→Bangla
│   │   ├── joint_multilingual.yaml      # All sources jointly
│   │   └── few_shot_variants.yaml       # Shared few-shot settings
│   │
│   ├── stage2_sequential/
│   │   ├── path_a_baseline.yaml         # Path A (Hi→Mr→Ta→Te→Bn)
│   │   ├── path_a_er.yaml               # Path A + Experience Replay
│   │   ├── path_a_lora.yaml             # Path A + LoRA
│   │   └── path_a_fewshot.yaml          # Path A few-shot variants
│   │
│   ├── stage3_curriculum/
│   │   ├── path_b_er.yaml               # Path B (Ta→Te→Hi→Mr→Bn)
│   │   ├── path_c_er.yaml               # Path C (Hi→Ta→Mr→Te→Bn)
│   │   └── path_comparison.yaml         # Shared settings for all paths
│   │
│   └── stage4_linguistic/
│       ├── probing_tasks.yaml           # Linguistic feature probing
│       └── distance_analysis.yaml       # Correlation analysis settings
│
├── results/
│   ├── stage1_baselines/
│   ├── stage2_sequential/
│   ├── stage3_curriculum/
│   └── stage4_linguistic/
│
└── scripts/
    ├── train.py                         # Main training script
    ├── config_loader.py                 # YAML loader + validator
    ├── experiment_runner.py             # Batch runner
    └── validate_config.py               # Schema validation
```

---

## Core Configuration Schema (base.yaml)

```yaml
# experiments/configs/base/base.yaml
# Global defaults inherited by all experiments

# ============================================================================
# PROJECT & TRACKING
# ============================================================================
project:
  name: "bangla_continual_learning"
  description: "Sequential continual learning for cross-lingual transfer to Bangla"
  version: "1.0"
  tracking_platform: "wandb"  # or "mlflow"

tracking:
  wandb:
    project: "bangla-cl"
    entity: "research-team"  # Change to your W&B team
    tags: ["stage1", "baseline"]  # Override per experiment
    notes: "Baseline experiments - determining performance envelope"

  checkpoint_dir: "./checkpoints/"
  logs_dir: "./logs/"
  results_dir: "./results/"

# ============================================================================
# RANDOM SEEDS (for reproducibility)
# ============================================================================
seed: 42
deterministic: true

# ============================================================================
# DATASET & DATA LOADING
# ============================================================================
data:
  source_languages: ["hi"]  # Override per experiment
  target_language: "bn"
  tasks: ["sentiment_analysis"]  # Override per experiment

  # Data normalization strategy
  normalization:
    strategy: "equal_tokens"  # Normalize all languages to same token count
    target_tokens: 100000  # Total tokens per language
    random_sampling: true
    seed: 42

  # Few-shot configuration
  few_shot:
    enabled: false  # Override per experiment
    target_language_examples:
      - 100
      - 500
      - 1000

  # Dataset sources
  sources:
    xnli_indic:
      repo: "datasets/xnli"
      languages: ["hi", "mr", "ta", "te", "bn"]

    blub_sentiment:
      repo: "local"  # BLUB requires local setup
      path: "./data/BLUB/sentiment_analysis/"

    blub_ner:
      repo: "local"
      path: "./data/BLUB/ner/"

    blub_news:
      repo: "local"
      path: "./data/BLUB/news_classification/"

  # Data splits
  splits:
    train: 0.8
    val: 0.1
    test: 0.1

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model:
  base_model: "sarvamai/IndicBERT"  # Primary model

  # Model task-specific heads
  heads:
    sentiment_analysis:
      type: "sequence_classification"
      num_labels: 2
      loss: "cross_entropy"

    ner:
      type: "token_classification"
      num_labels: 9  # Standard NER tag set
      loss: "cross_entropy"

    nli:
      type: "sequence_classification"
      num_labels: 3  # entailment, neutral, contradiction
      loss: "cross_entropy"

    news_classification:
      type: "sequence_classification"
      num_labels: 4  # Task-dependent
      loss: "cross_entropy"

  # Initialization
  initialization:
    pretrained_weights: true
    freeze_base: false  # Allow full fine-tuning by default
    dropout: 0.1

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
training:
  # Optimizer
  optimizer: "adamw"
  learning_rate: 2e-5
  weight_decay: 0.0
  warmup_steps: 500
  max_grad_norm: 1.0

  # Batch size & epochs
  batch_size: 16
  gradient_accumulation_steps: 1
  num_epochs: 3  # Override per stage

  # Evaluation
  eval_strategy: "epoch"  # Evaluate every epoch
  eval_steps: null  # Or evaluate every N steps
  save_strategy: "epoch"

  # Early stopping
  early_stopping:
    enabled: false  # Override if needed
    metric: "eval_f1"
    patience: 3
    min_delta: 0.001

  # Hardware
  use_gpu: true
  mixed_precision: "fp16"  # or "bf16" on newer GPUs
  num_workers: 4
  pin_memory: true

  # Checkpointing
  save_total_limit: 3  # Keep only last 3 checkpoints
  resume_from_checkpoint: null

# ============================================================================
# EVALUATION METRICS
# ============================================================================
evaluation:
  task_metrics:
    sentiment_analysis: ["accuracy", "f1_macro", "precision", "recall"]
    ner: ["f1_token", "precision", "recall", "ner_tags"]
    nli: ["accuracy", "f1_macro"]
    news_classification: ["accuracy", "f1_macro", "precision", "recall"]

  # Transfer-specific metrics
  transfer_metrics:
    forward_transfer: true  # Measure target language performance after source
    backward_transfer: true  # Measure source performance degradation
    adaptation_gain: true   # Constructive vs. catastrophic forgetting

  # Compute metrics on all languages
  eval_all_languages: true

  # Statistical testing (Stage 3+)
  statistical_tests:
    enabled: false
    alpha: 0.05  # Significance level

# ============================================================================
# LOGGING & MONITORING
# ============================================================================
logging:
  log_level: "INFO"
  log_to_file: true
  log_file: "experiment.log"

  # What to log
  log_frequency: 100  # Every N steps
  log_metrics:
    - "loss"
    - "learning_rate"
    - "gradient_norm"

  # Model architecture logging
  log_model_summary: true

# ============================================================================
# EXPERIMENT METADATA
# ============================================================================
metadata:
  researcher: "Unknown"  # Override per experiment
  advisor: "Unknown"
  experiment_date: null  # Auto-filled at runtime
  notes: ""
  hypothesis: ""

# ============================================================================
# VALIDATION FLAGS
# ============================================================================
validation:
  check_data_quality: true
  plot_data_distribution: true
  dry_run: false  # If true, run on 100 samples only
  validate_on_startup: true
```

---

## Stage 1: Baseline Experiments

### Direct Bangla (no transfer)

```yaml
# experiments/configs/stage1_baselines/direct_bangla.yaml
# Baseline: Direct fine-tuning on Bangla only

extends: "../base/base.yaml"

experiment:
  stage: 1
  name: "direct_bangla_sa"
  type: "baseline"
  description: "Direct fine-tuning on Bangla - establishes performance floor"

data:
  source_languages: null  # No source language
  target_language: "bn"
  tasks: ["sentiment_analysis"]

training:
  num_epochs: 5  # More epochs for baseline
  batch_size: 16
  learning_rate: 2e-5

tracking:
  wandb:
    tags: ["stage1", "baseline", "direct_transfer"]
    notes: "Direct Bangla fine-tuning - performance ceiling for no transfer"

metadata:
  hypothesis: "Direct fine-tuning establishes baseline performance"
```

### Pairwise Transfer (Hindi→Bangla)

```yaml
# experiments/configs/stage1_baselines/pairwise_hi_bn.yaml

extends: "../base/base.yaml"

experiment:
  stage: 1
  name: "pairwise_hi_bn_sa"
  type: "baseline_pairwise"
  description: "Single-source transfer: Hindi → Bangla"

data:
  source_languages: ["hi"]
  target_language: "bn"
  tasks: ["sentiment_analysis"]

  # Two-stage training
  training_pipeline:
    stage_1:
      language: "hi"
      epochs: 2
      checkpoint: true
    stage_2:
      language: "bn"
      epochs: 3
      checkpoint: true

training:
  batch_size: 16
  learning_rate: 2e-5

tracking:
  wandb:
    tags: ["stage1", "baseline", "pairwise_transfer", "hindi"]

metadata:
  hypothesis: "Hindi transfer provides meaningful Bangla performance improvement"
```

### Joint Multilingual Training

```yaml
# experiments/configs/stage1_baselines/joint_multilingual.yaml
# Train on all source languages simultaneously, then fine-tune on Bangla

extends: "../base/base.yaml"

experiment:
  stage: 1
  name: "joint_multilingual_sa"
  type: "baseline_joint"
  description: "Simultaneous training on all source languages, then fine-tune Bangla"

data:
  source_languages: ["hi", "mr", "ta", "te"]
  target_language: "bn"
  tasks: ["sentiment_analysis"]

  # Joint training then adaptation
  training_pipeline:
    stage_1:
      languages: ["hi", "mr", "ta", "te"]
      strategy: "mixed_sampling"  # Interleave languages
      epochs: 3
    stage_2:
      language: "bn"
      epochs: 2

training:
  batch_size: 32  # Larger batch for simultaneous learning
  learning_rate: 2e-5
  warmup_steps: 1000

tracking:
  wandb:
    tags: ["stage1", "baseline", "joint_training"]

metadata:
  hypothesis: "Joint training on all sources provides better initialization than pairwise"
```

### Few-Shot Variants Configuration

```yaml
# experiments/configs/stage1_baselines/few_shot_variants.yaml
# Shared configuration for few-shot experiments (100/500/1000 examples)

few_shot_configs:
  # Generated automatically for each baseline
  100_examples:
    description: "Extreme few-shot: 100 Bangla examples"
    target_examples: 100
    repetitions: 3  # Run 3x with different seeds for stability
    seeds: [42, 123, 456]

  500_examples:
    description: "Very few-shot: 500 Bangla examples"
    target_examples: 500
    repetitions: 3
    seeds: [42, 123, 456]

  1000_examples:
    description: "Few-shot: 1000 Bangla examples"
    target_examples: 1000
    repetitions: 3
    seeds: [42, 123, 456]

# When applied to any baseline config:
# python train.py --config pairwise_hi_bn.yaml --few_shot 100
# Creates 3 experiment variants with different random seeds
```

---

## Stage 2: Sequential Continual Learning

### Path A with Experience Replay

```yaml
# experiments/configs/stage2_sequential/path_a_er.yaml
# Core experiment: Sequential learning with experience replay

extends: "../base/base.yaml"

experiment:
  stage: 2
  name: "path_a_er_sa"
  type: "sequential_continual_learning"
  description: "Path A (Hi→Mr→Ta→Te→Bn) with Experience Replay"

data:
  # Sequential path definition
  sequential_path:
    - language: "hi"
      epochs: 2
      save_checkpoint: true
    - language: "mr"
      epochs: 2
      save_checkpoint: true
    - language: "ta"
      epochs: 2
      save_checkpoint: true
    - language: "te"
      epochs: 2
      save_checkpoint: true
    - language: "bn"
      epochs: 3
      save_checkpoint: true

  tasks: ["sentiment_analysis"]

# ============================================================================
# CONTINUAL LEARNING CONFIGURATION
# ============================================================================
continual_learning:
  method: "experience_replay"  # Options: ["replay", "lora", "ewc"]

  # Experience Replay settings
  replay:
    buffer_size: 0.2  # Keep 20% of previous language data
    sampling_strategy: "uniform"  # or "weighted_by_difficulty"
    merge_current_with_replay: true  # Mix current + replay
    replay_ratio: 0.5  # Ratio of replay samples in batch

  # Checkpoint management
  checkpoints:
    save_at_each_language: true
    eval_all_languages: true  # Evaluate on all previous languages
    checkpoint_format: "huggingface"

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: null  # Use per-language epochs instead

  # Might need reduced LR after first language
  adaptive_lr:
    enabled: true
    decay_factor: 0.9  # Reduce by 10% each language transition
    min_lr: 1e-5

# ============================================================================
# EVALUATION FOR CL EXPERIMENTS
# ============================================================================
evaluation:
  # Evaluate on ALL languages at each checkpoint
  comprehensive_eval: true

  eval_languages:
    at_step_0: ["hi"]  # After Hindi training
    at_step_1: ["hi", "mr"]  # After Marathi training
    at_step_2: ["hi", "mr", "ta"]
    at_step_3: ["hi", "mr", "ta", "te"]
    at_step_4: ["hi", "mr", "ta", "te", "bn"]

  # Key metrics for CL
  transfer_metrics:
    forward_transfer: true
    backward_transfer: true
    forgetting_rate: true
    adaptation_gain: true

# ============================================================================
# OUTPUT & ANALYSIS
# ============================================================================
output:
  save_forgetting_curves: true
  save_checkpoint_analysis: true
  save_per_language_metrics: true

  # Auto-generate analysis plots
  plots:
    - "forgetting_curves"          # Line plot of forgetting over time
    - "forward_transfer_heatmap"   # Heatmap: language pairs × metrics
    - "checkpoint_performance"     # Performance at each transition
    - "adaptation_gain_bars"       # Constructive vs catastrophic

tracking:
  wandb:
    tags: ["stage2", "sequential", "experience_replay", "path_a"]
    notes: "Path A sequential training with 20% experience replay buffer"

metadata:
  hypothesis: "Sequential training with replay preserves source knowledge better than direct Bangla fine-tuning"
```

### Path A with LoRA (Low-Rank Adaptation)

```yaml
# experiments/configs/stage2_sequential/path_a_lora.yaml
# Path A with parameter-efficient LoRA instead of full fine-tuning

extends: "./path_a_er.yaml"

experiment:
  name: "path_a_lora_sa"
  description: "Path A (Hi→Mr→Ta→Te→Bn) with LoRA"

# Override continual learning method
continual_learning:
  method: "lora"

  lora:
    r: 8                          # LoRA rank
    lora_alpha: 16                # LoRA scaling
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.05
    bias: "none"
    task_type: "SEQ_CLS"          # Task type for PEFT

    # Per-language LoRA modules
    per_language_adapters: true  # Create separate adapter per language
    adapter_prefix: "lora_"

training:
  # LoRA usually needs higher LR due to smaller parameter updates
  learning_rate: 5e-4

  # Can use larger batch size with LoRA (lower memory)
  batch_size: 32

  # LoRA typically converges faster
  num_epochs_per_language: 1

tracking:
  wandb:
    tags: ["stage2", "sequential", "lora", "path_a", "parameter_efficient"]

metadata:
  hypothesis: "LoRA reduces memory footprint while achieving comparable forgetting patterns to full fine-tuning"
```

---

## Stage 3: Curriculum Learning Analysis

### Path B (Distant→Similar)

```yaml
# experiments/configs/stage3_curriculum/path_b_er.yaml

extends: "../base/base.yaml"

experiment:
  stage: 3
  name: "path_b_er_sa"
  type: "curriculum_learning"
  description: "Path B (Ta→Te→Hi→Mr→Bn): Distant→Similar ordering"

data:
  sequential_path:
    - language: "ta"      # Tamil (distant)
      epochs: 2
    - language: "te"      # Telugu (distant)
      epochs: 2
    - language: "hi"      # Hindi (similar)
      epochs: 2
    - language: "mr"      # Marathi (similar)
      epochs: 2
    - language: "bn"      # Bangla (target)
      epochs: 3

continual_learning:
  method: "experience_replay"
  replay:
    buffer_size: 0.2
    sampling_strategy: "uniform"

evaluation:
  comprehensive_eval: true
  curriculum_analysis: true

tracking:
  wandb:
    tags: ["stage3", "curriculum", "path_b", "distant_similar"]
    notes: "Curriculum: Distant languages first, then similar"
```

### Path C (Alternating)

```yaml
# experiments/configs/stage3_curriculum/path_c_er.yaml

extends: "../base/base.yaml"

experiment:
  stage: 3
  name: "path_c_er_sa"
  type: "curriculum_learning"
  description: "Path C (Hi→Ta→Mr→Te→Bn): Alternating diversity"

data:
  sequential_path:
    - language: "hi"      # Indo-Aryan
      epochs: 2
    - language: "ta"      # Dravidian
      epochs: 2
    - language: "mr"      # Indo-Aryan
      epochs: 2
    - language: "te"      # Dravidian
      epochs: 2
    - language: "bn"      # Target
      epochs: 3

continual_learning:
  method: "experience_replay"
  replay:
    buffer_size: 0.2

evaluation:
  comprehensive_eval: true
  curriculum_analysis: true

tracking:
  wandb:
    tags: ["stage3", "curriculum", "path_c", "alternating"]
    notes: "Curriculum: Alternating language families for diversity"
```

### Curriculum Comparison Configuration

```yaml
# experiments/configs/stage3_curriculum/path_comparison.yaml
# Settings for statistical comparison across Path A, B, C

comparison:
  paths: ["path_a_er", "path_b_er", "path_c_er"]

  # Statistical tests
  statistical_analysis:
    test_type: "anova"  # ANOVA across 3 groups
    null_hypothesis: "All paths achieve equivalent Bangla F1"
    alpha: 0.05

    # Post-hoc tests if significant
    posthoc:
      method: "tukey_hsd"

    # Additional comparisons
    pairwise_tests:
      - ["path_a_er", "path_b_er"]
      - ["path_a_er", "path_c_er"]
      - ["path_b_er", "path_c_er"]

  # Effect sizes
  effect_sizes:
    metric: "cohen_d"
    interpretation: "guidelines"

  # Visualization
  comparison_plots:
    - "paths_boxplot"              # Box plot: F1 scores across paths
    - "forgetting_heatmap_comparison"  # Side-by-side heatmaps
    - "statistical_summary"        # P-values and effect sizes
    - "curriculum_effect_sizes"    # Cohen's d visualization

output:
  decision_report: true  # Auto-generate decision report
```

---

## Stage 4: Linguistic Feature Analysis

### Linguistic Distance Analysis

```yaml
# experiments/configs/stage4_linguistic/distance_analysis.yaml

experiment:
  stage: 4
  name: "linguistic_distance_analysis"
  type: "interpretability_analysis"
  description: "Correlate linguistic distance metrics with forgetting patterns"

# ============================================================================
# LINGUISTIC DISTANCE METRICS
# ============================================================================
linguistic_analysis:
  distance_metrics:
    # Lexical similarity (cognate overlap, vocabulary similarity)
    lexical:
      enabled: true
      methods: ["cognate_overlap", "embedding_similarity", "alignment"]

      cognate_overlap:
        # Compare character n-grams between languages
        ngram_size: 3
        threshold: 0.7

      embedding_similarity:
        # Use model embeddings
        model: "sarvamai/IndicBERT"
        embedding_layer: -1

    # Syntactic similarity
    syntactic:
      enabled: true
      methods: ["treebank_distance", "pos_overlap", "dependency_patterns"]

      # Use UD parsers
      parser: "spacy_ud"

      # Compute POS tag similarity
      pos_overlap:
        # Measure how similar POS distributions are
        metric: "cosine_similarity"

    # Morphological complexity
    morphological:
      enabled: true
      methods: ["morpheme_per_word", "agglutination_index", "inflection_richness"]

  language_pairs:
    - ["hi", "mr"]
    - ["hi", "ta"]
    - ["hi", "te"]
    - ["mr", "ta"]
    - ["mr", "te"]
    - ["ta", "te"]
    - ["hi", "bn"]
    - ["mr", "bn"]
    - ["ta", "bn"]
    - ["te", "bn"]

# ============================================================================
# CORRELATION ANALYSIS (RQ3)
# ============================================================================
correlation_analysis:
  dependent_variable: "forgetting_rate"  # What we're predicting
  independent_variables:
    - "lexical_distance"
    - "syntactic_distance"
    - "morphological_distance"

  # Control for confounds
  confound_variables:
    - "source_data_size"
    - "target_data_size"
    - "task_type"

  # Statistics
  methods:
    - "pearson"           # Pearson correlation
    - "spearman"          # Spearman rank correlation
    - "ols_regression"    # OLS: Forgetting ~ distance + confounds

  success_criteria:
    pearson_r_threshold: 0.6
    regression_r2_threshold: 0.5

  output:
    - "correlation_matrix"        # Distance vs forgetting
    - "scatterplots"              # Scatter with regression lines
    - "regression_summary"        # Model coefficients, R²
    - "residual_plots"            # Check model assumptions

# ============================================================================
# FEATURE PROBING TASKS (RQ4)
# ============================================================================
feature_probing:
  enabled: true

  probing_tasks:
    # Morphological features
    morphology:
      task_type: "sequence_tagging"
      target_features: ["case", "gender", "number", "tense"]
      description: "Can we classify morphological features from layer activations?"

    # Syntactic features
    syntax:
      task_type: "dependency_prediction"
      target_features: ["pos", "dependency_type", "head_distance"]
      description: "Can we predict dependency relations from embeddings?"

    # Semantic features
    semantics:
      task_type: "lexical_entailment"
      target_features: ["synonym_detection", "antonym_detection"]
      description: "Can we classify semantic similarity from representations?"

  # Probing strategy
  probe_layers: "all"  # Analyze all transformer layers

  # At which checkpoints to probe
  checkpoint_probing:
    after_each_language: true
    checkpoints_to_analyze:
      - "after_language_0"  # After first language
      - "after_language_1"
      - "after_language_2"
      - "after_language_3"
      - "after_language_4"  # After target

  # Output
  output:
    - "probing_accuracy_curves"   # How accuracy changes per layer
    - "layer_wise_feature_importance"  # Which layers encode which features
    - "feature_degradation_heatmap"  # How features degrade at transitions
    - "resilience_ranking"        # Features ranked by resilience

# ============================================================================
# CONSTRUCTIVE vs. CATASTROPHIC FORGETTING FRAMEWORK
# ============================================================================
forgetting_analysis:
  framework_enabled: true

  # Define constructive vs catastrophic
  adaptation_gain_formula: |
    Adaptation_Gain = Bangla_F1_Gain - (Source_F1_Loss × proximity_weight)
    If Adaptation_Gain > 0: Constructive (beneficial)
    If Adaptation_Gain < 0: Catastrophic (harmful)

  proximity_weight:
    # Languages closer to target get higher weights
    linguistic_distance_based: true
    # Closer languages have higher penalty if forgotten

  # Visualization
  constructive_vs_catastrophic:
    - "adaptation_gain_bars"      # Show gain/loss for each transition
    - "forgetting_heatmap_colored"  # Color by constructive (green) or catastrophic (red)
    - "net_gain_trajectory"       # Running total of adaptation gains

# ============================================================================
# OUTPUT & REPORTING
# ============================================================================
output:
  generate_report: true
  report_sections:
    - "executive_summary"
    - "linguistic_distance_analysis"
    - "correlation_findings"
    - "feature_resilience_ranking"
    - "constructive_vs_catastrophic_assessment"
    - "recommendations_for_practitioners"

tracking:
  wandb:
    tags: ["stage4", "interpretability", "linguistic_analysis", "rq3_rq4"]
```

### Linguistic Feature Probing Configuration

```yaml
# experiments/configs/stage4_linguistic/probing_tasks.yaml

experiment:
  stage: 4
  name: "linguistic_feature_probing"
  type: "probing_analysis"
  description: "Classify linguistic features from model representations"

# ============================================================================
# PROBING CLASSIFIERS
# ============================================================================
probing_classifiers:
  # Each probing task uses a shallow classifier on frozen representations

  morphology:
    task: "sequence_classification"
    target_labels: ["case", "gender", "number", "tense", "aspect"]
    classifier: "linear_probe"  # Just linear layer on frozen embeddings
    dataset: "annotated_morphology_corpus"

  pos_tagging:
    task: "token_classification"
    target_labels: ["POS_tags"]  # Universal POS
    classifier: "linear_probe"

  dependency_parsing:
    task: "sequence_labeling"
    target_labels: ["dependency_type", "head_distance"]
    classifier: "linear_probe"

# ============================================================================
# LAYER ANALYSIS CONFIGURATION
# ============================================================================
layer_analysis:
  # Analyze each transformer layer separately
  num_layers: 12  # IndicBERT has 12 layers

  analysis_per_layer:
    - layer: 0
      name: "embedding_layer"
    - layer: [1, 2, 3]
      name: "early_layers"
    - layer: [4, 5, 6, 7, 8]
      name: "middle_layers"
    - layer: [9, 10, 11]
      name: "late_layers"

  # Aggregate analysis
  aggregation:
    - "per_layer"
    - "by_layer_group"
    - "mean_across_layers"

# ============================================================================
# OUTPUT
# ============================================================================
output:
  plots:
    - "probing_accuracy_per_layer"      # Accuracy vs layer depth
    - "feature_accessibility_heatmap"   # Which layers encode which features
    - "layer_wise_degradation"          # How features degrade per layer at transitions

  tables:
    - "layer_wise_feature_accuracy"     # Detailed accuracy table
    - "feature_resilience_summary"      # Which features survive best
```

---

## Configuration Loading & Validation

### config_loader.py

```python
"""
Configuration loader with validation and inheritance.
Usage:
    loader = ConfigLoader()
    config = loader.load("path_a_er_sa.yaml")
    config.validate()
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import jsonschema

class ConfigLoader:
    def __init__(self, config_dir: Path = Path("experiments/configs")):
        self.config_dir = config_dir

    def load(self, config_file: str) -> Dict[str, Any]:
        """Load config with inheritance support (extends: field)"""
        config_path = self.config_dir / config_file

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if "extends" in config:
            base_path = self.config_dir / config.pop("extends")
            base_config = self.load(str(base_path.relative_to(self.config_dir)))
            # Deep merge
            config = self._merge_dicts(base_config, config)

        return config

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate against JSON schema"""
        schema = self._get_schema()
        try:
            jsonschema.validate(config, schema)
            print("✅ Config validation passed")
            return True
        except jsonschema.ValidationError as e:
            print(f"❌ Config validation failed: {e.message}")
            return False

    @staticmethod
    def _merge_dicts(base: dict, override: dict) -> dict:
        """Deep merge override into base"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
```

### Example: Running an Experiment

```bash
# Run single experiment
python train.py --config stage2_sequential/path_a_er.yaml

# Run with few-shot override
python train.py --config stage1_baselines/pairwise_hi_bn.yaml --few_shot 100

# Run all Stage 1 baselines
python experiment_runner.py --stage 1

# Run Stage 3 curriculum comparison + statistical analysis
python experiment_runner.py --stage 3 --run_comparison true
```

---

## Configuration Workflow

```
1. User creates/modifies YAML config
   ↓
2. Config loader validates against schema
   ↓
3. Config merged with base configs (inheritance)
   ↓
4. Hyperparameters extracted and logged to W&B
   ↓
5. Training runs with full config provenance
   ↓
6. Results saved with config copy for reproducibility
   ↓
7. Analysis scripts use config to determine what to compute
```

---

## Best Practices

### ✅ DO:

- Pin all hyperparameters in YAML, not in code
- Use inheritance for common settings
- Include hypothesis in metadata
- Log all configs to W&B
- Version control all configs in git

### ❌ DON'T:

- Hardcode experiment parameters in Python
- Copy-paste configs without using inheritance
- Forget to update tracking.wandb.tags
- Run experiments without saving the config

---

## Files Reference

| File | Purpose |
|------|---------|
| `base.yaml` | Global defaults |
| `task_defaults.yaml` | Task-specific overrides |
| `model_defaults.yaml` | Model-specific overrides |
| `*_baselines/*.yaml` | Stage 1 configurations |
| `stage2_*/*.yaml` | Stage 2 configurations |
| `stage3_*/*.yaml` | Stage 3 configurations |
| `stage4_*/*.yaml` | Stage 4 configurations |

---

**Last Updated:** 2025-10-21
**Maintained By:** Research Team
**Next Review:** After first experiment run
