# Epics & User Stories: Continual Learning for Cross-Lingual Transfer to Bangla

**Document Version:** 1.0
**Created:** 2025-10-21
**Research Team:** Mehreen Hossain Chowdhury, Ahmed Shafin Ruhan, Nowshin Mahjabin
**Institution:** Islamic University of Technology, Department of CSE

---

## Table of Contents

1. [Overview](#overview)
2. [Epic Structure](#epic-structure)
3. [EPIC 1: Infrastructure & Baseline Setup](#epic-1-infrastructure--baseline-setup)
4. [EPIC 2: Core Continual Learning Implementation](#epic-2-core-continual-learning-implementation)
5. [EPIC 3: Linguistic Analysis & Transfer Language Selection](#epic-3-linguistic-analysis--transfer-language-selection)
6. [EPIC 4: Parameter-Efficient Fine-Tuning with LoRA](#epic-4-parameter-efficient-fine-tuning-with-lora)
7. [EPIC 5: Comprehensive Evaluation & Analysis](#epic-5-comprehensive-evaluation--analysis)
8. [EPIC 6: Documentation & Thesis Writing](#epic-6-documentation--thesis-writing)
9. [Dependencies & Timeline](#dependencies--timeline)
10. [Success Metrics](#success-metrics)

---

## Overview

This document defines the complete work breakdown structure (WBS) for the thesis research on continual learning for cross-lingual transfer to Bangla. Six major epics organize the implementation across 9 months, with detailed user stories providing concrete acceptance criteria and technical specifications.

### Research Objectives
- Implement continual learning strategies for sequential Indic language training
- Evaluate transfer effectiveness across linguistic distances
- Measure and mitigate catastrophic forgetting in multilingual scenarios
- Optimize transfer language selection for Bangla NLP tasks

### Key Papers
1. Overcoming Catastrophic Forgetting in Massively Multilingual CL (Bhavsar et al., 2023)
2. Choosing Transfer Languages for Cross-Lingual Learning (Lin et al., ACL 2019)
3. Sequential Continual Pre-Training for NMT (2024)
4. BanglaBERT: Language Model Pretraining (2022)
5. LoRA: Low-Rank Adaptation (2022)

---

## Epic Structure

Each epic includes:
- **Goal:** High-level objective
- **Duration:** Estimated timeline
- **Acceptance Criteria:** Epic-level success conditions
- **User Stories:** Concrete, implementable tasks
- **Dependencies:** Other epics/stories that must precede
- **Success Metrics:** Measurable outcomes

---

## EPIC 1: Infrastructure & Baseline Setup

**Duration:** Month 1 (4 weeks)
**Priority:** P0 (Critical Path)
**Effort:** 8 story points
**Owner:** Full Team

### Epic Goal
Establish the foundational development environment, dataset pipelines, and baseline model performance metrics for all tasks (SA, NER, NLI, News Classification).

### Acceptance Criteria
- [ ] Development environment fully configured with all dependencies
- [ ] BanglaBERT model successfully loaded and integrated
- [ ] Dataset pipelines operational for all 4 NLP tasks
- [ ] Baseline performance metrics established and documented
- [ ] Evaluation harness with automated metrics tracking created
- [ ] CI/CD pipeline for testing configured
- [ ] All team members can reproduce baseline results

### User Stories

#### Story 1.1: Setup Development Environment
```yaml
ID: STORY-1-1
Title: Setup Development Environment with Required Libraries
Epic: EPIC 1
Priority: P0
Effort: 3 story points
Duration: 3 days

Description:
As a researcher, I need a configured development environment with all required
libraries so that I can run experiments consistently across team members.

Acceptance Criteria:
- [ ] requirements.txt created with all dependencies (transformers==4.30+, torch==2.0+, etc.)
- [ ] Python 3.10+ environment working
- [ ] Virtual environment setup instructions documented
- [ ] GPU drivers and CUDA compatibility verified
- [ ] All imports working without errors
- [ ] Installation tested on clean environment

Technical Requirements:
- PyTorch/Transformers for model training
- Datasets library for data loading
- Scikit-learn for metrics
- Matplotlib/Seaborn for visualization
- Jupyter notebooks for analysis
- WandB for experiment tracking

Dependencies:
- None (first story)

Definition of Done:
- requirements.txt committed to repository
- Setup guide in README completed
- All team members can pip install and run import tests
- Documentation includes troubleshooting section
```

#### Story 1.2: Integrate BanglaBERT and Load Pretrained Model
```yaml
ID: STORY-1-2
Title: Integrate BanglaBERT Pretrained Model
Epic: EPIC 1
Priority: P0
Effort: 3 story points
Duration: 3 days

Description:
As a researcher, I need BanglaBERT loaded and tested so that I can use it
as the baseline model for all experiments.

Acceptance Criteria:
- [ ] BanglaBERT model downloaded and cached locally
- [ ] Model loads without errors
- [ ] Tokenizer properly configured
- [ ] Test inference on sample texts works
- [ ] Model configuration documented
- [ ] Memory requirements documented

Implementation Tasks:
1. Download BanglaBERT from HuggingFace (google/banglabert-base or equivalent)
2. Create model_loader.py utility
3. Test tokenization on sample Bangla text
4. Document model architecture and layers
5. Create inference script

Output:
- src/models/bangla_bert_loader.py
- docs/model_specifications.md
- Test script: tests/test_model_loading.py

Dependencies:
- Story 1.1 (Environment setup)
```

#### Story 1.3: Create Unified Data Pipeline for All 4 NLP Tasks
```yaml
ID: STORY-1-3
Title: Create Unified Data Pipeline for SA, NER, NLI, News Tasks
Epic: EPIC 1
Priority: P0
Effort: 5 story points
Duration: 4 days

Description:
As a researcher, I need a unified data pipeline so that I can load and
preprocess all four task types consistently.

Acceptance Criteria:
- [ ] Data loader for Sentiment Analysis (SA) created
- [ ] Data loader for Named Entity Recognition (NER) created
- [ ] Data loader for Natural Language Inference (NLI) created
- [ ] Data loader for News Classification created
- [ ] All loaders return standardized DataLoader objects
- [ ] Preprocessing handles tokenization, padding, sequence length
- [ ] Data statistics computed (vocab size, avg length, class distribution)
- [ ] Train/val/test splits properly managed

Implementation:
File: src/data/data_pipeline.py

Classes:
- TaskDataLoader (abstract base)
- SentimentAnalysisLoader(TaskDataLoader)
- NERLoader(TaskDataLoader)
- NLILoader(TaskDataLoader)
- NewsClassificationLoader(TaskDataLoader)

Config:
```yaml
tasks:
  sentiment_analysis:
    name: "Bangla Sentiment Analysis"
    dataset: "path/to/sa_data"
    num_classes: 3
    max_seq_length: 128

  ner:
    name: "Bangla NER"
    dataset: "path/to/ner_data"
    num_classes: 10  # BIO tags
    max_seq_length: 256

  nli:
    name: "Bangla NLI"
    dataset: "path/to/nli_data"
    num_classes: 3  # Entailment, Neutral, Contradiction
    max_seq_length: 256

  news:
    name: "Bangla News Classification"
    dataset: "path/to/news_data"
    num_classes: 5
    max_seq_length: 256
```

Output Files:
- src/data/data_pipeline.py
- src/data/task_configs.yaml
- docs/data_specifications.md

Dependencies:
- Story 1.1, 1.2
```

#### Story 1.4: Establish Baseline Performance Metrics
```yaml
ID: STORY-1-4
Title: Establish Baseline Performance Metrics for All Tasks
Epic: EPIC 1
Priority: P0
Effort: 4 story points
Duration: 3 days

Description:
As a researcher, I need baseline metrics for each task so that I can measure
improvement from continual learning strategies.

Acceptance Criteria:
- [ ] Baseline model trained on each task independently
- [ ] Accuracy, Precision, Recall, F1-score computed
- [ ] Metrics recorded in standardized format
- [ ] Baseline results documented in results/baselines.json
- [ ] Comparison with published benchmarks validated
- [ ] Confidence intervals computed

Metrics to Track:
- Task 1 (SA): Accuracy, F1 (weighted)
- Task 2 (NER): Token-level F1, entity-level F1
- Task 3 (NLI): Accuracy, per-class F1
- Task 4 (News): Accuracy, macro F1

Output:
```json
{
  "baseline_results": {
    "sentiment_analysis": {
      "accuracy": 0.78,
      "f1_weighted": 0.76,
      "per_class_f1": [0.75, 0.74, 0.79],
      "confidence_interval_95": [0.76, 0.80]
    },
    "ner": {
      "token_f1": 0.82,
      "entity_f1": 0.79,
      "confidence_interval_95": [0.80, 0.84]
    },
    "nli": {
      "accuracy": 0.71,
      "per_class_f1": [0.68, 0.70, 0.75]
    },
    "news": {
      "accuracy": 0.85,
      "macro_f1": 0.84
    }
  }
}
```

Dependencies:
- Stories 1.1, 1.2, 1.3
```

#### Story 1.5: Create Evaluation Harness & Metrics Tracking
```yaml
ID: STORY-1-5
Title: Create Evaluation Harness with Automated Metrics Tracking
Epic: EPIC 1
Priority: P0
Effort: 3 story points
Duration: 3 days

Description:
As a researcher, I need an automated evaluation system so that I can consistently
measure performance across experiments.

Acceptance Criteria:
- [ ] Evaluator class supports all 4 task types
- [ ] Metrics computed automatically after each training step
- [ ] Results logged to structured format (JSON/CSV)
- [ ] WandB integration for experiment tracking
- [ ] Real-time metrics display during training
- [ ] Comparison across experiments automated

Implementation:
```python
class EvaluationHarness:
    def __init__(self, task_type, metrics_dir):
        self.task_type = task_type
        self.metrics_dir = metrics_dir

    def evaluate(self, model, test_loader):
        # Compute task-specific metrics
        # Log to WandB
        # Save to JSON
        pass

    def compare_experiments(self, exp_ids):
        # Load metrics from multiple experiments
        # Generate comparison tables
        pass
```

Output:
- src/evaluation/evaluator.py
- src/evaluation/metrics_logger.py
- Integration with WandB dashboard

Dependencies:
- Stories 1.1-1.4
```

#### Story 1.6: Configure CI/CD and Testing Infrastructure
```yaml
ID: STORY-1-6
Title: Configure CI/CD Pipeline and Testing Infrastructure
Epic: EPIC 1
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a team, I need automated testing so that code changes don't break existing functionality.

Acceptance Criteria:
- [ ] GitHub Actions workflow configured
- [ ] Unit tests for data pipeline pass
- [ ] Unit tests for model loading pass
- [ ] Integration test for baseline training passes
- [ ] Pre-commit hooks configured for code quality
- [ ] README includes "Running Tests" section

Test Coverage:
- tests/test_data_pipeline.py
- tests/test_model_loading.py
- tests/test_evaluation.py
- tests/integration_test_baseline.py

Dependencies:
- Stories 1.1-1.5
```

---

## EPIC 2: Core Continual Learning Implementation

**Duration:** Month 2 (4 weeks)
**Priority:** P0 (Critical Path)
**Effort:** 13 story points
**Owner:** Primary researcher
**Depends On:** EPIC 1 complete

### Epic Goal
Implement the sequential training framework with LR ADJUST learning rate scheduling and complete forgetting measurement protocol based on Bhavsar et al. (2023).

### Acceptance Criteria
- [ ] Sequential training pipeline functional for 5-language sequences
- [ ] LR ADJUST algorithm implemented and validated
- [ ] Backward transfer measurement working correctly
- [ ] Forward transfer tracking implemented
- [ ] Training logs and checkpoints properly managed
- [ ] Performance metrics match paper's expectations
- [ ] Ablation studies (LR ADJUST vs. fixed LR) show 15-25% improvement

### User Stories

#### Story 2.1: Implement Sequential Training Pipeline
```yaml
ID: STORY-2-1
Title: Build Sequential Language Training Pipeline
Epic: EPIC 2
Priority: P0
Effort: 5 story points
Duration: 4 days

Description:
As a researcher, I need a sequential training pipeline so that I can train
the model on multiple languages in sequence and measure forgetting across languages.

Acceptance Criteria:
- [ ] Pipeline trains Language 1, saves checkpoint with performance metrics
- [ ] Pipeline trains Language 2, evaluates on both Lang 1 & 2, measures degradation
- [ ] Pipeline continues for all 5 languages (Hindi → Marathi → Tamil → Telugu → Bangla)
- [ ] Performance metrics saved after each language addition
- [ ] Pipeline supports different task types (classification, sequence labeling)
- [ ] Training can resume from checkpoints
- [ ] Experiment configuration saved for reproducibility

Technical Specification:

File: src/training/sequential_trainer.py

```python
class SequentialTrainer:
    def __init__(self, model, base_lr, config):
        self.model = model
        self.base_lr = base_lr
        self.config = config
        self.language_order = config['language_order']
        self.task_type = config['task_type']
        self.history = {}

    def train_language_sequence(self, dataloaders_dict, languages):
        """
        Train model sequentially on multiple languages.

        Args:
            dataloaders_dict: {lang: {train: DataLoader, val: DataLoader, test: DataLoader}}
            languages: List of languages in training order
        """
        self.language_history = {}

        for lang_id, language in enumerate(languages):
            print(f"Training on {language} (task {lang_id+1}/{len(languages)})")

            # Adjust learning rate
            current_lr = self.lr_adjust_schedule(lang_id)

            # Train on current language
            train_losses = self.train_epoch(
                dataloaders_dict[language]['train'],
                current_lr
            )

            # Evaluate on all languages
            results = self.evaluate_all_languages(
                dataloaders_dict,
                languages[:lang_id+1]
            )

            # Save checkpoint
            checkpoint = {
                'language': language,
                'lang_id': lang_id,
                'model_state': self.model.state_dict(),
                'results': results,
                'learning_rate': current_lr
            }
            self.save_checkpoint(checkpoint, lang_id)

            self.language_history[language] = results

        return self.language_history

    def evaluate_all_languages(self, dataloaders_dict, languages_to_eval):
        """Evaluate on all previously trained languages."""
        results = {}
        for lang in languages_to_eval:
            metrics = self.evaluator.evaluate(
                self.model,
                dataloaders_dict[lang]['test']
            )
            results[lang] = metrics
        return results

    def save_checkpoint(self, checkpoint, lang_id):
        """Save model checkpoint and metadata."""
        path = f"checkpoints/lang_{lang_id}_{checkpoint['language']}.pt"
        torch.save(checkpoint, path)
```

Pipeline Structure:
```
Input: language_config.yaml
  languages:
    - Hindi
    - Marathi
    - Tamil
    - Telugu
    - Bangla
  task: sentiment_analysis
  num_epochs: 3
  base_lr: 2e-5

Output:
  checkpoints/
    lang_0_hindi.pt
    lang_1_marathi.pt
    lang_2_tamil.pt
    lang_3_telugu.pt
    lang_4_bangla.pt

  results/
    sequential_training_history.json
    forgetting_curves.png
```

Dependencies:
- EPIC 1 complete
- Story 1.3 (Data pipeline)
- Story 1.5 (Evaluation harness)
```

#### Story 2.2: Implement Backward Transfer Measurement
```yaml
ID: STORY-2-2
Title: Implement Backward Transfer Measurement & Forgetting Quantification
Epic: EPIC 2
Priority: P0
Effort: 4 story points
Duration: 3 days

Description:
As a researcher, I need to quantify how much the model forgets previous languages
after learning new ones using the methodology from Bhavsar et al. (2023).

Acceptance Criteria:
- [ ] Backward Transfer (BWT) metric computed correctly per paper definition
- [ ] BWT formula: average performance change on previous tasks
- [ ] Metrics saved per-task and aggregated
- [ ] Comparison with baseline (independent training) created
- [ ] Visual forgetting curves generated
- [ ] Statistical significance computed

Technical Implementation:

File: src/metrics/forgetting_metrics.py

```python
def compute_backward_transfer(model, task_id, all_tasks, test_sets, checkpoints):
    """
    Measure forgetting on previously learned tasks.

    From Bhavsar et al. 2023:
    BWT = (1/T) * sum(performance_after - performance_right_after)

    Args:
        model: Current model
        task_id: Current task index
        all_tasks: All tasks trained so far
        test_sets: Test sets for all tasks
        checkpoints: Saved checkpoints after each task

    Returns:
        bwt: Backward transfer score (negative = forgetting)
    """
    if task_id == 0:
        return 0.0

    forgetting_scores = []

    for prev_task_id in range(task_id):
        # Load performance right after training on this task
        checkpoint = checkpoints[prev_task_id]
        original_perf = checkpoint['performance'][all_tasks[prev_task_id]]

        # Evaluate current model on this task
        current_perf = evaluate(model, test_sets[all_tasks[prev_task_id]])

        # Forgetting = degradation from original
        forgetting = current_perf - original_perf
        forgetting_scores.append(forgetting)

    bwt = np.mean(forgetting_scores)
    return bwt

def compute_forward_transfer(model, task_id, tasks, test_sets):
    """
    Measure forward transfer: performance on new task before training.

    FWT = performance_zero_shot - baseline_random
    """
    # Evaluate untrained model on task_id
    zero_shot_perf = evaluate(model, test_sets[tasks[task_id]])

    # Baseline is random guessing for task
    num_classes = get_num_classes(tasks[task_id])
    baseline_random = 1.0 / num_classes

    fwt = zero_shot_perf - baseline_random
    return fwt

def compute_average_accuracy(model, all_tasks, test_sets):
    """Average performance across all tasks."""
    accuracies = []
    for task in all_tasks:
        acc = evaluate(model, test_sets[task])
        accuracies.append(acc)
    return np.mean(accuracies)
```

Metrics Output Format:
```json
{
  "task_sequence": ["Hindi", "Marathi", "Tamil", "Telugu", "Bangla"],
  "metrics_per_language": {
    "Hindi": {
      "performance_at_end": 0.78,
      "performance_right_after": 0.85,
      "forgetting": -0.07
    },
    "Marathi": {
      "performance_at_end": 0.76,
      "performance_right_after": 0.81,
      "forgetting": -0.05
    }
  },
  "summary": {
    "BWT": -0.063,
    "FWT_avg": 0.12,
    "Average_Accuracy": 0.77
  }
}
```

Dependencies:
- Story 2.1 (Sequential trainer)
- Story 1.5 (Evaluation harness)
```

#### Story 2.3: Implement LR ADJUST Learning Rate Scheduling
```yaml
ID: STORY-2-3
Title: Implement LR ADJUST Learning Rate Scheduling Algorithm
Epic: EPIC 2
Priority: P0
Effort: 3 story points
Duration: 2 days

Description:
As a researcher, I need to implement the LR ADJUST algorithm from Bhavsar et al.
so that I can reduce catastrophic forgetting through adaptive learning rates.

Acceptance Criteria:
- [ ] LR ADJUST algorithm matches paper specification exactly
- [ ] Learning rate adjusts based on observed forgetting
- [ ] Implementation handles first task (no forgetting history)
- [ ] Comparison with fixed LR shows 15-25% improvement
- [ ] Algorithm configurable for different decay patterns

Technical Implementation:

File: src/training/lr_adjust.py

```python
class LRAdjustScheduler:
    """
    Implements LR ADJUST from Bhavsar et al. 2023.

    Key insight: Reduce learning rate for new tasks to minimize overwriting
    of previously learned knowledge.
    """

    def __init__(self, base_lr=2e-5, decay_factor_fn=None):
        self.base_lr = base_lr
        self.task_id = 0
        self.decay_factor_fn = decay_factor_fn or self.default_decay
        self.history = {}

    def get_lr(self, task_id, forgetting_rate=0.0):
        """
        Compute adjusted learning rate for task.

        Formula: lr_t = base_lr * 1 / (1 + task_id * forgetting_rate)

        Intuition:
        - Task 0 (first): lr = base_lr (full learning)
        - Task 1+: lr decreases based on observed forgetting
        - Higher forgetting → lower lr (more conservative)
        """
        if task_id == 0:
            adjusted_lr = self.base_lr
        else:
            decay = self.decay_factor_fn(task_id, forgetting_rate)
            adjusted_lr = self.base_lr * decay

        self.history[task_id] = {
            'adjusted_lr': adjusted_lr,
            'forgetting_rate': forgetting_rate
        }

        return adjusted_lr

    def default_decay(self, task_id, forgetting_rate):
        """Default decay function from paper."""
        if forgetting_rate == 0:
            forgetting_rate = 0.01  # Small default
        return 1.0 / (1.0 + task_id * forgetting_rate)

    def exponential_decay(self, task_id, forgetting_rate):
        """Alternative: exponential decay."""
        decay_rate = 0.95
        return decay_rate ** task_id

    def log_decay(self, task_id, forgetting_rate):
        """Alternative: logarithmic decay."""
        return 1.0 / np.log(2 + task_id)

# Usage in training loop:
lr_scheduler = LRAdjustScheduler(base_lr=2e-5)

for lang_id, language in enumerate(['Hindi', 'Marathi', 'Tamil', 'Telugu', 'Bangla']):
    # Measure forgetting from previous languages
    if lang_id > 0:
        forgetting_rate = compute_backward_transfer(...)
    else:
        forgetting_rate = 0.0

    # Get adjusted learning rate
    current_lr = lr_scheduler.get_lr(lang_id, forgetting_rate)

    # Train with adjusted LR
    train(model, dataloaders[language], lr=current_lr)
```

Comparison Experiment:
```
Experiment 1: Fixed LR (baseline)
  - LR = 2e-5 for all languages
  - Expected BWT: ~-15% to -20%

Experiment 2: LR ADJUST
  - LR adjusted per formula
  - Expected BWT: ~-5% to -10% (25% improvement)

Report:
  - Table comparing BWT, FWT, AA
  - Learning rate schedule plot
  - Statistical significance (t-test)
```

Dependencies:
- Story 2.1, 2.2
```

#### Story 2.4: Implement Checkpoint Management System
```yaml
ID: STORY-2-4
Title: Implement Checkpoint Management for Multi-Language Sequences
Epic: EPIC 2
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need robust checkpoint management so that I can resume
experiments and maintain reproducibility.

Acceptance Criteria:
- [ ] Checkpoints saved after each language
- [ ] Checkpoint includes: model state, optimizer state, metrics, config
- [ ] Can resume training from any checkpoint
- [ ] Checkpoints versioned and timestamped
- [ ] Automatic cleanup of old checkpoints (configurable)
- [ ] Checkpoint loading handles version mismatches

Implementation:
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = {}

    def save_checkpoint(self, model, optimizer, epoch, metrics, language, lang_id):
        timestamp = datetime.now().isoformat()
        checkpoint = {
            'timestamp': timestamp,
            'language': language,
            'lang_id': lang_id,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        path = f"{self.checkpoint_dir}/checkpoint_lang{lang_id}_{language}_{timestamp}.pt"
        torch.save(checkpoint, path)

        self.cleanup_old_checkpoints()
        return path

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        return checkpoint

    def cleanup_old_checkpoints(self):
        # Keep only max_checkpoints recent ones
        pass
```

Dependencies:
- Story 2.1
```

#### Story 2.5: Create Performance Tracking Across Language Transitions
```yaml
ID: STORY-2-5
Title: Create Performance Tracking & Visualization Across Language Transitions
Epic: EPIC 2
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need visualizations of performance across language transitions
so that I can understand forgetting patterns.

Acceptance Criteria:
- [ ] Forgetting curve plots generated for each task
- [ ] Heatmap of performance across languages
- [ ] Learning rate schedule visualized
- [ ] Comparison plots (fixed LR vs. LR ADJUST)
- [ ] Summary statistics table created

Output Visualizations:
- forgetting_curves.png (line plot)
- performance_heatmap.png (heatmap)
- lr_schedule.png (learning rate over time)
- comparative_analysis.png

Dependencies:
- Stories 2.1-2.4
```

---

## EPIC 3: Linguistic Analysis & Transfer Language Selection

**Duration:** Month 3 (4 weeks)
**Priority:** P1 (High)
**Effort:** 8 story points
**Owner:** Secondary researcher
**Depends On:** EPIC 1 complete, EPIC 2 in progress

### Epic Goal
Integrate LangRank tool, compute linguistic distance metrics, and analyze relationship between linguistic proximity and transfer effectiveness (RQ2 & RQ3).

### Acceptance Criteria
- [ ] LangRank tool integrated and operational
- [ ] Linguistic distance matrix computed for all 5 languages
- [ ] Features include phylogenetic, typological, lexical, embedding-based
- [ ] Correlation between linguistic distance and transfer success analyzed
- [ ] Indo-Aryan vs. Dravidian family effects quantified
- [ ] Transfer language ranking generated
- [ ] Results visualized and documented

### User Stories

#### Story 3.1: Integrate LangRank Tool
```yaml
ID: STORY-3-1
Title: Integrate LangRank Tool for Linguistic Distance Analysis
Epic: EPIC 3
Priority: P1
Effort: 3 story points
Duration: 2 days

Description:
As a researcher, I need the LangRank tool integrated so that I can compute
linguistic distance metrics between languages.

Acceptance Criteria:
- [ ] LangRank repository cloned and dependencies installed
- [ ] lang2vec integrated for WALS features
- [ ] Setup script created for easy integration
- [ ] Test run on all 5 languages successful
- [ ] Output format documented

Installation & Setup:
```bash
# Clone repositories
git clone https://github.com/neulab/langrank.git
git clone https://github.com/antonisa/lang2vec.git

# Install dependencies
pip install -r langrank/requirements.txt

# Place in project
cp -r langrank src/linguistic_analysis/langrank
cp -r lang2vec src/linguistic_analysis/lang2vec
```

Integration:
```python
# src/linguistic_analysis/langrank_integration.py
import sys
sys.path.insert(0, 'src/linguistic_analysis/langrank')

from langrank import LangRank

langrank = LangRank(lang2vec_path='src/linguistic_analysis/lang2vec')
```

Dependencies:
- EPIC 1 complete
```

#### Story 3.2: Compute Linguistic Features for All Languages
```yaml
ID: STORY-3-2
Title: Compute Linguistic Features: Phylogenetic, Typological, Lexical
Epic: EPIC 3
Priority: P1
Effort: 3 story points
Duration: 3 days

Description:
As a researcher, I need linguistic features for Hindi, Marathi, Tamil, Telugu,
and Bangla so that I can analyze linguistic distance effects.

Acceptance Criteria:
- [ ] Phylogenetic features extracted (family, genetic distance)
- [ ] Typological features from WALS database retrieved (~100 features)
- [ ] Lexical features computed (cognate detection, word overlap)
- [ ] Embedding-based similarity computed
- [ ] Feature matrix saved as features_matrix.json
- [ ] Feature importance ranked

Feature Categories:

A. Phylogenetic Features:
   - Language family (Indo-Aryan=1, Dravidian=0)
   - Genetic distance score (0-1, from linguistic literature)
   - Shared proto-language distance

B. Typological Features (from WALS):
   - Word order (SOV, SVO, VSO, etc.)
   - Morphological type (agglutinative, fusional, isolating)
   - Noun-adjective order
   - Subject-object-verb patterns
   - ~100 WALS features total

C. Lexical Features:
   - Lexical overlap on common vocabulary
   - Cognate detection score
   - Word frequency correlation
   - Common loan words

D. Embedding-Based Features:
   - Cross-lingual word embedding similarity
   - Language model representation similarity
   - Sentence embedding alignment

Implementation:
```python
# src/linguistic_analysis/feature_extractor.py

class LinguisticFeatureExtractor:
    def __init__(self, languages):
        self.languages = languages
        self.features = {}

    def extract_all_features(self):
        phylogenetic = self.extract_phylogenetic()
        typological = self.extract_typological()
        lexical = self.extract_lexical()
        embedding = self.extract_embedding_based()

        return {
            'phylogenetic': phylogenetic,
            'typological': typological,
            'lexical': lexical,
            'embedding': embedding
        }

    def extract_phylogenetic(self):
        """Extract language family and genetic distance."""
        data = {
            'Hindi': {'family': 'Indo-Aryan', 'genetic_distance_to_bangla': 0.15},
            'Marathi': {'family': 'Indo-Aryan', 'genetic_distance_to_bangla': 0.18},
            'Tamil': {'family': 'Dravidian', 'genetic_distance_to_bangla': 0.65},
            'Telugu': {'family': 'Dravidian', 'genetic_distance_to_bangla': 0.68},
            'Bangla': {'family': 'Indo-Aryan', 'genetic_distance_to_bangla': 0.0}
        }
        return data

    def extract_typological(self):
        """Extract WALS typological features."""
        # Use lang2vec to get ~100 WALS features
        pass

    def extract_lexical(self):
        """Compute lexical overlap."""
        pass

    def extract_embedding_based(self):
        """Compute embedding similarity."""
        pass
```

Output:
```json
{
  "linguistic_features": {
    "Hindi": {
      "family": "Indo-Aryan",
      "genetic_distance": 0.15,
      "typological_features": [...],
      "lexical_overlap": 0.42,
      "embedding_similarity": 0.78
    },
    ...
  }
}
```

Dependencies:
- Story 3.1
```

#### Story 3.3: Build Linguistic Distance Matrix
```yaml
ID: STORY-3-3
Title: Build Linguistic Distance Matrix for All Language Pairs
Epic: EPIC 3
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need a comprehensive linguistic distance matrix so that
I can correlate linguistic proximity with transfer success.

Acceptance Criteria:
- [ ] Distance matrix computed for all 5×5 language pairs
- [ ] Multiple distance metrics (phylogenetic, typological, lexical, combined)
- [ ] Distance matrices visualized as heatmaps
- [ ] Symmetry verified (distance(A,B) = distance(B,A))
- [ ] Output saved as CSV and JSON

Implementation:
```python
# src/linguistic_analysis/distance_matrix.py

import numpy as np
import pandas as pd

class LinguisticDistanceMatrix:
    def __init__(self, features):
        self.features = features
        self.languages = list(features.keys())

    def compute_distance_matrix(self, metric='combined'):
        """
        Compute distance matrix using specified metric.

        Metrics:
        - phylogenetic: genetic distance only
        - typological: WALS feature distance
        - lexical: word overlap distance
        - combined: weighted combination
        """
        n_langs = len(self.languages)
        distance_matrix = np.zeros((n_langs, n_langs))

        for i, lang1 in enumerate(self.languages):
            for j, lang2 in enumerate(self.languages):
                if i == j:
                    distance_matrix[i][j] = 0.0
                elif i < j:
                    dist = self.compute_pairwise_distance(
                        lang1, lang2, metric
                    )
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist  # Symmetric

        return pd.DataFrame(
            distance_matrix,
            index=self.languages,
            columns=self.languages
        )

    def compute_pairwise_distance(self, lang1, lang2, metric):
        if metric == 'phylogenetic':
            return self.features[lang1]['genetic_distance']
        elif metric == 'typological':
            # Euclidean distance on WALS features
            f1 = np.array(self.features[lang1]['typological'])
            f2 = np.array(self.features[lang2]['typological'])
            return np.linalg.norm(f1 - f2)
        # ... other metrics

    def save_matrices(self, output_dir):
        for metric in ['phylogenetic', 'typological', 'lexical', 'combined']:
            df = self.compute_distance_matrix(metric)
            df.to_csv(f"{output_dir}/distance_matrix_{metric}.csv")
```

Output Format:
```
Distance Matrix (Combined):
        Hindi  Marathi  Tamil  Telugu  Bangla
Hindi     0.0     0.08   0.65    0.68    0.15
Marathi   0.08    0.0    0.67    0.70    0.18
Tamil     0.65    0.67   0.0     0.15    0.65
Telugu    0.68    0.70   0.15    0.0     0.68
Bangla    0.15    0.18   0.65    0.68    0.0
```

Visualizations:
- heatmap_phylogenetic.png
- heatmap_typological.png
- heatmap_lexical.png
- heatmap_combined.png

Dependencies:
- Story 3.2
```

#### Story 3.4: Analyze Indo-Aryan vs. Dravidian Family Effects
```yaml
ID: STORY-3-4
Title: Analyze Indo-Aryan vs. Dravidian Language Family Effects
Epic: EPIC 3
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need to quantify how language family affects transfer
effectiveness for my RQ3 analysis.

Acceptance Criteria:
- [ ] Transfer effectiveness grouped by language family pairing
- [ ] Statistical comparison (Indo-Aryan→Bangla vs. Dravidian→Bangla)
- [ ] Effect size computed (Cohen's d or similar)
- [ ] Visualization showing family effects

Analysis:

```python
# src/linguistic_analysis/family_effects.py

class FamilyEffectsAnalysis:
    def __init__(self, transfer_results, linguistic_distance):
        self.transfer_results = transfer_results
        self.linguistic_distance = linguistic_distance

    def analyze_family_effects(self):
        """
        Compare transfer effectiveness by language family.

        Groups:
        - Indo-Aryan to Bangla (within-family): Hindi, Marathi → Bangla
        - Dravidian to Bangla (cross-family): Tamil, Telugu → Bangla
        """
        within_family = {}
        cross_family = {}

        for source_lang, results in self.transfer_results.items():
            if source_lang in ['Hindi', 'Marathi']:
                # Within Indo-Aryan family
                within_family[source_lang] = results
            else:
                # Cross to Dravidian
                cross_family[source_lang] = results

        within_f1 = np.mean([r['final_f1'] for r in within_family.values()])
        cross_f1 = np.mean([r['final_f1'] for r in cross_family.values()])

        effect_size = self.compute_cohens_d(
            within_family.values(),
            cross_family.values()
        )

        return {
            'within_family_avg_f1': within_f1,
            'cross_family_avg_f1': cross_f1,
            'effect_size': effect_size,
            'statistical_significance': self.t_test(within_family, cross_family)
        }
```

Report Output:
```
Language Family Effects on Transfer to Bangla:

Within Indo-Aryan Family (Hindi, Marathi → Bangla):
  - Average Forward Transfer: +4.2%
  - Average Backward Transfer: -2.1%
  - Mean F1 Score: 0.82

Cross to Dravidian Family (Tamil, Telugu → Bangla):
  - Average Forward Transfer: +1.3%
  - Average Backward Transfer: -8.5%
  - Mean F1 Score: 0.76

Effect Size (Cohen's d): 0.65 (medium effect)
Statistical Significance: p < 0.05 ✓

Interpretation: Within-family transfer is significantly more effective
than cross-family transfer, with a medium effect size.
```

Dependencies:
- Stories 3.1-3.3, EPIC 2 (experimental results)
```

#### Story 3.5: Correlate Linguistic Distance with Transfer Success
```yaml
ID: STORY-3-5
Title: Correlate Linguistic Distance with Empirical Transfer Effectiveness
Epic: EPIC 3
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need to validate that linguistic distance predicts
transfer success, answering RQ2.

Acceptance Criteria:
- [ ] Pearson correlation computed between distance and transfer success
- [ ] Multiple regression model analyzing feature importance
- [ ] Scatter plots with trend lines
- [ ] Statistical significance and R² reported
- [ ] Findings compared with Lin et al. (2019) expectations

Implementation:
```python
# src/linguistic_analysis/correlation_analysis.py

from scipy import stats
import numpy as np

class CorrelationAnalysis:
    def __init__(self, linguistic_distances, transfer_results):
        self.distances = linguistic_distances
        self.results = transfer_results

    def compute_correlation(self):
        """
        Compute correlation between linguistic distance and transfer effectiveness.
        """
        distance_pairs = []
        transfer_effectiveness = []

        for source_lang, empirical_f1 in self.results.items():
            if source_lang != 'Bangla':
                ling_dist = self.distances.loc[source_lang, 'Bangla']
                distance_pairs.append(ling_dist)
                transfer_effectiveness.append(empirical_f1)

        # Pearson correlation
        correlation, p_value = stats.pearsonr(distance_pairs, transfer_effectiveness)

        # R² from linear regression
        z = np.polyfit(distance_pairs, transfer_effectiveness, 1)
        p = np.poly1d(z)
        y_pred = p(distance_pairs)
        ss_res = np.sum((np.array(transfer_effectiveness) - y_pred) ** 2)
        ss_tot = np.sum((np.array(transfer_effectiveness) - np.mean(transfer_effectiveness)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'correlation': correlation,
            'p_value': p_value,
            'r_squared': r_squared,
            'regression_coefficients': z
        }

    def plot_correlation(self):
        """Generate scatter plot with trend line."""
        # Implementation for visualization
        pass
```

Output Plots:
- correlation_distance_vs_transfer.png (scatter + trend line)
- correlation_matrix.png (multiple features)

Dependencies:
- Stories 3.1-3.4
```

---

## EPIC 4: Parameter-Efficient Fine-Tuning with LoRA

**Duration:** Month 4-5 (4 weeks)
**Priority:** P1 (High)
**Effort:** 6 story points
**Owner:** Primary researcher
**Depends On:** EPIC 2 complete

### Epic Goal
Implement LoRA-based continual learning for parameter-efficient sequential training of Bangla models.

### Acceptance Criteria
- [ ] LoRA integration complete and operational
- [ ] Parameter efficiency validated (80%+ reduction)
- [ ] Performance comparable to full fine-tuning
- [ ] LoRA parameters optimized per language
- [ ] Memory usage significantly reduced
- [ ] Training speed improved

### User Stories

#### Story 4.1: Integrate LoRA into Training Pipeline
```yaml
ID: STORY-4-1
Title: Integrate LoRA (Low-Rank Adaptation) into Training Pipeline
Epic: EPIC 4
Priority: P1
Effort: 3 story points
Duration: 2 days

Description:
As a researcher, I need LoRA integrated so that I can train models with
significantly fewer parameters.

Acceptance Criteria:
- [ ] PEFT library integrated
- [ ] LoRA layers added to BanglaBERT
- [ ] Training code modified to use LoRA
- [ ] Checkpoint management handles LoRA weights
- [ ] Test run on sample data successful

Implementation:
```python
# src/training/lora_trainer.py
from peft import get_peft_model, LoraConfig

class LoRATrainer:
    def __init__(self, base_model, lora_config):
        self.base_model = base_model
        self.lora_config = lora_config
        self.model = self.setup_lora()

    def setup_lora(self):
        """Setup LoRA configuration."""
        lora_config = LoraConfig(
            r=self.lora_config['rank'],
            lora_alpha=self.lora_config['alpha'],
            target_modules=['query', 'value'],
            lora_dropout=0.05,
            bias='none'
        )

        model = get_peft_model(self.base_model, lora_config)
        model.print_trainable_parameters()
        return model

    def get_trainable_params(self):
        """Return number of trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total
```

Output:
```
LoRA Configuration:
  - Rank: 8
  - Alpha: 16
  - Target modules: query, value

Trainable Parameters:
  - LoRA params: 245,760 (0.5% of model)
  - Total params: 110M
  - Reduction: 99.5%
```

Dependencies:
- EPIC 2 complete
```

#### Story 4.2: Compare LoRA vs. Full Fine-Tuning
```yaml
ID: STORY-4-2
Title: Compare LoRA vs. Full Fine-Tuning Performance and Efficiency
Epic: EPIC 4
Priority: P1
Effort: 3 story points
Duration: 3 days

Description:
As a researcher, I need to validate that LoRA achieves comparable performance
with significantly reduced parameters and memory.

Acceptance Criteria:
- [ ] LoRA training on all 4 tasks completed
- [ ] Performance within 1-2% of full fine-tuning
- [ ] Memory usage reduced by 80%+
- [ ] Training time improved or comparable
- [ ] Comparison table generated

Comparison Study:

```python
# Compare on task 1 (Sentiment Analysis)
# Full fine-tuning: 110M parameters
# LoRA: 245K parameters

Results:
  Full Fine-Tuning:
    - Accuracy: 0.785
    - Memory: 8.2 GB
    - Time per epoch: 45 sec

  LoRA (r=8):
    - Accuracy: 0.778 (-0.7%)
    - Memory: 1.2 GB (85% reduction)
    - Time per epoch: 28 sec (38% faster)

  LoRA (r=16):
    - Accuracy: 0.782 (-0.3%)
    - Memory: 1.8 GB (78% reduction)
    - Time per epoch: 32 sec (29% faster)
```

Conclusion: LoRA with r=8 provides best trade-off between efficiency
and performance for this task.

Dependencies:
- Story 4.1
```

#### Story 4.3: Optimize LoRA Ranks and Alpha Parameters
```yaml
ID: STORY-4-3
Title: Optimize LoRA Rank and Alpha Parameters Per Language
Epic: EPIC 4
Priority: P2
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I want to find optimal LoRA hyperparameters for each language
to maximize performance efficiency trade-off.

Acceptance Criteria:
- [ ] Grid search over rank (4, 8, 16) and alpha (8, 16, 32)
- [ ] Per-language optimal parameters identified
- [ ] Configuration saved and used in subsequent experiments
- [ ] Report showing parameter sensitivity

Grid Search Results:
```
Hindi (Early in sequence):
  - Best: r=8, alpha=16 (F1=0.82)
  - Rank sensitivity: Low
  - Alpha sensitivity: Medium

Bangla (Final language, most parameters learned):
  - Best: r=16, alpha=32 (F1=0.81)
  - Rank sensitivity: High
  - Alpha sensitivity: High

Interpretation: Earlier languages need less LoRA capacity;
later languages benefit from higher rank as more knowledge accumulated.
```

Dependencies:
- Stories 4.1, 4.2
```

---

## EPIC 5: Comprehensive Evaluation & Analysis

**Duration:** Month 6-7 (6 weeks)
**Priority:** P0 (Critical Path)
**Effort:** 10 story points
**Owner:** Full team
**Depends On:** EPIC 2, 3, 4 complete

### Epic Goal
Execute complete experimental suite with all curriculum paths, ablation studies, and comprehensive analysis of continual learning effects.

### Acceptance Criteria
- [ ] All curriculum paths (A/B/C) executed successfully
- [ ] Ablation studies completed (LR ADJUST, replay, LoRA)
- [ ] Backward/forward transfer metrics comprehensive
- [ ] Statistical significance validated
- [ ] Comparative analysis matrices generated
- [ ] Results reproducible on clean environment

### User Stories

#### Story 5.1: Execute Curriculum Path A
```yaml
ID: STORY-5-1
Title: Execute Curriculum Path A - Hindi → Marathi → Tamil → Telugu → Bangla
Epic: EPIC 5
Priority: P0
Effort: 2 story points
Duration: 3 days (compute time)

Description:
As a researcher, I need to execute the standard curriculum ordering to
establish baseline results for the full sequential learning setup.

Acceptance Criteria:
- [ ] Training completes for all 5 languages
- [ ] All metrics (BWT, FWT, AA) computed
- [ ] Results saved to results/curriculum_a/
- [ ] Visualizations generated
- [ ] Experiment reproducible with fixed seed

Curriculum Path A:
```
Language Sequence: Hindi → Marathi → Tamil → Telugu → Bangla

Method: LR ADJUST + Replay (10%)
Hyperparameters:
  - Base LR: 2e-5
  - Batch size: 32
  - Epochs per language: 3
  - Replay buffer: 10% of previous data
  - Seed: 42

Stages:
1. Train on Hindi SA dataset (5000 samples)
2. Train on Marathi SA dataset (4000 samples) + 10% Hindi replay
3. Train on Tamil SA dataset (3500 samples) + replay from 1,2
4. Train on Telugu SA dataset (3200 samples) + replay from 1,2,3
5. Train on Bangla SA dataset (6000 samples) + replay from 1,2,3,4

Evaluate at each stage on all language test sets
```

Expected Results:
```
Backward Transfer (BWT): -0.08 ± 0.03
Forward Transfer (FWT): +0.12 ± 0.04
Average Accuracy (AA): 0.76 ± 0.02
```

Output Files:
- curriculum_a_training_log.txt
- curriculum_a_metrics.json
- curriculum_a_checkpoints/ (all language checkpoints)
- curriculum_a_plots/ (forgetting curves, heatmaps)

Dependencies:
- EPIC 2, 3 complete
```

#### Story 5.2: Execute Curriculum Path B
```yaml
ID: STORY-5-2
Title: Execute Curriculum Path B - Linguistically Optimized Ordering
Epic: EPIC 5
Priority: P0
Effort: 2 story points
Duration: 3 days

Description:
As a researcher, I need to execute an alternative curriculum based on
linguistic distance analysis from EPIC 3.

Acceptance Criteria:
- [ ] Path B ordering derived from linguistic analysis
- [ ] Training completes successfully
- [ ] Performance compared with Path A
- [ ] Improvement/degradation quantified

Curriculum Path B (Optimized Ordering):
```
Rationale: Order languages by linguistic distance to Bangla
(closest first, farthest last)

Proposed Order: Bangla (start) → Hindi → Marathi → Telugu → Tamil

Linguistic Proximity to Bangla:
1. Bangla (same language, 0.0)
2. Hindi (Indo-Aryan, distance 0.15)
3. Marathi (Indo-Aryan, distance 0.18)
4. Telugu (Dravidian, distance 0.68)
5. Tamil (Dravidian, distance 0.65)

Hypothesis: Training from linguistically similar to dissimilar
languages might better preserve knowledge and improve transfer.
```

Comparison with Path A:
```
Metric          Path A      Path B      Difference
BWT             -0.080      -0.055      +0.025 (25% improvement)
FWT             +0.120      +0.135      +0.015 (12.5% improvement)
AA              0.760       0.775       +0.015 (2% improvement)
```

Output Files:
- curriculum_b_training_log.txt
- curriculum_b_metrics.json
- curriculum_b_checkpoints/
- curriculum_b_plots/

Dependencies:
- EPIC 3 complete
```

#### Story 5.3: Execute Curriculum Path C - Reverse Order
```yaml
ID: STORY-5-3
Title: Execute Curriculum Path C - Reverse Ordering
Epic: EPIC 5
Priority: P0
Effort: 2 story points
Duration: 3 days

Description:
As a researcher, I need to test curriculum path C (reverse ordering) to
understand if starting with distant languages affects learning.

Acceptance Criteria:
- [ ] Reverse ordering executed
- [ ] Results compared with A and B
- [ ] Statistical significance tested

Curriculum Path C (Reverse Order):
```
Rationale: Start with linguistically distant languages,
end with close relatives

Order: Tamil → Telugu → Marathi → Hindi → Bangla

Hypothesis: Starting with diverse languages might improve
generalization, but could increase forgetting.
```

Expected Results: Path C should show higher forgetting due to
greater linguistic distance in early training, but potentially
better generalization.

Dependencies:
- EPIC 2, 3
```

#### Story 5.4: Run Ablation Studies
```yaml
ID: STORY-4-4
Title: Run Ablation Studies - Isolate Effect of Each Component
Epic: EPIC 5
Priority: P1
Effort: 3 story points
Duration: 4 days

Description:
As a researcher, I need ablation studies to quantify the contribution of
each continual learning component.

Acceptance Criteria:
- [ ] Baseline: Fixed LR, no replay, no LoRA
- [ ] +LR ADJUST: Dynamic LR only
- [ ] +Replay: LR ADJUST + 10% replay
- [ ] +LoRA: Replay + LoRA adaptation
- [ ] Statistical significance for each addition

Ablation Study Design:

```
Configuration 1: Baseline
  - Fixed LR: 2e-5
  - No replay
  - Full fine-tuning
  - Expected BWT: -0.20 (high forgetting)

Configuration 2: +LR ADJUST
  - Dynamic LR schedule
  - No replay
  - Full fine-tuning
  - Expected improvement: 15-25% in BWT

Configuration 3: +Replay
  - Dynamic LR
  - 10% replay buffer
  - Full fine-tuning
  - Expected improvement: Additional 10-15%

Configuration 4: +LoRA
  - Dynamic LR
  - 10% replay
  - LoRA (r=8)
  - Expected: Similar BWT, 80% less memory

Results Table:
```

| Method              | BWT    | FWT    | AA    | Parameters | Memory |
|-------------------|--------|--------|-------|-----------|--------|
| Baseline           | -0.195 | +0.105 | 0.74  | 110M      | 8.2GB  |
| +LR ADJUST         | -0.155 | +0.118 | 0.76  | 110M      | 8.2GB  |
| +Replay            | -0.125 | +0.125 | 0.77  | 110M      | 8.2GB  |
| +LoRA              | -0.128 | +0.122 | 0.76  | 245K      | 1.2GB  |

Conclusion: Each component contributes ~5-7% improvement in BWT
```

Dependencies:
- Stories 5.1-5.3
```

#### Story 5.5: Generate Comparative Analysis & Statistical Tests
```yaml
ID: STORY-5-5
Title: Generate Comparative Analysis & Statistical Significance Tests
Epic: EPIC 5
Priority: P1
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need statistical analysis to validate findings and
ensure they are not due to random variation.

Acceptance Criteria:
- [ ] T-tests comparing curriculum paths (A vs B, A vs C)
- [ ] ANOVA across all methods
- [ ] Effect sizes computed (Cohen's d)
- [ ] Confidence intervals reported
- [ ] Multiple comparisons correction applied (Bonferroni)

Statistical Analysis:

```python
# Comparing curriculum paths
from scipy import stats

# Path A vs Path B (BWT scores)
path_a_bwt = [-0.080, -0.075, -0.085]  # Multiple runs
path_b_bwt = [-0.055, -0.050, -0.060]

t_stat, p_value = stats.ttest_ind(path_a_bwt, path_b_bwt)
cohens_d = (np.mean(path_a_bwt) - np.mean(path_b_bwt)) / np.sqrt(
    (np.var(path_a_bwt) + np.var(path_b_bwt)) / 2
)

print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f} ({'small'|'medium'|'large'} effect)")
```

Output Report:
```
Statistical Significance Testing Results
==========================================

Path A vs Path B (BWT):
  - Mean difference: -0.025
  - t-statistic: 2.14
  - p-value: 0.045 *
  - Cohen's d: 0.72 (medium effect)
  - Conclusion: Path B significantly better (p < 0.05)

Path A vs Path C (BWT):
  - Mean difference: -0.055
  - t-statistic: 3.45
  - p-value: 0.008 **
  - Cohen's d: 1.23 (large effect)
  - Conclusion: Path A significantly better (p < 0.01)

ANOVA (Curriculum paths × Ablation methods):
  - F-statistic: 5.67
  - p-value: 0.002 **
  - Significant effects for both factors

* p < 0.05
** p < 0.01
```

Dependencies:
- Stories 5.1-5.4
```

---

## EPIC 6: Documentation & Thesis Writing

**Duration:** Month 8-9 (6 weeks)
**Priority:** P0 (Critical)
**Effort:** 8 story points
**Owner:** Full team
**Depends On:** EPIC 5 complete

### Epic Goal
Document methodology, results, and contributions; create thesis sections integrating research findings.

### Acceptance Criteria
- [ ] Methodology section complete with algorithm descriptions
- [ ] Results section with tables and figures
- [ ] Discussion section analyzing findings
- [ ] Contributions section clearly articulated
- [ ] Related work section incorporating paper insights
- [ ] Appendix with reproduction instructions
- [ ] Thesis draft sections integrated

### User Stories

#### Story 6.1: Document LR ADJUST Implementation in Methodology
```yaml
ID: STORY-6-1
Title: Document LR ADJUST Implementation in Thesis Methodology
Epic: EPIC 6
Priority: P0
Effort: 1 story point
Duration: 1 day

Description:
As a researcher, I need to document the LR ADJUST algorithm in the thesis
methodology section so readers understand the approach.

Acceptance Criteria:
- [ ] Algorithm pseudocode included
- [ ] Mathematical formulation explained
- [ ] Reference to Bhavsar et al. (2023) proper
- [ ] Adaptation to Bangla context explained

Content Template:

### Learning Rate Adjustment Strategy

We adopt the learning rate adjustment (LR ADJUST) strategy proposed by
Bhavsar et al. (2023) for massively multilingual continual learning. This
method dynamically adjusts the learning rate based on observed forgetting
during sequential language training.

**Algorithm:**
```
For each language L_t in sequence:
  1. Measure backward transfer (BWT) on previously trained languages
  2. Compute adjusted learning rate: lr_t = base_lr / (1 + t × BWT)
  3. Train on L_t with lr_t
  4. Save checkpoint and metrics
```

**Motivation:** By reducing the learning rate for new languages, we
balance learning new task knowledge with preserving previously learned
information, reducing catastrophic forgetting.

**Adaptation:** We apply this strategy to Indic language continual
learning, specifically examining transfer to Bangla from Hindi, Marathi,
Tamil, and Telugu.

Dependencies:
- EPIC 2 complete, EPIC 5 (results)
```

#### Story 6.2: Document Linguistic Analysis Findings
```yaml
ID: STORY-6-2
Title: Document Linguistic Analysis Findings in Related Work Section
Epic: EPIC 6
Priority: P0
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need to document the linguistic analysis methodology
and findings from EPIC 3.

Acceptance Criteria:
- [ ] LangRank methodology explained
- [ ] Linguistic features documented
- [ ] Linguistic distance analysis results included
- [ ] Family effects quantified
- [ ] Correlation findings reported

Content Structure:

### Linguistic Distance and Transfer Effectiveness

#### Methodology
We employ the LangRank framework (Lin et al., 2019) to analyze linguistic
distance between source and target languages. Linguistic proximity is
computed across four dimensions:

1. **Phylogenetic Distance:** Language family classification (Indo-Aryan
   vs. Dravidian)
2. **Typological Distance:** ~100 WALS features capturing morphological
   and syntactic properties
3. **Lexical Distance:** Cognate detection and word overlap analysis
4. **Embedding-Based Distance:** Cross-lingual word embedding similarity

#### Results

**Linguistic Distance Matrix:**
[Include distance matrix table/heatmap]

**Transfer Effectiveness by Language Pair:**
[Include transfer success rates by pair]

**Family Effects:**
Within-family transfer (Indo-Aryan → Bangla) shows X% higher accuracy
than cross-family transfer (Dravidian → Bangla), suggesting linguistic
proximity is a significant predictor of transfer success.

**Correlation Analysis:**
Pearson correlation between linguistic distance and empirical transfer
effectiveness: r = -0.72 (p < 0.05), confirming that greater linguistic
similarity predicts better transfer.

Dependencies:
- EPIC 3 complete, Story 6.1
```

#### Story 6.3: Create Results Tables and Figures
```yaml
ID: STORY-6-3
Title: Create Results Tables and Figures for Thesis
Epic: EPIC 6
Priority: P0
Effort: 2 story points
Duration: 2 days

Description:
As a researcher, I need publication-ready tables and figures summarizing
experimental results.

Acceptance Criteria:
- [ ] Main results table (BWT, FWT, AA across methods)
- [ ] Curriculum comparison table
- [ ] Ablation study summary table
- [ ] Forgetting curves figure
- [ ] Performance heatmaps
- [ ] Comparative bar charts

Table 1: Main Results Summary
```
| Method           | Curriculum | BWT   | FWT   | AA    | Memory |
|-----------------|-----------|-------|-------|-------|--------|
| Baseline        | A         | -20.0%| +10.5%| 74.2% | 8.2GB  |
| LR ADJUST       | A         | -15.5%| +11.8%| 76.1% | 8.2GB  |
| +Replay         | A         | -12.5%| +12.5%| 77.3% | 8.2GB  |
| +LoRA           | A         | -12.8%| +12.2%| 76.4% | 1.2GB  |
| LR ADJUST+Reply | B         | -5.5% | +13.5%| 77.5% | 8.2GB  |
| LR ADJUST+Reply | C         | -18.0%| +9.5% | 75.8% | 8.2GB  |
```

Figure 1: Forgetting Curves
[Line plot showing performance degradation across language sequence]

Figure 2: Performance Heatmap
[Heatmap of accuracy across all language pairs and methods]

Figure 3: Ablation Study Results
[Bar chart comparing BWT improvement for each component]

Dependencies:
- EPIC 5 complete
```

#### Story 6.4: Draft Contributions Section
```yaml
ID: STORY-6-4
Title: Draft Research Contributions Section
Epic: EPIC 6
Priority: P0
Effort: 1 story point
Duration: 1 day

Description:
As a researcher, I need to clearly articulate the novel contributions
of this research beyond existing work.

Acceptance Criteria:
- [ ] Contributions listed and explained
- [ ] Comparison with related work shown
- [ ] Novelty clearly articulated

Content Template:

### Research Contributions

This thesis makes the following contributions to continual learning for
cross-lingual NLP:

**1. Application of Continual Learning to Indic Languages**
- First systematic study of sequential transfer learning for Bangla from
  related Indic languages (Hindi, Marathi, Tamil, Telugu)
- Demonstrates that catastrophic forgetting is a significant challenge
  even when transfer languages are linguistically related
- Proposes practical solutions (LR ADJUST, replay) for this setting

**2. Quantification of Language Family Effects**
- Empirically demonstrates that within-family transfer (Indo-Aryan →
  Bangla) is X% more effective than cross-family transfer (Dravidian →
  Bangla)
- Validates linguistic distance as predictor of transfer success, extending
  Lin et al. (2019) to continual learning setting

**3. Optimized Curriculum Design**
- Shows that curriculum ordering based on linguistic proximity (Path B)
  outperforms standard ordering (Path A) by 25-30% in reducing forgetting
- Provides practical guidance for transfer language selection

**4. Parameter-Efficient Continual Learning**
- Demonstrates that LoRA-based fine-tuning maintains competitive
  performance while reducing parameters by 99.5% and memory by 80%
- Enables continual learning on resource-constrained devices

### Novelty Statement

While Bhavsar et al. (2023) address multilingual continual learning,
they do not:
- Focus on linguistically motivated sequential transfer to specific languages
- Analyze language family effects
- Propose curriculum optimization based on linguistic similarity
- Integrate parameter-efficient methods with continual learning

This work fills these gaps through systematic empirical study.

Dependencies:
- EPIC 5 complete
```

#### Story 6.5: Create Reproduction Instructions and Appendix
```yaml
ID: STORY-6-5
Title: Create Appendix with Complete Reproduction Instructions
Epic: EPIC 6
Priority: P1
Effort: 1 story point
Duration: 1 day

Description:
As a researcher, I need complete reproducibility instructions so readers
can verify and build on this work.

Acceptance Criteria:
- [ ] Step-by-step setup instructions included
- [ ] Hyperparameters documented
- [ ] Dataset links provided
- [ ] Command-line examples given
- [ ] Expected outputs described
- [ ] Computation requirements listed

Appendix Structure:

### Appendix A: Reproducibility Instructions

#### A.1 Environment Setup
```bash
# Clone repository
git clone <repo-url>
cd BanglaContinualLearning

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download BanglaBERT
python scripts/download_models.py
```

#### A.2 Dataset Acquisition
- Sentiment Analysis: [Link to dataset]
- NER: [Link to dataset]
- NLI: [Link to dataset]
- News Classification: [Link to dataset]

Place in `data/` directory.

#### A.3 Running Experiments

Curriculum Path A:
```bash
python experiments/run_curriculum.py \
  --config configs/curriculum_a.yaml \
  --output results/curriculum_a
```

Full Ablation:
```bash
python experiments/run_ablation.py \
  --curriculum_path a \
  --output results/ablation_full
```

#### A.4 Computation Requirements
- GPU: NVIDIA A100 40GB (or equivalent)
- CPU: 32+ cores recommended
- Storage: 200GB for datasets + models
- Time: ~48 hours for full experimental suite

#### A.5 Hyperparameters

Table A.1: Training Hyperparameters
```
Base Learning Rate: 2e-5
Batch Size: 32
Epochs per Language: 3
Replay Buffer Size: 10%
LoRA Rank: 8
LoRA Alpha: 16
Max Sequence Length: 256
Dropout: 0.1
```

Dependencies:
- All EPIC 6 stories
```

---

## Dependencies & Timeline

### Epic Dependency Graph

```
EPIC 1: Infrastructure & Baseline
    ↓
EPIC 2: Core CL Implementation
    ├─→ EPIC 3: Linguistic Analysis
    ├─→ EPIC 4: LoRA Implementation
    ↓
EPIC 5: Comprehensive Evaluation
    ↓
EPIC 6: Documentation & Writing
```

### Timeline Overview

```
Month 1: EPIC 1 (Infrastructure)
Month 2: EPIC 2 (Core CL)
Month 3: EPIC 3 (Linguistic Analysis) + EPIC 2 continuation
Month 4-5: EPIC 4 (LoRA) in parallel with EPIC 2/3
Month 6-7: EPIC 5 (Evaluation & Analysis)
Month 8-9: EPIC 6 (Documentation & Writing)
```

---

## Success Metrics

### Quantitative Metrics

1. **Backward Transfer Improvement:**
   - Target: 15-25% reduction in forgetting vs. baseline
   - Minimum: 10% improvement
   - Measured by: BWT score

2. **Forward Transfer:**
   - Target: Positive forward transfer (+10-15%)
   - Measured by: Zero-shot performance before training

3. **Linguistic Distance Correlation:**
   - Target: r > 0.60 correlation with transfer effectiveness
   - Statistical significance: p < 0.05

4. **Parameter Efficiency (LoRA):**
   - Target: 80%+ parameter reduction
   - Performance loss: < 2%

5. **Curriculum Optimization:**
   - Target: Path B/C 20-30% better than Path A
   - Statistical significance: p < 0.05

### Qualitative Metrics

1. **Reproducibility:**
   - All experiments reproducible with fixed seed
   - Code and data publicly available
   - Documentation complete

2. **Thesis Quality:**
   - Clear problem statement and contributions
   - Rigorous experimental design
   - Proper statistical analysis
   - Clear writing and presentation

3. **Research Impact:**
   - Findings actionable for practitioners
   - Extends existing theory (Bhavsar et al., Lin et al.)
   - Opens avenues for future work

---

## Final Checklist

### Before Starting EPIC 1
- [ ] Team members assigned
- [ ] Development environment tested
- [ ] Literature review completed
- [ ] Dataset access confirmed
- [ ] GitHub repository initialized

### Monthly Checkpoints
- [ ] EPIC 1 Month 1: Infrastructure complete, baseline established
- [ ] EPIC 2 Month 2: Sequential training working, LR ADJUST implemented
- [ ] EPIC 3 Month 3: Linguistic analysis complete, correlations computed
- [ ] EPIC 4 Month 4-5: LoRA integrated, efficiency validated
- [ ] EPIC 5 Month 6-7: All experiments complete, statistical tests done
- [ ] EPIC 6 Month 8-9: Thesis sections drafted, all documentation complete

---

## Conclusion

This document provides a comprehensive roadmap for executing the thesis research on continual learning for cross-lingual transfer to Bangla. By organizing work into 6 epics with detailed user stories, the team has clear objectives, acceptance criteria, and technical specifications for each phase.

Success requires disciplined execution, regular progress tracking, and iterative feedback. The dependencies between epics should be respected to maintain critical path and prevent rework.

Good luck with your research! 🚀

---

**Document Prepared By:** John (Product Manager)
**Date:** 2025-10-21
**Status:** Ready for Team Review
