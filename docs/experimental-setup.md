# Experimental Setup and Execution Plan

**Document Version:** 1.0
**Created:** 2025-10-23
**Purpose:** Complete breakdown of which models train on which languages/tasks and experiment execution order

---

## Table of Contents

1. [Understanding the Setup](#understanding-the-setup)
2. [Model-Language-Task Mapping](#model-language-task-mapping)
3. [Complete List of Experiments (In Order)](#complete-list-of-experiments-in-order)
4. [Experimental Conditions](#experimental-conditions)
5. [Detailed Execution Timeline](#detailed-execution-timeline)

---

## Understanding the Setup

### Key Concept: Sequential Continual Learning

**Important:** Each model is trained on **ALL 4 tasks** for each language **sequentially**. This is NOT about different models doing different tasks - it's about ONE model learning multiple languages one-by-one without forgetting.

### The Training Sequence

```
IndicBERT (starting model)
    ↓
Train on Hindi (all 4 tasks: SA, NER, NLI, News)
    ↓
Train on Marathi (all 4 tasks) + Remember Hindi
    ↓
Train on Tamil (all 4 tasks) + Remember Hindi & Marathi
    ↓
Train on Telugu (all 4 tasks) + Remember Hindi, Marathi & Tamil
    ↓
Train on Bangla (all 4 tasks) + Remember all previous languages
    ↓
FINAL MODEL (evaluated on Bangla tasks)
```

---

## Model-Language-Task Mapping

### Understanding: ONE Model, Multiple Languages, Sequential Training

| Model Start | Languages (Sequential Order) | Tasks (All Trained) | Final Target |
|-------------|------------------------------|---------------------|--------------|
| **IndicBERT** | Hindi → Marathi → Tamil → Telugu → **Bangla** | SA, NER, NLI, News (for each language) | Bangla Performance |

### What Each Model Does

#### 1. **IndicBERT (Sequential Continual Learning)**
- **Starting Point:** Pre-trained `ai4bharat/IndicBERT`
- **Training Path:**
  - **Step 1:** Hindi - Train on 4 tasks (SA, NER, NLI, News)
  - **Step 2:** Marathi - Train on 4 tasks (SA, NER, NLI, News)
  - **Step 3:** Tamil - Train on 4 tasks (SA, NER, NLI, News)
  - **Step 4:** Telugu - Train on 4 tasks (SA, NER, NLI, News)
  - **Step 5:** Bangla - Train on 4 tasks (SA, NER, NLI, News)
- **Goal:** Learn from 4 source languages sequentially, then perform well on Bangla
- **Challenge:** Prevent forgetting Hindi when learning Marathi, etc.

#### 2. **BanglaBERT (Baseline - No Sequential Training)**
- **Starting Point:** Pre-trained `csebuetnlp/banglabert`
- **Training Path:**
  - **Direct:** Train ONLY on Bangla (4 tasks: SA, NER, NLI, News)
- **Languages Used:** Bangla ONLY
- **Goal:** Upper bound performance (what a monolingual model can achieve)

#### 3. **IndicBERT Direct (Baseline - No Sequential Training)**
- **Starting Point:** Pre-trained `ai4bharat/IndicBERT`
- **Training Path:**
  - **Direct:** Train ONLY on Bangla (4 tasks: SA, NER, NLI, News)
- **Languages Used:** Bangla ONLY (no Hindi, Marathi, Tamil, Telugu)
- **Goal:** Compare against sequential approach - does sequential learning help?

#### 4. **mBERT (Baseline - No Sequential Training)**
- **Starting Point:** Pre-trained `bert-base-multilingual-cased`
- **Training Path:**
  - **Direct:** Train ONLY on Bangla (4 tasks)
- **Goal:** Standard multilingual baseline

#### 5. **XLM-R (Baseline - No Sequential Training)**
- **Starting Point:** Pre-trained `xlm-roberta-base`
- **Training Path:**
  - **Direct:** Train ONLY on Bangla (4 tasks)
- **Goal:** State-of-the-art multilingual baseline

---

## Complete List of Experiments (In Order)

### Phase 1: Baseline Experiments (Week 1-2)

These establish what performance to beat.

#### Experiment 1: **BanglaBERT Baseline**
```
Model: csebuetnlp/banglabert
Training: Direct fine-tuning on Bangla only
Tasks: SA, NER, NLI, News Classification
Purpose: Upper bound (monolingual performance)
Expected: ~79.54 F1 average (from paper)
```

#### Experiment 2: **IndicBERT Direct Baseline**
```
Model: ai4bharat/IndicBERT
Training: Direct fine-tuning on Bangla only (no sequential learning)
Tasks: SA, NER, NLI, News Classification
Purpose: Compare against sequential - does multi-language help?
Expected: ~70.40 F1 average (from paper)
```

#### Experiment 3: **mBERT Baseline**
```
Model: bert-base-multilingual-cased
Training: Direct fine-tuning on Bangla only
Tasks: SA, NER, NLI, News Classification
Purpose: Standard multilingual comparison
```

#### Experiment 4: **XLM-R Baseline**
```
Model: xlm-roberta-base
Training: Direct fine-tuning on Bangla only
Tasks: SA, NER, NLI, News Classification
Purpose: SOTA multilingual comparison
```

---

### Phase 2: Sequential Continual Learning - Core Methods (Week 3-6)

These are your main contributions.

#### Experiment 5: **Sequential Training (Naive - No Forgetting Prevention)**
```
Model: ai4bharat/IndicBERT
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
Tasks per Language: SA, NER, NLI, News (all 4)
Technique: Basic sequential fine-tuning (NO protection)
Purpose: Show how bad forgetting is without techniques
Expected: Significant forgetting of earlier languages
```

#### Experiment 6: **Sequential + Experience Replay**
```
Model: ai4bharat/IndicBERT
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
Tasks per Language: SA, NER, NLI, News (all 4)
Technique: Experience Replay (5% buffer from each language)
Purpose: Reduce forgetting by replaying past data
Expected: Better than naive, less forgetting
```

#### Experiment 7: **Sequential + LR ADJUST**
```
Model: ai4bharat/IndicBERT
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
Tasks per Language: SA, NER, NLI, News (all 4)
Technique: Learning rate scheduling based on forgetting
Purpose: Adaptive LR to prevent overwriting past knowledge
Expected: Comparable to Experience Replay
```

#### Experiment 8: **Sequential + Experience Replay + LR ADJUST (BEST)**
```
Model: ai4bharat/IndicBERT
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
Tasks per Language: SA, NER, NLI, News (all 4)
Technique: Combined approach (Replay + LR scheduling)
Purpose: Best performance - your main result
Expected: Closest to BanglaBERT, beats IndicBERT direct by ~7 F1 points
```

---

### Phase 3: Parameter-Efficient Methods (Week 7-8)

#### Experiment 9: **Sequential LoRA (No Replay)**
```
Model: ai4bharat/IndicBERT + LoRA adapters
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
Tasks per Language: SA, NER, NLI, News (all 4)
Technique: Language-specific LoRA adapters (rank=8)
Purpose: Reduce parameters, prevent forgetting via separate adapters
Expected: Good performance with 99.7% fewer trainable parameters
```

#### Experiment 10: **Sequential LoRA + Experience Replay (Hybrid)**
```
Model: ai4bharat/IndicBERT + LoRA adapters
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
Tasks per Language: SA, NER, NLI, News (all 4)
Technique: LoRA + small replay buffer
Purpose: Best of both worlds - efficiency + performance
Expected: Comparable to full fine-tuning with replay
```

---

### Phase 4: Curriculum Analysis (Week 9-10)

Test if language ORDER matters.

#### Experiment 11: **Path A - Close to Distant**
```
Model: ai4bharat/IndicBERT
Training Path: Hindi → Marathi → Tamil → Telugu → Bangla
             (Indo-Aryan → Indo-Aryan → Dravidian → Dravidian → Target)
Technique: Experience Replay + LR ADJUST
Purpose: Test close-to-distant curriculum
```

#### Experiment 12: **Path B - Distant to Close**
```
Model: ai4bharat/IndicBERT
Training Path: Tamil → Telugu → Hindi → Marathi → Bangla
             (Dravidian → Dravidian → Indo-Aryan → Indo-Aryan → Target)
Technique: Experience Replay + LR ADJUST
Purpose: Test distant-to-close curriculum
```

#### Experiment 13: **Path C - Mixed/Interleaved**
```
Model: ai4bharat/IndicBERT
Training Path: Hindi → Tamil → Marathi → Telugu → Bangla
             (Indo-Aryan → Dravidian → Indo-Aryan → Dravidian → Target)
Technique: Experience Replay + LR ADJUST
Purpose: Test alternating language families
```

---

### Phase 5: Joint Training Upper Bound (Week 11)

#### Experiment 14: **Joint Multilingual Training**
```
Model: ai4bharat/IndicBERT
Training: ALL languages mixed from the start (Hindi+Marathi+Tamil+Telugu+Bangla)
Tasks: SA, NER, NLI, News (all 4)
Purpose: Upper bound - best possible with all data available
Expected: Best multilingual performance (but unrealistic for continual learning)
```

---

## Experimental Conditions Summary

### Total Experiments: 14

| # | Experiment Name | Model | Method | Languages | Purpose |
|---|----------------|-------|--------|-----------|---------|
| 1 | BanglaBERT Baseline | BanglaBERT | Direct | Bangla only | Upper bound |
| 2 | IndicBERT Direct | IndicBERT | Direct | Bangla only | Baseline comparison |
| 3 | mBERT Baseline | mBERT | Direct | Bangla only | Standard baseline |
| 4 | XLM-R Baseline | XLM-R | Direct | Bangla only | SOTA baseline |
| 5 | Sequential Naive | IndicBERT | None | Hi→Mr→Ta→Te→Bn | Show forgetting |
| 6 | Sequential + Replay | IndicBERT | Experience Replay | Hi→Mr→Ta→Te→Bn | Reduce forgetting |
| 7 | Sequential + LR ADJUST | IndicBERT | Learning rate scheduling | Hi→Mr→Ta→Te→Bn | Adaptive learning |
| 8 | **Sequential + Replay + LR** | IndicBERT | **Combined (MAIN)** | Hi→Mr→Ta→Te→Bn | **Best approach** |
| 9 | Sequential LoRA | IndicBERT+LoRA | LoRA adapters | Hi→Mr→Ta→Te→Bn | Parameter-efficient |
| 10 | Sequential LoRA + Replay | IndicBERT+LoRA | LoRA + Replay | Hi→Mr→Ta→Te→Bn | Hybrid approach |
| 11 | Path A (Close→Distant) | IndicBERT | Replay + LR | Hi→Mr→Ta→Te→Bn | Curriculum test |
| 12 | Path B (Distant→Close) | IndicBERT | Replay + LR | Ta→Te→Hi→Mr→Bn | Curriculum test |
| 13 | Path C (Mixed) | IndicBERT | Replay + LR | Hi→Ta→Mr→Te→Bn | Curriculum test |
| 14 | Joint Multilingual | IndicBERT | Joint training | All mixed | Upper bound |

---

## Detailed Execution Timeline

### Week 1-2: Setup + Baselines
- Download all 20 datasets
- Setup training infrastructure
- Run Experiments 1-4 (all baselines)
- **Deliverable:** Baseline performance table

### Week 3-4: Core Sequential Methods
- Implement Experience Replay
- Implement LR ADJUST
- Run Experiments 5-8
- **Deliverable:** Forgetting curves, comparison charts

### Week 5-6: Analysis Phase 1
- Measure Backward Transfer (forgetting)
- Measure Forward Transfer (knowledge gain)
- Analyze which tasks/languages forget most
- **Deliverable:** Detailed analysis of forgetting patterns

### Week 7-8: Parameter-Efficient Methods
- Implement LoRA integration
- Run Experiments 9-10
- Compare memory/compute efficiency
- **Deliverable:** Efficiency vs. performance trade-off analysis

### Week 9-10: Curriculum Experiments
- Run Experiments 11-13 (Path A/B/C)
- Analyze linguistic distance correlations
- **Deliverable:** Curriculum ordering insights

### Week 11: Joint Training + Consolidation
- Run Experiment 14
- Create comprehensive comparison tables
- Generate all figures for paper
- **Deliverable:** Complete experimental results

### Week 12: Writing + Contingency
- Write thesis chapters
- Run additional ablations if needed
- Prepare defense presentation

---

## Evaluation Metrics (Applied to ALL Experiments)

For each experiment, measure:

1. **Final Bangla Performance:** F1 scores on 4 tasks
2. **Backward Transfer (BWT):** How much did earlier languages forget?
3. **Forward Transfer (FWT):** Did earlier languages help learn new ones?
4. **Average Accuracy (AA):** Overall performance across all languages
5. **Training Time:** Hours to complete
6. **Memory Usage:** GPU memory required

---

## Expected Results Summary

| Experiment | Expected Bangla F1 | Forgetting (BWT) | Key Insight |
|------------|-------------------|------------------|-------------|
| BanglaBERT | **79.54** (best) | N/A | Upper bound |
| IndicBERT Direct | 70.40 | N/A | Baseline to beat |
| Sequential Naive | ~65-68 | High (-15% to -20%) | Forgetting is real |
| Sequential + Replay | ~75-77 | Medium (-5% to -10%) | Replay helps |
| Sequential + LR ADJUST | ~74-76 | Medium (-5% to -10%) | LR scheduling helps |
| **Sequential + Both** | **~77-78** | Low (-2% to -5%) | **Best approach** |
| Sequential LoRA | ~74-76 | Low (-3% to -7%) | Efficient alternative |
| Path A vs B vs C | ~77-78 (varies) | Varies by order | Order matters |
| Joint Multilingual | ~78-80 | N/A | Unrealistic upper bound |

**Main Result:** Sequential CL (Experiment 8) should achieve ~77-78 F1, which is:
- **+7 points** better than IndicBERT Direct (70.40)
- **-2 points** below BanglaBERT (79.54)
- Demonstrates continual learning closes the gap between multilingual and monolingual

---

## Key Takeaways

1. **One Model (IndicBERT) learns 5 languages sequentially** - not 5 separate models
2. **All 4 tasks trained for each language** - comprehensive evaluation
3. **14 total experiments** - 4 baselines + 10 continual learning variants
4. **Main comparison:** Sequential CL vs. Direct Transfer vs. Monolingual
5. **Execution time:** ~11 weeks for full experimental suite

---

**Document Status:** Ready for implementation
**Next Steps:** Begin with baseline experiments (Week 1-2)
