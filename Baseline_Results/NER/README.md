# NER Cross-Lingual Transfer Experiments

## Overview

This folder contains Jupyter notebooks for Named Entity Recognition (NER) experiments to measure cross-lingual transfer learning effectiveness from source Indic languages to Bangla.

## Notebook Structure

### Training Setup
- **Model:** IndicBERT (ai4bharat/IndicBERTv2-MLM-only)
- **Dataset:** WikiANN
- **Training:** Source language (Hindi/Marathi/Tamil/Telugu)
- **Testing:** Bangla (target language for cross-lingual transfer evaluation)

### Available Notebooks

| Notebook | Training Language | Test Language | Purpose |
|----------|------------------|---------------|---------|
| [wikiann-indicbert_Bangla.ipynb](wikiann-indicbert_Bangla.ipynb) | Bangla | Bangla | Baseline (no transfer) |
| [wikiann-indicbert_Hindi.ipynb](wikiann-indicbert_Hindi.ipynb) | Hindi (hi) | Bangla | Cross-lingual transfer |
| [wikiann-indicbert_Marathi.ipynb](wikiann-indicbert_Marathi.ipynb) | Marathi (mr) | Bangla | Cross-lingual transfer |
| [wikiann-indicbert_Tamil.ipynb](wikiann-indicbert_Tamil.ipynb) | Tamil (ta) | Bangla | Cross-lingual transfer |
| [wikiann-indicbert_Telugu.ipynb](wikiann-indicbert_Telugu.ipynb) | Telugu (te) | Bangla | Cross-lingual transfer |
| [wikiann-banglabert.ipynb](wikiann-banglabert.ipynb) | Bangla | Bangla | BanglaBERT baseline |

## Experiment Goals

### Cross-Lingual Transfer Analysis
Each notebook (except Bangla baseline) measures:
1. **Source Language Performance:** How well the model learns the source language
2. **Cross-Lingual Transfer:** How well knowledge transfers to Bangla
3. **Transfer Gap:** Performance difference between source and target languages

### Expected Outcomes
- **High transfer:** Hindi/Marathi (Indo-Aryan languages, linguistically closer to Bangla)
- **Lower transfer:** Tamil/Telugu (Dravidian languages, linguistically more distant)

## Training Parameters

```python
learning_rate = 3e-5
batch_size = 16
num_epochs = 5
max_length = 128
optimizer = AdamW
weight_decay = 0.01
```

## Evaluation Metrics

All notebooks report:
- **Precision:** Token-level precision for NER tags
- **Recall:** Token-level recall for NER tags
- **F1 Score:** Harmonic mean (primary metric)
- **Accuracy:** Overall token classification accuracy

## NER Labels

All WikiANN datasets use the same 7 NER labels:
- `O` - Outside any named entity
- `B-PER` - Beginning of a person name
- `I-PER` - Inside a person name
- `B-ORG` - Beginning of an organization name
- `I-ORG` - Inside an organization name
- `B-LOC` - Beginning of a location name
- `I-LOC` - Inside a location name

## Output Structure

Each notebook produces:
1. **Trained model:** Saved to `./indicbert-{language}-ner-final/`
2. **Evaluation results:**
   - Validation results (source language)
   - Test results (Bangla - cross-lingual)
   - Transfer gap analysis
3. **Model artifacts:** Tokenizer, model weights, configuration

## Usage Instructions

### Running a Notebook

1. **Open notebook** in Jupyter/Colab/Kaggle
2. **Install dependencies** (Cell 1)
3. **Run all cells** sequentially
4. **Review results** in evaluation cells

### Expected Runtime
- **GPU (T4):** ~15-25 minutes per notebook
- **CPU:** Not recommended (very slow)

### Hardware Requirements
- **GPU Memory:** 16GB recommended
- **RAM:** 8GB minimum
- **Disk Space:** ~5GB per trained model

## Cross-Lingual Transfer Results

Results will be documented here after running experiments:

| Source Language | Source F1 | Bangla F1 | Transfer Gap | Linguistic Family |
|----------------|-----------|-----------|--------------|-------------------|
| Hindi | TBD | TBD | TBD | Indo-Aryan |
| Marathi | TBD | TBD | TBD | Indo-Aryan |
| Tamil | TBD | TBD | TBD | Dravidian |
| Telugu | TBD | TBD | TBD | Dravidian |
| Bangla (baseline) | TBD | TBD | 0.0 | Indo-Aryan |

## Research Questions

These experiments help answer:
1. **Does linguistic proximity affect transfer?** Compare Indo-Aryan vs. Dravidian transfer to Bangla
2. **Which source language transfers best?** Rank languages by cross-lingual F1 on Bangla
3. **What is the transfer gap?** Measure performance drop from source to target

## Next Steps

After completing these experiments:
1. Compare results across all source languages
2. Analyze linguistic features correlating with transfer success
3. Use insights to design sequential continual learning curriculum
4. Implement multi-source sequential training (Hindi → Marathi → Tamil → Telugu → Bangla)

## References

- **WikiANN Dataset:** Rahimi et al., 2019
- **IndicBERT:** Kakwani et al., 2020
- **BanglaBERT:** Bhattacharjee et al., 2022

---

**Created:** 2025-10-23
**Last Updated:** 2025-10-23
