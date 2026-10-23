# Neuro-Symbolic Causal Inference via Language Model Knowledge Extraction and Bayesian Networks

## Phase 1: Advanced Machine Learning (AML) Baseline

This repository contains the Phase 1 (ML-only) implementation of Project 1 for the CSAI course. The project extracts causal relations from the Corr2Cause dataset using traditional ML techniques, constructs a directed acyclic graph (DAG), and fits a Bayesian network for causal inference.

## Repository Structure

```
.
├── 00_context/              # Context log and decisions log (persistent memory)
├── 01_literatureReview/     # Literature review grounded in Pearl (1988) and Devlin et al. (2019)
├── 02_dataAcquisition/      # Data loading and split verification
├── 03_edaAndDataQuality/    # Exploratory data analysis and quality checks
├── 04_featureEngineering/   # Feature extraction (dependency patterns, NER, cue words)
├── 05_relationExtractionBaseline/  # Rule-based and ML relation classifiers
├── 06_dagConstruction/      # DAG construction with cycle resolution
├── 07_bayesianNetwork/      # Bayesian network fitting and causal queries
├── 08_evaluation/           # Consolidated evaluation metrics and analysis
├── 09_report/               # IEEE-format LaTeX report
├── 10_presentation/         # Slide outline with talking points
├── 11_docs/                 # How-to-run guide, tests, and viva preparation
├── screenshots/             # Screenshots organized by stage
├── corr2cause/              # Raw dataset (Corr2Cause from Hugging Face)
├── src/                     # Shared helper modules (config, seeds, paths)
├── masterNotebook.py        # Master script that runs all stages end-to-end
├── requirements.txt         # Pinned Python dependencies
└── README.md                # This file
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run the master notebook end-to-end
python masterNotebook.py

# 4. Run the automated test suite
python 11_docs/runTests.py
```

## Dataset

- **Corr2Cause** (causalnlp/corr2cause): A dataset for evaluating whether LLMs can distinguish correlation from causation
- Binary classification: label 0 (invalid causal claim) vs label 1 (valid causal claim)
- 6 template types covering different causal reasoning patterns
- Official train/dev/test splits

## Pipeline Overview

1. **Data Acquisition** → Load and verify Corr2Cause splits (no document leakage)
2. **EDA & Quality** → Label imbalance, text analysis, duplicate checks
3. **Feature Engineering** → Structural, cue word, and relation pattern features
4. **Relation Extraction** → Rule-based + 3 ML classifiers (fair comparison)
5. **DAG Construction** → Graph building, cycle detection and resolution
6. **Bayesian Network** → pgmpy BN with intervention and conditional queries
7. **Evaluation** → Consolidated metrics, plots, honest limitations

## Fixed Random Seed

All experiments use **SEED = 42** for full reproducibility.

## Documentation

- **How to Run**: See [`11_docs/HOW_TO_RUN_AND_TEST.md`](11_docs/HOW_TO_RUN_AND_TEST.md)
- **Tests**: Run `python 11_docs/runTests.py`
- **Viva Prep**: See [`11_docs/vivaPreparation.md`](11_docs/vivaPreparation.md)
- **Decisions Log**: See [`00_context/DECISIONS.md`](00_context/DECISIONS.md)

## Team

*Individual contributions to be filled by team members.*

## License

Academic use only — CSAI course project.
