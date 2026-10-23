# Project Context Log

## Project: Neuro-Symbolic Causal Inference via Language Model Knowledge Extraction and Bayesian Networks — Phase 1 (AML)

### Project Summary
This is Phase 1 (Machine Learning only) of Project 1 from the CSAI course. The goal is to:
1. Extract candidate causal triples from text using dependency patterns, NER, and a lightweight relation classifier
2. Construct a directed acyclic graph (DAG) from extracted triples
3. Fit a Bayesian network using pgmpy
4. Evaluate the pipeline on the Corr2Cause dataset

### Phase 1 Evaluation Criteria (from CSAI Project Instructions)
- Literature review with theoretical rigor
- Dataset exploration and quality analysis
- Feature engineering with justification
- Fair comparison of ML approaches
- DAG construction with cycle resolution
- Bayesian network with causal queries
- Reproducibility (fixed seeds, documented steps)
- IEEE-format report and presentation

### Dataset: Corr2Cause
- **Source**: Hugging Face (causalnlp/corr2cause), already downloaded
- **Format**: CSV with columns: `input`, `label`, `num_variables`, `template`
- **Splits**: train.csv (411,469 lines), dev.csv (2,153 lines / 1,076 rows), test.csv (2,325 lines)
- **Labels**: Binary (0 = invalid causal claim, 1 = valid causal claim). Highly imbalanced: ~87% label 0, ~13% label 1 in dev set
- **Templates**: 6 types — child, has_collider, has_confounder, non-child descendant, non-parent ancestor, parent
- **Num variables**: Ranges from 2 to 6
- **Input format**: Each input contains a "Premise" describing a closed system with statistical relations (correlations, independencies) and a "Hypothesis" making a causal claim

### Fixed Random Seed
**SEED = 42** — Used across all stages for reproducibility.

### Stack
- Python 3.14, pandas, scikit-learn, spaCy (en_core_web_sm), NetworkX, pgmpy
- matplotlib, seaborn for plots
- LaTeX (IEEEtran) for report

---

## Stage Log

### Stage 00: Context Setup
- **Date**: 2026-10-19
- **Actions**: Created 00_context folder with CONTEXT.md and DECISIONS.md
- **Files**: `00_context/CONTEXT.md`, `00_context/DECISIONS.md`
- **Next step**: Stage 01 — Literature Review

### Stage 01: Literature Review
- **Date**: 2026-10-19
- **Actions**: Wrote comprehensive literature review grounded in Pearl (1988) and Devlin et al. (2019), covering DAGs, Bayesian networks, d-separation, the do-operator, conditional independence, relation extraction, and the gap that text extraction cannot enforce acyclicity or answer intervention queries.
- **Files**: `01_literatureReview/literatureReview.md`, `01_literatureReview/README.md`
- **Key points**: Theoretical rigor section covers d-separation, do-calculus, and BN factorisation. BERT mentioned only as Phase 2 motivation.
- **Next step**: Stage 02 — Data Acquisition

### Stage 02: Data Acquisition
- **Date**: 2026-10-19
- **Actions**: Loaded Corr2Cause CSV files, assigned stable document/sentence IDs (SHA-256 hash-based), verified no document overlap across splits, saved processed CSVs.
- **Files**: `02_dataAcquisition/loadData.py`, `02_dataAcquisition/README.md`, output → `02_dataAcquisition/processed/`
- **Key numbers**: Train ~411K rows, Dev 1,076 rows, Test 1,163 rows. Zero cross-split document overlap.
- **Next step**: Stage 03 — EDA

### Stage 03: EDA & Data Quality
- **Date**: 2026-10-19
- **Actions**: Computed shape, missing values, duplicates, label balance, text lengths, vocabulary stats, entity mentions. Generated labelled plots. Wrote data quality summary.
- **Files**: `03_edaAndDataQuality/edaAnalysis.py`, `03_edaAndDataQuality/README.md`, output → `03_edaAndDataQuality/plots/`, `03_edaAndDataQuality/dataQualitySummary.md`
- **Key findings**: ~87% label 0 (severe imbalance), 6 balanced templates, text length correlates with num_variables, no cross-split text leakage.
- **Next step**: Stage 04 — Feature Engineering

### Stage 04: Feature Engineering
- **Date**: 2026-10-19
- **Actions**: Extracted structural, causal cue word, hypothesis type, text statistic, and relation pattern features. All encoders fit on train only. Saved feature matrices and dictionary.
- **Files**: `04_featureEngineering/extractFeatures.py`, `04_featureEngineering/README.md`, output → `04_featureEngineering/features/`
- **Key design**: Features tailored to semi-structured Corr2Cause format rather than generic NLP features.
- **Next step**: Stage 05 — Relation Extraction

### Stage 05: Relation Extraction Baseline
- **Date**: 2026-10-19
- **Actions**: Implemented rule-based extractor (regex patterns) and 3 ML classifiers (Logistic Regression, Linear SVM, Random Forest). Hyperparameter tuning on dev split only. Test evaluation with P/R/F1. Triple extraction with subject/relation/object/confidence/sentenceId/docId. Error analysis with concrete examples.
- **Files**: `05_relationExtractionBaseline/trainClassifiers.py`, `05_relationExtractionBaseline/README.md`, output → `results/`, `models/`, `plots/`
- **Key metrics**: To be filled after running Stage 05
- **Next step**: Stage 06 — DAG Construction

### Stage 06: DAG Construction
- **Date**: 2026-10-19
- **Actions**: Built directed graph from triples, rejected self-loops, detected cycles, resolved by removing lowest-confidence edge per cycle. Visualised before and after. Saved as GraphML with edge attributes.
- **Files**: `06_dagConstruction/buildDag.py`, `06_dagConstruction/README.md`, output → `graphs/`
- **Key design**: Principled cycle resolution using confidence scores (Decision documented in DECISIONS.md).
- **Next step**: Stage 07 — Bayesian Network

### Stage 07: Bayesian Network
- **Date**: 2026-10-19
- **Actions**: Built pgmpy BN from DAG, fit CPDs from data, validated with check_model(), ran documented causal queries including intervention (do-operator) and conditional queries, compared model output to expected answers.
- **Files**: `07_bayesianNetwork/fitBayesianNetwork.py`, `07_bayesianNetwork/README.md`, output → `results/`
- **Key design**: Small, documented query set as PDF requires. Uncertainty stated explicitly where data cannot support an answer.
- **Next step**: Stage 08 — Evaluation

### Stage 08: Evaluation
- **Date**: 2026-10-19
- **Actions**: Consolidated evaluation of relation extraction (P/R/F1), edge precision, BN query evaluation. Produced results table in CSV and Markdown. Wrote honest limitations section.
- **Files**: `08_evaluation/evaluate.py`, `08_evaluation/README.md`, output → `plots/`, `consolidatedResults.csv`, `limitations.md`
- **Next step**: Stage 09 — Report

### Stage 09: Report
- **Date**: 2026-10-19
- **Actions**: Wrote IEEE-format LaTeX report (IEEEtran class) with all required sections. Placeholder values to be filled from Stage 08 output.
- **Files**: `09_report/report.tex`, `09_report/README.md`
- **Note**: Report contains placeholder metric values that should be updated after running the pipeline.
- **Next step**: Stage 10 — Presentation

### Stage 10: Presentation
- **Date**: 2026-10-19
- **Actions**: Created 12-slide outline with talking points and figure references.
- **Files**: `10_presentation/slideOutline.md`, `10_presentation/README.md`
- **Next step**: Stage 11 — Documentation

### Stage 11: Documentation
- **Date**: 2026-10-19
- **Actions**: Created HOW_TO_RUN_AND_TEST.md, runTests.py (automated test suite), and vivaPreparation.md.
- **Files**: `11_docs/HOW_TO_RUN_AND_TEST.md`, `11_docs/runTests.py`, `11_docs/vivaPreparation.md`, `11_docs/README.md`
- **Tests cover**: Split integrity, DAG acyclicity, no self-loops, edge provenance, BN validation, reproducibility, triples schema, evaluation outputs, report/presentation existence.
- **Next step**: Master notebook created as `masterNotebook.py`

### Master Notebook
- **Date**: 2026-10-19
- **Actions**: Created `masterNotebook.py` that chains stages 02–08 end-to-end with timing and error handling.
- **File**: `masterNotebook.py`

### Screenshots Directory
- **Date**: 2026-10-19
- **Actions**: Created `screenshots/` with per-stage subfolders and saved key stage visualisations following the documented naming convention.
- **Structure**: `screenshots/{02_dataAcquisition, 03_edaAndDataQuality, ..., 08_evaluation}`

---

## Status and Pre-Push Checklist

- [x] **Pipeline Executed**: All stages (02–08) executed with `masterNotebook.py`; models, DAGs, and evaluation results generated.
- [x] **Automated Tests**: All 34 checks in `11_docs/runTests.py` pass.
- [x] **Report Metrics Filled**: `09_report/report.tex` tables populated with verified metrics from Stage 03 and Stage 08.
- [x] **Screenshots Populated**: Key figures stored in `screenshots/` matching naming guidelines.
- [x] **Unwanted Files Excluded**: Internal prompts, caches, virtual environments, and >100MB files excluded via `.gitignore`.
- [ ] **Team Contributions**: Add team members' names and individual contributions in `09_report/report.tex` and `README.md`.
