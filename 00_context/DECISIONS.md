# Design Decisions Log

## Decision 001: Dataset Selection
- **Date**: 2026-10-19
- **Decision**: Use Corr2Cause default config only (train.csv, dev.csv, test.csv), not the perturbation variants
- **Justification**: The Project Ideas PDF specifies Corr2Cause as the starter dataset. The default config contains the core causal reasoning data. Perturbation variants (paraphrasing, refactorization) are augmented versions not required for Phase 1.

## Decision 002: Random Seed
- **Date**: 2026-10-19
- **Decision**: Use SEED = 42 throughout all stages
- **Justification**: Standard reproducibility practice. Documented here and in CONTEXT.md.

## Decision 003: Use Official Splits As-Is
- **Date**: 2026-10-19
- **Decision**: Keep train/dev/test splits from the dataset without re-splitting
- **Justification**: The dataset ships with official splits. The project specification requires verifying no overlap across splits, which we verify. Re-splitting would discard the authors' careful construction.

## Decision 004: Treat Dataset as Structured Causal Reasoning Text
- **Date**: 2026-10-19
- **Decision**: The Corr2Cause dataset contains formal causal reasoning text (premises about variables and hypotheses about causal relations). Our extraction pipeline will parse these structured texts to extract variables, relations, and build a causal graph, rather than treating them as free-form natural language.
- **Justification**: The input format is semi-structured ("Suppose there is a closed system of N variables..."). Leveraging this structure will produce better triples than generic NER/dependency parsing on arbitrary text.

## Decision 005: Entity Schema
- **Date**: 2026-10-19
- **Decision**: Define entities as the variables mentioned in each example (A, B, C, D, E, F) and relations as: correlates_with, independent_of, independent_given, direct_cause, ancestor, descendant, collider, confounder
- **Justification**: These are the relation types naturally present in the Corr2Cause premise/hypothesis structure and align with the template categories in the dataset.

## Decision 006: Phase 1 Scope Boundary
- **Date**: 2026-10-19
- **Decision**: No DistilBERT fine-tuning, no instruction model labelling, no PubMed abstracts extension
- **Justification**: These belong to Phase 2 (DL). The course instructions explicitly exclude them from Phase 1 scope.

## Decision 007: Cycle Resolution Strategy
- **Date**: 2026-10-19
- **Decision**: Resolve cycles by iteratively removing the lowest-confidence edge in each detected cycle
- **Justification**: Principled approach: lower confidence means less reliable extraction, so these edges are the best candidates for removal. This is documented and reproducible.

## Decision 008: Classifier Comparison
- **Date**: 2026-10-19
- **Decision**: Compare three classifiers — Logistic Regression, Linear SVM, Random Forest — all on the same features and splits
- **Justification**: Satisfies the "fair comparison" requirement from the PDF. Covers linear (LR, SVM) and non-linear (RF) approaches. Hyperparameter tuning only on dev split.

## Decision 009: Feature Design Philosophy
- **Date**: 2026-10-19
- **Decision**: Extract features tailored to the semi-structured Corr2Cause format (structural, causal cue word, relation pattern features) rather than generic NLP features
- **Justification**: The Corr2Cause text follows templates. Regex-based extraction of formal variable relations is more effective than dependency parsing on templated sentences. This is an honest acknowledgement of the dataset's nature.

## Decision 010: Class Imbalance Handling
- **Date**: 2026-10-19
- **Decision**: Use `class_weight='balanced'` in classifiers and report both macro-averaged and per-class metrics
- **Justification**: ~87% of labels are class 0. Without balancing, classifiers would trivially predict the majority class. Macro averaging ensures minority class performance is not hidden.

## Decision 011: Bayesian Network CPD Estimation
- **Date**: 2026-10-19
- **Decision**: Use Maximum Likelihood / Bayesian estimation for CPDs, with explicit documentation of where the model is uncertain
- **Justification**: With limited data from the extracted triples, we must be honest about uncertainty. The small query set is documented with expected vs actual answers.

## Decision 012: Master Notebook as Python Script
- **Date**: 2026-10-19
- **Decision**: Implement the master notebook as `masterNotebook.py` (a runnable Python script) rather than a Jupyter notebook
- **Justification**: A .py script is more reproducible (no hidden state), can be run from the command line, and is easier to version control. It chains all stages via dynamic imports.
