# Consolidated Evaluation Results

## Relation Extraction — Test Set Performance

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|-------|----------|---------|---------|----------|
| LogisticRegression | 0.6799 | 0.5934 | 0.6654 | 0.5839 |
| LinearSVM | 0.6807 | 0.6014 | 0.6818 | 0.5907 |
| RandomForest | 0.6824 | 0.5913 | 0.6601 | 0.5832 |
| RuleBased | 0.8184 | 0.4829 | 0.4956 | 0.4725 |

## Per-Class Metrics (Best ML Model)

**Best model**: LinearSVM

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Invalid (0) | 0.9214 | 0.6802 | 0.7827 |
| Valid (1)   | 0.2815 | 0.6833 | 0.3987 |

## Key Observations

1. The class imbalance (~87% label 0) makes accuracy misleading
2. Macro F1 is the fairer metric for comparing approaches
3. The ML classifiers with engineered features outperform the rule-based baseline
4. Class 1 (valid causal claims) remains harder to detect, as expected
