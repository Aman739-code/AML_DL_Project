# Data Quality Summary — Corr2Cause

## 1. Dataset Shape

| Split | Rows | Columns |
|-------|------|---------|
| Train | 205,734 | 9 |
| Dev | 1,076 | 9 |
| Test | 1,162 | 9 |

## 2. Missing Values

No missing values were found in any split across all columns.

✅ All values present.

## 3. Duplicate Texts

| Split | Exact Duplicates |
|-------|------------------|
| Train | 0 |
| Dev | 0 |
| Test | 0 |

## 4. Label Balance

| Split | Label 0 (Invalid) | Label 1 (Valid) | % Label 1 |
|-------|-------------------|-----------------|----------|
| Train | 167,686 | 38,048 | 18.5% |
| Dev | 937 | 139 | 12.9% |
| Test | 982 | 180 | 15.5% |

**Finding**: The dataset is significantly imbalanced, with ~87% negative (label 0) and ~13% positive (label 1) examples. This imbalance must be considered during model training and evaluation.

## 5. Vocabulary Statistics

| Split | Total Tokens | Unique Tokens |
|-------|-------------|---------------|
| Train | 83,318,721 | 67 |
| Dev | 309,067 | 69 |
| Test | 242,748 | 69 |

## 6. Issues Found and Handling

1. **Class imbalance**: ~87/13 split. Will use stratified sampling and report per-class metrics (precision, recall, F1).
2. **Structured text format**: Inputs are semi-structured (Premise/Hypothesis), not free-form text. Feature engineering should leverage this structure.
3. **Limited vocabulary**: The text uses a restricted vocabulary of causal reasoning terms, which is expected for a synthetic/curated dataset.
4. **Variable naming convention**: Variables are always single letters (A–F), making entity extraction straightforward.

## 7. Plots

- `plots/labelDistribution.png` — Label distribution per split
- `plots/templateDistribution.png` — Template type distribution per split
- `plots/numVariablesDistribution.png` — Number of variables per example
- `plots/textLengthDistribution.png` — Character and word length distributions
- `plots/labelByTemplate.png` — Label distribution within each template
- `plots/labelByNumVars.png` — Label distribution by number of variables
- `plots/entityMentions.png` — Entity mention counts per example
