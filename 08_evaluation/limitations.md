# Limitations

## 1. Text Leakage Safeguards

- We verified that no exact input texts overlap between train/dev/test splits
- Some premises may share similar structures (same causal graph, different hypotheses), which is by design in the dataset but could inflate similarity metrics
- All feature fitting (scaler, vectoriser) is performed on training data only

## 2. Dataset Limitations

- **Synthetic nature**: Corr2Cause uses template-generated text about abstract variables (A, B, C...), not real-world causal claims from scientific literature
- **Class imbalance**: ~87% of examples are label 0 (invalid claims), making positive class detection challenging
- **Limited vocabulary**: The restricted, formulaic language limits the generalisability of learned features to free-form text
- **Abstract variables**: Real causal reasoning involves domain-specific entities (genes, drugs, economic indicators), not generic letters

## 3. AML Baseline Gaps (Motivating Phase 2)

- **No deep semantic understanding**: Our features capture surface patterns but cannot reason about the logical structure of conditional independence arguments
- **Independent edge extraction**: Each edge is extracted independently, without considering global consistency of the causal graph
- **CPDs from priors**: Without observational data for the abstract variables, CPDs are constructed from informative priors rather than estimated from data
- **Limited intervention queries**: The small set of queries demonstrates the capability but does not validate against known ground-truth interventional effects

## 4. Methodological Limitations

- The rule-based extractor is deliberately conservative (high precision, low recall)
- Feature engineering relies on domain knowledge of the dataset structure
- Hyperparameter tuning uses 3-fold CV on training data, which may be suboptimal
- The DAG construction treats all extracted edges equally (same confidence)

## 5. What Phase 2 Should Address

- Fine-tuned DistilBERT for end-to-end causal claim classification
- Learned representations capturing semantic entailment structure
- Potential for graph neural networks to enforce structural constraints
- Richer confidence calibration for extracted edges
