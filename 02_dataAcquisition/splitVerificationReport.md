# Split Verification Report

## Split Sizes

| Split | Size |
|-------|------|
| Train | 205,734 |
| Dev   | 1,076 |
| Test  | 1,162 |

## Document ID Overlap

| Pair | Overlap |
|------|--------|
| Train ↔ Dev  | 0 |
| Train ↔ Test | 0 |
| Dev ↔ Test   | 0 |

## Exact Text Overlap

| Pair | Overlap |
|------|--------|
| Train ↔ Dev  | 0 |
| Train ↔ Test | 0 |
| Dev ↔ Test   | 0 |

## Premise Overlap (Near-Duplicate Check)

| Pair | Overlap |
|------|--------|
| Train ↔ Dev  | 2 |
| Train ↔ Test | 0 |
| Dev ↔ Test   | 3 |

⚠️ **Potential leakage detected.** Review overlapping entries.

Note: Some overlap in premises may be expected if the same causal structure is tested with different hypotheses. What matters is that exact input texts (premise + hypothesis) do not overlap across splits.
