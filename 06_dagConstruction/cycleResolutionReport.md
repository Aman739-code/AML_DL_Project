# Cycle Resolution Report

## Graph Statistics

| Metric | Before | After |
|--------|--------|-------|
| Nodes | 6 | 6 |
| Edges | 30 | 15 |
| Is DAG | No | Yes |

## Cycles Found: 15

## Edges Removed: 15

| # | Source | Target | Confidence | Relation | Reason |
|---|--------|--------|-----------|----------|--------|
| 1 | E | D | 1.000 | direct_cause | Cycle 1: lowest confidence edge |
| 2 | E | C | 1.000 | direct_cause | Cycle 2: lowest confidence edge |
| 3 | B | E | 1.000 | confounder | Cycle 3: lowest confidence edge |
| 4 | B | D | 1.000 | confounder | Cycle 4: lowest confidence edge |
| 5 | B | C | 1.000 | confounder | Cycle 5: lowest confidence edge |
| 6 | B | F | 1.000 | confounder | Cycle 6: lowest confidence edge |
| 7 | B | A | 1.000 | direct_cause | Cycle 7: lowest confidence edge |
| 8 | E | A | 1.000 | direct_cause | Cycle 8: lowest confidence edge |
| 9 | E | F | 1.000 | direct_cause | Cycle 9: lowest confidence edge |
| 10 | D | A | 1.000 | direct_cause | Cycle 10: lowest confidence edge |
| 11 | D | C | 1.000 | direct_cause | Cycle 11: lowest confidence edge |
| 12 | D | F | 1.000 | direct_cause | Cycle 12: lowest confidence edge |
| 13 | C | F | 1.000 | confounder | Cycle 13: lowest confidence edge |
| 14 | C | A | 1.000 | direct_cause | Cycle 14: lowest confidence edge |
| 15 | F | A | 1.000 | direct_cause | Cycle 15: lowest confidence edge |

## Resolution Strategy

Cycles were resolved by iteratively finding cycles and removing the edge with the **lowest confidence score** in each cycle. This is a principled approach because lower-confidence edges are more likely to be extraction errors. If all edges in a cycle have equal confidence, the last edge discovered is removed.

## Gap Analysis

This stage highlights the central gap of the project: text extraction produces edges independently, without enforcing global structural constraints like acyclicity. The fact that cycle resolution may remove edges demonstrates that pure text extraction cannot guarantee a valid causal DAG — this is precisely what the Bayesian network stage addresses.
