# Causal Query Results

This document records the causal queries run on the Bayesian network, including both **conditional queries** (passive observation) and **intervention queries** (do-operator, simulating experiments).

## Query 1: P(B | E = low)

**Type**: Conditional

**Expected Reasoning**: Observing E=low updates belief about B via passive conditioning.

**Model Output**:
```
+---------+----------+
| B       |   phi(B) |
+=========+==========+
| B(low)  |   0.8100 |
+---------+----------+
| B(high) |   0.1900 |
+---------+----------+
```

**Interpretation**: Under passive observation of E=low, B is more likely to be low.

---

## Query 2: P(B | E = high)

**Type**: Conditional

**Expected Reasoning**: Observing E=high shifts belief about B towards high.

**Model Output**:
```
+---------+----------+
| B       |   phi(B) |
+=========+==========+
| B(low)  |   0.1760 |
+---------+----------+
| B(high) |   0.8240 |
+---------+----------+
```

**Interpretation**: Comparing Query 1 and Query 2 demonstrates strong associative dependence between E and B.

---

## Query 3: P(B | do(E = high))

**Type**: Intervention (do-operator)

**Expected Reasoning**: The do-operator actively sets E=high by cutting all incoming edges to E (graph surgery), removing confounding.

**Model Output**:
```
+---------+----------+
| B       |   phi(B) |
+=========+==========+
| B(low)  |   0.2807 |
+---------+----------+
| B(high) |   0.7193 |
+---------+----------+
```

**Interpretation**: The interventional distribution P(B | do(E=high)) isolates the pure causal effect of E on B. Confounding from common ancestors accounts for the difference from the conditional distribution P(B | E=high).

---

## Query 4: P(D | do(A = low))

**Type**: Intervention (do-operator)

**Expected Reasoning**: Intervening on root variable A=low propagates downstream through directed paths to affect D.

**Model Output**:
```
+---------+----------+
| D       |   phi(D) |
+=========+==========+
| D(low)  |   0.6851 |
+---------+----------+
| D(high) |   0.3149 |
+---------+----------+
```

**Interpretation**: Demonstrates that an upstream intervention on A causally shifts the downstream belief of D.

---

## Query 5: Compare P(D) vs P(D | A = high)

**Type**: Conditional independence test

**Expected Reasoning**: If D and A are d-separated, unconditional and conditional distributions are identical; if d-connected, they differ.

**Model Output**:
```
Unconditional:
+---------+----------+
| D       |   phi(D) |
+=========+==========+
| D(low)  |   0.4703 |
+---------+----------+
| D(high) |   0.5297 |
+---------+----------+

Conditional on A=high:
+---------+----------+
| D       |   phi(D) |
+=========+==========+
| D(low)  |   0.2554 |
+---------+----------+
| D(high) |   0.7446 |
+---------+----------+
```

**Interpretation**: The shift between unconditional and conditional distributions confirms d-connection (causal flow) between A and D.

---

## Key Insight: do-Operator vs Conditioning

The intervention queries (do-operator) differ from conditional queries because:
- **Conditioning**: P(Y | X=x) — passive observation of X=x
- **Intervention**: P(Y | do(X=x)) — actively setting X=x by graph surgery

Graph surgery removes all incoming edges to X, eliminating confounding. This is why a Bayesian network can answer questions that a simple correlation-based classifier cannot.
