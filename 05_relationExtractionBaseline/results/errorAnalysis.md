# Error Analysis — LinearSVM

## False Positives (314 total)

These are examples where the model predicted the causal claim is valid, but it is actually invalid.

### FP Example 1

**Input**: Premise: Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical relations among these 4 variables are as follows: A correlates with B. A correlates with C. A correlates with D. B correlates with C. B correlates with D. C correlates with D. However, C and D are independent given A and B.
Hypothesis: There exists at least one collider (i.e., common effect) of A and B.

**True Label**: 0 (Invalid)
**Predicted**: 1 (Valid)

---

### FP Example 2

**Input**: Premise: Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical relations among these 4 variables are as follows: A correlates with B. A correlates with C. A correlates with D. B correlates with C. B correlates with D. C correlates with D. However, C and D are independent given A and B.
Hypothesis: There exists at least one collider (i.e., common effect) of A and C.

**True Label**: 0 (Invalid)
**Predicted**: 1 (Valid)

---

### FP Example 3

**Input**: Premise: Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical relations among these 4 variables are as follows: A correlates with B. A correlates with C. A correlates with D. B correlates with C. B correlates with D. C correlates with D. However, C and D are independent given A and B.
Hypothesis: There exists at least one collider (i.e., common effect) of A and D.

**True Label**: 0 (Invalid)
**Predicted**: 1 (Valid)

---

### FP Example 4

**Input**: Premise: Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical relations among these 4 variables are as follows: A correlates with B. A correlates with C. A correlates with D. B correlates with C. B correlates with D. C correlates with D. However, C and D are independent given A and B.
Hypothesis: There exists at least one collider (i.e., common effect) of B and C.

**True Label**: 0 (Invalid)
**Predicted**: 1 (Valid)

---

### FP Example 5

**Input**: Premise: Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical relations among these 4 variables are as follows: A correlates with B. A correlates with C. A correlates with D. B correlates with C. B correlates with D. C correlates with D. However, C and D are independent given A and B.
Hypothesis: There exists at least one collider (i.e., common effect) of B and D.

**True Label**: 0 (Invalid)
**Predicted**: 1 (Valid)

---

## False Negatives (57 total)

These are examples where the model predicted the causal claim is invalid, but it is actually valid.

### FN Example 1

**Input**: Premise: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.
Hypothesis: A directly causes C.

**True Label**: 1 (Valid)
**Predicted**: 0 (Invalid)

---

### FN Example 2

**Input**: Premise: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.
Hypothesis: B directly causes C.

**True Label**: 1 (Valid)
**Predicted**: 0 (Invalid)

---

### FN Example 3

**Input**: Premise: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.
Hypothesis: There exists at least one collider (i.e., common effect) of A and B.

**True Label**: 1 (Valid)
**Predicted**: 0 (Invalid)

---

### FN Example 4

**Input**: Premise: Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, A is independent of B. B and D are independent given A and C. B and D are independent given A, C and E. C and E are independent given A and B. C and E are i...

**True Label**: 1 (Valid)
**Predicted**: 0 (Invalid)

---

### FN Example 5

**Input**: Premise: Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, A is independent of B. A and D are independent given B and C. A and D are independent given B, C and E. B and E are independent given A and C. B and E are i...

**True Label**: 1 (Valid)
**Predicted**: 0 (Invalid)

---

## Summary

- Total test examples: 1162
- False Positives: 314 (27.0%)
- False Negatives: 57 (4.9%)
- True Positives: 123
- True Negatives: 668
