# Classification Report — Relation Extraction

## 1. Rule-Based Pattern Extractor

```
              precision    recall  f1-score   support

 Invalid (0)       0.84      0.96      0.90       982
   Valid (1)       0.12      0.03      0.05       180

    accuracy                           0.82      1162
   macro avg       0.48      0.50      0.47      1162
weighted avg       0.73      0.82      0.77      1162
```

## 2. LogisticRegression

**Best Params**: {'C': 0.1, 'class_weight': 'balanced'}

**Dev Macro F1**: 0.5391

```
              precision    recall  f1-score   support

 Invalid (0)       0.91      0.69      0.78       982
   Valid (1)       0.27      0.64      0.38       180

    accuracy                           0.68      1162
   macro avg       0.59      0.67      0.58      1162
weighted avg       0.81      0.68      0.72      1162
```

## 2. LinearSVM

**Best Params**: {'C': 1.0, 'class_weight': 'balanced'}

**Dev Macro F1**: 0.5489

```
              precision    recall  f1-score   support

 Invalid (0)       0.92      0.68      0.78       982
   Valid (1)       0.28      0.68      0.40       180

    accuracy                           0.68      1162
   macro avg       0.60      0.68      0.59      1162
weighted avg       0.82      0.68      0.72      1162
```

## 2. RandomForest

**Best Params**: {'class_weight': 'balanced', 'max_depth': 10, 'n_estimators': 200}

**Dev Macro F1**: 0.5579

```
              precision    recall  f1-score   support

 Invalid (0)       0.91      0.69      0.79       982
   Valid (1)       0.27      0.63      0.38       180

    accuracy                           0.68      1162
   macro avg       0.59      0.66      0.58      1162
weighted avg       0.81      0.68      0.72      1162
```

## Summary Comparison

| Model | Macro Precision | Macro Recall | Macro F1 |
|-------|----------------|-------------|----------|
| Rule-Based | 0.4829 | 0.4956 | 0.4725 |
| LogisticRegression | 0.5934 | 0.6654 | 0.5839 |
| LinearSVM | 0.6014 | 0.6818 | 0.5907 |
| RandomForest | 0.5913 | 0.6601 | 0.5832 |
