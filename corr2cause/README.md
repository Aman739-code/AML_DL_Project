---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.csv
  - split: test
    path: test.csv
  - split: validation
    path: dev.csv
- config_name: perturbation_by_paraphrasing
  data_files:
  - split: train
    path: perturbation_by_paraphrasing_train.csv
  - split: test
    path: perturbation_by_paraphrasing_test.csv
  - split: validation
    path: perturbation_by_paraphrasing_dev.csv
- config_name: perturbation_by_refactorization
  data_files:
  - split: train
    path: perturbation_by_refactorization_train.csv
  - split: test
    path: perturbation_by_refactorization_test.csv
  - split: validation
    path: perturbation_by_refactorization_dev.csv
---

# Dataset card for corr2cause

TODO