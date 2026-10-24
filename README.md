<p align="center">
  <img src="assets/banner.svg" alt="Neuro-Symbolic Causal Inference" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/pgmpy-Bayesian%20Network-6c5ce7" alt="pgmpy"/>
  <img src="https://img.shields.io/badge/networkx-DAG-0d9488" alt="networkx"/>
  <img src="https://img.shields.io/badge/dataset-Corr2Cause-e11d48" alt="dataset"/>
  <img src="https://img.shields.io/badge/phase-1%20(ML%20part)-2563eb" alt="phase"/>
</p>

> **Idea in one line:** read causal claims out of text, assemble them into a directed acyclic graph, turn the graph into a Bayesian network, and ask it interventional questions such as `P(B | do(E = high))`.

---

## Pipeline

<p align="center">
  <img src="assets/pipeline.svg" alt="Pipeline stages 02 to 08" width="100%"/>
</p>

```mermaid
flowchart LR
    A[Corr2Cause CSVs] --> B[Load and verify splits]
    B --> C[EDA and data quality]
    C --> D[Feature engineering]
    D --> E[Rule-based + LR / SVM / RF]
    E --> F[Extracted causal triples]
    F --> G[Build DAG and break cycles]
    G --> H[Bayesian network - pgmpy]
    H --> I[Queries: observe vs do]
    E --> J[Evaluation]
    I --> J
```

## Results

<p align="center">
  <img src="assets/results.svg" alt="Test macro-F1 by model" width="100%"/>
</p>

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Predict all-0 (reference) | 0.845 | 0.458 |
| Rule-based | 0.818 | 0.472 |
| Logistic Regression | 0.680 | 0.584 |
| Random Forest | 0.682 | 0.583 |
| **Linear SVM** | 0.681 | **0.591** |

The test set is 84.5% label 0, so accuracy is misleading and **macro-F1** is the metric to read.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python masterNotebook.py      # runs stages 02-08 end to end (SEED = 42)
python runTests.py            # automated checks
```

## Repository map

<details>
<summary><b>Click to expand</b></summary>

| Folder | What it does |
|---|---|
| `00_context/` | Project context and design decisions |
| `02_dataAcquisition/` | Loads Corr2Cause, builds `processed/` splits, verifies them |
| `03_edaAndDataQuality/` | Label balance, length stats, plots, quality summary |
| `04_featureEngineering/` | `extractFeatures.py` and the feature dictionary |
| `05_relationExtractionBaseline/` | Rule-based classifier, LR / SVM / RF, triple extraction |
| `06_dagConstruction/` | Triples to NetworkX graph, cycle detection and removal |
| `07_bayesianNetwork/` | pgmpy network, observational vs `do()` queries |
| `08_evaluation/` | Consolidated results and limitations |
| `09_report/` | LaTeX report |
| `10_presentation/` | Slide outline |
| `11_docs/` | Codebase explanation, how to run, viva preparation |

</details>

## Known limitations (read before citing numbers)

- **CPDs are hand-set priors**, not learned from data, because the corpus has no observational samples. Query results show the mechanics of `P(Y | X)` versus `P(Y | do(X))`; they are not empirical causal estimates.
- **Triples carry a constant confidence of 1.0**, so cycle removal falls back to an arbitrary tie-break.
- **Claims from independent examples are merged into one graph** over the letters A to F, which have different meanings in every example.
- **Hyperparameters are tuned with 3-fold CV on the training split**; the dev split is not used for tuning.
- Several engineered columns are constant, and the `parent` and `child` hypothesis types are merged.

---

<p align="center"><sub>Advanced Machine Learning · Phase 1 · Seed 42</sub></p>
