# HOW TO RUN AND TEST — Neuro-Symbolic Causal Inference (Phase 1 AML)

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pip | latest |
| Git | latest (for version control) |
| LaTeX | optional — only for compiling `09_report/report.tex` |

## 2. Environment Setup

```bash
# 2a. Clone / navigate to the project root
cd aml_dl_project

# 2b. Create a virtual environment
python3 -m venv venv

# 2c. Activate it
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 2d. Install all dependencies
pip install -r requirements.txt

# 2e. Download the spaCy English model
python -m spacy download en_core_web_sm
```

## 3. Dataset

The Corr2Cause dataset must be present under `corr2cause/` with at least:
- `train.csv` (~383 MB)
- `dev.csv` (~1.4 MB)
- `test.csv` (~1.2 MB)

If missing, download from Hugging Face:
```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('causalnlp/corr2cause')
for split in ['train', 'validation', 'test']:
    ds[split].to_csv(f'corr2cause/{split}.csv', index=False)
"
```

## 4. Running Each Stage Individually

All stage scripts are designed to be run from the **project root directory**.

```bash
# Stage 02: Data Acquisition (load & verify splits)
python 02_dataAcquisition/loadData.py

# Stage 03: EDA & Data Quality
python 03_edaAndDataQuality/edaAnalysis.py

# Stage 04: Feature Engineering
python 04_featureEngineering/extractFeatures.py

# Stage 05: Relation Extraction Baseline (trains classifiers, extracts triples)
python 05_relationExtractionBaseline/trainClassifiers.py

# Stage 06: DAG Construction
python 06_dagConstruction/buildDag.py

# Stage 07: Bayesian Network
python 07_bayesianNetwork/fitBayesianNetwork.py

# Stage 08: Consolidated Evaluation
python 08_evaluation/evaluate.py
```

**Note**: Stages 01, 09, 10 are documentation/report stages with no runnable scripts.

## 5. Running the Master Notebook (End-to-End)

```bash
python masterNotebook.py
```

This runs stages 02 → 03 → 04 → 05 → 06 → 07 → 08 in order, with timing information printed for each.

## 6. Where Outputs Are Saved

| Stage | Output Location | Contents |
|---|---|---|
| 02 | `02_dataAcquisition/processed/` | Processed train/dev/test CSVs with doc IDs |
| 02 | `02_dataAcquisition/splitVerificationReport.md` | Split integrity report |
| 03 | `03_edaAndDataQuality/plots/` | EDA plots (PNG) |
| 03 | `03_edaAndDataQuality/dataQualitySummary.md` | Written quality summary |
| 04 | `04_featureEngineering/features/` | Feature matrices (CSV) + dictionary |
| 05 | `05_relationExtractionBaseline/results/` | Triples, classification reports |
| 05 | `05_relationExtractionBaseline/models/` | Saved scikit-learn models (pickle) |
| 05 | `05_relationExtractionBaseline/plots/` | Confusion matrices, feature importance |
| 06 | `06_dagConstruction/graphs/` | GraphML DAG, before/after PNGs |
| 06 | `06_dagConstruction/cycleResolutionReport.md` | Cycle resolution details |
| 07 | `07_bayesianNetwork/results/` | BN structure PNG, query results |
| 08 | `08_evaluation/plots/` | Model comparison plots |
| 08 | `08_evaluation/consolidatedResults.csv` | Merged metrics table |
| 08 | `08_evaluation/limitations.md` | Honest limitations analysis |
| screenshots | `screenshots/<stage_name>/` | Stage-specific screenshots |

## 7. Running the Automated Tests

```bash
python 11_docs/runTests.py
```

### What the tests verify:

1. **No document-ID overlap across splits** — Every `docId` appears in exactly one of train/dev/test.
2. **Final graph is acyclic** — The saved DAG contains no directed cycles.
3. **No self-loops** — Every edge connects two distinct nodes.
4. **Edge provenance** — Every edge in the DAG carries `sentenceId` and `docId` attributes.
5. **Bayesian network model check** — pgmpy's internal `check_model()` passes.
6. **Reproducibility** — Running with SEED = 42 reproduces the same feature matrix shapes.
7. **Causal query results** — The documented query results file exists and contains expected fields.

All tests print PASS / FAIL and the script exits with code 0 on full success, 1 on any failure.

## 8. Compiling the Report (Optional)

If you have a LaTeX distribution (e.g. TeX Live, MiKTeX):
```bash
cd 09_report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
cd ..
```

## 9. Fixed Random Seed

**SEED = 42** is used across all stages. It is set in `src/config.py` and imported by every stage script. To verify reproducibility, delete all output folders and re-run `python masterNotebook.py` — all numeric results should match exactly.

## 10. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run scripts from the project root, not from inside stage folders |
| `spacy.load('en_core_web_sm')` fails | Run `python -m spacy download en_core_web_sm` |
| `FileNotFoundError: ...train.csv` | Ensure the `corr2cause/` directory contains the dataset files |
| Memory issues on full train set | The train set is ~383 MB; ensure ≥4 GB RAM is available |
| pgmpy import errors | Ensure `pgmpy==1.1.2` is installed via `pip install -r requirements.txt` |
