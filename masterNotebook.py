"""
Master Notebook — Neuro-Symbolic Causal Inference (Phase 1: AML)
==================================================================
This is the reproducible master script that runs all stages in order,
end-to-end. It corresponds to the "master notebook" deliverable described
in the Project Ideas PDF.

Usage:
    python masterNotebook.py

All random seeds are fixed to SEED = 42 for full reproducibility.
"""

import os
import sys
import time

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import SEED, setSeed

setSeed(SEED)


def runStage(stageName, stageModule, mainFunc="main"):
    """
    Import and run a stage's main() function, with timing.
    
    Parameters
    ----------
    stageName : str
        Human-readable stage name for display.
    stageModule : str
        Dotted module path (e.g. '02_dataAcquisition.loadData').
    mainFunc : str
        Name of the entry-point function in the module.
    """
    print("\n")
    print("▓" * 70)
    print(f"  ▸ {stageName}")
    print("▓" * 70)
    
    startTime = time.time()
    
    try:
        # Dynamic import
        import importlib
        mod = importlib.import_module(stageModule)
        fn = getattr(mod, mainFunc)
        result = fn()
    except Exception as e:
        print(f"\n  ⚠️  {stageName} encountered an error: {e}")
        import traceback
        traceback.print_exc()
        result = None
    
    elapsed = time.time() - startTime
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\n  ⏱  {stageName} completed in {minutes}m {seconds:.1f}s")
    
    return result


def main():
    """Run the full Phase 1 AML pipeline."""
    
    overallStart = time.time()
    
    print("=" * 70)
    print("  MASTER NOTEBOOK — Phase 1: AML Pipeline")
    print("  Neuro-Symbolic Causal Inference via")
    print("  Language Model Knowledge Extraction and Bayesian Networks")
    print("=" * 70)
    print(f"\n  Random Seed: {SEED}")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Python: {sys.version}")
    
    # ── Stage 02: Data Acquisition ────────────────────────────────────────
    runStage(
        "Stage 02: Data Acquisition",
        "02_dataAcquisition.loadData"
    )
    
    # ── Stage 03: EDA & Data Quality ──────────────────────────────────────
    runStage(
        "Stage 03: EDA & Data Quality",
        "03_edaAndDataQuality.edaAnalysis"
    )
    
    # ── Stage 04: Feature Engineering ─────────────────────────────────────
    runStage(
        "Stage 04: Feature Engineering",
        "04_featureEngineering.extractFeatures"
    )
    
    # ── Stage 05: Relation Extraction Baseline ────────────────────────────
    runStage(
        "Stage 05: Relation Extraction Baseline",
        "05_relationExtractionBaseline.trainClassifiers"
    )
    
    # ── Stage 06: DAG Construction ────────────────────────────────────────
    runStage(
        "Stage 06: DAG Construction",
        "06_dagConstruction.buildDag"
    )
    
    # ── Stage 07: Bayesian Network ────────────────────────────────────────
    runStage(
        "Stage 07: Bayesian Network",
        "07_bayesianNetwork.fitBayesianNetwork"
    )
    
    # ── Stage 08: Evaluation ──────────────────────────────────────────────
    runStage(
        "Stage 08: Evaluation",
        "08_evaluation.evaluate"
    )
    
    # ── Summary ───────────────────────────────────────────────────────────
    overallElapsed = time.time() - overallStart
    minutes = int(overallElapsed // 60)
    seconds = overallElapsed % 60
    
    print("\n")
    print("═" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {minutes}m {seconds:.1f}s")
    print("═" * 70)
    print()
    print("  Next steps:")
    print("    1. Review outputs in each stage folder")
    print("    2. Run tests: python 11_docs/runTests.py")
    print("    3. Compile report: cd 09_report && pdflatex report.tex")
    print("    4. Review presentation: 10_presentation/slideOutline.md")
    print()


if __name__ == "__main__":
    main()
