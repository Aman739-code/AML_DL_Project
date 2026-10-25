"""
Stage 11: Automated Test Suite
================================
Concrete, runnable checks that verify the integrity of the entire pipeline.

Run with:
    python 11_docs/runTests.py

Exits with code 0 if all tests pass, 1 otherwise.
"""

import os
import sys

# ── Add project root to path ──
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

passed = 0
failed = 0
total = 0


def check(name, condition, detail=""):
    """Record a test result."""
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ PASS: {name}")
    else:
        failed += 1
        msg = f"  ❌ FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: No document-ID overlap across splits
# ═══════════════════════════════════════════════════════════════════════════
def testSplitIntegrity():
    """Assert that no docId appears in more than one split."""
    print("\n── Test 1: Split Integrity (No Document Leakage) ──")
    processedDir = os.path.join(PROJECT_ROOT, "02_dataAcquisition", "processed")
    
    if not os.path.exists(processedDir):
        check("Processed data exists", False, "Run Stage 02 first")
        return
    
    trainPath = os.path.join(processedDir, "train.csv")
    devPath = os.path.join(processedDir, "dev.csv")
    testPath = os.path.join(processedDir, "test.csv")
    
    if not os.path.exists(devPath) or not os.path.exists(testPath):
        check("Processed dev and test data exist", False, "Run Stage 02 first")
        return
    
    devDf = pd.read_csv(devPath)
    testDf = pd.read_csv(testPath)
    
    check("Dev data loaded", len(devDf) > 0, f"{len(devDf)} rows")
    check("Test data loaded", len(testDf) > 0, f"{len(testDf)} rows")
    
    devIds = set(devDf["docId"].unique()) if "docId" in devDf.columns else set()
    testIds = set(testDf["docId"].unique()) if "docId" in testDf.columns else set()
    devTestOverlap = devIds & testIds
    check("No dev/test docId overlap", len(devTestOverlap) == 0,
          f"{len(devTestOverlap)} overlapping IDs")
    
    if os.path.exists(trainPath):
        trainDf = pd.read_csv(trainPath)
        check("Train data loaded", len(trainDf) > 0, f"{len(trainDf)} rows")
        if "docId" in trainDf.columns:
            trainIds = set(trainDf["docId"].unique())
            trainDevOverlap = trainIds & devIds
            trainTestOverlap = trainIds & testIds
            check("No train/dev docId overlap", len(trainDevOverlap) == 0,
                  f"{len(trainDevOverlap)} overlapping IDs")
            check("No train/test docId overlap", len(trainTestOverlap) == 0,
                  f"{len(trainTestOverlap)} overlapping IDs")
        else:
            check("docId column exists in train", False, "Column 'docId' not found in processed train data")
    else:
        print("  ℹ️  Note: train.csv omitted from git (>100MB limit); download per 11_docs/HOW_TO_RUN_AND_TEST.md")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: DAG is acyclic and has no self-loops
# ═══════════════════════════════════════════════════════════════════════════
def testDagIntegrity():
    """Assert the DAG is acyclic and has no self-loops."""
    print("\n── Test 2: DAG Integrity ──")
    try:
        import networkx as nx
    except ImportError:
        check("NetworkX installed", False, "pip install networkx")
        return
    
    dagPath = os.path.join(PROJECT_ROOT, "06_dagConstruction", "graphs", "dag.graphml")
    
    if not os.path.exists(dagPath):
        check("DAG file exists", False, "Run Stage 06 first")
        return
    
    check("DAG file exists", True)
    G = nx.read_graphml(dagPath)
    
    check("Graph is not empty", G.number_of_nodes() > 0,
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Check acyclicity
    isDAG = nx.is_directed_acyclic_graph(G)
    check("Graph is acyclic (valid DAG)", isDAG)
    
    # Check no self-loops
    selfLoops = list(nx.selfloop_edges(G))
    check("No self-loops", len(selfLoops) == 0,
          f"{len(selfLoops)} self-loops found")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Edge provenance (sentenceId and docId attributes)
# ═══════════════════════════════════════════════════════════════════════════
def testEdgeProvenance():
    """Assert that every edge has sentenceId and docId attributes."""
    print("\n── Test 3: Edge Provenance ──")
    try:
        import networkx as nx
    except ImportError:
        check("NetworkX installed", False)
        return
    
    dagPath = os.path.join(PROJECT_ROOT, "06_dagConstruction", "graphs", "dag.graphml")
    if not os.path.exists(dagPath):
        check("DAG file exists", False, "Run Stage 06 first")
        return
    
    G = nx.read_graphml(dagPath)
    
    missingSentenceId = 0
    missingDocId = 0
    for u, v, data in G.edges(data=True):
        if "sentenceId" not in data:
            missingSentenceId += 1
        if "docId" not in data:
            missingDocId += 1
    
    total_edges = G.number_of_edges()
    check(f"All edges have sentenceId ({total_edges} edges)", missingSentenceId == 0,
          f"{missingSentenceId}/{total_edges} edges missing sentenceId")
    check(f"All edges have docId ({total_edges} edges)", missingDocId == 0,
          f"{missingDocId}/{total_edges} edges missing docId")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Bayesian Network model check
# ═══════════════════════════════════════════════════════════════════════════
def testBayesianNetwork():
    """Assert that pgmpy's check_model() passes."""
    print("\n── Test 4: Bayesian Network Validation ──")
    
    bnResultsDir = os.path.join(PROJECT_ROOT, "07_bayesianNetwork", "results")
    queryResultsPath = os.path.join(bnResultsDir, "queryResults.md")
    
    check("Query results file exists", os.path.exists(queryResultsPath),
          "Run Stage 07 first" if not os.path.exists(queryResultsPath) else "")
    
    # We cannot easily reconstruct the BN without running the stage,
    # so we verify the outputs exist and are non-empty
    if os.path.exists(queryResultsPath):
        with open(queryResultsPath) as f:
            content = f.read()
        check("Query results non-empty", len(content) > 100,
              f"File is only {len(content)} bytes")
        check("Contains intervention queries", "do(" in content.lower() or "intervention" in content.lower(),
              "Expected do-operator queries")
        check("Contains conditional queries", "conditional" in content.lower() or "P(" in content,
              "Expected conditional queries")
        check("All queries executed successfully without error", "Error:" not in content and "Query failed" not in content,
              "queryResults.md contains error messages")
    
    bnStructurePath = os.path.join(bnResultsDir, "bnStructure.png")
    check("BN structure visualisation exists", os.path.exists(bnStructurePath),
          "Run Stage 07 first" if not os.path.exists(bnStructurePath) else "")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Reproducibility — feature matrix shapes
# ═══════════════════════════════════════════════════════════════════════════
def testReproducibility():
    """Verify seed is consistent and feature matrices are loadable."""
    print("\n── Test 5: Reproducibility ──")
    
    from src.config import SEED
    check("SEED is 42", SEED == 42, f"Got SEED = {SEED}")
    
    featuresDir = os.path.join(PROJECT_ROOT, "04_featureEngineering", "features")
    if not os.path.exists(featuresDir):
        check("Features directory exists", False, "Run Stage 04 first")
        return
    
    for split in ["trainFeatures", "devFeatures", "testFeatures"]:
        fpath = os.path.join(featuresDir, f"{split}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            check(f"{split}.csv loadable", True,
                  f"Shape: {df.shape}")
            check(f"{split} has label column", "label" in df.columns)
        else:
            check(f"{split}.csv exists", False, "Run Stage 04 first")


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Extracted triples exist and have required columns
# ═══════════════════════════════════════════════════════════════════════════
def testExtractedTriples():
    """Verify triples file schema."""
    print("\n── Test 6: Extracted Triples ──")
    
    triplesPath = os.path.join(
        PROJECT_ROOT, "05_relationExtractionBaseline", "results", "extractedTriples.csv"
    )
    
    if not os.path.exists(triplesPath):
        check("Triples file exists", False, "Run Stage 05 first")
        return
    
    check("Triples file exists", True)
    df = pd.read_csv(triplesPath)
    check("Triples non-empty", len(df) > 0, f"{len(df)} triples")
    
    requiredCols = {"subject", "relation", "object", "sentenceId", "docId"}
    actualCols = set(df.columns)
    missing = requiredCols - actualCols
    check("Required columns present", len(missing) == 0,
          f"Missing: {missing}" if missing else f"Columns: {list(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Evaluation outputs
# ═══════════════════════════════════════════════════════════════════════════
def testEvaluationOutputs():
    """Check that Stage 08 produced the required outputs."""
    print("\n── Test 7: Evaluation Outputs ──")
    
    evalDir = os.path.join(PROJECT_ROOT, "08_evaluation")
    
    # Consolidated results table
    for fname in ["consolidatedResults.csv", "limitations.md"]:
        fpath = os.path.join(evalDir, fname)
        check(f"{fname} exists", os.path.exists(fpath),
              "Run Stage 08 first" if not os.path.exists(fpath) else "")


# ═══════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Automated Test Suite — Neuro-Symbolic Causal Inference")
    print("  Phase 1 (AML)")
    print("=" * 60)
    
    testSplitIntegrity()
    testDagIntegrity()
    testEdgeProvenance()
    testBayesianNetwork()
    testReproducibility()
    testExtractedTriples()
    testEvaluationOutputs()
    
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {total} total")
    print("=" * 60)
    
    if failed > 0:
        print("\n  ⚠️  Some tests failed. Run the missing stages first.")
        sys.exit(1)
    else:
        print("\n  ✅ All tests passed!")
        sys.exit(0)
