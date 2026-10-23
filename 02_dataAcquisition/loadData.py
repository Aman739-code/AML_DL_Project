"""
Stage 02: Data Acquisition
===========================
Load the Corr2Cause dataset from raw CSV files, assign stable document and
sentence identifiers, verify that no documents leak across splits, and save
the processed data for downstream stages.
"""

import os
import sys
import hashlib
import pandas as pd

# ── Add project root to path so we can import from src ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    SEED, setSeed, TRAIN_CSV, DEV_CSV, TEST_CSV,
    COL_INPUT, COL_LABEL, COL_NUM_VARS, COL_TEMPLATE, ensureDir
)

setSeed(SEED)


def generateDocId(text: str, split: str, index: int) -> str:
    """
    Generate a stable, unique document identifier from the text content.
    Uses a truncated SHA-256 hash of the text for stability, prefixed with
    the split name and row index for readability.
    """
    textHash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{split}_{index:06d}_{textHash}"


def loadAndProcess(csvPath: str, splitName: str) -> pd.DataFrame:
    """
    Load a raw CSV, assign document and sentence IDs, and validate schema.
    
    Each row in Corr2Cause contains one premise–hypothesis pair, which we 
    treat as one 'document'. The premise may contain multiple sentences 
    describing correlations/independencies, and we assign sentence IDs 
    to each within the premise.
    """
    df = pd.read_csv(csvPath)
    
    # Validate expected columns
    expectedCols = {COL_INPUT, COL_LABEL, COL_NUM_VARS, COL_TEMPLATE}
    actualCols = set(df.columns)
    assert expectedCols.issubset(actualCols), \
        f"Missing columns: {expectedCols - actualCols}"
    
    # Assign stable document IDs
    df["docId"] = [
        generateDocId(row[COL_INPUT], splitName, idx) 
        for idx, row in df.iterrows()
    ]
    
    # Assign split name
    df["split"] = splitName
    
    # Parse premise and hypothesis from the input text
    df["premise"] = df[COL_INPUT].apply(extractPremise)
    df["hypothesis"] = df[COL_INPUT].apply(extractHypothesis)
    
    # Assign sentence IDs within each document (premise sentences)
    df["premiseSentences"] = df["premise"].apply(splitIntoSentences)
    df["numPremiseSentences"] = df["premiseSentences"].apply(len)
    
    # Ensure label is integer
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    df[COL_NUM_VARS] = df[COL_NUM_VARS].astype(int)
    
    return df


def extractPremise(inputText: str) -> str:
    """Extract the premise part from the input text."""
    if "Hypothesis:" in inputText:
        parts = inputText.split("Hypothesis:")
        premise = parts[0].replace("Premise:", "").strip()
        return premise
    return inputText.replace("Premise:", "").strip()


def extractHypothesis(inputText: str) -> str:
    """Extract the hypothesis part from the input text."""
    if "Hypothesis:" in inputText:
        parts = inputText.split("Hypothesis:")
        return parts[1].strip()
    return ""


def splitIntoSentences(premiseText: str) -> list:
    """
    Split a premise into individual sentences describing relations.
    We split on period-space patterns but keep the content meaningful.
    """
    # Split on '. ' but keep sentences clean
    sentences = []
    current = ""
    for char in premiseText:
        current += char
        if char == '.' and len(current.strip()) > 5:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def verifySplitIntegrity(trainDf: pd.DataFrame, devDf: pd.DataFrame, 
                         testDf: pd.DataFrame) -> dict:
    """
    Verify that no document IDs or input texts overlap across splits.
    Returns a report dictionary.
    """
    report = {}
    
    # Check document ID overlap
    trainIds = set(trainDf["docId"])
    devIds = set(devDf["docId"])
    testIds = set(testDf["docId"])
    
    report["trainDevDocOverlap"] = len(trainIds & devIds)
    report["trainTestDocOverlap"] = len(trainIds & testIds)
    report["devTestDocOverlap"] = len(devIds & testIds)
    
    # Check input text overlap (exact duplicates across splits)
    trainTexts = set(trainDf[COL_INPUT])
    devTexts = set(devDf[COL_INPUT])
    testTexts = set(testDf[COL_INPUT])
    
    report["trainDevTextOverlap"] = len(trainTexts & devTexts)
    report["trainTestTextOverlap"] = len(trainTexts & testTexts)
    report["devTestTextOverlap"] = len(devTexts & testTexts)
    
    # Check near-duplicates (same premise appearing in different splits)
    trainPremises = set(trainDf["premise"])
    devPremises = set(devDf["premise"])
    testPremises = set(testDf["premise"])
    
    report["trainDevPremiseOverlap"] = len(trainPremises & devPremises)
    report["trainTestPremiseOverlap"] = len(trainPremises & testPremises)
    report["devTestPremiseOverlap"] = len(devPremises & testPremises)
    
    # Split sizes
    report["trainSize"] = len(trainDf)
    report["devSize"] = len(devDf)
    report["testSize"] = len(testDf)
    
    return report


def writeVerificationReport(report: dict, outputPath: str):
    """Write the split verification report to a markdown file."""
    with open(outputPath, "w") as f:
        f.write("# Split Verification Report\n\n")
        f.write("## Split Sizes\n\n")
        f.write(f"| Split | Size |\n")
        f.write(f"|-------|------|\n")
        f.write(f"| Train | {report['trainSize']:,} |\n")
        f.write(f"| Dev   | {report['devSize']:,} |\n")
        f.write(f"| Test  | {report['testSize']:,} |\n\n")
        
        f.write("## Document ID Overlap\n\n")
        f.write(f"| Pair | Overlap |\n")
        f.write(f"|------|--------|\n")
        f.write(f"| Train ↔ Dev  | {report['trainDevDocOverlap']} |\n")
        f.write(f"| Train ↔ Test | {report['trainTestDocOverlap']} |\n")
        f.write(f"| Dev ↔ Test   | {report['devTestDocOverlap']} |\n\n")
        
        f.write("## Exact Text Overlap\n\n")
        f.write(f"| Pair | Overlap |\n")
        f.write(f"|------|--------|\n")
        f.write(f"| Train ↔ Dev  | {report['trainDevTextOverlap']} |\n")
        f.write(f"| Train ↔ Test | {report['trainTestTextOverlap']} |\n")
        f.write(f"| Dev ↔ Test   | {report['devTestTextOverlap']} |\n\n")
        
        f.write("## Premise Overlap (Near-Duplicate Check)\n\n")
        f.write(f"| Pair | Overlap |\n")
        f.write(f"|------|--------|\n")
        f.write(f"| Train ↔ Dev  | {report['trainDevPremiseOverlap']} |\n")
        f.write(f"| Train ↔ Test | {report['trainTestPremiseOverlap']} |\n")
        f.write(f"| Dev ↔ Test   | {report['devTestPremiseOverlap']} |\n\n")
        
        allZero = all(v == 0 for k, v in report.items() if "Overlap" in k)
        if allZero:
            f.write("✅ **No leakage detected.** All overlaps are zero.\n")
        else:
            f.write("⚠️ **Potential leakage detected.** Review overlapping entries.\n")
            f.write("\nNote: Some overlap in premises may be expected if the same ")
            f.write("causal structure is tested with different hypotheses. ")
            f.write("What matters is that exact input texts (premise + hypothesis) ")
            f.write("do not overlap across splits.\n")


def main():
    """Main entry point for Stage 02."""
    print("=" * 60)
    print("Stage 02: Data Acquisition")
    print("=" * 60)
    
    stageDir = os.path.dirname(os.path.abspath(__file__))
    processedDir = ensureDir(os.path.join(stageDir, "processed"))
    
    # Load and process each split
    print("\nLoading training data...")
    trainDf = loadAndProcess(TRAIN_CSV, "train")
    print(f"  → {len(trainDf):,} rows loaded")
    
    print("Loading validation data...")
    devDf = loadAndProcess(DEV_CSV, "dev")
    print(f"  → {len(devDf):,} rows loaded")
    
    print("Loading test data...")
    testDf = loadAndProcess(TEST_CSV, "test")
    print(f"  → {len(testDf):,} rows loaded")
    
    # Verify split integrity
    print("\nVerifying split integrity...")
    report = verifySplitIntegrity(trainDf, devDf, testDf)
    reportPath = os.path.join(stageDir, "splitVerificationReport.md")
    writeVerificationReport(report, reportPath)
    print(f"  → Report saved to {reportPath}")
    
    # Print summary
    for key, val in report.items():
        if "Overlap" in key:
            status = "✅" if val == 0 else "⚠️"
            print(f"  {status} {key}: {val}")
    
    # Save processed data (drop the premiseSentences list column for CSV)
    print("\nSaving processed data...")
    for name, df in [("train", trainDf), ("dev", devDf), ("test", testDf)]:
        saveDf = df.drop(columns=["premiseSentences"])
        outPath = os.path.join(processedDir, f"{name}.csv")
        saveDf.to_csv(outPath, index=False)
        print(f"  → {outPath} ({len(saveDf):,} rows)")
    
    print("\n✅ Stage 02 complete.")
    return trainDf, devDf, testDf


if __name__ == "__main__":
    main()
