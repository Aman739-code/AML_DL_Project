"""
Stage 04: Feature Engineering
==============================
Extract features from Corr2Cause texts for the ML relation classifier.

Features are extracted from both the premise (statistical relations) and
hypothesis (causal claim), capturing:
- Structural features (num variables, num relations, num independencies)
- Causal cue word presence
- Hypothesis type encoding
- Text statistics
- Relation pattern features

All vectorisers/encoders are fit on training data only, then applied
to validation and test.
"""

import os
import sys
import re
import pickle

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    SEED, setSeed, COL_INPUT, COL_LABEL, COL_NUM_VARS, COL_TEMPLATE, ensureDir
)

setSeed(SEED)

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = ensureDir(os.path.join(STAGE_DIR, "features"))
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "02_dataAcquisition", "processed")

# ── Causal cue words ──────────────────────────────────────────────────────
CAUSAL_CUE_WORDS = [
    "cause", "causes", "caused", "causing",
    "effect", "affects", "affect",
    "leads", "lead",
    "results", "result",
    "due", "because",
    "therefore", "thus", "hence",
    "implies", "imply",
    "determines", "determine",
    "influences", "influence",
    "produces", "produce",
    "directly",
    "indirectly",
    "ancestor",
    "descendant",
    "collider",
    "confounder",
    "mediator",
]

# ── Relation patterns ────────────────────────────────────────────────────
CORRELATION_PATTERN = re.compile(r'\b([A-F])\s+correlates?\s+with\s+([A-F])\b', re.IGNORECASE)
INDEPENDENCE_PATTERN = re.compile(r'\b([A-F])\s+(?:is\s+)?independent\s+(?:of|from)\s+([A-F])\b', re.IGNORECASE)
CONDITIONAL_INDEP_PATTERN = re.compile(
    r'\b([A-F])\s+and\s+([A-F])\s+are\s+independent\s+given\s+((?:[A-F](?:\s+and\s+)?)+)',
    re.IGNORECASE
)
DIRECT_CAUSE_PATTERN = re.compile(r'\b([A-F])\s+directly\s+causes?\s+([A-F])\b', re.IGNORECASE)
COLLIDER_PATTERN = re.compile(r'collider.*?(?:common\s+effect).*?([A-F])\s+and\s+([A-F])', re.IGNORECASE)
CONFOUNDER_PATTERN = re.compile(r'confounder.*?(?:common\s+cause).*?([A-F])\s+and\s+([A-F])', re.IGNORECASE)

# ── Hypothesis type keywords ─────────────────────────────────────────────
HYPOTHESIS_TYPES = {
    "directCause": ["directly causes", "direct cause"],
    "ancestor": ["ancestor", "not a direct one", "indirect cause"],
    "descendant": ["descendant"],
    "collider": ["collider", "common effect"],
    "confounder": ["confounder", "common cause"],
    "parent": ["directly causes"],
    "child": ["directly caused by"],
}


def extractVariables(text: str) -> set:
    """Extract all variable names (A–F) mentioned in the text."""
    return set(re.findall(r'\b([A-F])\b', text))


def countCorrelations(premiseText: str) -> int:
    """Count the number of correlation statements in the premise."""
    return len(CORRELATION_PATTERN.findall(premiseText))


def countIndependencies(premiseText: str) -> int:
    """Count the number of independence statements in the premise."""
    simple = len(INDEPENDENCE_PATTERN.findall(premiseText))
    conditional = len(CONDITIONAL_INDEP_PATTERN.findall(premiseText))
    return simple + conditional


def countConditionalIndependencies(premiseText: str) -> int:
    """Count specifically conditional independence statements."""
    return len(CONDITIONAL_INDEP_PATTERN.findall(premiseText))


def countCausalCueWords(text: str) -> int:
    """Count occurrences of causal cue words in the text."""
    textLower = text.lower()
    return sum(1 for cue in CAUSAL_CUE_WORDS if cue in textLower)


def classifyHypothesisType(hypothesisText: str) -> str:
    """Classify the hypothesis into a type based on keywords."""
    hypoLower = hypothesisText.lower()
    for hType, keywords in HYPOTHESIS_TYPES.items():
        if any(kw in hypoLower for kw in keywords):
            return hType
    return "other"


def computeRelationDensity(numCorrelations: int, numVars: int) -> float:
    """
    Compute the density of correlations relative to possible pairs.
    For n variables, max pairs = n*(n-1)/2.
    """
    if numVars < 2:
        return 0.0
    maxPairs = numVars * (numVars - 1) / 2
    return numCorrelations / maxPairs if maxPairs > 0 else 0.0


def computeIndependenceRatio(numIndependencies: int, numCorrelations: int) -> float:
    """Ratio of independence statements to correlation statements."""
    total = numIndependencies + numCorrelations
    return numIndependencies / total if total > 0 else 0.0


def extractFeaturesFromRow(row: pd.Series) -> dict:
    """
    Extract all features from a single data row.
    
    Returns a dict of feature_name -> feature_value.
    """
    inputText = row[COL_INPUT]
    numVars = int(row[COL_NUM_VARS])
    
    # Split into premise and hypothesis
    if "Hypothesis:" in inputText:
        parts = inputText.split("Hypothesis:")
        premiseText = parts[0].replace("Premise:", "").strip()
        hypothesisText = parts[1].strip()
    else:
        premiseText = inputText.replace("Premise:", "").strip()
        hypothesisText = ""
    
    # ── Structural features ──
    variables = extractVariables(premiseText)
    numCorrelations = countCorrelations(premiseText)
    numIndependencies = countIndependencies(premiseText)
    numCondIndependencies = countConditionalIndependencies(premiseText)
    
    # ── Hypothesis features ──
    hypoVariables = extractVariables(hypothesisText)
    hypoType = classifyHypothesisType(hypothesisText)
    
    # ── Text statistics ──
    premiseWordCount = len(premiseText.split())
    hypothesisWordCount = len(hypothesisText.split())
    premiseCharCount = len(premiseText)
    
    # ── Cue words ──
    premiseCueWords = countCausalCueWords(premiseText)
    hypothesisCueWords = countCausalCueWords(hypothesisText)
    
    # ── Derived features ──
    relationDensity = computeRelationDensity(numCorrelations, numVars)
    indepRatio = computeIndependenceRatio(numIndependencies, numCorrelations)
    
    # Number of premise sentences (approximate by splitting on '. ')
    numPremiseSentences = premiseText.count('. ') + 1
    
    # Whether hypothesis mentions variables from the premise
    hypoVarsInPremise = len(hypoVariables & variables)
    
    # Average relations per variable
    totalRelations = numCorrelations + numIndependencies
    avgRelationsPerVar = totalRelations / numVars if numVars > 0 else 0
    
    features = {
        # Structural
        "numVariables": numVars,
        "numCorrelations": numCorrelations,
        "numIndependencies": numIndependencies,
        "numCondIndependencies": numCondIndependencies,
        "numPremiseSentences": numPremiseSentences,
        "totalRelations": totalRelations,
        
        # Density / ratio
        "relationDensity": round(relationDensity, 4),
        "independenceRatio": round(indepRatio, 4),
        "avgRelationsPerVar": round(avgRelationsPerVar, 4),
        
        # Text stats
        "premiseWordCount": premiseWordCount,
        "hypothesisWordCount": hypothesisWordCount,
        "premiseCharCount": premiseCharCount,
        "inputWordCount": premiseWordCount + hypothesisWordCount,
        
        # Cue words
        "premiseCueWords": premiseCueWords,
        "hypothesisCueWords": hypothesisCueWords,
        "totalCueWords": premiseCueWords + hypothesisCueWords,
        
        # Hypothesis type (one-hot encoded below)
        "hypoType": hypoType,
        
        # Variable overlap
        "numHypoVarsInPremise": hypoVarsInPremise,
        "numHypoVars": len(hypoVariables),
        
        # Has conditional independence (binary)
        "hasConditionalIndep": int(numCondIndependencies > 0),
        
        # Premise mentions "however" (signals independence constraints)
        "hasHowever": int("however" in premiseText.lower()),
    }
    
    return features


def oneHotEncodeHypoType(featuresDf: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the hypoType column."""
    # Get all possible types
    allTypes = ["directCause", "ancestor", "descendant", "collider", "confounder", "parent", "child", "other"]
    
    for hType in allTypes:
        featuresDf[f"hypoType_{hType}"] = (featuresDf["hypoType"] == hType).astype(int)
    
    featuresDf = featuresDf.drop(columns=["hypoType"])
    return featuresDf


def extractAllFeatures(df: pd.DataFrame, splitName: str) -> pd.DataFrame:
    """
    Extract features for all rows in a dataframe.
    Returns a DataFrame with the features, label, and metadata columns.
    """
    print(f"  Extracting features for {splitName} ({len(df):,} rows)...")
    
    featuresList = []
    for idx, row in df.iterrows():
        feats = extractFeaturesFromRow(row)
        feats["label"] = int(row[COL_LABEL])
        feats["docId"] = row.get("docId", f"{splitName}_{idx}")
        feats["template"] = row[COL_TEMPLATE]
        featuresList.append(feats)
    
    featuresDf = pd.DataFrame(featuresList)
    featuresDf = oneHotEncodeHypoType(featuresDf)
    
    print(f"    → {featuresDf.shape[1]} features extracted")
    return featuresDf


def writeFeatureDictionary(featuresDf: pd.DataFrame):
    """Write a markdown file describing every feature and its rationale."""
    path = os.path.join(STAGE_DIR, "featureDictionary.md")
    
    featureDescriptions = {
        "numVariables": ("Number of variables in the closed system (from dataset)", 
                        "Controls complexity; more variables → more possible causal structures"),
        "numCorrelations": ("Count of 'X correlates with Y' statements in premise",
                           "Captures the density of statistical associations"),
        "numIndependencies": ("Count of all independence statements (simple + conditional)",
                             "Independence constraints are key to narrowing causal structures"),
        "numCondIndependencies": ("Count of specifically conditional independence statements",
                                  "Conditional independencies are more informative for causal discovery"),
        "numPremiseSentences": ("Number of sentences in the premise",
                               "Proxy for information richness of the premise"),
        "totalRelations": ("numCorrelations + numIndependencies",
                          "Total amount of statistical evidence provided"),
        "relationDensity": ("numCorrelations / (numVars choose 2)",
                           "How dense the correlation graph is; sparser → more independence"),
        "independenceRatio": ("numIndependencies / totalRelations",
                             "Fraction of relations that are independencies vs correlations"),
        "avgRelationsPerVar": ("totalRelations / numVariables",
                              "Average information per variable"),
        "premiseWordCount": ("Word count of the premise text",
                            "Text length feature; longer premises have more complex structures"),
        "hypothesisWordCount": ("Word count of the hypothesis text",
                               "Hypothesis complexity indicator"),
        "premiseCharCount": ("Character count of the premise text",
                            "Complementary length feature"),
        "inputWordCount": ("Total word count (premise + hypothesis)",
                          "Overall text length"),
        "premiseCueWords": ("Count of causal cue words in premise",
                           "Causal language density in premise"),
        "hypothesisCueWords": ("Count of causal cue words in hypothesis",
                              "Causal language density in hypothesis"),
        "totalCueWords": ("Total causal cue words across premise and hypothesis",
                         "Overall causal language presence"),
        "numHypoVarsInPremise": ("Number of hypothesis variables also in premise",
                                "Whether the hypothesis refers to variables with known relations"),
        "numHypoVars": ("Number of variables mentioned in hypothesis",
                       "Hypothesis specificity"),
        "hasConditionalIndep": ("Binary: premise contains conditional independence statements",
                               "Conditional independencies strongly constrain possible DAGs"),
        "hasHowever": ("Binary: premise contains 'However'",
                      "Signals transition from correlations to independence constraints"),
    }
    
    # Add one-hot features
    for hType in ["directCause", "ancestor", "descendant", "collider", "confounder", "parent", "child", "other"]:
        featureDescriptions[f"hypoType_{hType}"] = (
            f"Binary: hypothesis type is '{hType}'",
            "Controls for different causal claim types which have different base rates"
        )
    
    with open(path, "w") as f:
        f.write("# Feature Dictionary\n\n")
        f.write("This document describes every feature used in the relation extraction classifier, ")
        f.write("along with the rationale for its inclusion.\n\n")
        f.write("| Feature | Description | Rationale |\n")
        f.write("|---------|-------------|----------|\n")
        
        for feat, (desc, rationale) in featureDescriptions.items():
            f.write(f"| `{feat}` | {desc} | {rationale} |\n")
        
        f.write(f"\n**Total features**: {featuresDf.shape[1] - 3} ")
        f.write("(excluding label, docId, template metadata)\n")
    
    print(f"  → Feature dictionary saved to {path}")


def main():
    """Main entry point for Stage 04."""
    print("=" * 60)
    print("Stage 04: Feature Engineering")
    print("=" * 60)
    
    # Load processed data
    trainDf = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    devDf = pd.read_csv(os.path.join(PROCESSED_DIR, "dev.csv"))
    testDf = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    # Extract features
    trainFeatures = extractAllFeatures(trainDf, "train")
    devFeatures = extractAllFeatures(devDf, "dev")
    testFeatures = extractAllFeatures(testDf, "test")
    
    # Save feature matrices
    trainFeatures.to_csv(os.path.join(FEATURES_DIR, "trainFeatures.csv"), index=False)
    devFeatures.to_csv(os.path.join(FEATURES_DIR, "devFeatures.csv"), index=False)
    testFeatures.to_csv(os.path.join(FEATURES_DIR, "testFeatures.csv"), index=False)
    
    print(f"\n  Feature matrices saved to {FEATURES_DIR}")
    print(f"  Train: {trainFeatures.shape}")
    print(f"  Dev:   {devFeatures.shape}")
    print(f"  Test:  {testFeatures.shape}")
    
    # Write feature dictionary
    writeFeatureDictionary(trainFeatures)
    
    # Print feature statistics
    print("\n  Feature summary (Dev set):")
    numericCols = trainFeatures.select_dtypes(include=[np.number]).columns
    numericCols = [c for c in numericCols if c != "label"]
    for col in numericCols[:10]:
        print(f"    {col}: mean={trainFeatures[col].mean():.3f}, "
              f"std={trainFeatures[col].std():.3f}")
    
    print("\n✅ Stage 04 complete.")
    return trainFeatures, devFeatures, testFeatures


if __name__ == "__main__":
    main()
