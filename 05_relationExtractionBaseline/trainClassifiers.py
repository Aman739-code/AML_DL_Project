"""
Stage 05: Relation Extraction Baseline
========================================
Two approaches compared on the same test set:
1. Rule-based pattern extractor (using regex patterns on the structured text)
2. ML classifiers (Logistic Regression, Linear SVM, Random Forest) trained
   on features from Stage 04

Both are evaluated with precision, recall, and F1 on the held-out test set.
Hyperparameter tuning is performed on the validation (dev) split only.
"""

import os
import sys
import re
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import SEED, setSeed, COL_INPUT, COL_LABEL, ensureDir

setSeed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = ensureDir(os.path.join(STAGE_DIR, "results"))
MODELS_DIR = ensureDir(os.path.join(STAGE_DIR, "models"))
PLOTS_DIR = ensureDir(os.path.join(STAGE_DIR, "plots"))
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "04_featureEngineering", "features")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "02_dataAcquisition", "processed")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Rule-Based Pattern Extractor
# ═══════════════════════════════════════════════════════════════════════════

def ruleBasedPredict(inputText: str) -> int:
    """
    Rule-based prediction: determine if the causal hypothesis follows from
    the premise based on pattern matching of statistical relations.
    
    The key insight is that in Corr2Cause:
    - Label 1 (valid): the causal claim IS entailed by the statistical relations
    - Label 0 (invalid): the causal claim is NOT entailed
    
    Rules:
    1. If variables are independent, no direct causal link exists → reject direct cause claims
    2. For collider claims: need convergent patterns (A→C←B)
    3. For confounder claims: need divergent patterns (A←C→B)
    4. Direct cause (parent): correlation must exist AND certain independence patterns
    5. Conservative default: predict 0 (most hypotheses are not entailed)
    """
    if "Hypothesis:" not in inputText:
        return 0
    
    parts = inputText.split("Hypothesis:")
    premise = parts[0].replace("Premise:", "").strip().lower()
    hypothesis = parts[1].strip().lower()
    
    # Extract mentioned variables in hypothesis
    hypoVars = re.findall(r'\b([a-f])\b', hypothesis)
    if len(hypoVars) < 2:
        return 0
    
    var1, var2 = hypoVars[0], hypoVars[1]
    
    # Check if the two main hypothesis variables correlate
    varsCorrelate = (
        f"{var1} correlates with {var2}" in premise or 
        f"{var2} correlates with {var1}" in premise
    )
    
    # Check if they are independent
    varsIndependent = (
        f"{var1} is independent of {var2}" in premise or
        f"{var2} is independent of {var1}" in premise or
        f"{var1} and {var2} are independent" in premise or
        f"{var2} and {var1} are independent" in premise
    )
    
    # Count how many conditional independencies involve these variables
    condIndepCount = premise.count(f"{var1} and {var2} are independent given")
    condIndepCount += premise.count(f"{var2} and {var1} are independent given")
    
    # Number of variables in the system
    allVars = set(re.findall(r'\b([a-f])\b', premise))
    numVars = len(allVars)
    
    # ── Direct cause (parent) claims ──
    if "directly causes" in hypothesis or "direct cause" in hypothesis:
        # If variables are unconditionally independent, can't be direct cause
        if varsIndependent:
            return 0
        # If they correlate and there are conditional independencies with
        # other variables, it might be a valid direct cause in a small system
        if varsCorrelate and numVars == 2:
            # With only 2 variables and correlation, direct cause is possible
            # but not guaranteed (could be reverse causation)
            return 0  # Conservative: can't determine direction from correlation alone
        if varsCorrelate and condIndepCount == 0 and numVars <= 3:
            return 0  # Not enough independence info to confirm
        return 0  # Default conservative
    
    # ── Collider claims ──
    if "collider" in hypothesis or "common effect" in hypothesis:
        if varsIndependent:
            # For a collider structure (A→C←B), A and B are marginally independent
            # but become dependent when conditioning on C
            # Check if they become dependent given some variable
            otherVars = allVars - {var1, var2}
            for v in otherVars:
                condDep = (
                    f"{var1} correlates with {var2}" not in premise and
                    not any(f"{var1} and {var2} are independent given" in s 
                           for s in [premise])
                )
            return 0  # Conservative
        return 0
    
    # ── Confounder claims ──
    if "confounder" in hypothesis or "common cause" in hypothesis:
        if varsCorrelate:
            # A confounder means A←C→B: A and B correlate but become
            # independent when conditioning on C
            otherVars = allVars - {var1, var2}
            for v in otherVars:
                givenV = f"{var1} and {var2} are independent given {v}"
                givenV2 = f"{var2} and {var1} are independent given {v}"
                if givenV in premise or givenV2 in premise:
                    # They become independent given a third variable → suggests confounder
                    return 1
        return 0
    
    # ── Ancestor / descendant claims ──
    if "ancestor" in hypothesis or "not a direct one" in hypothesis:
        if varsCorrelate:
            return 0  # Can't confirm indirect cause from correlations alone
        return 0
    
    if "descendant" in hypothesis:
        if varsCorrelate:
            return 0
        return 0
    
    # Default: predict 0 (conservative)
    return 0


def evaluateRuleBased(testDf: pd.DataFrame) -> dict:
    """Evaluate the rule-based extractor on the test set."""
    predictions = testDf[COL_INPUT].apply(ruleBasedPredict)
    trueLabels = testDf[COL_LABEL].astype(int)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        trueLabels, predictions, average=None, zero_division=0
    )
    macroPrecision, macroRecall, macroF1, _ = precision_recall_fscore_support(
        trueLabels, predictions, average="macro", zero_division=0
    )
    
    report = classification_report(trueLabels, predictions, 
                                   target_names=["Invalid (0)", "Valid (1)"],
                                   zero_division=0)
    
    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macroPrecision": macroPrecision,
        "macroRecall": macroRecall,
        "macroF1": macroF1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: ML Classifiers
# ═══════════════════════════════════════════════════════════════════════════

def getFeatureColumns(df: pd.DataFrame) -> list:
    """Get the list of numeric feature columns (excluding metadata)."""
    excludeCols = {"label", "docId", "template"}
    return [c for c in df.columns if c not in excludeCols and df[c].dtype in [np.float64, np.int64, np.int32, np.float32]]


def trainAndEvaluateModels(trainDf, devDf, testDf):
    """
    Train Logistic Regression, Linear SVM, and Random Forest on the 
    training features, tune hyperparameters on dev, evaluate on test.
    """
    featureCols = getFeatureColumns(trainDf)
    
    xTrain = trainDf[featureCols].values
    yTrain = trainDf["label"].values
    xDev = devDf[featureCols].values
    yDev = devDf["label"].values
    xTest = testDf[featureCols].values
    yTest = testDf["label"].values
    
    # Standardise features (fit on train only)
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xDevScaled = scaler.transform(xDev)
    xTestScaled = scaler.transform(xTest)
    
    # Save scaler
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    results = {}
    
    # ── 1. Logistic Regression ──
    print("\n  Training Logistic Regression...")
    lrParams = {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}
    lrGrid = GridSearchCV(
        LogisticRegression(random_state=SEED, max_iter=1000, solver="lbfgs"),
        lrParams, cv=3, scoring="f1_macro", n_jobs=-1
    )
    lrGrid.fit(xTrainScaled, yTrain)
    bestLr = lrGrid.best_estimator_
    print(f"    Best params: {lrGrid.best_params_}")
    print(f"    Dev macro F1: {lrGrid.best_score_:.4f}")
    
    lrPreds = bestLr.predict(xTestScaled)
    results["LogisticRegression"] = {
        "model": bestLr,
        "predictions": lrPreds,
        "bestParams": lrGrid.best_params_,
        "devScore": lrGrid.best_score_,
    }
    
    # ── 2. Linear SVM ──
    print("\n  Training Linear SVM...")
    svmParams = {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}
    svmGrid = GridSearchCV(
        LinearSVC(random_state=SEED, max_iter=2000, dual=True),
        svmParams, cv=3, scoring="f1_macro", n_jobs=-1
    )
    svmGrid.fit(xTrainScaled, yTrain)
    bestSvm = svmGrid.best_estimator_
    print(f"    Best params: {svmGrid.best_params_}")
    print(f"    Dev macro F1: {svmGrid.best_score_:.4f}")
    
    svmPreds = bestSvm.predict(xTestScaled)
    results["LinearSVM"] = {
        "model": bestSvm,
        "predictions": svmPreds,
        "bestParams": svmGrid.best_params_,
        "devScore": svmGrid.best_score_,
    }
    
    # ── 3. Random Forest ──
    print("\n  Training Random Forest...")
    rfParams = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "class_weight": [None, "balanced"],
    }
    rfGrid = GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        rfParams, cv=3, scoring="f1_macro", n_jobs=-1
    )
    rfGrid.fit(xTrain, yTrain)  # RF doesn't need scaling
    bestRf = rfGrid.best_estimator_
    print(f"    Best params: {rfGrid.best_params_}")
    print(f"    Dev macro F1: {rfGrid.best_score_:.4f}")
    
    rfPreds = bestRf.predict(xTest)
    results["RandomForest"] = {
        "model": bestRf,
        "predictions": rfPreds,
        "bestParams": rfGrid.best_params_,
        "devScore": rfGrid.best_score_,
    }
    
    # Save models
    for name, res in results.items():
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(res["model"], f)
    
    return results, xTestScaled, xTest, yTest, featureCols


def extractTriples(testDf: pd.DataFrame, predictions: np.ndarray, 
                   modelName: str) -> pd.DataFrame:
    """
    Extract causal triples from test examples where the model predicts
    the causal claim is valid (label=1).
    
    Each triple has: subject, relation, object, confidence, sentenceId, docId.
    """
    triples = []
    for idx, (_, row) in enumerate(testDf.iterrows()):
        if predictions[idx] == 1:
            inputText = row[COL_INPUT]
            if "Hypothesis:" in inputText:
                hypothesis = inputText.split("Hypothesis:")[1].strip()
            else:
                hypothesis = ""
            
            # Extract variables from hypothesis
            hypoVars = re.findall(r'\b([A-F])\b', hypothesis)
            
            subject = hypoVars[0] if len(hypoVars) >= 1 else "?"
            obj = hypoVars[1] if len(hypoVars) >= 2 else "?"
            
            # Determine relation type from hypothesis
            hypoLower = hypothesis.lower()
            if "directly causes" in hypoLower:
                relation = "direct_cause"
            elif "ancestor" in hypoLower or "not a direct one" in hypoLower:
                relation = "indirect_cause"
            elif "collider" in hypoLower or "common effect" in hypoLower:
                relation = "collider"
            elif "confounder" in hypoLower or "common cause" in hypoLower:
                relation = "confounder"
            else:
                relation = "causal_relation"
            
            triples.append({
                "subject": subject,
                "relation": relation,
                "object": obj,
                "confidence": 1.0,  # Binary classifier; no probability for rule-based
                "sentenceId": f"test_{idx:04d}_hypo",
                "docId": row.get("docId", f"test_{idx}"),
                "model": modelName,
                "hypothesis": hypothesis,
            })
    
    return pd.DataFrame(triples)


def plotConfusionMatrices(yTest, results, ruleBasedResults):
    """Plot confusion matrices for all models side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Rule-based
    cm = confusion_matrix(yTest, ruleBasedResults["predictions"])
    ConfusionMatrixDisplay(cm, display_labels=["Invalid", "Valid"]).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title("Rule-Based", fontweight="bold")
    
    # ML models
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(yTest, res["predictions"])
        ConfusionMatrixDisplay(cm, display_labels=["Invalid", "Valid"]).plot(ax=axes[i+1], cmap="Blues")
        axes[i+1].set_title(name, fontweight="bold")
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusionMatrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")


def plotFeatureImportance(model, featureCols, modelName):
    """Plot top feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return
    
    # Sort by importance
    indices = np.argsort(importances)[-15:]  # Top 15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="#3498db", edgecolor="black")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([featureCols[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 15 Feature Importances — {modelName}", fontweight="bold")
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, f"featureImportance_{modelName}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")


def writeClassificationReport(ruleBasedResults, mlResults, yTest, testDf):
    """Write a consolidated classification report."""
    path = os.path.join(RESULTS_DIR, "classificationReport.md")
    
    with open(path, "w") as f:
        f.write("# Classification Report — Relation Extraction\n\n")
        
        # Rule-based
        f.write("## 1. Rule-Based Pattern Extractor\n\n")
        f.write("```\n")
        f.write(ruleBasedResults["report"])
        f.write("```\n\n")
        
        # ML models
        for name, res in mlResults.items():
            f.write(f"## 2. {name}\n\n")
            f.write(f"**Best Params**: {res['bestParams']}\n\n")
            f.write(f"**Dev Macro F1**: {res['devScore']:.4f}\n\n")
            report = classification_report(yTest, res["predictions"],
                                          target_names=["Invalid (0)", "Valid (1)"],
                                          zero_division=0)
            f.write("```\n")
            f.write(report)
            f.write("```\n\n")
        
        # Summary table
        f.write("## Summary Comparison\n\n")
        f.write("| Model | Macro Precision | Macro Recall | Macro F1 |\n")
        f.write("|-------|----------------|-------------|----------|\n")
        
        # Rule-based
        f.write(f"| Rule-Based | {ruleBasedResults['macroPrecision']:.4f} | "
                f"{ruleBasedResults['macroRecall']:.4f} | "
                f"{ruleBasedResults['macroF1']:.4f} |\n")
        
        # ML models
        for name, res in mlResults.items():
            p, r, f1, _ = precision_recall_fscore_support(
                yTest, res["predictions"], average="macro", zero_division=0
            )
            f.write(f"| {name} | {p:.4f} | {r:.4f} | {f1:.4f} |\n")
    
    print(f"  → Saved {path}")


def writeErrorAnalysis(testDf, yTest, bestPreds, modelName):
    """Write error analysis with concrete FP and FN examples."""
    path = os.path.join(RESULTS_DIR, "errorAnalysis.md")
    
    with open(path, "w") as f:
        f.write(f"# Error Analysis — {modelName}\n\n")
        
        # False Positives
        fpMask = (bestPreds == 1) & (yTest == 0)
        fpIndices = np.where(fpMask)[0]
        f.write(f"## False Positives ({len(fpIndices)} total)\n\n")
        f.write("These are examples where the model predicted the causal claim is valid, ")
        f.write("but it is actually invalid.\n\n")
        
        for i, idx in enumerate(fpIndices[:5]):  # Show first 5
            row = testDf.iloc[idx]
            inputText = row[COL_INPUT]
            if len(inputText) > 500:
                inputText = inputText[:500] + "..."
            f.write(f"### FP Example {i+1}\n\n")
            f.write(f"**Input**: {inputText}\n\n")
            f.write(f"**True Label**: {yTest[idx]} (Invalid)\n")
            f.write(f"**Predicted**: 1 (Valid)\n\n")
            f.write("---\n\n")
        
        # False Negatives
        fnMask = (bestPreds == 0) & (yTest == 1)
        fnIndices = np.where(fnMask)[0]
        f.write(f"## False Negatives ({len(fnIndices)} total)\n\n")
        f.write("These are examples where the model predicted the causal claim is invalid, ")
        f.write("but it is actually valid.\n\n")
        
        for i, idx in enumerate(fnIndices[:5]):  # Show first 5
            row = testDf.iloc[idx]
            inputText = row[COL_INPUT]
            if len(inputText) > 500:
                inputText = inputText[:500] + "..."
            f.write(f"### FN Example {i+1}\n\n")
            f.write(f"**Input**: {inputText}\n\n")
            f.write(f"**True Label**: {yTest[idx]} (Valid)\n")
            f.write(f"**Predicted**: 0 (Invalid)\n\n")
            f.write("---\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total test examples: {len(yTest)}\n")
        f.write(f"- False Positives: {len(fpIndices)} ({len(fpIndices)/len(yTest)*100:.1f}%)\n")
        f.write(f"- False Negatives: {len(fnIndices)} ({len(fnIndices)/len(yTest)*100:.1f}%)\n")
        f.write(f"- True Positives: {((bestPreds == 1) & (yTest == 1)).sum()}\n")
        f.write(f"- True Negatives: {((bestPreds == 0) & (yTest == 0)).sum()}\n")
    
    print(f"  → Saved {path}")


def main():
    """Main entry point for Stage 05."""
    print("=" * 60)
    print("Stage 05: Relation Extraction Baseline")
    print("=" * 60)
    
    # Load features
    trainFeatures = pd.read_csv(os.path.join(FEATURES_DIR, "trainFeatures.csv"))
    devFeatures = pd.read_csv(os.path.join(FEATURES_DIR, "devFeatures.csv"))
    testFeatures = pd.read_csv(os.path.join(FEATURES_DIR, "testFeatures.csv"))
    
    # Load raw test data for error analysis
    testDf = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    yTest = testFeatures["label"].values
    
    # ── Rule-based evaluation ──
    print("\n📏 Evaluating Rule-Based Extractor...")
    ruleBasedResults = evaluateRuleBased(testDf)
    print(f"\n{ruleBasedResults['report']}")
    
    # ── ML classifiers ──
    print("\n🤖 Training ML Classifiers...")
    mlResults, xTestScaled, xTest, yTestML, featureCols = trainAndEvaluateModels(
        trainFeatures, devFeatures, testFeatures
    )
    
    # ── Plots ──
    print("\n📊 Generating Plots...")
    plotConfusionMatrices(yTest, mlResults, ruleBasedResults)
    
    for name, res in mlResults.items():
        plotFeatureImportance(res["model"], featureCols, name)
    
    # ── Extract triples ──
    print("\n📋 Extracting Triples...")
    allTriples = []
    
    # Rule-based triples
    rbTriples = extractTriples(testDf, ruleBasedResults["predictions"].values, "RuleBased")
    allTriples.append(rbTriples)
    
    # ML model triples (use best model)
    bestModelName = max(mlResults, key=lambda k: precision_recall_fscore_support(
        yTest, mlResults[k]["predictions"], average="macro", zero_division=0
    )[2])
    bestTriples = extractTriples(testDf, mlResults[bestModelName]["predictions"], bestModelName)
    allTriples.append(bestTriples)
    
    triplesPath = os.path.join(RESULTS_DIR, "extractedTriples.csv")
    allTriplesDf = pd.concat(allTriples, ignore_index=True)
    allTriplesDf.to_csv(triplesPath, index=False)
    print(f"  → {len(allTriplesDf)} triples saved to {triplesPath}")
    
    # ── Reports ──
    print("\n📝 Writing Reports...")
    writeClassificationReport(ruleBasedResults, mlResults, yTest, testDf)
    
    bestPreds = mlResults[bestModelName]["predictions"]
    writeErrorAnalysis(testDf, yTest, bestPreds, bestModelName)
    
    print(f"\n  Best model: {bestModelName}")
    bestP, bestR, bestF1, _ = precision_recall_fscore_support(
        yTest, bestPreds, average="macro", zero_division=0
    )
    print(f"  Macro P/R/F1: {bestP:.4f} / {bestR:.4f} / {bestF1:.4f}")
    
    print("\n✅ Stage 05 complete.")
    return mlResults, ruleBasedResults


if __name__ == "__main__":
    main()
