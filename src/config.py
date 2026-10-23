"""
Shared configuration and utility functions for the Corr2Cause AML pipeline.

This module centralises constants, paths, and helper functions used across
all stages so that no stage duplicates this logic.
"""

import os
import random
import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42

def setSeed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "corr2cause")
TRAIN_CSV = os.path.join(DATA_RAW_DIR, "train.csv")
DEV_CSV = os.path.join(DATA_RAW_DIR, "dev.csv")
TEST_CSV = os.path.join(DATA_RAW_DIR, "test.csv")

SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")

def stageDir(stageName: str) -> str:
    """Return the absolute path for a numbered stage folder."""
    return os.path.join(PROJECT_ROOT, stageName)

def ensureDir(path: str) -> str:
    """Create directory if it does not exist, return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def screenshotDir(stageName: str) -> str:
    """Return and create the screenshots subfolder for a given stage."""
    d = os.path.join(SCREENSHOTS_DIR, stageName)
    return ensureDir(d)

# ── Dataset column names ──────────────────────────────────────────────────
COL_INPUT = "input"
COL_LABEL = "label"
COL_NUM_VARS = "num_variables"
COL_TEMPLATE = "template"

# ── Label semantics ──────────────────────────────────────────────────────
LABEL_INVALID = 0   # The causal hypothesis does NOT follow from the premise
LABEL_VALID = 1     # The causal hypothesis DOES follow from the premise

# ── Template categories ──────────────────────────────────────────────────
TEMPLATES = [
    "child",
    "has_collider",
    "has_confounder",
    "non-child descendant",
    "non-parent ancestor",
    "parent",
]
