# tests/make_threshold_sweep.py
# Multi-model threshold sweep (LogReg, RandomForest, and your prod model if available)
# Saves results/threshold_metrics.csv and results/comparison.png

import os
import sys
import math
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 120})

# -------------------------
# Config (env overrides)
# -------------------------
DATA = os.environ.get("CAPSTONE_DATA", "data/diabetes_prediction_dataset.csv")
PROD_MODEL = os.environ.get("CAPSTONE_MODEL", "models/calibrated_diabetes_model_v5.pkl")
OUT_DIR = os.environ.get("CAPSTONE_OUT", "results")
TARGET_COL = os.environ.get("CAPSTONE_TARGET", "diabetes")

TH_MIN = float(os.environ.get("CAPSTONE_THRESH_MIN", "0.05"))
TH_MAX = float(os.environ.get("CAPSTONE_THRESH_MAX", "0.95"))
TH_STEP = float(os.environ.get("CAPSTONE_THRESH_STEP", "0.01"))

# Comma-separated thresholds to highlight (e.g., "0.26,0.36")
MARK_THRESHOLDS = os.environ.get("MARK_THRESHOLDS", "").strip()

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def normalize_smoking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "smoking_history" in df.columns:
        v = df["smoking_history"].astype(str).str.strip().str.lower()
        v = v.replace({
            "ever": "former",
            "not current": "former",
            "no info": "unknown",
            "none": "unknown",
            "nan": "unknown",
        })
        v = v.where(v.isin(["never", "former", "current", "unknown"]), "unknown")
        df["smoking_history"] = v
        df["ever_smoked"] = v.isin(["former", "current"]).astype(int)
    else:
        df["smoking_history"] = "unknown"
        df["ever_smoked"] = 0
    return df

def load_data(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(path):
        print(f"[ERROR] Data not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if target not in df.columns:
        print(f"[ERROR] Target column '{target}' not found in {path}", file=sys.stderr)
        sys.exit(1)

    # Keep only the columns used by the app/pipeline (and target)
    expected = [
        "age", "hypertension", "heart_disease", "bmi",
        "HbA1c_level", "blood_glucose_level", "gender", "smoking_history"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in data: {missing} — proceeding with what is available.", file=sys.stderr)

    df = normalize_smoking(df)

    # Basic cleaning: coerce numerics if needed
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing required fields
    required = [c for c in ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", "gender", "smoking_history", "ever_smoked"] if c in df.columns]
    df = df.dropna(subset=required)

    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Columns we expect
    num = [c for c in ["age", "bmi", "HbA1c_level", "blood_glucose_level"] if c in X.columns]
    bin_cols = [c for c in ["hypertension", "heart_disease", "ever_smoked"] if c in X.columns]
    cat = [c for c in ["gender", "smoking_history"] if c in X.columns]

    # Pipelines
    num_pipe = Pipeline([("scaler", StandardScaler())])
    # For binary we just passthrough; ensure ints
    # For cat we one-hot encode
    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num),
            ("bin", "passthrough", bin_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return ct

def fit_baselines(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None, class_weight=None, solver="lbfgs")),
    ])
    logreg.fit(X_train, y_train)
    models["LogisticRegression"] = logreg

    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )),
    ])
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    return models

def maybe_load_prod_model(path: str):
    if not os.path.exists(path):
        print(f"[INFO] Prod model not found at {path} — skipping.", file=sys.stderr)
        return None
    try:
        m = joblib.load(path)
        # Must have predict_proba
        _ = m.predict_proba
        print(f"[INFO] Loaded prod model from {path}", file=sys.stderr)
        return m
    except Exception as e:
        print(f"[WARN] Could not load prod model: {e}", file=sys.stderr)
        return None

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append((float(t), prec, rec, f1))
    return pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])

def parse_marks(raw: str) -> List[float]:
    if not raw:
        return []
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except:
            pass
    return vals

# -------------------------
# Main
# -------------------------
def main():
    print(f"[INFO] Reading data: {DATA}", file=sys.stderr)
    X, y = load_data(DATA, TARGET_COL)

    # Consistent split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    baselines = fit_baselines(X_train, y_train, preprocessor)

    models: Dict[str, object] = {}
    models.update(baselines)

    prod = maybe_load_prod_model(PROD_MODEL)
    if prod is not None:
        models["ProdModel"] = prod

    # Threshold grid
    thresholds = np.arange(TH_MIN, TH_MAX + 1e-9, TH_STEP)

    all_rows = []
    best_by_model = {}

    for name, model in models.items():
        if name == "ProdModel":
            # The prod pipeline expects raw cols; we already normalized smoking
            X_eval = normalize_smoking(X_test.copy())
            y_prob = model.predict_proba(X_eval)[:, 1]
        else:
            # scikit pipelines (already include preprocessor)
            y_prob = model.predict_proba(X_test)[:, 1]

        df_sweep = sweep_thresholds(y_test.values, y_prob, thresholds)
        df_sweep["model"] = name
        all_rows.append(df_sweep)

        # Track best F1
        best_idx = int(df_sweep["f1"].values.argmax())
        best_row = df_sweep.iloc[best_idx]
        best_by_model[name] = (best_row["threshold"], best_row["f1"])

    out_csv = os.path.join(OUT_DIR, "threshold_metrics.csv")
    metrics = pd.concat(all_rows, ignore_index=True)
    metrics.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote: {out_csv}", file=sys.stderr)

    # Plot F1 vs threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in metrics["model"].unique():
        sub = metrics[metrics["model"] == name]
        ax.plot(sub["threshold"].values, sub["f1"].values, label=name, linewidth=2)

    # Mark any requested thresholds
    marks = parse_marks(MARK_THRESHOLDS)
    for m in marks:
        ax.axvline(m, linestyle="--", linewidth=1, alpha=0.6)
        ax.text(m, ax.get_ylim()[1]*0.95, f"{m:.2f}", rotation=90,
                va="top", ha="right", fontsize=8)

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Threshold (Test Set)")
    ax.grid(True, linestyle=":")
    ax.legend()
    out_png = os.path.join(OUT_DIR, "comparison.png")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[INFO] Wrote: {out_png}", file=sys.stderr)

    # Console summary
    print("\nBest F1 by model:", file=sys.stderr)
    for name, (t, f1v) in best_by_model.items():
        print(f"  {name:>15s}  best_th={t:.2f}  F1={f1v:.3f}", file=sys.stderr)

if __name__ == "__main__":
    main()
