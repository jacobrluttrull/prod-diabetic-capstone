# tests/make_prod_threshold_metrics.py
# Plot F1, Precision, Recall vs threshold for the *prod* model only.
# Writes:
#   results/threshold_metrics_prod.csv
#   results/threshold_curve_prod_metrics.png   <-- new name (won’t overwrite comparison.png)

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 120})

# -------------------------
# Config (override via env)
# -------------------------
DATA = os.environ.get("CAPSTONE_DATA", "data/diabetes_prediction_dataset.csv")
PROD_MODEL = os.environ.get("CAPSTONE_MODEL", "models/calibrated_diabetes_model_v5.pkl")
OUT_DIR = os.environ.get("CAPSTONE_OUT", "results")
TARGET_COL = os.environ.get("CAPSTONE_TARGET", "diabetes")

TH_MIN = float(os.environ.get("CAPSTONE_THRESH_MIN", "0.05"))
TH_MAX = float(os.environ.get("CAPSTONE_THRESH_MAX", "0.95"))
TH_STEP = float(os.environ.get("CAPSTONE_THRESH_STEP", "0.01"))

# Optional vertical marker (e.g., your UI default 0.26)
PREFERRED_THRESHOLD = os.environ.get("PREFERRED_THRESHOLD", "0.245").strip()
PREFERRED_THRESHOLD = float(PREFERRED_THRESHOLD) if PREFERRED_THRESHOLD else None

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

def load_data(path: str, target: str):
    if not os.path.exists(path):
        print(f"[ERROR] Data not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if target not in df.columns:
        print(f"[ERROR] Target column '{target}' not found in {path}", file=sys.stderr)
        sys.exit(1)

    # Minimal cleaning to match app expectations
    df = normalize_smoking(df)

    for c in ["age", "bmi", "HbA1c_level", "blood_glucose_level"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = [c for c in [
        "age", "bmi", "HbA1c_level", "blood_glucose_level",
        "hypertension", "heart_disease", "gender", "smoking_history", "ever_smoked", target
    ] if c in df.columns]
    df = df.dropna(subset=required)

    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def sweep_metrics(y_true, y_prob, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append((float(t), prec, rec, f1))
    return pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])

# -------------------------
# Main
# -------------------------
def main():
    print(f"[INFO] Data: {DATA}", file=sys.stderr)
    print(f"[INFO] Model: {PROD_MODEL}", file=sys.stderr)

    X, y = load_data(DATA, TARGET_COL)

    # Holdout test set (to mimic what you show in the app docs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Load pre-trained prod model
    if not os.path.exists(PROD_MODEL):
        print(f"[ERROR] Model not found at {PROD_MODEL}", file=sys.stderr)
        sys.exit(1)
    model = joblib.load(PROD_MODEL)

    # The prod model consumes raw columns (we normalize smoking like in the app)
    X_eval = normalize_smoking(X_test.copy())
    y_prob = model.predict_proba(X_eval)[:, 1]

    thresholds = np.arange(TH_MIN, TH_MAX + 1e-9, TH_STEP)
    df = sweep_metrics(y_test.values, y_prob, thresholds)

    # Save CSV
    out_csv = os.path.join(OUT_DIR, "threshold_metrics_prod.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote: {out_csv}", file=sys.stderr)

    # Find best F1
    best_idx = int(df["f1"].values.argmax())
    best_row = df.iloc[best_idx]
    best_t = float(best_row["threshold"])
    best_f1 = float(best_row["f1"])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", linewidth=2)
    ax.plot(df["threshold"], df["recall"], label="Recall", linewidth=2)
    ax.plot(df["threshold"], df["f1"], label="F1", linewidth=2)

    # Mark best F1
    ax.axvline(best_t, linestyle="--", linewidth=1, alpha=0.7, label=f"Best F1 @ {best_t:.2f}")
    ax.text(best_t, ax.get_ylim()[1]*0.97, f"{best_t:.2f}", rotation=90, va="top", ha="right", fontsize=8)

    # Mark preferred threshold (e.g., 0.26 from your UI)
    if PREFERRED_THRESHOLD is not None:
        ax.axvline(PREFERRED_THRESHOLD, linestyle=":", linewidth=1, alpha=0.7, label=f"Preferred @ {PREFERRED_THRESHOLD:.2f}")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Prod Model — Precision/Recall/F1 vs Threshold (Test Set)")
    ax.grid(True, linestyle=":")
    ax.legend()

    out_png = os.path.join(OUT_DIR, "threshold_curve_prod_metrics.png")  # <— new name
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[INFO] Wrote: {out_png}", file=sys.stderr)
    print(f"[INFO] Best F1: {best_f1:.3f} at threshold {best_t:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
