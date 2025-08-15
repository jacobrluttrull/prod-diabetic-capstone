"""
Plot Threshold Curve
--------------------
Creates a chart of F1 / Precision / Recall vs. threshold and highlights the
best-F1 operating point. Also writes a CSV of the sweep.

Usage (PowerShell / CMD):
  python training/plot_threshold_curve.py ^
    --model "models\\calibrated_diabetes_model_v5.pkl" ^
    --data "data\\diabetes_prediction_dataset.csv" ^
    --out "models\\threshold_curve.png"

Notes:
- Expects target column 'diabetes' in the CSV.
- Works with your v5 model (adds 'ever_smoked' if missing).
"""
from __future__ import annotations
import os, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_MODEL = os.path.join("models", "calibrated_diabetes_model_v5.pkl")
DEFAULT_DATA  = os.path.join("data", "diabetes_prediction_dataset.csv")
DEFAULT_OUT   = os.path.join("models", "threshold_curve.png")
DEFAULT_SWEEP = os.path.join("models", "threshold_sweep.csv")
RANDOM_STATE  = 42

def normalize_smoking(df: pd.DataFrame) -> pd.DataFrame:
    """Match the v5 preprocessing: standardize smoking + add ever_smoked."""
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

def threshold_sweep(y_true, probs, start=0.05, stop=0.35, step=0.01) -> pd.DataFrame:
    ts = np.arange(start, stop + 1e-12, step)
    rows = []
    for t in ts:
        pred = (probs >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "f1": f1_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
        })
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out",  default=DEFAULT_OUT)
    p.add_argument("--sweep_csv", default=DEFAULT_SWEEP)
    p.add_argument("--start", type=float, default=0.05)
    p.add_argument("--stop",  type=float, default=0.95)
    p.add_argument("--step",  type=float, default=0.01)
    args = p.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    if "diabetes" not in df.columns:
        raise KeyError("Expected target column 'diabetes' in dataset.")
    y = df["diabetes"].astype(int).values
    X = df.drop(columns=["diabetes"]).copy()

    # Minimal type fixes
    for c in ["hypertension", "heart_disease"]:
        if c in X.columns:
            X[c] = X[c].astype(int)
    for c in ["gender", "smoking_history"]:
        if c in X.columns:
            X[c] = X[c].astype(str)

    # Split and normalize smoking (v5 feature)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_te = normalize_smoking(X_te)

    # Predict probabilities
    probs = model.predict_proba(X_te)[:, 1]

    # Sweep thresholds
    sweep = threshold_sweep(y_te, probs, start=args.start, stop=args.stop, step=args.step)
    sweep.to_csv(args.sweep_csv, index=False)

    # Best F1
    best = sweep.sort_values("f1", ascending=False).iloc[0]
    t_best = float(best["threshold"])

    # Plot (monochrome-friendly)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(sweep["threshold"], sweep["f1"],        label="F1")
    plt.plot(sweep["threshold"], sweep["precision"], label="Precision")
    plt.plot(sweep["threshold"], sweep["recall"],    label="Recall")
    plt.axvline(t_best, linestyle="--", color="black", alpha=0.6, label=f"Best F1 @ {t_best:.2f}")
    plt.title("Operating Threshold vs Metrics (Test)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    plt.close()

    # Also stash a small json next to the image
    meta = {
        "best_f1_threshold": t_best,
        "best_row": best.to_dict(),
        "sweep_csv": str(Path(args.sweep_csv).as_posix()),
        "model_path": str(Path(args.model).as_posix()),
        "data_path": str(Path(args.data).as_posix()),
    }
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved chart → {args.out}")
    print(f"Best F1 threshold: {t_best:.3f}")

if __name__ == "__main__":
    main()
