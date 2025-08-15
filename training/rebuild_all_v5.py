# training/rebuild_all_v5.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
import joblib


# ----------------------------
# Helpers
# ----------------------------
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


def make_splits(df: pd.DataFrame, target="diabetes", rs=42):
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=rs
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=rs
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def threshold_sweep(y_true, p, lo=0.05, hi=0.75, step=0.01):
    rows=[]
    for t in np.arange(lo, hi+1e-9, step):
        yhat = (p >= t).astype(int)
        rows.append({
            "threshold": round(float(t),3),
            "f1": f1_score(y_true, yhat),
            "precision": precision_score(y_true, yhat, zero_division=0),
            "recall": recall_score(y_true, yhat),
        })
    df = pd.DataFrame(rows)
    best = df.sort_values("f1", ascending=False).iloc[0]
    return df, best


def plot_confusion_matrix_clean(cm: np.ndarray, threshold: float, src: str, out_png: Path, title: str|None=None):
    """White background, counts + row-percent annotations."""
    # Row % (normalize by actual class)
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(row_sum > 0, cm / row_sum * 100.0, 0.0)

    acc = (cm.trace() / cm.sum()) if cm.sum() else 0.0
    # Safe metrics: (we calculate them outside if needed)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=90, va="center")

    ax.set_xticks([0, 1], ["Pred: No Diabetes", "Pred: Diabetes"], rotation=20, ha="right")
    ax.set_yticks([0, 1], ["Actual: No Diabetes", "Actual: Diabetes"])

    vmax = cm.max() if cm.size else 0
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            perc = pct[i, j]
            text_color = "white" if vmax and cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, f"{count}\n{perc:.1f}%", ha="center", va="center",
                    fontsize=12, color=text_color, fontweight="bold")

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    ttl = title or "Confusion Matrix"
    ax.set_title(f"{ttl}  @ threshold={threshold:.2f} ({src})", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_reliability_curves(y_test, p_list, labels, out_png: Path):
    fig, ax = plt.subplots(figsize=(5.8, 5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot([0,1],[0,1],"--",color="#888",label="Perfect")

    for lbl, p in zip(labels, p_list):
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=lbl)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability (Test)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ----------------------------
def main():
    ap = argparse.ArgumentParser("Rebuild model + artifacts (v5, ever_smoked + clean plots)")
    ap.add_argument("--data", required=True, help="Path to diabetes_prediction_dataset.csv")
    ap.add_argument("--out-model", default=os.path.join("models","calibrated_diabetes_model_v5.pkl"))
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--title", default="Confusion Matrix")
    args = ap.parse_args()

    DATA = Path(args.data)
    OUT_MODEL = Path(args.out_model)
    RS = args.random_state
    TITLE = args.title
    out_dir = OUT_MODEL.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    METRICS_JSON = out_dir / "test_metrics.json"
    REL_PNG = out_dir / "reliability_curves.png"
    CM_PNG = out_dir / "confusion_matrix_clean.png"
    SWEEP_CSV = out_dir / "threshold_sweep.csv"
    SMOKING_CSV = out_dir / "smoking_prevalence_before_after.csv"

    # Load
    df_raw = pd.read_csv(DATA)
    for c in ["hypertension","heart_disease","diabetes"]:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].astype(int)
    for c in ["gender","smoking_history"]:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].astype(str)

    # Re-bucket smoking + add ever_smoked
    df = normalize_smoking(df_raw)

    # Prevalence table (before/after) for documentation
    prev_before = (
        df_raw.groupby("smoking_history")["diabetes"].mean()
        .rename("prevalence_before").reset_index()
    )
    prev_after = (
        df.groupby("smoking_history")["diabetes"].mean()
        .rename("prevalence_after").reset_index()
    )
    pd.merge(prev_before, prev_after, on="smoking_history", how="outer").to_csv(SMOKING_CSV, index=False)

    # Splits
    X_train, X_cal, X_test, y_train, y_cal, y_test = make_splits(df, rs=RS)

    # Preprocessor
    num_cols = ["age","bmi","HbA1c_level","blood_glucose_level"]
    bin_cols = ["hypertension","heart_disease","ever_smoked"]
    cat_cols = ["gender"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", categories=[["Male","Female"]]), cat_cols),
        ],
        remainder="drop"
    )

    # Model
    neg, pos = np.bincount(y_train)
    spw = neg / max(pos,1)
    xgb = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        eval_metric="logloss",
        scale_pos_weight=spw,
        base_score=pos/(pos+neg),
        random_state=RS,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", pre),
        ("classifier", xgb),
    ])

    # Fit on TRAIN
    pipe.fit(X_train, y_train)

    # Uncalibrated on TEST
    p_uncal = pipe.predict_proba(X_test)[:,1]

    # Calibrate on CAL
    cal_iso = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv="prefit").fit(X_cal, y_cal)
    cal_sig = CalibratedClassifierCV(estimator=pipe, method="sigmoid",  cv="prefit").fit(X_cal, y_cal)
    p_iso = cal_iso.predict_proba(X_test)[:,1]
    p_sig = cal_sig.predict_proba(X_test)[:,1]

    def summarize(y, p):
        return {
            "brier": float(brier_score_loss(y,p)),
            "auc": float(roc_auc_score(y,p)),
            "pr_auc": float(average_precision_score(y,p)),
        }

    metrics = {
        "uncalibrated": summarize(y_test, p_uncal),
        "isotonic": summarize(y_test, p_iso),
        "sigmoid": summarize(y_test, p_sig),
    }
    best_name = min(["isotonic","sigmoid"], key=lambda k: metrics[k]["brier"])
    chosen = cal_iso if best_name=="isotonic" else cal_sig
    p_test = p_iso if best_name=="isotonic" else p_sig

    # Threshold sweep on TEST
    sweep, best = threshold_sweep(y_test, p_test)
    sweep.to_csv(SWEEP_CSV, index=False)
    best_t = float(best["threshold"])

    # Confusion matrix @ best t
    yhat = (p_test >= best_t).astype(int)
    cm = confusion_matrix(y_test, yhat, labels=[0,1])
    plot_confusion_matrix_clean(cm, best_t, src=f"best F1 ({best_name})", out_png=CM_PNG, title=TITLE)

    # Reliability curves (save)
    plot_reliability_curves(y_test, [p_uncal, p_iso, p_sig], ["Uncalibrated","Isotonic","Sigmoid"], REL_PNG)

    # Save model + metrics
    joblib.dump(chosen, OUT_MODEL)
    with open(METRICS_JSON, "w") as f:
        json.dump({
            "chosen_calibration": best_name,
            "threshold_best_f1": best_t,
            "metrics": metrics,
            "artifacts": {
                "model_path": str(OUT_MODEL),
                "reliability_png": str(REL_PNG),
                "confusion_matrix_png": str(CM_PNG),
                "threshold_sweep_csv": str(SWEEP_CSV),
                "smoking_prevalence_csv": str(SMOKING_CSV),
                "data_path": str(DATA),
            },
        }, f, indent=2)

    print("Saved →", OUT_MODEL)
    print("Best calibration:", best_name, "Best-F1 threshold:", f"{best_t:.2f}")
    print("Artifacts:")
    print("  ", REL_PNG)
    print("  ", CM_PNG)
    print("  ", SWEEP_CSV)
    print("  ", SMOKING_CSV)
    print("  ", METRICS_JSON)


if __name__ == "__main__":
    main()
