
from __future__ import annotations

import os
import json
import math
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ----------------------
# Paths / config
# ----------------------
DATA_CSV = os.path.join("../data", "diabetes_prediction_dataset.csv")  # <-- adjust if needed
CURRENT_MODEL_PATH = os.path.join("../models", "calibrated_diabetes_model.pkl")
OUTPUT_MODEL_PATH = os.path.join("../models", "calibrated_diabetes_model_v4.pkl")
METRICS_JSON = os.path.join("../models", "test_metrics.json")
CM_PNG = os.path.join("../models", "confusion_matrix.png")
RELIABILITY_PNG = os.path.join("../models", "reliability_curves.png")
THRESH_SWEEP_CSV = os.path.join("../models", "threshold_sweep.csv")
RANDOM_STATE = 42

# ----------------------
# Helpers
# ----------------------

def file_sig(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return "unknown"


def ensure_dirs():
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray


def make_splits(df: pd.DataFrame) -> Split:
    X = df.drop(columns=["diabetes"])
    y = df["diabetes"].astype(int).values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=RANDOM_STATE
    )
    return Split(X_train, X_cal, X_test, y_train, y_cal, y_test)


def plot_reliability(ax, y_true, probs, label, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)


def threshold_sweep(y_true, probs, start=0.05, stop=0.75, step=0.01):
    rows = []
    thr = np.arange(start, stop + 1e-9, step)
    for t in thr:
        pred = (probs >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 3),
                "f1": f1_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred),
            }
        )
    sweep = pd.DataFrame(rows)
    best_f1 = sweep.sort_values("f1", ascending=False).iloc[0]
    return sweep, best_f1


# ----------------------
# 1) Load data & current model
# ----------------------
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Could not find data CSV at {DATA_CSV}")

df = pd.read_csv(DATA_CSV)
# Expect column 'diabetes' as target (0/1). Map if needed:
if df["diabetes"].dtype != int:
    df["diabetes"] = df["diabetes"].astype(int)

# Basic sanity checks / type coercions
binary_cols = ["hypertension", "heart_disease", "diabetes"]
for c in binary_cols:
    if c in df.columns:
        df[c] = df[c].astype(int)

categorical_cols = ["gender", "smoking_history"]
for c in categorical_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)

print("Data shape:", df.shape)
print("Positive rate:", df["diabetes"].mean())

# Load current calibrated model to reuse its preprocessing (ensures compatibility with app)
assert os.path.exists(CURRENT_MODEL_PATH), f"Missing current model: {CURRENT_MODEL_PATH}"
current_model = joblib.load(CURRENT_MODEL_PATH)
estimator = getattr(current_model, "estimator", current_model)  # pipeline if calibrated
assert isinstance(estimator, Pipeline), "Expected a Pipeline in the existing model"
preprocessor = estimator.named_steps["preprocessor"]
base_classifier = estimator.named_steps["classifier"]

print("Loaded current model:", CURRENT_MODEL_PATH, file_sig(CURRENT_MODEL_PATH))

# ----------------------
# 2) Make splits
# ----------------------
split = make_splits(df)
print(
    "Split sizes:",
    {"train": len(split.X_train), "cal": len(split.X_cal), "test": len(split.X_test)},
)

# ----------------------
# 3) Rebuild fresh pipeline and fit on TRAIN only
# ----------------------
from sklearn.utils import class_weight
neg, pos = np.bincount(split.y_train)
scale_pos_weight = neg / max(pos, 1)

xgb_params = base_classifier.get_params()
# Update a couple of training-time params safely
xgb_params.update(
    dict(
        n_estimators=300 if xgb_params.get("n_estimators", 100) < 300 else xgb_params["n_estimators"],
        learning_rate=min(xgb_params.get("learning_rate", 0.1), 0.05),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        base_score=pos / (pos + neg),
        random_state=RANDOM_STATE,
    )
)

xgb = XGBClassifier(**xgb_params)
pipe = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("classifier", xgb)])

pipe.fit(split.X_train, split.y_train)

# Uncalibrated probabilities on TEST
probs_uncal = pipe.predict_proba(split.X_test)[:, 1]

# ----------------------
# 4) Calibrate on CAL only — compare isotonic vs sigmoid
# ----------------------
cal_iso = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv="prefit").fit(
    split.X_cal, split.y_cal
)
cal_sig = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv="prefit").fit(
    split.X_cal, split.y_cal
)

probs_iso = cal_iso.predict_proba(split.X_test)[:, 1]
probs_sig = cal_sig.predict_proba(split.X_test)[:, 1]

# ----------------------
# 5) Metrics & selection
# ----------------------
metrics = {}
for name, p in {
    "uncalibrated": probs_uncal,
    "isotonic": probs_iso,
    "sigmoid": probs_sig,
}.items():
    metrics[name] = {
        "brier": float(brier_score_loss(split.y_test, p)),
        "auc": float(roc_auc_score(split.y_test, p)),
        "pr_auc": float(average_precision_score(split.y_test, p)),
    }

best_name = min(metrics.keys(), key=lambda k: metrics[k]["brier"])
print("Calibration choice by Brier (lower is better):", best_name, metrics[best_name])

chosen_cal = cal_iso if best_name == "isotonic" else (cal_sig if best_name == "sigmoid" else pipe)
probs_test = probs_iso if best_name == "isotonic" else (probs_sig if best_name == "sigmoid" else probs_uncal)

# ----------------------
# 6) Threshold sweep on TEST
# ----------------------
sweep, best_f1 = threshold_sweep(split.y_test, probs_test, start=0.05, stop=0.75, step=0.01)
sweep.to_csv(THRESH_SWEEP_CSV, index=False)
print(
    f"Best threshold by F1: t={best_f1['threshold']:.2f} (F1={best_f1['f1']:.3f}, P={best_f1['precision']:.3f}, R={best_f1['recall']:.3f})"
)

# For demo: also pick a threshold with recall >= 0.85 if available
thr_high_recall = None
candidates = sweep[sweep["recall"] >= 0.85]
if len(candidates) > 0:
    thr_high_recall = float(candidates.sort_values("threshold").iloc[0]["threshold"])

# ----------------------
# 7) Confusion matrix at best F1 threshold
# ----------------------
from sklearn.preprocessing import label_binarize
best_t = float(best_f1["threshold"])
pred_best = (probs_test >= best_t).astype(int)
cm = confusion_matrix(split.y_test, pred_best)

plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix @ t={best_t:.2f}")
plt.xticks([0, 1], ["Pred 0", "Pred 1"]) ; plt.yticks([0, 1], ["True 0", "True 1"])
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=160)
plt.close()

# ----------------------
# 8) Reliability curves plot
# ----------------------
plt.figure(figsize=(5.5, 5))
ax = plt.gca()
ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect")
plot_reliability(ax, split.y_test, probs_uncal, "Uncalibrated")
plot_reliability(ax, split.y_test, probs_iso, "Isotonic")
plot_reliability(ax, split.y_test, probs_sig, "Sigmoid")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Reliability (Test)")
ax.legend()
plt.tight_layout()
plt.savefig(RELIABILITY_PNG, dpi=160)
plt.close()

# ----------------------
# 9) Save chosen calibrated model + metrics
# ----------------------
ensure_dirs()
joblib.dump(chosen_cal, OUTPUT_MODEL_PATH)

out = {
    "chosen_calibration": best_name,
    "threshold_best_f1": best_t,
    "threshold_recall_0.85": thr_high_recall,
    "metrics": metrics,
    "artifacts": {
        "model_path": OUTPUT_MODEL_PATH,
        "model_sig": file_sig(OUTPUT_MODEL_PATH),
        "confusion_matrix_png": CM_PNG,
        "reliability_png": RELIABILITY_PNG,
        "threshold_sweep_csv": THRESH_SWEEP_CSV,
    },
}
with open(METRICS_JSON, "w") as f:
    json.dump(out, f, indent=2)

print("Saved model →", OUTPUT_MODEL_PATH)
print(json.dumps(out, indent=2))
