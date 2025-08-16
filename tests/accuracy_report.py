# tests/accuracy_report.py
from pathlib import Path
import os, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.calibration import CalibrationDisplay

# ---- Resolve paths from repo root (parent of /tests) ----
ROOT = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data" / "diabetes_prediction_dataset.csv"
MODEL = ROOT / "models" / "calibrated_diabetes_model_v5.pkl"
OUT   = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

# (Optional) make CWD the repo root so any other relative paths behave
os.chdir(ROOT)

# ---- Load data and prep (must mirror training) ----
df = pd.read_csv(DATA)
df["smoking_history"] = (
    df["smoking_history"].astype(str).str.strip().str.lower()
      .replace({"ever":"former","not current":"former","no info":"unknown","none":"unknown","nan":"unknown"})
)
df["ever_smoked"] = df["smoking_history"].isin(["former","current"]).astype(int)

features = ["age","hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level",
            "gender","smoking_history","ever_smoked"]
X = df[features]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

# ---- Load your calibrated pipeline ----
clf = joblib.load(MODEL)

# ---- Evaluate at your threshold ----
proba = clf.predict_proba(X_test)[:, 1]
threshold = 0.26
pred  = (proba >= threshold).astype(int)

acc  = accuracy_score(y_test, pred)
bacc = balanced_accuracy_score(y_test, pred)
f1   = f1_score(y_test, pred)
prec = precision_score(y_test, pred)
rec  = recall_score(y_test, pred)
auc  = roc_auc_score(y_test, proba)

# ---- Write text report ----
with open(OUT / "accuracy_report.txt", "w") as f:
    f.write(
        f"Accuracy: {acc:.3f}\nBalanced accuracy: {bacc:.3f}\nF1: {f1:.3f}\n"
        f"Precision: {prec:.3f}\nRecall: {rec:.3f}\nROC AUC: {auc:.3f}\n\n"
    )
    f.write(classification_report(y_test, pred))

# ---- Plots ----
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix (threshold = {threshold})")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=150)

fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
ax.set_title(f"ROC Curve (AUC = {auc:.3f})")
fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150)

fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
ax.set_title("Precision–Recall Curve")
fig.tight_layout(); fig.savefig(OUT / "pr_curve.png", dpi=150)

fig, ax = plt.subplots()
CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, ax=ax)
ax.set_title("Calibration Curve")
fig.tight_layout(); fig.savefig(OUT / "calibration_curve.png", dpi=150)

ths = np.linspace(0.05, 0.95, 37)
f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in ths]
fig, ax = plt.subplots()
ax.plot(ths, f1s, marker="o", lw=1)
ax.axvline(threshold, ls="--")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1"); ax.set_title("Threshold sweep")
fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=150)

# 5-fold CV on train for the table
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
pd.DataFrame({"fold": range(1, 6), "AUC": cv_auc, "mean": [cv_auc.mean()]*5, "std": [cv_auc.std()]*5}) \
  .to_csv(OUT / "cv_table.csv", index=False)
print(f"Saved figures/tables to {OUT}")
