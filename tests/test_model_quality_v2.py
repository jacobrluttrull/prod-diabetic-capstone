import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

# Optional: SHAP smoke test will be skipped if shap is missing
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/calibrated_diabetes_model_v5.pkl"))
METRICS_JSON = Path(os.getenv("METRICS_JSON", "models/test_metrics.json"))

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def md5_sig(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def normalize_smoking(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize smoking_history to {never, former, current, unknown} and add ever_smoked (0/1)."""
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


@pytest.fixture(scope="session")
def model():
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    m = joblib.load(MODEL_PATH)
    # CalibratedClassifierCV or bare Pipeline both acceptable
    return m


@pytest.fixture()
def baseline_row() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "age": 40,
            "hypertension": 0,
            "heart_disease": 0,
            "bmi": 24.0,
            "HbA1c_level": 5.5,
            "blood_glucose_level": 95,
            "gender": "Male",
            "smoking_history": "never",
        }
    ])


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_model_loads_and_predicts_prob_in_0_1(model, baseline_row):
    X = normalize_smoking(baseline_row)
    proba = float(model.predict_proba(X)[0][1])
    assert 0.0 <= proba <= 1.0


def test_batch_prediction_shape(model, baseline_row):
    X = pd.concat([baseline_row] * 5, ignore_index=True)
    X = normalize_smoking(X)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_probability_increases_with_hba1c_and_glucose(model, baseline_row):
    # HbA1c sensitivity
    low = baseline_row.copy()
    high = baseline_row.copy()
    low["HbA1c_level"] = 5.5
    high["HbA1c_level"] = 8.5
    low = normalize_smoking(low)
    high = normalize_smoking(high)
    p_low = float(model.predict_proba(low)[0][1])
    p_high = float(model.predict_proba(high)[0][1])
    assert p_high >= p_low + 0.03, f"Expected higher prob with elevated HbA1c: {p_low:.3f} -> {p_high:.3f}"

    # Glucose sensitivity (hold HbA1c at 6.0, vary glucose)
    g1 = baseline_row.copy()
    g2 = baseline_row.copy()
    g1["HbA1c_level"] = 6.0
    g2["HbA1c_level"] = 6.0
    g1["blood_glucose_level"] = 90
    g2["blood_glucose_level"] = 180
    g1 = normalize_smoking(g1)
    g2 = normalize_smoking(g2)
    p1 = float(model.predict_proba(g1)[0][1])
    p2 = float(model.predict_proba(g2)[0][1])
    assert p2 >= p1 + 0.03, f"Expected higher prob with elevated glucose: {p1:.3f} -> {p2:.3f}"


def test_metrics_artifacts_and_consistency(model):
    # Metrics JSON is optional but recommended
    if not METRICS_JSON.exists():
        pytest.skip("metrics json not found; skipping artifact checks")

    with open(METRICS_JSON, "r") as f:
        meta = json.load(f)

    # Basic structure
    assert "metrics" in meta and isinstance(meta["metrics"], dict)
    assert "chosen_calibration" in meta
    assert "threshold_best_f1" in meta

    # AUC sanity
    auc = meta["metrics"].get("isotonic", {}).get("auc") or meta["metrics"].get("sigmoid", {}).get("auc")
    if auc is not None:
        assert 0.7 <= float(auc) <= 1.0

    # Model signature matches (if present)
    m_sig = meta.get("artifacts", {}).get("model_sig")
    if m_sig:
        assert m_sig == md5_sig(MODEL_PATH), "Model checksum mismatch vs test_metrics.json"


@pytest.mark.skipif(not _HAS_SHAP, reason="shap not installed; skipping explainability smoke test")
def test_shap_smoke_single_row(model, baseline_row):
    # Access underlying pipeline for SHAP
    estimator = getattr(model, "estimator", model)
    preprocessor = estimator.named_steps["preprocessor"]
    classifier = estimator.named_steps["classifier"]

    X_row = normalize_smoking(baseline_row)
    X_enc = preprocessor.transform(X_row)

    explainer = shap.Explainer(classifier, feature_names=preprocessor.get_feature_names_out(),
                               feature_perturbation="interventional")
    sv = explainer(X_enc)
    assert sv.values.shape[-1] == len(preprocessor.get_feature_names_out())

def test_high_risk_scenario_above_low_risk(model, baseline_row):
    # Baseline
    low = normalize_smoking(baseline_row.copy())

    # High-risk profile (edit row 0 in-place)
    high = baseline_row.copy()
    updates = {
        "age": 62,
        "hypertension": 1,
        "heart_disease": 1,
        "bmi": 33.0,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 170,
        "smoking_history": "current",
    }
    for k, v in updates.items():
        high.at[0, k] = v
    high = normalize_smoking(high)

    p_low = float(model.predict_proba(low)[0][1])
    p_high = float(model.predict_proba(high)[0][1])

    assert p_high >= p_low + 0.10, (
        f"Expected high-risk prob much higher than baseline: {p_low:.3f} -> {p_high:.3f}"
    )


