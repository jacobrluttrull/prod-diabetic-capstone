# Type 2 Diabetes Prediction App 🩺

An educational Streamlit app for WGU C964 showing how a trained ML model can estimate Type 2 Diabetes risk from basic inputs (age, BMI, HbA1c, blood glucose, etc.). It includes accuracy visuals (ROC, PR, threshold/F1), explainability (SHAP), and a clean user experience with logging and CSV export.

> **Not medical advice.** This is a capstone demo only.

---

## Features (mapped to Part C rubric)

* **Descriptive method**
  Model inspection/visuals — ROC curve, Precision–Recall curve, threshold sweep (F1/precision/recall), SHAP factor contributions.

* **Nondesc. method**
  **Predictive** classifier (calibrated scikit-learn pipeline).

* **Datasets**
  Sample diabetes CSV (see `data/`) + the trained model `.pkl`.

* **Decision support functionality**
  Adjustable decision threshold; risk tier; resources & next steps section.

* **Featurizing / parsing / cleaning**
  Automatic smoking normalization (adds `ever_smoked`), pipeline preprocessing.

* **Methods for exploration/prep**
  Model comparison and threshold sweep; HbA1c sensitivity plot.

* **Data visualization**
  ≥3 inside the app (ROC, PR, threshold sweep; plus SHAP bar and waterfall).

* **Interactive queries**
  UI controls, example cases, BMI calculator, DOB→Age helper, CSV export.

* **Machine learning**
  scikit-learn model + probability calibration; SHAP explanations.

* **Evaluate accuracy**
  AUC/AP, threshold sweep (F1/precision/recall), model comparison — shown in app.

* **Security**
  No PII collection; runs locally; optional on-disk logs; minimal dependencies.

* **Monitoring & maintenance**
  Lightweight local logs (`logs/predictions_log.csv`), clear file layout, reproducible plots.

* **User-friendly dashboard**
  Compact dark theme, accessible labels, expanders, and image zoom (**🔎 Full screen**) for visuals.

---

## Repo layout

```
.
├─ app/
│  └─ gui.py                          # Streamlit app (main entrypoint)
├─ models/
│  └─ calibrated_diabetes_model_v5.pkl
├─ data/
│  └─ diabetes_prediction_dataset.csv # (optional for scripts/plots)
├─ results/
│  ├─ roc_curve.png
│  ├─ pr_curve.png
│  ├─ comparison.png
│  └─ threshold_curve_prod_metrics.png
├─ tests/
│  ├─ make_threshold_sweep.py
│  └─ make_prod_threshold_metrics.py
├─ logs/                              # created at runtime (optional)
└─main.py                             # (optional; runs app)
├─ requirements.txt
└─ README.md
```

> If any `results/*.png` are missing, the app shows a friendly “Missing:” message in the **Model accuracy** expander.

---

## System requirements

* Windows 10/11 or macOS/Linux
* Python **3.10+** and pip
* Internet not required once installed

---

## Install & run

### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

> If PowerShell blocks scripts:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The browser will open at `http://localhost:8501`.

---

## Files the app expects

* `models/calibrated_diabetes_model_v5.pkl`
* `results/roc_curve.png`
* `results/pr_curve.png`
* `results/comparison.png`
* `results/threshold_curve_prod_metrics.png`
* `data/diabetes_prediction_dataset.csv` *(optional; used by plot scripts in `tests/`)*

---

## User guide

1. **About this App**
   Read the intro to understand the app’s purpose and limitations.

2. **Model accuracy (validation)**

   * View **ROC** (AUC) and **PR** (AP) curves.
   * See the **threshold sweep** for F1/precision/recall trade-offs.
   * Click **🔎 Full screen** to enlarge any plot inline.

3. **Inputs**
   Enter age, BMI, HbA1c, blood glucose, etc. Use the BMI calculator and DOB→Age helpers.

4. **Sidebar controls**

   * **Classification Threshold** slider for decision support.
   * **Example cases** (Low/Medium/High) to demo predictions.
   * **Log predictions locally (CSV)** toggle.
   * **Resources** links by region.

5. **Predict**
   Click **Predict** to get the probability and **Risk Tier** (Low/Medium/High).

6. **Explainability (SHAP)**

   * Factor contributions table with ↑/↓ and strength.
   * Waterfall plot in the **Advanced** expander.

7. **Download & logging**

   * **⬇️ Download Report (CSV)** for the current run.
   * If logging is enabled, each run is appended to `logs/predictions_log.csv`.

---

## Regenerate threshold/metrics plots (optional)

### PowerShell (Windows)

```powershell
$env:CAPSTONE_DATA="C:\path\to\repo\data\diabetes_prediction_dataset.csv"
$env:CAPSTONE_MODEL="C:\path\to\repo\models\calibrated_diabetes_model_v5.pkl"
$env:CAPSTONE_OUT="C:\path\to\repo\results"
$env:PREFERRED_THRESHOLD="0.245"

python tests\make_prod_threshold_metrics.py
```

### CMD (Windows)

```cmd
set CAPSTONE_DATA=C:\path\to\repo\data\diabetes_prediction_dataset.csv
set CAPSTONE_MODEL=C:\path\to\repo\models\calibrated_diabetes_model_v5.pkl
set CAPSTONE_OUT=C:\path\to\repo\results
set PREFERRED_THRESHOLD=0.245

python tests\make_prod_threshold_metrics.py
```

### macOS / Linux

```bash
export CAPSTONE_DATA="/path/to/repo/data/diabetes_prediction_dataset.csv"
export CAPSTONE_MODEL="/path/to/repo/models/calibrated_diabetes_model_v5.pkl"
export CAPSTONE_OUT="/path/to/repo/results"
export PREFERRED_THRESHOLD="0.245"

python tests/make_prod_threshold_metrics.py
```

> Quote paths that contain spaces.

---

## Troubleshooting

* **FileNotFoundError** for CSV or model: confirm the files exist at the listed paths.
* **Images not showing**: ensure the PNGs are in `results/` with the exact filenames.
* **PowerShell “Set-Variable … positional parameter …”**: use `$env:VAR=value` in PowerShell (not `set`, which is for CMD).
* **Port already in use**: `streamlit run app/gui.py --server.port 8081`

---

## License & attribution

* Educational use only. Not medical advice.
