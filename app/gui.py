import streamlit as st
import pandas as pd
import joblib
import os
import shap
from shap.plots import waterfall
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from datetime import datetime

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

# Friendly labels for smoking in the UI/report
_SMOKING_LABELS = {
    "never":   "Never",
    "former":  "Yes — former",
    "current": "Yes — current",
    "unknown": "Unknown",
}

# --- Resources helper (NEW) ---
def resource_links(region: str):
    global_links = [
        ("Find care near you (Google Maps)", "https://www.google.com/maps/search/primary+care+clinic+near+me"),
        ("Find a diabetes clinic (Google Maps)", "https://www.google.com/maps/search/diabetes+clinic+near+me"),
        ("WHO: Diabetes", "https://www.who.int/health-topics/diabetes"),
        ("International Diabetes Federation", "https://idf.org/"),
        ("Healthy eating (WHO)", "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"),
    ]
    by_region = {
        "United States": [
            ("Find a Community Health Center (HRSA)", "https://findahealthcenter.hrsa.gov/"),
            ("CDC: Prediabetes & Diabetes", "https://www.cdc.gov/diabetes/index.html"),
            ("American Diabetes Association", "https://diabetes.org/"),
        ],
        "United Kingdom": [
            ("NHS: Diabetes", "https://www.nhs.uk/conditions/diabetes/"),
            ("Find a GP service (NHS)", "https://www.nhs.uk/service-search/find-a-gp"),
        ],
        "Canada": [
            ("Government of Canada: Diabetes", "https://www.canada.ca/en/public-health/services/chronic-diseases/diabetes.html"),
            ("Find a clinic (Maps)", "https://www.google.com/maps/search/clinic+near+me"),
        ],
        "Australia": [
            ("Healthdirect: Diabetes", "https://www.healthdirect.gov.au/diabetes"),
            ("Find a GP (Healthdirect)", "https://www.healthdirect.gov.au/australian-health-services"),
        ],
        "India": [
            ("Indian Health Service", "https://www.ihs.gov/Diabetes/"),
            ("Find hospitals (Fortis Healthcare)", "https://www.fortishealthcare.com/"),
        ],
        "European Union": [
            ("IDF Europe)", "https://idf.org/europe/"),
        ],
    }
    return global_links + by_region.get(region, [])

# -------------------------
# Page + global style (Black & White / Sleek)
# -------------------------
st.set_page_config(page_title="Type 2 Diabetes Prediction", page_icon="🩺", layout="centered")

plt.rcParams.update({
    "figure.facecolor": "#0b0b0b",
    "axes.facecolor": "#0b0b0b",
    "axes.edgecolor": "#666666",
    "axes.labelcolor": "#EAEAEA",
    "text.color": "#EAEAEA",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "grid.color": "#333333",
    "axes.prop_cycle": cycler(color=["#FFFFFF"]),
})

st.markdown(
    """
    <style>
      :root { --bg:#0b0b0b; --panel:#121212; --text:#eaeaea; --muted:#a8a8a8; --border:#1e1e1e; }
      html,body,[data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
      [data-testid="stSidebar"]{ background:var(--panel); }
      h1,h2,h3,h4{ color:var(--text); }
      .stMarkdown,.stCaption,.stText,.stSelectbox,.stNumberInput{ color:var(--text); }
      div[data-testid="stMetric"] label, div[data-testid="stMetric"] div{ color:var(--text); }
      .block-container{ padding-top:2rem; }
      button[kind="primary"]{ background:#1a1a1a !important; color:var(--text) !important; border:1px solid var(--border) !important; }
      button[kind="secondary"]{ background:#111 !important; color:var(--text) !important; border:1px solid var(--border) !important; }
      .stSelectbox > div, .stNumberInput > div{ background:#111 !important; border:1px solid var(--border) !important; }
      [data-baseweb="slider"] div{ background-color:#222 !important; }
      [data-baseweb="slider"] div[role="slider"]{ background:#fff !important; }
      .stDataFrame{ border:1px solid var(--border); border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session defaults
# -------------------------
for k, v in {
    "age": 40,
    "hypertension": 0,
    "heart_disease": 0,
    "bmi": 25.0,
    "hba1c": 6.0,
    "glucose": 90,
    "gender": "Male",
    "smoking": "never",  # canonical options: never/former/current/unknown
    "show_prediction": False,
}.items():
    st.session_state.setdefault(k, v)

# -------------------------
# Load model + explainer
# -------------------------
MODEL_PATH = os.path.join("models", "calibrated_diabetes_model_v5.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def get_explainer(_trained_classifier, feature_names):
    return shap.Explainer(
        _trained_classifier,
        feature_names=feature_names,
        feature_perturbation="interventional",
    )

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.26, 0.01)
st.sidebar.caption(
    "Lower threshold catches more cases (higher sensitivity) but more false positives. "
    "Higher threshold is stricter but can miss cases. 0.26 was chosen from validation sweeps as a good balance. "
    "You can see the threshold justification in the results in the image below."
)
THRESHOLD_IMG_PATH = os.path.join("results", "threshold_curve.png")
if os.path.exists(THRESHOLD_IMG_PATH):
    st.sidebar.image(THRESHOLD_IMG_PATH, caption="Threshold sweep (F1 score)", use_container_width=True)

log_predictions = st.sidebar.checkbox("Log predictions locally (CSV)", value=False)

# --- Resources (sidebar picker) (NEW) ---
st.sidebar.header("Resources")
resources_region = st.sidebar.selectbox(
    "Show resources for:",
    ["Global", "United States", "United Kingdom", "Canada", "Australia", "India", "European Union"],
    index=0,
    key="resources_region",
)

with st.sidebar.expander("BMI Calculator", expanded=False):
    bmi_unit = st.radio("Units", ["US (lbs, ft/in)", "Metric (kg, cm)"], index=1, key="bmi_unit")

    def bmi_category(v: float) -> str:
        if v < 18.5: return "Underweight"
        elif v < 25: return "Normal weight"
        elif v < 30: return "Overweight"
        else: return "Obesity"

    bmi_calc = None
    if bmi_unit.startswith("US"):
        weight_lb = st.number_input("Weight (lbs):", min_value=50.0, max_value=550.0, step=0.5, key="bmi_weight_lbs")
        height_ft = st.number_input("Height (ft):", min_value=3, max_value=8, step=1, key="bmi_height_ft")
        height_in = st.number_input("Height (in):", min_value=0, max_value=11, step=1, key="bmi_height_in")
        total_in = (height_ft or 0) * 12 + (height_in or 0)
        if total_in > 0:
            bmi_calc = (weight_lb / (total_in ** 2)) * 703
    else:
        weight_kg = st.number_input("Weight (kg):", min_value=20.0, max_value=400.0, step=0.1, key="bmi_weight_kg")
        height_cm = st.number_input("Height (cm):", min_value=100.0, max_value=250.0, step=0.5, key="bmi_height_cm")
        if height_cm and height_cm > 0:
            bmi_calc = weight_kg / ((height_cm / 100) ** 2)

    if bmi_calc is not None and np.isfinite(bmi_calc):
        st.metric("Calculated BMI", f"{bmi_calc:.1f}", help="Body Mass Index = weight / height²")
        st.caption(f"Category: **{bmi_category(bmi_calc)}**")
        if st.button("Use this BMI"):
            st.session_state["bmi"] = float(round(bmi_calc, 1))
            st.rerun()

# Example cases quick-load
with st.sidebar.expander("Load example case"):
    example = st.selectbox("Examples", ["— None —", "Low risk", "Medium risk", "High risk"], index=0, key="example_case")
    if st.button("Use this example"):
        if example == "Low risk":
            ex = dict(age=25, hypertension=0, heart_disease=0, bmi=21.0, hba1c=5.2, glucose=85, gender="Female", smoking="never")
        elif example == "Medium risk":
            ex = dict(age=45, hypertension=1, heart_disease=1, bmi=29, hba1c=6.3, glucose=130, gender="Male", smoking="former")
        elif example == "High risk":
            ex = dict(age=62, hypertension=1, heart_disease=1, bmi=32.8, hba1c=7.1, glucose=160, gender="Female", smoking="current")
        else:
            ex = {}
        st.session_state.update(ex)
        st.session_state["show_prediction"] = False
        st.rerun()

    with st.sidebar.expander("DOB -> Age", expanded=False):
        from datetime import date
        dob = st.date_input("Date of birth", key="dob_input", min_value=date(1900, 1, 1), max_value=date.today())
        today = date.today()
        if dob > today:
            st.error("Date of birth cannot be in the future.")
        else:
            years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            months = (today.year - dob.year) * 12 + today.month - dob.month - (1 if today.day < dob.day else 0)
            months = max(months, 0)
            st.metric("Age", f"{years} years", help=f"≈ {months} months")
            if years < 1 or years > 120:
                st.caption("Age is calculated from Date of Birth. Please ensure it is correct and within 1–120 years.")
            if st.button("Use this age", key="use_dob_age"):
                st.session_state["age"] = int(years)
                st.rerun()

# -------------------------
# Header
# -------------------------

st.title("Type 2 Diabetes Prediction App")
with st.expander("ℹ️ About this App", expanded=False):
    st.write(
        """
        This is an educational tool I made to demonstrate an example of a machine learning model in practice
        for my Capstone project for WGU Computer Science program. The model predicts the risk of Type 2 Diabetes based on various health metrics.
        This tool is not intended for medical use and should not be used as a substitute for professional medical advice.
        This app was built for educational purposes only, showcasing how machine learning can be applied to health data.
        """
    )

# -------------------------
# Inputs (bound to session_state)
# -------------------------
age = st.number_input("Age:", min_value=1, max_value=120, key="age")
hypertension = st.selectbox("Hypertension:", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hypertension")
heart_disease = st.selectbox("Heart Disease:", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="heart_disease")
bmi = st.number_input("BMI:", min_value=10.0, max_value=60.0, step=0.1, key="bmi", help="Normal range: 18.5 - 24.9")
hba1c = st.number_input("HbA1c Level:", min_value=4.0, max_value=16.5, step=0.1, key="hba1c", help="Diabetes threshold: 6.5%")
glucose = st.number_input("Blood Sugar (mg/dL):", min_value=50, max_value=800, step=1, key="glucose", help="Diabetes threshold: 126 mg/dL")
gender = st.selectbox("Biological Gender:", ["Male", "Female"], key="gender")

# 🚬 Simplified smoking input (canonical values + friendly labels)
smoking = st.selectbox(
    "Smoking",
    ["never", "former", "current", "unknown"],
    format_func=lambda v: _SMOKING_LABELS[v],
    key="smoking",
)

# -------------------------
# Predict trigger
# -------------------------
if "show_prediction" not in st.session_state:
    st.session_state["show_prediction"] = False

def trigger_prediction():
    st.session_state["show_prediction"] = True

st.button("Predict", on_click=trigger_prediction)

# -------------------------
# Prediction block
# -------------------------
if st.session_state["show_prediction"]:
    input_df = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "gender": gender,
        "smoking_history": smoking,  # canonical value
    }])

    try:
        # Normalize to add ever_smoked + standardize categories
        input_df_norm = normalize_smoking(input_df.copy())

        prob = float(model.predict_proba(input_df_norm)[0][1])
        tier = "Low" if prob < 0.2 else ("Medium" if prob < 0.5 else "High")

        st.metric("Predicted Probability", f"{prob:.3f}")
        st.caption(f"Risk Tier: {tier}  •  Decision threshold = {threshold:.3f}")

        # --- Resources & next steps (only for Medium/High risk) (NEW) ---
        if tier in ("Medium", "High"):
            st.subheader("Resources & next steps")
            st.markdown(
                "These links can help you find care and learn more. "
                "**This app is educational only and not medical advice.**"
            )
            links = resource_links(resources_region)
            col1, col2 = st.columns(2)
            half = (len(links) + 1) // 2
            for i, (label, url) in enumerate(links):
                (col1 if i < half else col2).markdown(f"- [{label}]({url})")
            st.caption(
                "If you have very high blood sugar readings, new confusion, vision changes, "
                "chest pain, or trouble breathing, seek urgent medical care."
            )

        # Optional logging
        if log_predictions:
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", "predictions_log.csv")
            log_row = input_df_norm.copy()
            log_row["predicted_probability"] = prob
            log_row["risk_tier"] = tier
            log_row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            header = not os.path.exists(log_path)
            log_row.to_csv(log_path, mode="a", header=header, index=False)

        # Download (human-friendly)
        pretty = input_df.copy()
        pretty["hypertension"] = pretty["hypertension"].map({0: "No", 1: "Yes"})
        pretty["heart_disease"] = pretty["heart_disease"].map({0: "No", 1: "Yes"})
        pretty["Smoking"] = pretty["smoking_history"].map(_SMOKING_LABELS)
        pretty["Ever smoked"] = np.where(input_df_norm["ever_smoked"].values == 1, "Yes", "No")
        pretty.drop(columns=["smoking_history"], inplace=True)
        pretty.rename(columns={"HbA1c_level": "HbA1c", "blood_glucose_level": "Blood Sugar"}, inplace=True)
        pretty["Predicted Probability"] = prob
        pretty["Risk Tier"] = tier
        st.download_button("Download Report", data=pretty.to_csv(index=False), file_name="diabetes_prediction.csv", mime="text/csv")

        # Sensitivity plot (monochrome)
        with st.expander("Sensitivity: HbA1c"):
            hba1c_range = np.arange(4.0, 12.0, 0.1)
            probabilities = []
            for val in hba1c_range:
                temp_df = input_df.copy()
                temp_df["HbA1c_level"] = val
                temp_df = normalize_smoking(temp_df)
                try:
                    prob_val = float(model.predict_proba(temp_df)[0][1])
                except Exception:
                    prob_val = np.nan
                probabilities.append(prob_val)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(hba1c_range, probabilities, marker="o", linewidth=1.5, markersize=3)
            ax.axvline(6.5, linestyle="--", color="#BBBBBB", linewidth=1, label="Clinical cutoff (6.5%)")
            ax.set_title("Prediction sensitivity to HbA1c")
            ax.set_xlabel("HbA1c (%)")
            ax.set_ylabel("Probability of Diabetes")
            ax.legend(facecolor="#0b0b0b", edgecolor="#333")
            ax.grid(True, linestyle=":")
            for spine in ax.spines.values():
                spine.set_color("#666")
            st.pyplot(fig)

        # ---------------- SHAP Explainability ----------------
        st.subheader("Why this prediction?")
        try:
            pipeline = model.estimator
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]

            raw_feat_names = preprocessor.get_feature_names_out()

            # Friendly labels (v5 includes bin__ever_smoked)
            def friendly_label(raw_feature_name: str) -> str:
                if raw_feature_name.startswith("num__"):
                    col = raw_feature_name.split("__", 1)[1]
                    return {
                        "age": "Age",
                        "bmi": "BMI",
                        "HbA1c_level": "HbA1c",
                        "blood_glucose_level": "Blood Sugar",
                    }.get(col, col.replace("_", " ").title())
                if raw_feature_name.startswith("bin__"):
                    col = raw_feature_name.split("__", 1)[1]
                    return {
                        "hypertension": "Hypertension",
                        "heart_disease": "Heart Disease",
                        "ever_smoked": "Ever smoked",
                    }.get(col, col.replace("_", " ").title())
                if raw_feature_name.startswith("cat__"):
                    rest = raw_feature_name.split("__", 1)[1]
                    if rest.startswith("gender_"):
                        return "Gender"
                    return rest.replace("_", " ").title()
                return raw_feature_name.replace("_", " ").title()

            friendly_labels = [friendly_label(f) for f in raw_feat_names]

            explainer = get_explainer(classifier, raw_feat_names)
            X_row = preprocessor.transform(normalize_smoking(input_df.copy()))
            shap_raw = explainer(X_row)
            shap_vals = shap.Explanation(
                values=shap_raw.values[0],
                base_values=shap_raw.base_values[0],
                data=X_row[0],
                feature_names=friendly_labels,
            )

            raw_inputs = input_df.iloc[0].to_dict()

            # Map label -> user-friendly value
            def display_value(label: str) -> str:
                if label == "Gender":
                    return raw_inputs.get("gender", "unknown")
                if label == "Hypertension":
                    return "Yes" if int(raw_inputs.get("hypertension", 0)) == 1 else "No"
                if label == "Heart Disease":
                    return "Yes" if int(raw_inputs.get("heart_disease", 0)) == 1 else "No"
                if label == "Ever smoked":
                    es = int(normalize_smoking(input_df.copy())["ever_smoked"].iloc[0])
                    return "Yes" if es == 1 else "No"
                key_map = {"Age": "age", "BMI": "bmi", "HbA1c": "HbA1c_level", "Blood Sugar": "blood_glucose_level"}
                key = key_map.get(label)
                return raw_inputs.get(key, "unknown") if key else "unknown"

            # Rows (only active gender OHE; bin features pass through)
            rows, seen = [], set()
            for i, raw_name in enumerate(raw_feat_names):
                label = friendly_labels[i]
                sv = float(shap_vals.values[i])

                if raw_name.startswith("cat__"):
                    rest = raw_name.split("__", 1)[1]
                    if rest.startswith("gender_"):
                        active = f"cat__gender_{raw_inputs['gender']}"
                        if raw_name != active:
                            continue

                if (label,) in seen:
                    continue
                seen.add((label,))
                rows.append({"Feature": label, "SHAP value": sv, "Input value": display_value(label)})

            # Ensure Gender row exists even if SHAP=0
            if not any(r["Feature"] == "Gender" for r in rows):
                active_gender_col = f"cat__gender_{raw_inputs['gender']}"
                if active_gender_col in raw_feat_names:
                    j = list(raw_feat_names).index(active_gender_col)
                    sv = float(shap_vals.values[j])
                else:
                    sv = 0.0
                rows.append({"Feature": "Gender", "SHAP value": sv, "Input value": display_value("Gender")})

            # Build user-friendly view
            table_df = (
                pd.DataFrame(rows)
                .sort_values("SHAP value", key=lambda s: s.abs(), ascending=False)
                .reset_index(drop=True)
            )
            abs_vals = table_df["SHAP value"].abs()
            if len(abs_vals) >= 2:
                q1, q2 = np.quantile(abs_vals, [0.33, 0.66])
            else:
                q1 = q2 = abs_vals.max() if len(abs_vals) > 0 else 0

            def strength(v: float) -> str:
                a = abs(v)
                if a >= q2: return "strong"
                if a >= q1: return "medium"
                return "slight"

            def arrow(v: float) -> str:
                return "↑" if v > 0 else ("↓" if v < 0 else "→")

            table_df["Effect on risk"] = [f"{arrow(v)} {strength(v)}" for v in table_df["SHAP value"]]
            display_df = table_df[["Feature", "Input value", "Effect on risk"]].rename(columns={"Input value": "Your value"})
            display_df.index = display_df.index + 1
            display_df.index.name = "#"

            st.caption("**How to read this:** ↑ pushes probability *higher*, ↓ pushes it *lower*. Strength is relative across factors for *your* prediction.")
            st.dataframe(display_df, use_container_width=True)

            # Monochrome contribution bars (top 8)
            sub = table_df.head(8).iloc[::-1]
            fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.6 * len(sub))))
            ax2.barh(sub["Feature"], sub["SHAP value"], color="#FFFFFF", edgecolor="#FFFFFF")
            ax2.axvline(0, color="#777777", linewidth=1)
            ax2.set_xlabel("Contribution to risk (±)")
            ax2.set_ylabel("")
            for spine in ax2.spines.values():
                spine.set_color("#666")
            ax2.grid(True, axis="x", linestyle=":", linewidth=0.7)
            st.pyplot(fig2)

            with st.expander("See full details (advanced)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                waterfall(shap_vals, max_display=14, show=False)
                fig.patch.set_facecolor("#0b0b0b")
                ax.set_facecolor("#0b0b0b")
                st.pyplot(fig)
                st.caption(f"Internal explanation chart. Predicted probability = `{prob:.2f}`")

        except Exception as e:
            st.warning(f"SHAP explainability failed: {e}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
