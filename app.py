import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Heart Failure Prediction", page_icon="🫀", layout="wide")
st.title("🫀 Heart Failure Risk Prediction")
st.caption("Educational decision-support prototype — not a diagnostic tool.")

NUM = ["age", "restingbp", "cholesterol", "fastingbs", "maxhr", "oldpeak"]
CAT = ["sex", "chestpaintype", "restingecg", "exerciseangina", "st_slope"]
FEATURES = NUM + CAT
TARGET = "heartdisease"
DEFAULT_THRESHOLD = 0.40
BASE = Path(__file__).parent
ARTIFACT = Path(r"C:\Users\Data Professor\Downloads\Heart failure App\rf_model.joblib")
DATASET = Path(r"C:\Users\Data Professor\Downloads\Heart failure App\heart.csv")

def make_pipeline():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    processor = ColumnTransformer([("numerical", num_pipe, NUM),
                                   ("categorical", cat_pipe, CAT)], remainder="drop")
    return Pipeline([("processor", processor), ("model_name", RandomForestClassifier(random_state=42))])

@st.cache_resource
def load_or_train():
    if ARTIFACT.exists():
        obj = joblib.load(ARTIFACT)
        if not hasattr(obj, "predict_proba"):
            raise ValueError("model.joblib must contain a fitted pipeline with predict_proba.")
        return obj, "Loaded supplied model.joblib", None
    if not DATASET.exists():
        raise FileNotFoundError("No model.joblib or heart_failure.csv found. Add a fitted pipeline artifact or training CSV.")
    df = pd.read_csv(DATASET)
    df.columns = df.columns.str.lower().str.strip()
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].copy()
    # Preserve the notebook's explicit zero-value cleaning.
    if "cholesterol" in df:
        df["cholesterol"] = df["cholesterol"].replace(0, df["cholesterol"].median())
    if "restingbp" in df:
        df["restingbp"] = df["restingbp"].replace(0, df["restingbp"].mean())
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= DEFAULT_THRESHOLD).astype(int)
    metrics = {"Accuracy": accuracy_score(y_test, pred),
               "Precision": precision_score(y_test, pred, zero_division=0),
               "Recall": recall_score(y_test, pred, zero_division=0),
               "F1": f1_score(y_test, pred, zero_division=0)}
    return pipeline, "Trained from heart_failure.csv (80/20 split; random_state=42)", metrics

try:
    model, model_source, metrics = load_or_train()
except Exception as e:
    st.error(f"Model setup issue: {e}")
    st.info("Place a fitted pipeline in model.joblib or a compatible CSV named heart_failure.csv in this repository.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    threshold = st.slider("Prediction threshold", min_value=0.05, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.05,
                         help="Probability cutoff used to flag a positive prediction. The notebook tested 0.40; this is not a clinically validated cutoff.")
    st.caption(model_source)
    st.divider()
    st.caption("The threshold changes the classification flag, not the model's probability.")

if metrics:
    st.subheader("Holdout performance (local training mode)")
    cols = st.columns(4)
    for col, (label, val) in zip(cols, metrics.items()):
        col.metric(label, f"{val:.3f}")
    st.caption("These metrics are calculated from a fresh split of the supplied CSV, not from an independent external validation cohort.")

st.subheader("Patient information")
st.write("Enter the clinical and exercise-test values below. Use only values obtained and interpreted by qualified healthcare professionals.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", min_value=18, max_value=120, value=55)
    sex = c2.selectbox("Sex", ["M", "F"])
    chestpain = c3.selectbox("Chest pain type", ["ASY", "ATA", "NAP", "TA"],
                             help="ASY: asymptomatic; ATA: atypical angina; NAP: non-anginal pain; TA: typical angina.")
    c4, c5, c6 = st.columns(3)
    restingbp = c4.number_input("Resting blood pressure (mmHg)", min_value=1, max_value=300, value=130)
    cholesterol = c5.number_input("Cholesterol (mg/dL)", min_value=1, max_value=1000, value=200)
    fastingbs = c6.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    c7, c8, c9 = st.columns(3)
    restingecg = c7.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = c8.number_input("Maximum heart rate achieved (bpm)", min_value=1, max_value=300, value=150)
    exerciseangina = c9.selectbox("Exercise-induced angina", ["N", "Y"], format_func=lambda x: "No" if x == "N" else "Yes")
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-10.0, max_value=20.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])
    submitted = st.form_submit_button("Estimate model probability", type="primary")

if submitted:
    if age < 18:
        st.error("This prototype is intended for adult inputs.")
    else:
        patient = pd.DataFrame([{"age": age, "restingbp": restingbp, "cholesterol": cholesterol,
                                "fastingbs": fastingbs, "maxhr": maxhr, "oldpeak": oldpeak,
                                "sex": sex, "chestpaintype": chestpain, "restingecg": restingecg,
                                "exerciseangina": exerciseangina, "st_slope": st_slope}], columns=FEATURES)
        try:
            probability = float(model.predict_proba(patient)[0, 1])
            flagged = probability >= threshold
            st.divider()
            st.subheader("Prediction result")
            left, right = st.columns([1, 2])
            left.metric("Model probability", f"{probability:.1%}")
            if flagged:
                right.warning(f"Model flag: probability meets/exceeds the selected {threshold:.0%} threshold.")
            else:
                right.success(f"Model flag: probability is below the selected {threshold:.0%} threshold.")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption("This flag is generated by a statistical model and is not a diagnosis, clinical risk category, or treatment recommendation.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.subheader("Important clinical limitations")
st.warning("""
- This is an educational prototype, not a medical device and not a diagnostic system.
- The notebook used 918 records and a single random 80/20 split. External, prospective and subgroup validation were not demonstrated.
- Model performance may change across populations, institutions, measurement practices and disease prevalence.
- A probability is not the same as an individual's clinically calibrated risk. Calibration and threshold selection require further evaluation.
- The model may miss disease or flag people who do not have disease. Do not use it to make or withhold care, treatment, referral or emergency decisions.
- Symptoms such as chest pain, shortness of breath, fainting or other urgent concerns require appropriate clinical assessment regardless of this app's output.
""")
st.caption("Notebook SHAP analysis reported ST slope, chest pain type and exercise angina as the strongest drivers. SHAP values describe model associations, not causal effects.")
