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
ARTIFACT = BASE / "rf_model.joblib"
DATASET = BASE / "heart.csv"

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
    fastingbs = c6.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
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
        patient = pd.DataFrame([{
            "age": age,
            "restingbp": restingbp,
            "cholesterol": cholesterol,
            "fastingbs": fastingbs,
            "maxhr": maxhr,
            "oldpeak": oldpeak,
            "sex": sex,
            "chestpaintype": chestpain,
            "restingecg": restingecg,
            "exerciseangina": exerciseangina,
            "st_slope": st_slope
        }], columns=FEATURES)

        try:
            probability = float(model.predict_proba(patient)[0, 1])
            predicted_status = int(probability >= threshold)

            st.divider()
            st.subheader("Predicted Heart Disease Status")

            if predicted_status == 1:
                st.error("## Heart Disease Predicted")

                explanation = (
                    f"The model estimated a {probability:.1%} probability "
                    f"for the heart disease class. This meets or exceeds "
                    f"the selected decision threshold of {threshold:.0%}, "
                    f"so the model classified this input as heart disease."
                )

            else:
                st.success("## No Heart Disease Predicted")

                explanation = (
                    f"The model estimated a {probability:.1%} probability "
                    f"for the heart disease class. This is below the selected "
                    f"decision threshold of {threshold:.0%}, so the model "
                    f"classified this input as no heart disease."
                )

            st.write(explanation)

            st.divider()

            st.subheader("Model Probability")

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated probability of heart disease",
                f"{probability:.1%}"
            )

            col2.metric(
                "Decision threshold",
                f"{threshold:.0%}"
            )

            st.progress(min(max(probability, 0.0), 1.0))

            st.caption(
                "The probability represents the model's estimated likelihood "
                "of the heart disease class based on the input features. "
                "It is not necessarily a clinically calibrated individual risk."
            )

            st.info(
                "The predicted status is determined by comparing the model "
                "probability with the selected threshold. Changing the threshold "
                "may change the predicted status even when the probability remains unchanged."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
