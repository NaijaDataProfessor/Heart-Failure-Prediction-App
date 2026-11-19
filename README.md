# Heart Failure Prediction Streamlit App

A Streamlit educational prototype based on the Random Forest pipeline documented in the uploaded notebook.

## Notebook findings

- Model: Random Forest classifier in a scikit-learn pipeline
- Dataset: 918 records
- Split: 80/20, `random_state=42`
- Threshold explored: 0.40 (not a clinically validated cutoff)
- Numerical preprocessing: median imputation + standard scaling
- Categorical preprocessing: most-frequent imputation + ordinal encoding; unseen values encoded as -1
- Features: age, restingbp, cholesterol, fastingbs, maxhr, oldpeak, sex, chestpaintype, restingecg, exerciseangina, st_slope

The notebook's zero values in cholesterol were replaced with the median; zero resting blood pressure values were replaced with the median.

## Repository structure

```text
heart-failure-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── heart_failure.csv       
└── rf_model.joblib            
```

## Model/data setup

**Preferred:** export the notebook's fitted pipeline to `rf_model.joblib` and place it alongside `app.py`. The app will load it and will not retrain.

If no model artifact is present, place a compatible dataset CSV named `heart_failure.csv` in the project root. It must contain the following fields (case-insensitive; the app lowercases headers):

`age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope, heartdisease`

The app will clean the zero-coded values as described above, make a stratified-agnostic random 80/20 split with seed 42, fit a pipeline, and display holdout metrics at 0.40. This fallback is a deployment convenience, not a reproduction of an independently validated clinical model.

### Export the fitted pipeline from your notebook

After fitting `rf_model` in the notebook:

```python
import joblib
joblib.dump(rf_model, "model.joblib")
```

Copy `model.joblib` into the app repository root. Treat pickle/joblib artifacts as trusted files only; never load model artifacts from untrusted sources.

## Run locally in VS Code

1. Install Python 3.10+ and VS Code. Install the Python extension.
2. Open this folder in VS Code.
3. Create a virtual environment:

   **Windows PowerShell**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Streamlit opens the local app in your browser.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the project files.
2. Include `app.py` and `requirements.txt`. Include `model.joblib` if using the saved pipeline, or `heart_failure.csv` for fallback training.
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Select **Create app**, choose the repository, branch and `app.py`.
5. Deploy. Review the app logs if dependency or file-path issues arise.

## Clinical and ethical limitations

- This is a research/educational prototype, not a medical device or diagnostic tool.
- The notebook describes a single held-out split, not external or prospective validation.
- Performance and SHAP interpretations may not generalize to other populations, institutions, or data collection practices.
- Threshold selection is a model decision setting and is not a clinical cutoff.
- SHAP describes model associations, not causality.
- Do not use the output to make or withhold treatment, care, referral or emergency decisions.

## Notebook evaluation snapshot (threshold 0.40)

Reported test set: 184 records. Accuracy: approximately 0.89, precision: 0.89, recall: 0.92, F1: 0.90. Confusion matrix: 65 true negatives, 12 false positives, 9 false negatives, 98 true positives.

## SHAP findings reported in the notebook

| Feature | Mean absolute SHAP |
|---|---:|
| st_slope | 0.171 |
| chestpaintype | 0.096 |
| exerciseangina | 0.090 |
| oldpeak | 0.055 |
| maxhr | 0.041 |
| age | 0.037 |
| sex | 0.033 |
| fastingbs | 0.030 |
| cholesterol | 0.019 |
| restingbp | 0.014 |
| restingecg | 0.008 |

The notebook identified ST slope, chest pain type and exercise-induced angina as the most influential features. These are model-specific observations, not proof of causal or clinical importance.
