
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"
model = joblib.load(MODEL_PATH)

def predict_customer(customer: dict) -> tuple[int, float]:
    row = pd.DataFrame([customer])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])
    return pred, prob
