from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "risk_model.joblib"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
REFER_THRESHOLD = 0.25
URGENT_THRESHOLD = 0.60

_model = joblib.load(MODEL_PATH)


def predict_risk(values: dict) -> dict:
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    score = float(_model.predict_proba(row)[0, 1])

    if score >= URGENT_THRESHOLD:
        flag = "Refer to a doctor soon"
    elif score >= REFER_THRESHOLD:
        flag = "Refer to a doctor"
    else:
        flag = "No referral flag raised"

    return {
        "risk_score": round(score, 3),
        "prediction": "Higher risk" if score >= REFER_THRESHOLD else "Lower risk",
        "flag": flag,
        "disclaimer": "Screening aid only. Not a diagnosis. Always consult a doctor.",
    }
