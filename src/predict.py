
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "breast_cancer_wisconsin.csv"

MODEL_PATH = PROJECT_DIR / "models" / "logistic_regression.joblib"
SCALER_PATH = PROJECT_DIR / "models" / "standard_scaler.joblib"

FEATURE_NAMES = (
    pd.read_csv(DATA_PATH, nrows=0)
    .drop(columns=["target", "target_name"])
    .columns.tolist()
)


@lru_cache(maxsize=1)
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_one(input_data):
    model, scaler = load_models()

    input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        return "Malignant"
    else:
        return "Benign"


if __name__ == "__main__":
    print("Model and scaler loaded successfully!")
    print("Total features:", len(FEATURE_NAMES))
