
from pathlib import Path
from functools import lru_cache

import mlflow.sklearn
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "breast_cancer_wisconsin.csv"

FEATURE_NAMES = (
    pd.read_csv(DATA_PATH, nrows=0)
    .drop(columns=["target", "target_name"])
    .columns.tolist()
)

MODEL_URI = "models:/m-3595d51e65bd49db9b0e93897f00696c"
SCALER_URI = "models:/m-602c0400632b4fc88be7566237d33041"


@lru_cache(maxsize=1)
def load_models():
    model = mlflow.sklearn.load_model(MODEL_URI)
    scaler = mlflow.sklearn.load_model(SCALER_URI)
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
