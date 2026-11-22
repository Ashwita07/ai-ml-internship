
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_one, FEATURE_NAMES

app = FastAPI(
    title="MLOps Capstone API",
    version="0.1.0"
)


class PredictionInput(BaseModel):
    features: list[float] = Field(
        min_length=30,
        max_length=30,
        description="Enter exactly 30 feature values in the dataset's column order."
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        prediction = predict_one([data.features])

        return {
            "prediction": prediction,
            "total_features": len(FEATURE_NAMES)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
