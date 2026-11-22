
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint(monkeypatch):
    from app import main

    monkeypatch.setattr(
        main,
        "predict_one",
        lambda input_data: "Malignant"
    )

    sample_data = {
        "features": [1.0] * 30
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert response.json()["prediction"] == "Malignant"
    assert response.json()["total_features"] == 30


def test_prediction_rejects_invalid_feature_count():
    sample_data = {
        "features": [1.0, 2.0, 3.0]
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 422
