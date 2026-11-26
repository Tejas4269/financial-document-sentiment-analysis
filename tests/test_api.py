from fastapi.testclient import TestClient
from src.api import app
import src.api as api_module

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_model_returns_service_unavailable(monkeypatch):
    def missing_model(_text):
        raise FileNotFoundError("Model not trained yet")

    monkeypatch.setattr(api_module, "predict_sentiment", missing_model)
    response = client.post("/predict", json={"text": "Company reports record revenue"})
    assert response.status_code == 503
