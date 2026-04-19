"""Integration test for FastAPI — uses a stub model (no MLflow needed)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def _make_stub_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture()
def client():
    with patch("churn.serving.loader.load_model"), \
         patch("churn.serving.loader.get_model", return_value=_make_stub_model()), \
         patch("churn.serving.loader.get_model_version", return_value="1"):
        from churn.serving.app import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_returns_probability(client):
    payload = {
        "customer_id": "C001",
        "ticket_freq_7d": 3,
        "ticket_freq_30d": 8,
        "ticket_freq_90d": 15,
        "ticket_sentiment_score": -0.4,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["customer_id"] == "C001"
    assert "churn_label" in data


def test_predict_rejects_invalid_sentiment(client):
    payload = {
        "customer_id": "C001",
        "ticket_freq_7d": 3,
        "ticket_freq_30d": 8,
        "ticket_freq_90d": 15,
        "ticket_sentiment_score": 5.0,  # invalid
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    assert "features" in resp.json()
