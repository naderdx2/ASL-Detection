import pytest
import time
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "labels" in data
    assert data["num_classes"] > 0

def test_predict_landmarks_valid():
    response = client.post("/predict/landmarks", json={"landmarks": [0.0] * 63})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_letter" in data
    assert "confidence" in data
    assert len(data["top3"]) == 3

def test_predict_landmarks_invalid():
    response = client.post("/predict/landmarks", json={"landmarks": [0.0] * 10})
    assert response.status_code == 400

def test_clear_session():
    response = client.delete("/session")
    assert response.status_code == 200

def test_response_time():
    start = time.time()
    client.get("/")
    assert (time.time() - start) * 1000 < 500

def test_confidence_range():
    response = client.post("/predict/landmarks", json={"landmarks": [0.3] * 63})
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0