# tests/test_serve.py
import pytest
from fastapi.testclient import TestClient
from madewithml.serve import app

# Initialize the test client with our FastAPI app
client = TestClient(app)

def test_health_check():
    """Ensure the root endpoint returns a 200 OK status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status-code"] == 200
    assert "environment" in response.json()

def test_predict_validation_error():
    """Ensure the API rejects bad payloads that don't match the Pydantic schema."""
    # Missing the 'description' field
    bad_payload = {"title": "My Machine Learning Project"}
    
    response = client.post("/predict/", json=bad_payload)
    
    # 422 Unprocessable Entity is the standard FastAPI error for bad schemas
    assert response.status_code == 422