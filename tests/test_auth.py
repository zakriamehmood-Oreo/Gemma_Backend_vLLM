import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.inference import gemma

VALID_JSON = '{"sentimentLabel":"negative","tone":"frustrated","urgency":"high","sentimentScore":30,"queryType":"Order/Shipping Issue","churnRisk":65}'


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_model():
    with patch.object(gemma, "_loaded", True):
        yield


# --- auth disabled (no API_KEY set) ---

def test_no_auth_configured_generate_passes(client, loaded_model):
    with patch("app.config.settings.api_key", ""), \
         patch.object(gemma, "generate", return_value=("hi", 3, 2)):
        resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 200


def test_no_auth_configured_analyze_passes(client, loaded_model):
    with patch("app.config.settings.api_key", ""), \
         patch.object(gemma, "generate", return_value=(VALID_JSON, 50, 20)):
        resp = client.post("/analyze", json={"message": "test"})
    assert resp.status_code == 200


# --- auth enabled ---

def test_correct_api_key_accepted(client, loaded_model):
    with patch("app.auth.settings.api_key", "secret-key"), \
         patch.object(gemma, "generate", return_value=("hi", 3, 2)):
        resp = client.post("/generate", json={"prompt": "hello"},
                           headers={"X-API-Key": "secret-key"})
    assert resp.status_code == 200


def test_wrong_api_key_rejected(client, loaded_model):
    with patch("app.auth.settings.api_key", "secret-key"):
        resp = client.post("/generate", json={"prompt": "hello"},
                           headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


def test_missing_api_key_rejected(client, loaded_model):
    with patch("app.auth.settings.api_key", "secret-key"):
        resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 401


def test_health_does_not_require_api_key(client):
    with patch("app.auth.settings.api_key", "secret-key"):
        resp = client.get("/health")
    assert resp.status_code == 200


def test_analyze_requires_api_key(client, loaded_model):
    with patch("app.auth.settings.api_key", "secret-key"):
        resp = client.post("/analyze", json={"message": "test"})
    assert resp.status_code == 401
