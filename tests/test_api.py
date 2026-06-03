import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.inference import gemma


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_model():
    with patch.object(gemma, "_loaded", True):
        yield


@pytest.fixture
def unloaded_model():
    with patch.object(gemma, "_loaded", False):
        yield


# --- /health ---

def test_health_model_loaded(client, loaded_model):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_model_not_loaded(client, unloaded_model):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


# --- /metrics ---

def test_metrics_endpoint_available(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"gemma_model_loaded" in resp.content


def test_metrics_records_success(client, loaded_model):
    with patch.object(gemma, "generate", return_value=("hi", 4, 2)):
        client.post("/generate", json={"prompt": "test"})
    resp = client.get("/metrics")
    assert b'gemma_requests_total{status="success"}' in resp.content
    assert b"gemma_inference_duration_seconds_count" in resp.content
    assert b"gemma_prompt_tokens_total" in resp.content
    assert b"gemma_completion_tokens_total" in resp.content


def test_metrics_records_model_not_loaded(client, unloaded_model):
    client.post("/generate", json={"prompt": "test"})
    resp = client.get("/metrics")
    assert b'gemma_requests_total{status="model_not_loaded"}' in resp.content


def test_metrics_records_error(client, loaded_model):
    with patch.object(gemma, "generate", side_effect=RuntimeError("OOM")):
        client.post("/generate", json={"prompt": "test"})
    resp = client.get("/metrics")
    assert b'gemma_requests_total{status="error"}' in resp.content


# --- /generate ---

def test_generate_success(client, loaded_model):
    with patch.object(gemma, "generate", return_value=("hello there", 5, 2)):
        resp = client.post("/generate", json={"prompt": "say hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "hello there"
    assert data["prompt_tokens"] == 5
    assert data["completion_tokens"] == 2


def test_generate_model_not_loaded(client, unloaded_model):
    resp = client.post("/generate", json={"prompt": "say hello"})
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"]


def test_generate_empty_prompt(client, loaded_model):
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422


def test_generate_missing_prompt(client, loaded_model):
    resp = client.post("/generate", json={})
    assert resp.status_code == 422


def test_generate_with_params(client, loaded_model):
    with patch.object(gemma, "generate", return_value=("ok", 3, 1)) as mock_gen:
        resp = client.post("/generate", json={
            "prompt": "test",
            "max_new_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.8,
        })
    assert resp.status_code == 200
    mock_gen.assert_called_once_with(
        prompt="test",
        max_new_tokens=128,
        temperature=0.5,
        top_p=0.8,
    )


def test_generate_inference_error(client, loaded_model):
    with patch.object(gemma, "generate", side_effect=RuntimeError("OOM")):
        resp = client.post("/generate", json={"prompt": "test"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Inference error"


def test_generate_invalid_temperature(client, loaded_model):
    resp = client.post("/generate", json={"prompt": "test", "temperature": 5.0})
    assert resp.status_code == 422


def test_generate_invalid_top_p(client, loaded_model):
    resp = client.post("/generate", json={"prompt": "test", "top_p": 2.0})
    assert resp.status_code == 422


def test_generate_max_new_tokens_boundary(client, loaded_model):
    with patch.object(gemma, "generate", return_value=("ok", 1, 1)):
        resp = client.post("/generate", json={"prompt": "test", "max_new_tokens": 4096})
    assert resp.status_code == 200


def test_generate_max_new_tokens_exceeded(client, loaded_model):
    resp = client.post("/generate", json={"prompt": "test", "max_new_tokens": 4097})
    assert resp.status_code == 422
