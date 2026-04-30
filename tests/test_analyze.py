import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app, _extract_json
from app.inference import gemma

VALID_JSON = {
    "sentimentLabel": "negative",
    "tone": "frustrated",
    "urgency": "high",
    "sentimentScore": 30,
    "queryType": "Order/Shipping Issue",
    "churnRisk": 75,
}

VALID_RAW = json.dumps(VALID_JSON)
VALID_FENCED = f"```json\n{VALID_RAW}\n```"
VALID_FENCED_NO_LANG = f"```\n{VALID_RAW}\n```"


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


# --- _extract_json ---

def test_extract_json_plain():
    assert _extract_json(VALID_RAW) == VALID_JSON


def test_extract_json_fenced_with_lang():
    assert _extract_json(VALID_FENCED) == VALID_JSON


def test_extract_json_fenced_no_lang():
    assert _extract_json(VALID_FENCED_NO_LANG) == VALID_JSON


def test_extract_json_with_whitespace():
    assert _extract_json(f"  \n{VALID_RAW}\n  ") == VALID_JSON


def test_extract_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("not json at all")


# --- /analyze endpoint ---

def test_analyze_success(client, loaded_model):
    with patch.object(gemma, "generate", return_value=(VALID_FENCED, 50, 40)):
        resp = client.post("/analyze", json={"message": "My order is late!"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["sentimentLabel"] == "negative"
    assert data["result"]["tone"] == "frustrated"
    assert data["result"]["urgency"] == "high"
    assert data["result"]["sentimentScore"] == 30
    assert data["result"]["queryType"] == "Order/Shipping Issue"
    assert data["result"]["churnRisk"] == 75
    assert data["prompt_tokens"] == 50
    assert data["completion_tokens"] == 40
    assert data["raw_response"] == VALID_FENCED


def test_analyze_model_not_loaded(client, unloaded_model):
    resp = client.post("/analyze", json={"message": "My order is late!"})
    assert resp.status_code == 503


def test_analyze_empty_message(client, loaded_model):
    resp = client.post("/analyze", json={"message": ""})
    assert resp.status_code == 422


def test_analyze_missing_message(client, loaded_model):
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422


def test_analyze_invalid_json_from_model(client, loaded_model):
    with patch.object(gemma, "generate", return_value=("Sorry I cannot help.", 10, 5)):
        resp = client.post("/analyze", json={"message": "test"})
    assert resp.status_code == 422
    assert "invalid JSON" in resp.json()["detail"]


def test_analyze_invalid_sentiment_label(client, loaded_model):
    bad = {**VALID_JSON, "sentimentLabel": "confused"}
    with patch.object(gemma, "generate", return_value=(json.dumps(bad), 10, 5)):
        resp = client.post("/analyze", json={"message": "test"})
    assert resp.status_code == 422


def test_analyze_score_out_of_range(client, loaded_model):
    bad = {**VALID_JSON, "sentimentScore": 150}
    with patch.object(gemma, "generate", return_value=(json.dumps(bad), 10, 5)):
        resp = client.post("/analyze", json={"message": "test"})
    assert resp.status_code == 422


def test_analyze_inference_error(client, loaded_model):
    with patch.object(gemma, "generate", side_effect=RuntimeError("OOM")):
        resp = client.post("/analyze", json={"message": "test"})
    assert resp.status_code == 500


def test_analyze_uses_low_temperature(client, loaded_model):
    with patch.object(gemma, "generate", return_value=(VALID_RAW, 50, 30)) as mock_gen:
        client.post("/analyze", json={"message": "test"})
    _, kwargs = mock_gen.call_args
    assert kwargs["temperature"] == 0.1
