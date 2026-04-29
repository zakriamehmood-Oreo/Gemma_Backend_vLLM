import pytest
from pydantic import ValidationError
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse


def test_generate_request_minimal():
    req = GenerateRequest(prompt="hello")
    assert req.prompt == "hello"
    assert req.max_new_tokens is None
    assert req.temperature is None
    assert req.top_p is None


def test_generate_request_full():
    req = GenerateRequest(prompt="hi", max_new_tokens=100, temperature=0.5, top_p=0.8)
    assert req.max_new_tokens == 100
    assert req.temperature == 0.5
    assert req.top_p == 0.8


def test_generate_request_empty_prompt():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_generate_request_max_new_tokens_too_large():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", max_new_tokens=9999)


def test_generate_request_temperature_out_of_range():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", temperature=3.0)


def test_generate_request_top_p_out_of_range():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", top_p=1.5)


def test_generate_request_negative_tokens():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", max_new_tokens=0)


def test_generate_response():
    resp = GenerateResponse(response="hello world", prompt_tokens=5, completion_tokens=2)
    assert resp.response == "hello world"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 2


def test_health_response_loaded():
    h = HealthResponse(status="ok", model_id="google/gemma-4-E2B-it", model_loaded=True)
    assert h.status == "ok"
    assert h.model_loaded is True


def test_health_response_not_loaded():
    h = HealthResponse(status="ok", model_id="google/gemma-4-E2B-it", model_loaded=False)
    assert h.model_loaded is False
