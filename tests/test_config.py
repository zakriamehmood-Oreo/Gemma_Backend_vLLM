import os
import pytest
from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model_id == "google/gemma-4-E2B-it"
    assert s.device == "auto"
    assert s.max_new_tokens == 512
    assert s.temperature == 0.7
    assert s.top_p == 0.9
    assert s.load_in_4bit is True
    assert s.port == 8000
    assert s.inference_backend == "transformers"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "google/gemma-4-E2B-it")
    monkeypatch.setenv("MAX_NEW_TOKENS", "256")
    monkeypatch.setenv("TEMPERATURE", "0.5")
    s = Settings(_env_file=None)
    assert s.model_id == "google/gemma-4-E2B-it"
    assert s.max_new_tokens == 256
    assert s.temperature == 0.5


def test_load_in_4bit_false(monkeypatch):
    monkeypatch.setenv("LOAD_IN_4BIT", "false")
    s = Settings(_env_file=None)
    assert s.load_in_4bit is False
