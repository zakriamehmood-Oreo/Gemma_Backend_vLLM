from unittest.mock import patch, MagicMock
import pytest


def test_create_backend_transformers(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "transformers")
    from importlib import reload
    import app.config as cfg
    reload(cfg)
    import app.inference as inf
    reload(inf)
    assert isinstance(inf.gemma, inf.TransformersBackend)


def test_create_backend_vllm(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "vllm")
    from importlib import reload
    import app.config as cfg
    reload(cfg)
    import app.inference as inf
    reload(inf)
    assert isinstance(inf.gemma, inf.VLLMBackend)


def test_create_backend_default_is_transformers(monkeypatch):
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    from importlib import reload
    import app.config as cfg
    reload(cfg)
    import app.inference as inf
    reload(inf)
    assert isinstance(inf.gemma, inf.TransformersBackend)


def test_transformers_backend_not_loaded_initially():
    from app.inference import TransformersBackend
    b = TransformersBackend()
    assert b.is_loaded is False


def test_vllm_backend_not_loaded_initially():
    from app.inference import VLLMBackend
    b = VLLMBackend()
    assert b.is_loaded is False


def test_transformers_backend_load(monkeypatch):
    from app.inference import TransformersBackend

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with patch("app.inference.settings") as mock_settings, \
         patch("torch.cuda.is_available", return_value=False):
        mock_settings.model_id = "google/gemma-4-E2B-it"
        mock_settings.hf_token = ""
        mock_settings.load_in_4bit = False
        mock_settings.device = "auto"

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            b = TransformersBackend()
            b.load()

    assert b.is_loaded is True
    assert b._device == "cpu"
    mock_model.eval.assert_called_once()


def test_vllm_backend_load(monkeypatch):
    from app.inference import VLLMBackend

    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("app.inference.settings") as mock_settings:
        mock_settings.model_id = "google/gemma-4-E2B-it"
        mock_settings.hf_token = ""
        mock_settings.load_in_4bit = False

        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=MagicMock(return_value=mock_llm))}), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            b = VLLMBackend()
            b.load()

    assert b.is_loaded is True
