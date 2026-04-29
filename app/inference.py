import logging
from abc import ABC, abstractmethod

from app.config import settings

logger = logging.getLogger(__name__)


class BaseInferenceBackend(ABC):
    def __init__(self):
        self._loaded = False

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, int, int]: ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class TransformersBackend(BaseInferenceBackend):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self._device = "cpu"

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            bnb_available = True
        except ImportError:
            bnb_available = False

        cuda_available = torch.cuda.is_available()
        hf_token = settings.hf_token or None
        quantization_config = None

        if cuda_available and settings.load_in_4bit:
            if not bnb_available:
                logger.warning("bitsandbytes not available — loading in bfloat16 instead of 4-bit")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

        self._device = "cuda" if cuda_available else "cpu"
        device_map = settings.device if cuda_available else None
        dtype = torch.bfloat16 if quantization_config is None else None

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=dtype,
            token=hf_token,
        )
        self.model.eval()
        self._loaded = True
        logger.info("TransformersBackend loaded on %s", self._device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, int, int]:
        import torch

        max_new_tokens = max_new_tokens or settings.max_new_tokens
        temperature = temperature or settings.temperature
        top_p = top_p or settings.top_p

        messages = [{"role": "user", "content": prompt}]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = (tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized).to(self._device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_token_count = input_ids.shape[-1]
        completion_ids = output_ids[0][prompt_token_count:]
        response_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return response_text, prompt_token_count, len(completion_ids)


class VLLMBackend(BaseInferenceBackend):
    def __init__(self):
        super().__init__()
        self.llm = None
        self.tokenizer = None

    def load(self):
        from vllm import LLM
        from transformers import AutoTokenizer

        hf_token = settings.hf_token or None
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id, token=hf_token)
        self.llm = LLM(
            model=settings.model_id,
            dtype="bfloat16",
            quantization="bitsandbytes" if settings.load_in_4bit else None,
            token=hf_token,
        )
        self._loaded = True
        logger.info("VLLMBackend loaded")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, int, int]:
        from vllm import SamplingParams

        max_new_tokens = max_new_tokens or settings.max_new_tokens
        temperature = temperature or settings.temperature
        top_p = top_p or settings.top_p

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        outputs = self.llm.generate([formatted], params)

        result = outputs[0]
        completion_text = result.outputs[0].text
        prompt_tokens = len(result.prompt_token_ids)
        completion_tokens = len(result.outputs[0].token_ids)
        return completion_text, prompt_tokens, completion_tokens


def create_backend() -> BaseInferenceBackend:
    backend = settings.inference_backend.lower()
    if backend == "vllm":
        logger.info("Using vLLM backend")
        return VLLMBackend()
    logger.info("Using Transformers backend")
    return TransformersBackend()


gemma = create_backend()
