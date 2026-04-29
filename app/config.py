from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    model_id: str = "google/gemma-4-E2B-it"
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    hf_token: str = ""

    # "transformers" for local dev (Windows/CPU), "vllm" for production (Linux/GPU)
    inference_backend: str = "transformers"


settings = Settings()
