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

    # Leave empty to disable auth (local dev). Set a strong value in production.
    api_key: str = ""

    # Concurrency: max requests processed at once; rest queue up to max_queue_depth
    max_concurrent: int = 10
    max_queue_depth: int = 50

    # CORS: comma-separated list of allowed origins. Use * only in local dev.
    allowed_origins: str = "*"

    # Set true to allow running with no API_KEY (local dev only)
    allow_no_auth: bool = False

    # Set true when behind a reverse proxy — enables X-Forwarded-For for real IP
    trust_proxy: bool = False

    # Grafana admin password (used by docker-compose.monitoring.yml)
    grafana_password: str = "changeme"


settings = Settings()
