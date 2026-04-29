import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.config import settings
from app.inference import gemma
from app.metrics import (
    active_requests,
    completion_tokens_total,
    inference_duration_seconds,
    model_loaded_gauge,
    prompt_tokens_total,
    requests_total,
    tokens_per_second,
)
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        gemma.load()
        model_loaded_gauge.set(1)
        logger.info("Model loaded successfully: %s", settings.model_id)
    except Exception as exc:
        model_loaded_gauge.set(0)
        logger.error("Model failed to load: %s", exc)
    yield
    model_loaded_gauge.set(0)


app = FastAPI(
    title="Gemma 4 Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus scrape endpoint at /metrics
app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_id=settings.model_id,
        model_loaded=gemma.is_loaded,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not gemma.is_loaded:
        requests_total.labels(status="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    active_requests.inc()
    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    try:
        response_text, prompt_tokens, completion_tokens = await loop.run_in_executor(
            None,
            lambda: gemma.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
        )
    except Exception as e:
        requests_total.labels(status="error").inc()
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=repr(e))
    finally:
        active_requests.dec()

    duration = time.perf_counter() - start
    inference_duration_seconds.observe(duration)
    prompt_tokens_total.inc(prompt_tokens)
    completion_tokens_total.inc(completion_tokens)
    tokens_per_second.observe(completion_tokens / duration if duration > 0 else 0)
    requests_total.labels(status="success").inc()

    return GenerateResponse(
        response=response_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
