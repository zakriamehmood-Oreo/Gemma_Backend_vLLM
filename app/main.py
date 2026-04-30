import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.auth import verify_api_key
from app.preprocessing import preprocess_message
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
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResult,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

_ANALYZE_PROMPT = """\
You are a customer support analysis assistant.

Analyze the following customer message and return ONLY a valid JSON object with exactly these 6 keys:

- sentimentLabel (positive, neutral, negative, threatening)
- tone (calm, frustrated, angry, anxious, appreciative, demanding, sarcastic)
- urgency (low, medium, high, critical)
- sentimentScore (integer 0-100)
- queryType (one of: Order/Shipping Issue, Billing/Payment Issue, Product Defect, Account Access, General Inquiry, Refund/Return Request, Complaint, Compliment)
- churnRisk (integer 0-100)

Customer message:
\"\"\"
{message}
\"\"\"

Return ONLY JSON. No explanations. No extra text."""


def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from model output."""
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())
    return json.loads(clean)


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

app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_id=settings.model_id,
        model_loaded=gemma.is_loaded,
    )


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
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


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(verify_api_key)])
async def analyze(request: AnalyzeRequest):
    if not gemma.is_loaded:
        requests_total.labels(status="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    clean_message = preprocess_message(request.message)
    prompt = _ANALYZE_PROMPT.format(message=clean_message)

    active_requests.inc()
    start = time.perf_counter()

    loop = asyncio.get_event_loop()
    try:
        raw_text, prompt_tokens, completion_tokens = await loop.run_in_executor(
            None,
            lambda: gemma.generate(
                prompt=prompt,
                max_new_tokens=request.max_new_tokens or 256,
                temperature=0.1,
                top_p=0.9,
            ),
        )
    except Exception as e:
        requests_total.labels(status="error").inc()
        logger.exception("Analysis generation failed")
        raise HTTPException(status_code=500, detail=repr(e))
    finally:
        active_requests.dec()

    duration = time.perf_counter() - start
    inference_duration_seconds.observe(duration)
    prompt_tokens_total.inc(prompt_tokens)
    completion_tokens_total.inc(completion_tokens)
    tokens_per_second.observe(completion_tokens / duration if duration > 0 else 0)
    requests_total.labels(status="success").inc()

    try:
        parsed = _extract_json(raw_text)
        result = AnalysisResult(**parsed)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse model JSON output: %s | raw: %s", e, raw_text)
        raise HTTPException(
            status_code=422,
            detail=f"Model returned invalid JSON: {e}. Raw: {raw_text!r}",
        )

    return AnalyzeResponse(
        result=result,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw_response=raw_text,
        preprocessed_message=clean_message,
    )
