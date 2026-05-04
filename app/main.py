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
    queued_requests,
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

_semaphore: asyncio.Semaphore | None = None
_waiting: int = 0


@asynccontextmanager
async def _concurrency_slot():
    """Acquire a concurrency slot; queue if full; reject if queue is full."""
    global _waiting
    if _waiting >= settings.max_queue_depth:
        requests_total.labels(status="queue_full").inc()
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server queue is full ({_waiting} requests waiting). "
                "Try again shortly."
            ),
        )
    _waiting += 1
    queued_requests.set(_waiting)
    acquired = False
    try:
        await _semaphore.acquire()
        acquired = True
        _waiting -= 1
        queued_requests.set(_waiting)
        active_requests.inc()
        yield
    finally:
        if acquired:
            active_requests.dec()
            _semaphore.release()
        else:
            _waiting -= 1
            queued_requests.set(_waiting)

_ANALYZE_PROMPT = """\
You are a customer support analysis assistant. Analyze ONLY the customer message below and return a single JSON object. No explanation. No extra text.

## Rules

**1. sentiment** — pick one: positive | neutral | negative | threatening
- threatening: ANY mention of legal action, chargeback, refund dispute, authorities, or public exposure — ALWAYS overrides others
- negative: frustration, complaints, dissatisfaction
- neutral: factual, no emotion
- positive: satisfaction, praise, thanks

**2. tone** — pick one: calm | frustrated | angry | anxious | appreciative | demanding | sarcastic
- angry: insults, ALL CAPS, harsh or aggressive language
- frustrated: complaint without aggression
- demanding: direct commands ("fix this now", "send it immediately")
- anxious: worry or fear ("I’m concerned", "getting nervous")
- sarcastic: mockery or passive-aggressive language
- appreciative: gratitude or positive acknowledgment
- calm: composed, neutral delivery

**3. urgency** — pick one: low | medium | high | critical
- critical: deadline within 48 hours OR sentiment is threatening
- high: explicitly mentions urgency (ASAP, urgent) OR is a repeated follow-up
- medium: clear need, no urgency signals
- low: informational only

**4. sentiment_score** — integer 0–100
- 0–20: threatening or extreme hostility
- 21–35: strong negative (angry, multiple complaints)
- 36–45: mild negative (frustrated, disappointed)
- 46–54: neutral
- 55–70: mild positive (polite, appreciative)
- 71–100: strong positive (praise, very satisfied)

**5. query_type** — pick one: order-status | shipping-delay | address-change | order-modification | order-hold | refund | billing | product-inquiry | restock-inquiry | damaged-item | warranty | technical-issue | general-inquiry
- If multiple intents, pick the most urgent or irreversible
- Refund or cancellation always overrides other intents

**6. churn_risk** — integer 0–100, baseline 50
- Add: threatening +30, refund/cancellation request +20, repeated issue +15, strong negative sentiment +20, mentions a competitor +10
- Subtract: positive sentiment -15, appreciation -10
- Clamp to 0–100

## Customer Message

{message}

## Output

Return ONLY valid JSON, nothing else:
{"sentiment": "", "tone": "", "urgency": "", "sentiment_score": 0, "query_type": "", "churn_risk": 0}
"""


def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from model output."""
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean.strip())
    if not clean:
        raise json.JSONDecodeError("Model returned empty response", text, 0)
    # Find the JSON object boundaries in case there's surrounding text
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise json.JSONDecodeError("No JSON object found in response", clean, 0)
    return json.loads(clean[start:end])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore
    _semaphore = asyncio.Semaphore(settings.max_concurrent)
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
    slots_in_use = settings.max_concurrent - (_semaphore._value if _semaphore else 0)
    return HealthResponse(
        status="ok",
        model_id=settings.model_id,
        model_loaded=gemma.is_loaded,
        active_requests=max(slots_in_use, 0),
        queued_requests=_waiting,
        max_concurrent=settings.max_concurrent,
        max_queue_depth=settings.max_queue_depth,
    )


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    if not gemma.is_loaded:
        requests_total.labels(status="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    async with _concurrency_slot():
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
    prompt = _ANALYZE_PROMPT.replace("{message}", clean_message)

    async with _concurrency_slot():
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
