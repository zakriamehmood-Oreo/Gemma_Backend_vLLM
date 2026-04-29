import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.inference import gemma
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        gemma.load()
        logger.info("Model loaded successfully: %s", settings.model_id)
    except Exception as exc:
        logger.error("Model failed to load: %s", exc)
    yield


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
        raise HTTPException(status_code=503, detail="Model is not loaded")

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
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=repr(e))

    return GenerateResponse(
        response=response_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
