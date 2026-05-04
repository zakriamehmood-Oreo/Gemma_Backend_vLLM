from typing import Literal
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input prompt for the model")
    max_new_tokens: int | None = Field(None, gt=0, le=4096)
    temperature: float | None = Field(None, gt=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    response: str
    prompt_tokens: int
    completion_tokens: int


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    active_requests: int
    queued_requests: int
    max_concurrent: int
    max_queue_depth: int


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Raw customer message to analyze")
    max_new_tokens: int | None = Field(None, gt=0, le=512)


_VALID_SENTIMENTS = {"positive", "neutral", "negative", "threatening"}
_VALID_TONES = {"calm", "frustrated", "angry", "anxious", "appreciative", "demanding", "sarcastic"}
_VALID_URGENCIES = {"low", "medium", "high", "critical"}
_VALID_QUERY_TYPES = {
    "order-status", "shipping-delay", "address-change", "order-modification",
    "order-hold", "refund", "billing", "product-inquiry", "restock-inquiry",
    "damaged-item", "warranty", "technical-issue", "general-inquiry",
}


class AnalysisResult(BaseModel):
    sentiment: str
    tone: str
    urgency: str
    sentiment_score: int = Field(..., ge=0, le=100)
    query_type: str
    churn_risk: int = Field(..., ge=0, le=100)

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        return v if v in _VALID_SENTIMENTS else "neutral"

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        return v if v in _VALID_TONES else "calm"

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        return v if v in _VALID_URGENCIES else "medium"

    @field_validator("query_type")
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        return v if v in _VALID_QUERY_TYPES else "general-inquiry"


class AnalyzeResponse(BaseModel):
    result: AnalysisResult
    prompt_tokens: int
    completion_tokens: int
    raw_response: str
    preprocessed_message: str
