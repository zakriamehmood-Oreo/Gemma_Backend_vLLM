from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The input prompt for the model")
    max_new_tokens: int | None = Field(None, gt=0, le=4096)
    temperature: float | None = Field(None, gt=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    response: str
    prompt_tokens: int
    completion_tokens: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_requests: int
    queued_requests: int


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="Raw customer message to analyze")
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
    model_warnings: list[str] = Field(default_factory=list, description="Fields substituted due to invalid model output")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        if v not in _VALID_SENTIMENTS:
            # warning recorded in main.py after construction
            return "neutral"
        return v

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in _VALID_TONES:
            return "calm"
        return v

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        if v not in _VALID_URGENCIES:
            return "medium"
        return v

    @field_validator("query_type")
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        if v not in _VALID_QUERY_TYPES:
            return "general-inquiry"
        return v


class AnalyzeResponse(BaseModel):
    result: AnalysisResult
    prompt_tokens: int
    completion_tokens: int
    preprocessed_message: str
