from typing import Literal
from pydantic import BaseModel, Field


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


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Raw customer message to analyze")
    max_new_tokens: int | None = Field(None, gt=0, le=512)


class AnalysisResult(BaseModel):
    sentimentLabel: Literal["positive", "neutral", "negative", "threatening"]
    tone: Literal["calm", "frustrated", "angry", "anxious", "appreciative", "demanding", "sarcastic"]
    urgency: Literal["low", "medium", "high", "critical"]
    sentimentScore: int = Field(..., ge=0, le=100)
    queryType: str
    churnRisk: int = Field(..., ge=0, le=100)


class AnalyzeResponse(BaseModel):
    result: AnalysisResult
    prompt_tokens: int
    completion_tokens: int
    raw_response: str
    preprocessed_message: str
