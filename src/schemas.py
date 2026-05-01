from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationItem(BaseModel):
    movie_id: int
    predicted_rating: float = Field(ge=0.5, le=5.0)
    rank: int


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    timestamp: str
    model_version: str


class PredictionItem(BaseModel):
    user_id: int = Field(gt=0)
    movie_id: int = Field(gt=0)


class BatchPredictRequest(BaseModel):
    predictions: List[PredictionItem]


class PredictionResult(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float


class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResult]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    model_version: Optional[str] = None