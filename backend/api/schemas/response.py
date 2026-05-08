from pydantic import BaseModel
from typing import List, Dict, Optional

class SentimentResult(BaseModel):
    """
    Schema for a single prediction result.
    """
    label: str
    confidence: float
    probabilities: Dict[str, float]

class PredictResponse(BaseModel):
    """
    Response schema for a single text prediction.
    """
    text: str
    result: SentimentResult

class AspectInsight(BaseModel):
    """
    Insight generated for a specific aspect/buzzword.
    """
    aspect: str
    sentiment: str
    mention_count: int
    message: str

class BatchPredictResponse(BaseModel):
    """
    Response schema for a batch text prediction.
    """
    results: List[PredictResponse]
    insights: Optional[List[AspectInsight]] = []

class HealthResponse(BaseModel):
    """
    Response schema for the health check endpoint.
    """
    status: str
    model_loaded: bool
