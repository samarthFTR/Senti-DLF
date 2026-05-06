from pydantic import BaseModel
from typing import List, Dict

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

class BatchPredictResponse(BaseModel):
    """
    Response schema for a batch text prediction.
    """
    results: List[PredictResponse]

class HealthResponse(BaseModel):
    """
    Response schema for the health check endpoint.
    """
    status: str
    model_loaded: bool
