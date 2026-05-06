from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    """
    Schema for a single text prediction request.
    """
    text: str = Field(
        ..., 
        min_length=3, 
        max_length=1000, 
        description="The text to analyze for sentiment."
    )

class BatchPredictRequest(BaseModel):
    """
    Schema for a batch text prediction request.
    """
    texts: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=64, 
        description="A list of strings to analyze."
    )
