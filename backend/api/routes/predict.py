from fastapi import APIRouter, HTTPException, Depends
from typing import List

from backend.api.schemas.request import PredictRequest, BatchPredictRequest
from backend.api.schemas.response import PredictResponse, BatchPredictResponse, SentimentResult
from backend.core.inference import SentimentInferenceEngine, get_engine
from backend.core.config import settings

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict_single(
    request: PredictRequest,
    engine: SentimentInferenceEngine = Depends(get_engine)
):
    """
    Analyze the sentiment of a single piece of text.
    """
    try:
        results = engine.predict([request.text])
        result_data = results[0]
        
        return PredictResponse(
            text=request.text,
            result=SentimentResult(**result_data)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    engine: SentimentInferenceEngine = Depends(get_engine)
):
    """
    Analyze the sentiment of a batch of texts.
    """
    try:
        results = engine.predict(request.texts)
        
        responses = []
        for text, result_data in zip(request.texts, results):
            responses.append(
                PredictResponse(
                    text=text,
                    result=SentimentResult(**result_data)
                )
            )
            
        return BatchPredictResponse(results=responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
