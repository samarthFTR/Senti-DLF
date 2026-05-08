from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List
import tempfile
import os
import re

from model.src.reviewparser import ReviewParser, Config

from backend.api.schemas.request import PredictRequest, BatchPredictRequest
from backend.api.schemas.response import PredictResponse, BatchPredictResponse, SentimentResult, AspectInsight
from backend.core.inference import SentimentInferenceEngine, get_engine
from backend.core.config import settings

router = APIRouter()

ASPECTS = {
    "UI/UX": ["ui", "user interface", "design", "layout", "ux", "user experience", "look", "feel"],
    "Performance": ["speed", "fast", "slow", "lag", "performance", "crash", "buggy", "freeze", "loading"],
    "Customer Support": ["support", "customer service", "help", "response", "service", "agent"],
    "Pricing": ["price", "cost", "expensive", "cheap", "subscription", "value", "money", "fee"],
    "Features": ["feature", "functionality", "missing", "tool", "option", "setting", "update"]
}

def generate_aspect_insights(responses: List[PredictResponse]) -> List[AspectInsight]:
    aspect_sentiments = {aspect: {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0, "count": 0} for aspect in ASPECTS}
    
    for resp in responses:
        text_lower = resp.text.lower()
        pred_label = resp.result.label
        
        for aspect, keywords in ASPECTS.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in keywords):
                aspect_sentiments[aspect][pred_label] += 1
                aspect_sentiments[aspect]["count"] += 1
                
    insights = []
    for aspect, stats in aspect_sentiments.items():
        if stats["count"] >= 2: # At least 2 mentions needed for an insight
            counts = {
                "positive": stats["positive"],
                "negative": stats["negative"],
                "neutral": stats["neutral"],
                "mixed": stats["mixed"]
            }
            dominant = max(counts, key=counts.get)
            
            if dominant == "negative":
                message = f"Reviews about {aspect} are mostly negative."
            elif dominant == "positive":
                message = f"Users are generally praising the {aspect}."
            elif dominant == "mixed":
                message = f"Opinions on {aspect} are mixed."
            else:
                message = f"Feedback regarding {aspect} is mostly neutral."
                
            insights.append(
                AspectInsight(
                    aspect=aspect,
                    sentiment=dominant,
                    mention_count=stats["count"],
                    message=message
                )
            )
            
    # Sort insights by mention count descending
    insights.sort(key=lambda x: x.mention_count, reverse=True)
    return insights

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
            
        insights = generate_aspect_insights(responses)
        return BatchPredictResponse(results=responses, insights=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


@router.post("/predict/file", response_model=BatchPredictResponse)
async def predict_file(
    file: UploadFile = File(...),
    engine: SentimentInferenceEngine = Depends(get_engine)
):
    """
    Analyze the sentiment of reviews from an uploaded text file.
    The file should have reviews separated by blank lines.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as temp:
            content = await file.read()
            temp.write(content)
            temp_path = temp.name
            
        parser_config = Config(FILE_PATH=temp_path)
        parser = ReviewParser(config=parser_config)
        parser.load_reviews()
        reviews = parser.get_reviews()
        
        os.unlink(temp_path)
        
        if not reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found in the file.")
            
        results = engine.predict(reviews)
        
        responses = []
        for text, result_data in zip(reviews, results):
            responses.append(
                PredictResponse(
                    text=text,
                    result=SentimentResult(**result_data)
                )
            )
            
        insights = generate_aspect_insights(responses)
        return BatchPredictResponse(results=responses, insights=insights)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"File inference failed: {str(e)}")
