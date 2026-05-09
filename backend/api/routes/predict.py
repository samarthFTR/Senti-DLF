from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List
import tempfile
import os
import re

from model.src.reviewparser import ReviewParser, Config
from sklearn.feature_extraction.text import CountVectorizer

from backend.api.schemas.request import PredictRequest, BatchPredictRequest
from backend.api.schemas.response import PredictResponse, BatchPredictResponse, SentimentResult, AspectInsight
from backend.core.inference import SentimentInferenceEngine, get_engine
from backend.core.config import settings

router = APIRouter()

def extract_dynamic_aspects(texts: List[str], top_n: int = 6) -> List[str]:
    try:
        # We start with sklearn's english stop words and add common sentiment words/verbs
        base_stop_words = list(CountVectorizer(stop_words='english').get_stop_words())
        custom_stopwords = base_stop_words + [
            "good", "bad", "great", "awesome", "terrible", "excellent", "poor", 
            "love", "hate", "like", "dislike", "just", "really", "very", "much",
            "make", "get", "got", "go", "going", "know", "think", "see", "time",
            "people", "thing", "things", "way", "day", "don", "ve", "ll", "re",
            "did", "didn", "does", "doesn", "isn", "aren", "wasn", "weren",
            "best", "worst", "better", "worse", "amazing", "horrible", "nice"
        ]
        
        vectorizer = CountVectorizer(
            stop_words=custom_stopwords, 
            max_df=0.9, 
            min_df=2, # Word must appear in at least 2 reviews
            ngram_range=(1, 2) # Allow bigrams like "customer service"
        )
        
        X = vectorizer.fit_transform(texts)
        word_counts = X.sum(axis=0).A1
        
        word_freq = [(word, word_counts[idx]) for word, idx in vectorizer.vocabulary_.items()]
        word_freq.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, count in word_freq[:top_n]]
    except ValueError:
        # Happens if vocabulary is empty
        return []

def generate_aspect_insights(responses: List[PredictResponse]) -> List[AspectInsight]:
    texts = [resp.text for resp in responses]
    dynamic_aspects = extract_dynamic_aspects(texts, top_n=6)
    
    if not dynamic_aspects:
        return []
        
    aspect_sentiments = {aspect: {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0, "count": 0} for aspect in dynamic_aspects}
    
    for resp in responses:
        text_lower = resp.text.lower()
        pred_label = resp.result.label
        
        for aspect in dynamic_aspects:
            if re.search(r'\b' + re.escape(aspect) + r'\b', text_lower):
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
                message = f"Reviews mentioning '{aspect}' are mostly negative."
            elif dominant == "positive":
                message = f"Users generally speak positively about '{aspect}'."
            elif dominant == "mixed":
                message = f"Opinions on '{aspect}' are mixed."
            else:
                message = f"Feedback regarding '{aspect}' is mostly neutral."
                
            insights.append(
                AspectInsight(
                    aspect=aspect.title(),
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
